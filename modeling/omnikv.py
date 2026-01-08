"""
OmniKV 核心建模模块。

该模块实现了 OmniKV 的核心逻辑，包括多种 Token 选择策略、多阶段缓存管理逻辑
以及针对 Llama 架构的自定义层、模型和语言模型封装类。
"""
import torch

from modeling.offload_select_once import *
from modeling.spec_cache import OmniKVMultiStageCache, WOPackCache
import logging as lgt


def select_tokens_by_attn_universal(
    raw_attn,
    hidden_states,
    position_ids,
    past_key_value,
    num_selected_tokens,
    consider_len,
    layer_idx=None,
    selector_cls="last",
):
    """
    通用的 Token 选择函数，支持多种基于注意力的筛选策略。

    Args:
        raw_attn (nn.Module): 注意力模块实例。
        hidden_states (torch.Tensor): 输入的隐藏状态 [bsz, q_len, dim]。
        position_ids (torch.LongTensor): 位置编码 ID。
        past_key_value (Cache): 缓存对象，需包含全量 Key 缓存。
        num_selected_tokens (int): 目标选择的 Token 数量。
        consider_len (int): 考虑的历史序列长度。
        layer_idx (int, optional): 当前层索引。
        selector_cls (str, optional): 选择器类型。支持:
            - 'last': 仅使用最后一个 Query 产生的注意力分数。
            - 'softmax_before_last': 先做 Softmax 再取最后一个 Query 的分数。
            - 'uniform': 平等对待 Prefill 阶段的所有 Query 注意力分数并累加。
            - 'exp': 对 Prefill 阶段的 Query 赋予随位置指数衰减的权重并累加。

    Returns:
        torch.LongTensor: 选中的 Token 索引，按原始位置排序。
    """
    bsz, q_len, _ = hidden_states.size()
    assert past_key_value

    # 处理张量并行逻辑（目前未在 universal 中实现 TP）
    if raw_attn.config.get("pretraining_tp", -1) > 1:
        raise NotImplementedError
    else:
        # 计算当前输入的 Q 和 K
        query_states = raw_attn.q_proj(hidden_states)
        key_states = raw_attn.k_proj(hidden_states)

    # 形状变换 [bsz, heads, q_len, dim]
    query_states = query_states.view(
        bsz, q_len, raw_attn.num_heads, raw_attn.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        bsz, q_len, raw_attn.num_key_value_heads, raw_attn.head_dim
    ).transpose(1, 2)

    # 应用 RoPE 位置编码
    cos, sin = raw_attn.rotary_emb(key_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # 从缓存中提取参与计算的历史 Key [bsz, kv_heads, consider_len, dim]
    key_states = past_key_value.key_cache[raw_attn.layer_idx][:, :, :consider_len]
    # 如果是 GQA 结构，则重复 KV 头以匹配 Query 数量
    key_states = repeat_kv(key_states, raw_attn.num_key_value_groups)

    # 策略 1: 基于末位 Token 的注意力分数（原生视角）
    if selector_cls == "last":
        attn_score = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
            raw_attn.head_dim
        )
        # 在所有 Head 上取最大值，取 Q 序列的最后一个位置 [bsz, consider_len]
        attn_score = torch.max(
            attn_score[..., -1, :], dim=1
        ).values  # remove query, then head
        num_selected_tokens = min(num_selected_tokens, attn_score.shape[-1])
        v, idx = torch.topk(attn_score, k=num_selected_tokens, dim=-1, sorted=True)
        
    # 策略 2: 在末位 Token 决策前先应用 Softmax
    elif selector_cls == "softmax_before_last":
        attn_score = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
            raw_attn.head_dim
        )
        attn_score = torch.nn.functional.softmax(attn_score, dim=-1)  
        attn_score = torch.max(
            attn_score[..., -1, :], dim=1
        ).values  
        num_selected_tokens = min(num_selected_tokens, attn_score.shape[-1])
        v, idx = torch.topk(attn_score, k=num_selected_tokens, dim=-1, sorted=True)
        
    # 策略 3: 累加所有输入 Token 对历史的注意力贡献（均匀采样视角）
    elif selector_cls == "uniform":
        qs = torch.split(query_states, 1, dim=2)
        first_flag = True
        attn_sum = None
        logger.debug(f"before = {torch.cuda.memory_allocated() / 1e9} GB")
        for q in qs:
            attn_score = torch.matmul(q, key_states.transpose(2, 3)) / math.sqrt(
                raw_attn.head_dim
            )
            attn_score = torch.nn.functional.softmax(
                attn_score, dim=-1
            )  
            attn_score = torch.max(attn_score, dim=1).values  # 移除 Head 维度
            attn_score = torch.sum(attn_score, dim=-2)  # 累加 Query 维度
            if first_flag:
                first_flag = False
                attn_sum = attn_score
            else:
                attn_sum += attn_score
        num_selected_tokens = min(num_selected_tokens, attn_sum.shape[-1])
        v, idx = torch.topk(attn_sum, k=num_selected_tokens, dim=-1, sorted=True)
        logger.debug(f"after = {torch.cuda.memory_allocated() / 1e9} GB")
        
    # 策略 4: 对输入 Token 赋予随时间衰减的权重并累加
    elif selector_cls == "exp":
        qs = torch.split(query_states, 1, dim=2)
        first_flag = True
        attn_sum = None
        for q in qs:
            attn_score = torch.matmul(q, key_states.transpose(2, 3)) / math.sqrt(
                raw_attn.head_dim
            )
            attn_score = torch.nn.functional.softmax(
                attn_score, dim=-1
            )  
            attn_score = torch.max(attn_score, dim=1).values  
            q_len = attn_score.shape[-2]
            # 生成 2^(0), 2^(-1), ... 的衰减系数
            alpha = (
                2
                ** torch.arange(-q_len + 1, 1, device=attn_score.device)[None, :, None]
            )
            attn_score = torch.sum(attn_score * alpha, dim=-2)  
            if first_flag:
                first_flag = False
                attn_sum = attn_score
            else:
                # 实现衰减累加逻辑
                attn_sum = attn_sum * (2**-q_len) + attn_score
        num_selected_tokens = min(num_selected_tokens, attn_sum.shape[-1])
        v, idx = torch.topk(attn_sum, k=num_selected_tokens, dim=-1, sorted=True)
    else:
        raise NotImplementedError

    # 统一按 Token 原始位置排序后返回索引
    idx = torch.sort(idx, descending=False).values
    return idx


class OmniKVMulLayer(LlamaDecoderLayer):
    """
    针对 OmniKV 协同选择优化的 Llama 解码层封装。

    继承自 `LlamaDecoderLayer`，通过注入 Token 选择逻辑来控制缓存的筛选和多阶段加载。
    """
    def __init__(self, config, layer_idx):
        # 兼容性处理：注入 RoPE 缩放配置
        config.rope_scaling = config.rope_scaling_
        try:
            super().__init__(config, layer_idx)
        except Exception:
            # 针对 Llama 3.1 及以后版本的 Monkey Patch
            warning_once(logger, "检测到兼容性异常，确保正在使用 Llama 3.1 以上架构！")
            config.rope_scaling = None
            super().__init__(config, layer_idx)
            # 根据配置重新初始化 Attention 模块并应用 RoPE Patch
            self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](
                config=config, layer_idx=layer_idx
            )
            config.rope_scaling = config.rope_scaling_
            self.self_attn.rotary_emb = PatchLlamaRotaryEmbedding(config=config)

        self.config = config
        self.layer_idx = layer_idx
        self.prefill_len = None # 记录 Prefill 阶段的总长度
        self.cache_cls = get_cache_cls(config) # 获取当前选择的缓存类
        self.sparse_in_prefill = config.get("sparse_in_prefill", False)
        self.max_len_can_hold = config.get("max_len_can_hold", 32_000)
        self.attn_seg_sz = config.get("attn_seg_sz", 8000)
        # 解析需要执行 Token 选择的层索引
        self.do_select_layers = [
            int(i) for i in config.get("do_select_layers").split(",")
        ]
        self.hidden_state_window = None # 维护滑动窗口内的隐藏状态
        self.selector_cls = config.get("selector_cls", "softmax_before_last")
        self.window_size = config.get("window_size", 16)
        self.decode_step = 0

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[OmniKVMultiStageCache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        前向传播，嵌入了动态 KV 选择逻辑。
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Prefill 阶段逻辑
        if hidden_states.shape[1] > 1:
            self.prefill_len = hidden_states.shape[1]
            # 如果选择器需要滑动窗口历史且当前层是过滤层
            if (
                "last" not in self.selector_cls
                and self.layer_idx in self.do_select_layers
            ):
                self.hidden_state_window = hidden_states[:, -self.window_size :]
                self.decode_step = 1  # 标记进入生成阶段

        if past_key_value:
            assert isinstance(past_key_value, self.cache_cls)

        consider_len = self.prefill_len
        num_selected_tokens = self.config.get("num_of_selected_tokens", 4096)
        
        # 处理 Token 保留数量：可以是绝对值，也可以是相较于 Prefill 长度的比例
        if isinstance(num_selected_tokens, float):
            num_selected_tokens = int(num_selected_tokens * consider_len)

        # Decoding 阶段核心逻辑：触发 Token 选择
        if (
            hidden_states.shape[1] == 1
            and past_key_value
            and self.layer_idx in self.do_select_layers
        ):
            window_hs = hidden_states
            # 计算窗口内属于 Prefill 阶段的 Token 数量
            num_prefill_token_in_window = max(0, self.window_size - self.decode_step)
            
            if "last" not in self.selector_cls:
                # 更新滑动窗口隐藏状态
                self.hidden_state_window = torch.cat(
                    [self.hidden_state_window, hidden_states], dim=1
                )[
                    :, -self.window_size :
                ]  # noqa
                window_hs = self.hidden_state_window
                # 排除窗口内已有的 Prefill Token，避免重复选择
                consider_len -= num_prefill_token_in_window
                num_selected_tokens -= num_prefill_token_in_window
                num_selected_tokens = max(1, num_selected_tokens)
            
            # 调用通用选择器计算索引
            idx = select_tokens_by_attn_universal(
                self.self_attn,
                window_hs,
                position_ids,
                past_key_value,
                num_selected_tokens,
                consider_len,
                self.layer_idx,
                self.selector_cls,
            )
            
            # 如果使用非 'last' 策略，补全被剔除的窗口内 Prefill Token 索引
            if "last" not in self.selector_cls:
                idx = torch.cat(
                    [
                        idx,
                        torch.arange(
                            self.prefill_len - num_prefill_token_in_window,
                            self.prefill_len,
                            device=idx.device,
                        )[None, :].repeat(idx.shape[0], 1),
                    ],
                    dim=1,
                )  # noqa
            
            # 设置 GPU 缓存索引，触发异步搬运或筛选
            if self.config.get("dense_more", False):
                past_key_value.set_idx_on_gpu(idx, self.layer_idx)
            else:
                raise ValueError("OmniKV 目前仅支持 dense_more=True 模式")
            
            past_key_value.stage = "decoding"
            self.decode_step += 1

        # 执行原生的 Self Attention 计算（此时缓存对象内部会自动处理筛选逻辑）
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        hidden_states = residual + hidden_states

        # MLP 计算
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        # 对超长列进行分块计算，防止显存峰值过高
        hsl = torch.split(hidden_states, 4000, dim=1)
        hidden_states = [self.mlp(hs) for hs in hsl]
        hidden_states = torch.cat(hidden_states, dim=1)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class OmniKVMulModel(LlamaModel):
    """
    集成 OmniKV 层的 Llama 模型封装。
    """
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        # 关键改动：使用自定义的 OmniKVMulLayer 替换原生层
        self.layers = nn.ModuleList(
            [
                OmniKVMulLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # 初始化权重并应用最终处理
        self.post_init()


class OmniKVMulLM(LlamaForCausalLM):
    """
    OmniKV 推理的核心语言模型类。

    该类负责处理缓存类的初始化、多层筛选配置的分发以及最终的 Logits 生成。
    支持长文本推理所需的 RoPE Factor 动态调整。
    """
    def __init__(self, config: LlamaConfig):
        # 支持动态调整 RoPE Factor 以扩展上下文窗口
        if (fac := config.get("rope_factor", -1)) > 0:
            logger.warning(f"设置了自定义 rope_scaling (factor={fac})")
            config.rope_scaling = {"type": "dynamic", "factor": fac}
        
        # 缓存原始 RoPE 配置，用于后续层级别的 Patch
        config.rope_scaling_ = config.rope_scaling
        config.rope_scaling = None
        super().__init__(config)
        
        # 使用自定义的 OmniKVMulModel
        self.model = OmniKVMulModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.max_context_len = config.get("max_context_len", 50_000)
        self.cache_cls = get_cache_cls(config)
        
        # 初始化权重
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        前向传播，包含自动缓存转换逻辑。
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        n = input_ids.shape[1]
        
        # 缓存转换逻辑：将原生缓存转换为 OmniKV 专用的多阶段缓存
        if not isinstance(past_key_values, Cache):
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        
        if not isinstance(past_key_values, self.cache_cls):
            kwargs = {}
            # 根据配置名初始化多阶段或非拼接缓存
            if (
                cache_cls_name := self.config.get("cache_cls", "default")
            ) == "multi" or cache_cls_name == "without_pack":
                do_sel_layers = [
                    int(i) for i in self.config.get("do_select_layers").split(",")
                ]
                
                # 自动计算全量注意力层列表：
                # 包括选择层之前的所有层，选择层本身，以及最后一层
                full_attn_layers = (
                    list(range(0, do_sel_layers[0]))
                    + do_sel_layers
                    + [self.config.num_hidden_layers]
                )
                kwargs["full_attn_layers"] = full_attn_layers
                kwargs["num_hidden_layers"] = self.config.num_hidden_layers
                kwargs["num_wait_load_layers"] = self.config.get(
                    "num_wait_load_layers", 2
                )
                kwargs["real_offload"] = self.config.get("real_offload", True)
            else:
                raise NotImplementedError(f"不支持的缓存类型: {cache_cls_name}")
                
            past_key_values = self.cache_cls.from_dynamic_cache(
                past_key_values, **kwargs
            )

        # 标记当前推理阶段
        if n == 1:
            past_key_values.stage = "decoding"
        else:
            past_key_values.stage = "prefill"

        # 调用底层模型
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        # 生成预测 Logits
        hidden_states = outputs[0][:, -1:]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(
                self.vocab_size // self.config.pretraining_tp, dim=0
            )
            logits = [
                F.linear(hidden_states, lm_head_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            raise NotImplementedError("OmniKVMulLM 目前主要用于推理，未实现损失计算逻辑")

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        # 记录各阶段耗时
        if n > 1:
            logger.info(f"---prefill time {round(time_analyze(), 3)}s")
        else:
            logger.info(f"---decoding time {round(time_analyze(), 3)}s")

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
