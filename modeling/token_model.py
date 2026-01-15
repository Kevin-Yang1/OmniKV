"""
文件名: token_model.py
核心功能:
    实现 OmniKV 的基础版动态 Token 选择模型 (`TokenLM`)。
    该模型在推理 Decode 阶段，针对每一层计算 Attention Score，
    只保留 Top-K 个最重要的 KV Cache，从而大幅降低显存占用。
    适用于主要实验配置中的 `model_cls="token"`。

依赖关系:
    - transformers: LlamaForCausalLM, LlamaDecoderLayer
    - modeling.spec_cache: DynamicSubCache (核心缓存结构)
    - modeling.compressor: OmniKVCompressorConfig

主要逻辑流程:
    1. TokenLM 初始化时加载 TokenLayer 替代原生 DecoderLayer。
    2. 在 TokenLayer.forward中：
       - 判断是否处于 Decoding 阶段 (seq_len=1)。
       - 若是，调用 `select_tokens_by_attn` 计算当前 Query 对历史 Key 的关注度。
       - 获取 Top-K 索引，更新 `past_key_value` (DynamicSubCache) 的 `out_idx`。
    3. Self-Attention 计算只使用被选中的 Token。
"""
import torch
import time
from transformers.models.llama.modeling_llama import *
from modeling.compressor import OmniKVCompressorConfig
from modeling.spec_cache import DynamicSubCache
import os
import pickle
from tiny_tools.log import logger

# 用于全局耗时分析的变量
last_call_t = time.time()


def time_analyze():
    """
    计算并返回自上次调用以来的时间间隔，用于性能打点。

    Returns:
        float: 耗时秒数（保留四位小数）。
    """
    global last_call_t
    temp = round(time.time() - last_call_t, 4)
    last_call_t = time.time()
    return temp


# Token 分析器全局状态
token_analyser = None
large_tokens_percentage = {}
cal_cnt = 0
if os.environ.get("CHECK_TOKENS", False):
    token_analyser = {
        "idx": {}, # 存储各层选中的索引
        "iou": {}, # 存储跨步/跨层的交并比
    }


def tensor_union(t1, t2):
    """计算两个一维张量的并集节点数。"""
    t1_flat = t1.view(-1)
    t2_flat = t2.view(-1)
    # 合并张量
    combined = torch.cat((t1_flat, t2_flat))
    # 使用unique去除重复值，注意sorted参数保持结果有序
    union_result = torch.unique(combined, sorted=True)
    return union_result


def tensor_intersection(t1, t2):
    """计算两个一维张量的交集。"""
    t1_flat = t1.view(-1)
    t2_flat = t2.view(-1)
    # 计算交集，这里使用了isin方法来找出t1中也存在于t2中的元素
    intersection_result = torch.unique(t1_flat[torch.isin(t1_flat, t2_flat)])
    return intersection_result


def cal_iou(idx, iou):
    """
    计算不同层之间选中 Token 索引的 IoU (交并比)，用于分析 Token 选择的一致性。
    
    Args:
        idx (dict): 层索引到选定 Token ID 张量的映射。
        iou (dict): 存储结果的字典。
    """
    for i, a in idx.items():
        for j, b in idx.items():
            if f"{i} {j}" not in iou:
                iou[f"{i} {j}"] = []
            # 计算并记录交并比
            iou[f"{i} {j}"] += ([tensor_intersection(a, b).shape[0] / tensor_union(a, b).shape[0]])
    global cal_cnt
    cal_cnt += 1
    # 每 100 次计算持久化一次结果到磁盘
    if cal_cnt % 100 == 0:
        with open('debug_logs/iou.pkl', 'wb') as _out:
            pickle.dump(iou, _out)


def count_big_tokens(attn_weights, layer_idx):
    """
    统计注意力权重中超过阈值 (0.01) 的 Token 比例，分析稀疏度。
    """
    if layer_idx not in large_tokens_percentage:
        large_tokens_percentage[layer_idx] = []
    # 转换为 float32 进行 softmax 保证精度，再转回 fp16
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
    large_tokens_percentage[layer_idx] += [(attn_weights > 0.01).sum().item() / attn_weights.numel()]
    if cal_cnt % 100 == 0:
        with open('debug_logs/big_tokens.pkl', 'wb') as _out:
            pickle.dump(large_tokens_percentage, _out)


class TokenConfig(OmniKVCompressorConfig):
    """
    Token 选择模型的配置类，扩展了 LlamaConfig。
    """
    def set_config(self, **kwargs):
        """批量设置配置属性。"""
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _rope_scaling_validation(self):
        """跳过 RoPE Scaling 验证，以兼容 Llama-3.1 等特定版本。"""
        logger.warning("为llama3.1做的patch，不验证了")
        return


def select_tokens_by_attn(raw_attn, hidden_states, position_ids, past_key_value: DynamicCache,
                          num_selected_tokens, layer_idx=None, ensure_head_tail=False):
    """
    根据当前的注意力分数动态选择最重要的 Token。

    Purpose:
        通过计算当前 Token (Query) 对所有历史 KV (Key) 的注意力权重，筛选出权重最高的 K 个 Token 索引。

    Args:
        raw_attn: 当前层的 LlamaAttention 模块。
        hidden_states (tensor): 当前步输入的隐藏状态 [bsz, 1, dim]。
        position_ids (tensor): 位置编码 ID。
        past_key_value (DynamicCache): 当前缓存对象。
        num_selected_tokens (int): 计划选出的 Token 数量。
        layer_idx (int, optional): 当前层索引。
        ensure_head_tail (int/bool, optional): 是否强制包含句首和句尾特定长度的 Token。

    Returns:
        torch.LongTensor: 选中的 Token 索引 [bsz, num_selected_tokens]。
    """
    bsz, q_len, _ = hidden_states.size()
    # 确保是 Decoding 阶段（q_len=1）且存在缓存
    assert q_len == 1 and past_key_value

    # 处理张量并行逻辑（兼容原生 Transformers 实现）
    if raw_attn.config.pretraining_tp > 1:
        key_value_slicing = (raw_attn.num_key_value_heads * raw_attn.head_dim) // raw_attn.config.pretraining_tp
        query_slices = raw_attn.q_proj.weight.split(
            (raw_attn.num_heads * raw_attn.head_dim) // raw_attn.config.pretraining_tp, dim=0
        )
        key_slices = raw_attn.k_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(raw_attn.config.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(raw_attn.config.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)
    else:
        # 获取当前步的 Query 和 Key
        query_states = raw_attn.q_proj(hidden_states)
        key_states = raw_attn.k_proj(hidden_states)

    # 规范化张量形状 [bsz, heads, 1, dim]
    query_states = query_states.view(bsz, q_len, raw_attn.num_heads, raw_attn.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, raw_attn.num_key_value_heads, raw_attn.head_dim).transpose(1, 2)

    # 应用 RoPE 位置旋转
    cos, sin = raw_attn.rotary_emb(key_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    del key_states # 只需要全量 Key 缓存来做选择
    # 从缓存中提取并计算完整的注意力权重
    all_key_states = past_key_value.key_cache[raw_attn.layer_idx]
    # 对超长缓存进行切分处理，防止显存爆炸
    all_key_states = all_key_states.split(32000, dim=-2)
    attn_weights = []
    for key_states in all_key_states:
        key_states = repeat_kv(key_states, raw_attn.num_key_value_groups)
        # 矩阵乘法 QK^T
        _attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(raw_attn.head_dim)
        attn_weights += [_attn_weights]
    attn_weights = torch.cat(attn_weights, dim=-1)
    
    if os.environ.get("CHECK_TOKENS", False):
        count_big_tokens(attn_weights, layer_idx)

    # 在所有 Head 上取最大值，作为该 Token 的重要性分数 [bsz, seq_len]
    attn_weights = torch.max(attn_weights[..., -1, :], dim=1).values
    
    # 特殊保护：强制包含 Sink Token (句首) 和最新生成的 Token (句尾)
    if ensure_head_tail:
        attn_weights[:, :ensure_head_tail] += 10 # 极大值加成
        attn_weights[:, -ensure_head_tail:] += 10
        
    num_selected_tokens = min(num_selected_tokens, attn_weights.shape[-1])
    # 选取 Top-K 索引
    v, idx = torch.topk(attn_weights, k=num_selected_tokens, dim=-1)
    return idx


class TokenLayer(LlamaDecoderLayer):
    """
    针对 OmniKV 优化的 Llama 解码层。
    
    该层继承自 `LlamaDecoderLayer`，在 Decoding 阶段引入了动态 Token 选择逻辑。
    """
    def __init__(self, config, layer_idx):
        super(TokenLayer, self).__init__(config, layer_idx)
        self.config = config
        self.layer_idx = layer_idx
        # 从配置中获取保留的 Token 数量，默认为 4096
        self.default_k = self.config.get('num_of_selected_tokens', 4096)
        # 是否保护句首/句尾 Token
        self.ensure_ht = self.config.get('ensure_head_tail_len', False)
        # 记录 Prefill 阶段的序列长度，用于计算百分比模式下的 K
        self.prefill_len = None

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        解码层前向传播，包含 KV 缓存的选择逻辑。
        """
        residual = hidden_states

        # 层归一化
        hidden_states = self.input_layernorm(hidden_states)
        
        # 记录 Prefill 长度
        if hidden_states.shape[1] > 1:
            self.prefill_len = hidden_states.shape[1]
            
        # 缓存转换：将原生 DynamicCache 增强为支持索引筛选的 DynamicSubCache
        if past_key_value:
            assert isinstance(past_key_value, DynamicCache)
            if not isinstance(past_key_value, DynamicSubCache):
                past_key_value = DynamicSubCache.from_dynamic_cache(past_key_value)
                
        # 核心逻辑：在 Decoding 阶段 (len=1) 且达到起始层阈值时进行 Token 选择
        if hidden_states.shape[1] == 1 and past_key_value and \
                self.layer_idx >= self.config.get('decoder_start_layer_idx'):
            # 获取当前层特定的 K 值
            k = self.config.get(f'num_of_selected_tokens_{self.layer_idx}', self.default_k)
            # 如果 K 是浮点数，则视为原始长度的比例
            if isinstance(k, float):
                k = int(k * self.prefill_len)
            
            # 运行注意力权重分析并挑选索引
            idx = select_tokens_by_attn(self.self_attn, hidden_states, position_ids, past_key_value, k,
                                        layer_idx=self.layer_idx, ensure_head_tail=self.ensure_ht)
            
            # 将索引设置到缓存中，后续的 self_attn(forward) 将只使用这些选中的 KV
            if past_key_value:
                past_key_value.set_out_idx(idx, self.layer_idx)
                
            # 调试模式：记录索引并计算 IoU
            if os.environ.get("CHECK_TOKENS", False):
                token_analyser['idx'][self.layer_idx] = idx
                if self.layer_idx == self.config.num_hidden_layers - 1:
                    cal_iou(**token_analyser)

        # 自注意力计算
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

        # 全连接层 (MLP) 计算
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # 这里的 split 操作是对超长上下文的一种特殊处理，确保 MLP 计算不溢出或优化计算开销
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


class TokenModel(LlamaModel):
    """
    封装了 TokenLayer 的 Llama 模型。
    """
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        # 关键改动：使用 TokenLayer 替代标准的 LlamaDecoderLayer
        self.layers = nn.ModuleList(
            [TokenLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # 初始化权重
        self.post_init()


class TokenLM(LlamaForCausalLM):
    """
    OmniKV 推理的核心模型类 (Llama 架构)。
    
    集成了动态 KV 选择、可调节 NTK ROPE 缩放等特性。
    """
    def __init__(self, config):
        # 支持动态 RoPE 缩放 (NTK 扩展)，用于处理超出训练长度的上下文
        ntk_factor = config.get('ntk_factor', -1)
        if ntk_factor > 0:
            logger.warning("在TokenLM中直接设置了rope_scaling")
            config.rope_scaling = {"type": "dynamic", "factor": config.get('ntk_factor', 8.0)}
        super().__init__(config)
        
        # 使用自定义的 TokenModel
        self.model = TokenModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重
        self.post_init()

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # 记录输入 Token 数量
        n = input_ids.shape[1]
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用底层的 TokenModel 前向传播
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

        # 仅取最后一个 Token 的隐藏状态用于生成 Logits
        hidden_states = outputs[0][:, -1:]
        
        # 处理张量并行逻辑下的 LM Head
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        # 计算损失（如果提供了 labels）
        if labels is not None:
            # 将 logits 和 labels 错位对齐以计算交叉熵
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        # 打印耗时分析日志
        if n > 1:
            logger.info(f"---prefill time {round(time_analyze(), 3)}s")
        else:
            logger.info(f"---decoding time {round(time_analyze(), 3)}s")

        # 返回符合 Transformers 规范的输出对象
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
