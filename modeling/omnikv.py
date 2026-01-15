"""
文件名: omnikv.py
核心功能:
    实现 OmniKV 的进阶版多阶段动态选择模型 (`OmniKVMulLM`)。
    支持更复杂的 Token 选择策略 (Uniform, Exp, Last) 和多级缓存管理 (Multistage Cache)。
    适用于需要 Offloading 或复杂驱逐策略的场景 (`model_cls="multi"`).

主要类定义:
    1. OmniKVMulLayer (继承 LlamaDecoderLayer):
       - 【核心执行者】 覆写了 forward 方法，嵌入了 "Token 筛选" 和 "显存优化" 逻辑。
       - 负责调用 select_tokens_by_attn_universal 计算 Token 重要性。
       - 对 MLP 进行切片计算以降低显存峰值。

    2. OmniKVMulModel (继承 LlamaModel):
       - 【骨架搭建者】 Transformer 的主体结构。
       - 主要职责是将原生 Llama 的 Decoder Layers 强制替换为 OmniKVMulLayer。

    3. OmniKVMulLM (继承 LlamaForCausalLM):
       - 【顶层管理者】 面向用户的推理接口。
       - 负责 KV Cache 对象的生命周期管理 (如初始化 OmniKVMultiStageCache)。
       - 处理 Prefill/Decoding 阶段切换逻辑。
       - 支持 RoPE Factor 的动态调整以适应超长文本。

类之间的层级关系 (Hierarchy):
    OmniKVMulLM (顶层封装，含 LM Head)
      └── self.model: OmniKVMulModel (Transformer 主干)
            └── self.layers: List[OmniKVMulLayer] (N 个解码层堆叠)
                  └── self.self_attn: Attention (含 Token 选择逻辑)

依赖关系:
    - transformers: LlamaForCausalLM
    - modeling.spec_cache: OmniKVMultiStageCache (支持卸载/多级管理的缓存)
    - modeling.offload_select_once: 辅助功能

主要逻辑流程:
    1. OmniKVMulLM 初始化时加载 OmniKVMulLayer。
    2. 核心算法 `select_tokens_by_attn_universal`:
       - 提供多种策略计算 Token 重要性 (如将 Prefill 阶段所有 Query 的注意力能够累加)。
    3. 在 OmniKVMulLayer.forward 中：
       - 结合 `do_select_layers` 配置决定当前层是否需要筛选。
       - 更新 OmniKVMultiStageCache，可能涉及将 Token 移至 CPU 或丢弃。
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
        # 这里的注意力计算只是为了得到 Top-K 的索引 idx，后面还会调用FlashAttention再次计算，存在重复计算
        attn_score = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
            raw_attn.head_dim
        )
        # 在所有 Head 上取最大值，取 Q 序列的最后一个位置 [bsz, consider_len]
        # TODO: 这里所有头统一过滤可能不是最优的，可以考虑不同头选择不同 Token 的方案
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
        attn_score = torch.nn.functional.softmax(attn_score, dim=-1)  # to attn score
        attn_score = torch.max(
            attn_score[..., -1, :], dim=1
        ).values  # remove query, then head
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
        """
        OmniKVMulLayer 初始化函数。

        该函数在标准 `LlamaDecoderLayer` 初始化基础上进行了重要扩展，主要区别如下：
        1. **RoPE 兼容性与 Monkey Patch**: 
           - 增加了 try-except 块来处理 Llama 3.1+ 版本中 RoPE 配置可能导致的初始化异常。
           - 若标准初始化失败，则手动替换 Attention 模块的 Rotary Embedding (`PatchLlamaRotaryEmbedding`)，以支持自定义的 RoPE 因子或长上下文。
        2. **OmniKV 状态管理**:
           - 新增 `do_select_layers`: 解析配置，确定当前层是否属于 Token 筛选层。
           - 新增 `cache_cls`: 根据配置动态加载缓存策略类。
           - 新增 `hidden_state_window`: 用于存储滑动窗口内的隐藏状态，支持非 'last' 策略的 Token 选择。
           - 新增 `decode_step`: 追踪解码步数，配合滑动窗口使用。

        Args:
            config (LlamaConfig): 模型配置对象，包含 OmniKV 特有的参数 (如 `do_select_layers`, `selector_cls`)。
            layer_idx (int): 当前层索引。
        """
        # [兼容性处理] 注入 RoPE 缩放配置
        # 原始 config 可能被修改，这里使用 config.rope_scaling_ 恢复用于初始化的正确配置
        config.rope_scaling = config.rope_scaling_
        try:
            # 尝试调用父类 LlamaDecoderLayer 的标准初始化 在这里完成一层解码器中的所有模块创建
            super().__init__(config, layer_idx)
        except Exception:
            # [Llama 3.1+ 兼容补丁]
            # 如果标准初始化失败 (通常因 RoPE 参数格式变化)，则回退到无 RoPE 缩放配置进行初始化
            warning_once(logger, "检测到兼容性异常，确保正在使用 Llama 3.1 以上架构！")
            config.rope_scaling = None
            super().__init__(config, layer_idx)
            
            # 手动重新初始化 Attention 模块，并应用自定义的 Rotary Embedding Patch
            self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](
                config=config, layer_idx=layer_idx
            )
            config.rope_scaling = config.rope_scaling_
            self.self_attn.rotary_emb = PatchLlamaRotaryEmbedding(config=config)

        # 下面是 OmniKV 相关的初始化逻辑
        self.config = config # 保存完整的配置对象，以便在 forward 推理阶段动态读取超参数。父类 LlamaDecoderLayer 通常只保存 self.hidden_size 等特定属性，不保存整个 config 字典
        self.layer_idx = layer_idx # 让当前层知道“我是第几层”，虽然父类初始化时接收了 layer_idx。但它通常只将其传给 self.self_attn，Layer 本身往往不保存这个属性。如果不在这里显式保存，forward 函数里就拿不到这个索引了。
        
        # [OmniKV 特定属性初始化]
        self.prefill_len = None # 记录 Prefill 阶段的总长度
        self.cache_cls = get_cache_cls(config) # 获取当前选择的缓存类 (如 OmniKVMultiStageCache)
        
        # [OmniKV 专属配置参数] (非 Llama 原生)
        # 以下参数均为了支持 OmniKV 的长文本压缩和显存优化而引入：
        
        # sparse_in_prefill: 【KV 稀疏策略】 是否在 Prefill 阶段就启用稀疏注意力 (默认 False)。
        # 若为 True，则在处理 Prompt 时就开始丢弃 Token，可节省显存但可能影响理解。
        self.sparse_in_prefill = config.get("sparse_in_prefill", False)
        
        # max_len_can_hold: 【显存保护】 显存允许的最大上下文长度 (默认 32k)。用于防止 OOM 的安全阈值。
        self.max_len_can_hold = config.get("max_len_can_hold", 32_000)
        
        # attn_seg_sz: 【计算优化】 Attention/MLP 计算的分块大小 (默认 8000)。
        # 针对超长序列，将 Hidden States 切分为小块分别通过 MLP，以降低显存峰值 (时间换空间)。
        self.attn_seg_sz = config.get("attn_seg_sz", 8000)
        
        # 解析需要执行 Token 选择的层索引列表（过滤层）
        self.do_select_layers = [
            int(i) for i in config.get("do_select_layers").split(",")
        ]
        
        # [Token 选择策略相关的状态变量]
        # 维护滑动窗口内的隐藏状态 buffer。
        # 某些选择策略 (如 'uniform', 'exp') 不只看当前这一个 Token 的 Attention，
        # 而是需要参考过去一段时间窗口 (window_size) 内的 Attention 分布来决定谁重要。
        self.hidden_state_window = None 

        # selector_cls: 【选择器算法】
        # 决定计算 Token 重要分数的算法类型。
        # "last": 仅根据当前最新 Token 的注意力分布选择 (最快，由 Llama 原生行为启发)。
        # "uniform"/"exp": 综合考虑窗口内多个 Token 的注意力 (更稳健，但计算量稍大)。
        self.selector_cls = config.get("selector_cls", "softmax_before_last")

        # window_size: 【滑动窗口大小】 (默认 16)
        # 配合 'uniform'/'exp' 使用，指定回看多少个历史 Query 来计算平均注意力分数。使用 'last' 时无效。
        self.window_size = config.get("window_size", 16)

        # decode_step: 【解码步数计数器】
        # 用于跟踪当前处于解码的第几步。
        # 在处理滑动窗口时，这用于区分哪些 Hidden States 是新生成的，哪些是 Prompt 阶段遗留的，
        # 从而避免在窗口滑过 Prefill/Decoding 边界时重复计算或索引越界。
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
        [OmniKV 解码层前向传播]
        
        该函数是 Transformer 解码层的核心执行单元，负责处理输入特征、执行自注意力机制 (Self-Attention) 以及前馈网络 (MLP)。
        除此之外，它还集成了 OmniKV 的核心 Token 选择逻辑，在推理过程中动态地筛选和管理 KV Cache。

        Args:
            hidden_states (`torch.Tensor`):
                输入层级的隐藏状态，形状为 `(batch, seq_len, embed_dim)`。
            attention_mask (`torch.Tensor`, *可选*):
                注意力掩码，用于屏蔽 Padding Token 或在因果注意力中屏蔽未来 Token。
                形状通常为 `(batch, 1, query_len, key_len)`。
            position_ids (`torch.LongTensor`, *可选*):
                每个 Token 对应的位置 ID，用于 Rotary Embedding 计算。形状为 `(batch, seq_len)`。
            past_key_value (`OmniKVMultiStageCache`, *可选*):
                OmniKV 专用的多阶段缓存对象，存储了历史的 Key 和 Value 状态。
                如果 `use_cache=True`，则必须提供此参数。
            output_attentions (`bool`, *可选*, 默认为 `False`):
                是否返回注意力权重张量。
            use_cache (`bool`, *可选*, 默认为 `False`):
                是否使用并更新 KV 缓存。如果为 True，则必须传入 `past_key_value`。
            cache_position (`torch.LongTensor`, *可选*):
                当前输入 Token 在缓存序列中的绝对位置索引。

        Returns:
            `Tuple[torch.FloatTensor, ...]`:
            返回一个元组，包含：
            - `hidden_states` (`torch.FloatTensor`): 该层输出的隐藏状态。
            - `self_attn_weights` (`torch.FloatTensor`, *可选*): 如果 `output_attentions=True`，则包含注意力权重。
            - `present_key_value` (`Tuple[torch.FloatTensor]`, *可选*): 如果 `use_cache=True`，则包含当前层更新后的 KV 缓存。

        Raises:
            `AssertionError`: 如果传入的 `past_key_value` 类型与配置中的 `cache_cls` 不匹配。
            `ValueError`: 如果配置了不支持的模式 (如 non-dense_more 模式)。
        """
        # [残差连接准备] 保存原始输入用于后续的 Residual Add
        residual = hidden_states

        # [层归一化] Attention 前的 RMSNorm
        hidden_states = self.input_layernorm(hidden_states)

        # [Prefill 阶段逻辑]
        # 当输入序列长度 > 1 时，通常处于 Prompt 处理 (Prefill) 阶段
        if hidden_states.shape[1] > 1:
            # 记录 Prefill 的总长度，用于后续 Token 选择时确定范围
            self.prefill_len = hidden_states.shape[1]
            
            # [滑动窗口初始化]
            # 如果使用的 Token 选择策略不是仅依赖最后一个 Token ("last")，
            # 并且当前层是被标记为需要进行筛选的 "Filter Layer"
            if (
                "last" not in self.selector_cls
                and self.layer_idx in self.do_select_layers
            ):
                # 保存最近 window_size 个 Token 的 hidden state，作为后续计算注意力的参考
                self.hidden_state_window = hidden_states[:, -self.window_size :]
                # 初始化解码步数为 1，标记即将进入生成阶段
                self.decode_step = 1  

        # [缓存类型检查] 确保传入的 cache 对象符合预期
        if past_key_value:
            assert isinstance(past_key_value, self.cache_cls)

        # 确定需要考虑的历史 Token 长度 (通常就是 Prefill 长度)
        consider_len = self.prefill_len
        # 获取配置中的目标保留 Token 数量 (例如 4096)
        num_selected_tokens = self.config.get("num_of_selected_tokens", 4096)
        
        # [动态比例支持]
        # 如果 num_selected_tokens 是浮点数 (如 0.5)，则表示保留 Prefill 长度的一定比例
        if isinstance(num_selected_tokens, float):
            num_selected_tokens = int(num_selected_tokens * consider_len)

        # [Decoding 阶段核心逻辑：触发 Token 选择]
        # 条件：1. 输入长度为 1 (单步解码) 2. 有缓存对象 3. 当前层是筛选层
        if (
            hidden_states.shape[1] == 1
            and past_key_value
            and self.layer_idx in self.do_select_layers
        ):
            window_hs = hidden_states
            
            # 计算当前滑动窗口中有多少个 Token 实际上是属于 Prefill 阶段遗留下来的
            # 这是为了处理 "刚好从 Prefill 过渡到 Decoding" 的边界情况
            num_prefill_token_in_window = max(0, self.window_size - self.decode_step)
            
            # [非 Last 策略的窗口更新]
            if "last" not in self.selector_cls:
                # 将新 Token 的 hidden state 拼接到窗口 buffer 中
                self.hidden_state_window = torch.cat(
                    [self.hidden_state_window, hidden_states], dim=1
                )[
                    :, -self.window_size :
                ]  # noqa: 保持窗口固定大小
                window_hs = self.hidden_state_window
                
                # [排除重复计算]
                # 减去窗口中属于 Prefill 的长度，避免对那些已经处理过的 Token 重复计数
                consider_len -= num_prefill_token_in_window
                num_selected_tokens -= num_prefill_token_in_window
                num_selected_tokens = max(1, num_selected_tokens)
            
            # [执行 Token 选择算法]
            # 调用 select_tokens_by_attn_universal 计算出哪些 Token 是重要的，返回它们的索引 idx
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
            
            # [索引补全]
            # 如果使用了滑动窗口策略，因为上面排除了一部分 Prefill Token，这里需要把它们的索引补回来
            # 确保返回的索引是在完整序列中的正确位置
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
            
            # [应用筛选结果]
            # 将计算出的保留索引 idx 通知给 Cache 对象。
            # Cache 对象会根据此索引在后台异步地将重要 KV 从 CPU 搬回 GPU，或整理 GPU 显存。
            if self.config.get("dense_more", False):
                past_key_value.set_idx_on_gpu(idx, self.layer_idx)
            else:
                raise ValueError("OmniKV 目前仅支持 dense_more=True 模式")
            
            # 标记 Cache 状态为 decoding
            past_key_value.stage = "decoding"
            # 更新解码步数
            self.decode_step += 1

        # [执行 Self Attention 计算]
        # 这里调用标准的 LlamaAttention 。使用FlashAttention2时，即使output_attentions=True，也不会返回 attn 权重
        # 即使发生了 Token 筛选，Cache 对象内部的 update 方法会自动返回拼接好(筛选后历史 + 新 Token)的 KV。
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        
        # [残差连接一] Add Residual
        hidden_states = residual + hidden_states

        # [MLP 前的数据准备]
        # 保存用于下一个残差连接的输入
        residual = hidden_states
        # MLP 前的 RMSNorm
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # [MLP 计算与显存优化]
        # 使用 attn_seg_sz (如 4000) 对 Hidden States 进行切片。
        # 分块通过 MLP 可以显著降低中间激活值的显存峰值，对于超长序列推理至关重要。
        # hsl = torch.split(hidden_states, self.attn_seg_sz, dim=1)
        hsl = torch.split(hidden_states, 4000, dim=1)
        hidden_states = [self.mlp(hs) for hs in hsl]
        # 将分块计算结果拼接回完整张量
        hidden_states = torch.cat(hidden_states, dim=1)
        
        # [残差连接二] Add Residual
        hidden_states = residual + hidden_states

        # [构建返回值]
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
        """
        OmniKVMulModel 构造函数。
        搭建模型的骨架：Embedding -> N 个 OmniKV 层 -> RMSNorm。
        """
        # [继承与初始化]
        # 调用父类 LlamaModel 的初始化。
        # 注意：这里有点“浪费”，因为父类会先创建一套原生的 LlamaDecoderLayer 赋值给 self.layers。
        # 下面几行代码会立刻用 OmniKVMulLayer 覆盖掉它们。
        #TODO：super(LlamaModel, self).__init__(config) 可能更合适，可以避免层的重复创建 当前是先在父类创建了 LlamaDecoderLayer，再在这里用 OmniKVMulLayer 覆盖，其他代码全都一样，实际也覆盖了
        super().__init__(config)
        
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # [词嵌入层]
        # 负责将输入的 Token ID 映射为向量。
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        
        # [核心架构替换]
        # 覆盖 self.layers，将标准的 LlamaDecoderLayer 替换为支持 Token 选择的 OmniKVMulLayer。
        # 这是 OmniKV 能够拦截 forward 过程并实施 KV 压缩的关键注入点。
        self.layers = nn.ModuleList(
            [
                OmniKVMulLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        
        # [输出归一化]
        # Transformer 最后的 RMS Norm 层。
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # 默认关闭梯度检查点 (推理场景通常不需要，微调需开启)
        self.gradient_checkpointing = False

        # [初始化后处理]
        # 调用 Transformers 库的 standard hook，初始化权重 (如高斯分布) 并处理 tie_weights。
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
        # 保存缓存类以便后续检查
        # 并不存储KV Cache 实例，而是保存类本身以便在 forward 时进行类型检查，KV Cache 实例实际存储在后面的 past_key_values 中
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
        [中文功能描述]:
        OmniKVMulLM 的前向传播入口。
        负责协调 KV Cache 的状态初始化、分辨 Prefill/Decoding 阶段，并驱动底层 Transformer 模型进行特征提取，最后计算 Logits。

        Args:
            input_ids (`torch.LongTensor`, shape `(batch_size, sequence_length)`, *可选*):
                输入序列的 token indices。如果未提供，必须提供 `inputs_embeds`。
            attention_mask (`torch.Tensor`, shape `(batch_size, sequence_length)`, *可选*):
                用于避免对 padding token 进行注意力的掩码 (1 表示保留, 0 表示掩盖)。
            position_ids (`torch.LongTensor`, shape `(batch_size, sequence_length)`, *可选*):
                每个 token 的位置索引。
            past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *可选*):
                预先计算的 Key-Value 状态。在 OmniKV 中，这将被转换为特定的 `OmniKVMultiStageCache` 以支持分层存储和卸载。
            inputs_embeds (`torch.FloatTensor`, shape `(batch_size, sequence_length, hidden_size)`, *可选*):
                直接提供嵌入向量而不是 `input_ids`。
            labels (`torch.LongTensor`, shape `(batch_size, sequence_length)`, *可选*):
                用于计算语言建模损失的标签。注意：在此实现中若提供 labels 会抛出异常。
            use_cache (`bool`, *可选*):
                如果为 True，将返回 `past_key_values` 用于加速后续解码。
            output_attentions (`bool`, *可选*):
                是否返回所有层的注意力权重。
            output_hidden_states (`bool`, *可选*):
                是否返回所有层的隐藏状态。
            return_dict (`bool`, *可选*):
                是否返回 `CausalLMOutputWithPast` 对象而不是元组。
            cache_position (`torch.LongTensor`, *可选*):
                指示当前输入在 Cache 中的具体位置索引。

        Returns:
            `Union[Tuple, CausalLMOutputWithPast]`:
            默认返回 `CausalLMOutputWithPast` 对象，包含：
            - `loss` (None): 损失值 (未实现)。
            - `logits` (torch.FloatTensor): 词表大小的预测分数 [batch, seq_len, vocab_size]。
            - `past_key_values` (Cache): 更新后的 KV 缓存对象。
            - `hidden_states` (tuple, optional): 如果请求，返回各层的输出。
            - `attentions` (tuple, optional): 如果请求，返回各层的注意力权重。

        Raises:
            `NotImplementedError`: 如果提供了 `labels` 试图计算损失。
            `NotImplementedError`: 如果配置了不支持的 `cache_cls` 类型。
        """
        # [参数解析] 确定是否输出注意力权重
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        # [参数解析] 确定是否输出隐藏状态
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        # [参数解析] 确定返回值格式 (Dict or Tuple)
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        
        # 获取序列长度，用于判断所处阶段
        n = input_ids.shape[1]
        
        # [Cache 初始化与转换逻辑 - OmniKV 核心部分]
        # 步骤 1: 确保 past_key_values 是 DynamicCache 或兼容的新版 Cache 接口，而不是旧版的 tuple
        if not isinstance(past_key_values, Cache):
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        
        # 步骤 2: 确保 Cache 是 OmniKV 特有的 OmniKVMultiStageCache 类型
        # 如果不是 (例如第一次运行)，则需要根据配置进行初始化
        if not isinstance(past_key_values, self.cache_cls):
            kwargs = {}
            # 获取配置中的 Cache 类名，默认为 "default"
            if (
                cache_cls_name := self.config.get("cache_cls", "default")
            ) == "multi" or cache_cls_name == "without_pack":
                # 解析需要进行 Token 选择的层列表
                do_sel_layers = [
                    int(i) for i in self.config.get("do_select_layers").split(",")
                ]
                
                # 计算全量注意力层列表 (full_attn_layers):
                # 策略: 在第一个选择层之前的所有层 + 过滤层层本身 + 最后一层，这些层通常保留较完整的 Attention 信息
                full_attn_layers = (
                    list(range(0, do_sel_layers[0]))
                    + do_sel_layers
                    + [self.config.num_hidden_layers]
                )
                
                # 填充初始化参数
                kwargs["full_attn_layers"] = full_attn_layers
                kwargs["num_hidden_layers"] = self.config.num_hidden_layers
                # num_wait_load_layers: 异步卸载/加载时的缓冲层数
                kwargs["num_wait_load_layers"] = self.config.get(
                    "num_wait_load_layers", 2
                )
                # real_offload: 是否真正在物理内存上进行 CPU offloading
                kwargs["real_offload"] = self.config.get("real_offload", True)
            else:
                # 遇到未知的 Cache 类型配置则报错
                raise NotImplementedError(f"不支持的缓存类型: {cache_cls_name}")
            
            # 使用提取的参数初始化专用的 Cache 对象
            past_key_values = self.cache_cls.from_dynamic_cache(
                past_key_values, **kwargs
            )

        # [阶段标记]
        # 根据输入序列长度 n 判断当前是 "prefill" (n > 1) 还是 "decoding" (n == 1)
        # 这个标记会被传递给 Cache 对象，用于内部决定是否执行复杂的 Token 驱逐策略
        if n == 1:
            past_key_values.stage = "decoding"
        else:
            past_key_values.stage = "prefill"

        # [模型主体调用]
        # 将准备好的输入和 Cache 传入 Transformer 主干 (self.model)
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

        # [Logits 计算]
        # 提取最后一层的 Hidden States，只关注最后一个时间步 (next token prediction)
        # transformer 原始代码是 hidden_states = outputs[0]，在解码第一个token时会计算全部token的logits，浪费显存和计算，这里改为只计算最后一个token的logits
        hidden_states = outputs[0][:, -1:]
        
        # 处理张量并行 (Tensor Parallelism) 的情况
        if self.config.pretraining_tp > 1:
            # 如果启用了 TP，将 LM Head 权重切分并在不同部分计算，最后拼接
            lm_head_slices = self.lm_head.weight.split(
                self.vocab_size // self.config.pretraining_tp, dim=0
            )
            logits = [
                F.linear(hidden_states, lm_head_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            logits = torch.cat(logits, dim=-1)
        else:
            # 标准情况：直接通过线性层映射到词表大小
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        # [损失计算 - 未实现]
        # 作为一个专门用于长文本推理优化的模型，这里显式禁用了训练时的 Loss 计算
        loss = None
        if labels is not None:
            raise NotImplementedError("OmniKVMulLM 目前主要用于推理，未实现损失计算逻辑")

        # [返回结果处理]
        # 如果不需要字典格式，返回元组
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        # [性能分析日志]
        # 简单记录 Prefill 或 Decoding 阶段的耗时 (依赖全局函数 time_analyze)
        if n > 1:
            logger.info(f"---prefill time {round(time_analyze(), 3)}s")
        else:
            logger.info(f"---decoding time {round(time_analyze(), 3)}s")

        # 返回标准的 Output 对象
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
