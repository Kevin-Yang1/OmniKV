import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
from baselines.cachecraft import utils

# 全局上下文，用于在 Monkey Patch 的 Forward 中传递信息
CURRENT_CONTEXT: Dict[str, Any] = {
    "mode": "off",  # "capture", "reuse", "off"
    "chunk_boundaries": [],  # List of dict: [{'start': start, 'end': end, 'hash': hash}, ...]
    "captured_data": {},  # layer_idx -> {chunk_hash -> {k:..., v:..., scores:...}}
}

def cache_craft_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Any] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    Monkey Patch 版本的 LlamaAttention.forward。
    支持 Cache-Craft 的 capture 逻辑。
    """
    bsz, q_len, _ = hidden_states.size()

    # 1. Projections
    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split((self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0)
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)
    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    # ================= Cache-Craft Capture Logic (Pre-RoPE) =================
    # 在应用 RoPE 之前，如果处于 capture 模式，保存 Pre-RoPE Keys
    pre_rope_key_states = None
    if CURRENT_CONTEXT["mode"] == "capture":
        # 我们暂时将全量的 Pre-RoPE Key 放入 context 的临时区，稍后根据 chunk 切分
        # 注意：这里可能会占用显存，后续会由 storage 移动到 CPU
        # 为了避免干扰后续计算，使用 detach 可能是明智的，但如果不需要梯度，直接引用即可
        pre_rope_key_states = key_states
    # ========================================================================

    # 获取当前输入的序列长度（即本次 Forward 的 Key 数量）
    kv_seq_len = key_states.shape[-2]
    
    # 如果存在 KV Cache（past_key_values），说明这是解码（Decoding）阶段或长序列分段处理
    if past_key_values is not None:
        # cache 更新依赖 layer_idx，必须确保它存在
        if self.layer_idx is None:
            raise ValueError("Layer index is needed for cache update")
            
        # 加上 Cache 中已有的序列长度
        # get_usable_length 会根据 Cache 实现（如 Window Attention）返回有效的历史长度
        # 最终 kv_seq_len 表示：[历史 KV 长度] + [当前 KV 长度]
        # 这个总长度将用于生成正确的位置索引（Position IDs）和正弦/余弦嵌入（Cos/Sin）
        kv_seq_len += past_key_values.get_usable_length(kv_seq_len, self.layer_idx)
    
    # 2. RoPE
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    
    # 注意：past_key_values 处理逻辑略过，假设 capture 模式是在 Prefill 阶段运行（无 past），
    # 或者我们只关注当前的 fresh tokens。如果是有 past_key_values 的情况（Decode），通常不会触发 Capture 整块。
    # 这里为了简化，我们关注 Prefill 阶段的 Capture。
    
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs=kwargs)

    # 3. Repeat KV for GQA
    # [GQA 处理] 如果是 Grouped Query Attention，复制 KV 头以匹配 Query 头数
    # 将 [batch, kv_heads, seq, dim] -> [batch, q_heads, seq, dim]
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # 4.  # [注意力分数计算]
    # 使用 manual matmul 以获取 weights 用于计算 CCI
    # attn_weights 形状为 [Batch, Heads, Query_Len, Key_Len]
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
    # [应用掩码] 加上一般是负无穷大的 mask，使 padding 或 future tokens 的 softmax 概率趋近于 0
    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            attn_weights = attn_weights + attention_mask
        else:
             attn_weights = attn_weights + attention_mask

    # [Softmax 归一化] 计算注意力概率，upcast 到 fp32 以保证数值稳定性
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    # ================= Cache-Craft Capture Logic (Profiling) =================
    if CURRENT_CONTEXT["mode"] == "capture" and pre_rope_key_states is not None:
        layer_idx = self.layer_idx
        chunk_info_list = CURRENT_CONTEXT.get("chunk_boundaries", [])
        
        # 确保当前层在 captured_data 中初始化
        if layer_idx not in CURRENT_CONTEXT["captured_data"]:
            CURRENT_CONTEXT["captured_data"][layer_idx] = {}

        # 遍历所有定义的 Chunk 进行分析
        for i, chunk_info in enumerate(chunk_info_list):
            c_hash = chunk_info['hash']
            start = chunk_info['start']
            end = chunk_info['end']
            
            # 安全检查
            if start >= q_len: continue
            real_end = min(end, q_len)
            
            # 1. 提取 KV Cache (Pre-RoPE from Step 1, Value from Step 1)
            # 注意：pre_rope_key_states 是 (BS, Num_KV_Heads, Seq_Len, Dim)
            # 我们需要保存的是针对该 Chunk 的部分
            # Assuming BS=1. For safety we clone.
            chunk_k = pre_rope_key_states[:, :, start:real_end, :].clone() 
            chunk_v = value_states[:, :, start:real_end, :].clone()        
            
            # 准备计算分数所需的前缀列表
            # prefix_ranges 应该是当前 chunk 之前的所有 chunk
            prefix_ranges = [(c['start'], c['end']) for c in chunk_info_list[:i]]
            
            # 2. 计算 CCI 相关的 (a_l, b_l)
            # attn_weights: (BS, Num_Heads, Seq_Len, Seq_Len)
            a_l, b_l = utils.compute_layer_scores(attn_weights, start, real_end, prefix_ranges)
            
            # 3. 计算 Inter-Attention Tensor (用于 Top-K 筛选)
            # 目标：该 Chunk 每个 Token 对前文（External Context）的注意力总和
            # Inter-Region: [0, start)
            # Attn Slice: [start:real_end, 0:start]
            # Sum over last dim (K dim) -> [Chunk_Len]
            if start > 0:
                # 提取当前 Chunk (Query) 对 前缀部分 (Key) 的注意力
                inter_attn_slice = attn_weights[:, :, start:real_end, 0:start] 
                # 求和得到每个 Token 对外部的总关注度: (BS, Num_Heads, Chunk_Len)
                inter_sum = inter_attn_slice.sum(dim=-1)
                # 对 Batch 和 Head 维度取平均 -> (Chunk_Len)
                inter_scores_tensor = inter_sum.mean(dim=(0, 1))
            else:
                chunk_len = real_end - start
                inter_scores_tensor = torch.zeros(chunk_len, device=attn_weights.device)

            # 4. 存入 Context
            CURRENT_CONTEXT["captured_data"][layer_idx][c_hash] = {
                "k_cache": chunk_k,
                "v_cache": chunk_v,
                "layer_scores": (a_l, b_l),
                "inter_scores_tensor": inter_scores_tensor
            }
    # ========================================================================

    # [注意力聚合] Score * V
    attn_output = torch.matmul(attn_weights, value_states)

    # [形状恢复] 确保输出尺寸正确
    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    # [维度还原] [batch, heads, seq, head_dim] -> [batch, seq, heads, head_dim]
    attn_output = attn_output.transpose(1, 2).contiguous()
    # -> [batch, seq, hidden_size]
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    # [输出投影] W_o * result
    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_values

def apply_monkey_patch(model):
    """
    将模型中所有 Attention 层替换为 cache_craft_forward。
    并注入 layer_idx 属性。
    """
    print("Applying Cache-Craft Monkey Patch...")
    count = 0
    for name, module in model.named_modules():
        if "LlamaAttention" in module.__class__.__name__:
            # 绑定方法到实例
            import types
            module.forward = types.MethodType(cache_craft_forward, module)
            
            # 确保 layer_idx 存在 (Transformer 模型通常已有，但以防万一)
            if not hasattr(module, 'layer_idx') or module.layer_idx is None:
                # 尝试从 name 解析 layer_idx (e.g., model.layers.0.self_attn)
                try:
                    parts = name.split('.')
                    # 通常是 layers[i] or h[i]
                    for part in parts:
                         if part.isdigit():
                             module.layer_idx = int(part)
                             break
                except (ValueError, IndexError):
                    print(f"Warning: Could not infer layer_idx for module {name}")
            
            count += 1

    print(f"Monkey Patch Applied Successfully to {count} attention layers.")
