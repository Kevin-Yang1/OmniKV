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
    past_key_value: Optional[Any] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    Monkey Patch 版本的 LlamaAttention.forward。
    支持 Cache-Craft 的 capture 逻辑。
    Updated for Transformers 4.45 compatibility.
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

    # ================= Cache-Craft Reuse Logic (Pre-RoPE injection) =================
    # 如果处于 reuse 模式，我们需要将从存储中检索到的 Pre-RoPE KV 注入到当前流中
    # 这一步必须在 RoPE 之前做，以便统一应用正确的位置编码
    if CURRENT_CONTEXT["mode"] == "reuse":
        layer_idx = self.layer_idx
        # 检查当前层是否有需要复用的数据
        reuse_payload = CURRENT_CONTEXT.get("reuse_payload", {})
        if layer_idx in reuse_payload and reuse_payload[layer_idx]:
            # payload 应该是一个 list of (k_chunk, v_chunk)
            # 我们将它们拼接起来
            retrieved_k_list = [item[0].to(key_states.device) for item in reuse_payload[layer_idx]]
            retrieved_v_list = [item[1].to(value_states.device) for item in reuse_payload[layer_idx]]
            
            # 将 Retrieval 部分拼接在当前计算的 KV 之前
            # 注意：这里假设 reused chunks 确实是当前内容的前缀 (Prefix)
            # key_states: (Batch, Num_KV_Heads, Seq_Len_Total, Head_Dim)
            key_states = torch.cat(retrieved_k_list + [key_states], dim=2)
            value_states = torch.cat(retrieved_v_list + [value_states], dim=2)
            
            # [重要] 位置编码调整
            # 我们现在的 key_states 变长了，包含了历史。
            # 如果传入的 position_ids 只是针对当前新 token 的（例如 [100, 101]），
            # 那么我们需要扩展 position_ids 以覆盖前面的 retrieved tokens (例如 [0...99, 100, 101])。
            # 通常 Llama 的 RoPE 实现会根据 KV 长度和 position_ids 配合。
            # 这里我们需要确保 position_ids 正确对齐。
            if position_ids is not None:
                # 计算 retrieved 部分的总长度
                prefix_len = sum(k.shape[2] for k in retrieved_k_list)
                
                # 构造前缀的 position_ids
                # 假设 retrieved 是从 0 开始连续的（通常如此）
                # position_ids: (Batch, Seq_Len_New) -> 我们需要 (Batch, Seq_Len_Old + Seq_Len_New)
                # 注意：这只是一个简单的假设，复杂的非连续 Reuse 可能需要更复杂的 Position ID 管理
                start_pos = position_ids[0, 0].item() - prefix_len
                # 防御性检查：
                start_pos = max(0, int(start_pos)) 
                
                prefix_pos_ids = torch.arange(start_pos, start_pos + prefix_len, device=position_ids.device).unsqueeze(0)
                if position_ids.shape[0] > 1:
                     prefix_pos_ids = prefix_pos_ids.expand(position_ids.shape[0], -1)
                
                position_ids = torch.cat([prefix_pos_ids, position_ids], dim=1)

    # ================= Cache-Craft Capture Logic Part 1: Save Raw KV =================
    # 在应用 RoPE 之前，立即保存 Pre-RoPE Keys 和 Pre-Repeat Values
    # 这样可以尽早锁定原始数据，并支持后续可能的 key_states 显存释放优化
    if CURRENT_CONTEXT["mode"] == "capture":
        layer_idx = self.layer_idx
        chunk_info_list = CURRENT_CONTEXT.get("chunk_boundaries", [])
        
        # 确保当前层在 captured_data 中初始化
        if layer_idx not in CURRENT_CONTEXT["captured_data"]:
            CURRENT_CONTEXT["captured_data"][layer_idx] = {}

        for i, chunk_info in enumerate(chunk_info_list):
            c_hash = chunk_info['hash']
            start = chunk_info['start']
            end = chunk_info['end']
            
            if start >= q_len: continue
            real_end = min(end, q_len)
            
            # 1. 提取并保存 KV Cache (Pre-RoPE, Pre-Repeat)
            # Assuming BS=1. For safety we clone.
            chunk_k = key_states[:, :, start:real_end, :].clone() 
            chunk_v = value_states[:, :, start:real_end, :].clone()
            
            # 初始化该 chunk 的存储条目
            if c_hash not in CURRENT_CONTEXT["captured_data"][layer_idx]:
                 CURRENT_CONTEXT["captured_data"][layer_idx][c_hash] = {}
            
            CURRENT_CONTEXT["captured_data"][layer_idx][c_hash]["k_cache"] = chunk_k
            CURRENT_CONTEXT["captured_data"][layer_idx][c_hash]["v_cache"] = chunk_v
    # ========================================================================

    # 获取当前输入的序列长度（即本次 Forward 的 Key 数量）
    kv_seq_len = key_states.shape[-2]
    
    # 如果存在 KV Cache（past_key_value），说明这是解码（Decoding）阶段或长序列分段处理
    if past_key_value is not None:
        # cache 更新依赖 layer_idx，必须确保它存在
        if self.layer_idx is None:
            raise ValueError("Layer index is needed for cache update")
            
        # 加上 Cache 中已有的序列长度
        # get_usable_length 会根据 Cache 实现（如 Window Attention）返回有效的历史长度
        # 最终 kv_seq_len 表示：[历史 KV 长度] + [当前 KV 长度]
        # 这个总长度将用于生成正确的位置索引（Position IDs）和正弦/余弦嵌入（Cos/Sin）
        if hasattr(past_key_value, "get_usable_length"):
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        else:
            # Fallback for older transformers or simple tuple caches
            if isinstance(past_key_value, tuple):
                 kv_seq_len += past_key_value[self.layer_idx][0].shape[-2]
            else:
                 # Default behavior for Cache objects without explicit usable length method (unlikely in new versions)
                 kv_seq_len += past_key_value.get_seq_length(self.layer_idx)
    
    # 2. RoPE
    # Transformers 4.41+ LlamaRotaryEmbedding.forward(x, position_ids)
    cos, sin = self.rotary_emb(value_states, position_ids)
    
    # 注意：past_key_values 处理逻辑略过，假设 capture 模式是在 Prefill 阶段运行（无 past），
    # 或者我们只关注当前的 fresh tokens。如果是有 past_key_values 的情况（Decode），通常不会触发 Capture 整块。
    # 这里为了简化，我们关注 Prefill 阶段的 Capture。
    
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs=kwargs)

    # 3. Repeat KV for GQA
    # [GQA 处理] 如果是 Grouped Query Attention，复制 KV 头以匹配 Query 头数
    # 将 [batch, kv_heads, seq, dim] -> [batch, q_heads, seq, dim]
    # 显存优化：此时原始 V 已经被 Part 1 保存，这里可以直接覆盖变量名，释放旧内存
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

    # ================= Cache-Craft Capture Logic Part 2: Compute Scores =================
    if CURRENT_CONTEXT["mode"] == "capture":
        layer_idx = self.layer_idx
        chunk_info_list = CURRENT_CONTEXT.get("chunk_boundaries", [])
        
        # 遍历所有定义的 Chunk 计算分数
        for i, chunk_info in enumerate(chunk_info_list):
            c_hash = chunk_info['hash']
            start = chunk_info['start']
            end = chunk_info['end']
            
            if start >= q_len: continue
            real_end = min(end, q_len)
            
            # 准备计算分数所需的前缀列表
            prefix_ranges = [(c['start'], c['end']) for c in chunk_info_list[:i]]
            
            # 2. 计算 CCI 相关的 (a_l, b_l)
            a_l, b_l = utils.compute_layer_scores(attn_weights, start, real_end, prefix_ranges)
            
            # 3. 计算 Inter-Attention Tensor
            # TODO: 这里去掉注意力汇点是不是更好？
            # 计算当前 Chunk 中每个 Token 对前文（External Context）的总注意力权重
            if start > 0:
                # inter_attn_slice: (Batch, Num_Heads, Chunk_Len, Prefix_Len)
                inter_attn_slice = attn_weights[:, :, start:real_end, 0:start] 
                
                # 对最后一个维度（Key/Context）求和，得到每个 Token 对外部的总关注度
                # inter_sum: (Batch, Num_Heads, Chunk_Len)
                inter_sum = inter_attn_slice.sum(dim=-1)
                
                # 取 Batch 中的第一个样本 (index 0)，并对 Heads 维度求平均
                # 结果 inter_scores_tensor 形状为 (Chunk_Len,)，即每个 Token 一个分数
                inter_scores_tensor = inter_sum[0].mean(dim=0)
            else:
                chunk_len = real_end - start
                inter_scores_tensor = torch.zeros(chunk_len, device=attn_weights.device)

            # 4. 补充填入分数信息
            # 注意：k/v cache 已经在 Part 1 填入
            if c_hash in CURRENT_CONTEXT["captured_data"][layer_idx]:
                 CURRENT_CONTEXT["captured_data"][layer_idx][c_hash]["layer_scores"] = (a_l, b_l)
                 CURRENT_CONTEXT["captured_data"][layer_idx][c_hash]["inter_scores_tensor"] = inter_scores_tensor
    # ========================================================================

    # [注意力聚合] 使用 value_states 进行计算
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

    return attn_output, attn_weights, past_key_value

def apply_monkey_patch(model):
    """
    将模型中所有 Attention 层替换为 cache_craft_forward。
    并注入 layer_idx 属性。
    """
    print("Applying Cache-Craft Monkey Patch...")
    count = 0
    for name, module in model.named_modules():
        # 支持 LlamaAttention, LlamaFlashAttention2, LlamaSdpaAttention 等变体
        class_name = module.__class__.__name__
        if "Llama" in class_name and "Attention" in class_name:
            # 排除 CrossAttention (如果存在)
            if "Cross" in class_name:
                continue

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
