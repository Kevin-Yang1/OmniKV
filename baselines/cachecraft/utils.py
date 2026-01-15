import torch
import math
from typing import List, Tuple

def compute_layer_scores(
    attn_weights: torch.Tensor,
    curr_start: int,
    curr_end: int,
    prefix_ranges: List[Tuple[int, int]]
) -> Tuple[float, float]:
    """
    计算单层的归一化注意力得分 a_l(C_i) 和 b_l(C_i)。
    
    Args:
        attn_weights (torch.Tensor): 单层的 Attention 矩阵，形状为 (Batch, Num_Heads, Seq_Len, Seq_Len)。
        curr_start (int): 当前文档块 C_i 在序列中的起始索引（包含）。
        curr_end (int): 当前文档块 C_i 在序列中的结束索引（不包含）。
        prefix_ranges (List[Tuple[int, int]]): 一个列表，包含所有前缀块 C_j 的 (start, end) 元组（即 j < i 的块）。
        
    Returns:
        Tuple[float, float]: 返回元组 (a_l, b_l)，分别代表归一化的块间注意力和块内注意力得分。
    """
    # 1. 预处理 Attention 权重
    # attn_weights 形状: (Batch, Num_Heads, Seq_Len, Seq_Len)
    # 我们假设 Batch Size = 1
    # inter_scores_tensor = inter_sum[0].mean(dim=0) -> 取 batch index 0, 对 heads 求平均
    # 因此这里也应该保持一致：此时我们专注分析第一个样本的 Attention Pattern
    avg_attn = attn_weights[0].mean(dim=0).float()
    
    # 获取当前块 C_i 的长度 |C_i|
    c_i_len = curr_end - curr_start
    if c_i_len <= 0:
        # 如果长度无效，返回 0 以避免错误
        return 0.0, 0.0
        
    # --- A. 计算块间注意力 (Inter-Attention, Eq 3) ---
    # 逻辑：a_l = sum( inter(C_i, C_j) / (|C_i| * |C_j|) ) for all j < i
    
    total_normalized_inter = 0.0
    
    for (prev_start, prev_end) in prefix_ranges:
        c_j_len = prev_end - prev_start
        if c_j_len <= 0:
            continue
            
        # 提取从当前块 C_i (Query, 行) 到前缀块 C_j (Key, 列) 的注意力子矩阵
        # 行索引范围: [curr_start, curr_end)
        # 列索引范围: [prev_start, prev_end)
        inter_block = avg_attn[curr_start:curr_end, prev_start:prev_end]
        
        # 计算 inter(C_i, C_j)：子矩阵元素求和
        inter_score = inter_block.sum().item()
        
        # 归一化 (Eq 9 左)：除以两块长度的乘积
        norm_score = inter_score / (c_i_len * c_j_len)
        
        # 累加到总的 a_l
        total_normalized_inter += norm_score
        
    a_l = total_normalized_inter
    
    # --- B. 计算块内注意力 (Intra-Attention, Eq 4) ---
    # 逻辑：b_l = intra(C_i) / (|C_i|^2)
    
    # 提取当前块 C_i 对自身的注意力子矩阵
    # 行列索引均为: [curr_start, curr_end)
    intra_block = avg_attn[curr_start:curr_end, curr_start:curr_end]
    
    # 应用下三角掩码 (Causal Mask)
    # 只保留 k <= l 的部分（即矩阵的下三角部分，包括对角线）
    # diagonal=0 表示保留对角线
    mask = torch.tril(torch.ones_like(intra_block), diagonal=0)
    masked_intra = intra_block * mask
    
    # 计算 intra(C_i)：掩码后的子矩阵求和
    intra_score = masked_intra.sum().item()
    
    # 归一化 (Eq 9 右)：除以当前块长度的平方
    b_l = intra_score / (c_i_len * c_i_len)
    
    return a_l, b_l

def compute_final_cci(layer_scores: List[Tuple[float, float]], epsilon: float = 1e-9) -> float:
    """
    聚合所有层的结果计算最终 CCI (Cache Context Impact)。
    
    Args:
        layer_scores (List[Tuple[float, float]]): 一个列表，包含每一层的 (a_l, b_l) 元组。
        epsilon (float): 防止除零异常的小量。
        
    Returns:
        float: 计算得到的 CCI 分数。
    """
    if not layer_scores:
        return 0.0
        
    # 跨层平均 (Eq 10)
    # 计算所有层 a_l 的平均值 bar_a
    a_vals = [s[0] for s in layer_scores]
    bar_a = sum(a_vals) / len(a_vals)
    
    # 计算所有层 b_l 的平均值 bar_b
    b_vals = [s[1] for s in layer_scores]
    bar_b = sum(b_vals) / len(b_vals)
    
    # 计算 CCI (Eq 11)
    # CCI = 1 / (1 + exp(- bar_a / bar_b))
    
    if bar_b < epsilon:
        # 边界情况：如果 chunk 内部注意力极低 (接近0)，
        # 则 Inter/Intra 比值趋向正无穷，CCI 趋向于 sigmoid(inf) = 1.0
        # 这意味着该 chunk 几乎完全依赖外部上下文，非常重要。
        return 1.0
    
    ratio = bar_a / bar_b
    
    # 应用 Sigmoid 变体公式
    # math.exp 可能溢出，如果 ratio 很大，exp(-ratio) 接近 0，结果接近 1
    # 如果 ratio 很小负数，exp(-ratio) 很大，结果接近 0
    try:
        exp_val = math.exp(-ratio)
    except OverflowError:
        # 如果 -ratio 太大（即 ratio 是很大的负数），exp 溢出，分母很大，CCI -> 0
        return 0.0
        
    cci = 1.0 / (1.0 + exp_val)
    
    return cci
