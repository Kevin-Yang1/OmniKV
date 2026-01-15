import math
import torch
from typing import List, Set, Union
from baselines.cachecraft import utils

class CacheCraftController:
    def __init__(self, alpha: float = 1.0):
        """
        初始化控制器。
        
        Args:
            alpha (float): CFO 计算中的超参数 alpha (默认 1.0)。
        """
        self.alpha = alpha

    def calculate_cci(self, layer_scores: List[tuple]) -> float:
        """
        计算缓存上下文影响 (CCI)。
        直接调用 utils.compute_final_cci。
        
        Args:
            layer_scores: 包含每一层 (a_l, b_l) 的列表。
            
        Returns:
            float: 计算出的 CCI 值。
        """
        return utils.compute_final_cci(layer_scores)

    def calculate_beta_prime(self, chunk_hash: str, old_prefix_hashes: List[str], new_prefix_hashes: List[str]) -> float:
        """
        计算前缀重叠分数 (Beta Prime)。
        Beta' = |Old_Prefix ∩ New_Prefix| / |Old_Prefix|
        衡量生成 Cache 时依赖的旧上下文有多少在当前上下文中被保留了。
        
        Args:
            chunk_hash (str): 当前块的 Hash (为了接口完整性保留，暂时不用)。
            old_prefix_hashes (List[str]): 生成 Cache 时的前缀 Chunk Hash 列表。
            new_prefix_hashes (List[str]): 当前推理时的前缀 Chunk Hash 列表。
            
        Returns:
            float: 0.0 到 1.0 之间的重叠分数。
        """
        old_set = set(old_prefix_hashes)
        new_set = set(new_prefix_hashes)
        
        if not old_set:
            # 如果旧前缀为空（例如这是文档的第一个 chunk），意味着它之前没有依赖外部上下文。
            # 这种情况下，前缀是否变化对它没有负面影响。
            # 返回 1.0 表示 "状态完美保留" (即使是空的)。
            return 1.0

        intersection = old_set.intersection(new_set)
        
        # 根据逻辑：(交集数量 / 旧前缀集合大小)
        beta_prime = len(intersection) / len(old_set)
        return beta_prime

    def get_recompute_tokens(
        self, 
        chunk_len: int, 
        cci: float, 
        beta_prime: float, 
        inter_scores_tensor_list: List[torch.Tensor]
    ) -> List[int]:
        """
        决策函数：计算 CFO 并筛选需要重算的 Token 索引。
        
        Args:
            chunk_len (int): 当前块的长度。
            cci (float): 缓存上下文影响分数。
            beta_prime (float): 前缀重叠分数。
            inter_scores_tensor_list (List[torch.Tensor]): 每一层的注意力敏感度张量列表，每个形状为 [Chunk_Len]。
            
        Returns:
            List[int]: 需要重算的 Token 索引列表 (升序排列)。
        """
        # 1. 计算 Cache Fix Overhead (CFO)
        # CFO = alpha * CCI * (1 - beta')
        # 如果 beta' = 1 (前缀完全一致), CFO = 0, 不需要重算。
        # 如果 cci = 0 (不依赖上下文), CFO = 0, 不需要重算。
        cfo = self.alpha * cci * (1.0 - beta_prime)
        
        # 2. 计算需要重算的 token 数量
        # num_recompute = ceil(CFO * chunk_len)
        num_recompute = math.ceil(cfo * chunk_len)
        
        # 边界处理：不需要重算
        if num_recompute <= 0:
            return []
            
        # 边界处理：全部重算
        if num_recompute >= chunk_len:
            return list(range(chunk_len))
            
        # 3. 核心筛选逻辑：选择最敏感的 Token
        # inter_scores_tensor_list: List[Tensor], 每个 Tensor 形状 [Chunk_Len]
        # 首先确保 Tensor 在同一设备上 (通常 storage 里存的是 CPU 上的)
        
        # 在层维度上堆叠: [Num_Layers, Chunk_Len]
        stacked_scores = torch.stack(inter_scores_tensor_list)
        
        # 对所有层取平均，得到综合敏感度: [Chunk_Len]
        # 这代表了每个 Token 对外部前缀的平均注意力强度 (Inter-Attention)
        avg_scores = stacked_scores.mean(dim=0)
        
        # 选出数值最大的前 num_recompute 个 Token 的索引
        # topk 返回 (values, indices)
        _, top_indices = torch.topk(avg_scores, k=num_recompute)
        
        # 转换为列表并排序
        recompute_indices = top_indices.tolist()
        recompute_indices.sort()
        
        return recompute_indices
