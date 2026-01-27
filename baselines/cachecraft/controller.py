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

    def calculate_beta_prime(
        self,
        chunk_hash: str,
        old_prefix_hashes: List[str],
        new_prefix_hashes: List[str],
        prefix_inter_scores: List[List[float]] = None
    ) -> float:
        """
        先计算前缀重叠分数 (Beta Prime)。
        Beta = |Old_Prefix ∩ New_Prefix| / |Old_Prefix|
        衡量生成 Cache 时依赖的旧上下文有多少在当前上下文中被保留了。
        再乘以顺序惩罚因子 (1 - gamma)。
        gamma 使用归一化的 Kendall-Tau 距离。
        
        Args:
            chunk_hash (str): 当前块的 Hash (为了接口完整性保留，暂时不用)。
            old_prefix_hashes (List[str]): 生成 Cache 时的前缀 Chunk Hash 列表。
            new_prefix_hashes (List[str]): 当前推理时的前缀 Chunk Hash 列表。
            prefix_inter_scores (List[List[float]]): 每层对各前缀块的 inter 分数列表。
            
        Returns:
            float: 0.0 到 1.0 之间的重叠分数。 1.0 表示前缀完全一致，0.0 表示完全不重叠。
        """
        old_set = set(old_prefix_hashes)
        new_set = set(new_prefix_hashes)
        
        if not old_set:
            # 如果旧前缀为空（例如这是文档的第一个 chunk），意味着它之前没有依赖外部上下文。
            # 这种情况下，前缀是否变化对它没有负面影响。
            # 返回 1.0 表示 "状态完美保留" (即使是空的)。
            return 1.0
        # 论文公式 (6): beta = sum_{j in old∩new} inter(C_i, C_j) / sum_{j in old} inter(C_i, C_j)
        if prefix_inter_scores:
            # 前缀哈希与分数一一对应
            old_hashes = old_prefix_hashes
            num_layers = len(prefix_inter_scores)
            if not num_layers:
                return 1.0

            # 计算每个前缀块在多层上的分数总和
            prefix_score_sum = {}
            for layer_scores in prefix_inter_scores:
                for idx, h in enumerate(old_hashes):
                    if idx >= len(layer_scores):
                        continue
                    prefix_score_sum[h] = prefix_score_sum.get(h, 0.0) + float(layer_scores[idx])

            denom = 0.0 # 分母：旧前缀的总分数
            numer = 0.0 # 分子：旧前缀与新前缀交集的总分数
            for h, s in prefix_score_sum.items():
                denom += s
                if h in new_set:
                    numer += s

            if denom <= 0.0:
                return 1.0
            beta = numer / denom
            gamma = self._kendall_tau_distance(old_prefix_hashes, new_prefix_hashes)
            return beta * (1.0 - gamma)

        # 回退：基于 Hash 的简单集合交集 + 顺序惩罚
        intersection = old_set.intersection(new_set)
        beta = len(intersection) / len(old_set)
        gamma = self._kendall_tau_distance(old_prefix_hashes, new_prefix_hashes)
        return beta * (1.0 - gamma)

    @staticmethod
    def _kendall_tau_distance(old_prefix_hashes: List[str], new_prefix_hashes: List[str]) -> float:
        """
        归一化 Kendall-Tau 距离，仅在交集元素上比较顺序。
        """
        if not old_prefix_hashes or not new_prefix_hashes:
            return 0.0

        new_pos = {h: i for i, h in enumerate(new_prefix_hashes)}
        overlap = [h for h in old_prefix_hashes if h in new_pos]
        m = len(overlap)
        if m < 2:
            return 0.0

        D = 0
        for i in range(m):
            for j in range(i + 1, m):
                if new_pos[overlap[i]] > new_pos[overlap[j]]:
                    D += 1

        T = m * (m - 1) / 2
        return D / T

    def get_recompute_tokens(
        self, 
        chunk_len: int, 
        cci: float, 
        beta_prime: float, 
        inter_scores_tensor_list: List[torch.Tensor],
        fixed_ratio: float = None
    ) -> List[int]:
        """
        决策函数：计算 CFO (或使用固定比例) 并筛选需要重算的 Token 索引。
        
        Args:
            chunk_len (int): 当前块的长度。
            cci (float): 缓存上下文影响分数。
            beta_prime (float): 前缀重叠分数。
            inter_scores_tensor_list (List[torch.Tensor]): 每一层的注意力敏感度张量列表，每个形状为 [Chunk_Len]。
            fixed_ratio (float, optional): 强制覆盖的重算比例 (0.0-1.0)。若设置，忽略 CFO 逻辑。
            
        Returns:
            List[int]: 需要重算的 Token 索引列表 (升序排列)。
        """
        if fixed_ratio is not None:
            # 强制指定模式
            num_recompute = math.ceil(fixed_ratio * chunk_len)
        else:
            # 1. 计算 Cache Fix Overhead (CFO)
            # CFO = alpha * CCI * (1 - beta')
            # 如果 beta' = 1 (前缀完全一致), CFO = 0, 不需要重算。
            # 如果 cci = 0 (不依赖上下文), CFO = 0, 不需要重算。
            cfo = self.alpha * cci * (1.0 - beta_prime)
            
            # 2. 计算需要重算的 token 数量
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
        stacked_scores = torch.stack(inter_scores_tensor_list).float()
        
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
