import torch
from typing import List, Tuple, Dict, Any, Optional

class MetadataStore:
    """
    Cache-Craft 的元数据存储类。
    用于存储每个 RAG Block 的 KV Cache、统计分数以及原始文本。
    """
    def __init__(self):
        # 核心存储字典
        # Key: chunk_hash (str)
        # Value: 包含 k_cache, v_cache, layer_scores, inter_scores_tensor, text 的字典
        self.store: Dict[str, Dict[str, Any]] = {}

    def has_chunk(self, chunk_hash: str) -> bool:
        """
        检查是否已存储具有特定 Hash 的 Chunk。
        
        Args:
            chunk_hash (str): Chunk 的唯一 Hash 标识。
            
        Returns:
            bool: 如果存在返回 True，否则 False。
        """
        return chunk_hash in self.store

    def get_chunk(self, chunk_hash: str) -> Optional[Dict[str, Any]]:
        """
        获取 Chunk 的元数据。
        
        Args:
            chunk_hash (str): Chunk 的唯一 Hash 标识。
            
        Returns:
            Optional[Dict[str, Any]]: 包含 Chunk 数据的字典，如果不存在则返回 None。
        """
        return self.store.get(chunk_hash)

    def save_chunk(
        self,
        chunk_hash: str,
        k_cache: List[torch.Tensor],
        v_cache: List[torch.Tensor],
        layer_scores: List[Tuple[float, float]],
        inter_scores_tensor: List[torch.Tensor],
        text: str,
        prefix_hashes: Optional[List[str]] = None,
        prefix_inter_scores: Optional[List[List[float]]] = None,
        cci: Optional[float] = None
    ):
        """
        保存 Chunk 数据到存储中。
        重要：为了节省显存，所有传入的 Tensor 都会被强制移动到 CPU 存储。
        
        Args:
            chunk_hash (str): Chunk 的唯一 Hash 标识。
            k_cache (List[torch.Tensor]): 每一层的 Key Cache (Pre-RoPE)。
            v_cache (List[torch.Tensor]): 每一层的 Value Cache。
            layer_scores (List[Tuple[float, float]]): 每一层计算出的 (a_l, b_l) (块间，块内)分数，用于计算 CCI。
            inter_scores_tensor (List[torch.Tensor]): 每一层的 Token 级前缀注意力总和，形状 [Seq_Len]，用于后续筛选 Top-K 重算。
            text (str): 原始文本块内容。
            prefix_hashes (Optional[List[str]]): 生成该 Chunk 时的前缀 Hash 列表。
            prefix_inter_scores (Optional[List[List[float]]]): 每层对各前缀块的 inter 分数列表。
            cci (Optional[float]): 预先计算好的 CCI 分数。
        """
        
        # 1. 处理 KV Cache：移动到 CPU 并分离计算图
        # k_cache 和 v_cache 是 List[Tensor]
        cpu_k_cache = [k.detach().cpu() for k in k_cache]
        cpu_v_cache = [v.detach().cpu() for v in v_cache]
        
        # 2. 处理 inter_scores_tensor：移动到 CPU 并分离计算图
        cpu_inter_scores = [t.detach().cpu() for t in inter_scores_tensor]

        # 3. 存入字典
        self.store[chunk_hash] = {
            "k_cache": cpu_k_cache,
            "v_cache": cpu_v_cache,
            "layer_scores": layer_scores,
            "inter_scores_tensor": cpu_inter_scores,
            "text": text,
            "prefix_hashes": prefix_hashes or [],
            "prefix_inter_scores": prefix_inter_scores or [],
            "cci": cci
        }

    def clear(self):
        """清空存储。"""
        self.store.clear()
