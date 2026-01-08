import os
import time

import transformers
import torch
from typing import List, Optional, Dict, Tuple, Any
from tiny_tools.log import logger


"""
OmniKV 动态缓存管理模块。

该模块提供了一系列增强的 Cache 类，支持在 Decoding 阶段对 KV 缓存进行动态筛选、
CPU 卸载 (Offloading) 以及特定长度的截断等功能。
"""
import os
import time

import transformers
import torch
from typing import List, Optional, Dict, Tuple, Any
from tiny_tools.log import logger


def get_idx_iou_score(new_idx, old_idx):
    """
    计算两个 1D 索引张量之间的交并比 (IoU) 分数。

    Args:
        new_idx (torch.Tensor): 1D 张量，表示新的索引。
        old_idx (torch.Tensor): 1D 张量，表示旧的索引。

    Returns:
        float: IoU 分数。
    """
    # 确保张量是 1D 的
    assert new_idx.dim() == 1 and old_idx.dim() == 1, "两者都必须是 1D 张量"
    # 计算并集的大小（去重）
    un = torch.cat([new_idx, old_idx], dim=0).unique().shape[0]
    # 计算交集长度：A + B - (A U B)
    intersection_len = new_idx.shape[0] + old_idx.shape[0] - un
    # 计算简化版的 IoU (交集占新索引的比例)
    iou_score = intersection_len / new_idx.shape[0]
    return iou_score


class DynamicSubCache(transformers.cache_utils.DynamicCache):
    """
    支持自索引选择的动态缓存类。
    
    允许在 `update` 时根据外部指定的索引（idx）返回 KV 缓存的子集，
    用于在 Attention 计算中只关注选定的 Token。
    """
    def __init__(self):
        super(DynamicSubCache, self).__init__()
        # 存储每一层选中的索引 [layer_idx] -> Tensor
        self.idx = {}

    def set_out_idx(self, idx, layer_idx):
        """设置该层需要输出的 KV 索引序列"""
        self.idx[layer_idx] = idx

    def clear(self, layer_idx):
        """清除指定层之后的所有缓存内容"""
        while len(self.key_cache) > layer_idx + 1:
            del self.key_cache[-1], self.value_cache[-1]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        更新 KV 缓存。如果设置了索引，则返回筛选后的视图。

        Args:
            key_states: 新生成的 Key。
            value_states: 新生成的 Value。
            layer_idx: 当前层索引。
            cache_kwargs: 额外参数。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 用于本次推理的 (Key, Value) 缓存视图。
        """
        # 仅在第 0 层更新全局计数器
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # 将新 KV 拼接到现有缓存中
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=-2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=-2
            )

        # 如果该层配置了筛选索引，则使用 gather 操作提取对应的 KV
        if layer_idx in self.idx:
            # 扩展索引形状以匹配 [bsz, heads, seq_len, head_dim]
            idx = (
                self.idx[layer_idx]
                .unsqueeze(1)
                .unsqueeze(-1)
                .expand(
                    -1,
                    self.key_cache[layer_idx].shape[1],
                    -1,
                    self.key_cache[layer_idx].shape[-1],
                )
            )
            # 根据索引提取并返回
            return self.key_cache[layer_idx].gather(2, idx), self.value_cache[
                layer_idx
            ].gather(2, idx)
        else:
            # 否则返回全量缓存（例如在 Prefilling 阶段）
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

    @staticmethod
    def from_dynamic_cache(cache: transformers.cache_utils.DynamicCache):
        """从标准的 DynamicCache 对象快速转换为 DynamicSubCache"""
        c = DynamicSubCache()
        c.key_cache = cache.key_cache
        c.value_cache = cache.value_cache
        c._seen_tokens = cache._seen_tokens
        return c


class DynamicSubOffloadTrueCache(transformers.cache_utils.DynamicCache):
    """
    带有 CPU 卸载功能的动态子缓存类。
    
    该类将部分层的 KV 缓存存储在 CPU 内存中，并在 Decoding 阶段根据索引只将选中的部分搬运回 GPU。
    """
    def __init__(self, offload_start_id=12, num_total_layers=32):
        super().__init__()
        self.idx = {}
        self.stage = "prefill" # 标志当前是 prefill 还是 decoding 阶段
        self.offload_start_id = offload_start_id # 开始卸载的层索引
        self.num_total_layers = num_total_layers # 模型总层数
        self.part_key = {} # 存储从 CPU 搬运回 GPU 的 KV 缓存段
        self.part_value = {}
        self.tail_k = {} # 存储 Decoding 阶段新生成的 KV
        self.tail_v = {}
        self.mamba_k = None # CPU 上的大缓存（所有卸载层的拼接）
        self.mamba_v = None

    def set_out_idx(self, idx, sid):
        """
        根据选中的过滤索引，将对应的 CPU KV 缓存异步搬运到 GPU。
        """
        # 如果是保存索引模式，则不执行搬运
        if os.environ.get("SAVE_SELECTED_IDX", False):
            assert self.offload_start_id >= self.num_total_layers
            return

        # 转换并处理索引
        if (bs := idx.shape[0]) == 1:
            _idx = idx.cpu().view(-1)
        else:
            # 多 Batch 模式（待充分测试）
            _idx = idx.view(bs, -1, 1).repeat(
                1, 1, self.mamba_k.shape[-2], self.mamba_k.shape[-1]
            )
            
        st = time.time()
        # 更新所有后续层的索引
        for i in range(sid, self.num_total_layers):
            self.idx[i] = idx
            
        # 计算每个层在拼接大缓存中占用的长度
        sz = self.mamba_k.shape[2] // (self.num_total_layers - self.offload_start_id)
        
        # 核心搬运：从 CPU 缓存 (mamba_k) 中 gather 索引内容并送到 CUDA
        if bs == 1:
            k = self.mamba_k[:, _idx].cuda(non_blocking=True).split(sz, dim=2)
            v = self.mamba_v[:, _idx].cuda(non_blocking=True).split(sz, dim=2)
        else:
            # TODO not tested
            k = (
                torch.gather(self.mamba_k, 1, _idx)
                .cuda(non_blocking=True)
                .split(sz, dim=2)
            )
            v = (
                torch.gather(self.mamba_v, 1, _idx)
                .cuda(non_blocking=True)
                .split(sz, dim=2)
            )
            
        # 将结果分配回各个层
        for i in range(self.offload_start_id, self.num_total_layers):
            self.part_key[i] = k[i - self.offload_start_id].transpose(1, 2)
            self.part_value[i] = v[i - self.offload_start_id].transpose(1, 2)

        logger.info(f"index&to cuda used {time.time() - st}")

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        更新缓存。如果属于卸载层且在 prefill 阶段，则将 KV 送往 CPU。
        如果在 decoding 阶段，则返回 GPU 上的部分缓存与新生成的 KV 的拼接。
        """
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
            
        # 处理卸载层逻辑
        if layer_idx >= self.offload_start_id:
            if self.stage == "prefill":
                # Prefill 阶段：异步将 KV 存储到 CPU
                st = time.time()
                _key = (
                    key_states.transpose(1, 2)
                    .contiguous()
                    .to(
                        device="cpu",
                        non_blocking=(layer_idx != self.num_total_layers - 1),
                    )
                )
                _value = (
                    value_states.transpose(1, 2)
                    .contiguous()
                    .to(
                        device="cpu",
                        non_blocking=(layer_idx != self.num_total_layers - 1),
                    )
                )
                self.key_cache.append(_key)
                self.value_cache.append(_value)
                logger.info(f"L{layer_idx} offload to cpu {time.time() - st}")
                
                # 最后一层处理完后，将 CPU 上的 KV 拼接成大张量
                if layer_idx == self.num_total_layers - 1:
                    self.mamba_k = torch.cat(
                        self.key_cache[self.offload_start_id :], dim=2
                    ).contiguous()
                    self.mamba_v = torch.cat(
                        self.value_cache[self.offload_start_id :], dim=2
                    ).contiguous()
                return key_states, value_states
            else:
                # Decoding 阶段：更新本阶段生成的 Token (tail)
                if layer_idx not in self.tail_k:
                    self.tail_k[layer_idx] = key_states
                    self.tail_v[layer_idx] = value_states
                else:
                    self.tail_k[layer_idx] = torch.cat(
                        [self.tail_k[layer_idx], key_states], dim=-2
                    )
                    self.tail_v[layer_idx] = torch.cat(
                        [self.tail_v[layer_idx], value_states], dim=-2
                    )

                # 返回 历史选中部分 (part) + 最新生成部分 (tail)
                return (
                    torch.cat(
                        [self.part_key[layer_idx], self.tail_k[layer_idx]], dim=-2
                    ),
                    torch.cat(
                        [self.part_value[layer_idx], self.tail_v[layer_idx]], dim=-2
                    ),
                )

        else:
            # 非卸载层的标准处理逻辑
            if len(self.key_cache) <= layer_idx:
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
            else:
                self.key_cache[layer_idx] = torch.cat(
                    [self.key_cache[layer_idx], key_states], dim=-2
                )
                self.value_cache[layer_idx] = torch.cat(
                    [self.value_cache[layer_idx], value_states], dim=-2
                )

            if layer_idx in self.idx:
                idx = (
                    self.idx[layer_idx]
                    .unsqueeze(1)
                    .unsqueeze(-1)
                    .expand(
                        -1,
                        self.key_cache[layer_idx].shape[1],
                        -1,
                        self.key_cache[layer_idx].shape[-1],
                    )
                )
                return self.key_cache[layer_idx].gather(2, idx), self.value_cache[
                    layer_idx
                ].gather(2, idx)
            else:
                return self.key_cache[layer_idx], self.value_cache[layer_idx]

    @staticmethod
    def from_dynamic_cache(
        cache: transformers.cache_utils.DynamicCache,
        offload_start_id=12,
        num_total_layers=32,
    ):
        c = DynamicSubOffloadTrueCache(offload_start_id, num_total_layers)
        c.key_cache = cache.key_cache
        c.value_cache = cache.value_cache
        c._seen_tokens = cache._seen_tokens
        return c


class OmniKVMultiStageCache(transformers.cache_utils.DynamicCache):
    """
    OmniKV 多阶段动态缓存。
    
    支持将模型层划分为多个阶段，每个阶段可以有不同的 KV 留存策略和 CPU 卸载策略。
    主要用于处理多层过滤的情境。
    """
    def __init__(
        self,
        full_attn_layers: List,
        num_hidden_layers,
        num_wait_load_layers=2,
        real_offload=True,
    ):
        super().__init__()
        self.stage = "prefill"
        self.full_attn_layers = full_attn_layers # 触发全量注意力的层索引列表
        self.num_hidden_layers = num_hidden_layers
        self.num_wait_layers = num_wait_load_layers # 触发选择后等待加载的层数
        self.layer_state = {} 
        self.need_cat_layers = []
        self.device = "cpu" if real_offload else None
        
        # 构建层状态机：确定哪些层需要全量缓存，哪些层需要卸载，哪些层是子集缓存
        for _l in self.full_attn_layers:
            _r = num_hidden_layers
            for i in range(_l + 1, self.num_hidden_layers):
                if i in self.full_attn_layers:
                    _r = i
                    break
            # _l 到 _l + num_wait_layers 之间的层保持在 GPU 上（等待索引计算）
            for i in range(_l, min(_r, _l + self.num_wait_layers + 1)):
                self.layer_state[i] = (False, _l, _r) # False 表示不需要 offload
            # 后续的层可以卸载到 CPU，直到下一个 full_attn_layer
            for i in range(min(_r, _l + self.num_wait_layers + 1), _r):
                self.layer_state[i] = (True, _l, _r)  # True 表示需要 offload

        self.part_key = {}
        self.part_value = {}
        self.tail_k = {}
        self.tail_v = {}
        self.mamba_k = {}
        self.mamba_v = {}

    def set_idx_on_gpu(self, idx, sel_layer_idx):
        """
        在 GPU 上通过索引选取指定范围层的 KV 缓存。
        """
        st = time.time()
        _idx = idx.view(-1)
        if self.device == "cpu":
            _idx = _idx.cpu()
            
        # 找到受该索引影响的层范围
        _r = self.layer_state[sel_layer_idx][2]
        sid = sel_layer_idx + self.num_wait_layers + 1
        
        # 批量执行索引选择并送回 GPU
        for i in range(sid, _r):
            self.part_key[i] = torch.index_select(self.key_cache[i], 2, _idx).cuda(
                non_blocking=True
            )
            self.part_value[i] = torch.index_select(self.value_cache[i], 2, _idx).cuda(
                non_blocking=True
            )
        logger.info(
            f"index&to cuda for layer={sel_layer_idx} used={round(time.time() - st, 3)}s"
        )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        根据层状态更新缓存。
        """
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        if self.layer_state[layer_idx][0]:  # 如果该层被标记为需要 offload
            if self.stage == "prefill":
                # Prefill 阶段：卸载到指定设备
                st = time.time()
                is_last_layer = layer_idx == self.num_hidden_layers - 1
                _key = key_states.to(
                    device=self.device if self.device else key_states.device,
                    non_blocking=not is_last_layer,
                )
                _value = value_states.to(
                    device=self.device if self.device else value_states.device,
                    non_blocking=not is_last_layer,
                )

                self.key_cache.append(_key)
                self.value_cache.append(_value)
                logger.info(
                    f"Layer={layer_idx} offload to cpu {round(time.time() - st, 3)}s"
                )
                return key_states, value_states
            else:
                # Decoding 阶段：拼接选中的历史 KV 和新生成的 Tail KV
                if layer_idx not in self.tail_k:
                    self.tail_k[layer_idx] = key_states
                    self.tail_v[layer_idx] = value_states
                else:
                    self.tail_k[layer_idx] = torch.cat(
                        [self.tail_k[layer_idx], key_states], dim=-2
                    )
                    self.tail_v[layer_idx] = torch.cat(
                        [self.tail_v[layer_idx], value_states], dim=-2
                    )

                return (
                    torch.cat(
                        [self.part_key[layer_idx], self.tail_k[layer_idx]], dim=-2
                    ),
                    torch.cat(
                        [self.part_value[layer_idx], self.tail_v[layer_idx]], dim=-2
                    ),
                )

        else:
            # 标准层（非卸载层）：正常维护 KV 缓存
            if len(self.key_cache) <= layer_idx:
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
            else:
                self.key_cache[layer_idx] = torch.cat(
                    [self.key_cache[layer_idx], key_states], dim=-2
                )
                self.value_cache[layer_idx] = torch.cat(
                    [self.value_cache[layer_idx], value_states], dim=-2
                )
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

    @staticmethod
    def from_dynamic_cache(
        cache: transformers.cache_utils.DynamicCache,
        full_attn_layers: List,
        num_hidden_layers,
        num_wait_load_layers=2,
        real_offload=True,
    ):
        c = OmniKVMultiStageCache(
            full_attn_layers, num_hidden_layers, num_wait_load_layers, real_offload
        )
        c.key_cache = cache.key_cache
        c.value_cache = cache.value_cache
        c._seen_tokens = cache._seen_tokens
        return c


class WOPackCache(transformers.cache_utils.DynamicCache):
    # 默认DenseMore=True
    def __init__(
        self,
        full_attn_layers: List,
        num_hidden_layers,
        num_wait_load_layers=2,
        real_offload=True,
    ):
        super().__init__()
        # self.selected_indices = {}
        self.stage = "prefill"
        self.full_attn_layers = full_attn_layers
        self.num_hidden_layers = num_hidden_layers
        self.num_wait_layers = num_wait_load_layers
        self.layer_state = {}
        self.need_cat_layers = []
        self.device = "cpu" if real_offload else None
        # 设定层的状态
        for _l in self.full_attn_layers:
            _r = num_hidden_layers
            for i in range(_l + 1, self.num_hidden_layers):
                if i in self.full_attn_layers:
                    _r = i
                    break
            for i in range(_l, min(_r, _l + self.num_wait_layers + 1)):
                self.layer_state[i] = (False, _l, _r)
            for i in range(min(_r, _l + self.num_wait_layers + 1), _r):
                self.layer_state[i] = (True, _l, _r)  # 需要offload

        self.part_key = {}
        self.part_value = {}
        self.tail_k = {}
        self.tail_v = {}
        self.mamba_k = {}
        self.mamba_v = {}
        self.idx = None

    def set_idx_on_gpu(self, idx, sel_layer_idx):
        st = time.time()
        _idx = idx.cpu().view(-1)
        self.idx = _idx
        # _r = self.layer_state[sel_layer_idx][2]
        # sid = sel_layer_idx + self.num_wait_layers + 1
        # for i in range(sid, _r):
        #     self.part_key[i] = torch.index_select(self.key_cache[i], 2, _idx).cuda(non_blocking=True)
        #     self.part_value[i] = torch.index_select(self.value_cache[i], 2, _idx).cuda(non_blocking=True)
        # logger.info(f"index&to cuda for layer={sel_layer_idx} used={round(time.time() - st, 3)}s")

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        if self.layer_state[layer_idx][0]:  # 需要offload
            if self.stage == "prefill":
                st = time.time()
                is_last_layer = layer_idx == self.num_hidden_layers - 1
                _key = key_states.to(
                    device=self.device if self.device else key_states.device,
                    non_blocking=not is_last_layer,
                )
                _value = value_states.to(
                    device=self.device if self.device else value_states.device,
                    non_blocking=not is_last_layer,
                )

                self.key_cache.append(_key)
                self.value_cache.append(_value)
                logger.info(
                    f"Layer={layer_idx} offload to cpu {round(time.time() - st, 3)}s"
                )
                return key_states, value_states
            else:
                if layer_idx not in self.tail_k:
                    self.tail_k[layer_idx] = key_states
                    self.tail_v[layer_idx] = value_states
                else:
                    self.tail_k[layer_idx] = torch.cat(
                        [self.tail_k[layer_idx], key_states], dim=-2
                    )
                    self.tail_v[layer_idx] = torch.cat(
                        [self.tail_v[layer_idx], value_states], dim=-2
                    )
                if torch.max(key_states) > -1e4:  # to ensure sync
                    self.part_key[layer_idx] = torch.index_select(
                        self.key_cache[layer_idx], 2, self.idx
                    ).cuda()
                    self.part_value[layer_idx] = torch.index_select(
                        self.value_cache[layer_idx], 2, self.idx
                    ).cuda()
                return (
                    torch.cat(
                        [self.part_key[layer_idx], self.tail_k[layer_idx]], dim=-2
                    ),
                    torch.cat(
                        [self.part_value[layer_idx], self.tail_v[layer_idx]], dim=-2
                    ),
                )

        else:
            if len(self.key_cache) <= layer_idx:
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
            else:
                self.key_cache[layer_idx] = torch.cat(
                    [self.key_cache[layer_idx], key_states], dim=-2
                )
                self.value_cache[layer_idx] = torch.cat(
                    [self.value_cache[layer_idx], value_states], dim=-2
                )
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

    @staticmethod
    def from_dynamic_cache(
        cache: transformers.cache_utils.DynamicCache,
        full_attn_layers: List,
        num_hidden_layers,
        num_wait_load_layers=2,
        real_offload=True,
    ):
        c = WOPackCache(
            full_attn_layers, num_hidden_layers, num_wait_load_layers, real_offload
        )
        c.key_cache = cache.key_cache
        c.value_cache = cache.value_cache
        c._seen_tokens = cache._seen_tokens
        return c


class OmniKVLazyCache(DynamicSubOffloadTrueCache):

    def __init__(self, offload_start_id=12, num_total_layers=32, skip_threshold=0.4):
        super().__init__()
        self.idx = {}
        self.stage = "prefill"
        self.offload_start_id = offload_start_id
        self.num_total_layers = num_total_layers
        self.part_key = {}
        self.part_value = {}
        self.tail_k = {}
        self.tail_v = {}
        self.mamba_k = None
        self.mamba_v = None
        self.skip_threshold = skip_threshold

    def set_out_idx(self, idx, sid):
        assert idx.shape[0] == 1
        _idx = idx.cpu().view(-1)
        if sid in self.idx:
            sc = get_idx_iou_score(idx.view(-1), self.idx[sid].view(-1))
            logger.info(f"current score = {sc}")
            if sc > self.skip_threshold:
                return
        # 接下来进行更新
        st = time.time()
        for i in range(sid, self.num_total_layers):
            self.idx[i] = idx
        sz = self.mamba_k.shape[2] // (self.num_total_layers - self.offload_start_id)
        k = self.mamba_k[:, _idx].cuda(non_blocking=True).split(sz, dim=2)
        v = self.mamba_v[:, _idx].cuda(non_blocking=True).split(sz, dim=2)
        for i in range(self.offload_start_id, self.num_total_layers):
            self.part_key[i] = k[i - self.offload_start_id].transpose(1, 2)
            self.part_value[i] = v[i - self.offload_start_id].transpose(1, 2)
        logger.info(f"index&to cuda used {time.time() - st}")

    @staticmethod
    def from_dynamic_cache(
        cache: transformers.cache_utils.DynamicCache,
        offload_start_id=12,
        num_total_layers=32,
        skip_threshold=0.4,
    ):
        c = OmniKVLazyCache(
            offload_start_id, num_total_layers, skip_threshold=skip_threshold
        )
        c.key_cache = cache.key_cache
        c.value_cache = cache.value_cache
        c._seen_tokens = cache._seen_tokens
        return c


class OmniKVMoreEffCache(transformers.cache_utils.DynamicCache):
    def __init__(self, config):
        super().__init__()
        if cpu_cache is None:
            make_cpu_cache(config)
        self.idx = {}
        self.stage = "prefill"
        self.offload_start_id = config.get("offload_sid", 12)
        self.num_total_layers = config.get("num_hidden_layers", 32)
        self.part_key = {}
        self.part_value = {}
        self.tail_k = {}
        self.tail_v = {}

    def set_out_idx(self, idx, sid):
        st = time.time()
        for i in range(sid, self.num_total_layers):
            self.idx[i] = idx
        k, v = cpu_cache.get_cache(idx)
        for i in range(self.offload_start_id, self.num_total_layers):
            self.part_key[i] = k[i - self.offload_start_id].transpose(1, 2)
            self.part_value[i] = v[i - self.offload_start_id].transpose(1, 2)
        logger.debug(f"index&to cuda used {time.time() - st}")

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        if layer_idx >= self.offload_start_id:
            if self.stage == "prefill":
                st = time.time()
                _key, _value = (
                    key_states.transpose(1, 2).contiguous(),
                    value_states.transpose(1, 2).contiguous(),
                )
                cpu_cache.set_prefilled_cache(_key, _value, layer_idx)
                logger.info(f"L{layer_idx} offload to cpu {time.time() - st}")
                return key_states, value_states
            else:
                if layer_idx not in self.tail_k:
                    self.tail_k[layer_idx] = key_states
                    self.tail_v[layer_idx] = value_states
                else:
                    self.tail_k[layer_idx] = torch.cat(
                        [self.tail_k[layer_idx], key_states], dim=-2
                    )
                    self.tail_v[layer_idx] = torch.cat(
                        [self.tail_v[layer_idx], value_states], dim=-2
                    )

                return (
                    torch.cat(
                        [self.part_key[layer_idx], self.tail_k[layer_idx]], dim=-2
                    ),
                    torch.cat(
                        [self.part_value[layer_idx], self.tail_v[layer_idx]], dim=-2
                    ),
                )

        else:
            if len(self.key_cache) <= layer_idx:
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
            else:
                self.key_cache[layer_idx] = torch.cat(
                    [self.key_cache[layer_idx], key_states], dim=-2
                )
                self.value_cache[layer_idx] = torch.cat(
                    [self.value_cache[layer_idx], value_states], dim=-2
                )

            if layer_idx in self.idx:
                idx = (
                    self.idx[layer_idx]
                    .unsqueeze(1)
                    .unsqueeze(-1)
                    .expand(
                        -1,
                        self.key_cache[layer_idx].shape[1],
                        -1,
                        self.key_cache[layer_idx].shape[-1],
                    )
                )
                return self.key_cache[layer_idx].gather(2, idx), self.value_cache[
                    layer_idx
                ].gather(2, idx)
            else:
                return self.key_cache[layer_idx], self.value_cache[layer_idx]

    @staticmethod
    def from_dynamic_cache(cache: transformers.cache_utils.DynamicCache, config):
        c = OmniKVMoreEffCache(config)
        c.key_cache = cache.key_cache
        c.value_cache = cache.value_cache
        c._seen_tokens = cache._seen_tokens
        return c


class DynamicBrutalOffloadCache(transformers.cache_utils.DynamicCache):

    def __init__(self, offload_start_id=12, num_total_layers=32):
        super().__init__()
        self.idx = {}
        self.stage = "prefill"
        self.offload_start_id = offload_start_id
        self.num_total_layers = num_total_layers
        self.streams = {
            i: torch.cuda.Stream()
            for i in range(self.offload_start_id, self.num_total_layers)
        }
        self.part_key = {}
        self.part_value = {}
        self.tail_k = {}
        self.tail_v = {}
        self.offset = 4

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        # logger.debug(f"in layer {layer_idx}")
        if (
            self.stage == "decoding"
            and self.num_total_layers > layer_idx + self.offset >= self.offload_start_id
        ):
            self.part_key[layer_idx + self.offset] = self.key_cache[
                layer_idx + self.offset
            ].cuda(non_blocking=True)
            self.part_value[layer_idx + self.offset] = self.value_cache[
                layer_idx + self.offset
            ].cuda(non_blocking=True)
        if layer_idx >= self.offload_start_id:
            assert key_states.shape[0] == 1, "now only bsz==1"
            if self.stage == "prefill":
                st = time.time()
                _key = key_states.to(device="cpu", non_blocking=True)
                _value = value_states.to(device="cpu", non_blocking=True)
                # _key = key_states
                # _value = value_states
                self.key_cache.append(_key)
                self.value_cache.append(_value)
                logger.info(f"L{layer_idx} offload to cpu {time.time() - st}")
                return key_states, value_states
            else:
                if layer_idx not in self.tail_k:
                    self.tail_k[layer_idx] = key_states
                    self.tail_v[layer_idx] = value_states
                else:
                    self.tail_k[layer_idx] = torch.cat(
                        [self.tail_k[layer_idx], key_states], dim=-2
                    )
                    self.tail_v[layer_idx] = torch.cat(
                        [self.tail_v[layer_idx], value_states], dim=-2
                    )

                assert key_states.shape[-2] == 1
                temp = (
                    torch.cat(
                        [self.part_key[layer_idx], self.tail_k[layer_idx]], dim=-2
                    ),
                    torch.cat(
                        [self.part_value[layer_idx], self.tail_v[layer_idx]], dim=-2
                    ),
                )
                self.part_key[layer_idx] = None
                self.part_value[layer_idx] = None
                return temp

        else:
            if len(self.key_cache) <= layer_idx:
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
            else:
                self.key_cache[layer_idx] = torch.cat(
                    [self.key_cache[layer_idx], key_states], dim=-2
                )
                self.value_cache[layer_idx] = torch.cat(
                    [self.value_cache[layer_idx], value_states], dim=-2
                )

            return self.key_cache[layer_idx], self.value_cache[layer_idx]

    @staticmethod
    def from_dynamic_cache(
        cache: transformers.cache_utils.DynamicCache,
        offload_start_id=12,
        num_total_layers=32,
    ):
        c = DynamicBrutalOffloadCache(offload_start_id, num_total_layers)
        c.key_cache = cache.key_cache
        c.value_cache = cache.value_cache
        c._seen_tokens = cache._seen_tokens
        return c


class SinkCache(transformers.cache_utils.DynamicCache):
    def __init__(self, window_length, num_sink_tokens):
        super().__init__()
        self.window_len = window_length
        self.num_sink_tokens = num_sink_tokens

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            assert key_states.shape[2] > 1
            if key_states.shape[-2] > self.window_len + self.num_sink_tokens:
                key_states = torch.cat(
                    [
                        key_states[:, :, : self.num_sink_tokens],
                        key_states[:, :, -self.window_len :],
                    ],
                    dim=-2,
                )  # noqa
                value_states = torch.cat(
                    [
                        value_states[:, :, : self.num_sink_tokens],
                        value_states[:, :, -self.window_len :],
                    ],
                    dim=-2,
                )  # noqa
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            assert key_states.shape[2] == 1
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=-2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=-2
            )

        return self.key_cache[layer_idx], self.value_cache[layer_idx]


def get_cache_cls(config):
    name2cls = {
        "lazy": OmniKVLazyCache,
        "default": DynamicSubOffloadTrueCache,
        "eff": OmniKVMoreEffCache,
        "multi": OmniKVMultiStageCache,
        "without_pack": WOPackCache,
    }
    return name2cls[config.get("cache_cls", "default")]
