"""
OmniKV 压缩器配置模块。

该模块扩展了原生的 LlamaConfig，用于存储和管理 OmniKV 压缩算法特有的超参数，
例如树高度、最小长度和段大小等。
"""
import torch
from transformers.models.llama.modeling_llama import *
from tiny_tools.log import logger

# 用于文档引用的配置名称
_CONFIG_FOR_DOC = "LlamaConfig"

logger.debug("start")


class OmniKVCompressorConfig(LlamaConfig):
    """
    OmniKV 压缩策略的专用配置类。

    该类通过继承 LlamaConfig，允许在标准的 Llama 模型配置中注入
    动态上下文选择和 KV 压缩相关的参数。
    """
    def set_config_of_compressor(self, **kwargs):
        """
        批量设置压缩器相关的配置参数。

        Args:
            **kwargs: 包含配置名称和值的键值对。
                     必须包含 "tree_height", "minL", "segment_size"。

        Raises:
            AssertionError: 如果指定的必填键缺失。
        """
        # 验证核心参数是否存在
        for key in ["tree_height", "minL", "segment_size"]:
            assert key in kwargs, f"缺少压缩器必填参数: {key}"
            
        # 动态设置属性值
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get(self, key, default=None):
        """
        安全获取配置项的值，如果不存在则赋予默认值并记录警告。

        Args:
            key (str): 配置项名称。
            default (Any, optional): 默认值。默认为 None。

        Returns:
            Any: 配置项的值。
        """
        if not hasattr(self, key):
            setattr(self, key, default)
            logger.warning(f"配置项 {key} 不存在，已将其初始化为默认值: {default}")
        return getattr(self, key, default)
