"""
OmniKV 项目配置
自动检测不同主机的路径配置
"""

import os
from pathlib import Path


class OmniKVConfig:
    """项目配置管理"""
    
    def __init__(self):
        # 自动检测基础路径
        self.base_path = self._detect_base_path()
        self.host_type = self._detect_host_type()
        
        # 设置路径
        self.project_root = self.base_path / "projects" / "OmniKV"
        self.models_dir = self.base_path / "models"
        self.datasets_dir = self.project_root / "datasets"
        self.output_dir = self.project_root / "baselines" / "cachecraft" / "output"
    
    def _detect_base_path(self) -> Path:
        """自动检测基础路径"""
        # 优先使用环境变量
        if 'OMNIKV_BASE' in os.environ:
            return Path(os.environ['OMNIKV_BASE'])
        
        # 自动检测
        candidates = [
            Path("/NV1/ykw"),
            Path("/data/ykw"),
            Path.home(),  # 备选
        ]
        
        for path in candidates:
            if path.exists():
                return path
        
        # 默认使用当前项目路径的上级
        return Path(__file__).resolve().parent.parent.parent
    
    def _detect_host_type(self) -> str:
        """检测主机类型"""
        if "/NV1" in str(self.base_path):
            return "original"
        elif "/data" in str(self.base_path):
            return "new"
        else:
            return "unknown"
    
    def get_dataset_path(self, dataset_name: str) -> Path:
        """获取数据集路径"""
        return self.datasets_dir / "longbench" / dataset_name
    
    def get_model_path(self, model_name: str) -> str:
        """获取模型路径（可能是本地或 HuggingFace）"""
        local_path = self.models_dir / model_name
        if local_path.exists():
            return str(local_path)
        else:
            # 返回 HuggingFace 名称（会自动从缓存加载）
            return model_name
    
    def __repr__(self):
        return f"""OmniKVConfig(
  host_type={self.host_type}
  base_path={self.base_path}
  project_root={self.project_root}
  models_dir={self.models_dir}
  datasets_dir={self.datasets_dir}
)"""


# 全局配置实例
config = OmniKVConfig()


if __name__ == '__main__':
    # 测试配置
    print(config)
    print(f"\n测试路径:")
    print(f"  narrativeqa: {config.get_dataset_path('narrativeqa.json')}")
    print(f"  NV-Embed-v2: {config.get_model_path('NV-Embed-v2')}")
