"""
数据集配置管理模块

提供统一的数据集配置接口，包括：
- 字段映射（自动将数据集特定字段映射到统一字段）
- Context 格式转换（支持多种格式）
- 提示词模板管理
"""

import json
import os
from typing import Dict, List, Any, Optional, Tuple


class DatasetConfig:
    """数据集配置管理类"""
    
    def __init__(self, config_path: str = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径，默认使用当前目录下的 config/dataset2prompt.json
        """
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), 
                "config/dataset2prompt.json"
            )
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
    
    def get_dataset_config(self, dataset_name: str) -> Dict:
        """
        获取指定数据集的配置
        
        Args:
            dataset_name: 数据集名称
            
        Returns:
            数据集配置字典
            
        Raises:
            ValueError: 如果数据集配置不存在
        """
        if dataset_name not in self.config:
            raise ValueError(
                f"Dataset '{dataset_name}' not found in config. "
                f"Available datasets: {list(self.config.keys())}"
            )
        return self.config[dataset_name]
    
    def extract_fields(self, sample: Dict, dataset_name: str) -> Dict[str, Any]:
        """
        从样本中提取字段，自动进行字段映射
        
        Args:
            sample: 数据样本
            dataset_name: 数据集名称
            
        Returns:
            包含统一字段的字典，包括:
                - question: 问题文本
                - context: 上下文（原始格式）
                - answer: 答案
        """
        dataset_cfg = self.get_dataset_config(dataset_name)
        field_mapping = dataset_cfg["field_mapping"]
        
        result = {}
        
        # 映射问题字段
        question_field = field_mapping.get("question")
        if question_field and question_field in sample:
            result["question"] = sample[question_field]
        
        # 映射上下文字段
        context_field = field_mapping.get("context")
        if context_field and context_field in sample:
            result["context"] = sample[context_field]
        
        # 映射答案字段
        answer_field = field_mapping.get("answer")
        if answer_field and answer_field in sample:
            answer_value = sample[answer_field]
            # 如果是列表，取第一个元素
            if isinstance(answer_value, list) and len(answer_value) > 0:
                result["answer"] = answer_value[0]
            else:
                result["answer"] = answer_value
        
        return result
    
    def format_context(self, context: Any, dataset_name: str) -> List[str]:
        """
        格式化 context 为统一的文本块列表
        
        Args:
            context: 原始 context 数据
            dataset_name: 数据集名称
            
        Returns:
            格式化后的文本块列表
        """
        dataset_cfg = self.get_dataset_config(dataset_name)
        format_type = dataset_cfg.get("context_format", "plain_list")
        
        if format_type == "title_paragraphs":
            # HotpotQA 格式: [[标题, [句子1, 句子2]], ...]
            chunks = []
            for item in context:
                if isinstance(item, list) and len(item) >= 2:
                    title = item[0]
                    sentences = item[1]
                    if isinstance(sentences, list):
                        text = f"Title: {title}\n" + "".join(sentences) + "\n\n"
                    else:
                        text = f"Title: {title}\n{sentences}\n\n"
                    chunks.append(text)
            return chunks
        
        elif format_type == "plain_list":
            # LongBench 格式: [文本1, 文本2, ...]
            # 确保每个 chunk 以 \n\n 结尾，保证 token 分隔一致性
            chunks = []
            for chunk in context:
                if isinstance(chunk, str):
                    # 去除开头的空格/换行符，规范化结尾为统一的 \n\n
                    formatted_chunk = chunk.lstrip().rstrip('\n') + "\n\n"
                    chunks.append(formatted_chunk)
                else:
                    chunks.append(str(chunk) + "\n\n")
            return chunks
        
        else:
            raise ValueError(f"Unsupported context format: {format_type}")
    
    def get_template(
        self, 
        dataset_name: str, 
        template_name: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        获取提示词模板
        
        Args:
            dataset_name: 数据集名称
            template_name: 模板名称，如果为 None 则使用默认模板
            
        Returns:
            (system_prompt, user_template) 元组
        """
        dataset_cfg = self.get_dataset_config(dataset_name)
        
        if template_name is None:
            template_name = dataset_cfg.get("default_template", "answer_only")
        
        templates = dataset_cfg.get("templates", {})
        if template_name not in templates:
            raise ValueError(
                f"Template '{template_name}' not found for dataset '{dataset_name}'. "
                f"Available templates: {list(templates.keys())}"
            )
        
        template = templates[template_name]
        system_prompt = template.get("system", "")
        user_template = template.get("user", "")
        
        return system_prompt, user_template
    
    def list_available_datasets(self) -> List[str]:
        """获取所有可用的数据集名称"""
        return list(self.config.keys())
    
    def list_available_templates(self, dataset_name: str) -> List[str]:
        """获取指定数据集的所有可用模板"""
        dataset_cfg = self.get_dataset_config(dataset_name)
        return list(dataset_cfg.get("templates", {}).keys())


def auto_detect_dataset(data_path: str) -> Optional[str]:
    """
    根据数据路径自动检测数据集类型
    
    Args:
        data_path: 数据文件路径
        
    Returns:
        数据集名称，如果无法检测则返回 None
    """
    path_lower = data_path.lower()
    
    if "longbench" in path_lower:
        return "longbench"
    elif "hotpot" in path_lower:
        return "hotpotqa"
    elif "2wiki" in path_lower:
        return "2wikimqa"
    
    return None
