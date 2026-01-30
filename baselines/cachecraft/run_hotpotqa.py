# CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python baselines/cachecraft/run_hotpotqa.py --output_file baselines/cachecraft/output/hotpot_dense_remix_v1/craft_results.txt

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.omnikv_config import config

import json
import argparse
import os
import torch
from baselines.cachecraft.pipeline import CacheCraftPipeline
from baselines.cachecraft.dataset_config import DatasetConfig

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="在 HotpotQA 数据集上运行 Cache-Craft 基准测试")
    parser.add_argument("--data_path", type=str, default=str(config.get_dataset_path("2WikiMultihopQA_format/2WiKiMQA_dense_remix_v1.json")), help="数据集路径 (JSON格式)")
    parser.add_argument("--model_path", type=str, default=config.get_model_path("Meta-Llama-3.1-8B-Instruct"), help="HuggingFace 模型路径或名称")
    parser.add_argument("--num_samples", type=int, default=None, help="运行测试的样本数量 (用于快速调试)")
    parser.add_argument("--alpha", type=float, default=1.0, help="CFO (Context-Aware Fractional Offloading) 算法的 alpha 参数")
    parser.add_argument("--device", type=str, default="cuda", help="使用的计算设备 (如 cuda, cpu)")
    parser.add_argument("--disable_caching", action="store_true", help="禁用 KV 缓存功能 (MetadataStore)")
    parser.add_argument("--disable_recompute", action="store_true", help="禁用命中块的 token 重算，仅直接复用 KV")
    parser.add_argument("--output_file", type=str, default=None, help="输出文件路径")
    return parser.parse_args()

def load_data(path, num_samples):
    """
    加载数据集，并截取指定数量的样本用于测试
    """
    print(f"Loading data from {path}...")
    with open(path, 'r') as f:
        data = json.load(f)
    print(f"Total samples: {len(data)}")
    return data[:num_samples]

def main():
    args = parse_args()
    
    # 自动生成输出路径逻辑
    if args.output_file is None:
        base_output_dir = str(config.output_dir) + "/"
        # 1. 提取数据集名称 (datasets/ 之后的部分路径)
        if "datasets/" in args.data_path:
            relative_path = args.data_path.split("datasets/")[-1]
            dataset_rel_name = os.path.splitext(relative_path)[0]
        else:
            dataset_rel_name = os.path.splitext(os.path.basename(args.data_path))[0]
            
        # 2. 根据参数决定文件名
        if args.disable_caching:
            suffix = "nocache"
        elif args.disable_recompute:
            suffix = "norecompute"
        else:
            suffix = "recompute"
        filename = f"craft_{suffix}_results.txt"
            
        # 3. 拼接最终路径并创建目录
        args.output_file = os.path.join(base_output_dir, dataset_rel_name, filename)
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        print(f"Auto-generated output_file: {args.output_file}")

    # 检查模型路径是否存在（略过详细检查，假设用户提供路径正确）
    
    # 初始化配置系统
    dataset_cfg = DatasetConfig()
    dataset_name = "hotpotqa"
    
    # 获取提示词模板
    system_prompt, user_template = dataset_cfg.get_template(dataset_name)
    print(f"Loaded templates for {dataset_name}")

    # 初始化 CacheCraft Pipeline
    # 这将加载模型并应用 Monkey Patch
    pipeline = CacheCraftPipeline(
        model_name_or_path=args.model_path,
        alpha=args.alpha,
        device=args.device,
        enable_caching=not args.disable_caching,
        enable_recompute=not args.disable_recompute
    )
    
    # 加载测试数据
    samples = load_data(args.data_path, args.num_samples)
    
    # 逐个样本处理
    for i, sample in enumerate(samples):
        print(f"\n{'='*30} Processing Sample {i+1}/{len(samples)} {'='*30}")
        
        # 使用配置系统提取字段
        fields = dataset_cfg.extract_fields(sample, dataset_name)
        question = fields["question"]
        golden_answer = fields["answer"]
        
        print(f"Question: {question}")
        print(f"Golden Answer: {golden_answer}")
        
        # 使用配置系统格式化 context
        chunks = dataset_cfg.format_context(fields["context"], dataset_name)
        print(f"Number of chunks: {len(chunks)}")
        
        # 统一生成接口：处理 Hit/Miss，自动 Capture
        print("\n[Phase 1 & 2] Unified Generation with Cache Craft...")
        pred_answer, debug_info = pipeline.generate(
            chunks, 
            question=question, 
            prompt_template=user_template,
            system_prompt=system_prompt
        )
        
        print(f"\nSample {i+1} Prediction: {pred_answer}")
        print(f"{'='*70}")

        if args.output_file:
            with open(args.output_file, "a") as f:
                f.write(f"==================== Processing Sample {i+1}/{len(samples)} ====================\n")
                f.write(f"Question: {question}\n")
                f.write(f"Golden Answer: {golden_answer}\n")
                f.write(f"Generating Answer: {pred_answer.strip()}\n\n")

if __name__ == "__main__":
    main()
