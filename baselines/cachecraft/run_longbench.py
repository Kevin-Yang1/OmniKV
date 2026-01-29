# CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python baselines/cachecraft/run_longbench.py

import json
import argparse
import os
import torch
import random
from baselines.cachecraft.pipeline import CacheCraftPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="在 LongBench 格式数据集上运行 Cache-Craft 基准测试")
    parser.add_argument("--data_path", type=str, default="/NV1/ykw/projects/OmniKV/datasets/longbench/2wikimqa_e_dense_remix_v1.json", help="数据集路径 (JSON格式)")
    parser.add_argument("--model_path", type=str, default="/NV1/ykw/models/Meta-Llama-3.1-8B-Instruct", help="HuggingFace 模型路径或名称")
    parser.add_argument("--num_samples", type=int, default=None, help="运行测试的样本数量 (用于快速调试)")
    parser.add_argument("--alpha", type=float, default=1.0, help="CFO (Context-Aware Fractional Offloading) 算法的 alpha 参数")
    parser.add_argument("--device", type=str, default="cuda", help="使用的计算设备 (如 cuda, cpu)")
    parser.add_argument("--disable_caching", action="store_true", help="禁用 KV 缓存功能 (MetadataStore)")
    parser.add_argument("--disable_recompute", action="store_true", help="禁用命中块的 token 重算，仅直接复用 KV")
    parser.add_argument("--shuffle_chunks", action="store_true", help="打乱 chunks 中的文档顺序")
    parser.add_argument("--run_all_modes", action="store_true", help="一次运行所有三种模式 (nocache, norecompute, recompute)")
    parser.add_argument("--recompute_ratios", type=float, nargs='+', default=None, help="重算比例列表，例如: 0.1 0.3 0.5。启用此项将测试一系列固定比例")
    parser.add_argument("--output_file", type=str, default=None, help="输出文件路径")
    return parser.parse_args()

def load_data(path, num_samples):
    """
    加载数据集，并截取指定数量的样本用于测试
    """
    print(f"Loading data from {path}...")
    with open(path, 'r') as f:
        data = [json.loads(line) for line in f] if path.endswith(".jsonl") else json.load(f)
    print(f"Total samples: {len(data)}")
    return data[:num_samples]

def run_mode(args, mode_name, samples, prompt_template, shared_pipeline, shared_ground_truth=None, fixed_ratio=None):
    print(f"\n{'#'*40}")
    print(f"Running Mode: {mode_name} (Ratio: {fixed_ratio if fixed_ratio is not None else 'Auto/CFO'})")
    print(f"{'#'*40}\n")
    
    # 根据模式名称设置参数
    disable_caching = False
    disable_recompute = False
    record_gt = False
    
    if mode_name == "nocache":
        disable_caching = True
        if shared_ground_truth is not None:
             record_gt = True
    elif mode_name == "norecompute":
        disable_caching = False
        disable_recompute = True
    elif mode_name == "recompute" or mode_name.startswith("recompute_ratio_"):
        disable_caching = False
        disable_recompute = False
    else:
        # 兼容手动指定参数的情况
        disable_caching = args.disable_caching
        disable_recompute = args.disable_recompute
        
    # 自动生成输出路径逻辑
    base_output_dir = "/NV1/ykw/projects/OmniKV/baselines/cachecraft/output/"
    if "datasets/" in args.data_path:
        relative_path = args.data_path.split("datasets/")[-1]
        dataset_rel_name = os.path.splitext(relative_path)[0]
    else:
        dataset_rel_name = os.path.splitext(os.path.basename(args.data_path))[0]
        
    suffix = mode_name
    if args.shuffle_chunks:
        suffix += "_shuffle"
    filename = f"craft_{suffix}_results.txt"
        
    output_file = os.path.join(base_output_dir, dataset_rel_name, filename)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"Auto-generated output_file: {output_file}")
    
    # [NEW] 生成 JSONL 格式的测评输出文件
    # 使用 basename 避免创建多余的子目录
    dataset_basename = os.path.splitext(os.path.basename(args.data_path))[0]
    eval_output_dir = os.path.join(base_output_dir, "longbench_eval")
    os.makedirs(eval_output_dir, exist_ok=True)
    jsonl_output_file = os.path.join(eval_output_dir, f"{dataset_basename}_{suffix}.jsonl")
    print(f"JSONL output file for evaluation: {jsonl_output_file}")
    
    with open(output_file, "w") as f:
        f.write(f"Run Log for Mode: {mode_name} (Fixed Ratio: {fixed_ratio})\n\n")
    
    # 初始化 JSONL 输出文件（清空旧内容）
    with open(jsonl_output_file, "w", encoding="utf-8") as f:
        pass

    # [优化] 使用共享 Pipeline 并动态更新配置
    pipeline = shared_pipeline
    
    # 1. 每次模式运行前，清除 KV Store 以保证状态干净且一致 (Cache Hit 环境一致)
    pipeline.clear_kv_store()
    
    # 2. 更新配置
    pipeline.update_config(
        enable_caching=not disable_caching,
        enable_recompute=not disable_recompute,
        fixed_recompute_ratio=fixed_ratio,
        record_ground_truth=record_gt
    )
    
    # 逐个样本处理
    for i, sample in enumerate(samples):
        print(f"\n{'='*50} Processing Sample {i+1}/{len(samples)} {'='*50}")
        question = sample['input']  # LongBench 格式使用 'input'
        golden_answer = sample['answers'][0] if isinstance(sample['answers'], list) else sample['answers']
        
        print(f"Question: {question}")
        print(f"Golden Answer: {golden_answer}")
        
        # 确保每个 chunk 以 \n\n 结尾，以保证 token 分隔的一致性，防止边界合并
        # 注意：这里我们生成一个新的 list，不修改原 sample
        processed_chunks = []
        source_chunks = list(sample['context']) # 创建副本
        
        if args.shuffle_chunks:
            print("Shuffling chunks order...")
            random.seed(42)
            random.shuffle(source_chunks)
            
        processed_chunks = [c if c.endswith("\n\n") else (c + "\n\n" if c.endswith("\n") else c + "\n\n") for c in source_chunks]
        print(f"Number of chunks: {len(processed_chunks)}")
        
        # 统一生成接口
        print("\n[Phase 1 & 2] Unified Generation with Cache Craft...")
        pred_answer, debug_info = pipeline.generate(processed_chunks, question=question, prompt_template=prompt_template)
        
        print(f"\nSample {i+1} Prediction: {pred_answer}")
        print(f"{'='*100}")
        
        # [NEW] 计算上下文总长度（用于测评）
        total_length = sum([
            len(pipeline.tokenizer(chunk, add_special_tokens=False).input_ids)
            for chunk in processed_chunks
        ])
        
        # [NEW] 写入 JSONL 格式的测评文件
        with open(jsonl_output_file, "a", encoding="utf-8") as f:
            json.dump({
                "pred": pred_answer.strip(),
                "answers": sample['answers'],
                "all_classes": sample.get('all_classes', None),
                "length": total_length
            }, f, ensure_ascii=False)
            f.write('\n')

        if output_file:
            # 使用 "a" 模式追加结果 
            with open(output_file, "a") as f:
                f.write(f"\n{'='*50} Processing Sample {i+1}/{len(samples)} {'='*50}\n")
                f.write(f"Question: {question}\n")
                f.write(f"Golden Answer: {golden_answer}\n")
                f.write(f"Generating Answer: {pred_answer.strip()}\n")
                # 写入调试信息
                if debug_info:
                    f.write(f"Debug Hidden State: {debug_info.get('hidden_state', 'N/A')}\n")
                    f.write(f"Debug Logits: {debug_info.get('logits', 'N/A')}\n")
                    if debug_info.get("recompute_stats") and debug_info["recompute_stats"] != "N/A":
                        f.write(f"[Recompute Stats] {debug_info['recompute_stats']}\n")

                    if debug_info.get("chunk_details"):
                        f.write("\n[Chunk Details]\n")
                        for cd in debug_info["chunk_details"]:
                            tokens_str = str(cd['recomputed_tokens']) if cd.get('recomputed_tokens') else "None"
                            f.write(f"  Chunk {cd['chunk_idx']}: CCI={cd['cci']}, Beta={cd['beta_prime']}, Ratio={cd['recompute_ratio']}\n")
                            if cd.get('recomputed_tokens'):
                                f.write(f"    Indices: {tokens_str}\n")

                    if debug_info.get("deviation_analysis"):
                        f.write(f"{debug_info['deviation_analysis']}\n")

                    # if "attn_analysis" in debug_info:
                    #     f.write(f"{debug_info['attn_analysis']}\n")


def main():
    args = parse_args()
    
    # 加载 Prompt 模板
    config_path = os.path.join(os.path.dirname(__file__), "config/dataset2prompt.json")
    prompt_template = None
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            templates = json.load(f)
            # 默认使用通用或根据数据集名称匹配
            prompt_template = templates.get("longbench", templates.get("hotpotqa", None))

    # 加载测试数据 (只加载一次)
    samples = load_data(args.data_path, args.num_samples)

    # [优化] 预先加载模型和 Tokenizer，避免重复加载
    print(f"Loading Base Model: {args.model_path}")
    
    # [MODIFIED] 支持多 GPU 推理
    # 使用 device_map="auto" 自动将模型分布到多个可用 GPU
    # 这样可以解决单卡显存不足的问题
    if args.device == "cuda":
        device_map = "auto"  # 自动分配到多个 GPU
        print(f"Using multi-GPU with device_map='auto'")
    else:
        device_map = args.device
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16, 
        device_map=device_map,
        trust_remote_code=True
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    shared_gt = {} if (args.run_all_modes or args.recompute_ratios) else None
    
    # 初始化共享的 Pipeline 实例
    shared_pipeline = CacheCraftPipeline(
        model_name_or_path=args.model_path,
        alpha=args.alpha,
        device=args.device,
        ground_truth_store=shared_gt,
        # 传入已加载的对象
        model=model,
        tokenizer=tokenizer
    )

    # 1. 运行标准模式
    if args.run_all_modes:
        modes = ["nocache", "norecompute", "recompute"]
        for mode in modes:
            run_mode(args, mode, samples, prompt_template, shared_pipeline, shared_ground_truth=shared_gt)
    elif shared_gt is not None:
        pass
    else:
        # 单模式运行
        if args.disable_caching:
            mode = "nocache"
        elif args.disable_recompute:
            mode = "norecompute"
        else:
            mode = "recompute"
        run_mode(args, mode, samples, prompt_template, shared_pipeline)

    # 2. 运行扩充的固定比例模式
    if args.recompute_ratios:
        print(f"\n[Extended Test] Running fixed recompute ratios: {args.recompute_ratios}")
        
        for ratio in args.recompute_ratios:
            ratio_val = float(ratio)
            mode_name = f"recompute_ratio_{ratio_val}"
            run_mode(args, mode_name, samples, prompt_template, shared_pipeline, shared_ground_truth=shared_gt, fixed_ratio=ratio_val)

if __name__ == "__main__":
    main()
