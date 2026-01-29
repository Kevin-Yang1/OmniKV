"""
文件名: eval.py
描述: LongBench 评测脚本
功能: 根据模型生成的预测结果计算各种任务的评分（如 QA F1, Rouge, Classification 等）。
依赖: metrics.py (包含具体的评分函数实现)

主要逻辑:
1. 解析命令行参数，确定模型名称、配置和是否为 LongBench-E (E 结尾数据集)。
2. 遍历指定目录下的所有预测文件 (.jsonl)。
3. 读取预测结果 (pred) 和标准答案 (answers)。
4. 根据数据集类型选择对应的评分函数 (metric)。
5. 计算得分并保存到 result.json 文件中。
"""

import os
import json
import argparse
import numpy as np

# 从 metrics 模块导入各种评分函数
from metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

# 数据集名称到评分函数的映射字典
dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}


def parse_args(args=None):
    """
    解析命令行参数。

    Args:
        args (list, optional): 参数列表。默认为 None，从 sys.argv 获取。

    Returns:
        argparse.Namespace: 解析后的参数对象，包含 model, e, cfg 等属性。
    """
    parser = argparse.ArgumentParser()
    # 模型名称，用于构建文件路径
    parser.add_argument('--model', type=str, required=True)
    # 是否在 LongBench-E 数据集上进行评测
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    # 具体配置名称 (如 exp1)，也是路径的一部分
    parser.add_argument("--cfg", default=None, required=True)
    return parser.parse_args(args)


def scorer_e(dataset, predictions, answers, lengths, all_classes):
    """
    计算 LongBench-E 数据集的评分，按长度区间分段统计。
    
    Args:
        dataset (str): 数据集名称，用于查找对应的评分函数。
        predictions (list[str]): 模型生成的预测文本列表。
        answers (list[list[str]]): 标准答案列表，每个元素是该样本的所有可能答案列表。
        lengths (list[int]): 每个样本的上下文长度列表。
        all_classes (list[str] or None): 对于分类任务，包含所有类别标签；否则为 None。

    Returns:
        dict: 包含不同长度区间 ("0-4k", "4-8k", "8k+") 得分的字典。
    """
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
        score = 0.
        # 对于特定分类数据，只取预测的第一行
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        # 遍历所有可能的正确答案，取最高分
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        # 根据长度归类分数
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    # 计算每个区间的平均分
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores


def scorer(dataset, predictions, answers, all_classes):
    """
    计算标准 LongBench 数据集的平均分。

    Args:
        dataset (str): 数据集名称。
        predictions (list[str]): 预测文本列表。
        answers (list[list[str]]): 标准答案列表。
        all_classes (list[str] or None): 所有类别标签。

    Returns:
        float: 该数据集的平均得分 (百分制，保留两位小数)。
    """
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        # 对于特定分类数据，只取预测的第一行
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        # 遍历所有可能的正确答案，取最高分
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)


if __name__ == '__main__':
    args = parse_args()
    scores = dict()
    
    # [MODIFIED] 支持从 cachecraft 输出目录读取预测文件
    # 优先检查 cachecraft 输出目录，如果不存在则回退到标准目录
    cachecraft_pred_path = "baselines/cachecraft/output/longbench_eval"
    
    # 根据是否是 LongBench-E 选择预测结果的路径
    if args.e:
        standard_path = f"benchmark/long_bench/pred_e/{args.model}/{args.cfg}"
    else:
        standard_path = f"benchmark/long_bench/pred/{args.model}/{args.cfg}"
    
    # 如果指定了 cachecraft 模型，优先使用 cachecraft 输出目录
    if args.model == "cachecraft" and os.path.exists(cachecraft_pred_path):
        path = cachecraft_pred_path
        print(f"Using CacheCraft prediction path: {path}")
    elif os.path.exists(standard_path):
        path = standard_path
        print(f"Using standard prediction path: {path}")
    else:
        # 回退到标准路径（即使不存在，后续会报错）
        path = standard_path
        print(f"Warning: Path may not exist: {path}")
    
    # 获取目录下所有预测文件
    all_files = os.listdir(path)
    print("Evaluating on:", all_files)
    
    # 遍历每个文件进行评估
    for filename in all_files:
        if not filename.endswith("jsonl"):
            continue
        predictions, answers, lengths = [], [], []
        
        # [MODIFIED] 数据集名称提取逻辑
        # 去除 .jsonl 后缀
        dataset_full = filename.split('.')[0]
        
        # 仅去除模式后缀（_nocache, _norecompute, _recompute 等），保留变体后缀
        dataset = dataset_full
        mode_suffixes = ['_nocache', '_norecompute', '_recompute']
        
        # 支持固定比例的后缀，如 _recompute_ratio_0.3
        if '_recompute_ratio_' in dataset_full:
            dataset = dataset_full.split('_recompute_ratio_')[0]
        else:
            for suffix in mode_suffixes:
                if dataset_full.endswith(suffix):
                    dataset = dataset_full[:-len(suffix)]
                    break
        
        print(f"Processing file: {filename} -> dataset: {dataset}")
        
        # 读取 jsonl 文件
        with open(f"{path}/{filename}", "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                predictions.append(data["pred"])
                answers.append(data["answers"])
                all_classes = data["all_classes"]
                if "length" in data:
                    lengths.append(data["length"])
        # 计算得分
        try:
            # [MODIFIED] 保留完整数据集名称，仅在查找评分函数时匹配
            # 首先尝试直接查找
            if dataset in dataset2metric:
                metric_func = dataset2metric[dataset]
            else:
                # 如果找不到，尝试前缀匹配找到对应的基础数据集评分函数
                metric_func = None
                matched_base = None
                for base_dataset in dataset2metric.keys():
                    if dataset.startswith(base_dataset):
                        # 检查后缀是否符合常见模式（以 _ 开头）
                        suffix = dataset[len(base_dataset):]
                        if suffix == '' or suffix.startswith('_'):
                            metric_func = dataset2metric[base_dataset]
                            matched_base = base_dataset
                            print(f"  Using metric from '{matched_base}' for variant '{dataset}'")
                            break
                
                if metric_func is None:
                    raise ValueError(f"No matching metric function found for '{dataset}'")
            
            # 计算得分，结果中使用完整的数据集名称
            if args.e:
                score = scorer_e(matched_base or dataset, predictions, answers, lengths, all_classes)
            else:
                score = scorer(matched_base or dataset, predictions, answers, all_classes)
            scores[dataset] = score  # 使用完整名称作为 key
        except Exception as e:
            print(f"error in {dataset}: {e}")
            pass

    # [MODIFIED] 构造结果输出路径
    # 如果是 cachecraft 模型，输出到专门的评测结果目录
    if args.model == "cachecraft":
        eval_results_dir = "baselines/cachecraft/output/longbench_eval/eval_results"
        os.makedirs(eval_results_dir, exist_ok=True)
        
        # 使用 cfg 作为文件名的一部分
        result_filename = f"{args.cfg}_result.json" if args.cfg else "result.json"
        out_path = os.path.join(eval_results_dir, result_filename)
        print(f"Saving results to CacheCraft output directory: {out_path}")
    else:
        # 其他模型使用标准输出路径
        if args.e:
            out_path = f"benchmark/long_bench/pred_e/{args.model}/{args.cfg}/result.json"
        else:
            out_path = f"benchmark/long_bench/pred/{args.model}/{args.cfg}/result.json"
    
    # 打印并写入结果
    print("\nEvaluation Scores:")
    print(scores)
    print(f"\nResults saved to: {out_path}")
    with open(out_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
