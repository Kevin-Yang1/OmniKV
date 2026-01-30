"""
================================================================================
文件名: eval.py
描述: LongBench 评测脚本
================================================================================

功能:
    根据模型生成的预测结果计算各种任务的评分（如 QA F1, Rouge, Classification 等）。
    支持标准 LongBench 和 LongBench-E (长度分段评估) 两种模式。

依赖:
    - metrics.py: 包含具体的评分函数实现（qa_f1_score, rouge_score 等）
    - numpy: 用于计算平均分
    - json: 读写 JSONL 预测文件和结果文件

主要流程:
    1. 解析命令行参数（模型名称、配置、是否为 LongBench-E）
    2. 定位预测文件目录（支持 CacheCraft 和标准路径）
    3. 遍历目录下所有 .jsonl 预测文件
    4. 读取每个文件的预测结果 (pred) 和标准答案 (answers)
    5. 根据数据集名称选择对应的评分函数
    6. 计算得分并保存到 result.json 文件

================================================================================
使用方法:
================================================================================

基本用法:
---------
    python benchmark/long_bench/eval.py --model <MODEL_NAME> --cfg <CONFIG_NAME>

参数说明:
---------
    --model     (必需) 模型名称，用于构建预测文件路径
                       - 对于 CacheCraft: 使用 "cachecraft"
                       - 对于其他模型: 使用自定义模型名
    
    --cfg       (必需) 配置名称，对应实验的子目录名
                       例如: exp1, baseline, alpha_1.0 等
    
    --e         (可选) 是否在 LongBench-E 数据集上评测
                       - 添加此参数: 按长度区间 (0-4k, 4-8k, 8k+) 分段评分
                       - 不添加: 计算整体平均分

示例命令:
---------
1. 评估 CacheCraft 标准 LongBench 结果:
   python benchmark/long_bench/eval.py --model cachecraft --cfg exp1

2. 评估 CacheCraft LongBench-E 结果:
   python benchmark/long_bench/eval.py --model cachecraft --cfg exp1 --e

3. 评估其他模型的结果:
   python benchmark/long_bench/eval.py --model my_model --cfg baseline

输入文件格式:
-----------
预测文件应为 JSONL 格式（每行一个 JSON 对象），包含以下字段:
    {
        "pred": "模型的预测答案",
        "answers": ["标准答案1", "标准答案2"],  // 可能有多个正确答案
        "all_classes": ["类别1", "类别2"] or null,  // 仅分类任务需要
        "length": 5234  // 可选，LongBench-E 需要
    }

文件路径约定:
-----------
标准模型:
    - LongBench: benchmark/long_bench/pred/{model}/{cfg}/*.jsonl
    - LongBench-E: benchmark/long_bench/pred_e/{model}/{cfg}/*.jsonl

CacheCraft 模型:
    - 输入: baselines/cachecraft/output/longbench_eval/*.jsonl
    - 输出: baselines/cachecraft/output/longbench_eval/eval_results/{cfg}_result.json

预测文件命名规范:
--------------
    {dataset_name}[_variant][_mode].jsonl
    
    示例:
    - narrativeqa.jsonl              # 标准数据集
    - narrativeqa_rag.jsonl          # 带 RAG 变体
    - 2wikimqa_nocache.jsonl         # 无缓存模式
    - hotpotqa_recompute.jsonl       # 重算模式
    - narrativeqa_rag_norecompute.jsonl  # RAG 变体 + 无重算模式
    - qasper_recompute_ratio_0.3.jsonl   # 固定重算比例 30%

支持的数据集:
-----------
    QA 类 (F1 Score):
        - narrativeqa, qasper, multifieldqa_en, hotpotqa, 2wikimqa, 
          musique, triviaqa
    
    中文 QA (中文 F1):
        - multifieldqa_zh
    
    摘要类 (Rouge Score):
        - gov_report, qmsum, multi_news, samsum (英文)
        - dureader, vcsum (中文)
    
    分类类:
        - trec, lsht
    
    检索类:
        - passage_retrieval_en, passage_retrieval_zh
    
    计数类:
        - passage_count
    
    代码类 (Code Similarity):
        - lcc, repobench-p

输出结果:
--------
result.json 文件示例:
    {
        "narrativeqa": 23.45,          # 标准 LongBench: 单个分数
        "2wikimqa_rag": 45.67,
        "hotpotqa": {                  # LongBench-E: 分段分数
            "0-4k": 78.9,
            "4-8k": 65.4,
            "8k+": 52.3
        }
    }

注意事项:
--------
1. 数据集变体支持:
   - 数据集名可以包含后缀（如 _rag, _dense 等）
   - 评分时会自动匹配到基础数据集的评分函数
   - 例如: "2wikimqa_rag" 会使用 "2wikimqa" 的 qa_f1_score

2. 模式后缀自动去除:
   - _nocache, _norecompute, _recompute 等后缀会被自动识别
   - _recompute_ratio_X.X 格式的固定比例后缀也会被处理
   
3. 特殊数据集处理:
   - trec, triviaqa, samsum, lsht 等数据集只取预测的第一行
   - 多答案样本取最高分（多个标准答案，选择F1最高的）

4. CacheCraft 集成:
   - 使用 --model cachecraft 时，自动从专用目录读取/写入
   - 支持多种运行模式的结果同时评测

调试提示:
--------
- 如果找不到评分函数: 检查数据集名称是否在 dataset2metric 字典中
- 如果路径不存在: 检查 --model 和 --cfg 参数是否正确
- 如果分数异常: 检查 JSONL 文件格式是否正确，pred/answers 字段是否存在

================================================================================
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
        # 去除 .jsonl 后缀，保留完整数据集名称（包含模式后缀，包括小数点）
        if filename.endswith('.jsonl'):
            dataset_full = filename[:-6]  # 去除 '.jsonl'
        else:
            dataset_full = filename.split('.')[0]  # 回退到原逻辑
        
        # 提取基础数据集名称（用于查找评分函数）
        dataset_base = dataset_full
        mode_suffixes = ['_nocache', '_norecompute', '_recompute']
        
        # 支持固定比例的后缀，如 _recompute_ratio_0.3
        if '_recompute_ratio_' in dataset_full:
            dataset_base = dataset_full.split('_recompute_ratio_')[0]
        else:
            for suffix in mode_suffixes:
                if dataset_full.endswith(suffix):
                    dataset_base = dataset_full[:-len(suffix)]
                    break
        
        print(f"Processing file: {filename} -> dataset_full: {dataset_full}, dataset_base: {dataset_base}")
        
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
            # [MODIFIED] 使用 dataset_base 查找评分函数
            # 首先尝试直接查找
            if dataset_base in dataset2metric:
                metric_func = dataset2metric[dataset_base]
                matched_base = dataset_base
            else:
                # 如果找不到，尝试前缀匹配找到对应的基础数据集评分函数
                metric_func = None
                matched_base = None
                for base_dataset in dataset2metric.keys():
                    if dataset_base.startswith(base_dataset):
                        # 检查后缀是否符合常见模式（以 _ 开头）
                        suffix = dataset_base[len(base_dataset):]
                        if suffix == '' or suffix.startswith('_'):
                            metric_func = dataset2metric[base_dataset]
                            matched_base = base_dataset
                            print(f"  Using metric from '{matched_base}' for variant '{dataset_base}'")
                            break
                
                if metric_func is None:
                    raise ValueError(f"No matching metric function found for '{dataset_base}'")
            
            # 计算得分，使用完整的数据集名称（包含模式后缀）作为 key
            if args.e:
                score = scorer_e(matched_base, predictions, answers, lengths, all_classes)
            else:
                score = scorer(matched_base, predictions, answers, all_classes)
            scores[dataset_full] = score  # ⭐ 使用完整名称（含模式后缀）作为 key
        except Exception as e:
            print(f"error in {dataset_full}: {e}")
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
