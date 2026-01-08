"""LongBench 推理结果记录脚本。

核心功能：批量遍历 LongBench 任务数据（或 E 版），通过指定模型生成回答并写入预测文件；可选地利用 `idx_tracer` 打点分析 token 选择。
依赖：Transformer 模型/Tokenizer、`infer.get_any_chat_api`（自定义模型接口）、`tiny_tools.read_json` 配置加载、`torch.multiprocessing`（备用多进程逻辑）以及 `idx_tracer`。
主要流程：解析 CLI 参数 -> 加载模型权重/Tokenizer -> 为每个数据样本构建 prompt -> 调用 model.generate（或自定义 API）获取生成 -> 后处理并写入 JSONL -> 可选触发指标保存。
"""

import os
import json
from tqdm import tqdm
import numpy as np
import random
import argparse
from tiny_tools.read_json import read_config
import torch.multiprocessing as mp
from torch.multiprocessing import Process
import torch
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM, GenerationConfig
import torch.distributed as dist
from infer import get_any_chat_api
from tiny_tools.tensor_tools import idx_tracer


def parse_args(args=None):
    """解析命令行参数并返回统一的配置对象。

    Args:
        args (List[str], optional): CLI 参数列表，默认为 `sys.argv[1:]`。
    Returns:
        argparse.Namespace: 包含模型名、数据集类型、配置路径等字段的对象。
    """
    parser = argparse.ArgumentParser()
    # 移除过时的 choices 限制，允许用户在 model2path.json 中自由定义新模型
    parser.add_argument('--model', type=str, default=None,
                        help="指定本次推理使用的模型名称，需与 config/model2path.json 对应")
    parser.add_argument('--e', action='store_true', help="启用 LongBench-E 任务序列")
    parser.add_argument("--cfg", default=None, help="传入的 pipeline 配置文件路径")
    parser.add_argument("--ws", default=2, type=int, help='多进程 world size（目前只用于占位）')
    parser.add_argument("--task_start_id", default=0, type=int, help='按任务列表偏移，用于续跑')
    parser.add_argument("--task", default=None, type=str, help='逗号分割的子任务名，将覆盖默认序列')
    return parser.parse_args(args)


# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    """根据模型类型构造 Chat 模型的输入 prompt。

    Purpose:
        不同模型的 chat 接口需要不同的封装方式，本函数根据 `model_name` 选择对应模板。
    Args:
        tokenizer (PreTrainedTokenizer): 负责分词的实例，某些模型提供辅助构造方法。
        prompt (str): 原始 prompt。
        model_name (str): 当前模型名称，用于匹配对应模板。
    Returns:
        str: 构造完成的 prompt，以便直接送入模型。
    """
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name and "chatglm3" not in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama2" in model_name or 'llama-2' in model_name.lower():
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    return prompt


def post_process(response, model_name):
    """清理模型输出中的特殊标记，使其适合后续评估。

    Args:
        response (str): 模型生成的原始文本。
        model_name (str): 用于判定所需的清洗策略。
    Returns:
        str: 清洗后的文本。
    """
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response


def get_pred(rank, world_size, data, max_length, max_gen, prompt_format, dataset, device, model_name, model2path,
             out_path, args):
    """
    核心推理函数：执行数据集遍历、文本截断、模型生成并保存结果。

    Args:
        rank (int): 进程排名。
        world_size (int): 进程总数。
        data (List[dict]): 待推理的数据集列表。
        max_length (int): 输入 Token 的最大长度限制。
        max_gen (int): 生成结果的最大 Token 数量。
        prompt_format (str): 用于填充数据的 Prompt 模板。
        dataset (str): 数据集名称。
        device (int): 默认 GPU 设备序号。
        model_name (str): 模型名称。
        model2path (dict): 模型名称到本地路径的映射。
        out_path (str): 结果保存路径。
        args (Namespace): 命令行参数对象。
    """
    seed_everything(42)
    d_cfg = read_config(args.cfg)
    device = 0

    # 加载模型和分词器
    model, tokenizer, model_max_length = load_model_and_tokenizer(model2path[model_name], model_name, device, args.cfg)
    if model_max_length is not None:
        max_length = model_max_length
        print(f"max_length is set to {max_length}")

    # 遍历数据集进行预测
    for json_obj in tqdm(data, desc=f'{dataset}'):
        # 填充 prompt 模板
        prompt = prompt_format.format(**json_obj)
        
        # 针对超长文本进行中间截断 (Middle-out truncation)
        # 这种做法是为了同时保留开头（指令）和末尾（问题/最近信息），删掉中间的文档内容。
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = (tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) +
                      tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True))

        # 构建对话模板 (Chat Template)
        # 只有在特定的任务和模型下才包装对话指令格式
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc",
                           "repobench-p"]:
            if 'my_model' not in model_name:
                prompt = build_chat(tokenizer, prompt, model_name)
            else:
                prompt = build_chat(tokenizer, prompt, d_cfg['model_name'])

        # 准备模型输入
        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]
        
        # 执行推理生成
        if dataset == "samsum":
            # samsum 数据集需要特殊处理停止词（如换行符），防止生成冗余内容
            if "my_model" not in model_name:
                output = model.generate(
                    **input,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    min_length=context_length + 1,
                    eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
                )[0]
            else:
                output = model(
                    prompt, generation_config=None,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    min_length=context_length + 1,
                    eos_token_id=[tokenizer.eos_token_id,
                                  tokenizer.encode("\n", add_special_tokens=False)[-1]],
                    skip_special_tokens=True)
        else:
            # 标准生成逻辑
            if "my_model" not in model_name:
                output = model.generate(
                    **input,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                )[0]
            else:
                # 调用 OmniKV 的全权生成接口
                output = model(
                    prompt, generation_config=None,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    skip_special_tokens=True
                )
        
        # 结果解码与后处理
        if not isinstance(output, str):
            # 将生成的 token id 会转为字符串
            pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        else:
            pred = output

        pred = post_process(pred, model_name)
        
        # 将预测回答序列化并追加写入结果文件
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"],
                       "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')

        # 诊断逻辑：如果设置了环境变量，则保存 Token 选择的索引并提前终止
        if os.environ.get('SAVE_SELECTED_IDX', False):
            idx_tracer.save_idx()
            if idx_tracer.num_samples > 20: # 采样数限制
                return


def seed_everything(seed):
    """固定随机种子，保证 CUDA/CPU 生成过程可复现。"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(path, model_name, device, cfg):
    """装载模型与分词器，支持多种架构的推理后端集成。

    Purpose:
        根据模型名称识别对应的架构（如 Llama, ChatGLM, 或自定义的 OmniKV 模型），
        并完成权重加载、精度设置（bf16）以及设备分配。

    Args:
        path (str): 模型权重（Checkpoint）的本地路径或 HuggingFace ID。
        model_name (str): 模型标识符，用于匹配不同的加载逻辑。
        device (int/str): 推理运行的目标设备序号。
        cfg (str): 专门用于 'my_model' (OmniKV) 的详细 pipeline 配置文件路径。

    Returns:
        tuple: 包含三个元素:
            - model: 已加载并移动到对应设备的模型对象。
            - tokenizer: 对应的分词器实例。
            - max_length (int/None): 模型支持的最大上下文长度。
    """
    max_length = None
    
    # 支持 ChatGLM, InternLM, XGen 等需要远程代码执行的模型类
    if "chatglm" in model_name or "internlm" in model_name or "xgen" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(
            device)
            
    # 核心路径：加载自定义的推理接口（如 OmniKV 压缩模型）
    if 'my_model' in model_name:
        # 调用 infer.py 中的统一入口获取模型实例
        model, tokenizer, max_length, other_kwargs = get_any_chat_api(cfg)
        # 确保 EOS Token 设置正确，以防止生成过程中止或无限循环
        tokenizer.eos_token_id = other_kwargs['eos_token_id']
        print("EOS is", tokenizer.eos_token_id)
        
    # 标准 Llama-2 架构加载逻辑，启用 Flash Attention 2 加速
    elif "llama2" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = LlamaForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
                                                 attn_implementation="flash_attention_2").to(device)
                                                 
    # 对 LongChat 和 Vicuna 进行特殊处理（通常基于 FastChat 库加载）
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import load_model
        # 注意：这里可能需要对 Llama Attention 进行 Monkey Patch 以支持超长上下文
        replace_llama_attn_with_flash_attn() 
        model, _ = load_model(
            path,
            device='cpu', # 先在 CPU 加载防止显存溢出
            num_gpus=0,
            load_8bit=False,
            cpu_offloading=False,
            debug=False,
        )
        model = model.to(device)  # 移动到指定 GPU
        model = model.bfloat16() # 转换为 bfloat16 混合精度
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
        
    return model, tokenizer, max_length


def load_dataset(path, mode='r'):
    """从 JSONL 文件读取任务样本。

    Args:
        path (str): 目标 JSONL 文件路径。
        mode (str): 文件打开模式。
    Returns:
        List[dict]: 每行 JSON 解析出的字典对象。
    """
    data = [json.loads(line) for line in open(path, mode, encoding="utf-8")]
    return data


class CustomProcess(Process):
    def __init__(self, env_var_key, env_var_value, *args, **kwargs):
        """用于在 spawn 时提前设置环境变量的包装 Process。"""
        super().__init__(*args, **kwargs)
        self.env_var_key = env_var_key
        self.env_var_value = env_var_value
        os.environ[self.env_var_key] = self.env_var_value

    def run(self):
        # 在子进程中重新设置环境变量，确保与主进程一致
        os.environ[self.env_var_key] = self.env_var_value
        super().run()


if __name__ == '__main__':
    # 解析命令行参数
    args = parse_args()
    # 设置并行进程数（world size）
    world_size = args.ws
    # 强制使用 spawn 模式启动多进程，避免 CUDA 初始化冲突
    mp.set_start_method('spawn', force=True)

    # 加载模型名称到路径的映射配置文件
    model2path = json.load(open("benchmark/long_bench/config/model2path.json", "r"))
    # 加载模型对应的最大上下文长度配置
    model2maxlen = json.load(open("benchmark/long_bench/config/model2maxlen.json", "r"))
    model_name = args.model
    # 获取当前模型的最大上下文限制，若无则设为 -1
    max_length = model2maxlen.get(model_name, -1)
    # 读取本次运行的具体 pipeline 配置文件 (OmniKV 的核心配置)
    d_cfg = read_config(args.cfg)
    
    # 决定待评测的数据集列表
    if args.e:
        # LongBench-E 版本数据集（通常包含不同长度梯度的样本）
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news",
                    "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        # 标准 LongBench 数据集
        datasets = [
            "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique",
            "dureader", "gov_report", "qmsum",
            "vcsum", "trec", "triviaqa",
            "lsht",
            "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"
        ]
    
    # 支持从特定索引开始运行，方便断点续跑
    datasets = datasets[args.task_start_id:]
    # 如果手动指定了 --task，则仅运行指定的任务
    if args.task is not None:
        datasets = args.task.split(',')

    # 加载每个数据集对应的 prompt 模板
    dataset2prompt = json.load(open("benchmark/long_bench/config/dataset2prompt.json", "r"))
    # 加载每个数据集对应的最大生成长度限制
    dataset2maxlen = json.load(open("benchmark/long_bench/config/dataset2maxlen.json", "r"))
    
    # 处理基础路径偏移（兼容特定的 NAS 或计算集群环境）
    base_path = ''
    if os.environ.get("NO_NAS", False):
        base_path = '/jitai/'
        
    # 核心循环：依次处理每个选定的数据集
    for dataset in datasets:
        if args.e:
            # 加载 E 版数据
            data = load_dataset(f'benchmark/long_bench/data/{dataset}_e.jsonl', 'r')
            # 自动创建预测结果文件夹（包含模型名和配置名）
            if not os.path.exists(f"{base_path}benchmark/long_bench/pred_e/{model_name}/{args.cfg}"):
                os.makedirs(f"{base_path}benchmark/long_bench/pred_e/{model_name}/{args.cfg}", exist_ok=True)
            out_path = f"{base_path}benchmark/long_bench/pred_e/{model_name}/{args.cfg}/{dataset}.jsonl"
        else:
            # 加载标准版数据
            data = load_dataset(f'benchmark/long_bench/data/{dataset}.jsonl', 'r')
            if not os.path.exists(f"{base_path}benchmark/long_bench/pred/{model_name}/{args.cfg}"):
                os.makedirs(f"{base_path}benchmark/long_bench/pred/{model_name}/{args.cfg}", exist_ok=True)
            out_path = f"{base_path}benchmark/long_bench/pred/{model_name}/{args.cfg}/{dataset}.jsonl"
        
        # 初始化预测文件（清空旧内容）
        with open(out_path, 'w') as _in:
            pass
            
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]
        
        # 清理显存缓存，准备开始新一轮推理
        torch.cuda.empty_cache()
        # 调用核心推理函数 get_pred，直接开始顺序推理
        get_pred(0, world_size, data_all, max_length,
                 max_gen, prompt_format, dataset, None, model_name, model2path,
                 out_path, args)
        # 去掉多进程
        # for rank in range(world_size):
        #     # if d_cfg.get('use_multi_gpus', False):
        #     #     p = CustomProcess(target=get_pred,
        #     #                       args=(rank, world_size, data_subsets[rank], max_length,
        #     #                             max_gen, prompt_format, dataset, None, model_name, model2path,
        #     #                             out_path, args),
        #     #                       env_var_key="CUDA_VISIBLE_DEVICES",
        #     #                       env_var_value=f"{rank % torch.cuda.device_count()}")
        #     # else:
        #     #     p = mp.Process(target=get_pred, args=(rank, world_size, data_subsets[rank], max_length,
        #     #                                           max_gen, prompt_format, dataset, None, model_name, model2path,
        #     #                                           out_path, args))
        #
        #     p.start()
        #     processes.append(p)
        # for p in processes:
        #     p.join()
