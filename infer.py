import os

import torch
import json
import time
import transformers
from transformers import (
    AutoTokenizer,
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    set_seed,
)
from typing import Dict, AnyStr
from tiny_tools.read_json import read_config
from tiny_tools.log import logger
from tiny_tools.tensor_tools import dict_to_cuda
from torch.cuda.amp import autocast
from argparse import ArgumentParser
from tiny_tools.tensor_tools import idx_tracer

if transformers.__version__ >= "4.40":
    from transformers import (
        BitsAndBytesConfig,
        GPTQConfig,
    )
    from modeling._old_test import PreCheckLM
    from modeling.token_model import TokenLM, TokenConfig
    from modeling.select_once_model import TokenOnceLM, TokenOnceConfig
    from modeling.offload_select_once import TokenOnceOffloadLM
    from modeling.omnikv import OmniKVMulLM
    from modeling.brutal_offload_llama import BrutalOffloadLM
    from modeling.omnikv_config import LlamaCompressorConfig
    from configs.template_for_chat import get_chat_template
    from baselines.infllm import get_infllm_api

    try:
        # TODO 加入quest
        import sys

        # 为了让里面的import可以找到对应位置
        sys.path.append("baselines/quest")
        from baselines.quest.quest.models import llama as quest_bsl
    except:
        pass
else:
    from baselines.raw_h2o import get_h2o_api

set_seed(42)
# old_prompt = "You are a pirate chatbot who always responds in pirate speak!"
# fixed_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
#
# You are a helpful assistant.<|eot_id|>
# <|start_header_id|>user<|end_header_id|>
#
# {user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
# """
transformers.logging.set_verbosity_error()
if os.environ.get("BE_CAREFUL", False):
    transformers.logging.set_verbosity_debug()


def inference_bs1(
    prompt,
    tkn,
    model,
    generation_config=None,
    use_chat_template=False,
    model_name=None,
    use_fixed_prompt=False,
    use_cot=False,
    return_input_ids=False,
    **kwargs,
):
    st = time.time()
    with torch.no_grad():
        with autocast():
            terminators = [] + [tkn.eos_token_id] if tkn.eos_token_id else []
            for eos_token in ["<|eot_id|>", "<|endoftext|>", "<|im_end|>"]:
                if eos_token in tkn.vocab:
                    terminators += [tkn.convert_tokens_to_ids(eos_token)]
            if use_chat_template:
                template = get_chat_template(model_name, use_cot)
                prompt = template.format(
                    user_message=prompt, system_prompt="You are a helpful assistant."
                )
                # logger.debug(f"prompt is {prompt}")
                input_ids = tkn(prompt, return_tensors="pt")["input_ids"]
            else:
                input_ids = tkn(prompt, return_tensors="pt")["input_ids"]
            if return_input_ids:
                return input_ids
            n = input_ids.shape[1]
            temp = model.generate(
                input_ids.cuda(model.device),
                generation_config=generation_config,
                eos_token_id=terminators,
                **kwargs,
            )[:, n:]
            if os.environ.get("USE_TIMER", False):
                print(f"-------inference_bs1 time {round(time.time() - st, 4)} s")
            return temp


def get_ntk_llama_chat_api_with_tokenizer_bs1(config_path):
    d_config = read_config(config_path)
    model_name = d_config["model_name"]
    model_cls = d_config["model_cls"]
    tkn = AutoTokenizer.from_pretrained(model_name)
    device = 0
    if "qwen" in model_cls:
        raise NotImplementedError
    elif "llama" in model_cls or "262" in model_cls:
        cfg_cls = LlamaCompressorConfig
    else:
        raise ValueError

    cfg = cfg_cls.from_pretrained(model_name)
    if hasattr(cfg, "set_config_of_compressor"):
        cfg.set_config_of_compressor(**d_config)

    use_flash_attn = d_config.get("use_flash_attn", False)
    load_in_8bit = d_config.get("load_in_8bit", False)
    load_in_4bit = d_config.get("load_in_4bit", False)
    quant_config = None
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )
    if load_in_8bit:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)

    args = [model_name]
    kwargs = {
        "config": cfg,
        "quantization_config": quant_config,
        # "pretrained_model_name_or_path": model_name
    }
    if use_flash_attn:
        kwargs["attn_implementation"] = "flash_attention_2"
    if quant_config is None:
        kwargs["torch_dtype"] = torch.float16
    if mul_gpu := getattr(cfg, "use_multi_gpus", False):
        kwargs["device_map"] = "auto"

    if "qwen" in model_cls:
        raise NotImplementedError
    elif "llama" in model_cls or "262" in model_cls:
        model = PreCheckLM.from_pretrained(*args, **kwargs)
    else:
        raise ValueError
    if not load_in_8bit and not load_in_4bit and not mul_gpu:
        model = model.cuda(device)

    if tkn.pad_token is None:
        # 处理一下新加入的pad token，直接设置为全0
        logger.warning("因为llama没有pad token，设置tokenizer.pad_token=[PAD]")
        tkn.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tkn))

    use_cot = cfg.get("cot", False)
    use_chat_template = getattr(cfg, "use_chat_template", False)
    use_fixed_prompt = getattr(cfg, "use_fixed_prompt", True)
    assert use_fixed_prompt
    if use_cot:
        assert use_chat_template

    def chat(prompt, generation_config=None, skip_special_tokens=False, **kwargs):
        st = time.time()
        out_ids = inference_bs1(
            prompt,
            tkn,
            model,
            generation_config,
            use_chat_template=use_chat_template,
            use_fixed_prompt=use_fixed_prompt,
            use_cot=use_cot,
            model_name=model_name,
            **kwargs,
        )
        out = tkn.batch_decode(
            out_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )[0]
        logger.info(f"--- chat time: {round(time.time() - st, 3)}s")
        return out

    return chat, tkn, d_config["max_context_len"], {"eos_token_id": tkn.eos_token_id}


def get_token_select_llama_chat_api_with_tokenizer_bs1(config_path):
    """
    初始化并返回基于 Token 选择（OmniKV）策略的 Llama 推理 API。

    Purpose:
        该函数专门用于加载 OmniKV 系列模型。它会根据配置实例化对应的自定义模型类
        （如动态选择、一次性选择、卸载模式或多级选择），并处理量化配置与 Flash Attention 加速。

    Args:
        config_path (str): OmniKV 实验配置文件的路径 (JSON)。

    Returns:
        tuple: (chat_func, tokenizer, max_context_len, extra_info)
            - chat_func: 封装好的推理函数。
            - tokenizer: 适配 Llama 的分词器。
            - max_context_len: 配置中定义的上下文上限。
            - extra_info: 包含关键元数据的字典。
    """
    # 基础配置解析
    config = read_config(config_path)
    cls = config.get("model_cls", "token")
    model_name = config["model_name"]

    # 配置专属于 Token 选择模型的参数 (继承自 LlamaConfig)
    cfg_cls = TokenConfig
    cfg = read_config(config_path)
    config = cfg_cls.from_pretrained(model_name)
    config.set_config(**cfg)  # 设置OmniKV相关配置

    # 默认设备设置为 GPU 0
    device = 0

    # 检查是否启用 Flash Attention 2
    use_flash_attn = config.get("use_flash_attn", False)
    
    # 准备量化配置 (BitsAndBytes)
    load_in_8bit = config.get("load_in_8bit", False)
    load_in_4bit = config.get("load_in_4bit", False)
    quant_config = None
    if load_in_4bit:
        # 预留 GPTQ 支持接口
        if config.get("use_gptq", False):
            quant_config = GPTQConfig(load_in_4bit)
            raise NotImplementedError
        # 默认使用 BNB 4-bit 量化，计算精度设为 fp16
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )
    if load_in_8bit:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)

    # 封装模型初始化参数
    args = [model_name]
    kwargs = {"config": config, "quantization_config": quant_config}
    if use_flash_attn:
        kwargs["attn_implementation"] = "flash_attention_2"
    if quant_config is None:
        # 非量化模式下，强制使用半精度以节省显存
        kwargs["torch_dtype"] = torch.float16

    # 核心分发：根据 model_cls 实例化具体的 OmniKV 变体类
    if cls == "token":
        # 默认模式：每步动态筛选 Token
        model = TokenLM.from_pretrained(*args, **kwargs).cuda(device)
    elif cls == "token_once":
        # 一次性选择模式：仅在 Prefill 阶段确定保留的 Token
        model = TokenOnceLM.from_pretrained(*args, **kwargs).cuda(device)
    elif cls == "once_offload":
        # 卸载模式：保留重要 Token 到 GPU，其余卸载到内存
        logger.warning("该模式主要用于模型行为分析")
        model = TokenOnceOffloadLM.from_pretrained(*args, **kwargs)
    elif cls == "brutal_offload":
        # 暴力卸载模式：固定策略卸载
        model = BrutalOffloadLM.from_pretrained(*args, **kwargs)
    elif cls == "multi":
        # 多级 OmniKV 模式：支持更复杂的过滤层配置
        model = OmniKVMulLM.from_pretrained(*args, **kwargs)
    else:
        raise ValueError(f"不支持的模型类别: {cls}")

    # 对于非量化模型，确保已完全移动到目标 GPU
    if not load_in_8bit and not load_in_4bit:
        model = model.cuda(device)

    # 初始化分词器
    tkn = AutoTokenizer.from_pretrained(model_name)
    if tkn.pad_token is None:
        # Llama 原生没有 pad token，需手动添加并同步调整模型 Embedding 层大小
        logger.warning("检测到 Llama 缺省 pad token，正自动补齐 [PAD] 标记")
        tkn.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tkn))  

    # 读取任务控制逻辑配置
    use_cot = config.get("cot", False)
    use_chat_template = config.get("use_chat_template", False)
    use_fixed_prompt = config.get("use_fixed_prompt", True)
    
    # 目前主要支持固定 Prompt 模式，以确保实验一致性
    assert use_fixed_prompt
    if use_cot:
        assert use_chat_template

    def chat(prompt, generation_config=None, skip_special_tokens=False, **kwargs):
        """
        内部包装的对话闭包，将 prompt 转化为模型生成。
        """
        st = time.time()
        # 调用通用的批次推理函数 (BS=1)
        tkn_ids = inference_bs1(
            prompt,
            tkn,
            model,
            generation_config,
            use_cot=use_cot,
            model_name=model_name,
            use_chat_template=use_chat_template,
            use_fixed_prompt=use_fixed_prompt,
            **kwargs,
        )
        # 解码 Token IDs 回字符串
        res = tkn.batch_decode(
            tkn_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )[0]
        logger.info(f"--- 单次会话耗时: {round(time.time() - st, 3)}s")
        return res

    # 构造并返回 API 包
    o_dict = {}
    o_dict["eos_token_id"] = config.eos_token_id
    return chat, tkn, config.get("max_context_len", 100_000), o_dict


def get_infllm_chat_api_bs1(config_path):
    from baselines.infllm.inf_llm.utils.greedy_search import GreedySearch

    config = read_config(config_path)
    infllm_cfg_path = config["infllm_cfg_path"]
    model_name = config["model_name"]

    # prepare quantization config
    load_in_8bit = config.get("load_in_8bit", False)
    load_in_4bit = config.get("load_in_4bit", False)
    quant_config = None
    if load_in_4bit:
        if config.get("use_gptq", False):
            quant_config = GPTQConfig(load_in_4bit)
            raise NotImplementedError
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )
    if load_in_8bit:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)

    model, tkn = get_infllm_api.get_model_tokenizer_others(config_path, quant_config)

    if tkn.pad_token is None:
        logger.warning("因为llama没有pad token，设置tokenizer.pad_token=[PAD]")
        tkn.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tkn))

    searcher = GreedySearch(model, tkn)
    use_cot = config.get("cot", False)  # removed
    use_chat_template = config.get("use_chat_template", False)
    use_fixed_prompt = config.get("use_fixed_prompt", True)
    assert use_fixed_prompt
    if use_cot:
        assert use_chat_template

    def chat(prompt, generation_config=None, skip_special_tokens=False, **kwargs):
        st = time.time()
        input_ids = inference_bs1(
            prompt,
            tkn,
            model,
            generation_config,
            use_cot=use_cot,
            model_name=model_name,
            use_chat_template=use_chat_template,
            return_input_ids=True,
            **kwargs,
        )
        extra_end_token_ids = [
            tkn.convert_tokens_to_ids(t)
            for t in ["<|eot_id|>", "<|endoftext|>", "<|im_end|>"]
        ]
        res = searcher.generate(
            input_ids=input_ids, extra_end_token_ids=extra_end_token_ids
        )[0]
        searcher.clear()
        logger.info(f"--- chat time: {round(time.time() - st, 3)}s")
        return res

    o_dict = {"eos_token_id": config.get("eos_token_id", None)}
    return chat, tkn, config.get("max_context_len", 100_000), o_dict


def get_quest_chat_api(cfg_path):
    raise NotImplementedError


def get_any_chat_api(cfg_path):
    """
    根据配置文件加载对应的模型推理接口（Chat API）。

    Purpose:
        该函数是整个推理框架的统一入口。它读取指定的 JSON 配置文件，
        并根据其中的 `model_cls` 字段分发到不同的加载函数，从而支持
        原生模型、H2O、InfLLM 以及本项目核心的 OmniKV (Token Selection) 等多种算法。

    Args:
        cfg_path (str): 推理配置文件（JSON）的绝对或相对路径。

    Returns:
        tuple: (chat_func, tokenizer, max_context_len, extra_info)
            - chat_func: 一个可调用的函数，接收 prompt 并返回生成的字符串。
            - tokenizer: 分词器实例。
            - max_context_len: 该配置支持的最大上下文长度。
            - extra_info: 包含 eos_token_id 等额外信息的字典。
    """
    # 读取 JSON 配置文件
    cfg = read_config(cfg_path)
    
    # 设置 PyTorch 占用的 CPU 线程数，优化 CPU 端预处理性能
    torch.set_num_threads(cfg.get("cpu_num_threads", 12))
    
    # 根据模型类别（model_cls）选择对应的 API 构造器
    model_cls = cfg["model_cls"]
    
    # 逻辑分发：
    if "raw" in model_cls or "262" in model_cls:
        # 加载原生 Llama 或采用线性插值 (NTK-scaling) 的长文本模型
        return get_ntk_llama_chat_api_with_tokenizer_bs1(cfg_path)
    elif "h2o" in model_cls:
        # 加载 H2O 基准算法接口
        return get_h2o_api.get_chat_api(cfg_path)
    elif "quest" in model_cls:
        # Quest 算法支持（目前尚未完全实现）
        raise NotImplementedError
        # return get_quest_chat_api(cfg_path)
    elif "infllm" in model_cls:
        # 加载 InfLLM 基准算法接口
        # raise ValueError("难以适配")
        return get_infllm_chat_api_bs1(cfg_path)
    else:
        # 默认路径：加载基于 Token 选择的 OmniKV 系列模型 (TokenOnce, OmniKVMul 等)
        return get_token_select_llama_chat_api_with_tokenizer_bs1(cfg_path)


if __name__ == "__main__":
    pass
