# CUDA_VISIBLE_DEVICES=1 python baselines/cachecraft/test_original.py --output_file baselines/cachecraft/output/hotpot_dense_remix_v1/orig_results.txt

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.omnikv_config import config

import torch
import argparse
import time
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

def format_context_chunk(context_item):
    """
    将 HotpotQA 的上下文条目格式化为文本 Chunk。
    Args:
        context_item: [标题, [句子1, 句子2, ...]]
    """
    title = context_item[0]
    sentences = context_item[1]
    # 格式化:Title + 正文句子拼接
    text = f"Title: {title}\n" + "".join(sentences) + "\n\n"
    return text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=config.get_model_path("Meta-Llama-3.1-8B-Instruct"), help="Path to the model")
    parser.add_argument("--data_path", type=str, default="datasets/2WikiMultihopQA_format/2WiKiMQA_dense_remix_v1.json", help="Data path")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to run")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_file", type=str, default=None, help="Path to output file")
    args = parser.parse_args()

    # 1. 加载提示词模板
    # 尝试读取 config/dataset2prompt.json (相对于当前脚本或已知位置)
    # 假设脚本在 baselines/cachecraft/test_original.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "config/dataset2prompt.json")
    
    prompt_template = None
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            templates = json.load(f)
            prompt_template = templates.get("hotpotqa", None)
            if prompt_template:
                print(f"Loaded prompt template: {prompt_template[:50]}...")
    else:
        print(f"Warning: Config file not found at {config_path}")
        return

    if not prompt_template:
        print("Error: Could not load 'hotpotqa' template.")
        return

    # 2. 加载数据
    print(f"Loading data from {args.data_path}...")
    if not os.path.exists(args.data_path):
         print(f"Error: Data file not found at {args.data_path}")
         return
         
    with open(args.data_path, 'r') as f:
        data = json.load(f)
    
    samples = data[:args.num_samples]
    print(f"Total samples to process: {len(samples)}")

    # 3. 加载模型
    print(f"Loading native model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # [Fix] Reset Chat Template to remove date injection logic
    # This ensures deterministic behavior across dates and environments
    # Added explicit bos_token logic
    tokenizer.chat_template = (
        "{{ bos_token }}"
        "{% for message in messages %}"
        "<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n"
        "{{ message['content'] }}<|eot_id|>"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        "{% endif %}"
    )
    
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map=args.device,
        trust_remote_code=True
    )
    model.eval()

    # 4. 循环处理
    for i, sample in enumerate(samples):
        print(f"\n{'='*20} Processing Sample {i+1} {'='*20}")
        question = sample['question']
        print(f"Question: {question}")
        print(f"Golden Answer: {sample['answer']}")

        # 构造 Context
        chunks_text = []
        for ctx in sample['context']:
            chunk_str = format_context_chunk(ctx)
            chunks_text.append(chunk_str)
        
        full_context = "".join(chunks_text)
        
        # 构造 Prompt
        # [Fix] 强制在 Context 和 Question 之间加一个空格 (align with Pipeline fix)
        # 这对于 Llama 3 的 Tokenizer 边界行为非常敏感
        full_prompt = prompt_template.replace("{context}", full_context + " ").replace("{input}", question)

        # 使用聊天模板以匹配 Llama-3.1-Instruct 的预期输入格式
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Your job is to answer questions based on the given paragraph. Just provide the answer within 5 words. No need to explain the reasoning or include any other information."},
            {"role": "user", "content": full_prompt},
        ]
        chat_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(chat_text, add_special_tokens=False, return_tensors="pt").to(args.device)
        input_len = inputs.input_ids.shape[1]
        
        # --- DEBUG: Strict Numerical Check ---
        # Perform a manual forward pass on the entire prompt to get the exact hidden state
        # before generation. This allows comparison with the Pipeline's state.
        with torch.no_grad():
            debug_outputs = model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                output_hidden_states=True
            )
            # Last token's last hidden state
            final_hidden_state = debug_outputs.hidden_states[-1][0, -1, :] 
            print(f"[DEBUG Original] 最终 Prompt Hidden State - Mean: {final_hidden_state.mean().item():.8f}, Sum: {final_hidden_state.sum().item():.8f}")
            print(f"[DEBUG Original] 前 5 个 Logits: {debug_outputs.logits[0, -1, :5].tolist()}")
        # -------------------------------------

        # print("Generating (Greedy, max_new_tokens=50)...")
        t0 = time.time()
        
        with torch.no_grad():
            output_ids = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=50,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        t1 = time.time()
        
        # 解码
        generated_part = output_ids[0][input_len:]
        generated_text = tokenizer.decode(generated_part, skip_special_tokens=True)
        
        print(f"\n[Generation Result]:")
        print(f"'{generated_text}'")
        print("-" * 40)
        
        # 也可以打印原始 token id 看看是否输出了 eos
        raw_output_text = tokenizer.decode(generated_part, skip_special_tokens=False)
        print(f"[Raw with Special Tokens]: {repr(raw_output_text)}")
        print(f"Tokenized Input Length: {input_len}")

        if args.output_file:
            with open(args.output_file, "a") as f:
                f.write(f"==================== Processing Sample {i+1}/{len(samples)} ====================\n")
                f.write(f"Question: {question}\n")
                f.write(f"Golden Answer: {sample['answer']}\n")
                f.write(f"Generating Answer: {generated_text.strip()}\n\n")

if __name__ == "__main__":
    main()
