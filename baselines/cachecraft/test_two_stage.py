# CUDA_VISIBLE_DEVICES=1 python baselines/cachecraft/test_two_stage.py --num_samples 10 --output_file /NV1/ykw/projects/OmniKV/baselines/cachecraft/output/hotpot_dense_remix_v1/two_stage_results.txt

import torch
import argparse
import time
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

def format_context_chunk(context_item):
    """格式化上下文块"""
    title = context_item[0]
    sentences = context_item[1]
    text = f"Title: {title}\n" + "".join(sentences) + "\n\n"
    return text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/NV1/ykw/models/Meta-Llama-3.1-8B-Instruct", help="模型路径")
    parser.add_argument("--data_path", type=str, default="datasets/hotpotqa/hotpot_dev_distractor_v1.json", help="数据路径")
    parser.add_argument("--num_samples", type=int, default=1, help="运行样本数量")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_file", type=str, default=None, help="输出文件路径")
    args = parser.parse_args()

    # 加载提示词模板
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "config/dataset2prompt.json")
    prompt_template = None
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            templates = json.load(f)
            prompt_template = templates.get("hotpotqa", None)
    
    if not prompt_template:
        print("错误：无法加载模板。")
        return

    # 加载数据
    with open(args.data_path, 'r') as f:
        data = json.load(f)
    samples = data[:args.num_samples]

    # 加载模型
    print(f"正在从 {args.model_path} 加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # [对齐] 重置聊天模板以确保确定性
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

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map=args.device,
        trust_remote_code=True
    )
    model.eval()

    # 上下文占位符
    context_placeholder = "___CONTEXT_PLACEHOLDER___"

    for i, sample in enumerate(samples):
        print(f"\n{'='*30} 处理样本 {i+1} {'='*30}")
        question = sample['question']

        # 1. 准备完整的上下文字符串
        chunks_text = []
        for ctx in sample['context']:
            chunk_str = format_context_chunk(ctx)
            chunks_text.append(chunk_str)
        full_context = "".join(chunks_text)

        # 2. 构建带有占位符的完整提示词字符串
        # [对齐] 在 context 后强制添加空格以确保分词器边界安全
        safe_placeholder = context_placeholder + " "
        if "{context}" in prompt_template:
            user_content = prompt_template.replace("{context}", safe_placeholder).replace("{input}", question)
        else:
            user_content = f"Answer the question based on the context.\n\nContext:\n{safe_placeholder}\n\nQuestion: {question}"

        # 3. 应用聊天模板以获取完整的字符串结构（前缀 + 占位符 + 后缀）
        system_prompt = "You are a helpful assistant. Your job is to answer questions based on the given paragraph. Just provide the answer within 5 words. No need to explain the reasoning or include any other information."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        full_template_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # 4. 分割为第一阶段（上下文）和第二阶段（问题/后缀）
        # 注意：我们要通过占位符分割，但必须把真正的上下文注入到第一部分
        if context_placeholder in full_template_str:
            parts = full_template_str.split(context_placeholder)
            prefix_part = parts[0]   # System Prompt + Start of User Msg + "Context:\n"
            suffix_part = parts[1]   # " \n\nQuestion: ... Assistant: ..."
            
            # 第一阶段文本 = 前缀 + 真实上下文
            stage1_text = prefix_part + full_context
            # 第二阶段文本 = 后缀（由于 safe_placeholder，这里以空格开头）
            stage2_text = suffix_part
        else:
            print("错误：在模板输出中未找到占位符。")
            continue

        # --- 执行第一阶段：上下文 Prefill ---
        # 对第一阶段分词
        inputs1 = tokenizer(stage1_text, add_special_tokens=False, return_tensors="pt").to(args.device)
        
        # 前向传播获取 KV Cache
        with torch.no_grad():
            outputs1 = model(
                input_ids=inputs1.input_ids,
                use_cache=True,
                output_hidden_states=True # 调试对齐用
            )
        past_key_values = outputs1.past_key_values
        
        # --- 调试：检查第一阶段结束状态 ---
        final_hidden_state_s1 = outputs1.hidden_states[-1][0, -1, :]
        print(f"[DEBUG Two-Stage] 第一阶段结束 Hidden State - Mean: {final_hidden_state_s1.mean().item():.8f}, Sum: {final_hidden_state_s1.sum().item():.8f}")

        # --- 执行第二阶段：问题 Prefill ---
        # 对第二阶段分词
        # 注意：我们不添加特殊 token，依赖于分割后的字符串结构
        inputs2 = tokenizer(stage2_text, add_special_tokens=False, return_tensors="pt").to(args.device)
        
        # 为第二阶段准备 Position IDs
        # 必须从第一阶段的长度继续
        past_len = past_key_values[0][0].shape[2]
        seq_len2 = inputs2.input_ids.shape[1]
        position_ids2 = torch.arange(past_len, past_len + seq_len2, device=args.device).unsqueeze(0)
        attention_mask2 = torch.ones((1, past_len + seq_len2), dtype=torch.long, device=args.device)

        with torch.no_grad():
            outputs2 = model(
                input_ids=inputs2.input_ids,
                past_key_values=past_key_values,
                position_ids=position_ids2,
                attention_mask=attention_mask2,
                use_cache=True,
                output_hidden_states=True
            )
        past_key_values = outputs2.past_key_values

        # --- 调试：检查第二阶段结束状态 (准备生成) ---
        final_hidden_state_s2 = outputs2.hidden_states[-1][0, -1, :]
        print(f"[DEBUG Two-Stage] 最终 Prompt Hidden State - Mean: {final_hidden_state_s2.mean().item():.8f}, Sum: {final_hidden_state_s2.sum().item():.8f}")
        print(f"[DEBUG Two-Stage] 前 5 个 Logits: {outputs2.logits[0, -1, :5].tolist()}")

        # --- 执行第三阶段：解码 (Decoding) ---
        generated_tokens = []
        next_token = torch.argmax(outputs2.logits[:, -1, :], dim=-1).unsqueeze(1)
        generated_tokens.append(next_token)
        
        for _ in range(50):
            past_len = past_key_values[0][0].shape[2]
            position_ids = torch.tensor([[past_len]], device=args.device)
            
            with torch.no_grad():
                outputs = model(
                    input_ids=next_token,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    use_cache=True
                )
            
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(1)
            if next_token.item() == tokenizer.eos_token_id:
                break
            generated_tokens.append(next_token)
            past_key_values = outputs.past_key_values

        final_ids = torch.cat(generated_tokens, dim=1)
        result_text = tokenizer.decode(final_ids[0], skip_special_tokens=True)
        print(f"\n[生成结果]: '{result_text}'")

        if args.output_file:
            with open(args.output_file, "a") as f:
                f.write(f"==================== 处理样本 {i+1} ====================\n")
                f.write(f"Question: {question}\n")
                f.write(f"Generating Answer: {result_text.strip()}\n\n")

if __name__ == "__main__":
    main()
