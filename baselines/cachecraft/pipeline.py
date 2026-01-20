import torch
import hashlib
from typing import List, Tuple, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from baselines.cachecraft.modeling_cachecraft_llama import apply_monkey_patch, CURRENT_CONTEXT
from baselines.cachecraft.storage import MetadataStore
from baselines.cachecraft.controller import CacheCraftController
from baselines.cachecraft.utils import compute_final_cci

class CacheCraftPipeline:
    def __init__(self, model_name_or_path: str, alpha: float = 1.0, device="cuda", enable_caching: bool = True):
        """
        初始化 Cache-Craft Pipeline。
        
        Args:
            model_name_or_path (str): 模型路径。
            alpha (float): CFO 计算参数。
            device (str): 运行设备。
            enable_caching (bool): 是否启用 KV 缓存存储（读/写）。默认 True。
        """
        self.device = device
        self.enable_caching = enable_caching
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        
        # [Fix] Reset Chat Template to remove date injection logic alignment
        # Added explicit bos_token logic since we are overriding the template
        self.tokenizer.chat_template = (
            "{{ bos_token }}"
            "{% for message in messages %}"
            "<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n"
            "{{ message['content'] }}<|eot_id|>"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
            "{% endif %}"
        )
        
        print(f"Loading model from {model_name_or_path}...")
        
        # [Updated] Since we upgraded transformers, we can load natively.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16, 
            device_map=device,
            trust_remote_code=True # Add this for safety
        )
        self.model.eval()
        
        self.store = MetadataStore()
        self.controller = CacheCraftController(alpha=alpha)
        
        # 应用 Monkey Patch
        apply_monkey_patch(self.model)

    def _hash_chunk(self, text: str) -> str:
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    @torch.no_grad()
    def generate(self, chunks: List[str], question: str, prompt_template: str = None):
        """
        统一的生成入口。
        实现完整的 RAG 混合流水线 (Cache-Craft)：
        
        流程说明：
        1. [Phase 0] 解析模板与前缀处理 (Template Parse & Prefix Encoding)
           - 提取并编码 Prompt 前缀，这部分总是需要计算并作为 Cache 的起始部分。
           
        2. [Phase 1] 逐块处理文档 Chunk (Process Chunks)
           - 遍历 RAG 检索到的每个 Chunk。
           - [Cache Check] 计算 Hash 检查是否存在于 Store 中。
           - [Cache Hit] 
             - 直接从存储中加载 Pre-RoPE KV Cache。
             - 根据当前位置 (Seq Len) 施加旋转位置编码 (RoPE)。
             - 拼接到当前 Layer Caches 中 (复用，跳过重算)。
             - (未来可在此处加入 CFO/重算决策逻辑)
           - [Cache Miss] 
             - 构造当前必须的历史 KV Cache。
             - 运行模型 Forward 计算当前 Chunk，同时开启 Capture 模式。
             - Monkey Patch 拦截并保存 Pre-RoPE KV 到存储。
             - 将新计算出的结果更新到 Layer Caches。
             
        3. [Phase 2] 生成答案 (Generate Answer)
           - 拼接 Question 后缀。
           - 构造完整的 past_key_values。
           - 运行生成循环 (Decoding Loop) 输出最终回答。
        """
        print("\n--- Starting Cache-Craft Generation ---")
        
        # 获取模型层数
        num_layers = self.model.config.num_hidden_layers
        
        # 维护每一层的 KV Cache 列表，用于最终拼接
        # 结构: layer_caches[layer_idx] = list of (k_tensor, v_tensor)
        layer_caches = [[] for _ in range(num_layers)] 
        
        current_seq_len = 0
        current_prefix_hashes = []

        # --- Phase 0: 模板解析与前缀编码 (集成 Chat Template) ---
        # 定义 System Prompt (参考 test_original.py 为解决重复输出问题)
        system_prompt = "You are a helpful assistant. Your job is to answer questions based on the given paragraph. Just provide the answer within 5 words. No need to explain the reasoning or include any other information."
        context_placeholder = "___CONTEXT_PLACEHOLDER___"
        
        # 1. 构造 User Content
        # [Fix] 强制在 placeholder 后加一个空格，防止 Tokenizer 将 Chunk 结尾和 Suffix 开头合并 (如 \n\n)
        # 这对于保持 Cache 也就是 Split Tokenization 与 Global Tokenization 一致至关重要
        safe_placeholder = context_placeholder + " "
        
        if prompt_template and "{context}" in prompt_template:
            user_content = prompt_template.replace("{context}", safe_placeholder).replace("{input}", question)
        else:
            user_content = f"Answer the question based on the context.\n\nContext:\n{safe_placeholder}\n\nQuestion: {question}"

        # 2. 应用 Chat Template (如果可用)
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            print("Applying Chat Template to Prefix/Suffix...")
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
            # 这里已经添加了<|begin_of_text|>
            full_prompt_str = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            print("Chat Template not available or not set. Using raw template.")
            full_prompt_str = user_content

        # 3. 分割前缀和后缀
        # 注意: apply_chat_template 可能会保留我们在 user_content 里加的空格
        # 分割时我们使用原始的 context_placeholder (不带空格) 进行 locate
        # 这样 parts[1] (Suffix) 的开头将会包含那个 safe_placeholder 里的空格
        if context_placeholder in full_prompt_str:
            parts = full_prompt_str.split(context_placeholder)
            prefix_str = parts[0]
            # parts[1] starts with " " because we used safe_placeholder above
            suffix_part = parts[1] 
        else:
            prefix_str = ""
            suffix_part = full_prompt_str

        # 4. [优化] 将前缀视为第一个 Chunk 加入列表，参与统一的缓存管理
        # 这样无需每次都重新计算前缀的 KV，而是可以复用
        if prefix_str:
            print("Merging Prefix into Chunks stream...")
            chunks.insert(0, prefix_str)
            
        # === DEBUG: 分词一致性检查 ===
        full_text_joined = "".join(chunks) + suffix_part
        # 模拟真实的 attention_mask/special tokens 行为，这里尽量用最裸的逻辑
        full_tokens = self.tokenizer(full_text_joined, add_special_tokens=False).input_ids
        
        split_tokens_concat = []
        for ch in chunks:
            split_tokens_concat.extend(self.tokenizer(ch, add_special_tokens=False).input_ids)
        # Suffix is also tokenized separately in the pipeline
        split_tokens_concat.extend(self.tokenizer(suffix_part, add_special_tokens=False).input_ids)
        
        if split_tokens_concat != full_tokens:
            print(f"\n[CRITICAL WARNING] Tokenization Mismatch Detected!")
            print(f"Full Text Token Len: {len(full_tokens)}")
            print(f"Split & Concat Len:  {len(split_tokens_concat)}")
            
            min_len = min(len(full_tokens), len(split_tokens_concat))
            diff_idx = -1
            for i in range(min_len):
                if full_tokens[i] != split_tokens_concat[i]:
                    diff_idx = i
                    break
            
            if diff_idx != -1:
                print(f"First mismatch at index {diff_idx}:")
                print(f"  Native: {full_tokens[diff_idx:diff_idx+5]}")
                print(f"  Split:  {split_tokens_concat[diff_idx:diff_idx+5]}")
                print(f"  Context text near mismatch: {repr(self.tokenizer.decode(full_tokens[max(0, diff_idx-2):diff_idx+2]))}")
        else:
            print(f"\n[SUCCESS] Tokenization is consistent (Len: {len(full_tokens)}).")
        # ===============================================
        
        # --- Phase 1: 处理文档块 (命中或未命中通过 Batching 处理) ---
        idx = 0
        while idx < len(chunks):
            chunk_text = chunks[idx]
            c_hash = self._hash_chunk(chunk_text)
            
            # 检查当前块的缓存
            chunk_data = None
            if self.enable_caching:
                chunk_data = self.store.get_chunk(c_hash)
            
            if chunk_data:
                # === Case A: 缓存命中 (Cache Hit) ===
                chunk_len = chunk_data['k_cache'][0].shape[2]
                print(f"Chunk {idx} [Hit]: {c_hash[:8]} (Len: {chunk_len})")
                
                # 加载并复用
                self._reuse_chunk_data(layer_caches, chunk_data, current_seq_len, num_layers)
                
                current_seq_len += chunk_len
                current_prefix_hashes.append(c_hash)
                idx += 1
            else:
                # === Case B: 缓存未命中 (Cache Miss)，对连续未命中进行批处理 ===
                # 向前查找有多少个连续的未命中块
                batch_texts = []
                batch_hashes = []
                batch_boundaries = []
                batch_total_len = 0
                
                start_idx = idx
                while idx < len(chunks):
                    miss_text = chunks[idx]
                    miss_hash = self._hash_chunk(miss_text)
                    
                    # 如果发现命中的块，停止批处理
                    if self.enable_caching and self.store.get_chunk(miss_hash):
                        break
                    
                    # 未命中，加入批处理
                    
                    # [简化逻辑] apply_chat_template 已经负责了特殊 Token 的添加 (比如 <|begin_of_text|>)
                    # 以及后续文本的拼接，我们对所有 Chunk 都一视同仁，不额外添加特殊 Token。
                    use_special_tokens = False
                    
                    # 获取长度作为边界信息
                    miss_input_ids = self.tokenizer(miss_text, add_special_tokens=False).input_ids
                    miss_len = len(miss_input_ids)
                    
                    batch_texts.append(miss_text)
                    batch_hashes.append(miss_hash)
                    
                    # 记录相对于本批次 START 的边界
                    # 这告诉 Monkey Patch 在哪里分割每个 chunk
                    batch_boundaries.append({
                        'hash': miss_hash,
                        'start': batch_total_len,
                        'end': batch_total_len + miss_len,
                        'text': miss_text
                    })
                    
                    batch_total_len += miss_len
                    current_prefix_hashes.append(miss_hash) # Track hash
                    idx += 1
                
                print(f"Processing Batch Misses: Chunks {start_idx} to {idx-1} (Total Len: {batch_total_len})")
                
                # 1. 构造整个批次的输入 (Inputs)
                batch_input_ids_list = []
                for _, txt in enumerate(batch_texts):
                    # 使用我们在 loop 中判定好的 flag
                    batch_input_ids_list.append(self.tokenizer(txt, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device))
                
                batch_inputs_tensor = torch.cat(batch_input_ids_list, dim=1) # [1, Batch_Len]
                
                # 2. 构造 Past KV
                current_past = self._build_past_key_values(layer_caches)
                
                # 3. 设置捕获上下文 (一个批次，多个边界)
                CURRENT_CONTEXT["mode"] = "capture"
                CURRENT_CONTEXT["chunk_boundaries"] = batch_boundaries
                CURRENT_CONTEXT["captured_data"] = {}

                # 4. 构造 Position IDs 和 Attention Mask
                position_ids = torch.arange(
                    current_seq_len, current_seq_len + batch_total_len, device=self.device
                ).unsqueeze(0)
                
                past_len = current_past.get_seq_length() if current_past else 0
                if not past_len and isinstance(current_past, tuple): 
                     past_len = current_past[0][0].shape[-2]
                     
                attention_mask = torch.ones(
                    (1, past_len + batch_total_len), 
                    dtype=torch.long, 
                    device=self.device
                )

                # 5. 运行 Forward (一次性处理所有未命中块)
                outputs = self.model(
                    input_ids=batch_inputs_tensor,
                    past_key_values=current_past, 
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    use_cache=True
                )
                
                # 6. 保存捕获的数据 (内部迭代保存每个块)
                self._save_captured_data_to_store(CURRENT_CONTEXT["captured_data"], CURRENT_CONTEXT["chunk_boundaries"])
                
                # 7. 更新层缓存 (追加整个批次的 KV)
                new_kv = outputs.past_key_values
                self._append_kv_to_layer_caches(layer_caches, new_kv, take_last_tokens=batch_total_len)

                # 清理上下文
                CURRENT_CONTEXT["mode"] = "off"
                CURRENT_CONTEXT["captured_data"] = {}
                
                # 更新全局序列长度
                current_seq_len += batch_total_len

        # --- Phase 2: 生成最终答案 ---
        print("\nGenerating answer...")
        # [关键修正] suffix_part 是拼接在中间的，必须禁止自动添加 BOS (Begin Of Sentence) Token
        q_inputs = self.tokenizer(suffix_part, return_tensors="pt", add_special_tokens=False).to(self.device)
        
        # 构造最终完整的 Cache 对象
        final_past = self._build_past_key_values(layer_caches)
        
        # 生成循环 (Generation Loop)
        generated_tokens = []
        cache = final_past
        
        past_len = current_seq_len # 应该与 Cache 长度一致
        q_len = q_inputs.input_ids.shape[1]
        
        # 准备 Question 的位置 ID 和 Mask
        position_ids = torch.arange(past_len, past_len + q_len, device=self.device).unsqueeze(0)
        attention_mask = torch.ones((1, past_len + q_len), dtype=torch.long, device=self.device)

        # 首步 (Question Forward)
        with torch.no_grad():
            outputs = self.model(
                input_ids=q_inputs.input_ids,
                position_ids=position_ids,
                past_key_values=cache,
                attention_mask=attention_mask,
                use_cache=True,
                # --- DEBUG: Strict Numerical Check ---
                output_hidden_states=True
            )
            
            # --- DEBUG: Strict Numerical Check ---
            final_hidden_state = outputs.hidden_states[-1][0, -1, :]
            print(f"[DEBUG Pipeline] Final Token Hidden State - Mean: {final_hidden_state.mean().item():.8f}, Sum: {final_hidden_state.sum().item():.8f}")
            print(f"[DEBUG Pipeline] First 5 Logits: {outputs.logits[0, -1, :5].tolist()}")
            # -------------------------------------
        
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(1)
        generated_tokens.append(next_token)
        cache = outputs.past_key_values # 更新 Cache
        
        # 解码循环 (Decode Loop)
        for _ in range(50):
             # 获取最新 Cache 长度作为下一个 Position ID
             past_len = cache.get_seq_length() if hasattr(cache, 'get_seq_length') else (past_len + 1)
             position_ids = torch.tensor([[past_len]], device=self.device)
             
             with torch.no_grad():
                 outputs = self.model(
                     input_ids=next_token,
                     position_ids=position_ids,
                     past_key_values=cache,
                     use_cache=True
                 )
             
             next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(1)
             if next_token.item() == self.tokenizer.eos_token_id:
                 break
             generated_tokens.append(next_token)
             cache = outputs.past_key_values

        final_ids = torch.cat(generated_tokens, dim=1)
        result_text = self.tokenizer.decode(final_ids[0], skip_special_tokens=True)
        print(f"Result: {result_text}")
        return result_text

    def _append_kv_to_layer_caches(self, layer_caches, kv_obj, take_last_tokens=None):
        """辅助函数: 解包 HF KV 对象并追加到 layer_caches 列表"""
        num_layers = len(layer_caches)
        for i in range(num_layers):
            if hasattr(kv_obj, "key_cache"): # DynamicCache
                k = kv_obj.key_cache[i]
                v = kv_obj.value_cache[i]
            else: # Tuple (Legacy)
                k, v = kv_obj[i]
            
            if take_last_tokens:
                k = k[..., -take_last_tokens:, :]
                v = v[..., -take_last_tokens:, :]
            
            layer_caches[i].append((k, v))

    def _build_past_key_values(self, layer_caches):
        """辅助函数: 将 layer_caches 列表拼装回 HF 兼容的 Cache 对象"""
        final_past_key_values = []
        for l_idx, layer_ops in enumerate(layer_caches):
            ks = [op[0] for op in layer_ops]
            vs = [op[1] for op in layer_ops]
            if ks:
                layer_k = torch.cat(ks, dim=2)
                layer_v = torch.cat(vs, dim=2)
                final_past_key_values.append((layer_k, layer_v))
            else:
                return None
        
        # 转换为 DynamicCache (如果可用)
        try:
            from transformers import DynamicCache
            cache_obj = DynamicCache()
            cache_obj.key_cache = [item[0] for item in final_past_key_values]
            cache_obj.value_cache = [item[1] for item in final_past_key_values]
            # [关键修复] 设置 _seen_tokens 以便 HF 4.41+ 正确计算长度
            if hasattr(cache_obj, "_seen_tokens"):
                cache_obj._seen_tokens = cache_obj.key_cache[0].shape[-2]
            return cache_obj
        except ImportError:
            return tuple(final_past_key_values)

    def _save_captured_data_to_store(self, captured, chunk_boundaries):
        """辅助函数: 将捕获的数据保存到 Storage"""
        if not self.enable_caching:
            return

        for ch_info in chunk_boundaries:
            c_hash = ch_info['hash']
            c_text = ch_info['text']
            
            k_cache_list = []
            v_cache_list = []
            layer_scores_list = []
            inter_scores_list = []
            
            sorted_layers = sorted(captured.keys())
            for lay_idx in sorted_layers:
                layer_data = captured[lay_idx].get(c_hash)
                if layer_data:
                    k_cache_list.append(layer_data['k_cache'])
                    v_cache_list.append(layer_data['v_cache'])
                    # 处理可能的缺失数据 (如首层可能没有 Scores)
                    if "layer_scores" in layer_data:
                         layer_scores_list.append(layer_data['layer_scores'])
                         inter_scores_list.append(layer_data['inter_scores_tensor'])
                    else:
                         # 这里的逻辑可以根据是否是首层决定是否需要 padding
                         pass 

            if k_cache_list:
                # print(f"  -> Saving to disk: {c_hash[:8]}")
                self.store.save_chunk(
                    c_hash, k_cache_list, v_cache_list, 
                    layer_scores_list, inter_scores_list, c_text
                )

    def _reuse_chunk_data(self, layer_caches, chunk_data, current_seq_len, num_layers):
        """辅助函数: 处理 Cache Hit - 加载, 应用 RoPE, 追加"""
        chunk_len = chunk_data['k_cache'][0].shape[2]
        
        # 为 RoPE 准备当前位置范围
        chunk_positions = torch.arange(
            current_seq_len, current_seq_len + chunk_len, device=self.device
        ).unsqueeze(0)

        for layer_idx in range(num_layers):
            # 从存储获取无 RoPE 的 KV
            k_no_rope = chunk_data['k_cache'][layer_idx].to(self.device).to(torch.float16)
            v_state = chunk_data['v_cache'][layer_idx].to(self.device).to(torch.float16)
            
            # 重新应用 RoPE (基于当前位置)
            layer_module = self.model.model.layers[layer_idx].self_attn
            cos, sin = layer_module.rotary_emb(v_state, chunk_positions)
            k_roped = (k_no_rope * cos) + (self._rotate_half(k_no_rope) * sin)
            
            # 加入当前层缓存
            layer_caches[layer_idx].append((k_roped, v_state))



    def _rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

if __name__ == "__main__":
    # 简单的测试桩
    pass
