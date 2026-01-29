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
    def __init__(
        self,
        model_name_or_path: str,
        alpha: float = 1.0,
        device="cuda",
        enable_caching: bool = True,
        enable_recompute: bool = True,
        record_ground_truth: bool = False,
        ground_truth_store: Dict = None,
        fixed_recompute_ratio: float = None,
        # 新增可选参数
        model = None,
        tokenizer = None
    ):
        """
        初始化 Cache-Craft Pipeline。
        允许传入已加载的 model/tokenizer 以避免重复加载。
        
        Args:
            model_name_or_path (str): 模型路径。
            alpha (float): CFO 计算参数。
            device (str): 运行设备。
            enable_caching (bool): 是否启用 KV 缓存存储（读/写）。默认 True。
            enable_recompute (bool): 是否启用命中块的 token 重算。默认 True。
            record_ground_truth (bool): 是否记录当前运行的 KV 为 Ground Truth (用于 nocache 模式)。
            ground_truth_store (Dict): 共享的 Ground Truth 存储字典 (用于跨模式对比)。
            fixed_recompute_ratio (float): 指定固定的重算比例 (0.0-1.0)。若设置，不再使用 CFO 动态计算。
            model (Optional): 预加载的模型实例。
            tokenizer (Optional): 预加载的分词器实例。
        """
        self.device = device
        self.enable_caching = enable_caching
        self.enable_recompute = enable_recompute
        self.record_ground_truth = record_ground_truth
        self.ground_truth_store = ground_truth_store if ground_truth_store is not None else {}
        self.fixed_recompute_ratio = fixed_recompute_ratio
        
        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            # [Fix] Reset Chat Template only if we loaded a fresh tokenizer (or assume caller handled it)
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

        if model:
            print("Using pre-loaded model...")
            self.model = model
        else:
            print(f"Loading model from {model_name_or_path}...")
            # [Updated] Since we upgraded transformers, we can load natively.
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.float16, 
                device_map=device,
                trust_remote_code=True # Add this for safety
            )
            self.model.eval()
            # 应用 Monkey Patch (Apply only if newly loaded? Or check if already applied?)
            # Usually safe to re-apply or assume caller handles it if they pass model.
            # But here let's apply it just in case.
            apply_monkey_patch(self.model)
        
        # If model was passed, we assume Monkey Patch might have been applied before, or we apply it again.
        # Check if already patched to avoid double patching if logic is not idempotent
        # But our monkey patch replaces class methods, so it's idempotent-ish (replaces again).
        if model:
             # Just to be safe, let's re-apply to ensure it's our CacheCraft patch
             apply_monkey_patch(self.model)

        self.store = MetadataStore()
        self.controller = CacheCraftController(alpha=alpha)

    def update_config(self, enable_caching=None, enable_recompute=None, fixed_recompute_ratio=None, record_ground_truth=None):
        """动态更新配置"""
        if enable_caching is not None: self.enable_caching = enable_caching
        if enable_recompute is not None: self.enable_recompute = enable_recompute
        # Allow passing "None" to unset fixed ratio, so check if it's explicitly passed
        self.fixed_recompute_ratio = fixed_recompute_ratio
        if record_ground_truth is not None: self.record_ground_truth = record_ground_truth

    def clear_kv_store(self):
        """清除 KV Cache 存储，确保干净的测试状态"""
        # MetadataStore 默认是内存字典吗？
        # 如果是 Disk based, 需要删除文件。
        # 假设 storage.py 里 MetadataStore 主要是内存或简单的 file wrapper.
        # 我们直接重新初始化它。
        self.store = MetadataStore()
        print("[CacheCraftPipeline] KV Store cleared.")

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
        current_prefix_meta = []

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
            min_len = min(len(full_tokens), len(split_tokens_concat))
            diff_idx = -1
            for i in range(min_len):
                if full_tokens[i] != split_tokens_concat[i]:
                    diff_idx = i
                    break

            context_snippet = ""
            native_slice = []
            split_slice = []
            if diff_idx != -1:
                native_slice = full_tokens[diff_idx:diff_idx+5]
                split_slice = split_tokens_concat[diff_idx:diff_idx+5]
                context_snippet = repr(self.tokenizer.decode(full_tokens[max(0, diff_idx-2):diff_idx+2]))

            raise ValueError(
                "Tokenization mismatch detected. "
                f"Full len={len(full_tokens)}, Split len={len(split_tokens_concat)}, "
                f"diff_idx={diff_idx}, native_slice={native_slice}, split_slice={split_slice}, "
                f"context={context_snippet}"
            )
        else:
            print(f"\n[SUCCESS] Tokenization is consistent (Len: {len(full_tokens)}).")
        # ===============================================
        
        # --- Phase 1: 处理文档块 (命中或未命中通过 Batching 处理) ---
        idx = 0
        total_hit_tokens = 0
        total_recomputed_tokens = 0
        chunk_details = []

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
                print(f"Chunk {idx} [Hit]: {chunk_data['text'][:10]} (Len: {chunk_len})")
                
                total_hit_tokens += chunk_len

                # Variables for logging
                log_cci = "N/A"
                log_beta = "N/A"
                log_recompute_ratio = 0.0
                log_indices = []

                # 若前缀不完全一致，按 CFO 选择部分 token 进行重算
                old_prefix_hashes = chunk_data.get("prefix_hashes", [])
                new_prefix_hashes = current_prefix_hashes
                layer_scores = chunk_data.get("layer_scores", [])
                inter_scores = chunk_data.get("inter_scores_tensor", [])
                prefix_inter_scores = chunk_data.get("prefix_inter_scores", [])

                if self.enable_recompute:
                    recompute_indices = []
                    if layer_scores and inter_scores:
                        # [Optimization] Use pre-calculated CCI if available
                        if "cci" in chunk_data and chunk_data["cci"] is not None:
                             cci = chunk_data["cci"]
                        else:
                             cci = self.controller.calculate_cci(layer_scores)
                        
                        log_cci = f"{cci:.4f}"
                        beta_prime = self.controller.calculate_beta_prime(
                            c_hash, old_prefix_hashes, new_prefix_hashes, prefix_inter_scores
                        )
                        log_beta = f"{beta_prime:.4f}"
                        recompute_indices = self.controller.get_recompute_tokens(
                            chunk_len, cci, beta_prime, inter_scores,
                            fixed_ratio=self.fixed_recompute_ratio
                        )
                    
                    if recompute_indices:
                        num_recomputed = len(recompute_indices)
                        total_recomputed_tokens += num_recomputed
                        log_recompute_ratio = num_recomputed / chunk_len
                        log_indices = recompute_indices
                        print(f"  -> Recomputing {num_recomputed}/{chunk_len} tokens ({num_recomputed/chunk_len*100:.2f}%)")

                        self._recompute_chunk_kv(
                            chunk_text,
                            c_hash,
                            recompute_indices,
                            layer_caches,
                            current_seq_len,
                            chunk_data
                        )
                
                # Record detailed stats
                chunk_details.append({
                    "chunk_idx": idx,
                    "cci": log_cci,
                    "beta_prime": log_beta,
                    "recompute_ratio": f"{log_recompute_ratio*100:.2f}%",
                    "recomputed_tokens": log_indices
                })
                
                # === DEBUG: KV Deviation Analysis (vs Ground Truth) ===
                if self.ground_truth_store:
                    self._check_deviation_against_ground_truth(c_hash, chunk_data)

                # 加载并复用
                self._reuse_chunk_data(layer_caches, chunk_data, current_seq_len, num_layers)
                
                current_seq_len += chunk_len
                current_prefix_hashes.append(c_hash)
                current_prefix_meta.append({"hash": c_hash, "len": chunk_len})
                idx += 1
            else:
                # === Case B: 缓存未命中 (Cache Miss)，对连续未命中进行批处理 ===
                # 向前查找有多少个连续的未命中块
                history_prefix_meta = current_prefix_meta.copy()
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
                    
                    # [Memory Protection] 如果批次过大，强制截断，分批处理
                    # 1536 tokens 约为 1.5K，这是一个保守值，用于防止 OOM
                    # 注意：如果单个 chunk 本身就超过这个值，至少要处理一个
                    MAX_BATCH_TOKEN_LIMIT = 1536
                    if batch_total_len > 0 and (batch_total_len + miss_len) > MAX_BATCH_TOKEN_LIMIT:
                        print(f"[Memory Guard] Stopping batch at {batch_total_len} tokens (Next chunk {miss_len} too large)")
                        break

                    batch_texts.append(miss_text)
                    batch_hashes.append(miss_hash)
                    
                    # 记录相对于本批次 START 的边界
                    # 这告诉 Monkey Patch 在哪里分割每个 chunk
                    prefix_snapshot = current_prefix_hashes.copy()
                    batch_boundaries.append({
                        'hash': miss_hash,
                        'start': batch_total_len,
                        'end': batch_total_len + miss_len,
                        'text': miss_text,
                        'prefix_hashes': prefix_snapshot
                    })
                    
                    batch_total_len += miss_len
                    current_prefix_hashes.append(miss_hash) # Track hash
                    current_prefix_meta.append({"hash": miss_hash, "len": miss_len})
                    idx += 1
                
                print(f"Processing Batch Misses: Chunks {start_idx} to {idx-1} (Total Len: {batch_total_len})")
                
                # 1. 构造整个批次的输入 (Inputs)
                batch_input_ids_list = []
                for _, txt in enumerate(batch_texts):
                    
                    batch_input_ids_list.append(self.tokenizer(txt, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device))
                
                batch_inputs_tensor = torch.cat(batch_input_ids_list, dim=1) # [1, Batch_Len]
                
                # 2. 构造 Past KV
                current_past = self._build_past_key_values(layer_caches)
                
                # 3. 设置捕获上下文 (一个批次，多个边界)
                CURRENT_CONTEXT["mode"] = "capture"
                CURRENT_CONTEXT["chunk_boundaries"] = batch_boundaries
                CURRENT_CONTEXT["captured_data"] = {}
                CURRENT_CONTEXT["history_prefix_meta"] = history_prefix_meta

                # 4. 构造 Position IDs 和 Attention Mask
                position_ids = torch.arange(
                    current_seq_len, current_seq_len + batch_total_len, device=self.device
                ).unsqueeze(0)
                
                past_len = current_past.get_seq_length() if current_past else 0
                if not past_len and isinstance(current_past, tuple): 
                     past_len = current_past[0][0].shape[-2]
                CURRENT_CONTEXT["past_len"] = past_len
                     
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
                    use_cache=True,
                    # --- DEBUG: Strict Numerical Check ---
                    output_hidden_states=True
                )
                # --- DEBUG: Strict Numerical Check ---
                final_hidden_state_s1 = outputs.hidden_states[-1][0, -1, :]
                print(f"[DEBUG Pipeline] Batch Miss Final Token Hidden State - Mean: {final_hidden_state_s1.mean().item():.8f}, Sum: {final_hidden_state_s1.sum().item():.8f}")
                
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
                output_hidden_states=True,
                output_attentions=True
            )
            
            # --- DEBUG: Strict Numerical Check ---
            final_hidden_state = outputs.hidden_states[-1][0, -1, :]
            hs_info = f"Mean: {final_hidden_state.mean().item():.8f}, Sum: {final_hidden_state.sum().item():.8f}"
            logits_info = f"{outputs.logits[0, -1, :5].tolist()}"
            print(f"[DEBUG Pipeline] Final Token Hidden State - {hs_info}")
            print(f"[DEBUG Pipeline] First 5 Logits: {logits_info}")

            # --- DEBUG: 注意力分析 ---
            # outputs.attentions: 包含 [Batch, Heads, Q_Len, KV_Len] 张量的元组
            
            def get_attn_analysis_str(attn_tensor_1d, label):
                # attn_tensor_1d: 形状为 [KV_Len] 的 1D 张量
                lines = [f"\n[{label}] Attention Distribution (Last Query Token):"]
                running_idx = 0
                
                # 检查边界 vs 当前序列长度 (仅 Context)
                for idx, meta in enumerate(current_prefix_meta):
                    c_hash = meta['hash']
                    c_len = meta['len']
                    start = running_idx
                    end = running_idx + c_len
                    
                    if start >= attn_tensor_1d.shape[0]: break
                    
                    chunk_attn = attn_tensor_1d[start:end]
                    total_score = chunk_attn.sum().item()
                    
                    # 获取 Chunk 的文本信息
                    # 修正：即使 disable_caching，chunks 列表依然包含原始文本，优先使用它
                    if idx < len(chunks):
                        c_text = chunks[idx]
                    else:
                        chunk_data = self.store.get_chunk(c_hash)
                        c_text = chunk_data.get('text', "") if chunk_data else ""

                    first_word = c_text.strip().split()[0] if c_text and c_text.strip() else "???"
                    first_word = (first_word[:15] + '..') if len(first_word) > 15 else first_word
                    
                    # 获取关注度最高的 Top-K Token
                    k = min(5, c_len)
                    if k > 0:
                        vals, indices = torch.topk(chunk_attn, k)
                        # 重新分词以将索引映射回字符串
                        c_input_ids = self.tokenizer(c_text, add_special_tokens=False).input_ids
                        top_toks_str = []
                        for i in range(k):
                            local_idx = indices[i].item()
                            val = vals[i].item()
                            if local_idx < len(c_input_ids):
                                t_id = c_input_ids[local_idx]
                                t_str = self.tokenizer.decode([t_id]).strip()
                                t_str = t_str.replace('\n', '\\n')
                                top_toks_str.append(f"'{t_str}'({val:.4f})")
                        
                        tokens_msg = ", ".join(top_toks_str)
                    else:
                        tokens_msg = "None"
                        
                    lines.append(f"  Chunk {idx} ({first_word}): Score={total_score:.4f} | Top Tokens: {tokens_msg}")
                    running_idx += c_len
                return "\n".join(lines)

            # 堆叠所有层的注意力: [Layers, Batch, Heads, Q, K]
            # 取最后一个用户 Token 的注意力: [..., -1, :]
            # 使用循环避免创建巨大的完整张量
            last_token_attns = [l[..., -1, :].detach() for l in outputs.attentions]
            all_layers_tensor = torch.stack(last_token_attns) # [Layers, Batch, Heads, KV_Len]
            
            # 1. 所有层平均: 对层(0)和头(2)求平均 -> [Batch, KV_Len] -> 取第一个样本 [0]
            mean_all = all_layers_tensor.mean(dim=0).mean(dim=1)[0].float().cpu()
            
            # 2. 顶层(最后一层): 对头(1)求平均 -> [Batch, KV_Len] -> 取第一个样本 [0]
            top_layer = all_layers_tensor[-1].mean(dim=1)[0].float().cpu()
            
            attn_analysis_log = get_attn_analysis_str(mean_all, "Mean All Layers") + "\n" + get_attn_analysis_str(top_layer, "Top Layer")
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
        
        # 整理调试输出
        deviation_analysis_str = ""
        if hasattr(self, "_temp_deviation_logs") and self._temp_deviation_logs:
            deviation_analysis_str = "\n[KV Deviation Analysis]\n" + "\n".join(self._temp_deviation_logs)
            self._temp_deviation_logs = []  # Clear for next sample
        
        recompute_stats_str = "N/A"
        if total_hit_tokens > 0:
            ratio = total_recomputed_tokens / total_hit_tokens * 100
            recompute_stats_str = f"Total Hit Tokens: {total_hit_tokens}, Recomputed: {total_recomputed_tokens} ({ratio:.2f}%)"

        debug_output = {
            "hidden_state": hs_info,
            "logits": logits_info,
            "attn_analysis": attn_analysis_log,
            "deviation_analysis": deviation_analysis_str,
            "chunk_details": chunk_details,
            "recompute_stats": recompute_stats_str
        }
        return result_text, debug_output

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
            # [关键修复] 设置 _seen_tokens 以便 HF 4.41+ 正确计算长度。_seen_tokens 是一个计数器，记录了模型到目前为止已经处理了多少个 Token
            if hasattr(cache_obj, "_seen_tokens"):
                cache_obj._seen_tokens = cache_obj.key_cache[0].shape[-2]
            return cache_obj
        except ImportError:
            return tuple(final_past_key_values)

    def _save_captured_data_to_store(self, captured, chunk_boundaries):
        """辅助函数: 将捕获的数据保存到 Storage (和 Ground Truth Store)"""
        
        # 1. 记录 Ground Truth (如果开启)
        # 这通常在 Stage 1 (nocache) 运行
        if self.record_ground_truth:
            for ch_info in chunk_boundaries:
                c_hash = ch_info['hash']
                # 仅保存需要对比的 KV Tensors (Pre-RoPE)
                # 为了节省显存，必须移动到 CPU
                k_list = []
                v_list = []
                
                sorted_layers = sorted(captured.keys())
                valid_capture = False
                for lay_idx in sorted_layers:
                    layer_data = captured[lay_idx].get(c_hash)
                    if layer_data:
                        valid_capture = True
                        k_list.append(layer_data['k_cache'].detach().cpu())
                        v_list.append(layer_data['v_cache'].detach().cpu())
                
                if valid_capture:
                    # 保存到内存字典
                    self.ground_truth_store[c_hash] = {
                        "k": k_list,
                        "v": v_list
                    }

        # 2. 保存到 MetadataStore (如果开启 Cache)
        if not self.enable_caching:
            return

        for ch_info in chunk_boundaries:
            c_hash = ch_info['hash']
            c_text = ch_info['text']
            prefix_hashes = ch_info.get('prefix_hashes', [])
            
            k_cache_list = []
            v_cache_list = []
            layer_scores_list = []
            inter_scores_list = []
            prefix_inter_scores_list = []
            
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
                        prefix_inter_scores_list.append(layer_data.get('prefix_inter_scores', []))
                    else:
                         # 这里的逻辑可以根据是否是首层决定是否需要 padding
                         pass 

            if k_cache_list:
                # print(f"  -> Saving to disk: {c_hash[:8]}")
                
                # [Optimization] Pre-calculate CCI and store it
                cci = None
                if layer_scores_list:
                    cci = self.controller.calculate_cci(layer_scores_list)

                self.store.save_chunk(
                    c_hash, k_cache_list, v_cache_list, 
                    layer_scores_list, inter_scores_list, c_text, prefix_hashes, prefix_inter_scores_list,
                    cci=cci
                )

    def _recompute_chunk_kv(
        self,
        chunk_text: str,
        c_hash: str,
        recompute_indices: List[int],
        layer_caches,
        current_seq_len: int,
        chunk_data: Dict
    ):
        """
        功能说明：
        - 在缓存命中时，对当前块进行一次前向重算，捕获完整的 Pre-RoPE KV，
          然后仅回写 `recompute_indices` 对应位置的 KV 到缓存（self.store）。

        参数说明：
        - chunk_text (str): 当前命中块的原始文本内容，必填。
        - c_hash (str): 当前块的哈希标识，用于定位捕获结果，必填。
        - recompute_indices (List[int]): 需要重算的 token 索引列表（相对当前块），必填。
        - layer_caches (List[List[Tuple[Tensor, Tensor]]]): 已拼接的历史 KV 列表，用于构造 past，必填。
        - current_seq_len (int): 当前全局序列长度（已处理 token 数），用于 position_ids，必填。
        - chunk_data (Dict): 缓存中该块的元数据（含 k_cache/v_cache 等），必填。

        返回值：
        - None。本函数直接就地更新缓存中的 KV。
          注意：会修改 self.store 中对应块的 k_cache/v_cache（原地赋值）。

        可能抛出的异常：
        - ValueError: 若内部模型在 forward 中因输入不一致而报错。
        - 其它来自模型 forward 的运行时异常。
        """
        # TODO: 当前实现是全量重算整个块后再挑选索引回写，后续应改为仅针对指定 token 的局部重算。
        if not recompute_indices:
            # 无需重算，直接返回
            return

        # 过滤越界索引
        chunk_len = chunk_data['k_cache'][0].shape[2]
        recompute_indices = [i for i in recompute_indices if 0 <= i < chunk_len]
        if not recompute_indices:
            # 全部索引无效，直接返回
            return

        # 将当前块文本分词成 input_ids
        input_ids = self.tokenizer(
            chunk_text, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(self.device)

        # 构造 past_key_values（历史上下文）
        current_past = self._build_past_key_values(layer_caches)
        # 为当前块构造 position_ids，起点是全局序列长度
        position_ids = torch.arange(
            current_seq_len, current_seq_len + chunk_len, device=self.device
        ).unsqueeze(0)

        # 计算 past_len，用于 attention_mask 长度对齐
        past_len = current_past.get_seq_length() if current_past else 0
        if not past_len and isinstance(current_past, tuple):
            past_len = current_past[0][0].shape[-2]

        # attention_mask 采用全 1，长度为 past + 当前块
        attention_mask = torch.ones(
            (1, past_len + chunk_len), dtype=torch.long, device=self.device
        )

        # 设置捕获上下文，仅针对该 chunk
        CURRENT_CONTEXT["mode"] = "capture"
        CURRENT_CONTEXT["chunk_boundaries"] = [{
            "hash": c_hash,
            "start": 0,
            "end": chunk_len,
            "text": chunk_text
        }]
        CURRENT_CONTEXT["captured_data"] = {}

        # 运行前向，捕获当前块的 Pre-RoPE KV
        with torch.no_grad():
            _ = self.model(
                input_ids=input_ids,
                past_key_values=current_past,
                position_ids=position_ids,
                attention_mask=attention_mask,
                use_cache=True
            )

        # 读取捕获结果并清理上下文
        captured = CURRENT_CONTEXT["captured_data"]
        CURRENT_CONTEXT["mode"] = "off"
        CURRENT_CONTEXT["captured_data"] = {}

        # 回写指定 token 的 KV（Pre-RoPE）
        idx_tensor = torch.tensor(recompute_indices, dtype=torch.long)
        for layer_idx, layer_data in captured.items():
            new_layer = layer_data.get(c_hash)
            if not new_layer:
                continue
            # 捕获到的新 KV（CPU）
            new_k = new_layer["k_cache"].detach().cpu()
            new_v = new_layer["v_cache"].detach().cpu()

            if layer_idx >= len(chunk_data["k_cache"]):
                continue
            # 旧 KV 指向缓存中的张量
            old_k = chunk_data["k_cache"][layer_idx]
            old_v = chunk_data["v_cache"][layer_idx]

            # 这里实际上修改了self.store中的值
            # 仅对 recompute_indices 对应位置执行原地替换
            old_k[..., idx_tensor, :] = new_k[..., idx_tensor, :]
            old_v[..., idx_tensor, :] = new_v[..., idx_tensor, :]

    def _check_deviation_against_ground_truth(self, c_hash, chunk_data):
        """
        调试功能: 将当前复用的 KV 与 Ground Truth (nocache 运行结果) 对比。
        使用 L2 距离 (RMSE) 并在 CPU 上计算。
        """
        if c_hash not in self.ground_truth_store:
            return

        gt_data = self.ground_truth_store[c_hash]
        gt_k_list = gt_data["k"]
        gt_v_list = gt_data["v"]
        
        # Current data (on GPU usually)
        curr_k_list = chunk_data['k_cache']
        curr_v_list = chunk_data['v_cache']
        
        num_layers = len(curr_k_list)
        if len(gt_k_list) != num_layers:
            # Layer count mismatch? Skip
            return

        total_k_sq_diff = 0.0
        total_v_sq_diff = 0.0
        details = []

        try:
            for l_idx in range(num_layers):
                # Move current to CPU for comparison
                c_k = curr_k_list[l_idx].detach().cpu().float()
                c_v = curr_v_list[l_idx].detach().cpu().float()
                
                g_k = gt_k_list[l_idx].float()
                g_v = gt_v_list[l_idx].float()
                
                # Check shapes
                if c_k.shape != g_k.shape:
                    continue

                # L2 Deviation (RMSE: Root Mean Square Error)
                # diff = sqrt(mean((a-b)^2))
                diff_k = torch.sqrt(torch.pow(c_k - g_k, 2).mean()).item()
                diff_v = torch.sqrt(torch.pow(c_v - g_v, 2).mean()).item()
                
                total_k_sq_diff += diff_k
                total_v_sq_diff += diff_v
                
                # Option: detailed log per layer
                # details.append(f"L{l_idx}:{diff_k:.4f}/{diff_v:.4f}")

            avg_k_rmse = total_k_sq_diff / num_layers
            avg_v_rmse = total_v_sq_diff / num_layers
            
            print(f"  [KV Deviation vs GT] AvgRMSE K: {avg_k_rmse:.6f}, AvgRMSE V: {avg_v_rmse:.6f}")
            
            # Store inputs to be picked up by the main loop logging? 
            # Currently just print to stdout. The run_longbench.py redirects stdout/debug_info.
            # Ideally we put this into a property that generate() returns.
            # But generate() returns (text, debug_info). We can add to debug_info if we persist it.
            # For now, print is captured by user observation, but user asked for output file.
            # We can log to a temporary list and attach to returned debug_info.
            if not hasattr(self, "_temp_deviation_logs"):
                self._temp_deviation_logs = []
            self._temp_deviation_logs.append(f"ChunkHash {c_hash[:8]}: K_RMSE={avg_k_rmse:.6f} V_RMSE={avg_v_rmse:.6f}")

        except Exception as e:
            print(f"  [KV Deviation Check Error] {e}")

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
            # _, k_roped = apply_rotary_pos_emb(k_no_rope, k_no_rope, cos, sin, chunk_positions)
            
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
