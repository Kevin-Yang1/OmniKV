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

    # ====================================================================
    # Phase 0: 模板准备相关方法
    # ====================================================================
    
    def _initialize_generation_state(self):
        """初始化生成状态"""
        num_layers = self.model.config.num_hidden_layers
        return {
            'layer_caches': [[] for _ in range(num_layers)],
            'current_seq_len': 0,
            'current_prefix_hashes': [],
            'current_prefix_meta': [],
            'total_hit_tokens': 0,
            'total_recomputed_tokens': 0,
            'chunk_details': []
        }
    
    def _prepare_prompt_template(self, chunks: List[str], question: str, prompt_template: str = None):
        """
        准备 prompt 模板，应用 chat template，分割前缀和后缀
        
        Returns:
            Tuple[List[str], str]: (更新后的 chunks 列表, suffix_part)
        """
        # 定义 System Prompt
        system_prompt = "You are a helpful assistant. Your job is to answer questions based on the given paragraph. Just provide the answer within 5 words. No need to explain the reasoning or include any other information."
        context_placeholder = "___CONTEXT_PLACEHOLDER___"
        
        # 构造 User Content (强制在 placeholder 后加空格防止 token 合并)
        safe_placeholder = context_placeholder + " "
        
        if prompt_template and "{context}" in prompt_template:
            user_content = prompt_template.replace("{context}", safe_placeholder).replace("{input}", question)
        else:
            user_content = f"Answer the question based on the context.\n\nContext:\n{safe_placeholder}\n\nQuestion: {question}"

        # 应用 Chat Template (如果可用)
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            print("Applying Chat Template to Prefix/Suffix...")
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
            full_prompt_str = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            print("Chat Template not available or not set. Using raw template.")
            full_prompt_str = user_content

        # 分割前缀和后缀
        if context_placeholder in full_prompt_str:
            parts = full_prompt_str.split(context_placeholder)
            prefix_str = parts[0]
            suffix_part = parts[1]  # 包含我们添加的空格
        else:
            prefix_str = ""
            suffix_part = full_prompt_str

        # 将前缀作为第一个 Chunk 加入列表（优化：前缀也可缓存）
        if prefix_str:
            print("Merging Prefix into Chunks stream...")
            chunks.insert(0, prefix_str)
            
        return chunks, suffix_part
    
    def _validate_tokenization(self, chunks: List[str], suffix_part: str):
        """
        验证分词一致性：确保分别 tokenize 各 chunk 后拼接的结果
        与一次性 tokenize 整个文本的结果一致
        """
        full_text_joined = "".join(chunks) + suffix_part
        full_tokens = self.tokenizer(full_text_joined, add_special_tokens=False).input_ids
        
        split_tokens_concat = []
        for ch in chunks:
            split_tokens_concat.extend(self.tokenizer(ch, add_special_tokens=False).input_ids)
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

    @torch.no_grad()
    def generate(self, chunks: List[str], question: str, prompt_template: str = None):
        """
        统一的生成入口 - 协调各个阶段
        
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
             - 使用 CFO 决策是否需要重算部分 token。
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
        
        # Phase 0: 模板准备
        state = self._initialize_generation_state()
        chunks, suffix_part = self._prepare_prompt_template(chunks, question, prompt_template)
        self._validate_tokenization(chunks, suffix_part)
        
        # Phase 1: 块处理
        state = self._process_chunks(chunks, state)
        
        # Phase 2: 答案生成
        result_text, debug_output = self._generate_answer(suffix_part, state, chunks)
        
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

    # ====================================================================
    # Phase 1: 块处理相关方法
    # ====================================================================
    
    def _process_chunks(self, chunks: List[str], state: Dict):
        """
        处理所有文档块（缓存命中/未命中）
        
        Args:
            chunks: 文档块列表
            state: 生成状态字典
            
        Returns:
            Dict: 更新后的状态字典
        """
        idx = 0
        while idx < len(chunks):
            chunk_text = chunks[idx]
            c_hash = self._hash_chunk(chunk_text)
            chunk_data = self._get_chunk_from_cache(c_hash)
            
            if chunk_data:
                # 缓存命中
                idx = self._handle_cache_hit(
                    idx, chunk_text, c_hash, chunk_data, state
                )
            else:
                # 缓存未命中（批处理）
                idx = self._handle_cache_miss_batch(idx, chunks, state)
        
        return state
    
    def _get_chunk_from_cache(self, c_hash: str):
        """从缓存获取块数据"""
        if self.enable_caching:
            return self.store.get_chunk(c_hash)
        return None
    
    def _handle_cache_hit(self, idx: int, chunk_text: str, c_hash: str, 
                         chunk_data: Dict, state: Dict):
        """
        处理单个缓存命中的块
        
        Returns:
            int: 下一个要处理的块索引
        """
        chunk_len = chunk_data['k_cache'][0].shape[2]
        print(f"Chunk {idx} [Hit]: {chunk_data['text'][:10]} (Len: {chunk_len})")
        
        state['total_hit_tokens'] += chunk_len
        
        # 决策是否重算
        recompute_info = self._decide_recompute(
            chunk_data, state['current_prefix_hashes'], c_hash, chunk_len
        )
        
        # 执行重算（如果需要）
        if recompute_info['indices']:
            self._execute_recompute(
                chunk_text, c_hash, chunk_data, 
                recompute_info, state
            )
        
        # 记录详细信息
        recompute_info['log_data']['chunk_idx'] = idx
        state['chunk_details'].append(recompute_info['log_data'])
        
        # KV 偏差检查
        if self.ground_truth_store:
            self._check_deviation_against_ground_truth(c_hash, chunk_data)
        
        # 复用缓存数据
        num_layers = len(state['layer_caches'])
        self._reuse_chunk_data(
            state['layer_caches'], chunk_data, 
            state['current_seq_len'], num_layers
        )
        
        # 更新状态
        state['current_seq_len'] += chunk_len
        state['current_prefix_hashes'].append(c_hash)
        state['current_prefix_meta'].append({"hash": c_hash, "len": chunk_len})
        
        return idx + 1
    
    def _decide_recompute(self, chunk_data: Dict, current_prefix_hashes: List[str], 
                         c_hash: str, chunk_len: int):
        """
        决定是否需要重算以及重算哪些 token
        
        Returns:
            Dict: 包含 'indices' 和 'log_data' 的字典
        """
        log_cci = "N/A"
        log_beta = "N/A"
        log_recompute_ratio = 0.0
        recompute_indices = []
        
        if not self.enable_recompute:
            return {
                'indices': recompute_indices,
                'log_data': {
                    'chunk_idx': None,  # 由 caller 填充
                    'cci': log_cci,
                    'beta_prime': log_beta,
                    'recompute_ratio': f"{log_recompute_ratio*100:.2f}%",
                    'recomputed_tokens': recompute_indices
                }
            }
        
        # CFO 逻辑
        old_prefix_hashes = chunk_data.get("prefix_hashes", [])
        layer_scores = chunk_data.get("layer_scores", [])
        inter_scores = chunk_data.get("inter_scores_tensor", [])
        prefix_inter_scores = chunk_data.get("prefix_inter_scores", [])
        
        if layer_scores and inter_scores:
            # 使用预计算的 CCI（如果可用）
            cci = chunk_data.get("cci") or self.controller.calculate_cci(layer_scores)
            log_cci = f"{cci:.4f}"
            
            beta_prime = self.controller.calculate_beta_prime(
                c_hash, old_prefix_hashes, current_prefix_hashes, prefix_inter_scores
            )
            log_beta = f"{beta_prime:.4f}"
            
            recompute_indices = self.controller.get_recompute_tokens(
                chunk_len, cci, beta_prime, 
                inter_scores, fixed_ratio=self.fixed_recompute_ratio
            )
            
            if recompute_indices:
                log_recompute_ratio = len(recompute_indices) / chunk_len
        
        return {
            'indices': recompute_indices,
            'log_data': {
                'chunk_idx': None,
                'cci': log_cci,
                'beta_prime': log_beta,
                'recompute_ratio': f"{log_recompute_ratio*100:.2f}%",
                'recomputed_tokens': recompute_indices
            }
        }
    
    def _execute_recompute(self, chunk_text: str, c_hash: str, chunk_data: Dict,
                          recompute_info: Dict, state: Dict):
        """执行 token 重算"""
        indices = recompute_info['indices']
        num_recomputed = len(indices)
        chunk_len = chunk_data['k_cache'][0].shape[2]
        
        state['total_recomputed_tokens'] += num_recomputed
        print(f"  -> Recomputing {num_recomputed}/{chunk_len} tokens ({num_recomputed/chunk_len*100:.2f}%)")
        
        self._recompute_chunk_kv(
            chunk_text, c_hash, indices,
            state['layer_caches'], state['current_seq_len'], chunk_data
        )
    
    def _handle_cache_miss_batch(self, start_idx: int, chunks: List[str], state: Dict):
        """
        处理连续的缓存未命中块（批处理）
        
        Returns:
            int: 下一个要处理的块索引
        """
        # 收集批次
        batch_info = self._collect_miss_batch(
            start_idx, chunks, state['current_prefix_hashes'], state['current_prefix_meta']
        )
        
        if not batch_info['texts']:
            return start_idx
        
        print(f"Processing Batch Misses: Chunks {start_idx} to {batch_info['end_idx']-1} "
              f"(Total Len: {batch_info['total_len']})")
        
        # 执行批处理 forward
        self._execute_batch_forward(batch_info, state)
        
        # 更新状态
        state['current_seq_len'] += batch_info['total_len']
        
        return batch_info['end_idx']
    
    def _collect_miss_batch(self, start_idx: int, chunks: List[str], 
                           current_prefix_hashes: List[str], current_prefix_meta: List[Dict]):
        """
        收集连续未命中的块形成批次
        
        Returns:
            Dict: 批次信息字典
        """
        history_prefix_meta = current_prefix_meta.copy()
        batch_texts = []
        batch_hashes = []
        batch_boundaries = []
        batch_total_len = 0
        batch_meta = []
        
        idx = start_idx
        MAX_BATCH_TOKEN_LIMIT = 1536
        
        while idx < len(chunks):
            miss_text = chunks[idx]
            miss_hash = self._hash_chunk(miss_text)
            
            # 如果发现命中的块，停止批处理
            if self.enable_caching and self.store.get_chunk(miss_hash):
                break
            
            # 获取长度
            miss_input_ids = self.tokenizer(miss_text, add_special_tokens=False).input_ids
            miss_len = len(miss_input_ids)
            
            # Memory Protection: 检查批次大小
            if batch_total_len > 0 and (batch_total_len + miss_len) > MAX_BATCH_TOKEN_LIMIT:
                print(f"[Memory Guard] Stopping batch at {batch_total_len} tokens (Next chunk {miss_len} too large)")
                break

            batch_texts.append(miss_text)
            batch_hashes.append(miss_hash)
            
            # 记录边界信息
            prefix_snapshot = current_prefix_hashes.copy()
            batch_boundaries.append({
                'hash': miss_hash,
                'start': batch_total_len,
                'end': batch_total_len + miss_len,
                'text': miss_text,
                'prefix_hashes': prefix_snapshot
            })
            
            batch_total_len += miss_len
            current_prefix_hashes.append(miss_hash)
            batch_meta.append({"hash": miss_hash, "len": miss_len})
            idx += 1
        
        return {
            'texts': batch_texts,
            'hashes': batch_hashes,
            'boundaries': batch_boundaries,
            'total_len': batch_total_len,
            'end_idx': idx,
            'history_prefix_meta': history_prefix_meta,
            'meta': batch_meta
        }
    
    def _execute_batch_forward(self, batch_info: Dict, state: Dict):
        """执行批处理前向传播"""
        # 构造批次输入
        batch_input_ids_list = []
        for txt in batch_info['texts']:
            batch_input_ids_list.append(
                self.tokenizer(txt, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
            )
        
        batch_inputs_tensor = torch.cat(batch_input_ids_list, dim=1)  # [1, Batch_Len]
        
        # 构造 Past KV
        current_past = self._build_past_key_values(state['layer_caches'])
        
        # 设置捕获上下文
        CURRENT_CONTEXT["mode"] = "capture"
        CURRENT_CONTEXT["chunk_boundaries"] = batch_info['boundaries']
        CURRENT_CONTEXT["captured_data"] = {}
        CURRENT_CONTEXT["history_prefix_meta"] = batch_info['history_prefix_meta']

        # 构造 Position IDs 和 Attention Mask
        position_ids = torch.arange(
            state['current_seq_len'], state['current_seq_len'] + batch_info['total_len'], 
            device=self.device
        ).unsqueeze(0)
        
        past_len = current_past.get_seq_length() if current_past else 0
        if not past_len and isinstance(current_past, tuple): 
            past_len = current_past[0][0].shape[-2]
        CURRENT_CONTEXT["past_len"] = past_len
             
        attention_mask = torch.ones(
            (1, past_len + batch_info['total_len']), 
            dtype=torch.long, 
            device=self.device
        )

        # 运行 Forward
        outputs = self.model(
            input_ids=batch_inputs_tensor,
            past_key_values=current_past, 
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True
        )
        
        # DEBUG: Numerical Check
        final_hidden_state_s1 = outputs.hidden_states[-1][0, -1, :]
        print(f"[DEBUG Pipeline] Batch Miss Final Token Hidden State - "
              f"Mean: {final_hidden_state_s1.mean().item():.8f}, "
              f"Sum: {final_hidden_state_s1.sum().item():.8f}")
        
        # 保存捕获的数据
        self._save_captured_data_to_store(CURRENT_CONTEXT["captured_data"], CURRENT_CONTEXT["chunk_boundaries"])
        
        # 更新层缓存
        new_kv = outputs.past_key_values
        self._append_kv_to_layer_caches(state['layer_caches'], new_kv, take_last_tokens=batch_info['total_len'])

        # 清理上下文
        CURRENT_CONTEXT["mode"] = "off"
        CURRENT_CONTEXT["captured_data"] = {}
        
        # 更新 prefix meta（补充到 state）
        state['current_prefix_meta'].extend(batch_info['meta'])

    # ====================================================================
    # Phase 2: 答案生成相关方法
    # ====================================================================
    
    def _generate_answer(self, suffix_part: str, state: Dict, chunks: List[str]):
        """
        生成最终答案
        
        Args:
            suffix_part: Question 部分的文本
            state: 生成状态字典
            chunks: 原始 chunks 列表（用于注意力分析）
            
        Returns:
            Tuple[str, Dict]: (生成的答案文本, 调试信息)
        """
        print("\nGenerating answer...")
        
        # 准备输入
        q_inputs = self.tokenizer(
            suffix_part, return_tensors="pt", add_special_tokens=False
        ).to(self.device)
        
        final_past = self._build_past_key_values(state['layer_caches'])
        
        # 首步 forward
        outputs, attention_analysis = self._forward_question(
            q_inputs, final_past, state['current_seq_len'], 
            state['current_prefix_meta'], chunks
        )
        
        # 解码循环
        result_text = self._decode_loop(outputs)
        
        # 构建调试输出
        debug_output = self._build_debug_output(
            outputs, attention_analysis, state
        )
        
        return result_text, debug_output
    
    def _forward_question(self, q_inputs, cache, past_len: int, 
                         current_prefix_meta: List[Dict], chunks: List[str]):
        """
        Question 的前向传播（包含注意力分析）
        
        Returns:
            Tuple[outputs, attention_analysis_str]: 模型输出和注意力分析字符串
        """
        q_len = q_inputs.input_ids.shape[1]
        
        # 准备 Position IDs 和 Attention Mask
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
                output_hidden_states=True,
                output_attentions=True
            )
            
            # DEBUG: Strict Numerical Check
            final_hidden_state = outputs.hidden_states[-1][0, -1, :]
            hs_info = f"Mean: {final_hidden_state.mean().item():.8f}, Sum: {final_hidden_state.sum().item():.8f}"
            logits_info = f"{outputs.logits[0, -1, :5].tolist()}"
            print(f"[DEBUG Pipeline] Final Token Hidden State - {hs_info}")
            print(f"[DEBUG Pipeline] First 5 Logits: {logits_info}")

            # 注意力分析
            attention_analysis = self._analyze_attention(
                outputs.attentions, current_prefix_meta, chunks
            )
        
        return outputs, attention_analysis
    
    def _analyze_attention(self, attentions, current_prefix_meta: List[Dict], chunks: List[str]):
        """
        分析注意力分布（DEBUG 功能）
        
        Returns:
            str: 格式化的注意力分析字符串
        """
        def get_attn_analysis_str(attn_tensor_1d, label):
            """生成单个注意力分析字符串"""
            lines = [f"\n[{label}] Attention Distribution (Last Query Token):"]
            running_idx = 0
            
            for idx, meta in enumerate(current_prefix_meta):
                c_hash = meta['hash']
                c_len = meta['len']
                start = running_idx
                end = running_idx + c_len
                
                if start >= attn_tensor_1d.shape[0]: 
                    break
                
                chunk_attn = attn_tensor_1d[start:end]
                total_score = chunk_attn.sum().item()
                
                # 获取 Chunk 文本
                if idx < len(chunks):
                    c_text = chunks[idx]
                else:
                    chunk_data = self.store.get_chunk(c_hash)
                    c_text = chunk_data.get('text', "") if chunk_data else ""

                first_word = c_text.strip().split()[0] if c_text and c_text.strip() else "???"
                first_word = (first_word[:15] + '..') if len(first_word) > 15 else first_word
                
                # 获取 Top-K Token
                k = min(5, c_len)
                if k > 0:
                    vals, indices = torch.topk(chunk_attn, k)
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

        # 堆叠所有层的注意力
        last_token_attns = [l[..., -1, :].detach() for l in attentions]
        all_layers_tensor = torch.stack(last_token_attns)  # [Layers, Batch, Heads, KV_Len]
        
        # 1. 所有层平均
        mean_all = all_layers_tensor.mean(dim=0).mean(dim=1)[0].float().cpu()
        
        # 2. 顶层
        top_layer = all_layers_tensor[-1].mean(dim=1)[0].float().cpu()
        
        return get_attn_analysis_str(mean_all, "Mean All Layers") + "\n" + get_attn_analysis_str(top_layer, "Top Layer")
    
    def _decode_loop(self, initial_outputs):
        """
        自回归解码循环
        
        Args:
            initial_outputs: Question forward 的输出
            
        Returns:
            str: 解码后的文本
        """
        generated_tokens = []
        cache = initial_outputs.past_key_values
        
        # 第一个 token
        next_token = torch.argmax(initial_outputs.logits[:, -1, :], dim=-1).unsqueeze(1)
        generated_tokens.append(next_token)
        
        # 解码循环
        for _ in range(50):
            past_len = cache.get_seq_length() if hasattr(cache, 'get_seq_length') else (cache[0][0].shape[-2])
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
    
    def _build_debug_output(self, outputs, attention_analysis: str, state: Dict):
        """
        构建调试输出字典
        
        Returns:
            Dict: 包含各种调试信息的字典
        """
        # Hidden state info
        final_hidden_state = outputs.hidden_states[-1][0, -1, :]
        hs_info = f"Mean: {final_hidden_state.mean().item():.8f}, Sum: {final_hidden_state.sum().item():.8f}"
        logits_info = f"{outputs.logits[0, -1, :5].tolist()}"
        
        # Deviation analysis
        deviation_analysis_str = ""
        if hasattr(self, "_temp_deviation_logs") and self._temp_deviation_logs:
            deviation_analysis_str = "\n[KV Deviation Analysis]\n" + "\n".join(self._temp_deviation_logs)
            self._temp_deviation_logs = []  # Clear for next sample
        
        # Recompute stats
        recompute_stats_str = "N/A"
        if state['total_hit_tokens'] > 0:
            ratio = state['total_recomputed_tokens'] / state['total_hit_tokens'] * 100
            recompute_stats_str = f"Total Hit Tokens: {state['total_hit_tokens']}, Recomputed: {state['total_recomputed_tokens']} ({ratio:.2f}%)"

        return {
            "hidden_state": hs_info,
            "logits": logits_info,
            "attn_analysis": attention_analysis,
            "deviation_analysis": deviation_analysis_str,
            "chunk_details": state['chunk_details'],
            "recompute_stats": recompute_stats_str
        }



    def _rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

if __name__ == "__main__":
    # 简单的测试桩
    pass
