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
    def __init__(self, model_name_or_path: str, alpha: float = 1.0, device="cuda"):
        """
        初始化 Cache-Craft Pipeline。
        
        Args:
            model_name_or_path (str): 模型路径。
            alpha (float): CFO 计算参数。
            device (str): 运行设备。
        """
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        
        print(f"Loading model from {model_name_or_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            torch_dtype=torch.float16, 
            device_map=device
        )
        self.model.eval()
        
        self.store = MetadataStore()
        self.controller = CacheCraftController(alpha=alpha)
        
        # 应用 Monkey Patch
        apply_monkey_patch(self.model)

    def _hash_chunk(self, text: str) -> str:
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    @torch.no_grad()
    def prefill_and_capture(self, chunks: List[str], question: str):
        """
        Offline/Online Profiling 阶段。
        运行一次完整的 Forward，捕获每个 Chunk 的 KV Cache 和 Attention Profiling 数据。
        """
        print("\n--- Starting Prefill & Capture ---")
        
        # 1. 构造 Prompt 并计算 Token 边界
        input_text = ""
        chunk_boundaries = []
        current_token_offset = 0
        
        for chunk_text in chunks:
            # 简单拼接，实际应用可能需要分隔符
            input_text += chunk_text
            
            # 计算 token 数量 (简化版，实际中可能需要精确对齐)
            # 注意：这种简单相加的方式在此时仅做演示，精确边界需要对 full_ids 进行映射
            # 更好的做法是对每个 chunk 单独 tokenize 这里的 offset 计算仅供参考
            # 实际操作：
            chunk_tokens = self.tokenizer(chunk_text, add_special_tokens=False).input_ids
            length = len(chunk_tokens)
            
            c_hash = self._hash_chunk(chunk_text)
            chunk_boundaries.append({
                'hash': c_hash,
                'start': current_token_offset,
                'end': current_token_offset + length,
                'text': chunk_text
            })
            current_token_offset += length
            
        # 加上 Question
        input_text += question
        
        # 2. 准备上下文
        # 实际全量 Tokenize
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        
        # 校准边界 (因为 tokenizer merge 可能导致某些边界 token 合并，这里为了演示简化处理，假设边界是对齐的)
        # 如果需要严格对齐，需要更复杂的逻辑。
        
        # 设置全局 Context
        CURRENT_CONTEXT["mode"] = "capture"
        CURRENT_CONTEXT["chunk_boundaries"] = chunk_boundaries
        CURRENT_CONTEXT["captured_data"] = {} # Clear previous data
        
        # 3. 运行 Model Forward
        print(f"Forward pass with input length: {input_ids.shape[1]}")
        self.model(input_ids)
        
        # 4. 提取数据并保存
        captured = CURRENT_CONTEXT["captured_data"]
        
        for ch_info in chunk_boundaries:
            c_hash = ch_info['hash']
            c_text = ch_info['text']
            
            # 收集该 Chunk 所有层的数据
            k_cache_list = []
            v_cache_list = []
            layer_scores_list = []
            inter_scores_list = []
            
            # 遍历层 (根据 captured 中的 key)
            sorted_layers = sorted(captured.keys())
            for lay_idx in sorted_layers:
                layer_data = captured[lay_idx].get(c_hash)
                if layer_data:
                    k_cache_list.append(layer_data['k_cache'])
                    v_cache_list.append(layer_data['v_cache'])
                    layer_scores_list.append(layer_data['layer_scores'])
                    inter_scores_list.append(layer_data['inter_scores_tensor'])
            
            if k_cache_list:
                print(f"Saving chunk {c_hash[:8]}... (Layers: {len(k_cache_list)})")
                self.store.save_chunk(
                    c_hash,
                    k_cache_list,
                    v_cache_list,
                    layer_scores_list,
                    inter_scores_list,
                    c_text
                )
            else:
                print(f"Warning: No data captured for chunk {c_hash[:8]}")

        # 清理 Context
        CURRENT_CONTEXT["mode"] = "off"
        CURRENT_CONTEXT["captured_data"] = {}
        print("Capture complete.")

    @torch.no_grad()
    def inference_with_cache(self, new_chunks: List[str], question: str):
        """
        在线推理阶段。
        利用 Cache-Craft 逻辑：检查 Cache -> 计算 CFO -> 决策 -> 拼装 Cache。
        """
        print("\n--- Starting Inference with Cache ---")
        
        # 模拟生成 Past Key Values
        # List of tuple(key, value) for each layer
        past_key_values = [] 
        
        # 获取模型层数
        num_layers = self.model.config.num_hidden_layers
        num_heads = self.model.config.num_attention_heads
        head_dim = self.model.config.hidden_size // num_heads
        
        # 用于构建 layer-wise 的 cache 列表
        # structure: [[k_l0, v_l0], [k_l1, v_l1], ...] (will convert to tuple later)
        layer_caches = [[] for _ in range(num_layers)] 
        
        current_prefix_hashes = []
        current_seq_len = 0
        
        # 1. 遍历 New Chunks (RAG Retrieved Results)
        for chunk_text in new_chunks:
            c_hash = self._hash_chunk(chunk_text)
            
            # --- Cache Hit Check ---
            chunk_data = self.store.get_chunk(c_hash)
            
            if chunk_data:
                # Cache Hit!
                print(f"Cache Hit: {c_hash[:8]}")
                
                # --- Decision Making ---
                # 计算 CCI
                layer_scores = chunk_data['layer_scores']
                cci = self.controller.calculate_cci(layer_scores)
                
                # 计算 Beta' (Overlap)
                # 简化逻辑：这里假设 old_prefix 就是该 chunk 存储时记录的前缀
                # 但我们在 storage 里没存 old_prefix (论文里建议存)。
                # 作为 Baseline 简化：假设 old_prefix 为空或者我们只做增量判断
                # 暂时用一个 dummy old_prefix 来模拟 beta calculation (e.g., 假设上次是在同样位置)
                # 实际实现应在 save_chunk 时保存 prefix_hashes
                beta_prime = 1.0 # 假设上下文完全匹配以跳过重算，或者设为0.5测试
                
                # 计算 CFO & Recompute Tokens
                chunk_len = chunk_data['k_cache'][0].shape[2] # [BS, Head, Seq, Dim]
                inter_scores = chunk_data['inter_scores_tensor']
                
                recompute_indices = self.controller.get_recompute_tokens(
                    chunk_len, cci, beta_prime, inter_scores
                )
                
                print(f"  CCI: {cci:.4f}, Beta': {beta_prime:.2f}")
                print(f"  Decision: Recompute {len(recompute_indices)} / {chunk_len} tokens")
                
                # --- Cache Stitching (No-RoPE -> RoPE) ---
                # 我们需要把 No-RoPE 的 Key 拿出来，根据当前的新位置 (current_seq_len) 施加 RoPE
                
                for layer_idx in range(num_layers):
                    # 获取 No-RoPE Key 和 Value (CPU -> GPU)
                    k_no_rope = chunk_data['k_cache'][layer_idx].to(self.device).to(torch.float16)
                    v_state = chunk_data['v_cache'][layer_idx].to(self.device).to(torch.float16)
                    
                    # 准备 RoPE
                    # 我们需要为这部分 cache 生成 position_ids
                    # Range: [current_seq_len, current_seq_len + chunk_len)
                    chunk_positions = torch.arange(
                        current_seq_len, 
                        current_seq_len + chunk_len, 
                        device=self.device
                    ).unsqueeze(0) # [1, Chunk_Len]
                    
                    # 获取该层的 Attention Module 以调用 rotary_emb
                    # 假设 model.layers 是 iterable
                    layer_module = self.model.model.layers[layer_idx].self_attn
                    
                    # 生成 cos, sin
                    # rotary_emb 需要输入 value_states 来推断 dtype 和 device，seq_len 是总长
                    # 这里 seq_len 参数通常用于切片 cache，我们传 max_len 或者当前需要的长度
                    cos, sin = layer_module.rotary_emb(v_state, seq_len=current_seq_len + chunk_len)
                    
                    # 应用 RoPE
                    # query 传 None? No, apply_rotary_pos_emb expects query and key.
                    # 我们只想旋转 Key。我们可以传一个 dummy query 或者只计算 Key 部分的逻辑。
                    # transformers 的 apply_rotary_pos_emb 同时处理 q, k。
                    # 如果我们只处理 k，可以构造一个 dummy q。
                    dummy_q = torch.empty(1, 1, chunk_len, head_dim, device=self.device)
                    
                    # 注意：apply_rotary_pos_emb 实现可能依赖 slicing。
                    # 调用 transformers 的 def apply_rotary_pos_emb(q, k, cos, sin, position_ids)
                    # cos, sin 应该是对应位置的。
                    # 标准实现里 cos, sin 是全量的 [Max_Len, Dim]。apply 内部会根据 position_ids 索引吗？
                    # LlamaRotaryEmbedding 的 forward 返回的 cos, sin 是切好或者 cache 好的。
                    # 在 LlamaAttention.forward 里: cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
                    # 这里的 seq_len 决定了返回多长的 cos/sin。
                    # 我们需要的是对应 [current_seq_len : current_seq_len+chunk_len] 这段的 cos/sin。
                    
                    # 修正 RoPE 逻辑：
                    # 1. 获取全量 cos, sin 直到当前末尾
                    full_cos, full_sin = layer_module.rotary_emb(v_state, seq_len=current_seq_len + chunk_len)
                    # 2. 我们只需要当前 chunk 对应的 cos/sin
                    # cos shape: [1, 1, Seq_Len, Dim]
                    chunk_cos = full_cos[:, :, current_seq_len : current_seq_len + chunk_len, :]
                    chunk_sin = full_sin[:, :, current_seq_len : current_seq_len + chunk_len, :]
                    
                    # 3. 对 k_no_rope 应用旋转
                    # 手动实现旋转 (参考 apply_rotary_pos_emb 源码，避免 dummy q 的麻烦)
                    # output = (k * cos) + (rotate_half(k) * sin)
                    from transformers.models.llama.modeling_llama import rotate_half
                    k_roped = (k_no_rope * chunk_cos) + (rotate_half(k_no_rope) * chunk_sin)
                    
                    # 将处理好的 KV 加入当前层的列表
                    # 稍后拼接
                    layer_caches[layer_idx].append((k_roped, v_state))
                    
                current_seq_len += chunk_len
                current_prefix_hashes.append(c_hash)
            else:
                print(f"Cache Miss: {c_hash[:8]}")
                # 真实场景需要运行模型计算该 chunk 的 KV
                # 这里作为 Baseline 代码生成，暂时跳过或者报错
                raise NotImplementedError("Online calculation for cache miss not implemented in this demo.")

        # 2. 组装最终的 past_key_values
        final_past_key_values = []
        for l_idx in range(num_layers):
            # layer_caches[l_idx] is list of (k, v) tuples
            # Concat them along seq_len dimension (dim 2)
            ks = [item[0] for item in layer_caches[l_idx]]
            vs = [item[1] for item in layer_caches[l_idx]]
            
            if ks:
                layer_k = torch.cat(ks, dim=2)
                layer_v = torch.cat(vs, dim=2)
                final_past_key_values.append((layer_k, layer_v))
            else:
                # No cache (only question?)
                final_past_key_values = None
                break
                
        # 3. 处理 Question 并生成
        print("Generating answer...")
        q_inputs = self.tokenizer(question, return_tensors="pt").to(self.device)
        
        # 此时的 position_ids 需要接在 current_seq_len 之后
        q_len = q_inputs.input_ids.shape[1]
        
        # 运行生成
        # 传递 past_key_values
        # 注意：transformers 的 generate 会自动处理 position_ids 如果有了 past
        outputs = self.model.generate(
            input_ids=q_inputs.input_ids,
            past_key_values=final_past_key_values, # 传入我们精心拼接的 Cache
            max_new_tokens=50,
            do_sample=False
        )
        
        result_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Result: {result_text}")
        return result_text

if __name__ == "__main__":
    # 简单的测试桩
    pass
