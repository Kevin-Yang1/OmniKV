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
        实现 RAG 混合流水线：
        1. Parse Template & Prefix
        2. Iterate Chunks:
           - Check Cache
           - If Hit: Load -> Apply RoPE -> Reuse (Placeholder for Recompute)
           - If Miss: Run Model -> Capture -> Save -> Use
        3. Generate Answer
        """
        print("\n--- Starting Cache-Craft Generation ---")
        
        # 获取模型层数
        num_layers = self.model.config.num_hidden_layers
        
        # 维护每一层的 KV Cache 列表，用于最终拼接
        # Structure: layer_caches[layer_idx] = list of (k_tensor, v_tensor)
        layer_caches = [[] for _ in range(num_layers)] 
        
        current_seq_len = 0
        current_prefix_hashes = []

        # --- Phase 0: Template Parsing & Prefix Encoding ---
        suffix_part = question # Default fallback
        if prompt_template and "{context}" in prompt_template:
            parts = prompt_template.split("{context}")
            prefix_str = parts[0]
            suffix_template = parts[1]
            
            # Encode Prefix (Always Compute / Or we could cache prefix too, but assuming short)
            if prefix_str:
                print("Encoding Prefix instructions...")
                prefix_inputs = self.tokenizer(prefix_str, return_tensors="pt").to(self.device)
                # Run forward to get KV
                prefix_outputs = self.model(prefix_inputs.input_ids, use_cache=True)
                
                # Extract KV
                prefix_kv = prefix_outputs.past_key_values
                self._append_kv_to_layer_caches(layer_caches, prefix_kv)
                
                current_seq_len += prefix_inputs.input_ids.shape[1]
                print(f"Prefix encoded. Length: {current_seq_len}")

            # Prepare Suffix Template
            suffix_part = suffix_template.replace("{input}", question)
        else:
            print("No template used or invalid template format. Using question as suffix.")
        
        # --- Phase 1: Process Chunks (Hit or Miss) ---
        for i, chunk_text in enumerate(chunks):
            c_hash = self._hash_chunk(chunk_text)
            chunk_inputs = self.tokenizer(chunk_text, add_special_tokens=False, return_tensors="pt").to(self.device)
            chunk_len = chunk_inputs.input_ids.shape[1]
            
            # Check Store
            chunk_data = self.store.get_chunk(c_hash)
            
            if chunk_data:
                # === Cache Hit ===
                print(f"Chunk {i} [Hit]: {c_hash[:8]} (Len: {chunk_len})")
                
                # [Placeholder] CFO & Recompute Decision
                # recompute_indices = self.controller.get_recompute_tokens(...)
                # For now, we skip recompute logic and fully reuse
                
                # Load & Apply RoPE (Stitching)
                self._reuse_chunk_data(layer_caches, chunk_data, current_seq_len, num_layers)
                
            else:
                # === Cache Miss ===
                print(f"Chunk {i} [Miss]: {c_hash[:8]} (Len: {chunk_len}) -> Computing & Capturing")
                
                # 1. Construct current past_key_values from layer_caches
                current_past = self._build_past_key_values(layer_caches)
                
                # 2. Setup Capture Context
                # We want to capture the PRE-ROPE KV of this chunk
                CURRENT_CONTEXT["mode"] = "capture"
                CURRENT_CONTEXT["chunk_boundaries"] = [{
                    'hash': c_hash,
                    'start': 0, # Relative to the NEW input_ids
                    'end': chunk_len,
                    'text': chunk_text
                }]
                CURRENT_CONTEXT["captured_data"] = {} # Reset capture buffer (per layer)

                # 3. Run Forward (Compute)
                # Input: just the new chunk tokens
                # Position IDs: need to align with current_seq_len
                position_ids = torch.arange(
                    current_seq_len, current_seq_len + chunk_len, device=self.device
                ).unsqueeze(0)
                
                # Attention Mask Handling
                # When passing past_key_values, HF expects attention_mask to cover both past and new tokens
                # Shape should be [1, past_len + new_len]
                past_len = current_past.get_seq_length() if current_past else 0
                if not past_len and isinstance(current_past, tuple): # Fallback
                     past_len = current_past[0][0].shape[-2]
                     
                attention_mask = torch.ones(
                    (1, past_len + chunk_len), 
                    dtype=torch.long, 
                    device=self.device
                )

                outputs = self.model(
                    input_ids=chunk_inputs.input_ids,
                    past_key_values=current_past, 
                    position_ids=position_ids,
                    attention_mask=attention_mask, # ADDED THIS
                    use_cache=True
                )
                
                # 4. Save Captured Data (Pre-RoPE) to Store
                self._save_captured_data_to_store(CURRENT_CONTEXT["captured_data"], CURRENT_CONTEXT["chunk_boundaries"])
                
                # 5. Update layer_caches with the NEW Post-RoPE KV returned by model
                # Model returns full past_key_values (old + new) or incremental depending on implementation?
                # Usually HF returns the updated full/incremental cache object.
                # Since we passed `current_past`, the output `outputs.past_key_values` contains everything.
                # BUT, to keep our `layer_caches` clean (list of chunks), we prefer to extract JUST the new part?
                # Or easier: just perform the same extraction logic on the output KV, 
                # taking the last `chunk_len` tokens.
                
                new_kv = outputs.past_key_values
                self._append_kv_to_layer_caches(layer_caches, new_kv, take_last_tokens=chunk_len)

                # Clean Context
                CURRENT_CONTEXT["mode"] = "off"
                CURRENT_CONTEXT["captured_data"] = {}
                
            # Advance Seq Len
            current_seq_len += chunk_len
            current_prefix_hashes.append(c_hash)

        # --- Phase 2: Generate Answer ---
        print("\nGenerating answer...")
        q_inputs = self.tokenizer(suffix_part, return_tensors="pt").to(self.device)
        
        # Build final complete cache
        final_past = self._build_past_key_values(layer_caches)
        
        # Generation Loop
        generated_tokens = []
        cache = final_past
        
        past_len = current_seq_len # Should match cache length
        q_len = q_inputs.input_ids.shape[1]
        
        position_ids = torch.arange(past_len, past_len + q_len, device=self.device).unsqueeze(0)
        attention_mask = torch.ones((1, past_len + q_len), dtype=torch.long, device=self.device)

        # First step (Question)
        with torch.no_grad():
            outputs = self.model(
                input_ids=q_inputs.input_ids,
                position_ids=position_ids,
                past_key_values=cache,
                attention_mask=attention_mask,
                use_cache=True
            )
        
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(1)
        generated_tokens.append(next_token)
        cache = outputs.past_key_values # Update cache
        
        # Decode loop
        for _ in range(50):
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
        """Helper to unpack HF KV object and append to our layer_caches list"""
        num_layers = len(layer_caches)
        for i in range(num_layers):
            if hasattr(kv_obj, "key_cache"): # DynamicCache
                k = kv_obj.key_cache[i]
                v = kv_obj.value_cache[i]
            else: # Tuple
                k, v = kv_obj[i]
            
            if take_last_tokens:
                k = k[..., -take_last_tokens:, :]
                v = v[..., -take_last_tokens:, :]
            
            layer_caches[i].append((k, v))

    def _build_past_key_values(self, layer_caches):
        """Helper to stitch layer_caches into a HF compatible cache object"""
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
        
        # Convert to DynamicCache
        try:
            from transformers import DynamicCache
            cache_obj = DynamicCache()
            cache_obj.key_cache = [item[0] for item in final_past_key_values]
            cache_obj.value_cache = [item[1] for item in final_past_key_values]
            if hasattr(cache_obj, "_seen_tokens"):
                cache_obj._seen_tokens = cache_obj.key_cache[0].shape[-2]
            return cache_obj
        except ImportError:
            return tuple(final_past_key_values)

    def _save_captured_data_to_store(self, captured, chunk_boundaries):
        """Helper to save captured data to storage"""
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
                    # Default placeholders if missing (e.g. first layer might not have scores)
                    if "layer_scores" in layer_data:
                         layer_scores_list.append(layer_data['layer_scores'])
                         inter_scores_list.append(layer_data['inter_scores_tensor'])
                    else:
                         # first layer or error, append dummy
                         pass 

            if k_cache_list:
                print(f"  -> Saving to disk: {c_hash[:8]}")
                # Note: Assuming scores are captured. If not (first token?), handle gracefully.
                # For simplicity in this demo, we save what we have.
                self.store.save_chunk(
                    c_hash, k_cache_list, v_cache_list, 
                    layer_scores_list, inter_scores_list, c_text
                )

    def _reuse_chunk_data(self, layer_caches, chunk_data, current_seq_len, num_layers):
        """Helper to process Cache Hit: Load from store, Apply RoPE, Append"""
        chunk_len = chunk_data['k_cache'][0].shape[2]
        
        # Position Range for RoPE
        chunk_positions = torch.arange(
            current_seq_len, current_seq_len + chunk_len, device=self.device
        ).unsqueeze(0)

        for layer_idx in range(num_layers):
            # No-RoPE KV from Store
            k_no_rope = chunk_data['k_cache'][layer_idx].to(self.device).to(torch.float16)
            v_state = chunk_data['v_cache'][layer_idx].to(self.device).to(torch.float16)
            
            # Apply RoPE
            layer_module = self.model.model.layers[layer_idx].self_attn
            cos, sin = layer_module.rotary_emb(v_state, chunk_positions)
            k_roped = (k_no_rope * cos) + (self._rotate_half(k_no_rope) * sin)
            
            layer_caches[layer_idx].append((k_roped, v_state))



    def _rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

if __name__ == "__main__":
    # 简单的测试桩
    pass
