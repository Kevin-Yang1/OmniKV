import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaModel,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    LlamaConfig,
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.utils import logging

logger = logging.get_logger(__name__)

class CacheCraftLlamaAttention(LlamaAttention):
    """
    Cache-Craft modified LlamaAttention to support Pre-RoPE Key storage and retrieval.
    """
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        block_metadata_list: Optional[List[dict]] = None, # Added argument
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # --- Cache-Craft Logic: Pre-RoPE Interception ---
        # 此时的 key_states 是 Pre-RoPE 的。
        # 如果需要，可以将这些 Pre-RoPE key 存储下来供后续步骤使用（例如写入磁盘或特殊的 Cache 结构）。
        
        # 正常 Llama 逻辑是先 RoPE 再 update cache。
        # Cache-Craft 要求 Cache 存储 Pre-RoPE。
        # 这里我们假设 past_key_values 如果存在，里面存储的是 Pre-RoPE 的 KV。
        
        kv_seq_len = key_states.shape[-2]
        if past_key_values is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            # update_cache 这里的行为需要特别注意。
            # 如果我们希望 Cache 里存的是 Pre-RoPE，我们应该在 RoPE 之前 update。
            # 但 transformers 的 DynamicCache 等通常只是简单的 append。
            
            # 这里的逻辑修改为：先更新 Cache (存 Pre-RoPE)，再取出来应用 RoPE。
            if use_cache:
                # 注意：这里我们假设 past_key_values.update 会处理拼接逻辑
                # 存入的是原始的 key_states (Pre-RoPE) 和 value_states
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs=kwargs)
                
            kv_seq_len = key_states.shape[-2]

        # --- RoPE Application ---
        # 现在 key_states 包含了过去和现在的 Pre-RoPE keys。
        # 我们需要根据当前的 position_ids 对 *所有* 或 *需要计算的* keys 应用 RoPE。
        # LlamaRotaryEmbedding 计算 cos/sin
        
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        
        # apply_rotary_pos_emb 默认是对 query 和 key 做旋转。
        # query 是当前的 (q_len)，key 是全量的 (kv_seq_len)。
        # 注意：apply_rotary_pos_emb 需要对应的 position_ids。
        # 输入的 position_ids 通常只对应 hidden_states (q_len)。
        # 如果 key_states 是全量的，我们需要全量的 position_ids 吗？
        # 通常 Cached inference 时，position_ids 是当前 token 的位置。
        # 但如果我们是在 "重算" 或者 "RAG Context" 场景，可能需要构造正确的 pos_ids。
        
        # 这里为了保持通用性，如果 seq_len 变长了，说明有 history。
        # 我们假设 apply_rotary_pos_emb 能处理 broadcasting 或者我们需要扩展 position_ids。
        # 在标准 forward 中，position_ids 是 (bsz, q_len)。
        # 如果 key_states 长度 > q_len，说明前面的 keys 已经存在。
        # **关键点**：Cache-Craft 要求重用 Cache 时 "根据当前的新位置重新施加 RoPE"。
        # 这意味着我们不能复用之前计算好的 RoPEd Key，而是必须拿到 Pre-RoPE Key (上面已经拿到)，
        # 然后用当前的 position 重新算一遍。
        
        # 构造全量的 position_ids 
        # Case 1: Prefill / Processing Prompt。position_ids 覆盖整个序列。
        # Case 2: Decoding。position_ids 是当前 step。
        # 如果是 Decoding 且有 Cache (Pre-RoPE)，其实我们需要对 Cache 部分也应用 RoPE 吗？
        # 实际上，如果 Attention 是 `Q * K^T`，K 的位置编码对应它在序列中的绝对位置。
        # 只要 K 的绝对位置(index)没变，它的 RoPE 结果就不变。
        # Cache-Craft 的 "根据新位置" 通常指 retrieved chunk 被放置在 prompt 的不同位置。
        # 这种情况下，我们需要为这部分 Cache 生成新的 position_ids。
        
        # 简化实现：假设 position_ids 已经包含了所有参与 attention 的 token 的正确位置。
        # 如果是 generation step，apply_rotary_pos_emb 内部通常只处理 query 和 key 的后半部分(unsliced)?
        # 不，apply_rotary_pos_emb(q, k, cos, sin) 会对整个传入的 q, k 进行变换。
        # 因此，这里传入全量的 key_states (pre-rope) 和对应的全量 cos/sin，就会得到正确的 Post-RoPE keys。
        
        # 此时需要确保 position_ids 的长度能够覆盖 kv_seq_len，或者 cos/sin 足够长且对齐。
        # 标准 Llama 流程中，Decodig 时 q 只有 1，k 是 cache+new。
        # 这里的 cos, sin 是基于 kv_seq_len 生成的。
        # 默认 apply_rotary_pos_emb 会根据 cos/sin 的维度进行广播。
        # 我们必须显式地让 key_states 加上位置编码。
        
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if self.config.pretraining_tp > 1:
             # Restore time dimension for concat (omitted strictly matching standard impl if needed)
             pass 

        # SDPA or Manual Attention
        # ... (使用 transformers 标准实现，稍作简化)
        
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask
            if attention_mask.size(-1) != query_states.size(-2): # Check dims slightly loosely
                 pass # Adjust logic as needed for broadcasting
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_values


class CacheCraftLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = CacheCraftLlamaAttention(config=config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        block_metadata_list: Optional[List[dict]] = None, # Added argument
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            block_metadata_list=block_metadata_list, # Propagate
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class CacheCraftLlamaModel(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([CacheCraftLlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        block_metadata_list: Optional[List[dict]] = None, # Added argument
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # Need to propagate block_metadata_list to layers
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            block_metadata_list=block_metadata_list, # Passed via kwargs in super if not explicit?
            # super().forward calls layer(..., **kwargs) ?
            # LlamaModel.forward iterates layers. If we inherit, we rely on super().
            # BUT LlamaModel.forward definition in transformers might not pass unknown kwargs to layers.
            # We usually need to rewrite the loop in forward to pass the new arg.
        )
        return outputs
    
    # Overriding forward completely is safer to ensure arguments are passed
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        block_metadata_list: Optional[List[dict]] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is None):
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None: 
            # Initialize a simpler cache if needed, or rely on transformers default
            # For simplicity, passing None starts a new cache
            pass

        # ... (Simplified standard LlamaModel logic to iterate layers)
        
        hidden_states = inputs_embeds
        
        # Retrieve needed vars standard way if not passed
        if position_ids is None:
            position_ids = torch.arange(0, hidden_states.shape[1], dtype=torch.long, device=hidden_states.device)
            position_ids = position_ids.unsqueeze(0)

        # 2D attention mask logic omitted for brevity, assuming calling code handles mask creation or standard implementation 
        
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                # Gradient Checkpointing omitted
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    block_metadata_list=block_metadata_list,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    block_metadata_list=block_metadata_list,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, next_decoder_cache, all_hidden_states, all_self_attns] if v is not None)
            
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class CacheCraftLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = CacheCraftLlamaModel(config)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        block_metadata_list: Optional[List[dict]] = None, # Added argument
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            block_metadata_list=block_metadata_list,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
