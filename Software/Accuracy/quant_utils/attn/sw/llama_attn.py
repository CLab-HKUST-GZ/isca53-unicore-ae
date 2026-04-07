"""
PyTorch LLaMA Attention model from llama: 
https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
"""
import math
import warnings
from typing import List, Optional, Tuple, Union, Callable

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers.cache_utils import Cache, StaticCache
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    # is_flash_attn_greater_or_equal_2_10,
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs

from quant_utils.quant_utils import ActQuantizer, AdaptiveCodebookQuantizer

from quant_utils.attn.utils import SmoothK


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    if module.attnw_quantizer.bits < 16:
        gs = getattr(module.attnw_quantizer, "groupsize", -1)
        need_unpad = False
        if gs and gs > 0:
            L = attn_weights.shape[-1]
            rem = L % gs
            if rem != 0:
                pad = gs - rem
                pad_shape = list(attn_weights.shape)
                pad_shape[-1] = pad
                attn_weights = torch.cat(
                    [attn_weights, torch.zeros(pad_shape, dtype=attn_weights.dtype, device=attn_weights.device)],
                    dim=-1,
                )
                need_unpad = True
        module.attnw_quantizer.find_params(attn_weights)
        attn_weights = module.attnw_quantizer(attn_weights)
        if need_unpad:
            attn_weights = attn_weights[..., :L]
        module.attnw_quantizer.free()
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        
        # Default to standard ActQuantizer
        # Will be replaced with AdaptiveCodebookQuantizer if adaptive flag is set
        self.query_quantizer = ActQuantizer()
        self.key_quantizer = ActQuantizer()
        self.value_quantizer = ActQuantizer()
        self.attnw_quantizer = ActQuantizer()
        
        # Smooth K to reduce per-channel outliers
        # 'per_channel': subtract global mean for each channel (recommended for quantization)
        # 'per_token': subtract mean for each token independently
        self.smooth_K = SmoothK(enabled=True, strategy='per_channel')

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if self.smooth_K:
            key_states = self.smooth_K(key_states)

        # Q/K/V quantization (supports both standard FP quantization and adaptive codebook)
        # Use --q_adaptive, --k_adaptive, --v_adaptive to enable AdaptiveCodebookQuantizer
        if self.query_quantizer.bits < 16:
            self.query_quantizer.find_params(query_states)
            query_states = self.query_quantizer(query_states)
            self.query_quantizer.free()
        if self.key_quantizer.bits < 16:
            self.key_quantizer.find_params(key_states)
            key_states = self.key_quantizer(key_states)
            self.key_quantizer.free()
        if self.value_quantizer.bits < 16:
            # Transpose to make L the last dimension for quantization
            value_states_transposed = value_states.transpose(2, 3)  # [B, H, L, d] -> [B, H, d, L]
            gs = getattr(self.value_quantizer, "groupsize", -1)
            need_unpad = False
            if gs and gs > 0:
                L = value_states_transposed.shape[-1]
                rem = L % gs
                if rem != 0:
                    pad = gs - rem
                    pad_shape = list(value_states_transposed.shape)
                    pad_shape[-1] = pad
                    value_states_transposed = torch.cat(
                        [value_states_transposed, torch.zeros(pad_shape, dtype=value_states_transposed.dtype, device=value_states_transposed.device)],
                        dim=-1,
                    )
                    need_unpad = True
            self.value_quantizer.find_params(value_states_transposed)
            value_states_transposed = self.value_quantizer(value_states_transposed)
            if need_unpad:
                value_states_transposed = value_states_transposed[..., :L]
            value_states = value_states_transposed.transpose(2, 3)
            self.value_quantizer.free()

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        # if self.config._attn_implementation != "eager":
        #     if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
        #         logger.warning_once(
        #             "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
        #             'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
        #         )
        #     else:
        #         attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
