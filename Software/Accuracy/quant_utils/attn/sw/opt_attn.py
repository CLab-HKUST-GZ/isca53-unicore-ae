from functools import partial
from typing import Callable, Optional, Tuple, Union

import torch
from torch import nn

from transformers.cache_utils import Cache, StaticCache
from transformers.models.opt.configuration_opt import OPTConfig
# from transformers.models.opt.modeling_opt import (
#     LlamaRotaryEmbedding,
#     # is_flash_attn_greater_or_equal_2_10,
#     apply_rotary_pos_emb,
#     repeat_kv,
# )

from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs

from quant_utils.quant_utils import ActQuantizer, AdaptiveCodebookQuantizer

from quant_utils.attn.utils import SmoothK

class OPTAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: OPTConfig,
        layer_idx: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.dropout = config.attention_dropout
        self.enable_bias = config.enable_bias
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.head_dim = self.embed_dim // self.num_heads
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        
        self.query_quantizer = ActQuantizer()
        self.key_quantizer = ActQuantizer()
        self.value_quantizer = ActQuantizer()
        self.attnw_quantizer = ActQuantizer()
        
        self.smooth_K = SmoothK(enabled=True, strategy='per_channel')

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        # isn't needed in normal attention, but needed in flash attention so to keep the signature same
        position_ids: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""
        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        query_states = query_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)

        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)

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
            # save all key/value_states to cache to be re-used for fast auto-regressive generation
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, {"cache_position": cache_position}
            )

        attn_weights = torch.matmul(query_states, key_states.transpose(3, 2))
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast to fp32 if the weights are in fp16. Please see https://github.com/huggingface/transformers/pull/17437
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        
        if self.attnw_quantizer.bits < 16:
            gs = getattr(self.attnw_quantizer, "groupsize", -1)
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
            self.attnw_quantizer.find_params(attn_weights)
            attn_weights = self.attnw_quantizer(attn_weights)
            if need_unpad:
                attn_weights = attn_weights[..., :L]
            self.attnw_quantizer.free()
        attn_output = torch.matmul(attn_probs, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        if not output_attentions:
            attn_probs = None

        return attn_output, attn_probs
