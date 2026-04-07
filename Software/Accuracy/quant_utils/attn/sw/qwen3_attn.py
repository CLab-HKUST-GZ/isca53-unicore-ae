from functools import partial
from typing import Callable, Optional, Tuple, Union

import torch
from torch import nn

from transformers.cache_utils import Cache, StaticCache
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RotaryEmbedding,
    Qwen3RMSNorm,
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


class Qwen3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3Config, layer_idx: int):
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
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape
        self.sliding_window = config.sliding_window
        if not (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            self.sliding_window = None

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

        # Compute Q/K before norm
        query_states_before_norm = self.q_proj(hidden_states).view(hidden_shape)
        key_states_before_norm = self.k_proj(hidden_states).view(hidden_shape)
        
        # Apply QK norm
        query_states = self.q_norm(query_states_before_norm).transpose(1, 2)
        key_states = self.k_norm(key_states_before_norm).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # Statistics after RoPE: sample 1% to reduce overhead
        do_stats = torch.rand(1).item() < 0.01
        do_stats = False
        if do_stats:
            # Per-token per-head statistics for adaptive quantization
            # Q/K are already in [B, H, L, d] format
            
            # Group-wise crest factor analysis
            # Note: GQA models have different head counts for Q and K/V
            Bq, Hq, Lq, dq = query_states.shape
            Bk, Hk, Lk, dk = key_states.shape
            Bv, Hv, Lv, dv = value_states.shape
            group_size = 32
            
            # Compute group stats for Q if head_dim is large enough
            if dq >= group_size:
                num_groups_q = dq // group_size
                q_grouped = query_states[:, :, :, :num_groups_q*group_size].reshape(Bq, Hq, Lq, num_groups_q, group_size)
                q_std_per_group = q_grouped.std(dim=4)  # [Bq, Hq, Lq, num_groups]
                q_max_per_group = q_grouped.abs().max(dim=4).values
                q_crest_factor_per_group = q_max_per_group / (q_std_per_group + 1e-8)
            else:
                num_groups_q = 0
            
            # Compute group stats for K if head_dim is large enough
            if dk >= group_size:
                num_groups_k = dk // group_size
                k_grouped = key_states[:, :, :, :num_groups_k*group_size].reshape(Bk, Hk, Lk, num_groups_k, group_size)
                k_std_per_group = k_grouped.std(dim=4)  # [Bk, Hk, Lk, num_groups]
                k_max_per_group = k_grouped.abs().max(dim=4).values
                k_crest_factor_per_group = k_max_per_group / (k_std_per_group + 1e-8)
            else:
                num_groups_k = 0
            
            # Compute group stats for V along seq_len dimension (L)
            # Note: V is accumulated along L in attention: attn_weights @ V
            # V: [B, H, L, d] -> group along L dimension
            if Lv >= group_size:
                num_groups_v = Lv // group_size
                # Reshape to [B, H, num_groups, group_size, d] and compute stats along group_size (axis=3)
                v_grouped = value_states[:, :, :num_groups_v*group_size, :].reshape(Bv, Hv, num_groups_v, group_size, dv)
                v_std_per_group = v_grouped.std(dim=3)  # [Bv, Hv, num_groups, dv] - std over group_size tokens
                v_max_per_group = v_grouped.abs().max(dim=3).values
                v_crest_factor_per_group = v_max_per_group / (v_std_per_group + 1e-8)  # [Bv, Hv, num_groups, dv]
            else:
                num_groups_v = 0
            
            # Global stats
            q_std_global = query_states.std().item()
            k_std_global = key_states.std().item()
            v_std_global = value_states.std().item()
            
            # Print global statistics
            print(f"\n[Layer {self.layer_idx}] Global (after RoPE): Q std {q_std_global:.4f}, K std {k_std_global:.4f}, V std {v_std_global:.4f}")
            print(f"  Q: heads={Hq}, head_dim={dq}, groups_on_d={num_groups_q if dq >= group_size else 'N/A'}")
            print(f"  K: heads={Hk}, head_dim={dk}, groups_on_d={num_groups_k if dk >= group_size else 'N/A'} (GQA)")
            print(f"  V: heads={Hv}, seq_len={Lv}, groups_on_L={num_groups_v if Lv >= group_size else 'N/A'} (GQA, grouped on seq_len!)")
            
            # ===== Group-wise Statistics =====
            if num_groups_q > 1 or num_groups_k > 1 or num_groups_v > 1:  
                print(f"\n[Layer {self.layer_idx}] Group-wise Statistics (group_size={group_size}):")
            
                # Q group-wise statistics (grouped along head_dim)
                if num_groups_q > 1:
                    q_std_group_flat = q_std_per_group.flatten().float()  # [B*H*L*num_groups]
                    q_std_group_median = q_std_group_flat.median()
                    q_std_group_q25 = torch.quantile(q_std_group_flat, 0.25)
                    q_std_group_q75 = torch.quantile(q_std_group_flat, 0.75)
                    q_std_group_q95 = torch.quantile(q_std_group_flat, 0.95)
                    
                    q_cf_group_flat = q_crest_factor_per_group.flatten().float()  # [B*H*L*num_groups]
                    q_cf_group_median = q_cf_group_flat.median()
                    q_cf_group_q25 = torch.quantile(q_cf_group_flat, 0.25)
                    q_cf_group_q75 = torch.quantile(q_cf_group_flat, 0.75)
                    q_cf_group_q95 = torch.quantile(q_cf_group_flat, 0.95)
                    q_cf_group_max = q_cf_group_flat.max()
                    
                    # High crest factor group ratio
                    q_high_cf_ratio = (q_cf_group_flat > 4.0).float().mean()
                    q_extreme_cf_ratio = (q_cf_group_flat > 6.0).float().mean()
                    
                    print(f"  Q (per group on head_dim):")
                    print(f"    Std: Q25={q_std_group_q25:.4f}, Median={q_std_group_median:.4f}, Q75={q_std_group_q75:.4f}, Q95={q_std_group_q95:.4f}")
                    print(f"    Crest Factor: Q25={q_cf_group_q25:.2f}, Median={q_cf_group_median:.2f}, "
                          f"Q75={q_cf_group_q75:.2f}, Q95={q_cf_group_q95:.2f}, Max={q_cf_group_max:.2f}")
                    print(f"    High CF groups (>4.0): {q_high_cf_ratio:.2%}, Extreme (>6.0): {q_extreme_cf_ratio:.2%}")
                
                # K group-wise statistics (grouped along head_dim)
                if num_groups_k > 1:
                    k_std_group_flat = k_std_per_group.flatten().float()  # [B*H*L*num_groups]
                    k_std_group_median = k_std_group_flat.median()
                    k_std_group_q25 = torch.quantile(k_std_group_flat, 0.25)
                    k_std_group_q75 = torch.quantile(k_std_group_flat, 0.75)
                    k_std_group_q95 = torch.quantile(k_std_group_flat, 0.95)
                    
                    k_cf_group_flat = k_crest_factor_per_group.flatten().float()  # [B*H*L*num_groups]
                    k_cf_group_median = k_cf_group_flat.median()
                    k_cf_group_q25 = torch.quantile(k_cf_group_flat, 0.25)
                    k_cf_group_q75 = torch.quantile(k_cf_group_flat, 0.75)
                    k_cf_group_q95 = torch.quantile(k_cf_group_flat, 0.95)
                    k_cf_group_max = k_cf_group_flat.max()
                    
                    # High crest factor group ratio
                    k_high_cf_ratio = (k_cf_group_flat > 4.0).float().mean()
                    k_extreme_cf_ratio = (k_cf_group_flat > 6.0).float().mean()
                    
                    print(f"  K (per group on head_dim):")
                    print(f"    Std: Q25={k_std_group_q25:.4f}, Median={k_std_group_median:.4f}, Q75={k_std_group_q75:.4f}, Q95={k_std_group_q95:.4f}")
                    print(f"    Crest Factor: Q25={k_cf_group_q25:.2f}, Median={k_cf_group_median:.2f}, "
                          f"Q75={k_cf_group_q75:.2f}, Q95={k_cf_group_q95:.2f}, Max={k_cf_group_max:.2f}")
                    print(f"    High CF groups (>4.0): {k_high_cf_ratio:.2%}, Extreme (>6.0): {k_extreme_cf_ratio:.2%}")
                
                # V group-wise statistics (grouped along seq_len dimension)
                if num_groups_v > 1:
                    # v_std_per_group: [B, H, num_groups, d] - std over group_size tokens per channel
                    v_std_group_flat = v_std_per_group.flatten().float()  # [B*H*num_groups*d]
                    v_std_group_median = v_std_group_flat.median()
                    v_std_group_q25 = torch.quantile(v_std_group_flat, 0.25)
                    v_std_group_q75 = torch.quantile(v_std_group_flat, 0.75)
                    v_std_group_q95 = torch.quantile(v_std_group_flat, 0.95)
                    
                    v_cf_group_flat = v_crest_factor_per_group.flatten().float()  # [B*H*num_groups*d]
                    v_cf_group_median = v_cf_group_flat.median()
                    v_cf_group_q25 = torch.quantile(v_cf_group_flat, 0.25)
                    v_cf_group_q75 = torch.quantile(v_cf_group_flat, 0.75)
                    v_cf_group_q95 = torch.quantile(v_cf_group_flat, 0.95)
                    v_cf_group_max = v_cf_group_flat.max()
                    
                    # High crest factor group ratio
                    v_high_cf_ratio = (v_cf_group_flat > 4.0).float().mean()
                    v_extreme_cf_ratio = (v_cf_group_flat > 6.0).float().mean()
                    
                    print(f"  V (per group on seq_len, per channel):")
                    print(f"    Std: Q25={v_std_group_q25:.4f}, Median={v_std_group_median:.4f}, Q75={v_std_group_q75:.4f}, Q95={v_std_group_q95:.4f}")
                    print(f"    Crest Factor: Q25={v_cf_group_q25:.2f}, Median={v_cf_group_median:.2f}, "
                          f"Q75={v_cf_group_q75:.2f}, Q95={v_cf_group_q95:.2f}, Max={v_cf_group_max:.2f}")
                    print(f"    High CF groups (>4.0): {v_high_cf_ratio:.2%}, Extreme (>6.0): {v_extreme_cf_ratio:.2%}")
                    print(f"    Note: V grouped on L dimension (attn_weights @ V accumulates on L)")
        
        if self.smooth_K:
            key_states = self.smooth_K(key_states)


        # Q/K/V Quantization:
        # Two modes supported:
        # 1. Standard FP quantization (--q_fpq/--k_fpq/--v_fpq):
        #    Uses standard floating-point quantization with fixed E/M format
        # 2. Adaptive codebook quantization (--q_adaptive/--k_adaptive/--v_adaptive):
        #    Dynamically selects codebook based on per-group crest factor (CF = max/std):
        #    - CF <= 2.94:  E1M2 [-3.5 ~ 3.5]  (tight distribution)
        #    - CF <= 9.76:  E2M1 [-6.0 ~ 6.0]  (moderate distribution)
        #    - CF > 9.76:   E3M0 [-16.0 ~ 16.0] (wide distribution)
        
        # Print codebook statistics (sample 1% to reduce overhead)
        print_stats = torch.rand(1).item() < 0.01 and self.layer_idx == 0
        print_stats = False  # Set to True to enable statistics printing
        
        # Q quantization: along head_dim (last dim) - matches accumulation in Q@K^T
        if self.query_quantizer.bits < 16:
            self.query_quantizer.find_params(query_states)
            query_states = self.query_quantizer(query_states)
            
            if print_stats and hasattr(self.query_quantizer, 'get_codebook_stats'):
                q_stats = self.query_quantizer.get_codebook_stats()
                if q_stats:
                    print(f"\n[Layer {self.layer_idx}] Query Codebook Usage:")
                    print(f"  E1M2 (CF≤2.94): {q_stats['E1M2_ratio']:.1%} ({q_stats['E1M2_count']}/{q_stats['total_groups']})")
                    print(f"  E2M1 (2.94<CF≤9.76): {q_stats['E2M1_ratio']:.1%} ({q_stats['E2M1_count']}/{q_stats['total_groups']})")
                    print(f"  E3M0 (CF>9.76): {q_stats['E3M0_ratio']:.1%} ({q_stats['E3M0_count']}/{q_stats['total_groups']})")
                    print(f"  Mean Crest Factor: {q_stats['mean_crest_factor']:.2f}")
            
            self.query_quantizer.free()
        
        # K quantization: along head_dim (last dim) - matches accumulation in Q@K^T
        if self.key_quantizer.bits < 16:
            self.key_quantizer.find_params(key_states)
            key_states = self.key_quantizer(key_states)
            
            if print_stats and hasattr(self.key_quantizer, 'get_codebook_stats'):
                k_stats = self.key_quantizer.get_codebook_stats()
                if k_stats:
                    print(f"\n[Layer {self.layer_idx}] Key Codebook Usage:")
                    print(f"  E1M2 (CF≤2.94): {k_stats['E1M2_ratio']:.1%} ({k_stats['E1M2_count']}/{k_stats['total_groups']})")
                    print(f"  E2M1 (2.94<CF≤9.76): {k_stats['E2M1_ratio']:.1%} ({k_stats['E2M1_count']}/{k_stats['total_groups']})")
                    print(f"  E3M0 (CF>9.76): {k_stats['E3M0_ratio']:.1%} ({k_stats['E3M0_count']}/{k_stats['total_groups']})")
                    print(f"  Mean Crest Factor: {k_stats['mean_crest_factor']:.2f}")
            
            self.key_quantizer.free()
        
        # V quantization: along seq_len dim - matches accumulation in attn_weights@V
        # V: [B, H, L, d] -> transpose to [B, H, d, L] -> quantize -> transpose back
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
            value_states = value_states_transposed.transpose(2, 3)  # [B, H, d, L] -> [B, H, L, d]
            
            if print_stats and hasattr(self.value_quantizer, 'get_codebook_stats'):
                v_stats = self.value_quantizer.get_codebook_stats()
                if v_stats:
                    print(f"\n[Layer {self.layer_idx}] Value Codebook Usage:")
                    print(f"  E1M2 (CF≤2.94): {v_stats['E1M2_ratio']:.1%} ({v_stats['E1M2_count']}/{v_stats['total_groups']})")
                    print(f"  E2M1 (2.94<CF≤9.76): {v_stats['E2M1_ratio']:.1%} ({v_stats['E2M1_count']}/{v_stats['total_groups']})")
                    print(f"  E3M0 (CF>9.76): {v_stats['E3M0_ratio']:.1%} ({v_stats['E3M0_count']}/{v_stats['total_groups']})")
                    print(f"  Mean Crest Factor: {v_stats['mean_crest_factor']:.2f}")
            
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
            sliding_window=self.sliding_window,  # diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights