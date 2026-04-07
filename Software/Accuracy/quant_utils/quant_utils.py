import math
import torch
import torch.nn as nn
import transformers
from quant_utils import utils
from unicore_kernel.unicore_core import *

def fp_scale(tensor, S, M, bias, max_float, min_float):
    tensor_unscaled = (tensor / S)
    tensor_unscaled = torch.clamp(tensor_unscaled, min_float, max_float)
    tensor_log_scales = torch.clamp((torch.floor(torch.log2(torch.abs(tensor_unscaled)) + bias)).detach(), 1.0)
    scales = 2.0 ** (tensor_log_scales - M - bias)
    tensor_q = (tensor_unscaled / scales).round()
    tensor_q = tensor_q * scales
    return tensor_q


# Fake round from S1E5M10 to S1E4M3
# Direct Cast into FP8 instead of affine mapping
def fake_quantize_quarter_E4M3(w: torch.tensor) -> torch.tensor:
    # Ref: https://www.h-schmidt.net/FloatConverter/IEEE754.html
    # Sign:     1000 0000 0000 0000 HEX: 0x8000
    # Exponent: 0111 1100 0000 0000 HEX: 0x7C00
    # Mantissa: 0000 0011 1111 1111 HEX: 0x03FF
    assert w.dtype == torch.float16
    # Maximum number of FP8 E4M3 should be (0 1111 111) = 480
    w = w.cuda()
    w = torch.clamp(w, -480, 480)

    # Manipulate bits
    w = w.view(torch.int16)
    # print in hex
    # First round mantissa
    # Need consider rounding case
    # Just construct a float16 to see whether to round 1 bits
    mantissa = w & 0x03FF
    roundFloat = (((mantissa << 3) & 0x03FF) + 0x3C00).clone().view(torch.float16).cuda()
    roundingBits = (torch.round(roundFloat) - 1).to(dtype=torch.int16)
    mantissa = ((mantissa >> 7) + roundingBits) << 7

    # Deal with subnormal value
    # Round exponent in [-6, 8] + 15 = [9, 23]
    # Ref: https://arxiv.org/pdf/2209.05433.pdf
    # Min normal value:     0 0001 000 = 2^-6
    # Min Submormal Value:  0 0000 001 = 2^-9
    exponent = (w & 0x7C00) >> 10
    subNormalMask = (exponent - 15) < -6
    subNormal_min = torch.tensor(2**(-9), dtype=torch.float16, device='cuda')
    w = w.view(torch.float16)
    w[subNormalMask] = torch.round(w[subNormalMask] / subNormal_min).to(dtype=torch.int16) * subNormal_min
    
    # Deal with normal value
    w = w.view(torch.int16)
    exponent = torch.clamp(exponent, 9, 23)
    exponent = exponent << 10
    w[~subNormalMask] = (w[~subNormalMask] & 0x8000) + mantissa[~subNormalMask] + exponent[~subNormalMask]
    return w.view(torch.float16)


# Ref: GPTAQ
def get_minq_maxq(bits, sym, fpq):
    if fpq:
        if bits == 8:
            E = 4.
            M = 3.
        elif bits == 4:
            E = 2.
            M = 1.
        elif bits == 3:
            E = 2.
            M = 0.
        else:
            raise NotImplementedError(f'Unsupported bits: {bits}')
        bias = 2 ** (E - 1) - 1
        maxq = torch.tensor((2 - 2 ** (-M)) * 2 ** (2**E - 1 - bias))
        minq = -maxq

    else:
        if sym:
            maxq = torch.tensor(2**(bits-1)-1)
            minq = -maxq -1
        else:
            maxq = torch.tensor(2**bits - 1)
            minq = 0

    return minq, maxq

def asym_quant(x, scale, zero, maxq):
    scale = scale.to(x.device)
    zero = zero.to(x.device)
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return q, scale, zero

def asym_dequant(q, scale, zero):
    return scale * (q - zero)

def asym_quant_dequant(x, scale, zero, maxq):
    return asym_dequant(*asym_quant(x, scale, zero, maxq))

def sym_quant(x, scale, maxq):
    scale = scale.to(x.device)
    q = torch.clamp(torch.round(x / scale), -(maxq+1), maxq)
    return q, scale
def sym_dequant(q, scale):
    return scale * q

def sym_quant_dequant(x, scale, maxq):
    return sym_dequant(*sym_quant(x, scale, maxq))

def fpq_sym_quant(x, scale, maxq, bias, M):
    scale = scale.to(x.device)
    x_unscaled = x / scale
    x_unscaled = torch.clamp(x_unscaled, -maxq, maxq)
    subnormal_threshold = 2 ** (1 - bias)
    # Option 1: Push subnormals to threshold (original implementation)
    # x_unscaled = torch.where(x_unscaled > 0, 
    #                  torch.clamp(x_unscaled, min=subnormal_threshold, max=max_float),
    #                  torch.where(x_unscaled < 0,
    #                              torch.clamp(x_unscaled, min=-max_float, max=-subnormal_threshold),
    #                              x_unscaled))
    
    if M >= 3:
        # Option 2: Set subnormals to zero, keep normals unchanged
        x_unscaled = torch.where(
            torch.abs(x_unscaled) < subnormal_threshold,
            torch.zeros_like(x_unscaled),
            x_unscaled
        )
    x_log_scales = torch.clamp((torch.floor(torch.log2(torch.abs(x_unscaled)) + bias)).detach(), 1.0)
    scales = 2.0 ** (x_log_scales - M - bias)
    x_q = (x_unscaled / scales).round()
    x_q = x_q * scales
    return x_q, scale

def fpq_sym_dequant(q, scale):
    out = scale * q
    return out

def fpq_sym_quant_dequant(x, scale, maxq, bias, M):
    out = fpq_sym_dequant(*fpq_sym_quant(x, scale, maxq, bias, M))
    return out

def fpq_asym_quant(x, scale_pos, scale_neg, maxq, bias, M):
    scale_pos = scale_pos.to(x.device)
    scale_neg = scale_neg.to(x.device)
    if (scale_pos == 0).any() or (scale_neg == 0).any():
        raise ValueError("scale_pos or scale_neg contains zero")
    x_unscaled = torch.where(x >= 0, x / scale_pos, x / scale_neg)
    x_unscaled = torch.clamp(x_unscaled, -maxq, maxq)
    x_log_scales = torch.clamp((torch.floor(torch.log2(torch.abs(x_unscaled)) + bias)).detach(), 1.0)
    scales = 2.0 ** (x_log_scales - M - bias)
    x_q = (x_unscaled / scales).round()
    x_q = x_q * scales
    if torch.isnan(x_q).any():
        raise ValueError("x_q contains nan")
    return x_q, scale_pos, scale_neg

def fpq_asym_dequant(q, scale_pos, scale_neg):
    scale_pos = scale_pos.to(q.device)
    scale_neg = scale_neg.to(q.device)
    out = torch.where(q >= 0, scale_pos * q, scale_neg * q)
    return out 

def fpq_asym_quant_dequant(x, scale_pos, scale_neg, maxq, bias, M):
    return fpq_asym_dequant(*fpq_asym_quant(x, scale_pos, scale_neg, maxq, bias, M))

class ActQuantizer(torch.nn.Module):
    '''
        A class for quantizing the activations. We only support (both sym. and asym.) per-token quantization
        for the activations.
    '''

    def __init__(self):
        super(ActQuantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(1))
        self.register_buffer('zero', torch.zeros(1))
        self.bits = 16
        # Whether to apply STE: y = x + (x_q - x).detach()
        # This should be set by add_aq(model, args) based on args.aq_ste and training mode.
        self.use_ste: bool = False
        # Learnable activation scale (LSQ-style multiplicative gain)
        self.learn_scale: bool = False
        self.lsq_alpha: float = 1.0
        self.scale_gain: nn.Parameter | None = None  # shape (1,1,G,1) for groupwise, else scalar-broadcast
        
        # kv quantization codebook
        FP4_E3M0 = [-16.0, -8.0, -4.0, -2.0, -1.0, -0.5, -0.25, 0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
        FP4_E2M1 = [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
        FP4_E1M2 = [-3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]

    def free(self):
        self.zero = None
        self.scale = None
        if hasattr(self, 'scale_pos'):
            self.scale_pos = None
        if hasattr(self, 'scale_neg'):
            self.scale_neg = None
        torch.cuda.empty_cache()

    def forward(self, x):
        x_dtype = x.dtype
        # Compute quant-dequant result x_q across all branches
        if self.bits == 16:
            x_q = x
        elif self.fpq:
            if self.sym:
                x_q = fpq_sym_quant_dequant(x, self.scale, self.maxq, self.bias, self.M)
            else:
                x_q = fpq_asym_quant_dequant(x, self.scale_pos, self.scale_neg, self.maxq, self.bias, self.M)
        else:
            if self.sym:
                x_q = sym_quant_dequant(x, self.scale, self.maxq)
            else:
                x_q = asym_quant_dequant(x, self.scale, self.zero, self.maxq)

        x_q = x_q.to(x_dtype)
        if self.use_ste:
            # Straight-through estimator: forward uses quantized x_q; backward passes gradient of identity
            return x + (x_q - x).detach()
        return x_q

    # Different from `forward`, this method returns quantized integers, scales (and zeros if asymmetric).
    def quantize(self, x):
        if self.fpq:
            if self.sym:
                return fpq_sym_quant(x, self.scale, self.maxq, self.bias, self.M)
            else:
                return fpq_asym_quant(x, self.scale_pos, self.scale_neg, self.maxq, self.bias, self.M)
        else:
            if self.sym:
                return sym_quant(x, self.scale, self.maxq)
            else:
                return asym_quant(x, self.scale, self.zero, self.maxq)

    def configure(self, bits, groupsize=-1, fpq=False, sym=False, clip_ratio=1.0, datatype=""):
        _, self.maxq = get_minq_maxq(bits, sym, fpq)
        self.datatype = datatype
        self.fpq = fpq
        self.bits = bits
        if fpq:
            if bits == 8:
                self.E = 4.
                self.M = 3.
            elif bits == 4:
                self.E = 2.
                self.M = 1.
            elif bits == 3:
                self.E = 2.
                self.M = 0.
            else:
                raise NotImplementedError(f'Unsupported bits: {bits}')
            self.bias = 2 ** (self.E - 1) - 1
        self.groupsize = groupsize
        self.sym = sym
        self.clip_ratio = clip_ratio
        assert self.clip_ratio <= 1 and self.clip_ratio > 0, 'Clip ratio should be in (0, 1]'

    def find_params_per_token_groupwise(self, x):
        init_shape = x.shape
        # print(f"init_x: {x}")
        reshaped_x = x.reshape(-1, x.shape[-2], x.shape[-1] // self.groupsize, self.groupsize) # [batch_size, seq_len, num_groups, group_size]

        xmax = torch.amax(reshaped_x, dim=3, keepdim=True) * self.clip_ratio # [batch_size, seq_len, num_groups, 1]
        xmin = torch.amin(reshaped_x, dim=3, keepdim=True) * self.clip_ratio # [batch_size, seq_len, num_groups, 1]
        
        if self.fpq:
            if self.sym:
                xmax = torch.maximum(torch.abs(xmin), xmax)
                tmp = xmax == 0
                self.scale = xmax / self.maxq
                self.scale[tmp] = 1
                self.zero = torch.zeros_like(self.scale)
            else:
                tmp = (xmin == 0) & (xmax == 0)
                xmin[tmp] = -1
                xmax[tmp] = +1
                pos_mask = xmax > 0
                neg_mask = xmin < 0
                self.scale_pos = torch.where(pos_mask, xmax / self.maxq, torch.ones_like(xmax))
                self.scale_neg = torch.where(neg_mask, (-xmin) / self.maxq, torch.ones_like(xmin))
                self.zero = torch.zeros_like(self.scale_pos)
        else:   
            if self.sym:
                xmax = torch.maximum(torch.abs(xmin), xmax)
                tmp = xmax == 0
                self.scale = xmax / self.maxq
                self.scale[tmp] = 1
                self.zero = torch.zeros_like(self.scale)
            else:
                tmp = (xmin == 0) & (xmax == 0)
                xmin[tmp] = -1
                xmax[tmp] = +1
                self.scale = (xmax - xmin) / self.maxq
                self.zero = torch.round(-xmin / self.scale)
                
        if self.sym:
            if self.fpq:
                if self.groupsize == 16 and self.datatype == "nvfp": # nvfp
                    scale_max = torch.max(self.scale)
                    M = 3.; E = 4.
                    bias = 2 ** (E - 1) - 1
                    max_float = torch.tensor((2 - 2 ** (-M)) * 2 ** (2**E - 1 - bias), device=scale_max.device)
                    min_float = -max_float
                    scale_max_scale = scale_max.to(torch.float32) / max_float
                    eps = torch.tensor(torch.finfo(scale_max.dtype).tiny, device=scale_max.device, dtype=scale_max.dtype)
                    scale_max_scale = torch.clamp(scale_max_scale, min=eps)
                    
                    scale_new_unscaled = (self.scale / scale_max_scale)
                    scale_new_unscaled = torch.clamp(scale_new_unscaled, min_float, max_float)
                    log_scales = torch.clamp((torch.floor(torch.log2(torch.abs(scale_new_unscaled)) + bias)).detach(), 1.0)
                    scales = 2.0 ** (log_scales - M - bias)
                    scale_q = (scale_new_unscaled / scales).round()
                    scale_q = scale_q * scales
                    scale_q = scale_q * scale_max_scale
                    self.scale = scale_q.to(self.scale.dtype)
                    # self.scale = torch.clamp(self.scale, min=eps)
                elif self.groupsize == 32 and self.datatype == "mxfp": # mxfp
                    ...
            else:
                if self.groupsize == 16 and self.datatype == "nvint": # nvint
                    scale_max = torch.max(self.scale)
                    M = 3.; E = 4.
                    bias = 2 ** (E - 1) - 1
                    max_float = torch.tensor((2 - 2 ** (-M)) * 2 ** (2**E - 1 - bias), device=scale_max.device)
                    min_float = -max_float
                    scale_max_scale = scale_max.to(torch.float32) / max_float
                    eps = torch.tensor(torch.finfo(scale_max.dtype).tiny, device=scale_max.device, dtype=scale_max.dtype)
                    scale_max_scale = torch.clamp(scale_max_scale, min=eps)
                    
                    scale_new_unscaled = (self.scale / scale_max_scale)
                    scale_new_unscaled = torch.clamp(scale_new_unscaled, min_float, max_float)
                    log_scales = torch.clamp((torch.floor(torch.log2(torch.abs(scale_new_unscaled)) + bias)).detach(), 1.0)
                    scales = 2.0 ** (log_scales - M - bias)
                    scale_q = (scale_new_unscaled / scales).round()
                    scale_q = scale_q * scales
                    scale_q = scale_q * scale_max_scale
                    self.scale = scale_q.to(self.scale.dtype)

        if self.fpq and not self.sym and self.groupsize == 32 and self.datatype == "unicore":
            # UniCore double quant
            assert self.scale_pos.shape[2] % 4 == 0
            scale_pos_new = self.scale_pos.reshape(self.scale_pos.shape[0], self.scale_pos.shape[1], self.scale_pos.shape[2] // 4, 4)
            scale_neg_new = self.scale_neg.reshape(self.scale_neg.shape[0], self.scale_neg.shape[1], self.scale_neg.shape[2] // 4, 4)
            scale_pos_max = torch.amax(scale_pos_new, dim=3, keepdim=True)
            scale_neg_max = torch.amax(scale_neg_new, dim=3, keepdim=True)
            scale_max = torch.maximum(scale_pos_max, scale_neg_max)
            M = 0.; E = 3.
            bias = 2 ** (E - 1) - 1
            max_float = torch.tensor((2 - 2 ** (-M)) * 2 ** (2**E - 1 - bias), device=scale_max.device)
            min_float = -max_float
            scale_max_scale = scale_max / max_float
            eps = torch.tensor(torch.finfo(scale_max_scale.dtype).tiny, device=scale_max_scale.device, dtype=scale_max_scale.dtype)
            scale_max_scale = torch.clamp(scale_max_scale, min=eps)
            
            scale_pos_new_unscaled = (scale_pos_new / scale_max_scale)
            if torch.isnan(scale_pos_new_unscaled).any():
                raise ValueError("scale_pos_new_unscaled is nan")
            scale_pos_new_unscaled = torch.clamp(scale_pos_new_unscaled, min_float, max_float)
            pos_log_scales = torch.clamp((torch.floor(torch.log2(torch.abs(scale_pos_new_unscaled)) + bias)).detach(), 1.0)
            pos_scales = 2.0 ** (pos_log_scales - M - bias)
            scale_pos_new_q = (scale_pos_new_unscaled / pos_scales).round()
            scale_pos_new_q = scale_pos_new_q * pos_scales
            
            scale_neg_new_unscaled = (scale_neg_new / scale_max_scale)
            if torch.isnan(scale_neg_new_unscaled).any():
                raise ValueError("scale_neg_new_unscaled is nan")
            scale_neg_new_unscaled = torch.clamp(scale_neg_new_unscaled, min_float, max_float)
            neg_log_scales = torch.clamp((torch.floor(torch.log2(torch.abs(scale_neg_new_unscaled)) + bias)).detach(), 1.0)
            neg_scales = 2.0 ** (neg_log_scales - M - bias)
            scale_neg_new_q = (scale_neg_new_unscaled / neg_scales).round()
            scale_neg_new_q = scale_neg_new_q * neg_scales
            
            scale_pos_new = scale_pos_new_q * scale_max_scale
            scale_neg_new = scale_neg_new_q * scale_max_scale
            
            self.scale_pos = scale_pos_new.reshape(self.scale_pos.shape[0], self.scale_pos.shape[1], self.scale_pos.shape[2], 1).to(self.scale_pos.dtype)
            self.scale_neg = scale_neg_new.reshape(self.scale_neg.shape[0], self.scale_neg.shape[1], self.scale_neg.shape[2], 1).to(self.scale_neg.dtype)
            
            eps = torch.tensor(torch.finfo(self.scale_pos.dtype).tiny, device=self.scale_pos.device, dtype=self.scale_pos.dtype)
            self.scale_pos = torch.clamp(self.scale_pos, min=eps)
            self.scale_neg = torch.clamp(self.scale_neg, min=eps)
            
            self.scale_pos = self.scale_pos.repeat(1, 1, 1, self.groupsize).reshape(init_shape)
            self.scale_neg = self.scale_neg.repeat(1, 1, 1, self.groupsize).reshape(init_shape)
            # self.zero = self.zero.repeat(1, 1, 1, self.groupsize).reshape(init_shape)
            return
        eps = torch.tensor(torch.finfo(self.scale.dtype).tiny, device=self.scale.device, dtype=self.scale.dtype)
        self.scale = torch.clamp(self.scale, min=eps)
        self.scale = self.scale.repeat(1, 1, 1, self.groupsize).reshape(init_shape)
        self.zero = self.zero.repeat(1, 1, 1, self.groupsize).reshape(init_shape)

    def find_params(self, x):
        if self.bits == 16:
            return

        dev = x.device
        self.maxq = self.maxq.to(dev)

        init_shape = x.shape

        if self.groupsize > 0:
            # group-wise per-token quantization
            self.find_params_per_token_groupwise(x)
            # utils.cleanup_memory(verbos=False)
            return

        reshaped_x = x.reshape((-1, x.shape[-1]))

        tmp = torch.zeros(reshaped_x.shape[0], device=dev)
        xmin = torch.minimum(reshaped_x.min(1)[0], tmp) * self.clip_ratio
        xmax = torch.maximum(reshaped_x.max(1)[0], tmp) * self.clip_ratio
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmax == 0
            self.scale = (xmax / self.maxq).unsqueeze(1).repeat(1, reshaped_x.shape[-1])
            self.scale[tmp] = 1
            self.scale = self.scale.reshape(init_shape)
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

            self.scale = self.scale.unsqueeze(1).repeat(1, reshaped_x.shape[-1]).reshape(init_shape)
            self.zero = self.zero.unsqueeze(1).repeat(1, reshaped_x.shape[-1]).reshape(init_shape)

class AdaptiveCodebookQuantizer(ActQuantizer):
    """
    Adaptive codebook quantizer that selects FP4 codebook based on group-wise crest factor.
    
    Crest Factor (CF) = max / rms for each group
    - CF <= 2.0:  Use E1M2 codebook (tight distribution)
    - CF <= 10.42:  Use E2M1 codebook (moderate distribution) 
    - CF >  10.42:  Use E3M0 codebook (wide distribution) # actually not use in group size = 32
    
    This quantizer only supports group-wise quantization with fpq=True and sym=True.
    """
    
    # Define codebook constants
    FP4_E1M2 = torch.tensor([-3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 
                              0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    FP4_E2M1 = torch.tensor([-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 
                              0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    FP4_E3M0 = torch.tensor([-16.0, -8.0, -4.0, -2.0, -1.0, -0.5, -0.25, 0.0, 
                              0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
    
    # Crest factor thresholds
    CF_THRESHOLD_LOW = 1.0
    CF_THRESHOLD_HIGH = 5.2
    
    def __init__(self):
        super(AdaptiveCodebookQuantizer, self).__init__()
        # Store codebook selection for each group: 0=E1M2, 1=E2M1, 2=E3M0
        self.register_buffer('codebook_indices', torch.zeros(1, dtype=torch.long))
        # Store crest factors for debugging/analysis
        self.register_buffer('crest_factors', torch.zeros(1))
    
    def configure(self, bits, groupsize=-1, fpq=False, sym=False, clip_ratio=1.0, datatype=""):
        """Configure quantizer. Adaptive codebook only works with group-wise FP quantization."""
        if bits < 16 and (not fpq or not sym or groupsize <= 0):
            raise ValueError(
                "AdaptiveCodebookQuantizer requires fpq=True, sym=True, and groupsize > 0. "
                f"Got: fpq={fpq}, sym={sym}, groupsize={groupsize}"
            )
        super().configure(bits, groupsize, fpq, sym, clip_ratio, datatype)
    
    def _compute_crest_factors(self, reshaped_x):
        """
        Compute crest factor for each group.
        
        Args:
            reshaped_x: [batch_size, seq_len, num_groups, group_size]
        
        Returns:
            crest_factors: [batch_size, seq_len, num_groups, 1]
        """
        # Compute RMS and max per group
        rms_per_group = torch.sqrt(torch.mean(reshaped_x * reshaped_x, dim=3, keepdim=True))  # [B, L, G, 1]
        max_per_group = torch.max(torch.abs(reshaped_x), dim=3, keepdim=True).values  # [B, L, G, 1]
        
        # Crest factor = max / rms (add epsilon to avoid division by zero)
        crest_factors = max_per_group / (rms_per_group + 1e-8)
        
        return crest_factors
    
    def _select_codebook_indices(self, crest_factors):
        """
        Select codebook index based on crest factor thresholds.
        
        Args:
            crest_factors: [batch_size, seq_len, num_groups, 1]
        
        Returns:
            codebook_indices: [batch_size, seq_len, num_groups, 1] 
                             0=E1M2, 1=E2M1, 2=E3M0
        """
        indices = torch.zeros_like(crest_factors, dtype=torch.long)
        
        # E1M2 for CF <= 2.0
        mask_e1m2 = crest_factors <= self.CF_THRESHOLD_LOW
        indices[mask_e1m2] = 0
        
        # E2M1 for 2.0 < CF <= 10.42
        mask_e2m1 = (crest_factors > self.CF_THRESHOLD_LOW) & (crest_factors <= self.CF_THRESHOLD_HIGH)
        indices[mask_e2m1] = 1
        
        # E3M0 for CF > 10.42，when group size = 32, the largest cf is 5.66. So the is no E3M0 actually.
        mask_e3m0 = crest_factors > self.CF_THRESHOLD_HIGH
        indices[mask_e3m0] = 2
        
        return indices
    
    def _quantize_with_adaptive_codebook(self, reshaped_x, codebook_indices):
        """
        Quantize each group using its selected codebook.
        
        Args:
            reshaped_x: [batch_size, seq_len, num_groups, group_size]
            codebook_indices: [batch_size, seq_len, num_groups, 1]
        
        Returns:
            quantized: [batch_size, seq_len, num_groups, group_size]
        """
        device = reshaped_x.device
        dtype = reshaped_x.dtype
        
        # Convert codebooks to the same device and dtype as input
        codebooks = [
            self.FP4_E1M2.to(device=device, dtype=dtype),
            self.FP4_E2M1.to(device=device, dtype=dtype),
            self.FP4_E3M0.to(device=device, dtype=dtype)
        ]
        
        # 1) Per-group max with clipping (shared across methods)
        xmax = torch.amax(reshaped_x, dim=3, keepdim=True) * self.clip_ratio
        xmin = torch.amin(reshaped_x, dim=3, keepdim=True) * self.clip_ratio
        xmax = torch.maximum(torch.abs(xmin), xmax)
        
        # 2) Per-group, per-codebook qmax and scaling
        #    E1M2 -> 3.5, E2M1 -> 6.0, E3M0 -> 16.0
        qmax_per_cb = torch.tensor([3.5, 6.0, 16.0], device=device, dtype=dtype)
        # codebook_indices: [B, L, G, 1] -> [B, L, G]
        qmax_map = qmax_per_cb[codebook_indices.squeeze(-1)].unsqueeze(-1)  # [B, L, G, 1]
        
        # Avoid division-by-zero groups by setting scale to 1 where xmax==0
        scale = xmax / qmax_map
        scale = torch.where(xmax == 0, torch.ones_like(scale), scale)

        # # 3) Quantize scale with E1M6 floating-point format (per-group), similar to quant_weight.scale_fp
        # #    Compute in float32 for numerical stability, then cast back to input dtype
        # scale_fp = scale.to(torch.float32)
        # # Global maximum for dynamic scaling to E1M6 range
        # scale_fp_max = torch.amax(scale_fp.abs())
        # # If all zeros (unlikely due to above), skip quantization
        # if (scale_fp_max > 0).item():
        #     M, E = 6.0, 1.0
        #     bias = 2 ** (E - 1) - 1  # -> 0 for E=1
        #     max_float = (2 - 2 ** (-M)) * 2 ** (2 ** E - 1 - bias)
        #     min_float = -max_float
        #     # Dynamic scaling to map max to representable range
        #     eps32 = torch.finfo(torch.float32).tiny
        #     scale_fp_scale = torch.clamp(scale_fp_max / max_float, min=eps32)
        #     scale_unscaled = scale_fp / scale_fp_scale
        #     scale_unscaled = torch.clamp(scale_unscaled, min_float, max_float)
        #     # Exponent/mantissa rounding to E1M6
        #     tensor_log_scales = torch.floor(torch.log2(torch.abs(scale_unscaled)) + bias)
        #     tensor_log_scales = torch.clamp(tensor_log_scales, min=1.0)
        #     scales_e1m6 = torch.pow(torch.tensor(2.0, dtype=torch.float32, device=scale_unscaled.device), (tensor_log_scales - M - bias))
        #     scale_unscaled = (scale_unscaled / scales_e1m6).round() * scales_e1m6
        #     scale_q = (scale_unscaled * scale_fp_scale).to(dtype)
        #     scale = scale_q

        # 4) Normalize by scale
        x_normalized = reshaped_x / scale
        # 5) Midpoint-thresholding with bucketize for each selected codebook
        quantized = torch.zeros_like(reshaped_x)
        for codebook_idx in range(3):
            mask = (codebook_indices == codebook_idx).squeeze(-1)
            if not mask.any():
                continue
            codebook = codebooks[codebook_idx]
            # Midpoints between adjacent codebook entries (must be sorted ascending)
            mids = (codebook[:-1] + codebook[1:]) / 2

            mask_expanded = mask.unsqueeze(-1).expand_as(x_normalized)
            x_sel = x_normalized[mask_expanded].reshape(-1)
            if x_sel.numel() == 0:
                continue

            # bucketize returns index in [0, K-1] for K codebook entries
            idx = torch.bucketize(x_sel, mids, right=False)
            q_sel = codebook[idx]

            # Put back into quantized tensor (dtype already matches)
            quantized[mask_expanded] = q_sel

        # 6) De-normalize to original scale
        quantized = quantized * scale
        return quantized
    
    def find_params_per_token_groupwise(self, x):
        """
        Override parent method to add adaptive codebook selection.
        """
        init_shape = x.shape
        reshaped_x = x.reshape(-1, x.shape[-2], x.shape[-1] // self.groupsize, self.groupsize)
        
        # Compute crest factors
        self.crest_factors = self._compute_crest_factors(reshaped_x)
        
        # Select codebook for each group
        self.codebook_indices = self._select_codebook_indices(self.crest_factors)
        
        # For standard scale computation (used in _quantize_with_adaptive_codebook)
        xmax = torch.amax(reshaped_x, dim=3, keepdim=True) * self.clip_ratio
        xmin = torch.amin(reshaped_x, dim=3, keepdim=True) * self.clip_ratio
        xmax = torch.maximum(torch.abs(xmin), xmax)
        tmp = xmax == 0
        self.scale = xmax / self.maxq
        self.scale[tmp] = 1
        self.zero = torch.zeros_like(self.scale)
        
        # Expand scale to full shape for compatibility
        eps = torch.tensor(torch.finfo(self.scale.dtype).tiny, device=self.scale.device, dtype=self.scale.dtype)
        self.scale = torch.clamp(self.scale, min=eps)
        self.scale = self.scale.repeat(1, 1, 1, self.groupsize).reshape(init_shape)
        self.zero = self.zero.repeat(1, 1, 1, self.groupsize).reshape(init_shape)
    
    def forward(self, x):
        """
        Forward pass with adaptive codebook quantization.
        """
        if self.bits == 16:
            return x
        
        x_dtype = x.dtype
        
        # Reshape to group format
        init_shape = x.shape
        reshaped_x = x.reshape(-1, x.shape[-2], x.shape[-1] // self.groupsize, self.groupsize)
        
        # Quantize with adaptive codebook
        quantized_reshaped = self._quantize_with_adaptive_codebook(reshaped_x, self.codebook_indices)
        
        # Reshape back
        x_q = quantized_reshaped.reshape(init_shape)
        x_q = x_q.to(x_dtype)
        
        if self.use_ste:
            return x + (x_q - x).detach()
        return x_q
    
    def free(self):
        """Free memory."""
        super().free()
        self.codebook_indices = None
        self.crest_factors = None
        torch.cuda.empty_cache()
    
    def get_codebook_stats(self):
        """
        Get statistics about codebook usage for analysis.
        
        Returns:
            dict with codebook usage statistics
        """
        if self.codebook_indices is None:
            return None
        
        total_groups = self.codebook_indices.numel()
        e1m2_count = (self.codebook_indices == 0).sum().item()
        e2m1_count = (self.codebook_indices == 1).sum().item()
        e3m0_count = (self.codebook_indices == 2).sum().item()
        
        return {
            'total_groups': total_groups,
            'E1M2_count': e1m2_count,
            'E1M2_ratio': e1m2_count / total_groups if total_groups > 0 else 0,
            'E2M1_count': e2m1_count,
            'E2M1_ratio': e2m1_count / total_groups if total_groups > 0 else 0,
            'E3M0_count': e3m0_count,
            'E3M0_ratio': e3m0_count / total_groups if total_groups > 0 else 0,
            'mean_crest_factor': self.crest_factors.mean().item() if self.crest_factors is not None else None,
            'median_crest_factor': self.crest_factors.median().item() if self.crest_factors is not None else None,
        }


class ActQuantWrapper(torch.nn.Module):
    '''
        This class is a wrapper for the activation quantization.
        We extract the FP features in the forward pass and quantize the rest using
        the self.quantizer object.
        If a rotation Q is provided, the weight matrix will be rotated,
        a pre-forward hook will be registerd to rotate the activation before quantization.
    '''

    def __init__(self, module:torch.nn.Linear, module_path):
        super(ActQuantWrapper, self).__init__()
        assert isinstance(module, torch.nn.Linear)
        self.module = module
        self.module_path = module_path
        self.weight = module.weight
        self.bias = module.bias
        self.quantizer = ActQuantizer()
        self.out_quantizer = ActQuantizer()
        self.register_buffer('had_K', torch.tensor(0))
        self._buffers['had_K'] = None
        self.K = 1
        self.online_full_had = False
        self.online_partial_had = False
        self.had_dim = 0
        self.fp32_had = False

    def extra_repr(self) -> str:
        str_ = f'Input Quantizer Bits: {self.quantizer.bits}'
        if self.quantizer.bits < 16:
            str_ += f' (Asymmetric Per-Token)' if not self.quantizer.sym else f' (Symmetric Per-Token)'

        str_ += f'\nOutput Quantizer Bits: {self.out_quantizer.bits}'
        if self.out_quantizer.bits < 16:
            str_ += f' (Asymmetric Per-Token)' if not self.out_quantizer.sym else f' (Symmetric Per-Token)'

        return str_

    def forward(self, x):
        x_dtype = x.dtype
        # print(f"original x: {x} \nmodule path: {self.module_path}")
        if torch.isnan(x).any():
            raise ValueError("x is nan")

        if self.quantizer.bits < 16: #Quantize, if needed
            self.quantizer.find_params(x)
            x = self.quantizer(x)
            # print(f"quantized x: {x} \nmodule path: {self.module_path}")
            if torch.isnan(x).any():
                raise ValueError("x is nan")
            x = x.to(x_dtype)
            if torch.isnan(x).any():
                raise ValueError("x is nan")
            self.quantizer.free()
            
        x = self.module(x).to(x_dtype)
        
        if self.out_quantizer.bits < 16: #Quantize the output, if needed
            self.out_quantizer.find_params(x)
            x = self.out_quantizer(x).to(x_dtype)
            self.out_quantizer.free()

        return x
    
    
class UniCoreActQuantWrapper(torch.nn.Module):
    '''
        This class is a wrapper for the activation quantization.
        We extract the FP features in the forward pass and quantize the rest using
        the self.quantizer object.
        If a rotation Q is provided, the weight matrix will be rotated,
        a pre-forward hook will be registerd to rotate the activation before quantization.
    '''

    def __init__(self, module:torch.nn.Linear, module_path):
        super(UniCoreActQuantWrapper, self).__init__()
        assert isinstance(
            module,
            (
                torch.nn.Linear,
                UniCoreLinearFP8,
                UniCoreLinearGroupFP8,
                UniCoreLinearFP4,
                UniCoreLinearGroupFP4,
            ),
        )
        self.module = module
        self.module_path = module_path
        self.weight = module.weight
        self.bias = module.bias
        self.quantizer = ActQuantizer()
        self.out_quantizer = ActQuantizer()

    def extra_repr(self) -> str:
        str_ = f'Input Quantizer Bits: {self.quantizer.bits}'
        if self.quantizer.bits < 16:
            str_ += f' (Asymmetric Per-Token)' if not self.quantizer.sym else f' (Symmetric Per-Token)'

        str_ += f'\nOutput Quantizer Bits: {self.out_quantizer.bits}'
        if self.out_quantizer.bits < 16:
            str_ += f' (Asymmetric Per-Token)' if not self.out_quantizer.sym else f' (Symmetric Per-Token)'

        return str_

    def forward(self, x):
        x_dtype = x.dtype
        
        # Input quantization
        if self.quantizer.bits < 16:
            self.quantizer.find_params(x)
            
            # Check if module supports forward_with_scale (for FP quantization with scale passing)
            if hasattr(self.module, 'forward_with_scale') and self.quantizer.fpq and self.quantizer.sym:
                # Get quantized value and scale(s) without dequant
                quant_result = self.quantizer.quantize(x)
                
                if self.quantizer.sym:
                    # fpq_sym: returns (x_q, scale)
                    x_q, scale_full = quant_result
                    # Extract scale for forward_with_scale
                    # scale_full shape: [batch, seq_len, K] (repeated within each group)
                    if self.quantizer.groupsize <= 0:  # per-token
                        # Per-token: all K elements have same scale, take first
                        scale_per_token = scale_full[..., 0:1]  # [batch, seq_len, 1]
                    else:  # per-group
                        # Per-group: extract one scale per group (every groupsize elements)
                        # scale_full[..., ::groupsize] takes every groupsize-th element
                        scale_per_token = scale_full[..., ::self.quantizer.groupsize]  # [batch, seq_len, K//g]
                    x = self.module.forward_with_scale(x_q.to(self.weight.dtype), scale_per_token)
            else:
                # Standard quantize-dequant path
                x = self.quantizer(x).to(self.weight.dtype)
                x = self.module(x)
            
            self.quantizer.free()
        else:
            x = x.to(self.weight.dtype)
            x = self.module(x)
        
        x = x.to(x_dtype)
        
        # Output quantization
        if self.out_quantizer.bits < 16:
            self.out_quantizer.find_params(x)
            x = self.out_quantizer(x).to(x_dtype)
            self.out_quantizer.free()

        return x
    
        
def add_actquant(
    module,
    name='',
    layers_to_wrap=None,
    use_unicore=False,
):
    if layers_to_wrap is None:
        if use_unicore:
            layers_to_wrap = [
                torch.nn.Linear,
                UniCoreLinearFP8,
                UniCoreLinearGroupFP8,
                UniCoreLinearFP4,
                UniCoreLinearGroupFP4,
            ]
        else:
            layers_to_wrap = [torch.nn.Linear]

    wrapper_class = UniCoreActQuantWrapper if use_unicore else ActQuantWrapper
    
    if isinstance(module, (ActQuantWrapper, UniCoreActQuantWrapper)):
        return

    for child_name, child_module in module.named_children():
        full_child_name = f"{name}.{child_name}" if name else child_name

        if type(child_module) in layers_to_wrap:
            # Skip lm_head and output_layer
            if 'lm_head' not in full_child_name and 'output_layer' not in full_child_name:
                wrapper = wrapper_class(child_module, full_child_name)
                setattr(module, child_name, wrapper)
        else:
            add_actquant(child_module, name=full_child_name, layers_to_wrap=layers_to_wrap, use_unicore=use_unicore)
        
        
def find_qlayers(module, layers=[torch.nn.Linear,
                                ActQuantWrapper, UniCoreActQuantWrapper], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_qlayers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def disable_act_quant(module):
    bits_config = {}
    for name, m in module.named_modules():
        if isinstance(m, ActQuantWrapper):
            bits_config[name] = m.quantizer.bits
            m.quantizer.bits = 16

    return bits_config


def enable_act_quant(module, bits_config):
    for name, m in module.named_modules():
        if isinstance(m, ActQuantWrapper):
            m.quantizer.bits = bits_config[name]
            
def configure_attention_quantizers(model, args):
    """
    Configure quantizers for attention QKV and scores, and smooth_K operation.
    
    Args:
        model: The model containing attention modules
        args: Arguments containing quantization configuration
            - q_bits, k_bits, v_bits, attnw_bits: bit precision for each component
            - q_groupsize, k_groupsize, v_groupsize, attnw_groupsize: group sizes
            - q_asym, k_asym, v_asym, attnw_asym: asymmetric quantization flags
            - q_clip_ratio, k_clip_ratio, v_clip_ratio, attnw_clip_ratio: clipping ratios
            - smooth_k: enable smooth K operation
            - smooth_k_strategy: 'per_channel' or 'per_token'
            - aq_datatype: default datatype for all quantizers
    """
    # Configure smooth_K first (before quantization)
    smooth_k_enabled = getattr(args, 'smooth_k', False)
    smooth_k_strategy = getattr(args, 'smooth_k_strategy', 'per_channel')
    
    for name, module in model.named_modules():
        if hasattr(module, 'smooth_K'):
            module.smooth_K.enabled = smooth_k_enabled
            module.smooth_K.strategy = smooth_k_strategy
    
    # Configuration map: (quantizer_attr, bit_arg, groupsize_arg, asym_arg, clip_arg, datatype_arg, adaptive_arg)
    quant_configs = [
        ('query_quantizer', 'q_bits', 'q_groupsize', 'q_asym', 'q_clip_ratio', 'q_datatype', 'q_adaptive'),
        ('key_quantizer', 'k_bits', 'k_groupsize', 'k_asym', 'k_clip_ratio', 'k_datatype', 'k_adaptive'),
        ('value_quantizer', 'v_bits', 'v_groupsize', 'v_asym', 'v_clip_ratio', 'v_datatype', 'v_adaptive'),
        ('attnw_quantizer', 'attnw_bits', 'attnw_groupsize', 'attnw_asym', 'attnw_clip_ratio', 'attnw_datatype', None),
    ]
    
    for quant_attr, bit_arg, gs_arg, asym_arg, clip_arg, dtype_arg, adaptive_arg in quant_configs:
        bits = getattr(args, bit_arg, None)
        
        # Skip if bits not specified or >= 16
        if bits is None or bits >= 16:
            continue
            
        # Get configuration parameters with fallback to activation quantization defaults
        groupsize = getattr(args, gs_arg, args.a_groupsize)
        sym = not getattr(args, asym_arg, args.a_asym)
        clip_ratio = getattr(args, clip_arg, args.a_clip_ratio)
        datatype = getattr(args, dtype_arg, args.aq_datatype)
        # fpq = getattr(args, f'{bit_arg[0]}_fpq', False)  
        prefix = bit_arg.replace('_bits', '')
        fpq = getattr(args, f'{prefix}_fpq', False)  # e.g., q_fpq, k_fpq
        adaptive = getattr(args, adaptive_arg, False) if adaptive_arg else False
        
        # Apply configuration to all modules with this quantizer
        for name, module in model.named_modules():
            if hasattr(module, quant_attr):
                # Replace with AdaptiveCodebookQuantizer if adaptive is enabled
                if adaptive and fpq and sym and groupsize > 0:
                    # Replace the quantizer with AdaptiveCodebookQuantizer
                    setattr(module, quant_attr, AdaptiveCodebookQuantizer())
                
                quantizer = getattr(module, quant_attr)
                quantizer.configure(
                    bits=bits,
                    groupsize=groupsize,
                    fpq=fpq,
                    sym=sym,
                    clip_ratio=clip_ratio,
                    datatype=datatype
                )
                # Optionally apply STE if needed
                use_ste_flag = bool(getattr(args, 'aq_ste', False) and 
                                  ((getattr(args, 'qat_steps', 0) > 0) or 
                                   getattr(args, 'smoke_test', False)))
                quantizer.use_ste = use_ste_flag


def add_aq(model, args):
    # Add Input Quantization6
    if args.a_bits < 16:
        qlayers = find_qlayers(model, layers=[ActQuantWrapper, UniCoreActQuantWrapper])
        # down_proj_groupsize = -1
        # if args.a_groupsize > 0 and "llama" in args.model:
        #     down_proj_groupsize = utils.llama_down_proj_groupsize(model, args.a_groupsize)

        # Enable STE only during training (qat_steps>0 or smoke_test) and if explicitly requested
        use_ste_flag = bool(getattr(args, 'aq_ste', False) and ((getattr(args, 'qat_steps', 0) > 0) or getattr(args, 'smoke_test', False)))

        for name in qlayers:
            layer_input_bits = args.a_bits
            layer_groupsize = args.a_groupsize
            layer_a_fpq = args.a_fpq
            layer_a_sym = not(args.a_asym)
            layer_a_clip = args.a_clip_ratio

            # if 'v_proj' in name and args.v_bits < 16: #Set the v_proj precision
            #     qlayers[name].out_quantizer.configure(bits=args.v_bits,
            #                                   groupsize=args.v_groupsize,
            #                                   sym=not(args.v_asym),
            #                                   clip_ratio=args.v_clip_ratio)

            if 'lm_head' in name: #Skip lm_head quantization
                layer_input_bits = 16
                layer_a_fpq = False

            if 'down_proj' in name:  #Set the down_proj precision
                # if args.int8_down_proj:
                #     layer_input_bits = 8
                down_proj_groupsize = args.a_groupsize
                layer_groupsize = down_proj_groupsize
                # layer_input_bits = 8
                # layer_a_sym = True

            qlayers[name].quantizer.configure(bits=layer_input_bits,
                                              groupsize=layer_groupsize,
                                              fpq=layer_a_fpq,
                                              sym=layer_a_sym,
                                              clip_ratio=layer_a_clip,
                                              datatype=args.aq_datatype)
            # Apply STE option to both input and output quantizers
            qlayers[name].quantizer.use_ste = use_ste_flag
            qlayers[name].out_quantizer.use_ste = use_ste_flag
            
    # Configure attention quantizers (Q, K, V, Score)
    configure_attention_quantizers(model, args)
    
    # if args.k_bits < 16:
    #     if args.k_pre_rope:
    #         raise NotImplementedError("Pre-RoPE quantization is not supported yet!")
    #     else:
    #         rope_function_name = model_utils.get_rope_function_name(model)
    #         layers = model_utils.get_layers(model)
    #         k_quant_config = {'k_bits':args.k_bits, "k_groupsize": args.k_groupsize,
    #                                       "k_sym": not(args.k_asym), "k_clip_ratio": args.k_clip_ratio}
    #         for layer in layers:
    #             rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
    #                         layer.self_attn,
    #                         rope_function_name,
    #                         config=model.config,
    #                         **k_quant_config)
