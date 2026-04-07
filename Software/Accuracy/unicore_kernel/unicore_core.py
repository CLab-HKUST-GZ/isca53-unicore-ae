from requests import head
import torch
from torch.utils.cpp_extension import load
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir

unicore_core_cuda_module = load(
    name="unicore_core_gemm",
    sources=[os.path.join(project_root, "kernel", "unicore_core.cpp"),
            os.path.join(project_root, "kernel", "unicore_core.cu")],
    extra_cuda_cflags=['-O3', '--ptxas-options=-v', '-std=c++17'],
    extra_include_paths=[os.path.join(project_root, "include")],
    verbose=True
)

FP8_COMPENSATION_METHOD_IDS = {
    "fpma_cg": 0,
    "unicore": 1,
}

FP4_TO_FP8 = {
    0.0: 0.0,
    0.5: 2 ** -7,
    1.0: 2 ** -6,
    1.5: (2 ** -6) * 1.5,
    2.0: 2 ** -5,
    3.0: (2 ** -5) * 1.5,
    4.0: 2 ** -4,
    6.0: (2 ** -4) * 1.5,
    -0.5: -(2 ** -7),
    -1.0: -(2 ** -6),
    -1.5: -((2 ** -6) * 1.5),
    -2.0: -(2 ** -5),
    -3.0: -((2 ** -5) * 1.5),
    -4.0: -(2 ** -4),
    -6.0: -((2 ** -4) * 1.5),
}


def resolve_fp8_compensation_method(method: str) -> int:
    try:
        return FP8_COMPENSATION_METHOD_IDS[method]
    except KeyError as exc:
        choices = ", ".join(sorted(FP8_COMPENSATION_METHOD_IDS))
        raise ValueError(
            f"Unsupported FP8 compensation method '{method}'. Choices: {choices}"
        ) from exc


def map_fp4_to_fp8_lut_values_strict(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype != torch.bfloat16:
        tensor = tensor.to(torch.bfloat16)

    mapped = torch.empty_like(tensor, dtype=torch.bfloat16)
    unique_vals = torch.unique(tensor)
    lut_keys = set(float(k) for k in FP4_TO_FP8.keys())
    unmatched = []

    for value in unique_vals:
        value_float = float(value.item())
        if value_float not in lut_keys:
            unmatched.append(value_float)

    if unmatched:
        raise ValueError(
            f"Found values not in strict FP4 LUT: {sorted(set(unmatched))}"
        )

    for value in unique_vals:
        value_float = float(value.item())
        mapped_value = torch.tensor(
            FP4_TO_FP8[value_float], device=tensor.device, dtype=torch.bfloat16
        )
        mapped[tensor == value] = mapped_value

    return mapped

def pack_fp8_e4m3_from_bf16(w: torch.Tensor) -> torch.Tensor:
    """
    Pack bfloat16 tensor to FP8 E4M3 format (uint8).
    This function converts BF16 values to FP8 E4M3 format for efficient GEMM.
    """
    assert w.dtype == torch.bfloat16

    bits = w.view(torch.int16)
    sign = ((bits & 0x8000) >> 8).to(torch.uint8)
    abs_w = torch.abs(w)
    is_nonzero = ((bits & 0x7FFF) != 0)
    sub_mask = (abs_w < torch.tensor(2**(-6), dtype=torch.bfloat16, device=w.device)) & is_nonzero

    e_bf = ((bits & 0x7F80) >> 7).to(torch.int16)
    e_unb = (e_bf - 127).clamp(-6, 8)
    e4_bits = ((e_unb + 7).to(torch.uint8) & 0x0F) << 3
    m3_norm = (((bits & 0x007F) >> 4).to(torch.uint8)) & 0x07

    step = torch.tensor(2**(-9), dtype=torch.bfloat16, device=w.device)
    m3_sub = torch.clamp(torch.round((abs_w / step)).to(torch.int16), 0, 7).to(torch.uint8)

    out = sign.clone()
    out[~sub_mask] = (sign[~sub_mask] | e4_bits[~sub_mask] | m3_norm[~sub_mask])
    out[sub_mask] = (sign[sub_mask] | m3_sub[sub_mask])
    out[~is_nonzero] = sign[~is_nonzero]
    out = out.to(torch.uint8)
    return out

class UniCoreLinearBF16(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=False, dev='cuda'):
        super(UniCoreLinearBF16, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer('weight', torch.empty(self.out_features,
                                                   self.in_features, dtype=torch.bfloat16, requires_grad=False, device=dev))

        if bias:
            self.register_buffer('bias', torch.empty(
                (1, self.out_features), dtype=torch.bfloat16, requires_grad=False, device=dev))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        # Reshape input: [batch, seq_len, in_features] -> [batch*seq_len, in_features]
        original_shape = x.shape
        x_2d = x.view(-1, self.in_features)

        # Prepare dimensions for GEMM: C = A @ B
        # A: [M, K] = [batch*seq_len, in_features]
        # B: [K, N] = [in_features, out_features]  (weight.T)
        # C: [M, N] = [batch*seq_len, out_features]
        M = x_2d.shape[0]
        K = self.in_features
        N = self.out_features

        output = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
        unicore_core_cuda_module.unicore_GEMM_bf16(x_2d, self.weight.T.contiguous(), output, M, N, K)

        output = output.view(*original_shape[:-1], self.out_features)

        if self.bias is not None:
            output = output + self.bias

        return output

class UniCoreLinearFP16(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=False, dev='cuda'):
        super(UniCoreLinearFP16, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer('weight', torch.empty(self.out_features,
                                                   self.in_features, dtype=torch.float16, requires_grad=False, device=dev))

        if bias:
            self.register_buffer('bias', torch.empty(
                (1, self.out_features), dtype=torch.float16, requires_grad=False, device=dev))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        # Reshape input: [batch, seq_len, in_features] -> [batch*seq_len, in_features]
        original_shape = x.shape
        x_2d = x.view(-1, self.in_features)

        # Prepare dimensions for GEMM: C = A @ B
        # A: [M, K] = [batch*seq_len, in_features]
        # B: [K, N] = [in_features, out_features]  (weight.T)
        # C: [M, N] = [batch*seq_len, out_features]
        M = x_2d.shape[0]
        K = self.in_features
        N = self.out_features

        output = torch.empty((M, N), dtype=torch.float16, device=x.device)
        unicore_core_cuda_module.unicore_GEMM_fp16(x_2d, self.weight.T.contiguous(), output, M, N, K)

        output = output.view(*original_shape[:-1], self.out_features)

        if self.bias is not None:
            output = output + self.bias

        return output


class UniCoreLinearFP8(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=False,
        dev='cuda',
        layer_name='',
        fp8_compensation_method='unicore',
    ):
        super(UniCoreLinearFP8, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.layer_name = layer_name  # Store layer name for debugging
        self.fp8_compensation_method = fp8_compensation_method
        self.fp8_compensation_method_id = resolve_fp8_compensation_method(
            fp8_compensation_method
        )

        # Initialize weight parameter (BF16 for fallback)
        self.register_buffer('weight', torch.empty(self.out_features,
                                                   self.in_features, dtype=torch.bfloat16, requires_grad=False, device=dev))
        # Packed weight in uint8 FP8 format (for optimized kernel)
        self.register_buffer('weight_packed', torch.empty(self.out_features,
                                                          self.in_features, dtype=torch.uint8, requires_grad=False, device=dev))
        # per channel scaling factor
        self.register_buffer('wscales', torch.empty(self.out_features, 1, dtype=torch.bfloat16, requires_grad=False, device=dev))
        # Initialize bias parameter if required
        if bias:
            self.register_buffer('bias', torch.empty(
                (1, self.out_features), dtype=torch.bfloat16, requires_grad=False, device=dev))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        # Reshape input: [batch, seq_len, in_features] -> [batch*seq_len, in_features]
        original_shape = x.shape
        x_2d = x.view(-1, self.in_features)

        M = x_2d.shape[0]
        K = self.in_features
        N = self.out_features

        output = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
        # unicore_core_cuda_module.unicore_GEMM_fp8(x_2d, self.weight.T.contiguous(), output, M, N, K)
        output = output.view(*original_shape[:-1], self.out_features)

        if self.bias is not None:
            output = output + self.bias

        return output

    def forward_with_scale(self, x_q, x_scale):
        """
        Forward pass with quantized activation and per-token scale.
        Uses optimized FP8 GEMM with packed uint8 format.

        Computation:
        1. Pack x_q (BF16) -> x_packed (uint8)
        2. FP8 GEMM: output = x_packed @ weight_packed.T
        3. Dequantize: output = output * x_scale * w_scale

        Args:
            x_q: Quantized activation in BF16 (not yet packed to FP8)
                 Shape: [batch, seq_len, in_features]
            x_scale: Per-token scaling factor for activation dequantization
                     Shape: [batch, seq_len, 1] (broadcasts with output)
        Returns:
            Output tensor after GEMM and dequantization: [batch, seq_len, out_features]
        """
        # Reshape input
        original_shape = x_q.shape
        x_q_2d = x_q.view(-1, self.in_features)  # [M, K]

        # Pack activation to FP8 E4M3 uint8 format (online conversion)
        x_packed = pack_fp8_e4m3_from_bf16(x_q_2d)  # [M, K] uint8

        M = x_q_2d.shape[0]
        K = self.in_features
        N = self.out_features

        # FP8 GEMM with packed values
        output = torch.empty((M, N), dtype=torch.bfloat16, device=x_q.device)

        # TODO: Replace with optimized FP8 kernel
        unicore_core_cuda_module.unicore_GEMM_fp8(
            x_packed,
            self.weight_packed.T.contiguous(),
            output,
            M,
            N,
            K,
            self.fp8_compensation_method_id,
        )
        # For now, use fallback matmul with unpacked values
        # output = torch.matmul(x_q_2d, self.weight.T.to(x_q_2d.dtype))  # [M, N]

        # Reshape activation scale for broadcasting: [batch, seq_len, 1] -> [M, 1]
        x_scale_2d = x_scale.view(-1, 1) if x_scale.numel() > 1 else x_scale  # [M, 1]

        # Dequantize by multiplying both scales
        # x_scale: [M, 1], wscales: [N, 1] -> broadcasts to [M, N]
        output.mul_(x_scale_2d)  # [M, N] * [M, 1] -> [M, N]
        output.mul_(self.wscales.T)  # [M, N] * [1, N] -> [M, N]

        # Reshape back
        output = output.view(*original_shape[:-1], self.out_features)

        if self.bias is not None:
            output = output + self.bias.to(output.dtype)

        return output


class UniCoreLinearGroupFP8(torch.nn.Module):
    """
    only support per-group quantization.
    """
    def __init__(
        self,
        in_features,
        out_features,
        group_size=128,
        bias=False,
        dev='cuda',
        layer_name='',
        fp8_compensation_method='unicore',
    ):
        super(UniCoreLinearGroupFP8, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.layer_name = layer_name  # Store layer name for debugging
        self.fp8_compensation_method = fp8_compensation_method
        self.fp8_compensation_method_id = resolve_fp8_compensation_method(
            fp8_compensation_method
        )

        # Initialize weight parameter [N, K]
        self.register_buffer('weight', torch.empty(self.out_features,
                                                   self.in_features, dtype=torch.bfloat16, requires_grad=False, device=dev))
        # Packed weight in FP8 format [N, K] uint8
        self.register_buffer('weight_packed', torch.empty(self.out_features,
                                                          self.in_features, dtype=torch.uint8, requires_grad=False, device=dev))
        # per channel scaling factor [N, K//g]
        self.register_buffer('wscales', torch.empty(self.out_features, self.in_features // self.group_size, dtype=torch.bfloat16, requires_grad=False, device=dev))
        # Initialize bias parameter if required
        if bias:
            self.register_buffer('bias', torch.empty(
                (1, self.out_features), dtype=torch.bfloat16, requires_grad=False, device=dev))
        else:
            self.register_parameter('bias', None)

    def forward_with_scale(self, x_q, x_scale):
        """
        Forward pass with quantized activation and per-token scale.
        Computation: output = (x_q @ weight_q.T) * x_scale * w_scale

        Args:
            x_q: Quantized activation (in original dtype, not yet cast to fp8)
                 Shape: [batch, seq_len, in_features]
            x_scale: Per-token scaling factor for activation dequantization
                     Shape: [batch, seq_len, dim//g] (broadcasts with output)
        Returns:
            Output tensor after GEMM and dequantization: [batch, seq_len, out_features]
        """
        # Reshape input
        original_shape = x_q.shape
        x_q = x_q.view(-1, self.in_features)  # [M, K]

        # Pack activation to FP8 E4M3 uint8 format (online conversion)
        x_packed = pack_fp8_e4m3_from_bf16(x_q)  # [M, K] uint8

        # Reshape activation scale: support per-group
        # Per-group: x_scale shape [batch, seq_len, K//g] -> [M, K//g]
        x_scale = x_scale.view(-1, x_scale.shape[-1])  # [M, K//g]

        # Optimized FP8 GEMM kernel with group quantization
        # x_packed: [M, K] uint8, weight_packed: [N, K] uint8,
        # x_scale_2d: [M, K//g] bf16, wscales: [N, K//g] bf16
        M = x_q.shape[0]
        K = self.in_features
        N = self.out_features

        output = torch.empty((M, N), dtype=torch.bfloat16, device=x_q.device)
        # Use optimized grouped FP8 kernel
        unicore_core_cuda_module.unicore_GEMM_fp8_grouped(
            x_packed,                            # [M, K] uint8
            self.weight_packed.T.contiguous(),   # [K, N] uint8
            output,                              # [M, N] bf16
            x_scale,                          # [M, K//g] bf16
            self.wscales,                        # [N, K//g] bf16
            M, N, K, self.group_size, self.fp8_compensation_method_id
        )

        # output2 = torch.zeros((M, N), dtype=torch.bfloat16, device=x_q.device)
        # num_groups = K // self.group_size
        # for g in range(num_groups):
        #     # Extract the g-th group
        #     k_start = g * self.group_size
        #     k_end = (g + 1) * self.group_size

        #     # x_group = x_packed[:, k_start:k_end]  # [M, group_size]
        #     # w_group = self.weight_packed[:, k_start:k_end]  # [N, group_size]

        #     x_group = x_q[:, k_start:k_end]  # [M, group_size]
        #     w_group = self.weight[:, k_start:k_end]  # [N, group_size]

        #     # matmul: [M, group_size] @ [group_size, N] = [M, N]
        #     partial = torch.matmul(x_group, w_group.T)  # [M, N]
        #     # partial = torch.empty((M, N), dtype=torch.bfloat16, device=x_q.device)
        #     # unicore_core_cuda_module.unicore_GEMM_fp8(x_group, w_group.T.contiguous(), partial, M, N, self.group_size)
        #     # Dequantize by multiplying both per-group scales
        #     # x_scale[:, g]: [M] → broadcasts to [M, 1]
        #     # w_scale[:, g]: [N] → broadcasts to [1, N]
        #     partial = partial * x_scale[:, g:g+1]  # [M, N] * [M, 1]
        #     partial = partial * self.wscales[:, g:g+1].T  # [M, N] * [1, N]

        #     output2 += partial

        # Reshape back
        output = output.view(*original_shape[:-1], self.out_features)
        # output2 = output2.view(*original_shape[:-1], self.out_features)

        # Print layer information for debugging
        # layer_info = f"[Layer: {self.layer_name}]" if self.layer_name else "[Unknown Layer]"
        # print(f"\n{'='*80}")
        # print(f"{layer_info} UniCoreLinearGroupFP8 Forward")
        # print(f"Shape: [{M}, {K}] @ [{N}, {K}]^T -> [{M}, {N}]")
        # print(f"Group size: {self.group_size}, Num groups: {K // self.group_size}")
        # print(f"Input shape: {original_shape}, Output shape: {output.shape}")
        # print(f"{'='*80}")

        # print("output (CUDA kernel)", output)
        # print("output2 (PyTorch reference)", output2)
        # diff = torch.abs(output - output2)
        # max_diff = diff.max().item()
        # mean_diff = diff.mean().item()
        # rel_diff = (diff / (torch.abs(output2) + 1e-8)).mean().item()

        # print(f"Max absolute difference: {max_diff:.6e}")
        # print(f"Mean absolute difference: {mean_diff:.6e}")
        # print(f"Mean relative difference: {rel_diff:.6e}")


        # raise ValueError("CUDA kernel and PyTorch reference do not match")

        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        return output


class UniCoreLinearFP4(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=False, dev='cuda', layer_name=''):
        super(UniCoreLinearFP4, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.layer_name = layer_name

        self.register_buffer(
            'weight',
            torch.empty(
                self.out_features,
                self.in_features,
                dtype=torch.bfloat16,
                requires_grad=False,
                device=dev,
            ),
        )
        self.register_buffer(
            'weight_packed',
            torch.empty(
                self.out_features,
                self.in_features,
                dtype=torch.uint8,
                requires_grad=False,
                device=dev,
            ),
        )
        self.register_buffer(
            'wscales',
            torch.empty(
                self.out_features, 1, dtype=torch.bfloat16, requires_grad=False, device=dev
            ),
        )
        if bias:
            self.register_buffer(
                'bias',
                torch.empty((1, self.out_features), dtype=torch.bfloat16, requires_grad=False, device=dev),
            )
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        original_shape = x.shape
        x_2d = x.view(-1, self.in_features).to(torch.bfloat16)
        weight_dequant = self.weight.to(x_2d.device) * self.wscales.to(x_2d.device)
        output = torch.matmul(x_2d, weight_dequant.T)
        output = output.view(*original_shape[:-1], self.out_features)

        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        return output

    def forward_with_scale(self, x_q, x_scale):
        original_shape = x_q.shape
        x_q_2d = x_q.view(-1, self.in_features)
        x_mapped = map_fp4_to_fp8_lut_values_strict(x_q_2d)
        x_packed = pack_fp8_e4m3_from_bf16(x_mapped)

        M = x_q_2d.shape[0]
        K = self.in_features
        N = self.out_features

        output = torch.empty((M, N), dtype=torch.bfloat16, device=x_q.device)
        unicore_core_cuda_module.unicore_GEMM_fp4(
            x_packed,
            self.weight_packed.T.contiguous(),
            output,
            M,
            N,
            K,
        )

        output.mul_(x_scale.view(-1, 1))
        output.mul_(self.wscales.T)
        output = output.view(*original_shape[:-1], self.out_features)

        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        return output


class UniCoreLinearGroupFP4(torch.nn.Module):
    def __init__(self, in_features, out_features, group_size=128, bias=False, dev='cuda', layer_name=''):
        super(UniCoreLinearGroupFP4, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.layer_name = layer_name

        self.register_buffer(
            'weight',
            torch.empty(
                self.out_features,
                self.in_features,
                dtype=torch.bfloat16,
                requires_grad=False,
                device=dev,
            ),
        )
        self.register_buffer(
            'weight_packed',
            torch.empty(
                self.out_features,
                self.in_features,
                dtype=torch.uint8,
                requires_grad=False,
                device=dev,
            ),
        )
        self.register_buffer(
            'wscales',
            torch.empty(
                self.out_features,
                self.in_features // self.group_size,
                dtype=torch.bfloat16,
                requires_grad=False,
                device=dev,
            ),
        )
        if bias:
            self.register_buffer(
                'bias',
                torch.empty((1, self.out_features), dtype=torch.bfloat16, requires_grad=False, device=dev),
            )
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        original_shape = x.shape
        x_2d = x.view(-1, self.in_features).to(torch.bfloat16)
        weight = self.weight.to(x_2d.device)
        wscales = self.wscales.to(x_2d.device)
        weight_dequant = (
            weight.view(self.out_features, -1, self.group_size)
            * wscales.unsqueeze(-1)
        ).view(self.out_features, self.in_features)
        output = torch.matmul(x_2d, weight_dequant.T)
        output = output.view(*original_shape[:-1], self.out_features)

        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        return output

    def forward_with_scale(self, x_q, x_scale):
        original_shape = x_q.shape
        x_q_2d = x_q.view(-1, self.in_features)
        x_mapped = map_fp4_to_fp8_lut_values_strict(x_q_2d)
        x_packed = pack_fp8_e4m3_from_bf16(x_mapped)
        x_scale_2d = x_scale.view(-1, x_scale.shape[-1])

        M = x_q_2d.shape[0]
        K = self.in_features
        N = self.out_features
        num_groups = K // self.group_size

        output = torch.zeros((M, N), dtype=torch.bfloat16, device=x_q.device)
        for group_idx in range(num_groups):
            k_start = group_idx * self.group_size
            k_end = (group_idx + 1) * self.group_size
            partial = torch.empty((M, N), dtype=torch.bfloat16, device=x_q.device)
            unicore_core_cuda_module.unicore_GEMM_fp4(
                x_packed[:, k_start:k_end].contiguous(),
                self.weight_packed[:, k_start:k_end].T.contiguous(),
                partial,
                M,
                N,
                self.group_size,
            )
            partial.mul_(x_scale_2d[:, group_idx:group_idx + 1])
            partial.mul_(self.wscales[:, group_idx:group_idx + 1].T)
            output.add_(partial)

        output = output.view(*original_shape[:-1], self.out_features)
        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        return output
