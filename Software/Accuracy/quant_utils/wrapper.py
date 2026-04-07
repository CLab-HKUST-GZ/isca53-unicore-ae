import torch
import gc
from tqdm import tqdm

from .utils import get_parent_module
from unicore_kernel.unicore_core import *

def print_vram_usage():
    print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    print(torch.cuda.memory_summary())

def approximation_wrapper(model, args):
    # if 'llama' in model.config.architectures[0].lower():
    # bits = 8
    if args.wq_bits == 16 and args.a_bits == 16:
        for name, module in tqdm(model.named_modules()):
            if isinstance(module, torch.nn.Linear) and 'lm_head' not in name and 'output_layer' not in name:
                device = next(module.parameters()).device
                module.weight.data = module.weight.data.to('cpu')
                if module.bias is not None:
                    module.bias.data = module.bias.data.to('cpu')
                use_fp16 = hasattr(args, 'wq_datatype') and isinstance(args.wq_datatype, str) and args.wq_datatype.lower() == 'fp16'
                if use_fp16:
                    new_linear = UniCoreLinearFP16(module.in_features,
                                                    module.out_features,
                                                    module.bias is not None,
                                                    dev='cpu')
                else:
                    new_linear = UniCoreLinearBF16(module.in_features,
                                                    module.out_features,
                                                    module.bias is not None,
                                                    dev='cpu')

                with torch.no_grad():
                    copy_dtype = torch.float16 if use_fp16 else torch.bfloat16
                    new_linear.weight.copy_(module.weight.to(copy_dtype))
                    if module.bias is not None:
                        new_linear.bias.copy_(module.bias.to(copy_dtype))
                new_linear = new_linear.to(device)
                child_name = name.split('.')[-1]
                parent_module = get_parent_module(model, name)

                old_module = getattr(parent_module, child_name)
                setattr(parent_module, child_name, new_linear)
                del old_module

                gc.collect()
                torch.cuda.empty_cache()
    elif args.wq_bits == 8 and args.a_bits == 8:
        from .quant_weight import quant_fp8
        fp8_compensation_method = getattr(args, "fp8_compensation_method", "unicore")

        for name, module in tqdm(model.named_modules()):
            if isinstance(module, torch.nn.Linear) and 'lm_head' not in name and 'output_layer' not in name:
                device = next(module.parameters()).device
                module.weight.data = module.weight.data.to('cpu')
                if module.bias is not None:
                    module.bias.data = module.bias.data.to('cpu')

                # Quantize weight and get scale
                if args.wq_groupsize <= 0:
                    weight_q, weight_scale = quant_fp8(module.weight.data.to(torch.bfloat16),
                                                        group_size=-1,  # per-channel
                                                        return_scale=True)
                    new_linear = UniCoreLinearFP8(module.in_features,
                                                module.out_features,
                                                bias=module.bias is not None,
                                                dev='cpu',
                                                layer_name=name,
                                                fp8_compensation_method=fp8_compensation_method)  # Pass layer name for debugging
                else:
                    weight_q, weight_scale = quant_fp8(module.weight.data.to(torch.bfloat16),
                                                        group_size=args.wq_groupsize,  # per-group
                                                        return_scale=True)
                    new_linear = UniCoreLinearGroupFP8(module.in_features,
                                                        module.out_features,
                                                        group_size=args.wq_groupsize,
                                                        bias=module.bias is not None,
                                                        dev='cpu',
                                                        layer_name=name,
                                                        fp8_compensation_method=fp8_compensation_method)  # Pass layer name for debugging

                with torch.no_grad():
                    # Store BF16 quantized weight (for fallback)
                    new_linear.weight.copy_(weight_q)
                    # Pre-pack weight to uint8 FP8 format (optimization)
                    # if args.wq_groupsize <= 0:
                    from unicore_kernel.unicore_core import pack_fp8_e4m3_from_bf16
                    weight_packed = pack_fp8_e4m3_from_bf16(weight_q).cpu()
                    new_linear.weight_packed.copy_(weight_packed)
                    # Store per-channel/per-group scale
                    # For per-group: weight_scale shape is [out_features, in_features//g, 1], need to squeeze
                    new_linear.wscales.copy_(weight_scale.squeeze(-1) if weight_scale.dim() == 3 else weight_scale)
                    if module.bias is not None:
                        new_linear.bias.copy_(module.bias.to(torch.bfloat16))

                new_linear = new_linear.to(device)
                # Keep only packed weights + scales on GPU; offload BF16 fallback weights.
                new_linear.weight = new_linear.weight.to('cpu')
                child_name = name.split('.')[-1]
                parent_module = get_parent_module(model, name)

                old_module = getattr(parent_module, child_name)
                setattr(parent_module, child_name, new_linear)
                del old_module

                gc.collect()
                torch.cuda.empty_cache()
    elif args.wq_bits == 4 and args.a_bits == 4:
        from .quant_weight import quant_fp4

        for name, module in tqdm(model.named_modules()):
            if isinstance(module, torch.nn.Linear) and 'lm_head' not in name and 'output_layer' not in name:
                device = next(module.parameters()).device
                module.weight.data = module.weight.data.to('cpu')
                if module.bias is not None:
                    module.bias.data = module.bias.data.to('cpu')

                # Quantize weight and get scale
                if args.wq_groupsize <= 0:
                    weight_q, weight_scale = quant_fp4(module.weight.data.to(torch.bfloat16),
                                                        group_size=-1,  # per-channel
                                                        return_scale=True)
                    new_linear = UniCoreLinearFP4(module.in_features,
                                                module.out_features,
                                                bias=module.bias is not None,
                                                dev='cpu',
                                                layer_name=name)  # Pass layer name for debugging
                else:
                    weight_q, weight_scale = quant_fp4(module.weight.data.to(torch.bfloat16),
                                                        group_size=args.wq_groupsize,  # per-group
                                                        return_scale=True)
                    new_linear = UniCoreLinearGroupFP4(module.in_features,
                                                        module.out_features,
                                                        group_size=args.wq_groupsize,
                                                        bias=module.bias is not None,
                                                        dev='cpu',
                                                        layer_name=name)  # Pass layer name for debugging

                with torch.no_grad():
                    # Store BF16 quantized weight (for fallback)
                    new_linear.weight.copy_(weight_q)
                    # Map fp4 values into the kernel's uint8 encoding domain, then pack.
                    from unicore_kernel.unicore_core import (
                        map_fp4_to_fp8_lut_values_strict,
                        pack_fp8_e4m3_from_bf16,
                    )
                    weight_packed = pack_fp8_e4m3_from_bf16(
                        map_fp4_to_fp8_lut_values_strict(weight_q)
                    ).cpu()
                    new_linear.weight_packed.copy_(weight_packed)
                    # Store per-channel/per-group scale
                    # For per-group: weight_scale shape is [out_features, in_features//g, 1], need to squeeze
                    new_linear.wscales.copy_(weight_scale.squeeze(-1) if weight_scale.dim() == 3 else weight_scale)
                    if module.bias is not None:
                        new_linear.bias.copy_(module.bias.to(torch.bfloat16))

                new_linear = new_linear.to(device)
                # Keep only packed weights + scales on GPU; offload BF16 fallback weights.
                new_linear.weight = new_linear.weight.to('cpu')
                child_name = name.split('.')[-1]
                parent_module = get_parent_module(model, name)

                old_module = getattr(parent_module, child_name)
                setattr(parent_module, child_name, new_linear)
                del old_module

                gc.collect()
                torch.cuda.empty_cache()
    print(model)

def attention_wrapper(model, args):
    """
    Replace standard attention modules with quantization-aware attention modules.
    Follows the same pattern as approximation_wrapper for consistency.
    """
    if 'llama' in model.config.architectures[0].lower():
        from .attn.sw.llama_attn import LlamaAttention
        print("Replacing attention modules with quantization-aware LlamaAttention...")
        for name, module in tqdm(list(model.named_modules())):
            if name.endswith('self_attn') and hasattr(module, 'q_proj'):
                device = next(module.parameters()).device
                dtype = next(module.parameters()).dtype

                new_attn = LlamaAttention(
                    config=model.config,
                    layer_idx=module.layer_idx
                ).to('cpu')

                # Transfer weights from old attention to new attention
                # Move to CPU first to save GPU memory during replacement
                old_state_dict = {k: v.to('cpu') for k, v in module.state_dict().items()}
                new_attn.load_state_dict(old_state_dict, strict=False)

                new_attn = new_attn.to(dtype).to(device)

                child_name = name.split('.')[-1]
                parent_module = get_parent_module(model, name)

                old_module = getattr(parent_module, child_name)
                setattr(parent_module, child_name, new_attn)
                del old_module
                gc.collect()
                torch.cuda.empty_cache()
    elif 'qwen3' in model.config.architectures[0].lower():
        from .attn.sw.qwen3_attn import Qwen3Attention
        print("Replacing attention modules with quantization-aware Qwen3Attention...")
        for name, module in tqdm(list(model.named_modules())):
            if name.endswith('self_attn') and hasattr(module, 'q_proj'):
                device = next(module.parameters()).device
                dtype = next(module.parameters()).dtype

                new_attn = Qwen3Attention(
                    config=model.config,
                    layer_idx=module.layer_idx
                ).to('cpu')

                # Transfer weights from old attention to new attention
                # Move to CPU first to save GPU memory during replacement
                old_state_dict = {k: v.to('cpu') for k, v in module.state_dict().items()}
                new_attn.load_state_dict(old_state_dict, strict=False)

                new_attn = new_attn.to(dtype).to(device)

                child_name = name.split('.')[-1]
                parent_module = get_parent_module(model, name)

                old_module = getattr(parent_module, child_name)
                setattr(parent_module, child_name, new_attn)
                del old_module
                gc.collect()
                torch.cuda.empty_cache()

        print("Attention module replacement completed.")
    elif 'opt' in model.config.architectures[0].lower():
        from .attn.sw.opt_attn import OPTAttention
        print("Replacing attention modules with quantization-aware OPTAttention...")
        for name, module in tqdm(list(model.named_modules())):
            if name.endswith('self_attn') and hasattr(module, 'q_proj'):
                device = next(module.parameters()).device
                dtype = next(module.parameters()).dtype

                new_attn = OPTAttention(
                    config=model.config,
                    layer_idx=module.layer_idx
                ).to('cpu')

                # Transfer weights from old attention to new attention
                # Move to CPU first to save GPU memory during replacement
                old_state_dict = {k: v.to('cpu') for k, v in module.state_dict().items()}
                new_attn.load_state_dict(old_state_dict, strict=False)

                new_attn = new_attn.to(dtype).to(device)

                child_name = name.split('.')[-1]
                parent_module = get_parent_module(model, name)

                old_module = getattr(parent_module, child_name)
                setattr(parent_module, child_name, new_attn)
                del old_module
                gc.collect()
                torch.cuda.empty_cache()

        print("Attention module replacement completed.")
    else:
        print(f"Attention wrapper not implemented for {model.config.architectures[0]}")
