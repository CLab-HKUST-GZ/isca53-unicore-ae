import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

import random, argparse
from tqdm import tqdm

from quant_utils.quant_weight import quant_model
from quant_utils.write_results import write_results
from quant_utils.quant_utils import add_actquant, add_aq

from quant_utils.wrapper import attention_wrapper


parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument(
    "--model", "-m", type=str, default="hf", help="Name of model e.g. `hf`"
)
parser.add_argument(
    "--result_table", type=str, default="table1",
    help="Optional results table name, e.g. table1 or table3",
)
parser.add_argument(
    "--result_precision_tag", type=str, default="",
    help="Optional results directory prefix, e.g. 4a4kv16",
)
parser.add_argument(
    "--result_method", type=str, default="",
    help="Optional results directory method name, e.g. unicore/int/bitmod",
)
parser.add_argument(
    "--wq_datatype", type=str, default="", help="The weight datatype for weight quantization",
)
parser.add_argument(
    "--wq_bits", type=int, default=16, help="The weight precision for weight quantization",
)
parser.add_argument(
    "--wq_groupsize", type=int, default=None, help="The quantization group size for weight quantization",
)
parser.add_argument(
    "--wq_search_on_cpu", action=argparse.BooleanOptionalAction, default=False,
    help="Run mixed/search-based weight quantization on CPU to reduce GPU peak memory.",
)
parser.add_argument(
    "--wq_layerwise_offload", action=argparse.BooleanOptionalAction, default=True,
    help="When --wq_search_on_cpu is enabled, offload each Linear weight to CPU before search and move quantized weight back.",
)
parser.add_argument(
    "--a_bits", type=int, default=16, help="The activation precision for activation quantization")
parser.add_argument(
    "--a_groupsize", type=int, default=128, help="The quantization group size for activation quantization")
parser.add_argument(
    "--a_fpq", action=argparse.BooleanOptionalAction, default=True,
    help="Apply fpq activation quantization")
parser.add_argument(
    "--a_asym", action=argparse.BooleanOptionalAction, default=False,
    help="Apply asymmetric activation quantization")
parser.add_argument(
    "--a_clip_ratio", type=float, default=1.0,
    help="Clip ratio for activation quantization. new_max = max * clip_ratio")
parser.add_argument(
    "--aq_datatype", type=str, default="unicore", help="The activation datatype for weight-only quantization",
)

# Attention QKV quantization arguments
parser.add_argument(
    "--q_bits", type=int, default=16, help="Query quantization precision (default: 16, no quantization)")
parser.add_argument(
    "--q_groupsize", type=int, default=128, help="Query quantization group size")
parser.add_argument(
    "--q_asym", action=argparse.BooleanOptionalAction, default=False,
    help="Apply asymmetric query quantization")
parser.add_argument(
    "--q_clip_ratio", type=float, default=1.0, help="Query quantization clip ratio")
parser.add_argument(
    "--q_fpq", action=argparse.BooleanOptionalAction, default=False,
    help="Apply floating-point quantization for Query")
parser.add_argument(
    "--q_adaptive", action=argparse.BooleanOptionalAction, default=False,
    help="Use adaptive codebook selection for Query (requires q_fpq=True)")
parser.add_argument(
    "--k_bits", type=int, default=16, help="Key quantization precision (default: 16, no quantization)")
parser.add_argument(
    "--k_groupsize", type=int, default=128, help="Key quantization group size")
parser.add_argument(
    "--k_asym", action=argparse.BooleanOptionalAction, default=False,
    help="Apply asymmetric key quantization")
parser.add_argument(
    "--k_clip_ratio", type=float, default=1.0, help="Key quantization clip ratio")
parser.add_argument(
    "--k_fpq", action=argparse.BooleanOptionalAction, default=False,
    help="Apply floating-point quantization for Key")
parser.add_argument(
    "--k_adaptive", action=argparse.BooleanOptionalAction, default=False,
    help="Use adaptive codebook selection for Key (requires k_fpq=True)")
parser.add_argument(
    "--v_bits", type=int, default=16, help="Value quantization precision (default: 16, no quantization)")
parser.add_argument(
    "--v_groupsize", type=int, default=128, help="Value quantization group size")
parser.add_argument(
    "--v_asym", action=argparse.BooleanOptionalAction, default=False,
    help="Apply asymmetric value quantization")
parser.add_argument(
    "--v_clip_ratio", type=float, default=1.0, help="Value quantization clip ratio")
parser.add_argument(
    "--v_fpq", action=argparse.BooleanOptionalAction, default=False,
    help="Apply floating-point quantization for Value")
parser.add_argument(
    "--v_adaptive", action=argparse.BooleanOptionalAction, default=False,
    help="Use adaptive codebook selection for Value (requires v_fpq=True)")
parser.add_argument(
    "--attnw_bits", type=int, default=16, help="Attention weight quantization precision (default: 16, no quantization)")
parser.add_argument(
    "--attnw_groupsize", type=int, default=128, help="Attention weight quantization group size")
parser.add_argument(
    "--attnw_asym", action=argparse.BooleanOptionalAction, default=False,
    help="Apply asymmetric attention weight quantization")
parser.add_argument(
    "--attnw_clip_ratio", type=float, default=1.0, help="Attention weight quantization clip ratio")
parser.add_argument(
    "--attnw_fpq", action=argparse.BooleanOptionalAction, default=False,
    help="Apply floating-point quantization for Attention weight")

# Smooth K arguments
parser.add_argument(
    "--smooth_k", action=argparse.BooleanOptionalAction, default=False,
    help="Enable smooth K to reduce per-channel outliers before quantization")
parser.add_argument(
    "--smooth_k_strategy", type=str, default='per_channel', choices=['per_channel', 'per_token'],
    help="Smooth K strategy: 'per_channel' (global mean per channel) or 'per_token' (mean per token)")

args = parser.parse_args()
model_str = args.model
wq_bits = args.wq_bits
wq_groupsize = args.wq_groupsize
wq_datatype = args.wq_datatype
if not wq_datatype and wq_bits >= 16:
    wq_datatype = "bf16"

torch.set_grad_enabled(False)
torch.manual_seed(0)
random.seed(0)

if args.wq_bits < 16:
    model = AutoModelForCausalLM.from_pretrained(model_str, torch_dtype=torch.bfloat16, attn_implementation="eager", low_cpu_mem_usage=True, device_map='auto', trust_remote_code=True)
    quant_model(
        model,
        wq_bits,
        wq_datatype,
        wq_groupsize,
        search_on_cpu=args.wq_search_on_cpu,
        layerwise_offload=args.wq_layerwise_offload,
    )
else:
    model_dtype = torch.float16 if wq_datatype == "fp16" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_str, torch_dtype=model_dtype, attn_implementation="eager", low_cpu_mem_usage=True, device_map='auto', trust_remote_code=True)

if any(bits < 16 for bits in (args.q_bits, args.k_bits, args.v_bits, args.attnw_bits)):
    attention_wrapper(model, args)  # Replace attention modules with quantization-aware versions
add_actquant(model, use_unicore=False)
add_aq(model, args)           
model.seqlen = 2048
model = model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_str, use_fast=False, trust_remote_code=True)
testenc = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
testenc = tokenizer("\n\n".join(testenc["text"]), return_tensors="pt")
testenc = testenc.input_ids.to(model.device)
nsamples = testenc.numel() // model.seqlen
loss_fct = torch.nn.CrossEntropyLoss()

nlls = []
for i in tqdm(range(nsamples), desc="evaluating..."):
    batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(
        model.device
    )
    with torch.no_grad():
        lm_logits = model(batch).logits
    shift_logits = lm_logits[:, :-1, :].contiguous().float()
    shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )
    neg_log_likelihood = loss.float() * model.seqlen
    nlls.append(neg_log_likelihood)

ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
print(f'wikitext perplexity: {ppl.item()}')

write_results(
    ppl.item(),
    model_str,
    "wikitext",
    wq_bits,
    wq_datatype,
    wq_groupsize,
    result_table=args.result_table,
    result_precision_tag=args.result_precision_tag,
    result_method=args.result_method,
)
