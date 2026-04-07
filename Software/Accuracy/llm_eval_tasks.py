import argparse
import math
import os
import random

import lm_eval
import torch
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table
from transformers import AutoModelForCausalLM

from quant_utils.quant_utils import add_actquant, add_aq
from quant_utils.quant_weight import quant_model
from quant_utils.wrapper import approximation_wrapper, attention_wrapper

TASK_GROUPS = {
    "group1": {
        "tasks": ["piqa", "hellaswag", "winogrande", "arc_easy"],
        "batch_size": 32,
        "num_fewshot": 0,
    },
    "group2": {"tasks": ["gsm8k"], "batch_size": 32, "num_fewshot": 8},
    "group3": {"tasks": ["mmlu"], "batch_size": 8, "num_fewshot": 5},
    "group4": {"tasks": ["wikitext"], "batch_size": 2, "num_fewshot": 0},
    "group5": {
        "tasks": ["truthfulqa_mc2", "triviaqa"],
        "batch_size": 32,
        "num_fewshot": 0,
    },
}


def get_result_path(
    model_name: str,
    result_table: str,
    precision_tag: str,
    method: str,
) -> str:
    parts = [part for part in model_name.replace("\\", "/").split("/") if part]
    if len(parts) >= 2:
        model_file = f"{parts[-2]}__{parts[-1]}.txt"
    elif len(parts) == 1:
        model_file = f"{parts[0]}__.txt"
    else:
        model_file = "model__.txt"
    return f"./results/{result_table}/{precision_tag}/{method}/{model_file}"


parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument(
    "--model", "-m", type=str, default="hf", help="Name of model e.g. `hf`"
)
parser.add_argument(
    "--result_table",
    type=str,
    default="table2",
    help="Optional results table name, e.g. table2 or table4",
)
parser.add_argument(
    "--result_precision_tag",
    type=str,
    default="",
    help="Optional results directory prefix, e.g. 4a4kv16",
)
parser.add_argument(
    "--result_method",
    type=str,
    default="",
    help="Optional results directory method name, e.g. unicore/int/bitmod",
)
parser.add_argument(
    "--task_group",
    type=str,
    choices=sorted(TASK_GROUPS.keys()),
    default="group1",
    help="LM-Eval task bundle to run",
)
parser.add_argument(
    "--use_unicore",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Enable the UniCore path used by table2 unicore baselines",
)
parser.add_argument(
    "--wq_datatype",
    type=str,
    default="",
    help="The weight datatype for weight quantization",
)
parser.add_argument(
    "--wq_bits",
    type=int,
    default=4,
    help="The weight precision for weight quantization",
)
parser.add_argument(
    "--wq_groupsize",
    type=int,
    default=None,
    help="The quantization group size for weight quantization",
)
parser.add_argument(
    "--wq_search_on_cpu",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Run mixed-weight search on CPU to save GPU memory",
)
parser.add_argument(
    "--wq_layerwise_offload",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Offload layers during mixed-weight search to reduce peak memory",
)
parser.add_argument(
    "--a_bits", type=int, default=4, help="The activation precision for activation quantization"
)
parser.add_argument(
    "--a_groupsize",
    type=int,
    default=32,
    help="The quantization group size for activation quantization",
)
parser.add_argument(
    "--a_fpq",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Apply fpq activation quantization",
)
parser.add_argument(
    "--a_asym",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Apply asymmetric activation quantization",
)
parser.add_argument(
    "--a_clip_ratio",
    type=float,
    default=1.0,
    help="Clip ratio for activation quantization. new_max = max * clip_ratio",
)
parser.add_argument(
    "--aq_datatype",
    type=str,
    default="unicore",
    help="The activation datatype for weight-only quantization",
)

parser.add_argument(
    "--q_bits",
    type=int,
    default=16,
    help="Query quantization precision (default: 16, no quantization)",
)
parser.add_argument(
    "--q_groupsize", type=int, default=128, help="Query quantization group size"
)
parser.add_argument(
    "--q_asym",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Apply asymmetric query quantization",
)
parser.add_argument(
    "--q_clip_ratio", type=float, default=1.0, help="Query quantization clip ratio"
)
parser.add_argument(
    "--q_fpq",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Apply floating-point quantization for Query",
)
parser.add_argument(
    "--q_adaptive",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Use adaptive codebook selection for Query (requires q_fpq=True)",
)
parser.add_argument(
    "--k_bits",
    type=int,
    default=16,
    help="Key quantization precision (default: 16, no quantization)",
)
parser.add_argument(
    "--k_groupsize", type=int, default=128, help="Key quantization group size"
)
parser.add_argument(
    "--k_asym",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Apply asymmetric key quantization",
)
parser.add_argument(
    "--k_clip_ratio", type=float, default=1.0, help="Key quantization clip ratio"
)
parser.add_argument(
    "--k_fpq",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Apply floating-point quantization for Key",
)
parser.add_argument(
    "--k_adaptive",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Use adaptive codebook selection for Key (requires k_fpq=True)",
)
parser.add_argument(
    "--v_bits",
    type=int,
    default=16,
    help="Value quantization precision (default: 16, no quantization)",
)
parser.add_argument(
    "--v_groupsize", type=int, default=128, help="Value quantization group size"
)
parser.add_argument(
    "--v_asym",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Apply asymmetric value quantization",
)
parser.add_argument(
    "--v_clip_ratio", type=float, default=1.0, help="Value quantization clip ratio"
)
parser.add_argument(
    "--v_fpq",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Apply floating-point quantization for Value",
)
parser.add_argument(
    "--v_adaptive",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Use adaptive codebook selection for Value (requires v_fpq=True)",
)
parser.add_argument(
    "--attnw_bits",
    type=int,
    default=16,
    help="Attention weight quantization precision (default: 16, no quantization)",
)
parser.add_argument(
    "--attnw_groupsize",
    type=int,
    default=128,
    help="Attention weight quantization group size",
)
parser.add_argument(
    "--attnw_asym",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Apply asymmetric attention weight quantization",
)
parser.add_argument(
    "--attnw_clip_ratio",
    type=float,
    default=1.0,
    help="Attention weight quantization clip ratio",
)
parser.add_argument(
    "--attnw_fpq",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Apply floating-point quantization for Attention weight",
)
parser.add_argument(
    "--smooth_k",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Enable smooth K to reduce per-channel outliers before quantization",
)
parser.add_argument(
    "--smooth_k_strategy",
    type=str,
    default="per_channel",
    choices=["per_channel", "per_token"],
    help="Smooth K strategy: 'per_channel' or 'per_token'",
)

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

model_dtype = torch.float16 if wq_bits >= 16 and wq_datatype == "fp16" else torch.bfloat16
model = AutoModelForCausalLM.from_pretrained(
    model_str,
    torch_dtype=model_dtype,
    attn_implementation="eager",
    low_cpu_mem_usage=True,
    device_map="auto",
    trust_remote_code=True,
)

if wq_bits < 16:
    quant_model(
        model,
        wq_bits,
        wq_datatype,
        wq_groupsize,
        search_on_cpu=args.wq_search_on_cpu,
        layerwise_offload=args.wq_layerwise_offload,
    )
elif args.use_unicore:
    approximation_wrapper(model, args)

if any(bits < 16 for bits in (args.q_bits, args.k_bits, args.v_bits, args.attnw_bits)):
    attention_wrapper(model, args)

add_actquant(model, use_unicore=args.use_unicore)
add_aq(model, args)
model = model.eval()

task_group = TASK_GROUPS[args.task_group]
lm = HFLM(model, backend="causal", batch_size=task_group["batch_size"])
results = lm_eval.simple_evaluate(
    lm,
    tasks=task_group["tasks"],
    num_fewshot=task_group["num_fewshot"],
    batch_size=task_group["batch_size"],
    gen_kwargs="do_sample=False",
)

metric_vals = {}
std_vals = {}
for task, result in results["results"].items():
    if task == "AVERAGE":
        continue
    if "acc_norm,none" in result:
        acc_key = "acc_norm,none"
        stderr_key = "acc_norm_stderr,none"
    else:
        acc_key = "acc,none"
        stderr_key = "acc_stderr,none"

    metric_vals[task] = result[acc_key]
    std_vals[task] = result.get(stderr_key, 0.0)

mean_acc_val = round(sum(metric_vals.values()) / len(metric_vals), 4)
mean_std_val = round(
    math.sqrt(sum(std ** 2 for std in std_vals.values())) / len(std_vals),
    4,
)

results["results"]["AVERAGE"] = {
    "acc,none": mean_acc_val,
    "acc_stderr,none": mean_std_val,
}
results["versions"]["AVERAGE"] = "macro_avg_over_tasks"

table = make_table(results)
print("Evaluation Results:")
print(table)

if args.result_precision_tag and args.result_method:
    output_path = get_result_path(
        model_str,
        args.result_table,
        args.result_precision_tag,
        args.result_method,
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as handle:
        handle.write(f"model: {args.model}\n")
        handle.write(f"task_group: {args.task_group}\n")
        handle.write(table)
        handle.write("\n")
    print(f"Successfully written results to {output_path}.\n")
else:
    os.makedirs("./results", exist_ok=True)
    with open("./results/lm_eval_results.txt", "a") as handle:
        handle.write(f"model: {args.model}\n")
        handle.write(f"task_group: {args.task_group}\n")
        handle.write(table)
        handle.write("\n")
