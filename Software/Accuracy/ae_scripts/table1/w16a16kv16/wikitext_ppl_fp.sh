#!/usr/bin/env bash

set -euo pipefail

: "${device:=0}"
export CUDA_VISIBLE_DEVICES=${device}

models=(
  "facebook/opt-6.7b"
  "meta-llama/Llama-2-7b-hf"
  "meta-llama/Meta-Llama-3-8B"
  "Qwen/Qwen3-8B"
  "Qwen/Qwen3-14B"
)

configs=(
  "16 fp16 16"
)

for cfg in "${configs[@]}"; do
  read -r wq_bits wq_datatype a_bits <<<"${cfg}"

  for model in "${models[@]}"; do
    echo "============================================"
    echo "Starting ${model} FP16 baseline"
    echo "============================================"

    python llm_eval_wikitext.py \
      --model "${model}" \
      --result_precision_tag 16a16kv16 \
      --result_method fp \
      --wq_datatype "${wq_datatype}" \
      --wq_bits "${wq_bits}" \
      --a_bits "${a_bits}"

    echo "Finished ${model} ${wq_datatype} ${wq_bits}-bit"
    echo ""
  done
done

echo "All models completed!"
