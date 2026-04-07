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
  "4 int4 32 4 32"
)

for cfg in "${configs[@]}"; do
  read -r wq_bits wq_datatype wq_groupsize a_bits a_groupsize <<<"${cfg}"

  for model in "${models[@]}"; do
    echo "============================================"
    echo "Starting ${model} INT4 baseline W4A4KV16"
    echo "============================================"

    python llm_eval_wikitext.py \
      --model "${model}" \
      --result_precision_tag 4a4kv16 \
      --result_method int \
      --wq_datatype "${wq_datatype}" \
      --wq_bits "${wq_bits}" \
      --wq_groupsize "${wq_groupsize}" \
      --a_bits "${a_bits}" \
      --a_groupsize "${a_groupsize}" \
      --no-a_fpq

    echo "Finished ${model} ${wq_datatype} wq=${wq_bits} g=${wq_groupsize} aq=${a_bits} ag=${a_groupsize}"
    echo ""
  done
done

echo "All models completed!"
