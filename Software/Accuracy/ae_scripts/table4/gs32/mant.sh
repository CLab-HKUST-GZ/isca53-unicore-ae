#!/usr/bin/env bash

# Simple evaluation script for mant baseline on LM-Eval tasks.

set -euo pipefail

: "${device:=0}"
export CUDA_VISIBLE_DEVICES=${device}

models=(
  "meta-llama/Meta-Llama-3-8B"
  "Qwen/Qwen3-8B"
)

# Quantization sweep settings: "wq_bits wq_datatype wq_groupsize a_bits a_groupsize"
configs=(
  "4 mixed_mant 32 4 32"
)

for cfg in "${configs[@]}"; do
  read -r wq_bits wq_datatype wq_groupsize a_bits a_groupsize <<<"${cfg}"

  for model in "${models[@]}"; do
    echo "============================================"
    echo "Starting ${model} mant baseline GS32"
    echo "============================================"

    python llm_eval_tasks.py \
      --task_group group1 \
      --result_table table4 \
      --result_precision_tag gs32 \
      --result_method mant \
      --model "${model}" \
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
