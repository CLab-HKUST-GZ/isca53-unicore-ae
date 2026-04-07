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
  "8 fp8 -1 8 -1"
)

for cfg in "${configs[@]}"; do
  read -r wq_bits wq_datatype wq_groupsize a_bits a_groupsize <<<"${cfg}"

  for model in "${models[@]}"; do
    echo "============================================"
    echo "Starting ${model} UniCore FP8 W8A8KV16"
    echo "============================================"

    python llm_eval_wikitext_fpma.py \
      --model "${model}" \
      --result_table table3 \
      --result_precision_tag 8 \
      --result_method unicore \
      --fp8_compensation_method unicore \
      --wq_datatype "${wq_datatype}" \
      --wq_bits "${wq_bits}" \
      --wq_groupsize "${wq_groupsize}" \
      --a_bits "${a_bits}" \
      --a_groupsize "${a_groupsize}" \
      --a_fpq

    echo "Finished ${model} ${wq_datatype} wq=${wq_bits} aq=${a_bits} comp=unicore"
    echo ""
  done
done

echo "All models completed!"
