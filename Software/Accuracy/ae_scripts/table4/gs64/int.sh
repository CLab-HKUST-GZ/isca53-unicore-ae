#!/usr/bin/env bash

# WikiText-2 PPL evaluation script for INT4 baseline.

set -euo pipefail

: "${device:=0}"
export CUDA_VISIBLE_DEVICES=${device}

models=(
  "meta-llama/Llama-2-7b-hf"
)

# Quantization sweep settings: "wq_bits wq_datatype wq_groupsize a_bits a_groupsize"
configs=(
  "4 int4 64 4 64"
)

for cfg in "${configs[@]}"; do
  read -r wq_bits wq_datatype wq_groupsize a_bits a_groupsize <<<"${cfg}"

  for model in "${models[@]}"; do
    echo "============================================"
    echo "Starting ${model} INT4 baseline GS64"
    echo "============================================"

    python llm_eval_wikitext_fpma.py \
      --result_table table4 \
      --result_precision_tag gs64 \
      --result_method int \
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
