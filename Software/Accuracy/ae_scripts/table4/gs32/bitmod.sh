#!/usr/bin/env bash

# WikiText-2 PPL evaluation script for BitMoD baseline.

set -euo pipefail

: "${device:=0}"
export CUDA_VISIBLE_DEVICES=${device}

models=(
  "meta-llama/Llama-2-7b-hf"
)

# Quantization sweep settings: "wq_bits wq_datatype wq_groupsize a_bits a_groupsize"
configs=(
  "4 mixed_bitmod 32 4 32"
)

for cfg in "${configs[@]}"; do
  read -r wq_bits wq_datatype wq_groupsize a_bits a_groupsize <<<"${cfg}"

  for model in "${models[@]}"; do
    echo "============================================"
    echo "Starting ${model} bitmod baseline GS32"
    echo "============================================"

    python llm_eval_wikitext_fpma.py \
      --result_table table4 \
      --result_precision_tag gs32 \
      --result_method bitmod \
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
