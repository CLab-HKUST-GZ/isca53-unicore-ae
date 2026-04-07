#!/usr/bin/env bash

set -euo pipefail

: "${device:=0,1,2,3}"
export CUDA_VISIBLE_DEVICES=${device}

models=(
  "meta-llama/Llama-2-70b-hf"
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
      --result_precision_tag 4a4kv4 \
      --result_method int \
      --wq_datatype "${wq_datatype}" \
      --wq_bits "${wq_bits}" \
      --wq_groupsize "${wq_groupsize}" \
      --a_bits "${a_bits}" \
      --a_groupsize "${a_groupsize}" \
      --no-a_fpq \
      --q_bits 4 --q_groupsize 32 --no-q_fpq \
      --k_bits 4 --k_groupsize 32 --no-k_fpq \
      --v_bits 4 --v_groupsize 32 --v_fpq \
      --attnw_bits 4 --attnw_groupsize 32 --no-attnw_fpq \

    echo "Finished ${model} ${wq_datatype} wq=${wq_bits} g=${wq_groupsize} aq=${a_bits} ag=${a_groupsize}"
    echo ""
  done
done

echo "All models completed!"
