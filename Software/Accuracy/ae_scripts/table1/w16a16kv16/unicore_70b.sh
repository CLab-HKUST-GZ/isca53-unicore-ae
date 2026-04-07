#!/usr/bin/env bash

set -euo pipefail

: "${device:=0,1,2,3}"
export CUDA_VISIBLE_DEVICES=${device}

models=(
  "meta-llama/Llama-2-70b-hf"
)

configs=(
  "16 bf16 16"
)

for cfg in "${configs[@]}"; do
  read -r wq_bits wq_datatype a_bits <<<"${cfg}"

  for model in "${models[@]}"; do
    echo "============================================"
    echo "Starting ${model} UniCore BF16"
    echo "============================================"

    python llm_eval_wikitext_fpma.py \
      --model "${model}" \
      --result_precision_tag 16a16kv16 \
      --result_method unicore \
      --wq_datatype "${wq_datatype}" \
      --wq_bits "${wq_bits}" \
      --a_bits "${a_bits}"

    echo "Finished ${model} ${wq_datatype} ${wq_bits}-bit"
    echo ""
  done
done

echo "All models completed!"
