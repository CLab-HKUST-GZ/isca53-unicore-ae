#!/usr/bin/env bash

# WikiText-2 PPL evaluation script for UniCore FP4.

set -euo pipefail

: "${device:=0}"
export CUDA_VISIBLE_DEVICES=${device}

models=(
  "meta-llama/Llama-2-7b-hf"
)

BUDGET=${BUDGET:-32}

# Memory-optimized mixed-search knobs (keep quantization logic unchanged).
WQ_SEARCH_ON_CPU=${WQ_SEARCH_ON_CPU:-0}
WQ_LAYERWISE_OFFLOAD=${WQ_LAYERWISE_OFFLOAD:-0}

wq_search_args="--no-wq_search_on_cpu"
if [ "${WQ_SEARCH_ON_CPU}" = "1" ]; then
  wq_search_args="--wq_search_on_cpu"
fi

wq_offload_args="--no-wq_layerwise_offload"
if [ "${WQ_LAYERWISE_OFFLOAD}" = "1" ]; then
  wq_offload_args="--wq_layerwise_offload"
fi

quant_datatype="mixed_unicore_auto_b${BUDGET}"

# Quantization sweep settings: "wq_bits wq_groupsize a_bits a_groupsize"
configs=(
  "4 64 4 64"
)

for cfg in "${configs[@]}"; do
  read -r wq_bits wq_groupsize a_bits a_groupsize <<< "${cfg}"

  for model in "${models[@]}"; do
    echo "============================================"
    echo "Starting ${model} UniCore FP4 GS64"
    echo "============================================"

    python llm_eval_wikitext_fpma.py \
      --result_table table4 \
      --result_precision_tag gs64 \
      --result_method unicore \
      --model "${model}" \
      --wq_bits ${wq_bits} \
      --wq_datatype ${quant_datatype} \
      --wq_groupsize ${wq_groupsize} \
      ${wq_search_args} \
      ${wq_offload_args} \
      --a_bits ${a_bits} \
      --a_groupsize ${a_groupsize} \
      --a_fpq

    echo "Finished ${model} ${quant_datatype} ${wq_bits} ${wq_groupsize} ${a_bits} ${a_groupsize}"
    echo ""
  done
done

echo "All models completed!"
