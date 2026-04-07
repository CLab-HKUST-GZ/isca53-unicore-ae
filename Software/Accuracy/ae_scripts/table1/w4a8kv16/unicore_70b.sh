#!/usr/bin/env bash

# Simple evaluation script for UniCore FP4 evaluation on WikiText.

set -euo pipefail

: "${device:=0,1,2,3}"
export CUDA_VISIBLE_DEVICES=${device}

# Choose a model. Update to your local path if needed.

models=(
  "meta-llama/Llama-2-70b-hf"
)

# Quantization settings

# Budgeted DSE: use 32 types (=16 a pairs).
# Override examples:
#   BUDGET=16 ./scripts/wikitext_ppl_dse.sh
#   COUNT=1  ./scripts/wikitext_ppl_dse.sh   # use count-based variant
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

quant_datatype="fp4"

# Quantization sweep settings: "wq_bits wq_groupsize a_bits a_groupsize"
configs=(
  "4 32 8 -1"
)

for cfg in "${configs[@]}"; do
  read -r wq_bits wq_groupsize a_bits a_groupsize <<< "${cfg}"

  for model in "${models[@]}"; do
    echo "Running UniCore FP4 evaluation on WikiText with settings:"
    echo "  device           = ${CUDA_VISIBLE_DEVICES}"
    echo "  model            = ${model}"
    echo "  wq_bits          = ${wq_bits}"
    echo "  wq_datatype      = ${quant_datatype}"
    echo "  wq_groupsize     = ${wq_groupsize}"
    echo "  a_bits           = ${a_bits}"
    echo "  a_groupsize      = ${a_groupsize}"

    python llm_eval_wikitext.py \
      --model ${model} \
      --result_precision_tag 4a8kv16 \
      --result_method unicore \
      --wq_bits ${wq_bits} \
      --wq_datatype ${quant_datatype} \
      --wq_groupsize ${wq_groupsize} \
      ${wq_search_args} \
      ${wq_offload_args} \
      --a_bits ${a_bits} \
      --a_groupsize ${a_groupsize} \
      --a_fpq \


    echo "Finished ${model} ${quant_datatype} ${wq_bits} ${wq_groupsize} ${a_bits} ${a_groupsize}"
    echo ""
  done
done

echo "All models completed!"
