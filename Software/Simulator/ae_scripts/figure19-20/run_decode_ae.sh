#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../common.sh"

MEMORY="both"
CXT_LEN="8192"
BATCH_SIZE="128"
DRY_RUN=0
SKIP_DRAMSIM=0

usage() {
  cat <<'EOF'
Usage: run_decode_ae.sh [options]

Options:
  --memory <ddr4|hbm2|both>   Which memory system(s) to run. Default: both
  --cxt-len <int>             Context length to plot. Default: 8192
  --batch-size <int>          Decode batch size. Default: 128
  --skip-dramsim              Skip DRAMsim3 build and summary generation.
  --dry-run                   Print commands without executing them.
  -h, --help                  Show this help message.

Outputs:
  - Decode CSVs under `results_*_ctx_sweep_decode_bs<batch>/`
  - Decode figures under `figure/`
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --memory)
      MEMORY="$2"
      shift 2
      ;;
    --cxt-len)
      CXT_LEN="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --skip-dramsim)
      SKIP_DRAMSIM=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: Unknown argument '$1'" >&2
      usage >&2
      exit 1
      ;;
  esac
done

case "${MEMORY}" in
  ddr4|hbm2|both)
    ;;
  *)
    echo "ERROR: --memory must be one of: ddr4, hbm2, both" >&2
    exit 1
    ;;
esac

log "Using current shell Python environment."

if [[ "${SKIP_DRAMSIM}" != "1" ]]; then
  build_dramsim3
  run_dramsim_summary "${MEMORY}"
fi

run_decode_for_memory() {
  local memory="$1"
  local results_dir=""
  local output_stem=""
  local dram_channels=""
  local dram_r_cost=""
  local dram_w_cost=""
  local dram_bg_power=""

  case "${memory}" in
    ddr4)
      results_dir="results_ddr4_8Gb_x8_3200_ctx_sweep_decode_bs${BATCH_SIZE}"
      output_stem="figure/figure19"
      dram_channels="1"
      dram_r_cost="4454.4"
      dram_w_cost="3763.2"
      dram_bg_power="923.90592"
      ;;
    hbm2)
      results_dir="results_hbm2_8Gb_x128_ctx_sweep_decode_bs${BATCH_SIZE}"
      output_stem="figure/figure20"
      dram_channels="8"
      dram_r_cost="804.0"
      dram_w_cost="1068.0"
      dram_bg_power="427.81803"
      ;;
    *)
      echo "ERROR: Unsupported memory '${memory}'" >&2
      return 1
      ;;
  esac

  log "Running decode simulation for ${memory} (batch_size=${BATCH_SIZE})..."
  run_in_repo python run_from_configuration.py \
    --results-dir "${results_dir}" \
    --dram-rw-bw 512 \
    --dram-channels "${dram_channels}" \
    --dram-r-cost "${dram_r_cost}" \
    --dram-w-cost "${dram_w_cost}" \
    --dram-bg-power "${dram_bg_power}" \
    --is-generation 1 \
    --batch-size "${BATCH_SIZE}"

  log "Plotting decode figure for ${memory} (cxt_len=${CXT_LEN})..."
  run_in_repo python plot/figure19-20.py \
    --results-dir "${results_dir}" \
    --cxt-len "${CXT_LEN}" \
    --output-stem "${output_stem}"
}

case "${MEMORY}" in
  ddr4)
    run_decode_for_memory ddr4
    ;;
  hbm2)
    run_decode_for_memory hbm2
    ;;
  both)
    run_decode_for_memory ddr4
    run_decode_for_memory hbm2
    ;;
esac

log "Decode AE flow complete."
