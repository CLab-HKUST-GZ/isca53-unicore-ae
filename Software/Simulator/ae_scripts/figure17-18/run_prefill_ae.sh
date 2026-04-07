#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../common.sh"

MEMORY="both"
CXT_LEN="8192"
DRY_RUN=0
SKIP_DRAMSIM=0
FUSED_ATTN=0

usage() {
  cat <<'EOF'
Usage: run_prefill_ae.sh [options]

Options:
  --memory <ddr4|hbm2|both>   Which memory system(s) to run. Default: both
  --cxt-len <int>             Context length to use for plotting. Default: 8192
  --fused-attn                Enable prefill fused-attention DRAM approximation.
  --skip-dramsim              Skip DRAMsim3 build and summary generation.
  --dry-run                   Print commands without executing them.
  -h, --help                  Show this help message.

Outputs:
  - Prefill CSVs under `results_ddr4_8Gb_x8_3200_ctx_sweep*/` and/or
    `results_hbm2_8Gb_x128_ctx_sweep*/`
  - Prefill figures under `figure/`
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
    --fused-attn)
      FUSED_ATTN=1
      shift
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

run_prefill_for_memory() {
  local memory="$1"
  local results_dir=""
  local energy_output_stem=""
  local speedup_output_stem=""
  local dram_channels=""
  local dram_r_cost=""
  local dram_w_cost=""
  local dram_bg_power=""
  local fused_suffix=""

  if [[ "${FUSED_ATTN}" == "1" ]]; then
    fused_suffix="_fused_attn"
  fi

  case "${memory}" in
    ddr4)
      results_dir="results_ddr4_8Gb_x8_3200_ctx_sweep${fused_suffix}"
      if [[ "${FUSED_ATTN}" == "1" ]]; then
        energy_output_stem="figure/figure18"
        speedup_output_stem="figure/figure17"
      else
        energy_output_stem="figure/figure18_ddr4"
        speedup_output_stem="figure/figure17_ddr4"
      fi
      dram_channels="1"
      dram_r_cost="4454.4"
      dram_w_cost="3763.2"
      dram_bg_power="923.90592"
      ;;
    hbm2)
      results_dir="results_hbm2_8Gb_x128_ctx_sweep${fused_suffix}"
      if [[ "${FUSED_ATTN}" == "1" ]]; then
        energy_output_stem="figure/figure18_hbm2_fused_attn"
        speedup_output_stem="figure/figure17_hbm2_fused_attn"
      else
        energy_output_stem="figure/figure18_hbm2"
        speedup_output_stem="figure/figure17_hbm2"
      fi
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

  log "Running prefill simulation for ${memory}..."
  run_in_repo python run_from_configuration.py \
    --results-dir "${results_dir}" \
    --dram-rw-bw 512 \
    --dram-channels "${dram_channels}" \
    --dram-r-cost "${dram_r_cost}" \
    --dram-w-cost "${dram_w_cost}" \
    --dram-bg-power "${dram_bg_power}" \
    --fused-attn "${FUSED_ATTN}"

  log "Plotting legacy-style prefill figure for ${memory} (cxt_len=${CXT_LEN})..."
  run_in_repo python plot/figure18_prefill_energy.py \
    --results-dir "${results_dir}" \
    --cxt-len "${CXT_LEN}" \
    --output-stem "${energy_output_stem}"

  log "Plotting legacy-style prefill speedup figure for ${memory} (cxt_len=${CXT_LEN})..."
  run_in_repo python plot/figure17_prefill_speedup.py \
    --results-dir "${results_dir}" \
    --cxt-len "${CXT_LEN}" \
    --output-stem "${speedup_output_stem}"
}

case "${MEMORY}" in
  ddr4)
    run_prefill_for_memory ddr4
    ;;
  hbm2)
    run_prefill_for_memory hbm2
    ;;
  both)
    run_prefill_for_memory ddr4
    run_prefill_for_memory hbm2
    ;;
esac

log "Prefill AE flow complete."
