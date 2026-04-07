#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: clean_ae.sh [options]

Options:
  --dry-run   Print cleanup commands without executing them.
  -h, --help  Show this help message.

This script removes the generated artifacts produced by the simulator AE flows:
  - built DRAMsim3 binaries / object files
  - generated DRAMsim3 summary outputs
  - generated result CSV directories
  - generated figure PDFs / PNGs
  - Python __pycache__ directories
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
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

log "Cleaning DRAMsim3 build artifacts..."
run_in_repo make -C DRAMsim3 clean

log "Cleaning generated DRAMsim3 summaries..."
run_in_repo bash -lc 'rm -rf -- DRAMsim3/runs_by_type/DDR4 DRAMsim3/runs_by_type/HBM2 DRAMsim3/runs_by_type/summary.csv'

log "Cleaning generated AE result directories..."
run_in_repo bash -lc 'rm -rf -- \
  results_ddr4_8Gb_x8_3200_ctx_sweep \
  results_ddr4_8Gb_x8_3200_ctx_sweep_fused_attn \
  results_ddr4_8Gb_x8_3200_ctx_sweep_decode_bs* \
  results_hbm2_8Gb_x128_ctx_sweep \
  results_hbm2_8Gb_x128_ctx_sweep_fused_attn \
  results_hbm2_8Gb_x128_ctx_sweep_decode_bs*'

log "Cleaning generated figures..."
run_in_repo bash -lc 'mkdir -p figure && rm -rf -- \
  figure/*.pdf figure/*.png \
  plot/figure*.pdf plot/figure*.png \
  plot/new_*.pdf plot/new_*.png \
  plot/Speedup_Stacked_Clean.pdf plot/Speedup_Stacked_Clean.png'

log "Cleaning Python cache directories..."
run_in_repo bash -lc 'find . -type d -name __pycache__ -prune -exec rm -rf {} +'

log "AE cleanup complete."
