#!/usr/bin/env bash
set -euo pipefail

AE_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${AE_SCRIPTS_DIR}/.." && pwd)"

log() {
  echo "[ae] $*"
}

run_cmd() {
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '[dry-run]'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

run_in_repo() {
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '[dry-run][cwd=.]'
    printf ' %q' "$@"
    printf '\n'
  else
    (
      cd "${REPO_ROOT}"
      "$@"
    )
  fi
}

build_dramsim3() {
  log "Building DRAMsim3..."
  run_in_repo make -j -C DRAMsim3
}

append_config_for_memory() {
  local memory="$1"
  case "${memory}" in
    ddr4)
      DRAM_CONFIG_ARGS+=("configs/DDR4_8Gb_x8_3200.ini")
      ;;
    hbm2)
      DRAM_CONFIG_ARGS+=("configs/HBM2_8Gb_x128.ini")
      ;;
    both)
      DRAM_CONFIG_ARGS+=("configs/DDR4_8Gb_x8_3200.ini" "configs/HBM2_8Gb_x128.ini")
      ;;
    *)
      echo "ERROR: Unsupported memory '${memory}'. Use ddr4, hbm2, or both." >&2
      return 1
      ;;
  esac
}

run_dramsim_summary() {
  local memory="$1"
  DRAM_CONFIG_ARGS=()
  append_config_for_memory "${memory}"

  log "Generating DRAM summary for memory=${memory}..."
  run_in_repo python DRAMsim3/scripts/generate_runs_by_type_summary.py \
    --dramsim-root DRAMsim3 \
    --exe dramsim3main.out \
    --trace tests/example.trace \
    --cycles 200000 \
    --output-root DRAMsim3/runs_by_type \
    --configs "${DRAM_CONFIG_ARGS[@]}"
}
