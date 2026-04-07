#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "Usage: $0 <table_dir> [script_filter]" >&2
  echo "Env: INCLUDE_70B=1 to include *_70b.sh, DRY_RUN=1 to print only." >&2
  exit 1
fi

table_dir=$(cd -- "$1" && pwd)
script_filter=${2:-${SCRIPT_FILTER:-}}
include_70b=${INCLUDE_70B:-0}
dry_run=${DRY_RUN:-0}

if [ ! -d "${table_dir}" ]; then
  echo "Table directory not found: ${table_dir}" >&2
  exit 1
fi

mapfile -t scripts < <(find "${table_dir}" -mindepth 2 -maxdepth 2 -type f -name "*.sh" | sort)

selected_scripts=()
for script in "${scripts[@]}"; do
  base_name=$(basename "${script}")

  if [ "${include_70b}" != "1" ] && [[ "${base_name}" == *_70b.sh ]]; then
    continue
  fi

  if [ -n "${script_filter}" ] && [[ "${script}" != *"${script_filter}"* ]]; then
    continue
  fi

  selected_scripts+=("${script}")
done

if [ "${#selected_scripts[@]}" -eq 0 ]; then
  echo "No scripts matched in ${table_dir}" >&2
  exit 1
fi

echo "Running table scripts from ${table_dir}"
echo "  include_70b = ${include_70b}"
if [ -n "${script_filter}" ]; then
  echo "  filter      = ${script_filter}"
fi
echo "  script_count = ${#selected_scripts[@]}"

for script in "${selected_scripts[@]}"; do
  echo "=================================================="
  echo "Running $(realpath --relative-to="${table_dir}" "${script}")"
  echo "=================================================="

  if [ "${dry_run}" = "1" ]; then
    continue
  fi

  bash "${script}"
done

echo "Completed all scripts for ${table_dir}"
