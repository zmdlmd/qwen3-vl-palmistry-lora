#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ $# -gt 1 ]]; then
  echo "Usage: bash scripts/palmistry/train_grpo_report.sh [path/to/grpo_report.env]" >&2
  exit 1
fi

if [[ $# -eq 1 ]]; then
  # shellcheck disable=SC1090
  source "$1"
fi

export DATA_PATH="${DATA_PATH:-${PROJECT_ROOT}/artifacts/palmistry_llava.report_grpo.json}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output/palmistry_grpo_report_qwen3_vl_8b}"
export REWARD_FUNCS_MODULE="${REWARD_FUNCS_MODULE:-src.palmistry.reward_funcs_report}"
export MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-1400}"

bash "${PROJECT_ROOT}/scripts/palmistry/train_grpo.sh"
