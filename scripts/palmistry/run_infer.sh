#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash scripts/palmistry/run_infer.sh /path/to/image [path/to/inference.env]" >&2
  exit 1
fi

IMAGE_PATH="$1"

if [[ $# -eq 2 ]]; then
  # shellcheck disable=SC1090
  source "$2"
fi

BASE_MODEL_PATH="${BASE_MODEL_PATH:-/path/to/Qwen3-VL-8B-Instruct}"
LORA_PATH="${LORA_PATH:-${PROJECT_ROOT}/output/palmistry_grpo_report_qwen3_vl_8b_strict_v2/checkpoint-200-clean-adapter}"
GATE_CLASSIFIER_PATH="${GATE_CLASSIFIER_PATH:-${PROJECT_ROOT}/output/palmistry_gate_classifier_efficientnet_b0_v1/best.pt}"
GATE_CLASSIFIER_DEVICE="${GATE_CLASSIFIER_DEVICE:-cuda}"
GATE_CLASSIFIER_MIN_CONFIDENCE="${GATE_CLASSIFIER_MIN_CONFIDENCE:-0.55}"
GATE_CLASSIFIER_CONTINUE_MIN_CONFIDENCE="${GATE_CLASSIFIER_CONTINUE_MIN_CONFIDENCE:-0.65}"
GATE_CLASSIFIER_RETAKE_MIN_CONFIDENCE="${GATE_CLASSIFIER_RETAKE_MIN_CONFIDENCE:-0.65}"
GATE_CLASSIFIER_MIN_MARGIN="${GATE_CLASSIFIER_MIN_MARGIN:-0.10}"
TORCH_DTYPE="${TORCH_DTYPE:-bf16}"
DEVICE="${DEVICE:-cuda}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1200}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH:-}"

python -m tools.infer_palmistry \
  --base-model "${BASE_MODEL_PATH}" \
  --lora-path "${LORA_PATH}" \
  --gate-classifier-path "${GATE_CLASSIFIER_PATH}" \
  --gate-classifier-device "${GATE_CLASSIFIER_DEVICE}" \
  --gate-classifier-min-confidence "${GATE_CLASSIFIER_MIN_CONFIDENCE}" \
  --gate-classifier-continue-min-confidence "${GATE_CLASSIFIER_CONTINUE_MIN_CONFIDENCE}" \
  --gate-classifier-retake-min-confidence "${GATE_CLASSIFIER_RETAKE_MIN_CONFIDENCE}" \
  --gate-classifier-min-margin "${GATE_CLASSIFIER_MIN_MARGIN}" \
  --device "${DEVICE}" \
  --device-map "${DEVICE_MAP}" \
  --torch-dtype "${TORCH_DTYPE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --image "${IMAGE_PATH}"
