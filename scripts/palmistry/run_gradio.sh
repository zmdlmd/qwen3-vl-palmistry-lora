#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ $# -gt 1 ]]; then
  echo "Usage: bash scripts/palmistry/run_gradio.sh [path/to/inference.env]" >&2
  exit 1
fi

if [[ $# -eq 1 ]]; then
  # shellcheck disable=SC1090
  source "$1"
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
GRADIO_SERVER_NAME="${GRADIO_SERVER_NAME:-0.0.0.0}"
GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-7860}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH:-}"

python -m apps.gradio_palmistry \
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
  --server-name "${GRADIO_SERVER_NAME}" \
  --server-port "${GRADIO_SERVER_PORT}"
