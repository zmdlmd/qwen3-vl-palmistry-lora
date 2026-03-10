#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ $# -eq 1 && ( "$1" == "--help" || "$1" == "-h" ) ]]; then
  echo "Usage: bash scripts/palmistry/evaluate_pipeline.sh [path/to/evaluate_pipeline.env]" >&2
  exit 0
fi

if [[ $# -gt 1 ]]; then
  echo "Usage: bash scripts/palmistry/evaluate_pipeline.sh [path/to/evaluate_pipeline.env]" >&2
  exit 1
fi

if [[ $# -eq 1 ]]; then
  # shellcheck disable=SC1090
  source "$1"
fi

BASE_MODEL_PATH="${BASE_MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
LORA_PATH="${LORA_PATH:-${PROJECT_ROOT}/output/palmistry_lora_qwen3_vl_8b_clean_v1}"
PALMISTRY_DATA_ROOT="${PALMISTRY_DATA_ROOT:-/root/autodl-tmp/data/Palmistry.v2i.coco}"
IMAGE_ROOT="${IMAGE_ROOT:-${PALMISTRY_DATA_ROOT}}"
VAL_JSON="${VAL_JSON:-${PROJECT_ROOT}/artifacts/palmistry_llava.generated.clean.qwen3_5_plus.val.json}"
HARD_MANIFEST="${HARD_MANIFEST:-${PALMISTRY_DATA_ROOT}/manifests/teacher_train.hard_cases.jsonl}"
OUTPUT_JSON="${OUTPUT_JSON:-${PROJECT_ROOT}/artifacts/evals/palmistry_eval.summary.json}"
OUTPUT_JSONL="${OUTPUT_JSONL:-${PROJECT_ROOT}/artifacts/evals/palmistry_eval.samples.jsonl}"

VAL_LIMIT="${VAL_LIMIT:-}"
HARD_LIMIT="${HARD_LIMIT:-}"
DEVICE="${DEVICE:-cuda:0}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
TORCH_DTYPE="${TORCH_DTYPE:-auto}"
STYLE="${STYLE:-balanced}"
REPORT_MAX_NEW_TOKENS="${REPORT_MAX_NEW_TOKENS:-900}"
STRUCTURED_MAX_NEW_TOKENS="${STRUCTURED_MAX_NEW_TOKENS:-1400}"
TEMPERATURE="${TEMPERATURE:-0.2}"
TOP_P="${TOP_P:-0.9}"
HARD_MODE="${HARD_MODE:-gate_only}"
PROGRESS_EVERY="${PROGRESS_EVERY:-10}"
SUMMARY_EVERY="${SUMMARY_EVERY:-25}"

if [[ -z "${VAL_JSON}" && -z "${HARD_MANIFEST}" ]]; then
  echo "At least one of VAL_JSON or HARD_MANIFEST must be set." >&2
  exit 1
fi

if [[ -n "${VAL_JSON}" && ! -f "${VAL_JSON}" ]]; then
  echo "VAL_JSON does not exist: ${VAL_JSON}" >&2
  exit 1
fi

if [[ -n "${HARD_MANIFEST}" && ! -f "${HARD_MANIFEST}" ]]; then
  echo "HARD_MANIFEST does not exist: ${HARD_MANIFEST}" >&2
  exit 1
fi

if [[ ! -d "${IMAGE_ROOT}" ]]; then
  echo "IMAGE_ROOT does not exist: ${IMAGE_ROOT}" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT_JSON}")"
if [[ -n "${OUTPUT_JSONL}" ]]; then
  mkdir -p "$(dirname "${OUTPUT_JSONL}")"
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH:-}"

CMD=(
  python -m tools.evaluate_palmistry_pipeline
  --base-model "${BASE_MODEL_PATH}"
  --lora-path "${LORA_PATH}"
  --image-root "${IMAGE_ROOT}"
  --output-json "${OUTPUT_JSON}"
  --device "${DEVICE}"
  --device-map "${DEVICE_MAP}"
  --torch-dtype "${TORCH_DTYPE}"
  --style "${STYLE}"
  --report-max-new-tokens "${REPORT_MAX_NEW_TOKENS}"
  --structured-max-new-tokens "${STRUCTURED_MAX_NEW_TOKENS}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --hard-mode "${HARD_MODE}"
  --progress-every "${PROGRESS_EVERY}"
  --summary-every "${SUMMARY_EVERY}"
)

if [[ -n "${VAL_JSON}" ]]; then
  CMD+=(--val-json "${VAL_JSON}")
fi

if [[ -n "${HARD_MANIFEST}" ]]; then
  CMD+=(--hard-manifest "${HARD_MANIFEST}")
fi

if [[ -n "${OUTPUT_JSONL}" ]]; then
  CMD+=(--output-jsonl "${OUTPUT_JSONL}")
fi

if [[ -n "${VAL_LIMIT}" ]]; then
  CMD+=(--val-limit "${VAL_LIMIT}")
fi

if [[ -n "${HARD_LIMIT}" ]]; then
  CMD+=(--hard-limit "${HARD_LIMIT}")
fi

"${CMD[@]}"
