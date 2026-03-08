#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ $# -gt 1 ]]; then
  echo "Usage: bash scripts/palmistry/generate_teacher_data.sh [path/to/teacher_generation.env]" >&2
  exit 1
fi

if [[ $# -eq 1 ]]; then
  # shellcheck disable=SC1090
  source "$1"
fi

OPENAI_API_KEY="${OPENAI_API_KEY:-${DASHSCOPE_API_KEY:-}}"
TEACHER_API_BASE="${TEACHER_API_BASE:-https://api.openai.com/v1}"
TEACHER_MODEL="${TEACHER_MODEL:-}"
PALMISTRY_DATA_ROOT="${PALMISTRY_DATA_ROOT:-/root/autodl-tmp/data/Palmistry.v2i.coco}"
TEACHER_MANIFEST="${TEACHER_MANIFEST:-${PALMISTRY_DATA_ROOT}/manifests/teacher_all.jsonl}"
TEACHER_IMAGE_DIR="${TEACHER_IMAGE_DIR:-${PALMISTRY_DATA_ROOT}}"
TEACHER_OUTPUT_JSON="${TEACHER_OUTPUT_JSON:-${PROJECT_ROOT}/artifacts/palmistry_llava.generated.json}"
TEACHER_OUTPUT_JSONL="${TEACHER_OUTPUT_JSONL:-${PROJECT_ROOT}/artifacts/palmistry_teacher_generations.jsonl}"
TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE:-0.2}"
TEACHER_MAX_TOKENS="${TEACHER_MAX_TOKENS:-2200}"
TEACHER_MAX_RETRIES="${TEACHER_MAX_RETRIES:-3}"
TEACHER_REQUEST_TIMEOUT="${TEACHER_REQUEST_TIMEOUT:-180}"
TEACHER_SLEEP_SECONDS="${TEACHER_SLEEP_SECONDS:-0.0}"
TEACHER_NUM_WORKERS="${TEACHER_NUM_WORKERS:-4}"
TEACHER_JSON_MODE="${TEACHER_JSON_MODE:-false}"

if [[ -z "${OPENAI_API_KEY}" ]]; then
  echo "An OpenAI-compatible API key is required. Set OPENAI_API_KEY or DASHSCOPE_API_KEY." >&2
  exit 1
fi

if [[ -z "${TEACHER_MODEL}" ]]; then
  echo "TEACHER_MODEL is required." >&2
  exit 1
fi

if [[ -n "${TEACHER_MANIFEST}" && ! -f "${TEACHER_MANIFEST}" ]]; then
  echo "TEACHER_MANIFEST does not exist: ${TEACHER_MANIFEST}" >&2
  exit 1
fi

if [[ ! -d "${TEACHER_IMAGE_DIR}" ]]; then
  echo "TEACHER_IMAGE_DIR does not exist: ${TEACHER_IMAGE_DIR}" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH:-}"

CMD=(
  python -m tools.generate_teacher_dataset
  --api-base "${TEACHER_API_BASE}"
  --model "${TEACHER_MODEL}"
  --output-json "${TEACHER_OUTPUT_JSON}"
  --output-jsonl "${TEACHER_OUTPUT_JSONL}"
  --image-dir "${TEACHER_IMAGE_DIR}"
  --temperature "${TEACHER_TEMPERATURE}"
  --max-tokens "${TEACHER_MAX_TOKENS}"
  --max-retries "${TEACHER_MAX_RETRIES}"
  --request-timeout "${TEACHER_REQUEST_TIMEOUT}"
  --sleep-seconds "${TEACHER_SLEEP_SECONDS}"
  --num-workers "${TEACHER_NUM_WORKERS}"
)

if [[ -n "${TEACHER_MANIFEST}" ]]; then
  CMD+=(--manifest "${TEACHER_MANIFEST}")
fi

if [[ "${TEACHER_JSON_MODE}" == "true" ]]; then
  CMD+=(--json-mode)
fi

"${CMD[@]}"
