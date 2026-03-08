#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ $# -gt 1 ]]; then
  echo "Usage: bash scripts/palmistry/prepare_report_grpo_dataset.sh [path/to/report_grpo_data.env]" >&2
  exit 1
fi

if [[ $# -eq 1 ]]; then
  # shellcheck disable=SC1090
  source "$1"
fi

SOURCE_DATA_PATH="${SOURCE_DATA_PATH:-${PROJECT_ROOT}/artifacts/palmistry_llava.generated.json}"
OUTPUT_DATA_PATH="${OUTPUT_DATA_PATH:-${PROJECT_ROOT}/artifacts/palmistry_llava.report_grpo.json}"
REPORT_STYLE="${REPORT_STYLE:-balanced}"
REPORT_PROMPT_FILE="${REPORT_PROMPT_FILE:-}"
ID_SUFFIX="${ID_SUFFIX:--report-grpo}"

if [[ ! -f "${SOURCE_DATA_PATH}" ]]; then
  echo "SOURCE_DATA_PATH does not exist: ${SOURCE_DATA_PATH}" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH:-}"

CMD=(
  python -m tools.build_report_grpo_dataset
  --input-json "${SOURCE_DATA_PATH}"
  --output-json "${OUTPUT_DATA_PATH}"
  --style "${REPORT_STYLE}"
  --id-suffix "${ID_SUFFIX}"
)

if [[ -n "${REPORT_PROMPT_FILE}" ]]; then
  CMD+=(--prompt-file "${REPORT_PROMPT_FILE}")
fi

"${CMD[@]}"
