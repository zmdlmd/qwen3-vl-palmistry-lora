#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ $# -gt 1 ]]; then
  echo "Usage: bash scripts/palmistry/train_grpo.sh [path/to/grpo.env]" >&2
  exit 1
fi

if [[ $# -eq 1 ]]; then
  # shellcheck disable=SC1090
  source "$1"
fi

BASE_MODEL_PATH="${BASE_MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
LORA_WEIGHT_PATH="${LORA_WEIGHT_PATH:-}"
DATA_PATH="${DATA_PATH:-${PROJECT_ROOT}/artifacts/palmistry_llava.generated.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-${PROJECT_ROOT}/data/images}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output/palmistry_grpo_qwen3_vl_8b}"

NUM_DEVICES="${NUM_DEVICES:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
BATCH_PER_DEVICE="${BATCH_PER_DEVICE:-1}"

LORA_ENABLE="${LORA_ENABLE:-true}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
NUM_LORA_MODULES="${NUM_LORA_MODULES:--1}"
UNFREEZE_TOPK_LLM="${UNFREEZE_TOPK_LLM:-1}"
UNFREEZE_TOPK_VISION="${UNFREEZE_TOPK_VISION:-1}"
FREEZE_VISION_TOWER="${FREEZE_VISION_TOWER:-true}"
FREEZE_LLM="${FREEZE_LLM:-true}"
FREEZE_MERGER="${FREEZE_MERGER:-false}"
VISION_LORA="${VISION_LORA:-true}"

NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:--1}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
MERGER_LR="${MERGER_LR:-1e-5}"
VISION_LR="${VISION_LR:-1e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"

IMAGE_MIN_PIXELS="${IMAGE_MIN_PIXELS:-43904}"
IMAGE_MAX_PIXELS="${IMAGE_MAX_PIXELS:-43904}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${PROJECT_ROOT}/scripts/zero3.json}"
REWARD_FUNCS_MODULE="${REWARD_FUNCS_MODULE:-src.palmistry.reward_funcs_structured}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-1024}"
BITS="${BITS:-16}"
BETA="${BETA:-0.04}"
SAVE_STRATEGY="${SAVE_STRATEGY:-epoch}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
OVERWRITE_OUTPUT_DIR="${OVERWRITE_OUTPUT_DIR:-false}"

if [[ ! -f "${DATA_PATH}" ]]; then
  echo "DATA_PATH does not exist: ${DATA_PATH}" >&2
  exit 1
fi

if [[ ! -d "${IMAGE_FOLDER}" ]]; then
  echo "IMAGE_FOLDER does not exist: ${IMAGE_FOLDER}" >&2
  exit 1
fi

if (( GLOBAL_BATCH_SIZE % (BATCH_PER_DEVICE * NUM_DEVICES) != 0 )); then
  echo "GLOBAL_BATCH_SIZE must be divisible by BATCH_PER_DEVICE * NUM_DEVICES." >&2
  exit 1
fi

GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH:-}"

CMD=(
  deepspeed
  --num_gpus="${NUM_DEVICES}"
  "${PROJECT_ROOT}/src/train/train_grpo.py"
  --deepspeed "${DEEPSPEED_CONFIG}"
  --model_id "${BASE_MODEL_PATH}"
  --data_path "${DATA_PATH}"
  --image_folder "${IMAGE_FOLDER}"
  --freeze_vision_tower "${FREEZE_VISION_TOWER}"
  --freeze_llm "${FREEZE_LLM}"
  --freeze_merger "${FREEZE_MERGER}"
  --lora_enable "${LORA_ENABLE}"
  --vision_lora "${VISION_LORA}"
  --lora_rank "${LORA_RANK}"
  --lora_alpha "${LORA_ALPHA}"
  --lora_dropout "${LORA_DROPOUT}"
  --num_lora_modules "${NUM_LORA_MODULES}"
  --lora_weight_path "${LORA_WEIGHT_PATH}"
  --bits "${BITS}"
  --unfreeze_topk_llm "${UNFREEZE_TOPK_LLM}"
  --unfreeze_topk_vision "${UNFREEZE_TOPK_VISION}"
  --bf16 True
  --fp16 False
  --disable_flash_attn2 True
  --output_dir "${OUTPUT_DIR}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS}"
  --max_steps "${MAX_STEPS}"
  --num_generations "${NUM_GENERATIONS}"
  --per_device_train_batch_size "${BATCH_PER_DEVICE}"
  --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}"
  --max_completion_length "${MAX_COMPLETION_LENGTH}"
  --max_prompt_length "${MAX_PROMPT_LENGTH}"
  --image_min_pixels "${IMAGE_MIN_PIXELS}"
  --image_max_pixels "${IMAGE_MAX_PIXELS}"
  --learning_rate "${LEARNING_RATE}"
  --merger_lr "${MERGER_LR}"
  --vision_lr "${VISION_LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --warmup_ratio "${WARMUP_RATIO}"
  --lr_scheduler_type cosine
  --logging_steps "${LOGGING_STEPS}"
  --gradient_checkpointing True
  --remove_unused_columns False
  --report_to none
  --save_strategy "${SAVE_STRATEGY}"
  --save_total_limit "${SAVE_TOTAL_LIMIT}"
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
  --reward_funcs_module "${REWARD_FUNCS_MODULE}"
  --beta "${BETA}"
)

if [[ "${OVERWRITE_OUTPUT_DIR}" == "true" ]]; then
  CMD+=(--overwrite_output_dir True)
fi

"${CMD[@]}"
