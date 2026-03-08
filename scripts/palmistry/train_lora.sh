#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ $# -gt 1 ]]; then
  echo "Usage: bash scripts/palmistry/train_lora.sh [path/to/train_lora.env]" >&2
  exit 1
fi

if [[ $# -eq 1 ]]; then
  # shellcheck disable=SC1090
  source "$1"
fi

BASE_MODEL_PATH="${BASE_MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
PALMISTRY_DATA_ROOT="${PALMISTRY_DATA_ROOT:-/root/autodl-tmp/data/Palmistry.v2i.coco}"
DATA_PATH="${DATA_PATH:-${PROJECT_ROOT}/artifacts/palmistry_llava.generated.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-${PALMISTRY_DATA_ROOT}}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output/palmistry_lora_qwen3_vl_8b}"

NUM_DEVICES="${NUM_DEVICES:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
BATCH_PER_DEVICE="${BATCH_PER_DEVICE:-1}"

LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
NUM_LORA_MODULES="${NUM_LORA_MODULES:--1}"

UNFREEZE_TOPK_LLM="${UNFREEZE_TOPK_LLM:-1}"
UNFREEZE_TOPK_VISION="${UNFREEZE_TOPK_VISION:-1}"
FREEZE_VISION_TOWER="${FREEZE_VISION_TOWER:-True}"
FREEZE_LLM="${FREEZE_LLM:-True}"
FREEZE_MERGER="${FREEZE_MERGER:-False}"
VISION_LORA="${VISION_LORA:-True}"
USE_DORA="${USE_DORA:-False}"
USE_LIGER="${USE_LIGER:-False}"

NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-8}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
MERGER_LR="${MERGER_LR:-1e-4}"
VISION_LR="${VISION_LR:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"

IMAGE_MIN_PIXELS="${IMAGE_MIN_PIXELS:-43904}"
IMAGE_MAX_PIXELS="${IMAGE_MAX_PIXELS:-43904}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${PROJECT_ROOT}/scripts/zero2.json}"

if [[ ! -f "${DATA_PATH}" ]]; then
  echo "DATA_PATH does not exist: ${DATA_PATH}" >&2
  exit 1
fi

if [[ ! -d "${IMAGE_FOLDER}" ]]; then
  echo "IMAGE_FOLDER does not exist: ${IMAGE_FOLDER}" >&2
  exit 1
fi

if [[ $((BATCH_PER_DEVICE * NUM_DEVICES)) -le 0 ]]; then
  echo "Invalid batch settings." >&2
  exit 1
fi

if (( GLOBAL_BATCH_SIZE % (BATCH_PER_DEVICE * NUM_DEVICES) != 0 )); then
  echo "GLOBAL_BATCH_SIZE must be divisible by BATCH_PER_DEVICE * NUM_DEVICES." >&2
  exit 1
fi

GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH:-}"

deepspeed --num_gpus="${NUM_DEVICES}" "${PROJECT_ROOT}/src/train/train_sft.py" \
  --use_liger "${USE_LIGER}" \
  --lora_enable True \
  --vision_lora "${VISION_LORA}" \
  --use_dora "${USE_DORA}" \
  --lora_rank "${LORA_RANK}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --num_lora_modules "${NUM_LORA_MODULES}" \
  --deepspeed "${DEEPSPEED_CONFIG}" \
  --model_id "${BASE_MODEL_PATH}" \
  --data_path "${DATA_PATH}" \
  --image_folder "${IMAGE_FOLDER}" \
  --remove_unused_columns False \
  --freeze_vision_tower "${FREEZE_VISION_TOWER}" \
  --freeze_llm "${FREEZE_LLM}" \
  --freeze_merger "${FREEZE_MERGER}" \
  --unfreeze_topk_llm "${UNFREEZE_TOPK_LLM}" \
  --unfreeze_topk_vision "${UNFREEZE_TOPK_VISION}" \
  --bf16 True \
  --fp16 False \
  --disable_flash_attn2 True \
  --output_dir "${OUTPUT_DIR}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --per_device_train_batch_size "${BATCH_PER_DEVICE}" \
  --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
  --image_min_pixels "${IMAGE_MIN_PIXELS}" \
  --image_max_pixels "${IMAGE_MAX_PIXELS}" \
  --learning_rate "${LEARNING_RATE}" \
  --merger_lr "${MERGER_LR}" \
  --vision_lr "${VISION_LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --lr_scheduler_type cosine \
  --logging_steps "${LOGGING_STEPS}" \
  --gradient_checkpointing True \
  --report_to none \
  --lazy_preprocess True \
  --save_strategy epoch \
  --save_total_limit 3 \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
