#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [env-file]" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -eq 1 ]]; then
  # shellcheck disable=SC1090
  source "$1"
fi

export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"

GATE_INPUT_JSONL="${GATE_INPUT_JSONL:-$ROOT_DIR/artifacts/palmistry_gate_policy.jsonl}"
GATE_TRAIN_JSONL="${GATE_TRAIN_JSONL:-$ROOT_DIR/artifacts/palmistry_gate_policy.train.jsonl}"
GATE_VAL_JSONL="${GATE_VAL_JSONL:-$ROOT_DIR/artifacts/palmistry_gate_policy.val.jsonl}"
GATE_SPLIT_SUMMARY="${GATE_SPLIT_SUMMARY:-$ROOT_DIR/artifacts/palmistry_gate_policy.split.summary.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/root/autodl-tmp/data/Palmistry.v2i.coco}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/output/palmistry_gate_classifier_v1}"
MODEL_NAME="${MODEL_NAME:-resnet18}"
VAL_RATIO="${VAL_RATIO:-0.2}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-12}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
PATIENCE="${PATIENCE:-4}"
PRETRAINED="${PRETRAINED:-true}"

python tools/split_gate_policy_dataset.py \
  --input-jsonl "$GATE_INPUT_JSONL" \
  --train-jsonl "$GATE_TRAIN_JSONL" \
  --val-jsonl "$GATE_VAL_JSONL" \
  --summary-json "$GATE_SPLIT_SUMMARY" \
  --val-ratio "$VAL_RATIO" \
  --seed "$SEED"

PRETRAINED_FLAG="--pretrained"
if [[ "$PRETRAINED" != "true" ]]; then
  PRETRAINED_FLAG="--no-pretrained"
fi

python tools/train_gate_classifier.py \
  --train-jsonl "$GATE_TRAIN_JSONL" \
  --val-jsonl "$GATE_VAL_JSONL" \
  --image-folder "$IMAGE_FOLDER" \
  --output-dir "$OUTPUT_DIR" \
  --model-name "$MODEL_NAME" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --learning-rate "$LEARNING_RATE" \
  --weight-decay "$WEIGHT_DECAY" \
  --num-workers "$NUM_WORKERS" \
  --image-size "$IMAGE_SIZE" \
  --patience "$PATIENCE" \
  --seed "$SEED" \
  "$PRETRAINED_FLAG"
