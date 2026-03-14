# Gate Classifier Training Notes

## Objective

This experiment trains a standalone three-class gate policy classifier for the palmistry pipeline. The classifier predicts one of:

- `continue`
- `cautious`
- `retake`

The goal is to move gate control out of the main generative pipeline and provide a cheaper, more stable front-end decision before structured analysis or full report generation.

## Dataset Construction

The classifier dataset was built from two sources:

- structured teacher data: [palmistry_llava.generated.clean.qwen3_5_plus.json](/root/autodl-tmp/Qwen/Qwen3_FineTune/Qwen-VL-Series-Finetune/artifacts/palmistry_llava.generated.clean.qwen3_5_plus.json)
- hard-case manifest: [teacher_train.hard_cases.jsonl](/root/autodl-tmp/data/Palmistry.v2i.coco/manifests/teacher_train.hard_cases.jsonl)

Builder:

- [build_gate_policy_dataset.py](/root/autodl-tmp/Qwen/Qwen3_FineTune/Qwen-VL-Series-Finetune/tools/build_gate_policy_dataset.py)

Generated dataset:

- [palmistry_gate_policy.jsonl](/root/autodl-tmp/Qwen/Qwen3_FineTune/Qwen-VL-Series-Finetune/artifacts/palmistry_gate_policy.jsonl)
- [palmistry_gate_policy.summary.json](/root/autodl-tmp/Qwen/Qwen3_FineTune/Qwen-VL-Series-Finetune/artifacts/palmistry_gate_policy.summary.json)

Dataset summary:

- total rows: `2450`
- structured rows: `2109`
- hard rows: `341`
- max per cluster: `1`

Class distribution:

- `continue`: `425`
- `cautious`: `927`
- `retake`: `1098`

## Train/Val Split

Split tool:

- [split_gate_policy_dataset.py](/root/autodl-tmp/Qwen/Qwen3_FineTune/Qwen-VL-Series-Finetune/tools/split_gate_policy_dataset.py)

Split artifacts:

- train: [palmistry_gate_policy.train.jsonl](/root/autodl-tmp/Qwen/Qwen3_FineTune/Qwen-VL-Series-Finetune/artifacts/palmistry_gate_policy.train.jsonl)
- val: [palmistry_gate_policy.val.jsonl](/root/autodl-tmp/Qwen/Qwen3_FineTune/Qwen-VL-Series-Finetune/artifacts/palmistry_gate_policy.val.jsonl)
- summary: [palmistry_gate_policy.split.summary.json](/root/autodl-tmp/Qwen/Qwen3_FineTune/Qwen-VL-Series-Finetune/artifacts/palmistry_gate_policy.split.summary.json)

Split configuration:

- seed: `42`
- validation ratio: `0.2`

Split result:

- train rows: `1960`
- val rows: `490`

Train label counts:

- `continue`: `340`
- `cautious`: `742`
- `retake`: `878`

Val label counts:

- `continue`: `85`
- `cautious`: `185`
- `retake`: `220`

## Training Implementation

Training code:

- [train_gate_classifier.py](/root/autodl-tmp/Qwen/Qwen3_FineTune/Qwen-VL-Series-Finetune/tools/train_gate_classifier.py)

Wrapper script:

- [train_gate_classifier.sh](/root/autodl-tmp/Qwen/Qwen3_FineTune/Qwen-VL-Series-Finetune/scripts/palmistry/train_gate_classifier.sh)

Env example:

- [train_gate_classifier.env.example](/root/autodl-tmp/Qwen/Qwen3_FineTune/Qwen-VL-Series-Finetune/configs/palmistry/train_gate_classifier.env.example)

Training setup:

- image root: `/root/autodl-tmp/data/Palmistry.v2i.coco`
- input resolution: `224`
- optimizer: `AdamW`
- loss: weighted cross entropy
- early stopping patience: `5`
- pretrained ImageNet initialization: enabled

## Backbone Comparison

Two lightweight backbones were trained as initial baselines.

### ResNet-18

Output directory:

- [palmistry_gate_classifier_resnet18_v1](/root/autodl-tmp/Qwen/Qwen3_FineTune/Qwen-VL-Series-Finetune/output/palmistry_gate_classifier_resnet18_v1)

Summary:

- [summary.json](/root/autodl-tmp/Qwen/Qwen3_FineTune/Qwen-VL-Series-Finetune/output/palmistry_gate_classifier_resnet18_v1/summary.json)

Best validation result:

- best epoch: `2`
- validation accuracy: `0.4959`
- validation macro precision: `0.4976`
- validation macro recall: `0.5199`
- validation macro F1: `0.4780`

Per-class F1:

- `continue`: `0.4103`
- `cautious`: `0.3869`
- `retake`: `0.6368`

Main issue:

- `cautious` recall is low and the model tends to confuse `cautious` with both neighboring classes.

### EfficientNet-B0

Output directory:

- [palmistry_gate_classifier_efficientnet_b0_v1](/root/autodl-tmp/Qwen/Qwen3_FineTune/Qwen-VL-Series-Finetune/output/palmistry_gate_classifier_efficientnet_b0_v1)

Summary:

- [summary.json](/root/autodl-tmp/Qwen/Qwen3_FineTune/Qwen-VL-Series-Finetune/output/palmistry_gate_classifier_efficientnet_b0_v1/summary.json)

Best validation result:

- best epoch: `6`
- validation accuracy: `0.5347`
- validation macro precision: `0.5178`
- validation macro recall: `0.5437`
- validation macro F1: `0.5116`

Per-class F1:

- `continue`: `0.4362`
- `cautious`: `0.4241`
- `retake`: `0.6746`

Compared with ResNet-18, EfficientNet-B0 improved all three classes, especially:

- `continue`: `0.4103 -> 0.4362`
- `cautious`: `0.3869 -> 0.4241`
- `retake`: `0.6368 -> 0.6746`

## Current Best Checkpoint

Recommended current gate classifier:

- [best.pt](/root/autodl-tmp/Qwen/Qwen3_FineTune/Qwen-VL-Series-Finetune/output/palmistry_gate_classifier_efficientnet_b0_v1/best.pt)

Checkpoint metadata:

- model: `efficientnet_b0`
- best epoch: `6`
- label mapping:
  - `continue -> 0`
  - `cautious -> 1`
  - `retake -> 2`

## Interpretation

This classifier is already useful as a first standalone gate baseline, but it is not yet strong enough to fully replace all heuristic safeguards.

Observed behavior:

- `retake` is the easiest class
- `cautious` is the hardest class
- the main confusion is around the middle decision boundary:
  - `continue <-> cautious`
  - `cautious <-> retake`

This is expected because `cautious` is a transitional regime rather than a visually extreme category.

## Recommended Next Steps

1. Integrate the EfficientNet-B0 checkpoint into the runtime gate path in [pipeline.py](/root/autodl-tmp/Qwen/Qwen3_FineTune/Qwen-VL-Series-Finetune/src/palmistry/pipeline.py).
2. Add test-time decision smoothing:
   - low confidence classifier outputs should fall back to conservative heuristics.
3. Improve the training labels:
   - refine `cautious` pseudo-label rules
   - reduce noisy borderline samples
4. Add a holdout evaluation set with real user photos, not only the current dataset domain.
5. Consider a second round with:
   - stronger augmentations
   - focal loss
   - class-balanced sampling

## Repro Commands

Split:

```bash
cd /root/autodl-tmp/Qwen/Qwen3_FineTune/Qwen-VL-Series-Finetune
python tools/split_gate_policy_dataset.py \
  --input-jsonl artifacts/palmistry_gate_policy.jsonl \
  --train-jsonl artifacts/palmistry_gate_policy.train.jsonl \
  --val-jsonl artifacts/palmistry_gate_policy.val.jsonl \
  --summary-json artifacts/palmistry_gate_policy.split.summary.json \
  --val-ratio 0.2 \
  --seed 42
```

Train ResNet-18:

```bash
cd /root/autodl-tmp/Qwen/Qwen3_FineTune/Qwen-VL-Series-Finetune
OMP_NUM_THREADS=4 python tools/train_gate_classifier.py \
  --train-jsonl artifacts/palmistry_gate_policy.train.jsonl \
  --val-jsonl artifacts/palmistry_gate_policy.val.jsonl \
  --image-folder /root/autodl-tmp/data/Palmistry.v2i.coco \
  --output-dir output/palmistry_gate_classifier_resnet18_v1 \
  --model-name resnet18 \
  --epochs 15 \
  --batch-size 64 \
  --learning-rate 1e-4 \
  --weight-decay 1e-4 \
  --num-workers 4 \
  --image-size 224 \
  --patience 5 \
  --seed 42 \
  --pretrained
```

Train EfficientNet-B0:

```bash
cd /root/autodl-tmp/Qwen/Qwen3_FineTune/Qwen-VL-Series-Finetune
OMP_NUM_THREADS=4 python tools/train_gate_classifier.py \
  --train-jsonl artifacts/palmistry_gate_policy.train.jsonl \
  --val-jsonl artifacts/palmistry_gate_policy.val.jsonl \
  --image-folder /root/autodl-tmp/data/Palmistry.v2i.coco \
  --output-dir output/palmistry_gate_classifier_efficientnet_b0_v1 \
  --model-name efficientnet_b0 \
  --epochs 15 \
  --batch-size 64 \
  --learning-rate 1e-4 \
  --weight-decay 1e-4 \
  --num-workers 4 \
  --image-size 224 \
  --patience 5 \
  --seed 42 \
  --pretrained
```
