# Strict SFT V2 Training Report

## Overview

This report records the end-to-end `strict v2` SFT run built on the new `teacher + judge/filter` pipeline.

- Base model: `Qwen3-VL-8B-Instruct`
- Task: palm image to structured palmistry JSON
- Training style: LoRA SFT
- Training date: `2026-03-15`

## Dataset Version

### Final judged teacher dataset

- File: `artifacts/palmistry_llava.generated.clean.qwen3_5_plus.judged_v2.json`
- Total records: `3906`

Final latest-status summary reconstructed from `artifacts/palmistry_teacher_generations.clean.qwen3_5_plus.judged_v2.jsonl`:

- `ok = 3906`
- `filtered = 36`
- `error = 0`

Judge decision split inside the accepted set:

- `accept = 2171`
- `accept_cautious = 1735`

### Stage split

Cluster-disjoint split generated with:

- `eval_ratio = 0.15`
- `grpo_ratio = 0.25`
- `seed = 42`
- `cluster_regex = ^(?P<base>.+)\\.rf\\.[^.]+$`

Artifacts:

- `artifacts/palmistry_llava.generated.clean.qwen3_5_plus.judged_v2.stage_split.sft_train.json`
- `artifacts/palmistry_llava.generated.clean.qwen3_5_plus.judged_v2.stage_split.grpo_train.json`
- `artifacts/palmistry_llava.generated.clean.qwen3_5_plus.judged_v2.stage_split.eval_holdout.json`
- `artifacts/palmistry_llava.report_grpo.judged_v2.stage_split.grpo_train.json`
- `artifacts/palmistry_llava.generated.clean.qwen3_5_plus.judged_v2.stage_split.summary.json`

Split result:

- `total_records = 3906`
- `total_clusters = 2108`
- `sft_records = 2344`
- `grpo_records = 976`
- `eval_records = 586`
- `sft_clusters = 1269`
- `grpo_clusters = 523`
- `eval_clusters = 316`
- `cluster_overlap = 0 / 0 / 0`

## Training Configuration

Training config file:

- `configs/palmistry/train_lora_strict_v2.env`

Key settings:

- `NUM_DEVICES = 1`
- `GLOBAL_BATCH_SIZE = 4`
- `BATCH_PER_DEVICE = 1`
- `gradient_accumulation_steps = 4`
- `NUM_TRAIN_EPOCHS = 3`
- `LEARNING_RATE = 5e-5`
- `MERGER_LR = 1e-4`
- `VISION_LR = 1e-5`
- `LORA_RANK = 32`
- `LORA_ALPHA = 32`
- `LORA_DROPOUT = 0.05`
- `FREEZE_VISION_TOWER = True`
- `FREEZE_LLM = True`
- `FREEZE_MERGER = False`
- `VISION_LORA = True`
- `DEEPSPEED_CONFIG = ./scripts/zero2.json`
- `IMAGE_MIN_PIXELS = 43904`
- `IMAGE_MAX_PIXELS = 43904`

Training command entry:

- `scripts/palmistry/train_lora.sh`

Training log:

- `artifacts/train_lora_strict_v2.log`

Output directory:

- `output/palmistry_lora_qwen3_vl_8b_strict_v2`

## Run Summary

The run completed the full optimization schedule:

- total steps: `1758 / 1758`
- completed epochs: `3.0`

Observed training progression:

- early logged loss: `2.0596` at `epoch 0.02`
- late logged loss: `0.4566` at `epoch 2.99`
- final logged eval loss: `0.49681615829467773` at `epoch 3.0`

The run therefore reached the final evaluation stage successfully.

## Checkpoints

Checkpoints present under `output/palmistry_lora_qwen3_vl_8b_strict_v2`:

- `checkpoint-586`
- `checkpoint-1172`
- `checkpoint-1758`

The most important artifact is:

- `output/palmistry_lora_qwen3_vl_8b_strict_v2/checkpoint-1758`

This checkpoint contains:

- `adapter_model.safetensors`
- `adapter_config.json`
- `non_lora_state_dict.bin`
- tokenizer / processor files

Practical conclusion:

- `checkpoint-1758` is usable as the strict `SFT v2` baseline checkpoint.

## Failure During Save

The training process exited with return code `1`, but the failure happened after the training and final eval had already completed.

Root cause:

- disk full during checkpoint write on `/root/autodl-tmp`

Relevant log errors:

- `PytorchStreamWriter failed writing file data/701: file write failed`
- `unexpected pos ...`

At failure time, the filesystem state was:

- `/root/autodl-tmp` reached `100%`

This was later resolved by disk expansion, but the training run itself was not rerun because `checkpoint-1758` had already been written.

## Post-Training Smoke Check

The strict `SFT v2` checkpoint was used for inference smoke checks with the standalone gate classifier enabled.

Observed behavior:

- clear sample: `gate = continue`, structured JSON and full report both generated
- medium-quality sample: `gate = cautious`, structured JSON preserved and report flow stayed conservative
- poor-quality sample: `gate = retake`, analysis was blocked with a retake message

This indicates that the strict `SFT v2` checkpoint is functionally usable and compatible with the current conservative inference pipeline.

## Current Conclusion

Strict `SFT v2` is now the cleanest supervised baseline built on:

- judged teacher data
- cluster-disjoint `SFT / GRPO / eval` split
- standalone three-class gate policy

Recommended next step:

1. use `checkpoint-1758` as the strict baseline
2. run strict holdout evaluation and side-by-side comparison against the previous non-strict baseline
3. continue with strict `report GRPO` on `judged_v2.stage_split.grpo_train.json`
