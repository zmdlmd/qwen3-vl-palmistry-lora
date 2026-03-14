# Distillation And GRPO Workflow

## 1. Automated SFT Distillation

The new teacher-data pipeline automates the step that was previously done manually:

1. Read a manifest of hand images, or scan an image folder directly.
2. Call an OpenAI-compatible multimodal API.
3. Validate that the teacher output matches the palmistry JSON schema.
4. Optionally run a judge pass that scores visual grounding, uncertainty honesty, line consistency, and schema quality.
5. Save the canonical teacher response to JSONL logs.
6. Export a LLaVA-style SFT dataset for small-model distillation.

Main files:

- [tools/generate_teacher_dataset.py](../tools/generate_teacher_dataset.py)
- [tools/build_gate_policy_dataset.py](../tools/build_gate_policy_dataset.py)
- [src/palmistry/teacher.py](../src/palmistry/teacher.py)
- [src/palmistry/schema.py](../src/palmistry/schema.py)
- [scripts/palmistry/generate_teacher_data.sh](../scripts/palmistry/generate_teacher_data.sh)

Typical usage:

```bash
cp configs/palmistry/teacher_generation.env.example configs/palmistry/teacher_generation.env
bash scripts/palmistry/generate_teacher_data.sh configs/palmistry/teacher_generation.env
```

On AutoDL, the default teacher manifest is `/root/autodl-tmp/data/Palmistry.v2i.coco/manifests/teacher_all.jsonl`, and images are resolved from `/root/autodl-tmp/data/Palmistry.v2i.coco`.

The teacher endpoint only needs to be OpenAI-compatible. For DashScope, the teacher can be a Qwen3.5 multimodal model, while the API `model` value can be `qwen-plus` or `qwen-vl-plus`.
Use `TEACHER_NUM_WORKERS` or `--num-workers` to increase throughput with concurrent API requests.
If `JUDGE_MODEL` is configured, the pipeline adds a second pass that labels each teacher sample as `accept`, `accept_cautious`, or `reject`, and filters low-trust samples before SFT export.

Output artifacts:

- canonical SFT dataset json
- generation log jsonl
- train/val split jsons for SFT evaluation

## 2. Why This Matters

This turns the project into a repeatable teacher-student pipeline:

- large model teacher via API
- structured palmistry labels
- Qwen3-VL student via LoRA SFT

That makes it much easier to iterate on:

- better prompts
- better schema constraints
- more images
- better filtering rules

Before running SFT, split the generated dataset with [tools/split_sft_dataset.py](../tools/split_sft_dataset.py). It keeps augmented variants from the same source palm image in the same split, so `eval_path` reflects real generalization instead of leakage from near-duplicate samples.

The project now also includes a separate three-class gate-policy path:

- `continue`
- `cautious`
- `retake`

Use [tools/build_gate_policy_dataset.py](../tools/build_gate_policy_dataset.py) to bootstrap pseudo-labeled gate-policy data from the structured teacher dataset plus hard-case manifests.

## 3. GRPO For Palmistry

The repo already had a generic GRPO training stack. The palmistry extension adds:

- configurable reward module loading
- palmistry-specific structured rewards
- a dedicated palmistry GRPO launch script
- support for initializing GRPO from an existing SFT LoRA adapter

Main files:

- [src/train/train_grpo.py](../src/train/train_grpo.py)
- [src/palmistry/reward_funcs_structured.py](../src/palmistry/reward_funcs_structured.py)
- [src/palmistry/reward_funcs_report.py](../src/palmistry/reward_funcs_report.py)
- [scripts/palmistry/train_grpo.sh](../scripts/palmistry/train_grpo.sh)
- [scripts/palmistry/train_grpo_report.sh](../scripts/palmistry/train_grpo_report.sh)
- [tools/build_report_grpo_dataset.py](../tools/build_report_grpo_dataset.py)

## 4. Current Reward Design

The structured palmistry reward module currently combines:

- JSON schema validity
- core line-field coverage
- report-field coverage
- similarity to the teacher reference answer
- safety language checks

This is designed for the current structured-JSON supervision format. If you later switch GRPO to optimize long-form natural reports instead, the clean extension path is:

1. convert the teacher dataset into a report-oriented GRPO dataset
2. point `reward_funcs_module` to `src.palmistry.reward_funcs_report`
3. keep the teacher JSON as the reward reference while changing the prompt into a natural-report prompt

The report reward module now combines:

- report-format checks that reject JSON / code blocks
- section-order and line-coverage checks
- n-gram alignment against the structured teacher reference
- uncertainty honesty when the palm image is hard to read
- safety and non-diagnostic language checks

Typical report-stage preparation:

```bash
cp configs/palmistry/report_grpo_data.env.example configs/palmistry/report_grpo_data.env
bash scripts/palmistry/prepare_report_grpo_dataset.sh configs/palmistry/report_grpo_data.env
```

Typical report-stage training:

```bash
cp configs/palmistry/grpo_report.env.example configs/palmistry/grpo_report.env
bash scripts/palmistry/train_grpo_report.sh configs/palmistry/grpo_report.env
```

## 5. Recommended Training Order

The practical order is:

1. API teacher generation
2. LoRA SFT on the generated dataset
3. structured GRPO initialized from the SFT adapter
4. report GRPO initialized from the structured adapter
5. final inference prompting for natural report style

That order is much more stable than trying to jump directly from base model into GRPO.
