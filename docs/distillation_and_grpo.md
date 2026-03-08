# Distillation And GRPO Workflow

## 1. Automated SFT Distillation

The new teacher-data pipeline automates the step that was previously done manually:

1. Read a manifest of hand images, or scan an image folder directly.
2. Call an OpenAI-compatible multimodal API.
3. Validate that the teacher output matches the palmistry JSON schema.
4. Save the canonical teacher response to JSONL logs.
5. Export a LLaVA-style SFT dataset for small-model distillation.

Main files:

- [tools/generate_teacher_dataset.py](../tools/generate_teacher_dataset.py)
- [src/palmistry/teacher.py](../src/palmistry/teacher.py)
- [src/palmistry/schema.py](../src/palmistry/schema.py)
- [scripts/palmistry/generate_teacher_data.sh](../scripts/palmistry/generate_teacher_data.sh)

Typical usage:

```bash
cp configs/palmistry/teacher_generation.env.example configs/palmistry/teacher_generation.env
bash scripts/palmistry/generate_teacher_data.sh configs/palmistry/teacher_generation.env
```

Output artifacts:

- canonical SFT dataset json
- generation log jsonl

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

## 3. GRPO For Palmistry

The repo already had a generic GRPO training stack. The palmistry extension adds:

- configurable reward module loading
- palmistry-specific structured rewards
- a dedicated palmistry GRPO launch script
- support for initializing GRPO from an existing SFT LoRA adapter

Main files:

- [src/train/train_grpo.py](../src/train/train_grpo.py)
- [src/palmistry/reward_funcs_structured.py](../src/palmistry/reward_funcs_structured.py)
- [scripts/palmistry/train_grpo.sh](../scripts/palmistry/train_grpo.sh)

## 4. Current Reward Design

The structured palmistry reward module currently combines:

- JSON schema validity
- core line-field coverage
- report-field coverage
- similarity to the teacher reference answer
- safety language checks

This is designed for the current structured-JSON supervision format. If you later switch GRPO to optimize long-form natural reports instead, the clean extension path is:

1. create a new reward module such as `src.palmistry.reward_funcs_report`
2. point `reward_funcs_module` to that module
3. use report-style reference data instead of JSON-only references

## 5. Recommended Training Order

The practical order is:

1. API teacher generation
2. LoRA SFT on the generated dataset
3. GRPO initialized from the SFT adapter
4. final inference prompting for natural report style

That order is much more stable than trying to jump directly from base model into GRPO.
