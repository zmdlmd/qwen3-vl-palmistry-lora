<p align="center">
  <img src="docs/assets/repo-hero.svg" alt="Qwen3-VL Palmistry LoRA" width="100%" />
</p>

<h1 align="center">Qwen3-VL Palmistry LoRA</h1>

<p align="center">
  A GitHub-ready palmistry fine-tuning project built on top of Qwen-VL-Series-Finetune.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Qwen3--VL-2B%20%7C%204B%20%7C%208B-orange" alt="Qwen3-VL" />
  <img src="https://img.shields.io/badge/Training-LoRA%20SFT-blue" alt="LoRA SFT" />
  <img src="https://img.shields.io/badge/Input-Hand%20Images-0f766e" alt="Hand Images" />
  <img src="https://img.shields.io/badge/Output-Chinese%20Palmistry%20Report-7c3aed" alt="Chinese Palmistry Report" />
  <img src="https://img.shields.io/badge/License-Apache%202.0-lightgrey" alt="Apache 2.0" />
</p>

## 中文简介

这是一个基于 `Qwen3-VL` 的手相图像 LoRA 微调项目。其训练流程包括：

- 用 GPT-5 生成手相结构化标注作为 teacher 数据
- 用 `Qwen3-VL` 作为 student 模型
- 用 LoRA SFT 学习“读取手部图片并输出掌纹分析”
- 在推理阶段再通过 prompt，把结构化理解展开成自然中文手相报告

该仓库保留了上游 `Qwen-VL-Series-Finetune` 的通用训练内核，同时将 palmistry 任务相关的数据、训练、推理与展示层整理为独立模块，适合公开发布与持续迭代。

## English Overview

This repository adapts the upstream `Qwen-VL-Series-Finetune` framework into a palmistry-focused multimodal fine-tuning project.

The current pipeline is:

- GPT-5 generated palmistry annotations as teacher data
- `Qwen3-VL` as the student model
- LoRA SFT for image-conditioned palm-line understanding
- Prompt-based report generation for final natural Chinese responses

## Why This Repo

- Clear separation between upstream training core and palmistry-specific application code
- GitHub-safe structure with datasets, checkpoints, logs, and local symlinks excluded from version control
- Reusable CLI and Gradio entrypoints for demo and deployment
- Config-driven training wrapper instead of machine-specific hardcoded shell scripts
- Documentation that explains what is actually being learned by the LoRA

## Project Snapshot

```text
GPT-5 palmistry labels
        ↓
LLaVA-style hand-image dataset
        ↓
src/train/train_sft.py
        ↓
Qwen3-VL + LoRA adapters
        ↓
Palmistry inference pipeline
        ↓
CLI / Gradio Chinese report generation
```

## Extended Workflow

This repository now supports two complementary stages beyond basic SFT:

1. Automated teacher-data distillation  
   Call a large multimodal model through an OpenAI-compatible API, validate the returned palmistry JSON, and write the result directly into a LLaVA-style SFT dataset.

2. GRPO post-training  
   Start from a base model or an existing SFT LoRA adapter, then optimize with palmistry-specific reward functions.

## What Is Actually In Here

- Core SFT trainer: [src/train/train_sft.py](src/train/train_sft.py)
- Palmistry training wrapper: [scripts/palmistry/train_lora.sh](scripts/palmistry/train_lora.sh)
- Palmistry prompts: [src/palmistry/prompts.py](src/palmistry/prompts.py)
- Palmistry schema + teacher pipeline: [src/palmistry/schema.py](src/palmistry/schema.py), [src/palmistry/teacher.py](src/palmistry/teacher.py)
- Palmistry inference pipeline: [src/palmistry/pipeline.py](src/palmistry/pipeline.py)
- CLI inference: [tools/infer_palmistry.py](tools/infer_palmistry.py)
- Teacher data generation CLI: [tools/generate_teacher_dataset.py](tools/generate_teacher_dataset.py)
- Gradio demo: [apps/gradio_palmistry.py](apps/gradio_palmistry.py)
- Adapter export tool: [tools/export_peft_adapter.py](tools/export_peft_adapter.py)
- Architecture notes: [docs/architecture.md](docs/architecture.md)
- Distillation + GRPO notes: [docs/distillation_and_grpo.md](docs/distillation_and_grpo.md)
- Dataset notes: [data/README.md](data/README.md)

## Repository Layout

```text
.
├── apps/
│   └── gradio_palmistry.py
├── configs/
│   └── palmistry/
│       ├── inference.env.example
│       └── train_lora.env.example
├── data/
│   ├── README.md
│   └── examples/
├── docs/
│   ├── architecture.md
│   └── assets/
├── scripts/
│   ├── palmistry/
│   └── zero*.json
├── src/
│   ├── palmistry/
│   └── train/
└── tools/
```

## Training Framework

The training backbone remains the upstream multimodal SFT stack. The palmistry layer mainly standardizes:

- model path
- dataset path
- image folder path
- LoRA settings
- DeepSpeed launch config
- inference prompt style

One important project-specific detail:

- the current `data/palmistry_llava.json` labels are mainly GPT-5 generated structured JSON strings
- so the LoRA is learning visual palm-line interpretation and structured analysis first
- the final long-form natural Chinese report style is mostly controlled by inference prompts

That means this repo is best understood as:

- a hand-image understanding LoRA
- plus a palmistry report generation prompt layer

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt -f https://download.pytorch.org/whl/cu128
pip install qwen-vl-utils
pip install flash-attn --no-build-isolation
```

Or:

```bash
conda env create -f environment.yaml
conda activate train
pip install qwen-vl-utils
pip install flash-attn --no-build-isolation
```

### 2. Configure Training

```bash
cp configs/palmistry/train_lora.env.example configs/palmistry/train_lora.env
```

Then edit:

- `BASE_MODEL_PATH`
- `DATA_PATH`
- `IMAGE_FOLDER`
- `OUTPUT_DIR`

### 3. Launch LoRA Training

```bash
bash scripts/palmistry/train_lora.sh configs/palmistry/train_lora.env
```

### 3.5. Generate Teacher Distillation Data

```bash
cp configs/palmistry/teacher_generation.env.example configs/palmistry/teacher_generation.env
bash scripts/palmistry/generate_teacher_data.sh configs/palmistry/teacher_generation.env
```

The teacher API is OpenAI-compatible. For DashScope, Qwen3.5 multimodal models fit this workflow, while the actual API `model` value can be `qwen-plus` or `qwen-vl-plus`.

Main entrypoints:

- [tools/generate_teacher_dataset.py](tools/generate_teacher_dataset.py)
- [scripts/palmistry/generate_teacher_data.sh](scripts/palmistry/generate_teacher_data.sh)
- [docs/distillation_and_grpo.md](docs/distillation_and_grpo.md)

### 4. Run CLI Inference

```bash
python -m tools.infer_palmistry \
  --base-model /path/to/Qwen3-VL-8B-Instruct \
  --lora-path ./output/palmistry_lora_qwen3_vl_8b \
  --image /path/to/hand.png \
  --style balanced
```

### 5. Run Gradio Demo

```bash
python -m apps.gradio_palmistry \
  --base-model /path/to/Qwen3-VL-8B-Instruct \
  --lora-path ./output/palmistry_lora_qwen3_vl_8b
```

## GRPO Reinforcement Learning

The repository now supports palmistry-specific GRPO training with configurable reward modules.

Recommended usage:

```bash
cp configs/palmistry/grpo.env.example configs/palmistry/grpo.env
bash scripts/palmistry/train_grpo.sh configs/palmistry/grpo.env
```

Key points:

- `reward_funcs_module` is now configurable
- `src.palmistry.reward_funcs_structured` provides structured palmistry rewards
- `lora_weight_path` can be used to initialize GRPO from an existing SFT LoRA adapter

## Data Format

The training data follows a LLaVA-style single-image conversation format:

- `image`: relative image filename
- `conversations[0]`: user prompt containing `<image>`
- `conversations[1]`: assistant target string

Tracked example:

- [data/examples/palmistry_llava.sample.json](data/examples/palmistry_llava.sample.json)

Data notes:

- Real hand images are not included in this public repo
- Full GPT-5 annotation files are not included either
- Local testing symlinks, checkpoints, and private assets are ignored by git

## Public Repository Safety

This repository intentionally does not track:

- `data/images/`
- `data/palmistry_llava.json`
- `data/test_palmdata.json`
- `output/`
- `scripts/train_log/`
- `ssh.txt`
- local test symlinks under `data/`

See [.gitignore](.gitignore) for the current public-safe rules.

## Notes On Scope

This project is for research, experimentation, and creative interaction design. Palmistry outputs are not medical, legal, or factual diagnoses.

If you want stronger final prose quality, the next obvious upgrade is a second-stage SFT dataset where the target outputs are already polished natural-language reports instead of structured JSON.

## Acknowledgements

- Upstream base project: `2U1/Qwen-VL-Series-Finetune`
- Multimodal model family: `Qwen3-VL`

## License

This repository keeps the upstream Apache 2.0 license.
