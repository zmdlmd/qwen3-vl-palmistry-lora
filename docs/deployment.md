# Deployment Guide

This document freezes the current recommended deployment stack for the repository.

## Default Deployment Stack

Use the following combination as the current default release:

- Base model: `Qwen3-VL-8B-Instruct`
- Report adapter: `output/palmistry_grpo_report_qwen3_vl_8b_strict_v2/checkpoint-200-clean-adapter`
- Gate classifier: `output/palmistry_gate_classifier_efficientnet_b0_v1/best.pt`
- Inference profile: `configs/palmistry/inference.strict_grpo_v2.env.example`

Why this stack:

- `strict SFT v2` is the most stable grounding-first baseline
- `strict GRPO v2` is the best current report branch for deployment
- the standalone gate classifier is a better default front-end gate than the older generative gate

Do not use `strict GRPO v3` as the default release branch yet. It is still an experimental line.

## Required Artifacts

Before deployment, confirm these files exist:

- base model directory
- `output/palmistry_grpo_report_qwen3_vl_8b_strict_v2/checkpoint-200-clean-adapter`
- `output/palmistry_gate_classifier_efficientnet_b0_v1/best.pt`

Important:

- use `checkpoint-200-clean-adapter`
- do not deploy directly from the raw `checkpoint-200`
- the clean adapter removes incompatible `.base_layer.weight` tensors from the exported adapter directory

## Recommended Runtime Behavior

The default inference path should be:

1. standalone gate classifier
2. structured palmistry analysis
3. final report generation

Expected gate behavior:

- `continue`: run the full report path
- `cautious`: keep the conservative path and allow manual continuation from the UI
- `retake`: stop and ask the user to reshoot

The standalone gate classifier also applies threshold fallback:

- low-confidence `continue` becomes `cautious`
- low-confidence `retake` becomes `cautious`

This is intentional and should remain enabled in production.

## Quick Start

### Gradio

```bash
cp configs/palmistry/inference.strict_grpo_v2.env.example configs/palmistry/inference.strict_grpo_v2.env
bash scripts/palmistry/run_gradio.sh configs/palmistry/inference.strict_grpo_v2.env
```

### CLI

```bash
cp configs/palmistry/inference.strict_grpo_v2.env.example configs/palmistry/inference.strict_grpo_v2.env
bash scripts/palmistry/run_infer.sh /path/to/hand.png configs/palmistry/inference.strict_grpo_v2.env
```

## Configuration Notes

The main configuration file is:

- `configs/palmistry/inference.strict_grpo_v2.env.example`

Typical fields to update locally:

- `BASE_MODEL_PATH`
- `DEVICE`
- `DEVICE_MAP`
- `GRADIO_SERVER_PORT`

Usually you should not change these without a reason:

- `LORA_PATH`
- `GATE_CLASSIFIER_PATH`
- `GATE_CLASSIFIER_MIN_CONFIDENCE`
- `GATE_CLASSIFIER_CONTINUE_MIN_CONFIDENCE`
- `GATE_CLASSIFIER_RETAKE_MIN_CONFIDENCE`
- `GATE_CLASSIFIER_MIN_MARGIN`

## AutoDL Example

Example launch pattern on AutoDL:

```bash
cd /root/autodl-tmp/Qwen/Qwen3_FineTune/Qwen-VL-Series-Finetune
cp configs/palmistry/inference.strict_grpo_v2.env.example configs/palmistry/inference.strict_grpo_v2.env
bash scripts/palmistry/run_gradio.sh configs/palmistry/inference.strict_grpo_v2.env
```

If you need a public demo endpoint, map the chosen Gradio port in the AutoDL console.

## Release Checklist

Before calling a version deployable, confirm:

- the base model path is correct
- the adapter path points to `strict_v2/checkpoint-200-clean-adapter`
- the gate classifier path is correct
- Gradio can load and answer at least one `continue` sample
- Gradio can reject at least one `retake` sample
- the JSON tabs in the UI still render correctly
- the follow-up chat appears only after a report is generated

## Known Limitations

- the model still depends on the current palmistry prompt design
- `strict GRPO v2` improves uncertainty honesty, but it is not perfect
- the current evaluation set is still derived from the same main dataset family
- a real external photo set is still needed for final generalization validation

## Next Planned Upgrade

After this release baseline is frozen, the next research step should be:

- `strict GRPO v4`

Its goal should be narrow and explicit:

- preserve uncertainty honesty
- recover report reference alignment
- avoid unnecessary over-cautious behavior on normal images
