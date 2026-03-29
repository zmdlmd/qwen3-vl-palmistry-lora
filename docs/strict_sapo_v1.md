# Strict SAPO v1

## Overview

`strict SAPO v1` is the first soft-adaptive policy optimization prototype built on top of the strict `SFT v2` palmistry baseline. It keeps the existing strict report reward stack and replaces the hard-clipped GRPO/DAPO policy loss with a soft-gated update rule.

This experiment is designed to answer one narrow question:

- Can a softer update rule preserve report coherence and uncertainty honesty while recovering some of the report grounding/alignment lost in later strict GRPO reward iterations?

The implementation is intentionally minimal:

- Same base model as the strict GRPO runs
- Same report GRPO dataset
- Same reward functions as the current strict GRPO branch
- Only the policy loss changes

## Motivation

The strict palmistry RL runs exposed a recurring pattern:

- `strict GRPO v2` remained the best default deployment branch
- `strict GRPO v3/v4` improved some grounding-related behaviors
- but long natural-language reports still suffered from local section drift

In practice, the problem was not full training instability. Instead, it was more often:

- most of the report is acceptable
- but one or two line-level sections become over-confident or under-supported
- hard clipping then treats the update too coarsely

That makes this task a reasonable candidate for a soft adaptive policy update rather than another reward-only iteration.

## What Changed

### 1. Custom SAPO loss in the trainer

File:
- `src/trainer/grpo_trainer.py`

A new helper computes the soft gate from the importance ratio:

```python
center = self.args.sapo_center
temperature = self.args.sapo_temperature
min_weight = self.args.sapo_min_weight

distance = torch.abs(ratio - center)
gate = torch.exp(-temperature * distance)
gate = min_weight + (1.0 - min_weight) * gate
```

Interpretation:

- when the importance ratio stays close to the target center (`1.0`), the raw ratio is trusted more
- when the ratio drifts away, the update is gradually pulled toward the clipped ratio
- unlike hard clipping, the transition is continuous

The SAPO branch overrides `_compute_loss(...)` only when `loss_type == "sapo"`.

### 2. Replacing hard clipping with soft interpolation

Upstream GRPO/DAPO uses the usual clipped form:

```text
ratio = exp(logpi - logpi_old)
ratio_clip = clip(ratio, 1 - eps_low, 1 + eps_high)
loss = -min(ratio * A, ratio_clip * A)
```

In `strict SAPO v1`, that becomes:

```text
gate = soft_gate(ratio)
ratio_soft = gate * ratio + (1 - gate) * ratio_clip
loss = -(ratio_soft * A)
```

This means:

- small deviations keep more of the raw policy update
- larger deviations are softly damped rather than abruptly truncated
- the objective is still compatible with the existing reward pipeline

### 3. New SAPO hyperparameters

File:
- `src/params.py`

Added:

- `sapo_temperature`
- `sapo_center`
- `sapo_min_weight`

Default values:

- `sapo_temperature = 4.0`
- `sapo_center = 1.0`
- `sapo_min_weight = 0.2`

These control how aggressively the soft gate suppresses updates far from the desired ratio region.

### 4. Training script support

File:
- `scripts/palmistry/train_grpo.sh`

New env/CLI options:

- `LOSS_TYPE`
- `IMPORTANCE_SAMPLING_LEVEL`
- `SAPO_TEMPERATURE`
- `SAPO_CENTER`
- `SAPO_MIN_WEIGHT`

This allows SAPO to be switched on without changing the data or reward stack.

### 5. Dedicated strict SAPO config template

File:
- `configs/palmistry/grpo_report_strict_sapo_v1.env.example`

Key settings:

- `LOSS_TYPE="sapo"`
- `IMPORTANCE_SAMPLING_LEVEL="sequence"`
- `SAPO_TEMPERATURE=4.0`
- `SAPO_CENTER=1.0`
- `SAPO_MIN_WEIGHT=0.2`

The `sequence` importance level was chosen deliberately because the target behavior is report-level quality, not token-level short-form optimization.

## Why Sequence-Level SAPO

For this palmistry task, the model outputs long multi-section reports. The desired behavior is largely sequence-level:

- preserve overall report structure
- remain grounded in teacher evidence
- stay honest about uncertainty
- avoid unsupported over-specific claims

Using `importance_sampling_level="sequence"` makes the ratio correspond to the whole completion more naturally. SAPO then softens that sequence-level update instead of sharply clipping it.

This is a better fit than purely token-level clipping for a task where a report can be globally good but locally noisy.

## What Did Not Change

To keep the experiment attributable, the following were intentionally not changed:

- no new teacher data
- no new reward module
- no prompt rewrite
- no new gate classifier
- no new dataset split

`strict SAPO v1` reuses the current strict report reward stack and the strict `SFT v2` initialization.

## Runtime Instrumentation

The trainer logs three additional SAPO metrics:

- `sapo_gate/mean`
- `sapo_gate/min`
- `sapo_gate/max`

These are useful for checking whether SAPO is actually doing anything:

- values near `1.0` mean the batch is staying near the center region
- lower values indicate the update is being softened toward the clipped path

## Smoke Validation

A GPU `1-step` smoke run completed successfully.

Observed signals:

- training completed with `exit 0`
- reward computation worked normally
- backward/update path worked
- SAPO metrics were logged:
  - `sapo_gate/mean = 1.0`
  - `sapo_gate/min = 1.0`
  - `sapo_gate/max = 1.0`

This confirmed that:

- the `loss_type="sapo"` branch was actually executed
- the trainer, reward stack, and optimizer all remained compatible

## Experiment Goal

`strict SAPO v1` is not meant to outperform all previous runs automatically.

Its narrow target is:

- keep the report-level uncertainty behavior competitive with strict GRPO
- recover some report grounding/alignment without collapsing structure

Success should be evaluated on the same strict holdout used for:

- `strict SFT v2`
- `strict GRPO v2`
- `strict GRPO v3`
- `strict GRPO v4`

## Expected Comparison Focus

The most important metrics for SAPO should be:

- `report.reference_alignment`
- `report.uncertainty_honesty`
- `report.section_structure`
- `structured.reference_alignment`

A useful SAPO outcome would be:

- higher `report.reference_alignment` than `strict GRPO v4`
- no major collapse in `uncertainty_honesty`
- report structure remaining near the current strict GRPO runs

## Current Status

Implementation is complete.

Files changed for the initial SAPO prototype:

- `src/trainer/grpo_trainer.py`
- `src/params.py`
- `scripts/palmistry/train_grpo.sh`
- `configs/palmistry/grpo_report_strict_sapo_v1.env.example`

Formal training and evaluation should continue using the strict deployment/experiment workflow already established in this repository.
