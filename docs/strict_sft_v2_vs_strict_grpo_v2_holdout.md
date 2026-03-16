# Strict SFT v2 vs Strict GRPO v2 Holdout Comparison

## Scope

This note compares the strict holdout evaluation results of:

- `Strict SFT v2`
  - checkpoint: `output/palmistry_lora_qwen3_vl_8b_strict_v2/checkpoint-1758`
- `Strict GRPO v2`
  - training output: `output/palmistry_grpo_report_qwen3_vl_8b_strict_v2/checkpoint-200`
  - evaluation adapter: `output/palmistry_grpo_report_qwen3_vl_8b_strict_v2/checkpoint-200-clean-adapter`

Both evaluations use the same holdout split:

- `artifacts/palmistry_llava.generated.clean.qwen3_5_plus.judged_v2.stage_split.eval_holdout.json`
- `586` samples

Summary files:

- SFT: `artifacts/evals/palmistry_eval.strict_v2_holdout.summary.json`
- GRPO: `artifacts/evals/palmistry_eval.strict_grpo_v2_holdout.summary.json`


## Why a Clean GRPO Adapter Was Needed

`checkpoint-200` from strict GRPO contained extra `.base_layer.weight` tensors under `lm_head` and `embed_tokens`, which caused PEFT loading failures during evaluation.

A clean adapter was exported by removing those base-layer tensors while keeping the LoRA weights:

- `checkpoint-200-clean-adapter`

The comparison below is based on that cleaned adapter.


## Headline Result

On the strict holdout set, `Strict GRPO v2` did **not** improve the main grounding metrics over `Strict SFT v2`.

What changed:

- report-side `uncertainty_honesty` improved
- report-side `safety_language` improved to a perfect score

What regressed slightly:

- structured `reference_alignment`
- report `reference_alignment`
- report `section_structure`

What stayed exactly the same:

- all gate-related metrics
- `structured_available_rate`
- `full_report_rate`


## Metric Table

| Metric | Strict SFT v2 | Strict GRPO v2 | Delta |
|---|---:|---:|---:|
| `low_confidence_rate` | 0.8840 | 0.8840 | 0.0000 |
| `expected_low_confidence_rate` | 0.6553 | 0.6553 | 0.0000 |
| `gate_match_rate` | 0.7235 | 0.7235 | 0.0000 |
| `gate_continue_rate` | 0.1160 | 0.1160 | 0.0000 |
| `gate_cautious_rate` | 0.7799 | 0.7799 | 0.0000 |
| `gate_retake_rate` | 0.1041 | 0.1041 | 0.0000 |
| `structured_available_rate` | 0.8959 | 0.8959 | 0.0000 |
| `full_report_rate` | 0.1160 | 0.1160 | 0.0000 |
| `structured.reference_alignment` | 0.5269 | 0.5200 | -0.0070 |
| `structured.safety_language` | 0.9276 | 0.9432 | +0.0156 |
| `report.section_structure` | 0.9967 | 0.9824 | -0.0143 |
| `report.reference_alignment` | 0.4238 | 0.4111 | -0.0126 |
| `report.uncertainty_honesty` | 0.7390 | 0.7710 | +0.0320 |
| `report.safety_language` | 0.9412 | 1.0000 | +0.0588 |


## Interpretation

### 1. Gate metrics are unchanged

This is expected.

Both evaluations use the same standalone gate classifier:

- `output/palmistry_gate_classifier_efficientnet_b0_v1/best.pt`

So the gate decision (`continue / cautious / retake`) is mostly determined before the report model is involved. That is why:

- `gate_match_rate`
- `gate_continue_rate`
- `gate_cautious_rate`
- `gate_retake_rate`
- `structured_available_rate`
- `full_report_rate`

all remained unchanged.

This means the current strict GRPO result should be read as:

- **a report-quality update**
- **not a gate-policy update**


### 2. GRPO improved honesty and safety

The clearest gains are:

- `report.uncertainty_honesty`: `0.7390 -> 0.7710`
- `report.safety_language`: `0.9412 -> 1.0000`

This is aligned with the strict GRPO reward design:

- stronger uncertainty reward
- contradiction penalty
- safety reward

So the model is more likely to:

- explicitly acknowledge uncertainty
- avoid over-assertive wording
- use safer phrasing in the final report


### 3. GRPO slightly weakened grounding

Two alignment metrics fell:

- `structured.reference_alignment`: `0.5269 -> 0.5200`
- `report.reference_alignment`: `0.4238 -> 0.4111`

And `section_structure` also dropped slightly:

- `0.9967 -> 0.9824`

These are not catastrophic drops, but they show the current GRPO objective has started to trade some factual alignment for:

- safer language
- more uncertainty disclosure

That tradeoff is acceptable only if the product goal values conservative reporting more than tight textual alignment.


## Practical Conclusion

At the current stage:

- `Strict SFT v2` is still the better **grounding-first baseline**
- `Strict GRPO v2` is the better **safety-and-honesty tuned report model**

If the immediate priority is:

- **stable structured grounding**
  - prefer `Strict SFT v2`
- **more conservative and safer final reports**
  - prefer `Strict GRPO v2`


## Recommendation for the Next Iteration

The next strict GRPO iteration should not keep pushing only on conservative language.

It should explicitly rebalance:

1. **keep the uncertainty honesty gain**
2. **recover report/structured alignment**
3. **avoid losing section quality**

Concretely, the next reward revision should focus on:

- stronger positive reward for line-level grounding
- stronger penalty for unsupported expansions
- keeping current uncertainty contradiction penalties
- not over-rewarding generic caution language


## Current Decision

Recommended default roles:

- `Strict SFT v2`: default baseline for further controlled experiments
- `Strict GRPO v2`: report-style branch for honesty/safety optimization
