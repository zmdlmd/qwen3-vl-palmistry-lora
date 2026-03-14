# Report GRPO v2 vs v3 Fixed-Eval Comparison

## Scope

This note compares `report GRPO v2` and `report GRPO v3` on the same fixed evaluation subsets:

- `fixed_val50`
- `fixed_hard200`
- `uncertainty_challenge_200`

Summary sources:

- `v2 fixed val50`: `/root/autodl-tmp/Qwen/Qwen3_FineTune/Qwen-VL-Series-Finetune/artifacts/evals/palmistry_eval.v2_fixed_val50.summary.json`
- `v3 fixed val50`: `/root/autodl-tmp/Qwen/Qwen3_FineTune/Qwen-VL-Series-Finetune/artifacts/evals/palmistry_eval.v3_fixed_val50.summary.json`
- `v2 fixed hard200`: `/root/autodl-tmp/Qwen/Qwen3_FineTune/Qwen-VL-Series-Finetune/artifacts/evals/palmistry_eval.v2_fixed_hard200.summary.json`
- `v3 fixed hard200`: `/root/autodl-tmp/Qwen/Qwen3_FineTune/Qwen-VL-Series-Finetune/artifacts/evals/palmistry_eval.v3_fixed_hard200.summary.json`
- `v2 uncertainty200`: `/root/autodl-tmp/Qwen/Qwen3_FineTune/Qwen-VL-Series-Finetune/artifacts/evals/palmistry_eval.v2_uncertainty200.summary.json`
- `v3 uncertainty200`: `/root/autodl-tmp/Qwen/Qwen3_FineTune/Qwen-VL-Series-Finetune/artifacts/evals/palmistry_eval.v3_uncertainty200.summary.json`

## High-Level Conclusion

`v3` did not produce a universal improvement over `v2`.

- On normal-but-usable samples (`fixed_val50`), `v2` remains better overall.
- On the uncertainty-focused challenge set (`uncertainty_challenge_200`), `v3` is directionally better: it becomes more conservative, reports less aggressively, and slightly improves uncertainty honesty.
- On hard gating-only samples (`fixed_hard200`), the two versions are effectively tied, with `v2` marginally more conservative.

Practical takeaway:

- If the immediate priority is better coverage and stronger report quality on ordinary usable images, keep `v2`.
- If the immediate priority is improving behavior on ambiguous images and uncertainty-sensitive cases, keep iterating from `v3`.

## Metric Tables

### 1. fixed_val50

| Metric | v2 | v3 | Delta (v3 - v2) |
| --- | ---: | ---: | ---: |
| low_confidence_rate | 0.30 | 0.48 | +0.18 |
| gate_match_rate | 0.44 | 0.46 | +0.02 |
| structured_available_rate | 0.96 | 0.96 | 0.00 |
| full_report_rate | 0.70 | 0.52 | -0.18 |
| report uncertainty_honesty | 0.4932 | 0.4582 | -0.0350 |
| report reference_alignment | 0.4025 | 0.3948 | -0.0078 |

Interpretation:

- `v3` is clearly more conservative on usable images.
- That conservatism did not translate into better report quality on this split.
- `v2` keeps more report coverage and slightly better honesty/alignment on normal samples.

### 2. fixed_hard200

| Metric | v2 | v3 | Delta (v3 - v2) |
| --- | ---: | ---: | ---: |
| low_confidence_rate | 0.935 | 0.925 | -0.01 |
| visibility_cautious_rate | 0.93 | 0.92 | -0.01 |
| visibility_retake_rate | 0.10 | 0.10 | 0.00 |
| full_report_rate | 0.00 | 0.00 | 0.00 |

Interpretation:

- Both versions already behave conservatively on clearly hard cases.
- `v2` is marginally stricter, but the difference is too small to matter operationally.
- This split is not where `v3` shows its main gain.

### 3. uncertainty_challenge_200

| Metric | v2 | v3 | Delta (v3 - v2) |
| --- | ---: | ---: | ---: |
| low_confidence_rate | 0.16 | 0.35 | +0.19 |
| gate_match_rate | 0.16 | 0.35 | +0.19 |
| structured_available_rate | 0.995 | 0.995 | 0.00 |
| full_report_rate | 0.84 | 0.65 | -0.19 |
| report uncertainty_honesty | 0.2762 | 0.2927 | +0.0165 |
| report reference_alignment | 0.3893 | 0.3919 | +0.0026 |

Interpretation:

- This is the split where `v3` does what it was designed to do.
- `v3` is substantially more willing to mark uncertain samples as low-confidence.
- It reduces aggressive full-report generation on ambiguity-heavy samples.
- It also improves uncertainty honesty, but only slightly.

## What Changed Between v2 and v3

The main difference was the reward design, not the training setup:

- same base model
- same SFT starting adapter
- same report-GRPO data
- same training scale

`v3` added stronger uncertainty-focused reward shaping:

- line-level uncertainty honesty
- contradiction penalty for "cautious wording + assertive detail"
- expanded uncertainty lexicon

These changes improved challenge-set behavior, but they also made the model more conservative on normal samples.

## Recommended Decision

Current recommendation:

- Treat `v2` as the stronger general-use reporting checkpoint.
- Treat `v3` as the better uncertainty-aware experimental branch.

If only one branch should be continued for the next uncertainty-focused RL round, choose `v3`.

Reason:

- The target problem was uncertainty handling.
- `v3` improves the relevant split (`uncertainty_challenge_200`) without breaking hard-case gating.
- The remaining gap is not "more of the same reward"; it is calibration quality on ambiguous-but-not-hopeless images.

## Next Steps

1. Build a mixed calibration objective instead of only pushing uncertainty harder.
2. Add a reward term that separates:
   - cautious analysis
   - retake recommendation
3. Add a penalty for unnecessary low-confidence decisions on normal usable images.
4. Keep using the fixed subsets above for all future A/B comparisons.

## Bottom Line

`v2` is still better as the current "balanced default" model.

`v3` is better as the current "uncertainty-aware research branch".

The next iteration should start from `v3`, but the optimization target should shift from "be more conservative" to "be selectively conservative only when justified".
