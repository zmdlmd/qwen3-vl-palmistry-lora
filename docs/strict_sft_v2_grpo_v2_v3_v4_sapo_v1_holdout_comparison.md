# Strict SFT v2 vs Strict GRPO v2/v3/v4 vs Strict SAPO v1

## Scope

This note compares five report-generation branches on the same strict holdout:

- `strict SFT v2`
- `strict GRPO v2`
- `strict GRPO v3`
- `strict GRPO v4`
- `strict SAPO v1`

All five evaluations use the same:

- holdout set: `judged_v2.stage_split.eval_holdout.json`
- base model family: `Qwen3-VL-8B-Instruct`
- standalone gate classifier

Because the gate classifier is fixed across all runs, gate-related metrics are expected to stay constant. The real comparison is in structured grounding quality and report behavior.

## Holdout Metrics

| Variant | Structured Ref Align | Structured Safety | Report Section | Report Ref Align | Report Uncertainty Honesty | Report Safety |
|---|---:|---:|---:|---:|---:|---:|
| strict SFT v2 | 0.5269 | 0.9276 | 0.9967 | 0.4238 | 0.7390 | 0.9412 |
| strict GRPO v2 | 0.5200 | 0.9432 | 0.9824 | 0.4111 | 0.7710 | 1.0000 |
| strict GRPO v3 | 0.5261 | 0.9390 | 0.9942 | 0.2979 | 0.7607 | 1.0000 |
| strict GRPO v4 | 0.5230 | 0.9124 | 0.9826 | 0.3401 | 0.7449 | 0.9853 |
| strict SAPO v1 | 0.5245 | 0.8038 | 0.9926 | 0.3577 | 0.7676 | 0.9412 |

Shared gate metrics for all five runs:

- `low_confidence_rate = 0.8840`
- `expected_low_confidence_rate = 0.6553`
- `gate_match_rate = 0.7235`
- `structured_available_rate = 0.8959`
- `full_report_rate = 0.1160`

## Main Findings

### 1. Strict SFT v2 remains the strongest grounding-first baseline

`strict SFT v2` still has the best `report_reference_alignment` among all five branches:

- `strict SFT v2`: `0.4238`
- `strict GRPO v2`: `0.4111`
- `strict SAPO v1`: `0.3577`
- `strict GRPO v4`: `0.3401`
- `strict GRPO v3`: `0.2979`

This means no RL branch has yet surpassed the strict SFT baseline on report-level grounding.

### 2. Strict GRPO v2 remains the best default deployment branch

`strict GRPO v2` still has the best overall balance for deployment:

- second-best report alignment after strict SFT
- best `report_uncertainty_honesty` among the GRPO family
- perfect `report_safety_language = 1.0`
- better section quality than `v4`

It remains the best compromise between:

- alignment
- conservative uncertainty behavior
- safe language

### 3. Strict GRPO v3 over-corrected toward structure

`strict GRPO v3` recovered section organization:

- `report_section_structure = 0.9942`

and preserved strong safety:

- `report_safety_language = 1.0`

but its report grounding collapsed too far:

- `report_reference_alignment = 0.2979`

This made it unsuitable as a replacement for `strict GRPO v2`.

### 4. Strict GRPO v4 partially repaired the v3 collapse

`strict GRPO v4` improved report grounding over `v3`:

- `v3`: `0.2979`
- `v4`: `0.3401`

but still remained clearly behind `strict GRPO v2`.

It also did not meaningfully surpass `v2` on uncertainty behavior:

- `v4 uncertainty_honesty = 0.7449`
- `v2 uncertainty_honesty = 0.7710`

So `v4` was a useful research iteration, but not a deployment winner.

### 5. Strict SAPO v1 is better than GRPO v4, but not yet better than GRPO v2

`strict SAPO v1` is the first optimizer-side change rather than another reward-only iteration. Compared with `strict GRPO v4`, it moved in the right direction:

- `report_reference_alignment`: `0.3401 -> 0.3577`
- `report_uncertainty_honesty`: `0.7449 -> 0.7676`
- `report_section_structure`: `0.9826 -> 0.9926`

This is a real improvement. SAPO v1 recovered:

- better report grounding than `v4`
- better section structure than `v4`
- stronger uncertainty honesty than `v4`

However, it still does not beat `strict GRPO v2` on the most important report metric:

- `strict GRPO v2 report_reference_alignment = 0.4111`
- `strict SAPO v1 report_reference_alignment = 0.3577`

## SAPO-Specific Interpretation

The SAPO hypothesis was:

- keep the reward stack from strict GRPO
- replace hard clipping with a softer policy update
- recover report grounding without breaking uncertainty handling

The holdout result suggests:

- this direction is promising
- optimizer-side changes helped more than `strict GRPO v4`
- but the first SAPO prototype is still not strong enough to replace `strict GRPO v2`

The most encouraging point is that SAPO v1 improved multiple report metrics at once relative to `v4`, which is a better sign than the `v3 -> v4` reward-only path.

## Current Ranking

### Deployment ranking

1. `strict GRPO v2`
2. `strict SFT v2`
3. `strict SAPO v1`
4. `strict GRPO v4`
5. `strict GRPO v3`

### Research ranking

1. `strict SAPO v1`
2. `strict GRPO v4`
3. `strict GRPO v3`

Reason:

- `strict GRPO v2` is already the default deployment winner
- `strict SAPO v1` is the most promising research branch because it improved over `v4` without changing the data or reward stack

## Recommendation

Keep the default deployment stack unchanged:

- `strict GRPO v2 clean adapter`
- standalone gate classifier

Use `strict SAPO v1` as the next research branch.

The next iteration should focus on:

- recovering another step of `report_reference_alignment`
- keeping `uncertainty_honesty` at or above the current SAPO level
- improving structured/report safety language regression seen in SAPO v1

## Files

- `artifacts/evals/palmistry_eval.strict_v2_holdout.summary.json`
- `artifacts/evals/palmistry_eval.strict_grpo_v2_holdout.summary.json`
- `artifacts/evals/palmistry_eval.strict_grpo_v3_holdout.summary.json`
- `artifacts/evals/palmistry_eval.strict_grpo_v4_holdout.summary.json`
- `artifacts/evals/palmistry_eval.strict_sapo_v1_holdout.summary.json`
