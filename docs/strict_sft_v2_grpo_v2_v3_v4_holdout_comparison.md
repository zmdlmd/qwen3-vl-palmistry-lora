# Strict SFT v2 / GRPO v2 / GRPO v3 / GRPO v4 Holdout Comparison

## Scope

This note compares four models on the same strict holdout split:

- `Strict SFT v2`
- `Strict GRPO v2`
- `Strict GRPO v3`
- `Strict GRPO v4`

Shared evaluation set:

- `artifacts/palmistry_llava.generated.clean.qwen3_5_plus.judged_v2.stage_split.eval_holdout.json`
- `586` samples

Shared gate:

- `output/palmistry_gate_classifier_efficientnet_b0_v1/best.pt`

Summary files:

- SFT v2: `artifacts/evals/palmistry_eval.strict_v2_holdout.summary.json`
- GRPO v2: `artifacts/evals/palmistry_eval.strict_grpo_v2_holdout.summary.json`
- GRPO v3: `artifacts/evals/palmistry_eval.strict_grpo_v3_holdout.summary.json`
- GRPO v4: `artifacts/evals/palmistry_eval.strict_grpo_v4_holdout.summary.json`


## Headline Result

All four runs share exactly the same gate metrics, because they use the same standalone gate classifier before report generation.

The meaningful differences are therefore concentrated in:

- structured grounding
- report grounding
- uncertainty honesty
- section quality
- safety language

Current high-level conclusion:

- `Strict SFT v2` remains the strongest grounding-first baseline
- `Strict GRPO v2` remains the best deployable report branch overall
- `Strict GRPO v3` improved some structure/structured alignment behavior, but report grounding regressed too much
- `Strict GRPO v4` recovered part of the `v3` report-grounding regression, but still does not beat `GRPO v2`


## Shared Gate Metrics

These values are identical across all four models:

- `low_confidence_rate = 0.883959`
- `expected_low_confidence_rate = 0.655290`
- `gate_match_rate = 0.723549`
- `gate_continue_rate = 0.116041`
- `gate_cautious_rate = 0.779863`
- `gate_retake_rate = 0.104096`
- `structured_available_rate = 0.895904`
- `full_report_rate = 0.116041`

Interpretation:

- this is a **report-quality comparison under a fixed gate policy**
- it is **not** a gate-policy comparison


## Comparison Table

| Metric | Strict SFT v2 | Strict GRPO v2 | Strict GRPO v3 | Strict GRPO v4 |
|---|---:|---:|---:|---:|
| `structured.reference_alignment` | 0.5269 | 0.5200 | 0.5261 | 0.5230 |
| `structured.safety_language` | 0.9276 | 0.9432 | 0.9390 | 0.9124 |
| `report.section_structure` | 0.9967 | 0.9824 | 0.9942 | 0.9826 |
| `report.reference_alignment` | 0.4238 | 0.4111 | 0.2979 | 0.3401 |
| `report.uncertainty_honesty` | 0.7390 | 0.7710 | 0.7607 | 0.7449 |
| `report.safety_language` | 0.9412 | 1.0000 | 1.0000 | 0.9853 |


## Delta vs Strict SFT v2

### Strict GRPO v2

Improvements:

- `report.uncertainty_honesty`: `0.7390 -> 0.7710`
- `report.safety_language`: `0.9412 -> 1.0000`
- `structured.safety_language`: `0.9276 -> 0.9432`

Regressions:

- `structured.reference_alignment`: `0.5269 -> 0.5200`
- `report.section_structure`: `0.9967 -> 0.9824`
- `report.reference_alignment`: `0.4238 -> 0.4111`

Interpretation:

- `GRPO v2` successfully pushed the model toward safer and more uncertainty-aware reports
- but it paid a small grounding cost to get there


### Strict GRPO v3

Improvements:

- `report.uncertainty_honesty`: `0.7390 -> 0.7607`
- `report.safety_language`: `0.9412 -> 1.0000`
- `structured.safety_language`: `0.9276 -> 0.9390`

Recovered from the `GRPO v2` regression:

- `structured.reference_alignment`: `0.5200 -> 0.5261`
- `report.section_structure`: `0.9824 -> 0.9942`

Major remaining problem:

- `report.reference_alignment`: `0.4238 -> 0.2979`

Interpretation:

- `GRPO v3` improved structure quality and partially restored structured grounding
- but report text drifted too far from the teacher reference


### Strict GRPO v4

Improvements:

- `report.uncertainty_honesty`: `0.7390 -> 0.7449`
- `report.safety_language`: `0.9412 -> 0.9853`

Recovered relative to `GRPO v3`:

- `report.reference_alignment`: `0.2979 -> 0.3401`

Still weaker than both `SFT v2` and `GRPO v2` on:

- `report.reference_alignment`
- `report.section_structure`
- `structured.safety_language`

Interpretation:

- `GRPO v4` succeeded in partially pulling report grounding back after the `v3` drop
- but the recovery was incomplete, and the new reward mix also weakened safety/structure relative to `GRPO v2`


## Direct GRPO Branch Comparison

### GRPO v2 vs GRPO v3

`GRPO v3` is better than `GRPO v2` on:

- `structured.reference_alignment`
  - `0.5200 -> 0.5261`
- `report.section_structure`
  - `0.9824 -> 0.9942`

`GRPO v3` is worse than `GRPO v2` on:

- `report.reference_alignment`
  - `0.4111 -> 0.2979`
- `report.uncertainty_honesty`
  - `0.7710 -> 0.7607`
- `structured.safety_language`
  - `0.9432 -> 0.9390`


### GRPO v3 vs GRPO v4

`GRPO v4` is better than `GRPO v3` on:

- `report.reference_alignment`
  - `0.2979 -> 0.3401`

`GRPO v4` is worse than `GRPO v3` on:

- `structured.reference_alignment`
  - `0.5261 -> 0.5230`
- `report.section_structure`
  - `0.9942 -> 0.9826`
- `report.uncertainty_honesty`
  - `0.7607 -> 0.7449`
- `report.safety_language`
  - `1.0000 -> 0.9853`

Interpretation:

- `v4` moved in the intended direction on report grounding
- but it traded away too much of the honesty/structure/safety gains that made `v2` the strongest report branch


## Practical Model Choice

### Current default deployable report branch

Choose:

- `Strict GRPO v2`

Reason:

- best overall balance between report grounding, uncertainty honesty, and safety
- only mild report/reference regression relative to the strict `SFT v2` baseline
- still the strongest practical report model in the strict line


### Strongest grounding-first baseline

Choose:

- `Strict SFT v2`

Reason:

- best `report.reference_alignment`
- best `report.section_structure`
- strongest overall grounding behavior


### Current experimental branch

Continue from:

- `Strict GRPO v4`

Reason:

- it already recovers part of the report-grounding collapse seen in `v3`
- but still needs another iteration to restore `v2`-level honesty/safety balance


## Recommended Next Step

The next strict GRPO iteration should optimize for:

1. keep `report.reference_alignment >= 0.38`
2. keep `report.uncertainty_honesty >= 0.76`
3. keep `report.safety_language >= 0.99`
4. keep `report.section_structure >= 0.99`

Concretely, the next reward revision should:

- keep the new evidence-coverage signal
- reduce the weight of generic caution-friendly reward paths
- strengthen penalties when evidence coverage is low but abstract caution wording is high
- restore stronger section/structure incentives for the final report text


## Current Decision

Recommended status:

- `Strict SFT v2`: baseline
- `Strict GRPO v2`: current best deployable report branch
- `Strict GRPO v3`: historical experimental branch
- `Strict GRPO v4`: current active research branch
