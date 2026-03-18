# Strict SFT v2 / GRPO v2 / GRPO v3 Holdout Comparison

## Scope

This note compares three models on the same strict holdout split:

- `Strict SFT v2`
- `Strict GRPO v2`
- `Strict GRPO v3`

Shared evaluation set:

- `artifacts/palmistry_llava.generated.clean.qwen3_5_plus.judged_v2.stage_split.eval_holdout.json`
- `586` samples

Shared gate:

- `output/palmistry_gate_classifier_efficientnet_b0_v1/best.pt`

Summary files:

- SFT v2: `artifacts/evals/palmistry_eval.strict_v2_holdout.summary.json`
- GRPO v2: `artifacts/evals/palmistry_eval.strict_grpo_v2_holdout.summary.json`
- GRPO v3: `artifacts/evals/palmistry_eval.strict_grpo_v3_holdout.summary.json`


## Headline Result

The three runs share exactly the same gate metrics, because they all use the same standalone gate classifier before the report model is invoked.

The meaningful differences are therefore concentrated in:

- structured grounding
- report grounding
- uncertainty honesty
- section quality
- safety language

Current high-level conclusion:

- `Strict SFT v2` remains the best grounding-first baseline
- `Strict GRPO v2` remains the best honesty/safety-tuned branch
- `Strict GRPO v3` recovered part of the grounding regression seen in `GRPO v2`, but it still underperforms on report grounding and does not beat `GRPO v2` on uncertainty honesty


## Shared Gate Metrics

These values are identical across all three models:

- `low_confidence_rate = 0.883959`
- `expected_low_confidence_rate = 0.655290`
- `gate_match_rate = 0.723549`
- `gate_continue_rate = 0.116041`
- `gate_cautious_rate = 0.779863`
- `gate_retake_rate = 0.104096`
- `structured_available_rate = 0.895904`
- `full_report_rate = 0.116041`

Interpretation:

- this comparison is effectively a **report-quality comparison under a fixed gate policy**
- it is **not** a gate-policy comparison


## Comparison Table

| Metric | Strict SFT v2 | Strict GRPO v2 | Strict GRPO v3 |
|---|---:|---:|---:|
| `structured.reference_alignment` | 0.5269 | 0.5200 | 0.5261 |
| `structured.safety_language` | 0.9276 | 0.9432 | 0.9390 |
| `report.section_structure` | 0.9967 | 0.9824 | 0.9942 |
| `report.reference_alignment` | 0.4238 | 0.4111 | 0.2979 |
| `report.uncertainty_honesty` | 0.7390 | 0.7710 | 0.7607 |
| `report.safety_language` | 0.9412 | 1.0000 | 1.0000 |


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

- `GRPO v3` did improve structure and partially restore grounding on the structured side
- however, the report text itself drifted much further from the teacher reference than either `SFT v2` or `GRPO v2`


## Direct GRPO v2 vs GRPO v3

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

Equivalent:

- `report.safety_language = 1.0000`

Interpretation:

- the `v3` reward revision succeeded in reducing some of the section/structure regression
- but it over-corrected toward a weaker report/reference coupling
- so `v3` is not yet a replacement for `v2`


## Practical Model Choice

### If you want the current default model

Choose:

- `Strict GRPO v2`

Reason:

- best report-side uncertainty honesty
- perfect report safety score
- only mild alignment regression relative to `SFT v2`


### If you want the safest grounding-first baseline

Choose:

- `Strict SFT v2`

Reason:

- best report/reference alignment among the three
- best overall grounding balance
- still reasonably strong uncertainty honesty


### If you want the next research branch

Continue from:

- `Strict GRPO v3`

Reason:

- its reward redesign already recovered structure quality and structured alignment
- but it still needs a new iteration specifically to restore report/reference alignment


## Recommended Next Step

The next strict GRPO iteration should optimize for:

1. keep `report.uncertainty_honesty >= 0.76`
2. keep `report.safety_language = 1.0`
3. push `report.reference_alignment` back above `0.38`
4. keep `report.section_structure >= 0.99`

Concretely, the next reward revision should:

- reduce the degree to which generic caution wording can score well on its own
- reward line-grounded paraphrase more than abstract caution framing
- add stronger penalties when the report omits concrete teacher-supported line evidence


## Current Decision

Recommended status:

- `Strict SFT v2`: baseline
- `Strict GRPO v2`: current best deployable report branch
- `Strict GRPO v3`: active experimental branch, not yet promoted
