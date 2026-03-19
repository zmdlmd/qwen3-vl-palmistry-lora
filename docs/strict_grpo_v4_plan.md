# Strict GRPO v4 Plan

## Goal

`strict GRPO v4` is a targeted follow-up to `strict GRPO v3`.

The goal is not to make the report more conservative in general.

The goal is:

- keep `report uncertainty_honesty` near the `strict GRPO v2` level
- recover `report reference_alignment` from the `v3` regression
- keep `report safety_language = 1.0`

## Why v4 Is Needed

From the shared strict holdout comparison:

- `strict GRPO v2` is still the best default deployable report branch
- `strict GRPO v3` recovered part of the structural regression
- but `strict GRPO v3` dropped too far on `report reference_alignment`

That means the current reward still lets generic caution wording score too well relative to line-grounded report content.

## v4 Reward Changes

`strict GRPO v4` focuses on three changes inside `src/palmistry/reward_funcs_report.py`.

### 1. Stronger line evidence coverage

Added:

- `evidence_coverage_reward`

Purpose:

- reward reports that actually mention teacher-supported line evidence
- reward line sections that carry concrete detail instead of only abstract caution framing

Implementation idea:

- extract teacher-supported evidence terms from each line reference
- measure whether the generated line section covers those terms
- combine this with the existing per-line grounding score

### 2. Reference alignment now uses evidence coverage

Updated:

- `reference_alignment_reward`

Change:

- the line-level component now uses evidence coverage instead of the older looser grounding-only score

Purpose:

- pull the report closer to teacher-supported line detail
- reduce the chance that a structurally correct but weakly grounded report scores too well

### 3. Caution balance reward

Added:

- `caution_balance_reward`

Purpose:

- penalize overuse of generic caution wording on samples that should still contain concrete line evidence
- keep `retake` and genuinely uncertain samples conservative
- stop rewarding vague reports on otherwise analyzable samples

## Expected Direction

If `v4` works as intended:

- `report reference_alignment` should improve relative to `v3`
- `report uncertainty_honesty` should stay close to the `v2` level
- `report section_structure` should remain high
- `report safety_language` should remain at `1.0`

## Training Config

Suggested config file:

- `configs/palmistry/grpo_report_strict_v4.env.example`

Base initialization:

- strict `SFT v2` checkpoint
- same strict `grpo_train` split

Keep the rest close to `strict GRPO v3` so the comparison remains attributable to reward changes.

## Evaluation Protocol

Evaluate `v4` on the same strict holdout used by:

- `strict SFT v2`
- `strict GRPO v2`
- `strict GRPO v3`

Primary metrics:

- `report reference_alignment`
- `report uncertainty_honesty`
- `report safety_language`
- `report section_structure`

## Promotion Rule

Promote `strict GRPO v4` only if:

- it clearly beats `v3` on `report reference_alignment`
- it does not fall below `strict GRPO v2` by a large margin on `report uncertainty_honesty`
- it keeps `report safety_language = 1.0`
