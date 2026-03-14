# Gate Classifier vs Generative Gate

## Purpose

This comparison evaluates the standalone three-class gate classifier against the older generative gate on the same fixed subsets.

The comparison only measures gate behavior. Both runs use:

- `val_mode=gate_only`
- `hard_mode=gate_only`
- the same base model and LoRA adapter
- the same fixed evaluation subsets

This isolates the gate decision itself from downstream structured parsing and report generation.

## Evaluation Sets

Fixed subsets:

- val set: [fixed_val50.json](/root/autodl-tmp/Qwen/Qwen3_FineTune/Qwen-VL-Series-Finetune/artifacts/evals/fixed_val50.json)
- hard set: [fixed_hard200.jsonl](/root/autodl-tmp/Qwen/Qwen3_FineTune/Qwen-VL-Series-Finetune/artifacts/evals/fixed_hard200.jsonl)

## Compared Systems

### 1. Generative Gate

Summary:

- [palmistry_eval.gate_ab.generative_fast.summary.json](/root/autodl-tmp/Qwen/Qwen3_FineTune/Qwen-VL-Series-Finetune/artifacts/evals/palmistry_eval.gate_ab.generative_fast.summary.json)

Behavior:

- no standalone classifier
- gate decision is produced by the multimodal model through JSON generation

### 2. Standalone Gate Classifier

Summary:

- [palmistry_eval.gate_ab.classifier_fast.summary.json](/root/autodl-tmp/Qwen/Qwen3_FineTune/Qwen-VL-Series-Finetune/artifacts/evals/palmistry_eval.gate_ab.classifier_fast.summary.json)

Checkpoint:

- [best.pt](/root/autodl-tmp/Qwen/Qwen3_FineTune/Qwen-VL-Series-Finetune/output/palmistry_gate_classifier_efficientnet_b0_v1/best.pt)

Behavior:

- gate decision comes from the standalone EfficientNet-B0 classifier
- inference runtime falls back to the generative gate only if classifier inference fails

## Results

### Fixed Val 50

Generative gate:

- `low_confidence_rate = 1.00`
- `gate_match_rate = 0.74`
- `gate_continue_rate = 0.00`
- `gate_cautious_rate = 1.00`
- `gate_retake_rate = 0.00`

Classifier gate:

- `low_confidence_rate = 0.82`
- `gate_match_rate = 0.84`
- `gate_continue_rate = 0.18`
- `gate_cautious_rate = 0.40`
- `gate_retake_rate = 0.42`

Interpretation:

- the generative gate is too conservative on this subset
- it never predicts `continue`
- the classifier is still conservative, but it has a much healthier decision spread
- the classifier improves gate-label matching from `0.74` to `0.84`

### Fixed Hard 200

Generative gate:

- `low_confidence_rate = 1.00`
- `gate_continue_rate = 0.00`
- `gate_cautious_rate = 1.00`
- `gate_retake_rate = 0.00`

Classifier gate:

- `low_confidence_rate = 0.915`
- `gate_continue_rate = 0.085`
- `gate_cautious_rate = 0.48`
- `gate_retake_rate = 0.435`

Interpretation:

- the generative gate collapses to a single output mode: always `cautious`
- the classifier differentiates between moderate and severe failure cases
- the classifier is able to assign `retake` to sharper failure buckets instead of flattening everything into one class

### Reject-Reason Breakdown

Classifier gate is especially better on strong failure reasons:

- `below_quality_floor`: `retake_rate = 0.7273`
- `below_sharpness_floor`: `retake_rate = 0.8333`
- `too_dark`: `retake_rate = 1.0`

This is exactly the behavior we want from a front-end gate:

- extreme quality failures should mostly become `retake`
- borderline failures can remain `cautious`

## Runtime Difference

The speed gap is large.

Observed behavior:

- generative gate took roughly `13+ minutes` for `50 + 200` samples
- classifier gate completed the same evaluation in about `1 second` after the shared Qwen model load

This means the standalone classifier is not only better calibrated, but also far more deployable as a front-end filter.

## Main Conclusion

The standalone gate classifier is the better default gate implementation.

Why:

- better decision diversity
- better alignment on fixed validation samples
- more meaningful separation between `cautious` and `retake`
- much faster runtime

The generative gate remains useful as a fallback or debugging path, but it is not a good default front-end gate because it collapses too easily into a single conservative mode.

## Recommended Deployment Policy

1. Use the standalone classifier as the default gate.
2. Keep the generative gate as a fallback path when the classifier checkpoint is missing or runtime inference fails.
3. Continue improving the classifier with better `cautious` labels and a real-photo holdout set.

## Next Steps

1. Add a dedicated gate-eval command for classifier-vs-generative comparison.
2. Add classifier confidence thresholds:
   - very low confidence should fall back to conservative heuristics
   - medium confidence can map to `cautious`
3. Rebuild the gate dataset after teacher judge/filter is applied, so the gate labels inherit cleaner upstream supervision.
