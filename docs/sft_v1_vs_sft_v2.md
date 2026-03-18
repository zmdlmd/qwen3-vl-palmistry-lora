# SFT v1 vs Strict SFT v2

## Scope

This note compares the two SFT baselines currently present in the project:

- `SFT v1`
  - output: `output/palmistry_lora_qwen3_vl_8b_clean_v1`
- `Strict SFT v2`
  - output: `output/palmistry_lora_qwen3_vl_8b_strict_v2`
  - usable checkpoint: `output/palmistry_lora_qwen3_vl_8b_strict_v2/checkpoint-1758`

This is **not** a same-protocol benchmark.

The two baselines differ in:

- teacher data quality control
- dataset split protocol
- gate stack used during evaluation
- downstream experiment design

So the comparison below is intended to answer:

- how the project evolved from `v1` to `v2`
- why `v2` is now the recommended baseline

not to claim a perfectly controlled apples-to-apples win.


## 1. Data Pipeline Difference

### SFT v1

Source data:

- `artifacts/palmistry_llava.generated.clean.qwen3_5_plus.json`

Split:

- train: `3509`
- val: `390`
- total: `3899`

Summary:

- `artifacts/palmistry_llava.generated.clean.qwen3_5_plus.split_summary.json`

Characteristics:

- based on the earlier `clean qwen3.5-plus` teacher generation pipeline
- no strict `teacher -> judge -> filter` stage in the final production form
- standard train/val split


### Strict SFT v2

Source data:

- `artifacts/palmistry_llava.generated.clean.qwen3_5_plus.judged_v2.json`

Split:

- `sft_train = 2344`
- `grpo_train = 976`
- `eval_holdout = 586`
- total: `3906`

Summary:

- `artifacts/palmistry_llava.generated.clean.qwen3_5_plus.judged_v2.stage_split.summary.json`

Characteristics:

- teacher outputs are rebuilt through `teacher -> judge -> filter`
- explicit three-stage split:
  - SFT train
  - GRPO train
  - final holdout eval
- cluster overlap is forced to zero across stages


## 2. Why Strict SFT v2 Is More Credible

`Strict SFT v2` is the stronger experimental baseline mainly because the protocol is cleaner.

Improvements over `v1`:

- teacher labels pass through a judge/filter stage
- SFT, GRPO, and final evaluation are no longer mixed together
- holdout evaluation is separated from GRPO training data
- the final pipeline is evaluated under the standalone gate-classifier setup used by the product path

So even before looking at metrics:

- `v1` is a useful historical baseline
- `v2` is the baseline you can actually build the next experiments on


## 3. Training Comparison

### SFT v1

Training output:

- `output/palmistry_lora_qwen3_vl_8b_clean_v1`

Recorded values:

- epochs: `3`
- total steps: `2634`
- final `train_loss = 0.5444`
- final `eval_loss = 0.4591`

References:

- `output/palmistry_lora_qwen3_vl_8b_clean_v1/trainer_state.json`
- `docs/experiment_iterations.md`


### Strict SFT v2

Training output:

- `output/palmistry_lora_qwen3_vl_8b_strict_v2`

Recorded values:

- epochs: `3`
- total steps: `1758 / 1758`
- final `eval_loss = 0.4968`

References:

- `artifacts/train_lora_strict_v2.log`
- `docs/sft_strict_v2_training.md`

Important note:

- the strict v2 run completed training but hit a disk-full error during the final save phase
- `checkpoint-1758` is still usable and was used for later evaluation


## 4. Why Loss Cannot Be Compared Directly

At first glance:

- `v1 eval_loss = 0.4591`
- `v2 eval_loss = 0.4968`

This does **not** mean `v1` is better.

Reason:

- the validation/evaluation sets are not the same
- `v2` uses a stricter holdout protocol
- the data generation pipeline is different

So this part should only be read as:

- both SFT runs converged normally
- strict v2 did not collapse
- the lower `v1` eval loss is not a meaningful standalone win


## 5. Evaluation Comparison

### SFT v1

Evaluated through the older formal calibration workflow:

- file: `artifacts/evals/palmistry_eval.formal_v2.summary.json`
- val samples: `390`

Key values:

- `gate_match_rate = 0.6538`
- `structured_available_rate = 0.0282`
- `full_report_rate = 0.0282`
- `structured reference_alignment = 0.5709`
- `report reference_alignment = 0.4239`
- `report uncertainty_honesty = 0.0636`


### Strict SFT v2

Evaluated on the strict holdout:

- file: `artifacts/evals/palmistry_eval.strict_v2_holdout.summary.json`
- holdout samples: `586`

Key values:

- `gate_match_rate = 0.7235`
- `structured_available_rate = 0.8959`
- `full_report_rate = 0.1160`
- `structured reference_alignment = 0.5269`
- `report reference_alignment = 0.4238`
- `report uncertainty_honesty = 0.7390`


## 6. What Can and Cannot Be Concluded

### Can be concluded

1. `Strict SFT v2` is the better baseline for future work

Reason:

- stricter data protocol
- judge-filtered teacher labels
- clean separation between SFT, GRPO, and final evaluation

2. `Strict SFT v2` is much better aligned with the current product pipeline

Reason:

- standalone gate classifier
- cautious/retake/continue triage
- strict holdout evaluation path

3. `Strict SFT v2` has far better uncertainty behavior under the current evaluation stack

The biggest practical gain is:

- `report uncertainty_honesty`
  - `v1: 0.0636`
  - `v2: 0.7390`

Even though this number also reflects pipeline changes, it still shows the newer system is far more compatible with conservative reporting.


### Cannot be concluded

1. You cannot say `v2` strictly beats `v1` on a fully controlled same-set benchmark

Because:

- data differs
- split differs
- gate stack differs
- evaluation protocol differs

2. You cannot use `eval_loss` alone to claim `v1` is stronger

Because:

- the validation sets are different


## 7. Practical Recommendation

Use the following roles:

- `SFT v1`
  - historical reference baseline
- `Strict SFT v2`
  - official baseline for all future controlled experiments

That means:

- future GRPO should continue from `Strict SFT v2`
- future holdout comparisons should use the strict holdout protocol
- `v1` should be kept for documentation and project evolution context only


## 8. One-Sentence Summary

`SFT v1` was the original workable baseline, while `Strict SFT v2` is the first version with a clean enough data and evaluation protocol to serve as the project’s real long-term baseline.
