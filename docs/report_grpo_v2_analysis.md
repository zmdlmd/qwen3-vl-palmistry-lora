# Report GRPO v2 分析记录

## 1. 目标

本轮实验的目标不是继续提升结构化 JSON 拟合能力，而是验证 `report-stage GRPO` 是否能够把模型往以下方向推动：

- 在正常样本上更自然地输出完整报告
- 在低质量样本上更保守地触发低置信度门控
- 降低报告阶段的幻觉扩写
- 让自然报告与 teacher 结构化 JSON 保持更强一致性

本次分析对应的模型与结果文件如下：

- SFT 基线 LoRA：`output/palmistry_lora_qwen3_vl_8b_clean_v1`
- GRPO v2 LoRA：`output/palmistry_grpo_report_qwen3_vl_8b_clean_v2`
- SFT 校准结果：`artifacts/evals/palmistry_eval.calib.summary.json`
- GRPO v2 校准结果：`artifacts/evals/palmistry_eval.grpo_v2_calib.summary.json`
- GRPO v2 训练日志：`artifacts/grpo_logs/report_grpo_train_v2.log`

## 2. 训练设置

### 基础配置

- 基座模型：`Qwen3-VL-8B-Instruct`
- 初始权重：`palmistry_lora_qwen3_vl_8b_clean_v1`
- 训练数据：`artifacts/palmistry_llava.report_grpo.train.json`
- 训练方式：单卡 `RTX 5090`、`4-bit`、LoRA、DeepSpeed
- `max_steps = 200`
- `num_generations = 2`

### v2 新增 reward

相比早期 report GRPO 版本，v2 重点加入了三类约束：

- `gate_decision_reward`
  - 奖励“继续分析 / 谨慎分析 / 建议重拍”决策与参考信息一致
- `hallucination_penalty_reward`
  - 惩罚 teacher 未提供依据但报告中强行扩写的内容
- `line_level_consistency_reward`
  - 按掌纹逐线检查报告与 teacher JSON 的一致性

这三项 reward 的目标，是把优化重点从“报告像不像一篇文章”转成“报告该不该说、该说多少、说得是否克制”。

## 3. 训练结果

从 `report_grpo_train_v2.log` 可以确认，这轮训练已经完整结束：

- `200 / 200 step`
- `train_runtime = 10337.24s`，约 `2小时52分`
- `train_loss = 0.0347`

最后一步日志中的 reward 读数为：

- `reward ≈ 2.9773`
- `reference_alignment_reward ≈ 0.3773`
- `line_level_consistency_reward = 0.0`
- `hallucination_penalty_reward = 0.0`
- `gate_decision_reward = 0.0`
- `uncertainty_honesty_reward = 0.0`
- `safety_language_reward = 0.6`

需要注意的是，GRPO 的单 step reward 波动很大，最后一步不能代表整体趋势。训练过程中也出现过以下更积极的 batch：

- `gate_decision_reward = 1.0`
- `hallucination_penalty_reward ≈ 0.9`
- `line_level_consistency_reward ≈ 0.25`

这说明新 reward 不是恒定加分，而是在真实区分不同样本与不同生成结果。

## 4. 校准评测说明

`calib` 的作用是用一批固定的小样本快速判断方向是否正确，而不是代替正式全量评测。

本次 `GRPO v2 calib` 的设置为：

- `val = 50`
- `hard_cases = 200`
- `hard_mode = gate_only`

其中：

- `val` 评测会跑完整推理链，适合观察“正常图像是否被过度拦截”
- `hard_cases` 只测试门控，不生成完整报告，适合快速观察“困难图像是否被保守处理”

## 5. 对比结果

### 5.1 `val` 子集对比

`val` 的 50 条样本在两次校准中完全相同，因此这一部分可以视为严格可比的 A/B 对照。

| 指标 | SFT baseline | GRPO v2 | 变化 |
| --- | ---: | ---: | ---: |
| `low_confidence_rate` | `0.56` | `0.34` | `-0.22` |
| `gate_match_rate` | `0.58` | `0.56` | `-0.02` |
| `structured_available_rate` | `0.96` | `0.94` | `-0.02` |
| `full_report_rate` | `0.44` | `0.66` | `+0.22` |
| `structured.reference_alignment` | `0.5299` | `0.5296` | `≈ 0` |
| `report.reference_alignment` | `0.3953` | `0.4016` | `+0.0063` |
| `report.uncertainty_honesty` | `0.0318` | `0.0212` | `-0.0106` |

#### 解释

- `GRPO v2` 在正常样本上更愿意给出完整报告，`full_report_rate` 明显上升
- 报告和 teacher JSON 的对齐略有提升
- 结构化输出能力基本没有明显退化
- 但 `low_confidence_rate` 明显下降，说明它在正常样本上更激进
- `uncertainty_honesty` 没有改善，反而略降，说明“不确定时显式说明信息有限”仍然是薄弱点

### 5.2 `hard_cases` 对比

`hard_cases` 两次 `calib` 使用的 200 条样本并不是同一子集。经核对：

- `val` 样本重合：`50 / 50`
- `hard_cases` 样本重合：`2 / 200`

因此 `hard_cases` 结果只能用于方向判断，不能视为严格的逐样本 A/B 对照。

尽管如此，整体趋势仍然清晰：

| 指标 | SFT baseline | GRPO v2 | 变化 |
| --- | ---: | ---: | ---: |
| `low_confidence_rate` | `0.80` | `0.925` | `+0.125` |
| `visibility_cautious_rate` | `0.79` | `0.92` | `+0.13` |
| `visibility_retake_rate` | `0.045` | `0.095` | `+0.05` |
| `full_report_rate` | `0.0` | `0.0` | `0` |

#### 解释

- `GRPO v2` 对困难样本更保守
- `low_confidence_rate`、`visibility_cautious_rate`、`visibility_retake_rate` 都明显上升
- 这说明新增 reward 确实在推动模型对低质量图像更谨慎地门控

不过，由于样本子集不同，下一轮如果要做严格结论，应该使用固定 `hard_cases` 清单重新同时评估 `SFT baseline` 和 `GRPO v2`。

## 6. 当前结论

综合训练与校准结果，可以得到以下判断：

1. `GRPO v2` 已经学到更强的“困难样本保守化”倾向。
2. 在正常样本上，模型更愿意输出完整报告，报告对齐指标也略有提升。
3. 结构化能力没有明显塌陷，说明 report-stage GRPO 没有破坏已有的 SFT 基线。
4. 当前最明显的短板仍然是 `uncertainty_honesty`，也就是“该不确定时是否会明确承认不确定”。

换句话说，`GRPO v2` 当前更像是在优化：

- 是否继续分析
- 是否减少明显幻觉
- 是否保持报告结构

但还没有充分优化：

- 当信息不足时，如何把“不确定”说得自然且明确

## 7. 下一步建议

下一阶段不建议简单继续增加 GRPO 步数，更合理的是针对 `uncertainty_honesty` 做定向优化：

- 强化对“难以判断 / 可见信息有限 / 建议重拍”表达的奖励
- 增加“teacher 明确不确定但报告仍写得很满”的惩罚
- 用固定 `hard_cases` 子集重跑 `SFT baseline` 与 `GRPO v2` 的严格 A/B 对比

如果要继续做 report GRPO，建议把下一轮实验目标明确设成：

- 保持当前 `hard_cases` 保守性
- 尽量不牺牲 `val` 上的完整报告率
- 重点拉高 `report.uncertainty_honesty`
