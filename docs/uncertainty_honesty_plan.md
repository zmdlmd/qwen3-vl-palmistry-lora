# Uncertainty Honesty 优化计划

## 1. 目标

当前 `report GRPO v2` 已经在两件事上取得了进展：

- `hard_cases` 上的保守门控更强
- `val` 上的完整报告率和报告对齐略有提升

但核心短板仍然存在：

- `report_metrics.uncertainty_honesty` 没有提升，反而略有下降

因此，下一阶段的目标不是继续泛化地“优化报告质量”，而是定向优化：

1. 当 teacher 明确给出“不确定 / 难以判断 / 可见信息有限”时，模型能在自然报告中明确表达这一点
2. 当某条线不确定时，只对该线保守，不把整篇报告都写成模糊话术
3. 当图像质量不足时，模型更自然地输出“谨慎分析”或“建议重拍”，而不是继续写满细节

## 2. 当前基线

参考文档：

- `docs/report_grpo_v2_analysis.md`
- `artifacts/evals/palmistry_eval.calib.summary.json`
- `artifacts/evals/palmistry_eval.grpo_v2_calib.summary.json`

当前可作为对照的两版模型：

- SFT baseline：`output/palmistry_lora_qwen3_vl_8b_clean_v1`
- GRPO v2：`output/palmistry_grpo_report_qwen3_vl_8b_clean_v2`

当前关键指标：

### `val`

- SFT baseline
  - `full_report_rate = 0.44`
  - `report.reference_alignment = 0.3953`
  - `report.uncertainty_honesty = 0.0318`

- GRPO v2
  - `full_report_rate = 0.66`
  - `report.reference_alignment = 0.4016`
  - `report.uncertainty_honesty = 0.0212`

### `hard_cases`

- SFT baseline
  - `low_confidence_rate = 0.80`
  - `visibility_cautious_rate = 0.79`

- GRPO v2
  - `low_confidence_rate = 0.925`
  - `visibility_cautious_rate = 0.92`

结论：

- `GRPO v2` 已经把模型推向了“更保守的门控”
- 但没有成功把模型推向“更诚实地表达不确定性”

## 3. 问题拆解

当前 `uncertainty_honesty` 表现不佳，主要可能来自三类问题：

### 3.1 reward 强度不够

现有 `uncertainty_honesty_reward` 只是整体文级别奖励，没有足够强地惩罚：

- teacher 写“难以判断”，报告却给出明确结论
- teacher 只对某一条线不确定，报告却把该线写得很具体

### 3.2 reward 粒度不够

当前判断更偏“整篇报告是否出现了谨慎词”，但任务本质上是逐线条的：

- 生命线不确定
- 事业线不确定
- 婚姻线不确定

这些应该逐线核对，而不是只看全文有没有“可见信息有限”。

### 3.3 数据分布不够聚焦

如果训练数据中“teacher 明确不确定”的样本比例不够，或者这类样本在 GRPO 中权重不够，那么策略优化会优先朝：

- 格式更像报告
- 结构更完整
- 整体更流畅

而不是朝：

- 该保守时更诚实

## 4. 实验原则

为了让下一轮结论可信，这一轮必须遵守三个原则：

1. 固定评测子集
   - `val 50` 和 `hard_cases 200` 必须固化成同一份清单
   - 后续所有对比都在同一批样本上跑

2. 单变量推进
   - 先只改 reward，不同时改 prompt、门控策略和训练步数
   - 避免无法判断改进来源

3. 分阶段验证
   - 先做 reward smoke / calib
   - 再决定是否值得开正式 GRPO

## 5. 数据与评测准备

### 5.1 固定对照评测清单

需要先生成两份固定清单：

- `artifacts/evals/fixed_val50.json`
- `artifacts/evals/fixed_hard200.jsonl`

要求：

- 后续所有 SFT / GRPO 对比都使用这两份固定子集
- 禁止每次重新随机抽样

### 5.2 构建不确定性挑战集

从现有 `val` 或 teacher 数据中额外筛出一份 challenge set，优先保留：

- 主线中至少 `2` 条被 teacher 标为不确定
- 或 `visibility_assessment` 为 `谨慎分析`
- 或 `teacher` 中出现高频不确定词：
  - `难以判断`
  - `模糊`
  - `可见信息有限`
  - `无法准确判断`

建议规模：

- `uncertainty_challenge_200`

用途：

- 专门评估“报告是否诚实表达不确定性”

## 6. reward 改造方案

下一轮先在 `reward_funcs_report.py` 中做三类定向修改。

### 6.1 强化逐线不确定性奖励

新增或替换为更细粒度的 `line_uncertainty_honesty_reward`：

- 按 `生命线 / 智慧线 / 感情线 / 事业线 / 婚姻线` 逐线比对
- 如果 teacher 对某条线不确定：
  - 报告该线必须出现谨慎表达，才给分
- 如果 teacher 对某条线明确：
  - 报告该线不应无端写成“完全看不清”，否则扣分

目标：

- 避免“整篇报告一句模糊免责声明”骗到 reward

### 6.2 增加矛盾惩罚

新增 `uncertainty_contradiction_penalty_reward`：

- 如果报告先说“图像不够清晰 / 难以判断”
- 但后面又给出：
  - “深长”
  - “明显断裂”
  - “清晰分叉”
  - “贯穿掌心”
  这类强判断，则扣分

目标：

- 惩罚“口头保守，内容冒进”

### 6.3 提高不确定样本权重

对 teacher 不确定样本做更高 reward 权重或更高采样权重：

- 主线不确定数 >= 2 的样本
- `visibility_assessment` 为 `谨慎分析` 的样本

目标：

- 让 GRPO 更关注当前最薄弱的行为模式

## 7. 实验设计

### 阶段 A：reward 单独校准

目标：

- 不开正式长训练，先验证 reward 是否朝正确方向打分

步骤：

1. 对现有 `SFT baseline` 抽样生成报告
2. 对同一批报告离线计算旧 reward 和新 reward
3. 检查新 reward 是否满足：
   - teacher 不确定且报告冒进的样本，分数显著下降
   - teacher 不确定且报告诚实的样本，分数显著上升

通过标准：

- 至少 `80%` 的人工抽检样本符合预期排序

### 阶段 B：1-step / 10-step smoke

目标：

- 确认新 reward 可以稳定进入训练，不出现：
  - reward 恒为 0
  - reward 爆炸
  - 明显训练不稳定

步骤：

1. `1-step smoke`
2. `10-step mini run`
3. 检查日志中：
   - 新 reward 是否有波动
   - `uncertainty` 相关 reward 是否确实参与区分

### 阶段 C：calib 对照实验

目标：

- 用固定 `val50 + hard200 + challenge200` 评估是否值得开正式训练

候选 run：

- `GRPO v3a`: 只强化逐线不确定性 reward
- `GRPO v3b`: 逐线不确定性 + 矛盾惩罚
- `GRPO v3c`: 再叠加不确定样本加权

决策标准：

- `uncertainty_honesty` 提升
- `hard_cases.low_confidence_rate` 不明显回退
- `val.full_report_rate` 不大幅崩掉

### 阶段 D：正式训练

只有 `v3a / v3b / v3c` 中有一版在 `calib` 明显优于 `GRPO v2` 时，才开正式训练。

建议正式训练设置：

- `max_steps = 200`
- 保持当前 batch / generations 不变
- 先不要同时改吞吐配置

原因：

- 先保证实验结论可比
- 速度优化放在 reward 方向稳定之后

## 8. 成功标准

下一轮如果要认为“有效”，建议至少满足以下条件：

### 必达指标

- `val.report.uncertainty_honesty` 相比 `GRPO v2` 提升至少 `+0.03`
- `hard_cases.low_confidence_rate >= 0.90`
- `hard_cases.full_report_rate = 0.0`

### 保底指标

- `val.full_report_rate >= 0.55`
- `val.report.reference_alignment >= 0.39`
- `structured_available_rate >= 0.90`

### 失败判定

如果出现以下任一情况，则判定该方案不值得继续：

- `val.full_report_rate` 大幅降回 `< 0.45`
- `hard_cases.low_confidence_rate` 明显回落到 `< 0.85`
- `uncertainty_honesty` 无提升或继续下降

## 9. 执行顺序

建议按以下顺序实施：

1. 固化评测子集
2. 构建 `uncertainty_challenge_200`
3. 改造 reward
4. 做离线 reward 排序检查
5. 跑 `1-step / 10-step smoke`
6. 跑固定 `calib`
7. 选择最优方案进入正式训练

## 10. 本轮不做的事

为了避免实验变量过多，以下内容暂不和本轮 reward 实验同时推进：

- 改 teacher prompt
- 改 report prompt
- 重做门控规则
- 调大 batch / num_generations / 吞吐配置
- 引入新的 teacher-as-judge 评分链

这些都可能有价值，但不适合和本轮 `uncertainty_honesty` 定向优化同时进行。
