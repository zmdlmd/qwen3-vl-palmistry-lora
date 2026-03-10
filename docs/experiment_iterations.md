# Palmistry 项目实验迭代总结

## 1. 实验目标

本轮实验的核心目标是围绕手掌图像分析任务，逐步建立一条可落地的训练与推理链路：

- 从原始掌纹图像中筛出更适合训练的高质量样本
- 通过教师模型自动生成结构化 palmistry 标注
- 基于 Qwen3-VL 进行 LoRA SFT
- 在推理阶段加入更保守的可见性门控，避免对模糊图像过度解读
- 建立可复现、可观测的评测流程，为后续 GRPO 提供基线

## 2. 数据清洗与训练样本构建

原始训练清单共有 `19587` 条记录，包含大量增强图、相似图和低清晰度样本。为了让训练分布更接近“清晰手掌照片输入”的目标场景，先进行了质量过滤和聚类限额筛选。

### 清洗结果

- 原始样本数：`19587`
- 原图簇数：`2450`
- 保留 `clean` 样本：`3942`
- 归入 `hard_cases` 样本：`15645`
- 平均每簇样本数：`7.99`
- 每簇最多保留：`2`

### 主要过滤原因

- `cluster_quota_exceeded`: `14687`
- `cluster_relative_quality_low`: `12828`
- `below_quality_floor`: `8813`
- `below_sharpness_floor`: `7834`
- `too_dark`: `245`
- `too_bright`: `6`

### 结论

- 数据清洗有效压缩了重复增强样本的权重
- 主训练集开始聚焦于清晰、可辨掌纹的图像
- `hard_cases` 被单独留出，用于后续门控和拒答能力评估

## 3. Teacher 数据生成迭代

教师数据生成阶段，先后对比了多种 DashScope 模型。

### 模型对比结论

- `qwen-plus`
  - 在掌纹图像上基本不可用
  - 多数样本退化为“难以判断”
- `qwen-vl-plus`
  - 能稳定读取图像
  - 但输出较模板化，细节多样性不足
- `qwen3.5-plus`
  - 细节更丰富，多样性更好
  - 但早期存在格式脏输出和低质量图像上的过度解释问题

### 工程改进

- 增加了 JSON 清洗逻辑，处理 `</think>`、重复 JSON 等异常输出
- 增加质量过滤与自动重试
- 把 teacher 生成改成并发请求版本，提高吞吐

### 最终可用结果

- `clean` 集合总样本：`3942`
- 成功生成可用 teacher 样本：`3899`
- `filtered`: `41`
- `error`: `2`

### 结论

- 最终采用 `qwen3.5-plus` 作为 teacher
- 通过“清晰样本优先 + 清洗/过滤 + 重试”的策略，得到一版质量可接受的 SFT 数据

## 4. SFT 基线训练

基于生成后的结构化数据，构建了 SFT 训练集与验证集。

### 数据切分

- 总样本：`3899`
- `train`: `3509`
- `val`: `390`
- 总簇数：`2109`
- `train` 簇：`1895`
- `val` 簇：`214`
- 簇重叠：`0`

本次切分按照原图簇进行，避免同一张手掌的增强变体同时进入 `train` 与 `val`。

### 训练结果

- 基座模型：`Qwen3-VL-8B-Instruct`
- 训练方式：`LoRA SFT`
- 训练轮数：`3 epochs`
- 总 step：`2634`
- 最终 `train_loss`: `0.5444`
- 训练时长：约 `90.6` 分钟

### 结论

- 结构化理解能力已经建立起来
- 模型更擅长“先输出结构化 palmistry JSON”，而不是直接对低质量图像生成自然长报告

## 5. 推理链保守化改造

SFT 抽检后发现一个关键问题：模型在模糊图上容易继续输出完整报告，并出现过度自信解读。因此，推理链被改造成“先判断图像可见性，再决定是否继续分析”的两阶段流程。

### 推理链重构

1. 图像可见性质检
2. 结构化 JSON 生成
3. 自然报告展开

其中可见性判断结果分成三档：

- `继续分析`
- `谨慎分析`
- `建议重拍`

### 目标

- 清晰样本继续分析
- 边缘样本保守输出
- 明显模糊、遮挡或关键掌纹不可见样本直接拦截

## 6. 评测脚本迭代

为了验证门控效果，评测脚本也做了两轮工程改进。

### 改进内容

- 增加实时进度输出
- 增加 `samples.jsonl` 边跑边写
- 增加 `summary.json` 周期性刷新
- 增加 `hard_mode=gate_only`
- 在 hard case 评测中区分：
  - `visibility_cautious`
  - `visibility_retake`

### 价值

- 评测不再是长时间黑盒运行
- 可以实时读取中间结果
- 可以更精确地区分“谨慎分析”和“建议重拍”

## 7. 门控实验三轮对比

### 第一轮：门控过松

正式评测结果：

- `val.gate_match_rate = 0.459`
- `val.full_report_rate = 0.869`
- `hard_cases.low_confidence_rate = 0.275`
- `hard_cases.full_report_rate = 0.725`

结论：

- `val` 上通过率高，但过于乐观
- `hard_cases` 中仍有大量样本生成完整报告
- 模型对困难图像拦截不足

### 第二轮：门控过严

新版正式评测结果：

- `val.gate_match_rate = 0.654`
- `val.low_confidence_rate = 0.972`
- `val.full_report_rate = 0.028`
- `hard_cases.low_confidence_rate = 0.840`
- `hard_cases.full_report_rate = 0.0`

结论：

- `hard_cases` 拦截效果明显提升
- 但 `val` 大部分正常样本也被一起拦掉
- 这版不适合作为最终产品门控

### 第三轮：分层门控校准

校准结果：

- `val.num_samples = 50`
- `val.low_confidence_rate = 0.56`
- `val.expected_low_confidence_rate = 0.74`
- `val.gate_match_rate = 0.58`
- `val.visibility_cautious_rate = 0.98`
- `val.visibility_retake_rate = 0.04`
- `val.structured_available_rate = 0.96`
- `val.full_report_rate = 0.44`

- `hard_cases.num_samples = 200`
- `hard_cases.low_confidence_rate = 0.80`
- `hard_cases.visibility_cautious_rate = 0.79`
- `hard_cases.visibility_retake_rate = 0.045`
- `hard_cases.full_report_rate = 0.0`

结论：

- 与前两轮相比，这一版已经更平衡
- `val` 不再被大面积误杀，结构化输出基本恢复
- `hard_cases` 基本被保守处理，不再继续完整报告

## 8. 当前阶段结论

当前版本已经形成一条较完整的可用链路：

1. 数据清洗，保留高质量清晰样本
2. `qwen3.5-plus` teacher 自动生成结构化标注
3. Qwen3-VL LoRA SFT 学习掌纹结构化理解
4. 推理阶段加入可见性门控与谨慎分析逻辑
5. 用实时评测脚本验证 `val` 与 `hard_cases` 表现

从结果看，当前版本已经比最初的直接报告生成方案更稳：

- 结构化输出稳定
- 对困难样本更保守
- 推理与评测链路都更可观测

## 9. 当前仍待优化的问题

虽然门控已经明显改善，但还有两个指标说明后续仍需继续优化：

- `gate_match_rate` 仍然不高，说明当前门控与 teacher 的理想保守判断还没有完全对齐
- `report uncertainty_honesty` 仍偏低，说明模型在自然报告中对“不确定性”的表达还不够充分

## 10. 下一步建议

当前最合理的下一步是进入 `report-stage GRPO`，重点优化以下方向：

- 当图像不清晰时，更自然地表达“不确定性”
- 对边缘样本优先输出保守结论，而不是具体断言
- 让结构化结果与自然报告之间保持更强一致性
- 继续把 `hard_cases` 作为拒答与谨慎回答的对抗集

这意味着后续训练重点不再是“让模型看见更多内容”，而是“让模型在看不清时更诚实、更克制”。
