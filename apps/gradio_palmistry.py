from __future__ import annotations

import argparse
import html
import json

import gradio as gr

from src.palmistry.config import default_device
from src.palmistry import PalmistryPipeline, STYLE_OPTIONS


CSS = """
body {
  background: radial-gradient(circle at top, #f97316 0, #0f172a 58%, #020617 100%);
}

.gradio-container {
  max-width: 1180px !important;
  margin: 28px auto !important;
}

#app-shell {
  background: #ffffff;
  border-radius: 24px;
  box-shadow: 0 24px 64px rgba(15, 23, 42, 0.24);
  padding: 24px 28px 20px 28px;
}

#title-bar {
  display: flex;
  gap: 14px;
  align-items: center;
  border-bottom: 1px solid #e2e8f0;
  padding-bottom: 10px;
  margin-bottom: 14px;
}

#title-icon {
  font-size: 34px;
}

#title-text {
  font-size: 27px;
  font-weight: 800;
  letter-spacing: 0.02em;
}

#subtitle {
  font-size: 13px;
  color: #64748b;
}

button.primary {
  background: linear-gradient(135deg, #f97316, #fb923c) !important;
  color: #ffffff !important;
  border: none !important;
  border-radius: 999px !important;
  box-shadow: 0 10px 24px rgba(249, 115, 22, 0.35);
}

#report-card {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 18px;
  padding: 10px 12px 6px 12px;
}

#status-shell {
  border-radius: 18px;
  padding: 14px 16px;
  margin-bottom: 12px;
  border: 1px solid #e2e8f0;
}

.status-ready {
  background: linear-gradient(135deg, #ecfccb, #f0fdf4);
  border-color: #84cc16 !important;
}

.status-retake {
  background: linear-gradient(135deg, #fff7ed, #fffbeb);
  border-color: #f97316 !important;
}

.status-wait {
  background: linear-gradient(135deg, #eff6ff, #f8fafc);
  border-color: #cbd5e1 !important;
}

.status-badge {
  display: inline-flex;
  align-items: center;
  border-radius: 999px;
  padding: 4px 10px;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.03em;
}

.badge-ready {
  background: #65a30d;
  color: #ffffff;
}

.badge-retake {
  background: #ea580c;
  color: #ffffff;
}

.badge-wait {
  background: #475569;
  color: #ffffff;
}

.status-title {
  font-size: 18px;
  font-weight: 800;
  margin: 10px 0 6px 0;
  color: #0f172a;
}

.status-copy {
  color: #334155;
  line-height: 1.6;
  font-size: 14px;
}

.status-meta {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px;
  margin-top: 12px;
}

.status-meta-item {
  background: rgba(255, 255, 255, 0.7);
  border: 1px solid #dbe3ef;
  border-radius: 12px;
  padding: 8px 10px;
}

.status-meta-label {
  font-size: 11px;
  font-weight: 700;
  color: #64748b;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  margin-bottom: 3px;
}

.status-meta-value {
  font-size: 13px;
  font-weight: 600;
  color: #0f172a;
  word-break: break-word;
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Palmistry Gradio demo")
    parser.add_argument("--base-model", required=True, help="Base Qwen3-VL model path or HF id")
    parser.add_argument("--lora-path", default=None, help="Optional LoRA adapter path")
    parser.add_argument("--device", default=None, help="Runtime device, e.g. cuda or cpu")
    parser.add_argument("--device-map", default="auto", help="Transformers device_map")
    parser.add_argument("--torch-dtype", default="auto", help="auto | bf16 | fp16 | fp32")
    parser.add_argument("--gate-classifier-path", default=None, help="Optional standalone gate classifier checkpoint path")
    parser.add_argument("--gate-classifier-device", default=None, help="Standalone gate classifier runtime device")
    parser.add_argument("--gate-classifier-min-confidence", type=float, default=0.55)
    parser.add_argument("--gate-classifier-continue-min-confidence", type=float, default=0.65)
    parser.add_argument("--gate-classifier-retake-min-confidence", type=float, default=0.65)
    parser.add_argument("--gate-classifier-min-margin", type=float, default=0.10)
    parser.add_argument("--server-name", default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=7860)
    return parser.parse_args()


def build_app(pipeline: PalmistryPipeline) -> gr.Blocks:
    def format_probability(value):
        if value is None:
            return "N/A"
        return f"{float(value):.1%}"

    def compute_margin(probabilities):
        if not isinstance(probabilities, dict) or not probabilities:
            return None
        ranked = sorted((float(v) for v in probabilities.values()), reverse=True)
        if len(ranked) < 2:
            return ranked[0]
        return ranked[0] - ranked[1]

    def gate_source_label(visibility_assessment):
        source = str((visibility_assessment or {}).get("source", "")).strip()
        if source == "standalone_gate_classifier":
            return "独立 Classifier"
        if source == "generative_gate":
            return "生成式 Gate"
        if source == "error_fallback":
            return "错误回退"
        return "未标注"

    def build_status_meta_html(visibility_assessment):
        if visibility_assessment is None:
            return ""

        probabilities = visibility_assessment.get("classifier_probabilities")
        confidence = visibility_assessment.get("classifier_confidence")
        margin = compute_margin(probabilities)
        raw_decision = visibility_assessment.get("classifier_raw_decision")
        threshold_applied = visibility_assessment.get("classifier_threshold_applied")
        decision_label = visibility_assessment.get("decision_label") or visibility_assessment.get("建议") or ""

        meta_items = [
            ("Gate Source", gate_source_label(visibility_assessment)),
            ("当前决策", str(decision_label or visibility_assessment.get("decision", "N/A"))),
        ]

        if confidence is not None:
            meta_items.append(("Confidence", format_probability(confidence)))
        if margin is not None:
            meta_items.append(("Margin", format_probability(margin)))
        if raw_decision:
            meta_items.append(("Raw Decision", str(raw_decision)))
        if threshold_applied is not None:
            meta_items.append(("Threshold", "已触发" if threshold_applied else "未触发"))

        blocks = []
        for label, value in meta_items:
            blocks.append(
                f"""
                <div class="status-meta-item">
                  <div class="status-meta-label">{html.escape(label)}</div>
                  <div class="status-meta-value">{html.escape(str(value))}</div>
                </div>
                """.strip()
            )
        return f'<div class="status-meta">{"".join(blocks)}</div>'

    def format_status_html(gate_decision, caution_message, visibility_assessment):
        if visibility_assessment is None:
            return """
            <div id="status-shell" class="status-wait">
              <div class="status-badge badge-wait">等待分析</div>
              <div class="status-title">上传手掌照片后开始质检</div>
              <div class="status-copy">系统会先判断掌纹可见性，再决定继续分析还是建议重拍。</div>
            </div>
            """.strip()

        if gate_decision == "retake":
            badge = "建议重拍"
            status_class = "status-retake"
            badge_class = "badge-retake"
            title = "当前照片不适合继续做完整手相解读"
        elif gate_decision == "cautious":
            badge = "谨慎分析"
            status_class = "status-ready"
            badge_class = "badge-wait"
            title = "当前照片只适合保守掌纹观察"
        else:
            badge = "可继续分析"
            status_class = "status-ready"
            badge_class = "badge-ready"
            title = "当前照片已通过保守质检"

        extra = caution_message or "图像质量和主线可见性已达到继续分析的最低要求。"
        meta_html = build_status_meta_html(visibility_assessment)
        return f"""
        <div id="status-shell" class="{status_class}">
          <div class="status-badge {badge_class}">{badge}</div>
          <div class="status-title">{title}</div>
          <div class="status-copy">{extra}</div>
          {meta_html}
        </div>
        """.strip()

    def format_visibility_json(visibility_assessment):
        if visibility_assessment is None:
            return ""
        return json.dumps({"visibility_assessment": visibility_assessment}, ensure_ascii=False, indent=2)

    def generate_report(image, style):
        if image is None:
            message = "请先上传清晰的手掌照片。"
            return (
                format_status_html("continue", "", None),
                message,
                "",
                "",
                "",
                "",
                [],
                [],
            )

        result = pipeline.analyze_detailed(image, style=style)
        report_state = "" if result.gate_decision != "continue" else result.report
        return (
            format_status_html(result.gate_decision, result.caution_message, result.visibility_assessment),
            result.report,
            result.caution_message,
            result.structured_json,
            format_visibility_json(result.visibility_assessment),
            report_state,
            [],
            [],
        )

    def ask_followup(user_question, history, report):
        history = history or []
        if not user_question or not user_question.strip():
            return history, history, ""

        if not report:
            answer = "请先生成一份手相报告，再继续追问。"
        else:
            answer = pipeline.answer_followup(report, user_question.strip())

        updated_history = history + [[user_question, answer]]
        return updated_history, updated_history, ""

    def clear_all():
        return None, "balanced", format_status_html("continue", "", None), "", "", "", "", "", [], [], ""

    with gr.Blocks(css=CSS, title="Palmistry LoRA Demo") as demo:
        report_state = gr.State("")
        history_state = gr.State([])

        with gr.Column(elem_id="app-shell"):
            gr.HTML(
                """
                <div id="title-bar">
                  <div id="title-icon">🖐</div>
                    <div>
                      <div id="title-text">Qwen3-VL Palmistry LoRA Demo</div>
                    <div id="subtitle">先做掌纹可见性质检，再决定继续分析还是建议重拍；通过后再生成中文报告并支持追问。</div>
                  </div>
                </div>
                """
            )

            with gr.Row():
                with gr.Column(scale=1, min_width=360):
                    image_input = gr.Image(
                        type="pil",
                        label="手掌图像",
                        height=420,
                    )
                    style_input = gr.Radio(
                        choices=list(STYLE_OPTIONS.keys()),
                        value="balanced",
                        label="报告风格",
                        info="balanced: 均衡 | soft: 疗愈 | professional: 专业",
                    )
                    with gr.Row():
                        generate_btn = gr.Button("生成报告", elem_classes="primary")
                        clear_btn = gr.Button("清空")

                    gr.Markdown(
                        """
拍摄建议：
- 掌心朝上，手掌尽量完整入镜
- 纹路要清晰，避免强反光和过暗
- 本结果仅供参考，不替代医学或职业建议
                        """.strip()
                    )

                with gr.Column(scale=1, min_width=360):
                    status_box = gr.HTML(value=format_status_html(False, "", None))
                    with gr.Column(elem_id="report-card"):
                        with gr.Tabs():
                            with gr.Tab("分析报告"):
                                report_box = gr.Textbox(
                                    label="手相分析报告",
                                    lines=18,
                                    show_copy_button=True,
                                )
                                caution_box = gr.Textbox(
                                    label="保守模式提示",
                                    lines=4,
                                    show_copy_button=True,
                                )
                            with gr.Tab("结构化 JSON"):
                                structured_box = gr.Textbox(
                                    label="结构化掌纹分析 JSON",
                                    lines=18,
                                    show_copy_button=True,
                                )
                            with gr.Tab("图像质检"):
                                visibility_box = gr.Textbox(
                                    label="可见性质检 JSON",
                                    lines=12,
                                    show_copy_button=True,
                                )

            gr.Markdown("### 报告追问")
            chatbot = gr.Chatbot(label="继续追问", height=320)
            with gr.Row():
                question_box = gr.Textbox(
                    label="继续提问",
                    placeholder="例如：感情线部分能不能再展开一点？",
                    scale=5,
                )
                send_btn = gr.Button("发送", elem_classes="primary", scale=1)

        generate_btn.click(
            generate_report,
            inputs=[image_input, style_input],
            outputs=[status_box, report_box, caution_box, structured_box, visibility_box, report_state, chatbot, history_state],
        )
        send_btn.click(
            ask_followup,
            inputs=[question_box, history_state, report_state],
            outputs=[chatbot, history_state, question_box],
        )
        question_box.submit(
            ask_followup,
            inputs=[question_box, history_state, report_state],
            outputs=[chatbot, history_state, question_box],
        )
        clear_btn.click(
            clear_all,
            outputs=[image_input, style_input, status_box, report_box, caution_box, structured_box, visibility_box, report_state, chatbot, history_state, question_box],
        )

    return demo


def main() -> None:
    args = parse_args()
    pipeline = PalmistryPipeline(
        base_model_path=args.base_model,
        lora_path=args.lora_path,
        device=args.device or default_device(),
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        gate_classifier_path=args.gate_classifier_path,
        gate_classifier_device=args.gate_classifier_device,
        gate_classifier_min_confidence=args.gate_classifier_min_confidence,
        gate_classifier_continue_min_confidence=args.gate_classifier_continue_min_confidence,
        gate_classifier_retake_min_confidence=args.gate_classifier_retake_min_confidence,
        gate_classifier_min_margin=args.gate_classifier_min_margin,
    )
    demo = build_app(pipeline)
    demo.launch(server_name=args.server_name, server_port=args.server_port, share=False)


if __name__ == "__main__":
    main()
