from __future__ import annotations

import argparse

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
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Palmistry Gradio demo")
    parser.add_argument("--base-model", required=True, help="Base Qwen3-VL model path or HF id")
    parser.add_argument("--lora-path", default=None, help="Optional LoRA adapter path")
    parser.add_argument("--device", default=None, help="Runtime device, e.g. cuda or cpu")
    parser.add_argument("--device-map", default="auto", help="Transformers device_map")
    parser.add_argument("--torch-dtype", default="auto", help="auto | bf16 | fp16 | fp32")
    parser.add_argument("--server-name", default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=7860)
    return parser.parse_args()


def build_app(pipeline: PalmistryPipeline) -> gr.Blocks:
    def generate_report(image, style):
        if image is None:
            message = "请先上传清晰的手掌照片。"
            return message, "", "", "", [], []

        result = pipeline.analyze_detailed(image, style=style)
        return result.report, result.caution_message, result.structured_json, result.report, [], []

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
        return None, "balanced", "", "", "", "", [], [], ""

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
                    <div id="subtitle">上传手掌图像，生成中文手相报告，并基于报告继续追问。</div>
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
                    with gr.Column(elem_id="report-card"):
                        caution_box = gr.Textbox(
                            label="保守模式提示",
                            lines=4,
                            show_copy_button=True,
                        )
                        report_box = gr.Textbox(
                            label="手相分析报告",
                            lines=18,
                            show_copy_button=True,
                        )
                        structured_box = gr.Textbox(
                            label="结构化掌纹分析 JSON",
                            lines=14,
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
            outputs=[report_box, caution_box, structured_box, report_state, chatbot, history_state],
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
            outputs=[image_input, style_input, report_box, caution_box, structured_box, report_state, chatbot, history_state, question_box],
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
    )
    demo = build_app(pipeline)
    demo.launch(server_name=args.server_name, server_port=args.server_port, share=False)


if __name__ == "__main__":
    main()
