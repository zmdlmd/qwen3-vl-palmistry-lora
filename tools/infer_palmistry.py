from __future__ import annotations

import argparse

from src.palmistry.config import default_device
from src.palmistry import PalmistryPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run palmistry inference on one hand image.")
    parser.add_argument("--base-model", required=True, help="Base Qwen3-VL model path or HF id")
    parser.add_argument("--lora-path", default=None, help="Optional LoRA adapter path")
    parser.add_argument("--image", required=True, help="Path to the hand image")
    parser.add_argument("--style", default="balanced", choices=["balanced", "soft", "professional"])
    parser.add_argument("--device", default=None, help="Runtime device")
    parser.add_argument("--device-map", default="auto", help="Transformers device_map")
    parser.add_argument("--torch-dtype", default="auto", help="auto | bf16 | fp16 | fp32")
    parser.add_argument("--gate-classifier-path", default=None, help="Optional standalone gate classifier checkpoint path")
    parser.add_argument("--gate-classifier-device", default=None, help="Standalone gate classifier runtime device")
    parser.add_argument("--gate-classifier-min-confidence", type=float, default=0.55)
    parser.add_argument("--gate-classifier-continue-min-confidence", type=float, default=0.65)
    parser.add_argument("--gate-classifier-retake-min-confidence", type=float, default=0.65)
    parser.add_argument("--gate-classifier-min-margin", type=float, default=0.10)
    parser.add_argument("--max-new-tokens", type=int, default=1200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--show-structured", action="store_true", help="Print the intermediate structured JSON analysis")
    parser.add_argument("--question", default=None, help="Optional follow-up question after the main report")
    return parser.parse_args()


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

    result = pipeline.analyze_detailed(
        args.image,
        style=args.style,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(f"[gate_policy] {result.gate_decision}")
    if result.visibility_assessment:
        reason = result.visibility_assessment.get("依据")
        if reason:
            print(f"[gate_reason] {reason}")
        print()

    if result.caution_message:
        print(result.caution_message)
        print()

    print(result.report)

    if args.show_structured and result.structured_json:
        print("\n" + "=" * 80)
        print(result.structured_json)

    if args.question:
        answer = pipeline.answer_followup(
            result.report,
            args.question,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print("\n" + "=" * 80)
        print(answer)


if __name__ == "__main__":
    main()
