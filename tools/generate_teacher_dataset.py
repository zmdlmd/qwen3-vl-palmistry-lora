from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.palmistry.prompts import (
    DEFAULT_STUDENT_STRUCTURED_PROMPT,
    build_teacher_judge_prompt,
    build_teacher_structured_prompt,
)
from src.palmistry.teacher import TeacherGenerationConfig, default_num_workers, generate_sft_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate palmistry SFT data by calling an OpenAI-compatible teacher model API.",
    )
    parser.add_argument("--api-base", default="https://api.openai.com/v1", help="OpenAI-compatible API base URL")
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenAI-compatible API key. If omitted, OPENAI_API_KEY or DASHSCOPE_API_KEY will be used.",
    )
    parser.add_argument("--model", required=True, help="Teacher model name")
    parser.add_argument("--manifest", default=None, help="Manifest path in .json or .jsonl format")
    parser.add_argument("--image-dir", default=None, help="Image directory used to resolve relative image paths")
    parser.add_argument("--output-json", required=True, help="Output LLaVA-style JSON dataset path")
    parser.add_argument("--output-jsonl", default=None, help="Optional raw generation log JSONL path")
    parser.add_argument("--teacher-prompt-file", default=None, help="Optional file containing the teacher prompt")
    parser.add_argument("--student-prompt", default=DEFAULT_STUDENT_STRUCTURED_PROMPT)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=2200)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=default_num_workers(), help="Number of concurrent teacher API requests")
    parser.add_argument("--json-mode", action="store_true", help="Enable response_format=json_object if the provider supports it")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true", help="Do not reuse successful generations found in output-jsonl")
    parser.add_argument("--no-quality-filter", action="store_true", help="Disable heuristic quality filtering for low-confidence teacher outputs")
    parser.add_argument("--max-uncertain-main-lines", type=int, default=1, help="Maximum number of main lines that may remain uncertain before filtering a sample")
    parser.add_argument("--judge-model", default=None, help="Optional OpenAI-compatible judge model used to review teacher outputs")
    parser.add_argument("--judge-prompt-file", default=None, help="Optional file containing the teacher judge prompt")
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-max-tokens", type=int, default=900)
    parser.add_argument("--judge-min-average-score", type=float, default=3.5)
    parser.add_argument("--judge-min-visual-grounding", type=float, default=3.0)
    parser.add_argument("--judge-min-uncertainty-honesty", type=float, default=3.0)
    parser.add_argument("--reject-cautious", action="store_true", help="Treat judge decision accept_cautious as filtered")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise SystemExit("API key is required. Pass --api-key or set OPENAI_API_KEY / DASHSCOPE_API_KEY.")

    if not args.manifest and not args.image_dir:
        raise SystemExit("Either --manifest or --image-dir must be provided.")

    if args.teacher_prompt_file:
        teacher_prompt = Path(args.teacher_prompt_file).read_text(encoding="utf-8").strip()
    else:
        teacher_prompt = build_teacher_structured_prompt()

    if args.judge_prompt_file:
        judge_prompt = Path(args.judge_prompt_file).read_text(encoding="utf-8").strip()
    else:
        judge_prompt = build_teacher_judge_prompt()

    config = TeacherGenerationConfig(
        api_base=args.api_base,
        api_key=api_key,
        model=args.model,
        output_json=Path(args.output_json),
        output_jsonl=Path(args.output_jsonl) if args.output_jsonl else None,
        manifest_path=Path(args.manifest) if args.manifest else None,
        image_dir=Path(args.image_dir).resolve() if args.image_dir else None,
        teacher_prompt=teacher_prompt,
        student_prompt=args.student_prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        request_timeout=args.request_timeout,
        sleep_seconds=args.sleep_seconds,
        num_workers=args.num_workers,
        json_mode=args.json_mode,
        limit=args.limit,
        resume=not args.no_resume,
        filter_low_quality=not args.no_quality_filter,
        max_uncertain_main_lines=args.max_uncertain_main_lines,
        judge_model=args.judge_model,
        judge_prompt=judge_prompt,
        judge_temperature=args.judge_temperature,
        judge_max_tokens=args.judge_max_tokens,
        judge_min_average_score=args.judge_min_average_score,
        judge_min_visual_grounding=args.judge_min_visual_grounding,
        judge_min_uncertainty_honesty=args.judge_min_uncertainty_honesty,
        allow_cautious_accept=not args.reject_cautious,
    )

    summary = generate_sft_dataset(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
