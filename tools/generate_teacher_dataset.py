from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.palmistry import DEFAULT_STUDENT_STRUCTURED_PROMPT, build_teacher_structured_prompt
from src.palmistry.teacher import TeacherGenerationConfig, generate_sft_dataset


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
    parser.add_argument("--json-mode", action="store_true", help="Enable response_format=json_object if the provider supports it")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true", help="Do not reuse successful generations found in output-jsonl")
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
        json_mode=args.json_mode,
        limit=args.limit,
        resume=not args.no_resume,
    )

    summary = generate_sft_dataset(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
