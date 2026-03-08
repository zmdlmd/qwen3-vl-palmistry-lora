from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.palmistry.prompts import STYLE_OPTIONS, build_report_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a structured palmistry dataset into a report-oriented GRPO dataset.",
    )
    parser.add_argument("--input-json", required=True, help="Input LLaVA-style JSON dataset path")
    parser.add_argument("--output-json", required=True, help="Output GRPO dataset path")
    parser.add_argument("--style", default="balanced", choices=sorted(STYLE_OPTIONS.keys()))
    parser.add_argument("--prompt-file", default=None, help="Optional file containing a custom report prompt")
    parser.add_argument("--id-suffix", default="-report-grpo", help="Suffix appended to each record id")
    return parser.parse_args()


def build_user_prompt(record: dict[str, Any], prompt_text: str) -> str:
    if "video" in record:
        return f"<video>\n{prompt_text}"
    if "image" in record:
        return f"<image>\n{prompt_text}"
    return prompt_text


def convert_record(record: dict[str, Any], prompt_text: str, id_suffix: str) -> dict[str, Any]:
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or len(conversations) < 2:
        raise ValueError(f"Invalid conversations field for record: {record.get('id', '<unknown>')}")

    converted = dict(record)
    converted_conversations = [dict(turn) for turn in conversations]
    converted_conversations[0]["value"] = build_user_prompt(record, prompt_text)
    converted["conversations"] = converted_conversations
    converted["grpo_mode"] = "report"

    record_id = converted.get("id")
    if isinstance(record_id, str) and id_suffix:
        converted["id"] = f"{record_id}{id_suffix}"

    return converted


def main() -> None:
    args = parse_args()
    source_records = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    if not isinstance(source_records, list):
        raise SystemExit("Input dataset must be a JSON array.")

    if args.prompt_file:
        prompt_text = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    else:
        prompt_text = build_report_prompt(args.style)

    converted_records = [convert_record(record, prompt_text, args.id_suffix) for record in source_records]

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(converted_records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "total": len(source_records),
                "output_json": str(output_path),
                "style": args.style,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
