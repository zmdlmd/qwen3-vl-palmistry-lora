from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from src.palmistry.gate_policy import (
    GATE_DECISION_CAUTIOUS,
    GATE_DECISION_CONTINUE,
    GATE_DECISION_RETAKE,
)
from src.palmistry.schema import REQUIRED_LINE_NAMES, load_palmistry_payload
from tools.split_sft_dataset import DEFAULT_CLUSTER_REGEX, cluster_key_for_record


SEVERE_RETAKE_REASONS = {
    "below_quality_floor",
    "below_sharpness_floor",
    "too_dark",
    "too_bright",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a pseudo-labeled three-class gate-policy dataset from teacher JSON and hard manifests.",
    )
    parser.add_argument("--structured-json", required=True, help="Structured SFT dataset JSON path")
    parser.add_argument("--hard-manifest", default=None, help="Optional hard-case manifest JSONL path")
    parser.add_argument("--output-jsonl", required=True, help="Output gate dataset JSONL path")
    parser.add_argument("--output-summary", default=None, help="Optional summary JSON path")
    parser.add_argument("--cluster-regex", default=DEFAULT_CLUSTER_REGEX)
    parser.add_argument("--max-per-cluster", type=int, default=2)
    parser.add_argument("--continue-max-uncertain-main-lines", type=int, default=0)
    parser.add_argument("--cautious-max-uncertain-main-lines", type=int, default=3)
    return parser.parse_args()


def load_json_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("structured-json must contain a JSON list.")
    return payload


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def count_uncertain_main_lines(payload: dict[str, Any]) -> int:
    analysis = payload.get("palmistry_analysis", {})
    lines = analysis.get("lines", {})
    count = 0
    for line_name in REQUIRED_LINE_NAMES:
        line_payload = lines.get(line_name, {})
        text = "\n".join(str(line_payload.get(field, "")).strip() for field in line_payload)
        if any(marker in text for marker in ("难以判断", "无法", "模糊", "不可见", "不清晰", "噪点", "遮挡")):
            count += 1
    return count


def extract_teacher_json(record: dict[str, Any]) -> str:
    for turn in record.get("conversations", []):
        if turn.get("from") == "gpt":
            value = turn.get("value")
            if isinstance(value, str) and value.strip():
                return value.strip()
    raise ValueError(f"Record {record.get('id')} has no teacher assistant turn.")


def gate_decision_from_structured(
    uncertain_main_lines: int,
    *,
    continue_max_uncertain_main_lines: int,
    cautious_max_uncertain_main_lines: int,
) -> str:
    if uncertain_main_lines <= continue_max_uncertain_main_lines:
        return GATE_DECISION_CONTINUE
    if uncertain_main_lines <= cautious_max_uncertain_main_lines:
        return GATE_DECISION_CAUTIOUS
    return GATE_DECISION_RETAKE


def gate_decision_from_hard_reasons(reject_reasons: list[str]) -> str:
    if any(reason in SEVERE_RETAKE_REASONS for reason in reject_reasons):
        return GATE_DECISION_RETAKE
    return GATE_DECISION_CAUTIOUS


def main() -> None:
    args = parse_args()
    cluster_pattern = re.compile(args.cluster_regex)
    cluster_counts: Counter[str] = Counter()
    gate_rows: list[dict[str, Any]] = []

    structured_records = load_json_records(Path(args.structured_json))
    for record in structured_records:
        cluster_key = cluster_key_for_record(record, cluster_pattern)
        if cluster_counts[cluster_key] >= args.max_per_cluster:
            continue
        teacher_payload = load_palmistry_payload(extract_teacher_json(record))
        uncertain_main_lines = count_uncertain_main_lines(teacher_payload)
        decision = gate_decision_from_structured(
            uncertain_main_lines,
            continue_max_uncertain_main_lines=args.continue_max_uncertain_main_lines,
            cautious_max_uncertain_main_lines=args.cautious_max_uncertain_main_lines,
        )
        gate_rows.append(
            {
                "id": record["id"],
                "image": record["image"],
                "cluster_key": cluster_key,
                "gate_decision": decision,
                "uncertain_main_lines": uncertain_main_lines,
                "source": "structured_teacher",
            }
        )
        cluster_counts[cluster_key] += 1

    hard_count = 0
    if args.hard_manifest:
        for row in load_jsonl_records(Path(args.hard_manifest)):
            cluster_key = cluster_key_for_record(row, cluster_pattern)
            if cluster_counts[cluster_key] >= args.max_per_cluster:
                continue
            reject_reasons = [str(reason) for reason in row.get("reject_reasons", [])]
            decision = gate_decision_from_hard_reasons(reject_reasons)
            gate_rows.append(
                {
                    "id": row.get("id"),
                    "image": row.get("image"),
                    "cluster_key": cluster_key,
                    "gate_decision": decision,
                    "reject_reasons": reject_reasons,
                    "quality_bucket": row.get("quality_bucket"),
                    "quality_score": row.get("quality_score"),
                    "source": "hard_manifest",
                }
            )
            cluster_counts[cluster_key] += 1
            hard_count += 1

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in gate_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "structured_json": args.structured_json,
        "hard_manifest": args.hard_manifest,
        "output_jsonl": str(output_jsonl),
        "total_rows": len(gate_rows),
        "structured_rows": len(gate_rows) - hard_count,
        "hard_rows": hard_count,
        "max_per_cluster": args.max_per_cluster,
        "decision_counts": dict(Counter(row["gate_decision"] for row in gate_rows)),
        "source_counts": dict(Counter(row["source"] for row in gate_rows)),
    }
    if args.output_summary:
        Path(args.output_summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
