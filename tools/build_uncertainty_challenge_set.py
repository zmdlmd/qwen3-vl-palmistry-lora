from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from src.palmistry.schema import REQUIRED_LINE_NAMES, load_palmistry_payload
from tools.split_sft_dataset import DEFAULT_CLUSTER_REGEX, cluster_key_for_record


UNCERTAINTY_TERMS = (
    "难以判断",
    "无法",
    "不可见",
    "不可辨",
    "模糊",
    "不清晰",
    "噪点",
    "遮挡",
    "可见信息有限",
    "无法准确判断",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a fixed uncertainty-focused challenge set from the teacher/SFT dataset.",
    )
    parser.add_argument("--input-json", required=True, help="Input LLaVA-style SFT dataset JSON")
    parser.add_argument("--output-json", required=True, help="Output challenge JSON path")
    parser.add_argument("--output-summary", required=True, help="Output summary JSON path")
    parser.add_argument("--limit", type=int, default=200, help="Maximum number of challenge samples to keep")
    parser.add_argument("--cluster-regex", default=DEFAULT_CLUSTER_REGEX, help="Regex used to strip augmentation suffixes")
    parser.add_argument("--max-per-cluster", type=int, default=1, help="Maximum samples to keep per source-image cluster")
    parser.add_argument(
        "--selection-mode",
        default="stratified",
        choices=["stratified", "top"],
        help="top picks the globally highest-scoring samples; stratified mixes samples across uncertainty levels",
    )
    parser.add_argument(
        "--min-uncertain-main-lines",
        type=int,
        default=2,
        help="Minimum number of uncertain required palm lines to qualify by default",
    )
    parser.add_argument(
        "--min-uncertainty-hits",
        type=int,
        default=4,
        help="Minimum number of uncertainty term hits in the teacher payload to qualify by default",
    )
    parser.add_argument(
        "--exclude-sample-jsonl",
        action="append",
        default=[],
        help="Optional evaluation samples.jsonl file(s) whose ids should be excluded",
    )
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Input JSON must be a list.")
    return payload


def extract_teacher_json(record: dict[str, Any]) -> str:
    for turn in record.get("conversations", []):
        if turn.get("from") == "gpt":
            value = turn.get("value")
            if isinstance(value, str) and value.strip():
                return value.strip()
    raise ValueError(f"Record {record.get('id')} has no teacher assistant turn.")


def line_uncertainty_text(line_payload: dict[str, Any]) -> str:
    return "\n".join(str(line_payload.get(field, "")).strip() for field in line_payload)


def line_is_uncertain(line_payload: dict[str, Any]) -> bool:
    text = line_uncertainty_text(line_payload)
    return any(term in text for term in UNCERTAINTY_TERMS)


def count_uncertain_lines(payload: dict[str, Any]) -> tuple[int, int]:
    analysis = payload.get("palmistry_analysis", {})
    lines = analysis.get("lines", {})
    uncertain_required = 0
    uncertain_total = 0
    for line_name, line_payload in lines.items():
        if not isinstance(line_payload, dict):
            continue
        if not line_is_uncertain(line_payload):
            continue
        uncertain_total += 1
        if line_name in REQUIRED_LINE_NAMES:
            uncertain_required += 1
    return uncertain_required, uncertain_total


def count_uncertainty_hits(payload: dict[str, Any]) -> int:
    text = json.dumps(payload, ensure_ascii=False)
    return sum(text.count(term) for term in UNCERTAINTY_TERMS)


def extract_report_uncertainty_hits(payload: dict[str, Any]) -> int:
    analysis = payload.get("palmistry_analysis", {})
    report_text = "\n".join(
        str(analysis.get(field, "")).strip()
        for field in (
            "traditional_report",
            "japanese_report",
            "mystic_energy_report",
            "medical_report",
        )
    )
    return sum(report_text.count(term) for term in UNCERTAINTY_TERMS)


def load_excluded_ids(sample_jsonl_paths: list[str]) -> set[str]:
    excluded: set[str] = set()
    for raw_path in sample_jsonl_paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Exclude samples JSONL does not exist: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = row.get("id")
            if row_id:
                excluded.add(str(row_id))
    return excluded


def challenge_score(
    *,
    uncertain_required: int,
    uncertain_total: int,
    uncertainty_hits: int,
    report_uncertainty_hits: int,
) -> float:
    score = 0.0
    score += uncertain_required * 10.0
    score += uncertain_total * 3.0
    score += min(uncertainty_hits, 12) * 0.5
    score += min(report_uncertainty_hits, 8) * 0.75
    if uncertain_required >= 3:
        score += 4.0
    if uncertain_total >= 4:
        score += 2.0
    return score


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    records = load_records(Path(args.input_json))
    cluster_pattern = re.compile(args.cluster_regex)
    excluded_ids = load_excluded_ids(args.exclude_sample_jsonl)

    candidates: list[dict[str, Any]] = []
    skipped_excluded = 0

    for record in records:
        record_id = str(record.get("id", ""))
        if not record_id:
            continue
        if record_id in excluded_ids:
            skipped_excluded += 1
            continue

        teacher_json = extract_teacher_json(record)
        payload = load_palmistry_payload(teacher_json)
        uncertain_required, uncertain_total = count_uncertain_lines(payload)
        uncertainty_hits = count_uncertainty_hits(payload)
        report_hits = extract_report_uncertainty_hits(payload)

        qualifies = (
            uncertain_required >= args.min_uncertain_main_lines
            or uncertainty_hits >= args.min_uncertainty_hits
        )
        if not qualifies:
            continue

        candidates.append(
            {
                "record": record,
                "cluster_key": cluster_key_for_record(record, cluster_pattern),
                "uncertain_required_lines": uncertain_required,
                "uncertain_total_lines": uncertain_total,
                "uncertainty_hits": uncertainty_hits,
                "report_uncertainty_hits": report_hits,
                "challenge_score": challenge_score(
                    uncertain_required=uncertain_required,
                    uncertain_total=uncertain_total,
                    uncertainty_hits=uncertainty_hits,
                    report_uncertainty_hits=report_hits,
                ),
            }
        )

    ranked = sorted(
        candidates,
        key=lambda row: (
            row["challenge_score"],
            row["uncertain_required_lines"],
            row["uncertain_total_lines"],
            row["uncertainty_hits"],
            row["record"]["id"],
        ),
        reverse=True,
    )

    selected: list[dict[str, Any]] = []
    selected_meta: list[dict[str, Any]] = []
    cluster_counts: dict[str, int] = {}

    def try_take(row: dict[str, Any]) -> bool:
        cluster_key = row["cluster_key"]
        if cluster_counts.get(cluster_key, 0) >= args.max_per_cluster:
            return False
        cluster_counts[cluster_key] = cluster_counts.get(cluster_key, 0) + 1
        selected.append(row["record"])
        selected_meta.append(
            {
                "id": row["record"]["id"],
                "image": row["record"]["image"],
                "cluster_key": cluster_key,
                "uncertain_required_lines": row["uncertain_required_lines"],
                "uncertain_total_lines": row["uncertain_total_lines"],
                "uncertainty_hits": row["uncertainty_hits"],
                "report_uncertainty_hits": row["report_uncertainty_hits"],
                "challenge_score": row["challenge_score"],
            }
        )
        return True

    if args.selection_mode == "top":
        for row in ranked:
            try_take(row)
            if len(selected) >= args.limit:
                break
    else:
        buckets: dict[int, list[dict[str, Any]]] = {}
        for row in ranked:
            buckets.setdefault(row["uncertain_required_lines"], []).append(row)
        bucket_levels = sorted(
            [level for level in buckets if level >= args.min_uncertain_main_lines],
            reverse=True,
        )
        bucket_positions = {level: 0 for level in bucket_levels}

        while len(selected) < args.limit and bucket_levels:
            made_progress = False
            for level in bucket_levels:
                rows = buckets[level]
                pos = bucket_positions[level]
                while pos < len(rows):
                    row = rows[pos]
                    pos += 1
                    if try_take(row):
                        made_progress = True
                        break
                bucket_positions[level] = pos
                if len(selected) >= args.limit:
                    break
            if not made_progress:
                break

    output_json = Path(args.output_json)
    output_summary = Path(args.output_summary)

    write_json(output_json, selected)
    summary = {
        "input_json": str(Path(args.input_json).resolve()),
        "output_json": str(output_json.resolve()),
        "requested_limit": args.limit,
        "selected_count": len(selected),
        "candidate_count": len(candidates),
        "excluded_ids_count": len(excluded_ids),
        "skipped_excluded": skipped_excluded,
        "cluster_regex": args.cluster_regex,
        "max_per_cluster": args.max_per_cluster,
        "selection_mode": args.selection_mode,
        "min_uncertain_main_lines": args.min_uncertain_main_lines,
        "min_uncertainty_hits": args.min_uncertainty_hits,
        "selected_clusters": len(cluster_counts),
        "avg_uncertain_required_lines": (
            sum(row["uncertain_required_lines"] for row in selected_meta) / len(selected_meta)
            if selected_meta
            else 0.0
        ),
        "avg_uncertain_total_lines": (
            sum(row["uncertain_total_lines"] for row in selected_meta) / len(selected_meta)
            if selected_meta
            else 0.0
        ),
        "avg_uncertainty_hits": (
            sum(row["uncertainty_hits"] for row in selected_meta) / len(selected_meta)
            if selected_meta
            else 0.0
        ),
        "selected_by_uncertain_required_lines": {
            str(level): sum(1 for row in selected_meta if row["uncertain_required_lines"] == level)
            for level in sorted({row["uncertain_required_lines"] for row in selected_meta})
        },
        "preview": selected_meta[:20],
    }
    write_json(output_summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
