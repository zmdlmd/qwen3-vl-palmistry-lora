from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from src.palmistry.prompts import STYLE_OPTIONS, build_report_prompt
from tools.split_sft_dataset import DEFAULT_CLUSTER_REGEX, cluster_key_for_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a structured palmistry dataset into cluster-disjoint "
            "SFT-train / GRPO-train / eval-holdout subsets."
        ),
    )
    parser.add_argument("--input-json", required=True, help="Input structured SFT dataset JSON path")
    parser.add_argument("--output-sft-json", required=True, help="Output SFT-train JSON path")
    parser.add_argument("--output-grpo-json", required=True, help="Output GRPO-train structured JSON path")
    parser.add_argument("--output-eval-json", required=True, help="Output eval-holdout JSON path")
    parser.add_argument("--output-summary", default=None, help="Optional split summary JSON path")
    parser.add_argument(
        "--output-grpo-report-json",
        default=None,
        help="Optional report-style GRPO dataset converted from the GRPO-train structured split",
    )
    parser.add_argument("--eval-ratio", type=float, default=0.15, help="Holdout eval ratio by total record count")
    parser.add_argument(
        "--grpo-ratio",
        type=float,
        default=0.25,
        help="GRPO-train ratio by total record count after reserving eval via cluster split",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cluster-regex", default=DEFAULT_CLUSTER_REGEX, help="Regex used to strip augmentation suffixes")
    parser.add_argument("--report-style", default="balanced", choices=sorted(STYLE_OPTIONS.keys()))
    parser.add_argument("--report-id-suffix", default="-report-grpo", help="Suffix for optional report GRPO records")
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Input JSON must be a list.")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def choose_cluster_subset(
    candidate_cluster_keys: list[str],
    cluster_sizes: Counter[str],
    *,
    target_records: int,
    seed: int,
) -> set[str]:
    shuffled = list(candidate_cluster_keys)
    random.Random(seed).shuffle(shuffled)

    selected: set[str] = set()
    selected_records = 0
    for cluster_key in shuffled:
        if selected_records >= target_records:
            break
        selected.add(cluster_key)
        selected_records += cluster_sizes[cluster_key]
    return selected


def build_cluster_sizes(records: list[dict[str, Any]], cluster_pattern: re.Pattern[str]) -> Counter[str]:
    sizes: Counter[str] = Counter()
    for record in records:
        sizes[cluster_key_for_record(record, cluster_pattern)] += 1
    return sizes


def build_user_prompt(record: dict[str, Any], prompt_text: str) -> str:
    if "video" in record:
        return f"<video>\n{prompt_text}"
    if "image" in record:
        return f"<image>\n{prompt_text}"
    return prompt_text


def convert_to_report_grpo(record: dict[str, Any], prompt_text: str, id_suffix: str) -> dict[str, Any]:
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
    if not 0.0 < args.eval_ratio < 1.0:
        raise SystemExit("--eval-ratio must be between 0 and 1.")
    if not 0.0 < args.grpo_ratio < 1.0:
        raise SystemExit("--grpo-ratio must be between 0 and 1.")
    if args.eval_ratio + args.grpo_ratio >= 1.0:
        raise SystemExit("--eval-ratio + --grpo-ratio must be < 1.0.")

    input_path = Path(args.input_json)
    output_sft_path = Path(args.output_sft_json)
    output_grpo_path = Path(args.output_grpo_json)
    output_eval_path = Path(args.output_eval_json)
    output_summary_path = Path(args.output_summary) if args.output_summary else None
    output_grpo_report_path = Path(args.output_grpo_report_json) if args.output_grpo_report_json else None

    records = load_records(input_path)
    cluster_pattern = re.compile(args.cluster_regex)
    cluster_sizes = build_cluster_sizes(records, cluster_pattern)
    cluster_keys = sorted(cluster_sizes)

    total_records = len(records)
    eval_target = max(1, int(round(total_records * args.eval_ratio)))
    grpo_target = max(1, int(round(total_records * args.grpo_ratio)))

    eval_clusters = choose_cluster_subset(cluster_keys, cluster_sizes, target_records=eval_target, seed=args.seed)
    remaining_clusters = [key for key in cluster_keys if key not in eval_clusters]
    grpo_clusters = choose_cluster_subset(
        remaining_clusters,
        cluster_sizes,
        target_records=grpo_target,
        seed=args.seed + 1,
    )

    sft_records: list[dict[str, Any]] = []
    grpo_records: list[dict[str, Any]] = []
    eval_records: list[dict[str, Any]] = []

    sft_cluster_keys: set[str] = set()
    grpo_cluster_keys: set[str] = set()
    eval_cluster_keys: set[str] = set()

    for record in records:
        cluster_key = cluster_key_for_record(record, cluster_pattern)
        if cluster_key in eval_clusters:
            eval_records.append(record)
            eval_cluster_keys.add(cluster_key)
        elif cluster_key in grpo_clusters:
            grpo_records.append(record)
            grpo_cluster_keys.add(cluster_key)
        else:
            sft_records.append(record)
            sft_cluster_keys.add(cluster_key)

    write_json(output_sft_path, sft_records)
    write_json(output_grpo_path, grpo_records)
    write_json(output_eval_path, eval_records)

    report_grpo_count = 0
    if output_grpo_report_path is not None:
        prompt_text = build_report_prompt(args.report_style)
        report_records = [
            convert_to_report_grpo(record, prompt_text, args.report_id_suffix)
            for record in grpo_records
        ]
        write_json(output_grpo_report_path, report_records)
        report_grpo_count = len(report_records)

    summary = {
        "input_json": str(input_path),
        "output_sft_json": str(output_sft_path),
        "output_grpo_json": str(output_grpo_path),
        "output_eval_json": str(output_eval_path),
        "output_grpo_report_json": str(output_grpo_report_path) if output_grpo_report_path else None,
        "total_records": total_records,
        "total_clusters": len(cluster_sizes),
        "sft_records": len(sft_records),
        "grpo_records": len(grpo_records),
        "eval_records": len(eval_records),
        "sft_clusters": len(sft_cluster_keys),
        "grpo_clusters": len(grpo_cluster_keys),
        "eval_clusters": len(eval_cluster_keys),
        "eval_ratio_target": args.eval_ratio,
        "grpo_ratio_target": args.grpo_ratio,
        "sft_ratio_actual": len(sft_records) / total_records if total_records else 0.0,
        "grpo_ratio_actual": len(grpo_records) / total_records if total_records else 0.0,
        "eval_ratio_actual": len(eval_records) / total_records if total_records else 0.0,
        "seed": args.seed,
        "cluster_regex": args.cluster_regex,
        "cluster_overlap": {
            "sft_grpo": len(sft_cluster_keys & grpo_cluster_keys),
            "sft_eval": len(sft_cluster_keys & eval_cluster_keys),
            "grpo_eval": len(grpo_cluster_keys & eval_cluster_keys),
        },
        "report_style": args.report_style,
        "report_grpo_records": report_grpo_count,
    }
    if output_summary_path is not None:
        write_json(output_summary_path, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
