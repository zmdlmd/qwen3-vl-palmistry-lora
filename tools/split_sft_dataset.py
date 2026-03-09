from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_CLUSTER_REGEX = r"^(?P<base>.+)\.rf\.[^.]+$"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split an LLaVA-style SFT dataset into train/val sets, grouped by source-image cluster.",
    )
    parser.add_argument("--input-json", required=True, help="Input SFT dataset JSON path")
    parser.add_argument("--output-train-json", required=True, help="Output train JSON path")
    parser.add_argument("--output-val-json", required=True, help="Output validation JSON path")
    parser.add_argument("--output-summary", default=None, help="Optional split summary JSON path")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio by record count target")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cluster-regex", default=DEFAULT_CLUSTER_REGEX, help="Regex used to strip augmentation suffixes from image stems")
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Input JSON must be a list.")
    return payload


def cluster_key_for_record(record: dict[str, Any], cluster_pattern: re.Pattern[str]) -> str:
    image_value = record.get("image")
    if not image_value:
        raise ValueError("Each record must contain an 'image' field.")
    stem = Path(str(image_value)).stem
    match = cluster_pattern.match(stem)
    return match.group("base") if match else stem


def choose_val_clusters(
    records: list[dict[str, Any]],
    *,
    val_ratio: float,
    seed: int,
    cluster_pattern: re.Pattern[str],
) -> tuple[set[str], Counter[str]]:
    cluster_sizes: Counter[str] = Counter()
    for record in records:
        cluster_sizes[cluster_key_for_record(record, cluster_pattern)] += 1

    cluster_keys = sorted(cluster_sizes)
    random.Random(seed).shuffle(cluster_keys)

    target_val_records = max(1, int(round(len(records) * val_ratio)))
    selected: set[str] = set()
    selected_records = 0

    for cluster_key in cluster_keys:
        if selected_records >= target_val_records:
            break
        selected.add(cluster_key)
        selected_records += cluster_sizes[cluster_key]

    return selected, cluster_sizes


def write_json(path: Path, payload: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not 0.0 < args.val_ratio < 1.0:
        raise SystemExit("--val-ratio must be between 0 and 1.")

    input_path = Path(args.input_json)
    output_train_path = Path(args.output_train_json)
    output_val_path = Path(args.output_val_json)
    output_summary_path = Path(args.output_summary) if args.output_summary else None

    records = load_records(input_path)
    cluster_pattern = re.compile(args.cluster_regex)
    val_clusters, cluster_sizes = choose_val_clusters(
        records,
        val_ratio=args.val_ratio,
        seed=args.seed,
        cluster_pattern=cluster_pattern,
    )

    train_records: list[dict[str, Any]] = []
    val_records: list[dict[str, Any]] = []
    train_cluster_keys: set[str] = set()
    val_cluster_keys: set[str] = set()

    for record in records:
        cluster_key = cluster_key_for_record(record, cluster_pattern)
        if cluster_key in val_clusters:
            val_records.append(record)
            val_cluster_keys.add(cluster_key)
        else:
            train_records.append(record)
            train_cluster_keys.add(cluster_key)

    write_json(output_train_path, train_records)
    write_json(output_val_path, val_records)

    if output_summary_path is not None:
        summary = {
            "input_json": str(input_path),
            "output_train_json": str(output_train_path),
            "output_val_json": str(output_val_path),
            "total_records": len(records),
            "train_records": len(train_records),
            "val_records": len(val_records),
            "total_clusters": len(cluster_sizes),
            "train_clusters": len(train_cluster_keys),
            "val_clusters": len(val_cluster_keys),
            "val_ratio_target": args.val_ratio,
            "val_ratio_actual": len(val_records) / len(records) if records else 0.0,
            "seed": args.seed,
            "cluster_regex": args.cluster_regex,
        }
        output_summary_path.parent.mkdir(parents=True, exist_ok=True)
        output_summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "total_records": len(records),
                "train_records": len(train_records),
                "val_records": len(val_records),
                "total_clusters": len(cluster_sizes),
                "train_clusters": len(train_cluster_keys),
                "val_clusters": len(val_cluster_keys),
                "val_ratio_actual": len(val_records) / len(records) if records else 0.0,
                "seed": args.seed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
