from __future__ import annotations

import argparse
import os
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
from itertools import repeat
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_CLUSTER_REGEX = r"^(?P<base>.+)\.rf\.[^.]+$"


def default_num_workers() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, min(8, cpu_count - 1 if cpu_count > 2 else 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster augmented palm images, score quality, and export clean/hard-case manifests.",
    )
    parser.add_argument("--manifest", required=True, help="Input manifest in JSON or JSONL format")
    parser.add_argument("--image-dir", required=True, help="Image root used to resolve relative image paths")
    parser.add_argument("--output-clean-manifest", required=True, help="Output JSONL manifest for clean samples")
    parser.add_argument("--output-hard-manifest", required=True, help="Output JSONL manifest for hard/rejected samples")
    parser.add_argument("--output-summary", required=True, help="Output JSON summary path")
    parser.add_argument("--cluster-regex", default=DEFAULT_CLUSTER_REGEX, help="Regex used to strip augmentation suffixes")
    parser.add_argument("--max-per-cluster", type=int, default=2, help="Maximum clean samples to keep per source-image cluster")
    parser.add_argument("--min-quality-quantile", type=float, default=0.45, help="Global quality-score quantile floor for clean samples")
    parser.add_argument("--min-sharpness-quantile", type=float, default=0.40, help="Global sharpness quantile floor for clean samples")
    parser.add_argument("--secondary-cluster-ratio", type=float, default=0.88, help="Required quality ratio against the best sample for non-top cluster variants")
    parser.add_argument("--min-brightness", type=float, default=45.0, help="Minimum acceptable grayscale brightness")
    parser.add_argument("--max-brightness", type=float, default=215.0, help="Maximum acceptable grayscale brightness")
    parser.add_argument("--resize-max-side", type=int, default=512, help="Resize larger images before scoring to speed up evaluation")
    parser.add_argument("--num-workers", type=int, default=default_num_workers(), help="Number of worker threads used for image scoring")
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError("Manifest JSON must be a list.")
        return payload

    return [json.loads(line) for line in text.splitlines() if line.strip()]


def resolve_image_path(image_dir: Path, record: dict[str, Any]) -> Path:
    image_path = Path(record["image"])
    if image_path.is_absolute():
        return image_path
    return (image_dir / image_path).resolve()


def cluster_key_for_image(image_name: str, cluster_pattern: re.Pattern[str]) -> str:
    stem = Path(image_name).stem
    match = cluster_pattern.match(stem)
    return match.group("base") if match else stem


def resize_image(image: Image.Image, resize_max_side: int) -> Image.Image:
    width, height = image.size
    max_side = max(width, height)
    if max_side <= resize_max_side:
        return image

    scale = resize_max_side / max_side
    resized_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return image.resize(resized_size, Image.Resampling.BILINEAR)


def percentile_rank(sorted_values: list[float], value: float) -> float:
    if not sorted_values:
        return 0.0
    index = int(np.searchsorted(sorted_values, value, side="right"))
    return index / len(sorted_values)


def quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    q = max(0.0, min(1.0, q))
    index = min(len(sorted_values) - 1, max(0, int(math.floor(q * (len(sorted_values) - 1)))))
    return sorted_values[index]


def score_image(image_path: Path, resize_max_side: int) -> dict[str, float]:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        rgb = resize_image(rgb, resize_max_side)
        gray = np.asarray(rgb.convert("L"), dtype=np.float32)

    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)

    sharpness = float(np.mean(np.abs(gx)) + np.mean(np.abs(gy)))
    contrast = float(np.std(gray))
    brightness = float(np.mean(gray))

    hist, _ = np.histogram(gray, bins=256, range=(0, 255), density=True)
    hist = hist[hist > 0]
    entropy = float(-(hist * np.log2(hist)).sum())

    return {
        "sharpness": sharpness,
        "contrast": contrast,
        "brightness": brightness,
        "entropy": entropy,
    }


def enrich_record_for_scoring(
    record: dict[str, Any],
    image_dir: str,
    cluster_regex: str,
    resize_max_side: int,
) -> dict[str, Any]:
    image_dir_path = Path(image_dir)
    cluster_pattern = re.compile(cluster_regex)
    resolved_image = resolve_image_path(image_dir_path, record)
    metrics = score_image(resolved_image, resize_max_side)
    return {
        **record,
        "cluster_key": cluster_key_for_image(Path(record["image"]).name, cluster_pattern),
        "image_path": str(resolved_image),
        **metrics,
    }


def score_records_in_parallel(
    records: list[dict[str, Any]],
    *,
    image_dir: Path,
    cluster_regex: str,
    resize_max_side: int,
    num_workers: int,
) -> list[dict[str, Any]]:
    if num_workers <= 1:
        return [
            enrich_record_for_scoring(
                record,
                str(image_dir),
                cluster_regex,
                resize_max_side,
            )
            for record in records
        ]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        return list(
            executor.map(
                enrich_record_for_scoring,
                records,
                repeat(str(image_dir)),
                repeat(cluster_regex),
                repeat(resize_max_side),
                chunksize=32,
            )
        )


def build_quality_score(
    metrics: dict[str, float],
    sorted_sharpness: list[float],
    sorted_contrast: list[float],
    sorted_entropy: list[float],
) -> float:
    sharp_rank = percentile_rank(sorted_sharpness, metrics["sharpness"])
    contrast_rank = percentile_rank(sorted_contrast, metrics["contrast"])
    entropy_rank = percentile_rank(sorted_entropy, metrics["entropy"])

    brightness_penalty = min(1.0, abs(metrics["brightness"] - 128.0) / 128.0)
    score = 0.60 * sharp_rank + 0.25 * contrast_rank + 0.15 * entropy_rank - 0.12 * brightness_penalty
    return max(0.0, min(1.0, score))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    image_dir = Path(args.image_dir).resolve()
    output_clean_manifest = Path(args.output_clean_manifest)
    output_hard_manifest = Path(args.output_hard_manifest)
    output_summary = Path(args.output_summary)

    records = load_manifest(manifest_path)
    enriched_rows = score_records_in_parallel(
        records,
        image_dir=image_dir,
        cluster_regex=args.cluster_regex,
        resize_max_side=args.resize_max_side,
        num_workers=max(1, args.num_workers),
    )

    sorted_sharpness = sorted(row["sharpness"] for row in enriched_rows)
    sorted_contrast = sorted(row["contrast"] for row in enriched_rows)
    sorted_entropy = sorted(row["entropy"] for row in enriched_rows)

    for row in enriched_rows:
        row["quality_score"] = build_quality_score(row, sorted_sharpness, sorted_contrast, sorted_entropy)

    sorted_quality = sorted(row["quality_score"] for row in enriched_rows)
    quality_floor = quantile(sorted_quality, args.min_quality_quantile)
    sharpness_floor = quantile(sorted_sharpness, args.min_sharpness_quantile)

    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in enriched_rows:
        clusters[row["cluster_key"]].append(row)

    clean_rows: list[dict[str, Any]] = []
    hard_rows: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()

    for cluster_key, rows in clusters.items():
        ranked_rows = sorted(rows, key=lambda row: (row["quality_score"], row["sharpness"], row["contrast"]), reverse=True)
        best_quality = ranked_rows[0]["quality_score"]
        kept = 0

        for index, row in enumerate(ranked_rows):
            reasons: list[str] = []
            if row["brightness"] < args.min_brightness:
                reasons.append("too_dark")
            if row["brightness"] > args.max_brightness:
                reasons.append("too_bright")
            if row["quality_score"] < quality_floor:
                reasons.append("below_quality_floor")
            if row["sharpness"] < sharpness_floor:
                reasons.append("below_sharpness_floor")
            if index >= args.max_per_cluster:
                reasons.append("cluster_quota_exceeded")
            if index > 0 and row["quality_score"] < best_quality * args.secondary_cluster_ratio:
                reasons.append("cluster_relative_quality_low")

            enriched_record = {
                key: value
                for key, value in row.items()
                if key not in {"image_path"}
            }

            if reasons:
                enriched_record["quality_bucket"] = "hard_case"
                enriched_record["reject_reasons"] = reasons
                hard_rows.append(enriched_record)
                for reason in reasons:
                    reason_counts[reason] += 1
                continue

            kept += 1
            enriched_record["quality_bucket"] = "clean"
            enriched_record["cluster_rank"] = kept
            clean_rows.append(enriched_record)

    summary = {
        "input_manifest": str(manifest_path),
        "image_dir": str(image_dir),
        "total_records": len(records),
        "total_clusters": len(clusters),
        "clean_records": len(clean_rows),
        "hard_records": len(hard_rows),
        "avg_cluster_size": len(records) / max(1, len(clusters)),
        "max_per_cluster": args.max_per_cluster,
        "min_quality_quantile": args.min_quality_quantile,
        "min_sharpness_quantile": args.min_sharpness_quantile,
        "secondary_cluster_ratio": args.secondary_cluster_ratio,
        "brightness_range": [args.min_brightness, args.max_brightness],
        "num_workers": max(1, args.num_workers),
        "quality_floor": quality_floor,
        "sharpness_floor": sharpness_floor,
        "reject_reason_counts": reason_counts,
        "top_cluster_sizes": Counter(len(rows) for rows in clusters.values()).most_common(10),
        "output_clean_manifest": str(output_clean_manifest),
        "output_hard_manifest": str(output_hard_manifest),
    }

    write_jsonl(output_clean_manifest, clean_rows)
    write_jsonl(output_hard_manifest, hard_rows)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
