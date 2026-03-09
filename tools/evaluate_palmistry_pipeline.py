from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.palmistry import PalmistryPipeline
from src.palmistry.reward_funcs_report import (
    reference_alignment_reward as report_reference_alignment_reward,
    report_format_reward,
    safety_language_reward as report_safety_language_reward,
    section_structure_reward,
    uncertainty_honesty_reward,
)
from src.palmistry.reward_funcs_structured import (
    json_schema_reward,
    line_field_coverage_reward,
    reference_alignment_reward as structured_reference_alignment_reward,
    report_field_coverage_reward,
    safety_language_reward as structured_safety_language_reward,
)
from src.palmistry.schema import REQUIRED_LINE_NAMES, load_palmistry_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the palmistry inference pipeline on val data and hard-case manifests.",
    )
    parser.add_argument("--base-model", required=True, help="Base Qwen3-VL model path or HF id")
    parser.add_argument("--lora-path", default=None, help="Optional LoRA adapter path")
    parser.add_argument("--image-root", required=True, help="Root directory for palm images")
    parser.add_argument("--val-json", default=None, help="LLaVA-style validation JSON with teacher labels")
    parser.add_argument("--hard-manifest", default=None, help="Hard-case JSONL manifest")
    parser.add_argument("--output-json", required=True, help="Output summary JSON path")
    parser.add_argument("--output-jsonl", default=None, help="Optional per-sample evaluation log JSONL path")
    parser.add_argument("--val-limit", type=int, default=None, help="Optional cap for validation examples")
    parser.add_argument("--hard-limit", type=int, default=None, help="Optional cap for hard-case examples")
    parser.add_argument("--device", default=None, help="Runtime device")
    parser.add_argument("--device-map", default="auto", help="Transformers device_map")
    parser.add_argument("--torch-dtype", default="auto", help="auto | bf16 | fp16 | fp32")
    parser.add_argument("--style", default="balanced", choices=["balanced", "soft", "professional"])
    parser.add_argument("--report-max-new-tokens", type=int, default=900)
    parser.add_argument("--structured-max-new-tokens", type=int, default=1400)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    return parser.parse_args()


def load_json_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list.")
    return payload


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def maybe_limit(records: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None:
        return records
    return records[:limit]


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


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


def structured_metrics(predicted_json: str, reference_json: str) -> dict[str, float]:
    return {
        "json_schema": json_schema_reward([predicted_json])[0],
        "line_field_coverage": line_field_coverage_reward([predicted_json])[0],
        "report_field_coverage": report_field_coverage_reward([predicted_json])[0],
        "reference_alignment": structured_reference_alignment_reward([predicted_json], [reference_json])[0],
        "safety_language": structured_safety_language_reward([predicted_json])[0],
    }


def report_metrics(predicted_report: str, reference_json: str) -> dict[str, float]:
    return {
        "format": report_format_reward([predicted_report])[0],
        "section_structure": section_structure_reward([predicted_report])[0],
        "reference_alignment": report_reference_alignment_reward([predicted_report], [reference_json])[0],
        "uncertainty_honesty": uncertainty_honesty_reward([predicted_report], [reference_json])[0],
        "safety_language": report_safety_language_reward([predicted_report])[0],
    }


def evaluate_val_split(
    pipeline: PalmistryPipeline,
    records: list[dict[str, Any]],
    *,
    image_root: Path,
    style: str,
    report_max_new_tokens: int,
    structured_max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    per_sample: list[dict[str, Any]] = []
    structured_summary: defaultdict[str, list[float]] = defaultdict(list)
    report_summary: defaultdict[str, list[float]] = defaultdict(list)
    counts = Counter()

    for record in records:
        reference_json = extract_teacher_json(record)
        reference_payload = load_palmistry_payload(reference_json)
        expected_uncertain_main_lines = count_uncertain_main_lines(reference_payload)
        expected_low_confidence = expected_uncertain_main_lines > 1

        result = pipeline.analyze_detailed(
            image_root / record["image"],
            style=style,
            max_new_tokens=report_max_new_tokens,
            structured_max_new_tokens=structured_max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        sample_row: dict[str, Any] = {
            "split": "val",
            "id": record["id"],
            "image": record["image"],
            "expected_low_confidence": expected_low_confidence,
            "expected_uncertain_main_lines": expected_uncertain_main_lines,
            "pred_low_confidence": result.low_confidence,
            "pred_uncertain_main_lines": result.uncertain_main_lines,
            "pred_uncertain_lines": result.uncertain_lines,
            "visibility_assessment": result.visibility_assessment,
            "caution_message": result.caution_message,
            "error": result.error,
        }
        visibility_retake = bool(
            result.visibility_assessment
            and pipeline._visibility_requires_retake(result.visibility_assessment)
        )
        full_report_generated = bool(result.report) and result.report != result.caution_message

        counts["samples"] += 1
        counts["low_confidence"] += int(result.low_confidence)
        counts["expected_low_confidence"] += int(expected_low_confidence)
        counts["gate_match"] += int(result.low_confidence == expected_low_confidence)
        counts["visibility_retake"] += int(visibility_retake)
        counts["full_report_generated"] += int(full_report_generated)

        if result.structured_json:
            sample_structured = structured_metrics(result.structured_json, reference_json)
            sample_row["structured_metrics"] = sample_structured
            for key, value in sample_structured.items():
                structured_summary[key].append(value)
            counts["structured_available"] += 1

        if full_report_generated:
            sample_report = report_metrics(result.report, reference_json)
            sample_row["report_metrics"] = sample_report
            for key, value in sample_report.items():
                report_summary[key].append(value)
            counts["report_available"] += 1

        per_sample.append(sample_row)

    summary = {
        "num_samples": counts["samples"],
        "low_confidence_rate": counts["low_confidence"] / counts["samples"] if counts["samples"] else 0.0,
        "expected_low_confidence_rate": counts["expected_low_confidence"] / counts["samples"] if counts["samples"] else 0.0,
        "gate_match_rate": counts["gate_match"] / counts["samples"] if counts["samples"] else 0.0,
        "visibility_retake_rate": counts["visibility_retake"] / counts["samples"] if counts["samples"] else 0.0,
        "structured_available_rate": counts["structured_available"] / counts["samples"] if counts["samples"] else 0.0,
        "full_report_rate": counts["full_report_generated"] / counts["samples"] if counts["samples"] else 0.0,
        "structured_metrics": {key: safe_mean(values) for key, values in structured_summary.items()},
        "report_metrics": {key: safe_mean(values) for key, values in report_summary.items()},
    }
    return summary, per_sample


def evaluate_hard_cases(
    pipeline: PalmistryPipeline,
    rows: list[dict[str, Any]],
    *,
    image_root: Path,
    style: str,
    report_max_new_tokens: int,
    structured_max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    per_sample: list[dict[str, Any]] = []
    counts = Counter()
    by_reason: dict[str, Counter[str]] = defaultdict(Counter)

    for row in rows:
        result = pipeline.analyze_detailed(
            image_root / row["image"],
            style=style,
            max_new_tokens=report_max_new_tokens,
            structured_max_new_tokens=structured_max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        reject_reasons = [str(reason) for reason in row.get("reject_reasons", [])]
        sample_row = {
            "split": "hard_cases",
            "id": row.get("id"),
            "image": row.get("image"),
            "reject_reasons": reject_reasons,
            "quality_bucket": row.get("quality_bucket"),
            "quality_score": row.get("quality_score"),
            "pred_low_confidence": result.low_confidence,
            "pred_uncertain_main_lines": result.uncertain_main_lines,
            "pred_uncertain_lines": result.uncertain_lines,
            "visibility_assessment": result.visibility_assessment,
            "caution_message": result.caution_message,
            "error": result.error,
        }
        per_sample.append(sample_row)
        visibility_retake = bool(
            result.visibility_assessment
            and pipeline._visibility_requires_retake(result.visibility_assessment)
        )
        full_report_generated = bool(result.report) and result.report != result.caution_message

        counts["samples"] += 1
        counts["low_confidence"] += int(result.low_confidence)
        counts["full_report_generated"] += int(full_report_generated)
        counts["visibility_retake"] += int(visibility_retake)

        for reason in reject_reasons or ["__none__"]:
            by_reason[reason]["samples"] += 1
            by_reason[reason]["low_confidence"] += int(result.low_confidence)
            by_reason[reason]["full_report_generated"] += int(full_report_generated)
            by_reason[reason]["visibility_retake"] += int(visibility_retake)

    summary = {
        "num_samples": counts["samples"],
        "low_confidence_rate": counts["low_confidence"] / counts["samples"] if counts["samples"] else 0.0,
        "full_report_rate": counts["full_report_generated"] / counts["samples"] if counts["samples"] else 0.0,
        "visibility_retake_rate": counts["visibility_retake"] / counts["samples"] if counts["samples"] else 0.0,
        "by_reject_reason": {
            reason: {
                "samples": counter["samples"],
                "low_confidence_rate": counter["low_confidence"] / counter["samples"] if counter["samples"] else 0.0,
                "full_report_rate": counter["full_report_generated"] / counter["samples"] if counter["samples"] else 0.0,
                "visibility_retake_rate": counter["visibility_retake"] / counter["samples"] if counter["samples"] else 0.0,
            }
            for reason, counter in sorted(by_reason.items())
        },
    }
    return summary, per_sample


def main() -> None:
    args = parse_args()
    image_root = Path(args.image_root).resolve()
    if not image_root.exists():
        raise SystemExit(f"Image root does not exist: {image_root}")

    pipeline = PalmistryPipeline(
        base_model_path=args.base_model,
        lora_path=args.lora_path,
        device=args.device,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
    )

    summary: dict[str, Any] = {
        "base_model": args.base_model,
        "lora_path": args.lora_path,
        "image_root": str(image_root),
        "style": args.style,
    }
    per_sample_rows: list[dict[str, Any]] = []

    if args.val_json:
        val_records = maybe_limit(load_json_records(Path(args.val_json)), args.val_limit)
        val_summary, val_rows = evaluate_val_split(
            pipeline,
            val_records,
            image_root=image_root,
            style=args.style,
            report_max_new_tokens=args.report_max_new_tokens,
            structured_max_new_tokens=args.structured_max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        summary["val"] = val_summary
        per_sample_rows.extend(val_rows)

    if args.hard_manifest:
        hard_rows = maybe_limit(load_jsonl_records(Path(args.hard_manifest)), args.hard_limit)
        hard_summary, hard_sample_rows = evaluate_hard_cases(
            pipeline,
            hard_rows,
            image_root=image_root,
            style=args.style,
            report_max_new_tokens=args.report_max_new_tokens,
            structured_max_new_tokens=args.structured_max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        summary["hard_cases"] = hard_summary
        per_sample_rows.extend(hard_sample_rows)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.output_jsonl:
        dump_jsonl(Path(args.output_jsonl), per_sample_rows)


if __name__ == "__main__":
    main()
