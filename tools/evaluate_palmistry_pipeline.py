from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

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
    parser.add_argument("--gate-classifier-path", default=None, help="Optional standalone gate classifier checkpoint path")
    parser.add_argument("--gate-classifier-device", default=None, help="Standalone gate classifier runtime device")
    parser.add_argument("--style", default="balanced", choices=["balanced", "soft", "professional"])
    parser.add_argument("--report-max-new-tokens", type=int, default=900)
    parser.add_argument("--structured-max-new-tokens", type=int, default=1400)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--hard-mode",
        default="gate_only",
        choices=["gate_only", "full"],
        help="gate_only only runs visibility gating on hard cases; full runs the full pipeline.",
    )
    parser.add_argument(
        "--val-mode",
        default="full",
        choices=["gate_only", "full"],
        help="gate_only only runs gate evaluation on val samples; full runs the complete pipeline.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N samples.",
    )
    parser.add_argument(
        "--summary-every",
        type=int,
        default=25,
        help="Rewrite the summary JSON every N samples.",
    )
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


def append_jsonl_row(path: Path | None, row: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def count_gate_decision(counts: Counter[str], decision: str) -> None:
    counts[f"gate_{decision}"] += 1


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


def summarize_val(
    counts: Counter[str],
    structured_summary: dict[str, list[float]],
    report_summary: dict[str, list[float]],
) -> dict[str, Any]:
    return {
        "num_samples": counts["samples"],
        "low_confidence_rate": counts["low_confidence"] / counts["samples"] if counts["samples"] else 0.0,
        "expected_low_confidence_rate": counts["expected_low_confidence"] / counts["samples"] if counts["samples"] else 0.0,
        "gate_match_rate": counts["gate_match"] / counts["samples"] if counts["samples"] else 0.0,
        "gate_continue_rate": counts["gate_continue"] / counts["samples"] if counts["samples"] else 0.0,
        "gate_cautious_rate": counts["gate_cautious"] / counts["samples"] if counts["samples"] else 0.0,
        "gate_retake_rate": counts["gate_retake"] / counts["samples"] if counts["samples"] else 0.0,
        "visibility_cautious_rate": counts["visibility_cautious"] / counts["samples"] if counts["samples"] else 0.0,
        "visibility_retake_rate": counts["visibility_retake"] / counts["samples"] if counts["samples"] else 0.0,
        "structured_available_rate": counts["structured_available"] / counts["samples"] if counts["samples"] else 0.0,
        "full_report_rate": counts["full_report_generated"] / counts["samples"] if counts["samples"] else 0.0,
        "structured_metrics": {key: safe_mean(values) for key, values in structured_summary.items()},
        "report_metrics": {key: safe_mean(values) for key, values in report_summary.items()},
    }


def summarize_hard(counts: Counter[str], by_reason: dict[str, Counter[str]]) -> dict[str, Any]:
    return {
        "num_samples": counts["samples"],
        "low_confidence_rate": counts["low_confidence"] / counts["samples"] if counts["samples"] else 0.0,
        "gate_continue_rate": counts["gate_continue"] / counts["samples"] if counts["samples"] else 0.0,
        "gate_cautious_rate": counts["gate_cautious"] / counts["samples"] if counts["samples"] else 0.0,
        "gate_retake_rate": counts["gate_retake"] / counts["samples"] if counts["samples"] else 0.0,
        "visibility_cautious_rate": counts["visibility_cautious"] / counts["samples"] if counts["samples"] else 0.0,
        "full_report_rate": counts["full_report_generated"] / counts["samples"] if counts["samples"] else 0.0,
        "visibility_retake_rate": counts["visibility_retake"] / counts["samples"] if counts["samples"] else 0.0,
        "by_reject_reason": {
            reason: {
                "samples": counter["samples"],
                "low_confidence_rate": counter["low_confidence"] / counter["samples"] if counter["samples"] else 0.0,
                "gate_continue_rate": counter["gate_continue"] / counter["samples"] if counter["samples"] else 0.0,
                "gate_cautious_rate": counter["gate_cautious"] / counter["samples"] if counter["samples"] else 0.0,
                "gate_retake_rate": counter["gate_retake"] / counter["samples"] if counter["samples"] else 0.0,
                "visibility_cautious_rate": counter["visibility_cautious"] / counter["samples"] if counter["samples"] else 0.0,
                "full_report_rate": counter["full_report_generated"] / counter["samples"] if counter["samples"] else 0.0,
                "visibility_retake_rate": counter["visibility_retake"] / counter["samples"] if counter["samples"] else 0.0,
            }
            for reason, counter in sorted(by_reason.items())
        },
    }


def build_root_summary(
    base_summary: dict[str, Any],
    *,
    val_summary: dict[str, Any] | None = None,
    hard_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = dict(base_summary)
    if val_summary is not None:
        summary["val"] = val_summary
    if hard_summary is not None:
        summary["hard_cases"] = hard_summary
    return summary


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def should_emit(index: int, total: int, every: int) -> bool:
    return index == 1 or index == total or (every > 0 and index % every == 0)


def print_progress(split_name: str, index: int, total: int, start_time: float) -> None:
    elapsed = max(time.time() - start_time, 1e-6)
    rate = index / elapsed
    eta = (total - index) / rate if rate > 0 else 0.0
    print(
        f"[{split_name}] {index}/{total} ({index / total:.1%}) "
        f"elapsed={format_duration(elapsed)} eta={format_duration(eta)} "
        f"speed={rate:.2f} samples/s",
        flush=True,
    )


def evaluate_visibility_only(
    pipeline: PalmistryPipeline,
    image_path: Path,
    *,
    temperature: float,
    top_p: float,
    ) -> dict[str, Any]:
    try:
        gate_policy = pipeline.assess_gate_policy(
            image_path,
            temperature=min(temperature, 0.2),
            top_p=top_p,
        )
        visibility_assessment = gate_policy.to_visibility_assessment()
        visibility_cautious = bool(
            visibility_assessment and pipeline._visibility_is_cautious(visibility_assessment)
        )
        visibility_retake = bool(
            visibility_assessment and pipeline._visibility_requires_retake(visibility_assessment)
        )
        caution_message = ""
        if visibility_retake:
            reason = str(visibility_assessment.get("依据", "")).strip()
            caution_message = (
                "这张照片的掌纹可见性不足，继续生成完整手相报告容易出现过度解读。"
                f"{reason if reason else '建议重新拍摄：自然光、掌心完整入镜、避免遮挡和强反光。'}"
                " 手相仅供参考，请以生活实际为准。"
            )
        elif visibility_cautious:
            caution_message = (
                "这张照片目前只适合做保守掌纹观察，不适合直接输出强结论的完整报告。"
                "如需更稳定结果，建议在自然光下重拍一张更清晰的掌心照片。"
                " 手相仅供参考，请以生活实际为准。"
            )
        return {
            "pred_low_confidence": visibility_retake or visibility_cautious,
            "pred_gate_decision": gate_policy.decision,
            "gate_source": str(visibility_assessment.get("source", "generative_gate")),
            "pred_uncertain_main_lines": len(REQUIRED_LINE_NAMES) if visibility_retake else int(visibility_cautious),
            "pred_uncertain_lines": list(REQUIRED_LINE_NAMES) if visibility_retake else ([] if not visibility_cautious else ["主要掌纹"]),
            "visibility_assessment": visibility_assessment,
            "caution_message": caution_message,
            "error": None,
            "visibility_cautious": visibility_cautious,
            "visibility_retake": visibility_retake,
            "full_report_generated": False,
        }
    except Exception as exc:
        message = pipeline._build_retake_message([], error=str(exc))
        return {
            "pred_low_confidence": True,
            "pred_gate_decision": "retake",
            "gate_source": "error_fallback",
            "pred_uncertain_main_lines": len(REQUIRED_LINE_NAMES),
            "pred_uncertain_lines": list(REQUIRED_LINE_NAMES),
            "visibility_assessment": None,
            "caution_message": message,
            "error": str(exc),
            "visibility_cautious": False,
            "visibility_retake": True,
            "full_report_generated": False,
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
    val_mode: str,
    progress_every: int,
    summary_every: int,
    sample_writer: Callable[[dict[str, Any]], None] | None = None,
    summary_writer: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    structured_summary: defaultdict[str, list[float]] = defaultdict(list)
    report_summary: defaultdict[str, list[float]] = defaultdict(list)
    counts = Counter()
    total = len(records)
    start_time = time.time()

    for index, record in enumerate(records, start=1):
        reference_json = extract_teacher_json(record)
        reference_payload = load_palmistry_payload(reference_json)
        expected_uncertain_main_lines = count_uncertain_main_lines(reference_payload)
        expected_low_confidence = expected_uncertain_main_lines > 1

        if val_mode == "gate_only":
            result_row = evaluate_visibility_only(
                pipeline,
                image_root / record["image"],
                temperature=temperature,
                top_p=top_p,
            )
            sample_row: dict[str, Any] = {
                "split": "val",
                "evaluation_mode": val_mode,
                "id": record["id"],
                "image": record["image"],
                "expected_low_confidence": expected_low_confidence,
                "expected_uncertain_main_lines": expected_uncertain_main_lines,
                **result_row,
            }
            visibility_retake = result_row["visibility_retake"]
            visibility_cautious = result_row["visibility_cautious"]
            full_report_generated = result_row["full_report_generated"]
            result_structured_json = ""
            result_report = ""
        else:
            result = pipeline.analyze_detailed(
                image_root / record["image"],
                style=style,
                max_new_tokens=report_max_new_tokens,
                structured_max_new_tokens=structured_max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            sample_row = {
                "split": "val",
                "evaluation_mode": val_mode,
                "id": record["id"],
                "image": record["image"],
                "expected_low_confidence": expected_low_confidence,
                "expected_uncertain_main_lines": expected_uncertain_main_lines,
                "pred_low_confidence": result.low_confidence,
                "pred_gate_decision": result.gate_decision,
                "gate_source": str((result.visibility_assessment or {}).get("source", "generative_gate")),
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
            visibility_cautious = bool(
                result.visibility_assessment
                and pipeline._visibility_is_cautious(result.visibility_assessment)
            )
            full_report_generated = bool(result.report) and result.report != result.caution_message
            result_structured_json = result.structured_json
            result_report = result.report

        counts["samples"] += 1
        counts["low_confidence"] += int(sample_row["pred_low_confidence"])
        counts["expected_low_confidence"] += int(expected_low_confidence)
        counts["gate_match"] += int(sample_row["pred_low_confidence"] == expected_low_confidence)
        count_gate_decision(counts, sample_row["pred_gate_decision"])
        counts["visibility_cautious"] += int(visibility_cautious)
        counts["visibility_retake"] += int(visibility_retake)
        counts["full_report_generated"] += int(full_report_generated)

        if result_structured_json:
            sample_structured = structured_metrics(result_structured_json, reference_json)
            sample_row["structured_metrics"] = sample_structured
            for key, value in sample_structured.items():
                structured_summary[key].append(value)
            counts["structured_available"] += 1

        if full_report_generated and result_report:
            sample_report = report_metrics(result_report, reference_json)
            sample_row["report_metrics"] = sample_report
            for key, value in sample_report.items():
                report_summary[key].append(value)
            counts["report_available"] += 1

        if sample_writer is not None:
            sample_writer(sample_row)

        if should_emit(index, total, progress_every):
            print_progress("val", index, total, start_time)

        if summary_writer is not None and should_emit(index, total, summary_every):
            summary_writer(summarize_val(counts, structured_summary, report_summary))

    return summarize_val(counts, structured_summary, report_summary)


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
    hard_mode: str,
    progress_every: int,
    summary_every: int,
    sample_writer: Callable[[dict[str, Any]], None] | None = None,
    summary_writer: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    counts = Counter()
    by_reason: dict[str, Counter[str]] = defaultdict(Counter)
    total = len(rows)
    start_time = time.time()

    for index, row in enumerate(rows, start=1):
        image_path = image_root / row["image"]
        if hard_mode == "gate_only":
            result_row = evaluate_visibility_only(
                pipeline,
                image_path,
                temperature=temperature,
                top_p=top_p,
            )
        else:
            result = pipeline.analyze_detailed(
                image_path,
                style=style,
                max_new_tokens=report_max_new_tokens,
                structured_max_new_tokens=structured_max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            visibility_retake = bool(
                result.visibility_assessment
                and pipeline._visibility_requires_retake(result.visibility_assessment)
            )
            visibility_cautious = bool(
                result.visibility_assessment
                and pipeline._visibility_is_cautious(result.visibility_assessment)
            )
            full_report_generated = bool(result.report) and result.report != result.caution_message
            result_row = {
                "pred_low_confidence": result.low_confidence,
                "pred_gate_decision": result.gate_decision,
                "gate_source": str((result.visibility_assessment or {}).get("source", "generative_gate")),
                "pred_uncertain_main_lines": result.uncertain_main_lines,
                "pred_uncertain_lines": result.uncertain_lines,
                "visibility_assessment": result.visibility_assessment,
                "caution_message": result.caution_message,
                "error": result.error,
                "visibility_cautious": visibility_cautious,
                "visibility_retake": visibility_retake,
                "full_report_generated": full_report_generated,
            }

        reject_reasons = [str(reason) for reason in row.get("reject_reasons", [])]
        sample_row = {
            "split": "hard_cases",
            "evaluation_mode": hard_mode,
            "id": row.get("id"),
            "image": row.get("image"),
            "reject_reasons": reject_reasons,
            "quality_bucket": row.get("quality_bucket"),
            "quality_score": row.get("quality_score"),
            **result_row,
        }

        counts["samples"] += 1
        counts["low_confidence"] += int(result_row["pred_low_confidence"])
        count_gate_decision(counts, result_row["pred_gate_decision"])
        counts["visibility_cautious"] += int(result_row["visibility_cautious"])
        counts["full_report_generated"] += int(result_row["full_report_generated"])
        counts["visibility_retake"] += int(result_row["visibility_retake"])

        for reason in reject_reasons or ["__none__"]:
            by_reason[reason]["samples"] += 1
            by_reason[reason]["low_confidence"] += int(result_row["pred_low_confidence"])
            count_gate_decision(by_reason[reason], result_row["pred_gate_decision"])
            by_reason[reason]["visibility_cautious"] += int(result_row["visibility_cautious"])
            by_reason[reason]["full_report_generated"] += int(result_row["full_report_generated"])
            by_reason[reason]["visibility_retake"] += int(result_row["visibility_retake"])

        if sample_writer is not None:
            sample_writer(sample_row)

        if should_emit(index, total, progress_every):
            print_progress(f"hard:{hard_mode}", index, total, start_time)

        if summary_writer is not None and should_emit(index, total, summary_every):
            summary_writer(summarize_hard(counts, by_reason))

    return summarize_hard(counts, by_reason)


def main() -> None:
    args = parse_args()
    image_root = Path(args.image_root).resolve()
    if not image_root.exists():
        raise SystemExit(f"Image root does not exist: {image_root}")

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl_path = Path(args.output_jsonl) if args.output_jsonl else None
    if output_jsonl_path is not None:
        output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        output_jsonl_path.write_text("", encoding="utf-8")

    pipeline = PalmistryPipeline(
        base_model_path=args.base_model,
        lora_path=args.lora_path,
        device=args.device,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        gate_classifier_path=args.gate_classifier_path,
        gate_classifier_device=args.gate_classifier_device,
    )

    base_summary: dict[str, Any] = {
        "base_model": args.base_model,
        "lora_path": args.lora_path,
        "image_root": str(image_root),
        "style": args.style,
        "hard_mode": args.hard_mode,
        "val_mode": args.val_mode,
        "gate_classifier_path": args.gate_classifier_path,
        "gate_classifier_device": args.gate_classifier_device,
        "gate_mode": "classifier" if args.gate_classifier_path else "generative",
    }

    val_summary: dict[str, Any] | None = None
    hard_summary: dict[str, Any] | None = None

    sample_writer = None
    if output_jsonl_path is not None:
        sample_writer = lambda row: append_jsonl_row(output_jsonl_path, row)

    if args.val_json:
        val_records = maybe_limit(load_json_records(Path(args.val_json)), args.val_limit)

        def val_summary_writer(current_summary: dict[str, Any]) -> None:
            write_json(
                output_path,
                build_root_summary(base_summary, val_summary=current_summary, hard_summary=hard_summary),
            )

        val_summary = evaluate_val_split(
            pipeline,
            val_records,
            image_root=image_root,
            style=args.style,
            report_max_new_tokens=args.report_max_new_tokens,
            structured_max_new_tokens=args.structured_max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            val_mode=args.val_mode,
            progress_every=args.progress_every,
            summary_every=args.summary_every,
            sample_writer=sample_writer,
            summary_writer=val_summary_writer,
        )

    if args.hard_manifest:
        hard_rows = maybe_limit(load_jsonl_records(Path(args.hard_manifest)), args.hard_limit)

        def hard_summary_writer(current_summary: dict[str, Any]) -> None:
            write_json(
                output_path,
                build_root_summary(base_summary, val_summary=val_summary, hard_summary=current_summary),
            )

        hard_summary = evaluate_hard_cases(
            pipeline,
            hard_rows,
            image_root=image_root,
            style=args.style,
            report_max_new_tokens=args.report_max_new_tokens,
            structured_max_new_tokens=args.structured_max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            hard_mode=args.hard_mode,
            progress_every=args.progress_every,
            summary_every=args.summary_every,
            sample_writer=sample_writer,
            summary_writer=hard_summary_writer,
        )

    summary = build_root_summary(base_summary, val_summary=val_summary, hard_summary=hard_summary)
    write_json(output_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
