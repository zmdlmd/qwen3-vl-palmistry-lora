from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from .prompts import DEFAULT_STUDENT_STRUCTURED_PROMPT, build_teacher_structured_prompt
from .schema import (
    OPTIONAL_LINE_NAMES,
    REQUIRED_LINE_FIELDS,
    REQUIRED_LINE_NAMES,
    REQUIRED_REPORT_FIELDS,
    TOP_LEVEL_KEY,
    build_llava_sft_record,
    canonicalize_palmistry_json,
    load_palmistry_payload,
    validate_palmistry_payload,
)


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
POSITIVE_BASIS_PATTERNS = (
    re.compile(r"可见"),
    re.compile(r"观察到"),
    re.compile(r"清晰"),
    re.compile(r"明显"),
    re.compile(r"连续"),
    re.compile(r"起始点"),
    re.compile(r"主体"),
    re.compile(r"走向"),
    re.compile(r"延伸"),
    re.compile(r"弧形"),
    re.compile(r"横向"),
    re.compile(r"纵向"),
    re.compile(r"主纹路"),
)
HARD_UNCERTAINTY_PATTERNS = (
    re.compile(r"输入图片未提供"),
    re.compile(r"图像未提供"),
    re.compile(r"未观察到明显"),
    re.compile(r"未见明显"),
    re.compile(r"无法"),
    re.compile(r"不可见"),
    re.compile(r"不可辨识"),
    re.compile(r"无法辨认"),
    re.compile(r"难以判断"),
    re.compile(r"难以准确判断"),
    re.compile(r"无法清晰"),
    re.compile(r"细节缺失"),
    re.compile(r"完全不可见"),
)
REPORT_UNCERTAINTY_PATTERNS = (
    re.compile(r"图像质量"),
    re.compile(r"难以"),
    re.compile(r"无法"),
    re.compile(r"模糊"),
    re.compile(r"不可辨识"),
    re.compile(r"不清晰"),
)
_THREAD_LOCAL = threading.local()


def is_remote_image(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://") or value.startswith("data:image")


def default_num_workers() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, min(8, cpu_count))


@dataclass
class TeacherGenerationConfig:
    api_base: str
    api_key: str
    model: str
    output_json: Path
    output_jsonl: Path | None = None
    manifest_path: Path | None = None
    image_dir: Path | None = None
    teacher_prompt: str = build_teacher_structured_prompt()
    student_prompt: str = DEFAULT_STUDENT_STRUCTURED_PROMPT
    temperature: float = 0.2
    max_tokens: int = 2200
    max_retries: int = 3
    request_timeout: int = 180
    sleep_seconds: float = 0.0
    json_mode: bool = False
    limit: int | None = None
    resume: bool = True
    filter_low_quality: bool = True
    max_uncertain_main_lines: int = 1
    num_workers: int = default_num_workers()


@dataclass(frozen=True)
class TeacherTaskContext:
    index: int
    record_id: str
    image_value: str
    student_prompt: str
    teacher_prompt: str
    image_path: Path | str


@dataclass
class TeacherTaskResult:
    index: int
    llava_record: dict[str, Any] | None = None
    log_records: list[dict[str, Any]] = field(default_factory=list)
    generated: int = 0
    failed: int = 0
    filtered: int = 0


def _load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    text = manifest_path.read_text(encoding="utf-8")
    if manifest_path.suffix.lower() == ".json":
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError("Manifest JSON must be a list.")
        return payload

    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def _scan_image_dir(image_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for image_path in sorted(image_dir.rglob("*")):
        if image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            continue
        relative_path = image_path.relative_to(image_dir).as_posix()
        record_id = relative_path.replace("/", "_").rsplit(".", 1)[0]
        records.append({"id": record_id, "image": relative_path})
    return records


def load_teacher_records(manifest_path: Path | None, image_dir: Path | None, limit: int | None = None) -> list[dict[str, Any]]:
    if manifest_path is None and image_dir is None:
        raise ValueError("Either manifest_path or image_dir must be provided.")

    if manifest_path is not None:
        records = _load_manifest(manifest_path)
    else:
        records = _scan_image_dir(image_dir.resolve())

    if limit is not None:
        return records[:limit]
    return records


def resolve_image_path(record: dict[str, Any], *, manifest_path: Path | None, image_dir: Path | None) -> Path | str:
    image_value = record["image"]
    if isinstance(image_value, str) and is_remote_image(image_value):
        return image_value

    image_path = Path(image_value)
    if image_path.is_absolute():
        return image_path

    if image_dir is not None:
        return (image_dir / image_path).resolve()
    if manifest_path is not None:
        return (manifest_path.parent / image_path).resolve()
    return image_path.resolve()


def encode_image_as_data_url(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(image_path.name)
    mime_type = mime_type or "application/octet-stream"
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def extract_message_text(response_payload: dict[str, Any]) -> str:
    choices = response_payload.get("choices")
    if not choices:
        raise ValueError("Teacher API response has no choices.")

    message = choices[0].get("message", {})
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"]
        combined = "\n".join(text.strip() for text in texts if text.strip())
        if combined:
            return combined

    raise ValueError("Teacher API response did not contain text content.")


def build_image_url_payload(image_path: Path | str) -> str:
    if isinstance(image_path, str):
        return image_path
    return encode_image_as_data_url(image_path)


def _get_requests_session() -> requests.Session:
    session = getattr(_THREAD_LOCAL, "requests_session", None)
    if session is None:
        session = requests.Session()
        _THREAD_LOCAL.requests_session = session
    return session


def call_openai_compatible_api(
    *,
    api_base: str,
    api_key: str,
    model: str,
    prompt: str,
    image_path: Path | str,
    temperature: float,
    max_tokens: int,
    request_timeout: int,
    json_mode: bool,
) -> str:
    url = api_base.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": build_image_url_payload(image_path)}},
                ],
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    response = _get_requests_session().post(url, headers=headers, json=payload, timeout=request_timeout)
    response.raise_for_status()
    response_payload = response.json()
    return extract_message_text(response_payload)


def _load_success_map(jsonl_path: Path) -> dict[str, dict[str, Any]]:
    success_map: dict[str, dict[str, Any]] = {}
    if not jsonl_path.exists():
        return success_map

    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("status") == "ok" and record.get("id"):
            success_map[record["id"]] = record
    return success_map


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _sleep_if_configured(config: TeacherGenerationConfig) -> None:
    if config.sleep_seconds > 0:
        time.sleep(config.sleep_seconds)


def _matches_any(text: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def _is_unknown_field(value: Any) -> bool:
    text = str(value).strip()
    return not text or "难以判断" in text


def _build_unknown_line_payload(reason: str) -> dict[str, str]:
    payload = {field_name: "难以判断" for field_name in REQUIRED_LINE_FIELDS}
    payload["图像依据"] = reason or "难以判断"
    return payload


def _is_line_unknown(line_payload: dict[str, Any]) -> bool:
    return all(_is_unknown_field(line_payload.get(field_name)) for field_name in REQUIRED_LINE_FIELDS if field_name != "图像依据")


def sanitize_palmistry_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    sanitized = deepcopy(payload)
    issues: list[str] = []

    analysis = sanitized.get(TOP_LEVEL_KEY)
    if not isinstance(analysis, dict):
        return sanitized, issues

    lines = analysis.get("lines")
    if not isinstance(lines, dict):
        return sanitized, issues

    for line_name in REQUIRED_LINE_NAMES + OPTIONAL_LINE_NAMES:
        line_payload = lines.get(line_name)
        if not isinstance(line_payload, dict):
            continue

        image_basis = str(line_payload.get("图像依据", "")).strip()
        if not image_basis:
            continue

        has_positive_basis = _matches_any(image_basis, POSITIVE_BASIS_PATTERNS)
        has_hard_uncertainty = _matches_any(image_basis, HARD_UNCERTAINTY_PATTERNS)

        if has_hard_uncertainty and not has_positive_basis:
            lines[line_name] = _build_unknown_line_payload(image_basis)
            issues.append(f"{line_name}: downgraded_to_unknown")

    return sanitized, issues


def evaluate_palmistry_quality(payload: dict[str, Any], *, max_uncertain_main_lines: int) -> list[str]:
    issues: list[str] = []
    analysis = payload.get(TOP_LEVEL_KEY)
    if not isinstance(analysis, dict):
        return ["missing_analysis"]

    lines = analysis.get("lines")
    if not isinstance(lines, dict):
        return ["missing_lines"]

    uncertain_main_lines = 0
    for line_name in REQUIRED_LINE_NAMES:
        line_payload = lines.get(line_name)
        if not isinstance(line_payload, dict):
            uncertain_main_lines += 1
            continue
        if _is_line_unknown(line_payload):
            uncertain_main_lines += 1

    if uncertain_main_lines > max_uncertain_main_lines:
        issues.append(f"too_many_uncertain_main_lines:{uncertain_main_lines}")

    report_mentions_uncertainty = any(
        _matches_any(str(analysis.get(field_name, "")).strip(), REPORT_UNCERTAINTY_PATTERNS)
        for field_name in REQUIRED_REPORT_FIELDS
    )
    if uncertain_main_lines > 0 and not report_mentions_uncertainty:
        issues.append("uncertain_lines_without_report_uncertainty")

    return issues


def _process_teacher_record(context: TeacherTaskContext, config: TeacherGenerationConfig) -> TeacherTaskResult:
    result = TeacherTaskResult(index=context.index)

    for attempt in range(1, config.max_retries + 1):
        raw_text = ""
        try:
            raw_text = call_openai_compatible_api(
                api_base=config.api_base,
                api_key=config.api_key,
                model=config.model,
                prompt=context.teacher_prompt,
                image_path=context.image_path,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                request_timeout=config.request_timeout,
                json_mode=config.json_mode,
            )
            payload = load_palmistry_payload(raw_text)
            payload, sanitation_issues = sanitize_palmistry_payload(payload)
            errors = validate_palmistry_payload(payload)
            if errors:
                raise ValueError("; ".join(errors[:5]))

            canonical_text = canonicalize_palmistry_json(payload)
            quality_issues = (
                evaluate_palmistry_quality(payload, max_uncertain_main_lines=config.max_uncertain_main_lines)
                if config.filter_low_quality
                else []
            )
            if quality_issues:
                result.log_records.append(
                    {
                        "id": context.record_id,
                        "image": context.image_value,
                        "student_prompt": context.student_prompt,
                        "teacher_prompt": context.teacher_prompt,
                        "assistant": canonical_text,
                        "teacher_model": config.model,
                        "attempt": attempt,
                        "status": "filtered",
                        "quality_issues": sanitation_issues + quality_issues,
                    }
                )
                if attempt == config.max_retries:
                    result.filtered = 1
                    _sleep_if_configured(config)
                    return result
                _sleep_if_configured(config)
                continue

            log_record = {
                "id": context.record_id,
                "image": context.image_value,
                "student_prompt": context.student_prompt,
                "teacher_prompt": context.teacher_prompt,
                "assistant": canonical_text,
                "teacher_model": config.model,
                "status": "ok",
            }
            if sanitation_issues:
                log_record["sanitation_issues"] = sanitation_issues
            result.log_records.append(log_record)
            result.llava_record = build_llava_sft_record(
                context.record_id,
                context.image_value,
                canonical_text,
                student_prompt=context.student_prompt,
            )
            result.generated = 1
            _sleep_if_configured(config)
            return result
        except Exception as exc:
            result.log_records.append(
                {
                    "id": context.record_id,
                    "image": context.image_value,
                    "teacher_model": config.model,
                    "attempt": attempt,
                    "status": "error",
                    "error": str(exc),
                    "raw_text": raw_text[:4000],
                }
            )
            if attempt == config.max_retries:
                result.failed = 1
                _sleep_if_configured(config)
                return result
            _sleep_if_configured(config)

    return result


def generate_sft_dataset(config: TeacherGenerationConfig) -> dict[str, Any]:
    records = load_teacher_records(config.manifest_path, config.image_dir, config.limit)
    success_map = _load_success_map(config.output_jsonl) if config.resume and config.output_jsonl else {}

    llava_records: list[dict[str, Any] | None] = [None] * len(records)
    pending_contexts: list[TeacherTaskContext] = []
    generated = 0
    skipped = 0
    failed = 0
    filtered = 0

    for index, record in enumerate(records):
        record_id = str(record.get("id") or Path(record["image"]).stem)
        image_value = str(record["image"])
        student_prompt = str(record.get("student_prompt") or config.student_prompt)
        teacher_prompt = str(record.get("teacher_prompt") or config.teacher_prompt)

        if record_id in success_map:
            llava_records[index] = (
                build_llava_sft_record(
                    record_id,
                    image_value,
                    success_map[record_id]["assistant"],
                    student_prompt=student_prompt,
                )
            )
            skipped += 1
            continue

        image_path = resolve_image_path(record, manifest_path=config.manifest_path, image_dir=config.image_dir)
        pending_contexts.append(
            TeacherTaskContext(
                index=index,
                record_id=record_id,
                image_value=image_value,
                student_prompt=student_prompt,
                teacher_prompt=teacher_prompt,
                image_path=image_path,
            )
        )

    if pending_contexts:
        with ThreadPoolExecutor(max_workers=max(1, config.num_workers)) as executor:
            future_to_context: dict[Future[TeacherTaskResult], TeacherTaskContext] = {
                executor.submit(_process_teacher_record, context, config): context for context in pending_contexts
            }
            for future in as_completed(future_to_context):
                context = future_to_context[future]
                try:
                    result = future.result()
                except Exception as exc:
                    failed += 1
                    if config.output_jsonl:
                        _append_jsonl(
                            config.output_jsonl,
                            {
                                "id": context.record_id,
                                "image": context.image_value,
                                "teacher_model": config.model,
                                "status": "error",
                                "error": f"worker_crash: {exc}",
                            },
                        )
                    continue

                if config.output_jsonl:
                    for log_record in result.log_records:
                        _append_jsonl(config.output_jsonl, log_record)

                if result.llava_record is not None:
                    llava_records[result.index] = result.llava_record
                generated += result.generated
                failed += result.failed
                filtered += result.filtered

    config.output_json.parent.mkdir(parents=True, exist_ok=True)
    config.output_json.write_text(
        json.dumps([record for record in llava_records if record is not None], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "total": len(records),
        "generated": generated,
        "skipped": skipped,
        "failed": failed,
        "filtered": filtered,
        "num_workers": max(1, config.num_workers),
        "output_json": str(config.output_json),
        "output_jsonl": str(config.output_jsonl) if config.output_jsonl else None,
    }
