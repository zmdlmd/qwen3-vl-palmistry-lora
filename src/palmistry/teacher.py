from __future__ import annotations

import base64
import json
import mimetypes
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from .prompts import DEFAULT_STUDENT_STRUCTURED_PROMPT, build_teacher_structured_prompt
from .schema import build_llava_sft_record, canonicalize_palmistry_json, load_palmistry_payload, validate_palmistry_payload


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


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


def resolve_image_path(record: dict[str, Any], *, manifest_path: Path | None, image_dir: Path | None) -> Path:
    image_value = record["image"]
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


def call_openai_compatible_api(
    *,
    api_base: str,
    api_key: str,
    model: str,
    prompt: str,
    image_path: Path,
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
                    {"type": "image_url", "image_url": {"url": encode_image_as_data_url(image_path)}},
                ],
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    response = requests.post(url, headers=headers, json=payload, timeout=request_timeout)
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


def generate_sft_dataset(config: TeacherGenerationConfig) -> dict[str, Any]:
    records = load_teacher_records(config.manifest_path, config.image_dir, config.limit)
    success_map = _load_success_map(config.output_jsonl) if config.resume and config.output_jsonl else {}

    llava_records: list[dict[str, Any]] = []
    generated = 0
    skipped = 0
    failed = 0

    for record in records:
        record_id = str(record.get("id") or Path(record["image"]).stem)
        image_value = str(record["image"])
        student_prompt = str(record.get("student_prompt") or config.student_prompt)
        teacher_prompt = str(record.get("teacher_prompt") or config.teacher_prompt)

        if record_id in success_map:
            llava_records.append(
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
        last_error = ""
        for attempt in range(1, config.max_retries + 1):
            raw_text = ""
            try:
                raw_text = call_openai_compatible_api(
                    api_base=config.api_base,
                    api_key=config.api_key,
                    model=config.model,
                    prompt=teacher_prompt,
                    image_path=image_path,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    request_timeout=config.request_timeout,
                    json_mode=config.json_mode,
                )
                payload = load_palmistry_payload(raw_text)
                errors = validate_palmistry_payload(payload)
                if errors:
                    raise ValueError("; ".join(errors[:5]))

                canonical_text = canonicalize_palmistry_json(payload)
                log_record = {
                    "id": record_id,
                    "image": image_value,
                    "student_prompt": student_prompt,
                    "teacher_prompt": teacher_prompt,
                    "assistant": canonical_text,
                    "teacher_model": config.model,
                    "status": "ok",
                }
                if config.output_jsonl:
                    _append_jsonl(config.output_jsonl, log_record)

                llava_records.append(
                    build_llava_sft_record(
                        record_id,
                        image_value,
                        canonical_text,
                        student_prompt=student_prompt,
                    )
                )
                generated += 1
                break
            except Exception as exc:
                last_error = str(exc)
                if config.output_jsonl:
                    _append_jsonl(
                        config.output_jsonl,
                        {
                            "id": record_id,
                            "image": image_value,
                            "teacher_model": config.model,
                            "attempt": attempt,
                            "status": "error",
                            "error": last_error,
                            "raw_text": raw_text[:4000],
                        },
                    )
                if attempt == config.max_retries:
                    failed += 1
                elif config.sleep_seconds > 0:
                    time.sleep(config.sleep_seconds)

        if config.sleep_seconds > 0:
            time.sleep(config.sleep_seconds)

    config.output_json.parent.mkdir(parents=True, exist_ok=True)
    config.output_json.write_text(
        json.dumps(llava_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "total": len(records),
        "generated": generated,
        "skipped": skipped,
        "failed": failed,
        "output_json": str(config.output_json),
        "output_jsonl": str(config.output_jsonl) if config.output_jsonl else None,
    }
