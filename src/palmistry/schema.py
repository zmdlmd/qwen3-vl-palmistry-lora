from __future__ import annotations

import json
import re
from typing import Any


TOP_LEVEL_KEY = "palmistry_analysis"
REQUIRED_LINE_NAMES = ("生命线", "智慧线", "感情线", "事业线")
OPTIONAL_LINE_NAMES = ("婚姻线",)
REQUIRED_LINE_FIELDS = (
    "位置判断",
    "清晰度",
    "长度",
    "深浅",
    "特殊结构",
    "日系分类",
    "图像依据",
)
REQUIRED_REPORT_FIELDS = (
    "traditional_report",
    "japanese_report",
    "mystic_energy_report",
    "medical_report",
    "blessing",
)

_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json_object(text: str) -> str:
    candidate = text.strip()
    code_block = _CODE_BLOCK_RE.search(candidate)
    if code_block:
        candidate = code_block.group(1).strip()

    if candidate.startswith("{") and candidate.endswith("}"):
        return candidate

    start = candidate.find("{")
    if start == -1:
        raise ValueError("No JSON object found in response.")

    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(candidate)):
        char = candidate[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return candidate[start : index + 1]

    raise ValueError("Could not extract a balanced JSON object from response.")


def load_palmistry_payload(text: str) -> dict[str, Any]:
    extracted = extract_json_object(text)
    payload = json.loads(extracted)
    if not isinstance(payload, dict):
        raise ValueError("Palmistry payload must be a JSON object.")
    return payload


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_palmistry_payload(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    analysis = payload.get(TOP_LEVEL_KEY)
    if not isinstance(analysis, dict):
        return [f"Missing object: {TOP_LEVEL_KEY}"]

    lines = analysis.get("lines")
    if not isinstance(lines, dict):
        errors.append("Missing object: palmistry_analysis.lines")
        return errors

    for line_name in REQUIRED_LINE_NAMES:
        line_payload = lines.get(line_name)
        if not isinstance(line_payload, dict):
            errors.append(f"Missing line: {line_name}")
            continue
        for field_name in REQUIRED_LINE_FIELDS:
            if not _is_non_empty_string(line_payload.get(field_name)):
                errors.append(f"Missing or empty field: {line_name}.{field_name}")

    for line_name in OPTIONAL_LINE_NAMES:
        if line_name not in lines:
            continue
        line_payload = lines.get(line_name)
        if not isinstance(line_payload, dict):
            errors.append(f"Invalid optional line object: {line_name}")
            continue
        for field_name in REQUIRED_LINE_FIELDS:
            if not _is_non_empty_string(line_payload.get(field_name)):
                errors.append(f"Missing or empty field: {line_name}.{field_name}")

    for field_name in REQUIRED_REPORT_FIELDS:
        if not _is_non_empty_string(analysis.get(field_name)):
            errors.append(f"Missing or empty field: palmistry_analysis.{field_name}")

    return errors


def normalize_palmistry_payload(payload: dict[str, Any]) -> dict[str, Any]:
    analysis = payload[TOP_LEVEL_KEY]
    lines = analysis["lines"]

    normalized_lines: dict[str, Any] = {}
    for line_name in REQUIRED_LINE_NAMES + OPTIONAL_LINE_NAMES:
        if line_name not in lines:
            continue
        line_payload = lines[line_name]
        normalized_lines[line_name] = {
            field_name: str(line_payload.get(field_name, "")).strip()
            for field_name in REQUIRED_LINE_FIELDS
        }

    normalized_analysis: dict[str, Any] = {"lines": normalized_lines}
    for field_name in REQUIRED_REPORT_FIELDS:
        normalized_analysis[field_name] = str(analysis.get(field_name, "")).strip()

    return {TOP_LEVEL_KEY: normalized_analysis}


def canonicalize_palmistry_json(text_or_payload: str | dict[str, Any]) -> str:
    payload = load_palmistry_payload(text_or_payload) if isinstance(text_or_payload, str) else text_or_payload
    normalized = normalize_palmistry_payload(payload)
    return json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))


def build_llava_sft_record(
    record_id: str,
    image: str,
    assistant_text: str,
    *,
    student_prompt: str,
) -> dict[str, Any]:
    return {
        "id": record_id,
        "image": image,
        "conversations": [
            {
                "from": "human",
                "value": f"<image>\n{student_prompt}",
            },
            {
                "from": "gpt",
                "value": assistant_text,
            },
        ],
    }


def flatten_palmistry_text(payload: dict[str, Any]) -> str:
    chunks: list[str] = []

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)
        elif isinstance(node, str):
            stripped = node.strip()
            if stripped:
                chunks.append(stripped)

    visit(payload)
    return "\n".join(chunks)
