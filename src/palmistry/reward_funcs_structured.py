from __future__ import annotations

import re
from typing import Any

from .schema import (
    REQUIRED_LINE_FIELDS,
    REQUIRED_LINE_NAMES,
    REQUIRED_REPORT_FIELDS,
    flatten_palmistry_text,
    load_palmistry_payload,
    validate_palmistry_payload,
)


def _safe_parse_payload(text: str) -> dict[str, Any] | None:
    try:
        return load_palmistry_payload(text)
    except Exception:
        return None


def _line_field_coverage(payload: dict[str, Any]) -> float:
    analysis = payload.get("palmistry_analysis", {})
    lines = analysis.get("lines", {})
    hits = 0
    total = len(REQUIRED_LINE_NAMES) * len(REQUIRED_LINE_FIELDS)
    for line_name in REQUIRED_LINE_NAMES:
        line_payload = lines.get(line_name, {})
        for field_name in REQUIRED_LINE_FIELDS:
            value = line_payload.get(field_name)
            if isinstance(value, str) and value.strip():
                hits += 1
    return hits / total if total else 0.0


def _report_field_coverage(payload: dict[str, Any]) -> float:
    analysis = payload.get("palmistry_analysis", {})
    hits = 0
    total = len(REQUIRED_REPORT_FIELDS)
    for field_name in REQUIRED_REPORT_FIELDS:
        value = analysis.get(field_name)
        if isinstance(value, str) and value.strip():
            hits += 1
    return hits / total if total else 0.0


def _char_ngram_set(text: str, n: int = 2) -> set[str]:
    compact = re.sub(r"\s+", "", text)
    if len(compact) < n:
        return {compact} if compact else set()
    return {compact[index : index + n] for index in range(len(compact) - n + 1)}


def json_schema_reward(completions, **kwargs):
    rewards = []
    for completion in completions:
        payload = _safe_parse_payload(completion)
        rewards.append(1.0 if payload is not None and not validate_palmistry_payload(payload) else 0.0)
    return rewards


def line_field_coverage_reward(completions, **kwargs):
    rewards = []
    for completion in completions:
        payload = _safe_parse_payload(completion)
        rewards.append(_line_field_coverage(payload) if payload is not None else 0.0)
    return rewards


def report_field_coverage_reward(completions, **kwargs):
    rewards = []
    for completion in completions:
        payload = _safe_parse_payload(completion)
        rewards.append(_report_field_coverage(payload) if payload is not None else 0.0)
    return rewards


def reference_alignment_reward(completions, assistant, **kwargs):
    rewards = []
    for completion, reference in zip(completions, assistant):
        pred_payload = _safe_parse_payload(completion)
        ref_payload = _safe_parse_payload(reference)
        if pred_payload is None or ref_payload is None:
            rewards.append(0.0)
            continue

        pred_text = flatten_palmistry_text(pred_payload)
        ref_text = flatten_palmistry_text(ref_payload)
        pred_ngrams = _char_ngram_set(pred_text)
        ref_ngrams = _char_ngram_set(ref_text)
        if not pred_ngrams or not ref_ngrams:
            rewards.append(0.0)
            continue

        overlap = len(pred_ngrams & ref_ngrams) / len(pred_ngrams | ref_ngrams)
        coverage = 0.5 * _line_field_coverage(pred_payload) + 0.5 * _report_field_coverage(pred_payload)
        rewards.append(0.6 * overlap + 0.4 * coverage)
    return rewards


def safety_language_reward(completions, **kwargs):
    safe_terms = ("仅供参考", "请以生活实际为准", "请以现实实际为准", "不构成诊断", "不可替代医学")
    banned_terms = ("保证", "绝对", "确诊", "治愈", "包治")

    rewards = []
    for completion in completions:
        payload = _safe_parse_payload(completion)
        if payload is None:
            rewards.append(0.0)
            continue

        analysis = payload["palmistry_analysis"]
        report_text = "\n".join(
            [
                analysis.get("medical_report", ""),
                analysis.get("blessing", ""),
            ]
        )

        if any(term in report_text for term in banned_terms):
            rewards.append(0.0)
            continue

        rewards.append(1.0 if any(term in report_text for term in safe_terms) else 0.2)
    return rewards
