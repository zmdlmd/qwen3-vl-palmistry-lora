from __future__ import annotations

import re
from typing import Any

from .schema import OPTIONAL_LINE_NAMES, REQUIRED_LINE_FIELDS, REQUIRED_LINE_NAMES, REQUIRED_REPORT_FIELDS, load_palmistry_payload


SECTION_KEYWORDS = (
    ("整体印象",),
    ("生命线",),
    ("智慧线",),
    ("感情线",),
    ("事业线", "事业线与发展节奏"),
    ("整体能量", "近期运势", "整体能量与近期运势"),
    ("现实建议", "温和提醒", "现实建议与温和提醒"),
    ("总结祝福", "总结", "祝福"),
)
SAFE_TERMS = ("手相仅供参考", "仅供参考", "请以生活实际为准", "请以现实实际为准", "真正重要的是当下选择和行动")
BANNED_TERMS = ("保证", "绝对", "包治", "治愈", "确诊", "注定", "百分之百")
UNCERTAINTY_TERMS = ("难以判断", "图像不够清晰", "可见信息有限", "看不清", "信息有限", "无法准确判断", "手掌图像模糊")
CHINESE_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        content = value.get("content", "")
        if isinstance(content, list):
            chunks: list[str] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text")
                    if isinstance(text, str) and text.strip():
                        chunks.append(text.strip())
            return "\n".join(chunks)
        return _as_text(content)
    if isinstance(value, list):
        chunks = [_as_text(item) for item in value]
        return "\n".join(chunk for chunk in chunks if chunk)
    return ""


def _safe_parse_payload(text: str) -> dict[str, Any] | None:
    try:
        return load_palmistry_payload(_as_text(text))
    except Exception:
        return None


def _strip_report(text: str) -> str:
    return re.sub(r"\s+", "", _as_text(text))


def _char_ngram_set(text: str, n: int = 2) -> set[str]:
    compact = _strip_report(text)
    if len(compact) < n:
        return {compact} if compact else set()
    return {compact[index : index + n] for index in range(len(compact) - n + 1)}


def _report_char_count(text: str) -> int:
    return len(CHINESE_CHAR_RE.findall(_as_text(text)))


def _reference_text(payload: dict[str, Any]) -> str:
    analysis = payload.get("palmistry_analysis", {})
    lines = analysis.get("lines", {})
    chunks: list[str] = []

    for line_name in REQUIRED_LINE_NAMES + OPTIONAL_LINE_NAMES:
        line_payload = lines.get(line_name)
        if not isinstance(line_payload, dict):
            continue
        chunks.append(line_name)
        for field_name in REQUIRED_LINE_FIELDS:
            value = line_payload.get(field_name)
            if isinstance(value, str) and value.strip():
                chunks.append(value.strip())

    for field_name in REQUIRED_REPORT_FIELDS:
        value = analysis.get(field_name)
        if isinstance(value, str) and value.strip():
            chunks.append(value.strip())

    return "\n".join(chunks)


def _ordered_section_score(text: str) -> float:
    text = _as_text(text)
    last_pos = -1
    hits = 0

    for keyword_group in SECTION_KEYWORDS:
        positions = []
        for keyword in keyword_group:
            match = re.search(re.escape(keyword), text)
            if match:
                positions.append(match.start())

        if not positions:
            continue

        current_pos = min(positions)
        if current_pos > last_pos:
            hits += 1
            last_pos = current_pos

    return hits / len(SECTION_KEYWORDS)


def _required_line_mention_score(text: str) -> float:
    text = _as_text(text)
    hits = sum(1 for line_name in REQUIRED_LINE_NAMES if line_name in text)
    return hits / len(REQUIRED_LINE_NAMES)


def _reference_uncertainty_score(payload: dict[str, Any]) -> float:
    reference_text = _reference_text(payload)
    if not reference_text:
        return 0.0

    uncertainty_hits = sum(reference_text.count(term) for term in UNCERTAINTY_TERMS)
    return min(1.0, uncertainty_hits / 6.0)


def report_format_reward(completions, **kwargs):
    rewards = []
    for completion in completions:
        completion = _as_text(completion)
        if "{" in completion or "}" in completion or "```" in completion:
            rewards.append(0.0)
            continue

        char_count = _report_char_count(completion)
        if char_count < 120:
            rewards.append(char_count / 120.0)
            continue

        rewards.append(1.0 if char_count <= 1800 else 0.6)
    return rewards


def section_structure_reward(completions, **kwargs):
    rewards = []
    for completion in completions:
        completion = _as_text(completion)
        rewards.append(0.6 * _ordered_section_score(completion) + 0.4 * _required_line_mention_score(completion))
    return rewards


def reference_alignment_reward(completions, assistant, **kwargs):
    rewards = []
    for completion, reference in zip(completions, assistant):
        completion = _as_text(completion)
        ref_payload = _safe_parse_payload(reference)
        if ref_payload is None:
            rewards.append(0.0)
            continue

        pred_ngrams = _char_ngram_set(completion)
        ref_ngrams = _char_ngram_set(_reference_text(ref_payload))
        if not pred_ngrams or not ref_ngrams:
            rewards.append(0.0)
            continue

        overlap = len(pred_ngrams & ref_ngrams) / len(pred_ngrams | ref_ngrams)
        structure = 0.5 * _ordered_section_score(completion) + 0.5 * _required_line_mention_score(completion)
        rewards.append(0.7 * overlap + 0.3 * structure)
    return rewards


def uncertainty_honesty_reward(completions, assistant, **kwargs):
    rewards = []
    for completion, reference in zip(completions, assistant):
        completion = _as_text(completion)
        ref_payload = _safe_parse_payload(reference)
        if ref_payload is None:
            rewards.append(0.0)
            continue

        expected_uncertainty = _reference_uncertainty_score(ref_payload)
        mentions_uncertainty = any(term in completion for term in UNCERTAINTY_TERMS)

        if expected_uncertainty >= 0.5:
            rewards.append(1.0 if mentions_uncertainty else 0.0)
        else:
            rewards.append(0.7 if not mentions_uncertainty else 0.4)
    return rewards


def safety_language_reward(completions, **kwargs):
    rewards = []
    for completion in completions:
        completion = _as_text(completion)
        if any(term in completion for term in BANNED_TERMS):
            rewards.append(0.0)
            continue

        rewards.append(1.0 if any(term in completion for term in SAFE_TERMS) else 0.2)
    return rewards
