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
UNCERTAINTY_TERMS = (
    "难以判断",
    "图像不够清晰",
    "可见信息有限",
    "看不清",
    "信息有限",
    "无法准确判断",
    "手掌图像模糊",
    "模糊",
    "不清晰",
    "噪点",
    "遮挡",
)
CAUTIOUS_TERMS = UNCERTAINTY_TERMS + (
    "谨慎分析",
    "保守观察",
    "仅作保守观察",
    "建议结合更清晰照片",
    "建议在自然光下重拍",
)
RETAKE_TERMS = (
    "建议重拍",
    "重新拍摄",
    "重新拍一张",
    "重拍一张",
    "更清晰的掌心照片",
)
DETAIL_TERMS = (
    "岛纹",
    "断裂",
    "分叉",
    "双重",
    "二股",
    "三股",
    "锁链",
    "链状",
    "羽毛状",
    "十字",
    "佛眼",
    "神秘十字",
    "三奇纹",
)
ASSERTIVE_PATTERNS = (
    re.compile(r"(明显|清晰|分明).{0,4}(断裂|分叉|岛纹|双重|二股|三股|十字|贯穿)"),
    re.compile(r"(深|较深|很深).{0,2}(长|较长|很长)"),
    re.compile(r"(线条|走势).{0,4}(清晰|深刻|明显)"),
    re.compile(r"(贯穿掌心|清晰可见|深刻有力|深长有力)"),
)
LINE_SECTION_KEYWORDS = {
    "生命线": ("生命线",),
    "智慧线": ("智慧线",),
    "感情线": ("感情线",),
    "事业线": ("事业线", "事业线与发展节奏"),
}
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


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


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


def _line_reference_text(payload: dict[str, Any], line_name: str) -> str:
    analysis = payload.get("palmistry_analysis", {})
    lines = analysis.get("lines", {})
    line_payload = lines.get(line_name)
    if not isinstance(line_payload, dict):
        return ""

    chunks = [line_name]
    for field_name in REQUIRED_LINE_FIELDS:
        value = line_payload.get(field_name)
        if isinstance(value, str) and value.strip():
            chunks.append(value.strip())
    return "\n".join(chunks)


def _is_uncertain_reference_line(payload: dict[str, Any], line_name: str) -> bool:
    return _contains_any(_line_reference_text(payload, line_name), UNCERTAINTY_TERMS)


def _first_keyword_pos(text: str, keywords: tuple[str, ...]) -> int | None:
    positions = []
    for keyword in keywords:
        match = re.search(re.escape(keyword), text)
        if match:
            positions.append(match.start())
    return min(positions) if positions else None


def _extract_line_sections(text: str) -> dict[str, str]:
    text = _as_text(text)
    starts: list[tuple[str, int]] = []
    for line_name, keywords in LINE_SECTION_KEYWORDS.items():
        pos = _first_keyword_pos(text, keywords)
        if pos is not None:
            starts.append((line_name, pos))

    starts.sort(key=lambda item: item[1])
    sections: dict[str, str] = {}
    for index, (line_name, start) in enumerate(starts):
        end = starts[index + 1][1] if index + 1 < len(starts) else len(text)
        sections[line_name] = text[start:end].strip()
    return sections


def _estimate_reference_gate(payload: dict[str, Any]) -> str:
    uncertain_required = sum(1 for line_name in REQUIRED_LINE_NAMES if _is_uncertain_reference_line(payload, line_name))
    uncertain_optional = sum(1 for line_name in OPTIONAL_LINE_NAMES if _is_uncertain_reference_line(payload, line_name))
    uncertain_total = uncertain_required + uncertain_optional

    if uncertain_required >= 3 or uncertain_total >= 4:
        return "retake"
    if uncertain_required >= 1 or uncertain_total >= 2:
        return "cautious"
    return "continue"


def _estimate_report_gate(text: str) -> str:
    text = _as_text(text)
    if _contains_any(text, RETAKE_TERMS):
        return "retake"
    if _contains_any(text, CAUTIOUS_TERMS):
        return "cautious"
    return "continue"


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


def _unsupported_detail_count(section_text: str, reference_text: str) -> int:
    return sum(1 for term in DETAIL_TERMS if term in section_text and term not in reference_text)


def _assertive_signal_count(section_text: str) -> int:
    return sum(len(pattern.findall(section_text)) for pattern in ASSERTIVE_PATTERNS)


def _line_uncertainty_honesty_score(section_text: str, reference_text: str) -> float:
    reference_uncertain = _contains_any(reference_text, UNCERTAINTY_TERMS)
    cautious = _contains_any(section_text, CAUTIOUS_TERMS)
    retake = _contains_any(section_text, RETAKE_TERMS)
    unsupported_details = _unsupported_detail_count(section_text, reference_text)
    assertive_signals = _assertive_signal_count(section_text)

    if reference_uncertain:
        score = 1.0 if cautious or retake else 0.0
        if unsupported_details:
            score -= min(0.6, unsupported_details * 0.2)
        if assertive_signals:
            score -= min(0.6, assertive_signals * 0.25)
        return max(0.0, min(1.0, score))

    if not section_text:
        return 0.0

    if cautious or retake:
        return 0.55
    return 0.9


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


def line_level_consistency_reward(completions, assistant, **kwargs):
    rewards = []
    for completion, reference in zip(completions, assistant):
        completion = _as_text(completion)
        ref_payload = _safe_parse_payload(reference)
        if ref_payload is None:
            rewards.append(0.0)
            continue

        line_sections = _extract_line_sections(completion)
        line_scores = []
        for line_name in REQUIRED_LINE_NAMES:
            section_text = line_sections.get(line_name, "")
            reference_text = _line_reference_text(ref_payload, line_name)
            if not section_text or not reference_text:
                line_scores.append(0.0)
                continue

            if _contains_any(reference_text, UNCERTAINTY_TERMS):
                line_scores.append(1.0 if _contains_any(section_text, CAUTIOUS_TERMS) else 0.0)
                continue

            pred_ngrams = _char_ngram_set(section_text)
            ref_ngrams = _char_ngram_set(reference_text)
            if not pred_ngrams or not ref_ngrams:
                line_scores.append(0.0)
                continue

            overlap = len(pred_ngrams & ref_ngrams) / len(pred_ngrams | ref_ngrams)
            line_scores.append(min(1.0, 0.2 + 0.8 * overlap))

        rewards.append(sum(line_scores) / len(line_scores) if line_scores else 0.0)
    return rewards


def hallucination_penalty_reward(completions, assistant, **kwargs):
    rewards = []
    for completion, reference in zip(completions, assistant):
        completion = _as_text(completion)
        ref_payload = _safe_parse_payload(reference)
        if ref_payload is None:
            rewards.append(0.0)
            continue

        line_sections = _extract_line_sections(completion)
        checks = 0
        penalties = 0.0

        for line_name in REQUIRED_LINE_NAMES:
            section_text = line_sections.get(line_name, "")
            reference_text = _line_reference_text(ref_payload, line_name)
            if not section_text or not reference_text:
                continue

            checks += 1
            if _contains_any(reference_text, UNCERTAINTY_TERMS) and not _contains_any(section_text, CAUTIOUS_TERMS):
                penalties += 1.0

            unsupported_details = sum(
                1 for term in DETAIL_TERMS if term in section_text and term not in reference_text
            )
            penalties += min(1.0, unsupported_details * 0.25)

        if checks == 0:
            rewards.append(0.0)
            continue

        rewards.append(max(0.0, 1.0 - penalties / checks))
    return rewards


def gate_decision_reward(completions, assistant, **kwargs):
    reward_matrix = {
        "continue": {"continue": 1.0, "cautious": 0.6, "retake": 0.0},
        "cautious": {"continue": 0.2, "cautious": 1.0, "retake": 0.6},
        "retake": {"continue": 0.0, "cautious": 0.8, "retake": 1.0},
    }

    rewards = []
    for completion, reference in zip(completions, assistant):
        ref_payload = _safe_parse_payload(reference)
        if ref_payload is None:
            rewards.append(0.0)
            continue

        expected_gate = _estimate_reference_gate(ref_payload)
        predicted_gate = _estimate_report_gate(completion)
        reward = reward_matrix[expected_gate][predicted_gate]

        if expected_gate == "retake" and _report_char_count(completion) > 600 and predicted_gate == "continue":
            reward = 0.0
        rewards.append(reward)
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
        line_sections = _extract_line_sections(completion)

        line_scores = []
        for line_name in REQUIRED_LINE_NAMES:
            reference_text = _line_reference_text(ref_payload, line_name)
            if not reference_text:
                continue
            section_text = line_sections.get(line_name, "")
            line_scores.append(_line_uncertainty_honesty_score(section_text, reference_text))

        if line_scores:
            line_score = sum(line_scores) / len(line_scores)
        else:
            if expected_uncertainty >= 0.5:
                line_score = 1.0 if mentions_uncertainty else 0.0
            else:
                line_score = 0.7 if not mentions_uncertainty else 0.4

        predicted_gate = _estimate_report_gate(completion)
        expected_gate = _estimate_reference_gate(ref_payload)
        gate_bonus = 0.0
        if expected_gate == "retake" and predicted_gate == "retake":
            gate_bonus = 0.1
        elif expected_gate == "cautious" and predicted_gate in {"cautious", "retake"}:
            gate_bonus = 0.1
        elif expected_gate == "continue" and predicted_gate == "continue":
            gate_bonus = 0.05

        rewards.append(min(1.0, line_score + gate_bonus))
    return rewards


def uncertainty_contradiction_penalty_reward(completions, assistant, **kwargs):
    rewards = []
    for completion, reference in zip(completions, assistant):
        completion = _as_text(completion)
        ref_payload = _safe_parse_payload(reference)
        if ref_payload is None:
            rewards.append(0.0)
            continue

        line_sections = _extract_line_sections(completion)
        checks = 0
        penalties = 0.0

        for line_name in REQUIRED_LINE_NAMES:
            section_text = line_sections.get(line_name, "")
            reference_text = _line_reference_text(ref_payload, line_name)
            if not section_text or not reference_text:
                continue

            checks += 1
            cautious = _contains_any(section_text, CAUTIOUS_TERMS)
            assertive_signals = _assertive_signal_count(section_text)
            unsupported_details = _unsupported_detail_count(section_text, reference_text)
            reference_uncertain = _contains_any(reference_text, UNCERTAINTY_TERMS)

            if cautious and (assertive_signals > 0 or unsupported_details > 0):
                penalties += 1.0

            if reference_uncertain and (assertive_signals > 0 or unsupported_details > 0):
                penalties += min(1.0, 0.5 + unsupported_details * 0.2 + assertive_signals * 0.2)

        if checks == 0:
            rewards.append(0.0)
            continue

        rewards.append(max(0.0, 1.0 - penalties / checks))
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
