from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from .schema import extract_json_object


GATE_DECISION_CONTINUE = "continue"
GATE_DECISION_CAUTIOUS = "cautious"
GATE_DECISION_RETAKE = "retake"
GATE_DECISIONS = (
    GATE_DECISION_CONTINUE,
    GATE_DECISION_CAUTIOUS,
    GATE_DECISION_RETAKE,
)
GATE_DECISION_LABELS = {
    GATE_DECISION_CONTINUE: "继续分析",
    GATE_DECISION_CAUTIOUS: "谨慎分析",
    GATE_DECISION_RETAKE: "建议重拍",
}


def normalize_gate_decision(value: Any) -> str:
    text = str(value).strip().lower()
    mapping = {
        "continue": GATE_DECISION_CONTINUE,
        "继续分析": GATE_DECISION_CONTINUE,
        "cautious": GATE_DECISION_CAUTIOUS,
        "谨慎分析": GATE_DECISION_CAUTIOUS,
        "retake": GATE_DECISION_RETAKE,
        "建议重拍": GATE_DECISION_RETAKE,
    }
    if text in mapping:
        return mapping[text]
    return GATE_DECISION_CAUTIOUS


def gate_decision_label(decision: str) -> str:
    return GATE_DECISION_LABELS[normalize_gate_decision(decision)]


@dataclass(frozen=True)
class GatePolicyDecision:
    decision: str
    overall_clarity: str
    visible_main_lines: int
    occlusion_noise: str
    rationale: str
    raw_payload: dict[str, Any]

    @property
    def decision_label(self) -> str:
        return gate_decision_label(self.decision)

    @property
    def low_confidence(self) -> bool:
        return normalize_gate_decision(self.decision) != GATE_DECISION_CONTINUE

    def to_visibility_assessment(self) -> dict[str, Any]:
        assessment = dict(self.raw_payload)
        assessment["decision"] = normalize_gate_decision(self.decision)
        assessment["decision_label"] = self.decision_label
        assessment["建议"] = self.decision_label
        assessment["整体清晰度"] = self.overall_clarity
        assessment["主线可辨识数量"] = self.visible_main_lines
        assessment["遮挡或噪点"] = self.occlusion_noise
        assessment["依据"] = self.rationale
        return assessment


def _coerce_visible_main_lines(value: Any) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return 0


def parse_gate_policy_payload(text: str) -> GatePolicyDecision:
    payload = json.loads(extract_json_object(text))
    if not isinstance(payload, dict):
        raise ValueError("Gate policy payload must be a JSON object.")

    gate_payload = payload.get("gate_policy")
    if not isinstance(gate_payload, dict):
        gate_payload = payload.get("visibility_assessment")
    if not isinstance(gate_payload, dict):
        raise ValueError("Missing gate_policy object.")

    decision = normalize_gate_decision(
        gate_payload.get("decision") or gate_payload.get("decision_label") or gate_payload.get("建议")
    )
    overall_clarity = str(gate_payload.get("整体清晰度", "")).strip()
    occlusion_noise = str(gate_payload.get("遮挡或噪点", "")).strip()
    rationale = str(gate_payload.get("依据", "")).strip()
    visible_main_lines = _coerce_visible_main_lines(gate_payload.get("主线可辨识数量"))

    if not overall_clarity:
        raise ValueError("Missing gate field: 整体清晰度")
    if not occlusion_noise:
        raise ValueError("Missing gate field: 遮挡或噪点")
    if not rationale:
        raise ValueError("Missing gate field: 依据")

    normalized_payload = {
        "decision": decision,
        "decision_label": gate_decision_label(decision),
        "整体清晰度": overall_clarity,
        "主线可辨识数量": visible_main_lines,
        "遮挡或噪点": occlusion_noise,
        "依据": rationale,
    }
    return GatePolicyDecision(
        decision=decision,
        overall_clarity=overall_clarity,
        visible_main_lines=visible_main_lines,
        occlusion_noise=occlusion_noise,
        rationale=rationale,
        raw_payload=normalized_payload,
    )
