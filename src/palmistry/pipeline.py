from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration

from .config import default_device, env_or_default, resolve_torch_dtype
from .gate_classifier_runtime import StandaloneGateClassifier
from .gate_policy import (
    GATE_DECISION_CAUTIOUS,
    GATE_DECISION_CONTINUE,
    GATE_DECISION_RETAKE,
    GatePolicyDecision,
    parse_gate_policy_payload,
)
from .prompts import (
    DEFAULT_STUDENT_STRUCTURED_PROMPT,
    build_visibility_guard_prompt,
    build_followup_prompt,
    build_structured_report_prompt,
)
from .schema import (
    REQUIRED_LINE_NAMES,
    canonicalize_palmistry_json,
    load_palmistry_payload,
    validate_palmistry_payload,
)
from .teacher import sanitize_palmistry_payload


_UNCERTAINTY_MARKERS = (
    "难以判断",
    "无法",
    "不可见",
    "不可辨",
    "模糊",
    "不清晰",
    "噪点",
    "遮挡",
    "低",
    "极低",
    "较低",
    "不足",
    "缺失",
)


@dataclass
class PalmistryAnalysisResult:
    structured_json: str
    report: str
    low_confidence: bool
    uncertain_main_lines: int
    gate_decision: str = GATE_DECISION_RETAKE
    uncertain_lines: list[str] = field(default_factory=list)
    caution_message: str = ""
    gate_policy: dict[str, Any] | None = None
    structured_payload: dict[str, Any] | None = None
    visibility_assessment: dict[str, Any] | None = None
    sanitation_issues: list[str] = field(default_factory=list)
    error: str | None = None


class PalmistryPipeline:
    def __init__(
        self,
        base_model_path: str,
        lora_path: str | None = None,
        *,
        device: str | None = None,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        gate_classifier_path: str | None = None,
        gate_classifier_device: str | None = None,
    ) -> None:
        self.base_model_path = base_model_path
        self.lora_path = lora_path
        self.device = device or default_device()
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            self.device = "cpu"
        self.device_map = device_map
        self.dtype = resolve_torch_dtype(torch_dtype, device=self.device)
        self.gate_classifier_path = gate_classifier_path
        self.gate_classifier_device = gate_classifier_device or self.device
        self.gate_classifier: StandaloneGateClassifier | None = None
        if self.gate_classifier_path:
            self.gate_classifier = StandaloneGateClassifier(
                self.gate_classifier_path,
                device=self.gate_classifier_device,
            )

        self.processor = AutoProcessor.from_pretrained(self.base_model_path)
        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.base_model_path,
            torch_dtype=self.dtype,
            device_map=self.device_map,
        )

        if self.lora_path:
            self.model = PeftModel.from_pretrained(base_model, self.lora_path)
        else:
            self.model = base_model

        self.model.eval()

    @classmethod
    def from_env(cls) -> "PalmistryPipeline":
        base_model_path = env_or_default("BASE_MODEL_PATH")
        if not base_model_path:
            raise ValueError("BASE_MODEL_PATH is required.")
        return cls(
            base_model_path=base_model_path,
            lora_path=env_or_default("LORA_PATH"),
            device=env_or_default("DEVICE", default_device()),
            device_map=env_or_default("DEVICE_MAP", "auto") or "auto",
            torch_dtype=env_or_default("TORCH_DTYPE", "auto") or "auto",
            gate_classifier_path=env_or_default("GATE_CLASSIFIER_PATH"),
            gate_classifier_device=env_or_default("GATE_CLASSIFIER_DEVICE"),
        )

    @staticmethod
    def _string_has_uncertainty(value: Any) -> bool:
        text = str(value).strip()
        if not text:
            return True
        return any(marker in text for marker in _UNCERTAINTY_MARKERS)

    @classmethod
    def _line_is_uncertain(cls, line_payload: dict[str, Any]) -> bool:
        clarity = str(line_payload.get("清晰度", "")).strip()
        basis = str(line_payload.get("图像依据", "")).strip()
        uncertain_core_fields = sum(
            cls._string_has_uncertainty(line_payload.get(field_name))
            for field_name in ("长度", "深浅", "特殊结构", "日系分类")
        )
        if cls._string_has_uncertainty(clarity) and uncertain_core_fields >= 1:
            return True
        if cls._string_has_uncertainty(basis) and uncertain_core_fields >= 1:
            return True
        if uncertain_core_fields >= 3:
            return True
        return False

    @classmethod
    def _collect_uncertain_lines(cls, payload: dict[str, Any]) -> tuple[list[str], list[str]]:
        analysis = payload.get("palmistry_analysis", {})
        lines = analysis.get("lines", {})
        uncertain_main_lines: list[str] = []
        uncertain_optional_lines: list[str] = []
        for line_name, line_payload in lines.items():
            if not isinstance(line_payload, dict):
                continue
            if not cls._line_is_uncertain(line_payload):
                continue
            if line_name in REQUIRED_LINE_NAMES:
                uncertain_main_lines.append(line_name)
            else:
                uncertain_optional_lines.append(line_name)
        return uncertain_main_lines, uncertain_optional_lines

    @staticmethod
    def _build_retake_message(uncertain_main_lines: list[str], error: str | None = None) -> str:
        if error:
            return (
                "这张手掌照片暂时没能稳定完成结构化解析。"
                "建议重新拍摄一张更清晰的掌心照片：自然光、掌心完整入镜、避免遮挡和强反光。"
                "手相仅供参考，请以生活实际为准。"
            )

        line_text = "、".join(uncertain_main_lines) if uncertain_main_lines else "主要掌纹"
        return (
            f"当前照片里 {line_text} 的可见信息不足，继续生成完整手相报告容易出现过度解读。"
            "建议重新拍摄：自然光下掌心完整入镜、镜头对焦掌纹、避免遮挡和强反光。"
            "手相仅供参考，请以生活实际为准。"
        )

    @staticmethod
    def _build_caution_message(uncertain_lines: list[str]) -> str:
        if not uncertain_lines:
            return ""
        line_text = "、".join(uncertain_lines)
        return f"以下线条可见信息有限：{line_text}。报告已按保守方式处理，不确定处不会强行展开。"

    @staticmethod
    def _normalize_image(image: str | Path | Any) -> str | Any:
        if isinstance(image, Path):
            return str(image)
        return image

    def _structured_messages(self, image: str | Path | Any) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": self._normalize_image(image)},
                    {"type": "text", "text": DEFAULT_STUDENT_STRUCTURED_PROMPT},
                ],
            }
        ]

    def _visibility_messages(self, image: str | Path | Any) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": self._normalize_image(image)},
                    {"type": "text", "text": build_visibility_guard_prompt()},
                ],
            }
        ]

    @staticmethod
    def _extract_gate_decision(assessment: dict[str, Any]) -> str:
        return str(assessment.get("decision", "")).strip() or str(assessment.get("建议", "")).strip()

    @staticmethod
    def _visible_line_count(assessment: dict[str, Any]) -> int:
        visible_lines_raw = str(assessment.get("主线可辨识数量", "")).strip()
        try:
            return int(visible_lines_raw)
        except ValueError:
            return 0

    @classmethod
    def _visibility_is_cautious(cls, assessment: dict[str, Any]) -> bool:
        suggestion = cls._extract_gate_decision(assessment)
        clarity = str(assessment.get("整体清晰度", "")).strip()
        occlusion = str(assessment.get("遮挡或噪点", "")).strip()
        visible_lines = cls._visible_line_count(assessment)

        if suggestion in {GATE_DECISION_CAUTIOUS, "谨慎分析"}:
            return True
        if clarity == "一般":
            return True
        if occlusion == "轻微":
            return True
        if visible_lines in (2, 3):
            return True
        return False

    @classmethod
    def _visibility_requires_retake(cls, assessment: dict[str, Any]) -> bool:
        suggestion = cls._extract_gate_decision(assessment)
        clarity = str(assessment.get("整体清晰度", "")).strip()
        occlusion = str(assessment.get("遮挡或噪点", "")).strip()
        visible_lines = cls._visible_line_count(assessment)

        if suggestion in {GATE_DECISION_RETAKE, "建议重拍"}:
            return True
        if clarity == "模糊":
            return True
        if occlusion == "明显":
            return True
        if visible_lines < 2:
            return True
        return False

    @classmethod
    def _should_force_low_confidence(
        cls,
        assessment: dict[str, Any] | None,
        uncertain_main_line_count: int,
        uncertain_line_count: int,
    ) -> bool:
        if uncertain_main_line_count >= 2:
            return True

        if not assessment:
            return False

        visible_lines = cls._visible_line_count(assessment)
        cautious_visibility = cls._visibility_is_cautious(assessment)

        if uncertain_main_line_count == 1 and (cautious_visibility or visible_lines < 4):
            return True

        if uncertain_line_count >= 3 and cautious_visibility:
            return True
        return False

    def assess_visibility(
        self,
        image: str | Path | Any,
        *,
        max_new_tokens: int = 260,
        temperature: float = 0.1,
        top_p: float = 0.9,
    ) -> dict[str, Any]:
        return self.assess_gate_policy(
            image,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        ).to_visibility_assessment()

    def _assess_gate_policy_with_generator(
        self,
        image: str | Path | Any,
        *,
        max_new_tokens: int = 260,
        temperature: float = 0.1,
        top_p: float = 0.9,
    ) -> GatePolicyDecision:
        raw_output = self._generate(
            self._visibility_messages(image),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return parse_gate_policy_payload(raw_output)

    def assess_gate_policy(
        self,
        image: str | Path | Any,
        *,
        max_new_tokens: int = 260,
        temperature: float = 0.1,
        top_p: float = 0.9,
    ) -> GatePolicyDecision:
        if self.gate_classifier is not None:
            try:
                return self.gate_classifier.predict_decision(image)
            except Exception:
                pass
        return self._assess_gate_policy_with_generator(
            image,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    @classmethod
    def _refine_gate_decision(
        cls,
        *,
        initial_gate_decision: str,
        visibility_assessment: dict[str, Any] | None,
        uncertain_main_line_count: int,
        uncertain_line_count: int,
    ) -> str:
        if initial_gate_decision == GATE_DECISION_RETAKE:
            return GATE_DECISION_RETAKE
        if uncertain_main_line_count >= 2:
            return GATE_DECISION_CAUTIOUS
        if initial_gate_decision == GATE_DECISION_CAUTIOUS:
            return GATE_DECISION_CAUTIOUS
        if cls._should_force_low_confidence(
            visibility_assessment,
            uncertain_main_line_count,
            uncertain_line_count,
        ):
            return GATE_DECISION_CAUTIOUS
        return GATE_DECISION_CONTINUE

    @staticmethod
    def _build_gate_caution_message(uncertain_lines: list[str]) -> str:
        if uncertain_lines:
            return (
                "当前照片已进入“谨慎分析”路径：可以做保守掌纹观察，但不适合输出强结论的完整手相报告。"
                f" 可见信息有限的线条包括：{'、'.join(uncertain_lines)}。"
                "如需更稳定结果，建议在自然光下重拍更清晰的掌心照片。"
                " 手相仅供参考，请以生活实际为准。"
            )
        return (
            "当前照片已进入“谨慎分析”路径：可以做保守掌纹观察，但不适合输出强结论的完整手相报告。"
            " 如需更稳定结果，建议在自然光下重拍更清晰的掌心照片。"
            " 手相仅供参考，请以生活实际为准。"
        )

    def analyze_structured(
        self,
        image: str | Path | Any,
        *,
        max_new_tokens: int = 1400,
        temperature: float = 0.1,
        top_p: float = 0.9,
    ) -> PalmistryAnalysisResult:
        raw_output = self._generate(
            self._structured_messages(image),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        payload = load_palmistry_payload(raw_output)
        payload, sanitation_issues = sanitize_palmistry_payload(payload)
        errors = validate_palmistry_payload(payload)
        if errors:
            raise ValueError("; ".join(errors[:5]))

        canonical_json = canonicalize_palmistry_json(payload)
        structured_payload = load_palmistry_payload(canonical_json)
        uncertain_main_lines, uncertain_optional_lines = self._collect_uncertain_lines(structured_payload)
        uncertain_lines = uncertain_main_lines + uncertain_optional_lines
        low_confidence = len(uncertain_main_lines) >= 2
        caution_message = (
            self._build_retake_message(uncertain_main_lines)
            if low_confidence
            else self._build_caution_message(uncertain_lines)
        )
        return PalmistryAnalysisResult(
            structured_json=canonical_json,
            report="",
            low_confidence=low_confidence,
            uncertain_main_lines=len(uncertain_main_lines),
            uncertain_lines=uncertain_lines,
            caution_message=caution_message,
            structured_payload=structured_payload,
            sanitation_issues=sanitation_issues,
        )

    def _prepare_inputs(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }

    def _generate(
        self,
        messages: list[dict[str, Any]],
        *,
        max_new_tokens: int = 1200,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        inputs = self._prepare_inputs(messages)
        do_sample = temperature > 0
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p

        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                **generation_kwargs,
            )

        return self.processor.batch_decode(
            generated[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )[0].strip()

    def analyze_detailed(
        self,
        image: str | Path | Any,
        *,
        style: str = "balanced",
        max_new_tokens: int = 1200,
        temperature: float = 0.7,
        top_p: float = 0.9,
        structured_max_new_tokens: int = 1400,
    ) -> PalmistryAnalysisResult:
        try:
            gate_policy = self.assess_gate_policy(
                image,
                temperature=min(temperature, 0.2),
                top_p=top_p,
            )
            visibility_assessment = gate_policy.to_visibility_assessment()
        except Exception as exc:
            message = self._build_retake_message([], error=str(exc))
            return PalmistryAnalysisResult(
                structured_json="",
                structured_payload=None,
                report=message,
                low_confidence=True,
                uncertain_main_lines=len(REQUIRED_LINE_NAMES),
                gate_decision=GATE_DECISION_RETAKE,
                uncertain_lines=list(REQUIRED_LINE_NAMES),
                caution_message=message,
                gate_policy=None,
                visibility_assessment=None,
                error=str(exc),
            )

        if gate_policy.decision == GATE_DECISION_RETAKE:
            reason = str(visibility_assessment.get("依据", "")).strip()
            message = (
                "这张照片的掌纹可见性不足，继续生成完整手相报告容易出现过度解读。"
                f"{reason if reason else '建议重新拍摄：自然光、掌心完整入镜、避免遮挡和强反光。'}"
                " 手相仅供参考，请以生活实际为准。"
            )
            return PalmistryAnalysisResult(
                structured_json="",
                structured_payload=None,
                report=message,
                low_confidence=True,
                uncertain_main_lines=len(REQUIRED_LINE_NAMES),
                gate_decision=GATE_DECISION_RETAKE,
                uncertain_lines=list(REQUIRED_LINE_NAMES),
                caution_message=message,
                gate_policy=visibility_assessment,
                visibility_assessment=visibility_assessment,
            )

        try:
            result = self.analyze_structured(
                image,
                max_new_tokens=structured_max_new_tokens,
                temperature=min(temperature, 0.2),
                top_p=top_p,
            )
        except Exception as exc:
            message = self._build_retake_message([], error=str(exc))
            return PalmistryAnalysisResult(
                structured_json="",
                structured_payload=None,
                report=message,
                low_confidence=True,
                uncertain_main_lines=len(REQUIRED_LINE_NAMES),
                gate_decision=GATE_DECISION_RETAKE,
                uncertain_lines=list(REQUIRED_LINE_NAMES),
                caution_message=message,
                gate_policy=visibility_assessment,
                visibility_assessment=visibility_assessment,
                error=str(exc),
            )

        result.gate_policy = visibility_assessment
        result.visibility_assessment = visibility_assessment
        result.gate_decision = self._refine_gate_decision(
            initial_gate_decision=gate_policy.decision,
            visibility_assessment=visibility_assessment,
            uncertain_main_line_count=result.uncertain_main_lines,
            uncertain_line_count=len(result.uncertain_lines),
        )
        result.low_confidence = result.gate_decision != GATE_DECISION_CONTINUE

        if result.gate_decision == GATE_DECISION_CAUTIOUS:
            result.caution_message = self._build_gate_caution_message(result.uncertain_lines)
            result.report = result.caution_message
            return result

        if result.gate_decision == GATE_DECISION_RETAKE or result.low_confidence:
            if not result.caution_message:
                uncertain_main_lines, _ = self._collect_uncertain_lines(result.structured_payload or {})
                result.caution_message = self._build_retake_message(
                    uncertain_main_lines or list(REQUIRED_LINE_NAMES[:1]),
                )
            result.report = result.caution_message
            return result

        prompt = build_structured_report_prompt(
            result.structured_json,
            style=style,
            caution_hint=result.caution_message or None,
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        result.report = self._generate(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return result

    def analyze(
        self,
        image: str | Path | Any,
        *,
        style: str = "balanced",
        max_new_tokens: int = 1200,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        result = self.analyze_detailed(
            image,
            style=style,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return result.report

    def answer_followup(
        self,
        last_report: str,
        user_question: str,
        *,
        max_new_tokens: int = 600,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": build_followup_prompt(last_report, user_question),
                    }
                ],
            }
        ]
        return self._generate(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
