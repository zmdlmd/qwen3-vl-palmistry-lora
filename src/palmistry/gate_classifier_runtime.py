from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0, resnet18

from .gate_policy import (
    GATE_DECISION_CAUTIOUS,
    GATE_DECISION_CONTINUE,
    GATE_DECISION_RETAKE,
    GatePolicyDecision,
    gate_decision_label,
    normalize_gate_decision,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_DECISION_METADATA = {
    GATE_DECISION_CONTINUE: {
        "overall_clarity": "清晰",
        "visible_main_lines": 4,
        "occlusion_noise": "无",
        "rationale": "独立 gate classifier 判断当前手掌图像质量足以进入完整分析路径。",
    },
    GATE_DECISION_CAUTIOUS: {
        "overall_clarity": "一般",
        "visible_main_lines": 3,
        "occlusion_noise": "轻微",
        "rationale": "独立 gate classifier 判断当前图像仅适合保守掌纹观察，建议限制强结论展开。",
    },
    GATE_DECISION_RETAKE: {
        "overall_clarity": "模糊",
        "visible_main_lines": 1,
        "occlusion_noise": "明显",
        "rationale": "独立 gate classifier 判断当前图像质量不足，建议重新拍摄后再分析。",
    },
}


def _build_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _build_model(model_name: str, num_labels: int) -> torch.nn.Module:
    if model_name == "resnet18":
        model = resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_labels)
        return model
    if model_name == "efficientnet_b0":
        model = efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, num_labels)
        return model
    raise ValueError(f"Unsupported gate classifier backbone: {model_name}")


def _load_image(image: str | Path | Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, Path):
        return Image.open(image).convert("RGB")
    if isinstance(image, str):
        return Image.open(image).convert("RGB")
    raise TypeError(f"Unsupported image type for gate classifier: {type(image)!r}")


@dataclass(frozen=True)
class GateClassifierPrediction:
    decision: str
    confidence: float
    probabilities: dict[str, float]


class StandaloneGateClassifier:
    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "cpu",
    ) -> None:
        self.checkpoint_path = str(checkpoint_path)
        self.device = device
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        self.model_name = str(checkpoint.get("model_name", "resnet18"))
        self.image_size = int(checkpoint.get("image_size", 224))
        self.label_to_id = {
            normalize_gate_decision(label): int(idx)
            for label, idx in checkpoint["label_to_id"].items()
        }
        self.id_to_label = {
            int(idx): normalize_gate_decision(label)
            for idx, label in checkpoint["id_to_label"].items()
        }
        self.model = _build_model(self.model_name, num_labels=len(self.label_to_id))
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.transform = _build_eval_transform(self.image_size)

    def predict(self, image: str | Path | Any) -> GateClassifierPrediction:
        pil_image = _load_image(image)
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=-1)[0].detach().cpu()
        pred_idx = int(torch.argmax(probs).item())
        decision = normalize_gate_decision(self.id_to_label[pred_idx])
        probabilities = {
            self.id_to_label[int(idx)]: float(prob)
            for idx, prob in enumerate(probs.tolist())
        }
        return GateClassifierPrediction(
            decision=decision,
            confidence=float(probabilities[decision]),
            probabilities=probabilities,
        )

    def predict_decision(self, image: str | Path | Any) -> GatePolicyDecision:
        prediction = self.predict(image)
        metadata = dict(_DECISION_METADATA[prediction.decision])
        raw_payload = {
            "source": "standalone_gate_classifier",
            "checkpoint_path": self.checkpoint_path,
            "backbone": self.model_name,
            "decision": prediction.decision,
            "decision_label": gate_decision_label(prediction.decision),
            "整体清晰度": metadata["overall_clarity"],
            "主线可辨识数量": metadata["visible_main_lines"],
            "遮挡或噪点": metadata["occlusion_noise"],
            "依据": metadata["rationale"],
            "classifier_confidence": round(prediction.confidence, 4),
            "classifier_probabilities": {
                label: round(score, 4) for label, score in prediction.probabilities.items()
            },
        }
        return GatePolicyDecision(
            decision=prediction.decision,
            overall_clarity=metadata["overall_clarity"],
            visible_main_lines=metadata["visible_main_lines"],
            occlusion_noise=metadata["occlusion_noise"],
            rationale=metadata["rationale"],
            raw_payload=raw_payload,
        )
