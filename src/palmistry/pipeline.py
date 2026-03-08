from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration

from .config import default_device, env_or_default, resolve_torch_dtype
from .prompts import build_followup_prompt, build_report_prompt


class PalmistryPipeline:
    def __init__(
        self,
        base_model_path: str,
        lora_path: str | None = None,
        *,
        device: str | None = None,
        device_map: str = "auto",
        torch_dtype: str = "auto",
    ) -> None:
        self.base_model_path = base_model_path
        self.lora_path = lora_path
        self.device = device or default_device()
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            self.device = "cpu"
        self.device_map = device_map
        self.dtype = resolve_torch_dtype(torch_dtype, device=self.device)

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

    def analyze(
        self,
        image: str | Path | Any,
        *,
        style: str = "balanced",
        max_new_tokens: int = 1200,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        if isinstance(image, Path):
            image = str(image)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": build_report_prompt(style)},
                ],
            }
        ]
        return self._generate(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

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
