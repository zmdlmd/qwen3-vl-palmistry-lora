from __future__ import annotations

import os

import torch


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_torch_dtype(dtype_name: str = "auto", device: str | None = None) -> torch.dtype:
    device = device or default_device()
    normalized = dtype_name.lower()

    if normalized == "auto":
        if device.startswith("cuda"):
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32

    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return mapping[normalized]


def env_or_default(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value
