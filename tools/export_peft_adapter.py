from __future__ import annotations

import argparse
from pathlib import Path

from peft import PeftModel
from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration

from src.palmistry.config import resolve_torch_dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a clean PEFT adapter from a training output directory.")
    parser.add_argument("--base-model", required=True, help="Base Qwen3-VL model path")
    parser.add_argument("--checkpoint-root", required=True, help="Training output root or checkpoint root")
    parser.add_argument("--output-dir", required=True, help="Directory to save the exported adapter")
    parser.add_argument("--torch-dtype", default="bf16", help="bf16 | fp16 | fp32")
    return parser.parse_args()


def find_adapter_source(root: Path) -> Path:
    if (root / "adapter_config.json").exists():
        return root

    candidates = sorted({path.parent for path in root.rglob("adapter_config.json")})
    if not candidates:
        raise FileNotFoundError(f"No adapter_config.json found under {root}")

    checkpoint_candidates = [path for path in candidates if path.name.startswith("checkpoint-")]
    if checkpoint_candidates:
        return checkpoint_candidates[-1]
    return candidates[-1]


def main() -> None:
    args = parse_args()
    checkpoint_root = Path(args.checkpoint_root)
    output_dir = Path(args.output_dir)
    adapter_source = find_adapter_source(checkpoint_root)

    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=resolve_torch_dtype(args.torch_dtype, device="cpu"),
        device_map="cpu",
    )
    model = PeftModel.from_pretrained(
        base_model,
        str(adapter_source),
        local_files_only=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)

    print(f"adapter_source={adapter_source}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
