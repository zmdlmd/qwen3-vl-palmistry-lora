import argparse
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a clean LoRA adapter directory by removing incompatible base-layer weights."
    )
    parser.add_argument("--src", required=True, help="Source checkpoint/adpater directory.")
    parser.add_argument("--dst", required=True, help="Destination clean adapter directory.")
    parser.add_argument(
        "--drop-substring",
        action="append",
        default=[".base_layer.weight"],
        help="Substring pattern to drop from adapter weight keys. Can be provided multiple times.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination if it already exists.",
    )
    return parser.parse_args()


def should_drop(key: str, patterns: list[str]) -> bool:
    return any(pattern in key for pattern in patterns)


def main():
    args = parse_args()
    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()

    if not src.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {src}")

    if dst.exists():
        if not args.overwrite:
            raise FileExistsError(f"Destination already exists: {dst}")
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    adapter_path = src / "adapter_model.safetensors"
    if not adapter_path.is_file():
        raise FileNotFoundError(f"adapter_model.safetensors not found in: {src}")

    weights = load_file(str(adapter_path), device="cpu")
    filtered = {key: value for key, value in weights.items() if not should_drop(key, args.drop_substring)}
    save_file(filtered, str(dst / "adapter_model.safetensors"))

    for name in ("adapter_config.json", "README.md"):
        src_file = src / name
        if src_file.is_file():
            shutil.copy2(src_file, dst / name)

    removed = len(weights) - len(filtered)
    print(f"Exported clean adapter to {dst}")
    print(f"Original keys: {len(weights)}")
    print(f"Removed keys: {removed}")
    print(f"Remaining keys: {len(filtered)}")


if __name__ == "__main__":
    main()
