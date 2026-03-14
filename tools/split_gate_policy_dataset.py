from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.palmistry.gate_policy import GATE_DECISIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a gate-policy JSONL dataset into stratified train/val splits."
    )
    parser.add_argument("--input-jsonl", required=True, help="Input gate-policy JSONL path")
    parser.add_argument("--train-jsonl", required=True, help="Output train JSONL path")
    parser.add_argument("--val-jsonl", required=True, help="Output val JSONL path")
    parser.add_argument("--summary-json", required=True, help="Output split summary JSON path")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio per class")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    rows = load_jsonl(Path(args.input_jsonl))

    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        label = str(row["gate_decision"])
        if label not in GATE_DECISIONS:
            raise ValueError(f"Unsupported gate_decision: {label}")
        by_label[label].append(row)

    train_rows: list[dict] = []
    val_rows: list[dict] = []
    label_counts: dict[str, dict[str, int]] = {}

    for label in GATE_DECISIONS:
        label_rows = by_label[label]
        rng.shuffle(label_rows)
        if not label_rows:
            label_counts[label] = {"train": 0, "val": 0}
            continue
        val_count = max(1, int(round(len(label_rows) * args.val_ratio)))
        if val_count >= len(label_rows):
            val_count = max(1, len(label_rows) - 1)
        label_val = label_rows[:val_count]
        label_train = label_rows[val_count:]
        train_rows.extend(label_train)
        val_rows.extend(label_val)
        label_counts[label] = {"train": len(label_train), "val": len(label_val)}

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)

    train_path = Path(args.train_jsonl)
    val_path = Path(args.val_jsonl)
    summary_path = Path(args.summary_json)

    dump_jsonl(train_path, train_rows)
    dump_jsonl(val_path, val_rows)

    summary = {
        "input_jsonl": args.input_jsonl,
        "train_jsonl": str(train_path),
        "val_jsonl": str(val_path),
        "summary_json": str(summary_path),
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_label_counts": dict(Counter(row["gate_decision"] for row in train_rows)),
        "val_label_counts": dict(Counter(row["gate_decision"] for row in val_rows)),
        "label_counts": label_counts,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
