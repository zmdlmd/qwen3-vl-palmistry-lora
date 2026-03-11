from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze fixed val / hard-case evaluation subsets from an existing evaluation samples.jsonl file.",
    )
    parser.add_argument("--sample-jsonl", required=True, help="Evaluation samples.jsonl used as the source of selected ids")
    parser.add_argument("--val-json", required=True, help="Source val JSON file")
    parser.add_argument("--hard-manifest", required=True, help="Source hard-case JSONL manifest")
    parser.add_argument("--output-val-json", required=True, help="Output fixed val JSON path")
    parser.add_argument("--output-hard-jsonl", required=True, help="Output fixed hard-case JSONL path")
    parser.add_argument("--output-summary-json", required=True, help="Output summary JSON path")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def main() -> None:
    args = parse_args()

    sample_rows = load_jsonl(Path(args.sample_jsonl))
    val_rows = load_json(Path(args.val_json))
    hard_rows = load_jsonl(Path(args.hard_manifest))

    if not isinstance(val_rows, list):
        raise SystemExit("--val-json must contain a JSON list.")

    selected_val_ids = ordered_unique(
        [str(row["id"]) for row in sample_rows if row.get("split") == "val" and row.get("id")]
    )
    selected_hard_ids = ordered_unique(
        [str(row["id"]) for row in sample_rows if row.get("split") == "hard_cases" and row.get("id")]
    )

    val_index = {str(row["id"]): row for row in val_rows if isinstance(row, dict) and row.get("id")}
    hard_index = {str(row["id"]): row for row in hard_rows if isinstance(row, dict) and row.get("id")}

    missing_val_ids = [row_id for row_id in selected_val_ids if row_id not in val_index]
    missing_hard_ids = [row_id for row_id in selected_hard_ids if row_id not in hard_index]
    if missing_val_ids or missing_hard_ids:
        raise SystemExit(
            json.dumps(
                {
                    "missing_val_ids": missing_val_ids[:10],
                    "missing_hard_ids": missing_hard_ids[:10],
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    fixed_val_rows = [val_index[row_id] for row_id in selected_val_ids]
    fixed_hard_rows = [hard_index[row_id] for row_id in selected_hard_ids]

    output_val_path = Path(args.output_val_json)
    output_hard_path = Path(args.output_hard_jsonl)
    output_summary_path = Path(args.output_summary_json)

    output_val_path.parent.mkdir(parents=True, exist_ok=True)
    output_hard_path.parent.mkdir(parents=True, exist_ok=True)
    output_summary_path.parent.mkdir(parents=True, exist_ok=True)

    output_val_path.write_text(
        json.dumps(fixed_val_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with output_hard_path.open("w", encoding="utf-8") as handle:
        for row in fixed_hard_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "sample_jsonl": str(Path(args.sample_jsonl).resolve()),
        "val_json": str(Path(args.val_json).resolve()),
        "hard_manifest": str(Path(args.hard_manifest).resolve()),
        "output_val_json": str(output_val_path.resolve()),
        "output_hard_jsonl": str(output_hard_path.resolve()),
        "fixed_val_count": len(fixed_val_rows),
        "fixed_hard_count": len(fixed_hard_rows),
        "fixed_val_ids_preview": selected_val_ids[:5],
        "fixed_hard_ids_preview": selected_hard_ids[:5],
    }
    output_summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
