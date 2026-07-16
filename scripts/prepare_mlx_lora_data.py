#!/usr/bin/env python3
"""Export validated Polish records to the MLX-LM LoRA format."""

import argparse
import json
from pathlib import Path
from typing import Any

from polish.data_contract import DataContractError, validate_data_manifest

_SPLIT_FILES = {"train": "train.jsonl", "validation": "valid.jsonl"}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export train and validation records for MLX-LM LoRA."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _prompt(record: dict[str, Any]) -> str:
    input_data = record["input"]
    sections = [f"Task: {record['task']}", input_data["brief"]]
    if input_data.get("draft"):
        sections.extend(["Draft:", input_data["draft"]])
    constraints = input_data.get("constraints", [])
    if constraints:
        sections.append(
            "Constraints:\n" + "\n".join(f"- {item}" for item in constraints)
        )
    return "\n\n".join(sections)


def _load_records(manifest_path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_root = (manifest_path.parent / manifest["data_root"]).resolve()
    raw_path = data_root / manifest["raw_records"]
    return [
        json.loads(line)
        for line in raw_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    args = _parser().parse_args()
    try:
        summary = validate_data_manifest(args.manifest)
    except (DataContractError, OSError, json.JSONDecodeError) as error:
        _parser().error(str(error))

    if args.output.exists() and any(args.output.iterdir()):
        _parser().error(f"output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    records = _load_records(args.manifest.resolve())
    exported: dict[str, int] = {}
    for split, filename in _SPLIT_FILES.items():
        rows = [
            {"prompt": _prompt(record), "completion": record["target"]}
            for record in records
            if record["split"] == split
        ]
        path = args.output / filename
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        exported[split] = len(rows)

    if exported["train"] == 0 or exported["validation"] == 0:
        _parser().error("train and validation exports must both be non-empty")
    print(
        f"exported {summary.dataset_id}@{summary.revision}: "
        f"{exported['train']} train and {exported['validation']} validation records"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
