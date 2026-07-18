#!/usr/bin/env python3
"""Build a target-free evaluation manifest from a validated data manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from polish.data_contract import DataContractError, validate_data_manifest
except ModuleNotFoundError:  # pragma: no cover - convenient source checkout use
    sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
    from polish.data_contract import DataContractError, validate_data_manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split",
        action="append",
        choices=("validation", "holdout"),
        required=True,
    )
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


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = validate_data_manifest(args.data_manifest)
        manifest = json.loads(args.data_manifest.read_text(encoding="utf-8"))
        data_root = (args.data_manifest.parent / manifest["data_root"]).resolve()
        raw_path = data_root / manifest["raw_records"]
        records = [
            json.loads(line)
            for line in raw_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        selected = [record for record in records if record["split"] in args.split]
        if not selected:
            _parser().error("requested splits contain no records")
        cases = []
        for record in selected:
            metadata = record.get("metadata", {})
            cases.append(
                {
                    "id": record["record_id"],
                    "split": record["split"],
                    "prompt": _prompt(record),
                    "metadata": {
                        "task_type": record["task"],
                        "genre": metadata.get("genre", "unspecified"),
                        "tone": metadata.get("tone", "unspecified"),
                        "dataset_revision": summary.revision,
                    },
                }
            )
        output = {
            "schema_version": "0.1",
            "revision": f"{summary.revision}-eval",
            "cases": cases,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    except (DataContractError, OSError, ValueError, json.JSONDecodeError) as error:
        _parser().error(str(error))
    print(f"wrote evaluation manifest {args.output} ({len(cases)} cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
