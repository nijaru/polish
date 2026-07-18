#!/usr/bin/env python3
"""Create a tracked dataset manifest from an ignored JSONL record file."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from polish.data_contract import record_digest
except ModuleNotFoundError:  # pragma: no cover - convenient source checkout use
    sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
    from polish.data_contract import record_digest


SPLITS = ("train", "validation", "holdout")
ALLOWED_SOURCE_KINDS = (
    "authored",
    "permissive-license",
    "public-domain",
    "user-provided",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--record-schema", type=Path, required=True)
    return parser


def _load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            raise ValueError(f"blank line in records at line {line_number}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"record at line {line_number} is not an object")
        records.append(value)
    if not records:
        raise ValueError("records file is empty")
    return records


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        records = _load_records(args.records)
        sources: dict[str, dict[str, Any]] = {}
        refs: list[dict[str, Any]] = []
        counts = Counter()
        seen_ids: set[str] = set()
        seen_digests: set[str] = set()
        for line_number, record in enumerate(records, 1):
            record_id = record.get("record_id")
            split = record.get("split")
            source = record.get("source")
            task = record.get("task")
            if not (
                isinstance(record_id, str)
                and record_id
                and isinstance(split, str)
                and split
                and isinstance(task, str)
                and task
            ):
                raise ValueError(f"record {line_number} is missing identity metadata")
            if split not in SPLITS:
                raise ValueError(f"record {record_id} has invalid split {split!r}")
            if not isinstance(source, dict) or not isinstance(source.get("id"), str):
                raise ValueError(f"record {record_id} has invalid source metadata")
            source_id = source["id"]
            if source_id in sources and sources[source_id] != source:
                raise ValueError(f"source {source_id!r} changes across records")
            sources[source_id] = source
            digest = record_digest(record)
            if record_id in seen_ids:
                raise ValueError(f"duplicate record_id {record_id!r}")
            if digest in seen_digests:
                raise ValueError(f"duplicate record content at {record_id!r}")
            seen_ids.add(record_id)
            seen_digests.add(digest)
            counts[split] += 1
            refs.append(
                {
                    "record_id": record_id,
                    "split": split,
                    "task": task,
                    "source_id": source_id,
                    "line_number": line_number,
                    "sha256": digest,
                }
            )
        manifest_dir = args.output.parent.resolve()
        manifest = {
            "schema_version": "0.1",
            "dataset_id": args.dataset_id,
            "revision": args.revision,
            "record_schema": os.path.relpath(
                args.record_schema.resolve(), manifest_dir
            ),
            "data_root": os.path.relpath(args.data_root.resolve(), manifest_dir),
            "raw_records": args.records.name,
            "source_policy": {
                "raw_content_policy": "local-only",
                "allowed_source_kinds": list(ALLOWED_SOURCE_KINDS),
            },
            "sources": list(sources.values()),
            "preprocessing": [
                (
                    "Combined the existing project-authored fixture with six "
                    "excerpts each from three pinned Standard Ebooks source "
                    "repositories."
                ),
                (
                    "Public-domain excerpts were selected as complete paragraphs "
                    "and normalized only for whitespace and invisible word-joiner "
                    "characters."
                ),
                (
                    "Briefs and task metadata were authored locally; source prose "
                    "was not paraphrased or mixed with model output."
                ),
            ],
            "deduplication": [
                (
                    "Canonical content hashes ignore record identity, split "
                    "assignment, and provenance."
                ),
                "Duplicate content is rejected across the entire dataset.",
            ],
            "split_policy": {
                "train": {
                    "purpose": "adapter fitting candidates only",
                    "allowed_for_training": True,
                    "used_for_model_selection": False,
                    "sealed": False,
                },
                "validation": {
                    "purpose": "model and hyperparameter selection",
                    "allowed_for_training": False,
                    "used_for_model_selection": True,
                    "sealed": False,
                },
                "holdout": {
                    "purpose": "final comparison only",
                    "allowed_for_training": False,
                    "used_for_model_selection": False,
                    "sealed": True,
                },
            },
            "counts": {split: counts[split] for split in SPLITS},
            "records": refs,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        _parser().error(str(error))
    print(f"wrote dataset manifest {args.output} ({len(records)} records)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
