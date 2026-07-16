import json
from pathlib import Path

import pytest

from polish.cli import main
from polish.data_contract import (
    DataContractError,
    record_digest,
    validate_data_manifest,
)

SOURCE = {
    "id": "fixture-source",
    "kind": "authored",
    "origin": "local://fixture-source",
    "license": "CC BY 4.0",
    "rights_note": "Project-authored fixture for research and SFT with attribution.",
    "acquired_at": "2026-07-16",
}


def _record(record_id: str, split: str, target: str) -> dict[str, object]:
    return {
        "record_id": record_id,
        "split": split,
        "source": SOURCE,
        "task": "generation",
        "input": {"brief": f"Write a short passage about {record_id}."},
        "target": target,
        "metadata": {"genre": "profile"},
    }


def _write_fixture(tmp_path: Path) -> tuple[Path, list[dict[str, object]]]:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    records = [
        _record("one", "train", "A careful passage."),
        _record("two", "validation", "A measured passage."),
        _record("three", "holdout", "A specific passage."),
    ]
    (raw_root / "records.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    (tmp_path / "training-record.schema.json").write_text("{}", encoding="utf-8")
    manifest = {
        "schema_version": "0.1",
        "dataset_id": "fixture",
        "revision": "fixture-v0",
        "record_schema": "training-record.schema.json",
        "data_root": "raw",
        "raw_records": "records.jsonl",
        "source_policy": {
            "raw_content_policy": "local-only",
            "allowed_source_kinds": [
                "authored",
                "permissive-license",
                "public-domain",
                "user-provided",
            ],
        },
        "sources": [SOURCE],
        "preprocessing": ["authored directly"],
        "deduplication": ["content hashes checked across all splits"],
        "split_policy": {
            "train": {
                "purpose": "adapter fitting",
                "allowed_for_training": True,
                "used_for_model_selection": False,
                "sealed": False,
            },
            "validation": {
                "purpose": "model selection",
                "allowed_for_training": False,
                "used_for_model_selection": True,
                "sealed": False,
            },
            "holdout": {
                "purpose": "final comparison",
                "allowed_for_training": False,
                "used_for_model_selection": False,
                "sealed": True,
            },
        },
        "counts": {"train": 1, "validation": 1, "holdout": 1},
        "records": [
            {
                "record_id": record["record_id"],
                "split": record["split"],
                "task": record["task"],
                "source_id": SOURCE["id"],
                "line_number": line_number,
                "sha256": record_digest(record),
            }
            for line_number, record in enumerate(records, start=1)
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, records


def test_validate_data_manifest_checks_hashes_and_splits(tmp_path: Path) -> None:
    manifest_path, _ = _write_fixture(tmp_path)

    summary = validate_data_manifest(manifest_path)

    assert summary.record_count == 3
    assert summary.source_count == 1
    assert summary.split_counts == {"train": 1, "validation": 1, "holdout": 1}


def test_validate_data_manifest_rejects_duplicate_content(tmp_path: Path) -> None:
    manifest_path, records = _write_fixture(tmp_path)
    duplicate = dict(records[0])
    duplicate["record_id"] = "duplicate"
    duplicate["split"] = "holdout"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["records"][2] = {
        **data["records"][2],
        "record_id": "duplicate",
        "split": "holdout",
        "sha256": record_digest(duplicate),
    }
    manifest_path.write_text(json.dumps(data), encoding="utf-8")
    (tmp_path / "raw" / "records.jsonl").write_text(
        "".join(
            json.dumps(record) + "\n" for record in [records[0], records[1], duplicate]
        ),
        encoding="utf-8",
    )

    with pytest.raises(DataContractError, match="duplicate record content"):
        validate_data_manifest(manifest_path)


def test_validate_data_manifest_rejects_trainable_holdout(tmp_path: Path) -> None:
    manifest_path, _ = _write_fixture(tmp_path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["split_policy"]["holdout"]["allowed_for_training"] = True
    manifest_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(DataContractError, match="split_policy.holdout"):
        validate_data_manifest(manifest_path)


def test_validate_data_command_reports_success(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest_path, _ = _write_fixture(tmp_path)

    assert main(["validate-data", "--manifest", str(manifest_path)]) == 0
    assert (
        "validated data manifest fixture@fixture-v0 (3 records)"
        in capsys.readouterr().out
    )


def test_record_digest_ignores_record_identity_and_split() -> None:
    first = _record("one", "train", "Same content.")
    second = _record("two", "holdout", "Same content.")
    second["input"] = first["input"]

    assert record_digest(first) == record_digest(second)
