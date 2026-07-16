"""Validation for tracked dataset manifests and local training records."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

_SOURCE_KINDS = {
    "authored",
    "permissive-license",
    "public-domain",
    "user-provided",
}
_TASKS = {"critique", "generation", "outline", "revision"}
_RECORD_KEYS = {
    "input",
    "metadata",
    "record_id",
    "source",
    "split",
    "target",
    "task",
}
_SOURCE_KEYS = {
    "acquired_at",
    "id",
    "kind",
    "license",
    "origin",
    "rights_note",
}
_REF_KEYS = {"line_number", "record_id", "sha256", "source_id", "split", "task"}
_SPLIT_POLICY_KEYS = {
    "allowed_for_training",
    "purpose",
    "sealed",
    "used_for_model_selection",
}
_DATA_SPLITS = ("train", "validation", "holdout")
_MANIFEST_KEYS = {
    "counts",
    "data_root",
    "dataset_id",
    "deduplication",
    "preprocessing",
    "raw_records",
    "record_schema",
    "records",
    "revision",
    "schema_version",
    "source_policy",
    "sources",
    "split_policy",
}


class DataContractError(ValueError):
    """Raised when a dataset manifest or local record violates the contract."""


@dataclass(frozen=True, slots=True)
class DataManifestSummary:
    """Summary emitted after a complete data-gate validation."""

    dataset_id: str
    revision: str
    manifest_path: Path
    record_count: int
    source_count: int
    split_counts: Mapping[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "revision": self.revision,
            "manifest_path": str(self.manifest_path),
            "record_count": self.record_count,
            "source_count": self.source_count,
            "split_counts": dict(self.split_counts),
        }


def record_digest(record: Mapping[str, Any]) -> str:
    """Hash training content while ignoring identity, split, and provenance."""

    content = {
        key: value
        for key, value in record.items()
        if key not in {"record_id", "source", "split"}
    }
    canonical = json.dumps(
        content,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(canonical).hexdigest()


def validate_data_manifest(path: Path) -> DataManifestSummary:
    """Validate a tracked manifest and every referenced local JSONL record."""

    manifest_path = path.resolve()
    manifest = _load_object(manifest_path, "manifest")
    if set(manifest) != _MANIFEST_KEYS:
        raise DataContractError(
            f"manifest must contain exactly {sorted(_MANIFEST_KEYS)}"
        )
    if manifest["schema_version"] != "0.1":
        raise DataContractError("manifest schema_version must be '0.1'")

    dataset_id = _string(manifest["dataset_id"], "manifest.dataset_id")
    revision = _string(manifest["revision"], "manifest.revision")
    _validate_string_list(manifest["preprocessing"], "manifest.preprocessing")
    _validate_string_list(manifest["deduplication"], "manifest.deduplication")
    _validate_source_policy(manifest["source_policy"])
    source_map = _validate_sources(manifest["sources"])
    _validate_split_policy(manifest["split_policy"])
    declared_counts = _validate_counts(manifest["counts"])

    manifest_dir = manifest_path.parent
    schema_path = _relative_path(
        manifest_dir, manifest["record_schema"], "record_schema"
    )
    if not schema_path.is_file():
        raise DataContractError(f"record schema does not exist: {schema_path}")
    _load_object(schema_path, "record schema")

    data_root = _relative_path(manifest_dir, manifest["data_root"], "data_root")
    if not data_root.is_dir():
        raise DataContractError(f"data_root does not exist: {data_root}")
    raw_records_name = _string(manifest["raw_records"], "manifest.raw_records")
    raw_records_path = (data_root / raw_records_name).resolve()
    try:
        raw_records_path.relative_to(data_root.resolve())
    except ValueError as error:
        raise DataContractError("raw_records must stay inside data_root") from error
    if not raw_records_path.is_file():
        raise DataContractError(f"raw records do not exist: {raw_records_path}")

    refs = _validate_record_refs(manifest["records"], source_map)
    actual_counts = dict.fromkeys(_DATA_SPLITS, 0)
    seen_digests: set[str] = set()
    seen_record_ids: set[str] = set()
    seen_lines: set[int] = set()

    try:
        lines = raw_records_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise DataContractError(
            f"could not read raw records: {raw_records_path}"
        ) from error

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise DataContractError(f"blank line in raw records at line {line_number}")
        try:
            record = _load_object_from_text(line, f"raw record line {line_number}")
        except json.JSONDecodeError as error:
            raise DataContractError(
                f"invalid JSON in raw records at line {line_number}: {error.msg}"
            ) from error
        ref = refs.get(line_number)
        if ref is None:
            raise DataContractError(
                f"raw record line {line_number} is missing from manifest records"
            )
        _validate_record(record, ref, source_map, line_number)
        record_id = _string(record["record_id"], f"record line {line_number}.record_id")
        digest = record_digest(record)
        if record_id in seen_record_ids:
            raise DataContractError(f"duplicate record_id: {record_id}")
        if digest in seen_digests:
            raise DataContractError(
                f"duplicate record content across records: {record_id}"
            )
        if digest != ref["sha256"]:
            raise DataContractError(
                f"sha256 mismatch for {record_id}: "
                f"expected {ref['sha256']}, got {digest}"
            )
        seen_record_ids.add(record_id)
        seen_digests.add(digest)
        seen_lines.add(line_number)
        actual_counts[_string(record["split"], "record.split")] += 1

    if seen_lines != set(refs):
        missing = sorted(set(refs) - seen_lines)
        raise DataContractError(
            f"manifest references missing raw record lines: {missing}"
        )
    if actual_counts != declared_counts:
        raise DataContractError(
            f"manifest counts {declared_counts} do not match "
            f"raw records {actual_counts}"
        )

    return DataManifestSummary(
        dataset_id=dataset_id,
        revision=revision,
        manifest_path=manifest_path,
        record_count=len(seen_record_ids),
        source_count=len(source_map),
        split_counts=actual_counts,
    )


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise DataContractError(f"could not read {label}: {path}") from error
    try:
        return _load_object_from_text(text, label)
    except json.JSONDecodeError as error:
        raise DataContractError(f"invalid JSON in {label}: {error.msg}") from error


def _load_object_from_text(text: str, label: str) -> dict[str, Any]:
    value = json.loads(text)
    if not isinstance(value, dict):
        raise DataContractError(f"{label} must be a JSON object")
    return cast(dict[str, Any], value)


def _relative_path(base: Path, value: Any, label: str) -> Path:
    path = Path(_string(value, f"manifest.{label}"))
    if path.is_absolute():
        raise DataContractError(f"manifest.{label} must be relative")
    return (base / path).resolve()


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DataContractError(f"{label} must be a non-empty string")
    return value


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DataContractError(f"{label} must be an object")
    return cast(dict[str, Any], value)


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise DataContractError(f"{label} must be an array")
    return value


def _validate_string_list(value: Any, label: str) -> None:
    values = _list(value, label)
    if not values or any(
        not isinstance(item, str) or not item.strip() for item in values
    ):
        raise DataContractError(f"{label} must contain non-empty strings")


def _validate_source_policy(value: Any) -> None:
    policy = _object(value, "manifest.source_policy")
    if set(policy) != {"allowed_source_kinds", "raw_content_policy"}:
        raise DataContractError("source_policy has unexpected keys")
    if policy["raw_content_policy"] != "local-only":
        raise DataContractError("source_policy.raw_content_policy must be 'local-only'")
    kinds = {
        _string(item, "source_policy.allowed_source_kinds[]")
        for item in _list(
            policy["allowed_source_kinds"], "source_policy.allowed_source_kinds"
        )
    }
    if kinds != _SOURCE_KINDS:
        raise DataContractError(
            f"source_policy.allowed_source_kinds must be {sorted(_SOURCE_KINDS)}"
        )


def _validate_sources(value: Any) -> dict[str, dict[str, Any]]:
    sources = _list(value, "manifest.sources")
    if not sources:
        raise DataContractError("manifest.sources must not be empty")
    source_map: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(sources):
        source = _object(value, f"manifest.sources[{index}]")
        if set(source) != _SOURCE_KEYS:
            raise DataContractError(
                f"manifest.sources[{index}] must contain exactly {sorted(_SOURCE_KEYS)}"
            )
        source_id = _string(source["id"], f"manifest.sources[{index}].id")
        kind = _string(source["kind"], f"manifest.sources[{index}].kind")
        if kind not in _SOURCE_KINDS:
            raise DataContractError(f"unknown source kind: {kind}")
        _string(source["origin"], f"manifest.sources[{index}].origin")
        _string(source["license"], f"manifest.sources[{index}].license")
        _string(source["rights_note"], f"manifest.sources[{index}].rights_note")
        acquired_at = _string(
            source["acquired_at"], f"manifest.sources[{index}].acquired_at"
        )
        if len(acquired_at) != 10 or acquired_at[4] != "-" or acquired_at[7] != "-":
            raise DataContractError(f"invalid acquired_at for source {source_id}")
        if source_id in source_map:
            raise DataContractError(f"duplicate source id: {source_id}")
        source_map[source_id] = source
    return source_map


def _validate_split_policy(value: Any) -> None:
    policy = _object(value, "manifest.split_policy")
    expected = {
        "train": (True, False, False),
        "validation": (False, True, False),
        "holdout": (False, False, True),
    }
    if set(policy) != set(expected):
        raise DataContractError(
            "split_policy must define train, validation, and holdout"
        )
    for split, flags in expected.items():
        entry = _object(policy[split], f"split_policy.{split}")
        if set(entry) != _SPLIT_POLICY_KEYS:
            raise DataContractError(f"split_policy.{split} has unexpected keys")
        _string(entry["purpose"], f"split_policy.{split}.purpose")
        actual = (
            entry["allowed_for_training"],
            entry["used_for_model_selection"],
            entry["sealed"],
        )
        if any(type(flag) is not bool for flag in actual) or actual != flags:
            raise DataContractError(f"split_policy.{split} has invalid permissions")


def _validate_counts(value: Any) -> dict[str, int]:
    counts = _object(value, "manifest.counts")
    expected_keys = set(_DATA_SPLITS)
    if set(counts) != expected_keys:
        raise DataContractError("counts must define train, validation, and holdout")
    result: dict[str, int] = {}
    for key in expected_keys:
        count = counts[key]
        if type(count) is not int or count <= 0:
            raise DataContractError(f"counts.{key} must be a positive integer")
        result[key] = count
    return result


def _validate_record_refs(
    value: Any, source_map: Mapping[str, Mapping[str, Any]]
) -> dict[int, dict[str, Any]]:
    refs = _list(value, "manifest.records")
    if not refs:
        raise DataContractError("manifest.records must not be empty")
    by_line: dict[int, dict[str, Any]] = {}
    ids: set[str] = set()
    for index, value in enumerate(refs):
        ref = _object(value, f"manifest.records[{index}]")
        if set(ref) != _REF_KEYS:
            raise DataContractError(f"manifest.records[{index}] has unexpected keys")
        record_id = _string(ref["record_id"], f"manifest.records[{index}].record_id")
        split = _string(ref["split"], f"manifest.records[{index}].split")
        task = _string(ref["task"], f"manifest.records[{index}].task")
        source_id = _string(ref["source_id"], f"manifest.records[{index}].source_id")
        line_number = ref["line_number"]
        digest = _string(ref["sha256"], f"manifest.records[{index}].sha256")
        if split not in _DATA_SPLITS:
            raise DataContractError(f"unknown record split: {split}")
        if task not in _TASKS:
            raise DataContractError(f"unknown record task: {task}")
        if source_id not in source_map:
            raise DataContractError(
                f"record {record_id} references unknown source {source_id}"
            )
        if type(line_number) is not int or line_number <= 0:
            raise DataContractError(f"record {record_id} line_number must be positive")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise DataContractError(f"record {record_id} sha256 must be lowercase hex")
        if record_id in ids:
            raise DataContractError(f"duplicate manifest record id: {record_id}")
        if line_number in by_line:
            raise DataContractError(f"duplicate manifest line number: {line_number}")
        ids.add(record_id)
        by_line[line_number] = ref
    return by_line


def _validate_record(
    record: Mapping[str, Any],
    ref: Mapping[str, Any],
    source_map: Mapping[str, Mapping[str, Any]],
    line_number: int,
) -> None:
    if not _RECORD_KEYS.issuperset(record):
        raise DataContractError(f"raw record line {line_number} has unexpected keys")
    for key in {"record_id", "split", "source", "task", "input", "target"}:
        if key not in record:
            raise DataContractError(f"raw record line {line_number} is missing {key}")
    record_id = _string(record["record_id"], f"record line {line_number}.record_id")
    split = _string(record["split"], f"record line {line_number}.split")
    task = _string(record["task"], f"record line {line_number}.task")
    if record_id != ref["record_id"] or split != ref["split"] or task != ref["task"]:
        raise DataContractError(f"manifest metadata does not match record {record_id}")
    source = _object(record["source"], f"record {record_id}.source")
    if set(source) != _SOURCE_KEYS or source != source_map[ref["source_id"]]:
        raise DataContractError(f"record {record_id} has mismatched provenance")
    if task not in _TASKS or split not in _DATA_SPLITS:
        raise DataContractError(f"record {record_id} has invalid task or split")
    input_value = _object(record["input"], f"record {record_id}.input")
    if not {"brief"}.issubset(input_value) or not set(input_value).issubset(
        {"brief", "constraints", "draft"}
    ):
        raise DataContractError(f"record {record_id}.input has invalid keys")
    _string(input_value["brief"], f"record {record_id}.input.brief")
    if "draft" in input_value and not isinstance(input_value["draft"], str):
        raise DataContractError(f"record {record_id}.input.draft must be a string")
    if "constraints" in input_value:
        constraints = _list(
            input_value["constraints"], f"record {record_id}.input.constraints"
        )
        if any(not isinstance(item, str) or not item.strip() for item in constraints):
            raise DataContractError(
                f"record {record_id}.input.constraints must be strings"
            )
    _string(record["target"], f"record {record_id}.target")
    if "metadata" in record:
        metadata = _object(record["metadata"], f"record {record_id}.metadata")
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in metadata.items()
        ):
            raise DataContractError(f"record {record_id}.metadata must contain strings")
