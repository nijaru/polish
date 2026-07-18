#!/usr/bin/env python3
"""Build a deterministic, model-blinded A/B review packet from reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REVIEW_SCHEMA_VERSION = "0.1"
DIMENSIONS = (
    "specificity",
    "coherence",
    "meaningful_detail",
    "repetition",
    "rhythm",
    "instruction_adherence",
    "overall",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key-output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON object {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _case_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise ValueError("evaluation manifest must contain a cases array")
    result: dict[str, dict[str, Any]] = {}
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str):
            raise ValueError("evaluation manifest contains an invalid case")
        result[case["id"]] = case
    return result


def _report_pairs(report: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    models = report.get("models")
    outputs = report.get("outputs")
    split = report.get("split")
    if not isinstance(split, str) or not isinstance(models, list) or len(models) != 2:
        raise ValueError("each report must contain exactly two models and one split")
    if not isinstance(outputs, list):
        raise ValueError("each report must contain an outputs array")
    model_ids = []
    for model in models:
        if not isinstance(model, dict) or not isinstance(model.get("id"), str):
            raise ValueError("report contains an invalid model")
        model_ids.append(model["id"])
    if len(set(model_ids)) != 2:
        raise ValueError("report model IDs must be unique")
    by_case: dict[str, dict[str, Any]] = {}
    for output in outputs:
        if not isinstance(output, dict):
            raise ValueError("report contains an invalid output")
        case_id = output.get("case_id")
        model_id = output.get("model_id")
        text = output.get("text")
        if not all(isinstance(value, str) for value in (case_id, model_id, text)):
            raise ValueError("report output is missing case_id, model_id, or text")
        by_case.setdefault(case_id, {})[model_id] = text
    return split, {"models": model_ids, "outputs": by_case}


def _review_id(
    manifest: dict[str, Any], reports: list[dict[str, Any]], seed: int
) -> str:
    payload = {
        "manifest_revision": manifest.get("revision"),
        "reports": [report.get("report_id") for report in reports],
        "seed": seed,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = _load(args.manifest)
        cases = _case_map(manifest)
        reports = [_load(path) for path in args.report]
        settings = reports[0].get("settings")
        if any(report.get("settings") != settings for report in reports[1:]):
            raise ValueError("reports must use identical generation settings")
        review_id = _review_id(manifest, reports, args.seed)
        review_cases: list[dict[str, Any]] = []
        key_cases: list[dict[str, Any]] = []
        seen_case_ids: set[str] = set()
        for report in reports:
            split, pair_data = _report_pairs(report)
            for case_id, outputs in sorted(pair_data["outputs"].items()):
                if case_id in seen_case_ids:
                    raise ValueError(
                        f"case {case_id!r} appears in more than one report"
                    )
                seen_case_ids.add(case_id)
                if case_id not in cases:
                    raise ValueError(
                        f"report case {case_id!r} is absent from the manifest"
                    )
                if set(outputs) != set(pair_data["models"]):
                    raise ValueError(
                        f"report case {case_id!r} does not have both model outputs"
                    )
                digest = hashlib.sha256(
                    f"{args.seed}:{split}:{case_id}".encode()
                ).digest()
                left_model, right_model = pair_data["models"]
                if digest[0] % 2:
                    left_model, right_model = right_model, left_model
                model_to_text = outputs
                case = cases[case_id]
                review_cases.append(
                    {
                        "case_id": case_id,
                        "split": split,
                        "prompt": case["prompt"],
                        "metadata": case.get("metadata", {}),
                        "responses": {
                            "A": model_to_text[left_model],
                            "B": model_to_text[right_model],
                        },
                        "review": {
                            "preferred": None,
                            "scores": {
                                label: {dimension: None for dimension in DIMENSIONS}
                                for label in ("A", "B")
                            },
                            "notes": "",
                        },
                    }
                )
                key_cases.append(
                    {
                        "case_id": case_id,
                        "split": split,
                        "labels": {"A": left_model, "B": right_model},
                    }
                )
        if not review_cases:
            raise ValueError("reports produced no review cases")
        common_manifest_revision = {
            report.get("manifest_revision") for report in reports
        }
        if len(common_manifest_revision) != 1:
            raise ValueError("reports must use one evaluation manifest revision")
        packet = {
            "schema_version": REVIEW_SCHEMA_VERSION,
            "review_id": review_id,
            "blind": True,
            "manifest_revision": next(iter(common_manifest_revision)),
            "generation_settings": settings,
            "instructions": {
                "purpose": (
                    "Compare the two responses for writing quality, not model identity."
                ),
                "do_not_infer_identity": True,
                "scale": (
                    "1-5, where 1 is poor and 5 is strong; leave a score blank "
                    "if the dimension is not applicable."
                ),
                "dimensions": list(DIMENSIONS),
                "preferred_values": ["A", "B", "tie", "neither"],
                "review_order": (
                    "Read the prompt, then both responses, then score independently."
                ),
            },
            "cases": review_cases,
        }
        key = {
            "schema_version": REVIEW_SCHEMA_VERSION,
            "review_id": review_id,
            "warning": "Keep this file separate until the blind scores are recorded.",
            "models": sorted(
                {model for item in key_cases for model in item["labels"].values()}
            ),
            "cases": key_cases,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.key_output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(packet, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        args.key_output.write_text(
            json.dumps(key, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    except ValueError as error:
        _parser().error(str(error))
    print(f"wrote blind review packet {args.output} ({review_id})")
    print(f"wrote unblinding key {args.key_output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
