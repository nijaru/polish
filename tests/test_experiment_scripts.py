from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_blind_review_main():
    path = Path(__file__).parents[1] / "scripts/build_blind_review.py"
    spec = importlib.util.spec_from_file_location("build_blind_review", path)
    if spec is None or spec.loader is None:
        raise AssertionError("could not load blind review script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


build_blind_review = _load_blind_review_main()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_blind_review_packet_separates_model_key(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    report = tmp_path / "report.json"
    packet = tmp_path / "packet.json"
    key = tmp_path / "key.json"
    _write_json(
        manifest,
        {
            "schema_version": "0.1",
            "revision": "fixture-v1",
            "cases": [
                {
                    "id": "case-1",
                    "split": "holdout",
                    "prompt": "Write a scene.",
                    "metadata": {"genre": "scene"},
                }
            ],
        },
    )
    _write_json(
        report,
        {
            "report_id": "report-1",
            "manifest_revision": "fixture-v1",
            "split": "holdout",
            "settings": {
                "enable_thinking": True,
                "max_new_tokens": 16384,
                "temperature": 0.7,
                "top_p": 0.95,
                "seed": 0,
            },
            "models": [{"id": "base-model"}, {"id": "adapter-model"}],
            "outputs": [
                {"case_id": "case-1", "model_id": "base-model", "text": "Base."},
                {
                    "case_id": "case-1",
                    "model_id": "adapter-model",
                    "text": "Adapter.",
                },
            ],
        },
    )

    assert (
        build_blind_review(
            [
                "--manifest",
                str(manifest),
                "--report",
                str(report),
                "--output",
                str(packet),
                "--key-output",
                str(key),
            ]
        )
        == 0
    )

    packet_value = json.loads(packet.read_text(encoding="utf-8"))
    key_value = json.loads(key.read_text(encoding="utf-8"))
    assert packet_value["blind"] is True
    assert "base-model" not in json.dumps(packet_value)
    assert "adapter-model" not in json.dumps(packet_value)
    assert key_value["review_id"] == packet_value["review_id"]
    assert set(key_value["cases"][0]["labels"].values()) == {
        "base-model",
        "adapter-model",
    }
