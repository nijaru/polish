import json
from pathlib import Path

from polish.cli import main
from polish.evaluation import GenerationSettings


def test_evaluate_command_writes_a_report(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "0.1",
                "revision": "fixture-v1",
                "cases": [
                    {
                        "id": "case-1",
                        "split": "holdout",
                        "prompt": "Write one sentence.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    settings = GenerationSettings(seed=7)
    recorded_paths = []
    for model_id, text in (("model-a", "First."), ("model-b", "Second.")):
        recorded_path = tmp_path / f"{model_id}.json"
        recorded_path.write_text(
            json.dumps(
                {
                    "schema_version": "0.1",
                    "manifest_revision": "fixture-v1",
                    "split": "holdout",
                    "settings": settings.to_dict(),
                    "model": {"id": model_id, "revision": "fixture"},
                    "outputs": [{"case_id": "case-1", "text": text}],
                }
            ),
            encoding="utf-8",
        )
        recorded_paths.append(recorded_path)

    report_path = tmp_path / "report.json"
    exit_code = main(
        [
            "evaluate",
            "--manifest",
            str(manifest_path),
            "--recorded-output",
            str(recorded_paths[0]),
            "--recorded-output",
            str(recorded_paths[1]),
            "--output",
            str(report_path),
            "--seed",
            "7",
        ]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["manifest_revision"] == "fixture-v1"
    assert [model["id"] for model in report["models"]] == ["model-a", "model-b"]
