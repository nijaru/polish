import json
from pathlib import Path

import pytest

from polish.evaluation import (
    EvaluationError,
    EvaluationRunner,
    GenerationSettings,
    ModelSpec,
    PromptCase,
    PromptManifest,
    RecordedOutputAdapter,
    Split,
    surface_metrics,
)


class FixtureAdapter:
    def __init__(self, model_id: str, outputs: dict[str, str]) -> None:
        self.spec = ModelSpec(model_id, revision="fixture", quantization="none")
        self.outputs = outputs

    def generate(self, case: PromptCase, settings: GenerationSettings) -> str:
        del settings
        return self.outputs[case.case_id]


@pytest.fixture
def manifest() -> PromptManifest:
    return PromptManifest(
        "prompts-v1",
        (
            PromptCase("holdout-1", Split.HOLDOUT, "Write a short scene."),
            PromptCase("holdout-2", Split.HOLDOUT, "Explain a useful habit."),
            PromptCase("dev-1", Split.DEVELOPMENT, "Draft a title."),
        ),
    )


def test_runner_uses_the_same_ordered_holdout_cases_for_each_model(
    manifest: PromptManifest,
) -> None:
    report = EvaluationRunner().run(
        manifest,
        (
            FixtureAdapter(
                "model-a", {"holdout-1": "A scene.", "holdout-2": "A habit."}
            ),
            FixtureAdapter(
                "model-b",
                {"holdout-1": "Another scene.", "holdout-2": "Another habit."},
            ),
        ),
        settings=GenerationSettings(seed=42),
    )

    assert [output.case_id for output in report.outputs] == [
        "holdout-1",
        "holdout-2",
        "holdout-1",
        "holdout-2",
    ]
    assert {output.model_id for output in report.outputs} == {"model-a", "model-b"}
    assert report.split is Split.HOLDOUT
    assert report.to_dict()["aggregate_metrics"]["model-a"]["words"] == 2.0


def test_report_id_is_stable_for_identical_inputs(manifest: PromptManifest) -> None:
    adapters = (FixtureAdapter("model-a", {"holdout-1": "A.", "holdout-2": "B."}),)
    first = EvaluationRunner().run(manifest, adapters)
    second = EvaluationRunner().run(manifest, adapters)

    assert first.report_id == second.report_id
    assert first.to_dict() == second.to_dict()


def test_manifest_rejects_duplicate_case_ids() -> None:
    with pytest.raises(EvaluationError, match="case ids must be unique"):
        PromptManifest(
            "bad",
            (
                PromptCase("same", Split.HOLDOUT, "One."),
                PromptCase("same", Split.HOLDOUT, "Two."),
            ),
        )


def test_surface_metrics_are_transparent_and_deterministic() -> None:
    metrics = surface_metrics("One small test. One small test.")

    assert metrics["words"] == 6.0
    assert metrics["sentences"] == 2.0
    assert metrics["unique_word_ratio"] == pytest.approx(3 / 6)
    assert metrics["repeated_trigram_rate"] > 0


def test_recorded_output_adapter_loads_model_metadata(tmp_path: Path) -> None:
    path = tmp_path / "outputs.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "0.1",
                "manifest_revision": "prompts-v1",
                "split": "holdout",
                "settings": GenerationSettings().to_dict(),
                "model": {
                    "id": "local/model",
                    "revision": "abc123",
                    "quantization": "q4_k_m",
                },
                "outputs": [
                    {
                        "case_id": "case-1",
                        "text": "Recorded.",
                        "raw_text": "<thought>Recorded.</thought>Recorded.",
                        "finish_reason": "stop",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    adapter = RecordedOutputAdapter.from_path(path)

    assert adapter.spec.revision == "abc123"
    assert adapter.raw_text_for("case-1") == "<thought>Recorded.</thought>Recorded."
    assert adapter.finish_reason_for("case-1") == "stop"
    assert (
        adapter.generate(
            PromptCase("case-1", Split.HOLDOUT, "Prompt."), GenerationSettings()
        )
        == "Recorded."
    )
