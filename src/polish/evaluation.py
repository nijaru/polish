"""Data contracts and execution primitives for reproducible evaluations."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, Self

SCHEMA_VERSION = "0.1"
_WORD_RE = re.compile(r"\b[\w\u2019'-]+\b", re.UNICODE)
_SENTENCE_RE = re.compile(r"[.!?]+(?:[\"'\u2019)\]]+)?")


class EvaluationError(ValueError):
    """Raised when an evaluation manifest, output, or run is invalid."""


class Split(StrEnum):
    """The evaluation split a prompt belongs to."""

    DEVELOPMENT = "development"
    VALIDATION = "validation"
    HOLDOUT = "holdout"


def _require_non_empty(value: str, field_name: str) -> str:
    if not value.strip():
        raise EvaluationError(f"{field_name} must not be empty")
    return value


def _as_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise EvaluationError(f"{field_name} must be a string")
    return value


def _as_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvaluationError(f"{field_name} must be an integer")
    return value


def _as_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationError(f"{field_name} must be a number")
    return float(value)


def _as_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise EvaluationError(f"{field_name} must be a boolean")
    return value


def _validate_schema_version(value: Mapping[str, Any], source: str) -> None:
    schema_version = value.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise EvaluationError(
            f"{source} uses schema version {schema_version!r}, "
            f"expected {SCHEMA_VERSION!r}"
        )


@dataclass(frozen=True, slots=True)
class PromptCase:
    """One fixed writing prompt in an evaluation manifest."""

    case_id: str
    split: Split
    prompt: str
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.case_id, "case_id")
        _require_non_empty(self.prompt, "prompt")
        if not isinstance(self.split, Split):
            raise EvaluationError("split must be a Split")
        for key, value in self.metadata.items():
            _require_non_empty(key, "metadata key")
            _require_non_empty(value, f"metadata[{key!r}]")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        try:
            split = Split(_as_string(value["split"], "split"))
        except KeyError as error:
            raise EvaluationError("prompt case is missing split") from error
        except ValueError as error:
            raise EvaluationError(
                f"unknown evaluation split: {value['split']!r}"
            ) from error

        raw_metadata = value.get("metadata", {})
        if not isinstance(raw_metadata, dict) or not all(
            isinstance(key, str) and isinstance(item, str)
            for key, item in raw_metadata.items()
        ):
            raise EvaluationError("metadata must be an object of string values")

        try:
            case_id = _as_string(value["id"], "id")
            prompt = _as_string(value["prompt"], "prompt")
        except KeyError as error:
            raise EvaluationError(f"prompt case is missing {error.args[0]}") from error
        return cls(case_id, split, prompt, raw_metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.case_id,
            "split": self.split.value,
            "prompt": self.prompt,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class PromptManifest:
    """A versioned, fixed collection of evaluation prompts."""

    revision: str
    cases: tuple[PromptCase, ...]

    def __post_init__(self) -> None:
        _require_non_empty(self.revision, "manifest revision")
        if not self.cases:
            raise EvaluationError("manifest must contain at least one prompt case")
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise EvaluationError("manifest case ids must be unique")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        _validate_schema_version(value, "manifest")
        revision = _as_string(value.get("revision"), "revision")
        raw_cases = value.get("cases")
        if not isinstance(raw_cases, list):
            raise EvaluationError("manifest cases must be an array")
        cases_list: list[PromptCase] = []
        for case in raw_cases:
            if not isinstance(case, Mapping):
                raise EvaluationError("each case must be an object")
            cases_list.append(PromptCase.from_dict(case))
        cases = tuple(cases_list)
        return cls(revision, cases)

    @classmethod
    def from_path(cls, path: Path) -> Self:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except OSError as error:
            raise EvaluationError(f"cannot read manifest {path}: {error}") from error
        except json.JSONDecodeError as error:
            raise EvaluationError(
                f"invalid JSON in manifest {path}: {error}"
            ) from error
        if not isinstance(value, dict):
            raise EvaluationError("manifest root must be an object")
        return cls.from_dict(value)

    def for_split(self, split: Split) -> tuple[PromptCase, ...]:
        selected = tuple(case for case in self.cases if case.split is split)
        if not selected:
            raise EvaluationError(
                f"manifest {self.revision!r} has no cases in {split.value!r} split"
            )
        return selected

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "revision": self.revision,
            "cases": [case.to_dict() for case in self.cases],
        }


@dataclass(frozen=True, slots=True)
class GenerationSettings:
    """Settings that must be recorded alongside every generated output."""

    max_new_tokens: int = 768
    temperature: float = 0.7
    top_p: float = 0.95
    seed: int = 0
    enable_thinking: bool = False

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        return cls(
            max_new_tokens=_as_int(
                value.get("max_new_tokens", 512), "settings.max_new_tokens"
            ),
            temperature=_as_float(
                value.get("temperature", 0.7), "settings.temperature"
            ),
            top_p=_as_float(value.get("top_p", 0.95), "settings.top_p"),
            seed=_as_int(value.get("seed", 0), "settings.seed"),
            enable_thinking=_as_bool(
                value.get("enable_thinking", False), "settings.enable_thinking"
            ),
        )

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise EvaluationError("max_new_tokens must be positive")
        if not self.temperature >= 0.0:
            raise EvaluationError("temperature must not be negative")
        if not 0.0 < self.top_p <= 1.0:
            raise EvaluationError("top_p must be greater than 0 and at most 1")

    def to_dict(self) -> dict[str, int | float | bool]:
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "enable_thinking": self.enable_thinking,
        }


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Identity and serving details for one evaluated model."""

    model_id: str
    revision: str = "unknown"
    quantization: str = "none"

    def __post_init__(self) -> None:
        _require_non_empty(self.model_id, "model_id")
        _require_non_empty(self.revision, "model revision")
        _require_non_empty(self.quantization, "quantization")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        try:
            model_id = _as_string(value["id"], "model.id")
        except KeyError as error:
            raise EvaluationError("model metadata is missing id") from error
        revision = _as_string(value.get("revision", "unknown"), "model.revision")
        quantization = _as_string(
            value.get("quantization", "none"), "model.quantization"
        )
        return cls(model_id, revision, quantization)

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.model_id,
            "revision": self.revision,
            "quantization": self.quantization,
        }


@dataclass(frozen=True, slots=True)
class OutputRecord:
    """One model output and transparent structural measurements."""

    case_id: str
    model_id: str
    text: str
    metrics: Mapping[str, float]
    raw_text: str | None = None
    finish_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "case_id": self.case_id,
            "model_id": self.model_id,
            "text": self.text,
            "metrics": dict(self.metrics),
        }
        if self.raw_text is not None:
            result["raw_text"] = self.raw_text
        if self.finish_reason is not None:
            result["finish_reason"] = self.finish_reason
        return result


class ModelAdapter(Protocol):
    """Minimal interface required to plug a local model into the runner."""

    @property
    def spec(self) -> ModelSpec: ...

    def generate(self, case: PromptCase, settings: GenerationSettings) -> str: ...


def surface_metrics(text: str) -> dict[str, float]:
    """Return reproducible surface metrics, not a claim of writing quality."""

    words = [match.group(0).casefold() for match in _WORD_RE.finditer(text)]
    sentences = len(_SENTENCE_RE.findall(text))
    if words and sentences == 0:
        sentences = 1
    paragraphs = [paragraph for paragraph in text.split("\n\n") if paragraph.strip()]
    trigrams = list(zip(words, words[1:], words[2:], strict=False))
    repeated_trigrams = len(trigrams) - len(set(trigrams))
    return {
        "characters": float(len(text)),
        "words": float(len(words)),
        "sentences": float(sentences),
        "paragraphs": float(len(paragraphs)),
        "unique_word_ratio": len(set(words)) / len(words) if words else 0.0,
        "repeated_trigram_rate": (
            repeated_trigrams / len(trigrams) if trigrams else 0.0
        ),
    }


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """A stable report containing the run contract, outputs, and aggregates."""

    report_id: str
    manifest_revision: str
    split: Split
    settings: GenerationSettings
    models: tuple[ModelSpec, ...]
    outputs: tuple[OutputRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        aggregates: dict[str, dict[str, float]] = {}
        for model in self.models:
            model_outputs = [
                output for output in self.outputs if output.model_id == model.model_id
            ]
            if not model_outputs:
                continue
            metric_names = model_outputs[0].metrics.keys()
            aggregates[model.model_id] = {
                metric: sum(output.metrics[metric] for output in model_outputs)
                / len(model_outputs)
                for metric in metric_names
            }
        return {
            "schema_version": SCHEMA_VERSION,
            "report_id": self.report_id,
            "manifest_revision": self.manifest_revision,
            "split": self.split.value,
            "settings": self.settings.to_dict(),
            "models": [model.to_dict() for model in self.models],
            "outputs": [output.to_dict() for output in self.outputs],
            "aggregate_metrics": aggregates,
        }

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _report_id(
    manifest: PromptManifest,
    split: Split,
    settings: GenerationSettings,
    models: Sequence[ModelSpec],
    outputs: Sequence[OutputRecord],
) -> str:
    payload = {
        "manifest_revision": manifest.revision,
        "split": split.value,
        "settings": settings.to_dict(),
        "models": [model.to_dict() for model in models],
        "outputs": [output.to_dict() for output in outputs],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


class EvaluationRunner:
    """Run every adapter against the exact same ordered prompt cases."""

    def run(
        self,
        manifest: PromptManifest,
        adapters: Sequence[ModelAdapter],
        *,
        split: Split = Split.HOLDOUT,
        settings: GenerationSettings | None = None,
    ) -> EvaluationReport:
        if not adapters:
            raise EvaluationError("at least one model adapter is required")
        settings = settings or GenerationSettings()
        models = tuple(adapter.spec for adapter in adapters)
        model_ids = [model.model_id for model in models]
        if len(model_ids) != len(set(model_ids)):
            raise EvaluationError("model ids must be unique within an evaluation")

        outputs: list[OutputRecord] = []
        for adapter in adapters:
            for case in manifest.for_split(split):
                text = adapter.generate(case, settings)
                if not isinstance(text, str):
                    raise EvaluationError(
                        f"adapter {adapter.spec.model_id!r} returned non-string output"
                    )
                outputs.append(
                    OutputRecord(
                        case.case_id,
                        adapter.spec.model_id,
                        text,
                        surface_metrics(text),
                        (
                            adapter.raw_text_for(case.case_id)
                            if isinstance(adapter, RecordedOutputAdapter)
                            else None
                        ),
                        (
                            adapter.finish_reason_for(case.case_id)
                            if isinstance(adapter, RecordedOutputAdapter)
                            else None
                        ),
                    )
                )

        return EvaluationReport(
            _report_id(manifest, split, settings, models, outputs),
            manifest.revision,
            split,
            settings,
            models,
            tuple(outputs),
        )


@dataclass(frozen=True, slots=True)
class RecordedOutputAdapter:
    """Adapter for outputs captured from an external local model runtime."""

    spec: ModelSpec
    manifest_revision: str
    split: Split
    settings: GenerationSettings
    outputs: Mapping[str, str]
    raw_outputs: Mapping[str, str] = field(default_factory=dict)
    finish_reasons: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_path(cls, path: Path) -> Self:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except OSError as error:
            raise EvaluationError(
                f"cannot read recorded outputs {path}: {error}"
            ) from error
        except json.JSONDecodeError as error:
            raise EvaluationError(
                f"invalid JSON in recorded outputs {path}: {error}"
            ) from error
        if not isinstance(value, dict):
            raise EvaluationError("recorded outputs root must be an object")
        _validate_schema_version(value, "recorded outputs")
        raw_model = value.get("model")
        raw_revision = value.get("manifest_revision")
        raw_split = value.get("split")
        raw_settings = value.get("settings")
        raw_output_items = value.get("outputs")
        if not isinstance(raw_model, Mapping):
            raise EvaluationError("recorded outputs must contain a model object")
        manifest_revision = _as_string(raw_revision, "manifest_revision")
        try:
            split = Split(_as_string(raw_split, "split"))
        except ValueError as error:
            raise EvaluationError(f"unknown evaluation split: {raw_split!r}") from error
        if not isinstance(raw_settings, Mapping):
            raise EvaluationError("recorded outputs must contain a settings object")
        settings = GenerationSettings.from_dict(raw_settings)
        if not isinstance(raw_output_items, list):
            raise EvaluationError("recorded outputs must contain an outputs array")
        outputs: dict[str, str] = {}
        raw_outputs: dict[str, str] = {}
        finish_reasons: dict[str, str] = {}
        for item in raw_output_items:
            if not isinstance(item, Mapping):
                raise EvaluationError("each recorded output must be an object")
            try:
                case_id = _as_string(item["case_id"], "output.case_id")
                text = _as_string(item["text"], "output.text")
            except KeyError as error:
                raise EvaluationError(
                    f"recorded output is missing {error.args[0]}"
                ) from error
            if case_id in outputs:
                raise EvaluationError(f"duplicate recorded output for {case_id!r}")
            outputs[case_id] = text
            raw_text = item.get("raw_text")
            if raw_text is not None:
                raw_outputs[case_id] = _as_string(raw_text, "output.raw_text")
            finish_reason = item.get("finish_reason")
            if finish_reason is not None:
                finish_reasons[case_id] = _as_string(
                    finish_reason, "output.finish_reason"
                )
        return cls(
            ModelSpec.from_dict(raw_model),
            manifest_revision,
            split,
            settings,
            outputs,
            raw_outputs,
            finish_reasons,
        )

    def validate_contract(
        self,
        manifest: PromptManifest,
        split: Split,
        settings: GenerationSettings,
    ) -> None:
        if self.manifest_revision != manifest.revision:
            raise EvaluationError(
                f"recorded outputs for {self.spec.model_id!r} use manifest "
                f"{self.manifest_revision!r}, expected {manifest.revision!r}"
            )
        if self.split is not split:
            raise EvaluationError(
                f"recorded outputs for {self.spec.model_id!r} use split "
                f"{self.split.value!r}, expected {split.value!r}"
            )
        if self.settings != settings:
            raise EvaluationError(
                f"recorded outputs for {self.spec.model_id!r} use different "
                "generation settings"
            )
        expected_case_ids = {case.case_id for case in manifest.for_split(split)}
        actual_case_ids = set(self.outputs)
        missing = expected_case_ids - actual_case_ids
        extra = actual_case_ids - expected_case_ids
        if missing:
            raise EvaluationError(
                f"recorded outputs for {self.spec.model_id!r} are missing "
                f"cases: {', '.join(sorted(missing))}"
            )
        if extra:
            raise EvaluationError(
                f"recorded outputs for {self.spec.model_id!r} contain unexpected "
                f"cases: {', '.join(sorted(extra))}"
            )

    def generate(self, case: PromptCase, settings: GenerationSettings) -> str:
        del settings
        try:
            return self.outputs[case.case_id]
        except KeyError as error:
            raise EvaluationError(
                f"recorded outputs for {self.spec.model_id!r} are missing "
                f"{case.case_id!r}"
            ) from error

    def raw_text_for(self, case_id: str) -> str | None:
        """Return optional runtime text retained alongside visible output."""

        return self.raw_outputs.get(case_id)

    def finish_reason_for(self, case_id: str) -> str | None:
        """Return the runtime stop reason when the capture recorded one."""

        return self.finish_reasons.get(case_id)
