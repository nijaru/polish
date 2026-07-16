#!/usr/bin/env python3
"""Capture MLX-LM generations into Polish's recorded-output contract.

This is intentionally a runtime-specific script outside the package. It keeps
the evaluator usable with MLX-LM, llama.cpp, or a remote runner without making
any of them part of the core experiment contract.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any

from polish.evaluation import (
    SCHEMA_VERSION,
    EvaluationError,
    GenerationSettings,
    PromptManifest,
    Split,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture a Hugging Face model with MLX-LM."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Hugging Face model ID or local MLX model directory.",
    )
    parser.add_argument(
        "--model-id",
        help="Identity written to the report; defaults to --model.",
    )
    parser.add_argument("--revision", help="Hub commit or local model revision.")
    parser.add_argument(
        "--quantization",
        default="mlx-4bit",
        help="Serving quantization label recorded in the report.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=[split.value for split in Split],
        default=Split.HOLDOUT.value,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--adapter-path",
        type=Path,
        help="Optional MLX-LM LoRA adapter directory.",
    )
    return parser


def _resolve_revision(model: str, revision: str | None) -> str:
    if revision:
        return revision
    if Path(model).exists():
        raise EvaluationError(
            "--revision is required when --model points to a local directory"
        )

    hf_api: Any = importlib.import_module("huggingface_hub").HfApi()
    info = hf_api.model_info(model)
    if not info.sha:
        raise EvaluationError(f"Hugging Face model {model!r} has no commit SHA")
    return info.sha


def _runtime_info(
    model: str, revision: str, adapter_path: Path | None
) -> dict[str, str]:
    return {
        "runtime": "mlx-lm",
        "runtime_version": importlib.metadata.version("mlx-lm"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "model_source": model,
        "model_revision": revision,
        "adapter_path": str(adapter_path) if adapter_path else "none",
    }


def _visible_text(raw_text: str) -> str:
    """Remove Gemma 4's internal thought channel from the presented answer."""

    if "<channel|>" in raw_text:
        return raw_text.rsplit("<channel|>", 1)[1].strip()
    return raw_text.strip()


def _capture(args: argparse.Namespace) -> dict[str, Any]:
    # Keep optional runtime imports out of the core package and its tests.
    mx: Any = importlib.import_module("mlx.core")
    mlx_lm: Any = importlib.import_module("mlx_lm")
    make_sampler: Any = importlib.import_module("mlx_lm.sample_utils").make_sampler

    manifest = PromptManifest.from_path(args.manifest)
    split = Split(args.split)
    settings = GenerationSettings(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )
    revision = _resolve_revision(args.model, args.revision)
    model_id = args.model_id or args.model
    is_local_model = Path(args.model).exists()
    model, tokenizer = mlx_lm.load(
        args.model,
        adapter_path=str(args.adapter_path) if args.adapter_path else None,
        revision=None if is_local_model else revision,
    )
    sampler = make_sampler(settings.temperature, settings.top_p)
    outputs: list[dict[str, str]] = []
    for case in manifest.for_split(split):
        # Reset per case so the same case receives the same RNG stream across
        # model runs, independent of how long earlier outputs were.
        mx.random.seed(settings.seed)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": case.prompt}],
            add_generation_prompt=True,
        )
        responses = mlx_lm.stream_generate(
            model,
            tokenizer,
            prompt,
            max_tokens=settings.max_new_tokens,
            sampler=sampler,
        )
        chunks: list[str] = []
        finish_reason: str | None = None
        for response in responses:
            chunks.append(response.text)
            finish_reason = response.finish_reason
        text = "".join(chunks)
        outputs.append(
            {
                "case_id": case.case_id,
                "text": _visible_text(text),
                "raw_text": text,
                "finish_reason": finish_reason or "unknown",
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_revision": manifest.revision,
        "split": split.value,
        "settings": settings.to_dict(),
        "model": {
            "id": model_id,
            "revision": revision,
            "quantization": args.quantization,
        },
        "runtime_info": _runtime_info(args.model, revision, args.adapter_path),
        "outputs": outputs,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = _capture(args)
    except EvaluationError as error:
        _parser().error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote MLX capture {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
