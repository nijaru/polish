#!/usr/bin/env python3
"""Render a model-blinded JSON review packet as readable Markdown."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read review packet {path}") from error
    if not isinstance(value, dict) or value.get("blind") is not True:
        raise ValueError("input must be a blind review packet")
    if not isinstance(value.get("cases"), list):
        raise ValueError("review packet must contain cases")
    return value


def _indented(text: str) -> str:
    return "\n".join(f"    {line}" for line in text.splitlines())


def _score_line(label: str, dimensions: list[str]) -> str:
    return f"- {label}: " + "; ".join(f"{dimension}: __" for dimension in dimensions)


def _render(packet: dict[str, Any]) -> str:
    instructions = packet["instructions"]
    dimensions = [str(item) for item in instructions["dimensions"]]
    lines = [
        "# Blind A/B writing review",
        "",
        f"Review ID: `{packet['review_id']}`",
        f"Manifest: `{packet['manifest_revision']}`",
        "",
        (
            "Review each pair without trying to infer which system produced either "
            "response."
        ),
        f"Scale: {instructions['scale']}",
        "",
    ]
    settings = packet.get("generation_settings")
    if isinstance(settings, dict):
        lines.extend(
            [
                "Generation condition: "
                f"thinking={settings.get('enable_thinking')}, "
                f"max_new_tokens={settings.get('max_new_tokens')}, "
                f"temperature={settings.get('temperature')}, "
                f"seed={settings.get('seed')}",
                "",
            ]
        )
    for index, case in enumerate(packet["cases"], 1):
        if not isinstance(case, dict):
            raise ValueError(f"case {index} is not an object")
        responses = case.get("responses")
        if not isinstance(responses, dict) or set(responses) != {"A", "B"}:
            raise ValueError(f"case {index} must contain A and B responses")
        lines.extend(
            [
                f"## {index}. {case['case_id']} ({case['split']})",
                "",
                "Prompt:",
                _indented(str(case["prompt"])),
                "",
                "### A",
                "",
                _indented(str(responses["A"])),
                "",
                "### B",
                "",
                _indented(str(responses["B"])),
                "",
                "Review fields:",
                "",
                "- Preferred: `A` / `B` / `tie` / `neither`",
                _score_line("A scores", dimensions),
                _score_line("B scores", dimensions),
                "- Notes: ",
                "",
            ]
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        markdown = _render(_load(args.input))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
    except (OSError, ValueError) as error:
        _parser().error(str(error))
    print(f"wrote readable blind review {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
