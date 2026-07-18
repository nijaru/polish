#!/usr/bin/env python3
"""Build the local writing seed from pinned public-domain source checkouts.

The source repositories are deliberately kept outside the Polish repository.
This script records the exact source commit and paragraph reference in every
generated record, while the resulting JSONL remains under the ignored
``data/raw`` directory.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

# Prompt and provenance strings are intentionally kept readable as one unit.
# The functional code is still checked by Ruff's other rules.
# ruff: noqa: E501

ACQUIRED_AT = "2026-07-17"

SOURCE_SPECS: dict[str, dict[str, str]] = {
    "standard-wind": {
        "root_arg": "wind_root",
        "repo": "standardebooks/kenneth-grahame_the-wind-in-the-willows",
        "commit": "3377cb42ec6a711e3463fff957131eafbf80d547",
        "title": "The Wind in the Willows",
        "author": "Kenneth Grahame",
    },
    "standard-pride": {
        "root_arg": "pride_root",
        "repo": "standardebooks/jane-austen_pride-and-prejudice",
        "commit": "59e21f2da09be16b2bdf0bb9a94ce7911452ee55",
        "title": "Pride and Prejudice",
        "author": "Jane Austen",
    },
    "standard-alice": {
        "root_arg": "alice_root",
        "repo": "standardebooks/lewis-carroll_alices-adventures-in-wonderland",
        "commit": "dfe3e6099b7633b729bec1ce65a66997c515c845",
        "title": "Alice's Adventures in Wonderland",
        "author": "Lewis Carroll",
    },
}


EXCERPTS: tuple[dict[str, Any], ...] = (
    {
        "source_id": "standard-wind",
        "record_id": "public-wind-0001",
        "split": "train",
        "path": "src/epub/text/chapter-1.xhtml",
        "paragraph": 3,
        "task": "generation",
        "brief": "Write an 80-110 word scene about someone leaving spring cleaning to step outside.",
        "constraints": [
            "Use physical sensations.",
            "End with the character moving toward a hedge.",
        ],
        "genre": "scene",
        "tone": "bright",
    },
    {
        "source_id": "standard-wind",
        "record_id": "public-wind-0002",
        "split": "train",
        "path": "src/epub/text/chapter-1.xhtml",
        "paragraph": 5,
        "task": "generation",
        "brief": "Write a 80-110 word reflective paragraph about taking a holiday while everyone else is working.",
        "constraints": [
            "Use an outdoor setting.",
            "Let the conclusion emerge from observation rather than advice.",
        ],
        "genre": "reflection",
        "tone": "light",
    },
    {
        "source_id": "standard-wind",
        "record_id": "public-wind-0003",
        "split": "validation",
        "path": "src/epub/text/chapter-1.xhtml",
        "paragraph": 6,
        "task": "generation",
        "brief": "Write a 130-180 word first encounter with a river, treating the water as an active presence.",
        "constraints": [
            "Use sound and motion.",
            "Keep the observer's response physical.",
        ],
        "genre": "description",
        "tone": "wondering",
    },
    {
        "source_id": "standard-wind",
        "record_id": "public-wind-0004",
        "split": "validation",
        "path": "src/epub/text/chapter-1.xhtml",
        "paragraph": 7,
        "task": "generation",
        "brief": "Write a 120-160 word quiet observation of a possible home seen across a river.",
        "constraints": [
            "Begin with a small visual detail.",
            "Delay the identity of the watcher until the end.",
        ],
        "genre": "observation",
        "tone": "curious",
    },
    {
        "source_id": "standard-wind",
        "record_id": "public-wind-0005",
        "split": "holdout",
        "path": "src/epub/text/chapter-1.xhtml",
        "paragraph": 25,
        "task": "generation",
        "brief": "Write a 100-140 word conversation about taking an unplanned day on the water.",
        "constraints": [
            "Include a minor mishap.",
            "Keep the invitation open-ended.",
            "Do not summarize the lesson.",
        ],
        "genre": "dialogue",
        "tone": "easygoing",
    },
    {
        "source_id": "standard-wind",
        "record_id": "public-wind-0006",
        "split": "holdout",
        "path": "src/epub/text/chapter-1.xhtml",
        "paragraph": 50,
        "task": "generation",
        "brief": "Write a 90-130 word description of a boat entering a quiet waterway near a mill.",
        "constraints": [
            "Layer water, plant, and machine sounds.",
            "End with a brief spoken reaction.",
        ],
        "genre": "description",
        "tone": "serene",
    },
    {
        "source_id": "standard-pride",
        "record_id": "public-pride-0001",
        "split": "train",
        "path": "src/epub/text/chapter-2.xhtml",
        "paragraph": 1,
        "task": "generation",
        "brief": "Write a 60-90 word family conversation in which news of a visitor reveals competing temperaments.",
        "constraints": [
            "Use dialogue and one small domestic action.",
            "Let one speaker be dryly amused.",
        ],
        "genre": "dialogue",
        "tone": "comic",
    },
    {
        "source_id": "standard-pride",
        "record_id": "public-pride-0002",
        "split": "train",
        "path": "src/epub/text/chapter-2.xhtml",
        "paragraph": 16,
        "task": "generation",
        "brief": "Write a 55-80 word reply in which a speaker answers social caution with affectionate irony.",
        "constraints": [
            "Keep the exchange courteous on the surface.",
            "Make the final sentence turn the argument.",
        ],
        "genre": "dialogue",
        "tone": "witty",
    },
    {
        "source_id": "standard-pride",
        "record_id": "public-pride-0003",
        "split": "validation",
        "path": "src/epub/text/chapter-3.xhtml",
        "paragraph": 5,
        "task": "generation",
        "brief": "Write a 150-190 word social profile that contrasts a charming newcomer with his proud friend at a public gathering.",
        "constraints": [
            "Use the crowd's changing judgment.",
            "Include status as a social fact, not a moral lecture.",
        ],
        "genre": "profile",
        "tone": "observant",
    },
    {
        "source_id": "standard-pride",
        "record_id": "public-pride-0004",
        "split": "validation",
        "path": "src/epub/text/chapter-3.xhtml",
        "paragraph": 6,
        "task": "generation",
        "brief": "Write a 130-170 word account of two guests at a dance, contrasting one person's ease with another's reserve.",
        "constraints": [
            "Use specific actions rather than labels alone.",
            "Show how the room forms an opinion.",
        ],
        "genre": "social-observation",
        "tone": "wry",
    },
    {
        "source_id": "standard-pride",
        "record_id": "public-pride-0005",
        "split": "holdout",
        "path": "src/epub/text/chapter-3.xhtml",
        "paragraph": 14,
        "task": "generation",
        "brief": "Write a 35-50 word aftermath in which a lively person retells an awkward exchange to friends.",
        "constraints": [
            "Keep the event small.",
            "Show the storyteller's pleasure in the ridiculous.",
        ],
        "genre": "aftermath",
        "tone": "playful",
    },
    {
        "source_id": "standard-pride",
        "record_id": "public-pride-0006",
        "split": "holdout",
        "path": "src/epub/text/chapter-4.xhtml",
        "paragraph": 14,
        "task": "generation",
        "brief": "Write a 110-140 word comparison of two friends whose temperaments differ despite their loyalty.",
        "constraints": [
            "Use behavior and circumstance.",
            "Avoid turning the contrast into a simple verdict.",
        ],
        "genre": "character-sketch",
        "tone": "measured",
    },
    {
        "source_id": "standard-alice",
        "record_id": "public-alice-0001",
        "split": "train",
        "path": "src/epub/text/chapter-1.xhtml",
        "paragraph": 4,
        "task": "generation",
        "brief": "Write a 120-160 word comic scene in which an ordinary animal behaves as if it has an urgent appointment.",
        "constraints": [
            "Let the observer accept the impossible briefly.",
            "End with pursuit.",
        ],
        "genre": "comic-scene",
        "tone": "playful",
    },
    {
        "source_id": "standard-alice",
        "record_id": "public-alice-0002",
        "split": "train",
        "path": "src/epub/text/chapter-1.xhtml",
        "paragraph": 7,
        "task": "generation",
        "brief": "Write a 110-150 word passage about falling through a strange vertical space while noticing useful objects.",
        "constraints": [
            "Keep the danger understated.",
            "Include one failed attempt to use an object.",
        ],
        "genre": "fantasy-scene",
        "tone": "matter-of-fact",
    },
    {
        "source_id": "standard-alice",
        "record_id": "public-alice-0003",
        "split": "validation",
        "path": "src/epub/text/chapter-1.xhtml",
        "paragraph": 12,
        "task": "generation",
        "brief": "Write a 100-140 word chase through an unfamiliar building after a fleeting glimpse of someone ahead.",
        "constraints": [
            "Use pace and a changing interior.",
            "End in an unexpectedly empty room.",
        ],
        "genre": "chase",
        "tone": "urgent",
    },
    {
        "source_id": "standard-alice",
        "record_id": "public-alice-0004",
        "split": "validation",
        "path": "src/epub/text/chapter-1.xhtml",
        "paragraph": 15,
        "task": "generation",
        "brief": "Write a 120-160 word scene in which a person discovers a beautiful place that is physically out of reach.",
        "constraints": [
            "Use the doorway as the central image.",
            "End with a practical but impossible wish.",
        ],
        "genre": "fantasy-scene",
        "tone": "yearning",
    },
    {
        "source_id": "standard-alice",
        "record_id": "public-alice-0005",
        "split": "holdout",
        "path": "src/epub/text/chapter-2.xhtml",
        "paragraph": 2,
        "task": "generation",
        "brief": "Write a 120-160 word comic interior monologue about a person whose changing body creates practical problems.",
        "constraints": [
            "Address one body part directly.",
            "Mix worry with an absurd plan.",
        ],
        "genre": "comic-monologue",
        "tone": "inventive",
    },
    {
        "source_id": "standard-alice",
        "record_id": "public-alice-0006",
        "split": "holdout",
        "path": "src/epub/text/chapter-2.xhtml",
        "paragraph": 9,
        "task": "generation",
        "brief": "Write a 90-130 word encounter in which a frightened person asks a hurried stranger for help.",
        "constraints": [
            "Use a dropped object to interrupt the exchange.",
            "Let the stranger escape without explanation.",
        ],
        "genre": "encounter",
        "tone": "flustered",
    },
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authored-records", type=Path, required=True)
    parser.add_argument("--wind-root", type=Path, required=True)
    parser.add_argument("--pride-root", type=Path, required=True)
    parser.add_argument("--alice-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _head(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"cannot read source checkout revision at {root}") from error
    return result.stdout.strip()


def _paragraphs(path: Path) -> list[str]:
    try:
        root = ElementTree.parse(path).getroot()
    except (OSError, ElementTree.ParseError) as error:
        raise ValueError(f"cannot parse source XHTML {path}") from error
    paragraphs: list[str] = []
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] != "p":
            continue
        text = re.sub(
            r"\s+", " ", "".join(element.itertext()).replace("\u2060", "")
        ).strip()
        if text:
            paragraphs.append(text)
    return paragraphs


def _load_records(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"cannot read authored records {path}") from error
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid authored JSON on line {line_number}") from error
        if not isinstance(record, dict):
            raise ValueError(f"authored record on line {line_number} is not an object")
        records.append(record)
    return records


def _public_record(
    spec: dict[str, Any], source_roots: dict[str, Path]
) -> dict[str, Any]:
    source = SOURCE_SPECS[spec["source_id"]]
    root = source_roots[spec["source_id"]]
    actual_commit = _head(root)
    if actual_commit != source["commit"]:
        raise ValueError(
            f"source {spec['source_id']!r} is at {actual_commit}, "
            f"expected pinned commit {source['commit']}"
        )
    source_path = root / spec["path"]
    paragraphs = _paragraphs(source_path)
    paragraph = spec["paragraph"]
    if paragraph < 1 or paragraph > len(paragraphs):
        raise ValueError(f"paragraph {paragraph} is outside {source_path}")
    origin = f"https://github.com/{source['repo']}/tree/{source['commit']}"
    return {
        "record_id": spec["record_id"],
        "split": spec["split"],
        "source": {
            "id": spec["source_id"],
            "kind": "public-domain",
            "origin": origin,
            "license": "CC0 1.0 / U.S. public domain",
            "rights_note": (
                "Standard Ebooks releases its edition work under CC0 1.0; "
                "the underlying work is believed to be in the U.S. public domain."
            ),
            "acquired_at": ACQUIRED_AT,
        },
        "task": spec["task"],
        "input": {"brief": spec["brief"], "constraints": spec["constraints"]},
        "target": paragraphs[paragraph - 1],
        "metadata": {
            "genre": spec["genre"],
            "tone": spec["tone"],
            "source_title": source["title"],
            "source_author": source["author"],
            "source_repo": source["repo"],
            "source_commit": source["commit"],
            "source_path": spec["path"],
            "source_paragraph": str(paragraph),
            "source_transform": "single paragraph; whitespace and word-joiner normalization only",
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    roots = {
        "standard-wind": args.wind_root,
        "standard-pride": args.pride_root,
        "standard-alice": args.alice_root,
    }
    try:
        authored = _load_records(args.authored_records)
        public_records = [_public_record(spec, roots) for spec in EXCERPTS]
        records = authored + public_records
        record_ids = [record["record_id"] for record in records]
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("combined records contain duplicate record IDs")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "".join(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                for record in records
            ),
            encoding="utf-8",
        )
    except ValueError as error:
        _parser().error(str(error))
    print(f"wrote {len(records)} records to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
