"""Command-line entry point for the Polish research toolkit."""

from argparse import ArgumentParser
from collections.abc import Sequence
from pathlib import Path

from polish import __version__
from polish.evaluation import (
    EvaluationError,
    EvaluationRunner,
    GenerationSettings,
    PromptManifest,
    RecordedOutputAdapter,
    Split,
)


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="polish",
        description="Research tooling for improving local language-model writing.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command")
    evaluate = subparsers.add_parser(
        "evaluate", help="build a report from recorded local-model outputs"
    )
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument(
        "--recorded-output", type=Path, action="append", required=True
    )
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument(
        "--split",
        choices=[split.value for split in Split],
        default=Split.HOLDOUT.value,
    )
    evaluate.add_argument("--max-new-tokens", type=int, default=512)
    evaluate.add_argument("--temperature", type=float, default=0.7)
    evaluate.add_argument("--top-p", type=float, default=0.95)
    evaluate.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a Polish command and return its process exit code."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command != "evaluate":
        parser.print_help()
        return 0

    try:
        manifest = PromptManifest.from_path(args.manifest)
        split = Split(args.split)
        settings = GenerationSettings(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
        )
        adapters = tuple(
            RecordedOutputAdapter.from_path(path) for path in args.recorded_output
        )
        for adapter in adapters:
            adapter.validate_contract(manifest, split, settings)
        report = EvaluationRunner().run(
            manifest, adapters, split=split, settings=settings
        )
        report.write(args.output)
    except EvaluationError as error:
        parser.error(str(error))
    print(f"wrote evaluation report {args.output} ({report.report_id})")
    return 0
