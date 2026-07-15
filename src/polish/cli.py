"""Command-line entry point for the Polish research toolkit."""

from argparse import ArgumentParser

from polish import __version__


def main() -> None:
    """Run the minimal project command until research workflows land."""
    parser = ArgumentParser(
        prog="polish",
        description="Research tooling for improving local language-model writing.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.parse_args()
