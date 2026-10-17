"""Small command-line entry point for the packaged demonstration."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from lensuq import __version__


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested command, returning a process exit status."""
    parser = argparse.ArgumentParser(
        prog="lensuq",
        description="Audit prediction sensitivity to preprocessing choices.",
    )
    parser.add_argument("--version", action="version", version=f"LensUQ {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "demo", help="Run the offline artificial-missingness experiment"
    )
    parser.parse_args(argv)
    from lensuq.demo import run_demo

    run_demo()
    return 0
