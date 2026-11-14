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
    demo = subparsers.add_parser("demo", help="Run an offline demonstration")
    demo.add_argument(
        "name",
        nargs="?",
        choices=("breast-cancer", "wine", "regression"),
        default="breast-cancer",
    )
    subparsers.add_parser("info", help="Show version and supported experiments")
    args = parser.parse_args(argv)
    if args.command == "info":
        print(
            f"LensUQ {__version__}\n"
            "Tasks: binary classification, multiclass classification, regression\n"
            "Demos: breast-cancer, wine, regression"
        )
    elif args.name == "breast-cancer":
        from lensuq.demo import run_demo

        run_demo()
    else:
        from lensuq.extended_demos import run_extended_demo

        run_extended_demo(args.name)
    return 0
