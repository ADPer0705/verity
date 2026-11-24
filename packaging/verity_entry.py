"""Standalone executable entry point for PyInstaller builds of Verity."""
from __future__ import annotations

import multiprocessing
import sys


def main() -> None:
    multiprocessing.freeze_support()
    from verity.cli import cli

    sys.exit(cli(prog_name="verity"))


if __name__ == "__main__":
    main()
