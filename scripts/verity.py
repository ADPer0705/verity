#!/usr/bin/env python3
"""Verity detector environment CLI shim.

Delegates to ``verity.runtime.env_manager`` so development checkouts, Docker
containers, and the standalone PyInstaller binary all share one implementation.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from verity.runtime.env_manager import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
