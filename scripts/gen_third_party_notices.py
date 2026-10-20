#!/usr/bin/env python3
"""Regenerate THIRD_PARTY_NOTICES.md's checkpoint tables from the weights manifest.

    uv run python scripts/gen_third_party_notices.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from verity.weights import load_manifest  # noqa: E402

NOTICE = ROOT / "THIRD_PARTY_NOTICES.md"


def main() -> None:
    m = load_manifest(ROOT / "detectors" / "weights_manifest.yaml")
    rows = []
    for f in sorted(m.files.values(), key=lambda f: (f.source, f.name)):
        src = m.sources.get(f.source, {})
        rows.append(f"| `{f.name}` | {src.get('name', f.source)} | {f.effective_license(m)} | "
                    f"{'mirrored' if f.redistributable else 'authors’ URL only'} | `{f.sha256[:12]}…` |")
    weights_table = "| File | Source | License | Distribution | SHA-256 |\n|---|---|---|---|---|\n" + "\n".join(rows)
    runtime = "| Detector | Models and licenses |\n|---|---|\n" + "\n".join(
        f"| `{k}` | {v} |" for k, v in sorted(m.runtime_fetched.items()))

    text = NOTICE.read_text()
    text = re.sub(r"(## Checkpoints fetched by `verity weights pull`\n\n)\|.*?\|\n(?=\n)",
                  lambda mo: mo.group(1) + weights_table + "\n", text, flags=re.S)
    text = re.sub(r"(## Models fetched at first run by the detectors themselves\n\n)\|.*?\|\n(?=\n)",
                  lambda mo: mo.group(1) + runtime + "\n", text, flags=re.S)
    NOTICE.write_text(text)
    print(f"updated {NOTICE.name}: {len(rows)} checkpoint rows, {len(m.runtime_fetched)} runtime rows")


if __name__ == "__main__":
    main()
