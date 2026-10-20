"""Assemble the Hugging Face upload for the redistributable checkpoints.

Nothing here talks to the network. It lays out ``dist-weights/<repo>/`` so
that a single ``hf upload`` publishes a complete, attributed, checksummed
model repo; the commands to run are printed at the end.
"""
from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

from verity.weights.fetch import sha256_of
from verity.weights.manifest import FileSpec, Manifest


def _copy_or_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.unlink(missing_ok=True)
    try:
        import os
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _local_source(manifest: Manifest, detectors_dir: Path, spec: FileSpec) -> Optional[Path]:
    """A verified local copy of ``spec`` from any detector that uses it."""
    from verity.weights.fetch import dest_path
    for targets in manifest.detectors.values():
        for t in targets:
            if t.spec.name != spec.name:
                continue
            p = dest_path(detectors_dir, t)
            if p.is_file() and p.stat().st_size == spec.size and sha256_of(p) == spec.sha256:
                return p
    return None


def _model_card(manifest: Manifest, repo: str, files: List[FileSpec]) -> str:
    by_source: Dict[str, List[FileSpec]] = {}
    for f in files:
        by_source.setdefault(f.source, []).append(f)

    rows = "\n".join(
        f"| `{f.hub_path}` | {f.size / 1e6:.1f} MB | {f.effective_license(manifest)} | "
        f"{f.description} | `{f.sha256[:16]}…` |"
        for f in files
    )
    sources = ""
    for key, specs in by_source.items():
        src = manifest.sources.get(key, {})
        sources += (
            f"\n### {src.get('name', key)}\n\n"
            f"- Upstream: {src.get('url', '')}\n"
            f"- License: {src.get('license', '')}\n"
            + (f"- {src['copyright']}\n" if src.get("copyright") else "")
            + (f"- Citation: {src['citation'].strip()}\n" if src.get("citation") else "")
            + (f"- Note: {src['note'].strip()}\n" if src.get("note") else "")
            + "- Files: " + ", ".join(f"`{s.hub_path}`" for s in specs) + "\n"
        )

    return f"""---
license: cc-by-nc-4.0
license_name: cc-by-nc-4.0-plus-upstream
license_link: LICENSE
tags:
  - deepfake-detection
  - media-forensics
  - deepfakebench
  - verity
library_name: pytorch
---

# Verity detector checkpoints

Mirror of the pretrained checkpoints used by **Verity**, an open deepfake
examination pipeline for investigators and researchers. Verity downloads the
files below on first run (`verity weights pull`) and verifies each one
against the SHA-256 in `manifest.json` before it is used.

**This repo re-hosts other people's work.** Every file keeps the license its
authors released it under; the repo-level tag is the most restrictive one
(CC BY-NC 4.0). Nothing here may be used commercially. See `NOTICE.md` for
per-file attribution and `LICENSE` for the license texts.

Checkpoints that upstream authors did not license for redistribution (NPR,
MLEP) are **not** in this repo; Verity fetches those from the authors'
repositories directly.

Assembled {date.today().isoformat()} from Verity's `detectors/weights_manifest.yaml`.

## Files

| Path | Size | License | What it is | SHA-256 |
|---|---|---|---|---|
{rows}

## Sources
{sources}
## Use

```bash
pip install -U "huggingface_hub[cli]"
hf download {repo} --local-dir weights
# or let Verity do it:
VERITY_WEIGHTS_HUB={repo} verity weights pull --trained
```

These models are statistical indicators, not proof. Their outputs must be
interpreted by a qualified examiner alongside the explainability artifacts
Verity produces, and never cited as a standalone determination.
"""


def _notice(manifest: Manifest, files: List[FileSpec]) -> str:
    lines = ["# Third-party notices", "",
             "Each file in this repository is the work of its upstream authors and is",
             "redistributed under the license stated below, unmodified.", ""]
    for f in files:
        src = manifest.sources.get(f.source, {})
        lines += [f"## {f.hub_path}", "",
                  f"- Origin: {f.origin}" + (f" (member `{f.archive.member}`)" if f.archive else ""),
                  f"- License: {f.effective_license(manifest)}",
                  *( [f"- {src['copyright']}"] if src.get("copyright") else [] ),
                  *( [f"- Citation: {src['citation'].strip()}"] if src.get("citation") else [] ),
                  f"- SHA-256: `{f.sha256}`", ""]
    return "\n".join(lines)


LICENSE_TEXTS = {
    "CC-BY-NC-4.0": "Creative Commons Attribution-NonCommercial 4.0 International.\n"
                    "Full text: https://creativecommons.org/licenses/by-nc/4.0/legalcode\n",
    "CC-BY-NC-SA-4.0": "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International.\n"
                       "Full text: https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode\n",
    "MIT": "MIT License. Full text: https://opensource.org/license/mit\n",
    "BSD-3-Clause": "BSD 3-Clause License. Full text: https://opensource.org/license/bsd-3-clause\n",
    "Apache-2.0": ("Apache License 2.0.\n"
                   "Copyright 1999-2025 the respective authors.\n"
                   "Full text: https://www.apache.org/licenses/LICENSE-2.0.txt\n"),
}


def build_bundle(manifest: Manifest, detectors_dir: Path, out_dir: Path, repo: str) -> Dict[str, object]:
    files = manifest.redistributable_files()
    out_dir.mkdir(parents=True, exist_ok=True)
    copied, missing = [], []
    for spec in files:
        src = _local_source(manifest, detectors_dir, spec)
        if src is None:
            missing.append(spec.hub_path)
            continue
        _copy_or_link(src, out_dir / spec.hub_path)
        copied.append(spec.hub_path)

    (out_dir / "README.md").write_text(_model_card(manifest, repo, files))
    (out_dir / "NOTICE.md").write_text(_notice(manifest, files))
    licenses = sorted({f.effective_license(manifest) for f in files})
    (out_dir / "LICENSE").write_text(
        "This repository redistributes files under the following licenses; the\n"
        "license of each file is stated in NOTICE.md and manifest.json.\n\n"
        + "\n".join(f"== {lic} ==\n{LICENSE_TEXTS.get(lic, '')}" for lic in licenses))
    (out_dir / "manifest.json").write_text(json.dumps({
        "version": manifest.version,
        "repo": repo,
        "files": {f.hub_path: {"sha256": f.sha256, "size": f.size, "source": f.source,
                               "license": f.effective_license(manifest), "origin": f.origin,
                               "description": f.description} for f in files},
    }, indent=2))
    (out_dir / "sha256sums.txt").write_text(
        "".join(f"{f.sha256}  {f.hub_path}\n" for f in files))
    (out_dir / ".gitattributes").write_text("*.pth filter=lfs diff=lfs merge=lfs -text\n")
    return {"out_dir": out_dir, "copied": copied, "missing": missing,
            "total_bytes": sum(f.size for f in files if f.hub_path in copied)}
