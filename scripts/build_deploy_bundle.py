#!/usr/bin/env python3
"""Assemble the lab deployment bundle attached to a GitHub release.

    python scripts/build_deploy_bundle.py --version 0.5.0

Produces dist-deploy/verity-deploy-<version>.tar.gz plus a stable
verity-deploy.tar.gz alias and SHA256SUMS. The bundle contains the compose
files, installer, doctor, systemd unit and deployment documentation; no
Python packages, checkpoints or images are included.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import tarfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dist-deploy"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="", help="Release version (default: pyproject.toml)")
    parser.add_argument("--out", default=str(OUT), help="Output directory")
    args = parser.parse_args()

    version = args.version.lstrip("v")
    if not version:
        with open(ROOT / "pyproject.toml", "rb") as f:
            version = tomllib.load(f)["project"]["version"]

    out = Path(args.out)
    stage = out / "verity-deploy"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    files = {
        "docker-compose.yml": "docker-compose.yml",
        "docker-compose.cpu.yml": "docker-compose.cpu.yml",
        ".env.example": ".env.example",
        "install.sh": "deploy/install.sh",
        "doctor.sh": "deploy/doctor.sh",
        "verity.service": "deploy/verity.service",
        "README.md": "deploy/README.md",
        "LICENSE": "LICENSE",
        "THIRD_PARTY_NOTICES.md": "THIRD_PARTY_NOTICES.md",
        "deployment.md": "docs/deployment.md",
    }
    missing = [src for src in files.values() if not (ROOT / src).exists()]
    if missing:
        raise SystemExit(f"missing bundle inputs: {', '.join(missing)}")

    for dest, src in files.items():
        shutil.copy2(ROOT / src, stage / dest)

    (stage / "VERSION").write_text(version + "\n")
    (stage / "install.sh").chmod(0o755)
    (stage / "doctor.sh").chmod(0o755)

    out.mkdir(parents=True, exist_ok=True)
    archive = out / f"verity-deploy-{version}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(stage, arcname="verity-deploy")

    stable = out / "verity-deploy.tar.gz"
    shutil.copy2(archive, stable)

    checksums = out / "SHA256SUMS"
    checksums.write_text(f"{sha256(stable)}  {stable.name}\n")
    versioned_checksums = out / f"SHA256SUMS-{version}"
    versioned_checksums.write_text(f"{sha256(archive)}  {archive.name}\n")

    print(f"wrote {archive} ({archive.stat().st_size / 1024:.0f} KB)")
    print(f"wrote {stable}")
    print(f"wrote {checksums} (stable alias)")
    print(f"wrote {versioned_checksums} (pinned archive)")


if __name__ == "__main__":
    main()
