#!/usr/bin/env python3
"""Assemble a Verity release.

Usage:
    python scripts/release.py v0.3.0            # build wheel
    python scripts/release.py v0.3.0 --no-build # skip uv build (use existing dist/)

Produces dist-bundle/<tag>/ containing:
    verity-<version>-py3-none-any.whl   the installable package (code + detector definitions)
    sha256sums.txt                      checksums for verification
"""
import argparse
import hashlib
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
BUNDLE_ROOT = ROOT / "dist-bundle"


def _run(cmd: list[str], cwd: Path) -> None:
    print(f"  $ {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=cwd)


def version_from_tag(tag: str) -> str:
    return tag[1:] if tag.startswith("v") else tag


def pyproject_version() -> str:
    with open(ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]["version"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag", help="Release tag, e.g. v0.2.0")
    parser.add_argument("--no-build", action="store_true", help="Skip uv build")
    args = parser.parse_args()

    version = version_from_tag(args.tag)
    if pyproject_version() != version:
        print(
            f"[error] pyproject version is {pyproject_version()}, tag implies {version}. "
            f"Bump pyproject.toml first.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not args.no_build:
        print("[build] building wheel")
        _run(["uv", "build"], cwd=ROOT)

    wheel = DIST / f"verity-{version}-py3-none-any.whl"
    if not wheel.exists():
        print(f"[error] wheel not found: {wheel}", file=sys.stderr)
        sys.exit(1)

    bundle = BUNDLE_ROOT / args.tag
    bundle.mkdir(parents=True, exist_ok=True)
    shutil.copy2(wheel, bundle / wheel.name)

    manifest = [(sha256(bundle / wheel.name), wheel.name)]

    manifest_path = bundle / "sha256sums.txt"
    with open(manifest_path, "w") as f:
        for digest, name in sorted(manifest, key=lambda kv: kv[1]):
            f.write(f"{digest}  {name}\n")

    total = sum(p.stat().st_size for p in bundle.iterdir() if p.is_file())
    print(f"[done]  {bundle}  ({total / 1e6:.1f} MB)")
    print(f"        manifest: {manifest_path}")
    print()
    print("Deploy (air-gapped or otherwise):")
    print(f"  1. copy {bundle} to the target")
    print(f"  2. cd {bundle.name} && sha256sum -c sha256sums.txt")
    print(f"  3. uv tool install verity-{version}-py3-none-any.whl")


if __name__ == "__main__":
    main()
