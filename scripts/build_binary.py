#!/usr/bin/env python3
"""Build a single standalone ``verity`` executable using PyInstaller.

The resulting executable embeds:
- Python runtime + ``verity`` orchestrator + ``verity_api`` FastAPI server
- Pre-built React Web Console SPA (``src/verity_ui/dist``)
- All 58 detector definitions, scripts, ``detectors/common`` and
  ``weights_manifest.yaml`` (no binary checkpoints)
- OpenCV Haar cascades for deterministic face detection

At runtime the binary stores environments, verified checkpoints, caches and
cases cleanly via ``platformdirs`` (or ``$VERITY_HOME``), bootstraps ``uv``
and managed Python runtimes on ``verity setup``, and can then run 100%
air-gapped.

Usage:
    uv run python scripts/build_binary.py
    uv run python scripts/build_binary.py --out dist-release --rebuild-ui
"""
from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stage_detectors(src: Path, dst: Path) -> int:
    """Copy detector source code, specs and lock/project configs (no weight binaries)."""
    count = 0
    for item in sorted(src.rglob("*")):
        rel = item.relative_to(src)
        if "__pycache__" in rel.parts or item.suffix in (".pyc", ".pyo"):
            continue
        # Never bundle checkpoint binaries into the executable
        if "weights" in rel.parts and item.is_file() and item.name != "README.md":
            continue
        if item.is_dir():
            (dst / rel).mkdir(parents=True, exist_ok=True)
        elif item.is_file():
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="", help="Release version (default: pyproject.toml)")
    parser.add_argument("--out", default=str(ROOT / "dist-release"), help="Output directory")
    parser.add_argument("--rebuild-ui", action="store_true", help="Force rebuilding the React UI")
    args = parser.parse_args()

    version = args.version.lstrip("v")
    if not version:
        with open(ROOT / "pyproject.toml", "rb") as f:
            version = tomllib.load(f)["project"]["version"]

    ui_dist = ROOT / "src" / "verity_ui" / "dist"
    if args.rebuild_ui or not (ui_dist / "index.html").is_file():
        print("[build] building Web UI SPA...")
        subprocess.run(["npm", "--prefix", str(ROOT / "src" / "verity_ui"), "run", "build"], check=True)

    sys_name = platform.system().lower()
    machine = platform.machine().lower()
    if machine in ("amd64", "x86_64"):
        arch = "x86_64"
    elif machine in ("arm64", "aarch64"):
        arch = "aarch64"
    else:
        arch = machine

    sep = ";" if os.name == "nt" else ":"
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Keep temporary PyInstaller build artifacts on the target filesystem
    # (not /tmp, which is often a small RAM-backed tmpfs on Linux).
    with tempfile.TemporaryDirectory(prefix=".pyinstaller_", dir=str(out_dir)) as tmp:
        tmp_path = Path(tmp)
        staged_detectors = tmp_path / "detectors"
        n_files = stage_detectors(ROOT / "detectors", staged_detectors)
        print(f"[build] staged {n_files} detector source/spec files (0 weight binaries)")

        work_dir = tmp_path / "work"
        dist_dir = tmp_path / "dist"
        spec_dir = tmp_path / "spec"

        cmd = [
            sys.executable,
            "-m",
            "PyInstaller",
            "--onefile",
            "--clean",
            "--noconfirm",
            "--name",
            "verity",
            "--workpath",
            str(work_dir),
            "--distpath",
            str(dist_dir),
            "--specpath",
            str(spec_dir),
            "--paths",
            str(ROOT / "src"),
            "--add-data",
            f"{staged_detectors}{sep}detectors",
            "--add-data",
            f"{ui_dist}{sep}verity_ui_dist",
            "--add-data",
            f"{ROOT / 'config'}{sep}config",
            "--collect-data",
            "cv2",
            "--collect-data",
            "weasyprint",
            "--collect-submodules",
            "verity",
            "--collect-submodules",
            "verity_api",
            "--collect-submodules",
            "uvicorn",
            "--collect-submodules",
            "fastapi",
            "--collect-submodules",
            "starlette",
            "--collect-submodules",
            "pydantic",
            "--hidden-import",
            "yaml",
            "--hidden-import",
            "platformdirs",
            "--hidden-import",
            "filetype",
            "--hidden-import",
            "matplotlib.backends.backend_agg",
            "--hidden-import",
            "multipart",
            "--hidden-import",
            "python_multipart",
            "--hidden-import",
            "email.mime.multipart",
            "--hidden-import",
            "email.mime.text",
            str(ROOT / "packaging" / "verity_entry.py"),
        ]
        print("[build] running PyInstaller...")
        subprocess.run(cmd, check=True)

        built_name = "verity.exe" if os.name == "nt" else "verity"
        built_bin = dist_dir / built_name
        if not built_bin.is_file():
            raise RuntimeError(f"Expected binary not produced: {built_bin}")

        ext = ".exe" if os.name == "nt" else ""
        stable = out_dir / f"verity-{sys_name}-{arch}{ext}"
        pinned = out_dir / f"verity-{version}-{sys_name}-{arch}{ext}"
        shutil.copy2(built_bin, pinned)
        shutil.copy2(built_bin, stable)
        pinned.chmod(0o755)
        stable.chmod(0o755)

        (out_dir / f"{stable.name}.sha256").write_text(f"{sha256(stable)}  {stable.name}\n")
        (out_dir / f"{pinned.name}.sha256").write_text(f"{sha256(pinned)}  {pinned.name}\n")

        size_mb = pinned.stat().st_size / (1024 * 1024)
        print(f"[build] wrote {pinned} ({size_mb:.1f} MB)")
        print(f"[build] wrote {stable}")
        print(f"[build] wrote {stable.name}.sha256 & {pinned.name}.sha256")
        print(f"[build] sha256: {sha256(pinned)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
