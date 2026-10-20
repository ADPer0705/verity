#!/usr/bin/env python3
"""Verity detector environment CLI (uv-backed, hardware-adaptive).

Runs inside the Verity container (or natively on a dev machine): detects
whether a GPU is available, creates per-detector uv virtual environments,
installs the exact pinned dependencies, and launches detector inference.

Hardware profiles
-----------------
auto   (default)   GPU when `nvidia-smi` works, else CPU
cuda               PyPI wheels (CUDA-enabled torch builds)
cpu                +cpu torch/torchvision wheels from download.pytorch.org

The profile is decided at runtime, so the same container image works on GPU
and CPU machines. Override with VERITY_HW_PROFILE=cuda|cpu|auto.

Layout
------
$VERITY_DETECTOR_ENVS/<profile>/<detector_id>/    per-profile project copy:
                                                  pyproject.toml, uv.lock, .venv
                                                  (isolated per profile)
$VERITY_DETECTOR_ENVS/cache/                      uv package cache (shared, hardlinked)
$VERITY_WEIGHTS_ROOT/<detector_id>/               weights (symlinked into package)

Each detector's pyproject.toml (detectors/<id>/pyproject.toml) declares the
environment that fits that detector alone: its own Python version and only
the dependencies its code actually imports (plus the detectors/common
runtime closure). On the cpu profile the project copy is generated with
torch/torchvision pinned to +cpu builds ([tool.uv.override-dependencies])
plus the pytorch cpu index, so a plain `uv sync` needs no extra flags and
works on any uv version.

Subcommands
-----------
verity.py profile [detector_id]
verity.py setup [--detector ID[,ID...]] [--trained] [--all] [--force] [--concurrency N]
verity.py run <detector_id> [input_dir] [output_dir]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DETECTORS = ROOT / "detectors"
CPU_INDEX = "https://download.pytorch.org/whl/cpu"

PIN = re.compile(r"^(torch|torchvision)(==[\d.]+)$")


def log(msg: str) -> None:
    print(f"[verity] {msg}", flush=True)


def detect_profile(requested: str | None) -> str:
    if requested in ("cuda", "cpu"):
        return requested
    if requested not in (None, "", "auto"):
        log(f"WARN unknown VERITY_HW_PROFILE={requested!r}, treating as auto")
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi and subprocess.run([nvidia_smi], capture_output=True).returncode == 0:
        return "cuda"
    return "cpu"


def detector_envs_root() -> Path:
    root = Path(os.environ.get("VERITY_DETECTOR_ENVS", ROOT / ".detectorenvs"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def baked_envs_root() -> Path | None:
    """Read-only environments baked into the container image
    (``VERITY_DETECTOR_ENVS_BAKED``). Used when they match the detector's
    current pyproject; anything else is provisioned under the writable root."""
    env = os.environ.get("VERITY_DETECTOR_ENVS_BAKED", "")
    if env and Path(env).is_dir():
        return Path(env)
    return None


def project_dir(det_id: str, profile: str) -> Path:
    """The uv project directory to run ``det_id`` from: the baked copy when it
    is up to date, else the writable per-profile copy."""
    baked = baked_envs_root()
    if baked is not None:
        proj = baked / profile / det_id
        marker = proj / ".verity-synced"
        want = hashlib.sha256(project_pyproject(det_id, profile).encode()).hexdigest()
        if marker.exists() and marker.read_text() == want and (proj / ".venv").exists():
            return proj
    return detector_envs_root() / profile / det_id


def weights_root() -> Path | None:
    env = os.environ.get("VERITY_WEIGHTS_ROOT", "")
    if not env:
        return None
    root = Path(env)
    root.mkdir(parents=True, exist_ok=True)
    return root


def detector_ids() -> list[str]:
    return sorted(p.name for p in DETECTORS.iterdir() if (p / "pyproject.toml").exists())


def trained_detector_ids() -> list[str]:
    """Detectors whose classification head is trained (see detector.yaml
    ``weights.kind`` / ``weights.trained``). Mirrors verity.detect.base_detector."""
    import yaml  # available in the orchestrator env; only needed for this filter

    ids = []
    for det_id in detector_ids():
        spec = yaml.safe_load((DETECTORS / det_id / "detector.yaml").read_text()) or {}
        weights = spec.get("weights") or {}
        trained = weights.get("trained")
        if trained is None:
            trained = weights.get("kind", "official") == "official"
        if trained:
            ids.append(det_id)
    return ids


def cpu_pyproject(src_text: str) -> str:
    """Add +cpu override-dependencies and the pytorch cpu index to a copy of
    the project's pyproject.toml. Torch/torchvision pins are mapped 1:1 to
    their +cpu equivalents (2.2.2 -> 2.2.2+cpu, 2.8.0 -> 2.8.0+cpu, ...)."""
    data = tomllib.loads(src_text)
    overrides: dict[str, str] = {}
    for dep in data["project"]["dependencies"]:
        m = PIN.match(dep)
        if m:
            overrides[m.group(1)] = f"{m.group(2)}+cpu"
    if not overrides:
        log("WARN cpu profile: no torch/torchvision pins found to override")
    src_text = src_text.replace("[tool.uv]\npackage = false\n", "", 1)
    deps_override = "[" + ", ".join(f'"{pkg}{spec}"' for pkg, spec in sorted(overrides.items())) + "]"
    lines = ["# Generated by scripts/verity.py: cpu hardware profile copy.",
             "# Adds +cpu torch/torchvision wheels from download.pytorch.org.", ""]
    lines += ["[tool.uv]", "package = false", f"override-dependencies = {deps_override}", ""]
    lines += ["[[tool.uv.index]]", 'name = "pytorch-cpu"', f'url = "{CPU_INDEX}"', ""]
    return src_text.rstrip() + "\n\n" + "\n".join(lines) + "\n"


def project_pyproject(det_id: str, profile: str) -> str:
    src = (DETECTORS / det_id / "pyproject.toml").read_text()
    return cpu_pyproject(src) if profile == "cpu" else src


def sync_one(det_id: str, profile: str, force: bool = False) -> bool:
    want = project_pyproject(det_id, profile)
    want_hash = hashlib.sha256(want.encode()).hexdigest()
    if not force:
        baked_proj = project_dir(det_id, profile)
        if baked_envs_root() is not None and baked_proj.is_relative_to(baked_envs_root()):
            return True  # baked and current: nothing to do
    proj_dir = detector_envs_root() / profile / det_id
    proj_dir.mkdir(parents=True, exist_ok=True)
    marker = proj_dir / ".verity-synced"
    if not force and marker.exists() and marker.read_text() == want_hash:
        return True

    (proj_dir / "pyproject.toml").write_text(want)
    if (proj_dir / "uv.lock").exists():
        (proj_dir / "uv.lock").unlink()

    env = dict(os.environ)
    env["UV_CACHE_DIR"] = str(detector_envs_root() / "cache")
    env.setdefault("UV_LINK_MODE", "hardlink")
    env.setdefault("UV_NO_PROGRESS", "1")

    log(f"syncing {det_id} ({profile})")
    result = subprocess.run(["uv", "sync", "--project", str(proj_dir)], env=env)
    if result.returncode != 0:
        log(f"FAILED to sync {det_id} ({profile})")
        return False
    marker.write_text(hashlib.sha256(want.encode()).hexdigest())
    return True


def ensure_weights_links(det_id: str) -> None:
    """In the container, weights/ is a symlink to the /weights volume so the
    download script (curl into detectors/<id>/weights/...) persists across runs.
    Local dev (no VERITY_WEIGHTS_ROOT) keeps weights in the package."""
    root = weights_root()
    if root is None:
        return
    wdir = DETECTORS / det_id / "weights"
    target = root / det_id
    if wdir.is_symlink():
        return
    target.mkdir(parents=True, exist_ok=True)
    readme = wdir / "README.md"
    if readme.exists() and not (target / "README.md").exists():
        shutil.copy2(readme, target / "README.md")
    if wdir.exists():
        shutil.rmtree(wdir)
    wdir.symlink_to(target, target_is_directory=True)
    log(f"weights: {wdir.relative_to(ROOT)} -> {target}")


def cmd_setup(args) -> int:
    profile = detect_profile(args.profile)
    log(f"hardware profile: {profile}")
    if args.detector:
        ids = [d.strip() for d in args.detector.split(",") if d.strip()]
    elif args.all:
        ids = detector_ids()
    elif args.trained:
        ids = trained_detector_ids()
    else:
        ids = []
    unknown = sorted(set(ids) - set(detector_ids()))
    if unknown:
        log(f"unknown detector(s): {', '.join(unknown)}")
        return 2
    if not ids:
        log("no detectors selected (use --detector ID[,ID...], --trained, --all, or run <id>)")
        return 2
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(sync_one, det_id, profile, args.force): det_id for det_id in ids}
        for fut in futures:
            det_id = futures[fut]
            if not fut.result():
                failed.append(det_id)
    for det_id in ids:
        ensure_weights_links(det_id)
    if failed:
        log(f"failed: {', '.join(failed)}")
        return 1
    log(f"setup complete: {len(ids)} detector(s) ready on {profile}")
    return 0


def cmd_run(args) -> int:
    profile = detect_profile(args.profile)
    if not sync_one(args.detector, profile):
        return 1
    ensure_weights_links(args.detector)

    # Resolve to absolute paths: the engine runs with cwd=detectors/<id>, so a
    # relative INPUT_DIR/OUTPUT_DIR (as passed by the orchestrator, whose cwd
    # is the repo root) would silently write into the detector package.
    input_dir = os.path.abspath(args.input_dir or os.environ.get("INPUT_DIR", "/input"))
    output_dir = os.path.abspath(args.output_dir or os.environ.get("OUTPUT_DIR", "/output"))
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    env = dict(os.environ)
    env.update({"INPUT_DIR": input_dir, "OUTPUT_DIR": output_dir, "VERITY_HW_PROFILE": profile})
    env["UV_CACHE_DIR"] = str(detector_envs_root() / "cache")
    env.setdefault("UV_NO_PROGRESS", "1")

    proj_dir = project_dir(args.detector, profile)
    cmd = ["uv", "run", "--project", str(proj_dir), "--no-sync", "python",
           str(DETECTORS / args.detector / "inference.py")]
    log(f"running {args.detector} on {profile} ({input_dir} -> {output_dir})")
    return subprocess.run(cmd, env=env, cwd=DETECTORS / args.detector).returncode


def cmd_profile(args) -> int:
    print(detect_profile(args.profile))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", choices=["auto", "cuda", "cpu"],
                        default=os.environ.get("VERITY_HW_PROFILE", "auto"),
                        help="hardware profile (default: $VERITY_HW_PROFILE or auto-detect)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_setup = sub.add_parser("setup", help="create/refresh detector environments")
    p_setup.add_argument("--detector", help="sync one or more detectors (comma-separated)")
    p_setup.add_argument("--all", action="store_true", help="sync all detectors")
    p_setup.add_argument("--trained", action="store_true",
                         help="sync only detectors with a trained classification head")
    p_setup.add_argument("--force", action="store_true", help="re-sync even if up to date")
    p_setup.add_argument("--concurrency", type=int, default=8)

    p_run = sub.add_parser("run", help="run one detector")
    p_run.add_argument("detector")
    p_run.add_argument("input_dir", nargs="?", help="default: $INPUT_DIR or /input")
    p_run.add_argument("output_dir", nargs="?", help="default: $OUTPUT_DIR or /output")

    sub.add_parser("profile", help="print detected hardware profile")

    args = parser.parse_args()
    if args.command == "setup":
        return cmd_setup(args)
    if args.command == "run":
        return cmd_run(args)
    return cmd_profile(args)


if __name__ == "__main__":
    sys.exit(main())
