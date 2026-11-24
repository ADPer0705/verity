"""Verity detector environment manager (uv-backed, hardware-adaptive, air-gap safe).

Runs directly in-process inside the standalone ``verity`` binary as well as via
``scripts/verity.py`` in development and Docker containers:
- Auto-bootstraps the official standalone ``uv`` binary into ``<data_dir>/bin/uv``
  if ``uv`` is not already present on the host.
- Uses uv-managed standalone Python builds in ``<data_dir>/python`` so detector
  environments are 100% independent of host system Python.
- Keeps per-detector ``uv.lock`` files so ``verity airgap import`` can recreate
  every virtual environment offline (``uv sync --offline --no-python-downloads``)
  via hardlinks from the local wheel cache in seconds.
- Automatically enforces ``HF_HUB_OFFLINE=1``, ``TRANSFORMERS_OFFLINE=1``,
  ``DIFFUSERS_OFFLINE=1``, and ``UV_OFFLINE=1`` whenever Verity is in offline
  mode.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, List, Optional

import yaml

from verity import paths

CPU_INDEX = "https://download.pytorch.org/whl/cpu"
PIN = re.compile(r"^(torch|torchvision)(==[\d.]+)$")

LogFn = Callable[[str], None]


def _default_log(msg: str) -> None:
    print(f"[verity] {msg}", flush=True)


# --------------------------------------------------------------------------- #
#  Tool bootstrapping (uv + optional static ffmpeg/ffprobe)                   #
# --------------------------------------------------------------------------- #


def find_uv() -> Optional[str]:
    """Return path to ``uv`` (checking Verity's private bin/ first, then PATH)."""
    paths.ensure_runtime_path()
    private = paths.bin_dir() / ("uv.exe" if os.name == "nt" else "uv")
    if private.is_file() and os.access(private, os.X_OK):
        return str(private)
    return shutil.which("uv")


def _uv_release_url() -> str:
    sys_name = platform.system().lower()
    machine = platform.machine().lower()
    if sys_name == "linux":
        arch = "aarch64" if machine in ("aarch64", "arm64") else "x86_64"
        return (
            f"https://github.com/astral-sh/uv/releases/latest/download/"
            f"uv-{arch}-unknown-linux-gnu.tar.gz"
        )
    if sys_name == "darwin":
        arch = "aarch64" if machine in ("aarch64", "arm64") else "x86_64"
        return (
            f"https://github.com/astral-sh/uv/releases/latest/download/"
            f"uv-{arch}-apple-darwin.tar.gz"
        )
    raise RuntimeError(f"Automatic uv bootstrap is not supported on {sys_name}/{machine}")


def ensure_uv(allow_download: bool = True, log: LogFn = _default_log) -> str:
    """Locate ``uv`` or download the official Astral binary into ``<data_dir>/bin``."""
    existing = find_uv()
    if existing:
        return existing
    if not allow_download or paths.is_offline():
        raise RuntimeError(
            "uv is not installed and Verity is in offline mode. "
            "Install uv (https://docs.astral.sh/uv/) or import an airgap kit."
        )
    url = _uv_release_url()
    bdir = paths.bin_dir()
    bdir.mkdir(parents=True, exist_ok=True)
    log(f"bootstrapping uv from {url}")
    with tempfile.TemporaryDirectory(prefix="verity_uv_") as tmp:
        archive = Path(tmp) / "uv.tar.gz"
        req = urllib.request.Request(url, headers={"User-Agent": "verity-setup/0.6"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(archive, "wb") as out:
            shutil.copyfileobj(resp, out)
        with tarfile.open(archive, "r:gz") as tf:
            for member in tf.getmembers():
                name = Path(member.name).name
                if name in ("uv", "uvx") and member.isfile():
                    src = tf.extractfile(member)
                    if src is None:
                        continue
                    dest = bdir / name
                    with open(dest, "wb") as out:
                        shutil.copyfileobj(src, out)
                    dest.chmod(0o755)
    paths.ensure_runtime_path()
    found = find_uv()
    if not found:
        raise RuntimeError("Failed to extract uv binary into " + str(bdir))
    log(f"uv installed at {found}")
    return found


def ensure_ffmpeg(allow_download: bool = True, log: LogFn = _default_log) -> tuple[Optional[str], Optional[str]]:
    """Locate ``ffmpeg`` and ``ffprobe``, optionally fetching static Linux x86_64 builds."""
    paths.ensure_runtime_path()
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg and ffprobe:
        return ffmpeg, ffprobe
    if not allow_download or paths.is_offline():
        return ffmpeg, ffprobe
    if platform.system().lower() != "linux" or platform.machine().lower() not in ("x86_64", "amd64"):
        return ffmpeg, ffprobe

    # Static GPL-free/LGPL or standard static build from BtbN GitHub Releases
    url = (
        "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/"
        "ffmpeg-master-latest-linux64-gpl.tar.xz"
    )
    bdir = paths.bin_dir()
    bdir.mkdir(parents=True, exist_ok=True)
    log("ffmpeg/ffprobe not found on host; downloading static Linux x86_64 builds into " + str(bdir))
    try:
        with tempfile.TemporaryDirectory(prefix="verity_ffmpeg_") as tmp:
            archive = Path(tmp) / "ffmpeg.tar.xz"
            req = urllib.request.Request(url, headers={"User-Agent": "verity-setup/0.6"})
            with urllib.request.urlopen(req, timeout=180) as resp, open(archive, "wb") as out:
                shutil.copyfileobj(resp, out)
            with tarfile.open(archive, "r:xz") as tf:
                for member in tf.getmembers():
                    name = Path(member.name).name
                    if name in ("ffmpeg", "ffprobe") and member.isfile():
                        src = tf.extractfile(member)
                        if src is None:
                            continue
                        dest = bdir / name
                        with open(dest, "wb") as out:
                            shutil.copyfileobj(src, out)
                        dest.chmod(0o755)
        paths.ensure_runtime_path()
        ffmpeg = shutil.which("ffmpeg")
        ffprobe = shutil.which("ffprobe")
        if ffmpeg and ffprobe:
            log(f"static ffmpeg/ffprobe installed in {bdir}")
    except Exception as exc:
        log(f"WARN could not auto-download static ffmpeg ({exc}); install ffmpeg via your OS package manager")
    return ffmpeg, ffprobe


# --------------------------------------------------------------------------- #
#  Hardware profile & environment layout                                      #
# --------------------------------------------------------------------------- #


def detect_profile(requested: Optional[str] = None, log: LogFn = _default_log) -> str:
    if requested in ("cuda", "cpu"):
        return requested
    cfg_profile = paths.load_user_config().get("hw_profile")
    if requested in (None, "", "auto") and cfg_profile in ("cuda", "cpu"):
        return cfg_profile
    if requested not in (None, "", "auto"):
        log(f"WARN unknown VERITY_HW_PROFILE={requested!r}, treating as auto")
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi and subprocess.run([nvidia_smi], capture_output=True).returncode == 0:
        return "cuda"
    return "cpu"


def baked_envs_root() -> Optional[Path]:
    env = os.environ.get("VERITY_DETECTOR_ENVS_BAKED", "")
    if env and Path(env).is_dir():
        return Path(env)
    return None


def detector_ids(detectors_dir: Optional[Path] = None) -> List[str]:
    ddir = detectors_dir or paths.detectors_dir()
    if not ddir.is_dir():
        return []
    return sorted(p.name for p in ddir.iterdir() if p.is_dir() and (p / "pyproject.toml").exists())


def trained_detector_ids(detectors_dir: Optional[Path] = None) -> List[str]:
    ddir = detectors_dir or paths.detectors_dir()
    ids: List[str] = []
    for det_id in detector_ids(ddir):
        yaml_path = ddir / det_id / "detector.yaml"
        if not yaml_path.exists():
            continue
        spec = yaml.safe_load(yaml_path.read_text()) or {}
        weights = spec.get("weights") or {}
        trained = weights.get("trained")
        if trained is None:
            trained = weights.get("kind", "official") == "official"
        if trained:
            ids.append(det_id)
    return ids


def cpu_pyproject(src_text: str, log: LogFn = _default_log) -> str:
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
    lines = [
        "# Generated by verity env_manager: cpu hardware profile copy.",
        "# Adds +cpu torch/torchvision wheels from download.pytorch.org.",
        "",
        "[tool.uv]",
        "package = false",
        f"override-dependencies = {deps_override}",
        "",
        "[[tool.uv.index]]",
        'name = "pytorch-cpu"',
        f'url = "{CPU_INDEX}"',
        "",
    ]
    return src_text.rstrip() + "\n\n" + "\n".join(lines) + "\n"


def project_pyproject(det_id: str, profile: str, detectors_dir: Optional[Path] = None) -> str:
    ddir = detectors_dir or paths.detectors_dir()
    src = (ddir / det_id / "pyproject.toml").read_text()
    return cpu_pyproject(src) if profile == "cpu" else src


def project_dir(det_id: str, profile: str, detectors_dir: Optional[Path] = None) -> Path:
    baked = baked_envs_root()
    if baked is not None:
        proj = baked / profile / det_id
        marker = proj / ".verity-synced"
        want = hashlib.sha256(project_pyproject(det_id, profile, detectors_dir).encode()).hexdigest()
        if marker.exists() and marker.read_text() == want and (proj / ".venv").exists():
            return proj
    return paths.envs_root() / profile / det_id


def build_subprocess_env(profile: str, offline: Optional[bool] = None) -> dict[str, str]:
    """Build deterministic environment variables for uv and detector subprocesses."""
    paths.ensure_runtime_path()
    eff_offline = paths.is_offline() if offline is None else offline

    uv_cache = paths.uv_cache_dir()
    hf_cache = paths.hf_cache_dir()
    torch_cache = paths.torch_cache_dir()
    py_dir = paths.python_dir()
    for d in (uv_cache, hf_cache, torch_cache):
        d.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["UV_CACHE_DIR"] = str(uv_cache)
    env.setdefault("UV_LINK_MODE", "hardlink")
    env.setdefault("UV_NO_PROGRESS", "1")
    env["HF_HOME"] = str(hf_cache)
    env["TORCH_HOME"] = str(torch_cache)
    env["XDG_CACHE_HOME"] = str(paths.cache_dir())
    env["VERITY_HW_PROFILE"] = profile

    # Standalone/air-gap installs keep managed Python runtimes inside <data_dir>/python
    if paths.is_standalone_mode() or py_dir.is_dir():
        py_dir.mkdir(parents=True, exist_ok=True)
        env["UV_PYTHON_INSTALL_DIR"] = str(py_dir)
        env.setdefault("UV_PYTHON_PREFERENCE", "only-managed")

    wroot = paths.weights_root()
    if wroot is not None:
        wroot.mkdir(parents=True, exist_ok=True)
        env["VERITY_WEIGHTS_ROOT"] = str(wroot)

    cfg = paths.load_user_config()
    if "max_frames" in cfg and "VERITY_MAX_FRAMES" not in env:
        env["VERITY_MAX_FRAMES"] = str(cfg["max_frames"])

    if eff_offline:
        env["VERITY_OFFLINE"] = "1"
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["DIFFUSERS_OFFLINE"] = "1"
        env["UV_OFFLINE"] = "1"
        env["UV_NO_PYTHON_DOWNLOADS"] = "1"

    return env


def ensure_weights_links(det_id: str, detectors_dir: Optional[Path] = None) -> None:
    wroot = paths.weights_root()
    if wroot is None:
        return
    ddir = detectors_dir or paths.detectors_dir()
    paths.link_weights_dir(ddir / det_id, wroot / det_id)


def sync_one(
    det_id: str,
    profile: str,
    force: bool = False,
    offline: Optional[bool] = None,
    detectors_dir: Optional[Path] = None,
    log: LogFn = _default_log,
) -> bool:
    eff_offline = paths.is_offline() if offline is None else offline
    want = project_pyproject(det_id, profile, detectors_dir)
    want_hash = hashlib.sha256(want.encode()).hexdigest()

    if not force:
        baked_proj = project_dir(det_id, profile, detectors_dir)
        baked = baked_envs_root()
        if baked is not None and baked_proj.is_relative_to(baked):
            return True

    proj_dir = paths.envs_root() / profile / det_id
    proj_dir.mkdir(parents=True, exist_ok=True)
    marker = proj_dir / ".verity-synced"
    if not force and marker.exists() and marker.read_text() == want_hash and (proj_dir / ".venv").exists():
        return True

    pyproject_path = proj_dir / "pyproject.toml"
    pyproject_changed = (not pyproject_path.exists()) or (pyproject_path.read_text() != want)
    pyproject_path.write_text(want)
    # Only drop uv.lock when pyproject changed AND we are online (in offline import mode we keep the transferred lock!)
    if pyproject_changed and not eff_offline and (proj_dir / "uv.lock").exists():
        (proj_dir / "uv.lock").unlink()

    uv_bin = ensure_uv(allow_download=not eff_offline, log=log)
    env = build_subprocess_env(profile, offline=eff_offline)

    cmd = [uv_bin, "sync", "--project", str(proj_dir)]
    if eff_offline:
        cmd.extend(["--offline", "--no-python-downloads"])

    log(f"syncing {det_id} ({profile}{', offline' if eff_offline else ''})")
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        log(f"FAILED to sync {det_id} ({profile})")
        return False
    marker.write_text(want_hash)
    return True


def setup_detectors(
    detector_ids_list: Optional[List[str]] = None,
    trained_only: bool = False,
    all_detectors: bool = False,
    profile: str = "auto",
    force: bool = False,
    concurrency: int = 4,
    offline: Optional[bool] = None,
    detectors_dir: Optional[Path] = None,
    log: LogFn = _default_log,
) -> int:
    ddir = detectors_dir or paths.detectors_dir()
    eff_profile = detect_profile(profile, log=log)
    eff_offline = paths.is_offline() if offline is None else offline
    log(f"hardware profile: {eff_profile}")

    ensure_uv(allow_download=not eff_offline, log=log)

    available = detector_ids(ddir)
    if detector_ids_list:
        ids = list(detector_ids_list)
    elif all_detectors:
        ids = available
    elif trained_only:
        ids = trained_detector_ids(ddir)
    else:
        ids = []

    unknown = sorted(set(ids) - set(available))
    if unknown:
        log(f"unknown detector(s): {', '.join(unknown)}")
        return 2
    if not ids:
        log("no detectors selected (use --detector ID[,ID...], --trained, or --all)")
        return 2

    failed: List[str] = []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {
            pool.submit(sync_one, det_id, eff_profile, force, eff_offline, ddir, log): det_id
            for det_id in ids
        }
        for fut in futures:
            det_id = futures[fut]
            if not fut.result():
                failed.append(det_id)

    for det_id in ids:
        ensure_weights_links(det_id, ddir)

    if failed:
        log(f"failed: {', '.join(sorted(failed))}")
        return 1
    log(f"setup complete: {len(ids)} detector(s) ready on {eff_profile}")
    return 0


def run_detector_process(
    detector_id: str,
    input_dir: Path,
    output_dir: Path,
    profile: Optional[str] = None,
    timeout: int = 3600,
    capture: bool = True,
    detectors_dir: Optional[Path] = None,
    log: LogFn = _default_log,
) -> subprocess.CompletedProcess:
    """Sync (if needed) and execute ``detectors/<detector_id>/inference.py``."""
    ddir = detectors_dir or paths.detectors_dir()
    eff_profile = detect_profile(profile or os.environ.get("VERITY_HW_PROFILE", "auto"), log=log)
    eff_offline = paths.is_offline()

    if not sync_one(detector_id, eff_profile, force=False, offline=eff_offline, detectors_dir=ddir, log=log):
        raise RuntimeError(f"Failed to provision environment for detector '{detector_id}'")
    ensure_weights_links(detector_id, ddir)

    in_abs = os.path.abspath(str(input_dir))
    out_abs = os.path.abspath(str(output_dir))
    os.makedirs(in_abs, exist_ok=True)
    os.makedirs(out_abs, exist_ok=True)

    uv_bin = ensure_uv(allow_download=not eff_offline, log=log)
    env = build_subprocess_env(eff_profile, offline=eff_offline)
    env["INPUT_DIR"] = in_abs
    env["OUTPUT_DIR"] = out_abs

    proj = project_dir(detector_id, eff_profile, ddir)
    cmd = [
        uv_bin,
        "run",
        "--project",
        str(proj),
        "--no-sync",
        "python",
        str(ddir / detector_id / "inference.py"),
    ]
    return subprocess.run(
        cmd,
        env=env,
        cwd=str(ddir / detector_id),
        capture_output=capture,
        text=True,
        timeout=timeout,
    )


def warmup_detectors(
    detector_ids_list: Optional[List[str]] = None,
    trained_only: bool = True,
    profile: str = "auto",
    detectors_dir: Optional[Path] = None,
    log: LogFn = _default_log,
) -> dict[str, str]:
    """Pre-download runtime-fetched HF / Torch Hub / FAN foundation caches for
    air-gapped readiness by running ``build_model`` on CPU inside each
    detector's uv environment.

    Returns a dict of ``{detector_id: "ok" | "skipped: ..." | "failed: ..."}``.
    """
    from verity.weights import load_manifest

    ddir = detectors_dir or paths.detectors_dir()
    manifest = load_manifest(ddir / "weights_manifest.yaml")
    eff_profile = detect_profile(profile, log=log)
    uv_bin = ensure_uv(allow_download=True, log=log)
    env = build_subprocess_env(eff_profile, offline=False)

    if detector_ids_list:
        candidates = list(detector_ids_list)
    elif trained_only:
        candidates = trained_detector_ids(ddir)
    else:
        candidates = detector_ids(ddir)

    # Only detectors that fetch external models at runtime (HF / Torch Hub / FAN)
    to_warm = [d for d in candidates if d in manifest.runtime_fetched]
    statuses: dict[str, str] = {}

    if not to_warm:
        log("no runtime-fetched detectors in selection")
        return statuses

    warmup_snippet = (
        "import sys, yaml, torch\n"
        "from pathlib import Path\n"
        "det_dir = Path(sys.argv[1])\n"
        "sys.path.insert(0, str(det_dir))\n"
        "sys.path.insert(0, str(det_dir.parent))\n"
        "spec = yaml.safe_load((det_dir / 'detector.yaml').read_text()) or {}\n"
        "wfile = (spec.get('weights') or {}).get('file') or ''\n"
        "wpath = str(det_dir / 'weights' / wfile) if wfile and (det_dir / 'weights' / wfile).exists() else None\n"
        "import model\n"
        "m = model.build_model(wpath, torch.device('cpu'))\n"
        "del m\n"
        "print('WARMUP_OK')\n"
    )

    for det_id in to_warm:
        note = manifest.runtime_fetched.get(det_id, "")
        if "nothing is fetched today" in note.lower():
            statuses[det_id] = "skipped (no runtime download)"
            continue
        if not sync_one(det_id, eff_profile, force=False, offline=False, detectors_dir=ddir, log=log):
            statuses[det_id] = "failed: env sync failed"
            continue
        ensure_weights_links(det_id, ddir)
        proj = project_dir(det_id, eff_profile, ddir)
        log(f"warming runtime cache for {det_id} ({note[:70]})")
        proc = subprocess.run(
            [uv_bin, "run", "--project", str(proj), "--no-sync", "python", "-c", warmup_snippet, str(ddir / det_id)],
            env=env,
            cwd=str(ddir / det_id),
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if proc.returncode == 0 and "WARMUP_OK" in (proc.stdout or ""):
            statuses[det_id] = "ok"
            log(f"  {det_id}: cached")
        else:
            tail = ((proc.stderr or proc.stdout or "").strip().splitlines() or ["unknown error"])[-1]
            statuses[det_id] = f"failed: {tail[:160]}"
            log(f"  {det_id}: WARN warmup failed ({tail[:160]})")

    return statuses


# --------------------------------------------------------------------------- #
#  CLI entrypoint (used by scripts/verity.py compatibility shim)              #
# --------------------------------------------------------------------------- #


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verity detector environment CLI (uv-backed, hardware-adaptive)."
    )
    parser.add_argument(
        "--profile",
        choices=["auto", "cuda", "cpu"],
        default=os.environ.get("VERITY_HW_PROFILE", "auto"),
        help="hardware profile (default: $VERITY_HW_PROFILE or auto-detect)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_setup = sub.add_parser("setup", help="create/refresh detector environments")
    p_setup.add_argument("--detector", help="sync one or more detectors (comma-separated)")
    p_setup.add_argument("--all", action="store_true", help="sync all detectors")
    p_setup.add_argument(
        "--trained",
        action="store_true",
        help="sync only detectors with a trained classification head",
    )
    p_setup.add_argument("--force", action="store_true", help="re-sync even if up to date")
    p_setup.add_argument("--concurrency", type=int, default=8)

    p_run = sub.add_parser("run", help="run one detector")
    p_run.add_argument("detector")
    p_run.add_argument("input_dir", nargs="?", help="default: $INPUT_DIR or /input")
    p_run.add_argument("output_dir", nargs="?", help="default: $OUTPUT_DIR or /output")

    sub.add_parser("profile", help="print detected hardware profile")

    args = parser.parse_args(argv)
    if args.command == "setup":
        ids = [d.strip() for d in args.detector.split(",") if d.strip()] if args.detector else None
        return setup_detectors(
            detector_ids_list=ids,
            trained_only=args.trained,
            all_detectors=args.all,
            profile=args.profile,
            force=args.force,
            concurrency=args.concurrency,
        )
    if args.command == "run":
        in_dir = Path(args.input_dir or os.environ.get("INPUT_DIR", "/input"))
        out_dir = Path(args.output_dir or os.environ.get("OUTPUT_DIR", "/output"))
        proc = run_detector_process(
            args.detector, in_dir, out_dir, profile=args.profile, capture=False
        )
        return proc.returncode
    print(detect_profile(args.profile))
    return 0
