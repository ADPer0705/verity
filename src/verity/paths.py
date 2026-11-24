"""Canonical directory and configuration resolution for Verity.

Supports two seamless operating modes:
1. **Standalone binary / installed workstation mode** (frozen PyInstaller
   executable, or when ``VERITY_HOME`` is explicitly set):
   Uses ``platformdirs`` (``~/.local/share/verity``, ``~/.cache/verity``,
   ``~/.config/verity`` on Linux; or ``$VERITY_HOME/{data,cache,config}`` when
   ``VERITY_HOME`` is set) so the executable never pollutes the current working
   directory and can be sealed for air-gapped forensic use.
2. **Repository development mode** (editable checkout with ``detectors/`` next
   to ``src/`` and ``VERITY_HOME`` unset):
   Preserves repo-local defaults (``detectors/``, ``workspace/``,
   ``.detectorenvs/``) for zero-friction local development and tests.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from platformdirs import user_cache_dir, user_config_dir, user_data_dir

from verity import __version__

APP_NAME = "verity"
_TRUTHY = {"1", "true", "yes", "on"}


def is_frozen() -> bool:
    """True when running inside a PyInstaller standalone executable."""
    return bool(getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"))


def bundle_root() -> Path:
    """Read-only root containing bundled resources (detectors, UI dist)."""
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS"))
    return Path(__file__).resolve().parents[2]


def is_standalone_mode() -> bool:
    """True when Verity should store runtime state under platformdirs / VERITY_HOME."""
    if is_frozen():
        return True
    if os.environ.get("VERITY_HOME"):
        return True
    # Pip-installed wheel outside a git checkout (no adjacent detectors/ folder)
    if not (bundle_root() / "detectors").is_dir():
        return True
    return False


def data_dir() -> Path:
    """Persistent application data: detectors, weights, envs, python, bin, workspace."""
    override = os.environ.get("VERITY_HOME")
    if override:
        return Path(override).expanduser().resolve() / "data"
    return Path(user_data_dir(APP_NAME))


def cache_dir() -> Path:
    """Reusable caches: uv wheels, Hugging Face models, Torch Hub models."""
    override = os.environ.get("VERITY_HOME")
    if override:
        return Path(override).expanduser().resolve() / "cache"
    return Path(user_cache_dir(APP_NAME))


def config_dir() -> Path:
    """Persistent user/workstation configuration (config.yaml)."""
    override = os.environ.get("VERITY_HOME")
    if override:
        return Path(override).expanduser().resolve() / "config"
    return Path(user_config_dir(APP_NAME))


def config_file() -> Path:
    return config_dir() / "config.yaml"


def load_user_config() -> Dict[str, Any]:
    """Load persistent workstation config from ``<config_dir>/config.yaml``."""
    path = config_file()
    if not path.is_file():
        return {}
    try:
        data = yaml.safe_load(path.read_text()) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_user_config(patch: Dict[str, Any]) -> Path:
    """Merge ``patch`` into ``<config_dir>/config.yaml`` and persist it."""
    current = load_user_config()
    for k, v in patch.items():
        if v is None:
            current.pop(k, None)
        else:
            current[k] = v
    path = config_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(current, sort_keys=True))
    return path


def is_offline() -> bool:
    """Whether Verity is running in strict air-gapped offline mode."""
    env = os.environ.get("VERITY_OFFLINE")
    if env is not None and env.strip() != "":
        return env.strip().lower() in _TRUTHY
    cfg = load_user_config()
    return bool(cfg.get("offline", False))


def bin_dir() -> Path:
    """Location for self-bootstrapped tools (uv, optional static ffmpeg/ffprobe)."""
    return data_dir() / "bin"


def python_dir() -> Path:
    """Location for uv-managed standalone Python runtimes (air-gap safe)."""
    return data_dir() / "python"


def uv_cache_dir() -> Path:
    if os.environ.get("UV_CACHE_DIR"):
        return Path(os.environ["UV_CACHE_DIR"]).expanduser().resolve()
    if is_standalone_mode():
        return cache_dir() / "uv"
    return envs_root() / "cache"


def hf_cache_dir() -> Path:
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]).expanduser().resolve()
    return cache_dir() / "huggingface"


def torch_cache_dir() -> Path:
    if os.environ.get("TORCH_HOME"):
        return Path(os.environ["TORCH_HOME"]).expanduser().resolve()
    return cache_dir() / "torch"


def envs_root() -> Path:
    """Root directory holding isolated per-detector uv environments."""
    env = os.environ.get("VERITY_DETECTOR_ENVS")
    if env:
        return Path(env).expanduser().resolve()
    if is_standalone_mode():
        return data_dir() / "envs"
    return bundle_root() / ".detectorenvs"


def weights_root() -> Optional[Path]:
    """External weights root symlinked into ``detectors/<id>/weights``.

    In standalone/container mode, weights live persistently under
    ``<data_dir>/weights/<id>`` (or ``$VERITY_WEIGHTS_ROOT``) so refreshing the
    bundled detector scripts never touches downloaded checkpoints.
    """
    env = os.environ.get("VERITY_WEIGHTS_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    if is_standalone_mode():
        return data_dir() / "weights"
    return None


def detectors_dir() -> Path:
    """Directory containing detector definitions and inference scripts."""
    env = os.environ.get("VERITY_DETECTORS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    cfg = load_user_config()
    if cfg.get("detectors_dir"):
        return Path(str(cfg["detectors_dir"])).expanduser().resolve()
    if is_standalone_mode():
        return ensure_bundled_detectors_extracted()
    return bundle_root() / "detectors"


def workspace_dir() -> Path:
    """Directory holding ingested cases, artifacts, outputs and reports."""
    env = os.environ.get("VERITY_WORKSPACE")
    if env:
        return Path(env).expanduser().resolve()
    cfg = load_user_config()
    if cfg.get("workspace"):
        return Path(str(cfg["workspace"])).expanduser().resolve()
    if is_standalone_mode():
        return data_dir() / "workspace"
    return (Path.cwd() / "workspace").resolve()


def output_dir(ws: Optional[Path] = None) -> Path:
    env = os.environ.get("VERITY_OUTPUT_DIR")
    if env:
        return Path(env).expanduser().resolve()
    cfg = load_user_config()
    if cfg.get("output_dir"):
        return Path(str(cfg["output_dir"])).expanduser().resolve()
    return (ws or workspace_dir()) / "output"


def ui_dist_dir() -> Optional[Path]:
    """Directory holding the built React Web Console static bundle."""
    env = os.environ.get("VERITY_UI_DIST")
    if env:
        p = Path(env).expanduser().resolve()
        return p if (p / "index.html").is_file() else None
    candidates = [
        bundle_root() / "verity_ui_dist",
        bundle_root() / "src" / "verity_ui" / "dist",
        data_dir() / "ui_dist",
    ]
    for c in candidates:
        if (c / "index.html").is_file():
            return c
    return None


def ensure_runtime_path() -> None:
    """Prepend Verity's private bin/ directory to PATH so bootstrapped tools
    (``uv``, static ``ffmpeg``/``ffprobe``) are discovered automatically."""
    bdir = bin_dir()
    if bdir.is_dir():
        bstr = str(bdir)
        current = os.environ.get("PATH", "")
        parts = current.split(os.pathsep) if current else []
        if bstr not in parts:
            os.environ["PATH"] = bstr + (os.pathsep + current if current else "")


def ensure_bundled_detectors_extracted(force: bool = False) -> Path:
    """Extract embedded detector definitions/scripts into ``<data_dir>/detectors``
    when running from a standalone binary or ``VERITY_HOME`` install.

    Preserves any existing ``weights/`` symlinks or downloaded checkpoints so
    upgrading the Verity binary never forces re-downloading models.
    """
    target = data_dir() / "detectors"
    bundled = bundle_root() / "detectors"
    marker = target / ".verity-bundle-version"
    want_stamp = f"{__version__}"

    if not bundled.is_dir():
        target.mkdir(parents=True, exist_ok=True)
        return target

    if not force and marker.is_file() and marker.read_text().strip() == want_stamp:
        return target

    target.mkdir(parents=True, exist_ok=True)
    wroot = weights_root()
    if wroot is not None:
        wroot.mkdir(parents=True, exist_ok=True)

    for src in bundled.rglob("*"):
        rel = src.relative_to(bundled)
        # Skip __pycache__ and binary checkpoints inside bundled tree
        if "__pycache__" in rel.parts or src.suffix in (".pyc", ".pyo"):
            continue
        dst = target / rel
        if src.is_dir():
            # Never overwrite an existing weights/ symlink with a directory
            if dst.is_symlink():
                continue
            dst.mkdir(parents=True, exist_ok=True)
        else:
            # If copying weights/README.md into a symlinked weights dir, follow safely
            if dst.parent.is_symlink():
                dst.parent.resolve().mkdir(parents=True, exist_ok=True)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    # Wire every detector's weights/ directory to persistent <data_dir>/weights/<id>
    if wroot is not None:
        for det_dir in sorted(target.iterdir()):
            if not det_dir.is_dir() or not (det_dir / "detector.yaml").is_file():
                continue
            link_weights_dir(det_dir, wroot / det_dir.name)

    marker.write_text(want_stamp + "\n")
    return target


def link_weights_dir(detector_dir: Path, persistent_target: Path) -> Path:
    """Ensure ``<detector_dir>/weights`` is a symlink to ``persistent_target``."""
    wdir = detector_dir / "weights"
    persistent_target.mkdir(parents=True, exist_ok=True)
    if wdir.is_symlink():
        if wdir.resolve() == persistent_target.resolve():
            return persistent_target
        wdir.unlink()
    elif wdir.exists():
        readme = wdir / "README.md"
        if readme.exists() and not (persistent_target / "README.md").exists():
            shutil.copy2(readme, persistent_target / "README.md")
        # Move any existing local checkpoint files into persistent_target before symlinking
        for item in wdir.rglob("*"):
            if item.is_file() and item.name != "README.md":
                rel = item.relative_to(wdir)
                dest_item = persistent_target / rel
                dest_item.parent.mkdir(parents=True, exist_ok=True)
                if not dest_item.exists():
                    shutil.move(str(item), str(dest_item))
        shutil.rmtree(wdir)
    wdir.symlink_to(persistent_target, target_is_directory=True)
    return persistent_target
