"""Settings resolution for the API.

Mirrors ``verity.cli._load_config`` / ``_detectors_dir`` so the UI and the CLI
always agree on where detectors, the workspace and reports live. One
difference, deliberate: relative paths from ``config/pipeline.yaml`` are
resolved against the repository root rather than the process working
directory, because a long-lived server should not change meaning depending on
where uvicorn happened to be launched.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

#: ``src/verity_api/config.py`` -> repository root
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Extractor keys understood by the pipeline, in pipeline order.
ALL_EXTRACTORS: List[str] = ["frame", "face", "metadata", "audio"]

DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "pipeline.yaml"

_TRUTHY = {"1", "true", "yes", "on"}


def _as_bool(value: Optional[str]) -> bool:
    return (value or "").strip().lower() in _TRUTHY


def _resolve(value: str | os.PathLike[str], base: Path = REPO_ROOT) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


@dataclass
class Settings:
    """Effective pipeline configuration for this server process."""

    detectors_dir: Path
    workspace: Path
    output_dir: Path
    extractors: List[str] = field(default_factory=lambda: list(ALL_EXTRACTORS))
    offline: bool = False
    config_path: Optional[Path] = None
    #: Directory holding the built SPA, when serving the UI from this process.
    ui_dist: Optional[Path] = None

    # -- derived locations -------------------------------------------- #

    @property
    def media_dir(self) -> Path:
        return self.workspace / "media"

    @property
    def artifacts_dir(self) -> Path:
        return self.workspace / "artifacts"

    @property
    def detector_output_dir(self) -> Path:
        """Where ``DetectorRunner`` writes ``<media_id>_<detector_id>/``.

        Hardcoded to ``workspace/output`` by ``ReportGenerator``, so it is not
        the same thing as :attr:`output_dir` (which holds reports).
        """
        return self.workspace / "output"

    def report_dir(self, media_id: str) -> Path:
        return self.output_dir / media_id

    @property
    def readable_roots(self) -> List[Path]:
        """Directories the API is allowed to serve files from."""
        return [self.workspace, self.output_dir, self.detectors_dir]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detectors_dir": str(self.detectors_dir),
            "workspace": str(self.workspace),
            "output_dir": str(self.output_dir),
            "extractors": list(self.extractors),
            "offline": self.offline,
            "config_path": str(self.config_path) if self.config_path else None,
        }


def load_settings(config_path: Optional[Path] = None) -> Settings:
    """Build :class:`Settings` from the config file, then environment overrides."""
    path = config_path or (DEFAULT_CONFIG_PATH if DEFAULT_CONFIG_PATH.exists() else None)

    raw: Dict[str, Any] = {}
    if path and Path(path).exists():
        loaded = yaml.safe_load(Path(path).read_text()) or {}
        if isinstance(loaded, dict):
            raw = loaded

    detectors_dir = os.environ.get("VERITY_DETECTORS_DIR") or raw.get("detectors_dir") or "detectors"
    workspace = os.environ.get("VERITY_WORKSPACE") or raw.get("workspace") or "workspace"
    output_dir = os.environ.get("VERITY_OUTPUT_DIR") or raw.get("output_dir") or ""

    workspace_path = _resolve(workspace)
    output_path = _resolve(output_dir) if output_dir else workspace_path / "output"

    extractors = raw.get("extractors") or ALL_EXTRACTORS
    extractors = [e for e in extractors if e in ALL_EXTRACTORS] or list(ALL_EXTRACTORS)

    ui_dist_env = os.environ.get("VERITY_UI_DIST")
    ui_dist = _resolve(ui_dist_env) if ui_dist_env else REPO_ROOT / "src" / "verity_ui" / "dist"

    return Settings(
        detectors_dir=_resolve(detectors_dir),
        workspace=workspace_path,
        output_dir=output_path,
        extractors=extractors,
        offline=_as_bool(os.environ.get("VERITY_OFFLINE")),
        config_path=Path(path) if path else None,
        ui_dist=ui_dist if ui_dist.exists() else None,
    )


# --------------------------------------------------------------------- #
#  Process-wide singleton (mutable from the Settings screen)             #
# --------------------------------------------------------------------- #

_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings


def update_settings(**changes: Any) -> Settings:
    """Apply a partial update to the live settings and return the new value."""
    global _settings
    current = get_settings()

    patch: Dict[str, Any] = {}
    for key in ("detectors_dir", "workspace", "output_dir"):
        if changes.get(key):
            patch[key] = _resolve(changes[key])
    if changes.get("extractors") is not None:
        wanted = [e for e in changes["extractors"] if e in ALL_EXTRACTORS]
        patch["extractors"] = wanted
    if changes.get("offline") is not None:
        patch["offline"] = bool(changes["offline"])

    _settings = replace(current, **patch)
    return _settings


# --------------------------------------------------------------------- #
#  Environment capability probing (surfaced on the Settings screen)      #
# --------------------------------------------------------------------- #


def _module_available(name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


@lru_cache(maxsize=1)
def _weasyprint_status() -> tuple[bool, str]:
    """Whether PDF rendering will actually work, not merely whether the
    package is installed.

    WeasyPrint is a pure-Python package that binds to native Pango/GObject
    libraries at import time. On Windows those are frequently absent, so
    ``find_spec`` says yes and ``HTML(...).write_pdf()`` then fails with an
    opaque ctypes error. Import it once here so the UI can disable PDF export
    up front instead of surfacing that error from inside a job.
    """
    if not _module_available("weasyprint"):
        return False, "WeasyPrint is not installed."
    try:
        import weasyprint  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - any import failure disables PDF
        return False, (
            f"WeasyPrint is installed but cannot load its native dependencies "
            f"({exc}). On Windows this needs the GTK runtime; on Linux, "
            f"libpango and libgdk-pixbuf."
        )
    return True, ""


def probe_environment(settings: Settings) -> Dict[str, Any]:
    """Report which optional pieces of the pipeline are actually usable.

    The UI uses this to explain *why* a capability is unavailable instead of
    failing with an opaque error deep inside a job.
    """
    detectors_present = settings.detectors_dir.exists()
    detector_count = 0
    if detectors_present:
        detector_count = sum(
            1 for d in settings.detectors_dir.iterdir() if (d / "detector.yaml").exists()
        )

    runner_script = REPO_ROOT / "scripts" / "verity.py"
    runner = os.environ.get("VERITY_RUNNER") or (
        str(runner_script) if runner_script.exists() else shutil.which("verity-run")
    )

    weasyprint_ok, weasyprint_error = _weasyprint_status()

    return {
        "detectors_dir_present": detectors_present,
        "detector_count": detector_count,
        "detector_runner": runner,
        "ffprobe": shutil.which("ffprobe"),
        "ffmpeg": shutil.which("ffmpeg"),
        "uv": shutil.which("uv"),
        "weasyprint": weasyprint_ok,
        "weasyprint_error": weasyprint_error,
        "opencv": _module_available("cv2"),
        "hw_profile": os.environ.get("VERITY_HW_PROFILE", "auto"),
    }
