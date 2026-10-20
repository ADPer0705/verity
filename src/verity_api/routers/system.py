"""System status, capability probing and live settings."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter

from verity_api.config import ALL_EXTRACTORS, get_settings, probe_environment, update_settings
from verity_api.jobs import manager
from verity_api.schemas import SettingsUpdate
from verity_api.service import build_registry

router = APIRouter(prefix="/api/system", tags=["system"])


def _readiness(environment: Dict[str, Any]) -> Dict[str, Any]:
    """What the UI needs to know before offering an action.

    Each blocker names the thing that is missing and what it stops, so the
    interface can explain the gap rather than let a job fail obscurely.
    """
    blockers = []
    if not environment["detectors_dir_present"]:
        blockers.append(
            {
                "capability": "detection",
                "message": "No detectors directory found. Set it under Settings, "
                "or point VERITY_DETECTORS_DIR at your detector definitions.",
            }
        )
    elif environment["detector_count"] == 0:
        blockers.append(
            {
                "capability": "detection",
                "message": "The detectors directory contains no detector.yaml definitions.",
            }
        )
    if not environment["detector_runner"]:
        blockers.append(
            {
                "capability": "detection",
                "message": "Detector runtime (scripts/verity.py or verity-run) not found. "
                "Set VERITY_RUNNER, or run inside the container image.",
            }
        )
    if not environment["uv"]:
        blockers.append(
            {
                "capability": "setup",
                "message": "uv is not on PATH, so detector environments cannot be provisioned.",
            }
        )
    if not environment["ffprobe"]:
        blockers.append(
            {
                "capability": "analysis",
                "message": "ffprobe not found. Duration, resolution, FPS and codec will be blank; "
                "face scanning still works.",
            }
        )
    if not environment["weasyprint"]:
        blockers.append(
            {
                "capability": "report-pdf",
                "message": environment.get("weasyprint_error")
                or "WeasyPrint is unavailable, so PDF export will fail. HTML reports work.",
            }
        )

    return {
        "can_ingest": True,
        "can_analyze": bool(environment["opencv"]),
        "can_detect": environment["detector_count"] > 0 and bool(environment["detector_runner"]),
        "can_setup": bool(environment["uv"]) and environment["detectors_dir_present"],
        "can_report_html": True,
        "can_report_pdf": bool(environment["weasyprint"]),
        "blockers": blockers,
    }


@router.get("")
def system_status() -> Dict[str, Any]:
    settings = get_settings()
    environment = probe_environment(settings)
    registry = build_registry(settings)
    return {
        "settings": settings.to_dict(),
        "available_extractors": ALL_EXTRACTORS,
        "environment": environment,
        "readiness": _readiness(environment),
        "detectors_registered": len(registry.list_detectors()),
        "active_jobs": manager.active_count(),
    }


@router.put("/settings")
def put_settings(update: SettingsUpdate) -> Dict[str, Any]:
    settings = update_settings(**update.model_dump(exclude_none=True))
    environment = probe_environment(settings)
    return {
        "settings": settings.to_dict(),
        "environment": environment,
        "readiness": _readiness(environment),
    }
