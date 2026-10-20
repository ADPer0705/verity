"""Detector registry and environment provisioning (``verity setup``)."""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from verity_api.config import get_settings
from verity_api.jobs import JobContext, manager
from verity_api.schemas import SetupRequest
from verity_api.service import build_registry, coverage, detector_catalog, run_setup

router = APIRouter(prefix="/api/detectors", tags=["detectors"])


@router.get("")
def list_detectors() -> Dict[str, Any]:
    settings = get_settings()
    catalog = detector_catalog(settings)
    return {
        "detectors_dir": str(settings.detectors_dir),
        "detectors_dir_present": settings.detectors_dir.exists(),
        "count": len(catalog),
        "weights_ready": sum(1 for d in catalog if d["weights"]["available"]),
        "detectors": catalog,
    }


@router.get("/coverage")
def get_coverage() -> Dict[str, Any]:
    """Manipulation families and which detectors (by tier) cover each."""
    families = coverage(get_settings())
    return {"families": families, "covered": sum(1 for f in families if f["covered"]),
            "total": len(families)}


@router.get("/{detector_id}")
def get_detector(detector_id: str) -> Dict[str, Any]:
    settings = get_settings()
    for entry in detector_catalog(settings):
        if entry["detector_id"] == detector_id:
            return entry
    raise HTTPException(status_code=404, detail=f"Unknown detector: {detector_id}")


@router.post("/setup")
def setup_detectors(request: SetupRequest) -> Dict[str, Any]:
    settings = get_settings()
    registry = build_registry(settings)
    known = set(registry.list_detectors())

    requested: List[str] = [d for d in (request.detectors or []) if d]
    unknown = sorted(set(requested) - known)
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown detector(s): {', '.join(unknown)}")

    target = requested or None
    label = (
        f"Provision {', '.join(requested)}" if requested else "Provision all detector environments"
    )

    def worker(ctx: JobContext) -> Dict[str, Any]:
        ctx.declare_steps([("setup", label)])
        ctx.start_step("setup")
        result = run_setup(ctx, settings, target, request.force)
        ctx.finish_step("setup", "Environments ready")
        return result

    job = manager.submit(
        kind="setup",
        label=label,
        worker=worker,
        params={"detectors": requested, "force": request.force},
    )
    return job.to_dict()
