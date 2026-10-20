"""Pipeline operations (each returns a job) and job inspection."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from verity_api.config import ALL_EXTRACTORS, get_settings
from verity_api.jobs import JobContext, manager
from verity_api.schemas import (
    AnalyzeRequest,
    CompileRequest,
    ExtractRequest,
    ReportRequest,
    RunRequest,
)
from verity_api.service import (
    build_registry,
    consensus,
    load_media_info,
    run_analyze,
    run_compile,
    run_extract,
    run_full_pipeline,
    run_report,
    run_select,
)

router = APIRouter(prefix="/api", tags=["jobs"])

#: How often the SSE stream checks for new log lines.
STREAM_POLL_SECONDS = 0.4


def _case(media_id: str):
    settings = get_settings()
    info = load_media_info(settings, media_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"No ingested media with id {media_id}")
    return settings, info


def _file_label(settings, media_id: str) -> str:
    info = load_media_info(settings, media_id)
    if info is None:
        return media_id
    return info.metadata.get("original_name") or Path(info.original_path).name


def _validate_detectors(settings, detectors: Optional[List[str]]) -> Optional[List[str]]:
    if not detectors:
        return None
    known = set(build_registry(settings).list_detectors())
    unknown = sorted(set(detectors) - known)
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown detector(s): {', '.join(unknown)}")
    return sorted(set(detectors))


def _validate_extractors(extractors: Optional[List[str]]) -> Optional[List[str]]:
    if not extractors:
        return None
    unknown = sorted(set(extractors) - set(ALL_EXTRACTORS))
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown extractor(s): {', '.join(unknown)}")
    return [e for e in ALL_EXTRACTORS if e in extractors]


# --------------------------------------------------------------------- #
#  Operations                                                            #
# --------------------------------------------------------------------- #


@router.post("/run", status_code=202)
def post_run(request: RunRequest) -> Dict[str, Any]:
    """``verity run`` — the full pipeline for one case."""
    settings, _ = _case(request.media_id)
    detectors = _validate_detectors(settings, request.detectors)
    extractors = _validate_extractors(request.extractors)
    name = _file_label(settings, request.media_id)

    def worker(ctx: JobContext) -> Dict[str, Any]:
        return run_full_pipeline(
            ctx,
            settings,
            request.media_id,
            detector_override=detectors,
            extractors=extractors,
            fmt=request.format,
            skip_report=request.skip_report,
            include_untrained=request.include_untrained,
        )

    job = manager.submit(
        kind="run",
        label=f"Examine {name}",
        worker=worker,
        params={
            "detectors": detectors,
            "extractors": extractors,
            "format": request.format,
            "skip_report": request.skip_report,
            "include_untrained": request.include_untrained,
        },
        media_ids=[request.media_id],
        steps=[
            ("extract", "Extract artifacts"),
            ("forensics", "Forensic analyses"),
            ("analyze", "Characterize media"),
            ("select", "Select detectors"),
            ("compile", "Compile findings"),
        ]
        + ([] if request.skip_report else [("report", "Generate report")]),
    )
    return job.to_dict()


@router.post("/analyze", status_code=202)
def post_analyze(request: AnalyzeRequest) -> Dict[str, Any]:
    """``verity analyze`` — characterize the media and explain detector eligibility."""
    settings, info = _case(request.media_id)
    detectors = _validate_detectors(settings, request.detectors)
    name = _file_label(settings, request.media_id)

    def worker(ctx: JobContext) -> Dict[str, Any]:
        ctx.declare_steps([("analyze", "Characterize media"), ("select", "Select detectors")])
        ctx.start_step("analyze")
        analysis = run_analyze(ctx, settings, request.media_id, info.media_type)
        ctx.finish_step(
            "analyze",
            f"{analysis.media.face_count} face(s) across "
            f"{analysis.media.sampled_frames} sampled frames",
        )
        ctx.start_step("select")
        selection, selected = run_select(ctx, settings, analysis, detectors,
                                         include_untrained=request.include_untrained)
        ctx.finish_step("select", f"{len(selected)} eligible")
        return {
            "media_id": request.media_id,
            "selected": selected,
            "excluded": len([d for d in selection.decisions if not d.included]),
        }

    job = manager.submit(
        kind="analyze",
        label=f"Analyze {name}",
        worker=worker,
        params={"detectors": detectors, "include_untrained": request.include_untrained},
        media_ids=[request.media_id],
    )
    return job.to_dict()


@router.post("/extract", status_code=202)
def post_extract(request: ExtractRequest) -> Dict[str, Any]:
    """``verity extract``."""
    settings, _ = _case(request.media_id)
    extractors = _validate_extractors(request.extractors)
    name = _file_label(settings, request.media_id)

    def worker(ctx: JobContext) -> Dict[str, Any]:
        ctx.declare_steps([("extract", "Extract artifacts")])
        ctx.start_step("extract")
        result = run_extract(ctx, settings, request.media_id, extractors)
        ctx.finish_step("extract", f"{result['total_artifacts']} artifact(s)")
        return result

    job = manager.submit(
        kind="extract",
        label=f"Extract artifacts from {name}",
        worker=worker,
        params={"extractors": extractors},
        media_ids=[request.media_id],
    )
    return job.to_dict()


@router.post("/compile", status_code=202)
def post_compile(request: CompileRequest) -> Dict[str, Any]:
    """``verity compile``."""
    settings, _ = _case(request.media_id)
    name = _file_label(settings, request.media_id)

    def worker(ctx: JobContext) -> Dict[str, Any]:
        ctx.declare_steps([("compile", "Compile findings")])
        ctx.start_step("compile")
        collection = run_compile(ctx, settings, request.media_id)
        ctx.finish_step("compile", f"{len(collection.findings)} finding(s)")
        return {
            "media_id": request.media_id,
            "findings_count": len(collection.findings),
            "aggregated_score": collection.aggregated_score,
            "consensus": consensus(collection.findings),
        }

    job = manager.submit(
        kind="compile",
        label=f"Compile findings for {name}",
        worker=worker,
        media_ids=[request.media_id],
    )
    return job.to_dict()


@router.post("/report", status_code=202)
def post_report(request: ReportRequest) -> Dict[str, Any]:
    """``verity report``."""
    settings, _ = _case(request.media_id)
    name = _file_label(settings, request.media_id)

    def worker(ctx: JobContext) -> Dict[str, Any]:
        ctx.declare_steps([("report", f"Generate {request.format.upper()} report")])
        ctx.start_step("report")
        result = run_report(ctx, settings, request.media_id, request.format)
        ctx.finish_step("report", Path(result.get("pdf") or result["html"]).name)
        return result

    job = manager.submit(
        kind="report",
        label=f"Report for {name} ({request.format.upper()})",
        worker=worker,
        params={"format": request.format},
        media_ids=[request.media_id],
    )
    return job.to_dict()


# --------------------------------------------------------------------- #
#  Job inspection                                                        #
# --------------------------------------------------------------------- #


@router.get("/jobs")
def list_jobs(
    limit: int = Query(50, ge=1, le=250),
    media_id: Optional[str] = None,
) -> Dict[str, Any]:
    jobs = manager.list(limit=limit, media_id=media_id)
    return {
        "active": manager.active_count(),
        "count": len(jobs),
        "jobs": [j.to_dict(include_logs=False) for j in jobs],
    }


@router.get("/jobs/{job_id}")
def get_job(job_id: str, since: int = Query(0, ge=0)) -> Dict[str, Any]:
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}")
    return job.to_dict(include_logs=True, since=since)


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> Dict[str, Any]:
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}")
    if not manager.cancel(job_id):
        raise HTTPException(status_code=409, detail="Job has already finished")
    return {
        "job_id": job_id,
        "cancelling": True,
        "note": "The current detector finishes before the job stops.",
    }


@router.get("/jobs/{job_id}/events")
async def stream_job(job_id: str, request: Request) -> StreamingResponse:
    """Server-sent events: incremental log lines plus step/status changes."""
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}")

    async def generator():
        since = 0
        last_signature: Optional[str] = None
        while True:
            if await request.is_disconnected():
                return

            current = manager.get(job_id)
            if current is None:
                return

            payload = current.to_dict(include_logs=True, since=since)
            since = payload["log_seq"]

            signature = json.dumps(
                {"status": payload["status"], "steps": payload["steps"]}, sort_keys=True
            )
            if payload["logs"] or signature != last_signature:
                last_signature = signature
                yield f"data: {json.dumps(payload)}\n\n"

            if payload["status"] in ("succeeded", "failed", "cancelled"):
                yield "event: done\ndata: {}\n\n"
                return

            await asyncio.sleep(STREAM_POLL_SECONDS)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
