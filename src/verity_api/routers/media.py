"""Media cases: ingest, inspect, verify, browse artifacts, fetch reports."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from verity_api.config import get_settings
from verity_api.files import file_response, resolve_within_roots
from verity_api.service import (
    PipelineError,
    artifact_tree,
    build_registry,
    case_summary,
    consensus,
    delete_case,
    detector_assets,
    detector_logs,
    detector_raw,
    finding_to_dict,
    ingest_file,
    list_cases,
    load_analysis,
    load_findings,
    load_forensics,
    load_media_info,
    load_selection,
    report_files,
    resolve_media_file,
    verify_integrity,
)

router = APIRouter(prefix="/api/media", tags=["media"])

#: Upload ceiling. Forensic video is large, but an unbounded upload endpoint
#: on a workstation tool is a footgun; raise this deliberately if needed.
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB
CHUNK = 1024 * 1024


def _require_case(media_id: str):
    settings = get_settings()
    info = load_media_info(settings, media_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"No ingested media with id {media_id}")
    return settings, info


@router.get("")
def get_cases() -> Dict[str, Any]:
    settings = get_settings()
    return {"count": len(list_media_ids_safe(settings)), "cases": list_cases(settings)}


def list_media_ids_safe(settings) -> List[str]:
    from verity_api.service import list_media_ids

    return list_media_ids(settings)


@router.post("", status_code=201)
async def upload_media(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Ingest an uploaded file.

    The upload is streamed to a temp file first so the checksum is computed
    over exactly the bytes that land on disk — ``MediaIngestor`` then makes
    the read-only working copy and derives the media id from that hash.
    """
    settings = get_settings()
    suffix = Path(file.filename or "upload").suffix

    tmp_dir = Path(tempfile.mkdtemp(prefix="verity_upload_"))
    tmp_path = tmp_dir / (Path(file.filename or "upload").name or f"upload{suffix}")

    written = 0
    try:
        with open(tmp_path, "wb") as handle:
            while chunk := await file.read(CHUNK):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Upload exceeds the {MAX_UPLOAD_BYTES // (1024**3)} GiB limit",
                    )
                handle.write(chunk)

        if written == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        copy = ingest_file(settings, tmp_path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if copy.original.media_type.value == "unknown":
        # Not fatal — the selector will exclude every detector and say why —
        # but the UI should be able to warn at the point of upload.
        pass

    return {
        "media_id": copy.media_id,
        "media_type": copy.original.media_type.value,
        "mime_type": copy.original.mime_type,
        "checksum": copy.original.checksum,
        "size_bytes": copy.original.size_bytes,
        "file_name": copy.original.metadata.get("original_name"),
        "copy_path": str(copy.copy_path),
        "case": case_summary(settings, copy.media_id),
    }


@router.get("/{media_id}")
def get_case(media_id: str) -> Dict[str, Any]:
    settings, info = _require_case(media_id)

    analysis = load_analysis(settings, media_id)
    selection = load_selection(settings, media_id)
    findings = load_findings(settings, media_id)
    definitions = build_registry(settings).list_detectors()

    def detector_name(detector_id: str) -> str:
        definition = definitions.get(detector_id)
        return definition.name if definition and definition.name else detector_id

    def detector_targets(detector_id: str) -> list:
        definition = definitions.get(detector_id)
        return list(definition.targets) if definition else []

    decisions = []
    if selection:
        for decision in selection.decisions:
            decisions.append(
                {
                    "detector_id": decision.detector_id,
                    "detector_name": detector_name(decision.detector_id),
                    "included": decision.included,
                    "reasons": list(decision.reasons),
                }
            )

    return {
        "media_id": media_id,
        "summary": case_summary(settings, media_id),
        "media": {
            "original_path": str(info.original_path),
            "file_name": info.metadata.get("original_name") or Path(info.original_path).name,
            "media_type": info.media_type.value,
            "mime_type": info.mime_type,
            "checksum": info.checksum,
            "size_bytes": info.size_bytes,
            "metadata": info.metadata,
            "working_copy": str(resolve_media_file(settings, media_id) or ""),
        },
        "analysis": analysis.model_dump(mode="json") if analysis else None,
        "selection": {
            "media_type": selection.media_type.value,
            "decisions": decisions,
            "selected": selection.selected,
        }
        if selection
        else None,
        "findings": [
            finding_to_dict(settings, media_id, f, detector_name(f.detector_id),
                            targets=detector_targets(f.detector_id))
            for f in findings.findings
        ],
        "aggregated_score": findings.aggregated_score,
        "consensus": consensus(findings.findings),
        "artifacts": artifact_tree(settings, media_id),
        "forensics": load_forensics(settings, media_id),
        "reports": report_files(settings, media_id),
    }


@router.delete("/{media_id}")
def remove_case(
    media_id: str,
    confirm: bool = Query(False, description="Must be true; deletion is irreversible."),
) -> Dict[str, Any]:
    settings, _ = _require_case(media_id)
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Deletion removes the working copy, artifacts, findings and reports. "
            "Re-send with confirm=true.",
        )
    return delete_case(settings, media_id)


@router.get("/{media_id}/file")
def get_media_file(media_id: str, download: bool = False):
    """Stream the read-only working copy (video/image preview, or download)."""
    settings, info = _require_case(media_id)
    path = resolve_media_file(settings, media_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Working copy missing from the workspace")
    name = info.metadata.get("original_name") or path.name
    return file_response(path, download_name=name, inline=not download)


@router.post("/{media_id}/verify")
def verify_case(media_id: str) -> Dict[str, Any]:
    settings, _ = _require_case(media_id)
    try:
        return verify_integrity(settings, media_id)
    except PipelineError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/{media_id}/artifacts")
def get_artifacts(media_id: str) -> Dict[str, Any]:
    settings, _ = _require_case(media_id)
    groups = artifact_tree(settings, media_id)
    return {
        "media_id": media_id,
        "total": sum(g["count"] for g in groups),
        "groups": groups,
    }


@router.get("/{media_id}/findings")
def get_findings(media_id: str) -> Dict[str, Any]:
    settings, _ = _require_case(media_id)
    findings = load_findings(settings, media_id)
    definitions = build_registry(settings).list_detectors()
    return {
        "media_id": media_id,
        "aggregated_score": findings.aggregated_score,
        "consensus": consensus(findings.findings),
        "findings": [
            finding_to_dict(
                settings,
                media_id,
                f,
                definitions[f.detector_id].name if f.detector_id in definitions else "",
                targets=definitions[f.detector_id].targets if f.detector_id in definitions else None,
            )
            for f in findings.findings
        ],
    }


@router.get("/{media_id}/detectors/{detector_id}/assets")
def get_detector_assets(media_id: str, detector_id: str) -> Dict[str, Any]:
    settings, _ = _require_case(media_id)
    return detector_assets(settings, media_id, detector_id)


@router.get("/{media_id}/detectors/{detector_id}/logs")
def get_detector_logs(media_id: str, detector_id: str) -> Dict[str, str]:
    settings, _ = _require_case(media_id)
    return detector_logs(settings, media_id, detector_id)


@router.get("/{media_id}/detectors/{detector_id}/raw")
def get_detector_raw(media_id: str, detector_id: str) -> Dict[str, Any]:
    settings, _ = _require_case(media_id)
    payload = detector_raw(settings, media_id, detector_id)
    if not payload:
        raise HTTPException(
            status_code=404, detail=f"No raw output stored for {detector_id} on {media_id}"
        )
    return payload


@router.get("/{media_id}/report")
def get_report(media_id: str, format: str = Query("html", pattern="^(pdf|html)$"),
               download: bool = False):
    settings, info = _require_case(media_id)
    reports = report_files(settings, media_id)
    entry = reports.get(format)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=f"No {format.upper()} report has been generated for this case yet",
        )
    stem = Path(info.metadata.get("original_name") or media_id).stem
    return file_response(
        Path(entry["path"]),
        download_name=f"verity-{stem}-{media_id}.{format}",
        inline=not download,
    )


@router.get("/{media_id}/files")
def get_case_file(media_id: str, path: str = Query(..., description="Absolute path on disk")):
    """Serve one artifact by absolute path, confined to the configured roots."""
    settings, _ = _require_case(media_id)
    resolved = resolve_within_roots(settings, path)
    return file_response(resolved)
