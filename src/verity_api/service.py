"""Service layer: the pipeline objects the CLI uses, exposed as API operations.

Every function here is a thin adapter over ``verity.*``. Where the CLI prints
to stdout, these report through a :class:`~verity_api.jobs.JobContext` instead.
Nothing in this module decides anything a detector or the selector has not
already decided.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from verity.analyze.detector_selector import DetectorSelector
from verity.analyze.media_analyzer import MediaAnalyzer
from verity.analyze.models import DetectorSelection, MediaAnalysis
from verity.compile.findings_compiler import FindingsCompiler
from verity.detect.detector_registry import DetectorRegistry
from verity.detect.detector_runner import DetectorRunner
from verity.extract.audio_extractor import AudioExtractor
from verity.extract.face_extractor import FaceExtractor
from verity.extract.frame_extractor import FrameExtractor
from verity.extract.metadata_extractor import MetadataExtractor
from verity.forensics import ForensicsRunner, load_report as _load_forensics_report
from verity.ingest.media_ingestor import MediaIngestor
from verity.models.findings import Classification, Finding, FindingsCollection
from verity.models.media import MediaCopy, MediaInfo, MediaType
from verity.report.report_generator import ReportGenerator

from verity_api.config import Settings
from verity_api.jobs import JobContext

EXTRACTOR_CLASSES = {
    "frame": FrameExtractor,
    "face": FaceExtractor,
    "metadata": MetadataExtractor,
    "audio": AudioExtractor,
}

#: Probability above which a per-detector mean reads as "leaning manipulated".
LEAN_FAKE = 0.6
LEAN_REAL = 0.4


class PipelineError(RuntimeError):
    """A pipeline precondition the user can act on (surfaced verbatim)."""


# --------------------------------------------------------------------- #
#  Registry access                                                       #
# --------------------------------------------------------------------- #


def build_registry(settings: Settings) -> DetectorRegistry:
    """Scan ``detectors/`` fresh.

    Deliberately uncached: provisioning a detector or dropping in weights
    should show up on the next request without restarting the server.
    """
    return DetectorRegistry(settings.detectors_dir)


def coverage(settings: Settings) -> List[Dict[str, Any]]:
    """Per manipulation family: which detectors cover it, by tier."""
    from verity.models.taxonomy import FAMILIES
    registry = build_registry(settings)
    out = []
    for fam_id, meta in FAMILIES.items():
        trained, untrained = [], []
        for det_id, d in sorted(registry.list_detectors().items()):
            if fam_id in d.targets:
                (trained if d.weights.is_trained else untrained).append(det_id)
        out.append({"id": fam_id, **meta, "trained": trained, "untrained": untrained,
                    "covered": bool(trained)})
    return out


def detector_catalog(settings: Settings) -> List[Dict[str, Any]]:
    registry = build_registry(settings)
    catalog: List[Dict[str, Any]] = []
    for detector_id, definition in sorted(registry.list_detectors().items()):
        caps = definition.capabilities
        weights = definition.weights
        catalog.append(
            {
                "detector_id": detector_id,
                "name": definition.name or detector_id,
                "version": definition.version,
                "description": definition.description,
                "media_types": caps.media_types,
                "faces_required": caps.faces_required,
                "video_mode": definition.video_mode,
                "clip_size": definition.clip_size,
                "min_face": definition.min_face,
                "resolution": definition.resolution,
                "explainability": definition.cam,
                "tier": weights.tier,
                "trained": weights.is_trained,
                "targets": list(definition.targets),
                "weights": {
                    "file": weights.file,
                    "kind": weights.kind,
                    "license": weights.license,
                    "url": weights.url,
                    "size": weights.size,
                    "note": weights.note,
                    "self_contained": weights.self_contained,
                    "available": registry.weights_available(detector_id),
                },
            }
        )
    return catalog


# --------------------------------------------------------------------- #
#  Media / case listing                                                  #
# --------------------------------------------------------------------- #


def _media_meta_path(settings: Settings, media_id: str) -> Path:
    return settings.media_dir / f".{media_id}.json"


def load_media_info(settings: Settings, media_id: str) -> Optional[MediaInfo]:
    path = _media_meta_path(settings, media_id)
    if not path.exists():
        return None
    return MediaInfo(**json.loads(path.read_text()))


def resolve_media_file(settings: Settings, media_id: str) -> Optional[Path]:
    if not settings.media_dir.exists():
        return None
    for candidate in sorted(settings.media_dir.iterdir()):
        if candidate.is_file() and candidate.stem == media_id:
            return candidate
    return None


def list_media_ids(settings: Settings) -> List[str]:
    if not settings.media_dir.exists():
        return []
    ids = {p.stem.lstrip(".") for p in settings.media_dir.glob(".*.json")}
    return sorted(ids)


def _read_model(path: Path, model):
    if not path.exists():
        return None
    try:
        return model(**json.loads(path.read_text()))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def load_analysis(settings: Settings, media_id: str) -> Optional[MediaAnalysis]:
    return _read_model(settings.report_dir(media_id) / "analysis.json", MediaAnalysis)


def load_selection(settings: Settings, media_id: str) -> Optional[DetectorSelection]:
    return _read_model(settings.report_dir(media_id) / "selection.json", DetectorSelection)


def load_forensics(settings: Settings, media_id: str) -> Optional[Dict[str, Any]]:
    """The model-free forensic analyses (``artifacts/<id>/forensics/forensics.json``)."""
    report = _load_forensics_report(settings.workspace, media_id)
    return report.model_dump(mode="json") if report is not None else None


def load_findings(settings: Settings, media_id: str) -> FindingsCollection:
    return FindingsCompiler(settings.workspace, build_registry(settings).list_detectors()).compile(media_id)


def report_files(settings: Settings, media_id: str) -> Dict[str, Any]:
    report_dir = settings.report_dir(media_id)
    out: Dict[str, Any] = {}
    for fmt in ("html", "pdf"):
        path = report_dir / f"report.{fmt}"
        out[fmt] = (
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "generated_at": datetime.fromtimestamp(
                    path.stat().st_mtime, tz=timezone.utc
                ).isoformat(),
            }
            if path.exists()
            else None
        )
    return out


# --------------------------------------------------------------------- #
#  Findings interpretation                                               #
# --------------------------------------------------------------------- #


def mean_fake_probability(finding: Finding) -> float:
    """Same derivation the PDF report uses, so UI and report never disagree."""
    per_face = [
        f.get("fake_prob")
        for f in (finding.raw_output.get("per_face") or [])
        if isinstance(f.get("fake_prob"), (int, float))
    ]
    if per_face:
        return sum(per_face) / len(per_face)
    prediction = finding.raw_output.get("prediction")
    if isinstance(prediction, (int, float)):
        return float(prediction)
    return 0.0


def faces_analyzed(finding: Finding) -> Optional[int]:
    stats = (finding.raw_output.get("stats") or {})
    if stats.get("count") is not None:
        return int(stats["count"])
    per_face = (finding.raw_output.get("per_face") or [])
    return len(per_face) or None


def finding_to_dict(
    settings: Settings, media_id: str, finding: Finding, definition_name: str = "",
    targets: Optional[List[str]] = None,
) -> Dict[str, Any]:
    raw = finding.raw_output
    out_dir = settings.detector_output_dir / f"{media_id}_{finding.detector_id}"
    return {
        "detector_id": finding.detector_id,
        "detector_name": definition_name or finding.detector_id,
        "detector_version": finding.detector_version,
        "targets": list(targets or []),
        "tier": finding.tier,
        "trained": finding.is_trained,
        "classification": finding.classification.value,
        "confidence": finding.confidence,
        "mean_fake_probability": mean_fake_probability(finding),
        "faces_analyzed": faces_analyzed(finding),
        "reasoning": {
            "summary": finding.reasoning.summary,
            "confidence_factors": list(finding.reasoning.confidence_factors),
            "evidence": [str(p) for p in finding.reasoning.evidence],
        },
        "stats": (raw.get("stats") or {}),
        "sampling": (raw.get("sampling") or {}),
        "region_aggregate": (raw.get("region_aggregate") or {}),
        "per_face": (raw.get("per_face") or []),
        "timestamp": finding.timestamp.isoformat(),
        "output_dir": str(out_dir) if out_dir.exists() else None,
        "has_logs": (out_dir / "stderr.log").exists() or (out_dir / "stdout.log").exists(),
    }


def consensus(findings: Iterable[Finding]) -> Dict[str, Any]:
    """Aggregate independent detector findings without overriding any of them.

    Verity's premise is that a single score hides the disagreement that
    matters, so this reports the split as a first-class fact: the band is
    never stronger than the agreement behind it, and a divided panel resolves
    to ``inconclusive`` no matter how confident individual detectors are.
    """
    all_findings = list(findings)
    # Untrained heads never take part in the consensus: they are research
    # artifacts, and a panel vote that counts noise is not a panel vote.
    findings = [f for f in all_findings if f.is_trained]
    untrained_excluded = len(all_findings) - len(findings)
    total = len(findings)
    if total == 0:
        return {
            "band": "no_analysis",
            "total": 0,
            "fake": 0,
            "real": 0,
            "uncertain": 0,
            "agreement": 0.0,
            "mean_fake_probability": None,
            "disagreement": False,
            "leaning_split": False,
            "untrained_excluded": untrained_excluded,
        }

    fake = sum(1 for f in findings if f.classification is Classification.FAKE)
    real = sum(1 for f in findings if f.classification is Classification.REAL)
    uncertain = total - fake - real

    probabilities = [mean_fake_probability(f) for f in findings]
    mean_probability = sum(probabilities) / total

    # A detector "leans" when its own mean probability is decisive, regardless
    # of the label it emitted; a panel that leans both ways is a real split.
    leans_fake = sum(1 for p in probabilities if p >= LEAN_FAKE)
    leans_real = sum(1 for p in probabilities if p <= LEAN_REAL)
    leaning_split = leans_fake > 0 and leans_real > 0

    agreement = max(fake, real) / total
    disagreement = fake > 0 and real > 0

    if disagreement and min(fake, real) / total >= 0.25:
        band = "inconclusive"
    elif fake / total >= 0.7 and mean_probability >= LEAN_FAKE:
        band = "strong_manipulation"
    elif fake > real:
        band = "likely_manipulation"
    elif real / total >= 0.7 and mean_probability <= LEAN_REAL:
        band = "strong_authentic"
    elif real > fake:
        band = "likely_authentic"
    else:
        band = "inconclusive"

    return {
        "band": band,
        "total": total,
        "fake": fake,
        "real": real,
        "uncertain": uncertain,
        "agreement": round(agreement, 4),
        "mean_fake_probability": round(mean_probability, 4),
        "disagreement": disagreement,
        "leaning_split": leaning_split,
        "leans_fake": leans_fake,
        "leans_real": leans_real,
        "untrained_excluded": untrained_excluded,
    }


def case_summary(settings: Settings, media_id: str) -> Optional[Dict[str, Any]]:
    """One row in the investigator case list / the citizen's result header."""
    info = load_media_info(settings, media_id)
    if info is None:
        return None

    findings = load_findings(settings, media_id)
    selection = load_selection(settings, media_id)
    analysis = load_analysis(settings, media_id)
    reports = report_files(settings, media_id)

    selected = len(selection.selected) if selection else 0
    excluded = (
        len([d for d in selection.decisions if not d.included]) if selection else 0
    )

    return {
        "media_id": media_id,
        "file_name": info.metadata.get("original_name") or Path(info.original_path).name,
        "media_type": info.media_type.value,
        "mime_type": info.mime_type,
        "size_bytes": info.size_bytes,
        "checksum": info.checksum,
        "ingested_at": info.metadata.get("ingested_at"),
        "has_analysis": analysis is not None,
        "has_selection": selection is not None,
        "detectors_selected": selected,
        "detectors_excluded": excluded,
        "findings_count": len(findings.findings),
        "aggregated_score": findings.aggregated_score,
        "consensus": consensus(findings.findings),
        "reports": reports,
        "face_count": analysis.media.face_count if analysis else None,
        "duration_seconds": analysis.media.duration_seconds if analysis else None,
        "width": analysis.media.width if analysis else None,
        "height": analysis.media.height if analysis else None,
    }


def list_cases(settings: Settings) -> List[Dict[str, Any]]:
    cases = [case_summary(settings, mid) for mid in list_media_ids(settings)]
    cases = [c for c in cases if c]
    cases.sort(key=lambda c: (c.get("ingested_at") or "", c["media_id"]), reverse=True)
    return cases


# --------------------------------------------------------------------- #
#  Artifacts and detector assets                                         #
# --------------------------------------------------------------------- #


def _file_entry(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {
        "name": path.name,
        "path": str(path),
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "suffix": path.suffix.lstrip(".").lower(),
    }


def artifact_tree(settings: Settings, media_id: str) -> List[Dict[str, Any]]:
    """Extracted artifacts grouped by their directory under ``artifacts/<id>/``."""
    root = settings.artifacts_dir / media_id
    if not root.exists():
        return []

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        group = str(path.parent.relative_to(root)) or "."
        groups.setdefault(group, []).append(_file_entry(path))

    return [
        {"group": group, "count": len(files), "files": files}
        for group, files in sorted(groups.items())
    ]


def detector_assets(settings: Settings, media_id: str, detector_id: str) -> Dict[str, Any]:
    """Explainability images produced by one detector run.

    Mirrors what :class:`ReportGenerator` embeds in the PDF, but served as
    individual files so the investigator can zoom and compare them.
    """
    out_dir = settings.detector_output_dir / f"{media_id}_{detector_id}"
    if not out_dir.exists():
        return {"exists": False, "timeline": None, "faces": [], "attention_maps": [], "reconstructions": []}

    def images(directory: Path, pattern: str) -> List[Dict[str, Any]]:
        if not directory.exists():
            return []
        return [_file_entry(p) for p in sorted(directory.glob(pattern), key=lambda p: p.name)]

    timeline = out_dir / "timeline.png"
    return {
        "exists": True,
        "output_dir": str(out_dir),
        "timeline": _file_entry(timeline) if timeline.exists() else None,
        "faces": images(out_dir / "faces", "*.png"),
        "attention_maps": images(out_dir / "attention_maps", "*_attention.png"),
        "reconstructions": images(out_dir / "reconstructions", "*_recon.png"),
    }


def detector_logs(settings: Settings, media_id: str, detector_id: str) -> Dict[str, str]:
    out_dir = settings.detector_output_dir / f"{media_id}_{detector_id}"
    result: Dict[str, str] = {}
    for name in ("stdout.log", "stderr.log"):
        path = out_dir / name
        result[name.replace(".log", "")] = path.read_text(errors="replace") if path.exists() else ""
    return result


def detector_raw(settings: Settings, media_id: str, detector_id: str) -> Dict[str, Any]:
    out_dir = settings.detector_output_dir / f"{media_id}_{detector_id}"
    payload: Dict[str, Any] = {}
    for name in ("result.json", "finding.json"):
        path = out_dir / name
        if path.exists():
            try:
                payload[name.replace(".json", "")] = json.loads(path.read_text())
            except json.JSONDecodeError as exc:
                payload[name.replace(".json", "")] = {"_parse_error": str(exc)}
    return payload


# --------------------------------------------------------------------- #
#  Integrity verification                                                #
# --------------------------------------------------------------------- #


def verify_integrity(settings: Settings, media_id: str) -> Dict[str, Any]:
    """Recompute the SHA-256 of the working copy and compare it to ingest.

    This is the manual check an investigator is expected to be able to run on
    demand and cite: it proves the file the detectors read is byte-identical
    to the file that was admitted.
    """
    info = load_media_info(settings, media_id)
    if info is None:
        raise PipelineError(f"No ingested media with id {media_id}")

    copy_path = resolve_media_file(settings, media_id)
    if copy_path is None:
        raise PipelineError(f"Working copy for {media_id} is missing from the workspace")

    ingestor = MediaIngestor(settings.workspace)
    actual = ingestor._compute_checksum(copy_path)  # noqa: SLF001 - same package contract

    original = Path(info.original_path)
    original_state = "absent"
    original_matches: Optional[bool] = None
    if original.exists() and original.is_file():
        original_state = "present"
        original_matches = ingestor._compute_checksum(original) == info.checksum  # noqa: SLF001

    return {
        "media_id": media_id,
        "expected_checksum": info.checksum,
        "actual_checksum": actual,
        "matches": actual == info.checksum,
        "working_copy": str(copy_path),
        "working_copy_size": copy_path.stat().st_size,
        "recorded_size": info.size_bytes,
        "size_matches": copy_path.stat().st_size == info.size_bytes,
        "original_path": str(original),
        "original_state": original_state,
        "original_matches": original_matches,
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }


# --------------------------------------------------------------------- #
#  Mutating operations (job workers)                                     #
# --------------------------------------------------------------------- #


def ingest_file(settings: Settings, source: Path) -> MediaCopy:
    return MediaIngestor(settings.workspace).ingest(source)


def delete_case(settings: Settings, media_id: str) -> Dict[str, Any]:
    """Remove one case: working copy, metadata, artifacts, outputs, reports."""
    removed: List[str] = []

    copy_path = resolve_media_file(settings, media_id)
    if copy_path is not None:
        copy_path.chmod(0o666)
        copy_path.unlink()
        removed.append(str(copy_path))

    meta_path = _media_meta_path(settings, media_id)
    if meta_path.exists():
        meta_path.unlink()
        removed.append(str(meta_path))

    artifacts = settings.artifacts_dir / media_id
    if artifacts.exists():
        shutil.rmtree(artifacts, ignore_errors=True)
        removed.append(str(artifacts))

    if settings.detector_output_dir.exists():
        for directory in sorted(settings.detector_output_dir.iterdir()):
            if directory.is_dir() and directory.name.startswith(f"{media_id}_"):
                shutil.rmtree(directory, ignore_errors=True)
                removed.append(str(directory))

    report_dir = settings.report_dir(media_id)
    if report_dir.exists():
        shutil.rmtree(report_dir, ignore_errors=True)
        removed.append(str(report_dir))

    return {"media_id": media_id, "removed": removed}


def _extractors(settings: Settings, keys: Optional[List[str]] = None):
    chosen = keys or settings.extractors
    return [EXTRACTOR_CLASSES[k](settings.workspace) for k in chosen if k in EXTRACTOR_CLASSES]


def run_extract(ctx: JobContext, settings: Settings, media_id: str,
                keys: Optional[List[str]] = None) -> Dict[str, Any]:
    """``verity extract`` — frames, faces, metadata, audio.

    Each extractor is isolated: one failing (a missing external tool, an
    unreadable stream) is reported loudly and leaves the others intact,
    because losing an entire examination to an optional artifact type is
    never the right trade.
    """
    results: Dict[str, int] = {}
    failures: Dict[str, str] = {}
    total = 0

    for extractor in _extractors(settings, keys):
        ctx.raise_if_cancelled()
        name = extractor.__class__.__name__
        try:
            collection = extractor.extract(media_id)
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            failures[name] = f"{type(exc).__name__}: {exc}"
            results[name] = 0
            ctx.warn(f"  {name} failed: {failures[name]}")
            continue

        count = len(collection.artifacts)
        results[name] = count
        total += count
        ctx.log(f"  {name}: {count} artifact(s)")

    return {
        "media_id": media_id,
        "per_extractor": results,
        "failed_extractors": failures,
        "total_artifacts": total,
    }


def run_forensics(ctx: JobContext, settings: Settings, media_id: str,
                  media_type: MediaType) -> Dict[str, Any]:
    """``verity forensics`` — model-free analyses; each analyzer fails soft."""
    report = ForensicsRunner(settings.workspace).run(media_id, media_type)
    for result in report.results:
        ctx.log(f"  {result.label}: {len(result.observations)} observation(s)"
                + ("" if result.ok else " (failed)"))
    return {
        "media_id": media_id,
        "analyzers": [r.analyzer for r in report.results],
        "counts": report.counts,
        "path": str(ForensicsRunner(settings.workspace).report_path(media_id)),
    }


def run_analyze(ctx: JobContext, settings: Settings, media_id: str,
                media_type: MediaType) -> MediaAnalysis:
    """``verity analyze`` — characterize the media and scan for faces."""
    analyzer = MediaAnalyzer(settings.media_dir)
    analysis = analyzer.analyze(media_id, media_type)
    characteristics = analysis.media
    ctx.log(
        f"  {characteristics.format or 'n/a'}, "
        f"{characteristics.face_count} face(s) in "
        f"{characteristics.sampled_frames} sampled frames"
    )
    analyzer.save(analysis, settings.report_dir(media_id))
    return analysis


def run_select(ctx: JobContext, settings: Settings, analysis: MediaAnalysis,
               override: Optional[List[str]] = None,
               include_untrained: bool = False) -> tuple[DetectorSelection, List[str]]:
    """``verity analyze``'s selection stage, including the CLI's override rules.

    An explicit override list is an explicit request, so untrained detectors
    named in it run (flagged); auto-selection stays evidence-grade unless
    ``include_untrained`` is set.
    """
    registry = build_registry(settings)
    selector = DetectorSelector(
        registry, offline=settings.offline,
        include_untrained=include_untrained or bool(override),
    )
    selection = selector.select(analysis)

    known = set(registry.list_detectors())
    if override:
        unknown = [d for d in override if d not in known]
        if unknown:
            ctx.warn(f"  Unknown detectors ignored: {', '.join(sorted(unknown))}")
        selected = sorted(set(override) & known)
        # Keep the selector's reasoning intact, but mark what actually ran —
        # identical to the CLI so the saved selection matches the report.
        selection = selection.model_copy(deep=True)
        override_set = set(selected)
        for decision in selection.decisions:
            decision.included = decision.detector_id in override_set
    else:
        selected = selection.selected

    excluded = len([d for d in selection.decisions if not d.included])
    ctx.log(
        f"  {len(selected)} detector(s) selected"
        + ("" if override else " (auto)")
        + f", {excluded} excluded"
    )
    selector.save(selection, settings.report_dir(analysis.media.media_id))
    return selection, selected


def run_detectors(ctx: JobContext, settings: Settings, media_id: str,
                  detector_ids: List[str]) -> List[Dict[str, Any]]:
    registry = build_registry(settings)
    runner = DetectorRunner(settings.workspace, registry)
    results: List[Dict[str, Any]] = []

    for index, detector_id in enumerate(detector_ids, start=1):
        ctx.raise_if_cancelled()
        step_key = f"detect:{detector_id}"
        definition = registry.get(detector_id)
        label = (definition.name if definition and definition.name else detector_id)
        ctx.start_step(step_key, f"Run {label} ({index}/{len(detector_ids)})")
        try:
            finding = runner.run(media_id, detector_id)
        except Exception as exc:  # noqa: BLE001 - one detector must not sink the run
            ctx.fail_step(step_key, f"{type(exc).__name__}: {exc}")
            results.append({"detector_id": detector_id, "error": f"{type(exc).__name__}: {exc}"})
            continue

        probability = mean_fake_probability(finding)
        ctx.finish_step(
            step_key,
            f"{finding.classification.value} — mean fake probability "
            f"{probability:.4f}, confidence {finding.confidence:.4f}",
        )
        results.append(
            {
                "detector_id": detector_id,
                "classification": finding.classification.value,
                "confidence": finding.confidence,
                "mean_fake_probability": probability,
            }
        )
    return results


def run_compile(ctx: JobContext, settings: Settings, media_id: str) -> FindingsCollection:
    """``verity compile``."""
    collection = FindingsCompiler(settings.workspace, build_registry(settings).list_detectors()).compile(media_id)
    for finding in collection.findings:
        ctx.log(
            f"  {finding.detector_id}: {finding.classification.value} "
            f"({finding.confidence:.4f})"
        )
    return collection


def run_report(ctx: JobContext, settings: Settings, media_id: str, fmt: str = "pdf",
               analysis: Optional[MediaAnalysis] = None,
               selection: Optional[DetectorSelection] = None) -> Dict[str, Any]:
    """``verity report`` — always writes HTML; PDF additionally when asked."""
    registry = build_registry(settings)
    generator = ReportGenerator(settings.workspace, registry.list_detectors())

    analysis = analysis or load_analysis(settings, media_id)
    selection = selection or load_selection(settings, media_id)
    report = generator.generate(media_id, analysis=analysis, selection=selection)

    report_dir = settings.report_dir(media_id)
    report_dir.mkdir(parents=True, exist_ok=True)
    html_path = report_dir / "report.html"

    if fmt == "html":
        generator.save_html(report, html_path)
        ctx.log(f"  HTML report: {html_path}")
        return {"media_id": media_id, "format": "html", "html": str(html_path), "pdf": None}

    # ``save`` writes the HTML and renders the PDF beside it.
    generator.save(report, html_path)
    pdf_path = html_path.with_suffix(".pdf")
    ctx.log(f"  PDF report: {pdf_path}")
    return {"media_id": media_id, "format": "pdf", "html": str(html_path), "pdf": str(pdf_path)}


def run_full_pipeline(
    ctx: JobContext,
    settings: Settings,
    media_id: str,
    detector_override: Optional[List[str]] = None,
    include_untrained: bool = False,
    extractors: Optional[List[str]] = None,
    fmt: str = "pdf",
    skip_report: bool = False,
) -> Dict[str, Any]:
    """``verity run`` for one already-ingested media item."""
    info = load_media_info(settings, media_id)
    if info is None:
        raise PipelineError(f"No ingested media with id {media_id}")

    ctx.add_media_id(media_id)
    ctx.declare_steps(
        [
            ("extract", "Extract artifacts"),
            ("forensics", "Forensic analyses"),
            ("analyze", "Characterize media"),
            ("select", "Select detectors"),
            ("compile", "Compile findings"),
        ]
        + ([] if skip_report else [("report", "Generate report")])
    )

    ctx.start_step("extract")
    extraction = run_extract(ctx, settings, media_id, extractors)
    failed = extraction["failed_extractors"]
    ctx.finish_step(
        "extract",
        f"{extraction['total_artifacts']} artifact(s)"
        + (f" · {len(failed)} extractor(s) failed" if failed else ""),
    )

    ctx.start_step("forensics")
    forensics = run_forensics(ctx, settings, media_id, info.media_type)
    ctx.finish_step(
        "forensics",
        f"{len(forensics['analyzers'])} analyzer(s) · {forensics['counts'].get('warning', 0)} warning(s), "
        f"{forensics['counts'].get('notice', 0)} notice(s)",
    )

    ctx.start_step("analyze")
    analysis = run_analyze(ctx, settings, media_id, info.media_type)
    ctx.finish_step(
        "analyze",
        f"{analysis.media.face_count} face(s) across "
        f"{analysis.media.sampled_frames} sampled frames",
    )

    ctx.start_step("select")
    selection, selected = run_select(ctx, settings, analysis, detector_override,
                                     include_untrained=include_untrained)
    ctx.finish_step("select", f"{len(selected)} selected")

    if not selected:
        ctx.warn(
            "No detector is eligible for this media. Open the selection tab "
            "to see the exclusion reason for each one."
        )

    detector_results = run_detectors(ctx, settings, media_id, selected)

    ctx.start_step("compile")
    collection = run_compile(ctx, settings, media_id)
    ctx.finish_step("compile", f"{len(collection.findings)} finding(s)")

    report: Optional[Dict[str, Any]] = None
    if not skip_report:
        ctx.start_step("report")
        report = run_report(ctx, settings, media_id, fmt, analysis, selection)
        ctx.finish_step("report", Path(report.get("pdf") or report["html"]).name)

    return {
        "media_id": media_id,
        "selected": selected,
        "detectors": detector_results,
        "findings_count": len(collection.findings),
        "consensus": consensus(collection.findings),
        "forensics": forensics,
        "report": report,
    }


def run_setup(ctx: JobContext, settings: Settings, detector_ids: Optional[List[str]],
              force: bool) -> Dict[str, Any]:
    """``verity setup`` — provision detector environments via uv."""
    registry = build_registry(settings)
    runner = DetectorRunner(settings.workspace, registry)
    target = ", ".join(detector_ids) if detector_ids else "all detectors"
    ctx.log(f"Provisioning {target}{' (forced)' if force else ''} …")
    code = runner.setup(detector_ids=detector_ids, force=force)
    if code != 0:
        raise PipelineError(f"Detector provisioning exited with code {code}")
    ctx.log("Provisioning complete.")
    return {"detector_ids": detector_ids, "force": force, "exit_code": code}
