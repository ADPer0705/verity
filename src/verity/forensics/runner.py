"""Runs every applicable forensic analyzer for one media item and writes the
combined ``forensics.json``. Analyzers fail soft: an exception turns into a
``notice`` observation so the rest of the examination proceeds."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Type

from verity.forensics.audio_analyzer import AudioAnalyzer
from verity.forensics.base import ForensicAnalyzer
from verity.forensics.compression_analyzer import CompressionAnalyzer
from verity.forensics.metadata_analyzer import MetadataAnalyzer
from verity.forensics.models import ForensicResult, ForensicsReport, Observation
from verity.forensics.noise_analyzer import NoiseAnalyzer
from verity.forensics.video_temporal_analyzer import VideoTemporalAnalyzer

#: Pipeline order; results are stored sorted by analyzer id regardless.
ANALYZERS: Sequence[Type[ForensicAnalyzer]] = (
    MetadataAnalyzer,
    CompressionAnalyzer,
    NoiseAnalyzer,
    VideoTemporalAnalyzer,
    AudioAnalyzer,
)


class ForensicsRunner:
    def __init__(self, workspace: Path, analyzers: Optional[Sequence[Type[ForensicAnalyzer]]] = None):
        self.workspace = Path(workspace)
        self.analyzers = [cls(self.workspace) for cls in (analyzers or ANALYZERS)]

    def forensics_dir(self, media_id: str) -> Path:
        return self.workspace / "artifacts" / media_id / "forensics"

    def report_path(self, media_id: str) -> Path:
        return self.forensics_dir(media_id) / "forensics.json"

    def run(self, media_id: str, media_type) -> ForensicsReport:
        media_type = getattr(media_type, "value", media_type) or "unknown"
        results: List[ForensicResult] = []
        for analyzer in self.analyzers:
            if media_type not in analyzer.media_types:
                continue
            try:
                results.append(analyzer.analyze(media_id, media_type))
            except Exception as exc:  # noqa: BLE001 - fail soft, by design
                results.append(ForensicResult(
                    analyzer=analyzer.analyzer_id, label=analyzer.label, method=analyzer.method,
                    ok=False,
                    observations=[Observation(
                        analyzer=analyzer.analyzer_id, severity="notice",
                        statement=f"analyzer {analyzer.analyzer_id} failed: "
                                  f"{type(exc).__name__}: {str(exc)[:300]}",
                        evidence={"error": type(exc).__name__})],
                ))
        report = ForensicsReport.from_results(media_id, media_type, results)
        self.save(report)
        return report

    def save(self, report: ForensicsReport) -> Path:
        path = self.report_path(report.media_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
        return path

    def load(self, media_id: str) -> Optional[ForensicsReport]:
        return load_report(self.workspace, media_id)


def load_report(workspace: Path, media_id: str) -> Optional[ForensicsReport]:
    path = Path(workspace) / "artifacts" / media_id / "forensics" / "forensics.json"
    if not path.exists():
        return None
    try:
        return ForensicsReport(**json.loads(path.read_text()))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
