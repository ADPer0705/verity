from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
from verity.models.findings import Classification, Reasoning, Finding


def discover_evidence_paths(output_path: Path, raw: Dict[str, Any]) -> List[Path]:
    """Collect generated explainability artifacts from a detector output folder."""
    evidence_paths: List[Path] = []
    if "heatmap" in raw:
        ev = Path(raw["heatmap"])
        if ev.exists():
            evidence_paths.append(ev)
    if not output_path or not output_path.exists():
        return evidence_paths

    timeline = output_path / "timeline.png"
    if timeline.exists():
        evidence_paths.append(timeline)

    for folder, pattern, limit in (
        ("full_frames", "*_frame.png", 3),
        ("attention_maps", "*_attention.png", 4),
        ("reconstructions", "*_recon.png", 4),
        ("faces", "*.png", 4),
        ("frames", "*.png", 4),
    ):
        subdir = output_path / folder
        if subdir.is_dir():
            files = sorted(subdir.glob(pattern), key=lambda p: p.name)[:limit]
            evidence_paths.extend(files)
    return evidence_paths


def parse(
    raw: Dict[str, Any],
    output_path: Path,
    version: str,
    detector_id: str,
) -> Finding:
    prediction = raw.get("prediction", 0.5)
    confidence = raw.get("confidence", 0.0)

    if prediction > 0.7:
        classification = Classification.FAKE
    elif prediction < 0.3:
        classification = Classification.REAL
    else:
        classification = Classification.UNCERTAIN

    reasoning_text = raw.get("reasoning", "No reasoning provided.")
    evidence_paths = discover_evidence_paths(output_path, raw)

    reasoning = Reasoning(
        summary=reasoning_text,
        evidence=evidence_paths,
        confidence_factors=(raw.get("confidence_factors") or []),
    )

    return Finding(
        detector_id=detector_id,
        detector_version=version,
        classification=classification,
        confidence=confidence,
        reasoning=reasoning,
        raw_output=raw,
    )
