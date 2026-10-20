from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
from verity.models.findings import Classification, Reasoning, Finding


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
    evidence_paths = []
    if "heatmap" in raw:
        ev = Path(raw["heatmap"])
        if ev.exists():
            evidence_paths.append(ev)

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
