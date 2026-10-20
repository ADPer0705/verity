from __future__ import annotations
import json
from pathlib import Path
from typing import List, Optional
from verity.models.findings import Finding, FindingsCollection, Classification


class FindingsCompiler:
    """Collects ``finding.json`` files for a media id into one sorted collection.

    ``definitions`` (detector id -> DetectorDefinition) lets findings written
    before trust tiers existed be labelled from the current registry instead
    of silently passing as ``trained``.
    """

    def __init__(self, workspace: Path, definitions: Optional[dict] = None):
        self.workspace = workspace
        self.output_dir = workspace / "output"
        self.definitions = definitions or {}

    def compile(self, media_id: str) -> FindingsCollection:
        findings: List[Finding] = []

        self.output_dir.mkdir(parents=True, exist_ok=True)
        for subdir in sorted(self.output_dir.iterdir()):
            if not subdir.name.startswith(f"{media_id}_"):
                continue
            result_path = subdir / "finding.json"
            if result_path.exists():
                with open(result_path) as f:
                    data = json.load(f)
                if "tier" not in data:
                    definition = self.definitions.get(data.get("detector_id"))
                    if definition is not None:
                        data["tier"] = definition.weights.tier
                findings.append(Finding(**data))

        # Deterministic ordering: findings are presented in detector-id order.
        findings.sort(key=lambda f: f.detector_id)

        aggregated = self._aggregate(findings) if findings else None

        return FindingsCollection(
            media_id=media_id,
            findings=findings,
            aggregated_score=aggregated,
        )

    @staticmethod
    def _aggregate(findings: List[Finding]) -> Optional[float]:
        # Untrained heads are research artifacts; they never move the score.
        findings = [f for f in findings if f.is_trained]
        if not findings:
            return None
        weighted = sum(f.confidence * (1.0 if f.classification == Classification.FAKE else 0.0)
                       for f in findings)
        total = sum(f.confidence for f in findings)
        return weighted / total if total > 0 else None
