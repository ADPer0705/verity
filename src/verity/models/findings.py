from __future__ import annotations
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field


class Classification(str, Enum):
    REAL = "REAL"
    FAKE = "FAKE"
    UNCERTAIN = "UNCERTAIN"


class Reasoning(BaseModel):
    summary: str
    evidence: List[Path] = Field(default_factory=list)
    confidence_factors: List[str] = Field(default_factory=list)


class Finding(BaseModel):
    detector_id: str
    detector_version: str
    classification: Classification
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: Reasoning
    raw_output: Dict[str, Any] = Field(default_factory=dict)
    # trained | backbone-only | random-init | untrained (see Weights.tier).
    # Anything but "trained" is a research artifact, not evidence.
    tier: str = "trained"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_trained(self) -> bool:
        return self.tier == "trained"


class FindingsCollection(BaseModel):
    media_id: str
    findings: List[Finding] = Field(default_factory=list)
    aggregated_score: Optional[float] = None
