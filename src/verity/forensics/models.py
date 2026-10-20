"""Pydantic models for the model-free forensic analyses.

These analyses complement the ML detectors: each one is a classic signal
processing or metadata technique whose output an investigator can explain
without reference to a trained model. They never produce a real/fake
verdict; they produce *observations* (a sentence plus the numbers behind it)
with a severity that says how much attention the observation deserves.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field

Severity = Literal["info", "notice", "warning"]

#: Sort rank: the most attention-worthy observations come first.
SEVERITY_RANK: Dict[str, int] = {"warning": 0, "notice": 1, "info": 2}


class Observation(BaseModel):
    """One explainable statement backed by numbers.

    ``severity`` is a reading aid, not a verdict:

    * ``info``: a neutral fact worth recording (e.g. "no audio track").
    * ``notice``: something that deviates from what an untouched camera file
      would show, with benign explanations that remain plausible.
    * ``warning``: an internal inconsistency that is hard to explain without
      some form of processing or editing.
    """

    analyzer: str
    severity: Severity = "info"
    statement: str
    evidence: Dict[str, Any] = Field(default_factory=dict)


class ForensicResult(BaseModel):
    """Everything one analyzer produced for one media item."""

    analyzer: str
    label: str
    method: str = ""
    ok: bool = True
    observations: List[Observation] = Field(default_factory=list)
    artifacts: List[str] = Field(default_factory=list)
    data: Dict[str, Any] = Field(default_factory=dict)


class ForensicsReport(BaseModel):
    """All forensic results for a media item plus a flat, sorted view of the
    observations (severity first, then analyzer, then statement)."""

    media_id: str
    media_type: str
    results: List[ForensicResult] = Field(default_factory=list)
    observations: List[Observation] = Field(default_factory=list)
    counts: Dict[str, int] = Field(default_factory=dict)

    @classmethod
    def from_results(cls, media_id: str, media_type: str,
                     results: List[ForensicResult]) -> "ForensicsReport":
        results = sorted(results, key=lambda r: r.analyzer)
        flat = [obs for r in results for obs in r.observations]
        flat.sort(key=lambda o: (SEVERITY_RANK.get(o.severity, 9), o.analyzer, o.statement))
        counts = {sev: sum(1 for o in flat if o.severity == sev)
                  for sev in ("warning", "notice", "info")}
        return cls(media_id=media_id, media_type=media_type, results=results,
                   observations=flat, counts=counts)
