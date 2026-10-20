"""Request bodies for the API.

Responses are plain dicts assembled in :mod:`verity_api.service` — they mirror
the pydantic models the pipeline already defines, and re-declaring them here
would only create a second thing to keep in sync.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from verity_api.config import ALL_EXTRACTORS


class RunRequest(BaseModel):
    """``verity run`` for one already-ingested media item."""

    media_id: str
    detectors: Optional[List[str]] = Field(
        default=None,
        description="Detector ids to force. Omit for rule-based auto-selection.",
    )
    extractors: Optional[List[str]] = Field(
        default=None, description=f"Subset of {ALL_EXTRACTORS}. Omit for the configured set."
    )
    format: str = Field(default="pdf", pattern="^(pdf|html)$")
    skip_report: bool = False
    include_untrained: bool = Field(
        default=False,
        description="Also auto-select detectors with an untrained head (research only; "
                    "their findings are flagged and never count toward consensus).",
    )


class AnalyzeRequest(BaseModel):
    """``verity analyze``: characterize + select, without running detectors."""

    media_id: str
    detectors: Optional[List[str]] = None
    include_untrained: bool = False


class ExtractRequest(BaseModel):
    """``verity extract``."""

    media_id: str
    extractors: Optional[List[str]] = None


class CompileRequest(BaseModel):
    """``verity compile``."""

    media_id: str


class ReportRequest(BaseModel):
    """``verity report``."""

    media_id: str
    format: str = Field(default="pdf", pattern="^(pdf|html)$")


class SetupRequest(BaseModel):
    """``verity setup``."""

    detectors: Optional[List[str]] = Field(
        default=None, description="Omit or leave empty to provision every detector."
    )
    force: bool = False


class SettingsUpdate(BaseModel):
    detectors_dir: Optional[str] = None
    workspace: Optional[str] = None
    output_dir: Optional[str] = None
    extractors: Optional[List[str]] = None
    offline: Optional[bool] = None
