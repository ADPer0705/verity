from __future__ import annotations
from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field
from verity.models.media import MediaInfo
from verity.models.findings import FindingsCollection
from verity.analyze.models import MediaAnalysis, DetectorSelection


class Report(BaseModel):
    media: MediaInfo
    findings: FindingsCollection
    analysis: Optional[MediaAnalysis] = None
    selection: Optional[DetectorSelection] = None
    summary: str = ""
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
