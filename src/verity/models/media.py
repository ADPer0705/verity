from __future__ import annotations
from enum import Enum
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class MediaType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    UNKNOWN = "unknown"


class MediaInfo(BaseModel):
    original_path: Path
    media_type: MediaType = MediaType.UNKNOWN
    checksum: str = ""
    size_bytes: int = 0
    mime_type: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MediaCopy(BaseModel):
    media_id: str
    original: MediaInfo
    copy_path: Path
    verified: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
