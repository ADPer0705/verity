from __future__ import annotations
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List
from pydantic import BaseModel, Field


class ArtifactType(str, Enum):
    FRAME = "frame"
    FACE_CROP = "face_crop"
    AUDIO_CLIP = "audio_clip"
    METADATA = "metadata"
    LANDMARKS = "landmarks"
    FREQUENCY_SPECTRUM = "frequency_spectrum"


class Artifact(BaseModel):
    artifact_type: ArtifactType
    path: Path
    source_media_id: str
    params: Dict[str, Any] = Field(default_factory=dict)


class ArtifactCollection(BaseModel):
    media_id: str
    artifacts: List[Artifact] = Field(default_factory=list)
