from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field
from verity.models.media import MediaType


class MediaCharacteristics(BaseModel):
    """Deterministic characterization of one media file."""
    media_id: str
    media_type: MediaType
    format: str = ""
    duration_seconds: Optional[float] = None
    fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    frame_count: Optional[int] = None
    video_codec: str = ""
    has_audio_track: bool = False
    audio_codec: str = ""
    face_present: bool = False
    face_count: int = 0
    sampled_frames: int = 0
    # Largest detected face box dimension (px) across sampled frames, or 0
    # when no faces were found. Used to exclude detectors whose minimum face
    # size exceeds what the media provides.
    max_face_size: int = 0


class MediaAnalysis(BaseModel):
    media: MediaCharacteristics
    # Per-frame face counts, keyed by frame index, sorted ascending. Included
    # so the report can cite exactly which frames contained faces.
    faces_per_frame: List[dict] = Field(default_factory=list)


class SelectionDecision(BaseModel):
    detector_id: str
    included: bool
    reasons: List[str] = Field(default_factory=list)


class DetectorSelection(BaseModel):
    media_id: str
    media_type: MediaType
    decisions: List[SelectionDecision] = Field(default_factory=list)

    @property
    def selected(self) -> List[str]:
        return [d.detector_id for d in self.decisions if d.included]

    def decision(self, detector_id: str) -> Optional[SelectionDecision]:
        for d in self.decisions:
            if d.detector_id == detector_id:
                return d
        return None
