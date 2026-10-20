from __future__ import annotations
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class ArtifactRequirement(BaseModel):
    type: str
    params: Dict[str, Any] = Field(default_factory=dict)


class OutputSchema(BaseModel):
    format: str = "json"
    schema_: Dict[str, str] = Field(default_factory=dict, alias="schema")


TIER_TRAINED = "trained"
TIER_BACKBONE = "backbone-only"
TIER_RANDOM = "random-init"
TIER_UNTRAINED = "untrained"


class Weights(BaseModel):
    """Weight provenance for one detector (mirrors the `weights:` block).

    ``kind`` says where the checkpoint comes from; ``trained`` says whether
    the *classification head* has actually been trained for deepfake
    detection. Only trained detectors are selected by default: a randomly
    initialised head emits noise, and noise must never sit next to real
    evidence in an investigator's report without being labelled as such.
    """
    file: str = ""
    kind: str = "official"  # official | backbone | external | random
    url: str = ""
    size: int | None = None
    md5: str = ""
    sha256: str = ""
    note: str = ""
    license: str = ""
    # Explicit override of the trained/untrained derivation below. Needed
    # for ``external`` detectors, where the head may be a released detector
    # (trained), a training-free method (trained), or a random head bolted
    # onto a pretrained backbone (untrained).
    trained: bool | None = None

    @property
    def self_contained(self) -> bool:
        """True when no local checkpoint file is required (weights are either
        downloaded at first run or the model is randomly initialized)."""
        return self.kind in ("external", "random")

    @property
    def is_trained(self) -> bool:
        if self.trained is not None:
            return self.trained
        return self.kind == "official"

    @property
    def tier(self) -> str:
        if self.is_trained:
            return TIER_TRAINED
        if self.kind == "backbone":
            return TIER_BACKBONE
        if self.kind == "random":
            return TIER_RANDOM
        return TIER_UNTRAINED

class DetectorCapabilities(BaseModel):
    """What kinds of media a detector can consume.

    Derived from ``artifacts.required`` in the detector definition:
      - ``video``      -> video detectors: need a video, analyze faces temporally
      - ``face_crop``  -> frame detectors: analyze faces in images and videos
      - ``audio``      -> audio detectors (none shipped yet)
    """
    media_types: List[str] = Field(default_factory=list)  # subset of image|video|audio
    faces_required: bool = False
    audio_required: bool = False  # needs an audio track (audio detectors on video)

    def accepts(self, media_type: str) -> bool:
        return media_type in self.media_types


class DetectorDefinition(BaseModel):
    detector_id: str
    name: str = ""
    version: str
    description: str = ""
    architecture: str = ""
    pretrained: str = ""
    # Manipulation families this detector was trained to recognise
    # (ids from verity.models.taxonomy.FAMILIES). Empty = unspecified.
    targets: List[str] = Field(default_factory=list)
    artifacts: Dict[str, List[ArtifactRequirement]] = Field(default_factory=dict)
    output: OutputSchema = Field(default_factory=OutputSchema)
    weights: Weights = Field(default_factory=Weights)
    video_mode: bool = False
    clip_size: Optional[int] = None
    sample_interval: int = 1
    min_face: int = 30
    cam: bool = True
    resolution: int = 256
    # face_crop | face_clip | full_frame | full_clip | audio (see detectors/common/spec.py)
    input_mode: str = ""

    @property
    def effective_input_mode(self) -> str:
        if self.input_mode:
            return self.input_mode
        return "face_clip" if self.video_mode else "face_crop"

    @property
    def capabilities(self) -> DetectorCapabilities:
        mode = self.effective_input_mode
        required = {r.type for r in self.artifacts.get("required", [])}
        if mode == "audio" or "audio" in required:
            # Audio detectors consume standalone audio and the soundtrack of videos.
            return DetectorCapabilities(media_types=["audio", "video"], faces_required=False,
                                        audio_required=True)
        if mode == "full_clip":
            return DetectorCapabilities(media_types=["video"], faces_required=False)
        if mode == "full_frame":
            return DetectorCapabilities(media_types=["image", "video"], faces_required=False)
        # Face-based detectors: a `video` requirement means temporal face
        # analysis, `face_crop` means per-frame face analysis.
        faces_required = bool(required & {"video", "face_crop", "face"}) or mode in ("face_crop", "face_clip")
        if "video" in required or mode == "face_clip":
            media_types = ["video"]
        else:
            media_types = ["image", "video"]
        return DetectorCapabilities(media_types=media_types, faces_required=faces_required)
