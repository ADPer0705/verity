from __future__ import annotations
import json
import os
from pathlib import Path
from typing import List
from urllib.parse import urlparse

from verity.analyze.models import DetectorSelection, MediaAnalysis, SelectionDecision
from verity.detect.detector_registry import DetectorRegistry
from verity.models.media import MediaType


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


class DetectorSelector:
    """Rule-based, deterministic selection of detectors suitable for one media
    file. Every decision records its reasons, so the report can explain both
    what was run and what was skipped and why.

    Decision inputs come from the default analysis (ffprobe characterization
    + Haar face scan) and the per-detector definition (``detector.yaml``):
    media type, face presence/count/scale, estimated frame count, video clip
    size, minimum face size and weight availability.

    Rules (applied in order; the first failure excludes the detector):
      0. detectors whose classification head is not trained (random init, or
         a pretrained backbone with an untrained head) are skipped unless
         ``include_untrained`` is set (``VERITY_INCLUDE_UNTRAINED=1``); they
         exist for researchers who bring their own checkpoint, not as evidence
      1. media type must be consumable by the detector (image/video/audio)
      2. detectors that need faces are skipped when the face scan found none
      3. detectors whose minimum face size exceeds the largest detected face
         are skipped (the detector's own face detector would find nothing)
      4. video-only detectors are skipped when the media is too short to
         provide a full clip (the engine would only score padded frames)
      5. detectors whose checkpoint is missing are skipped; weights that are
         downloaded at first run are allowed unless offline mode is set
         (``VERITY_OFFLINE=1``), in which case only locally-verified weights
         are accepted
    """

    def __init__(self, registry: DetectorRegistry, offline: bool = False,
                 include_untrained: bool = False):
        self.registry = registry
        self.offline = offline or _env_flag("VERITY_OFFLINE")
        self.include_untrained = include_untrained or _env_flag("VERITY_INCLUDE_UNTRAINED")

    def select(self, analysis: MediaAnalysis) -> DetectorSelection:
        decisions: List[SelectionDecision] = []
        for detector_id in sorted(self.registry.list_detectors()):
            definition = self.registry.get(detector_id)
            if definition is None:
                continue
            decisions.append(self._decide(analysis, detector_id))

        return DetectorSelection(
            media_id=analysis.media.media_id,
            media_type=analysis.media.media_type,
            decisions=decisions,
        )

    def save(self, selection: DetectorSelection, analysis_dir: Path) -> Path:
        analysis_dir.mkdir(parents=True, exist_ok=True)
        path = analysis_dir / "selection.json"
        path.write_text(selection.model_dump_json(indent=2))
        return path

    # ------------------------------------------------------------------ #

    def _decide(self, analysis: MediaAnalysis, detector_id: str) -> SelectionDecision:
        definition = self.registry.get(detector_id)
        caps = definition.capabilities
        media = analysis.media
        media_type = media.media_type.value
        reasons: List[str] = []

        # 0. trained head
        if not definition.weights.is_trained and not self.include_untrained:
            return self._exclude(
                detector_id,
                f"{definition.weights.tier} head: not evidence-grade; "
                f"excluded by default (use --include-untrained to run it anyway)",
            )
        if not definition.weights.is_trained:
            reasons.append(f"{definition.weights.tier} head included on request (not evidence-grade)")

        # 1. media type compatibility
        if not caps.accepts(media_type):
            return self._exclude(
                detector_id,
                f"requires {self._media_label(caps.media_types)}, media is {media_type}",
            )
        reasons.append(f"consumes {self._media_label(caps.media_types)}")

        # 1b. audio detectors need a soundtrack when the media is a video
        if caps.audio_required and media_type == "video" and not media.has_audio_track:
            return self._exclude(detector_id, "requires an audio track; the video has none")
        if caps.audio_required:
            reasons.append("audio track present" if media_type == "video" else "audio media")

        # 2. face presence
        if caps.faces_required and not media.face_present:
            return self._exclude(
                detector_id,
                f"requires faces; none detected in {media.sampled_frames} sampled frames",
            )
        if caps.faces_required:
            reasons.append(
                f"faces present ({media.face_count} in {media.sampled_frames} "
                f"sampled frames, largest {media.max_face_size}px)"
            )

        # 3. face scale
        if caps.faces_required and definition.min_face > 0 and media.max_face_size < definition.min_face:
            return self._exclude(
                detector_id,
                f"faces too small: largest {media.max_face_size}px "
                f"< required min {definition.min_face}px",
            )

        # 4. temporal sufficiency for video-only detectors
        if set(caps.media_types) == {"video"} and not caps.audio_required:
            clip = definition.clip_size or 1
            available = self._available_frames(media)
            if available is not None and available < max(clip, 2):
                return self._exclude(
                    detector_id,
                    f"video too short: ~{available} frames available "
                    f"< clip size {clip}",
                )
            if available is not None:
                reasons.append(f"~{available} frames >= clip size {clip}")

        # 5. weights
        status = self._weight_status(detector_id)
        if not status[0]:
            return self._exclude(detector_id, status[1])
        reasons.append(status[1])

        return SelectionDecision(detector_id=detector_id, included=True, reasons=reasons)

    # ------------------------------------------------------------------ #

    def _weight_status(self, detector_id: str) -> tuple[bool, str]:
        """(eligible, reason) for one detector's weights.

        - checkpoint file configured: local presence required (``official`` /
          ``backbone``); a missing file excludes with the download source.
        - ``external`` or no-file: weights are fetched at first run unless
          offline mode is set.
        - ``random``: self-contained (random init), always eligible.
        """
        definition = self.registry.get(detector_id)
        weights = definition.weights

        if weights.kind == "random":
            return True, "weights: random initialization (no checkpoint)"

        if weights.file:
            if self.registry.weights_available(detector_id):
                return True, f"weights available: {weights.file}"
            host = urlparse(weights.url).netloc if weights.url else ""
            hint = f" (download from {host})" if host else ""
            return False, f"checkpoint missing: weights/{weights.file}{hint}"

        if weights.self_contained or weights.kind == "external":
            if self.offline:
                return False, "offline mode; weights not cached locally"
            return True, "weights: downloaded at first run"

        if self.offline:
            return False, "offline mode; weights not cached locally"
        return True, "weights: downloaded at first run"

    @staticmethod
    def _available_frames(media) -> int | None:
        if media.frame_count:
            return int(media.frame_count)
        if media.duration_seconds and media.fps:
            return int(round(media.duration_seconds * media.fps))
        return media.sampled_frames or None

    @staticmethod
    def _exclude(detector_id: str, reason: str) -> SelectionDecision:
        return SelectionDecision(detector_id=detector_id, included=False, reasons=[reason])

    @staticmethod
    def _media_label(media_types: List[str]) -> str:
        return " or ".join(media_types)
