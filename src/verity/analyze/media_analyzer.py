from __future__ import annotations
import json
import cv2
from pathlib import Path
from typing import Optional

from verity.models.media import MediaType
from verity.analyze.models import MediaAnalysis, MediaCharacteristics

MAX_SAMPLED_FRAMES = 30


class MediaAnalyzer:
    """Deterministically characterize a media file: format, video/audio
    properties (via ffprobe when available) and a face-presence scan using
    OpenCV's Haar cascade on evenly sampled frames.

    All iteration is sorted and sampling is deterministic: the same media
    file always yields the same analysis.
    """

    def __init__(self, media_dir: Path):
        self.media_dir = media_dir
        self._cascade = None
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if hasattr(cv2, "CascadeClassifier"):
            try:
                self._cascade = cv2.CascadeClassifier(str(cascade_path))
            except Exception:
                self._cascade = None

    def analyze(self, media_id: str, media_type: MediaType) -> MediaAnalysis:
        media_path = self._resolve_media(media_id)
        characteristics = self._characterize(media_id, media_path, media_type)
        faces_per_frame, sampled_frames = self._scan_faces(media_path, media_type)
        characteristics.sampled_frames = sampled_frames
        characteristics.face_count = sum(f["face_count"] for f in faces_per_frame)
        characteristics.face_present = bool(faces_per_frame)
        characteristics.max_face_size = max(
            (f["max_face_size"] for f in faces_per_frame), default=0
        )
        return MediaAnalysis(
            media=characteristics,
            faces_per_frame=faces_per_frame,
        )

    def save(self, analysis: MediaAnalysis, analysis_dir: Path) -> Path:
        analysis_dir.mkdir(parents=True, exist_ok=True)
        path = analysis_dir / "analysis.json"
        path.write_text(analysis.model_dump_json(indent=2))
        return path

    # ------------------------------------------------------------------ #

    def _resolve_media(self, media_id: str) -> Path:
        for p in sorted(self.media_dir.iterdir()):
            if p.stem == media_id and p.is_file():
                return p
        raise FileNotFoundError(f"No media for id: {media_id}")

    def _characterize(self, media_id: str, path: Path, media_type: MediaType) -> MediaCharacteristics:
        char = MediaCharacteristics(
            media_id=media_id,
            media_type=media_type,
            format=path.suffix.lstrip(".").lower(),
        )
        probe = self._ffprobe(path)
        if probe is not None:
            stream = self._first_video_stream(probe)
            if stream is not None:
                char.video_codec = stream.get("codec_name", "")
                char.fps = self._parse_fps(stream)
                char.width = stream.get("width")
                char.height = stream.get("height")
                char.frame_count = self._parse_int(stream.get("nb_frames")) or self._estimate_frames(probe)
            fmt = probe.get("format", {})
            char.duration_seconds = self._parse_duration(fmt.get("duration"))
            char.has_audio_track = self._first_audio_stream(probe) is not None
        if char.media_type == MediaType.VIDEO and char.fps and char.duration_seconds:
            char.frame_count = char.frame_count or int(char.fps * char.duration_seconds)
        return char

    @staticmethod
    def _parse_int(value) -> Optional[int]:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    # ----- ffprobe ----- #

    def _ffprobe(self, path: Path) -> Optional[dict]:
        import subprocess
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json",
                 "-show_format", "-show_streams", str(path)],
                capture_output=True, text=True, timeout=60,
            )
            if result.returncode != 0:
                return None
            return json.loads(result.stdout)
        except Exception:
            return None

    @staticmethod
    def _first_video_stream(probe: dict) -> Optional[dict]:
        for s in probe.get("streams", []):
            if s.get("codec_type") == "video":
                return s
        return None

    @staticmethod
    def _first_audio_stream(probe: dict) -> Optional[dict]:
        for s in probe.get("streams", []):
            if s.get("codec_type") == "audio":
                return s
        return None

    @staticmethod
    def _parse_fps(stream: dict) -> Optional[float]:
        r = stream.get("r_frame_rate", "")
        try:
            num, den = r.split("/")
            if float(den) > 0:
                return round(float(num) / float(den), 3)
        except (ValueError, ZeroDivisionError):
            pass
        return None

    @staticmethod
    def _parse_duration(value) -> Optional[float]:
        try:
            return round(float(value), 3)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _estimate_frames(probe: dict) -> Optional[int]:
        fmt = probe.get("format", {})
        try:
            duration = float(fmt.get("duration", 0))
        except (TypeError, ValueError):
            return None
        if duration <= 0:
            return None
        for s in probe.get("streams", []):
            if s.get("codec_type") == "video":
                fps = MediaAnalyzer._parse_fps(s)
                if fps:
                    return int(round(duration * fps))
        return None

    # ----- face scan ----- #

    def _scan_faces(self, path: Path, media_type: MediaType) -> tuple[list[dict], int]:
        if self._cascade is None:
            return [], 0
        if media_type == MediaType.VIDEO:
            frames = self._sample_video_frames(path)
        elif media_type == MediaType.IMAGE:
            frames = [(0, cv2.imread(str(path)))]
        else:
            return [], 0

        faces_per_frame: list[dict] = []
        for frame_idx, frame in frames:
            if frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            boxes = self._cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
            )
            if len(boxes):
                largest = max(max(w, h) for (_, _, w, h) in boxes)
                faces_per_frame.append({
                    "frame": frame_idx,
                    "face_count": int(len(boxes)),
                    "max_face_size": int(largest),
                })
        return faces_per_frame, len(frames)

    def _sample_video_frames(self, path: Path) -> list[tuple[int, object]]:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        interval = max(1, total // MAX_SAMPLED_FRAMES) if total else 1

        frames: list[tuple[int, object]] = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % interval == 0:
                frames.append((idx, frame))
            idx += 1
        cap.release()
        return frames
