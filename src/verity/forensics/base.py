"""Shared plumbing for forensic analyzers: media lookup, output directories,
deterministic frame sampling and small image helpers."""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from verity.forensics.models import ForensicResult, Observation

#: Longest image side the pixel-level analyses work on. Larger inputs are
#: downscaled (area interpolation) so run time stays bounded; the JSON records
#: the scale that was applied.
MAX_SIDE = 2048


def resolve_media(workspace: Path, media_id: str) -> Path:
    """The read-only working copy the ingestor made for ``media_id``."""
    media_dir = workspace / "media"
    if media_dir.exists():
        for p in sorted(media_dir.iterdir()):
            if p.is_file() and p.stem == media_id:
                return p
    raise FileNotFoundError(f"No media for id: {media_id}")


def sample_video_frames(path: Path, count: int, first: bool = False) -> List[Tuple[int, np.ndarray]]:
    """Decode ``count`` frames (BGR) from a video, deterministically.

    ``first=True`` takes the first ``count`` decoded frames; otherwise the
    frames are spread evenly over the stream (index 0, stride, 2*stride ...).
    Frames are read sequentially rather than by seeking, because seeking lands
    on different frames depending on the decoder's keyframe handling.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    stride = 1 if first or total <= 0 else max(1, total // count)
    frames: List[Tuple[int, np.ndarray]] = []
    idx = 0
    while len(frames) < count:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            frames.append((idx, frame))
        idx += 1
    cap.release()
    return frames


def to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def bounded(image: np.ndarray, max_side: int = MAX_SIDE) -> Tuple[np.ndarray, float]:
    """Downscale so that the longest side is at most ``max_side``.

    Returns the (possibly unchanged) image and the scale factor applied."""
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image, 1.0
    scale = max_side / longest
    resized = cv2.resize(image, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                         interpolation=cv2.INTER_AREA)
    return resized, scale


def normalize_u8(arr: np.ndarray, lo: Optional[float] = None, hi: Optional[float] = None) -> np.ndarray:
    """Map a float array to 0..255 for display (robust percentiles by default)."""
    arr = np.asarray(arr, dtype=np.float64)
    if lo is None:
        lo = float(np.percentile(arr, 1))
    if hi is None:
        hi = float(np.percentile(arr, 99))
    if hi <= lo:
        hi = lo + 1e-9
    return np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def write_png(path: Path, image: np.ndarray) -> Path:
    """cv2.imwrite with no ancillary chunks: byte-identical across runs."""
    cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    return path


def write_json(path: Path, payload: Dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))
    return path


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def r(value, digits: int = 4):
    """Round for JSON so tiny floating-point noise never changes an output."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    value = float(value)
    if not np.isfinite(value):
        return None
    return round(value, digits)


class ForensicAnalyzer(ABC):
    """One model-free technique. Subclasses implement :meth:`analyze` and set
    ``analyzer_id``, ``label`` and ``method`` (a plain-language note on what
    the technique can and cannot show, rendered verbatim in reports)."""

    analyzer_id: str = ""
    label: str = ""
    method: str = ""
    #: Media types the analyzer applies to.
    media_types: Tuple[str, ...] = ("image", "video")

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace)
        self.artifact_dir = self.workspace / "artifacts"

    def output_dir(self, media_id: str) -> Path:
        out = self.artifact_dir / media_id / "forensics" / self.analyzer_id
        out.mkdir(parents=True, exist_ok=True)
        return out

    def media_path(self, media_id: str) -> Path:
        return resolve_media(self.workspace, media_id)

    def observation(self, severity: str, statement: str, **evidence) -> Observation:
        return Observation(analyzer=self.analyzer_id, severity=severity,
                           statement=statement, evidence=evidence)

    def result(self, observations: List[Observation], artifacts: List[Path],
               data: Optional[Dict[str, Any]] = None, ok: bool = True) -> ForensicResult:
        return ForensicResult(
            analyzer=self.analyzer_id, label=self.label, method=self.method, ok=ok,
            observations=list(observations),
            artifacts=[str(p) for p in artifacts],
            data=data or {},
        )

    @abstractmethod
    def analyze(self, media_id: str, media_type: str) -> ForensicResult: ...
