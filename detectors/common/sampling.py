from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class VideoReader:
    """Decode a video once into a list of (sampled) frames.

    Video detectors need ``clip_size`` consecutive frames per forward pass;
    keeping the sampled frames in memory (a few dozen for real-world clips) is
    far cheaper than re-reading the file per clip.
    """

    def __init__(self, path: Path, sample_interval: int = 1):
        self.path = path
        self.sample_interval = sample_interval if sample_interval and sample_interval > 0 else 1
        self.frames: list[np.ndarray] = []
        self.frame_indices: list[int] = []
        self.fps, self.total_frames = self._read()

    def _read(self) -> tuple[float, int]:
        cap = cv2.VideoCapture(str(self.path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % self.sample_interval == 0:
                self.frames.append(frame)
                self.frame_indices.append(idx)
            idx += 1
        cap.release()
        if fps <= 0:
            fps = 30.0
        return fps, total

    def clip(self, center_pos: int, clip_size: int) -> list[np.ndarray]:
        """Return ``clip_size`` sampled frames centered on ``center_pos``
        (in sampled-frame index space), clamped to the video bounds."""
        n = len(self.frames)
        half = clip_size // 2
        start = max(0, center_pos - half)
        end = min(n, start + clip_size)
        start = max(0, end - clip_size)
        seg = self.frames[start:end]
        while len(seg) < clip_size:
            seg.append(self.frames[min(len(self.frames) - 1, start)])
        return seg


def crop_rgb(frames: list[np.ndarray], bbox: tuple[int, int, int, int],
             resolution: int) -> list[np.ndarray]:
    """Crop ``bbox`` from each (BGR) frame, resize to ``resolution``, and
    return a list of float32 RGB arrays normalized to [0, 1]."""
    x, y, w, h = bbox
    h_im, w_im = frames[0].shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w_im, x + w), min(h_im, y + h)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("degenerate face bbox")
    out = []
    for fr in frames:
        crop = fr[y1:y2, x1:x2]
        crop = cv2.resize(crop, (resolution, resolution), interpolation=cv2.INTER_LINEAR)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        out.append(crop.astype("float32") / 255.0)
    return out


def build_image_tensor(rgb_float01, mean, std) -> np.ndarray:
    """Apply ``(x - mean) / std`` to a (H, W, 3) RGB array in [0, 1]."""
    arr = np.asarray(rgb_float01, dtype="float32")
    arr = (arr - np.asarray(mean, dtype="float32")) / np.asarray(std, dtype="float32")
    return arr.transpose(2, 0, 1)
