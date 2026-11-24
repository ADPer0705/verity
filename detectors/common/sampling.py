from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np


class VideoReader:
    """Decode a video once and keep only the frames a detector will score.

    Memory is bounded: at most ``max_frames`` evenly spaced sample positions
    are kept (plus, in clip mode, the ``clip_size`` consecutive frames around
    each of them), instead of every frame of the file.  A 1080p minute of
    video is ~11 GB decoded, which is what used to exhaust laptop RAM.
    ``VERITY_MAX_FRAMES`` overrides the budget for a run (``0`` = unlimited).
    """

    def __init__(self, path: Path, sample_interval: int = 1, max_frames: int | None = 32,
                 clip_size: int | None = None):
        self.path = path
        self.sample_interval = sample_interval if sample_interval and sample_interval > 0 else 1
        env = os.environ.get("VERITY_MAX_FRAMES")
        if env is not None and env.strip() != "":
            max_frames = int(env)
        self.max_frames = max_frames if max_frames and max_frames > 0 else None
        self.clip_size = clip_size if clip_size and clip_size > 1 else None
        self.frames: list[np.ndarray] = []          # the sampled frames that are scored
        self.frame_indices: list[int] = []          # their absolute frame indices
        self._windows: dict[int, list[np.ndarray]] = {}   # clip mode: consecutive frames per position
        self.fps, self.total_frames = self._read()

    def _plan(self, total: int) -> tuple[list[int], set[int]]:
        """Absolute indices of the frames to score and the set of frames to keep."""
        sampled = list(range(0, total, self.sample_interval))
        if self.max_frames is not None and len(sampled) > self.max_frames:
            k = self.max_frames
            sampled = sorted({sampled[int(round(i * (len(sampled) - 1) / (k - 1)))] for i in range(k)}) \
                if k > 1 else [sampled[len(sampled) // 2]]
        keep = set(sampled)
        if self.clip_size:
            half = self.clip_size // 2
            for idx in sampled:
                start = max(0, idx - half)
                end = min(total, start + self.clip_size)
                start = max(0, end - self.clip_size)
                keep.update(range(start, end))
        return sampled, keep

    def _read(self) -> tuple[float, int]:
        cap = cv2.VideoCapture(str(self.path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0:
            fps = 30.0
        sampled, keep = self._plan(total) if total > 0 else ([], set())
        unbounded = total <= 0          # unknown length: fall back to interval sampling
        kept: dict[int, np.ndarray] = {}
        idx = 0
        last = max(keep) if keep else -1
        while True:
            if not unbounded and idx > last:
                break
            ret, frame = cap.read()
            if not ret:
                break
            if unbounded:
                if idx % self.sample_interval == 0:
                    self.frames.append(frame)
                    self.frame_indices.append(idx)
            elif idx in keep:
                kept[idx] = frame
            idx += 1
        cap.release()
        if unbounded:
            return fps, idx
        actual_total = total
        for pos in sampled:
            if pos not in kept:
                continue      # the container over-reported its length
            self.frames.append(kept[pos])
            self.frame_indices.append(pos)
            if self.clip_size:
                half = self.clip_size // 2
                start = max(0, pos - half)
                end = min(total, start + self.clip_size)
                start = max(0, end - self.clip_size)
                window = [kept[i] for i in range(start, end) if i in kept]
                while window and len(window) < self.clip_size:
                    window.append(window[-1])
                self._windows[pos] = window
        return fps, actual_total

    def clip(self, center_pos: int, clip_size: int) -> list[np.ndarray]:
        """``clip_size`` consecutive frames around sampled position ``center_pos``
        (index into ``self.frames``), clamped to the video bounds."""
        if self.clip_size and self._windows:
            window = self._windows.get(self.frame_indices[center_pos])
            if window:
                return list(window[:clip_size]) if len(window) >= clip_size else \
                    list(window) + [window[-1]] * (clip_size - len(window))
        n = len(self.frames)
        half = clip_size // 2
        start = max(0, center_pos - half)
        end = min(n, start + clip_size)
        start = max(0, end - clip_size)
        seg = list(self.frames[start:end])
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
