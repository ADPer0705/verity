"""Frame-level temporal structure of a video.

Rationale: edits that operate on individual frames leave traces in how
consecutive frames relate to each other. We decode the stream once at
reduced resolution and record, per frame, the mean absolute difference to
the previous frame (difference energy), a perceptual hash and the mean
luminance. From these:

* **Scene-cut candidates** are frames whose difference energy is an outlier
  (robust z-score). Cuts are normal in edited footage and absent in a
  single continuous take; their positions are compared with the keyframe
  positions from the metadata analysis.
* **Frozen or duplicated runs** are consecutive frames with identical
  perceptual hashes and near-zero difference: frame-rate conversion, dropped
  frames, or a still frame held to cover an edit.
* **Flicker** is the mean second difference of the luminance series: a
  frame-to-frame alternation that face-swap and frame-wise generators
  produce and cameras do not.
* **Face-size jitter** uses the crops the face extractor saved: the extractor
  stores crop images, not box coordinates, so only the size series is
  available and positional jitter is not computed.

Limits: a hand-held camera or a panning shot raises the difference energy
everywhere, so thresholds are relative to the clip itself; a video with one
cut is simply an edited video, which says nothing about the content.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from verity.forensics.base import ForensicAnalyzer, r, write_json
from verity.forensics.models import ForensicResult, Observation

#: Frames are analysed at this width (aspect preserved) for speed.
WORK_WIDTH = 320
#: Upper bound on decoded frames; longer clips are strided.
MAX_FRAMES = 1500
#: Robust z-score of difference energy that marks a scene-cut candidate.
CUT_Z = 6.0
CUT_MIN_ENERGY = 12.0
#: Difference energy below which two hash-identical frames count as frozen.
FROZEN_MAX_ENERGY = 0.1
#: ... and well below the clip's own median difference.
FROZEN_MEDIAN_FRACTION = 0.25
FLICKER_THRESHOLD = 1.5


def phash(gray: np.ndarray) -> int:
    """64-bit perceptual hash: low-frequency DCT of a 32x32 thumbnail."""
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)[:8, :8]
    med = np.median(dct[1:, 1:])
    bits = (dct > med).astype(np.uint8).ravel()
    return int("".join("1" if b else "0" for b in bits), 2)


class VideoTemporalAnalyzer(ForensicAnalyzer):
    analyzer_id = "video_temporal"
    label = "Temporal structure"
    media_types = ("video",)
    method = (
        "Frames are decoded sequentially at reduced size; for each one the mean "
        "absolute difference to the previous frame, a perceptual hash and the mean "
        "luminance are recorded. Outlying differences are scene-cut candidates, "
        "runs of identical hashes are frozen or duplicated frames, and the mean "
        "second difference of luminance is a flicker measure. Camera motion and "
        "lighting changes raise difference energy naturally, so thresholds are "
        "relative to the clip itself; the analysis shows where a clip changes "
        "abruptly, not why."
    )

    def analyze(self, media_id: str, media_type: str) -> ForensicResult:
        path = self.media_path(media_id)
        out_dir = self.output_dir(media_id)
        series = self._scan(path)
        if series["frames_processed"] < 2:
            raise RuntimeError("fewer than two decodable frames")
        observations: List[Observation] = []
        data: Dict[str, Any] = dict(series)

        energy = np.array(series["diff_energy"], dtype=np.float64)
        lum = np.array(series["mean_luminance"], dtype=np.float64)
        idx = np.array(series["frame_index"])
        fps = series["fps"] or 25.0

        cuts = self._cuts(energy, idx, fps)
        data["scene_cuts"] = cuts
        frozen = self._frozen_runs(series["phash"], energy, idx, fps)
        data["frozen_runs"] = frozen
        flicker = float(np.mean(np.abs(lum[1:-1] - (lum[:-2] + lum[2:]) / 2.0))) if len(lum) >= 3 else 0.0
        data["flicker"] = {"mean_second_difference": r(flicker, 4), "threshold": FLICKER_THRESHOLD,
                           "luminance_std": r(lum.std(), 4)}
        faces = self._face_sizes(media_id)
        data["faces"] = faces
        data["frame_index"] = [int(i) for i in idx]
        data["diff_energy"] = [r(e, 3) for e in energy]
        data["mean_luminance"] = [r(v, 3) for v in lum]
        data["phash"] = [f"{h:016x}" for h in series["phash"]]

        observations.extend(self._observations(data, cuts, frozen, flicker, faces))
        timeline = self._plot(out_dir / "timeline.png", data, cuts, frozen, faces)
        artifacts = [timeline, write_json(out_dir / "temporal.json", data)]
        return self.result(observations, artifacts, data={
            "frames_processed": series["frames_processed"],
            "scene_cuts": len(cuts),
            "frozen_runs": len(frozen),
            "flicker": r(flicker, 4),
        })

    # ------------------------------------------------------------------ #

    def _scan(self, path: Path) -> Dict[str, Any]:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError("OpenCV cannot open the video")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        stride = max(1, int(np.ceil(total / MAX_FRAMES))) if total > 0 else 1
        prev = None
        frame_index: List[int] = []
        energy: List[float] = []
        hashes: List[int] = []
        lum: List[float] = []
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % stride == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                h, w = gray.shape
                if w > WORK_WIDTH:
                    gray = cv2.resize(gray, (WORK_WIDTH, max(1, int(round(h * WORK_WIDTH / w)))),
                                      interpolation=cv2.INTER_AREA)
                g = gray.astype(np.float32)
                energy.append(float(np.abs(g - prev).mean()) if prev is not None else 0.0)
                hashes.append(phash(gray))
                lum.append(float(g.mean()))
                frame_index.append(i)
                prev = g
            i += 1
        cap.release()
        return {
            "fps": r(fps, 3),
            "frames_total": total,
            "frames_processed": len(frame_index),
            "stride": stride,
            "work_width": WORK_WIDTH,
            "frame_index": frame_index,
            "diff_energy": energy,
            "phash": hashes,
            "mean_luminance": lum,
        }

    @staticmethod
    def _cuts(energy: np.ndarray, idx: np.ndarray, fps: float) -> List[Dict[str, Any]]:
        if len(energy) < 3:
            return []
        body = energy[1:]
        med = float(np.median(body))
        mad = float(1.4826 * np.median(np.abs(body - med))) + 1e-6
        cuts = []
        for k in range(1, len(energy)):
            z = (energy[k] - med) / mad
            if z > CUT_Z and energy[k] > CUT_MIN_ENERGY:
                cuts.append({"frame": int(idx[k]), "time": r(idx[k] / fps, 3),
                             "energy": r(energy[k], 3), "z": r(z, 2)})
        return cuts

    @staticmethod
    def _frozen_runs(hashes: List[int], energy: np.ndarray, idx: np.ndarray, fps: float) -> List[Dict[str, Any]]:
        runs = []
        start = 0
        limit = min(FROZEN_MAX_ENERGY, FROZEN_MEDIAN_FRACTION * float(np.median(energy[1:]))) if len(energy) > 1 else FROZEN_MAX_ENERGY
        for k in range(1, len(hashes) + 1):
            same = k < len(hashes) and hashes[k] == hashes[k - 1] and energy[k] <= limit
            if not same:
                if k - start >= 2:
                    runs.append({"start_frame": int(idx[start]), "end_frame": int(idx[k - 1]),
                                 "frames": int(k - start), "start_time": r(idx[start] / fps, 3)})
                start = k
        return runs

    def _face_sizes(self, media_id: str) -> Dict[str, Any]:
        faces_dir = self.artifact_dir / media_id / "faces"
        if not faces_dir.exists():
            return {"available": False, "reason": "no face crops extracted"}
        per_frame: Dict[int, int] = {}
        pattern = re.compile(r"frame_(\d+)_face_\d+\.png$")
        for p in sorted(faces_dir.glob("frame_*_face_*.png")):
            m = pattern.search(p.name)
            if not m:
                continue
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            key = int(m.group(1))
            per_frame[key] = max(per_frame.get(key, 0), int(max(img.shape[:2])))
        if len(per_frame) < 3:
            return {"available": False, "reason": f"only {len(per_frame)} frame(s) with face crops"}
        keys = sorted(per_frame)
        sizes = np.array([per_frame[k] for k in keys], dtype=np.float64)
        rel = np.abs(np.diff(sizes)) / np.maximum(sizes[:-1], 1.0)
        return {
            "available": True,
            "note": "the face extractor stores crops, not box coordinates: size series only",
            "frames_with_faces": len(keys),
            "frame_keys": keys,
            "largest_face_size": [int(s) for s in sizes],
            "size_jitter_median": r(np.median(rel), 4),
            "size_jitter_max": r(rel.max(), 4),
            "size_jumps_over_30pct": int((rel > 0.3).sum()),
        }

    def _observations(self, data, cuts, frozen, flicker, faces) -> List[Observation]:
        obs: List[Observation] = []
        n = data["frames_processed"]
        obs.append(self.observation(
            "info",
            f"{n} frame(s) analysed (stride {data['stride']} of {data['frames_total']} declared, "
            f"{data['fps']} fps); median frame-to-frame difference "
            f"{np.median(data['diff_energy'][1:]) if n > 1 else 0:.2f} grey levels.",
            frames_processed=n, stride=data["stride"], fps=data["fps"]))
        if cuts:
            times = ", ".join(f"{c['time']:.2f}s" for c in cuts[:8])
            obs.append(self.observation(
                "notice",
                f"{len(cuts)} abrupt frame change(s) at {times}{', ...' if len(cuts) > 8 else ''}: "
                "scene cuts or splice points; a continuous recording has none.",
                scene_cuts=len(cuts), first_cuts=cuts[:8]))
        else:
            obs.append(self.observation("info", "No abrupt frame changes: the clip reads as one continuous shot.",
                                        scene_cuts=0))
        if frozen:
            longest = max(frozen, key=lambda x: (x["frames"], -x["start_frame"]))
            total_frozen = sum(x["frames"] for x in frozen)
            obs.append(self.observation(
                "notice",
                f"{len(frozen)} run(s) of identical consecutive frames ({total_frozen} frames in total, "
                f"longest {longest['frames']} frames at {longest['start_time']:.2f}s): frame-rate "
                "conversion, dropped frames, or a held frame covering an edit.",
                frozen_runs=len(frozen), frozen_frames=total_frozen, longest=longest))
        if flicker > FLICKER_THRESHOLD:
            obs.append(self.observation(
                "notice",
                f"Luminance flickers between consecutive frames (mean second difference "
                f"{flicker:.2f} grey levels, threshold {FLICKER_THRESHOLD}); frame-wise generation "
                "and face replacement produce this, as do fluorescent lighting and auto-exposure.",
                flicker=r(flicker, 4)))
        if faces.get("available"):
            jumps = faces["size_jumps_over_30pct"]
            obs.append(self.observation(
                "notice" if jumps >= 2 else "info",
                f"Largest detected face size changes by a median {faces['size_jitter_median']:.0%} "
                f"between successive extracted frames ({jumps} jump(s) above 30%).",
                size_jitter_median=faces["size_jitter_median"], size_jumps_over_30pct=jumps,
                frames_with_faces=faces["frames_with_faces"]))
        else:
            obs.append(self.observation(
                "info", f"Face jitter not computed: {faces.get('reason')}.", reason=faces.get("reason")))
        return obs

    def _plot(self, path: Path, data, cuts, frozen, faces) -> Path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fps = data["fps"] or 25.0
        t = np.array(data["frame_index"]) / fps
        rows = 3 if faces.get("available") else 2
        fig, axes = plt.subplots(rows, 1, figsize=(10, 2.2 * rows), dpi=100)
        axes = np.atleast_1d(axes)
        ax = axes[0]
        ax.plot(t, data["diff_energy"], color="#2a6f97", linewidth=0.9)
        for c in cuts:
            ax.axvline(c["time"], color="#d1495b", linewidth=1.0, alpha=0.9)
        for run in frozen:
            ax.axvspan(run["start_time"], run["end_frame"] / fps, color="#f4a261", alpha=0.35, linewidth=0)
        ax.set_ylabel("frame diff")
        ax.set_title("Frame difference energy (red: cut candidates, orange: frozen runs)", fontsize=9)
        ax = axes[1]
        ax.plot(t, data["mean_luminance"], color="#386641", linewidth=0.9)
        ax.set_ylabel("mean luma")
        ax.set_xlabel("time (s)")
        ax.set_title(f"Mean luminance (flicker {data['flicker']['mean_second_difference']})", fontsize=9)
        if faces.get("available"):
            ax = axes[2]
            keys = faces["frame_keys"]
            ax.plot(range(len(keys)), faces["largest_face_size"], color="#6a4c93", marker=".", linewidth=0.9)
            ax.set_ylabel("face px")
            ax.set_xlabel("extracted frame #")
            ax.set_title("Largest face crop size per extracted frame", fontsize=9)
        for a in axes:
            a.grid(True, alpha=0.3)
            a.tick_params(labelsize=8)
        fig.tight_layout()
        fig.savefig(path, metadata={"Software": None})
        plt.close(fig)
        return path
