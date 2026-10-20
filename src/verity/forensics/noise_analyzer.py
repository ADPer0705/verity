"""Sensor-noise consistency and frequency-domain periodicity.

Rationale: a camera sensor adds noise whose level is roughly uniform across
a frame (modulated by brightness and ISO). Content pasted from another
picture, smoothed by a generator or re-rendered by a face-swap model carries
a different noise level, so the *local* noise estimate becomes inconsistent.
We take the residual (image minus a median-filtered copy), estimate the
noise level per grid cell as the RMS over non-edge pixels (edges would
otherwise be mistaken for noise) and flag cells whose level is far from the
picture's typical level.

The log-magnitude 2-D spectrum is a second, independent view: natural
images have a smooth, radially decaying spectrum. Upsampling, GAN decoders
and tiled textures introduce regular grids of peaks (Zhang, Karaman & Chang,
2019; Durall et al., 2020). JPEG and video codecs also add peaks at
multiples of 1/8 cycles per pixel, so a peak alone is not a verdict.

Limits: textured regions (hair, foliage) have a higher residual than flat
ones regardless of origin, strong denoising on phones flattens the noise
everywhere, and recompression masks all of it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from verity.forensics.base import (ForensicAnalyzer, bounded, normalize_u8, r, sample_video_frames,
                                   to_gray, write_json, write_png)
from verity.forensics.models import ForensicResult, Observation

GRID_CELL = 48
SPECTRUM_SIDE = 1024
SPECTRUM_FRAMES = 5
#: Robust z-score beyond which a cell's noise level is called inconsistent.
Z_THRESHOLD = 3.5
#: Sobel magnitude above which a pixel is an edge and excluded from the noise estimate.
EDGE_THRESHOLD = 40.0
#: Prominence (log units above the local background) for a spectral peak.
PEAK_PROMINENCE = 1.6
PEAK_MIN_RADIUS = 10


class NoiseAnalyzer(ForensicAnalyzer):
    analyzer_id = "noise"
    label = "Noise and spectrum"
    media_types = ("image", "video")
    method = (
        "The noise residual (picture minus a 3x3 median-filtered copy) is measured "
        "per 48px cell as the RMS over non-edge pixels, so that edges are not "
        "mistaken for noise; cells whose level is far from the picture's typical level are "
        "outlined. A uniformly captured photograph has a uniform level, while "
        "pasted, generated or denoised content differs. The log-magnitude Fourier "
        "spectrum (windowed, centred) is scanned for isolated peaks away from the "
        "centre: natural images decay smoothly, whereas upsampling and generative "
        "decoders leave regular grids of peaks. Texture, lighting, phone denoising "
        "and codec block grids all influence both maps, so they are read together "
        "with the picture, not as scores."
    )

    def analyze(self, media_id: str, media_type: str) -> ForensicResult:
        path = self.media_path(media_id)
        out_dir = self.output_dir(media_id)
        frames = self._load_frames(path, media_type)
        if not frames:
            raise RuntimeError("no decodable image data")
        observations: List[Observation] = []
        artifacts: List[Path] = []
        data: Dict[str, Any] = {"source": media_type, "frame_index": frames[0][0]}

        first, scale = bounded(frames[0][1])
        gray = to_gray(first).astype(np.float64)
        data["analysis_scale"] = r(scale, 4)

        residual = gray - cv2.medianBlur(to_gray(first), 3).astype(np.float64)
        residual_vis = np.clip(residual * 6.0 + 128.0, 0, 255).astype(np.uint8)
        artifacts.append(write_png(out_dir / "noise_residual.png", residual_vis))

        variance_vis, noise_stats = self._noise_map(residual, gray)
        artifacts.append(write_png(out_dir / "noise_variance.png", variance_vis))
        data["noise"] = noise_stats
        observations.extend(self._noise_observations(noise_stats))

        spectrum_vis, spec_stats = self._spectrum([to_gray(f) for _, f in frames[:SPECTRUM_FRAMES]])
        artifacts.append(write_png(out_dir / "spectrum.png", spectrum_vis))
        data["spectrum"] = spec_stats
        observations.extend(self._spectrum_observations(spec_stats, media_type))

        artifacts.append(write_json(out_dir / "noise.json", data))
        return self.result(observations, artifacts, data={
            "noise_sigma_median": noise_stats["sigma_median"],
            "inconsistent_cell_fraction": noise_stats["inconsistent_cell_fraction"],
            "spectral_peaks": spec_stats["peak_count"],
        })

    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_frames(path: Path, media_type: str) -> List[Tuple[int, np.ndarray]]:
        if media_type == "video":
            return sample_video_frames(path, SPECTRUM_FRAMES)
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            from PIL import Image
            with Image.open(path) as pil:
                img = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
        return [(0, img)]

    def _noise_map(self, residual: np.ndarray, gray: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Per-cell noise level: RMS of the residual over non-edge pixels.

        Edges (Sobel magnitude of a 5x5-smoothed copy above ``EDGE_THRESHOLD``)
        are masked because a median filter treats them as noise; a cell needs
        30% non-edge, non-saturated pixels to count. Levels are compared on a
        log scale with a robust z-score against the picture's median cell."""
        h, w = residual.shape
        cell = GRID_CELL
        rows, cols = max(1, h // cell), max(1, w // cell)
        gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
        blur = cv2.GaussianBlur(gray_u8, (5, 5), 0)
        sx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
        flat = (np.hypot(sx, sy) <= EDGE_THRESHOLD) & (gray > 8) & (gray < 247)

        def cells(arr: np.ndarray) -> np.ndarray:
            return arr[:rows * cell, :cols * cell].reshape(rows, cell, cols, cell).transpose(0, 2, 1, 3).reshape(rows, cols, -1)

        res_c, flat_c = cells(residual), cells(flat)
        flat_fraction = flat_c.mean(axis=2)
        masked_sq = np.where(flat_c, res_c ** 2, 0.0).sum(axis=2)
        counts = np.maximum(flat_c.sum(axis=2), 1)
        sigma = np.sqrt(masked_sq / counts)
        valid = flat_fraction >= 0.3
        n_valid = int(valid.sum())
        log_sigma = np.log(sigma + 0.05)
        ref = log_sigma[valid] if n_valid >= 4 else log_sigma.ravel()
        centre = float(np.median(ref))
        spread = float(1.4826 * np.median(np.abs(ref - centre))) + 1e-3
        z = (log_sigma - centre) / spread
        inconsistent = valid & (np.abs(z) > Z_THRESHOLD)
        frac = float(inconsistent.sum() / n_valid) if n_valid else 0.0

        vis = normalize_u8(np.where(valid, sigma, 0.0), lo=0.0, hi=max(1.0, float(np.percentile(sigma[valid], 99)) if n_valid else 1.0))
        vis = cv2.resize(vis, (cols * cell, rows * cell), interpolation=cv2.INTER_NEAREST)
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_MAGMA)
        invalid = cv2.resize((~valid).astype(np.uint8), (cols * cell, rows * cell),
                             interpolation=cv2.INTER_NEAREST).astype(bool)
        vis[invalid] = (70, 70, 70)  # no flat, unsaturated pixels: no estimate
        for i, j in zip(*np.nonzero(inconsistent)):
            colour = (255, 255, 0) if z[i, j] < 0 else (0, 255, 255)
            cv2.rectangle(vis, (int(j * cell), int(i * cell)), (int((j + 1) * cell - 1), int((i + 1) * cell - 1)),
                          colour, 2)
        low = [[int(i), int(j)] for i, j in zip(*np.nonzero(inconsistent & (z < 0)))]
        high = [[int(i), int(j)] for i, j in zip(*np.nonzero(inconsistent & (z > 0)))]
        stats = {
            "cell_size": cell,
            "cell_grid": [int(rows), int(cols)],
            "edge_threshold": EDGE_THRESHOLD,
            "sigma_median": r(np.exp(centre) - 0.05, 3),
            "sigma_min": r(sigma[valid].min() if n_valid else sigma.min(), 3),
            "sigma_max": r(sigma[valid].max() if n_valid else sigma.max(), 3),
            "z_threshold": Z_THRESHOLD,
            "valid_cells": n_valid,
            "inconsistent_cells": int(inconsistent.sum()),
            "inconsistent_cell_fraction": r(frac, 4),
            "max_abs_z": r(np.abs(z[valid]).max() if n_valid else 0.0, 3),
            "low_noise_cells": low[:64],
            "high_noise_cells": high[:64],
        }
        return vis, stats

    def _noise_observations(self, stats: Dict[str, Any]) -> List[Observation]:
        obs: List[Observation] = []
        frac = stats["inconsistent_cell_fraction"] or 0.0
        n = stats["inconsistent_cells"]
        if n >= 2 and frac >= 0.03:
            low, high = len(stats["low_noise_cells"]), len(stats["high_noise_cells"])
            obs.append(self.observation(
                "notice",
                f"Noise level is inconsistent in {n} of {stats['valid_cells']} cells ({frac:.0%}): "
                f"{low} unusually smooth, {high} unusually noisy (robust z beyond {Z_THRESHOLD}); "
                "smooth islands are what generated or heavily retouched regions look like, noisy "
                "islands what fine texture or a pasted noisier source looks like.",
                inconsistent_cells=n, valid_cells=stats["valid_cells"], fraction=frac,
                max_abs_z=stats["max_abs_z"], sigma_median=stats["sigma_median"]))
        else:
            obs.append(self.observation(
                "info",
                f"Noise level is spatially consistent (median sigma {stats['sigma_median']} grey "
                f"levels, {n} outlying cell(s) of {stats['valid_cells']}).",
                sigma_median=stats["sigma_median"], inconsistent_cells=n))
        if stats["sigma_median"] is not None and stats["sigma_median"] < 0.6:
            obs.append(self.observation(
                "notice",
                f"Very low overall noise (median sigma {stats['sigma_median']}): the picture was "
                "strongly denoised, rendered, or downscaled, so noise-based checks have little "
                "signal to work with.",
                sigma_median=stats["sigma_median"]))
        return obs

    def _spectrum(self, grays: List[np.ndarray]) -> Tuple[np.ndarray, Dict[str, Any]]:
        acc = None
        used = 0
        for g in grays:
            crop = self._centre_crop(g, SPECTRUM_SIDE).astype(np.float64)
            crop -= crop.mean()
            wy = np.hanning(crop.shape[0])[:, None]
            wx = np.hanning(crop.shape[1])[None, :]
            f = np.fft.fftshift(np.fft.fft2(crop * wy * wx))
            mag = np.log1p(np.abs(f))
            acc = mag if acc is None else acc + mag
            used += 1
        spec = acc / max(1, used)
        h, w = spec.shape
        cy, cx = h // 2, w // 2
        background = cv2.blur(spec, (21, 21))
        prominence = spec - background
        yy, xx = np.mgrid[0:h, 0:w]
        radius = np.hypot(yy - cy, xx - cx)
        mask = radius >= PEAK_MIN_RADIUS
        # half-plane: the spectrum of a real image is point-symmetric
        mask &= (yy < cy) | ((yy == cy) & (xx > cx))
        local_max = cv2.dilate(prominence, np.ones((7, 7), np.uint8)) == prominence
        peaks = np.argwhere(local_max & mask & (prominence > PEAK_PROMINENCE))
        found = []
        for y, x in peaks:
            found.append({
                "fx": r((x - cx) / w, 4), "fy": r((y - cy) / h, 4),
                "prominence": r(prominence[y, x], 3),
                "radius": r(radius[y, x] / (min(h, w) / 2), 4),
            })
        found.sort(key=lambda p: (-p["prominence"], p["fx"], p["fy"]))
        # peaks exactly on the k/8 block-grid frequencies are codec artefacts
        grid_peaks = sum(1 for p in found if self._near_grid(p["fx"]) and self._near_grid(p["fy"]))
        vis = normalize_u8(spec, lo=float(np.percentile(spec, 1)), hi=float(np.percentile(spec, 99.9)))
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
        for p in found[:40]:
            x = int(round(p["fx"] * w + cx))
            y = int(round(p["fy"] * h + cy))
            cv2.circle(vis, (x, y), 9, (255, 255, 255), 1)
            cv2.circle(vis, (w - x, h - y), 9, (255, 255, 255), 1)
        # radial profile: energy by normalised frequency, summarises the fall-off
        bins = np.minimum((radius / (min(h, w) / 2) * 20).astype(int), 20)
        profile = [r(spec[bins == b].mean(), 4) if (bins == b).any() else None for b in range(20)]
        stats = {
            "frames_averaged": used,
            "crop": [int(h), int(w)],
            "peak_prominence_threshold": PEAK_PROMINENCE,
            "peak_count": len(found),
            "grid_aligned_peaks": grid_peaks,
            "peaks": found[:40],
            "radial_profile": profile,
        }
        return vis, stats

    @staticmethod
    def _near_grid(f: float, tol: float = 0.006) -> bool:
        k = round(f * 8)
        return abs(f - k / 8) < tol

    @staticmethod
    def _centre_crop(gray: np.ndarray, side: int) -> np.ndarray:
        h, w = gray.shape
        ch, cw = min(h, side), min(w, side)
        y0, x0 = (h - ch) // 2, (w - cw) // 2
        return gray[y0:y0 + ch, x0:x0 + cw]

    def _spectrum_observations(self, stats: Dict[str, Any], media_type: str) -> List[Observation]:
        obs: List[Observation] = []
        n, grid = stats["peak_count"], stats["grid_aligned_peaks"]
        other = n - grid
        if other >= 2:
            top = stats["peaks"][0]
            obs.append(self.observation(
                "notice",
                f"{other} isolated peak(s) in the Fourier spectrum away from the 1/8 codec grid "
                f"(strongest at {top['fx']:+.3f}, {top['fy']:+.3f} cycles/px, prominence "
                f"{top['prominence']:.1f}); regular peaks are left by upsampling, generative decoders, "
                "halftone or screen moire and tiled textures.",
                peak_count=n, grid_aligned_peaks=grid, strongest=top))
        elif n:
            obs.append(self.observation(
                "info",
                f"{n} spectral peak(s), {grid} on the 1/8 cycles/px block grid that lossy "
                "compression leaves; nothing beyond codec structure.",
                peak_count=n, grid_aligned_peaks=grid))
        else:
            obs.append(self.observation(
                "info", "The Fourier spectrum decays smoothly with no isolated peaks.", peak_count=0))
        if media_type == "video" and stats["frames_averaged"] > 1:
            obs.append(self.observation(
                "info", f"Spectrum averaged over {stats['frames_averaged']} evenly spaced frames.",
                frames_averaged=stats["frames_averaged"]))
        return obs
