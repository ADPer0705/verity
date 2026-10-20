"""Compression history: error level analysis, JPEG quantization tables,
double-JPEG periodicity and block-grid consistency.

Rationale: lossy compression leaves a structured footprint. Every JPEG save
quantizes 8x8 DCT blocks with a known table, so

* **Error Level Analysis (ELA)** re-saves the image at a fixed quality and
  looks at what changes. Content that has already been through that
  quality changes little; freshly pasted or re-rendered regions change more.
  Flat and highly textured regions respond differently too, so ELA is a
  map to read, not a score to trust.
* **Quantization tables** reveal the encoder's quality setting and whether
  the tables are the standard IJG ones (libjpeg, most phones) or custom
  (Photoshop, some cameras).
* **Double-JPEG periodicity**: compressing once gives a smooth, roughly
  Laplacian histogram of quantized DCT coefficients. Compressing a second
  time with a different table leaves periodic gaps and peaks in that
  histogram (Popescu & Farid, 2004; Lukas & Fridrich, 2003). We measure the
  periodicity as the strongest non-trivial peak in the Fourier transform of
  the de-trended histogram of the lowest-frequency AC coefficients.
* **Blockiness** compares pixel differences across 8x8 block boundaries with
  differences elsewhere; a map of it shows where the JPEG grid is strong,
  weak or shifted. A pasted JPEG region whose grid does not align with the
  host's is a classic splicing trace.

None of this identifies *what* was changed; re-saving a whole image also
double-compresses it, and heavy social-media recompression blurs every one
of these traces.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from verity.forensics.base import (ForensicAnalyzer, bounded, normalize_u8, r, sample_video_frames,
                                   to_gray, write_json, write_png)
from verity.forensics.models import ForensicResult, Observation

ELA_QUALITY = 90
#: ELA display gain is chosen so the 99th percentile maps to full scale,
#: clamped to this range (the raw statistics are gain-independent).
ELA_GAIN_MIN, ELA_GAIN_MAX = 4.0, 40.0
#: Grey-level floor for the blockiness contrast denominator: flat areas have
#: near-zero differences at every phase and would otherwise explode.
CONTRAST_FLOOR = 0.5
#: Decoded video frames inspected (the first ones, as the GOP structure is
#: identical across the stream and ELA on video is only indicative).
VIDEO_FRAMES = 5
GRID_CELL = 64
#: Peak-to-background ratio in the histogram spectrum above which the
#: quantized-coefficient histogram is called periodic.
PERIODICITY_THRESHOLD = 5.0
HIST_RANGE = 40
MIN_POPULATED_BINS = 12
#: Fraction of textured cells without a block grid (under a strong global
#: grid) above which the picture is flagged.
GRIDLESS_FRACTION = 0.15

#: IJG standard luminance table (natural order), Annex K of ITU-T T.81.
STD_LUMA = np.array([
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
], dtype=np.float64).reshape(8, 8)

#: Low-frequency AC positions (row, col) used for the periodicity test.
AC_POSITIONS = [(0, 1), (1, 0), (1, 1), (0, 2), (2, 0)]


def ijg_table(quality: int) -> np.ndarray:
    """The IJG luminance table for a given quality factor (libjpeg scaling)."""
    q = max(1, min(100, int(quality)))
    scale = 5000.0 / q if q < 50 else 200.0 - 2 * q
    table = np.floor((STD_LUMA * scale + 50) / 100)
    return np.clip(table, 1, 255)


def estimate_quality(table: np.ndarray) -> Tuple[int, float]:
    """Quality whose IJG table is closest (L1) to ``table``; 0 distance means
    the table is exactly the standard one."""
    best_q, best_d = 1, float("inf")
    for q in range(1, 101):
        d = float(np.abs(ijg_table(q) - table).sum())
        if d < best_d:
            best_q, best_d = q, d
    return best_q, best_d


def _dct_matrix() -> np.ndarray:
    n = 8
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    mat = np.sqrt(2.0 / n) * np.cos(np.pi * (2 * i + 1) * k / (2 * n))
    mat[0, :] = np.sqrt(1.0 / n)
    return mat


DCT8 = _dct_matrix()


def block_dct(gray: np.ndarray) -> np.ndarray:
    """8x8 block DCT-II (JPEG convention, level-shifted) -> (H/8, W/8, 8, 8)."""
    h, w = gray.shape
    h8, w8 = h - h % 8, w - w % 8
    blocks = gray[:h8, :w8].astype(np.float64).reshape(h8 // 8, 8, w8 // 8, 8).transpose(0, 2, 1, 3) - 128.0
    return np.einsum("ij,abjk,lk->abil", DCT8, blocks, DCT8)


def periodicity_score(values: np.ndarray, span: int = HIST_RANGE,
                      min_bins: int = MIN_POPULATED_BINS) -> Tuple[Optional[float], Optional[float], int]:
    """Peak-to-background ratio of the spectrum of the de-trended histogram
    of integer coefficient values in [-span, span], the period in bins, and
    the number of populated bins.

    The histogram of a single-compressed coefficient is smooth; a second
    quantization with a different step stamps a comb onto it. Removing the
    smooth envelope (a 5-bin moving average) and looking for a dominant
    Fourier component makes the comb measurable independently of content.
    Coarsely quantized coefficients populate only a handful of bins, which
    leaves no room for a comb to show, so the test is skipped (``None``)
    when fewer than ``min_bins`` bins are populated."""
    hist, _ = np.histogram(values, bins=np.arange(-span - 0.5, span + 1.5, 1.0))
    hist = hist.astype(np.float64)
    hist[span] = (hist[span - 1] + hist[span + 1]) / 2.0  # zero bin is dominated by flat content
    populated = np.nonzero(hist >= max(5.0, 0.002 * hist.max()))[0]
    if len(populated) < min_bins:
        return None, None, int(len(populated))
    lo, hi = int(populated.min()), int(populated.max()) + 1
    h = hist[lo:hi]
    trend = np.convolve(h, np.ones(5) / 5.0, mode="same")
    detrended = (h - trend) / (trend + 1.0)
    detrended *= np.hanning(len(detrended))
    spectrum = np.abs(np.fft.rfft(detrended))
    if len(spectrum) < 6:
        return None, None, int(len(populated))
    band = spectrum[2:]
    background = float(np.median(band)) + 1e-9
    peak_idx = int(np.argmax(band)) + 2
    return float(band.max() / background), float(len(detrended) / peak_idx), int(len(populated))


class CompressionAnalyzer(ForensicAnalyzer):
    analyzer_id = "compression"
    label = "Compression history"
    media_types = ("image", "video")
    method = (
        "Error level analysis re-saves the picture as JPEG at quality 90 and maps "
        "how much each pixel changes; regions that respond differently from their "
        "surroundings have a different compression history, but flat versus "
        "textured content also responds differently, so the map needs a human "
        "reader. For JPEG files the quantization tables give the encoder's quality "
        "setting, and the histogram of low-frequency quantized DCT coefficients is "
        "tested for the periodic comb a second JPEG save leaves behind. The "
        "blockiness map shows where the 8x8 grid is strong, weak or shifted. "
        "Re-saving a whole image produces the same traces as editing part of it, "
        "and aggressive platform recompression erases them."
    )

    def analyze(self, media_id: str, media_type: str) -> ForensicResult:
        path = self.media_path(media_id)
        out_dir = self.output_dir(media_id)
        observations: List[Observation] = []
        data: Dict[str, Any] = {"source": media_type}
        artifacts: List[Path] = []

        frames = self._load_frames(path, media_type)
        if not frames:
            raise RuntimeError("no decodable image data")
        first_idx, first = frames[0]
        first, scale = bounded(first)
        data["analysis_scale"] = r(scale, 4)
        data["frame_index"] = first_idx

        # --- ELA ---------------------------------------------------------- #
        ela, ela_stats = self._ela(first)
        artifacts.append(write_png(out_dir / "ela.png", ela))
        overlay = self._overlay(first, ela)
        artifacts.append(write_png(out_dir / "ela_overlay.png", overlay))
        data["ela"] = ela_stats
        observations.extend(self._ela_observations(ela_stats, media_type))

        # --- Blockiness grid ---------------------------------------------- #
        gray = to_gray(first)
        block_map, block_stats = self._blockiness(gray)
        artifacts.append(write_png(out_dir / "blockiness.png", block_map))
        data["blockiness"] = block_stats
        if media_type == "video" and len(frames) > 1:
            per_frame = []
            for idx, frame in frames[1:]:
                frame, _ = bounded(frame)
                _, stats = self._blockiness(to_gray(frame))
                per_frame.append({"frame_index": idx, "global_blockiness": stats["global_blockiness"],
                                  "gridless_cell_fraction": stats["gridless_cell_fraction"]})
            data["blockiness_per_frame"] = per_frame
        observations.extend(self._blockiness_observations(block_stats, path, media_type))

        # --- JPEG specifics ----------------------------------------------- #
        jpeg = self._jpeg_info(path) if media_type == "image" else None
        if jpeg is not None:
            data["jpeg"] = jpeg
            observations.extend(self._jpeg_observations(jpeg))
        elif media_type == "image":
            data["jpeg"] = None
            observations.append(self.observation(
                "info", f"Not a JPEG ({path.suffix.lower()}): quantization-table and double-JPEG "
                "tests do not apply; the blockiness map still reveals a prior JPEG grid.",
                suffix=path.suffix.lower()))

        artifacts.append(write_json(out_dir / "compression.json", data))
        return self.result(observations, artifacts, data={
            "ela_mean": ela_stats["mean"],
            "ela_cell_ratio": ela_stats["cell_max_over_median"],
            "global_blockiness": block_stats["global_blockiness"],
            "jpeg_quality": (jpeg or {}).get("estimated_quality"),
            "double_jpeg_score": (jpeg or {}).get("periodicity_score"),
        })

    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_frames(path: Path, media_type: str) -> List[Tuple[int, np.ndarray]]:
        if media_type == "video":
            return sample_video_frames(path, VIDEO_FRAMES, first=True)
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            with Image.open(path) as pil:
                img = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
        return [(0, img)]

    def _ela(self, bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=ELA_QUALITY, subsampling=0)
        buf.seek(0)
        resaved = np.array(Image.open(buf).convert("RGB"))
        diff = np.abs(rgb.astype(np.int16) - resaved.astype(np.int16)).max(axis=2).astype(np.float64)
        p99 = float(np.percentile(diff, 99))
        gain = float(np.clip(255.0 / max(p99, 1.0), ELA_GAIN_MIN, ELA_GAIN_MAX))
        amplified = np.clip(diff * gain, 0, 255).astype(np.uint8)
        cells = self._cell_means(diff, GRID_CELL)
        median_cell = float(np.median(cells)) + 1e-6
        ratio = float(cells.max() / median_cell)
        hot = int(np.argmax(cells))
        rows, cols = cells.shape
        stats = {
            "quality": ELA_QUALITY,
            "gain": r(gain, 2),
            "mean": r(diff.mean()),
            "p95": r(np.percentile(diff, 95)),
            "max": r(diff.max()),
            "cell_size": GRID_CELL,
            "cell_grid": [int(rows), int(cols)],
            "cell_max_over_median": r(ratio, 3),
            "hottest_cell": [int(hot // cols), int(hot % cols)],
            "cells_above_3x_median": int((cells > 3 * median_cell).sum()),
        }
        return amplified, stats

    @staticmethod
    def _overlay(bgr: np.ndarray, ela: np.ndarray) -> np.ndarray:
        """Heat-map blended onto a desaturated copy, with opacity following the
        ELA strength so the picture stays readable where nothing responds."""
        base = cv2.cvtColor(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR).astype(np.float64)
        heat = cv2.applyColorMap(ela, cv2.COLORMAP_INFERNO).astype(np.float64)
        alpha = np.clip(ela.astype(np.float64) / 255.0 * 1.5, 0.0, 0.9)[:, :, None]
        return np.clip(base * (1.0 - alpha) * 0.85 + 20 + heat * alpha, 0, 255).astype(np.uint8)

    @staticmethod
    def _cell_means(arr: np.ndarray, cell: int) -> np.ndarray:
        h, w = arr.shape[:2]
        rows, cols = max(1, h // cell), max(1, w // cell)
        trimmed = arr[:rows * cell, :cols * cell]
        return trimmed.reshape(rows, cell, cols, cell).mean(axis=(1, 3))

    def _ela_observations(self, stats: Dict[str, Any], media_type: str) -> List[Observation]:
        obs: List[Observation] = []
        ratio = stats["cell_max_over_median"] or 0.0
        if ratio >= 4.0 and stats["cells_above_3x_median"] >= 1:
            obs.append(self.observation(
                "notice",
                f"ELA is uneven: the strongest {GRID_CELL}px cell responds {ratio:.1f}x more to "
                f"recompression than the median cell ({stats['cells_above_3x_median']} cell(s) above "
                "3x); such regions have a different compression history or much finer texture "
                "than their surroundings, so compare the overlay with the content.",
                cell_max_over_median=ratio, hottest_cell=stats["hottest_cell"],
                cells_above_3x_median=stats["cells_above_3x_median"]))
        else:
            obs.append(self.observation(
                "info",
                f"ELA response is spatially even (strongest cell {ratio:.1f}x the median), as "
                "expected for a picture with a single compression history.",
                cell_max_over_median=ratio, mean=stats["mean"]))
        if media_type == "video":
            obs.append(self.observation(
                "info", "ELA on a decoded video frame reflects the video codec's own quantization, "
                "not a JPEG history; it is shown for spatial unevenness only."))
        return obs

    def _blockiness(self, gray: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Local 8x8 blockiness map.

        Blockiness is the contrast between pixel differences across the block
        boundary phase (one of eight column/row offsets) that dominates the
        whole picture and the differences at the other seven phases. Cells
        with texture but no grid under a strong global grid are outlined:
        content that was never JPEG-compressed, or was resampled, pasted into
        a compressed host and saved losslessly looks like that. Local phase
        estimates on 64px cells are too noisy to call misalignment on their
        own, so the map is for reading, with the gridless count as the only
        derived statistic."""
        g = gray.astype(np.float64)
        dx = np.abs(np.diff(g, axis=1))  # difference between column x and x+1
        dy = np.abs(np.diff(g, axis=0))
        h, w = g.shape
        col_phase = np.array([dx[:, p::8].mean() for p in range(8)])
        row_phase = np.array([dy[p::8, :].mean() for p in range(8)])
        gx, gy = int(np.argmax(col_phase)), int(np.argmax(row_phase))
        global_blockiness = (self._contrast(col_phase, gx) + self._contrast(row_phase, gy)) / 2
        cell = GRID_CELL
        rows, cols = max(1, h // cell), max(1, w // cell)
        local = np.zeros((rows, cols), dtype=np.float64)
        textured = np.zeros((rows, cols), dtype=bool)
        for i in range(rows):
            for j in range(cols):
                sx = dx[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell]
                sy = dy[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell]
                cp = np.array([sx[:, p::8].mean() for p in range(8)])
                rp = np.array([sy[p::8, :].mean() for p in range(8)])
                local[i, j] = (self._contrast(cp, gx) + self._contrast(rp, gy)) / 2
                textured[i, j] = (sx.mean() + sy.mean()) / 2 > 1.5
        strong_global = global_blockiness > 0.3
        gridless = textured & (local < 0.1) if strong_global else np.zeros_like(textured)
        n_textured = int(textured.sum())
        frac = float(gridless.sum() / n_textured) if n_textured else 0.0
        vis = normalize_u8(local, lo=0.0, hi=1.0)  # fixed scale: comparable across pictures
        vis = cv2.resize(vis, (cols * cell, rows * cell), interpolation=cv2.INTER_NEAREST)
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_VIRIDIS)
        for i, j in zip(*np.nonzero(gridless)):
            cv2.rectangle(vis, (int(j * cell), int(i * cell)), (int((j + 1) * cell - 1), int((i + 1) * cell - 1)),
                          (0, 0, 255), 2)
        stats = {
            "cell_size": cell,
            "cell_grid": [int(rows), int(cols)],
            "grid_phase": [gx, gy],
            "global_blockiness": r(global_blockiness, 4),
            "local_mean": r(local.mean(), 4),
            "local_max": r(local.max(), 4),
            "local_min": r(local.min(), 4),
            "textured_cells": n_textured,
            "gridless_cells": int(gridless.sum()),
            "gridless_cell_fraction": r(frac, 4),
            "gridless_threshold": GRIDLESS_FRACTION,
        }
        return vis, stats

    @staticmethod
    def _contrast(phases: np.ndarray, p: int) -> float:
        rest = np.delete(phases, p)
        return float((phases[p] - rest.mean()) / (rest.mean() + CONTRAST_FLOOR))

    def _blockiness_observations(self, stats: Dict[str, Any], path: Path, media_type: str) -> List[Observation]:
        obs: List[Observation] = []
        gb = stats["global_blockiness"] or 0.0
        suffix = path.suffix.lower()
        lossless = media_type == "image" and suffix in (".png", ".bmp", ".tif", ".tiff", ".webp")
        if gb > 0.15:
            if lossless:
                obs.append(self.observation(
                    "notice",
                    f"A clear 8x8 block grid (blockiness {gb:.2f}) is present in a {suffix} file; the "
                    "picture was JPEG-compressed before being saved in a lossless format.",
                    global_blockiness=gb, grid_phase=stats["grid_phase"]))
            else:
                obs.append(self.observation(
                    "info", f"8x8 block grid visible (blockiness {gb:.2f}), consistent with the "
                    "file's lossy compression.", global_blockiness=gb))
        else:
            obs.append(self.observation(
                "info", f"Weak or no 8x8 block grid (blockiness {gb:.2f}); the picture is high "
                "quality, resized after compression, or never block-compressed.",
                global_blockiness=gb))
        if stats["textured_cells"] >= 6 and stats["gridless_cell_fraction"] >= GRIDLESS_FRACTION:
            obs.append(self.observation(
                "notice",
                f"{stats['gridless_cells']} of {stats['textured_cells']} textured cells "
                f"({stats['gridless_cell_fraction']:.0%}) show no block grid although the picture as a "
                "whole has a strong one; content that was pasted in losslessly, resampled or rendered "
                "after the last JPEG save looks like this (cells outlined in the blockiness map).",
                gridless_cells=stats["gridless_cells"], textured_cells=stats["textured_cells"],
                gridless_cell_fraction=stats["gridless_cell_fraction"]))
        return obs

    def _jpeg_info(self, path: Path) -> Optional[Dict[str, Any]]:
        try:
            with Image.open(path) as img:
                if img.format != "JPEG":
                    return None
                tables = getattr(img, "quantization", None) or {}
                tables = {int(k): [int(v) for v in vals] for k, vals in tables.items()}
                info = {
                    "mode": img.mode,
                    "progressive": bool(img.info.get("progressive") or img.info.get("progression")),
                    "subsampling": self._subsampling(img),
                    "quantization_tables": {str(k): v for k, v in sorted(tables.items())},
                }
                img.draft("L", img.size)  # luma plane straight from libjpeg, no colour conversion
                luma = np.asarray(img.convert("L"))
        except Exception:  # noqa: BLE001 - not a JPEG Pillow can parse
            return None
        if 0 in tables and len(tables[0]) == 64:
            table = np.array(tables[0], dtype=np.float64).reshape(8, 8)
            q, dist = estimate_quality(table)
            info["estimated_quality"] = q
            info["table_distance_to_standard"] = r(dist, 1)
            info["standard_tables"] = dist == 0
            coeffs = block_dct(luma)
            quantized = np.round(coeffs / table[None, None, :, :])
            scores: List[float] = []
            per_pos = {}
            for (i, j) in AC_POSITIONS:
                score, period, bins = periodicity_score(quantized[:, :, i, j].ravel())
                per_pos[f"{i},{j}"] = {"score": r(score, 3), "period_bins": r(period, 2), "populated_bins": bins}
                if score is not None:
                    scores.append(score)
            info["periodicity_per_position"] = per_pos
            # the two most periodic positions carry the comb; coarse positions are gated out
            top = sorted(scores, reverse=True)[:2]
            info["periodicity_score"] = r(float(np.mean(top)), 3) if len(scores) >= 2 else None
            info["periodicity_positions_tested"] = len(scores)
            info["periodicity_threshold"] = PERIODICITY_THRESHOLD
            info["blocks_analyzed"] = int(coeffs.shape[0] * coeffs.shape[1])
        else:
            info["estimated_quality"] = None
            info["periodicity_score"] = None
        return info

    @staticmethod
    def _subsampling(img: Image.Image) -> Optional[str]:
        try:
            from PIL import JpegImagePlugin
            code = JpegImagePlugin.get_sampling(img)
        except Exception:  # noqa: BLE001
            return None
        return {0: "4:4:4", 1: "4:2:2", 2: "4:2:0"}.get(code, None if code == -1 else str(code))

    def _jpeg_observations(self, jpeg: Dict[str, Any]) -> List[Observation]:
        obs: List[Observation] = []
        q = jpeg.get("estimated_quality")
        if q is not None:
            kind = "the standard IJG table" if jpeg["standard_tables"] else "a custom table"
            obs.append(self.observation(
                "info",
                f"JPEG luminance quantization is closest to quality {q} using {kind}"
                + ("" if jpeg["standard_tables"] else
                   f" (L1 distance {jpeg['table_distance_to_standard']} from the standard one, "
                   "as written by Photoshop, some cameras and many web services)")
                + f"; chroma subsampling {jpeg.get('subsampling') or 'n/a'}.",
                estimated_quality=q, standard_tables=jpeg["standard_tables"],
                subsampling=jpeg.get("subsampling"), progressive=jpeg["progressive"]))
        score = jpeg.get("periodicity_score")
        if q is not None and score is None:
            obs.append(self.observation(
                "info",
                "Double-JPEG test not applicable: the quantization is too coarse for the "
                "low-frequency coefficient histograms to show a comb.",
                positions_tested=jpeg.get("periodicity_positions_tested", 0)))
        elif score is not None:
            if score >= PERIODICITY_THRESHOLD:
                obs.append(self.observation(
                    "notice",
                    f"The histogram of low-frequency quantized DCT coefficients is periodic (score "
                    f"{score:.1f}, threshold {PERIODICITY_THRESHOLD:.0f}), the signature of a JPEG that "
                    "was saved at least twice with different quality settings; this follows from any "
                    "re-save, including a crop or a platform re-upload, not only from local editing.",
                    periodicity_score=score, threshold=PERIODICITY_THRESHOLD,
                    positions=jpeg["periodicity_per_position"]))
            else:
                obs.append(self.observation(
                    "info",
                    f"No double-JPEG periodicity in the quantized DCT histogram (score {score:.1f}, "
                    f"threshold {PERIODICITY_THRESHOLD:.0f}); a single compression or a re-save at "
                    "the same or a much lower quality, which this test cannot see.",
                    periodicity_score=score, threshold=PERIODICITY_THRESHOLD))
        return obs
