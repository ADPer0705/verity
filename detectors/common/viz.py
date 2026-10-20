from __future__ import annotations

import numpy as np
from PIL import Image
from matplotlib.patches import Rectangle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_attention_overlay(orig_pil: Image.Image, att_map: np.ndarray,
                           save_path, face_box=None, title="Attention") -> None:
    """Three-panel explainability figure: original, heatmap, overlay.

    ``att_map`` is a 2D map in [0, 1] at the original image resolution.
    """
    if att_map.ndim != 2:
        raise ValueError("attention map must be 2D")
    if float(att_map.std()) < 1e-6:
        raise RuntimeError("attention map has no spatial variation")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax in axes:
        ax.axis("off")

    axes[0].imshow(orig_pil)
    axes[0].set_title("Original", fontsize=10)
    if face_box:
        x, y, w, h = face_box
        axes[0].add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="lime", linewidth=2))

    axes[1].imshow(att_map, cmap="hot", vmin=0, vmax=1)
    axes[1].set_title(f"{title} Heatmap", fontsize=10)

    overlay = np.array(orig_pil.convert("RGB")).astype("float32") / 255.0
    for c in range(3):
        overlay[:, :, c] = overlay[:, :, c] * 0.5 + att_map * 0.5
    axes[2].imshow(np.clip(overlay, 0, 1))
    axes[2].set_title("Overlay", fontsize=10)

    plt.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_timeline(per_face, save_path, total_frames: int, threshold: float = 0.5) -> None:
    """Per-frame prediction timeline: mean/range line plot + bar chart."""
    frames = sorted({r["frame_idx"] for r in per_face})
    frame_probs: dict[int, list[float]] = {}
    for r in per_face:
        frame_probs.setdefault(r["frame_idx"], []).append(r["fake_prob"])
    frame_numbers = sorted(frame_probs)
    mean_probs = [float(np.mean(frame_probs[f])) for f in frame_numbers]
    max_probs = [float(np.max(frame_probs[f])) for f in frame_numbers]
    min_probs = [float(np.min(frame_probs[f])) for f in frame_numbers]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle("Detector Per-Face Prediction Timeline", fontsize=14, fontweight="bold")

    if frame_numbers:
        ax1.fill_between(frame_numbers, min_probs, max_probs, alpha=0.2, color="orange", label="Range")
    ax1.plot(frame_numbers, mean_probs, "o-", color="red", linewidth=2, markersize=6, label="Mean")
    ax1.axhline(threshold, color="gray", linestyle="--", alpha=0.5, label=f"Threshold ({threshold})")
    ax1.axhline(1 - threshold, color="gray", linestyle="--", alpha=0.3)
    ax1.set_ylabel("Fake Probability")
    ax1.set_xlim(0, max(frame_numbers) if frame_numbers else 1)
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    colors = ["red" if p > threshold else "green" if p < 1 - threshold else "gray" for p in mean_probs]
    ax2.bar(frame_numbers, mean_probs, color=colors, width=0.6, alpha=0.8)
    ax2.axhline(threshold, color="gray", linestyle="--", alpha=0.5)
    ax2.axhline(1 - threshold, color="gray", linestyle="--", alpha=0.3)
    ax2.set_xlabel("Frame Number")
    ax2.set_ylabel("Mean Fake Prob")
    ax2.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_top_frames(frame_cache, per_face, output_dir, top_n: int = 3) -> None:
    """Save full frames (with the face box drawn) for the most-suspicious
    faces. ``frame_cache`` maps sampled frame_idx -> RGB PIL image."""
    top = sorted(per_face, key=lambda r: r["fake_prob"], reverse=True)[:top_n]
    full_dir = output_dir / "full_frames"
    full_dir.mkdir(exist_ok=True)
    for face_info in top:
        f_idx = face_info["frame_idx"]
        rgb = frame_cache.get(f_idx)
        if rgb is None:
            continue
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.imshow(rgb)
        x, y, w, h = face_info["bbox"]
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="lime", linewidth=3))
        ax.set_title(f"{face_info['face_id']} — fake prob: {face_info['fake_prob']:.4f}", fontsize=12)
        ax.axis("off")
        fig.savefig(str(full_dir / f"{face_info['face_id']}_frame.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


_REGION_TOPO = {
    "left_eye": {33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246},
    "right_eye": {362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398},
    "nose": {1, 2, 98, 97, 19, 239, 458, 49, 279, 429, 437, 438, 420, 445, 275, 440,
             5, 4, 45, 44, 195, 220, 237, 244, 275, 279, 429, 437, 438, 420, 445, 440},
    "mouth": {0, 17, 61, 146, 91, 181, 84, 314, 405, 321, 375, 291, 409, 270, 269, 267,
              13, 14, 78, 95, 88, 178, 87, 317, 402, 318, 324, 308, 415, 310, 311, 312},
    "jaw": {152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162,
            454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 356, 389, 251, 284},
    "forehead": {10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
                 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
                 234, 127, 162, 21, 54, 103, 67, 109, 10},
}


def compute_region_energy(face_pil: Image.Image, att_map: np.ndarray,
                          face_mesh) -> dict | None:
    """Share of attention energy concentrated in each facial region.

    ``att_map`` is a 2D map in [0, 1] at the face crop's native resolution.
    Returns ``None`` if mediapipe is unavailable or no face mesh is found.
    """
    if face_mesh is None or att_map is None or att_map.ndim != 2:
        return None
    import cv2

    rgb = np.array(face_pil.convert("RGB"))
    h_orig, w_orig = rgb.shape[:2]
    results = face_mesh.process(rgb)
    if not results or not results.multi_face_landmarks:
        return None
    lm = results.multi_face_landmarks[0]
    pts = np.array([(p.x * w_orig, p.y * h_orig) for p in lm.landmark])

    from PIL import Image as _Image
    att_resized = np.array(
        _Image.fromarray((att_map * 255).astype("uint8")).resize((w_orig, h_orig), _Image.BILINEAR)
    ) / 255.0

    total = att_resized.sum() + 1e-8
    result = {}
    for name, idx_set in _REGION_TOPO.items():
        valid = [i for i in idx_set if i < len(pts)]
        if len(valid) < 3:
            continue
        hull = cv2.convexHull(pts[valid].astype(np.float32).reshape(-1, 1, 2)).astype(np.int32)
        mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
        cv2.fillPoly(mask, [hull], 1)
        if mask.sum() == 0:
            continue
        result[name] = round(float((att_resized * mask).sum() / total * 100), 1)
    return result


def save_audio_timeline(waveform, sample_rate: int, windows: list[dict], save_path,
                        threshold: float = 0.5) -> None:
    """Log-mel-ish spectrogram (plain STFT in dB) with per-window fake
    probability drawn above it. ``windows``: dicts with start_s, end_s, fake_prob."""
    x = np.asarray(waveform, dtype="float32")
    n_fft, hop = 1024, 256
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    win = np.hanning(n_fft).astype("float32")
    n_frames = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    spec = np.abs(np.fft.rfft(x[idx] * win, axis=1)).T          # (freq, time)
    spec_db = 20 * np.log10(spec + 1e-6)
    duration = len(x) / float(sample_rate)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={"height_ratios": [1, 2]},
                                   sharex=True)
    fig.suptitle("Audio Window Prediction Timeline", fontsize=14, fontweight="bold")
    for w in windows:
        p = w["fake_prob"]
        color = "red" if p > threshold else "green" if p < 1 - threshold else "gray"
        ax1.barh(0, w["end_s"] - w["start_s"], left=w["start_s"], height=p, color=color, alpha=0.7)
    ax1.axhline(threshold, color="gray", linestyle="--", alpha=0.5)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Fake prob")
    ax2.imshow(spec_db, origin="lower", aspect="auto", cmap="magma",
               extent=[0, duration, 0, sample_rate / 2000.0])
    ax2.set_ylabel("kHz")
    ax2.set_xlabel("Time (s)")
    plt.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
