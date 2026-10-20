from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image

from .cam import grad_cam, upscale_to_pil
from .face_utils import FaceDetector
from .preprocess import pil_to_tensor
from .reasoning import aggregate, aggregate_regions, build_reasoning
from .sampling import VideoReader, build_image_tensor, crop_rgb
from .spec import DetectorSpec, load_spec
from .viz import (compute_region_energy, save_attention_overlay, save_audio_timeline,
                  save_timeline, save_top_frames)

# Deterministic inference: fixed seed for any stochastic ops, and conservative
# determinism flags for CUDA. In eval mode the published checkpoints are
# already deterministic on a given device; these pins make it explicit.
SEED = 0xC0FFEE

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".3gp", ".ts")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif")
AUDIO_EXTS = (".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma", ".amr")

INPUT_DIR = Path(os.environ.get("INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/output"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_face_mesh():
    try:
        import mediapipe as mp
        return mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5
        )
    except Exception:
        return None


def _tensor_to_clip(x: np.ndarray, mean, std, device) -> torch.Tensor:
    """(T, H, W, 3) float01 -> (1, 3, T, H, W) normalized tensor."""
    arr = (x - np.asarray(mean, dtype="float32")) / np.asarray(std, dtype="float32")
    t = torch.from_numpy(arr).permute(3, 0, 1, 2).unsqueeze(0).to(device)
    return t


def run_detector(build_model: Callable, spec_path: Path) -> None:
    """Standalone detector entrypoint shared by every detector.

    ``build_model(weights_path, device)`` comes from the detector's ``model.py``.
    """
    spec = load_spec(spec_path)
    detector_dir = spec_path.parent
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    media_files = sorted((INPUT_DIR / "media").iterdir()) if (INPUT_DIR / "media").exists() else []
    if not media_files:
        (OUTPUT_DIR / "result.json").write_text(json.dumps({
            "prediction": 0.5, "confidence": 0.0,
            "reasoning": "No media file found.",
            "confidence_factors": ["No input media"],
        }, indent=2))
        return

    weights_path = _resolve_weights(spec, detector_dir)
    weight_status = _weight_status(spec, weights_path)

    start = time.time()
    t_load = time.time()
    model = build_model(str(weights_path) if weights_path else None, DEVICE)
    load_time = time.time() - t_load

    mode = spec.input_mode
    unit = {"face_crop": "face", "face_clip": "face", "full_frame": "frame",
            "full_clip": "clip", "audio": "window"}.get(mode, "face")
    needs_faces = mode in ("face_crop", "face_clip")
    face_detector = FaceDetector(device=str(DEVICE), min_face=spec.min_face) if needs_faces else None
    face_mesh = _make_face_mesh() if needs_faces else None

    all_per_face: list[dict] = []
    timing = {"face_detection": 0.0, "preprocess": 0.0, "inference": 0.0, "viz": 0.0}
    total_frames = 0
    sample_interval_ = spec.sample_interval
    coverage_pct = 0.0
    audio_info: dict | None = None

    for media_path in media_files:
        print(f"  Processing: {media_path.name} ({mode})", file=sys.stderr)
        ext = media_path.suffix.lower()
        is_video = ext in VIDEO_EXTS
        is_image = ext in IMAGE_EXTS
        is_audio = ext in AUDIO_EXTS
        if mode == "audio":
            if not (is_video or is_audio):
                print(f"  [skip] {ext} carries no audio", file=sys.stderr)
                continue
            try:
                pf, info = _process_audio(model, media_path, spec, OUTPUT_DIR, timing)
            except AudioUnavailable as exc:
                print(f"  [skip] {exc}", file=sys.stderr)
                audio_info = {"error": str(exc)}
                continue
            all_per_face.extend(pf)
            audio_info = info
        elif mode in ("full_frame", "full_clip"):
            if is_video:
                pf, n_total, interval, cov = _process_video_full(
                    model, media_path, spec, OUTPUT_DIR, timing)
                all_per_face.extend(pf)
                total_frames = max(total_frames, n_total)
                sample_interval_ = interval
                coverage_pct = cov
            elif is_image and mode == "full_frame":
                all_per_face.extend(_process_image_full(model, media_path, spec, OUTPUT_DIR, timing))
            else:
                print(f"  [skip] {ext} not usable in {mode} mode", file=sys.stderr)
        elif is_video:
            pf, tim, n_total, interval, cov = _process_video(
                model, media_path, spec, face_detector, face_mesh, OUTPUT_DIR, timing)
            all_per_face.extend(pf)
            total_frames = max(total_frames, n_total)
            sample_interval_ = interval
            coverage_pct = cov
        elif is_image:
            pf, tim = _process_image(
                model, media_path, spec, face_detector, face_mesh, OUTPUT_DIR, timing)
            all_per_face.extend(pf)
        else:
            print(f"  [skip] unsupported extension {ext}", file=sys.stderr)

    mean_p, confidence, stats = aggregate(all_per_face, threshold=0.5)
    summary, factors = build_reasoning(
        spec.name, mean_p, stats, timing, str(DEVICE),
        per_face=all_per_face, weight_status=weight_status,
        cam_available=spec.cam and mode != "audio", threshold=0.5, unit=unit,
    )

    result = {
        "prediction": round(mean_p, 4),
        "confidence": round(confidence, 4),
        "reasoning": summary,
        "confidence_factors": factors,
        "stats": stats,
        "weights": {
            "status": weight_status,
            "file": spec.weights_file,
            "note": spec.weight_note,
        },
        "timing_seconds": {
            "model_load": round(load_time, 3),
            "face_detection": round(timing["face_detection"], 3),
            "preprocess": round(timing["preprocess"], 3),
            "inference": round(timing["inference"], 3),
            "visualization": round(timing["viz"], 3),
            "total": round(time.time() - start, 3),
        },
        "device": str(DEVICE),
        "model": {
            "name": spec.name,
            "version": spec.version,
            "architecture": spec.architecture,
            "parameters": spec.parameters,
            "pretrained": spec.pretrained,
        },
        "unit": unit,
        "input_mode": mode,
        "per_face": all_per_face,
        "audio": audio_info,
        "total_video_frames": total_frames,
        "sampling": {
            "sample_interval": sample_interval_,
            "coverage_pct": round(coverage_pct, 1),
        } if total_frames else None,
        "region_aggregate": aggregate_regions(all_per_face),
        "execution_context": {
            "runtime": os.environ.get("VERITY_RUNTIME", "native"),
            "sif": os.environ.get("VERITY_SIF", ""),
            "detector_id": spec.detector_id,
        },
    }
    (OUTPUT_DIR / "result.json").write_text(json.dumps(result, indent=2))
    (OUTPUT_DIR / "per_face_results.json").write_text(json.dumps(all_per_face, indent=2))
    print(f"Done. {len(all_per_face)} faces, {time.time() - start:.1f}s", file=sys.stderr)


def _resolve_weights(spec: DetectorSpec, detector_dir: Path) -> Path | None:
    if not spec.weights_file:
        return None
    return detector_dir / "weights" / spec.weights_file


def _weight_status(spec: DetectorSpec, weights_path: Path | None) -> str:
    if weights_path is not None and weights_path.exists():
        return spec.weights_kind
    if weights_path is not None:
        return "missing"
    return spec.weights_kind  # external/random (weights loaded by build_model)


def _score_one(model, x: torch.Tensor) -> tuple[float, dict]:
    with torch.no_grad():
        out = model.forward_detailed(x)
    logits = out["logits"]
    prob = torch.softmax(logits, dim=1)[0, 1].item()
    return prob, out


def _cam_for(spec, model, x: torch.Tensor, out: dict) -> np.ndarray | None:
    """Prefer the detector's native attention map; fall back to Grad-CAM."""
    native = out.get("attention_map")
    if native is not None:
        try:
            t = native.detach().cpu().squeeze()
            if t.ndim == 3:
                t = t.mean(dim=0)
            arr = t.numpy()
            if arr.ndim == 2 and float(arr.std()) > 1e-6:
                mx, mn = arr.max(), arr.min()
                return ((arr - mn) / (mx - mn + 1e-8)).astype("float32")
        except Exception:
            pass
    if not spec.cam:
        return None
    return grad_cam(model, x)


def _process_image(model, media_path, spec, face_detector, face_mesh,
                   output_dir, timing):
    per_face: list[dict] = []
    pil = Image.open(media_path).convert("RGB")

    t0 = time.time()
    boxes = face_detector.detect(pil)
    timing["face_detection"] += time.time() - t0

    for i, (x, y, w, h) in enumerate(boxes):
        face = pil.crop((x, y, x + w, y + h))
        t1 = time.time()
        tensor = pil_to_tensor(face, spec.resolution, spec.mean, spec.std).to(DEVICE)
        timing["preprocess"] += time.time() - t1

        t2 = time.time()
        prob, out = _score_one(model, tensor)
        timing["inference"] += time.time() - t2

        face_id = f"img{i:03d}"
        record = {"face_id": face_id, "frame_idx": 0, "face_index": i,
                  "bbox": [x, y, w, h], "fake_prob": round(prob, 4)}
        per_face.append(record)

        (output_dir / "faces").mkdir(exist_ok=True)
        face.save(output_dir / "faces" / f"{face_id}.png")

        t3 = time.time()
        cam = _cam_for(spec, model, tensor, out)
        if cam is not None:
            (output_dir / "attention_maps").mkdir(exist_ok=True)
            att = upscale_to_pil(cam, face.width, face.height)
            save_attention_overlay(face, att, output_dir / "attention_maps" / f"{face_id}_attention.png")
            if spec.region_analysis and face_mesh is not None:
                record["region_energy"] = compute_region_energy(face, att, face_mesh)
        timing["viz"] += time.time() - t3

    return per_face, timing


def _process_video(model, media_path, spec, face_detector, face_mesh,
                   output_dir, timing):
    reader = VideoReader(media_path, sample_interval=spec.sample_interval)
    n_sampled = len(reader.frames)
    if n_sampled == 0:
        return [], timing, reader.total_frames, spec.sample_interval, 0.0

    faces_dir = output_dir / "faces"
    attn_dir = output_dir / "attention_maps"
    faces_dir.mkdir(parents=True, exist_ok=True)
    attn_dir.mkdir(parents=True, exist_ok=True)

    per_face: list[dict] = []
    frame_cache: dict[int, Image.Image] = {}
    clip_mode = spec.video_mode

    for pos, frame in enumerate(reader.frames):
        rgb = frame[:, :, ::-1]  # BGR -> RGB
        pil_img = Image.fromarray(rgb)

        t0 = time.time()
        boxes = face_detector.detect(pil_img)
        timing["face_detection"] += time.time() - t0
        if not boxes:
            continue

        frame_cache[reader.frame_indices[pos]] = pil_img
        for i, (x, y, w, h) in enumerate(boxes):
            face = pil_img.crop((x, y, x + w, y + h))
            t1 = time.time()
            if clip_mode:
                clip_frames = reader.clip(pos, spec.clip_size)
                crops = crop_rgb(clip_frames, (x, y, w, h), spec.resolution)
                tensor = _tensor_to_clip(np.stack(crops, 0), spec.mean, spec.std, DEVICE)
            else:
                tensor = pil_to_tensor(face, spec.resolution, spec.mean, spec.std).to(DEVICE)
            timing["preprocess"] += time.time() - t1

            t2 = time.time()
            prob, out = _score_one(model, tensor)
            timing["inference"] += time.time() - t2

            face_id = f"f{reader.frame_indices[pos]:05d}_face{i}"
            record = {"face_id": face_id, "frame_idx": reader.frame_indices[pos],
                      "face_index": i, "bbox": [x, y, w, h], "fake_prob": round(prob, 4)}
            per_face.append(record)

            face.save(faces_dir / f"{face_id}.png")

            t3 = time.time()
            cam = _cam_for(spec, model, tensor, out)
            if cam is not None:
                att = upscale_to_pil(cam, face.width, face.height)
                save_attention_overlay(face, att, attn_dir / f"{face_id}_attention.png")
                if spec.region_analysis and face_mesh is not None:
                    record["region_energy"] = compute_region_energy(face, att, face_mesh)
            timing["viz"] += time.time() - t3

    n_scored = len({r["frame_idx"] for r in per_face})
    coverage = min(100.0, n_scored * spec.sample_interval / reader.total_frames * 100) if reader.total_frames else 0.0

    timeline_path = output_dir / "timeline.png"
    save_timeline(per_face, timeline_path, reader.total_frames, threshold=0.5)
    save_top_frames(frame_cache, per_face, output_dir, top_n=3)

    return per_face, timing, reader.total_frames, spec.sample_interval, coverage


# --------------------------------------------------------------------------- #
#  Full-frame / full-clip modes (no face detector)                             #
# --------------------------------------------------------------------------- #


def _frame_to_tensor(model, pil: Image.Image, spec: DetectorSpec) -> torch.Tensor:
    """Whole-image preprocessing. A detector's model.py may define
    ``preprocess(pil) -> (1, C, H, W) tensor`` to own this entirely (SAFE's
    wavelet crop, Community Forensics' resize-440/crop-384, ...); otherwise:
    resize shorter side to ``resize_short`` (or square-resize to
    ``resolution``), optional center crop, then (x - mean) / std."""
    custom = getattr(model, "preprocess", None)
    if callable(custom):
        return custom(pil).to(DEVICE)
    pil = pil.convert("RGB")
    if spec.resize_short:
        w, h = pil.size
        scale = spec.resize_short / float(min(w, h))
        pil = pil.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC)
    if spec.center_crop:
        w, h = pil.size
        r = spec.resolution
        left, top = max(0, (w - r) // 2), max(0, (h - r) // 2)
        pil = pil.crop((left, top, left + r, top + r))
    return pil_to_tensor(pil, spec.resolution, spec.mean, spec.std).to(DEVICE)


def _evenly(n: int, budget: int) -> list[int]:
    if n <= budget:
        return list(range(n))
    return sorted({int(round(i * (n - 1) / (budget - 1))) for i in range(budget)})


def _process_image_full(model, media_path, spec, output_dir, timing) -> list[dict]:
    pil = Image.open(media_path).convert("RGB")
    t1 = time.time()
    tensor = _frame_to_tensor(model, pil, spec)
    timing["preprocess"] += time.time() - t1
    t2 = time.time()
    prob, out = _score_one(model, tensor)
    timing["inference"] += time.time() - t2
    record = {"face_id": "img00000", "frame_idx": 0, "face_index": 0,
              "bbox": [0, 0, pil.width, pil.height], "fake_prob": round(prob, 4)}
    (output_dir / "frames").mkdir(exist_ok=True)
    thumb = pil.copy()
    thumb.thumbnail((512, 512))
    thumb.save(output_dir / "frames" / "img00000.png")
    t3 = time.time()
    cam = _cam_for(spec, model, tensor, out)
    if cam is not None:
        (output_dir / "attention_maps").mkdir(exist_ok=True)
        att = upscale_to_pil(cam, thumb.width, thumb.height)
        save_attention_overlay(thumb, att, output_dir / "attention_maps" / "img00000_attention.png")
    timing["viz"] += time.time() - t3
    return [record]


def _process_video_full(model, media_path, spec, output_dir, timing):
    reader = VideoReader(media_path, sample_interval=spec.sample_interval)
    n = len(reader.frames)
    if n == 0:
        return [], reader.total_frames, spec.sample_interval, 0.0
    frames_dir = output_dir / "frames"
    attn_dir = output_dir / "attention_maps"
    frames_dir.mkdir(parents=True, exist_ok=True)
    per: list[dict] = []
    clip_mode = spec.input_mode == "full_clip"
    positions = _evenly(n, spec.max_frames)
    for pos in positions:
        frame = reader.frames[pos]
        pil = Image.fromarray(frame[:, :, ::-1])
        t1 = time.time()
        if clip_mode:
            clip = reader.clip(pos, spec.clip_size or 16)
            h, w = clip[0].shape[:2]
            crops = crop_rgb(clip, (0, 0, w, h), spec.resolution)
            tensor = _tensor_to_clip(np.stack(crops, 0), spec.mean, spec.std, DEVICE)
        else:
            tensor = _frame_to_tensor(model, pil, spec)
        timing["preprocess"] += time.time() - t1
        t2 = time.time()
        prob, out = _score_one(model, tensor)
        timing["inference"] += time.time() - t2
        fid = f"f{reader.frame_indices[pos]:05d}"
        per.append({"face_id": fid, "frame_idx": reader.frame_indices[pos], "face_index": 0,
                    "bbox": [0, 0, pil.width, pil.height], "fake_prob": round(prob, 4)})
        thumb = pil.copy()
        thumb.thumbnail((512, 512))
        thumb.save(frames_dir / f"{fid}.png")
        t3 = time.time()
        if not clip_mode:
            cam = _cam_for(spec, model, tensor, out)
            if cam is not None:
                attn_dir.mkdir(exist_ok=True)
                att = upscale_to_pil(cam, thumb.width, thumb.height)
                save_attention_overlay(thumb, att, attn_dir / f"{fid}_attention.png")
        timing["viz"] += time.time() - t3
    coverage = min(100.0, len(positions) * spec.sample_interval / reader.total_frames * 100) \
        if reader.total_frames else 0.0
    save_timeline(per, output_dir / "timeline.png", reader.total_frames, threshold=0.5)
    return per, reader.total_frames, spec.sample_interval, coverage


# --------------------------------------------------------------------------- #
#  Audio mode                                                                  #
# --------------------------------------------------------------------------- #


class AudioUnavailable(RuntimeError):
    """The media has no decodable audio track."""


def _load_audio(media_path: Path, sample_rate: int) -> np.ndarray:
    """Mono float32 waveform at ``sample_rate``. Prefers the orchestrator's
    extracted ``artifacts/audio/audio.wav``; otherwise decodes with ffmpeg."""
    import subprocess
    import tempfile
    import wave

    candidates = [INPUT_DIR / "artifacts" / "audio" / "audio.wav"]
    wav_path = next((c for c in candidates if c.exists()), None)
    tmp = None
    if wav_path is None or _wav_rate(wav_path) != sample_rate:
        tmp = Path(tempfile.mkdtemp(prefix="verity_audio_")) / "audio.wav"
        proc = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(media_path), "-vn",
                               "-acodec", "pcm_s16le", "-ar", str(sample_rate), "-ac", "1", str(tmp)],
                              capture_output=True, text=True)
        if proc.returncode != 0 or not tmp.exists():
            raise AudioUnavailable(f"no decodable audio track in {media_path.name}: "
                                   f"{(proc.stderr or '').strip()[-200:]}")
        wav_path = tmp
    with wave.open(str(wav_path), "rb") as wf:
        n = wf.getnframes()
        data = np.frombuffer(wf.readframes(n), dtype="<i2").astype("float32") / 32768.0
    return data


def _wav_rate(path: Path) -> int | None:
    import wave
    try:
        with wave.open(str(path), "rb") as wf:
            return wf.getframerate() if wf.getnchannels() == 1 else None
    except Exception:
        return None


def _process_audio(model, media_path, spec, output_dir, timing):
    sr = spec.audio_sample_rate
    t1 = time.time()
    wave_ = _load_audio(media_path, sr)
    timing["preprocess"] += time.time() - t1
    n = len(wave_)
    duration = n / float(sr)
    if n == 0:
        return [], {"sample_rate": sr, "duration_seconds": 0.0, "windows": 0}

    win = int(round((spec.audio_window_seconds or duration) * sr)) or n
    hop = int(round((spec.audio_hop_seconds or spec.audio_window_seconds or duration) * sr)) or win
    starts = list(range(0, max(1, n - win + 1), hop)) if n >= win else [0]
    if n > win and starts[-1] + win < n:
        starts.append(n - win)

    per: list[dict] = []
    windows: list[dict] = []
    for i, st in enumerate(starts):
        seg = wave_[st:st + win]
        if len(seg) < win:          # pad short signals by tiling (ASVspoof convention)
            reps = int(np.ceil(win / max(1, len(seg))))
            seg = np.tile(seg, reps)[:win]
        tensor = torch.from_numpy(seg).unsqueeze(0).to(DEVICE)
        t2 = time.time()
        prob, _out = _score_one(model, tensor)
        timing["inference"] += time.time() - t2
        start_s, end_s = st / sr, min(n, st + win) / sr
        wid = f"w{i:04d}"
        per.append({"face_id": wid, "frame_idx": i, "face_index": 0,
                    "bbox": [round(start_s, 3), round(end_s, 3), 0, 0], "fake_prob": round(prob, 4),
                    "start_s": round(start_s, 3), "end_s": round(end_s, 3)})
        windows.append({"start_s": start_s, "end_s": end_s, "fake_prob": prob})

    t3 = time.time()
    save_audio_timeline(wave_, sr, windows, output_dir / "timeline.png", threshold=0.5)
    timing["viz"] += time.time() - t3
    return per, {"sample_rate": sr, "duration_seconds": round(duration, 3),
                 "window_seconds": win / sr, "hop_seconds": hop / sr, "windows": len(per)}
