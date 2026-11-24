---
type: subsystem
title: Artifact extraction
description: Deterministic extraction of frames, face crops, container metadata, and 16 kHz mono audio into workspace/artifacts/<media_id>/, plus fail-soft isolation and temporary input staging for detectors.
tags: [artifacts, extraction, ffmpeg, opencv, pipeline]
verified:
  - by: openwiki/0.9.0
    at: 2026-11-24T11:23:46.564Z
sources:
  - id: openwiki-source-61ad6b16674ac0599a5a1e59
    resource: repo://src/verity/detect/detector_runner.py
  - id: openwiki-source-3c00bfddd4822d4f366b5696
    resource: repo://src/verity/extract/audio_extractor.py
  - id: openwiki-source-30123873926de405285e5c91
    resource: repo://src/verity/extract/base_extractor.py
  - id: openwiki-source-5cfca22ef6495938f266c846
    resource: repo://src/verity/extract/face_extractor.py
  - id: openwiki-source-d8222e6d42baef1c82c31108
    resource: repo://src/verity/extract/frame_extractor.py
  - id: openwiki-source-a921a2bb3cc695a37bd11609
    resource: repo://src/verity/extract/metadata_extractor.py
generated: { by: "antigravity", at: "2026-11-24T11:23:46.564Z" }
---

# Artifact extraction

Artifact extraction runs immediately after ingest and before model-free
forensics, analysis, and detector selection (`verity extract`, `verity run`, and
`POST /api/extract`). Implemented under `src/verity/extract/`, it converts raw
media into normalized intermediate exhibits without making any authenticity claims.

## The `BaseExtractor` & `Artifact` contract

Every extractor inherits `BaseExtractor` (`src/verity/extract/base_extractor.py`),
initializes `workspace/artifacts/`, and implements
`extract(media_id: str) -> ArtifactCollection`. `src/verity/models/artifacts.py`
defines `ArtifactType` (`frame`, `face_crop`, `audio_clip`, `metadata`, `landmarks`,
`frequency_spectrum`), `Artifact`, and `ArtifactCollection`.

Extractors distinguish missing prerequisites from absent content: missing ingested
media raises `FileNotFoundError`, whereas benign mismatches (images passed to
frame/audio extraction, or silent videos without soundtracks) return an empty
`ArtifactCollection` cleanly.

## Shipped extractors in `config/pipeline.yaml` order

`config/pipeline.yaml` runs `frame`, `face`, `metadata`, and `audio` in fixed sequence:

1. **`FrameExtractor` (`frame_extractor.py`)**: Confirms video MIME via `filetype`
   and stream presence via `ffprobe`, then extracts PNG frames via
   `ffmpeg -i <media> -vf fps=<fps> -frame_pts 1 frame_%05d.png -y` (default 1 fps)
   under `artifacts/<id>/frames/`.
2. **`FaceExtractor` (`face_extractor.py`)**: Iterates sorted `frame_*.png` images
   with OpenCV's `haarcascade_frontalface_default.xml` (`scaleFactor=1.1`,
   `minNeighbors=5`, `minSize=(50, 50)`), clamps boxes, rejects `< 20x20` px boxes,
   and writes `artifacts/<id>/faces/<frame_stem>_face_<i>.png` with bounding-box
   params. Because it reads `frames/`, it naturally depends on `FrameExtractor`
   running first.
3. **`MetadataExtractor` (`metadata_extractor.py`)**: Records file name/size and
   captures `ffprobe -show_format -show_streams` output into
   `artifacts/<id>/metadata/metadata.json`, recording `ffprobe: null` gracefully if
   `ffprobe` is absent.
4. **`AudioExtractor` (`audio_extractor.py`)**: Exports normalized 16 kHz mono
   16-bit PCM WAV via `ffmpeg -vn -acodec pcm_s16le -ar 16000 -ac 1 audio.wav`. If
   no audio stream exists or `ffmpeg` fails, returns an empty `ArtifactCollection`
   so audio-dependent detectors are excluded cleanly by `DetectorSelector` rather
   than failing the case. *(Note: the model-free `audio` forensic analyzer decodes
   at native sample rate separately so high-frequency `> 8 kHz` spectral traces are
   not erased by 16 kHz resampling.)*

## Fail-soft API behavior & isolated detector staging

- **Fail-soft isolation**: `verity_api.service.run_extract` wraps each extractor in
  its own `try/except`, logging failures under `failed_extractors` so losing one
  optional extractor never aborts the entire case.
- **Staged detector inputs**: When `DetectorRunner.run` launches a detector,
  `_prepare_input_dir` (`src/verity/detect/detector_runner.py`) creates a fresh
  temporary directory containing a copy of the media file (`media/`) and
  `artifacts/<id>/` (`artifacts/`). Detectors never hold direct write paths to
  workspace evidence.

## Related pages

- [Examination pipeline](pipeline.md) — full stage order across CLI & API.
- [Ingest and media analysis](ingest-and-analysis.md) — SHA-256 admission and `ffprobe`/Haar scan.
- [Shared detector engine](detector-engine.md) — how staged audio and frames are consumed.
- [Detector runtime and isolation](detector-runtime.md) — `DetectorRunner` staging and cleanup.
