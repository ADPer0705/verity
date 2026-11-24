---
type: subsystem
title: Ingest and media analysis
description: Content-addressed media admission under SHA-256 IDs with read-only working copies, plus deterministic ffprobe and sequential Haar face-presence characterization driving selector decisions.
tags: [ingest, media, checksum, analysis, ffprobe, opencv]
verified:
  - by: openwiki/0.9.0
    at: 2026-11-24T11:23:46.564Z
sources:
  - id: openwiki-source-f9eed95309c4e6211dbd78a2
    resource: repo://src/verity/analyze/media_analyzer.py
  - id: openwiki-source-c914ce3e5aa30a558c56c3cb
    resource: repo://src/verity/ingest/media_ingestor.py
generated: { by: "antigravity", at: "2026-11-24T11:23:46.564Z" }
---

# Ingest and media analysis

Every Verity case begins with two cheap, foundational stages before any neural
detector spins up: admitting the file immutably (`MediaIngestor`) and measuring
its intrinsic technical characteristics (`MediaAnalyzer`). Together, they establish
byte identity and feed `DetectorSelector` with the facts needed to exclude
unsuitable detectors upfront.

## Ingest: content-addressed admission (`MediaIngestor`)

`src/verity/ingest/media_ingestor.py` admits media files deterministically:

1. Resolves the source path (`FileNotFoundError` if missing) and streams bytes in
   64 KiB chunks to compute a full **SHA-256** digest.
2. Assigns `media_id = checksum[:12]` (`MEDIA_ID_LENGTH = 12`).
3. Classifies container MIME type using `filetype` into `MediaType` (`image`,
   `video`, `audio`, or `unknown`).
4. Copies the file once into `workspace/media/<media_id><suffix>` (skipping if
   already present) and locks permissions to **read-only (`chmod 0o444`)**.
5. Persists a `MediaInfo` sidecar at `workspace/media/.<media_id>.json` recording
   `original_path`, `media_type`, full `checksum`, `size_bytes`, `mime_type`,
   `original_name`, and UTC `ingested_at`.

Because the ID derives strictly from file bytes, re-ingesting an identical exhibit
is idempotent, while any single-byte edit creates a separate case instead of
silently mutating an existing case record.

## Analysis: deterministic characterization (`MediaAnalyzer`)

`src/verity/analyze/media_analyzer.py` inspects the read-only working copy without
modifying any workspace state:

- **Container & stream characterization (`_ffprobe`)**: invokes `ffprobe` for
  `video_codec`, `fps`, `width`, `height`, `frame_count` (or duration * fps
  estimate), `duration_seconds`, and `has_audio_track`. If `ffprobe` is absent or
  fails, `MediaAnalyzer` keeps filename-derived defaults and continues smoothly.
- **Deterministic Haar face scan (`_scan_faces`)**: initializes OpenCV's
  `haarcascade_frontalface_default.xml` (bundled cleanly via `cv2.data.haarcascades`
  even inside PyInstaller binaries) with `scaleFactor=1.1`, `minNeighbors=5`,
  `minSize=(40, 40)`.
  - For images, inspects the single frame.
  - For videos, decodes sequentially via `cv2.VideoCapture` reading frames at
    `interval = max(1, total // 30)` (`MAX_SAMPLED_FRAMES = 30`). **Why sequential
    reading instead of keyframe seeking?** Seeking lands on decoder-dependent
    keyframes across codecs/ffmpeg builds, whereas sequential modulus stride always
    inspects identical frame indices.
- Computes `face_present`, `face_count`, `max_face_size`, and `sampled_frames`,
  saving `<output_dir>/<media_id>/analysis.json`.

## Why analysis is separate from extraction

`verity analyze` (and `POST /api/analyze`) runs cleanly without heavy extraction:
an examiner can characterize media and inspect all 58 `selection.json` inclusion
and exclusion reasons even on a lightweight laptop or before provisioning `ffmpeg`
or detector environments. Keeping characterization cheap ensures expensive
subprocess runs only launch after eligibility is verified.

## Related pages

- [Artifact extraction](artifact-extraction.md) — extracting full PNG frames, crops, and audio.
- [Detector selection and trust tiers](detector-selection.md) — how `MediaAnalysis` drives selection rules.
- [Security, integrity and chain of custody](security-and-integrity.md) — `verify_integrity` and `verity export`.
