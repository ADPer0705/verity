---
type: subsystem
title: Model-free forensic analyses
description: The five deterministic signal-processing and metadata analyzers (metadata, compression, noise, video temporal, audio), their severity model, fail-soft execution, boundaries, standalone API trigger, and Web UI inspection.
tags: [forensics, signal-processing, metadata, compression, noise, audio, explainability]
verified:
  - by: openwiki/0.9.0
    at: 2026-11-21T06:14:40.994Z
sources:
  - id: openwiki-source-50e32edfedf5ed35b14b1fcc
    resource: repo://src/verity_api/routers/jobs.py
  - id: openwiki-source-ba6ee0d18b6931b109031a39
    resource: repo://src/verity_ui/src/routes/tabs/ForensicsTab.tsx
  - id: openwiki-source-4efd8cf2f72de430f6b102ff
    resource: repo://src/verity/forensics/audio_analyzer.py
  - id: openwiki-source-bebc1a0fbb3d31b067b13577
    resource: repo://src/verity/forensics/base.py
  - id: openwiki-source-bea854b0e15c8c3c1751b82a
    resource: repo://src/verity/forensics/compression_analyzer.py
  - id: openwiki-source-d114e9ad1af3fb8fa6ba0883
    resource: repo://src/verity/forensics/metadata_analyzer.py
  - id: openwiki-source-8b113eaf5d27f5888b560dec
    resource: repo://src/verity/forensics/models.py
  - id: openwiki-source-7cd5967330964f010516891b
    resource: repo://src/verity/forensics/noise_analyzer.py
  - id: openwiki-source-ee6ff565ef667b397d40fa2a
    resource: repo://src/verity/forensics/runner.py
  - id: openwiki-source-dc09948f63078fa94e25ec39
    resource: repo://src/verity/forensics/video_temporal_analyzer.py
generated: { by: "antigravity", at: "2026-11-21T06:14:40.994Z" }
---

# Model-free forensic analyses

Verity's forensic analyzers run alongside the machine-learning detectors to
provide classic, model-independent measurements that an examiner can defend in
review without referring to neural network weights. Crucially, they **never emit
a real/fake verdict**; they emit explainable **observations** — a plain-language
statement backed by structured numerical evidence — tagged with a severity level
that prioritizes review attention.

Why run them before neural inference (`verity run` and `POST /api/run`)? Because
they are fast, CPU-only, deterministic, and occasionally decisive on their own
(for example, an EXIF modification date earlier than capture date or an explicit
editor/AI-generator encoder tag can settle provenance before GPU models start).
They are also runnable independently via `verity forensics --media-id <id>`,
`POST /api/forensics`, or directly from the dedicated **Forensics** tab in the
Web Console.

## Observation and severity model

`src/verity/forensics/models.py` defines the structured schema:

| Type | Fields & Purpose |
|---|---|
| `Observation` | `analyzer`, `severity` (`info` \| `notice` \| `warning`), `statement`, `evidence` dict |
| `ForensicResult` | `analyzer`, `label`, `method` (plain-language scope & sensitivity note rendered verbatim in reports), `ok`, `observations`, `artifacts`, `data` |
| `ForensicsReport` | `media_id`, `media_type`, `results`, `observations` (sorted warning -> notice -> info, then analyzer & statement), `counts` |

Severity guides triage rather than judging authenticity:

- **`info`** — neutral fact worth documenting (e.g., no audio track, standard pixel format).
- **`notice`** — deviates from an untouched camera capture, though benign delivery/transcoding explanations remain plausible.
- **`warning`** — internal contradiction (such as camera-claiming EXIF missing Make/Model or modify timestamp preceding capture) hard to explain without editing or reprocessing.

## Fail-soft execution & bounded determinism

`ForensicsRunner` (`src/verity/forensics/runner.py`) runs the five analyzers in
pipeline order:

`metadata -> compression -> noise -> video_temporal -> audio`

Each analyzer runs inside an explicit `try/except`: a failure in one analyzer
records `ok=False` and emits a `notice` observation (`analyzer X failed: ...`),
allowing the remaining analyzers and the rest of the examination to complete.

`src/verity/forensics/base.py` enforces deterministic, bounded work:

- `sample_video_frames` reads sequentially via `cv2.VideoCapture` rather than
  random seeking so decoder keyframe behavior cannot shift frame selection.
- `bounded` downscales pixel-level frames to at most `MAX_SIDE = 2048` px.
- `write_png` strips ancillary PNG metadata for byte-identical artifacts; `write_json`
  sorts keys and rounds floats via `r()` (`4` decimal places).

Results persist under `workspace/artifacts/<media_id>/forensics/forensics.json`,
with per-analyzer JSON and PNG figures alongside it.

## The five analyzers, methods, and honest limits

### 1. `metadata` (`image`, `video`, `audio`)
Checks container/stream tags via `ffprobe` and EXIF headers via Pillow:

- Scans `encoder`/`software` tags against 60+ signatures (`ffmpeg`, `x264`/`x265`,
  `HandBrake`, `Adobe`, `Final Cut`, `CapCut`, `OBS`, generative tools, social
  platforms).
- Flags non-4:2:0 pixel formats (`yuv444p`, unusual profiles) and container/codec mismatches.
- Inspects GOP picture types (`I`/`P`/`B`) for all-intra or edit-aligned keyframes.
- Cross-checks EXIF Make/Model/DateTimeOriginal/ModifyDate (warning when a
  camera-claiming file lacks Make/Model or modifies before capture).
- **Limit**: metadata is trivially stripped or forged; absence proves nothing, and
  warnings reflect processing history rather than malicious intent.
- **Artifacts**: `metadata/metadata.json`, `metadata/observations.json`.

### 2. `compression` (`image`, `video`)
Examines JPEG quantization and 8x8 DCT structure (or the first 5 decoded video
frames as indicative samples):

- **Error Level Analysis (ELA)**: re-saves at fixed JPEG quality 90 and visualizes
  absolute error (`ela.png`, `ela_overlay.png`). Flat vs textured areas naturally
  respond differently — ELA is a localization aid, never an isolated score.
- **Quantization tables**: estimates closest IJG quality and L1 distance, noting
  standard (`libjpeg`) vs custom encoder tables.
- **Double-JPEG periodicity**: computes FFT peak-to-background ratio on de-trended
  low-frequency AC DCT coefficient histograms (coarse tables with too few bins are
  honestly reported as non-measurable rather than zero).
- **8x8 blockiness (`blockiness.png`)**: compares inter-block vs intra-block
  gradients per 64 px cell to surface splicing grid mismatches.
- **Limit**: full-image re-saving double-compresses everything; heavy social-media
  re-encoding erases weak traces.

### 3. `noise` (`image`, `video`)
Inspects pixel noise residuals and spatial frequency spectra:

- **Local noise residual (`noise_residual.png`, `noise_variance.png`)**: computes
  `image - median_blur(3x3)` and measures RMS over non-edge pixels per 48 px cell
  (Sobel-masked to avoid confusing sharp edges with noise), outlining cells with
  robust `|z| > 3.5`.
- **2D Fourier spectrum (`spectrum.png`)**: examines windowed log-magnitude DFT
  (up to 1024 px) for isolated periodic grid peaks indicative of upsampling/decoders,
  while noting codec 1/8-cycle peaks as benign confounders.
- **Limit**: textured regions (hair/foliage) raise residuals natively, and modern
  computational-photography denoising flattens sensor noise uniformly.

### 4. `video_temporal` (`video`)
Decodes up to 1,500 sequential frames at 320 px width and computes frame-to-frame
mean absolute difference, 64-bit DCT perceptual hash, and mean luminance:

- **Scene-cut candidates**: outlying difference spikes (robust `z >= 6.0` and
  energy `>= 12`), cross-referenced against metadata keyframes.
- **Frozen / duplicated runs**: consecutive identical perceptual hashes with
  near-zero difference (dropped frames, frame-rate conversion, held edit frames).
- **Luminance flicker**: mean second difference of frame luminance.
- **Face-size jitter**: computed from extracted `faces/` sizes (noting explicitly
  that box coordinates are not stored post-crop so positional jitter is omitted).
- **Artifacts**: `video_temporal/temporal.json`, `video_temporal/timeline.png`.

### 5. `audio` (`video`, `audio`)
Decodes the **original** audio stream at its native sample rate (not the pipeline's
16 kHz detector WAV, which would destroy high-band evidence above 8 kHz), up to 180 s:

- Computes long-term high-band (`> 8 kHz`) vs low-band energy ratio (flagging
  band-limited speech below `-45 dB`), 25 ms RMS silence/noise-floor (`-50 dBFS`),
  DC offset, clipping headroom, and renders a 64-band log-mel `spectrogram.png`.
- **Limit**: telephony, Bluetooth, and low-bitrate AAC compression also band-limit
  audio innocently; studio booths naturally exhibit clean silence floors.
- **Artifacts**: `audio/audio.json`, `audio/audio_native.wav`, `audio/spectrogram.png`.

## Surfacing in API, UI, and reports

- **Standalone trigger**: `POST /api/forensics` launches forensics on demand for an
  ingested case without re-running neural detectors.
- **Web Console (`ForensicsTab.tsx`)**: dedicated **Forensics** tab on Case Detail
  with severity filters (`Warning` / `Notice` / `Info`), itemized evidence badges,
  method caveats, full-resolution diagnostic gallery, and a one-click re-run button.
- **HTML / PDF Reports**: `ReportGenerator._forensics_html` renders severity counts,
  every analyzer's `method` string verbatim, sorted observation tables, and wide/tile
  diagnostic figures cleanly ahead of the ML detector findings.

## Related pages

- [Examination pipeline](pipeline.md) — stage order and storage layout.
- [Report generation](reporting.md) — how forensic observations and PNG figures embed in dossiers.
- [Web API and investigator console](web-api-and-console.md) — `POST /api/forensics` and `ForensicsTab`.
- [Explainability artifacts and on-disk formats](../reference/explainability-artifacts.md) — complete file catalog.
