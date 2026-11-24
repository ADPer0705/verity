---
type: reference
title: Explainability artifacts and on-disk formats
description: Complete catalog of every JSON record, visual explainability image, forensic plot, HTML/PDF report, and checksum-sealed chain-of-custody export bundle written during a Verity examination.
tags: [explainability, artifacts, schemas, chain-of-custody, forensics, reports]
verified:
  - by: openwiki/0.9.0
    at: 2026-11-21T06:14:40.994Z
sources:
  - id: openwiki-source-2fbc661d38d1c9f143c7b320
    resource: repo://detectors/common/engine.py
  - id: openwiki-source-c5cd8824ffe1e5595fda06ac
    resource: repo://src/verity/detect/output_parser.py
  - id: openwiki-source-6d817ce39f4036ea92df24bc
    resource: repo://src/verity/export.py
  - id: openwiki-source-8b113eaf5d27f5888b560dec
    resource: repo://src/verity/forensics/models.py
  - id: openwiki-source-479291a034ea1b3214af9e14
    resource: repo://src/verity/report/report_generator.py
generated: { by: "antigravity", at: "2026-11-21T06:14:40.994Z" }
---

# Explainability artifacts and on-disk formats

Every output in Verity is built to be **supervisor-reviewable**: no detector
result is ever stored as a bare scalar score detached from what the model
actually saw. This reference catalogs every on-disk file across the five stages
of an examination, why each exists, and how the UI and PDF dossier surface them.

All workspace paths live under `$VERITY_WORKSPACE` (`~/.local/share/verity/workspace`
in standalone binary mode; `./workspace` in repository dev) and `$VERITY_OUTPUT_DIR`
(defaulting to `<workspace>/output`). Case identifiers (`media_id`) equal the
first 12 hex characters of the file's SHA-256 hash.

## Artifact overview & custody design invariants

| Category | Stage / Owner | Primary Files | Evidence Role |
|---|---|---|---|
| **Case record** | Ingest, Extract, Analyze, Select, Compile | `.{media_id}.json`, `analysis.json`, `selection.json`, `finding.json` | Bytes-to-ID link, eligibility proof, and raw-to-finding provenance |
| **Forensic exhibits** | `ForensicsRunner` (5 analyzers) | `forensics.json`, `ela.png`, `ela_overlay.png`, `blockiness.png`, `noise_*.png`, `spectrum.png`, `timeline.png`, `spectrogram.png` | Model-free signal & metadata measurements |
| **Detector exhibits** | `detectors/common/engine.py` + `output_parser.py` | `result.json`, `faces/`, `frames/`, `full_frames/`, `attention_maps/`, `reconstructions/`, `timeline.png`, logs | Exact cropped/full inputs, spatial attribution, auxiliary sub-scores, verbatim logs |
| **Compiled deliverables** | `ReportGenerator` | `report.html`, `report.pdf` | Portable self-contained exhibit (base64 images, zero aggregate verdict) |
| **Preservation bundle** | `verity.export` | `verity-evidence-<stem>-<id>.zip` (`MANIFEST.json`, `SHA256SUMS.txt`) | Standard `sha256sum -c` verifiable archive for court/lab custody |

Four invariants govern this layout:

1. **Determinism**: Media IDs derive from SHA-256; frame readers iterate
   sequentially; PNG compression strips timestamps; galleries sort by descending
   probability with filename tie-breakers.
2. **Traceability**: Every prose sentence in `Finding.reasoning` traces back to
   raw keys inside `result.json` or `forensics.json`.
3. **Examiner-first inspection**: Images (cropped inputs, overlays, bounding boxes,
   spectrograms) accompany every table so reviewers verify *what the detector saw*
   instead of trusting numbers blindly.
4. **Honest absence**: Missing Grad-CAMs, dead decoders (`recce`), or missing audio
   tracks are explicitly documented rather than fabricated or silently hidden.

---

## 1. Case record files

- **`workspace/media/<media_id>.<ext>` (`0o444`) & `.{media_id}.json`**: Immutable
  working copy and `MediaInfo` sidecar (`original_path`, `media_type`, `checksum`,
  `size_bytes`, `mime_type`, `original_name`, `ingested_at`).
- **`<output_dir>/<media_id>/analysis.json`**: Deterministic `MediaAnalysis`
  (`ffprobe` stream characteristics plus OpenCV Haar face scan over up to 30
  evenly sampled frames).
- **`<output_dir>/<media_id>/selection.json`**: Sorted `SelectionDecision` list
  for all 58 registered detectors, proving which ran and why each excluded
  detector failed rule 0..5.
- **`workspace/output/<media_id>_<detector_id>/finding.json`**: Parsed `Finding`
  preserving `classification`, `confidence`, `tier`, `reasoning`, normalized
  `raw_output`, and discovered `evidence` paths.

---

## 2. Model-free forensic artifacts (`artifacts/<media_id>/forensics/`)

Indexed by `forensics.json` (`ForensicsReport`, sorted by severity `warning` ->
`notice` -> `info` via `src/verity/forensics/models.py`):

| Analyzer | Artifacts Written |
|---|---|
| `metadata` | `metadata/metadata.json`, `metadata/observations.json` |
| `compression` | `compression/ela.png`, `compression/ela_overlay.png`, `compression/blockiness.png`, `compression/compression.json` |
| `noise` | `noise/noise_residual.png`, `noise/noise_variance.png`, `noise/spectrum.png`, `noise/noise.json` |
| `video_temporal` | `video_temporal/temporal.json`, `video_temporal/timeline.png` |
| `audio` | `audio/audio.json`, `audio/audio_native.wav`, `audio/spectrogram.png` |

---

## 3. Detector artifacts (`workspace/output/<media_id>_<detector_id>/`)

Produced by `detectors/common/engine.py` and collected automatically by
`verity.detect.output_parser.discover_evidence_paths`:

| Artifact | Producer Modes | Evidence Value |
|---|---|---|
| `result.json` & `per_face_results.json` | All modes | Unmodified `prediction`, `confidence`, `stats`, `unit`, `input_mode`, `aux_aggregate`, `region_aggregate`, and `per_face` unit records |
| `faces/<face_id>.png` | `face_crop`, `face_clip`, `av_clip` | Exact MTCNN face crop fed to the network (exposes detector false-positives or bad framing) |
| `frames/<id>.png` | `full_frame`, `full_clip`, `av_clip` | Thumbnails of whole frames / windows scored directly without face cropping |
| `full_frames/<face_id>_frame.png` | Video face detectors (top-3 suspicious) | Full uncropped frame with bounding box overlay (shows background/scene context in PDF & UI) |
| `attention_maps/<id>_attention.png` | Native attention or `cam: true` backbones | 3-panel figure (Original -> Heatmap -> Overlay) highlighting spatial attribution |
| `reconstructions/<id>_recon.png` | RECCE (only when `decoder_alive`) | Original vs reconstruction vs `|Delta|` residual (suppressed cleanly on dead checkpoints) |
| `timeline.png` | Video, audio, `av_clip` | Frame probability mean/range plot, or waveform/log-mel spectrogram overlaid with window fake scores |
| `stdout.log`, `stderr.log` | `DetectorRunner` (last 20,000 chars each) | Verbatim subprocess execution audit trail |

### Region-energy attribution & native attention vs Grad-CAM

- When `region_analysis: true` and MediaPipe FaceMesh locates landmarks on a face
  crop, normalized attention energy maps onto six convex-hull facial regions
  (`left_eye`, `right_eye`, `nose`, `mouth`, `jaw`, `forehead`) in `region_energy`
  and `region_aggregate`.
- `_cam_for` prefers native `attention_map` tensors (LAA-Net boundary map,
  Forensics Adapter / VLAForge / DF40 ViT patch attention, SRM spatial attention,
  RECCE guided attention) and falls back to Grad-CAM only when `spec.cam` is `True`.
- Models whose physics/input make spatial Grad-CAM meaningless (`aeroblade`, `univfd`,
  `safe`, `timesformer`, `videomae`, `xclip`, `tall`, audio detectors) set
  `cam: false` and emit `aux_aggregate` / timelines instead.

### Modality-aware presentation (`FindingTables` & `ReportGenerator`)

Instead of forcing audio and whole-image models into face-bounding-box columns:

- When `unit == "window"` (`audio`, `av_clip`), tables render **`Window ID`** +
  **`Time Span (s)`** (`start_s`–`end_s`) + telemetry/info columns (`av_offset_seconds`,
  `confidence`, `region_weight_max`).
- When `unit` is `"frame"` or `"clip"`, tables render **`Dimensions (w × h)`**
  alongside `Frame`.
- When `aux_aggregate` is non-empty (ProDet branches, VLAForge dual heads,
  AEROBLADE SD1.1/SD2.1/Kandinsky LPIPS distances, SyncNet measurements), both
  `ReportGenerator._key_figures_html` and UI `AuxSignalsTable` render each metric.

---

## 4. Exported reports & Preservation Bundles

### Self-contained reports (`<output_dir>/<media_id>/report.{html,pdf}`)

Embeds top-3 `full_frames`, top-6 `faces` (or top-6 `frames` when no face crops
exist), top-4 `attention_maps`, top-4 `reconstructions`, collapsible full sets,
forensics galleries, and `AuxSignalsTable` / `Per-Window` / `Per-Face` tables —
with zero aggregate verdict.

### Evidence Preservation Bundle (`verity-evidence-<stem>-<media_id>.zip`)

Produced by `export_evidence_bundle` (`src/verity/export.py`) via
`verity export --media-id <id>` or `GET /api/media/{id}/export`:

```text
verity-evidence-<stem>-<media_id>.zip
├── MANIFEST.json          # verity-evidence-bundle/1 metadata, checksums, findings_summary
├── SHA256SUMS.txt         # sorted sha256sum-compatible checksums for EVERY file + MANIFEST.json
├── media/                 # <original_name> (0o444 bytes) + ingest_record.json
├── report/                # report.html, report.pdf, analysis.json, selection.json
├── forensics/             # forensics.json + per-analyzer JSON and PNG figures
└── detectors/<det_id>/    # result.json, finding.json, logs, faces, frames, full_frames, attention_maps, timeline.png
```

Running `sha256sum -c SHA256SUMS.txt` inside the unpacked bundle verifies every
retained byte independently of Verity.

## Related pages

- [Shared detector engine](../architecture/detector-engine.md) — how engine artifacts are emitted.
- [Model-free forensic analyses](../architecture/forensics.md) — measurement methods and figure interpretations.
- [Report generation](../architecture/reporting.md) — HTML/PDF rendering and determinism guarantees.
- [Security, integrity and chain of custody](../architecture/security-and-integrity.md) — bundle verification and confinement rules.
