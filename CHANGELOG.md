# Changelog

All notable changes to Verity are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Planned
- Authentication and persisted job history for the API.
- More audio detectors and an audio-forensics temporal view.
- Optional signed evidence bundles (case export with hashes).

## [0.4.1] — 2026-10-21

### Fixed
- Compose files and the README quick start now use the **published slim
  image** (`ghcr.io/adper0705/verity:slim`) instead of an unpublished
  pre-baked `:cuda` tag, which caused `docker compose up` to attempt a
  multi-gigabyte local build. The full `:cuda`/`:cpu` targets remain available
  as documented manual builds.
- `latest` is documented as an alias for the newest slim image.

## [0.4.0] — 2026-10-21

First public release.

### Added
- **Deterministic pipeline** — ingest by SHA-256, frame/face/metadata/audio
  extraction, media analysis, rule-based detector selection, isolated
  detector execution, finding compilation and HTML/PDF reporting.
- **47 ported detector architectures** (26 evidence-grade) spanning image,
  video and audio deepfakes, including Xception/EfficientNet/MesoNet family,
  F3Net, SPSL, SRM, CORE, RECCE, LipForensics, UnivFD, GenD, DF40-CLIP,
  Effort, Community Forensics, NPR, MLEP, SAFE, AEROBLADE, I3D/FTCN/
  AltFreezing, TimeSformer, VideoMAE, X-CLIP, AASIST, AntiDeepfake-MMS and
  XLS-R+SLS.
- **Trust tiers** (`trained`, `backbone-only`, `random-init`, `untrained`)
  with tier-aware selection; untrained heads never contribute to consensus.
- **Five model-free forensic analyzers** — metadata/EXIF consistency,
  compression (ELA, quantization, double-JPEG, block grid), noise/spectrum,
  video temporal structure, and audio spectrum/noise-floor analysis.
- **Explainability artifacts** — face crops, frames, attention/Grad-CAM
  overlays, per-face region energy, per-frame/prediction timelines,
  RECCE reconstruction maps with decoder-liveness audit, raw detector logs.
- **Web console + HTTP API** — background jobs with streaming logs, case
  list/detail, detector evidence panels, consensus panel, system readiness,
  forensic observations, secure artifact serving.
- **Manifest-driven weights manager** — SHA-256 verification, archive
  extraction, shared-file hard-linking, offline mode, Hugging Face mirror
  bundling, per-file license tracking.
- **Per-detector uv environments** with hardware-adaptive CUDA/CPU profiles
  and pre-baked image support.
- **Container images** — slim (CI), plus documented full CUDA/CPU builds;
  compose overlays; air-gapped operation.
- **Tests and tooling** — unit suite, detector table and third-party notice
  generators, release builder, CI workflow (tests, UI build, slim image).
- **Documentation** — technical overview, detector catalog, explainability
  artifact catalog, forensics reference, web UI/API guide, weights system,
  publishing runbook, contributing/security/conduct guides.

### Known limitations
- No authentication; run behind a trusted proxy or locally.
- Jobs are in-memory; the workspace is the durable record.
- The exported report deliberately contains no aggregate verdict.
- Some ports ship without trained heads (documented per detector).
