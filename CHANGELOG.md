# Changelog

All notable changes to Verity are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Planned
- Authentication and persisted job history for the API.
- More audio detectors and an audio-forensics temporal view.
- Optional signed evidence bundles (case export with hashes).

## [0.5.1] — 2026-10-21

### Fixed
- Release checksums are split correctly: `SHA256SUMS` covers the stable
  `verity-deploy.tar.gz` alias used by the quick start, and
  `SHA256SUMS-<version>` covers the pinned archive. Previously both names
  were listed in one file, so verifying only the downloaded alias reported a
  missing file.

## [0.5.0] — 2026-10-21

### Added
- **Lab deploy bundle** (`verity-deploy.tar.gz` release asset): compose files,
  `install.sh`, `doctor.sh`, systemd unit, deployment guide, license/notices
  and a `VERSION` marker.
- **One-command install** that auto-detects GPU/CPU, pulls the image,
  downloads and verifies every trained checkpoint, builds **all**
  evidence-grade detector environments ahead of time, starts the stack and
  waits for `/api/system` before running health checks.
- **`verity setup --concurrency`** to bound parallel environment builds.
- **Deployment guide** covering hardware, disk, weights, environments,
  exposure, upgrades, offline/air-gapped installs and troubleshooting.
- **Doctor script** for runtime, container, API readiness, checkpoints,
  environments, disk and GPU.

### Changed
- Compose binds to **127.0.0.1 by default** (the API has no authentication);
  LAN exposure requires `--allow-network` and an authenticating proxy.
- Compose adds `restart: unless-stopped`, an API healthcheck, and a
  `${VERITY_IMAGE}` selector; local builds moved to a separate overlay.
- Container logs now show checkpoint downloads instead of hiding them.
- Releases attach the deploy bundle and `SHA256SUMS` only.

### Removed
- Wheel/sdist release assets and the release-wheel script: they cannot run an
  examination and were not a usable deployment path.

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
