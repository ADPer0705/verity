---
type: quickstart
title: Quickstart and documentation map
description: Entry point for forensic investigators, lab operators, and developers covering one-line standalone binary installation, air-gapped workstation provisioning, core workflows, and Verity's complete OpenWiki documentation map.
tags: [quickstart, install, airgap, cli, documentation]
verified:
  - by: openwiki/0.9.0
    at: 2026-11-21T06:14:40.994Z
sources:
  - id: openwiki-source-03ffc32a0ca502ab67c54b25
    resource: repo://install.sh
  - id: openwiki-source-e4dc7d4c95cc68eb2f4b4656
    resource: repo://src/verity/cli.py
  - id: openwiki-source-184a2f452d290bbb1fb99bd9
    resource: repo://src/verity/paths.py
generated: { by: "antigravity", at: "2026-11-24T11:23:46.564Z" }
---

# Verity quickstart

**Multi-detector deepfake examination for single-investigator workstations and
air-gapped forensic labs.** Verity characterizes images, videos, and audio files,
runs model-free forensic analyzers alongside 39 evidence-grade detectors (across
58 ported architectures) in isolated `uv` environments, preserves every finding and
visual exhibit separately, and seals case dossiers into checksum-verifiable evidence
bundles.

> Detector outputs are statistical indicators, not proof. Verity never collapses a
> panel into a single verdict — panel disagreement is itself a first-class finding.

## One-command workstation install (Linux x86_64)

No Docker, no system Python, and no pre-installed `uv` required:

```bash
# 1. Download, SHA-256 verify, and install ~/.local/bin/verity
curl -fsSL https://raw.githubusercontent.com/ADPer0705/verity/main/install.sh | bash

# 2. Provision everything once while online:
#    - self-bootstraps uv + managed standalone Pythons into ~/.local/share/verity
#    - builds all 39 isolated evidence-grade detector virtualenvs
#    - downloads & SHA-256 verifies checkpoints (ADPer/verity-weights + origins)
#    - warms Hugging Face / Torch Hub foundation caches
#    - verifies integrity and seals offline=true in ~/.config/verity/config.yaml
verity setup --trained --airgap-ready

# 3. Disconnect network permanently, audit readiness, and launch Web Console
verity doctor --verify
verity serve                 # Web Console at http://127.0.0.1:8000
```

Need to deploy onto a machine that **never** touches the internet? Build a USB
transfer kit on an online staging machine and import it offline:

```bash
# Staging workstation (online):
verity airgap export --out /mnt/usb/verity-kit

# Air-gapped workstation (offline):
/mnt/usb/verity-kit/verity airgap import --from /mnt/usb/verity-kit
verity doctor --verify && verity serve
```

Host libraries recommended for video/audio decoding and PDF export: `ffmpeg`
(auto-bootstrapped statically on Linux x86_64 if missing online) plus WeasyPrint's
Pango/Cairo packages (`libpango-1.0-0 libpangocairo-1.0-0 libcairo2`).

## Essential investigator commands

| Goal | Command |
|---|---|
| Launch Web Console & HTTP API | `verity serve [--host 127.0.0.1] [--port 8000] [--offline]` |
| Examine a file or directory via CLI | `verity run --media suspect.mp4 [--detector x,y] [--format pdf\|html]` |
| Dry-run capability & selection reasons | `verity analyze --media suspect.mp4 [--json]` |
| Run model-free forensics alone | `verity forensics --media-id <12-char-id>` |
| Export chain-of-custody zip bundle | `verity export --media-id <12-char-id> [--out evidence.zip]` |
| Audit readiness & checkpoint hashes | `verity doctor --verify` / `verity weights status --verify` |

In standalone binary or `$VERITY_HOME` mode (`src/verity/paths.py`), mutable data,
caches, and configuration resolve cleanly via `platformdirs` (`~/.local/share/verity`,
`~/.cache/verity`, `~/.config/verity`), whereas editable Git checkouts keep
repo-local `./workspace` and `./detectors` paths untouched.

## Complete OpenWiki documentation map

OpenWiki is the sole, repository-grounded documentation source for Verity:

### Core Architecture & Pipeline

| Page | Covers |
|---|---|
| [Architecture overview](architecture/overview.md) | Package boundaries, dual-mode `platformdirs` layout, why single binary + `uv`, core invariants |
| [Examination pipeline](architecture/pipeline.md) | End-to-end stage flow (`ingest` -> `extract` -> `forensics` -> `analyze` -> `select` -> `detect` -> `compile` -> `report` -> `export`) |
| [Ingest and media analysis](architecture/ingest-and-analysis.md) | Content-addressed SHA-256 IDs, `0o444` working copies, `ffprobe`, deterministic Haar face scan |
| [Artifact extraction](architecture/artifact-extraction.md) | `FrameExtractor`, `FaceExtractor`, `MetadataExtractor`, `AudioExtractor`, and detector input staging |
| [Model-free forensic analyses](architecture/forensics.md) | Five deterministic analyzers (`metadata`, `compression`, `noise`, `video_temporal`, `audio`), severity model & limits |
| [Detector selection and trust tiers](architecture/detector-selection.md) | Ordered eligibility rules, `trained` vs scaffold tiers, overrides, and manipulation-family coverage |
| [Detector runtime and isolation](architecture/detector-runtime.md) | `DetectorRegistry`, `DetectorRunner`, in-process `env_manager`, hardware profiles, offline sync |
| [Shared detector engine](architecture/detector-engine.md) | `detectors/common/` seeding, all 6 input modes, Grad-CAM vs native attention, `aux_aggregate`, RECCE decoder audit |
| [Findings, consensus and aggregation](architecture/findings-and-consensus.md) | Tier-gated `FindingsCompiler`, `consensus()` split bands, and UI probability spectrum |
| [Report generation](architecture/reporting.md) | Self-contained HTML/PDF dossiers, modality-aware exhibits, and why verdicts stay omitted |
| [Web API and investigator console](architecture/web-api-and-console.md) | FastAPI `JobManager`/SSE endpoints, frozen `uvicorn` serving, capability probes, and React/Mantine UX |
| [Weights and checkpoints](architecture/weights-and-checkpoints.md) | Manifest v2, SHA-256 verification, `ADPer/verity-weights` mirror attribution, origin-only licensing, warmup |
| [Security, integrity and chain of custody](architecture/security-and-integrity.md) | Working-copy verification, `MANIFEST.json` + `SHA256SUMS.txt` export bundles, confined file serving, loopback posture |

### Reference & Operations

| Page | Covers |
|---|---|
| [Detector catalog (all 58 ports)](reference/detector-catalog.md) | Field guide to all 39 evidence-grade & 19 scaffold architectures, caveats, and recommended panels |
| [Explainability artifacts and on-disk formats](reference/explainability-artifacts.md) | Every JSON schema, PNG overlay, forensic plot, and archive structure |
| [Deployment, configuration and air-gap operations](operations/deployment-and-airgap.md) | One-line `install.sh`, `setup --airgap-ready`, USB `airgap export/import`, Docker Compose Option B, env vars, troubleshooting |
| [Releasing, publishing and extending Verity](operations/releasing-and-publishing.md) | `ubuntu-22.04` glibc builds, Hugging Face & GHCR publishing, version bumps, tests, adding detectors |
