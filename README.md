# Verity

**Multi-detector deepfake examination for investigators, students and researchers.**

[![CI](https://github.com/ADPer0705/verity/actions/workflows/ci.yml/badge.svg)](https://github.com/ADPer0705/verity/actions/workflows/ci.yml)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Container](https://img.shields.io/badge/container-ghcr.io%2Fadper0705%2Fverity-blue)](https://github.com/ADPer0705/verity/pkgs/container/verity)
[![Weights](https://img.shields.io/badge/weights-Hugging%20Face-yellow)](https://huggingface.co/ADPer/verity-weights)

Verity takes any image, video or audio file, characterises it, picks the
detectors that can actually consume it, runs each one in its own isolated,
version-pinned environment, and writes a deterministic report in which every
detector's verdict, confidence, reasoning and explainability artifacts (faces,
attention maps, timelines) are preserved separately. **It never collapses the
panel into a single real/fake score — disagreement between methods is the
finding.**

It ships as one container with a web console, an HTTP API and a CLI, plus a
checksummed, license-tracked set of pretrained checkpoints. Model-free
forensic analyzers (metadata, compression, noise, temporal, audio) run
alongside the detectors and report observations, never verdicts.

> Detector outputs are statistical indicators, not proof. They must be read
> by a qualified examiner together with the artifacts Verity produces.

## Contents

- [Quick start](#quick-start)
- [What it examines](#what-it-examines)
- [How it works](#how-it-works)
- [Detectors and trust tiers](#detectors-and-trust-tiers)
- [Forensics and explainability](#forensics-and-explainability)
- [CLI](#cli)
- [HTTP API and web console](#http-api-and-web-console)
- [Checkpoints](#checkpoints)
- [Native install](#native-install-development)
- [Containers](#containers)
- [Project layout](#project-layout)
- [Documentation](#documentation)
- [Development](#development)
- [Limitations and ethics](#limitations-and-ethics)
- [License and citation](#license-and-citation)

## Quick start

Prerequisites: Docker (or Podman) and, for GPU hosts, the NVIDIA Container
Toolkit. A GPU with 8 GB or more is recommended; everything also runs on CPU.

```bash
mkdir verity && cd verity
curl -LO https://raw.githubusercontent.com/ADPer0705/verity/main/docker-compose.yml
docker compose up -d           # pulls ghcr.io/adper0705/verity:cuda, http://localhost:8000
```

Open <http://localhost:8000>, drop a photo, video or audio file, and choose
which detectors run (the recommended set, or exactly the ones you pick). Each
case opens with a plain-language summary and the full examination record
underneath. The first start fetches the checkpoints into the `verity-weights`
volume.

One-off, no web console:

```bash
docker compose run --rm verity run --media /input/suspect.mp4 --format pdf
# -> ./output/<media_id>/report.pdf, analysis.json, selection.json
```

CPU-only host:

```bash
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d
```

## What it examines

| Input | Analyzed by |
|---|---|
| Images (JPEG/PNG/WebP/TIFF/…) | whole-image and face-crop detectors, metadata/EXIF, compression, noise/spectrum |
| Videos (MP4/MOV/MKV/WebM/…) | face-crop, face-clip/temporal and whole-frame detectors, all image forensics on sampled frames, temporal structure, audio track |
| Audio (WAV/MP3/M4A/FLAC/…) | voice-clone detectors, audio spectrum/noise-floor/spectrogram forensics |

Manipulation families tracked: face swap, face reenactment, lip sync,
synthetic face, AI-generated image, local image editing, AI-generated video,
synthetic/cloned voice (`src/verity/models/taxonomy.py`).

## How it works

```
media ─► ingest ─► extract ─► analyze ─► select ─► detect ─► compile ─► report
         sha256    frames,    ffprobe +  per-      one uv     sorted     HTML / PDF
         copy      faces,     face scan  detector  env each,  findings   + UI / API
                   metadata,             rules     tier-aware
                   audio
                   └─ model-free forensics (metadata, compression, noise, temporal, audio)
```

- **Deterministic.** Media ids are SHA-256 checksums, inference uses fixed
  seeds, every list is explicitly sorted: the same media and detector set
  always yield the same report (the only wall-clock value is the report's
  generation timestamp).
- **Isolated.** Each detector is its own uv project
  (`detectors/<id>/pyproject.toml`) with exact pins. Incompatible torch
  versions coexist; a broken detector cannot poison another.
- **Explained.** The selector records why each detector was included or
  excluded; each finding carries its reasoning, confidence factors, faces,
  attention maps and a prediction timeline. Forensics carries its numbers and
  figures.
- **No single score.** Per-detector verdicts stay separate; the UI shows a
  consensus panel with the agreement bar and downgrades weak agreement to
  `inconclusive`. The exported report renders no aggregated verdict at all.

Details: [technical overview](docs/technical-overview.md) ·
[architecture of the UI/API](docs/web-ui.md).

## Detectors and trust tiers

47 detector architectures are ported. **26 are evidence-grade** — released
checkpoints with trained classification heads (or training-free methods) —
and are the only ones auto-selected:

| Tier | Meaning | Default |
|---|---|---|
| `trained` | released deepfake-trained checkpoint, or a training-free method | selected |
| `backbone-only` | pretrained backbone (ImageNet, Kinetics) with an untrained head | excluded |
| `random-init` | no released checkpoint at all; weights are random | excluded |
| `untrained` | foundation model fetched at first run, detector head untrained | excluded |

The rest can be run with `--include-untrained` (or by naming them explicitly).
Their findings are flagged in the report and the UI, and they never count
toward consensus or the aggregate. They exist for researchers who want to plug
in their own checkpoint or study an architecture.

Coverage includes Xception, EfficientNet-B4, MesoNet, F3Net, SPSL, SRM,
RECCE, LipForensics, CLIP-based detectors (UnivFD, GenD, DF40, Effort),
Community Forensics, NPR, MLEP, SAFE, AEROBLADE, I3D/FTCN/AltFreezing,
TimeSformer, VideoMAE, X-CLIP, and audio anti-spoofing (AASIST,
AntiDeepfake-MMS, XLS-R+SLS).

- Machine-generated table: [docs/detectors.md](docs/detectors.md)
- Per-detector field guide, artifacts, caveats:
  [docs/detector-catalog.md](docs/detector-catalog.md)
- `verity weights status` prints the full table with licenses and provenance.

## Forensics and explainability

Verity's **five model-free analyzers** produce observations with evidence
numbers, never verdicts:

| Analyzer | Checks |
|---|---|
| `metadata` | container/encoder tags, pixel format, GOP structure, EXIF consistency |
| `compression` | error-level analysis, JPEG quantization tables, double-JPEG periodicity, block grid |
| `noise` | sensor-noise uniformity per cell, Fourier spectrum peaks |
| `video_temporal` | scene cuts, frozen/duplicated frames, flicker, face-size jitter |
| `audio` | high-band energy, noise floor, DC/clipping, mel spectrogram |

Every detector additionally writes explainability artifacts: the exact face
crops or frames it scored, attention/Grad-CAM overlays, per-face region
energy, per-frame timelines, and — for RECCE — reconstruction maps when the
decoder is alive (the port audits liveness and honestly omits dead-channel
claims). Audio detectors produce per-window probability timelines over a
spectrogram.

- [Forensics reference](docs/forensics.md)
- [Explainability artifact catalog](docs/explainability-artifacts.md)

## CLI

```bash
verity run      --media <file|dir> [--detector a,b,c] [--include-untrained] [--format pdf|html]
verity analyze  --media <file|dir> [--json]      # characteristics + why each detector was (not) selected
verity setup    --trained | --all | --detector <id>[,<id>]   # provision detector environments
verity weights  pull --trained | status [--verify] | verify | bundle --repo <hf-repo>
verity ingest | extract | compile | report       # individual stages
verity-server                                    # API + web console
```

All commands read `config/pipeline.yaml` and the `VERITY_*` environment
variables (see `.env.example`). Reports land in
`$VERITY_OUTPUT_DIR/<media_id>/` (`report.html`, `report.pdf`,
`analysis.json`, `selection.json`).

## HTTP API and web console

- **API** — FastAPI; interactive docs at `/api/docs`. Long operations return
  a job id and stream logs; read endpoints expose cases, findings, artifacts,
  reports and system readiness.
- **Web console** — React/Mantine; case list, run panel, detector evidence
  panels, consensus view, forensic observations, system screen.

See [docs/web-ui.md](docs/web-ui.md) for the endpoint map, job model, security
posture and known gaps.

## Checkpoints

Weights are **never committed**. `detectors/weights_manifest.yaml` is the
single source of truth: SHA-256, size, origin, license, and whether Verity may
mirror the file. Everything is verified before use, and `VERITY_OFFLINE=1`
forbids downloads entirely.

```bash
verity weights pull --trained     # fetch + verify what the trained panel needs
verity weights status --verify    # hashes on disk vs. manifest
```

Redistributable checkpoints are mirrored on Hugging Face
(`ADPer/verity-weights`). Files whose authors granted no redistribution right
(NPR, MLEP, DF40, Effort, XLS-R-SLS) are fetched only from the authors' own
releases. See [docs/weights.md](docs/weights.md) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Native install (development)

```bash
uv sync --extra api                  # orchestrator + API
cd src/verity_ui && npm ci && npm run build && cd -   # web console
uv run verity weights pull --trained
uv run verity setup --trained        # one uv environment per detector (GPU auto-detected)
uv run verity run --media sample.mp4 --format html
uv run verity-server                 # http://127.0.0.1:8000
uv run pytest
```

Requirements: Linux, [`uv`](https://docs.astral.sh/uv/) ≥ 0.5, `ffmpeg`, and
the WeasyPrint system libraries (`libpango-1.0-0 libpangocairo-1.0-0
libcairo2`) for PDF reports.

## Containers

| Tag | Contents | Use |
|---|---|---|
| `ghcr.io/adper0705/verity:cuda` | everything + all trained-detector environments pre-baked (CUDA wheels) | servers/workstations with NVIDIA GPUs |
| `ghcr.io/adper0705/verity:cpu` | everything + trained-detector environments pre-baked (CPU wheels) | CPU hosts, classrooms, air-gapped labs |
| `ghcr.io/adper0705/verity:slim` | orchestrator, API, web console; detector environments provisioned into a volume on first use | development, custom detector sets |

Checkpoints are never inside the image: they live in the `verity-weights`
volume and are fetched on first start (or seeded for offline use). The full
images are built manually — see [docs/publishing.md](docs/publishing.md).

## Project layout

```text
config/pipeline.yaml          pipeline paths + enabled extractors
detectors/<id>/               one isolated uv project per detector
  detector.yaml               capabilities, input mode, trust tier, artifact contract
  model.py / inference.py     architecture + entrypoint
  output_parser.py            raw result -> Finding
  pyproject.toml              exact pins for this detector alone
detectors/common/             shared engine (face detection, sampling, CAM, viz, reasoning)
detectors/weights_manifest.yaml
src/verity/                   pipeline: ingest, extract, forensics, analyze, detect,
                              compile, report, weights, CLI
src/verity_api/               FastAPI service + background jobs
src/verity_ui/                React web console
scripts/verity.py             detector environment runner (uv, hardware-adaptive)
scripts/                      generators, release and matrix utilities
tests/                        unit suite (pipeline, selection, forensics, weights, API)
docs/                         technical, detector, forensics, artifact, API, publishing docs
```

## Documentation

| Document | Contents |
|---|---|
| [Technical overview](docs/technical-overview.md) | pipeline stages, config, environment, determinism |
| [Detector table](docs/detectors.md) | generated list of all 47 ports with tier/modality/license |
| [Detector catalog](docs/detector-catalog.md) | what each detector looks for, its artifacts and caveats |
| [Explainability artifacts](docs/explainability-artifacts.md) | every artifact written, where, and its evidence value |
| [Forensics](docs/forensics.md) | the five model-free analyzers, methods, limits |
| [Web UI and API](docs/web-ui.md) | jobs, endpoints, consensus, security posture |
| [Weights system](docs/weights.md) | manifest, verification, offline use, mirroring |
| [Publishing](docs/publishing.md) | distribution choices, release runbook |
| [Third-party notices](THIRD_PARTY_NOTICES.md) | code and checkpoint provenance/licenses |
| [Changelog](CHANGELOG.md) | release history |

## Development

```bash
uv sync                       # orchestrator environment
uv run pytest                 # tests
uv run python scripts/gen_detector_table.py        # regenerate docs/detectors.md
uv run python scripts/gen_third_party_notices.py   # refresh notice tables
uv run python scripts/release.py v0.4.0            # build release wheel
```

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md), the
[Code of Conduct](CODE_OF_CONDUCT.md), and [SECURITY.md](SECURITY.md) for
vulnerability reports. Detector ports follow the contract documented in
[docs/detector-catalog.md](docs/detector-catalog.md); every port must declare
its capabilities, trust tier and artifact contract in `detector.yaml`.

## Limitations and ethics

- **No detector is proof.** Every model has failure modes (new generators,
  heavy recompression, low resolution, adversarial input). The report is
  built for a human examiner to weigh, not for automated accusation.
- **Consensus is orientation, not authority.** A divided panel resolves to
  inconclusive; weak agreement is downgraded. The exported report deliberately
  contains no aggregate verdict.
- **Licenses restrict use.** Verity is CC BY-NC 4.0; most checkpoints are
  non-commercial and several derive from research-only datasets. Do not use
  the output for commercial or carceral purposes.
- **Privacy and consent.** Media examined here may contain personal data.
  Handle the workspace, exports and logs under your jurisdiction's evidence
  rules; Verity does not provide chain-of-custody or authentication.

## License and citation

Verity is released under **CC BY-NC 4.0** (non-commercial, attribution). The
detector implementations are inference-only ports of research code — most
from [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench) (CC BY-NC 4.0) —
and the checkpoints belong to their authors under their own licenses, listed
per file in `detectors/weights_manifest.yaml` and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

If you use Verity in academic work, cite it together with the upstream
detector papers relevant to your panel; see `CITATION.cff` for the project
entry.
