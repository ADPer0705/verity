# Verity

**Multi-detector deepfake examination for investigators, labs and researchers.**

[![CI](https://github.com/ADPer0705/verity/actions/workflows/ci.yml/badge.svg)](https://github.com/ADPer0705/verity/actions/workflows/ci.yml)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Release](https://img.shields.io/github/v/release/ADPer0705/verity?label=release)](https://github.com/ADPer0705/verity/releases/latest)
[![Container](https://img.shields.io/badge/container-ghcr.io%2Fadper0705%2Fverity-blue)](https://github.com/ADPer0705/verity/pkgs/container/verity)
[![Weights](https://img.shields.io/badge/weights-Hugging%20Face-yellow)](https://huggingface.co/ADPer/verity-weights)

Verity takes any image, video or audio file, characterises it, picks the
detectors that can actually consume it, runs each one in its own isolated,
version-pinned environment, and writes a deterministic report in which every
detector's verdict, confidence, reasoning and explainability artifacts (faces,
attention maps, timelines) are preserved separately. **It never collapses the
panel into a single real/fake score — disagreement between methods is the
finding.**

It ships as a **standalone single Linux executable** (built for single-investigator,
air-gapped forensic workstations — no Docker or system Python required), with an
embedded web console, HTTP API, CLI, chain-of-custody evidence bundle exporter,
and a checksummed, license-tracked checkpoint mirror. Model-free forensic
analyzers (metadata, compression, noise, temporal, audio) run alongside the
detectors and report observations, never verdicts.

> Detector outputs are statistical indicators, not proof. They must be read
> by a qualified examiner together with the artifacts Verity produces.

## Quick start (single command, air-gap ready)

Install the standalone `verity` binary on Linux x86_64 (verifies SHA-256
automatically and installs to `~/.local/bin/verity`):

```bash
curl -fsSL https://raw.githubusercontent.com/ADPer0705/verity/main/install.sh | bash
```

Then let `verity` provision everything once while online (self-bootstrapped
`uv` + managed Python runtimes, all 39 evidence-grade detector environments,
SHA-256-verified checkpoints, and Hugging Face / Torch Hub foundation caches
stored cleanly via `platformdirs`):

```bash
verity setup --trained --airgap-ready
```

Now disconnect the workstation from the network permanently, verify integrity,
and launch the forensic Web Console:

```bash
verity doctor --verify
verity serve                 # Web Console at http://127.0.0.1:8000
```

Need to deploy to a machine that **never** touches the internet? Export a
self-contained USB kit on an online staging host and import it offline:

```bash
# On online staging workstation:
verity airgap export --out /mnt/usb/verity-kit

# On air-gapped forensic workstation:
/mnt/usb/verity-kit/verity airgap import --from /mnt/usb/verity-kit
verity doctor --verify && verity serve
```

> Host prerequisites for video/audio extraction & PDF reports: `ffmpeg` and
> WeasyPrint's Pango/Cairo libraries (`sudo apt-get install -y ffmpeg
> libpango-1.0-0 libpangocairo-1.0-0 libcairo2` on Ubuntu/Debian). An NVIDIA GPU
> with 8 GB+ VRAM is recommended; everything also runs on CPU. Budget ~60 GB of
> disk for checkpoints and all 39 detector environments.

## What it examines

| Input | Analyzed by |
|---|---|
| Images (JPEG/PNG/WebP/TIFF/…) | whole-image and face-crop detectors, metadata/EXIF, compression, noise/spectrum |
| Videos (MP4/MOV/MKV/WebM/…) | face-crop, face-clip/temporal, audio-visual lip-sync and whole-frame detectors, image forensics on sampled frames, temporal structure, audio track |
| Audio (WAV/MP3/M4A/FLAC/…) | voice-clone anti-spoofing detectors, audio spectrum/noise-floor/spectrogram forensics |

Manipulation families tracked: face swap, face reenactment, lip sync,
synthetic face, AI-generated image, local image editing, AI-generated video,
synthetic/cloned voice (`src/verity/models/taxonomy.py`).

## How it works

```text
media ─► ingest ─► extract ─► forensics ─► analyze ─► select ─► detect ─► compile ─► report ─► export
         sha256    frames,    5 model-free ffprobe +  rule-based 58 uv     sorted     HTML/PDF  chain-of-custody
         r/o copy  faces,     analyzers    face scan  tier-aware envs,     findings   + Web UI  .zip + SHA256SUMS
                   meta/audio                         selection  isolated
```

- **Deterministic.** Media IDs are SHA-256 checksums, inference uses fixed
  seeds, every list is explicitly sorted: the same media and detector set
  always yield the same report.
- **Isolated.** Each detector is its own `uv` project
  (`detectors/<id>/pyproject.toml`) with exact pins. Incompatible PyTorch
  versions coexist; a broken detector cannot poison another.
- **Explained.** The selector records why each detector was included or
  excluded; each finding carries its reasoning, confidence factors, face/frame
  crops, attention/Grad-CAM maps, auxiliary sub-scores, and prediction timelines.
- **No single score.** Per-detector verdicts stay separate; the UI shows an
  agreement bar and downgrades weak or split panels to `inconclusive`. The
  exported report renders no aggregated verdict at all.
- **Evidence preservation.** `verity export --media-id <id>` (and the UI's
  **Export evidence bundle** button) writes a deterministic `.zip` containing
  `MANIFEST.json`, `SHA256SUMS.txt`, the read-only working copy, HTML/PDF
  dossier, model-free forensic figures, and all detector findings & exhibits.

## Detectors and trust tiers

58 detector architectures are ported. **39 are evidence-grade** — released
checkpoints with trained classification heads (or training-free methods) —
and are the only ones auto-selected:

| Tier | Meaning | Default |
|---|---|---|
| `trained` | released deepfake-trained checkpoint, or a training-free method | selected |
| `backbone-only` | pretrained backbone (ImageNet, Kinetics) with an untrained head | excluded |
| `random-init` | no released checkpoint at all; weights are random | excluded |
| `untrained` | foundation model fetched at first run, detector head untrained | excluded |

Untrained scaffolds run only on explicit request (`--include-untrained` or by
name), are visually flagged, and never count toward consensus.

## CLI reference

```bash
verity setup    --trained [--airgap-ready] [--all] [--detector a,b]
verity doctor   [--verify] [--json]
verity serve    [--host 127.0.0.1] [--port 8000]
verity run      --media <file|dir> [--detector a,b] [--include-untrained] [--format pdf|html]
verity analyze  --media <file|dir> [--json]
verity forensics --media-id <id>
verity export   --media-id <id> [--out case.zip]
verity weights  pull --trained | warmup --trained | status [--verify] | verify | bundle --repo <hf-repo>
verity airgap   export --out <dir> | import --from <dir>
verity ingest | extract | compile | report
```

All paths resolve through `platformdirs` (`~/.local/share/verity`,
`~/.cache/verity`, `~/.config/verity`) when installed as a binary, or relative
to the repository in development, and can be overridden via `VERITY_HOME` or
`VERITY_*` environment variables (see `.env.example`).

## Checkpoints & attribution

Weights are **never committed** to git and **never authored by Verity**. Every
checkpoint belongs to its original academic/open-source authors and is governed
by its upstream license — tracked per file in
`detectors/weights_manifest.yaml` and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

- Redistributable checkpoints (28 files, 6.3 GB) are mirrored with full author
  attribution on Hugging Face at
  [`ADPer/verity-weights`](https://huggingface.co/ADPer/verity-weights) for
  reproducible, reliable downloads.
- Files whose authors granted no redistribution right or requested origin-only
  downloads (NPR, MLEP, DF40, Effort, XLS-R-SLS, FTCN, D³, ProDet, LipFD, SBI,
  LAA-Net) are **never mirrored** and are fetched strictly from the authors'
  official release URLs.
- Every downloaded file is SHA-256 verified before use; `VERITY_OFFLINE=1` (or
  `offline: true` in `~/.config/verity/config.yaml`) forbids all network calls.

## Alternative installs (Docker Compose & development)

### Docker / Podman Compose

Each GitHub Release also attaches `verity-deploy.tar.gz` + `SHA256SUMS` and
publishes `ghcr.io/adper0705/verity:slim`:

```bash
curl -fsSLO https://github.com/ADPer0705/verity/releases/latest/download/verity-deploy.tar.gz
curl -fsSLO https://github.com/ADPer0705/verity/releases/latest/download/SHA256SUMS
sha256sum -c SHA256SUMS
tar xzf verity-deploy.tar.gz && cd verity-deploy
./install.sh
```

### From source (development)

```bash
uv sync --extra api
cd src/verity_ui && npm ci && npm run build && cd -
uv run verity weights pull --trained
uv run verity setup --trained
uv run verity serve
uv run pytest -q
```

## Documentation (OpenWiki)

Verity's complete architecture, engineering rationale, detector catalog,
forensics reference, explainability catalog, deployment/air-gap guide, and
release runbook live in **`openwiki/`**:

- **[Quickstart & documentation map](openwiki/quickstart.md)** — start here
- **[Architecture & design rationale](openwiki/architecture/overview.md)** — package boundaries, determinism, isolation, no-verdict principle
- **[Detector catalog (all 58 ports)](openwiki/reference/detector-catalog.md)** — modality groups, trust tiers, caveats, panel selection
- **[Explainability & evidence artifacts](openwiki/reference/explainability-artifacts.md)** — artifact contracts, Grad-CAM/attention, chain-of-custody bundles
- **[Model-free forensics](openwiki/architecture/forensics.md)** — the 5 signal-processing/metadata analyzers and their limits
- **[Weights & checkpoint provenance](openwiki/architecture/weights-and-checkpoints.md)** — manifest v2, SHA-256 gate, redistribution policy
- **[Web API & console](openwiki/architecture/web-api-and-console.md)** — FastAPI job model, confined file serving, React UX
- **[Deployment & air-gap operations](openwiki/operations/deployment-and-airgap.md)** — standalone binary, `platformdirs`, USB air-gap kit, Docker Compose
- **[Releasing & publishing](openwiki/operations/releasing-and-publishing.md)** — PyInstaller glibc baseline, Hugging Face mirror, GHCR, testing
- **[Third-party notices](THIRD_PARTY_NOTICES.md)** · **[Changelog](CHANGELOG.md)** · **[Contributing](CONTRIBUTING.md)** · **[Security](SECURITY.md)**

## Limitations and ethics

- **No detector is proof.** Every model has failure modes (new generators,
  heavy recompression, low resolution, adversarial input). The report is built
  for a human examiner to weigh, not for automated accusation.
- **Consensus is orientation, not authority.** A divided panel resolves to
  `inconclusive`; weak agreement is downgraded. The exported report deliberately
  contains no aggregate verdict.
- **Licenses restrict use.** Verity is CC BY-NC 4.0; most checkpoints are
  non-commercial and several derive from research-only datasets (such as
  FaceForensics++). Do not use Verity for commercial or automated carceral
  decisions.
- **Privacy and custody.** Verity computes SHA-256 working-copy hashes and
  exports verifiable evidence bundles (`MANIFEST.json` + `SHA256SUMS.txt`),
  but legal admissibility and access control depend on your lab's chain-of-custody
  procedures and workstation security.

## License and citation

Verity is released under **CC BY-NC 4.0** (non-commercial, attribution).
Detector implementations are inference-only ports of upstream research code —
many from [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench) (CC BY-NC 4.0)
— and checkpoints remain the intellectual property of their respective authors
under their upstream licenses, listed per file in
`detectors/weights_manifest.yaml` and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

If you use Verity in academic work, cite Verity (`CITATION.cff`) together with
the upstream detector papers relevant to your panel.
