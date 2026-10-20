# Verity: technical overview

Containerized deepfake detection pipeline for investigations and research.
Verity ingests media from the host, analyzes it, selects the detectors that
are suitable for it, runs each detector in its own isolated uv-managed
environment, and produces a deterministic, explainable report.

License: CC BY-NC 4.0 (see [LICENSE](../LICENSE) and
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)).

Companion documents: [detector catalog](detector-catalog.md) ·
[explainability artifacts](explainability-artifacts.md) ·
[forensics](forensics.md) · [weights system](weights.md) ·
[web UI/API](web-ui.md) · [publishing](publishing.md).

## What Verity does

```
host /input ─► ingest ─► extract ─► analyze ─► select ─► detect ─► compile ─► report ─► /output
                │           │          │          │          │           │
              checksummed  frames,   ffprobe +  rule-based  uv venvs,   findings
              copy         faces,    face scan   suitability 47 isolated  sorted
                           metadata,                          + per-detector
                           audio                              explainability
                └────────── model-free forensics: metadata, compression, noise,
                            video temporal, audio ──────────────────────────────┘
```

- **Analyze** — characterizes the media deterministically: format, duration,
  fps, resolution, audio track, and a face-presence scan over up to 30 evenly
  sampled frames.
- **Select** — every detector's capabilities (media types, faces required,
  weights availability) are matched against the analysis. Each decision is
  recorded, so the report explains both what ran and what was skipped and why.
  Rule zero is the **trust tier**: a detector whose classification head is
  not trained (`backbone-only`, `random-init`) is excluded unless
  `--include-untrained` / `VERITY_INCLUDE_UNTRAINED=1` is set or it is named
  explicitly; its findings are then flagged in the report, the API and the
  UI and never count toward the consensus or the aggregated score.
- **Detect** — each detector runs in its own uv project
  (`detectors/<id>/pyproject.toml` is the single source of truth — exact pins,
  no shared global env). Every detector's environment is tailored to it alone:
  the Python version (3.12 everywhere today — the pinned wheels cap at 3.12)
  and only the dependencies its code actually imports, on top of the
  `detectors/common` runtime (face detection, sampling, overlays).
  `scripts/verity.py` provisions and runs them, hardware-adaptive: `auto`
  (GPU when `nvidia-smi` works, else CPU), `cuda`, or `cpu` (+cpu torch wheels,
  ~12x smaller installs). Environments are cached per `(profile, detector)`
  under `$VERITY_DETECTOR_ENVS` (`cuda/<id>/` and `cpu/<id>/`); weights live
  in a persistent `$VERITY_WEIGHTS_ROOT` volume.
- **Forensics** — five model-free analyzers (metadata/EXIF, compression,
  noise/spectrum, video temporal structure, audio spectrum) run before the
  detectors and fail soft: an analyzer error becomes a `notice`, never a lost
  case. They emit observations with evidence numbers and severity, not
  verdicts. See [forensics.md](forensics.md).
- **Report** — a trimmed, detector-agnostic report: verdict, reasoning,
  confidence factors, and explainability artifacts. Top faces / attention
  maps / reconstructions are shown; all the rest are in dropdowns. Detector
  metadata, timings, and benchmark noise are gone.

Reports are **deterministic**: media IDs derive from the SHA-256 checksum,
findings and every list are explicitly sorted, and inference runs with fixed
seeds. The same media and detectors always produce the same report (the only
wall-clock value is the report's generation timestamp).

## Quick start (container)

See the [README](../README.md) for the pull-and-run instructions and
[publishing.md](publishing.md) for how the images and the weights mirror are
built. In short: `docker compose up -d` starts the web console and API on
port 8000; `docker compose run --rm verity run --media /input/<file>` runs
the CLI once. Three images exist: `:cuda` and `:cpu` with every trained
detector's environment pre-baked, and `:slim` which provisions environments
into the `/envs` volume on first use. Checkpoints live in the `/weights`
volume and are fetched on first start (`verity weights pull --trained`).

## CLI

```bash
verity run --media <file|dir> [--detector a,b,c] [--include-untrained] [--format html|pdf]
verity analyze --media <file|dir> [--json] [--include-untrained]   # characteristics + selection rationale
verity setup [--trained | --all | --detector <id>[,<id>]] [--force] # provision detector environments
verity weights pull [--trained | --all | --detector <id>] [--source auto|hub|origin] [--force]
verity weights status [--verify] | verify | bundle --repo <hf-repo>
verity ingest|extract|compile|report                                # step-by-step debugging
verity-server                                                       # API + web console
```

`run` and `analyze` default `--media` to `$INPUT_DIR` (set to `/input` in the
container). Reports land in `$VERITY_OUTPUT_DIR/<media_id>/` with `report.html`
+ `report.pdf`, `analysis.json` (media characteristics), and `selection.json`
(detector decisions with reasons). Per-detector artifacts live in the
workspace (`$VERITY_WORKSPACE`).

## Requirements

- Linux with **Docker** (or podman) for the runtime image
- `uv` (>= 0.5) for native runs and development
- System packages for WeasyPrint: `libpango-1.0-0 libpangocairo-1.0-0
  libcairo2`, and `ffmpeg` for extraction (all baked into the image)
- GPU optional — the runtime auto-selects the CUDA or CPU torch profile

## Native usage (development)

```bash
uv sync                          # orchestrator env (click, pydantic, cv2, weasyprint...)
uv run pytest                    # run the test suite
uv run verity analyze --media sample.mp4 --workspace ws
uv run verity run --media sample.mp4 --detector recce --workspace ws --output out
python scripts/verity.py --profile cpu setup --detector <id>   # one detector env
```

## Detectors

47 detector architectures are ported under `detectors/`, each with `model.py` /
`inference.py` / `detector.yaml` / `pyproject.toml` / `output_parser.py`.
26 have trained classification heads (or are training-free methods) and are
the only ones auto-selected. Coverage: images, videos, GANs, diffusion
(including AEROBLADE and GenD), universal detectors, face-swap/re-enactment,
lip sync, and audio anti-spoofing. The per-detector table (tier, modality,
weights, license) is in [detectors.md](detectors.md); the field guide with
artifacts and caveats is in [detector-catalog.md](detector-catalog.md);
provenance and checksums are in `detectors/weights_manifest.yaml`.

Selection is driven by each detector's `detector.yaml`. `input_mode` says
what the shared engine feeds the model and therefore what media it can
consume:

| `input_mode` | fed to the model | media | typical detectors |
|---|---|---|---|
| `face_crop` (default) | one face crop per sampled frame | image, video with faces | FF++-era CNNs, CLIP face probes |
| `face_clip` (`video_mode: true`) | `clip_size` consecutive face crops | video with faces | I3D / FTCN / AltFreezing, LipForensics |
| `full_frame` | the whole image or sampled whole frames (`resize_short`, `center_crop`, `max_frames`) | image, video | AI-generated-image detectors (UnivFD, NPR, SAFE, Community Forensics) |
| `full_clip` | `clip_size` consecutive whole frames | video | generated-video detectors |
| `audio` | 16 kHz mono waveform windows (`audio_window_seconds`, `audio_hop_seconds`) | audio, video with a soundtrack | voice-clone detectors |

`targets` lists the manipulation families (see `src/verity/models/taxonomy.py`)
the detector was trained for, and the `weights` block determines the trust
tier and whether a local checkpoint is needed:

| `weights.kind` | `weights.trained` | tier | meaning |
|---|---|---|---|
| `official` | (default true) | `trained` | released deepfake-trained checkpoint |
| `external` | `true` | `trained` | released detector or training-free method, fetched at first run |
| `external` | (default false) | `untrained` | pretrained backbone fetched at first run, head untrained |
| `backbone` | (default false) | `backbone-only` | local pretrained backbone, head untrained |
| `random` | (default false) | `random-init` | no checkpoint anywhere; random weights |

Only `trained` detectors are auto-selected. The others ship for researchers:
drop a checkpoint into `detectors/<id>/weights/`, set `trained: true`, and the
detector becomes evidence-grade for that install.

### RECCE honest-degradation note

The official DeepfakeBench v1.0.1 checkpoint ships with an **untrained
reconstruction decoder**. Verity audits decoder liveness at load and, when
dead, omits reconstruction visualizations/claims entirely and says so in the
reasoning. The verdict itself is valid — it rests on the trained classifier
head and attention maps.

## Configuration

`config/pipeline.yaml` (all overridable via env / CLI flags):

```yaml
detectors_dir: "detectors"     # detector definitions
workspace: "workspace"         # runtime data (ingested media, artifacts)
output_dir: "output"           # reports
extractors: [frame, face, metadata, audio]
```

Environment variables: `VERITY_DETECTORS_DIR`, `VERITY_WORKSPACE`,
`VERITY_OUTPUT_DIR`, `VERITY_DETECTOR_ENVS`, `VERITY_WEIGHTS_ROOT`,
`VERITY_HW_PROFILE` (auto|cuda|cpu), `VERITY_DETECTOR_TIMEOUT` (seconds,
default 3600), `VERITY_INCLUDE_UNTRAINED`, `VERITY_OFFLINE`,
`VERITY_WEIGHTS_HUB` (Hugging Face mirror), `VERITY_WEIGHTS_AUTO` (container:
fetch missing trained checkpoints on start), `VERITY_DETECTOR_ENVS_BAKED`
(read-only pre-baked environments inside the full images), plus the
detector-specific `VERITY_ANTIDEEPFAKE_REPO` (XLS-R-1B variant) and
`VERITY_ORT_FORCE_CPU` (ONNX runtime on CPU). See `.env.example`.

## Development

```bash
uv sync                    # create .venv
uv run pytest              # run tests
python scripts/verity.py --profile cpu setup --detector <id>   # one detector env
python scripts/verity.py --profile cpu run <id> /input /output # run it natively
```
