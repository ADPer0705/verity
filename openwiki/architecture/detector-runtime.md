---
type: subsystem
title: Detector runtime and isolation
description: How Verity discovers detector definitions, provisions and runs each detector in its own uv-managed environment via verity.runtime.env_manager and scripts/verity.py, adapts CUDA/CPU wheels, supports air-gapped execution, and parses findings.
tags: [detectors, runtime, uv, isolation, registry, airgap]
verified:
  - by: openwiki/0.9.0
    at: 2026-11-21T06:14:40.994Z
sources:
  - id: openwiki-source-8991b61e6ff0834683a1f059
    resource: repo://scripts/verity.py
  - id: openwiki-source-081142855503f021a5dfbb83
    resource: repo://src/verity/detect/detector_registry.py
  - id: openwiki-source-61ad6b16674ac0599a5a1e59
    resource: repo://src/verity/detect/detector_runner.py
  - id: openwiki-source-c5cd8824ffe1e5595fda06ac
    resource: repo://src/verity/detect/output_parser.py
  - id: openwiki-source-1c0176c5ec21efddcddaa30c
    resource: repo://src/verity/runtime/env_manager.py
generated: { by: "antigravity", at: "2026-11-21T06:14:40.994Z" }
---

# Detector runtime and isolation

Verity never imports detector models into the orchestrator or API process. Each of
the 58 detector ports under `detectors/<id>/` owns an isolated Python virtual
environment and communicates with `DetectorRunner` solely through staged input
files, stdout/stderr logs, and JSON/PNG outputs.

Keeping every detector in its own `uv` project solves a severe practical problem in
media forensics: published research detectors pin incompatible versions of PyTorch,
CUDA wheels, `timm`, `transformers`, `mediapipe`, and `facenet-pytorch`. Subprocess
isolation allows all 58 architectures to coexist without dependency conflicts and
ensures that one failing detector cannot poison another or corrupt the admitted
evidence.

## Component ownership

- **`DetectorRegistry` (`src/verity/detect/detector_registry.py`)** — scans
  `detectors/` in sorted directory order, parses each `detector.yaml` into a
  `DetectorDefinition`, and reports whether required local weights exist.
- **`DetectorRunner` (`src/verity/detect/detector_runner.py`)** — stages read-only
  inputs into a temporary directory, executes the detector environment, captures
  stdout/stderr tails, parses outputs into a `Finding`, and stamps the weight trust
  tier.
- **`verity.runtime.env_manager` (`src/verity/runtime/env_manager.py`)** — the
  single in-process `uv` runtime manager used directly by the standalone single
  binary and CLI/API, while `scripts/verity.py` delegates to `env_manager.main` for
  container (`verity-run`) and native script invocations.
- **`verity.detect.output_parser` (`src/verity/detect/output_parser.py`)** —
  default `result.json` -> `Finding` mapper and explainability artifact collector.

## The detector package contract

Each `detectors/<id>/` directory defines:

| File | Required | Role |
|---|---|---|
| `detector.yaml` | yes | Capabilities, `input_mode`, preprocessing, `cam`/`region_analysis`, weights block, `targets`, optional `caveat` |
| `pyproject.toml` | yes | Pinned Python version and exact package requirements for this detector alone |
| `model.py` | yes | Network definition and `build_model(weights_path, device)` returning normalized `[real, fake]` logits |
| `inference.py` | yes | Thin entrypoint invoking `common.engine.run_detector(build_model, detector.yaml)` |
| `output_parser.py` | no | Optional custom `parse(raw, output_path, version)` override |
| `weights/` | when applicable | Checkpoint file(s) plus `README.md` provenance (symlinked to persistent storage in standalone/container mode) |

## Registration and live visibility

`DetectorRegistry` scans `detectors/` cleanly on instantiation and exposes
`get()`, `list_detectors()`, `register()`, and `weights_available()`. Because the
API constructs a fresh registry per request via `build_registry()`, dropping a new
checkpoint into `weights/` or adding a detector directory takes effect immediately
without restarting the server.

## Execution lifecycle in `DetectorRunner`

When `DetectorRunner.run(media_id, detector_id)` executes:

1. Resolves the read-only working copy under `workspace/media/` and ensures
   `workspace/output/<media_id>_<detector_id>/` exists.
2. `_prepare_input_dir` stages a temporary directory containing a copy of the media
   and the case's `artifacts/<media_id>/` tree so the detector cannot mutate the
   workspace working copy.
3. By default (when `VERITY_RUNNER` is unset), `_run_detector` calls
   `verity.runtime.env_manager.run_detector_process` in-process so frozen PyInstaller
   binaries never rely on an external repository checkout or system Python. If
   `VERITY_RUNNER` is set, `_invoke_verity` executes the configured external script.
4. Enforces `VERITY_DETECTOR_TIMEOUT` (default `3600` seconds), surfacing timeouts as
   an explicit `RuntimeError`.
5. Saves bounded `stdout.log` and `stderr.log` (last 20,000 characters each) alongside
   `result.json`.
6. If `result.json` was not created (e.g., non-zero exit), returns a neutral fallback
   finding (`prediction: 0.5`, `confidence: 0.0`) embedding the exit code and stderr
   tail so examination of the remaining panel continues uninterrupted.
7. Calls `_parse_output`, stamps `finding.tier = definition.weights.tier`, and writes
   `finding.json`.

## Environment provisioning and air-gap safety (`env_manager`)

`src/verity/runtime/env_manager.py` replaces external shell scripts with a
portable Python implementation designed for both containers and zero-dependency
single-binary workstations:

- **Self-bootstrapped `uv` and managed Python**: `ensure_uv()` locates `uv` in
  Verity's private `<data_dir>/bin/` or `PATH`, and if absent online, downloads the
  official Astral binary into `paths.bin_dir()`. In standalone/air-gap mode,
  `build_subprocess_env()` sets `UV_PYTHON_INSTALL_DIR=<data_dir>/python` and
  `UV_PYTHON_PREFERENCE=only-managed` so detector environments do not depend on any
  host system Python installation.
- **Optional static `ffmpeg`/`ffprobe`**: `ensure_ffmpeg()` can fetch static Linux
  x86_64 builds into `paths.bin_dir()` when missing from `PATH`.
- **Hardware profiles (`auto` / `cuda` / `cpu`)**: `detect_profile()` checks user
  config (`hw_profile`) and `nvidia-smi` to choose `cuda` or `cpu`. For `cpu`,
  `cpu_pyproject()` rewrites `torch`/`torchvision` exact pins with
  `override-dependencies` (`+cpu`) and adds the PyTorch CPU index so plain `uv sync`
  yields ~12x smaller environments.
- **Baked container environments**: If `VERITY_DETECTOR_ENVS_BAKED` is set (full
  Docker builds), `project_dir()` checks whether the baked `.verity-synced` SHA-256
  marker matches `project_pyproject()` and reuses that read-only environment directly.
- **Deterministic hash-gated sync and offline rebuilds**: `sync_one()` compares
  the SHA-256 of the effective `pyproject.toml` against `.verity-synced`. When
  changed (or `--force`), online syncs refresh `uv.lock` and run
  `uv sync --project <dir>`, while offline syncs preserve transferred `uv.lock`
  files and pass `--offline --no-python-downloads`.
- **Air-gap runtime sealing**: When `paths.is_offline()` is true,
  `build_subprocess_env()` forces `VERITY_OFFLINE=1`, `HF_HUB_OFFLINE=1`,
  `TRANSFORMERS_OFFLINE=1`, `DIFFUSERS_OFFLINE=1`, `UV_OFFLINE=1`, and
  `UV_NO_PYTHON_DOWNLOADS=1` on every detector subprocess.
- **Persistent weights symlinking**: `ensure_weights_links()` maintains
  `detectors/<id>/weights -> $VERITY_WEIGHTS_ROOT/<id>` in container and standalone
  modes so refreshing bundled detector code never touches downloaded checkpoints.

## Output parsing and evidence discovery

`DetectorRunner._parse_output` checks first for a detector-local
`detectors/<id>/output_parser.py` defining `parse(raw, output_path, version)`, and
falls back to `verity.detect.output_parser.parse`, which maps `prediction > 0.7` to
`FAKE`, `< 0.3` to `REAL`, and `[0.3, 0.7]` to `UNCERTAIN`. Even when a custom
parser omits explicit paths, `discover_evidence_paths()` scans `timeline.png`,
`full_frames/*_frame.png`, `attention_maps/*_attention.png`,
`reconstructions/*_recon.png`, `faces/*.png`, and `frames/*.png` in sorted order so
all explainability exhibits reach `Finding.reasoning.evidence`.

## Related pages

- [Shared detector engine](detector-engine.md) — what executes inside each isolated
  environment.
- [Weights and checkpoints](weights-and-checkpoints.md) — checkpoint verification and
  symlink targets.
- [Deployment, configuration and air-gap operations](../operations/deployment-and-airgap.md)
  — how `uv`, managed Pythons, wheel caches, and lockfiles transfer via `airgap export`.
