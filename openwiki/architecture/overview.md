---
type: architecture
title: Architecture overview
description: Major packages, unidirectional dependency boundaries, dual-mode storage layout via platformdirs, entrypoints, and the engineering rationale behind Verity's design invariants.
tags: [architecture, packages, design, platformdirs, workspace, isolation]
verified:
  - by: openwiki/0.9.0
    at: 2026-11-21T06:14:40.994Z
sources:
  - id: openwiki-source-29c60552d63929b7e3ce579e
    resource: repo://config/pipeline.yaml
  - id: openwiki-source-05ccef8d4cf1698187f20464
    resource: repo://pyproject.toml
  - id: openwiki-source-8991b61e6ff0834683a1f059
    resource: repo://scripts/verity.py
  - id: openwiki-source-392cb142cea6fef3de3d5452
    resource: repo://src/verity_api/config.py
  - id: openwiki-source-256a301831e925392a4a4725
    resource: repo://src/verity_api/jobs.py
  - id: openwiki-source-8fd3e8c7a518a6182d2aef95
    resource: repo://src/verity_api/main.py
  - id: openwiki-source-27c03b24282b02c206c94fdd
    resource: repo://src/verity_api/service.py
  - id: openwiki-source-e4dc7d4c95cc68eb2f4b4656
    resource: repo://src/verity/cli.py
  - id: openwiki-source-61ad6b16674ac0599a5a1e59
    resource: repo://src/verity/detect/detector_runner.py
  - id: openwiki-source-184a2f452d290bbb1fb99bd9
    resource: repo://src/verity/paths.py
  - id: openwiki-source-1c0176c5ec21efddcddaa30c
    resource: repo://src/verity/runtime/env_manager.py
generated: { by: "antigravity", at: "2026-11-21T06:14:40.994Z" }
---

# Architecture overview

Verity is an investigative deepfake examination workbench engineered around two
foundational decisions:

1. **Preserve every detector's voice separately**: never collapse multi-detector
   panels into a single opaque verdict — disagreement across architectures is the
   primary analytical finding.
2. **Zero external runtime dependencies in deployment**: ship one self-contained
   Linux binary (alongside a Docker option) that manages `uv`, Python runtimes,
   58 isolated environments, SHA-256-verified checkpoints, and case workspaces
   cleanly via `platformdirs`, then operates 100% air-gapped.

The repository consists of three core software surfaces plus isolated out-of-process
detector runtimes and chain-of-custody exporters:

| Component | Location | Role |
|---|---|---|
| `verity` | `src/verity/` | Core pipeline: `paths`, `ingest`, `extract`, `forensics`, `analyze`, `detect`, `compile`, `report`, `weights`, `runtime`, `export`, `cli` |
| `verity_api` | `src/verity_api/` | FastAPI HTTP service and thread-pool `JobManager` wrapping pipeline objects |
| `verity_ui` | `src/verity_ui/` | React / Mantine / Vite SPA served directly by FastAPI on port 8000 |
| Detector environments | `detectors/<id>/` | 58 independent `uv` projects launched out of process |

## Unidirectional dependency direction

<!-- openwiki: mermaid parse failed and this diagram was converted to a text fence so it does not break rendering. Fix the diagram source and restore the mermaid fence. Parser error: Heuristic: a semicolon inside a label breaks rendering; rephrase the label. -->
```text
graph TD
  UI["verity_ui (React SPA)"] --> API["verity_api (FastAPI + JobManager)"]
  API --> CORE["verity (Pipeline + paths + runtime)"]
  CORE --> MODELS["verity.models"]
  CORE -. "uv subprocess / env_manager" .-> DET["detectors/&lt;id&gt; (58 isolated uv envs)"]
```

- `verity` knows nothing about FastAPI or React.
- `verity_api` is strictly a thin adapter (`src/verity_api/service.py`) composing the
  exact same pipeline objects (`MediaIngestor`, `ForensicsRunner`, `MediaAnalyzer`,
  `DetectorSelector`, `DetectorRunner`, `FindingsCompiler`, `ReportGenerator`,
  `export_evidence_bundle`) and streaming steps via `JobContext` instead of stdout.
  This guarantees the CLI, HTTP API, Web Console, and PDF report cannot disagree on
  any probability or selection decision.
- `detectors/<id>/` run out of process via `verity.runtime.env_manager` (or
  `scripts/verity.py` / `verity-run`), exchanging staged media copies,
  `result.json`, `finding.json`, PNG artifacts, and logs. Out-of-process execution
  is why dozens of mutually incompatible PyTorch, CUDA, `timm`, and `transformers`
  versions coexist cleanly.

## Why single binary + `platformdirs` + `uv` (instead of a 20 GB fat binary or Docker-only)

Lab forensic workstations are frequently offline or air-gapped after initial
provisioning, run under locked-down non-root investigator accounts, and cannot
always run a Docker daemon. Packaging everything directly inside a single binary
would weigh 15–25 GB and break CUDA/PyTorch isolation. Instead, Verity uses a
hybrid architecture cleanly implemented in `src/verity/paths.py` and
`src/verity/runtime/env_manager.py`:

- **Small single executable (`~130 MB`)**: PyInstaller (`scripts/build_binary.py`)
  bundles only Python 3.12, `verity`, `verity_api`, the built `verity_ui` SPA,
  config/Haar cascades, and the source/YAML/lock specs of all 58 detectors (never
  weight binaries).
- **Automatic dual-mode path resolution (`src/verity/paths.py`)**:
  - *Standalone binary / `VERITY_HOME` mode*: stores all mutable runtime state via
    `platformdirs` (`~/.local/share/verity`, `~/.cache/verity`,
    `~/.config/verity`, or `$VERITY_HOME/{data,cache,config}`). On startup,
    `ensure_bundled_detectors_extracted()` unpacks bundled detector scripts while
    symlinking every `detectors/<id>/weights` to persistent `<data_dir>/weights/<id>`
    so upgrading the binary never deletes downloaded checkpoints.
  - *Repository development mode*: when running from an editable checkout with no
    `VERITY_HOME`, preserves repo-local defaults (`detectors/`, `workspace/`,
    `.detectorenvs/`) so local tests and hacking need zero configuration.

## Core package responsibilities (`src/verity/`)

| Module | Responsibility |
|---|---|
| `verity.paths` | Dual-mode `platformdirs` / `VERITY_HOME` / repo layout, persistent `config.yaml`, offline seal check |
| `verity.ingest` | SHA-256 content-addressed media IDs (first 12 hex chars) and `0o444` read-only copies |
| `verity.extract` | Deterministic `frame`, `face`, `metadata`, and `audio` extractors |
| `verity.forensics` | Five model-free analyzers emitting severity-sorted observations without verdicts |
| `verity.analyze` | `ffprobe` + Haar face scan characterization and ordered rule-based `DetectorSelector` |
| `verity.detect` | `DetectorRegistry`, staged `DetectorRunner`, and `output_parser` |
| `verity.runtime` | `env_manager` — self-bootstraps `uv` + managed Pythons, builds profiles, warms HF/Torch caches |
| `verity.compile` | `FindingsCompiler` sorting findings and computing trained-only aggregate scores |
| `verity.report` | Deterministic self-contained HTML + optional WeasyPrint PDF reports |
| `verity.weights` | Version-2 manifest, SHA-256 verification gate, hub mirroring, HF bundling |
| `verity.export` | Chain-of-custody `.zip` evidence bundles (`MANIFEST.json` + `SHA256SUMS.txt`) and USB `airgap` kits |

## Workspace & report layout

```text
<workspace>/                      (data_dir/workspace in binary mode; ./workspace in dev)
  media/
    <media_id>.<ext>              0o444 read-only working copy
    .<media_id>.json              MediaInfo sidecar (full SHA-256 + source metadata)
  artifacts/<media_id>/
    {frames,faces,metadata,audio}/
    forensics/                    forensics.json + per-analyzer JSON & PNGs
  output/
    <media_id>_<detector_id>/     result.json, finding.json, logs, attention/face PNGs, timeline.png
<output_dir>/                     (configurable; defaults to <workspace>/output)
  <media_id>/                     analysis.json, selection.json, report.html, report.pdf, evidence zip
```

Note the intentional separation documented in `src/verity_api/config.py`: raw
detector run directories always live under `workspace/output/`, whereas case-level
reports and analysis JSON live under configurable `output_dir`.

## Core engineering principles

1. **Determinism**: Media IDs derive from SHA-256 bytes, video readers iterate
   sequentially, seeds (`0xC0FFEE`) and cuDNN flags are pinned, JSON keys are sorted,
   and galleries sort by probability + filename tie-breaker.
2. **Honest absence & trust tiers**: Untrained (`backbone-only`, `random-init`,
   `untrained`) heads are excluded by default and blocked from consensus; broken
   sub-modules (e.g., RECCE's untrained decoder) are audited at load and omitted.
3. **Fail-soft isolation**: Individual extractors, forensic analyzers, or detector
   subprocesses fail visibly as `notice` / neutral findings without losing the rest
   of the case.
4. **Examiner sovereignty**: Reports omit verdicts and aggregate scores; forensic
   analyzers emit observations with exact method caveats; evidence exports seal
   every file under `SHA256SUMS.txt`.

## Entrypoints

- `verity` (`src/verity/cli.py`) — unified CLI (`run`, `analyze`, `setup`, `doctor`,
  `serve`, `weights`, `airgap`, `export`, and stage subcommands).
- `verity-server` (`src/verity_api/main.py`) — FastAPI server serving both `/api/*`
  and the compiled React SPA cleanly from source or PyInstaller frozen binaries.
- `scripts/verity.py` / `verity-run` — thin delegation shim onto
  `verity.runtime.env_manager.main` for container and shell workflows.
- `docker/entrypoint.sh` — container entrypoint pulling missing trained weights when
  online before launching `server` or CLI jobs.

## Related pages

- [Examination pipeline](pipeline.md) — stage-by-stage orchestration flow.
- [Web API and investigator console](web-api-and-console.md) — FastAPI endpoints and React UX.
- [Explainability artifacts and on-disk formats](../reference/explainability-artifacts.md) — file contracts and schemas.
- [Deployment, configuration and air-gap operations](../operations/deployment-and-airgap.md) — `platformdirs`, `setup --airgap-ready`, and USB kits.
