---
type: architecture
title: Web API and investigator console
description: FastAPI HTTP service, SSE thread-pool job model, endpoint map, environment readiness probes, frozen-binary SPA serving, security boundaries, and the redesigned React/Mantine investigator web console.
tags: [api, fastapi, react, mantine, jobs, sse, console]
verified:
  - by: openwiki/0.9.0
    at: 2026-11-21T06:14:40.994Z
sources:
  - id: openwiki-source-392cb142cea6fef3de3d5452
    resource: repo://src/verity_api/config.py
  - id: openwiki-source-256a301831e925392a4a4725
    resource: repo://src/verity_api/jobs.py
  - id: openwiki-source-8fd3e8c7a518a6182d2aef95
    resource: repo://src/verity_api/main.py
  - id: openwiki-source-50e32edfedf5ed35b14b1fcc
    resource: repo://src/verity_api/routers/jobs.py
  - id: openwiki-source-da2b46acf7b47bb969866230
    resource: repo://src/verity_ui/src/components/FindingTables.tsx
  - id: openwiki-source-76bdf262853ccb4dab3106ae
    resource: repo://src/verity_ui/src/routes/CaseDetail.tsx
  - id: openwiki-source-ba6ee0d18b6931b109031a39
    resource: repo://src/verity_ui/src/routes/tabs/ForensicsTab.tsx
generated: { by: "antigravity", at: "2026-11-21T06:14:40.994Z" }
---

# Web API and investigator console

The `verity` CLI remains the reference implementation of the pipeline. Two thin
presentation layers sit cleanly above it without duplicating detection or scoring
logic:

- **`src/verity_api/`** — a FastAPI server and bounded thread-pool job manager
  driving the same `verity.*` stage objects used by the CLI.
- **`src/verity_ui/`** — a React / TypeScript / Mantine web console compiled via
  Vite into static assets (`dist/`) and served by FastAPI from the exact same port,
  so both source installs and the standalone single executable expose a unified
  one-port workstation console (`http://127.0.0.1:8000`).

## Background job architecture (`JobManager`)

Media probing, Haar face scans, and running up to 39 evidence-grade detector
subprocesses can take minutes to tens of minutes. Instead of blocking HTTP requests,
every mutating pipeline endpoint returns `202 Accepted` with a job descriptor:

```text
POST /api/run            ->  202 { id, status: "pending", steps: [...] }
GET  /api/jobs/{id}      ->  poll snapshot (?since=<seq> for incremental logs)
GET  /api/jobs/{id}/events -> server-sent events (logs + step transitions)
POST /api/jobs/{id}/cancel -> sets cooperative cancel_event (stops between detectors)
```

`JobManager` (`src/verity_api/jobs.py`) intentionally holds up to `250` recent jobs
in memory using a bounded worker pool (`max_workers=2` by default: heavy detector
runs serialize to protect 8 GB VRAM laptops, while fast analyze/report queries stay
responsive). If the server restarts, the ephemeral job queue clears while the durable
examination record on disk stays intact. Two concurrent non-terminal jobs for the
same `media_id` are rejected with `409 Conflict` (`JobConflict`) so runs cannot race
on shared case directories.

## Endpoint map (1:1 parity with the CLI)

| CLI Command | HTTP Endpoint | Semantics |
|---|---|---|
| `verity ingest` | `POST /api/media` | Streamed multipart ingest (2 GiB cap, 413 overflow) |
| `verity extract` | `POST /api/extract` | Extract frames, faces, metadata, audio (fail-soft per extractor) |
| `verity forensics` | `POST /api/forensics` | Run all applicable model-free forensic analyzers on demand |
| `verity analyze` | `POST /api/analyze` | Characterize media + compute `selection.json` |
| `verity run` | `POST /api/run` | End-to-end async examination (`extract` -> `forensics` -> `analyze` -> `select` -> `detect` -> `compile` -> `report`) |
| `verity compile` | `POST /api/compile` | Recompile findings + panel consensus |
| `verity report` | `POST /api/report` | Generate deterministic `report.html` + optional `report.pdf` |
| `verity export` | `GET /api/media/{id}/export` | Build & download verifiable `.zip` Evidence Preservation Bundle (`MANIFEST.json` + `SHA256SUMS.txt`) |
| `verity setup` | `POST /api/detectors/setup` | Provision detector virtual environments via `env_manager` |

Case inspection & artifact endpoints (`src/verity_api/routers/media.py`, `system.py`,
`detectors.py`):

- `GET /api/system` / `GET /api/health` — effective settings and live probe status.
- `GET /api/detectors` / `GET /api/detectors/coverage` — fresh registry scan per
  request (`build_registry`) so newly added checkpoints appear without restarting.
- `GET /api/media` & `GET /api/media/{id}` — case list and full case record
  (media info, analysis, selection decisions, `finding_to_dict`, consensus, extracted
  artifacts, `forensics`, report availability).
- `GET /api/media/{id}/file`, `/report`, `/detectors/{det}/assets|logs|raw`, and
  `GET /api/media/{id}/files?path=...` (path-confined & suffix-checked).
- `POST /api/media/{id}/verify` — recomputes working copy SHA-256 on demand.
- `DELETE /api/media/{id}?confirm=true` — deletes case artifacts only (never the
  original source media).

Interactive OpenAPI documentation is always served at `/api/docs`.

## Frozen-binary & configuration engineering

`verity_api.main.main()` passes the ASGI `app` object directly to `uvicorn.run(app,
...)` whenever `VERITY_RELOAD` is disabled (`src/verity_api/main.py`). Passing the
object instead of the `"verity_api.main:app"` import string ensures PyInstaller
single-file executables boot FastAPI cleanly without dynamic module resolution errors.

`verity_api.config` (`src/verity_api/config.py`) resolves paths via
`verity.paths.is_standalone_mode()`: in standalone/binary mode it uses `platformdirs`
(`~/.local/share/verity`, etc.) and locates the SPA via `paths.ui_dist_dir()`
(checking `$VERITY_UI_DIST`, bundled `verity_ui_dist`, `src/verity_ui/dist`, and
`<data_dir>/ui_dist`); in repository development mode, relative paths resolve
against `REPO_ROOT` rather than process CWD so server behavior never depends on
where uvicorn launched. Settings edits submitted from the UI mutate process state
only and never overwrite checked-in `config/pipeline.yaml`.

## Proactive environment readiness probing

`probe_environment()` (`src/verity_api/config.py`) inspects the host before jobs run
so the UI can explain missing capabilities clearly rather than failing inside a
worker:

- `detectors_dir_present` & `detector_count` (`detector.yaml` scan);
- `detector_runner` (external `$VERITY_RUNNER`, `scripts/verity.py`, or builtin
  `verity.runtime.env_manager`);
- `uv` (`find_uv()`, checking private `<data_dir>/bin` first, then `PATH`),
  `ffmpeg`, `ffprobe`, OpenCV (`cv2`), and `hw_profile`;
- WeasyPrint — `_weasyprint_status()` imports `weasyprint` cleanly rather than relying
  solely on `find_spec`, catching missing native Pango/GObject libraries up front.

## Investigator Web Console (`src/verity_ui/`)

`CaseDetail.tsx` structures every case into eight URL-synchronized tabs (`?tab=...`):

1. **Overview (`OverviewTab.tsx`)**: Consensus panel + panel-split limitations, live
   SHA-256 integrity check button, one-click **Export evidence bundle (.zip)**,
   media preview, examination state metrics, and the **Detector probability spectrum**
   deep-linking straight into any detector's explainability exhibits.
2. **Forensics (`ForensicsTab.tsx`)**: Dedicated model-free inspection tab with
   severity filtering (`warning` / `notice` / `info`), inline key-value `EvidenceChips`,
   verbatim analyzer method caveats, high-resolution diagnostic galleries (ELA, block
   grid, noise residual/variance, 2D FFT spectrum, temporal timeline, log-mel
   spectrogram), and an on-demand **Run forensics** button (`POST /api/forensics`).
3. **Findings (`FindingsTab.tsx`)** & **Evidence & XAI (`DetectorEvidence.tsx`)**:
   - Displays `AuxSignalsTable` (`aux_aggregate` across secondary branches, LPIPS VAEs,
     or SyncNet AV offset/LSE-C confidence) and `RegionEnergyTable`;
   - Galleries sort units by descending `fake_prob` (`full_frames`, `faces`, `frames`,
     `attention_maps`, `reconstructions`) plus temporal/audio `timeline.png`;
   - **`PerFaceTable` (`FindingTables.tsx`)** renders modality-aware schemas automatically:
     time spans in seconds (`start_s`–`end_s`) for `audio`/`av_clip` windows, dimensions
     (`w × h`) for whole frames/clips, and face bounding boxes + peak MediaPipe facial
     regions for face detectors.
4. **Analysis**, **Selection** (inclusion & exclusion rationale for all 58 ports),
   **Report** (embedded HTML/PDF preview + evidence bundle export), and **Activity**
   (live SSE job console).

## Related pages

- [Architecture overview](overview.md) — package boundaries and dependency rules.
- [Examination pipeline](pipeline.md) — CLI vs API orchestration parity.
- [Findings, consensus and aggregation](findings-and-consensus.md) — `consensus()` and
  disagreement handling.
- [Security, integrity and chain of custody](security-and-integrity.md) — confined file
  serving, upload limits, and loopback security posture.
