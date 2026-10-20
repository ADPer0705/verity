# Web UI and HTTP API

The `verity` CLI remains the reference implementation of the pipeline. Two
packages sit on top of it:

- **`src/verity_api/`** — FastAPI service wrapping the same pipeline objects
  the CLI drives. It contains no detection logic of its own.
- **`src/verity_ui/`** — React front end (Mantine components, Inter typeface; no vendor design system).

## Why a job API

Every meaningful pipeline operation is long-running and blocking: ffprobe,
OpenCV face scans, and detector subprocesses that can run for tens of minutes.
So each operation returns a **job** immediately and runs on a background
thread, reporting named steps and streaming log lines.

```
POST /api/run       ->  202  { id: "a390839f288f", status: "pending", steps: [...] }
GET  /api/jobs/{id}      poll, with ?since=<seq> for incremental logs
GET  /api/jobs/{id}/events   server-sent events
POST /api/jobs/{id}/cancel   cooperative — the current detector finishes first
```

Jobs live in memory and are deliberately lost on restart. The durable record
is what each stage wrote to the workspace, and every stage writes as it
completes — so a run stopped part-way leaves everything produced up to that
point intact and readable.

## Endpoint map

| CLI command | Endpoint |
|---|---|
| `verity ingest` | `POST /api/media` (multipart) |
| `verity extract` | `POST /api/extract` |
| `verity analyze` | `POST /api/analyze` |
| `verity run` | `POST /api/run` |
| `verity compile` | `POST /api/compile` |
| `verity report` | `POST /api/report` |
| `verity setup` | `POST /api/detectors/setup` |

Read endpoints:

| Endpoint | Returns |
|---|---|
| `GET /api/system` | Effective settings, environment probe, readiness with per-capability blockers |
| `GET /api/detectors` | Registry with capabilities and weight availability |
| `GET /api/media` | Case list with consensus, detector counts, report state |
| `GET /api/media/{id}` | Full case: media, analysis, selection, findings, artifacts, reports |
| `GET /api/media/{id}/file` | The read-only working copy (preview or download) |
| `GET /api/media/{id}/report?format=pdf\|html` | Generated report |
| `GET /api/media/{id}/detectors/{det}/assets\|logs\|raw` | Explainability images, stdout/stderr, verbatim JSON |
| `POST /api/media/{id}/verify` | Recomputes SHA-256 and compares against the ingest record |

Interactive docs: `/api/docs`.

## Configuration

`verity_api.config` mirrors `verity.cli`'s resolution — `config/pipeline.yaml`
then `VERITY_*` environment variables — with one deliberate difference:
**relative paths resolve against the repository root**, not the process working
directory, so a long-lived server does not change meaning depending on where
uvicorn was launched.

Settings are editable at runtime from the Settings screen. Those edits apply to
the running process only and are **not** written back to `pipeline.yaml`;
silently rewriting a checked-in pipeline configuration from a browser is not
something a forensics tool should do.

## Environment readiness

`GET /api/system` probes what is actually usable and returns a blocker per
missing capability, so the UI can explain a gap up front instead of letting a
job fail with an opaque error:

- detectors directory present, and how many `detector.yaml` definitions
- detector runtime (`scripts/verity.py` / `verity-run` / `$VERITY_RUNNER`)
- `uv` — required to provision detector environments
- `ffprobe` / `ffmpeg` — duration, resolution, FPS, and audio extraction
- OpenCV — face scanning
- WeasyPrint — **imported, not merely located**, because it binds to native
  Pango/GObject libraries at import time and is frequently installable but
  unusable on Windows

## Consensus, and why it is presented the way it is

`verity_api.service.consensus()` aggregates independent findings without
overriding any of them. Two rules matter:

1. **A divided panel resolves to `inconclusive`**, no matter how confident the
   individual detectors are. When a quarter or more of the detectors dissent,
   no single band is honest.
2. **The band is never stronger than the agreement behind it.** `reliability()`
   downgrades results that rest on few detectors or a split panel, and the UI
   shows that downgrade next to the headline.

The agreement bar is always shown. This follows the project's premise
— findings are preserved with their provenance rather than collapsed into one
score — and the real sample report is exactly why: seven detectors, with mean
fake probabilities from 0.0000 to 0.9811 on the same video.

The aggregate is presented for orientation. The per-detector findings are the
record, and the exported report still renders no verdict at all.

## Security posture

- **File serving is confined.** Artifacts are served by absolute path through
  `GET /api/media/{id}/files?path=…`, which resolves the path and rejects
  anything outside the workspace, output and detector roots (403), plus any
  extension the pipeline does not emit (415). An evidence tool must not become
  an arbitrary-file reader via a crafted query string.
- **Uploads are bounded** at 2 GiB and streamed to a temp file, so the checksum
  is computed over exactly the bytes that land on disk.
- **Deletion requires `confirm=true`** and never touches the original source
  file — only the workspace copy and derived data.
- **Profiles are presentation, not authorisation.** Both reach the same API.
  Restricting the investigator surface has to be enforced server-side.

## Known gaps

- The role picker is not authentication; there is no user model.
- Job history is per-process and not persisted.
- CORS defaults to localhost dev origins; set `VERITY_CORS_ORIGINS` for any
  other deployment.
