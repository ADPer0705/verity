---
type: architecture
title: Examination pipeline
description: Stage-by-stage examination flow from ingest through extraction, forensics, selection, isolated detection, compilation, reporting, and chain-of-custody bundle export across CLI and HTTP API.
tags: [pipeline, orchestration, cli, api, workspace, export]
verified:
  - by: openwiki/0.9.0
    at: 2026-11-21T06:14:40.994Z
sources:
  - id: openwiki-source-4e5e7f4f3cd60b5ede40fbb5
    resource: repo://src/verity_api/routers/media.py
  - id: openwiki-source-27c03b24282b02c206c94fdd
    resource: repo://src/verity_api/service.py
  - id: openwiki-source-e4dc7d4c95cc68eb2f4b4656
    resource: repo://src/verity/cli.py
  - id: openwiki-source-6d817ce39f4036ea92df24bc
    resource: repo://src/verity/export.py
generated: { by: "antigravity", at: "2026-11-24T11:23:46.564Z" }
---

# Examination pipeline

An examination applies a fixed, deterministic sequence of stages to a single
content-addressed media item. Both the `verity` CLI (`src/verity/cli.py`) and
FastAPI service (`src/verity_api/service.py`) compose the exact same stage
objects — only progress reporting, concurrency/cancellation checks, and failure
isolation differ.

## End-to-end stage sequence

```text
ingest ─► extract ─► forensics ─► analyze ─► select ─► detect (58 envs) ─► compile ─► report ─► export
```

| Stage | Primary Class / Function | On-disk Output |
|---|---|---|
| 1. Ingest | `MediaIngestor` | `workspace/media/<id>.<ext>` (`0o444`), `workspace/media/.<id>.json` |
| 2. Extract | `FrameExtractor`, `FaceExtractor`, `MetadataExtractor`, `AudioExtractor` | `workspace/artifacts/<id>/{frames,faces,metadata,audio}/` |
| 3. Forensics | `ForensicsRunner` + 5 analyzers | `workspace/artifacts/<id>/forensics/forensics.json` + PNGs |
| 4. Analyze | `MediaAnalyzer` | `<output_dir>/<id>/analysis.json` |
| 5. Select | `DetectorSelector` | `<output_dir>/<id>/selection.json` |
| 6. Detect | `DetectorRunner` + `env_manager` | `workspace/output/<id>_<detector_id>/` (`result.json`, `finding.json`, logs, PNGs) |
| 7. Compile | `FindingsCompiler` | Sorted `FindingsCollection` in memory (reads `finding.json`s) |
| 8. Report | `ReportGenerator` | `<output_dir>/<id>/report.html` + optional `report.pdf` |
| 9. Export *(optional)* | `export_evidence_bundle` | `<output_dir>/<id>/verity-evidence-<stem>-<id>.zip` (`MANIFEST.json`, `SHA256SUMS.txt`) |

Notice the intentional storage split: per-detector outputs live under
`workspace/output/`, whereas case-level `analysis.json`, `selection.json`,
reports, and exported evidence zips live under configurable `output_dir` (which
defaults to `<workspace>/output`). In standalone single-binary mode
(`src/verity/paths.py`), `<workspace>` defaults cleanly to `<data_dir>/workspace`
(`~/.local/share/verity/workspace` or `$VERITY_HOME/data/workspace`).

## Content-addressed case identity

A case is not a mutable database row — it is the directory tree keyed by the
first 12 hex characters of the source file's SHA-256 checksum (`media_id`).
Ingesting identical bytes twice reuses the same working copy idempotently, while
modifying even a single bit creates a distinct case ID.

## The CLI orchestration path (`verity run` & stage subcommands)

`verity.cli.run` executes synchronously for a file or directory of sorted files:

1. **Ingest** via `MediaIngestor.ingest()`.
2. **Extract** using every configured extractor in `config/pipeline.yaml`
   (`frame`, `face`, `metadata`, `audio`).
3. **Forensics** unless `--no-forensics` is passed, summarizing severity counts.
4. **Analyze & Select** deterministically via `MediaAnalyzer` and
   `DetectorSelector` (honoring `--include-untrained`, `--detector` overrides,
   and `paths.is_offline()`). With `--detector a,b`, `selection.decisions` keeps
   all reasoning while rewriting `included` flags to reflect what ran.
5. **Run** selected detectors sequentially via `DetectorRunner.run()`, writing
   `finding.json` beside each `result.json`.
6. **Compile & Report**, saving `analysis.json`, `selection.json`, `report.html`,
   and (when `--format pdf`) `report.pdf`.

Individual stages are exposed cleanly on the CLI (`verity ingest`, `extract`,
`forensics`, `analyze`, `compile`, `report`, `export`) for resumable or scripted
investigations. Note that running `verity extract` automatically triggers
`_run_forensics` afterward so forensic artifacts stay synchronized.

## The API orchestration path (`run_full_pipeline`)

`verity_api.service.run_full_pipeline` wraps the same pipeline classes around a
`JobContext` after upload (`POST /api/media`), adding three workflow guarantees:

- **Named steps & SSE streaming**: declares `extract`, `forensics`, `analyze`,
  `select`, per-detector `detect:<id>`, `compile`, and `report`, streaming
  incremental logs over `GET /api/jobs/{id}/events`.
- **Cooperative cancellation**: checks `ctx.raise_if_cancelled()` between
  extractors and between detectors.
- **Stage-level fail-soft containment**: `run_extract` catches individual
  extractor errors (`failed_extractors`) and `run_detectors` catches individual
  detector crashes as failed steps while continuing the remaining panel. Even if
  zero detectors are eligible, the job completes with forensics, `selection.json`,
  and an explicit report showing nothing ran.

Every stage can also be launched standalone via HTTP (`POST /api/extract`,
`POST /api/forensics`, `POST /api/analyze`, `POST /api/compile`, `POST /api/report`),
and any completed or partial case can be sealed into a verifiable chain-of-custody
archive via `GET /api/media/{id}/export` or `verity export --media-id <id>`.

## Ordering guarantees

1. Selection depends on `analysis.json`.
2. Isolated detectors depend on staged `artifacts/` copies.
3. Compilation depends on every `<media_id>_*/finding.json`.
4. Reports depend on `analysis.json`, `selection.json`, `forensics.json`, and
   `FindingsCollection` filtered strictly to `selection`.
5. Exports hash and seal every artifact under `MANIFEST.json` + `SHA256SUMS.txt`.

Within every stage, media files, detectors, extractors, analyzers, and findings are
iterated in explicit sorted order.

## Related pages

- [Architecture overview](overview.md) — package boundaries and `platformdirs` storage.
- [Ingest and media analysis](ingest-and-analysis.md) — SHA-256 identity and `ffprobe`/Haar scan.
- [Artifact extraction](artifact-extraction.md) — extractor contracts and staging.
- [Model-free forensic analyses](forensics.md) — five deterministic analyzers.
- [Detector selection and trust tiers](detector-selection.md) — ordered selection rules.
- [Report generation](reporting.md) — HTML/PDF rendering and exclusion rules.
- [Web API and investigator console](web-api-and-console.md) — HTTP endpoints and SSE jobs.
