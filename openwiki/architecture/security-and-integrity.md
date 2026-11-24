---
type: architecture
title: Security, integrity and chain of custody
description: Content-addressed media identity, read-only working copies, integrity verification, deterministic chain-of-custody zip export bundles, offline sealing, confined file serving, and authentication boundaries.
tags: [security, integrity, checksums, custody, airgap, deployment]
verified:
  - by: openwiki/0.9.0
    at: 2026-11-21T06:14:40.994Z
sources:
  - id: openwiki-source-10066bffec1f610f3650c047
    resource: repo://deploy/install.sh
  - id: openwiki-source-b79fbbd921df689b4bbdc82f
    resource: repo://docker-compose.yml
  - id: openwiki-source-3a44815832a872f4778f822b
    resource: repo://SECURITY.md
  - id: openwiki-source-fa2c5a103d433f485a60c655
    resource: repo://src/verity_api/files.py
  - id: openwiki-source-256a301831e925392a4a4725
    resource: repo://src/verity_api/jobs.py
  - id: openwiki-source-4e5e7f4f3cd60b5ede40fbb5
    resource: repo://src/verity_api/routers/media.py
  - id: openwiki-source-27c03b24282b02c206c94fdd
    resource: repo://src/verity_api/service.py
  - id: openwiki-source-e4dc7d4c95cc68eb2f4b4656
    resource: repo://src/verity/cli.py
  - id: openwiki-source-61ad6b16674ac0599a5a1e59
    resource: repo://src/verity/detect/detector_runner.py
  - id: openwiki-source-6d817ce39f4036ea92df24bc
    resource: repo://src/verity/export.py
  - id: openwiki-source-c914ce3e5aa30a558c56c3cb
    resource: repo://src/verity/ingest/media_ingestor.py
  - id: openwiki-source-184a2f452d290bbb1fb99bd9
    resource: repo://src/verity/paths.py
  - id: openwiki-source-3b6208258bae1bc25b3e1366
    resource: repo://src/verity/weights/manifest.py
  - id: openwiki-source-6951631dfc297700ab3f73ac
    resource: repo://tests/test_packaging_and_export.py
generated: { by: "antigravity", at: "2026-11-24T11:23:46.564Z" }
---

# Security, integrity and chain of custody

An evidence analysis workbench must be completely precise about what it can and
cannot guarantee. Verity guarantees that every artifact read by its detectors,
forensic analyzers, web console, and reports is byte-identical to the admitted
working copy — and packages complete case records with standard `SHA256SUMS.txt`
proofs for external custody transfer. It does **not** certify pre-ingest capture
chain of custody, and it delegates user authentication to loopback isolation or a
reverse proxy.

## Content-addressed identity & read-only admission

`MediaIngestor.ingest` (`src/verity/ingest/media_ingestor.py`) streams every
incoming file in 64 KiB chunks to compute its full SHA-256 checksum, derives
`media_id = checksum[:12]`, copies the bytes once into
`workspace/media/<media_id><suffix>`, marks the working copy read-only
(`chmod 0o444`), and records `original_path`, `size_bytes`, full `checksum`, and
UTC ingest timestamp in `.{media_id}.json`.

When `DetectorRunner._prepare_input_dir` launches a detector subprocess, the
detector receives an ephemeral *copy* of the working copy inside a temp
directory — never the `workspace/media/` file itself — so even a malfunctioning
detector cannot mutate admitted evidence.

## Live integrity verification

`verify_integrity` (`src/verity_api/service.py`) backing `POST /api/media/{id}/verify`
re-hashes the read-only working copy on demand and compares both size and SHA-256
against the ingest record (plus the original source path if still reachable on
disk). This allows an investigator to demonstrate integrity live during review.

## Chain-of-custody Evidence Preservation Bundle (`verity export`)

For court exhibits, cross-team peer review, and immutable offline archiving,
`export_evidence_bundle` (`src/verity/export.py`), exposed via
`verity export --media-id <id>`, `GET /api/media/{id}/export`, and the Web UI's
**Export evidence bundle (.zip)** action:

1. Verifies the read-only working copy's byte size and SHA-256 against
   `.{media_id}.json` before archiving (exiting non-zero via CLI on mismatch).
2. Packages cleanly organized files:
   - `media/<original_name>` and `media/ingest_record.json`
   - `report/{report.html,report.pdf,analysis.json,selection.json}`
   - `forensics/*` (JSON and PNG diagnostic figures)
   - `detectors/<id>/*` (`result.json`, `finding.json`, `faces/`, `frames/`,
     `full_frames/`, `attention_maps/`, `reconstructions/`, `timeline.png`, logs)
3. Writes `MANIFEST.json` (`schema: "verity-evidence-bundle/1"`, media hashes,
   offline flag, findings summary, forensics counts, and exact per-file metadata)
   plus `SHA256SUMS.txt` sorted lexicographically across every archived entry and
   `MANIFEST.json` itself.

An external auditor can verify every byte inside an unpacked archive using plain
`sha256sum -c SHA256SUMS.txt` (`tests/test_packaging_and_export.py`), with zero
dependency on Verity software.

## Offline air-gap seal (`config.yaml` & `VERITY_OFFLINE`)

`src/verity/paths.py` enforces air-gap discipline via `is_offline()`:

- Setting `VERITY_OFFLINE=1` immediately overrides runtime behavior.
- Running `verity setup --trained --airgap-ready` (or `verity airgap import`)
  verifies local checkpoints and persists `offline: true` inside
  `<config_dir>/config.yaml` via `save_user_config()`.
- Once sealed, `DetectorSelector` excludes uncached external models,
  `env_manager` forces `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`,
  `DIFFUSERS_OFFLINE=1`, `UV_OFFLINE=1`, and `UV_NO_PYTHON_DOWNLOADS=1` on every
  subprocess, and the API never initiates outbound model fetches.

## Confined file serving, upload bounds, and safe deletion

- **Path confinement (`src/verity_api/files.py`)**: `resolve_within_roots()`
  resolves query paths with `strict=True` (following symlinks first) and returns
  `403` outside `Settings.readable_roots` (`workspace`, `output_dir`,
  `detectors_dir`) and `415` for any suffix not in `ALLOWED_SUFFIXES` (blocking
  arbitrary filesystem reads).
- **Streaming upload cap**: `POST /api/media` streams multipart uploads to disk
  in 1 MiB chunks up to `MAX_UPLOAD_BYTES` (2 GiB, else `413`).
- **Explicit deletion**: `DELETE /api/media/{id}` requires `?confirm=true`,
  removes only workspace/output artifacts, and never touches the original
  source file outside the workspace.
- **Per-case lock**: `JobManager` raises `409 Conflict` if a non-terminal job
  already targets the same `media_id`.

## Authentication and workstation exposure

Because Verity is built for single-investigator workstations and air-gapped labs,
the API ships without authentication or user roles (`docker-compose.yml` and
`verity serve` bind to `127.0.0.1:8000` by default, and `deploy/install.sh`
refuses non-loopback binds without `--allow-network`). Remote operators should
use SSH port forwarding (`ssh -L 8000:localhost:8000 user@host`) or place TLS +
authentication at a reverse proxy (`Caddy` / `nginx`).

## Related pages

- [Ingest and media analysis](ingest-and-analysis.md) — initial SHA-256 admission.
- [Weights and checkpoints](weights-and-checkpoints.md) — SHA-256 gate on model weights.
- [Web API and investigator console](web-api-and-console.md) — endpoints and integrity checks.
- [Deployment, configuration and air-gap operations](../operations/deployment-and-airgap.md) — `setup --airgap-ready` and USB transfer kits.
