# Security policy

## Supported versions

Verity is pre-1.0. Security fixes are applied to the latest release on
`main`; older tags are not maintained.

| Version | Supported |
|---|---|
| 0.4.x | yes |
| < 0.4 | no |

## Reporting a vulnerability

**Do not open a public issue.** Use GitHub's private vulnerability reporting:
<https://github.com/ADPer0705/verity/security/advisories/new>.

Include, where possible:

- affected version/tag and deployment (container vs. native);
- the component (API, UI, pipeline, detector environment, weights fetch);
- reproduction steps or a proof of concept;
- impact assessment — especially anything that could serve arbitrary files,
  execute code, bypass checksum verification, or mutate evidence.

You can expect an acknowledgement within a few days. Please give us time to
ship a fix before public disclosure; we will credit reporters who wish to be
named.

## Security posture (what Verity already does)

- **Checksummed evidence.** Media ids are SHA-256 checksums; every
  checkpoint is verified against its manifest SHA-256 and size before use;
  `GET /api/media/{id}/verify` recomputes the media hash against the ingest
  record.
- **Confined file serving.** Artifact paths are resolved and rejected unless
  they stay inside the workspace/output/detector roots, and only pipeline
  file types are served.
- **Bounded uploads.** Uploads are capped (2 GiB) and streamed to a temp file,
  so the checksum covers exactly the bytes stored.
- **No implicit destructive actions.** Deletion requires `confirm=true` and
  never touches the original source media.
- **No silent downloads.** `VERITY_OFFLINE=1` forbids all network fetches;
  weights have an explicit source/checksum policy.
- **Detector isolation.** Each detector runs in its own uv environment, which
  contains the blast radius of a compromised detector dependency.

## Known gaps (by design or pending)

- There is **no authentication or user model**; the web console's role picker
  is presentation only. Do not expose Verity directly to an untrusted
  network — put it behind an authenticating reverse proxy.
- Jobs are in-memory and lost on restart; the workspace is the durable record.
- CORS defaults to localhost origins; configure `VERITY_CORS_ORIGINS`
  explicitly for any other deployment.
- The container may run with GPU access and bind mounts to host directories;
  treat the compose file as a template and restrict mounts for production.
- Detector subprocesses are trusted code in the repository. Do not run
  untrusted detector directories.

## Hardening recommendations

- Run the API bound to `127.0.0.1` or behind TLS + authentication.
- Mount `/input` read-only and keep `/output` and the workspace on
  access-controlled storage.
- Pin images by digest (not only tag) and verify the release artifacts.
- Keep `VERITY_OFFLINE=1` on air-gapped installations and pre-seed verified
  weights.
- Review `THIRD_PARTY_NOTICES.md` and the manifest before redistributing any
  output that embeds third-party checkpoints.
