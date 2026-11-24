---
type: subsystem
title: Weights and checkpoints
description: Single-source-of-truth weights manifest v2, SHA-256 integrity verification, redistributable versus origin-only licensing policy, the ADPer/verity-weights Hugging Face mirror, runtime cache warmup, and persistent storage layout.
tags: [weights, checkpoints, manifest, sha256, huggingface, licensing, airgap]
verified:
  - by: openwiki/0.9.0
    at: 2026-11-21T06:14:40.994Z
sources:
  - id: openwiki-source-d6b6ec4e1d20c1d7ac5ea76a
    resource: repo://detectors/weights_manifest.yaml
  - id: openwiki-source-184a2f452d290bbb1fb99bd9
    resource: repo://src/verity/paths.py
  - id: openwiki-source-ebbe77dc7a099dcf315b9a12
    resource: repo://src/verity/weights/bundle.py
  - id: openwiki-source-3f16ed36efcd342551f843ef
    resource: repo://src/verity/weights/fetch.py
  - id: openwiki-source-3b6208258bae1bc25b3e1366
    resource: repo://src/verity/weights/manifest.py
  - id: openwiki-source-6951631dfc297700ab3f73ac
    resource: repo://tests/test_packaging_and_export.py
generated: { by: "antigravity", at: "2026-11-21T06:14:40.994Z" }
---

# Weights and checkpoints

Verity never commits binary model checkpoints to git and never bakes gigabytes of
weights into its PyInstaller executable or `:slim` container image. Instead,
`detectors/weights_manifest.yaml` acts as the single authoritative inventory for
every checkpoint — recording origin URL, byte size, SHA-256 checksum, upstream
copyright/citation, license terms, and whether Verity is permitted to re-host a
mirror copy.

## Design principles & engineering rationale

1. **Verify everything before execution.** A silent weight substitution or
   truncated download would invalidate an evidence pipeline. `_fetch_file`
   (`src/verity/weights/fetch.py`) streams every download to a `.part` file,
   verifies exact byte size and SHA-256 hash against the manifest, deletes any
   mismatch immediately, and only renames on verification pass.
2. **Respect upstream authorship and licenses.** Re-hosting checkpoints without
   clear redistribution permission risks legal violations and misattribution. Every
   file defaults strictly to author attribution in both `NOTICE.md` and
   `ADPer/verity-weights` (`README.md` explicitly states that Verity mirrors
   upstream work for reproducible lab provisioning and trained none of the weights).
   Files with `redistributable: false` are never copied by `verity weights bundle`
   and always fetch solely from upstream origin URLs.
3. **Zero duplication via hard-links.** Shared checkpoints (such as
   `I3D_8x8_R50.pth` used across `i3d` and `altfreezing`) are downloaded once and
   hard-linked (`os.link`, falling back to `shutil.copy2` across filesystems).
4. **Upgrade-safe persistence.** In standalone binary (`platformdirs`) and
   Docker setups, `detectors/<id>/weights` is a symlink to
   `<weights_root>/<id>`, ensuring binary upgrades or detector script refreshes
   never delete verified multi-gigabyte checkpoints.

## Version-2 manifest architecture (`src/verity/weights/manifest.py`)

`load_manifest()` enforces `version >= 2` and parses seven sections from
`detectors/weights_manifest.yaml`:

| Section | Purpose |
|---|---|
| `hub` | Mirror configuration (`huggingface_repo: "ADPer/verity-weights"`, `revision: main`, overridable by `VERITY_WEIGHTS_HUB`) |
| `licenses` | Canonical license ID -> URL map |
| `sources` | Upstream attribution blocks (`name`, `url`, `license`, `copyright`, `citation`, `note`) |
| `files` | Per-file `sha256`, `size`, `source`, `origin`, `redistributable`, optional `hub_path`, `license` override, and `archive` member extraction spec |
| `detectors` | Per-detector target mapping (string or `{file, dest}`) under `detectors/<id>/weights/` |
| `runtime_fetched` | Documentation & license tracking for Hugging Face / Torch Hub / FAN models pulled at first run (`clip`, `gend`, `univfd`, `antideepfake_mms`, `aeroblade`, `realforensics`, etc.) |
| `no_checkpoint` | Explicit record of architectures shipped with random initialization |

For archive-hosted releases whose server-generated zip bytes vary on the fly (for
example LAA-Net's Dropbox folder download), `FileSpec.archive` treats the outer zip
checksum as advisory and verifies the extracted inner member's exact size and
SHA-256 before acceptance.

## Mirroring & licensing policy (`ADPer/verity-weights`)

`detectors/weights_manifest.yaml` configures `ADPer/verity-weights` as the mirror
for the **28 redistributable files (6.32 GB)**:

| Checkpoint group | License | Mirror policy |
|---|---|---|
| DeepfakeBench FF++ c23 detectors (`xception`, `efficientnetb4`, `f3net`, `spsl`, `srm`, `ffd`, `core`, `ucf`, `recce`, `capsule_net`, `meso4`, `meso4Inception`, `resnet34`) & `I3D_8x8_R50` | `CC-BY-NC-4.0` | Mirrored (`ADPer/verity-weights`) |
| Backbones (`hrnetv2_w48`, `xception-b5690688`) | `MIT` / `BSD-3-Clause` | Mirrored |
| `aasist`, `rawnet2`, `lipforensics`, `realforensics`, `syncnet`, `univfd`, `community_forensics`, `m2tr`, `vlaforge`, `m2f2_det`, `safe` | `MIT` / `Apache-2.0` | Mirrored |
| `forensics_adapter` | `CC-BY-NC-4.0` | Mirrored (embeds frozen MIT CLIP) |
| **11 origin-only files**: `NPR` (`npr`), `MLEP` (`mlep`), `FTCN` (`ftcn`), `D³` (`d3`), `ProDet` (`prodet`), `LipFD` (`lipfd`), `XLS-R+SLS` (`xlsr_sls`), `DF40` (`df40_clip`), `Effort` (`effort`), `SBI` (`sbi`), `LAA-Net` (`laa_net`) | `none-stated`, `SBI-academic`, `SnT-academic`, or origin-only `CC-BY-NC-4.0` | **Never mirrored** — fetched strictly from upstream Drive / GitHub / Dropbox / HF author URLs |

Note that `FileSpec.effective_license()` applies file-level license overrides cleanly
whenever an upstream repository's release metadata differs from its model card
(`xlsr-sls.onnx`). `Manifest.hub_url()` returns `None` for any `redistributable: false`
file, guaranteeing origin-only enforcement even with `--source hub`.

## CLI commands (`verity weights ...`)

- `verity weights pull [--trained | --all | --detector a,b] [--source auto|hub|origin] [--force]`
  — downloads using hub-first order (for `redistributable: true`), falls back to
  `origin`, skips verified files, replaces corrupt local files, and hard-links twins.
- `verity weights warmup [--trained | --all | --detector a,b] [--profile auto|cuda|cpu]`
  — invokes `env_manager.warmup_detectors` to execute each runtime-fetched model's
  `build_model()` once on CPU so Hugging Face / Torch Hub / FAN weights land in
  persistent `<cache_dir>/huggingface` and `<cache_dir>/torch` before disconnecting
  from the internet.
- `verity weights status [--verify]` / `verity weights verify` — reports `missing`,
  `present`, `verified`, or `CORRUPT`, exiting non-zero on corruption.
- `verity weights bundle --repo ADPer/verity-weights --out dist-weights`
  (`src/verity/weights/bundle.py`) — offline assembler that links only
  `redistributable: true` verified files, generates a full model card (`README.md`)
  explicitly crediting all upstream projects and noting excluded origin-only repos,
  writes `NOTICE.md`, `LICENSE`, `manifest.json`, `sha256sums.txt`, and an LFS
  `.gitattributes` covering `*.pth`, `*.pt`, `*.safetensors`, `*.onnx`, `*.model`,
  `*.tar`, and `*.bin`.

## Air-gap & container behavior

- **Standalone binary**: `verity setup --trained --airgap-ready` combines `pull`
  + `warmup` + `verify` and seals `offline: true` in `<config_dir>/config.yaml`.
  `verity airgap export` packages `weights/` alongside `cache-models.tar.gz` (HF +
  Torch Hub caches) for physical media transfer.
- **Docker**: `docker/entrypoint.sh` checks `VERITY_WEIGHTS_AUTO=1` (unless
  `VERITY_OFFLINE=1`) to pull missing trained checkpoints into `/weights` before
  starting `server` or analysis commands.

## Related pages

- [Detector runtime and isolation](detector-runtime.md) — `warmup_detectors` and
  offline environment variables.
- [Detector selection and trust tiers](detector-selection.md) — offline weight gating.
- [Deployment, configuration and air-gap operations](../operations/deployment-and-airgap.md) — `airgap export/import`.
- [Releasing, publishing and extending Verity](../operations/releasing-and-publishing.md) — publishing checkpoints to `ADPer/verity-weights`.
