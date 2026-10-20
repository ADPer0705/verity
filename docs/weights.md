# Checkpoints and the weights system

Verity never commits binary checkpoints to git. Instead, one YAML file —
`detectors/weights_manifest.yaml` — is the **single source of truth** for
every checkpoint: where it comes from, its size and SHA-256, its license, and
whether Verity may re-host it. This page explains how the system works; the
maintainer-facing publishing procedure is in [publishing.md](publishing.md);
the per-file license table is surfaced by `verity weights status` and
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

## Design rules

1. **Verify everything.** Every downloaded file is checked against its
   recorded SHA-256 and size before it is used. A mismatch means the file is
   deleted, and the run fails loudly — a silent substitution of weights is
   the nightmare scenario for an evidence pipeline.
2. **Respect licenses.** `redistributable: false` files (NPR, MLEP, DF40,
   Effort, XLS-R-SLS — no license statement, or upstream asks not to mirror)
   are *only ever fetched from the authors' own release points*. `true`
   files may live on the Verity Hugging Face mirror.
3. **Offline knob.** `VERITY_OFFLINE=1` forbids all downloads; the selector
   then only admits detectors whose checkpoint is already present and
   verified.
4. **No weight, no finding.** A detector without its trained checkpoint is
   excluded (with the reason recorded in `selection.json`), offered as
   research-mode via `--include-untrained`, or — if the YAML says
   `kind: random` — flagged as random-init in its own findings.

## Manifest layout

```yaml
version: 2

hub:                       # optional mirror (set in repo, or via VERITY_WEIGHTS_HUB)
  huggingface_repo: ""     # e.g. ADPer0705/verity-weights
  revision: main

licenses: {CC-BY-NC-4.0: url, MIT: url, ...}

sources:                   # attribution blocks per upstream project
  deepfakebench: {name, url, license, copyright, citation, note}
  aasist: {...}
  ...

files:                     # what can be fetched
  xception_best.pth:
    sha256: dd61…          # authoritative checksum
    size: 87763531
    source: deepfakebench  # attribution block key
    description: Xception (FF++ c23)
    origin: https://…      # authors' own URL (Release asset, HF resolve, Google Drive)
    redistributable: true  # may be re-hosted on the Verity hub
    hub_path: deepfakebench/xception_best.pth
  hrnetv2_w48_imagenet_pretrained.pth:
    …                      # files inside a zip add an `archive:` block with
                           # the archive checksum + member path (extracted + verified)

detectors:                 # which files each detector needs and where they land
  xception: [xception_best.pth]
  facexray: [{file: hrnetv2_w48_imagenet_pretrained.pth,
              dest: pretrained/hrnetv2_w48_imagenet_pretrained.pth}]

runtime_fetched:           # models the detector fetches itself (HF hub) —
  clip: openai/clip-vit-base-patch16 (HF, MIT)   # license tracking only
  …

no_checkpoint:             # honest statement per architecture with no checkpoint
  sbi: "No released trained checkpoint; random initialisation. …"
```

Files shared by several detectors (the Kinetics I3D backbone, consumed by
`i3d`/`ftcn`/`altfreezing`) are downloaded **once and hard-linked**, falling
back to a copy across filesystems.

## When a detector's weights come from where

| `weights` block in `detector.yaml` | Verity fetches | Tier |
|---|---|---|
| `kind: official` + `file` | the manifest (hub first if configured, then `origin`) | `trained` |
| `kind: external` + `file` | origin (authors' URL; license-aware re-hosting decision) | `trained` if `trained: true` |
| `kind: external`, no file | the detector fetches its model itself at first run (HF hub) — tracked under `runtime_fetched` | `trained` when proven, else `untrained` |
| `kind: backbone` + `file` | manifest (usually a non-deepfake backbone) | `backbone-only` |
| `kind: random` / nothing | nothing — random initialisation | `random-init` |

Tier assignment lives in `src/verity/detect/base_detector.py` and is derived
from this block; the selector's rule zero reads it.

## The CLI

```bash
verity weights status                     # table: file → source, license, present?, sha ok?
verity weights status --verify            # + re-hash every local file (slow)
verity weights pull [--trained| --all | --detector id[,id]] [--source auto|hub|origin] [--force]
verity weights verify                     # strict check of every mapped checkpoint
verity weights bundle --repo ADPer0705/verity-weights --out dist-weights
```

- `pull` is idempotent: verified-present files are skipped; corrupt local
  files are detected (checksum mismatch) and replaced.
- `--source hub` forces mirror-only (air-gapped setups point `hub` at a
  private mirror); `--source origin` forces the authors' URLs.
- `bundle` builds the HF upload directory offline: files laid out per
  `hub_path`, a model card, `NOTICE.md`, `LICENSE`, `manifest.json` and
  `sha256sums.txt`. It refuses files with no local verified copy ("missing")
  and never includes `redistributable: false` files.

## Weights locations

| Context | Where checkpoints live |
|---|---|
| Native dev (default) | `detectors/<id>/weights/…` (gitignored) |
| `VERITY_WEIGHTS_ROOT` set (containers) | `<root>/<id>/…`, with `detectors/<id>/weights` a symlink to it — downloads persist across container restarts |
| Air-gapped deployments | pre-seed the volume, or set `hub.huggingface_repo` to an internal mirror and run `verity weights pull --source hub` |

Runtime-fetched models (CLIP towers, VideoMAE, AntiDeepfake MMS …) go to the
standard HF cache (`HF_HOME`), shared across detector environments inside the
container.

## Current redistribution decisions (as of this writing)

| Checkpoint group | License | Verity mirror? |
|---|---|---|
| DeepfakeBench detector heads + I3D backbone | CC BY-NC 4.0 | yes |
| HRNet-W48 (Face X-ray backbone), Xception-Cadene (IID backbone) | MIT / BSD-3 | yes |
| UnivFD probe, GenD note, AASIST, Community Forensics, SAFE | MIT / Apache-2.0 | yes |
| LipForensics | MIT (Drive-hosted) | yes |
| NPR, MLEP | none stated | no — authors' URLs only |
| DF40, Effort | CC BY-NC 4.0, upstream asks not to re-host | no — Google Drive only |
| XLS-R-SLS ONNX export | HF card MIT, upstream none stated | no — HF `SpeechAntiSpoofingBenchmarks/XLSR-SLS` only |

DeepfakeBench checkpoints were trained on FaceForensics++ (c23), a
research-only dataset — the model card says so, and the project license is
CC BY-NC 4.0 accordingly.
