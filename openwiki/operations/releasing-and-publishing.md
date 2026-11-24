---
type: operations
title: Releasing, publishing and extending Verity
description: Maintainer release runbook and distribution engineering rationale covering standalone PyInstaller binary builds on ubuntu-22.04, Hugging Face checkpoint mirroring, GHCR container images, deploy bundles, version synchronization, rollback, testing, and adding new detectors.
tags: [releasing, publishing, pyinstaller, glibc, ci, ghcr, huggingface, contributing]
verified:
  - by: openwiki/0.9.0
    at: 2026-11-21T06:14:40.994Z
sources:
  - id: openwiki-source-164e2da859b5277df81c7d94
    resource: repo://.github/workflows/ci.yml
  - id: openwiki-source-4d1d392666be6dfdd7a91a2e
    resource: repo://.github/workflows/release.yml
  - id: openwiki-source-f317ee207e1653d2033c81a4
    resource: repo://CONTRIBUTING.md
  - id: openwiki-source-bb1ebe868e35e9e500714501
    resource: repo://Dockerfile
  - id: openwiki-source-fe6aea552edf9d759e727b95
    resource: repo://scripts/build_binary.py
  - id: openwiki-source-d19b5b99402b0a05de7a8907
    resource: repo://scripts/build_deploy_bundle.py
generated: { by: "antigravity", at: "2026-11-21T06:14:40.994Z" }
---

# Releasing, publishing and extending Verity

Verity distributes across three coordinated channels, each matched to its
artifact class:

| Artifact | Host | Identifier | Engineering Rationale |
|---|---|---|---|
| Standalone single binary (`verity-linux-x86_64`), checksums, `install.sh`, & Compose bundle | GitHub Releases | `ADPer0705/verity` | Versioned binary + checksum assets with CDN downloads; avoids PyPI's single-env constraints |
| Container images | GitHub Container Registry | `ghcr.io/adper0705/verity` | Native `GITHUB_TOKEN` pushes, public pulls with zero rate limits, semver + `:slim` tagging |
| Redistributable checkpoints (28 files, 6.32 GB) | Hugging Face Hub | `ADPer/verity-weights` | Native Git LFS handling for >2 GB sets, model cards with CC BY-NC attribution, `resolve/<rev>/<path>` CDN URLs |

PyPI (`sdist`/`wheel`) is intentionally skipped: labs install a single binary +
`install.sh`, letting Verity itself manage isolated `uv` environments per detector.

## Standalone single binary & glibc portability

### Why `ubuntu-22.04` (glibc 2.35) matters

PyInstaller bundles Python interpreter binaries linked dynamically against the build
host's GNU C Library (`glibc`). Building a release binary on a rolling/cutting-edge
distribution (for example Fedora with `glibc 2.44`) produces an executable that
fails immediately with `GLIBC_2.xx not found` on LTS workstations.
`.github/workflows/release.yml` pins the `binary` job to **`runs-on: ubuntu-22.04`
(`glibc 2.35`)**, producing binaries portable across Ubuntu 22.04/24.04, Debian
12+, RHEL 9+, Fedora 38+, and newer systems.

### What `scripts/build_binary.py` embeds

`scripts/build_binary.py` compiles `packaging/verity_entry.py` via PyInstaller
(`--onefile`) embedding:

- Python 3.12 runtime + `verity` + `verity_api` + Uvicorn/FastAPI/WeasyPrint;
- Pre-built React Web Console SPA (`src/verity_ui/dist`);
- Source code, `detector.yaml`, `pyproject.toml`, `output_parser.py`, `weights_manifest.yaml`,
  and `weights/README.md` for all 58 detectors (explicitly excluding all checkpoint binaries);
- OpenCV Haar cascades (`--collect-data cv2`).

Two subtle build/release engineering choices keep builds reliable:

1. **Disk-backed temporary directory**: `build_binary.py` creates its temp
   working directory inside `dist-release/` (`dir=str(out_dir)`) instead of `/tmp`,
   preventing `objcopy` / extraction failures on systems with small RAM-backed `/tmp`
   tmpfs mounts.
2. **Collision-free checksum filenames**: `build_binary.py` emits
   `verity-linux-x86_64.sha256` and `verity-<version>-linux-x86_64.sha256`, while
   `scripts/build_deploy_bundle.py` emits `SHA256SUMS` and `SHA256SUMS-<version>`.
   Separate release jobs can attach all assets to the same GitHub Release without
   overwriting each other's checksum files.

Build and smoke-test locally:

```bash
uv sync --frozen --extra api --extra build
uv run python scripts/build_binary.py --version 0.6.0 --out dist-release
(cd dist-release && sha256sum -c verity-linux-x86_64.sha256)
VERITY_HOME=/tmp/verity-smoke ./dist-release/verity-linux-x86_64 --version
```

## Hugging Face checkpoint mirror (`ADPer/verity-weights`)

`verity weights bundle` (`src/verity/weights/bundle.py`) lays out an attributed,
offline-verified mirror containing **only** `redistributable: true` files
(28 checkpoints, ~6.32 GB), a generated `README.md` crediting upstream authors and
clearly stating zero checkpoints were trained by Verity's maintainer, `NOTICE.md`,
`LICENSE`, `manifest.json`, `sha256sums.txt`, and LFS `.gitattributes`
(`*.pth`, `*.pt`, `*.safetensors`, `*.onnx`, `*.model`, `*.tar`, `*.bin`):

```bash
uv run verity weights pull --all
uv run verity weights bundle --repo ADPer/verity-weights --out dist-weights
hf repos create ADPer/verity-weights --type model --public --exist-ok
hf upload ADPer/verity-weights dist-weights/verity-weights . --repo-type model \
  --commit-message "Verity v0.6.0 redistributable checkpoints mirror"
```

Verify mirror resolution cleanly:

```bash
VERITY_WEIGHTS_HUB=ADPer/verity-weights \
  uv run verity weights pull --detector univfd --force --source hub
VERITY_OFFLINE=1 uv run verity weights verify
```

The 11 origin-only / un redistributable checkpoints (`npr`, `mlep`, `df40_clip`,
`effort`, `xlsr_sls`, `ftcn`, `d3`, `prodet`, `lipfd`, `sbi`, `laa_net`) are never
copied into `dist-weights/` and continue resolving exclusively from upstream URLs.

## Containers (GHCR `:slim` vs manual `:cuda`/`:cpu`)

`Dockerfile` defines a multi-stage build:

- **`slim` (automated by CI on tags `v*`)**: bundles orchestrator, FastAPI API,
  SPA, `ffmpeg`, Pango/Cairo, and detector definitions; detector virtualenvs build on
  first use or via `deploy/install.sh`. Built automatically in `.github/workflows/release.yml`
  because GitHub hosted runners lack disk for full pre-baked images.
- **`full` (`--target full --build-arg VERITY_PROFILE=cuda|cpu`, manual)**:
  pre-bakes every trained detector environment inside `/opt/verity/envs` (10–20 GB)
  for organizations that want instant-boot offline containers. Build and push from a
  high-disk workstation (`>= 60 GB` free).

## Release process & version synchronization

Before pushing a `vX.Y.Z` tag, keep the same version string synchronized across:

1. `pyproject.toml` (`[project] version`)
2. `src/verity/__init__.py` (`__version__`)
3. `src/verity_ui/package.json` (+ `package-lock.json`)
4. `CITATION.cff` and `CHANGELOG.md`
5. `src/verity_api/main.py` (reads `verity.__version__` dynamically)

### Release checklist

1. `uv run pytest -q` green; `npm --prefix src/verity_ui run typecheck && npm run build` green.
2. `uv run verity weights status --verify` clean (`0` corrupt).
3. Hugging Face `ADPer/verity-weights` mirror uploaded and public resolve verified.
4. `scripts/build_deploy_bundle.py` & `bash -n install.sh deploy/install.sh deploy/doctor.sh` green.
5. CI workflow (`.github/workflows/ci.yml`: `test`, `ui`, `image-slim`) passes on `main`.
6. Tag `vX.Y.Z` and push; verify `.github/workflows/release.yml` (`slim`, `binary`
   on `ubuntu-22.04`, `deploy-bundle`) attaches `verity-linux-x86_64`,
   `verity-<version>-linux-x86_64`, `.sha256` files, `install.sh`,
   `verity-deploy-*.tar.gz`, and `SHA256SUMS*`.

## Rollback

- **Weights**: Revert `hub.revision` in `detectors/weights_manifest.yaml` or
  restore the prior Hugging Face commit; Verity's SHA-256 check refuses mismatches.
- **Container & Binary**: Re-tag or roll back to the previous release semver asset;
  existing `<data_dir>/weights` and Docker volumes persist independently of binaries.

## Extending Verity & testing strategy (`CONTRIBUTING.md`)

To add a new detector port:

1. Create `detectors/<id>/` with `detector.yaml` (capabilities, `input_mode`,
   `targets`, honest `weights` tier), `pyproject.toml`, `model.py`
   (`build_model`, normalized `[real, fake]` logits), `inference.py` delegating to
   `common.engine.run_detector`, optional `output_parser.py`, and `weights/README.md`.
2. Add checkpoint metadata to `detectors/weights_manifest.yaml` (truthfully setting
   `redistributable: false` unless upstream license explicitly allows redistribution).
3. Regenerate notices (`uv run python scripts/gen_third_party_notices.py`) and
   verify `uv run python scripts/gen_detector_table.py`.
4. Add lightweight unit coverage in `tests/` (registry parsing, selection decisions,
   output mapping, and deterministic reports) — unit tests run cleanly without GPU or
   heavy weight downloads. Heavy per-detector runtime profiling is handled separately
   via `scripts/run_detector_matrix.py` (`output/matrix_runs.md`).

## Related pages

- [Deployment, configuration and air-gap operations](deployment-and-airgap.md) — lab
  setup and USB transfer workflows.
- [Weights and checkpoints](../architecture/weights-and-checkpoints.md) — manifest schema
  and license gate.
- [Detector runtime and isolation](../architecture/detector-runtime.md) — detector
  package contract.
