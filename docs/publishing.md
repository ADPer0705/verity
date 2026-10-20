# Publishing Verity

This is the maintainer runbook for taking Verity from a local checkout to a
public, installable release. It covers **where** each artifact should live and
**why**, then the exact commands. Nothing here runs automatically except the
tag-triggered workflows.

Current targets for this repository:

| Artifact | Host | Identifier |
|---|---|---|
| Source + docs + releases | GitHub | `ADPer0705/verity` |
| Container images | GitHub Container Registry | `ghcr.io/adper0705/verity` |
| Checkpoints | Hugging Face Hub (primary) | `ADPer/verity-weights` |
| Wheel (optional) | PyPI | `verity` (only if the name is available) |

Nothing in the project hard-codes a host: the manifest accepts any HTTPS URL
for weights, and the image name is a compose-file field, so you can change
these later without touching code.

---

## 0. Distribution choices (read this before publishing)

### Weights: Hugging Face is the right primary

The checkpoints are 1 MB–1.4 GB binaries that need licence metadata,
checksums, immutable revisions and a CDN in front of them. Hugging Face Hub
wins on every axis that matters here:

- **LFS-native** with per-file SHA-256 available, and our own SHA-256 gate on
  top (`verity weights verify`), so a corrupted or substituted file can never
  be used;
- **model cards** carry per-file licence + attribution — exactly the CC
  BY-NC 4.0 §3 duty of the DeepfakeBench-derived files;
- **free for public repos**, no egress bill, fast global CDN;
- **`resolve/<revision>/<path>` URLs**, which is what
  `detectors/weights_manifest.yaml` already speaks;
- **revision pinning** (branch, tag or commit) so a report generated in six
  months can be reproduced with the same bytes;
- **gated/private repos** if you later need an access-controlled distribution.

Choose these instead only for a specific reason:

| Host | Good for | Trade-offs vs HF |
|---|---|---|
| **GitHub Releases** | tiny weights (< 2 GB/file), one-off mirrors | no model card, no LFS semantics, release assets are less discoverable; repo LFS has bandwidth quotas |
| **Cloudflare R2 / S3** | private or air-gapped deployments, custom domain, zero-egress R2 | you own ACLs, lifecycle, TLS, integrity dashboards; manifest still works by putting the object URL in `origin` |
| **Zenodo** | a citable DOI snapshot at paper/release time | not a package registry; upload limits and slower updates |
| **Kaggle Datasets** | community visibility | unstable direct-download URLs, not suitable for a manifest |
| **Git LFS in git** | keeping weights next to code | clone size/bandwidth pain; git history is the wrong versioning model for weights |

Recommended topology: **HF as canonical mirror for redistributable files**,
authors' URLs for files we may not re-host (NPR, MLEP, DF40, Effort,
XLS-R-SLS), and optionally an R2/S3 bucket with the same object layout for
private/offline installs.

### Containers: GHCR is the right primary

- **Same identity and permissions as the repo** — the release workflow
  authenticates with the built-in `GITHUB_TOKEN`, no long-lived secrets;
- **free for public images**, anonymous pulls, no Docker Hub pull-rate limit;
- **per-version tags next to the source tag**, which makes provenance easy to
  explain;
- **Sigstore provenance/attestations** available from `docker/build-push-action`
  without extra infrastructure.

Mirror to **Docker Hub** only if discoverability matters more than rate limits
(it is still the default `docker pull` namespace for many users). **Quay.io**
is a reasonable alternative if you want built-in vulnerability scanning and
Red Hat-style robot accounts. Publish to a cloud registry (ECR/GAR/ACR) only
for a specific deployment pipeline.

GHCR currently hosts the **slim** image from CI. The full `:cuda` and `:cpu`
images are 10–20 GB and are built manually (below) — or by a self-hosted
runner.

### The Python package is optional

The wheel is useful for native installs and for a stable `verity` entry
point, but it does **not** contain detector environments or checkpoints.
Attach it to the GitHub release (the workflow does) and publish to PyPI only
if `verity` is available there; `uv publish` is enough. Conda/conda-forge is
not worth the maintenance burden at this stage.

---

## 1. One-time setup

```bash
gh auth login                     # GitHub CLI
gh auth status                    # needs repo scope; add write:packages for manual GHCR pushes
hf auth login                     # Hugging Face CLI
hf auth whoami
podman login ghcr.io -u ADPer0705 # PAT with write:packages (skip if CI-only)
```

## 2. Source repository

```bash
cd /path/to/verity
git status                        # weights, workspace, output, .env, node_modules must be ignored
git log --oneline                 # the history should read as a sequence of real changes
git push origin main
```

CI (`.github/workflows/ci.yml`) runs on every push: unit tests, the UI
typecheck/build, a manifest sanity check, and a no-push build of the slim
image. Fix CI **before** tagging — the release workflow trusts the tag.

## 3. Checkpoints — Hugging Face

Build the upload from the manifest (offline; nothing is pushed here):

```bash
uv run verity weights pull --all            # every file present + verified
uv run verity weights bundle --repo ADPer/verity-weights --out dist-weights
# -> dist-weights/verity-weights/
#      deepfakebench/… backbones/… univfd/… lipforensics/… aasist/…
#      community_forensics/… safe/…
#      README.md (model card)  NOTICE.md  LICENSE  manifest.json  sha256sums.txt
```

Only `redistributable: true` files are included. `bundle` reports missing
files instead of silently shipping a partial repository.

Create and upload:

```bash
hf repo create ADPer/verity-weights --type model --public
hf upload ADPer/verity-weights dist-weights/verity-weights . --repo-type model \
  --commit-message "Verity weights $(date -I)"
```

Then point the manifest at the mirror:

```yaml
# detectors/weights_manifest.yaml
hub:
  huggingface_repo: "ADPer/verity-weights"
  revision: main        # or a tag/commit for a frozen release
```

Verify from a clean state (a different machine or a cleared weights dir):

```bash
VERITY_WEIGHTS_HUB=ADPer/verity-weights \
  uv run verity weights pull --detector xception --force --source hub
VERITY_OFFLINE=1 uv run verity weights verify
```

Licence duties: the model card and `NOTICE.md` must keep naming the upstream
authors, link the original releases, state each licence, and say the files are
unmodified. DeepfakeBench checkpoints were trained on FaceForensics++ (c23),
a research-only dataset — say so on the card. Never upload the
`redistributable: false` files.

## 4. Containers — GHCR

### Slim image (automated)

```bash
git tag v0.4.0
git push origin v0.4.0
```

The release workflow builds and pushes:

- `ghcr.io/adper0705/verity:slim`
- `ghcr.io/adper0705/verity:0.4.1-slim`

`latest` is a moving alias for the newest slim image; the full `:cuda` and
`:cpu` images are built by hand (below). Make the package public once in
**GitHub → Packages → verity → Package settings** (first publish only).

### Full images (manual: too large for hosted runners)

Build on a machine with ≥ 60 GB free disk, from a clean checkout of the tag:

```bash
# CUDA — recommended for investigators with an NVIDIA GPU
podman build --target full --build-arg VERITY_PROFILE=cuda \
  -t ghcr.io/adper0705/verity:cuda -t ghcr.io/adper0705/verity:0.4.0-cuda .
podman push ghcr.io/adper0705/verity:cuda
podman push ghcr.io/adper0705/verity:0.4.0-cuda

# CPU — classrooms / air-gapped labs
podman build --target full --build-arg VERITY_PROFILE=cpu \
  -t ghcr.io/adper0705/verity:cpu -t ghcr.io/adper0705/verity:0.4.0-cpu .
podman push ghcr.io/adper0705/verity:cpu
podman push ghcr.io/adper0705/verity:0.4.0-cpu
```

`docker build`/`docker push` are identical, with `--gpus all` in place of
Podman's device flag when running.

### Smoke test before announcing

```bash
mkdir -p /tmp/vt/input /tmp/vt/output
cp misc/001_870.mp4 /tmp/vt/input/
podman run --rm --device nvidia.com/gpu=all \
  -v verity-envs:/envs -v verity-weights:/weights -v verity-cache:/cache \
  -v verity-workspace:/workspace \
  -v /tmp/vt/input:/input:ro -v /tmp/vt/output:/output \
  ghcr.io/adper0705/verity:slim run --media /input/001_870.mp4 --format html
ls /tmp/vt/output/*/report.html
```

This exercises the published image end to end (the first run provisions the
detector environments it needs). On CPU, drop the device flag; with Docker use
`--gpus all`. To smoke-test a locally built full image, substitute its tag.

## 5. Version bumps

Three places carry the version and must agree before tagging:

- `pyproject.toml` → `[project] version`
- `src/verity_ui/package.json` → `"version"`
- `src/verity_api/main.py` → `FastAPI(version=…)`

`scripts/release.py` refuses to build a wheel whose tag disagrees with
`pyproject.toml`. The current release line is **0.4.x** (0.4.1 as of this
writing).

## 6. Release checklist

- [ ] `uv run pytest -q` green; `uv run verity weights status` clean
- [ ] `npm run typecheck && npm run build` in `src/verity_ui`
- [ ] README quickstart works from a clean directory (compose pull path)
- [ ] HF mirror uploaded; `VERITY_WEIGHTS_HUB` set in the manifest; clean
      `weights pull --source hub` verified
- [ ] tag pushed; GitHub release has the wheel + generated notes
- [ ] slim image public; full images built/pushed if you ship them
- [ ] `CHANGELOG.md` section for the release
- [ ] no secrets, checkpoints, `workspace/` or `output/` in the tag

## 7. Rollback / revocation

- **Weights** — replace the HF revision or restore the previous commit; pin
  `hub.revision` back to the last good commit. Verity refuses checksum
  mismatches, so a bad upload cannot silently become evidence.
- **Images** — delete the bad tag in GHCR and re-push from the previous
  source tag. Keep semver tags immutable; consumers pin by tag or digest.
- **Wheel** — yank the PyPI release and delete the GitHub release asset;
  attach a fixed build under a new patch version.
