# Publishing Verity

Maintainer runbook for taking Verity from a local checkout to a release that a
lab can install. It covers where each artifact lives and why, then the exact
commands.

Current targets:

| Artifact | Host | Identifier |
|---|---|---|
| Source + docs + deploy bundle | GitHub | `ADPer0705/verity` releases |
| Container image | GitHub Container Registry | `ghcr.io/adper0705/verity` |
| Checkpoints | Hugging Face Hub | `ADPer/verity-weights` |

No Python wheel or sdist is published: the product is the image + checkpoints
+ deploy bundle, and a wheel without `detectors/` or environments cannot run
an examination.

---

## 0. Distribution choices

### Weights: Hugging Face

Checkpoints are 1 MB–1.4 GB binaries needing licence metadata, checksums,
immutable revisions and a CDN. Hugging Face Hub provides:

- LFS with per-file hashes, plus Verity's own SHA-256 gate before use;
- model cards carrying the CC BY-NC attribution duties of the
  DeepfakeBench-derived files;
- free public hosting and a global CDN;
- `resolve/<revision>/<path>` URLs — what the manifest already uses;
- revision pinning for reproducible re-runs.

Reasonable alternatives only for a specific need:

| Host | Good for | Trade-offs |
|---|---|---|
| **Cloudflare R2 / S3** | private or air-gapped mirrors, custom domain | you own ACLs, lifecycle, billing; manifest accepts the object URL |
| **Zenodo** | a citable DOI snapshot at paper time | not a registry; slower updates |
| **GitHub Releases** | tiny one-off mirrors | 2 GB/asset, no model card or LFS semantics |

Non-redistributable files (NPR, MLEP, DF40, Effort, XLS-R-SLS) are never
mirrored; the manifest points at the authors' URLs only.

### Containers: GHCR

GHCR gives repo-scoped permissions, `GITHUB_TOKEN` signing from CI, free
public pulls with no Docker Hub rate limit, and version tags next to the
source tag. Docker Hub is only worth a mirror if discoverability matters more
than rate limits.

Only the **slim** image is built in CI. The pre-baked `:cuda` / `:cpu` images
are 10–20 GB and exceed hosted-runner disk; build and push them manually
(section 4) or from a self-hosted runner. The deploy installer pre-builds all
detector environments, so labs still get a fully provisioned install from
`:slim`; a full image only shortens first install.

### Releases: deploy bundle, not Python artifacts

Every release attaches:

- `verity-deploy-<version>.tar.gz` — compose files, installer, doctor,
  systemd unit, deployment docs, license/notices, `VERSION`;
- `verity-deploy.tar.gz` — stable alias for the latest release;
- `SHA256SUMS`.

That is the lab-facing artifact. It contains no secrets, images or weights.

---

## 1. One-time setup

```bash
gh auth login
gh auth status                    # repo scope; write:packages for manual GHCR pushes
hf auth login
hf auth whoami
podman login ghcr.io -u ADPer0705 # PAT with write:packages (only for manual pushes)
```

## 2. Source repository

```bash
cd /path/to/verity
git status                        # weights, workspace, output, .env, node_modules ignored
uv run pytest -q
cd src/verity_ui && npm run typecheck && npm run build && cd -
git push origin main
```

CI runs tests, the UI build, a manifest sanity check and a no-push slim image
build. Fix CI before tagging.

## 3. Checkpoints — Hugging Face

```bash
uv run verity weights pull --all
uv run verity weights bundle --repo ADPer/verity-weights --out dist-weights
hf repo create ADPer/verity-weights --type model --public   # first time only
hf upload ADPer/verity-weights dist-weights/verity-weights . --repo-type model \
  --commit-message "Verity weights $(date -I)"
```

Only `redistributable: true` files are bundled; `bundle` reports missing files
rather than shipping a partial repo. The manifest already points at
`ADPer/verity-weights`; bump `hub.revision` if you freeze a release to a
tag/commit.

Verify from a clean directory:

```bash
VERITY_WEIGHTS_HUB=ADPer/verity-weights \
  uv run verity weights pull --detector xception --force --source hub
VERITY_OFFLINE=1 uv run verity weights verify
```

Keep the model card and `NOTICE.md` attribution intact; state that
DeepfakeBench checkpoints were trained on FaceForensics++ (c23), research-only.

## 4. Container images — GHCR

### Slim (automated)

```bash
git tag v0.5.0
git push origin v0.5.0
```

The release workflow pushes `ghcr.io/adper0705/verity:slim` and
`ghcr.io/adper0705/verity:0.5.0-slim`; `latest` follows the newest slim image.

### Full images (manual)

On a machine with ≥ 60 GB free disk:

```bash
podman build --target full --build-arg VERITY_PROFILE=cuda \
  -t ghcr.io/adper0705/verity:cuda -t ghcr.io/adper0705/verity:0.5.0-cuda .
podman push ghcr.io/adper0705/verity:cuda
podman push ghcr.io/adper0705/verity:0.5.0-cuda

podman build --target full --build-arg VERITY_PROFILE=cpu \
  -t ghcr.io/adper0705/verity:cpu -t ghcr.io/adper0705/verity:0.5.0-cpu .
podman push ghcr.io/adper0705/verity:cpu
podman push ghcr.io/adper0705/verity:0.5.0-cpu
```

Make the package public once (GitHub → Packages → verity → settings).

### Smoke test

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

## 5. Deploy bundle

Built by CI from the tag; build it locally to check:

```bash
uv run python scripts/build_deploy_bundle.py --version 0.5.0
tar tzf dist-deploy/verity-deploy-0.5.0.tar.gz
```

It must contain `docker-compose.yml`, `docker-compose.cpu.yml`, `.env.example`,
`install.sh`, `doctor.sh`, `verity.service`, `README.md`, `deployment.md`,
`VERSION`, `LICENSE`, `THIRD_PARTY_NOTICES.md`.

Test the installer path in a clean directory (or VM):

```bash
tar xzf dist-deploy/verity-deploy-0.5.0.tar.gz
cd verity-deploy && ./install.sh --dry-run
# then a real GPU/CPU install; run ./doctor.sh afterwards
```

## 6. Version bumps

Carry the same version in:

- `pyproject.toml` → `[project] version`
- `src/verity_ui/package.json` (+ `package-lock.json`)
- `src/verity_api/main.py` → `FastAPI(version=…)`
- `CITATION.cff` and `CHANGELOG.md`

The deploy bundle records the tag in `VERSION`; `install.sh` derives the
default image tag from it.

## 7. Release checklist

- [ ] `uv run pytest -q` green; UI typecheck/build green
- [ ] `uv run verity weights status --verify` clean
- [ ] HF mirror uploaded; clean `weights pull --source hub` verified
- [ ] deploy bundle builds; `./install.sh --dry-run` clean
- [ ] tag pushed; release has the deploy bundle + `SHA256SUMS` only
- [ ] slim image public, tags verified
- [ ] `CHANGELOG.md` has the release section
- [ ] no secrets, checkpoints, workspace/output or personal data in the tag

## 8. Rollback

- **Weights** — restore the previous HF commit; pin `hub.revision` back.
  Checksum verification prevents a bad upload from becoming evidence.
- **Images** — delete the bad tag and re-push from the previous source tag.
  Keep semver tags immutable; deploy by tag or digest.
- **Deploy bundle** — keep the previous tarball; the named volumes persist
  weights, environments and cases across a rollback.
