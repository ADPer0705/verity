# Deploying Verity for a lab or investigation team

This is the operator guide: what to install, what gets downloaded, how the
web console is exposed, and how to verify that the stack is actually ready.
It assumes a Linux host and a container runtime; no Python knowledge is
required.

If you only want the short version, it is:

```bash
tar xzf verity-deploy-<version>.tar.gz
cd verity-deploy
./install.sh                      # run as a user in the docker group, not root
# open http://127.0.0.1:8000 (or tunnel from your workstation)
```

Everything below explains what that does and how to operate it.

## 1. What you are deploying

One container runs the orchestrator, API and web console. Four named volumes
hold the state:

| Volume | Contents | Survives upgrade |
|---|---|---|
| `verity-weights` | detector checkpoints (verified against SHA-256) | yes |
| `verity-envs` | one isolated uv environment per detector | yes |
| `verity-cache` | Hugging Face / torch model caches | yes |
| `verity-workspace` | ingested media, per-case artifacts, compiled findings | yes |

Host bind mounts:

| Path | Purpose |
|---|---|
| `./input` | media you want examined (mounted read-only) |
| `./output` | generated reports (`report.html`, `report.pdf`, `analysis.json`, `selection.json`) |

The original media is copied into the workspace and identified by its SHA-256
checksum; the host file is never modified.

## 2. Host requirements

| Item | Minimum | Recommended |
|---|---|---|
| OS | Linux x86_64 | current Ubuntu LTS / Debian / RHEL-family |
| Runtime | Docker Engine 24 + Compose v2, or Podman + compose provider | Docker Engine + Compose plugin |
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 16–32 GB |
| GPU | optional | NVIDIA, 8 GB+ VRAM, current driver |
| GPU runtime | — | NVIDIA Container Toolkit |
| Disk | 20 GB free | 60 GB+ free |
| Network | outbound HTTPS during install | outbound HTTPS for model downloads |

Notes:

- On CPU everything runs, but the large video/audio models take minutes per
  clip rather than seconds.
- 8 GB VRAM is enough for the evidence-grade panel in normal use, but a few
  models (CLIP-L-based detectors, wav2vec/MMS audio, AEROBLADE's diffusion
  autoencoders) are the heaviest; run one at a time if you see CUDA OOM.
- The exact disk footprint depends on which runtime models get fetched by
  detectors themselves. Use `docker system df` and `doctor.sh` after install.

### Supported runtime variants

| Image | Published | Behaviour |
|---|---|---|
| `:slim` | yes (CI, every release) | environments are built into `verity-envs` on install/first use |
| `:cuda`, `:cpu` | not published; manual build | all trained environments pre-baked; fastest start, 10–20 GB image |

`slim` is what `install.sh` uses. The pre-baked images are optional and built
from the Dockerfile (`--target full`); see [publishing.md](publishing.md).

## 3. What happens during installation

`install.sh` performs these steps in order, with progress visible:

1. **Preflight** — container runtime + compose, GPU/CPU detection, disk space,
   and refuses a non-loopback bind unless explicitly allowed.
2. **Image pull** — `ghcr.io/adper0705/verity:<version>-slim`.
3. **Weights** — `verity weights pull --trained`. Reads
   `detectors/weights_manifest.yaml`, downloads from the Verity Hugging Face
   mirror first and the authors' URLs for non-redistributable files, verifies
   every SHA-256, and stores them in `verity-weights`.
4. **Environments** — `verity setup --trained --concurrency N`. Builds one
   isolated uv environment per evidence-grade detector into `verity-envs`.
   This is the long step: tens of minutes on a fresh host.
5. **Start** — `docker compose up -d`, then polls `/api/system` until ready.
6. **Doctor** — checks runtime, container, API readiness, checkpoints,
   environments, disk and GPU.

After this, an analysis does **not** need to install anything: weights and
environments are cached. Re-running the installer is safe and idempotent.

### Which detectors run

There is no reduced "recommended" profile. On every image/video/audio case,
Verity selects **all evidence-grade detectors** that can consume that media
(26 architectures with trained heads or training-free methods). The remaining
21 ports are research scaffolds (untrained heads); they are excluded by
default, labelled if explicitly requested, and never count toward consensus.
To prepare those too:

```bash
docker compose run --rm verity setup --all
# or install everything up front:
./install.sh --all-envs
# then run with --include-untrained (or name the detectors)
```

## 4. Exposure and access

The API has **no authentication**. The compose file therefore binds to
loopback only:

```yaml
ports:
  - "${VERITY_BIND:-127.0.0.1}:8000:8000"
```

### Single workstation

Open <http://127.0.0.1:8000>.

### Remote server, single examiner (recommended default)

Keep the loopback bind and use an SSH tunnel from your workstation:

```bash
ssh -L 8000:localhost:8000 user@lab-server
# then open http://localhost:8000 locally
```

Traffic is encrypted by SSH; nothing is exposed on the lab network.

### Shared lab access (read this before exposing)

If several people must reach it, put an authenticating reverse proxy in front
and bind Verity to loopback (best) or to a private interface:

```bash
./install.sh --bind 0.0.0.0 --allow-network
```

Minimal Caddy example (TLS + basic auth):

```caddyfile
verity.lab.example {
    basicauth {
        examiner $2a$14$REPLACE_WITH_CADDY_HASH
    }
    reverse_proxy 127.0.0.1:8000
}
```

Generate the hash with `caddy hash-password`. Nginx/oidc-proxy/Tailscale are
equally valid. Do **not** expose port 8000 directly to an untrusted network:
anyone who can reach it can read every case in the workspace and upload media.
There are no user accounts, roles or audit log.

### Firewall checklist

- only 22/443 reachable from user networks;
- 8000 bound to loopback (or a private interface behind the proxy);
- restrict egress if policy requires it, after the initial download.

## 5. Weights management

Weights are never baked into the published image.

```bash
# what is configured, present, licensed and verified
docker compose run --rm verity weights status
docker compose run --rm verity weights verify

# fetch what the evidence-grade panel needs
docker compose run --rm verity weights pull --trained

# force a specific detector / re-fetch
docker compose run --rm verity weights pull --detector xception --force
```

- Source of truth: `detectors/weights_manifest.yaml` (SHA-256, size, origin,
  license, redistribution policy).
- Redistributable files come from the Hugging Face mirror
  [`ADPer/verity-weights`](https://huggingface.co/ADPer/verity-weights);
  non-redistributable files (NPR, MLEP, DF40, Effort, XLSR-SLS) come from the
  authors' own URLs.
- The `server` entrypoint also pulls missing trained checkpoints on start, but
  the installer does it explicitly so you can see progress and failures.
- `VERITY_OFFLINE=1` forbids all downloads; a detector whose weights are
  missing is excluded with a recorded reason instead of failing the case.

## 6. Environment management

Each detector owns a uv project (`detectors/<id>/pyproject.toml`) and gets a
dedicated environment under `verity-envs/<profile>/<id>`. This is what lets
incompatible torch versions coexist.

```bash
docker compose run --rm verity setup --trained --concurrency 4
docker compose run --rm verity setup --detector xception,f3net --force
docker compose run --rm verity setup --all
```

If you skip pre-provisioning (or add a detector later), the environment is
built on first use — the analysis simply takes longer. Watch progress with
`docker compose logs -f`.

## 7. Operating the stack

```bash
docker compose ps
docker compose logs -f --tail=200
./doctor.sh

# one-off CLI run (no web console)
docker compose run --rm verity run --media /input/case.mp4 --format pdf
```

The server restarts automatically (`restart: unless-stopped`) and has a
healthcheck on `/api/system`. Optional boot integration: copy
`verity.service` to `/etc/systemd/system/`, set `WorkingDirectory` to the
bundle path, then `systemctl enable --now verity`.

### Backups

Case data lives in the workspace volume and `./output`. Back up what your
evidence policy requires:

```bash
docker run --rm -v verity-workspace:/w -v "$PWD":/backup alpine \
  tar czf /backup/verity-workspace-$(date -I).tar.gz -C /w .
```

Weights and environments are reproducible from the manifest and image; they do
not need backup, only re-download/re-build time.

### Upgrade

1. Download the new `verity-deploy-<version>.tar.gz`, verify `SHA256SUMS`.
2. `docker compose down` in the old directory.
3. Unpack the new bundle (same directory keeps the compose project name and
   volumes) and run `./install.sh`.
4. `./doctor.sh`; keep the previous bundle for rollback.

Pin the image tag from the bundle's `VERSION`; avoid `latest` for evidence
work.

## 8. Offline / air-gapped labs

The first install needs HTTPS. For an isolated network, prepare on a connected
machine with the same architecture:

```bash
docker pull ghcr.io/adper0705/verity:<version>-slim
docker save ghcr.io/adper0705/verity:<version>-slim -o verity-image.tar
docker compose run --rm verity weights pull --trained
docker run --rm -v verity-weights:/w alpine tar czf - -C /w . > verity-weights.tar.gz
```

Move both archives plus the deploy bundle, then:

```bash
docker load -i verity-image.tar
docker volume create verity-weights
docker run --rm -v verity-weights:/w -v "$PWD":/src alpine \
  sh -c 'cd /w && tar xzf /src/verity-weights.tar.gz'
./install.sh --skip-weights
# set VERITY_OFFLINE=1 in .env, restart
```

Environment builds still need the Python wheels; use a full `:cuda`/`:cpu`
image if the air-gapped host has no package mirror, or mirror the uv cache
under `verity-envs`.

## 9. Verification checklist

```bash
./doctor.sh

curl -s http://127.0.0.1:8000/api/system | python3 -m json.tool
# expect readiness.can_detect / can_setup / can_report_pdf = true,
# no blockers, and detectors_registered = 47

docker compose run --rm verity weights status | head
# every trained checkpoint present; licenses shown

docker compose run --rm verity run --media /input/<small-test-file> --format html
ls output/*/report.html
```

Then open the console, upload the same file, and confirm that the case shows
per-detector findings, attention/overlay artifacts and forensic observations.

## 10. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `docker compose up` fails with an NVIDIA device error | NVIDIA Container Toolkit missing, or using the CPU overlay on a GPU host (or vice versa). `./install.sh` picks automatically; `--cpu` forces CPU. |
| API not reachable | Container still starting; `docker compose logs`. Check `VERITY_BIND` and use `doctor.sh --host`. |
| “container is unhealthy” | `/api/system` failing; check disk space and logs. |
| First analysis very slow | Environments are being built; pre-build with `setup --trained`. |
| “checkpoint missing … detector excluded” | Run `weights pull --trained`; check network or `VERITY_WEIGHTS_HUB`. |
| PDF export fails | WeasyPrint system libraries missing — the published image includes them; custom images must too. |
| `permission denied` on `./input`/`./output` | SELinux: the compose file sets `label=disable`; otherwise fix ownership or use `:Z`. |
| Out of disk | `docker system df`; move the Docker root or prune old images. Keep the four volumes. |

## 11. Known limits

- No built-in authentication, accounts, roles or audit log. Use a proxy.
- Jobs are in memory and lost on restart; the workspace is the durable record.
- The report contains no aggregate verdict by design.
- CPU operation is functional but not suitable for high-throughput labs.
- Full pre-baked images are manual builds; only `:slim` is published.
