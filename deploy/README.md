# Verity deploy bundle

This directory is a self-contained installer for a Linux host. It pulls the
Verity image, downloads and verifies every trained-detector checkpoint, builds
all detector environments once, and starts the web console + API.

## Requirements

- Linux x86_64 (laptop, workstation or server)
- Docker Engine 24+ with Compose v2, **or** Podman with a compose provider
- 20 GB free disk minimum, **60 GB recommended** (weights + environments)
- 8 GB RAM minimum; NVIDIA GPU with 8 GB+ VRAM recommended (CPU works, slower)
- NVIDIA Container Toolkit for GPU operation
- Outbound HTTPS for the first install (image, checkpoints, model weights)

## Install

```bash
tar xzf verity-deploy-*.tar.gz
cd verity-deploy
./install.sh                    # as a user in the docker group, not root
```

If you downloaded from the release page, verify it first:

```bash
sha256sum -c SHA256SUMS
```

`install.sh` auto-detects GPU vs CPU, then:

1. pulls `ghcr.io/adper0705/verity:<version>-slim`
2. runs `verity weights pull --trained` (progress visible)
3. runs `verity setup --trained` (all 26 evidence-grade detector environments)
4. starts the stack and waits for `/api/system`
5. runs `doctor.sh`

Add `--all-envs` to build the research-scaffold environments too (all 47
architectures). Re-run `./install.sh` at any time; every step is idempotent.

## Access

By default the console binds to **127.0.0.1:8000** because the API has no
authentication. From a remote workstation:

```bash
ssh -L 8000:localhost:8000 user@server
# open http://localhost:8000
```

For LAN access, put an authenticating reverse proxy in front and run:

```bash
./install.sh --bind 0.0.0.0 --allow-network
```

## Operations

```bash
docker compose ps
docker compose logs -f
docker compose run --rm verity weights status
docker compose run --rm verity weights verify
docker compose run --rm verity setup --trained          # resume/refresh envs
docker compose down                                     # stop (volumes persist)
./doctor.sh                                             # health check
```

Volumes: `verity-weights` (checkpoints), `verity-envs` (detector
environments), `verity-cache` (HF/torch caches), `verity-workspace` (case
data). Host folders: `./input` (read-only media), `./output` (reports).

## Upgrade

Download the new bundle, stop the old stack (`docker compose down`), unpack
the new bundle in the same directory or move the volumes' project name, then
run `./install.sh` again. The named volumes carry weights, environments and
case data across the upgrade. Pin the image tag and keep a copy of the old
bundle for rollback.

## Offline / air-gapped

On a connected machine with the same image:

```bash
docker pull ghcr.io/adper0705/verity:<version>-slim
docker save ghcr.io/adper0705/verity:<version>-slim -o verity-image.tar
docker compose run --rm verity weights pull --trained   # fills verity-weights
docker run --rm -v verity-weights:/w alpine tar czf - -C /w . > verity-weights.tar.gz
```

Move the image tar and weights archive to the air-gapped host:

```bash
docker load -i verity-image.tar
docker volume create verity-weights
docker run --rm -v verity-weights:/w -v "$PWD":/src alpine \
  sh -c 'cd /w && tar xzf /src/verity-weights.tar.gz'
./install.sh --skip-weights          # builds environments, starts server
# then set VERITY_OFFLINE=1 in .env and restart
```

## Troubleshooting

- `doctor.sh` checks runtime, container, API readiness, weights, envs, disk, GPU.
- GPU device errors → install NVIDIA Container Toolkit; verify
  `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi`.
- API unreachable → check `docker compose ps` and `docker compose logs`.
- Slow first analysis → environments are still being built; pre-build with
  `docker compose run --rm verity setup --trained --concurrency 4`.
- Disk pressure → `docker system df`; weights and envs live in the volumes.
