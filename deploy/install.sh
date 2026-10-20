#!/usr/bin/env bash
# Verity installer for a Linux host (laptop, workstation or server).
#
#   ./install.sh                    # auto-detect NVIDIA GPU, all defaults
#   ./install.sh --cpu              # force CPU profile
#   ./install.sh --bind 0.0.0.0     # LAN exposure (read the warning!)
#   ./install.sh --dry-run          # print actions without doing them
#
# What it does, in order:
#   1. checks docker/podman + compose, disk, and GPU availability
#   2. writes .env (image tag, bind address, hardware profile)
#   3. pulls the Verity image
#   4. downloads and verifies all trained-detector checkpoints (~3 GB + runtime
#      models fetched by detectors themselves)
#   5. builds every trained detector's isolated environment (one-time; the
#      /envs volume is reused afterwards)
#   6. starts the server, waits for /api/system, runs the doctor
#
# Re-running is safe: image pull, weights and environments are idempotent.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

VERSION="$(cat VERSION 2>/dev/null || echo dev)"
DEFAULT_IMAGE="ghcr.io/adper0705/verity:${VERSION}-slim"
IMAGE="$DEFAULT_IMAGE"
BIND="127.0.0.1"
PROFILE="auto"
DRY_RUN=0
SKIP_WEIGHTS=0
SKIP_ENVS=0
ALL_ENVS=0
ALLOW_NETWORK=0
CONCURRENCY="${VERITY_SETUP_CONCURRENCY:-4}"

usage() {
  cat <<'EOF'
Usage: ./install.sh [options]

  --image IMAGE       Container image (default: versioned slim tag from this bundle)
  --bind ADDRESS      Host address for the web console (default: 127.0.0.1)
  --cpu               Force the CPU profile
  --cuda              Force the CUDA profile
  --allow-network     Required to bind to anything other than loopback
  --skip-weights      Do not pre-download checkpoints
  --skip-envs         Do not pre-build detector environments
  --all-envs          Also build the research-scaffold environments (all 47)
  --concurrency N     Parallel environment builds (default: 4)
  --dry-run           Print commands without executing them
  -h, --help          This text
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --image) IMAGE="$2"; shift 2 ;;
    --bind) BIND="$2"; shift 2 ;;
    --cpu) PROFILE="cpu"; shift ;;
    --cuda) PROFILE="cuda"; shift ;;
    --allow-network) ALLOW_NETWORK=1; shift ;;
    --skip-weights) SKIP_WEIGHTS=1; shift ;;
    --skip-envs) SKIP_ENVS=1; shift ;;
    --all-envs) ALL_ENVS=1; shift ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

log()  { printf '\n\033[1m[verity]\033[0m %s\n' "$*"; }
warn() { printf '\n\033[33m[verity] WARNING:\033[0m %s\n' "$*" >&2; }
die()  { printf '\n\033[31m[verity] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

run() {
  if [ "$DRY_RUN" = 1 ]; then
    printf '  + %s\n' "$*"
  else
    "$@"
  fi
}

# ---------------------------------------------------------------- container runtime
RUNTIME=""
if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  RUNTIME="docker"
elif command -v podman >/dev/null 2>&1 && podman compose version >/dev/null 2>&1; then
  RUNTIME="podman"
elif [ "$DRY_RUN" = 1 ]; then
  RUNTIME="docker"
  warn "no usable compose runtime found; dry run will show the docker commands."
else
  die "Docker Engine with Compose v2 (or Podman with a compose provider) is required.
  Docker: https://docs.docker.com/engine/install/  (install the compose plugin)
  Podman: install podman-compose or the docker-compose provider."
fi

if [ "$(id -u)" = 0 ] && [ "$DRY_RUN" != 1 ]; then
  warn "running as root: files and volumes will be root-owned. Prefer a normal
  user in the 'docker' group (sudo usermod -aG docker <user>)."
fi

if [ "$RUNTIME" = "docker" ]; then
  COMPOSE=(docker compose)
else
  COMPOSE=(podman compose)
fi

command -v curl >/dev/null 2>&1 || warn "curl not found on the host; readiness polling will be skipped."

# ---------------------------------------------------------------- hardware profile
if [ "$PROFILE" = "auto" ]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    PROFILE="cuda"
  else
    PROFILE="cpu"
  fi
fi

if [ "$PROFILE" = "cuda" ]; then
  command -v nvidia-smi >/dev/null 2>&1 || die "--cuda requested but nvidia-smi is not available."
  nvidia-smi >/dev/null 2>&1 || die "nvidia-smi failed; install/repair the NVIDIA driver first."
  if [ "$RUNTIME" = "docker" ] && [ "$DRY_RUN" != 1 ] && ! docker info 2>/dev/null | grep -qi 'nvidia'; then
    warn "NVIDIA runtime not visible in 'docker info'. Install the NVIDIA Container Toolkit
  (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  or the container will fail to start with a GPU device error."
  fi
fi

# ---------------------------------------------------------------- exposure
case "$BIND" in
  127.0.0.1|localhost) ;;
  *)
    if [ "$ALLOW_NETWORK" != 1 ]; then
      die "refusing to bind to '$BIND': the Verity API has no authentication.
  Use 127.0.0.1 + an SSH tunnel, or put an authenticating reverse proxy in front
  and re-run with --allow-network."
    fi
    warn "binding to $BIND. The API has no built-in authentication; anyone who can
  reach this port can read every case in the workspace. Use a firewall and an
  authenticating reverse proxy (see docs/deployment.md)."
    ;;
esac

# ---------------------------------------------------------------- disk preflight
mkdir -p input output
AVAIL_KB="$(df -Pk . | awk 'NR==2 {print $4}')"
AVAIL_GB=$((AVAIL_KB / 1024 / 1024))
if [ "$AVAIL_GB" -lt 20 ]; then
  if [ "$DRY_RUN" = 1 ]; then
    warn "only ${AVAIL_GB} GB free in $(pwd); at least 20 GB is required, 60 GB recommended."
  else
    die "only ${AVAIL_GB} GB free in $(pwd); at least 20 GB is required, 60 GB recommended."
  fi
elif [ "$AVAIL_GB" -lt 60 ]; then
  warn "only ${AVAIL_GB} GB free; 60 GB is recommended for all trained detector environments."
fi

log "Verity $VERSION"
echo "  runtime : $RUNTIME"
echo "  image   : $IMAGE"
echo "  profile : $PROFILE"
echo "  bind    : $BIND:8000"
echo "  dir     : $HERE"
echo "  free    : ${AVAIL_GB} GB"

# ---------------------------------------------------------------- .env
write_env() {
  : > .env
  cat >> .env <<EOF
# Generated by install.sh — safe to edit.
VERITY_IMAGE=$IMAGE
VERITY_BIND=$BIND
VERITY_HW_PROFILE=$PROFILE
VERITY_WEIGHTS_AUTO=1
VERITY_WEIGHTS_HUB=
VERITY_OFFLINE=
VERITY_DETECTOR_TIMEOUT=3600
EOF
}
if [ "$DRY_RUN" = 1 ]; then
  echo "  + write .env"
else
  if [ -f .env ]; then
    cp .env ".env.bak.$(date +%Y%m%d-%H%M%S)"
    warn "existing .env backed up before rewriting the generated values."
  fi
  write_env
fi

COMPOSE_FILES=(-f docker-compose.yml)
if [ "$PROFILE" = "cpu" ]; then
  COMPOSE_FILES+=(-f docker-compose.cpu.yml)
fi

compose() { run "${COMPOSE[@]}" "${COMPOSE_FILES[@]}" "$@"; }

# ---------------------------------------------------------------- install
log "Pulling the image"
compose pull

if [ "$SKIP_WEIGHTS" != 1 ]; then
  log "Downloading and verifying trained checkpoints (progress below)"
  compose run --rm verity weights pull --trained
else
  warn "skipping checkpoint download; detectors with missing weights will be excluded at run time."
fi

if [ "$SKIP_ENVS" != 1 ]; then
  if [ "$ALL_ENVS" = 1 ]; then
    log "Building every detector environment, including research scaffolds (one-time)"
    compose run --rm verity setup --all --concurrency "$CONCURRENCY"
  else
    log "Building every evidence-grade detector environment (one-time; takes a while)"
    compose run --rm verity setup --trained --concurrency "$CONCURRENCY"
  fi
else
  warn "skipping environment provisioning; the first analysis will build the
  environments it needs, one detector at a time."
fi

log "Starting the server"
compose up -d

if [ "$DRY_RUN" = 1 ]; then
  log "Dry run complete; nothing was changed."
  exit 0
fi

# ---------------------------------------------------------------- wait for readiness
PROBE_HOST="$BIND"
[ "$PROBE_HOST" = "0.0.0.0" ] && PROBE_HOST="127.0.0.1"
log "Waiting for http://$PROBE_HOST:8000/api/system"
READY=0
if command -v curl >/dev/null 2>&1; then
  for _ in $(seq 1 90); do
    if curl -fsS "http://$PROBE_HOST:8000/api/system" >/dev/null 2>&1; then READY=1; break; fi
    sleep 2
  done
fi
if [ "$READY" != 1 ]; then
  warn "the API did not become ready in 3 minutes (or curl is unavailable). Inspect: ${COMPOSE[*]} ${COMPOSE_FILES[*]} logs --tail=200"
fi

log "Running checks"
"$HERE/doctor.sh" --host "$PROBE_HOST" || true

cat <<EOF

────────────────────────────────────────────────────────────────────
Verity is installed.

  Web console : http://$PROBE_HOST:8000
  API docs    : http://$PROBE_HOST:8000/api/docs

  From your workstation (if bound to loopback on a remote server):
      ssh -L 8000:localhost:8000 <user>@<server>
      then open http://localhost:8000

  Logs        : ${COMPOSE[*]} ${COMPOSE_FILES[*]} logs -f
  Status      : ${COMPOSE[*]} ${COMPOSE_FILES[*]} ps
  Weights     : ${COMPOSE[*]} ${COMPOSE_FILES[*]} run --rm verity weights status
  Stop        : ${COMPOSE[*]} ${COMPOSE_FILES[*]} down
  Upgrade     : replace this directory with a newer bundle and re-run ./install.sh

  Data: ./input (read-only media), ./output (reports),
        docker volumes verity-weights, verity-envs, verity-workspace.
────────────────────────────────────────────────────────────────────
EOF
