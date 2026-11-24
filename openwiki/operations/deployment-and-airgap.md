---
type: operations
title: Deployment, configuration and air-gap operations
description: Operator and lab deployment guide covering the one-line standalone binary installer, platformdirs workstation layout, online-to-airgap provisioning, USB air-gap kit transfer, Docker Compose fallback, configuration variables, and troubleshooting.
tags: [deployment, airgap, platformdirs, installer, docker, configuration, operations]
verified:
  - by: openwiki/0.9.0
    at: 2026-11-21T06:14:40.994Z
sources:
  - id: openwiki-source-10066bffec1f610f3650c047
    resource: repo://deploy/install.sh
  - id: openwiki-source-b79fbbd921df689b4bbdc82f
    resource: repo://docker-compose.yml
  - id: openwiki-source-03ffc32a0ca502ab67c54b25
    resource: repo://install.sh
  - id: openwiki-source-e4dc7d4c95cc68eb2f4b4656
    resource: repo://src/verity/cli.py
  - id: openwiki-source-6d817ce39f4036ea92df24bc
    resource: repo://src/verity/export.py
generated: { by: "antigravity", at: "2026-11-21T06:14:40.994Z" }
---

# Deployment, configuration and air-gap operations

Verity targets single-investigator forensic workstations and isolated labs that
need strict reproducibility, clean uninstall/upgrade separation, and complete
network isolation after setup. It provides two supported deployment paths:

1. **Option A — Standalone Single-Binary Workstation (Recommended)**: one Linux
   x86_64 binary installed with a single command (`install.sh`), managing tools,
   Python runtimes, 58 isolated detector virtualenvs, verified checkpoints, and
   workspaces cleanly under `platformdirs` (`~/.local/share/verity`,
   `~/.cache/verity`, `~/.config/verity`, or `$VERITY_HOME`). No Docker, system
   Python, or pre-installed `uv` is needed.
2. **Option B — Containerized Stack (Docker / Podman Compose)**: `:slim` image
   plus persistent volumes for hosts standardized on container orchestration.

## Why single binary + `platformdirs` + `uv` (instead of a monolithic 20 GB binary or PyPI)

- **Why not a fat single binary embedding all 58 virtualenvs & checkpoints?**
  39 evidence-grade detector environments + 6.3 GB of checkpoints + foundation
  hub weights exceed 20 GB and combine incompatible PyTorch/CUDA/mediapipe wheels
  that cannot inhabit one Python interpreter. Packing that into PyInstaller
  would break per-detector isolation and force multi-gigabyte decompression into
  RAM/tmpfs on every CLI invocation.
- **Why not PyPI (`pip install verity`)?** A plain Python wheel cannot reliably
  bundle the compiled React SPA, isolated Python 3.10–3.12 detector toolchains,
  or verified external weights without conflicting with host Python packages.
- **How `platformdirs` + self-bootstrapped `uv` solves both**: A ~130 MB single
  executable unpacks detector source specs cleanly into `<data_dir>/detectors`
  while symlinking `weights/` to persistent `<data_dir>/weights/<id>`, downloads
  Astral's `uv` (`bin/uv`) and standalone Pythons (`python/`) on demand, provisions
  all isolated `.venv`s into `<data_dir>/envs/<profile>/<id>`, and warms HF/Torch
  Hub caches into `<cache_dir>/` — leaving the workstation clean and ready to
  unplug permanently.

---

## Option A: Standalone binary installation

### 1. One-line verified install

```bash
curl -fsSL https://raw.githubusercontent.com/ADPer0705/verity/main/install.sh | bash
```

Root `install.sh` detects Linux x86_64, downloads both `verity-linux-x86_64` and
`verity-linux-x86_64.sha256` from GitHub Releases (`ADPer0705/verity`), aborts on
any SHA-256 mismatch, installs to `~/.local/bin/verity` (or `--dir PATH`), warns if
host `ffmpeg` / WeasyPrint libs are missing, and optionally triggers setup directly
via `--setup` or `--airgap-ready`.

### 2. Provision once while online, then seal for air-gapped use

```bash
# 1. Provision uv, standalone Pythons, 39 trained detector envs,
#    static ffmpeg (if absent), SHA-256 checkpoints, HF/Torch caches, and seal offline=true:
verity setup --trained --airgap-ready

# 2. Permanently disconnect Ethernet / Wi-Fi, then audit readiness & hashes:
verity doctor --verify

# 3. Launch the forensic Web Console (loopback http://127.0.0.1:8000):
verity serve
```

What `verity setup --trained --airgap-ready` executes in sequence (`src/verity/cli.py`):

1. Extracts bundled detector specs cleanly via `paths.ensure_bundled_detectors_extracted()`.
2. Bootstraps `uv` (`env_manager.ensure_uv`) and optional static `ffmpeg`/`ffprobe` (`ensure_ffmpeg`) into `<data_dir>/bin/`.
3. Builds isolated detector `.venv`s via `env_manager.setup_detectors()` (CUDA or CPU auto-detected; `--cpu` builds ~12x smaller `+cpu` wheels).
4. Downloads and SHA-256-verifies checkpoints (`weights.pull`) from `ADPer/verity-weights` and origin URLs.
5. Pre-warms runtime HF/Torch/FAN foundation caches on CPU (`env_manager.warmup_detectors`).
6. Re-verifies every checkpoint hash and writes `offline: true` + `hw_profile` to `<config_dir>/config.yaml`.

### 3. Transferring to a permanently offline workstation (never touches internet)

Python virtualenvs embed absolute shebangs tied to the build host's Python path
and username. `export_airgap_kit` / `import_airgap_kit` (`src/verity/export.py`)
avoids copying fragile `.venv` trees directly: instead, the online staging host
archives the standalone `verity` binary, `bin.tar.gz`, managed `python.tar.gz`,
verified `weights.tar.gz`, per-detector `locks.tar.gz` (`pyproject.toml` +
`uv.lock`), `cache-uv.tar.gz` wheel archive, and `cache-models.tar.gz` (HF/Torch
caches) with `airgap-manifest.json` + `SHA256SUMS`:

```bash
# On online staging workstation (same OS & arch):
verity setup --trained --airgap-ready
verity airgap export --out /mnt/usb/verity-kit

# Carry USB / optical media to the air-gapped forensic workstation:
/mnt/usb/verity-kit/verity airgap import --from /mnt/usb/verity-kit
verity doctor --verify
verity serve
```

`airgap import` verifies `SHA256SUMS`, unpacks archives safely (`filter="data"`),
seals `offline: true`, and rebuilds all 39 detector virtualenvs locally from the
cached wheels via `uv sync --offline --no-python-downloads` using hardlinks and
native target paths — zero network requests.

### 4. Upgrading the standalone binary

Replace `~/.local/bin/verity` with a newer release binary. When the new
`__version__` runs, `paths.ensure_bundled_detectors_extracted()` sees the version
stamp changed, updates detector code while leaving `<data_dir>/weights/` symlinks
and downloaded checkpoints completely untouched, and `env_manager.sync_one` only
re-syncs detectors whose `pyproject.toml` hash changed.

---

## Option B: Containerized deployment (Docker / Podman Compose)

For labs preferring containers (`deploy/install.sh` and `docker-compose.yml`):

```bash
curl -fsSLO https://github.com/ADPer0705/verity/releases/latest/download/verity-deploy.tar.gz
curl -fsSLO https://github.com/ADPer0705/verity/releases/latest/download/SHA256SUMS
sha256sum -c SHA256SUMS
tar xzf verity-deploy.tar.gz && cd verity-deploy
./install.sh                    # run as non-root user in docker group; --cpu forces CPU
```

Four persistent volumes retain state across upgrades (`verity-weights`,
`verity-envs`, `verity-cache`, `verity-workspace`), while `./input` mounts
read-only and `./output` receives HTML/PDF reports. `deploy/doctor.sh` checks the
running container, `/api/system` readiness, checkpoints, provisioned envs, free
disk, and GPU visibility. `deploy/verity.service` provides an optional `systemd`
unit.

---

## Host requirements

| Resource | Minimum | Recommended |
|---|---|---|
| OS | Linux x86_64 (`glibc >= 2.35`, e.g., Ubuntu 22.04+, Debian 12+, RHEL 9+) | Current LTS Linux workstation |
| Disk | 20 GB free | 60 GB+ free (39 envs + ~6.3 GB weights + HF caches + workspaces) |
| Memory | 8 GB RAM | 16–32 GB RAM |
| GPU | Optional (`--profile cpu` works everywhere) | NVIDIA GPU with 8 GB+ VRAM & current driver |
| Host libs *(for video/audio + PDF)* | `ffmpeg`, `libpango-1.0-0`, `libpangocairo-1.0-0`, `libcairo2` | Auto-bootstrapped static `ffmpeg` on Linux x86_64 if online |

---

## Configuration & environment variables

All path defaults come from `src/verity/paths.py`, `config/pipeline.yaml`,
`<config_dir>/config.yaml`, and `.env.example`:

| Variable | Effect | Default (Standalone / Binary) | Default (Repo Dev / Container) |
|---|---|---|---|
| `VERITY_HOME` | Root relocating `data/`, `cache/`, `config/` together | Unset (`platformdirs` XDG dirs) | Container / explicit override |
| `VERITY_WORKSPACE` | Case workspace (`media/`, `artifacts/`, `output/`) | `<data_dir>/workspace` | `./workspace` / `/workspace` |
| `VERITY_OUTPUT_DIR` | Report & bundle destination | `<workspace>/output` | `<workspace>/output` / `/output` |
| `VERITY_DETECTORS_DIR` | Unpacked detector definitions | `<data_dir>/detectors` | `./detectors` / `/app/detectors` |
| `VERITY_DETECTOR_ENVS` | Per-profile `uv` venvs | `<data_dir>/envs` | `./.detectorenvs` / `/envs` |
| `VERITY_WEIGHTS_ROOT` | Persistent weights target for `weights/` symlinks | `<data_dir>/weights` | Unset (in-package) / `/weights` |
| `VERITY_OFFLINE` | `1` forbids all network downloads & enables HF/UV offline flags | Reads `config.yaml::offline` | Unset |
| `VERITY_HW_PROFILE` | `auto`, `cuda`, or `cpu` | Reads `config.yaml::hw_profile` | `auto` |
| `VERITY_MAX_FRAMES` | Per-run sampled video frame budget (`0` = unlimited) | `32` (lower e.g. `16` on tight GPUs) | `32` |
| `VERITY_INCLUDE_UNTRAINED` | Auto-select untrained research scaffolds (flagged) | Unset | Unset |
| `VERITY_WEIGHTS_HUB` | Override Hugging Face weights mirror repo | `ADPer/verity-weights` | `ADPer/verity-weights` |
| `VERITY_DETECTOR_TIMEOUT` | Subprocess timeout in seconds | `3600` | `3600` |
| `VERITY_HOST` / `VERITY_PORT` | Console server bind address | `127.0.0.1` / `8000` | `0.0.0.0` (mapped to `127.0.0.1` by compose) |

---

## Exposure, backup & troubleshooting

- **Access security**: Never bind port 8000 directly to untrusted networks without
  an authenticating reverse proxy (TLS + Caddy/Nginx Auth). Remote single users
  should keep loopback (`127.0.0.1:8000`) and use `ssh -L 8000:localhost:8000 user@host`.
- **Verification**: Run `verity doctor --verify` (or `./doctor.sh` for Compose) and
  inspect `GET /api/system` to confirm `can_detect`, `can_setup`, `can_report_pdf`,
  and `detectors_registered = 58`.
- **Common fixes**:
  - *Out of `/tmp` space running PyInstaller binary on memory-constrained systems*:
    PyInstaller `--onefile` unpacks to `$TMPDIR` (~150 MB); export `TMPDIR=$HOME/.cache/tmp`
    if `/tmp` is a tiny tmpfs.
  - *CUDA OOM on 8 GB VRAM laptops*: Keep `JobManager` `VERITY_MAX_JOBS=1` (default)
    and reduce `VERITY_MAX_FRAMES=16`, or pass `--detector <id>` to examine heavy
    foundation detectors individually.
  - *PDF export unavailable*: Install `libpango-1.0-0 libpangocairo-1.0-0 libcairo2`
    (HTML reports still generate cleanly without WeasyPrint).
  - *SELinux host mounts in Docker*: Compose sets `security_opt: [label=disable]`.

## Related pages

- [Quickstart and documentation map](../quickstart.md) — fast commands for investigators.
- [Detector runtime and isolation](../architecture/detector-runtime.md) — internals of `env_manager`.
- [Weights and checkpoints](../architecture/weights-and-checkpoints.md) — manifest & SHA-256 checks.
- [Security, integrity and chain of custody](../architecture/security-and-integrity.md) — `export.py` zip bundles and network rules.
