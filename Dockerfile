# syntax=docker/dockerfile:1.7
#
# Verity: deepfake examination pipeline for investigators and researchers.
#
# Targets
#   slim  (default)  orchestrator + API + web console. Detector environments
#                    are provisioned into the /envs volume on first use.
#   full             everything above plus every trained detector's Python
#                    environment pre-baked for one hardware profile, so a
#                    fresh host runs an examination with no provisioning
#                    step. Build arg VERITY_PROFILE=cuda|cpu picks the torch
#                    wheels (cuda wheels run on CPU hosts too, just larger).
#
#   docker build -t verity:slim .
#   docker build -t verity:cuda --target full --build-arg VERITY_PROFILE=cuda .
#   docker build -t verity:cpu  --target full --build-arg VERITY_PROFILE=cpu  .
#
# Checkpoints are never baked in: they are fetched into the /weights volume on
# first run (`verity weights pull`), with licenses tracked per file in
# detectors/weights_manifest.yaml.

# ---------------------------------------------------------------- UI build
FROM node:26-bookworm-slim AS ui
WORKDIR /ui
COPY src/verity_ui/package.json src/verity_ui/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY src/verity_ui/ ./
RUN npm run build

# ---------------------------------------------------------------- runtime (slim)
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS slim

LABEL org.opencontainers.image.title="Verity" \
      org.opencontainers.image.description="Multi-detector deepfake examination pipeline with explainable, per-detector findings" \
      org.opencontainers.image.licenses="CC-BY-NC-4.0" \
      org.opencontainers.image.source="https://github.com/ADPer0705/verity"

ENV UV_LINK_MODE=hardlink \
    UV_NO_PROGRESS=1 \
    UV_PYTHON_PREFERENCE=only-system \
    PYTHONUNBUFFERED=1 \
    VERITY_DETECTOR_ENVS=/envs \
    VERITY_WEIGHTS_ROOT=/weights \
    VERITY_HW_PROFILE=auto \
    VERITY_WORKSPACE=/workspace \
    VERITY_DETECTORS_DIR=/app/detectors \
    VERITY_OUTPUT_DIR=/output \
    VERITY_HOST=0.0.0.0 \
    VERITY_PORT=8000 \
    INPUT_DIR=/input \
    OUTPUT_DIR=/output \
    HOME=/cache \
    TORCH_HOME=/cache/torch \
    HF_HOME=/cache/huggingface \
    XDG_CACHE_HOME=/cache

# ffmpeg/ffprobe: extraction + characterization. pango/cairo: WeasyPrint PDF.
# libgl1/libglib2.0: OpenCV. git: a few detector deps install from VCS.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libpango-1.0-0 libpangocairo-1.0-0 libcairo2 libgdk-pixbuf-2.0-0 \
        libgl1 libglib2.0-0 \
        git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Orchestrator env first (cache-friendly): only pyproject/lock invalidate it.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --extra api

COPY config ./config
COPY scripts ./scripts
COPY detectors ./detectors
COPY src/verity ./src/verity
COPY src/verity_api ./src/verity_api
COPY docker/entrypoint.sh /usr/local/bin/verity-entrypoint
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra api \
    && chmod +x /usr/local/bin/verity-entrypoint scripts/verity.py \
    && ln -s /app/scripts/verity.py /usr/local/bin/verity-run

COPY --from=ui /ui/dist ./src/verity_ui/dist
ENV PATH="/app/.venv/bin:${PATH}" \
    VERITY_UI_DIST=/app/src/verity_ui/dist

RUN mkdir -p /envs /weights /cache /workspace /input /output

VOLUME ["/envs", "/weights", "/cache", "/workspace"]
EXPOSE 8000
ENTRYPOINT ["verity-entrypoint"]
CMD ["server"]

# ---------------------------------------------------------------- full
# Pre-bake the per-detector environments for every trained detector. The
# envs live inside the image (/opt/verity/envs), not in the /envs volume, so
# the image is self-sufficient; VERITY_DETECTOR_ENVS still points at a
# writable volume for anything provisioned later (untrained detectors,
# --force re-syncs), and the runner looks in both.
FROM slim AS full
ARG VERITY_PROFILE=cuda
ENV VERITY_HW_PROFILE=${VERITY_PROFILE} \
    VERITY_DETECTOR_ENVS_BAKED=/opt/verity/envs
RUN --mount=type=cache,target=/opt/verity/envs/cache \
    VERITY_DETECTOR_ENVS=/opt/verity/envs \
    python scripts/verity.py --profile ${VERITY_PROFILE} setup --trained --concurrency 4 \
    && find /opt/verity/envs -name '__pycache__' -type d -prune -exec rm -rf {} + \
    && du -sh /opt/verity/envs
