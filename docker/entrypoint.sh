#!/usr/bin/env bash
# Verity container entrypoint.
#
#   verity ... (default)   -> the CLI (run | analyze | setup | weights | ...)
#   server                 -> API + web console on $VERITY_PORT (default 8000)
#   bash / sh              -> a shell
#
# Before a server or analysis command, missing checkpoints for trained
# detectors are fetched into the /weights volume unless VERITY_WEIGHTS_AUTO=0
# or VERITY_OFFLINE=1. Progress is visible in the container logs. Failures are
# reported, not fatal: detectors whose checkpoint is still missing are
# excluded by the selector with a reason.
set -euo pipefail

auto_weights() {
  case "${VERITY_WEIGHTS_AUTO:-1}" in 0|false|no) return 0;; esac
  case "${VERITY_OFFLINE:-}" in 1|true|yes) return 0;; esac
  echo "[verity] checking trained checkpoints in /weights (manifest: detectors/weights_manifest.yaml)" >&2
  if ! verity weights pull --trained; then
    echo "[verity] some checkpoints could not be fetched; affected detectors will be skipped" >&2
  fi
}

cmd="${1:-run}"
case "$cmd" in
  server)
    shift
    auto_weights
    exec verity-server "$@"
    ;;
  bash|sh)
    exec "$@"
    ;;
  run|analyze|ingest|extract|compile|report)
    auto_weights
    exec verity "$@"
    ;;
  *)
    exec verity "$@"
    ;;
esac
