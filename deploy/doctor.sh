#!/usr/bin/env bash
# Verity deployment doctor: verifies the pieces a fresh install depends on.
#
#   ./doctor.sh                 # check the stack in this directory
#   ./doctor.sh --host 10.0.0.5 # probe a non-loopback bind
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
HOST="127.0.0.1"

while [ $# -gt 0 ]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    -h|--help) echo "Usage: ./doctor.sh [--host ADDRESS]"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

FAIL=0
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
bad()  { printf '  \033[31m✗\033[0m %s\n' "$*"; FAIL=1; }
note() { printf '  \033[33m•\033[0m %s\n' "$*"; }

if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  COMPOSE=(docker compose); ok "docker compose available"
elif command -v podman >/dev/null 2>&1 && podman compose version >/dev/null 2>&1; then
  COMPOSE=(podman compose); ok "podman compose available"
else
  bad "no container runtime with a compose provider (docker compose / podman compose)"
  exit 1
fi

FILES=(-f docker-compose.yml)
PROFILE="$(grep -E '^VERITY_HW_PROFILE=' .env 2>/dev/null | cut -d= -f2- || true)"
[ "$PROFILE" = "cpu" ] && FILES+=(-f docker-compose.cpu.yml)

if "${COMPOSE[@]}" "${FILES[@]}" ps --status running --services 2>/dev/null | grep -qx verity; then
  ok "verity container running"
else
  bad "verity container not running (start with: ${COMPOSE[*]} ${FILES[*]} up -d)"
fi

if curl -fsS "http://$HOST:8000/api/system" >/tmp/verity-system.json 2>/dev/null; then
  ok "API reachable at http://$HOST:8000/api/system"
  python3 - <<'EOF' 2>/dev/null || note "install python3 to summarise readiness"
import json
d = json.load(open('/tmp/verity-system.json'))
r = d.get("readiness", {})
print("    detectors registered:", d.get("detectors_registered", "?"))
for key in ("can_detect", "can_setup", "can_analyze", "can_report_pdf"):
    print(f"    {key}: {'ok' if r.get(key) else 'BLOCKED'}")
for b in r.get("blockers", []):
    print(f"    blocker [{b.get('capability')}]: {b.get('message')}")
EOF
else
  bad "API not reachable at http://$HOST:8000 (bound elsewhere? use --host)"
fi

if "${COMPOSE[@]}" "${FILES[@]}" exec -T verity verity weights status >/tmp/verity-weights-status.txt 2>/dev/null; then
  MISSING="$(grep -ciE 'missing|MISSING' /tmp/verity-weights-status.txt || true)"
  if [ "${MISSING:-0}" -gt 0 ]; then
    bad "some checkpoints are missing (run: ${COMPOSE[*]} ${FILES[*]} run --rm verity weights pull --trained)"
  else
    ok "checkpoints present"
  fi
else
  note "could not read weights status (container may still be starting)"
fi

ENVS="$("${COMPOSE[@]}" "${FILES[@]}" exec -T verity sh -lc 'ls /envs/*/ 2>/dev/null | head -50' 2>/dev/null | grep -c '/$' || true)"
if [ "${ENVS:-0}" -gt 0 ]; then
  ok "detector environments provisioned"
else
  note "no detector environments yet; first analysis will build them (or run: setup --trained)"
fi

AVAIL_KB="$(df -Pk . | awk 'NR==2 {print $4}')"
AVAIL_GB=$((AVAIL_KB / 1024 / 1024))
if [ "$AVAIL_GB" -lt 20 ]; then bad "only ${AVAIL_GB} GB free"; else ok "${AVAIL_GB} GB free disk"; fi

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  ok "NVIDIA GPU visible on the host"
else
  note "no NVIDIA GPU detected (CPU profile)"
fi

if [ "$FAIL" = 0 ]; then
  printf '\n\033[32mAll critical checks passed.\033[0m\n'; exit 0
else
  printf '\n\033[31mSome checks failed — see messages above.\033[0m\n'; exit 1
fi
