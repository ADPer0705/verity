#!/usr/bin/env bash
# Verity one-line installer for Linux x86_64 forensic workstations.
#
#   curl -fsSL https://raw.githubusercontent.com/ADPer0705/verity/main/install.sh | bash
#
# Options (pass after `bash -s --`):
#   --version TAG      Install a specific release tag (e.g. v0.6.0; default: latest)
#   --dir PATH         Install directory (default: ~/.local/bin or $VERITY_INSTALL_DIR)
#   --setup            Run `verity setup --trained` after installing
#   --airgap-ready     Run `verity setup --trained --airgap-ready` after installing
#   -h, --help         Show this help text
#
# Downloads the standalone `verity` executable and its SHA-256 checksum from
# GitHub Releases, verifies the checksum before installation, and places the
# binary on disk. From there, `verity` itself bootstraps `uv`, managed Python
# runtimes, isolated detector environments, and SHA-256-verified checkpoints.
set -euo pipefail

REPO="${VERITY_GITHUB_REPO:-ADPer0705/verity}"
VERSION="${VERITY_VERSION:-latest}"
INSTALL_DIR="${VERITY_INSTALL_DIR:-$HOME/.local/bin}"
RUN_SETUP=0
AIRGAP_READY=0

usage() {
  cat <<'EOF'
Verity standalone installer (Linux x86_64)

Usage:
  curl -fsSL https://raw.githubusercontent.com/ADPer0705/verity/main/install.sh | bash
  curl -fsSL https://raw.githubusercontent.com/ADPer0705/verity/main/install.sh | bash -s -- [options]

Options:
  --version TAG      Pin a release tag (e.g. v0.6.0; default: latest)
  --dir PATH         Destination directory for the `verity` binary (default: ~/.local/bin)
  --setup            Provision trained detector environments & checkpoints right away
  --airgap-ready     Full air-gap provisioning (environments + checkpoints + hub caches + offline seal)
  -h, --help         Show this message
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --version) VERSION="$2"; shift 2 ;;
    --dir) INSTALL_DIR="$2"; shift 2 ;;
    --setup) RUN_SETUP=1; shift ;;
    --airgap-ready) RUN_SETUP=1; AIRGAP_READY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

log()  { printf '\033[1m[verity]\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[verity] WARNING:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31m[verity] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

OS="$(uname -s)"
ARCH="$(uname -m)"
if [ "$OS" != "Linux" ]; then
  die "Standalone prebuilt binaries currently target Linux x86_64 (detected OS: $OS).
  On macOS/Windows, run from source with uv (https://github.com/$REPO)."
fi
case "$ARCH" in
  x86_64|amd64) ASSET="verity-linux-x86_64" ;;
  *)
    die "No prebuilt binary for architecture '$ARCH' (currently Linux x86_64 only).
  Install from source with uv or use Docker Compose (https://github.com/$REPO)."
    ;;
esac

command -v curl >/dev/null 2>&1 || die "'curl' is required to download Verity."

SHA_CMD=""
if command -v sha256sum >/dev/null 2>&1; then
  SHA_CMD="sha256sum"
elif command -v shasum >/dev/null 2>&1; then
  SHA_CMD="shasum -a 256"
else
  die "Neither 'sha256sum' nor 'shasum' was found; refusing to install without checksum verification."
fi

if [ "$VERSION" = "latest" ]; then
  BASE_URL="https://github.com/${REPO}/releases/latest/download"
else
  case "$VERSION" in
    v*) TAG="$VERSION" ;;
    *)  TAG="v${VERSION}" ;;
  esac
  BASE_URL="https://github.com/${REPO}/releases/download/${TAG}"
fi

TMPDIR_WORK="$(mktemp -d)"
cleanup() { rm -rf "$TMPDIR_WORK"; }
trap cleanup EXIT

BIN_URL="${BASE_URL}/${ASSET}"
SUM_URL="${BASE_URL}/${ASSET}.sha256"

log "Downloading ${ASSET} (${VERSION}) from GitHub Releases..."
curl -fL --progress-bar -o "${TMPDIR_WORK}/${ASSET}" "$BIN_URL" \
  || die "Failed to download ${BIN_URL}. Verify the release tag exists."

log "Downloading SHA-256 checksum..."
curl -fsSL -o "${TMPDIR_WORK}/${ASSET}.sha256" "$SUM_URL" \
  || die "Failed to download checksum file ${SUM_URL}."

log "Verifying SHA-256 integrity..."
(cd "$TMPDIR_WORK" && $SHA_CMD -c "${ASSET}.sha256") \
  || die "SHA-256 verification FAILED! Aborting installation."

mkdir -p "$INSTALL_DIR"
TARGET="${INSTALL_DIR}/verity"
install -m 0755 "${TMPDIR_WORK}/${ASSET}" "$TARGET"

INSTALLED_VER="$("$TARGET" --version 2>/dev/null || echo "verity")"
log "Installed ${INSTALLED_VER} to ${TARGET}"

# Friendly check for optional host tools needed for video/audio extraction & PDF reports
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  warn "ffmpeg/ffprobe not found on PATH (needed for video/audio examination).
  Debian/Ubuntu: sudo apt-get install -y ffmpeg libpango-1.0-0 libpangocairo-1.0-0 libcairo2
  Fedora/RHEL:   sudo dnf install -y ffmpeg pango cairo"
fi

PATH_NOTE=""
case ":${PATH}:" in
  *":${INSTALL_DIR}:"*) ;;
  *)
    PATH_NOTE="
  Add ${INSTALL_DIR} to your PATH (for this shell and future sessions):
    export PATH=\"${INSTALL_DIR}:\$PATH\"
    echo 'export PATH=\"${INSTALL_DIR}:\$PATH\"' >> ~/.bashrc
"
    ;;
esac

if [ "$RUN_SETUP" = 1 ]; then
  if [ "$AIRGAP_READY" = 1 ]; then
    log "Running full air-gap provisioning (environments + checkpoints + hub caches)..."
    "$TARGET" setup --trained --airgap-ready
  else
    log "Provisioning trained detector environments and verified checkpoints..."
    "$TARGET" setup --trained
  fi
fi

cat <<EOF

────────────────────────────────────────────────────────────────────
Verity binary is ready: ${TARGET}
${PATH_NOTE}
Next steps (single-machine online -> air-gapped workstation):
  1. Provision everything once while online (uv, Python runtimes,
     39 evidence-grade detector envs, verified checkpoints & caches):
       verity setup --trained --airgap-ready

  2. Disconnect from the network permanently, verify readiness:
       verity doctor --verify

  3. Start the forensic Web Console (http://127.0.0.1:8000):
       verity serve

Transfer to a permanently offline workstation (via USB/staging host):
  verity airgap export --out /mnt/usb/verity-kit
  # on the offline workstation:
  /mnt/usb/verity-kit/verity airgap import --from /mnt/usb/verity-kit

Checkpoints are SHA-256 verified from ADPer/verity-weights (mirroring
redistributable upstream checkpoints with full author attribution) and
authors' official release URLs for origin-only weights.
────────────────────────────────────────────────────────────────────
EOF
