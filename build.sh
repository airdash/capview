#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="capview"
[ -f "${SCRIPT_DIR}/../../.env" ] && source "${SCRIPT_DIR}/../../.env"
# OUTPUT_DIR="${1:-${OUTPUT_DIR:-${SCRIPT_DIR}/build-output}}"
OUTPUT_DIR="/home/build/capview"
PLATFORM="$(uname -s)"

# ── macOS ────────────────────────────────────────────────────────────

build_macos() {
    echo "==> macOS build (native cargo)"

    # Verify toolchain
    if ! command -v cargo &>/dev/null; then
        if [ -x "$HOME/.cargo/bin/cargo" ]; then
            export PATH="$HOME/.cargo/bin:$PATH"
        else
            echo "error: cargo not found. install rust: https://rustup.rs" >&2
            exit 1
        fi
    fi

    # Homebrew deps — check and report missing
    local missing=()
    for pkg in sdl2 jpeg-turbo; do
        if ! brew --prefix "$pkg" &>/dev/null; then
            missing+=("$pkg")
        fi
    done
    if [ ${#missing[@]} -gt 0 ]; then
        echo "error: missing brew packages: ${missing[*]}" >&2
        echo "  brew install ${missing[*]}" >&2
        exit 1
    fi

    # SDL2 from Homebrew may not be symlinked into /opt/homebrew/lib.
    # Set LIBRARY_PATH so the linker finds it.
    local sdl2_lib
    sdl2_lib="$(brew --prefix sdl2)/lib"
    export LIBRARY_PATH="${sdl2_lib}:${LIBRARY_PATH:-}"

    mkdir -p "${OUTPUT_DIR}"

    echo "  cargo build --release"
    cargo build --release --manifest-path "${SCRIPT_DIR}/Cargo.toml"

    local bin="${SCRIPT_DIR}/target/release/capview"
    cp "${bin}" "${OUTPUT_DIR}/capview"
    chmod 755 "${OUTPUT_DIR}/capview"

    echo ""
    echo "==> done."
    file "${OUTPUT_DIR}/capview"
    ls -lh "${OUTPUT_DIR}/capview"
    echo ""
    echo "run it:"
    echo "  ${OUTPUT_DIR}/capview --device 0"
}

# ── Linux (Docker) ───────────────────────────────────────────────────

build_linux() {
    local IMAGE_NAME="capview-rs-builder"
    local HOST_UID HOST_GID
    HOST_UID="$(id -u)"
    HOST_GID="$(id -g)"

    echo "==> Linux build (Docker)"

    if ! command -v docker &>/dev/null; then
        echo "error: docker not found" >&2
        exit 1
    fi

    mkdir -p "${OUTPUT_DIR}"

    echo "  building ${IMAGE_NAME} image..."
    docker build \
        --progress=plain \
        --target builder \
        -t "${IMAGE_NAME}" \
        "${SCRIPT_DIR}"

    echo "  extracting binary to ${OUTPUT_DIR}..."
    docker run --rm \
        -v "${OUTPUT_DIR}:/output:rw" \
        "${IMAGE_NAME}" \
        sh -c 'rm -f /output/capview && cp /build/target/release/capview /output/capview && chmod 755 /output/capview'

    # Under rootless Docker the container's root maps to a high subuid,
    # so the extracted file is owned by that mapped uid.  Fix it up from
    # the host side where we have real ownership of the directory.
    if [ -f "${OUTPUT_DIR}/capview" ]; then
        chown "${HOST_UID}:${HOST_GID}" "${OUTPUT_DIR}/capview" 2>/dev/null || true
    fi

    echo ""
    echo "==> done."
    file "${OUTPUT_DIR}/capview"
    ls -lh "${OUTPUT_DIR}/capview"
    echo ""
    echo "run it:"
    echo "  ${OUTPUT_DIR}/capview --device /dev/video0 --fps 120 --format NV12"
}

# ── Dispatch ─────────────────────────────────────────────────────────

case "${PLATFORM}" in
    Darwin)  build_macos ;;
    Linux)   build_linux ;;
    *)
        echo "error: unsupported platform '${PLATFORM}'" >&2
        exit 1
        ;;
esac
