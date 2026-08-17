#!/usr/bin/env bash
# Compile all compute shaders to SPIR-V. The fg_*.spv are .gitignored and
# required by include_bytes! at compile time, so on a fresh clone this step
# is mandatory on BOTH platforms — the build will not link without it.
# Kept in lockstep with the Dockerfile (source of truth).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

for comp in src/shaders/*.comp; do
    glslangValidator -V "${comp}" -o "${comp%.comp}.spv"
done
