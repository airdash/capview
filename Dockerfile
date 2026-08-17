# syntax=docker/dockerfile:1
FROM rust:1.94-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsdl2-dev \
    libwayland-dev \
    wayland-protocols \
    zlib1g-dev \
    libpulse-dev \
    libgl-dev \
    libegl-dev \
    libvulkan-dev \
    libturbojpeg0-dev \
    glslang-tools \
    pkg-config \
    file \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY Cargo.toml Cargo.lock build.rs ./
COPY src/ src/
COPY protocol/ protocol/

# Compile compute shaders to SPIR-V
RUN glslangValidator -V src/shaders/cas.comp -o src/shaders/cas.spv \
    && glslangValidator -V src/shaders/fsr_easu.comp -o src/shaders/fsr_easu.spv \
    && glslangValidator -V src/shaders/fsr_rcas.comp -o src/shaders/fsr_rcas.spv \
    && glslangValidator -V src/shaders/osd_blend.comp -o src/shaders/osd_blend.spv \
    && glslangValidator -V src/shaders/fg_flow.comp -o src/shaders/fg_flow.spv \
    && glslangValidator -V src/shaders/fg_mv_filter.comp -o src/shaders/fg_mv_filter.spv \
    && glslangValidator -V src/shaders/fg_synth.comp -o src/shaders/fg_synth.spv

RUN cargo generate-lockfile \
    && cargo build --release \
    && strip target/release/capview \
    && echo "--- build info ---" \
    && file target/release/capview \
    && ls -lh target/release/capview \
    && echo "--- dynamic deps ---" \
    && ldd target/release/capview 

FROM scratch AS export
COPY --from=builder /build/target/release/capview /capview
