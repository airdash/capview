//! GLSL 4.30 compute shader sources for frame generation.
//!
//! Two passes:
//! 1. `BLOCK_MATCH_SRC` — hierarchical block-matching optical flow (coarse → fine)
//! 2. `WARP_SYNTH_SRC`  — motion-compensated frame synthesis

/// Hierarchical block matching optical flow.
///
/// Each work group processes one 16×16 block.  16 threads (one per row)
/// cooperatively compute SAD via shared memory reduction.  Uses `texelFetch`
/// with luminance-only comparison for speed.
///
/// Inputs (samplers):
///   layout(binding = 0) uniform sampler2D uPrev;    — previous frame (mipmapped)
///   layout(binding = 1) uniform sampler2D uCurr;    — current frame (mipmapped)
///   layout(binding = 2) uniform sampler2D uHintMV;  — coarser level's MV (NEAREST)
///
/// Output (image):
///   layout(rgba16f, binding = 0) writeonly uniform image2D uMV;
///
/// Uniforms:
///   uniform ivec2 uSize;     — image dimensions at this pyramid level
///   uniform int   uRadius;   — search radius (pixels at this level)
///   uniform int   uHasHint;  — 0 at coarsest level, 1 when using hint
///   uniform int   uLevel;    — mipmap LOD to sample
pub const BLOCK_MATCH_SRC: &str = r#"#version 430
layout(local_size_x = 16, local_size_y = 1) in;

layout(binding = 0) uniform sampler2D uPrev;
layout(binding = 1) uniform sampler2D uCurr;
layout(binding = 2) uniform sampler2D uHintMV;
layout(rgba16f, binding = 0) writeonly uniform image2D uMV;

uniform ivec2 uSize;
uniform int uRadius;
uniform int uHasHint;
uniform int uLevel;

const int BLOCK = 16;

// Each thread computes a partial SAD for one row; reduced in shared memory.
shared float rowSAD[16];

void main() {
    // gl_WorkGroupID.xy = block index in the MV grid
    // gl_LocalInvocationID.x = row within the 16×16 block (0..15)
    ivec2 blockId = ivec2(gl_WorkGroupID.xy);
    int row = int(gl_LocalInvocationID.x);
    ivec2 origin = blockId * BLOCK;

    int bw = min(BLOCK, uSize.x - origin.x);
    int bh = min(BLOCK, uSize.y - origin.y);

    // Seed from coarser level (motion scaled ×2 for finer grid)
    ivec2 hintOff = ivec2(0);
    if (uHasHint != 0) {
        ivec2 hintBlock = blockId / 2;
        vec4 hintData = texelFetch(uHintMV, hintBlock, 0);
        hintOff = ivec2(round(hintData.xy * 2.0));
    }

    float bestSAD = 1e30;
    ivec2 bestOff = hintOff;

    for (int dy = -uRadius; dy <= uRadius; dy++) {
        for (int dx = -uRadius; dx <= uRadius; dx++) {
            ivec2 testOff = hintOff + ivec2(dx, dy);

            // Each thread computes SAD for its row
            float mySAD = 0.0;
            if (row < bh) {
                int py = origin.y + row;
                int qy = clamp(py + testOff.y, 0, uSize.y - 1);
                for (int bx = 0; bx < bw; bx++) {
                    int px = origin.x + bx;
                    int qx = clamp(px + testOff.x, 0, uSize.x - 1);

                    // Luminance via green channel (fast integer texel fetch)
                    float lP = texelFetch(uPrev, ivec2(px, py), uLevel).g;
                    float lQ = texelFetch(uCurr, ivec2(qx, qy), uLevel).g;
                    mySAD += abs(lP - lQ);
                }
            }

            // Reduce row SADs in shared memory
            rowSAD[row] = mySAD;
            barrier();

            if (row == 0) {
                float total = 0.0;
                for (int i = 0; i < 16; i++) total += rowSAD[i];
                if (total < bestSAD) {
                    bestSAD = total;
                    bestOff = testOff;
                }
            }
            barrier();
        }
    }

    if (row == 0) {
        // Normalize SAD to [0, ~1]: divide by block area (256 texels).
        // 0 = perfect match, 1.0 = significant difference.
        float normSAD = bestSAD / 256.0;
        imageStore(uMV, blockId, vec4(float(bestOff.x), float(bestOff.y), normSAD, 0.0));
    }
}
"#;

/// 3×3 weighted median filter on the motion vector field with temporal dampening.
///
/// Reduces block boundary artifacts by smoothing outlier MVs while
/// preserving dominant motion direction.  Uses the SAD confidence
/// (stored in MV.z) to weight the median — high-SAD (poor match)
/// vectors are treated as less reliable.
///
/// Temporal dampening: blends result with previous frame's filtered MVs
/// (70% new, 30% old) to reduce frame-to-frame MV jitter/flicker.
///
/// Runs in-place via ping-pong: reads from uMVIn, writes to uMVOut.
pub const MV_FILTER_SRC: &str = r#"#version 430
layout(local_size_x = 8, local_size_y = 8) in;

layout(binding = 0) uniform sampler2D uMVIn;
layout(binding = 1) uniform sampler2D uMVPrev;  // previous frame's filtered MVs
layout(rgba16f, binding = 0) writeonly uniform image2D uMVOut;

uniform ivec2 uMVSize; // MV texture dimensions (blocks, not pixels)
uniform int uHasPrev;  // 0 = no previous MVs, 1 = blend with previous

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    if (pos.x >= uMVSize.x || pos.y >= uMVSize.y) return;

    // Gather 3×3 neighbourhood
    vec2 mvs[9];
    float weights[9];
    int count = 0;

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            ivec2 p = clamp(pos + ivec2(dx, dy), ivec2(0), uMVSize - 1);
            vec4 data = texelFetch(uMVIn, p, 0);
            mvs[count] = data.xy;
            // Weight: center pixel gets 2×, low-SAD gets more influence
            float w = (dx == 0 && dy == 0) ? 2.0 : 1.0;
            w /= (data.z + 0.001); // inverse SAD confidence
            weights[count] = w;
            count++;
        }
    }

    // Weighted median: pick the MV that minimises weighted distance to all others
    float bestCost = 1e30;
    vec2 bestMV = mvs[4]; // center as default
    for (int i = 0; i < 9; i++) {
        float cost = 0.0;
        for (int j = 0; j < 9; j++) {
            vec2 d = mvs[i] - mvs[j];
            cost += weights[j] * (abs(d.x) + abs(d.y));
        }
        if (cost < bestCost) {
            bestCost = cost;
            bestMV = mvs[i];
        }
    }

    // Temporal dampening: blend with previous frame's MVs
    if (uHasPrev != 0) {
        vec4 prevData = texelFetch(uMVPrev, pos, 0);
        bestMV = mix(prevData.xy, bestMV, 0.7);
    }

    // Preserve SAD from center for downstream use
    float centerSAD = texelFetch(uMVIn, pos, 0).z;
    imageStore(uMVOut, pos, vec4(bestMV, centerSAD, 0.0));
}
"#;

/// Motion-compensated frame synthesis.
///
/// One invocation per output pixel (8×8 work groups).  Reads the finest-level
/// motion vectors and warps from the source frame(s).
///
/// Inputs (samplers):
///   layout(binding = 0) uniform sampler2D uPrev;
///   layout(binding = 1) uniform sampler2D uCurr;
///   layout(binding = 2) uniform sampler2D uMV;
///
/// Output (image):
///   layout(rgba8, binding = 0) writeonly uniform image2D uOut;
///
/// Uniforms:
///   uniform ivec2 uSize;   — frame width, height in pixels (level 0)
///   uniform float uT;      — interpolation time (0.0 = prev, 1.0 = curr)
///   uniform int   uMode;   — 0 = extrapolate, 1 = interpolate
pub const WARP_SYNTH_SRC: &str = r#"#version 430
layout(local_size_x = 8, local_size_y = 8) in;

layout(binding = 0) uniform sampler2D uPrev;
layout(binding = 1) uniform sampler2D uCurr;
layout(binding = 2) uniform sampler2D uMV;
layout(rgba8, binding = 0) writeonly uniform image2D uOut;

uniform ivec2 uSize;
uniform float uT;    // 0.0 = prev, 1.0 = curr, 0.5 = midpoint
uniform int   uMode; // 0 = extrapolate, 1 = interpolate

const int BLOCK = 16;

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    if (pos.x >= uSize.x || pos.y >= uSize.y) return;

    vec2 invSize = 1.0 / vec2(uSize);

    // Bilinear interpolation of MVs across block boundaries for smooth warps.
    // Sample MV field at the pixel's position (with half-block offset for centering).
    vec2 mvSize = vec2(textureSize(uMV, 0));
    vec2 mvUV = (vec2(pos) / float(BLOCK) + 0.5) / mvSize;
    vec4 mvData = texture(uMV, mvUV); // bilinear-filtered MV
    vec2 mv = mvData.xy;
    float sad = mvData.z;

    // Confidence: smooth sigmoid — avoids hard pixel-snapping at boundaries.
    // Centered at SAD=0.15 with gentle falloff; fully trusted below ~0.05,
    // fully distrusted above ~0.3.
    float confidence = 1.0 / (1.0 + exp((sad - 0.15) * 20.0));

    vec4 warped;
    vec4 source;

    if (uMode == 0) {
        // Extrapolation: predict frame beyond curr by continuing motion.
        // Inverse warp: output pixel at pos came from pos - mv*t in curr.
        vec2 srcPos = vec2(pos) - mv * uT;
        vec2 srcUV = (srcPos + 0.5) * invSize;
        warped = textureLod(uCurr, srcUV, 0.0);
        source = textureLod(uCurr, (vec2(pos) + 0.5) * invSize, 0.0);
    } else {
        // Interpolation: blend forward-warped prev and backward-warped curr.
        // Forward warp from prev: pixel at p in prev is at p+mv*t at time t.
        // Inverse: output pos came from pos - mv*t in prev.
        vec2 fwdPos = vec2(pos) - mv * uT;
        vec2 fwdUV = (fwdPos + 0.5) * invSize;
        vec4 fwd = textureLod(uPrev, fwdUV, 0.0);

        // Backward warp from curr: pixel at p+mv in curr was at p+mv*t at time t.
        // Inverse: output pos came from pos + mv*(1-t) in curr.
        vec2 bwdPos = vec2(pos) + mv * (1.0 - uT);
        vec2 bwdUV = (bwdPos + 0.5) * invSize;
        vec4 bwd = textureLod(uCurr, bwdUV, 0.0);

        warped = mix(fwd, bwd, uT);
        // Source fallback: blend of prev and curr at the original position
        vec2 origUV = (vec2(pos) + 0.5) * invSize;
        source = mix(textureLod(uPrev, origUV, 0.0), textureLod(uCurr, origUV, 0.0), uT);
    }

    // Blend warped result with source fallback based on confidence
    vec4 result = mix(source, warped, confidence);
    imageStore(uOut, pos, result);
}
"#;
