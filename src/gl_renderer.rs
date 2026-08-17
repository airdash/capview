//! OpenGL render path using glow.
//!
//! Uploads the raw V4L2 frame as a texture and performs YUV→RGB
//! colorspace conversion in a fragment shader.  Supports YUYV, UYVY
//! and NV12.  Falls back gracefully — if GL init fails the caller
//! keeps using the SDL renderer.

use glow::HasContext;
use std::num::NonZeroU32;

use crate::capture::{V4L2_PIX_FMT_NV12, V4L2_PIX_FMT_YUYV, V4L2_PIX_FMT_UYVY, V4L2_PIX_FMT_XRGB32, V4L2_PIX_FMT_P010, PIXFMT_RGB24};

// ── Shaders ─────────────────────────────────────────────────────────

const VERT_SRC: &str = r#"#version 150
in vec2 aPos;
in vec2 aUV;
out vec2 vUV;
uniform vec4 uViewport; // x, y, w, h in NDC
void main() {
    vec2 p = uViewport.xy + aPos * uViewport.zw;
    gl_Position = vec4(p, 0.0, 1.0);
    vUV = aUV;
}
"#;

/// YUYV / UYVY: packed 4:2:2 uploaded as RGBA (width/2 × height).
/// Each texel holds two luma samples + one chroma pair.
const FRAG_YUYV: &str = r#"#version 150
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uTex;
uniform float uTexelW;   // 1.0 / (width/2)
uniform float uBrightness;
uniform float uContrast; // contrast multiplier (1.0 = no change)
uniform float uInvGamma; // 1.0/gamma (1.0 = linear)
uniform int uSwap;       // 0=YUYV  1=UYVY

vec3 yuv2rgb(float y, float u, float v) {
    y = y - 0.0625;
    u = u - 0.5;
    v = v - 0.5;
    return vec3(
        clamp(1.164 * y + 1.596 * v,           0.0, 1.0),
        clamp(1.164 * y - 0.392 * u - 0.813 * v, 0.0, 1.0),
        clamp(1.164 * y + 2.017 * u,           0.0, 1.0)
    );
}

void main() {
    // vUV.x [0,1] maps directly to packed texture [0,1]
    // (each packed texel covers 2 output pixels)
    vec4 t = texture(uTex, vUV);
    // Determine even/odd output pixel within the YUYV pair
    float oddPixel = fract(vUV.x / uTexelW);

    float y, u, v;
    if (uSwap == 0) {
        // YUYV: R=Y0 G=U A=Y1 B=V  (RGBA upload of YUYV bytes)
        u = t.g; v = t.a;
        y = (oddPixel < 0.5) ? t.r : t.b;
    } else {
        // UYVY: R=U G=Y0 B=V A=Y1
        u = t.r; v = t.b;
        y = (oddPixel < 0.5) ? t.g : t.a;
    }

    // Contrast-compensate luma (matches CPU LUT: (y - 0.5) * contrast + 0.5)
    y = clamp((y - 0.5) * uContrast + 0.5, 0.0, 1.0);

    vec3 rgb = pow(yuv2rgb(y, u, v) * uBrightness, vec3(uInvGamma));
    fragColor = vec4(rgb, 1.0);
}
"#;

/// NV12: Y plane (R8) + interleaved UV plane (RG8), two textures.
const FRAG_NV12: &str = r#"#version 150
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uTexY;
uniform sampler2D uTexUV;
uniform float uBrightness;
uniform float uContrast; // contrast multiplier (1.0 = no change)
uniform float uInvGamma; // 1.0/gamma (1.0 = linear)

vec3 yuv2rgb(float y, float u, float v) {
    y = y - 0.0625;
    u = u - 0.5;
    v = v - 0.5;
    return vec3(
        clamp(1.164 * y + 1.596 * v,           0.0, 1.0),
        clamp(1.164 * y - 0.392 * u - 0.813 * v, 0.0, 1.0),
        clamp(1.164 * y + 2.017 * u,           0.0, 1.0)
    );
}

void main() {
    float y = texture(uTexY,  vUV).r;
    // Contrast-compensate luma (matches CPU LUT: (y - 0.5) * contrast + 0.5)
    y = clamp((y - 0.5) * uContrast + 0.5, 0.0, 1.0);
    vec2  uv = texture(uTexUV, vUV).rg;
    vec3 rgb = pow(yuv2rgb(y, uv.r, uv.g) * uBrightness, vec3(uInvGamma));
    fragColor = vec4(rgb, 1.0);
}
"#;

/// XRGB (BGRX in memory): single BGRA texture, swap R/B in shader.
const FRAG_XRGB: &str = r#"#version 150
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uTex;
uniform float uBrightness;
uniform float uContrast;
uniform float uInvGamma;
void main() {
    vec4 t = texture(uTex, vUV);
    vec3 rgb = vec3(t.b, t.g, t.r);
    rgb = clamp((rgb - 0.5) * uContrast + 0.5, 0.0, 1.0);
    rgb = pow(rgb * uBrightness, vec3(uInvGamma));
    fragColor = vec4(rgb, 1.0);
}
"#;

/// P010 (10-bit NV12): Y plane as R16, UV plane as RG16.
/// Hardware normalises the 16-bit value to [0,1]; P010 stores 10-bit values
/// in the upper bits so the mapping is close enough for BT.601 math.
const FRAG_P010: &str = r#"#version 150
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uTexY;
uniform sampler2D uTexUV;
uniform float uBrightness;
uniform float uContrast;
uniform float uInvGamma;

vec3 yuv2rgb(float y, float u, float v) {
    y = y - 0.0625;
    u = u - 0.5;
    v = v - 0.5;
    return vec3(
        clamp(1.164 * y + 1.596 * v,           0.0, 1.0),
        clamp(1.164 * y - 0.392 * u - 0.813 * v, 0.0, 1.0),
        clamp(1.164 * y + 2.017 * u,           0.0, 1.0)
    );
}

void main() {
    float y = texture(uTexY, vUV).r;
    y = clamp((y - 0.5) * uContrast + 0.5, 0.0, 1.0);
    vec2 uv = texture(uTexUV, vUV).rg;
    vec3 rgb = pow(yuv2rgb(y, uv.r, uv.g) * uBrightness, vec3(uInvGamma));
    fragColor = vec4(rgb, 1.0);
}
"#;

/// RGB24 (decoded MJPEG): direct RGB passthrough with brightness/contrast.
const FRAG_RGB: &str = r#"#version 150
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uTex;
uniform float uBrightness;
uniform float uContrast;
uniform float uInvGamma;
void main() {
    vec3 rgb = texture(uTex, vUV).rgb;
    rgb = clamp((rgb - 0.5) * uContrast + 0.5, 0.0, 1.0);
    rgb = pow(rgb * uBrightness, vec3(uInvGamma));
    fragColor = vec4(rgb, 1.0);
}
"#;

/// OSD vertex shader: screen-space quads with UV remapping.
/// Note: computes UV from aPos directly (not aUV which is Y-flipped for video).
const VERT_OSD: &str = r#"#version 150
in vec2 aPos;
in vec2 aUV;
out vec2 vUV;
uniform vec4 uRect;    // ndc_x, ndc_y, ndc_w, ndc_h
uniform vec4 uUVRect;  // u0, v0, u_size, v_size
void main() {
    vec2 p = uRect.xy + aPos * uRect.zw;
    gl_Position = vec4(p, 0.0, 1.0);
    // Use aPos for UV (0..1 in both axes, not the flipped aUV)
    vUV = uUVRect.xy + aPos * uUVRect.zw;
}
"#;

/// OSD fragment shader: alpha-masked texture × solid colour.
const FRAG_OSD: &str = r#"#version 150
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uTex;
uniform vec4 uColor;
void main() {
    float a = texture(uTex, vUV).r;
    fragColor = vec4(uColor.rgb, uColor.a * a);
}
"#;

/// Passthrough RGBA fragment shader for rendering pre-computed textures
/// (e.g. frame-gen output).  Shares VERT_SRC with the video shaders.
/// Note: vUV comes from aUV which is Y-flipped for v4l2 video data.
/// The FBO blit already produced a correctly-oriented texture, so we
/// undo the flip with (vUV.x, 1.0 - vUV.y).
const FRAG_PASSTHROUGH: &str = r#"#version 150
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uTex;
uniform float uBrightness;
uniform float uInvGamma;
void main() {
    vec3 rgb = pow(texture(uTex, vec2(vUV.x, 1.0 - vUV.y)).rgb * uBrightness, vec3(uInvGamma));
    fragColor = vec4(rgb, 1.0);
}
"#;

// ── Scaling fragment shaders ────────────────────────────────────────

/// Sharp bilinear: snaps to texel grid then uses hardware bilinear for
/// sub-pixel, giving pixel-crisp edges for pixel art at non-integer scales.
// (Removed: indistinguishable from bilinear for natural video content)

/// Bicubic Mitchell-Netravali: 4×4 tap interpolation with tunable sharpness.
// (Removed: indistinguishable from bilinear for natural video content)

/// Lanczos2: 4×4 tap windowed sinc for high-quality resampling.
// (Removed: indistinguishable from bilinear for natural video content)

/// AMD Contrast Adaptive Sharpening (CAS) — standalone post-process sharpening.
/// Applied after bilinear upscale to add crispness without a full FSR pipeline.
const FRAG_CAS: &str = r#"#version 150
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uTex;
uniform float uBrightness;
uniform float uInvGamma;
uniform float uSharpness; // 0.0 = off, 1.0 = max CAS
void main() {
    vec2 ts = vec2(1.0) / vec2(textureSize(uTex, 0));
    vec2 uv = vec2(vUV.x, 1.0 - vUV.y);

    // 5-tap CAS kernel: center + 4 cardinal neighbors
    vec3 c = texture(uTex, uv).rgb;
    vec3 n = texture(uTex, uv + vec2(0.0, ts.y)).rgb;
    vec3 s = texture(uTex, uv - vec2(0.0, ts.y)).rgb;
    vec3 e = texture(uTex, uv + vec2(ts.x, 0.0)).rgb;
    vec3 w = texture(uTex, uv - vec2(ts.x, 0.0)).rgb;

    // Per-channel min/max for adaptive sharpening strength
    vec3 mn = min(c, min(min(n, s), min(e, w)));
    vec3 mx = max(c, max(max(n, s), max(e, w)));
    // Compute adaptive weight: sharpen more in low-contrast areas
    vec3 d = max(vec3(0.0), 1.0 - mn / mx);
    vec3 wt = d * (-0.125 * uSharpness); // negative lobe weight

    // Apply sharpening: center + weight*(sum_of_neighbors - 4*center)
    vec3 result = (c + (n + s + e + w) * wt) / (1.0 + 4.0 * wt);
    result = clamp(result, 0.0, 1.0);

    fragColor = vec4(pow(result * uBrightness, vec3(uInvGamma)), 1.0);
}
"#;

/// AMD FidelityFX FSR 1.0 reference shader sources, included at compile time.
const FFX_A_SRC: &str = include_str!("shaders/ffx_a.h");
const FFX_FSR1_SRC: &str = include_str!("shaders/ffx_fsr1.h");

/// FSR GLSL version: macOS caps at GL 4.1, Linux has 4.3+.
#[cfg(target_os = "macos")]
const FSR_GLSL_VER: &str = "#version 410\n";
#[cfg(not(target_os = "macos"))]
const FSR_GLSL_VER: &str = "#version 430\n";

/// Vertex shader for FSR passes — fills entire viewport (no uViewport transform).
const FSR_VERT_SRC: &str = "#version 410\n\
in vec2 aPos;\n\
in vec2 aUV;\n\
out vec2 vUV;\n\
void main() {\n\
    gl_Position = vec4(aPos * 2.0 - 1.0, 0.0, 1.0);\n\
    vUV = aUV;\n\
}\n";

/// EASU wrapper: callbacks + main, concatenated after ffx_a.h + ffx_fsr1.h.
/// Uses texelFetch to emulate textureGather for maximum driver compatibility.
const FSR_EASU_WRAPPER: &str = "\
uniform sampler2D uTex;\n\
uniform uvec4 uCon0;\n\
uniform uvec4 uCon1;\n\
uniform uvec4 uCon2;\n\
uniform uvec4 uCon3;\n\
out vec4 fragColor;\n\
// Emulate textureGather with texelFetch for Intel iGPU compatibility.\n\
// textureGather returns the 2x2 bilinear footprint in order:\n\
//   .x = (i0, j1)  left-upper\n\
//   .y = (i1, j1)  right-upper\n\
//   .z = (i1, j0)  right-lower\n\
//   .w = (i0, j0)  left-lower\n\
AF4 FsrEasuRF(AF2 p) {\n\
    ivec2 b = ivec2(floor(p * vec2(textureSize(uTex, 0)) - 0.5));\n\
    return AF4(\n\
        texelFetch(uTex, b + ivec2(0,1), 0).r,\n\
        texelFetch(uTex, b + ivec2(1,1), 0).r,\n\
        texelFetch(uTex, b + ivec2(1,0), 0).r,\n\
        texelFetch(uTex, b              , 0).r);}\n\
AF4 FsrEasuGF(AF2 p) {\n\
    ivec2 b = ivec2(floor(p * vec2(textureSize(uTex, 0)) - 0.5));\n\
    return AF4(\n\
        texelFetch(uTex, b + ivec2(0,1), 0).g,\n\
        texelFetch(uTex, b + ivec2(1,1), 0).g,\n\
        texelFetch(uTex, b + ivec2(1,0), 0).g,\n\
        texelFetch(uTex, b              , 0).g);}\n\
AF4 FsrEasuBF(AF2 p) {\n\
    ivec2 b = ivec2(floor(p * vec2(textureSize(uTex, 0)) - 0.5));\n\
    return AF4(\n\
        texelFetch(uTex, b + ivec2(0,1), 0).b,\n\
        texelFetch(uTex, b + ivec2(1,1), 0).b,\n\
        texelFetch(uTex, b + ivec2(1,0), 0).b,\n\
        texelFetch(uTex, b              , 0).b);}\n\
void main() {\n\
    AU2 ip = AU2(gl_FragCoord.xy);\n\
    AF3 c;\n\
    FsrEasuF(c, ip, uCon0, uCon1, uCon2, uCon3);\n\
    fragColor = vec4(c, 1.0);\n\
}\n";

/// RCAS wrapper: callbacks + main, concatenated after ffx_a.h + ffx_fsr1.h.
const FSR_RCAS_WRAPPER: &str = "\
uniform sampler2D uTex;\n\
uniform uvec4 uRcasCon;\n\
uniform float uBrightness;\n\
uniform float uInvGamma;\n\
uniform vec2 uViewportOrigin;\n\
out vec4 fragColor;\n\
AF4 FsrRcasLoadF(ASU2 p) { return texelFetch(uTex, ivec2(p), 0); }\n\
void FsrRcasInputF(inout AF1 r, inout AF1 g, inout AF1 b) {}\n\
void main() {\n\
    AU2 ip = AU2(gl_FragCoord.xy - uViewportOrigin);\n\
    AF1 r, g, b;\n\
    FsrRcasF(r, g, b, ip, uRcasCon);\n\
    vec3 c = pow(vec3(r, g, b) * uBrightness, vec3(uInvGamma));\n\
    fragColor = vec4(c, 1.0);\n\
}\n";

/// Polyfills for GLSL 4.2 pack/unpack functions used in ffx_a.h helper
/// definitions.  These helpers are never called by FSR1 (they're for the
/// A_HALF path) but the compiler still parses their definitions.
/// Provides stub implementations that satisfy the compiler; correctness
/// doesn't matter since these code paths are never executed.
const PACK_POLYFILL: &str = "\
#if __VERSION__ < 420\n\
uint packHalf2x16(vec2 v){return uint(v.x*65535.0);}\n\
vec2 unpackHalf2x16(uint v){return vec2(float(v)/65535.0,0.0);}\n\
uint packUnorm2x16(vec2 v){return uint(v.x*65535.0)|(uint(v.y*65535.0)<<16);}\n\
vec2 unpackUnorm2x16(uint v){return vec2(float(v&0xFFFFu)/65535.0,float(v>>16)/65535.0);}\n\
uint packUnorm4x8(vec4 v){return uint(v.x*255.0)|(uint(v.y*255.0)<<8)|(uint(v.z*255.0)<<16)|(uint(v.w*255.0)<<24);}\n\
vec4 unpackUnorm4x8(uint v){return vec4(float(v&0xFFu)/255.0,float((v>>8)&0xFFu)/255.0,float((v>>16)&0xFFu)/255.0,float(v>>24)/255.0);}\n\
#endif\n";

/// Build the full EASU fragment shader source.
fn build_fsr_easu_frag() -> String {
    format!(
        "{}#define A_GPU 1\n#define A_GLSL 1\n{}{}\n#define FSR_EASU_F 1\n{}\n{}",
        FSR_GLSL_VER, PACK_POLYFILL, FFX_A_SRC, FFX_FSR1_SRC, FSR_EASU_WRAPPER
    )
}

/// Build the full RCAS fragment shader source.
fn build_fsr_rcas_frag() -> String {
    format!(
        "{}#define A_GPU 1\n#define A_GLSL 1\n{}{}\n#define FSR_RCAS_F 1\n{}\n{}",
        FSR_GLSL_VER, PACK_POLYFILL, FFX_A_SRC, FFX_FSR1_SRC, FSR_RCAS_WRAPPER
    )
}

/// Compute FsrEasuCon constants on the CPU.
/// Returns (con0, con1, con2, con3) as [u32; 4] arrays,
/// where each u32 stores the bit pattern of a f32.
pub fn fsr_easu_con(
    input_w: f32, input_h: f32,
    output_w: f32, output_h: f32,
) -> ([u32; 4], [u32; 4], [u32; 4], [u32; 4]) {
    let f2u = |f: f32| -> u32 { f.to_bits() };
    let rcp = |x: f32| -> f32 { 1.0 / x };

    // viewport = input (no dynamic resolution)
    let vp_w = input_w;
    let vp_h = input_h;

    let con0 = [
        f2u(vp_w * rcp(output_w)),
        f2u(vp_h * rcp(output_h)),
        f2u(0.5 * vp_w * rcp(output_w) - 0.5),
        f2u(0.5 * vp_h * rcp(output_h) - 0.5),
    ];
    let con1 = [
        f2u(rcp(input_w)),
        f2u(rcp(input_h)),
        f2u(1.0 * rcp(input_w)),
        f2u(-1.0 * rcp(input_h)),
    ];
    let con2 = [
        f2u(-1.0 * rcp(input_w)),
        f2u(2.0 * rcp(input_h)),
        f2u(1.0 * rcp(input_w)),
        f2u(2.0 * rcp(input_h)),
    ];
    let con3 = [
        f2u(0.0 * rcp(input_w)),
        f2u(4.0 * rcp(input_h)),
        0,
        0,
    ];
    (con0, con1, con2, con3)
}

/// Compute FsrRcasCon constant on the CPU.
/// `sharpness`: 0.0 = maximum sharpness, higher = less sharp (in "stops").
pub fn fsr_rcas_con(sharpness_linear: f32) -> [u32; 4] {
    // Direct linear sharpness multiplier for RCAS lobe.
    // 0.0 = no sharpening, 1.0 = standard FSR max, 2.0 = extra aggressive
    [sharpness_linear.to_bits(), 0, 0, 0]
}

// ── Scaling mode ────────────────────────────────────────────────────

/// GL upscaling / downscaling algorithm for the final render pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScaleMode {
    Nearest,
    Bilinear,
    IntegerScale,
    Cas,
    Fsr,
    IntegerFsr,
}

impl ScaleMode {
    #[allow(dead_code)]
    pub fn label(self) -> &'static str {
        match self {
            Self::Nearest => "Nearest",
            Self::Bilinear => "Bilinear",
            Self::IntegerScale => "Integer",
            Self::Cas => "CAS",
            Self::Fsr => "FSR",
            Self::IntegerFsr => "Integer+FSR",
        }
    }

    pub fn from_index(i: usize) -> Self {
        match i {
            0 => Self::Nearest,
            1 => Self::Bilinear,
            2 => Self::IntegerScale,
            3 => Self::Cas,
            4 => Self::Fsr,
            5 => Self::IntegerFsr,
            _ => Self::Bilinear,
        }
    }

    pub fn index(self) -> usize {
        match self {
            Self::Nearest => 0,
            Self::Bilinear => 1,
            Self::IntegerScale => 2,
            Self::Cas => 3,
            Self::Fsr => 4,
            Self::IntegerFsr => 5,
        }
    }

    pub fn config_name(self) -> &'static str {
        match self {
            Self::Nearest => "nearest",
            Self::Bilinear => "bilinear",
            Self::IntegerScale => "integer_scale",
            Self::Cas => "cas",
            Self::Fsr => "fsr",
            Self::IntegerFsr => "integer_fsr",
        }
    }

    pub fn from_config(s: &str) -> Option<Self> {
        match s {
            "nearest" => Some(Self::Nearest),
            "bilinear" | "sharp_bilinear" | "bicubic" | "lanczos" => Some(Self::Bilinear),
            "integer_scale" => Some(Self::IntegerScale),
            "cas" => Some(Self::Cas),
            "fsr" => Some(Self::Fsr),
            "integer_fsr" => Some(Self::IntegerFsr),
            _ => None,
        }
    }

    /// Whether this mode has a sharpness tunable.
    pub fn has_sharpness(self) -> bool {
        matches!(self, Self::Cas | Self::Fsr | Self::IntegerFsr)
    }

    /// Whether this mode requires the GL/VK compute path. The SDL backend
    /// silently falls back to its built-in linear scaling for these modes.
    pub fn requires_shader(self) -> bool {
        matches!(self, Self::IntegerScale | Self::Cas | Self::Fsr | Self::IntegerFsr)
    }
}

// ── Public interface ────────────────────────────────────────────────

pub struct GlRenderer {
    gl: glow::Context,
    prog_packed: glow::Program,   // YUYV / UYVY
    prog_nv12: glow::Program,     // NV12
    vao: glow::VertexArray,
    _vbo: glow::Buffer,
    tex_packed: glow::Texture,    // for YUYV/UYVY
    tex_y: glow::Texture,         // for NV12 Y
    tex_uv: glow::Texture,        // for NV12 UV
    width: u32,
    height: u32,
    pixfmt: u32,
    smooth: bool,
    // Cached video uniform locations (set once at init, used every frame)
    loc_packed_viewport: Option<glow::UniformLocation>,
    loc_packed_brightness: Option<glow::UniformLocation>,
    loc_packed_contrast: Option<glow::UniformLocation>,
    loc_packed_inv_gamma: Option<glow::UniformLocation>,
    loc_packed_swap: Option<glow::UniformLocation>,
    loc_nv12_viewport: Option<glow::UniformLocation>,
    loc_nv12_brightness: Option<glow::UniformLocation>,
    loc_nv12_contrast: Option<glow::UniformLocation>,
    loc_nv12_inv_gamma: Option<glow::UniformLocation>,
    // XRGB / RGB24 / P010 shaders (compiled on demand, only the active format's program is used)
    prog_xrgb: glow::Program,
    loc_xrgb_viewport: Option<glow::UniformLocation>,
    loc_xrgb_brightness: Option<glow::UniformLocation>,
    loc_xrgb_contrast: Option<glow::UniformLocation>,
    loc_xrgb_inv_gamma: Option<glow::UniformLocation>,
    prog_rgb: glow::Program,
    loc_rgb_viewport: Option<glow::UniformLocation>,
    loc_rgb_brightness: Option<glow::UniformLocation>,
    loc_rgb_contrast: Option<glow::UniformLocation>,
    loc_rgb_inv_gamma: Option<glow::UniformLocation>,
    prog_p010: glow::Program,
    loc_p010_viewport: Option<glow::UniformLocation>,
    loc_p010_brightness: Option<glow::UniformLocation>,
    loc_p010_contrast: Option<glow::UniformLocation>,
    loc_p010_inv_gamma: Option<glow::UniformLocation>,
    // OSD rendering
    prog_osd: glow::Program,
    atlas_tex: glow::Texture,
    white_tex: glow::Texture,
    loc_osd_rect: Option<glow::UniformLocation>,
    loc_osd_uv_rect: Option<glow::UniformLocation>,
    loc_osd_color: Option<glow::UniformLocation>,
    // DMA-BUF zero-copy import (optional, Linux only)
    #[cfg(target_os = "linux")]
    dmabuf: Option<crate::dmabuf::DmaBufImporter>,
    #[cfg(target_os = "linux")]
    using_dmabuf: bool,
    pixel_store_set: bool,
    // Passthrough RGBA render (for frame-gen output)
    prog_passthrough: glow::Program,
    loc_pt_viewport: Option<glow::UniformLocation>,
    loc_pt_brightness: Option<glow::UniformLocation>,
    loc_pt_inv_gamma: Option<glow::UniformLocation>,
    // Scaling shaders (upscale pass for render_texture)
    scale_mode: ScaleMode,
    sharpness: f32,
    // CAS (Contrast Adaptive Sharpening) shader
    prog_cas: glow::Program,
    loc_cas_viewport: Option<glow::UniformLocation>,
    loc_cas_brightness: Option<glow::UniformLocation>,
    loc_cas_inv_gamma: Option<glow::UniformLocation>,
    loc_cas_sharpness: Option<glow::UniformLocation>,
    prog_fsr_easu: Option<glow::Program>,
    prog_fsr_rcas: Option<glow::Program>,
    // EASU uniform locations
    loc_easu_con0: Option<glow::UniformLocation>,
    loc_easu_con1: Option<glow::UniformLocation>,
    loc_easu_con2: Option<glow::UniformLocation>,
    loc_easu_con3: Option<glow::UniformLocation>,
    // RCAS uniform locations
    loc_rcas_con: Option<glow::UniformLocation>,
    loc_rcas_brightness: Option<glow::UniformLocation>,
    loc_rcas_inv_gamma: Option<glow::UniformLocation>,
    loc_rcas_viewport_origin: Option<glow::UniformLocation>,
    // EASU intermediate FBO (at output content-area resolution)
    fsr_easu_fbo: Option<glow::Framebuffer>,
    fsr_easu_tex: Option<glow::Texture>,
    fsr_easu_w: u32,
    fsr_easu_h: u32,
    fsr_diag_done: bool,
    pub aspect_mode: crate::config::AspectMode,
    // Intermediate FBO for two-pass scaling (YUV→RGB → scale shader)
    scale_fbo: Option<glow::Framebuffer>,
    scale_rgb_tex: Option<glow::Texture>,
    scale_fbo_w: u32,
    scale_fbo_h: u32,
}

impl GlRenderer {
    /// Create GL context + compile shaders.  `gl_get_proc` is typically
    /// `|s| video.gl_get_proc_address(s) as *const _`.
    pub fn new<F: FnMut(&str) -> *const std::ffi::c_void>(
        mut gl_get_proc: F,
        width: u32,
        height: u32,
        pixfmt: u32,
        smooth: bool,
        debug: bool,
    ) -> anyhow::Result<(Self, GlSavedState)> {
        let gl = unsafe {
            glow::Context::from_loader_function(|s| gl_get_proc(s))
        };

        // Capture SDL's GL state BEFORE we modify anything.
        let sdl_state = save_gl_state(&gl);

        // Print GL context info
        if debug {
            unsafe {
                let vendor = gl.get_parameter_string(glow::VENDOR);
                let renderer = gl.get_parameter_string(glow::RENDERER);
                let version = gl.get_parameter_string(glow::VERSION);
                let glsl = gl.get_parameter_string(glow::SHADING_LANGUAGE_VERSION);
                eprintln!("debug: GL vendor={}", vendor);
                eprintln!("debug: GL renderer={}", renderer);
                eprintln!("debug: GL version={}", version);
                eprintln!("debug: GLSL version={}", glsl);
            }
        }

        let prog_packed = compile_program(&gl, VERT_SRC, FRAG_YUYV)?;
        if debug { eprintln!("debug: GL packed shader compiled OK"); }
        let prog_nv12 = compile_program(&gl, VERT_SRC, FRAG_NV12)?;
        if debug { eprintln!("debug: GL nv12 shader compiled OK"); }
        let prog_xrgb = compile_program(&gl, VERT_SRC, FRAG_XRGB)?;
        let prog_rgb = compile_program(&gl, VERT_SRC, FRAG_RGB)?;
        let prog_p010 = compile_program(&gl, VERT_SRC, FRAG_P010)?;

        check_gl_error(&gl, "after shader compile")?;

        // Fullscreen quad  (pos XY + uv ST)
        let verts: [f32; 16] = [
            // pos       uv
            0.0, 0.0,   0.0, 1.0,
            1.0, 0.0,   1.0, 1.0,
            0.0, 1.0,   0.0, 0.0,
            1.0, 1.0,   1.0, 0.0,
        ];

        let (vao, vbo) = unsafe {
            let vao = gl.create_vertex_array().map_err(|e| anyhow::anyhow!("{}", e))?;
            gl.bind_vertex_array(Some(vao));

            let vbo = gl.create_buffer().map_err(|e| anyhow::anyhow!("{}", e))?;
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
            gl.buffer_data_u8_slice(
                glow::ARRAY_BUFFER,
                bytemuck_cast_slice(&verts),
                glow::STATIC_DRAW,
            );

            // aPos (location 0)
            gl.enable_vertex_attrib_array(0);
            gl.vertex_attrib_pointer_f32(0, 2, glow::FLOAT, false, 16, 0);
            // aUV  (location 1)
            gl.enable_vertex_attrib_array(1);
            gl.vertex_attrib_pointer_f32(1, 2, glow::FLOAT, false, 16, 8);

            gl.bind_vertex_array(None);
            (vao, vbo)
        };

        let filter = if smooth { glow::LINEAR } else { glow::NEAREST } as i32;

        let tex_packed = create_texture(&gl, filter);
        let tex_y = create_texture(&gl, filter);
        let tex_uv = create_texture(&gl, filter);

        // Pre-allocate texture storage
        unsafe {
            match pixfmt {
                V4L2_PIX_FMT_YUYV | V4L2_PIX_FMT_UYVY => {
                    gl.bind_texture(glow::TEXTURE_2D, Some(tex_packed));
                    gl.tex_image_2d(
                        glow::TEXTURE_2D, 0, glow::RGBA8 as i32,
                        (width / 2) as i32, height as i32,
                        0, glow::RGBA, glow::UNSIGNED_BYTE, None,
                    );
                }
                V4L2_PIX_FMT_NV12 => {
                    gl.bind_texture(glow::TEXTURE_2D, Some(tex_y));
                    gl.tex_image_2d(
                        glow::TEXTURE_2D, 0, glow::R8 as i32,
                        width as i32, height as i32,
                        0, glow::RED, glow::UNSIGNED_BYTE, None,
                    );
                    gl.bind_texture(glow::TEXTURE_2D, Some(tex_uv));
                    gl.tex_image_2d(
                        glow::TEXTURE_2D, 0, glow::RG8 as i32,
                        (width / 2) as i32, (height / 2) as i32,
                        0, glow::RG, glow::UNSIGNED_BYTE, None,
                    );
                }
                V4L2_PIX_FMT_XRGB32 => {
                    gl.bind_texture(glow::TEXTURE_2D, Some(tex_packed));
                    gl.tex_image_2d(
                        glow::TEXTURE_2D, 0, glow::RGBA8 as i32,
                        width as i32, height as i32,
                        0, glow::BGRA, glow::UNSIGNED_BYTE, None,
                    );
                }
                V4L2_PIX_FMT_P010 => {
                    gl.bind_texture(glow::TEXTURE_2D, Some(tex_y));
                    gl.tex_image_2d(
                        glow::TEXTURE_2D, 0, glow::R16 as i32,
                        width as i32, height as i32,
                        0, glow::RED, glow::UNSIGNED_SHORT, None,
                    );
                    gl.bind_texture(glow::TEXTURE_2D, Some(tex_uv));
                    gl.tex_image_2d(
                        glow::TEXTURE_2D, 0, glow::RG16 as i32,
                        (width / 2) as i32, (height / 2) as i32,
                        0, glow::RG, glow::UNSIGNED_SHORT, None,
                    );
                }
                PIXFMT_RGB24 => {
                    gl.bind_texture(glow::TEXTURE_2D, Some(tex_packed));
                    gl.tex_image_2d(
                        glow::TEXTURE_2D, 0, glow::RGB8 as i32,
                        width as i32, height as i32,
                        0, glow::RGB, glow::UNSIGNED_BYTE, None,
                    );
                }
                _ => {}
            }
            gl.bind_texture(glow::TEXTURE_2D, None);
        }

        check_gl_error(&gl, "after texture alloc")?;
        if debug {
            let fmt_name = match pixfmt {
                V4L2_PIX_FMT_YUYV => "YUYV",
                V4L2_PIX_FMT_UYVY => "UYVY",
                V4L2_PIX_FMT_NV12 => "NV12",
                V4L2_PIX_FMT_XRGB32 => "XRGB",
                V4L2_PIX_FMT_P010 => "P010",
                PIXFMT_RGB24 => "RGB24",
                _ => "unknown",
            };
            eprintln!("debug: GL textures allocated for {}x{} {}", width, height, fmt_name);
        }

        // Set uniforms that don't change per-frame
        unsafe {
            gl.use_program(Some(prog_packed));
            if let Some(loc) = gl.get_uniform_location(prog_packed, "uTex") {
                gl.uniform_1_i32(Some(&loc), 0);
            }
            if let Some(loc) = gl.get_uniform_location(prog_packed, "uTexelW") {
                gl.uniform_1_f32(Some(&loc), 1.0 / (width as f32 / 2.0));
            }

            gl.use_program(Some(prog_nv12));
            if let Some(loc) = gl.get_uniform_location(prog_nv12, "uTexY") {
                gl.uniform_1_i32(Some(&loc), 0);
            }
            if let Some(loc) = gl.get_uniform_location(prog_nv12, "uTexUV") {
                gl.uniform_1_i32(Some(&loc), 1);
            }

            // XRGB + RGB: single texture on unit 0
            for prog in [prog_xrgb, prog_rgb] {
                gl.use_program(Some(prog));
                if let Some(loc) = gl.get_uniform_location(prog, "uTex") {
                    gl.uniform_1_i32(Some(&loc), 0);
                }
            }

            // P010: same layout as NV12 (Y on unit 0, UV on unit 1)
            gl.use_program(Some(prog_p010));
            if let Some(loc) = gl.get_uniform_location(prog_p010, "uTexY") {
                gl.uniform_1_i32(Some(&loc), 0);
            }
            if let Some(loc) = gl.get_uniform_location(prog_p010, "uTexUV") {
                gl.uniform_1_i32(Some(&loc), 1);
            }

            gl.use_program(None);
        }

        // Cache per-frame uniform locations (avoids glGetUniformLocation every frame)
        let (
            loc_packed_viewport, loc_packed_brightness, loc_packed_contrast, loc_packed_inv_gamma, loc_packed_swap,
            loc_nv12_viewport, loc_nv12_brightness, loc_nv12_contrast, loc_nv12_inv_gamma,
        ) = unsafe {
            gl.use_program(Some(prog_packed));
            let pv  = gl.get_uniform_location(prog_packed, "uViewport");
            let pb  = gl.get_uniform_location(prog_packed, "uBrightness");
            let pc  = gl.get_uniform_location(prog_packed, "uContrast");
            let pg  = gl.get_uniform_location(prog_packed, "uInvGamma");
            let ps  = gl.get_uniform_location(prog_packed, "uSwap");
            if let Some(ref loc) = pc { gl.uniform_1_f32(Some(loc), 1.0); }
            if let Some(ref loc) = pg { gl.uniform_1_f32(Some(loc), 1.0); }

            gl.use_program(Some(prog_nv12));
            let nv  = gl.get_uniform_location(prog_nv12, "uViewport");
            let nb  = gl.get_uniform_location(prog_nv12, "uBrightness");
            let nc  = gl.get_uniform_location(prog_nv12, "uContrast");
            let ng  = gl.get_uniform_location(prog_nv12, "uInvGamma");
            if let Some(ref loc) = nc { gl.uniform_1_f32(Some(loc), 1.0); }
            if let Some(ref loc) = ng { gl.uniform_1_f32(Some(loc), 1.0); }

            gl.use_program(None);
            (pv, pb, pc, pg, ps, nv, nb, nc, ng)
        };

        // Cache uniform locations for XRGB, RGB, P010 shaders
        macro_rules! cache_uniforms {
            ($prog:expr) => {{
                unsafe {
                    gl.use_program(Some($prog));
                    let v = gl.get_uniform_location($prog, "uViewport");
                    let b = gl.get_uniform_location($prog, "uBrightness");
                    let c = gl.get_uniform_location($prog, "uContrast");
                    let g = gl.get_uniform_location($prog, "uInvGamma");
                    if let Some(ref loc) = c { gl.uniform_1_f32(Some(loc), 1.0); }
                    if let Some(ref loc) = g { gl.uniform_1_f32(Some(loc), 1.0); }
                    gl.use_program(None);
                    (v, b, c, g)
                }
            }};
        }
        let (loc_xrgb_viewport, loc_xrgb_brightness, loc_xrgb_contrast, loc_xrgb_inv_gamma) = cache_uniforms!(prog_xrgb);
        let (loc_rgb_viewport, loc_rgb_brightness, loc_rgb_contrast, loc_rgb_inv_gamma) = cache_uniforms!(prog_rgb);
        let (loc_p010_viewport, loc_p010_brightness, loc_p010_contrast, loc_p010_inv_gamma) = cache_uniforms!(prog_p010);

        // ── OSD resources ───────────────────────────────────────────
        let prog_osd = compile_program(&gl, VERT_OSD, FRAG_OSD)?;
        if debug { eprintln!("debug: GL osd shader compiled OK"); }

        let loc_osd_rect = unsafe { gl.get_uniform_location(prog_osd, "uRect") };
        let loc_osd_uv_rect = unsafe { gl.get_uniform_location(prog_osd, "uUVRect") };
        let loc_osd_color = unsafe { gl.get_uniform_location(prog_osd, "uColor") };

        // Set texture sampler once
        unsafe {
            gl.use_program(Some(prog_osd));
            if let Some(loc) = gl.get_uniform_location(prog_osd, "uTex") {
                gl.uniform_1_i32(Some(&loc), 0);
            }
            gl.use_program(None);
        }

        // Font glyph atlas (R8)
        let atlas_pixels = crate::osd::build_gl_atlas();
        let atlas_tex = create_texture(&gl, glow::NEAREST as i32);
        unsafe {
            gl.pixel_store_i32(glow::UNPACK_ROW_LENGTH, 0);
            gl.pixel_store_i32(glow::UNPACK_SKIP_PIXELS, 0);
            gl.pixel_store_i32(glow::UNPACK_SKIP_ROWS, 0);
            gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
            gl.bind_texture(glow::TEXTURE_2D, Some(atlas_tex));
            gl.tex_image_2d(
                glow::TEXTURE_2D, 0, glow::R8 as i32,
                crate::osd::ATLAS_W as i32, crate::osd::ATLAS_H as i32,
                0, glow::RED, glow::UNSIGNED_BYTE, Some(&atlas_pixels),
            );
        }

        // 1×1 white pixel for solid rectangles
        let white_tex = create_texture(&gl, glow::NEAREST as i32);
        unsafe {
            gl.bind_texture(glow::TEXTURE_2D, Some(white_tex));
            gl.tex_image_2d(
                glow::TEXTURE_2D, 0, glow::R8 as i32,
                1, 1, 0, glow::RED, glow::UNSIGNED_BYTE, Some(&[255u8]),
            );
            gl.bind_texture(glow::TEXTURE_2D, None);
        }

        if debug { eprintln!("debug: GL OSD atlas {}x{} + white tex created", crate::osd::ATLAS_W, crate::osd::ATLAS_H); }

        // ── Passthrough RGBA program (for frame-gen output) ─────────
        let prog_passthrough = compile_program(&gl, VERT_SRC, FRAG_PASSTHROUGH)?;
        let (loc_pt_viewport, loc_pt_brightness, loc_pt_inv_gamma) = unsafe {
            gl.use_program(Some(prog_passthrough));
            if let Some(loc) = gl.get_uniform_location(prog_passthrough, "uTex") {
                gl.uniform_1_i32(Some(&loc), 0);
            }
            let v = gl.get_uniform_location(prog_passthrough, "uViewport");
            let b = gl.get_uniform_location(prog_passthrough, "uBrightness");
            let g = gl.get_uniform_location(prog_passthrough, "uInvGamma");
            if let Some(ref loc) = g { gl.uniform_1_f32(Some(loc), 1.0); }
            gl.use_program(None);
            (v, b, g)
        };
        if debug { eprintln!("debug: GL passthrough shader compiled OK"); }

        // Compile CAS (Contrast Adaptive Sharpening) shader
        let prog_cas = compile_program(&gl, VERT_SRC, FRAG_CAS)?;
        let (loc_cas_viewport, loc_cas_brightness, loc_cas_inv_gamma, loc_cas_sharpness) = unsafe {
            gl.use_program(Some(prog_cas));
            if let Some(loc) = gl.get_uniform_location(prog_cas, "uTex") {
                gl.uniform_1_i32(Some(&loc), 0);
            }
            let v = gl.get_uniform_location(prog_cas, "uViewport");
            let b = gl.get_uniform_location(prog_cas, "uBrightness");
            let g = gl.get_uniform_location(prog_cas, "uInvGamma");
            let s = gl.get_uniform_location(prog_cas, "uSharpness");
            if let Some(ref loc) = g { gl.uniform_1_f32(Some(loc), 1.0); }
            gl.use_program(None);
            (v, b, g, s)
        };
        if debug { eprintln!("debug: GL CAS shader compiled OK"); }

        // Compile AMD FSR 1.0 EASU + RCAS shaders
        let (prog_fsr_easu, prog_fsr_rcas,
             loc_easu_con0, loc_easu_con1, loc_easu_con2, loc_easu_con3,
             loc_rcas_con, loc_rcas_brightness, loc_rcas_inv_gamma, loc_rcas_viewport_origin) = {
            let easu_frag = build_fsr_easu_frag();
            let rcas_frag = build_fsr_rcas_frag();
            if debug {
                eprintln!("debug: FSR EASU shader: {} lines, {} bytes", easu_frag.lines().count(), easu_frag.len());
                eprintln!("debug: FSR RCAS shader: {} lines, {} bytes", rcas_frag.lines().count(), rcas_frag.len());
                // Dump shader sources for inspection
                let _ = std::fs::write("/tmp/capview_fsr_easu.glsl", &easu_frag);
                let _ = std::fs::write("/tmp/capview_fsr_rcas.glsl", &rcas_frag);
                eprintln!("debug: FSR shader sources dumped to /tmp/capview_fsr_{{easu,rcas}}.glsl");
            }
            match (
                compile_program(&gl, FSR_VERT_SRC, &easu_frag),
                compile_program(&gl, FSR_VERT_SRC, &rcas_frag),
            ) {
                (Ok(easu), Ok(rcas)) => {
                    let (c0, c1, c2, c3) = unsafe {
                        gl.use_program(Some(easu));
                        if let Some(loc) = gl.get_uniform_location(easu, "uTex") {
                            gl.uniform_1_i32(Some(&loc), 0);
                        }
                        let c0 = gl.get_uniform_location(easu, "uCon0");
                        let c1 = gl.get_uniform_location(easu, "uCon1");
                        let c2 = gl.get_uniform_location(easu, "uCon2");
                        let c3 = gl.get_uniform_location(easu, "uCon3");
                        gl.use_program(None);
                        (c0, c1, c2, c3)
                    };
                    let (rc, rb, rg, rv) = unsafe {
                        gl.use_program(Some(rcas));
                        if let Some(loc) = gl.get_uniform_location(rcas, "uTex") {
                            gl.uniform_1_i32(Some(&loc), 0);
                        }
                        let rc = gl.get_uniform_location(rcas, "uRcasCon");
                        let rb = gl.get_uniform_location(rcas, "uBrightness");
                        let rg = gl.get_uniform_location(rcas, "uInvGamma");
                        let rv = gl.get_uniform_location(rcas, "uViewportOrigin");
                        if let Some(ref loc) = rg { gl.uniform_1_f32(Some(loc), 1.0); }
                        gl.use_program(None);
                        (rc, rb, rg, rv)
                    };
                    eprintln!("fsr: AMD FSR 1.0 EASU+RCAS shaders compiled OK");
                    if c0.is_none() || c1.is_none() || c2.is_none() || c3.is_none() {
                        eprintln!("fsr: WARN: EASU uniform locations missing: con0={} con1={} con2={} con3={}",
                            c0.is_some(), c1.is_some(), c2.is_some(), c3.is_some());
                    }
                    if rc.is_none() {
                        eprintln!("fsr: WARN: RCAS uRcasCon uniform location missing");
                    }
                    (Some(easu), Some(rcas), c0, c1, c2, c3, rc, rb, rg, rv)
                }
                (Err(e), _) | (_, Err(e)) => {
                    eprintln!("warn: FSR shader compilation failed (GL < 4.0?): {e}");
                    eprintln!("warn: FSR scaling mode will fall back to Bilinear");
                    (None, None, None, None, None, None, None, None, None, None)
                }
            }
        };
        if debug { eprintln!("debug: GL scaling shaders compiled OK"); }

        let scale_mode = if smooth { ScaleMode::Bilinear } else { ScaleMode::Nearest };

        // Create intermediate FBO at capture resolution for two-pass scaling
        let (scale_fbo, scale_rgb_tex) = unsafe {
            let tex = gl.create_texture().map_err(|e| anyhow::anyhow!("scale tex: {}", e))?;
            gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            gl.tex_image_2d(
                glow::TEXTURE_2D, 0, glow::RGBA8 as i32,
                width as i32, height as i32, 0,
                glow::RGBA, glow::UNSIGNED_BYTE, None,
            );
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);

            let fbo = gl.create_framebuffer().map_err(|e| anyhow::anyhow!("scale fbo: {}", e))?;
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
            gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER, glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D, Some(tex), 0,
            );
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            gl.bind_texture(glow::TEXTURE_2D, None);
            (Some(fbo), Some(tex))
        };

        Ok((Self {
            gl,
            prog_packed,
            prog_nv12,
            vao,
            _vbo: vbo,
            tex_packed,
            tex_y,
            tex_uv,
            width,
            height,
            pixfmt,
            smooth,
            loc_packed_viewport,
            loc_packed_brightness,
            loc_packed_contrast,
            loc_packed_inv_gamma,
            loc_packed_swap,
            loc_nv12_viewport,
            loc_nv12_brightness,
            loc_nv12_contrast,
            loc_nv12_inv_gamma,
            prog_xrgb, loc_xrgb_viewport, loc_xrgb_brightness, loc_xrgb_contrast, loc_xrgb_inv_gamma,
            prog_rgb, loc_rgb_viewport, loc_rgb_brightness, loc_rgb_contrast, loc_rgb_inv_gamma,
            prog_p010, loc_p010_viewport, loc_p010_brightness, loc_p010_contrast, loc_p010_inv_gamma,
            prog_osd,
            atlas_tex,
            white_tex,
            loc_osd_rect,
            loc_osd_uv_rect,
            loc_osd_color,
            #[cfg(target_os = "linux")]
            dmabuf: None,
            #[cfg(target_os = "linux")]
            using_dmabuf: false,
            pixel_store_set: false,
            prog_passthrough,
            loc_pt_viewport,
            loc_pt_brightness,
            loc_pt_inv_gamma,
            scale_mode,
            sharpness: 0.5,
            prog_cas,
            loc_cas_viewport, loc_cas_brightness, loc_cas_inv_gamma, loc_cas_sharpness,
            prog_fsr_easu,
            prog_fsr_rcas,
            loc_easu_con0, loc_easu_con1, loc_easu_con2, loc_easu_con3,
            loc_rcas_con, loc_rcas_brightness, loc_rcas_inv_gamma, loc_rcas_viewport_origin,
            fsr_easu_fbo: None,
            fsr_easu_tex: None,
            fsr_easu_w: 0,
            fsr_easu_h: 0,
            fsr_diag_done: false,
            aspect_mode: crate::config::AspectMode::Preserve,
            scale_fbo,
            scale_rgb_tex,
            scale_fbo_w: width,
            scale_fbo_h: height,
        }, sdl_state))
    }

    /// Try to initialise DMA-BUF zero-copy import.
    ///
    /// `gl_get_proc` — SDL's `gl_get_proc_address`.
    /// `fds` — one DMA-BUF FD per V4L2 buffer (from `VIDIOC_EXPBUF`).
    ///
    /// On failure the renderer continues to work via `upload()`.
    #[cfg(target_os = "linux")]
    pub fn init_dmabuf<F: FnMut(&str) -> *const std::ffi::c_void>(
        &mut self,
        gl_get_proc: F,
        fds: &[std::os::unix::io::RawFd],
        debug: bool,
    ) -> anyhow::Result<()> {
        let imp = crate::dmabuf::DmaBufImporter::new(
            gl_get_proc, fds, self.width, self.height, self.pixfmt, self.smooth, debug,
        )?;
        self.dmabuf = Some(imp);
        Ok(())
    }

    /// Bind DMA-BUF-backed textures for the given V4L2 buffer index.
    /// Returns `true` if DMA-BUF was used, `false` if unavailable.
    #[cfg(target_os = "linux")]
    pub fn bind_dmabuf(&mut self, buf_index: u32) -> bool {
        if self.dmabuf.is_none() {
            return false;
        }

        let textures: Vec<glow::Texture> = match self.pixfmt {
            V4L2_PIX_FMT_NV12 => vec![self.tex_y, self.tex_uv],
            V4L2_PIX_FMT_YUYV | V4L2_PIX_FMT_UYVY => vec![self.tex_packed],
            V4L2_PIX_FMT_XRGB32 => vec![self.tex_packed],
            V4L2_PIX_FMT_P010 => vec![self.tex_y, self.tex_uv],
            PIXFMT_RGB24 => vec![self.tex_packed],
            _ => return false,
        };

        // Split borrow: self.dmabuf (mut) and self.gl (shared) are disjoint fields.
        let gl = &self.gl;
        if let Some(ref mut dmabuf) = self.dmabuf {
            dmabuf.bind(buf_index, &textures, gl);
        }
        self.using_dmabuf = true;
        true
    }

    #[cfg(target_os = "macos")]
    pub fn bind_dmabuf(&mut self, _buf_index: u32) -> bool { false }

    /// Returns `true` when DMA-BUF zero-copy is available.
    #[cfg(target_os = "linux")]
    pub fn has_dmabuf(&self) -> bool {
        self.dmabuf.is_some()
    }

    #[cfg(target_os = "macos")]
    pub fn has_dmabuf(&self) -> bool { false }

    /// Re-allocate texture storage after switching away from DMA-BUF.
    /// `glEGLImageTargetTexture2DOES` redefines the texture image;
    /// we must call `glTexImage2D` (with NULL data) to reclaim normal
    /// mutable storage before `glTexSubImage2D` will work again.
    fn realloc_textures(&self) {
        let filter = if self.smooth { glow::LINEAR } else { glow::NEAREST } as i32;
        unsafe {
            match self.pixfmt {
                V4L2_PIX_FMT_YUYV | V4L2_PIX_FMT_UYVY => {
                    self.gl.bind_texture(glow::TEXTURE_2D, Some(self.tex_packed));
                    self.gl.tex_image_2d(
                        glow::TEXTURE_2D, 0, glow::RGBA8 as i32,
                        (self.width / 2) as i32, self.height as i32,
                        0, glow::RGBA, glow::UNSIGNED_BYTE, None,
                    );
                    self.gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, filter);
                    self.gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, filter);
                }
                V4L2_PIX_FMT_NV12 => {
                    self.gl.bind_texture(glow::TEXTURE_2D, Some(self.tex_y));
                    self.gl.tex_image_2d(
                        glow::TEXTURE_2D, 0, glow::R8 as i32,
                        self.width as i32, self.height as i32,
                        0, glow::RED, glow::UNSIGNED_BYTE, None,
                    );
                    self.gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, filter);
                    self.gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, filter);
                    self.gl.bind_texture(glow::TEXTURE_2D, Some(self.tex_uv));
                    self.gl.tex_image_2d(
                        glow::TEXTURE_2D, 0, glow::RG8 as i32,
                        (self.width / 2) as i32, (self.height / 2) as i32,
                        0, glow::RG, glow::UNSIGNED_BYTE, None,
                    );
                    self.gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, filter);
                    self.gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, filter);
                }
                V4L2_PIX_FMT_XRGB32 | PIXFMT_RGB24 => {
                    self.gl.bind_texture(glow::TEXTURE_2D, Some(self.tex_packed));
                    let (ifmt, efmt, ety) = if self.pixfmt == V4L2_PIX_FMT_XRGB32 {
                        (glow::RGBA8 as i32, glow::BGRA, glow::UNSIGNED_BYTE)
                    } else {
                        (glow::RGB8 as i32, glow::RGB, glow::UNSIGNED_BYTE)
                    };
                    self.gl.tex_image_2d(glow::TEXTURE_2D, 0, ifmt, self.width as i32, self.height as i32, 0, efmt, ety, None);
                    self.gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, filter);
                    self.gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, filter);
                }
                V4L2_PIX_FMT_P010 => {
                    self.gl.bind_texture(glow::TEXTURE_2D, Some(self.tex_y));
                    self.gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::R16 as i32, self.width as i32, self.height as i32, 0, glow::RED, glow::UNSIGNED_SHORT, None);
                    self.gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, filter);
                    self.gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, filter);
                    self.gl.bind_texture(glow::TEXTURE_2D, Some(self.tex_uv));
                    self.gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::RG16 as i32, (self.width / 2) as i32, (self.height / 2) as i32, 0, glow::RG, glow::UNSIGNED_SHORT, None);
                    self.gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, filter);
                    self.gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, filter);
                }
                _ => {}
            }
            self.gl.bind_texture(glow::TEXTURE_2D, None);
        }
    }

    /// Upload a raw frame (YUYV / UYVY / NV12 bytes).
    pub fn upload(&mut self, data: &[u8]) {
        // If we were previously using DMA-BUF-backed textures, re-allocate
        // normal storage so that glTexSubImage2D works again.
        #[cfg(target_os = "linux")]
        if self.using_dmabuf {
            self.realloc_textures();
            self.using_dmabuf = false;
            self.pixel_store_set = false; // force re-set after DMA-BUF switch
        }
        unsafe {
            // Set pixel-store state once (SDL's NV12 path can clobber
            // UNPACK_ROW_LENGTH, but when GL is active we own the context).
            if !self.pixel_store_set {
                self.gl.pixel_store_i32(glow::UNPACK_ROW_LENGTH, 0);
                self.gl.pixel_store_i32(glow::UNPACK_SKIP_PIXELS, 0);
                self.gl.pixel_store_i32(glow::UNPACK_SKIP_ROWS, 0);
                self.gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
                self.pixel_store_set = true;
            }

            match self.pixfmt {
                V4L2_PIX_FMT_YUYV | V4L2_PIX_FMT_UYVY => {
                    self.gl.bind_texture(glow::TEXTURE_2D, Some(self.tex_packed));
                    self.gl.tex_sub_image_2d(
                        glow::TEXTURE_2D, 0,
                        0, 0,
                        (self.width / 2) as i32, self.height as i32,
                        glow::RGBA, glow::UNSIGNED_BYTE,
                        glow::PixelUnpackData::Slice(data),
                    );
                }
                V4L2_PIX_FMT_NV12 => {
                    let y_size = (self.width * self.height) as usize;
                    self.gl.bind_texture(glow::TEXTURE_2D, Some(self.tex_y));
                    self.gl.tex_sub_image_2d(
                        glow::TEXTURE_2D, 0,
                        0, 0,
                        self.width as i32, self.height as i32,
                        glow::RED, glow::UNSIGNED_BYTE,
                        glow::PixelUnpackData::Slice(&data[..y_size]),
                    );
                    self.gl.bind_texture(glow::TEXTURE_2D, Some(self.tex_uv));
                    self.gl.tex_sub_image_2d(
                        glow::TEXTURE_2D, 0,
                        0, 0,
                        (self.width / 2) as i32, (self.height / 2) as i32,
                        glow::RG, glow::UNSIGNED_BYTE,
                        glow::PixelUnpackData::Slice(&data[y_size..]),
                    );
                }
                V4L2_PIX_FMT_XRGB32 => {
                    self.gl.bind_texture(glow::TEXTURE_2D, Some(self.tex_packed));
                    self.gl.tex_sub_image_2d(
                        glow::TEXTURE_2D, 0, 0, 0,
                        self.width as i32, self.height as i32,
                        glow::BGRA, glow::UNSIGNED_BYTE,
                        glow::PixelUnpackData::Slice(data),
                    );
                }
                V4L2_PIX_FMT_P010 => {
                    let y_size = (self.width * self.height * 2) as usize;
                    self.gl.bind_texture(glow::TEXTURE_2D, Some(self.tex_y));
                    self.gl.tex_sub_image_2d(
                        glow::TEXTURE_2D, 0, 0, 0,
                        self.width as i32, self.height as i32,
                        glow::RED, glow::UNSIGNED_SHORT,
                        glow::PixelUnpackData::Slice(&data[..y_size]),
                    );
                    self.gl.bind_texture(glow::TEXTURE_2D, Some(self.tex_uv));
                    self.gl.tex_sub_image_2d(
                        glow::TEXTURE_2D, 0, 0, 0,
                        (self.width / 2) as i32, (self.height / 2) as i32,
                        glow::RG, glow::UNSIGNED_SHORT,
                        glow::PixelUnpackData::Slice(&data[y_size..]),
                    );
                }
                PIXFMT_RGB24 => {
                    self.gl.bind_texture(glow::TEXTURE_2D, Some(self.tex_packed));
                    self.gl.tex_sub_image_2d(
                        glow::TEXTURE_2D, 0, 0, 0,
                        self.width as i32, self.height as i32,
                        glow::RGB, glow::UNSIGNED_BYTE,
                        glow::PixelUnpackData::Slice(data),
                    );
                }
                _ => {}
            }
        }
    }

    /// Compute NDC (nw, nh) for the given source and window dimensions,
    /// respecting the current aspect mode.
    fn ndc_aspect(&self, src_w: u32, src_h: u32, win_w: u32, win_h: u32) -> (f32, f32) {
        use crate::config::AspectMode;
        match self.aspect_mode {
            AspectMode::Stretch => (2.0, 2.0),
            AspectMode::Zoom => {
                let src_aspect = src_w as f32 / src_h as f32;
                let win_aspect = win_w as f32 / win_h as f32;
                // Use max scale: the larger dimension fills, excess is cropped
                if win_aspect > src_aspect {
                    (2.0, 2.0 * win_aspect / src_aspect)
                } else {
                    (2.0 * src_aspect / win_aspect, 2.0)
                }
            }
            AspectMode::Preserve => {
                let src_aspect = src_w as f32 / src_h as f32;
                let win_aspect = win_w as f32 / win_h as f32;
                if win_aspect > src_aspect {
                    (2.0 * src_aspect / win_aspect, 2.0)
                } else {
                    (2.0, 2.0 / src_aspect * win_aspect)
                }
            }
        }
    }

    /// Compute pixel-space content rect for the given source and window dimensions.
    fn pixel_aspect(&self, src_w: u32, src_h: u32, win_w: u32, win_h: u32) -> (u32, u32, i32, i32) {
        use crate::config::AspectMode;
        match self.aspect_mode {
            AspectMode::Stretch => (win_w, win_h, 0, 0),
            AspectMode::Zoom => {
                let src_aspect = src_w as f32 / src_h as f32;
                let win_aspect = win_w as f32 / win_h as f32;
                let (cw, ch) = if win_aspect > src_aspect {
                    (win_w, (win_w as f32 / src_aspect).round() as u32)
                } else {
                    ((win_h as f32 * src_aspect).round() as u32, win_h)
                };
                (cw, ch, (win_w as i32 - cw as i32) / 2, (win_h as i32 - ch as i32) / 2)
            }
            AspectMode::Preserve => {
                let src_aspect = src_w as f32 / src_h as f32;
                let win_aspect = win_w as f32 / win_h as f32;
                let (cw, ch) = if win_aspect > src_aspect {
                    ((win_h as f32 * src_aspect).round() as u32, win_h)
                } else {
                    (win_w, (win_w as f32 / src_aspect).round() as u32)
                };
                (cw, ch, ((win_w - cw) / 2) as i32, ((win_h - ch) / 2) as i32)
            }
        }
    }

    /// Draw the uploaded frame, letterboxed into `win_w × win_h`.
    /// `contrast` is the luma contrast multiplier (1.0 = normal).
    pub fn render(&mut self, win_w: u32, win_h: u32, brightness: f32, contrast: f32, gamma: f32) {
        let needs_two_pass = matches!(
            self.scale_mode,
            ScaleMode::Fsr | ScaleMode::IntegerFsr
        );

        // ── Pass 1: YUV → RGB ──────────────────────────────────────
        // For advanced scaling: render to intermediate FBO at capture res.
        // For Nearest/Bilinear: render directly to screen.
        let (target_w, target_h, ndc_x, ndc_y, ndc_w, ndc_h) = if needs_two_pass {
            if let (Some(fbo), Some(_)) = (self.scale_fbo, self.scale_rgb_tex) {
                unsafe {
                    self.gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
                    self.gl.viewport(0, 0, self.scale_fbo_w as i32, self.scale_fbo_h as i32);
                    self.gl.clear_color(0.0, 0.0, 0.0, 1.0);
                    self.gl.clear(glow::COLOR_BUFFER_BIT);
                }
                // Fill entire FBO — no letterbox in intermediate
                (self.scale_fbo_w, self.scale_fbo_h, -1.0_f32, -1.0_f32, 2.0_f32, 2.0_f32)
            } else {
                return; // shouldn't happen
            }
        } else {
            // Direct to screen: preserve (letterbox), zoom (crop), or stretch
            let (nw, nh) = self.ndc_aspect(self.width, self.height, win_w, win_h);
            unsafe {
                self.gl.bind_framebuffer(glow::FRAMEBUFFER, None);
                self.gl.viewport(0, 0, win_w as i32, win_h as i32);
                self.gl.clear_color(0.0, 0.0, 0.0, 1.0);
                self.gl.clear(glow::COLOR_BUFFER_BIT);
            }
            (win_w, win_h, -nw / 2.0, -nh / 2.0, nw, nh)
        };
        let _ = target_w;
        let _ = target_h;

        unsafe {
            self.gl.bind_vertex_array(Some(self.vao));

            match self.pixfmt {
                V4L2_PIX_FMT_YUYV | V4L2_PIX_FMT_UYVY => {
                    self.gl.use_program(Some(self.prog_packed));
                    self.gl.active_texture(glow::TEXTURE0);
                    self.gl.bind_texture(glow::TEXTURE_2D, Some(self.tex_packed));

                    if let Some(ref loc) = self.loc_packed_viewport {
                        self.gl.uniform_4_f32(Some(loc), ndc_x, ndc_y, ndc_w, ndc_h);
                    }
                    let b = if needs_two_pass { 1.0 } else { brightness };
                    if let Some(ref loc) = self.loc_packed_brightness {
                        self.gl.uniform_1_f32(Some(loc), b);
                    }
                    let c = if needs_two_pass { 1.0 } else { contrast };
                    if let Some(ref loc) = self.loc_packed_contrast {
                        self.gl.uniform_1_f32(Some(loc), c);
                    }
                    let ig = if needs_two_pass { 1.0 } else { 1.0 / gamma };
                    if let Some(ref loc) = self.loc_packed_inv_gamma {
                        self.gl.uniform_1_f32(Some(loc), ig);
                    }
                    let swap = if self.pixfmt == V4L2_PIX_FMT_UYVY { 1 } else { 0 };
                    if let Some(ref loc) = self.loc_packed_swap {
                        self.gl.uniform_1_i32(Some(loc), swap);
                    }
                }
                V4L2_PIX_FMT_NV12 => {
                    self.gl.use_program(Some(self.prog_nv12));
                    self.gl.active_texture(glow::TEXTURE0);
                    self.gl.bind_texture(glow::TEXTURE_2D, Some(self.tex_y));
                    self.gl.active_texture(glow::TEXTURE1);
                    self.gl.bind_texture(glow::TEXTURE_2D, Some(self.tex_uv));

                    if let Some(ref loc) = self.loc_nv12_viewport {
                        self.gl.uniform_4_f32(Some(loc), ndc_x, ndc_y, ndc_w, ndc_h);
                    }
                    let b = if needs_two_pass { 1.0 } else { brightness };
                    if let Some(ref loc) = self.loc_nv12_brightness {
                        self.gl.uniform_1_f32(Some(loc), b);
                    }
                    let c = if needs_two_pass { 1.0 } else { contrast };
                    if let Some(ref loc) = self.loc_nv12_contrast {
                        self.gl.uniform_1_f32(Some(loc), c);
                    }
                    let ig = if needs_two_pass { 1.0 } else { 1.0 / gamma };
                    if let Some(ref loc) = self.loc_nv12_inv_gamma {
                        self.gl.uniform_1_f32(Some(loc), ig);
                    }
                }
                V4L2_PIX_FMT_XRGB32 => {
                    self.gl.use_program(Some(self.prog_xrgb));
                    self.gl.active_texture(glow::TEXTURE0);
                    self.gl.bind_texture(glow::TEXTURE_2D, Some(self.tex_packed));
                    if let Some(ref loc) = self.loc_xrgb_viewport { self.gl.uniform_4_f32(Some(loc), ndc_x, ndc_y, ndc_w, ndc_h); }
                    let b = if needs_two_pass { 1.0 } else { brightness };
                    if let Some(ref loc) = self.loc_xrgb_brightness { self.gl.uniform_1_f32(Some(loc), b); }
                    let c = if needs_two_pass { 1.0 } else { contrast };
                    if let Some(ref loc) = self.loc_xrgb_contrast { self.gl.uniform_1_f32(Some(loc), c); }
                    let ig = if needs_two_pass { 1.0 } else { 1.0 / gamma };
                    if let Some(ref loc) = self.loc_xrgb_inv_gamma { self.gl.uniform_1_f32(Some(loc), ig); }
                }
                PIXFMT_RGB24 => {
                    self.gl.use_program(Some(self.prog_rgb));
                    self.gl.active_texture(glow::TEXTURE0);
                    self.gl.bind_texture(glow::TEXTURE_2D, Some(self.tex_packed));
                    if let Some(ref loc) = self.loc_rgb_viewport { self.gl.uniform_4_f32(Some(loc), ndc_x, ndc_y, ndc_w, ndc_h); }
                    let b = if needs_two_pass { 1.0 } else { brightness };
                    if let Some(ref loc) = self.loc_rgb_brightness { self.gl.uniform_1_f32(Some(loc), b); }
                    let c = if needs_two_pass { 1.0 } else { contrast };
                    if let Some(ref loc) = self.loc_rgb_contrast { self.gl.uniform_1_f32(Some(loc), c); }
                    let ig = if needs_two_pass { 1.0 } else { 1.0 / gamma };
                    if let Some(ref loc) = self.loc_rgb_inv_gamma { self.gl.uniform_1_f32(Some(loc), ig); }
                }
                V4L2_PIX_FMT_P010 => {
                    self.gl.use_program(Some(self.prog_p010));
                    self.gl.active_texture(glow::TEXTURE0);
                    self.gl.bind_texture(glow::TEXTURE_2D, Some(self.tex_y));
                    self.gl.active_texture(glow::TEXTURE1);
                    self.gl.bind_texture(glow::TEXTURE_2D, Some(self.tex_uv));
                    if let Some(ref loc) = self.loc_p010_viewport { self.gl.uniform_4_f32(Some(loc), ndc_x, ndc_y, ndc_w, ndc_h); }
                    let b = if needs_two_pass { 1.0 } else { brightness };
                    if let Some(ref loc) = self.loc_p010_brightness { self.gl.uniform_1_f32(Some(loc), b); }
                    let c = if needs_two_pass { 1.0 } else { contrast };
                    if let Some(ref loc) = self.loc_p010_contrast { self.gl.uniform_1_f32(Some(loc), c); }
                    let ig = if needs_two_pass { 1.0 } else { 1.0 / gamma };
                    if let Some(ref loc) = self.loc_p010_inv_gamma { self.gl.uniform_1_f32(Some(loc), ig); }
                }
                _ => {}
            }

            self.gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);

            self.gl.bind_vertex_array(None);
            self.gl.use_program(None);
        }

        // ── Pass 2: scale shader → screen ──────────────────────────
        if needs_two_pass {
            if let Some(tex) = self.scale_rgb_tex {
                unsafe { self.gl.bind_framebuffer(glow::FRAMEBUFFER, None); }
                match self.scale_mode {
                    ScaleMode::Fsr if self.has_fsr() => {
                        self.render_fsr(
                            tex,
                            self.scale_fbo_w, self.scale_fbo_h,
                            win_w, win_h,
                            brightness,
                            gamma,
                        );
                    }
                    ScaleMode::IntegerFsr if self.has_fsr() => {
                        self.render_integer_fsr(
                            tex,
                            self.scale_fbo_w, self.scale_fbo_h,
                            win_w, win_h,
                            brightness,
                            gamma,
                        );
                    }
                    _ => {
                        self.render_texture(
                            tex,
                            self.scale_fbo_w, self.scale_fbo_h,
                            win_w, win_h,
                            brightness,
                            gamma,
                        );
                    }
                }
            }
        }
    }

    /// Check for GL errors after a frame (debug path).
    pub fn check_frame_error(&self) -> Option<String> {
        unsafe {
            let err = self.gl.get_error();
            if err != glow::NO_ERROR {
                Some(format!("GL error 0x{:04X} after draw", err))
            } else {
                None
            }
        }
    }

    /// Returns whether smooth filtering is on.
    #[allow(dead_code)]
    pub fn smooth(&self) -> bool { self.smooth }

    /// Current upscaling algorithm.
    #[allow(dead_code)]
    pub fn scale_mode(&self) -> ScaleMode { self.scale_mode }

    /// Set upscaling algorithm.
    pub fn set_scale_mode(&mut self, mode: ScaleMode) {
        self.scale_mode = mode;
        self.smooth = !matches!(mode, ScaleMode::Nearest | ScaleMode::IntegerScale | ScaleMode::IntegerFsr);
        self.fsr_diag_done = false;
        // Update texture filter without reallocating (preserves DMA-BUF)
        let filter = if self.smooth { glow::LINEAR } else { glow::NEAREST } as i32;
        unsafe {
            for &tex in &[self.tex_packed, self.tex_y, self.tex_uv] {
                self.gl.bind_texture(glow::TEXTURE_2D, Some(tex));
                self.gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, filter);
                self.gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, filter);
            }
            self.gl.bind_texture(glow::TEXTURE_2D, None);
        }
    }

    /// Returns true if AMD FSR 1.0 shaders are available (GL 4.0+).
    pub fn has_fsr(&self) -> bool {
        self.prog_fsr_easu.is_some() && self.prog_fsr_rcas.is_some()
    }

    /// Set sharpness for scaling shaders (0–10 integer from OSD).
    pub fn set_sharpness(&mut self, level: u32) {
        self.sharpness = (level.min(10) as f32) / 10.0;
        self.fsr_diag_done = false;
    }

    /// Get current sharpness level (0–10).
    #[allow(dead_code)]
    pub fn sharpness_level(&self) -> u32 {
        (self.sharpness * 10.0).round() as u32
    }

    /// Ensure the FSR EASU FBO exists at the requested size.
    fn ensure_fsr_easu_fbo(&mut self, w: u32, h: u32) {
        if self.fsr_easu_w == w && self.fsr_easu_h == h && self.fsr_easu_fbo.is_some() {
            return;
        }
        let gl = &self.gl;
        unsafe {
            // Clean up old resources
            if let Some(tex) = self.fsr_easu_tex.take() { gl.delete_texture(tex); }
            if let Some(fbo) = self.fsr_easu_fbo.take() { gl.delete_framebuffer(fbo); }

            let tex = gl.create_texture().expect("fsr easu tex");
            gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            gl.tex_image_2d(
                glow::TEXTURE_2D, 0, glow::RGBA8 as i32,
                w as i32, h as i32, 0,
                glow::RGBA, glow::UNSIGNED_BYTE, None,
            );
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);

            let fbo = gl.create_framebuffer().expect("fsr easu fbo");
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
            gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER, glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D, Some(tex), 0,
            );
            let status = gl.check_framebuffer_status(glow::FRAMEBUFFER);
            if status != glow::FRAMEBUFFER_COMPLETE {
                eprintln!("fsr: WARN: EASU FBO incomplete (status=0x{:04X})", status);
            }
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            gl.bind_texture(glow::TEXTURE_2D, None);

            self.fsr_easu_tex = Some(tex);
            self.fsr_easu_fbo = Some(fbo);
            self.fsr_easu_w = w;
            self.fsr_easu_h = h;
        }
    }

    /// AMD FSR 1.0 pipeline: EASU (upscale) → RCAS (sharpen) → screen.
    ///
    /// `src_tex` is the RGB texture at capture resolution (from scale_fbo).
    /// Renders the final output letterboxed into `win_w × win_h`.
    fn render_fsr(
        &mut self,
        src_tex: glow::Texture,
        tex_w: u32, tex_h: u32,
        win_w: u32, win_h: u32,
        brightness: f32,
        gamma: f32,
    ) {
        let (easu_prog, rcas_prog) = match (self.prog_fsr_easu, self.prog_fsr_rcas) {
            (Some(e), Some(r)) => (e, r),
            _ => {
                self.render_texture(src_tex, tex_w, tex_h, win_w, win_h, brightness, gamma);
                return;
            }
        };

        // Compute content area in pixels
        let (content_w, content_h, lb_x, lb_y) = self.pixel_aspect(tex_w, tex_h, win_w, win_h);

        // One-shot diagnostic
        if !self.fsr_diag_done {
            self.fsr_diag_done = true;
            eprintln!("fsr: render_fsr active — input {}x{} → content {}x{} (window {}x{}, letterbox +{},+{}, sharpness {:.2})",
                tex_w, tex_h, content_w, content_h, win_w, win_h, lb_x, lb_y, self.sharpness);
        }

        // Ensure EASU FBO is the right size
        self.ensure_fsr_easu_fbo(content_w, content_h);

        let easu_fbo = match self.fsr_easu_fbo {
            Some(f) => f,
            None => return,
        };
        let easu_tex = match self.fsr_easu_tex {
            Some(t) => t,
            None => return,
        };

        let gl = &self.gl;

        // Compute EASU constants
        let (con0, con1, con2, con3) = fsr_easu_con(
            tex_w as f32, tex_h as f32,
            content_w as f32, content_h as f32,
        );
        // RCAS sharpness: linear 0.0→0.0 (off) to 1.0→1.2 (slight oversharp, no artifacts)
        let rcas_strength = self.sharpness * 1.2;
        let rcas_con = fsr_rcas_con(rcas_strength);

        unsafe {
            // ── EASU pass: src_tex → easu_fbo ──────────────────────
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(easu_fbo));
            gl.viewport(0, 0, content_w as i32, content_h as i32);
            gl.bind_vertex_array(Some(self.vao));
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(src_tex));
            // EASU uses textureGather — needs LINEAR for correct operation
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);

            gl.use_program(Some(easu_prog));
            if let Some(ref loc) = self.loc_easu_con0 {
                gl.uniform_4_u32(Some(loc), con0[0], con0[1], con0[2], con0[3]);
            }
            if let Some(ref loc) = self.loc_easu_con1 {
                gl.uniform_4_u32(Some(loc), con1[0], con1[1], con1[2], con1[3]);
            }
            if let Some(ref loc) = self.loc_easu_con2 {
                gl.uniform_4_u32(Some(loc), con2[0], con2[1], con2[2], con2[3]);
            }
            if let Some(ref loc) = self.loc_easu_con3 {
                gl.uniform_4_u32(Some(loc), con3[0], con3[1], con3[2], con3[3]);
            }
            gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);

            // ── RCAS pass: easu_tex → screen ───────────────────────
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            gl.viewport(0, 0, win_w as i32, win_h as i32);
            gl.clear_color(0.0, 0.0, 0.0, 1.0);
            gl.clear(glow::COLOR_BUFFER_BIT);
            gl.viewport(lb_x, lb_y, content_w as i32, content_h as i32);
            gl.bind_texture(glow::TEXTURE_2D, Some(easu_tex));

            if rcas_strength < 0.01 {
                // Sharpness off — blit EASU output directly with passthrough
                gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR as i32);
                gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
                gl.use_program(Some(self.prog_passthrough));
                if let Some(ref loc) = self.loc_pt_viewport {
                    gl.uniform_4_f32(Some(loc), -1.0, -1.0, 2.0, 2.0);
                }
                if let Some(ref loc) = self.loc_pt_brightness {
                    gl.uniform_1_f32(Some(loc), brightness);
                }
                if let Some(ref loc) = self.loc_pt_inv_gamma {
                    gl.uniform_1_f32(Some(loc), 1.0 / gamma);
                }
            } else {
                gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::NEAREST as i32);
                gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::NEAREST as i32);
                gl.use_program(Some(rcas_prog));
                if let Some(ref loc) = self.loc_rcas_con {
                    gl.uniform_4_u32(Some(loc), rcas_con[0], rcas_con[1], rcas_con[2], rcas_con[3]);
                }
                if let Some(ref loc) = self.loc_rcas_brightness {
                    gl.uniform_1_f32(Some(loc), brightness);
                }
                if let Some(ref loc) = self.loc_rcas_inv_gamma {
                    gl.uniform_1_f32(Some(loc), 1.0 / gamma);
                }
                if let Some(ref loc) = self.loc_rcas_viewport_origin {
                    gl.uniform_2_f32(Some(loc), lb_x as f32, lb_y as f32);
                }
            }
            gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);

            // Restore full viewport for OSD etc.
            gl.viewport(0, 0, win_w as i32, win_h as i32);
            gl.bind_vertex_array(None);
            gl.use_program(None);
        }
    }

    /// Integer pre-scale + FSR RCAS sharpening only (skip EASU).
    /// Nearest-neighbor integer scale to the largest whole multiplier, then
    /// apply RCAS for sharpening. Pixel-perfect for integer ratios.
    fn render_integer_fsr(
        &mut self,
        src_tex: glow::Texture,
        tex_w: u32, tex_h: u32,
        win_w: u32, win_h: u32,
        brightness: f32,
        gamma: f32,
    ) {
        let rcas_prog = match self.prog_fsr_rcas {
            Some(r) => r,
            None => {
                self.render_texture(src_tex, tex_w, tex_h, win_w, win_h, brightness, gamma);
                return;
            }
        };

        // Compute integer scale factor
        let (content_w, content_h, _, _) = self.pixel_aspect(tex_w, tex_h, win_w, win_h);
        let is_fill = self.aspect_mode != crate::config::AspectMode::Preserve;
        let scale_x = content_w / tex_w.max(1);
        let scale_y = content_h / tex_h.max(1);
        let scale = scale_x.min(scale_y).max(1);
        let int_w = if is_fill { content_w } else { tex_w * scale };
        let int_h = if is_fill { content_h } else { tex_h * scale };
        let lb_x = (win_w as i32 - int_w as i32) / 2;
        let lb_y = (win_h as i32 - int_h as i32) / 2;

        if !self.fsr_diag_done {
            self.fsr_diag_done = true;
            eprintln!("integer_fsr: {}x{} → {}x integer → {}x{} (window {}x{}, sharpness {:.2})",
                tex_w, tex_h, scale, int_w, int_h, win_w, win_h, self.sharpness);
        }

        // Ensure EASU FBO is sized to integer-scaled output for nearest blit
        self.ensure_fsr_easu_fbo(int_w, int_h);
        let easu_fbo = match self.fsr_easu_fbo { Some(f) => f, None => return };
        let easu_tex = match self.fsr_easu_tex { Some(t) => t, None => return };

        let rcas_strength = self.sharpness * 1.2;
        let rcas_con = fsr_rcas_con(rcas_strength);

        let gl = &self.gl;
        unsafe {
            // Pass 1: nearest-neighbor blit into FBO at integer scale
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(easu_fbo));
            gl.viewport(0, 0, int_w as i32, int_h as i32);
            gl.bind_vertex_array(Some(self.vao));
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(src_tex));
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::NEAREST as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::NEAREST as i32);
            gl.use_program(Some(self.prog_passthrough));
            if let Some(ref loc) = self.loc_pt_viewport {
                gl.uniform_4_f32(Some(loc), -1.0, -1.0, 2.0, 2.0);
            }
            if let Some(ref loc) = self.loc_pt_brightness {
                gl.uniform_1_f32(Some(loc), 1.0); // brightness applied in RCAS pass
            }
            if let Some(ref loc) = self.loc_pt_inv_gamma {
                gl.uniform_1_f32(Some(loc), 1.0); // gamma applied in RCAS pass
            }
            gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);

            // Pass 2: RCAS sharpening → screen
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            gl.viewport(0, 0, win_w as i32, win_h as i32);
            gl.clear_color(0.0, 0.0, 0.0, 1.0);
            gl.clear(glow::COLOR_BUFFER_BIT);
            gl.viewport(lb_x, lb_y, int_w as i32, int_h as i32);
            gl.bind_texture(glow::TEXTURE_2D, Some(easu_tex));

            if rcas_strength < 0.01 {
                // No sharpening — just blit
                gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::NEAREST as i32);
                gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::NEAREST as i32);
                gl.use_program(Some(self.prog_passthrough));
                if let Some(ref loc) = self.loc_pt_viewport {
                    gl.uniform_4_f32(Some(loc), -1.0, -1.0, 2.0, 2.0);
                }
                if let Some(ref loc) = self.loc_pt_brightness {
                    gl.uniform_1_f32(Some(loc), brightness);
                }
                if let Some(ref loc) = self.loc_pt_inv_gamma {
                    gl.uniform_1_f32(Some(loc), 1.0 / gamma);
                }
            } else {
                gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::NEAREST as i32);
                gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::NEAREST as i32);
                gl.use_program(Some(rcas_prog));
                if let Some(ref loc) = self.loc_rcas_con {
                    gl.uniform_4_u32(Some(loc), rcas_con[0], rcas_con[1], rcas_con[2], rcas_con[3]);
                }
                if let Some(ref loc) = self.loc_rcas_brightness {
                    gl.uniform_1_f32(Some(loc), brightness);
                }
                if let Some(ref loc) = self.loc_rcas_inv_gamma {
                    gl.uniform_1_f32(Some(loc), 1.0 / gamma);
                }
                if let Some(ref loc) = self.loc_rcas_viewport_origin {
                    gl.uniform_2_f32(Some(loc), lb_x as f32, lb_y as f32);
                }
            }
            gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);

            gl.viewport(0, 0, win_w as i32, win_h as i32);
            gl.bind_vertex_array(None);
            gl.use_program(None);
        }
    }

    /// Render a pre-computed RGBA texture (e.g. frame-gen output),
    /// letterboxed into `win_w × win_h`, using the selected scaling mode.
    pub fn render_texture(
        &self, tex: glow::Texture,
        tex_w: u32, tex_h: u32,
        win_w: u32, win_h: u32,
        brightness: f32,
        gamma: f32,
    ) {
        let gl = &self.gl;

        let (ndc_x, ndc_y, ndc_w, ndc_h, filter) = if self.scale_mode == ScaleMode::IntegerScale
            && self.aspect_mode == crate::config::AspectMode::Preserve
        {
            let scale_x = win_w / tex_w.max(1);
            let scale_y = win_h / tex_h.max(1);
            let scale = scale_x.min(scale_y).max(1);
            let out_w = tex_w * scale;
            let out_h = tex_h * scale;
            let nw = 2.0 * out_w as f32 / win_w as f32;
            let nh = 2.0 * out_h as f32 / win_h as f32;
            (-nw / 2.0, -nh / 2.0, nw, nh, glow::NEAREST as i32)
        } else {
            let (nw, nh) = self.ndc_aspect(tex_w, tex_h, win_w, win_h);
            let f = match self.scale_mode {
                ScaleMode::Nearest | ScaleMode::IntegerScale | ScaleMode::IntegerFsr => glow::NEAREST,
                _ => glow::LINEAR,
            } as i32;
            (-nw / 2.0, -nh / 2.0, nw, nh, f)
        };

        unsafe {
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            gl.viewport(0, 0, win_w as i32, win_h as i32);
            gl.clear_color(0.0, 0.0, 0.0, 1.0);
            gl.clear(glow::COLOR_BUFFER_BIT);
            gl.bind_vertex_array(Some(self.vao));
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, filter);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, filter);

            match self.scale_mode {
                ScaleMode::Cas => {
                    gl.use_program(Some(self.prog_cas));
                    if let Some(ref loc) = self.loc_cas_viewport {
                        gl.uniform_4_f32(Some(loc), ndc_x, ndc_y, ndc_w, ndc_h);
                    }
                    if let Some(ref loc) = self.loc_cas_brightness {
                        gl.uniform_1_f32(Some(loc), brightness);
                    }
                    if let Some(ref loc) = self.loc_cas_inv_gamma {
                        gl.uniform_1_f32(Some(loc), 1.0 / gamma);
                    }
                    if let Some(ref loc) = self.loc_cas_sharpness {
                        gl.uniform_1_f32(Some(loc), self.sharpness);
                    }
                }
                _ => {
                    // Passthrough for Nearest, Bilinear, IntegerScale, and FSR/IntegerFsr fallbacks
                    gl.use_program(Some(self.prog_passthrough));
                    if let Some(ref loc) = self.loc_pt_viewport {
                        gl.uniform_4_f32(Some(loc), ndc_x, ndc_y, ndc_w, ndc_h);
                    }
                    if let Some(ref loc) = self.loc_pt_brightness {
                        gl.uniform_1_f32(Some(loc), brightness);
                    }
                    if let Some(ref loc) = self.loc_pt_inv_gamma {
                        gl.uniform_1_f32(Some(loc), 1.0 / gamma);
                    }
                }
            }

            gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
            gl.bind_vertex_array(None);
            gl.use_program(None);
        }
    }

    // ── OSD rendering (native GL) ───────────────────────────────────

    /// Set up GL state for 2D OSD overlay rendering.
    pub fn begin_osd(&self, win_w: u32, win_h: u32) {
        unsafe {
            self.gl.viewport(0, 0, win_w as i32, win_h as i32);
            self.gl.enable(glow::BLEND);
            self.gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);
            self.gl.use_program(Some(self.prog_osd));
            self.gl.bind_vertex_array(Some(self.vao));
            self.gl.active_texture(glow::TEXTURE0);
        }
    }

    /// Draw a solid filled rectangle (screen pixel coords, origin top-left).
    pub fn osd_rect(&self, x: i32, y: i32, w: u32, h: u32, color: [f32; 4], win_w: u32, win_h: u32) {
        let (nx, ny, nw, nh) = pixel_to_ndc(x, y, w, h, win_w, win_h);
        unsafe {
            self.gl.bind_texture(glow::TEXTURE_2D, Some(self.white_tex));
            if let Some(ref loc) = self.loc_osd_rect {
                self.gl.uniform_4_f32(Some(loc), nx, ny, nw, nh);
            }
            if let Some(ref loc) = self.loc_osd_uv_rect {
                // White 1x1 texture — UV doesn't matter, but keep consistent
                self.gl.uniform_4_f32(Some(loc), 0.0, 0.0, 1.0, 1.0);
            }
            if let Some(ref loc) = self.loc_osd_color {
                self.gl.uniform_4_f32(Some(loc), color[0], color[1], color[2], color[3]);
            }
            self.gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
        }
    }

    /// Draw a text string using the glyph atlas (screen pixel coords, origin top-left).
    pub fn osd_text(&self, text: &str, x: i32, y: i32, scale: u32, color: [f32; 4], win_w: u32, win_h: u32) {
        let gw = crate::osd::GLYPH_W * scale;
        let gh = crate::osd::GLYPH_H * scale;
        let atlas_w = crate::osd::ATLAS_W as f32;
        let glyph_u_size = crate::osd::GLYPH_W as f32 / atlas_w;

        unsafe {
            self.gl.bind_texture(glow::TEXTURE_2D, Some(self.atlas_tex));
            if let Some(ref loc) = self.loc_osd_color {
                self.gl.uniform_4_f32(Some(loc), color[0], color[1], color[2], color[3]);
            }

            for (i, ch) in text.chars().enumerate() {
                let code = ch as u32;
                if code < crate::osd::FIRST_CHAR as u32 || code > crate::osd::LAST_CHAR as u32 {
                    continue;
                }
                let ci = (code - crate::osd::FIRST_CHAR as u32) as usize;
                let cx = x + (i as u32 * gw) as i32;
                let (nx, ny, nw, nh) = pixel_to_ndc(cx, y, gw, gh, win_w, win_h);
                let u0 = ci as f32 * glyph_u_size;

                if let Some(ref loc) = self.loc_osd_rect {
                    self.gl.uniform_4_f32(Some(loc), nx, ny, nw, nh);
                }
                if let Some(ref loc) = self.loc_osd_uv_rect {
                    // V flipped: aPos.y goes bottom→top but atlas row 0 is top
                    self.gl.uniform_4_f32(Some(loc), u0, 1.0, glyph_u_size, -1.0);
                }
                self.gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
            }
        }
    }

    /// Tear down OSD rendering state.
    pub fn end_osd(&self) {
        unsafe {
            self.gl.disable(glow::BLEND);
            self.gl.bind_vertex_array(None);
            self.gl.use_program(None);
        }
    }

    // ── GL state save / restore (SDL2 compatibility) ────────────────
    //
    // SDL2's accelerated renderer caches GL state internally.  If we
    // change programs, textures, blend mode, etc. via raw GL, the cache
    // goes stale and OSD rendering breaks.  Save before our GL work,
    // restore after.

    /// Call after switching back from SDL renderer to force GL state re-setup.
    /// SDL's internal renderer clobbers pixel-store params and may bind its own FBO.
    pub fn reclaim_context(&mut self) {
        self.pixel_store_set = false;
        unsafe {
            self.gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        }
    }

    /// Snapshot the GL state that SDL2's renderer tracks.
    #[allow(dead_code)]
    pub fn save_state(&self) -> GlSavedState {
        save_gl_state(&self.gl)
    }

    /// Restore previously saved GL state.
    pub fn restore_state(&self, s: &GlSavedState) {
        let gl = &self.gl;
        unsafe {
            let prog = NonZeroU32::new(s.program as u32)
                .map(glow::NativeProgram);
            gl.use_program(prog);

            gl.active_texture(glow::TEXTURE0);
            let t0 = NonZeroU32::new(s.texture0 as u32)
                .map(glow::NativeTexture);
            gl.bind_texture(glow::TEXTURE_2D, t0);

            gl.active_texture(glow::TEXTURE1);
            let t1 = NonZeroU32::new(s.texture1 as u32)
                .map(glow::NativeTexture);
            gl.bind_texture(glow::TEXTURE_2D, t1);

            gl.active_texture(s.active_tex as u32);

            let va = NonZeroU32::new(s.vao as u32)
                .map(glow::NativeVertexArray);
            gl.bind_vertex_array(va);

            let ab = NonZeroU32::new(s.array_buf as u32)
                .map(glow::NativeBuffer);
            gl.bind_buffer(glow::ARRAY_BUFFER, ab);

            if s.blend { gl.enable(glow::BLEND); } else { gl.disable(glow::BLEND); }
            gl.blend_func(s.blend_src as u32, s.blend_dst as u32);

            gl.viewport(s.viewport[0], s.viewport[1], s.viewport[2], s.viewport[3]);
            gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, s.unpack_align);
            gl.pixel_store_i32(glow::UNPACK_ROW_LENGTH, s.unpack_row_length);
        }
    }
}

/// Saved GL state for SDL2 renderer compatibility.
pub struct GlSavedState {
    program: i32,
    active_tex: i32,
    texture0: i32,
    texture1: i32,
    vao: i32,
    array_buf: i32,
    blend: bool,
    blend_src: i32,
    blend_dst: i32,
    viewport: [i32; 4],
    unpack_align: i32,
    unpack_row_length: i32,
}

impl Drop for GlRenderer {
    fn drop(&mut self) {
        unsafe {
            self.gl.delete_program(self.prog_packed);
            self.gl.delete_program(self.prog_nv12);
            self.gl.delete_program(self.prog_osd);
            self.gl.delete_program(self.prog_passthrough);
            if let Some(p) = self.prog_fsr_easu { self.gl.delete_program(p); }
            if let Some(p) = self.prog_fsr_rcas { self.gl.delete_program(p); }
            self.gl.delete_vertex_array(self.vao);
            self.gl.delete_buffer(self._vbo);
            self.gl.delete_texture(self.tex_packed);
            self.gl.delete_texture(self.tex_y);
            self.gl.delete_texture(self.tex_uv);
            self.gl.delete_texture(self.atlas_tex);
            self.gl.delete_texture(self.white_tex);
            if let Some(tex) = self.scale_rgb_tex {
                self.gl.delete_texture(tex);
            }
            if let Some(fbo) = self.scale_fbo {
                self.gl.delete_framebuffer(fbo);
            }
            if let Some(tex) = self.fsr_easu_tex {
                self.gl.delete_texture(tex);
            }
            if let Some(fbo) = self.fsr_easu_fbo {
                self.gl.delete_framebuffer(fbo);
            }
        }
    }
}

// ── Helpers ─────────────────────────────────────────────────────────

fn compile_program(
    gl: &glow::Context,
    vert_src: &str,
    frag_src: &str,
) -> anyhow::Result<glow::Program> {
    unsafe {
        let prog = gl.create_program().map_err(|e| anyhow::anyhow!("{}", e))?;

        let vs = gl.create_shader(glow::VERTEX_SHADER).map_err(|e| anyhow::anyhow!("{}", e))?;
        gl.shader_source(vs, vert_src);
        gl.compile_shader(vs);
        if !gl.get_shader_compile_status(vs) {
            let log = gl.get_shader_info_log(vs);
            gl.delete_shader(vs);
            anyhow::bail!("vertex shader: {}", log);
        }

        let fs = gl.create_shader(glow::FRAGMENT_SHADER).map_err(|e| anyhow::anyhow!("{}", e))?;
        gl.shader_source(fs, frag_src);
        gl.compile_shader(fs);
        if !gl.get_shader_compile_status(fs) {
            let log = gl.get_shader_info_log(fs);
            gl.delete_shader(vs);
            gl.delete_shader(fs);
            anyhow::bail!("fragment shader: {}", log);
        }

        // Bind attrib locations before linking
        gl.bind_attrib_location(prog, 0, "aPos");
        gl.bind_attrib_location(prog, 1, "aUV");

        gl.attach_shader(prog, vs);
        gl.attach_shader(prog, fs);
        gl.link_program(prog);
        gl.delete_shader(vs);
        gl.delete_shader(fs);

        if !gl.get_program_link_status(prog) {
            let log = gl.get_program_info_log(prog);
            gl.delete_program(prog);
            anyhow::bail!("link: {}", log);
        }

        Ok(prog)
    }
}

fn create_texture(gl: &glow::Context, filter: i32) -> glow::Texture {
    unsafe {
        let tex = gl.create_texture().expect("GL texture");
        gl.bind_texture(glow::TEXTURE_2D, Some(tex));
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, filter);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, filter);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
        tex
    }
}

/// Zero-copy cast &[f32] → &[u8] (same as bytemuck::cast_slice).
fn bytemuck_cast_slice(data: &[f32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * std::mem::size_of::<f32>(),
        )
    }
}

/// Check for GL errors and bail with context string.
fn check_gl_error(gl: &glow::Context, context: &str) -> anyhow::Result<()> {
    unsafe {
        let err = gl.get_error();
        if err != glow::NO_ERROR {
            anyhow::bail!("GL error 0x{:04X} at {}", err, context);
        }
    }
    Ok(())
}

/// Snapshot the current GL state (call BEFORE modifying it).
fn save_gl_state(gl: &glow::Context) -> GlSavedState {
    unsafe {
        let program = gl.get_parameter_i32(glow::CURRENT_PROGRAM);
        let active_tex = gl.get_parameter_i32(glow::ACTIVE_TEXTURE);
        gl.active_texture(glow::TEXTURE0);
        let texture0 = gl.get_parameter_i32(glow::TEXTURE_BINDING_2D);
        gl.active_texture(glow::TEXTURE1);
        let texture1 = gl.get_parameter_i32(glow::TEXTURE_BINDING_2D);
        gl.active_texture(active_tex as u32);
        let vao = gl.get_parameter_i32(glow::VERTEX_ARRAY_BINDING);
        let array_buf = gl.get_parameter_i32(glow::ARRAY_BUFFER_BINDING);
        let blend = gl.is_enabled(glow::BLEND);
        let blend_src = gl.get_parameter_i32(glow::BLEND_SRC_RGB);
        let blend_dst = gl.get_parameter_i32(glow::BLEND_DST_RGB);
        let mut viewport = [0i32; 4];
        gl.get_parameter_i32_slice(glow::VIEWPORT, &mut viewport);
        let unpack_align = gl.get_parameter_i32(glow::UNPACK_ALIGNMENT);
        let unpack_row_length = gl.get_parameter_i32(glow::UNPACK_ROW_LENGTH);
        GlSavedState {
            program, active_tex, texture0, texture1,
            vao, array_buf, blend, blend_src, blend_dst,
            viewport, unpack_align, unpack_row_length,
        }
    }
}

/// Convert screen pixel rect (origin top-left) to NDC for the unit quad.
/// The quad's pos attribute goes from (0,0) at NDC bottom-left to (1,1)
/// at NDC top-right.  Screen-space has Y-down, NDC has Y-up.
fn pixel_to_ndc(x: i32, y: i32, w: u32, h: u32, win_w: u32, win_h: u32) -> (f32, f32, f32, f32) {
    let wf = win_w as f32;
    let hf = win_h as f32;
    // NDC x: map pixel x [0, win_w] to [-1, 1]
    let ndc_x = (x as f32 / wf) * 2.0 - 1.0;
    // NDC y: screen y is top-down. The quad's pos (0,0) is the bottom
    // of the rect in NDC (most negative Y), and pos (0,1) is the top.
    // So ndc_y = bottom edge in NDC, ndc_h is positive upward.
    let ndc_y = 1.0 - ((y as f32 + h as f32) / hf) * 2.0;
    let ndc_w = (w as f32 / wf) * 2.0;
    let ndc_h = (h as f32 / hf) * 2.0;
    (ndc_x, ndc_y, ndc_w, ndc_h)
}
