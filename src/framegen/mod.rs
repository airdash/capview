//! GPU frame generation — synthesize intermediate frames via OpenGL compute.
//!
//! Produces frames between real capture frames to fill gaps when the display
//! runs at a higher refresh rate than the capture device (e.g. 120 Hz display,
//! 60 fps capture).
//!
//! Two modes:
//! - **Extrapolate**: predict frame N+0.5 from N-1 and N (low latency)
//! - **Interpolate**: synthesize N+0.5 from N and N+1 (better quality, +1 frame latency)
//!
//! Uses a multi-level image pyramid for coarse-to-fine optical flow, giving an
//! effective motion search range of ±46 pixels at Quality preset.
//!
//! Requires OpenGL 4.3 (compute shaders).  Returns `None` if unavailable.

use glow::HasContext;

mod shaders;
pub mod vk;
#[cfg(feature = "rife")]
pub mod rife;

// ── Constants ───────────────────────────────────────────────────────

/// Number of pyramid levels (L0 = full, L1 = ½, L2 = ¼, L3 = ⅛).
const NUM_PYRAMID_LEVELS: usize = 4;

// ── Public types ────────────────────────────────────────────────────

/// Frame generation operating mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameGenMode {
    Off,
    Extrapolate,
    Interpolate,
    #[cfg(feature = "rife")]
    Rife,
}

impl FrameGenMode {
    pub fn next(self) -> Self {
        match self {
            Self::Off => Self::Extrapolate,
            Self::Extrapolate => Self::Interpolate,
            #[cfg(feature = "rife")]
            Self::Interpolate => Self::Rife,
            #[cfg(feature = "rife")]
            Self::Rife => Self::Off,
            #[cfg(not(feature = "rife"))]
            Self::Interpolate => Self::Off,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Off => "Off",
            Self::Extrapolate => "Extrapolate",
            Self::Interpolate => "Interpolate",
            #[cfg(feature = "rife")]
            Self::Rife => "RIFE",
        }
    }

    /// Number of variants (used for OSD index calculations).
    #[allow(dead_code)]
    pub fn count() -> usize {
        #[cfg(feature = "rife")]
        { 4 }
        #[cfg(not(feature = "rife"))]
        { 3 }
    }

    pub fn from_index(i: usize) -> Self {
        match i {
            0 => Self::Off,
            1 => Self::Extrapolate,
            2 => Self::Interpolate,
            #[cfg(feature = "rife")]
            3 => Self::Rife,
            _ => Self::Off,
        }
    }

    pub fn index(self) -> usize {
        match self {
            Self::Off => 0,
            Self::Extrapolate => 1,
            Self::Interpolate => 2,
            #[cfg(feature = "rife")]
            Self::Rife => 3,
        }
    }
}

/// Quality preset controlling pyramid depth and search radii.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameGenQuality {
    Fast,
    Balanced,
    Quality,
}

impl FrameGenQuality {
    /// Active pyramid levels (starting from L0).
    pub fn levels(self) -> usize {
        match self {
            Self::Fast => 2,
            Self::Balanced => 3,
            Self::Quality => 4,
        }
    }

    /// Search radius per level [L0..L3]. Only `levels()` entries are used.
    pub fn radii(self) -> [i32; NUM_PYRAMID_LEVELS] {
        match self {
            Self::Fast =>     [2, 4, 0, 0],
            Self::Balanced => [2, 2, 4, 0],
            Self::Quality =>  [2, 2, 2, 4],
        }
    }

    #[allow(dead_code)]
    pub fn label(self) -> &'static str {
        match self {
            Self::Fast => "Fast",
            Self::Balanced => "Balanced",
            Self::Quality => "Quality",
        }
    }
}

/// Cumulative debug statistics for frame generation.
#[derive(Clone, Debug, Default)]
pub struct FrameGenStats {
    /// Time of last generate() call in microseconds.
    pub last_gen_us: u64,
    /// Number of synthetic frames produced.
    pub synth_count: u64,
    /// Number of times generate() returned false (missed frames).
    pub miss_count: u64,
}

// ── Cached uniform locations ────────────────────────────────────────

struct FlowUniforms {
    size: Option<glow::UniformLocation>,
    radius: Option<glow::UniformLocation>,
    has_hint: Option<glow::UniformLocation>,
    level: Option<glow::UniformLocation>,
}

struct SynthUniforms {
    size: Option<glow::UniformLocation>,
    t: Option<glow::UniformLocation>,
    mode: Option<glow::UniformLocation>,
}

// ── Frame generator ─────────────────────────────────────────────────

pub struct FrameGen {
    gl: glow::Context,
    mode: FrameGenMode,
    quality: FrameGenQuality,
    width: u32,
    height: u32,
    /// Capture (source) resolution — internal textures never exceed this.
    cap_w: u32,
    cap_h: u32,

    /// Previous and current frames (RGBA8 with mipmap pyramid).
    tex_prev: glow::Texture,
    tex_curr: glow::Texture,
    /// Synthesised output (RGBA8, no mipmaps).
    tex_out: glow::Texture,
    /// Motion vector field per pyramid level (RGBA16F, RG = dx/dy).
    mv_levels: Vec<glow::Texture>,
    /// Second MV texture for ping-pong median filter at L0.
    tex_mv_filtered: glow::Texture,
    /// Previous frame's filtered MVs for temporal dampening.
    tex_mv_prev: glow::Texture,
    /// Whether we have valid previous MVs (false until second frame).
    has_prev_mv: bool,
    /// Image dimensions at each pyramid level.
    level_dims: Vec<(u32, u32)>,

    fbo_capture: glow::Framebuffer,
    frame_count: u64,
    /// True when cached MVs are valid for the current frame pair.
    /// Set false in push_frame(), set true after first generate() computes MVs.
    mv_valid: bool,

    prog_flow: glow::Program,
    prog_synth: glow::Program,
    prog_mv_filter: glow::Program,
    flow_locs: FlowUniforms,
    synth_locs: SynthUniforms,
    filter_loc_mv_size: Option<glow::UniformLocation>,
    filter_loc_has_prev: Option<glow::UniformLocation>,

    stats: FrameGenStats,
}

impl FrameGen {
    /// Create the frame generator.  Returns `None` if GL 4.3 compute is
    /// unavailable or shader compilation fails.
    pub fn new<F: FnMut(&str) -> *const std::ffi::c_void>(
        mut gl_get_proc: F,
        width: u32,
        height: u32,
        debug: bool,
    ) -> Option<Self> {
        let gl = unsafe {
            glow::Context::from_loader_function(|s| gl_get_proc(s))
        };

        if !check_compute_support(&gl) {
            if debug { eprintln!("framegen: GL 4.3 compute not available"); }
            return None;
        }

        // Compile shaders first (fail before allocating textures)
        let prog_flow = match compile_compute(&gl, shaders::BLOCK_MATCH_SRC) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("framegen: flow shader: {}", e);
                return None;
            }
        };
        let prog_synth = match compile_compute(&gl, shaders::WARP_SYNTH_SRC) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("framegen: synth shader: {}", e);
                unsafe { gl.delete_program(prog_flow); }
                return None;
            }
        };
        let prog_mv_filter = match compile_compute(&gl, shaders::MV_FILTER_SRC) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("framegen: mv filter shader: {}", e);
                unsafe { gl.delete_program(prog_flow); gl.delete_program(prog_synth); }
                return None;
            }
        };

        // Compute pyramid level dimensions (at capture resolution)
        let mut level_dims = Vec::with_capacity(NUM_PYRAMID_LEVELS);
        let (mut w, mut h) = (width, height);
        for _ in 0..NUM_PYRAMID_LEVELS {
            level_dims.push((w, h));
            w = (w + 1) / 2;
            h = (h + 1) / 2;
        }

        // Source textures with mipmap storage for the pyramid
        let tex_prev = unsafe { create_rgba_texture_mip(&gl, width, height) };
        let tex_curr = unsafe { create_rgba_texture_mip(&gl, width, height) };
        let tex_out = unsafe { create_rgba_texture(&gl, width, height) };

        // One MV texture per pyramid level (LINEAR for bilinear interpolation in warp)
        let mv_levels: Vec<_> = level_dims.iter().map(|&(lw, lh)| {
            unsafe { create_mv_texture(&gl, (lw + 15) / 16, (lh + 15) / 16) }
        }).collect();

        // Second MV texture at L0 for median filter ping-pong
        let mv0_w = (width + 15) / 16;
        let mv0_h = (height + 15) / 16;
        let tex_mv_filtered = unsafe { create_mv_texture(&gl, mv0_w, mv0_h) };
        let tex_mv_prev = unsafe { create_mv_texture(&gl, mv0_w, mv0_h) };

        let fbo_capture = unsafe {
            gl.create_framebuffer().expect("FBO for framegen")
        };

        // Cache uniform locations
        let flow_locs = unsafe { FlowUniforms {
            size: gl.get_uniform_location(prog_flow, "uSize"),
            radius: gl.get_uniform_location(prog_flow, "uRadius"),
            has_hint: gl.get_uniform_location(prog_flow, "uHasHint"),
            level: gl.get_uniform_location(prog_flow, "uLevel"),
        }};
        let synth_locs = unsafe { SynthUniforms {
            size: gl.get_uniform_location(prog_synth, "uSize"),
            t: gl.get_uniform_location(prog_synth, "uT"),
            mode: gl.get_uniform_location(prog_synth, "uMode"),
        }};
        let filter_loc_mv_size = unsafe {
            gl.get_uniform_location(prog_mv_filter, "uMVSize")
        };
        let filter_loc_has_prev = unsafe {
            gl.get_uniform_location(prog_mv_filter, "uHasPrev")
        };

        if debug {
            eprintln!("framegen: init {}x{}, {} pyramid levels", width, height, NUM_PYRAMID_LEVELS);
        }

        Some(Self {
            gl, mode: FrameGenMode::Off,
            quality: FrameGenQuality::Balanced,
            cap_w: width, cap_h: height,
            width, height,
            tex_prev, tex_curr, tex_out,
            mv_levels, tex_mv_filtered, tex_mv_prev, has_prev_mv: false, level_dims,
            fbo_capture, frame_count: 0, mv_valid: false,
            prog_flow, prog_synth, prog_mv_filter,
            flow_locs, synth_locs, filter_loc_mv_size, filter_loc_has_prev,
            stats: FrameGenStats::default(),
        })
    }

    // ── Public API ──────────────────────────────────────────────────

    #[allow(dead_code)]
    pub fn mode(&self) -> FrameGenMode { self.mode }

    pub fn set_mode(&mut self, mode: FrameGenMode) { self.mode = mode; }

    /// Current texture dimensions (matches last push_frame source).
    pub fn dimensions(&self) -> (u32, u32) { (self.width, self.height) }

    #[allow(dead_code)]
    pub fn quality(&self) -> FrameGenQuality { self.quality }

    pub fn set_quality(&mut self, q: FrameGenQuality) { self.quality = q; }

    pub fn stats(&self) -> &FrameGenStats { &self.stats }

    pub fn can_generate(&self) -> bool {
        self.mode != FrameGenMode::Off && self.frame_count >= 2
    }

    /// Resize internal textures if the framebuffer dimensions changed.
    /// Caps to capture resolution — no point running compute at higher res
    /// than the source content.  Resets frame counter.
    fn resize(&mut self, w: u32, h: u32) {
        let (w, h) = cap_resolution(w, h, self.cap_w, self.cap_h);
        if w == self.width && h == self.height { return; }

        // Delete old textures (keep programs and FBO)
        unsafe {
            self.gl.delete_texture(self.tex_prev);
            self.gl.delete_texture(self.tex_curr);
            self.gl.delete_texture(self.tex_out);
            for mv in &self.mv_levels {
                self.gl.delete_texture(*mv);
            }
            self.gl.delete_texture(self.tex_mv_filtered);
            self.gl.delete_texture(self.tex_mv_prev);
        }

        // Recompute level dimensions
        self.level_dims.clear();
        let (mut lw, mut lh) = (w, h);
        for _ in 0..NUM_PYRAMID_LEVELS {
            self.level_dims.push((lw, lh));
            lw = (lw + 1) / 2;
            lh = (lh + 1) / 2;
        }

        self.tex_prev = unsafe { create_rgba_texture_mip(&self.gl, w, h) };
        self.tex_curr = unsafe { create_rgba_texture_mip(&self.gl, w, h) };
        self.tex_out = unsafe { create_rgba_texture(&self.gl, w, h) };

        self.mv_levels = self.level_dims.iter().map(|&(dw, dh)| {
            unsafe { create_mv_texture(&self.gl, (dw + 15) / 16, (dh + 15) / 16) }
        }).collect();

        let mv0_w = (w + 15) / 16;
        let mv0_h = (h + 15) / 16;
        self.tex_mv_filtered = unsafe { create_mv_texture(&self.gl, mv0_w, mv0_h) };
        self.tex_mv_prev = unsafe { create_mv_texture(&self.gl, mv0_w, mv0_h) };
        self.has_prev_mv = false;

        self.width = w;
        self.height = h;
        self.frame_count = 0;
    }

    /// Capture the current default framebuffer as a new "real" frame.
    /// Call right after the YUV→RGB render, before OSD / present.
    pub fn push_frame(&mut self, src_w: u32, src_h: u32) {
        // If window dimensions changed, recreate textures to match.
        self.resize(src_w, src_h);

        std::mem::swap(&mut self.tex_prev, &mut self.tex_curr);

        unsafe {
            // Blit default framebuffer → tex_curr level 0 (1:1 — same size)
            self.gl.bind_framebuffer(glow::READ_FRAMEBUFFER, None);
            self.gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(self.fbo_capture));
            self.gl.framebuffer_texture_2d(
                glow::DRAW_FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                Some(self.tex_curr),
                0,
            );
            self.gl.blit_framebuffer(
                0, 0, src_w as i32, src_h as i32,
                0, 0, self.width as i32, self.height as i32,
                glow::COLOR_BUFFER_BIT,
                glow::LINEAR,
            );
            self.gl.bind_framebuffer(glow::FRAMEBUFFER, None);

            // Build mipmap pyramid for the new frame
            self.gl.bind_texture(glow::TEXTURE_2D, Some(self.tex_curr));
            self.gl.generate_mipmap(glow::TEXTURE_2D);
            self.gl.bind_texture(glow::TEXTURE_2D, None);
        }

        self.frame_count += 1;
        self.mv_valid = false; // Force MV recomputation on next generate()
    }

    /// Run frame synthesis (and motion estimation if needed).
    /// Motion vectors are computed once per frame pair and cached;
    /// subsequent calls with different `t` only run the cheap warp pass.
    ///
    /// `t` is the interpolation parameter (0.5 = midpoint between frames).
    pub fn generate(&mut self, t: f32) -> bool {
        if !self.can_generate() {
            self.stats.miss_count += 1;
            return false;
        }

        let start = std::time::Instant::now();

        // Only compute motion vectors once per frame pair
        if !self.mv_valid {
            self.compute_motion_vectors();
            self.mv_valid = true;
        }

        // Warp synthesis (cheap — only pass that depends on t)
        self.run_synthesis(t);

        self.stats.last_gen_us = start.elapsed().as_micros() as u64;
        self.stats.synth_count += 1;
        true
    }

    /// Hierarchical block matching + MV filtering. Expensive — run once per frame pair.
    fn compute_motion_vectors(&mut self) {
        let active_levels = self.quality.levels().min(NUM_PYRAMID_LEVELS);
        let radii = self.quality.radii();

        unsafe {
            // ── Pass 1: hierarchical block matching (coarse → fine) ──
            self.gl.use_program(Some(self.prog_flow));

            self.gl.active_texture(glow::TEXTURE0);
            self.gl.bind_texture(glow::TEXTURE_2D, Some(self.tex_prev));
            self.gl.active_texture(glow::TEXTURE1);
            self.gl.bind_texture(glow::TEXTURE_2D, Some(self.tex_curr));

            for level in (0..active_levels).rev() {
                let (lw, lh) = self.level_dims[level];
                let mv_w = (lw + 15) / 16;
                let mv_h = (lh + 15) / 16;

                if let Some(ref loc) = self.flow_locs.size {
                    self.gl.uniform_2_i32(Some(loc), lw as i32, lh as i32);
                }
                if let Some(ref loc) = self.flow_locs.radius {
                    self.gl.uniform_1_i32(Some(loc), radii[level]);
                }
                if let Some(ref loc) = self.flow_locs.level {
                    self.gl.uniform_1_i32(Some(loc), level as i32);
                }

                if level < active_levels - 1 {
                    self.gl.active_texture(glow::TEXTURE2);
                    self.gl.bind_texture(glow::TEXTURE_2D, Some(self.mv_levels[level + 1]));
                    if let Some(ref loc) = self.flow_locs.has_hint {
                        self.gl.uniform_1_i32(Some(loc), 1);
                    }
                } else {
                    if let Some(ref loc) = self.flow_locs.has_hint {
                        self.gl.uniform_1_i32(Some(loc), 0);
                    }
                }

                self.gl.bind_image_texture(
                    0, self.mv_levels[level], 0, false, 0,
                    glow::WRITE_ONLY, glow::RGBA16F,
                );

                self.gl.dispatch_compute(mv_w, mv_h, 1);
                self.gl.memory_barrier(
                    glow::SHADER_IMAGE_ACCESS_BARRIER_BIT
                    | glow::TEXTURE_FETCH_BARRIER_BIT,
                );
            }

            // ── Pass 1.5: median filter on finest-level MV ─────────
            {
                let mv0_w = (self.width + 15) / 16;
                let mv0_h = (self.height + 15) / 16;

                self.gl.use_program(Some(self.prog_mv_filter));

                self.gl.active_texture(glow::TEXTURE0);
                self.gl.bind_texture(glow::TEXTURE_2D, Some(self.mv_levels[0]));

                self.gl.active_texture(glow::TEXTURE1);
                self.gl.bind_texture(glow::TEXTURE_2D, Some(self.tex_mv_prev));

                self.gl.bind_image_texture(
                    0, self.tex_mv_filtered, 0, false, 0,
                    glow::WRITE_ONLY, glow::RGBA16F,
                );

                if let Some(ref loc) = self.filter_loc_mv_size {
                    self.gl.uniform_2_i32(Some(loc), mv0_w as i32, mv0_h as i32);
                }
                if let Some(ref loc) = self.filter_loc_has_prev {
                    self.gl.uniform_1_i32(Some(loc), if self.has_prev_mv { 1 } else { 0 });
                }

                let fg_x = (mv0_w + 7) / 8;
                let fg_y = (mv0_h + 7) / 8;
                self.gl.dispatch_compute(fg_x, fg_y, 1);
                self.gl.memory_barrier(
                    glow::SHADER_IMAGE_ACCESS_BARRIER_BIT
                    | glow::TEXTURE_FETCH_BARRIER_BIT,
                );
            }

            // Swap filtered MVs → previous for temporal dampening
            std::mem::swap(&mut self.tex_mv_filtered, &mut self.tex_mv_prev);
            self.has_prev_mv = true;

            self.gl.use_program(None);
        }
    }

    /// Warp synthesis only — uses cached MVs to produce output at parameter `t`.
    fn run_synthesis(&mut self, t: f32) {
        unsafe {
            self.gl.use_program(Some(self.prog_synth));

            self.gl.active_texture(glow::TEXTURE0);
            self.gl.bind_texture(glow::TEXTURE_2D, Some(self.tex_prev));
            self.gl.active_texture(glow::TEXTURE1);
            self.gl.bind_texture(glow::TEXTURE_2D, Some(self.tex_curr));
            self.gl.active_texture(glow::TEXTURE2);
            // After MV swap, tex_mv_prev holds the filtered MVs
            self.gl.bind_texture(glow::TEXTURE_2D, Some(self.tex_mv_prev));

            self.gl.bind_image_texture(
                0, self.tex_out, 0, false, 0,
                glow::WRITE_ONLY, glow::RGBA8,
            );

            if let Some(ref loc) = self.synth_locs.size {
                self.gl.uniform_2_i32(Some(loc), self.width as i32, self.height as i32);
            }
            if let Some(ref loc) = self.synth_locs.t {
                self.gl.uniform_1_f32(Some(loc), t);
            }
            if let Some(ref loc) = self.synth_locs.mode {
                let mode_int = match self.mode {
                    FrameGenMode::Extrapolate => 0,
                    FrameGenMode::Interpolate => 1,
                    _ => return, // Off or Rife (uses separate pipeline)
                };
                self.gl.uniform_1_i32(Some(loc), mode_int);
            }

            let groups_x = (self.width + 7) / 8;
            let groups_y = (self.height + 7) / 8;
            self.gl.dispatch_compute(groups_x, groups_y, 1);
            self.gl.memory_barrier(glow::SHADER_IMAGE_ACCESS_BARRIER_BIT);
            // Flush to start GPU execution immediately; without this the
            // driver may batch the compute dispatch and only execute it
            // during the blocking eglSwapBuffers call.
            self.gl.flush();

            self.gl.use_program(None);
        }
    }

    pub fn output_texture(&self) -> glow::Texture { self.tex_out }

    /// Return the previous and current frame textures for external use (e.g. RIFE).
    #[allow(dead_code)]
    pub fn prev_curr_textures(&self) -> (glow::Texture, glow::Texture) {
        (self.tex_prev, self.tex_curr)
    }
}

impl Drop for FrameGen {
    fn drop(&mut self) {
        unsafe {
            self.gl.delete_texture(self.tex_prev);
            self.gl.delete_texture(self.tex_curr);
            self.gl.delete_texture(self.tex_out);
            for mv in &self.mv_levels {
                self.gl.delete_texture(*mv);
            }
            self.gl.delete_texture(self.tex_mv_filtered);
            self.gl.delete_texture(self.tex_mv_prev);
            self.gl.delete_framebuffer(self.fbo_capture);
            self.gl.delete_program(self.prog_flow);
            self.gl.delete_program(self.prog_synth);
            self.gl.delete_program(self.prog_mv_filter);
        }
    }
}

// ── Helpers ─────────────────────────────────────────────────────────

/// Scale dimensions down to fit within max_w × max_h while preserving
/// aspect ratio.  Used to cap framegen at capture resolution.
fn cap_resolution(w: u32, h: u32, max_w: u32, max_h: u32) -> (u32, u32) {
    if w <= max_w && h <= max_h {
        return (w, h);
    }
    let scale_w = max_w as f32 / w as f32;
    let scale_h = max_h as f32 / h as f32;
    let scale = scale_w.min(scale_h);
    // Round down to even for mipmap compatibility
    let nw = ((w as f32 * scale) as u32) & !1;
    let nh = ((h as f32 * scale) as u32) & !1;
    (nw.max(2), nh.max(2))
}

fn check_compute_support(gl: &glow::Context) -> bool {
    let version = unsafe { gl.get_parameter_string(glow::VERSION) };
    let parts: Vec<&str> = version.split(|c: char| !c.is_ascii_digit())
        .filter(|s| !s.is_empty())
        .collect();
    if parts.len() >= 2 {
        if let (Ok(major), Ok(minor)) = (parts[0].parse::<u32>(), parts[1].parse::<u32>()) {
            return major > 4 || (major == 4 && minor >= 3);
        }
    }
    false
}

/// RGBA8 texture with mipmap storage for pyramid (NUM_PYRAMID_LEVELS levels).
unsafe fn create_rgba_texture_mip(gl: &glow::Context, w: u32, h: u32) -> glow::Texture {
    let tex = gl.create_texture().expect("framegen RGBA mip texture");
    gl.bind_texture(glow::TEXTURE_2D, Some(tex));
    gl.tex_storage_2d(
        glow::TEXTURE_2D,
        NUM_PYRAMID_LEVELS as i32,
        glow::RGBA8,
        w as i32, h as i32,
    );
    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR_MIPMAP_LINEAR as i32);
    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
    gl.bind_texture(glow::TEXTURE_2D, None);
    tex
}

/// RGBA8 texture without mipmaps (for synthesis output).
unsafe fn create_rgba_texture(gl: &glow::Context, w: u32, h: u32) -> glow::Texture {
    let tex = gl.create_texture().expect("framegen RGBA texture");
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
    gl.bind_texture(glow::TEXTURE_2D, None);
    tex
}

unsafe fn create_mv_texture(gl: &glow::Context, w: u32, h: u32) -> glow::Texture {
    let tex = gl.create_texture().expect("framegen MV texture");
    gl.bind_texture(glow::TEXTURE_2D, Some(tex));
    gl.tex_image_2d(
        glow::TEXTURE_2D, 0, glow::RGBA16F as i32,
        w as i32, h as i32, 0,
        glow::RGBA, glow::FLOAT, None,
    );
    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR as i32);
    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
    gl.bind_texture(glow::TEXTURE_2D, None);
    tex
}

fn compile_compute(gl: &glow::Context, src: &str) -> anyhow::Result<glow::Program> {
    unsafe {
        let prog = gl.create_program().map_err(|e| anyhow::anyhow!("{}", e))?;
        let cs = gl.create_shader(glow::COMPUTE_SHADER)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        gl.shader_source(cs, src);
        gl.compile_shader(cs);
        if !gl.get_shader_compile_status(cs) {
            let log = gl.get_shader_info_log(cs);
            gl.delete_shader(cs);
            gl.delete_program(prog);
            anyhow::bail!("compute shader: {}", log);
        }
        gl.attach_shader(prog, cs);
        gl.link_program(prog);
        gl.delete_shader(cs);
        if !gl.get_program_link_status(prog) {
            let log = gl.get_program_info_log(prog);
            gl.delete_program(prog);
            anyhow::bail!("compute link: {}", log);
        }
        Ok(prog)
    }
}
