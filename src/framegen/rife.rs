//! RIFE (Real-Time Intermediate Flow Estimation) neural frame interpolation.
//!
//! Uses ONNX Runtime to run a pre-trained RIFE model for high-quality
//! frame interpolation.  Requires a RIFE ONNX model file.
//!
//! Enable with `--features rife` at build time.
//!
//! # Model source
//!
//! Use models from the official Practical-RIFE repository by Zheqiang Huang
//! et al. (Megvii Research / ECNU):
//!
//!   Repository: github.com/hzwer/Practical-RIFE
//!   Paper:      "Real-Time Intermediate Flow Estimation for Video Frame
//!               Interpolation" (ECCV 2022)
//!
//! Export to ONNX using the repo's export script, or use a pre-exported
//! model from the Releases page.  Recommended: RIFE v4.6+ (flownet).
//!
//! Place as:  ~/.config/capview/rife.onnx
//!    or set: CAPVIEW_RIFE_MODEL=/path/to/model.onnx
//!
//! # Supported input formats (auto-detected)
//!
//! - Two inputs `[1, 3, H, W]` each (img0, img1) — standard export
//! - Single input `[1, 6, H, W]` (concatenated prev+curr)
//! - Optional `timestep` input `[1, 1, H, W]` (if absent, always t=0.5)

use anyhow::{bail, Context, Result};
use glow::HasContext;
use ndarray::Array4;
use ort::session::Session;
use ort::value::Tensor;

/// RIFE neural frame interpolator.
pub struct RifeInterpolator {
    session: Session,
    /// Scratch buffers for frame readback (RGBA u8, H×W×4).
    buf_prev: Vec<u8>,
    buf_curr: Vec<u8>,
    /// Scratch buffer for output upload.
    buf_out: Vec<u8>,
    /// Pre-allocated NCHW input arrays (avoids per-frame allocation).
    arr_prev: Array4<f32>,
    arr_curr: Array4<f32>,
    width: u32,
    height: u32,
    pad_w: u32,
    pad_h: u32,
    /// GL texture for the interpolated output.
    tex_out: glow::Texture,
    /// Persistent FBO for texture readback (avoids create/delete per frame).
    fbo: glow::Framebuffer,
    gl: glow::Context,
    diag_done: bool,
    /// Model input format.
    concat_input: bool,
    has_timestep: bool,
    input_names: Vec<String>,
}

impl RifeInterpolator {
    /// Create a new RIFE interpolator from an ONNX model file.
    ///
    /// The model should accept inputs named "img0" and "img1" (both
    /// `[1, 3, H, W]` float32, range [0,1]) and produce output named
    /// "output" (`[1, 3, H, W]` float32).
    pub fn new<F: FnMut(&str) -> *const std::ffi::c_void>(
        mut gl_get_proc: F,
        model_path: &std::path::Path,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        if !model_path.exists() {
            bail!(
                "RIFE model not found at {}.\n\
                 Download from the official Practical-RIFE repository\n\
                 (github.com/hzwer/Practical-RIFE) and place as\n\
                 ~/.config/capview/rife.onnx",
                model_path.display()
            );
        }

        let gl = unsafe {
            glow::Context::from_loader_function(|s| gl_get_proc(s))
        };
        let pad_h = ((height + 31) / 32) * 32;
        let pad_w = ((width + 31) / 32) * 32;

        eprintln!("rife: loading model from {:?} (input {}x{}, padded to {}x{})",
            model_path, width, height, pad_w, pad_h);

        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("ort session builder: {}", e))?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("ort optimization: {}", e))?
            .with_intra_threads(4)
            .map_err(|e| anyhow::anyhow!("ort threads: {}", e))?
            .commit_from_file(&model_path)
            .map_err(|e| anyhow::anyhow!("failed to load RIFE model: {}", e))?;

        // Detect model input format from metadata
        let inputs = session.inputs();
        if inputs.is_empty() {
            bail!("RIFE model has no inputs");
        }
        let input_names: Vec<String> = inputs.iter().map(|i| i.name().to_string()).collect();

        // Check first input's channel dimension to detect concat vs two-input format
        let first_shape = inputs[0].dtype().tensor_shape();
        let chan_dim = first_shape.as_ref().and_then(|s| {
            let dims: Vec<i64> = s.iter().copied().collect();
            dims.get(1).copied()
        });
        let concat_input = inputs.len() == 1 || chan_dim == Some(6);
        let has_timestep = input_names.iter()
            .any(|n| { let l = n.to_lowercase(); l.contains("timestep") || l.contains("time") });

        for input in inputs.iter() {
            eprintln!("rife:   input: {:?} {:?}", input.name(), input.dtype());
        }
        for output in session.outputs().iter() {
            eprintln!("rife:   output: {:?} {:?}", output.name(), output.dtype());
        }
        eprintln!("rife: format={}, timestep={}", if concat_input { "concat[1,6,H,W]" } else { "two[1,3,H,W]" }, has_timestep);

        let npix = (width * height) as usize;
        let pw = pad_w as usize;
        let ph = pad_h as usize;

        // Persistent FBO for readback
        let fbo = unsafe { gl.create_framebuffer().map_err(|e| anyhow::anyhow!("{}", e))? };

        // Create output texture
        let tex_out = unsafe {
            let tex = gl.create_texture().map_err(|e| anyhow::anyhow!("{}", e))?;
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
            gl.bind_texture(glow::TEXTURE_2D, None);
            tex
        };

        eprintln!("rife: model loaded, session ready");

        Ok(Self {
            session,
            buf_prev: vec![0u8; npix * 4],
            buf_curr: vec![0u8; npix * 4],
            buf_out: vec![0u8; npix * 4],
            arr_prev: Array4::zeros((1, 3, ph, pw)),
            arr_curr: Array4::zeros((1, 3, ph, pw)),
            width,
            height,
            pad_w,
            pad_h,
            tex_out,
            fbo,
            gl,
            diag_done: false,
            concat_input,
            has_timestep,
            input_names,
        })
    }

    /// Read an RGBA GL texture into a CPU buffer using persistent FBO.
    fn read_texture(
        gl: &glow::Context,
        fbo: glow::Framebuffer,
        tex: glow::Texture,
        w: u32,
        h: u32,
        buf: &mut [u8],
    ) {
        unsafe {
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
            gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER, glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D, Some(tex), 0,
            );
            gl.read_pixels(
                0, 0, w as i32, h as i32,
                glow::RGBA, glow::UNSIGNED_BYTE,
                glow::PixelPackData::Slice(buf),
            );
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        }
    }

    /// Convert RGBA u8 buffer [H×W×4] to NCHW float32 [1,3,pad_H,pad_W].
    /// Padding region uses edge replication (not zeros) to reduce border artifacts.
    fn rgba_to_nchw(buf: &[u8], w: u32, h: u32, arr: &mut Array4<f32>) {
        let w = w as usize;
        let h = h as usize;
        let ph = arr.shape()[2];
        let pw = arr.shape()[3];
        for y in 0..ph {
            let sy = y.min(h - 1); // edge replicate
            for x in 0..pw {
                let sx = x.min(w - 1);
                let idx = (sy * w + sx) * 4;
                arr[[0, 0, y, x]] = buf[idx] as f32 / 255.0;
                arr[[0, 1, y, x]] = buf[idx + 1] as f32 / 255.0;
                arr[[0, 2, y, x]] = buf[idx + 2] as f32 / 255.0;
            }
        }
    }

    /// Run RIFE interpolation between two GL textures.
    /// Returns true if interpolation succeeded and `output_texture()` is valid.
    pub fn interpolate(&mut self, tex_prev: glow::Texture, tex_curr: glow::Texture) -> bool {
        let start = std::time::Instant::now();

        // Step 1: Read textures from GPU → CPU
        Self::read_texture(&self.gl, self.fbo, tex_prev, self.width, self.height, &mut self.buf_prev);
        Self::read_texture(&self.gl, self.fbo, tex_curr, self.width, self.height, &mut self.buf_curr);
        let readback_us = start.elapsed().as_micros();

        // Step 2: Convert to NCHW (reuses pre-allocated arrays, edge-replicates padding)
        Self::rgba_to_nchw(&self.buf_prev, self.width, self.height, &mut self.arr_prev);
        Self::rgba_to_nchw(&self.buf_curr, self.width, self.height, &mut self.arr_curr);

        // Step 3: Run inference
        let infer_start = std::time::Instant::now();
        let ph = self.pad_h as usize;
        let pw = self.pad_w as usize;

        // Build input tensors (outside closure to avoid lifetime issues)
        let mk = |e: ort::Error| anyhow::anyhow!("{}", e);
        let t_prev = match Tensor::from_array(self.arr_prev.clone()) { Ok(t) => t, Err(e) => { eprintln!("rife: tensor: {}", e); return false; } };
        let t_curr = match Tensor::from_array(self.arr_curr.clone()) { Ok(t) => t, Err(e) => { eprintln!("rife: tensor: {}", e); return false; } };

        let outputs = if self.concat_input {
            let mut concat = Array4::<f32>::zeros((1, 6, ph, pw));
            concat.slice_mut(ndarray::s![.., 0..3, .., ..]).assign(&self.arr_prev);
            concat.slice_mut(ndarray::s![.., 3..6, .., ..]).assign(&self.arr_curr);
            let t_concat = match Tensor::from_array(concat) { Ok(t) => t, Err(e) => { eprintln!("rife: tensor: {}", e); return false; } };
            self.session.run(ort::inputs![t_concat]).map_err(mk)
        } else {
            self.session.run(ort::inputs![t_prev, t_curr]).map_err(mk)
        };

        let outputs = match outputs {
            Ok(o) => o,
            Err(e) => {
                if !self.diag_done { eprintln!("rife: inference error: {}", e); self.diag_done = true; }
                return false;
            }
        };
        let infer_us = infer_start.elapsed().as_micros();

        // Step 4: Extract output tensor → [1, 3, H, W]
        let (_shape, out_slice) = match outputs[0].try_extract_tensor::<f32>() {
            Ok(t) => t,
            Err(e) => {
                if !self.diag_done { eprintln!("rife: output extraction error: {}", e); self.diag_done = true; }
                return false;
            }
        };
        let plane = pw * ph;
        let w = self.width as usize;
        let h = self.height as usize;
        if out_slice.len() >= 3 * plane {
            for row in 0..h {
                for col in 0..w {
                    let si = row * pw + col;
                    let di = (row * w + col) * 4;
                    self.buf_out[di]     = (out_slice[si].clamp(0.0, 1.0) * 255.0) as u8;
                    self.buf_out[di + 1] = (out_slice[plane + si].clamp(0.0, 1.0) * 255.0) as u8;
                    self.buf_out[di + 2] = (out_slice[2 * plane + si].clamp(0.0, 1.0) * 255.0) as u8;
                    self.buf_out[di + 3] = 255;
                }
            }
        } else {
            if !self.diag_done {
                eprintln!("rife: output too small ({} floats, need {})", out_slice.len(), 3 * plane);
                self.diag_done = true;
            }
            return false;
        }

        unsafe {
            self.gl.bind_texture(glow::TEXTURE_2D, Some(self.tex_out));
            self.gl.tex_sub_image_2d(
                glow::TEXTURE_2D, 0,
                0, 0, self.width as i32, self.height as i32,
                glow::RGBA, glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(&self.buf_out),
            );
            self.gl.bind_texture(glow::TEXTURE_2D, None);
        }

        let total_us = start.elapsed().as_micros();

        if !self.diag_done {
            self.diag_done = true;
            eprintln!("rife: first frame {}x{} — readback {}µs, inference {}µs, total {}µs ({:.1}ms)",
                self.width, self.height, readback_us, infer_us, total_us,
                total_us as f64 / 1000.0);
        }

        true
    }

    /// Get the output texture containing the interpolated frame.
    pub fn output_texture(&self) -> glow::Texture {
        self.tex_out
    }

    /// Get the frame dimensions.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

impl Drop for RifeInterpolator {
    fn drop(&mut self) {
        unsafe {
            self.gl.delete_texture(self.tex_out);
            self.gl.delete_framebuffer(self.fbo);
        }
    }
}
