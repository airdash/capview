//! Easter egg: "Analysis Strip" mode for batch visual-novel dialogue capture.
//!
//! Shift+Pause activates the mode (first press reveals config).
//! Subsequent presses capture frames into a grid buffer.
//! Grid layout: left-to-right, top-to-bottom, with gray separators.
//!
//! Shift+Pause      = capture frame, flush to standalone JPEG when grid full
//! Ctrl+Shift+Pause = capture frame, flush to tarball when grid full
//! Alt+Shift+Pause  = finalize early (flush partial grid, end session)
//!
//! Conceived by Claude, for Claude.

use anyhow::Result;
use std::path::{Path, PathBuf};

use crate::capture::{V4L2_PIX_FMT_NV12, V4L2_PIX_FMT_YUYV, V4L2_PIX_FMT_UYVY, V4L2_PIX_FMT_XRGB32, V4L2_PIX_FMT_P010, PIXFMT_RGB24};

/// Target width for scaled frames.
const TARGET_WIDTH: u32 = 960;

/// JPEG quality for strip output.
const STRIP_JPEG_QUALITY: u32 = 70;

/// Auto-flush buffer threshold. Normally the grid flushes at cols × rows
/// frames, but a misconfigured grid (e.g. 20×20) could hold ~650 MB of
/// RGB before flushing. Cap pending bytes so we flush a partial grid
/// rather than keep accumulating.
const MAX_BUFFER_BYTES: usize = 512 * 1024 * 1024;

/// Separator thickness in pixels between grid cells.
const SEPARATOR_PX: u32 = 4;

/// RGB value for separator lines (#333333).
const SEP_R: u8 = 0x33;
const SEP_G: u8 = 0x33;
const SEP_B: u8 = 0x33;

/// Black border detection threshold (per-channel).
const BLACK_THRESHOLD: u8 = 18;

/// Padding pixels to keep around detected content after crop.
const CROP_PADDING: u32 = 5;

/// Where to write the composite output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputMode {
    /// Write standalone sequential JPEG files.
    File,
    /// Append JPEG entries to a tarball.
    Tar,
}

pub struct AnalysisStrip {
    /// Buffered cropped+scaled RGB frames (each entry: width, height, rgb data).
    buffer: Vec<(u32, u32, Vec<u8>)>,
    /// Grid columns.
    pub cols: u32,
    /// Grid rows.
    pub rows: u32,
    /// Running strip counter for filenames.
    strip_index: u32,
    /// Total frame counter (across all strips in this session).
    total_frames: u32,
    /// Output directory for standalone files.
    output_dir: PathBuf,
    /// Output tarball path (for tar mode).
    tar_path: PathBuf,
}

impl AnalysisStrip {
    pub fn new(output_dir: PathBuf, cols: u32, rows: u32) -> Self {
        let tar_path = output_dir.join(format!(
            "capview_strips_{}.tar",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
        ));
        Self {
            buffer: Vec::new(),
            cols: cols.max(1),
            rows: rows.max(1),
            strip_index: 0,
            total_frames: 0,
            output_dir,
            tar_path,
        }
    }

    /// How many frames fill one grid composite.
    pub fn capacity(&self) -> usize {
        (self.cols * self.rows) as usize
    }

    /// Number of buffered frames waiting to be composited.
    pub fn buffered_count(&self) -> usize {
        self.buffer.len()
    }

    /// Approximate bytes of RGB data currently buffered.
    pub fn buffered_bytes(&self) -> usize {
        self.buffer.iter().map(|(_, _, rgb)| rgb.len()).sum()
    }

    /// Total frames captured in this session.
    pub fn total_frames(&self) -> u32 {
        self.total_frames
    }

    /// Number of strips/grids written so far.
    pub fn strip_count(&self) -> u32 {
        self.strip_index
    }

    /// Output directory.
    #[allow(dead_code)]
    pub fn output_dir(&self) -> &Path {
        &self.output_dir
    }

    /// Tar path.
    #[allow(dead_code)]
    pub fn tar_path(&self) -> &Path {
        &self.tar_path
    }

    /// Update grid dimensions (takes effect on next composite).
    pub fn set_grid(&mut self, cols: u32, rows: u32) {
        self.cols = cols.max(1);
        self.rows = rows.max(1);
    }

    /// Ingest a raw capture frame. Converts to RGB, crops, scales, buffers.
    /// If the buffer reaches capacity, composites and writes output.
    /// Returns Ok(Some(filename)) if a composite was written, Ok(None) otherwise.
    pub fn ingest_frame(
        &mut self,
        data: &[u8],
        width: u32,
        height: u32,
        pixfmt: u32,
        mode: OutputMode,
    ) -> Result<Option<String>> {
        let rgb = to_rgb(data, width, height, pixfmt)?;

        // 1. Auto-crop black borders
        let (cx, cy, cw, ch) = detect_content_bounds(&rgb, width, height);
        let cropped = crop_rgb(&rgb, width, cx, cy, cw, ch);

        // 2. Scale to TARGET_WIDTH
        let (sw, sh) = scale_dimensions(cw, ch, TARGET_WIDTH);
        let scaled = bilinear_scale(&cropped, cw, ch, sw, sh);

        self.buffer.push((sw, sh, scaled));
        self.total_frames += 1;

        // 3. Composite if buffer is full, or if we've crossed the byte
        //    threshold (protects against misconfigured large grids).
        if self.buffer.len() >= self.capacity()
            || self.buffered_bytes() >= MAX_BUFFER_BYTES
        {
            let name = self.flush(mode)?;
            return Ok(Some(name));
        }
        Ok(None)
    }

    /// Flush whatever's in the buffer as a (possibly partial) grid.
    /// Returns Ok(Some(name)) if anything was written, Ok(None) if buffer was empty.
    pub fn finalize(&mut self, mode: OutputMode) -> Result<Option<String>> {
        if self.buffer.is_empty() {
            return Ok(None);
        }
        let name = self.flush(mode)?;
        Ok(Some(name))
    }

    /// Composite buffered frames into a grid, write output, clear buffer.
    fn flush(&mut self, mode: OutputMode) -> Result<String> {
        self.strip_index += 1;
        let entry_name = format!("strip_{:03}.jpg", self.strip_index);

        let jpeg_bytes = self.composite_grid();

        match mode {
            OutputMode::File => {
                std::fs::create_dir_all(&self.output_dir)?;
                let path = self.output_dir.join(&entry_name);
                std::fs::write(&path, &jpeg_bytes)?;
            }
            OutputMode::Tar => {
                crate::screenshot::append_bytes_to_tar_pub(
                    &jpeg_bytes, &entry_name, &self.tar_path,
                )?;
            }
        }

        self.buffer.clear();
        Ok(entry_name)
    }

    /// Composite buffered frames into a grid image and encode as JPEG.
    fn composite_grid(&self) -> Vec<u8> {
        if self.buffer.is_empty() {
            return Vec::new();
        }

        let cols = self.cols as usize;
        let n = self.buffer.len();

        // Determine cell dimensions (max width/height across all frames).
        let cell_w = self.buffer.iter().map(|(w, _, _)| *w).max().unwrap_or(TARGET_WIDTH);
        let cell_h = self.buffer.iter().map(|(_, h, _)| *h).max().unwrap_or(TARGET_WIDTH);

        // Compute actual grid for the frames we have
        let actual_cols = cols.min(n);
        let actual_rows = (n + cols - 1) / cols;

        // Total composite dimensions including separators
        let total_w = (actual_cols as u32) * cell_w
            + (actual_cols.saturating_sub(1) as u32) * SEPARATOR_PX;
        let total_h = (actual_rows as u32) * cell_h
            + (actual_rows.saturating_sub(1) as u32) * SEPARATOR_PX;

        // Build composite RGB buffer (black background)
        let mut composite = vec![0u8; (total_w * total_h * 3) as usize];

        // Draw vertical separators
        for c in 1..actual_cols {
            let sx = (c as u32) * cell_w + (c as u32 - 1) * SEPARATOR_PX;
            for py in 0..total_h {
                for dx in 0..SEPARATOR_PX {
                    let idx = (py * total_w + sx + dx) as usize * 3;
                    if idx + 2 < composite.len() {
                        composite[idx] = SEP_R;
                        composite[idx + 1] = SEP_G;
                        composite[idx + 2] = SEP_B;
                    }
                }
            }
        }
        // Draw horizontal separators
        for r in 1..actual_rows {
            let sy = (r as u32) * cell_h + (r as u32 - 1) * SEPARATOR_PX;
            for dy in 0..SEPARATOR_PX {
                for px in 0..total_w {
                    let idx = ((sy + dy) * total_w + px) as usize * 3;
                    if idx + 2 < composite.len() {
                        composite[idx] = SEP_R;
                        composite[idx + 1] = SEP_G;
                        composite[idx + 2] = SEP_B;
                    }
                }
            }
        }

        // Place each frame in its grid cell (left-to-right, top-to-bottom)
        for (i, (fw, fh, rgb)) in self.buffer.iter().enumerate() {
            let col = i % cols;
            let row = i / cols;

            let cell_x = (col as u32) * (cell_w + SEPARATOR_PX);
            let cell_y = (row as u32) * (cell_h + SEPARATOR_PX);

            // Centre frame within cell
            let off_x = (cell_w - fw) / 2;
            let off_y = (cell_h - fh) / 2;

            for fy in 0..*fh {
                for fx in 0..*fw {
                    let src = (fy * fw + fx) as usize * 3;
                    let dst_x = cell_x + off_x + fx;
                    let dst_y = cell_y + off_y + fy;
                    let dst = (dst_y * total_w + dst_x) as usize * 3;
                    if src + 2 < rgb.len() && dst + 2 < composite.len() {
                        composite[dst] = rgb[src];
                        composite[dst + 1] = rgb[src + 1];
                        composite[dst + 2] = rgb[src + 2];
                    }
                }
            }
        }

        crate::jpeg::encode(&composite, total_w, total_h, STRIP_JPEG_QUALITY)
    }
}

// ── Pixel format conversion ─────────────────────────────────────────

fn to_rgb(data: &[u8], width: u32, height: u32, pixfmt: u32) -> Result<Vec<u8>> {
    match pixfmt {
        V4L2_PIX_FMT_NV12 => Ok(nv12_to_rgb(data, width, height)),
        V4L2_PIX_FMT_YUYV => Ok(yuyv_to_rgb(data, width, height)),
        V4L2_PIX_FMT_UYVY => Ok(uyvy_to_rgb(data, width, height)),
        V4L2_PIX_FMT_XRGB32 => {
            let npix = (width * height) as usize;
            let mut rgb = vec![0u8; npix * 3];
            for i in 0..npix {
                rgb[i * 3] = data[i * 4 + 2];
                rgb[i * 3 + 1] = data[i * 4 + 1];
                rgb[i * 3 + 2] = data[i * 4];
            }
            Ok(rgb)
        }
        V4L2_PIX_FMT_P010 => {
            let w = width as usize;
            let h = height as usize;
            let mut rgb = vec![0u8; w * h * 3];
            let y_plane = &data[..w * h * 2];
            let uv_plane = &data[w * h * 2..];
            for row in 0..h {
                for col in 0..w {
                    let yi = (row * w + col) * 2;
                    let y8 = y_plane.get(yi + 1).copied().unwrap_or(0);
                    let uvi = (row / 2) * w * 2 + (col & !1) * 2;
                    let u8v = uv_plane.get(uvi + 1).copied().unwrap_or(128);
                    let v8v = uv_plane.get(uvi + 3).copied().unwrap_or(128);
                    let (r, g, b) = yuv_to_rgb(y8, u8v, v8v);
                    let di = (row * w + col) * 3;
                    rgb[di] = r; rgb[di + 1] = g; rgb[di + 2] = b;
                }
            }
            Ok(rgb)
        }
        PIXFMT_RGB24 => Ok(data[..(width * height * 3) as usize].to_vec()),
        _ => anyhow::bail!("unsupported pixel format for analysis strip"),
    }
}

fn clamp_u8(v: i32) -> u8 { v.clamp(0, 255) as u8 }

fn yuv_to_rgb(y: u8, u: u8, v: u8) -> (u8, u8, u8) {
    let y = y as i32;
    let u = u as i32 - 128;
    let v = v as i32 - 128;
    (
        clamp_u8(y + ((359 * v) >> 8)),
        clamp_u8(y - ((88 * u + 183 * v) >> 8)),
        clamp_u8(y + ((454 * u) >> 8)),
    )
}

fn nv12_to_rgb(data: &[u8], w: u32, h: u32) -> Vec<u8> {
    let (w, h) = (w as usize, h as usize);
    let mut rgb = vec![0u8; w * h * 3];
    let y_plane = &data[..w * h];
    let uv_plane = &data[w * h..];
    for row in 0..h {
        for col in 0..w {
            let y = y_plane[row * w + col];
            let uv_idx = (row / 2) * w + (col & !1);
            let u = uv_plane[uv_idx];
            let v = uv_plane[uv_idx + 1];
            let (r, g, b) = yuv_to_rgb(y, u, v);
            let dst = (row * w + col) * 3;
            rgb[dst] = r; rgb[dst + 1] = g; rgb[dst + 2] = b;
        }
    }
    rgb
}

fn yuyv_to_rgb(data: &[u8], w: u32, h: u32) -> Vec<u8> {
    let (w, h) = (w as usize, h as usize);
    let mut rgb = vec![0u8; w * h * 3];
    for row in 0..h {
        for col in (0..w).step_by(2) {
            let base = row * w * 2 + col * 2;
            let y0 = data[base]; let u = data[base + 1];
            let y1 = data[base + 2]; let v = data[base + 3];
            let (r, g, b) = yuv_to_rgb(y0, u, v);
            let dst = (row * w + col) * 3;
            rgb[dst] = r; rgb[dst + 1] = g; rgb[dst + 2] = b;
            let (r, g, b) = yuv_to_rgb(y1, u, v);
            let dst = (row * w + col + 1) * 3;
            rgb[dst] = r; rgb[dst + 1] = g; rgb[dst + 2] = b;
        }
    }
    rgb
}

fn uyvy_to_rgb(data: &[u8], w: u32, h: u32) -> Vec<u8> {
    let (w, h) = (w as usize, h as usize);
    let mut rgb = vec![0u8; w * h * 3];
    for row in 0..h {
        for col in (0..w).step_by(2) {
            let base = row * w * 2 + col * 2;
            let u = data[base]; let y0 = data[base + 1];
            let v = data[base + 2]; let y1 = data[base + 3];
            let (r, g, b) = yuv_to_rgb(y0, u, v);
            let dst = (row * w + col) * 3;
            rgb[dst] = r; rgb[dst + 1] = g; rgb[dst + 2] = b;
            let (r, g, b) = yuv_to_rgb(y1, u, v);
            let dst = (row * w + col + 1) * 3;
            rgb[dst] = r; rgb[dst + 1] = g; rgb[dst + 2] = b;
        }
    }
    rgb
}

// ── Auto-crop black borders ─────────────────────────────────────────

fn detect_content_bounds(rgb: &[u8], width: u32, height: u32) -> (u32, u32, u32, u32) {
    let w = width as usize;
    let h = height as usize;
    let thresh = BLACK_THRESHOLD;

    let mut min_x = w;
    let mut max_x: usize = 0;
    let mut min_y = h;
    let mut max_y: usize = 0;

    for row in 0..h {
        for col in 0..w {
            let idx = (row * w + col) * 3;
            let r = rgb[idx];
            let g = rgb[idx + 1];
            let b = rgb[idx + 2];
            if r > thresh || g > thresh || b > thresh {
                if col < min_x { min_x = col; }
                if col > max_x { max_x = col; }
                if row < min_y { min_y = row; }
                if row > max_y { max_y = row; }
            }
        }
    }

    if max_x < min_x || max_y < min_y {
        return (0, 0, width, height);
    }

    let pad = CROP_PADDING as usize;
    let x = min_x.saturating_sub(pad) as u32;
    let y = min_y.saturating_sub(pad) as u32;
    let x2 = ((max_x + pad + 1).min(w)) as u32;
    let y2 = ((max_y + pad + 1).min(h)) as u32;

    (x, y, x2 - x, y2 - y)
}

fn crop_rgb(rgb: &[u8], src_width: u32, cx: u32, cy: u32, cw: u32, ch: u32) -> Vec<u8> {
    let sw = src_width as usize;
    let mut out = vec![0u8; (cw * ch * 3) as usize];
    for row in 0..ch as usize {
        let src_row = (cy as usize + row) * sw;
        let src_start = (src_row + cx as usize) * 3;
        let dst_start = row * cw as usize * 3;
        let len = cw as usize * 3;
        out[dst_start..dst_start + len].copy_from_slice(&rgb[src_start..src_start + len]);
    }
    out
}

// ── Bilinear scaling ────────────────────────────────────────────────

fn scale_dimensions(w: u32, h: u32, target_w: u32) -> (u32, u32) {
    if w == 0 || h == 0 { return (target_w, target_w); }
    let scale = target_w as f32 / w as f32;
    let new_h = (h as f32 * scale).round() as u32;
    (target_w, new_h.max(1))
}

fn bilinear_scale(src: &[u8], sw: u32, sh: u32, dw: u32, dh: u32) -> Vec<u8> {
    let mut dst = vec![0u8; (dw * dh * 3) as usize];
    let x_ratio = if dw > 1 { (sw - 1) as f32 / (dw - 1) as f32 } else { 0.0 };
    let y_ratio = if dh > 1 { (sh - 1) as f32 / (dh - 1) as f32 } else { 0.0 };

    for dy in 0..dh {
        let sy_f = dy as f32 * y_ratio;
        let sy = sy_f as u32;
        let sy1 = (sy + 1).min(sh - 1);
        let fy = sy_f - sy as f32;

        for dx in 0..dw {
            let sx_f = dx as f32 * x_ratio;
            let sx = sx_f as u32;
            let sx1 = (sx + 1).min(sw - 1);
            let fx = sx_f - sx as f32;

            for c in 0..3usize {
                let p00 = src[(sy * sw + sx) as usize * 3 + c] as f32;
                let p10 = src[(sy * sw + sx1) as usize * 3 + c] as f32;
                let p01 = src[(sy1 * sw + sx) as usize * 3 + c] as f32;
                let p11 = src[(sy1 * sw + sx1) as usize * 3 + c] as f32;

                let top = p00 + (p10 - p00) * fx;
                let bot = p01 + (p11 - p01) * fx;
                let val = top + (bot - top) * fy;

                dst[(dy * dw + dx) as usize * 3 + c] = val.round() as u8;
            }
        }
    }
    dst
}
