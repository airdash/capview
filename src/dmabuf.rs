//! EGL DMA-BUF zero-copy frame import.
//!
//! Uses `EGL_EXT_image_dma_buf_import` to create EGLImage objects from
//! V4L2 DMA-BUF file descriptors, then binds them to GL textures via
//! `glEGLImageTargetTexture2DOES`.  This eliminates the per-frame
//! CPU→GPU texture upload in the GL render path.
//!
//! Entirely optional — if libEGL is missing, the extension is absent,
//! or the driver cannot export DMA-BUFs, the GL renderer falls back to
//! the normal `glTexSubImage2D` upload path.

use std::ffi::{c_void, CStr};
use std::os::unix::io::RawFd;

use crate::capture::{V4L2_PIX_FMT_NV12, V4L2_PIX_FMT_YUYV, V4L2_PIX_FMT_UYVY};

// ── EGL types ───────────────────────────────────────────────────────

type EGLDisplay = *mut c_void;
type EGLContext = *mut c_void;
type EGLImage = *mut c_void;
type EGLint = i32;

const EGL_NO_DISPLAY: EGLDisplay = std::ptr::null_mut();
const EGL_NO_IMAGE: EGLImage = std::ptr::null_mut();
const EGL_NO_CONTEXT: EGLContext = std::ptr::null_mut();

// EGL constants
const EGL_EXTENSIONS: EGLint = 0x3055;
const EGL_NONE: EGLint = 0x3038;
const EGL_WIDTH: EGLint = 0x3057;
const EGL_HEIGHT: EGLint = 0x3058;
const EGL_LINUX_DMA_BUF_EXT: u32 = 0x3270;
const EGL_LINUX_DRM_FOURCC_EXT: EGLint = 0x3271;
const EGL_DMA_BUF_PLANE0_FD_EXT: EGLint = 0x3272;
const EGL_DMA_BUF_PLANE0_OFFSET_EXT: EGLint = 0x3273;
const EGL_DMA_BUF_PLANE0_PITCH_EXT: EGLint = 0x3274;

// DRM fourcc codes (same byte-order encoding as V4L2)
const DRM_FORMAT_R8: u32 = 0x20203852;
const DRM_FORMAT_GR88: u32 = 0x38385247;
const DRM_FORMAT_ABGR8888: u32 = 0x34324241;

// GL
const GL_TEXTURE_2D: u32 = 0x0DE1;

// ── EGL function pointer types ──────────────────────────────────────

type FnEglGetCurrentDisplay = unsafe extern "C" fn() -> EGLDisplay;
type FnEglQueryString = unsafe extern "C" fn(EGLDisplay, EGLint) -> *const i8;
type FnEglCreateImageKHR = unsafe extern "C" fn(
    EGLDisplay, EGLContext, u32, *mut c_void, *const EGLint,
) -> EGLImage;
type FnEglDestroyImageKHR = unsafe extern "C" fn(EGLDisplay, EGLImage) -> u32;
type FnEglGetError = unsafe extern "C" fn() -> EGLint;
type FnGlEGLImageTargetTexture2DOES = unsafe extern "C" fn(u32, EGLImage);

struct EglFns {
    get_current_display: FnEglGetCurrentDisplay,
    query_string: FnEglQueryString,
    create_image: FnEglCreateImageKHR,
    destroy_image: FnEglDestroyImageKHR,
    get_error: FnEglGetError,
    image_target_tex: FnGlEGLImageTargetTexture2DOES,
}

// ── Per-buffer EGLImage set ─────────────────────────────────────────

/// One set of EGLImages for a single V4L2 buffer.
///
/// For NV12: `planes[0]` = Y (R8), `planes[1]` = UV (GR88).
/// For YUYV/UYVY: `planes[0]` = packed (ABGR8888 at width/2).
struct BufferImages {
    planes: Vec<EGLImage>,
}

// ── Public interface ────────────────────────────────────────────────

pub struct DmaBufImporter {
    lib: *mut c_void, // dlopen'd libEGL handle
    egl: EglFns,
    display: EGLDisplay,
    buffers: Vec<BufferImages>,
    #[allow(dead_code)]
    pixfmt: u32,
    filter: i32, // GL_NEAREST or GL_LINEAR based on smooth config
    params_set: Vec<bool>, // track whether tex params were applied per buffer
}

impl DmaBufImporter {
    /// Try to set up zero-copy DMA-BUF import.
    ///
    /// `gl_get_proc` — SDL's `gl_get_proc_address` (for loading GL extensions).
    /// `fds` — one DMA-BUF FD per V4L2 buffer (from `VIDIOC_EXPBUF`).
    ///
    /// Returns `Err` if any prerequisite is missing (libEGL, extensions, etc).
    pub fn new<F: FnMut(&str) -> *const c_void>(
        mut gl_get_proc: F,
        fds: &[RawFd],
        width: u32,
        height: u32,
        pixfmt: u32,
        smooth: bool,
        debug: bool,
    ) -> anyhow::Result<Self> {
        // ── Load libEGL ─────────────────────────────────────────────
        let lib = unsafe {
            libc::dlopen(b"libEGL.so.1\0".as_ptr() as *const _, libc::RTLD_LAZY)
        };
        if lib.is_null() {
            anyhow::bail!("cannot load libEGL.so.1");
        }

        // ── Resolve EGL functions ───────────────────────────────────
        let egl = unsafe {
            macro_rules! load {
                ($name:expr, $ty:ty) => {{
                    let p = libc::dlsym(lib, concat!($name, "\0").as_ptr() as *const _);
                    if p.is_null() {
                        libc::dlclose(lib);
                        anyhow::bail!(concat!("missing EGL symbol: ", $name));
                    }
                    std::mem::transmute::<*mut c_void, $ty>(p)
                }};
            }
            EglFns {
                get_current_display: load!("eglGetCurrentDisplay", FnEglGetCurrentDisplay),
                query_string: load!("eglQueryString", FnEglQueryString),
                create_image: load!("eglCreateImageKHR", FnEglCreateImageKHR),
                destroy_image: load!("eglDestroyImageKHR", FnEglDestroyImageKHR),
                get_error: load!("eglGetError", FnEglGetError),
                image_target_tex: {
                    let p = gl_get_proc("glEGLImageTargetTexture2DOES");
                    if p.is_null() {
                        libc::dlclose(lib);
                        anyhow::bail!("glEGLImageTargetTexture2DOES not available");
                    }
                    std::mem::transmute::<*const c_void, FnGlEGLImageTargetTexture2DOES>(p)
                },
            }
        };

        // ── Get current EGL display ─────────────────────────────────
        let display = unsafe { (egl.get_current_display)() };
        if display == EGL_NO_DISPLAY {
            unsafe { libc::dlclose(lib); }
            anyhow::bail!("no current EGL display (X11/GLX?)");
        }

        // ── Check for DMA-BUF import extension ─────────────────────
        let extensions = unsafe {
            let ptr = (egl.query_string)(display, EGL_EXTENSIONS);
            if ptr.is_null() {
                libc::dlclose(lib);
                anyhow::bail!("eglQueryString(EGL_EXTENSIONS) returned NULL");
            }
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        };

        if !extensions.contains("EGL_EXT_image_dma_buf_import") {
            unsafe { libc::dlclose(lib); }
            anyhow::bail!("EGL_EXT_image_dma_buf_import not supported");
        }

        if debug {
            eprintln!("debug: EGL display {:?}, DMA-BUF import extension present", display);
        }

        let filter = if smooth { 0x2601 /* GL_LINEAR */ } else { 0x2600 /* GL_NEAREST */ } as i32;

        // ── Create EGLImages for each buffer ────────────────────────
        let mut importer = Self {
            lib,
            egl,
            display,
            buffers: Vec::with_capacity(fds.len()),
            pixfmt,
            filter,
            params_set: vec![false; fds.len()],
        };

        for (i, &fd) in fds.iter().enumerate() {
            match importer.create_images(fd, width, height, pixfmt) {
                Ok(images) => {
                    if debug {
                        eprintln!("debug: DMA-BUF buffer {} → {} EGLImage(s)", i, images.planes.len());
                    }
                    importer.buffers.push(images);
                }
                Err(e) => {
                    // Clean up already-created images
                    importer.destroy_all_images();
                    unsafe { libc::dlclose(importer.lib); }
                    // Prevent Drop from double-close
                    importer.lib = std::ptr::null_mut();
                    anyhow::bail!("DMA-BUF buffer {}: {}", i, e);
                }
            }
        }

        Ok(importer)
    }

    /// Bind the EGLImages for `buf_index` to the given GL textures.
    ///
    /// For NV12: `textures` = `[tex_y, tex_uv]`.
    /// For YUYV/UYVY: `textures` = `[tex_packed]`.
    ///
    /// After this call the textures are backed by the DMA-BUF memory —
    /// no `glTexSubImage2D` needed.
    pub fn bind(&mut self, buf_index: u32, textures: &[glow::Texture], gl: &glow::Context) {
        let idx = buf_index as usize;
        let images = &self.buffers[idx];
        let need_params = !self.params_set[idx];
        for (img, &tex) in images.planes.iter().zip(textures.iter()) {
            unsafe {
                use glow::HasContext;
                gl.bind_texture(glow::TEXTURE_2D, Some(tex));
                (self.egl.image_target_tex)(GL_TEXTURE_2D, *img);
                // Re-apply texture parameters only on first bind per buffer —
                // glEGLImageTargetTexture2DOES may reset them on some drivers.
                if need_params {
                    gl.tex_parameter_i32(
                        glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, self.filter,
                    );
                    gl.tex_parameter_i32(
                        glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, self.filter,
                    );
                    gl.tex_parameter_i32(
                        glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32,
                    );
                    gl.tex_parameter_i32(
                        glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32,
                    );
                }
            }
        }
        self.params_set[idx] = true;
    }

    /// How many planes does a single bind-set have.
    #[allow(dead_code)]
    pub fn planes_per_buffer(&self) -> usize {
        self.buffers.first().map_or(0, |b| b.planes.len())
    }

    // ── Internal ────────────────────────────────────────────────────

    fn create_images(
        &self,
        fd: RawFd,
        width: u32,
        height: u32,
        pixfmt: u32,
    ) -> anyhow::Result<BufferImages> {
        match pixfmt {
            V4L2_PIX_FMT_NV12 => {
                let y = self.create_one(fd, width, height, DRM_FORMAT_R8, width, 0)?;
                let uv_offset = (width * height) as i32;
                let uv = self.create_one(
                    fd, width / 2, height / 2, DRM_FORMAT_GR88, width, uv_offset,
                )?;
                Ok(BufferImages { planes: vec![y, uv] })
            }
            V4L2_PIX_FMT_YUYV | V4L2_PIX_FMT_UYVY => {
                let packed = self.create_one(
                    fd, width / 2, height, DRM_FORMAT_ABGR8888, width * 2, 0,
                )?;
                Ok(BufferImages { planes: vec![packed] })
            }
            _ => anyhow::bail!("unsupported pixel format for DMA-BUF import"),
        }
    }

    fn create_one(
        &self,
        fd: RawFd,
        width: u32,
        height: u32,
        fourcc: u32,
        pitch: u32,
        offset: i32,
    ) -> anyhow::Result<EGLImage> {
        let attrs: [EGLint; 13] = [
            EGL_WIDTH,                       width as EGLint,
            EGL_HEIGHT,                      height as EGLint,
            EGL_LINUX_DRM_FOURCC_EXT,        fourcc as EGLint,
            EGL_DMA_BUF_PLANE0_FD_EXT,      fd as EGLint,
            EGL_DMA_BUF_PLANE0_OFFSET_EXT,  offset,
            EGL_DMA_BUF_PLANE0_PITCH_EXT,   pitch as EGLint,
            EGL_NONE,
        ];

        let img = unsafe {
            (self.egl.create_image)(
                self.display,
                EGL_NO_CONTEXT,
                EGL_LINUX_DMA_BUF_EXT,
                std::ptr::null_mut(), // no client buffer
                attrs.as_ptr(),
            )
        };

        if img == EGL_NO_IMAGE {
            let err = unsafe { (self.egl.get_error)() };
            anyhow::bail!(
                "eglCreateImageKHR failed (fourcc 0x{:08x} {}x{} pitch={} off={}): EGL error 0x{:04x}",
                fourcc, width, height, pitch, offset, err,
            );
        }

        Ok(img)
    }

    fn destroy_all_images(&self) {
        for buf in &self.buffers {
            for &img in &buf.planes {
                unsafe {
                    (self.egl.destroy_image)(self.display, img);
                }
            }
        }
    }
}

impl Drop for DmaBufImporter {
    fn drop(&mut self) {
        self.destroy_all_images();
        if !self.lib.is_null() {
            unsafe { libc::dlclose(self.lib); }
        }
    }
}
