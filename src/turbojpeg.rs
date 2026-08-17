//! Fast JPEG encode/decode via libturbojpeg FFI.
//!
//! Used for streaming — the pure-Rust encoder in `jpeg.rs` is fine for
//! one-off screenshots but too slow for per-frame 1080p at 60fps.
//! Falls back gracefully: if libturbojpeg is not installed, `init()`
//! returns `Err` and the caller can decide what to do.

use std::ffi::c_void;
use std::ptr;

// ── TurboJPEG constants ─────────────────────────────────────────────

const TJPF_RGB: i32 = 0;
const TJSAMP_420: i32 = 1;

// ── FFI types ───────────────────────────────────────────────────────

type TjHandle = *mut c_void;

type FnTjInitCompress = unsafe extern "C" fn() -> TjHandle;
type FnTjInitDecompress = unsafe extern "C" fn() -> TjHandle;
type FnTjDestroy = unsafe extern "C" fn(TjHandle) -> i32;
type FnTjCompress2 = unsafe extern "C" fn(
    handle: TjHandle,
    src: *const u8,
    width: i32,
    pitch: i32,
    height: i32,
    pixel_format: i32,
    jpeg_buf: *mut *mut u8,
    jpeg_size: *mut libc::c_ulong,
    jpeg_subsamp: i32,
    jpeg_qual: i32,
    flags: i32,
) -> i32;
type FnTjDecompressHeader2 = unsafe extern "C" fn(
    handle: TjHandle,
    jpeg_buf: *const u8,
    jpeg_size: libc::c_ulong,
    width: *mut i32,
    height: *mut i32,
    jpeg_subsamp: *mut i32,
) -> i32;
type FnTjDecompress2 = unsafe extern "C" fn(
    handle: TjHandle,
    jpeg_buf: *const u8,
    jpeg_size: libc::c_ulong,
    dst: *mut u8,
    width: i32,
    pitch: i32,
    height: i32,
    pixel_format: i32,
    flags: i32,
) -> i32;
type FnTjFree = unsafe extern "C" fn(*mut u8);
type FnTjGetErrorStr = unsafe extern "C" fn() -> *const i8;

struct TjFns {
    init_compress: FnTjInitCompress,
    init_decompress: FnTjInitDecompress,
    destroy: FnTjDestroy,
    compress2: FnTjCompress2,
    decompress_header2: FnTjDecompressHeader2,
    decompress2: FnTjDecompress2,
    free: FnTjFree,
    get_error_str: FnTjGetErrorStr,
}

// ── Public API ──────────────────────────────────────────────────────

pub struct TurboJpeg {
    lib: *mut c_void,
    fns: TjFns,
    comp: TjHandle,
    decomp: TjHandle,
}

// Safety: TurboJPEG handles are per-instance (not global), and we own them.
// The caller must ensure single-threaded access per instance.
unsafe impl Send for TurboJpeg {}

impl TurboJpeg {
    /// Load libturbojpeg and create compressor + decompressor handles.
    pub fn new() -> anyhow::Result<Self> {
        let lib = unsafe {
            #[cfg(target_os = "linux")]
            const LIB_NAMES: &[&[u8]] = &[
                b"libturbojpeg.so.0\0",
                b"libturbojpeg.so\0",
            ];
            #[cfg(target_os = "macos")]
            const LIB_NAMES: &[&[u8]] = &[
                b"libturbojpeg.dylib\0",
                b"/opt/homebrew/opt/jpeg-turbo/lib/libturbojpeg.dylib\0",
                b"/usr/local/opt/jpeg-turbo/lib/libturbojpeg.dylib\0",
            ];
            let mut h: *mut libc::c_void = std::ptr::null_mut();
            for name in LIB_NAMES {
                h = libc::dlopen(name.as_ptr() as *const _, libc::RTLD_LAZY);
                if !h.is_null() { break; }
            }
            if h.is_null() {
                anyhow::bail!("cannot load libturbojpeg");
            }
            h
        };

        let fns = unsafe {
            macro_rules! load {
                ($name:expr, $ty:ty) => {{
                    let p = libc::dlsym(lib, concat!($name, "\0").as_ptr() as *const _);
                    if p.is_null() {
                        libc::dlclose(lib);
                        anyhow::bail!(concat!("missing turbojpeg symbol: ", $name));
                    }
                    std::mem::transmute::<*mut c_void, $ty>(p)
                }};
            }
            TjFns {
                init_compress: load!("tjInitCompress", FnTjInitCompress),
                init_decompress: load!("tjInitDecompress", FnTjInitDecompress),
                destroy: load!("tjDestroy", FnTjDestroy),
                compress2: load!("tjCompress2", FnTjCompress2),
                decompress_header2: load!("tjDecompressHeader2", FnTjDecompressHeader2),
                decompress2: load!("tjDecompress2", FnTjDecompress2),
                free: load!("tjFree", FnTjFree),
                get_error_str: load!("tjGetErrorStr", FnTjGetErrorStr),
            }
        };

        let comp = unsafe { (fns.init_compress)() };
        if comp.is_null() {
            unsafe { libc::dlclose(lib); }
            anyhow::bail!("tjInitCompress failed");
        }

        let decomp = unsafe { (fns.init_decompress)() };
        if decomp.is_null() {
            unsafe { (fns.destroy)(comp); libc::dlclose(lib); }
            anyhow::bail!("tjInitDecompress failed");
        }

        Ok(Self { lib, fns, comp, decomp })
    }

    /// Compress RGB data to JPEG.  Returns the JPEG bytes.
    pub fn compress(&self, rgb: &[u8], width: u32, height: u32, quality: u32) -> anyhow::Result<Vec<u8>> {
        let mut jpeg_buf: *mut u8 = ptr::null_mut();
        let mut jpeg_size: libc::c_ulong = 0;

        let ret = unsafe {
            (self.fns.compress2)(
                self.comp,
                rgb.as_ptr(),
                width as i32,
                (width * 3) as i32, // pitch
                height as i32,
                TJPF_RGB,
                &mut jpeg_buf,
                &mut jpeg_size,
                TJSAMP_420,
                quality.clamp(1, 100) as i32,
                0, // flags
            )
        };

        if ret != 0 {
            let msg = self.error_string();
            if !jpeg_buf.is_null() {
                unsafe { (self.fns.free)(jpeg_buf); }
            }
            anyhow::bail!("tjCompress2: {}", msg);
        }

        let out = unsafe {
            std::slice::from_raw_parts(jpeg_buf, jpeg_size as usize).to_vec()
        };
        unsafe { (self.fns.free)(jpeg_buf); }

        Ok(out)
    }

    /// Decompress JPEG into a pre-allocated buffer, avoiding per-frame allocation.
    /// Buffer is resized if needed.  Returns `(width, height)`.
    pub fn decompress_into(&self, jpeg: &[u8], buf: &mut Vec<u8>) -> anyhow::Result<(u32, u32)> {
        let mut width: i32 = 0;
        let mut height: i32 = 0;
        let mut subsamp: i32 = 0;

        let ret = unsafe {
            (self.fns.decompress_header2)(
                self.decomp,
                jpeg.as_ptr(),
                jpeg.len() as libc::c_ulong,
                &mut width,
                &mut height,
                &mut subsamp,
            )
        };
        if ret != 0 {
            anyhow::bail!("tjDecompressHeader2: {}", self.error_string());
        }

        let rgb_size = (width as usize) * (height as usize) * 3;
        buf.resize(rgb_size, 0);

        let ret = unsafe {
            (self.fns.decompress2)(
                self.decomp,
                jpeg.as_ptr(),
                jpeg.len() as libc::c_ulong,
                buf.as_mut_ptr(),
                width,
                (width * 3) as i32,
                height,
                TJPF_RGB,
                0,
            )
        };
        if ret != 0 {
            anyhow::bail!("tjDecompress2: {}", self.error_string());
        }

        Ok((width as u32, height as u32))
    }

    /// Decompress JPEG data to RGB.  Returns `(rgb_data, width, height)`.
    pub fn decompress(&self, jpeg: &[u8]) -> anyhow::Result<(Vec<u8>, u32, u32)> {
        let mut width: i32 = 0;
        let mut height: i32 = 0;
        let mut subsamp: i32 = 0;

        let ret = unsafe {
            (self.fns.decompress_header2)(
                self.decomp,
                jpeg.as_ptr(),
                jpeg.len() as libc::c_ulong,
                &mut width,
                &mut height,
                &mut subsamp,
            )
        };
        if ret != 0 {
            anyhow::bail!("tjDecompressHeader2: {}", self.error_string());
        }

        let rgb_size = (width as usize) * (height as usize) * 3;
        let mut rgb = vec![0u8; rgb_size];

        let ret = unsafe {
            (self.fns.decompress2)(
                self.decomp,
                jpeg.as_ptr(),
                jpeg.len() as libc::c_ulong,
                rgb.as_mut_ptr(),
                width,
                (width * 3) as i32,
                height,
                TJPF_RGB,
                0,
            )
        };
        if ret != 0 {
            anyhow::bail!("tjDecompress2: {}", self.error_string());
        }

        Ok((rgb, width as u32, height as u32))
    }

    fn error_string(&self) -> String {
        unsafe {
            let ptr = (self.fns.get_error_str)();
            if ptr.is_null() {
                "unknown error".to_string()
            } else {
                std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
            }
        }
    }
}

impl Drop for TurboJpeg {
    fn drop(&mut self) {
        unsafe {
            if !self.comp.is_null() { (self.fns.destroy)(self.comp); }
            if !self.decomp.is_null() { (self.fns.destroy)(self.decomp); }
            if !self.lib.is_null() { libc::dlclose(self.lib); }
        }
    }
}
