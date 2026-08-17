//! Runtime-loaded filter plugins via dlopen.
//!
//! Each plugin is a shared library (.so) exporting C-ABI functions defined
//! in `capview-plugin.h`. Plugins are loaded from paths listed in the
//! config file (`plugins = /path/to/filter.so:optional,args`).

use anyhow::{bail, Result};
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::Path;

const ABI_VERSION: i32 = 1;

// ── dlopen / dlsym FFI ──────────────────────────────────────────────

const RTLD_NOW: libc::c_int = 0x2;

extern "C" {
    fn dlopen(filename: *const c_char, flags: libc::c_int) -> *mut libc::c_void;
    fn dlsym(handle: *mut libc::c_void, symbol: *const c_char) -> *mut libc::c_void;
    fn dlclose(handle: *mut libc::c_void) -> libc::c_int;
    fn dlerror() -> *const c_char;
}

fn dl_error() -> String {
    unsafe {
        let p = dlerror();
        if p.is_null() {
            "unknown dlopen error".into()
        } else {
            CStr::from_ptr(p).to_string_lossy().to_string()
        }
    }
}

// ── Plugin vtable (resolved function pointers) ─────────────────────

type FnAbiVersion = unsafe extern "C" fn() -> i32;
type FnName = unsafe extern "C" fn() -> *const c_char;
type FnInit = unsafe extern "C" fn(u32, u32, u32, u32, *const c_char) -> i32;
type FnProcess = unsafe extern "C" fn(
    *const u8, u32,          // input, input_len
    *mut u8, u32,            // output, output_cap
    *mut u32,                // output_len
    u32, u32,                // width, height
) -> i32;
type FnDestroy = unsafe extern "C" fn();

pub struct FilterPlugin {
    _handle: *mut libc::c_void,
    name: String,
    path: String,
    args: Option<String>,
    fn_init: FnInit,
    fn_process: FnProcess,
    fn_destroy: FnDestroy,
}

// We make sure to clean up properly; the handle is managed exclusively.
unsafe impl Send for FilterPlugin {}

impl FilterPlugin {
    /// Load a plugin from a shared library path.
    /// `spec` is "path/to/lib.so" or "path/to/lib.so:arg1,arg2".
    pub fn load(spec: &str) -> Result<Self> {
        let (path, args) = match spec.split_once(':') {
            Some((p, a)) => (p.trim(), Some(a.trim().to_string())),
            None => (spec.trim(), None),
        };

        if !Path::new(path).exists() {
            bail!("plugin not found: {}", path);
        }

        let c_path = CString::new(path)?;

        unsafe {
            // Clear any previous error
            dlerror();

            let handle = dlopen(c_path.as_ptr(), RTLD_NOW);
            if handle.is_null() {
                bail!("dlopen({}): {}", path, dl_error());
            }

            // Resolve required symbols
            let fn_abi = resolve::<FnAbiVersion>(handle, "capview_filter_abi_version", path)?;
            let fn_name = resolve::<FnName>(handle, "capview_filter_name", path)?;
            let fn_init = resolve::<FnInit>(handle, "capview_filter_init", path)?;
            let fn_process = resolve::<FnProcess>(handle, "capview_filter_process", path)?;
            let fn_destroy = resolve::<FnDestroy>(handle, "capview_filter_destroy", path)?;

            // Check ABI version
            let abi = fn_abi();
            if abi != ABI_VERSION {
                dlclose(handle);
                bail!("plugin {}: ABI version mismatch (plugin={}, expected={})",
                      path, abi, ABI_VERSION);
            }

            let name_ptr = fn_name();
            let name = if name_ptr.is_null() {
                "unnamed".to_string()
            } else {
                CStr::from_ptr(name_ptr).to_string_lossy().to_string()
            };

            Ok(Self {
                _handle: handle,
                name,
                path: path.to_string(),
                args,
                fn_init,
                fn_process,
                fn_destroy,
            })
        }
    }

    /// Name of the filter (from the plugin).
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Path the plugin was loaded from.
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Initialise the filter with capture parameters.
    pub fn init(&self, width: u32, height: u32, fps: u32, pixfmt: u32) -> Result<()> {
        let c_args = match &self.args {
            Some(a) => Some(CString::new(a.as_str())?),
            None => None,
        };
        let args_ptr = c_args.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());

        let ret = unsafe { (self.fn_init)(width, height, fps, pixfmt, args_ptr) };
        if ret != 0 {
            bail!("plugin '{}' init failed (returned {})", self.name, ret);
        }
        Ok(())
    }

    /// Process one frame. Returns a Vec of output frames (each is a raw byte
    /// buffer the same pixel format as input). Typically returns 1 frame, but
    /// interpolation plugins may return 2+.
    pub fn process(&self, input: &[u8], width: u32, height: u32) -> Result<Vec<Vec<u8>>> {
        let frame_size = input.len();
        // Allocate room for up to 4 output frames (generous for interpolation)
        let max_frames: usize = 4;
        let mut output = vec![0u8; frame_size * max_frames];
        let mut output_len: u32 = 0;

        let n = unsafe {
            (self.fn_process)(
                input.as_ptr(), input.len() as u32,
                output.as_mut_ptr(), output.len() as u32,
                &mut output_len,
                width, height,
            )
        };

        if n < 0 {
            bail!("plugin '{}' process error (returned {})", self.name, n);
        }

        if n == 0 {
            return Ok(Vec::new()); // frame skipped
        }

        let n = n as usize;
        let total = output_len as usize;

        if total > output.len() || total != n * frame_size {
            bail!("plugin '{}': output size mismatch (got {} bytes for {} frames, expected {})",
                  self.name, total, n, n * frame_size);
        }

        let mut frames = Vec::with_capacity(n);
        for i in 0..n {
            let start = i * frame_size;
            frames.push(output[start..start + frame_size].to_vec());
        }
        Ok(frames)
    }
}

impl Drop for FilterPlugin {
    fn drop(&mut self) {
        unsafe {
            (self.fn_destroy)();
            dlclose(self._handle);
        }
    }
}

unsafe fn resolve<T>(handle: *mut libc::c_void, sym: &str, path: &str) -> Result<T> {
    let c_sym = CString::new(sym).unwrap();
    dlerror(); // clear
    let ptr = dlsym(handle, c_sym.as_ptr());
    if ptr.is_null() {
        // dlsym can legitimately return NULL if the symbol's value is NULL,
        // but for function pointers that's never valid — check dlerror.
        let err = dlerror();
        if !err.is_null() {
            let msg = CStr::from_ptr(err).to_string_lossy();
            bail!("plugin {}: missing symbol '{}': {}", path, sym, msg);
        }
        bail!("plugin {}: symbol '{}' resolved to NULL", path, sym);
    }
    Ok(std::mem::transmute_copy(&ptr))
}

// ── Pipeline: ordered list of plugins ───────────────────────────────

pub struct FilterPipeline {
    filters: Vec<FilterPlugin>,
}

impl FilterPipeline {
    /// Load all plugins from config specs and initialise them.
    pub fn load(specs: &[String], width: u32, height: u32, fps: u32, pixfmt: u32, debug: bool) -> Result<Self> {
        let mut filters = Vec::new();

        for spec in specs {
            if debug { eprintln!("debug: loading plugin: {}", spec); }
            let plugin = FilterPlugin::load(spec)?;
            if debug { eprintln!("debug: loaded '{}' from {}", plugin.name(), plugin.path()); }
            plugin.init(width, height, fps, pixfmt)?;
            if debug { eprintln!("debug: plugin '{}' initialised", plugin.name()); }
            eprintln!("plugin: {}", plugin.name());
            filters.push(plugin);
        }

        Ok(Self { filters })
    }

    /// True if there are no plugins loaded.
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.filters.is_empty()
    }

    /// Run the full filter pipeline on a frame.
    /// Returns the final list of output frames (empty = skip rendering).
    pub fn process(&self, input: &[u8], width: u32, height: u32) -> Result<Vec<Vec<u8>>> {
        if self.filters.is_empty() {
            return Ok(vec![input.to_vec()]);
        }

        let mut frames = vec![input.to_vec()];

        for filter in &self.filters {
            let mut next_frames = Vec::new();
            for frame in &frames {
                let mut out = filter.process(frame, width, height)?;
                next_frames.append(&mut out);
            }
            if next_frames.is_empty() {
                return Ok(Vec::new()); // chain produced nothing
            }
            frames = next_frames;
        }

        Ok(frames)
    }
}
