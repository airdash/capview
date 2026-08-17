//! macOS clipboard: copy PNG data via NSPasteboard.
//!
//! Replaces the Wayland clipboard module.  NSPasteboard is synchronous
//! and doesn't need a background thread.
//!
//! IMPORTANT: On ARM64 macOS, objc_msgSend must NOT be called through a
//! variadic declaration — see capture_mac.rs for details.

use anyhow::{bail, Result};

type Id = *mut libc::c_void;
type Sel = *mut libc::c_void;

#[link(name = "objc", kind = "dylib")]
extern "C" {
    fn objc_getClass(name: *const libc::c_char) -> Id;
    fn sel_registerName(name: *const libc::c_char) -> Sel;
    fn objc_msgSend();
}

macro_rules! cls {
    ($name:expr) => { objc_getClass(concat!($name, "\0").as_ptr() as *const _) };
}
macro_rules! sel {
    ($name:expr) => { sel_registerName(concat!($name, "\0").as_ptr() as *const _) };
}

macro_rules! msg {
    ($obj:expr, $sel:expr) => {{
        let f: unsafe extern "C" fn(Id, Sel) -> Id =
            std::mem::transmute(objc_msgSend as *const ());
        f($obj, $sel)
    }};
    ($obj:expr, $sel:expr, $a1:expr) => {{
        let f: unsafe extern "C" fn(Id, Sel, Id) -> Id =
            std::mem::transmute(objc_msgSend as *const ());
        f($obj, $sel, $a1)
    }};
    ($obj:expr, $sel:expr, $a1:expr, $a2:expr) => {{
        let f: unsafe extern "C" fn(Id, Sel, Id, Id) -> Id =
            std::mem::transmute(objc_msgSend as *const ());
        f($obj, $sel, $a1, $a2)
    }};
}

pub fn copy_to_clipboard(png_data: &[u8], debug: bool) -> Result<()> {
    if debug { eprintln!("debug: clipboard: {}B png → NSPasteboard", png_data.len()); }

    unsafe {
        let pasteboard = msg!(cls!("NSPasteboard"), sel!("generalPasteboard"));
        if pasteboard.is_null() { bail!("NSPasteboard.generalPasteboard returned nil"); }

        msg!(pasteboard, sel!("clearContents"));

        // dataWithBytes:length: takes (ptr, usize)
        let f: unsafe extern "C" fn(Id, Sel, *const libc::c_void, usize) -> Id =
            std::mem::transmute(objc_msgSend as *const ());
        let nsdata = f(cls!("NSData"), sel!("dataWithBytes:length:"),
            png_data.as_ptr() as *const libc::c_void, png_data.len());
        if nsdata.is_null() { bail!("NSData creation failed"); }

        let png_type = nsstring("public.png");

        // setData:forType: returns BOOL
        let f_set: unsafe extern "C" fn(Id, Sel, Id, Id) -> u8 =
            std::mem::transmute(objc_msgSend as *const ());
        let ok = f_set(pasteboard, sel!("setData:forType:"), nsdata, png_type);
        if ok == 0 { bail!("NSPasteboard setData:forType: failed"); }

        if debug { eprintln!("debug: clipboard: copied to pasteboard"); }
    }
    Ok(())
}

/// Clipboard is always available on macOS.
/// Maintains API compatibility with the Linux module's is_wayland() guard.
pub fn is_wayland() -> bool { true }

unsafe fn nsstring(s: &str) -> Id {
    let c = std::ffi::CString::new(s).unwrap();
    let f: unsafe extern "C" fn(Id, Sel, *const libc::c_char) -> Id =
        std::mem::transmute(objc_msgSend as *const ());
    f(cls!("NSString"), sel!("stringWithUTF8String:"), c.as_ptr())
}
