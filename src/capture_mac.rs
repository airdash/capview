//! macOS capture: AVFoundation via raw Objective-C FFI.
//!
//! Replaces V4L2 capture.  Uses AVCaptureSession + AVCaptureVideoDataOutput
//! with a delegate callback that signals a pipe fd for poll() compatibility
//! with the existing main loop.
//!
//! IMPORTANT: On ARM64 macOS, objc_msgSend must NOT be called through a
//! variadic extern "C" declaration — the ARM64 ABI uses different register
//! allocation for variadic vs non-variadic calls, and objc_msgSend is a
//! non-variadic trampoline.  Every call site must cast the raw function
//! pointer to a correctly-typed non-variadic fn pointer.

use anyhow::{bail, Result};
use std::ffi::CString;
use std::os::unix::io::RawFd;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU32, Ordering};
use std::sync::atomic::AtomicI32;

// Re-export pixel format constants so the rest of the codebase can use
// `capture::V4L2_PIX_FMT_*` regardless of platform.
pub const V4L2_PIX_FMT_NV12: u32 = 0x3231564e;
pub const V4L2_PIX_FMT_YUYV: u32 = 0x56595559;
pub const V4L2_PIX_FMT_UYVY: u32 = 0x59565955;

// AVFoundation pixel format constants (CoreVideo FourCC)
const K_CV_NV12: u32 = 0x3432_3076; // '420v' — NV12 video-range
const K_CV_UYVY: u32 = 0x3279_7576; // '2vuy' — UYVY

type Id = *mut libc::c_void;
type Sel = *mut libc::c_void;

// ── ObjC FFI ────────────────────────────────────────────────────────

#[link(name = "objc", kind = "dylib")]
extern "C" {
    fn objc_getClass(name: *const libc::c_char) -> Id;
    fn sel_registerName(name: *const libc::c_char) -> Sel;
    // Raw pointer — we NEVER call this directly.  Every call site transmutes
    // to a properly-typed non-variadic fn pointer.
    fn objc_msgSend();
    fn objc_allocateClassPair(superclass: Id, name: *const libc::c_char, extra: usize) -> Id;
    fn objc_registerClassPair(cls: Id);
    fn class_addMethod(cls: Id, sel: Sel, imp: *const libc::c_void, types: *const libc::c_char) -> bool;
    fn class_addProtocol(cls: Id, protocol: Id) -> bool;
    fn objc_getProtocol(name: *const libc::c_char) -> Id;
    fn object_setInstanceVariable(obj: Id, name: *const libc::c_char, value: Id) -> Id;
    fn object_getInstanceVariable(obj: Id, name: *const libc::c_char, out: *mut Id) -> Id;
    fn class_addIvar(cls: Id, name: *const libc::c_char, size: usize, alignment: u8, types: *const libc::c_char) -> bool;
}

// AVFoundation constants
#[link(name = "AVFoundation", kind = "framework")]
extern "C" {
    static AVCaptureSessionPresetInputPriority: Id;
}

// CoreFoundation / CoreMedia / CoreVideo C functions
extern "C" {
    fn CFRelease(cf: Id);
    fn CFRetain(cf: Id) -> Id;
    fn CMSampleBufferGetImageBuffer(buf: Id) -> Id;
    fn CVPixelBufferLockBaseAddress(buf: Id, flags: u64) -> i32;
    fn CVPixelBufferUnlockBaseAddress(buf: Id, flags: u64) -> i32;
    fn CVPixelBufferGetBaseAddress(buf: Id) -> *mut u8;
    fn CVPixelBufferGetBaseAddressOfPlane(buf: Id, plane: usize) -> *mut u8;
    fn CVPixelBufferGetBytesPerRowOfPlane(buf: Id, plane: usize) -> usize;
    fn CVPixelBufferGetBytesPerRow(buf: Id) -> usize;
    fn CVPixelBufferGetWidth(buf: Id) -> usize;
    fn CVPixelBufferGetHeight(buf: Id) -> usize;
    fn CVPixelBufferIsPlanar(buf: Id) -> bool;
}

#[repr(C)]
struct CMVideoDimensions { width: i32, height: i32 }

extern "C" {
    fn CMVideoFormatDescriptionGetDimensions(desc: Id) -> CMVideoDimensions;
    fn CMFormatDescriptionGetMediaSubType(desc: Id) -> u32;
}

// ── Typed objc_msgSend trampolines ──────────────────────────────────
//
// On ARM64 Apple, objc_msgSend is a non-variadic trampoline.  Rust's
// variadic `extern "C" fn(...)` emits a different (variadic) calling
// convention, which puts arguments in the wrong registers.  Fix: cast
// the raw symbol to correctly-typed fn pointers.

macro_rules! cls {
    ($name:expr) => { objc_getClass(concat!($name, "\0").as_ptr() as *const _) };
}
macro_rules! sel {
    ($name:expr) => { sel_registerName(concat!($name, "\0").as_ptr() as *const _) };
}

/// msg![obj, sel] — zero extra args, returns Id
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
    ($obj:expr, $sel:expr, $a1:expr, $a2:expr, $a3:expr) => {{
        let f: unsafe extern "C" fn(Id, Sel, Id, Id, Id) -> Id =
            std::mem::transmute(objc_msgSend as *const ());
        f($obj, $sel, $a1, $a2, $a3)
    }};
}

/// msg_uint![obj, sel, ...] — returns Id, takes one usize arg
macro_rules! msg_uint {
    ($obj:expr, $sel:expr, $a1:expr) => {{
        let f: unsafe extern "C" fn(Id, Sel, usize) -> Id =
            std::mem::transmute(objc_msgSend as *const ());
        f($obj, $sel, $a1)
    }};
}

/// msg_ptr_count![obj, sel, ptr, count] — for arrayWithObjects:count:
macro_rules! msg_ptr_count {
    ($obj:expr, $sel:expr, $ptr:expr, $count:expr) => {{
        let f: unsafe extern "C" fn(Id, Sel, *const Id, usize) -> Id =
            std::mem::transmute(objc_msgSend as *const ());
        f($obj, $sel, $ptr, $count)
    }};
}

/// msg_i64![obj, sel, ...] — for methods taking i64 (e.g. position enum)
macro_rules! msg_id_id_i64 {
    ($obj:expr, $sel:expr, $a1:expr, $a2:expr, $a3:expr) => {{
        let f: unsafe extern "C" fn(Id, Sel, Id, Id, i64) -> Id =
            std::mem::transmute(objc_msgSend as *const ());
        f($obj, $sel, $a1, $a2, $a3)
    }};
}

/// msg_cstr![obj, sel, cstr] — for stringWithUTF8String: etc
macro_rules! msg_cstr {
    ($obj:expr, $sel:expr, $cstr:expr) => {{
        let f: unsafe extern "C" fn(Id, Sel, *const libc::c_char) -> Id =
            std::mem::transmute(objc_msgSend as *const ());
        f($obj, $sel, $cstr)
    }};
}

/// msg_u32![obj, sel, val] — for numberWithUnsignedInt:
macro_rules! msg_u32 {
    ($obj:expr, $sel:expr, $val:expr) => {{
        let f: unsafe extern "C" fn(Id, Sel, u32) -> Id =
            std::mem::transmute(objc_msgSend as *const ());
        f($obj, $sel, $val)
    }};
}

/// msg_id_ptr![obj, sel, id, ptr] — for deviceInputWithDevice:error:
macro_rules! msg_id_ptr {
    ($obj:expr, $sel:expr, $a1:expr, $a2:expr) => {{
        let f: unsafe extern "C" fn(Id, Sel, Id, *mut Id) -> Id =
            std::mem::transmute(objc_msgSend as *const ());
        f($obj, $sel, $a1, $a2)
    }};
}

/// msg_bool![obj, sel, id] — for setAlwaysDiscardsLateVideoFrames: (BOOL arg)
macro_rules! msg_bool_arg {
    ($obj:expr, $sel:expr, $val:expr) => {{
        let f: unsafe extern "C" fn(Id, Sel, i8) -> Id =
            std::mem::transmute(objc_msgSend as *const ());
        f($obj, $sel, $val)
    }};
}

fn nsstring(s: &str) -> Id {
    let c = CString::new(s).unwrap();
    unsafe { msg_cstr!(cls!("NSString"), sel!("stringWithUTF8String:"), c.as_ptr()) }
}

fn nsnumber_u32(v: u32) -> Id {
    unsafe { msg_u32!(cls!("NSNumber"), sel!("numberWithUnsignedInt:"), v) }
}

// ── Shared state between delegate callback and Capture ──────────────

struct SharedState {
    latest: AtomicPtr<libc::c_void>,
    seq: AtomicU32,
    pipe_w: RawFd,
    running: AtomicBool,
}

// ── Public types (matching capture.rs API) ──────────────────────────

pub struct V4l2Buffer {
    pub index: u32,
    pub length: u32,
    sequence: u32,
    _sample_buf: Id,
    _pixel_buf: Id,
}

impl V4l2Buffer {
    pub fn sequence(&self) -> u32 { self.sequence }
}

pub struct MappedBuffer {
    pub ptr: *mut u8,
    pub len: usize,
}

pub struct Capture {
    session: Id,
    delegate: Id,
    shared: *mut SharedState,
    pipe_r: RawFd,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub pixfmt: u32,
    held_buf: std::cell::Cell<Option<(Id, Id)>>,
    linear_buf: std::cell::UnsafeCell<Vec<u8>>,
}

unsafe impl Send for Capture {}

impl Capture {
    pub fn open(
        device: &str, width: u32, height: u32, fps: u32,
        pixfmt: u32, _buf_count: u32,
    ) -> Result<Self> {
        let t0 = std::time::Instant::now();
        macro_rules! ts {
            ($($arg:tt)*) => {
                eprintln!("[{:7.3}s] {}", t0.elapsed().as_secs_f64(), format!($($arg)*));
            };
        }

        unsafe {
            // ── Camera authorization ────────────────────────────
            let media_type = nsstring("vide");
            let status = msg!(cls!("AVCaptureDevice"),
                sel!("authorizationStatusForMediaType:"), media_type) as i64;
            ts!("auth status = {} (0=undetermined 1=restricted 2=denied 3=authorized)", status);
            // 0=NotDetermined, 1=Restricted, 2=Denied, 3=Authorized
            if status == 2 || status == 1 {
                bail!("camera access denied — grant in System Settings > Privacy > Camera");
            }
            if status == 0 {
                ts!("requesting camera permission (check the system dialog)...");
                request_camera_access();
                for _ in 0..300 {
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    let s = msg!(cls!("AVCaptureDevice"),
                        sel!("authorizationStatusForMediaType:"), nsstring("vide")) as i64;
                    if s != 0 {
                        ts!("auth response = {}", s);
                        if s == 2 || s == 1 {
                            bail!("camera access denied — grant in System Settings > Privacy > Camera");
                        }
                        break;
                    }
                }
            }

            // ── Pipe for poll() signalling ──────────────────────
            let mut fds = [0i32; 2];
            if libc::pipe(fds.as_mut_ptr()) != 0 { bail!("pipe() failed"); }
            let flags = libc::fcntl(fds[0], libc::F_GETFL);
            libc::fcntl(fds[0], libc::F_SETFL, flags | libc::O_NONBLOCK);

            let shared = Box::into_raw(Box::new(SharedState {
                latest: AtomicPtr::new(std::ptr::null_mut()),
                seq: AtomicU32::new(0),
                pipe_w: fds[1],
                running: AtomicBool::new(true),
            }));

            // ── Find capture device ─────────────────────────────
            ts!("discovering devices...");
            let dev = find_device(device)?;
            if dev.is_null() { bail!("capture device is nil"); }

            // Print device name
            let name_ns = msg!(dev, sel!("localizedName"));
            if !name_ns.is_null() {
                let name_c = msg!(name_ns, sel!("UTF8String")) as *const libc::c_char;
                if !name_c.is_null() {
                    ts!("device: {}", std::ffi::CStr::from_ptr(name_c).to_string_lossy());
                }
            }

            // ── Create session ──────────────────────────────────
            ts!("creating capture session...");
            let session = msg!(msg!(cls!("AVCaptureSession"), sel!("alloc")), sel!("init"));
            if session.is_null() { bail!("AVCaptureSession alloc/init failed"); }

            // ── Add input ───────────────────────────────────────
            let mut err: Id = std::ptr::null_mut();
            let input = msg_id_ptr!(
                cls!("AVCaptureDeviceInput"),
                sel!("deviceInputWithDevice:error:"),
                dev, &mut err
            );
            if input.is_null() {
                let msg = if !err.is_null() {
                    let desc = msg!(err, sel!("localizedDescription"));
                    let c = msg!(desc, sel!("UTF8String")) as *const libc::c_char;
                    if !c.is_null() {
                        std::ffi::CStr::from_ptr(c).to_string_lossy().to_string()
                    } else { "unknown error".into() }
                } else { "unknown error".into() };
                bail!("AVCaptureDeviceInput failed: {}", msg);
            }

            let can_add = (msg!(session, sel!("canAddInput:"), input) as usize) != 0;
            if !can_add { bail!("cannot add capture input to session"); }

            // Batch all session changes together
            msg!(session, sel!("beginConfiguration"));

            msg!(session, sel!("addInput:"), input);
            ts!("input added");

            // ── Add output ──────────────────────────────────────
            let avf_pixfmt = match pixfmt {
                V4L2_PIX_FMT_NV12 => K_CV_NV12,
                V4L2_PIX_FMT_UYVY => K_CV_UYVY,
                _ => K_CV_NV12,
            };

            let output = msg!(msg!(cls!("AVCaptureVideoDataOutput"), sel!("alloc")), sel!("init"));
            if output.is_null() { bail!("AVCaptureVideoDataOutput alloc/init failed"); }

            // Request pixel format
            let settings = msg!(cls!("NSDictionary"), sel!("dictionaryWithObject:forKey:"),
                nsnumber_u32(avf_pixfmt), nsstring("PixelFormatType"));
            msg!(output, sel!("setVideoSettings:"), settings);

            // Drop late frames
            msg_bool_arg!(output, sel!("setAlwaysDiscardsLateVideoFrames:"), 1i8);

            // ── Create delegate ─────────────────────────────────
            let delegate_cls = create_delegate_class();
            let delegate = msg!(msg!(delegate_cls, sel!("alloc")), sel!("init"));
            if delegate.is_null() { bail!("delegate alloc/init failed"); }

            object_setInstanceVariable(
                delegate, b"_shared\0".as_ptr() as *const _, shared as Id,
            );

            extern "C" {
                fn dispatch_queue_create(label: *const libc::c_char, attr: Id) -> Id;
            }
            let queue = dispatch_queue_create(
                b"capview.capture\0".as_ptr() as *const _, std::ptr::null_mut(),
            );

            msg!(output, sel!("setSampleBufferDelegate:queue:"), delegate, queue);

            let can_add_out = (msg!(session, sel!("canAddOutput:"), output) as usize) != 0;
            if !can_add_out { bail!("cannot add video output to session"); }
            msg!(session, sel!("addOutput:"), output);

            msg!(session, sel!("commitConfiguration"));

            // ── Select format AFTER commitConfiguration ─────────
            // Setting activeFormat inside the configuration block can be
            // overridden by commitConfiguration.  Do it after commit so
            // the device format and frame duration actually stick.
            let (actual_w, actual_h, actual_fps) =
                select_device_format(dev, width, height, fps, pixfmt);
            ts!("format: {}x{}@{} (requested {}x{}@{})",
                actual_w, actual_h, actual_fps, width, height, fps);

            let actual_pixfmt = match avf_pixfmt {
                K_CV_NV12 => V4L2_PIX_FMT_NV12,
                K_CV_UYVY => V4L2_PIX_FMT_UYVY,
                _ => V4L2_PIX_FMT_NV12,
            };

            ts!("session configured, ready to start");

            Ok(Capture {
                session, delegate, shared,
                pipe_r: fds[0],
                width: actual_w, height: actual_h,
                fps: actual_fps, pixfmt: actual_pixfmt,
                held_buf: std::cell::Cell::new(None),
                linear_buf: std::cell::UnsafeCell::new(Vec::new()),
            })
        }
    }

    pub fn start(&self) -> Result<()> {
        unsafe {
            msg!(self.session, sel!("startRunning"));

            // After startRunning, the "High" session preset may have
            // overridden our frame duration.  Force InputPriority and
            // re-apply the frame rate.
            let inputs = msg!(self.session, sel!("inputs"));
            if !inputs.is_null() && (msg!(inputs, sel!("count")) as usize) > 0 {
                let input = msg_uint!(inputs, sel!("objectAtIndex:"), 0usize);
                let dev = msg!(input, sel!("device"));
                if !dev.is_null() {
                    #[repr(C)]
                    #[derive(Copy, Clone)]
                    struct CMTime { value: i64, timescale: i32, flags: u32, epoch: i64 }
                    type MsgCMTime = unsafe extern "C" fn(Id, Sel) -> CMTime;
                    let get_dur: MsgCMTime = std::mem::transmute(objc_msgSend as *const ());

                    let min_dur = get_dur(dev, sel!("activeVideoMinFrameDuration"));
                    let min_fps = if min_dur.value > 0 { min_dur.timescale as f64 / min_dur.value as f64 } else { 0.0 };

                    // If the session overrode our frame duration, re-apply it
                    // using the exact CMTime from the device's supported ranges
                    // (AVFoundation rejects non-exact CMTime values).
                    let target_fps = self.fps as f64;
                    if (min_fps - target_fps).abs() > 1.0 {
                        eprintln!("capture: session overrode fps to {:.1}, re-applying {}fps", min_fps, self.fps);

                        let msg_f64: unsafe extern "C" fn(Id, Sel) -> f64 =
                            std::mem::transmute(objc_msgSend as *const ());

                        // Find the exact CMTime from the active format's ranges
                        let active_fmt = msg!(dev, sel!("activeFormat"));
                        let ranges = if !active_fmt.is_null() {
                            msg!(active_fmt, sel!("videoSupportedFrameRateRanges"))
                        } else { std::ptr::null_mut() };
                        let rc = if !ranges.is_null() { msg!(ranges, sel!("count")) as usize } else { 0 };

                        let mut best_dur: Option<CMTime> = None;
                        let mut best_dist: f64 = f64::MAX;
                        for i in 0..rc {
                            let r = msg_uint!(ranges, sel!("objectAtIndex:"), i);
                            if r.is_null() { continue; }
                            let rm = msg_f64(r, sel!("maxFrameRate"));
                            let dist = (rm - target_fps).abs();
                            if dist < best_dist {
                                best_dist = dist;
                                best_dur = Some(get_dur(r, sel!("minFrameDuration")));
                            }
                        }

                        if let Some(dur) = best_dur {
                            type MsgSendBoolPtr = unsafe extern "C" fn(Id, Sel, Id) -> u8;
                            type MsgSendVoid = unsafe extern "C" fn(Id, Sel) -> Id;
                            let lock_fn: MsgSendBoolPtr = std::mem::transmute(objc_msgSend as *const ());
                            let unlock_fn: MsgSendVoid = std::mem::transmute(objc_msgSend as *const ());

                            if lock_fn(dev, sel!("lockForConfiguration:"), std::ptr::null_mut()) != 0 {
                                type MsgSendCMTime = unsafe extern "C" fn(Id, Sel, CMTime) -> Id;
                                let set_dur_fn: MsgSendCMTime = std::mem::transmute(objc_msgSend as *const ());
                                set_dur_fn(dev, sel!("setActiveVideoMinFrameDuration:"), dur);
                                set_dur_fn(dev, sel!("setActiveVideoMaxFrameDuration:"), dur);
                                unlock_fn(dev, sel!("unlockForConfiguration"));

                                let new_min = get_dur(dev, sel!("activeVideoMinFrameDuration"));
                                let new_fps = if new_min.value > 0 { new_min.timescale as f64 / new_min.value as f64 } else { 0.0 };
                                eprintln!("capture: re-applied frame duration {}/{} ({:.1}fps)",
                                    new_min.value, new_min.timescale, new_fps);
                            }
                        }
                    }

                    let min_dur = get_dur(dev, sel!("activeVideoMinFrameDuration"));
                    let max_dur = get_dur(dev, sel!("activeVideoMaxFrameDuration"));
                    let min_fps = if min_dur.value > 0 { min_dur.timescale as f64 / min_dur.value as f64 } else { 0.0 };
                    let max_fps = if max_dur.value > 0 { max_dur.timescale as f64 / max_dur.value as f64 } else { 0.0 };
                    eprintln!("capture: active minFrameDuration={}/{} ({:.1}fps) maxFrameDuration={}/{} ({:.1}fps)",
                        min_dur.value, min_dur.timescale, min_fps,
                        max_dur.value, max_dur.timescale, max_fps);

                    let preset = msg!(self.session, sel!("sessionPreset"));
                    if !preset.is_null() {
                        let c = msg!(preset, sel!("UTF8String")) as *const libc::c_char;
                        if !c.is_null() {
                            eprintln!("capture: session preset = {}",
                                std::ffi::CStr::from_ptr(c).to_string_lossy());
                        }
                    }
                }
            }
        }
        eprintln!("capture: streaming {}x{} @ {}fps ({})",
            self.width, self.height, self.fps, format_name(self.pixfmt));
        Ok(())
    }

    pub fn dequeue(&self) -> Result<Option<V4l2Buffer>> {
        let mut sink = [0u8; 64];
        unsafe { libc::read(self.pipe_r, sink.as_mut_ptr() as *mut _, sink.len()); }

        let sb = unsafe { (*self.shared).latest.swap(std::ptr::null_mut(), Ordering::AcqRel) };
        if sb.is_null() { return Ok(None); }
        let seq = unsafe { (*self.shared).seq.load(Ordering::Relaxed) };

        unsafe {
            let pb = CMSampleBufferGetImageBuffer(sb);
            if pb.is_null() { CFRelease(sb); return Ok(None); }

            CVPixelBufferLockBaseAddress(pb, 1);

            let w = CVPixelBufferGetWidth(pb) as usize;
            let h = CVPixelBufferGetHeight(pb) as u32;
            // Log stride info on first frame
            if seq == 1 {
                if CVPixelBufferIsPlanar(pb) {
                    let y_bpr = CVPixelBufferGetBytesPerRowOfPlane(pb, 0);
                    let uv_bpr = CVPixelBufferGetBytesPerRowOfPlane(pb, 1);
                    eprintln!("capture: first frame {}x{} planar y_stride={} uv_stride={} (w={})",
                        w, h, y_bpr, uv_bpr, w);
                } else {
                    let bpr = CVPixelBufferGetBytesPerRow(pb);
                    eprintln!("capture: first frame {}x{} packed stride={} (w*2={})",
                        w, h, bpr, w * 2);
                }
            }
            // Report tightly packed size (buffer_ptr strips stride padding)
            let length = if CVPixelBufferIsPlanar(pb) {
                w * h as usize + w * h as usize / 2
            } else {
                w * 2 * h as usize
            };

            if let Some((old_sb, old_pb)) = self.held_buf.get() {
                CVPixelBufferUnlockBaseAddress(old_pb, 1);
                CFRelease(old_sb);
            }
            self.held_buf.set(Some((sb, pb)));

            Ok(Some(V4l2Buffer {
                index: seq % 8, length: length as u32, sequence: seq,
                _sample_buf: sb, _pixel_buf: pb,
            }))
        }
    }

    pub fn queue(&self, _buf: &mut V4l2Buffer) -> Result<()> { Ok(()) }

    pub fn buffer_ptr(&self, _index: u32) -> *const u8 {
        if let Some((_, pb)) = self.held_buf.get() {
            unsafe {
                if CVPixelBufferIsPlanar(pb) {
                    let w = CVPixelBufferGetWidth(pb);
                    let h = CVPixelBufferGetHeight(pb);
                    let y_ptr = CVPixelBufferGetBaseAddressOfPlane(pb, 0) as *const u8;
                    let y_bpr = CVPixelBufferGetBytesPerRowOfPlane(pb, 0);
                    let uv_ptr = CVPixelBufferGetBaseAddressOfPlane(pb, 1) as *const u8;
                    let uv_bpr = CVPixelBufferGetBytesPerRowOfPlane(pb, 1);

                    // Fast path: tightly packed and contiguous
                    if y_bpr == w && uv_bpr == w && y_ptr.add(w * h) == uv_ptr {
                        return y_ptr;
                    }

                    // Strip stride padding: copy row-by-row to tightly packed buffer
                    let packed_size = w * h + w * (h / 2);
                    let buf = &mut *self.linear_buf.get();
                    buf.resize(packed_size, 0);
                    let dst = buf.as_mut_ptr();
                    for row in 0..h {
                        std::ptr::copy_nonoverlapping(
                            y_ptr.add(row * y_bpr), dst.add(row * w), w);
                    }
                    let uv_dst = dst.add(w * h);
                    for row in 0..(h / 2) {
                        std::ptr::copy_nonoverlapping(
                            uv_ptr.add(row * uv_bpr), uv_dst.add(row * w), w);
                    }
                    buf.as_ptr()
                } else {
                    let w = CVPixelBufferGetWidth(pb);
                    let h = CVPixelBufferGetHeight(pb);
                    let bpr = CVPixelBufferGetBytesPerRow(pb);
                    let packed_w = w * 2; // UYVY: 2 bytes per pixel
                    if bpr == packed_w {
                        return CVPixelBufferGetBaseAddress(pb) as *const u8;
                    }
                    // Strip stride padding for non-planar formats too
                    let buf = &mut *self.linear_buf.get();
                    buf.resize(packed_w * h, 0);
                    let src = CVPixelBufferGetBaseAddress(pb) as *const u8;
                    let dst = buf.as_mut_ptr();
                    for row in 0..h {
                        std::ptr::copy_nonoverlapping(
                            src.add(row * bpr), dst.add(row * packed_w), packed_w);
                    }
                    buf.as_ptr()
                }
            }
        } else { std::ptr::null() }
    }

    pub fn fd(&self) -> RawFd { self.pipe_r }

    /// Check if a frame is available without consuming it.
    pub fn has_frame(&self) -> bool {
        unsafe {
            !(*self.shared).latest.load(Ordering::Acquire).is_null()
        }
    }

    pub fn export_dmabuf_fds(&self) -> Result<Vec<RawFd>> {
        bail!("DMA-BUF not available on macOS")
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        unsafe {
            msg!(self.session, sel!("stopRunning"));
            if let Some((sb, pb)) = self.held_buf.get() {
                CVPixelBufferUnlockBaseAddress(pb, 1);
                CFRelease(sb);
            }
            let pending = (*self.shared).latest.swap(std::ptr::null_mut(), Ordering::AcqRel);
            if !pending.is_null() { CFRelease(pending); }
            (*self.shared).running.store(false, Ordering::Release);
            libc::close(self.pipe_r);
            libc::close((*self.shared).pipe_w);
            msg!(self.session, sel!("release"));
            msg!(self.delegate, sel!("release"));
        }
    }
}

pub fn format_name(pixfmt: u32) -> String {
    match pixfmt {
        V4L2_PIX_FMT_NV12 => "NV12".into(),
        V4L2_PIX_FMT_YUYV => "YUYV".into(),
        V4L2_PIX_FMT_UYVY => "UYVY".into(),
        _ => format!("0x{:08x}", pixfmt),
    }
}

// ── Device discovery ────────────────────────────────────────────────

unsafe fn find_device(device: &str) -> Result<Id> {
    let type_names: &[&str] = &[
        "AVCaptureDeviceTypeExternal",
        "AVCaptureDeviceTypeExternalUnknown",
        "AVCaptureDeviceTypeBuiltInWideAngleCamera",
    ];
    let type_ptrs: Vec<Id> = type_names.iter().map(|n| nsstring(n)).collect();

    let type_arr = msg_ptr_count!(
        cls!("NSArray"), sel!("arrayWithObjects:count:"),
        type_ptrs.as_ptr(), type_ptrs.len()
    );

    let discovery = msg_id_id_i64!(
        cls!("AVCaptureDeviceDiscoverySession"),
        sel!("discoverySessionWithDeviceTypes:mediaType:position:"),
        type_arr,
        nsstring("vide"),
        0i64 // AVCaptureDevicePositionUnspecified
    );

    let devices = if !discovery.is_null() { msg!(discovery, sel!("devices")) }
                  else { std::ptr::null_mut() };
    let count = if !devices.is_null() { msg!(devices, sel!("count")) as usize }
                else { 0 };

    if count == 0 {
        bail!("no capture devices found (is a capture card connected?)");
    }

    // Collect (index, name, device_ptr) and sort by name for stable indexing.
    // AVFoundation's enumeration order is non-deterministic.
    let mut dev_list: Vec<(String, Id)> = Vec::new();
    for i in 0..count {
        let d = msg_uint!(devices, sel!("objectAtIndex:"), i);
        let name_ns = msg!(d, sel!("localizedName"));
        let name = if !name_ns.is_null() {
            let c = msg!(name_ns, sel!("UTF8String")) as *const libc::c_char;
            if !c.is_null() {
                std::ffi::CStr::from_ptr(c).to_string_lossy().to_string()
            } else { format!("device-{}", i) }
        } else { format!("device-{}", i) };
        dev_list.push((name, d));
    }
    dev_list.sort_by(|a, b| a.0.cmp(&b.0));
    let names: Vec<&str> = dev_list.iter().map(|(n, _)| n.as_str()).collect();
    eprintln!("capture: found {} device(s): {}", count, names.join(", "));

    if let Ok(idx) = device.parse::<usize>() {
        if idx >= dev_list.len() { bail!("device index {} out of range (found {})", idx, count); }
        Ok(dev_list[idx].1)
    } else {
        let needle = device.to_lowercase();
        for (name, dev_ptr) in &dev_list {
            if name.to_lowercase().contains(&needle) {
                return Ok(*dev_ptr);
            }
        }
        bail!("device '{}' not found. available: {}", device, names.join(", "));
    }
}

// ── Camera authorization request ─────────────────────────────────────
//
// requestAccessForMediaType:completionHandler: needs an ObjC block.
// We construct one manually using the documented block ABI.

extern "C" {
    static _NSConcreteGlobalBlock: [*const libc::c_void; 0];
}

static AUTH_RESULT: AtomicI32 = AtomicI32::new(-1); // -1 = pending

#[repr(C)]
struct GlobalBlock {
    isa: *const libc::c_void,
    flags: i32,
    reserved: i32,
    invoke: unsafe extern "C" fn(*const GlobalBlock, u8),
    descriptor: *const BlockDescriptor,
}

#[repr(C)]
struct BlockDescriptor {
    reserved: libc::c_ulong,
    size: libc::c_ulong,
}

unsafe impl Sync for GlobalBlock {}

static BLOCK_DESCRIPTOR: BlockDescriptor = BlockDescriptor {
    reserved: 0,
    size: std::mem::size_of::<GlobalBlock>() as libc::c_ulong,
};

unsafe extern "C" fn auth_block_invoke(_block: *const GlobalBlock, granted: u8) {
    AUTH_RESULT.store(granted as i32, Ordering::Release);
}

static AUTH_BLOCK: GlobalBlock = GlobalBlock {
    isa: unsafe { _NSConcreteGlobalBlock.as_ptr() as *const libc::c_void },
    flags: (1 << 28) | (1 << 30), // BLOCK_IS_GLOBAL | BLOCK_HAS_SIGNATURE (required flags)
    reserved: 0,
    invoke: auth_block_invoke,
    descriptor: &BLOCK_DESCRIPTOR,
};

unsafe fn request_camera_access() {
    AUTH_RESULT.store(-1, Ordering::Release);
    let f: unsafe extern "C" fn(Id, Sel, Id, *const GlobalBlock) -> Id =
        std::mem::transmute(objc_msgSend as *const ());
    f(cls!("AVCaptureDevice"),
      sel!("requestAccessForMediaType:completionHandler:"),
      nsstring("vide"),
      &AUTH_BLOCK);
}

// ── Frame rate ──────────────────────────────────────────────────────

/// Iterate the device's formats, pick the best match for the requested
/// resolution / fps / pixel format, set it as activeFormat, and configure
/// the frame rate.  Returns (width, height, fps) actually configured.
unsafe fn select_device_format(
    dev: Id, want_w: u32, want_h: u32, want_fps: u32, want_pixfmt: u32,
) -> (u32, u32, u32) {
    let msg_f64: unsafe extern "C" fn(Id, Sel) -> f64 =
        std::mem::transmute(objc_msgSend as *const ());

    let formats = msg!(dev, sel!("formats"));
    let fmt_count = if !formats.is_null() { msg!(formats, sel!("count")) as usize } else { 0 };

    let want_avf = match want_pixfmt {
        V4L2_PIX_FMT_UYVY => K_CV_UYVY,
        _ => K_CV_NV12,
    };

    // Score each format: prefer exact resolution match, then closest, then fps
    let mut best_fmt: Id = std::ptr::null_mut();
    let mut best_score: i64 = i64::MIN;
    let mut best_w: u32 = 0;
    let mut best_h: u32 = 0;
    let mut best_max_fps: f64 = 0.0;

    for i in 0..fmt_count {
        let fmt = msg_uint!(formats, sel!("objectAtIndex:"), i);
        if fmt.is_null() { continue; }

        let desc = msg!(fmt, sel!("formatDescription"));
        if desc.is_null() { continue; }

        let dims = CMVideoFormatDescriptionGetDimensions(desc);
        let fw = dims.width as u32;
        let fh = dims.height as u32;
        let subtype = CMFormatDescriptionGetMediaSubType(desc);

        // Check frame rate support
        let ranges = msg!(fmt, sel!("videoSupportedFrameRateRanges"));
        let rc = if !ranges.is_null() { msg!(ranges, sel!("count")) as usize } else { 0 };
        let mut max_fps: f64 = 0.0;
        for j in 0..rc {
            let r = msg_uint!(ranges, sel!("objectAtIndex:"), j);
            if !r.is_null() {
                let rm = msg_f64(r, sel!("maxFrameRate"));
                if rm > max_fps { max_fps = rm; }
            }
        }

        let subtype_bytes = subtype.to_be_bytes();
        let subtype_str = std::str::from_utf8(&subtype_bytes).unwrap_or("????");
        eprintln!("  fmt[{}]: {}x{} {} max={:.1}fps",
            i, fw, fh, subtype_str, max_fps);

        // Scoring: heavily prefer exact resolution, then pixel format, then fps
        let mut score: i64 = 0;
        if fw == want_w && fh == want_h {
            score += 10_000_000;
        } else {
            // Penalize by pixel count distance
            let diff = (fw as i64 * fh as i64 - want_w as i64 * want_h as i64).abs();
            score -= diff;
        }
        if subtype == want_avf { score += 1_000_000; }
        // Bonus for supporting the requested fps
        if max_fps >= want_fps as f64 - 0.5 { score += 500_000; }
        // Tiebreak: prefer higher max fps
        score += max_fps as i64;

        if score > best_score {
            best_score = score;
            best_fmt = fmt;
            best_w = fw;
            best_h = fh;
            best_max_fps = max_fps;
        }
    }

    // If we found a better format than the current active, set it.
    // Format and frame rate MUST be set in the same lock transaction
    // (Apple docs: set format first, then min/max frame duration).
    if !best_fmt.is_null() && best_w > 0 {
        type MsgSendBoolPtr = unsafe extern "C" fn(Id, Sel, Id) -> u8;
        type MsgSendVoid = unsafe extern "C" fn(Id, Sel) -> Id;
        type MsgSendSetFmt = unsafe extern "C" fn(Id, Sel, Id) -> Id;
        let lock_fn: MsgSendBoolPtr = std::mem::transmute(objc_msgSend as *const ());
        let unlock_fn: MsgSendVoid = std::mem::transmute(objc_msgSend as *const ());
        let set_fmt_fn: MsgSendSetFmt = std::mem::transmute(objc_msgSend as *const ());

        let actual_fps;
        if lock_fn(dev, sel!("lockForConfiguration:"), std::ptr::null_mut()) != 0 {
            set_fmt_fn(dev, sel!("setActiveFormat:"), best_fmt);
            actual_fps = set_frame_duration(dev, want_fps, &msg_f64);
            unlock_fn(dev, sel!("unlockForConfiguration"));
        } else {
            eprintln!("capture: lockForConfiguration failed");
            actual_fps = want_fps;
        }
        if best_w != want_w || best_h != want_h {
            eprintln!("capture: closest format {}x{} (requested {}x{})",
                      best_w, best_h, want_w, want_h);
        }
        (best_w, best_h, actual_fps)
    } else {
        // Fallback: just read whatever activeFormat is and set fps
        let active_fmt = msg!(dev, sel!("activeFormat"));
        let (aw, ah) = if !active_fmt.is_null() {
            let desc = msg!(active_fmt, sel!("formatDescription"));
            if !desc.is_null() {
                let dims = CMVideoFormatDescriptionGetDimensions(desc);
                (dims.width as u32, dims.height as u32)
            } else { (want_w, want_h) }
        } else { (want_w, want_h) };
        type MsgSendBoolPtr = unsafe extern "C" fn(Id, Sel, Id) -> u8;
        type MsgSendVoid = unsafe extern "C" fn(Id, Sel) -> Id;
        let lock_fn: MsgSendBoolPtr = std::mem::transmute(objc_msgSend as *const ());
        let unlock_fn: MsgSendVoid = std::mem::transmute(objc_msgSend as *const ());
        let actual_fps = if lock_fn(dev, sel!("lockForConfiguration:"), std::ptr::null_mut()) != 0 {
            let fps = set_frame_duration(dev, want_fps, &msg_f64);
            unlock_fn(dev, sel!("unlockForConfiguration"));
            fps
        } else { want_fps };
        (aw, ah, actual_fps)
    }
}

/// Set frame duration on a device that is ALREADY locked for configuration.
/// Caller must hold lockForConfiguration.
unsafe fn set_frame_duration(
    dev: Id, fps: u32,
    msg_f64: &(unsafe extern "C" fn(Id, Sel) -> f64),
) -> u32 {
    #[repr(C)]
    #[derive(Copy, Clone)]
    struct CMTime { value: i64, timescale: i32, flags: u32, epoch: i64 }

    let active_fmt = msg!(dev, sel!("activeFormat"));
    if active_fmt.is_null() { return fps; }

    let ranges = msg!(active_fmt, sel!("videoSupportedFrameRateRanges"));
    if ranges.is_null() { return fps; }

    let range_count = msg!(ranges, sel!("count")) as usize;
    if range_count == 0 { return fps; }

    type MsgSendCMTimeRet = unsafe extern "C" fn(Id, Sel) -> CMTime;
    let msg_cmtime: MsgSendCMTimeRet = std::mem::transmute(objc_msgSend as *const ());

    // Find the range whose maxFrameRate is closest to our target
    let mut best_range: Id = std::ptr::null_mut();
    let mut best_fps: f64 = 0.0;
    let mut best_dist: f64 = f64::MAX;
    for i in 0..range_count {
        let range = msg_uint!(ranges, sel!("objectAtIndex:"), i);
        if range.is_null() { continue; }
        let range_max = msg_f64(range, sel!("maxFrameRate"));
        let range_min = msg_f64(range, sel!("minFrameRate"));
        eprintln!("  fps range[{}]: {:.1}-{:.1}", i, range_min, range_max);
        let clamped = (fps as f64).min(range_max).max(range_min);
        let dist = (clamped - fps as f64).abs();
        if dist < best_dist {
            best_dist = dist;
            best_fps = clamped;
            best_range = range;
        }
    }

    if best_range.is_null() { return fps; }

    let actual_fps = best_fps.round() as u32;
    if actual_fps != fps {
        eprintln!("capture: device supports {:.1} fps (best match), requested {}",
                  best_fps, fps);
    }

    // Use the range's own minFrameDuration (= max fps) if our target matches
    // the range max, otherwise construct a CMTime for the clamped value.
    let range_max = msg_f64(best_range, sel!("maxFrameRate"));
    let dur = if (best_fps - range_max).abs() < 0.01 {
        msg_cmtime(best_range, sel!("minFrameDuration"))
    } else {
        CMTime { value: 1000, timescale: (best_fps * 1000.0) as i32, flags: 1, epoch: 0 }
    };
    eprintln!("capture: setting frame duration {}/{} ({:.1}fps)",
        dur.value, dur.timescale,
        if dur.value > 0 { dur.timescale as f64 / dur.value as f64 } else { 0.0 });

    if dur.timescale <= 0 || dur.value <= 0 { return actual_fps; }

    type MsgSendCMTime = unsafe extern "C" fn(Id, Sel, CMTime) -> Id;
    let set_dur_fn: MsgSendCMTime = std::mem::transmute(objc_msgSend as *const ());

    set_dur_fn(dev, sel!("setActiveVideoMinFrameDuration:"), dur);
    set_dur_fn(dev, sel!("setActiveVideoMaxFrameDuration:"), dur);

    actual_fps
}

// ── Delegate class ──────────────────────────────────────────────────

fn create_delegate_class() -> Id {
    unsafe {
        static ONCE: std::sync::Once = std::sync::Once::new();
        static mut CLASS: Id = std::ptr::null_mut();

        ONCE.call_once(|| {
            let superclass = cls!("NSObject");
            let c = objc_allocateClassPair(superclass, b"CapviewDelegate\0".as_ptr() as *const _, 0);

            class_addIvar(c, b"_shared\0".as_ptr() as *const _,
                std::mem::size_of::<Id>(), std::mem::align_of::<Id>() as u8,
                b"^v\0".as_ptr() as *const _);

            let proto = objc_getProtocol(
                b"AVCaptureVideoDataOutputSampleBufferDelegate\0".as_ptr() as *const _);
            if !proto.is_null() { class_addProtocol(c, proto); }

            class_addMethod(c,
                sel!("captureOutput:didOutputSampleBuffer:fromConnection:"),
                delegate_callback as *const libc::c_void,
                b"v@:@@@\0".as_ptr() as *const _);

            objc_registerClassPair(c);
            CLASS = c;
        });
        CLASS
    }
}

extern "C" fn delegate_callback(
    this: Id, _sel: Sel, _output: Id, sample_buffer: Id, _connection: Id,
) {
    unsafe {
        if sample_buffer.is_null() { return; }
        let mut shared_ptr: Id = std::ptr::null_mut();
        object_getInstanceVariable(this, b"_shared\0".as_ptr() as *const _, &mut shared_ptr);
        if shared_ptr.is_null() { return; }
        let shared = &*(shared_ptr as *const SharedState);
        if !shared.running.load(Ordering::Relaxed) { return; }

        CFRetain(sample_buffer);
        let old = shared.latest.swap(sample_buffer, Ordering::AcqRel);
        if !old.is_null() { CFRelease(old); }
        shared.seq.fetch_add(1, Ordering::Relaxed);

        let byte = 1u8;
        libc::write(shared.pipe_w, &byte as *const u8 as *const _, 1);
    }
}
