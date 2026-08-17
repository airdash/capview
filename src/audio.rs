//! Audio passthrough: pipe a PulseAudio/PipeWire source to the default sink.
//!
//! Uses libpulse-simple for the read/write loop (blocking, runs on a
//! dedicated thread) and libpulse's introspect API to resolve a source
//! by partial description match.

use anyhow::{bail, Result};
use std::ffi::{CStr, CString};
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use crate::virtual_mic::VmicTee;

// ── PulseAudio FFI bindings (just what we need) ─────────────────────

#[allow(non_camel_case_types)]
type pa_simple = libc::c_void;
#[allow(non_camel_case_types)]
type pa_context = libc::c_void;
#[allow(non_camel_case_types)]
type pa_mainloop = libc::c_void;
#[allow(non_camel_case_types)]
type pa_mainloop_api = libc::c_void;
#[allow(non_camel_case_types)]
type pa_operation = libc::c_void;

const PA_STREAM_RECORD: i32 = 2;
const PA_STREAM_PLAYBACK: i32 = 1;
const PA_SAMPLE_S16LE: i32 = 3;

const PA_CONTEXT_READY: i32 = 4;
const PA_CONTEXT_FAILED: i32 = 5;
const PA_CONTEXT_TERMINATED: i32 = 6;

const PA_OPERATION_RUNNING: i32 = 0;

#[repr(C)]
#[derive(Clone, Copy)]
struct PaSampleSpec {
    format: i32,
    rate: u32,
    channels: u8,
}

#[repr(C)]
struct PaBufferAttr {
    maxlength: u32,
    tlength: u32,
    prebuf: u32,
    minreq: u32,
    fragsize: u32,
}

// pa_source_info is huge — we only read name + description via raw pointers
// so we define the fields we need and skip the rest.
#[repr(C)]
struct PaSourceInfo {
    name: *const libc::c_char,
    index: u32,
    description: *const libc::c_char,
    // ... many more fields we don't need
}

extern "C" {
    // simple API
    fn pa_simple_new(
        server: *const libc::c_char,
        name: *const libc::c_char,
        dir: i32,
        dev: *const libc::c_char,
        stream_name: *const libc::c_char,
        ss: *const PaSampleSpec,
        map: *const libc::c_void,
        attr: *const PaBufferAttr,
        error: *mut i32,
    ) -> *mut pa_simple;
    fn pa_simple_free(s: *mut pa_simple);
    fn pa_simple_read(s: *mut pa_simple, data: *mut libc::c_void, bytes: usize, error: *mut i32) -> i32;
    fn pa_simple_write(s: *mut pa_simple, data: *const libc::c_void, bytes: usize, error: *mut i32) -> i32;
    fn pa_simple_flush(s: *mut pa_simple, error: *mut i32) -> i32;
    #[allow(dead_code)]
    fn pa_simple_drain(s: *mut pa_simple, error: *mut i32) -> i32;
    fn pa_simple_get_latency(s: *mut pa_simple, error: *mut i32) -> u64;

    // error
    fn pa_strerror(error: i32) -> *const libc::c_char;

    // mainloop (for introspect)
    fn pa_mainloop_new() -> *mut pa_mainloop;
    fn pa_mainloop_free(m: *mut pa_mainloop);
    fn pa_mainloop_get_api(m: *mut pa_mainloop) -> *mut pa_mainloop_api;
    fn pa_mainloop_iterate(m: *mut pa_mainloop, block: i32, retval: *mut i32) -> i32;

    // context
    fn pa_context_new(api: *mut pa_mainloop_api, name: *const libc::c_char) -> *mut pa_context;
    fn pa_context_connect(c: *mut pa_context, server: *const libc::c_char, flags: i32, api: *const libc::c_void) -> i32;
    fn pa_context_disconnect(c: *mut pa_context);
    fn pa_context_unref(c: *mut pa_context);
    fn pa_context_get_state(c: *mut pa_context) -> i32;

    // introspect
    fn pa_context_get_source_info_list(
        c: *mut pa_context,
        cb: extern "C" fn(*mut pa_context, *const PaSourceInfo, i32, *mut libc::c_void),
        userdata: *mut libc::c_void,
    ) -> *mut pa_operation;
    fn pa_operation_unref(o: *mut pa_operation);
    fn pa_operation_get_state(o: *mut pa_operation) -> i32;
}

// ── Source matching ─────────────────────────────────────────────────

struct SourceMatch {
    needle: String,
    result: Option<String>,
    done: bool,
    debug: bool,
}

extern "C" fn source_info_cb(
    _c: *mut pa_context,
    info: *const PaSourceInfo,
    eol: i32,
    userdata: *mut libc::c_void,
) {
    let data = unsafe { &mut *(userdata as *mut SourceMatch) };

    if eol != 0 {
        if data.debug {
            eprintln!("debug: audio: source enumeration complete (eol)");
        }
        data.done = true;
        return;
    }
    if info.is_null() {
        return;
    }
    if data.result.is_some() {
        return; // already found
    }

    unsafe {
        let desc = if (*info).description.is_null() {
            std::borrow::Cow::Borrowed("")
        } else {
            CStr::from_ptr((*info).description).to_string_lossy()
        };

        let name = if (*info).name.is_null() {
            return;
        } else {
            CStr::from_ptr((*info).name).to_string_lossy()
        };

        if data.debug {
            eprintln!("debug: audio: source [{}] desc=\"{}\"", name, desc);
        }

        let needle_lower = data.needle.to_lowercase();

        // Match against description (case-insensitive)
        if desc.to_lowercase().contains(&needle_lower) {
            if data.debug {
                eprintln!("debug: audio: matched by description");
            }
            data.result = Some(name.to_string());
            return;
        }

        // Also match against source name
        if name.to_lowercase().contains(&needle_lower) {
            if data.debug {
                eprintln!("debug: audio: matched by name");
            }
            data.result = Some(name.to_string());
        }
    }
}

/// Resolve an audio device query to an exact PulseAudio source name.
/// The query is matched case-insensitively against source descriptions and names.
fn resolve_source(query: &str, debug: bool) -> Result<String> {
    if debug { eprintln!("debug: audio: connecting to PulseAudio server"); }

    unsafe {
        let ml = pa_mainloop_new();
        if ml.is_null() {
            bail!("pa_mainloop_new failed");
        }
        let api = pa_mainloop_get_api(ml);
        let app = CString::new("capview").unwrap();
        let ctx = pa_context_new(api, app.as_ptr());
        if ctx.is_null() {
            pa_mainloop_free(ml);
            bail!("pa_context_new failed");
        }

        if pa_context_connect(ctx, ptr::null(), 0, ptr::null()) < 0 {
            pa_context_unref(ctx);
            pa_mainloop_free(ml);
            bail!("pa_context_connect failed");
        }

        if debug { eprintln!("debug: audio: waiting for context ready"); }

        // Wait for context to be ready
        loop {
            pa_mainloop_iterate(ml, 1, ptr::null_mut());
            let state = pa_context_get_state(ctx);
            if debug { eprintln!("debug: audio: context state = {}", state); }
            if state == PA_CONTEXT_READY {
                break;
            }
            if state == PA_CONTEXT_FAILED || state == PA_CONTEXT_TERMINATED {
                pa_context_unref(ctx);
                pa_mainloop_free(ml);
                bail!("PulseAudio context failed (state {})", state);
            }
        }

        if debug { eprintln!("debug: audio: enumerating sources for '{}'", query); }

        let mut match_data = SourceMatch {
            needle: query.to_string(),
            result: None,
            done: false,
            debug,
        };

        let op = pa_context_get_source_info_list(
            ctx,
            source_info_cb,
            &mut match_data as *mut _ as *mut libc::c_void,
        );

        if op.is_null() {
            pa_context_disconnect(ctx);
            pa_context_unref(ctx);
            pa_mainloop_free(ml);
            bail!("pa_context_get_source_info_list returned NULL");
        }

        // Iterate until the operation completes (eol callback fires)
        while !match_data.done {
            if pa_operation_get_state(op) != PA_OPERATION_RUNNING {
                break;
            }
            pa_mainloop_iterate(ml, 1, ptr::null_mut());
        }

        if debug { eprintln!("debug: audio: source enumeration finished"); }

        pa_operation_unref(op);
        pa_context_disconnect(ctx);
        pa_context_unref(ctx);
        pa_mainloop_free(ml);

        match match_data.result {
            Some(name) => {
                if debug { eprintln!("debug: audio: resolved to '{}'", name); }
                Ok(name)
            }
            None => bail!("no PulseAudio source matching '{}'", query),
        }
    }
}

// ── PipeWire link passthrough ────────────────────────────────────────

struct PwLinkState {
    source_ports: Vec<String>,
    sink_ports: Vec<String>,
    linked: bool,
    debug: bool,
}

impl PwLinkState {
    fn start(source_query: &str, debug: bool) -> Result<Self> {
        let source_ports = pw_find_ports(source_query, "output", debug)?;
        if source_ports.is_empty() {
            bail!("no PipeWire output ports matching '{}'", source_query);
        }
        let sink_ports = pw_default_sink_ports(debug)?;
        if sink_ports.is_empty() {
            bail!("no default PipeWire sink ports found");
        }
        if debug {
            eprintln!("debug: pw-link: source ports: {:?}", source_ports);
            eprintln!("debug: pw-link: sink ports: {:?}", sink_ports);
        }
        let mut state = Self { source_ports, sink_ports, linked: false, debug };
        state.link()?;
        Ok(state)
    }

    fn link(&mut self) -> Result<()> {
        if self.linked { return Ok(()); }
        let pairs = self.source_ports.len().min(self.sink_ports.len());
        for i in 0..pairs {
            let out = std::process::Command::new("pw-link")
                .arg(&self.source_ports[i])
                .arg(&self.sink_ports[i])
                .output();
            match out {
                Ok(o) if o.status.success() => {
                    if self.debug {
                        eprintln!("debug: pw-link: linked {} → {}", self.source_ports[i], self.sink_ports[i]);
                    }
                }
                Ok(o) => {
                    let stderr = String::from_utf8_lossy(&o.stderr);
                    if !stderr.contains("already linked") {
                        eprintln!("pw-link: {}", stderr.trim());
                    }
                }
                Err(e) => bail!("pw-link command failed: {}", e),
            }
        }
        self.linked = true;
        eprintln!("audio: pw-link passthrough active ({} channels)", pairs);
        Ok(())
    }

    fn unlink(&mut self) {
        if !self.linked { return; }
        let pairs = self.source_ports.len().min(self.sink_ports.len());
        for i in 0..pairs {
            let _ = std::process::Command::new("pw-link")
                .arg("-d")
                .arg(&self.source_ports[i])
                .arg(&self.sink_ports[i])
                .output();
        }
        self.linked = false;
        if self.debug {
            eprintln!("debug: pw-link: unlinked {} channels", pairs);
        }
    }
}

impl Drop for PwLinkState {
    fn drop(&mut self) {
        self.unlink();
    }
}

fn pw_find_ports(query: &str, direction: &str, debug: bool) -> Result<Vec<String>> {
    let flag = if direction == "output" { "-o" } else { "-i" };
    let out = std::process::Command::new("pw-link")
        .arg(flag)
        .output()
        .map_err(|e| anyhow::anyhow!("pw-link not found: {}", e))?;
    if !out.status.success() {
        bail!("pw-link {} failed", flag);
    }
    let needle = query.to_lowercase();
    let ports: Vec<String> = String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty() && l.to_lowercase().contains(&needle))
        .collect();
    if debug {
        eprintln!("debug: pw-link {}: {} ports matching '{}'", flag, ports.len(), query);
    }
    Ok(ports)
}

fn pw_default_sink_ports(debug: bool) -> Result<Vec<String>> {
    let out = std::process::Command::new("pw-link")
        .arg("-i")
        .output()
        .map_err(|e| anyhow::anyhow!("pw-link not found: {}", e))?;
    if !out.status.success() {
        bail!("pw-link -i failed");
    }
    let all_ports: Vec<String> = String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect();
    // Find the default sink via wpctl
    let default_sink = std::process::Command::new("wpctl")
        .args(["inspect", "@DEFAULT_AUDIO_SINK@"])
        .output()
        .ok()
        .and_then(|o| if o.status.success() {
            let text = String::from_utf8_lossy(&o.stdout).to_string();
            // Look for node.name = "..." line
            text.lines()
                .find(|l| l.contains("node.name"))
                .and_then(|l| l.split('"').nth(1))
                .map(|s| s.to_string())
        } else { None });
    if debug {
        eprintln!("debug: pw-link: default sink node = {:?}", default_sink);
    }
    if let Some(ref sink_name) = default_sink {
        let needle = sink_name.to_lowercase();
        let ports: Vec<String> = all_ports.iter()
            .filter(|p| p.to_lowercase().contains(&needle) && (p.contains("playback_FL") || p.contains("playback_FR")))
            .cloned()
            .collect();
        if !ports.is_empty() { return Ok(ports); }
    }
    // Fallback: look for any playback_FL/FR ports from alsa_output
    let ports: Vec<String> = all_ports.into_iter()
        .filter(|p| p.starts_with("alsa_output") && (p.contains("playback_FL") || p.contains("playback_FR")))
        .collect();
    Ok(ports)
}

// ── Capture mode state ──────────────────────────────────────────────

struct CaptureState {
    running: Arc<AtomicBool>,
    volume: Arc<AtomicI32>,
    muted_flag: Arc<AtomicBool>,
    capture_buf_ms: Arc<AtomicI32>,
    playback_buf_ms: Arc<AtomicI32>,
    xruns: Arc<AtomicU32>,
    vmic_tee: Arc<Mutex<Option<Arc<VmicTee>>>>,
    max_volume: i32,
    thread: Option<thread::JoinHandle<()>>,
    source_name: String,
}

impl CaptureState {
    fn start(source_name: String, max_volume: u32, capture_buf: u32, playback_buf: u32, debug: bool) -> Self {
        let running = Arc::new(AtomicBool::new(true));
        let running2 = running.clone();
        let volume = Arc::new(AtomicI32::new(100));
        let volume2 = volume.clone();
        let muted_flag = Arc::new(AtomicBool::new(false));
        let muted_flag2 = muted_flag.clone();
        let capture_buf_ms = Arc::new(AtomicI32::new(capture_buf as i32));
        let capture_buf_ms2 = capture_buf_ms.clone();
        let playback_buf_ms = Arc::new(AtomicI32::new(playback_buf as i32));
        let playback_buf_ms2 = playback_buf_ms.clone();
        let xruns = Arc::new(AtomicU32::new(0));
        let xruns2 = xruns.clone();
        let vmic_tee: Arc<Mutex<Option<Arc<VmicTee>>>> = Arc::new(Mutex::new(None));
        let vmic_tee2 = vmic_tee.clone();
        let source = source_name.clone();

        let thread = thread::spawn(move || {
            crate::priority::avoid_render_core();
            if let Err(e) = passthrough_loop(&source, &running2, &volume2, &muted_flag2, &capture_buf_ms2, &playback_buf_ms2, &xruns2, &vmic_tee2, debug) {
                eprintln!("audio error: {}", e);
            }
        });

        Self {
            running, volume, muted_flag, capture_buf_ms, playback_buf_ms,
            xruns, vmic_tee, max_volume: max_volume as i32, thread: Some(thread), source_name,
        }
    }

    fn stop(&mut self) {
        self.running.store(false, Ordering::SeqCst);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }

    fn restart(&mut self, debug: bool) {
        self.stop();
        let running = Arc::new(AtomicBool::new(true));
        self.running = running.clone();
        let volume = self.volume.clone();
        let muted_flag = self.muted_flag.clone();
        let capture_buf_ms = self.capture_buf_ms.clone();
        let playback_buf_ms = self.playback_buf_ms.clone();
        let xruns = self.xruns.clone();
        let vmic_tee = self.vmic_tee.clone();
        let source = self.source_name.clone();
        self.thread = Some(thread::spawn(move || {
            crate::priority::avoid_render_core();
            if let Err(e) = passthrough_loop(&source, &running, &volume, &muted_flag, &capture_buf_ms, &playback_buf_ms, &xruns, &vmic_tee, debug) {
                eprintln!("audio error: {}", e);
            }
        }));
    }
}

impl Drop for CaptureState {
    fn drop(&mut self) {
        self.stop();
    }
}

// ── Audio passthrough handle ────────────────────────────────────────

enum AudioInner {
    Capture(CaptureState),
    Passthrough(PwLinkState),
}

pub struct AudioPassthrough {
    inner: AudioInner,
    source_name: String,
    source_query: String,
    max_volume: u32,
    capture_buf: u32,
    playback_buf: u32,
    debug: bool,
    muted: bool,
    mode: crate::config::AudioMode,
}

impl AudioPassthrough {
    pub fn start(source_query: &str, max_volume: u32, capture_buf: u32, playback_buf: u32, mode: crate::config::AudioMode, debug: bool) -> Result<Self> {
        let source_name = resolve_source(source_query, debug)?;
        eprintln!("audio: {} → default sink (mode: {:?})", source_name, mode);

        let inner = match mode {
            crate::config::AudioMode::Capture => {
                AudioInner::Capture(CaptureState::start(source_name.clone(), max_volume, capture_buf, playback_buf, debug))
            }
            crate::config::AudioMode::Passthrough => {
                AudioInner::Passthrough(PwLinkState::start(source_query, debug)?)
            }
        };

        Ok(Self {
            inner, source_name, source_query: source_query.to_string(),
            max_volume, capture_buf, playback_buf, debug, muted: false, mode,
        })
    }

    pub fn source_name(&self) -> &str {
        &self.source_name
    }

    pub fn mode(&self) -> crate::config::AudioMode {
        self.mode
    }

    /// Switch audio mode. Stops the current backend and starts the new one.
    pub fn set_mode(&mut self, mode: crate::config::AudioMode) -> Result<()> {
        if mode == self.mode { return Ok(()); }
        self.stop_inner();
        self.mode = mode;
        self.inner = match mode {
            crate::config::AudioMode::Capture => {
                AudioInner::Capture(CaptureState::start(self.source_name.clone(), self.max_volume, self.capture_buf, self.playback_buf, self.debug))
            }
            crate::config::AudioMode::Passthrough => {
                AudioInner::Passthrough(PwLinkState::start(&self.source_query, self.debug)?)
            }
        };
        eprintln!("audio: switched to {:?} mode", mode);
        Ok(())
    }

    pub fn volume_up(&self) -> i32 {
        match &self.inner {
            AudioInner::Capture(c) => {
                let v = (c.volume.load(Ordering::Relaxed) + 5).min(c.max_volume);
                c.volume.store(v, Ordering::Relaxed);
                v
            }
            AudioInner::Passthrough(_) => 100,
        }
    }

    pub fn volume_down(&self) -> i32 {
        match &self.inner {
            AudioInner::Capture(c) => {
                let v = (c.volume.load(Ordering::Relaxed) - 5).max(0);
                c.volume.store(v, Ordering::Relaxed);
                v
            }
            AudioInner::Passthrough(_) => 100,
        }
    }

    pub fn volume(&self) -> i32 {
        match &self.inner {
            AudioInner::Capture(c) => c.volume.load(Ordering::Relaxed),
            AudioInner::Passthrough(_) => 100,
        }
    }

    pub fn set_volume(&self, v: i32) {
        if let AudioInner::Capture(c) = &self.inner {
            c.volume.store(v.max(0).min(c.max_volume), Ordering::Relaxed);
        }
    }

    pub fn is_muted(&self) -> bool {
        self.muted
    }

    pub fn set_muted(&mut self, muted: bool) {
        self.muted = muted;
        match &mut self.inner {
            AudioInner::Capture(c) => c.muted_flag.store(muted, Ordering::SeqCst),
            AudioInner::Passthrough(p) => {
                if muted { p.unlink(); } else { let _ = p.link(); }
            }
        }
    }

    pub fn toggle_mute(&mut self) -> bool {
        self.muted = !self.muted;
        self.set_muted(self.muted);
        self.muted
    }

    pub fn capture_buf_ms(&self) -> i32 {
        match &self.inner {
            AudioInner::Capture(c) => c.capture_buf_ms.load(Ordering::Relaxed),
            AudioInner::Passthrough(_) => 0,
        }
    }

    pub fn playback_buf_ms(&self) -> i32 {
        match &self.inner {
            AudioInner::Capture(c) => c.playback_buf_ms.load(Ordering::Relaxed),
            AudioInner::Passthrough(_) => 0,
        }
    }

    pub fn xruns(&self) -> u32 {
        match &self.inner {
            AudioInner::Capture(c) => c.xruns.load(Ordering::Relaxed),
            AudioInner::Passthrough(_) => 0,
        }
    }

    pub fn set_buffers(&mut self, capture_ms: i32, playback_ms: i32) {
        self.capture_buf = capture_ms as u32;
        self.playback_buf = playback_ms as u32;
        if let AudioInner::Capture(c) = &mut self.inner {
            c.capture_buf_ms.store(capture_ms, Ordering::SeqCst);
            c.playback_buf_ms.store(playback_ms, Ordering::SeqCst);
            c.restart(self.debug);
        }
    }

    fn stop_inner(&mut self) {
        match &mut self.inner {
            AudioInner::Capture(c) => c.stop(),
            AudioInner::Passthrough(p) => p.unlink(),
        }
    }

    pub fn stop(&mut self) {
        self.stop_inner();
    }

    /// Set the virtual mic tee. `None` disables teeing. Only active in
    /// Capture mode — Passthrough (PipeWire link) bypasses this thread
    /// entirely, so the caller should check `mode()` first.
    pub fn set_virtual_mic(&self, tee: Option<Arc<VmicTee>>) {
        if let AudioInner::Capture(c) = &self.inner {
            *c.vmic_tee.lock().unwrap() = tee;
        }
    }
}

impl Drop for AudioPassthrough {
    fn drop(&mut self) {
        self.stop();
    }
}

fn pa_error(code: i32) -> String {
    unsafe {
        let p = pa_strerror(code);
        if p.is_null() {
            format!("PA error {}", code)
        } else {
            CStr::from_ptr(p).to_string_lossy().to_string()
        }
    }
}

/// Check if audio samples look like garbage from a dirty USB device.
/// Returns true if the audio appears corrupt.
fn probe_is_corrupt(buf: &[u8], debug: bool) -> bool {
    let samples: &[i16] = unsafe {
        std::slice::from_raw_parts(buf.as_ptr() as *const i16, buf.len() / 2)
    };
    if samples.is_empty() { return false; }

    // Count how many samples are at extreme values (clipping)
    let mut clipped = 0u32;
    let mut zero = 0u32;
    let mut prev = samples[0];
    let mut stuck = 0u32; // consecutive identical non-zero samples
    let mut max_stuck = 0u32;

    for &s in samples {
        if s == i16::MAX || s == i16::MIN { clipped += 1; }
        if s == 0 { zero += 1; }
        if s == prev && s != 0 {
            stuck += 1;
            max_stuck = max_stuck.max(stuck);
        } else {
            stuck = 0;
        }
        prev = s;
    }

    let n = samples.len() as u32;
    let clip_pct = clipped * 100 / n;
    let zero_pct = zero * 100 / n;
    let stuck_pct = max_stuck * 100 / n;

    // Heuristics: >30% clipped, >50% of longest run is same value, etc.
    let corrupt = clip_pct > 30 || stuck_pct > 50;

    if debug {
        eprintln!("debug: audio probe: {} samples, clipped={}%, zero={}%, max_stuck={}% → {}",
            n, clip_pct, zero_pct, stuck_pct, if corrupt { "CORRUPT" } else { "ok" });
    }

    corrupt
}

const MAX_PROBE_RETRIES: u32 = 5;
const PROBE_FRAMES: usize = 4;
const PROBE_RETRY_MS: u64 = 500;

fn passthrough_loop(source_name: &str, running: &AtomicBool, volume: &AtomicI32, muted: &AtomicBool, capture_buf_ms: &AtomicI32, playback_buf_ms: &AtomicI32, xruns: &AtomicU32, vmic_tee: &Mutex<Option<Arc<VmicTee>>>, debug: bool) -> Result<()> {
    use std::time::Instant;

    // 48kHz, 2ch, 16-bit = 192 bytes per ms
    const BYTES_PER_MS: u32 = 192;

    let cap_ms = capture_buf_ms.load(Ordering::Relaxed) as u32;
    let play_ms = playback_buf_ms.load(Ordering::Relaxed) as u32;
    let fragsize = cap_ms * BYTES_PER_MS;
    let tlength = play_ms * BYTES_PER_MS;

    let ss = PaSampleSpec {
        format: PA_SAMPLE_S16LE,
        rate: 48000,
        channels: 2,
    };

    let app_name = CString::new("capview").unwrap();
    let rec_stream = CString::new("capture input").unwrap();
    let play_stream = CString::new("capture output").unwrap();
    let c_source = CString::new(source_name)?;

    let mut err: i32 = 0;
    let buf_len = fragsize as usize;
    let mut buf = vec![0u8; buf_len];

    // Probe loop: open record stream, read a few frames, check for corruption.
    // If the USB device is in a dirty state, close and retry after a delay.
    let mut rec: *mut pa_simple = ptr::null_mut();
    for attempt in 0..=MAX_PROBE_RETRIES {
        if !running.load(Ordering::Relaxed) { bail!("stopped during probe"); }

        let rec_attr = PaBufferAttr {
            maxlength: u32::MAX,
            tlength: u32::MAX,
            prebuf: u32::MAX,
            minreq: u32::MAX,
            fragsize,
        };

        if debug { eprintln!("debug: audio: opening record stream from '{}' (attempt {}) ...", source_name, attempt + 1); }
        let t0 = Instant::now();

        rec = unsafe {
            pa_simple_new(
                ptr::null(),
                app_name.as_ptr(),
                PA_STREAM_RECORD,
                c_source.as_ptr(),
                rec_stream.as_ptr(),
                &ss,
                ptr::null(),
                &rec_attr,
                &mut err,
            )
        };
        if rec.is_null() {
            if attempt < MAX_PROBE_RETRIES {
                eprintln!("audio: source open failed (attempt {}), retrying in {}ms...", attempt + 1, PROBE_RETRY_MS);
                thread::sleep(std::time::Duration::from_millis(PROBE_RETRY_MS));
                continue;
            }
            bail!("pa_simple_new(record): {}", pa_error(err));
        }

        if debug { eprintln!("debug: audio: record stream opened ({:.0}ms)", t0.elapsed().as_secs_f64() * 1000.0); }

        // Read probe frames and check for corruption
        let mut corrupt = false;
        for i in 0..PROBE_FRAMES {
            let r = unsafe { pa_simple_read(rec, buf.as_mut_ptr() as *mut _, buf.len(), &mut err) };
            if r < 0 {
                eprintln!("audio: probe read failed: {}", pa_error(err));
                corrupt = true;
                break;
            }
            // Skip first frame (may contain transition artifacts), check the rest
            if i > 0 && probe_is_corrupt(&buf, debug) {
                corrupt = true;
                break;
            }
        }

        if !corrupt {
            if attempt > 0 {
                eprintln!("audio: source clean after {} retries", attempt);
            }
            break;
        }

        // Dirty — close and retry
        unsafe {
            pa_simple_flush(rec, &mut err);
            pa_simple_free(rec);
        }
        rec = ptr::null_mut();

        if attempt < MAX_PROBE_RETRIES {
            eprintln!("audio: source appears dirty (attempt {}), retrying in {}ms...", attempt + 1, PROBE_RETRY_MS);
            thread::sleep(std::time::Duration::from_millis(PROBE_RETRY_MS));
        } else {
            eprintln!("audio: source still dirty after {} retries, proceeding anyway", MAX_PROBE_RETRIES + 1);
        }
    }

    if rec.is_null() {
        bail!("pa_simple_new(record): failed all probe attempts");
    }

    if debug { eprintln!("debug: audio: opening playback stream to default sink ..."); }
    let t1 = Instant::now();

    let play_attr = PaBufferAttr {
        maxlength: u32::MAX,
        tlength,
        prebuf: 0,
        minreq: u32::MAX,
        fragsize: u32::MAX,
    };

    let play = unsafe {
        pa_simple_new(
            ptr::null(),
            app_name.as_ptr(),
            PA_STREAM_PLAYBACK,
            ptr::null(), // default sink
            play_stream.as_ptr(),
            &ss,
            ptr::null(),
            &play_attr,
            &mut err,
        )
    };
    if play.is_null() {
        unsafe { pa_simple_free(rec); }
        bail!("pa_simple_new(playback): {}", pa_error(err));
    }

    if debug {
        eprintln!("debug: audio: playback stream opened ({:.0}ms)", t1.elapsed().as_secs_f64() * 1000.0);
        let mut lat_err: i32 = 0;
        let rec_lat = unsafe { pa_simple_get_latency(rec, &mut lat_err) };
        let play_lat = unsafe { pa_simple_get_latency(play, &mut lat_err) };
        eprintln!("debug: audio: record latency={}µs playback latency={}µs", rec_lat, play_lat);
        eprintln!("debug: audio: starting read/write loop (buf={}B = {}ms)", fragsize, cap_ms);
    }

    let silence = vec![0u8; buf_len];
    let xrun_threshold_us = (cap_ms as u64) * 2000; // 2x buffer period in microseconds
    let mut last_read = Instant::now();

    while running.load(Ordering::Relaxed) {
        // Always read from the source to keep PA happy and drain its buffer.
        let r = unsafe { pa_simple_read(rec, buf.as_mut_ptr() as *mut _, buf.len(), &mut err) };
        if r < 0 {
            eprintln!("audio read error: {}", pa_error(err));
            break;
        }
        let now = Instant::now();
        let elapsed_us = (now - last_read).as_micros() as u64;
        last_read = now;
        if elapsed_us > xrun_threshold_us {
            xruns.fetch_add(1, Ordering::Relaxed);
        }

        // When muted, write silence instead of captured audio.
        if muted.load(Ordering::Relaxed) {
            let w = unsafe { pa_simple_write(play, silence.as_ptr() as *const _, silence.len(), &mut err) };
            if w < 0 {
                eprintln!("audio write error: {}", pa_error(err));
                break;
            }
            continue;
        }

        // Apply volume scaling to S16LE samples
        let vol = volume.load(Ordering::Relaxed);
        if vol != 100 {
            let samples: &mut [i16] = unsafe {
                std::slice::from_raw_parts_mut(
                    buf.as_mut_ptr() as *mut i16,
                    buf.len() / 2,
                )
            };
            for s in samples.iter_mut() {
                let v = (*s as i32 * vol) / 100;
                *s = v.clamp(-32768, 32767) as i16;
            }
        }

        // Tee into the virtual mic (if enabled). Skipped when muted — the
        // muted branch above `continue`s before reaching here.
        if let Ok(guard) = vmic_tee.try_lock() {
            if let Some(tee) = guard.as_ref() {
                if !tee.write(&buf) {
                    // Writer thread exited; best-effort clear so we don't
                    // keep hitting Disconnected on every iteration.
                    drop(guard);
                    if let Ok(mut g) = vmic_tee.try_lock() {
                        *g = None;
                    }
                }
            }
        }

        let w = unsafe { pa_simple_write(play, buf.as_ptr() as *const _, buf.len(), &mut err) };
        if w < 0 {
            eprintln!("audio write error: {}", pa_error(err));
            break;
        }
    }

    if debug {
        eprintln!("debug: audio: loop ended, cleaning up");
    }

    unsafe {
        let mut flush_err: i32 = 0;
        pa_simple_flush(play, &mut flush_err);
        pa_simple_free(play);
        pa_simple_free(rec);
    }

    Ok(())
}
