//! Virtual microphone: pipe capture audio into a PulseAudio null-sink so its
//! `.monitor` source appears as a microphone input in Discord / OBS / browsers.
//!
//! Lifecycle:
//!   1. `pactl load-module module-null-sink sink_name=<name>` on start
//!   2. `pa_simple` playback stream targeting that sink
//!   3. Dedicated writer thread reads pooled buffers from a channel
//!   4. `pactl unload-module <id>` on drop
//!
//! The audio capture thread tees into this via `VmicTee::write` — same
//! zero-alloc buffer-pool pattern used by recording / virtual_webcam.

use anyhow::{bail, Context, Result};
use std::ffi::{CStr, CString};
use std::os::raw::c_void;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};

#[allow(non_camel_case_types)]
type pa_simple = libc::c_void;

const PA_STREAM_PLAYBACK: i32 = 1;
const PA_SAMPLE_S16LE: i32 = 3;

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

extern "C" {
    fn pa_simple_new(
        server: *const libc::c_char,
        name: *const libc::c_char,
        dir: i32,
        dev: *const libc::c_char,
        stream_name: *const libc::c_char,
        ss: *const PaSampleSpec,
        map: *const c_void,
        attr: *const PaBufferAttr,
        error: *mut i32,
    ) -> *mut pa_simple;
    fn pa_simple_free(s: *mut pa_simple);
    fn pa_simple_write(
        s: *mut pa_simple,
        data: *const c_void,
        bytes: usize,
        error: *mut i32,
    ) -> i32;
    fn pa_strerror(err: i32) -> *const libc::c_char;
}

fn pa_err(code: i32) -> String {
    unsafe {
        let p = pa_strerror(code);
        if p.is_null() {
            format!("PA error {}", code)
        } else {
            CStr::from_ptr(p).to_string_lossy().to_string()
        }
    }
}

const CHANNEL_DEPTH: usize = 8;
const POOL_SIZE: usize = 12;

/// Clone-able handle the audio thread uses to tee samples into the writer.
/// Senders are `Option<_>` so `VirtualMic::drop` can close them while the
/// audio thread still holds an `Arc<VmicTee>` reference — preventing a
/// join-deadlock on shutdown.
pub struct VmicTee {
    full_tx: Mutex<Option<SyncSender<Vec<u8>>>>,
    free_rx: Mutex<Receiver<Vec<u8>>>,
    free_tx: Mutex<Option<SyncSender<Vec<u8>>>>,
    dropped: Arc<AtomicUsize>,
}

impl VmicTee {
    /// Non-blocking tee of audio samples. Returns false if the writer has
    /// exited (device gone); true otherwise (including silent sample drops).
    pub fn write(&self, data: &[u8]) -> bool {
        let mut buf = {
            let rx = self.free_rx.lock().unwrap();
            match rx.try_recv() {
                Ok(b) => b,
                Err(_) => {
                    self.dropped.fetch_add(1, Ordering::Relaxed);
                    return true;
                }
            }
        };
        buf.clear();
        buf.extend_from_slice(data);

        let tx_guard = self.full_tx.lock().unwrap();
        let tx = match tx_guard.as_ref() {
            Some(t) => t,
            None => return false,
        };
        match tx.try_send(buf) {
            Ok(()) => true,
            Err(TrySendError::Full(b)) => {
                drop(tx_guard);
                self.dropped.fetch_add(1, Ordering::Relaxed);
                if let Some(free_tx) = self.free_tx.lock().unwrap().as_ref() {
                    let _ = free_tx.try_send(b);
                }
                true
            }
            Err(TrySendError::Disconnected(_)) => false,
        }
    }
}

pub struct VirtualMic {
    tee: Arc<VmicTee>,
    writer_thread: Option<std::thread::JoinHandle<()>>,
    module_id: Option<u32>,
    sink_name: String,
}

impl VirtualMic {
    pub fn start(sink_name: &str) -> Result<Self> {
        // Drop any stale capview sinks left behind by a previous crash so
        // we don't accumulate duplicates each launch.
        unload_sinks_matching(sink_name);

        let module_id = load_null_sink(sink_name)?;

        // Fixed spec matches passthrough_loop's capture side: 48kHz/2ch/S16LE.
        // 20ms buffer period → 3840 bytes.
        let ss = PaSampleSpec {
            format: PA_SAMPLE_S16LE,
            rate: 48000,
            channels: 2,
        };
        let buf_len: usize = 48 * 2 * 2 * 20;
        let attr = PaBufferAttr {
            maxlength: u32::MAX,
            tlength: buf_len as u32,
            prebuf: 0,
            minreq: u32::MAX,
            fragsize: u32::MAX,
        };

        let app_name = CString::new("capview").unwrap();
        let stream_name = CString::new("virtual mic").unwrap();
        let sink_c = CString::new(sink_name).unwrap();
        let mut err: i32 = 0;

        let pa = unsafe {
            pa_simple_new(
                std::ptr::null(),
                app_name.as_ptr(),
                PA_STREAM_PLAYBACK,
                sink_c.as_ptr(),
                stream_name.as_ptr(),
                &ss,
                std::ptr::null(),
                &attr,
                &mut err,
            )
        };
        if pa.is_null() {
            let msg = pa_err(err);
            unload_null_sink(module_id);
            bail!("pa_simple_new(virtual mic): {}", msg);
        }

        let (full_tx, full_rx) = mpsc::sync_channel::<Vec<u8>>(CHANNEL_DEPTH);
        let (free_tx, free_rx) = mpsc::sync_channel::<Vec<u8>>(POOL_SIZE);
        for _ in 0..POOL_SIZE {
            free_tx
                .send(vec![0u8; buf_len])
                .expect("pool pre-populate fits channel capacity");
        }

        // Pass the pa_simple ptr to the writer thread as usize so the
        // closure is unambiguously Send without a wrapper newtype.
        let pa_addr = pa as usize;
        let free_tx_writer = free_tx.clone();
        let writer_thread = std::thread::spawn(move || {
            crate::priority::avoid_render_core();
            let pa = pa_addr as *mut pa_simple;
            let mut err: i32 = 0;
            while let Ok(buf) = full_rx.recv() {
                let w = unsafe {
                    pa_simple_write(pa, buf.as_ptr() as *const c_void, buf.len(), &mut err)
                };
                if w < 0 {
                    eprintln!("virtual_mic: pa_simple_write failed: {}", pa_err(err));
                    break;
                }
                let _ = free_tx_writer.try_send(buf);
            }
            unsafe { pa_simple_free(pa); }
        });

        let tee = Arc::new(VmicTee {
            full_tx: Mutex::new(Some(full_tx)),
            free_rx: Mutex::new(free_rx),
            free_tx: Mutex::new(Some(free_tx)),
            dropped: Arc::new(AtomicUsize::new(0)),
        });

        Ok(VirtualMic {
            tee,
            writer_thread: Some(writer_thread),
            module_id: Some(module_id),
            sink_name: sink_name.to_string(),
        })
    }

    pub fn tee(&self) -> Arc<VmicTee> {
        self.tee.clone()
    }

    pub fn monitor_source(&self) -> String {
        format!("{}.monitor", self.sink_name)
    }
}

impl Drop for VirtualMic {
    fn drop(&mut self) {
        // Closing senders → writer thread's recv() returns Err → pa_simple_free.
        // Using Mutex<Option<_>> lets us do this even if the audio thread still
        // holds an Arc<VmicTee> clone.
        *self.tee.full_tx.lock().unwrap() = None;
        *self.tee.free_tx.lock().unwrap() = None;

        if let Some(t) = self.writer_thread.take() {
            let _ = t.join();
        }
        if let Some(id) = self.module_id.take() {
            unload_null_sink(id);
        }
    }
}

fn load_null_sink(sink_name: &str) -> Result<u32> {
    let out = Command::new("pactl")
        .args([
            "load-module",
            "module-null-sink",
            &format!("sink_name={}", sink_name),
            "sink_properties=device.description=capview",
        ])
        .output()
        .context("failed to run pactl (is PulseAudio/PipeWire installed?)")?;
    if !out.status.success() {
        bail!(
            "pactl load-module failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    let id_str = String::from_utf8_lossy(&out.stdout);
    let id: u32 = id_str
        .trim()
        .parse()
        .with_context(|| format!("pactl returned unexpected output: {:?}", id_str))?;
    Ok(id)
}

fn unload_null_sink(module_id: u32) {
    let _ = Command::new("pactl")
        .args(["unload-module", &module_id.to_string()])
        .output();
}

/// Best-effort: unload any existing null-sink with the same sink_name, so a
/// previous capview crash doesn't leave a stale sink behind.
fn unload_sinks_matching(sink_name: &str) {
    let out = match Command::new("pactl")
        .args(["list", "modules", "short"])
        .output()
    {
        Ok(o) => o,
        Err(_) => return,
    };
    let needle = format!("sink_name={}", sink_name);
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        if line.contains("module-null-sink") && line.contains(&needle) {
            if let Some(id) = line.split_whitespace().next() {
                let _ = Command::new("pactl")
                    .args(["unload-module", id])
                    .output();
            }
        }
    }
}
