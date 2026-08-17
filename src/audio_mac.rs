//! CoreAudio passthrough: pipe capture device audio to default output.
//!
//! Replaces PulseAudio on macOS.  Uses AudioUnit (AUHAL) for input
//! capture and DefaultOutput for playback.

use anyhow::{bail, Result};
use std::ffi::CStr;
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::Arc;
use std::thread;

// ── CoreAudio FFI ──────────────────────────────────────────────────────

type AudioObjectID = u32;
type AudioDeviceID = u32;
type AudioUnit = *mut libc::c_void;
type OSStatus = i32;

const K_AUDIO_OBJECT_SYSTEM_OBJECT: AudioObjectID = 1;
const K_AUDIO_HARDWARE_PROPERTY_DEVICES: u32 = 0x64657623; // 'dev#'
const K_AUDIO_DEVICE_PROPERTY_DEVICE_NAME: u32 = 0x6e616d65; // 'name'
const K_AUDIO_DEVICE_PROPERTY_STREAMS: u32 = 0x73746d23; // 'stm#'
const K_AUDIO_OBJECT_PROPERTY_SCOPE_GLOBAL: u32 = 0x676c6f62; // 'glob'
const K_AUDIO_OBJECT_PROPERTY_SCOPE_INPUT: u32 = 0x696e7074; // 'inpt'
const K_AUDIO_OBJECT_PROPERTY_ELEMENT_MAIN: u32 = 0;

const K_AUDIO_UNIT_TYPE_OUTPUT: u32 = 0x61756f75; // 'auou'
const K_AUDIO_UNIT_SUBTYPE_HAL_OUTPUT: u32 = 0x6168616c; // 'ahal'
const K_AUDIO_UNIT_SUBTYPE_DEFAULT_OUTPUT: u32 = 0x64656620; // 'def '
const K_AUDIO_UNIT_MANUFACTURER_APPLE: u32 = 0x6170706c; // 'appl'

const K_AUDIO_UNIT_PROPERTY_STREAM_FORMAT: u32 = 8;
const K_AUDIO_OUTPUT_UNIT_PROPERTY_ENABLE_IO: u32 = 2003;
const K_AUDIO_OUTPUT_UNIT_PROPERTY_CURRENT_DEVICE: u32 = 2000;
const K_AUDIO_UNIT_PROPERTY_SET_RENDER_CALLBACK: u32 = 23;

const K_AUDIO_FORMAT_LINEAR_PCM: u32 = 0x6c70636d; // 'lpcm'
const K_AUDIO_FORMAT_FLAG_IS_SIGNED_INTEGER: u32 = 4;
const K_AUDIO_FORMAT_FLAG_IS_PACKED: u32 = 8;

#[repr(C)]
struct AudioObjectPropertyAddress {
    selector: u32,
    scope: u32,
    element: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AudioStreamBasicDescription {
    sample_rate: f64,
    format_id: u32,
    format_flags: u32,
    bytes_per_packet: u32,
    frames_per_packet: u32,
    bytes_per_frame: u32,
    channels_per_frame: u32,
    bits_per_channel: u32,
    reserved: u32,
}

#[repr(C)]
struct AudioComponentDescription {
    component_type: u32,
    component_sub_type: u32,
    component_manufacturer: u32,
    component_flags: u32,
    component_flags_mask: u32,
}

#[repr(C)]
struct AURenderCallbackStruct {
    input_proc: extern "C" fn(
        *mut libc::c_void, *mut u32, *const AudioTimeStamp, u32, u32, *mut AudioBufferList,
    ) -> OSStatus,
    input_proc_ref_con: *mut libc::c_void,
}

#[repr(C)]
struct AudioTimeStamp {
    sample_time: f64,
    host_time: u64,
    rate_scalar: f64,
    word_clock_time: u64,
    smpte_time: [u8; 24],
    flags: u32,
    reserved: u32,
}

#[repr(C)]
struct AudioBuffer {
    number_channels: u32,
    data_byte_size: u32,
    data: *mut libc::c_void,
}

#[repr(C)]
struct AudioBufferList {
    number_buffers: u32,
    buffers: [AudioBuffer; 1],
}

extern "C" {
    fn AudioObjectGetPropertyDataSize(
        id: AudioObjectID, address: *const AudioObjectPropertyAddress,
        qualifier_size: u32, qualifier: *const libc::c_void, out_size: *mut u32,
    ) -> OSStatus;
    fn AudioObjectGetPropertyData(
        id: AudioObjectID, address: *const AudioObjectPropertyAddress,
        qualifier_size: u32, qualifier: *const libc::c_void,
        io_size: *mut u32, out_data: *mut libc::c_void,
    ) -> OSStatus;
    fn AudioComponentFindNext(
        component: *mut libc::c_void, desc: *const AudioComponentDescription,
    ) -> *mut libc::c_void;
    fn AudioComponentInstanceNew(component: *mut libc::c_void, out: *mut AudioUnit) -> OSStatus;
    fn AudioComponentInstanceDispose(unit: AudioUnit) -> OSStatus;
    fn AudioUnitSetProperty(
        unit: AudioUnit, id: u32, scope: u32, element: u32,
        data: *const libc::c_void, size: u32,
    ) -> OSStatus;
    fn AudioUnitInitialize(unit: AudioUnit) -> OSStatus;
    fn AudioUnitUninitialize(unit: AudioUnit) -> OSStatus;
    fn AudioOutputUnitStart(unit: AudioUnit) -> OSStatus;
    fn AudioOutputUnitStop(unit: AudioUnit) -> OSStatus;
    fn AudioUnitRender(
        unit: AudioUnit, io_action_flags: *mut u32, in_time_stamp: *const AudioTimeStamp,
        bus: u32, frames: u32, io_data: *mut AudioBufferList,
    ) -> OSStatus;
}

// ── Device discovery ───────────────────────────────────────────────────

fn get_device_name(id: AudioDeviceID) -> String {
    let addr = AudioObjectPropertyAddress {
        selector: K_AUDIO_DEVICE_PROPERTY_DEVICE_NAME,
        scope: K_AUDIO_OBJECT_PROPERTY_SCOPE_GLOBAL,
        element: K_AUDIO_OBJECT_PROPERTY_ELEMENT_MAIN,
    };
    let mut buf = [0u8; 256];
    let mut size = buf.len() as u32;
    let st = unsafe { AudioObjectGetPropertyData(id, &addr, 0, ptr::null(), &mut size, buf.as_mut_ptr() as *mut _) };
    if st == 0 {
        unsafe { CStr::from_ptr(buf.as_ptr() as *const _) }.to_string_lossy().into_owned()
    } else {
        format!("device_{}", id)
    }
}

fn has_input_streams(id: AudioDeviceID) -> bool {
    let addr = AudioObjectPropertyAddress {
        selector: K_AUDIO_DEVICE_PROPERTY_STREAMS,
        scope: K_AUDIO_OBJECT_PROPERTY_SCOPE_INPUT,
        element: K_AUDIO_OBJECT_PROPERTY_ELEMENT_MAIN,
    };
    let mut size: u32 = 0;
    let st = unsafe { AudioObjectGetPropertyDataSize(id, &addr, 0, ptr::null(), &mut size) };
    st == 0 && size > 0
}

fn resolve_source(query: &str, debug: bool) -> Result<(AudioDeviceID, String)> {
    let addr = AudioObjectPropertyAddress {
        selector: K_AUDIO_HARDWARE_PROPERTY_DEVICES,
        scope: K_AUDIO_OBJECT_PROPERTY_SCOPE_GLOBAL,
        element: K_AUDIO_OBJECT_PROPERTY_ELEMENT_MAIN,
    };
    let mut size: u32 = 0;
    let st = unsafe { AudioObjectGetPropertyDataSize(K_AUDIO_OBJECT_SYSTEM_OBJECT, &addr, 0, ptr::null(), &mut size) };
    if st != 0 { bail!("AudioObjectGetPropertyDataSize failed: {}", st); }

    let count = size as usize / std::mem::size_of::<AudioDeviceID>();
    let mut devices = vec![0u32; count];
    let st = unsafe { AudioObjectGetPropertyData(K_AUDIO_OBJECT_SYSTEM_OBJECT, &addr, 0, ptr::null(), &mut size, devices.as_mut_ptr() as *mut _) };
    if st != 0 { bail!("AudioObjectGetPropertyData failed: {}", st); }

    let needle = query.to_lowercase();
    for &dev_id in &devices {
        if !has_input_streams(dev_id) { continue; }
        let name = get_device_name(dev_id);
        if debug { eprintln!("debug: audio: input device [{}] \"{}\"", dev_id, name); }
        if name.to_lowercase().contains(&needle) {
            if debug { eprintln!("debug: audio: matched"); }
            return Ok((dev_id, name));
        }
    }
    bail!("no CoreAudio input device matching '{}'", query)
}

// ── Public API (matching audio.rs) ─────────────────────────────────────

struct RenderCtx {
    input_unit: AudioUnit,
    volume: Arc<AtomicI32>,
}
unsafe impl Send for RenderCtx {}

pub struct AudioPassthrough {
    running: Arc<AtomicBool>,
    volume: Arc<AtomicI32>,
    max_volume: i32,
    thread: Option<thread::JoinHandle<()>>,
    source_name: String,
    debug: bool,
    muted: bool,
}

impl AudioPassthrough {
    pub fn start(source_query: &str, max_volume: u32, _capture_buf: u32, _playback_buf: u32, debug: bool) -> Result<Self> {
        let (device_id, source_name) = resolve_source(source_query, debug)?;
        eprintln!("audio: {} → default output", source_name);

        let running = Arc::new(AtomicBool::new(true));
        let running2 = running.clone();
        let volume = Arc::new(AtomicI32::new(100));
        let volume2 = volume.clone();

        let thread = thread::spawn(move || {
            crate::priority::avoid_render_core();
            if let Err(e) = passthrough_loop(device_id, &running2, &volume2, debug) {
                eprintln!("audio error: {}", e);
            }
        });

        Ok(Self { running, volume, max_volume: max_volume as i32,
            thread: Some(thread), source_name, debug, muted: false })
    }

    pub fn source_name(&self) -> &str { &self.source_name }

    pub fn volume_up(&self) -> i32 {
        let v = (self.volume.load(Ordering::Relaxed) + 5).min(self.max_volume);
        self.volume.store(v, Ordering::Relaxed);
        v
    }

    pub fn volume_down(&self) -> i32 {
        let v = (self.volume.load(Ordering::Relaxed) - 5).max(0);
        self.volume.store(v, Ordering::Relaxed);
        v
    }

    pub fn volume(&self) -> i32 { self.volume.load(Ordering::Relaxed) }

    pub fn set_volume(&self, v: i32) {
        self.volume.store(v.max(0).min(self.max_volume), Ordering::Relaxed);
    }

    pub fn is_muted(&self) -> bool { self.muted }

    pub fn set_muted(&mut self, muted: bool) { self.muted = muted; }

    pub fn capture_buf_ms(&self) -> i32 { 5 }

    pub fn playback_buf_ms(&self) -> i32 { 10 }

    pub fn set_buffers(&mut self, _capture_ms: i32, _playback_ms: i32) {}

    pub fn toggle_mute(&mut self) -> bool {
        if self.muted {
            self.muted = false;
            false
        } else {
            self.stop_thread();
            self.muted = true;
            true
        }
    }

    fn stop_thread(&mut self) {
        self.running.store(false, Ordering::SeqCst);
        if let Some(t) = self.thread.take() {
            let deadline = std::time::Instant::now() + std::time::Duration::from_millis(500);
            loop {
                if t.is_finished() { let _ = t.join(); return; }
                if std::time::Instant::now() >= deadline { drop(t); return; }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        }
    }

    pub fn stop(&mut self) { self.stop_thread(); }
}

impl Drop for AudioPassthrough {
    fn drop(&mut self) { self.stop(); }
}

// ── Passthrough loop ───────────────────────────────────────────────────

fn passthrough_loop(device_id: AudioDeviceID, running: &AtomicBool, volume: &AtomicI32, debug: bool) -> Result<()> {
    let format = AudioStreamBasicDescription {
        sample_rate: 48000.0, format_id: K_AUDIO_FORMAT_LINEAR_PCM,
        format_flags: K_AUDIO_FORMAT_FLAG_IS_SIGNED_INTEGER | K_AUDIO_FORMAT_FLAG_IS_PACKED,
        bytes_per_packet: 4, frames_per_packet: 1, bytes_per_frame: 4,
        channels_per_frame: 2, bits_per_channel: 16, reserved: 0,
    };

    unsafe {
        // Input unit (AUHAL)
        let mut input_unit: AudioUnit = ptr::null_mut();
        let desc = AudioComponentDescription {
            component_type: K_AUDIO_UNIT_TYPE_OUTPUT,
            component_sub_type: K_AUDIO_UNIT_SUBTYPE_HAL_OUTPUT,
            component_manufacturer: K_AUDIO_UNIT_MANUFACTURER_APPLE,
            component_flags: 0, component_flags_mask: 0,
        };
        let comp = AudioComponentFindNext(ptr::null_mut(), &desc);
        if comp.is_null() { bail!("no HAL output component"); }
        let st = AudioComponentInstanceNew(comp, &mut input_unit);
        if st != 0 { bail!("AudioComponentInstanceNew(input): {}", st); }

        // Enable input on bus 1
        let enable: u32 = 1;
        let st = AudioUnitSetProperty(input_unit, K_AUDIO_OUTPUT_UNIT_PROPERTY_ENABLE_IO,
            1, 1, &enable as *const _ as *const _, 4);
        if st != 0 { AudioComponentInstanceDispose(input_unit); bail!("enable input: {}", st); }

        // Disable output on bus 0
        let disable: u32 = 0;
        AudioUnitSetProperty(input_unit, K_AUDIO_OUTPUT_UNIT_PROPERTY_ENABLE_IO,
            2, 0, &disable as *const _ as *const _, 4);

        // Set device
        let st = AudioUnitSetProperty(input_unit, K_AUDIO_OUTPUT_UNIT_PROPERTY_CURRENT_DEVICE,
            0, 0, &device_id as *const _ as *const _, 4);
        if st != 0 { AudioComponentInstanceDispose(input_unit); bail!("set input device: {}", st); }

        // Set format
        AudioUnitSetProperty(input_unit, K_AUDIO_UNIT_PROPERTY_STREAM_FORMAT,
            2, 1, &format as *const _ as *const _,
            std::mem::size_of::<AudioStreamBasicDescription>() as u32);

        let st = AudioUnitInitialize(input_unit);
        if st != 0 { AudioComponentInstanceDispose(input_unit); bail!("init input: {}", st); }

        // Output unit
        let mut output_unit: AudioUnit = ptr::null_mut();
        let desc = AudioComponentDescription {
            component_type: K_AUDIO_UNIT_TYPE_OUTPUT,
            component_sub_type: K_AUDIO_UNIT_SUBTYPE_DEFAULT_OUTPUT,
            component_manufacturer: K_AUDIO_UNIT_MANUFACTURER_APPLE,
            component_flags: 0, component_flags_mask: 0,
        };
        let comp = AudioComponentFindNext(ptr::null_mut(), &desc);
        if comp.is_null() { bail!("no default output component"); }
        let st = AudioComponentInstanceNew(comp, &mut output_unit);
        if st != 0 { bail!("AudioComponentInstanceNew(output): {}", st); }

        AudioUnitSetProperty(output_unit, K_AUDIO_UNIT_PROPERTY_STREAM_FORMAT,
            1, 0, &format as *const _ as *const _,
            std::mem::size_of::<AudioStreamBasicDescription>() as u32);

        // Render callback
        let ctx = Box::into_raw(Box::new(RenderCtx {
            input_unit,
            volume: Arc::new(AtomicI32::new(volume.load(Ordering::Relaxed))),
        }));

        let cb = AURenderCallbackStruct {
            input_proc: render_callback,
            input_proc_ref_con: ctx as *mut _,
        };
        let st = AudioUnitSetProperty(output_unit, K_AUDIO_UNIT_PROPERTY_SET_RENDER_CALLBACK,
            1, 0, &cb as *const _ as *const _,
            std::mem::size_of::<AURenderCallbackStruct>() as u32);
        if st != 0 {
            let _ = Box::from_raw(ctx);
            AudioUnitUninitialize(input_unit);
            AudioComponentInstanceDispose(input_unit);
            AudioComponentInstanceDispose(output_unit);
            bail!("set render callback: {}", st);
        }

        let st = AudioUnitInitialize(output_unit);
        if st != 0 {
            let _ = Box::from_raw(ctx);
            AudioUnitUninitialize(input_unit);
            AudioComponentInstanceDispose(input_unit);
            AudioComponentInstanceDispose(output_unit);
            bail!("init output: {}", st);
        }

        AudioOutputUnitStart(input_unit);
        let st = AudioOutputUnitStart(output_unit);
        if st != 0 { bail!("start output: {}", st); }

        if debug { eprintln!("debug: audio: passthrough running (48kHz 2ch S16LE)"); }

        while running.load(Ordering::Relaxed) {
            (*ctx).volume.store(volume.load(Ordering::Relaxed), Ordering::Relaxed);
            thread::sleep(std::time::Duration::from_millis(50));
        }

        AudioOutputUnitStop(output_unit);
        AudioOutputUnitStop(input_unit);
        AudioUnitUninitialize(output_unit);
        AudioUnitUninitialize(input_unit);
        AudioComponentInstanceDispose(output_unit);
        AudioComponentInstanceDispose(input_unit);
        let _ = Box::from_raw(ctx);
    }
    Ok(())
}

extern "C" fn render_callback(
    in_ref_con: *mut libc::c_void, io_action_flags: *mut u32,
    in_time_stamp: *const AudioTimeStamp, _bus: u32,
    in_number_frames: u32, io_data: *mut AudioBufferList,
) -> OSStatus {
    let ctx = unsafe { &*(in_ref_con as *const RenderCtx) };
    if io_data.is_null() { return 0; }

    let st = unsafe { AudioUnitRender(ctx.input_unit, io_action_flags, in_time_stamp,
        1, in_number_frames, io_data) };
    if st != 0 { return st; }

    let vol = ctx.volume.load(Ordering::Relaxed);
    if vol != 100 {
        unsafe {
            let list = &*io_data;
            for i in 0..list.number_buffers as usize {
                let buf = &*(&list.buffers as *const AudioBuffer).add(i);
                let samples = std::slice::from_raw_parts_mut(
                    buf.data as *mut i16, buf.data_byte_size as usize / 2);
                for s in samples.iter_mut() {
                    *s = ((*s as i32 * vol) / 100).clamp(-32768, 32767) as i16;
                }
            }
        }
    }
    0
}
