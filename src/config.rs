use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::PathBuf;

use crate::capture::{V4L2_PIX_FMT_NV12, V4L2_PIX_FMT_YUYV, V4L2_PIX_FMT_UYVY, V4L2_PIX_FMT_XRGB32, V4L2_PIX_FMT_P010, V4L2_PIX_FMT_MJPEG};

const MAX_BUFFERS: u32 = 8;

const CONFIG_DIR: &str = "capview";
const CONFIG_FILE: &str = "capview.conf";

const DEFAULT_CONFIG: &str = "\
# capview defaults — edit to taste, CLI flags always override.
#
# Boolean values: true/yes/1 to enable, anything else to disable.
#
# Top-level keys are global defaults. Named [profile] sections override them.
# Usage: capview --profile hdmi
#
# Profile example:
# [hdmi]
# device       = /dev/video2
# fps          = 60
# audio_device = capture card
#
# audio_device matches by substring against PulseAudio/PipeWire source
# descriptions (case-insensitive). Use 'pactl list sources short' to see
# available sources.
#
# max_volume caps PageUp volume adjustment (default 100). Raise above
# 100 only if the source is unusually quiet — values over 100 amplify
# digitally and may clip.
# max_volume   = 100

# device       = /dev/video0
# width        = 1920
# height       = 1080
# fps          = 60
# format       = nv12
# buffers      = 2
# window       = 960x540
# audio_device =
# record_resolution = native
# target_fps = 0
vsync      = false
smooth     = false
fullscreen = false
quiet      = false
daemonize  = false

# Priority optimizations — applied at startup to reduce latency.
# Set to 'all' (default) or 'none', or pick individual flags as a
# comma-separated list:
#   timer_slack    - set kernel timer slack to 1ns (default 50us)
#   realtime (rt)  - request SCHED_RR real-time scheduling (needs rtprio ulimit)
#   cpu_pin        - pin render thread to the most idle core; background threads
#                    and other capview instances avoid it automatically
#   mlock          - lock current + future pages into RAM (no page faults ever)
#   idle_inhibit   - inhibit screensaver / idle timeout via D-Bus
#   no_compositor  - suspend KWin compositing (resumes on exit)
#   io_prio        - set best-effort I/O priority 0 (highest without root)
#   sig_mask       - block unnecessary signals on the render thread
# If any option causes problems on your system, disable it here.
# priority = all

# Virtual webcam: expose capture frames as a v4l2loopback device for
# Discord / OBS / browsers. Requires v4l2loopback installed:
#   sudo modprobe v4l2loopback video_nr=10 card_label=capview exclusive_caps=1
# virtual_webcam        = false
# virtual_webcam_device = /dev/video10

# Virtual mic: expose capture audio as a PulseAudio null-sink; apps see
# <sink_name>.monitor as a selectable microphone input.
# virtual_mic      = false
# virtual_mic_sink = capview_mic
";

pub struct Config {
    pub device: String,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub buffers: u32,
    pub pixfmt: u32,
    pub screenshot_dir: String,
    pub win_w: Option<u32>,
    pub win_h: Option<u32>,
    pub vsync: bool,
    pub smooth: bool,
    pub fullscreen: bool,
    pub quiet: bool,
    pub daemonize: bool,
    pub audio_device: Option<String>,
    pub max_volume: u32,
    pub record_resolution: RecordResolution,
    pub screenshot_format: ScreenshotFormat,
    pub jpeg_quality: u32,
    pub renderer: RendererBackend,
    pub plugins: Vec<String>,
    pub stream_port: u16,
    pub stream_quality: u32,
    pub stream_client_ip: [u8; 4],
    pub stream_client_port: u16,
    pub strip_cols: u32,
    pub strip_rows: u32,
    /// Target display FPS for frame generation (0 = auto: 2× capture fps).
    pub target_fps: u32,
    /// Frame generation mode (off, extrapolate, interpolate, rife).
    pub framegen_mode: String,
    /// Frame generation quality (fast, balanced, quality).
    pub framegen_quality: String,
    /// Scaling algorithm for OpenGL renderer.
    pub scale_mode: String,
    /// Sharpness for scaling shaders (0 = soft, 10 = sharpest).
    pub sharpness: u32,
    /// Vulkan present mode preference.
    pub vk_present_mode: VkPresentMode,
    /// Aspect mode: Preserve (letterbox), Stretch (fill, distort), Zoom (fill, crop).
    pub aspect_mode: AspectMode,
    /// Audio capture buffer size in ms.
    pub audio_capture_buf: u32,
    /// Audio playback buffer size in ms.
    pub audio_playback_buf: u32,
    /// Audio mode: "capture" (software passthrough) or "passthrough" (PipeWire link).
    pub audio_mode: AudioMode,
    /// Brightness multiplier (0–200, maps to 0.0–2.0). Default 100 = 1.0.
    pub brightness: u32,
    /// Contrast multiplier (0–200, maps to 0.0–2.0). Default 100 = 1.0.
    pub contrast: u32,
    /// Gamma (10–300, maps to 0.1–3.0). Default 100 = 1.0 (linear).
    pub gamma: u32,
    /// FPS display mode: "off", "simple", "verbose".
    pub fps_display: String,
    /// Pause capture when window is minimized.
    pub pause_on_minimize: bool,
    /// Pause capture when window loses focus (goes to background).
    pub pause_on_background: bool,
    /// OSD background opacity (0–100, maps to 0–255 alpha). Default 63 (~160/255).
    pub osd_opacity: u32,
    /// Priority optimizations bitmask.
    pub priority: PriorityFlags,
    /// Expose capture frames as a v4l2loopback virtual webcam.
    pub virtual_webcam: bool,
    /// v4l2loopback device path for virtual webcam output.
    pub virtual_webcam_device: String,
    /// Expose capture audio as a PulseAudio null-sink (.monitor = virtual mic).
    pub virtual_mic: bool,
    /// Sink name for the null-sink. The mic shows up as `<name>.monitor`.
    pub virtual_mic_sink: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AudioMode {
    Capture,
    Passthrough,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AspectMode {
    Preserve,
    Stretch,
    Zoom,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RendererBackend {
    Sdl,
    OpenGl,
    Vulkan,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VkPresentMode {
    Mailbox,
    Immediate,
    Fifo,
}

impl VkPresentMode {
    pub fn label(self) -> &'static str {
        match self {
            VkPresentMode::Mailbox => "Mailbox",
            VkPresentMode::Immediate => "Immediate",
            VkPresentMode::Fifo => "VSync (FIFO)",
        }
    }
}

/// Which priority optimizations to apply at startup.
/// Config key: `priority = all` (default) or comma-separated list of flags.
/// Use `priority = none` to disable all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PriorityFlags(pub u32);

impl PriorityFlags {
    pub const TIMER_SLACK:   PriorityFlags = PriorityFlags(1 << 0);
    pub const REALTIME:      PriorityFlags = PriorityFlags(1 << 1);
    pub const CPU_PIN:       PriorityFlags = PriorityFlags(1 << 2);
    pub const MLOCK:         PriorityFlags = PriorityFlags(1 << 3);
    pub const IDLE_INHIBIT:  PriorityFlags = PriorityFlags(1 << 4);
    pub const NO_COMPOSITOR: PriorityFlags = PriorityFlags(1 << 5);
    pub const IO_PRIO:       PriorityFlags = PriorityFlags(1 << 6);
    pub const SIG_MASK:      PriorityFlags = PriorityFlags(1 << 7);
    pub const ALL: PriorityFlags = PriorityFlags(0xFF);
    pub const NONE: PriorityFlags = PriorityFlags(0);

    pub fn has(self, flag: PriorityFlags) -> bool { self.0 & flag.0 != 0 }

    pub fn parse(s: &str) -> Self {
        let s = s.trim().to_lowercase();
        match s.as_str() {
            "all" | "true" | "1" | "yes" => Self::ALL,
            "none" | "false" | "0" | "no" | "off" => Self::NONE,
            _ => {
                let mut bits = 0u32;
                for token in s.split(',') {
                    match token.trim() {
                        "timer_slack" => bits |= Self::TIMER_SLACK.0,
                        "realtime" | "rt" => bits |= Self::REALTIME.0,
                        "cpu_pin" | "affinity" => bits |= Self::CPU_PIN.0,
                        "mlock" => bits |= Self::MLOCK.0,
                        "idle_inhibit" | "idle" => bits |= Self::IDLE_INHIBIT.0,
                        "no_compositor" | "no_comp" => bits |= Self::NO_COMPOSITOR.0,
                        "io_prio" | "ioprio" => bits |= Self::IO_PRIO.0,
                        "sig_mask" | "signals" => bits |= Self::SIG_MASK.0,
                        other => eprintln!("capview: unknown priority flag '{}'", other),
                    }
                }
                PriorityFlags(bits)
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecordResolution {
    Native,
    Window,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScreenshotFormat {
    Png,
    Jpeg,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            #[cfg(target_os = "linux")]
            device: "/dev/video0".into(),
            #[cfg(target_os = "macos")]
            device: "0".into(),
            width: 1920,
            height: 1080,
            fps: 60,
            buffers: 2,
            pixfmt: V4L2_PIX_FMT_NV12,
            screenshot_dir: "~/Pictures".into(),
            win_w: None,
            win_h: None,
            vsync: false,
            smooth: false,
            fullscreen: false,
            quiet: false,
            daemonize: false,
            audio_device: None,
            max_volume: 100,
            record_resolution: RecordResolution::Native,
            screenshot_format: ScreenshotFormat::Png,
            jpeg_quality: 90,
            renderer: RendererBackend::Sdl,
            plugins: Vec::new(),
            stream_port: 9000,
            stream_quality: 80,
            stream_client_ip: [192, 168, 1, 1],
            stream_client_port: 9000,
            strip_cols: 1,
            strip_rows: 6,
            target_fps: 0,
            framegen_mode: String::new(),
            framegen_quality: String::new(),
            scale_mode: String::new(),
            sharpness: 5,
            vk_present_mode: VkPresentMode::Mailbox,
            aspect_mode: AspectMode::Preserve,
            audio_capture_buf: 5,
            audio_playback_buf: 10,
            audio_mode: AudioMode::Capture,
            brightness: 100,
            contrast: 100,
            gamma: 100,
            fps_display: "off".to_string(),
            pause_on_minimize: true,
            pause_on_background: false,
            osd_opacity: 63,
            priority: PriorityFlags::ALL,
            virtual_webcam: false,
            virtual_webcam_device: "/dev/video10".to_string(),
            virtual_mic: false,
            virtual_mic_sink: "capview_mic".to_string(),
        }
    }
}

fn parse_bool(val: &str) -> bool {
    matches!(val, "true" | "1" | "yes")
}

impl Config {
    pub fn load(profile: Option<&str>) -> Result<Self> {
        let path = config_path();
        ensure_config(&path);

        if !path.exists() {
            return Ok(Self::default());
        }

        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;

        // Parse into sections: "" (global) and named "[section]" blocks
        let mut sections: HashMap<String, Vec<(String, String, u32)>> = HashMap::new();
        let mut current_section = String::new();
        let mut lineno = 0u32;

        for line in text.lines() {
            lineno += 1;
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            // Section header
            if line.starts_with('[') && line.ends_with(']') {
                current_section = line[1..line.len()-1].trim().to_lowercase();
                continue;
            }
            if let Some((k, v)) = line.split_once('=') {
                let v = v.trim();
                let v = v.strip_prefix('"').and_then(|s| s.strip_suffix('"'))
                    .or_else(|| v.strip_prefix('\'').and_then(|s| s.strip_suffix('\'')))
                    .unwrap_or(v);
                sections.entry(current_section.clone()).or_default().push(
                    (k.trim().to_lowercase(), v.to_string(), lineno)
                );
            } else {
                eprintln!("capview.conf:{}: malformed line", lineno);
            }
        }

        // Start with defaults
        let mut cfg = Self::default();

        // Apply global (top-level) settings
        if let Some(globals) = sections.get("") {
            for (key, val, lineno) in globals {
                apply_config_key(&mut cfg, key, val, *lineno)?;
            }
        }

        // Apply profile settings on top (if requested)
        if let Some(name) = profile {
            let lname = name.to_lowercase();
            if let Some(profile_kvs) = sections.get(&lname) {
                for (key, val, lineno) in profile_kvs {
                    apply_config_key(&mut cfg, key, val, *lineno)?;
                }
            } else {
                anyhow::bail!("profile '{}' not found in {}", name, path.display());
            }
        }

        Ok(cfg)
    }

    /// List available profile names from the config file.
    pub fn list_profiles() -> Vec<String> {
        let path = config_path();
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };
        let mut profiles = Vec::new();
        for line in text.lines() {
            let line = line.trim();
            if line.starts_with('[') && line.ends_with(']') {
                let name = line[1..line.len()-1].trim().to_string();
                if !name.is_empty() {
                    profiles.push(name);
                }
            }
        }
        profiles
    }

    /// Expand ~ in screenshot_dir.
    #[allow(dead_code)]
    pub fn screenshot_path(&self) -> PathBuf {
        let expanded = if self.screenshot_dir.starts_with("~/") {
            if let Ok(home) = std::env::var("HOME") {
                format!("{}{}", home, &self.screenshot_dir[1..])
            } else {
                self.screenshot_dir.clone()
            }
        } else {
            self.screenshot_dir.clone()
        };
        PathBuf::from(expanded)
    }
}

fn config_path() -> PathBuf {
    if let Ok(xdg) = std::env::var("XDG_CONFIG_HOME") {
        PathBuf::from(xdg).join(CONFIG_DIR).join(CONFIG_FILE)
    } else if let Ok(home) = std::env::var("HOME") {
        PathBuf::from(&home).join(".config").join(CONFIG_DIR).join(CONFIG_FILE)
    } else {
        PathBuf::from(CONFIG_FILE)
    }
}

/// Create config directory and default config file if they don't exist.
fn ensure_config(path: &PathBuf) {
    if let Some(dir) = path.parent() {
        if !dir.exists() {
            if let Err(e) = std::fs::create_dir_all(dir) {
                eprintln!("capview: could not create {}: {}", dir.display(), e);
                return;
            }
        }
    }
    if !path.exists() {
        match std::fs::write(path, DEFAULT_CONFIG) {
            Ok(()) => eprintln!("capview: created {}", path.display()),
            Err(e) => eprintln!("capview: could not create {}: {}", path.display(), e),
        }
    }
}

fn parse_pixfmt(s: &str, lineno: u32) -> Result<u32> {
    match s.to_lowercase().as_str() {
        "nv12" => Ok(V4L2_PIX_FMT_NV12),
        "yuyv" | "yuy2" => Ok(V4L2_PIX_FMT_YUYV),
        "uyvy" => Ok(V4L2_PIX_FMT_UYVY),
        "xrgb" | "rgbx" | "bgrx" => Ok(V4L2_PIX_FMT_XRGB32),
        "p010" => Ok(V4L2_PIX_FMT_P010),
        "mjpeg" | "mjpg" => Ok(V4L2_PIX_FMT_MJPEG),
        _ => {
            eprintln!("capview.conf:{}: unknown format '{}'", lineno, s);
            Ok(V4L2_PIX_FMT_NV12)
        }
    }
}

fn apply_config_key(cfg: &mut Config, key: &str, val: &str, lineno: u32) -> Result<()> {
    match key {
        "device" => cfg.device = val.to_string(),
        "width" => cfg.width = val.parse().context("invalid width")?,
        "height" => cfg.height = val.parse().context("invalid height")?,
        "fps" => cfg.fps = val.parse().context("invalid fps")?,
        "buffers" => {
            let n: u32 = val.parse().context("invalid buffers")?;
            cfg.buffers = n.clamp(2, MAX_BUFFERS);
        }
        "format" => cfg.pixfmt = parse_pixfmt(val, lineno)?,
        "screenshot_dir" => cfg.screenshot_dir = val.to_string(),
        "window" => {
            if let Some((w, h)) = val.split_once('x') {
                match (w.trim().parse::<u32>(), h.trim().parse::<u32>()) {
                    (Ok(w), Ok(h)) => { cfg.win_w = Some(w); cfg.win_h = Some(h); }
                    _ => eprintln!("capview.conf:{}: bad window size '{}' (want WxH)", lineno, val),
                }
            } else {
                eprintln!("capview.conf:{}: bad window size '{}' (want WxH)", lineno, val);
            }
        }
        "vsync" => cfg.vsync = parse_bool(val),
        "smooth" => cfg.smooth = parse_bool(val),
        "fullscreen" => cfg.fullscreen = parse_bool(val),
        "quiet" => cfg.quiet = parse_bool(val),
        "daemonize" => cfg.daemonize = parse_bool(val),
        "audio_device" => {
            let v = val.trim();
            cfg.audio_device = if v.is_empty() { None } else { Some(v.to_string()) };
        }
        "max_volume" => {
            let n: u32 = val.parse().context("invalid max_volume")?;
            cfg.max_volume = n.max(100);  // must be at least 100
        }
        "record_resolution" | "record_res" => {
            match val.to_lowercase().as_str() {
                "native" | "full" => cfg.record_resolution = RecordResolution::Native,
                "window" | "win" => cfg.record_resolution = RecordResolution::Window,
                _ => eprintln!("capview.conf:{}: unknown record_resolution '{}' (native|window)", lineno, val),
            }
        }
        "screenshot_format" | "screenshot_fmt" => {
            match val.to_lowercase().as_str() {
                "png" => cfg.screenshot_format = ScreenshotFormat::Png,
                "jpeg" | "jpg" => cfg.screenshot_format = ScreenshotFormat::Jpeg,
                _ => eprintln!("capview.conf:{}: unknown screenshot_format '{}' (png|jpeg)", lineno, val),
            }
        }
        "jpeg_quality" => {
            let n: u32 = val.parse().context("invalid jpeg_quality")?;
            cfg.jpeg_quality = n.clamp(1, 100);
        }
        "renderer" | "render_backend" => {
            match val.to_lowercase().as_str() {
                "sdl" | "sdl2" => cfg.renderer = RendererBackend::Sdl,
                "gl" | "opengl" => cfg.renderer = RendererBackend::OpenGl,
                "vk" | "vulkan" => cfg.renderer = RendererBackend::Vulkan,
                _ => eprintln!("capview.conf:{}: unknown renderer '{}' (sdl|opengl|vulkan)", lineno, val),
            }
        }
        "vk_present_mode" | "present_mode" => {
            match val.to_lowercase().as_str() {
                "mailbox" => cfg.vk_present_mode = VkPresentMode::Mailbox,
                "immediate" => cfg.vk_present_mode = VkPresentMode::Immediate,
                "fifo" | "vsync" => cfg.vk_present_mode = VkPresentMode::Fifo,
                _ => eprintln!("capview.conf:{}: unknown present mode '{}' (mailbox|immediate|fifo)", lineno, val),
            }
        }
        "plugins" | "plugin" => {
            let v = val.trim();
            if !v.is_empty() {
                cfg.plugins.push(v.to_string());
            }
        }
        "stream_port" => {
            let n: u16 = val.parse().context("invalid stream_port")?;
            cfg.stream_port = n;
        }
        "stream_quality" => {
            let n: u32 = val.parse().context("invalid stream_quality")?;
            cfg.stream_quality = n.clamp(1, 100);
        }
        "stream_client_ip" => {
            let parts: Vec<&str> = val.split('.').collect();
            if parts.len() == 4 {
                if let (Ok(a), Ok(b), Ok(c), Ok(d)) = (
                    parts[0].trim().parse::<u8>(),
                    parts[1].trim().parse::<u8>(),
                    parts[2].trim().parse::<u8>(),
                    parts[3].trim().parse::<u8>(),
                ) {
                    cfg.stream_client_ip = [a, b, c, d];
                } else {
                    eprintln!("capview.conf:{}: bad stream_client_ip '{}'", lineno, val);
                }
            } else {
                eprintln!("capview.conf:{}: bad stream_client_ip '{}' (want A.B.C.D)", lineno, val);
            }
        }
        "stream_client_port" => {
            let n: u16 = val.parse().context("invalid stream_client_port")?;
            cfg.stream_client_port = n;
        }
        "strip_cols" => {
            let n: u32 = val.parse().context("invalid strip_cols")?;
            cfg.strip_cols = n.clamp(1, 6);
        }
        "strip_rows" => {
            let n: u32 = val.parse().context("invalid strip_rows")?;
            cfg.strip_rows = n.clamp(1, 6);
        }
        "target_fps" => {
            let n: u32 = val.parse().context("invalid target_fps")?;
            cfg.target_fps = n.clamp(0, 480);
        }
        "framegen_mode" => {
            cfg.framegen_mode = val.to_string();
        }
        "framegen_quality" => {
            cfg.framegen_quality = val.to_string();
        }
        "scale_mode" => {
            cfg.scale_mode = val.to_string();
        }
        "sharpness" => {
            let n: u32 = val.parse().context("invalid sharpness")?;
            cfg.sharpness = n.clamp(0, 10);
        }
        "stretch" => {
            // Legacy: stretch = true maps to Stretch mode
            if parse_bool(val) { cfg.aspect_mode = AspectMode::Stretch; }
        }
        "aspect_mode" => {
            cfg.aspect_mode = match val.to_lowercase().as_str() {
                "stretch" => AspectMode::Stretch,
                "zoom" => AspectMode::Zoom,
                _ => AspectMode::Preserve,
            };
        }
        "audio_capture_buf" => {
            let n: u32 = val.parse().context("invalid audio_capture_buf")?;
            cfg.audio_capture_buf = n.clamp(1, 100);
        }
        "audio_playback_buf" => {
            let n: u32 = val.parse().context("invalid audio_playback_buf")?;
            cfg.audio_playback_buf = n.clamp(1, 100);
        }
        "audio_mode" => {
            cfg.audio_mode = match val {
                "passthrough" => AudioMode::Passthrough,
                _ => AudioMode::Capture,
            };
        }
        "brightness" => {
            let n: u32 = val.parse().context("invalid brightness")?;
            cfg.brightness = n.clamp(5, 200);
        }
        "contrast" => {
            let n: u32 = val.parse().context("invalid contrast")?;
            cfg.contrast = n.clamp(5, 200);
        }
        "gamma" => {
            let n: u32 = val.parse().context("invalid gamma")?;
            cfg.gamma = n.clamp(10, 300);
        }
        "fps_display" => {
            cfg.fps_display = match val {
                "simple" => "simple",
                "verbose" => "verbose",
                _ => "off",
            }.to_string();
        }
        "pause_on_minimize" => {
            cfg.pause_on_minimize = parse_bool(val);
        }
        "pause_on_background" => {
            cfg.pause_on_background = parse_bool(val);
        }
        "osd_opacity" => {
            let n: u32 = val.parse().context("invalid osd_opacity")?;
            cfg.osd_opacity = n.clamp(0, 100);
        }
        "priority" => {
            cfg.priority = PriorityFlags::parse(val);
        }
        "virtual_webcam" => {
            cfg.virtual_webcam = parse_bool(val);
        }
        "virtual_webcam_device" => {
            cfg.virtual_webcam_device = val.trim().to_string();
        }
        "virtual_mic" => {
            cfg.virtual_mic = parse_bool(val);
        }
        "virtual_mic_sink" => {
            cfg.virtual_mic_sink = val.trim().to_string();
        }
        _ => eprintln!("capview.conf:{}: unknown key '{}'", lineno, key),
    }
    Ok(())
}

/// Persist a single key=value to the config file.
///
/// If `profile` is `Some`, writes into that `[profile]` section.
/// If `None`, writes into the global (top-level) section.
/// Creates the section if it doesn't exist; updates in place if already set.
pub fn save_key(profile: Option<&str>, key: &str, value: &str) {
    let path = config_path();
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(_) => DEFAULT_CONFIG.to_string(),
    };

    let target = profile.map(|p| p.to_lowercase()).unwrap_or_default();
    let mut lines: Vec<String> = text.lines().map(|l| l.to_string()).collect();

    let mut current = String::new();
    let mut key_line: Option<usize> = None;
    let mut insert_at: Option<usize> = None;
    let mut section_found = target.is_empty(); // global always exists

    for (i, line) in lines.iter().enumerate() {
        let t = line.trim();
        if t.starts_with('[') && t.ends_with(']') {
            // Leaving previous section — record insert point if it was our target
            if current == target && section_found && key_line.is_none() && insert_at.is_none() {
                insert_at = Some(i);
            }
            current = t[1..t.len()-1].trim().to_lowercase();
            if current == target {
                section_found = true;
            }
            continue;
        }
        if current != target { continue; }
        if t.is_empty() || t.starts_with('#') { continue; }
        if let Some((k, _)) = t.split_once('=') {
            if k.trim().to_lowercase() == key.to_lowercase() {
                key_line = Some(i);
            }
        }
    }

    // If target section goes to end of file
    if current == target && section_found && key_line.is_none() && insert_at.is_none() {
        insert_at = Some(lines.len());
    }

    let formatted = format!("{:<16}= {}", key, value);

    if let Some(i) = key_line {
        lines[i] = formatted;
    } else if section_found {
        let pos = insert_at.unwrap_or(lines.len());
        lines.insert(pos, formatted);
    } else {
        // Section doesn't exist — create it
        lines.push(String::new());
        if let Some(name) = profile {
            lines.push(format!("[{}]", name));
        }
        lines.push(formatted);
    }

    let mut out = lines.join("\n");
    if !out.ends_with('\n') { out.push('\n'); }
    if let Err(e) = atomic_write(&path, out.as_bytes()) {
        eprintln!("config: failed to save: {}", e);
    }
}

/// Write `contents` to `path` atomically via a same-directory temp file +
/// rename. A crash mid-write leaves either the old file intact or the new
/// file in place — never a half-written config.
fn atomic_write(path: &std::path::Path, contents: &[u8]) -> std::io::Result<()> {
    let dir = path.parent().unwrap_or_else(|| std::path::Path::new("."));
    let file_name = path.file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("capview.conf");
    let tmp = dir.join(format!(".{}.{}.tmp", file_name, std::process::id()));
    match std::fs::write(&tmp, contents) {
        Ok(()) => {}
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            return Err(e);
        }
    }
    if let Err(e) = std::fs::rename(&tmp, path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_bool_accepts_truthy_only() {
        for v in ["true", "1", "yes"] {
            assert!(parse_bool(v), "{v} should be true");
        }
        for v in ["false", "0", "no", "off", "TRUE", ""] {
            assert!(!parse_bool(v), "{v} should be false");
        }
    }

    #[test]
    fn priority_flags_keywords() {
        assert_eq!(PriorityFlags::parse("all"), PriorityFlags::ALL);
        assert_eq!(PriorityFlags::parse("  ALL "), PriorityFlags::ALL);
        assert_eq!(PriorityFlags::parse("none"), PriorityFlags::NONE);
        assert_eq!(PriorityFlags::parse("off"), PriorityFlags::NONE);
    }

    #[test]
    fn priority_flags_list_and_aliases() {
        let f = PriorityFlags::parse("rt, mlock,ioprio");
        assert!(f.has(PriorityFlags::REALTIME));
        assert!(f.has(PriorityFlags::MLOCK));
        assert!(f.has(PriorityFlags::IO_PRIO));
        assert!(!f.has(PriorityFlags::CPU_PIN));
        // Unknown tokens are warned about but ignored, not fatal.
        assert_eq!(PriorityFlags::parse("bogus"), PriorityFlags::NONE);
    }

    #[test]
    fn pixfmt_known_and_fallback() {
        assert_eq!(parse_pixfmt("nv12", 1).unwrap(), V4L2_PIX_FMT_NV12);
        assert_eq!(parse_pixfmt("YUY2", 1).unwrap(), V4L2_PIX_FMT_YUYV);
        assert_eq!(parse_pixfmt("mjpg", 1).unwrap(), V4L2_PIX_FMT_MJPEG);
        // Unknown formats fall back to NV12 rather than erroring.
        assert_eq!(parse_pixfmt("bogus", 1).unwrap(), V4L2_PIX_FMT_NV12);
    }

    #[test]
    fn apply_config_key_parses_and_clamps() {
        let mut cfg = Config::default();
        apply_config_key(&mut cfg, "width", "2560", 1).unwrap();
        assert_eq!(cfg.width, 2560);
        apply_config_key(&mut cfg, "buffers", "99", 1).unwrap();
        assert_eq!(cfg.buffers, MAX_BUFFERS);
        apply_config_key(&mut cfg, "window", "960x540", 1).unwrap();
        assert_eq!((cfg.win_w, cfg.win_h), (Some(960), Some(540)));
        apply_config_key(&mut cfg, "vsync", "yes", 1).unwrap();
        assert!(cfg.vsync);
        assert!(apply_config_key(&mut cfg, "fps", "not-a-number", 1).is_err());
    }
}
