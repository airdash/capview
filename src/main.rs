mod analysis_strip;
#[cfg(target_os = "linux")]
mod audio;
#[cfg(target_os = "macos")]
#[path = "audio_mac.rs"]
mod audio;
#[cfg(target_os = "linux")]
mod capture;
#[cfg(target_os = "macos")]
#[path = "capture_mac.rs"]
mod capture;
#[cfg(target_os = "linux")]
mod clipboard;
#[cfg(target_os = "macos")]
#[path = "clipboard_mac.rs"]
mod clipboard;
mod config;
#[cfg(target_os = "linux")]
mod dmabuf;
mod framegen;
mod gl_renderer;
mod jpeg;
mod net;
mod osd;
mod plugin;
mod priority;
mod recording;
mod screenshot;
mod stream_rx;
mod stream_tx;
mod turbojpeg;
mod vk_renderer;
#[cfg(target_os = "linux")]
mod wayland_minimize;
#[cfg(target_os = "linux")]
mod virtual_webcam;
#[cfg(target_os = "linux")]
mod virtual_mic;

use anyhow::{bail, Result};
use clap::Parser;
use osd::Slot;
use sdl2::event::Event;
use sdl2::keyboard::{Keycode, Mod};
use sdl2::pixels::PixelFormatEnum;
use sdl2::rect::Rect;

use capture::{Capture, V4L2_PIX_FMT_NV12, V4L2_PIX_FMT_YUYV, V4L2_PIX_FMT_UYVY, V4L2_PIX_FMT_XRGB32, V4L2_PIX_FMT_P010, V4L2_PIX_FMT_MJPEG, PIXFMT_RGB24};

/// GL proc address loader.  On macOS, SDL's gl_get_proc_address may return NULL
/// when the accelerated canvas uses Metal internally.  Fall back to loading
/// directly from the OpenGL framework.
#[cfg(target_os = "macos")]
fn gl_proc_address(video: &sdl2::VideoSubsystem, name: &str) -> *const std::ffi::c_void {
    let p = video.gl_get_proc_address(name) as *const std::ffi::c_void;
    if !p.is_null() { return p; }
    // Fallback: load from OpenGL.framework
    use std::sync::OnceLock;
    static LIB: OnceLock<Option<libloading::Library>> = OnceLock::new();
    let lib = LIB.get_or_init(|| unsafe {
        libloading::Library::new("/System/Library/Frameworks/OpenGL.framework/OpenGL").ok()
    });
    if let Some(lib) = lib {
        let cname = std::ffi::CString::new(name).unwrap();
        unsafe { lib.get::<*const std::ffi::c_void>(cname.as_bytes()).ok().map(|s| *s).unwrap_or(std::ptr::null()) }
    } else { std::ptr::null() }
}

#[cfg(not(target_os = "macos"))]
fn gl_proc_address(video: &sdl2::VideoSubsystem, name: &str) -> *const std::ffi::c_void {
    video.gl_get_proc_address(name) as *const std::ffi::c_void
}

#[derive(Parser)]
#[command(name = "capview", about = "Minimal low-latency v4l2 capture card viewer")]
struct Cli {
    /// Video device path
    #[arg(short, long)]
    device: Option<String>,
    /// Capture width
    #[arg(short = 'W', long)]
    width: Option<u32>,
    /// Capture height
    #[arg(short = 'H', long)]
    height: Option<u32>,
    /// Target framerate
    #[arg(short, long)]
    fps: Option<u32>,
    /// Pixel format (NV12, YUYV, UYVY)
    #[arg(short = 'F', long)]
    format: Option<String>,
    /// Suppress all output
    #[arg(short, long)]
    quiet: bool,
    /// Fork to background and detach from terminal
    #[arg(long)]
    fork: bool,
    /// Config profile name (section in capview.conf)
    #[arg(short, long)]
    profile: Option<String>,
    /// Enable debug output
    #[arg(long)]
    debug: bool,
    /// List available profiles and exit
    #[arg(long)]
    list_profiles: bool,
    /// Run as streaming server (bind address, e.g. 0.0.0.0:9000 or just :9000)
    #[arg(long, alias = "stream")]
    server: Option<String>,
    /// Connect to a streaming server and display its frames (e.g. 192.168.1.50:9000)
    #[arg(long, alias = "listen")]
    connect: Option<String>,
    /// Write per-frame performance CSV to file (e.g. --perf perf.csv)
    #[arg(long)]
    perf: Option<String>,
}

fn pixfmt_to_sdl(pixfmt: u32) -> Result<PixelFormatEnum> {
    match pixfmt {
        V4L2_PIX_FMT_NV12 => Ok(PixelFormatEnum::NV12),
        V4L2_PIX_FMT_YUYV => Ok(PixelFormatEnum::YUY2),
        V4L2_PIX_FMT_UYVY => Ok(PixelFormatEnum::UYVY),
        V4L2_PIX_FMT_XRGB32 => Ok(PixelFormatEnum::BGRX8888),
        PIXFMT_RGB24 => Ok(PixelFormatEnum::RGB24),
        V4L2_PIX_FMT_P010 => bail!("P010 requires OpenGL or Vulkan renderer"),
        _ => bail!("unsupported pixel format for SDL texture"),
    }
}

/// Compute a letterboxed destination rect preserving source aspect ratio.
/// Disable vsync for framegen pacing — try adaptive (-1) first to avoid
/// compositor blocking on Wayland, fall back to immediate (0).
fn set_framegen_swap(video: &sdl2::VideoSubsystem) {
    use sdl2::video::SwapInterval;
    // LateSwapTearing (-1): immediate swap but tear if behind vsync.
    // Prevents eglSwapBuffers blocking on compositor buffer release.
    if video.gl_set_swap_interval(SwapInterval::LateSwapTearing).is_err() {
        video.gl_set_swap_interval(SwapInterval::Immediate).ok();
    }
}

fn fit_rect(src_w: u32, src_h: u32, win_w: u32, win_h: u32, mode: config::AspectMode) -> Rect {
    match mode {
        config::AspectMode::Stretch => Rect::new(0, 0, win_w, win_h),
        config::AspectMode::Zoom => {
            // Fill window, crop excess (use max scale instead of min)
            let src_aspect = src_w as f32 / src_h as f32;
            let win_aspect = win_w as f32 / win_h as f32;
            let (dst_w, dst_h) = if win_aspect > src_aspect {
                (win_w, (win_w as f32 / src_aspect) as u32)
            } else {
                ((win_h as f32 * src_aspect) as u32, win_h)
            };
            Rect::new(
                (win_w as i32 - dst_w as i32) / 2,
                (win_h as i32 - dst_h as i32) / 2,
                dst_w,
                dst_h,
            )
        }
        config::AspectMode::Preserve => {
            let src_aspect = src_w as f32 / src_h as f32;
            let win_aspect = win_w as f32 / win_h as f32;
            let (dst_w, dst_h) = if win_aspect > src_aspect {
                ((win_h as f32 * src_aspect) as u32, win_h)
            } else {
                (win_w, (win_w as f32 / src_aspect) as u32)
            };
            Rect::new(
                ((win_w - dst_w) / 2) as i32,
                ((win_h - dst_h) / 2) as i32,
                dst_w,
                dst_h,
            )
        }
    }
}

/// Snap window size to maintain source aspect ratio on user resize.
fn enforce_aspect(win: &mut sdl2::video::Window,
                  src_w: u32, src_h: u32,
                  new_w: i32, new_h: i32) {
    let correct_h = new_w * src_h as i32 / src_w as i32;
    if (new_h - correct_h).abs() > 1 {
        let _ = win.set_size(new_w as u32, correct_h as u32);
    }
}

// ── Screenshot menu ─────────────────────────────────────────────────

const QUALITY_OPTIONS: &[u32] = &[50, 60, 70, 80, 85, 90, 95, 100];
const QUALITY_LABELS: &[&str] = &["50%", "60%", "70%", "80%", "85%", "90%", "95%", "100%"];

const STREAM_PORT_OPTIONS: &[u16] = &[9000, 9001, 9002, 9010, 9100, 8000, 8080];
const STREAM_PORT_LABELS: &[&str] = &["9000", "9001", "9002", "9010", "9100", "8000", "8080"];
const STREAM_QUALITY_OPTIONS: &[u32] = &[40, 50, 60, 70, 80, 90, 95];
const STREAM_QUALITY_LABELS: &[&str] = &["40%", "50%", "60%", "70%", "80%", "90%", "95%"];

const STRIP_GRID_OPTIONS: &[u32] = &[1, 2, 3, 4, 5, 6];
const STRIP_GRID_LABELS: &[&str] = &["1", "2", "3", "4", "5", "6"];

const AUDIO_BUF_OPTIONS: &[i32] = &[5, 10, 15, 20, 30, 40, 50];

fn buf_ms_index(ms: i32) -> usize {
    AUDIO_BUF_OPTIONS.iter().position(|&v| v == ms).unwrap_or(1)
}

fn build_audio_items(cfg: &config::Config, volume: i32, muted: bool, capture_ms: i32, playback_ms: i32) -> Vec<osd::MenuItem> {
    let mode_idx = match cfg.audio_mode {
        config::AudioMode::Capture => 0,
        config::AudioMode::Passthrough => 1,
    };
    let device_label = cfg.audio_device.as_deref().unwrap_or("(none)");
    let vol_labels: Vec<String> = (0..=cfg.max_volume).step_by(5).map(|v| format!("{}%", v)).collect();
    let vol_strs: Vec<&str> = vol_labels.iter().map(|s| s.as_str()).collect();
    let vol_idx = ((volume as u32 / 5).min(cfg.max_volume / 5)) as usize;
    let cap_labels: Vec<String> = AUDIO_BUF_OPTIONS.iter().map(|v| format!("{} ms", v)).collect();
    let cap_strs: Vec<&str> = cap_labels.iter().map(|s| s.as_str()).collect();
    let play_labels: Vec<String> = AUDIO_BUF_OPTIONS.iter().map(|v| format!("{} ms", v)).collect();
    let play_strs: Vec<&str> = play_labels.iter().map(|s| s.as_str()).collect();
    let mut items = vec![
        osd::MenuItem::value("Audio Mode", &["Capture", "Passthrough"], mode_idx),
        osd::MenuItem::value("Device", &[device_label], 0),
    ];
    if cfg.audio_mode == config::AudioMode::Capture {
        items.extend([
            osd::MenuItem::value("Volume", &vol_strs, vol_idx),
            osd::MenuItem::value("Mute", &["Off", "On"], if muted { 1 } else { 0 }),
            osd::MenuItem::separator(),
            osd::MenuItem::value("Sample Rate", &["48000 Hz"], 0),
            osd::MenuItem::value("Channels", &["2 (Stereo)"], 0),
            osd::MenuItem::value("Capture Buffer", &cap_strs, buf_ms_index(capture_ms)),
            osd::MenuItem::value("Playback Buffer", &play_strs, buf_ms_index(playback_ms)),
        ]);
    }
    items
}

fn build_screenshot_items(cfg: &config::Config) -> Vec<osd::MenuItem> {
    let fmt_idx = match cfg.screenshot_format {
        config::ScreenshotFormat::Png  => 0,
        config::ScreenshotFormat::Jpeg => 1,
    };
    let mut items = vec![
        osd::MenuItem::value("Format", &["PNG", "JPEG"], fmt_idx),
    ];
    if cfg.screenshot_format == config::ScreenshotFormat::Jpeg {
        let q_idx = QUALITY_OPTIONS.iter()
            .position(|&q| q == cfg.jpeg_quality)
            .unwrap_or(5); // default to 90%
        items.push(osd::MenuItem::value("JPEG Quality", QUALITY_LABELS, q_idx));
    }
    items
}

fn build_streaming_items(cfg: &config::Config, server_active: bool, client_active: bool) -> Vec<osd::MenuItem> {
    // Server submenu
    let port_idx = STREAM_PORT_OPTIONS.iter()
        .position(|&p| p == cfg.stream_port)
        .unwrap_or(0);
    let q_idx = STREAM_QUALITY_OPTIONS.iter()
        .position(|&q| q == cfg.stream_quality)
        .unwrap_or(4); // default to 80%
    let server_items = vec![
        osd::MenuItem::value("Port", STREAM_PORT_LABELS, port_idx),
        osd::MenuItem::value("Quality", STREAM_QUALITY_LABELS, q_idx),
        osd::MenuItem::action(if server_active { "Stop Server" } else { "Start Server" }),
    ];

    // Client submenu
    let ip = cfg.stream_client_ip;
    let ip_str = format!("{}.{}.{}.{}", ip[0], ip[1], ip[2], ip[3]);
    let port_str = cfg.stream_client_port.to_string();
    let client_items = vec![
        osd::MenuItem::text("Address", &ip_str),
        osd::MenuItem::text("Port", &port_str),
        osd::MenuItem::action(if client_active { "Disconnect" } else { "Connect" }),
    ];

    let mut items = vec![
        osd::MenuItem::submenu("Server", server_items),
        osd::MenuItem::submenu("Client", client_items),
    ];
    #[cfg(target_os = "linux")]
    {
        items.push(osd::MenuItem::value(
            "Virtual Webcam",
            &["Off", "On"],
            if cfg.virtual_webcam { 1 } else { 0 },
        ));
        items.push(osd::MenuItem::value(
            "Virtual Mic",
            &["Off", "On"],
            if cfg.virtual_mic { 1 } else { 0 },
        ));
    }
    items
}

fn build_renderer_items(
    cfg: &config::Config,
    fg_mode: framegen::FrameGenMode,
    fg_quality: framegen::FrameGenQuality,
    target_fps: u32,
    scale_mode: gl_renderer::ScaleMode,
    sharpness: u32,
    active_backend: config::RendererBackend,
    aspect_mode: config::AspectMode,
    brightness: f32,
    contrast: f32,
    gamma: f32,
) -> Vec<osd::MenuItem> {
    // Build backend labels with "(restart)" for backends that need a restart
    let needs_restart = |b: config::RendererBackend| -> bool {
        match (active_backend, b) {
            (config::RendererBackend::Vulkan, config::RendererBackend::Vulkan) => false,
            (config::RendererBackend::Vulkan, _) => true,
            (_, config::RendererBackend::Vulkan) => true,
            _ => false, // SDL ↔ OpenGL can hot-switch
        }
    };
    let label = |name: &str, b: config::RendererBackend| -> String {
        if needs_restart(b) { format!("{} (restart)", name) } else { name.to_string() }
    };
    let backend_labels = [
        label("SDL", config::RendererBackend::Sdl),
        label("OpenGL", config::RendererBackend::OpenGl),
        label("Vulkan", config::RendererBackend::Vulkan),
    ];
    let backend_strs: Vec<&str> = backend_labels.iter().map(|s| s.as_str()).collect();
    let renderer_idx = match cfg.renderer {
        config::RendererBackend::Sdl => 0,
        config::RendererBackend::OpenGl => 1,
        config::RendererBackend::Vulkan => 2,
    };
    let aspect_idx = match aspect_mode {
        config::AspectMode::Preserve => 0,
        config::AspectMode::Zoom => 1,
        config::AspectMode::Stretch => 2,
    };
    let bc_labels: Vec<String> = (1..=40).map(|i| format!("{}%", i * 5)).collect();
    let bc_strs: Vec<&str> = bc_labels.iter().map(|s| s.as_str()).collect();
    let bright_idx = ((brightness * 100.0).round() as u32 / 5).saturating_sub(1).min(39) as usize;
    let contrast_idx = ((contrast * 100.0).round() as u32 / 5).saturating_sub(1).min(39) as usize;
    let mut items = vec![
        osd::MenuItem::value("Backend", &backend_strs, renderer_idx),
        osd::MenuItem::value("Aspect", &["Preserve", "Zoom", "Stretch"], aspect_idx),
        osd::MenuItem::value("Brightness", &bc_strs, bright_idx),
        osd::MenuItem::value("Contrast", &bc_strs, contrast_idx),
    ];
    // Gamma: 0.1–3.0 in 0.1 steps (index 0 = 0.1, index 9 = 1.0, index 29 = 3.0)
    let gamma_labels: Vec<String> = (1..=30).map(|i| format!("{:.1}", i as f32 * 0.1)).collect();
    let gamma_strs: Vec<&str> = gamma_labels.iter().map(|s| s.as_str()).collect();
    let gamma_idx = ((gamma * 10.0).round() as u32).saturating_sub(1).min(29) as usize;
    items.push(osd::MenuItem::value("Gamma", &gamma_strs, gamma_idx));
    let selected_vk = matches!(cfg.renderer, config::RendererBackend::Vulkan);
    if selected_vk {
        let vk_scale_options = ["Nearest", "Bilinear", "CAS", "FSR"];
        let vk_scale_idx = match scale_mode {
            gl_renderer::ScaleMode::Nearest => 0,
            gl_renderer::ScaleMode::Cas => 2,
            gl_renderer::ScaleMode::Fsr | gl_renderer::ScaleMode::IntegerFsr => 3,
            _ => 1, // Bilinear default
        };
        items.push(osd::MenuItem::value("Scaling", &vk_scale_options, vk_scale_idx));
        if scale_mode.has_sharpness() {
            let sharp_options: Vec<String> = (0..=10).map(|i| i.to_string()).collect();
            let sharp_strs: Vec<&str> = sharp_options.iter().map(|s| s.as_str()).collect();
            items.push(osd::MenuItem::value("Sharpness", &sharp_strs, sharpness.min(10) as usize));
        }
        let has_imm = vk_renderer::VkRenderer::immediate_available();
        if has_imm {
            let pm_options = ["Mailbox", "Immediate", "VSync (FIFO)"];
            let pm_idx = match cfg.vk_present_mode {
                config::VkPresentMode::Mailbox => 0,
                config::VkPresentMode::Immediate => 1,
                config::VkPresentMode::Fifo => 2,
            };
            items.push(osd::MenuItem::value("Present Mode", &pm_options, pm_idx));
        } else {
            let pm_options = ["Mailbox", "VSync (FIFO)"];
            let pm_idx = match cfg.vk_present_mode {
                config::VkPresentMode::Mailbox | config::VkPresentMode::Immediate => 0,
                config::VkPresentMode::Fifo => 1,
            };
            items.push(osd::MenuItem::value("Present Mode", &pm_options, pm_idx));
        }
    }
    let selected_gl = matches!(cfg.renderer, config::RendererBackend::OpenGl);
    if selected_gl {
        let scale_options = ["Nearest", "Bilinear", "Integer", "CAS", "FSR", "Integer+FSR"];
        items.push(osd::MenuItem::value("Scaling", &scale_options, scale_mode.index()));
        // Show sharpness for modes that support it
        if scale_mode.has_sharpness() {
            let sharp_options: Vec<String> = (0..=10).map(|i| i.to_string()).collect();
            let sharp_strs: Vec<&str> = sharp_options.iter().map(|s| s.as_str()).collect();
            items.push(osd::MenuItem::value("Sharpness", &sharp_strs, sharpness.min(10) as usize));
        }
    }
    if selected_gl || selected_vk {
        let mode_idx = fg_mode.index();
        let quality_idx = match fg_quality {
            framegen::FrameGenQuality::Fast => 0,
            framegen::FrameGenQuality::Balanced => 1,
            framegen::FrameGenQuality::Quality => 2,
        };
        let fps_options = ["Auto", "60", "90", "120", "144", "165", "180", "240"];
        let fps_idx = match target_fps {
            60  => 1,
            90  => 2,
            120 => 3,
            144 => 4,
            165 => 5,
            180 => 6,
            240 => 7,
            _   => 0,
        };
        #[cfg(feature = "rife")]
        let fg_options = ["Off", "Extrapolate", "Interpolate", "RIFE"];
        #[cfg(not(feature = "rife"))]
        let fg_options = ["Off", "Extrapolate", "Interpolate"];
        items.push(osd::MenuItem::value("Frame Gen", &fg_options, mode_idx));
        if fg_mode != framegen::FrameGenMode::Off {
            #[cfg(feature = "rife")]
            let show_quality = !matches!(fg_mode, framegen::FrameGenMode::Rife);
            #[cfg(not(feature = "rife"))]
            let show_quality = true;
            if show_quality {
                items.push(osd::MenuItem::value("FG Quality", &["Fast", "Balanced", "Quality"], quality_idx));
            }
            items.push(osd::MenuItem::value("Target FPS", &fps_options, fps_idx));
        }
    }
    items
}

fn build_root_menu(
    cfg: &config::Config,
    server_active: bool,
    client_active: bool,
    strip_revealed: bool,
    fg_mode: framegen::FrameGenMode,
    fg_quality: framegen::FrameGenQuality,
    target_fps: u32,
    scale_mode: gl_renderer::ScaleMode,
    sharpness: u32,
    active_backend: config::RendererBackend,
    aspect_mode: config::AspectMode,
    audio_volume: i32,
    audio_muted: bool,
    capture_buf_ms: i32,
    playback_buf_ms: i32,
    brightness: f32,
    contrast: f32,
    gamma: f32,
) -> Vec<osd::MenuItem> {
    let fps_idx = match cfg.fps_display.as_str() {
        "simple" => 1,
        "verbose" => 2,
        _ => 0,
    };
    let opacity_idx = (cfg.osd_opacity / 5).min(20) as usize;
    let mut options_items = vec![
        osd::MenuItem::value("FPS", &["Off", "Simple", "Verbose"], fps_idx),
        osd::MenuItem::value("OSD Opacity", &["0%","5%","10%","15%","20%","25%","30%","35%","40%","45%","50%","55%","60%","65%","70%","75%","80%","85%","90%","95%","100%"], opacity_idx),
        osd::MenuItem::value("Pause on Minimize", &["Off", "On"], if cfg.pause_on_minimize { 1 } else { 0 }),
        osd::MenuItem::value("Pause on Background", &["Off", "On"], if cfg.pause_on_background { 1 } else { 0 }),
        osd::MenuItem::submenu("Screenshots", build_screenshot_items(cfg)),
    ];
    if strip_revealed {
        options_items.push(osd::MenuItem::submenu("Analysis Strip", build_strip_items(cfg)));
    }
    let mut items = vec![
        osd::MenuItem::submenu("Video", build_renderer_items(cfg, fg_mode, fg_quality, target_fps, scale_mode, sharpness, active_backend, aspect_mode, brightness, contrast, gamma)),
        osd::MenuItem::submenu("Audio", build_audio_items(cfg, audio_volume, audio_muted, capture_buf_ms, playback_buf_ms)),
        osd::MenuItem::separator(),
        osd::MenuItem::submenu("Streaming", build_streaming_items(cfg, server_active, client_active)),
    ];
    items.push(osd::MenuItem::submenu("Options", options_items));
    items.push(osd::MenuItem::info(&format!("v{}", env!("CAPVIEW_BUILD_TIME"))));
    items
}

fn build_strip_items(cfg: &config::Config) -> Vec<osd::MenuItem> {
    let cols_idx = STRIP_GRID_OPTIONS.iter()
        .position(|&v| v == cfg.strip_cols)
        .unwrap_or(0);
    let rows_idx = STRIP_GRID_OPTIONS.iter()
        .position(|&v| v == cfg.strip_rows)
        .unwrap_or(5); // default 6
    vec![
        osd::MenuItem::value("Grid Columns", STRIP_GRID_LABELS, cols_idx),
        osd::MenuItem::value("Grid Rows", STRIP_GRID_LABELS, rows_idx),
    ]
}

/// Update the persistent Streaming OSD slot to reflect current server/client state.
fn update_streaming_osd(
    osd: &mut osd::Osd,
    streamer: &Option<stream_tx::StreamSender>,
    client_receiver: &Option<stream_rx::StreamReceiver>,
    cfg: &config::Config,
) {
    let mut parts = Vec::new();
    if let Some(ref s) = streamer {
        let cc = s.client_count();
        parts.push(format!("Server :{} ({})", s.port(), cc));
    }
    if client_receiver.is_some() {
        let ip = cfg.stream_client_ip;
        parts.push(format!("-> {}.{}.{}.{}:{}", ip[0], ip[1], ip[2], ip[3], cfg.stream_client_port));
    }
    if parts.is_empty() {
        osd.clear(Slot::Streaming);
    } else {
        osd.pin(Slot::Streaming, parts.join(" | "));
    }
}

fn sync_menu_to_config(osd: &osd::Osd, cfg: &mut config::Config, profile: Option<&str>, aspect_mode: &mut config::AspectMode) {
    // Read format (searched recursively through submenus)
    if let Some((sel, _)) = osd.find_menu_value("Format") {
        let new_fmt = if sel == 1 {
            config::ScreenshotFormat::Jpeg
        } else {
            config::ScreenshotFormat::Png
        };
        if new_fmt != cfg.screenshot_format {
            cfg.screenshot_format = new_fmt;
            let s = match new_fmt {
                config::ScreenshotFormat::Png  => "png",
                config::ScreenshotFormat::Jpeg => "jpeg",
            };
            config::save_key(profile, "screenshot_format", s);
        }
    }
    // Read quality (only when JPEG)
    if cfg.screenshot_format == config::ScreenshotFormat::Jpeg {
        if let Some((sel, _)) = osd.find_menu_value("JPEG Quality") {
            if sel < QUALITY_OPTIONS.len() {
                let new_q = QUALITY_OPTIONS[sel];
                if new_q != cfg.jpeg_quality {
                    cfg.jpeg_quality = new_q;
                    config::save_key(profile, "jpeg_quality", &new_q.to_string());
                }
            }
        }
    }
    // Read renderer backend
    if let Some((sel, _)) = osd.find_menu_value("Backend") {
        let new_r = match sel {
            1 => config::RendererBackend::OpenGl,
            2 => config::RendererBackend::Vulkan,
            _ => config::RendererBackend::Sdl,
        };
        if new_r != cfg.renderer {
            cfg.renderer = new_r;
            let s = match new_r {
                config::RendererBackend::Sdl => "sdl",
                config::RendererBackend::OpenGl => "opengl",
                config::RendererBackend::Vulkan => "vulkan",
            };
            config::save_key(profile, "renderer", s);
        }
    }
    // Read aspect mode
    if let Some((sel, _)) = osd.find_menu_value("Aspect") {
        let new_mode = match sel {
            1 => config::AspectMode::Zoom,
            2 => config::AspectMode::Stretch,
            _ => config::AspectMode::Preserve,
        };
        if new_mode != *aspect_mode {
            *aspect_mode = new_mode;
            cfg.aspect_mode = new_mode;
            let s = match new_mode {
                config::AspectMode::Preserve => "preserve",
                config::AspectMode::Zoom => "zoom",
                config::AspectMode::Stretch => "stretch",
            };
            config::save_key(profile, "aspect_mode", s);
        }
    }
    // Read Vulkan present mode
    if let Some((sel, _)) = osd.find_menu_value("Present Mode") {
        let has_imm = vk_renderer::VkRenderer::immediate_available();
        let new_pm = if has_imm {
            match sel {
                1 => config::VkPresentMode::Immediate,
                2 => config::VkPresentMode::Fifo,
                _ => config::VkPresentMode::Mailbox,
            }
        } else {
            // 2-option menu: Mailbox, VSync (FIFO)
            match sel {
                1 => config::VkPresentMode::Fifo,
                _ => config::VkPresentMode::Mailbox,
            }
        };
        if new_pm != cfg.vk_present_mode {
            cfg.vk_present_mode = new_pm;
            let s = match new_pm {
                config::VkPresentMode::Mailbox => "mailbox",
                config::VkPresentMode::Immediate => "immediate",
                config::VkPresentMode::Fifo => "fifo",
            };
            config::save_key(profile, "vk_present_mode", s);
        }
    }
    // Read streaming port
    if let Some((sel, _)) = osd.find_menu_value("Port") {
        if sel < STREAM_PORT_OPTIONS.len() {
            let new_p = STREAM_PORT_OPTIONS[sel];
            if new_p != cfg.stream_port {
                cfg.stream_port = new_p;
                config::save_key(profile, "stream_port", &new_p.to_string());
            }
        }
    }
    // Read streaming quality
    if let Some((sel, _)) = osd.find_menu_value("Quality") {
        if sel < STREAM_QUALITY_OPTIONS.len() {
            let new_q = STREAM_QUALITY_OPTIONS[sel];
            if new_q != cfg.stream_quality {
                cfg.stream_quality = new_q;
                config::save_key(profile, "stream_quality", &new_q.to_string());
            }
        }
    }
    // Read strip grid columns
    if let Some((sel, _)) = osd.find_menu_value("Grid Columns") {
        if sel < STRIP_GRID_OPTIONS.len() {
            let new_c = STRIP_GRID_OPTIONS[sel];
            if new_c != cfg.strip_cols {
                cfg.strip_cols = new_c;
                config::save_key(profile, "strip_cols", &new_c.to_string());
            }
        }
    }
    // Read strip grid rows
    if let Some((sel, _)) = osd.find_menu_value("Grid Rows") {
        if sel < STRIP_GRID_OPTIONS.len() {
            let new_r = STRIP_GRID_OPTIONS[sel];
            if new_r != cfg.strip_rows {
                cfg.strip_rows = new_r;
                config::save_key(profile, "strip_rows", &new_r.to_string());
            }
        }
    }
    // Read audio mode
    if let Some((sel, _)) = osd.find_menu_value("Audio Mode") {
        let new_mode = match sel {
            1 => config::AudioMode::Passthrough,
            _ => config::AudioMode::Capture,
        };
        if new_mode != cfg.audio_mode {
            cfg.audio_mode = new_mode;
            let s = match new_mode {
                config::AudioMode::Capture => "capture",
                config::AudioMode::Passthrough => "passthrough",
            };
            config::save_key(profile, "audio_mode", s);
        }
    }
    // Read pause on minimize
    if let Some((sel, _)) = osd.find_menu_value("Pause on Minimize") {
        let new_val = sel == 1;
        if new_val != cfg.pause_on_minimize {
            cfg.pause_on_minimize = new_val;
            config::save_key(profile, "pause_on_minimize", if new_val { "true" } else { "false" });
        }
    }
    // Read pause on background
    if let Some((sel, _)) = osd.find_menu_value("Pause on Background") {
        let new_val = sel == 1;
        if new_val != cfg.pause_on_background {
            cfg.pause_on_background = new_val;
            config::save_key(profile, "pause_on_background", if new_val { "true" } else { "false" });
        }
    }
}

/// Read the Virtual Webcam menu toggle and reconcile the running instance.
/// Handles start/stop, persists the config key, and shows a toast.
#[cfg(target_os = "linux")]
fn sync_vcam_menu(
    osd: &mut osd::Osd,
    vcam: &mut Option<virtual_webcam::VirtualWebcam>,
    cfg: &mut config::Config,
    profile: Option<&str>,
    cap_w: u32,
    cap_h: u32,
    pixfmt: u32,
) {
    let sel = match osd.find_menu_value("Virtual Webcam") {
        Some((s, _)) => s,
        None => return,
    };
    let want_on = sel == 1;
    let is_on = vcam.is_some();
    if want_on == is_on { return; }
    if want_on {
        match virtual_webcam::VirtualWebcam::start(
            &cfg.virtual_webcam_device, cap_w, cap_h, pixfmt,
        ) {
            Ok(v) => {
                let path = v.device_path().to_string();
                *vcam = Some(v);
                cfg.virtual_webcam = true;
                config::save_key(profile, "virtual_webcam", "true");
                osd.show(Slot::Transient, format!("VCAM on: {}", path), 1500);
            }
            Err(e) => {
                eprintln!("virtual_webcam: {}", e);
                osd.show(Slot::Transient, format!("VCAM: {}", e), 3000);
                osd.set_menu_value("Virtual Webcam", 0);
            }
        }
    } else {
        *vcam = None;
        cfg.virtual_webcam = false;
        config::save_key(profile, "virtual_webcam", "false");
        osd.show(Slot::Transient, "VCAM off", 1500);
    }
}

/// Read the Virtual Mic menu toggle and reconcile the running instance.
/// Loads/unloads the PulseAudio null-sink, wires the audio-thread tee.
#[cfg(target_os = "linux")]
fn sync_vmic_menu(
    osd: &mut osd::Osd,
    vmic: &mut Option<virtual_mic::VirtualMic>,
    audio: Option<&audio::AudioPassthrough>,
    cfg: &mut config::Config,
    profile: Option<&str>,
) {
    let sel = match osd.find_menu_value("Virtual Mic") {
        Some((s, _)) => s,
        None => return,
    };
    let want_on = sel == 1;
    let is_on = vmic.is_some();
    if want_on == is_on { return; }

    if want_on {
        let audio = match audio {
            Some(a) if a.mode() == config::AudioMode::Capture => a,
            Some(_) => {
                eprintln!("virtual_mic: requires audio mode=capture (current: passthrough)");
                osd.show(Slot::Transient, "VMIC: needs capture mode", 3000);
                osd.set_menu_value("Virtual Mic", 0);
                return;
            }
            None => {
                eprintln!("virtual_mic: no audio device configured");
                osd.show(Slot::Transient, "VMIC: no audio device", 3000);
                osd.set_menu_value("Virtual Mic", 0);
                return;
            }
        };
        match virtual_mic::VirtualMic::start(&cfg.virtual_mic_sink) {
            Ok(v) => {
                audio.set_virtual_mic(Some(v.tee()));
                let monitor = v.monitor_source();
                *vmic = Some(v);
                cfg.virtual_mic = true;
                config::save_key(profile, "virtual_mic", "true");
                osd.show(Slot::Transient, format!("VMIC on: {}", monitor), 2000);
            }
            Err(e) => {
                eprintln!("virtual_mic: {}", e);
                osd.show(Slot::Transient, format!("VMIC: {}", e), 3000);
                osd.set_menu_value("Virtual Mic", 0);
            }
        }
    } else {
        if let Some(a) = audio {
            a.set_virtual_mic(None);
        }
        *vmic = None;
        cfg.virtual_mic = false;
        config::save_key(profile, "virtual_mic", "false");
        osd.show(Slot::Transient, "VMIC off", 1500);
    }
}

fn framegen_mode_str(m: framegen::FrameGenMode) -> &'static str {
    match m {
        framegen::FrameGenMode::Off => "off",
        framegen::FrameGenMode::Extrapolate => "extrapolate",
        framegen::FrameGenMode::Interpolate => "interpolate",
        #[cfg(feature = "rife")]
        framegen::FrameGenMode::Rife => "rife",
    }
}

fn framegen_quality_str(q: framegen::FrameGenQuality) -> &'static str {
    match q {
        framegen::FrameGenQuality::Fast => "fast",
        framegen::FrameGenQuality::Balanced => "balanced",
        framegen::FrameGenQuality::Quality => "quality",
    }
}

/// Read frame gen mode, quality, target FPS, and scaling mode from OSD menu.
fn read_framegen_menu(osd: &osd::Osd, use_vk: bool) -> (
    Option<framegen::FrameGenMode>,
    Option<framegen::FrameGenQuality>,
    Option<u32>,
    Option<gl_renderer::ScaleMode>,
    Option<u32>,
) {
    let mode = osd.find_menu_value("Frame Gen").map(|(sel, _)| {
        framegen::FrameGenMode::from_index(sel)
    });
    let quality = osd.find_menu_value("FG Quality").map(|(sel, _)| match sel {
        0 => framegen::FrameGenQuality::Fast,
        2 => framegen::FrameGenQuality::Quality,
        _ => framegen::FrameGenQuality::Balanced,
    });
    let tfps = osd.find_menu_value("Target FPS").map(|(sel, _)| match sel {
        1 => 60,
        2 => 90,
        3 => 120,
        4 => 144,
        5 => 165,
        6 => 180,
        7 => 240,
        _ => 0,
    });
    let scaling = osd.find_menu_value("Scaling").map(|(sel, _)| {
        if use_vk {
            // VK menu: 0=Nearest, 1=Bilinear, 2=CAS, 3=FSR
            match sel {
                0 => gl_renderer::ScaleMode::Nearest,
                2 => gl_renderer::ScaleMode::Cas,
                3 => gl_renderer::ScaleMode::Fsr,
                _ => gl_renderer::ScaleMode::Bilinear,
            }
        } else {
            gl_renderer::ScaleMode::from_index(sel)
        }
    });
    let sharpness = osd.find_menu_value("Sharpness").map(|(sel, _)| sel as u32);
    (mode, quality, tfps, scaling, sharpness)
}

fn daemonize() -> Result<()> {
    unsafe {
        let pid = libc::fork();
        if pid < 0 {
            bail!("fork: {}", std::io::Error::last_os_error());
        }
        if pid > 0 {
            libc::_exit(0);
        }
        libc::setsid();
        // Redirect stdin/stdout/stderr to /dev/null
        libc::close(0);
        libc::open(b"/dev/null\0".as_ptr() as *const _, libc::O_RDONLY);
        libc::close(1);
        libc::open(b"/dev/null\0".as_ptr() as *const _, libc::O_WRONLY);
        libc::close(2);
        libc::open(b"/dev/null\0".as_ptr() as *const _, libc::O_WRONLY);
    }
    Ok(())
}

fn silence_output() {
    unsafe {
        let devnull = libc::open(b"/dev/null\0".as_ptr() as *const _, libc::O_WRONLY);
        if devnull >= 0 {
            libc::dup2(devnull, 1);
            libc::dup2(devnull, 2);
            libc::close(devnull);
        }
    }
}

/// Receiver mode — connect to a remote capview sender and display the stream.
fn run_receiver(sender_addr: &str, debug: bool) -> Result<()> {
    let mut receiver = stream_rx::StreamReceiver::start(sender_addr, debug)?;

    // Wait for first frame to learn dimensions
    eprintln!("streaming: waiting for first frame from {}…", sender_addr);
    let first = loop {
        if let Some(f) = receiver.try_recv() {
            break f;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    };

    let (src_w, src_h) = (first.width, first.height);
    eprintln!("streaming: receiving {}x{} @ {}fps", src_w, src_h, first.fps);

    // SDL init
    #[cfg(target_os = "linux")]
    sdl2::hint::set("SDL_VIDEODRIVER", "wayland,x11");
    sdl2::hint::set("SDL_RENDER_VSYNC", "1");
    let sdl = sdl2::init().map_err(|e| anyhow::anyhow!(e))?;
    let video = sdl.video().map_err(|e| anyhow::anyhow!(e))?;

    let title = format!("capview <stream from {}>", sender_addr);
    let window = video.window(&title, src_w, src_h)
        .position_centered().resizable().allow_highdpi()
        .build()?;
    let mut canvas = window.into_canvas().accelerated().present_vsync().build()?;
    let tc = canvas.texture_creator();

    let mut texture = tc
        .create_texture_streaming(PixelFormatEnum::RGB24, src_w, src_h)
        .map_err(|e| anyhow::anyhow!(e))?;

    // Upload first frame
    let pitch = (src_w * 3) as usize;
    let _ = texture.update(None, &first.rgb, pitch);

    let mut event_pump = sdl.event_pump().map_err(|e| anyhow::anyhow!(e))?;
    let mut dirty = true;

    'main: loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. } => break 'main,
                Event::KeyDown { keycode: Some(Keycode::Q), .. }
                | Event::KeyDown { keycode: Some(Keycode::Escape), .. } => break 'main,
                Event::KeyDown { keycode: Some(Keycode::F), .. } => {
                    let win = canvas.window_mut();
                    use sdl2::video::FullscreenType;
                    if win.fullscreen_state() == FullscreenType::Off {
                        let _ = win.set_fullscreen(FullscreenType::Desktop);
                    } else {
                        let _ = win.set_fullscreen(FullscreenType::Off);
                    }
                    dirty = true;
                }
                _ => {}
            }
        }

        // Poll for new frames
        while let Some(frame) = receiver.try_recv() {
            let p = (frame.width * 3) as usize;
            let _ = texture.update(None, &frame.rgb, p);
            dirty = true;
        }

        if dirty {
            let (win_w, win_h) = canvas.output_size().unwrap_or((src_w, src_h));
            let dst = fit_rect(src_w, src_h, win_w, win_h, config::AspectMode::Preserve);
            canvas.set_draw_color(sdl2::pixels::Color::RGB(0, 0, 0));
            canvas.clear();
            canvas.copy(&texture, None, Some(dst)).map_err(|e| anyhow::anyhow!(e))?;
            canvas.present();
            dirty = false;
        } else {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    receiver.stop();
    Ok(())
}

/// Search for RIFE ONNX model file in standard locations.
#[cfg(feature = "rife")]
fn rife_model_path() -> std::path::PathBuf {
    // 1. Environment variable
    if let Ok(p) = std::env::var("CAPVIEW_RIFE_MODEL") {
        let path = std::path::PathBuf::from(p);
        if path.exists() { return path; }
    }
    // 2. XDG config / ~/.config/capview/rife.onnx
    if let Ok(home) = std::env::var("HOME") {
        let xdg = std::env::var("XDG_CONFIG_HOME")
            .unwrap_or_else(|_| format!("{}/.config", home));
        let p = std::path::PathBuf::from(&xdg).join("capview").join("rife.onnx");
        if p.exists() { return p; }
    }
    // 3. Next to executable
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let p = dir.join("rife.onnx");
            if p.exists() { return p; }
        }
    }
    // 4. Current directory
    let p = std::path::PathBuf::from("rife.onnx");
    if p.exists() { return p; }
    // Fallback: config dir path (will fail with a descriptive error)
    if let Ok(home) = std::env::var("HOME") {
        let xdg = std::env::var("XDG_CONFIG_HOME")
            .unwrap_or_else(|_| format!("{}/.config", home));
        return std::path::PathBuf::from(&xdg).join("capview").join("rife.onnx");
    }
    std::path::PathBuf::from("rife.onnx")
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    if cli.list_profiles {
        let profiles = config::Config::list_profiles();
        if profiles.is_empty() {
            eprintln!("no profiles defined in config file");
        } else {
            for p in profiles {
                println!("{}", p);
            }
        }
        return Ok(());
    }

    let mut cfg = config::Config::load(cli.profile.as_deref())?;

    // CLI overrides config file
    if let Some(d) = cli.device { cfg.device = d; }
    if let Some(w) = cli.width { cfg.width = w; }
    if let Some(h) = cli.height { cfg.height = h; }
    if let Some(f) = cli.fps { cfg.fps = f; }
    if let Some(fmt) = cli.format {
        cfg.pixfmt = match fmt.to_uppercase().as_str() {
            "NV12" => V4L2_PIX_FMT_NV12,
            "YUYV" | "YUY2" => V4L2_PIX_FMT_YUYV,
            "UYVY" => V4L2_PIX_FMT_UYVY,
            "XRGB" | "RGBX" | "BGRX" => V4L2_PIX_FMT_XRGB32,
            "P010" => V4L2_PIX_FMT_P010,
            "MJPEG" | "MJPG" => V4L2_PIX_FMT_MJPEG,
            _ => bail!("unknown format '{}' (NV12, YUYV, YUY2, UYVY, XRGB, P010, MJPEG)", fmt),
        };
    }

    // CLI flags override config booleans
    let debug = cli.debug;
    let quiet = !debug && (cli.quiet || cfg.quiet);
    let do_fork = !debug && (cli.fork || cfg.daemonize);

    if quiet {
        silence_output();
    }

    if !quiet {
        let fmtname = capture::format_name(cfg.pixfmt);
        let win_str = match (cfg.win_w, cfg.win_h) {
            (Some(w), Some(h)) => format!(" win={}x{}", w, h),
            _ => String::new(),
        };
        let profile_str = match &cli.profile {
            Some(p) => format!(" profile={}", p),
            None => String::new(),
        };
        let audio_str = match &cfg.audio_device {
            Some(d) => format!(" audio={}", d),
            None => String::new(),
        };
        eprintln!("capview: {} {}x{}@{} {} buf={}{}{}{}{}{}{}",
                  cfg.device, cfg.width, cfg.height, cfg.fps,
                  fmtname, cfg.buffers, win_str,
                  if cfg.vsync { " vsync" } else { "" },
                  if cfg.smooth { " smooth" } else { "" },
                  if cfg.fullscreen { " fullscreen" } else { "" },
                  profile_str, audio_str);
    }

    // Fork BEFORE opening any devices or threads — forking with active
    // threads (e.g. audio) kills the child copies and corrupts PA state.
    if do_fork {
        daemonize()?;
    }

    // ── Receiver mode (--connect) ───────────────────────────────────
    if let Some(ref addr) = cli.connect {
        return run_receiver(addr, debug);
    }

    // ── Auto-start streaming server if --server given ───────────────
    let auto_stream = cli.server.clone();

    let t0 = std::time::Instant::now();
    let cap = Capture::open(
        &cfg.device, cfg.width, cfg.height, cfg.fps, cfg.pixfmt, cfg.buffers,
    )?;
    cap.start()?;
    eprintln!("[{:7.3}s] capture started", t0.elapsed().as_secs_f64());

    // For MJPEG, decode to RGB24 before any renderer sees the data.
    // eff_pixfmt is the format renderers actually receive.
    let is_mjpeg = cap.pixfmt == V4L2_PIX_FMT_MJPEG;
    let eff_pixfmt = if is_mjpeg { PIXFMT_RGB24 } else { cap.pixfmt };
    let mjpeg_decoder: Option<turbojpeg::TurboJpeg> = if is_mjpeg {
        match turbojpeg::TurboJpeg::new() {
            Ok(tj) => Some(tj),
            Err(e) => bail!("MJPEG requires libturbojpeg: {}", e),
        }
    } else { None };
    let mut mjpeg_rgb_buf: Vec<u8> = Vec::new();

    // Export DMA-BUF file descriptors for zero-copy GL rendering (Linux only).
    #[cfg(target_os = "linux")]
    let dmabuf_fds: Vec<std::os::unix::io::RawFd> = match cap.export_dmabuf_fds() {
        Ok(fds) => {
            if debug { eprintln!("debug: exported {} DMA-BUF fd(s)", fds.len()); }
            fds
        }
        Err(e) => {
            if debug { eprintln!("debug: DMA-BUF export unavailable: {}", e); }
            Vec::new()
        }
    };

    // Start audio passthrough if configured
    let mut _audio: Option<audio::AudioPassthrough> = None;
    if let Some(ref audio_query) = cfg.audio_device {
        if debug { eprintln!("debug: resolving audio device '{}'", audio_query); }
        match audio::AudioPassthrough::start(audio_query, cfg.max_volume, cfg.audio_capture_buf, cfg.audio_playback_buf, cfg.audio_mode, debug) {
            Ok(a) => _audio = Some(a),
            Err(e) => eprintln!("audio: {}", e),
        }
    }

    // Load filter plugins
    let pipeline = if !cfg.plugins.is_empty() {
        match plugin::FilterPipeline::load(&cfg.plugins, cap.width, cap.height, cfg.fps, eff_pixfmt, debug) {
            Ok(p) => {
                if debug { eprintln!("debug: {} plugin(s) loaded", cfg.plugins.len()); }
                Some(p)
            }
            Err(e) => {
                eprintln!("plugin error: {}", e);
                None
            }
        }
    } else {
        None
    };

    eprintln!("[{:7.3}s] initializing SDL", t0.elapsed().as_secs_f64());

    // SDL init — prefer native Wayland over XWayland for lower latency
    #[cfg(target_os = "linux")]
    sdl2::hint::set("SDL_VIDEODRIVER", "wayland,x11");
    // macOS: help SDL find the Vulkan loader from Homebrew, and point
    // the loader at MoltenVK's ICD manifest.
    #[cfg(target_os = "macos")]
    {
        for path in &[
            "/opt/homebrew/lib/libvulkan.1.dylib",   // Apple Silicon
            "/usr/local/lib/libvulkan.1.dylib",       // Intel
        ] {
            if std::path::Path::new(path).exists() {
                sdl2::hint::set("SDL_VULKAN_LIBRARY", path);
                break;
            }
        }
        if std::env::var("VK_ICD_FILENAMES").is_err() {
            for path in &[
                "/opt/homebrew/share/vulkan/icd.d/MoltenVK_icd.json",
                "/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json",
                "/usr/local/share/vulkan/icd.d/MoltenVK_icd.json",
            ] {
                if std::path::Path::new(path).exists() {
                    std::env::set_var("VK_ICD_FILENAMES", path);
                    break;
                }
            }
        }
    }
    sdl2::hint::set("SDL_FRAMEBUFFER_ACCELERATION", "1");
    sdl2::hint::set("SDL_RENDER_VSYNC", if cfg.vsync { "1" } else { "0" });
    let sdl = sdl2::init().map_err(|e| anyhow::anyhow!(e))?;
    let video = sdl.video().map_err(|e| anyhow::anyhow!(e))?;
    eprintln!("[{:7.3}s] SDL video initialized", t0.elapsed().as_secs_f64());

    // Enable SDL text input so we receive TextInput events for menu text fields
    video.text_input().start();

    // Window size: config override, else capture resolution
    let (init_w, init_h) = match (cfg.win_w, cfg.win_h) {
        (Some(w), Some(h)) => (w, h),
        _ => (cap.width, cap.height),
    };

    let use_vk = cfg.renderer == config::RendererBackend::Vulkan;

    // macOS requires GL 3.2 core profile for #version 150 shaders.
    // Only set for OpenGL renderer — SDL mode uses Metal or its own GL context.
    #[cfg(target_os = "macos")]
    if !use_vk && cfg.renderer == config::RendererBackend::OpenGl {
        let gl_attr = video.gl_attr();
        gl_attr.set_context_version(3, 2);
        gl_attr.set_context_profile(sdl2::video::GLProfile::Core);
        gl_attr.set_context_flags().forward_compatible().set();
    }

    eprintln!("[{:7.3}s] creating window (vk={})", t0.elapsed().as_secs_f64(), use_vk);
    let mut win_builder = video.window("capview", init_w, init_h);
    win_builder.position_centered().resizable().allow_highdpi();
    if use_vk {
        win_builder.vulkan();
    } else {
        win_builder.opengl();
    }
    if cfg.fullscreen {
        win_builder.fullscreen_desktop();
    }

    // Vulkan mode: keep raw Window for VkRenderer surface.
    // GL mode on macOS: keep raw Window + GL context (SDL canvas uses Metal).
    // SDL mode: consume Window into Canvas.
    let mut vk_window: Option<sdl2::video::Window> = None;
    let mut canvas: Option<sdl2::render::Canvas<sdl2::video::Window>> = None;
    let mut vk_renderer_inst: Option<vk_renderer::VkRenderer> = None;
    // macOS: hold GL context alive (dropped = context destroyed)
    let mut _gl_ctx: Option<sdl2::video::GLContext> = None;

    // On macOS, OpenGL mode needs a raw GL context — SDL's canvas uses Metal.
    #[cfg(target_os = "macos")]
    let macos_gl_mode = !use_vk && cfg.renderer == config::RendererBackend::OpenGl;
    #[cfg(not(target_os = "macos"))]
    let macos_gl_mode = false;

    if use_vk {
        // On Wayland, IMMEDIATE mode is unavailable — fall back to Mailbox
        if !vk_renderer::VkRenderer::immediate_available()
            && cfg.vk_present_mode == config::VkPresentMode::Immediate
        {
            cfg.vk_present_mode = config::VkPresentMode::Mailbox;
        }
        // Window build can fail if Vulkan loader (MoltenVK) is missing
        match win_builder.build() {
            Ok(window) => {
                eprintln!("[{:7.3}s] VK window built", t0.elapsed().as_secs_f64());
                match vk_renderer::VkRenderer::new(
                    &window, cap.width, cap.height, eff_pixfmt,
                    Some(vk_renderer::VkRenderer::config_to_vk_present_mode(cfg.vk_present_mode)),
                    debug,
                ) {
                    Ok(mut vr) => {
                        eprintln!("[{:7.3}s] vulkan renderer initialized (mailbox={})", t0.elapsed().as_secs_f64(), vr.is_mailbox());
                        vr.aspect_mode = cfg.aspect_mode;
                        vk_renderer_inst = Some(vr);
                        vk_window = Some(window);
                    }
                    Err(e) => {
                        eprintln!("vulkan: init failed: {} — falling back to SDL", e);
                        cfg.renderer = config::RendererBackend::Sdl;
                        config::save_key(cli.profile.as_deref(), "renderer", "sdl");
                        let mut fb = video.window("capview", init_w, init_h);
                        fb.position_centered().resizable().allow_highdpi().opengl();
                        if cfg.fullscreen { fb.fullscreen_desktop(); }
                        let c = fb.build()?.into_canvas().accelerated().build()?;
                        canvas = Some(c);
                    }
                }
            }
            Err(e) => {
                eprintln!("vulkan: {} — falling back to SDL", e);
                cfg.renderer = config::RendererBackend::Sdl;
                config::save_key(cli.profile.as_deref(), "renderer", "sdl");
                let mut fb = video.window("capview", init_w, init_h);
                fb.position_centered().resizable().allow_highdpi().opengl();
                if cfg.fullscreen { fb.fullscreen_desktop(); }
                let c = fb.build()?.into_canvas().accelerated().build()?;
                canvas = Some(c);
            }
        }
    } else if macos_gl_mode {
        // macOS GL path: create window + GL 3.2 context directly
        let window = win_builder.build()?;
        eprintln!("[{:7.3}s] GL window built (macOS direct context)", t0.elapsed().as_secs_f64());
        match window.gl_create_context() {
            Ok(ctx) => {
                window.gl_make_current(&ctx).map_err(|e| anyhow::anyhow!(e))?;
                if cfg.vsync {
                    video.gl_set_swap_interval(sdl2::video::SwapInterval::VSync).ok();
                }
                _gl_ctx = Some(ctx);
                vk_window = Some(window);
            }
            Err(e) => {
                eprintln!("opengl: GL context creation failed: {} — falling back to SDL", e);
                cfg.renderer = config::RendererBackend::Sdl;
                config::save_key(cli.profile.as_deref(), "renderer", "sdl");
                let c = window.into_canvas().accelerated().build()?;
                canvas = Some(c);
            }
        }
    } else {
        let window = win_builder.build()?;
        eprintln!("[{:7.3}s] SDL window built", t0.elapsed().as_secs_f64());
        let c = window.into_canvas().accelerated().build()?;
        eprintln!("[{:7.3}s] SDL canvas created", t0.elapsed().as_secs_f64());
        if cfg.vsync {
            video.gl_set_swap_interval(sdl2::video::SwapInterval::VSync).ok();
        }
        canvas = Some(c);
    }

    // Helper closures for window access
    macro_rules! with_window {
        ($f:expr) => {
            if let Some(ref c) = canvas {
                $f(c.window())
            } else if let Some(ref w) = vk_window {
                $f(w)
            } else {
                unreachable!()
            }
        };
    }
    macro_rules! with_window_mut {
        ($f:expr) => {
            if let Some(ref mut c) = canvas {
                $f(c.window_mut())
            } else if let Some(ref mut w) = vk_window {
                $f(w)
            } else {
                unreachable!()
            }
        };
    }

    // SDL texture + OSD (only for non-Vulkan mode)
    let sdl_fmt = pixfmt_to_sdl(eff_pixfmt)?;
    let texture_creator = canvas.as_ref().map(|c| c.texture_creator());
    let mut texture = match texture_creator {
        Some(ref tc) => {
            let tex = tc
                .create_texture_streaming(sdl_fmt, cap.width, cap.height)
                .map_err(|e| anyhow::anyhow!(e))?;
            if cfg.smooth {
                unsafe {
                    sdl2_sys::SDL_SetTextureScaleMode(
                        tex.raw(),
                        sdl2_sys::SDL_ScaleMode::SDL_ScaleModeLinear,
                    );
                }
            }
            Some(tex)
        }
        None => None,
    };

    let mut event_pump = sdl.event_pump().map_err(|e| anyhow::anyhow!(e))?;

    // OSD: in non-Vulkan mode, uses SDL texture atlas. In Vulkan mode, uses VkRenderer OSD.
    let mut osd = if let Some(ref tc) = texture_creator {
        osd::Osd::new(tc)?
    } else {
        // Vulkan mode: create a dummy OSD — actual rendering done via VkRenderer
        // We still need the data model for menu/slot state.
        // Create a temporary off-screen surface to satisfy the texture creator requirement.
        osd::Osd::new_headless()?
    };

    // Analysis strip easter egg state (declared early for menu init)
    let mut strip_revealed = false;

    let mut scale_mode = gl_renderer::ScaleMode::from_config(&cfg.scale_mode)
        .unwrap_or(if cfg.smooth {
            gl_renderer::ScaleMode::Bilinear
        } else {
            gl_renderer::ScaleMode::Nearest
        });
    let mut sharpness: u32 = cfg.sharpness;

    // Set up centre menu with screenshot options
    let active_backend = cfg.renderer;
    osd.set_opacity(cfg.osd_opacity);
    osd.set_menu_items(build_root_menu(&cfg, false, false, strip_revealed,
        framegen::FrameGenMode::Off, framegen::FrameGenQuality::Balanced,
        cfg.target_fps, scale_mode, sharpness,
        active_backend, cfg.aspect_mode, 100, false, cfg.audio_capture_buf as i32, cfg.audio_playback_buf as i32, cfg.brightness as f32 / 100.0, cfg.contrast as f32 / 100.0, cfg.gamma as f32 / 100.0));

    eprintln!("[{:7.3}s] OSD initialized, renderer={:?}", t0.elapsed().as_secs_f64(), cfg.renderer);

    // GL renderer (optional — created on demand when OpenGL backend selected)
    let mut gl_renderer: Option<gl_renderer::GlRenderer> = None;
    let mut use_gl = cfg.renderer == config::RendererBackend::OpenGl;
    let mut saved_sdl_state: Option<gl_renderer::GlSavedState> = None;
    if use_gl {
        eprintln!("[{:7.3}s] initializing OpenGL renderer...", t0.elapsed().as_secs_f64());
        match gl_renderer::GlRenderer::new(
            |s| gl_proc_address(&video, s),
            cap.width, cap.height, eff_pixfmt, cfg.smooth, debug,
        ) {
            Ok((mut r, sdl_state)) => {
                if debug { eprintln!("debug: opengl renderer initialized"); }
                // Try DMA-BUF zero-copy import (Linux only)
                #[cfg(target_os = "linux")]
                if !dmabuf_fds.is_empty() {
                    match r.init_dmabuf(
                        |s| gl_proc_address(&video, s),
                        &dmabuf_fds, debug,
                    ) {
                        Ok(()) => {
                            eprintln!("dmabuf: zero-copy enabled ({} buffers)", dmabuf_fds.len());
                        }
                        Err(e) => {
                            if debug { eprintln!("debug: DMA-BUF init: {}", e); }
                        }
                    }
                }
                saved_sdl_state = Some(sdl_state);
                r.set_scale_mode(scale_mode);
                r.set_sharpness(sharpness);
                r.aspect_mode = cfg.aspect_mode;
                gl_renderer = Some(r);
            }
            Err(e) => {
                eprintln!("opengl: {}, falling back to SDL", e);
                use_gl = false;
                cfg.renderer = config::RendererBackend::Sdl;
                // On macOS GL mode, we have a raw window — need to rebuild as canvas
                if macos_gl_mode {
                    if let Some(w) = vk_window.take() {
                        _gl_ctx = None;
                        let c = w.into_canvas().accelerated().build()?;
                        canvas = Some(c);
                    }
                }
            }
        }
    }

    eprintln!("[{:7.3}s] GL init done (use_gl={})", t0.elapsed().as_secs_f64(), use_gl);

    let mut last_frame: Vec<u8> = Vec::new();
    let mut adjusted_frame: Vec<u8> = Vec::new();
    // Deferred snapshot action — avoid copying frame data every iteration.
    // Only snapshot when the user actually presses S or C.
    let mut pending_action: Option<u8> = None; // b'S' or b'C'
    let mut dirty = true;
    let mut paused = false;
    let mut auto_paused = false;     // true when paused by minimize/background (auto-unpause on restore/focus)
    let mut auto_paused_by_minimize = false; // distinguish minimize from background for wayland
    let mut audio_was_muted = false;  // track pre-pause mute state

    // Wayland minimize detection via wlr-foreign-toplevel-management protocol
    #[cfg(target_os = "linux")]
    let wl_minimize = if clipboard::is_wayland() {
        wayland_minimize::MinimizeWatcher::start(debug)
    } else {
        None
    };
    #[cfg(target_os = "linux")]
    let mut wl_was_minimized = false;
    let mut renderer_mismatch_shown = false;

    // Fullscreen toggle state: saved (x, y, w, h) from before entering fullscreen
    let mut saved_windowed_geom: Option<(i32, i32, u32, u32)> = None;
    let mut brightness: f32 = cfg.brightness as f32 / 100.0;
    let mut contrast: f32 = cfg.contrast as f32 / 100.0;
    let mut gamma: f32 = cfg.gamma as f32 / 100.0;
    let mut aspect_mode = cfg.aspect_mode;
    let mut cached_lut: Option<(f32, f32, f32, [u8; 256])> = None; // (brightness, contrast, gamma, lut)
    const BRIGHTNESS_STEP: f32 = 0.05;
    const BRIGHTNESS_MIN: f32 = 0.05;
    const BRIGHTNESS_MAX: f32 = 2.0;
    const _CONTRAST_MIN: f32 = 0.05;
    const _CONTRAST_MAX: f32 = 2.0;

    // Recording state
    let recording_dir = recording::video_dir();
    let mut recorder: Option<recording::Recorder> = None;

    // Streaming state
    let mut streamer: Option<stream_tx::StreamSender> = None;
    let mut stream_last_clients: u32 = 0;
    let mut stream_title_dirty = false;

    // Client receiver state (connect to remote server via menu)
    let mut client_receiver: Option<stream_rx::StreamReceiver> = None;
    let mut client_tex: Option<sdl2::render::Texture> = None;
    let mut client_dim: (u32, u32) = (0, 0);

    // Auto-start streaming server if --server was given
    if let Some(ref addr) = auto_stream {
        let bind = if addr.contains(':') {
            addr.clone()
        } else {
            format!("0.0.0.0:{}", cfg.stream_port)
        };
        match stream_tx::StreamSender::start(
            &bind, cap.width, cap.height, cfg.fps, eff_pixfmt, cfg.stream_quality, debug,
        ) {
            Ok(s) => {
                stream_title_dirty = true;
                streamer = Some(s);
                osd.set_action_label("Start Server", "Stop Server");
                update_streaming_osd(&mut osd, &streamer, &client_receiver, &cfg);
            }
            Err(e) => eprintln!("streaming: {}", e),
        }
    }

    #[cfg(target_os = "linux")]
    let mut vcam: Option<virtual_webcam::VirtualWebcam> = None;
    #[cfg(target_os = "linux")]
    if cfg.virtual_webcam {
        match virtual_webcam::VirtualWebcam::start(
            &cfg.virtual_webcam_device, cap.width, cap.height, eff_pixfmt,
        ) {
            Ok(v) => {
                eprintln!("virtual_webcam: started on {}", v.device_path());
                vcam = Some(v);
            }
            Err(e) => eprintln!("virtual_webcam: {}", e),
        }
    }

    #[cfg(target_os = "linux")]
    let mut vmic: Option<virtual_mic::VirtualMic> = None;
    #[cfg(target_os = "linux")]
    if cfg.virtual_mic {
        if let Some(ref a) = _audio {
            if a.mode() == config::AudioMode::Capture {
                match virtual_mic::VirtualMic::start(&cfg.virtual_mic_sink) {
                    Ok(v) => {
                        a.set_virtual_mic(Some(v.tee()));
                        eprintln!("virtual_mic: started ({})", v.monitor_source());
                        vmic = Some(v);
                    }
                    Err(e) => eprintln!("virtual_mic: {}", e),
                }
            } else {
                eprintln!("virtual_mic: requires audio_mode=capture; skipping");
            }
        } else {
            eprintln!("virtual_mic: no audio device configured; skipping");
        }
    }

    // Screenshot state
    let pictures_dir = screenshot::pictures_dir();
    let session_tar = pictures_dir.join(format!(
        "capview_{}.tar",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    ));

    // Analysis strip easter egg state (continued)
    let mut strip_mode: Option<analysis_strip::AnalysisStrip> = None;
    // Track last output mode used for strip captures (for finalize on exit)
    let mut strip_last_mode = analysis_strip::OutputMode::File;

    // FPS counter state: 0=off, 1=simple, 2=verbose
    let mut fps_show: u8 = match cfg.fps_display.as_str() {
        "simple" => 1,
        "verbose" => 2,
        _ => 0,
    };
    let mut fps_count: u32 = 0;
    let mut cap_fps_count: u32 = 0;
    let mut fps_last = std::time::Instant::now();
    let mut last_seq: Option<u32> = None;
    let mut seq_delta: u32 = 0;
    let mut upload_us_sum: u64 = 0;
    let mut upload_us_count: u32 = 0;
    let mut present_us_sum: u64 = 0;
    let mut present_us_count: u32 = 0;
    let mut unique_frame_count: u32 = 0;
    let mut last_frame_hash: u64 = 0;
    let mut stutter_count: u32 = 0;
    let mut total_dupes: u64 = 0;
    let mut last_present_time = std::time::Instant::now();
    let mut last_cpu_ticks: u64 = 0;
    // Read initial CPU ticks
    if let Ok(stat) = std::fs::read_to_string("/proc/self/stat") {
        let fields: Vec<&str> = stat.split_whitespace().collect();
        if fields.len() > 14 {
            let utime: u64 = fields[13].parse().unwrap_or(0);
            let stime: u64 = fields[14].parse().unwrap_or(0);
            last_cpu_ticks = utime + stime;
        }
    }
    let mut last_cpu_time = std::time::Instant::now();
    let mut cpu_pct: f64 = 0.0;
    // Frame timing jitter tracking (rolling window)
    let mut frame_gap_us_buf: [u64; 64] = [0; 64];
    let mut frame_gap_idx: usize = 0;
    let mut frame_gap_count: usize = 0;
    let mut last_capture_time = std::time::Instant::now();
    let mut max_frame_gap_us: u64 = 0;
    // CPU frequency monitoring (checked once per FPS update, not per frame)
    let mut cpu_freq_mhz: u32 = 0;
    let mut _cpu_max_freq_mhz: u32 = 0;
    let mut cpu_throttled = false;
    let mut last_repin_check = std::time::Instant::now();
    // Perf CSV: background thread drains a channel so the render loop never
    // blocks on disk I/O.  The channel is bounded (256 entries, ~25KB) — if the
    // writer can't keep up we drop rows rather than stall the renderer.
    let perf_tx: Option<std::sync::mpsc::SyncSender<[u64; 12]>> = cli.perf.as_ref().and_then(|path| {
        match std::fs::File::create(path) {
            Ok(f) => {
                let (tx, rx) = std::sync::mpsc::sync_channel::<[u64; 12]>(256);
                std::thread::Builder::new().name("perf-csv".into()).spawn(move || {
                    priority::avoid_render_core();
                    use std::io::Write;
                    let mut w = std::io::BufWriter::new(f);
                    let _ = writeln!(w, "timestamp_us,poll_us,dequeue_us,upload_us,render_us,present_us,frame_gap_us,cpu_freq_mhz,cpu_pct,jitter_us,throttled,drained");
                    while let Ok(row) = rx.recv() {
                        let _ = writeln!(w, "{},{},{},{},{},{},{},{},{},{},{},{}",
                            row[0], row[1], row[2], row[3], row[4], row[5],
                            row[6], row[7], row[8], row[9], row[10], row[11]);
                    }
                    let _ = w.flush();
                }).ok();
                eprintln!("perf: writing CSV to {}", path);
                Some(tx)
            }
            Err(e) => { eprintln!("perf: cannot create {}: {}", path, e); None }
        }
    });
    let perf_active = perf_tx.is_some();
    let perf_t0 = std::time::Instant::now();
    if fps_show > 0 {
        osd.pin(Slot::Fps, if fps_show == 2 { "Device:  --\nCapture: --\nRender:  --" } else { "-- / -- / -- fps" });
    }

    // Frame generation state — lazily initialized on first enable (G key)
    // to avoid allocating GPU resources (4 textures + FBO + compute shaders)
    // when frame gen is never used.
    let mut frame_gen: Option<framegen::FrameGen> = None;
    #[cfg(feature = "rife")]
    let mut rife_interp: Option<framegen::rife::RifeInterpolator> = None;
    let mut framegen_mode = match cfg.framegen_mode.as_str() {
        "extrapolate" => framegen::FrameGenMode::Extrapolate,
        "interpolate" => framegen::FrameGenMode::Interpolate,
        #[cfg(feature = "rife")]
        "rife" => framegen::FrameGenMode::Rife,
        _ => framegen::FrameGenMode::Off,
    };
    let mut framegen_quality = match cfg.framegen_quality.as_str() {
        "fast" => framegen::FrameGenQuality::Fast,
        "quality" => framegen::FrameGenQuality::Quality,
        _ => framegen::FrameGenQuality::Balanced,
    };
    // Track whether a real frame was uploaded this iteration
    let mut real_frame_this_tick: bool;
    // Throttle synthetic frame generation to ~display rate.
    // Deadline-based: advances by exact intervals to prevent drift.
    let mut next_synth_deadline = std::time::Instant::now();
    let mut _last_present = std::time::Instant::now();
    // Effective target FPS: config value, or 2× capture rate if auto (0).
    let mut target_fps: u32 = if cfg.target_fps > 0 {
        cfg.target_fps
    } else {
        (cfg.fps * 2).min(240)
    };
    let source_fps = cfg.fps.max(1);
    // Interval between presents when framegen active.
    let mut present_interval = std::time::Duration::from_micros(
        (1_000_000 / target_fps.max(1)) as u64
    );
    // Expected real frame interval from source.
    let _real_frame_interval = std::time::Duration::from_micros(
        (1_000_000 / source_fps) as u64
    );
    // Timestamp of last real frame arrival — used to compute interpolation `t`.
    let mut _last_real_frame = std::time::Instant::now();
    // Counter for synthetic frames since last real frame (for stable t).
    let mut synth_count_since_real: u32 = 0;
    // Number of synthetic frames expected per real frame interval.
    let mut synths_per_real = ((target_fps + source_fps.max(1) - 1) / source_fps.max(1)).max(1).saturating_sub(1);
    // Exponential moving average of present() time (µs) — used to detect
    // compositor throttling and adapt synthetic frame rate.
    let mut present_ema_us: f64 = 0.0;

    let v4l2_fd = cap.fd();
    // poll() fd for v4l2 readability — timeout = one frame interval so we
    // never spin or sleep longer than necessary.
    let poll_timeout_ms = (1000 / source_fps) as i32;

    // Apply process/thread priority optimizations (timer slack, RT scheduling,
    // CPU affinity, mlock).  Best-effort — failures are logged in debug mode.
    // Controlled by `priority` config key (default: all).
    let prio = cfg.priority;
    priority::apply_all(prio, debug);
    let _idle_inhibit = if prio.has(config::PriorityFlags::IDLE_INHIBIT) {
        priority::inhibit_idle(debug)
    } else { None };
    let _compositing_suspend = if prio.has(config::PriorityFlags::NO_COMPOSITOR) {
        priority::try_suspend_compositing(debug)
    } else { None };

    eprintln!("[{:7.3}s] entering main loop (poll_fd={} timeout={}ms)", t0.elapsed().as_secs_f64(), v4l2_fd, poll_timeout_ms);

    'main: loop {
        real_frame_this_tick = false;
        for event in event_pump.poll_iter() {
            if debug {
                if let Event::Window { win_event, .. } = &event {
                    let flags = with_window!(|w: &sdl2::video::Window| w.window_flags());
                    eprintln!("debug: SDL window event: {:?}  flags=0x{:08x}", win_event, flags);
                }
            }
            match event {
                Event::Quit { .. } => {
                    break 'main;
                }
                // ── Text field editing mode ─────────────────────────
                // When a text field is active, intercept most keys for text input.
                _ if osd.is_editing_text() => {
                    match event {
                        Event::KeyDown { keycode: Some(Keycode::Escape), .. } => {
                            osd.cancel_editing_text();
                            dirty = true;
                        }
                        Event::KeyDown { keycode: Some(Keycode::Return), .. } => {
                            osd.stop_editing_text();
                            dirty = true;
                        }
                        Event::KeyDown { keycode: Some(Keycode::Backspace), .. } => {
                            osd.text_backspace();
                            dirty = true;
                        }
                        Event::KeyDown { keycode: Some(Keycode::Left), .. } => {
                            osd.text_cursor_left();
                            dirty = true;
                        }
                        Event::KeyDown { keycode: Some(Keycode::Right), .. } => {
                            osd.text_cursor_right();
                            dirty = true;
                        }
                        Event::TextInput { text, .. } => {
                            for ch in text.chars() {
                                // Allow digits, dots, colons (for IPv6/port)
                                if ch.is_ascii_digit() || ch == '.' || ch == ':' {
                                    osd.text_insert(ch);
                                }
                            }
                            dirty = true;
                        }
                        _ => {}
                    }
                }
                // ── Normal key handling ─────────────────────────────
                Event::KeyDown { keycode: Some(Keycode::Q), .. } => {
                    if !osd.menu_open() { break 'main; }
                }
                Event::KeyDown { keycode: Some(Keycode::Escape), .. } => {
                    if osd.menu_open() {
                        osd.menu_back();
                        dirty = true;
                    }
                }
                Event::KeyDown { keycode: Some(Keycode::F12), keymod, .. } => {
                    if !osd.menu_open() {
                        let shift = keymod.intersects(Mod::LSHIFTMOD | Mod::RSHIFTMOD);
                        let ctrl = keymod.intersects(Mod::LCTRLMOD | Mod::RCTRLMOD);
                        if shift && ctrl {
                            pending_action = Some(b'T'); // tar
                        } else if shift {
                            pending_action = Some(b'S'); // save file
                        } else {
                            pending_action = Some(b'C'); // clipboard
                        }
                    }
                }
                Event::KeyDown { keycode: Some(Keycode::Tab), .. } => {
                    osd.toggle_menu();
                    dirty = true;
                }
                Event::KeyDown { keycode: Some(Keycode::F1), .. } => {
                    osd.toggle_help();
                    dirty = true;
                }
                Event::KeyDown { keycode: Some(Keycode::Up), .. } => {
                    if osd.menu_open() { osd.menu_up(); dirty = true; }
                }
                Event::KeyDown { keycode: Some(Keycode::Down), .. } => {
                    if osd.menu_open() { osd.menu_down(); dirty = true; }
                }
                Event::KeyDown { keycode: Some(Keycode::Left), keymod, .. } => {
                    if osd.menu_open() {
                        let old_fmt = cfg.screenshot_format;
                        let old_backend = cfg.renderer;
                        let step = if keymod.intersects(Mod::LSHIFTMOD | Mod::RSHIFTMOD) { 10 } else { 1 };
                        osd.menu_adjust(-step);
                        sync_menu_to_config(&osd, &mut cfg, cli.profile.as_deref(), &mut aspect_mode);
                        #[cfg(target_os = "linux")]
                        sync_vcam_menu(&mut osd, &mut vcam, &mut cfg, cli.profile.as_deref(),
                                       cap.width, cap.height, eff_pixfmt);
                        #[cfg(target_os = "linux")]
                        sync_vmic_menu(&mut osd, &mut vmic, _audio.as_ref(),
                                       &mut cfg, cli.profile.as_deref());
                        if cfg.screenshot_format != old_fmt {
                            osd.set_submenu_items("Screenshots", build_screenshot_items(&cfg));
                        }
                        if cfg.renderer != old_backend {
                            osd.set_submenu_items("Video", build_renderer_items(&cfg, framegen_mode, framegen_quality, target_fps, scale_mode, sharpness, active_backend, aspect_mode, brightness, contrast, gamma));
                        }
                        // Sync framegen menu
                        let (fg_m, fg_q, fg_tfps, fg_scale, fg_sharp) = read_framegen_menu(&osd, cfg.renderer == config::RendererBackend::Vulkan);
                        if let Some(m) = fg_m {
                            if m != framegen_mode {
                                framegen_mode = m;
                                #[cfg(feature = "rife")]
                                let is_rife = m == framegen::FrameGenMode::Rife;
                                #[cfg(not(feature = "rife"))]
                                #[allow(unused_variables)]
                                let is_rife = false;

                                if m != framegen::FrameGenMode::Off && frame_gen.is_none() && use_gl {
                                    frame_gen = framegen::FrameGen::new(
                                        |s| gl_proc_address(&video, s),
                                        cap.width, cap.height, debug,
                                    );
                                }
                                if m != framegen::FrameGenMode::Off && use_vk {
                                    if let Some(ref mut vk) = vk_renderer_inst {
                                        if !vk.fg_can_generate() { vk.enable_framegen(debug); }
                                    }
                                }
                                if let Some(ref mut fg) = frame_gen { fg.set_mode(m); }

                                #[cfg(feature = "rife")]
                                if is_rife && rife_interp.is_none() && use_gl {
                                    let model_path = rife_model_path();
                                    match framegen::rife::RifeInterpolator::new(
                                        |s| gl_proc_address(&video, s),
                                        &model_path, cap.width, cap.height,
                                    ) {
                                        Ok(r) => { rife_interp = Some(r); }
                                        Err(e) => {
                                            eprintln!("rife: failed to init: {}", e);
                                            osd.show(Slot::Transient, &format!("RIFE: {}", e), 3000);
                                            framegen_mode = framegen::FrameGenMode::Off;
                                        }
                                    }
                                }

                                if m == framegen::FrameGenMode::Off {
                                    frame_gen = None;
                                    #[cfg(feature = "rife")]
                                    { rife_interp = None; }
                                    if use_vk {
                                        if let Some(ref mut vk) = vk_renderer_inst { vk.disable_framegen(); }
                                    }
                                    if cfg.vsync {
                                        video.gl_set_swap_interval(sdl2::video::SwapInterval::VSync).ok();
                                    }
                                } else {
                                    set_framegen_swap(&video);
                                }
                                config::save_key(cli.profile.as_deref(), "framegen_mode", framegen_mode_str(framegen_mode));
                                if m != framegen::FrameGenMode::Off && !use_gl && !use_vk {
                                    osd.show(Slot::Transient,
                                        "Frame Gen needs GL or Vulkan backend", 2500);
                                }
                                // Rebuild submenu so FG Quality/Target FPS appear/disappear
                                osd.set_submenu_items("Video", build_renderer_items(&cfg, framegen_mode, framegen_quality, target_fps, scale_mode, sharpness, active_backend, aspect_mode, brightness, contrast, gamma));
                            }
                        }
                        if let Some(q) = fg_q {
                            if q != framegen_quality {
                                framegen_quality = q;
                                if let Some(ref mut fg) = frame_gen { fg.set_quality(q); }
                                config::save_key(cli.profile.as_deref(), "framegen_quality", framegen_quality_str(q));
                            }
                        }
                        if let Some(new_tfps) = fg_tfps {
                            if new_tfps != target_fps {
                                target_fps = if new_tfps > 0 { new_tfps } else { (source_fps * 2).min(240) };
                                present_interval = std::time::Duration::from_micros(
                                    (1_000_000 / target_fps.max(1)) as u64
                                );
                                synths_per_real = ((target_fps + source_fps.max(1) - 1) / source_fps.max(1)).max(1).saturating_sub(1);
                                cfg.target_fps = new_tfps;
                                config::save_key(cli.profile.as_deref(), "target_fps", &new_tfps.to_string());
                            }
                        }
                        if let Some(sm) = fg_scale {
                            if sm != scale_mode {
                                scale_mode = sm;
                                if let Some(ref mut gl) = gl_renderer {
                                    gl.set_scale_mode(sm);
                                }
                                if let Some(ref mut vk) = vk_renderer_inst {
                                    vk.set_scale_mode(sm);
                                }
                                config::save_key(cli.profile.as_deref(), "scale_mode", sm.config_name());
                                if sm.requires_shader() && !use_gl && !use_vk {
                                    osd.show(Slot::Transient,
                                        format!("{} needs GL or Vulkan backend", sm.label()), 2500);
                                }
                                // Rebuild to show/hide Sharpness item
                                osd.set_submenu_items("Video", build_renderer_items(&cfg, framegen_mode, framegen_quality, target_fps, scale_mode, sharpness, active_backend, aspect_mode, brightness, contrast, gamma));
                            }
                        }
                        if let Some(s) = fg_sharp {
                            if s != sharpness {
                                sharpness = s;
                                if let Some(ref mut gl) = gl_renderer {
                                    gl.set_sharpness(s);
                                }
                                if let Some(ref mut vk) = vk_renderer_inst {
                                    vk.set_sharpness(s);
                                }
                                cfg.sharpness = s;
                                config::save_key(cli.profile.as_deref(), "sharpness", &s.to_string());
                            }
                        }
                        if let Some(ref mut gl) = gl_renderer { gl.aspect_mode = aspect_mode; }
                        if let Some(ref mut vk) = vk_renderer_inst { vk.aspect_mode = aspect_mode; }
                        // Sync brightness/contrast from menu
                        if let Some((idx, _)) = osd.find_menu_value("Brightness") {
                            let new_b = (idx as f32 + 1.0) * 0.05;
                            if (new_b - brightness).abs() > 0.001 {
                                brightness = new_b;
                                cached_lut = None;
                                cfg.brightness = (new_b * 100.0).round() as u32;
                                config::save_key(cli.profile.as_deref(), "brightness", &cfg.brightness.to_string());
                                osd.show(Slot::Transient, format!("Brightness: {:.0}%", new_b * 100.0), 1000);
                            }
                        }
                        if let Some((idx, _)) = osd.find_menu_value("Contrast") {
                            let new_c = (idx as f32 + 1.0) * 0.05;
                            if (new_c - contrast).abs() > 0.001 {
                                contrast = new_c;
                                cached_lut = None;
                                cfg.contrast = (new_c * 100.0).round() as u32;
                                config::save_key(cli.profile.as_deref(), "contrast", &cfg.contrast.to_string());
                                osd.show(Slot::Transient, format!("Contrast: {:.0}%", new_c * 100.0), 1000);
                            }
                        }
                        if let Some((idx, _)) = osd.find_menu_value("Gamma") {
                            let new_g = (idx as f32 + 1.0) * 0.1;
                            if (new_g - gamma).abs() > 0.001 {
                                gamma = new_g;
                                cached_lut = None;
                                cfg.gamma = (new_g * 100.0).round() as u32;
                                config::save_key(cli.profile.as_deref(), "gamma", &cfg.gamma.to_string());
                                osd.show(Slot::Transient, format!("Gamma: {:.1}", new_g), 1000);
                            }
                        }
                        // Sync audio mode from menu
                        if let Some(ref mut a) = _audio {
                            if let Some((idx, _)) = osd.find_menu_value("Audio Mode") {
                                let new_mode = match idx {
                                    1 => config::AudioMode::Passthrough,
                                    _ => config::AudioMode::Capture,
                                };
                                if new_mode != a.mode() {
                                    match a.set_mode(new_mode) {
                                        Ok(()) => {
                                            cfg.audio_mode = new_mode;
                                            let s = match new_mode {
                                                config::AudioMode::Capture => "capture",
                                                config::AudioMode::Passthrough => "passthrough",
                                            };
                                            config::save_key(cli.profile.as_deref(), "audio_mode", s);
                                            osd.set_submenu_items("Audio", build_audio_items(&cfg, a.volume(), a.is_muted(), a.capture_buf_ms(), a.playback_buf_ms()));
                                            osd.show(Slot::Transient, format!("Audio: {} mode", s), 1500);
                                        }
                                        Err(e) => {
                                            eprintln!("audio mode switch failed: {}", e);
                                            osd.show(Slot::Transient, format!("Audio mode error: {}", e), 2500);
                                        }
                                    }
                                }
                            }
                        }
                        // Sync audio menu → audio state (capture mode controls)
                        if let Some(ref mut a) = _audio {
                            if let Some((idx, _)) = osd.find_menu_value("Volume") {
                                let new_vol = (idx as i32) * 5;
                                if new_vol != a.volume() {
                                    a.set_volume(new_vol);
                                    osd.show(Slot::Transient, format!("Volume: {}%", new_vol), 1000);
                                }
                            }
                            if let Some((idx, _)) = osd.find_menu_value("Mute") {
                                let want_muted = idx == 1;
                                if want_muted != a.is_muted() {
                                    a.set_muted(want_muted);
                                    osd.show(Slot::Transient, if want_muted { "Audio: Muted" } else { "Audio: Unmuted" }, 1500);
                                }
                            }
                            let mut new_cap = a.capture_buf_ms();
                            let mut new_play = a.playback_buf_ms();
                            if let Some((idx, _)) = osd.find_menu_value("Capture Buffer") {
                                if let Some(&ms) = AUDIO_BUF_OPTIONS.get(idx) {
                                    new_cap = ms;
                                }
                            }
                            if let Some((idx, _)) = osd.find_menu_value("Playback Buffer") {
                                if let Some(&ms) = AUDIO_BUF_OPTIONS.get(idx) {
                                    new_play = ms;
                                }
                            }
                            if new_cap != a.capture_buf_ms() || new_play != a.playback_buf_ms() {
                                a.set_buffers(new_cap, new_play);
                                config::save_key(cli.profile.as_deref(), "audio_capture_buf", &new_cap.to_string());
                                config::save_key(cli.profile.as_deref(), "audio_playback_buf", &new_play.to_string());
                                osd.show(Slot::Transient, format!("Audio: {}ms cap / {}ms play", new_cap, new_play), 1500);
                            }
                        }
                        // Sync FPS display mode from menu
                        if let Some((idx, _)) = osd.find_menu_value("FPS") {
                            let new_fps_show = idx as u8;
                            if new_fps_show != fps_show {
                                fps_show = new_fps_show;
                                cfg.fps_display = match fps_show { 1 => "simple", 2 => "verbose", _ => "off" }.to_string();
                                config::save_key(cli.profile.as_deref(), "fps_display", &cfg.fps_display);
                                if fps_show > 0 {
                                    fps_count = 0; cap_fps_count = 0; seq_delta = 0; last_seq = None;
                                    upload_us_sum = 0; upload_us_count = 0;
                                    present_us_sum = 0; present_us_count = 0;
                                    unique_frame_count = 0; stutter_count = 0;
                                    last_present_time = std::time::Instant::now();
                                    fps_last = std::time::Instant::now();
                                    osd.pin(Slot::Fps, if fps_show == 2 { "Device:  --\nCapture: --\nRender:  --" } else { "-- / -- / -- fps" });
                                } else {
                                    osd.clear(Slot::Fps);
                                }
                            }
                        }
                        if let Some((idx, _)) = osd.find_menu_value("OSD Opacity") {
                            let new_opacity = idx as u32 * 5;
                            if new_opacity != cfg.osd_opacity {
                                cfg.osd_opacity = new_opacity;
                                osd.set_opacity(new_opacity);
                                config::save_key(cli.profile.as_deref(), "osd_opacity", &new_opacity.to_string());
                            }
                        }
                        dirty = true;
                    }
                }
                Event::KeyDown { keycode: Some(Keycode::Right), keymod, .. } => {
                    if osd.menu_open() {
                        let old_fmt = cfg.screenshot_format;
                        let old_backend = cfg.renderer;
                        let step = if keymod.intersects(Mod::LSHIFTMOD | Mod::RSHIFTMOD) { 10 } else { 1 };
                        osd.menu_right_by(step);
                        sync_menu_to_config(&osd, &mut cfg, cli.profile.as_deref(), &mut aspect_mode);
                        #[cfg(target_os = "linux")]
                        sync_vcam_menu(&mut osd, &mut vcam, &mut cfg, cli.profile.as_deref(),
                                       cap.width, cap.height, eff_pixfmt);
                        #[cfg(target_os = "linux")]
                        sync_vmic_menu(&mut osd, &mut vmic, _audio.as_ref(),
                                       &mut cfg, cli.profile.as_deref());
                        if cfg.screenshot_format != old_fmt {
                            osd.set_submenu_items("Screenshots", build_screenshot_items(&cfg));
                        }
                        if cfg.renderer != old_backend {
                            osd.set_submenu_items("Video", build_renderer_items(&cfg, framegen_mode, framegen_quality, target_fps, scale_mode, sharpness, active_backend, aspect_mode, brightness, contrast, gamma));
                        }
                        // Sync framegen menu
                        let (fg_m, fg_q, fg_tfps, fg_scale, fg_sharp) = read_framegen_menu(&osd, cfg.renderer == config::RendererBackend::Vulkan);
                        if let Some(m) = fg_m {
                            if m != framegen_mode {
                                framegen_mode = m;
                                #[cfg(feature = "rife")]
                                let is_rife = m == framegen::FrameGenMode::Rife;
                                #[cfg(not(feature = "rife"))]
                                #[allow(unused_variables)]
                                let is_rife = false;

                                if m != framegen::FrameGenMode::Off && frame_gen.is_none() && use_gl {
                                    frame_gen = framegen::FrameGen::new(
                                        |s| gl_proc_address(&video, s),
                                        cap.width, cap.height, debug,
                                    );
                                }
                                if m != framegen::FrameGenMode::Off && use_vk {
                                    if let Some(ref mut vk) = vk_renderer_inst {
                                        if !vk.fg_can_generate() { vk.enable_framegen(debug); }
                                    }
                                }
                                if let Some(ref mut fg) = frame_gen { fg.set_mode(m); }

                                #[cfg(feature = "rife")]
                                if is_rife && rife_interp.is_none() && use_gl {
                                    let model_path = rife_model_path();
                                    match framegen::rife::RifeInterpolator::new(
                                        |s| gl_proc_address(&video, s),
                                        &model_path, cap.width, cap.height,
                                    ) {
                                        Ok(r) => { rife_interp = Some(r); }
                                        Err(e) => {
                                            eprintln!("rife: failed to init: {}", e);
                                            osd.show(Slot::Transient, &format!("RIFE: {}", e), 3000);
                                            framegen_mode = framegen::FrameGenMode::Off;
                                        }
                                    }
                                }

                                if m == framegen::FrameGenMode::Off {
                                    frame_gen = None;
                                    #[cfg(feature = "rife")]
                                    { rife_interp = None; }
                                    if use_vk {
                                        if let Some(ref mut vk) = vk_renderer_inst { vk.disable_framegen(); }
                                    }
                                    if cfg.vsync {
                                        video.gl_set_swap_interval(sdl2::video::SwapInterval::VSync).ok();
                                    }
                                } else {
                                    set_framegen_swap(&video);
                                }
                                config::save_key(cli.profile.as_deref(), "framegen_mode", framegen_mode_str(framegen_mode));
                                if m != framegen::FrameGenMode::Off && !use_gl && !use_vk {
                                    osd.show(Slot::Transient,
                                        "Frame Gen needs GL or Vulkan backend", 2500);
                                }
                                osd.set_submenu_items("Video", build_renderer_items(&cfg, framegen_mode, framegen_quality, target_fps, scale_mode, sharpness, active_backend, aspect_mode, brightness, contrast, gamma));
                            }
                        }
                        if let Some(q) = fg_q {
                            if q != framegen_quality {
                                framegen_quality = q;
                                if let Some(ref mut fg) = frame_gen { fg.set_quality(q); }
                                config::save_key(cli.profile.as_deref(), "framegen_quality", framegen_quality_str(q));
                            }
                        }
                        if let Some(new_tfps) = fg_tfps {
                            if new_tfps != target_fps {
                                target_fps = if new_tfps > 0 { new_tfps } else { (source_fps * 2).min(240) };
                                present_interval = std::time::Duration::from_micros(
                                    (1_000_000 / target_fps.max(1)) as u64
                                );
                                synths_per_real = ((target_fps + source_fps.max(1) - 1) / source_fps.max(1)).max(1).saturating_sub(1);
                                cfg.target_fps = new_tfps;
                                config::save_key(cli.profile.as_deref(), "target_fps", &new_tfps.to_string());
                            }
                        }
                        if let Some(sm) = fg_scale {
                            if sm != scale_mode {
                                scale_mode = sm;
                                if let Some(ref mut gl) = gl_renderer {
                                    gl.set_scale_mode(sm);
                                }
                                if let Some(ref mut vk) = vk_renderer_inst {
                                    vk.set_scale_mode(sm);
                                }
                                config::save_key(cli.profile.as_deref(), "scale_mode", sm.config_name());
                                if sm.requires_shader() && !use_gl && !use_vk {
                                    osd.show(Slot::Transient,
                                        format!("{} needs GL or Vulkan backend", sm.label()), 2500);
                                }
                                // Rebuild to show/hide Sharpness item
                                osd.set_submenu_items("Video", build_renderer_items(&cfg, framegen_mode, framegen_quality, target_fps, scale_mode, sharpness, active_backend, aspect_mode, brightness, contrast, gamma));
                            }
                        }
                        if let Some(s) = fg_sharp {
                            if s != sharpness {
                                sharpness = s;
                                if let Some(ref mut gl) = gl_renderer {
                                    gl.set_sharpness(s);
                                }
                                if let Some(ref mut vk) = vk_renderer_inst {
                                    vk.set_sharpness(s);
                                }
                                cfg.sharpness = s;
                                config::save_key(cli.profile.as_deref(), "sharpness", &s.to_string());
                            }
                        }
                        if let Some(ref mut gl) = gl_renderer { gl.aspect_mode = aspect_mode; }
                        if let Some(ref mut vk) = vk_renderer_inst { vk.aspect_mode = aspect_mode; }
                        // Sync brightness/contrast from menu
                        if let Some((idx, _)) = osd.find_menu_value("Brightness") {
                            let new_b = (idx as f32 + 1.0) * 0.05;
                            if (new_b - brightness).abs() > 0.001 {
                                brightness = new_b;
                                cached_lut = None;
                                cfg.brightness = (new_b * 100.0).round() as u32;
                                config::save_key(cli.profile.as_deref(), "brightness", &cfg.brightness.to_string());
                                osd.show(Slot::Transient, format!("Brightness: {:.0}%", new_b * 100.0), 1000);
                            }
                        }
                        if let Some((idx, _)) = osd.find_menu_value("Contrast") {
                            let new_c = (idx as f32 + 1.0) * 0.05;
                            if (new_c - contrast).abs() > 0.001 {
                                contrast = new_c;
                                cached_lut = None;
                                cfg.contrast = (new_c * 100.0).round() as u32;
                                config::save_key(cli.profile.as_deref(), "contrast", &cfg.contrast.to_string());
                                osd.show(Slot::Transient, format!("Contrast: {:.0}%", new_c * 100.0), 1000);
                            }
                        }
                        if let Some((idx, _)) = osd.find_menu_value("Gamma") {
                            let new_g = (idx as f32 + 1.0) * 0.1;
                            if (new_g - gamma).abs() > 0.001 {
                                gamma = new_g;
                                cached_lut = None;
                                cfg.gamma = (new_g * 100.0).round() as u32;
                                config::save_key(cli.profile.as_deref(), "gamma", &cfg.gamma.to_string());
                                osd.show(Slot::Transient, format!("Gamma: {:.1}", new_g), 1000);
                            }
                        }
                        // Sync audio mode from menu
                        if let Some(ref mut a) = _audio {
                            if let Some((idx, _)) = osd.find_menu_value("Audio Mode") {
                                let new_mode = match idx {
                                    1 => config::AudioMode::Passthrough,
                                    _ => config::AudioMode::Capture,
                                };
                                if new_mode != a.mode() {
                                    match a.set_mode(new_mode) {
                                        Ok(()) => {
                                            cfg.audio_mode = new_mode;
                                            let s = match new_mode {
                                                config::AudioMode::Capture => "capture",
                                                config::AudioMode::Passthrough => "passthrough",
                                            };
                                            config::save_key(cli.profile.as_deref(), "audio_mode", s);
                                            osd.set_submenu_items("Audio", build_audio_items(&cfg, a.volume(), a.is_muted(), a.capture_buf_ms(), a.playback_buf_ms()));
                                            osd.show(Slot::Transient, format!("Audio: {} mode", s), 1500);
                                        }
                                        Err(e) => {
                                            eprintln!("audio mode switch failed: {}", e);
                                            osd.show(Slot::Transient, format!("Audio mode error: {}", e), 2500);
                                        }
                                    }
                                }
                            }
                        }
                        // Sync audio menu → audio state (capture mode controls)
                        if let Some(ref mut a) = _audio {
                            if let Some((idx, _)) = osd.find_menu_value("Volume") {
                                let new_vol = (idx as i32) * 5;
                                if new_vol != a.volume() {
                                    a.set_volume(new_vol);
                                    osd.show(Slot::Transient, format!("Volume: {}%", new_vol), 1000);
                                }
                            }
                            if let Some((idx, _)) = osd.find_menu_value("Mute") {
                                let want_muted = idx == 1;
                                if want_muted != a.is_muted() {
                                    a.set_muted(want_muted);
                                    osd.show(Slot::Transient, if want_muted { "Audio: Muted" } else { "Audio: Unmuted" }, 1500);
                                }
                            }
                            let mut new_cap = a.capture_buf_ms();
                            let mut new_play = a.playback_buf_ms();
                            if let Some((idx, _)) = osd.find_menu_value("Capture Buffer") {
                                if let Some(&ms) = AUDIO_BUF_OPTIONS.get(idx) {
                                    new_cap = ms;
                                }
                            }
                            if let Some((idx, _)) = osd.find_menu_value("Playback Buffer") {
                                if let Some(&ms) = AUDIO_BUF_OPTIONS.get(idx) {
                                    new_play = ms;
                                }
                            }
                            if new_cap != a.capture_buf_ms() || new_play != a.playback_buf_ms() {
                                a.set_buffers(new_cap, new_play);
                                config::save_key(cli.profile.as_deref(), "audio_capture_buf", &new_cap.to_string());
                                config::save_key(cli.profile.as_deref(), "audio_playback_buf", &new_play.to_string());
                                osd.show(Slot::Transient, format!("Audio: {}ms cap / {}ms play", new_cap, new_play), 1500);
                            }
                        }
                        // Sync FPS display mode from menu
                        if let Some((idx, _)) = osd.find_menu_value("FPS") {
                            let new_fps_show = idx as u8;
                            if new_fps_show != fps_show {
                                fps_show = new_fps_show;
                                cfg.fps_display = match fps_show { 1 => "simple", 2 => "verbose", _ => "off" }.to_string();
                                config::save_key(cli.profile.as_deref(), "fps_display", &cfg.fps_display);
                                if fps_show > 0 {
                                    fps_count = 0; cap_fps_count = 0; seq_delta = 0; last_seq = None;
                                    upload_us_sum = 0; upload_us_count = 0;
                                    present_us_sum = 0; present_us_count = 0;
                                    unique_frame_count = 0; stutter_count = 0;
                                    last_present_time = std::time::Instant::now();
                                    fps_last = std::time::Instant::now();
                                    osd.pin(Slot::Fps, if fps_show == 2 { "Device:  --\nCapture: --\nRender:  --" } else { "-- / -- / -- fps" });
                                } else {
                                    osd.clear(Slot::Fps);
                                }
                            }
                        }
                        if let Some((idx, _)) = osd.find_menu_value("OSD Opacity") {
                            let new_opacity = idx as u32 * 5;
                            if new_opacity != cfg.osd_opacity {
                                cfg.osd_opacity = new_opacity;
                                osd.set_opacity(new_opacity);
                                config::save_key(cli.profile.as_deref(), "osd_opacity", &new_opacity.to_string());
                            }
                        }
                        dirty = true;
                    }
                }
                Event::KeyDown { keycode: Some(Keycode::Return), .. } => {
                    if osd.menu_open() {
                        let action = osd.menu_action_label();
                        if debug {
                            eprintln!("debug: menu enter: action={:?}", action);
                        }
                        match action.as_deref() {
                            Some("Start Server") => {
                                let bind = format!("0.0.0.0:{}", cfg.stream_port);
                                match stream_tx::StreamSender::start(
                                    &bind, cap.width, cap.height, cfg.fps,
                                    eff_pixfmt, cfg.stream_quality, debug,
                                ) {
                                    Ok(s) => {
                                        let port = s.port();
                                        streamer = Some(s);
                                        osd.set_action_label("Start Server", "Stop Server");
                                        eprintln!("streaming: server started on :{}", port);
                                        osd.show(Slot::Transient, format!("Server on :{}", port), 1500);
                                        stream_title_dirty = true;
                                        update_streaming_osd(&mut osd, &streamer, &client_receiver, &cfg);
                                    }
                                    Err(e) => {
                                        eprintln!("streaming: {}", e);
                                        osd.show(Slot::Transient, "Server failed", 1500);
                                    }
                                }
                            }
                            Some("Stop Server") => {
                                if let Some(mut s) = streamer.take() {
                                    s.stop();
                                }
                                osd.set_action_label("Stop Server", "Start Server");
                                eprintln!("streaming: server stopped");
                                osd.show(Slot::Transient, "Server stopped", 1500);
                                stream_title_dirty = true;
                                update_streaming_osd(&mut osd, &streamer, &client_receiver, &cfg);
                            }
                            Some("Connect") => {
                                let ip_str = osd.find_menu_text("Address")
                                    .unwrap_or_else(|| "192.168.1.1".to_string());
                                let port_str = osd.find_menu_text("Port")
                                    .unwrap_or_else(|| "9000".to_string());
                                let port: u16 = match port_str.trim().parse() {
                                    Ok(p) if p > 0 => p,
                                    _ => {
                                        osd.show(Slot::Transient, "Invalid port", 1500);
                                        dirty = true;
                                        continue;
                                    }
                                };
                                let addr = format!("{}:{}", ip_str.trim(), port);
                                // Validate IP by trying to parse the full address
                                if addr.parse::<std::net::SocketAddr>().is_err() {
                                    osd.show(Slot::Transient, format!("Bad address: {}", addr), 2000);
                                    dirty = true;
                                    continue;
                                }
                                eprintln!("streaming: connect to {}", addr);
                                match stream_rx::StreamReceiver::start(&addr, debug) {
                                    Ok(rx) => {
                                        client_receiver = Some(rx);
                                        osd.set_action_label("Connect", "Disconnect");
                                        eprintln!("streaming: receiver started for {}", addr);
                                        osd.show(Slot::Transient, format!("Connecting to {}", addr), 2000);
                                        stream_title_dirty = true;
                                        // Persist last-used address
                                        let parts: Vec<&str> = ip_str.trim().split('.').collect();
                                        if parts.len() == 4 {
                                            if let (Ok(a), Ok(b), Ok(c), Ok(d)) = (
                                                parts[0].parse::<u8>(), parts[1].parse::<u8>(),
                                                parts[2].parse::<u8>(), parts[3].parse::<u8>(),
                                            ) {
                                                cfg.stream_client_ip = [a, b, c, d];
                                            }
                                        }
                                        cfg.stream_client_port = port;
                                        config::save_key(cli.profile.as_deref(), "stream_client_ip",
                                            ip_str.trim());
                                        config::save_key(cli.profile.as_deref(), "stream_client_port",
                                            &port.to_string());
                                        update_streaming_osd(&mut osd, &streamer, &client_receiver, &cfg);
                                    }
                                    Err(e) => {
                                        eprintln!("streaming: connect failed: {}", e);
                                        osd.show(Slot::Transient, "Connect failed", 1500);
                                    }
                                }
                            }
                            Some("Disconnect") => {
                                if let Some(mut rx) = client_receiver.take() {
                                    rx.stop();
                                }
                                client_tex = None;
                                client_dim = (0, 0);
                                osd.set_action_label("Disconnect", "Connect");
                                eprintln!("streaming: disconnected");
                                osd.show(Slot::Transient, "Disconnected", 1500);
                                stream_title_dirty = true;
                                update_streaming_osd(&mut osd, &streamer, &client_receiver, &cfg);
                            }
                            _ => {
                                // Try entering a submenu, or start editing a text field
                                if !osd.menu_enter() {
                                    osd.start_editing_text();
                                }
                            }
                        }
                        dirty = true;
                    }
                }
                Event::KeyDown { keycode: Some(Keycode::Equals), .. }
                | Event::KeyDown { keycode: Some(Keycode::Plus), .. }
                | Event::KeyDown { keycode: Some(Keycode::KpPlus), .. } => {
                    brightness = (brightness + BRIGHTNESS_STEP).min(BRIGHTNESS_MAX);
                    cached_lut = None;
                    cfg.brightness = (brightness * 100.0).round() as u32;
                    config::save_key(cli.profile.as_deref(), "brightness", &cfg.brightness.to_string());
                    eprintln!("brightness: {:.0}%", brightness * 100.0);
                    osd.show(Slot::Transient, format!("Brightness: {:.0}%", brightness * 100.0), 1500);
                    osd.set_submenu_items("Video", build_renderer_items(&cfg, framegen_mode, framegen_quality, target_fps, scale_mode, sharpness, active_backend, aspect_mode, brightness, contrast, gamma));
                    dirty = true;
                }
                Event::KeyDown { keycode: Some(Keycode::Minus), .. }
                | Event::KeyDown { keycode: Some(Keycode::KpMinus), .. } => {
                    brightness = (brightness - BRIGHTNESS_STEP).max(BRIGHTNESS_MIN);
                    cached_lut = None;
                    cfg.brightness = (brightness * 100.0).round() as u32;
                    config::save_key(cli.profile.as_deref(), "brightness", &cfg.brightness.to_string());
                    eprintln!("brightness: {:.0}%", brightness * 100.0);
                    osd.show(Slot::Transient, format!("Brightness: {:.0}%", brightness * 100.0), 1500);
                    osd.set_submenu_items("Video", build_renderer_items(&cfg, framegen_mode, framegen_quality, target_fps, scale_mode, sharpness, active_backend, aspect_mode, brightness, contrast, gamma));
                    dirty = true;
                }
                Event::KeyDown { keycode: Some(Keycode::F), .. } => {
                    with_window_mut!(|win: &mut sdl2::video::Window| {
                        use sdl2::video::FullscreenType;
                        if win.fullscreen_state() == FullscreenType::Off {
                            let pos = win.position();
                            let sz = win.size();
                            saved_windowed_geom = Some((pos.0, pos.1, sz.0, sz.1));
                            let _ = win.set_fullscreen(FullscreenType::Desktop);
                        } else {
                            let _ = win.set_fullscreen(FullscreenType::Off);
                            if let Some((x, y, w, h)) = saved_windowed_geom {
                                let _ = win.set_size(w, h);
                                let _ = win.set_position(
                                    sdl2::video::WindowPos::Positioned(x),
                                    sdl2::video::WindowPos::Positioned(y),
                                );
                            } else {
                                let _ = win.set_size(init_w, init_h);
                                let _ = win.set_position(
                                    sdl2::video::WindowPos::Centered,
                                    sdl2::video::WindowPos::Centered,
                                );
                            }
                        }
                    });
                    dirty = true;
                }
                Event::Window { win_event: sdl2::event::WindowEvent::Resized(w, h), .. } => {
                    if aspect_mode == config::AspectMode::Preserve {
                        with_window_mut!(|win: &mut sdl2::video::Window| {
                            enforce_aspect(win, cap.width, cap.height, w, h);
                        });
                    }
                    dirty = true;
                }
                // X11 only: Minimized/Restored fire on minimize/restore.
                // Wayland never fires these — minimize is only visible as FocusLost.
                Event::Window { win_event: sdl2::event::WindowEvent::Minimized, .. }
                | Event::Window { win_event: sdl2::event::WindowEvent::Hidden, .. } => {
                    if cfg.pause_on_minimize && !paused {
                        paused = true;
                        auto_paused = true;
                        audio_was_muted = _audio.as_ref().map_or(true, |a| a.is_muted());
                        if let Some(ref mut a) = _audio {
                            if !audio_was_muted { a.set_muted(true); }
                        }
                        osd.set_bottom_right(Some("Paused".into()));
                        if use_vk {
                            if let Some(ref mut vk) = vk_renderer_inst {
                                vk.dim_staging(0.7);
                            }
                        }
                        dirty = true;
                    }
                }
                Event::Window { win_event: sdl2::event::WindowEvent::Restored, .. }
                | Event::Window { win_event: sdl2::event::WindowEvent::Shown, .. } => {
                    if paused && auto_paused {
                        paused = false;
                        auto_paused = false;
                        auto_paused_by_minimize = false;
                        if let Some(ref mut a) = _audio {
                            if !audio_was_muted { a.set_muted(false); }
                        }
                        osd.set_bottom_right(None);
                        if use_vk {
                            if let Some(ref mut vk) = vk_renderer_inst {
                                vk.restore_staging();
                            }
                        }
                        dirty = true;
                    }
                }
                Event::Window { win_event: sdl2::event::WindowEvent::FocusLost, .. } => {
                    // On Wayland without foreign-toplevel protocol (e.g. KDE Plasma 5.x),
                    // minimize is indistinguishable from alt-tab. Pause on Minimize
                    // falls back to FocusLost in that case.
                    #[cfg(target_os = "linux")]
                    let minimize_fallback = cfg.pause_on_minimize && wl_minimize.is_none();
                    #[cfg(not(target_os = "linux"))]
                    let minimize_fallback = false;
                    if (cfg.pause_on_background || minimize_fallback) && !paused {
                        paused = true;
                        auto_paused = true;
                        audio_was_muted = _audio.as_ref().map_or(true, |a| a.is_muted());
                        if let Some(ref mut a) = _audio {
                            if !audio_was_muted { a.set_muted(true); }
                        }
                        osd.set_bottom_right(Some("Paused".into()));
                        if use_vk {
                            if let Some(ref mut vk) = vk_renderer_inst {
                                vk.dim_staging(0.7);
                            }
                        }
                        dirty = true;
                    }
                }
                Event::Window { win_event: sdl2::event::WindowEvent::FocusGained, .. } => {
                    // Don't unpause if wlr watcher still reports minimized
                    #[cfg(target_os = "linux")]
                    let wlr_still_minimized = wl_minimize.as_ref().map_or(false, |w| w.is_minimized());
                    #[cfg(not(target_os = "linux"))]
                    let wlr_still_minimized = false;
                    if paused && auto_paused && !wlr_still_minimized {
                        paused = false;
                        auto_paused = false;
                        auto_paused_by_minimize = false;
                        if let Some(ref mut a) = _audio {
                            if !audio_was_muted { a.set_muted(false); }
                        }
                        osd.set_bottom_right(None);
                        if use_vk {
                            if let Some(ref mut vk) = vk_renderer_inst {
                                vk.restore_staging();
                            }
                        }
                        dirty = true;
                    }
                }
                Event::KeyDown { keycode: Some(Keycode::M), .. } => {
                    if let Some(ref mut a) = _audio {
                        let muted = a.toggle_mute();
                        eprintln!("audio: {}", if muted { "muted" } else { "unmuted" });
                        osd.show(Slot::Transient, if muted { "Audio: Muted" } else { "Audio: Unmuted" }, 1500);
                        osd.set_submenu_items("Audio", build_audio_items(&cfg, a.volume(), muted, a.capture_buf_ms(), a.playback_buf_ms()));
                        dirty = true;
                    }
                }
                Event::KeyDown { keycode: Some(Keycode::P), .. } => {
                    paused = !paused;
                    auto_paused = false; // manual toggle clears auto-pause
                    auto_paused_by_minimize = false;
                    if paused {
                        audio_was_muted = _audio.as_ref().map_or(true, |a| a.is_muted());
                        if let Some(ref mut a) = _audio {
                            if !audio_was_muted { a.set_muted(true); }
                        }
                        osd.set_bottom_right(Some("Paused".into()));
                        // Dim VK staging for non-compute path
                        if use_vk {
                            if let Some(ref mut vk) = vk_renderer_inst {
                                vk.dim_staging(0.7);
                            }
                        }
                    } else {
                        if let Some(ref mut a) = _audio {
                            if !audio_was_muted { a.set_muted(false); }
                        }
                        osd.set_bottom_right(None);
                        // Restore VK staging for non-compute path
                        if use_vk {
                            if let Some(ref mut vk) = vk_renderer_inst {
                                vk.restore_staging();
                            }
                        }
                    }
                    dirty = true;
                }
                // Frame generation mode toggle (Off → Extrapolate → Interpolate → Off)
                Event::KeyDown { keycode: Some(Keycode::G), .. } => {
                    if use_gl || use_vk {
                        let new_mode = framegen_mode.next();
                        let mut can_enable = true;

                        #[cfg(feature = "rife")]
                        let is_rife = new_mode == framegen::FrameGenMode::Rife;
                        #[cfg(not(feature = "rife"))]
                        #[allow(unused_variables)]
                        let is_rife = false;

                        if new_mode != framegen::FrameGenMode::Off && frame_gen.is_none() && use_gl {
                            frame_gen = framegen::FrameGen::new(
                                |s| gl_proc_address(&video, s),
                                cap.width, cap.height, debug,
                            );
                            if frame_gen.is_none() {
                                can_enable = false;
                            }
                        }
                        if new_mode != framegen::FrameGenMode::Off && use_vk {
                            if let Some(ref mut vk) = vk_renderer_inst {
                                if !vk.enable_framegen(debug) {
                                    can_enable = false;
                                }
                            }
                        }

                        #[cfg(feature = "rife")]
                        if is_rife && rife_interp.is_none() && use_gl {
                            let model_path = rife_model_path();
                            match framegen::rife::RifeInterpolator::new(
                                |s| gl_proc_address(&video, s),
                                &model_path, cap.width, cap.height,
                            ) {
                                Ok(r) => { rife_interp = Some(r); }
                                Err(e) => {
                                    eprintln!("rife: failed to init: {}", e);
                                    osd.show(Slot::Transient, &format!("RIFE: {}", e), 3000);
                                    can_enable = false;
                                }
                            }
                        }

                        if new_mode == framegen::FrameGenMode::Off || can_enable {
                            framegen_mode = new_mode;
                            if let Some(ref mut fg) = frame_gen {
                                fg.set_mode(new_mode);
                            }
                            if new_mode == framegen::FrameGenMode::Off {
                                frame_gen = None;
                                #[cfg(feature = "rife")]
                                { rife_interp = None; }
                                if use_vk {
                                    if let Some(ref mut vk) = vk_renderer_inst { vk.disable_framegen(); }
                                }
                                // Restore vsync if configured
                                if cfg.vsync {
                                    video.gl_set_swap_interval(sdl2::video::SwapInterval::VSync).ok();
                                }
                            } else {
                                // Disable vsync so present() doesn't block — we pace frames ourselves
                                set_framegen_swap(&video);
                            }
                            config::save_key(cli.profile.as_deref(), "framegen_mode", framegen_mode_str(framegen_mode));
                            osd.show(Slot::Transient,
                                format!("Frame Gen: {}", framegen_mode.label()), 1500);
                            dirty = true;
                        } else {
                            osd.show(Slot::Transient, "Frame Gen: need GL 4.3 or VK", 2000);
                        }
                    } else {
                        osd.show(Slot::Transient, "Frame Gen requires GL or VK", 1500);
                    }
                }
                Event::KeyDown { keycode: Some(Keycode::PageUp), .. } => {
                    if let Some(ref a) = _audio {
                        let v = a.volume_up();
                        osd.show(Slot::Transient, format!("Volume: {}%", v), 1000);
                        osd.set_submenu_items("Audio", build_audio_items(&cfg, v, a.is_muted(), a.capture_buf_ms(), a.playback_buf_ms()));
                        dirty = true;
                    }
                }
                Event::KeyDown { keycode: Some(Keycode::PageDown), .. } => {
                    if let Some(ref a) = _audio {
                        let v = a.volume_down();
                        osd.show(Slot::Transient, format!("Volume: {}%", v), 1000);
                        osd.set_submenu_items("Audio", build_audio_items(&cfg, v, a.is_muted(), a.capture_buf_ms(), a.playback_buf_ms()));
                        dirty = true;
                    }
                }
                Event::KeyDown { keycode: Some(Keycode::F4), .. } => {
                    // Toggle: if currently showing, turn off; if off, show configured mode (default simple)
                    if fps_show > 0 {
                        fps_show = 0;
                        osd.clear(Slot::Fps);
                    } else {
                        fps_show = match cfg.fps_display.as_str() {
                            "verbose" => 2,
                            _ => 1, // "simple" or "off" both default to simple
                        };
                        fps_count = 0;
                        cap_fps_count = 0;
                        seq_delta = 0;
                        last_seq = None;
                        upload_us_sum = 0; upload_us_count = 0;
                        present_us_sum = 0; present_us_count = 0;
                        unique_frame_count = 0; stutter_count = 0;
                        last_present_time = std::time::Instant::now();
                        fps_last = std::time::Instant::now();
                        osd.pin(Slot::Fps, if fps_show == 2 { "Device:  --\nCapture: --\nRender:  --" } else { "-- / -- / -- fps" });
                    }
                    cfg.fps_display = match fps_show { 1 => "simple", 2 => "verbose", _ => "off" }.to_string();
                    config::save_key(cli.profile.as_deref(), "fps_display", &cfg.fps_display);
                    // Keep menu in sync so left/right sync doesn't fight F4
                    osd.set_menu_value("FPS", fps_show as usize);
                    dirty = true;
                }
                Event::KeyDown { keycode: Some(Keycode::F9), .. } => {
                    if let Some(ref mut rec) = recorder {
                        let path = rec.stop();
                        recorder = None;
                        osd.clear(Slot::Status);
                        let name = path.file_name()
                            .map(|n| n.to_string_lossy().into_owned())
                            .unwrap_or_default();
                        eprintln!("recording saved: {}", path.display());
                        osd.show(Slot::Transient, format!("Saved {}", name), 2000);
                    } else {
                        let audio_src = _audio.as_ref().map(|a| a.source_name().to_owned());
                        let output_size = if cfg.record_resolution == config::RecordResolution::Window {
                            let sz = with_window!(|win: &sdl2::video::Window| win.size());
                            Some(sz)
                        } else {
                            None
                        };
                        match recording::Recorder::start(
                            cap.width, cap.height, cfg.fps, eff_pixfmt,
                            &recording_dir, audio_src.as_deref(), output_size, debug,
                        ) {
                            Ok(r) => {
                                recorder = Some(r);
                                eprintln!("recording: started");
                                osd.pin(Slot::Status, "Recording");
                            }
                            Err(e) => {
                                eprintln!("recording: {}", e);
                                osd.show(Slot::Transient, "REC failed", 1500);
                            }
                        }
                    }
                }
                Event::KeyDown { keycode: Some(Keycode::F5), .. } => {
                    if streamer.is_some() {
                        // Stop streaming
                        if let Some(mut s) = streamer.take() {
                            s.stop();
                        }
                        osd.set_action_label("Stop Server", "Start Server");
                        eprintln!("streaming: stopped");
                        osd.show(Slot::Transient, "Server stopped", 1500);
                        stream_title_dirty = true;
                        update_streaming_osd(&mut osd, &streamer, &client_receiver, &cfg);
                    } else {
                        // Start streaming
                        let bind = format!("0.0.0.0:{}", cfg.stream_port);
                        match stream_tx::StreamSender::start(
                            &bind, cap.width, cap.height, cfg.fps,
                            eff_pixfmt, cfg.stream_quality, debug,
                        ) {
                            Ok(s) => {
                                let port = s.port();
                                streamer = Some(s);
                                osd.set_action_label("Start Server", "Stop Server");
                                eprintln!("streaming: started on :{}", port);
                                osd.show(Slot::Transient, format!("Server on :{}", port), 1500);
                                stream_title_dirty = true;
                                update_streaming_osd(&mut osd, &streamer, &client_receiver, &cfg);
                            }
                            Err(e) => {
                                eprintln!("streaming: {}", e);
                                osd.show(Slot::Transient, "Server failed", 1500);
                            }
                        }
                    }
                }
                // ── Easter egg: Analysis strip mode (Pause combos) ──
                Event::KeyDown { keycode: Some(Keycode::Pause), keymod, .. } => {
                    let shift = keymod.intersects(Mod::LSHIFTMOD | Mod::RSHIFTMOD);
                    let ctrl = keymod.intersects(Mod::LCTRLMOD | Mod::RCTRLMOD);
                    let alt = keymod.intersects(Mod::LALTMOD | Mod::RALTMOD);

                    if !osd.menu_open() && shift && alt && !ctrl {
                        // Alt+Shift+Pause: finalize early and end session
                        if let Some(ref mut strip) = strip_mode {
                            match strip.finalize(strip_last_mode) {
                                Ok(Some(name)) => {
                                    eprintln!("analysis strip: flushed final {}", name);
                                }
                                Ok(None) => {}
                                Err(e) => eprintln!("analysis strip flush: {}", e),
                            }
                            let total = strip.total_frames();
                            let strips = strip.strip_count();
                            eprintln!("analysis strip: done -- {} frames, {} strips",
                                total, strips);
                            osd.show(Slot::Transient,
                                format!("Strip done ({} frames, {} strips)", total, strips), 2500);
                        }
                        strip_mode = None;
                        osd.clear(Slot::Strip);
                        dirty = true;
                    } else if !osd.menu_open() && shift && ctrl {
                        // Ctrl+Shift+Pause: capture frame -> tarball output
                        if !strip_revealed {
                            // First ever press: just reveal the feature
                            strip_revealed = true;
                            let (av, am, acb, apb) = _audio.as_ref().map(|a| (a.volume(), a.is_muted(), a.capture_buf_ms(), a.playback_buf_ms())).unwrap_or((100, false, cfg.audio_capture_buf as i32, cfg.audio_playback_buf as i32));
                            osd.set_menu_items(build_root_menu(&cfg,
                                streamer.is_some(), client_receiver.is_some(), true,
                                framegen_mode, framegen_quality,
                                target_fps, scale_mode, sharpness,
                                active_backend, aspect_mode, av, am, acb, apb, brightness, contrast, gamma));
                            osd.set_extra_help(vec![
                                "".into(),
                                "--- Analysis Strip ---".into(),
                                "S+Pause     Capture (file)".into(),
                                "C+S+Pause   Capture (tar)".into(),
                                "A+S+Pause   End strip early".into(),
                            ]);
                            osd.show(Slot::Transient, "Enabled strip output", 2500);
                        } else {
                            // Start session if needed, then capture
                            if strip_mode.is_none() {
                                strip_mode = Some(analysis_strip::AnalysisStrip::new(
                                    pictures_dir.clone(), cfg.strip_cols, cfg.strip_rows,
                                ));
                                eprintln!("analysis strip: started (tar mode, {}x{})",
                                    cfg.strip_cols, cfg.strip_rows);
                            }
                            strip_last_mode = analysis_strip::OutputMode::Tar;
                            pending_action = Some(b'A'); // strip capture
                        }
                        dirty = true;
                    } else if !osd.menu_open() && shift {
                        // Shift+Pause: first press reveals, subsequent capture -> file output
                        if !strip_revealed {
                            strip_revealed = true;
                            let (av, am, acb, apb) = _audio.as_ref().map(|a| (a.volume(), a.is_muted(), a.capture_buf_ms(), a.playback_buf_ms())).unwrap_or((100, false, cfg.audio_capture_buf as i32, cfg.audio_playback_buf as i32));
                            osd.set_menu_items(build_root_menu(&cfg,
                                streamer.is_some(), client_receiver.is_some(), true,
                                framegen_mode, framegen_quality,
                                target_fps, scale_mode, sharpness,
                                active_backend, aspect_mode, av, am, acb, apb, brightness, contrast, gamma));
                            osd.set_extra_help(vec![
                                "".into(),
                                "--- Analysis Strip ---".into(),
                                "S+Pause     Capture (file)".into(),
                                "C+S+Pause   Capture (tar)".into(),
                                "A+S+Pause   End strip early".into(),
                            ]);
                            osd.show(Slot::Transient, "Enabled strip output", 2500);
                        } else {
                            // Start session if needed, then capture
                            if strip_mode.is_none() {
                                strip_mode = Some(analysis_strip::AnalysisStrip::new(
                                    pictures_dir.clone(), cfg.strip_cols, cfg.strip_rows,
                                ));
                                eprintln!("analysis strip: started (file mode, {}x{})",
                                    cfg.strip_cols, cfg.strip_rows);
                            }
                            strip_last_mode = analysis_strip::OutputMode::File;
                            pending_action = Some(b'A'); // strip capture
                        }
                        dirty = true;
                    }
                }
                _ => {}
            }
        }

        // Handle renderer toggle (after event processing, before frame work)
        // Vulkan requires a restart — show persistent message if selected.
        // Only update OSD on actual transitions to avoid clobbering Transient
        // notifications every frame.
        {
            let mismatch = cfg.renderer != active_backend;
            if mismatch != renderer_mismatch_shown {
                renderer_mismatch_shown = mismatch;
                if mismatch {
                    let msg = if cfg.renderer == config::RendererBackend::Vulkan {
                        "Restart to switch to Vulkan".to_string()
                    } else {
                        let name = match cfg.renderer {
                            config::RendererBackend::Sdl => "SDL",
                            config::RendererBackend::OpenGl => "OpenGL",
                            _ => "selected backend",
                        };
                        format!("Restart to switch to {}", name)
                    };
                    osd.show(Slot::Transient, msg, 2500);
                }
            }
        }

        // Vulkan present mode hot-switch (swapchain recreation)
        if use_vk {
            if let Some(ref mut vr) = vk_renderer_inst {
                let wanted = vk_renderer::VkRenderer::config_to_vk_present_mode(cfg.vk_present_mode);
                if wanted != vr.present_mode() {
                    let (ww, wh) = with_window!(|w: &sdl2::video::Window| w.size());
                    match vr.recreate_swapchain(ww, wh, Some(wanted), debug) {
                        Ok(()) => {
                            let label = cfg.vk_present_mode.label();
                            osd.show(Slot::Transient, format!("Present: {}", label), 1500);
                            osd.set_submenu_items("Video", build_renderer_items(&cfg, framegen_mode, framegen_quality, target_fps, scale_mode, sharpness, active_backend, aspect_mode, brightness, contrast, gamma));
                            dirty = true;
                        }
                        Err(e) => {
                            eprintln!("vulkan: present mode switch failed: {}", e);
                            osd.show(Slot::Transient, "Present mode switch failed", 2000);
                            // Revert config to match actual present mode
                            cfg.vk_present_mode = match vr.present_mode() {
                                ash::vk::PresentModeKHR::MAILBOX => config::VkPresentMode::Mailbox,
                                ash::vk::PresentModeKHR::IMMEDIATE => config::VkPresentMode::Immediate,
                                _ => config::VkPresentMode::Fifo,
                            };
                            osd.set_submenu_items("Video", build_renderer_items(&cfg, framegen_mode, framegen_quality, target_fps, scale_mode, sharpness, active_backend, aspect_mode, brightness, contrast, gamma));
                        }
                    }
                }
            }
        }

        // GL ↔ SDL hot-switching (only when not involving Vulkan)
        let want_gl = cfg.renderer == config::RendererBackend::OpenGl;
        if want_gl != use_gl && active_backend != config::RendererBackend::Vulkan
            && cfg.renderer != config::RendererBackend::Vulkan
        {
            // macOS: SDL canvas is Metal-backed, GL needs a raw window.
            // Can't switch between them at runtime — save config for next launch.
            #[cfg(target_os = "macos")]
            {
                let label = if want_gl { "opengl" } else { "sdl" };
                config::save_key(cli.profile.as_deref(), "renderer", label);
                osd.show(Slot::Transient, "Restart to apply renderer change", 2500);
                osd.set_submenu_items("Video", build_renderer_items(&cfg, framegen_mode, framegen_quality, target_fps, scale_mode, sharpness, active_backend, aspect_mode, brightness, contrast, gamma));
            }
            #[cfg(not(target_os = "macos"))]
            if want_gl {
                if gl_renderer.is_none() {
                    match gl_renderer::GlRenderer::new(
                        |s| gl_proc_address(&video, s),
                        cap.width, cap.height, eff_pixfmt, cfg.smooth, debug,
                    ) {
                        Ok((mut r, sdl_state)) => {
                            // Try DMA-BUF zero-copy import (Linux only)
                            #[cfg(target_os = "linux")]
                            if !dmabuf_fds.is_empty() {
                                match r.init_dmabuf(
                                    |s| gl_proc_address(&video, s),
                                    &dmabuf_fds, debug,
                                ) {
                                    Ok(()) => {
                                        eprintln!("dmabuf: zero-copy enabled");
                                    }
                                    Err(e) => {
                                        if debug { eprintln!("debug: DMA-BUF init: {}", e); }
                                    }
                                }
                            }
                            saved_sdl_state = Some(sdl_state);
                            r.set_scale_mode(scale_mode);
                            r.set_sharpness(sharpness);
                            gl_renderer = Some(r);
                            use_gl = true;
                            osd.show(Slot::Transient, "Renderer: OpenGL", 1500);
                            osd.set_submenu_items("Video", build_renderer_items(&cfg, framegen_mode, framegen_quality, target_fps, scale_mode, sharpness, active_backend, aspect_mode, brightness, contrast, gamma));
                        }
                        Err(e) => {
                            eprintln!("opengl: {}", e);
                            osd.show(Slot::Transient, "OpenGL init failed", 2000);
                            cfg.renderer = config::RendererBackend::Sdl;
                        }
                    }
                } else {
                    // Reclaim GL context after SDL used it
                    // Save current SDL GL state (viewport may have changed since init)
                    if let Some(ref mut gl) = gl_renderer {
                        saved_sdl_state = Some(gl.save_state());
                        gl.reclaim_context();
                    }
                    use_gl = true;
                    osd.show(Slot::Transient, "Renderer: OpenGL", 1500);
                    osd.set_submenu_items("Video", build_renderer_items(&cfg, framegen_mode, framegen_quality, target_fps, scale_mode, sharpness, active_backend, aspect_mode, brightness, contrast, gamma));
                }
            } else {
                // Switching back to SDL — restore its cached GL state
                if let (Some(ref gl), Some(ref state)) = (&gl_renderer, &saved_sdl_state) {
                    gl.restore_state(state);
                }
                // Force SDL's renderer to recalculate its viewport for the
                // current window size (the saved state may have a stale viewport
                // from before a fullscreen toggle).
                canvas.as_mut().map(|c| c.set_viewport(None));
                use_gl = false;
                // Framegen requires GL — disable and release resources
                if framegen_mode != framegen::FrameGenMode::Off {
                    framegen_mode = framegen::FrameGenMode::Off;
                    frame_gen = None;
                    // Restore vsync before leaving GL
                    if cfg.vsync {
                        video.gl_set_swap_interval(sdl2::video::SwapInterval::VSync).ok();
                    }
                }
                osd.show(Slot::Transient, "Renderer: SDL", 1500);
                osd.set_submenu_items("Video", build_renderer_items(&cfg, framegen_mode, framegen_quality, target_fps, scale_mode, sharpness, active_backend, aspect_mode, brightness, contrast, gamma));
            }
            dirty = true;
        }

        // Wayland minimize detection via foreign-toplevel protocol
        #[cfg(target_os = "linux")]
        if let Some(ref watcher) = wl_minimize {
            let is_min = watcher.is_minimized();
            if is_min && !wl_was_minimized {
                // Just minimized
                wl_was_minimized = true;
                if cfg.pause_on_minimize && !paused {
                    paused = true;
                    auto_paused = true;
                    auto_paused_by_minimize = true;
                    audio_was_muted = _audio.as_ref().map_or(true, |a| a.is_muted());
                    if let Some(ref mut a) = _audio {
                        if !audio_was_muted { a.set_muted(true); }
                    }
                    osd.set_bottom_right(Some("Paused".into()));
                    if use_vk {
                        if let Some(ref mut vk) = vk_renderer_inst {
                            vk.dim_staging(0.7);
                        }
                    }
                    dirty = true;
                }
            } else if !is_min && wl_was_minimized {
                // Just restored from minimize
                wl_was_minimized = false;
                if paused && auto_paused && auto_paused_by_minimize {
                    paused = false;
                    auto_paused = false;
                    auto_paused_by_minimize = false;
                    if let Some(ref mut a) = _audio {
                        if !audio_was_muted { a.set_muted(false); }
                    }
                    osd.set_bottom_right(None);
                    if use_vk {
                        if let Some(ref mut vk) = vk_renderer_inst {
                            vk.restore_staging();
                        }
                    }
                    dirty = true;
                }
            }
        }

        // When paused, skip capture but still render (for OSD updates)
        if paused {
            if !dirty {
                std::thread::sleep(std::time::Duration::from_millis(16));
                continue;
            }
            // Fall through to render the frozen frame with OSD
        }

        let _poll_t0 = std::time::Instant::now();
        let mut _poll_us = 0u128;
        let mut _dequeue_us = 0u128;
        let mut drained_count: u32 = 0;

        if !paused {
        // Wait for capture frame.
        // Linux: ppoll() on v4l2 fd for sub-millisecond precision.
        // macOS: poll() on pipe fd from AVFoundation delegate callback.
        #[cfg(target_os = "macos")]
        unsafe {
            let mut pfd = libc::pollfd {
                fd: v4l2_fd,
                events: libc::POLLIN,
                revents: 0,
            };
            if framegen_mode != framegen::FrameGenMode::Off {
                let now = std::time::Instant::now();
                if now >= next_synth_deadline {
                    libc::poll(&mut pfd, 1, 0);
                } else {
                    let remaining = next_synth_deadline - now;
                    libc::poll(&mut pfd, 1, remaining.as_millis() as i32);
                }
            } else {
                libc::poll(&mut pfd, 1, poll_timeout_ms);
            }
        }
        #[cfg(target_os = "linux")]
        unsafe {
            let mut pfd = libc::pollfd {
                fd: v4l2_fd,
                events: libc::POLLIN,
                revents: 0,
            };
            if framegen_mode != framegen::FrameGenMode::Off {
                let now = std::time::Instant::now();
                if now >= next_synth_deadline {
                    libc::poll(&mut pfd, 1, 0);
                } else {
                    let remaining = next_synth_deadline - now;
                    let ts = libc::timespec {
                        tv_sec: remaining.as_secs() as libc::time_t,
                        tv_nsec: remaining.subsec_nanos() as libc::c_long,
                    };
                    libc::ppoll(&mut pfd, 1, &ts, std::ptr::null());
                }
            } else {
                libc::poll(&mut pfd, 1, poll_timeout_ms);
            }
        }
        _poll_us = _poll_t0.elapsed().as_micros();

        // Drain all available frames, keep latest (minimize latency)
        let dequeue_t0 = if fps_show == 2 || perf_active { std::time::Instant::now() } else { _poll_t0 };
        let mut latest: Option<capture::V4l2Buffer> = None;
        loop {
            match cap.dequeue()? {
                Some(buf) => {
                    if let Some(mut prev) = latest.take() {
                        cap.queue(&mut prev)?;
                    }
                    latest = Some(buf);
                    drained_count += 1;
                }
                None => break,
            }
        }
        if fps_show == 2 || perf_active { _dequeue_us = dequeue_t0.elapsed().as_micros(); }

        if let Some(buf) = &latest {
            // Track inter-frame gap for jitter analysis (only when verbose FPS or perf active)
            if fps_show == 2 || perf_active {
                let now_cap = std::time::Instant::now();
                let gap = (now_cap - last_capture_time).as_micros() as u64;
                last_capture_time = now_cap;
                frame_gap_us_buf[frame_gap_idx] = gap;
                frame_gap_idx = (frame_gap_idx + 1) & 63;
                if frame_gap_count < 64 { frame_gap_count += 1; }
                if gap > max_frame_gap_us { max_frame_gap_us = gap; }
            }
            if fps_show > 0 {
                cap_fps_count += 1;
                let s = buf.sequence();
                if let Some(prev) = last_seq {
                    seq_delta += s.wrapping_sub(prev);
                }
                last_seq = Some(s);
            }
            let ptr = cap.buffer_ptr(buf.index);
            let raw_len = if is_mjpeg { buf.bytesused as usize } else { buf.length as usize };
            let raw_data = unsafe { std::slice::from_raw_parts(ptr, raw_len) };
            // MJPEG: decode JPEG → RGB before anything else sees the data
            let frame_data: &[u8] = if is_mjpeg {
                if let Some(ref tj) = mjpeg_decoder {
                    match tj.decompress_into(raw_data, &mut mjpeg_rgb_buf) {
                        Ok(_) => &mjpeg_rgb_buf,
                        Err(e) => {
                            if debug { eprintln!("debug: MJPEG decode failed: {}", e); }
                            continue; // skip this frame
                        }
                    }
                } else { raw_data }
            } else { raw_data };
            let frame_len = frame_data.len();

            // Fast frame content hash for unique frame detection (verbose FPS)
            if fps_show == 2 {
                let mut h: u64 = 0xcbf29ce484222325; // FNV-1a offset basis
                let stride = (frame_len / 128).max(1);
                let mut i = 0;
                while i < frame_len {
                    h ^= frame_data[i] as u64;
                    h = h.wrapping_mul(0x100000001b3); // FNV-1a prime
                    i += stride;
                }
                if h != last_frame_hash {
                    unique_frame_count += 1;
                    last_frame_hash = h;
                }
            }

            // Recorder copies the slice into its own pooled buffer (no alloc).
            if let Some(ref rec) = recorder {
                if !rec.write_frame(frame_data) {
                    eprintln!("recording: pipe broken, stopping");
                    recorder = None;
                }
            }
            #[cfg(target_os = "linux")]
            if let Some(ref vc) = vcam {
                if !vc.write_frame(frame_data) {
                    eprintln!("virtual_webcam: device error, stopping");
                    vcam = None;
                }
            }
            if let Some(ref streamer) = streamer {
                // Arc::from(slice) copies the frame into a refcounted buffer
                // handed off to the sender thread. One full-frame memcpy per
                // tick when streaming is enabled — not zero-cost.
                streamer.send_frame(std::sync::Arc::from(frame_data));
            }

            // Only snapshot frame data when screenshot/clipboard was requested
            let need_snapshot = pending_action.is_some();
            if need_snapshot {
                last_frame.clear();
                last_frame.extend_from_slice(frame_data);
            }

            // Run through filter pipeline if any plugins are loaded
            let plugin_frames: Option<Vec<Vec<u8>>> = if let Some(ref pipe) = pipeline {
                match pipe.process(frame_data, cap.width, cap.height) {
                    Ok(out) if out.is_empty() => Some(Vec::new()), // skip
                    Ok(out) => {
                        if need_snapshot {
                            if let Some(first) = out.first() {
                                last_frame.clear();
                                last_frame.extend_from_slice(first);
                            }
                        }
                        Some(out)
                    }
                    Err(e) => {
                        eprintln!("plugin error: {}", e);
                        None // fall through to normal rendering
                    }
                }
            } else {
                None
            };

            // Determine upload source: plugin output or raw capture
            let upload_src: Option<&[u8]> = match &plugin_frames {
                Some(frames) if !frames.is_empty() => Some(frames.last().unwrap()),
                Some(_) => None,
                None => Some(frame_data),
            };

            // Zero-copy DMA-BUF path (Linux only): when GL is active, no
            // plugins are modifying the frame, and we have DMA-BUF import,
            // just bind the V4L2 buffer's EGLImage to the GL texture.
            #[cfg(target_os = "linux")]
            let dmabuf_used = use_gl
                && !is_mjpeg  // MJPEG buffers contain JPEG data, not pixels
                && pipeline.is_none()
                && upload_src.is_some()
                && gl_renderer.as_mut().map_or(false, |gl| gl.bind_dmabuf(buf.index));
            #[cfg(not(target_os = "linux"))]
            let dmabuf_used = false;

            if !dmabuf_used {
                if let Some(src) = upload_src {
                    // GL path: contrast is handled in the fragment shader uniform.
                    // SDL path: no shader control, so apply the LUT on the CPU.
                    let need_sdl_lut = !use_gl && ((contrast - 1.0).abs() > 0.001 || (gamma - 1.0).abs() > 0.001);
                    let upload_data = if need_sdl_lut {
                        let lut = match cached_lut {
                            Some((b, c, g, ref lut)) if b == brightness && c == contrast && g == gamma => lut,
                            _ => {
                                cached_lut = Some((brightness, contrast, gamma, build_contrast_lut(contrast, gamma)));
                                &cached_lut.as_ref().unwrap().3
                            }
                        };
                        adjusted_frame.resize(src.len(), 0);
                        adjusted_frame.copy_from_slice(src);
                        apply_y_lut(&mut adjusted_frame, cap.width, cap.height, eff_pixfmt, lut);
                        &adjusted_frame[..]
                    } else {
                        src
                    };

                    let upload_t0 = std::time::Instant::now();
                    if use_vk {
                        if let Some(ref mut vk) = vk_renderer_inst {
                            vk.upload(upload_data, brightness, contrast, gamma);
                        }
                    } else if use_gl {
                        if let Some(ref mut gl) = gl_renderer {
                            gl.upload(upload_data);
                        }
                    } else if let Some(ref mut tex) = texture {
                        update_texture_from_slice(tex, upload_data, cap.width, cap.height, eff_pixfmt);
                    }
                    if fps_show == 2 {
                        upload_us_sum += upload_t0.elapsed().as_micros() as u64;
                        upload_us_count += 1;
                    }
                }
            }
            dirty = true;
            real_frame_this_tick = true;
            _last_real_frame = std::time::Instant::now();
        }

        // Requeue buffer immediately after texture upload — the mmap data has
        // been consumed by SDL at this point, so give the buffer back to the
        // driver BEFORE present() which may block on vsync.
        if let Some(mut buf) = latest {
            cap.queue(&mut buf)?;
        }
        } // if !paused

        // Poll client receiver for frames (when connected via menu)
        if let Some(ref rx) = client_receiver {
            while let Some(frame) = rx.try_recv() {
                // Create or resize the RGB texture on demand
                if client_tex.is_none() || client_dim != (frame.width, frame.height) {
                    if let Some(ref tc) = texture_creator {
                        client_tex = Some(tc
                            .create_texture_streaming(PixelFormatEnum::RGB24, frame.width, frame.height)
                            .map_err(|e| anyhow::anyhow!(e))?);
                        client_dim = (frame.width, frame.height);
                    }
                }
                if let Some(ref mut tex) = client_tex {
                    let pitch = (frame.width * 3) as usize;
                    let _ = tex.update(None, &frame.rgb, pitch);
                }
                dirty = true;
            }
        }

        // Execute deferred clipboard action (runs after buffer requeue
        // so we don't hold the mmap buffer hostage during PNG encoding)
        if let Some(action) = pending_action.take() {
            if !last_frame.is_empty() {
                match action {
                    b'C' => {
                        if !clipboard::is_wayland() {
                            eprintln!("clipboard: wayland only");
                        } else {
                            match screenshot::encode_png_bytes(
                                &last_frame, cap.width, cap.height, eff_pixfmt,
                            ) {
                                Ok(bytes) => match clipboard::copy_to_clipboard(&bytes, debug) {
                                    Ok(()) => {
                                        eprintln!("copied to clipboard");
                                        osd.show(Slot::Transient, "Screenshot copied", 1500);
                                    }
                                    Err(e) => eprintln!("clipboard: {}", e),
                                },
                                Err(e) => eprintln!("clipboard png: {}", e),
                            }
                        }
                    }
                    b'S' => {
                        match screenshot::save_screenshot(
                            &last_frame, cap.width, cap.height, eff_pixfmt,
                            &pictures_dir, cfg.screenshot_format, cfg.jpeg_quality,
                        ) {
                            Ok(path) => {
                                eprintln!("screenshot saved: {}", path);
                                osd.show(Slot::Transient, "Screenshot saved", 1500);
                            }
                            Err(e) => eprintln!("screenshot save: {}", e),
                        }
                    }
                    b'T' => {
                        match screenshot::append_to_tar(
                            &last_frame, cap.width, cap.height, eff_pixfmt,
                            &session_tar, cfg.screenshot_format, cfg.jpeg_quality,
                        ) {
                            Ok(entry) => {
                                let tar_name = session_tar.file_name()
                                    .map(|n| n.to_string_lossy().into_owned())
                                    .unwrap_or_default();
                                eprintln!("screenshot added to {}: {}", tar_name, entry);
                                osd.show(Slot::Transient, format!("Added to {}", tar_name), 1500);
                            }
                            Err(e) => eprintln!("screenshot tar: {}", e),
                        }
                    }
                    b'A' => {
                        // Analysis strip: ingest frame with current output mode
                        if let Some(ref mut strip) = strip_mode {
                            // Sync grid config in case user changed it in OSD
                            strip.set_grid(cfg.strip_cols, cfg.strip_rows);
                            match strip.ingest_frame(
                                &last_frame, cap.width, cap.height, eff_pixfmt,
                                strip_last_mode,
                            ) {
                                Ok(Some(strip_name)) => {
                                    let mode_str = match strip_last_mode {
                                        analysis_strip::OutputMode::File => "file",
                                        analysis_strip::OutputMode::Tar => "tar",
                                    };
                                    eprintln!("analysis strip: wrote {} ({})", strip_name, mode_str);
                                    osd.show(Slot::Transient,
                                        format!("{} ({})", strip_name, mode_str), 1500);
                                }
                                Ok(None) => {
                                    // Frame buffered, not yet a full grid
                                    osd.show(Slot::Transient, "Frame captured", 800);
                                }
                                Err(e) => eprintln!("analysis strip: {}", e),
                            }
                            // Update the persistent counter
                            let buf = strip.buffered_count();
                            let total = strip.total_frames();
                            let cap = strip.capacity();
                            osd.pin(Slot::Strip,
                                format!("Strip: {} ({}/{})", total, buf, cap));
                        }
                    }
                    _ => {}
                }
            }
        }

        // Repaint when OSD content changed (new message, expiry, menu interaction)
        if osd.take_dirty() { dirty = true; }

        // FPS counter update (once per second)
        if fps_show > 0 {
            let elapsed = fps_last.elapsed();
            if elapsed.as_secs() >= 1 {
                let secs = elapsed.as_secs_f64();
                let render_fps = fps_count as f64 / secs;
                let capture_fps = cap_fps_count as f64 / secs;
                let device_fps = seq_delta as f64 / secs;
                let tag = if use_vk {
                    match vk_renderer_inst.as_ref().map(|v| v.present_mode()) {
                        Some(ash::vk::PresentModeKHR::MAILBOX) => "VK+MB",
                        Some(ash::vk::PresentModeKHR::IMMEDIATE) => "VK+IMM",
                        Some(ash::vk::PresentModeKHR::FIFO) => "VK+FIFO",
                        _ => "VK",
                    }
                } else if use_gl {
                    if gl_renderer.as_ref().map_or(false, |g| g.has_dmabuf()) {
                        "GL+DMA"
                    } else {
                        "GL"
                    }
                } else {
                    "SDL"
                };
                let fps_text = if fps_show == 2 {
                    // Verbose: multi-line labeled stats
                    let avg_upload = if upload_us_count > 0 {
                        upload_us_sum as f64 / upload_us_count as f64 / 1000.0
                    } else { 0.0 };
                    let avg_present = if present_us_count > 0 {
                        present_us_sum as f64 / present_us_count as f64 / 1000.0
                    } else { 0.0 };
                    let scale_tag = if use_vk {
                        match scale_mode {
                            gl_renderer::ScaleMode::Cas => " CAS",
                            gl_renderer::ScaleMode::Fsr | gl_renderer::ScaleMode::IntegerFsr => " FSR",
                            _ => "",
                        }
                    } else { "" };
                    let unique_fps = unique_frame_count as f64 / secs;
                    let dropped = cap_fps_count.saturating_sub(unique_frame_count);
                    let dupes_total = total_dupes + dropped as u64;
                    let mut lines = format!(
                        "Device:  {:.0} fps\nCapture: {:.0} fps\nUnique:  {:.0} fps (dupes: {}/{})\nRender:  {:.0} fps [{}{}]\nUpload:  {:.1}ms  Present: {:.1}ms",
                        device_fps, capture_fps, unique_fps, dropped, dupes_total,
                        render_fps, tag, scale_tag, avg_upload, avg_present);
                    lines.push_str(&format!("\nStutters: {}", stutter_count));
                    if framegen_mode != framegen::FrameGenMode::Off {
                        if let Some(ref fg) = frame_gen {
                            let s = fg.stats();
                            lines.push_str(&format!(
                                "\nFrameGen: {}fps {:.1}ms (synth:{} miss:{})",
                                target_fps,
                                s.last_gen_us as f64 / 1000.0,
                                s.synth_count,
                                s.miss_count,
                            ));
                        }
                        if let Some(ref vk) = vk_renderer_inst {
                            if let Some(s) = vk.fg_stats() {
                                lines.push_str(&format!(
                                    "\nFrameGen[VK]: {}fps (synth:{} miss:{})",
                                    target_fps,
                                    s.synth_count,
                                    s.miss_count,
                                ));
                            }
                        }
                    }
                    // CPU usage from /proc/self/stat
                    if let Ok(stat) = std::fs::read_to_string("/proc/self/stat") {
                        let fields: Vec<&str> = stat.split_whitespace().collect();
                        if fields.len() > 14 {
                            let utime: u64 = fields[13].parse().unwrap_or(0);
                            let stime: u64 = fields[14].parse().unwrap_or(0);
                            let ticks = utime + stime;
                            let dt = (std::time::Instant::now() - last_cpu_time).as_secs_f64();
                            if dt > 0.0 {
                                let tick_hz = 100.0; // sysconf(_SC_CLK_TCK), typically 100
                                let dticks = (ticks - last_cpu_ticks) as f64;
                                cpu_pct = (dticks / tick_hz / dt) * 100.0;
                            }
                            last_cpu_ticks = ticks;
                            last_cpu_time = std::time::Instant::now();
                        }
                    }
                    lines.push_str(&format!("\nCPU: {:.1}%", cpu_pct));
                    // CPU frequency and throttle detection
                    {
                        let (cur, max, thr) = priority::check_throttle();
                        cpu_freq_mhz = cur;
                        _cpu_max_freq_mhz = max;
                        cpu_throttled = thr;
                        let cpu_id = priority::render_cpu_id();
                        if max > 0 {
                            lines.push_str(&format!("\nCPU{}: {}/{}MHz{}", cpu_id, cur, max,
                                if thr { " THROTTLED" } else { "" }));
                        } else if cpu_id >= 0 {
                            lines.push_str(&format!("\nCPU{}", cpu_id));
                        }
                    }
                    // Frame jitter stats
                    if frame_gap_count >= 2 {
                        let n = frame_gap_count.min(64);
                        let sum: u64 = frame_gap_us_buf[..n].iter().sum();
                        let mean = sum / n as u64;
                        let var: u64 = frame_gap_us_buf[..n].iter()
                            .map(|&g| { let d = if g > mean { g - mean } else { mean - g }; d * d })
                            .sum::<u64>() / n as u64;
                        let jitter_us = (var as f64).sqrt();
                        lines.push_str(&format!("\nJitter: {:.1}ms (max gap: {:.1}ms)",
                            jitter_us / 1000.0, max_frame_gap_us as f64 / 1000.0));
                        if drained_count > 1 {
                            lines.push_str(&format!(" burst:{}", drained_count));
                        }
                    }
                    // Auto re-pin if throttled (check every 10 seconds)
                    if cpu_throttled && last_repin_check.elapsed().as_secs() >= 10 {
                        last_repin_check = std::time::Instant::now();
                        if let Some(new_cpu) = priority::repin_if_throttled(debug) {
                            lines.push_str(&format!("\nRe-pinned to CPU{}", new_cpu));
                        }
                    }
                    // Audio underruns
                    if let Some(ref a) = _audio {
                        let xr = a.xruns();
                        if xr > 0 {
                            lines.push_str(&format!("\nAudio xruns: {}", xr));
                        }
                    }
                    // Reset max gap each display interval
                    max_frame_gap_us = 0;
                    lines
                } else {
                    // Simple: single line (existing format)
                    let mut fps_line = format!("{:.0} / {:.0} / {:.0} fps [{}]", device_fps, capture_fps, render_fps, tag);
                    if framegen_mode != framegen::FrameGenMode::Off {
                        if let Some(ref fg) = frame_gen {
                            let s = fg.stats();
                            fps_line.push_str(&format!(" | t:{}fps {:.1}ms p:{:.1}ms s:{} m:{}",
                                target_fps,
                                s.last_gen_us as f64 / 1000.0,
                                present_ema_us / 1000.0,
                                s.synth_count,
                                s.miss_count,
                            ));
                        }
                        if let Some(ref vk) = vk_renderer_inst {
                            if let Some(s) = vk.fg_stats() {
                                fps_line.push_str(&format!(" | VK t:{}fps p:{:.1}ms s:{} m:{}",
                                    target_fps,
                                    present_ema_us / 1000.0,
                                    s.synth_count,
                                    s.miss_count,
                                ));
                            }
                        }
                    }
                    fps_line
                };
                osd.pin(Slot::Fps, fps_text);
                total_dupes += cap_fps_count.saturating_sub(unique_frame_count) as u64;
                fps_count = 0;
                cap_fps_count = 0;
                seq_delta = 0;
                last_seq = None;
                upload_us_sum = 0; upload_us_count = 0;
                present_us_sum = 0; present_us_count = 0;
                unique_frame_count = 0;
                fps_last = std::time::Instant::now();
            }
        }

        // Update window title for streaming status
        {
            let mut needs_update = stream_title_dirty;
            if let Some(ref s) = streamer {
                let cc = s.client_count();
                if cc != stream_last_clients {
                    needs_update = true;
                    stream_last_clients = cc;
                    // Also refresh the persistent OSD indicator
                    update_streaming_osd(&mut osd, &streamer, &client_receiver, &cfg);
                }
            }
            if needs_update {
                stream_title_dirty = false;
                let mut parts = Vec::new();
                if let Some(ref s) = streamer {
                    let cc = s.client_count();
                    parts.push(format!("streaming on :{}, {} client{}",
                        s.port(), cc, if cc == 1 { "" } else { "s" }));
                }
                if client_receiver.is_some() {
                    let ip = cfg.stream_client_ip;
                    parts.push(format!("connected to {}.{}.{}.{}:{}",
                        ip[0], ip[1], ip[2], ip[3], cfg.stream_client_port));
                }
                let title = if parts.is_empty() {
                    "capview".to_string()
                } else {
                    format!("capview <{}>", parts.join(" | "))
                };
                with_window_mut!(|win: &mut sdl2::video::Window| {
                    let _ = win.set_title(&title);
                });
            }
        }

        // Frame generation: when no real frame arrived but framegen is active,
        // synthesize intermediate frames paced by present_interval.
        if !real_frame_this_tick && framegen_mode != framegen::FrameGenMode::Off
            && std::time::Instant::now() >= next_synth_deadline
        {
            #[cfg(feature = "rife")]
            let is_rife_mode = framegen_mode == framegen::FrameGenMode::Rife;
            #[cfg(not(feature = "rife"))]
            #[allow(unused_variables)]
            let is_rife_mode = false;

            if is_rife_mode {
                #[cfg(feature = "rife")]
                if rife_interp.is_some() {
                    dirty = true;
                }
            } else if let Some(ref mut fg) = frame_gen {
                if fg.can_generate() {
                    dirty = true;
                }
            } else if use_vk {
                if let Some(ref vk) = vk_renderer_inst {
                    if vk.fg_can_generate() {
                        dirty = true;
                    }
                }
            }
        }

        let eff_brightness = if paused { brightness * 0.7 } else { brightness };

        if dirty {
            let (win_w, win_h) = if use_vk {
                if let Some(ref vk) = vk_renderer_inst {
                    vk.extent()
                } else {
                    (cap.width, cap.height)
                }
            } else if let Some(ref c) = canvas {
                c.output_size().unwrap_or((cap.width, cap.height))
            } else if let Some(ref w) = vk_window {
                // macOS GL mode: use drawable size (physical pixels, not points)
                w.drawable_size()
            } else {
                (cap.width, cap.height)
            };

            // Client receiver mode: render received stream via SDL (RGB24)
            // (not supported in VK mode — requires SDL textures)
            if !use_vk && client_receiver.is_some() && client_tex.is_some() {
                // Temporarily use SDL path if GL is active
                if use_gl {
                    if let (Some(ref gl), Some(ref state)) = (&gl_renderer, &saved_sdl_state) {
                        gl.restore_state(state);
                    }
                }
                let (cw, ch) = client_dim;
                let dst = fit_rect(cw, ch, win_w, win_h, aspect_mode);
                if let Some(ref mut c) = canvas {
                    c.set_draw_color(sdl2::pixels::Color::RGB(0, 0, 0));
                    c.set_blend_mode(sdl2::render::BlendMode::None);
                    c.clear();
                    if let Some(ref tex) = client_tex {
                        c.copy(tex, None, Some(dst)).map_err(|e| anyhow::anyhow!(e))?;
                    }
                    osd.render(c, win_w, win_h);
                    c.present();
                }
            } else if use_vk {
                // Vulkan render path
                if let Some(ref mut vk) = vk_renderer_inst {
                    osd.render_vk(vk, win_w, win_h);
                    if real_frame_this_tick {
                        // Real frame: render_and_present handles fg capture internally
                        if !vk.render_and_present(win_w, win_h, eff_brightness, contrast, gamma) {
                            eprintln!("vulkan: present failed");
                        }
                        synth_count_since_real = 0;
                    } else if framegen_mode != framegen::FrameGenMode::Off && vk.fg_can_generate() {
                        // Synth frame
                        synth_count_since_real += 1;
                        let t = (synth_count_since_real as f32
                            / (synths_per_real + 1) as f32)
                            .clamp(0.01, 0.99);
                        if !vk.render_synth_and_present(
                            win_w, win_h, eff_brightness, contrast, gamma,
                            t, framegen_mode, framegen_quality,
                        ) {
                            eprintln!("vulkan: synth present failed");
                        }
                    } else {
                        // No synth available — re-render last frame
                        if !vk.render_and_present(win_w, win_h, eff_brightness, contrast, gamma) {
                            eprintln!("vulkan: present failed");
                        }
                    }
                }
            } else if use_gl {
                if let Some(ref mut gl) = gl_renderer {
                    if real_frame_this_tick {
                        // Real frame: render YUV→RGB as usual
                        gl.render(win_w, win_h, eff_brightness, contrast, gamma);

                        // Capture rendered frame for framegen (before OSD)
                        if framegen_mode != framegen::FrameGenMode::Off {
                            if let Some(ref mut fg) = frame_gen {
                                fg.push_frame(win_w, win_h);
                            }
                        }
                        synth_count_since_real = 0;
                    } else {
                        // No real frame — render synthesised frame.
                        #[cfg(feature = "rife")]
                        let is_rife_mode = framegen_mode == framegen::FrameGenMode::Rife;
                        #[cfg(not(feature = "rife"))]
                        #[allow(unused_variables)]
                        let is_rife_mode = false;

                        #[allow(unused_mut)]
                        let mut rendered = false;

                        #[cfg(feature = "rife")]
                        if is_rife_mode {
                            if let (Some(ref mut ri), Some(ref fg)) = (&mut rife_interp, &frame_gen) {
                                if fg.can_generate() {
                                    let (tp, tc) = fg.prev_curr_textures();
                                    if ri.interpolate(tp, tc) {
                                        let (rw, rh) = ri.dimensions();
                                        gl.render_texture(
                                            ri.output_texture(), rw, rh,
                                            win_w, win_h, eff_brightness, gamma,
                                        );
                                        rendered = true;
                                    }
                                }
                            }
                        }

                        if !rendered {
                            if let Some(ref mut fg) = frame_gen {
                                synth_count_since_real += 1;
                                let t = (synth_count_since_real as f32
                                    / (synths_per_real + 1) as f32)
                                    .clamp(0.01, 0.99);
                                if fg.generate(t) {
                                    let (fg_w, fg_h) = fg.dimensions();
                                    gl.render_texture(
                                        fg.output_texture(),
                                        fg_w, fg_h,
                                        win_w, win_h,
                                        eff_brightness, gamma,
                                    );
                                    rendered = true;
                                }
                            }
                        }

                        // No new frame and no synthetic frame — re-render
                        // the last texture to prevent double-buffer ghosting
                        // (swapping without drawing shows stale back-buffer
                        // content from 2 frames ago).
                        if !rendered {
                            gl.render(win_w, win_h, eff_brightness, contrast, gamma);
                        }
                    }

                    if debug {
                        if let Some(msg) = gl.check_frame_error() {
                            eprintln!("debug: {}", msg);
                        }
                    }

                    // OSD via native GL (avoids SDL/GL state conflicts)
                    osd.render_gl(gl, win_w, win_h);
                }
            } else if let Some(ref mut c) = canvas {
                let dst = fit_rect(cap.width, cap.height, win_w, win_h, aspect_mode);
                c.set_draw_color(sdl2::pixels::Color::RGB(0, 0, 0));
                c.set_blend_mode(sdl2::render::BlendMode::None);
                c.clear();

                // Apply brightness via texture color modulation
                if let Some(ref mut tex) = texture {
                    if eff_brightness <= 1.0 {
                        let bv = (eff_brightness * 255.0) as u8;
                        tex.set_color_mod(bv, bv, bv);
                        tex.set_blend_mode(sdl2::render::BlendMode::None);
                        c.copy(tex, None, Some(dst)).map_err(|e| anyhow::anyhow!(e))?;
                    } else {
                        // Base pass at full brightness
                        tex.set_color_mod(255, 255, 255);
                        tex.set_blend_mode(sdl2::render::BlendMode::None);
                        c.copy(tex, None, Some(dst)).map_err(|e| anyhow::anyhow!(e))?;
                        // Additive pass for the extra brightness
                        let extra = ((eff_brightness - 1.0) * 255.0).min(255.0) as u8;
                        tex.set_color_mod(extra, extra, extra);
                        tex.set_blend_mode(sdl2::render::BlendMode::Add);
                        c.copy(tex, None, Some(dst)).map_err(|e| anyhow::anyhow!(e))?;
                    }
                }

                // OSD overlay via SDL
                osd.render(c, win_w, win_h);
            }

            let pre_present = std::time::Instant::now();
            let _render_us = (pre_present - _poll_t0).as_micros();
            // VK already presented in its render branch
            if !use_vk {
                if let Some(ref mut c) = canvas {
                    c.present();
                } else if let Some(ref w) = vk_window {
                    // macOS GL mode: swap via raw GL context
                    w.gl_swap_window();
                }
            }
            let now = std::time::Instant::now();
            let present_us = (now - pre_present).as_micros() as f64;
            present_ema_us = present_ema_us * 0.9 + present_us * 0.1;
            if fps_show == 2 {
                present_us_sum += present_us as u64;
                present_us_count += 1;
            }
            // Per-frame perf: non-blocking send to background writer thread.
            // try_send drops the row if the channel is full — never blocks.
            if let Some(ref tx) = perf_tx {
                let ts = perf_t0.elapsed().as_micros() as u64;
                let gap = if frame_gap_count > 0 { frame_gap_us_buf[(frame_gap_idx + 63) & 63] } else { 0 };
                let _ = tx.try_send([
                    ts, _poll_us as u64, _dequeue_us as u64, 0,
                    _render_us as u64, present_us as u64, gap,
                    cpu_freq_mhz as u64, (cpu_pct * 10.0) as u64,
                    0, if cpu_throttled { 1 } else { 0 }, drained_count as u64,
                ]);
            }
            // Debug: log timing every 2 seconds
            if debug && (now - _poll_t0).as_micros() > 0 {
                static DBGN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                static DBGT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                let n = DBGN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if n == 0 { DBGT.store(now.elapsed().as_secs(), std::sync::atomic::Ordering::Relaxed); }
                if n == 60 {
                    eprintln!("debug: loop poll={}us render={}us present={}us total={}us (real={})",
                        _poll_us, _render_us, present_us as u64,
                        (now - _poll_t0).as_micros(), real_frame_this_tick);
                    DBGN.store(0, std::sync::atomic::Ordering::Relaxed);
                }
            }
            if real_frame_this_tick || framegen_mode == framegen::FrameGenMode::Off {
                // Real frame: reset deadline from actual present time
                next_synth_deadline = now + present_interval;
            } else {
                // Synthetic frame: advance deadline by exact interval (no drift)
                next_synth_deadline += present_interval;
                // If we fell behind (deadline already in the past), snap forward
                // to avoid burst-generating to catch up.
                if next_synth_deadline < now {
                    next_synth_deadline = now + present_interval;
                }
            }
            _last_present = now;

            if fps_show > 0 {
                fps_count += 1;
                if fps_show == 2 {
                    let gap = (now - last_present_time).as_micros() as u64;
                    let expected = 1_000_000 / source_fps.max(1) as u64;
                    if gap > expected * 3 / 2 { stutter_count += 1; }
                    last_present_time = now;
                }
            }
            dirty = false;

            // After presenting a real frame, immediately generate synth frames
            // that are due. When present() blocks on the compositor (Wayland),
            // v4l2 frames accumulate during the block, causing the main loop to
            // always have a real frame ready — which prevents the synth check
            // from ever firing. This inner loop ensures synths get rendered
            // right after the real frame, before returning to ppoll.
            if real_frame_this_tick && framegen_mode != framegen::FrameGenMode::Off && use_gl {
                while std::time::Instant::now() >= next_synth_deadline {
                    let can_gen = frame_gen.as_ref().map_or(false, |fg| fg.can_generate());
                    if !can_gen { break; }
                    if let (Some(ref mut fg), Some(ref mut gl)) = (&mut frame_gen, &mut gl_renderer) {
                        synth_count_since_real += 1;
                        let t = (synth_count_since_real as f32
                            / (synths_per_real + 1) as f32)
                            .clamp(0.01, 0.99);
                        if fg.generate(t) {
                            let (fg_w, fg_h) = fg.dimensions();
                            gl.render_texture(
                                fg.output_texture(),
                                fg_w, fg_h,
                                win_w, win_h,
                                eff_brightness, gamma,
                            );
                            osd.render_gl(gl, win_w, win_h);
                        }
                    }
                    let pre = std::time::Instant::now();
                    if let Some(ref mut c) = canvas { c.present(); }
                    let now2 = std::time::Instant::now();
                    let pus = (now2 - pre).as_micros() as f64;
                    present_ema_us = present_ema_us * 0.9 + pus * 0.1;
                    next_synth_deadline += present_interval;
                    if next_synth_deadline < now2 {
                        next_synth_deadline = now2 + present_interval;
                    }
                    _last_present = now2;
                    if fps_show > 0 {
                        fps_count += 1;
                        if fps_show == 2 { last_present_time = now2; }
                    }
                }
            }
            // VK post-present synth loop
            if real_frame_this_tick && framegen_mode != framegen::FrameGenMode::Off && use_vk {
                while std::time::Instant::now() >= next_synth_deadline {
                    if let Some(ref vk) = vk_renderer_inst {
                        if !vk.fg_can_generate() { break; }
                    } else {
                        break;
                    }
                    if let Some(ref mut vk) = vk_renderer_inst {
                        synth_count_since_real += 1;
                        let t = (synth_count_since_real as f32
                            / (synths_per_real + 1) as f32)
                            .clamp(0.01, 0.99);
                        osd.render_vk(vk, win_w, win_h);
                        let pre = std::time::Instant::now();
                        vk.render_synth_and_present(
                            win_w, win_h, eff_brightness, contrast, gamma,
                            t, framegen_mode, framegen_quality,
                        );
                        let now2 = std::time::Instant::now();
                        let pus = (now2 - pre).as_micros() as f64;
                        present_ema_us = present_ema_us * 0.9 + pus * 0.1;
                        next_synth_deadline += present_interval;
                        if next_synth_deadline < now2 {
                            next_synth_deadline = now2 + present_interval;
                        }
                        _last_present = now2;
                        if fps_show > 0 {
                            fps_count += 1;
                            if fps_show == 2 { last_present_time = now2; }
                        }
                    }
                }
            }
        }
    }

    // Stop recording if active
    if let Some(ref mut rec) = recorder {
        let path = rec.stop();
        eprintln!("recording saved: {}", path.display());
    }

    // Finalize analysis strip if active
    if let Some(ref mut strip) = strip_mode {
        match strip.finalize(strip_last_mode) {
            Ok(Some(name)) => eprintln!("analysis strip: flushed final {}", name),
            Ok(None) => {}
            Err(e) => eprintln!("analysis strip flush: {}", e),
        }
        eprintln!("analysis strip: {} frames, {} strips",
            strip.total_frames(), strip.strip_count());
    }

    // Stop streaming if active
    if let Some(ref mut s) = streamer {
        s.stop();
        eprintln!("streaming: stopped");
    }

    // Stop client receiver if active
    if let Some(ref mut rx) = client_receiver {
        rx.stop();
    }
    drop(client_tex);

    // Drop frame generator before GL renderer (uses same GL context)
    drop(frame_gen);

    // Drop GL renderer first (destroys EGLImages before we close the FDs)
    drop(gl_renderer);

    // Close DMA-BUF file descriptors (Linux only)
    #[cfg(target_os = "linux")]
    for fd in &dmabuf_fds {
        unsafe { libc::close(*fd); }
    }

    Ok(())
}

/// NV12 texture update via raw SDL2 (SDL_UpdateNVTexture).
fn update_nv12_texture(
    texture: &mut sdl2::render::Texture,
    ptr: *const u8,
    width: u32,
    height: u32,
) {
    let y_pitch = width as i32;
    let uv_pitch = width as i32;
    let y_plane_size = (width * height) as usize;

    unsafe {
        sdl2_sys::SDL_UpdateNVTexture(
            texture.raw(),
            std::ptr::null(),
            ptr,
            y_pitch,
            ptr.add(y_plane_size),
            uv_pitch,
        );
    }
}

/// Upload a frame from a byte slice to the texture (used for plugin output).
fn update_texture_from_slice(
    texture: &mut sdl2::render::Texture,
    data: &[u8],
    width: u32,
    height: u32,
    pixfmt: u32,
) {
    match pixfmt {
        V4L2_PIX_FMT_NV12 => {
            update_nv12_texture(texture, data.as_ptr(), width, height);
        }
        V4L2_PIX_FMT_YUYV | V4L2_PIX_FMT_UYVY => {
            let pitch = width as usize * 2;
            let _ = texture.update(None, data, pitch);
        }
        V4L2_PIX_FMT_XRGB32 => {
            let pitch = width as usize * 4;
            let _ = texture.update(None, data, pitch);
        }
        PIXFMT_RGB24 => {
            let pitch = width as usize * 3;
            let _ = texture.update(None, data, pitch);
        }
        _ => {}
    }
}

/// Build a 256-entry contrast+gamma LUT for the Y (luma) channel.
/// Contrast is applied around the midpoint (128), then gamma correction.
/// Brightness is handled separately by SDL texture color modulation.
fn build_contrast_lut(contrast: f32, gamma: f32) -> [u8; 256] {
    let inv_gamma = 1.0 / gamma;
    let mut lut = [0u8; 256];
    for i in 0..256 {
        let v = i as f32;
        let c = ((v - 128.0) * contrast + 128.0).clamp(0.0, 255.0);
        let g = (c / 255.0).powf(inv_gamma) * 255.0;
        lut[i] = g.clamp(0.0, 255.0) as u8;
    }
    lut
}

/// Apply a LUT to the Y (luma) channel of a frame buffer, in-place.
fn apply_y_lut(data: &mut [u8], width: u32, height: u32, pixfmt: u32, lut: &[u8; 256]) {
    match pixfmt {
        V4L2_PIX_FMT_NV12 => {
            let y_size = (width * height) as usize;
            let end = y_size.min(data.len());
            for b in &mut data[..end] {
                *b = lut[*b as usize];
            }
        }
        V4L2_PIX_FMT_YUYV => {
            for i in (0..data.len()).step_by(2) {
                data[i] = lut[data[i] as usize];
            }
        }
        V4L2_PIX_FMT_UYVY => {
            for i in (1..data.len()).step_by(2) {
                data[i] = lut[data[i] as usize];
            }
        }
        _ => {}
    }
}
