use anyhow::{bail, Result};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::capture::{V4L2_PIX_FMT_NV12, V4L2_PIX_FMT_YUYV, V4L2_PIX_FMT_UYVY, V4L2_PIX_FMT_XRGB32, V4L2_PIX_FMT_P010, PIXFMT_RGB24};

/// Full-channel depth — filled buffers pending write to ffmpeg stdin.
const CHANNEL_DEPTH: usize = 4;
/// Total pool size. Must exceed CHANNEL_DEPTH so render always has a free
/// buffer to fill while the writer is draining the full channel.
const POOL_SIZE: usize = 6;

fn frame_byte_size(width: u32, height: u32, pixfmt: u32) -> usize {
    let pixels = (width as usize) * (height as usize);
    match pixfmt {
        V4L2_PIX_FMT_NV12 => pixels * 3 / 2,
        V4L2_PIX_FMT_YUYV | V4L2_PIX_FMT_UYVY => pixels * 2,
        V4L2_PIX_FMT_XRGB32 => pixels * 4,
        V4L2_PIX_FMT_P010 => pixels * 3, // 4:2:0 × 2 bytes/sample
        PIXFMT_RGB24 => pixels * 3,
        _ => pixels * 4, // conservative upper bound
    }
}

pub struct Recorder {
    full_tx: Option<SyncSender<Vec<u8>>>,   // render → writer (filled)
    free_rx: Option<Receiver<Vec<u8>>>,     // writer → render (returned)
    free_tx: Option<SyncSender<Vec<u8>>>,   // render-side return-on-failure
    writer_thread: Option<std::thread::JoinHandle<()>>,
    child: Option<Child>,
    path: PathBuf,
    dropped: Arc<AtomicUsize>,
}

/// Default output directory for recordings.
pub fn video_dir() -> PathBuf {
    if let Ok(home) = std::env::var("HOME") {
        PathBuf::from(home).join("Videos")
    } else {
        PathBuf::from(".")
    }
}

impl Recorder {
    pub fn start(
        width: u32,
        height: u32,
        fps: u32,
        pixfmt: u32,
        output_dir: &Path,
        audio_source: Option<&str>,
        output_size: Option<(u32, u32)>,
        debug: bool,
    ) -> Result<Self> {
        let pix_fmt = match pixfmt {
            V4L2_PIX_FMT_NV12 => "nv12",
            V4L2_PIX_FMT_YUYV => "yuyv422",
            V4L2_PIX_FMT_UYVY => "uyvy422",
            V4L2_PIX_FMT_XRGB32 => "bgr0",
            V4L2_PIX_FMT_P010 => "p010le",
            PIXFMT_RGB24 => "rgb24",
            _ => bail!("unsupported pixel format for recording"),
        };

        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let filename = format!("capview_{}.mp4", ts);
        let path = output_dir.join(&filename);
        std::fs::create_dir_all(output_dir)?;

        let stderr = if debug { Stdio::inherit() } else { Stdio::null() };

        let mut args: Vec<String> = vec![
            // Video input (pipe)
            "-f".into(), "rawvideo".into(),
            "-pix_fmt".into(), pix_fmt.into(),
            "-s".into(), format!("{}x{}", width, height),
            "-r".into(), format!("{}", fps),
            "-i".into(), "pipe:0".into(),
        ];

        // Optional PulseAudio audio input
        let has_audio = audio_source.is_some();
        if let Some(src) = audio_source {
            args.extend([
                "-f".into(), "pulse".into(),
                "-i".into(), src.into(),
            ]);
        }

        // Optional output scaling (window-size recording)
        if let Some((ow, oh)) = output_size {
            // Force even dimensions for h264
            let ow = ow & !1;
            let oh = oh & !1;
            args.extend([
                "-vf".into(), format!("scale={}:{}", ow, oh),
            ]);
        }

        // Encoding settings
        args.extend([
            "-c:v".into(), "libx264".into(),
            "-preset".into(), "ultrafast".into(),
            "-tune".into(), "zerolatency".into(),
            "-crf".into(), "18".into(),
            "-pix_fmt".into(), "yuv420p".into(),
        ]);

        if has_audio {
            args.extend([
                "-map".into(), "0:v".into(),
                "-map".into(), "1:a".into(),
                "-c:a".into(), "aac".into(),
                "-b:a".into(), "192k".into(),
                "-shortest".into(),
            ]);
        }

        args.extend([
            "-movflags".into(), "+faststart".into(),
            "-y".into(),
        ]);
        args.push(path.to_string_lossy().into());

        let mut child = Command::new("ffmpeg")
            .args(&args)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(stderr)
            .spawn()?;

        let mut stdin = child.stdin.take()
            .ok_or_else(|| anyhow::anyhow!("ffmpeg stdin not available"))?;

        // Pool: pre-allocated frame buffers reused for the whole session.
        // Keeps frame-sized mallocs out of the render thread's hot path.
        let frame_size = frame_byte_size(width, height, pixfmt);
        let (full_tx, full_rx) = mpsc::sync_channel::<Vec<u8>>(CHANNEL_DEPTH);
        let (free_tx, free_rx) = mpsc::sync_channel::<Vec<u8>>(POOL_SIZE);
        for _ in 0..POOL_SIZE {
            free_tx.send(vec![0u8; frame_size])
                .expect("pool pre-populate fits channel capacity");
        }

        let dropped = Arc::new(AtomicUsize::new(0));
        let dropped2 = dropped.clone();
        let is_debug = debug;
        let free_tx_writer = free_tx.clone();

        let writer_thread = std::thread::spawn(move || {
            crate::priority::avoid_render_core();
            while let Ok(buf) = full_rx.recv() {
                if stdin.write_all(&buf).is_err() {
                    break;
                }
                // Return buffer to pool; ignoring full means the pool is oversized,
                // which can't happen given POOL_SIZE == capacity(free_rx).
                let _ = free_tx_writer.try_send(buf);
            }
            drop(stdin); // close pipe → ffmpeg finishes
            let d = dropped2.load(Ordering::Relaxed);
            if is_debug && d > 0 {
                eprintln!("recording: dropped {} frames (writer couldn't keep up)", d);
            }
        });

        Ok(Recorder {
            full_tx: Some(full_tx),
            free_rx: Some(free_rx),
            free_tx: Some(free_tx),
            writer_thread: Some(writer_thread),
            child: Some(child),
            path,
            dropped,
        })
    }

    /// Queue a raw frame for the writer thread (non-blocking, zero-alloc).
    /// Copies into a pooled buffer; drops the frame if the pool is empty or
    /// the writer is backed up. Returns false only if the pipe broke.
    pub fn write_frame(&self, data: &[u8]) -> bool {
        let (Some(full_tx), Some(free_rx), Some(free_tx)) = (
            self.full_tx.as_ref(),
            self.free_rx.as_ref(),
            self.free_tx.as_ref(),
        ) else {
            return false;
        };

        let mut buf = match free_rx.try_recv() {
            Ok(b) => b,
            Err(_) => {
                // Writer hasn't returned a buffer yet — drop frame, no alloc.
                self.dropped.fetch_add(1, Ordering::Relaxed);
                return true;
            }
        };

        buf.clear();
        buf.extend_from_slice(data);

        match full_tx.try_send(buf) {
            Ok(()) => true,
            Err(TrySendError::Full(buf)) => {
                // Full channel saturated; return buffer to pool, drop frame.
                self.dropped.fetch_add(1, Ordering::Relaxed);
                let _ = free_tx.try_send(buf);
                true
            }
            Err(TrySendError::Disconnected(_)) => false,
        }
    }

    /// Stop recording: close the channel, join writer thread, reap ffmpeg.
    pub fn stop(&mut self) -> PathBuf {
        self.cleanup();
        self.path.clone()
    }

    fn cleanup(&mut self) {
        // Drop senders → writer_rx.recv() returns Err → writer drains and exits
        self.full_tx.take();
        self.free_tx.take();

        // Wait for writer thread to finish flushing
        if let Some(t) = self.writer_thread.take() {
            let _ = t.join();
        }

        let d = self.dropped.load(Ordering::Relaxed);
        if d > 0 {
            eprintln!("recording: {} frames dropped total", d);
        }

        // Reap ffmpeg in background (it may still be finalizing the container)
        if let Some(mut child) = self.child.take() {
            std::thread::spawn(move || {
                let _ = child.wait();
            });
        }
    }
}

impl Drop for Recorder {
    fn drop(&mut self) {
        self.cleanup();
    }
}
