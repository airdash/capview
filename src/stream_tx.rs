//! UDP streaming sender.
//!
//! Runs a background thread that:
//! 1. Receives raw YUV frames from the capture loop via a bounded channel
//! 2. Converts to RGB + JPEG-encodes via libturbojpeg
//! 3. Fragments and sends over UDP to all connected clients
//!
//! Clients "connect" by sending any UDP packet to the sender's port.
//! The sender tracks client addresses and removes stale ones after a timeout.

use std::collections::HashMap;
use std::net::{SocketAddr, UdpSocket};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc::{self, SyncSender, TrySendError};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::capture::{V4L2_PIX_FMT_NV12, V4L2_PIX_FMT_YUYV, V4L2_PIX_FMT_UYVY, V4L2_PIX_FMT_XRGB32, V4L2_PIX_FMT_P010, PIXFMT_RGB24};
use crate::net;

/// Channel depth — small to minimize latency.
const CHANNEL_DEPTH: usize = 2;

/// Clients are removed after this many seconds of no heartbeat.
const CLIENT_TIMEOUT_SECS: u64 = 10;

/// Hard cap on tracked clients. New registrations past this are dropped to
/// keep the HashMap bounded under hostile traffic (spoofed source addresses).
const MAX_CLIENTS: usize = 256;

/// Heartbeat / client-registration recv size (tiny).
const HEARTBEAT_BUF: usize = 64;

pub struct StreamSender {
    tx: Option<SyncSender<Arc<[u8]>>>,
    thread: Option<std::thread::JoinHandle<()>>,
    running: Arc<AtomicBool>,
    client_count: Arc<AtomicU32>,
    port: u16,
}

impl StreamSender {
    /// Start the sender on `bind_addr` (e.g. "0.0.0.0:9000").
    ///
    /// `width`, `height`, `fps`, `pixfmt` describe the incoming frames.
    /// `quality` is the JPEG quality (1-100).
    pub fn start(
        bind_addr: &str,
        width: u32,
        height: u32,
        fps: u32,
        pixfmt: u32,
        quality: u32,
        debug: bool,
    ) -> anyhow::Result<Self> {
        let sock = UdpSocket::bind(bind_addr)?;
        let actual_port = sock.local_addr()?.port();
        sock.set_nonblocking(true)?;

        // Increase send buffer for burst of fragments
        let _ = set_sock_sndbuf(&sock, 1024 * 1024);

        let running = Arc::new(AtomicBool::new(true));
        let client_count = Arc::new(AtomicU32::new(0));
        let (tx, rx) = mpsc::sync_channel::<Arc<[u8]>>(CHANNEL_DEPTH);

        let r = running.clone();
        let cc = client_count.clone();

        let thread = std::thread::Builder::new()
            .name("stream-tx".into())
            .spawn(move || {
                sender_loop(sock, rx, r, cc, width, height, fps, pixfmt, quality, debug);
            })?;

        eprintln!("streaming: sender listening on :{}", actual_port);

        Ok(Self {
            tx: Some(tx),
            thread: Some(thread),
            running,
            client_count,
            port: actual_port,
        })
    }

    /// Send a frame to the streaming thread (non-blocking, drops if full).
    /// Accepts a shared Arc so the caller can avoid duplicate copies.
    pub fn send_frame(&self, data: Arc<[u8]>) -> bool {
        if let Some(ref tx) = self.tx {
            match tx.try_send(data) {
                Ok(()) => true,
                Err(TrySendError::Full(_)) => true, // drop frame, not an error
                Err(TrySendError::Disconnected(_)) => false,
            }
        } else {
            false
        }
    }

    /// Number of currently connected clients.
    pub fn client_count(&self) -> u32 {
        self.client_count.load(Ordering::Relaxed)
    }

    /// The port we're actually bound to.
    pub fn port(&self) -> u16 {
        self.port
    }

    /// Stop the sender and join the thread.
    pub fn stop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        // Drop the sender side to unblock recv
        self.tx.take();
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

impl Drop for StreamSender {
    fn drop(&mut self) {
        self.stop();
    }
}

// ── Background thread ───────────────────────────────────────────────

fn sender_loop(
    sock: UdpSocket,
    rx: mpsc::Receiver<Arc<[u8]>>,
    running: Arc<AtomicBool>,
    client_count: Arc<AtomicU32>,
    width: u32,
    height: u32,
    fps: u32,
    pixfmt: u32,
    quality: u32,
    debug: bool,
) {
    // Init turbojpeg
    let tj = match crate::turbojpeg::TurboJpeg::new() {
        Ok(t) => t,
        Err(e) => {
            eprintln!("streaming: turbojpeg init failed: {}", e);
            return;
        }
    };

    let mut seq: u32 = 0;
    let mut clients: HashMap<SocketAddr, Instant> = HashMap::new();
    let mut heartbeat_buf = [0u8; HEARTBEAT_BUF];
    let mut pkt_buf = vec![0u8; net::HEADER_SIZE + net::MAX_PAYLOAD];
    let mut rgb_buf: Vec<u8> = Vec::new();
    let mut got_first_frame_in = false;
    let mut got_first_send = false;

    while running.load(Ordering::Relaxed) {
        // Check for new/returning clients (non-blocking)
        loop {
            match sock.recv_from(&mut heartbeat_buf) {
                Ok((_n, addr)) => {
                    let is_new = !clients.contains_key(&addr);
                    if is_new && clients.len() >= MAX_CLIENTS {
                        // Known client count already at cap; ignore new addrs
                        // until existing ones time out. Prevents unbounded
                        // growth under spoofed-source UDP traffic.
                        if debug {
                            eprintln!("streaming: client cap reached, ignoring: {}", addr);
                        }
                        continue;
                    }
                    clients.insert(addr, Instant::now());
                    if is_new {
                        client_count.store(clients.len() as u32, Ordering::Relaxed);
                        eprintln!("streaming: client connected: {}", addr);
                    }
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(_) => break,
            }
        }

        // Expire stale clients
        let now = Instant::now();
        let timeout = Duration::from_secs(CLIENT_TIMEOUT_SECS);
        clients.retain(|addr, last_seen| {
            let alive = now.duration_since(*last_seen) < timeout;
            if !alive && debug {
                eprintln!("streaming: client timed out: {}", addr);
            }
            alive
        });
        client_count.store(clients.len() as u32, Ordering::Relaxed);

        // Wait for a frame (with timeout so we can check heartbeats)
        let frame = match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(f) => f,
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        };

        if !got_first_frame_in {
            eprintln!(
                "streaming: first frame queued for send ({}x{}, pixfmt 0x{:08x}, {} bytes)",
                width, height, pixfmt, frame.len(),
            );
            got_first_frame_in = true;
        }

        // No clients? Skip encoding entirely.
        if clients.is_empty() {
            continue;
        }

        // Convert YUV→RGB
        yuv_to_rgb(&frame, width, height, pixfmt, &mut rgb_buf);

        // JPEG encode
        let jpeg = match tj.compress(&rgb_buf, width, height, quality) {
            Ok(j) => j,
            Err(e) => {
                eprintln!("streaming: jpeg encode error: {}", e);
                continue;
            }
        };

        // Fragment and send
        let frag_count = (jpeg.len() + net::MAX_PAYLOAD - 1) / net::MAX_PAYLOAD;
        if frag_count > net::MAX_FRAGMENT_COUNT {
            eprintln!("streaming: frame too large to fragment ({} bytes, {} fragments — max {})",
                jpeg.len(), frag_count, net::MAX_FRAGMENT_COUNT);
            continue;
        }

        for frag_idx in 0..frag_count {
            let start = frag_idx * net::MAX_PAYLOAD;
            let end = (start + net::MAX_PAYLOAD).min(jpeg.len());
            let payload = &jpeg[start..end];

            net::write_header(
                &mut pkt_buf,
                seq,
                frag_idx as u16,
                frag_count as u16,
                width as u16,
                height as u16,
                fps.min(255) as u8,
                quality.min(255) as u8,
            );
            pkt_buf[net::HEADER_SIZE..net::HEADER_SIZE + payload.len()]
                .copy_from_slice(payload);

            let pkt = &pkt_buf[..net::HEADER_SIZE + payload.len()];
            for addr in clients.keys() {
                let _ = sock.send_to(pkt, addr);
            }
        }

        if !got_first_send {
            eprintln!(
                "streaming: first frame broadcast to {} client(s) (jpeg {}b, {} fragment(s))",
                clients.len(), jpeg.len(), frag_count,
            );
            got_first_send = true;
        }

        seq = seq.wrapping_add(1);
    }
}

// ── YUV→RGB conversion (duplicated from screenshot.rs to avoid coupling) ──

fn yuv_to_rgb(data: &[u8], width: u32, height: u32, pixfmt: u32, out: &mut Vec<u8>) {
    let npixels = (width * height) as usize;
    out.resize(npixels * 3, 0);

    match pixfmt {
        V4L2_PIX_FMT_NV12 => {
            let w = width as usize;
            let h = height as usize;
            for y in 0..h {
                for x in 0..w {
                    let yi = y * w + x;
                    let uv_base = w * h + (y / 2) * w + (x & !1);
                    let yv = data[yi] as f32;
                    let u = data[uv_base] as f32 - 128.0;
                    let v = data[uv_base + 1] as f32 - 128.0;
                    let r = (yv + 1.402 * v).clamp(0.0, 255.0) as u8;
                    let g = (yv - 0.344 * u - 0.714 * v).clamp(0.0, 255.0) as u8;
                    let b = (yv + 1.772 * u).clamp(0.0, 255.0) as u8;
                    let oi = yi * 3;
                    out[oi] = r;
                    out[oi + 1] = g;
                    out[oi + 2] = b;
                }
            }
        }
        V4L2_PIX_FMT_YUYV => {
            let mut oi = 0;
            for i in (0..data.len()).step_by(4) {
                if i + 3 >= data.len() { break; }
                let y0 = data[i] as f32;
                let u  = data[i + 1] as f32 - 128.0;
                let y1 = data[i + 2] as f32;
                let v  = data[i + 3] as f32 - 128.0;
                for &yv in &[y0, y1] {
                    if oi + 2 < out.len() {
                        out[oi]     = (yv + 1.402 * v).clamp(0.0, 255.0) as u8;
                        out[oi + 1] = (yv - 0.344 * u - 0.714 * v).clamp(0.0, 255.0) as u8;
                        out[oi + 2] = (yv + 1.772 * u).clamp(0.0, 255.0) as u8;
                        oi += 3;
                    }
                }
            }
        }
        V4L2_PIX_FMT_UYVY => {
            let mut oi = 0;
            for i in (0..data.len()).step_by(4) {
                if i + 3 >= data.len() { break; }
                let u  = data[i] as f32 - 128.0;
                let y0 = data[i + 1] as f32;
                let v  = data[i + 2] as f32 - 128.0;
                let y1 = data[i + 3] as f32;
                for &yv in &[y0, y1] {
                    if oi + 2 < out.len() {
                        out[oi]     = (yv + 1.402 * v).clamp(0.0, 255.0) as u8;
                        out[oi + 1] = (yv - 0.344 * u - 0.714 * v).clamp(0.0, 255.0) as u8;
                        out[oi + 2] = (yv + 1.772 * u).clamp(0.0, 255.0) as u8;
                        oi += 3;
                    }
                }
            }
        }
        V4L2_PIX_FMT_XRGB32 => {
            for i in 0..(npixels.min(data.len() / 4)) {
                let si = i * 4;
                out[i * 3] = data[si + 2];     // R
                out[i * 3 + 1] = data[si + 1]; // G
                out[i * 3 + 2] = data[si];     // B
            }
        }
        V4L2_PIX_FMT_P010 => {
            let w = width as usize;
            let h = height as usize;
            let y_plane = &data[..w * h * 2];
            let uv_plane = &data[w * h * 2..];
            for row in 0..h {
                for col in 0..w {
                    let yi = (row * w + col) * 2;
                    let yv = y_plane.get(yi + 1).copied().unwrap_or(0) as f32;
                    let uvi = (row / 2) * w * 2 + (col & !1) * 2;
                    let u = uv_plane.get(uvi + 1).copied().unwrap_or(128) as f32 - 128.0;
                    let v = uv_plane.get(uvi + 3).copied().unwrap_or(128) as f32 - 128.0;
                    let oi = (row * w + col) * 3;
                    out[oi] = (yv + 1.402 * v).clamp(0.0, 255.0) as u8;
                    out[oi + 1] = (yv - 0.344 * u - 0.714 * v).clamp(0.0, 255.0) as u8;
                    out[oi + 2] = (yv + 1.772 * u).clamp(0.0, 255.0) as u8;
                }
            }
        }
        PIXFMT_RGB24 => {
            let n = (npixels * 3).min(data.len());
            out[..n].copy_from_slice(&data[..n]);
        }
        _ => {}
    }
}

fn set_sock_sndbuf(sock: &UdpSocket, size: i32) -> std::io::Result<()> {
    use std::os::unix::io::AsRawFd;
    let fd = sock.as_raw_fd();
    let ret = unsafe {
        libc::setsockopt(
            fd,
            libc::SOL_SOCKET,
            libc::SO_SNDBUF,
            &size as *const _ as *const _,
            std::mem::size_of::<i32>() as u32,
        )
    };
    if ret != 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}
