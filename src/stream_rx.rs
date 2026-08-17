//! UDP streaming receiver.
//!
//! Receives JPEG-fragmented frames from a capview sender, reassembles
//! them, and provides complete JPEG frames to the main loop for display.
//!
//! Runs a background receive thread that writes complete frames to a
//! channel.  The main loop polls for decoded RGB frames.

use std::net::{SocketAddr, UdpSocket};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::net;

/// Channel depth for complete frames.
const FRAME_DEPTH: usize = 2;

/// How often to send heartbeat/registration to the sender.
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(3);

/// Maximum reassembly buffer for one frame.
const MAX_FRAME_BYTES: usize = 4 * 1024 * 1024; // 4MB

/// A complete decoded frame ready for display.
pub struct RxFrame {
    pub rgb: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub fps: u8,
}

pub struct StreamReceiver {
    rx: Receiver<RxFrame>,
    thread: Option<std::thread::JoinHandle<()>>,
    running: Arc<AtomicBool>,
}

impl StreamReceiver {
    /// Start receiving from `sender_addr` (e.g. "192.168.1.50:9000").
    ///
    /// Binds to `0.0.0.0:0` (any port) and sends a registration packet
    /// to the sender so it knows where to stream to.
    pub fn start(sender_addr: &str, debug: bool) -> anyhow::Result<Self> {
        let sender: SocketAddr = sender_addr.parse()
            .map_err(|e| anyhow::anyhow!("bad address '{}': {}", sender_addr, e))?;

        let sock = UdpSocket::bind("0.0.0.0:0")?;
        sock.set_read_timeout(Some(Duration::from_millis(200)))?;

        // Increase receive buffer
        let _ = set_sock_rcvbuf(&sock, 2 * 1024 * 1024);

        // Send initial registration
        let _ = sock.send_to(b"hello", sender);

        let running = Arc::new(AtomicBool::new(true));
        let (tx, rx) = mpsc::sync_channel::<RxFrame>(FRAME_DEPTH);

        let r = running.clone();
        let thread = std::thread::Builder::new()
            .name("stream-rx".into())
            .spawn(move || {
                receiver_loop(sock, sender, tx, r, debug);
            })?;

        eprintln!("streaming: connecting to {}", sender_addr);

        Ok(Self { rx, thread: Some(thread), running })
    }

    /// Poll for a new frame (non-blocking).
    pub fn try_recv(&self) -> Option<RxFrame> {
        self.rx.try_recv().ok()
    }

    /// Stop the receiver.
    pub fn stop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

impl Drop for StreamReceiver {
    fn drop(&mut self) {
        self.stop();
    }
}

// ── Background receive thread ───────────────────────────────────────

fn receiver_loop(
    sock: UdpSocket,
    sender: SocketAddr,
    tx: SyncSender<RxFrame>,
    running: Arc<AtomicBool>,
    debug: bool,
) {
    let tj = match crate::turbojpeg::TurboJpeg::new() {
        Ok(t) => t,
        Err(e) => {
            eprintln!("streaming: turbojpeg init failed: {}", e);
            return;
        }
    };

    let mut recv_buf = vec![0u8; net::HEADER_SIZE + net::MAX_PAYLOAD + 128];
    let mut last_heartbeat = Instant::now();

    // Reassembly state
    let mut asm_seq: u32 = 0;
    let mut asm_count: usize = 0;
    let mut asm_received: usize = 0;
    let mut asm_frags: Vec<Option<Vec<u8>>> = Vec::new();
    let mut _asm_width: u16 = 0;
    let mut _asm_height: u16 = 0;
    let mut asm_fps: u8 = 0;

    // Always-on diagnostic markers so the user can tell where the pipeline
    // broke without having to enable --debug.
    let mut got_first_packet = false;
    let mut got_first_frame = false;
    let mut last_packet_at = Instant::now();
    let mut stall_warned = false;

    while running.load(Ordering::Relaxed) {
        // Periodic heartbeat
        if last_heartbeat.elapsed() >= HEARTBEAT_INTERVAL {
            let _ = sock.send_to(b"ping", sender);
            last_heartbeat = Instant::now();
        }

        // Stall warning if we got at least one packet then stopped seeing them.
        if got_first_packet && !stall_warned
            && last_packet_at.elapsed() > Duration::from_secs(5)
        {
            eprintln!(
                "streaming: stalled — no packets for {}s (sender stopped, packet loss, or path MTU dropping fragments)",
                last_packet_at.elapsed().as_secs()
            );
            stall_warned = true;
        }

        // Receive packet
        let n = match sock.recv(&mut recv_buf) {
            Ok(n) => n,
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock
                || e.kind() == std::io::ErrorKind::TimedOut => continue,
            Err(e) => {
                if running.load(Ordering::Relaxed) {
                    if debug { eprintln!("streaming: recv error: {}", e); }
                }
                continue;
            }
        };

        let hdr = match net::parse_header(&recv_buf[..n]) {
            Some(h) => h,
            None => {
                if !got_first_packet {
                    eprintln!("streaming: got {}-byte datagram with no/bad CVSW header", n);
                }
                continue;
            }
        };

        if !got_first_packet {
            eprintln!(
                "streaming: first frame packet from sender (seq={}, frag={}/{}, {}x{})",
                hdr.seq, hdr.frag_idx, hdr.frag_count, hdr.width, hdr.height,
            );
            got_first_packet = true;
        }
        last_packet_at = Instant::now();
        stall_warned = false;

        let payload = &recv_buf[net::HEADER_SIZE..n];

        // New frame? Reset reassembly.
        if hdr.seq != asm_seq || hdr.frag_count as usize != asm_count {
            // If we were mid-assembly and got a newer sequence, discard
            asm_seq = hdr.seq;
            asm_count = hdr.frag_count as usize;
            asm_received = 0;
            asm_frags.clear();
            asm_frags.resize(asm_count, None);
            _asm_width = hdr.width;
            _asm_height = hdr.height;
            asm_fps = hdr.fps;
        }

        // Store fragment
        let idx = hdr.frag_idx as usize;
        if idx < asm_count {
            if asm_frags[idx].is_none() {
                asm_received += 1;
            }
            asm_frags[idx] = Some(payload.to_vec());
        }

        // Complete frame?
        if asm_received == asm_count && asm_count > 0 {
            // Reassemble JPEG
            let total_size: usize = asm_frags.iter()
                .filter_map(|f| f.as_ref())
                .map(|f| f.len())
                .sum();

            if total_size > MAX_FRAME_BYTES {
                if debug { eprintln!("streaming: frame too large ({})", total_size); }
                asm_received = 0;
                asm_count = 0;
                continue;
            }

            let mut jpeg = Vec::with_capacity(total_size);
            for frag in &asm_frags {
                if let Some(data) = frag {
                    jpeg.extend_from_slice(data);
                }
            }

            // Decode JPEG → RGB
            match tj.decompress(&jpeg) {
                Ok((rgb, w, h)) => {
                    if !got_first_frame {
                        eprintln!(
                            "streaming: first complete frame decoded ({}x{}, jpeg {}b → rgb {}b)",
                            w, h, jpeg.len(), rgb.len(),
                        );
                        got_first_frame = true;
                    }
                    let frame = RxFrame {
                        rgb,
                        width: w,
                        height: h,
                        fps: asm_fps,
                    };
                    match tx.try_send(frame) {
                        Ok(()) => {}
                        Err(TrySendError::Full(_)) => {} // drop old frame
                        Err(TrySendError::Disconnected(_)) => break,
                    }
                }
                Err(e) => {
                    eprintln!("streaming: jpeg decode error: {} ({} fragments, {}b)",
                        e, asm_count, jpeg.len());
                }
            }

            // Reset for next frame
            asm_received = 0;
            asm_count = 0;
        }
    }
}

fn set_sock_rcvbuf(sock: &UdpSocket, size: i32) -> std::io::Result<()> {
    use std::os::unix::io::AsRawFd;
    let fd = sock.as_raw_fd();
    let ret = unsafe {
        libc::setsockopt(
            fd,
            libc::SOL_SOCKET,
            libc::SO_RCVBUF,
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
