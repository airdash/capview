use anyhow::{bail, Context, Result};
use std::ffi::CString;
use std::os::unix::io::RawFd;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};
use std::sync::Arc;

use crate::capture::{
    V4L2_PIX_FMT_NV12, V4L2_PIX_FMT_YUYV, V4L2_PIX_FMT_UYVY,
    V4L2_PIX_FMT_XRGB32, V4L2_PIX_FMT_P010, PIXFMT_RGB24,
};

// V4L2 ioctl numbers and flags needed for an output device. Kernel ABI;
// kept local to keep capture.rs focused on the capture path.
const VIDIOC_QUERYCAP: libc::c_ulong = 0x80685600;
const VIDIOC_S_FMT:    libc::c_ulong = 0xc0d05605;

const V4L2_CAP_VIDEO_CAPTURE: u32 = 0x00000001;
const V4L2_CAP_VIDEO_OUTPUT:  u32 = 0x00000002;

const V4L2_BUF_TYPE_VIDEO_OUTPUT: u32 = 2;
const V4L2_FIELD_NONE: u32 = 1;

#[repr(C)]
struct V4l2Capability {
    driver: [u8; 16],
    card: [u8; 32],
    bus_info: [u8; 32],
    version: u32,
    capabilities: u32,
    device_caps: u32,
    reserved: [u32; 3],
}

#[repr(C)]
struct V4l2PixFormat {
    width: u32,
    height: u32,
    pixelformat: u32,
    field: u32,
    bytesperline: u32,
    sizeimage: u32,
    colorspace: u32,
    private: u32,
    flags: u32,
    ycbcr_enc: u32,
    quantization: u32,
    xfer_func: u32,
}

#[repr(C)]
struct V4l2Format {
    type_: u32,
    _pad: u32,
    pix: V4l2PixFormat,
    _remainder: [u8; 152],
}

const CHANNEL_DEPTH: usize = 4;
const POOL_SIZE: usize = 6;

fn frame_byte_size(width: u32, height: u32, pixfmt: u32) -> usize {
    let pixels = (width as usize) * (height as usize);
    match pixfmt {
        V4L2_PIX_FMT_NV12 => pixels * 3 / 2,
        V4L2_PIX_FMT_YUYV | V4L2_PIX_FMT_UYVY => pixels * 2,
        V4L2_PIX_FMT_XRGB32 => pixels * 4,
        V4L2_PIX_FMT_P010 => pixels * 3,
        PIXFMT_RGB24 => pixels * 3,
        _ => pixels * 4,
    }
}

fn bytes_per_line(width: u32, pixfmt: u32) -> u32 {
    match pixfmt {
        V4L2_PIX_FMT_NV12 => width,
        V4L2_PIX_FMT_YUYV | V4L2_PIX_FMT_UYVY => width * 2,
        V4L2_PIX_FMT_XRGB32 => width * 4,
        V4L2_PIX_FMT_P010 => width * 2,
        PIXFMT_RGB24 => width * 3,
        _ => width * 4,
    }
}

pub struct VirtualWebcam {
    full_tx: Option<SyncSender<Vec<u8>>>,
    free_rx: Option<Receiver<Vec<u8>>>,
    free_tx: Option<SyncSender<Vec<u8>>>,
    writer_thread: Option<std::thread::JoinHandle<()>>,
    device_path: String,
    dropped: Arc<AtomicUsize>,
}

impl VirtualWebcam {
    pub fn start(
        device_path: &str,
        width: u32,
        height: u32,
        pixfmt: u32,
    ) -> Result<Self> {
        let fd = open_output_device(device_path)?;

        // From here, we own fd; clean up on any early error.
        if let Err(e) = configure_output(fd, width, height, pixfmt) {
            unsafe { libc::close(fd); }
            return Err(e);
        }

        let frame_size = frame_byte_size(width, height, pixfmt);
        let (full_tx, full_rx) = mpsc::sync_channel::<Vec<u8>>(CHANNEL_DEPTH);
        let (free_tx, free_rx) = mpsc::sync_channel::<Vec<u8>>(POOL_SIZE);
        for _ in 0..POOL_SIZE {
            free_tx.send(vec![0u8; frame_size])
                .expect("pool pre-populate fits channel capacity");
        }

        let dropped = Arc::new(AtomicUsize::new(0));
        let dropped2 = dropped.clone();
        let free_tx_writer = free_tx.clone();
        let thread_fd = fd;
        let thread_path = device_path.to_string();

        let writer_thread = std::thread::spawn(move || {
            crate::priority::avoid_render_core();
            while let Ok(buf) = full_rx.recv() {
                let mut off = 0;
                let mut broke = false;
                while off < buf.len() {
                    let ret = unsafe {
                        libc::write(
                            thread_fd,
                            buf.as_ptr().add(off) as *const libc::c_void,
                            buf.len() - off,
                        )
                    };
                    if ret < 0 {
                        let err = std::io::Error::last_os_error();
                        if err.raw_os_error() == Some(libc::EINTR) { continue; }
                        eprintln!("virtual_webcam: write to {} failed: {}", thread_path, err);
                        broke = true;
                        break;
                    }
                    off += ret as usize;
                }
                if broke { break; }
                let _ = free_tx_writer.try_send(buf);
            }
            unsafe { libc::close(thread_fd); }
            let d = dropped2.load(Ordering::Relaxed);
            if d > 0 {
                eprintln!("virtual_webcam: dropped {} frames total", d);
            }
        });

        Ok(VirtualWebcam {
            full_tx: Some(full_tx),
            free_rx: Some(free_rx),
            free_tx: Some(free_tx),
            writer_thread: Some(writer_thread),
            device_path: device_path.to_string(),
            dropped,
        })
    }

    /// Copy a frame into a pooled buffer for the writer thread (non-blocking,
    /// zero-alloc). Drops the frame if the pool is empty or the writer is
    /// backed up. Returns false only if the writer exited (device error).
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
                self.dropped.fetch_add(1, Ordering::Relaxed);
                return true;
            }
        };

        buf.clear();
        buf.extend_from_slice(data);

        match full_tx.try_send(buf) {
            Ok(()) => true,
            Err(TrySendError::Full(buf)) => {
                self.dropped.fetch_add(1, Ordering::Relaxed);
                let _ = free_tx.try_send(buf);
                true
            }
            Err(TrySendError::Disconnected(_)) => false,
        }
    }

    pub fn device_path(&self) -> &str {
        &self.device_path
    }

    fn cleanup(&mut self) {
        // Dropping senders → writer_rx.recv() errors → writer drains and exits,
        // then closes the device fd itself.
        self.full_tx.take();
        self.free_tx.take();
        if let Some(t) = self.writer_thread.take() {
            let _ = t.join();
        }
    }
}

impl Drop for VirtualWebcam {
    fn drop(&mut self) {
        self.cleanup();
    }
}

fn open_output_device(path: &str) -> Result<RawFd> {
    let c_path = CString::new(path)
        .with_context(|| format!("invalid device path: {}", path))?;
    let fd = unsafe { libc::open(c_path.as_ptr(), libc::O_WRONLY | libc::O_CLOEXEC) };
    if fd < 0 {
        let err = std::io::Error::last_os_error();
        bail!(
            "cannot open {} ({}). Is the v4l2loopback module loaded? Try:\n  \
             sudo modprobe v4l2loopback video_nr=10 card_label=capview exclusive_caps=1",
            path, err
        );
    }
    Ok(fd)
}

fn configure_output(fd: RawFd, width: u32, height: u32, pixfmt: u32) -> Result<()> {
    // Verify this is a V4L2 output device.
    let mut cap: V4l2Capability = unsafe { std::mem::zeroed() };
    let ret = unsafe { libc::ioctl(fd, VIDIOC_QUERYCAP, &mut cap) };
    if ret != 0 {
        bail!("VIDIOC_QUERYCAP failed: {}", std::io::Error::last_os_error());
    }
    // device_caps is preferred over capabilities on modern kernels; fall back if unset.
    let caps = if cap.device_caps != 0 { cap.device_caps } else { cap.capabilities };
    if caps & V4L2_CAP_VIDEO_OUTPUT == 0 {
        // v4l2loopback with exclusive_caps=1 tracks role per open-file. If a
        // consumer is still attached when we re-open after a toggle-off,
        // QUERYCAP can report CAPTURE-only on our O_WRONLY fd until the
        // consumer detaches. S_FMT below is authoritative — if the driver
        // really won't accept output, that ioctl will reject it.
        eprintln!(
            "virtual_webcam: QUERYCAP reports no VIDEO_OUTPUT on {} \
             (consumers still attached?); proceeding to S_FMT anyway",
            if cap.device_caps != 0 { "device_caps" } else { "capabilities" }
        );
    } else if caps & V4L2_CAP_VIDEO_CAPTURE != 0 {
        // v4l2loopback without exclusive_caps=1: device advertises both
        // capture AND output. Chrome/Discord won't list it as a webcam.
        eprintln!(
            "virtual_webcam: WARNING device advertises both VIDEO_CAPTURE and \
             VIDEO_OUTPUT. Most consumers (Discord, Chrome) will not list it. \
             Reload v4l2loopback with exclusive_caps=1."
        );
    }

    // Set the output format to match the capture format pass-through.
    let mut fmt: V4l2Format = unsafe { std::mem::zeroed() };
    fmt.type_ = V4L2_BUF_TYPE_VIDEO_OUTPUT;
    fmt.pix.width = width;
    fmt.pix.height = height;
    fmt.pix.pixelformat = pixfmt;
    fmt.pix.field = V4L2_FIELD_NONE;
    fmt.pix.bytesperline = bytes_per_line(width, pixfmt);
    fmt.pix.sizeimage = frame_byte_size(width, height, pixfmt) as u32;

    let ret = unsafe { libc::ioctl(fd, VIDIOC_S_FMT, &mut fmt) };
    if ret != 0 {
        bail!("VIDIOC_S_FMT failed: {}", std::io::Error::last_os_error());
    }
    if fmt.pix.pixelformat != pixfmt {
        bail!(
            "device did not accept requested pixel format (wanted 0x{:08x}, got 0x{:08x})",
            pixfmt, fmt.pix.pixelformat
        );
    }
    Ok(())
}
