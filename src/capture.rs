use anyhow::{bail, Result};
use std::ffi::CStr;
use std::os::unix::io::RawFd;
use std::ptr;

// ── v4l2 constants (x86_64, from linux/videodev2.h) ──────────────────

const VIDIOC_QUERYCAP: libc::c_ulong = 0x80685600;
const VIDIOC_S_FMT: libc::c_ulong = 0xc0d05605;
const VIDIOC_S_PARM: libc::c_ulong = 0xc0cc5616;
const VIDIOC_REQBUFS: libc::c_ulong = 0xc0145608;
const VIDIOC_QUERYBUF: libc::c_ulong = 0xc0585609;
const VIDIOC_QBUF: libc::c_ulong = 0xc058560f;
const VIDIOC_DQBUF: libc::c_ulong = 0xc0585611;
const VIDIOC_STREAMON: libc::c_ulong = 0x40045612;
const VIDIOC_STREAMOFF: libc::c_ulong = 0x40045613;
const VIDIOC_EXPBUF: libc::c_ulong = 0xc0405610;

const V4L2_CAP_VIDEO_CAPTURE: u32 = 0x1;
const V4L2_CAP_STREAMING: u32 = 0x4000000;
const V4L2_BUF_TYPE_VIDEO_CAPTURE: u32 = 1;
const V4L2_MEMORY_MMAP: u32 = 1;
const V4L2_FIELD_NONE: u32 = 1;

pub const V4L2_PIX_FMT_NV12: u32 = 0x3231564e;
pub const V4L2_PIX_FMT_YUYV: u32 = 0x56595559;
pub const V4L2_PIX_FMT_UYVY: u32 = 0x59565955;
pub const V4L2_PIX_FMT_XRGB32: u32 = 0x34324258; // 'XB24'
pub const V4L2_PIX_FMT_P010: u32 = 0x30313050;   // 'P010'
pub const V4L2_PIX_FMT_MJPEG: u32 = 0x47504a4d;  // 'MJPG'
pub const PIXFMT_RGB24: u32 = 0x33424752;         // internal: decoded MJPEG

// ── v4l2 kernel structs (#[repr(C)] matching kernel layout) ──────────

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
const _: () = assert!(std::mem::size_of::<V4l2Capability>() == 104);

#[repr(C)]
#[derive(Clone, Copy)]
struct V4l2PixFormat {
    width: u32,
    height: u32,
    pixelformat: u32,
    field: u32,
    bytesperline: u32,
    sizeimage: u32,
    colorspace: u32,
    priv_: u32,
    flags: u32,
    // encoding, quantization, xfer_func
    _pad: [u32; 3],
}

#[repr(C)]
struct V4l2Format {
    type_: u32,
    _pad: u32,
    pix: V4l2PixFormat,
    _remainder: [u8; 152], // rest of the 200-byte fmt union
}
const _: () = assert!(std::mem::size_of::<V4l2Format>() == 208);

#[repr(C)]
struct V4l2RequestBuffers {
    count: u32,
    type_: u32,
    memory: u32,
    capabilities: u32,
    flags: u32,
}
const _: () = assert!(std::mem::size_of::<V4l2RequestBuffers>() == 20);

#[repr(C)]
#[derive(Clone, Copy)]
struct V4l2Timecode {
    type_: u32,
    flags: u32,
    frames: u8,
    seconds: u8,
    minutes: u8,
    hours: u8,
    userbits: [u8; 4],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct V4l2Buffer {
    pub index: u32,
    type_: u32,
    pub bytesused: u32,
    flags: u32,
    field: u32,
    _pad0: u32, // alignment for timeval
    timestamp: libc::timeval,
    timecode: V4l2Timecode,
    sequence: u32,
    memory: u32,
    m_offset: u64, // union m (8 bytes on 64-bit)
    pub length: u32,
    reserved2: u32,
    request_fd: i32,
    reserved: u32,
}
const _: () = assert!(std::mem::size_of::<V4l2Buffer>() == 88);

impl V4l2Buffer {
    /// Driver-assigned monotonic frame sequence number.
    pub fn sequence(&self) -> u32 {
        self.sequence
    }

    /// Driver-assigned capture timestamp (seconds + microseconds).
    #[allow(dead_code)]
    pub fn timestamp_us(&self) -> u64 {
        self.timestamp.tv_sec as u64 * 1_000_000 + self.timestamp.tv_usec as u64
    }
}

#[repr(C)]
struct V4l2CaptureParm {
    capability: u32,
    capturemode: u32,
    timeperframe_numerator: u32,
    timeperframe_denominator: u32,
    extendedmode: u32,
    readbuffers: u32,
    reserved: [u32; 4],
}

#[repr(C)]
struct V4l2StreamParm {
    type_: u32,
    parm: V4l2CaptureParm,
    _remainder: [u8; 160], // rest of the union
}
const _: () = assert!(std::mem::size_of::<V4l2StreamParm>() == 204);

#[repr(C)]
struct V4l2ExportBuffer {
    type_: u32,
    index: u32,
    plane: u32,
    flags: u32,
    fd: i32,
    reserved: [u32; 11],
}
const _: () = assert!(std::mem::size_of::<V4l2ExportBuffer>() == 64);

// ── helpers ──────────────────────────────────────────────────────────

unsafe fn xioctl(fd: RawFd, request: libc::c_ulong, arg: *mut libc::c_void) -> i32 {
    loop {
        let r = libc::ioctl(fd, request, arg);
        if r != -1 || *libc::__errno_location() != libc::EINTR {
            return r;
        }
    }
}

// ── public types ─────────────────────────────────────────────────────

pub struct MappedBuffer {
    pub ptr: *mut u8,
    pub len: usize,
}

pub struct Capture {
    fd: RawFd,
    buffers: Vec<MappedBuffer>,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub pixfmt: u32,
}

impl Capture {
    pub fn open(
        device: &str,
        width: u32,
        height: u32,
        fps: u32,
        pixfmt: u32,
        buf_count: u32,
    ) -> Result<Self> {
        let c_path = std::ffi::CString::new(device)?;

        let fd = unsafe {
            libc::open(c_path.as_ptr(), libc::O_RDWR | libc::O_NONBLOCK)
        };
        if fd < 0 {
            bail!("cannot open {}: {}", device, std::io::Error::last_os_error());
        }

        let mut cap = Self {
            fd,
            buffers: Vec::new(),
            width,
            height,
            fps,
            pixfmt,
        };

        cap.query_caps(device)?;
        cap.set_format()?;
        cap.init_buffers(buf_count)?;

        Ok(cap)
    }

    fn query_caps(&self, device: &str) -> Result<()> {
        let mut caps: V4l2Capability = unsafe { std::mem::zeroed() };
        if unsafe { xioctl(self.fd, VIDIOC_QUERYCAP, &mut caps as *mut _ as *mut _) } < 0 {
            bail!("{}: not a v4l2 device", device);
        }
        if caps.device_caps & V4L2_CAP_VIDEO_CAPTURE == 0 {
            bail!("{}: not a capture device", device);
        }
        if caps.device_caps & V4L2_CAP_STREAMING == 0 {
            bail!("{}: does not support streaming", device);
        }

        let card = CStr::from_bytes_until_nul(&caps.card)
            .map(|s| s.to_string_lossy())
            .unwrap_or_else(|_| "unknown".into());
        let driver = CStr::from_bytes_until_nul(&caps.driver)
            .map(|s| s.to_string_lossy())
            .unwrap_or_else(|_| "unknown".into());
        println!("device: {} ({})", card, driver);
        Ok(())
    }

    fn set_format(&mut self) -> Result<()> {
        let mut fmt: V4l2Format = unsafe { std::mem::zeroed() };
        fmt.type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        fmt.pix.width = self.width;
        fmt.pix.height = self.height;
        fmt.pix.pixelformat = self.pixfmt;
        fmt.pix.field = V4L2_FIELD_NONE;

        if unsafe { xioctl(self.fd, VIDIOC_S_FMT, &mut fmt as *mut _ as *mut _) } < 0 {
            bail!("VIDIOC_S_FMT: {}", std::io::Error::last_os_error());
        }

        if fmt.pix.width != self.width || fmt.pix.height != self.height {
            eprintln!("warning: driver adjusted resolution to {}x{}",
                      fmt.pix.width, fmt.pix.height);
            self.width = fmt.pix.width;
            self.height = fmt.pix.height;
        }
        if fmt.pix.pixelformat != self.pixfmt {
            eprintln!("warning: driver changed pixel format");
            self.pixfmt = fmt.pix.pixelformat;
        }

        // set framerate
        let mut parm: V4l2StreamParm = unsafe { std::mem::zeroed() };
        parm.type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        parm.parm.timeperframe_numerator = 1;
        parm.parm.timeperframe_denominator = self.fps;

        if unsafe { xioctl(self.fd, VIDIOC_S_PARM, &mut parm as *mut _ as *mut _) } < 0 {
            eprintln!("warning: could not set framerate: {}",
                      std::io::Error::last_os_error());
        } else {
            let actual = parm.parm.timeperframe_denominator
                / parm.parm.timeperframe_numerator.max(1);
            if actual != self.fps {
                eprintln!("warning: driver set framerate to {}", actual);
                self.fps = actual;
            }
        }

        let fourcc = self.pixfmt.to_le_bytes();
        println!("format: {} {}x{} @ {}fps",
                 std::str::from_utf8(&fourcc).unwrap_or("????"),
                 self.width, self.height, self.fps);
        Ok(())
    }

    fn init_buffers(&mut self, count: u32) -> Result<()> {
        let mut req: V4l2RequestBuffers = unsafe { std::mem::zeroed() };
        req.count = count;
        req.type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = V4L2_MEMORY_MMAP;

        if unsafe { xioctl(self.fd, VIDIOC_REQBUFS, &mut req as *mut _ as *mut _) } < 0 {
            bail!("VIDIOC_REQBUFS: {}", std::io::Error::last_os_error());
        }

        for i in 0..req.count {
            let mut buf: V4l2Buffer = unsafe { std::mem::zeroed() };
            buf.type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            buf.index = i;

            if unsafe { xioctl(self.fd, VIDIOC_QUERYBUF, &mut buf as *mut _ as *mut _) } < 0 {
                bail!("VIDIOC_QUERYBUF: {}", std::io::Error::last_os_error());
            }

            let ptr = unsafe {
                libc::mmap(
                    ptr::null_mut(),
                    buf.length as usize,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED,
                    self.fd,
                    buf.m_offset as i64,
                )
            };
            if ptr == libc::MAP_FAILED {
                bail!("mmap: {}", std::io::Error::last_os_error());
            }

            let ptr = ptr as *mut u8;
            let len = buf.length as usize;
            crate::priority::advise_hugepages(ptr, len);
            self.buffers.push(MappedBuffer { ptr, len });
        }

        println!("buffers: {} (target was 2 for minimum latency)", req.count);
        Ok(())
    }

    pub fn start(&self) -> Result<()> {
        for i in 0..self.buffers.len() {
            let mut buf: V4l2Buffer = unsafe { std::mem::zeroed() };
            buf.type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            buf.index = i as u32;
            if unsafe { xioctl(self.fd, VIDIOC_QBUF, &mut buf as *mut _ as *mut _) } < 0 {
                bail!("VIDIOC_QBUF: {}", std::io::Error::last_os_error());
            }
        }

        let mut type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if unsafe { xioctl(self.fd, VIDIOC_STREAMON, &mut type_ as *mut _ as *mut _) } < 0 {
            bail!("VIDIOC_STREAMON: {}", std::io::Error::last_os_error());
        }
        Ok(())
    }

    /// Dequeue a frame. Returns None if EAGAIN (no frame ready), Err on failure.
    pub fn dequeue(&self) -> Result<Option<V4l2Buffer>> {
        let mut buf: V4l2Buffer = unsafe { std::mem::zeroed() };
        buf.type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;

        if unsafe { xioctl(self.fd, VIDIOC_DQBUF, &mut buf as *mut _ as *mut _) } < 0 {
            let err = std::io::Error::last_os_error();
            if err.raw_os_error() == Some(libc::EAGAIN) {
                return Ok(None);
            }
            bail!("VIDIOC_DQBUF: {}", err);
        }
        Ok(Some(buf))
    }

    pub fn queue(&self, buf: &mut V4l2Buffer) -> Result<()> {
        if unsafe { xioctl(self.fd, VIDIOC_QBUF, buf as *mut _ as *mut _) } < 0 {
            bail!("VIDIOC_QBUF: {}", std::io::Error::last_os_error());
        }
        Ok(())
    }

    /// Get the raw pointer for a buffer index.
    pub fn buffer_ptr(&self, index: u32) -> *const u8 {
        self.buffers[index as usize].ptr
    }

    pub fn fd(&self) -> RawFd {
        self.fd
    }

    /// Export each MMAP buffer as a DMA-BUF file descriptor via VIDIOC_EXPBUF.
    /// Returns one FD per buffer.  Caller must close them when done (or use
    /// the returned Vec with `close_dmabuf_fds()`).
    pub fn export_dmabuf_fds(&self) -> Result<Vec<RawFd>> {
        let mut fds = Vec::with_capacity(self.buffers.len());
        for i in 0..self.buffers.len() {
            let mut expbuf: V4l2ExportBuffer = unsafe { std::mem::zeroed() };
            expbuf.type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            expbuf.index = i as u32;
            expbuf.flags = libc::O_RDONLY as u32;

            if unsafe { xioctl(self.fd, VIDIOC_EXPBUF, &mut expbuf as *mut _ as *mut _) } < 0 {
                // Close any FDs we already exported
                for fd in &fds {
                    unsafe { libc::close(*fd); }
                }
                bail!("VIDIOC_EXPBUF buffer {}: {}", i, std::io::Error::last_os_error());
            }
            fds.push(expbuf.fd);
        }
        Ok(fds)
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        let mut type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        unsafe {
            xioctl(self.fd, VIDIOC_STREAMOFF, &mut type_ as *mut _ as *mut _);
        }
        for buf in &self.buffers {
            unsafe { libc::munmap(buf.ptr as *mut _, buf.len); }
        }
        unsafe { libc::close(self.fd); }
    }
}

#[allow(dead_code)]
pub fn format_name(pixfmt: u32) -> String {
    let b = pixfmt.to_le_bytes();
    String::from_utf8_lossy(&b).to_string()
}
