//! Network protocol for capview UDP streaming.
//!
//! Frames are JPEG-encoded and split into MTU-sized fragments.
//! Each fragment carries a small header so the receiver can
//! reassemble complete frames and discard partial ones.
//!
//! ## Packet layout
//!
//! ```text
//! Offset  Size  Field
//! 0       4     magic (0x43565357 = "CVSW")
//! 4       4     sequence (frame number, wrapping u32)
//! 8       2     fragment_index (0-based)
//! 10      2     fragment_count (total fragments in this frame)
//! 12      2     width
//! 14      2     height
//! 16      1     fps
//! 17      1     quality
//! 18      2     reserved (zero)
//! 20      ..    payload (JPEG fragment data)
//! ```
//!
//! Maximum fragment payload = MTU - IP(20) - UDP(8) - header(20) = ~1444 bytes
//! at MTU 1500.  We use 1400 bytes as a safe payload size.

pub const MAGIC: u32 = 0x4356_5357; // "CVSW"
pub const HEADER_SIZE: usize = 20;
pub const MAX_PAYLOAD: usize = 1400;
#[allow(dead_code)]
pub const DEFAULT_PORT: u16 = 9000;
pub const MAX_FRAGMENT_COUNT: usize = 256; // sanity cap

/// Encode a packet header into the first HEADER_SIZE bytes of `buf`.
pub fn write_header(
    buf: &mut [u8],
    seq: u32,
    frag_idx: u16,
    frag_count: u16,
    width: u16,
    height: u16,
    fps: u8,
    quality: u8,
) {
    buf[0..4].copy_from_slice(&MAGIC.to_be_bytes());
    buf[4..8].copy_from_slice(&seq.to_be_bytes());
    buf[8..10].copy_from_slice(&frag_idx.to_be_bytes());
    buf[10..12].copy_from_slice(&frag_count.to_be_bytes());
    buf[12..14].copy_from_slice(&width.to_be_bytes());
    buf[14..16].copy_from_slice(&height.to_be_bytes());
    buf[16] = fps;
    buf[17] = quality;
    buf[18] = 0;
    buf[19] = 0;
}

/// Parsed packet header.
#[derive(Debug, Clone)]
pub struct PacketHeader {
    pub seq: u32,
    pub frag_idx: u16,
    pub frag_count: u16,
    pub width: u16,
    pub height: u16,
    pub fps: u8,
    #[allow(dead_code)]
    pub quality: u8,
}

/// Parse a packet header from received data.
/// Returns `None` if too short or magic mismatch.
pub fn parse_header(buf: &[u8]) -> Option<PacketHeader> {
    if buf.len() < HEADER_SIZE { return None; }
    let magic = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]);
    if magic != MAGIC { return None; }
    Some(PacketHeader {
        seq: u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]),
        frag_idx: u16::from_be_bytes([buf[8], buf[9]]),
        frag_count: u16::from_be_bytes([buf[10], buf[11]]),
        width: u16::from_be_bytes([buf[12], buf[13]]),
        height: u16::from_be_bytes([buf[14], buf[15]]),
        fps: buf[16],
        quality: buf[17],
    })
}
