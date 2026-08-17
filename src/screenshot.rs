use anyhow::{bail, Result};
use flate2::write::ZlibEncoder;
use flate2::Compression;
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::capture::{V4L2_PIX_FMT_NV12, V4L2_PIX_FMT_YUYV, V4L2_PIX_FMT_UYVY, V4L2_PIX_FMT_XRGB32, V4L2_PIX_FMT_P010, PIXFMT_RGB24};
use crate::config::ScreenshotFormat;

/// Default output directory for screenshots.
pub fn pictures_dir() -> PathBuf {
    if let Ok(home) = std::env::var("HOME") {
        PathBuf::from(home).join("Pictures")
    } else {
        PathBuf::from(".")
    }
}

/// Convert capture pixel format to packed RGB.
fn to_rgb(data: &[u8], width: u32, height: u32, pixfmt: u32) -> Result<Vec<u8>> {
    match pixfmt {
        V4L2_PIX_FMT_NV12 => Ok(nv12_to_rgb(data, width, height)),
        V4L2_PIX_FMT_YUYV => Ok(yuyv_to_rgb(data, width, height)),
        V4L2_PIX_FMT_UYVY => Ok(uyvy_to_rgb(data, width, height)),
        V4L2_PIX_FMT_XRGB32 => Ok(xrgb_to_rgb(data, width, height)),
        V4L2_PIX_FMT_P010 => Ok(p010_to_rgb(data, width, height)),
        PIXFMT_RGB24 => Ok(data[..(width * height * 3) as usize].to_vec()),
        _ => bail!("unsupported pixel format for screenshot"),
    }
}

/// Encode a frame to bytes in the specified format.
/// Returns `(bytes, file_extension)`.
pub fn encode_bytes(
    data: &[u8], width: u32, height: u32, pixfmt: u32,
    format: ScreenshotFormat, quality: u32,
) -> Result<(Vec<u8>, &'static str)> {
    let rgb = to_rgb(data, width, height, pixfmt)?;
    match format {
        ScreenshotFormat::Png  => Ok((write_png_bytes(&rgb, width, height)?, "png")),
        ScreenshotFormat::Jpeg => Ok((crate::jpeg::encode(&rgb, width, height, quality), "jpg")),
    }
}

/// Save a frame to a file in the specified format.
pub fn save_screenshot(
    data: &[u8], width: u32, height: u32, pixfmt: u32,
    dir: &Path, format: ScreenshotFormat, quality: u32,
) -> Result<String> {
    let (bytes, ext) = encode_bytes(data, width, height, pixfmt, format, quality)?;
    std::fs::create_dir_all(dir)?;
    let filename = format!(
        "capview_{}.{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        ext,
    );
    let path = dir.join(&filename);
    std::fs::write(&path, &bytes)?;
    Ok(path.to_string_lossy().to_string())
}

/// Append a screenshot to a tar archive in the specified format.
pub fn append_to_tar(
    data: &[u8], width: u32, height: u32, pixfmt: u32,
    tar_path: &Path, format: ScreenshotFormat, quality: u32,
) -> Result<String> {
    let (img_bytes, ext) = encode_bytes(data, width, height, pixfmt, format, quality)?;
    let entry_name = format!(
        "capview_{}.{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        ext,
    );
    append_bytes_to_tar(&img_bytes, &entry_name, tar_path)?;
    Ok(entry_name)
}

fn clamp_u8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

fn yuv_to_rgb(y: u8, u: u8, v: u8) -> (u8, u8, u8) {
    let y = y as i32;
    let u = u as i32 - 128;
    let v = v as i32 - 128;
    let r = clamp_u8(y + ((359 * v) >> 8));
    let g = clamp_u8(y - ((88 * u + 183 * v) >> 8));
    let b = clamp_u8(y + ((454 * u) >> 8));
    (r, g, b)
}

fn nv12_to_rgb(data: &[u8], w: u32, h: u32) -> Vec<u8> {
    let (w, h) = (w as usize, h as usize);
    let mut rgb = vec![0u8; w * h * 3];
    let y_plane = &data[..w * h];
    let uv_plane = &data[w * h..];

    for row in 0..h {
        for col in 0..w {
            let y = y_plane[row * w + col];
            let uv_idx = (row / 2) * w + (col & !1);
            let u = uv_plane[uv_idx];
            let v = uv_plane[uv_idx + 1];
            let (r, g, b) = yuv_to_rgb(y, u, v);
            let dst = (row * w + col) * 3;
            rgb[dst] = r;
            rgb[dst + 1] = g;
            rgb[dst + 2] = b;
        }
    }
    rgb
}

fn yuyv_to_rgb(data: &[u8], w: u32, h: u32) -> Vec<u8> {
    let (w, h) = (w as usize, h as usize);
    let mut rgb = vec![0u8; w * h * 3];
    for row in 0..h {
        for col in (0..w).step_by(2) {
            let base = row * w * 2 + col * 2;
            let y0 = data[base];
            let u = data[base + 1];
            let y1 = data[base + 2];
            let v = data[base + 3];

            let (r, g, b) = yuv_to_rgb(y0, u, v);
            let dst = (row * w + col) * 3;
            rgb[dst] = r;
            rgb[dst + 1] = g;
            rgb[dst + 2] = b;

            let (r, g, b) = yuv_to_rgb(y1, u, v);
            let dst = (row * w + col + 1) * 3;
            rgb[dst] = r;
            rgb[dst + 1] = g;
            rgb[dst + 2] = b;
        }
    }
    rgb
}

fn uyvy_to_rgb(data: &[u8], w: u32, h: u32) -> Vec<u8> {
    let (w, h) = (w as usize, h as usize);
    let mut rgb = vec![0u8; w * h * 3];
    for row in 0..h {
        for col in (0..w).step_by(2) {
            let base = row * w * 2 + col * 2;
            let u = data[base];
            let y0 = data[base + 1];
            let v = data[base + 2];
            let y1 = data[base + 3];

            let (r, g, b) = yuv_to_rgb(y0, u, v);
            let dst = (row * w + col) * 3;
            rgb[dst] = r;
            rgb[dst + 1] = g;
            rgb[dst + 2] = b;

            let (r, g, b) = yuv_to_rgb(y1, u, v);
            let dst = (row * w + col + 1) * 3;
            rgb[dst] = r;
            rgb[dst + 1] = g;
            rgb[dst + 2] = b;
        }
    }
    rgb
}

fn xrgb_to_rgb(data: &[u8], w: u32, h: u32) -> Vec<u8> {
    let npix = (w * h) as usize;
    let mut rgb = vec![0u8; npix * 3];
    for i in 0..npix {
        let si = i * 4;
        // BGRX → RGB
        rgb[i * 3] = data[si + 2];
        rgb[i * 3 + 1] = data[si + 1];
        rgb[i * 3 + 2] = data[si];
    }
    rgb
}

fn p010_to_rgb(data: &[u8], w: u32, h: u32) -> Vec<u8> {
    let w = w as usize;
    let h = h as usize;
    let mut rgb = vec![0u8; w * h * 3];
    let y_plane = &data[..w * h * 2];
    let uv_plane = &data[w * h * 2..];
    for row in 0..h {
        for col in 0..w {
            let yi = (row * w + col) * 2;
            let y8 = y_plane.get(yi + 1).copied().unwrap_or(0);
            let uvi = (row / 2) * w * 2 + (col & !1) * 2;
            let u8v = uv_plane.get(uvi + 1).copied().unwrap_or(128);
            let v8v = uv_plane.get(uvi + 3).copied().unwrap_or(128);
            let (r, g, b) = yuv_to_rgb(y8, u8v, v8v);
            let di = (row * w + col) * 3;
            rgb[di] = r; rgb[di + 1] = g; rgb[di + 2] = b;
        }
    }
    rgb
}

/// Encode frame data as PNG bytes in memory (for clipboard).
pub fn encode_png_bytes(data: &[u8], width: u32, height: u32, pixfmt: u32) -> Result<Vec<u8>> {
    let rgb = to_rgb(data, width, height, pixfmt)?;
    write_png_bytes(&rgb, width, height)
}

fn write_png_bytes(rgb: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
    let mut out = Vec::new();

    // PNG signature
    out.extend_from_slice(&[137, 80, 78, 71, 13, 10, 26, 10]);

    // IHDR
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.push(8); ihdr.push(2); ihdr.push(0); ihdr.push(0); ihdr.push(0);
    write_chunk_to_vec(&mut out, b"IHDR", &ihdr);

    // IDAT
    let row_len = width as usize * 3;
    let mut raw = Vec::with_capacity((row_len + 1) * height as usize);
    for row in 0..height as usize {
        raw.push(0);
        raw.extend_from_slice(&rgb[row * row_len..(row + 1) * row_len]);
    }

    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&raw)?;
    let compressed = encoder.finish()?;
    write_chunk_to_vec(&mut out, b"IDAT", &compressed);

    write_chunk_to_vec(&mut out, b"IEND", &[]);
    Ok(out)
}

fn write_chunk_to_vec(out: &mut Vec<u8>, tag: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(tag);
    out.extend_from_slice(data);
    let mut crc_data = Vec::with_capacity(4 + data.len());
    crc_data.extend_from_slice(tag);
    crc_data.extend_from_slice(data);
    let crc = png_crc32(&crc_data);
    out.extend_from_slice(&crc.to_be_bytes());
}

/// Write a minimal PNG file (8-bit RGB, no interlacing).
#[allow(dead_code)]
fn write_png(path: &Path, rgb: &[u8], width: u32, height: u32) -> Result<()> {
    let mut f = std::fs::File::create(path)?;

    // PNG signature
    f.write_all(&[137, 80, 78, 71, 13, 10, 26, 10])?;

    // IHDR
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.push(8); // bit depth
    ihdr.push(2); // color type: RGB
    ihdr.push(0); // compression
    ihdr.push(0); // filter
    ihdr.push(0); // interlace
    write_chunk(&mut f, b"IHDR", &ihdr)?;

    // IDAT: filter byte (0=None) + row data, zlib compressed
    let row_len = width as usize * 3;
    let mut raw = Vec::with_capacity((row_len + 1) * height as usize);
    for row in 0..height as usize {
        raw.push(0); // filter: None
        raw.extend_from_slice(&rgb[row * row_len..(row + 1) * row_len]);
    }

    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&raw)?;
    let compressed = encoder.finish()?;
    write_chunk(&mut f, b"IDAT", &compressed)?;

    // IEND
    write_chunk(&mut f, b"IEND", &[])?;

    Ok(())
}

#[allow(dead_code)]
fn write_chunk(f: &mut std::fs::File, tag: &[u8; 4], data: &[u8]) -> Result<()> {
    f.write_all(&(data.len() as u32).to_be_bytes())?;
    f.write_all(tag)?;
    f.write_all(data)?;

    let mut crc_data = Vec::with_capacity(4 + data.len());
    crc_data.extend_from_slice(tag);
    crc_data.extend_from_slice(data);
    let crc = png_crc32(&crc_data);
    f.write_all(&crc.to_be_bytes())?;
    Ok(())
}

fn png_crc32(data: &[u8]) -> u32 {
    static TABLE: std::sync::OnceLock<[u32; 256]> = std::sync::OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let mut t = [0u32; 256];
        for n in 0..256u32 {
            let mut c = n;
            for _ in 0..8 {
                if c & 1 != 0 {
                    c = 0xedb88320 ^ (c >> 1);
                } else {
                    c >>= 1;
                }
            }
            t[n as usize] = c;
        }
        t
    });

    let mut crc = 0xffffffff_u32;
    for &byte in data {
        crc = table[((crc ^ byte as u32) & 0xff) as usize] ^ (crc >> 8);
    }
    crc ^ 0xffffffff
}

/// Append a PNG screenshot to a tar archive, creating it if it doesn't exist.
/// Each entry is named `capview_<timestamp>.png` inside the tar.
/// Returns the entry filename on success.
#[allow(dead_code)]
pub fn append_png_to_tar(
    data: &[u8],
    width: u32,
    height: u32,
    pixfmt: u32,
    tar_path: &Path,
) -> Result<String> {
    // Encode the PNG into memory first
    let png_bytes = encode_png_bytes(data, width, height, pixfmt)?;

    let entry_name = format!(
        "capview_{}.png",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    );

    append_bytes_to_tar(&png_bytes, &entry_name, tar_path)?;
    Ok(entry_name)
}

/// Public wrapper for low-level tar append (used by analysis_strip).
pub fn append_bytes_to_tar_pub(content: &[u8], entry_name: &str, tar_path: &Path) -> Result<()> {
    append_bytes_to_tar(content, entry_name, tar_path)
}

/// Low-level tar append: writes arbitrary bytes as a new entry.
fn append_bytes_to_tar(content: &[u8], entry_name: &str, tar_path: &Path) -> Result<()> {

    // Ensure parent directory exists
    if let Some(dir) = tar_path.parent() {
        std::fs::create_dir_all(dir)?;
    }

    // Open file for append (or create). We need to position before the
    // end-of-archive marker (two 512-byte zero blocks) if the file already
    // has content. For a new file, just start writing.
    use std::io::{Seek, SeekFrom};
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open(tar_path)?;

    let file_len = f.metadata()?.len();
    if file_len >= 1024 {
        // Seek back past the two 512-byte zero end-of-archive blocks
        f.seek(SeekFrom::End(-1024))?;
    } else {
        f.seek(SeekFrom::Start(0))?;
    }

    // Write a POSIX tar header (ustar)
    let mut header = [0u8; 512];

    // name (0..100)
    let name_bytes = entry_name.as_bytes();
    let copy_len = name_bytes.len().min(99);
    header[..copy_len].copy_from_slice(&name_bytes[..copy_len]);

    // mode (100..108) — 0644
    header[100..107].copy_from_slice(b"0000644");

    // uid (108..116)
    header[108..115].copy_from_slice(b"0001000");

    // gid (116..124)
    header[116..123].copy_from_slice(b"0001000");

    // size (124..136) — octal
    let size_str = format!("{:011o}", content.len());
    header[124..135].copy_from_slice(size_str.as_bytes());

    // mtime (136..148)
    let mtime = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let mtime_str = format!("{:011o}", mtime);
    header[136..147].copy_from_slice(mtime_str.as_bytes());

    // typeflag (156) — '0' = regular file
    header[156] = b'0';

    // magic (257..263) — "ustar\0"
    header[257..263].copy_from_slice(b"ustar\0");

    // version (263..265) — "00"
    header[263..265].copy_from_slice(b"00");

    // checksum placeholder (148..156) — spaces
    header[148..156].copy_from_slice(b"        ");

    // Compute checksum (sum of all bytes in header, treating checksum field as spaces)
    let cksum: u32 = header.iter().map(|&b| b as u32).sum();
    let cksum_str = format!("{:06o}\0 ", cksum);
    header[148..156].copy_from_slice(cksum_str.as_bytes());

    // Write header
    f.write_all(&header)?;

    // Write file data
    f.write_all(content)?;

    // Pad to 512-byte boundary
    let remainder = content.len() % 512;
    if remainder != 0 {
        let padding = 512 - remainder;
        f.write_all(&vec![0u8; padding])?;
    }

    // Write two 512-byte zero blocks (end-of-archive marker)
    f.write_all(&[0u8; 1024])?;

    Ok(())
}
