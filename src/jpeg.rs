//! Minimal baseline JPEG encoder (4:4:4, standard Huffman tables).
//!
//! Produces valid JFIF files from RGB input with configurable quality (1–100).
//! No external dependencies — all tables are from ITU-T T.81 Annex K.

use std::f32::consts::PI;

// ── Zigzag scan order ───────────────────────────────────────────────

static ZIGZAG: [u8; 64] = [
     0, 1, 8,16, 9, 2, 3,10, 17,24,32,25,18,11, 4, 5,
    12,19,26,33,40,48,41,34, 27,20,13, 6, 7,14,21,28,
    35,42,49,56,57,50,43,36, 29,22,15,23,30,37,44,51,
    58,59,52,45,38,31,39,46, 53,60,61,54,47,55,62,63,
];

// ── Standard quantization tables (Tables K.1 & K.2) ────────────────

static STD_LUM_QUANT: [u8; 64] = [
    16, 11, 10, 16,  24,  40,  51,  61,
    12, 12, 14, 19,  26,  58,  60,  55,
    14, 13, 16, 24,  40,  57,  69,  56,
    14, 17, 22, 29,  51,  87,  80,  62,
    18, 22, 37, 56,  68, 109, 103,  77,
    24, 35, 55, 64,  81, 104, 113,  92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103,  99,
];

static STD_CHR_QUANT: [u8; 64] = [
    17, 18, 24, 47, 99, 99, 99, 99,
    18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
];

// ── Standard Huffman tables (Tables K.3–K.6) ────────────────────────

static DC_LUM_BITS: [u8; 16] = [0,1,5,1,1,1,1,1,1,0,0,0,0,0,0,0];
static DC_LUM_VALS: [u8; 12]  = [0,1,2,3,4,5,6,7,8,9,10,11];

static DC_CHR_BITS: [u8; 16] = [0,3,1,1,1,1,1,1,1,1,1,0,0,0,0,0];
static DC_CHR_VALS: [u8; 12]  = [0,1,2,3,4,5,6,7,8,9,10,11];

static AC_LUM_BITS: [u8; 16] = [0,2,1,3,3,2,4,3,5,5,4,4,0,0,1,0x7d];
static AC_LUM_VALS: [u8; 162] = [
    0x01,0x02,0x03,0x00,0x04,0x11,0x05,0x12,
    0x21,0x31,0x41,0x06,0x13,0x51,0x61,0x07,
    0x22,0x71,0x14,0x32,0x81,0x91,0xa1,0x08,
    0x23,0x42,0xb1,0xc1,0x15,0x52,0xd1,0xf0,
    0x24,0x33,0x62,0x72,0x82,0x09,0x0a,0x16,
    0x17,0x18,0x19,0x1a,0x25,0x26,0x27,0x28,
    0x29,0x2a,0x34,0x35,0x36,0x37,0x38,0x39,
    0x3a,0x43,0x44,0x45,0x46,0x47,0x48,0x49,
    0x4a,0x53,0x54,0x55,0x56,0x57,0x58,0x59,
    0x5a,0x63,0x64,0x65,0x66,0x67,0x68,0x69,
    0x6a,0x73,0x74,0x75,0x76,0x77,0x78,0x79,
    0x7a,0x83,0x84,0x85,0x86,0x87,0x88,0x89,
    0x8a,0x92,0x93,0x94,0x95,0x96,0x97,0x98,
    0x99,0x9a,0xa2,0xa3,0xa4,0xa5,0xa6,0xa7,
    0xa8,0xa9,0xaa,0xb2,0xb3,0xb4,0xb5,0xb6,
    0xb7,0xb8,0xb9,0xba,0xc2,0xc3,0xc4,0xc5,
    0xc6,0xc7,0xc8,0xc9,0xca,0xd2,0xd3,0xd4,
    0xd5,0xd6,0xd7,0xd8,0xd9,0xda,0xe1,0xe2,
    0xe3,0xe4,0xe5,0xe6,0xe7,0xe8,0xe9,0xea,
    0xf1,0xf2,0xf3,0xf4,0xf5,0xf6,0xf7,0xf8,
    0xf9,0xfa,
];

static AC_CHR_BITS: [u8; 16] = [0,2,1,2,4,4,3,4,7,5,4,4,0,1,2,0x77];
static AC_CHR_VALS: [u8; 162] = [
    0x00,0x01,0x02,0x03,0x11,0x04,0x05,0x21,
    0x31,0x06,0x12,0x41,0x51,0x07,0x61,0x71,
    0x13,0x22,0x32,0x81,0x08,0x14,0x42,0x91,
    0xa1,0xb1,0xc1,0x09,0x23,0x33,0x52,0xf0,
    0x15,0x62,0x72,0xd1,0x0a,0x16,0x24,0x34,
    0xe1,0x25,0xf1,0x17,0x18,0x19,0x1a,0x26,
    0x27,0x28,0x29,0x2a,0x35,0x36,0x37,0x38,
    0x39,0x3a,0x43,0x44,0x45,0x46,0x47,0x48,
    0x49,0x4a,0x53,0x54,0x55,0x56,0x57,0x58,
    0x59,0x5a,0x63,0x64,0x65,0x66,0x67,0x68,
    0x69,0x6a,0x73,0x74,0x75,0x76,0x77,0x78,
    0x79,0x7a,0x82,0x83,0x84,0x85,0x86,0x87,
    0x88,0x89,0x8a,0x92,0x93,0x94,0x95,0x96,
    0x97,0x98,0x99,0x9a,0xa2,0xa3,0xa4,0xa5,
    0xa6,0xa7,0xa8,0xa9,0xaa,0xb2,0xb3,0xb4,
    0xb5,0xb6,0xb7,0xb8,0xb9,0xba,0xc2,0xc3,
    0xc4,0xc5,0xc6,0xc7,0xc8,0xc9,0xca,0xd2,
    0xd3,0xd4,0xd5,0xd6,0xd7,0xd8,0xd9,0xda,
    0xe2,0xe3,0xe4,0xe5,0xe6,0xe7,0xe8,0xe9,
    0xea,0xf2,0xf3,0xf4,0xf5,0xf6,0xf7,0xf8,
    0xf9,0xfa,
];

// ── Huffman code lookup table ───────────────────────────────────────

struct HuffTable {
    code: [u16; 256],
    size: [u8; 256],
}

fn build_huff(bits: &[u8; 16], vals: &[u8]) -> HuffTable {
    let mut t = HuffTable { code: [0; 256], size: [0; 256] };
    let mut code = 0u16;
    let mut k = 0usize;
    for (len, &count) in bits.iter().enumerate() {
        for _ in 0..count {
            t.code[vals[k] as usize] = code;
            t.size[vals[k] as usize] = (len + 1) as u8;
            code += 1;
            k += 1;
        }
        code <<= 1;
    }
    t
}

// ── Bitstream writer with JPEG byte-stuffing ────────────────────────

struct BitWriter {
    buf: Vec<u8>,
    acc: u32,
    bits: u8,
}

impl BitWriter {
    fn new() -> Self {
        Self { buf: Vec::with_capacity(65536), acc: 0, bits: 0 }
    }

    fn put(&mut self, val: u16, n: u8) {
        self.acc = (self.acc << n) | val as u32;
        self.bits += n;
        while self.bits >= 8 {
            self.bits -= 8;
            let b = (self.acc >> self.bits) as u8;
            self.buf.push(b);
            if b == 0xFF { self.buf.push(0); } // byte stuffing
        }
    }

    fn flush(&mut self) {
        if self.bits > 0 {
            // Pad with 1-bits (JPEG convention)
            let pad = 8 - self.bits;
            let ones = (1u16 << pad) - 1;
            self.put(ones, pad);
        }
    }
}

// ── Quality scaling (IJG formula) ───────────────────────────────────

fn scale_table(base: &[u8; 64], quality: u32) -> [u16; 64] {
    let q = quality.clamp(1, 100);
    let scale = if q < 50 { 5000 / q } else { 200 - 2 * q };
    let mut out = [0u16; 64];
    for i in 0..64 {
        let v = ((base[i] as u32 * scale + 50) / 100).clamp(1, 255);
        out[i] = v as u16;
    }
    out
}

// ── Forward DCT (separable, naive 8×8) ──────────────────────────────

fn fdct(block: &mut [f32; 64]) {
    const FRAC_1_SQRT2: f32 = std::f32::consts::FRAC_1_SQRT_2;
    let mut tmp = [0f32; 64];

    // Row transform
    for r in 0..8usize {
        for u in 0..8usize {
            let c = if u == 0 { FRAC_1_SQRT2 } else { 1.0 };
            let mut s = 0f32;
            for x in 0..8usize {
                s += block[r * 8 + x]
                    * ((2 * x + 1) as f32 * u as f32 * PI / 16.0).cos();
            }
            tmp[r * 8 + u] = c * s * 0.5;
        }
    }

    // Column transform
    for u in 0..8usize {
        for v in 0..8usize {
            let c = if v == 0 { FRAC_1_SQRT2 } else { 1.0 };
            let mut s = 0f32;
            for y in 0..8usize {
                s += tmp[y * 8 + u]
                    * ((2 * y + 1) as f32 * v as f32 * PI / 16.0).cos();
            }
            block[v * 8 + u] = c * s * 0.5;
        }
    }
}

// ── Coefficient categorisation ──────────────────────────────────────

fn categorize(val: i16) -> (u8, u16) {
    if val == 0 {
        return (0, 0);
    }
    let a = val.unsigned_abs();
    let cat = 16 - a.leading_zeros() as u8;
    let bits = if val > 0 { val as u16 } else { (val - 1) as u16 };
    (cat, bits & ((1u16 << cat) - 1))
}

// ── Encode one 8×8 block (quantised, zigzag-ordered coefficients) ───

fn encode_block(
    bw: &mut BitWriter,
    coeffs: &[i16; 64],
    prev_dc: &mut i16,
    dc_ht: &HuffTable,
    ac_ht: &HuffTable,
) {
    // DC (differential)
    let diff = coeffs[0] - *prev_dc;
    *prev_dc = coeffs[0];
    let (cat, bits) = categorize(diff);
    bw.put(dc_ht.code[cat as usize], dc_ht.size[cat as usize]);
    if cat > 0 { bw.put(bits, cat); }

    // AC
    let mut zeros = 0u8;
    for i in 1..64 {
        if coeffs[i] == 0 {
            zeros += 1;
        } else {
            while zeros >= 16 {
                bw.put(ac_ht.code[0xF0], ac_ht.size[0xF0]); // ZRL
                zeros -= 16;
            }
            let (cat, bits) = categorize(coeffs[i]);
            let sym = (zeros << 4) | cat;
            bw.put(ac_ht.code[sym as usize], ac_ht.size[sym as usize]);
            if cat > 0 { bw.put(bits, cat); }
            zeros = 0;
        }
    }
    if zeros > 0 {
        bw.put(ac_ht.code[0x00], ac_ht.size[0x00]); // EOB
    }
}

// ── JPEG marker helpers ─────────────────────────────────────────────

fn write_dqt(out: &mut Vec<u8>, id: u8, table: &[u16; 64]) {
    out.extend_from_slice(&[0xFF, 0xDB]);
    out.extend_from_slice(&67u16.to_be_bytes()); // length
    out.push(id);
    for i in 0..64 {
        out.push(table[ZIGZAG[i] as usize] as u8);
    }
}

fn write_dht(out: &mut Vec<u8>, class_id: u8, bits: &[u8; 16], vals: &[u8]) {
    out.extend_from_slice(&[0xFF, 0xC4]);
    let len = 2 + 1 + 16 + vals.len() as u16;
    out.extend_from_slice(&len.to_be_bytes());
    out.push(class_id);
    out.extend_from_slice(bits);
    out.extend_from_slice(vals);
}

// ── Public API ──────────────────────────────────────────────────────

/// Encode packed RGB (3 bytes/pixel, row-major) as baseline JPEG.
///
/// `quality` is 1–100 (IJG scale: 1 = worst, 100 = best).
/// Returns a complete JFIF byte stream.
pub fn encode(rgb: &[u8], width: u32, height: u32, quality: u32) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let quality = quality.clamp(1, 100);

    let lum_qt = scale_table(&STD_LUM_QUANT, quality);
    let chr_qt = scale_table(&STD_CHR_QUANT, quality);

    let dc_lum = build_huff(&DC_LUM_BITS, &DC_LUM_VALS);
    let dc_chr = build_huff(&DC_CHR_BITS, &DC_CHR_VALS);
    let ac_lum = build_huff(&AC_LUM_BITS, &AC_LUM_VALS);
    let ac_chr = build_huff(&AC_CHR_BITS, &AC_CHR_VALS);

    let mut out = Vec::with_capacity(w * h);

    // SOI
    out.extend_from_slice(&[0xFF, 0xD8]);

    // APP0 (JFIF)
    out.extend_from_slice(&[0xFF, 0xE0]);
    out.extend_from_slice(&16u16.to_be_bytes());
    out.extend_from_slice(b"JFIF\0");
    out.extend_from_slice(&[1, 1, 0]); // version 1.1, aspect-ratio units: none
    out.extend_from_slice(&1u16.to_be_bytes()); // X density
    out.extend_from_slice(&1u16.to_be_bytes()); // Y density
    out.push(0); out.push(0); // thumbnail 0×0

    // DQT (luminance table 0, chrominance table 1)
    write_dqt(&mut out, 0, &lum_qt);
    write_dqt(&mut out, 1, &chr_qt);

    // SOF0 (baseline DCT, 4:4:4)
    out.extend_from_slice(&[0xFF, 0xC0]);
    out.extend_from_slice(&17u16.to_be_bytes()); // length
    out.push(8); // precision
    out.extend_from_slice(&(height as u16).to_be_bytes());
    out.extend_from_slice(&(width as u16).to_be_bytes());
    out.push(3); // component count
    out.extend_from_slice(&[1, 0x11, 0]); // Y:  H=1 V=1 Tq=0
    out.extend_from_slice(&[2, 0x11, 1]); // Cb: H=1 V=1 Tq=1
    out.extend_from_slice(&[3, 0x11, 1]); // Cr: H=1 V=1 Tq=1

    // DHT (4 tables)
    write_dht(&mut out, 0x00, &DC_LUM_BITS, &DC_LUM_VALS);
    write_dht(&mut out, 0x10, &AC_LUM_BITS, &AC_LUM_VALS);
    write_dht(&mut out, 0x01, &DC_CHR_BITS, &DC_CHR_VALS);
    write_dht(&mut out, 0x11, &AC_CHR_BITS, &AC_CHR_VALS);

    // SOS
    out.extend_from_slice(&[0xFF, 0xDA]);
    out.extend_from_slice(&12u16.to_be_bytes());
    out.push(3);
    out.extend_from_slice(&[1, 0x00]); // Y:  Td=0 Ta=0
    out.extend_from_slice(&[2, 0x11]); // Cb: Td=1 Ta=1
    out.extend_from_slice(&[3, 0x11]); // Cr: Td=1 Ta=1
    out.extend_from_slice(&[0x00, 0x3F, 0x00]); // spectral selection + approx

    // ── Entropy-coded scan data ─────────────────────────────────────

    let mut bw = BitWriter::new();
    let mut dc_y:  i16 = 0;
    let mut dc_cb: i16 = 0;
    let mut dc_cr: i16 = 0;

    let blocks_w = (w + 7) / 8;
    let blocks_h = (h + 7) / 8;

    for by in 0..blocks_h {
        for bx in 0..blocks_w {
            // Extract 8×8 block, RGB → YCbCr, level-shift
            let mut y_blk  = [0f32; 64];
            let mut cb_blk = [0f32; 64];
            let mut cr_blk = [0f32; 64];

            for dy in 0..8usize {
                for dx in 0..8usize {
                    let px = (bx * 8 + dx).min(w - 1);
                    let py = (by * 8 + dy).min(h - 1);
                    let idx = (py * w + px) * 3;
                    let r = rgb[idx]     as f32;
                    let g = rgb[idx + 1] as f32;
                    let b = rgb[idx + 2] as f32;

                    let y  =  0.2990 * r + 0.5870 * g + 0.1140 * b;
                    let cb = -0.1687 * r - 0.3313 * g + 0.5000 * b + 128.0;
                    let cr =  0.5000 * r - 0.4187 * g - 0.0813 * b + 128.0;

                    let i = dy * 8 + dx;
                    y_blk[i]  = y  - 128.0;
                    cb_blk[i] = cb - 128.0;
                    cr_blk[i] = cr - 128.0;
                }
            }

            // Forward DCT
            fdct(&mut y_blk);
            fdct(&mut cb_blk);
            fdct(&mut cr_blk);

            // Quantise + zigzag reorder
            let mut y_coeff  = [0i16; 64];
            let mut cb_coeff = [0i16; 64];
            let mut cr_coeff = [0i16; 64];

            for zz in 0..64 {
                let nat = ZIGZAG[zz] as usize;
                y_coeff[zz]  = (y_blk[nat]  / lum_qt[nat] as f32).round() as i16;
                cb_coeff[zz] = (cb_blk[nat] / chr_qt[nat] as f32).round() as i16;
                cr_coeff[zz] = (cr_blk[nat] / chr_qt[nat] as f32).round() as i16;
            }

            // Entropy-code each component
            encode_block(&mut bw, &y_coeff,  &mut dc_y,  &dc_lum, &ac_lum);
            encode_block(&mut bw, &cb_coeff, &mut dc_cb, &dc_chr, &ac_chr);
            encode_block(&mut bw, &cr_coeff, &mut dc_cr, &dc_chr, &ac_chr);
        }
    }

    bw.flush();
    out.extend_from_slice(&bw.buf);

    // EOI
    out.extend_from_slice(&[0xFF, 0xD9]);

    out
}
