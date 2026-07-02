//! Pure-`std` GGUF *writer* — serialize a Ferrum `LlamaModel` back to a GGUF v3
//! file that runs in the wider ecosystem (llama.cpp / ollama / LM Studio).
//!
//! The reader in [`crate::gguf`] is the specification: every block encoder here
//! is the exact inverse of the matching `dequant_*` decoder, and is verified by
//! round-tripping through [`crate::gguf::Gguf`].

use crate::error::{InferError, Result};
use crate::gguf::{
    Gguf, MetaValue, DEFAULT_ALIGNMENT, GGML_F16, GGML_F32, GGML_Q4_0, GGML_Q4_1, GGML_Q4_K,
    GGML_Q5_K, GGML_Q6_K, GGML_Q8_0, GGML_Q8_1, GGUF_MAGIC, QK, QK_K, VT_ARRAY, VT_BOOL, VT_F32,
    VT_F64, VT_I16, VT_I32, VT_I64, VT_I8, VT_STRING, VT_U16, VT_U32, VT_U64, VT_U8,
};
use crate::layer::Linear;
use crate::llm::LlamaModel;

/// The on-disk GGUF tensor type the writer should emit for weight matrices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgufQuant {
    F32,
    F16,
    Q8_0,
    Q8_1,
    Q4_0,
    Q4_1,
    Q4K,
    Q5K,
    Q6K,
}

impl GgufQuant {
    /// Parse a CLI/API name (case-insensitive). Accepts the reader's spellings.
    /// API is deliberately `GgufQuant::from_str`, not implementing `FromStr` trait.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "f32" | "none" => Some(Self::F32),
            "f16" | "fp16" => Some(Self::F16),
            "q8_0" | "q8" => Some(Self::Q8_0),
            "q8_1" => Some(Self::Q8_1),
            "q4_0" | "q4" => Some(Self::Q4_0),
            "q4_1" => Some(Self::Q4_1),
            "q4_k" | "q4k" => Some(Self::Q4K),
            "q5_k" | "q5k" => Some(Self::Q5K),
            "q6_k" | "q6k" => Some(Self::Q6K),
            _ => None,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn ggml_type(self) -> u32 {
        match self {
            Self::F32 => GGML_F32,
            Self::F16 => GGML_F16,
            Self::Q8_0 => GGML_Q8_0,
            Self::Q8_1 => GGML_Q8_1,
            Self::Q4_0 => GGML_Q4_0,
            Self::Q4_1 => GGML_Q4_1,
            Self::Q4K => GGML_Q4_K,
            Self::Q5K => GGML_Q5_K,
            Self::Q6K => GGML_Q6_K,
        }
    }

    /// The GGUF `general.file_type` enum id closest to this target (metadata hint).
    #[allow(dead_code)]
    pub(crate) fn file_type(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q8_0 => 7,
            Self::Q8_1 => 9,
            Self::Q4K => 15, // MOSTLY_Q4_K_M
            Self::Q5K => 17, // MOSTLY_Q5_K_M
            Self::Q6K => 18, // MOSTLY_Q6_K
        }
    }

    #[allow(dead_code)]
    pub(crate) fn is_kquant(self) -> bool {
        matches!(self, Self::Q4K | Self::Q5K | Self::Q6K)
    }
}

/// Encode an `f32` as an IEEE-754 half (`u16`), round-to-nearest-ties-to-even.
/// Inf/NaN, subnormals, and overflow-to-Inf are all handled. Exact inverse of
/// [`crate::gguf::f16_to_f32`] on values representable as f16.
pub fn f32_to_f16(value: f32) -> u16 {
    let x = value.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp32 = ((x >> 23) & 0xFF) as i32;
    let mant32 = x & 0x007F_FFFF;

    if exp32 == 0xFF {
        // Inf (mant 0) or NaN (mant != 0 → a canonical quiet NaN).
        return if mant32 != 0 {
            sign | 0x7E00
        } else {
            sign | 0x7C00
        };
    }

    let mut exp16 = exp32 - 127 + 15;
    if exp16 >= 0x1F {
        return sign | 0x7C00; // Overflow → Inf.
    }
    if exp16 <= 0 {
        // Subnormal f16, or underflow to signed zero.
        if exp16 < -10 {
            return sign;
        }
        let mant = mant32 | 0x0080_0000; // restore implicit leading 1
        let shift = (14 - exp16) as u32; // 14..=24
        let mut half = mant >> shift;
        let rem = mant & ((1u32 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        if rem > halfway || (rem == halfway && (half & 1) == 1) {
            half += 1;
        }
        return sign | half as u16;
    }

    // Normalized f16.
    let mut mant16 = (mant32 >> 13) as u16;
    let rem = mant32 & 0x1FFF;
    if rem > 0x1000 || (rem == 0x1000 && (mant16 & 1) == 1) {
        mant16 += 1;
        if mant16 == 0x0400 {
            mant16 = 0;
            exp16 += 1;
            if exp16 >= 0x1F {
                return sign | 0x7C00;
            }
        }
    }
    sign | ((exp16 as u16) << 10) | mant16
}

/// Encode a whole tensor's f32 values into GGUF on-disk bytes for `ggml_type`.
/// Quantized types require the length to be a multiple of their block size.
#[allow(dead_code)] // consumed by later tasks
pub(crate) fn encode_tensor(data: &[f32], ggml_type: u32) -> Result<Vec<u8>> {
    let n = data.len();
    let need_mult = |m: usize| -> Result<()> {
        if !n.is_multiple_of(m) {
            return Err(InferError::Format(format!(
                "tensor length {n} is not a multiple of block size {m} for ggml type {ggml_type}"
            )));
        }
        Ok(())
    };
    Ok(match ggml_type {
        GGML_F32 => enc_f32(data),
        GGML_F16 => enc_f16(data),
        GGML_Q8_0 => {
            need_mult(QK)?;
            enc_q8_0(data)
        }
        GGML_Q8_1 => {
            need_mult(QK)?;
            enc_q8_1(data)
        }
        GGML_Q4_0 => {
            need_mult(QK)?;
            enc_q4_0(data)
        }
        GGML_Q4_1 => {
            need_mult(QK)?;
            enc_q4_1(data)
        }
        GGML_Q4_K => {
            need_mult(QK_K)?;
            enc_q4_k(data)
        }
        GGML_Q5_K => {
            need_mult(QK_K)?;
            enc_q5_k(data)
        }
        GGML_Q6_K => {
            need_mult(QK_K)?;
            enc_q6_k(data)
        }
        other => {
            return Err(InferError::Format(format!(
                "encode_tensor: unsupported ggml type {other}"
            )))
        }
    })
}

#[allow(dead_code)] // consumed by later tasks
fn enc_f32(data: &[f32]) -> Vec<u8> {
    let mut o = Vec::with_capacity(data.len() * 4);
    for &x in data {
        o.extend_from_slice(&x.to_le_bytes());
    }
    o
}

#[allow(dead_code)] // consumed by later tasks
fn enc_f16(data: &[f32]) -> Vec<u8> {
    let mut o = Vec::with_capacity(data.len() * 2);
    for &x in data {
        o.extend_from_slice(&f32_to_f16(x).to_le_bytes());
    }
    o
}

/// Q8_0: per 32-element block, `d = amax/127` (f16), then 32 × i8 with
/// `q = round(x/d)`. Decode is `d * q`.
#[allow(dead_code)] // consumed by later tasks
fn enc_q8_0(data: &[f32]) -> Vec<u8> {
    let mut o = Vec::with_capacity(data.len() / QK * (2 + QK));
    for blk in data.chunks_exact(QK) {
        let amax = blk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        o.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        for &x in blk {
            let q = (x * id).round().clamp(-127.0, 127.0) as i8;
            o.push(q as u8);
        }
    }
    o
}

/// Q8_1: like Q8_0 plus a per-block sum `s = d * Σq` (f16). The reader ignores
/// `s` on decode, but ggml stores it, so we compute it faithfully.
#[allow(dead_code)] // consumed by later tasks
fn enc_q8_1(data: &[f32]) -> Vec<u8> {
    let mut o = Vec::with_capacity(data.len() / QK * (2 + 2 + QK));
    for blk in data.chunks_exact(QK) {
        let amax = blk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let mut qs = [0i8; QK];
        let mut sum = 0i32;
        for (j, &x) in blk.iter().enumerate() {
            let q = (x * id).round().clamp(-127.0, 127.0) as i8;
            qs[j] = q;
            sum += q as i32;
        }
        o.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        o.extend_from_slice(&f32_to_f16(d * sum as f32).to_le_bytes());
        for &q in &qs {
            o.push(q as u8);
        }
    }
    o
}

/// Q4_0: per 32-element block, `d = max/-8` where `max` is the value of largest
/// magnitude; `q = round(x/d)+8` clamped to 0..15. Elements `j` and `j+16` share
/// byte `j` (low/high nibble). Decode is `d * (q - 8)`.
#[allow(dead_code)] // consumed by later tasks
fn enc_q4_0(data: &[f32]) -> Vec<u8> {
    let half = QK / 2;
    let mut o = Vec::with_capacity(data.len() / QK * (2 + half));
    for blk in data.chunks_exact(QK) {
        // Pick the extreme value (largest |x|) to anchor the scale, as ggml does.
        let mut max = 0.0f32;
        let mut amax = 0.0f32;
        for &x in blk {
            if x.abs() > amax {
                amax = x.abs();
                max = x;
            }
        }
        let d = max / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        o.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        for j in 0..half {
            let q0 = ((blk[j] * id + 8.5) as i32).clamp(0, 15) as u8;
            let q1 = ((blk[j + half] * id + 8.5) as i32).clamp(0, 15) as u8;
            o.push(q0 | (q1 << 4));
        }
    }
    o
}

/// Q4_1: per 32-element block, `d = (max-min)/15`, `q = round((x-min)/d)` in
/// 0..15, storing `d` and `min` (both f16). Decode is `d * q + min`.
#[allow(dead_code)] // consumed by later tasks
fn enc_q4_1(data: &[f32]) -> Vec<u8> {
    let half = QK / 2;
    let mut o = Vec::with_capacity(data.len() / QK * (2 + 2 + half));
    for blk in data.chunks_exact(QK) {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for &x in blk {
            lo = lo.min(x);
            hi = hi.max(x);
        }
        let d = (hi - lo) / 15.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        o.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        o.extend_from_slice(&f32_to_f16(lo).to_le_bytes());
        for j in 0..half {
            let q0 = (((blk[j] - lo) * id + 0.5) as i32).clamp(0, 15) as u8;
            let q1 = (((blk[j + half] - lo) * id + 0.5) as i32).clamp(0, 15) as u8;
            o.push(q0 | (q1 << 4));
        }
    }
    o
}

/// Q4_K super-block length in encoded bytes.
#[allow(dead_code)] // consumed by later tasks
const Q4_K_BLOCK_LEN: usize = 2 + 2 + 12 + QK_K / 2; // 144

/// Q5_K super-block length in encoded bytes.
#[allow(dead_code)] // consumed by later tasks
const Q5_K_BLOCK_LEN: usize = 2 + 2 + 12 + QK_K / 8 + QK_K / 2; // 176

/// Q6_K super-block length in encoded bytes.
#[allow(dead_code)] // consumed by later tasks
const Q6_K_BLOCK_LEN: usize = QK_K / 2 + QK_K / 4 + QK_K / 16 + 2; // 210

/// Pack 8 six-bit sub-block scales and mins into the 12-byte layout that
/// [`crate::gguf::get_scale_min_k4`] reads back. Exact inverse of that function.
#[allow(dead_code)] // consumed by later tasks
fn put_scale_min_k4(sc: &[u8; 8], m: &[u8; 8]) -> [u8; 12] {
    let mut q = [0u8; 12];
    for j in 0..4 {
        q[j] = sc[j] & 63;
        q[j + 4] = m[j] & 63;
    }
    for j in 4..8 {
        q[j + 4] = (sc[j] & 0x0F) | ((m[j] & 0x0F) << 4);
        q[j - 4] |= (sc[j] >> 4) << 6; // top 2 bits of the 6-bit scale
        q[j] |= (m[j] >> 4) << 6; // top 2 bits of the 6-bit min
    }
    q
}

/// Q4_K: per 256-element super-block, 8 sub-blocks of 32. Each sub-block gets an
/// affine (scale, min) chosen to cover its [lo, hi]; those reals are then
/// quantized to 6-bit via super-block `d`/`dmin`. Decode: `d*sc_s*q - dmin*m_s`.
#[allow(dead_code)] // consumed by later tasks
fn enc_q4_k(data: &[f32]) -> Vec<u8> {
    let mut o = Vec::with_capacity(data.len() / QK_K * Q4_K_BLOCK_LEN);
    for sblk in data.chunks_exact(QK_K) {
        // 1. Per sub-block real scale/min covering [lo, hi], with min >= 0.
        let mut scale_r = [0.0f32; 8];
        let mut min_r = [0.0f32; 8];
        for s in 0..8 {
            let seg = &sblk[s * 32..s * 32 + 32];
            let mut lo = f32::INFINITY;
            let mut hi = f32::NEG_INFINITY;
            for &v in seg {
                lo = lo.min(v);
                hi = hi.max(v);
            }
            let min = if lo < 0.0 { -lo } else { 0.0 }; // >= 0
            let base = -min; // <= 0
            let scale = (hi - base) / 15.0;
            scale_r[s] = if scale > 0.0 { scale } else { 0.0 };
            min_r[s] = min;
        }
        // 2. Super-block factors and 6-bit sub-block codes.
        let dmax = scale_r.iter().cloned().fold(0.0f32, f32::max);
        let mmax = min_r.iter().cloned().fold(0.0f32, f32::max);
        let d = dmax / 63.0;
        let dmin = mmax / 63.0;
        let (mut sc, mut m) = ([0u8; 8], [0u8; 8]);
        for s in 0..8 {
            sc[s] = if d > 0.0 {
                ((scale_r[s] / d).round() as i32).clamp(0, 63) as u8
            } else {
                0
            };
            m[s] = if dmin > 0.0 {
                ((min_r[s] / dmin).round() as i32).clamp(0, 63) as u8
            } else {
                0
            };
        }
        // 3. Quantize each element against the ACTUAL reconstructed scale/min.
        let mut qs = [0u8; QK_K / 2];
        for s in 0..8 {
            let a_scale = d * sc[s] as f32;
            let a_min = dmin * m[s] as f32;
            for l in 0..32 {
                let x = sblk[s * 32 + l];
                let q = if a_scale > 0.0 {
                    (((x + a_min) / a_scale).round() as i32).clamp(0, 15) as u8
                } else {
                    0
                };
                let byte = 32 * (s / 2) + l;
                if s % 2 == 0 {
                    qs[byte] |= q;
                } else {
                    qs[byte] |= q << 4;
                }
            }
        }
        // 4. Emit: d, dmin (f16), 12-byte packed scales, 128-byte qs.
        o.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        o.extend_from_slice(&f32_to_f16(dmin).to_le_bytes());
        o.extend_from_slice(&put_scale_min_k4(&sc, &m));
        o.extend_from_slice(&qs);
    }
    o
}

/// Q5_K: like Q4_K but 5-bit quants (`qv ∈ 0..31`). The low nibble is packed as
/// in Q4_K; the 5th bit rides in `qh` (bit `s` of `qh[l]`). Decode:
/// `d*sc_s*qv - dmin*m_s`.
#[allow(dead_code)] // consumed by later tasks
fn enc_q5_k(data: &[f32]) -> Vec<u8> {
    let mut o = Vec::with_capacity(data.len() / QK_K * Q5_K_BLOCK_LEN);
    for sblk in data.chunks_exact(QK_K) {
        let mut scale_r = [0.0f32; 8];
        let mut min_r = [0.0f32; 8];
        for s in 0..8 {
            let seg = &sblk[s * 32..s * 32 + 32];
            let mut lo = f32::INFINITY;
            let mut hi = f32::NEG_INFINITY;
            for &v in seg {
                lo = lo.min(v);
                hi = hi.max(v);
            }
            let min = if lo < 0.0 { -lo } else { 0.0 };
            let base = -min;
            let scale = (hi - base) / 31.0; // 5-bit → 31 levels
            scale_r[s] = if scale > 0.0 { scale } else { 0.0 };
            min_r[s] = min;
        }
        let d = scale_r.iter().cloned().fold(0.0f32, f32::max) / 63.0;
        let dmin = min_r.iter().cloned().fold(0.0f32, f32::max) / 63.0;
        let (mut sc, mut m) = ([0u8; 8], [0u8; 8]);
        for s in 0..8 {
            sc[s] = if d > 0.0 {
                ((scale_r[s] / d).round() as i32).clamp(0, 63) as u8
            } else {
                0
            };
            m[s] = if dmin > 0.0 {
                ((min_r[s] / dmin).round() as i32).clamp(0, 63) as u8
            } else {
                0
            };
        }
        let mut qs = [0u8; QK_K / 2];
        let mut qh = [0u8; QK_K / 8]; // 32 bytes, one bit per sub-block
        for s in 0..8 {
            let a_scale = d * sc[s] as f32;
            let a_min = dmin * m[s] as f32;
            for l in 0..32 {
                let x = sblk[s * 32 + l];
                let qv = if a_scale > 0.0 {
                    (((x + a_min) / a_scale).round() as i32).clamp(0, 31) as u8
                } else {
                    0
                };
                let low = qv & 0x0F;
                let high = (qv >> 4) & 1;
                let byte = 32 * (s / 2) + l;
                if s % 2 == 0 {
                    qs[byte] |= low;
                } else {
                    qs[byte] |= low << 4;
                }
                if high != 0 {
                    qh[l] |= 1 << s;
                }
            }
        }
        o.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        o.extend_from_slice(&f32_to_f16(dmin).to_le_bytes());
        o.extend_from_slice(&put_scale_min_k4(&sc, &m));
        o.extend_from_slice(&qh);
        o.extend_from_slice(&qs);
    }
    o
}

/// Q6_K: per 256-element super-block, 16 groups of 16 with signed 6-bit quants
/// (`q ∈ -32..31`) and an `i8` scale per group, all multiplied by a super-block
/// `d` (f16). Decode: `d * scale_g * q`.
#[allow(dead_code)] // consumed by later tasks
fn enc_q6_k(data: &[f32]) -> Vec<u8> {
    let mut o = Vec::with_capacity(data.len() / QK_K * Q6_K_BLOCK_LEN);
    for sblk in data.chunks_exact(QK_K) {
        // 1. Per 16-element group, a real scale mapping amax → q = 31.
        let mut gscale = [0.0f32; 16];
        for g in 0..16 {
            let seg = &sblk[g * 16..g * 16 + 16];
            let amax = seg.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            gscale[g] = amax / 31.0;
        }
        // 2. Super-block d and i8 group scales.
        let dmax = gscale.iter().cloned().fold(0.0f32, f32::max);
        let d = dmax / 127.0;
        let mut scales = [0i8; 16];
        for g in 0..16 {
            scales[g] = if d > 0.0 {
                ((gscale[g] / d).round() as i32).clamp(-127, 127) as i8
            } else {
                0
            };
        }
        // 3. Quantize each element to a signed 6-bit code, stored as u = q + 32.
        let mut u = [0u8; QK_K];
        for g in 0..16 {
            let a = d * scales[g] as f32;
            for l in 0..16 {
                let x = sblk[g * 16 + l];
                let q = if a != 0.0 {
                    ((x / a).round() as i32).clamp(-32, 31)
                } else {
                    0
                };
                u[g * 16 + l] = (q + 32) as u8; // 0..63
            }
        }
        // 4. Pack ql/qh per the reader's two-half layout.
        let mut ql = [0u8; QK_K / 2]; // 128
        let mut qh = [0u8; QK_K / 4]; // 64
        for half in 0..2 {
            let (oo, qlo, qho) = (half * 128, half * 64, half * 32);
            for l in 0..32 {
                let ua = u[oo + l];
                let ub = u[oo + l + 32];
                let uc = u[oo + l + 64];
                let ud = u[oo + l + 96];
                ql[qlo + l] = (ua & 0x0F) | ((uc & 0x0F) << 4);
                ql[qlo + l + 32] = (ub & 0x0F) | ((ud & 0x0F) << 4);
                qh[qho + l] = ((ua >> 4) & 3)
                    | (((ub >> 4) & 3) << 2)
                    | (((uc >> 4) & 3) << 4)
                    | (((ud >> 4) & 3) << 6);
            }
        }
        // 5. Emit: ql, qh, 16 i8 scales, d (f16) — matching the reader's order.
        o.extend_from_slice(&ql);
        o.extend_from_slice(&qh);
        for &s in &scales {
            o.push(s as u8);
        }
        o.extend_from_slice(&f32_to_f16(d).to_le_bytes());
    }
    o
}

/// Decide the on-disk type for one tensor. 1-D tensors (norms, biases) always
/// stay F32. 2-D matrices take `quant`, unless their block axis is not divisible
/// by the block size, in which case they fall back to F16. Returns
/// `(ggml_type, fell_back)`.
#[allow(dead_code)] // consumed by later tasks
pub(crate) fn plan_tensor_type(is_2d: bool, block_axis: usize, quant: GgufQuant) -> (u32, bool) {
    if !is_2d {
        return (GGML_F32, false);
    }
    match quant {
        GgufQuant::F32 => (GGML_F32, false),
        GgufQuant::F16 => (GGML_F16, false),
        q if q.is_kquant() => {
            if block_axis.is_multiple_of(QK_K) {
                (q.ggml_type(), false)
            } else {
                (GGML_F16, true)
            }
        }
        q => {
            if block_axis.is_multiple_of(QK) {
                (q.ggml_type(), false)
            } else {
                (GGML_F16, true)
            }
        }
    }
}

fn align_up(x: usize, a: usize) -> usize {
    if a == 0 {
        x
    } else {
        x.div_ceil(a) * a
    }
}

/// A tensor queued for emission: name, dims (GGML order), type id, encoded bytes.
struct TensorOut {
    name: String,
    dims: Vec<u64>,
    ggml_type: u32,
    data: Vec<u8>,
}

/// Accumulates metadata + tensors, then emits a byte-exact GGUF v3 file.
/// The exact inverse of [`crate::gguf`]'s `parse_header`.
pub struct GgufBuilder {
    metadata: Vec<(String, MetaValue)>,
    tensors: Vec<TensorOut>,
    alignment: usize,
}

impl Default for GgufBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl GgufBuilder {
    pub fn new() -> Self {
        Self {
            metadata: Vec::new(),
            tensors: Vec::new(),
            alignment: DEFAULT_ALIGNMENT as usize,
        }
    }

    pub fn meta(&mut self, key: &str, val: MetaValue) -> &mut Self {
        self.metadata.push((key.to_string(), val));
        self
    }

    pub fn tensor(&mut self, name: &str, dims: &[u64], ggml_type: u32, data: Vec<u8>) -> &mut Self {
        self.tensors.push(TensorOut {
            name: name.to_string(),
            dims: dims.to_vec(),
            ggml_type,
            data,
        });
        self
    }

    pub fn into_bytes(self) -> Vec<u8> {
        let mut o = Vec::new();
        // Header.
        put_u32(&mut o, GGUF_MAGIC);
        put_u32(&mut o, 3); // version
        put_u64(&mut o, self.tensors.len() as u64);
        put_u64(&mut o, self.metadata.len() as u64);

        // Metadata KV table, in insertion order.
        for (k, v) in &self.metadata {
            put_str(&mut o, k);
            put_value(&mut o, v);
        }

        // Tensor directory. Offsets are relative to the (aligned) data section
        // and each is aligned to `self.alignment`.
        let mut running = 0usize;
        let mut offsets = Vec::with_capacity(self.tensors.len());
        for t in &self.tensors {
            let off = align_up(running, self.alignment);
            offsets.push(off);
            running = off + t.data.len();
        }
        for (t, &off) in self.tensors.iter().zip(&offsets) {
            put_str(&mut o, &t.name);
            put_u32(&mut o, t.dims.len() as u32);
            for &d in &t.dims {
                put_u64(&mut o, d);
            }
            put_u32(&mut o, t.ggml_type);
            put_u64(&mut o, off as u64);
        }

        // Pad to the data-section start, then write each tensor at its offset.
        let data_start = align_up(o.len(), self.alignment);
        o.resize(data_start, 0);
        for (t, &off) in self.tensors.iter().zip(&offsets) {
            let abs = data_start + off;
            if o.len() < abs {
                o.resize(abs, 0); // inter-tensor alignment padding
            }
            o.extend_from_slice(&t.data);
        }
        o
    }

    #[allow(dead_code)]
    pub fn write(self, path: &str) -> crate::error::Result<()> {
        let bytes = self.into_bytes();
        std::fs::write(path, bytes).map_err(InferError::from)
    }
}

// ── Little-endian put helpers ────────────────────────────────────────────────

fn put_u32(o: &mut Vec<u8>, v: u32) {
    o.extend_from_slice(&v.to_le_bytes());
}

fn put_u64(o: &mut Vec<u8>, v: u64) {
    o.extend_from_slice(&v.to_le_bytes());
}

fn put_str(o: &mut Vec<u8>, s: &str) {
    put_u64(o, s.len() as u64);
    o.extend_from_slice(s.as_bytes());
}

/// Serialize one metadata value (type tag + payload). Mirrors `read_value`.
fn put_value(o: &mut Vec<u8>, v: &MetaValue) {
    match v {
        MetaValue::U8(x) => {
            put_u32(o, VT_U8);
            o.push(*x);
        }
        MetaValue::I8(x) => {
            put_u32(o, VT_I8);
            o.push(*x as u8);
        }
        MetaValue::U16(x) => {
            put_u32(o, VT_U16);
            o.extend_from_slice(&x.to_le_bytes());
        }
        MetaValue::I16(x) => {
            put_u32(o, VT_I16);
            o.extend_from_slice(&x.to_le_bytes());
        }
        MetaValue::U32(x) => {
            put_u32(o, VT_U32);
            o.extend_from_slice(&x.to_le_bytes());
        }
        MetaValue::I32(x) => {
            put_u32(o, VT_I32);
            o.extend_from_slice(&x.to_le_bytes());
        }
        MetaValue::F32(x) => {
            put_u32(o, VT_F32);
            o.extend_from_slice(&x.to_le_bytes());
        }
        MetaValue::Bool(x) => {
            put_u32(o, VT_BOOL);
            o.push(*x as u8);
        }
        MetaValue::String(s) => {
            put_u32(o, VT_STRING);
            put_str(o, s);
        }
        MetaValue::U64(x) => {
            put_u32(o, VT_U64);
            o.extend_from_slice(&x.to_le_bytes());
        }
        MetaValue::I64(x) => {
            put_u32(o, VT_I64);
            o.extend_from_slice(&x.to_le_bytes());
        }
        MetaValue::F64(x) => {
            put_u32(o, VT_F64);
            o.extend_from_slice(&x.to_le_bytes());
        }
        MetaValue::Array(items) => {
            put_u32(o, VT_ARRAY);
            // Element type tag from the first item (empty arrays default to STRING,
            // matching the common tokenizer-metadata case).
            let elem_tag = items.first().map(value_type_tag).unwrap_or(VT_STRING);
            put_u32(o, elem_tag);
            put_u64(o, items.len() as u64);
            for it in items {
                put_value_payload(o, it);
            }
        }
    }
}

/// The GGUF type tag for a value (without writing it).
fn value_type_tag(v: &MetaValue) -> u32 {
    match v {
        MetaValue::U8(_) => VT_U8,
        MetaValue::I8(_) => VT_I8,
        MetaValue::U16(_) => VT_U16,
        MetaValue::I16(_) => VT_I16,
        MetaValue::U32(_) => VT_U32,
        MetaValue::I32(_) => VT_I32,
        MetaValue::F32(_) => VT_F32,
        MetaValue::Bool(_) => VT_BOOL,
        MetaValue::String(_) => VT_STRING,
        MetaValue::U64(_) => VT_U64,
        MetaValue::I64(_) => VT_I64,
        MetaValue::F64(_) => VT_F64,
        MetaValue::Array(_) => VT_ARRAY,
    }
}

/// Write only a value's payload (no type tag) — used for array elements, whose
/// type is written once for the whole array.
fn put_value_payload(o: &mut Vec<u8>, v: &MetaValue) {
    match v {
        MetaValue::U8(x) => o.push(*x),
        MetaValue::I8(x) => o.push(*x as u8),
        MetaValue::U16(x) => o.extend_from_slice(&x.to_le_bytes()),
        MetaValue::I16(x) => o.extend_from_slice(&x.to_le_bytes()),
        MetaValue::U32(x) => o.extend_from_slice(&x.to_le_bytes()),
        MetaValue::I32(x) => o.extend_from_slice(&x.to_le_bytes()),
        MetaValue::F32(x) => o.extend_from_slice(&x.to_le_bytes()),
        MetaValue::Bool(x) => o.push(*x as u8),
        MetaValue::String(s) => put_str(o, s),
        MetaValue::U64(x) => o.extend_from_slice(&x.to_le_bytes()),
        MetaValue::I64(x) => o.extend_from_slice(&x.to_le_bytes()),
        MetaValue::F64(x) => o.extend_from_slice(&x.to_le_bytes()),
        MetaValue::Array(_) => {
            // The reader rejects nested arrays; the writer never produces them.
        }
    }
}

/// Serialize a `LlamaModel` to GGUF bytes, carrying the source GGUF's metadata
/// (hyperparameters + tokenizer) forward verbatim. See module docs.
pub fn llama_gguf_bytes(model: &LlamaModel, source: &Gguf, quant: GgufQuant) -> Result<Vec<u8>> {
    let arch = source
        .architecture()
        .ok_or_else(|| InferError::Format("source GGUF missing general.architecture".into()))?
        .to_string();
    if arch != "llama" && arch != "qwen2" {
        return Err(InferError::Format(format!(
            "GGUF export supports llama/qwen2 only (source architecture = '{arch}')"
        )));
    }

    let mut b = GgufBuilder::new();

    // 1. Metadata: copy every source key verbatim, except the file-type hints we
    //    refresh to match the new target.
    for (k, v) in &source.metadata {
        if k == "general.file_type" || k == "general.quantization_version" {
            continue;
        }
        b.meta(k, v.clone());
    }
    b.meta("general.file_type", MetaValue::U32(quant.file_type()));
    b.meta("general.quantization_version", MetaValue::U32(2));

    // 2. Tensors.
    let cfg = &model.cfg;
    // token_embd: Ferrum [vocab, dim] is already raw order; dims [dim, vocab].
    add_tensor(
        &mut b,
        "token_embd.weight",
        &model.tok_emb,
        cfg.model_dim, // block axis = fastest dim = model_dim
        &[cfg.model_dim as u64, cfg.vocab_size as u64],
        true,
        quant,
    )?;

    for (i, blk) in model.blocks.iter().enumerate() {
        let p = format!("blk.{i}");
        add_norm(
            &mut b,
            &format!("{p}.attn_norm.weight"),
            &blk.attn_norm.weight,
        );
        add_linear(&mut b, &p, "attn_q", &blk.attn.wq, source, true, quant)?;
        add_linear(&mut b, &p, "attn_k", &blk.attn.wk, source, true, quant)?;
        add_linear(&mut b, &p, "attn_v", &blk.attn.wv, source, true, quant)?;
        add_linear(
            &mut b,
            &p,
            "attn_output",
            &blk.attn.wo,
            source,
            false,
            quant,
        )?;
        add_norm(
            &mut b,
            &format!("{p}.ffn_norm.weight"),
            &blk.ffn_norm.weight,
        );
        add_linear(&mut b, &p, "ffn_gate", &blk.ffn.gate, source, false, quant)?;
        add_linear(&mut b, &p, "ffn_up", &blk.ffn.up, source, false, quant)?;
        add_linear(&mut b, &p, "ffn_down", &blk.ffn.down, source, false, quant)?;
    }

    add_norm(&mut b, "output_norm.weight", &model.final_norm.weight);
    // Emit an explicit LM head only if the source had one (else tying is kept).
    if source.tensor("output.weight").is_some() {
        add_weight_2d(&mut b, "output.weight", &model.lm_head, quant)?;
    }

    Ok(b.into_bytes())
}

/// Write `llama_gguf_bytes` to a file.
pub fn write_llama_gguf(
    model: &LlamaModel,
    source: &Gguf,
    quant: GgufQuant,
    path: &str,
) -> Result<()> {
    let bytes = llama_gguf_bytes(model, source, quant)?;
    std::fs::write(path, bytes).map_err(InferError::from)
}

// ── tensor emission helpers ──────────────────────────────────────────────────

/// Transpose row-major `[rows, cols]` → `[cols, rows]`.
fn transpose(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

/// A 1-D norm weight — always F32.
fn add_norm(b: &mut GgufBuilder, name: &str, weight: &[f32]) {
    // encode_tensor(F32) is infallible for any length; unwrap is safe here.
    let bytes = encode_tensor(weight, GGML_F32).unwrap();
    b.tensor(name, &[weight.len() as u64], GGML_F32, bytes);
}

/// A projection weight (+ optional bias) named `{prefix}.{stem}.weight`.
///
/// `can_have_bias` must be `true` only for `attn_q`/`attn_k`/`attn_v`: those are
/// the only stems [`crate::gguf::Gguf::load_llama_prec`] ever loads a bias for
/// (it passes `bias_name: None` when building the `attn_output`/`ffn_*`
/// `Linear`s). For those stems `lin.bias.data` is therefore always a zero-filled
/// placeholder, not source data — emitting it would silently synthesize a bogus
/// all-zeros bias tensor, so callers for those stems must pass `false`.
fn add_linear(
    b: &mut GgufBuilder,
    prefix: &str,
    stem: &str,
    lin: &Linear,
    source: &Gguf,
    can_have_bias: bool,
    quant: GgufQuant,
) -> Result<()> {
    let wname = format!("{prefix}.{stem}.weight");
    add_weight_2d(b, &wname, lin, quant)?;
    // Emit the bias only if this stem can carry one *and* the source had it.
    if can_have_bias {
        let bname = format!("{prefix}.{stem}.bias");
        if source.tensor(&bname).is_some() {
            let bias = &lin.bias.data;
            let bytes = encode_tensor(bias, GGML_F32).unwrap();
            b.tensor(&bname, &[bias.len() as u64], GGML_F32, bytes);
        }
    }
    Ok(())
}

/// Emit a 2-D weight: Ferrum `[n_in, n_out]` → raw `[n_out, n_in]`, dims
/// `[n_in, n_out]`, quantized per policy.
fn add_weight_2d(b: &mut GgufBuilder, name: &str, lin: &Linear, quant: GgufQuant) -> Result<()> {
    let (n_in, n_out) = (lin.in_features(), lin.out_features());
    let w = lin.weight_f32(); // [n_in, n_out]
    let raw = transpose(&w, n_in, n_out); // [n_out, n_in]
    add_tensor(
        b,
        name,
        &raw,
        n_in,
        &[n_in as u64, n_out as u64],
        true,
        quant,
    )
}

/// Encode `raw` (already in GGUF byte order) as a tensor of `dims`, choosing the
/// type via `plan_tensor_type` on `block_axis` (= dims[0]).
fn add_tensor(
    b: &mut GgufBuilder,
    name: &str,
    raw: &[f32],
    block_axis: usize,
    dims: &[u64],
    is_2d: bool,
    quant: GgufQuant,
) -> Result<()> {
    let (ggml_type, fell_back) = plan_tensor_type(is_2d, block_axis, quant);
    if fell_back {
        vprintln!(
            "note: tensor '{name}' (axis {block_axis}) is not block-aligned for the chosen \
             quant; storing it as F16"
        );
    }
    let bytes = encode_tensor(raw, ggml_type)?;
    b.tensor(name, dims, ggml_type, bytes);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::f16_to_f32;

    #[test]
    fn f32_to_f16_is_exact_inverse_of_f16_to_f32() {
        // For every representable f16 bit pattern, encoding its f32 value must
        // return the same bits. NaNs are canonicalized; ±0 compared by bits.
        for h in 0u32..=0xFFFF {
            let h = h as u16;
            let exp = (h >> 10) & 0x1F;
            let mant = h & 0x3FF;
            if exp == 0x1F && mant != 0 {
                continue; // skip NaN encodings (many bit patterns, one canonical out)
            }
            let f = f16_to_f32(h);
            assert_eq!(f32_to_f16(f), h, "roundtrip failed for f16 bits {h:#06x}");
        }
    }

    #[test]
    fn f32_to_f16_spot_values() {
        assert_eq!(f32_to_f16(0.0), 0x0000);
        assert_eq!(f32_to_f16(-0.0), 0x8000);
        assert_eq!(f32_to_f16(1.0), 0x3C00);
        assert_eq!(f32_to_f16(-2.0), 0xC000);
        assert_eq!(f32_to_f16(65504.0), 0x7BFF); // max normal f16
        assert_eq!(f32_to_f16(1e5), 0x7C00); // overflow → +Inf
        assert_eq!(f32_to_f16(f32::INFINITY), 0x7C00);
    }

    #[test]
    fn gguf_quant_parses_and_maps_types() {
        use crate::gguf::{GGML_F16, GGML_F32, GGML_Q4_0, GGML_Q4_K, GGML_Q6_K, GGML_Q8_0};
        assert_eq!(GgufQuant::from_str("f32").unwrap().ggml_type(), GGML_F32);
        assert_eq!(GgufQuant::from_str("f16").unwrap().ggml_type(), GGML_F16);
        assert_eq!(GgufQuant::from_str("q8_0").unwrap().ggml_type(), GGML_Q8_0);
        assert_eq!(GgufQuant::from_str("q4_0").unwrap().ggml_type(), GGML_Q4_0);
        assert_eq!(GgufQuant::from_str("q4_k").unwrap().ggml_type(), GGML_Q4_K);
        assert_eq!(GgufQuant::from_str("q6_k").unwrap().ggml_type(), GGML_Q6_K);
        assert!(GgufQuant::from_str("q4_k").unwrap().is_kquant());
        assert!(!GgufQuant::from_str("q8_0").unwrap().is_kquant());
        assert!(GgufQuant::from_str("bogus").is_none());
    }

    #[test]
    fn builder_roundtrips_through_reader() {
        use crate::gguf::{Gguf, MetaValue, GGML_F32};

        // Two f32 tensors + scalar/string/array metadata.
        let t0: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let t1: Vec<f32> = (0..4).map(|i| -(i as f32)).collect();

        let mut b = GgufBuilder::new();
        b.meta("general.architecture", MetaValue::String("llama".into()));
        b.meta("llama.block_count", MetaValue::U32(2));
        b.meta(
            "tokenizer.ggml.tokens",
            MetaValue::Array(vec![
                MetaValue::String("a".into()),
                MetaValue::String("b".into()),
            ]),
        );
        b.tensor("t0", &[8], GGML_F32, f32s_to_le_bytes(&t0));
        b.tensor("t1", &[4], GGML_F32, f32s_to_le_bytes(&t1));
        let bytes = b.into_bytes();

        let g = Gguf::parse(bytes).unwrap();
        assert_eq!(g.version, 3);
        assert_eq!(g.architecture(), Some("llama"));
        assert_eq!(g.meta("llama.block_count").unwrap().as_usize(), Some(2));
        // Tensors present with correct dims and decoded values.
        assert_eq!(g.dequantize(g.tensor("t0").unwrap()).unwrap(), t0);
        assert_eq!(g.dequantize(g.tensor("t1").unwrap()).unwrap(), t1);
        // Array metadata preserved.
        match g.meta("tokenizer.ggml.tokens").unwrap() {
            MetaValue::Array(a) => {
                assert_eq!(a.len(), 2);
                assert_eq!(a[0].as_str(), Some("a"));
            }
            _ => panic!("tokens not an array"),
        }
    }

    // Test-only helper: little-endian f32 bytes.
    fn f32s_to_le_bytes(xs: &[f32]) -> Vec<u8> {
        let mut o = Vec::with_capacity(xs.len() * 4);
        for &x in xs {
            o.extend_from_slice(&x.to_le_bytes());
        }
        o
    }

    // Round-trip a tensor through the real writer+reader stack and return the
    // reconstructed f32 values. This is the encoder correctness oracle.
    pub(crate) fn dequant_via_reader(data: &[f32], ggml_type: u32, dims: &[u64]) -> Vec<f32> {
        use crate::gguf::{Gguf, MetaValue};
        let mut b = GgufBuilder::new();
        b.meta("general.architecture", MetaValue::String("llama".into()));
        b.tensor(
            "t",
            dims,
            ggml_type,
            encode_tensor(data, ggml_type).unwrap(),
        );
        let g = Gguf::parse(b.into_bytes()).unwrap();
        g.dequantize(g.tensor("t").unwrap()).unwrap()
    }

    fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn enc_f32_is_exact() {
        use crate::gguf::GGML_F32;
        let x: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        let back = dequant_via_reader(&x, GGML_F32, &[64]);
        assert_eq!(back, x);
    }

    #[test]
    fn enc_f16_within_half_ulp() {
        use crate::gguf::GGML_F16;
        let x: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.01).collect();
        let back = dequant_via_reader(&x, GGML_F16, &[64]);
        // Each value within f16 relative resolution near this magnitude.
        assert!(max_abs_err(&x, &back) < 0.001);
    }

    #[test]
    fn enc_q8_0_within_block_bound() {
        use crate::gguf::GGML_Q8_0;
        let x: Vec<f32> = (0..64).map(|i| ((i * 7) % 13) as f32 - 6.0).collect();
        let back = dequant_via_reader(&x, GGML_Q8_0, &[64]);
        let amax = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(max_abs_err(&x, &back) <= amax / 127.0 + 1e-4);
    }

    #[test]
    fn enc_q8_1_within_block_bound() {
        use crate::gguf::GGML_Q8_1;
        let x: Vec<f32> = (0..64).map(|i| ((i * 5) % 11) as f32 - 5.0).collect();
        let back = dequant_via_reader(&x, GGML_Q8_1, &[64]);
        let amax = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(max_abs_err(&x, &back) <= amax / 127.0 + 1e-4);
    }

    #[test]
    fn encode_tensor_rejects_ragged_quantized() {
        use crate::gguf::GGML_Q8_0;
        // 20 is not a multiple of QK=32.
        let x = vec![0.0f32; 20];
        assert!(encode_tensor(&x, GGML_Q8_0).is_err());
    }

    #[test]
    fn enc_q4_0_within_block_bound() {
        use crate::gguf::GGML_Q4_0;
        let x: Vec<f32> = (0..64).map(|i| ((i * 3) % 17) as f32 - 8.0).collect();
        let back = dequant_via_reader(&x, GGML_Q4_0, &[64]);
        // Symmetric 4-bit: 15 levels across [-amax, amax] → step ≈ amax/7.5.
        let amax = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(max_abs_err(&x, &back) <= amax / 7.0 + 1e-4);
    }

    #[test]
    fn enc_q4_1_within_block_bound() {
        use crate::gguf::GGML_Q4_1;
        let x: Vec<f32> = (0..64).map(|i| (i as f32) * 0.25 - 3.0).collect();
        let back = dequant_via_reader(&x, GGML_Q4_1, &[64]);
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for &v in &x {
            lo = lo.min(v);
            hi = hi.max(v);
        }
        assert!(max_abs_err(&x, &back) <= (hi - lo) / 15.0 + 1e-4);
    }

    #[test]
    fn put_scale_min_k4_inverts_get_scale_min_k4() {
        use crate::gguf::get_scale_min_k4;
        // Distinct 6-bit values per sub-block, exercising all bit positions.
        let sc = [1u8, 2, 3, 62, 33, 40, 63, 17];
        let m = [7u8, 8, 63, 4, 21, 60, 1, 34];
        let packed = put_scale_min_k4(&sc, &m);
        for j in 0..8 {
            let (d, mn) = get_scale_min_k4(j, &packed);
            assert_eq!(d, sc[j], "scale mismatch at sub-block {j}");
            assert_eq!(mn, m[j], "min mismatch at sub-block {j}");
        }
    }

    #[test]
    fn enc_q4_k_constant_block_is_near_exact() {
        use crate::gguf::GGML_Q4_K;
        for &c in &[0.5f32, -0.75, 0.0] {
            let x = vec![c; QK_K];
            let back = dequant_via_reader(&x, GGML_Q4_K, &[QK_K as u64]);
            let err = max_abs_err(&x, &back);
            assert!(err <= c.abs() * 0.02 + 1e-3, "c={c} err={err}");
        }
    }

    #[test]
    #[allow(clippy::excessive_precision)]
    fn enc_q4_k_within_bound() {
        use crate::gguf::GGML_Q4_K;
        // Deterministic pseudo-random-ish values in [-1, 1].
        let x: Vec<f32> = (0..QK_K)
            .map(|i| ((i as f32 * 12.9898).sin() * 43758.5453).fract() * 2.0 - 1.0)
            .collect();
        let back = dequant_via_reader(&x, GGML_Q4_K, &[QK_K as u64]);
        let amax = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(max_abs_err(&x, &back) <= amax * 0.15, "q4_k error too high");
    }

    #[test]
    fn enc_q5_k_constant_block_is_near_exact() {
        use crate::gguf::GGML_Q5_K;
        for &c in &[0.5f32, -0.75] {
            let x = vec![c; QK_K];
            let back = dequant_via_reader(&x, GGML_Q5_K, &[QK_K as u64]);
            assert!(max_abs_err(&x, &back) <= c.abs() * 0.02 + 1e-3, "c={c}");
        }
    }

    #[test]
    #[allow(clippy::excessive_precision)]
    fn enc_q5_k_within_bound() {
        use crate::gguf::GGML_Q5_K;
        let x: Vec<f32> = (0..QK_K)
            .map(|i| ((i as f32 * 7.13).sin() * 1234.5).fract() * 2.0 - 1.0)
            .collect();
        let back = dequant_via_reader(&x, GGML_Q5_K, &[QK_K as u64]);
        let amax = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(max_abs_err(&x, &back) <= amax * 0.10, "q5_k error too high");
    }

    #[test]
    fn enc_q6_k_constant_block_is_near_exact() {
        use crate::gguf::GGML_Q6_K;
        for &c in &[0.5f32, -0.75] {
            let x = vec![c; QK_K];
            let back = dequant_via_reader(&x, GGML_Q6_K, &[QK_K as u64]);
            assert!(max_abs_err(&x, &back) <= c.abs() * 0.02 + 1e-3, "c={c}");
        }
    }

    #[test]
    #[allow(clippy::excessive_precision)]
    fn enc_q6_k_within_bound() {
        use crate::gguf::GGML_Q6_K;
        let x: Vec<f32> = (0..QK_K)
            .map(|i| ((i as f32 * 3.7).cos() * 987.65).fract() * 2.0 - 1.0)
            .collect();
        let back = dequant_via_reader(&x, GGML_Q6_K, &[QK_K as u64]);
        let amax = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(max_abs_err(&x, &back) <= amax * 0.05, "q6_k error too high");
    }

    #[test]
    fn plan_tensor_type_applies_policy() {
        use crate::gguf::{GGML_F16, GGML_F32, GGML_Q4_K, GGML_Q8_0};
        // 1-D tensors (norms/biases) are always F32.
        assert_eq!(
            plan_tensor_type(false, 8, GgufQuant::Q4K),
            (GGML_F32, false)
        );
        // 2-D, divisible by 256 → k-quant target.
        assert_eq!(
            plan_tensor_type(true, 256, GgufQuant::Q4K),
            (GGML_Q4_K, false)
        );
        // 2-D, NOT divisible by 256 → k-quant falls back to F16.
        assert_eq!(
            plan_tensor_type(true, 100, GgufQuant::Q4K),
            (GGML_F16, true)
        );
        // 2-D, divisible by 32 → legacy target.
        assert_eq!(
            plan_tensor_type(true, 64, GgufQuant::Q8_0),
            (GGML_Q8_0, false)
        );
        // 2-D, NOT divisible by 32 → legacy falls back to F16.
        assert_eq!(
            plan_tensor_type(true, 20, GgufQuant::Q8_0),
            (GGML_F16, true)
        );
        // F16/F32 targets have no divisibility requirement.
        assert_eq!(
            plan_tensor_type(true, 20, GgufQuant::F16),
            (GGML_F16, false)
        );
    }

    // ── write_llama_gguf ─────────────────────────────────────────────────────

    // Deterministic filler in [-0.1, 0.1], shared by fixture builders below and
    // by tests that need to know a fixture tensor's exact source values.
    fn det_fill(seed: usize, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (((seed * 131 + i * 17) % 101) as f32 / 500.0) - 0.1)
            .collect()
    }

    // Build a minimal but valid `llama` GGUF (2 layers, dim 8) as bytes, with all
    // tensors F32 and a small gpt2 tokenizer block. Deterministic weights.
    fn tiny_llama_gguf_bytes() -> Vec<u8> {
        tiny_llama_gguf_bytes_ex(false)
    }

    // Same fixture as `tiny_llama_gguf_bytes`, but when `with_qkv_bias` is true
    // also emits deterministic F32 `blk.{i}.attn_{q,k,v}.bias` tensors (used by
    // `write_llama_gguf_roundtrips_qkv_biases` to exercise the bias-emission
    // path, which the plain fixture never touches).
    fn tiny_llama_gguf_bytes_ex(with_qkv_bias: bool) -> Vec<u8> {
        use crate::gguf::{MetaValue, GGML_F32};
        let (dim, n_layers, n_heads, ffn, vocab, ctx) =
            (8usize, 2usize, 2usize, 16usize, 32usize, 16usize);
        let head_dim = dim / n_heads;

        let gen = det_fill;

        let mut b = GgufBuilder::new();
        b.meta("general.architecture", MetaValue::String("llama".into()));
        b.meta("general.name", MetaValue::String("tiny".into()));
        b.meta("llama.embedding_length", MetaValue::U32(dim as u32));
        b.meta("llama.block_count", MetaValue::U32(n_layers as u32));
        b.meta("llama.attention.head_count", MetaValue::U32(n_heads as u32));
        b.meta(
            "llama.attention.head_count_kv",
            MetaValue::U32(n_heads as u32),
        );
        b.meta("llama.feed_forward_length", MetaValue::U32(ffn as u32));
        b.meta("llama.context_length", MetaValue::U32(ctx as u32));
        b.meta(
            "llama.attention.layer_norm_rms_epsilon",
            MetaValue::F32(1e-5),
        );
        // Minimal gpt2 tokenizer block (metadata only — round-trip target).
        let toks: Vec<MetaValue> = (0..vocab)
            .map(|i| MetaValue::String(format!("t{i}")))
            .collect();
        b.meta("tokenizer.ggml.model", MetaValue::String("gpt2".into()));
        b.meta("tokenizer.ggml.tokens", MetaValue::Array(toks));

        let f32t = |b: &mut GgufBuilder, name: &str, dims: &[u64], seed: usize| {
            let n: usize = dims.iter().product::<u64>() as usize;
            b.tensor(name, dims, GGML_F32, f32s_to_le_bytes(&gen(seed, n)));
        };

        // token_embd: dims [dim, vocab], data row-major [vocab, dim].
        f32t(&mut b, "token_embd.weight", &[dim as u64, vocab as u64], 1);
        for i in 0..n_layers {
            let p = format!("blk.{i}");
            f32t(
                &mut b,
                &format!("{p}.attn_norm.weight"),
                &[dim as u64],
                10 + i,
            );
            // Projections: GGUF dims [n_in, n_out].
            f32t(
                &mut b,
                &format!("{p}.attn_q.weight"),
                &[dim as u64, (n_heads * head_dim) as u64],
                20 + i,
            );
            if with_qkv_bias {
                f32t(
                    &mut b,
                    &format!("{p}.attn_q.bias"),
                    &[(n_heads * head_dim) as u64],
                    300 + i,
                );
            }
            f32t(
                &mut b,
                &format!("{p}.attn_k.weight"),
                &[dim as u64, (n_heads * head_dim) as u64],
                30 + i,
            );
            if with_qkv_bias {
                f32t(
                    &mut b,
                    &format!("{p}.attn_k.bias"),
                    &[(n_heads * head_dim) as u64],
                    310 + i,
                );
            }
            f32t(
                &mut b,
                &format!("{p}.attn_v.weight"),
                &[dim as u64, (n_heads * head_dim) as u64],
                40 + i,
            );
            if with_qkv_bias {
                f32t(
                    &mut b,
                    &format!("{p}.attn_v.bias"),
                    &[(n_heads * head_dim) as u64],
                    320 + i,
                );
            }
            f32t(
                &mut b,
                &format!("{p}.attn_output.weight"),
                &[(n_heads * head_dim) as u64, dim as u64],
                50 + i,
            );
            if with_qkv_bias {
                // The reader never loads a bias for attn_output (it calls
                // `linear_from` with `bias_name: None`), so this tensor is inert
                // for `load_llama_prec` — it exists only so the writer's "never
                // emit for non-qkv stems" invariant (Fix 1) is a genuine
                // regression guard below: without it, `source.tensor(&bname)`
                // would be `None` for this stem regardless of the `can_have_bias`
                // gate, and a wrongly-gated regression would go undetected.
                f32t(
                    &mut b,
                    &format!("{p}.attn_output.bias"),
                    &[dim as u64],
                    330 + i,
                );
            }
            f32t(
                &mut b,
                &format!("{p}.ffn_norm.weight"),
                &[dim as u64],
                60 + i,
            );
            f32t(
                &mut b,
                &format!("{p}.ffn_gate.weight"),
                &[dim as u64, ffn as u64],
                70 + i,
            );
            f32t(
                &mut b,
                &format!("{p}.ffn_up.weight"),
                &[dim as u64, ffn as u64],
                80 + i,
            );
            f32t(
                &mut b,
                &format!("{p}.ffn_down.weight"),
                &[ffn as u64, dim as u64],
                90 + i,
            );
        }
        f32t(&mut b, "output_norm.weight", &[dim as u64], 200);
        f32t(&mut b, "output.weight", &[dim as u64, vocab as u64], 201);
        b.into_bytes()
    }

    #[test]
    fn write_llama_gguf_f32_roundtrip_is_exact() {
        use crate::gguf::Gguf;

        let g0 = Gguf::parse(tiny_llama_gguf_bytes()).unwrap();
        let model = g0.load_llama_prec(None).unwrap();

        let out = llama_gguf_bytes(&model, &g0, GgufQuant::F32).unwrap();
        let g1 = Gguf::parse(out).unwrap();
        let model2 = g1.load_llama_prec(None).unwrap();

        // Same logits for a fixed token sequence (F32 → bit-exact weights).
        let toks = [1usize, 5, 2, 7];
        let l0 = model.forward_tokens(&toks).unwrap();
        let l1 = model2.forward_tokens(&toks).unwrap();
        assert_eq!(l0.data, l1.data);
    }

    #[test]
    fn write_llama_gguf_preserves_tokenizer_metadata() {
        use crate::gguf::Gguf;
        let g0 = Gguf::parse(tiny_llama_gguf_bytes()).unwrap();
        let model = g0.load_llama_prec(None).unwrap();
        let out = llama_gguf_bytes(&model, &g0, GgufQuant::Q8_0).unwrap();
        let g1 = Gguf::parse(out).unwrap();
        assert_eq!(
            g1.meta("tokenizer.ggml.model").unwrap().as_str(),
            Some("gpt2")
        );
        assert_eq!(
            g0.meta("tokenizer.ggml.tokens"),
            g1.meta("tokenizer.ggml.tokens")
        );
    }

    // Positive-path counterpart to the gating in `add_linear` (Task 11 review,
    // Fix 1/2): `tiny_llama_gguf_bytes` never carries a `*.bias` tensor, so
    // without this test the bias-emission branch is only ever exercised as
    // `false`. This fixture adds deterministic q/k/v biases and checks both
    // that they round-trip and that no other stem ever emits one.
    #[test]
    fn write_llama_gguf_roundtrips_qkv_biases() {
        use crate::gguf::Gguf;

        // Mirrors the (dim, n_layers, n_heads, head_dim) fixed inside
        // `tiny_llama_gguf_bytes_ex`: dim=8, n_heads=2 → head_dim=4, so the q/k/v
        // output dim (n_heads*head_dim) is 8.
        let n_layers = 2usize;
        let qkv_dim = 8usize;

        let g0 = Gguf::parse(tiny_llama_gguf_bytes_ex(true)).unwrap();
        let model = g0.load_llama_prec(None).unwrap();

        let out = llama_gguf_bytes(&model, &g0, GgufQuant::F32).unwrap();
        let g1 = Gguf::parse(out).unwrap();

        // (a) q/k/v biases are present and match the source values exactly
        // (F32 in, F32 out — lossless).
        for i in 0..n_layers {
            let p = format!("blk.{i}");
            for (stem, seed) in [
                ("attn_q", 300 + i),
                ("attn_k", 310 + i),
                ("attn_v", 320 + i),
            ] {
                let name = format!("{p}.{stem}.bias");
                let expected = det_fill(seed, qkv_dim);
                let t = g1
                    .tensor(&name)
                    .unwrap_or_else(|| panic!("expected '{name}' to be emitted"));
                let got = g1.dequantize(t).unwrap();
                assert_eq!(got.len(), expected.len(), "{name}: length mismatch");
                for (g, e) in got.iter().zip(&expected) {
                    assert!((g - e).abs() < 1e-6, "{name}: got {g}, want {e}");
                }
            }
        }

        // (b) non-qkv stems never emit a bias, even though every Linear (incl.
        // attn_output/ffn_*) carries a zero-filled in-memory `bias` vector.
        assert!(g1.tensor("blk.0.attn_output.bias").is_none());
        assert!(g1.tensor("blk.0.ffn_gate.bias").is_none());
        assert!(g1.tensor("blk.0.ffn_up.bias").is_none());
        assert!(g1.tensor("blk.0.ffn_down.bias").is_none());
        assert!(g1.tensor("blk.1.attn_output.bias").is_none());
    }
}
