//! Pure-`std` GGUF *writer* — serialize a Ferrum `LlamaModel` back to a GGUF v3
//! file that runs in the wider ecosystem (llama.cpp / ollama / LM Studio).
//!
//! The reader in [`crate::gguf`] is the specification: every block encoder here
//! is the exact inverse of the matching `dequant_*` decoder, and is verified by
//! round-tripping through [`crate::gguf::Gguf`].

use crate::error::{InferError, Result};
use crate::gguf::{
    MetaValue, DEFAULT_ALIGNMENT, GGML_F16, GGML_F32, GGML_Q4_0, GGML_Q4_1, GGML_Q4_K, GGML_Q5_K,
    GGML_Q6_K, GGML_Q8_0, GGML_Q8_1, GGUF_MAGIC, QK, VT_ARRAY, VT_BOOL, VT_F32, VT_F64, VT_I16,
    VT_I32, VT_I64, VT_I8, VT_STRING, VT_U16, VT_U32, VT_U64, VT_U8,
};

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
}
