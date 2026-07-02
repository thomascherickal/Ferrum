//! Pure-`std` GGUF reader — import externally pretrained weights into Ferrum.
//!
//! GGUF (the llama.cpp model container) is the lingua franca for distributing
//! quantized LLM/SLM weights. This module parses it with **zero dependencies**
//! and **no `unsafe`**: header, the full typed metadata key/value table, the
//! tensor directory, and dequantization of the common block formats (F32, F16,
//! Q8_0, Q4_0, Q4_1, Q8_1) into f32 — from which [`crate::quant::QWeight`] packs
//! them to Ferrum's in-memory int4/int8 (Opt#1).
//!
//! There is no `mmap` (it would need `unsafe`), so [`Gguf::from_path`] reads the
//! file into memory; that is fine for the import step, where the int4 result is
//! what stays resident.
//!
//! ## Architecture gap — read this before expecting a Llama to *run*
//!
//! Importing weights ≠ running the model. Ferrum's [`crate::layer::TransformerBlock`]
//! is a **learned-positional-embedding + LayerNorm + ReLU-FFN** decoder. Modern
//! GGUF models (Llama/Qwen/Mistral/Phi) use **RoPE** positions, **RMSNorm**,
//! **SwiGLU** gated FFNs, and often **grouped-query attention** — none of which
//! this block implements. So this reader will faithfully import the *tensors* of
//! such a model, but running it end-to-end additionally requires a matching
//! block (RoPE + RMSNorm + SwiGLU + GQA). [`Gguf::into_ferrum_slm`] therefore
//! converts only models whose GGUF `*.architecture` Ferrum can actually execute,
//! and returns a clear error otherwise rather than producing wrong output.

use crate::error::{InferError, Result};
use crate::layer::Linear;
use crate::llm::{Attention, FeedForward, LlamaBlock, LlamaConfig, LlamaModel, RmsNorm, RopeType};
use crate::quant::{QKind, QWeight};
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::sync::Mutex;

pub(crate) const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" little-endian
pub(crate) const DEFAULT_ALIGNMENT: u64 = 32;

// GGML tensor element types (subset we can dequantize).
pub(crate) const GGML_F32: u32 = 0;
pub(crate) const GGML_F16: u32 = 1;
pub(crate) const GGML_Q4_0: u32 = 2;
pub(crate) const GGML_Q4_1: u32 = 3;
pub(crate) const GGML_Q8_0: u32 = 8;
pub(crate) const GGML_Q8_1: u32 = 9;
// k-quant super-block formats (G-K). These dominate modern GGUF downloads:
// `Q4_K_M` mixes Q4_K / Q5_K / Q6_K across tensors, so all three are needed to
// load one real checkpoint.
pub(crate) const GGML_Q4_K: u32 = 12;
pub(crate) const GGML_Q5_K: u32 = 13;
pub(crate) const GGML_Q6_K: u32 = 14;

pub(crate) const QK: usize = 32; // GGML block length for the legacy quant formats
pub(crate) const QK_K: usize = 256; // super-block length for the k-quant formats
                                    // On-disk bytes per k-quant super-block of QK_K weights (must match ggml).
pub(crate) const Q4_K_BLOCK: usize = 2 + 2 + 12 + QK_K / 2; // d, dmin, 6-bit scales, 4-bit qs = 144
pub(crate) const Q5_K_BLOCK: usize = 2 + 2 + 12 + QK_K / 8 + QK_K / 2; // + qh high bits = 176
pub(crate) const Q6_K_BLOCK: usize = QK_K / 2 + QK_K / 4 + QK_K / 16 + 2; // ql, qh, scales, d = 210

// GGUF metadata value type tags.
pub(crate) const VT_U8: u32 = 0;
pub(crate) const VT_I8: u32 = 1;
pub(crate) const VT_U16: u32 = 2;
pub(crate) const VT_I16: u32 = 3;
pub(crate) const VT_U32: u32 = 4;
pub(crate) const VT_I32: u32 = 5;
pub(crate) const VT_F32: u32 = 6;
pub(crate) const VT_BOOL: u32 = 7;
pub(crate) const VT_STRING: u32 = 8;
pub(crate) const VT_ARRAY: u32 = 9;
pub(crate) const VT_U64: u32 = 10;
pub(crate) const VT_I64: u32 = 11;
pub(crate) const VT_F64: u32 = 12;

/// A typed GGUF metadata value.
#[derive(Clone, Debug, PartialEq)]
pub enum MetaValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<MetaValue>),
}

impl MetaValue {
    /// Best-effort `usize` view of any integer scalar.
    pub fn as_usize(&self) -> Option<usize> {
        match *self {
            MetaValue::U8(v) => Some(v as usize),
            MetaValue::I8(v) if v >= 0 => Some(v as usize),
            MetaValue::U16(v) => Some(v as usize),
            MetaValue::I16(v) if v >= 0 => Some(v as usize),
            MetaValue::U32(v) => Some(v as usize),
            MetaValue::I32(v) if v >= 0 => Some(v as usize),
            MetaValue::U64(v) => Some(v as usize),
            MetaValue::I64(v) if v >= 0 => Some(v as usize),
            _ => None,
        }
    }
    /// `f32` view of any float/integer scalar.
    pub fn as_f32(&self) -> Option<f32> {
        match *self {
            MetaValue::F32(v) => Some(v),
            MetaValue::F64(v) => Some(v as f32),
            _ => self.as_usize().map(|u| u as f32),
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            MetaValue::String(s) => Some(s),
            _ => None,
        }
    }
}

/// One tensor's directory entry (name, shape, element type, data offset).
#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    /// Dimensions in GGML order (`dims[0]` is the fastest-varying axis).
    pub dims: Vec<u64>,
    pub ggml_type: u32,
    /// Byte offset of this tensor within the tensor-data section.
    pub offset: u64,
}

impl TensorInfo {
    /// Total element count (product of dimensions). Saturates at `usize::MAX`
    /// instead of overflowing, so a malicious dims table cannot panic a debug
    /// build (or truncate on 32-bit); the impossible size then fails the
    /// checked sizing in `type_nbytes` with a clean error rather than wrapping.
    pub fn num_elements(&self) -> usize {
        self.dims.iter().fold(1usize, |acc, &d| {
            acc.saturating_mul(usize::try_from(d).unwrap_or(usize::MAX))
        })
    }
}

/// Where a [`Gguf`]'s tensor data lives. `Memory` holds the whole file (the
/// `from_path`/`parse` path); `Streamed` keeps only an open file handle and
/// reads each tensor's bytes on demand (the `open` path — see [`Gguf::open`]),
/// so peak RAM during import is one tensor at a time, not the whole multi-GB
/// file. No `mmap` (that needs `unsafe`); a plain `File` + `seek` + `read`,
/// guarded by a `Mutex` so reads work through a shared `&self`.
enum Source {
    Memory(Vec<u8>),
    Streamed { file: Mutex<File>, len: usize },
}

/// A parsed GGUF file: metadata, tensor directory, and a handle to the tensor
/// data (in memory or streamed from disk — see [`Source`]).
pub struct Gguf {
    pub version: u32,
    pub metadata: BTreeMap<String, MetaValue>,
    pub tensors: Vec<TensorInfo>,
    /// Absolute byte offset where the (aligned) tensor-data section begins.
    data_offset: usize,
    source: Source,
}

/// Little-endian cursor over the GGUF byte stream.
struct Cur<'a> {
    b: &'a [u8],
    pos: usize,
}
impl<'a> Cur<'a> {
    fn new(b: &'a [u8]) -> Self {
        Self { b, pos: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| fmt("offset overflow"))?;
        if end > self.b.len() {
            return Err(fmt(&format!("GGUF EOF at +{}: need {n}", self.pos)));
        }
        let s = &self.b[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }
    fn u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn u64(&mut self) -> Result<u64> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }
    fn i8(&mut self) -> Result<i8> {
        Ok(self.u8()? as i8)
    }
    fn i16(&mut self) -> Result<i16> {
        Ok(self.u16()? as i16)
    }
    fn i32(&mut self) -> Result<i32> {
        Ok(self.u32()? as i32)
    }
    fn i64(&mut self) -> Result<i64> {
        Ok(self.u64()? as i64)
    }
    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_bits(self.u32()?))
    }
    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_bits(self.u64()?))
    }
    /// A GGUF string: `u64` byte length followed by UTF-8 bytes.
    fn string(&mut self) -> Result<String> {
        let n = self.u64()? as usize;
        let s = self.take(n)?;
        Ok(String::from_utf8_lossy(s).into_owned())
    }
}

fn fmt(msg: &str) -> InferError {
    InferError::Format(msg.to_string())
}

fn read_value(c: &mut Cur, vtype: u32) -> Result<MetaValue> {
    Ok(match vtype {
        VT_U8 => MetaValue::U8(c.u8()?),
        VT_I8 => MetaValue::I8(c.i8()?),
        VT_U16 => MetaValue::U16(c.u16()?),
        VT_I16 => MetaValue::I16(c.i16()?),
        VT_U32 => MetaValue::U32(c.u32()?),
        VT_I32 => MetaValue::I32(c.i32()?),
        VT_F32 => MetaValue::F32(c.f32()?),
        VT_BOOL => MetaValue::Bool(c.u8()? != 0),
        VT_STRING => MetaValue::String(c.string()?),
        VT_U64 => MetaValue::U64(c.u64()?),
        VT_I64 => MetaValue::I64(c.i64()?),
        VT_F64 => MetaValue::F64(c.f64()?),
        VT_ARRAY => {
            let elem_type = c.u32()?;
            if elem_type == VT_ARRAY {
                return Err(fmt("GGUF nested arrays are not supported"));
            }
            let count = c.u64()? as usize;
            // Guard against absurd counts before reserving.
            if count > c.b.len() {
                return Err(fmt("GGUF array count exceeds file size"));
            }
            let mut items = Vec::with_capacity(count.min(1024));
            for _ in 0..count {
                items.push(read_value(c, elem_type)?);
            }
            MetaValue::Array(items)
        }
        other => return Err(fmt(&format!("GGUF unknown metadata value type {other}"))),
    })
}

/// Convert an IEEE-754 half (`u16`) to `f32` (normals, subnormals, inf/NaN).
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as i32;
    let mant = (h & 0x3FF) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            // Subnormal f16: value = mant · 2⁻²⁴. Normalize so bit 10 (the hidden
            // 1) is set, tracking the exponent from the subnormal base −14.
            let mut m = mant;
            let mut e = -14i32;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3FF;
            let f32_exp = (e + 127) as u32;
            (sign << 31) | (f32_exp << 23) | (m << 13)
        }
    } else if exp == 0x1F {
        // Inf / NaN.
        (sign << 31) | (0xFF << 23) | (mant << 13)
    } else {
        let f32_exp = (exp - 15 + 127) as u32;
        (sign << 31) | (f32_exp << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

/// Bytes one tensor of `n` elements of `ggml_type` occupies on disk.
fn type_nbytes(ggml_type: u32, n: usize) -> Result<usize> {
    let blocks = || -> Result<usize> {
        if !n.is_multiple_of(QK) {
            return Err(fmt(&format!(
                "quantized tensor element count {n} is not a multiple of {QK}"
            )));
        }
        Ok(n / QK)
    };
    let kblocks = || -> Result<usize> {
        if !n.is_multiple_of(QK_K) {
            return Err(fmt(&format!(
                "k-quant tensor element count {n} is not a multiple of {QK_K}"
            )));
        }
        Ok(n / QK_K)
    };
    // Checked multiplications: a hostile dims table (whose element count
    // saturates in `num_elements`) must produce a clean error, never an
    // overflow panic (debug) or a wrapped size (release).
    let mul = |a: usize, b: usize| {
        a.checked_mul(b)
            .ok_or_else(|| fmt("tensor byte size overflows"))
    };
    match ggml_type {
        GGML_F32 => mul(n, 4),
        GGML_F16 => mul(n, 2),
        GGML_Q8_0 => mul(blocks()?, 2 + QK),     // f16 d + 32×i8
        GGML_Q8_1 => mul(blocks()?, 2 + 2 + QK), // f16 d + f16 s + 32×i8
        GGML_Q4_0 => mul(blocks()?, 2 + QK / 2), // f16 d + 16 bytes
        GGML_Q4_1 => mul(blocks()?, 2 + 2 + QK / 2), // f16 d + f16 m + 16 bytes
        GGML_Q4_K => mul(kblocks()?, Q4_K_BLOCK),
        GGML_Q5_K => mul(kblocks()?, Q5_K_BLOCK),
        GGML_Q6_K => mul(kblocks()?, Q6_K_BLOCK),
        other => Err(fmt(&format!(
            "GGUF tensor type {other} is unsupported for sizing"
        ))),
    }
}

fn align_up(x: usize, a: usize) -> usize {
    if a == 0 {
        x
    } else {
        x.div_ceil(a) * a
    }
}

/// `(version, metadata, tensor directory, absolute tensor-data offset)`.
type ParsedHeader = (u32, BTreeMap<String, MetaValue>, Vec<TensorInfo>, usize);

/// Parse the header, metadata table, and tensor directory from a buffer that
/// contains at least the whole header region (it may be a prefix of a larger
/// file — tensor data after `data_offset` is not required).
fn parse_header(b: &[u8]) -> Result<ParsedHeader> {
    let mut c = Cur::new(b);
    if c.u32()? != GGUF_MAGIC {
        return Err(fmt("not a GGUF file (bad magic)"));
    }
    let version = c.u32()?;
    if version != 2 && version != 3 {
        return Err(fmt(&format!(
            "unsupported GGUF version {version} (need 2 or 3)"
        )));
    }
    let tensor_count = c.u64()? as usize;
    let kv_count = c.u64()? as usize;

    let mut metadata = BTreeMap::new();
    for _ in 0..kv_count {
        let key = c.string()?;
        let vtype = c.u32()?;
        let value = read_value(&mut c, vtype)?;
        metadata.insert(key, value);
    }

    let mut tensors = Vec::with_capacity(tensor_count.min(4096));
    for _ in 0..tensor_count {
        let name = c.string()?;
        let n_dims = c.u32()? as usize;
        if n_dims > 8 {
            return Err(fmt(&format!(
                "tensor '{name}' has implausible {n_dims} dims"
            )));
        }
        let mut dims = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            dims.push(c.u64()?);
        }
        let ggml_type = c.u32()?;
        let offset = c.u64()?;
        tensors.push(TensorInfo {
            name,
            dims,
            ggml_type,
            offset,
        });
    }

    let alignment = metadata
        .get("general.alignment")
        .and_then(|v| v.as_usize())
        .map(|a| a as u64)
        .unwrap_or(DEFAULT_ALIGNMENT) as usize;
    let data_offset = align_up(c.pos, alignment);
    Ok((version, metadata, tensors, data_offset))
}

/// Whether an error is the "ran off the end of the buffer" kind (vs. a real
/// format error). Used by [`Gguf::open`] to know when to read a larger prefix.
fn is_truncation(e: &InferError) -> bool {
    matches!(e, InferError::Format(m) if m.starts_with("GGUF EOF"))
}

/// Read into `buf` until it is full or EOF (handling short/interrupted reads).
/// Returns the number of bytes actually read.
fn read_fully(f: &mut File, buf: &mut [u8]) -> Result<usize> {
    let mut filled = 0;
    while filled < buf.len() {
        match f.read(&mut buf[filled..]) {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e.into()),
        }
    }
    Ok(filled)
}

impl Gguf {
    /// Read and parse a GGUF file from disk **fully into memory**. Convenient for
    /// small files and tests; for large checkpoints prefer [`Gguf::open`], which
    /// streams tensor data from disk instead of holding the whole file resident.
    pub fn from_path(path: &str) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        Self::parse(bytes)
    }

    /// Open a GGUF file in **streamed** mode (G-mmap): parse only the header
    /// prefix into memory and keep the file open, reading each tensor's bytes on
    /// demand during [`Self::dequantize`] / [`Self::load_llama`]. Peak resident
    /// memory for the parse step is the header (typically a few MB), not the
    /// whole multi-GB file. The prefix is grown geometrically until the header
    /// fits, so tensor *data* is never read here.
    pub fn open(path: &str) -> Result<Self> {
        let mut file = File::open(path)?;
        let len = file.metadata()?.len() as usize;
        let mut cap = (1usize << 20).min(len.max(1)); // start at 1 MiB
        loop {
            file.seek(SeekFrom::Start(0))?;
            let mut prefix = vec![0u8; cap];
            let got = read_fully(&mut file, &mut prefix)?;
            prefix.truncate(got);
            match parse_header(&prefix) {
                Ok((version, metadata, tensors, data_offset)) => {
                    if data_offset > len {
                        return Err(fmt("GGUF tensor-data section begins past EOF"));
                    }
                    return Ok(Self {
                        version,
                        metadata,
                        tensors,
                        data_offset,
                        source: Source::Streamed {
                            file: Mutex::new(file),
                            len,
                        },
                    });
                }
                // The header didn't fit in the prefix yet: read a bigger one.
                Err(ref e) if cap < len && is_truncation(e) => {
                    cap = cap.saturating_mul(2).min(len);
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Parse GGUF from an owned byte buffer (held resident; see [`Self::open`]
    /// for the streamed alternative).
    pub fn parse(bytes: Vec<u8>) -> Result<Self> {
        let (version, metadata, tensors, data_offset) = parse_header(&bytes)?;
        if data_offset > bytes.len() {
            return Err(fmt("GGUF tensor-data section begins past EOF"));
        }
        Ok(Self {
            version,
            metadata,
            tensors,
            data_offset,
            source: Source::Memory(bytes),
        })
    }

    /// The model architecture string (`general.architecture`), e.g. `"llama"`.
    pub fn architecture(&self) -> Option<&str> {
        self.metadata
            .get("general.architecture")
            .and_then(|v| v.as_str())
    }

    /// Look up a metadata value by exact key.
    pub fn meta(&self, key: &str) -> Option<&MetaValue> {
        self.metadata.get(key)
    }

    /// Find a tensor by name.
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// The raw on-disk bytes of one tensor — borrowed from the in-memory buffer,
    /// or read from disk into an owned buffer when streamed.
    fn tensor_bytes(&self, t: &TensorInfo) -> Result<Cow<'_, [u8]>> {
        let n = t.num_elements();
        let len = type_nbytes(t.ggml_type, n)?;
        let start = self
            .data_offset
            .checked_add(t.offset as usize)
            .ok_or_else(|| fmt("tensor offset overflow"))?;
        let end = start
            .checked_add(len)
            .ok_or_else(|| fmt("tensor length overflow"))?;
        match &self.source {
            Source::Memory(bytes) => {
                if end > bytes.len() {
                    return Err(fmt(&format!("tensor '{}' data runs past EOF", t.name)));
                }
                Ok(Cow::Borrowed(&bytes[start..end]))
            }
            Source::Streamed { file, len: flen } => {
                if end > *flen {
                    return Err(fmt(&format!("tensor '{}' data runs past EOF", t.name)));
                }
                let mut f = file.lock().map_err(|_| fmt("GGUF file lock poisoned"))?;
                f.seek(SeekFrom::Start(start as u64))?;
                let mut buf = vec![0u8; len];
                f.read_exact(&mut buf)?;
                Ok(Cow::Owned(buf))
            }
        }
    }

    /// Dequantize a tensor to row-major f32 (in GGML storage order). Handles the
    /// legacy block formats and the common k-quant super-blocks (Q4_K/Q5_K/Q6_K).
    pub fn dequantize(&self, t: &TensorInfo) -> Result<Vec<f32>> {
        let raw = self.tensor_bytes(t)?;
        let raw = raw.as_ref();
        let n = t.num_elements();
        match t.ggml_type {
            GGML_F32 => Ok(raw
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()),
            GGML_F16 => Ok(raw
                .chunks_exact(2)
                .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                .collect()),
            GGML_Q8_0 => dequant_q8_0(raw, n),
            GGML_Q8_1 => dequant_q8_1(raw, n),
            GGML_Q4_0 => dequant_q4_0(raw, n),
            GGML_Q4_1 => dequant_q4_1(raw, n),
            GGML_Q4_K => dequant_q4_k(raw, n),
            GGML_Q5_K => dequant_q5_k(raw, n),
            GGML_Q6_K => dequant_q6_k(raw, n),
            other => Err(fmt(&format!(
                "GGUF tensor type {other} not yet supported for dequant \
                 (Q2_K/Q3_K and the IQ* formats need their own decoders)"
            ))),
        }
    }

    /// Dequantize a tensor by name, then pack it to a Ferrum [`QWeight`] of the
    /// requested in-memory precision (int4/int8). `rows` is the weight's input
    /// dimension (the per-row scale axis); `rows × cols` must equal the tensor's
    /// element count.
    pub fn quantized_weight(&self, name: &str, rows: usize, kind: QKind) -> Result<QWeight> {
        let t = self
            .tensor(name)
            .ok_or_else(|| fmt(&format!("GGUF tensor '{name}' not found")))?;
        let f = self.dequantize(t)?;
        if rows == 0 || f.len() % rows != 0 {
            return Err(fmt(&format!(
                "tensor '{name}' length {} not divisible by rows {rows}",
                f.len()
            )));
        }
        let cols = f.len() / rows;
        Ok(QWeight::from_f32(&f, rows, cols, kind))
    }

    fn meta_usize(&self, key: &str) -> Result<usize> {
        self.metadata
            .get(key)
            .and_then(|v| v.as_usize())
            .ok_or_else(|| fmt(&format!("GGUF missing/invalid metadata '{key}'")))
    }
    fn meta_opt_usize(&self, key: &str) -> Option<usize> {
        self.metadata.get(key).and_then(|v| v.as_usize())
    }
    fn meta_f32_or(&self, key: &str, default: f32) -> f32 {
        self.metadata
            .get(key)
            .and_then(|v| v.as_f32())
            .unwrap_or(default)
    }

    fn dequant_named(&self, name: &str) -> Result<Vec<f32>> {
        let t = self
            .tensor(name)
            .ok_or_else(|| fmt(&format!("GGUF tensor '{name}' not found")))?;
        self.dequantize(t)
    }

    /// Import a 2-D weight tensor as a Ferrum [`Linear`], transposing GGUF's
    /// `[n_out, n_in]` storage into Ferrum's `[n_in, n_out]`. With `prec =
    /// Some(kind)` the matrix is packed to in-memory int4/int8; with `prec =
    /// None` it is kept full **f32** — the precision-preserving import (G-Q) that
    /// avoids re-quantizing an already-quantized GGUF onto Ferrum's coarser
    /// per-row grid, at the cost of resident RAM. An optional bias is loaded if
    /// present.
    fn linear_from(
        &self,
        name: &str,
        bias_name: Option<&str>,
        prec: Option<QKind>,
    ) -> Result<Linear> {
        let t = self
            .tensor(name)
            .ok_or_else(|| fmt(&format!("GGUF tensor '{name}' not found")))?
            .clone();
        if t.dims.len() != 2 {
            return Err(fmt(&format!("tensor '{name}' is not 2-D")));
        }
        let n_in = t.dims[0] as usize;
        let n_out = t.dims[1] as usize;
        let flat = self.dequantize(&t)?; // row-major [n_out, n_in]
        let w = transpose_2d(&flat, n_out, n_in); // → [n_in, n_out]
        let bias = match bias_name.and_then(|b| self.tensor(b)).cloned() {
            Some(bt) => self.dequantize(&bt)?,
            None => vec![0.0; n_out],
        };
        if bias.len() != n_out {
            return Err(fmt(&format!("bias for '{name}' has wrong length")));
        }
        match prec {
            Some(kind) => {
                Linear::quantized(n_in, n_out, QWeight::from_f32(&w, n_in, n_out, kind), bias)
            }
            None => Linear::new(n_in, n_out, w, bias),
        }
    }

    /// Build a runnable [`LlamaModel`] from a `llama`- or `qwen2`-architecture
    /// GGUF, packing weight matrices to `kind` (int4/int8) in memory.
    ///
    /// RoPE uses [`RopeType::Norm`] — the convention llama.cpp's GGUF conversion
    /// permutes Q/K for. Other architectures are rejected. Bit-exact parity with
    /// llama.cpp on a real checkpoint is not asserted (it needs the actual file),
    /// but the metadata/tensor mapping, the transpose, and the dequant are all
    /// unit-covered, and the resulting model's cached decode matches its own full
    /// forward.
    pub fn load_llama(&self, kind: QKind) -> Result<LlamaModel> {
        self.load_llama_prec(Some(kind))
    }

    /// Like [`Self::load_llama`] but with explicit import precision: `Some(kind)`
    /// packs weights to in-memory int4/int8 (smaller, doubly-quantized);
    /// `None` keeps them full f32 (G-Q: no second quantization, larger RAM).
    pub fn load_llama_prec(&self, prec: Option<QKind>) -> Result<LlamaModel> {
        let arch = self
            .architecture()
            .ok_or_else(|| fmt("GGUF missing general.architecture"))?
            .to_string();
        if arch != "llama" && arch != "qwen2" {
            return Err(fmt(&format!(
                "architecture '{arch}' is not supported by load_llama (llama / qwen2 only)"
            )));
        }
        let dim = self.meta_usize(&format!("{arch}.embedding_length"))?;
        let n_layers = self.meta_usize(&format!("{arch}.block_count"))?;
        let n_heads = self.meta_usize(&format!("{arch}.attention.head_count"))?;
        let n_kv = self
            .meta_opt_usize(&format!("{arch}.attention.head_count_kv"))
            .unwrap_or(n_heads);
        let ffn = self.meta_usize(&format!("{arch}.feed_forward_length"))?;
        let head_dim = self
            .meta_opt_usize(&format!("{arch}.attention.key_length"))
            .unwrap_or(dim / n_heads.max(1));
        let rope_dim = self
            .meta_opt_usize(&format!("{arch}.rope.dimension_count"))
            .unwrap_or(head_dim);
        let rope_base = self.meta_f32_or(&format!("{arch}.rope.freq_base"), 10000.0);
        let eps = self.meta_f32_or(&format!("{arch}.attention.layer_norm_rms_epsilon"), 1e-5);
        let ctx = self
            .meta_opt_usize(&format!("{arch}.context_length"))
            .unwrap_or(2048);

        // Token embedding: GGUF `[dim, vocab]` (ne) → row-major `[vocab, dim]`,
        // which is exactly the lookup layout LlamaModel wants. Kept f32.
        let emb_t = self
            .tensor("token_embd.weight")
            .ok_or_else(|| fmt("GGUF missing token_embd.weight"))?;
        let vocab = if emb_t.dims.len() == 2 {
            emb_t.dims[1] as usize
        } else {
            self.meta_usize(&format!("{arch}.vocab_size"))?
        };
        let tok_emb = self.dequantize(emb_t)?;
        if tok_emb.len() != vocab * dim {
            return Err(fmt("token_embd.weight size does not match vocab × dim"));
        }

        let mut blocks = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let p = format!("blk.{i}");
            // Ferrum loads biases only for q/k/v (the qwen2 convention). A file
            // carrying a bias on any other projection would run silently wrong
            // with that bias dropped — refuse loudly instead.
            for stem in ["attn_output", "ffn_gate", "ffn_up", "ffn_down"] {
                let bname = format!("{p}.{stem}.bias");
                if self.tensor(&bname).is_some() {
                    return Err(fmt(&format!(
                        "GGUF tensor '{bname}' is not supported: Ferrum loads biases \
                         only for attn_q/attn_k/attn_v"
                    )));
                }
            }
            let attn_norm =
                RmsNorm::new(self.dequant_named(&format!("{p}.attn_norm.weight"))?, eps);
            // Qwen2 carries q/k/v biases; Llama does not (loaded only if present).
            let wq = self.linear_from(
                &format!("{p}.attn_q.weight"),
                Some(&format!("{p}.attn_q.bias")),
                prec,
            )?;
            let wk = self.linear_from(
                &format!("{p}.attn_k.weight"),
                Some(&format!("{p}.attn_k.bias")),
                prec,
            )?;
            let wv = self.linear_from(
                &format!("{p}.attn_v.weight"),
                Some(&format!("{p}.attn_v.bias")),
                prec,
            )?;
            let wo = self.linear_from(&format!("{p}.attn_output.weight"), None, prec)?;
            let attn = Attention::new(
                wq,
                wk,
                wv,
                wo,
                n_heads,
                n_kv,
                head_dim,
                rope_dim,
                rope_base,
                RopeType::Norm,
            )?;
            let ffn_norm = RmsNorm::new(self.dequant_named(&format!("{p}.ffn_norm.weight"))?, eps);
            let gate = self.linear_from(&format!("{p}.ffn_gate.weight"), None, prec)?;
            let up = self.linear_from(&format!("{p}.ffn_up.weight"), None, prec)?;
            let down = self.linear_from(&format!("{p}.ffn_down.weight"), None, prec)?;
            blocks.push(LlamaBlock {
                attn_norm,
                attn,
                ffn_norm,
                ffn: FeedForward::new(gate, up, down),
            });
        }

        let final_norm = RmsNorm::new(self.dequant_named("output_norm.weight")?, eps);
        // Same rule for the LM head: a bias there is never loaded, so refuse it.
        if self.tensor("output.bias").is_some() {
            return Err(fmt(
                "GGUF tensor 'output.bias' is not supported: Ferrum loads biases \
                 only for attn_q/attn_k/attn_v",
            ));
        }
        // LM head: explicit `output.weight`, or tied to the token embedding.
        let lm_head = if self.tensor("output.weight").is_some() {
            self.linear_from("output.weight", None, prec)?
        } else {
            let w = transpose_2d(&tok_emb, vocab, dim); // [vocab,dim] → [dim,vocab]
            match prec {
                Some(kind) => Linear::quantized(
                    dim,
                    vocab,
                    QWeight::from_f32(&w, dim, vocab, kind),
                    vec![0.0; vocab],
                )?,
                None => Linear::new(dim, vocab, w, vec![0.0; vocab])?,
            }
        };

        let cfg = LlamaConfig {
            vocab_size: vocab,
            model_dim: dim,
            n_layers,
            n_heads,
            n_kv_heads: n_kv,
            head_dim,
            ffn_dim: ffn,
            rope_dim,
            rope_base,
            rope_type: RopeType::Norm,
            norm_eps: eps,
            context_len: ctx,
        };
        Ok(LlamaModel {
            cfg,
            tok_emb,
            blocks,
            final_norm,
            lm_head,
        })
    }
}

/// Transpose a row-major `[rows, cols]` matrix to `[cols, rows]`.
fn transpose_2d(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

// ── Block dequantizers (GGML legacy formats) ──────────────────────────────────

fn dequant_q8_0(raw: &[u8], n: usize) -> Result<Vec<f32>> {
    let nb = n / QK;
    let mut out = vec![0.0f32; n];
    let mut c = Cur::new(raw);
    for blk in 0..nb {
        let d = f16_to_f32(c.u16()?);
        let qs = c.take(QK)?;
        for (j, &q) in qs.iter().enumerate() {
            out[blk * QK + j] = d * (q as i8) as f32;
        }
    }
    Ok(out)
}

fn dequant_q8_1(raw: &[u8], n: usize) -> Result<Vec<f32>> {
    let nb = n / QK;
    let mut out = vec![0.0f32; n];
    let mut c = Cur::new(raw);
    for blk in 0..nb {
        let d = f16_to_f32(c.u16()?);
        let _s = f16_to_f32(c.u16()?); // block sum (for dot products); unused here
        let qs = c.take(QK)?;
        for (j, &q) in qs.iter().enumerate() {
            out[blk * QK + j] = d * (q as i8) as f32;
        }
    }
    Ok(out)
}

fn dequant_q4_0(raw: &[u8], n: usize) -> Result<Vec<f32>> {
    let nb = n / QK;
    let mut out = vec![0.0f32; n];
    let mut c = Cur::new(raw);
    for blk in 0..nb {
        let d = f16_to_f32(c.u16()?);
        let qs = c.take(QK / 2)?;
        for (j, &byte) in qs.iter().enumerate() {
            // Low nibble → element j, high nibble → element j+16; centered at 8.
            let x0 = (byte & 0x0F) as i32 - 8;
            let x1 = (byte >> 4) as i32 - 8;
            out[blk * QK + j] = d * x0 as f32;
            out[blk * QK + j + QK / 2] = d * x1 as f32;
        }
    }
    Ok(out)
}

fn dequant_q4_1(raw: &[u8], n: usize) -> Result<Vec<f32>> {
    let nb = n / QK;
    let mut out = vec![0.0f32; n];
    let mut c = Cur::new(raw);
    for blk in 0..nb {
        let d = f16_to_f32(c.u16()?);
        let m = f16_to_f32(c.u16()?);
        let qs = c.take(QK / 2)?;
        for (j, &byte) in qs.iter().enumerate() {
            let x0 = (byte & 0x0F) as f32;
            let x1 = (byte >> 4) as f32;
            out[blk * QK + j] = d * x0 + m;
            out[blk * QK + j + QK / 2] = d * x1 + m;
        }
    }
    Ok(out)
}

// ── k-quant super-block dequantizers (G-K) ────────────────────────────────────
//
// These follow ggml's `dequantize_row_q{4,5,6}_K` byte-for-byte. Each super-block
// holds QK_K (256) weights with sub-block scales, which is what gives k-quants
// their better accuracy-per-bit than the legacy 32-wide formats. Layouts must
// match ggml exactly or every imported weight is garbage, so the indexing here
// is deliberately literal; the unit tests construct hand-laid-out super-blocks
// and assert the decoded values.

/// Unpack the 6-bit scale `d` and min `m` for sub-block `j` (0..8) from the
/// 12-byte packed `scales` of a Q4_K/Q5_K super-block (ggml `get_scale_min_k4`).
#[inline]
pub(crate) fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

fn dequant_q4_k(raw: &[u8], n: usize) -> Result<Vec<f32>> {
    let nb = n / QK_K;
    let mut out = vec![0.0f32; n];
    let mut c = Cur::new(raw);
    for blk in 0..nb {
        let d = f16_to_f32(c.u16()?);
        let dmin = f16_to_f32(c.u16()?);
        let scales = c.take(12)?;
        let qs = c.take(QK_K / 2)?; // 128 bytes
        let base = blk * QK_K;
        let (mut y, mut q_off, mut is) = (0usize, 0usize, 0usize);
        while y < QK_K {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (d1, mn1) = (d * sc1 as f32, dmin * m1 as f32);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let (d2, mn2) = (d * sc2 as f32, dmin * m2 as f32);
            for l in 0..32 {
                out[base + y + l] = d1 * (qs[q_off + l] & 0x0F) as f32 - mn1;
                out[base + y + 32 + l] = d2 * (qs[q_off + l] >> 4) as f32 - mn2;
            }
            y += 64;
            q_off += 32;
            is += 2;
        }
    }
    Ok(out)
}

fn dequant_q5_k(raw: &[u8], n: usize) -> Result<Vec<f32>> {
    let nb = n / QK_K;
    let mut out = vec![0.0f32; n];
    let mut c = Cur::new(raw);
    for blk in 0..nb {
        let d = f16_to_f32(c.u16()?);
        let dmin = f16_to_f32(c.u16()?);
        let scales = c.take(12)?;
        let qh = c.take(QK_K / 8)?; // 32 bytes of high bits
        let qs = c.take(QK_K / 2)?; // 128 bytes of low nibbles
        let base = blk * QK_K;
        let (mut y, mut q_off, mut is) = (0usize, 0usize, 0usize);
        let (mut u1, mut u2) = (1u8, 2u8);
        while y < QK_K {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (d1, mn1) = (d * sc1 as f32, dmin * m1 as f32);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let (d2, mn2) = (d * sc2 as f32, dmin * m2 as f32);
            for l in 0..32 {
                let hi1 = if qh[l] & u1 != 0 { 16 } else { 0 };
                let hi2 = if qh[l] & u2 != 0 { 16 } else { 0 };
                out[base + y + l] = d1 * ((qs[q_off + l] & 0x0F) as i32 + hi1) as f32 - mn1;
                out[base + y + 32 + l] = d2 * ((qs[q_off + l] >> 4) as i32 + hi2) as f32 - mn2;
            }
            y += 64;
            q_off += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    Ok(out)
}

fn dequant_q6_k(raw: &[u8], n: usize) -> Result<Vec<f32>> {
    let nb = n / QK_K;
    let mut out = vec![0.0f32; n];
    let mut c = Cur::new(raw);
    for blk in 0..nb {
        let ql = c.take(QK_K / 2)?; // 128 bytes, lower 4 bits
        let qh = c.take(QK_K / 4)?; // 64 bytes, upper 2 bits
        let scales = c.take(QK_K / 16)?; // 16 signed scales
        let d = f16_to_f32(c.u16()?);
        let base = blk * QK_K;
        // Two 128-wide halves; ql/qh/scales advance by 64/32/8 per half.
        for half in 0..2 {
            let (qlo, qho, sco, oo) = (half * 64, half * 32, half * 8, base + half * 128);
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[qlo + l] & 0x0F) as i32 | ((qh[qho + l] & 3) as i32) << 4) - 32;
                let q2 = ((ql[qlo + l + 32] & 0x0F) as i32
                    | (((qh[qho + l] >> 2) & 3) as i32) << 4)
                    - 32;
                let q3 = ((ql[qlo + l] >> 4) as i32 | (((qh[qho + l] >> 4) & 3) as i32) << 4) - 32;
                let q4 =
                    ((ql[qlo + l + 32] >> 4) as i32 | (((qh[qho + l] >> 6) & 3) as i32) << 4) - 32;
                let sc = |k: usize| (scales[sco + k] as i8) as f32;
                out[oo + l] = d * sc(is) * q1 as f32;
                out[oo + l + 32] = d * sc(is + 2) * q2 as f32;
                out[oo + l + 64] = d * sc(is + 4) * q3 as f32;
                out[oo + l + 96] = d * sc(is + 6) * q4 as f32;
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── A minimal in-memory GGUF writer, just enough to exercise the reader ───

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
    fn f32_to_f16(x: f32) -> u16 {
        // Round-to-nearest-even is unnecessary for the small exact values used
        // in tests; a truncating conversion suffices and stays exact for them.
        let bits = x.to_bits();
        let sign = ((bits >> 31) & 1) as u16;
        let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
        let mant = (bits >> 13) & 0x3FF;
        if x == 0.0 {
            return sign << 15;
        }
        (sign << 15) | ((exp as u16) << 10) | mant as u16
    }

    /// Build a GGUF with one metadata string, one F32, one Q8_0, and one Q4_0
    /// tensor. Returns the bytes plus the exact f32 values each tensor encodes.
    fn synth_gguf() -> (Vec<u8>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut header = Vec::new();
        put_u32(&mut header, GGUF_MAGIC);
        put_u32(&mut header, 3); // version
        put_u64(&mut header, 3); // tensor_count
        put_u64(&mut header, 1); // kv_count

        // metadata: general.architecture = "ferrum-test"
        put_str(&mut header, "general.architecture");
        put_u32(&mut header, VT_STRING);
        put_str(&mut header, "ferrum-test");

        // Tensor data we will encode.
        let f32_vals: Vec<f32> = (0..QK as i32).map(|i| (i - 16) as f32 * 0.25).collect();
        // Q8_0: exact multiples of d.
        let q8_d = 0.5f32;
        let q8_vals: Vec<f32> = (0..QK as i32).map(|i| q8_d * (i - 16) as f32).collect();
        // Q4_0: d * (nibble-8), nibble 0..15 → value in [-8,7]·d.
        let q4_d = 0.25f32;
        let q4_vals: Vec<f32> = (0..QK as i32)
            .map(|i| q4_d * ((i % 15) - 8) as f32)
            .collect();

        let mut data = Vec::new();
        // F32 tensor.
        let off_f32 = data.len() as u64;
        for &v in &f32_vals {
            data.extend_from_slice(&v.to_le_bytes());
        }
        // Q8_0 tensor: f16 d, then 32 i8.
        let off_q8 = data.len() as u64;
        data.extend_from_slice(&f32_to_f16(q8_d).to_le_bytes());
        for i in 0..QK as i32 {
            data.push(((i - 16) as i8) as u8);
        }
        // Q4_0 tensor: f16 d, then 16 packed bytes (low→j, high→j+16).
        let off_q4 = data.len() as u64;
        data.extend_from_slice(&f32_to_f16(q4_d).to_le_bytes());
        for j in 0..(QK / 2) {
            // Stored nibble = decoded_value + 8; here decoded_value = (idx%15) - 8,
            // so the nibble is simply idx % 15.
            let lo = (j as i32 % 15) as u8 & 0x0F;
            let hi = ((j + QK / 2) as i32 % 15) as u8 & 0x0F;
            data.push((hi << 4) | lo);
        }

        // Tensor directory.
        let mut dir = Vec::new();
        for (name, ty, off) in [
            ("t_f32", GGML_F32, off_f32),
            ("t_q8", GGML_Q8_0, off_q8),
            ("t_q4", GGML_Q4_0, off_q4),
        ] {
            put_str(&mut dir, name);
            put_u32(&mut dir, 1); // n_dims
            put_u64(&mut dir, QK as u64); // dims[0]
            put_u32(&mut dir, ty);
            put_u64(&mut dir, off);
        }

        let mut bytes = header;
        bytes.extend_from_slice(&dir);
        // Pad to default alignment (32) before the data section.
        let pad = align_up(bytes.len(), DEFAULT_ALIGNMENT as usize) - bytes.len();
        bytes.extend(std::iter::repeat_n(0u8, pad));
        bytes.extend_from_slice(&data);

        // Recompute q4 expected values to match the writer's nibble layout.
        let mut q4_expected = vec![0.0f32; QK];
        for j in 0..(QK / 2) {
            q4_expected[j] = q4_d * ((j as i32 % 15) - 8) as f32;
            q4_expected[j + QK / 2] = q4_d * (((j + QK / 2) as i32 % 15) - 8) as f32;
        }
        let _ = (&q8_vals, &q4_vals);
        (bytes, f32_vals, q8_vals, q4_expected)
    }

    #[test]
    fn parses_header_metadata_and_directory() {
        let (bytes, _, _, _) = synth_gguf();
        let g = Gguf::parse(bytes).unwrap();
        assert_eq!(g.version, 3);
        assert_eq!(g.architecture(), Some("ferrum-test"));
        assert_eq!(g.tensors.len(), 3);
        assert!(g.tensor("t_f32").is_some());
        assert_eq!(g.tensor("t_q8").unwrap().ggml_type, GGML_Q8_0);
        assert_eq!(g.tensor("t_q4").unwrap().num_elements(), QK);
    }

    // A hostile dims table must degrade to a clean error, never an overflow
    // panic (debug builds) or a wrapped size (release builds).
    #[test]
    fn hostile_dims_error_instead_of_panicking() {
        let t = TensorInfo {
            name: "x".into(),
            dims: vec![u64::MAX, 8],
            ggml_type: GGML_F32,
            offset: 0,
        };
        assert_eq!(t.num_elements(), usize::MAX); // saturated, not wrapped
        assert!(type_nbytes(GGML_F32, t.num_elements()).is_err());
        assert!(type_nbytes(GGML_F16, t.num_elements()).is_err());
    }

    #[test]
    fn dequantizes_f32_q8_and_q4() {
        let (bytes, f32_vals, q8_vals, q4_vals) = synth_gguf();
        let g = Gguf::parse(bytes).unwrap();

        let f = g.dequantize(g.tensor("t_f32").unwrap()).unwrap();
        assert_eq!(f, f32_vals);

        let q8 = g.dequantize(g.tensor("t_q8").unwrap()).unwrap();
        for (a, b) in q8.iter().zip(&q8_vals) {
            assert!((a - b).abs() < 1e-4, "q8 {a} vs {b}");
        }

        let q4 = g.dequantize(g.tensor("t_q4").unwrap()).unwrap();
        for (a, b) in q4.iter().zip(&q4_vals) {
            assert!((a - b).abs() < 1e-4, "q4 {a} vs {b}");
        }
    }

    #[test]
    fn imports_tensor_as_int4_qweight() {
        let (bytes, _, _, _) = synth_gguf();
        let g = Gguf::parse(bytes).unwrap();
        // 32 elements as a [4, 8] matrix → int4 QWeight.
        let qw = g.quantized_weight("t_q8", 4, QKind::Int4).unwrap();
        assert_eq!((qw.rows, qw.cols), (4, 8));
        assert_eq!(qw.kind, QKind::Int4);
        // Round-trips within the int4 step of each row.
        let back = qw.to_f32();
        let direct = g.dequantize(g.tensor("t_q8").unwrap()).unwrap();
        let mae: f32 = back
            .iter()
            .zip(&direct)
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / 32.0;
        assert!(mae < 0.3, "int4 import mae {mae}");
    }

    #[test]
    fn f16_roundtrip_of_simple_values() {
        for &v in &[0.0f32, 1.0, -1.0, 0.5, -2.5, 16.0] {
            let h = f32_to_f16(v);
            assert!((f16_to_f32(h) - v).abs() < 1e-3, "f16 {v}");
        }
    }

    #[test]
    fn rejects_bad_magic_and_version() {
        assert!(Gguf::parse(vec![0, 1, 2, 3, 4, 5, 6, 7]).is_err());
        let (mut bytes, _, _, _) = synth_gguf();
        bytes[4] = 99; // version byte
        assert!(Gguf::parse(bytes).is_err());
    }

    #[test]
    fn unsupported_architecture_is_rejected_clearly() {
        let (bytes, _, _, _) = synth_gguf();
        let g = Gguf::parse(bytes).unwrap();
        // "ferrum-test" is neither llama nor qwen2.
        match g.load_llama(QKind::Int8) {
            Err(InferError::Format(_)) => {}
            _ => panic!("expected a Format error for an unsupported architecture"),
        }
    }

    #[test]
    fn transpose_2d_is_correct() {
        // [[1,2,3],[4,5,6]] (2×3) → [[1,4],[2,5],[3,6]] (3×2)
        let m = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(transpose_2d(&m, 2, 3), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    // ── Synthetic llama-architecture GGUF (end-to-end import) ─────────────────

    fn put_f32(o: &mut Vec<u8>, v: f32) {
        o.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    fn kv_u32(o: &mut Vec<u8>, key: &str, v: u32) {
        put_str(o, key);
        put_u32(o, VT_U32);
        put_u32(o, v);
    }
    fn kv_f32(o: &mut Vec<u8>, key: &str, v: f32) {
        put_str(o, key);
        put_u32(o, VT_F32);
        put_f32(o, v);
    }
    fn kv_str(o: &mut Vec<u8>, key: &str, v: &str) {
        put_str(o, key);
        put_u32(o, VT_STRING);
        put_str(o, v);
    }

    struct TBuf {
        infos: Vec<u8>,
        data: Vec<u8>,
        count: u32,
    }
    impl TBuf {
        fn new() -> Self {
            Self {
                infos: Vec::new(),
                data: Vec::new(),
                count: 0,
            }
        }
        /// Append an F32 tensor with GGUF dims `ne` (fastest axis first) and
        /// row-major values.
        fn add(&mut self, name: &str, ne: &[u64], vals: &[f32]) {
            let off = self.data.len() as u64;
            for &v in vals {
                put_f32(&mut self.data, v);
            }
            put_str(&mut self.infos, name);
            put_u32(&mut self.infos, ne.len() as u32);
            for &d in ne {
                put_u64(&mut self.infos, d);
            }
            put_u32(&mut self.infos, GGML_F32);
            put_u64(&mut self.infos, off);
            self.count += 1;
        }
    }

    /// A tiny but complete llama/qwen2-arch GGUF (2 layers, MHA), f32 weights.
    /// `with_output` controls whether an explicit LM head is written (else the
    /// loader ties it to the embedding); `with_bias` adds Qwen2-style q/k/v
    /// attention biases. `n_meta` must equal the metadata KV count written.
    fn synth_llama_cfg(arch: &str, with_output: bool, with_bias: bool) -> Vec<u8> {
        let (vocab, dim, n_heads, n_layers, ffn) = (10usize, 8usize, 2usize, 2usize, 16usize);
        let head_dim = dim / n_heads;
        let qd = n_heads * head_dim; // = dim here
        let mut seed = 12345u64;
        let mut rnd = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    (seed >> 40) as f32 / (1u64 << 23) as f32 - 1.0
                })
                .collect()
        };

        let mut t = TBuf::new();
        t.add(
            "token_embd.weight",
            &[dim as u64, vocab as u64],
            &rnd(vocab * dim),
        );
        for i in 0..n_layers {
            let p = format!("blk.{i}");
            t.add(
                &format!("{p}.attn_norm.weight"),
                &[dim as u64],
                &vec![1.0; dim],
            );
            t.add(
                &format!("{p}.attn_q.weight"),
                &[dim as u64, qd as u64],
                &rnd(dim * qd),
            );
            t.add(
                &format!("{p}.attn_k.weight"),
                &[dim as u64, qd as u64],
                &rnd(dim * qd),
            );
            t.add(
                &format!("{p}.attn_v.weight"),
                &[dim as u64, qd as u64],
                &rnd(dim * qd),
            );
            if with_bias {
                t.add(&format!("{p}.attn_q.bias"), &[qd as u64], &rnd(qd));
                t.add(&format!("{p}.attn_k.bias"), &[qd as u64], &rnd(qd));
                t.add(&format!("{p}.attn_v.bias"), &[qd as u64], &rnd(qd));
            }
            t.add(
                &format!("{p}.attn_output.weight"),
                &[qd as u64, dim as u64],
                &rnd(qd * dim),
            );
            t.add(
                &format!("{p}.ffn_norm.weight"),
                &[dim as u64],
                &vec![1.0; dim],
            );
            t.add(
                &format!("{p}.ffn_gate.weight"),
                &[dim as u64, ffn as u64],
                &rnd(dim * ffn),
            );
            t.add(
                &format!("{p}.ffn_up.weight"),
                &[dim as u64, ffn as u64],
                &rnd(dim * ffn),
            );
            t.add(
                &format!("{p}.ffn_down.weight"),
                &[ffn as u64, dim as u64],
                &rnd(ffn * dim),
            );
        }
        t.add("output_norm.weight", &[dim as u64], &vec![1.0; dim]);
        if with_output {
            t.add(
                "output.weight",
                &[dim as u64, vocab as u64],
                &rnd(dim * vocab),
            );
        }

        let mut meta = Vec::new();
        kv_str(&mut meta, "general.architecture", arch);
        kv_u32(&mut meta, &format!("{arch}.embedding_length"), dim as u32);
        kv_u32(&mut meta, &format!("{arch}.block_count"), n_layers as u32);
        kv_u32(
            &mut meta,
            &format!("{arch}.attention.head_count"),
            n_heads as u32,
        );
        kv_u32(
            &mut meta,
            &format!("{arch}.attention.head_count_kv"),
            n_heads as u32,
        );
        kv_u32(
            &mut meta,
            &format!("{arch}.feed_forward_length"),
            ffn as u32,
        );
        kv_u32(&mut meta, &format!("{arch}.context_length"), 32);
        kv_f32(
            &mut meta,
            &format!("{arch}.attention.layer_norm_rms_epsilon"),
            1e-5,
        );
        kv_f32(&mut meta, &format!("{arch}.rope.freq_base"), 10000.0);
        let n_meta = 9u64;

        let mut bytes = Vec::new();
        put_u32(&mut bytes, GGUF_MAGIC);
        put_u32(&mut bytes, 3);
        put_u64(&mut bytes, t.count as u64);
        put_u64(&mut bytes, n_meta);
        bytes.extend_from_slice(&meta);
        bytes.extend_from_slice(&t.infos);
        let pad = align_up(bytes.len(), DEFAULT_ALIGNMENT as usize) - bytes.len();
        bytes.extend(std::iter::repeat_n(0u8, pad));
        bytes.extend_from_slice(&t.data);
        bytes
    }

    fn synth_llama() -> Vec<u8> {
        synth_llama_cfg("llama", true, false)
    }

    #[test]
    fn loads_synthetic_llama_and_runs() {
        let g = Gguf::parse(synth_llama()).unwrap();
        assert_eq!(g.architecture(), Some("llama"));
        let model = g.load_llama(QKind::Int8).unwrap();
        assert_eq!(model.cfg.vocab_size, 10);
        assert_eq!(model.cfg.model_dim, 8);
        assert_eq!(model.blocks.len(), 2);

        // Full forward produces finite [seq, vocab] logits.
        let tokens = [1usize, 4, 2, 7, 0];
        let logits = model.forward_tokens(&tokens).unwrap();
        assert_eq!(logits.shape, vec![tokens.len(), model.cfg.vocab_size]);
        assert!(logits.data.iter().all(|v| v.is_finite()));

        // The imported model's KV-cached decode matches its own full forward.
        let mut cache = crate::llm::LlamaCache::new(model.blocks.len());
        let (_, vocab) = logits.matrix_dims().unwrap();
        for (ti, &tok) in tokens.iter().enumerate() {
            let step = model.forward_one(tok, &mut cache).unwrap();
            for (v, &s) in step.iter().enumerate() {
                assert!(
                    (s - logits.data[ti * vocab + v]).abs() < 1e-2,
                    "imported model cached vs full mismatch at {ti},{v}"
                );
            }
        }

        // Generation runs end to end.
        let out = model
            .generate(
                &[1, 2],
                6,
                &crate::llm::SamplingParams::with_temperature(0.8),
                None,
                &mut crate::rng::Rng::new(1),
            )
            .unwrap();
        assert_eq!(out.len(), 6);
        assert!(out.iter().all(|&t| t < model.cfg.vocab_size));
    }

    #[test]
    fn linear_from_applies_transpose_so_y_equals_w_times_x() {
        // A single GGUF weight [n_out, n_in]; the imported Ferrum Linear must
        // compute y = W·x (so the transpose is right). int8 keeps error tiny.
        let (n_in, n_out) = (4usize, 3usize);
        // GGUF row-major [n_out, n_in].
        let w_gguf: Vec<f32> = (0..n_out * n_in).map(|i| (i as f32 * 0.1) - 0.5).collect();
        let mut t = TBuf::new();
        t.add("w", &[n_in as u64, n_out as u64], &w_gguf);
        let mut meta = Vec::new();
        kv_str(&mut meta, "general.architecture", "llama");
        let mut bytes = Vec::new();
        put_u32(&mut bytes, GGUF_MAGIC);
        put_u32(&mut bytes, 3);
        put_u64(&mut bytes, 1);
        put_u64(&mut bytes, 1);
        bytes.extend_from_slice(&meta);
        bytes.extend_from_slice(&t.infos);
        let pad = align_up(bytes.len(), DEFAULT_ALIGNMENT as usize) - bytes.len();
        bytes.extend(std::iter::repeat_n(0u8, pad));
        bytes.extend_from_slice(&t.data);

        let g = Gguf::parse(bytes).unwrap();
        let lin = g.linear_from("w", None, Some(QKind::Int8)).unwrap();
        use crate::layer::Layer;
        let x = vec![0.3f32, -0.7, 0.2, 0.9];
        let y = lin
            .forward(&crate::tensor::Tensor::matrix(1, n_in, x.clone()).unwrap())
            .unwrap();
        // Reference y[o] = Σ_i w_gguf[o*n_in + i] · x[i].
        for o in 0..n_out {
            let want: f32 = (0..n_in).map(|i| w_gguf[o * n_in + i] * x[i]).sum();
            assert!(
                (y.data[o] - want).abs() < 0.05,
                "y[{o}]={} want {want}",
                y.data[o]
            );
        }
    }

    #[test]
    fn loads_tied_embedding_llama() {
        // No output.weight → the LM head is tied to the token embedding.
        let g = Gguf::parse(synth_llama_cfg("llama", false, false)).unwrap();
        assert!(g.tensor("output.weight").is_none());
        let model = g.load_llama(QKind::Int8).unwrap();
        let logits = model.forward_tokens(&[1, 2, 3]).unwrap();
        assert_eq!(logits.shape, vec![3, model.cfg.vocab_size]);
        assert!(logits.data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn loads_qwen2_with_attention_biases() {
        let g = Gguf::parse(synth_llama_cfg("qwen2", true, true)).unwrap();
        assert_eq!(g.architecture(), Some("qwen2"));
        let model = g.load_llama(QKind::Int4).unwrap();
        let logits = model.forward_tokens(&[0, 5, 9]).unwrap();
        assert!(logits.data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn load_llama_errors_on_missing_metadata_and_tensors() {
        // arch=llama but no shape metadata and no tensors.
        let mut bytes = Vec::new();
        put_u32(&mut bytes, GGUF_MAGIC);
        put_u32(&mut bytes, 3);
        put_u64(&mut bytes, 0); // 0 tensors
        put_u64(&mut bytes, 1); // 1 metadata kv
        kv_str(&mut bytes, "general.architecture", "llama");
        // Pad to the alignment boundary so the (empty) data section starts in-bounds.
        let pad = align_up(bytes.len(), DEFAULT_ALIGNMENT as usize) - bytes.len();
        bytes.extend(std::iter::repeat_n(0u8, pad));
        let g = Gguf::parse(bytes).unwrap();
        assert!(
            g.load_llama(QKind::Int8).is_err(),
            "missing embedding_length must error"
        );
    }

    // ── Block dequantizers Q8_1 / Q4_1 ────────────────────────────────────────

    #[test]
    fn dequant_q8_1_block() {
        let d = 0.5f32;
        let mut raw = Vec::new();
        raw.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        raw.extend_from_slice(&f32_to_f16(0.0).to_le_bytes()); // s (unused)
        for i in 0..QK as i32 {
            raw.push(((i - 16) as i8) as u8);
        }
        let out = dequant_q8_1(&raw, QK).unwrap();
        for (i, &o) in out.iter().enumerate() {
            assert!((o - d * (i as i32 - 16) as f32).abs() < 1e-3);
        }
    }

    #[test]
    fn dequant_q4_1_block() {
        let (d, m) = (0.25f32, 1.0f32);
        let mut raw = Vec::new();
        raw.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        raw.extend_from_slice(&f32_to_f16(m).to_le_bytes());
        // nibble j → lo, nibble j+16 → hi.
        let los: Vec<u8> = (0..16).map(|j| (j % 15) as u8).collect();
        let his: Vec<u8> = (0..16).map(|j| ((j + 3) % 15) as u8).collect();
        for j in 0..16 {
            raw.push((his[j] << 4) | los[j]);
        }
        let out = dequant_q4_1(&raw, QK).unwrap();
        for j in 0..16 {
            assert!((out[j] - (d * los[j] as f32 + m)).abs() < 1e-3);
            assert!((out[j + 16] - (d * his[j] as f32 + m)).abs() < 1e-3);
        }
    }

    #[test]
    fn quantized_tensor_count_must_be_block_multiple() {
        // 30 is not a multiple of 32 → sizing rejects it.
        assert!(type_nbytes(GGML_Q8_0, 30).is_err());
        assert!(type_nbytes(GGML_F32, 30).is_ok());
        assert!(type_nbytes(999, 32).is_err());
    }

    // ── k-quant super-blocks (G-K) ────────────────────────────────────────────

    #[test]
    fn k_quant_block_sizes() {
        assert_eq!(type_nbytes(GGML_Q4_K, QK_K).unwrap(), Q4_K_BLOCK); // 144
        assert_eq!(type_nbytes(GGML_Q5_K, QK_K).unwrap(), Q5_K_BLOCK); // 176
        assert_eq!(type_nbytes(GGML_Q6_K, QK_K).unwrap(), Q6_K_BLOCK); // 210
        assert!(type_nbytes(GGML_Q4_K, 200).is_err()); // not a multiple of 256
    }

    #[test]
    fn dequant_q6_k_constant_block() {
        // ql nibbles = 8, qh 2-bit fields = 2 ⇒ q = (8|2<<4) − 32 = 8.
        // scales = 2, d = 0.5 ⇒ every value = 0.5·2·8 = 8.0.
        let mut raw = vec![0x88u8; QK_K / 2];
        raw.extend_from_slice(&[0xAAu8; QK_K / 4]);
        raw.extend(std::iter::repeat_n(2i8 as u8, QK_K / 16));
        raw.extend_from_slice(&f32_to_f16(0.5).to_le_bytes());
        let out = dequant_q6_k(&raw, QK_K).unwrap();
        assert_eq!(out.len(), QK_K);
        for &v in &out {
            assert!((v - 8.0).abs() < 1e-3, "q6_k got {v}");
        }
    }

    /// 12-byte packed scales that decode (via get_scale_min_k4) to d = 2, m = 0
    /// for all 8 sub-blocks.
    const K_SCALES_D2_M0: [u8; 12] = [2, 2, 2, 2, 0, 0, 0, 0, 2, 2, 2, 2];

    #[test]
    fn get_scale_min_k4_unpacks_all_subblocks() {
        for j in 0..8 {
            let (d, m) = get_scale_min_k4(j, &K_SCALES_D2_M0);
            assert_eq!((d, m), (2, 0), "sub-block {j}");
        }
    }

    #[test]
    fn dequant_q4_k_constant_block() {
        // d·sc = 0.5·2 = 1.0, min = 0, nibble = 3 ⇒ every value = 3.0.
        let mut raw = Vec::new();
        raw.extend_from_slice(&f32_to_f16(0.5).to_le_bytes()); // d
        raw.extend_from_slice(&f32_to_f16(0.0).to_le_bytes()); // dmin
        raw.extend_from_slice(&K_SCALES_D2_M0);
        raw.extend_from_slice(&[0x33u8; QK_K / 2]); // qs: both nibbles = 3
        let out = dequant_q4_k(&raw, QK_K).unwrap();
        for &v in &out {
            assert!((v - 3.0).abs() < 1e-3, "q4_k got {v}");
        }
    }

    #[test]
    fn dequant_q5_k_low_and_high_bits() {
        let build = |qh: u8| {
            let mut raw = Vec::new();
            raw.extend_from_slice(&f32_to_f16(0.5).to_le_bytes());
            raw.extend_from_slice(&f32_to_f16(0.0).to_le_bytes());
            raw.extend_from_slice(&K_SCALES_D2_M0);
            raw.extend_from_slice(&[qh; QK_K / 8]); // 32 bytes qh
            raw.extend_from_slice(&[0x33u8; QK_K / 2]); // qs nibble = 3
            raw
        };
        // No high bit: value = 1.0·3 = 3.0.
        for &v in &dequant_q5_k(&build(0x00), QK_K).unwrap() {
            assert!((v - 3.0).abs() < 1e-3, "q5_k low {v}");
        }
        // Every high bit set: value = 1.0·(3 + 16) = 19.0.
        for &v in &dequant_q5_k(&build(0xFF), QK_K).unwrap() {
            assert!((v - 19.0).abs() < 1e-3, "q5_k high {v}");
        }
    }

    // ── Streamed reader (G-mmap) & f32 import (G-Q) ───────────────────────────

    #[test]
    fn streamed_open_matches_in_memory_parse() {
        let bytes = synth_llama();
        let path =
            std::env::temp_dir().join(format!("ferrum_gguf_stream_{}.gguf", std::process::id()));
        std::fs::write(&path, &bytes).unwrap();

        let mem = Gguf::parse(bytes).unwrap();
        let streamed = Gguf::open(path.to_str().unwrap()).unwrap();
        assert_eq!(streamed.tensors.len(), mem.tensors.len());
        assert_eq!(streamed.architecture(), Some("llama"));

        // A tensor dequantizes identically whether held in memory or streamed.
        let tm = mem.tensor("token_embd.weight").unwrap().clone();
        let ts = streamed.tensor("token_embd.weight").unwrap().clone();
        assert_eq!(
            mem.dequantize(&tm).unwrap(),
            streamed.dequantize(&ts).unwrap()
        );

        // And a full model loads through the streamed path.
        let model = streamed.load_llama(QKind::Int8).unwrap();
        assert_eq!(model.cfg.vocab_size, 10);
        assert!(model
            .forward_tokens(&[1, 2, 3])
            .unwrap()
            .data
            .iter()
            .all(|v| v.is_finite()));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_llama_f32_precision_skips_requantization() {
        // prec = None keeps weights f32 (no second quantization on import).
        let g = Gguf::parse(synth_llama()).unwrap();
        let model = g.load_llama_prec(None).unwrap();
        let logits = model.forward_tokens(&[1, 2, 3]).unwrap();
        assert_eq!(logits.shape, vec![3, model.cfg.vocab_size]);
        assert!(logits.data.iter().all(|v| v.is_finite()));
    }

    // ── f16 edge cases ────────────────────────────────────────────────────────

    #[test]
    fn f16_special_values() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x8000), 0.0); // negative zero
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert!(f16_to_f32(0x7C00).is_infinite() && f16_to_f32(0x7C00) > 0.0);
        assert!(f16_to_f32(0xFC00).is_infinite() && f16_to_f32(0xFC00) < 0.0);
        assert!(f16_to_f32(0x7E00).is_nan());
        // Smallest positive subnormal = 2^-24.
        let sub = f16_to_f32(0x0001);
        assert!(sub > 0.0 && (sub - 2f32.powi(-24)).abs() < 1e-12);
    }

    // ── MetaValue accessors ───────────────────────────────────────────────────

    #[test]
    fn metavalue_accessors() {
        assert_eq!(MetaValue::U8(5).as_usize(), Some(5));
        assert_eq!(MetaValue::U64(7).as_usize(), Some(7));
        assert_eq!(MetaValue::I32(9).as_usize(), Some(9));
        assert_eq!(MetaValue::I32(-1).as_usize(), None);
        assert_eq!(MetaValue::I8(-3).as_usize(), None);
        assert_eq!(MetaValue::F32(2.5).as_f32(), Some(2.5));
        assert_eq!(MetaValue::U16(4).as_f32(), Some(4.0));
        assert_eq!(MetaValue::String("hi".into()).as_str(), Some("hi"));
        assert_eq!(MetaValue::Bool(true).as_usize(), None);
        assert_eq!(MetaValue::Array(vec![MetaValue::U8(1)]).as_f32(), None);
    }

    #[test]
    fn parse_rejects_truncation() {
        let mut bytes = synth_llama();
        bytes.truncate(bytes.len() - 20); // chop tensor data
                                          // Parse succeeds (directory intact) but dequantizing the last tensor
                                          // must fail because its data ran off the end.
        let g = Gguf::parse(bytes);
        if let Ok(g) = g {
            let last = g.tensors.last().unwrap().clone();
            assert!(g.dequantize(&last).is_err());
        }
    }

    #[test]
    fn parse_rejects_nested_arrays_and_unknown_types() {
        // metadata kv: key, value_type=ARRAY, elem_type=ARRAY → rejected.
        let mut bytes = Vec::new();
        put_u32(&mut bytes, GGUF_MAGIC);
        put_u32(&mut bytes, 3);
        put_u64(&mut bytes, 0);
        put_u64(&mut bytes, 1);
        put_str(&mut bytes, "k");
        put_u32(&mut bytes, VT_ARRAY);
        put_u32(&mut bytes, VT_ARRAY); // nested
        put_u64(&mut bytes, 1);
        assert!(Gguf::parse(bytes).is_err());
    }
}
