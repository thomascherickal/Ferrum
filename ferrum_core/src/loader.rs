//! FINF binary format v4/v5: weights + normalizer + ModelMetadata JSON.
//!
//! Layout (all integers little-endian):
//!   4 bytes  b"FINF"
//!   u32      version = 4 or 5
//!   u32      norm_len;    [bytes] normalizer string  (empty string for SLM)
//!   u32      meta_len;    [bytes] ModelMetadata JSON
//!   u32      num_layers
//!   per layer: u8 tag, then layer bytes
//!
//! Layer Tags:
//!   0 = Linear
//!   1 = ActivationLayer
//!   2 = Embedding
//!   3 = LayerNorm
//!   4 = TransformerBlock
//!   5 = Flatten (v5 only, no payload)
//!
//! v5 is a tag-compatible extension of v4: each weight vector is prefixed by a
//! one-byte encoding marker —
//!   0 = raw f32,
//!   1 = int8 symmetric per-tensor (f32 scale, then one i8 per value),
//!   2 = int8 symmetric per-channel (one f32 scale per input row),
//!   3 = int4 symmetric per-tensor (f32 scale, then packed nibbles),
//!   4 = int4 symmetric per-channel (one f32 scale per input row) — the default
//!       for `to_bytes_quantized_int4`.
//! value = q × scale. `to_bytes` writes v4 whenever the model is expressible in
//! it; `to_bytes_quantized` writes v5 with large weight matrices stored int8
//! (≈4× smaller), and `to_bytes_quantized_int4` stores them int4 (≈8× smaller).
//!
//! Crucially, the **loader keeps quantized matrices packed in memory** (Opt#1):
//! a Linear / TransformerBlock projection read from an int8/int4 file becomes a
//! [`crate::quant::QWeight`], never an expanded f32 buffer, so a large model
//! both fits in RAM and streams fewer bytes per generated token. Biases,
//! LayerNorm parameters and embeddings stay f32 in memory (the embedding table
//! is dequantized on load).
use crate::activation::Activation;
use crate::error::{InferError, Result};
use crate::gguf::{transpose_2d, Gguf, MetaValue, GGML_F32, GGML_Q4_0, GGML_Q8_0};
use crate::gguf_write::{encode_tensor, GgufBuilder};
use crate::layer::Layer;
use crate::layer::{ActivationLayer, Embedding, Flatten, LayerNorm, Linear, TransformerBlock};
use crate::meta::TaskType;
use crate::meta::{ModelMetadata, Normalizer};
use crate::model::Sequential;
use crate::quant::{
    int8_scale, int8_scales_per_channel, q8_0_eligible, QKind, QWeight, QUANT_MIN_LEN,
};
use crate::tokenizer::{byte_to_unicode, ByteBpeTokenizer};
use std::collections::HashMap;

const MAGIC: &[u8; 4] = b"FINF";
const VERSION: u32 = 4;
const VERSION_QUANT: u32 = 5;
const TAG_LINEAR: u8 = 0;
const TAG_ACTIVATION: u8 = 1;
const TAG_EMBEDDING: u8 = 2;
const TAG_LAYERNORM: u8 = 3;
const TAG_TRANSFORMER_BLOCK: u8 = 4;
const TAG_FLATTEN: u8 = 5;

/// v5 weight-vector encoding markers.
const ENC_F32: u8 = 0;
const ENC_INT8: u8 = 1;
/// Per-channel int8 (§7): `u32` channel count, then one `f32` scale per channel,
/// then one `i8` per value. Value `i` dequantises as `i8 × scale[i / row_len]`
/// with `row_len = n / channels`. A tag-compatible extension of v5 — older v5
/// readers reject the unknown marker rather than misreading.
const ENC_INT8_PER_CHANNEL: u8 = 2;
/// Per-tensor int4 (§Opt#1): `f32` scale, then `rows·ceil(cols/2)` bytes of
/// packed nibbles in [`crate::quant::QWeight`]'s **split-half** layout (per row,
/// byte `b`'s low nibble = column `b`, high nibble = column `half + b`), so the
/// loader copies straight into a `QWeight` without re-quantizing. Symmetric,
/// levels −7..=7.
const ENC_INT4: u8 = 3;
/// Per-channel int4: `u32` channel count, one `f32` scale per channel, then the
/// packed nibbles. Channels are the matrix's input rows (`= in_features`).
const ENC_INT4_PER_CHANNEL: u8 = 4;

// ─────────────────────────────────────────────────────────────────────────────
// Reader helper
// ─────────────────────────────────────────────────────────────────────────────

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
    /// FINF v5: weight vectors carry a per-vector encoding marker.
    v5: bool,
}
impl<'a> Reader<'a> {
    fn new(b: &'a [u8]) -> Self {
        Self {
            bytes: b,
            pos: 0,
            v5: false,
        }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos + n;
        if end > self.bytes.len() {
            return Err(InferError::Format(format!(
                "EOF at +{}: need {n}",
                self.pos
            )));
        }
        let s = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn usize(&mut self) -> Result<usize> {
        Ok(self.u32()? as usize)
    }
    /// Read `n` f32 values. The byte length is bounds-checked against the
    /// remaining buffer BEFORE any allocation, so corrupt or malicious files
    /// with huge dimension fields fail fast instead of attempting a giant
    /// allocation.
    fn f32_vec(&mut self, n: usize) -> Result<Vec<f32>> {
        let byte_len = n
            .checked_mul(4)
            .ok_or_else(|| InferError::Format(format!("f32 vec length {n} overflows")))?;
        let raw = self.take(byte_len)?;
        Ok(raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect())
    }
    fn utf8(&mut self, n: usize) -> Result<&'a str> {
        std::str::from_utf8(self.take(n)?)
            .map_err(|_| InferError::Format("invalid UTF-8 in blob".into()))
    }
    /// Read one weight vector of `n` values. In v4 this is raw f32; in v5 a
    /// one-byte marker selects raw f32 or int8 symmetric (scale + i8 × n).
    fn weights(&mut self, n: usize) -> Result<Vec<f32>> {
        if !self.v5 {
            return self.f32_vec(n);
        }
        match self.u8()? {
            ENC_F32 => self.f32_vec(n),
            ENC_INT8 => {
                let scale = f32::from_le_bytes({
                    let b = self.take(4)?;
                    [b[0], b[1], b[2], b[3]]
                });
                let raw = self.take(n)?;
                Ok(raw.iter().map(|&b| (b as i8) as f32 * scale).collect())
            }
            ENC_INT8_PER_CHANNEL => {
                let channels = self.usize()?;
                if channels == 0 || !n.is_multiple_of(channels) {
                    return Err(InferError::Format(format!(
                        "per-channel weights: {channels} channels do not divide {n} values"
                    )));
                }
                let row = n / channels;
                let mut scales = Vec::with_capacity(channels);
                for _ in 0..channels {
                    let b = self.take(4)?;
                    scales.push(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
                }
                let raw = self.take(n)?;
                Ok(raw
                    .iter()
                    .enumerate()
                    .map(|(i, &b)| (b as i8) as f32 * scales[i / row])
                    .collect())
            }
            ENC_INT4 | ENC_INT4_PER_CHANNEL => Err(InferError::Format(
                "int4 weights are only valid for matrices (read via weights_q)".into(),
            )),
            m => Err(InferError::Format(format!(
                "bad weight encoding marker {m}"
            ))),
        }
    }

    /// Read a `[rows, cols]` weight matrix, **preserving quantization** when the
    /// file is int8/int4 so the weights stay packed in memory (Opt#1). Returns
    /// either an f32 buffer (v4, or an `ENC_F32` v5 tensor) or a [`QWeight`].
    ///
    /// The FINF per-channel convention for matrices is one scale per input row
    /// (`channels = in_features = rows`), which maps exactly onto `QWeight`'s
    /// per-row scales; a per-tensor encoding becomes a `QWeight` whose row scales
    /// are all equal. An unexpected `channels != rows` (never produced by this
    /// writer) is dequantized to f32 defensively.
    fn weights_q(&mut self, rows: usize, cols: usize) -> Result<LoadedWeights> {
        let n = mul_dims(rows, cols)?;
        if !self.v5 {
            return Ok(LoadedWeights::F32(self.f32_vec(n)?));
        }
        match self.u8()? {
            ENC_F32 => Ok(LoadedWeights::F32(self.f32_vec(n)?)),
            ENC_INT8 => {
                let scale = self.read_f32()?;
                let raw = self.take(n)?.to_vec();
                Ok(LoadedWeights::Quant(QWeight {
                    rows,
                    cols,
                    kind: QKind::Int8,
                    scales: vec![scale; rows],
                    q: raw,
                }))
            }
            ENC_INT8_PER_CHANNEL => {
                let channels = self.usize()?;
                if channels == 0 || n % channels != 0 {
                    return Err(InferError::Format(format!(
                        "per-channel int8: {channels} channels do not divide {n}"
                    )));
                }
                let scales = self.read_scales(channels)?;
                let raw = self.take(n)?;
                if channels == rows {
                    Ok(LoadedWeights::Quant(QWeight {
                        rows,
                        cols,
                        kind: QKind::Int8,
                        scales,
                        q: raw.to_vec(),
                    }))
                } else {
                    let row = n / channels;
                    Ok(LoadedWeights::F32(
                        raw.iter()
                            .enumerate()
                            .map(|(i, &b)| (b as i8) as f32 * scales[i / row])
                            .collect(),
                    ))
                }
            }
            ENC_INT4 => {
                let scale = self.read_f32()?;
                let packed = mul_dims(rows, cols.div_ceil(2))?;
                let raw = self.take(packed)?.to_vec();
                Ok(LoadedWeights::Quant(QWeight {
                    rows,
                    cols,
                    kind: QKind::Int4,
                    scales: vec![scale; rows],
                    q: raw,
                }))
            }
            ENC_INT4_PER_CHANNEL => {
                let channels = self.usize()?;
                let packed = mul_dims(rows, cols.div_ceil(2))?;
                if channels == rows {
                    let scales = self.read_scales(channels)?;
                    let raw = self.take(packed)?.to_vec();
                    Ok(LoadedWeights::Quant(QWeight {
                        rows,
                        cols,
                        kind: QKind::Int4,
                        scales,
                        q: raw,
                    }))
                } else {
                    // Defensive: not produced by this writer for matrices.
                    let scales = self.read_scales(channels.max(1))?;
                    let qw = QWeight {
                        rows: channels.max(1),
                        cols: n / channels.max(1),
                        kind: QKind::Int4,
                        scales,
                        q: self.take(packed)?.to_vec(),
                    };
                    Ok(LoadedWeights::F32(qw.to_f32()))
                }
            }
            m => Err(InferError::Format(format!(
                "bad weight encoding marker {m}"
            ))),
        }
    }

    fn read_f32(&mut self) -> Result<f32> {
        let b = self.take(4)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn read_scales(&mut self, n: usize) -> Result<Vec<f32>> {
        (0..n).map(|_| self.read_f32()).collect()
    }
}

/// A weight matrix as read from FINF: either full f32, or kept quantized.
enum LoadedWeights {
    F32(Vec<f32>),
    Quant(QWeight),
}

impl LoadedWeights {
    /// Materialize to f32 (used where the in-memory layer stays f32, e.g.
    /// embeddings).
    fn into_f32(self) -> Vec<f32> {
        match self {
            LoadedWeights::F32(v) => v,
            LoadedWeights::Quant(q) => q.to_f32(),
        }
    }
}

/// Build a `Linear` from a loaded weight matrix, keeping it quantized in memory
/// when the file was quantized (Opt#1) and falling back to f32 otherwise.
fn linear_from_loaded(
    in_f: usize,
    out_f: usize,
    w: LoadedWeights,
    bias: Vec<f32>,
) -> Result<Linear> {
    match w {
        LoadedWeights::Quant(qw) => Linear::quantized(in_f, out_f, qw, bias),
        LoadedWeights::F32(data) => Linear::new(in_f, out_f, data, bias),
    }
}

/// Multiply two dimension fields read from an untrusted file, rejecting
/// overflow (relevant on 32-bit/wasm targets where usize is 32 bits).
fn mul_dims(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| InferError::Format(format!("dimension product {a}×{b} overflows")))
}

// ─────────────────────────────────────────────────────────────────────────────
// Writer helpers
// ─────────────────────────────────────────────────────────────────────────────

fn push_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn push_usize(out: &mut Vec<u8>, v: usize) {
    push_u32(out, v as u32);
}
fn push_f32s(out: &mut Vec<u8>, data: &[f32]) {
    for &x in data {
        out.extend_from_slice(&x.to_le_bytes());
    }
}
fn push_str(out: &mut Vec<u8>, s: &str) {
    push_u32(out, s.len() as u32);
    out.extend_from_slice(s.as_bytes());
}
/// Encode `data` as one i8 per value against `scale` (the symmetric int8 grid).
fn push_int8(out: &mut Vec<u8>, data: &[f32], scale: f32) {
    if scale == 0.0 {
        out.extend(std::iter::repeat_n(0u8, data.len()));
    } else {
        for &v in data {
            out.push((v / scale).round().clamp(-127.0, 127.0) as i8 as u8);
        }
    }
}

/// On-disk weight precision selected at save time. In GGUF, `Int8` is Q8_0
/// and `Int4` is Q4_0 (matrices whose GGML row length is a multiple of 32).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QPrec {
    /// Full f32 (FINF v4, or v5 `ENC_F32`; GGUF F32).
    F32,
    /// int8: FINF per-channel int8, GGUF Q8_0.
    Int8,
    /// int4: FINF per-channel int4 (~8× smaller than f32), GGUF Q4_0.
    Int4,
}

/// Write one weight vector. v4: raw f32. v5: a marker byte then either raw f32,
/// per-tensor/-channel int8, or per-channel int4. `channels` is the weight
/// matrix's row count (`1` for biases / 1-D parameters). Small or non-finite
/// vectors stay f32 even when quantisation is requested.
///
/// int4 is always written per-row (`ENC_INT4_PER_CHANNEL`, `channels` = rows)
/// with [`QWeight`]'s exact packing, so the loader reconstructs the `QWeight`
/// byte-for-byte with no re-quantization.
fn push_weights(out: &mut Vec<u8>, data: &[f32], channels: usize, v5: bool, prec: QPrec) {
    if !v5 {
        push_f32s(out, data);
        return;
    }
    let finite = data.iter().all(|v| v.is_finite());
    if prec == QPrec::F32 || !(data.len() >= QUANT_MIN_LEN && finite) {
        out.push(ENC_F32);
        push_f32s(out, data);
        return;
    }
    match prec {
        QPrec::F32 => unreachable!(),
        QPrec::Int8 => {
            // Per-channel when the matrix splits into >1 even rows; else per-tensor.
            if channels > 1 && data.len().is_multiple_of(channels) {
                let row = data.len() / channels;
                let scales = int8_scales_per_channel(data, channels);
                out.push(ENC_INT8_PER_CHANNEL);
                push_u32(out, channels as u32);
                for &s in &scales {
                    out.extend_from_slice(&s.to_le_bytes());
                }
                for (c, chunk) in data.chunks(row).enumerate() {
                    push_int8(out, chunk, scales[c]);
                }
            } else {
                let scale = int8_scale(data);
                out.push(ENC_INT8);
                out.extend_from_slice(&scale.to_le_bytes());
                push_int8(out, data, scale);
            }
        }
        QPrec::Int4 => {
            let rows = if channels > 0 && data.len().is_multiple_of(channels) {
                channels
            } else {
                1
            };
            let cols = data.len() / rows;
            let qw = QWeight::from_f32(data, rows, cols, QKind::Int4);
            out.push(ENC_INT4_PER_CHANNEL);
            push_u32(out, rows as u32);
            for &s in &qw.scales {
                out.extend_from_slice(&s.to_le_bytes());
            }
            out.extend_from_slice(&qw.q);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Serialization (to_bytes)
// ─────────────────────────────────────────────────────────────────────────────

/// Serialize with full f32 weights. Writes FINF v4 unless the model contains
/// a layer only expressible in v5 (`Flatten`), in which case v5 is written.
pub fn to_bytes(model: &Sequential, norm: &Normalizer, meta: &ModelMetadata) -> Result<Vec<u8>> {
    let needs_v5 = model
        .layers()
        .iter()
        .any(|l| l.as_any().downcast_ref::<Flatten>().is_some());
    let version = if needs_v5 { VERSION_QUANT } else { VERSION };
    to_bytes_impl(model, norm, meta, version, QPrec::F32)
}

/// Serialize as FINF v5 with int8 post-training quantisation: every weight
/// vector of ≥ 64 values is stored as one i8 per value plus an f32 scale
/// (≈4× smaller). Biases and LayerNorm parameters stay f32. The loader
/// dequantises transparently on read.
pub fn to_bytes_quantized(
    model: &Sequential,
    norm: &Normalizer,
    meta: &ModelMetadata,
) -> Result<Vec<u8>> {
    to_bytes_impl(model, norm, meta, VERSION_QUANT, QPrec::Int8)
}

/// Serialize as FINF v5 with **int4** post-training quantisation (~8× smaller
/// than f32). Matrices are stored per-row at 4 bits/weight; biases and
/// LayerNorm parameters stay int8/f32. This is the recommended on-disk format
/// for large (≥1B) models, and the loader keeps the matrices packed in memory.
pub fn to_bytes_quantized_int4(
    model: &Sequential,
    norm: &Normalizer,
    meta: &ModelMetadata,
) -> Result<Vec<u8>> {
    to_bytes_impl(model, norm, meta, VERSION_QUANT, QPrec::Int4)
}

fn to_bytes_impl(
    model: &Sequential,
    norm: &Normalizer,
    meta: &ModelMetadata,
    version: u32,
    prec: QPrec,
) -> Result<Vec<u8>> {
    let v5 = version == VERSION_QUANT;
    // Matrices quantise per input-row (`channels = in_features`). 1-D parameters
    // (biases, LayerNorm) use `channels = 1` and are read back through the f32
    // path, which has no int4 decoder — so they never go below int8 (int4 is
    // demoted to int8 for them). Both stay f32 below QUANT_MIN_LEN anyway.
    let vec_prec = if prec == QPrec::Int4 {
        QPrec::Int8
    } else {
        prec
    };
    let push_mat = |out: &mut Vec<u8>, data: &[f32], channels: usize| {
        push_weights(out, data, channels, v5, prec)
    };
    let push_vec = |out: &mut Vec<u8>, data: &[f32]| push_weights(out, data, 1, v5, vec_prec);
    vprintln!(
        "[loader::to_bytes] Serializing FINF v{} model ({} layers, prec=int{})",
        version,
        model.len(),
        match prec {
            QPrec::F32 => 32,
            QPrec::Int8 => 8,
            QPrec::Int4 => 4,
        }
    );
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    push_u32(&mut out, version);

    // Normalizer (can be empty string for SLM tasks)
    push_str(&mut out, &norm.encode());

    // Metadata JSON
    push_str(&mut out, &meta.to_json());

    push_u32(&mut out, model.len() as u32);
    for (layer_idx, layer) in model.layers().iter().enumerate() {
        vprintln!(
            "[loader::to_bytes]   layer[{}]: {}",
            layer_idx,
            layer.name()
        );
        let any = layer.as_any();

        if let Some(lin) = any.downcast_ref::<Linear>() {
            out.push(TAG_LINEAR);
            push_usize(&mut out, lin.in_features());
            push_usize(&mut out, lin.out_features());
            // `weight_f32` dequantizes if the Linear is already quantized in
            // memory, so a load-quantized model re-serializes correctly.
            push_mat(&mut out, &lin.weight_f32(), lin.in_features());
            push_vec(&mut out, &lin.bias.data);
        } else if let Some(act) = any.downcast_ref::<ActivationLayer>() {
            out.push(TAG_ACTIVATION);
            out.push(act.kind.tag());
        } else if any.downcast_ref::<Flatten>().is_some() {
            out.push(TAG_FLATTEN);
        } else if let Some(emb) = any.downcast_ref::<Embedding>() {
            out.push(TAG_EMBEDDING);
            push_usize(&mut out, emb.vocab_size());
            push_usize(&mut out, emb.max_seq_len());
            push_usize(&mut out, emb.embedding_dim());
            push_mat(&mut out, &emb.token_weight.data, emb.vocab_size());
            push_mat(&mut out, &emb.pos_weight.data, emb.max_seq_len());
        } else if let Some(ln) = any.downcast_ref::<LayerNorm>() {
            out.push(TAG_LAYERNORM);
            push_usize(&mut out, ln.dim());
            push_vec(&mut out, &ln.gamma.data);
            push_vec(&mut out, &ln.beta.data);
        } else if let Some(tb) = any.downcast_ref::<TransformerBlock>() {
            out.push(TAG_TRANSFORMER_BLOCK);
            push_usize(&mut out, tb.context_len());
            push_usize(&mut out, tb.num_heads());
            push_usize(&mut out, tb.embedding_dim());
            push_usize(&mut out, tb.hidden_dim());
            // Serialize all projection weights in order: ln1, q, k, v, out, ln2,
            // ffn1, ffn2. `weight_f32` dequantizes any in-memory-quantized proj.
            push_vec(&mut out, &tb.ln1.gamma.data);
            push_vec(&mut out, &tb.ln1.beta.data);
            push_mat(&mut out, &tb.q_proj.weight_f32(), tb.q_proj.in_features());
            push_vec(&mut out, &tb.q_proj.bias.data);
            push_mat(&mut out, &tb.k_proj.weight_f32(), tb.k_proj.in_features());
            push_vec(&mut out, &tb.k_proj.bias.data);
            push_mat(&mut out, &tb.v_proj.weight_f32(), tb.v_proj.in_features());
            push_vec(&mut out, &tb.v_proj.bias.data);
            push_mat(
                &mut out,
                &tb.out_proj.weight_f32(),
                tb.out_proj.in_features(),
            );
            push_vec(&mut out, &tb.out_proj.bias.data);
            push_vec(&mut out, &tb.ln2.gamma.data);
            push_vec(&mut out, &tb.ln2.beta.data);
            push_mat(&mut out, &tb.ffn1.weight_f32(), tb.ffn1.in_features());
            push_vec(&mut out, &tb.ffn1.bias.data);
            push_mat(&mut out, &tb.ffn2.weight_f32(), tb.ffn2.in_features());
            push_vec(&mut out, &tb.ffn2.bias.data);
        } else {
            return Err(InferError::Format(format!(
                "unknown layer type: {}",
                layer.name()
            )));
        }
    }
    vprintln!("[loader::to_bytes] Total size: {} bytes", out.len());
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Deserialization (from_bytes)
// ─────────────────────────────────────────────────────────────────────────────

pub fn from_bytes(bytes: &[u8]) -> Result<(Sequential, Normalizer, ModelMetadata)> {
    vprintln!("[loader::from_bytes] Deserializing {} bytes", bytes.len());
    let mut r = Reader::new(bytes);
    if r.take(4)? != MAGIC {
        return Err(InferError::Format("bad FINF magic".into()));
    }
    let ver = r.u32()?;
    vprintln!("[loader::from_bytes] FINF version: {}", ver);
    if ver != VERSION && ver != VERSION_QUANT {
        return Err(InferError::Format(format!(
            "unsupported FINF v{ver} (need v{VERSION} or v{VERSION_QUANT})"
        )));
    }
    r.v5 = ver == VERSION_QUANT;

    let norm_len = r.usize()?;
    let norm_str = r.utf8(norm_len)?;
    // SLM models may have empty normalizer — return trivial identity norm
    let norm = if norm_str.is_empty() {
        Normalizer {
            means: vec![],
            stds: vec![],
        }
    } else {
        Normalizer::decode(norm_str)?
    };

    let meta_len = r.usize()?;
    let meta = ModelMetadata::from_json(r.utf8(meta_len)?)?;

    let num_layers = r.usize()?;
    vprintln!("[loader::from_bytes] Loading {} layers", num_layers);
    let mut model = Sequential::new();
    for layer_i in 0..num_layers {
        match r.u8()? {
            TAG_LINEAR => {
                let in_f = r.usize()?;
                let out_f = r.usize()?;
                vprintln!(
                    "[loader::from_bytes]   layer[{}]: Linear({}→{})",
                    layer_i,
                    in_f,
                    out_f
                );
                let w = r.weights_q(in_f, out_f)?;
                let bias = r.weights(out_f)?;
                model.push(Box::new(linear_from_loaded(in_f, out_f, w, bias)?));
            }
            TAG_ACTIVATION => {
                let t = r.u8()?;
                let act = Activation::from_tag(t)
                    .ok_or_else(|| InferError::Format(format!("bad act tag {t}")))?;
                vprintln!(
                    "[loader::from_bytes]   layer[{}]: Activation({:?})",
                    layer_i,
                    act
                );
                model.push(Box::new(ActivationLayer::new(act)));
            }
            TAG_EMBEDDING => {
                let vocab_size = r.usize()?;
                let max_seq_len = r.usize()?;
                let embedding_dim = r.usize()?;
                vprintln!(
                    "[loader::from_bytes]   layer[{}]: Embedding(vocab={}, seq={}, dim={})",
                    layer_i,
                    vocab_size,
                    max_seq_len,
                    embedding_dim
                );
                // Embeddings stay f32 in memory (lookup + add, not a GEMV), but
                // we still read a quantized table by dequantizing on load.
                let tok = r.weights_q(vocab_size, embedding_dim)?.into_f32();
                let pos = r.weights_q(max_seq_len, embedding_dim)?.into_f32();
                model.push(Box::new(Embedding::new(
                    vocab_size,
                    max_seq_len,
                    embedding_dim,
                    tok,
                    pos,
                )?));
            }
            TAG_LAYERNORM => {
                let dim = r.usize()?;
                vprintln!(
                    "[loader::from_bytes]   layer[{}]: LayerNorm(dim={})",
                    layer_i,
                    dim
                );
                model.push(Box::new(LayerNorm::new(
                    dim,
                    r.weights(dim)?,
                    r.weights(dim)?,
                )?));
            }
            TAG_TRANSFORMER_BLOCK => {
                let context_len = r.usize()?;
                let num_heads = r.usize()?;
                let embedding_dim = r.usize()?;
                let hidden_dim = r.usize()?;
                vprintln!("[loader::from_bytes]   layer[{}]: TransformerBlock(ctx={}, heads={}, dim={}, hidden={})",
                    layer_i, context_len, num_heads, embedding_dim, hidden_dim);
                let c = embedding_dim;
                let h = hidden_dim;
                // Read in the exact serialization order. The six projection
                // matrices stay quantized in memory (Opt#1) — they are the bulk
                // of a large model's weights; LayerNorm and biases stay f32.
                let ln1_g = r.weights(c)?;
                let ln1_b = r.weights(c)?;
                let q_w = r.weights_q(c, c)?;
                let q_b = r.weights(c)?;
                let k_w = r.weights_q(c, c)?;
                let k_b = r.weights(c)?;
                let v_w = r.weights_q(c, c)?;
                let v_b = r.weights(c)?;
                let o_w = r.weights_q(c, c)?;
                let o_b = r.weights(c)?;
                let ln2_g = r.weights(c)?;
                let ln2_b = r.weights(c)?;
                let f1_w = r.weights_q(c, h)?;
                let f1_b = r.weights(h)?;
                let f2_w = r.weights_q(h, c)?;
                let f2_b = r.weights(c)?;
                model.push(Box::new(TransformerBlock::from_parts(
                    context_len,
                    num_heads,
                    embedding_dim,
                    LayerNorm::new(c, ln1_g, ln1_b)?,
                    linear_from_loaded(c, c, q_w, q_b)?,
                    linear_from_loaded(c, c, k_w, k_b)?,
                    linear_from_loaded(c, c, v_w, v_b)?,
                    linear_from_loaded(c, c, o_w, o_b)?,
                    LayerNorm::new(c, ln2_g, ln2_b)?,
                    linear_from_loaded(c, h, f1_w, f1_b)?,
                    linear_from_loaded(h, c, f2_w, f2_b)?,
                )?));
            }
            TAG_FLATTEN => {
                vprintln!("[loader::from_bytes]   layer[{}]: Flatten", layer_i);
                model.push(Box::new(Flatten::new()));
            }
            t => return Err(InferError::Format(format!("bad layer tag {t}"))),
        }
    }
    vprintln!("[loader::from_bytes] Deserialized: {} layers", model.len());
    Ok((model, norm, meta))
}

pub fn save(model: &Sequential, norm: &Normalizer, meta: &ModelMetadata, path: &str) -> Result<()> {
    vprintln!("[loader::save] Saving model to: {}", path);
    let bytes = to_bytes(model, norm, meta)?;
    vprintln!("[loader::save] Writing {} bytes to disk", bytes.len());
    std::fs::write(path, bytes)?;
    vprintln!("[loader::save] Done");
    Ok(())
}
/// Like [`save`] but writes FINF v5 with int8-quantised weights (≈4× smaller).
pub fn save_quantized(
    model: &Sequential,
    norm: &Normalizer,
    meta: &ModelMetadata,
    path: &str,
) -> Result<()> {
    vprintln!(
        "[loader::save_quantized] Saving quantized model to: {}",
        path
    );
    let bytes = to_bytes_quantized(model, norm, meta)?;
    vprintln!(
        "[loader::save_quantized] Writing {} bytes to disk",
        bytes.len()
    );
    std::fs::write(path, bytes)?;
    Ok(())
}
/// Like [`save_quantized`] but writes **int4** weights (~8× smaller than f32).
pub fn save_quantized_int4(
    model: &Sequential,
    norm: &Normalizer,
    meta: &ModelMetadata,
    path: &str,
) -> Result<()> {
    let bytes = to_bytes_quantized_int4(model, norm, meta)?;
    std::fs::write(path, bytes)?;
    Ok(())
}

pub fn load(path: &str) -> Result<(Sequential, Normalizer, ModelMetadata)> {
    vprintln!("[loader::load] Loading model from: {}", path);
    let bytes = std::fs::read(path)?;
    vprintln!("[loader::load] Read {} bytes from disk", bytes.len());
    from_bytes(&bytes)
}

// ─────────────────────────────────────────────────────────────────────────────
// GGUF codecs
//
// A BPE transformer SLM (`Embedding, TransformerBlock×N, LayerNorm, Linear,
// Softmax`) is written in llama.cpp's `gpt2` layout, so llama.cpp / Ollama /
// LM Studio run it as-is. Every other model uses the generic `ferrum` layout
// (`ferrum.layer.{i}.*` keys, `l{i}.*` tensors). Both store matrices as F32,
// Q8_0 or Q4_0 by the same rule: quantized only when the GGML row length
// (`ne0`, the input dim) is a multiple of 32; vectors and `position_embd` are
// always F32. Ferrum extras live under `ferrum.*`, which llama.cpp ignores.
// ─────────────────────────────────────────────────────────────────────────────

/// GPT-2's LayerNorm epsilon, which Ferrum's LayerNorm also uses.
const LN_EPS: f32 = 1e-5;

impl QPrec {
    fn name(self) -> &'static str {
        match self {
            QPrec::F32 => "f32",
            QPrec::Int8 => "int8",
            QPrec::Int4 => "int4",
        }
    }
    /// GGML type for a matrix whose GGML rows hold `ne0` values.
    fn ggml_type(self, ne0: usize) -> u32 {
        match self {
            QPrec::Int8 if q8_0_eligible(ne0) => GGML_Q8_0,
            QPrec::Int4 if q8_0_eligible(ne0) => GGML_Q4_0,
            _ => GGML_F32,
        }
    }
}

fn bad(msg: String) -> InferError {
    InferError::Format(msg)
}

/// Accumulates one model's GGUF keys and tensors.
struct GgufOut {
    b: GgufBuilder,
    prec: QPrec,
    /// Whether any tensor was actually stored quantized.
    quantized: bool,
}

impl GgufOut {
    fn str(&mut self, key: &str, v: &str) {
        self.b.meta(key, MetaValue::String(v.into()));
    }
    fn u32(&mut self, key: &str, v: usize) {
        self.b.meta(key, MetaValue::U32(v as u32));
    }
    /// An F32 tensor with GGML dims `dims`.
    fn f32(&mut self, name: &str, dims: &[usize], data: &[f32]) {
        let dims: Vec<u64> = dims.iter().map(|&d| d as u64).collect();
        let bytes = encode_tensor(data, GGML_F32).expect("F32 encodes any length");
        self.b.tensor(name, &dims, GGML_F32, bytes);
    }
    /// A 1-D tensor (bias, norm): always F32.
    fn vec(&mut self, name: &str, data: &[f32]) {
        self.f32(name, &[data.len()], data);
    }
    /// `rows` GGML rows of `ne0` values (already in GGML order), quantized by
    /// the precision when `ne0 % 32 == 0`.
    fn rows(&mut self, name: &str, data: &[f32], rows: usize, ne0: usize) {
        let ty = self.prec.ggml_type(ne0);
        self.quantized |= ty != GGML_F32;
        let bytes = encode_tensor(data, ty).expect("ne0 % 32 == 0 for quantized types");
        self.b.tensor(name, &[ne0 as u64, rows as u64], ty, bytes);
    }
    /// A Ferrum row-major `[rows_in, cols_out]` weight. GGML's `ne0` is the
    /// input dim, so GGML row `o` is column `o`: the matrix is transposed.
    fn matrix(&mut self, name: &str, w: &[f32], rows_in: usize, cols_out: usize) {
        self.rows(name, &transpose_2d(w, rows_in, cols_out), cols_out, rows_in);
    }
}

/// The layers of a model the GPT-2 codec can carry.
struct Gpt2Parts<'a> {
    emb: &'a Embedding,
    blocks: Vec<&'a TransformerBlock>,
    lnf: &'a LayerNorm,
    head: &'a Linear,
}

/// `Some` when `model` is a BPE transformer SLM in exactly GPT-2's shape:
/// `Embedding, TransformerBlock×N (N ≥ 1, uniform), LayerNorm, Linear (no
/// bias, vocab wide), Softmax`.
fn gpt2_parts<'a>(model: &'a Sequential, meta: &ModelMetadata) -> Option<Gpt2Parts<'a>> {
    let l = model.layers();
    if meta.task != TaskType::TransformerSLM || meta.tokenizer_state.is_empty() || l.len() < 5 {
        return None;
    }
    let n = l.len();
    let emb = l[0].as_any().downcast_ref::<Embedding>()?;
    let blocks = l[1..n - 3]
        .iter()
        .map(|b| b.as_any().downcast_ref::<TransformerBlock>())
        .collect::<Option<Vec<_>>>()?;
    let lnf = l[n - 3].as_any().downcast_ref::<LayerNorm>()?;
    let head = l[n - 2].as_any().downcast_ref::<Linear>()?;
    let soft = l[n - 1].as_any().downcast_ref::<ActivationLayer>()?;
    let (d, b0) = (emb.embedding_dim(), blocks[0]);
    let uniform = blocks.iter().all(|b| {
        b.context_len() == emb.max_seq_len()
            && b.embedding_dim() == d
            && b.num_heads() == b0.num_heads()
            && b.hidden_dim() == b0.hidden_dim()
    });
    let fits = uniform
        && soft.kind == Activation::Softmax
        && lnf.dim() == d
        && head.in_features() == d
        && head.out_features() == emb.vocab_size()
        && head.bias.data.iter().all(|&x| x == 0.0);
    fits.then_some(Gpt2Parts {
        emb,
        blocks,
        lnf,
        head,
    })
}

impl GgufOut {
    fn gpt2(&mut self, p: &Gpt2Parts, meta: &ModelMetadata) -> Result<()> {
        let tok = ByteBpeTokenizer::from_state(&meta.tokenizer_state)?;
        let (v, d, ctx) = (
            p.emb.vocab_size(),
            p.emb.embedding_dim(),
            p.emb.max_seq_len(),
        );
        if tok.vocab_size() != v {
            return Err(bad(format!(
                "tokenizer has {} tokens but the model vocab is {v}",
                tok.vocab_size()
            )));
        }
        let b0 = p.blocks[0];
        for (k, x) in [
            ("context_length", ctx),
            ("embedding_length", d),
            ("block_count", p.blocks.len()),
            ("feed_forward_length", b0.hidden_dim()),
            ("attention.head_count", b0.num_heads()),
        ] {
            self.u32(&format!("gpt2.{k}"), x);
        }
        self.b
            .meta("gpt2.attention.layer_norm_epsilon", MetaValue::F32(LN_EPS));
        self.str("tokenizer.ggml.model", "gpt2");
        self.str("tokenizer.ggml.pre", "gpt-2");
        let strings =
            |v: Vec<String>| MetaValue::Array(v.into_iter().map(MetaValue::String).collect());
        self.b
            .meta("tokenizer.ggml.tokens", strings(tok.gguf_tokens()));
        self.b
            .meta("tokenizer.ggml.merges", strings(tok.gguf_merges()));
        let types = tok.gguf_token_types().into_iter().map(MetaValue::I32);
        self.b.meta(
            "tokenizer.ggml.token_type",
            MetaValue::Array(types.collect()),
        );
        if let Some(eos) = tok.eos_id() {
            self.u32("tokenizer.ggml.bos_token_id", eos);
            self.u32("tokenizer.ggml.eos_token_id", eos);
        }
        self.b
            .meta("tokenizer.ggml.add_bos_token", MetaValue::Bool(false));
        self.b
            .meta("tokenizer.ggml.add_eos_token", MetaValue::Bool(false));

        let tok_rows = &p.emb.token_weight.data;
        self.rows("token_embd.weight", tok_rows, v, d);
        self.f32("position_embd.weight", &[d, ctx], &p.emb.pos_weight.data);
        for (i, b) in p.blocks.iter().enumerate() {
            let n = |s: &str| format!("blk.{i}.{s}");
            let (q, k, vp) = (&b.q_proj, &b.k_proj, &b.v_proj);
            self.vec(&n("attn_norm.weight"), &b.ln1.gamma.data);
            self.vec(&n("attn_norm.bias"), &b.ln1.beta.data);
            let qkv: Vec<f32> = [q, k, vp]
                .iter()
                .flat_map(|l| transpose_2d(&l.weight_f32(), d, d))
                .collect();
            self.rows(&n("attn_qkv.weight"), &qkv, 3 * d, d);
            let qkv_b = [&q.bias.data[..], &k.bias.data, &vp.bias.data].concat();
            self.vec(&n("attn_qkv.bias"), &qkv_b);
            self.matrix(&n("attn_output.weight"), &b.out_proj.weight_f32(), d, d);
            self.vec(&n("attn_output.bias"), &b.out_proj.bias.data);
            self.vec(&n("ffn_norm.weight"), &b.ln2.gamma.data);
            self.vec(&n("ffn_norm.bias"), &b.ln2.beta.data);
            let h = b.hidden_dim();
            self.matrix(&n("ffn_up.weight"), &b.ffn1.weight_f32(), d, h);
            self.vec(&n("ffn_up.bias"), &b.ffn1.bias.data);
            self.matrix(&n("ffn_down.weight"), &b.ffn2.weight_f32(), h, d);
            self.vec(&n("ffn_down.bias"), &b.ffn2.bias.data);
        }
        self.vec("output_norm.weight", &p.lnf.gamma.data);
        self.vec("output_norm.bias", &p.lnf.beta.data);
        // Tied head: omitted, llama.cpp falls back to token_embd.
        let head = transpose_2d(&p.head.weight_f32(), d, v);
        if head != *tok_rows {
            self.rows("output.weight", &head, v, d);
        }
        Ok(())
    }

    fn generic(&mut self, model: &Sequential) -> Result<()> {
        self.u32("ferrum.layer_count", model.len());
        for (i, layer) in model.layers().iter().enumerate() {
            let k = |s: &str| format!("ferrum.layer.{i}.{s}");
            let t = |s: &str| format!("l{i}.{s}");
            let any = layer.as_any();
            let ty = if let Some(lin) = any.downcast_ref::<Linear>() {
                let (a, b) = (lin.in_features(), lin.out_features());
                self.u32(&k("in"), a);
                self.u32(&k("out"), b);
                self.matrix(&t("weight"), &lin.weight_f32(), a, b);
                self.vec(&t("bias"), &lin.bias.data);
                "linear"
            } else if let Some(act) = any.downcast_ref::<ActivationLayer>() {
                self.u32(&k("kind"), act.kind.tag() as usize);
                "activation"
            } else if any.downcast_ref::<Flatten>().is_some() {
                "flatten"
            } else if let Some(emb) = any.downcast_ref::<Embedding>() {
                let (v, s, d) = (emb.vocab_size(), emb.max_seq_len(), emb.embedding_dim());
                self.u32(&k("vocab"), v);
                self.u32(&k("seq"), s);
                self.u32(&k("dim"), d);
                self.rows(&t("token"), &emb.token_weight.data, v, d);
                self.f32(&t("pos"), &[d, s], &emb.pos_weight.data);
                "embedding"
            } else if let Some(ln) = any.downcast_ref::<LayerNorm>() {
                self.u32(&k("dim"), ln.dim());
                self.vec(&t("gamma"), &ln.gamma.data);
                self.vec(&t("beta"), &ln.beta.data);
                "layernorm"
            } else if let Some(tb) = any.downcast_ref::<TransformerBlock>() {
                let (c, h) = (tb.embedding_dim(), tb.hidden_dim());
                self.u32(&k("context"), tb.context_len());
                self.u32(&k("heads"), tb.num_heads());
                self.u32(&k("dim"), c);
                self.u32(&k("hidden"), h);
                let lins = [
                    ("q", &tb.q_proj, c, c),
                    ("k", &tb.k_proj, c, c),
                    ("v", &tb.v_proj, c, c),
                    ("out", &tb.out_proj, c, c),
                    ("ffn1", &tb.ffn1, c, h),
                    ("ffn2", &tb.ffn2, h, c),
                ];
                for (p, ln) in [("ln1", &tb.ln1), ("ln2", &tb.ln2)] {
                    self.vec(&t(&format!("{p}.gamma")), &ln.gamma.data);
                    self.vec(&t(&format!("{p}.beta")), &ln.beta.data);
                }
                for (p, lin, a, b) in lins {
                    self.matrix(&t(&format!("{p}.weight")), &lin.weight_f32(), a, b);
                    self.vec(&t(&format!("{p}.bias")), &lin.bias.data);
                }
                "transformer_block"
            } else {
                return Err(bad(format!("unknown layer type: {}", layer.name())));
            };
            self.str(&k("type"), ty);
        }
        Ok(())
    }
}

/// Serialize a model as GGUF: llama.cpp's `gpt2` layout for a BPE transformer
/// SLM, the generic `ferrum` layout otherwise. `prec` picks F32 / Q8_0 / Q4_0
/// for the matrices (see the section comment for the rule).
pub fn to_gguf(
    model: &Sequential,
    norm: &Normalizer,
    meta: &ModelMetadata,
    prec: QPrec,
) -> Result<Vec<u8>> {
    let gpt2 = gpt2_parts(model, meta);
    let mut o = GgufOut {
        b: GgufBuilder::new(),
        prec,
        quantized: false,
    };
    let mut stored = meta.clone();
    if gpt2.is_some() {
        stored.tokenizer_state.clear(); // lives in tokenizer.ggml.*
    }
    o.str(
        "general.architecture",
        if gpt2.is_some() { "gpt2" } else { "ferrum" },
    );
    o.str("general.name", &meta.dataset_name);
    o.u32("ferrum.format", 1);
    o.str("ferrum.quant", prec.name());
    o.str("ferrum.metadata", &stored.to_json());
    o.str("ferrum.normalizer", &norm.encode());
    match &gpt2 {
        Some(p) => o.gpt2(p, meta)?,
        None => o.generic(model)?,
    }
    // general.file_type: 0 = all F32, 7 = mostly Q8_0, 2 = mostly Q4_0
    // (honest: 0 when no tensor could be quantized).
    let ft = match (o.quantized, prec) {
        (true, QPrec::Int8) => 7,
        (true, QPrec::Int4) => 2,
        _ => 0,
    };
    o.u32("general.file_type", ft as usize);
    o.u32("general.quantization_version", 2);
    Ok(o.b.into_bytes())
}

/// Typed, error-reporting reads from a parsed GGUF.
struct GgufIn<'a>(&'a Gguf);

impl GgufIn<'_> {
    fn usize(&self, key: &str) -> Result<usize> {
        self.0
            .meta(key)
            .and_then(MetaValue::as_usize)
            .ok_or_else(|| bad(format!("GGUF key '{key}' missing or not an integer")))
    }
    fn str(&self, key: &str) -> Result<&str> {
        self.0
            .meta(key)
            .and_then(MetaValue::as_str)
            .ok_or_else(|| bad(format!("GGUF key '{key}' missing or not a string")))
    }
    fn array<T>(&self, key: &str, item: impl Fn(&MetaValue) -> Option<T>) -> Result<Vec<T>> {
        match self.0.meta(key) {
            Some(MetaValue::Array(a)) => a.iter().map(item).collect(),
            _ => None,
        }
        .ok_or_else(|| bad(format!("GGUF key '{key}' missing or of the wrong type")))
    }
    /// Tensor `name` dequantized to f32, after checking its GGML dims.
    fn tensor(&self, name: &str, dims: &[usize]) -> Result<Vec<f32>> {
        let t = self
            .0
            .tensor(name)
            .ok_or_else(|| bad(format!("missing tensor '{name}'")))?;
        if !t.dims.iter().copied().eq(dims.iter().map(|&d| d as u64)) {
            return Err(bad(format!(
                "tensor '{name}' has dims {:?}, expected {dims:?}",
                t.dims
            )));
        }
        self.0.dequantize(t)
    }
    fn vec(&self, name: &str, n: usize) -> Result<Vec<f32>> {
        self.tensor(name, &[n])
    }
    fn rows(&self, name: &str, rows: usize, ne0: usize) -> Result<Vec<f32>> {
        self.tensor(name, &[ne0, rows])
    }
    /// Inverse of [`GgufOut::matrix`]: back to Ferrum's `[rows_in, cols_out]`.
    fn matrix(&self, name: &str, rows_in: usize, cols_out: usize) -> Result<Vec<f32>> {
        let g = self.rows(name, cols_out, rows_in)?;
        Ok(transpose_2d(&g, cols_out, rows_in))
    }
    fn linear(&self, w: &str, b: &str, rows_in: usize, cols_out: usize) -> Result<Linear> {
        Linear::new(
            rows_in,
            cols_out,
            self.matrix(w, rows_in, cols_out)?,
            self.vec(b, cols_out)?,
        )
    }
    fn layernorm(&self, g: &str, b: &str, d: usize) -> Result<LayerNorm> {
        LayerNorm::new(d, self.vec(g, d)?, self.vec(b, d)?)
    }
    /// `ferrum.normalizer`; absent (foreign file) or empty = identity.
    fn normalizer(&self) -> Result<Normalizer> {
        match self.0.meta("ferrum.normalizer").and_then(MetaValue::as_str) {
            None | Some("") => Ok(Normalizer {
                means: vec![],
                stds: vec![],
            }),
            Some(s) => Normalizer::decode(s),
        }
    }
}

/// Map a GGUF BPE vocab onto Ferrum's id layout (256 byte tokens in byte
/// order, one token per merge in rank order, then control tokens), so a
/// foreign GPT-2 vocab (which orders its byte tokens differently) loads too.
/// Returns the tokenizer and `perm[ferrum_id] = file_id`.
fn ferrum_vocab(
    tokens: &[String],
    merges: &[String],
    types: &[i32],
) -> Result<(ByteBpeTokenizer, Vec<usize>)> {
    if types.len() != tokens.len() {
        return Err(bad(format!(
            "gguf vocab: {} tokens but {} token types",
            tokens.len(),
            types.len()
        )));
    }
    let id: HashMap<&str, usize> = tokens
        .iter()
        .enumerate()
        .map(|(i, t)| (t.as_str(), i))
        .collect();
    let mut used = vec![false; tokens.len()];
    let mut perm = Vec::with_capacity(tokens.len());
    // Tokens never hold a raw space (byte_to_unicode maps it to 'Ġ'), so the
    // merge's single space is the separator and dropping it gives the result.
    let wanted = (0..=255u8)
        .map(|b| byte_to_unicode(b).to_string())
        .chain(merges.iter().map(|m| m.replacen(' ', "", 1)));
    for w in wanted {
        match id.get(w.as_str()) {
            Some(&i) if !used[i] => {
                used[i] = true;
                perm.push(i);
            }
            _ => return Err(bad(format!("gguf vocab: no unique token {w:?}"))),
        }
    }
    perm.extend((0..tokens.len()).filter(|&i| !used[i]));
    let ordered: Vec<String> = perm.iter().map(|&i| tokens[i].clone()).collect();
    let ordered_types: Vec<i32> = perm.iter().map(|&i| types[i]).collect();
    let tok = ByteBpeTokenizer::from_gguf_vocab(&ordered, merges, &ordered_types)?;
    Ok((tok, perm))
}

/// Reorder `ne0`-wide rows: output row `i` is input row `perm[i]`.
fn permute_rows(data: &[f32], perm: &[usize], ne0: usize) -> Vec<f32> {
    perm.iter()
        .flat_map(|&i| &data[i * ne0..(i + 1) * ne0])
        .copied()
        .collect()
}

fn gpt2_from(r: &GgufIn) -> Result<(Sequential, Normalizer, ModelMetadata)> {
    let k = |s: &str| r.usize(&format!("gpt2.{s}"));
    let (ctx, d, n_layer) = (
        k("context_length")?,
        k("embedding_length")?,
        k("block_count")?,
    );
    let (ff, heads) = (k("feed_forward_length")?, k("attention.head_count")?);
    let string = |v: &MetaValue| v.as_str().map(String::from);
    let tokens = r.array("tokenizer.ggml.tokens", string)?;
    let merges = r.array("tokenizer.ggml.merges", string)?;
    let types = r.array("tokenizer.ggml.token_type", |v| match v {
        MetaValue::I32(x) => Some(*x),
        _ => None,
    })?;
    let (tok, perm) = ferrum_vocab(&tokens, &merges, &types)?;
    let v = perm.len();

    let tok_rows = permute_rows(&r.rows("token_embd.weight", v, d)?, &perm, d);
    let head_rows = match r.0.tensor("output.weight") {
        Some(_) => permute_rows(&r.rows("output.weight", v, d)?, &perm, d),
        None => tok_rows.clone(),
    };
    let pos = r.rows("position_embd.weight", ctx, d)?;
    let mut model = Sequential::new().with(Box::new(Embedding::new(v, ctx, d, tok_rows, pos)?));
    for i in 0..n_layer {
        let n = |s: &str| format!("blk.{i}.{s}");
        let qkv = r.rows(&n("attn_qkv.weight"), mul_dims(3, d)?, d)?;
        let qkv_b = r.vec(&n("attn_qkv.bias"), 3 * d)?;
        // GGML rows Q, then K, then V; each block of d rows is [out, in].
        let proj = |j: usize| {
            let w = &qkv[j * d * d..(j + 1) * d * d];
            Linear::new(
                d,
                d,
                transpose_2d(w, d, d),
                qkv_b[j * d..(j + 1) * d].to_vec(),
            )
        };
        model.push(Box::new(TransformerBlock::from_parts(
            ctx,
            heads,
            d,
            r.layernorm(&n("attn_norm.weight"), &n("attn_norm.bias"), d)?,
            proj(0)?,
            proj(1)?,
            proj(2)?,
            r.linear(&n("attn_output.weight"), &n("attn_output.bias"), d, d)?,
            r.layernorm(&n("ffn_norm.weight"), &n("ffn_norm.bias"), d)?,
            r.linear(&n("ffn_up.weight"), &n("ffn_up.bias"), d, ff)?,
            r.linear(&n("ffn_down.weight"), &n("ffn_down.bias"), ff, d)?,
        )?));
    }
    model.push(Box::new(r.layernorm(
        "output_norm.weight",
        "output_norm.bias",
        d,
    )?));
    let head_w = transpose_2d(&head_rows, v, d);
    model.push(Box::new(Linear::new(d, v, head_w, vec![0.0; v])?));
    model.push(Box::new(ActivationLayer::new(Activation::Softmax)));

    let mut meta = match r.0.meta("ferrum.metadata").and_then(MetaValue::as_str) {
        Some(json) => ModelMetadata::from_json(json)?,
        // A foreign GPT-2 file: rebuild what Ferrum's trainer would record.
        None => ModelMetadata {
            dataset_name: r.str("general.name").unwrap_or("gpt2").into(),
            task: TaskType::TransformerSLM,
            feature_names: (0..ctx).map(|i| format!("c_{i}")).collect(),
            feature_ranges: vec![[0.0, v as f32]; ctx],
            class_names: vec![],
            target_name: "next_char".into(),
            target_range: [0.0, v as f32],
            input_dim: ctx,
            output_dim: v,
            tokenizer_state: String::new(),
        },
    };
    meta.tokenizer_state = tok.encode_state();
    Ok((model, r.normalizer()?, meta))
}

fn generic_from(r: &GgufIn) -> Result<(Sequential, Normalizer, ModelMetadata)> {
    let format = r.usize("ferrum.format")?;
    if format != 1 {
        return Err(bad(format!("unsupported ferrum.format {format} (need 1)")));
    }
    let meta = ModelMetadata::from_json(r.str("ferrum.metadata")?)?;
    let mut model = Sequential::new();
    for i in 0..r.usize("ferrum.layer_count")? {
        let k = |s: &str| r.usize(&format!("ferrum.layer.{i}.{s}"));
        let t = |s: &str| format!("l{i}.{s}");
        let layer: Box<dyn Layer> = match r.str(&format!("ferrum.layer.{i}.type"))? {
            "linear" => {
                let (a, b) = (k("in")?, k("out")?);
                Box::new(r.linear(&t("weight"), &t("bias"), a, b)?)
            }
            "activation" => {
                let tag = k("kind")?;
                let act = u8::try_from(tag)
                    .ok()
                    .and_then(Activation::from_tag)
                    .ok_or_else(|| bad(format!("bad activation kind {tag}")))?;
                Box::new(ActivationLayer::new(act))
            }
            "flatten" => Box::new(Flatten::new()),
            "embedding" => {
                let (v, s, d) = (k("vocab")?, k("seq")?, k("dim")?);
                let token = r.rows(&t("token"), v, d)?;
                Box::new(Embedding::new(v, s, d, token, r.rows(&t("pos"), s, d)?)?)
            }
            "layernorm" => Box::new(r.layernorm(&t("gamma"), &t("beta"), k("dim")?)?),
            "transformer_block" => {
                let (ctx, heads, c, h) = (k("context")?, k("heads")?, k("dim")?, k("hidden")?);
                let lin = |p: &str, a: usize, b: usize| {
                    r.linear(&t(&format!("{p}.weight")), &t(&format!("{p}.bias")), a, b)
                };
                let ln =
                    |p: &str| r.layernorm(&t(&format!("{p}.gamma")), &t(&format!("{p}.beta")), c);
                Box::new(TransformerBlock::from_parts(
                    ctx,
                    heads,
                    c,
                    ln("ln1")?,
                    lin("q", c, c)?,
                    lin("k", c, c)?,
                    lin("v", c, c)?,
                    lin("out", c, c)?,
                    ln("ln2")?,
                    lin("ffn1", c, h)?,
                    lin("ffn2", h, c)?,
                )?)
            }
            other => return Err(bad(format!("unknown layer type '{other}'"))),
        };
        model.push(layer);
    }
    Ok((model, r.normalizer()?, meta))
}

/// Load a model written by [`to_gguf`], or a foreign GPT-2 GGUF (e.g. one
/// converted by llama.cpp). Weights are dequantized to f32 in memory.
// ponytail: Q8_0/Q4_0 matrices are expanded to f32 (4×/8× the file's size in
// RAM); keep them packed once a Q8_0 kernel lands, if large models need it.
pub fn from_gguf(bytes: &[u8]) -> Result<(Sequential, Normalizer, ModelMetadata)> {
    let g = Gguf::parse(bytes.to_vec())?;
    let r = GgufIn(&g);
    match g.architecture() {
        Some("gpt2") => gpt2_from(&r),
        Some("ferrum") => generic_from(&r),
        other => Err(bad(format!(
            "not a Ferrum-loadable model: architecture {other:?} (need gpt2 or ferrum)"
        ))),
    }
}

/// The precision a GGUF model was saved at: `ferrum.quant` ("f32" | "int8" |
/// "int4"), else derived from `general.file_type` (0 / 7 / 2).
pub fn gguf_quant(bytes: &[u8]) -> Result<String> {
    let g = Gguf::parse(bytes.to_vec())?;
    if let Some(q) = g.meta("ferrum.quant").and_then(MetaValue::as_str) {
        return Ok(q.to_string());
    }
    match g.meta("general.file_type").and_then(MetaValue::as_usize) {
        Some(0) => Ok("f32".into()),
        Some(7) => Ok("int8".into()),
        Some(2) => Ok("int4".into()),
        ft => Err(bad(format!(
            "no ferrum.quant and unrecognised general.file_type {ft:?}"
        ))),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::Activation;
    use crate::layer::ActivationLayer;
    use crate::meta::TaskType;
    use crate::tensor::Tensor;

    fn make_bundle() -> (Sequential, Normalizer, ModelMetadata) {
        let l1 = Linear::new(
            4,
            8,
            (0..32).map(|i| i as f32 * 0.01).collect(),
            vec![0.0; 8],
        )
        .unwrap();
        let l2 = Linear::new(
            8,
            3,
            (0..24).map(|i| i as f32 * -0.01).collect(),
            vec![0.1; 3],
        )
        .unwrap();
        let model = Sequential::new()
            .with(Box::new(l1))
            .with(Box::new(ActivationLayer::new(Activation::ReLU)))
            .with(Box::new(l2))
            .with(Box::new(ActivationLayer::new(Activation::Softmax)));
        let norm = Normalizer {
            means: vec![5.8, 3.1, 3.7, 1.2],
            stds: vec![0.8, 0.4, 1.7, 0.8],
        };
        let meta = ModelMetadata {
            dataset_name: "test".into(),
            task: TaskType::Classification,
            feature_names: vec!["a".into(), "b".into(), "c".into(), "d".into()],
            feature_ranges: vec![[0.0, 10.0]; 4],
            class_names: vec!["X".into(), "Y".into(), "Z".into()],
            target_name: "".into(),
            target_range: [0.0, 2.0],
            input_dim: 4,
            output_dim: 3,
            tokenizer_state: String::new(),
        };
        (model, norm, meta)
    }

    /// A transformer bundle whose matrices are ≥ QUANT_MIN_LEN, so int4/int8
    /// quantization actually engages (the small `make_embedding_bundle` stays
    /// f32). `Embedding → TransformerBlock → LayerNorm → Linear(head) → Softmax`.
    fn make_embedding_bundle_big() -> (Sequential, Normalizer, ModelMetadata) {
        let vocab = 10usize;
        let ctx = 4usize;
        let dim = 64usize;
        let heads = 4usize;
        let hidden = 128usize;
        let f =
            |n: usize, s: f32| -> Vec<f32> { (0..n).map(|i| (i as f32 * s).sin() * 0.2).collect() };
        let emb =
            Embedding::new(vocab, ctx, dim, f(vocab * dim, 0.01), f(ctx * dim, 0.02)).unwrap();
        let tb = TransformerBlock::new(
            ctx,
            heads,
            dim,
            vec![1.0; dim],
            vec![0.0; dim],
            f(dim * dim, 0.013),
            vec![0.0; dim],
            f(dim * dim, 0.017),
            vec![0.0; dim],
            f(dim * dim, 0.019),
            vec![0.0; dim],
            f(dim * dim, 0.023),
            vec![0.0; dim],
            vec![1.0; dim],
            vec![0.0; dim],
            f(dim * hidden, 0.007),
            vec![0.0; hidden],
            f(hidden * dim, 0.009),
            vec![0.0; dim],
        )
        .unwrap();
        let lnf = LayerNorm::new(dim, vec![1.0; dim], vec![0.0; dim]).unwrap();
        let head = Linear::new(dim, vocab, f(dim * vocab, 0.005), vec![0.0; vocab]).unwrap();
        let model = Sequential::new()
            .with(Box::new(emb))
            .with(Box::new(tb))
            .with(Box::new(lnf))
            .with(Box::new(head))
            .with(Box::new(ActivationLayer::new(Activation::Softmax)));
        let norm = Normalizer {
            means: vec![],
            stds: vec![],
        };
        let meta = ModelMetadata {
            dataset_name: "big".into(),
            task: TaskType::TransformerSLM,
            feature_names: (0..ctx).map(|i| format!("c_{i}")).collect(),
            feature_ranges: vec![[0.0, vocab as f32]; ctx],
            class_names: (0..vocab).map(|i| format!("{i:x}")).collect(),
            target_name: "next".into(),
            target_range: [0.0, vocab as f32],
            input_dim: ctx,
            output_dim: vocab,
            tokenizer_state: String::new(),
        };
        (model, norm, meta)
    }

    fn make_embedding_bundle() -> (Sequential, Normalizer, ModelMetadata) {
        let vocab_size = 5;
        let max_seq_len = 4;
        let embedding_dim = 8;
        let emb = Embedding::new(
            vocab_size,
            max_seq_len,
            embedding_dim,
            (0..vocab_size * embedding_dim)
                .map(|i| i as f32 * 0.1)
                .collect(),
            (0..max_seq_len * embedding_dim)
                .map(|i| i as f32 * 0.01)
                .collect(),
        )
        .unwrap();
        let lm_head = Linear::new(
            embedding_dim,
            vocab_size,
            (0..embedding_dim * vocab_size)
                .map(|i| i as f32 * 0.01)
                .collect(),
            vec![0.0; vocab_size],
        )
        .unwrap();
        let model = Sequential::new()
            .with(Box::new(emb))
            .with(Box::new(ActivationLayer::new(Activation::Softmax)))
            .with(Box::new(lm_head));
        let norm = Normalizer {
            means: vec![],
            stds: vec![],
        };
        let vocab: Vec<String> = "abcde".chars().map(|c| c.to_string()).collect();
        let meta = ModelMetadata {
            dataset_name: "tiny_slm".into(),
            task: TaskType::TransformerSLM,
            feature_names: (0..max_seq_len).map(|i| format!("c_{i}")).collect(),
            feature_ranges: vec![[0.0, vocab_size as f32]; max_seq_len],
            class_names: vocab,
            target_name: "next_char".into(),
            target_range: [0.0, vocab_size as f32],
            input_dim: max_seq_len,
            output_dim: vocab_size,
            tokenizer_state: String::new(),
        };
        (model, norm, meta)
    }

    #[test]
    fn roundtrip_preserves_outputs() {
        let (model, norm, meta) = make_bundle();
        let raw = Tensor::row(vec![5.1f32, 3.5, 1.4, 0.2]).unwrap();
        let before = model.forward(&norm.transform(&raw).unwrap()).unwrap();
        let bytes = to_bytes(&model, &norm, &meta).unwrap();
        let (m2, n2, _) = from_bytes(&bytes).unwrap();
        let after = m2.forward(&n2.transform(&raw).unwrap()).unwrap();
        for (a, b) in before.data.iter().zip(&after.data) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn metadata_survives_roundtrip() {
        let (model, norm, meta) = make_bundle();
        let bytes = to_bytes(&model, &norm, &meta).unwrap();
        let (_, _, m2) = from_bytes(&bytes).unwrap();
        assert_eq!(m2.task, TaskType::Classification);
        assert_eq!(m2.class_names, vec!["X", "Y", "Z"]);
        assert_eq!(m2.feature_names, vec!["a", "b", "c", "d"]);
    }

    #[test]
    fn embedding_bundle_roundtrips() {
        let (model, norm, meta) = make_embedding_bundle();
        let bytes = to_bytes(&model, &norm, &meta).unwrap();
        let (m2, _, m2meta) = from_bytes(&bytes).unwrap();
        assert_eq!(m2meta.task, TaskType::TransformerSLM);
        assert_eq!(m2meta.class_names, vec!["a", "b", "c", "d", "e"]);
        // Forward should still work
        let x = Tensor::matrix(1, 4, vec![0.0, 1.0, 2.0, 3.0]).unwrap();
        assert!(m2.forward(&x).is_ok());
    }

    #[test]
    fn quantized_roundtrip_is_smaller_and_close() {
        let (model, norm, meta) = make_bundle();
        let raw = Tensor::row(vec![5.1f32, 3.5, 1.4, 0.2]).unwrap();
        let before = model.forward(&norm.transform(&raw).unwrap()).unwrap();

        let full = to_bytes(&model, &norm, &meta).unwrap();
        let quant = to_bytes_quantized(&model, &norm, &meta).unwrap();
        // make_bundle's largest tensor is 32 values (< QUANT_MIN_LEN), so use
        // a bigger Linear to actually exercise the int8 path.
        let big = Linear::new(
            64,
            64,
            (0..64 * 64)
                .map(|i| (i as f32 * 0.37).sin() * 0.1)
                .collect(),
            vec![0.0; 64],
        )
        .unwrap();
        let big_model = Sequential::new().with(Box::new(big));
        let big_norm = Normalizer {
            means: vec![],
            stds: vec![],
        };
        let big_full = to_bytes(&big_model, &big_norm, &meta).unwrap();
        let big_quant = to_bytes_quantized(&big_model, &big_norm, &meta).unwrap();
        assert!(
            (big_quant.len() as f32) < (big_full.len() as f32) * 0.35,
            "quantized {} not ≈4× smaller than {}",
            big_quant.len(),
            big_full.len()
        );

        // Small model: everything stays f32 (marker overhead only) and the
        // outputs must match the v4 file closely.
        assert!(quant.len() <= full.len() + model.len() * 8);
        let (m2, n2, _) = from_bytes(&quant).unwrap();
        let after = m2.forward(&n2.transform(&raw).unwrap()).unwrap();
        for (a, b) in before.data.iter().zip(&after.data) {
            assert!((a - b).abs() < 1e-6);
        }

        // Big model: int8 error is bounded by scale/2 per weight.
        let x = Tensor::row(vec![0.1f32; 64]).unwrap();
        let y_full = big_model.forward(&x).unwrap();
        let (m3, _, _) = from_bytes(&big_quant).unwrap();
        let y_quant = m3.forward(&x).unwrap();
        for (a, b) in y_full.data.iter().zip(&y_quant.data) {
            assert!((a - b).abs() < 0.05, "quantized output drifted: {a} vs {b}");
        }
    }

    #[test]
    fn int4_roundtrip_keeps_weights_quantized_in_memory() {
        // A Linear big enough to quantize (≥ QUANT_MIN_LEN). After an int4
        // save→load the layer must still be quantized *in memory* (Opt#1: no f32
        // expansion) and produce output close to the f32 original.
        let (in_f, out_f) = (96usize, 96usize);
        let w: Vec<f32> = (0..in_f * out_f)
            .map(|i| (i as f32 * 0.011).sin() * 0.4)
            .collect();
        let big = Linear::new(in_f, out_f, w, vec![0.0; out_f]).unwrap();
        let model = Sequential::new().with(Box::new(big));
        let norm = Normalizer {
            means: vec![],
            stds: vec![],
        };
        let (_, _, meta) = make_bundle();

        let x = Tensor::matrix(
            1,
            in_f,
            (0..in_f).map(|i| (i as f32 * 0.03).cos()).collect(),
        )
        .unwrap();
        let y_f32 = model.forward(&x).unwrap();

        let bytes = to_bytes_quantized_int4(&model, &norm, &meta).unwrap();
        assert_eq!(&bytes[4..8], &VERSION_QUANT.to_le_bytes());
        // ~8× smaller than the f32 file for the weight payload.
        let f32_bytes = to_bytes(&model, &norm, &meta).unwrap();
        assert!(
            (bytes.len() as f32) < (f32_bytes.len() as f32) * 0.4,
            "int4 file {} not far smaller than f32 {}",
            bytes.len(),
            f32_bytes.len()
        );

        let (loaded, _, _) = from_bytes(&bytes).unwrap();
        let lin = loaded.layers()[0]
            .as_any()
            .downcast_ref::<Linear>()
            .expect("layer 0 is Linear");
        let qw = lin.qweight().expect("weights stay quantized in memory");
        assert_eq!(qw.kind, crate::quant::QKind::Int4);
        assert!(lin.weight.data.is_empty(), "f32 copy must be dropped");

        let y_q = loaded.forward(&x).unwrap();
        let mae: f32 = y_f32
            .data
            .iter()
            .zip(&y_q.data)
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / out_f as f32;
        assert!(mae < 0.2, "int4 forward drifted too far: mae={mae}");
    }

    #[test]
    fn int4_transformer_block_projections_stay_quantized() {
        // The bulk of a large model is the transformer block projections; verify
        // they survive an int4 round trip as in-memory QWeights.
        let (model, norm, meta) = make_embedding_bundle_big();
        let bytes = to_bytes_quantized_int4(&model, &norm, &meta).unwrap();
        let (loaded, _, _) = from_bytes(&bytes).unwrap();
        let tb = loaded
            .layers()
            .iter()
            .find_map(|l| l.as_any().downcast_ref::<TransformerBlock>())
            .expect("has a transformer block");
        assert!(tb.q_proj.qweight().is_some(), "q_proj should be quantized");
        assert!(tb.ffn1.qweight().is_some(), "ffn1 should be quantized");
        // Forward still runs and is finite.
        let x = Tensor::matrix(1, meta.input_dim, vec![1.0; meta.input_dim]).unwrap();
        let y = loaded.forward(&x).unwrap();
        assert!(y.data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn per_channel_weight_vector_roundtrips_and_isolates_outlier() {
        // 4 channels × 32 values; channel 2 holds a large outlier.
        let channels = 4;
        let row = 32;
        let mut data: Vec<f32> = (0..channels * row)
            .map(|i| (i as f32 * 0.05).sin() * 0.3)
            .collect();
        data[2 * row + 1] = 25.0;

        let mut out = Vec::new();
        push_weights(&mut out, &data, channels, /*v5=*/ true, QPrec::Int8);
        assert_eq!(
            out[0], ENC_INT8_PER_CHANNEL,
            "matrix should use the per-channel marker"
        );

        let mut r = Reader::new(&out);
        r.v5 = true;
        let decoded = r.weights(data.len()).unwrap();
        assert_eq!(r.pos, out.len(), "reader did not consume the whole vector");

        // Each value is within its own channel's scale/2.
        let scales = int8_scales_per_channel(&data, channels);
        for (i, (o, q)) in data.iter().zip(&decoded).enumerate() {
            assert!(
                (o - q).abs() <= scales[i / row] * 0.5 + 1e-6,
                "value {i}: {o} vs {q}"
            );
        }
        // The outlier did NOT inflate the clean channels' scale: channel 0 stays
        // accurate (this is the per-channel win over per-tensor).
        let clean_err: f32 = data[..row]
            .iter()
            .zip(&decoded[..row])
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            clean_err < 0.05,
            "clean channel error too large: {clean_err}"
        );
    }

    #[test]
    fn one_dimensional_weight_uses_per_tensor_marker() {
        // channels = 1 (a bias-like vector) must fall back to the single-scale
        // int8 marker, not the per-channel one.
        let data: Vec<f32> = (0..QUANT_MIN_LEN)
            .map(|i| (i as f32 * 0.1).cos() * 0.2)
            .collect();
        let mut out = Vec::new();
        push_weights(&mut out, &data, 1, true, QPrec::Int8);
        assert_eq!(
            out[0], ENC_INT8,
            "1-D vector should use the per-tensor marker"
        );
        let mut r = Reader::new(&out);
        r.v5 = true;
        let decoded = r.weights(data.len()).unwrap();
        let scale = int8_scale(&data);
        for (o, q) in data.iter().zip(&decoded) {
            assert!((o - q).abs() <= scale * 0.5 + 1e-6);
        }
    }

    #[test]
    fn quantized_transformer_roundtrips() {
        // All five v4 layer types through the v5 quantized writer/reader.
        let (model, norm, meta) = make_embedding_bundle();
        let bytes = to_bytes_quantized(&model, &norm, &meta).unwrap();
        assert_eq!(&bytes[4..8], &5u32.to_le_bytes());
        let (m2, _, m2meta) = from_bytes(&bytes).unwrap();
        assert_eq!(m2meta.task, TaskType::TransformerSLM);
        let x = Tensor::matrix(1, 4, vec![0.0, 1.0, 2.0, 3.0]).unwrap();
        assert!(m2.forward(&x).is_ok());
    }

    #[test]
    fn flatten_roundtrips_as_v5() {
        let lin = Linear::new(
            8,
            3,
            (0..24).map(|i| i as f32 * 0.01).collect(),
            vec![0.0; 3],
        )
        .unwrap();
        let model = Sequential::new()
            .with(Box::new(Flatten::new()))
            .with(Box::new(lin));
        let norm = Normalizer {
            means: vec![],
            stds: vec![],
        };
        let (_, _, meta) = make_bundle();

        let bytes = to_bytes(&model, &norm, &meta).unwrap();
        // Flatten is not expressible in v4, so to_bytes must upgrade to v5.
        assert_eq!(&bytes[4..8], &5u32.to_le_bytes());

        let (m2, _, _) = from_bytes(&bytes).unwrap();
        assert_eq!(m2.len(), 2);
        assert_eq!(m2.layers()[0].name(), "Flatten");
        let x = Tensor::matrix(2, 4, vec![0.1; 8]).unwrap(); // [2,4] → [1,8]
        let before = model.forward(&x).unwrap();
        let after = m2.forward(&x).unwrap();
        assert_eq!(before.shape, vec![1, 3]);
        for (a, b) in before.data.iter().zip(&after.data) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn bad_weight_marker_errors() {
        let (model, norm, meta) = make_embedding_bundle();
        let mut bytes = to_bytes_quantized(&model, &norm, &meta).unwrap();
        // First weight marker sits right after the first layer's tag + dims:
        // header(4+4) + norm str(4+0) + meta(4+len) + num_layers(4) + tag(1) + 3 dims(12).
        let meta_len = meta.to_json().len();
        let marker_pos = 8 + 4 + (4 + meta_len) + 4 + 1 + 12;
        assert!(bytes[marker_pos] == ENC_F32 || bytes[marker_pos] == ENC_INT8);
        bytes[marker_pos] = 7; // invalid marker
        assert!(matches!(from_bytes(&bytes), Err(InferError::Format(_))));
    }

    #[test]
    fn bad_magic_errors() {
        assert!(matches!(
            from_bytes(b"XXXX\x04\x00\x00\x00"),
            Err(InferError::Format(_))
        ));
    }

    #[test]
    fn wrong_version_errors() {
        let (m, n, meta) = make_bundle();
        let mut bytes = to_bytes(&m, &n, &meta).unwrap();
        bytes[4] = 99;
        bytes[5] = 0;
        bytes[6] = 0;
        bytes[7] = 0;
        assert!(matches!(from_bytes(&bytes), Err(InferError::Format(_))));
    }

    #[test]
    fn truncation_errors() {
        let (m, n, meta) = make_bundle();
        let mut bytes = to_bytes(&m, &n, &meta).unwrap();
        bytes.truncate(bytes.len() - 8);
        assert!(from_bytes(&bytes).is_err());
    }

    #[derive(Debug)]
    struct DummyLayer;
    impl crate::layer::Layer for DummyLayer {
        fn forward(&self, x: &Tensor) -> Result<Tensor> {
            Ok(x.clone())
        }
        fn name(&self) -> String {
            "DummyLayer".to_string()
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    #[test]
    fn to_bytes_unknown_layer_error() {
        let mut model = Sequential::new();
        model.push(Box::new(DummyLayer));
        let norm = Normalizer {
            means: vec![],
            stds: vec![],
        };
        let meta = ModelMetadata {
            dataset_name: "test".into(),
            task: TaskType::Classification,
            feature_names: vec![],
            feature_ranges: vec![],
            class_names: vec![],
            target_name: "".into(),
            target_range: [0.0, 0.0],
            input_dim: 0,
            output_dim: 0,
            tokenizer_state: String::new(),
        };
        assert!(matches!(
            to_bytes(&model, &norm, &meta),
            Err(InferError::Format(_))
        ));
    }

    #[test]
    fn huge_dims_error_fast_without_alloc() {
        // A Linear layer claiming u32::MAX × u32::MAX weights must be rejected
        // by the bounds check, not by an allocation attempt.
        let (_, _, meta) = make_bundle();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        push_u32(&mut bytes, VERSION);
        push_str(&mut bytes, "");
        push_str(&mut bytes, &meta.to_json());
        push_u32(&mut bytes, 1);
        bytes.push(TAG_LINEAR);
        push_u32(&mut bytes, u32::MAX);
        push_u32(&mut bytes, u32::MAX);
        assert!(matches!(from_bytes(&bytes), Err(InferError::Format(_))));
    }

    #[test]
    fn oversized_embedding_dims_error() {
        let (_, _, meta) = make_bundle();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        push_u32(&mut bytes, VERSION);
        push_str(&mut bytes, "");
        push_str(&mut bytes, &meta.to_json());
        push_u32(&mut bytes, 1);
        bytes.push(TAG_EMBEDDING);
        push_u32(&mut bytes, 1_000_000); // vocab
        push_u32(&mut bytes, 1_000_000); // max_seq_len
        push_u32(&mut bytes, 1_000_000); // embedding_dim — 4 TB of weights claimed
        assert!(matches!(from_bytes(&bytes), Err(InferError::Format(_))));
    }

    #[test]
    fn unknown_layer_tag_error() {
        let (_, _, meta) = make_bundle();
        let mut malformed = Vec::new();
        malformed.extend_from_slice(MAGIC);
        push_u32(&mut malformed, VERSION);
        push_str(&mut malformed, ""); // empty normalizer
        push_str(&mut malformed, &meta.to_json()); // metadata JSON
        push_u32(&mut malformed, 1); // 1 layer
        malformed.push(99); // bad layer tag!
        assert!(matches!(from_bytes(&malformed), Err(InferError::Format(_))));
    }

    #[test]
    fn layernorm_and_transformerblock_roundtrip() {
        let context_len = 4;
        let num_heads = 2;
        let embedding_dim = 8;
        let hidden_dim = 16;
        let c = embedding_dim;
        let h = hidden_dim;

        let ln = LayerNorm::new(embedding_dim, vec![1.0; c], vec![0.0; c]).unwrap();
        let tb = TransformerBlock::new(
            context_len,
            num_heads,
            embedding_dim,
            vec![1.0; c],
            vec![0.0; c],
            vec![0.1; c * c],
            vec![0.0; c],
            vec![0.1; c * c],
            vec![0.0; c],
            vec![0.1; c * c],
            vec![0.0; c],
            vec![0.1; c * c],
            vec![0.0; c],
            vec![1.0; c],
            vec![0.0; c],
            vec![0.1; c * h],
            vec![0.0; h],
            vec![0.1; h * c],
            vec![0.0; c],
        )
        .unwrap();

        let mut model = Sequential::new();
        model.push(Box::new(ln));
        model.push(Box::new(tb));

        let norm = Normalizer {
            means: vec![],
            stds: vec![],
        };
        let meta = ModelMetadata {
            dataset_name: "tb_test".into(),
            task: TaskType::TransformerSLM,
            feature_names: vec![],
            feature_ranges: vec![],
            class_names: vec![],
            target_name: "".into(),
            target_range: [0.0, 0.0],
            input_dim: embedding_dim,
            output_dim: embedding_dim,
            tokenizer_state: String::new(),
        };

        let bytes = to_bytes(&model, &norm, &meta).unwrap();
        let (m2, _, m2meta) = from_bytes(&bytes).unwrap();
        assert_eq!(m2.len(), 2);
        assert_eq!(m2meta.task, TaskType::TransformerSLM);

        let input = Tensor::matrix(4, embedding_dim, vec![0.5; 4 * embedding_dim]).unwrap();
        let out1 = model.forward(&input).unwrap();
        let out2 = m2.forward(&input).unwrap();
        for (a, b) in out1.data.iter().zip(&out2.data) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    // ── GGUF codecs ──────────────────────────────────────────────────────────

    use crate::gguf::{Gguf, MetaValue, GGML_F32, GGML_Q4_0, GGML_Q8_0};
    use crate::gguf_write::{encode_tensor, GgufBuilder};
    use crate::slm::{GenerativeSLM, TransformerConfig};
    use crate::tokenizer::ByteBpeTokenizer;

    fn gguf_corpus() -> String {
        let words = [
            "the", "cat", "sat", "on", "a", "mat", "and", "dog", "ran", "far", "away", "home",
        ];
        (0..600)
            .map(|i| {
                let end = if i % 4 == 3 { ". " } else { " " };
                format!(
                    "{} {} {} {}{end}",
                    words[i % 12],
                    words[(i * 7 + 3) % 12],
                    i % 50,
                    words[(i * 5 + 1) % 12]
                )
            })
            .collect()
    }

    /// Distinct, deterministic parameter values (a swapped tensor shows up).
    fn vals(n: usize, seed: usize) -> Vec<f32> {
        (0..n)
            .map(|i| ((i * 31 + seed * 977) as f32 * 0.137).sin() * 0.3)
            .collect()
    }

    fn slm_meta(ctx: usize, vocab: usize, tokenizer_state: String) -> ModelMetadata {
        ModelMetadata {
            dataset_name: "gguf test".into(),
            task: TaskType::TransformerSLM,
            feature_names: (0..ctx).map(|i| format!("c_{i}")).collect(),
            feature_ranges: vec![[0.0, vocab as f32]; ctx],
            class_names: vec![],
            target_name: "next_char".into(),
            target_range: [0.0, vocab as f32],
            input_dim: ctx,
            output_dim: vocab,
            tokenizer_state,
        }
    }

    type Layers = Vec<Box<dyn crate::layer::Layer>>;

    fn block(ctx: usize, d: usize, h: usize, s: usize) -> Box<dyn crate::layer::Layer> {
        Box::new(
            TransformerBlock::new(
                ctx,
                4,
                d,
                vals(d, s),
                vals(d, s + 1),
                vals(d * d, s + 2),
                vals(d, s + 3),
                vals(d * d, s + 4),
                vals(d, s + 5),
                vals(d * d, s + 6),
                vals(d, s + 7),
                vals(d * d, s + 8),
                vals(d, s + 9),
                vals(d, s + 10),
                vals(d, s + 11),
                vals(d * h, s + 12),
                vals(h, s + 13),
                vals(h * d, s + 14),
                vals(d, s + 15),
            )
            .unwrap(),
        )
    }

    fn seq(layers: Layers) -> Sequential {
        layers.into_iter().fold(Sequential::new(), |m, l| m.with(l))
    }

    /// The layers of a GPT-2-codec model built directly (fast, every parameter
    /// distinct): `Embedding, TransformerBlock×2, LayerNorm, Linear, Softmax`,
    /// plus its BPE metadata.
    fn bpe_layers(embed: usize, hidden: usize, tie: bool) -> (Layers, ModelMetadata) {
        let tok = ByteBpeTokenizer::train(&gguf_corpus(), 300).unwrap();
        let (v, ctx, d, h) = (tok.vocab_size(), 8, embed, hidden);
        let token = vals(v * d, 1);
        let head = if tie {
            crate::gguf::transpose_2d(&token, v, d)
        } else {
            vals(d * v, 92)
        };
        let layers: Layers = vec![
            Box::new(Embedding::new(v, ctx, d, token, vals(ctx * d, 2)).unwrap()),
            block(ctx, d, h, 10),
            block(ctx, d, h, 30),
            Box::new(LayerNorm::new(d, vals(d, 90), vals(d, 91)).unwrap()),
            Box::new(Linear::new(d, v, head, vec![0.0; v]).unwrap()),
            Box::new(ActivationLayer::new(Activation::Softmax)),
        ];
        (layers, slm_meta(ctx, v, tok.encode_state()))
    }

    fn empty_norm() -> Normalizer {
        Normalizer {
            means: vec![],
            stds: vec![],
        }
    }

    fn bpe_parts(
        embed: usize,
        hidden: usize,
        tie: bool,
    ) -> (Sequential, Normalizer, ModelMetadata) {
        let (layers, meta) = bpe_layers(embed, hidden, tie);
        (seq(layers), empty_norm(), meta)
    }

    fn ids_input(n: usize, vocab: usize) -> Tensor {
        Tensor::matrix(
            1,
            n,
            (0..n).map(|i| ((i * 37 + 5) % vocab) as f32).collect(),
        )
        .unwrap()
    }

    fn fwd(m: &Sequential, meta: &ModelMetadata) -> Vec<f32> {
        m.forward(&ids_input(meta.input_dim, meta.output_dim))
            .unwrap()
            .data
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    /// The reference tiny config (0.1M params), trained briefly.
    fn trained_slm(qat: bool) -> GenerativeSLM {
        let cfg = TransformerConfig {
            context_len: 64,
            embed_dim: 64,
            num_heads: 4,
            num_blocks: 2,
            hidden_dim: 128,
            vocab_size: 512,
            weight_tying: true,
            max_tokens: 8 * 16 * 64, // 8 optimizer steps: enough to move every weight
            qat,
            ..TransformerConfig::default()
        };
        GenerativeSLM::train_transformer_config(
            &gguf_corpus(),
            &cfg,
            &mut crate::rng::Rng::new(1),
            |_, _| {},
        )
        .unwrap()
    }

    fn parse(bytes: &[u8]) -> Gguf {
        Gguf::parse(bytes.to_vec()).unwrap()
    }

    fn arch(bytes: &[u8]) -> String {
        parse(bytes).architecture().unwrap().to_string()
    }

    type Tensors = Vec<(String, Vec<u64>, Vec<f32>)>;
    type Kv = std::collections::BTreeMap<String, MetaValue>;
    type GgufEdit = Box<dyn Fn(&mut Kv, &mut Tensors)>;
    type LayerEdit = Box<dyn Fn(&mut Layers)>;

    /// Re-emit an F32-only GGUF after editing its metadata and tensors.
    fn rebuild(bytes: &[u8], edit: impl FnOnce(&mut Kv, &mut Tensors)) -> Vec<u8> {
        let g = parse(bytes);
        let mut meta = g.metadata.clone();
        let mut ts: Tensors = g
            .tensors
            .iter()
            .map(|t| (t.name.clone(), t.dims.clone(), g.dequantize(t).unwrap()))
            .collect();
        edit(&mut meta, &mut ts);
        let mut b = GgufBuilder::new();
        for (k, v) in meta {
            b.meta(&k, v);
        }
        for (n, d, v) in ts {
            b.tensor(&n, &d, GGML_F32, encode_tensor(&v, GGML_F32).unwrap());
        }
        b.into_bytes()
    }

    fn load_err(bytes: &[u8]) -> String {
        match from_gguf(bytes) {
            Ok(_) => panic!("expected an error"),
            Err(e) => e.to_string(),
        }
    }

    #[test]
    fn f32_gguf_roundtrip_is_exact() {
        let slm = trained_slm(false);
        let bytes = to_gguf(&slm.model, &slm.norm, &slm.meta, QPrec::F32).unwrap();
        assert_eq!(&bytes[..4], b"GGUF");
        assert_eq!(arch(&bytes), "gpt2");
        let (m2, n2, meta2) = from_gguf(&bytes).unwrap();
        assert_eq!(fwd(&slm.model, &slm.meta), fwd(&m2, &meta2));
        assert_eq!(
            meta2.to_json(),
            slm.meta.to_json(),
            "tokenizer state survives"
        );
        assert_eq!(n2.encode(), "");
        assert_eq!(gguf_quant(&bytes).unwrap(), "f32");
    }

    #[test]
    fn qat_model_reloads_bit_identically_from_int8_gguf() {
        let slm = trained_slm(true);
        let bytes = to_gguf(&slm.model, &slm.norm, &slm.meta, QPrec::Int8).unwrap();
        let g = parse(&bytes);
        assert_eq!(g.tensor("token_embd.weight").unwrap().ggml_type, GGML_Q8_0);
        assert_eq!(
            g.tensor("position_embd.weight").unwrap().ggml_type,
            GGML_F32
        );
        let (m2, n2, meta2) = from_gguf(&bytes).unwrap();
        let back = GenerativeSLM::new(m2, n2, meta2);
        let text = &gguf_corpus()[..1200]; // > context + 1 tokens, kept short for speed
        assert_eq!(slm.evaluate(text).unwrap(), back.evaluate(text).unwrap());
        assert_eq!(gguf_quant(&bytes).unwrap(), "int8");
    }

    #[test]
    fn int4_and_ineligible_dims() {
        // embed 48 / hidden 80: no GGML row length is a multiple of 32.
        let (m, n, meta) = bpe_parts(48, 80, false);
        let bytes = to_gguf(&m, &n, &meta, QPrec::Int8).unwrap();
        let g = parse(&bytes);
        assert!(g.tensors.iter().all(|t| t.ggml_type == GGML_F32));
        assert_eq!(g.meta("general.file_type"), Some(&MetaValue::U32(0)));
        assert_eq!(gguf_quant(&bytes).unwrap(), "int8");
        let (m2, _, meta2) = from_gguf(&bytes).unwrap();
        assert_eq!(fwd(&m, &meta), fwd(&m2, &meta2));

        let (m, n, meta) = bpe_parts(64, 128, true);
        let bytes = to_gguf(&m, &n, &meta, QPrec::Int4).unwrap();
        let g = parse(&bytes);
        assert_eq!(
            g.tensor("blk.0.ffn_down.weight").unwrap().ggml_type,
            GGML_Q4_0
        );
        assert_eq!(
            g.tensor("position_embd.weight").unwrap().ggml_type,
            GGML_F32
        );
        assert_eq!(g.meta("general.file_type"), Some(&MetaValue::U32(2)));
        assert_eq!(gguf_quant(&bytes).unwrap(), "int4");
        let (m2, _, meta2) = from_gguf(&bytes).unwrap();
        let err = max_abs_diff(&fwd(&m, &meta), &fwd(&m2, &meta2));
        assert!(err < 0.1, "int4 forward drifted {err}");
    }

    #[test]
    fn gpt2_layout_matches_llama_cpp() {
        for tie in [true, false] {
            let (m, n, meta) = bpe_parts(64, 128, tie);
            let bytes = to_gguf(&m, &n, &meta, QPrec::Int8).unwrap();
            let g = parse(&bytes);
            let tok = ByteBpeTokenizer::from_state(&meta.tokenizer_state).unwrap();
            let (v, d, ctx, ff) = (tok.vocab_size() as u64, 64u64, 8u64, 128u64);
            let u = |k: &str| g.meta(k).cloned();
            let s = |x: &str| Some(MetaValue::String(x.into()));
            assert_eq!(u("general.architecture"), s("gpt2"));
            assert_eq!(u("general.name"), s("gguf test"));
            assert_eq!(u("gpt2.context_length"), Some(MetaValue::U32(8)));
            assert_eq!(u("gpt2.embedding_length"), Some(MetaValue::U32(64)));
            assert_eq!(u("gpt2.block_count"), Some(MetaValue::U32(2)));
            assert_eq!(u("gpt2.feed_forward_length"), Some(MetaValue::U32(128)));
            assert_eq!(u("gpt2.attention.head_count"), Some(MetaValue::U32(4)));
            assert_eq!(
                u("gpt2.attention.layer_norm_epsilon"),
                Some(MetaValue::F32(1e-5))
            );
            assert_eq!(u("tokenizer.ggml.model"), s("gpt2"));
            assert_eq!(u("tokenizer.ggml.pre"), s("gpt-2"));
            let strs = |x: Vec<String>| {
                Some(MetaValue::Array(
                    x.into_iter().map(MetaValue::String).collect(),
                ))
            };
            assert_eq!(u("tokenizer.ggml.tokens"), strs(tok.gguf_tokens()));
            assert_eq!(u("tokenizer.ggml.merges"), strs(tok.gguf_merges()));
            assert_eq!(
                u("tokenizer.ggml.token_type"),
                Some(MetaValue::Array(
                    tok.gguf_token_types()
                        .into_iter()
                        .map(MetaValue::I32)
                        .collect()
                ))
            );
            let eos = tok.eos_id().unwrap() as u32;
            assert_eq!(u("tokenizer.ggml.bos_token_id"), Some(MetaValue::U32(eos)));
            assert_eq!(u("tokenizer.ggml.eos_token_id"), Some(MetaValue::U32(eos)));
            assert_eq!(
                u("tokenizer.ggml.add_bos_token"),
                Some(MetaValue::Bool(false))
            );
            assert_eq!(
                u("tokenizer.ggml.add_eos_token"),
                Some(MetaValue::Bool(false))
            );
            assert_eq!(u("general.file_type"), Some(MetaValue::U32(7)));
            assert_eq!(u("general.quantization_version"), Some(MetaValue::U32(2)));
            assert_eq!(u("ferrum.format"), Some(MetaValue::U32(1)));
            assert_eq!(u("ferrum.quant"), s("int8"));
            assert_eq!(u("ferrum.normalizer"), s(""));
            let mut no_tok = meta.clone();
            no_tok.tokenizer_state.clear();
            assert_eq!(u("ferrum.metadata"), s(&no_tok.to_json()));

            let mut want: Vec<(String, Vec<u64>, u32)> = vec![
                ("token_embd.weight".into(), vec![d, v], GGML_Q8_0),
                ("position_embd.weight".into(), vec![d, ctx], GGML_F32),
            ];
            for i in 0..2 {
                let b = |x: &str| format!("blk.{i}.{x}");
                want.extend([
                    (b("attn_norm.weight"), vec![d], GGML_F32),
                    (b("attn_norm.bias"), vec![d], GGML_F32),
                    (b("attn_qkv.weight"), vec![d, 3 * d], GGML_Q8_0),
                    (b("attn_qkv.bias"), vec![3 * d], GGML_F32),
                    (b("attn_output.weight"), vec![d, d], GGML_Q8_0),
                    (b("attn_output.bias"), vec![d], GGML_F32),
                    (b("ffn_norm.weight"), vec![d], GGML_F32),
                    (b("ffn_norm.bias"), vec![d], GGML_F32),
                    (b("ffn_up.weight"), vec![d, ff], GGML_Q8_0),
                    (b("ffn_up.bias"), vec![ff], GGML_F32),
                    (b("ffn_down.weight"), vec![ff, d], GGML_Q8_0),
                    (b("ffn_down.bias"), vec![d], GGML_F32),
                ]);
            }
            want.push(("output_norm.weight".into(), vec![d], GGML_F32));
            want.push(("output_norm.bias".into(), vec![d], GGML_F32));
            if !tie {
                want.push(("output.weight".into(), vec![d, v], GGML_Q8_0));
            }
            let got: Vec<(String, Vec<u64>, u32)> = g
                .tensors
                .iter()
                .map(|t| (t.name.clone(), t.dims.clone(), t.ggml_type))
                .collect();
            assert_eq!(got, want, "tie={tie}");

            // Q|K|V rows of attn_qkv are the transposed q/k/v projections.
            let f32_bytes = to_gguf(&m, &n, &meta, QPrec::F32).unwrap();
            let gf = parse(&f32_bytes);
            let qkv = gf
                .dequantize(gf.tensor("blk.1.attn_qkv.weight").unwrap())
                .unwrap();
            let tb = m.layers()[2]
                .as_any()
                .downcast_ref::<TransformerBlock>()
                .unwrap();
            let dd = 64 * 64;
            for (j, p) in [&tb.q_proj, &tb.k_proj, &tb.v_proj].iter().enumerate() {
                let t = crate::gguf::transpose_2d(&p.weight.data, 64, 64);
                assert_eq!(&qkv[j * dd..(j + 1) * dd], &t[..]);
            }
            let (m2, _, meta2) = from_gguf(&f32_bytes).unwrap();
            assert_eq!(fwd(&m, &meta), fwd(&m2, &meta2), "f32 exact, tie={tie}");
        }
    }

    #[test]
    fn tabular_mlp_roundtrips_with_normalizer() {
        let (model, norm, meta) = make_bundle();
        let bytes = to_gguf(&model, &norm, &meta, QPrec::F32).unwrap();
        let g = parse(&bytes);
        assert_eq!(g.architecture(), Some("ferrum"));
        assert_eq!(g.meta("ferrum.layer_count"), Some(&MetaValue::U32(4)));
        assert_eq!(
            g.meta("ferrum.layer.0.type"),
            Some(&MetaValue::String("linear".into()))
        );
        assert_eq!(g.meta("ferrum.layer.1.kind"), Some(&MetaValue::U32(1)));
        assert_eq!(g.tensor("l0.weight").unwrap().dims, vec![4, 8]);
        let (m2, n2, meta2) = from_gguf(&bytes).unwrap();
        assert_eq!(n2.encode(), norm.encode());
        assert_eq!(meta2.to_json(), meta.to_json());
        let x = Tensor::row(vec![0.1, -0.2, 0.3, 0.9]).unwrap();
        assert_eq!(
            model.forward(&x).unwrap().data,
            m2.forward(&x).unwrap().data
        );
    }

    #[test]
    fn generic_codec_covers_every_layer_kind() {
        // Flatten + Linear.
        let lin = Linear::new(8, 3, vals(24, 3), vals(3, 4)).unwrap();
        let model = Sequential::new()
            .with(Box::new(Flatten::new()))
            .with(Box::new(lin));
        let (_, norm, meta) = make_bundle();
        let bytes = to_gguf(&model, &norm, &meta, QPrec::F32).unwrap();
        let (m2, _, _) = from_gguf(&bytes).unwrap();
        assert_eq!(m2.layers()[0].name(), "Flatten");
        let x = Tensor::matrix(2, 4, vals(8, 5)).unwrap();
        assert_eq!(
            model.forward(&x).unwrap().data,
            m2.forward(&x).unwrap().data
        );

        // A char-level transformer (no tokenizer state) → generic codec with
        // Embedding / TransformerBlock / LayerNorm, at every precision.
        let (model, norm, meta) = make_embedding_bundle_big();
        let x = Tensor::matrix(1, 4, vec![0.0, 3.0, 9.0, 1.0]).unwrap();
        let want = model.forward(&x).unwrap().data;
        for (prec, ty, tol) in [
            (QPrec::F32, GGML_F32, 0.0),
            (QPrec::Int8, GGML_Q8_0, 0.05),
            (QPrec::Int4, GGML_Q4_0, 0.1),
        ] {
            let bytes = to_gguf(&model, &norm, &meta, prec).unwrap();
            let g = parse(&bytes);
            assert_eq!(g.architecture(), Some("ferrum"));
            assert_eq!(g.tensor("l1.ffn1.weight").unwrap().dims, vec![64, 128]);
            assert_eq!(g.tensor("l1.ffn2.weight").unwrap().ggml_type, ty);
            assert_eq!(g.tensor("l0.token").unwrap().ggml_type, ty);
            assert_eq!(g.tensor("l0.pos").unwrap().ggml_type, GGML_F32);
            assert_eq!(g.tensor("l1.ln2.beta").unwrap().ggml_type, GGML_F32);
            let (m2, _, meta2) = from_gguf(&bytes).unwrap();
            assert_eq!(meta2.to_json(), meta.to_json());
            let err = max_abs_diff(&want, &m2.forward(&x).unwrap().data);
            assert!(err <= tol, "{prec:?}: {err}");
        }
    }

    #[test]
    fn gpt2_codec_needs_the_exact_bpe_slm_shape() {
        let arch_of = |layers: Layers, meta: &ModelMetadata| {
            arch(&to_gguf(&seq(layers), &empty_norm(), meta, QPrec::F32).unwrap())
        };
        let (_, meta) = bpe_layers(32, 64, true);
        let v = meta.output_dim;
        let base = || bpe_layers(32, 64, true).0;
        assert_eq!(arch_of(base(), &meta), "gpt2");

        let mut cls = meta.clone();
        cls.task = TaskType::Classification;
        assert_eq!(arch_of(base(), &cls), "ferrum", "not an SLM");
        let mut chars = meta.clone();
        chars.tokenizer_state.clear();
        assert_eq!(arch_of(base(), &chars), "ferrum", "char-level");

        let ln = || -> Box<dyn crate::layer::Layer> {
            Box::new(LayerNorm::new(32, vals(32, 1), vals(32, 2)).unwrap())
        };
        let variants: Vec<(&str, LayerEdit)> = vec![
            (
                "too few layers",
                Box::new(|l: &mut Layers| {
                    l.remove(1);
                    l.remove(1);
                }),
            ),
            (
                "first is not Embedding",
                Box::new(move |l: &mut Layers| l[0] = ln()),
            ),
            (
                "middle is not a block",
                Box::new(move |l: &mut Layers| l[2] = ln()),
            ),
            (
                "no final LayerNorm",
                Box::new(|l: &mut Layers| l[3] = block(8, 32, 64, 50)),
            ),
            (
                "head is not Linear",
                Box::new(move |l: &mut Layers| l[4] = ln()),
            ),
            (
                "last is not an activation",
                Box::new(move |l: &mut Layers| l[5] = ln()),
            ),
            (
                "last is not Softmax",
                Box::new(|l: &mut Layers| l[5] = Box::new(ActivationLayer::new(Activation::ReLU))),
            ),
            (
                "head has a bias",
                Box::new(move |l: &mut Layers| {
                    l[4] = Box::new(Linear::new(32, v, vals(32 * v, 7), vals(v, 8)).unwrap())
                }),
            ),
            (
                "head width is not the vocab",
                Box::new(move |l: &mut Layers| {
                    l[4] = Box::new(Linear::new(32, 3, vals(96, 7), vec![0.0; 3]).unwrap())
                }),
            ),
            (
                "blocks differ",
                Box::new(|l: &mut Layers| l[2] = block(8, 32, 96, 50)),
            ),
            (
                "block context differs",
                Box::new(|l: &mut Layers| {
                    l[2] = Box::new(
                        TransformerBlock::from_parts(
                            4,
                            4,
                            32,
                            LayerNorm::new(32, vals(32, 1), vals(32, 2)).unwrap(),
                            Linear::new(32, 32, vals(1024, 3), vals(32, 4)).unwrap(),
                            Linear::new(32, 32, vals(1024, 5), vals(32, 6)).unwrap(),
                            Linear::new(32, 32, vals(1024, 7), vals(32, 8)).unwrap(),
                            Linear::new(32, 32, vals(1024, 9), vals(32, 10)).unwrap(),
                            LayerNorm::new(32, vals(32, 11), vals(32, 12)).unwrap(),
                            Linear::new(32, 64, vals(2048, 13), vals(64, 14)).unwrap(),
                            Linear::new(64, 32, vals(2048, 15), vals(32, 16)).unwrap(),
                        )
                        .unwrap(),
                    )
                }),
            ),
        ];
        for (why, edit) in variants {
            let mut l = base();
            edit(&mut l);
            let bytes = to_gguf(&seq(l), &empty_norm(), &meta, QPrec::F32).unwrap();
            assert_eq!(arch(&bytes), "ferrum", "{why}");
            from_gguf(&bytes).unwrap_or_else(|e| panic!("{why}: {e}"));
        }
    }

    #[test]
    fn to_gguf_rejects_inconsistent_models() {
        let (m, n, meta) = bpe_parts(32, 64, true);
        let mut other = meta.clone();
        other.tokenizer_state = ByteBpeTokenizer::train(&gguf_corpus(), 280)
            .unwrap()
            .encode_state();
        assert!(to_gguf(&m, &n, &other, QPrec::F32)
            .unwrap_err()
            .to_string()
            .contains("tokenizer has"));
        let mut broken = meta.clone();
        broken.tokenizer_state = "x".into();
        assert!(to_gguf(&m, &n, &broken, QPrec::F32).is_err());

        let model = Sequential::new().with(Box::new(DummyLayer));
        let (_, n, meta) = make_bundle();
        assert!(to_gguf(&model, &n, &meta, QPrec::F32)
            .unwrap_err()
            .to_string()
            .contains("unknown layer type"));
    }

    #[test]
    fn gpt2_without_eos_omits_the_special_ids() {
        // A tokenizer with merges but no special tokens.
        let (layers, mut meta) = bpe_layers(32, 64, true);
        let tok = ByteBpeTokenizer::from_state(&meta.tokenizer_state).unwrap();
        let plain = tok.encode_state().split('\n').next().unwrap().to_string();
        let plain_tok = ByteBpeTokenizer::from_state(&plain).unwrap();
        assert_eq!(plain_tok.eos_id(), None);
        // Model vocab must match: drop the trailing special rows.
        let v = plain_tok.vocab_size();
        let emb = layers[0].as_any().downcast_ref::<Embedding>().unwrap();
        let d = 32;
        let token = emb.token_weight.data[..v * d].to_vec();
        let mut l = layers;
        l[0] = Box::new(Embedding::new(v, 8, d, token.clone(), vals(8 * d, 2)).unwrap());
        l[4] = Box::new(
            Linear::new(d, v, crate::gguf::transpose_2d(&token, v, d), vec![0.0; v]).unwrap(),
        );
        meta.tokenizer_state = plain;
        meta.output_dim = v;
        let model = seq(l);
        let bytes = to_gguf(&model, &empty_norm(), &meta, QPrec::F32).unwrap();
        let g = parse(&bytes);
        assert_eq!(g.architecture(), Some("gpt2"));
        assert!(g.meta("tokenizer.ggml.eos_token_id").is_none());
        assert!(g.meta("tokenizer.ggml.bos_token_id").is_none());
        let (m2, _, meta2) = from_gguf(&bytes).unwrap();
        assert_eq!(meta2.tokenizer_state, meta.tokenizer_state);
        assert_eq!(fwd(&model, &meta), fwd(&m2, &meta2));
    }

    #[test]
    fn foreign_vocab_order_is_remapped_onto_ferrum_ids() {
        // Simulate a foreign GPT-2 file: same tokens, but the 256 byte tokens
        // in a different order (GPT-2 puts "!" first). The loader must map them
        // back and permute the embedding rows to match.
        let (m, n, meta) = bpe_parts(32, 64, false);
        let bytes = to_gguf(&m, &n, &meta, QPrec::F32).unwrap();
        let d = 32;
        let foreign = rebuild(&bytes, |kv, ts| {
            let MetaValue::Array(tokens) = kv.get_mut("tokenizer.ggml.tokens").unwrap() else {
                panic!()
            };
            tokens[..256].reverse();
            for (name, _, data) in ts.iter_mut() {
                if name == "token_embd.weight" || name == "output.weight" {
                    let mut rows: Vec<Vec<f32>> = data.chunks(d).map(|r| r.to_vec()).collect();
                    rows[..256].reverse();
                    *data = rows.concat();
                }
            }
            kv.remove("ferrum.metadata");
            kv.remove("ferrum.normalizer");
            kv.remove("general.name");
        });
        let (m2, n2, meta2) = from_gguf(&foreign).unwrap();
        assert_eq!(fwd(&m, &meta), fwd(&m2, &meta2));
        assert_eq!(meta2.tokenizer_state, meta.tokenizer_state);
        // Without ferrum.metadata the metadata is rebuilt from gpt2.* keys.
        assert_eq!(meta2.task, TaskType::TransformerSLM);
        assert_eq!(meta2.input_dim, 8);
        assert_eq!(meta2.output_dim, meta.output_dim);
        assert_eq!(meta2.dataset_name, "gpt2");
        assert_eq!(meta2.feature_names.len(), 8);
        assert_eq!(n2.encode(), "");
        // ...and gguf_quant falls back to general.file_type.
        assert_eq!(gguf_quant(&foreign).unwrap(), "f32");
    }

    #[test]
    fn gguf_quant_falls_back_to_file_type() {
        let mk = |ft: Option<u32>| {
            let mut b = GgufBuilder::new();
            if let Some(ft) = ft {
                b.meta("general.file_type", MetaValue::U32(ft));
            }
            b.into_bytes()
        };
        assert_eq!(gguf_quant(&mk(Some(0))).unwrap(), "f32");
        assert_eq!(gguf_quant(&mk(Some(7))).unwrap(), "int8");
        assert_eq!(gguf_quant(&mk(Some(2))).unwrap(), "int4");
        assert!(gguf_quant(&mk(Some(1))).is_err());
        assert!(gguf_quant(&mk(None)).is_err());
        assert!(gguf_quant(b"FINF\x05\x00\x00\x00").is_err());
    }

    #[test]
    fn rejects_foreign_and_truncated_gguf() {
        let mut b = GgufBuilder::new();
        b.meta("general.architecture", MetaValue::String("llama".into()));
        assert!(load_err(&b.into_bytes()).contains("not a Ferrum-loadable model"));
        assert!(load_err(&GgufBuilder::new().into_bytes()).contains("not a Ferrum-loadable model"));
        assert!(from_gguf(b"FINF\x05\x00\x00\x00").is_err());
        for prec in [QPrec::Int8, QPrec::F32] {
            let (m, n, meta) = bpe_parts(32, 64, true);
            let good = to_gguf(&m, &n, &meta, prec).unwrap();
            // Every cut point through the header, then a stride through the data.
            let cuts = (0..good.len()).filter(|&c| c < 16384 || c % 97 == 0);
            for cut in cuts {
                assert!(
                    from_gguf(&good[..cut]).is_err(),
                    "truncated at {cut} must error"
                );
            }
        }
    }

    #[test]
    fn from_gguf_reports_bad_fields() {
        let s = |x: &str| MetaValue::String(x.into());
        let check = |bytes: &[u8], why: &str, want: &str, edit: &dyn Fn(&mut Kv, &mut Tensors)| {
            let bad = rebuild(bytes, |kv, ts| edit(kv, ts));
            let e = load_err(&bad);
            assert!(e.contains(want), "{why}: {e:?} lacks {want:?}");
        };
        let (m, n, meta) = bpe_parts(32, 64, true);
        let gpt2 = to_gguf(&m, &n, &meta, QPrec::F32).unwrap();
        let set_tokens = |kv: &mut Kv, f: &dyn Fn(&mut Vec<MetaValue>)| {
            let Some(MetaValue::Array(t)) = kv.get_mut("tokenizer.ggml.tokens") else {
                panic!()
            };
            f(t)
        };
        let cases: Vec<(&str, &str, GgufEdit)> = vec![
            (
                "missing dim",
                "gpt2.embedding_length",
                Box::new(|kv, _| {
                    kv.remove("gpt2.embedding_length");
                }),
            ),
            (
                "dim not an int",
                "gpt2.block_count",
                Box::new(move |kv, _| {
                    kv.insert("gpt2.block_count".into(), s("2"));
                }),
            ),
            (
                "no tokens",
                "tokenizer.ggml.tokens",
                Box::new(|kv, _| {
                    kv.remove("tokenizer.ggml.tokens");
                }),
            ),
            (
                "tokens not strings",
                "tokenizer.ggml.tokens",
                Box::new(move |kv, _| set_tokens(kv, &|t| *t = vec![MetaValue::U32(3)])),
            ),
            (
                "no token types",
                "tokenizer.ggml.token_type",
                Box::new(|kv, _| {
                    kv.remove("tokenizer.ggml.token_type");
                }),
            ),
            (
                "token types not i32",
                "tokenizer.ggml.token_type",
                Box::new(move |kv, _| {
                    kv.insert(
                        "tokenizer.ggml.token_type".into(),
                        MetaValue::Array(vec![s("1")]),
                    );
                }),
            ),
            (
                "type count",
                "token types",
                Box::new(|kv, _| {
                    let Some(MetaValue::Array(t)) = kv.get_mut("tokenizer.ggml.token_type") else {
                        panic!()
                    };
                    t.pop();
                }),
            ),
            (
                "merge result missing",
                "no unique token",
                Box::new(move |kv, _| {
                    let Some(MetaValue::Array(mg)) = kv.get_mut("tokenizer.ggml.merges") else {
                        panic!()
                    };
                    mg[0] = s("q z");
                }),
            ),
            (
                "merge result reused",
                "no unique token",
                Box::new(|kv, _| {
                    let Some(MetaValue::Array(mg)) = kv.get_mut("tokenizer.ggml.merges") else {
                        panic!()
                    };
                    mg[1] = mg[0].clone();
                }),
            ),
            (
                "special not control",
                "not control",
                Box::new(|kv, _| {
                    let Some(MetaValue::Array(t)) = kv.get_mut("tokenizer.ggml.token_type") else {
                        panic!()
                    };
                    *t.last_mut().unwrap() = MetaValue::I32(1);
                }),
            ),
            (
                "missing tensor",
                "missing tensor 'blk.1.ffn_up.bias'",
                Box::new(|_, ts| ts.retain(|t| t.0 != "blk.1.ffn_up.bias")),
            ),
            (
                "wrong dims",
                "position_embd.weight",
                Box::new(|_, ts| {
                    for t in ts.iter_mut().filter(|t| t.0 == "position_embd.weight") {
                        t.1 = vec![t.1[1], t.1[0]];
                    }
                }),
            ),
            (
                "bad metadata",
                "missing task",
                Box::new(move |kv, _| {
                    kv.insert("ferrum.metadata".into(), s("not json"));
                }),
            ),
            (
                "bad normalizer",
                "bad token",
                Box::new(move |kv, _| {
                    kv.insert("ferrum.normalizer".into(), s("x"));
                }),
            ),
            (
                "heads do not divide the width",
                "divisible by num_heads",
                Box::new(|kv, _| {
                    kv.insert("gpt2.attention.head_count".into(), MetaValue::U32(3));
                }),
            ),
        ];
        for (why, want, edit) in &cases {
            check(&gpt2, why, want, edit.as_ref());
        }

        let (model, norm, cmeta) = make_bundle();
        let tab = to_gguf(&model, &norm, &cmeta, QPrec::F32).unwrap();
        let (model, norm, emeta) = make_embedding_bundle_big();
        let big = to_gguf(&model, &norm, &emeta, QPrec::F32).unwrap();
        let generic: Vec<(&[u8], &str, &str, GgufEdit)> = vec![
            (
                &tab,
                "format",
                "ferrum.format",
                Box::new(|kv, _| {
                    kv.insert("ferrum.format".into(), MetaValue::U32(2));
                }),
            ),
            (
                &tab,
                "no metadata",
                "ferrum.metadata",
                Box::new(|kv, _| {
                    kv.remove("ferrum.metadata");
                }),
            ),
            (
                &tab,
                "no layer count",
                "ferrum.layer_count",
                Box::new(|kv, _| {
                    kv.remove("ferrum.layer_count");
                }),
            ),
            (
                &tab,
                "type not a string",
                "ferrum.layer.0.type",
                Box::new(|kv, _| {
                    kv.insert("ferrum.layer.0.type".into(), MetaValue::U32(0));
                }),
            ),
            (
                &tab,
                "unknown type",
                "unknown layer type 'conv'",
                Box::new(move |kv, _| {
                    kv.insert("ferrum.layer.0.type".into(), s("conv"));
                }),
            ),
            (
                &tab,
                "bad activation",
                "bad activation kind 99",
                Box::new(|kv, _| {
                    kv.insert("ferrum.layer.1.kind".into(), MetaValue::U32(99));
                }),
            ),
            (
                &tab,
                "activation > u8",
                "bad activation kind 300",
                Box::new(|kv, _| {
                    kv.insert("ferrum.layer.1.kind".into(), MetaValue::U32(300));
                }),
            ),
            (
                &tab,
                "missing bias",
                "missing tensor 'l2.bias'",
                Box::new(|_, ts| ts.retain(|t| t.0 != "l2.bias")),
            ),
            (
                &big,
                "missing block tensor",
                "missing tensor 'l1.ffn2.weight'",
                Box::new(|_, ts| ts.retain(|t| t.0 != "l1.ffn2.weight")),
            ),
            (
                &big,
                "missing embedding",
                "missing tensor 'l0.pos'",
                Box::new(|_, ts| ts.retain(|t| t.0 != "l0.pos")),
            ),
            (
                &big,
                "missing layernorm",
                "missing tensor 'l2.gamma'",
                Box::new(|_, ts| ts.retain(|t| t.0 != "l2.gamma")),
            ),
            (
                &big,
                "block heads do not divide the width",
                "divisible by num_heads",
                Box::new(|kv, _| {
                    kv.insert("ferrum.layer.1.heads".into(), MetaValue::U32(3));
                }),
            ),
        ];
        for (bytes, why, want, edit) in &generic {
            check(bytes, why, want, edit.as_ref());
        }
    }

    #[test]
    fn loads_foreign_gpt2_gguf() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../.ferrum-dev/gpt2gguf/gpt2-f32.gguf"
        );
        let Ok(bytes) = std::fs::read(path) else {
            eprintln!("skipping loads_foreign_gpt2_gguf: {path} not found");
            return;
        };
        let (m, _, meta) = from_gguf(&bytes).unwrap();
        drop(bytes);
        let slm = GenerativeSLM::new(m, empty_norm(), meta);
        let greedy = crate::slm::SamplingParams {
            top_k: 1,
            ..crate::slm::SamplingParams::with_temperature(1.0)
        };
        let want = " John. I'm a writer";
        let got = slm
            .generate_continuation_with(
                "Hello, my name is",
                want.chars().count(),
                &greedy,
                &mut crate::rng::Rng::new(0),
            )
            .unwrap();
        // What llama.cpp and HF produce greedily (FACTS.md).
        assert_eq!(got, want);
    }
}
