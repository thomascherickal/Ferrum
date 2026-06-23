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
use crate::csv::{ModelMetadata, Normalizer};
use crate::error::{InferError, Result};
use crate::layer::{ActivationLayer, Embedding, Flatten, LayerNorm, Linear, TransformerBlock};
use crate::model::Sequential;
use crate::quant::{int8_scale, int8_scales_per_channel, QKind, QWeight, QUANT_MIN_LEN};

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
        Self { bytes: b, pos: 0, v5: false }
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
                if channels == 0 || n % channels != 0 {
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
            m => Err(InferError::Format(format!("bad weight encoding marker {m}"))),
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
            m => Err(InferError::Format(format!("bad weight encoding marker {m}"))),
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
fn linear_from_loaded(in_f: usize, out_f: usize, w: LoadedWeights, bias: Vec<f32>) -> Result<Linear> {
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
        out.extend(std::iter::repeat(0u8).take(data.len()));
    } else {
        for &v in data {
            out.push((v / scale).round().clamp(-127.0, 127.0) as i8 as u8);
        }
    }
}

/// On-disk weight precision selected at save time.
#[derive(Clone, Copy, PartialEq, Eq)]
enum QPrec {
    /// Full f32 (FINF v4, or v5 `ENC_F32`).
    F32,
    /// int8 symmetric, per-channel for matrices (the historical `to_bytes_quantized`).
    Int8,
    /// int4 symmetric, per-channel for matrices (~8× smaller than f32).
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
            if channels > 1 && data.len() % channels == 0 {
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
            let rows = if channels > 0 && data.len() % channels == 0 { channels } else { 1 };
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
    let vec_prec = if prec == QPrec::Int4 { QPrec::Int8 } else { prec };
    let push_mat = |out: &mut Vec<u8>, data: &[f32], channels: usize| {
        push_weights(out, data, channels, v5, prec)
    };
    let push_vec = |out: &mut Vec<u8>, data: &[f32]| push_weights(out, data, 1, v5, vec_prec);
    vprintln!("[loader::to_bytes] Serializing FINF v{} model ({} layers, prec=int{})",
        version, model.len(), match prec { QPrec::F32 => 32, QPrec::Int8 => 8, QPrec::Int4 => 4 });
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    push_u32(&mut out, version);

    // Normalizer (can be empty string for SLM tasks)
    push_str(&mut out, &norm.encode());

    // Metadata JSON
    push_str(&mut out, &meta.to_json());

    push_u32(&mut out, model.len() as u32);
    for (layer_idx, layer) in model.layers().iter().enumerate() {
        vprintln!("[loader::to_bytes]   layer[{}]: {}", layer_idx, layer.name());
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
            push_mat(&mut out, &tb.out_proj.weight_f32(), tb.out_proj.in_features());
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
        Normalizer { means: vec![], stds: vec![] }
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
                vprintln!("[loader::from_bytes]   layer[{}]: Linear({}→{})", layer_i, in_f, out_f);
                let w = r.weights_q(in_f, out_f)?;
                let bias = r.weights(out_f)?;
                model.push(Box::new(linear_from_loaded(in_f, out_f, w, bias)?));
            }
            TAG_ACTIVATION => {
                let t = r.u8()?;
                let act = Activation::from_tag(t)
                    .ok_or_else(|| InferError::Format(format!("bad act tag {t}")))?;
                vprintln!("[loader::from_bytes]   layer[{}]: Activation({:?})", layer_i, act);
                model.push(Box::new(ActivationLayer::new(act)));
            }
            TAG_EMBEDDING => {
                let vocab_size = r.usize()?;
                let max_seq_len = r.usize()?;
                let embedding_dim = r.usize()?;
                vprintln!("[loader::from_bytes]   layer[{}]: Embedding(vocab={}, seq={}, dim={})", layer_i, vocab_size, max_seq_len, embedding_dim);
                // Embeddings stay f32 in memory (lookup + add, not a GEMV), but
                // we still read a quantized table by dequantizing on load.
                let tok = r.weights_q(vocab_size, embedding_dim)?.into_f32();
                let pos = r.weights_q(max_seq_len, embedding_dim)?.into_f32();
                model.push(Box::new(Embedding::new(
                    vocab_size, max_seq_len, embedding_dim, tok, pos,
                )?));
            }
            TAG_LAYERNORM => {
                let dim = r.usize()?;
                vprintln!("[loader::from_bytes]   layer[{}]: LayerNorm(dim={})", layer_i, dim);
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
    vprintln!("[loader::save_quantized] Saving quantized model to: {}", path);
    let bytes = to_bytes_quantized(model, norm, meta)?;
    vprintln!("[loader::save_quantized] Writing {} bytes to disk", bytes.len());
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
// Tests
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::Activation;
    use crate::csv::TaskType;
    use crate::layer::ActivationLayer;
    use crate::tensor::Tensor;

    fn make_bundle() -> (Sequential, Normalizer, ModelMetadata) {
        let l1 = Linear::new(
            4, 8,
            (0..32).map(|i| i as f32 * 0.01).collect(),
            vec![0.0; 8],
        ).unwrap();
        let l2 = Linear::new(
            8, 3,
            (0..24).map(|i| i as f32 * -0.01).collect(),
            vec![0.1; 3],
        ).unwrap();
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
        let f = |n: usize, s: f32| -> Vec<f32> {
            (0..n).map(|i| (i as f32 * s).sin() * 0.2).collect()
        };
        let emb = Embedding::new(vocab, ctx, dim, f(vocab * dim, 0.01), f(ctx * dim, 0.02)).unwrap();
        let tb = TransformerBlock::new(
            ctx, heads, dim,
            vec![1.0; dim], vec![0.0; dim],
            f(dim * dim, 0.013), vec![0.0; dim],
            f(dim * dim, 0.017), vec![0.0; dim],
            f(dim * dim, 0.019), vec![0.0; dim],
            f(dim * dim, 0.023), vec![0.0; dim],
            vec![1.0; dim], vec![0.0; dim],
            f(dim * hidden, 0.007), vec![0.0; hidden],
            f(hidden * dim, 0.009), vec![0.0; dim],
        ).unwrap();
        let lnf = LayerNorm::new(dim, vec![1.0; dim], vec![0.0; dim]).unwrap();
        let head = Linear::new(dim, vocab, f(dim * vocab, 0.005), vec![0.0; vocab]).unwrap();
        let model = Sequential::new()
            .with(Box::new(emb))
            .with(Box::new(tb))
            .with(Box::new(lnf))
            .with(Box::new(head))
            .with(Box::new(ActivationLayer::new(Activation::Softmax)));
        let norm = Normalizer { means: vec![], stds: vec![] };
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
            vocab_size, max_seq_len, embedding_dim,
            (0..vocab_size * embedding_dim).map(|i| i as f32 * 0.1).collect(),
            (0..max_seq_len * embedding_dim).map(|i| i as f32 * 0.01).collect(),
        ).unwrap();
        let lm_head = Linear::new(
            embedding_dim, vocab_size,
            (0..embedding_dim * vocab_size).map(|i| i as f32 * 0.01).collect(),
            vec![0.0; vocab_size],
        ).unwrap();
        let model = Sequential::new()
            .with(Box::new(emb))
            .with(Box::new(ActivationLayer::new(Activation::Softmax)))
            .with(Box::new(lm_head));
        let norm = Normalizer { means: vec![], stds: vec![] };
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
            64, 64,
            (0..64 * 64).map(|i| (i as f32 * 0.37).sin() * 0.1).collect(),
            vec![0.0; 64],
        ).unwrap();
        let big_model = Sequential::new().with(Box::new(big));
        let big_norm = Normalizer { means: vec![], stds: vec![] };
        let big_full = to_bytes(&big_model, &big_norm, &meta).unwrap();
        let big_quant = to_bytes_quantized(&big_model, &big_norm, &meta).unwrap();
        assert!(
            (big_quant.len() as f32) < (big_full.len() as f32) * 0.35,
            "quantized {} not ≈4× smaller than {}", big_quant.len(), big_full.len()
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
        let w: Vec<f32> = (0..in_f * out_f).map(|i| (i as f32 * 0.011).sin() * 0.4).collect();
        let big = Linear::new(in_f, out_f, w, vec![0.0; out_f]).unwrap();
        let model = Sequential::new().with(Box::new(big));
        let norm = Normalizer { means: vec![], stds: vec![] };
        let (_, _, meta) = make_bundle();

        let x = Tensor::matrix(1, in_f, (0..in_f).map(|i| (i as f32 * 0.03).cos()).collect()).unwrap();
        let y_f32 = model.forward(&x).unwrap();

        let bytes = to_bytes_quantized_int4(&model, &norm, &meta).unwrap();
        assert_eq!(&bytes[4..8], &VERSION_QUANT.to_le_bytes());
        // ~8× smaller than the f32 file for the weight payload.
        let f32_bytes = to_bytes(&model, &norm, &meta).unwrap();
        assert!(
            (bytes.len() as f32) < (f32_bytes.len() as f32) * 0.4,
            "int4 file {} not far smaller than f32 {}", bytes.len(), f32_bytes.len()
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
        let mae: f32 = y_f32.data.iter().zip(&y_q.data).map(|(a, b)| (a - b).abs()).sum::<f32>()
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
        let mut data: Vec<f32> = (0..channels * row).map(|i| (i as f32 * 0.05).sin() * 0.3).collect();
        data[2 * row + 1] = 25.0;

        let mut out = Vec::new();
        push_weights(&mut out, &data, channels, /*v5=*/ true, QPrec::Int8);
        assert_eq!(out[0], ENC_INT8_PER_CHANNEL, "matrix should use the per-channel marker");

        let mut r = Reader::new(&out);
        r.v5 = true;
        let decoded = r.weights(data.len()).unwrap();
        assert_eq!(r.pos, out.len(), "reader did not consume the whole vector");

        // Each value is within its own channel's scale/2.
        let scales = int8_scales_per_channel(&data, channels);
        for (i, (o, q)) in data.iter().zip(&decoded).enumerate() {
            assert!((o - q).abs() <= scales[i / row] * 0.5 + 1e-6, "value {i}: {o} vs {q}");
        }
        // The outlier did NOT inflate the clean channels' scale: channel 0 stays
        // accurate (this is the per-channel win over per-tensor).
        let clean_err: f32 = data[..row].iter().zip(&decoded[..row]).map(|(a, b)| (a - b).abs()).sum();
        assert!(clean_err < 0.05, "clean channel error too large: {clean_err}");
    }

    #[test]
    fn one_dimensional_weight_uses_per_tensor_marker() {
        // channels = 1 (a bias-like vector) must fall back to the single-scale
        // int8 marker, not the per-channel one.
        let data: Vec<f32> = (0..QUANT_MIN_LEN).map(|i| (i as f32 * 0.1).cos() * 0.2).collect();
        let mut out = Vec::new();
        push_weights(&mut out, &data, 1, true, QPrec::Int8);
        assert_eq!(out[0], ENC_INT8, "1-D vector should use the per-tensor marker");
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
            8, 3,
            (0..24).map(|i| i as f32 * 0.01).collect(),
            vec![0.0; 3],
        ).unwrap();
        let model = Sequential::new()
            .with(Box::new(Flatten::new()))
            .with(Box::new(lin));
        let norm = Normalizer { means: vec![], stds: vec![] };
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
        fn forward(&self, x: &Tensor) -> Result<Tensor> { Ok(x.clone()) }
        fn name(&self) -> String { "DummyLayer".to_string() }
        fn as_any(&self) -> &dyn std::any::Any { self }
    }

    #[test]
    fn to_bytes_unknown_layer_error() {
        let mut model = Sequential::new();
        model.push(Box::new(DummyLayer));
        let norm = Normalizer { means: vec![], stds: vec![] };
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
        assert!(matches!(to_bytes(&model, &norm, &meta), Err(InferError::Format(_))));
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
            context_len, num_heads, embedding_dim,
            vec![1.0; c], vec![0.0; c],
            vec![0.1; c*c], vec![0.0; c],
            vec![0.1; c*c], vec![0.0; c],
            vec![0.1; c*c], vec![0.0; c],
            vec![0.1; c*c], vec![0.0; c],
            vec![1.0; c], vec![0.0; c],
            vec![0.1; c*h], vec![0.0; h],
            vec![0.1; h*c], vec![0.0; c],
        ).unwrap();

        let mut model = Sequential::new();
        model.push(Box::new(ln));
        model.push(Box::new(tb));

        let norm = Normalizer { means: vec![], stds: vec![] };
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
}

