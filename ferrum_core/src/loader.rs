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
//! one-byte encoding marker — 0 = raw f32, 1 = int8 symmetric quantisation
//! (f32 scale followed by one i8 per value, value = i8 × scale). `to_bytes`
//! writes v4 whenever the model is expressible in it; `to_bytes_quantized`
//! writes v5 with large weight tensors stored int8 (≈4× smaller files).
use crate::activation::Activation;
use crate::csv::{ModelMetadata, Normalizer};
use crate::error::{InferError, Result};
use crate::layer::{ActivationLayer, Embedding, Flatten, LayerNorm, Linear, TransformerBlock};
use crate::model::Sequential;

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
/// Vectors shorter than this stay f32 even in quantized files: biases and
/// LayerNorm parameters are small (no size win) and accuracy-sensitive.
const QUANT_MIN_LEN: usize = 64;

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
            m => Err(InferError::Format(format!("bad weight encoding marker {m}"))),
        }
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
/// Write one weight vector. v4: raw f32. v5: marker byte, then either raw f32
/// or int8 symmetric (f32 scale + i8 per value). Small or non-finite vectors
/// stay f32 even when quantisation is requested.
fn push_weights(out: &mut Vec<u8>, data: &[f32], v5: bool, quantize: bool) {
    if !v5 {
        push_f32s(out, data);
        return;
    }
    let finite = data.iter().all(|v| v.is_finite());
    if quantize && data.len() >= QUANT_MIN_LEN && finite {
        let max_abs = data.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let scale = max_abs / 127.0;
        out.push(ENC_INT8);
        out.extend_from_slice(&scale.to_le_bytes());
        if scale == 0.0 {
            out.extend(std::iter::repeat(0u8).take(data.len()));
        } else {
            for &v in data {
                out.push((v / scale).round().clamp(-127.0, 127.0) as i8 as u8);
            }
        }
    } else {
        out.push(ENC_F32);
        push_f32s(out, data);
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
    to_bytes_impl(model, norm, meta, version, false)
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
    to_bytes_impl(model, norm, meta, VERSION_QUANT, true)
}

fn to_bytes_impl(
    model: &Sequential,
    norm: &Normalizer,
    meta: &ModelMetadata,
    version: u32,
    quantize: bool,
) -> Result<Vec<u8>> {
    let v5 = version == VERSION_QUANT;
    let push_f32s = |out: &mut Vec<u8>, data: &[f32]| push_weights(out, data, v5, quantize);
    vprintln!("[loader::to_bytes] Serializing FINF v{} model ({} layers, quantize={})",
        version, model.len(), quantize);
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
            push_f32s(&mut out, &lin.weight.data);
            push_f32s(&mut out, &lin.bias.data);

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
            push_f32s(&mut out, &emb.token_weight.data);
            push_f32s(&mut out, &emb.pos_weight.data);

        } else if let Some(ln) = any.downcast_ref::<LayerNorm>() {
            out.push(TAG_LAYERNORM);
            push_usize(&mut out, ln.dim());
            push_f32s(&mut out, &ln.gamma.data);
            push_f32s(&mut out, &ln.beta.data);

        } else if let Some(tb) = any.downcast_ref::<TransformerBlock>() {
            out.push(TAG_TRANSFORMER_BLOCK);
            push_usize(&mut out, tb.context_len());
            push_usize(&mut out, tb.num_heads());
            push_usize(&mut out, tb.embedding_dim());
            push_usize(&mut out, tb.hidden_dim());
            // Serialize all projection weights in order: ln1, q, k, v, out, ln2, ffn1, ffn2
            push_f32s(&mut out, &tb.ln1.gamma.data);
            push_f32s(&mut out, &tb.ln1.beta.data);
            push_f32s(&mut out, &tb.q_proj.weight.data);
            push_f32s(&mut out, &tb.q_proj.bias.data);
            push_f32s(&mut out, &tb.k_proj.weight.data);
            push_f32s(&mut out, &tb.k_proj.bias.data);
            push_f32s(&mut out, &tb.v_proj.weight.data);
            push_f32s(&mut out, &tb.v_proj.bias.data);
            push_f32s(&mut out, &tb.out_proj.weight.data);
            push_f32s(&mut out, &tb.out_proj.bias.data);
            push_f32s(&mut out, &tb.ln2.gamma.data);
            push_f32s(&mut out, &tb.ln2.beta.data);
            push_f32s(&mut out, &tb.ffn1.weight.data);
            push_f32s(&mut out, &tb.ffn1.bias.data);
            push_f32s(&mut out, &tb.ffn2.weight.data);
            push_f32s(&mut out, &tb.ffn2.bias.data);

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
                model.push(Box::new(Linear::new(
                    in_f, out_f,
                    r.weights(mul_dims(in_f, out_f)?)?,
                    r.weights(out_f)?,
                )?));
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
                model.push(Box::new(Embedding::new(
                    vocab_size, max_seq_len, embedding_dim,
                    r.weights(mul_dims(vocab_size, embedding_dim)?)?,
                    r.weights(mul_dims(max_seq_len, embedding_dim)?)?,
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
                let cc = mul_dims(c, c)?;
                let ch = mul_dims(c, h)?;
                model.push(Box::new(TransformerBlock::new(
                    context_len, num_heads, embedding_dim,
                    r.weights(c)?,  r.weights(c)?,    // ln1 gamma, beta
                    r.weights(cc)?, r.weights(c)?,    // q weight, bias
                    r.weights(cc)?, r.weights(c)?,    // k weight, bias
                    r.weights(cc)?, r.weights(c)?,    // v weight, bias
                    r.weights(cc)?, r.weights(c)?,    // out weight, bias
                    r.weights(c)?,  r.weights(c)?,    // ln2 gamma, beta
                    r.weights(ch)?, r.weights(h)?,    // ffn1 weight, bias
                    r.weights(ch)?, r.weights(c)?,    // ffn2 weight, bias
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

