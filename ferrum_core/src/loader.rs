//! Model files: Ferrum models are stored as GGUF (weights + metadata +
//! tokenizer in one file). Weights are dequantized to f32 in memory.
//!
//! A BPE transformer SLM is written in llama.cpp's `gpt2` layout, so
//! llama.cpp / Ollama / LM Studio run it as-is. Every other model uses the
//! generic `ferrum` layout (`ferrum.format = 1`):
//!
//! | Key / tensor | Type | Meaning |
//! |---|---|---|
//! | `general.architecture` | string | `"ferrum"` |
//! | `general.name` | string | `ModelMetadata.dataset_name` |
//! | `ferrum.format` | u32 | `1` |
//! | `ferrum.quant` | string | `"f32"`, `"int8"` or `"int4"` (matrix precision) |
//! | `ferrum.metadata` | string | `ModelMetadata::to_json()` |
//! | `ferrum.normalizer` | string | `Normalizer::encode()` (empty for SLMs) |
//! | `ferrum.layer_count` | u32 | number of layers |
//! | `ferrum.layer.{i}.type` | string | `linear`, `activation`, `flatten`, `embedding`, `layernorm`, `transformer_block` |
//! | `ferrum.layer.{i}.{dim}` | u32 | `linear`: `in`, `out`; `activation`: `kind`; `embedding`: `vocab`, `seq`, `dim`; `layernorm`: `dim`; `transformer_block`: `context`, `heads`, `dim`, `hidden` |
//! | `l{i}.{param}` | F32 / Q8_0 / Q4_0 | GGML dims `[cols, rows]` for a row-major `[rows, cols]` matrix, `[n]` for a vector; 2-D weights quantized at the requested precision when GGML ne0 % 32 == 0, otherwise F32 |
//!
//! Parameter names: `linear`: `weight` `[in,out]`, `bias`; `embedding`:
//! `token`, `pos`; `layernorm`: `gamma`, `beta`; `transformer_block`: `ln1.*`,
//! `q.*`, `k.*`, `v.*`, `out.*`, `ln2.*`, `ffn1.*`, `ffn2.*` (each
//! `weight`/`bias`, or `gamma`/`beta` for the norms).
//!
//! Quantization: matrices whose GGML row length is a multiple of 32 are
//! stored as Q8_0 / Q4_0 when requested; vectors are always F32. The pre-GGUF
//! `.bin` model format is no longer readable.

use crate::activation::Activation;
use crate::error::{InferError, Result};
use crate::gguf::{bounded, transpose_2d, Gguf, MetaValue, GGML_F32, GGML_Q4_0, GGML_Q8_0};
use crate::gguf_write::{encode_tensor, GgufBuilder};
use crate::layer::Layer;
use crate::layer::{ActivationLayer, Embedding, Flatten, LayerNorm, Linear, TransformerBlock};
use crate::meta::TaskType;
use crate::meta::{ModelMetadata, Normalizer};
use crate::model::Sequential;
use crate::quant::q8_0_eligible;
use crate::tokenizer::{byte_to_unicode, ByteBpeTokenizer};
use std::collections::HashMap;

/// Multiply two dimension fields read from an untrusted file, rejecting
/// overflow (relevant on 32-bit/wasm targets where usize is 32 bits).
fn mul_dims(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| InferError::Format(format!("dimension product {a}×{b} overflows")))
}

/// On-disk weight precision selected at save time. In GGUF, `Int8` is Q8_0
/// and `Int4` is Q4_0 (matrices whose GGML row length is a multiple of 32).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QPrec {
    /// Full f32.
    F32,
    /// int8 (GGUF Q8_0).
    Int8,
    /// int4 (GGUF Q4_0, ~8× smaller than f32).
    Int4,
}

/// Serialize with full f32 weights (Ferrum GGUF).
pub fn to_bytes(model: &Sequential, norm: &Normalizer, meta: &ModelMetadata) -> Result<Vec<u8>> {
    to_gguf(model, norm, meta, QPrec::F32)
}

/// Serialize with int8 (Q8_0) matrices, ≈4× smaller.
pub fn to_bytes_quantized(
    model: &Sequential,
    norm: &Normalizer,
    meta: &ModelMetadata,
) -> Result<Vec<u8>> {
    to_gguf(model, norm, meta, QPrec::Int8)
}

/// Serialize with int4 (Q4_0) matrices, ≈8× smaller.
pub fn to_bytes_quantized_int4(
    model: &Sequential,
    norm: &Normalizer,
    meta: &ModelMetadata,
) -> Result<Vec<u8>> {
    to_gguf(model, norm, meta, QPrec::Int4)
}

/// Load a model from GGUF bytes; see [`from_gguf`].
pub fn from_bytes(bytes: &[u8]) -> Result<(Sequential, Normalizer, ModelMetadata)> {
    from_gguf(bytes)
}

pub fn save(model: &Sequential, norm: &Normalizer, meta: &ModelMetadata, path: &str) -> Result<()> {
    std::fs::write(path, to_bytes(model, norm, meta)?)?;
    Ok(())
}

/// Like [`save`] with int8 matrices.
pub fn save_quantized(
    model: &Sequential,
    norm: &Normalizer,
    meta: &ModelMetadata,
    path: &str,
) -> Result<()> {
    std::fs::write(path, to_bytes_quantized(model, norm, meta)?)?;
    Ok(())
}

/// Like [`save`] with int4 matrices.
pub fn save_quantized_int4(
    model: &Sequential,
    norm: &Normalizer,
    meta: &ModelMetadata,
    path: &str,
) -> Result<()> {
    std::fs::write(path, to_bytes_quantized_int4(model, norm, meta)?)?;
    Ok(())
}

pub fn load(path: &str) -> Result<(Sequential, Normalizer, ModelMetadata)> {
    from_gguf(&std::fs::read(path)?)
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
    fn f32(&self, key: &str) -> Result<f32> {
        self.0
            .meta(key)
            .and_then(MetaValue::as_f32)
            .ok_or_else(|| bad(format!("GGUF key '{key}' missing or not a number")))
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
    // llama.cpp requires the key; Ferrum's LayerNorm is fixed at 1e-5.
    let eps = r.f32("gpt2.attention.layer_norm_epsilon")?;
    if (eps - LN_EPS).abs() > 1e-9 {
        return Err(bad(format!(
            "gpt2.attention.layer_norm_epsilon is {eps}; Ferrum's LayerNorm uses {LN_EPS}"
        )));
    }
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
    model.push(Box::new(Linear::without_bias(d, v, head_w)?));
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
                // No tensor carries a block's context, yet it sizes the KV cache
                // (2 · ctx · dim floats): bound ctx · dim by the file. A real
                // model's F32 [dim, ctx] position table alone takes 4 · ctx · dim
                // bytes, so valid files pass.
                bounded(mul_dims(ctx, c)?, r.0.file_len(), "block context × dim")?;
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
    let g = parse_gguf(bytes)?;
    let r = GgufIn(&g);
    match g.architecture() {
        Some("gpt2") => gpt2_from(&r),
        Some("ferrum") => generic_from(&r),
        other => Err(bad(format!(
            "not a Ferrum-loadable model: architecture {other:?} (need gpt2 or ferrum)"
        ))),
    }
}

/// Parse GGUF bytes, with a pointed error for the removed FINF `.bin` format.
fn parse_gguf(bytes: &[u8]) -> Result<Gguf> {
    if bytes.starts_with(b"FINF") {
        return Err(bad(
            "old Ferrum .bin (FINF) model — this format was removed; retrain to produce a .gguf"
                .into(),
        ));
    }
    Gguf::parse(bytes.to_vec())
}

/// The precision a GGUF model was saved at: `ferrum.quant` ("f32" | "int8" |
/// "int4"), else derived from `general.file_type` (0 / 1 / 7 / 2 -> f32 / f16 /
/// int8 / int4). Any other file type is reported as `other(N)`, and a file with
/// neither key as `unknown`, so `info` never refuses a loadable model.
pub fn gguf_quant(bytes: &[u8]) -> Result<String> {
    Ok(quant_of(&parse_gguf(bytes)?))
}

fn quant_of(g: &Gguf) -> String {
    if let Some(q) = g.meta("ferrum.quant").and_then(MetaValue::as_str) {
        return q.to_string();
    }
    match g.meta("general.file_type").and_then(MetaValue::as_usize) {
        Some(0) => "f32".into(),
        Some(7) => "int8".into(),
        Some(1) => "f16".into(),
        Some(2) => "int4".into(),
        Some(ft) => format!("other({ft})"),
        None => "unknown".into(),
    }
}

/// One-line description for `info`: `GGUF v3 (gpt2, int8)`.
pub fn gguf_format(bytes: &[u8]) -> Result<String> {
    let g = parse_gguf(bytes)?;
    Ok(format!(
        "GGUF v{} ({}, {})",
        g.version,
        g.architecture().unwrap_or("unknown"),
        quant_of(&g)
    ))
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
    fn save_and_load_use_gguf_files() {
        let (model, norm, meta) = make_bundle();
        let dir = std::env::temp_dir();
        let path = |n: &str| {
            dir.join(format!("ferrum_loader_{n}_{}.gguf", std::process::id()))
                .to_str()
                .unwrap()
                .to_string()
        };
        type Save = fn(&Sequential, &Normalizer, &ModelMetadata, &str) -> Result<()>;
        let saves: [(&str, Save, &str); 3] = [
            ("f32", save, "f32"),
            ("i8", save_quantized, "int8"),
            ("i4", save_quantized_int4, "int4"),
        ];
        for (name, f, quant) in saves {
            let p = path(name);
            f(&model, &norm, &meta, &p).unwrap();
            let bytes = std::fs::read(&p).unwrap();
            assert_eq!(&bytes[..4], b"GGUF");
            assert_eq!(gguf_quant(&bytes).unwrap(), quant);
            let (m2, _, meta2) = load(&p).unwrap();
            assert_eq!(m2.len(), model.len());
            assert_eq!(meta2.class_names, meta.class_names);
            std::fs::remove_file(&p).unwrap();
        }
        assert!(matches!(load(&path("missing")), Err(InferError::Io(_))));
        assert!(save(&model, &norm, &meta, "/nonexistent-dir/x.gguf").is_err());
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

        // Small model: everything stays f32 and the outputs match closely.
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
    fn int4_roundtrip_is_smaller_and_close() {
        // A Linear big enough to quantize: int4 is far smaller and close.
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
        assert_eq!(&bytes[..4], b"GGUF");
        // ~8× smaller than the f32 file for the weight payload.
        let f32_bytes = to_bytes(&model, &norm, &meta).unwrap();
        assert!(
            (bytes.len() as f32) < (f32_bytes.len() as f32) * 0.4,
            "int4 file {} not far smaller than f32 {}",
            bytes.len(),
            f32_bytes.len()
        );

        let (loaded, _, _) = from_bytes(&bytes).unwrap();
        let y_q = loaded.forward(&x).unwrap();
        let mae: f32 = y_f32
            .data
            .iter()
            .zip(&y_q.data)
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / out_f as f32;
        // 0.4 tolerance: the sinusoid spans ~5 periods per 32-element input
        // block, so every Q4_0 block scale is near the global max and the
        // asymmetric -8..7 grid clips the opposite-sign extreme to 7/8*amax.
        // Measured MAE ~0.27 (old per-row int4: ~0.05); 0.4 still catches sign
        // errors (~1.2) and scrambled weights.
        assert!(mae < 0.4, "int4 forward drifted too far: mae={mae}");
    }

    #[test]
    fn int4_transformer_block_roundtrips_and_runs() {
        let (model, norm, meta) = make_embedding_bundle_big();
        let bytes = to_bytes_quantized_int4(&model, &norm, &meta).unwrap();
        let (loaded, _, _) = from_bytes(&bytes).unwrap();
        // Forward still runs and is finite.
        let x = Tensor::matrix(1, meta.input_dim, vec![1.0; meta.input_dim]).unwrap();
        let y = loaded.forward(&x).unwrap();
        assert!(y.data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn quantized_transformer_roundtrips() {
        // Every SLM layer type through the int8 writer/reader.
        let (model, norm, meta) = make_embedding_bundle();
        let bytes = to_bytes_quantized(&model, &norm, &meta).unwrap();
        assert_eq!(&bytes[..4], b"GGUF");
        let (m2, _, m2meta) = from_bytes(&bytes).unwrap();
        assert_eq!(m2meta.task, TaskType::TransformerSLM);
        let x = Tensor::matrix(1, 4, vec![0.0, 1.0, 2.0, 3.0]).unwrap();
        assert!(m2.forward(&x).is_ok());
    }

    #[test]
    fn flatten_roundtrips() {
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
        assert_eq!(&bytes[..4], b"GGUF");

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
            Box::new(Linear::without_bias(d, v, head).unwrap()),
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
            // The reloaded head is bias-free too, so the parameter count survives.
            assert_eq!(m2.num_params(), m.num_params(), "tie={tie}");
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
        // rotated by one (a permutation that is not its own inverse). The loader must map them
        // back and permute the embedding rows to match.
        let (m, n, meta) = bpe_parts(32, 64, false);
        let bytes = to_gguf(&m, &n, &meta, QPrec::F32).unwrap();
        let d = 32;
        let foreign = rebuild(&bytes, |kv, ts| {
            let MetaValue::Array(tokens) = kv.get_mut("tokenizer.ggml.tokens").unwrap() else {
                panic!()
            };
            tokens[..256].rotate_left(1);
            for (name, _, data) in ts.iter_mut() {
                if name == "token_embd.weight" || name == "output.weight" {
                    let mut rows: Vec<Vec<f32>> = data.chunks(d).map(|r| r.to_vec()).collect();
                    rows[..256].rotate_left(1);
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
        assert_eq!(gguf_quant(&mk(Some(1))).unwrap(), "f16");
        assert_eq!(gguf_quant(&mk(Some(15))).unwrap(), "other(15)");
        assert_eq!(gguf_quant(&mk(None)).unwrap(), "unknown");
        assert!(load_err_of(gguf_quant(b"FINF\x05\x00\x00\x00")).contains(FINF_HINT));
    }

    #[test]
    fn gguf_format_describes_arch_and_quant() {
        let (m, n, meta) = make_bundle();
        let b = to_bytes_quantized(&m, &n, &meta).unwrap();
        assert_eq!(gguf_format(&b).unwrap(), "GGUF v3 (ferrum, int8)");
        let slm = trained_slm(false);
        let b = to_gguf(&slm.model, &slm.norm, &slm.meta, QPrec::F32).unwrap();
        assert_eq!(gguf_format(&b).unwrap(), "GGUF v3 (gpt2, f32)");
        // No architecture key and no quant info.
        let b = GgufBuilder::new().into_bytes();
        assert_eq!(gguf_format(&b).unwrap(), "GGUF v3 (unknown, unknown)");
        assert!(load_err_of(gguf_format(b"FINF\x05\x00\x00\x00")).contains(FINF_HINT));
        // gguf_format derives quant from the same parse: every arm agrees with gguf_quant.
        for ft in [Some(0), Some(7), Some(1), Some(2), Some(15), None] {
            let mut b = GgufBuilder::new();
            b.meta("general.architecture", MetaValue::String("gpt2".into()));
            if let Some(ft) = ft {
                b.meta("general.file_type", MetaValue::U32(ft));
            }
            let bytes = b.into_bytes();
            let q = gguf_quant(&bytes).unwrap();
            assert_eq!(gguf_format(&bytes).unwrap(), format!("GGUF v3 (gpt2, {q})"));
        }
    }

    const FINF_HINT: &str =
        "old Ferrum .bin (FINF) model — this format was removed; retrain to produce a .gguf";

    fn load_err_of<T>(r: Result<T>) -> String {
        match r {
            Ok(_) => panic!("expected error"),
            Err(e) => e.to_string(),
        }
    }

    #[test]
    fn finf_input_gets_removed_format_hint() {
        assert!(load_err_of(from_gguf(b"FINF\x05\x00\x00\x00")).contains(FINF_HINT));
    }

    #[test]
    fn rejects_foreign_and_truncated_gguf() {
        let mut b = GgufBuilder::new();
        b.meta("general.architecture", MetaValue::String("llama".into()));
        assert!(load_err(&b.into_bytes()).contains("not a Ferrum-loadable model"));
        assert!(load_err(&GgufBuilder::new().into_bytes()).contains("not a Ferrum-loadable model"));
        assert!(load_err_of(from_gguf(b"FINF\x05\x00\x00\x00")).contains(FINF_HINT));
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

    /// Two tensors sharing bytes are rejected at parse time for both codecs
    /// (the same file loads fine before the edit).
    #[test]
    fn from_gguf_rejects_overlapping_tensors() {
        let (m, n, meta) = bpe_parts(32, 64, true);
        let gpt2 = to_gguf(&m, &n, &meta, QPrec::F32).unwrap();
        let (model, norm, emeta) = make_embedding_bundle_big();
        let generic = to_gguf(&model, &norm, &emeta, QPrec::F32).unwrap();
        for (bytes, arch_name) in [(gpt2, "gpt2"), (generic, "ferrum")] {
            assert_eq!(arch(&bytes), arch_name);
            from_gguf(&bytes).unwrap();
            let mut bad = bytes.clone();
            crate::gguf::patch_tensor_offset(&mut bad, 1, 0); // onto tensor 0
            let e = load_err(&bad);
            assert!(e.contains("overlap"), "{arch_name}: {e}");
        }
    }

    /// Round 3, finding 2: a foreign gpt2 file with `embedding_length = 0`,
    /// a 2^40 context and zero-size `[0, n]` tensors used to load and then
    /// allocate feature names / ranges by the context (TBs from ~2 KB). Such
    /// tensors are illegal now (only 1-D `[0]` may be empty), so it fails at
    /// parse, fast.
    #[test]
    fn zero_width_gpt2_with_huge_context_fails_fast() {
        let (m, n, meta) = bpe_parts(32, 64, true);
        let gpt2 = to_gguf(&m, &n, &meta, QPrec::F32).unwrap();
        let v = parse(&gpt2).tensor("token_embd.weight").unwrap().dims[1];
        let bad = rebuild(&gpt2, |kv, ts| {
            kv.insert("gpt2.embedding_length".into(), MetaValue::U32(0));
            kv.insert("gpt2.block_count".into(), MetaValue::U32(0));
            kv.insert("gpt2.context_length".into(), MetaValue::U64(1 << 40));
            kv.remove("ferrum.metadata");
            *ts = vec![
                ("token_embd.weight".into(), vec![0, v], vec![]),
                ("position_embd.weight".into(), vec![0, 1 << 40], vec![]),
                ("output_norm.weight".into(), vec![0], vec![]),
                ("output_norm.bias".into(), vec![0], vec![]),
            ];
        });
        let t = std::time::Instant::now();
        let e = load_err(&bad);
        assert!(e.contains("only a 1-D [0]"), "{e}");
        assert!(t.elapsed().as_secs() < 5, "took {:?}", t.elapsed());
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
                "missing epsilon",
                "layer_norm_epsilon",
                Box::new(|kv, _| {
                    kv.remove("gpt2.attention.layer_norm_epsilon");
                }),
            ),
            (
                "epsilon not a number",
                "layer_norm_epsilon",
                Box::new(move |kv, _| {
                    kv.insert("gpt2.attention.layer_norm_epsilon".into(), s("x"));
                }),
            ),
            (
                "epsilon is not Ferrum's",
                "uses 0.00001",
                Box::new(|kv, _| {
                    kv.insert(
                        "gpt2.attention.layer_norm_epsilon".into(),
                        MetaValue::F32(1e-6),
                    );
                }),
            ),
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
            // Round 3: every metadata dim is tied to a (file-bounded) tensor
            // before it sizes anything, so huge values fail on a shape check.
            (
                "huge context",
                "position_embd.weight",
                Box::new(|kv, _| {
                    kv.insert("gpt2.context_length".into(), MetaValue::U64(1 << 40));
                }),
            ),
            (
                "zero width",
                "token_embd.weight",
                Box::new(|kv, _| {
                    kv.insert("gpt2.embedding_length".into(), MetaValue::U32(0));
                }),
            ),
            (
                "huge width",
                "token_embd.weight",
                Box::new(|kv, _| {
                    kv.insert("gpt2.embedding_length".into(), MetaValue::U64(1 << 40));
                }),
            ),
            (
                "huge ffn",
                "ffn_up.weight",
                Box::new(|kv, _| {
                    kv.insert("gpt2.feed_forward_length".into(), MetaValue::U64(1 << 40));
                }),
            ),
            (
                "huge block count",
                "missing tensor 'blk.2.attn_qkv.weight'",
                Box::new(|kv, _| {
                    kv.insert("gpt2.block_count".into(), MetaValue::U64(1 << 40));
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
                "huge block context",
                "block context × dim",
                Box::new(|kv, _| {
                    kv.insert("ferrum.layer.1.context".into(), MetaValue::U64(1 << 40));
                }),
            ),
            (
                &big,
                "block context × dim overflows",
                "overflows",
                Box::new(|kv, _| {
                    kv.insert("ferrum.layer.1.context".into(), MetaValue::U64(u64::MAX));
                }),
            ),
            (
                &big,
                "huge embedding seq",
                "l0.pos",
                Box::new(|kv, _| {
                    kv.insert("ferrum.layer.0.seq".into(), MetaValue::U64(1 << 40));
                }),
            ),
            (
                &tab,
                "huge linear width",
                "l0.weight",
                Box::new(|kv, _| {
                    kv.insert("ferrum.layer.0.out".into(), MetaValue::U64(1 << 40));
                }),
            ),
            (
                &tab,
                "huge layer count",
                "ferrum.layer.4.type",
                Box::new(|kv, _| {
                    kv.insert("ferrum.layer_count".into(), MetaValue::U64(1 << 40));
                }),
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
