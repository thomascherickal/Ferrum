//! Llama/Qwen-family decoder blocks — the architecture modern GGUF checkpoints
//! use, which Ferrum's original [`crate::layer::TransformerBlock`] (learned
//! positions + LayerNorm + ReLU) does not implement. This module adds:
//!
//! * [`RmsNorm`]   — RMS normalization (no mean subtraction, weight-only).
//! * [`apply_rope`] — rotary position embeddings (NORM and NEOX conventions).
//! * [`Attention`] — grouped-query attention (GQA) with RoPE and a KV cache.
//! * [`FeedForward`] — the SwiGLU gated FFN (`down(silu(gate(x)) · up(x))`).
//! * [`LlamaBlock`] — a pre-norm decoder block wiring the two above.
//! * [`LlamaModel`] — token embedding → blocks → final RMSNorm → LM head, with
//!   both a full-sequence forward and an O(context)/token KV-cached path, plus
//!   sampling-based generation.
//!
//! All projections are Ferrum [`Linear`]s, so they inherit the int4/int8
//! in-memory path and the fused/threaded kernels automatically. The
//! [`crate::gguf`] importer maps a GGUF file's tensors onto a [`LlamaModel`].
//!
//! Correctness here is established two ways: each primitive is checked against
//! its closed-form definition, and the cached decode path is checked to match an
//! **independent** full-attention implementation row-for-row (see the tests).
//! Bit-exact parity with llama.cpp on a real checkpoint is *not* claimed — that
//! needs a multi-GB file to run against — but the math and the GGUF tensor
//! mapping are unit-covered.

use crate::error::{InferError, Result};
use crate::layer::{Layer, Linear};
use crate::rng::Rng;
use crate::slm::sample_with_params;
pub use crate::slm::SamplingParams;
use crate::tensor::Tensor;

// ─────────────────────────────────────────────────────────────────────────────
// RMSNorm
// ─────────────────────────────────────────────────────────────────────────────

/// Root-mean-square layer norm: `y = x / sqrt(mean(x²) + eps) · weight`. Unlike
/// [`crate::layer::LayerNorm`] there is no mean subtraction and no bias.
#[derive(Clone, Debug)]
pub struct RmsNorm {
    pub weight: Vec<f32>,
    eps: f32,
}

impl RmsNorm {
    pub fn new(weight: Vec<f32>, eps: f32) -> Self {
        Self { weight, eps }
    }

    pub fn dim(&self) -> usize {
        self.weight.len()
    }

    /// Normalize one row of length `dim` into `out` (length `dim`).
    pub fn forward_row(&self, x: &[f32], out: &mut [f32]) {
        let dim = self.weight.len();
        let ms = x.iter().map(|&v| v * v).sum::<f32>() / dim as f32;
        let inv = 1.0 / (ms + self.eps).sqrt();
        for i in 0..dim {
            out[i] = x[i] * inv * self.weight[i];
        }
    }

    /// Normalize every row of a `[rows, dim]` tensor.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (rows, cols) = x.matrix_dims()?;
        if cols != self.weight.len() {
            return Err(InferError::DimMismatch(format!(
                "RmsNorm expects width {}, got {cols}",
                self.weight.len()
            )));
        }
        let mut out = vec![0.0f32; rows * cols];
        for r in 0..rows {
            let (xs, os) = (
                &x.data[r * cols..(r + 1) * cols],
                &mut out[r * cols..(r + 1) * cols],
            );
            self.forward_row(xs, os);
        }
        Tensor::matrix(rows, cols, out)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Rotary position embeddings (RoPE)
// ─────────────────────────────────────────────────────────────────────────────

/// Which pairing RoPE rotates. Llama/Qwen GGUF checkpoints are converted for
/// [`RopeType::Norm`] (interleaved adjacent pairs); HF in-memory weights use
/// [`RopeType::Neox`] (split halves). The two are not interchangeable for a
/// given set of weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeType {
    /// Rotate interleaved pairs `(2i, 2i+1)` — llama.cpp `NORM`.
    Norm,
    /// Rotate split halves `(i, i + d/2)` — GPT-NeoX / HF `rotate_half`.
    Neox,
}

/// Apply RoPE in place to a `[n_heads · head_dim]` vector at sequence position
/// `pos`. Only the first `rope_dim` (≤ `head_dim`, even) channels of each head
/// are rotated; the rest pass through (partial-rotary models).
pub fn apply_rope(
    x: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    pos: usize,
    base: f32,
    rope_type: RopeType,
) {
    let half = rope_dim / 2;
    for h in 0..n_heads {
        let off = h * head_dim;
        for i in 0..half {
            let freq = (base as f64).powf(-2.0 * i as f64 / rope_dim as f64) as f32;
            let theta = pos as f32 * freq;
            let (s, c) = theta.sin_cos();
            let (ia, ib) = match rope_type {
                RopeType::Norm => (off + 2 * i, off + 2 * i + 1),
                RopeType::Neox => (off + i, off + i + half),
            };
            let a = x[ia];
            let b = x[ib];
            x[ia] = a * c - b * s;
            x[ib] = a * s + b * c;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Grouped-query attention with RoPE and a KV cache
// ─────────────────────────────────────────────────────────────────────────────

/// Per-layer key/value cache: appended-to as tokens stream through.
#[derive(Clone, Debug, Default)]
pub struct KvLayer {
    /// `[len · kv_dim]` row-major (one `kv_dim`-wide row per cached position).
    k: Vec<f32>,
    v: Vec<f32>,
}

impl KvLayer {
    pub fn clear(&mut self) {
        self.k.clear();
        self.v.clear();
    }
}

/// Grouped-query causal self-attention. `n_kv_heads ≤ n_heads`; each KV head is
/// shared by `n_heads / n_kv_heads` query heads.
#[derive(Clone, Debug)]
pub struct Attention {
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    rope_base: f32,
    rope_type: RopeType,
}

impl Attention {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        wq: Linear,
        wk: Linear,
        wv: Linear,
        wo: Linear,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        rope_dim: usize,
        rope_base: f32,
        rope_type: RopeType,
    ) -> Result<Self> {
        if n_kv_heads == 0 || !n_heads.is_multiple_of(n_kv_heads) {
            return Err(InferError::DimMismatch(format!(
                "n_heads {n_heads} must be a multiple of n_kv_heads {n_kv_heads}"
            )));
        }
        if rope_dim > head_dim || !rope_dim.is_multiple_of(2) {
            return Err(InferError::DimMismatch(format!(
                "rope_dim {rope_dim} must be even and ≤ head_dim {head_dim}"
            )));
        }
        Ok(Self {
            wq,
            wk,
            wv,
            wo,
            n_heads,
            n_kv_heads,
            head_dim,
            rope_dim,
            rope_base,
            rope_type,
        })
    }

    fn q_dim(&self) -> usize {
        self.n_heads * self.head_dim
    }
    fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }

    /// Project a `[1, model_dim]` row to RoPE'd q (`[q_dim]`) and k/v (`[kv_dim]`).
    fn project(&self, x: &Tensor, pos: usize) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let mut q = self.wq.forward(x)?.data;
        let mut k = self.wk.forward(x)?.data;
        let v = self.wv.forward(x)?.data;
        apply_rope(
            &mut q,
            self.n_heads,
            self.head_dim,
            self.rope_dim,
            pos,
            self.rope_base,
            self.rope_type,
        );
        apply_rope(
            &mut k,
            self.n_kv_heads,
            self.head_dim,
            self.rope_dim,
            pos,
            self.rope_base,
            self.rope_type,
        );
        Ok((q, k, v))
    }

    /// Attend a single query row `q` (`[q_dim]`) over `len` cached positions in
    /// `kc`/`vc` (`[len · kv_dim]`), writing the `[q_dim]` context to `out`.
    fn attend(&self, q: &[f32], kc: &[f32], vc: &[f32], len: usize, out: &mut [f32]) {
        let hd = self.head_dim;
        let kv_dim = self.kv_dim();
        let group = self.n_heads / self.n_kv_heads;
        let scale = 1.0 / (hd as f32).sqrt();
        let mut scores = vec![0.0f32; len];
        for h in 0..self.n_heads {
            let kvh = h / group;
            let qh = &q[h * hd..h * hd + hd];
            for (j, s) in scores.iter_mut().enumerate() {
                let kj = &kc[j * kv_dim + kvh * hd..j * kv_dim + kvh * hd + hd];
                let mut dot = 0.0f32;
                for d in 0..hd {
                    dot += qh[d] * kj[d];
                }
                *s = dot * scale;
            }
            // softmax over the cached prefix (causal: only ≤ current position)
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - max).exp();
                sum += *s;
            }
            let oh = &mut out[h * hd..h * hd + hd];
            for d in oh.iter_mut() {
                *d = 0.0;
            }
            for (j, &s) in scores.iter().enumerate() {
                let p = s / sum;
                let vj = &vc[j * kv_dim + kvh * hd..j * kv_dim + kvh * hd + hd];
                for d in 0..hd {
                    oh[d] += p * vj[d];
                }
            }
        }
    }

    /// Incremental attention for one new token at position `pos`, appending its
    /// K/V to `cache`. Returns the `[model_dim]` block output.
    pub fn forward_one(&self, x: &[f32], pos: usize, cache: &mut KvLayer) -> Result<Vec<f32>> {
        let dim_in = self.wq.in_features();
        let xt = Tensor::matrix(1, dim_in, x.to_vec())?;
        let (q, k, v) = self.project(&xt, pos)?;
        cache.k.extend_from_slice(&k);
        cache.v.extend_from_slice(&v);
        let len = cache.k.len() / self.kv_dim();
        let mut ctx = vec![0.0f32; self.q_dim()];
        self.attend(&q, &cache.k, &cache.v, len, &mut ctx);
        Ok(self
            .wo
            .forward(&Tensor::matrix(1, self.q_dim(), ctx)?)?
            .data)
    }

    /// Independent full-sequence causal attention over a `[seq, model_dim]`
    /// input. Used for prefill and as the cross-check for [`Self::forward_one`].
    pub fn forward_full(&self, x: &Tensor) -> Result<Tensor> {
        let (seq, dim_in) = x.matrix_dims()?;
        let kv_dim = self.kv_dim();
        let mut kc = vec![0.0f32; seq * kv_dim];
        let mut vc = vec![0.0f32; seq * kv_dim];
        let mut q_all = vec![0.0f32; seq * self.q_dim()];
        for t in 0..seq {
            let row = Tensor::matrix(1, dim_in, x.data[t * dim_in..(t + 1) * dim_in].to_vec())?;
            let (q, k, v) = self.project(&row, t)?;
            q_all[t * self.q_dim()..(t + 1) * self.q_dim()].copy_from_slice(&q);
            kc[t * kv_dim..(t + 1) * kv_dim].copy_from_slice(&k);
            vc[t * kv_dim..(t + 1) * kv_dim].copy_from_slice(&v);
        }
        let mut out = vec![0.0f32; seq * dim_in];
        let mut ctx = vec![0.0f32; self.q_dim()];
        for t in 0..seq {
            // Causal: position t attends to cached positions 0..=t.
            self.attend(
                &q_all[t * self.q_dim()..(t + 1) * self.q_dim()],
                &kc,
                &vc,
                t + 1,
                &mut ctx,
            );
            let o = self
                .wo
                .forward(&Tensor::matrix(1, self.q_dim(), ctx.clone())?)?;
            out[t * dim_in..(t + 1) * dim_in].copy_from_slice(&o.data);
        }
        Tensor::matrix(seq, dim_in, out)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SwiGLU feed-forward
// ─────────────────────────────────────────────────────────────────────────────

/// SwiGLU gated FFN: `down( silu(gate(x)) ⊙ up(x) )`.
#[derive(Clone, Debug)]
pub struct FeedForward {
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
}

impl FeedForward {
    pub fn new(gate: Linear, up: Linear, down: Linear) -> Self {
        Self { gate, up, down }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let g = self.gate.forward(x)?;
        let u = self.up.forward(x)?;
        let h: Vec<f32> = g
            .data
            .iter()
            .zip(&u.data)
            .map(|(&gi, &ui)| (gi / (1.0 + (-gi).exp())) * ui)
            .collect();
        let (rows, ffn) = g.matrix_dims()?;
        self.down.forward(&Tensor::matrix(rows, ffn, h)?)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Decoder block
// ─────────────────────────────────────────────────────────────────────────────

/// A pre-norm Llama/Qwen decoder block:
/// `h = x + attn(rmsnorm(x))`, then `out = h + ffn(rmsnorm(h))`.
#[derive(Clone, Debug)]
pub struct LlamaBlock {
    pub attn_norm: RmsNorm,
    pub attn: Attention,
    pub ffn_norm: RmsNorm,
    pub ffn: FeedForward,
}

impl LlamaBlock {
    /// Incremental one-token forward at position `pos`, updating `cache`.
    pub fn forward_one(&self, x: &[f32], pos: usize, cache: &mut KvLayer) -> Result<Vec<f32>> {
        let dim = x.len();
        let mut normed = vec![0.0f32; dim];
        self.attn_norm.forward_row(x, &mut normed);
        let a = self.attn.forward_one(&normed, pos, cache)?;
        let h: Vec<f32> = (0..dim).map(|i| x[i] + a[i]).collect();
        let mut normed2 = vec![0.0f32; dim];
        self.ffn_norm.forward_row(&h, &mut normed2);
        let f = self.ffn.forward(&Tensor::matrix(1, dim, normed2)?)?;
        Ok((0..dim).map(|i| h[i] + f.data[i]).collect())
    }

    /// Full-sequence forward (prefill / cross-check).
    pub fn forward_full(&self, x: &Tensor) -> Result<Tensor> {
        let (seq, dim) = x.matrix_dims()?;
        let n1 = self.attn_norm.forward(x)?;
        let a = self.attn.forward_full(&n1)?;
        let h = Tensor::matrix(
            seq,
            dim,
            (0..seq * dim).map(|i| x.data[i] + a.data[i]).collect(),
        )?;
        let n2 = self.ffn_norm.forward(&h)?;
        let f = self.ffn.forward(&n2)?;
        Tensor::matrix(
            seq,
            dim,
            (0..seq * dim).map(|i| h.data[i] + f.data[i]).collect(),
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Model
// ─────────────────────────────────────────────────────────────────────────────

/// Shape/hyperparameters of a [`LlamaModel`].
#[derive(Clone, Debug)]
pub struct LlamaConfig {
    pub vocab_size: usize,
    pub model_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
    pub rope_dim: usize,
    pub rope_base: f32,
    pub rope_type: RopeType,
    pub norm_eps: f32,
    pub context_len: usize,
}

/// All layers' KV caches, plus the absolute position of the next token.
pub struct LlamaCache {
    layers: Vec<KvLayer>,
    pos: usize,
}

impl LlamaCache {
    pub fn new(n_layers: usize) -> Self {
        Self {
            layers: vec![KvLayer::default(); n_layers],
            pos: 0,
        }
    }
    pub fn clear(&mut self) {
        for l in &mut self.layers {
            l.clear();
        }
        self.pos = 0;
    }
    pub fn pos(&self) -> usize {
        self.pos
    }
}

/// A runnable Llama/Qwen-family decoder model (standalone — not a [`crate::model::Sequential`]).
pub struct LlamaModel {
    pub cfg: LlamaConfig,
    /// Token embedding table, `[vocab, model_dim]` row-major (kept f32 in memory).
    pub tok_emb: Vec<f32>,
    pub blocks: Vec<LlamaBlock>,
    pub final_norm: RmsNorm,
    /// LM head `[model_dim, vocab]` (may be tied to the embedding).
    pub lm_head: Linear,
}

impl LlamaModel {
    /// The `[model_dim]` embedding of `token`.
    fn embed(&self, token: usize) -> Result<Vec<f32>> {
        let d = self.cfg.model_dim;
        if token >= self.cfg.vocab_size {
            return Err(InferError::DimMismatch(format!(
                "token {token} ≥ vocab_size {}",
                self.cfg.vocab_size
            )));
        }
        Ok(self.tok_emb[token * d..(token + 1) * d].to_vec())
    }

    /// Final hidden state → vocabulary logits (`[vocab]`).
    fn head(&self, h: &[f32]) -> Result<Vec<f32>> {
        let d = self.cfg.model_dim;
        let mut normed = vec![0.0f32; d];
        self.final_norm.forward_row(h, &mut normed);
        Ok(self.lm_head.forward(&Tensor::matrix(1, d, normed)?)?.data)
    }

    /// Feed one token at `cache.pos` through every block (KV-cached), returning
    /// next-token logits and advancing the cache position. This is the
    /// O(context)/token decode path.
    pub fn forward_one(&self, token: usize, cache: &mut LlamaCache) -> Result<Vec<f32>> {
        let pos = cache.pos;
        let mut x = self.embed(token)?;
        for (li, block) in self.blocks.iter().enumerate() {
            x = block.forward_one(&x, pos, &mut cache.layers[li])?;
        }
        cache.pos += 1;
        self.head(&x)
    }

    /// Full-sequence forward over `tokens`, returning `[seq, vocab]` logits.
    /// Independent of the cached path (used for scoring and as its cross-check).
    pub fn forward_tokens(&self, tokens: &[usize]) -> Result<Tensor> {
        let seq = tokens.len();
        let d = self.cfg.model_dim;
        let mut x = vec![0.0f32; seq * d];
        for (t, &tok) in tokens.iter().enumerate() {
            x[t * d..(t + 1) * d].copy_from_slice(&self.embed(tok)?);
        }
        let mut h = Tensor::matrix(seq, d, x)?;
        for block in &self.blocks {
            h = block.forward_full(&h)?;
        }
        let normed = self.final_norm.forward(&h)?;
        self.lm_head.forward(&normed)
    }

    /// Greedy/temperature generation. Primes the cache with `prompt`, then
    /// samples up to `max_new` tokens (stopping at `eos` if given). Returns the
    /// newly generated token IDs (excluding the prompt).
    pub fn generate(
        &self,
        prompt: &[usize],
        max_new: usize,
        params: &SamplingParams,
        eos: Option<usize>,
        rng: &mut Rng,
    ) -> Result<Vec<usize>> {
        if prompt.is_empty() {
            return Err(InferError::DimMismatch("prompt must be non-empty".into()));
        }
        let mut cache = LlamaCache::new(self.blocks.len());
        let mut logits = vec![0.0f32; self.cfg.vocab_size];
        for &tok in prompt {
            logits = self.forward_one(tok, &mut cache)?;
        }
        let mut out = Vec::new();
        let mut recent: Vec<usize> = prompt.to_vec();
        for _ in 0..max_new {
            let next = sample_with_params(&logits, params, &recent, rng);
            out.push(next);
            if Some(next) == eos {
                break;
            }
            recent.push(next);
            logits = self.forward_one(next, &mut cache)?;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(n: usize, seed: u64) -> Vec<f32> {
        let mut x = seed | 1;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x >> 40) as f32 / (1u64 << 23) as f32 - 1.0
            })
            .collect()
    }

    fn lin(in_f: usize, out_f: usize, seed: u64) -> Linear {
        Linear::new(in_f, out_f, det(in_f * out_f, seed), vec![0.0; out_f]).unwrap()
    }

    // ── RMSNorm ───────────────────────────────────────────────────────────────

    #[test]
    fn rmsnorm_matches_formula() {
        let w = vec![2.0, 0.5, 1.0, 1.5];
        let rn = RmsNorm::new(w.clone(), 1e-6);
        let x = vec![1.0, -2.0, 3.0, 0.5];
        let mut out = vec![0.0; 4];
        rn.forward_row(&x, &mut out);
        let ms = x.iter().map(|v| v * v).sum::<f32>() / 4.0;
        let inv = 1.0 / (ms + 1e-6).sqrt();
        for i in 0..4 {
            assert!((out[i] - x[i] * inv * w[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn rmsnorm_unit_weight_gives_unit_rms() {
        let rn = RmsNorm::new(vec![1.0; 8], 1e-6);
        let x = det(8, 5);
        let mut out = vec![0.0; 8];
        rn.forward_row(&x, &mut out);
        let rms = (out.iter().map(|v| v * v).sum::<f32>() / 8.0).sqrt();
        assert!((rms - 1.0).abs() < 1e-3, "rms={rms}");
    }

    // ── RoPE ──────────────────────────────────────────────────────────────────

    #[test]
    fn rope_at_position_zero_is_identity() {
        for rt in [RopeType::Norm, RopeType::Neox] {
            let mut x = det(16, 9);
            let orig = x.clone();
            apply_rope(&mut x, 2, 8, 8, 0, 10000.0, rt);
            for (a, b) in x.iter().zip(&orig) {
                assert!((a - b).abs() < 1e-6, "rope@0 changed values for {rt:?}");
            }
        }
    }

    #[test]
    fn rope_preserves_per_pair_norm() {
        // Each 2-D rotation is orthogonal, so per-head norm is invariant.
        for rt in [RopeType::Norm, RopeType::Neox] {
            let mut x = det(8, 3);
            let n0 = x.iter().map(|v| v * v).sum::<f32>();
            apply_rope(&mut x, 1, 8, 8, 5, 10000.0, rt);
            let n1 = x.iter().map(|v| v * v).sum::<f32>();
            assert!(
                (n0 - n1).abs() < 1e-4,
                "rope changed norm for {rt:?}: {n0} vs {n1}"
            );
        }
    }

    #[test]
    fn rope_norm_and_neox_differ() {
        let base = det(8, 7);
        let mut a = base.clone();
        let mut b = base.clone();
        apply_rope(&mut a, 1, 8, 8, 3, 10000.0, RopeType::Norm);
        apply_rope(&mut b, 1, 8, 8, 3, 10000.0, RopeType::Neox);
        assert!(a.iter().zip(&b).any(|(x, y)| (x - y).abs() > 1e-3));
    }

    // ── Attention ─────────────────────────────────────────────────────────────

    fn attention(dim: usize, n_heads: usize, n_kv: usize, hd: usize, seed: u64) -> Attention {
        let qd = n_heads * hd;
        let kvd = n_kv * hd;
        Attention::new(
            lin(dim, qd, seed),
            lin(dim, kvd, seed + 1),
            lin(dim, kvd, seed + 2),
            lin(qd, dim, seed + 3),
            n_heads,
            n_kv,
            hd,
            hd,
            10000.0,
            RopeType::Norm,
        )
        .unwrap()
    }

    #[test]
    fn cached_attention_matches_full_mha() {
        // Multi-head (no GQA): the looped incremental path must equal the
        // independent full causal attention row for row.
        let (dim, seq) = (16, 5);
        let attn = attention(dim, 4, 4, 4, 11);
        let x = Tensor::matrix(seq, dim, det(seq * dim, 21)).unwrap();
        let full = attn.forward_full(&x).unwrap();

        let mut cache = KvLayer::default();
        for t in 0..seq {
            let row = &x.data[t * dim..(t + 1) * dim];
            let inc = attn.forward_one(row, t, &mut cache).unwrap();
            for (d, &got) in inc.iter().enumerate() {
                assert!(
                    (got - full.data[t * dim + d]).abs() < 1e-4,
                    "row {t} dim {d}: cached {got} vs full {}",
                    full.data[t * dim + d]
                );
            }
        }
    }

    #[test]
    fn cached_attention_matches_full_gqa() {
        // Grouped-query attention: 6 query heads share 2 KV heads (group=3).
        let (dim, seq) = (24, 6);
        let attn = attention(dim, 6, 2, 4, 31);
        let x = Tensor::matrix(seq, dim, det(seq * dim, 41)).unwrap();
        let full = attn.forward_full(&x).unwrap();
        let mut cache = KvLayer::default();
        for t in 0..seq {
            let inc = attn
                .forward_one(&x.data[t * dim..(t + 1) * dim], t, &mut cache)
                .unwrap();
            for (d, &got) in inc.iter().enumerate() {
                assert!((got - full.data[t * dim + d]).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn attention_rejects_bad_gqa() {
        // 5 query heads cannot evenly share 2 KV heads.
        let r = Attention::new(
            lin(8, 20, 1),
            lin(8, 8, 2),
            lin(8, 8, 3),
            lin(20, 8, 4),
            5,
            2,
            4,
            4,
            10000.0,
            RopeType::Norm,
        );
        assert!(r.is_err());
    }

    // ── FeedForward (SwiGLU) ──────────────────────────────────────────────────

    #[test]
    fn swiglu_matches_manual() {
        let (dim, ffn) = (4, 6);
        let ff = FeedForward::new(lin(dim, ffn, 1), lin(dim, ffn, 2), lin(ffn, dim, 3));
        let x = Tensor::matrix(1, dim, det(dim, 9)).unwrap();
        let got = ff.forward(&x).unwrap();

        let g = ff.gate.forward(&x).unwrap();
        let u = ff.up.forward(&x).unwrap();
        let h: Vec<f32> = (0..ffn)
            .map(|i| (g.data[i] / (1.0 + (-g.data[i]).exp())) * u.data[i])
            .collect();
        let want = ff
            .down
            .forward(&Tensor::matrix(1, ffn, h).unwrap())
            .unwrap();
        for (a, b) in got.data.iter().zip(&want.data) {
            assert!((a - b).abs() < 1e-6);
        }
        assert_eq!(got.shape, vec![1, dim]);
    }

    // ── Block + Model ─────────────────────────────────────────────────────────

    fn block(
        dim: usize,
        n_heads: usize,
        n_kv: usize,
        hd: usize,
        ffn: usize,
        seed: u64,
    ) -> LlamaBlock {
        LlamaBlock {
            attn_norm: RmsNorm::new(vec![1.0; dim], 1e-6),
            attn: attention(dim, n_heads, n_kv, hd, seed),
            ffn_norm: RmsNorm::new(vec![1.0; dim], 1e-6),
            ffn: FeedForward::new(
                lin(dim, ffn, seed + 10),
                lin(dim, ffn, seed + 11),
                lin(ffn, dim, seed + 12),
            ),
        }
    }

    #[test]
    fn block_cached_matches_full() {
        let (dim, seq) = (16, 5);
        let b = block(dim, 4, 2, 4, 32, 51);
        let x = Tensor::matrix(seq, dim, det(seq * dim, 61)).unwrap();
        let full = b.forward_full(&x).unwrap();
        let mut cache = KvLayer::default();
        for t in 0..seq {
            let inc = b
                .forward_one(&x.data[t * dim..(t + 1) * dim], t, &mut cache)
                .unwrap();
            for (d, &got) in inc.iter().enumerate() {
                assert!(
                    (got - full.data[t * dim + d]).abs() < 1e-4,
                    "block row {t} dim {d}"
                );
            }
        }
    }

    fn tiny_model() -> LlamaModel {
        let (vocab, dim, hd, ffn) = (12, 16, 4, 32);
        let cfg = LlamaConfig {
            vocab_size: vocab,
            model_dim: dim,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: hd,
            ffn_dim: ffn,
            rope_dim: hd,
            rope_base: 10000.0,
            rope_type: RopeType::Norm,
            norm_eps: 1e-6,
            context_len: 32,
        };
        LlamaModel {
            cfg,
            tok_emb: det(vocab * dim, 71),
            blocks: vec![block(dim, 4, 2, hd, ffn, 81), block(dim, 4, 2, hd, ffn, 91)],
            final_norm: RmsNorm::new(vec![1.0; dim], 1e-6),
            lm_head: lin(dim, vocab, 101),
        }
    }

    #[test]
    fn model_cached_decode_matches_full_forward() {
        // End-to-end: feeding tokens one at a time (KV-cached) must reproduce the
        // last-row logits of the independent full forward at each step.
        let model = tiny_model();
        let tokens = [3usize, 7, 1, 9, 4, 0];
        let full = model.forward_tokens(&tokens).unwrap();
        let (_, vocab) = full.matrix_dims().unwrap();

        let mut cache = LlamaCache::new(model.blocks.len());
        for (t, &tok) in tokens.iter().enumerate() {
            let logits = model.forward_one(tok, &mut cache).unwrap();
            let row = &full.data[t * vocab..(t + 1) * vocab];
            for v in 0..vocab {
                assert!(
                    (logits[v] - row[v]).abs() < 1e-3,
                    "step {t} vocab {v}: cached {} vs full {}",
                    logits[v],
                    row[v]
                );
            }
        }
        assert_eq!(cache.pos(), tokens.len());
    }

    #[test]
    fn generate_is_deterministic_and_respects_eos() {
        let model = tiny_model();
        let params = SamplingParams::with_temperature(0.8);
        let a = model
            .generate(&[1, 2, 3], 10, &params, None, &mut Rng::new(7))
            .unwrap();
        let b = model
            .generate(&[1, 2, 3], 10, &params, None, &mut Rng::new(7))
            .unwrap();
        assert_eq!(a, b, "generation must be deterministic for a fixed seed");
        assert_eq!(a.len(), 10);
        assert!(a.iter().all(|&t| t < model.cfg.vocab_size));

        // Greedy with an eos that will be hit stops early (temperature→0 ≈ argmax).
        let greedy = SamplingParams::with_temperature(0.01);
        let first = model
            .generate(&[1, 2, 3], 1, &greedy, None, &mut Rng::new(1))
            .unwrap()[0];
        let stopped = model
            .generate(&[1, 2, 3], 10, &greedy, Some(first), &mut Rng::new(1))
            .unwrap();
        assert_eq!(
            stopped,
            vec![first],
            "should stop at eos on the first token"
        );
    }

    #[test]
    fn embed_rejects_out_of_range_token() {
        let model = tiny_model();
        assert!(model.embed(999).is_err());
    }

    #[test]
    fn rmsnorm_forward_multirow_and_width_check() {
        let rn = RmsNorm::new(vec![1.0; 4], 1e-6);
        let x = Tensor::matrix(2, 4, det(8, 1)).unwrap();
        let y = rn.forward(&x).unwrap();
        assert_eq!(y.shape, vec![2, 4]);
        // Wrong width is rejected.
        let bad = Tensor::matrix(1, 3, vec![0.0; 3]).unwrap();
        assert!(rn.forward(&bad).is_err());
    }

    #[test]
    fn attention_rejects_odd_or_oversized_rope_dim() {
        // rope_dim must be even and ≤ head_dim.
        assert!(Attention::new(
            lin(8, 8, 1),
            lin(8, 8, 2),
            lin(8, 8, 3),
            lin(8, 8, 4),
            2,
            2,
            4,
            5,
            10000.0,
            RopeType::Norm,
        )
        .is_err());
        assert!(Attention::new(
            lin(8, 8, 1),
            lin(8, 8, 2),
            lin(8, 8, 3),
            lin(8, 8, 4),
            2,
            2,
            4,
            6,
            10000.0,
            RopeType::Norm,
        )
        .is_err());
    }

    #[test]
    fn generate_rejects_empty_prompt() {
        let model = tiny_model();
        let p = SamplingParams::with_temperature(1.0);
        assert!(model.generate(&[], 4, &p, None, &mut Rng::new(1)).is_err());
    }

    #[test]
    fn cache_clear_resets_state() {
        let model = tiny_model();
        let mut cache = LlamaCache::new(model.blocks.len());
        model.forward_one(1, &mut cache).unwrap();
        model.forward_one(2, &mut cache).unwrap();
        assert_eq!(cache.pos(), 2);
        cache.clear();
        assert_eq!(cache.pos(), 0);
        // After clear, decoding starts fresh and equals a single-token forward.
        let a = model.forward_one(3, &mut cache).unwrap();
        let mut fresh = LlamaCache::new(model.blocks.len());
        let b = model.forward_one(3, &mut fresh).unwrap();
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-6);
        }
    }
}
