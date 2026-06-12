//! Trainable decoder-only causal Transformer: full backprop through token +
//! positional embeddings, LayerNorm, multi-head self-attention, and the FFN,
//! optimised with Adam. Mirrors the `train.rs` pattern: a dedicated trainable
//! struct (`TransformerNet`) that exports to an inference `Sequential` whose
//! layers (`Embedding`, `TransformerBlock`, `LayerNorm`, `Linear`) serialize
//! to FINF and run in WASM.

use crate::error::{InferError, Result};
use crate::layer::{
    matmul_naive_helper, matmul_transpose_b_helper, ActivationLayer, Embedding, LayerNorm,
    Linear, TransformerBlock,
};
use crate::model::Sequential;
use crate::ops;
use crate::optim::Adam;
use crate::rng::Rng;
use crate::tensor::Tensor;
use crate::verbose;

const LN_EPS: f32 = 1e-5; // must match layer::LayerNorm

// ─────────────────────────────────────────────────────────────────────────────
// Parameter with gradient and Adam moment buffers
// ─────────────────────────────────────────────────────────────────────────────

struct Param {
    data: Tensor,
    grad: Tensor,
    m: Tensor,
    v: Tensor,
}

impl Param {
    fn new(shape: Vec<usize>, init: Vec<f32>) -> Result<Self> {
        Ok(Self {
            grad: Tensor::zeros(shape.clone()),
            m: Tensor::zeros(shape.clone()),
            v: Tensor::zeros(shape.clone()),
            data: Tensor::new(shape, init)?,
        })
    }
    fn randn(shape: Vec<usize>, scale: f32, rng: &mut Rng) -> Self {
        let n: usize = shape.iter().product();
        let init: Vec<f32> = (0..n).map(|_| rng.next_normal() * scale).collect();
        Self::new(shape, init).unwrap()
    }
    fn constant(shape: Vec<usize>, value: f32) -> Self {
        let n: usize = shape.iter().product();
        Self::new(shape, vec![value; n]).unwrap()
    }
    fn zero_grad(&mut self) {
        for g in &mut self.grad.data {
            *g = 0.0;
        }
    }
    fn accumulate(&mut self, g: &Tensor) {
        for (a, b) in self.grad.data.iter_mut().zip(&g.data) {
            *a += b;
        }
    }
    fn step(&mut self, adam: &Adam, t: u64) -> Result<()> {
        adam.step(t, &mut self.data, &self.grad, &mut self.m, &mut self.v)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Differentiable building blocks
// ─────────────────────────────────────────────────────────────────────────────

fn linear_fwd(x: &Tensor, w: &Tensor, b: &Tensor) -> Result<Tensor> {
    ops::add_bias(&ops::matmul(x, w)?, b)
}

/// Returns (dW, db, dx) for y = xW + b given dy.
fn linear_bwd(x: &Tensor, w: &Tensor, dy: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
    let dw = ops::matmul(&ops::transpose(x)?, dy)?;
    let db = ops::sum_axis0(dy)?;
    let dx = ops::matmul(dy, &ops::transpose(w)?)?;
    Ok((dw, db, dx))
}

/// LayerNorm forward. Returns (y, x̂, inv_std) — x̂ and inv_std are needed by
/// the backward pass.
fn ln_fwd(x: &Tensor, g: &Tensor, b: &Tensor) -> Result<(Tensor, Tensor, Vec<f32>)> {
    let (rows, cols) = x.matrix_dims()?;
    let mut y = vec![0.0f32; rows * cols];
    let mut xhat = vec![0.0f32; rows * cols];
    let mut inv_std = vec![0.0f32; rows];
    for r in 0..rows {
        let base = r * cols;
        let row = &x.data[base..base + cols];
        let mean = row.iter().sum::<f32>() / cols as f32;
        let var = row.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / cols as f32;
        let is = 1.0 / (var + LN_EPS).sqrt();
        inv_std[r] = is;
        for c in 0..cols {
            let xh = (row[c] - mean) * is;
            xhat[base + c] = xh;
            y[base + c] = xh * g.data[c] + b.data[c];
        }
    }
    Ok((
        Tensor::matrix(rows, cols, y)?,
        Tensor::matrix(rows, cols, xhat)?,
        inv_std,
    ))
}

/// LayerNorm backward. Returns (dx, dγ, dβ).
fn ln_bwd(
    dy: &Tensor,
    xhat: &Tensor,
    inv_std: &[f32],
    g: &Tensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    let (rows, cols) = dy.matrix_dims()?;
    let n = cols as f32;
    let mut dx = vec![0.0f32; rows * cols];
    let mut dg = vec![0.0f32; cols];
    let mut db = vec![0.0f32; cols];
    for r in 0..rows {
        let base = r * cols;
        let mut sum_dxhat = 0.0f32;
        let mut sum_dxhat_xhat = 0.0f32;
        for c in 0..cols {
            let dyv = dy.data[base + c];
            let xh = xhat.data[base + c];
            dg[c] += dyv * xh;
            db[c] += dyv;
            let dxh = dyv * g.data[c];
            sum_dxhat += dxh;
            sum_dxhat_xhat += dxh * xh;
        }
        let mean_dxhat = sum_dxhat / n;
        let mean_dxhat_xhat = sum_dxhat_xhat / n;
        for c in 0..cols {
            let dxh = dy.data[base + c] * g.data[c];
            dx[base + c] =
                inv_std[r] * (dxh - mean_dxhat - xhat.data[base + c] * mean_dxhat_xhat);
        }
    }
    Ok((
        Tensor::matrix(rows, cols, dx)?,
        Tensor::vector(dg),
        Tensor::vector(db),
    ))
}

/// out [m,n] = aᵀ·b where a is [k,m] and b is [k,n].
fn matmul_transpose_a_helper(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for r in 0..k {
        let a_row = r * m;
        let b_row = r * n;
        for i in 0..m {
            let a_ri = a[a_row + i];
            let o_row = i * n;
            for j in 0..n {
                out[o_row + j] += a_ri * b[b_row + j];
            }
        }
    }
    out
}

/// Causal multi-head attention forward over [M=B·T, C] projections.
/// Returns (concatenated head outputs [M, C], softmax probs [B·H·T·T]).
fn attn_fwd(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    batch: usize,
    t: usize,
    heads: usize,
) -> Result<(Tensor, Vec<f32>)> {
    let (m, c) = q.matrix_dims()?;
    let head_dim = c / heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0.0f32; m * c];
    let mut probs = vec![0.0f32; batch * heads * t * t];

    let mut q_h = vec![0.0f32; t * head_dim];
    let mut k_h = vec![0.0f32; t * head_dim];
    let mut v_h = vec![0.0f32; t * head_dim];
    for b in 0..batch {
        for h in 0..heads {
            let hs = h * head_dim;
            for r in 0..t {
                let src = (b * t + r) * c + hs;
                let dst = r * head_dim;
                q_h[dst..dst + head_dim].copy_from_slice(&q.data[src..src + head_dim]);
                k_h[dst..dst + head_dim].copy_from_slice(&k.data[src..src + head_dim]);
                v_h[dst..dst + head_dim].copy_from_slice(&v.data[src..src + head_dim]);
            }
            // S = scale·QKᵀ with causal mask, then row-softmax.
            let mut s = matmul_transpose_b_helper(&q_h, &k_h, t, t, head_dim);
            for i in 0..t {
                for j in 0..t {
                    let idx = i * t + j;
                    s[idx] *= scale;
                    if j > i {
                        s[idx] = -1e9;
                    }
                }
            }
            for i in 0..t {
                let row = &mut s[i * t..(i + 1) * t];
                let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for val in row.iter_mut() {
                    *val = (*val - max).exp();
                    sum += *val;
                }
                for val in row.iter_mut() {
                    *val /= sum;
                }
            }
            let p_base = (b * heads + h) * t * t;
            probs[p_base..p_base + t * t].copy_from_slice(&s);
            // O = P·V
            let o = matmul_naive_helper(&s, &v_h, t, head_dim, t);
            for r in 0..t {
                let dst = (b * t + r) * c + hs;
                let src = r * head_dim;
                out[dst..dst + head_dim].copy_from_slice(&o[src..src + head_dim]);
            }
        }
    }
    Ok((Tensor::matrix(m, c, out)?, probs))
}

/// Backward through causal multi-head attention. Returns (dq, dk, dv), each [M, C].
fn attn_bwd(
    d_out: &Tensor,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    probs: &[f32],
    batch: usize,
    t: usize,
    heads: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    let (m, c) = q.matrix_dims()?;
    let head_dim = c / heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut dq = vec![0.0f32; m * c];
    let mut dk = vec![0.0f32; m * c];
    let mut dv = vec![0.0f32; m * c];

    let mut q_h = vec![0.0f32; t * head_dim];
    let mut k_h = vec![0.0f32; t * head_dim];
    let mut v_h = vec![0.0f32; t * head_dim];
    let mut do_h = vec![0.0f32; t * head_dim];
    for b in 0..batch {
        for h in 0..heads {
            let hs = h * head_dim;
            for r in 0..t {
                let src = (b * t + r) * c + hs;
                let dst = r * head_dim;
                q_h[dst..dst + head_dim].copy_from_slice(&q.data[src..src + head_dim]);
                k_h[dst..dst + head_dim].copy_from_slice(&k.data[src..src + head_dim]);
                v_h[dst..dst + head_dim].copy_from_slice(&v.data[src..src + head_dim]);
                do_h[dst..dst + head_dim].copy_from_slice(&d_out.data[src..src + head_dim]);
            }
            let p = &probs[(b * heads + h) * t * t..(b * heads + h + 1) * t * t];

            // dV = Pᵀ·dO
            let dv_h = matmul_transpose_a_helper(p, &do_h, t, head_dim, t);
            // dP = dO·Vᵀ
            let dp = matmul_transpose_b_helper(&do_h, &v_h, t, t, head_dim);
            // Softmax backward: dS_ij = P_ij·(dP_ij − Σ_k dP_ik·P_ik).
            // Masked entries have P = 0, so dS = 0 there automatically.
            let mut ds = vec![0.0f32; t * t];
            for i in 0..t {
                let base = i * t;
                let row_dot: f32 = (0..t).map(|j| dp[base + j] * p[base + j]).sum();
                for j in 0..t {
                    ds[base + j] = p[base + j] * (dp[base + j] - row_dot) * scale;
                }
            }
            // dQ = dS·K ;  dK = dSᵀ·Q
            let dq_h = matmul_naive_helper(&ds, &k_h, t, head_dim, t);
            let dk_h = matmul_transpose_a_helper(&ds, &q_h, t, head_dim, t);

            for r in 0..t {
                let dst = (b * t + r) * c + hs;
                let src = r * head_dim;
                dq[dst..dst + head_dim].copy_from_slice(&dq_h[src..src + head_dim]);
                dk[dst..dst + head_dim].copy_from_slice(&dk_h[src..src + head_dim]);
                dv[dst..dst + head_dim].copy_from_slice(&dv_h[src..src + head_dim]);
            }
        }
    }
    Ok((
        Tensor::matrix(m, c, dq)?,
        Tensor::matrix(m, c, dk)?,
        Tensor::matrix(m, c, dv)?,
    ))
}

// ─────────────────────────────────────────────────────────────────────────────
// Trainable block + forward cache
// ─────────────────────────────────────────────────────────────────────────────

struct BlockT {
    ln1_g: Param,
    ln1_b: Param,
    q_w: Param,
    q_b: Param,
    k_w: Param,
    k_b: Param,
    v_w: Param,
    v_b: Param,
    o_w: Param,
    o_b: Param,
    ln2_g: Param,
    ln2_b: Param,
    f1_w: Param,
    f1_b: Param,
    f2_w: Param,
    f2_b: Param,
}

impl BlockT {
    fn new(c: usize, hidden: usize, scale: f32, rng: &mut Rng) -> Self {
        Self {
            ln1_g: Param::constant(vec![c], 1.0),
            ln1_b: Param::constant(vec![c], 0.0),
            q_w: Param::randn(vec![c, c], scale, rng),
            q_b: Param::constant(vec![c], 0.0),
            k_w: Param::randn(vec![c, c], scale, rng),
            k_b: Param::constant(vec![c], 0.0),
            v_w: Param::randn(vec![c, c], scale, rng),
            v_b: Param::constant(vec![c], 0.0),
            o_w: Param::randn(vec![c, c], scale, rng),
            o_b: Param::constant(vec![c], 0.0),
            ln2_g: Param::constant(vec![c], 1.0),
            ln2_b: Param::constant(vec![c], 0.0),
            f1_w: Param::randn(vec![c, hidden], scale, rng),
            f1_b: Param::constant(vec![hidden], 0.0),
            f2_w: Param::randn(vec![hidden, c], scale, rng),
            f2_b: Param::constant(vec![c], 0.0),
        }
    }

    fn params_mut(&mut self) -> [&mut Param; 16] {
        [
            &mut self.ln1_g, &mut self.ln1_b,
            &mut self.q_w, &mut self.q_b,
            &mut self.k_w, &mut self.k_b,
            &mut self.v_w, &mut self.v_b,
            &mut self.o_w, &mut self.o_b,
            &mut self.ln2_g, &mut self.ln2_b,
            &mut self.f1_w, &mut self.f1_b,
            &mut self.f2_w, &mut self.f2_b,
        ]
    }
}

struct BlockCache {
    xhat1: Tensor,
    inv_std1: Vec<f32>,
    norm1: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    probs: Vec<f32>,  // [B·H·T·T]
    concat: Tensor,   // attention head outputs before out_proj [M, C]
    xhat2: Tensor,
    inv_std2: Vec<f32>,
    norm2: Tensor,
    h_relu: Tensor,   // FFN hidden after ReLU [M, hidden]
}

/// Opaque record of one forward pass, consumed by `backward`.
pub struct FwdCache {
    tokens: Vec<usize>,
    batch: usize,
    blocks: Vec<BlockCache>,
    xhat_f: Tensor,
    inv_std_f: Vec<f32>,
    norm_f: Tensor, // input to the LM head
}

// ─────────────────────────────────────────────────────────────────────────────
// TransformerNet
// ─────────────────────────────────────────────────────────────────────────────

/// A trainable decoder-only causal Transformer:
/// `Embedding → [TransformerBlock × N] → LayerNorm → Linear(vocab)`.
///
/// Train with `forward` / `backward` / `step` (Adam), then `to_inference()`
/// to obtain a `Sequential` that serializes to FINF and runs anywhere
/// (including WASM with KV-cached generation).
pub struct TransformerNet {
    vocab_size: usize,
    context_len: usize,
    embed_dim: usize,
    num_heads: usize,
    tok_emb: Param, // [vocab, C]
    pos_emb: Param, // [T, C]
    blocks: Vec<BlockT>,
    lnf_g: Param,
    lnf_b: Param,
    head_w: Param, // [C, vocab]
    head_b: Param, // [vocab]
    step_t: u64,
    /// Quantization Aware Training: when enabled, `train_transformer_epoch`
    /// runs forward/backward against int8-snapped weights while Adam updates
    /// full-precision master weights (straight-through estimator).
    qat: bool,
}

impl TransformerNet {
    pub fn new(
        vocab_size: usize,
        context_len: usize,
        embed_dim: usize,
        num_heads: usize,
        hidden_dim: usize,
        num_blocks: usize,
        rng: &mut Rng,
    ) -> Result<Self> {
        if num_heads == 0 || embed_dim % num_heads != 0 {
            return Err(InferError::DimMismatch(format!(
                "embedding_dim {embed_dim} must be divisible by num_heads {num_heads}"
            )));
        }
        if vocab_size == 0 || context_len == 0 || num_blocks == 0 {
            return Err(InferError::DimMismatch(
                "vocab_size, context_len and num_blocks must be > 0".into(),
            ));
        }
        let scale = 0.02f32; // GPT-style init
        vprintln!("[train_transformer::TransformerNet::new] vocab={}, ctx={}, dim={}, heads={}, hidden={}, blocks={}",
            vocab_size, context_len, embed_dim, num_heads, hidden_dim, num_blocks);
        Ok(Self {
            vocab_size,
            context_len,
            embed_dim,
            num_heads,
            tok_emb: Param::randn(vec![vocab_size, embed_dim], scale, rng),
            pos_emb: Param::randn(vec![context_len, embed_dim], scale, rng),
            blocks: (0..num_blocks)
                .map(|_| BlockT::new(embed_dim, hidden_dim, scale, rng))
                .collect(),
            lnf_g: Param::constant(vec![embed_dim], 1.0),
            lnf_b: Param::constant(vec![embed_dim], 0.0),
            head_w: Param::randn(vec![embed_dim, vocab_size], scale, rng),
            head_b: Param::constant(vec![vocab_size], 0.0),
            step_t: 0,
            qat: false,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    pub fn context_len(&self) -> usize {
        self.context_len
    }

    /// Enable or disable int8 Quantization Aware Training (see the `qat`
    /// field docs). Off by default; `GenerativeSLM` turns it on.
    pub fn set_qat(&mut self, enabled: bool) {
        self.qat = enabled;
    }
    pub fn qat_enabled(&self) -> bool {
        self.qat
    }

    /// Every parameter tensor, in a fixed order (used by the QAT snapshot /
    /// fake-quantize / restore cycle).
    fn param_tensors_mut(&mut self) -> Vec<&mut Tensor> {
        let mut v: Vec<&mut Tensor> = vec![&mut self.tok_emb.data, &mut self.pos_emb.data];
        for b in &mut self.blocks {
            for p in b.params_mut() {
                v.push(&mut p.data);
            }
        }
        v.push(&mut self.lnf_g.data);
        v.push(&mut self.lnf_b.data);
        v.push(&mut self.head_w.data);
        v.push(&mut self.head_b.data);
        v
    }

    /// Copy of every parameter tensor's data (the fp32 master weights).
    pub(crate) fn snapshot_weights(&mut self) -> Vec<Vec<f32>> {
        self.param_tensors_mut()
            .iter()
            .map(|t| t.data.clone())
            .collect()
    }

    /// Snap every (large-enough) parameter tensor onto the int8 grid in place.
    pub(crate) fn fake_quantize_weights(&mut self) {
        for t in self.param_tensors_mut() {
            crate::quant::fake_quantize_int8(&mut t.data);
        }
    }

    /// Restore master weights captured by [`TransformerNet::snapshot_weights`].
    pub(crate) fn restore_weights(&mut self, snapshot: &[Vec<f32>]) {
        for (t, s) in self.param_tensors_mut().into_iter().zip(snapshot) {
            t.data.copy_from_slice(s);
        }
    }

    pub fn num_params(&self) -> usize {
        let block = |b: &BlockT| -> usize {
            [
                &b.ln1_g, &b.ln1_b, &b.q_w, &b.q_b, &b.k_w, &b.k_b, &b.v_w, &b.v_b,
                &b.o_w, &b.o_b, &b.ln2_g, &b.ln2_b, &b.f1_w, &b.f1_b, &b.f2_w, &b.f2_b,
            ]
            .iter()
            .map(|p| p.data.numel())
            .sum()
        };
        self.tok_emb.data.numel()
            + self.pos_emb.data.numel()
            + self.blocks.iter().map(block).sum::<usize>()
            + self.lnf_g.data.numel()
            + self.lnf_b.data.numel()
            + self.head_w.data.numel()
            + self.head_b.data.numel()
    }

    /// Forward pass over `tokens` (length must be a multiple of `context_len`;
    /// each `context_len` chunk is one sequence in the batch). Returns logits
    /// of shape [B·T, vocab] plus the activation cache for `backward`.
    pub fn forward(&self, tokens: &[usize]) -> Result<(Tensor, FwdCache)> {
        let t = self.context_len;
        if tokens.is_empty() || tokens.len() % t != 0 {
            return Err(InferError::DimMismatch(format!(
                "token count {} must be a non-zero multiple of context_len {t}",
                tokens.len()
            )));
        }
        let batch = tokens.len() / t;
        let m = tokens.len();
        let c = self.embed_dim;

        // Embedding: token + positional lookup.
        let mut x = vec![0.0f32; m * c];
        for (r, &tok) in tokens.iter().enumerate() {
            if tok >= self.vocab_size {
                return Err(InferError::DimMismatch(format!(
                    "token {tok} out of range for vocab {}", self.vocab_size
                )));
            }
            let tb = tok * c;
            let pb = (r % t) * c;
            let xb = r * c;
            for d in 0..c {
                x[xb + d] = self.tok_emb.data.data[tb + d] + self.pos_emb.data.data[pb + d];
            }
        }
        let mut cur = Tensor::matrix(m, c, x)?;

        let mut block_caches = Vec::with_capacity(self.blocks.len());
        for blk in &self.blocks {
            let x_in = cur;
            let (norm1, xhat1, inv_std1) = ln_fwd(&x_in, &blk.ln1_g.data, &blk.ln1_b.data)?;
            let q = linear_fwd(&norm1, &blk.q_w.data, &blk.q_b.data)?;
            let k = linear_fwd(&norm1, &blk.k_w.data, &blk.k_b.data)?;
            let v = linear_fwd(&norm1, &blk.v_w.data, &blk.v_b.data)?;
            let (concat, probs) = attn_fwd(&q, &k, &v, batch, t, self.num_heads)?;
            let proj = linear_fwd(&concat, &blk.o_w.data, &blk.o_b.data)?;
            let x_attn = ops::add(&x_in, &proj)?;
            let (norm2, xhat2, inv_std2) = ln_fwd(&x_attn, &blk.ln2_g.data, &blk.ln2_b.data)?;
            let h_pre = linear_fwd(&norm2, &blk.f1_w.data, &blk.f1_b.data)?;
            let h_relu = h_pre.map(|v| v.max(0.0));
            let ff2 = linear_fwd(&h_relu, &blk.f2_w.data, &blk.f2_b.data)?;
            cur = ops::add(&x_attn, &ff2)?;
            block_caches.push(BlockCache {
                xhat1,
                inv_std1,
                norm1,
                q,
                k,
                v,
                probs,
                concat,
                xhat2,
                inv_std2,
                norm2,
                h_relu,
            });
        }

        let (norm_f, xhat_f, inv_std_f) = ln_fwd(&cur, &self.lnf_g.data, &self.lnf_b.data)?;
        let logits = linear_fwd(&norm_f, &self.head_w.data, &self.head_b.data)?;

        Ok((
            logits,
            FwdCache {
                tokens: tokens.to_vec(),
                batch,
                blocks: block_caches,
                xhat_f,
                inv_std_f,
                norm_f,
            },
        ))
    }

    /// Backprop `dlogits` (e.g. from `softmax_cross_entropy`) through the
    /// whole network, accumulating into every parameter's `.grad`.
    pub fn backward(&mut self, dlogits: &Tensor, cache: &FwdCache) -> Result<()> {
        let t = self.context_len;
        let batch = cache.batch;

        // LM head
        let (dw, db, d_normf) = linear_bwd(&cache.norm_f, &self.head_w.data, dlogits)?;
        self.head_w.accumulate(&dw);
        self.head_b.accumulate(&db);

        // Final LayerNorm
        let (mut dx, dg, dbeta) = ln_bwd(&d_normf, &cache.xhat_f, &cache.inv_std_f, &self.lnf_g.data)?;
        self.lnf_g.accumulate(&dg);
        self.lnf_b.accumulate(&dbeta);

        // Blocks, in reverse
        for (blk, bc) in self.blocks.iter_mut().zip(&cache.blocks).rev() {
            // out = x_attn + ffn2(relu(ffn1(norm2)))
            let d_out = dx;
            let (dw2, db2, dh) = linear_bwd(&bc.h_relu, &blk.f2_w.data, &d_out)?;
            blk.f2_w.accumulate(&dw2);
            blk.f2_b.accumulate(&db2);
            // ReLU backward via the cached post-activation values
            let dh_pre = Tensor::new(
                dh.shape.clone(),
                dh.data
                    .iter()
                    .zip(&bc.h_relu.data)
                    .map(|(&g, &h)| if h > 0.0 { g } else { 0.0 })
                    .collect(),
            )?;
            let (dw1, db1, d_norm2) = linear_bwd(&bc.norm2, &blk.f1_w.data, &dh_pre)?;
            blk.f1_w.accumulate(&dw1);
            blk.f1_b.accumulate(&db1);
            let (d_xattn_ln, dg2, dbeta2) = ln_bwd(&d_norm2, &bc.xhat2, &bc.inv_std2, &blk.ln2_g.data)?;
            blk.ln2_g.accumulate(&dg2);
            blk.ln2_b.accumulate(&dbeta2);
            // residual 2: gradient flows both into the FFN branch and straight through
            let d_xattn = ops::add(&d_out, &d_xattn_ln)?;

            // x_attn = x + out_proj(concat)
            let (dwo, dbo, d_concat) = linear_bwd(&bc.concat, &blk.o_w.data, &d_xattn)?;
            blk.o_w.accumulate(&dwo);
            blk.o_b.accumulate(&dbo);
            let (dq, dk, dv) = attn_bwd(
                &d_concat, &bc.q, &bc.k, &bc.v, &bc.probs, batch, t, self.num_heads,
            )?;
            let (dwq, dbq, d_n1a) = linear_bwd(&bc.norm1, &blk.q_w.data, &dq)?;
            let (dwk, dbk, d_n1b) = linear_bwd(&bc.norm1, &blk.k_w.data, &dk)?;
            let (dwv, dbv, d_n1c) = linear_bwd(&bc.norm1, &blk.v_w.data, &dv)?;
            blk.q_w.accumulate(&dwq);
            blk.q_b.accumulate(&dbq);
            blk.k_w.accumulate(&dwk);
            blk.k_b.accumulate(&dbk);
            blk.v_w.accumulate(&dwv);
            blk.v_b.accumulate(&dbv);
            let d_norm1 = ops::add(&ops::add(&d_n1a, &d_n1b)?, &d_n1c)?;
            let (d_x_ln, dg1, dbeta1) = ln_bwd(&d_norm1, &bc.xhat1, &bc.inv_std1, &blk.ln1_g.data)?;
            blk.ln1_g.accumulate(&dg1);
            blk.ln1_b.accumulate(&dbeta1);
            // residual 1
            dx = ops::add(&d_xattn, &d_x_ln)?;
        }

        // Embedding: scatter-add row gradients into the tables.
        let c = self.embed_dim;
        for (r, &tok) in cache.tokens.iter().enumerate() {
            let xb = r * c;
            let tb = tok * c;
            let pb = (r % t) * c;
            for d in 0..c {
                self.tok_emb.grad.data[tb + d] += dx.data[xb + d];
                self.pos_emb.grad.data[pb + d] += dx.data[xb + d];
            }
        }
        Ok(())
    }

    pub fn zero_grad(&mut self) {
        self.tok_emb.zero_grad();
        self.pos_emb.zero_grad();
        for b in &mut self.blocks {
            for p in b.params_mut() {
                p.zero_grad();
            }
        }
        self.lnf_g.zero_grad();
        self.lnf_b.zero_grad();
        self.head_w.zero_grad();
        self.head_b.zero_grad();
    }

    /// Apply one Adam update to every parameter.
    pub fn step(&mut self, adam: &Adam) -> Result<()> {
        self.step_t += 1;
        let t = self.step_t;
        self.tok_emb.step(adam, t)?;
        self.pos_emb.step(adam, t)?;
        for b in &mut self.blocks {
            for p in b.params_mut() {
                p.step(adam, t)?;
            }
        }
        self.lnf_g.step(adam, t)?;
        self.lnf_b.step(adam, t)?;
        self.head_w.step(adam, t)?;
        self.head_b.step(adam, t)?;
        Ok(())
    }

    /// Export to an inference `Sequential`:
    /// `Embedding → blocks → LayerNorm → Linear → Softmax`.
    /// Serializes to FINF v4 and is compatible with `TransformerSLMModel`.
    pub fn to_inference(&self) -> Result<Sequential> {
        let mut m = Sequential::new();
        m.push(Box::new(Embedding::new(
            self.vocab_size,
            self.context_len,
            self.embed_dim,
            self.tok_emb.data.data.clone(),
            self.pos_emb.data.data.clone(),
        )?));
        for b in &self.blocks {
            m.push(Box::new(TransformerBlock::new(
                self.context_len,
                self.num_heads,
                self.embed_dim,
                b.ln1_g.data.data.clone(), b.ln1_b.data.data.clone(),
                b.q_w.data.data.clone(), b.q_b.data.data.clone(),
                b.k_w.data.data.clone(), b.k_b.data.data.clone(),
                b.v_w.data.data.clone(), b.v_b.data.data.clone(),
                b.o_w.data.data.clone(), b.o_b.data.data.clone(),
                b.ln2_g.data.data.clone(), b.ln2_b.data.data.clone(),
                b.f1_w.data.data.clone(), b.f1_b.data.data.clone(),
                b.f2_w.data.data.clone(), b.f2_b.data.data.clone(),
            )?));
        }
        m.push(Box::new(LayerNorm::new(
            self.embed_dim,
            self.lnf_g.data.data.clone(),
            self.lnf_b.data.data.clone(),
        )?));
        m.push(Box::new(Linear::new(
            self.embed_dim,
            self.vocab_size,
            self.head_w.data.data.clone(),
            self.head_b.data.data.clone(),
        )?));
        m.push(Box::new(ActivationLayer::new(crate::activation::Activation::Softmax)));
        Ok(m)
    }
}

/// One epoch of minibatch Adam over a token stream. Each example is a window
/// of `context_len` tokens with next-token targets at every position.
/// Returns the mean train loss.
///
/// If the net has QAT enabled ([`TransformerNet::set_qat`]), each step runs
/// forward and backward against int8-snapped weights, then applies the Adam
/// update to the full-precision master weights (straight-through estimator),
/// so the trained model is robust to int8 export.
pub fn train_transformer_epoch(
    net: &mut TransformerNet,
    tokens: &[usize],
    batch_size: usize,
    adam: &Adam,
    rng: &mut Rng,
) -> Result<f32> {
    use crate::loss::softmax_cross_entropy;
    let t = net.context_len();
    if tokens.len() < t + 1 {
        return Err(InferError::DimMismatch(format!(
            "need at least context_len+1 = {} tokens, got {}",
            t + 1,
            tokens.len()
        )));
    }
    let num_windows = tokens.len() - t;
    let steps = num_windows.div_ceil(batch_size);
    let mut total = 0.0f32;
    for _ in 0..steps {
        let mut input = Vec::with_capacity(batch_size * t);
        let mut targets = Vec::with_capacity(batch_size * t);
        for _ in 0..batch_size {
            let start = (rng.next_u64() as usize) % num_windows;
            input.extend_from_slice(&tokens[start..start + t]);
            targets.extend_from_slice(&tokens[start + 1..start + t + 1]);
        }
        // QAT: gradients are computed at the int8-snapped weights, but the
        // optimizer updates the fp32 masters (straight-through estimator).
        let masters = if net.qat_enabled() {
            let snapshot = net.snapshot_weights();
            net.fake_quantize_weights();
            Some(snapshot)
        } else {
            None
        };
        let (logits, cache) = net.forward(&input)?;
        let (loss, dlogits) = softmax_cross_entropy(&logits, &targets)?;
        net.zero_grad();
        net.backward(&dlogits, &cache)?;
        if let Some(snapshot) = &masters {
            net.restore_weights(snapshot);
        }
        net.step(adam)?;
        total += loss;
    }
    let avg = total / steps as f32;
    if verbose::is_verbose() {
        vprintln!("[train_transformer::epoch] steps={}, mean loss={:.6}", steps, avg);
    }
    Ok(avg)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use crate::loss::softmax_cross_entropy;

    fn tiny_net(rng: &mut Rng) -> TransformerNet {
        // vocab=5, T=3, C=4, heads=2, hidden=6, 1 block
        TransformerNet::new(5, 3, 4, 2, 6, 1, rng).unwrap()
    }

    fn loss_of(net: &TransformerNet, tokens: &[usize], targets: &[usize]) -> f32 {
        let (logits, _) = net.forward(tokens).unwrap();
        softmax_cross_entropy(&logits, targets).unwrap().0
    }

    /// Finite-difference check of one parameter tensor against analytic grads.
    fn check_param<G, S>(
        net: &mut TransformerNet,
        tokens: &[usize],
        targets: &[usize],
        get_grad: G,
        set_data: S,
        indices: &[usize],
        label: &str,
    ) where
        G: Fn(&TransformerNet, usize) -> f32,
        S: Fn(&mut TransformerNet, usize, f32),
    {
        let (logits, cache) = net.forward(tokens).unwrap();
        let (_, dl) = softmax_cross_entropy(&logits, targets).unwrap();
        net.zero_grad();
        net.backward(&dl, &cache).unwrap();

        let eps = 1e-2f32;
        for &i in indices {
            let analytic = get_grad(net, i);
            // Central difference: nudge +eps, then −2eps, then restore.
            set_data(net, i, eps);
            let lp = loss_of(net, tokens, targets);
            set_data(net, i, -2.0 * eps);
            let lm = loss_of(net, tokens, targets);
            set_data(net, i, eps);
            let numeric = (lp - lm) / (2.0 * eps);
            assert!(
                (numeric - analytic).abs() < 6e-3 + 0.03 * analytic.abs(),
                "{label}[{i}]: analytic={analytic:.6} numeric={numeric:.6}"
            );
        }
    }

    #[test]
    fn gradient_check_all_param_groups() {
        let mut rng = Rng::new(3);
        let mut net = tiny_net(&mut rng);
        let tokens = [1usize, 2, 3];
        let targets = [2usize, 3, 4];

        macro_rules! check {
            ($field:ident, $label:expr, $idx:expr) => {
                check_param(
                    &mut net,
                    &tokens,
                    &targets,
                    |n, i| n.$field.grad.data[i],
                    |n, i, d| n.$field.data.data[i] += d,
                    $idx,
                    $label,
                );
            };
        }
        macro_rules! check_blk {
            ($field:ident, $label:expr, $idx:expr) => {
                check_param(
                    &mut net,
                    &tokens,
                    &targets,
                    |n, i| n.blocks[0].$field.grad.data[i],
                    |n, i, d| n.blocks[0].$field.data.data[i] += d,
                    $idx,
                    $label,
                );
            };
        }

        check!(tok_emb, "tok_emb", &[4, 9, 13]);   // rows of tokens 1,2,3
        check!(pos_emb, "pos_emb", &[0, 5, 11]);
        check!(lnf_g, "lnf_g", &[0, 3]);
        check!(lnf_b, "lnf_b", &[1]);
        check!(head_w, "head_w", &[0, 7, 19]);
        check!(head_b, "head_b", &[2]);
        check_blk!(ln1_g, "ln1_g", &[0, 2]);
        check_blk!(ln1_b, "ln1_b", &[1]);
        check_blk!(q_w, "q_w", &[0, 5, 15]);
        check_blk!(k_w, "k_w", &[3, 10]);
        check_blk!(v_w, "v_w", &[2, 12]);
        check_blk!(v_b, "v_b", &[1]);
        check_blk!(o_w, "o_w", &[4, 9]);
        check_blk!(ln2_g, "ln2_g", &[0, 3]);
        check_blk!(f1_w, "f1_w", &[0, 11, 20]);
        check_blk!(f1_b, "f1_b", &[2]);
        check_blk!(f2_w, "f2_w", &[5, 17]);
        check_blk!(f2_b, "f2_b", &[0]);
    }

    #[test]
    fn training_reduces_loss() {
        let mut rng = Rng::new(7);
        // vocab=4, T=4, C=8, heads=2, hidden=16, 1 block
        let mut net = TransformerNet::new(4, 4, 8, 2, 16, 1, &mut rng).unwrap();
        // A perfectly periodic token stream — trivially learnable.
        let tokens: Vec<usize> = (0..200).map(|i| i % 4).collect();
        let adam = Adam::new(0.01);
        let first = train_transformer_epoch(&mut net, &tokens, 8, &adam, &mut rng).unwrap();
        let mut last = first;
        for _ in 0..30 {
            last = train_transformer_epoch(&mut net, &tokens, 8, &adam, &mut rng).unwrap();
        }
        assert!(
            last < first * 0.5,
            "loss did not halve: {first:.4} → {last:.4}"
        );
    }

    #[test]
    fn qat_training_reduces_loss() {
        let mut rng = Rng::new(7);
        let mut net = TransformerNet::new(4, 4, 8, 2, 16, 1, &mut rng).unwrap();
        net.set_qat(true);
        assert!(net.qat_enabled());
        let tokens: Vec<usize> = (0..200).map(|i| i % 4).collect();
        let adam = Adam::new(0.01);
        let first = train_transformer_epoch(&mut net, &tokens, 8, &adam, &mut rng).unwrap();
        let mut last = first;
        for _ in 0..30 {
            last = train_transformer_epoch(&mut net, &tokens, 8, &adam, &mut rng).unwrap();
        }
        assert!(
            last < first * 0.5,
            "QAT loss did not halve: {first:.4} → {last:.4}"
        );
    }

    #[test]
    fn qat_snapshot_quantize_restore_cycle() {
        let mut rng = Rng::new(11);
        let mut net = TransformerNet::new(8, 4, 8, 2, 16, 1, &mut rng).unwrap();
        let masters = net.snapshot_weights();
        net.fake_quantize_weights();
        // Large tensors (≥ QUANT_MIN_LEN) must now sit on the int8 grid.
        let quantized = net.snapshot_weights();
        let mut any_changed = false;
        for (m, q) in masters.iter().zip(&quantized) {
            if q.len() >= crate::quant::QUANT_MIN_LEN {
                let scale = crate::quant::int8_scale(q);
                if scale > 0.0 {
                    for &v in q {
                        let steps = v / scale;
                        assert!((steps - steps.round()).abs() < 1e-3, "{v} off the int8 grid");
                    }
                }
            }
            if m != q {
                any_changed = true;
            }
        }
        assert!(any_changed, "fake quantization changed no weights");
        // Restore must bring back the fp32 masters exactly.
        net.restore_weights(&masters);
        assert_eq!(net.snapshot_weights(), masters);
    }

    #[test]
    fn qat_model_survives_int8_export_with_small_drift() {
        use crate::csv::{ModelMetadata, Normalizer, TaskType};
        use crate::loader::{from_bytes, to_bytes_quantized};
        let mut rng = Rng::new(13);
        let mut net = TransformerNet::new(4, 4, 8, 2, 16, 1, &mut rng).unwrap();
        net.set_qat(true);
        let tokens: Vec<usize> = (0..200).map(|i| i % 4).collect();
        let adam = Adam::new(0.01);
        for _ in 0..20 {
            train_transformer_epoch(&mut net, &tokens, 8, &adam, &mut rng).unwrap();
        }
        let model = net.to_inference().unwrap();
        let meta = ModelMetadata {
            dataset_name: "qat".into(),
            task: TaskType::TransformerSLM,
            feature_names: vec![],
            feature_ranges: vec![],
            class_names: (0..4).map(|i| i.to_string()).collect(),
            target_name: "next".into(),
            target_range: [0.0, 4.0],
            input_dim: 4,
            output_dim: 4,
            tokenizer_state: String::new(),
        };
        let norm = Normalizer { means: vec![], stds: vec![] };
        let bytes = to_bytes_quantized(&model, &norm, &meta).unwrap();
        let (m2, _, _) = from_bytes(&bytes).unwrap();
        let x = Tensor::matrix(1, 4, vec![0.0, 1.0, 2.0, 3.0]).unwrap();
        let before = model.forward(&x).unwrap();
        let after = m2.forward(&x).unwrap();
        // Output probabilities of the int8 export must track the fp32 model.
        for (a, b) in before.data.iter().zip(&after.data) {
            assert!((a - b).abs() < 0.05, "int8 export drifted: {a} vs {b}");
        }
    }

    #[test]
    fn to_inference_matches_trainable_forward() {
        let mut rng = Rng::new(21);
        let net = TransformerNet::new(6, 4, 8, 2, 12, 2, &mut rng).unwrap();
        let tokens = [0usize, 3, 1, 5];
        let (logits, _) = net.forward(&tokens).unwrap();

        let model = net.to_inference().unwrap();
        let ids: Vec<f32> = tokens.iter().map(|&t| t as f32).collect();
        let x = Tensor::matrix(1, 4, ids).unwrap();
        let out = model.forward(&x).unwrap(); // softmaxed [T, vocab]

        // Softmax the trainable logits and compare row by row.
        let probs = ops::softmax_rows(&logits).unwrap();
        assert_eq!(out.shape, probs.shape);
        for (a, b) in out.data.iter().zip(&probs.data) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn exported_model_roundtrips_finf() {
        use crate::csv::{ModelMetadata, Normalizer, TaskType};
        use crate::loader::{from_bytes, to_bytes};
        let mut rng = Rng::new(33);
        let net = TransformerNet::new(6, 4, 8, 2, 12, 1, &mut rng).unwrap();
        let model = net.to_inference().unwrap();
        let meta = ModelMetadata {
            dataset_name: "t".into(),
            task: TaskType::TransformerSLM,
            feature_names: vec![],
            feature_ranges: vec![],
            class_names: (0..6).map(|i| i.to_string()).collect(),
            target_name: "next".into(),
            target_range: [0.0, 6.0],
            input_dim: 4,
            output_dim: 6,
            tokenizer_state: String::new(),
        };
        let norm = Normalizer { means: vec![], stds: vec![] };
        let bytes = to_bytes(&model, &norm, &meta).unwrap();
        let (m2, _, _) = from_bytes(&bytes).unwrap();
        let x = Tensor::matrix(1, 4, vec![0.0, 3.0, 1.0, 5.0]).unwrap();
        let a = model.forward(&x).unwrap();
        let b = m2.forward(&x).unwrap();
        for (p, q) in a.data.iter().zip(&b.data) {
            assert!((p - q).abs() < 1e-6);
        }
    }

    #[test]
    fn invalid_configs_rejected() {
        let mut rng = Rng::new(1);
        assert!(TransformerNet::new(5, 3, 7, 2, 6, 1, &mut rng).is_err()); // 7 % 2 ≠ 0
        assert!(TransformerNet::new(5, 3, 4, 0, 6, 1, &mut rng).is_err());
        assert!(TransformerNet::new(0, 3, 4, 2, 6, 1, &mut rng).is_err());
        assert!(TransformerNet::new(5, 3, 4, 2, 6, 0, &mut rng).is_err());
    }

    #[test]
    fn forward_rejects_bad_token_counts() {
        let mut rng = Rng::new(1);
        let net = tiny_net(&mut rng);
        assert!(net.forward(&[]).is_err());
        assert!(net.forward(&[1, 2]).is_err()); // not a multiple of T=3
        assert!(net.forward(&[1, 2, 99]).is_err()); // token out of vocab
    }
}
