//! Trainable decoder-only causal Transformer: full backprop through token +
//! positional embeddings, LayerNorm, multi-head self-attention, and the FFN,
//! optimised with Adam. Mirrors the `train.rs` pattern: a dedicated trainable
//! struct (`TransformerNet`) that exports to an inference `Sequential` whose
//! layers (`Embedding`, `TransformerBlock`, `LayerNorm`, `Linear`) serialize
//! to GGUF and run in WASM.

use crate::ckpt;
use crate::error::{InferError, Result};
use crate::layer::{
    matmul_naive_helper, matmul_transpose_b_helper, ActivationLayer, Embedding, LayerNorm, Linear,
    TransformerBlock,
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

#[derive(Clone)]
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
    fn step(&mut self, adam: &Adam, t: u64) -> Result<()> {
        // Decoupled weight decay (AdamW) applies to weight matrices only — never
        // to biases or LayerNorm gains/biases (rank-1 params), per convention.
        if adam.weight_decay != 0.0 && self.data.shape.len() < 2 {
            let mut a = *adam;
            a.weight_decay = 0.0;
            return a.step(t, &mut self.data, &self.grad, &mut self.m, &mut self.v);
        }
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
    // `r` indexes inv_std and derives the row offset for x/y/xhat — an iterator
    // would obscure the per-row math without removing the offset arithmetic.
    #[allow(clippy::needless_range_loop)]
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
    // `r` derives the row offset and indexes inv_std; see ln_fwd.
    #[allow(clippy::needless_range_loop)]
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
            dx[base + c] = inv_std[r] * (dxh - mean_dxhat - xhat.data[base + c] * mean_dxhat_xhat);
        }
    }
    Ok((
        Tensor::matrix(rows, cols, dx)?,
        Tensor::vector(dg),
        Tensor::vector(db),
    ))
}

/// out [m,n] = aᵀ·b where a is [k,m] and b is [k,n].
/// Rows `r0..r1` of `Aᵀ·B`: `out[i,j] = Σ_r a[r,i] · b[r,j]` (A is `[k, m]`,
/// B is `[k, n]`), written into a locally-indexed `out`. Output rows (i) are
/// independent and the sum over r keeps its original order, so the split is
/// exact. Shared by the serial and pooled paths.
#[allow(clippy::too_many_arguments)]
fn transpose_a_block(
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
    r0: usize,
    r1: usize,
    out: &mut [f32],
) {
    for i in r0..r1 {
        let o_row = (i - r0) * n;
        for r in 0..k {
            let a_ri = a[r * m + i];
            let b_row = r * n;
            for j in 0..n {
                out[o_row + j] += a_ri * b[b_row + j];
            }
        }
    }
}

fn matmul_transpose_a_helper(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let cost = m.saturating_mul(n).saturating_mul(k);
    if crate::parallel::should_parallelize(m, cost) {
        let a_arc = std::sync::Arc::<[f32]>::from(a);
        let b_arc = std::sync::Arc::<[f32]>::from(b);
        crate::parallel::run(m, n, move |r0, r1, block| {
            transpose_a_block(&a_arc, &b_arc, m, n, k, r0, r1, block);
        })
    } else {
        let mut out = vec![0.0f32; m * n];
        transpose_a_block(a, b, m, n, k, 0, m, &mut out);
        out
    }
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
#[allow(clippy::too_many_arguments)] // the attention backward genuinely needs q/k/v, probs, and the batch/t/heads shape
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

#[derive(Clone)]
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
            &mut self.ln1_g,
            &mut self.ln1_b,
            &mut self.q_w,
            &mut self.q_b,
            &mut self.k_w,
            &mut self.k_b,
            &mut self.v_w,
            &mut self.v_b,
            &mut self.o_w,
            &mut self.o_b,
            &mut self.ln2_g,
            &mut self.ln2_b,
            &mut self.f1_w,
            &mut self.f1_b,
            &mut self.f2_w,
            &mut self.f2_b,
        ]
    }

    fn params(&self) -> [&Param; 16] {
        [
            &self.ln1_g,
            &self.ln1_b,
            &self.q_w,
            &self.q_b,
            &self.k_w,
            &self.k_b,
            &self.v_w,
            &self.v_b,
            &self.o_w,
            &self.o_b,
            &self.ln2_g,
            &self.ln2_b,
            &self.f1_w,
            &self.f1_b,
            &self.f2_w,
            &self.f2_b,
        ]
    }
}

/// Owned gradient tensors for one `BlockT`, produced by
/// [`TransformerNet::backward_grads`] so a worker can compute gradients without
/// mutating (or cloning) the shared parameters. Fields mirror `BlockT`.
struct BlockGrads {
    ln1_g: Tensor,
    ln1_b: Tensor,
    q_w: Tensor,
    q_b: Tensor,
    k_w: Tensor,
    k_b: Tensor,
    v_w: Tensor,
    v_b: Tensor,
    o_w: Tensor,
    o_b: Tensor,
    ln2_g: Tensor,
    ln2_b: Tensor,
    f1_w: Tensor,
    f1_b: Tensor,
    f2_w: Tensor,
    f2_b: Tensor,
}

struct BlockCache {
    xhat1: Tensor,
    inv_std1: Vec<f32>,
    norm1: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    probs: Vec<f32>, // [B·H·T·T]
    concat: Tensor,  // attention head outputs before out_proj [M, C]
    xhat2: Tensor,
    inv_std2: Vec<f32>,
    norm2: Tensor,
    /// FFN-1 pre-activation [M, hidden].
    h_pre: Tensor,
    /// FFN-2 input [M, hidden]: GELU(h) — or, when dropout is active, the
    /// dropped-and-scaled hidden `GELU(h) ⊙ mask`.
    h_act: Tensor,
    /// Inverted-dropout mask×scale applied to the GELU output (T7), present only when
    /// dropout was active for this forward; `None` means no dropout.
    ffn_dropout: Option<Vec<f32>>,
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
/// to obtain a `Sequential` that serializes to GGUF and runs anywhere
/// (including WASM with KV-cached generation).
///
/// `Clone` produces a deep copy of every weight, gradient, and Adam moment
/// buffer; [`train_transformer_epoch_threaded`] uses it to give each worker
/// thread an independent network to accumulate gradients into.
#[derive(Clone)]
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
    step_t: u64,
    /// Quantization Aware Training (int8 QAT simulating GGUF Q8_0): when
    /// enabled, `train_transformer_epoch` runs forward/backward against
    /// Q8_0-snapped weights while Adam updates
    /// full-precision master weights (straight-through estimator).
    qat: bool,
    /// Global-norm gradient clipping threshold. When `Some(max_norm)`, every
    /// training step rescales the gradients so their global L2 norm is at most
    /// `max_norm` before the Adam update, preventing divergence at higher
    /// learning rates / larger models. `None` (default) leaves gradients
    /// untouched, so the optimizer step is bit-identical to before.
    grad_clip: Option<f32>,
    /// Optional learning-rate schedule (warmup + decay). When `Some`, each
    /// [`TransformerNet::step`] overrides the Adam learning rate with
    /// `schedule.lr_at(step_t)`; `None` (default) uses the optimizer's fixed
    /// `lr`, so the step is unchanged.
    lr_schedule: Option<crate::optim::LrSchedule>,
    /// Weight tying (T9): when `true`, the LM head weight is kept equal to the
    /// transpose of the token-embedding table (`head_w = tok_embᵀ`) and their
    /// gradients are summed into the shared parameter, saving `vocab × embed`
    /// independent parameters. `false` (default) trains them independently.
    tie_weights: bool,
    /// Decoupled (AdamW) weight-decay coefficient applied to weight matrices
    /// each step (T7). `0.0` (default) disables it. Folded into the Adam used by
    /// [`TransformerNet::step`].
    weight_decay: f32,
    /// FFN-hidden dropout probability used during training forward passes (T7).
    /// `0.0` (default) disables it; inference is always dropout-free.
    dropout: f32,
    /// Distance in tokens between consecutive training-window starts. `1`
    /// (default) is the legacy fully-overlapping sliding window, where one epoch
    /// trains on every token about `context_len` times; `context_len` gives
    /// non-overlapping windows, so one epoch sees each token once.
    window_stride: usize,
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
        if num_heads == 0 || !embed_dim.is_multiple_of(num_heads) {
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
            step_t: 0,
            qat: false,
            grad_clip: None,
            lr_schedule: None,
            tie_weights: false,
            weight_decay: 0.0,
            dropout: 0.0,
            window_stride: 1,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    pub fn context_len(&self) -> usize {
        self.context_len
    }

    /// Enable or disable int8 QAT simulating GGUF Q8_0 (see the `qat`
    /// field docs). Off by default; `GenerativeSLM` turns it on.
    pub fn set_qat(&mut self, enabled: bool) {
        self.qat = enabled;
    }
    pub fn qat_enabled(&self) -> bool {
        self.qat
    }

    /// Set the global-norm gradient-clipping threshold (see the `grad_clip`
    /// field docs). `Some(max_norm)` clips before each step; `None` disables.
    pub fn set_grad_clip(&mut self, max_norm: Option<f32>) {
        self.grad_clip = max_norm;
    }
    pub fn grad_clip(&self) -> Option<f32> {
        self.grad_clip
    }

    /// Rescale all gradients so their global L2 norm is at most `max_norm`,
    /// returning the pre-clip norm. Operates on the gradients currently held in
    /// every parameter (i.e. after [`TransformerNet::backward`]). Exposed for
    /// callers driving their own training loop; the built-in epoch helpers call
    /// it automatically when [`TransformerNet::set_grad_clip`] is set.
    pub fn clip_grad_norm(&mut self, max_norm: f32) -> f32 {
        let mut grads = self.grad_tensors_mut();
        crate::optim::clip_grad_norm(&mut grads, max_norm)
    }

    /// Set the learning-rate schedule (warmup + decay). When `Some`, it
    /// overrides the Adam learning rate at every [`TransformerNet::step`] using
    /// the network's internal 1-based step counter; `None` keeps the fixed `lr`.
    pub fn set_lr_schedule(&mut self, schedule: Option<crate::optim::LrSchedule>) {
        self.lr_schedule = schedule;
    }
    pub fn lr_schedule(&self) -> Option<crate::optim::LrSchedule> {
        self.lr_schedule
    }

    /// The number of optimizer steps applied so far (the schedule's timestep).
    pub fn step_count(&self) -> u64 {
        self.step_t
    }

    /// Set the decoupled (AdamW) weight-decay coefficient applied to weight
    /// matrices each step (T7). `0.0` disables it.
    pub fn set_weight_decay(&mut self, weight_decay: f32) {
        self.weight_decay = weight_decay;
    }
    pub fn weight_decay(&self) -> f32 {
        self.weight_decay
    }

    /// Set the FFN-hidden dropout probability used in training forward passes
    /// (T7), in `[0, 1)`. `0.0` disables it; values are clamped to `< 1`.
    pub fn set_dropout(&mut self, p: f32) {
        self.dropout = p.clamp(0.0, 0.95);
    }
    pub fn dropout(&self) -> f32 {
        self.dropout
    }

    /// Set the training-window stride (clamped to `1..=context_len`); see the
    /// field docs. Larger strides make an epoch proportionally cheaper.
    pub fn set_window_stride(&mut self, stride: usize) {
        self.window_stride = stride.clamp(1, self.context_len);
    }
    pub fn window_stride(&self) -> usize {
        self.window_stride
    }

    /// Training-window start positions for one epoch over `n_tokens`, in a
    /// fresh random order. With stride > 1 the grid gets a random phase each
    /// epoch, so window boundaries differ across epochs. Stride 1 draws exactly
    /// as before (one shuffle), keeping existing runs reproducible.
    fn epoch_window_starts(&self, n_tokens: usize, rng: &mut Rng) -> Result<Vec<usize>> {
        let t = self.context_len;
        if n_tokens < t + 1 {
            return Err(InferError::DimMismatch(format!(
                "need at least context_len+1 = {} tokens, got {}",
                t + 1,
                n_tokens
            )));
        }
        let last = n_tokens - t - 1; // largest valid start
        let s = self.window_stride;
        let phase = if s > 1 {
            (rng.next_u64() % s as u64) as usize % (last + 1)
        } else {
            0
        };
        let count = (last - phase) / s + 1;
        Ok(rng
            .shuffled_indices(count)
            .into_iter()
            .map(|i| phase + i * s)
            .collect())
    }

    /// Enable or disable weight tying between the token embedding and the LM
    /// head (T9). When enabled, [`TransformerNet::sync_tied_head`] mirrors the
    /// embedding into the head before each step and
    /// [`TransformerNet::fold_tied_head_grad`] sums the head gradient back into
    /// the embedding, so the two share parameters.
    pub fn set_weight_tying(&mut self, enabled: bool) {
        self.tie_weights = enabled;
        if enabled {
            self.sync_tied_head();
        }
    }
    pub fn weight_tying(&self) -> bool {
        self.tie_weights
    }

    /// Copy `tok_embᵀ` into the LM-head weight so the head mirrors the embedding
    /// (no-op unless weight tying is enabled). Called before each forward.
    pub fn sync_tied_head(&mut self) {
        if !self.tie_weights {
            return;
        }
        let (vocab, embed) = (self.vocab_size, self.embed_dim);
        // head_w is [embed, vocab] = transpose of tok_emb [vocab, embed].
        for e in 0..embed {
            for v in 0..vocab {
                self.head_w.data.data[e * vocab + v] = self.tok_emb.data.data[v * embed + e];
            }
        }
    }

    /// Fold the LM-head gradient into the token-embedding gradient
    /// (`tok_emb.grad += head_w.gradᵀ`) so the shared parameter receives both
    /// contributions, then zero the head gradient so the redundant head copy is
    /// not stepped independently. No-op unless weight tying is enabled.
    pub fn fold_tied_head_grad(&mut self) {
        if !self.tie_weights {
            return;
        }
        let (vocab, embed) = (self.vocab_size, self.embed_dim);
        for e in 0..embed {
            for v in 0..vocab {
                self.tok_emb.grad.data[v * embed + e] += self.head_w.grad.data[e * vocab + v];
                self.head_w.grad.data[e * vocab + v] = 0.0;
            }
        }
    }

    /// FFN hidden width (read from the first block's first FFN layer).
    fn hidden_dim(&self) -> usize {
        self.blocks[0].f1_b.data.numel()
    }

    /// Every `Param` (weights + Adam moments), in canonical order. Used by the
    /// checkpoint codec (T6).
    fn params_all(&self) -> Vec<&Param> {
        let mut v: Vec<&Param> = vec![&self.tok_emb, &self.pos_emb];
        for b in &self.blocks {
            v.extend(b.params());
        }
        v.push(&self.lnf_g);
        v.push(&self.lnf_b);
        v.push(&self.head_w);
        v
    }

    fn params_mut_all(&mut self) -> Vec<&mut Param> {
        let mut v: Vec<&mut Param> = vec![&mut self.tok_emb, &mut self.pos_emb];
        for b in &mut self.blocks {
            v.extend(b.params_mut());
        }
        v.push(&mut self.lnf_g);
        v.push(&mut self.lnf_b);
        v.push(&mut self.head_w);
        v
    }

    /// Serialize the **full training state** (T6): architecture, QAT/tying
    /// flags, optimizer timestep, the supplied RNG's state, and every
    /// parameter's weights *and* Adam moment buffers (`m`, `v`). Pair with
    /// [`TransformerNet::load_checkpoint`] to resume an interrupted run exactly
    /// where it stopped — unlike a finished-model save, this preserves the
    /// optimizer state so momentum/adaptive scales are not lost.
    ///
    /// (The learning-rate schedule and grad-clip threshold are training-driver
    /// settings, not weights; re-apply them on the resumed net.)
    pub fn save_checkpoint(&self, rng: &Rng) -> Vec<u8> {
        let mut b = ckpt::begin("ferrum-checkpoint");
        for (k, v) in [
            ("vocab", self.vocab_size),
            ("context", self.context_len),
            ("embed", self.embed_dim),
            ("heads", self.num_heads),
            ("hidden", self.hidden_dim()),
            ("blocks", self.blocks.len()),
        ] {
            ckpt::put_u32(&mut b, k, v);
        }
        ckpt::put_bool(&mut b, "qat", self.qat);
        ckpt::put_bool(&mut b, "tie", self.tie_weights);
        ckpt::put_u64(&mut b, "step", self.step_t);
        ckpt::put_u64(&mut b, "rng", rng.state());
        for (i, p) in self.params_all().into_iter().enumerate() {
            ckpt::put_param(&mut b, i, &p.data.data, &p.m.data, &p.v.data);
        }
        b.into_bytes()
    }

    /// Parameter tensors per transformer block (`BlockT::params`).
    const BLOCK_PARAMS: usize = 16;

    /// Element count of every parameter, in `params_all` order; `None` on
    /// overflow. Allocates `5 + 16·blocks` entries, so callers bound `blocks`
    /// first.
    fn param_lens(
        vocab: usize,
        ctx: usize,
        embed: usize,
        hidden: usize,
        blocks: usize,
    ) -> Option<Vec<usize>> {
        let (e, h) = (embed, hidden);
        let sq = e.checked_mul(e)?;
        let eh = e.checked_mul(h)?;
        let block = [e, e, sq, e, sq, e, sq, e, sq, e, e, e, eh, h, eh, e];
        debug_assert_eq!(block.len(), Self::BLOCK_PARAMS);
        let ev = vocab.checked_mul(e)?; // token embedding / LM head
        let mut lens = vec![ev, ctx.checked_mul(e)?]; // + positions
        for _ in 0..blocks {
            lens.extend(block);
        }
        lens.extend([e, e, ev]); // final LayerNorm scale + shift, head
        Some(lens)
    }

    /// Reject a checkpoint whose header dims do not account for exactly the
    /// tensors it carries (`dims` = vocab, context, embed, heads, hidden,
    /// blocks). Runs before any header-sized allocation.
    fn check_header(g: &crate::gguf::Gguf, dims: [usize; 6]) -> Result<()> {
        let bad = || InferError::Format("checkpoint header does not match its tensors".into());
        let [vocab, ctx, embed, heads, hidden, blocks] = dims;
        if dims.contains(&0) || !embed.is_multiple_of(heads) {
            return Err(bad());
        }
        // 5 non-block params (2 embeddings, final LN scale + shift, head).
        let n_params = blocks
            .checked_mul(Self::BLOCK_PARAMS)
            .and_then(|b| b.checked_add(5))
            .ok_or_else(bad)?;
        if n_params.checked_mul(3) != Some(g.tensors.len()) {
            return Err(bad());
        }
        let lens = Self::param_lens(vocab, ctx, embed, hidden, blocks).ok_or_else(bad)?;
        // Exactly `p{i}.{data,m,v}`, F32, of the expected length. With the
        // count check above and distinct names, this accounts for every
        // tensor, so (non-overlapping, in-file) extents bound what `new` may
        // allocate by the file size.
        let by_name: std::collections::HashMap<&str, &crate::gguf::TensorInfo> =
            g.tensors.iter().map(|t| (t.name.as_str(), t)).collect();
        for (i, &len) in lens.iter().enumerate() {
            for s in ["data", "m", "v"] {
                match by_name.get(format!("p{i}.{s}").as_str()) {
                    Some(t) if t.ggml_type == crate::gguf::GGML_F32 && t.dims == [len as u64] => {}
                    _ => return Err(bad()),
                }
            }
        }
        Ok(())
    }

    /// Rebuild a net and RNG from [`TransformerNet::save_checkpoint`] bytes,
    /// restoring weights, Adam moments, optimizer timestep, and RNG state so
    /// training resumes deterministically. Pre-GGUF (`FCKP`) checkpoints are
    /// rejected with a pointed error.
    pub fn load_checkpoint(bytes: &[u8]) -> Result<(Self, Rng)> {
        let g = ckpt::open(bytes, b"FCKP", "ferrum-checkpoint")?;
        let n = |k| ckpt::get_usize(&g, k);
        let (vocab, ctx, embed) = (n("vocab")?, n("context")?, n("embed")?);
        let (heads, hidden, blocks) = (n("heads")?, n("hidden")?, n("blocks")?);
        // The header must be consistent with the (file-bounded) tensors before
        // `new` allocates anything sized by it.
        Self::check_header(&g, [vocab, ctx, embed, heads, hidden, blocks])?;
        let mut seed_rng = Rng::new(1); // throwaway: param data is overwritten
        let mut net = TransformerNet::new(vocab, ctx, embed, heads, hidden, blocks, &mut seed_rng)?;
        net.qat = ckpt::get_bool(&g, "qat")?;
        net.tie_weights = ckpt::get_bool(&g, "tie")?;
        net.step_t = ckpt::get_u64(&g, "step")?;
        let rng_state = ckpt::get_u64(&g, "rng")?;
        for (i, p) in net.params_mut_all().into_iter().enumerate() {
            // m and v share the parameter's shape, so each is the same length.
            let len = p.data.data.len();
            p.data.data = ckpt::get_f32(&g, &format!("p{i}.data"), len)?;
            p.m.data = ckpt::get_f32(&g, &format!("p{i}.m"), len)?;
            p.v.data = ckpt::get_f32(&g, &format!("p{i}.v"), len)?;
        }
        Ok((net, Rng::from_state(rng_state)))
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
        v
    }

    /// Every gradient tensor, in the same fixed order as
    /// [`TransformerNet::param_tensors_mut`]. Used to reduce per-shard
    /// gradients in [`train_transformer_epoch_threaded`].
    fn grad_tensors_mut(&mut self) -> Vec<&mut Tensor> {
        let mut v: Vec<&mut Tensor> = vec![&mut self.tok_emb.grad, &mut self.pos_emb.grad];
        for b in &mut self.blocks {
            v.push(&mut b.ln1_g.grad);
            v.push(&mut b.ln1_b.grad);
            v.push(&mut b.q_w.grad);
            v.push(&mut b.q_b.grad);
            v.push(&mut b.k_w.grad);
            v.push(&mut b.k_b.grad);
            v.push(&mut b.v_w.grad);
            v.push(&mut b.v_b.grad);
            v.push(&mut b.o_w.grad);
            v.push(&mut b.o_b.grad);
            v.push(&mut b.ln2_g.grad);
            v.push(&mut b.ln2_b.grad);
            v.push(&mut b.f1_w.grad);
            v.push(&mut b.f1_b.grad);
            v.push(&mut b.f2_w.grad);
            v.push(&mut b.f2_b.grad);
        }
        v.push(&mut self.lnf_g.grad);
        v.push(&mut self.lnf_b.grad);
        v.push(&mut self.head_w.grad);
        v
    }

    /// Overwrite every gradient tensor from a flat vector in canonical
    /// parameter order (as produced by [`TransformerNet::backward_grads`], e.g.
    /// a summed cross-shard gradient), so a subsequent [`TransformerNet::step`]
    /// applies it. Panics if `flat` has the wrong length.
    pub(crate) fn load_grads(&mut self, flat: &[f32]) {
        let mut offset = 0usize;
        for g in self.grad_tensors_mut() {
            let n = g.data.len();
            g.data.copy_from_slice(&flat[offset..offset + n]);
            offset += n;
        }
        debug_assert_eq!(offset, flat.len(), "load_grads: length mismatch");
    }

    /// Copy of every parameter tensor's data (the fp32 master weights).
    pub(crate) fn snapshot_weights(&mut self) -> Vec<Vec<f32>> {
        self.param_tensors_mut()
            .iter()
            .map(|t| t.data.clone())
            .collect()
    }

    /// Snap exactly the tensors the GGUF writer stores as Q8_0 onto their
    /// Q8_0 values (32-blocks, f16 scale, same axis as GGML): the token
    /// embedding along its rows, `[in, out]` weights down each column, the
    /// untied head likewise. 1-D params, the position embedding (GGUF keeps
    /// `position_embd` F32, the llama-quantize convention) and matrices whose
    /// GGML row length is not a multiple of 32 (stored F32) are left alone.
    /// With weight tying the head is re-mirrored from the snapped embedding.
    pub(crate) fn fake_quantize_weights(&mut self) {
        use crate::quant::{fake_quantize_q8_0_cols, fake_quantize_q8_0_rows};
        let e = self.embed_dim;
        fake_quantize_q8_0_rows(&mut self.tok_emb.data.data, e);
        for b in &mut self.blocks {
            for w in [
                &mut b.q_w,
                &mut b.k_w,
                &mut b.v_w,
                &mut b.o_w,
                &mut b.f1_w,
                &mut b.f2_w,
            ] {
                let (rows, cols) = (w.data.shape[0], w.data.shape[1]);
                fake_quantize_q8_0_cols(&mut w.data.data, rows, cols);
            }
        }
        if self.tie_weights {
            self.sync_tied_head();
        } else {
            let v = self.vocab_size;
            fake_quantize_q8_0_cols(&mut self.head_w.data.data, e, v);
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
                &b.ln1_g, &b.ln1_b, &b.q_w, &b.q_b, &b.k_w, &b.k_b, &b.v_w, &b.v_b, &b.o_w, &b.o_b,
                &b.ln2_g, &b.ln2_b, &b.f1_w, &b.f1_b, &b.f2_w, &b.f2_b,
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
    }

    /// Forward pass over `tokens` (length must be a multiple of `context_len`;
    /// each `context_len` chunk is one sequence in the batch). Returns logits
    /// of shape [B·T, vocab] plus the activation cache for `backward`.
    ///
    /// Dropout-free — equivalent to [`TransformerNet::forward_train`] with
    /// `dropout = 0`. Used by inference, evaluation, and gradient checks.
    pub fn forward(&self, tokens: &[usize]) -> Result<(Tensor, FwdCache)> {
        self.forward_train(tokens, 0.0, 0)
    }

    /// Forward pass with optional FFN-hidden dropout (T7). When `dropout > 0`,
    /// each block's post-GELU hidden activations are inverted-dropout masked
    /// using a local generator seeded by `seed` (so the masks are deterministic
    /// for a given seed and a worker can reproduce them in `backward` from the
    /// cache). `dropout = 0` is the plain, deterministic forward.
    pub fn forward_train(
        &self,
        tokens: &[usize],
        dropout: f32,
        seed: u64,
    ) -> Result<(Tensor, FwdCache)> {
        let t = self.context_len;
        if tokens.is_empty() || !tokens.len().is_multiple_of(t) {
            return Err(InferError::DimMismatch(format!(
                "token count {} must be a non-zero multiple of context_len {t}",
                tokens.len()
            )));
        }
        let batch = tokens.len() / t;
        let m = tokens.len();
        let c = self.embed_dim;
        let dropout = dropout.clamp(0.0, 0.95);
        let mut drng = Rng::from_state(seed | 1); // local mask generator

        // Embedding: token + positional lookup.
        let mut x = vec![0.0f32; m * c];
        for (r, &tok) in tokens.iter().enumerate() {
            if tok >= self.vocab_size {
                return Err(InferError::DimMismatch(format!(
                    "token {tok} out of range for vocab {}",
                    self.vocab_size
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
            let h_act = h_pre.map(crate::activation::gelu);
            // Inverted FFN dropout: zero each hidden unit with prob `dropout`,
            // scale survivors by 1/(1-dropout) so the expectation is unchanged
            // and inference needs no rescaling.
            let (h_ffn, ffn_dropout) = if dropout > 0.0 {
                let scale = 1.0 / (1.0 - dropout);
                let mask: Vec<f32> = (0..h_act.data.len())
                    .map(|_| {
                        if drng.next_f32() < dropout {
                            0.0
                        } else {
                            scale
                        }
                    })
                    .collect();
                let dropped: Vec<f32> =
                    h_act.data.iter().zip(&mask).map(|(&h, &m)| h * m).collect();
                (Tensor::new(h_act.shape.clone(), dropped)?, Some(mask))
            } else {
                (h_act, None)
            };
            let ff2 = linear_fwd(&h_ffn, &blk.f2_w.data, &blk.f2_b.data)?;
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
                h_pre,
                h_act: h_ffn,
                ffn_dropout,
            });
        }

        let (norm_f, xhat_f, inv_std_f) = ln_fwd(&cur, &self.lnf_g.data, &self.lnf_b.data)?;
        // LM head has no bias (GPT-2 / llama.cpp graph).
        let no_bias = Tensor::vector(vec![0.0; self.vocab_size]);
        let logits = linear_fwd(&norm_f, &self.head_w.data, &no_bias)?;

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
    /// whole network, storing the result in every parameter's `.grad`.
    ///
    /// (Each training step calls [`TransformerNet::zero_grad`] then a single
    /// `backward`, so storing is equivalent to the previous accumulate.)
    pub fn backward(&mut self, dlogits: &Tensor, cache: &FwdCache) -> Result<()> {
        let flat = self.backward_grads(dlogits, cache)?;
        self.load_grads(&flat);
        Ok(())
    }

    /// Backprop `dlogits` against the (read-only) weights and return the flat
    /// gradient vector in canonical parameter order — the same layout
    /// [`TransformerNet::load_grads`] consumes.
    ///
    /// Takes `&self` and allocates only the gradient buffer (≈4 B/param), so the
    /// data-parallel workers in [`train_transformer_epoch_threaded`] can share
    /// the master weights instead of deep-cloning the whole network
    /// (weights+grad+Adam moments ≈16 B/param per worker). `backward` is the
    /// in-place wrapper that stores the result into the owned parameters.
    pub(crate) fn backward_grads(&self, dlogits: &Tensor, cache: &FwdCache) -> Result<Vec<f32>> {
        let t = self.context_len;
        let batch = cache.batch;
        let c = self.embed_dim;

        // LM head + final LayerNorm (each produced exactly once).
        let (g_head_w, _g_head_b, d_normf) = linear_bwd(&cache.norm_f, &self.head_w.data, dlogits)?;
        let (mut dx, g_lnf_g, g_lnf_b) =
            ln_bwd(&d_normf, &cache.xhat_f, &cache.inv_std_f, &self.lnf_g.data)?;

        // Per-block grads, computed in reverse but stored by block index so they
        // can be emitted in forward (canonical) order.
        let mut block_grads: Vec<Option<BlockGrads>> =
            (0..self.blocks.len()).map(|_| None).collect();
        for (bi, (blk, bc)) in self.blocks.iter().zip(&cache.blocks).enumerate().rev() {
            // out = x_attn + ffn2(gelu(ffn1(norm2)))
            let d_out = dx;
            let (g_f2_w, g_f2_b, dh) = linear_bwd(&bc.h_act, &blk.f2_w.data, &d_out)?;
            // Backward through (optional) dropout then GELU. The dropout
            // mask×scale carries the inverted-dropout scaling into the gradient.
            let dh_pre = Tensor::new(
                dh.shape.clone(),
                match &bc.ffn_dropout {
                    Some(mask) => dh
                        .data
                        .iter()
                        .zip(&bc.h_pre.data)
                        .zip(mask)
                        .map(|((&g, &h), &m)| g * m * crate::activation::gelu_grad(h))
                        .collect(),
                    None => dh
                        .data
                        .iter()
                        .zip(&bc.h_pre.data)
                        .map(|(&g, &h)| g * crate::activation::gelu_grad(h))
                        .collect(),
                },
            )?;
            let (g_f1_w, g_f1_b, d_norm2) = linear_bwd(&bc.norm2, &blk.f1_w.data, &dh_pre)?;
            let (d_xattn_ln, g_ln2_g, g_ln2_b) =
                ln_bwd(&d_norm2, &bc.xhat2, &bc.inv_std2, &blk.ln2_g.data)?;
            // residual 2: gradient flows both into the FFN branch and straight through
            let d_xattn = ops::add(&d_out, &d_xattn_ln)?;

            // x_attn = x + out_proj(concat)
            let (g_o_w, g_o_b, d_concat) = linear_bwd(&bc.concat, &blk.o_w.data, &d_xattn)?;
            let (dq, dk, dv) = attn_bwd(
                &d_concat,
                &bc.q,
                &bc.k,
                &bc.v,
                &bc.probs,
                batch,
                t,
                self.num_heads,
            )?;
            let (g_q_w, g_q_b, d_n1a) = linear_bwd(&bc.norm1, &blk.q_w.data, &dq)?;
            let (g_k_w, g_k_b, d_n1b) = linear_bwd(&bc.norm1, &blk.k_w.data, &dk)?;
            let (g_v_w, g_v_b, d_n1c) = linear_bwd(&bc.norm1, &blk.v_w.data, &dv)?;
            let d_norm1 = ops::add(&ops::add(&d_n1a, &d_n1b)?, &d_n1c)?;
            let (d_x_ln, g_ln1_g, g_ln1_b) =
                ln_bwd(&d_norm1, &bc.xhat1, &bc.inv_std1, &blk.ln1_g.data)?;
            // residual 1
            dx = ops::add(&d_xattn, &d_x_ln)?;

            block_grads[bi] = Some(BlockGrads {
                ln1_g: g_ln1_g,
                ln1_b: g_ln1_b,
                q_w: g_q_w,
                q_b: g_q_b,
                k_w: g_k_w,
                k_b: g_k_b,
                v_w: g_v_w,
                v_b: g_v_b,
                o_w: g_o_w,
                o_b: g_o_b,
                ln2_g: g_ln2_g,
                ln2_b: g_ln2_b,
                f1_w: g_f1_w,
                f1_b: g_f1_b,
                f2_w: g_f2_w,
                f2_b: g_f2_b,
            });
        }

        // Embedding: scatter-add row gradients into local tables.
        let mut g_tok = vec![0.0f32; self.tok_emb.data.data.len()];
        let mut g_pos = vec![0.0f32; self.pos_emb.data.data.len()];
        for (r, &tok) in cache.tokens.iter().enumerate() {
            let xb = r * c;
            let tb = tok * c;
            let pb = (r % t) * c;
            for d in 0..c {
                g_tok[tb + d] += dx.data[xb + d];
                g_pos[pb + d] += dx.data[xb + d];
            }
        }

        // Assemble the flat gradient in canonical parameter order.
        let mut out = Vec::with_capacity(self.num_params());
        out.extend_from_slice(&g_tok);
        out.extend_from_slice(&g_pos);
        for bg in block_grads {
            let bg = bg.expect("every block's gradient was computed");
            for g in [
                &bg.ln1_g, &bg.ln1_b, &bg.q_w, &bg.q_b, &bg.k_w, &bg.k_b, &bg.v_w, &bg.v_b,
                &bg.o_w, &bg.o_b, &bg.ln2_g, &bg.ln2_b, &bg.f1_w, &bg.f1_b, &bg.f2_w, &bg.f2_b,
            ] {
                out.extend_from_slice(&g.data);
            }
        }
        out.extend_from_slice(&g_lnf_g.data);
        out.extend_from_slice(&g_lnf_b.data);
        out.extend_from_slice(&g_head_w.data);
        Ok(out)
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
    }

    /// Apply one Adam update to every parameter. When a learning-rate schedule
    /// is set ([`TransformerNet::set_lr_schedule`]), the Adam learning rate for
    /// this step is taken from `schedule.lr_at(step_t)` instead of the fixed
    /// `adam.lr`.
    pub fn step(&mut self, adam: &Adam) -> Result<()> {
        self.step_t += 1;
        let t = self.step_t;
        // Adam is `Copy`; override the learning rate when scheduled and fold in
        // the net's decoupled weight decay (AdamW).
        let mut adam = *adam;
        if let Some(sched) = self.lr_schedule {
            adam.lr = sched.lr_at(t);
            vprintln!(
                "[train_transformer::step] step {t}: scheduled lr={:.6e}",
                adam.lr
            );
        }
        if self.weight_decay != 0.0 {
            adam.weight_decay = self.weight_decay;
        }
        let adam = &adam;
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
        Ok(())
    }

    /// Export to an inference `Sequential`:
    /// `Embedding → blocks → LayerNorm → Linear → Softmax`.
    /// Serializes to GGUF v3 and is compatible with `TransformerSLMModel`.
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
                b.ln1_g.data.data.clone(),
                b.ln1_b.data.data.clone(),
                b.q_w.data.data.clone(),
                b.q_b.data.data.clone(),
                b.k_w.data.data.clone(),
                b.k_b.data.data.clone(),
                b.v_w.data.data.clone(),
                b.v_b.data.data.clone(),
                b.o_w.data.data.clone(),
                b.o_b.data.data.clone(),
                b.ln2_g.data.data.clone(),
                b.ln2_b.data.data.clone(),
                b.f1_w.data.data.clone(),
                b.f1_b.data.data.clone(),
                b.f2_w.data.data.clone(),
                b.f2_b.data.data.clone(),
            )?));
        }
        m.push(Box::new(LayerNorm::new(
            self.embed_dim,
            self.lnf_g.data.data.clone(),
            self.lnf_b.data.data.clone(),
        )?));
        // With weight tying the shipped head is the (current) transpose of the
        // token embedding; otherwise it is the independently trained head.
        let head_w_data = if self.tie_weights {
            let (vocab, embed) = (self.vocab_size, self.embed_dim);
            let mut w = vec![0.0f32; embed * vocab];
            for e in 0..embed {
                for v in 0..vocab {
                    w[e * vocab + v] = self.tok_emb.data.data[v * embed + e];
                }
            }
            w
        } else {
            self.head_w.data.data.clone()
        };
        m.push(Box::new(Linear::new(
            self.embed_dim,
            self.vocab_size,
            head_w_data,
            vec![0.0; self.vocab_size],
        )?));
        m.push(Box::new(ActivationLayer::new(
            crate::activation::Activation::Softmax,
        )));
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
    train_transformer_steps(
        net,
        tokens,
        batch_size,
        adam,
        rng,
        1,
        usize::MAX,
        &mut |_, _| Ok(()),
    )
}

/// Callback run after every optimizer step of [`train_transformer_steps`],
/// with the updated net and the RNG (e.g. to write a checkpoint).
pub type StepHook<'a> = dyn FnMut(&TransformerNet, &Rng) -> Result<()> + 'a;

/// One epoch (see [`train_transformer_epoch_threaded`]) that stops after at
/// most `max_steps` optimizer steps and calls `on_step` after each one, so a
/// driver can enforce a token/step budget mid-epoch and checkpoint as it goes.
/// Returns the mean loss over the steps actually run.
#[allow(clippy::too_many_arguments)]
pub fn train_transformer_steps(
    net: &mut TransformerNet,
    tokens: &[usize],
    batch_size: usize,
    adam: &Adam,
    rng: &mut Rng,
    threads: usize,
    max_steps: usize,
    on_step: &mut StepHook,
) -> Result<f32> {
    // One shard (or a single-sequence batch, or wasm32) is the serial path.
    #[cfg(not(target_arch = "wasm32"))]
    if threads.max(1).min(batch_size.max(1)) > 1 {
        return epoch_sharded(
            net, tokens, batch_size, adam, rng, threads, max_steps, on_step,
        );
    }
    let _ = threads;
    epoch_serial(net, tokens, batch_size, adam, rng, max_steps, on_step)
}

fn epoch_serial(
    net: &mut TransformerNet,
    tokens: &[usize],
    batch_size: usize,
    adam: &Adam,
    rng: &mut Rng,
    max_steps: usize,
    on_step: &mut StepHook,
) -> Result<f32> {
    use crate::loss::softmax_cross_entropy;
    let t = net.context_len();
    // Shuffle the epoch's windows once and draw minibatches without
    // replacement (T4), so an epoch visits every window exactly once.
    let perm = net.epoch_window_starts(tokens.len(), rng)?;
    let num_windows = perm.len();
    let steps = num_windows.div_ceil(batch_size).min(max_steps);
    let mut total = 0.0f32;
    for step in 0..steps {
        let batch = &perm[step * batch_size..((step + 1) * batch_size).min(num_windows)];
        let mut input = Vec::with_capacity(batch.len() * t);
        let mut targets = Vec::with_capacity(batch.len() * t);
        for &start in batch {
            input.extend_from_slice(&tokens[start..start + t]);
            targets.extend_from_slice(&tokens[start + 1..start + t + 1]);
        }
        // Weight tying: mirror tok_embᵀ into the head before the forward (T9).
        net.sync_tied_head();
        // QAT: gradients are computed at the int8-snapped weights, but the
        // optimizer updates the fp32 masters (straight-through estimator).
        let masters = if net.qat_enabled() {
            let snapshot = net.snapshot_weights();
            net.fake_quantize_weights();
            Some(snapshot)
        } else {
            None
        };
        // FFN dropout (T7): draw a per-step mask seed only when enabled, so the
        // RNG stream (and thus reproducibility) is untouched when dropout is off.
        let dropout = net.dropout;
        let drop_seed = if dropout > 0.0 { rng.next_u64() } else { 0 };
        let (logits, cache) = net.forward_train(&input, dropout, drop_seed)?;
        let (loss, dlogits) = softmax_cross_entropy(&logits, &targets)?;
        net.zero_grad();
        net.backward(&dlogits, &cache)?;
        if let Some(snapshot) = &masters {
            net.restore_weights(snapshot);
        }
        // Sum the head gradient back into the shared embedding (T9).
        net.fold_tied_head_grad();
        if let Some(max_norm) = net.grad_clip {
            net.clip_grad_norm(max_norm);
        }
        net.step(adam)?;
        on_step(net, rng)?;
        total += loss;
    }
    let avg = total / steps.max(1) as f32;
    if verbose::is_verbose() {
        vprintln!(
            "[train_transformer::epoch] steps={}, mean loss={:.6}",
            steps,
            avg
        );
    }
    Ok(avg)
}

/// Data-parallel version of [`train_transformer_epoch`]: each minibatch is split
/// into up to `threads` shards processed concurrently on separate OS threads,
/// then their gradients are summed and a single Adam update is applied.
///
/// Parallelism is **across training examples** (sequences), complementing the
/// per-matmul row parallelism in [`crate::ops::matmul`]. It is built only on
/// `std::thread::scope` — no external crates, no `unsafe`. Each shard clones the
/// network (cheap relative to forward/backward for non-trivial models), computes
/// gradients over its sequences against the shared weights, and the partial
/// gradients are reduced in a **fixed shard order**, so a run is reproducible for
/// a given `threads` value. With `threads <= 1` (or a single sequence) this is
/// bit-for-bit identical to [`train_transformer_epoch`]; with more threads the
/// only differences are the floating-point regrouping of the gradient sum.
///
/// The RNG is drawn exactly as in the serial path (the whole minibatch is
/// sampled up front, then sharded), so the sequence of training windows does not
/// depend on the thread count. QAT semantics are preserved: weights are
/// int8-snapped on the shared master before the shards fork, and the fp32
/// masters are restored before the optimizer step.
///
/// On `wasm32` (no threads) this transparently runs the serial path.
pub fn train_transformer_epoch_threaded(
    net: &mut TransformerNet,
    tokens: &[usize],
    batch_size: usize,
    adam: &Adam,
    rng: &mut Rng,
    threads: usize,
) -> Result<f32> {
    train_transformer_steps(
        net,
        tokens,
        batch_size,
        adam,
        rng,
        threads,
        usize::MAX,
        &mut |_, _| Ok(()),
    )
}

#[cfg(not(target_arch = "wasm32"))]
#[allow(clippy::too_many_arguments)]
fn epoch_sharded(
    net: &mut TransformerNet,
    tokens: &[usize],
    batch_size: usize,
    adam: &Adam,
    rng: &mut Rng,
    threads: usize,
    max_steps: usize,
    on_step: &mut StepHook,
) -> Result<f32> {
    {
        use crate::loss::softmax_cross_entropy;
        let nshards = threads.max(1).min(batch_size.max(1));
        let t = net.context_len();
        // Same per-epoch shuffle as the serial path (T4): identical RNG use, so
        // a fixed seed gives the same minibatches regardless of thread count.
        let perm = net.epoch_window_starts(tokens.len(), rng)?;
        let num_windows = perm.len();
        let steps = num_windows.div_ceil(batch_size).min(max_steps);
        let mut total = 0.0f32;

        for step in 0..steps {
            // The step's minibatch is the next slice of the permutation (the
            // final batch may be smaller than `batch_size`).
            let batch = &perm[step * batch_size..((step + 1) * batch_size).min(num_windows)];
            let cur_bs = batch.len();
            let total_rows = (cur_bs * t) as f32;
            let mut input = Vec::with_capacity(cur_bs * t);
            let mut targets = Vec::with_capacity(cur_bs * t);
            for &start in batch {
                input.extend_from_slice(&tokens[start..start + t]);
                targets.extend_from_slice(&tokens[start + 1..start + t + 1]);
            }

            // Weight tying: mirror tok_embᵀ into the shared head before forking,
            // so every shard's forward sees the tied head (T9).
            net.sync_tied_head();
            // QAT: snap the shared master weights before forking the shards, so
            // every shard's forward/backward runs at the int8 grid.
            let masters = if net.qat_enabled() {
                let snapshot = net.snapshot_weights();
                net.fake_quantize_weights();
                Some(snapshot)
            } else {
                None
            };

            // FFN dropout (T7): one base mask seed per step, offset per shard so
            // each shard's examples get independent masks. RNG is untouched off.
            let dropout = net.dropout;
            let base_seed = if dropout > 0.0 { rng.next_u64() } else { 0 };

            let per = cur_bs.div_ceil(nshards);
            let net_ref: &TransformerNet = net;
            let input_ref = &input;
            let targets_ref = &targets;

            // Each shard returns its weighted (loss, flat-gradients). Weighting by
            // shard_rows / total_rows makes the summed result equal the serial
            // gradient, which normalises by the full batch's row count.
            let results: Vec<Result<(f32, Vec<f32>)>> = std::thread::scope(|scope| {
                let mut handles = Vec::new();
                let mut w0 = 0usize;
                while w0 < cur_bs {
                    let w1 = (w0 + per).min(cur_bs);
                    handles.push(scope.spawn(move || -> Result<(f32, Vec<f32>)> {
                        // Share the master weights read-only (no clone): forward
                        // and backward_grads borrow `net_ref` and allocate only a
                        // flat gradient buffer (≈4 B/param) instead of a full
                        // network copy (weights+grad+Adam moments ≈16 B/param).
                        let sub_in = &input_ref[w0 * t..w1 * t];
                        let sub_tg = &targets_ref[w0 * t..w1 * t];
                        let (logits, cache) = net_ref.forward_train(
                            sub_in,
                            dropout,
                            base_seed.wrapping_add(w0 as u64 + 1),
                        )?;
                        let (loss, dlogits) = softmax_cross_entropy(&logits, sub_tg)?;
                        let shard_rows = ((w1 - w0) * t) as f32;
                        let weight = shard_rows / total_rows;
                        let mut g = net_ref.backward_grads(&dlogits, &cache)?;
                        for x in &mut g {
                            *x *= weight;
                        }
                        Ok((loss * weight, g))
                    }));
                    w0 = w1;
                }
                handles
                    .into_iter()
                    .map(|h| h.join().expect("ferrum: training worker thread panicked"))
                    .collect()
            });

            // Reduce in deterministic shard order (independent of scheduling).
            let mut summed: Option<Vec<f32>> = None;
            let mut step_loss = 0.0f32;
            for r in results {
                let (loss, g) = r?;
                step_loss += loss;
                match &mut summed {
                    None => summed = Some(g),
                    Some(acc) => {
                        for (a, b) in acc.iter_mut().zip(&g) {
                            *a += b;
                        }
                    }
                }
            }
            let summed = summed.expect("at least one shard always runs");

            if let Some(snapshot) = &masters {
                net.restore_weights(snapshot);
            }
            net.load_grads(&summed);
            // Sum the head gradient back into the shared embedding (T9).
            net.fold_tied_head_grad();
            if let Some(max_norm) = net.grad_clip {
                net.clip_grad_norm(max_norm);
            }
            net.step(adam)?;
            on_step(net, rng)?;
            total += step_loss;
        }

        let avg = total / steps.max(1) as f32;
        if verbose::is_verbose() {
            vprintln!("[train_transformer::epoch_threaded] shards={nshards}, steps={steps}, mean loss={avg:.6}");
        }
        Ok(avg)
    }
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

        check!(tok_emb, "tok_emb", &[4, 9, 13]); // rows of tokens 1,2,3
        check!(pos_emb, "pos_emb", &[0, 5, 11]);
        check!(lnf_g, "lnf_g", &[0, 3]);
        check!(lnf_b, "lnf_b", &[1]);
        check!(head_w, "head_w", &[0, 7, 19]);
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
    fn qat_snaps_only_q8_0_eligible_matrices() {
        let mut rng = Rng::new(2);
        // embed 48: ne0 = 48 is not a multiple of 32, so those tensors stay f32;
        // hidden 64: ffn2 has ne0 = 64, so it is snapped.
        let mut net = TransformerNet::new(40, 8, 48, 2, 64, 1, &mut rng).unwrap();
        let before = net.snapshot_weights();
        net.fake_quantize_weights();
        let after = net.snapshot_weights();
        assert!(
            before.iter().zip(&after).any(|(a, b)| a != b),
            "ffn2 must be snapped"
        );
        let shapes: Vec<Vec<usize>> = net
            .param_tensors_mut()
            .iter()
            .map(|t| t.shape.clone())
            .collect();
        let mut changed_2d = 0;
        for ((sh, a), b) in shapes.iter().zip(&before).zip(&after) {
            if sh.len() == 1 {
                assert_eq!(a, b, "1-D params must never be snapped");
            } else if a != b {
                changed_2d += 1;
                assert_eq!(sh, &vec![64, 48], "only ffn2 [64,48] is eligible");
            }
        }
        assert_eq!(changed_2d, 1);
    }

    #[test]
    fn qat_slm_holds_snapped_weights_and_tied_head_follows() {
        let mut rng = Rng::new(3);
        let mut net = TransformerNet::new(8, 4, 32, 2, 64, 1, &mut rng).unwrap();
        net.set_weight_tying(true);
        let pos_before = net.pos_emb.data.data.clone();
        net.fake_quantize_weights();
        // position_embd stays F32 in GGUF (llama-quantize convention): never snapped.
        assert_eq!(
            net.pos_emb.data.data, pos_before,
            "pos_emb must not be snapped"
        );
        let mut once = net.snapshot_weights();
        net.fake_quantize_weights();
        assert_eq!(net.snapshot_weights(), once, "snap must be idempotent");
        // head mirrors the snapped embedding
        once.clear();
        let (v, e) = (8, 32);
        for i in 0..e {
            for j in 0..v {
                assert_eq!(
                    net.head_w.data.data[i * v + j],
                    net.tok_emb.data.data[j * e + i]
                );
            }
        }
    }

    #[test]
    fn qat_snapshot_quantize_restore_cycle() {
        let mut rng = Rng::new(11);
        let mut net = TransformerNet::new(8, 4, 32, 2, 64, 1, &mut rng).unwrap();
        let masters = net.snapshot_weights();
        net.fake_quantize_weights();
        let quantized = net.snapshot_weights();
        // Quantization actually moved (large) weights off their fp32 values.
        assert!(
            masters.iter().zip(&quantized).any(|(m, q)| m != q),
            "fake quantization changed no weights"
        );
        // Per-channel snapping is idempotent — re-quantizing lands on the same
        // grid (so every value already sits on its channel's int8 grid).
        net.fake_quantize_weights();
        assert_eq!(
            net.snapshot_weights(),
            quantized,
            "fake quantization is not idempotent"
        );
        // Restore must bring back the fp32 masters exactly.
        net.restore_weights(&masters);
        assert_eq!(net.snapshot_weights(), masters);
    }

    #[test]
    fn qat_model_survives_int8_export_with_small_drift() {
        use crate::loader::{from_bytes, to_bytes_quantized};
        use crate::meta::{ModelMetadata, Normalizer, TaskType};
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
        let norm = Normalizer {
            means: vec![],
            stds: vec![],
        };
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
    fn head_has_no_bias() {
        let mut rng = Rng::new(9);
        let net = TransformerNet::new(40, 8, 16, 2, 32, 2, &mut rng).unwrap();
        // tok 640 + pos 128 + 2 blocks * 2224 + lnf 32 + head_w 640; no `+ vocab` bias term.
        assert_eq!(net.num_params(), 640 + 128 + 2 * 2224 + 32 + 640);
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
    fn exported_model_roundtrips_gguf() {
        use crate::loader::{from_bytes, to_bytes};
        use crate::meta::{ModelMetadata, Normalizer, TaskType};
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
        let norm = Normalizer {
            means: vec![],
            stds: vec![],
        };
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

    // ─────────────────────────────────────────────────────────────────────────
    // Multi-threaded (data-parallel) training
    // ─────────────────────────────────────────────────────────────────────────

    /// `threads <= 1` must be the serial path, bit-for-bit.
    #[test]
    fn threaded_one_shard_matches_serial_bitwise() {
        let mut rng0 = Rng::new(7);
        let base = TransformerNet::new(6, 4, 8, 2, 16, 1, &mut rng0).unwrap();
        let tokens: Vec<usize> = (0..200).map(|i| (i * 7) % 6).collect();
        let adam = Adam::new(0.01);

        let mut serial = base.clone();
        let mut rs = Rng::new(99);
        let ls = train_transformer_epoch(&mut serial, &tokens, 8, &adam, &mut rs).unwrap();

        let mut threaded = base.clone();
        let mut rt = Rng::new(99);
        let lt =
            train_transformer_epoch_threaded(&mut threaded, &tokens, 8, &adam, &mut rt, 1).unwrap();

        assert_eq!(ls, lt, "loss differs with a single shard");
        assert_eq!(
            serial.snapshot_weights(),
            threaded.snapshot_weights(),
            "weights differ with a single shard"
        );
    }

    /// Multi-shard data-parallel must match the serial gradient/step to within
    /// floating-point regrouping error after one step (batch = all windows).
    #[test]
    fn threaded_multishard_matches_serial_one_step() {
        let mut rng0 = Rng::new(7);
        let base = TransformerNet::new(6, 4, 8, 2, 16, 1, &mut rng0).unwrap();
        let tokens: Vec<usize> = (0..200).map(|i| (i * 7) % 6).collect();
        let adam = Adam::new(0.05);
        let batch = tokens.len() - 4; // a single optimizer step over every window

        let mut serial = base.clone();
        let mut rs = Rng::new(1);
        let ls = train_transformer_epoch(&mut serial, &tokens, batch, &adam, &mut rs).unwrap();

        let mut par = base.clone();
        let mut rp = Rng::new(1);
        let lp =
            train_transformer_epoch_threaded(&mut par, &tokens, batch, &adam, &mut rp, 4).unwrap();

        assert!(
            (ls - lp).abs() < 1e-4,
            "loss diverged: serial {ls} vs threaded {lp}"
        );
        let sw = serial.snapshot_weights();
        let pw = par.snapshot_weights();
        for (a, b) in sw.iter().zip(&pw) {
            for (x, y) in a.iter().zip(b) {
                assert!((x - y).abs() < 1e-4, "weight diverged: {x} vs {y}");
            }
        }
    }

    /// The QAT path must also behave identically for a single shard.
    #[test]
    fn threaded_qat_one_shard_matches_serial() {
        let mut rng0 = Rng::new(13);
        let mut base = TransformerNet::new(6, 4, 8, 2, 16, 1, &mut rng0).unwrap();
        base.set_qat(true);
        let tokens: Vec<usize> = (0..200).map(|i| (i * 3) % 6).collect();
        let adam = Adam::new(0.01);

        let mut serial = base.clone();
        let mut rs = Rng::new(5);
        for _ in 0..3 {
            train_transformer_epoch(&mut serial, &tokens, 8, &adam, &mut rs).unwrap();
        }
        let mut threaded = base.clone();
        let mut rt = Rng::new(5);
        for _ in 0..3 {
            train_transformer_epoch_threaded(&mut threaded, &tokens, 8, &adam, &mut rt, 1).unwrap();
        }
        assert_eq!(serial.snapshot_weights(), threaded.snapshot_weights());
    }

    /// QAT and gradient clipping must also run (and stay finite, restoring the
    /// f32 masters) on the multi-shard path.
    #[test]
    fn threaded_multishard_qat_and_clip_train_and_keep_masters() {
        let mut rng0 = Rng::new(21);
        let mut net = TransformerNet::new(4, 32, 32, 2, 32, 1, &mut rng0).unwrap();
        net.set_qat(true);
        net.set_grad_clip(Some(0.05));
        let tokens: Vec<usize> = (0..200).map(|i| (i * 3) % 4).collect();
        let adam = Adam::new(0.01);
        let mut rng = Rng::new(9);
        let before = net.snapshot_weights();
        let first =
            train_transformer_epoch_threaded(&mut net, &tokens, 16, &adam, &mut rng, 3).unwrap();
        let mut last = first;
        for _ in 0..5 {
            last = train_transformer_epoch_threaded(&mut net, &tokens, 16, &adam, &mut rng, 3)
                .unwrap();
        }
        assert!(first.is_finite() && last < first, "{first} -> {last}");
        let after = net.snapshot_weights();
        assert!(after.iter().flatten().all(|v| v.is_finite()));
        assert_ne!(before, after);
        // masters are restored after each step: they are not sitting on the Q8_0 grid
        let mut snapped = after.clone();
        net.fake_quantize_weights();
        let grid = net.snapshot_weights();
        assert_ne!(
            snapped.remove(0),
            grid[0],
            "f32 masters must not be left snapped"
        );
    }

    /// Data-parallel training reduces loss like the serial path.
    #[test]
    fn threaded_multishard_reduces_loss() {
        let mut rng0 = Rng::new(7);
        let mut net = TransformerNet::new(4, 4, 8, 2, 16, 1, &mut rng0).unwrap();
        let tokens: Vec<usize> = (0..400).map(|i| i % 4).collect();
        let adam = Adam::new(0.01);
        let mut rng = Rng::new(123);
        let first =
            train_transformer_epoch_threaded(&mut net, &tokens, 16, &adam, &mut rng, 4).unwrap();
        let mut last = first;
        for _ in 0..30 {
            last = train_transformer_epoch_threaded(&mut net, &tokens, 16, &adam, &mut rng, 4)
                .unwrap();
        }
        assert!(
            last < first * 0.5,
            "threaded loss did not halve: {first:.4} → {last:.4}"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // AdamW weight decay + FFN dropout (T7)
    // ─────────────────────────────────────────────────────────────────────────

    /// Decoupled weight decay shrinks weight matrices but never biases or
    /// LayerNorm parameters (rank-1).
    #[test]
    fn weight_decay_decays_matrices_only() {
        let mut rng = Rng::new(1);
        let mut net = TransformerNet::new(6, 4, 8, 2, 16, 1, &mut rng).unwrap();
        net.set_weight_decay(0.1);
        assert!((net.weight_decay() - 0.1).abs() < 1e-9);
        // Known values; zero gradients so only weight decay acts.
        for v in net.tok_emb.data.data.iter_mut() {
            *v = 2.0;
        } // 2-D weight
        for v in net.blocks[0].f1_b.data.data.iter_mut() {
            *v = 2.0;
        } // 1-D bias
        for v in net.lnf_g.data.data.iter_mut() {
            *v = 2.0;
        } // 1-D LN gain
        net.zero_grad();
        let adam = Adam::new(0.5); // lr; decay coefficient comes from the net
        net.step(&adam).unwrap();
        // Weight matrix: 2 − 0.5·0.1·2 = 1.9
        assert!(
            (net.tok_emb.data.data[0] - 1.9).abs() < 1e-5,
            "tok_emb not decayed"
        );
        // Bias and LN gain untouched.
        assert!(
            (net.blocks[0].f1_b.data.data[0] - 2.0).abs() < 1e-6,
            "bias was decayed"
        );
        assert!(
            (net.lnf_g.data.data[0] - 2.0).abs() < 1e-6,
            "LN gain was decayed"
        );
    }

    #[test]
    fn weight_decay_trains_and_shrinks_weight_matrices() {
        let base = {
            let mut r = Rng::new(7);
            TransformerNet::new(6, 4, 8, 2, 16, 1, &mut r).unwrap()
        };
        let tokens: Vec<usize> = (0..200).map(|i| (i * 7) % 6).collect();
        let adam = Adam::new(0.01);
        // L2 norm over the actual weight matrices (where decay applies), not the
        // un-decayed LayerNorm gains/biases.
        let matrix_norm = |net: &TransformerNet| -> f32 {
            let b = &net.blocks[0];
            [
                &b.q_w,
                &b.k_w,
                &b.v_w,
                &b.o_w,
                &b.f1_w,
                &b.f2_w,
                &net.head_w,
                &net.tok_emb,
            ]
            .iter()
            .flat_map(|p| p.data.data.iter())
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt()
        };

        let mut plain = base.clone();
        let mut rp = Rng::new(3);
        let mut decayed = base.clone();
        decayed.set_weight_decay(0.3);
        let mut rd = Rng::new(3);

        let mut first = f32::NAN;
        let mut last = 0.0;
        for e in 0..30 {
            train_transformer_epoch(&mut plain, &tokens, 8, &adam, &mut rp).unwrap();
            let l = train_transformer_epoch(&mut decayed, &tokens, 8, &adam, &mut rd).unwrap();
            if e == 0 {
                first = l;
            }
            last = l;
        }
        assert!(
            last < first,
            "decayed run should still reduce loss: {first:.4} → {last:.4}"
        );
        assert!(
            matrix_norm(&decayed) < matrix_norm(&plain),
            "weight decay should shrink the weight matrices: {} vs {}",
            matrix_norm(&decayed),
            matrix_norm(&plain)
        );
    }

    #[test]
    fn dropout_forward_is_deterministic_per_seed_and_off_matches_plain() {
        let mut rng = Rng::new(1);
        let net = TransformerNet::new(6, 4, 8, 2, 32, 1, &mut rng).unwrap();
        let tokens = [1usize, 2, 3, 4];
        let (a, _) = net.forward_train(&tokens, 0.5, 123).unwrap();
        let (b, _) = net.forward_train(&tokens, 0.5, 123).unwrap();
        assert_eq!(a.data, b.data, "same seed must reproduce the dropout mask");
        let (c, _) = net.forward_train(&tokens, 0.5, 999).unwrap();
        assert_ne!(
            a.data, c.data,
            "a different seed must change the dropout pattern"
        );
        // dropout = 0 is exactly the plain forward.
        let (d, _) = net.forward_train(&tokens, 0.0, 123).unwrap();
        let (e, _) = net.forward(&tokens).unwrap();
        assert_eq!(d.data, e.data);
    }

    /// The dropout backward is correct: against a fixed mask (fixed seed), the
    /// analytic gradient matches central differences both for the FFN-2 weight
    /// (directly downstream of dropout) and the FFN-1 weight (upstream, so the
    /// mask must propagate back through GELU).
    #[test]
    fn dropout_gradient_is_correct_for_fixed_mask() {
        let mut rng = Rng::new(3);
        let mut net = TransformerNet::new(5, 3, 4, 2, 8, 1, &mut rng).unwrap();
        let tokens = [1usize, 2, 3];
        let targets = [2usize, 3, 4];
        let (seed, p) = (77u64, 0.5f32);

        let (logits, cache) = net.forward_train(&tokens, p, seed).unwrap();
        let (_, dl) = softmax_cross_entropy(&logits, &targets).unwrap();
        net.zero_grad();
        net.backward(&dl, &cache).unwrap();

        let eps = 1e-2f32;
        macro_rules! check {
            ($field:ident, $idx:expr) => {
                for &i in $idx {
                    let analytic = net.blocks[0].$field.grad.data[i];
                    net.blocks[0].$field.data.data[i] += eps;
                    let lp = softmax_cross_entropy(
                        &net.forward_train(&tokens, p, seed).unwrap().0,
                        &targets,
                    )
                    .unwrap()
                    .0;
                    net.blocks[0].$field.data.data[i] -= 2.0 * eps;
                    let lm = softmax_cross_entropy(
                        &net.forward_train(&tokens, p, seed).unwrap().0,
                        &targets,
                    )
                    .unwrap()
                    .0;
                    net.blocks[0].$field.data.data[i] += eps;
                    let numeric = (lp - lm) / (2.0 * eps);
                    assert!(
                        (numeric - analytic).abs() < 6e-3 + 0.03 * analytic.abs(),
                        "{}[{i}]: analytic={analytic:.6} numeric={numeric:.6}",
                        stringify!($field)
                    );
                }
            };
        }
        check!(f2_w, &[0usize, 5, 11]);
        check!(f1_w, &[0usize, 7, 15]);
    }

    #[test]
    fn training_with_dropout_reduces_loss() {
        let mut rng = Rng::new(7);
        let mut net = TransformerNet::new(4, 4, 8, 2, 16, 1, &mut rng).unwrap();
        net.set_dropout(0.2);
        assert!((net.dropout() - 0.2).abs() < 1e-6);
        let tokens: Vec<usize> = (0..200).map(|i| i % 4).collect();
        let adam = Adam::new(0.01);
        let first = train_transformer_epoch(&mut net, &tokens, 8, &adam, &mut rng).unwrap();
        let mut last = first;
        for _ in 0..40 {
            last = train_transformer_epoch(&mut net, &tokens, 8, &adam, &mut rng).unwrap();
        }
        assert!(
            last < first * 0.7,
            "dropout training did not reduce loss: {first:.4} → {last:.4}"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Checkpoint / resume mid-training (T6)
    // ─────────────────────────────────────────────────────────────────────────

    /// Resuming from a mid-training checkpoint reproduces uninterrupted training
    /// bit-for-bit — proving weights, Adam moments, step counter, and RNG state
    /// are all preserved.
    #[test]
    fn checkpoint_resume_matches_uninterrupted_training() {
        let base = {
            let mut r = Rng::new(7);
            let mut n = TransformerNet::new(6, 4, 8, 2, 16, 2, &mut r).unwrap();
            n.set_qat(true); // exercise the QAT path through the checkpoint
            n
        };
        let tokens: Vec<usize> = (0..300).map(|i| (i * 7) % 6).collect();
        let adam = Adam::new(0.02);

        // Uninterrupted: 6 epochs straight through.
        let mut net_a = base.clone();
        let mut rng_a = Rng::new(123);
        for _ in 0..6 {
            train_transformer_epoch(&mut net_a, &tokens, 16, &adam, &mut rng_a).unwrap();
        }

        // Interrupted: 3 epochs, checkpoint, reload, 3 more.
        let mut net_b = base.clone();
        let mut rng_b = Rng::new(123);
        for _ in 0..3 {
            train_transformer_epoch(&mut net_b, &tokens, 16, &adam, &mut rng_b).unwrap();
        }
        let bytes = net_b.save_checkpoint(&rng_b);
        drop(net_b);
        let (mut net_c, mut rng_c) = TransformerNet::load_checkpoint(&bytes).unwrap();
        assert!(net_c.qat_enabled(), "QAT flag must survive the checkpoint");
        for _ in 0..3 {
            train_transformer_epoch(&mut net_c, &tokens, 16, &adam, &mut rng_c).unwrap();
        }

        assert_eq!(net_a.step_count(), net_c.step_count());
        assert_eq!(
            net_a.snapshot_weights(),
            net_c.snapshot_weights(),
            "resumed training diverged from the uninterrupted run"
        );
    }

    #[test]
    fn checkpoint_roundtrip_preserves_flags_and_rng() {
        let mut r = Rng::new(1);
        let mut net = TransformerNet::new(6, 4, 8, 2, 16, 2, &mut r).unwrap();
        net.set_qat(true);
        net.set_weight_tying(true);
        let tokens: Vec<usize> = (0..100).map(|i| i % 6).collect();
        let adam = Adam::new(0.01);
        let mut rng = Rng::new(5);
        train_transformer_epoch(&mut net, &tokens, 8, &adam, &mut rng).unwrap();

        let bytes = net.save_checkpoint(&rng);
        let (mut net2, rng2) = TransformerNet::load_checkpoint(&bytes).unwrap();
        assert_eq!(net.snapshot_weights(), net2.snapshot_weights());
        assert_eq!(net.step_count(), net2.step_count());
        assert!(net2.qat_enabled());
        assert!(net2.weight_tying());
        assert_eq!(rng.state(), rng2.state(), "RNG state must round-trip");
    }

    #[test]
    fn load_checkpoint_rejects_corrupt_data() {
        assert!(TransformerNet::load_checkpoint(b"").is_err());
        assert!(TransformerNet::load_checkpoint(b"XXXXjunkdata").is_err());
        let mut r = Rng::new(1);
        let net = TransformerNet::new(6, 4, 8, 2, 16, 1, &mut r).unwrap();
        let bytes = net.save_checkpoint(&r);
        // Truncated payload is rejected, not silently zero-filled.
        assert!(TransformerNet::load_checkpoint(&bytes[..bytes.len() / 2]).is_err());
    }

    fn trained_net(tie: bool, qat: bool) -> (TransformerNet, Rng) {
        let mut r = Rng::new(3);
        let mut net = TransformerNet::new(6, 4, 8, 2, 16, 2, &mut r).unwrap();
        net.set_qat(qat);
        net.set_weight_tying(tie);
        let tokens: Vec<usize> = (0..100).map(|i| (i * 5) % 6).collect();
        let adam = Adam::new(0.01);
        let mut rng = Rng::new(11);
        train_transformer_epoch(&mut net, &tokens, 8, &adam, &mut rng).unwrap();
        (net, rng)
    }

    #[test]
    fn checkpoint_is_gguf_and_roundtrips() {
        let mut rng = Rng::new(5);
        let net = TransformerNet::new(30, 8, 16, 2, 32, 1, &mut rng).unwrap();
        let bytes = net.save_checkpoint(&rng);
        assert_eq!(&bytes[..4], b"GGUF");
        let g = crate::gguf::Gguf::parse(bytes.clone()).unwrap();
        assert_eq!(g.architecture(), Some("ferrum-checkpoint"));
        let (net2, rng2) = TransformerNet::load_checkpoint(&bytes).unwrap();
        assert_eq!(net2.save_checkpoint(&rng2), bytes);
    }

    #[test]
    fn checkpoint_roundtrips_every_field_bit_exactly() {
        for (tie, qat) in [(false, false), (true, false), (false, true), (true, true)] {
            let (net, rng) = trained_net(tie, qat);
            assert!(net.step_count() > 0);
            let (net2, rng2) = TransformerNet::load_checkpoint(&net.save_checkpoint(&rng)).unwrap();
            assert_eq!(net2.qat_enabled(), qat);
            assert_eq!(net2.weight_tying(), tie);
            assert_eq!(net2.step_count(), net.step_count());
            assert_eq!(rng2.state(), rng.state());
            assert_eq!(net2.hidden_dim(), net.hidden_dim());
            let (a, b) = (net.params_all(), net2.params_all());
            assert_eq!(a.len(), b.len());
            let mut moments = 0.0f32;
            for (p, q) in a.iter().zip(&b) {
                for (x, y) in [(&p.data, &q.data), (&p.m, &q.m), (&p.v, &q.v)] {
                    assert_eq!(x.shape, y.shape);
                    assert!(x
                        .data
                        .iter()
                        .zip(&y.data)
                        .all(|(u, w)| u.to_bits() == w.to_bits()));
                }
                moments += p.m.data.iter().map(|v| v.abs()).sum::<f32>();
            }
            assert!(moments > 0.0, "test must exercise non-zero Adam moments");
        }
    }

    #[test]
    fn old_checkpoint_magic_is_a_clear_error() {
        let err = TransformerNet::load_checkpoint(b"FCKP\x01\x00\x00\x00")
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("pre-GGUF checkpoint (FCKP)"), "{err}");
        assert!(err.contains("retrain or start a new checkpoint"), "{err}");
    }

    #[test]
    fn checkpoint_rejects_foreign_truncated_and_wrong_architecture() {
        for junk in [&b""[..], b"XXXXjunkdata", b"GGUF"] {
            assert!(TransformerNet::load_checkpoint(junk).is_err());
        }
        let (net, rng) = trained_net(false, false);
        let bytes = net.save_checkpoint(&rng);
        for cut in [bytes.len() / 2, bytes.len() - 1, 100] {
            assert!(
                TransformerNet::load_checkpoint(&bytes[..cut]).is_err(),
                "cut {cut}"
            );
        }
        let other = crate::ckpt::rewrite(
            &bytes,
            |m| {
                m.insert(
                    "general.architecture".into(),
                    crate::gguf::MetaValue::String("gpt2".into()),
                );
            },
            None,
        );
        let err = TransformerNet::load_checkpoint(&other)
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("not a ferrum-checkpoint"), "{err}");
    }

    #[test]
    fn checkpoint_rejects_inconsistent_headers_and_tensors() {
        use crate::gguf::MetaValue;
        let (net, rng) = trained_net(true, true);
        let bytes = net.save_checkpoint(&rng);
        let key = |k: &str| format!("ferrum.ckpt.{k}");
        // A missing or mistyped scalar names its key.
        for k in [
            "vocab", "context", "embed", "heads", "hidden", "blocks", "step", "rng",
        ] {
            let kk = key(k);
            let b = crate::ckpt::rewrite(&bytes, |m| drop(m.remove(&kk)), None);
            let e = TransformerNet::load_checkpoint(&b)
                .err()
                .unwrap()
                .to_string();
            assert!(e.contains(&kk), "{k}: {e}");
        }
        for k in ["qat", "tie"] {
            let kk = key(k);
            let b = crate::ckpt::rewrite(&bytes, |m| drop(m.remove(&kk)), None);
            assert!(TransformerNet::load_checkpoint(&b).is_err(), "{k}");
            let b = crate::ckpt::rewrite(
                &bytes,
                |m| {
                    m.insert(kk.clone(), MetaValue::U32(1));
                },
                None,
            );
            assert!(TransformerNet::load_checkpoint(&b).is_err(), "{k} type");
        }
        // Header says a different shape than the tensors hold.
        let b = crate::ckpt::rewrite(
            &bytes,
            |m| {
                m.insert(key("embed"), MetaValue::U32(16));
            },
            None,
        );
        let e = TransformerNet::load_checkpoint(&b)
            .err()
            .unwrap()
            .to_string();
        assert!(e.contains("does not match its tensors"), "{e}");
        // Invalid architecture numbers are rejected by the constructor.
        for (k, v) in [("vocab", 0), ("heads", 3)] {
            let b = crate::ckpt::rewrite(
                &bytes,
                |m| {
                    m.insert(key(k), MetaValue::U32(v));
                },
                None,
            );
            assert!(TransformerNet::load_checkpoint(&b).is_err(), "{k}");
        }
        // A dropped moment tensor is an error (the directory no longer matches
        // the header), never zero-filled.
        let b = crate::ckpt::rewrite(&bytes, |_| {}, Some("p3.v"));
        let e = TransformerNet::load_checkpoint(&b)
            .err()
            .unwrap()
            .to_string();
        assert!(e.contains("does not match its tensors"), "{e}");
    }

    #[test]
    fn param_lens_match_real_nets_and_detect_overflow() {
        for (v, c, e, h, hid, b) in [
            (6, 4, 8, 2, 16, 2),
            (30, 8, 16, 2, 32, 1),
            (5, 3, 4, 1, 7, 3),
        ] {
            let net = TransformerNet::new(v, c, e, h, hid, b, &mut Rng::new(1)).unwrap();
            let real: Vec<usize> = net.params_all().iter().map(|p| p.data.data.len()).collect();
            assert_eq!(TransformerNet::param_lens(v, c, e, hid, b), Some(real));
        }
        let big = usize::MAX / 2;
        // u64-based literal: `1 << 40` / `1 << 62` do not fit a 32-bit usize.
        let huge = usize::try_from(1u64 << 40).unwrap_or(usize::MAX);
        let huger = usize::try_from(1u64 << 62).unwrap_or(usize::MAX);
        for args in [
            (big, 1, 4, 1, 1),
            (1, big, 4, 1, 1),
            (1, 1, big, 1, 1),
            (1, 1, huge, 1, 1),
            (1, 1, 4, huger, 1),
        ] {
            let (v, c, e, hid, b) = args;
            assert_eq!(
                TransformerNet::param_lens(v, c, e, hid, b),
                None,
                "{args:?}"
            );
        }
    }

    /// Crafted headers must fail fast with `Err`; if any of them drove an
    /// allocation (4e9 x 4e9 elements, 4e9 blocks) the test process would abort.
    #[test]
    fn crafted_headers_are_rejected_before_any_allocation() {
        use crate::gguf::MetaValue;
        let mut r = Rng::new(1);
        let net = TransformerNet::new(6, 4, 8, 2, 16, 2, &mut r).unwrap();
        let bytes = net.save_checkpoint(&r);
        let with = |k: &str, v: u32| {
            let k = format!("ferrum.ckpt.{k}");
            crate::ckpt::rewrite(
                &bytes,
                move |m| {
                    m.insert(k, MetaValue::U32(v));
                },
                None,
            )
        };
        let cases: [(&str, u32); 14] = [
            ("vocab", 4_000_000_000),
            ("context", 4_000_000_000),
            ("embed", 4_000_000_000),
            ("hidden", 4_000_000_000),
            ("blocks", 4_000_000_000),
            ("heads", 4_000_000_000),
            ("embed", 0),
            ("context", 0),
            ("hidden", 0),
            ("blocks", 0),
            ("heads", 0),
            ("heads", 3),  // does not divide embed 8
            ("blocks", 1), // fewer blocks than the tensor directory holds
            ("blocks", 3), // more blocks than it holds
        ];
        for (k, v) in cases {
            let t = std::time::Instant::now();
            let e = TransformerNet::load_checkpoint(&with(k, v)).err();
            assert!(e.is_some(), "{k}={v} must be rejected");
            assert!(t.elapsed().as_secs() < 5, "{k}={v} took too long");
        }
        // Tensor extents beyond EOF (header untouched).
        for cut in [4, 40] {
            assert!(TransformerNet::load_checkpoint(&bytes[..bytes.len() - cut]).is_err());
        }
        // A self-consistent claim of a huge net with no tensors at all.
        let mut huge = crate::ckpt::begin("ferrum-checkpoint");
        for k in ["vocab", "context", "embed", "heads", "hidden", "blocks"] {
            crate::ckpt::put_u32(&mut huge, k, 1 << 30);
        }
        crate::ckpt::put_bool(&mut huge, "qat", false);
        crate::ckpt::put_bool(&mut huge, "tie", false);
        crate::ckpt::put_u64(&mut huge, "step", 0);
        crate::ckpt::put_u64(&mut huge, "rng", 1);
        assert!(TransformerNet::load_checkpoint(&huge.into_bytes()).is_err());
    }

    /// Rebuild a checkpoint's GGUF with `edit` applied to each tensor
    /// `(name, dims, type)`; data is zero-filled to the *edited* extent, so
    /// the result is always a structurally valid, non-overlapping GGUF.
    fn mangled(bytes: &[u8], edit: impl Fn(&mut (String, Vec<u64>, u32))) -> Vec<u8> {
        let g = crate::gguf::Gguf::parse(bytes.to_vec()).unwrap();
        let mut b = crate::gguf_write::GgufBuilder::new();
        for (k, v) in &g.metadata {
            b.meta(k, v.clone());
        }
        for t in &g.tensors {
            let mut e = (t.name.clone(), t.dims.clone(), t.ggml_type);
            edit(&mut e);
            let n: u64 = e.1.iter().product();
            let width = if e.2 == crate::gguf::GGML_F32 { 4 } else { 2 };
            b.tensor(&e.0, &e.1, e.2, vec![0u8; n as usize * width]);
        }
        b.into_bytes()
    }

    /// `check_header` demands exactly `p{i}.{data,m,v}` F32 tensors of the
    /// header-implied length, even when the extents are all valid.
    #[test]
    fn check_header_requires_exact_tensor_names_dims_and_types() {
        let mut r = Rng::new(1);
        let net = TransformerNet::new(6, 4, 8, 2, 16, 2, &mut r).unwrap();
        let bytes = net.save_checkpoint(&r);
        TransformerNet::load_checkpoint(&bytes).unwrap(); // control
        type Edit = Box<dyn Fn(&mut (String, Vec<u64>, u32))>;
        let cases: Vec<(&str, Edit)> = vec![
            (
                "renamed tensor",
                Box::new(|t| {
                    if t.0 == "p3.m" {
                        t.0 = "q3.m".into();
                    }
                }),
            ),
            (
                "wrong dims",
                Box::new(|t| {
                    if t.0 == "p5.data" {
                        t.1 = vec![t.1[0] + 1];
                    }
                }),
            ),
            (
                "wrong rank",
                Box::new(|t| {
                    if t.0 == "p5.v" {
                        t.1 = vec![1, t.1[0]];
                    }
                }),
            ),
            (
                "wrong type (extent changes)",
                Box::new(|t| {
                    if t.0 == "p5.m" {
                        t.2 = crate::gguf::GGML_F16;
                    }
                }),
            ),
        ];
        for (why, edit) in cases {
            let e = TransformerNet::load_checkpoint(&mangled(&bytes, &edit))
                .err()
                .unwrap_or_else(|| panic!("{why} must be rejected"));
            assert!(
                e.to_string().contains("does not match its tensors"),
                "{why}: {e}"
            );
        }
    }

    /// The round-1 bypass: every `*.data` tensor aliases the same bytes, with
    /// a header chosen so the summed element count equals the architecture's
    /// parameter count. Rejected centrally (overlap) before any net exists,
    /// and by `check_header` on its own once the aliasing is removed.
    #[test]
    fn aliased_tensor_checkpoint_is_rejected_quickly() {
        const BLOCKS: usize = 100;
        const L: u64 = 1001; // elements per aliased `*.data` tensor (odd)
        let n_params = 5 + 16 * BLOCKS;
        // embed = heads = hidden = ctx = 1: total params = 2*vocab + 16*blocks + 3
        // must equal n_params * L, so vocab is solved for.
        let vocab = (n_params as u64 * L - (16 * BLOCKS as u64 + 3)) / 2;
        let mut b = crate::ckpt::begin("ferrum-checkpoint");
        for (k, v) in [
            ("vocab", vocab as usize),
            ("context", 1),
            ("embed", 1),
            ("heads", 1),
            ("hidden", 1),
            ("blocks", BLOCKS),
        ] {
            crate::ckpt::put_u32(&mut b, k, v);
        }
        crate::ckpt::put_bool(&mut b, "qat", false);
        crate::ckpt::put_bool(&mut b, "tie", false);
        crate::ckpt::put_u64(&mut b, "step", 0);
        crate::ckpt::put_u64(&mut b, "rng", 1);
        let data = vec![0.0f32; L as usize];
        for i in 0..n_params {
            crate::ckpt::put_param(&mut b, i, &data, &[0.0], &[0.0]);
        }
        let clean = b.into_bytes();
        // The file is ~(3 * n_params) directory entries plus ~L*4 data bytes
        // after aliasing; the net it claims would be ~2*vocab*16 B = ~51 MB.
        let mut bytes = clean.clone();
        for i in 0..3 * n_params {
            crate::gguf::patch_tensor_offset(&mut bytes, i, 0);
        }
        let t = std::time::Instant::now();
        let e = TransformerNet::load_checkpoint(&bytes).err().unwrap();
        assert!(e.to_string().contains("overlap"), "{e}");
        assert!(t.elapsed().as_secs() < 5, "took {:?}", t.elapsed());
        // Without aliasing the tensors are legitimate extents but wrong dims
        // for this header (data tensors are L long, the header wants 1 or
        // vocab): check_header refuses them before `new` allocates.
        let e = TransformerNet::load_checkpoint(&clean).err().unwrap();
        assert!(e.to_string().contains("does not match its tensors"), "{e}");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Weight tying (T9)
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn sync_tied_head_mirrors_embedding_transpose() {
        let mut rng = Rng::new(5);
        let mut net = TransformerNet::new(6, 4, 8, 2, 16, 1, &mut rng).unwrap();
        net.set_weight_tying(true); // enabling syncs immediately
        assert!(net.weight_tying());
        let (vocab, embed) = (6usize, 8usize);
        for v in 0..vocab {
            for e in 0..embed {
                assert_eq!(
                    net.head_w.data.data[e * vocab + v],
                    net.tok_emb.data.data[v * embed + e],
                    "head is not tok_embᵀ at ({v},{e})"
                );
            }
        }
    }

    /// The folded gradient must be the true gradient of the loss w.r.t. the
    /// shared embedding — accounting for the embedding affecting BOTH the input
    /// lookup and the (tied) output head. Verified by central differences that
    /// re-sync the head after each perturbation.
    #[test]
    fn weight_tying_gradient_is_correct() {
        let mut rng = Rng::new(3);
        let mut net = TransformerNet::new(5, 3, 4, 2, 6, 1, &mut rng).unwrap();
        net.set_weight_tying(true);
        let tokens = [1usize, 2, 3];
        let targets = [2usize, 3, 4];

        net.sync_tied_head();
        let (logits, cache) = net.forward(&tokens).unwrap();
        let (_, dl) = softmax_cross_entropy(&logits, &targets).unwrap();
        net.zero_grad();
        net.backward(&dl, &cache).unwrap();
        net.fold_tied_head_grad();

        let loss_at = |net: &mut TransformerNet, idx: usize, delta: f32| -> f32 {
            net.tok_emb.data.data[idx] += delta;
            net.sync_tied_head(); // the head tracks the embedding
            let (logits, _) = net.forward(&tokens).unwrap();
            let l = softmax_cross_entropy(&logits, &targets).unwrap().0;
            net.tok_emb.data.data[idx] -= delta; // restore
            net.sync_tied_head();
            l
        };

        let eps = 1e-2f32;
        for &i in &[4usize, 9, 13] {
            let analytic = net.tok_emb.grad.data[i];
            let lp = loss_at(&mut net, i, eps);
            let lm = loss_at(&mut net, i, -eps);
            let numeric = (lp - lm) / (2.0 * eps);
            assert!(
                (numeric - analytic).abs() < 6e-3 + 0.03 * analytic.abs(),
                "tied grad[{i}]: analytic={analytic:.6} numeric={numeric:.6}"
            );
        }
    }

    #[test]
    fn weight_tying_trains_and_exports_transposed_head() {
        let mut rng = Rng::new(7);
        let mut net = TransformerNet::new(6, 4, 8, 2, 16, 1, &mut rng).unwrap();
        net.set_weight_tying(true);
        let tokens: Vec<usize> = (0..200).map(|i| i % 6).collect();
        let adam = Adam::new(0.01);
        let first = train_transformer_epoch(&mut net, &tokens, 8, &adam, &mut rng).unwrap();
        let mut last = first;
        for _ in 0..30 {
            last = train_transformer_epoch(&mut net, &tokens, 8, &adam, &mut rng).unwrap();
        }
        assert!(
            last < first * 0.6,
            "tied training did not reduce loss: {first:.4} → {last:.4}"
        );

        // The exported head must equal the final tok_embᵀ.
        let model = net.to_inference().unwrap();
        let head = model
            .layers()
            .iter()
            .rev()
            .find_map(|l| l.as_any().downcast_ref::<Linear>())
            .expect("model has an LM head");
        let (vocab, embed) = (6usize, 8usize);
        for v in 0..vocab {
            for e in 0..embed {
                assert!(
                    (head.weight.data[e * vocab + v] - net.tok_emb.data.data[v * embed + e]).abs()
                        < 1e-6,
                    "exported head ≠ tok_embᵀ at ({v},{e})"
                );
            }
        }
    }

    /// Tied training is still bit-identical across the serial and single-shard
    /// threaded paths (the sync/fold hooks run in both).
    #[test]
    fn tied_threaded_one_shard_matches_serial() {
        let mut rng0 = Rng::new(7);
        let mut base = TransformerNet::new(6, 4, 8, 2, 16, 1, &mut rng0).unwrap();
        base.set_weight_tying(true);
        let tokens: Vec<usize> = (0..200).map(|i| (i * 7) % 6).collect();
        let adam = Adam::new(0.01);
        let mut serial = base.clone();
        let mut rs = Rng::new(9);
        let mut threaded = base.clone();
        let mut rt = Rng::new(9);
        for _ in 0..3 {
            train_transformer_epoch(&mut serial, &tokens, 8, &adam, &mut rs).unwrap();
            train_transformer_epoch_threaded(&mut threaded, &tokens, 8, &adam, &mut rt, 1).unwrap();
        }
        assert_eq!(serial.snapshot_weights(), threaded.snapshot_weights());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Shuffle without replacement / true epochs (T4)
    // ─────────────────────────────────────────────────────────────────────────

    /// A true epoch covers every window exactly once: with `batch_size = 1` it
    /// takes exactly `num_windows` optimizer steps, and with a batch larger than
    /// the corpus exactly one step — never sampling with replacement.
    #[test]
    fn epoch_makes_one_pass_over_all_windows() {
        let mut rng0 = Rng::new(7);
        let mut net = TransformerNet::new(5, 4, 8, 2, 16, 1, &mut rng0).unwrap();
        let tokens: Vec<usize> = (0..50).map(|i| i % 5).collect();
        let num_windows = (tokens.len() - 4) as u64;
        let adam = Adam::new(0.01);
        let mut rng = Rng::new(1);

        let before = net.step_count();
        train_transformer_epoch(&mut net, &tokens, 1, &adam, &mut rng).unwrap();
        assert_eq!(
            net.step_count() - before,
            num_windows,
            "batch_size=1 must take exactly num_windows steps"
        );

        let before = net.step_count();
        train_transformer_epoch(&mut net, &tokens, 10_000, &adam, &mut rng).unwrap();
        assert_eq!(
            net.step_count() - before,
            1,
            "an over-sized batch must take exactly one full-corpus step"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Learning-rate schedule (T2)
    // ─────────────────────────────────────────────────────────────────────────

    /// A warmup+decay schedule is actually applied during training: it drives a
    /// different trajectory than the same base LR held fixed, advances the step
    /// counter, and still reduces loss.
    #[test]
    fn lr_schedule_is_applied_and_reduces_loss() {
        use crate::optim::LrSchedule;
        let mut rng0 = Rng::new(7);
        let base = TransformerNet::new(4, 4, 8, 2, 16, 1, &mut rng0).unwrap();
        let tokens: Vec<usize> = (0..200).map(|i| i % 4).collect();
        let adam = Adam::new(0.02);

        // Fixed-LR reference.
        let mut fixed = base.clone();
        let mut rf = Rng::new(5);
        for _ in 0..10 {
            train_transformer_epoch(&mut fixed, &tokens, 8, &adam, &mut rf).unwrap();
        }

        // Scheduled run: warmup then cosine decay over the same total steps.
        let steps_per_epoch = (tokens.len() - 4).div_ceil(8) as u64;
        let total = steps_per_epoch * 10;
        let mut sched_net = base.clone();
        sched_net.set_lr_schedule(Some(LrSchedule::warmup_cosine(
            0.02,
            steps_per_epoch * 2,
            total,
        )));
        assert!(sched_net.lr_schedule().is_some());
        let mut rs = Rng::new(5);
        let first = train_transformer_epoch(&mut sched_net, &tokens, 8, &adam, &mut rs).unwrap();
        let mut last = first;
        for _ in 0..9 {
            last = train_transformer_epoch(&mut sched_net, &tokens, 8, &adam, &mut rs).unwrap();
        }

        assert_eq!(
            sched_net.step_count(),
            total,
            "step counter must drive the schedule"
        );
        assert!(
            last < first,
            "scheduled run should reduce loss: {first:.4} → {last:.4}"
        );
        // Warmup means the scheduled run takes a genuinely different path.
        assert_ne!(
            fixed.snapshot_weights(),
            sched_net.snapshot_weights(),
            "schedule had no effect on training"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Shared-weight data-parallel gradients (T3)
    // ─────────────────────────────────────────────────────────────────────────

    /// `backward_grads(&self)` returns exactly the gradients the in-place
    /// `backward` stores, and leaves the network unmodified. This purity is what
    /// lets the data-parallel workers borrow the master weights read-only
    /// instead of deep-cloning the whole network per shard.
    #[test]
    fn backward_grads_matches_inplace_backward_and_leaves_weights_intact() {
        let mut rng = Rng::new(7);
        let mut net = TransformerNet::new(6, 4, 8, 2, 16, 2, &mut rng).unwrap();
        let tokens = [1usize, 2, 3, 4, 0, 5, 2, 1]; // two windows (T=4)
        let targets = [2usize, 3, 4, 5, 1, 0, 3, 2];
        let (logits, cache) = net.forward(&tokens).unwrap();
        let (_, dl) = softmax_cross_entropy(&logits, &targets).unwrap();

        // Pure: no weight or optimizer-moment mutation.
        let weights_before = net.snapshot_weights();
        let flat = net.backward_grads(&dl, &cache).unwrap();
        assert_eq!(
            net.snapshot_weights(),
            weights_before,
            "backward_grads mutated weights"
        );
        assert_eq!(flat.len(), net.num_params());

        // The in-place wrapper stores exactly those gradients (canonical order).
        net.zero_grad();
        net.backward(&dl, &cache).unwrap();
        let mut stored = Vec::new();
        for g in net.grad_tensors_mut() {
            stored.extend_from_slice(&g.data);
        }
        assert_eq!(
            stored, flat,
            "in-place backward differs from backward_grads"
        );
    }

    /// The shared-weight (no-clone) data-parallel path stays numerically in
    /// step with the serial path across several epochs — the memory fix changes
    /// only allocation, not the math.
    #[test]
    fn threaded_shared_weights_track_serial_over_epochs() {
        let mut rng0 = Rng::new(7);
        let base = TransformerNet::new(6, 4, 8, 2, 16, 2, &mut rng0).unwrap();
        let tokens: Vec<usize> = (0..200).map(|i| (i * 7) % 6).collect();
        let adam = Adam::new(0.02);
        let batch = tokens.len() - 4; // one optimizer step per epoch (all windows)

        let mut serial = base.clone();
        let mut rs = Rng::new(3);
        let mut par = base.clone();
        let mut rp = Rng::new(3);
        for _ in 0..5 {
            train_transformer_epoch(&mut serial, &tokens, batch, &adam, &mut rs).unwrap();
            train_transformer_epoch_threaded(&mut par, &tokens, batch, &adam, &mut rp, 4).unwrap();
        }
        let sw = serial.snapshot_weights();
        let pw = par.snapshot_weights();
        for (a, b) in sw.iter().zip(&pw) {
            for (x, y) in a.iter().zip(b) {
                assert!(
                    (x - y).abs() < 1e-3,
                    "shared-weight path drifted from serial: {x} vs {y}"
                );
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Gradient clipping (T1)
    // ─────────────────────────────────────────────────────────────────────────

    /// `clip_grad_norm` caps the post-backward global gradient norm at the
    /// requested budget and returns the (larger) pre-clip norm.
    #[test]
    fn clip_grad_norm_caps_global_norm() {
        let mut rng = Rng::new(7);
        let mut net = TransformerNet::new(6, 4, 8, 2, 16, 1, &mut rng).unwrap();
        let tokens = [1usize, 2, 3, 4];
        let targets = [2usize, 3, 4, 5];
        let (logits, cache) = net.forward(&tokens).unwrap();
        let (_, dl) = softmax_cross_entropy(&logits, &targets).unwrap();
        net.zero_grad();
        net.backward(&dl, &cache).unwrap();

        let max_norm = 0.1f32;
        let pre = net.clip_grad_norm(max_norm);
        // Recompute the global norm from the (now clipped) gradients.
        let mut sumsq = 0.0f64;
        for g in net.grad_tensors_mut() {
            for &x in &g.data {
                sumsq += (x as f64) * (x as f64);
            }
        }
        let post = sumsq.sqrt() as f32;
        assert!(
            pre > max_norm,
            "test needs an exploding-enough grad: pre={pre}"
        );
        assert!(
            post <= max_norm + 1e-4,
            "post-clip norm {post} exceeds budget {max_norm}"
        );
    }

    /// Gradient clipping is actually wired into the epoch: with the same seed,
    /// a clipped run takes a different trajectory than an unclipped one (because
    /// the early gradients exceed the budget), and stays finite while still
    /// reducing loss.
    #[test]
    fn grad_clip_changes_trajectory_and_stays_finite() {
        let mut rng0 = Rng::new(7);
        let base = TransformerNet::new(6, 6, 12, 2, 24, 2, &mut rng0).unwrap();
        let tokens: Vec<usize> = (0..300).map(|i| (i * 7) % 6).collect();
        let adam = Adam::new(0.05);

        let finite = |net: &mut TransformerNet| {
            net.snapshot_weights()
                .iter()
                .all(|t| t.iter().all(|v| v.is_finite()))
        };

        let mut unclipped = base.clone();
        let mut r1 = Rng::new(1);
        for _ in 0..5 {
            train_transformer_epoch(&mut unclipped, &tokens, 16, &adam, &mut r1).unwrap();
        }

        let mut clipped = base.clone();
        clipped.set_grad_clip(Some(0.01)); // tiny budget → clips most steps
        assert_eq!(clipped.grad_clip(), Some(0.01));
        let mut r2 = Rng::new(1);
        let first = train_transformer_epoch(&mut clipped, &tokens, 16, &adam, &mut r2).unwrap();
        let mut last = first;
        for _ in 0..4 {
            last = train_transformer_epoch(&mut clipped, &tokens, 16, &adam, &mut r2).unwrap();
        }

        assert!(finite(&mut clipped), "clipped run must stay finite");
        assert!(
            last < first,
            "clipped run should still reduce loss: {first:.4} → {last:.4}"
        );
        // The clip threshold changed the optimization path.
        assert_ne!(
            unclipped.snapshot_weights(),
            clipped.snapshot_weights(),
            "grad clipping had no effect on the weights"
        );
    }

    /// For a fixed `threads` value, training is reproducible regardless of how
    /// the OS schedules the worker threads (deterministic shard reduction).
    #[test]
    fn threaded_training_is_reproducible() {
        let mut rng0 = Rng::new(5);
        let base = TransformerNet::new(5, 4, 8, 2, 16, 1, &mut rng0).unwrap();
        let tokens: Vec<usize> = (0..300).map(|i| (i * 11) % 5).collect();
        let adam = Adam::new(0.01);
        let run = || {
            let mut n = base.clone();
            let mut r = Rng::new(321);
            for _ in 0..10 {
                train_transformer_epoch_threaded(&mut n, &tokens, 12, &adam, &mut r, 4).unwrap();
            }
            n.snapshot_weights()
        };
        assert_eq!(run(), run(), "threaded training is not reproducible");
    }
}
