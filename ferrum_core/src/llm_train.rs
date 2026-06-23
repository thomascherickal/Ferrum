//! Training (the backward pass) for the Llama/Qwen decoder in [`crate::llm`].
//!
//! `llm.rs` is forward-only; without gradients the imported architecture cannot
//! be trained or fine-tuned. This module adds hand-derived backprop for every
//! primitive that block uses — **RMSNorm, RoPE, grouped-query attention (with the
//! softmax), the SwiGLU FFN, the token embedding, and the LM head** — plus a
//! next-token cross-entropy loss and an SGD [`LlamaTrainer::train_step`]. Each
//! primitive's gradient is checked against finite differences in the tests, and
//! an end-to-end test shows the loss falling on a memorisable sequence.
//!
//! Training runs in **f32**: the model must hold full-precision weights
//! (`Gguf::load_llama_prec(None)`, or built f32 directly). Quantized (`QWeight`)
//! Linears have no f32 master to update, so [`LlamaTrainer::new`] rejects them.
//!
//! Scope/reality check: this makes the architecture *trainable* and is exercised
//! on small models. It does **not** make training a 1B model on a CPU feasible —
//! that stays bounded by compute and RAM (see `ferrum_review.md §4.3`). It adds
//! the missing capability (gradients), not a claim about scale.

use crate::error::{InferError, Result};
use crate::layer::Linear;
use crate::llm::{Attention, FeedForward, LlamaBlock, LlamaConfig, LlamaModel, RopeType};

// ─────────────────────────────────────────────────────────────────────────────
// Per-parameter gradient buffers (mirror the model's trainable tensors)
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone)]
struct LinGrad {
    dw: Vec<f32>, // [in·out] row-major, like Linear::weight
    db: Vec<f32>, // [out]
}
impl LinGrad {
    fn zeros(lin: &Linear) -> Self {
        Self { dw: vec![0.0; lin.in_features() * lin.out_features()], db: vec![0.0; lin.out_features()] }
    }
    fn clear(&mut self) {
        self.dw.iter_mut().for_each(|x| *x = 0.0);
        self.db.iter_mut().for_each(|x| *x = 0.0);
    }
}

struct BlockGrad {
    attn_norm: Vec<f32>,
    wq: LinGrad,
    wk: LinGrad,
    wv: LinGrad,
    wo: LinGrad,
    ffn_norm: Vec<f32>,
    gate: LinGrad,
    up: LinGrad,
    down: LinGrad,
}

struct Grads {
    tok_emb: Vec<f32>,
    blocks: Vec<BlockGrad>,
    final_norm: Vec<f32>,
    lm_head: LinGrad,
}

impl Grads {
    fn zeros(model: &LlamaModel) -> Self {
        let blocks = model
            .blocks
            .iter()
            .map(|b| BlockGrad {
                attn_norm: vec![0.0; b.attn_norm.weight.len()],
                wq: LinGrad::zeros(&b.attn.wq),
                wk: LinGrad::zeros(&b.attn.wk),
                wv: LinGrad::zeros(&b.attn.wv),
                wo: LinGrad::zeros(&b.attn.wo),
                ffn_norm: vec![0.0; b.ffn_norm.weight.len()],
                gate: LinGrad::zeros(&b.ffn.gate),
                up: LinGrad::zeros(&b.ffn.up),
                down: LinGrad::zeros(&b.ffn.down),
            })
            .collect();
        Self {
            tok_emb: vec![0.0; model.tok_emb.len()],
            blocks,
            final_norm: vec![0.0; model.final_norm.weight.len()],
            lm_head: LinGrad::zeros(&model.lm_head),
        }
    }
    fn clear(&mut self) {
        self.tok_emb.iter_mut().for_each(|x| *x = 0.0);
        self.final_norm.iter_mut().for_each(|x| *x = 0.0);
        self.lm_head.clear();
        for b in &mut self.blocks {
            b.attn_norm.iter_mut().for_each(|x| *x = 0.0);
            b.ffn_norm.iter_mut().for_each(|x| *x = 0.0);
            for g in [&mut b.wq, &mut b.wk, &mut b.wv, &mut b.wo, &mut b.gate, &mut b.up, &mut b.down] {
                g.clear();
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Primitive forward/backward helpers (f32, row-major)
// ─────────────────────────────────────────────────────────────────────────────

/// `y[r,o] = Σ_i x[r,i]·W[i,o] + b[o]`.
fn lin_forward(lin: &Linear, x: &[f32], rows: usize) -> Vec<f32> {
    let (k, n) = (lin.in_features(), lin.out_features());
    let (w, b) = (&lin.weight.data, &lin.bias.data);
    let mut y = vec![0.0f32; rows * n];
    for r in 0..rows {
        let xr = &x[r * k..r * k + k];
        let yr = &mut y[r * n..r * n + n];
        yr.copy_from_slice(b);
        for (i, &xi) in xr.iter().enumerate() {
            let wr = &w[i * n..i * n + n];
            for (o, &wio) in wr.iter().enumerate() {
                yr[o] += xi * wio;
            }
        }
    }
    y
}

/// Backward of [`lin_forward`]: accumulates `dW`/`db` into `g`, returns `dX`.
fn lin_backward(lin: &Linear, x: &[f32], dy: &[f32], rows: usize, g: &mut LinGrad) -> Vec<f32> {
    let (k, n) = (lin.in_features(), lin.out_features());
    let w = &lin.weight.data;
    let mut dx = vec![0.0f32; rows * k];
    for r in 0..rows {
        let xr = &x[r * k..r * k + k];
        let dyr = &dy[r * n..r * n + n];
        let dxr = &mut dx[r * k..r * k + k];
        for (o, &dyo) in dyr.iter().enumerate() {
            g.db[o] += dyo;
        }
        for i in 0..k {
            let wr = &w[i * n..i * n + n];
            let dwr = &mut g.dw[i * n..i * n + n];
            let xi = xr[i];
            let mut acc = 0.0f32;
            for o in 0..n {
                acc += dyr[o] * wr[o];
                dwr[o] += xi * dyr[o];
            }
            dxr[i] = acc;
        }
    }
    dx
}

/// RMSNorm of one row: `(y, inv)` with `inv = 1/sqrt(mean(x²)+eps)`.
fn rmsnorm_forward_row(x: &[f32], w: &[f32], eps: f32) -> (Vec<f32>, f32) {
    let d = x.len();
    let ms = x.iter().map(|&v| v * v).sum::<f32>() / d as f32;
    let inv = 1.0 / (ms + eps).sqrt();
    ((0..d).map(|i| x[i] * inv * w[i]).collect(), inv)
}

/// Backward of RMSNorm for one row; adds the weight gradient into `dw`, returns
/// `dx`. With `g_i = dy_i·w_i`: `dx_j = inv·g_j − (inv³·x_j/d)·Σ_i g_i x_i`.
fn rmsnorm_backward_row(dy: &[f32], x: &[f32], w: &[f32], inv: f32, dw: &mut [f32]) -> Vec<f32> {
    let d = x.len();
    let mut gx = 0.0f32;
    for i in 0..d {
        gx += dy[i] * w[i] * x[i];
        dw[i] += dy[i] * x[i] * inv;
    }
    let inv3 = inv * inv * inv / d as f32;
    (0..d).map(|j| inv * dy[j] * w[j] - inv3 * x[j] * gx).collect()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}
/// d/dx silu(x) = σ(x)·(1 + x·(1−σ(x))).
fn silu_grad(x: f32) -> f32 {
    let s = sigmoid(x);
    s * (1.0 + x * (1.0 - s))
}

/// Apply RoPE to one `[n_heads·head_dim]` row; `transpose = true` rotates by −θ
/// (the backward direction, since RoPE is an orthogonal rotation).
#[allow(clippy::too_many_arguments)]
fn rope_apply(
    x: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    pos: usize,
    base: f32,
    rope_type: RopeType,
    transpose: bool,
) {
    let half = rope_dim / 2;
    for h in 0..n_heads {
        let off = h * head_dim;
        for i in 0..half {
            let freq = (base as f64).powf(-2.0 * i as f64 / rope_dim as f64) as f32;
            let theta = pos as f32 * freq;
            let (mut s, c) = theta.sin_cos();
            if transpose {
                s = -s;
            }
            let (ia, ib) = match rope_type {
                RopeType::Norm => (off + 2 * i, off + 2 * i + 1),
                RopeType::Neox => (off + i, off + i + half),
            };
            let (a, b) = (x[ia], x[ib]);
            x[ia] = a * c - b * s;
            x[ib] = a * s + b * c;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Caches
// ─────────────────────────────────────────────────────────────────────────────

struct AttnCache {
    xn: Vec<f32>,         // [seq, dim] projection input (= rmsnorm1 output)
    q: Vec<f32>,          // [seq, q_dim] RoPE'd queries
    k: Vec<f32>,          // [seq, kv_dim] RoPE'd keys
    v: Vec<f32>,          // [seq, kv_dim] values
    ctx: Vec<f32>,        // [seq, q_dim] attention context (wo input)
    probs: Vec<Vec<f32>>, // [seq*n_heads] softmax weights, entry (t,h) has len t+1
}

struct FfnCache {
    xn: Vec<f32>, // [seq, dim] FFN input (= rmsnorm2 output)
    g: Vec<f32>,  // [seq, ffn] gate pre-activation
    u: Vec<f32>,  // [seq, ffn] up
    h: Vec<f32>,  // [seq, ffn] silu(g)·u (down input)
}

struct BlockCache {
    x: Vec<f32>,      // [seq, dim] block input
    n1_inv: Vec<f32>, // [seq]
    attn: AttnCache,
    h: Vec<f32>,      // [seq, dim] x + attn output
    n2_inv: Vec<f32>, // [seq]
    ffn: FfnCache,
}

struct FwdCache {
    x0: Vec<f32>, // [seq, dim] embeddings
    blocks: Vec<BlockCache>,
    xfinal: Vec<f32>, // [seq, dim] last block output (final-norm input)
    nf_inv: Vec<f32>, // [seq]
    xn: Vec<f32>,     // [seq, dim] final-norm output (lm_head input)
    logits: Vec<f32>, // [seq, vocab]
}

// ─────────────────────────────────────────────────────────────────────────────
// Sublayer forward (with cache) / backward — free functions to keep borrows simple
// ─────────────────────────────────────────────────────────────────────────────

fn attn_forward(cfg: &LlamaConfig, attn: &Attention, xn: &[f32], seq: usize) -> AttnCache {
    let (nh, nkv, hd) = (cfg.n_heads, cfg.n_kv_heads, cfg.head_dim);
    let (q_dim, kv_dim, group) = (nh * hd, nkv * hd, nh / nkv);
    let scale = 1.0 / (hd as f32).sqrt();

    let mut q = lin_forward(&attn.wq, xn, seq);
    let mut k = lin_forward(&attn.wk, xn, seq);
    let v = lin_forward(&attn.wv, xn, seq);
    for t in 0..seq {
        rope_apply(&mut q[t * q_dim..t * q_dim + q_dim], nh, hd, cfg.rope_dim, t, cfg.rope_base, cfg.rope_type, false);
        rope_apply(&mut k[t * kv_dim..t * kv_dim + kv_dim], nkv, hd, cfg.rope_dim, t, cfg.rope_base, cfg.rope_type, false);
    }

    let mut ctx = vec![0.0f32; seq * q_dim];
    let mut probs: Vec<Vec<f32>> = vec![Vec::new(); seq * nh];
    for t in 0..seq {
        for h in 0..nh {
            let kvh = h / group;
            let qh = &q[t * q_dim + h * hd..t * q_dim + h * hd + hd];
            let mut sc = vec![0.0f32; t + 1];
            for (j, scj) in sc.iter_mut().enumerate() {
                let kj = &k[j * kv_dim + kvh * hd..j * kv_dim + kvh * hd + hd];
                let mut dot = 0.0f32;
                for d in 0..hd {
                    dot += qh[d] * kj[d];
                }
                *scj = dot * scale;
            }
            let m = sc.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in sc.iter_mut() {
                *s = (*s - m).exp();
                sum += *s;
            }
            for s in sc.iter_mut() {
                *s /= sum;
            }
            let oh = &mut ctx[t * q_dim + h * hd..t * q_dim + h * hd + hd];
            for (j, &p) in sc.iter().enumerate() {
                let vj = &v[j * kv_dim + kvh * hd..j * kv_dim + kvh * hd + hd];
                for d in 0..hd {
                    oh[d] += p * vj[d];
                }
            }
            probs[t * nh + h] = sc;
        }
    }
    AttnCache { xn: xn.to_vec(), q, k, v, ctx, probs }
}

/// Returns the attention sublayer output `o = wo(ctx)` for the cache.
fn attn_output(attn: &Attention, c: &AttnCache, seq: usize) -> Vec<f32> {
    lin_forward(&attn.wo, &c.ctx, seq)
}

fn attn_backward(cfg: &LlamaConfig, attn: &Attention, c: &AttnCache, d_o: &[f32], seq: usize, g: &mut BlockGrad) -> Vec<f32> {
    let (nh, nkv, hd, dim) = (cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.model_dim);
    let (q_dim, kv_dim, group) = (nh * hd, nkv * hd, nh / nkv);
    let scale = 1.0 / (hd as f32).sqrt();

    let d_ctx = lin_backward(&attn.wo, &c.ctx, d_o, seq, &mut g.wo);
    let mut dq = vec![0.0f32; seq * q_dim];
    let mut dk = vec![0.0f32; seq * kv_dim];
    let mut dv = vec![0.0f32; seq * kv_dim];

    for t in 0..seq {
        for h in 0..nh {
            let kvh = h / group;
            let p = &c.probs[t * nh + h];
            let dctx = &d_ctx[t * q_dim + h * hd..t * q_dim + h * hd + hd];
            // dp_j = dctx · v_j ; dv_j += p_j · dctx
            let mut dp = vec![0.0f32; t + 1];
            for (j, dpj) in dp.iter_mut().enumerate() {
                let vbase = j * kv_dim + kvh * hd;
                let vj = &c.v[vbase..vbase + hd];
                let mut acc = 0.0f32;
                for d in 0..hd {
                    acc += dctx[d] * vj[d];
                    dv[vbase + d] += p[j] * dctx[d];
                }
                *dpj = acc;
            }
            // softmax backward: dscore_j = p_j (dp_j − Σ_l p_l dp_l)
            let dot: f32 = (0..=t).map(|j| p[j] * dp[j]).sum();
            let qbase = t * q_dim + h * hd;
            for j in 0..=t {
                let ds = p[j] * (dp[j] - dot) * scale;
                let kbase = j * kv_dim + kvh * hd;
                for d in 0..hd {
                    dq[qbase + d] += ds * c.k[kbase + d];
                    dk[kbase + d] += ds * c.q[qbase + d];
                }
            }
        }
    }

    // RoPE backward (transpose rotation) on dq, dk.
    for t in 0..seq {
        rope_apply(&mut dq[t * q_dim..t * q_dim + q_dim], nh, hd, cfg.rope_dim, t, cfg.rope_base, cfg.rope_type, true);
        rope_apply(&mut dk[t * kv_dim..t * kv_dim + kv_dim], nkv, hd, cfg.rope_dim, t, cfg.rope_base, cfg.rope_type, true);
    }

    let dxq = lin_backward(&attn.wq, &c.xn, &dq, seq, &mut g.wq);
    let dxk = lin_backward(&attn.wk, &c.xn, &dk, seq, &mut g.wk);
    let dxv = lin_backward(&attn.wv, &c.xn, &dv, seq, &mut g.wv);
    (0..seq * dim).map(|i| dxq[i] + dxk[i] + dxv[i]).collect()
}

fn ffn_forward(cfg: &LlamaConfig, ffn: &FeedForward, xn: &[f32], seq: usize) -> FfnCache {
    let f = cfg.ffn_dim;
    let g = lin_forward(&ffn.gate, xn, seq);
    let u = lin_forward(&ffn.up, xn, seq);
    let h: Vec<f32> = (0..seq * f).map(|i| silu(g[i]) * u[i]).collect();
    FfnCache { xn: xn.to_vec(), g, u, h }
}

fn ffn_output(ffn: &FeedForward, c: &FfnCache, seq: usize) -> Vec<f32> {
    lin_forward(&ffn.down, &c.h, seq)
}

fn ffn_backward(cfg: &LlamaConfig, ffn: &FeedForward, c: &FfnCache, d_f: &[f32], seq: usize, g: &mut BlockGrad) -> Vec<f32> {
    let (dim, f) = (cfg.model_dim, cfg.ffn_dim);
    let d_h = lin_backward(&ffn.down, &c.h, d_f, seq, &mut g.down);
    let mut dg = vec![0.0f32; seq * f];
    let mut du = vec![0.0f32; seq * f];
    for i in 0..seq * f {
        du[i] = d_h[i] * silu(c.g[i]);
        dg[i] = d_h[i] * c.u[i] * silu_grad(c.g[i]);
    }
    let dxg = lin_backward(&ffn.gate, &c.xn, &dg, seq, &mut g.gate);
    let dxu = lin_backward(&ffn.up, &c.xn, &du, seq, &mut g.up);
    (0..seq * dim).map(|i| dxg[i] + dxu[i]).collect()
}

fn block_backward(cfg: &LlamaConfig, block: &LlamaBlock, bc: &BlockCache, d_out: &[f32], seq: usize, g: &mut BlockGrad) -> Vec<f32> {
    let dim = cfg.model_dim;
    // out = h + ffn(rmsnorm2(h)); d_f = d_out (residual to h handled below).
    let d_n2 = ffn_backward(cfg, &block.ffn, &bc.ffn, d_out, seq, g);
    let mut d_h = d_out.to_vec();
    for t in 0..seq {
        let s = t * dim;
        let dxr = rmsnorm_backward_row(&d_n2[s..s + dim], &bc.h[s..s + dim], &block.ffn_norm.weight, bc.n2_inv[t], &mut g.ffn_norm);
        for d in 0..dim {
            d_h[s + d] += dxr[d];
        }
    }
    // h = x + attn(rmsnorm1(x)); d_o = d_h (residual to x handled below).
    let d_n1 = attn_backward(cfg, &block.attn, &bc.attn, &d_h, seq, g);
    let mut d_x = d_h.clone();
    for t in 0..seq {
        let s = t * dim;
        let dxr = rmsnorm_backward_row(&d_n1[s..s + dim], &bc.x[s..s + dim], &block.attn_norm.weight, bc.n1_inv[t], &mut g.attn_norm);
        for d in 0..dim {
            d_x[s + d] += dxr[d];
        }
    }
    d_x
}

fn apply_lin(lin: &mut Linear, g: &LinGrad, lr: f32) {
    for (w, gg) in lin.weight.data.iter_mut().zip(&g.dw) {
        *w -= lr * gg;
    }
    for (b, gg) in lin.bias.data.iter_mut().zip(&g.db) {
        *b -= lr * gg;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Trainer
// ─────────────────────────────────────────────────────────────────────────────

/// Wraps an f32 [`LlamaModel`] and trains it by next-token cross-entropy.
pub struct LlamaTrainer {
    pub model: LlamaModel,
    grads: Grads,
}

impl LlamaTrainer {
    /// Build a trainer over `model`. Errors if any projection is still quantized
    /// (training needs f32 masters — load with `Gguf::load_llama_prec(None)`).
    pub fn new(model: LlamaModel) -> Result<Self> {
        for (i, b) in model.blocks.iter().enumerate() {
            for (name, lin) in [
                ("attn_q", &b.attn.wq), ("attn_k", &b.attn.wk), ("attn_v", &b.attn.wv),
                ("attn_output", &b.attn.wo), ("ffn_gate", &b.ffn.gate), ("ffn_up", &b.ffn.up),
                ("ffn_down", &b.ffn.down),
            ] {
                if lin.qweight().is_some() {
                    return Err(InferError::DimMismatch(format!(
                        "block {i} {name} is quantized; training needs f32 weights \
                         (load with load_llama_prec(None))"
                    )));
                }
            }
        }
        if model.lm_head.qweight().is_some() {
            return Err(InferError::DimMismatch("lm_head is quantized; training needs f32 weights".into()));
        }
        let grads = Grads::zeros(&model);
        Ok(Self { model, grads })
    }

    fn forward_train(&self, tokens: &[usize]) -> Result<FwdCache> {
        let cfg = &self.model.cfg;
        let (seq, dim) = (tokens.len(), cfg.model_dim);
        let mut x0 = vec![0.0f32; seq * dim];
        for (t, &tok) in tokens.iter().enumerate() {
            if tok >= cfg.vocab_size {
                return Err(InferError::DimMismatch(format!("token {tok} ≥ vocab {}", cfg.vocab_size)));
            }
            x0[t * dim..(t + 1) * dim].copy_from_slice(&self.model.tok_emb[tok * dim..(tok + 1) * dim]);
        }

        let mut x = x0.clone();
        let mut blocks = Vec::with_capacity(self.model.blocks.len());
        for block in &self.model.blocks {
            let mut n1 = vec![0.0f32; seq * dim];
            let mut n1_inv = vec![0.0f32; seq];
            for t in 0..seq {
                let s = t * dim;
                let (y, inv) = rmsnorm_forward_row(&x[s..s + dim], &block.attn_norm.weight, cfg.norm_eps);
                n1[s..s + dim].copy_from_slice(&y);
                n1_inv[t] = inv;
            }
            let attn = attn_forward(cfg, &block.attn, &n1, seq);
            let o = attn_output(&block.attn, &attn, seq);
            let h: Vec<f32> = (0..seq * dim).map(|i| x[i] + o[i]).collect();

            let mut n2 = vec![0.0f32; seq * dim];
            let mut n2_inv = vec![0.0f32; seq];
            for t in 0..seq {
                let s = t * dim;
                let (y, inv) = rmsnorm_forward_row(&h[s..s + dim], &block.ffn_norm.weight, cfg.norm_eps);
                n2[s..s + dim].copy_from_slice(&y);
                n2_inv[t] = inv;
            }
            let ffn = ffn_forward(cfg, &block.ffn, &n2, seq);
            let f = ffn_output(&block.ffn, &ffn, seq);
            let out: Vec<f32> = (0..seq * dim).map(|i| h[i] + f[i]).collect();

            blocks.push(BlockCache { x: x.clone(), n1_inv, attn, h, n2_inv, ffn });
            x = out;
        }

        let xfinal = x.clone();
        let mut xn = vec![0.0f32; seq * dim];
        let mut nf_inv = vec![0.0f32; seq];
        for t in 0..seq {
            let s = t * dim;
            let (y, inv) = rmsnorm_forward_row(&xfinal[s..s + dim], &self.model.final_norm.weight, cfg.norm_eps);
            xn[s..s + dim].copy_from_slice(&y);
            nf_inv[t] = inv;
        }
        let logits = lin_forward(&self.model.lm_head, &xn, seq);
        Ok(FwdCache { x0, blocks, xfinal, nf_inv, xn, logits })
    }

    /// Mean next-token cross-entropy over `tokens` (position `t` predicts
    /// `tokens[t+1]`). Pure forward — no gradients.
    pub fn loss(&self, tokens: &[usize]) -> Result<f32> {
        let cache = self.forward_train(tokens)?;
        Ok(cross_entropy(&cache.logits, tokens, self.model.cfg.vocab_size).0)
    }

    /// Forward + backward over `tokens`; fills `self.grads`, returns the loss.
    fn loss_and_backward(&mut self, tokens: &[usize]) -> Result<f32> {
        let cache = self.forward_train(tokens)?;
        let cfg = self.model.cfg.clone();
        let (seq, dim, vocab) = (tokens.len(), cfg.model_dim, cfg.vocab_size);
        let (loss, d_logits) = cross_entropy(&cache.logits, tokens, vocab);
        if !loss.is_finite() {
            return Err(InferError::DimMismatch("non-finite loss".into()));
        }

        self.grads.clear();
        let d_xn = lin_backward(&self.model.lm_head, &cache.xn, &d_logits, seq, &mut self.grads.lm_head);

        let mut d_x = vec![0.0f32; seq * dim];
        for t in 0..seq {
            let s = t * dim;
            let dxr = rmsnorm_backward_row(
                &d_xn[s..s + dim],
                &cache.xfinal[s..s + dim],
                &self.model.final_norm.weight,
                cache.nf_inv[t],
                &mut self.grads.final_norm,
            );
            d_x[s..s + dim].copy_from_slice(&dxr);
        }

        for bi in (0..self.model.blocks.len()).rev() {
            let block = &self.model.blocks[bi];
            let bc = &cache.blocks[bi];
            let bg = &mut self.grads.blocks[bi];
            d_x = block_backward(&cfg, block, bc, &d_x, seq, bg);
        }

        // Embedding: d_x is the gradient w.r.t. x0 (the embeddings).
        for (t, &tok) in tokens.iter().enumerate() {
            let s = t * dim;
            for d in 0..dim {
                self.grads.tok_emb[tok * dim + d] += d_x[s + d];
            }
        }
        let _ = &cache.x0;
        Ok(loss)
    }

    /// One SGD step on `tokens` at learning rate `lr`. Returns the pre-step loss.
    pub fn train_step(&mut self, tokens: &[usize], lr: f32) -> Result<f32> {
        let loss = self.loss_and_backward(tokens)?;
        // tok_emb / norms / heads
        for (w, g) in self.model.tok_emb.iter_mut().zip(&self.grads.tok_emb) {
            *w -= lr * g;
        }
        for (w, g) in self.model.final_norm.weight.iter_mut().zip(&self.grads.final_norm) {
            *w -= lr * g;
        }
        apply_lin(&mut self.model.lm_head, &self.grads.lm_head, lr);
        for (b, bg) in self.model.blocks.iter_mut().zip(&self.grads.blocks) {
            for (w, g) in b.attn_norm.weight.iter_mut().zip(&bg.attn_norm) {
                *w -= lr * g;
            }
            for (w, g) in b.ffn_norm.weight.iter_mut().zip(&bg.ffn_norm) {
                *w -= lr * g;
            }
            apply_lin(&mut b.attn.wq, &bg.wq, lr);
            apply_lin(&mut b.attn.wk, &bg.wk, lr);
            apply_lin(&mut b.attn.wv, &bg.wv, lr);
            apply_lin(&mut b.attn.wo, &bg.wo, lr);
            apply_lin(&mut b.ffn.gate, &bg.gate, lr);
            apply_lin(&mut b.ffn.up, &bg.up, lr);
            apply_lin(&mut b.ffn.down, &bg.down, lr);
        }
        Ok(loss)
    }
}

/// Mean next-token cross-entropy and its gradient `dL/d logits` (`[seq, vocab]`).
/// Position `t` predicts `tokens[t+1]`; the last position has no target.
fn cross_entropy(logits: &[f32], tokens: &[usize], vocab: usize) -> (f32, Vec<f32>) {
    let seq = tokens.len();
    let n_targets = seq.saturating_sub(1).max(1);
    let mut d = vec![0.0f32; seq * vocab];
    let mut loss = 0.0f32;
    for t in 0..seq.saturating_sub(1) {
        let row = &logits[t * vocab..t * vocab + vocab];
        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        let mut probs = vec![0.0f32; vocab];
        for j in 0..vocab {
            probs[j] = (row[j] - m).exp();
            sum += probs[j];
        }
        for p in probs.iter_mut() {
            *p /= sum;
        }
        let target = tokens[t + 1];
        loss += -(probs[target].max(1e-12)).ln();
        for j in 0..vocab {
            d[t * vocab + j] = (probs[j] - if j == target { 1.0 } else { 0.0 }) / n_targets as f32;
        }
    }
    (loss / n_targets as f32, d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{Attention, FeedForward, LlamaBlock, LlamaConfig, RmsNorm};

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
        // Small weights keep the toy model's logits in a sane range.
        let w: Vec<f32> = det(in_f * out_f, seed).iter().map(|v| v * 0.3).collect();
        Linear::new(in_f, out_f, w, vec![0.0; out_f]).unwrap()
    }

    fn tiny(vocab: usize, dim: usize, n_heads: usize, n_kv: usize, ffn: usize, n_layers: usize) -> LlamaModel {
        let hd = dim / n_heads;
        let cfg = LlamaConfig {
            vocab_size: vocab, model_dim: dim, n_layers, n_heads, n_kv_heads: n_kv,
            head_dim: hd, ffn_dim: ffn, rope_dim: hd, rope_base: 10000.0,
            rope_type: RopeType::Norm, norm_eps: 1e-5, context_len: 64,
        };
        let blocks = (0..n_layers)
            .map(|l| {
                let s = 100 + l as u64 * 50;
                LlamaBlock {
                    attn_norm: RmsNorm::new(vec![1.0; dim], 1e-5),
                    attn: Attention::new(
                        lin(dim, n_heads * hd, s), lin(dim, n_kv * hd, s + 1),
                        lin(dim, n_kv * hd, s + 2), lin(n_heads * hd, dim, s + 3),
                        n_heads, n_kv, hd, hd, 10000.0, RopeType::Norm,
                    ).unwrap(),
                    ffn_norm: RmsNorm::new(vec![1.0; dim], 1e-5),
                    ffn: FeedForward::new(lin(dim, ffn, s + 4), lin(dim, ffn, s + 5), lin(ffn, dim, s + 6)),
                }
            })
            .collect();
        LlamaModel {
            cfg,
            tok_emb: det(vocab * dim, 7).iter().map(|v| v * 0.3).collect(),
            blocks,
            final_norm: RmsNorm::new(vec![1.0; dim], 1e-5),
            lm_head: lin(dim, vocab, 999),
        }
    }

    // ── Primitive gradient checks (finite differences) ────────────────────────

    #[test]
    fn rmsnorm_backward_matches_finite_difference() {
        let d = 6;
        let x = det(d, 1);
        let w = det(d, 2).iter().map(|v| v + 1.0).collect::<Vec<_>>();
        let dy = det(d, 3);
        let (_, inv) = rmsnorm_forward_row(&x, &w, 1e-5);
        let mut dw = vec![0.0; d];
        let dx = rmsnorm_backward_row(&dy, &x, &w, inv, &mut dw);
        // Loss = Σ dy_i y_i ⇒ dL/dx, dL/dw are exactly the backward outputs.
        let loss = |x: &[f32], w: &[f32]| {
            let (y, _) = rmsnorm_forward_row(x, w, 1e-5);
            y.iter().zip(&dy).map(|(a, b)| a * b).sum::<f32>()
        };
        let eps = 1e-3;
        for i in 0..d {
            let mut xp = x.clone();
            xp[i] += eps;
            let mut xm = x.clone();
            xm[i] -= eps;
            let fd = (loss(&xp, &w) - loss(&xm, &w)) / (2.0 * eps);
            assert!((fd - dx[i]).abs() < 1e-2, "dx[{i}] fd={fd} got={}", dx[i]);
            let mut wp = w.clone();
            wp[i] += eps;
            let mut wm = w.clone();
            wm[i] -= eps;
            let fdw = (loss(&x, &wp) - loss(&x, &wm)) / (2.0 * eps);
            assert!((fdw - dw[i]).abs() < 1e-2, "dw[{i}] fd={fdw} got={}", dw[i]);
        }
    }

    #[test]
    fn silu_grad_matches_finite_difference() {
        for &g in &[-2.0f32, -0.5, 0.0, 0.7, 3.0] {
            let fd = (silu(g + 1e-3) - silu(g - 1e-3)) / 2e-3;
            assert!((fd - silu_grad(g)).abs() < 1e-3, "silu' at {g}");
        }
    }

    #[test]
    fn rope_transpose_is_the_inverse_rotation() {
        // Backward rotation undoes the forward one (orthogonality).
        let mut x = det(8, 5);
        let orig = x.clone();
        rope_apply(&mut x, 2, 4, 4, 3, 10000.0, RopeType::Norm, false);
        rope_apply(&mut x, 2, 4, 4, 3, 10000.0, RopeType::Norm, true);
        for (a, b) in x.iter().zip(&orig) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    // ── End-to-end gradient check on a tiny model ─────────────────────────────

    #[test]
    fn end_to_end_gradient_matches_finite_difference() {
        let model = tiny(7, 8, 2, 1, 16, 2); // GQA: 2 q-heads share 1 kv-head
        let mut tr = LlamaTrainer::new(model).unwrap();
        let tokens = [1usize, 3, 0, 5, 2];
        tr.loss_and_backward(&tokens).unwrap();

        let eps = 1e-3;
        // Check a sampling of parameters across distinct tensors.
        // 1) an embedding element
        {
            let g = tr.grads.tok_emb[3 * 8 + 2];
            tr.model.tok_emb[3 * 8 + 2] += eps;
            let lp = tr.loss(&tokens).unwrap();
            tr.model.tok_emb[3 * 8 + 2] -= 2.0 * eps;
            let lm = tr.loss(&tokens).unwrap();
            tr.model.tok_emb[3 * 8 + 2] += eps;
            let fd = (lp - lm) / (2.0 * eps);
            assert!((fd - g).abs() < 3e-2, "tok_emb fd={fd} got={g}");
        }
        // 2) an attention wq weight in block 0
        {
            let g = tr.grads.blocks[0].wq.dw[5];
            tr.model.blocks[0].attn.wq.weight.data[5] += eps;
            let lp = tr.loss(&tokens).unwrap();
            tr.model.blocks[0].attn.wq.weight.data[5] -= 2.0 * eps;
            let lm = tr.loss(&tokens).unwrap();
            tr.model.blocks[0].attn.wq.weight.data[5] += eps;
            let fd = (lp - lm) / (2.0 * eps);
            assert!((fd - g).abs() < 3e-2, "wq fd={fd} got={g}");
        }
        // 3) an FFN down weight in the last block
        {
            let g = tr.grads.blocks[1].down.dw[9];
            tr.model.blocks[1].ffn.down.weight.data[9] += eps;
            let lp = tr.loss(&tokens).unwrap();
            tr.model.blocks[1].ffn.down.weight.data[9] -= 2.0 * eps;
            let lm = tr.loss(&tokens).unwrap();
            tr.model.blocks[1].ffn.down.weight.data[9] += eps;
            let fd = (lp - lm) / (2.0 * eps);
            assert!((fd - g).abs() < 3e-2, "down fd={fd} got={g}");
        }
        // 4) an attn_norm weight
        {
            let g = tr.grads.blocks[0].attn_norm[3];
            tr.model.blocks[0].attn_norm.weight[3] += eps;
            let lp = tr.loss(&tokens).unwrap();
            tr.model.blocks[0].attn_norm.weight[3] -= 2.0 * eps;
            let lm = tr.loss(&tokens).unwrap();
            tr.model.blocks[0].attn_norm.weight[3] += eps;
            let fd = (lp - lm) / (2.0 * eps);
            assert!((fd - g).abs() < 3e-2, "attn_norm fd={fd} got={g}");
        }
        // 5) lm_head weight
        {
            let g = tr.grads.lm_head.dw[4];
            tr.model.lm_head.weight.data[4] += eps;
            let lp = tr.loss(&tokens).unwrap();
            tr.model.lm_head.weight.data[4] -= 2.0 * eps;
            let lm = tr.loss(&tokens).unwrap();
            tr.model.lm_head.weight.data[4] += eps;
            let fd = (lp - lm) / (2.0 * eps);
            assert!((fd - g).abs() < 3e-2, "lm_head fd={fd} got={g}");
        }
    }

    #[test]
    fn training_reduces_loss_on_a_memorizable_sequence() {
        let model = tiny(6, 16, 4, 2, 32, 2);
        let mut tr = LlamaTrainer::new(model).unwrap();
        let tokens = [1usize, 2, 3, 4, 5, 0, 1, 2]; // short, repeatable pattern
        let l0 = tr.loss(&tokens).unwrap();
        let mut last = l0;
        for _ in 0..300 {
            last = tr.train_step(&tokens, 0.05).unwrap();
        }
        let lf = tr.loss(&tokens).unwrap();
        assert!(lf < l0 * 0.5, "loss did not fall enough: {l0} → {lf} (last step {last})");
        assert!(lf.is_finite());
    }

    #[test]
    fn rejects_quantized_model() {
        let mut model = tiny(6, 8, 2, 2, 16, 1);
        model.lm_head = model.lm_head.quantize(crate::quant::QKind::Int8);
        assert!(LlamaTrainer::new(model).is_err());
    }
}
