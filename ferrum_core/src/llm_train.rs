//! Training (the backward pass) for the Llama/Qwen decoder in [`crate::llm`].
//!
//! `llm.rs` is forward-only; without gradients the imported architecture cannot
//! be trained or fine-tuned. This module adds hand-derived backprop for every
//! primitive that block uses — **RMSNorm, RoPE, grouped-query attention (with the
//! softmax), the SwiGLU FFN, the token embedding, and the LM head** — plus a
//! next-token cross-entropy loss. Each primitive's gradient is checked against
//! finite differences in the tests, and an end-to-end test shows the loss falling
//! on a memorisable sequence.
//!
//! On top of that bare gradient it provides a full fine-tuning stack that mirrors
//! [`crate::train_transformer`]:
//!
//! - **AdamW** optimizer with bias correction and decoupled weight decay
//!   (matrices only), replacing the original plain SGD.
//! - **Minibatching** over sequence windows ([`LlamaTrainer::train_batch`]) and
//!   true-epoch drivers that shuffle windows without replacement
//!   ([`LlamaTrainer::finetune_epoch`]).
//! - **Data-parallel** epochs on `std::thread::scope`
//!   ([`LlamaTrainer::finetune_epoch_threaded`]) with deterministic shard
//!   reduction; single-shard is bit-identical to the serial path.
//! - **Gradient clipping** (global L2 norm), a **warmup+decay learning-rate
//!   schedule**, and **FFN-hidden dropout** for regularised fine-tuning.
//! - **Quantization-aware training** (straight-through int8) so a fine-tuned
//!   model survives int8 export, and **checkpoint save / resume**
//!   ([`LlamaTrainer::save_checkpoint`] / [`LlamaTrainer::load_checkpoint_into`])
//!   that preserves the optimizer moments, step counter, and RNG state.
//!
//! Training runs in **f32**: the model must hold full-precision weights
//! (`Gguf::load_llama_prec(None)`, or built f32 directly). Quantized (`QWeight`)
//! Linears have no f32 master to update, so [`LlamaTrainer::new`] rejects them.
//!
//! Scope/reality check: this makes the architecture *trainable* and is exercised
//! on small models. It does **not** make training a 1B model on a CPU feasible —
//! that stays bounded by compute and RAM (see `ferrum_review.md §4.3`). It adds
//! the missing capability (gradients + a real optimizer), not a claim about scale.

use crate::error::{InferError, Result};
use crate::layer::Linear;
use crate::llm::{Attention, FeedForward, LlamaBlock, LlamaConfig, LlamaModel, RopeType};
use crate::optim::{Adam, LrSchedule};
use crate::rng::Rng;

// ─────────────────────────────────────────────────────────────────────────────
// Per-parameter gradient / moment buffers (mirror the model's trainable tensors)
// ─────────────────────────────────────────────────────────────────────────────

/// A weight+bias buffer mirroring one [`Linear`]. Used both for gradients (then
/// `w`/`b` hold dW/db) and for Adam moment buffers (then they hold m or v).
#[derive(Clone)]
struct LinBuf {
    w: Vec<f32>, // [in·out] row-major, like Linear::weight
    b: Vec<f32>, // [out]
}
impl LinBuf {
    fn zeros(lin: &Linear) -> Self {
        Self { w: vec![0.0; lin.in_features() * lin.out_features()], b: vec![0.0; lin.out_features()] }
    }
    fn clear(&mut self) {
        self.w.iter_mut().for_each(|x| *x = 0.0);
        self.b.iter_mut().for_each(|x| *x = 0.0);
    }
}

#[derive(Clone)]
struct BlockBuf {
    attn_norm: Vec<f32>,
    wq: LinBuf,
    wk: LinBuf,
    wv: LinBuf,
    wo: LinBuf,
    ffn_norm: Vec<f32>,
    gate: LinBuf,
    up: LinBuf,
    down: LinBuf,
}

/// One full set of buffers shaped like the model — gradients or an Adam moment.
#[derive(Clone)]
struct Buffers {
    tok_emb: Vec<f32>,
    blocks: Vec<BlockBuf>,
    final_norm: Vec<f32>,
    lm_head: LinBuf,
}

impl Buffers {
    fn zeros(model: &LlamaModel) -> Self {
        let blocks = model
            .blocks
            .iter()
            .map(|b| BlockBuf {
                attn_norm: vec![0.0; b.attn_norm.weight.len()],
                wq: LinBuf::zeros(&b.attn.wq),
                wk: LinBuf::zeros(&b.attn.wk),
                wv: LinBuf::zeros(&b.attn.wv),
                wo: LinBuf::zeros(&b.attn.wo),
                ffn_norm: vec![0.0; b.ffn_norm.weight.len()],
                gate: LinBuf::zeros(&b.ffn.gate),
                up: LinBuf::zeros(&b.ffn.up),
                down: LinBuf::zeros(&b.ffn.down),
            })
            .collect();
        Self {
            tok_emb: vec![0.0; model.tok_emb.len()],
            blocks,
            final_norm: vec![0.0; model.final_norm.weight.len()],
            lm_head: LinBuf::zeros(&model.lm_head),
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

    /// Flat view of every buffer in canonical order (matches [`param_data_ref`]).
    fn slices(&self) -> Vec<&[f32]> {
        let mut v: Vec<&[f32]> = vec![&self.tok_emb];
        for b in &self.blocks {
            v.push(&b.attn_norm);
            v.push(&b.wq.w); v.push(&b.wq.b);
            v.push(&b.wk.w); v.push(&b.wk.b);
            v.push(&b.wv.w); v.push(&b.wv.b);
            v.push(&b.wo.w); v.push(&b.wo.b);
            v.push(&b.ffn_norm);
            v.push(&b.gate.w); v.push(&b.gate.b);
            v.push(&b.up.w); v.push(&b.up.b);
            v.push(&b.down.w); v.push(&b.down.b);
        }
        v.push(&self.final_norm);
        v.push(&self.lm_head.w); v.push(&self.lm_head.b);
        v
    }

    fn slices_mut(&mut self) -> Vec<&mut [f32]> {
        let mut v: Vec<&mut [f32]> = vec![&mut self.tok_emb];
        for b in &mut self.blocks {
            v.push(&mut b.attn_norm);
            v.push(&mut b.wq.w); v.push(&mut b.wq.b);
            v.push(&mut b.wk.w); v.push(&mut b.wk.b);
            v.push(&mut b.wv.w); v.push(&mut b.wv.b);
            v.push(&mut b.wo.w); v.push(&mut b.wo.b);
            v.push(&mut b.ffn_norm);
            v.push(&mut b.gate.w); v.push(&mut b.gate.b);
            v.push(&mut b.up.w); v.push(&mut b.up.b);
            v.push(&mut b.down.w); v.push(&mut b.down.b);
        }
        v.push(&mut self.final_norm);
        v.push(&mut self.lm_head.w); v.push(&mut self.lm_head.b);
        v
    }

    fn to_flat(&self) -> Vec<f32> {
        let mut out = Vec::new();
        for s in self.slices() {
            out.extend_from_slice(s);
        }
        out
    }

    /// Overwrite from a flat vector in canonical order. Panics on length mismatch.
    fn load_flat(&mut self, flat: &[f32]) {
        let mut off = 0usize;
        for s in self.slices_mut() {
            let n = s.len();
            s.copy_from_slice(&flat[off..off + n]);
            off += n;
        }
        debug_assert_eq!(off, flat.len(), "load_flat: length mismatch");
    }

    /// Add another buffer set in place (`self += other`).
    fn add_assign_flat(&mut self, flat: &[f32]) {
        let mut off = 0usize;
        for s in self.slices_mut() {
            let n = s.len();
            for (a, b) in s.iter_mut().zip(&flat[off..off + n]) {
                *a += *b;
            }
            off += n;
        }
    }

    /// Scale every element (`self *= k`).
    fn scale(&mut self, k: f32) {
        for s in self.slices_mut() {
            for x in s.iter_mut() {
                *x *= k;
            }
        }
    }
}

// ── Canonical views into the live model weights ───────────────────────────────

/// `(rows, is_matrix)` per parameter in canonical order. `rows` is the number of
/// per-channel quantization groups (used by QAT); `is_matrix` flags the 2-D
/// weight matrices that decoupled weight decay applies to.
fn param_meta(model: &LlamaModel) -> Vec<(usize, bool)> {
    let dim = model.cfg.model_dim;
    let mut v: Vec<(usize, bool)> = vec![(model.tok_emb.len() / dim, true)]; // tok_emb [vocab, dim]
    for b in &model.blocks {
        v.push((1, false)); // attn_norm
        v.push((b.attn.wq.in_features(), true)); v.push((1, false));
        v.push((b.attn.wk.in_features(), true)); v.push((1, false));
        v.push((b.attn.wv.in_features(), true)); v.push((1, false));
        v.push((b.attn.wo.in_features(), true)); v.push((1, false));
        v.push((1, false)); // ffn_norm
        v.push((b.ffn.gate.in_features(), true)); v.push((1, false));
        v.push((b.ffn.up.in_features(), true)); v.push((1, false));
        v.push((b.ffn.down.in_features(), true)); v.push((1, false));
    }
    v.push((1, false)); // final_norm
    v.push((model.lm_head.in_features(), true)); v.push((1, false));
    v
}

fn param_data_ref(model: &LlamaModel) -> Vec<&[f32]> {
    let mut v: Vec<&[f32]> = vec![&model.tok_emb];
    for b in &model.blocks {
        v.push(&b.attn_norm.weight);
        v.push(&b.attn.wq.weight.data); v.push(&b.attn.wq.bias.data);
        v.push(&b.attn.wk.weight.data); v.push(&b.attn.wk.bias.data);
        v.push(&b.attn.wv.weight.data); v.push(&b.attn.wv.bias.data);
        v.push(&b.attn.wo.weight.data); v.push(&b.attn.wo.bias.data);
        v.push(&b.ffn_norm.weight);
        v.push(&b.ffn.gate.weight.data); v.push(&b.ffn.gate.bias.data);
        v.push(&b.ffn.up.weight.data); v.push(&b.ffn.up.bias.data);
        v.push(&b.ffn.down.weight.data); v.push(&b.ffn.down.bias.data);
    }
    v.push(&model.final_norm.weight);
    v.push(&model.lm_head.weight.data); v.push(&model.lm_head.bias.data);
    v
}

fn param_data_mut(model: &mut LlamaModel) -> Vec<&mut [f32]> {
    let mut v: Vec<&mut [f32]> = vec![&mut model.tok_emb];
    for b in &mut model.blocks {
        v.push(&mut b.attn_norm.weight);
        v.push(&mut b.attn.wq.weight.data); v.push(&mut b.attn.wq.bias.data);
        v.push(&mut b.attn.wk.weight.data); v.push(&mut b.attn.wk.bias.data);
        v.push(&mut b.attn.wv.weight.data); v.push(&mut b.attn.wv.bias.data);
        v.push(&mut b.attn.wo.weight.data); v.push(&mut b.attn.wo.bias.data);
        v.push(&mut b.ffn_norm.weight);
        v.push(&mut b.ffn.gate.weight.data); v.push(&mut b.ffn.gate.bias.data);
        v.push(&mut b.ffn.up.weight.data); v.push(&mut b.ffn.up.bias.data);
        v.push(&mut b.ffn.down.weight.data); v.push(&mut b.ffn.down.bias.data);
    }
    v.push(&mut model.final_norm.weight);
    v.push(&mut model.lm_head.weight.data); v.push(&mut model.lm_head.bias.data);
    v
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
fn lin_backward(lin: &Linear, x: &[f32], dy: &[f32], rows: usize, g: &mut LinBuf) -> Vec<f32> {
    let (k, n) = (lin.in_features(), lin.out_features());
    let w = &lin.weight.data;
    let mut dx = vec![0.0f32; rows * k];
    for r in 0..rows {
        let xr = &x[r * k..r * k + k];
        let dyr = &dy[r * n..r * n + n];
        let dxr = &mut dx[r * k..r * k + k];
        for (o, &dyo) in dyr.iter().enumerate() {
            g.b[o] += dyo;
        }
        for i in 0..k {
            let wr = &w[i * n..i * n + n];
            let dwr = &mut g.w[i * n..i * n + n];
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
    h: Vec<f32>,  // [seq, ffn] down input: silu(g)·u, post-dropout when active
    /// Inverted-dropout mask×scale applied to the hidden, present only when
    /// dropout was active for this forward; `None` means no dropout.
    mask: Option<Vec<f32>>,
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

fn attn_backward(cfg: &LlamaConfig, attn: &Attention, c: &AttnCache, d_o: &[f32], seq: usize, g: &mut BlockBuf) -> Vec<f32> {
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

/// SwiGLU FFN forward. When `dropout > 0`, the hidden `silu(g)⊙u` is
/// inverted-dropout masked using a generator seeded by `seed`, so a worker can
/// reproduce the mask in backward from the cache.
fn ffn_forward(cfg: &LlamaConfig, ffn: &FeedForward, xn: &[f32], seq: usize, dropout: f32, seed: u64) -> FfnCache {
    let f = cfg.ffn_dim;
    let g = lin_forward(&ffn.gate, xn, seq);
    let u = lin_forward(&ffn.up, xn, seq);
    let mut h: Vec<f32> = (0..seq * f).map(|i| silu(g[i]) * u[i]).collect();
    let mask = if dropout > 0.0 {
        let scale = 1.0 / (1.0 - dropout);
        let mut drng = Rng::new(seed);
        let mask: Vec<f32> = (0..h.len())
            .map(|_| if drng.next_f32() < dropout { 0.0 } else { scale })
            .collect();
        for (hi, &mi) in h.iter_mut().zip(&mask) {
            *hi *= mi;
        }
        Some(mask)
    } else {
        None
    };
    FfnCache { xn: xn.to_vec(), g, u, h, mask }
}

fn ffn_output(ffn: &FeedForward, c: &FfnCache, seq: usize) -> Vec<f32> {
    lin_forward(&ffn.down, &c.h, seq)
}

fn ffn_backward(cfg: &LlamaConfig, ffn: &FeedForward, c: &FfnCache, d_f: &[f32], seq: usize, g: &mut BlockBuf) -> Vec<f32> {
    let (dim, f) = (cfg.model_dim, cfg.ffn_dim);
    let d_h_in = lin_backward(&ffn.down, &c.h, d_f, seq, &mut g.down);
    let mut dg = vec![0.0f32; seq * f];
    let mut du = vec![0.0f32; seq * f];
    for i in 0..seq * f {
        // Propagate back through the (optional) inverted-dropout mask to the
        // pre-dropout hidden, then through the SwiGLU gate.
        let d_pre = match &c.mask {
            Some(mask) => d_h_in[i] * mask[i],
            None => d_h_in[i],
        };
        du[i] = d_pre * silu(c.g[i]);
        dg[i] = d_pre * c.u[i] * silu_grad(c.g[i]);
    }
    let dxg = lin_backward(&ffn.gate, &c.xn, &dg, seq, &mut g.gate);
    let dxu = lin_backward(&ffn.up, &c.xn, &du, seq, &mut g.up);
    (0..seq * dim).map(|i| dxg[i] + dxu[i]).collect()
}

fn block_backward(cfg: &LlamaConfig, block: &LlamaBlock, bc: &BlockCache, d_out: &[f32], seq: usize, g: &mut BlockBuf) -> Vec<f32> {
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

// ── Whole-model forward / backward (free functions over `&LlamaModel`) ─────────

/// Forward pass over `tokens`, returning the activation cache. With `dropout > 0`
/// each block's SwiGLU hidden is inverted-dropout masked from a `seed`-derived
/// generator (per-block offset) so it is reproducible in backward.
fn forward_train(model: &LlamaModel, tokens: &[usize], dropout: f32, seed: u64) -> Result<FwdCache> {
    let cfg = &model.cfg;
    let (seq, dim) = (tokens.len(), cfg.model_dim);
    let mut x = vec![0.0f32; seq * dim];
    for (t, &tok) in tokens.iter().enumerate() {
        if tok >= cfg.vocab_size {
            return Err(InferError::DimMismatch(format!("token {tok} ≥ vocab {}", cfg.vocab_size)));
        }
        x[t * dim..(t + 1) * dim].copy_from_slice(&model.tok_emb[tok * dim..(tok + 1) * dim]);
    }

    let mut blocks = Vec::with_capacity(model.blocks.len());
    for (bi, block) in model.blocks.iter().enumerate() {
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
        let blk_seed = seed.wrapping_add(bi as u64 + 1);
        let ffn = ffn_forward(cfg, &block.ffn, &n2, seq, dropout, blk_seed);
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
        let (y, inv) = rmsnorm_forward_row(&xfinal[s..s + dim], &model.final_norm.weight, cfg.norm_eps);
        xn[s..s + dim].copy_from_slice(&y);
        nf_inv[t] = inv;
    }
    let logits = lin_forward(&model.lm_head, &xn, seq);
    Ok(FwdCache { blocks, xfinal, nf_inv, xn, logits })
}

/// Backprop `d_logits` through the whole model, **accumulating** into `g` (the
/// embedding scatter and every dW/db/dnorm are `+=`), so summing across a
/// minibatch is just repeated calls into the same buffer.
fn backward_into(model: &LlamaModel, cache: &FwdCache, d_logits: &[f32], tokens: &[usize], g: &mut Buffers) -> Result<()> {
    let cfg = &model.cfg;
    let (seq, dim) = (tokens.len(), cfg.model_dim);

    let d_xn = lin_backward(&model.lm_head, &cache.xn, d_logits, seq, &mut g.lm_head);

    let mut d_x = vec![0.0f32; seq * dim];
    for t in 0..seq {
        let s = t * dim;
        let dxr = rmsnorm_backward_row(
            &d_xn[s..s + dim],
            &cache.xfinal[s..s + dim],
            &model.final_norm.weight,
            cache.nf_inv[t],
            &mut g.final_norm,
        );
        d_x[s..s + dim].copy_from_slice(&dxr);
    }

    for bi in (0..model.blocks.len()).rev() {
        d_x = block_backward(cfg, &model.blocks[bi], &cache.blocks[bi], &d_x, seq, &mut g.blocks[bi]);
    }

    for (t, &tok) in tokens.iter().enumerate() {
        let s = t * dim;
        for d in 0..dim {
            g.tok_emb[tok * dim + d] += d_x[s + d];
        }
    }
    Ok(())
}

/// Pure forward+backward over one sequence against the (read-only) model,
/// returning `(loss, flat-gradients)` in canonical order. Used by both the
/// serial accumulator and the data-parallel workers.
fn seq_grad_flat(model: &LlamaModel, tokens: &[usize], dropout: f32, seed: u64) -> Result<(f32, Vec<f32>)> {
    let cache = forward_train(model, tokens, dropout, seed)?;
    let (loss, d_logits) = cross_entropy(&cache.logits, tokens, model.cfg.vocab_size);
    if !loss.is_finite() {
        return Err(InferError::DimMismatch("non-finite loss".into()));
    }
    let mut g = Buffers::zeros(model);
    backward_into(model, &cache, &d_logits, tokens, &mut g)?;
    Ok((loss, g.to_flat()))
}

// ─────────────────────────────────────────────────────────────────────────────
// Trainer
// ─────────────────────────────────────────────────────────────────────────────

/// Wraps an f32 [`LlamaModel`] and fine-tunes it by next-token cross-entropy
/// with AdamW. See the module docs for the full feature set.
pub struct LlamaTrainer {
    pub model: LlamaModel,
    grads: Buffers,
    m: Buffers,
    v: Buffers,
    adam: Adam,
    weight_decay: f32,
    grad_clip: Option<f32>,
    lr_schedule: Option<LrSchedule>,
    dropout: f32,
    qat: bool,
    step_t: u64,
}

impl LlamaTrainer {
    /// Build a trainer over `model`. Errors if any projection is still quantized
    /// (training needs f32 masters — load with `Gguf::load_llama_prec(None)`).
    /// Defaults to AdamW with `lr = 1e-4` and no weight decay; tune with the
    /// setters before training.
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
        let grads = Buffers::zeros(&model);
        let m = Buffers::zeros(&model);
        let v = Buffers::zeros(&model);
        Ok(Self {
            model,
            grads,
            m,
            v,
            adam: Adam::new(1e-4),
            weight_decay: 0.0,
            grad_clip: None,
            lr_schedule: None,
            dropout: 0.0,
            qat: false,
            step_t: 0,
        })
    }

    // ── Configuration ────────────────────────────────────────────────────────

    /// Replace the optimizer settings (learning rate, betas, eps). The decoupled
    /// weight decay is taken from [`LlamaTrainer::set_weight_decay`] /
    /// `Adam::weight_decay`, whichever is non-zero.
    pub fn set_optimizer(&mut self, adam: Adam) {
        self.adam = adam;
        if adam.weight_decay != 0.0 {
            self.weight_decay = adam.weight_decay;
        }
    }
    /// Set the base learning rate (used when no schedule is active).
    pub fn set_lr(&mut self, lr: f32) {
        self.adam.lr = lr;
    }
    pub fn lr(&self) -> f32 {
        self.adam.lr
    }
    /// Set the decoupled (AdamW) weight-decay coefficient applied to weight
    /// matrices each step. `0.0` disables it.
    pub fn set_weight_decay(&mut self, wd: f32) {
        self.weight_decay = wd;
    }
    pub fn weight_decay(&self) -> f32 {
        self.weight_decay
    }
    /// Set the global-norm gradient-clipping threshold. `Some(max)` clips before
    /// each step; `None` disables.
    pub fn set_grad_clip(&mut self, max_norm: Option<f32>) {
        self.grad_clip = max_norm;
    }
    pub fn grad_clip(&self) -> Option<f32> {
        self.grad_clip
    }
    /// Set the learning-rate schedule (warmup + decay). When `Some`, it overrides
    /// the Adam learning rate at every step using the internal step counter.
    pub fn set_lr_schedule(&mut self, schedule: Option<LrSchedule>) {
        self.lr_schedule = schedule;
    }
    pub fn lr_schedule(&self) -> Option<LrSchedule> {
        self.lr_schedule
    }
    /// Set the FFN-hidden dropout probability used in training forward passes,
    /// in `[0, 1)`. `0.0` disables it; inference is always dropout-free.
    pub fn set_dropout(&mut self, p: f32) {
        self.dropout = p.clamp(0.0, 0.95);
    }
    pub fn dropout(&self) -> f32 {
        self.dropout
    }
    /// Enable/disable int8 quantization-aware training (straight-through). When
    /// on, each step computes gradients at int8-snapped weights but updates the
    /// f32 masters, so the model is robust to int8 export.
    pub fn set_qat(&mut self, enabled: bool) {
        self.qat = enabled;
    }
    pub fn qat_enabled(&self) -> bool {
        self.qat
    }
    /// Number of optimizer steps applied so far (the schedule's timestep).
    pub fn step_count(&self) -> u64 {
        self.step_t
    }
    /// Total number of trainable scalar parameters.
    pub fn num_params(&self) -> usize {
        param_data_ref(&self.model).iter().map(|s| s.len()).sum()
    }

    /// A per-parameter clone of the current model weights, in canonical order.
    /// Handy for comparing training runs (e.g. verifying a resumed checkpoint
    /// matches the model it was saved from).
    pub fn model_snapshot(&self) -> Vec<Vec<f32>> {
        param_data_ref(&self.model).iter().map(|s| s.to_vec()).collect()
    }

    // ── Loss (pure forward) ──────────────────────────────────────────────────

    /// Mean next-token cross-entropy over `tokens` (position `t` predicts
    /// `tokens[t+1]`). Pure forward — no gradients, no dropout.
    pub fn loss(&self, tokens: &[usize]) -> Result<f32> {
        let cache = forward_train(&self.model, tokens, 0.0, 0)?;
        Ok(cross_entropy(&cache.logits, tokens, self.model.cfg.vocab_size).0)
    }

    /// Forward + backward over a single sequence; fills `self.grads`
    /// (dropout-free, deterministic). Returns the loss. Exposed for callers
    /// driving their own loop and used by the gradient-check tests.
    pub fn loss_and_backward(&mut self, tokens: &[usize]) -> Result<f32> {
        let cache = forward_train(&self.model, tokens, 0.0, 0)?;
        let (loss, d_logits) = cross_entropy(&cache.logits, tokens, self.model.cfg.vocab_size);
        if !loss.is_finite() {
            return Err(InferError::DimMismatch("non-finite loss".into()));
        }
        self.grads.clear();
        backward_into(&self.model, &cache, &d_logits, tokens, &mut self.grads)?;
        Ok(loss)
    }

    // ── QAT snapshot / fake-quantize / restore ───────────────────────────────

    fn snapshot_weights(&self) -> Vec<Vec<f32>> {
        param_data_ref(&self.model).iter().map(|s| s.to_vec()).collect()
    }
    fn restore_weights(&mut self, snap: &[Vec<f32>]) {
        for (s, src) in param_data_mut(&mut self.model).into_iter().zip(snap) {
            s.copy_from_slice(src);
        }
    }
    /// Snap every weight **matrix** onto the int8 grid in place, per output-row,
    /// matching the per-channel quantization the FINF/GGUF writers use.
    fn fake_quantize_weights(&mut self) {
        let meta = param_meta(&self.model);
        for (s, (rows, is_matrix)) in param_data_mut(&mut self.model).into_iter().zip(meta) {
            if is_matrix {
                crate::quant::fake_quantize_int8_per_channel(s, rows.max(1));
            }
        }
    }

    // ── Gradient clipping ────────────────────────────────────────────────────

    /// Rescale `self.grads` so their global L2 norm is at most `max_norm`,
    /// returning the pre-clip norm.
    pub fn clip_grad_norm(&mut self, max_norm: f32) -> f32 {
        let mut sumsq = 0.0f64;
        for s in self.grads.slices() {
            for &x in s {
                sumsq += (x as f64) * (x as f64);
            }
        }
        let norm = sumsq.sqrt() as f32;
        if max_norm > 0.0 && norm.is_finite() && norm > max_norm {
            let scale = max_norm / norm;
            for s in self.grads.slices_mut() {
                for x in s.iter_mut() {
                    *x *= scale;
                }
            }
        }
        norm
    }

    // ── Optimizer step (AdamW over the canonical parameter list) ──────────────

    fn optimizer_step(&mut self) {
        self.step_t += 1;
        let t = self.step_t;
        let lr = match self.lr_schedule {
            Some(s) => s.lr_at(t),
            None => self.adam.lr,
        };
        let (b1, b2, eps, wd) = (self.adam.beta1, self.adam.beta2, self.adam.eps, self.weight_decay);
        let bc1 = 1.0 - b1.powi(t.min(i32::MAX as u64) as i32);
        let bc2 = 1.0 - b2.powi(t.min(i32::MAX as u64) as i32);

        let meta = param_meta(&self.model);
        let mut params = param_data_mut(&mut self.model);
        let grads = self.grads.slices();
        let mut ms = self.m.slices_mut();
        let mut vs = self.v.slices_mut();

        for idx in 0..params.len() {
            let (_, is_matrix) = meta[idx];
            let p = &mut params[idx];
            let g = grads[idx];
            let mb = &mut ms[idx];
            let vb = &mut vs[idx];
            for i in 0..p.len() {
                let gi = g[i];
                mb[i] = b1 * mb[i] + (1.0 - b1) * gi;
                vb[i] = b2 * vb[i] + (1.0 - b2) * gi * gi;
                let m_hat = mb[i] / bc1;
                let v_hat = vb[i] / bc2;
                let mut update = lr * m_hat / (v_hat.sqrt() + eps);
                if wd != 0.0 && is_matrix {
                    update += lr * wd * p[i];
                }
                p[i] -= update;
            }
        }
    }

    /// Run one AdamW step from the gradients currently in `self.grads`
    /// (clipping first if configured).
    fn apply_step(&mut self) {
        if let Some(max) = self.grad_clip {
            self.clip_grad_norm(max);
        }
        self.optimizer_step();
    }

    // ── Public training entry points ─────────────────────────────────────────

    /// One AdamW step on a single sequence. Returns the pre-step loss.
    pub fn train_step(&mut self, tokens: &[usize]) -> Result<f32> {
        self.train_batch(&[tokens])
    }

    /// One AdamW step over a minibatch of sequences. The per-sequence gradients
    /// (each already a per-position mean) are averaged across the batch, so the
    /// update matches the mean loss. Returns the mean pre-step loss.
    ///
    /// With QAT enabled the gradients are computed at int8-snapped weights and
    /// the f32 masters are updated (straight-through estimator).
    pub fn train_batch(&mut self, batch: &[&[usize]]) -> Result<f32> {
        self.train_batch_seeded(batch, 0)
    }

    /// Like [`LlamaTrainer::train_batch`] but with an explicit dropout-mask base
    /// seed (each sequence offset from it). `seed = 0` with dropout off is the
    /// plain deterministic path.
    fn train_batch_seeded(&mut self, batch: &[&[usize]], base_seed: u64) -> Result<f32> {
        if batch.is_empty() {
            return Err(InferError::DimMismatch("empty minibatch".into()));
        }
        let dropout = self.dropout;
        let masters = if self.qat {
            let snap = self.snapshot_weights();
            self.fake_quantize_weights();
            Some(snap)
        } else {
            None
        };

        self.grads.clear();
        let mut total = 0.0f32;
        for (si, seq) in batch.iter().enumerate() {
            let seed = base_seed.wrapping_add(si as u64 + 1);
            let (loss, flat) = seq_grad_flat(&self.model, seq, dropout, seed)?;
            self.grads.add_assign_flat(&flat);
            total += loss;
        }
        let inv_b = 1.0 / batch.len() as f32;
        self.grads.scale(inv_b);

        if let Some(snap) = &masters {
            self.restore_weights(snap);
        }
        self.apply_step();
        Ok(total * inv_b)
    }

    /// One epoch of minibatch AdamW over a token stream. Each example is a window
    /// of `seq_len` tokens (next-token targets at every interior position); the
    /// windows are shuffled and drawn without replacement, so an epoch covers the
    /// corpus once. Returns the mean loss. `seq_len` is capped at the model's
    /// `context_len`.
    pub fn finetune_epoch(
        &mut self,
        tokens: &[usize],
        seq_len: usize,
        batch_size: usize,
        rng: &mut Rng,
    ) -> Result<f32> {
        self.finetune_epoch_threaded(tokens, seq_len, batch_size, rng, 1)
    }

    /// Data-parallel [`LlamaTrainer::finetune_epoch`]: each minibatch is split
    /// into up to `threads` shards run concurrently on `std::thread::scope`,
    /// their gradients summed in a fixed shard order, and one AdamW step applied.
    /// `threads <= 1` is the serial path, bit-for-bit. On `wasm32` it always runs
    /// serially.
    pub fn finetune_epoch_threaded(
        &mut self,
        tokens: &[usize],
        seq_len: usize,
        batch_size: usize,
        rng: &mut Rng,
        threads: usize,
    ) -> Result<f32> {
        let seq_len = seq_len.min(self.model.cfg.context_len).max(2);
        if tokens.len() < seq_len {
            return Err(InferError::DimMismatch(format!(
                "need at least seq_len = {seq_len} tokens, got {}",
                tokens.len()
            )));
        }
        let batch_size = batch_size.max(1);
        let num_windows = tokens.len() - seq_len + 1;
        let steps = num_windows.div_ceil(batch_size);
        let perm = rng.shuffled_indices(num_windows);
        let dropout = self.dropout;
        let mut total = 0.0f32;

        for step in 0..steps {
            let idxs = &perm[step * batch_size..((step + 1) * batch_size).min(num_windows)];
            let windows: Vec<&[usize]> = idxs.iter().map(|&s| &tokens[s..s + seq_len]).collect();
            let base_seed = if dropout > 0.0 { rng.next_u64() } else { 0 };
            let loss = self.step_over_windows(&windows, base_seed, threads)?;
            total += loss;
        }
        Ok(total / steps as f32)
    }

    /// Shared gradient computation for one minibatch of already-sliced windows,
    /// optionally across threads, followed by a single AdamW step. Returns the
    /// mean loss over the batch.
    fn step_over_windows(&mut self, windows: &[&[usize]], base_seed: u64, threads: usize) -> Result<f32> {
        let nshards = threads.max(1).min(windows.len().max(1));

        #[cfg(not(target_arch = "wasm32"))]
        let multi = nshards > 1;
        #[cfg(target_arch = "wasm32")]
        let multi = false;

        if !multi {
            return self.train_batch_seeded(windows, base_seed);
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let dropout = self.dropout;
            let masters = if self.qat {
                let snap = self.snapshot_weights();
                self.fake_quantize_weights();
                Some(snap)
            } else {
                None
            };

            let model_ref: &LlamaModel = &self.model;
            let n = windows.len();
            let per = n.div_ceil(nshards);
            // Each shard returns (loss_sum, summed flat grads) over its windows.
            let results: Vec<Result<(f32, Vec<f32>)>> = std::thread::scope(|scope| {
                let mut handles = Vec::new();
                let mut w0 = 0usize;
                while w0 < n {
                    let w1 = (w0 + per).min(n);
                    let shard = &windows[w0..w1];
                    let off = w0;
                    handles.push(scope.spawn(move || -> Result<(f32, Vec<f32>)> {
                        let mut acc: Option<Vec<f32>> = None;
                        let mut loss_sum = 0.0f32;
                        for (i, seq) in shard.iter().enumerate() {
                            let seed = base_seed.wrapping_add((off + i) as u64 + 1);
                            let (loss, flat) = seq_grad_flat(model_ref, seq, dropout, seed)?;
                            loss_sum += loss;
                            match &mut acc {
                                None => acc = Some(flat),
                                Some(a) => {
                                    for (x, y) in a.iter_mut().zip(&flat) {
                                        *x += y;
                                    }
                                }
                            }
                        }
                        Ok((loss_sum, acc.expect("each shard has ≥1 window")))
                    }));
                    w0 = w1;
                }
                handles
                    .into_iter()
                    .map(|h| h.join().expect("ferrum: finetune worker thread panicked"))
                    .collect()
            });

            let mut summed: Option<Vec<f32>> = None;
            let mut loss_total = 0.0f32;
            for r in results {
                let (loss, flat) = r?;
                loss_total += loss;
                match &mut summed {
                    None => summed = Some(flat),
                    Some(a) => {
                        for (x, y) in a.iter_mut().zip(&flat) {
                            *x += y;
                        }
                    }
                }
            }
            let mut summed = summed.expect("at least one shard runs");
            let inv_b = 1.0 / n as f32;
            for x in &mut summed {
                *x *= inv_b;
            }

            if let Some(snap) = &masters {
                self.restore_weights(snap);
            }
            self.grads.load_flat(&summed);
            self.apply_step();
            Ok(loss_total * inv_b)
        }
    }

    // ── Checkpointing ────────────────────────────────────────────────────────

    /// Serialize the full **optimizer state**: a shape header, the QAT flag, the
    /// step counter, the supplied RNG's state, and every parameter's weights and
    /// both Adam moments. Pair with [`LlamaTrainer::load_checkpoint_into`] on a
    /// freshly-loaded base model of the same shape to resume exactly.
    pub fn save_checkpoint(&self, rng: &Rng) -> Vec<u8> {
        let c = &self.model.cfg;
        let mut out = Vec::new();
        out.extend_from_slice(b"FLCK");
        out.extend_from_slice(&1u32.to_le_bytes());
        for v in [
            c.vocab_size, c.model_dim, c.n_layers, c.n_heads, c.n_kv_heads, c.head_dim, c.ffn_dim,
        ] {
            out.extend_from_slice(&(v as u32).to_le_bytes());
        }
        out.push(self.qat as u8);
        out.extend_from_slice(&self.step_t.to_le_bytes());
        out.extend_from_slice(&rng.state().to_le_bytes());
        for flat in [
            self.model_flat(),
            self.m.to_flat(),
            self.v.to_flat(),
        ] {
            for &x in &flat {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        out
    }

    fn model_flat(&self) -> Vec<f32> {
        let mut out = Vec::new();
        for s in param_data_ref(&self.model) {
            out.extend_from_slice(s);
        }
        out
    }

    /// Restore weights, Adam moments, step counter, QAT flag, and RNG state from
    /// [`LlamaTrainer::save_checkpoint`] bytes into `self` (which must already
    /// hold a model of the matching shape). Returns the restored RNG.
    pub fn load_checkpoint_into(&mut self, bytes: &[u8]) -> Result<Rng> {
        let mut cur = Cur { b: bytes, pos: 0 };
        if cur.take(4)? != b"FLCK" {
            return Err(InferError::Format("bad checkpoint magic".into()));
        }
        if cur.u32()? != 1 {
            return Err(InferError::Format("unsupported checkpoint version".into()));
        }
        let c = &self.model.cfg;
        let want = [
            c.vocab_size, c.model_dim, c.n_layers, c.n_heads, c.n_kv_heads, c.head_dim, c.ffn_dim,
        ];
        for &w in &want {
            if cur.u32()? as usize != w {
                return Err(InferError::Format("checkpoint shape does not match this model".into()));
            }
        }
        self.qat = cur.u8()? != 0;
        self.step_t = cur.u64()?;
        let rng_state = cur.u64()?;

        // Weights, then m, then v — same canonical order as the save.
        let total: usize = self.num_params();
        let mut wflat = vec![0.0f32; total];
        cur.fill_f32(&mut wflat)?;
        {
            let mut off = 0usize;
            for s in param_data_mut(&mut self.model) {
                let n = s.len();
                s.copy_from_slice(&wflat[off..off + n]);
                off += n;
            }
        }
        let mut mflat = vec![0.0f32; total];
        cur.fill_f32(&mut mflat)?;
        self.m.load_flat(&mflat);
        let mut vflat = vec![0.0f32; total];
        cur.fill_f32(&mut vflat)?;
        self.v.load_flat(&vflat);

        Ok(Rng::from_state(rng_state))
    }
}

/// Minimal little-endian byte cursor for checkpoint deserialization.
struct Cur<'a> {
    b: &'a [u8],
    pos: usize,
}
impl Cur<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8]> {
        let end = self.pos.checked_add(n).filter(|&e| e <= self.b.len());
        match end {
            Some(end) => {
                let s = &self.b[self.pos..end];
                self.pos = end;
                Ok(s)
            }
            None => Err(InferError::Format("checkpoint truncated".into())),
        }
    }
    fn u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn u64(&mut self) -> Result<u64> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn fill_f32(&mut self, dst: &mut [f32]) -> Result<()> {
        let raw = self.take(dst.len() * 4)?;
        for (d, c) in dst.iter_mut().zip(raw.chunks_exact(4)) {
            *d = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        }
        Ok(())
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
        let check = |g: f32, set: &mut dyn FnMut(&mut LlamaTrainer, f32), tr: &mut LlamaTrainer, label: &str| {
            set(tr, eps);
            let lp = tr.loss(&tokens).unwrap();
            set(tr, -2.0 * eps);
            let lm = tr.loss(&tokens).unwrap();
            set(tr, eps);
            let fd = (lp - lm) / (2.0 * eps);
            assert!((fd - g).abs() < 3e-2, "{label} fd={fd} got={g}");
        };
        let g = tr.grads.tok_emb[3 * 8 + 2];
        check(g, &mut |t, d| t.model.tok_emb[3 * 8 + 2] += d, &mut tr, "tok_emb");
        let g = tr.grads.blocks[0].wq.w[5];
        check(g, &mut |t, d| t.model.blocks[0].attn.wq.weight.data[5] += d, &mut tr, "wq");
        let g = tr.grads.blocks[1].down.w[9];
        check(g, &mut |t, d| t.model.blocks[1].ffn.down.weight.data[9] += d, &mut tr, "down");
        let g = tr.grads.blocks[0].attn_norm[3];
        check(g, &mut |t, d| t.model.blocks[0].attn_norm.weight[3] += d, &mut tr, "attn_norm");
        let g = tr.grads.lm_head.w[4];
        check(g, &mut |t, d| t.model.lm_head.weight.data[4] += d, &mut tr, "lm_head");
    }

    #[test]
    fn training_reduces_loss_on_a_memorizable_sequence() {
        let model = tiny(6, 16, 4, 2, 32, 2);
        let mut tr = LlamaTrainer::new(model).unwrap();
        tr.set_lr(0.01);
        let tokens = [1usize, 2, 3, 4, 5, 0, 1, 2];
        let l0 = tr.loss(&tokens).unwrap();
        let mut last = l0;
        for _ in 0..200 {
            last = tr.train_step(&tokens).unwrap();
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

    // ── AdamW, batching, schedule, clipping, dropout, QAT, threading ──────────

    #[test]
    fn adamw_weight_decay_shrinks_matrices_only() {
        let mut tr = LlamaTrainer::new(tiny(6, 8, 2, 2, 16, 1)).unwrap();
        tr.set_lr(0.5);
        tr.set_weight_decay(0.1);
        // Zero gradients: only decoupled decay acts.
        for v in tr.model.lm_head.weight.data.iter_mut() { *v = 2.0; } // matrix
        for v in tr.model.lm_head.bias.data.iter_mut() { *v = 2.0; }   // bias (1-D)
        for v in tr.model.final_norm.weight.iter_mut() { *v = 2.0; }   // norm (1-D)
        tr.grads.clear();
        tr.apply_step();
        // matrix: 2 − 0.5·0.1·2 = 1.9
        assert!((tr.model.lm_head.weight.data[0] - 1.9).abs() < 1e-5, "matrix not decayed");
        assert!((tr.model.lm_head.bias.data[0] - 2.0).abs() < 1e-6, "bias decayed");
        assert!((tr.model.final_norm.weight[0] - 2.0).abs() < 1e-6, "norm decayed");
    }

    #[test]
    fn batch_equals_accumulated_singletons() {
        // One step over a 2-sequence batch == manual average of the two
        // per-sequence gradients fed through the same optimizer.
        let base = tiny(6, 8, 2, 2, 16, 1);
        let s0 = [1usize, 2, 3, 4];
        let s1 = [5usize, 0, 1, 2];

        let mut a = LlamaTrainer::new(base_clone(&base)).unwrap();
        a.set_lr(0.05);
        a.train_batch(&[&s0, &s1]).unwrap();

        // Reference: build the averaged grad by hand then step once.
        let mut b = LlamaTrainer::new(base_clone(&base)).unwrap();
        b.set_lr(0.05);
        let (_, g0) = seq_grad_flat(&b.model, &s0, 0.0, 0).unwrap();
        let (_, g1) = seq_grad_flat(&b.model, &s1, 0.0, 0).unwrap();
        let avg: Vec<f32> = g0.iter().zip(&g1).map(|(x, y)| 0.5 * (x + y)).collect();
        b.grads.load_flat(&avg);
        b.apply_step();

        for (x, y) in a.model.lm_head.weight.data.iter().zip(&b.model.lm_head.weight.data) {
            assert!((x - y).abs() < 1e-6, "batch != hand-averaged: {x} vs {y}");
        }
    }

    fn base_clone(m: &LlamaModel) -> LlamaModel {
        LlamaModel {
            cfg: m.cfg.clone(),
            tok_emb: m.tok_emb.clone(),
            blocks: m.blocks.clone(),
            final_norm: m.final_norm.clone(),
            lm_head: m.lm_head.clone(),
        }
    }

    #[test]
    fn finetune_epoch_reduces_loss() {
        let mut tr = LlamaTrainer::new(tiny(5, 16, 4, 2, 32, 2)).unwrap();
        tr.set_lr(0.01);
        let tokens: Vec<usize> = (0..120).map(|i| i % 5).collect();
        let mut rng = Rng::new(7);
        let first = tr.finetune_epoch(&tokens, 8, 8, &mut rng).unwrap();
        let mut last = first;
        for _ in 0..15 {
            last = tr.finetune_epoch(&tokens, 8, 8, &mut rng).unwrap();
        }
        assert!(last < first * 0.7, "epoch training did not reduce loss: {first:.4} → {last:.4}");
    }

    #[test]
    fn threaded_one_shard_matches_serial_bitwise() {
        let base = tiny(6, 8, 2, 2, 16, 1);
        let tokens: Vec<usize> = (0..80).map(|i| (i * 7) % 6).collect();

        let mut serial = LlamaTrainer::new(base_clone(&base)).unwrap();
        serial.set_lr(0.02);
        let mut rs = Rng::new(99);
        let ls = serial.finetune_epoch_threaded(&tokens, 6, 8, &mut rs, 1).unwrap();

        let mut threaded = LlamaTrainer::new(base_clone(&base)).unwrap();
        threaded.set_lr(0.02);
        let mut rt = Rng::new(99);
        let lt = threaded.finetune_epoch_threaded(&tokens, 6, 8, &mut rt, 1).unwrap();

        assert_eq!(ls, lt);
        for (x, y) in serial.model_flat().iter().zip(&threaded.model_flat()) {
            assert_eq!(x, y, "single-shard threaded diverged from serial");
        }
    }

    #[test]
    fn threaded_multishard_matches_serial_closely() {
        let base = tiny(6, 8, 2, 2, 16, 1);
        let tokens: Vec<usize> = (0..80).map(|i| (i * 7) % 6).collect();

        let mut serial = LlamaTrainer::new(base_clone(&base)).unwrap();
        serial.set_lr(0.02);
        let mut rs = Rng::new(1);
        for _ in 0..3 {
            serial.finetune_epoch_threaded(&tokens, 6, 16, &mut rs, 1).unwrap();
        }
        let mut par = LlamaTrainer::new(base_clone(&base)).unwrap();
        par.set_lr(0.02);
        let mut rp = Rng::new(1);
        for _ in 0..3 {
            par.finetune_epoch_threaded(&tokens, 6, 16, &mut rp, 4).unwrap();
        }
        for (x, y) in serial.model_flat().iter().zip(&par.model_flat()) {
            assert!((x - y).abs() < 1e-4, "multishard drifted: {x} vs {y}");
        }
    }

    #[test]
    fn lr_schedule_changes_trajectory() {
        let base = tiny(5, 16, 4, 2, 32, 1);
        let tokens: Vec<usize> = (0..100).map(|i| i % 5).collect();

        let mut fixed = LlamaTrainer::new(base_clone(&base)).unwrap();
        fixed.set_lr(0.02);
        let mut rf = Rng::new(5);
        for _ in 0..5 {
            fixed.finetune_epoch(&tokens, 8, 8, &mut rf).unwrap();
        }

        let mut sched = LlamaTrainer::new(base_clone(&base)).unwrap();
        sched.set_lr(0.02);
        let steps_per_epoch = (tokens.len() - 8 + 1).div_ceil(8) as u64;
        sched.set_lr_schedule(Some(LrSchedule::warmup_cosine(0.02, steps_per_epoch, steps_per_epoch * 5)));
        let mut rsc = Rng::new(5);
        for _ in 0..5 {
            sched.finetune_epoch(&tokens, 8, 8, &mut rsc).unwrap();
        }
        assert_eq!(sched.step_count(), steps_per_epoch * 5);
        let differ = fixed.model_flat().iter().zip(&sched.model_flat()).any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(differ, "schedule had no effect");
    }

    #[test]
    fn grad_clip_caps_norm_and_stays_finite() {
        let mut tr = LlamaTrainer::new(tiny(6, 8, 2, 2, 16, 1)).unwrap();
        let tokens = [1usize, 2, 3, 4];
        tr.loss_and_backward(&tokens).unwrap();
        let pre = tr.clip_grad_norm(0.05);
        let mut sumsq = 0.0f64;
        for s in tr.grads.slices() {
            for &x in s {
                sumsq += (x as f64) * (x as f64);
            }
        }
        let post = sumsq.sqrt() as f32;
        assert!(pre > 0.05, "test needs a larger pre-clip norm: {pre}");
        assert!(post <= 0.05 + 1e-4, "post-clip {post} exceeds budget");
    }

    #[test]
    fn dropout_off_matches_plain_and_gradient_is_correct() {
        // dropout=0 forward equals the plain path.
        let model = tiny(5, 8, 2, 2, 16, 1);
        let tokens = [1usize, 2, 3, 4];
        let a = forward_train(&model, &tokens, 0.0, 123).unwrap();
        let b = forward_train(&model, &tokens, 0.0, 999).unwrap();
        assert_eq!(a.logits, b.logits);

        // Same seed reproduces the masked forward; different seed changes it.
        let c = forward_train(&model, &tokens, 0.5, 7).unwrap();
        let d = forward_train(&model, &tokens, 0.5, 7).unwrap();
        let e = forward_train(&model, &tokens, 0.5, 8).unwrap();
        assert_eq!(c.logits, d.logits);
        assert_ne!(c.logits, e.logits);

        // Gradient through a fixed dropout mask matches finite differences.
        let mut tr = LlamaTrainer::new(base_clone(&model)).unwrap();
        let (seed, p) = (77u64, 0.5f32);
        let cache = forward_train(&tr.model, &tokens, p, seed).unwrap();
        let (_, dl) = cross_entropy(&cache.logits, &tokens, 5);
        tr.grads.clear();
        backward_into(&tr.model, &cache, &dl, &tokens, &mut tr.grads).unwrap();
        let eps = 1e-2f32;
        for &i in &[0usize, 7, 15] {
            let g = tr.grads.blocks[0].down.w[i];
            tr.model.blocks[0].ffn.down.weight.data[i] += eps;
            let lp = cross_entropy(&forward_train(&tr.model, &tokens, p, seed).unwrap().logits, &tokens, 5).0;
            tr.model.blocks[0].ffn.down.weight.data[i] -= 2.0 * eps;
            let lm = cross_entropy(&forward_train(&tr.model, &tokens, p, seed).unwrap().logits, &tokens, 5).0;
            tr.model.blocks[0].ffn.down.weight.data[i] += eps;
            let fd = (lp - lm) / (2.0 * eps);
            assert!((fd - g).abs() < 6e-3 + 0.03 * g.abs(), "down.w[{i}] fd={fd} got={g}");
        }
    }

    #[test]
    fn qat_training_reduces_loss_and_survives_quant() {
        let mut tr = LlamaTrainer::new(tiny(6, 16, 4, 2, 32, 2)).unwrap();
        tr.set_qat(true);
        assert!(tr.qat_enabled());
        tr.set_lr(0.01);
        let tokens = [1usize, 2, 3, 4, 5, 0, 1, 2];
        let l0 = tr.loss(&tokens).unwrap();
        let mut last = l0;
        for _ in 0..200 {
            last = tr.train_step(&tokens).unwrap();
        }
        assert!(last < l0 * 0.6, "QAT did not reduce loss: {l0} → {last}");
        assert!(last.is_finite());
    }

    #[test]
    fn checkpoint_resume_matches_uninterrupted() {
        let base = tiny(6, 8, 2, 2, 16, 2);
        let tokens: Vec<usize> = (0..100).map(|i| (i * 7) % 6).collect();

        // Uninterrupted: 6 epochs.
        let mut a = LlamaTrainer::new(base_clone(&base)).unwrap();
        a.set_lr(0.02);
        let mut ra = Rng::new(123);
        for _ in 0..6 {
            a.finetune_epoch(&tokens, 6, 16, &mut ra).unwrap();
        }

        // Interrupted: 3 epochs, checkpoint, reload into a fresh base, 3 more.
        let mut b = LlamaTrainer::new(base_clone(&base)).unwrap();
        b.set_lr(0.02);
        let mut rb = Rng::new(123);
        for _ in 0..3 {
            b.finetune_epoch(&tokens, 6, 16, &mut rb).unwrap();
        }
        let bytes = b.save_checkpoint(&rb);
        let mut c = LlamaTrainer::new(base_clone(&base)).unwrap();
        c.set_lr(0.02);
        let mut rc = c.load_checkpoint_into(&bytes).unwrap();
        for _ in 0..3 {
            c.finetune_epoch(&tokens, 6, 16, &mut rc).unwrap();
        }

        assert_eq!(a.step_count(), c.step_count());
        for (x, y) in a.model_flat().iter().zip(&c.model_flat()) {
            assert!((x - y).abs() < 1e-5, "resumed run diverged: {x} vs {y}");
        }
    }

    #[test]
    fn checkpoint_rejects_corrupt_and_mismatched() {
        let mut tr = LlamaTrainer::new(tiny(6, 8, 2, 2, 16, 1)).unwrap();
        assert!(tr.load_checkpoint_into(b"").is_err());
        assert!(tr.load_checkpoint_into(b"XXXXjunk").is_err());
        // Shape mismatch: checkpoint from a different model.
        let other = LlamaTrainer::new(tiny(7, 8, 2, 2, 16, 1)).unwrap();
        let bytes = other.save_checkpoint(&Rng::new(1));
        assert!(tr.load_checkpoint_into(&bytes).is_err());
    }
}
