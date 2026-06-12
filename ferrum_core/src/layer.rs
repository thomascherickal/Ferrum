//! The `Layer` trait and standard architectures: Linear, Activation, LayerNorm, Embedding, and TransformerBlock.
use crate::activation::Activation;
use crate::error::{InferError, Result};
use crate::ops;
use crate::tensor::Tensor;
use crate::verbose;
use std::any::Any;
use std::cell::RefCell;

/// Everything a layer must provide. `as_any` allows the loader to downcast
/// a trait object back to its concrete type for serialisation.
pub trait Layer {
    fn forward(&self, input: &Tensor) -> Result<Tensor>;
    fn name(&self) -> String;
    fn as_any(&self) -> &dyn Any;
}

// ─────────────────────────────────────────────────────────────────────────────
// Affine Linear Layer
// ─────────────────────────────────────────────────────────────────────────────

/// Fully-connected affine layer: y = x · W + b.
/// Weight shape: [in_features, out_features] — no transpose needed in forward.
pub struct Linear {
    pub weight: Tensor,
    pub bias: Tensor,
    in_f: usize,
    out_f: usize,
}

impl Linear {
    pub fn new(in_f: usize, out_f: usize, weight: Vec<f32>, bias: Vec<f32>) -> Result<Self> {
        if bias.len() != out_f {
            return Err(InferError::DimMismatch(format!(
                "bias length {} ≠ out_features {out_f}",
                bias.len()
            )));
        }
        vprintln!("[layer::Linear::new] in={}, out={}, weight_len={}, bias_len={}", in_f, out_f, weight.len(), bias.len());
        Ok(Self {
            weight: Tensor::matrix(in_f, out_f, weight)?,
            bias: Tensor::vector(bias),
            in_f,
            out_f,
        })
    }
    pub fn in_features(&self) -> usize {
        self.in_f
    }
    pub fn out_features(&self) -> usize {
        self.out_f
    }
}

impl Layer for Linear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (_, cols) = input.matrix_dims()?;
        if cols != self.in_f {
            return Err(InferError::DimMismatch(format!(
                "Linear expects width {}, got {cols}",
                self.in_f
            )));
        }
        vprintln!("[layer::Linear::forward] input={:?}, weight=[{},{}]", input.shape, self.in_f, self.out_f);
        let result = ops::add_bias(&ops::matmul(input, &self.weight)?, &self.bias)?;
        if verbose::is_verbose() {
            let (vmin, vmax, vmean) = verbose::stats(&result.data);
            vprintln!("[layer::Linear::forward]   output={:?}, stats: min={:.6}, max={:.6}, mean={:.6}", result.shape, vmin, vmax, vmean);
            verbose::check_nan_inf(&result.data, &format!("Linear({}→{}) output", self.in_f, self.out_f));
        }
        Ok(result)
    }
    fn name(&self) -> String {
        format!("Linear({}→{})", self.in_f, self.out_f)
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Activation Layer
// ─────────────────────────────────────────────────────────────────────────────

/// Parameter-free activation wrapper so it lives in the same layer list.
pub struct ActivationLayer {
    pub kind: Activation,
}

impl ActivationLayer {
    pub fn new(kind: Activation) -> Self {
        Self { kind }
    }
}

impl Layer for ActivationLayer {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.kind.apply(input)
    }
    fn name(&self) -> String {
        format!("Activation({:?})", self.kind)
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Flatten
// ─────────────────────────────────────────────────────────────────────────────

/// Flattens any input into a single row: [r, c] → [1, r·c].
///
/// Used between `Embedding` and `Linear` in embedded-MLP language models so
/// the per-position embeddings become one flat feature vector. Serialised in
/// FINF v5 as tag 5 with no payload.
#[derive(Default)]
pub struct Flatten;

impl Flatten {
    pub fn new() -> Self {
        Self
    }
}

impl Layer for Flatten {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        vprintln!("[layer::Flatten::forward] {:?} → [1, {}]", input.shape, input.data.len());
        Tensor::matrix(1, input.data.len(), input.data.clone())
    }
    fn name(&self) -> String {
        "Flatten".to_string()
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Layer Normalization
// ─────────────────────────────────────────────────────────────────────────────

/// Layer Normalization: y = (x - mean) / sqrt(var + eps) * gamma + beta.
pub struct LayerNorm {
    pub gamma: Tensor, // scale, shape [dim]
    pub beta: Tensor,  // shift, shape [dim]
    dim: usize,
}

impl LayerNorm {
    pub fn new(dim: usize, gamma: Vec<f32>, beta: Vec<f32>) -> Result<Self> {
        if gamma.len() != dim || beta.len() != dim {
            return Err(InferError::DimMismatch(format!(
                "LayerNorm weights len {}/{} != dim {}", gamma.len(), beta.len(), dim
            )));
        }
        vprintln!("[layer::LayerNorm::new] dim={}", dim);
        Ok(Self {
            gamma: Tensor::vector(gamma),
            beta: Tensor::vector(beta),
            dim,
        })
    }
    pub fn dim(&self) -> usize {
        self.dim
    }
}

impl Layer for LayerNorm {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (rows, cols) = input.matrix_dims()?;
        if cols != self.dim {
            return Err(InferError::DimMismatch(format!(
                "LayerNorm expects width {}, got {}", self.dim, cols
            )));
        }
        vprintln!("[layer::LayerNorm::forward] [{},{}]", rows, cols);
        let mut out = vec![0.0f32; rows * cols];
        for r in 0..rows {
            let base = r * cols;
            let row_slice = &input.data[base..base + cols];
            let mean = row_slice.iter().sum::<f32>() / cols as f32;
            let var = row_slice.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / cols as f32;
            let std = (var + 1e-5).sqrt();

            if verbose::is_verbose() && rows <= 8 {
                vprintln!("[layer::LayerNorm::forward]   row[{}]: mean={:.6}, var={:.6}, std={:.6}", r, mean, var, std);
            }

            for c in 0..cols {
                out[base + c] = ((row_slice[c] - mean) / std) * self.gamma.data[c] + self.beta.data[c];
            }
        }
        if verbose::is_verbose() {
            verbose::check_nan_inf(&out, "LayerNorm output");
        }
        Tensor::matrix(rows, cols, out)
    }
    fn name(&self) -> String {
        format!("LayerNorm({})", self.dim)
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Embedding Layer (Token + Position)
// ─────────────────────────────────────────────────────────────────────────────

/// Embedding table mapping token IDs and positional indices to continuous vectors.
pub struct Embedding {
    pub token_weight: Tensor, // Shape [vocab_size, embedding_dim]
    pub pos_weight: Tensor,   // Shape [max_seq_len, embedding_dim]
    vocab_size: usize,
    max_seq_len: usize,
    embedding_dim: usize,
}

impl Embedding {
    pub fn new(
        vocab_size: usize,
        max_seq_len: usize,
        embedding_dim: usize,
        token_weight: Vec<f32>,
        pos_weight: Vec<f32>,
    ) -> Result<Self> {
        if token_weight.len() != vocab_size * embedding_dim {
            return Err(InferError::ShapeMismatch { expected: vocab_size * embedding_dim, got: token_weight.len() });
        }
        if pos_weight.len() != max_seq_len * embedding_dim {
            return Err(InferError::ShapeMismatch { expected: max_seq_len * embedding_dim, got: pos_weight.len() });
        }
        vprintln!("[layer::Embedding::new] vocab={}, max_seq={}, dim={}", vocab_size, max_seq_len, embedding_dim);
        Ok(Self {
            token_weight: Tensor::matrix(vocab_size, embedding_dim, token_weight)?,
            pos_weight: Tensor::matrix(max_seq_len, embedding_dim, pos_weight)?,
            vocab_size,
            max_seq_len,
            embedding_dim,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Embed a single token at a given sequence position → [1, embedding_dim].
    /// Used by the incremental (KV-cached) generation path.
    pub fn embed_one(&self, token: usize, pos: usize) -> Result<Tensor> {
        if token >= self.vocab_size {
            return Err(InferError::DimMismatch(format!(
                "Token index {} out of bounds for vocab_size {}", token, self.vocab_size
            )));
        }
        if pos >= self.max_seq_len {
            return Err(InferError::DimMismatch(format!(
                "Position {} exceeds max_seq_len {}", pos, self.max_seq_len
            )));
        }
        let d = self.embedding_dim;
        let tok_base = token * d;
        let pos_base = pos * d;
        let data: Vec<f32> = (0..d)
            .map(|i| self.token_weight.data[tok_base + i] + self.pos_weight.data[pos_base + i])
            .collect();
        Tensor::matrix(1, d, data)
    }
}

impl Layer for Embedding {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Input has shape [B, T] (where T is sequence length)
        let (batch, seq_len) = match input.shape.as_slice() {
            [s] => (1, *s),
            [b, s] => (*b, *s),
            _ => return Err(InferError::DimMismatch("Embedding expects rank-1 or rank-2 inputs".into())),
        };
        if seq_len > self.max_seq_len {
            return Err(InferError::DimMismatch(format!(
                "Sequence length {} exceeds max_seq_len {}", seq_len, self.max_seq_len
            )));
        }
        vprintln!("[layer::Embedding::forward] batch={}, seq_len={}, vocab={}, dim={}",
            batch, seq_len, self.vocab_size, self.embedding_dim);

        let out_cols = self.embedding_dim;
        let mut out_data = vec![0.0f32; batch * seq_len * out_cols];
        for b in 0..batch {
            for t in 0..seq_len {
                let tok_idx = input.data[b * seq_len + t].round() as usize;
                if tok_idx >= self.vocab_size {
                    return Err(InferError::DimMismatch(format!(
                        "Token index {} out of bounds for vocab_size {}", tok_idx, self.vocab_size
                    )));
                }
                let tok_base = tok_idx * self.embedding_dim;
                let pos_base = t * self.embedding_dim;
                let out_base = (b * seq_len + t) * self.embedding_dim;
                for d in 0..self.embedding_dim {
                    out_data[out_base + d] = self.token_weight.data[tok_base + d] + self.pos_weight.data[pos_base + d];
                }
            }
        }
        if verbose::is_verbose() {
            let (vmin, vmax, vmean) = verbose::stats(&out_data);
            vprintln!("[layer::Embedding::forward]   output=[{},{}], stats: min={:.6}, max={:.6}, mean={:.6}",
                batch * seq_len, out_cols, vmin, vmax, vmean);
        }
        Tensor::matrix(batch * seq_len, out_cols, out_data)
    }
    fn name(&self) -> String {
        format!("Embedding(vocab={}, dim={})", self.vocab_size, self.embedding_dim)
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// KV Cache (incremental generation)
// ─────────────────────────────────────────────────────────────────────────────

/// Per-block key/value cache for token-at-a-time generation. Holding the K/V
/// rows of already-processed positions turns each new-token forward pass from
/// O(T²) into O(T).
pub struct KvCache {
    k: Vec<f32>, // [len, dim] row-major
    v: Vec<f32>,
    len: usize,
    capacity: usize,
    dim: usize,
}

impl KvCache {
    /// `capacity` should be the block's `context_len`; `dim` its embedding dim.
    pub fn new(capacity: usize, dim: usize) -> Self {
        Self {
            k: Vec::with_capacity(capacity * dim),
            v: Vec::with_capacity(capacity * dim),
            len: 0,
            capacity,
            dim,
        }
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn is_full(&self) -> bool {
        self.len >= self.capacity
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    /// Drop all cached positions (start a fresh sequence).
    pub fn clear(&mut self) {
        self.k.clear();
        self.v.clear();
        self.len = 0;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Transformer Block Layer
// ─────────────────────────────────────────────────────────────────────────────

/// A Decoder-Only Causal Transformer block containing:
///   LayerNorm -> Causal Multi-Head Self-Attention -> Residual
///   -> LayerNorm -> Feed-Forward Network -> Residual
pub struct TransformerBlock {
    pub ln1: LayerNorm,
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub out_proj: Linear,
    pub ln2: LayerNorm,
    pub ffn1: Linear,
    pub ffn2: Linear,
    context_len: usize,
    num_heads: usize,
    embedding_dim: usize,
    pub last_attention: RefCell<Vec<f32>>,
}

impl TransformerBlock {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        context_len: usize,
        num_heads: usize,
        embedding_dim: usize,
        ln1_g: Vec<f32>, ln1_b: Vec<f32>,
        q_w: Vec<f32>, q_b: Vec<f32>,
        k_w: Vec<f32>, k_b: Vec<f32>,
        v_w: Vec<f32>, v_b: Vec<f32>,
        out_w: Vec<f32>, out_b: Vec<f32>,
        ln2_g: Vec<f32>, ln2_b: Vec<f32>,
        ffn1_w: Vec<f32>, ffn1_b: Vec<f32>,
        ffn2_w: Vec<f32>, ffn2_b: Vec<f32>,
    ) -> Result<Self> {
        if context_len == 0 {
            return Err(InferError::DimMismatch("context_len must be > 0".into()));
        }
        if num_heads == 0 || embedding_dim % num_heads != 0 {
            return Err(InferError::DimMismatch(format!(
                "embedding_dim {embedding_dim} must be divisible by num_heads {num_heads}"
            )));
        }
        vprintln!("[layer::TransformerBlock::new] ctx={}, heads={}, dim={}, hidden={}",
            context_len, num_heads, embedding_dim, ffn1_b.len());

        let ln1 = LayerNorm::new(embedding_dim, ln1_g, ln1_b)?;
        let q_proj = Linear::new(embedding_dim, embedding_dim, q_w, q_b)?;
        let k_proj = Linear::new(embedding_dim, embedding_dim, k_w, k_b)?;
        let v_proj = Linear::new(embedding_dim, embedding_dim, v_w, v_b)?;
        let out_proj = Linear::new(embedding_dim, embedding_dim, out_w, out_b)?;
        let ln2 = LayerNorm::new(embedding_dim, ln2_g, ln2_b)?;
        let hidden_dim = ffn1_b.len();
        let ffn1 = Linear::new(embedding_dim, hidden_dim, ffn1_w, ffn1_b)?;
        let ffn2 = Linear::new(hidden_dim, embedding_dim, ffn2_w, ffn2_b)?;

        Ok(Self {
            ln1,
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            ln2,
            ffn1,
            ffn2,
            context_len,
            num_heads,
            embedding_dim,
            last_attention: RefCell::new(Vec::new()),
        })
    }

    pub fn context_len(&self) -> usize {
        self.context_len
    }
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
    pub fn hidden_dim(&self) -> usize {
        self.ffn1.out_features()
    }

    /// Incremental forward pass for one new token using a KV cache.
    ///
    /// `x` is the embedding of the newest position, shape [1, embedding_dim].
    /// The new K/V rows are appended to `cache` and the query attends over the
    /// whole cached prefix (causality holds automatically — only past
    /// positions are cached). Returns the block output for this position,
    /// shape [1, embedding_dim]. Produces the same values as a full
    /// `forward` over the sequence, but in O(T) per token instead of O(T²).
    ///
    /// Errors if the cache is full (`context_len` positions reached) — call
    /// `cache.clear()` and re-prime with a fresh context to continue.
    pub fn forward_with_cache(&self, x: &Tensor, cache: &mut KvCache) -> Result<Tensor> {
        let (rows, c) = x.matrix_dims()?;
        if rows != 1 || c != self.embedding_dim {
            return Err(InferError::DimMismatch(format!(
                "forward_with_cache expects [1,{}], got {:?}", self.embedding_dim, x.shape
            )));
        }
        if cache.dim != c {
            return Err(InferError::DimMismatch(format!(
                "KvCache dim {} ≠ block embedding_dim {}", cache.dim, c
            )));
        }
        if cache.is_full() {
            return Err(InferError::DimMismatch(format!(
                "KvCache full ({} positions): clear and re-prime to continue",
                cache.capacity
            )));
        }

        let norm1 = self.ln1.forward(x)?;
        let q = self.q_proj.forward(&norm1)?;
        let k = self.k_proj.forward(&norm1)?;
        let v = self.v_proj.forward(&norm1)?;
        cache.k.extend_from_slice(&k.data);
        cache.v.extend_from_slice(&v.data);
        cache.len += 1;

        let l = cache.len; // positions visible to the query (incl. itself)
        let heads = self.num_heads;
        let head_dim = c / heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut attn_out = vec![0.0f32; c];
        let mut scores = vec![0.0f32; l];
        for h in 0..heads {
            let hs = h * head_dim;
            // scores[j] = scale · q_h · k_h(j)
            for (j, s) in scores.iter_mut().enumerate() {
                let kb = j * c + hs;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q.data[hs + d] * cache.k[kb + d];
                }
                *s = dot * scale;
            }
            // softmax over the cached prefix
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - max).exp();
                sum += *s;
            }
            // out_h = Σ_j p_j · v_h(j)
            for (j, &s) in scores.iter().enumerate() {
                let p = s / sum;
                let vb = j * c + hs;
                for d in 0..head_dim {
                    attn_out[hs + d] += p * cache.v[vb + d];
                }
            }
        }

        let projected = self.out_proj.forward(&Tensor::matrix(1, c, attn_out)?)?;
        let x_attn: Vec<f32> = (0..c).map(|i| x.data[i] + projected.data[i]).collect();
        let x_attn = Tensor::matrix(1, c, x_attn)?;

        let norm2 = self.ln2.forward(&x_attn)?;
        let hidden = self.ffn1.forward(&norm2)?.map(|v| v.max(0.0));
        let ff2 = self.ffn2.forward(&hidden)?;
        let out: Vec<f32> = (0..c).map(|i| x_attn.data[i] + ff2.data[i]).collect();
        Tensor::matrix(1, c, out)
    }
}

impl Layer for TransformerBlock {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (m, c) = input.matrix_dims()?;
        if c != self.embedding_dim {
            return Err(InferError::DimMismatch(format!(
                "TransformerBlock expects dim {}, got {}", self.embedding_dim, c
            )));
        }
        let t = self.context_len;
        if m % t != 0 {
            return Err(InferError::DimMismatch(format!(
                "Input rows {} must be divisible by context_len {}", m, t
            )));
        }
        let b = m / t;

        vprintln!("[layer::TransformerBlock::forward] input=[{},{}], batch={}, ctx={}, heads={}, dim={}",
            m, c, b, t, self.num_heads, self.embedding_dim);

        // ── 1. LayerNorm 1 ───────────────────────────────────────────────────────
        vprintln!("[layer::TransformerBlock::forward]   ┌─ LayerNorm1");
        let norm1 = self.ln1.forward(input)?;
        if verbose::is_verbose() {
            let (vmin, vmax, vmean) = verbose::stats(&norm1.data);
            vprintln!("[layer::TransformerBlock::forward]   │  LN1 output: min={:.6}, max={:.6}, mean={:.6}", vmin, vmax, vmean);
            verbose::check_nan_inf(&norm1.data, "TransformerBlock LN1");
        }

        // ── 2. Q, K, V Projections ────────────────────────────────────────────────
        vprintln!("[layer::TransformerBlock::forward]   ├─ Q/K/V projections");
        let q = self.q_proj.forward(&norm1)?;
        let k = self.k_proj.forward(&norm1)?;
        let v = self.v_proj.forward(&norm1)?;
        if verbose::is_verbose() {
            let (qmin, qmax, qmean) = verbose::stats(&q.data);
            let (kmin, kmax, kmean) = verbose::stats(&k.data);
            let (vmin, vmax, vmean) = verbose::stats(&v.data);
            vprintln!("[layer::TransformerBlock::forward]   │  Q: min={:.6}, max={:.6}, mean={:.6}", qmin, qmax, qmean);
            vprintln!("[layer::TransformerBlock::forward]   │  K: min={:.6}, max={:.6}, mean={:.6}", kmin, kmax, kmean);
            vprintln!("[layer::TransformerBlock::forward]   │  V: min={:.6}, max={:.6}, mean={:.6}", vmin, vmax, vmean);
            verbose::check_nan_inf(&q.data, "TransformerBlock Q");
            verbose::check_nan_inf(&k.data, "TransformerBlock K");
            verbose::check_nan_inf(&v.data, "TransformerBlock V");
        }

        // ── 3. Multi-Head Attention ──────────────────────────────────────────────
        vprintln!("[layer::TransformerBlock::forward]   ├─ Multi-Head Attention ({} heads, head_dim={})",
            self.num_heads, self.embedding_dim / self.num_heads);
        let num_heads = self.num_heads;
        let head_dim = self.embedding_dim / num_heads;
        let head_scale = 1.0 / (head_dim as f32).sqrt();

        let mut attn_out = vec![0.0f32; m * c];
        let mut all_attns = vec![0.0f32; b * num_heads * t * t];

        for batch_idx in 0..b {
            for head_idx in 0..num_heads {
                // Extract Q, K, V for this head and batch item
                let mut q_head = vec![0.0f32; t * head_dim];
                let mut k_head = vec![0.0f32; t * head_dim];
                let mut v_head = vec![0.0f32; t * head_dim];

                for r in 0..t {
                    let src_row = batch_idx * t + r;
                    let head_start = head_idx * head_dim;
                    let src_idx = src_row * self.embedding_dim + head_start;
                    let dst_idx = r * head_dim;
                    q_head[dst_idx..dst_idx + head_dim].copy_from_slice(&q.data[src_idx..src_idx + head_dim]);
                    k_head[dst_idx..dst_idx + head_dim].copy_from_slice(&k.data[src_idx..src_idx + head_dim]);
                    v_head[dst_idx..dst_idx + head_dim].copy_from_slice(&v.data[src_idx..src_idx + head_dim]);
                }

                // Compute causal self-attention scores: S = Q * K^T
                let mut s = matmul_transpose_b_helper(&q_head, &k_head, t, t, head_dim);

                // Scale and apply causal mask
                for i in 0..t {
                    for j in 0..t {
                        let idx = i * t + j;
                        s[idx] *= head_scale;
                        if j > i {
                            s[idx] = -1e9; // Causal mask
                        }
                    }
                }

                // Softmax row-wise
                for i in 0..t {
                    let base = i * t;
                    let row = &mut s[base..base + t];
                    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0f32;
                    for val in row.iter_mut() {
                        let e = (*val - max).exp();
                        *val = e;
                        sum += e;
                    }
                    for val in row.iter_mut() {
                        *val /= sum;
                    }
                }

                if verbose::is_verbose() {
                    let (amin, amax, amean) = verbose::stats(&s);
                    vprintln!("[layer::TransformerBlock::forward]   │  batch[{}] head[{}]: attn_weights: min={:.6}, max={:.6}, mean={:.6}",
                        batch_idx, head_idx, amin, amax, amean);
                    verbose::check_nan_inf(&s, &format!("TransformerBlock attn_weights batch={} head={}", batch_idx, head_idx));
                }

                // Store attention weights for visualization
                let attn_store_base = (batch_idx * num_heads + head_idx) * t * t;
                all_attns[attn_store_base..attn_store_base + t * t].copy_from_slice(&s);

                // Compute output of head: O = S * V
                let o = matmul_naive_helper(&s, &v_head, t, head_dim, t);

                // Copy output back to flat output matrix
                for r in 0..t {
                    let dst_row = batch_idx * t + r;
                    let head_start = head_idx * head_dim;
                    let dst_idx = dst_row * self.embedding_dim + head_start;
                    let src_idx = r * head_dim;
                    attn_out[dst_idx..dst_idx + head_dim].copy_from_slice(&o[src_idx..src_idx + head_dim]);
                }
            }
        }

        // Store attention maps
        self.last_attention.replace(all_attns);

        let attn_out_tensor = Tensor::matrix(m, c, attn_out)?;
        vprintln!("[layer::TransformerBlock::forward]   ├─ Output projection");
        let projected = self.out_proj.forward(&attn_out_tensor)?;

        // Residual connection
        vprintln!("[layer::TransformerBlock::forward]   ├─ Residual connection 1 (input + attention)");
        let mut x_attn_data = vec![0.0f32; m * c];
        for i in 0..m * c {
            x_attn_data[i] = input.data[i] + projected.data[i];
        }
        let x_attn = Tensor::matrix(m, c, x_attn_data)?;
        if verbose::is_verbose() {
            let (vmin, vmax, vmean) = verbose::stats(&x_attn.data);
            vprintln!("[layer::TransformerBlock::forward]   │  post-residual-1: min={:.6}, max={:.6}, mean={:.6}", vmin, vmax, vmean);
            verbose::check_nan_inf(&x_attn.data, "TransformerBlock post-residual-1");
        }

        // ── 4. LayerNorm 2 & FFN ─────────────────────────────────────────────────
        vprintln!("[layer::TransformerBlock::forward]   ├─ LayerNorm2");
        let norm2 = self.ln2.forward(&x_attn)?;

        vprintln!("[layer::TransformerBlock::forward]   ├─ FFN (in → hidden → out)");
        let ff1 = self.ffn1.forward(&norm2)?;
        // ReLU
        let ffn_hidden = ff1.map(|x| x.max(0.0));
        if verbose::is_verbose() {
            let zeros = ffn_hidden.data.iter().filter(|&&v| v == 0.0).count();
            vprintln!("[layer::TransformerBlock::forward]   │  FFN ReLU: {}/{} zeros ({:.1}% dead)",
                zeros, ffn_hidden.data.len(), 100.0 * zeros as f32 / ffn_hidden.data.len() as f32);
        }
        let ff2 = self.ffn2.forward(&ffn_hidden)?;

        // Residual connection
        vprintln!("[layer::TransformerBlock::forward]   └─ Residual connection 2 (attn + ffn)");
        let mut out_data = vec![0.0f32; m * c];
        for i in 0..m * c {
            out_data[i] = x_attn.data[i] + ff2.data[i];
        }
        if verbose::is_verbose() {
            let (vmin, vmax, vmean) = verbose::stats(&out_data);
            vprintln!("[layer::TransformerBlock::forward]   Final output: min={:.6}, max={:.6}, mean={:.6}", vmin, vmax, vmean);
            verbose::check_nan_inf(&out_data, "TransformerBlock final output");
        }
        Tensor::matrix(m, c, out_data)
    }

    fn name(&self) -> String {
        format!(
            "TransformerBlock(heads={}, dim={}, hidden={})",
            self.num_heads, self.embedding_dim, self.hidden_dim()
        )
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

pub(crate) fn matmul_transpose_b_helper(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    crate::parallel::for_row_blocks(m, n, m * n * k, &mut out, |row0, block| {
        let rows = block.len() / n;
        for li in 0..rows {
            let a_row = (row0 + li) * k;
            let o_row = li * n;
            for j in 0..n {
                let b_row = j * k;
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a[a_row + p] * b[b_row + p];
                }
                block[o_row + j] = sum;
            }
        }
    });
    out
}

pub(crate) fn matmul_naive_helper(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    crate::parallel::for_row_blocks(m, n, m * n * k, &mut out, |row0, block| {
        let rows = block.len() / n;
        for li in 0..rows {
            let a_row = (row0 + li) * k;
            let o_row = li * n;
            for p in 0..k {
                let a_ip = a[a_row + p];
                let b_row = p * n;
                for j in 0..n {
                    block[o_row + j] += a_ip * b[b_row + j];
                }
            }
        }
    });
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ───────────────────────────────────────────────────────────────

    fn identity_linear(n: usize) -> Linear {
        let mut w = vec![0.0f32; n * n];
        for i in 0..n { w[i * n + i] = 1.0; }
        Linear::new(n, n, w, vec![0.0; n]).unwrap()
    }

    fn minimal_block(ctx: usize, heads: usize, dim: usize, hidden: usize) -> TransformerBlock {
        let scale = 0.01f32;
        let c = dim;
        let h = hidden;
        TransformerBlock::new(
            ctx, heads, dim,
            vec![1.0; c], vec![0.0; c],       // ln1
            vec![scale; c*c], vec![0.0; c],   // q
            vec![scale; c*c], vec![0.0; c],   // k
            vec![scale; c*c], vec![0.0; c],   // v
            vec![scale; c*c], vec![0.0; c],   // out
            vec![1.0; c], vec![0.0; c],       // ln2
            vec![scale; c*h], vec![0.0; h],   // ffn1
            vec![scale; h*c], vec![0.0; c],   // ffn2
        ).unwrap()
    }

    // ── Linear ───────────────────────────────────────────────────────────────

    #[test]
    fn linear_identity_transform() {
        let l = identity_linear(3);
        let x = Tensor::row(vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(l.forward(&x).unwrap().data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn linear_applies_bias() {
        let l = Linear::new(2, 2, vec![1.0, 0.0, 0.0, 1.0], vec![10.0, 20.0]).unwrap();
        let x = Tensor::row(vec![0.0, 0.0]).unwrap();
        assert_eq!(l.forward(&x).unwrap().data, vec![10.0, 20.0]);
    }

    #[test]
    fn linear_new_wrong_bias_len_errors() {
        let result = Linear::new(2, 3, vec![0.0; 6], vec![0.0; 2]); // bias should be 3
        assert!(matches!(result, Err(InferError::DimMismatch(_))));
    }

    #[test]
    fn linear_forward_wrong_input_width_errors() {
        let l = identity_linear(3);
        let x = Tensor::row(vec![1.0, 2.0]).unwrap(); // width 2, expects 3
        assert!(matches!(l.forward(&x), Err(InferError::DimMismatch(_))));
    }

    #[test]
    fn linear_accessors() {
        let l = Linear::new(4, 8, vec![0.0; 32], vec![0.0; 8]).unwrap();
        assert_eq!(l.in_features(), 4);
        assert_eq!(l.out_features(), 8);
    }

    #[test]
    fn linear_name_contains_dims() {
        let l = Linear::new(4, 8, vec![0.0; 32], vec![0.0; 8]).unwrap();
        let name = l.name();
        assert!(name.contains("4") && name.contains("8"));
    }

    #[test]
    fn linear_batch_forward_shape() {
        let l = Linear::new(3, 2, vec![1.0; 6], vec![0.0; 2]).unwrap();
        let x = Tensor::matrix(5, 3, vec![1.0; 15]).unwrap();
        let y = l.forward(&x).unwrap();
        assert_eq!(y.shape, vec![5, 2]);
    }

    // ── ActivationLayer ───────────────────────────────────────────────────────

    #[test]
    fn activation_layer_relu_name() {
        let a = ActivationLayer::new(Activation::ReLU);
        assert!(a.name().contains("ReLU"));
    }

    #[test]
    fn activation_layer_forward_delegates() {
        let a = ActivationLayer::new(Activation::ReLU);
        let x = Tensor::vector(vec![-1.0, 2.0]);
        let y = a.forward(&x).unwrap();
        assert_eq!(y.data, vec![0.0, 2.0]);
    }

    // ── LayerNorm ─────────────────────────────────────────────────────────────

    #[test]
    fn layernorm_preserves_mean_and_var() {
        let x = Tensor::matrix(2, 4, vec![
            1.0, 2.0, 3.0, 4.0,
            10.0, -10.0, 20.0, -20.0,
        ]).unwrap();
        let ln = LayerNorm::new(4, vec![1.0; 4], vec![0.0; 4]).unwrap();
        let y = ln.forward(&x).unwrap();
        assert_eq!(y.shape, vec![2, 4]);
        for r in 0..2 {
            let row = &y.data[r * 4..(r + 1) * 4];
            let mean = row.iter().sum::<f32>() / 4.0;
            let var = row.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / 4.0;
            assert!(mean.abs() < 1e-4, "row {r} mean={mean}");
            assert!((var - 1.0).abs() < 1e-3, "row {r} var={var}");
        }
    }

    #[test]
    fn layernorm_gamma_beta_applied() {
        // gamma=2, beta=1 on a single-row [0, 0, 0, 0] → all outputs should be beta=1
        // Actually zero-variance input: normalised = 0, output = 0*gamma + beta = beta
        let x = Tensor::matrix(1, 4, vec![5.0; 4]).unwrap(); // constant row → std≈0, normalised≈0
        let ln = LayerNorm::new(4, vec![2.0; 4], vec![1.0; 4]).unwrap();
        let y = ln.forward(&x).unwrap();
        // Each output should be near beta=1 (normalised×gamma=0)
        for &v in &y.data { assert!((v - 1.0).abs() < 1e-3, "got {v}"); }
    }

    #[test]
    fn layernorm_new_wrong_gamma_len_errors() {
        assert!(LayerNorm::new(4, vec![1.0; 3], vec![0.0; 4]).is_err());
    }

    #[test]
    fn layernorm_new_wrong_beta_len_errors() {
        assert!(LayerNorm::new(4, vec![1.0; 4], vec![0.0; 3]).is_err());
    }

    #[test]
    fn layernorm_forward_wrong_width_errors() {
        let ln = LayerNorm::new(4, vec![1.0; 4], vec![0.0; 4]).unwrap();
        let x = Tensor::matrix(1, 3, vec![1.0; 3]).unwrap();
        assert!(matches!(ln.forward(&x), Err(InferError::DimMismatch(_))));
    }

    #[test]
    fn layernorm_dim_accessor() {
        let ln = LayerNorm::new(8, vec![1.0; 8], vec![0.0; 8]).unwrap();
        assert_eq!(ln.dim(), 8);
    }

    #[test]
    fn layernorm_name() {
        let ln = LayerNorm::new(6, vec![1.0; 6], vec![0.0; 6]).unwrap();
        assert!(ln.name().contains("6"));
    }

    // ── Embedding ─────────────────────────────────────────────────────────────

    #[test]
    fn embedding_lookup_and_positional_addition() {
        let emb = Embedding::new(
            3, 4, 2,
            vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0],
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        ).unwrap();
        let x = Tensor::matrix(2, 3, vec![0.0, 2.0, 1.0, 1.0, 0.0, 2.0]).unwrap();
        let y = emb.forward(&x).unwrap();
        assert_eq!(y.shape, vec![6, 2]);
        // seq 0, pos 0, tok 0: token[0]=[1,10] + pos[0]=[0.1,0.2] = [1.1, 10.2]
        assert!((y.data[0] - 1.1).abs() < 1e-5);
        assert!((y.data[1] - 10.2).abs() < 1e-5);
    }

    #[test]
    fn embedding_rank1_input_accepted() {
        let emb = Embedding::new(4, 8, 2,
            vec![0.0; 8], vec![0.0; 16]).unwrap();
        let x = Tensor::vector(vec![0.0, 1.0, 2.0]);
        // rank-1 = single sequence of 3 tokens
        assert!(emb.forward(&x).is_ok());
    }

    #[test]
    fn embedding_new_wrong_token_weight_len_errors() {
        assert!(Embedding::new(3, 4, 2, vec![0.0; 5], vec![0.0; 8]).is_err());
    }

    #[test]
    fn embedding_new_wrong_pos_weight_len_errors() {
        assert!(Embedding::new(3, 4, 2, vec![0.0; 6], vec![0.0; 7]).is_err());
    }

    #[test]
    fn embedding_seq_exceeds_max_seq_len_errors() {
        let emb = Embedding::new(5, 4, 2, vec![0.0; 10], vec![0.0; 8]).unwrap();
        // seq_len=6 > max_seq_len=4
        let x = Tensor::matrix(1, 6, vec![0.0; 6]).unwrap();
        assert!(matches!(emb.forward(&x), Err(InferError::DimMismatch(_))));
    }

    #[test]
    fn embedding_oob_token_index_errors() {
        let emb = Embedding::new(3, 4, 2, vec![0.0; 6], vec![0.0; 8]).unwrap();
        let x = Tensor::matrix(1, 2, vec![0.0, 9.0]).unwrap(); // token 9 ≥ vocab_size=3
        assert!(matches!(emb.forward(&x), Err(InferError::DimMismatch(_))));
    }

    #[test]
    fn embedding_rank3_input_errors() {
        let emb = Embedding::new(5, 4, 2, vec![0.0; 10], vec![0.0; 8]).unwrap();
        let x = Tensor::new(vec![1, 2, 2], vec![0.0; 4]).unwrap();
        assert!(emb.forward(&x).is_err());
    }

    #[test]
    fn embedding_accessors() {
        let emb = Embedding::new(10, 32, 64, vec![0.0; 640], vec![0.0; 2048]).unwrap();
        assert_eq!(emb.vocab_size(), 10);
        assert_eq!(emb.max_seq_len(), 32);
        assert_eq!(emb.embedding_dim(), 64);
    }

    #[test]
    fn embedding_name() {
        let emb = Embedding::new(5, 8, 16, vec![0.0; 80], vec![0.0; 128]).unwrap();
        let name = emb.name();
        assert!(name.contains("vocab=5") && name.contains("dim=16"));
    }

    // ── TransformerBlock ──────────────────────────────────────────────────────

    #[test]
    fn transformer_block_forward_output_shape() {
        // ctx=4, heads=2, dim=8, hidden=16
        let block = minimal_block(4, 2, 8, 16);
        // Input: [B*T, dim] = [1*4, 8]
        let x = Tensor::matrix(4, 8, vec![0.1f32; 32]).unwrap();
        let y = block.forward(&x).unwrap();
        assert_eq!(y.shape, vec![4, 8]);
    }

    #[test]
    fn transformer_block_forward_batch_shape() {
        // batch=2, ctx=4, dim=8 → input [8, 8]
        let block = minimal_block(4, 2, 8, 16);
        let x = Tensor::matrix(8, 8, vec![0.05f32; 64]).unwrap();
        let y = block.forward(&x).unwrap();
        assert_eq!(y.shape, vec![8, 8]);
    }

    #[test]
    fn transformer_block_output_is_finite() {
        let block = minimal_block(4, 2, 8, 16);
        let x = Tensor::matrix(4, 8, (0..32).map(|i| i as f32 * 0.01).collect()).unwrap();
        let y = block.forward(&x).unwrap();
        assert!(y.data.iter().all(|v| v.is_finite()), "NaN/Inf in output");
    }

    #[test]
    fn transformer_block_wrong_embedding_dim_errors() {
        let block = minimal_block(4, 2, 8, 16);
        // width 5 instead of 8
        let x = Tensor::matrix(4, 5, vec![0.0; 20]).unwrap();
        assert!(matches!(block.forward(&x), Err(InferError::DimMismatch(_))));
    }

    #[test]
    fn transformer_block_rows_not_divisible_by_context_len_errors() {
        let block = minimal_block(4, 2, 8, 16);
        // 6 rows, context_len=4: 6 % 4 ≠ 0
        let x = Tensor::matrix(6, 8, vec![0.0; 48]).unwrap();
        assert!(matches!(block.forward(&x), Err(InferError::DimMismatch(_))));
    }

    #[test]
    fn transformer_block_attention_stored_after_forward() {
        let block = minimal_block(4, 2, 8, 16);
        let x = Tensor::matrix(4, 8, vec![0.1f32; 32]).unwrap();
        block.forward(&x).unwrap();
        let attn = block.last_attention.borrow();
        // Shape: [batch=1, heads=2, T=4, T=4] = 32 elements
        assert_eq!(attn.len(), 1 * 2 * 4 * 4);
    }

    #[test]
    fn transformer_block_causal_mask_enforced() {
        // Causal mask: attn[i, j] should be 0 for j > i.
        let block = minimal_block(4, 1, 8, 16);
        let x = Tensor::matrix(4, 8, (0..32).map(|i| (i as f32) * 0.1).collect()).unwrap();
        block.forward(&x).unwrap();
        let attn = block.last_attention.borrow();
        let t = 4;
        for qi in 0..t {
            for ki in 0..t {
                if ki > qi {
                    // Future tokens must have zero attention weight
                    assert!(
                        attn[qi * t + ki] < 1e-6,
                        "attn[{qi},{ki}]={} should be ~0 (causal mask)",
                        attn[qi * t + ki]
                    );
                }
            }
        }
    }

    #[test]
    fn transformer_block_attention_rows_sum_to_one() {
        let block = minimal_block(4, 2, 8, 16);
        let x = Tensor::matrix(4, 8, vec![0.3f32; 32]).unwrap();
        block.forward(&x).unwrap();
        let attn = block.last_attention.borrow();
        // Each query row of the attention matrix should sum to 1 (softmax)
        let t = 4;
        let heads = 2;
        for h in 0..heads {
            for qi in 0..t {
                let base = (h * t + qi) * t;
                let row_sum: f32 = attn[base..base + t].iter().sum();
                assert!((row_sum - 1.0).abs() < 1e-5,
                    "head {h} row {qi}: attn sum = {row_sum}");
            }
        }
    }

    #[test]
    fn transformer_block_accessors() {
        let block = minimal_block(8, 4, 16, 64);
        assert_eq!(block.context_len(), 8);
        assert_eq!(block.num_heads(), 4);
        assert_eq!(block.embedding_dim(), 16);
        assert_eq!(block.hidden_dim(), 64);
    }

    #[test]
    fn transformer_block_name_contains_config() {
        let block = minimal_block(4, 2, 8, 32);
        let name = block.name();
        assert!(name.contains("heads=2") && name.contains("dim=8"));
    }

    #[test]
    fn kv_cache_matches_full_forward() {
        // Token-at-a-time generation through the cache must reproduce the
        // full-sequence forward pass row for row.
        use crate::rng::Rng;
        let (t, heads, dim, hidden) = (6, 2, 8, 16);
        let mut rng = Rng::new(11);
        let mut randn = |n: usize| -> Vec<f32> {
            (0..n).map(|_| rng.next_normal() * 0.2).collect()
        };
        let block = TransformerBlock::new(
            t, heads, dim,
            vec![1.0; dim], vec![0.0; dim],
            randn(dim * dim), randn(dim),
            randn(dim * dim), randn(dim),
            randn(dim * dim), randn(dim),
            randn(dim * dim), randn(dim),
            vec![1.0; dim], vec![0.0; dim],
            randn(dim * hidden), randn(hidden),
            randn(hidden * dim), randn(dim),
        ).unwrap();

        let x_data = randn(t * dim);
        let x = Tensor::matrix(t, dim, x_data.clone()).unwrap();
        let full = block.forward(&x).unwrap();

        let mut cache = KvCache::new(t, dim);
        for r in 0..t {
            let row = Tensor::matrix(1, dim, x_data[r * dim..(r + 1) * dim].to_vec()).unwrap();
            let inc = block.forward_with_cache(&row, &mut cache).unwrap();
            for d in 0..dim {
                let a = full.data[r * dim + d];
                let b = inc.data[d];
                assert!((a - b).abs() < 1e-4, "row {r} dim {d}: full={a} cached={b}");
            }
        }
        assert!(cache.is_full());
    }

    #[test]
    fn kv_cache_overflow_errors() {
        let block = minimal_block(2, 1, 4, 8);
        let mut cache = KvCache::new(2, 4);
        let row = Tensor::matrix(1, 4, vec![0.1; 4]).unwrap();
        block.forward_with_cache(&row, &mut cache).unwrap();
        block.forward_with_cache(&row, &mut cache).unwrap();
        assert!(block.forward_with_cache(&row, &mut cache).is_err());
        cache.clear();
        assert!(block.forward_with_cache(&row, &mut cache).is_ok());
    }

    #[test]
    fn embed_one_matches_batch_forward() {
        let emb = Embedding::new(
            3, 4, 2,
            vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0],
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        ).unwrap();
        let x = Tensor::matrix(1, 3, vec![0.0, 2.0, 1.0]).unwrap();
        let batch = emb.forward(&x).unwrap();
        for pos in 0..3 {
            let tok = x.data[pos] as usize;
            let one = emb.embed_one(tok, pos).unwrap();
            for d in 0..2 {
                assert!((one.data[d] - batch.data[pos * 2 + d]).abs() < 1e-6);
            }
        }
        assert!(emb.embed_one(9, 0).is_err());  // token OOB
        assert!(emb.embed_one(0, 9).is_err());  // position OOB
    }

    #[test]
    fn transformer_block_invalid_heads_rejected() {
        // 3 heads do not divide dim=8; 0 heads is invalid; 0 context invalid.
        let c = 8usize;
        let mk = |ctx: usize, heads: usize| TransformerBlock::new(
            ctx, heads, c,
            vec![1.0; c], vec![0.0; c],
            vec![0.0; c*c], vec![0.0; c],
            vec![0.0; c*c], vec![0.0; c],
            vec![0.0; c*c], vec![0.0; c],
            vec![0.0; c*c], vec![0.0; c],
            vec![1.0; c], vec![0.0; c],
            vec![0.0; c*c], vec![0.0; c],
            vec![0.0; c*c], vec![0.0; c],
        );
        assert!(mk(4, 3).is_err());
        assert!(mk(4, 0).is_err());
        assert!(mk(0, 2).is_err());
        assert!(mk(4, 2).is_ok());
    }

    #[test]
    fn transformer_block_residual_adds_input() {
        // With near-zero weights the block output ≈ input (residual dominates).
        let block = minimal_block(2, 1, 4, 8);
        let input_data: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let x = Tensor::matrix(2, 4, input_data.clone()).unwrap();
        let y = block.forward(&x).unwrap();
        // Residual ensures y is not all-zero even with tiny weights
        assert!(y.data.iter().any(|&v| v.abs() > 0.01));
    }
}
