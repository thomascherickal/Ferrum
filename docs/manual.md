# 🧬 Ferrum Reference Manual

This document serves as the comprehensive technical reference for the `ferrum_core` API and structural kernels.

---

## 1. Mathematical Kernels (`ops.rs`)

Matrix calculations are hand-implemented to ensure compatibility across all CPU architectures and WASM sandboxes.

### Matrix Multiplication
Multiplies two matrices $A \in \mathbb{R}^{M \times K}$ and $B \in \mathbb{R}^{K \times N}$ yielding matrix $C \in \mathbb{R}^{M \times N}$ in row-major layout:
$$C_{i, j} = \sum_{p=0}^{K-1} A_{i, p} \cdot B_{p, j}$$

### Softmax
Applies stable row-wise softmax over logits matrix:
$$\text{Softmax}(z)_{i, j} = \frac{e^{z_{i, j} - \max(z_{i, \cdot})}}{\sum_k e^{z_{i, k} - \max(z_{i, \cdot})}}$$

---

## 2. Structural Layers (`layer.rs`)

Layers implement the `Layer` trait:
```rust
pub trait Layer {
    fn forward(&self, input: &Tensor) -> Result<Tensor>;
    fn name(&self) -> String;
    fn as_any(&self) -> &dyn Any;
}
```

### Linear
Affine projection:
$$Y = X \cdot W + b$$
- **Weight**: Shape $[d_{in}, d_{out}]$
- **Bias**: Shape $[d_{out}]$

### LayerNorm
Normalizes each row independently, stabilizing neural activations:
$$\mu = \frac{1}{d} \sum_{c=0}^{d-1} x_c, \quad \sigma^2 = \frac{1}{d} \sum_{c=0}^{d-1} (x_c - \mu)^2$$
$$y_c = \frac{x_c - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma_c + \beta_c$$

### Embedding
Combines discrete token lookup with positional lookup:
$$\text{Embedding}(t, p) = \text{Lookup}(t) + \text{Lookup}(p)$$
- **Token Weight**: Shape $[V, d_{emb}]$
- **Positional Weight**: Shape $[S_{max}, d_{emb}]$

### TransformerBlock
Decoder-only causal Multi-Head Self-Attention (MHSA) coupled with a feedforward residual network:
1. **LayerNorm 1**: $X_{norm} = \text{LayerNorm}(X)$
2. **Q, K, V Projections**: $Q = X_{norm} W_q, \quad K = X_{norm} W_k, \quad V = X_{norm} W_v$
3. **Causal Attention**: $S = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_{head}}} + M\right)$ where $M$ is the causal mask ($M_{i,j} = 0$ for $j \le i$, $-\infty$ otherwise).
4. **Attention Output**: $O = S V$
5. **Project & Residual**: $X_{attn} = X + O W_{out}$
6. **FFN Residual**: $Y = X_{attn} + \text{ReLU}( \text{LayerNorm}(X_{attn}) W_1 + b_1 ) W_2 + b_2$

---

## 3. Loss & Backpropagation Kernels (`loss.rs` & `train.rs`)

### Softmax Cross-Entropy Loss
Computes loss and analytical gradients for training classification systems:
$$L_i = -\ln P(y_i)$$
$$\frac{\partial L}{\partial z_{i, j}} = P(j) - \mathbb{I}(y_i = j)$$

### SGD Optimizer with Momentum
Updates weights $w$ utilizing gradients $g$ and velocity $v$:
$$v_{t+1} = \beta \cdot v_t + \eta \cdot g$$
$$w_{t+1} = w_t - v_{t+1}$$

Used by the MLP training path (`train.rs`).

### Adam Optimizer (`optim::Adam`)
Bias-corrected adaptive moments, used by the transformer trainer:
$$m_{t+1} = \beta_1 m_t + (1 - \beta_1) g, \quad v_{t+1} = \beta_2 v_t + (1 - \beta_2) g^2$$
$$\hat{m} = \frac{m_{t+1}}{1 - \beta_1^t}, \quad \hat{v} = \frac{v_{t+1}}{1 - \beta_2^t}, \quad w_{t+1} = w_t - \eta \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}$$

### Transformer Training (`train_transformer.rs`)
`TransformerNet` implements full backpropagation through token + positional
embeddings, LayerNorm, causal multi-head attention (softmax backward through
the mask), and the FFN, with next-token loss at every position. The training
loop is `forward` → `softmax_cross_entropy` → `backward` → `step(&Adam)`, and
`to_inference()` exports a FINF-serialisable `Sequential`. The high-level
entry point is `GenerativeSLM::train_transformer`. Gradients are verified by
finite-difference checks across all parameter groups in the test suite.

### KV-Cached Generation (`layer::KvCache`)
`TransformerBlock::forward_with_cache` and `Embedding::embed_one` enable
O(T)-per-token incremental generation instead of re-running the full context
(O(T²)) every step. The WASM bindings expose this as `prime(context)` /
`predict_next_cached(token_id)` / `cached_len()`. The cached path produces
distributions identical to the full forward pass.

---

## 4. The FINF v4 Serialization Format

Models are serialized to self-contained binary streams containing all layer shapes, weights, and dataset normalization metrics:

```text
Offsets (bytes):
[0 .. 4]     Magic sequence: b"FINF"
[4 .. 8]     Format Version: u32 = 4
[8 .. 12]    Normalizer length: u32 (N)
[12 .. 12+N] Z-score statistics: string representation (means;stds)
[12+N .. 16+N] Metadata length: u32 (M)
[16+N .. 16+N+M] Model Metadata: JSON string
[16+N+M .. 20+N+M] Number of Layers: u32 (L)
[20+N+M .. ] Layer weights: byte streams prefaced by u8 tag mapping:
              0 -> Linear
              1 -> Activation Layer
              2 -> Embedding
              3 -> LayerNorm
              4 -> TransformerBlock
```

The reader is a forward-only, bounds-checked cursor: byte lengths are verified
against the remaining buffer before allocating, and dimension products use
checked multiplication, so corrupt or truncated files yield a `Format` error
instead of a panic or a huge allocation.
