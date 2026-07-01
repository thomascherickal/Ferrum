//! Trainable network: EmbedT, DenseT, ReluT, Net, backprop, and the bridge to inference.
use crate::activation::Activation;
use crate::error::{InferError, Result};
use crate::layer::{ActivationLayer, Embedding, Flatten, Linear};
use crate::model::Sequential;
use crate::ops;
use crate::optim::Sgd;
use crate::rng::Rng;
use crate::tensor::Tensor;
use crate::verbose;

// ---------------------------------------------------------------------------
// Trainable dense layer
// ---------------------------------------------------------------------------

pub struct DenseT {
    pub weight: Tensor,
    pub bias: Tensor,
    grad_w: Tensor,
    grad_b: Tensor,
    vel_w: Tensor,
    vel_b: Tensor,
    input: Option<Tensor>,
    in_f: usize,
    out_f: usize,
}

impl DenseT {
    pub fn new_random(in_f: usize, out_f: usize, scale: f32, rng: &mut Rng) -> Self {
        vprintln!(
            "[train::DenseT::new_random] Creating layer: in={}, out={}, scale={:.6}",
            in_f,
            out_f,
            scale
        );
        let w: Vec<f32> = (0..in_f * out_f)
            .map(|_| rng.next_normal() * scale)
            .collect();
        let layer = Self {
            weight: Tensor::matrix(in_f, out_f, w).unwrap(),
            bias: Tensor::zeros(vec![out_f]),
            grad_w: Tensor::zeros(vec![in_f, out_f]),
            grad_b: Tensor::zeros(vec![out_f]),
            vel_w: Tensor::zeros(vec![in_f, out_f]),
            vel_b: Tensor::zeros(vec![out_f]),
            input: None,
            in_f,
            out_f,
        };
        if verbose::is_verbose() {
            let (wmin, wmax, wmean) = verbose::stats(&layer.weight.data);
            vprintln!(
                "[train::DenseT::new_random]   weight stats: min={:.6}, max={:.6}, mean={:.6}",
                wmin,
                wmax,
                wmean
            );
        }
        layer
    }

    fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        vprintln!(
            "[train::DenseT::forward] input shape={:?}, weight=[{},{}]",
            x.shape,
            self.in_f,
            self.out_f
        );
        self.input = Some(x.clone());
        let result = ops::add_bias(&ops::matmul(x, &self.weight)?, &self.bias)?;
        if verbose::is_verbose() {
            let (rmin, rmax, rmean) = verbose::stats(&result.data);
            vprintln!("[train::DenseT::forward]   output shape={:?}, stats: min={:.6}, max={:.6}, mean={:.6}", result.shape, rmin, rmax, rmean);
            verbose::check_nan_inf(&result.data, "DenseT::forward output");
        }
        Ok(result)
    }

    /// Backprop: dW = xᵀ·dy, db = Σ_rows(dy), dx = dy·Wᵀ
    fn backward(&mut self, dy: &Tensor) -> Result<Tensor> {
        vprintln!("[train::DenseT::backward] dy shape={:?}", dy.shape);
        let x = self
            .input
            .as_ref()
            .ok_or_else(|| InferError::Format("backward before forward".into()))?;
        self.grad_w = ops::matmul(&ops::transpose(x)?, dy)?;
        self.grad_b = ops::sum_axis0(dy)?;
        let dx = ops::matmul(dy, &ops::transpose(&self.weight)?)?;
        if verbose::is_verbose() {
            let (gmin, gmax, gmean) = verbose::stats(&self.grad_w.data);
            vprintln!(
                "[train::DenseT::backward]   grad_w stats: min={:.6e}, max={:.6e}, mean={:.6e}",
                gmin,
                gmax,
                gmean
            );
            let (gbmin, gbmax, gbmean) = verbose::stats(&self.grad_b.data);
            vprintln!(
                "[train::DenseT::backward]   grad_b stats: min={:.6e}, max={:.6e}, mean={:.6e}",
                gbmin,
                gbmax,
                gbmean
            );
            verbose::check_nan_inf(&self.grad_w.data, "DenseT::backward grad_w");
            verbose::check_nan_inf(&self.grad_b.data, "DenseT::backward grad_b");
            verbose::check_nan_inf(&dx.data, "DenseT::backward dx");
        }
        Ok(dx)
    }

    fn step(&mut self, opt: &Sgd) -> Result<()> {
        vprintln!(
            "[train::DenseT::step] Updating weight=[{},{}], bias=[{}]",
            self.in_f,
            self.out_f,
            self.out_f
        );
        opt.step(&mut self.weight, &self.grad_w, &mut self.vel_w)?;
        opt.step(&mut self.bias, &self.grad_b, &mut self.vel_b)?;
        if verbose::is_verbose() {
            let (wmin, wmax, wmean) = verbose::stats(&self.weight.data);
            vprintln!(
                "[train::DenseT::step]   post-update weight: min={:.6e}, max={:.6e}, mean={:.6e}",
                wmin,
                wmax,
                wmean
            );
            verbose::check_nan_inf(&self.weight.data, "DenseT::step weight");
            verbose::check_nan_inf(&self.bias.data, "DenseT::step bias");
        }
        Ok(())
    }

    fn to_linear(&self) -> Result<Linear> {
        Linear::new(
            self.in_f,
            self.out_f,
            self.weight.data.clone(),
            self.bias.data.clone(),
        )
    }
}

// ---------------------------------------------------------------------------
// Trainable embedding (token-ID lookup, flattened)
// ---------------------------------------------------------------------------

/// Trainable token-embedding layer for the embedded-MLP language-model path.
///
/// Forward maps a batch of token IDs `[B, T]` to flattened embeddings
/// `[B, T·E]` by table lookup — position is encoded by the output slot, so no
/// positional table is needed. Backward scatter-adds the upstream gradient
/// into the embedding table rows.
pub struct EmbedT {
    pub table: Tensor, // [vocab_size, embed_dim]
    grad: Tensor,
    vel: Tensor,
    input: Option<Tensor>,
    vocab_size: usize,
    embed_dim: usize,
}

impl EmbedT {
    pub fn new_random(vocab_size: usize, embed_dim: usize, rng: &mut Rng) -> Self {
        let scale = (1.0 / embed_dim as f32).sqrt();
        vprintln!(
            "[train::EmbedT::new_random] vocab={}, dim={}, scale={:.6}",
            vocab_size,
            embed_dim,
            scale
        );
        let w: Vec<f32> = (0..vocab_size * embed_dim)
            .map(|_| rng.next_normal() * scale)
            .collect();
        Self {
            table: Tensor::matrix(vocab_size, embed_dim, w).unwrap(),
            grad: Tensor::zeros(vec![vocab_size, embed_dim]),
            vel: Tensor::zeros(vec![vocab_size, embed_dim]),
            input: None,
            vocab_size,
            embed_dim,
        }
    }

    fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        let (batch, seq_len) = x.matrix_dims()?;
        vprintln!(
            "[train::EmbedT::forward] input=[{},{}], vocab={}, dim={}",
            batch,
            seq_len,
            self.vocab_size,
            self.embed_dim
        );
        self.input = Some(x.clone());
        let e = self.embed_dim;
        let mut out = vec![0.0f32; batch * seq_len * e];
        for b in 0..batch {
            for t in 0..seq_len {
                let tok = x.data[b * seq_len + t].round() as usize;
                if tok >= self.vocab_size {
                    return Err(InferError::DimMismatch(format!(
                        "token id {tok} out of bounds for vocab_size {}",
                        self.vocab_size
                    )));
                }
                let src = tok * e;
                let dst = (b * seq_len + t) * e;
                out[dst..dst + e].copy_from_slice(&self.table.data[src..src + e]);
            }
        }
        Tensor::matrix(batch, seq_len * e, out)
    }

    /// dTable[token] += dy slot; token IDs receive no gradient (returns zeros).
    fn backward(&mut self, dy: &Tensor) -> Result<Tensor> {
        let x = self
            .input
            .as_ref()
            .ok_or_else(|| InferError::Format("backward before forward".into()))?;
        let (batch, seq_len) = x.matrix_dims()?;
        let e = self.embed_dim;
        let (dy_rows, dy_cols) = dy.matrix_dims()?;
        if dy_rows != batch || dy_cols != seq_len * e {
            return Err(InferError::DimMismatch(format!(
                "EmbedT backward: dy [{dy_rows},{dy_cols}] ≠ [{batch},{}]",
                seq_len * e
            )));
        }
        for v in &mut self.grad.data {
            *v = 0.0;
        }
        for b in 0..batch {
            for t in 0..seq_len {
                let tok = x.data[b * seq_len + t].round() as usize;
                let src = (b * seq_len + t) * e;
                let dst = tok * e;
                for d in 0..e {
                    self.grad.data[dst + d] += dy.data[src + d];
                }
            }
        }
        Ok(Tensor::zeros(vec![batch, seq_len]))
    }

    fn step(&mut self, opt: &Sgd) -> Result<()> {
        opt.step(&mut self.table, &self.grad, &mut self.vel)
    }
}

// ---------------------------------------------------------------------------
// Trainable ReLU
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct ReluT {
    mask: Option<Tensor>,
}

impl ReluT {
    pub fn new() -> Self {
        Self::default()
    }

    fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        self.mask = Some(x.map(|v| if v > 0.0 { 1.0 } else { 0.0 }));
        let result = x.map(|v| v.max(0.0));
        if verbose::is_verbose() {
            let zeros = result.data.iter().filter(|&&v| v == 0.0).count();
            vprintln!(
                "[train::ReluT::forward] shape={:?}, zeroed={}/{} ({:.1}% dead)",
                result.shape,
                zeros,
                result.data.len(),
                100.0 * zeros as f32 / result.data.len() as f32
            );
        }
        Ok(result)
    }

    fn backward(&mut self, dy: &Tensor) -> Result<Tensor> {
        let mask = self
            .mask
            .as_ref()
            .ok_or_else(|| InferError::Format("backward before forward".into()))?;
        let result = ops::mul(dy, mask)?;
        vprintln!("[train::ReluT::backward] dy shape={:?}", dy.shape);
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Heterogeneous layer enum
// ---------------------------------------------------------------------------

enum TLayer {
    Embed(Box<EmbedT>),
    Dense(Box<DenseT>),
    Relu(ReluT),
}

impl TLayer {
    fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        match self {
            TLayer::Embed(e) => e.forward(x),
            TLayer::Dense(d) => d.forward(x),
            TLayer::Relu(r) => r.forward(x),
        }
    }
    fn backward(&mut self, dy: &Tensor) -> Result<Tensor> {
        match self {
            TLayer::Embed(e) => e.backward(dy),
            TLayer::Dense(d) => d.backward(dy),
            TLayer::Relu(r) => r.backward(dy),
        }
    }
    fn step(&mut self, opt: &Sgd) -> Result<()> {
        match self {
            TLayer::Embed(e) => e.step(opt),
            TLayer::Dense(d) => d.step(opt),
            TLayer::Relu(_) => Ok(()),
        }
    }
}

// ---------------------------------------------------------------------------
// Net
// ---------------------------------------------------------------------------

/// A trainable one-hidden-layer MLP.
pub struct Net {
    layers: Vec<TLayer>,
    input_dim: usize,
    output_dim: usize,
    /// Quantization Aware Training: when enabled, `train_epoch` runs
    /// forward/backward against int8-snapped weights while SGD updates
    /// full-precision master weights (straight-through estimator).
    qat: bool,
    /// Global-norm gradient clipping threshold. When `Some(max_norm)`,
    /// `train_epoch` rescales the gradients so their global L2 norm is at most
    /// `max_norm` before the SGD update. `None` (default) leaves the update
    /// unchanged.
    grad_clip: Option<f32>,
}

impl Net {
    /// `input_dim → hidden (ReLU) → output_dim (logits)`
    pub fn mlp(input_dim: usize, hidden: usize, output_dim: usize, rng: &mut Rng) -> Self {
        vprintln!(
            "[train::Net::mlp] Building MLP: input={} → hidden={} → output={}",
            input_dim,
            hidden,
            output_dim
        );
        let s1 = (2.0 / input_dim as f32).sqrt(); // Kaiming init for ReLU
        let s2 = (1.0 / hidden as f32).sqrt();
        let layers = vec![
            TLayer::Dense(Box::new(DenseT::new_random(input_dim, hidden, s1, rng))),
            TLayer::Relu(ReluT::new()),
            TLayer::Dense(Box::new(DenseT::new_random(hidden, output_dim, s2, rng))),
        ];
        let net = Self {
            layers,
            input_dim,
            output_dim,
            qat: false,
            grad_clip: None,
        };
        vprintln!("[train::Net::mlp] Total params: {}", net.num_params());
        net
    }

    /// Token-ID language-model MLP:
    /// `[B, context_len] ids → embed (E per token) → hidden (ReLU) → output_dim (logits)`.
    ///
    /// Compared to a one-hot MLP this shrinks the first layer from
    /// `context_len × vocab × hidden` to `vocab × E + context_len × E × hidden`
    /// parameters — drastically smaller for any non-trivial vocabulary.
    pub fn embedding_mlp(
        vocab_size: usize,
        context_len: usize,
        embed_dim: usize,
        hidden: usize,
        output_dim: usize,
        rng: &mut Rng,
    ) -> Self {
        vprintln!(
            "[train::Net::embedding_mlp] vocab={} ctx={} E={} → hidden={} → out={}",
            vocab_size,
            context_len,
            embed_dim,
            hidden,
            output_dim
        );
        let flat = context_len * embed_dim;
        let s1 = (2.0 / flat as f32).sqrt(); // Kaiming init for ReLU
        let s2 = (1.0 / hidden as f32).sqrt();
        let layers = vec![
            TLayer::Embed(Box::new(EmbedT::new_random(vocab_size, embed_dim, rng))),
            TLayer::Dense(Box::new(DenseT::new_random(flat, hidden, s1, rng))),
            TLayer::Relu(ReluT::new()),
            TLayer::Dense(Box::new(DenseT::new_random(hidden, output_dim, s2, rng))),
        ];
        let net = Self {
            layers,
            input_dim: context_len,
            output_dim,
            qat: false,
            grad_clip: None,
        };
        vprintln!(
            "[train::Net::embedding_mlp] Total params: {}",
            net.num_params()
        );
        net
    }

    /// Enable or disable int8 Quantization Aware Training (see the `qat`
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

    /// Every gradient tensor, in layer order (embedding table, then each dense
    /// layer's weight and bias). Used by [`Net::clip_grad_norm`].
    fn grad_tensors_mut(&mut self) -> Vec<&mut Tensor> {
        let mut v = Vec::new();
        for l in &mut self.layers {
            match l {
                TLayer::Embed(e) => v.push(&mut e.grad),
                TLayer::Dense(d) => {
                    v.push(&mut d.grad_w);
                    v.push(&mut d.grad_b);
                }
                TLayer::Relu(_) => {}
            }
        }
        v
    }

    /// Rescale all gradients so their global L2 norm is at most `max_norm`,
    /// returning the pre-clip norm. Call after [`Net::backward`] and before
    /// [`Net::step`]; [`train_epoch`] does this automatically when
    /// [`Net::set_grad_clip`] is set.
    pub fn clip_grad_norm(&mut self, max_norm: f32) -> f32 {
        let mut grads = self.grad_tensors_mut();
        crate::optim::clip_grad_norm(&mut grads, max_norm)
    }

    /// Every parameter tensor, in a fixed order (used by the QAT snapshot /
    /// fake-quantize / restore cycle).
    fn param_tensors_mut(&mut self) -> Vec<&mut Tensor> {
        let mut v = Vec::new();
        for l in &mut self.layers {
            match l {
                TLayer::Embed(e) => v.push(&mut e.table),
                TLayer::Dense(d) => {
                    v.push(&mut d.weight);
                    v.push(&mut d.bias);
                }
                TLayer::Relu(_) => {}
            }
        }
        v
    }

    /// Copy of every parameter tensor's data (the fp32 master weights).
    pub(crate) fn snapshot_weights(&mut self) -> Vec<Vec<f32>> {
        self.param_tensors_mut()
            .iter()
            .map(|t| t.data.clone())
            .collect()
    }

    /// Snap every (large-enough) parameter tensor onto the int8 grid in place,
    /// per output-row (§7) to match the per-channel FINF v5 quantization. 1-D
    /// parameters fall back to a single scale.
    pub(crate) fn fake_quantize_weights(&mut self) {
        for t in self.param_tensors_mut() {
            let channels = if t.shape.len() >= 2 {
                t.shape[0].max(1)
            } else {
                1
            };
            crate::quant::fake_quantize_int8_per_channel(&mut t.data, channels);
        }
    }

    /// Restore master weights captured by [`Net::snapshot_weights`].
    pub(crate) fn restore_weights(&mut self, snapshot: &[Vec<f32>]) {
        for (t, s) in self.param_tensors_mut().into_iter().zip(snapshot) {
            t.data.copy_from_slice(s);
        }
    }

    pub fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        vprintln!("[train::Net::forward] input shape={:?}", x.shape);
        let mut cur = x.clone();
        for (i, l) in self.layers.iter_mut().enumerate() {
            cur = l.forward(&cur)?;
            if verbose::is_verbose() {
                let name = match l {
                    TLayer::Embed(_) => "Embed",
                    TLayer::Dense(_) => "Dense",
                    TLayer::Relu(_) => "ReLU",
                };
                vprintln!(
                    "[train::Net::forward]   layer[{}] {} → shape={:?}",
                    i,
                    name,
                    cur.shape
                );
            }
        }
        Ok(cur)
    }

    pub fn backward(&mut self, dlogits: &Tensor) -> Result<()> {
        vprintln!("[train::Net::backward] dlogits shape={:?}", dlogits.shape);
        let mut grad = dlogits.clone();
        for (i, l) in self.layers.iter_mut().rev().enumerate() {
            grad = l.backward(&grad)?;
            vprintln!(
                "[train::Net::backward]   layer[rev-{}] → grad shape={:?}",
                i,
                grad.shape
            );
        }
        Ok(())
    }

    pub fn step(&mut self, opt: &Sgd) -> Result<()> {
        vprintln!(
            "[train::Net::step] Optimizer step (lr={}, momentum={})",
            opt.lr,
            opt.momentum
        );
        for l in &mut self.layers {
            l.step(opt)?;
        }
        Ok(())
    }

    pub fn num_params(&self) -> usize {
        self.layers
            .iter()
            .map(|l| match l {
                TLayer::Embed(e) => e.table.numel(),
                TLayer::Dense(d) => d.weight.numel() + d.bias.numel(),
                TLayer::Relu(_) => 0,
            })
            .sum()
    }

    pub fn input_dim(&self) -> usize {
        self.input_dim
    }
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    /// Convert to an inference `Sequential`, appending Softmax.
    pub fn to_inference(&self) -> Result<Sequential> {
        self.to_inference_task(crate::csv::TaskType::Classification)
    }

    /// Export to inference model. Classification appends Softmax; regression appends Identity.
    pub fn to_inference_task(&self, task: crate::csv::TaskType) -> Result<Sequential> {
        vprintln!(
            "[train::Net::to_inference_task] Converting to inference model, task={:?}",
            task
        );
        let mut m = Sequential::new();
        for l in &self.layers {
            match l {
                TLayer::Embed(e) => {
                    // Inference Embedding adds a positional table; training used
                    // none (position is encoded by the flattened slot), so it
                    // exports as all-zeros. Flatten restores the [1, T·E] shape
                    // the next Linear expects.
                    let context_len = self.input_dim;
                    m.push(Box::new(Embedding::new(
                        e.vocab_size,
                        context_len,
                        e.embed_dim,
                        e.table.data.clone(),
                        vec![0.0; context_len * e.embed_dim],
                    )?));
                    m.push(Box::new(Flatten::new()));
                }
                TLayer::Dense(d) => m.push(Box::new(d.to_linear()?)),
                TLayer::Relu(_) => m.push(Box::new(ActivationLayer::new(Activation::ReLU))),
            }
        }
        let final_act = match task {
            crate::csv::TaskType::Classification => Activation::Softmax,
            crate::csv::TaskType::Regression => Activation::Identity,
            crate::csv::TaskType::TransformerSLM => Activation::Softmax,
        };
        m.push(Box::new(ActivationLayer::new(final_act)));
        vprintln!(
            "[train::Net::to_inference_task] Inference model: {} layers",
            m.len()
        );
        Ok(m)
    }
}

// ---------------------------------------------------------------------------
// Training loop helpers
// ---------------------------------------------------------------------------

/// Run one epoch of minibatch SGD. Returns mean train loss over the epoch.
///
/// If the net has QAT enabled ([`Net::set_qat`]), each step runs forward and
/// backward against int8-snapped weights, then applies the SGD update to the
/// full-precision master weights (straight-through estimator), so the trained
/// model is robust to int8 export.
pub fn train_epoch(
    net: &mut Net,
    x: &Tensor,
    y: &[usize],
    batch_size: usize,
    opt: &Sgd,
    rng: &mut Rng,
) -> Result<f32> {
    use crate::loss::softmax_cross_entropy;
    let n = y.len();
    let mut total_loss = 0.0f32;
    let mut batches = 0usize;

    // Shuffle a permutation of every sample once per epoch and draw minibatches
    // without replacement (T4): an epoch now covers the whole dataset exactly
    // once instead of ≈63% in expectation under sampling with replacement.
    let steps = n.div_ceil(batch_size);
    let perm = rng.shuffled_indices(n);
    vprintln!(
        "[train::train_epoch] samples={}, batch_size={}, steps={}, lr={}, momentum={}",
        n,
        batch_size,
        steps,
        opt.lr,
        opt.momentum
    );

    let epoch_start = std::time::Instant::now();

    for step in 0..steps {
        let step_start = std::time::Instant::now();

        let indices = &perm[step * batch_size..((step + 1) * batch_size).min(n)];
        let cur_bs = indices.len();

        let mut xb_data = Vec::with_capacity(cur_bs * net.input_dim);
        let mut yb = Vec::with_capacity(cur_bs);
        let (_, cols) = x.matrix_dims()?;
        for &i in indices {
            xb_data.extend_from_slice(&x.data[i * cols..(i + 1) * cols]);
            yb.push(y[i]);
        }
        let xb = Tensor::matrix(cur_bs, cols, xb_data)?;

        vprintln!(
            "[train::train_epoch]   step {}/{}: batch shape=[{},{}]",
            step + 1,
            steps,
            cur_bs,
            cols
        );

        // QAT: gradients are computed at the int8-snapped weights, but the
        // optimizer updates the fp32 masters (straight-through estimator).
        let masters = if net.qat_enabled() {
            let snapshot = net.snapshot_weights();
            net.fake_quantize_weights();
            Some(snapshot)
        } else {
            None
        };

        // Forward
        let logits = net.forward(&xb)?;
        if verbose::is_verbose() {
            verbose::check_nan_inf(
                &logits.data,
                &format!("train_epoch step {} forward logits", step + 1),
            );
        }

        // Loss
        let (loss, dlogits) = softmax_cross_entropy(&logits, &yb)?;
        vprintln!(
            "[train::train_epoch]   step {}/{}: loss={:.6}",
            step + 1,
            steps,
            loss
        );

        if verbose::is_verbose() {
            if loss.is_nan() {
                crate::verbose::log_line(&format!(
                    "[ferrum_core::WARN] ⚠️  NaN loss detected at step {}! Training may diverge.",
                    step + 1
                ));
            }
            if loss.is_infinite() {
                crate::verbose::log_line(&format!("[ferrum_core::WARN] ⚠️  Infinite loss detected at step {}! Training may diverge.", step+1));
            }
            verbose::check_nan_inf(
                &dlogits.data,
                &format!("train_epoch step {} dlogits", step + 1),
            );
        }

        // Backward
        net.backward(&dlogits)?;

        // Step (against restored fp32 masters when QAT is on)
        if let Some(snapshot) = &masters {
            net.restore_weights(snapshot);
        }
        if let Some(max_norm) = net.grad_clip {
            net.clip_grad_norm(max_norm);
        }
        net.step(opt)?;

        total_loss += loss;
        batches += 1;

        if verbose::is_verbose() {
            let step_ms = step_start.elapsed().as_secs_f64() * 1000.0;
            vprintln!(
                "[train::train_epoch]   step {}/{}: done in {:.1}ms, running avg loss={:.6}",
                step + 1,
                steps,
                step_ms,
                total_loss / batches as f32
            );
        }
    }

    let epoch_ms = epoch_start.elapsed().as_secs_f64() * 1000.0;
    let avg_loss = total_loss / batches as f32;
    vprintln!(
        "[train::train_epoch] Epoch done in {:.1}ms, mean loss={:.6}",
        epoch_ms,
        avg_loss
    );

    Ok(avg_loss)
}

/// Compute accuracy on a full dataset (no gradient tracking).
pub fn accuracy(net: &mut Net, x: &Tensor, y: &[usize]) -> Result<f32> {
    vprintln!(
        "[train::accuracy] Computing accuracy on {} samples",
        y.len()
    );
    let logits = net.forward(x)?;
    let preds = crate::ops::argmax_rows(&logits)?;
    let correct = preds.iter().zip(y).filter(|(p, t)| p == t).count();
    let acc = correct as f32 / y.len() as f32;
    vprintln!(
        "[train::accuracy] Result: {}/{} correct = {:.4} ({:.1}%)",
        correct,
        y.len(),
        acc,
        acc * 100.0
    );
    Ok(acc)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::loss::softmax_cross_entropy;

    fn small_net() -> Net {
        Net::mlp(4, 8, 3, &mut Rng::new(1))
    }

    #[test]
    fn forward_output_shape() {
        let mut net = small_net();
        let x = Tensor::matrix(5, 4, vec![0.5f32; 20]).unwrap();
        let y = net.forward(&x).unwrap();
        assert_eq!(y.shape, vec![5, 3]);
    }

    #[test]
    fn num_params_is_positive() {
        assert!(small_net().num_params() > 0);
    }

    #[test]
    fn to_inference_has_softmax() {
        let net = small_net();
        let model = net.to_inference().unwrap();
        // Dense + ReLU + Dense + Softmax = 4 layers
        assert_eq!(model.len(), 4);
        let x = Tensor::row(vec![0.0f32; 4]).unwrap();
        let out = model.forward(&x).unwrap();
        let sum: f32 = out.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    /// Analytic gradient vs finite differences for the whole network.
    #[test]
    fn backprop_gradient_check() {
        let mut net = Net::mlp(4, 5, 3, &mut Rng::new(2));
        let x = Tensor::matrix(2, 4, vec![0.5, -0.3, 0.8, 0.1, -0.2, 0.4, -0.9, 0.6]).unwrap();
        let targets = [2usize, 0];

        let logits = net.forward(&x).unwrap();
        let (_, dl) = softmax_cross_entropy(&logits, &targets).unwrap();
        net.backward(&dl).unwrap();

        let analytic = match &net.layers[0] {
            TLayer::Dense(d) => d.grad_w.clone(),
            _ => unreachable!(),
        };

        let eps = 1e-3f32;
        for &k in &[0usize, 2, 5, 8, 12, 15, 19] {
            let orig = match &net.layers[0] {
                TLayer::Dense(d) => d.weight.data[k],
                _ => unreachable!(),
            };

            if let TLayer::Dense(d) = &mut net.layers[0] {
                d.weight.data[k] = orig + eps;
            }
            let lp = softmax_cross_entropy(&net.forward(&x).unwrap(), &targets)
                .unwrap()
                .0;

            if let TLayer::Dense(d) = &mut net.layers[0] {
                d.weight.data[k] = orig - eps;
            }
            let lm = softmax_cross_entropy(&net.forward(&x).unwrap(), &targets)
                .unwrap()
                .0;

            if let TLayer::Dense(d) = &mut net.layers[0] {
                d.weight.data[k] = orig;
            }
            let numeric = (lp - lm) / (2.0 * eps);
            assert!(
                (numeric - analytic.data[k]).abs() < 1e-2,
                "weight[{k}]: analytic={} numeric={numeric}",
                analytic.data[k]
            );
        }
    }

    #[test]
    fn training_reduces_loss_on_iris_like_data() {
        // 4 features, 3 classes, 30 fake examples.
        let mut rng = Rng::new(42);
        let x_data: Vec<f32> = (0..30 * 4).map(|_| rng.next_normal()).collect();
        let x = Tensor::matrix(30, 4, x_data).unwrap();
        let y: Vec<usize> = (0..30).map(|i| i % 3).collect();

        let mut net = Net::mlp(4, 16, 3, &mut rng);
        let opt = Sgd::with_momentum(0.1, 0.9);
        let (loss0, _) = softmax_cross_entropy(&net.forward(&x).unwrap(), &y).unwrap();

        for _ in 0..300 {
            let logits = net.forward(&x).unwrap();
            let (_, dl) = softmax_cross_entropy(&logits, &y).unwrap();
            net.backward(&dl).unwrap();
            net.step(&opt).unwrap();
        }
        let (loss1, _) = softmax_cross_entropy(&net.forward(&x).unwrap(), &y).unwrap();
        assert!(loss1 < loss0 * 0.5, "loss: {loss0:.4} → {loss1:.4}");
    }

    #[test]
    fn clip_grad_norm_caps_net_global_norm() {
        // After backward, clipping must cap the global gradient norm.
        let mut rng = Rng::new(42);
        let x = Tensor::matrix(8, 4, (0..32).map(|_| rng.next_normal() * 50.0).collect()).unwrap();
        let y: Vec<usize> = (0..8).map(|i| i % 3).collect();
        let mut net = Net::mlp(4, 16, 3, &mut rng);
        let logits = net.forward(&x).unwrap();
        let (_, dl) = softmax_cross_entropy(&logits, &y).unwrap();
        net.backward(&dl).unwrap();

        let max_norm = 0.5f32;
        let pre = net.clip_grad_norm(max_norm);
        let mut sumsq = 0.0f64;
        for g in net.grad_tensors_mut() {
            for &v in &g.data {
                sumsq += (v as f64) * (v as f64);
            }
        }
        let post = sumsq.sqrt() as f32;
        assert!(
            pre > max_norm,
            "pre-clip norm {pre} should exceed the budget"
        );
        assert!(
            post <= max_norm + 1e-4,
            "post-clip norm {post} exceeds {max_norm}"
        );
    }

    #[test]
    fn train_epoch_respects_grad_clip_flag() {
        // With a destructive LR the unclipped run diverges to non-finite
        // weights; enabling grad-clip keeps the MLP finite.
        let mut rng = Rng::new(42);
        let x = Tensor::matrix(30, 4, (0..120).map(|_| rng.next_normal()).collect()).unwrap();
        let y: Vec<usize> = (0..30).map(|i| i % 3).collect();
        let opt = Sgd::new(50.0);

        let finite = |net: &mut Net| {
            net.snapshot_weights()
                .iter()
                .all(|t| t.iter().all(|v| v.is_finite()))
        };

        let mut diverging = Net::mlp(4, 16, 3, &mut Rng::new(1));
        let mut r1 = Rng::new(9);
        for _ in 0..20 {
            let _ = train_epoch(&mut diverging, &x, &y, 10, &opt, &mut r1);
        }
        assert!(
            !finite(&mut diverging),
            "high LR expected to diverge without clipping"
        );

        let mut clipped = Net::mlp(4, 16, 3, &mut Rng::new(1));
        clipped.set_grad_clip(Some(1.0));
        assert_eq!(clipped.grad_clip(), Some(1.0));
        let mut r2 = Rng::new(9);
        for _ in 0..20 {
            train_epoch(&mut clipped, &x, &y, 10, &opt, &mut r2).unwrap();
        }
        assert!(finite(&mut clipped), "clipped run must keep weights finite");
    }

    #[test]
    fn qat_train_epoch_reduces_loss() {
        // Same shape as the iris-like test, but through train_epoch with QAT
        // enabled: gradients flow at int8-snapped weights, SGD updates masters.
        let mut rng = Rng::new(42);
        let x_data: Vec<f32> = (0..30 * 4).map(|_| rng.next_normal()).collect();
        let x = Tensor::matrix(30, 4, x_data).unwrap();
        let y: Vec<usize> = (0..30).map(|i| i % 3).collect();

        let mut net = Net::mlp(4, 32, 3, &mut rng);
        net.set_qat(true);
        assert!(net.qat_enabled());
        let opt = Sgd::with_momentum(0.1, 0.9);
        let first = train_epoch(&mut net, &x, &y, 10, &opt, &mut rng).unwrap();
        let mut last = first;
        for _ in 0..60 {
            last = train_epoch(&mut net, &x, &y, 10, &opt, &mut rng).unwrap();
        }
        assert!(last < first * 0.5, "QAT loss: {first:.4} → {last:.4}");
    }

    #[test]
    fn qat_snapshot_quantize_restore_cycle() {
        let mut rng = Rng::new(5);
        let mut net = Net::embedding_mlp(20, 4, 8, 32, 20, &mut rng);
        let masters = net.snapshot_weights();
        net.fake_quantize_weights();
        let quantized = net.snapshot_weights();
        assert!(
            masters.iter().zip(&quantized).any(|(m, q)| m != q),
            "fake quantization changed no weights"
        );
        net.restore_weights(&masters);
        assert_eq!(net.snapshot_weights(), masters);
    }

    #[test]
    fn accuracy_bounds() {
        let mut net = small_net();
        let x = Tensor::matrix(10, 4, vec![0.0f32; 40]).unwrap();
        let y: Vec<usize> = (0..10).map(|i| i % 3).collect();
        let acc = accuracy(&mut net, &x, &y).unwrap();
        assert!((0.0..=1.0).contains(&acc));
    }

    #[test]
    fn to_inference_task_branches() {
        let net = small_net();
        let m_reg = net
            .to_inference_task(crate::csv::TaskType::Regression)
            .unwrap();
        assert_eq!(m_reg.len(), 4);
        assert_eq!(m_reg.layers()[3].name(), "Activation(Identity)");

        let m_slm = net
            .to_inference_task(crate::csv::TaskType::TransformerSLM)
            .unwrap();
        assert_eq!(m_slm.len(), 4);
        assert_eq!(m_slm.layers()[3].name(), "Activation(Softmax)");
    }

    #[test]
    fn embedding_mlp_trains_and_matches_inference() {
        // Tiny next-token task over a 5-token vocabulary.
        let mut rng = Rng::new(3);
        let vocab = 5;
        let ctx = 3;
        let tokens: Vec<usize> = (0..40).map(|i| i % vocab).collect();
        let n = tokens.len() - ctx;
        let mut x_data = Vec::new();
        let mut y = Vec::new();
        for i in 0..n {
            x_data.extend(tokens[i..i + ctx].iter().map(|&t| t as f32));
            y.push(tokens[i + ctx]);
        }
        let x = Tensor::matrix(n, ctx, x_data).unwrap();

        let mut net = Net::embedding_mlp(vocab, ctx, 4, 16, vocab, &mut rng);
        let opt = Sgd::with_momentum(0.1, 0.9);
        let (loss0, _) = softmax_cross_entropy(&net.forward(&x).unwrap(), &y).unwrap();
        for _ in 0..200 {
            let logits = net.forward(&x).unwrap();
            let (_, dl) = softmax_cross_entropy(&logits, &y).unwrap();
            net.backward(&dl).unwrap();
            net.step(&opt).unwrap();
        }
        let (loss1, _) = softmax_cross_entropy(&net.forward(&x).unwrap(), &y).unwrap();
        assert!(loss1 < loss0 * 0.5, "loss: {loss0:.4} → {loss1:.4}");

        // The exported inference model (Embedding + Flatten + Linear…Softmax)
        // must produce the softmax of the training network's logits.
        let model = net.to_inference().unwrap();
        let one = Tensor::matrix(1, ctx, vec![0.0, 1.0, 2.0]).unwrap();
        let train_logits = net.forward(&one).unwrap();
        let infer_probs = model.forward(&one).unwrap();
        let max = train_logits
            .data
            .iter()
            .fold(f32::NEG_INFINITY, |m, &v| m.max(v));
        let exps: Vec<f32> = train_logits.data.iter().map(|&v| (v - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        for (p, e) in infer_probs.data.iter().zip(&exps) {
            assert!((p - e / sum).abs() < 1e-5, "inference/training mismatch");
        }
    }

    /// Finite-difference gradient check for the embedding table.
    #[test]
    fn embedding_gradient_check() {
        let mut rng = Rng::new(9);
        let mut net = Net::embedding_mlp(4, 2, 3, 6, 4, &mut rng);
        let x = Tensor::matrix(2, 2, vec![0.0, 2.0, 3.0, 1.0]).unwrap();
        let targets = [1usize, 3];

        let logits = net.forward(&x).unwrap();
        let (_, dl) = softmax_cross_entropy(&logits, &targets).unwrap();
        net.backward(&dl).unwrap();

        let analytic = match &net.layers[0] {
            TLayer::Embed(e) => e.grad.clone(),
            _ => unreachable!(),
        };

        let eps = 1e-3f32;
        for &k in &[0usize, 3, 6, 7, 9, 11] {
            let orig = match &net.layers[0] {
                TLayer::Embed(e) => e.table.data[k],
                _ => unreachable!(),
            };
            if let TLayer::Embed(e) = &mut net.layers[0] {
                e.table.data[k] = orig + eps;
            }
            let lp = softmax_cross_entropy(&net.forward(&x).unwrap(), &targets)
                .unwrap()
                .0;
            if let TLayer::Embed(e) = &mut net.layers[0] {
                e.table.data[k] = orig - eps;
            }
            let lm = softmax_cross_entropy(&net.forward(&x).unwrap(), &targets)
                .unwrap()
                .0;
            if let TLayer::Embed(e) = &mut net.layers[0] {
                e.table.data[k] = orig;
            }
            let numeric = (lp - lm) / (2.0 * eps);
            assert!(
                (numeric - analytic.data[k]).abs() < 1e-2,
                "table[{k}]: analytic={} numeric={numeric}",
                analytic.data[k]
            );
        }
    }

    #[test]
    fn embed_token_out_of_bounds_errors() {
        let mut e = EmbedT::new_random(4, 3, &mut Rng::new(1));
        let x = Tensor::matrix(1, 2, vec![0.0, 9.0]).unwrap();
        assert!(matches!(e.forward(&x), Err(InferError::DimMismatch(_))));
    }

    #[test]
    fn backward_before_forward_errors() {
        let mut d = DenseT::new_random(4, 3, 0.1, &mut Rng::new(1));
        let dy = Tensor::zeros(vec![1, 3]);
        assert!(matches!(d.backward(&dy), Err(InferError::Format(_))));

        let mut r = ReluT::new();
        assert!(matches!(r.backward(&dy), Err(InferError::Format(_))));
    }
}
