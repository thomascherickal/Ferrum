//! Trainable network: DenseT, ReluT, Net, backprop, and the bridge to inference.
use crate::activation::Activation;
use crate::error::{InferError, Result};
use crate::layer::{ActivationLayer, Linear};
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
        vprintln!("[train::DenseT::new_random] Creating layer: in={}, out={}, scale={:.6}", in_f, out_f, scale);
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
            vprintln!("[train::DenseT::new_random]   weight stats: min={:.6}, max={:.6}, mean={:.6}", wmin, wmax, wmean);
        }
        layer
    }

    fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        vprintln!("[train::DenseT::forward] input shape={:?}, weight=[{},{}]", x.shape, self.in_f, self.out_f);
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
            vprintln!("[train::DenseT::backward]   grad_w stats: min={:.6e}, max={:.6e}, mean={:.6e}", gmin, gmax, gmean);
            let (gbmin, gbmax, gbmean) = verbose::stats(&self.grad_b.data);
            vprintln!("[train::DenseT::backward]   grad_b stats: min={:.6e}, max={:.6e}, mean={:.6e}", gbmin, gbmax, gbmean);
            verbose::check_nan_inf(&self.grad_w.data, "DenseT::backward grad_w");
            verbose::check_nan_inf(&self.grad_b.data, "DenseT::backward grad_b");
            verbose::check_nan_inf(&dx.data, "DenseT::backward dx");
        }
        Ok(dx)
    }

    fn step(&mut self, opt: &Sgd) -> Result<()> {
        vprintln!("[train::DenseT::step] Updating weight=[{},{}], bias=[{}]", self.in_f, self.out_f, self.out_f);
        opt.step(&mut self.weight, &self.grad_w, &mut self.vel_w)?;
        opt.step(&mut self.bias, &self.grad_b, &mut self.vel_b)?;
        if verbose::is_verbose() {
            let (wmin, wmax, wmean) = verbose::stats(&self.weight.data);
            vprintln!("[train::DenseT::step]   post-update weight: min={:.6e}, max={:.6e}, mean={:.6e}", wmin, wmax, wmean);
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
            vprintln!("[train::ReluT::forward] shape={:?}, zeroed={}/{} ({:.1}% dead)",
                result.shape, zeros, result.data.len(),
                100.0 * zeros as f32 / result.data.len() as f32);
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
    Dense(Box<DenseT>),
    Relu(ReluT),
}

impl TLayer {
    fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        match self {
            TLayer::Dense(d) => d.forward(x),
            TLayer::Relu(r) => r.forward(x),
        }
    }
    fn backward(&mut self, dy: &Tensor) -> Result<Tensor> {
        match self {
            TLayer::Dense(d) => d.backward(dy),
            TLayer::Relu(r) => r.backward(dy),
        }
    }
    fn step(&mut self, opt: &Sgd) -> Result<()> {
        match self {
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
}

impl Net {
    /// `input_dim → hidden (ReLU) → output_dim (logits)`
    pub fn mlp(input_dim: usize, hidden: usize, output_dim: usize, rng: &mut Rng) -> Self {
        vprintln!("[train::Net::mlp] Building MLP: input={} → hidden={} → output={}", input_dim, hidden, output_dim);
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
        };
        vprintln!("[train::Net::mlp] Total params: {}", net.num_params());
        net
    }

    pub fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        vprintln!("[train::Net::forward] input shape={:?}", x.shape);
        let mut cur = x.clone();
        for (i, l) in self.layers.iter_mut().enumerate() {
            cur = l.forward(&cur)?;
            if verbose::is_verbose() {
                let name = match l {
                    TLayer::Dense(_) => "Dense",
                    TLayer::Relu(_) => "ReLU",
                };
                vprintln!("[train::Net::forward]   layer[{}] {} → shape={:?}", i, name, cur.shape);
            }
        }
        Ok(cur)
    }

    pub fn backward(&mut self, dlogits: &Tensor) -> Result<()> {
        vprintln!("[train::Net::backward] dlogits shape={:?}", dlogits.shape);
        let mut grad = dlogits.clone();
        for (i, l) in self.layers.iter_mut().rev().enumerate() {
            grad = l.backward(&grad)?;
            vprintln!("[train::Net::backward]   layer[rev-{}] → grad shape={:?}", i, grad.shape);
        }
        Ok(())
    }

    pub fn step(&mut self, opt: &Sgd) -> Result<()> {
        vprintln!("[train::Net::step] Optimizer step (lr={}, momentum={})", opt.lr, opt.momentum);
        for l in &mut self.layers {
            l.step(opt)?;
        }
        Ok(())
    }

    pub fn num_params(&self) -> usize {
        self.layers
            .iter()
            .map(|l| match l {
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
        vprintln!("[train::Net::to_inference_task] Converting to inference model, task={:?}", task);
        let mut m = Sequential::new();
        for l in &self.layers {
            match l {
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
        vprintln!("[train::Net::to_inference_task] Inference model: {} layers", m.len());
        Ok(m)
    }
}

// ---------------------------------------------------------------------------
// Training loop helpers
// ---------------------------------------------------------------------------

/// Run one epoch of minibatch SGD. Returns mean train loss over the epoch.
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

    // Random minibatch indices (with replacement).
    let steps = n.div_ceil(batch_size);
    vprintln!("[train::train_epoch] samples={}, batch_size={}, steps={}, lr={}, momentum={}",
        n, batch_size, steps, opt.lr, opt.momentum);

    let epoch_start = std::time::Instant::now();

    for step in 0..steps {
        let step_start = std::time::Instant::now();

        let indices: Vec<usize> = (0..batch_size)
            .map(|_| (rng.next_u64() as usize) % n)
            .collect();

        let mut xb_data = Vec::with_capacity(batch_size * net.input_dim);
        let mut yb = Vec::with_capacity(batch_size);
        let (_, cols) = x.matrix_dims()?;
        for &i in &indices {
            xb_data.extend_from_slice(&x.data[i * cols..(i + 1) * cols]);
            yb.push(y[i]);
        }
        let xb = Tensor::matrix(batch_size, cols, xb_data)?;

        vprintln!("[train::train_epoch]   step {}/{}: batch shape=[{},{}]", step+1, steps, batch_size, cols);

        // Forward
        let logits = net.forward(&xb)?;
        if verbose::is_verbose() {
            verbose::check_nan_inf(&logits.data, &format!("train_epoch step {} forward logits", step+1));
        }

        // Loss
        let (loss, dlogits) = softmax_cross_entropy(&logits, &yb)?;
        vprintln!("[train::train_epoch]   step {}/{}: loss={:.6}", step+1, steps, loss);

        if verbose::is_verbose() {
            if loss.is_nan() {
                println!("[ferrum_core::WARN] ⚠️  NaN loss detected at step {}! Training may diverge.", step+1);
            }
            if loss.is_infinite() {
                println!("[ferrum_core::WARN] ⚠️  Infinite loss detected at step {}! Training may diverge.", step+1);
            }
            verbose::check_nan_inf(&dlogits.data, &format!("train_epoch step {} dlogits", step+1));
        }

        // Backward
        net.backward(&dlogits)?;

        // Step
        net.step(opt)?;

        total_loss += loss;
        batches += 1;

        if verbose::is_verbose() {
            let step_ms = step_start.elapsed().as_secs_f64() * 1000.0;
            vprintln!("[train::train_epoch]   step {}/{}: done in {:.1}ms, running avg loss={:.6}",
                step+1, steps, step_ms, total_loss / batches as f32);
        }
    }

    let epoch_ms = epoch_start.elapsed().as_secs_f64() * 1000.0;
    let avg_loss = total_loss / batches as f32;
    vprintln!("[train::train_epoch] Epoch done in {:.1}ms, mean loss={:.6}", epoch_ms, avg_loss);

    Ok(avg_loss)
}

/// Compute accuracy on a full dataset (no gradient tracking).
pub fn accuracy(net: &mut Net, x: &Tensor, y: &[usize]) -> Result<f32> {
    vprintln!("[train::accuracy] Computing accuracy on {} samples", y.len());
    let logits = net.forward(x)?;
    let preds = crate::ops::argmax_rows(&logits)?;
    let correct = preds.iter().zip(y).filter(|(p, t)| p == t).count();
    let acc = correct as f32 / y.len() as f32;
    vprintln!("[train::accuracy] Result: {}/{} correct = {:.4} ({:.1}%)", correct, y.len(), acc, acc * 100.0);
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
        let m_reg = net.to_inference_task(crate::csv::TaskType::Regression).unwrap();
        assert_eq!(m_reg.len(), 4);
        assert_eq!(m_reg.layers()[3].name(), "Activation(Identity)");

        let m_slm = net.to_inference_task(crate::csv::TaskType::TransformerSLM).unwrap();
        assert_eq!(m_slm.len(), 4);
        assert_eq!(m_slm.layers()[3].name(), "Activation(Softmax)");
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
