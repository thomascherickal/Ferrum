//! SGD with optional momentum. Stateless — each trainable layer owns its
//! own velocity buffers and calls `step` once per parameter tensor.
use crate::error::{InferError, Result};
use crate::tensor::Tensor;
use crate::verbose;

// ─────────────────────────────────────────────────────────────────────────────
// Gradient clipping (global L2 norm)
// ─────────────────────────────────────────────────────────────────────────────

/// Rescale a set of gradient tensors in place so their **global** L2 norm
/// (the norm of all gradients concatenated into one vector) is at most
/// `max_norm`, and return the pre-clip global norm.
///
/// This is the standard `clip_grad_norm_` operation: if the gradients are
/// already within budget it is a no-op (but still reports the norm); otherwise
/// every gradient is multiplied by `max_norm / norm`, preserving direction
/// while capping magnitude. A non-positive `max_norm` disables clipping
/// (norm is still computed and returned). Bounding the update this way is what
/// prevents a single exploding step from diverging training into `NaN`/`Inf`.
///
/// The sum of squares is accumulated in `f64` so the norm stays accurate even
/// for models with millions of parameters.
pub fn clip_grad_norm(grads: &mut [&mut Tensor], max_norm: f32) -> f32 {
    let mut sumsq = 0.0f64;
    for g in grads.iter() {
        for &x in &g.data {
            sumsq += (x as f64) * (x as f64);
        }
    }
    let norm = sumsq.sqrt() as f32;
    if max_norm > 0.0 && norm.is_finite() && norm > max_norm {
        let scale = max_norm / norm;
        for g in grads.iter_mut() {
            for x in &mut g.data {
                *x *= scale;
            }
        }
        vprintln!(
            "[optim::clip_grad_norm] clipped: norm={:.6e} → max_norm={:.6e} (scale={:.6e})",
            norm,
            max_norm,
            scale
        );
    }
    norm
}

#[derive(Clone, Copy, Debug)]
pub struct Sgd {
    pub lr: f32,
    pub momentum: f32,
}

impl Sgd {
    pub fn new(lr: f32) -> Self {
        vprintln!("[optim::Sgd::new] lr={}", lr);
        Self { lr, momentum: 0.0 }
    }
    pub fn with_momentum(lr: f32, m: f32) -> Self {
        vprintln!("[optim::Sgd::with_momentum] lr={}, momentum={}", lr, m);
        Self { lr, momentum: m }
    }

    /// v ← momentum·v + grad;  param ← param − lr·v
    pub fn step(&self, param: &mut Tensor, grad: &Tensor, vel: &mut Tensor) -> Result<()> {
        if param.shape != grad.shape || param.shape != vel.shape {
            return Err(InferError::DimMismatch(format!(
                "step: param {:?}, grad {:?}, vel {:?} shapes differ",
                param.shape, grad.shape, vel.shape
            )));
        }
        if verbose::is_verbose() {
            let (gmin, gmax, gmean) = verbose::stats(&grad.data);
            vprintln!("[optim::Sgd::step] shape={:?}, lr={}, momentum={}, grad stats: min={:.6e}, max={:.6e}, mean={:.6e}",
                param.shape, self.lr, self.momentum, gmin, gmax, gmean);
        }

        let mut max_update = 0.0f32;
        for i in 0..param.data.len() {
            vel.data[i] = self.momentum * vel.data[i] + grad.data[i];
            let update = self.lr * vel.data[i];
            if update.abs() > max_update {
                max_update = update.abs();
            }
            param.data[i] -= update;
        }

        if verbose::is_verbose() {
            vprintln!("[optim::Sgd::step]   max |update|={:.6e}", max_update);
            let (pmin, pmax, pmean) = verbose::stats(&param.data);
            vprintln!(
                "[optim::Sgd::step]   post-step param: min={:.6e}, max={:.6e}, mean={:.6e}",
                pmin,
                pmax,
                pmean
            );
            verbose::check_nan_inf(&param.data, "Sgd::step param");
            verbose::check_nan_inf(&vel.data, "Sgd::step velocity");
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Learning-rate schedule (warmup + decay)
// ─────────────────────────────────────────────────────────────────────────────

/// Post-warmup decay shape for [`LrSchedule`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LrDecay {
    /// Smooth half-cosine from `base_lr` down to `min_lr`.
    Cosine,
    /// Straight line from `base_lr` down to `min_lr`.
    Linear,
}

/// Learning-rate schedule: a linear warmup from 0 up to `base_lr` over the
/// first `warmup_steps`, then a `decay` (cosine or linear) down to `min_lr` by
/// `total_steps`. This is the standard transformer recipe — warmup keeps the
/// early, high-variance Adam steps from destabilising the model, and decay lets
/// it settle near a minimum. A fixed learning rate (the previous behaviour)
/// makes larger models fragile and slower to converge.
///
/// Steps are 1-based (matching the optimizer's timestep `t`): `lr_at(1)` is the
/// first step. `lr_at` is a pure function of the step, so it is reproducible and
/// independent of how the optimizer is driven.
#[derive(Clone, Copy, Debug)]
pub struct LrSchedule {
    pub base_lr: f32,
    pub min_lr: f32,
    pub warmup_steps: u64,
    pub total_steps: u64,
    pub decay: LrDecay,
}

impl LrSchedule {
    /// Warmup-then-cosine-decay to zero — the most common transformer schedule.
    pub fn warmup_cosine(base_lr: f32, warmup_steps: u64, total_steps: u64) -> Self {
        Self {
            base_lr,
            min_lr: 0.0,
            warmup_steps,
            total_steps,
            decay: LrDecay::Cosine,
        }
    }

    /// Warmup-then-linear-decay to zero.
    pub fn warmup_linear(base_lr: f32, warmup_steps: u64, total_steps: u64) -> Self {
        Self {
            base_lr,
            min_lr: 0.0,
            warmup_steps,
            total_steps,
            decay: LrDecay::Linear,
        }
    }

    /// Learning rate at 1-based `step`. Ramps 0 → `base_lr` across the warmup,
    /// peaks at `base_lr` exactly at `warmup_steps`, decays to `min_lr` by
    /// `total_steps`, and stays at `min_lr` thereafter.
    pub fn lr_at(&self, step: u64) -> f32 {
        let step = step.max(1);
        if step <= self.warmup_steps {
            // Linear warmup; reaches base_lr exactly at warmup_steps.
            return self.base_lr * (step as f32 / self.warmup_steps.max(1) as f32);
        }
        if step >= self.total_steps {
            return self.min_lr;
        }
        let decay_steps = self.total_steps.saturating_sub(self.warmup_steps).max(1);
        let progress = (step - self.warmup_steps) as f32 / decay_steps as f32; // (0, 1)
        let factor = match self.decay {
            LrDecay::Linear => 1.0 - progress,
            LrDecay::Cosine => 0.5 * (1.0 + (std::f32::consts::PI * progress).cos()),
        };
        self.min_lr + (self.base_lr - self.min_lr) * factor
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Adam
// ─────────────────────────────────────────────────────────────────────────────

/// Adam optimizer (Kingma & Ba, 2015) with bias correction, plus optional
/// **decoupled weight decay** (AdamW; Loshchilov & Hutter, 2019). Stateless like
/// `Sgd` — each parameter owns its first/second moment buffers and passes the
/// shared 1-based timestep `t` to every `step` call.
///
/// `weight_decay == 0.0` (the default) is plain Adam, bit-identical to before.
/// A positive value pulls each weight toward zero by `lr · weight_decay · w`
/// *in addition to* the Adam step — decoupled from the gradient/`v̂` scaling,
/// which is what distinguishes AdamW from naive L2 regularization.
#[derive(Clone, Copy, Debug)]
pub struct Adam {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
}

impl Adam {
    /// Standard defaults: β1=0.9, β2=0.999, ε=1e-8, no weight decay.
    pub fn new(lr: f32) -> Self {
        vprintln!("[optim::Adam::new] lr={}", lr);
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        }
    }

    /// AdamW with decoupled `weight_decay` (see the struct docs).
    pub fn with_weight_decay(lr: f32, weight_decay: f32) -> Self {
        Self {
            weight_decay,
            ..Self::new(lr)
        }
    }

    /// m ← β1·m + (1−β1)·g;  v ← β2·v + (1−β2)·g²;
    /// param ← param − lr·(m̂/(√v̂ + ε) + weight_decay·param)  with bias-corrected
    /// m̂, v̂ (the weight-decay term is the decoupled AdamW addition).
    pub fn step(
        &self,
        t: u64,
        param: &mut Tensor,
        grad: &Tensor,
        m: &mut Tensor,
        v: &mut Tensor,
    ) -> Result<()> {
        if param.shape != grad.shape || param.shape != m.shape || param.shape != v.shape {
            return Err(InferError::DimMismatch(format!(
                "adam step: param {:?}, grad {:?}, m {:?}, v {:?} shapes differ",
                param.shape, grad.shape, m.shape, v.shape
            )));
        }
        if t == 0 {
            return Err(InferError::DimMismatch("adam timestep must be ≥ 1".into()));
        }
        let bc1 = 1.0 - self.beta1.powi(t.min(i32::MAX as u64) as i32);
        let bc2 = 1.0 - self.beta2.powi(t.min(i32::MAX as u64) as i32);
        for i in 0..param.data.len() {
            let g = grad.data[i];
            m.data[i] = self.beta1 * m.data[i] + (1.0 - self.beta1) * g;
            v.data[i] = self.beta2 * v.data[i] + (1.0 - self.beta2) * g * g;
            let m_hat = m.data[i] / bc1;
            let v_hat = v.data[i] / bc2;
            // Keep the plain-Adam term in its original form so a zero
            // weight_decay is bit-identical to pre-AdamW; only add the decoupled
            // decay when it is non-zero.
            let mut update = self.lr * m_hat / (v_hat.sqrt() + self.eps);
            if self.weight_decay != 0.0 {
                update += self.lr * self.weight_decay * param.data[i];
            }
            param.data[i] -= update;
        }
        if verbose::is_verbose() {
            verbose::check_nan_inf(&param.data, "Adam::step param");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_sgd_descends() {
        let opt = Sgd::new(0.1);
        let mut p = Tensor::vector(vec![2.0]);
        let g = Tensor::vector(vec![10.0]);
        let mut v = Tensor::zeros(vec![1]);
        opt.step(&mut p, &g, &mut v).unwrap();
        assert!((p.data[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn momentum_accumulates_over_steps() {
        let opt = Sgd::with_momentum(1.0, 0.9);
        let mut p = Tensor::vector(vec![0.0]);
        let g = Tensor::vector(vec![1.0]);
        let mut v = Tensor::zeros(vec![1]);
        opt.step(&mut p, &g, &mut v).unwrap(); // v=1,   p=-1
        opt.step(&mut p, &g, &mut v).unwrap(); // v=1.9, p=-2.9
        assert!((v.data[0] - 1.9).abs() < 1e-6);
        assert!((p.data[0] + 2.9).abs() < 1e-6);
    }

    #[test]
    fn shape_mismatch_errors() {
        let opt = Sgd::new(0.1);
        let mut p = Tensor::vector(vec![1.0, 2.0]);
        let g = Tensor::vector(vec![1.0]);
        let mut v = Tensor::zeros(vec![2]);
        assert!(opt.step(&mut p, &g, &mut v).is_err());
    }

    #[test]
    fn zero_lr_no_update() {
        let opt = Sgd::new(0.0);
        let mut p = Tensor::vector(vec![5.0]);
        let g = Tensor::vector(vec![100.0]);
        let mut v = Tensor::zeros(vec![1]);
        opt.step(&mut p, &g, &mut v).unwrap();
        assert_eq!(p.data[0], 5.0);
    }

    #[test]
    fn adam_first_step_moves_by_lr() {
        // With bias correction, the very first Adam step is ≈ lr (for eps≈0).
        let opt = Adam::new(0.1);
        let mut p = Tensor::vector(vec![1.0]);
        let g = Tensor::vector(vec![3.0]);
        let mut m = Tensor::zeros(vec![1]);
        let mut v = Tensor::zeros(vec![1]);
        opt.step(1, &mut p, &g, &mut m, &mut v).unwrap();
        assert!((p.data[0] - 0.9).abs() < 1e-4, "got {}", p.data[0]);
    }

    #[test]
    fn adam_minimises_quadratic() {
        // Minimise f(x) = (x - 3)², gradient 2(x - 3).
        let opt = Adam::new(0.1);
        let mut p = Tensor::vector(vec![0.0]);
        let mut m = Tensor::zeros(vec![1]);
        let mut v = Tensor::zeros(vec![1]);
        for t in 1..=500u64 {
            let g = Tensor::vector(vec![2.0 * (p.data[0] - 3.0)]);
            opt.step(t, &mut p, &g, &mut m, &mut v).unwrap();
        }
        assert!((p.data[0] - 3.0).abs() < 0.01, "converged to {}", p.data[0]);
    }

    #[test]
    fn adam_shape_mismatch_errors() {
        let opt = Adam::new(0.1);
        let mut p = Tensor::vector(vec![1.0, 2.0]);
        let g = Tensor::vector(vec![1.0]);
        let mut m = Tensor::zeros(vec![2]);
        let mut v = Tensor::zeros(vec![2]);
        assert!(opt.step(1, &mut p, &g, &mut m, &mut v).is_err());
    }

    // ── AdamW weight decay (T7) ───────────────────────────────────────────────

    #[test]
    fn adamw_decoupled_decay_shrinks_toward_zero() {
        // With a zero gradient the Adam step is 0, so only the decoupled decay
        // acts: param -= lr · weight_decay · param.
        let opt = Adam::with_weight_decay(0.1, 0.5);
        let mut p = Tensor::vector(vec![10.0]);
        let g = Tensor::zeros(vec![1]);
        let mut m = Tensor::zeros(vec![1]);
        let mut v = Tensor::zeros(vec![1]);
        opt.step(1, &mut p, &g, &mut m, &mut v).unwrap();
        // 10 − 0.1·0.5·10 = 9.5
        assert!((p.data[0] - 9.5).abs() < 1e-5, "got {}", p.data[0]);
    }

    #[test]
    fn adamw_zero_decay_matches_plain_adam() {
        let wd0 = Adam::with_weight_decay(0.1, 0.0);
        let plain = Adam::new(0.1);
        let run = |opt: &Adam| {
            let mut p = Tensor::vector(vec![1.0]);
            let g = Tensor::vector(vec![3.0]);
            let mut m = Tensor::zeros(vec![1]);
            let mut v = Tensor::zeros(vec![1]);
            opt.step(1, &mut p, &g, &mut m, &mut v).unwrap();
            p.data[0]
        };
        assert_eq!(run(&wd0), run(&plain));
    }

    #[test]
    fn adam_zero_timestep_errors() {
        let opt = Adam::new(0.1);
        let mut p = Tensor::vector(vec![1.0]);
        let g = Tensor::vector(vec![1.0]);
        let mut m = Tensor::zeros(vec![1]);
        let mut v = Tensor::zeros(vec![1]);
        assert!(opt.step(0, &mut p, &g, &mut m, &mut v).is_err());
    }

    // ── Gradient clipping (T1) ────────────────────────────────────────────────

    #[test]
    fn clip_reports_global_norm_across_tensors() {
        // Two tensors: [3,4] and [12] → global norm = √(9+16+144) = 13.
        let mut a = Tensor::vector(vec![3.0, 4.0]);
        let mut b = Tensor::vector(vec![12.0]);
        let mut grads: Vec<&mut Tensor> = vec![&mut a, &mut b];
        let norm = clip_grad_norm(&mut grads, 1e9); // budget huge → no scaling
        assert!((norm - 13.0).abs() < 1e-5, "global norm = {norm}");
        // Within budget → untouched.
        assert_eq!(a.data, vec![3.0, 4.0]);
        assert_eq!(b.data, vec![12.0]);
    }

    #[test]
    fn clip_rescales_to_max_norm_preserving_direction() {
        // Global norm 13, clip to 6.5 → every component halved, norm becomes 6.5.
        let mut a = Tensor::vector(vec![3.0, 4.0]);
        let mut b = Tensor::vector(vec![12.0]);
        let mut grads: Vec<&mut Tensor> = vec![&mut a, &mut b];
        let pre = clip_grad_norm(&mut grads, 6.5);
        assert!((pre - 13.0).abs() < 1e-5);
        assert!((a.data[0] - 1.5).abs() < 1e-5);
        assert!((a.data[1] - 2.0).abs() < 1e-5);
        assert!((b.data[0] - 6.0).abs() < 1e-5);
        let post = (a.data[0].powi(2) + a.data[1].powi(2) + b.data[0].powi(2)).sqrt();
        assert!((post - 6.5).abs() < 1e-4, "post-clip norm = {post}");
    }

    #[test]
    fn clip_noop_when_within_budget() {
        let mut a = Tensor::vector(vec![0.3, 0.4]); // norm 0.5
        let mut grads: Vec<&mut Tensor> = vec![&mut a];
        let norm = clip_grad_norm(&mut grads, 1.0);
        assert!((norm - 0.5).abs() < 1e-6);
        assert_eq!(a.data, vec![0.3, 0.4]); // unchanged
    }

    #[test]
    fn clip_disabled_for_nonpositive_max_norm() {
        let mut a = Tensor::vector(vec![3.0, 4.0]); // norm 5
        let mut grads: Vec<&mut Tensor> = vec![&mut a];
        let norm = clip_grad_norm(&mut grads, 0.0);
        assert!((norm - 5.0).abs() < 1e-6);
        assert_eq!(a.data, vec![3.0, 4.0]); // 0 budget disables clipping
    }

    // ── Learning-rate schedule (T2) ───────────────────────────────────────────

    #[test]
    fn schedule_warmup_ramps_linearly_to_base() {
        let s = LrSchedule::warmup_cosine(0.1, 10, 100);
        // Linear ramp 0 → base across the warmup; peaks at base at warmup_steps.
        assert!((s.lr_at(1) - 0.01).abs() < 1e-6);
        assert!((s.lr_at(5) - 0.05).abs() < 1e-6);
        assert!((s.lr_at(10) - 0.1).abs() < 1e-6, "peak at warmup end");
        // Just past warmup the LR has begun to decay (≤ base).
        assert!(s.lr_at(11) < 0.1);
    }

    #[test]
    fn cosine_decays_from_base_to_min() {
        let s = LrSchedule::warmup_cosine(0.2, 10, 110); // 100 decay steps
                                                         // Midpoint of decay (step 60) → halfway down for cosine: base/2.
        assert!(
            (s.lr_at(60) - 0.1).abs() < 1e-3,
            "cosine midpoint = {}",
            s.lr_at(60)
        );
        // Reaches min_lr (0) by total_steps and stays there.
        assert!(s.lr_at(110).abs() < 1e-6);
        assert!(s.lr_at(500).abs() < 1e-6);
        // Monotonic non-increasing through the decay phase.
        let mut prev = s.lr_at(10);
        for step in 11..=110 {
            let cur = s.lr_at(step);
            assert!(cur <= prev + 1e-6, "decay not monotonic at {step}");
            prev = cur;
        }
    }

    #[test]
    fn linear_decay_hits_midpoint_and_floor() {
        let s = LrSchedule {
            base_lr: 1.0,
            min_lr: 0.2,
            warmup_steps: 0,
            total_steps: 100,
            decay: LrDecay::Linear,
        };
        // No warmup: step 1 ≈ base (one step into a 100-step linear decay).
        assert!((s.lr_at(1) - 0.992).abs() < 1e-3);
        // Halfway → halfway between base and min.
        assert!(
            (s.lr_at(50) - 0.6).abs() < 1e-3,
            "linear midpoint = {}",
            s.lr_at(50)
        );
        // Floors at min_lr.
        assert!((s.lr_at(100) - 0.2).abs() < 1e-6);
        assert!((s.lr_at(200) - 0.2).abs() < 1e-6);
    }

    #[test]
    fn schedule_clamps_step_below_one() {
        let s = LrSchedule::warmup_linear(0.5, 4, 40);
        // Step 0 is clamped to step 1.
        assert_eq!(s.lr_at(0), s.lr_at(1));
    }
}
