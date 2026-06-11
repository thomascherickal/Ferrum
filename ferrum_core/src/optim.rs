//! SGD with optional momentum. Stateless — each trainable layer owns its
//! own velocity buffers and calls `step` once per parameter tensor.
use crate::error::{InferError, Result};
use crate::tensor::Tensor;
use crate::verbose;

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
            if update.abs() > max_update { max_update = update.abs(); }
            param.data[i] -= update;
        }

        if verbose::is_verbose() {
            vprintln!("[optim::Sgd::step]   max |update|={:.6e}", max_update);
            let (pmin, pmax, pmean) = verbose::stats(&param.data);
            vprintln!("[optim::Sgd::step]   post-step param: min={:.6e}, max={:.6e}, mean={:.6e}", pmin, pmax, pmean);
            verbose::check_nan_inf(&param.data, "Sgd::step param");
            verbose::check_nan_inf(&vel.data, "Sgd::step velocity");
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Adam
// ─────────────────────────────────────────────────────────────────────────────

/// Adam optimizer (Kingma & Ba, 2015) with bias correction. Stateless like
/// `Sgd` — each parameter owns its first/second moment buffers and passes the
/// shared 1-based timestep `t` to every `step` call.
#[derive(Clone, Copy, Debug)]
pub struct Adam {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
}

impl Adam {
    /// Standard defaults: β1=0.9, β2=0.999, ε=1e-8.
    pub fn new(lr: f32) -> Self {
        vprintln!("[optim::Adam::new] lr={}", lr);
        Self { lr, beta1: 0.9, beta2: 0.999, eps: 1e-8 }
    }

    /// m ← β1·m + (1−β1)·g;  v ← β2·v + (1−β2)·g²;
    /// param ← param − lr·m̂/(√v̂ + ε)  with bias-corrected m̂, v̂.
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
            param.data[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
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

    #[test]
    fn adam_zero_timestep_errors() {
        let opt = Adam::new(0.1);
        let mut p = Tensor::vector(vec![1.0]);
        let g = Tensor::vector(vec![1.0]);
        let mut m = Tensor::zeros(vec![1]);
        let mut v = Tensor::zeros(vec![1]);
        assert!(opt.step(0, &mut p, &g, &mut m, &mut v).is_err());
    }
}
