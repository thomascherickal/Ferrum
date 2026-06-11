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
}
