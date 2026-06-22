//! Activation functions as a serialisable enum.
use crate::error::Result;
use crate::ops;
use crate::tensor::Tensor;
use crate::verbose;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    Identity,
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    /// SiLU / swish: `x · sigmoid(x)`. The gate nonlinearity in SwiGLU FFNs
    /// (Llama / Qwen / Mistral).
    SiLU,
}

impl Activation {
    pub fn apply(&self, input: &Tensor) -> Result<Tensor> {
        vprintln!("[activation::apply] {:?} on shape={:?}", self, input.shape);
        let result = match self {
            Activation::Identity => Ok(input.clone()),
            Activation::ReLU => Ok(input.map(|x| x.max(0.0))),
            Activation::Sigmoid => Ok(input.map(|x| 1.0 / (1.0 + (-x).exp()))),
            Activation::Tanh => Ok(input.map(|x| x.tanh())),
            Activation::Softmax => ops::softmax_rows(input),
            Activation::SiLU => Ok(input.map(|x| x / (1.0 + (-x).exp()))),
        }?;
        if verbose::is_verbose() {
            let (vmin, vmax, vmean) = verbose::stats(&result.data);
            vprintln!("[activation::apply] {:?} output: min={:.6}, max={:.6}, mean={:.6}", self, vmin, vmax, vmean);
            verbose::check_nan_inf(&result.data, &format!("Activation::{:?}", self));
        }
        Ok(result)
    }

    pub fn tag(&self) -> u8 {
        match self {
            Activation::Identity => 0,
            Activation::ReLU => 1,
            Activation::Sigmoid => 2,
            Activation::Tanh => 3,
            Activation::Softmax => 4,
            Activation::SiLU => 5,
        }
    }

    pub fn from_tag(t: u8) -> Option<Self> {
        match t {
            0 => Some(Activation::Identity),
            1 => Some(Activation::ReLU),
            2 => Some(Activation::Sigmoid),
            3 => Some(Activation::Tanh),
            4 => Some(Activation::Softmax),
            5 => Some(Activation::SiLU),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relu_clamps_negatives() {
        let t = Tensor::vector(vec![-2.0, 0.0, 3.0]);
        let out = Activation::ReLU.apply(&t).unwrap();
        assert_eq!(out.data, vec![0.0, 0.0, 3.0]);
    }

    #[test]
    fn sigmoid_at_zero_is_half() {
        let t = Tensor::vector(vec![0.0]);
        let out = Activation::Sigmoid.apply(&t).unwrap();
        assert!((out.data[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn tanh_at_zero_is_zero() {
        let t = Tensor::vector(vec![0.0]);
        assert!((Activation::Tanh.apply(&t).unwrap().data[0]).abs() < 1e-7);
    }

    #[test]
    fn softmax_sums_to_one() {
        let t = Tensor::matrix(1, 4, vec![1.0, 2.0, -1.0, 0.5]).unwrap();
        let p = Activation::Softmax.apply(&t).unwrap();
        let s: f32 = p.data.iter().sum();
        assert!((s - 1.0).abs() < 1e-6);
    }

    #[test]
    fn identity_is_noop() {
        let t = Tensor::vector(vec![1.0, 2.0, 3.0]);
        assert_eq!(Activation::Identity.apply(&t).unwrap(), t);
    }

    #[test]
    fn tag_roundtrips_all_variants() {
        for a in [
            Activation::Identity,
            Activation::ReLU,
            Activation::Sigmoid,
            Activation::Tanh,
            Activation::Softmax,
            Activation::SiLU,
        ] {
            assert_eq!(Activation::from_tag(a.tag()), Some(a));
        }
    }

    #[test]
    fn silu_matches_x_times_sigmoid() {
        // SiLU(0) = 0; SiLU(x) = x·σ(x); monotone-ish and smooth.
        let t = Tensor::vector(vec![0.0, 1.0, -1.0, 2.0]);
        let out = Activation::SiLU.apply(&t).unwrap();
        let expect: Vec<f32> = t.data.iter().map(|&x| x / (1.0 + (-x).exp())).collect();
        assert_eq!(out.data[0], 0.0);
        for (a, b) in out.data.iter().zip(&expect) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn unknown_tag_returns_none() {
        assert_eq!(Activation::from_tag(99), None);
    }
}
