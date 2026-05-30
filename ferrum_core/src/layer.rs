//! The `Layer` trait and its two inference implementations.
use crate::activation::Activation;
use crate::error::{InferError, Result};
use crate::ops;
use crate::tensor::Tensor;
use std::any::Any;

/// Everything a layer must provide. `as_any` allows the loader to downcast
/// a trait object back to its concrete type for serialisation.
pub trait Layer {
    fn forward(&self, input: &Tensor) -> Result<Tensor>;
    fn name(&self) -> String;
    fn as_any(&self) -> &dyn Any;
}

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
        ops::add_bias(&ops::matmul(input, &self.weight)?, &self.bias)
    }
    fn name(&self) -> String {
        format!("Linear({}→{})", self.in_f, self.out_f)
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_linear(n: usize) -> Linear {
        let mut w = vec![0.0f32; n * n];
        for i in 0..n {
            w[i * n + i] = 1.0;
        }
        Linear::new(n, n, w, vec![0.0; n]).unwrap()
    }

    #[test]
    fn linear_identity_transform() {
        let l = identity_linear(3);
        let x = Tensor::row(vec![1.0, 2.0, 3.0]).unwrap();
        let y = l.forward(&x).unwrap();
        assert_eq!(y.data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn linear_applies_bias() {
        let l = Linear::new(2, 2, vec![1.0, 0.0, 0.0, 1.0], vec![10.0, 20.0]).unwrap();
        let x = Tensor::row(vec![0.0, 0.0]).unwrap();
        assert_eq!(l.forward(&x).unwrap().data, vec![10.0, 20.0]);
    }

    #[test]
    fn linear_rejects_wrong_input_width() {
        let l = Linear::new(2, 2, vec![0.0; 4], vec![0.0, 0.0]).unwrap();
        let x = Tensor::row(vec![1.0, 2.0, 3.0]).unwrap(); // width 3, expects 2
        assert!(l.forward(&x).is_err());
    }

    #[test]
    fn linear_bias_length_mismatch() {
        assert!(Linear::new(2, 2, vec![0.0; 4], vec![0.0]).is_err());
    }

    #[test]
    fn linear_output_shape() {
        let l = Linear::new(4, 3, vec![0.0; 12], vec![0.0; 3]).unwrap();
        let x = Tensor::matrix(2, 4, vec![0.0; 8]).unwrap();
        assert_eq!(l.forward(&x).unwrap().shape, vec![2, 3]);
    }

    #[test]
    fn activation_layer_relu() {
        let l = ActivationLayer::new(Activation::ReLU);
        let x = Tensor::row(vec![-1.0, 0.5]).unwrap();
        assert_eq!(l.forward(&x).unwrap().data, vec![0.0, 0.5]);
    }

    #[test]
    fn linear_name_contains_dims() {
        let l = Linear::new(4, 3, vec![0.0; 12], vec![0.0; 3]).unwrap();
        assert!(l.name().contains("4") && l.name().contains("3"));
    }
}
