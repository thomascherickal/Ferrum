//! Sequential model: an ordered pipeline of layers.
use crate::error::Result;
use crate::layer::Layer;
use crate::tensor::Tensor;

pub struct Sequential {
    layers: Vec<Box<dyn Layer>>,
}

impl Sequential {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn with(mut self, l: Box<dyn Layer>) -> Self {
        self.layers.push(l);
        self
    }
    pub fn push(&mut self, l: Box<dyn Layer>) {
        self.layers.push(l);
    }

    pub fn len(&self) -> usize {
        self.layers.len()
    }
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }
    pub fn layers(&self) -> &[Box<dyn Layer>] {
        &self.layers
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut cur = input.clone();
        for l in &self.layers {
            cur = l.forward(&cur)?;
        }
        Ok(cur)
    }

    pub fn summary(&self) -> String {
        let mut s = format!("Sequential ({} layers)\n", self.layers.len());
        for (i, l) in self.layers.iter().enumerate() {
            s.push_str(&format!("  [{i}] {}\n", l.name()));
        }
        s
    }
}

impl Default for Sequential {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::Activation;
    use crate::layer::{ActivationLayer, Linear};

    fn two_layer_model() -> Sequential {
        // Linear(2→2): double each feature, bias shifts second to negative
        let lin = Linear::new(2, 2, vec![2.0, 0.0, 0.0, 2.0], vec![0.0, -10.0]).unwrap();
        Sequential::new()
            .with(Box::new(lin))
            .with(Box::new(ActivationLayer::new(Activation::ReLU)))
    }

    #[test]
    fn forward_threads_through_layers() {
        let model = two_layer_model();
        let x = Tensor::row(vec![3.0, 1.0]).unwrap();
        // After linear: [6, 2-10]=[6,-8]; after ReLU: [6, 0]
        assert_eq!(model.forward(&x).unwrap().data, vec![6.0, 0.0]);
    }

    #[test]
    fn len_counts_layers() {
        assert_eq!(two_layer_model().len(), 2);
    }

    #[test]
    fn empty_model_is_noop() {
        let m = Sequential::new();
        let x = Tensor::row(vec![1.0, 2.0]).unwrap();
        assert_eq!(m.forward(&x).unwrap().data, vec![1.0, 2.0]);
    }

    #[test]
    fn summary_contains_layer_names() {
        let s = two_layer_model().summary();
        assert!(s.contains("Linear") && s.contains("Activation"));
    }

    #[test]
    fn push_increases_len() {
        let mut m = Sequential::new();
        assert_eq!(m.len(), 0);
        m.push(Box::new(ActivationLayer::new(Activation::ReLU)));
        assert_eq!(m.len(), 1);
    }

    #[test]
    fn batch_forward() {
        // Same model, batch of 3 examples
        let model = two_layer_model();
        let x = Tensor::matrix(3, 2, vec![1.0, 1.0, 3.0, 1.0, 0.0, 0.0]).unwrap();
        let y = model.forward(&x).unwrap();
        assert_eq!(y.shape, vec![3, 2]);
    }
}
