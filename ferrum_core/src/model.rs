//! Sequential model: an ordered pipeline of layers.
use crate::error::Result;
use crate::layer::Layer;
use crate::tensor::Tensor;
use crate::verbose;

pub struct Sequential {
    layers: Vec<Box<dyn Layer>>,
}

impl Sequential {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn with(mut self, l: Box<dyn Layer>) -> Self {
        vprintln!("[model::Sequential::with] Adding layer: {}", l.name());
        self.layers.push(l);
        self
    }
    pub fn push(&mut self, l: Box<dyn Layer>) {
        vprintln!("[model::Sequential::push] Adding layer: {}", l.name());
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
        vprintln!(
            "[model::Sequential::forward] input shape={:?}, {} layers",
            input.shape,
            self.layers.len()
        );
        let mut cur = input.clone();
        for (i, l) in self.layers.iter().enumerate() {
            cur = l.forward(&cur)?;
            if verbose::is_verbose() {
                let (vmin, vmax, vmean) = verbose::stats(&cur.data);
                vprintln!("[model::Sequential::forward]   [{}/{}] {} → shape={:?}, stats: min={:.6}, max={:.6}, mean={:.6}",
                    i+1, self.layers.len(), l.name(), cur.shape, vmin, vmax, vmean);
                verbose::check_nan_inf(
                    &cur.data,
                    &format!("Sequential::forward layer[{}] {}", i, l.name()),
                );
            }
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
    fn is_empty_on_new() {
        assert!(Sequential::new().is_empty());
    }

    #[test]
    fn not_empty_after_push() {
        let mut m = Sequential::new();
        m.push(Box::new(ActivationLayer::new(Activation::ReLU)));
        assert!(!m.is_empty());
    }

    #[test]
    fn default_is_same_as_new() {
        let d = Sequential::default();
        assert_eq!(d.len(), Sequential::new().len());
    }

    #[test]
    fn layers_slice_has_correct_len() {
        let m = two_layer_model();
        assert_eq!(m.layers().len(), 2);
    }
}
