//! ferrum_core — zero-dependency neural-network engine (inference + training).
pub mod activation;
pub mod csv;
pub mod error;
pub mod layer;
pub mod loader;
pub mod loss;
pub mod model;
pub mod ops;
pub mod optim;
pub mod rng;
pub mod tensor;
pub mod train;

pub use activation::Activation;
pub use csv::{
    fit_normalizer_with_target, train_val_split, CsvDataset, ModelMetadata, Normalizer, TaskType,
};
pub use error::{InferError, Result};
pub use layer::{ActivationLayer, Layer, Linear};
pub use loader::{from_bytes, load, save, to_bytes};
pub use loss::{mse, softmax_cross_entropy};
pub use model::Sequential;
pub use ops::argmax_rows;
pub use optim::Sgd;
pub use rng::Rng;
pub use tensor::Tensor;
pub use train::{accuracy, train_epoch, Net};
