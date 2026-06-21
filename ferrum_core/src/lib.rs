//! # ferrum_core — Edge Transformer & MLP Engine
//!
//! Zero-dependency, pure-Rust library for building, training, and running
//! **hand-crafted causal Transformer models**, Small Language Models, and
//! classical MLPs on CPU-only, edge, and WebAssembly targets. No GPU required.
//! No external crates — `std` only.
//!
//! ## Architecture at a glance
//!
//! ```text
//! Tensor ──► ops (matmul, softmax, …)  ──► parallel (std-only persistent CPU worker pool)
//!      │
//!      └──► Layer trait
//!             ├── Linear          (y = xW + b)
//!             ├── ActivationLayer (ReLU / Softmax / …)
//!             ├── LayerNorm       (per-row normalisation)
//!             ├── Embedding       (token + positional lookup)
//!             └── TransformerBlock (causal multi-head self-attention + FFN)
//!
//! Sequential ──► ordered pipeline of Layers
//!
//! loader     ──► FINF v4/v5 binary format (save / load, int8 quantized)
//! quant      ──► int8 fake-quantization for QAT and serialization
//! tokenizer  ──► ByteBpeTokenizer (byte-level BPE; char-level fallback)
//! csv        ──► CsvDataset, Normalizer, ModelMetadata
//! train      ──► Net (trainable MLP), train_epoch, accuracy
//! train_transformer ──► TransformerNet, train_transformer_epoch
//! loss       ──► softmax_cross_entropy, mse
//! optim      ──► Sgd (with optional momentum), Adam
//! rng        ──► seeded xorshift64* PRNG (deterministic)
//! slm        ──► GenerativeSLM: train / train_embedded / train_transformer (int8 QAT, BPE) / generate
//! ```
//!
//! ## Quick start — Transformer inference
//!
//! ```rust,no_run
//! use ferrum_core::*;
//! use ferrum_core::layer::{Embedding, LayerNorm, TransformerBlock};
//! use ferrum_core::model::Sequential;
//!
//! // Load a pre-trained model from FINF v4 bytes
//! let bytes = std::fs::read("model.bin").unwrap();
//! let (model, _norm, meta) = from_bytes(&bytes).unwrap();
//!
//! // Build a context of token IDs and run one forward pass
//! let context = Tensor::matrix(1, meta.input_dim,
//!     vec![0.0f32; meta.input_dim]).unwrap();
//! let logits = model.forward(&context).unwrap();
//! ```

#![forbid(unsafe_code)]

#[macro_use]
pub mod verbose;

pub mod activation;
pub mod csv;
pub mod dataset;
pub mod error;
pub mod layer;
pub mod loader;
pub mod loss;
pub mod model;
pub mod ops;
pub mod optim;
pub mod parallel;
pub mod quant;
pub mod rng;
pub mod slm;
pub mod tensor;
pub mod tokenizer;
pub mod train;
pub mod train_transformer;

pub use activation::Activation;
pub use csv::{
    fit_normalizer_with_target, train_val_split, CsvDataset, ModelMetadata, Normalizer, TaskType,
};
pub use dataset::{clean_corpus, corpus_stats, validate_for_training, CleanOptions, CorpusStats};
pub use error::{InferError, Result};
pub use layer::{
    ActivationLayer, Embedding, Flatten, KvCache, Layer, LayerNorm, Linear, TransformerBlock,
};
pub use loader::{from_bytes, load, save, save_quantized, to_bytes, to_bytes_quantized};
pub use loss::{mse, softmax_cross_entropy};
pub use model::Sequential;
pub use ops::argmax_rows;
pub use optim::{Adam, Sgd};
pub use parallel::num_threads;
pub use quant::{fake_quantize_int8, QUANT_MIN_LEN};
pub use rng::Rng;
pub use slm::{Evaluation, GenerativeSLM, TransformerConfig};
pub use tensor::Tensor;
pub use tokenizer::ByteBpeTokenizer;
pub use train::{accuracy, train_epoch, EmbedT, Net};
pub use train_transformer::{
    train_transformer_epoch, train_transformer_epoch_threaded, TransformerNet,
};
pub use verbose::{clear_log_sink, is_verbose, log_line, set_log_sink, set_verbose};
