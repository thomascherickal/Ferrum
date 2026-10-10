//! HTTP inference server for Ferrum SLMs: loglikelihood, rolling
//! loglikelihood, and generation — the three primitives eval harnesses such as
//! LightEval need. One thread owns the model (it is not `Sync`); requests are
//! queued to it over a channel and answered through oneshot replies.
#![forbid(unsafe_code)]

pub mod cli;

use axum::extract::{DefaultBodyLimit, State};
use axum::routing::get;
use axum::{Json, Router};
use ferrum_core::GenerativeSLM;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::sync::Arc;

/// Request limits, enforced before any model compute.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Limits {
    pub max_batch: usize,
    pub max_chars: usize,
    pub max_body: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_batch: 64,
            max_chars: 65_536,
            max_body: 8 << 20,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Info {
    pub params: usize,
    pub context_len: usize,
    pub vocab: usize,
    pub quant: String,
    pub tokenizer: String,
    pub model_sha256: String,
}

/// Parameter count of `model`.
// ponytail: real count after Phase 1A merge (Sequential::num_params).
fn num_params(_model: &GenerativeSLM) -> usize {
    0
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Describe `model`, loaded from `bytes` (hashed for `model_sha256`).
pub fn info_for(model: &GenerativeSLM, bytes: &[u8]) -> Info {
    Info {
        params: num_params(model),
        context_len: model.meta.input_dim,
        vocab: model.meta.output_dim,
        quant: ferrum_core::loader::gguf_quant(bytes).unwrap_or_else(|_| "unknown".into()),
        tokenizer: if model.meta.tokenizer_state.is_empty() {
            "char".into()
        } else {
            "bpe".into()
        },
        model_sha256: sha256_hex(bytes),
    }
}

#[derive(Clone)]
struct AppState {
    info: Arc<Info>,
}

/// Build the router.
pub fn app(_model: GenerativeSLM, info: Info, limits: Limits) -> Router {
    let state = AppState {
        info: Arc::new(info),
    };
    Router::new()
        .route("/health", get(|| async { Json("ok") }))
        .route(
            "/v1/info",
            get(|State(s): State<AppState>| async move { Json((*s.info).clone()) }),
        )
        .layer(DefaultBodyLimit::max(limits.max_body))
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_core::{Rng, TransformerConfig};

    fn model(vocab_size: usize) -> GenerativeSLM {
        let cfg = TransformerConfig {
            context_len: 8,
            embed_dim: 8,
            num_heads: 2,
            num_blocks: 1,
            hidden_dim: 16,
            epochs: 1,
            vocab_size,
            ..TransformerConfig::default()
        };
        GenerativeSLM::train_transformer_config(
            &"abcabc abc\n".repeat(30),
            &cfg,
            &mut Rng::new(1),
            |_, _| {},
        )
        .unwrap()
    }

    #[test]
    fn limits_default_values() {
        let d = Limits::default();
        assert_eq!(
            (d.max_batch, d.max_chars, d.max_body),
            (64, 65_536, 8 << 20)
        );
    }

    #[test]
    fn sha256_known_vectors() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn num_params_placeholder_is_zero() {
        assert_eq!(num_params(&model(0)), 0);
    }

    #[test]
    fn info_for_bpe_model() {
        let m = model(270);
        let bytes = m.to_bytes_quantized().unwrap();
        let i = info_for(&m, &bytes);
        assert_eq!(
            (i.context_len, i.vocab, i.tokenizer.as_str()),
            (8, m.meta.output_dim, "bpe")
        );
        assert!(i.vocab > 256);
        assert_eq!(i.model_sha256.len(), 64);
        assert_eq!(i.model_sha256, sha256_hex(&bytes));
    }

    #[test]
    fn info_for_char_model() {
        let m = model(0);
        let i = info_for(&m, &m.to_bytes_quantized().unwrap());
        assert_eq!(i.tokenizer, "char");
    }

    #[test]
    fn info_quant_falls_back_to_unknown_on_non_gguf_bytes() {
        let m = model(0);
        let i = info_for(&m, b"not a model file");
        assert_eq!(i.quant, "unknown");
        assert_eq!(i.model_sha256, sha256_hex(b"not a model file"));
    }

    #[test]
    fn info_quant_reported_for_gguf_bytes() {
        let m = model(0);
        let bytes = m.to_bytes_quantized().unwrap();
        let want = ferrum_core::loader::gguf_quant(&bytes).unwrap_or_else(|_| "unknown".into());
        assert_eq!(info_for(&m, &bytes).quant, want);
    }
}
