use axum::body::Body;
use axum::http::{Request, StatusCode};
use ferrum_core::{GenerativeSLM, Rng, TransformerConfig};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use slm_server::{app, info_for, Limits};
use tower::ServiceExt;

fn tiny_model() -> GenerativeSLM {
    let corpus = "the cat sat on the mat. the dog sat on the log. ".repeat(20);
    let cfg = TransformerConfig {
        context_len: 8,
        embed_dim: 16,
        num_heads: 2,
        num_blocks: 1,
        hidden_dim: 32,
        epochs: 2,
        vocab_size: 270,
        ..TransformerConfig::default()
    };
    GenerativeSLM::train_transformer_config(&corpus, &cfg, &mut Rng::new(3), |_, _| {}).unwrap()
}

fn router() -> axum::Router {
    let m = tiny_model();
    let bytes = m.to_bytes_quantized().unwrap();
    let info = info_for(&m, &bytes);
    app(m, info, Limits::default())
}

async fn call(r: &axum::Router, method: &str, uri: &str) -> (StatusCode, Value) {
    let req = Request::builder()
        .method(method)
        .uri(uri)
        .body(Body::empty())
        .unwrap();
    let resp = r.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let v = serde_json::from_slice(&bytes)
        .unwrap_or(Value::String(String::from_utf8_lossy(&bytes).into()));
    (status, v)
}

#[tokio::test]
async fn health_and_info() {
    let r = router();
    let (s, v) = call(&r, "GET", "/health").await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(v, json!("ok"));
    let (s, v) = call(&r, "GET", "/v1/info").await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(v["context_len"], 8);
    assert_eq!(v["vocab"], 270);
    assert_eq!(v["tokenizer"], "bpe");
    assert_eq!(v["model_sha256"].as_str().unwrap().len(), 64);
    // ponytail: `> 0` once Sequential::num_params lands (Phase 1A).
    assert!(v["params"].is_u64());
    assert!(v["quant"].is_string());
}

#[tokio::test]
async fn unknown_route_is_404_and_wrong_method_405() {
    let r = router();
    assert_eq!(call(&r, "GET", "/nope").await.0, StatusCode::NOT_FOUND);
    assert_eq!(
        call(&r, "POST", "/health").await.0,
        StatusCode::METHOD_NOT_ALLOWED
    );
}
