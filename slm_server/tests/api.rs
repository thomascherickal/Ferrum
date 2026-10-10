use axum::body::Body;
use axum::http::{Request, StatusCode};
use ferrum_core::{GenerativeSLM, Rng, TransformerConfig};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use slm_server::{app, Limits};
use tower::ServiceExt;

fn train(corpus: &str, vocab_size: usize, embed_dim: usize) -> GenerativeSLM {
    let cfg = TransformerConfig {
        context_len: 8,
        embed_dim,
        num_heads: 2,
        num_blocks: 1,
        hidden_dim: embed_dim * 2,
        epochs: 2,
        vocab_size,
        ..TransformerConfig::default()
    };
    GenerativeSLM::train_transformer_config(corpus, &cfg, &mut Rng::new(3), |_, _| {}).unwrap()
}

/// Model-file bytes of the BPE test model.
fn tiny_bytes() -> Vec<u8> {
    let corpus = "the cat sat on the mat. the dog sat on the log. ".repeat(20);
    train(&corpus, 270, 16).to_bytes_quantized().unwrap()
}

/// The model exactly as the server loads it (from the quantized bytes).
fn reference() -> GenerativeSLM {
    GenerativeSLM::from_bytes(&tiny_bytes()).unwrap()
}

fn router() -> axum::Router {
    router_with(Limits::default())
}

fn router_with(limits: Limits) -> axum::Router {
    app(tiny_bytes(), limits).unwrap().0
}

/// Char-level model (no BPE): unknown characters are a model error.
fn char_router() -> axum::Router {
    let m = train(&"abcabc abc\n".repeat(30), 0, 8);
    app(m.to_bytes_quantized().unwrap(), Limits::default())
        .unwrap()
        .0
}

async fn send(r: &axum::Router, req: Request<Body>) -> (StatusCode, Value) {
    let resp = r.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let v = serde_json::from_slice(&bytes)
        .unwrap_or(Value::String(String::from_utf8_lossy(&bytes).into()));
    (status, v)
}

async fn call(
    r: &axum::Router,
    method: &str,
    uri: &str,
    body: Option<Value>,
) -> (StatusCode, Value) {
    let req = Request::builder()
        .method(method)
        .uri(uri)
        .header("content-type", "application/json");
    let req = match body {
        Some(b) => req.body(Body::from(b.to_string())).unwrap(),
        None => req.body(Body::empty()).unwrap(),
    };
    send(r, req).await
}

async fn post(r: &axum::Router, uri: &str, body: Value) -> (StatusCode, Value) {
    call(r, "POST", uri, Some(body)).await
}

#[tokio::test]
async fn health_and_info() {
    let r = router();
    let (s, v) = call(&r, "GET", "/health", None).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(v, json!("ok"));
    let (s, v) = call(&r, "GET", "/v1/info", None).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(v["context_len"], 8);
    assert_eq!(v["vocab"], reference().meta.output_dim);
    assert_eq!(v["tokenizer"], "bpe");
    assert_eq!(v["model_sha256"].as_str().unwrap().len(), 64);
    // ponytail: `> 0` once Sequential::num_params lands (Phase 1A).
    assert!(v["params"].is_u64());
    assert!(v["quant"].is_string());
}

#[tokio::test]
async fn unknown_route_is_404_and_wrong_method_405_as_json() {
    let r = router();
    for (m, u, want) in [
        ("GET", "/nope", StatusCode::NOT_FOUND),
        ("POST", "/health", StatusCode::METHOD_NOT_ALLOWED),
        ("GET", "/v1/generate", StatusCode::METHOD_NOT_ALLOWED),
    ] {
        let (s, v) = call(&r, m, u, None).await;
        assert_eq!(s, want, "{m} {u}");
        assert!(v["error"].is_string(), "{m} {u}: {v}");
    }
}

#[tokio::test]
async fn invalid_sampling_params_are_400() {
    let r = router();
    for (field, val) in [
        ("temperature", json!(-1.0)),
        ("top_p", json!(0.0)),
        ("top_p", json!(1.5)),
    ] {
        let mut req = json!({"prompt": "the"});
        req[field] = val;
        let (s, v) = post(&r, "/v1/generate", json!({"requests": [req]})).await;
        assert_eq!(s, StatusCode::BAD_REQUEST, "{field}");
        assert!(v["error"].as_str().unwrap().contains(field), "{v}");
    }
}

#[test]
fn app_rejects_unloadable_model_bytes() {
    let e = app(b"garbage".to_vec(), Limits::default()).err().unwrap();
    assert!(!e.is_empty());
}

#[tokio::test]
async fn loglikelihood_matches_library() {
    let r = router();
    let (s, v) = post(
        &r,
        "/v1/loglikelihood",
        json!({"requests": [
            {"context": "the cat sat on", "continuation": " the mat."},
            {"context": "the cat sat on", "continuation": " zebra"}
        ]}),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    let res = v["results"].as_array().unwrap();
    assert_eq!(res.len(), 2);
    let want = reference().score("the cat sat on", " the mat.").unwrap();
    assert!((res[0]["logprob"].as_f64().unwrap() - want.logprob).abs() < 1e-6);
    assert_eq!(res[0]["n_tokens"], want.n_tokens);
    assert_eq!(res[0]["greedy"], want.greedy);
    assert!(res[0]["logprob"].as_f64().unwrap() > res[1]["logprob"].as_f64().unwrap());
}

#[tokio::test]
async fn long_rolling_text_is_scored_whole() {
    let r = router();
    let text = "the dog sat on the log. ".repeat(40);
    let (s, v) = post(
        &r,
        "/v1/loglikelihood_rolling",
        json!({"texts": [text.clone()]}),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    let want = reference().score_rolling(&text).unwrap();
    assert_eq!(v["results"][0]["n_tokens"], want.n_tokens);
    assert!((v["results"][0]["logprob"].as_f64().unwrap() - want.logprob).abs() < 1e-6);
    assert!(want.n_tokens > 8 * 5);
}

#[tokio::test]
async fn generate_is_deterministic_and_honours_stop() {
    let r = router();
    let body = json!({"requests": [{"prompt": "the cat", "max_tokens": 20}]});
    let (_, a) = post(&r, "/v1/generate", body.clone()).await;
    let (_, b) = post(&r, "/v1/generate", body).await;
    assert_eq!(a, b);
    let text = a["results"][0]["text"].as_str().unwrap().to_string();
    assert_eq!(a["results"][0]["n_tokens"], 20);
    let stop: String = text.chars().skip(2).take(2).collect();
    let (_, c) = post(
        &r,
        "/v1/generate",
        json!({"requests": [{"prompt": "the cat", "max_tokens": 20, "stop": [stop.clone()]}]}),
    )
    .await;
    let cut = c["results"][0]["text"].as_str().unwrap();
    assert!(!cut.contains(&stop) && text.starts_with(cut));
}

#[tokio::test]
async fn generate_defaults_and_sampling_fields() {
    let r = router();
    // Only `prompt`: max_tokens defaults to 64, greedy.
    let (s, v) = post(
        &r,
        "/v1/generate",
        json!({"requests": [{"prompt": "the cat"}]}),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(v["results"][0]["n_tokens"], 64);
    // max_tokens 0 generates nothing.
    let (_, v) = post(
        &r,
        "/v1/generate",
        json!({"requests": [{"prompt": "the", "max_tokens": 0}]}),
    )
    .await;
    assert_eq!(v["results"][0]["n_tokens"], 0);
    assert_eq!(v["results"][0]["text"], "");
    // Sampling: same seed reproduces, all fields accepted.
    let req = json!({"requests": [{"prompt": "the", "max_tokens": 12, "temperature": 0.8,
        "top_k": 5, "top_p": 0.9, "seed": 7}]});
    let (s, a) = post(&r, "/v1/generate", req.clone()).await;
    assert_eq!(s, StatusCode::OK, "{a}");
    let (_, b) = post(&r, "/v1/generate", req).await;
    assert_eq!(a, b);
    // Several requests in one batch come back in order.
    let (_, v) = post(
        &r,
        "/v1/generate",
        json!({"requests": [{"prompt": "the cat", "max_tokens": 3}, {"prompt": "the dog", "max_tokens": 5}]}),
    )
    .await;
    assert_eq!(v["results"][0]["n_tokens"], 3);
    assert_eq!(v["results"][1]["n_tokens"], 5);
}

#[tokio::test]
async fn empty_batch_is_ok() {
    let r = router();
    for (uri, body) in [
        ("/v1/loglikelihood", json!({"requests": []})),
        ("/v1/loglikelihood_rolling", json!({"texts": []})),
        ("/v1/generate", json!({"requests": []})),
    ] {
        let (s, v) = post(&r, uri, body).await;
        assert_eq!(s, StatusCode::OK, "{uri}");
        assert_eq!(v, json!({"results": []}), "{uri}");
    }
}

#[tokio::test]
async fn bad_json_is_400() {
    let r = router();
    // missing field
    let (s, v) = post(
        &r,
        "/v1/loglikelihood",
        json!({"requests": [{"context": "x"}]}),
    )
    .await;
    assert_eq!(s, StatusCode::BAD_REQUEST, "{v}");
    assert!(v["error"].is_string());
    for (uri, body) in [
        ("/v1/loglikelihood_rolling", json!({"nope": []})),
        ("/v1/generate", json!({"requests": [{"max_tokens": 3}]})),
        (
            "/v1/generate",
            json!({"requests": [{"prompt": "x", "max_tokens": -1}]}),
        ),
    ] {
        assert_eq!(
            post(&r, uri, body).await.0,
            StatusCode::BAD_REQUEST,
            "{uri}"
        );
    }
    // syntactically invalid JSON
    let req = Request::builder()
        .method("POST")
        .uri("/v1/generate")
        .header("content-type", "application/json")
        .body(Body::from("{not json"))
        .unwrap();
    assert_eq!(send(&r, req).await.0, StatusCode::BAD_REQUEST);
    // missing content-type (axum would say 415)
    let req = Request::builder()
        .method("POST")
        .uri("/v1/generate")
        .body(Body::from(r#"{"requests":[]}"#))
        .unwrap();
    assert_eq!(send(&r, req).await.0, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn limits_are_enforced() {
    let r = router_with(Limits {
        max_batch: 2,
        max_chars: 50,
        max_body: 1024,
        max_tokens: 40,
    });
    let rolling = "/v1/loglikelihood_rolling";
    let (s, v) = post(&r, rolling, json!({"texts": ["a", "b", "c"]})).await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert!(v["error"].as_str().unwrap().contains("max_batch"));
    // exactly at the limits is fine
    let (s, _) = post(&r, rolling, json!({"texts": ["a", "x".repeat(50)]})).await;
    assert_eq!(s, StatusCode::OK);
    let (s, v) = post(&r, rolling, json!({"texts": ["x".repeat(51)]})).await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert!(v["error"].as_str().unwrap().contains("max_chars"));
    let (s, v) = post(&r, rolling, json!({"texts": ["y".repeat(2000)]})).await;
    assert_eq!(s, StatusCode::PAYLOAD_TOO_LARGE);
    assert!(v["error"].is_string());
    // loglikelihood: batch and context+continuation length
    let ll = |n: usize| json!({"requests": (0..n).map(|_| json!({"context": "a", "continuation": "b"})).collect::<Vec<_>>()});
    assert_eq!(
        post(&r, "/v1/loglikelihood", ll(3)).await.0,
        StatusCode::BAD_REQUEST
    );
    let split = json!({"requests": [{"context": "c".repeat(30), "continuation": "d".repeat(30)}]});
    assert_eq!(
        post(&r, "/v1/loglikelihood", split).await.0,
        StatusCode::BAD_REQUEST
    );
    // generate: batch, prompt length, max_tokens
    let gen = |n: usize| json!({"requests": (0..n).map(|_| json!({"prompt": "a", "max_tokens": 1})).collect::<Vec<_>>()});
    assert_eq!(
        post(&r, "/v1/generate", gen(3)).await.0,
        StatusCode::BAD_REQUEST
    );
    let long = json!({"requests": [{"prompt": "p".repeat(51)}]});
    assert_eq!(
        post(&r, "/v1/generate", long).await.0,
        StatusCode::BAD_REQUEST
    );
    let (s, v) = post(
        &r,
        "/v1/generate",
        json!({"requests": [{"prompt": "a", "max_tokens": 41}]}),
    )
    .await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert!(v["error"].as_str().unwrap().contains("max_tokens"));
    let (s, _) = post(
        &r,
        "/v1/generate",
        json!({"requests": [{"prompt": "a", "max_tokens": 40}]}),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    // ... and the server still answers afterwards
    assert_eq!(call(&r, "GET", "/health", None).await.0, StatusCode::OK);
}

#[tokio::test]
async fn model_error_is_422_and_server_survives() {
    let r = char_router();
    let (s, v) = post(&r, "/v1/loglikelihood_rolling", json!({"texts": ["xyz"]})).await;
    assert_eq!(s, StatusCode::UNPROCESSABLE_ENTITY, "{v}");
    assert!(v["error"].as_str().unwrap().contains("vocabulary"));
    // the same 422 on the other two routes
    let (s, _) = post(
        &r,
        "/v1/loglikelihood",
        json!({"requests": [{"context": "abc", "continuation": "xyz"}]}),
    )
    .await;
    assert_eq!(s, StatusCode::UNPROCESSABLE_ENTITY);
    let (s, _) = post(&r, "/v1/generate", json!({"requests": [{"prompt": "xyz"}]})).await;
    assert_eq!(s, StatusCode::UNPROCESSABLE_ENTITY);
    // server survives
    let (s, _) = post(&r, "/v1/loglikelihood_rolling", json!({"texts": ["abc"]})).await;
    assert_eq!(s, StatusCode::OK);
    let (s, v) = post(
        &r,
        "/v1/generate",
        json!({"requests": [{"prompt": "abc", "max_tokens": 4}]}),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");
}
