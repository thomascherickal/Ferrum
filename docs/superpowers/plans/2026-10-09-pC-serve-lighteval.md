# Phase 1C — `slm_server` and LightEval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve a Ferrum SLM over HTTP (loglikelihood, rolling loglikelihood, generation) and evaluate it with Hugging Face LightEval 0.13.0 through a custom model class.

**Architecture:**
- New workspace crate `slm_server` (axum 0.8 + tokio).
  - `GenerativeSLM` is not `Send`/`Sync`: `TransformerBlock` holds a `RefCell`. So one dedicated **model thread** owns the model and serves jobs from an `std::sync::mpsc` channel. Each job carries a `tokio::sync::oneshot` reply sender.
  - Requests are processed one at a time. Each matmul still uses every core.
- Python package `eval/lighteval/` with:
  - `FerrumModel(LightevalModel)` that calls the server with stdlib `urllib`
  - custom tasks built from the DVC test split
  - `run.sh`, which starts the server, runs `lighteval custom`, and writes `metrics/lighteval.json`

**Tech Stack:**
- Rust: `axum 0.8`, `tokio 1` (rt-multi-thread, macros, net, signal, sync), `serde 1`, `serde_json 1`, `sha2 0.11`; dev: `tower 0.5` (util), `http-body-util 0.1`.
- Python 3.12: `lighteval==0.13.0`, `xxhash<4`, CPU torch.

**Spec:** `docs/superpowers/specs/2026-10-09-dataops-serve-eval-design.md` (§6). Phase 0 plan: `docs/superpowers/plans/2026-10-09-p0-shared-surfaces.md`.

## Global Constraints

- Start from the Phase 0 tip in a worktree: `git worktree add ../Ferrum-C -b feat/dos-C feat/dataops-serve-eval`.
- `#![forbid(unsafe_code)]` at the top of `slm_server/src/lib.rs` and `main.rs`.
- Bind `127.0.0.1` by default. Enforce limits **before** any model compute: body size (default 8 MiB), batch length (default 64), per-text characters (default 65,536). Return `413` for an oversized body and `400` for other limit or validation failures, both with `{"error": "..."}`. Model errors return `422`.
- LightEval pinned to **0.13.0**, plus `xxhash<4` (xxhash 4.x crashes the details logger). In `FerrumModel`, do not call `super().__init__()` (raises `TypeError` in 0.13.0). Set `self._cache = SampleCache(config)` directly. Leave `loglikelihood_rolling` **un-cached** (0.13.0's PERPLEXITY cache drops results).
- Python is not in CI (torch is heavy). The stdlib `unittest` must run without LightEval installed by skipping cleanly. The real smoke run is part of this phase's gate.
- Do not edit `ReadMe.md`, `Manual.md`, `Status.md` (Phase 2).
- Models are GGUF (`.gguf`). Every feature lands with its unit tests in the same commit.
- Each Rust commit passes `cargo test --workspace --release`, `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`.

## Contract with Phase 1B

Write `eval/lighteval/run.sh --model <slm.gguf> --test <test.txt> --out <metrics.json> [--max-samples N] [--wikitext] [--port P]`. Phase 1B's frozen DVC `lighteval` stage calls it with the first three flags.

## Review Focus

1. A request whose `requests` array is empty — `200` with `{"results": []}`, not an error. → Task 2 test `empty_batch_is_ok`.
2. Malformed JSON or a missing field — `400` (axum's default is `422`; remap so model errors stay distinguishable). → Task 2 test `bad_json_is_400`.
3. Characters outside a char-level model's vocabulary — `422` with the model's message, and the server keeps serving. → Task 2 test `model_error_is_422_and_server_survives`.
4. The server is unreachable or returns an error during a LightEval run — `FerrumModel` raises a clear `RuntimeError` naming the URL and status instead of returning bogus scores. → Task 4 test `test_http_error_raises`.
5. Wikitext documents far longer than `context_len` — scored through the sliding windows of `score_rolling`, no truncation. → Task 2 test `long_rolling_text_is_scored_whole`.

---

### Task 1: Crate skeleton, model thread, `/health`, `/v1/info`

**Files:**
- Modify: `Cargo.toml` (workspace `members` += `"slm_server"`)
- Create: `slm_server/Cargo.toml`, `slm_server/src/lib.rs`, `slm_server/src/main.rs`, `slm_server/tests/api.rs`

**Interfaces:**
- Consumes: `ferrum_core::{GenerativeSLM, Score, SamplingParams, Generation, Rng}` (Phase 0).
- Produces:
  ```rust
  pub struct Limits { pub max_batch: usize, pub max_chars: usize, pub max_body: usize }
  impl Default for Limits { /* 64, 65_536, 8 << 20 */ }
  pub struct Info { pub params: usize, pub context_len: usize, pub vocab: usize,
                    pub quant: String, pub tokenizer: String, pub model_sha256: String }
  /// Spawn the model thread and build the router.
  pub fn app(model: GenerativeSLM, info: Info, limits: Limits) -> axum::Router;
  /// Info for a model file (sha256 of `bytes`).
  pub fn info_for(model: &GenerativeSLM, bytes: &[u8]) -> Info;
  ```

- [ ] **Step 1: Cargo setup**

`slm_server/Cargo.toml`:

```toml
[package]
name = "slm_server"
description = "HTTP inference server for Ferrum SLMs (loglikelihood + generation) for eval harnesses."
version.workspace = true
edition.workspace = true
license.workspace = true

[[bin]]
name = "ferrum-serve"
path = "src/main.rs"

[dependencies]
ferrum_core = { path = "../ferrum_core" }
axum = "0.8"
tokio = { version = "1", features = ["rt-multi-thread", "macros", "net", "signal", "sync"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
sha2 = "0.11"

[dev-dependencies]
tower = { version = "0.5", features = ["util"] }
http-body-util = "0.1"
```

- [ ] **Step 2: Failing test** — `slm_server/tests/api.rs`:

```rust
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
        context_len: 8, embed_dim: 16, num_heads: 2, num_blocks: 1, hidden_dim: 32,
        epochs: 2, vocab_size: 270, ..TransformerConfig::default()
    };
    GenerativeSLM::train_transformer_config(&corpus, &cfg, &mut Rng::new(3), |_, _| {}).unwrap()
}

pub fn router() -> axum::Router {
    router_with(Limits::default())
}

pub fn router_with(limits: Limits) -> axum::Router {
    let m = tiny_model();
    let bytes = m.to_bytes_quantized().unwrap();
    let info = info_for(&m, &bytes);
    app(m, info, limits)
}

pub async fn call(r: &axum::Router, method: &str, uri: &str, body: Option<Value>) -> (StatusCode, Value) {
    let req = Request::builder().method(method).uri(uri).header("content-type", "application/json");
    let req = match body {
        Some(b) => req.body(Body::from(b.to_string())).unwrap(),
        None => req.body(Body::empty()).unwrap(),
    };
    let resp = r.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let v = serde_json::from_slice(&bytes).unwrap_or(Value::String(String::from_utf8_lossy(&bytes).into()));
    (status, v)
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
    assert_eq!(v["vocab"], 270);
    assert_eq!(v["tokenizer"], "bpe");
    assert_eq!(v["model_sha256"].as_str().unwrap().len(), 64);
    assert!(v["params"].as_u64().unwrap() > 0);
}
```

- [ ] **Step 3: Verify failure** — `cargo test -p slm_server --release 2>&1 | tail -5`. Expected: compile error (`app` not found).

- [ ] **Step 4: Implement `slm_server/src/lib.rs`**

```rust
//! HTTP inference server for Ferrum SLMs: loglikelihood, rolling
//! loglikelihood, and generation — the three primitives eval harnesses such as
//! LightEval need. One thread owns the model (it is not `Sync`); requests are
//! queued to it over a channel and answered through oneshot replies.
#![forbid(unsafe_code)]

use axum::extract::{rejection::JsonRejection, DefaultBodyLimit, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use ferrum_core::{GenerativeSLM, Rng, SamplingParams};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::sync::{mpsc, Arc};
use tokio::sync::oneshot;

pub struct Limits {
    pub max_batch: usize,
    pub max_chars: usize,
    pub max_body: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self { max_batch: 64, max_chars: 65_536, max_body: 8 << 20 }
    }
}

#[derive(Clone, Serialize)]
pub struct Info {
    pub params: usize,
    pub context_len: usize,
    pub vocab: usize,
    pub quant: String,
    pub tokenizer: String,
    pub model_sha256: String,
}

/// Describe `model`, loaded from `bytes` (hashed for `model_sha256`).
pub fn info_for(model: &GenerativeSLM, bytes: &[u8]) -> Info {
    Info {
        params: model.model.num_params(),
        context_len: model.meta.input_dim,
        vocab: model.meta.output_dim,
        quant: ferrum_core::loader::gguf_quant(bytes).unwrap_or_else(|_| "unknown".into()),
        tokenizer: if model.meta.tokenizer_state.is_empty() { "char".into() } else { "bpe".into() },
        model_sha256: format!("{:x}", Sha256::digest(bytes)),
    }
}

enum Job {
    Score(Vec<(String, String)>, oneshot::Sender<Result<Vec<Value>, String>>),
    Rolling(Vec<String>, oneshot::Sender<Result<Vec<Value>, String>>),
    Generate(Vec<GenRequest>, oneshot::Sender<Result<Vec<Value>, String>>),
}

#[derive(Clone)]
struct AppState {
    jobs: mpsc::Sender<Job>,
    info: Arc<Info>,
    max_batch: usize,
    max_chars: usize,
}

/// Run `model` on its own thread, answering jobs until every sender is dropped.
fn spawn_model_thread(model: GenerativeSLM) -> mpsc::Sender<Job> {
    let (tx, rx) = mpsc::channel::<Job>();
    std::thread::Builder::new()
        .name("ferrum-model".into())
        .spawn(move || {
            for job in rx {
                match job {
                    Job::Score(reqs, reply) => {
                        let r = reqs
                            .iter()
                            .map(|(c, k)| model.score(c, k).map(|s| json!({"logprob": s.logprob, "n_tokens": s.n_tokens, "greedy": s.greedy})))
                            .collect::<Result<Vec<_>, _>>()
                            .map_err(|e| e.to_string());
                        let _ = reply.send(r);
                    }
                    Job::Rolling(texts, reply) => {
                        let r = texts
                            .iter()
                            .map(|t| model.score_rolling(t).map(|s| json!({"logprob": s.logprob, "n_tokens": s.n_tokens})))
                            .collect::<Result<Vec<_>, _>>()
                            .map_err(|e| e.to_string());
                        let _ = reply.send(r);
                    }
                    Job::Generate(reqs, reply) => {
                        let r = reqs
                            .iter()
                            .map(|g| {
                                let params = SamplingParams {
                                    temperature: g.temperature,
                                    top_k: g.top_k,
                                    top_p: g.top_p,
                                    repetition_penalty: 1.0,
                                };
                                model
                                    .generate_tokens(&g.prompt, g.max_tokens, &g.stop, &params, &mut Rng::new(g.seed))
                                    .map(|o| json!({"text": o.text, "n_tokens": o.n_tokens}))
                            })
                            .collect::<Result<Vec<_>, _>>()
                            .map_err(|e| e.to_string());
                        let _ = reply.send(r);
                    }
                }
            }
        })
        .expect("spawn model thread");
    tx
}

/// Build the router; spawns the model thread.
pub fn app(model: GenerativeSLM, info: Info, limits: Limits) -> Router {
    let state = AppState {
        jobs: spawn_model_thread(model),
        info: Arc::new(info),
        max_batch: limits.max_batch,
        max_chars: limits.max_chars,
    };
    Router::new()
        .route("/health", get(|| async { Json("ok") }))
        .route("/v1/info", get(|State(s): State<AppState>| async move { Json((*s.info).clone()) }))
        .route("/v1/loglikelihood", post(loglikelihood))
        .route("/v1/loglikelihood_rolling", post(loglikelihood_rolling))
        .route("/v1/generate", post(generate))
        .layer(DefaultBodyLimit::max(limits.max_body))
        .with_state(state)
}

/// JSON error response.
struct ApiError(StatusCode, String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(json!({"error": self.1}))).into_response()
    }
}

/// Map axum's JSON rejection: oversized body → 413, anything else → 400.
fn reject(e: JsonRejection) -> ApiError {
    let status = if e.status() == StatusCode::PAYLOAD_TOO_LARGE { StatusCode::PAYLOAD_TOO_LARGE } else { StatusCode::BAD_REQUEST };
    ApiError(status, e.body_text())
}

fn check(s: &AppState, n: usize, texts: impl Iterator<Item = usize>) -> Result<(), ApiError> {
    if n > s.max_batch {
        return Err(ApiError(StatusCode::BAD_REQUEST, format!("batch of {n} exceeds max_batch {}", s.max_batch)));
    }
    if let Some(len) = texts.into_iter().find(|&l| l > s.max_chars) {
        return Err(ApiError(StatusCode::BAD_REQUEST, format!("text of {len} chars exceeds max_chars {}", s.max_chars)));
    }
    Ok(())
}

async fn submit(s: &AppState, make: impl FnOnce(oneshot::Sender<Result<Vec<Value>, String>>) -> Job) -> Result<Json<Value>, ApiError> {
    let (tx, rx) = oneshot::channel();
    s.jobs
        .send(make(tx))
        .map_err(|_| ApiError(StatusCode::SERVICE_UNAVAILABLE, "model thread stopped".into()))?;
    match rx.await {
        Ok(Ok(results)) => Ok(Json(json!({"results": results}))),
        Ok(Err(e)) => Err(ApiError(StatusCode::UNPROCESSABLE_ENTITY, e)),
        Err(_) => Err(ApiError(StatusCode::SERVICE_UNAVAILABLE, "model thread stopped".into())),
    }
}

#[derive(Deserialize)]
struct LlRequest { context: String, continuation: String }
#[derive(Deserialize)]
struct LlBody { requests: Vec<LlRequest> }

async fn loglikelihood(State(s): State<AppState>, body: Result<Json<LlBody>, JsonRejection>) -> Result<Json<Value>, ApiError> {
    let Json(b) = body.map_err(reject)?;
    check(&s, b.requests.len(), b.requests.iter().map(|r| r.context.chars().count() + r.continuation.chars().count()))?;
    let reqs = b.requests.into_iter().map(|r| (r.context, r.continuation)).collect();
    submit(&s, |tx| Job::Score(reqs, tx)).await
}

#[derive(Deserialize)]
struct RollingBody { texts: Vec<String> }

async fn loglikelihood_rolling(State(s): State<AppState>, body: Result<Json<RollingBody>, JsonRejection>) -> Result<Json<Value>, ApiError> {
    let Json(b) = body.map_err(reject)?;
    check(&s, b.texts.len(), b.texts.iter().map(|t| t.chars().count()))?;
    submit(&s, |tx| Job::Rolling(b.texts, tx)).await
}

fn default_max_tokens() -> usize { 64 }
fn default_top_p() -> f32 { 1.0 }

#[derive(Deserialize)]
struct GenRequest {
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default)]
    stop: Vec<String>,
    #[serde(default)]
    temperature: f32,
    #[serde(default)]
    top_k: usize,
    #[serde(default = "default_top_p")]
    top_p: f32,
    #[serde(default)]
    seed: u64,
}
#[derive(Deserialize)]
struct GenBody { requests: Vec<GenRequest> }

async fn generate(State(s): State<AppState>, body: Result<Json<GenBody>, JsonRejection>) -> Result<Json<Value>, ApiError> {
    let Json(b) = body.map_err(reject)?;
    check(&s, b.requests.len(), b.requests.iter().map(|r| r.prompt.chars().count()))?;
    if let Some(r) = b.requests.iter().find(|r| r.max_tokens > s.max_chars) {
        return Err(ApiError(StatusCode::BAD_REQUEST, format!("max_tokens {} exceeds max_chars {}", r.max_tokens, s.max_chars)));
    }
    submit(&s, |tx| Job::Generate(b.requests, tx)).await
}
```

`ferrum_core` exposes `Sequential::num_params()` only after Phase 1A Task 3. Until the merge, replace `model.model.num_params()` with a small local helper that counts nothing (`0`) and tag it `// ponytail: real count after Phase 1A merge (Sequential::num_params)`. Loosen the test to `v["params"].is_u64()`. After the merge, Phase 2 restores the real call and the `> 0` assertion.

`slm_server/src/main.rs`:

```rust
//! `ferrum-serve --model <path> [--host 127.0.0.1] [--port 8080]
//!   [--max-batch 64] [--max-chars 65536] [--max-body 8388608]`
#![forbid(unsafe_code)]

use slm_server::{app, info_for, Limits};

fn flag<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> Result<T, String> {
    match args.iter().position(|a| a == name) {
        None => Ok(default),
        Some(i) => {
            let v = args.get(i + 1).ok_or(format!("{name} needs a value"))?;
            v.parse().map_err(|_| format!("{name}: cannot parse '{v}'"))
        }
    }
}

#[tokio::main]
async fn main() {
    if let Err(e) = run().await {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    const KNOWN: &[&str] = &["--model", "--host", "--port", "--max-batch", "--max-chars", "--max-body"];
    for (i, a) in args.iter().enumerate() {
        if a.starts_with("--") && !KNOWN.contains(&a.as_str()) {
            return Err(format!("unknown flag {a} (valid: {})", KNOWN.join(" ")).into());
        }
        if i % 2 == 0 && !a.starts_with("--") {
            return Err(format!("unexpected argument '{a}'").into());
        }
    }
    let model_path: String = flag(&args, "--model", String::new())?;
    if model_path.is_empty() {
        return Err("usage: ferrum-serve --model <model.gguf> [--host H] [--port P]".into());
    }
    let host: String = flag(&args, "--host", "127.0.0.1".to_string())?;
    let port: u16 = flag(&args, "--port", 8080)?;
    let d = Limits::default();
    let limits = Limits {
        max_batch: flag(&args, "--max-batch", d.max_batch)?,
        max_chars: flag(&args, "--max-chars", d.max_chars)?,
        max_body: flag(&args, "--max-body", d.max_body)?,
    };
    let bytes = std::fs::read(&model_path).map_err(|e| format!("cannot read {model_path}: {e}"))?;
    let model = ferrum_core::GenerativeSLM::from_bytes(&bytes)?;
    let info = info_for(&model, &bytes);
    eprintln!(
        "ferrum-serve: {model_path} ({} params, context {}, vocab {}) on http://{host}:{port}",
        info.params, info.context_len, info.vocab
    );
    let listener = tokio::net::TcpListener::bind((host.as_str(), port)).await?;
    axum::serve(listener, app(model, info, limits))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}
```

Add `"slm_server"` to the workspace `members`.

- [ ] **Step 5: Verify and commit**

```bash
cargo test -p slm_server --release 2>&1 | grep -E "^test result|FAILED"
cargo fmt --all && cargo clippy --workspace --all-targets -- -D warnings
git add Cargo.toml Cargo.lock slm_server
git commit -m "feat(server): slm_server crate — model thread, /health, /v1/info

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Loglikelihood, rolling, and generate routes with limits

**Files:**
- Modify: `slm_server/tests/api.rs` (route tests; the handlers were written in Task 1)
- Modify: `slm_server/src/lib.rs` only if a test exposes a bug

**Interfaces:**
- Consumes: the Task 1 router.
- Produces: the HTTP contract consumed by Task 3:
  - `POST /v1/loglikelihood {"requests":[{"context","continuation"}]}` → `{"results":[{"logprob","n_tokens","greedy"}]}`
  - `POST /v1/loglikelihood_rolling {"texts":[…]}` → `{"results":[{"logprob","n_tokens"}]}`
  - `POST /v1/generate {"requests":[{"prompt","max_tokens"?,"stop"?,"temperature"?,"top_k"?,"top_p"?,"seed"?}]}` → `{"results":[{"text","n_tokens"}]}`

- [ ] **Step 1: Write the tests** (append to `api.rs`)

```rust
#[tokio::test]
async fn loglikelihood_matches_library() {
    let r = router();
    let (s, v) = call(&r, "POST", "/v1/loglikelihood", Some(json!({"requests": [
        {"context": "the cat sat on", "continuation": " the mat."},
        {"context": "the cat sat on", "continuation": " zebra"}
    ]}))).await;
    assert_eq!(s, StatusCode::OK, "{v}");
    let res = v["results"].as_array().unwrap();
    assert_eq!(res.len(), 2);
    let want = tiny_model().score("the cat sat on", " the mat.").unwrap();
    assert!((res[0]["logprob"].as_f64().unwrap() - want.logprob).abs() < 1e-6);
    assert_eq!(res[0]["n_tokens"], want.n_tokens);
    assert!(res[0]["logprob"].as_f64().unwrap() > res[1]["logprob"].as_f64().unwrap());
}

#[tokio::test]
async fn long_rolling_text_is_scored_whole() {
    let r = router();
    let text = "the dog sat on the log. ".repeat(40);
    let (s, v) = call(&r, "POST", "/v1/loglikelihood_rolling", Some(json!({"texts": [text.clone()]}))).await;
    assert_eq!(s, StatusCode::OK, "{v}");
    let want = tiny_model().score_rolling(&text).unwrap();
    assert_eq!(v["results"][0]["n_tokens"], want.n_tokens);
    assert!(want.n_tokens > 8 * 5);
}

#[tokio::test]
async fn generate_is_deterministic_and_honours_stop() {
    let r = router();
    let body = json!({"requests": [{"prompt": "the cat", "max_tokens": 20}]});
    let (_, a) = call(&r, "POST", "/v1/generate", Some(body.clone())).await;
    let (_, b) = call(&r, "POST", "/v1/generate", Some(body)).await;
    assert_eq!(a, b);
    let text = a["results"][0]["text"].as_str().unwrap().to_string();
    assert_eq!(a["results"][0]["n_tokens"], 20);
    let stop: String = text.chars().skip(2).take(2).collect();
    let (_, c) = call(&r, "POST", "/v1/generate", Some(json!({"requests": [{"prompt": "the cat", "max_tokens": 20, "stop": [stop.clone()]}]}))).await;
    let cut = c["results"][0]["text"].as_str().unwrap();
    assert!(!cut.contains(&stop) && text.starts_with(cut));
}

#[tokio::test]
async fn empty_batch_is_ok() {
    let r = router();
    let (s, v) = call(&r, "POST", "/v1/loglikelihood", Some(json!({"requests": []}))).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(v, json!({"results": []}));
}

#[tokio::test]
async fn bad_json_is_400() {
    let r = router();
    let (s, v) = call(&r, "POST", "/v1/loglikelihood", Some(json!({"requests": [{"context": "x"}]}))).await;
    assert_eq!(s, StatusCode::BAD_REQUEST, "{v}");
    assert!(v["error"].is_string());
}

#[tokio::test]
async fn limits_are_enforced() {
    let r = router_with(Limits { max_batch: 2, max_chars: 50, max_body: 1024 });
    let three = json!({"texts": ["a", "b", "c"]});
    assert_eq!(call(&r, "POST", "/v1/loglikelihood_rolling", Some(three)).await.0, StatusCode::BAD_REQUEST);
    let long = json!({"texts": ["x".repeat(60)]});
    assert_eq!(call(&r, "POST", "/v1/loglikelihood_rolling", Some(long)).await.0, StatusCode::BAD_REQUEST);
    let huge = json!({"texts": ["y".repeat(2000)]});
    assert_eq!(call(&r, "POST", "/v1/loglikelihood_rolling", Some(huge)).await.0, StatusCode::PAYLOAD_TOO_LARGE);
}

#[tokio::test]
async fn model_error_is_422_and_server_survives() {
    let corpus = "abcabc abc\n".repeat(30);
    let cfg = TransformerConfig {
        context_len: 8, embed_dim: 8, num_heads: 2, num_blocks: 1, hidden_dim: 16,
        epochs: 1, vocab_size: 0, ..TransformerConfig::default()
    };
    let m = GenerativeSLM::train_transformer_config(&corpus, &cfg, &mut Rng::new(1), |_, _| {}).unwrap();
    let bytes = m.to_bytes_quantized().unwrap();
    let info = info_for(&m, &bytes);
    let r = app(m, info, Limits::default());
    let (s, v) = call(&r, "POST", "/v1/loglikelihood_rolling", Some(json!({"texts": ["xyz"]}))).await;
    assert_eq!(s, StatusCode::UNPROCESSABLE_ENTITY, "{v}");
    assert!(v["error"].as_str().unwrap().contains("vocabulary"));
    let (s, _) = call(&r, "POST", "/v1/loglikelihood_rolling", Some(json!({"texts": ["abc"]}))).await;
    assert_eq!(s, StatusCode::OK);
}
```

- [ ] **Step 2: Run** — `cargo test -p slm_server --release 2>&1 | grep -E "test |panicked"`.
Expected: all pass. If one fails, fix `lib.rs`, not the test. Likely spots:
  - `reject` mapping: axum 0.8 `JsonRejection::status()`
  - `DefaultBodyLimit` surfaces as `JsonRejection::BytesRejection` with status 413

- [ ] **Step 3: Commit**

```bash
cargo fmt --all && cargo clippy --workspace --all-targets -- -D warnings
git add slm_server
git commit -m "test(server): route, limit, and error-mapping coverage

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `FerrumModel` for LightEval 0.13.0

**Files:**
- Create: `eval/lighteval/ferrum_model.py`, `eval/lighteval/requirements.txt`, `eval/lighteval/test_ferrum_model.py`
- Modify: `.gitignore` (`eval/lighteval/.venv/`, `eval/lighteval/out/`, `__pycache__/`)

**Interfaces:**
- Consumes: the HTTP contract from Task 2.
- Produces: `class FerrumModel(LightevalModel)`, loaded by `lighteval custom ferrum eval/lighteval/ferrum_model.py …`. The server URL comes from env `FERRUM_URL` (default `http://127.0.0.1:8080`) and the batch size from `FERRUM_BATCH` (default 32).

- [ ] **Step 1: Requirements and venv**

`eval/lighteval/requirements.txt`:

```
--extra-index-url https://download.pytorch.org/whl/cpu
lighteval==0.13.0
xxhash<4
torch
```

```bash
python3 -m venv eval/lighteval/.venv
eval/lighteval/.venv/bin/pip install -q -r eval/lighteval/requirements.txt
eval/lighteval/.venv/bin/python -c "import lighteval; print(lighteval.__version__)"   # 0.13.0
```

- [ ] **Step 2: Failing test** — `eval/lighteval/test_ferrum_model.py` (stdlib `unittest` plus a stub HTTP server):

```python
"""FerrumModel against a stub server. Run:
eval/lighteval/.venv/bin/python -m unittest eval/lighteval/test_ferrum_model.py
Skips cleanly when lighteval is not installed."""
import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from lighteval.tasks.requests import Doc
    import ferrum_model
except ImportError:  # pragma: no cover
    ferrum_model = None

CALLS = []


class Stub(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send(200, {"context_len": 128, "vocab": 1024, "params": 486624})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        CALLS.append((self.path, body))
        if self.path == "/v1/loglikelihood":
            res = [{"logprob": -float(len(r["continuation"])), "n_tokens": 1, "greedy": r["continuation"] == " a"} for r in body["requests"]]
        elif self.path == "/v1/loglikelihood_rolling":
            res = [{"logprob": -2.0 * len(t), "n_tokens": len(t)} for t in body["texts"]]
        elif self.path == "/v1/generate":
            if body["requests"][0]["prompt"] == "boom":
                return self._send(422, {"error": "kaput"})
            res = [{"text": "out:" + r["prompt"], "n_tokens": 3} for r in body["requests"]]
        else:
            return self._send(404, {"error": "no"})
        self._send(200, {"results": res})


@unittest.skipIf(ferrum_model is None, "lighteval not installed")
class FerrumModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = HTTPServer(("127.0.0.1", 0), Stub)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        os.environ["FERRUM_URL"] = f"http://127.0.0.1:{cls.srv.server_port}"
        os.environ["FERRUM_BATCH"] = "2"
        from lighteval.models.custom.custom_model import CustomModelConfig
        cfg = CustomModelConfig(model_name="ferrum", model_definition_file_path=ferrum_model.__file__)
        cls.m = ferrum_model.FerrumModel(cfg)

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def setUp(self):
        CALLS.clear()

    def test_max_length_from_info(self):
        self.assertEqual(self.m.max_length, 128)

    def test_loglikelihood_one_logprob_per_choice_batched(self):
        docs = [Doc(query="q", choices=[" a", " bb", " ccc"], gold_index=0, task_name="t")]
        out = self.m.loglikelihood(docs)
        self.assertEqual(out[0].logprobs, [-2.0, -3.0, -4.0])
        self.assertEqual(out[0].argmax_logits_eq_gold, [True, False, False])
        self.assertEqual([len(b["requests"]) for _, b in CALLS], [2, 1])  # FERRUM_BATCH=2

    def test_rolling(self):
        docs = [Doc(query="hello", choices=[], gold_index=0, task_name="t")]
        out = self.m.loglikelihood_rolling(docs)
        self.assertEqual(out[0].logprobs, [-10.0])

    def test_greedy_until_passes_stops_and_budget(self):
        docs = [Doc(query="hi", choices=[], gold_index=0, task_name="t", generation_size=7, stop_sequences=["\n"])]
        out = self.m.greedy_until(docs)
        self.assertEqual(out[0].text, ["out:hi"])
        req = CALLS[0][1]["requests"][0]
        self.assertEqual((req["max_tokens"], req["stop"], req["temperature"]), (7, ["\n"], 0.0))

    def test_http_error_raises(self):
        docs = [Doc(query="boom", choices=[], gold_index=0, task_name="t")]
        with self.assertRaises(RuntimeError) as ctx:
            self.m.greedy_until(docs)
        self.assertIn("422", str(ctx.exception))
        self.assertIn("kaput", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
```

Run: `eval/lighteval/.venv/bin/python -m unittest eval/lighteval/test_ferrum_model.py`
Expected: `ModuleNotFoundError: ferrum_model`, or every test errors.

- [ ] **Step 3: Implement `eval/lighteval/ferrum_model.py`**

```python
"""LightEval 0.13.0 custom model that evaluates a Ferrum SLM served by
`ferrum-serve` (slm_server). Usage:

    FERRUM_URL=http://127.0.0.1:8080 lighteval custom ferrum \
        eval/lighteval/ferrum_model.py "ferrum_heldout|0" \
        --custom-tasks eval/lighteval/ferrum_tasks.py

LightEval 0.13.0 quirks handled here:
- LightevalModel has no __init__: calling super().__init__() raises TypeError.
- The pipeline calls model._cache._init_registry(...) unconditionally, so
  self._cache must be a SampleCache.
- @cached(SamplingMethod.PERPLEXITY) drops every result, so
  loglikelihood_rolling is deliberately not cached.
"""
import json
import os
import urllib.error
import urllib.request

from lighteval.models.abstract_model import LightevalModel
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.utils.cache_management import SampleCache, cached

URL = os.environ.get("FERRUM_URL", "http://127.0.0.1:8080").rstrip("/")


def _batches(items, n):
    for i in range(0, len(items), n):
        yield items[i : i + n]


class FerrumModel(LightevalModel):
    def __init__(self, config):
        self.config = config
        self._cache = SampleCache(config)
        self.url = os.environ.get("FERRUM_URL", URL).rstrip("/")
        self.batch = int(os.environ.get("FERRUM_BATCH", "32"))
        self._info = None

    # ── HTTP ────────────────────────────────────────────────────────────
    def _call(self, path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(
            self.url + path, data=data, headers={"content-type": "application/json"}, method="GET" if body is None else "POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            raise RuntimeError(f"ferrum-serve {self.url}{path} returned {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"cannot reach ferrum-serve at {self.url}{path}: {e.reason}") from e

    def _results(self, path, key, items):
        out = []
        for b in _batches(items, self.batch):
            out.extend(self._call(path, {key: b})["results"])
        return out

    # ── LightevalModel interface ────────────────────────────────────────
    @property
    def tokenizer(self):
        return None

    @property
    def add_special_tokens(self):
        return False

    @property
    def max_length(self):
        if self._info is None:
            self._info = self._call("/v1/info")
        return int(self._info["context_len"])

    @cached(SamplingMethod.GENERATIVE)
    def greedy_until(self, docs: list[Doc]) -> list[ModelResponse]:
        reqs = [
            {
                "prompt": d.query,
                "max_tokens": int(d.generation_size or 64),
                "stop": list(d.stop_sequences or []),
                "temperature": 0.0,
            }
            for d in docs
        ]
        res = self._results("/v1/generate", "requests", reqs)
        return [ModelResponse(text=[r["text"]], output_tokens=[[0] * r["n_tokens"]]) for r in res]

    @cached(SamplingMethod.LOGPROBS)
    def loglikelihood(self, docs: list[Doc]) -> list[ModelResponse]:
        flat = [{"context": d.query, "continuation": c} for d in docs for c in d.choices]
        res = iter(self._results("/v1/loglikelihood", "requests", flat))
        out = []
        for d in docs:
            rs = [next(res) for _ in d.choices]
            out.append(ModelResponse(logprobs=[r["logprob"] for r in rs], argmax_logits_eq_gold=[r["greedy"] for r in rs]))
        return out

    # Not cached: lighteval 0.13.0's PERPLEXITY cache drops all results.
    def loglikelihood_rolling(self, docs: list[Doc]) -> list[ModelResponse]:
        res = self._results("/v1/loglikelihood_rolling", "texts", [d.query for d in docs])
        return [ModelResponse(logprobs=[r["logprob"]]) for r in res]
```

LightEval picks the first `LightevalModel` subclass in the file in alphabetical `dir()` order. Keep `FerrumModel` the only subclass in this file.

- [ ] **Step 4: Verify**

Run: `eval/lighteval/.venv/bin/python -m unittest eval/lighteval/test_ferrum_model.py -v`
Expected: 5 tests OK. Also run with the system python (`python3 -m unittest eval/lighteval/test_ferrum_model.py`) → `skipped 'lighteval not installed'`, exit 0.

- [ ] **Step 5: Commit**

```bash
git add eval/lighteval/ferrum_model.py eval/lighteval/requirements.txt eval/lighteval/test_ferrum_model.py .gitignore
git commit -m "feat(eval): LightEval 0.13.0 FerrumModel over the ferrum-serve HTTP API

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Custom tasks, `run.sh`, and a real LightEval smoke run

**Files:**
- Create: `eval/lighteval/ferrum_tasks.py`, `eval/lighteval/make_task_data.py`, `eval/lighteval/run.sh`

**Interfaces:**
- Consumes: `FerrumModel` (Task 3), `ferrum-serve` (Task 1), and a test split (`data/prepared/test.txt` from Phase 1B, or any text file with `<|endoftext|>` separators).
- Produces: `eval/lighteval/run.sh --model M --test T --out O [--max-samples N] [--wikitext] [--port P]`, which writes `O` as `{"tasks": {"<task>": {"<metric>": value, …}}, "lighteval_version": "0.13.0", "model": "<path>", "max_samples": N|null}`.

- [ ] **Step 1: `make_task_data.py`** — builds the local datasets from the test split (deterministic, seeded):

```python
"""Build LightEval task data from a Ferrum test split.

    python make_task_data.py <test.txt> <out_dir>

Writes <out_dir>/heldout/test.jsonl ({"text"}) and
<out_dir>/cloze/test.jsonl ({"question","choices","gold"}): story prefix →
true next sentence vs 3 sentences from other stories."""
import json
import random
import re
import sys
from pathlib import Path

SEP = "<|endoftext|>"


def sentences(doc):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", doc) if s.strip()]


def main(test_path, out_dir):
    docs = [d.strip() for d in Path(test_path).read_text(encoding="utf-8").split(SEP) if d.strip()]
    out = Path(out_dir)
    (out / "heldout").mkdir(parents=True, exist_ok=True)
    (out / "cloze").mkdir(parents=True, exist_ok=True)
    with open(out / "heldout" / "test.jsonl", "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps({"text": d}) + "\n")
    rng = random.Random(1234)
    pool = [s for d in docs for s in sentences(d)]
    with open(out / "cloze" / "test.jsonl", "w", encoding="utf-8") as f:
        for d in docs:
            ss = sentences(d)
            if len(ss) < 3:
                continue
            k = rng.randrange(2, len(ss))
            gold = ss[k]
            distractors = []
            while len(distractors) < 3:
                c = rng.choice(pool)
                if c != gold and c not in distractors:
                    distractors.append(c)
            choices = distractors + [gold]
            rng.shuffle(choices)
            f.write(json.dumps({"question": " ".join(ss[:k]), "choices": choices, "gold": choices.index(gold)}) + "\n")
    print(f"{len(docs)} heldout docs → {out}/heldout, cloze → {out}/cloze")


if __name__ == "__main__":
    main(*sys.argv[1:3])
```

- [ ] **Step 2: `ferrum_tasks.py`** (the pattern the research agent verified on 0.13.0: `hf_repo` pointing at a local directory with `test.jsonl`):

```python
"""Custom LightEval tasks for Ferrum SLMs. Data is built by make_task_data.py
into $FERRUM_TASK_DATA (default eval/lighteval/out/data)."""
import os

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc

DATA = os.environ.get("FERRUM_TASK_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "data"))


def heldout_prompt(line, task_name=None):
    return Doc(task_name=task_name, query=line["text"], original_query=line["text"], choices=None, gold_index=None)


def cloze_prompt(line, task_name=None):
    return Doc(task_name=task_name, query=line["question"], choices=[" " + c for c in line["choices"]], gold_index=line["gold"])


def _local(sub):
    return dict(
        hf_repo=os.path.join(DATA, sub), hf_subset="default", hf_avail_splits=["test"],
        evaluation_splits=["test"], few_shots_split=None, few_shots_select=None, generation_size=-1,
    )


TASKS_TABLE = [
    LightevalTaskConfig(
        name="ferrum_heldout", prompt_function=heldout_prompt,
        metrics=[Metrics.word_perplexity, Metrics.byte_perplexity, Metrics.bits_per_byte], **_local("heldout"),
    ),
    LightevalTaskConfig(
        name="tinystories_cloze", prompt_function=cloze_prompt,
        metrics=[Metrics.loglikelihood_acc], **_local("cloze"),
    ),
]
```

- [ ] **Step 3: `run.sh`**

```bash
#!/usr/bin/env bash
# Evaluate a Ferrum SLM with LightEval 0.13.0 through ferrum-serve.
# Usage: eval/lighteval/run.sh --model M --test T --out O [--max-samples N] [--wikitext] [--port P]
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
model= test= out= max= port=8765 wikitext=0
while [ $# -gt 0 ]; do
  case $1 in
    --model) model=$2; shift 2;;
    --test) test=$2; shift 2;;
    --out) out=$2; shift 2;;
    --max-samples) max=$2; shift 2;;
    --port) port=$2; shift 2;;
    --wikitext) wikitext=1; shift;;
    *) echo "unknown argument $1" >&2; exit 2;;
  esac
done
: "${model:?--model required}" "${test:?--test required}" "${out:?--out required}"
py="$here/.venv/bin/python"; le="$here/.venv/bin/lighteval"
[ -x "$le" ] || { echo "run: python3 -m venv $here/.venv && $here/.venv/bin/pip install -r $here/requirements.txt" >&2; exit 1; }

work="$here/out"; rm -rf "$work"; mkdir -p "$work"
"$py" "$here/make_task_data.py" "$test" "$work/data"

(cd "$root" && cargo build --release -q -p slm_server)
"$root/target/release/ferrum-serve" --model "$model" --port "$port" 2>"$work/server.log" &
srv=$!
trap 'kill $srv 2>/dev/null || true' EXIT
for _ in $(seq 1 100); do curl -sf "http://127.0.0.1:$port/health" >/dev/null && break; sleep 0.1; done
curl -sf "http://127.0.0.1:$port/health" >/dev/null || { cat "$work/server.log" >&2; exit 1; }

tasks="ferrum_heldout|0,tinystories_cloze|0,blimp:anaphor_gender_agreement|0,blimp:determiner_noun_agreement_1|0,blimp:regular_plural_subject_verb_agreement_1|0,arc:easy|0"
[ "$wikitext" = 1 ] && tasks="$tasks,wikitext:103:document_level|0"
extra=(); [ -n "$max" ] && extra=(--max-samples "$max")

FERRUM_URL="http://127.0.0.1:$port" FERRUM_TASK_DATA="$work/data" \
  "$le" custom ferrum "$here/ferrum_model.py" "$tasks" \
  --custom-tasks "$here/ferrum_tasks.py" --output-dir "$work/results" "${extra[@]}"

"$py" - "$work/results" "$out" "$model" "${max:-}" <<'EOF'
import glob, json, os, sys
res_dir, out, model, maxs = sys.argv[1:5]
f = max(glob.glob(os.path.join(res_dir, "**", "results_*.json"), recursive=True), key=os.path.getmtime)
r = json.load(open(f))["results"]
tasks = {k: v for k, v in r.items() if k != "all"}
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
json.dump({"lighteval_version": "0.13.0", "model": model, "max_samples": int(maxs) if maxs else None, "tasks": tasks},
          open(out, "w"), indent=2, sort_keys=True)
print(json.dumps(tasks, indent=2, sort_keys=True))
EOF
```

`chmod +x eval/lighteval/run.sh`. Before relying on the task names:
- check that the `blimp:` and `arc:easy` names exist in 0.13.0 with `eval/lighteval/.venv/bin/lighteval tasks list 2>/dev/null | grep -E "blimp:(anaphor_gender|determiner_noun_agreement_1|regular_plural_subject_verb_agreement_1)|arc:easy"`
- replace any name that is missing with the listed one
- check the `custom` CLI syntax with `lighteval custom --help`, and adjust if 0.13.0 names an argument differently

- [ ] **Step 4: Smoke run on a real model**

```bash
S=/tmp/claude-1000/-home-thomas-Projects-ferrum-cpu/88623770-ced2-4a37-be7f-75cb7f78b99e/scratchpad/review
# Build a small test split with document separators from the review's held-out text.
python3 -c "
import re,sys
t=open('$S/heldout.txt').read()
docs=[d.strip() for d in re.split(r'\n\s*\n', t) if len(d.strip())>80][:60]
open('$S/heldout_docs.txt','w').write('\n<|endoftext|>\n'.join(docs))"
[ -f $S/ref.gguf ] || ./target/release/train_transformer train $S/train.txt $S/ref.gguf --context 128 --embed 96 --heads 4 --blocks 3 --hidden 320 --vocab 1024 --tokens 1.2e6 --lr 5e-3 --cosine --warmup 30 --clip 1 --tokenizer $S/ref_tok.gguf
eval/lighteval/run.sh --model $S/ref.gguf --test $S/heldout_docs.txt --out $S/lighteval.json --max-samples 5
cat $S/lighteval.json
```

Expected:
- `ferrum_heldout` reports finite `word_perplexity`, `byte_perplexity` and `bits_per_byte`.
- `tinystories_cloze` and the `blimp:*`/`arc:easy` tasks report `loglikelihood_acc` (or `acc`) between 0 and 1.
- The first run downloads the BLiMP and ARC datasets from the Hub.
- `$S/ref.gguf` is the 507,552-param reference model trained above.

- [ ] **Step 5: Commit**

```bash
git add eval/lighteval/ferrum_tasks.py eval/lighteval/make_task_data.py eval/lighteval/run.sh
git commit -m "feat(eval): LightEval custom tasks + run.sh (heldout ppl, cloze, BLiMP, ARC-easy)

Smoke run (--max-samples 5) on the 507,552-param reference model:
<paste lighteval.json tasks block>

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: CI and Phase 1C gate

**Files:**
- Modify: `.github/workflows/ci.yml` (nothing new is needed: `slm_server` is a workspace member, so build/test/clippy/fmt/deny already cover it. Add a step running `python3 -m unittest eval/lighteval/test_ferrum_model.py` in the `lint` job; it skips without lighteval and confirms the file imports.)

- [ ] Full battery from the Phase 0 plan's Task 6 Step 1, plus `cargo deny check` (add any new permissive licences from axum/tokio's tree to `deny.toml` only if they are MIT/Apache-2.0/BSD/ISC/Zlib/Unicode-3.0).
- [ ] Commit the CI change:

```bash
git add .github/workflows/ci.yml deny.toml
git commit -m "ci: cover slm_server and the LightEval module import

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] Report to the controller: the smoke-run JSON, any LightEval task names that differed from this plan, and the `deny.toml` changes.
