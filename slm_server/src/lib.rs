//! HTTP inference server for Ferrum SLMs: loglikelihood, rolling
//! loglikelihood, and generation — the three primitives eval harnesses such as
//! LightEval need. One thread owns the model (it is not `Sync`); requests are
//! queued to it over a channel and answered through oneshot replies.
#![forbid(unsafe_code)]

pub mod cli;

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

type Reply = oneshot::Sender<Result<Vec<Value>, String>>;

enum Job {
    Score(Vec<(String, String)>, Reply),
    Rolling(Vec<String>, Reply),
    Generate(Vec<GenRequest>, Reply),
}

#[derive(Clone)]
struct AppState {
    jobs: mpsc::Sender<Job>,
    info: Arc<Info>,
    max_batch: usize,
    max_chars: usize,
}

/// Apply `f` to every item, stopping at the first model error.
fn collect<T>(
    items: &[T],
    f: impl Fn(&T) -> ferrum_core::Result<Value>,
) -> Result<Vec<Value>, String> {
    items
        .iter()
        .map(f)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())
}

/// Answer one job on the model thread.
fn handle(model: &GenerativeSLM, job: Job) {
    match job {
        Job::Score(reqs, reply) => {
            let r = collect(&reqs, |(c, k)| {
                model.score(c, k).map(
                    |s| json!({"logprob": s.logprob, "n_tokens": s.n_tokens, "greedy": s.greedy}),
                )
            });
            let _ = reply.send(r);
        }
        Job::Rolling(texts, reply) => {
            let r = collect(&texts, |t| {
                model
                    .score_rolling(t)
                    .map(|s| json!({"logprob": s.logprob, "n_tokens": s.n_tokens}))
            });
            let _ = reply.send(r);
        }
        Job::Generate(reqs, reply) => {
            let r = collect(&reqs, |g| {
                let params = SamplingParams {
                    temperature: g.temperature,
                    top_k: g.top_k,
                    top_p: g.top_p,
                    repetition_penalty: 1.0,
                };
                model
                    .generate_tokens(
                        &g.prompt,
                        g.max_tokens,
                        &g.stop,
                        &params,
                        &mut Rng::new(g.seed),
                    )
                    .map(|o| json!({"text": o.text, "n_tokens": o.n_tokens}))
            });
            let _ = reply.send(r);
        }
    }
}

/// Load the model from `bytes` on a dedicated thread and answer jobs there
/// until every sender is dropped. The thread must build the model itself:
/// `GenerativeSLM` is not `Send`. Returns the job channel and the model's info.
fn spawn_model_thread(bytes: Vec<u8>) -> Result<(mpsc::Sender<Job>, Info), String> {
    let (tx, rx) = mpsc::channel::<Job>();
    let (ready_tx, ready_rx) = mpsc::channel::<Result<Info, String>>();
    std::thread::Builder::new()
        .name("ferrum-model".into())
        .spawn(move || {
            let model = match GenerativeSLM::from_bytes(&bytes) {
                Ok(m) => m,
                Err(e) => {
                    let _ = ready_tx.send(Err(e.to_string()));
                    return;
                }
            };
            let _ = ready_tx.send(Ok(info_for(&model, &bytes)));
            drop(bytes);
            for job in rx {
                handle(&model, job);
            }
        })
        .map_err(|e| format!("cannot spawn model thread: {e}"))?;
    let info = ready_rx
        .recv()
        .map_err(|_| "model thread died while loading".to_string())??;
    Ok((tx, info))
}

/// Load the model in `bytes` (a model file) and build the router, spawning the
/// model thread. Also returns the model's [`Info`].
pub fn app(bytes: Vec<u8>, limits: Limits) -> Result<(Router, Info), String> {
    let (jobs, info) = spawn_model_thread(bytes)?;
    let state = AppState {
        jobs,
        info: Arc::new(info.clone()),
        max_batch: limits.max_batch,
        max_chars: limits.max_chars,
    };
    let router = Router::new()
        .route("/health", get(|| async { Json("ok") }))
        .route(
            "/v1/info",
            get(|State(s): State<AppState>| async move { Json((*s.info).clone()) }),
        )
        .route("/v1/loglikelihood", post(loglikelihood))
        .route("/v1/loglikelihood_rolling", post(loglikelihood_rolling))
        .route("/v1/generate", post(generate))
        .layer(DefaultBodyLimit::max(limits.max_body))
        .with_state(state);
    Ok((router, info))
}

/// JSON error response.
#[derive(Debug)]
struct ApiError(StatusCode, String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(json!({"error": self.1}))).into_response()
    }
}

/// Map axum's JSON rejection: oversized body -> 413, anything else -> 400.
fn reject(e: JsonRejection) -> ApiError {
    let status = if e.status() == StatusCode::PAYLOAD_TOO_LARGE {
        StatusCode::PAYLOAD_TOO_LARGE
    } else {
        StatusCode::BAD_REQUEST
    };
    ApiError(status, e.body_text())
}

/// Reject a batch of `n` texts with the given char lengths if it exceeds the limits.
fn check(s: &AppState, n: usize, lens: impl Iterator<Item = usize>) -> Result<(), ApiError> {
    if n > s.max_batch {
        return Err(ApiError(
            StatusCode::BAD_REQUEST,
            format!("batch of {n} exceeds max_batch {}", s.max_batch),
        ));
    }
    if let Some(len) = lens.into_iter().find(|&l| l > s.max_chars) {
        return Err(ApiError(
            StatusCode::BAD_REQUEST,
            format!("text of {len} chars exceeds max_chars {}", s.max_chars),
        ));
    }
    Ok(())
}

/// Queue a job on the model thread and await its reply.
async fn submit(s: &AppState, make: impl FnOnce(Reply) -> Job) -> Result<Json<Value>, ApiError> {
    let stopped = || {
        ApiError(
            StatusCode::SERVICE_UNAVAILABLE,
            "model thread stopped".into(),
        )
    };
    let (tx, rx) = oneshot::channel();
    s.jobs.send(make(tx)).map_err(|_| stopped())?;
    match rx.await {
        Ok(Ok(results)) => Ok(Json(json!({"results": results}))),
        Ok(Err(e)) => Err(ApiError(StatusCode::UNPROCESSABLE_ENTITY, e)),
        Err(_) => Err(stopped()),
    }
}

#[derive(Deserialize)]
struct LlRequest {
    context: String,
    continuation: String,
}
#[derive(Deserialize)]
struct LlBody {
    requests: Vec<LlRequest>,
}

async fn loglikelihood(
    State(s): State<AppState>,
    body: Result<Json<LlBody>, JsonRejection>,
) -> Result<Json<Value>, ApiError> {
    let Json(b) = body.map_err(reject)?;
    let lens = b
        .requests
        .iter()
        .map(|r| r.context.chars().count() + r.continuation.chars().count());
    check(&s, b.requests.len(), lens)?;
    let reqs = b
        .requests
        .into_iter()
        .map(|r| (r.context, r.continuation))
        .collect();
    submit(&s, |tx| Job::Score(reqs, tx)).await
}

#[derive(Deserialize)]
struct RollingBody {
    texts: Vec<String>,
}

async fn loglikelihood_rolling(
    State(s): State<AppState>,
    body: Result<Json<RollingBody>, JsonRejection>,
) -> Result<Json<Value>, ApiError> {
    let Json(b) = body.map_err(reject)?;
    check(&s, b.texts.len(), b.texts.iter().map(|t| t.chars().count()))?;
    submit(&s, |tx| Job::Rolling(b.texts, tx)).await
}

fn default_max_tokens() -> usize {
    64
}
fn default_top_p() -> f32 {
    1.0
}

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
struct GenBody {
    requests: Vec<GenRequest>,
}

async fn generate(
    State(s): State<AppState>,
    body: Result<Json<GenBody>, JsonRejection>,
) -> Result<Json<Value>, ApiError> {
    let Json(b) = body.map_err(reject)?;
    check(
        &s,
        b.requests.len(),
        b.requests.iter().map(|r| r.prompt.chars().count()),
    )?;
    if let Some(r) = b.requests.iter().find(|r| r.max_tokens > s.max_chars) {
        return Err(ApiError(
            StatusCode::BAD_REQUEST,
            format!(
                "max_tokens {} exceeds max_chars {}",
                r.max_tokens, s.max_chars
            ),
        ));
    }
    submit(&s, |tx| Job::Generate(b.requests, tx)).await
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

    fn state(jobs: mpsc::Sender<Job>) -> AppState {
        AppState {
            jobs,
            info: Arc::new(info_for(&model(0), b"")),
            max_batch: 2,
            max_chars: 3,
        }
    }

    #[test]
    fn check_accepts_limits_and_rejects_over() {
        let (tx, _rx) = mpsc::channel();
        let s = state(tx);
        assert!(check(&s, 0, std::iter::empty()).is_ok());
        assert!(check(&s, 2, [3, 1].into_iter()).is_ok());
        let e = check(&s, 3, std::iter::empty()).unwrap_err();
        assert_eq!(e.0, StatusCode::BAD_REQUEST);
        assert!(e.1.contains("max_batch 2"));
        let e = check(&s, 1, [4].into_iter()).unwrap_err();
        assert_eq!(e.0, StatusCode::BAD_REQUEST);
        assert!(e.1.contains("4 chars") && e.1.contains("max_chars 3"));
    }

    #[tokio::test]
    async fn api_error_renders_json_body() {
        use http_body_util::BodyExt;
        let resp = ApiError(StatusCode::IM_A_TEAPOT, "boom".into()).into_response();
        assert_eq!(resp.status(), StatusCode::IM_A_TEAPOT);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        assert_eq!(
            serde_json::from_slice::<Value>(&bytes).unwrap(),
            json!({"error": "boom"})
        );
    }

    #[tokio::test]
    async fn reject_maps_413_and_everything_else_to_400() {
        use axum::extract::FromRequest;
        use axum::http::Request;
        // A body over the limit: the length-limited body errors while being read.
        let too_big = Request::builder()
            .header("content-type", "application/json")
            .body(axum::body::Body::new(http_body_util::Limited::new(
                axum::body::Body::from("x".repeat(100)),
                10,
            )))
            .unwrap();
        let rej = Json::<Value>::from_request(too_big, &()).await.unwrap_err();
        let e = reject(rej);
        assert_eq!(e.0, StatusCode::PAYLOAD_TOO_LARGE);
        assert!(!e.1.is_empty());
        let bad = Request::builder()
            .header("content-type", "application/json")
            .body(axum::body::Body::from("{nope"))
            .unwrap();
        let rej = Json::<Value>::from_request(bad, &()).await.unwrap_err();
        assert_eq!(reject(rej).0, StatusCode::BAD_REQUEST);
        let no_ct = Request::builder()
            .body(axum::body::Body::from("{}"))
            .unwrap();
        let rej = Json::<Value>::from_request(no_ct, &()).await.unwrap_err();
        assert_eq!(reject(rej).0, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn submit_reports_stopped_model_thread() {
        // Receiver gone: send fails.
        let (tx, rx) = mpsc::channel();
        drop(rx);
        let e = submit(&state(tx), |r| Job::Rolling(vec![], r))
            .await
            .unwrap_err();
        assert_eq!(
            (e.0, e.1.as_str()),
            (StatusCode::SERVICE_UNAVAILABLE, "model thread stopped")
        );
        // Thread takes the job but drops the reply without answering.
        let (tx, rx) = mpsc::channel();
        let t = std::thread::spawn(move || drop(rx.recv().unwrap()));
        let e = submit(&state(tx), |r| Job::Rolling(vec![], r))
            .await
            .unwrap_err();
        t.join().unwrap();
        assert_eq!(
            (e.0, e.1.as_str()),
            (StatusCode::SERVICE_UNAVAILABLE, "model thread stopped")
        );
    }

    #[tokio::test]
    async fn submit_maps_ok_and_model_error() {
        let (tx, rx) = mpsc::channel::<Job>();
        let t = std::thread::spawn(move || {
            for job in rx {
                let (texts, reply) = match job {
                    Job::Rolling(t, r) => (t, r),
                    Job::Score(_, r) | Job::Generate(_, r) => (vec!["other".into()], r),
                };
                let _ = reply.send(if texts.is_empty() {
                    Ok(vec![json!(1)])
                } else {
                    Err("bad".into())
                });
            }
        });
        let s = state(tx);
        let Json(v) = submit(&s, |r| Job::Rolling(vec![], r)).await.unwrap();
        assert_eq!(v, json!({"results": [1]}));
        let e = submit(&s, |r| Job::Rolling(vec!["x".into()], r))
            .await
            .unwrap_err();
        assert_eq!(
            (e.0, e.1.as_str()),
            (StatusCode::UNPROCESSABLE_ENTITY, "bad")
        );
        let e = submit(&s, |r| Job::Score(vec![], r)).await.unwrap_err();
        assert_eq!(
            (e.0, e.1.as_str()),
            (StatusCode::UNPROCESSABLE_ENTITY, "bad")
        );
        let e = submit(&s, |r| Job::Generate(vec![], r)).await.unwrap_err();
        assert_eq!(e.0, StatusCode::UNPROCESSABLE_ENTITY);
        drop(s);
        t.join().unwrap();
    }

    #[test]
    fn spawn_model_thread_rejects_bad_bytes_and_serves_good() {
        assert!(spawn_model_thread(b"nope".to_vec()).is_err());
        let m = model(0);
        let (tx, info) = spawn_model_thread(m.to_bytes_quantized().unwrap()).unwrap();
        assert_eq!(info.tokenizer, "char");
        let (rtx, rrx) = oneshot::channel();
        tx.send(Job::Score(vec![("abc".into(), " abc".into())], rtx))
            .unwrap();
        let r = rrx.blocking_recv().unwrap().unwrap();
        assert!(r[0]["logprob"].as_f64().unwrap() < 0.0);
    }

    #[test]
    fn collect_stops_at_first_error() {
        let ok = collect(&[1, 2], |n| Ok(json!(n))).unwrap();
        assert_eq!(ok, vec![json!(1), json!(2)]);
        let err = collect(&[1, 2], |n| {
            if *n == 2 {
                Err(ferrum_core::InferError::Format("bad".into()))
            } else {
                Ok(json!(n))
            }
        })
        .unwrap_err();
        assert!(!err.is_empty());
        assert!(collect::<u8>(&[], |_| unreachable!()).unwrap().is_empty());
    }

    #[test]
    fn default_helpers() {
        assert_eq!((default_max_tokens(), default_top_p()), (64, 1.0));
    }
}
