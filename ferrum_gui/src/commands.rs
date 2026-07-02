//! Tauri commands: the bridge between the webview and `ferrum_core`.
//!
//! Conventions:
//! - Every command returns `Result<T, String>` so failures surface as a clear,
//!   human-readable message in the GUI (requirement: clear error messages).
//! - Heavy/blocking work runs inside `spawn_blocking` so the UI thread stays
//!   responsive and events can stream while the command runs.

use crate::AppState;
use ferrum_core::{
    clean_corpus, corpus_stats, validate_for_training, Adam, CleanOptions, GenerativeSLM, Gguf,
    GgufTokenizer, LlamaTrainer, LrSchedule, QKind, Rng, SamplingParams, TaskType,
};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};
use tauri::{AppHandle, Emitter, Manager, State};

// ─────────────────────────────────────────────────────────────────────────────
// Small helpers
// ─────────────────────────────────────────────────────────────────────────────

fn time_seed() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0xC0FFEE)
        | 1
}

fn file_name(path: &str) -> String {
    Path::new(path)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string())
}

/// True on platforms where spawning an OS shell / reading the process table is
/// not possible (mobile + the web preview).
fn is_sandboxed() -> bool {
    cfg!(any(target_os = "android", target_os = "ios"))
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Datasets — download + clean (requirement #1)
// ─────────────────────────────────────────────────────────────────────────────

/// HTTP client with bounded connect/read timeouts so a hung or slow host can
/// never block a download task indefinitely (G2).
pub(crate) fn http_agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout_connect(std::time::Duration::from_secs(15))
        .timeout_read(std::time::Duration::from_secs(30))
        .timeout_write(std::time::Duration::from_secs(30))
        .build()
}

/// Synchronous core of [`download_text`]: validate the URL, fetch with timeouts,
/// and return at most `cap` bytes decoded as UTF-8. Separated from the Tauri
/// command so it is unit-testable without an `AppHandle`.
fn fetch_text(url: &str, cap: usize) -> Result<String, String> {
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        return Err("URL must start with http:// or https://".to_string());
    }
    let cap = cap.max(1);
    let resp = http_agent()
        .get(url)
        .call()
        .map_err(|e| format!("download failed: {e}"))?;
    let mut buf = Vec::new();
    resp.into_reader()
        .take(cap as u64)
        .read_to_end(&mut buf)
        .map_err(|e| format!("read failed: {e}"))?;
    if buf.is_empty() {
        return Err("downloaded resource was empty".to_string());
    }
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

/// Download a text resource (e.g. a Project Gutenberg `.txt`) and return it as a
/// UTF-8 string (lossily decoded), capped at `max_bytes` (default 8 MiB). Uses an
/// HTTP client with connect/read timeouts so a slow host cannot hang the task.
#[tauri::command]
pub async fn download_text(url: String, max_bytes: Option<usize>) -> Result<String, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let cap = max_bytes.unwrap_or(8 * 1024 * 1024);
        fetch_text(&url, cap)
    })
    .await
    .map_err(|e| format!("task error: {e}"))?
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CleanOpts {
    pub strip_gutenberg: bool,
    pub lowercase: bool,
    pub collapse_whitespace: bool,
    pub normalize_punctuation: bool,
    pub strip_control_chars: bool,
    pub max_chars: Option<usize>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CleanResult {
    pub cleaned: String,
    pub preview: String,
    pub chars: usize,
    pub bytes: usize,
    pub lines: usize,
    pub words: usize,
    pub unique_chars: usize,
}

/// Clean raw text into a training-ready corpus and report statistics. Pure;
/// does not touch disk.
#[tauri::command]
pub fn clean_text(raw: String, opts: CleanOpts) -> Result<CleanResult, String> {
    let options = CleanOptions {
        strip_gutenberg: opts.strip_gutenberg,
        lowercase: opts.lowercase,
        collapse_whitespace: opts.collapse_whitespace,
        normalize_punctuation: opts.normalize_punctuation,
        strip_control_chars: opts.strip_control_chars,
        max_chars: opts.max_chars.filter(|&n| n > 0),
    };
    let cleaned = clean_corpus(&raw, &options);
    if cleaned.trim().is_empty() {
        return Err("nothing left after cleaning — check the source and options".into());
    }
    let st = corpus_stats(&cleaned);
    let preview: String = cleaned.chars().take(4000).collect();
    Ok(CleanResult {
        preview,
        chars: st.chars,
        bytes: st.bytes,
        lines: st.lines,
        words: st.words,
        unique_chars: st.unique_chars,
        cleaned,
    })
}

/// Write a corpus to disk (creating parent directories), returning bytes written.
#[tauri::command]
pub fn save_corpus(text: String, path: String) -> Result<usize, String> {
    if path.trim().is_empty() {
        return Err("please provide an output file path".into());
    }
    if let Some(parent) = Path::new(&path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("cannot create {parent:?}: {e}"))?;
        }
    }
    std::fs::write(&path, text.as_bytes()).map_err(|e| format!("cannot write {path}: {e}"))?;
    Ok(text.len())
}

/// Read a UTF-8 text file (lossy), capped at `max_bytes` (default 16 MiB).
#[tauri::command]
pub fn read_text_file(path: String, max_bytes: Option<usize>) -> Result<String, String> {
    let data = std::fs::read(&path).map_err(|e| format!("cannot read {path}: {e}"))?;
    let cap = max_bytes.unwrap_or(16 * 1024 * 1024).min(data.len());
    Ok(String::from_utf8_lossy(&data[..cap]).into_owned())
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Training (requirement #6) — transformer / embedded / one-hot
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainParams {
    /// "transformer" | "embedded" | "onehot"
    pub method: String,
    pub corpus_path: String,
    pub model_path: String,
    pub context_len: usize,
    pub embed_dim: usize,
    pub num_heads: usize,
    pub num_blocks: usize,
    pub hidden_dim: usize,
    pub epochs: usize,
    pub lr: f32,
    pub momentum: f32,
    pub batch_size: usize,
    pub vocab_size: usize,
    pub seed: u64,
    pub threads: usize,
    pub verbose: bool,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct TrainResult {
    pub model_path: String,
    pub method: String,
    pub seconds: f64,
    pub final_loss: f32,
    pub input_dim: usize,
    pub output_dim: usize,
    pub tokenizer: String,
    pub layers: usize,
    pub bytes: u64,
}

/// Train an SLM in-process (any of the three engine paths), streaming
/// `train-progress` events and (if `verbose`) `engine-log` lines, then save it.
#[tauri::command]
pub async fn train_slm(app: AppHandle, params: TrainParams) -> Result<TrainResult, String> {
    tauri::async_runtime::spawn_blocking(move || train_inner(app, params))
        .await
        .map_err(|e| format!("task error: {e}"))?
}

fn train_inner(app: AppHandle, p: TrainParams) -> Result<TrainResult, String> {
    // ---- shared validation with clear messages ----
    if p.model_path.trim().is_empty() {
        return Err("please provide an output model path".into());
    }
    if p.context_len == 0 {
        return Err("context length must be ≥ 1".into());
    }
    if p.epochs == 0 {
        return Err("epochs must be ≥ 1".into());
    }
    if p.batch_size == 0 {
        return Err("batch size must be ≥ 1".into());
    }
    if p.hidden_dim == 0 {
        return Err("hidden width must be ≥ 1".into());
    }
    if !(p.lr.is_finite() && p.lr > 0.0) {
        return Err("learning rate must be a positive number".into());
    }
    let corpus = std::fs::read_to_string(&p.corpus_path)
        .map_err(|e| format!("cannot read corpus {}: {e}", p.corpus_path))?;
    if corpus.trim().is_empty() {
        return Err(format!("corpus {} is empty", p.corpus_path));
    }
    validate_for_training(&corpus, p.context_len).map_err(|e| e.to_string())?;

    ferrum_core::set_verbose(p.verbose);

    // Capture the most recent loss for the result without changing the callback
    // signature.
    let last_loss = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
    let last_cb = last_loss.clone();
    let app_cb = app.clone();
    let total = p.epochs;
    let progress = move |epoch: usize, loss: f32| {
        last_cb.store(loss.to_bits(), std::sync::atomic::Ordering::Relaxed);
        let _ = app_cb.emit(
            "train-progress",
            serde_json::json!({ "epoch": epoch, "total": total, "loss": loss }),
        );
    };

    let mut rng = Rng::new(p.seed);
    let t0 = std::time::Instant::now();

    let slm = match p.method.as_str() {
        "transformer" => {
            if p.num_heads == 0 || !p.embed_dim.is_multiple_of(p.num_heads) {
                ferrum_core::set_verbose(false);
                return Err(format!(
                    "embedding dim ({}) must be divisible by heads ({})",
                    p.embed_dim, p.num_heads
                ));
            }
            if p.num_blocks == 0 {
                ferrum_core::set_verbose(false);
                return Err("transformer blocks must be ≥ 1".into());
            }
            if p.vocab_size != 0 && p.vocab_size < 256 {
                ferrum_core::set_verbose(false);
                return Err("vocab must be 0 (character-level) or ≥ 256 (byte-level BPE)".into());
            }
            GenerativeSLM::train_transformer_threaded_with_callback(
                &corpus,
                p.context_len,
                p.embed_dim,
                p.num_heads,
                p.num_blocks,
                p.hidden_dim,
                p.epochs,
                p.lr,
                p.batch_size,
                p.vocab_size,
                p.threads,
                &mut rng,
                progress,
            )
        }
        "embedded" => {
            if p.vocab_size != 0 && p.vocab_size < 256 {
                ferrum_core::set_verbose(false);
                return Err("vocab must be 0 (character-level) or ≥ 256 (byte-level BPE)".into());
            }
            GenerativeSLM::train_embedded_with_callback(
                &corpus,
                p.context_len,
                p.embed_dim,
                p.hidden_dim,
                p.epochs,
                p.lr,
                p.momentum,
                p.batch_size,
                p.vocab_size,
                &mut rng,
                progress,
            )
        }
        "onehot" => GenerativeSLM::train_with_callback(
            &corpus,
            p.context_len,
            p.hidden_dim,
            p.epochs,
            p.lr,
            p.momentum,
            p.batch_size,
            &mut rng,
            progress,
        ),
        other => {
            ferrum_core::set_verbose(false);
            return Err(format!("unknown training method: {other}"));
        }
    }
    .map_err(|e| {
        ferrum_core::set_verbose(false);
        format!("training failed: {e}")
    })?;

    let seconds = t0.elapsed().as_secs_f64();
    slm.save(&p.model_path)
        .map_err(|e| format!("cannot save model {}: {e}", p.model_path))?;
    ferrum_core::set_verbose(false);

    let bytes = std::fs::metadata(&p.model_path)
        .map(|m| m.len())
        .unwrap_or(0);
    let tokenizer = if slm.meta.tokenizer_state.is_empty() {
        "character-level".to_string()
    } else {
        "byte-level BPE".to_string()
    };
    let result = TrainResult {
        model_path: p.model_path.clone(),
        method: p.method.clone(),
        seconds,
        final_loss: f32::from_bits(last_loss.load(std::sync::atomic::Ordering::Relaxed)),
        input_dim: slm.meta.input_dim,
        output_dim: slm.meta.output_dim,
        tokenizer,
        layers: slm.model.len(),
        bytes,
    };
    let _ = app.emit("train-done", result.clone());
    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Generation with streaming (requirement #6)
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenParams {
    pub model_path: String,
    pub seed_text: String,
    pub num_chars: usize,
    pub temp: f32,
    pub gen_seed: Option<u64>,
    pub stream: bool,
    pub verbose: bool,
}

/// Load a model and generate text. With `stream = true`, each fragment is
/// emitted as a `gen-fragment` event as it is produced; the full string is also
/// returned.
#[tauri::command]
pub async fn generate_slm(app: AppHandle, params: GenParams) -> Result<String, String> {
    tauri::async_runtime::spawn_blocking(move || gen_inner(app, params))
        .await
        .map_err(|e| format!("task error: {e}"))?
}

fn gen_inner(app: AppHandle, p: GenParams) -> Result<String, String> {
    if !(p.temp.is_finite() && p.temp > 0.0) {
        return Err("temperature must be a positive number".into());
    }
    let slm = GenerativeSLM::load(&p.model_path)
        .map_err(|e| format!("cannot load model {}: {e}", p.model_path))?;

    // Character-level models need a seed at least as long as the context window.
    if slm.meta.tokenizer_state.is_empty() {
        let ctx = if slm.meta.task == TaskType::TransformerSLM {
            slm.meta.input_dim
        } else {
            slm.meta.input_dim / slm.meta.output_dim.max(1)
        };
        let seed_chars = p.seed_text.chars().count();
        if seed_chars < ctx {
            return Err(format!(
                "seed is {seed_chars} characters but this model needs at least {ctx} \
                 (its context window). Provide a longer seed."
            ));
        }
    }

    ferrum_core::set_verbose(p.verbose);
    let mut rng = Rng::new(p.gen_seed.unwrap_or_else(time_seed));
    let result = if p.stream {
        let app_cb = app.clone();
        slm.generate_stream(&p.seed_text, p.num_chars, p.temp, &mut rng, move |frag| {
            let _ = app_cb.emit("gen-fragment", frag.to_string());
        })
    } else {
        slm.generate(&p.seed_text, p.num_chars, p.temp, &mut rng)
    };
    ferrum_core::set_verbose(false);
    result.map_err(|e| format!("generation failed: {e}"))
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Evaluation (requirement #5) — perplexity table rows
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EvalParams {
    pub model_path: String,
    pub text: Option<String>,
    pub text_path: Option<String>,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct EvalRow {
    pub model: String,
    pub num_predictions: usize,
    pub cross_entropy: f32,
    pub bits_per_token: f32,
    pub perplexity: f32,
    pub uniform_baseline: f32,
}

/// Score a model against held-out text; returns one table row.
#[tauri::command]
pub async fn evaluate_slm(params: EvalParams) -> Result<EvalRow, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let text = match (&params.text, &params.text_path) {
            (Some(t), _) if !t.trim().is_empty() => t.clone(),
            (_, Some(path)) if !path.trim().is_empty() => {
                std::fs::read_to_string(path).map_err(|e| format!("cannot read {path}: {e}"))?
            }
            _ => return Err("provide held-out text or a text file to evaluate against".into()),
        };
        let slm = GenerativeSLM::load(&params.model_path)
            .map_err(|e| format!("cannot load model {}: {e}", params.model_path))?;
        let ev = slm
            .evaluate(&text)
            .map_err(|e| format!("evaluation failed: {e}"))?;
        Ok(EvalRow {
            model: file_name(&params.model_path),
            num_predictions: ev.num_predictions,
            cross_entropy: ev.cross_entropy,
            bits_per_token: ev.bits_per_token,
            perplexity: ev.perplexity,
            uniform_baseline: slm.meta.output_dim as f32,
        })
    })
    .await
    .map_err(|e| format!("task error: {e}"))?
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Model info / load / reload (requirement #6)
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelInfo {
    pub path: String,
    pub bytes: u64,
    pub format: String,
    pub name: String,
    pub task: String,
    pub input_dim: usize,
    pub output_dim: usize,
    pub tokenizer: String,
    pub merges: usize,
    pub layers: usize,
}

/// Inspect a saved model file (mirrors `slm_cli info`).
#[tauri::command]
pub fn model_info(path: String) -> Result<ModelInfo, String> {
    let raw = std::fs::read(&path).map_err(|e| format!("cannot read {path}: {e}"))?;
    let version = if raw.len() >= 8 {
        u32::from_le_bytes([raw[4], raw[5], raw[6], raw[7]])
    } else {
        0
    };
    let slm = GenerativeSLM::from_bytes(&raw).map_err(|e| format!("not a valid model: {e}"))?;
    let m = &slm.meta;
    let (tokenizer, merges) = if m.tokenizer_state.is_empty() {
        (
            format!("character-level ({} chars)", m.class_names.len()),
            0,
        )
    } else {
        let merges = m
            .tokenizer_state
            .split(';')
            .filter(|s| !s.is_empty())
            .count();
        (format!("byte-level BPE ({} tokens)", m.output_dim), merges)
    };
    Ok(ModelInfo {
        path: path.clone(),
        bytes: raw.len() as u64,
        format: format!(
            "FINF v{version}{}",
            if version == 5 {
                " (int8-quantized)"
            } else {
                " (f32)"
            }
        ),
        name: m.dataset_name.clone(),
        task: format!("{:?}", m.task),
        input_dim: m.input_dim,
        output_dim: m.output_dim,
        tokenizer,
        merges,
        layers: slm.model.len(),
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// 5b. GGUF import & run (Llama/Qwen checkpoints)
// ─────────────────────────────────────────────────────────────────────────────

/// Map a UI quant string to an import precision (`None` = keep f32).
fn parse_quant(q: &str) -> Result<Option<QKind>, String> {
    match q {
        "int4" | "q4" => Ok(Some(QKind::Int4)),
        "int8" | "q8" => Ok(Some(QKind::Int8)),
        "f32" | "none" => Ok(None),
        other => Err(format!("quant must be int4 | int8 | f32 (got '{other}')")),
    }
}

/// Estimate resident bytes for a loaded model from the GGUF directory: the token
/// embedding stays f32, everything else packs to the chosen precision.
fn gguf_resident_bytes(g: &Gguf, prec: Option<QKind>) -> usize {
    let mut total = 0usize;
    for t in &g.tensors {
        let n = t.num_elements();
        let bytes = if t.name == "token_embd.weight" {
            n * 4
        } else {
            match prec {
                Some(QKind::Int4) => n.div_ceil(2),
                Some(QKind::Int8) => n,
                None => n * 4,
            }
        };
        total = total.saturating_add(bytes);
    }
    total
}

/// Best-effort available RAM (Linux `/proc/meminfo`); `None` elsewhere.
fn available_memory_bytes() -> Option<usize> {
    // Cross-platform via sysinfo (0.33 reports bytes). The previous /proc/meminfo
    // reader returned None on macOS/Windows, silently disabling the RAM guard on
    // the very desktops the Tauri app also ships to.
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    let avail = sys.available_memory();
    (avail > 0).then_some(avail as usize)
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GgufInfo {
    pub path: String,
    pub bytes: u64,
    pub version: u32,
    pub architecture: String,
    pub num_tensors: usize,
    /// Total parameters across all tensors (sum of element counts).
    pub param_count: u64,
    pub model_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub vocab_size: usize,
    pub tokenizer: String,
    /// Whether `run_gguf` can execute this file (arch llama/qwen2).
    pub runnable: bool,
    pub note: String,
    pub est_int4_mb: f64,
    pub est_int8_mb: f64,
    pub est_f32_mb: f64,
    pub avail_mb: Option<f64>,
}

/// Inspect a GGUF file without loading its weights (streamed header parse).
#[tauri::command]
pub async fn gguf_info(path: String) -> Result<GgufInfo, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        let g = Gguf::open(&path).map_err(|e| format!("cannot open {path}: {e}"))?;
        let arch = g.architecture().unwrap_or("?").to_string();
        let meta_usize = |k: &str| g.meta(k).and_then(|v| v.as_usize()).unwrap_or(0);
        let dim = meta_usize(&format!("{arch}.embedding_length"));
        let n_layers = meta_usize(&format!("{arch}.block_count"));
        let n_heads = meta_usize(&format!("{arch}.attention.head_count"));
        let n_kv = g
            .meta(&format!("{arch}.attention.head_count_kv"))
            .and_then(|v| v.as_usize())
            .unwrap_or(n_heads);
        let vocab = g
            .tensor("token_embd.weight")
            .filter(|t| t.dims.len() == 2)
            .map(|t| t.dims[1] as usize)
            .unwrap_or_else(|| meta_usize(&format!("{arch}.vocab_size")));
        let tokenizer = match GgufTokenizer::from_gguf(&g) {
            Ok(t) => format!("{:?} ({} tokens)", t.model(), t.vocab_size()),
            Err(_) => "none in file (use token IDs)".to_string(),
        };
        let runnable = arch == "llama" || arch == "qwen2";
        let note = if !runnable {
            format!("architecture '{arch}' is not runnable (llama / qwen2 only)")
        } else {
            "decode is a few tok/s on CPU; prefer int8 for speed, int4 for memory".to_string()
        };
        let mb = |b: usize| b as f64 / 1e6;
        let param_count: u64 = g.tensors.iter().map(|t| t.num_elements() as u64).sum();
        Ok(GgufInfo {
            path: path.clone(),
            bytes,
            version: g.version,
            architecture: arch,
            num_tensors: g.tensors.len(),
            param_count,
            model_dim: dim,
            n_layers,
            n_heads,
            n_kv_heads: n_kv,
            vocab_size: vocab,
            tokenizer,
            runnable,
            note,
            est_int4_mb: mb(gguf_resident_bytes(&g, Some(QKind::Int4))),
            est_int8_mb: mb(gguf_resident_bytes(&g, Some(QKind::Int8))),
            est_f32_mb: mb(gguf_resident_bytes(&g, None)),
            avail_mb: available_memory_bytes().map(mb),
        })
    })
    .await
    .map_err(|e| format!("task error: {e}"))?
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GgufRunParams {
    pub model_path: String,
    pub prompt: String,
    pub quant: String,
    pub max_new: usize,
    pub temp: f32,
    pub gen_seed: Option<u64>,
    /// Raw space-separated token IDs (used when the file has no tokenizer).
    pub ids: Option<String>,
    /// Load even if the memory estimate exceeds available RAM.
    pub force: bool,
    /// Optional fine-tune checkpoint (`.flck`) to overlay on the base model.
    /// When set, the model is loaded f32 (a checkpoint holds f32 weights).
    pub resume: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GgufRunResult {
    pub text: String,
    pub prompt_tokens: usize,
    pub generated: usize,
    pub seconds: f32,
    pub tokens_per_sec: f32,
}

/// Import a llama/qwen2 GGUF and generate from it (mirrors `slm_cli run-gguf`).
#[tauri::command]
pub async fn run_gguf(params: GgufRunParams) -> Result<GgufRunResult, String> {
    tauri::async_runtime::spawn_blocking(move || gguf_run_inner(params))
        .await
        .map_err(|e| format!("task error: {e}"))?
}

fn gguf_run_inner(p: GgufRunParams) -> Result<GgufRunResult, String> {
    if !(p.temp.is_finite() && p.temp > 0.0) {
        return Err("temperature must be a positive number".into());
    }
    // A fine-tune checkpoint holds f32 weights, so applying one forces f32.
    let resume = p.resume.as_ref().filter(|s| !s.trim().is_empty()).cloned();
    let prec = if resume.is_some() {
        None
    } else {
        parse_quant(&p.quant)?
    };
    let g = Gguf::open(&p.model_path).map_err(|e| format!("cannot open {}: {e}", p.model_path))?;

    // Memory guard before the (potentially large) load.
    let est = gguf_resident_bytes(&g, prec);
    if let Some(avail) = available_memory_bytes() {
        if (est as f64) > 0.9 * avail as f64 && !p.force {
            return Err(format!(
                "estimated resident memory ({:.2} GB) exceeds 90% of available ({:.2} GB) — \
                 choose a smaller quant or enable 'load anyway'.",
                est as f64 / 1e9,
                avail as f64 / 1e9
            ));
        }
    }

    let tok = GgufTokenizer::from_gguf(&g).ok();
    let mut model = g
        .load_llama_prec(prec)
        .map_err(|e| format!("cannot load model: {e}"))?;

    // Overlay fine-tuned weights from a checkpoint, if requested.
    if let Some(ckpt) = &resume {
        let bytes =
            std::fs::read(ckpt).map_err(|e| format!("cannot read checkpoint {ckpt}: {e}"))?;
        let mut tr = LlamaTrainer::new(model).map_err(|e| format!("cannot wrap model: {e}"))?;
        tr.load_checkpoint_into(&bytes)
            .map_err(|e| format!("cannot apply checkpoint {ckpt}: {e}"))?;
        model = tr.model;
    }

    let prompt_ids: Vec<usize> = if let Some(ids) = p.ids.as_ref().filter(|s| !s.trim().is_empty())
    {
        ids.split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect()
    } else if let Some(t) = &tok {
        let mut v = Vec::new();
        if let Some(bos) = t.bos() {
            v.push(bos);
        }
        v.extend(t.encode(&p.prompt));
        v
    } else {
        return Err("this GGUF has no tokenizer; enter space-separated token IDs instead".into());
    };
    if prompt_ids.is_empty() {
        return Err("empty prompt — enter text (with a tokenizer) or token IDs".into());
    }

    let mut rng = Rng::new(p.gen_seed.unwrap_or_else(time_seed));
    let params = SamplingParams::with_temperature(p.temp);
    let eos = tok.as_ref().and_then(GgufTokenizer::eos);
    let t0 = std::time::Instant::now();
    let out = model
        .generate(&prompt_ids, p.max_new, &params, eos, &mut rng)
        .map_err(|e| format!("generation failed: {e}"))?;
    let seconds = t0.elapsed().as_secs_f32();

    let text = match &tok {
        Some(t) => format!("{}{}", p.prompt, t.decode(&out)),
        None => out
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(" "),
    };
    Ok(GgufRunResult {
        text,
        prompt_tokens: prompt_ids.len(),
        generated: out.len(),
        tokens_per_sec: out.len() as f32 / seconds.max(1e-6),
        seconds,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// 5b. Fine-tune an imported GGUF (mirrors `slm_cli finetune-gguf`)
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FinetuneParams {
    pub model_path: String,
    pub corpus_path: String,
    pub out_path: String,
    pub epochs: usize,
    pub lr: f32,
    pub batch_size: usize,
    pub seq_len: usize,
    pub warmup: u64,
    pub clip: f32,
    pub weight_decay: f32,
    pub dropout: f32,
    pub qat: bool,
    pub seed: u64,
    pub threads: usize,
    pub resume: Option<String>,
    pub force: bool,
    pub verbose: bool,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct FinetuneResult {
    pub out_path: String,
    pub epochs: usize,
    pub first_loss: f32,
    pub final_loss: f32,
    pub seconds: f64,
    pub tokens: usize,
    pub steps: u64,
    pub bytes: u64,
}

/// AdamW-fine-tune an imported llama/qwen2 GGUF on a text corpus and write a
/// `.flck` checkpoint that `run_gguf { resume }` can overlay on the base model.
/// Per-epoch progress streams as `finetune-progress` events.
#[tauri::command]
pub async fn finetune_gguf(
    app: AppHandle,
    params: FinetuneParams,
) -> Result<FinetuneResult, String> {
    tauri::async_runtime::spawn_blocking(move || finetune_inner(app, params))
        .await
        .map_err(|e| format!("task error: {e}"))?
}

fn finetune_inner(app: AppHandle, p: FinetuneParams) -> Result<FinetuneResult, String> {
    if p.out_path.trim().is_empty() {
        return Err("please provide an output checkpoint path (.flck)".into());
    }
    if p.epochs == 0 {
        return Err("epochs must be ≥ 1".into());
    }
    if !(p.lr.is_finite() && p.lr > 0.0) {
        return Err("learning rate must be a positive number".into());
    }
    ferrum_core::set_verbose(p.verbose);
    let cleanup = || ferrum_core::set_verbose(false);

    // 1. Open + f32-load the base model.
    let g = Gguf::open(&p.model_path).map_err(|e| {
        cleanup();
        format!("cannot open {}: {e}", p.model_path)
    })?;
    let est = gguf_resident_bytes(&g, None); // f32
    let train_est = est.saturating_mul(4); // weights + grad + Adam m/v
    if let Some(avail) = available_memory_bytes() {
        if (train_est as f64) > 0.9 * avail as f64 && !p.force {
            cleanup();
            return Err(format!(
                "estimated training memory ({:.2} GB) exceeds 90% of available ({:.2} GB) — \
                 fine-tune a smaller model or enable 'train anyway'.",
                train_est as f64 / 1e9,
                avail as f64 / 1e9
            ));
        }
    }
    let tok = GgufTokenizer::from_gguf(&g).map_err(|_| {
        cleanup();
        "this GGUF has no tokenizer; text fine-tuning needs one".to_string()
    })?;
    let model = g.load_llama_prec(None).map_err(|e| {
        cleanup();
        format!("cannot load model: {e}")
    })?;

    // 2. Read + tokenize the corpus.
    let text = std::fs::read_to_string(&p.corpus_path).map_err(|e| {
        cleanup();
        format!("cannot read corpus {}: {e}", p.corpus_path)
    })?;
    if text.trim().is_empty() {
        cleanup();
        return Err(format!("corpus {} is empty", p.corpus_path));
    }
    let mut tokens: Vec<usize> = Vec::new();
    if let Some(bos) = tok.bos() {
        tokens.push(bos);
    }
    tokens.extend(tok.encode(&text));

    let seq = p.seq_len.min(model.cfg.context_len).max(2);
    let batch = p.batch_size.max(1);
    if tokens.len() < seq {
        cleanup();
        return Err(format!(
            "corpus tokenized to {} tokens but sequence length is {seq}; supply more text",
            tokens.len()
        ));
    }

    // 3. Build + configure the trainer.
    let mut tr = LlamaTrainer::new(model).map_err(|e| {
        cleanup();
        format!("cannot build trainer: {e}")
    })?;
    tr.set_optimizer(Adam::new(p.lr));
    tr.set_weight_decay(p.weight_decay.max(0.0));
    tr.set_dropout(p.dropout);
    tr.set_grad_clip(if p.clip > 0.0 { Some(p.clip) } else { None });
    tr.set_qat(p.qat);

    let num_windows = tokens.len() - seq + 1;
    let steps_per_epoch = num_windows.div_ceil(batch) as u64;

    // 4. Resume optimizer state if requested.
    let mut rng = if let Some(ckpt) = p.resume.as_ref().filter(|s| !s.trim().is_empty()) {
        let bytes = std::fs::read(ckpt).map_err(|e| {
            cleanup();
            format!("cannot read checkpoint {ckpt}: {e}")
        })?;
        tr.load_checkpoint_into(&bytes).map_err(|e| {
            cleanup();
            format!("cannot resume from {ckpt}: {e}")
        })?
    } else {
        Rng::new(p.seed)
    };

    // The schedule spans the *cumulative* step timeline (after any resume), so a
    // resumed run does not start past total_steps — which would pin the LR at 0.
    if p.warmup > 0 {
        let total = tr.step_count() + steps_per_epoch * p.epochs as u64;
        tr.set_lr_schedule(Some(LrSchedule::warmup_cosine(
            p.lr,
            p.warmup,
            total.max(p.warmup + 1),
        )));
    }

    let threads = if p.threads == 0 {
        ferrum_core::num_threads()
    } else {
        p.threads
    };

    // 5. Train, streaming per-epoch progress.
    let t0 = std::time::Instant::now();
    let mut first_loss = f32::NAN;
    let mut final_loss = f32::NAN;
    for e in 0..p.epochs {
        let loss = match tr.finetune_epoch_threaded(&tokens, seq, batch, &mut rng, threads) {
            Ok(l) => l,
            Err(err) => {
                cleanup();
                return Err(format!("fine-tuning failed: {err}"));
            }
        };
        if e == 0 {
            first_loss = loss;
        }
        final_loss = loss;
        let _ = app.emit(
            "finetune-progress",
            serde_json::json!({
                "epoch": e + 1, "total": p.epochs, "loss": loss, "ppl": loss.exp(),
            }),
        );
    }
    let seconds = t0.elapsed().as_secs_f64();

    // 6. Save the checkpoint.
    let bytes = tr.save_checkpoint(&rng);
    std::fs::write(&p.out_path, &bytes).map_err(|e| {
        cleanup();
        format!("cannot write {}: {e}", p.out_path)
    })?;
    cleanup();

    let result = FinetuneResult {
        out_path: p.out_path.clone(),
        epochs: p.epochs,
        first_loss,
        final_loss,
        seconds,
        tokens: tokens.len(),
        steps: tr.step_count(),
        bytes: bytes.len() as u64,
    };
    let _ = app.emit("finetune-done", result.clone());
    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
// GGUF export (Export tab): write a loaded — optionally fine-tuned — llama/
// qwen2 model back out as a runnable GGUF at a chosen quantization.
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExportParams {
    pub model_path: String,
    pub out_path: String,
    /// Output type name, parsed by `GgufQuant::from_str` ("q8_0", "q4_k", "f16", …).
    pub quant: String,
    /// Optional `.flck` fine-tune checkpoint to overlay before export.
    pub resume: Option<String>,
    /// Bypass the RAM guard.
    pub force: bool,
    pub verbose: bool,
}

#[derive(Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct ExportResult {
    pub out_path: String,
    pub bytes: u64,
    pub seconds: f64,
    /// Per-type tensor counts of the written file, e.g. "42× Q6_K, 19× F32".
    pub tensor_summary: String,
}

/// Output-buffer size estimate for the RAM guard: the whole encoded file is
/// built in memory before the atomic write, so a lossless f32 target costs a
/// second model's worth; f16 half; quantized targets ≤ ~a third (conservative).
fn export_output_estimate(est_model: usize, quant: ferrum_core::GgufQuant) -> usize {
    use ferrum_core::GgufQuant as Q;
    match quant {
        Q::F32 => est_model,
        Q::F16 => est_model / 2,
        _ => est_model / 3,
    }
}

/// Human name for the GGML tensor-type ids the exporter can emit.
fn ggml_type_name(t: u32) -> String {
    match t {
        0 => "F32".into(),
        1 => "F16".into(),
        2 => "Q4_0".into(),
        3 => "Q4_1".into(),
        8 => "Q8_0".into(),
        9 => "Q8_1".into(),
        12 => "Q4_K".into(),
        13 => "Q5_K".into(),
        14 => "Q6_K".into(),
        other => format!("type{other}"),
    }
}

/// The testable core of the Export tab: everything `export_gguf` does except
/// event emission. `progress` receives one human-readable line per phase
/// (prefixes: "opening", "loading", "applying", "encoding").
pub(crate) fn do_export(p: &ExportParams, progress: &dyn Fn(&str)) -> Result<ExportResult, String> {
    if p.out_path.trim().is_empty() {
        return Err("please provide an output path (.gguf)".into());
    }
    let quant = ferrum_core::GgufQuant::from_str(&p.quant)
        .ok_or_else(|| format!("unknown output type '{}'", p.quant))?;

    progress(&format!("opening {} (streamed)…", p.model_path));
    let g = Gguf::open(&p.model_path).map_err(|e| format!("cannot open {}: {e}", p.model_path))?;

    // Memory guard: f32 model + in-RAM output buffer.
    let est_model = gguf_resident_bytes(&g, None);
    let est = est_model.saturating_add(export_output_estimate(est_model, quant));
    if let Some(avail) = available_memory_bytes() {
        if (est as f64) > 0.9 * avail as f64 && !p.force {
            return Err(format!(
                "estimated peak memory ({:.2} GB: f32 model + output buffer) exceeds 90% of \
                 available ({:.2} GB) — pick a smaller output type or tick 'Export anyway'.",
                est as f64 / 1e9,
                avail as f64 / 1e9
            ));
        }
    }

    let t0 = std::time::Instant::now();
    progress("loading weights (f32)…");
    let mut model = g
        .load_llama_prec(None)
        .map_err(|e| format!("cannot load model: {e}"))?;

    if let Some(ckpt) = &p.resume {
        progress(&format!("applying {ckpt}…"));
        let bytes =
            std::fs::read(ckpt).map_err(|e| format!("cannot read checkpoint {ckpt}: {e}"))?;
        let mut tr = LlamaTrainer::new(model).map_err(|e| format!("cannot wrap model: {e}"))?;
        tr.load_checkpoint_into(&bytes)
            .map_err(|e| format!("cannot apply checkpoint: {e}"))?;
        model = tr.model;
    }

    progress(&format!("encoding + writing ({})…", p.quant));
    ferrum_core::write_llama_gguf(&model, &g, quant, &p.out_path)
        .map_err(|e| format!("export failed: {e}"))?;

    // Per-type summary, read back from the written file (also proves it
    // re-opens as valid GGUF).
    let gout =
        Gguf::open(&p.out_path).map_err(|e| format!("written file failed to re-open: {e}"))?;
    let mut counts: std::collections::BTreeMap<u32, usize> = std::collections::BTreeMap::new();
    for t in &gout.tensors {
        *counts.entry(t.ggml_type).or_default() += 1;
    }
    let tensor_summary = counts
        .iter()
        .map(|(ty, n)| format!("{n}× {}", ggml_type_name(*ty)))
        .collect::<Vec<_>>()
        .join(", ");

    let bytes = std::fs::metadata(&p.out_path).map(|m| m.len()).unwrap_or(0);
    Ok(ExportResult {
        out_path: p.out_path.clone(),
        bytes,
        seconds: t0.elapsed().as_secs_f64(),
        tensor_summary,
    })
}

/// Export a llama/qwen2 GGUF (optionally with a fine-tune checkpoint applied)
/// to a new GGUF at the requested quantization. Phase progress streams as
/// `export-progress` events (see [`do_export`] for the phases).
#[tauri::command]
pub async fn export_gguf(app: AppHandle, params: ExportParams) -> Result<ExportResult, String> {
    tauri::async_runtime::spawn_blocking(move || {
        ferrum_core::set_verbose(params.verbose);
        let emit = |m: &str| {
            let _ = app.emit("export-progress", m.to_string());
        };
        let r = do_export(&params, &emit);
        ferrum_core::set_verbose(false);
        r
    })
    .await
    .map_err(|e| format!("task error: {e}"))?
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. Interactive terminal (requirement #3)
// ─────────────────────────────────────────────────────────────────────────────

/// Run one shell command in the studio's working directory, streaming stdout and
/// stderr as `term-output` events. `cd` is handled internally so it persists
/// between calls (a minimal interactive shell). Returns the exit code.
#[tauri::command]
pub async fn run_terminal(app: AppHandle, command: String) -> Result<i32, String> {
    if is_sandboxed() {
        return Err("the interactive shell is not available on this platform".into());
    }
    tauri::async_runtime::spawn_blocking(move || term_inner(app, command))
        .await
        .map_err(|e| format!("task error: {e}"))?
}

fn term_inner(app: AppHandle, command: String) -> Result<i32, String> {
    let state = app.state::<AppState>();
    let cmd = command.trim().to_string();
    if cmd.is_empty() {
        return Ok(0);
    }

    // Built-in `cd` so the working directory persists across commands.
    if cmd == "cd" || cmd.starts_with("cd ") {
        let target = cmd[2..].trim();
        let mut cwd = state.cwd.lock().unwrap();
        let candidate = if target.is_empty() {
            std::env::var("HOME")
                .or_else(|_| std::env::var("USERPROFILE"))
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|_| cwd.clone())
        } else {
            let p = Path::new(target);
            if p.is_absolute() {
                p.to_path_buf()
            } else {
                cwd.join(p)
            }
        };
        match std::fs::canonicalize(&candidate) {
            Ok(c) if c.is_dir() => {
                *cwd = c;
                Ok(0)
            }
            _ => {
                let _ = app.emit(
                    "term-output",
                    serde_json::json!({ "line": format!("cd: no such directory: {target}"), "stream": "stderr" }),
                );
                Ok(1)
            }
        }
    } else {
        let cwd = state.cwd.lock().unwrap().clone();
        run_shell(&app, &cmd, &cwd)
    }
}

fn run_shell(app: &AppHandle, cmd: &str, cwd: &Path) -> Result<i32, String> {
    use std::process::{Command, Stdio};

    let mut command = if cfg!(target_os = "windows") {
        let mut c = Command::new("cmd");
        c.arg("/C").arg(cmd);
        c
    } else {
        let mut c = Command::new("sh");
        c.arg("-c").arg(cmd);
        c
    };

    let mut child = command
        .current_dir(cwd)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("failed to start shell: {e}"))?;

    let stdout = child.stdout.take();
    let stderr = child.stderr.take();

    let a_out = app.clone();
    let h_out = std::thread::spawn(move || {
        if let Some(out) = stdout {
            for line in BufReader::new(out).lines().map_while(Result::ok) {
                let _ = a_out.emit(
                    "term-output",
                    serde_json::json!({ "line": line, "stream": "stdout" }),
                );
            }
        }
    });
    let a_err = app.clone();
    let h_err = std::thread::spawn(move || {
        if let Some(err) = stderr {
            for line in BufReader::new(err).lines().map_while(Result::ok) {
                let _ = a_err.emit(
                    "term-output",
                    serde_json::json!({ "line": line, "stream": "stderr" }),
                );
            }
        }
    });

    let _ = h_out.join();
    let _ = h_err.join();
    let status = child.wait().map_err(|e| format!("wait failed: {e}"))?;
    Ok(status.code().unwrap_or(-1))
}

/// Current working directory of the embedded shell (for the prompt).
#[tauri::command]
pub fn term_cwd(state: State<'_, AppState>) -> String {
    state.cwd.lock().unwrap().display().to_string()
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. System monitor (requirement #7)
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SysStats {
    pub available: bool,
    pub cpu_total: f32,
    pub per_core: Vec<f32>,
    pub cores: usize,
    pub mem_used: u64,
    pub mem_total: u64,
    pub mem_percent: f32,
    pub ferrum_threads: usize,
}

/// Sample current CPU and memory load. Polled by the frontend on an interval.
#[tauri::command]
pub fn system_stats(state: State<'_, AppState>) -> Result<SysStats, String> {
    let mut sys = state.sys.lock().unwrap();
    sys.refresh_cpu_usage();
    sys.refresh_memory();
    let per_core: Vec<f32> = sys.cpus().iter().map(|c| c.cpu_usage()).collect();
    let mem_total = sys.total_memory();
    let mem_used = sys.used_memory();
    let mem_percent = if mem_total > 0 {
        mem_used as f32 / mem_total as f32 * 100.0
    } else {
        0.0
    };
    Ok(SysStats {
        available: true,
        cpu_total: sys.global_cpu_usage(),
        cores: per_core.len(),
        per_core,
        mem_used,
        mem_total,
        mem_percent,
        ferrum_threads: ferrum_core::num_threads(),
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests (X1): pure command logic, no GUI runtime required.
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    // ── G2: download timeouts + URL validation ────────────────────────────────

    #[test]
    fn http_agent_builds() {
        // Building the timeout-bounded agent must not panic.
        let _ = http_agent();
    }

    #[test]
    fn fetch_text_rejects_non_http_urls() {
        // Validation happens before any network access, so these never hit the
        // wire — they fail fast with a clear message.
        for url in [
            "ftp://example.com/x",
            "file:///etc/passwd",
            "javascript:1",
            "",
        ] {
            let err = fetch_text(url, 1024).unwrap_err();
            assert!(
                err.contains("http://"),
                "unexpected error for {url:?}: {err}"
            );
        }
    }

    // ── file_name helper ──────────────────────────────────────────────────────

    #[test]
    fn file_name_extracts_basename() {
        assert_eq!(file_name("/a/b/model.bin"), "model.bin");
        assert_eq!(file_name("model.bin"), "model.bin");
    }

    #[test]
    fn available_memory_is_reported_cross_platform() {
        // The RAM guard is only meaningful if this returns a real figure; the old
        // /proc/meminfo reader returned None off Linux, disabling the guard there.
        let avail = available_memory_bytes().expect("available memory should be reported");
        assert!(
            avail > 16 * 1024 * 1024,
            "implausibly small avail: {avail} bytes"
        );
    }

    #[test]
    fn time_seed_is_nonzero() {
        assert_ne!(time_seed(), 0);
    }

    // ── GGUF export (do_export) ──────────────────────────────────────────────

    use ferrum_core::GgufQuant;

    #[test]
    fn export_output_estimate_matches_cli_heuristic() {
        assert_eq!(export_output_estimate(900, GgufQuant::F32), 900);
        assert_eq!(export_output_estimate(900, GgufQuant::F16), 450);
        assert_eq!(export_output_estimate(900, GgufQuant::Q8_0), 300);
        assert_eq!(export_output_estimate(900, GgufQuant::Q4K), 300);
    }

    #[test]
    fn do_export_rejects_bad_inputs() {
        let base = ExportParams {
            model_path: "/nonexistent.gguf".into(),
            out_path: "/tmp/out.gguf".into(),
            quant: "q8_0".into(),
            resume: None,
            force: false,
            verbose: false,
        };
        let noop = |_: &str| {};

        let mut p = base;
        p.out_path = "  ".into();
        assert!(do_export(&p, &noop).unwrap_err().contains("output"));

        p.out_path = "/tmp/out.gguf".into();
        p.quant = "q9_9".into();
        assert!(do_export(&p, &noop).unwrap_err().contains("q9_9"));
    }

    // Build a minimal valid `llama` GGUF (dim 8, 2 layers, gpt2 tokenizer
    // metadata) with deterministic weights — same shape as ferrum_core's own
    // export-test fixture. Returns the file's bytes.
    fn tiny_llama_gguf() -> Vec<u8> {
        use ferrum_core::{GgufBuilder, MetaValue};
        const GGML_F32: u32 = 0;
        let (dim, n_layers, n_heads, ffn, vocab) = (8usize, 2usize, 2usize, 16usize, 32usize);
        let head_dim = dim / n_heads;
        let gen = |seed: usize, n: usize| -> Vec<f32> {
            (0..n)
                .map(|i| (((seed * 131 + i * 17) % 101) as f32 / 500.0) - 0.1)
                .collect()
        };
        let f32_bytes = |xs: &[f32]| -> Vec<u8> {
            let mut o = Vec::with_capacity(xs.len() * 4);
            for &x in xs {
                o.extend_from_slice(&x.to_le_bytes());
            }
            o
        };
        let mut b = GgufBuilder::new();
        b.meta("general.architecture", MetaValue::String("llama".into()));
        b.meta("llama.embedding_length", MetaValue::U32(dim as u32));
        b.meta("llama.block_count", MetaValue::U32(n_layers as u32));
        b.meta("llama.attention.head_count", MetaValue::U32(n_heads as u32));
        b.meta(
            "llama.attention.head_count_kv",
            MetaValue::U32(n_heads as u32),
        );
        b.meta("llama.feed_forward_length", MetaValue::U32(ffn as u32));
        b.meta("llama.context_length", MetaValue::U32(16));
        b.meta(
            "llama.attention.layer_norm_rms_epsilon",
            MetaValue::F32(1e-5),
        );
        b.meta("tokenizer.ggml.model", MetaValue::String("gpt2".into()));
        b.meta(
            "tokenizer.ggml.tokens",
            MetaValue::Array(
                (0..vocab)
                    .map(|i| MetaValue::String(format!("t{i}")))
                    .collect(),
            ),
        );
        let t = |b: &mut GgufBuilder, name: &str, dims: &[u64], seed: usize| {
            let n: usize = dims.iter().product::<u64>() as usize;
            b.tensor(name, dims, GGML_F32, f32_bytes(&gen(seed, n)));
        };
        t(&mut b, "token_embd.weight", &[dim as u64, vocab as u64], 1);
        for i in 0..n_layers {
            let p = format!("blk.{i}");
            t(
                &mut b,
                &format!("{p}.attn_norm.weight"),
                &[dim as u64],
                10 + i,
            );
            let qkv = (n_heads * head_dim) as u64;
            t(
                &mut b,
                &format!("{p}.attn_q.weight"),
                &[dim as u64, qkv],
                20 + i,
            );
            t(
                &mut b,
                &format!("{p}.attn_k.weight"),
                &[dim as u64, qkv],
                30 + i,
            );
            t(
                &mut b,
                &format!("{p}.attn_v.weight"),
                &[dim as u64, qkv],
                40 + i,
            );
            t(
                &mut b,
                &format!("{p}.attn_output.weight"),
                &[qkv, dim as u64],
                50 + i,
            );
            t(
                &mut b,
                &format!("{p}.ffn_norm.weight"),
                &[dim as u64],
                60 + i,
            );
            t(
                &mut b,
                &format!("{p}.ffn_gate.weight"),
                &[dim as u64, ffn as u64],
                70 + i,
            );
            t(
                &mut b,
                &format!("{p}.ffn_up.weight"),
                &[dim as u64, ffn as u64],
                80 + i,
            );
            t(
                &mut b,
                &format!("{p}.ffn_down.weight"),
                &[ffn as u64, dim as u64],
                90 + i,
            );
        }
        t(&mut b, "output_norm.weight", &[dim as u64], 200);
        t(&mut b, "output.weight", &[dim as u64, vocab as u64], 201);
        b.into_bytes()
    }

    #[test]
    fn do_export_roundtrips_a_tiny_model_with_phases() {
        use ferrum_core::Gguf;
        let dir = std::env::temp_dir();
        let pid = std::process::id();
        let src = dir.join(format!("ferrum_gui_export_src_{pid}.gguf"));
        let dst = dir.join(format!("ferrum_gui_export_dst_{pid}.gguf"));
        std::fs::write(&src, tiny_llama_gguf()).unwrap();

        let p = ExportParams {
            model_path: src.to_string_lossy().into_owned(),
            out_path: dst.to_string_lossy().into_owned(),
            quant: "f32".into(),
            resume: None,
            force: false,
            verbose: false,
        };
        let phases = std::cell::RefCell::new(Vec::<String>::new());
        let r = do_export(&p, &|m| phases.borrow_mut().push(m.to_string())).unwrap();

        // Phases arrive in order: opening → loading → encoding (no resume).
        let ph = phases.borrow();
        assert!(ph.len() >= 3, "want ≥3 phases, got {ph:?}");
        assert!(ph[0].starts_with("opening"), "{ph:?}");
        assert!(ph[1].starts_with("loading"), "{ph:?}");
        assert!(ph.last().unwrap().starts_with("encoding"), "{ph:?}");
        assert!(!ph.iter().any(|m| m.starts_with("applying")), "{ph:?}");

        // Result: real file, non-trivial summary naming F32 tensors.
        assert!(r.bytes > 0);
        assert!(r.seconds >= 0.0);
        assert!(r.tensor_summary.contains("F32"), "{}", r.tensor_summary);

        // The output re-opens and re-loads as a runnable llama model.
        let g = Gguf::open(&p.out_path).unwrap();
        assert_eq!(g.architecture(), Some("llama"));
        g.load_llama_prec(None).unwrap();

        let _ = std::fs::remove_file(&src);
        let _ = std::fs::remove_file(&dst);
    }
}
