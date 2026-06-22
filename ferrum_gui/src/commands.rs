//! Tauri commands: the bridge between the webview and `ferrum_core`.
//!
//! Conventions:
//! - Every command returns `Result<T, String>` so failures surface as a clear,
//!   human-readable message in the GUI (requirement: clear error messages).
//! - Heavy/blocking work runs inside `spawn_blocking` so the UI thread stays
//!   responsive and events can stream while the command runs.

use crate::AppState;
use ferrum_core::{
    clean_corpus, corpus_stats, validate_for_training, CleanOptions, GenerativeSLM, Rng, TaskType,
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
            std::fs::create_dir_all(parent).map_err(|e| format!("cannot create {parent:?}: {e}"))?;
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
            if p.num_heads == 0 || p.embed_dim % p.num_heads != 0 {
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
                &corpus, p.context_len, p.embed_dim, p.num_heads, p.num_blocks, p.hidden_dim,
                p.epochs, p.lr, p.batch_size, p.vocab_size, p.threads, &mut rng, progress,
            )
        }
        "embedded" => {
            if p.vocab_size != 0 && p.vocab_size < 256 {
                ferrum_core::set_verbose(false);
                return Err("vocab must be 0 (character-level) or ≥ 256 (byte-level BPE)".into());
            }
            GenerativeSLM::train_embedded_with_callback(
                &corpus, p.context_len, p.embed_dim, p.hidden_dim, p.epochs, p.lr, p.momentum,
                p.batch_size, p.vocab_size, &mut rng, progress,
            )
        }
        "onehot" => GenerativeSLM::train_with_callback(
            &corpus, p.context_len, p.hidden_dim, p.epochs, p.lr, p.momentum, p.batch_size,
            &mut rng, progress,
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

    let bytes = std::fs::metadata(&p.model_path).map(|m| m.len()).unwrap_or(0);
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
        let ev = slm.evaluate(&text).map_err(|e| format!("evaluation failed: {e}"))?;
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
        (format!("character-level ({} chars)", m.class_names.len()), 0)
    } else {
        let merges = m.tokenizer_state.split(';').filter(|s| !s.is_empty()).count();
        (format!("byte-level BPE ({} tokens)", m.output_dim), merges)
    };
    Ok(ModelInfo {
        path: path.clone(),
        bytes: raw.len() as u64,
        format: format!(
            "FINF v{version}{}",
            if version == 5 { " (int8-quantized)" } else { " (f32)" }
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
        for url in ["ftp://example.com/x", "file:///etc/passwd", "javascript:1", ""] {
            let err = fetch_text(url, 1024).unwrap_err();
            assert!(err.contains("http://"), "unexpected error for {url:?}: {err}");
        }
    }

    // ── file_name helper ──────────────────────────────────────────────────────

    #[test]
    fn file_name_extracts_basename() {
        assert_eq!(file_name("/a/b/model.bin"), "model.bin");
        assert_eq!(file_name("model.bin"), "model.bin");
    }

    #[test]
    fn time_seed_is_nonzero() {
        assert_ne!(time_seed(), 0);
    }
}
