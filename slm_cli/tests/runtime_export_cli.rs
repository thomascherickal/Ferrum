//! `export-ollama` / `export-lmstudio` against the real runtimes.
//!
//! Tool-gated: a missing tool skips the test unless `FERRUM_REQUIRE_RUNTIMES=1`.
//! Run with:
//! `FERRUM_REQUIRE_RUNTIMES=1 cargo test -p slm_cli --release --test runtime_export_cli -- --nocapture`
//! (tests that touch a runtime share one lock, so default parallel threads are fine).
//!
//! ponytail: the model under test is llama.cpp's GPT-2 124M Q8_0 GGUF until Ferrum's
//! own GPT-2 GGUF writer (G1) merges; the 0.1M Ferrum-trained model fixture then replaces it.
//! Override the model file with `FERRUM_TEST_GGUF=/path/to/gpt2.gguf`.

mod common;

use common::{is_required, require_or_skip, require_or_skip_with, tool};
use ferrum_core::{GgufBuilder, MetaValue};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

const DEV: &str = "/home/thomas/Projects/ferrum-cpu/.ferrum-dev";
const PROMPT: &str = "Hello, my name is";
const N: usize = 8;

/// One lock for everything that touches Ollama, LM Studio or their ports (threads and processes).
fn serial() -> common::RuntimeLock {
    common::runtime_lock()
}

fn gpt2() -> PathBuf {
    let p = std::env::var_os("FERRUM_TEST_GGUF")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(DEV).join("gpt2gguf/gpt2-q8_0.gguf"));
    assert!(p.is_file(), "missing test model {}", p.display());
    p
}

fn exporter(args: &[&str], envs: &[(&str, &str)]) -> (bool, String) {
    let mut c = Command::new(env!("CARGO_BIN_EXE_train_transformer"));
    c.args(args);
    for (k, v) in envs {
        c.env(k, v);
    }
    let out = c.output().expect("spawn train_transformer");
    let text =
        String::from_utf8_lossy(&out.stderr).to_string() + &String::from_utf8_lossy(&out.stdout);
    (out.status.success(), text)
}

/// llama.cpp's greedy continuation of `PROMPT` (the reference both runtimes must match).
fn llama_reference(model: &Path) -> Option<String> {
    let bin = require_or_skip(tool(
        "FERRUM_LLAMA_COMPLETION",
        "llama-completion",
        ".local/opt/llama.cpp-b11521/llama-completion",
    ))?;
    let out = Command::new(bin)
        .args([
            "-m",
            model.to_str().unwrap(),
            "-p",
            PROMPT,
            "-n",
            &N.to_string(),
        ])
        .args(["--temp", "0", "-no-cnv", "--no-warmup"])
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .expect("run llama-completion");
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    let rest = text
        .strip_prefix(PROMPT)
        .unwrap_or_else(|| panic!("llama output {text:?}"));
    assert!(!rest.trim().is_empty(), "llama.cpp produced nothing");
    Some(rest.to_string())
}

fn json_string_field(json: &str, key: &str) -> String {
    let pat = format!("\"{key}\":");
    let at = json
        .find(&pat)
        .unwrap_or_else(|| panic!("no {key} in {json}"))
        + pat.len();
    let start = at + json[at..].find('"').unwrap() + 1; // tolerate pretty-printed JSON
    let (mut out, mut it) = (String::new(), json[start..].chars());
    while let Some(c) = it.next() {
        match c {
            '"' => break,
            '\\' => match it.next().unwrap() {
                'n' => out.push('\n'),
                't' => out.push('\t'),
                'u' => {
                    let hex: String = it.by_ref().take(4).collect();
                    out.push(char::from_u32(u32::from_str_radix(&hex, 16).unwrap()).unwrap());
                }
                other => out.push(other),
            },
            c => out.push(c),
        }
    }
    out
}

fn curl_post(url: &str, body: &str) -> String {
    let out = Command::new("curl")
        .args([
            "-s",
            "-X",
            "POST",
            "-H",
            "Content-Type: application/json",
            url,
            "-d",
            body,
        ])
        .output()
        .expect("curl");
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn answers(host: &str) -> bool {
    host.parse().is_ok_and(|a| {
        std::net::TcpStream::connect_timeout(&a, std::time::Duration::from_millis(300)).is_ok()
    })
}

/// The Ollama server the test talks to; kills it on drop if we started it.
struct Ollama {
    host: String,
    bin: PathBuf,
    child: Option<Child>,
}

impl Ollama {
    #[allow(clippy::zombie_processes)] // Drop kills and waits
    fn start(bin: PathBuf) -> Self {
        let wanted = std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "127.0.0.1:11434".into());
        if answers(&wanted) {
            return Self {
                host: wanted,
                bin,
                child: None,
            };
        }
        // 11545 belongs to another agent; this suite owns 11546.
        let host = ["127.0.0.1:11546"]
            .into_iter()
            .find(|h| !answers(h))
            .expect("no free port for ollama serve")
            .to_string();
        let models = Path::new(DEV).join("scratch/ollama-models");
        std::fs::create_dir_all(&models).unwrap();
        let child = Command::new(&bin)
            .arg("serve")
            .env("OLLAMA_HOST", &host)
            .env("OLLAMA_MODELS", &models)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("ollama serve");
        // Owning `me` before waiting means Drop kills the child if the port never answers.
        let me = Self {
            host,
            bin,
            child: Some(child),
        };
        for _ in 0..100 {
            if answers(&me.host) {
                return me;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        panic!("ollama serve did not come up on {}", me.host);
    }
    fn rm(&self, name: &str) {
        let _ = Command::new(&self.bin)
            .args(["rm", name])
            .env("OLLAMA_HOST", &self.host)
            .output();
    }
}

impl Drop for Ollama {
    fn drop(&mut self) {
        if let Some(c) = &mut self.child {
            let _ = c.kill();
            let _ = c.wait();
        }
    }
}

/// Removes the Ollama model even if the test panics.
struct OllamaModel<'a>(&'a Ollama, &'static str);
impl Drop for OllamaModel<'_> {
    fn drop(&mut self) {
        self.0.rm(self.1);
    }
}

fn ollama_tool() -> Option<PathBuf> {
    require_or_skip(tool("FERRUM_OLLAMA", "ollama", ".local/bin/ollama"))
}
fn lms_tool() -> Option<PathBuf> {
    require_or_skip(tool("FERRUM_LMS", "lms", ".lmstudio/bin/lms"))
}

/// A path with a space in it, pointing at the model.
/// Removes its directory on drop.
struct TmpDir(PathBuf);
impl Drop for TmpDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn spaced_link(tag: &str) -> (TmpDir, PathBuf) {
    let dir = std::env::temp_dir().join(format!("ferrum e1 {tag} {}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir); // a stale dir from a crashed run would make symlink fail with EEXIST
    std::fs::create_dir_all(&dir).unwrap();
    let guard = TmpDir(dir.clone());
    let link = dir.join("gpt2 q8.gguf");
    std::os::unix::fs::symlink(gpt2(), &link).unwrap();
    (guard, link)
}

#[test]
fn spaced_link_survives_a_stale_dir_and_cleans_up() {
    let (first, link1) = spaced_link("stale");
    let dir = first.0.clone();
    std::mem::forget(first); // simulate a crashed run that never cleaned up
    assert!(link1.exists());
    let (second, link2) = spaced_link("stale"); // same tag + pid -> same dir; must not EEXIST
    assert_eq!(link1, link2);
    assert!(link2.exists());
    drop(second);
    assert!(!dir.exists(), "guard must remove the dir on drop");
}

#[test]
fn export_ollama_then_generate() {
    let _serial = serial();
    let Some(bin) = ollama_tool() else { return };
    let Some(want) = llama_reference(&gpt2()) else {
        return;
    };
    let srv = Ollama::start(bin);
    let (_tmp, model) = spaced_link("gen");
    let _guard = OllamaModel(&srv, "ferrum-e1-gen");
    let (ok, text) = exporter(
        &["export-ollama", model.to_str().unwrap(), "ferrum-e1-gen"],
        &[("OLLAMA_HOST", &srv.host)],
    );
    assert!(ok, "{text}");
    let body = format!(
        r#"{{"model":"ferrum-e1-gen","prompt":"{PROMPT}","raw":true,"stream":false,"options":{{"temperature":0,"num_predict":{N}}}}}"#
    );
    let got = json_string_field(
        &curl_post(&format!("http://{}/api/generate", srv.host), &body),
        "response",
    );
    eprintln!("llama.cpp: {want:?}\nollama:    {got:?}");
    assert!(!got.trim().is_empty());
    assert_eq!(got.trim_end(), want.trim_end()); // llama-completion appends newlines
}

/// Cleans up only what this test did: the import under `~/.lmstudio/models/<repo>`,
/// the identifier it loaded, and the server it started.
struct LmStudioImport<'a> {
    lms: &'a Path,
    repo: &'a str,
    loaded: Option<&'a str>,
    server_started: bool,
}
impl<'a> LmStudioImport<'a> {
    fn new(lms: &'a Path, repo: &'a str) -> Self {
        Self {
            lms,
            repo,
            loaded: None,
            server_started: false,
        }
    }
}
impl Drop for LmStudioImport<'_> {
    fn drop(&mut self) {
        if let Some(id) = self.loaded {
            let _ = Command::new(self.lms).args(["unload", id]).output();
        }
        if self.server_started {
            let _ = Command::new(self.lms).args(["server", "stop"]).output();
        }
        let home = std::env::var("HOME").unwrap();
        let models = Path::new(&home).join(".lmstudio/models");
        let _ = std::fs::remove_dir_all(models.join(self.repo));
        if let Some((publisher, _)) = self.repo.split_once('/') {
            let _ = std::fs::remove_dir(models.join(publisher)); // only if now empty
        }
    }
}

#[test]
fn export_lmstudio_then_generate() {
    let _serial = serial();
    let Some(lms) = lms_tool() else { return };
    let Some(want) = llama_reference(&gpt2()) else {
        return;
    };
    let model = gpt2();
    let (ok, text) = exporter(
        &[
            "export-lmstudio",
            model.to_str().unwrap(),
            "--repo",
            "ferrum-test/e1gen",
        ],
        &[],
    );
    let id = "ferrum-e1-gen";
    let mut guard = LmStudioImport::new(&lms, "ferrum-test/e1gen");
    assert!(ok, "{text}");
    let key = text
        .split("modelKey '")
        .nth(1)
        .and_then(|r| r.split('\'').next())
        .expect(&text)
        .to_string();
    let ls = Command::new(&lms).args(["ls", "--json"]).output().unwrap();
    assert!(String::from_utf8_lossy(&ls.stdout).contains(&key));
    let load = Command::new(&lms)
        .args(["load", &key, "--gpu", "off", "--identifier", id, "-y"])
        .output()
        .unwrap();
    assert!(
        load.status.success(),
        "{}",
        String::from_utf8_lossy(&load.stderr)
    );
    guard.loaded = Some(id);
    assert!(
        !answers("127.0.0.1:1234"),
        "port 1234 is already in use by a server this test did not start; stop it and rerun"
    );
    let srv = Command::new(&lms)
        .args(["server", "start", "-p", "1234"])
        .output()
        .unwrap();
    assert!(
        srv.status.success(),
        "{}",
        String::from_utf8_lossy(&srv.stderr)
    );
    guard.server_started = true;
    let body = format!(
        r#"{{"model":"{id}","prompt":"{PROMPT}","max_tokens":{},"temperature":0,"repeat_penalty":1.0}}"#,
        N + 1
    );
    let got = json_string_field(
        &curl_post("http://127.0.0.1:1234/v1/completions", &body),
        "text",
    );
    eprintln!("llama.cpp: {want:?}\nlmstudio:  {got:?}");
    assert!(!got.trim().is_empty());
    assert_eq!(got.trim_end(), want.trim_end()); // llama-completion appends newlines
}

#[test]
fn export_is_idempotent() {
    let _serial = serial();
    if let Some(bin) = ollama_tool() {
        let srv = Ollama::start(bin);
        let _guard = OllamaModel(&srv, "ferrum-e1-idem");
        let model = gpt2();
        for round in 0..2 {
            let (ok, text) = exporter(
                &["export-ollama", model.to_str().unwrap(), "ferrum-e1-idem"],
                &[("OLLAMA_HOST", &srv.host)],
            );
            assert!(ok, "ollama round {round}: {text}");
        }
    }
    if let Some(lms) = lms_tool() {
        let _guard = LmStudioImport::new(&lms, "ferrum-test/e1idem");
        let model = gpt2();
        for round in 0..2 {
            let (ok, text) = exporter(
                &[
                    "export-lmstudio",
                    model.to_str().unwrap(),
                    "--repo",
                    "ferrum-test/e1idem",
                ],
                &[],
            );
            assert!(ok, "lmstudio round {round}: {text}");
        }
    }
}

#[test]
fn ollama_down_gives_actionable_error() {
    let _serial = serial();
    let Some(bin) = ollama_tool() else { return };
    let (ok, text) = exporter(
        &[
            "export-ollama",
            gpt2().to_str().unwrap(),
            "ferrum-e1-down",
            "--bin",
            bin.to_str().unwrap(),
        ],
        &[("OLLAMA_HOST", "127.0.0.1:1")],
    );
    assert!(!ok, "{text}");
    assert!(text.contains("ollama serve"), "{text}");
}

#[test]
fn cli_refuses_non_runnable_gguf_and_writes_modelfile() {
    let dir = std::env::temp_dir().join(format!("ferrum_e1_cli_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let _cleanup = TmpDir(dir.clone()); // removed even if an assertion panics
    let mut b = GgufBuilder::new();
    b.meta("general.architecture", MetaValue::String("ferrum".into()));
    let bad = dir.join("generic.gguf");
    std::fs::write(&bad, b.into_bytes()).unwrap();
    for cmd in [["export-ollama", "n"], ["export-lmstudio", "--"]] {
        let args: Vec<&str> = if cmd[0] == "export-ollama" {
            vec![cmd[0], bad.to_str().unwrap(), "n"]
        } else {
            vec![cmd[0], bad.to_str().unwrap()]
        };
        let (ok, text) = exporter(&args, &[]);
        assert!(!ok && text.contains("not runnable in llama.cpp"), "{text}");
    }
    let out = dir.join("m.Modelfile");
    let (ok, text) = exporter(
        &[
            "export-ollama",
            gpt2().to_str().unwrap(),
            "n",
            "--modelfile-only",
            out.to_str().unwrap(),
        ],
        &[],
    );
    assert!(ok, "{text}");
    let mf = std::fs::read_to_string(&out).unwrap();
    assert!(
        mf.contains("num_ctx 1024") && mf.contains("FROM \""),
        "{mf}"
    );
    let (ok, text) = exporter(&["export-ollama", "only-one-arg"], &[]);
    assert!(!ok && text.contains("usage:"), "{text}");
}

#[test]
fn require_runtimes_turns_skip_into_failure() {
    assert_eq!(require_or_skip_with(Ok::<_, String>(7), true), Some(7));
    assert_eq!(require_or_skip_with(Ok::<_, String>(7), false), Some(7));
    assert_eq!(
        require_or_skip_with(Err::<u8, _>("nope".into()), false),
        None
    );
    let r = std::panic::catch_unwind(|| require_or_skip_with(Err::<u8, _>("nope".into()), true));
    assert!(
        r.is_err(),
        "a missing tool must fail when FERRUM_REQUIRE_RUNTIMES=1"
    );
    // The env-var reader (pure, so no process-wide env mutation) and the finder.
    assert!(is_required(Some("1")));
    assert!(!is_required(Some("0")) && !is_required(Some("")) && !is_required(None));
    assert!(tool("FERRUM_NO_SUCH_ENV", "no-such-tool-xyz", "nowhere").is_err());
    assert!(tool("FERRUM_NO_SUCH_ENV", "sh", "nowhere").is_ok());
}
