//! `export-ollama` / `export-lmstudio`: put a Ferrum GGUF into a local runtime
//! with one command. Only `gpt2`-architecture GGUF files run in llama.cpp,
//! so anything else is refused before any tool is touched.

use ferrum_core::Gguf;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Command;

// ponytail: `arch`/`name` are part of the brief's interface; only tests read them today.
#[allow(dead_code)]
pub struct GgufSummary {
    pub arch: String,
    pub context_len: u32,
    pub name: String,
}

/// Errors unless `path` is a runnable (gpt2-architecture) GGUF file.
pub fn runnable_summary(path: &Path) -> Result<GgufSummary, String> {
    let p = path.to_string_lossy();
    let g = Gguf::open(&p).map_err(|e| format!("cannot read GGUF {p}: {e}"))?;
    let arch = g.architecture().unwrap_or("").to_string();
    if arch != "gpt2" {
        return Err(format!(
            "{p}: architecture '{}' is not runnable in llama.cpp (Ollama / LM Studio run only \
             'gpt2' GGUF files; generic, character-level and tabular Ferrum models have no \
             llama.cpp architecture)",
            if arch.is_empty() { "?" } else { &arch }
        ));
    }
    let context_len = g
        .meta("gpt2.context_length")
        .and_then(|v| v.as_usize())
        .ok_or_else(|| format!("{p}: missing gpt2.context_length"))? as u32;
    let name = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    Ok(GgufSummary {
        arch,
        context_len,
        name,
    })
}

/// The Modelfile text. The path is double-quoted (spaces are fine); callers
/// reject paths containing `"` or newlines first.
pub fn ollama_modelfile(abs_model: &Path, s: &GgufSummary) -> String {
    format!(
        "FROM \"{}\"\nTEMPLATE \"{{{{ .Prompt }}}}\"\nPARAMETER stop \"<|endoftext|>\"\n\
         PARAMETER temperature 0.8\nPARAMETER num_ctx {}\n",
        abs_model.display(),
        s.context_len
    )
}

/// Tool lookup order: `--bin` flag, env var, `PATH`, default install location.
pub fn find_tool(
    flag: Option<&str>,
    env: &str,
    name: &str,
    fallback: Option<PathBuf>,
) -> Result<PathBuf, String> {
    find_tool_in(
        flag,
        std::env::var(env).ok().as_deref(),
        env,
        std::env::var_os("PATH").as_deref(),
        name,
        fallback,
    )
}

fn find_tool_in(
    flag: Option<&str>,
    env_val: Option<&str>,
    env: &str,
    path_var: Option<&OsStr>,
    name: &str,
    fallback: Option<PathBuf>,
) -> Result<PathBuf, String> {
    let from_flag = flag.map(PathBuf::from);
    let from_env = env_val.map(PathBuf::from);
    let from_path = path_var.and_then(|p| {
        std::env::split_paths(p)
            .map(|d| d.join(name))
            .find(|c| c.is_file())
    });
    let candidates = [
        from_flag.clone(),
        from_env.clone(),
        from_path,
        fallback.clone(),
    ];
    if let Some(found) = candidates.iter().flatten().find(|c| c.is_file()) {
        return Ok(found.clone());
    }
    let show = |p: &Option<PathBuf>| {
        p.as_ref()
            .map_or("not set".into(), |p| p.display().to_string())
    };
    Err(format!(
        "cannot find '{name}'. Checked: (1) --bin: {}; (2) ${env}: {}; (3) PATH; (4) default \
         location: {}",
        show(&from_flag),
        show(&from_env),
        show(&fallback)
    ))
}

fn home_fallback(rel: &str) -> Option<PathBuf> {
    std::env::var_os("HOME").map(|h| Path::new(&h).join(rel))
}

/// Run a tool; returns (success, stdout+stderr).
fn run_tool(bin: &Path, args: &[&OsStr]) -> Result<(bool, String), String> {
    let out = Command::new(bin)
        .args(args)
        .output()
        .map_err(|e| format!("cannot run {}: {e}", bin.display()))?;
    let text =
        String::from_utf8_lossy(&out.stdout).into_owned() + &String::from_utf8_lossy(&out.stderr);
    Ok((out.status.success(), text))
}

fn absolute_checked(model: &Path) -> Result<PathBuf, String> {
    let abs = std::path::absolute(model).map_err(|e| format!("{}: {e}", model.display()))?;
    let s = abs.to_string_lossy();
    if s.contains('"') || s.contains('\n') {
        return Err(format!(
            "{s}: path contains a double quote or newline, which a Modelfile cannot quote; \
             rename the file"
        ));
    }
    Ok(abs)
}

/// Absolute path + summary + Modelfile text (shared by export and `--modelfile-only`).
pub fn modelfile_for(model: &Path) -> Result<String, String> {
    let abs = absolute_checked(model)?;
    let s = runnable_summary(&abs)?;
    Ok(ollama_modelfile(&abs, &s))
}

pub fn export_ollama(model: &Path, name: &str, bin: Option<&str>) -> Result<(), String> {
    let modelfile = modelfile_for(model)?;
    let tool = find_tool(
        bin,
        "FERRUM_OLLAMA",
        "ollama",
        home_fallback(".local/bin/ollama"),
    )?;
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("ferrum_modelfile_{}_{n}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mf = dir.join(format!("{}.Modelfile", name.replace('/', "_")));
    std::fs::write(&mf, modelfile).map_err(|e| format!("{}: {e}", mf.display()))?;
    let res = run_tool(
        &tool,
        &[
            "create".as_ref(),
            name.as_ref(),
            "-f".as_ref(),
            mf.as_os_str(),
        ],
    );
    let _ = std::fs::remove_dir_all(&dir);
    let (ok, text) = res?;
    if ok {
        return Ok(());
    }
    let hint = if text.contains("connect") || text.contains("refused") {
        "\nthe Ollama server is not running: start it with `ollama serve`"
    } else {
        ""
    };
    Err(format!(
        "`ollama create {name}` failed:\n{}{hint}",
        text.trim()
    ))
}

/// First `"key": "value"` string value in `obj` (whitespace after the colon is optional).
fn json_str<'a>(obj: &'a str, key: &str) -> Option<&'a str> {
    let pat = format!("\"{key}\":");
    let rest = obj[obj.find(&pat)? + pat.len()..].trim_start();
    let rest = rest.strip_prefix('"')?;
    Some(&rest[..rest.find('"')?])
}

/// `path` ends with `suffix` on a `/` boundary (`xferrum/m` does not match `ferrum/m`).
fn path_matches(path: &str, suffix: &str) -> bool {
    path.strip_suffix(suffix)
        .is_some_and(|head| head.is_empty() || head.ends_with('/'))
}

/// `modelKey` of the `lms ls --json` entry whose `path` ends with `suffix`.
// ponytail: string scan, not a JSON parser — assumes unescaped quotes in `path`/`modelKey`
// (true for LM Studio's output); use serde_json if slm_cli ever gains it.
fn model_key_for(json: &str, suffix: &str) -> Option<String> {
    let (mut depth, mut start, mut in_str, mut esc) = (0usize, 0usize, false, false);
    for (i, c) in json.char_indices() {
        if in_str {
            match (esc, c) {
                (true, _) => esc = false,
                (false, '\\') => esc = true,
                (false, '"') => in_str = false,
                _ => {}
            }
            continue;
        }
        match c {
            '"' => in_str = true,
            '{' => {
                if depth == 0 {
                    start = i;
                }
                depth += 1;
            }
            '}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    let obj = &json[start..=i];
                    if json_str(obj, "path").is_some_and(|p| path_matches(p, suffix)) {
                        return json_str(obj, "modelKey").map(String::from);
                    }
                }
            }
            _ => {}
        }
    }
    None
}

pub fn export_lmstudio(model: &Path, repo: &str, bin: Option<&str>) -> Result<String, String> {
    let (publisher, rest) = repo.split_once('/').unwrap_or(("", ""));
    if publisher.is_empty() || rest.is_empty() || rest.contains('/') {
        return Err(format!("--repo '{repo}' must look like <publisher>/<name>"));
    }
    let abs = std::path::absolute(model).map_err(|e| format!("{}: {e}", model.display()))?;
    runnable_summary(&abs)?;
    let tool = find_tool(bin, "FERRUM_LMS", "lms", home_fallback(".lmstudio/bin/lms"))?;
    let (ok, text) = run_tool(
        &tool,
        &[
            "import".as_ref(),
            "-y".as_ref(),
            "-l".as_ref(),
            "--user-repo".as_ref(),
            repo.as_ref(),
            abs.as_os_str(),
        ],
    )?;
    // Re-importing is not an error.
    if !ok && !text.to_lowercase().contains("already") {
        return Err(format!("`lms import` failed:\n{}", text.trim()));
    }
    let file = abs.file_name().unwrap_or_default().to_string_lossy();
    let suffix = format!("{repo}/{file}");
    let (ok, json) = run_tool(&tool, &["ls".as_ref(), "--json".as_ref()])?;
    if !ok {
        return Err(format!("`lms ls --json` failed:\n{}", json.trim()));
    }
    model_key_for(&json, &suffix).ok_or_else(|| {
        format!("imported, but `lms ls --json` has no entry whose path ends with {suffix}")
    })
}

const USAGE_OLLAMA: &str =
    "usage: train_transformer export-ollama <model.gguf> <name> [--bin P] [--modelfile-only out.Modelfile]";
const USAGE_LMSTUDIO: &str =
    "usage: train_transformer export-lmstudio <model.gguf> [--repo ferrum/<stem>] [--bin P]";

pub fn cmd_export_ollama(
    positional: &[String],
    bin: Option<&str>,
    modelfile_only: Option<&str>,
) -> Result<(), String> {
    let [model, name] = positional else {
        return Err(USAGE_OLLAMA.into());
    };
    if let Some(out) = modelfile_only {
        let text = modelfile_for(Path::new(model))?;
        std::fs::write(out, text).map_err(|e| format!("{out}: {e}"))?;
        println!("wrote {out}");
        return Ok(());
    }
    export_ollama(Path::new(model), name, bin)?;
    println!("exported to Ollama as '{name}'  (try: ollama run {name})");
    Ok(())
}

pub fn cmd_export_lmstudio(
    positional: &[String],
    repo: Option<&str>,
    bin: Option<&str>,
) -> Result<(), String> {
    let [model] = positional else {
        return Err(USAGE_LMSTUDIO.into());
    };
    let default_repo;
    let repo = match repo {
        Some(r) => r,
        None => {
            let stem = Path::new(model)
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy();
            default_repo = format!("ferrum/{stem}");
            &default_repo
        }
    };
    let key = export_lmstudio(Path::new(model), repo, bin)?;
    println!("imported into LM Studio as modelKey '{key}'  (try: lms load {key})");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_core::{GgufBuilder, MetaValue};
    use std::os::unix::fs::PermissionsExt;

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("ferrum_rx_{}_{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn gguf(dir: &Path, file: &str, arch: Option<&str>, ctx: Option<u32>) -> PathBuf {
        let mut b = GgufBuilder::new();
        if let Some(a) = arch {
            b.meta("general.architecture", MetaValue::String(a.into()));
        }
        if let Some(c) = ctx {
            b.meta("gpt2.context_length", MetaValue::U32(c));
        }
        let p = dir.join(file);
        std::fs::write(&p, b.into_bytes()).unwrap();
        p
    }

    /// An executable shell script named `name` in `dir`.
    fn fake(dir: &Path, name: &str, body: &str) -> PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, format!("#!/bin/sh\n{body}\n")).unwrap();
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
        p
    }

    fn s(p: &Path) -> &str {
        p.to_str().unwrap()
    }

    fn summary(ctx: u32) -> GgufSummary {
        GgufSummary {
            arch: "gpt2".into(),
            context_len: ctx,
            name: "m".into(),
        }
    }

    // ---- runnable_summary ----

    #[test]
    fn summary_accepts_gpt2() {
        let d = tmp("sum_ok");
        let p = gguf(&d, "tiny model.gguf", Some("gpt2"), Some(64));
        let g = runnable_summary(&p).unwrap();
        assert_eq!((g.arch.as_str(), g.context_len), ("gpt2", 64));
        assert_eq!(g.name, "tiny model");
    }

    #[test]
    fn refuses_non_runnable_gguf() {
        let d = tmp("sum_refuse");
        for arch in ["ferrum", "llama"] {
            let p = gguf(&d, "m.gguf", Some(arch), Some(64));
            let e = runnable_summary(&p).err().unwrap();
            assert!(e.contains("not runnable in llama.cpp"), "{e}");
            assert!(e.contains(&format!("'{arch}'")), "{e}");
        }
        // No architecture key at all.
        let e = runnable_summary(&gguf(&d, "n.gguf", None, None))
            .err()
            .unwrap();
        assert!(
            e.contains("not runnable in llama.cpp") && e.contains("'?'"),
            "{e}"
        );
    }

    #[test]
    fn summary_needs_context_length() {
        let d = tmp("sum_ctx");
        let e = runnable_summary(&gguf(&d, "m.gguf", Some("gpt2"), None))
            .err()
            .unwrap();
        assert!(e.contains("missing gpt2.context_length"), "{e}");
    }

    #[test]
    fn summary_rejects_unreadable_truncated_and_missing() {
        let d = tmp("sum_bad");
        let full = std::fs::read(gguf(&d, "ok.gguf", Some("gpt2"), Some(8))).unwrap();
        let cut = d.join("cut.gguf");
        std::fs::write(&cut, &full[..full.len() / 2]).unwrap();
        let junk = d.join("junk.gguf");
        std::fs::write(&junk, b"not a gguf at all").unwrap();
        for p in [cut, junk, d.join("missing.gguf")] {
            let e = runnable_summary(&p).err().unwrap();
            assert!(e.contains("cannot read GGUF"), "{e}");
        }
    }

    // ---- Modelfile ----

    #[test]
    fn modelfile_uses_absolute_quoted_path() {
        let m = ollama_modelfile(Path::new("/tmp/my model.gguf"), &summary(64));
        assert!(m.contains("FROM \"/tmp/my model.gguf\"\n"), "{m}");
        assert!(m.contains("TEMPLATE \"{{ .Prompt }}\"\n"), "{m}");
        assert!(m.contains("PARAMETER stop \"<|endoftext|>\"\n"), "{m}");
        assert!(m.contains("PARAMETER temperature 0.8\n"), "{m}");
        assert!(m.contains("PARAMETER num_ctx 64\n"), "{m}");
    }

    #[test]
    fn modelfile_for_makes_relative_paths_absolute_and_rejects_quotes() {
        let d = tmp("mf");
        let p = gguf(&d, "a b.gguf", Some("gpt2"), Some(32));
        let m = modelfile_for(&p).unwrap();
        assert!(
            m.contains(&format!("FROM \"{}\"", s(&p))) && m.contains("num_ctx 32"),
            "{m}"
        );
        // Relative input is made absolute against the cwd.
        let rel = Path::new("Cargo.toml");
        let e = modelfile_for(rel).err().unwrap();
        assert!(
            e.contains(std::env::current_dir().unwrap().to_str().unwrap()),
            "{e}"
        );
        for bad in ["q\"uote.gguf", "new\nline.gguf"] {
            let e = modelfile_for(&d.join(bad)).err().unwrap();
            assert!(e.contains("cannot quote"), "{e}");
        }
    }

    // ---- find_tool ----

    #[test]
    fn find_tool_order() {
        let d = tmp("find");
        let mk = |n: &str| fake(&d, n, "true");
        let (flag, env, path_tool, fb) = (mk("t_flag"), mk("t_env"), mk("mytool"), mk("t_fb"));
        let pv = std::ffi::OsString::from(&d);
        let go = |f: Option<&Path>,
                  e: Option<&Path>,
                  pv: Option<&std::ffi::OsStr>,
                  fb: Option<&Path>| {
            find_tool_in(
                f.map(s),
                e.map(s),
                "E",
                pv,
                "mytool",
                fb.map(Path::to_path_buf),
            )
        };
        let all = go(Some(&flag), Some(&env), Some(&pv), Some(&fb)).unwrap();
        assert_eq!(all, flag);
        assert_eq!(go(None, Some(&env), Some(&pv), Some(&fb)).unwrap(), env);
        assert_eq!(go(None, None, Some(&pv), Some(&fb)).unwrap(), path_tool);
        assert_eq!(go(None, None, None, Some(&fb)).unwrap(), fb);
        // A flag/env that points nowhere falls through to the next place.
        let gone = d.join("gone");
        assert_eq!(
            go(Some(&gone), Some(&gone), Some(&pv), None).unwrap(),
            path_tool
        );
    }

    #[test]
    fn find_tool_not_found_lists_all_four_places() {
        let d = tmp("find_none");
        let gone = d.join("gone");
        let e = find_tool_in(
            Some("/x/flagbin"),
            Some("/x/envbin"),
            "FERRUM_FOO",
            Some(d.as_os_str()),
            "mytool",
            Some(gone.clone()),
        )
        .unwrap_err();
        for want in [
            "'mytool'",
            "--bin: /x/flagbin",
            "$FERRUM_FOO: /x/envbin",
            "PATH",
            s(&gone),
        ] {
            assert!(e.contains(want), "{e}");
        }
        let e = find_tool_in(None, None, "FERRUM_FOO", None, "mytool", None).unwrap_err();
        assert!(
            e.contains("--bin: not set") && e.contains("$FERRUM_FOO: not set"),
            "{e}"
        );
        assert!(e.contains("default location: not set"), "{e}");
    }

    #[test]
    fn find_tool_reads_the_real_env_and_path() {
        let d = tmp("find_env");
        let t = fake(&d, "envtool", "true");
        std::env::set_var("FERRUM_RX_TEST_ENV", &t);
        assert_eq!(
            find_tool(None, "FERRUM_RX_TEST_ENV", "nope-xyz", None).unwrap(),
            t
        );
        std::env::remove_var("FERRUM_RX_TEST_ENV");
        let e = find_tool(None, "FERRUM_RX_TEST_ENV", "nope-xyz", None).unwrap_err();
        assert!(e.contains("nope-xyz"), "{e}");
        assert!(find_tool(None, "X", "sh", None).is_ok(), "sh is on PATH");
    }

    #[test]
    fn home_fallback_joins_home() {
        let h = std::env::var_os("HOME").map(PathBuf::from);
        assert_eq!(home_fallback("a/b"), h.map(|h| h.join("a/b")));
    }

    // ---- JSON scan ----

    #[test]
    fn model_key_scan() {
        let json = concat!(
            "Waking up LM Studio service...\n",
            r#"[{"type":"llm","modelKey":"a/one","path":"pub/repo-a/x.gguf","q":{"name":"Q8_0"},"v":["a{b"]},"#,
            r#"{"modelKey":"ferrum/m","note":"has \" quote and } brace","path":"ferrum/m/m.gguf"},"#,
            r#"{"path":"ferrum/nokey/z.gguf"}]"#
        );
        assert_eq!(
            model_key_for(json, "ferrum/m/m.gguf").as_deref(),
            Some("ferrum/m")
        );
        assert_eq!(
            model_key_for(json, "repo-a/x.gguf").as_deref(),
            Some("a/one")
        );
        assert_eq!(model_key_for(json, "ferrum/other/m.gguf"), None);
        assert_eq!(
            model_key_for(json, "ferrum/nokey/z.gguf"),
            None,
            "match without a modelKey"
        );
        // Pretty-printed JSON (space after the colon) and a non-boundary suffix.
        let pretty = "[{\n \"modelKey\": \"p/k\",\n \"path\": \"pub/repo/f.gguf\"\n}]";
        assert_eq!(model_key_for(pretty, "repo/f.gguf").as_deref(), Some("p/k"));
        assert_eq!(
            model_key_for(pretty, "po/f.gguf"),
            None,
            "not on a / boundary"
        );
        assert_eq!(
            model_key_for(pretty, "pub/repo/f.gguf").as_deref(),
            Some("p/k")
        );
        assert_eq!(model_key_for(r#"{"path": 5}"#, "x"), None);
        assert_eq!(model_key_for("", "x"), None);
        assert_eq!(model_key_for("[]", "x"), None);
        assert_eq!(model_key_for("} stray", "x"), None);
        assert_eq!(model_key_for(r#"{"path":"unterminated"#, "x"), None);
    }

    // ---- export_ollama ----

    #[test]
    fn export_ollama_runs_create_with_a_modelfile() {
        let d = tmp("ol_ok");
        let model = gguf(&d, "my model.gguf", Some("gpt2"), Some(64));
        let tool = fake(
            &d,
            "ollama",
            &format!("echo \"$@\" > {0}/args; cp \"$4\" {0}/seen", s(&d)),
        );
        export_ollama(&model, "ferrum-x", Some(s(&tool))).unwrap();
        let args = std::fs::read_to_string(d.join("args")).unwrap();
        assert!(
            args.starts_with("create ferrum-x -f ") && args.contains("ferrum-x.Modelfile"),
            "{args}"
        );
        let seen = std::fs::read_to_string(d.join("seen")).unwrap();
        assert!(seen.contains(&format!("FROM \"{}\"", s(&model))), "{seen}");
        assert!(seen.contains("num_ctx 64"), "{seen}");
    }

    #[test]
    fn export_ollama_down_says_ollama_serve() {
        let d = tmp("ol_down");
        let model = gguf(&d, "m.gguf", Some("gpt2"), Some(8));
        let tool = fake(&d, "ollama", "echo 'Error: connection refused' >&2; exit 1");
        let e = export_ollama(&model, "n", Some(s(&tool))).unwrap_err();
        assert!(
            e.contains("ollama serve") && e.contains("connection refused"),
            "{e}"
        );
    }

    #[test]
    fn export_ollama_other_failure_has_no_serve_hint() {
        let d = tmp("ol_fail");
        let model = gguf(&d, "m.gguf", Some("gpt2"), Some(8));
        let tool = fake(&d, "ollama", "echo 'bad modelfile' >&2; exit 1");
        let e = export_ollama(&model, "n/with/slash", Some(s(&tool))).unwrap_err();
        assert!(
            e.contains("bad modelfile") && !e.contains("ollama serve"),
            "{e}"
        );
    }

    #[test]
    fn export_ollama_refuses_before_running_any_tool() {
        let d = tmp("ol_refuse");
        let marker = d.join("ran");
        let tool = fake(&d, "ollama", &format!("touch {}", s(&marker)));
        let bad = gguf(&d, "f.gguf", Some("ferrum"), Some(8));
        let e = export_ollama(&bad, "n", Some(s(&tool))).unwrap_err();
        assert!(e.contains("not runnable in llama.cpp"), "{e}");
        assert!(!marker.exists());
    }

    #[test]
    fn export_ollama_spawn_error() {
        let d = tmp("ol_tool");
        let model = gguf(&d, "m.gguf", Some("gpt2"), Some(8));
        // Present but not executable: spawn fails with a clear message.
        let noexec = d.join("noexec");
        std::fs::write(&noexec, "x").unwrap();
        let e = export_ollama(&model, "n", Some(s(&noexec))).unwrap_err();
        assert!(e.contains("cannot run"), "{e}");
    }

    // ---- export_lmstudio ----

    const LS: &str = r#"[{"modelKey":"ferrum/m","path":"ferrum/m/m.gguf"}]"#;

    fn lms(dir: &Path, import: &str, ls: &str) -> PathBuf {
        std::fs::write(dir.join("ls.json"), LS).unwrap();
        fake(
            dir,
            "lms",
            &format!(
                "case \"$1\" in import) echo \"$@\" > {0}/args; {import};; ls) {ls};; esac",
                s(dir)
            ),
        )
    }

    #[test]
    fn export_lmstudio_imports_and_resolves_key() {
        let d = tmp("lm_ok");
        let model = gguf(&d, "m.gguf", Some("gpt2"), Some(8));
        let tool = lms(&d, "exit 0", &format!("cat {}/ls.json", s(&d)));
        let key = export_lmstudio(&model, "ferrum/m", Some(s(&tool))).unwrap();
        assert_eq!(key, "ferrum/m");
        let args = std::fs::read_to_string(d.join("args")).unwrap();
        assert!(
            args.starts_with("import -y -l --user-repo ferrum/m ") && args.contains("m.gguf"),
            "{args}"
        );
    }

    #[test]
    fn export_lmstudio_already_imported_is_success() {
        let d = tmp("lm_again");
        let model = gguf(&d, "m.gguf", Some("gpt2"), Some(8));
        let ls = format!("cat {}/ls.json", s(&d));
        for import in [
            "echo 'Model already exists' >&2; exit 1",
            "echo Already imported; exit 0",
        ] {
            let tool = lms(&d, import, &ls);
            assert_eq!(
                export_lmstudio(&model, "ferrum/m", Some(s(&tool))).unwrap(),
                "ferrum/m"
            );
        }
    }

    #[test]
    fn export_lmstudio_failures() {
        let d = tmp("lm_fail");
        let model = gguf(&d, "m.gguf", Some("gpt2"), Some(8));
        let ls = format!("cat {}/ls.json", s(&d));
        let tool = lms(&d, "echo disk full >&2; exit 1", &ls);
        let e = export_lmstudio(&model, "ferrum/m", Some(s(&tool))).unwrap_err();
        assert!(
            e.contains("`lms import` failed") && e.contains("disk full"),
            "{e}"
        );
        let tool = lms(&d, "exit 0", "echo daemon down >&2; exit 1");
        let e = export_lmstudio(&model, "ferrum/m", Some(s(&tool))).unwrap_err();
        assert!(
            e.contains("`lms ls --json` failed") && e.contains("daemon down"),
            "{e}"
        );
        let tool = lms(&d, "exit 0", "echo '[]'");
        let e = export_lmstudio(&model, "ferrum/m", Some(s(&tool))).unwrap_err();
        assert!(
            e.contains("no entry whose path ends with ferrum/m/m.gguf"),
            "{e}"
        );
    }

    #[test]
    fn export_lmstudio_spawn_error() {
        let d = tmp("lm_spawn");
        let model = gguf(&d, "m.gguf", Some("gpt2"), Some(8));
        let noexec = d.join("noexec");
        std::fs::write(&noexec, "x").unwrap();
        let e = export_lmstudio(&model, "ferrum/m", Some(s(&noexec))).unwrap_err();
        assert!(e.contains("cannot run"), "{e}");
    }

    #[test]
    fn export_lmstudio_validates_repo_and_model() {
        let d = tmp("lm_val");
        let model = gguf(&d, "m.gguf", Some("gpt2"), Some(8));
        let marker = d.join("ran");
        let tool = fake(&d, "lms", &format!("touch {}", s(&marker)));
        for repo in ["noslash", "/x", "x/", "a/b/c", ""] {
            let e = export_lmstudio(&model, repo, Some(s(&tool))).unwrap_err();
            assert!(e.contains("<publisher>/<name>"), "{repo}: {e}");
        }
        let bad = gguf(&d, "f.gguf", Some("ferrum"), Some(8));
        let e = export_lmstudio(&bad, "ferrum/f", Some(s(&tool))).unwrap_err();
        assert!(e.contains("not runnable in llama.cpp"), "{e}");
        assert!(!marker.exists());
    }

    // ---- CLI glue ----

    fn v(a: &[&str]) -> Vec<String> {
        a.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn cmd_export_ollama_paths() {
        let d = tmp("cmd_ol");
        let model = gguf(&d, "m.gguf", Some("gpt2"), Some(16));
        let tool = fake(&d, "ollama", "exit 0");
        assert!(cmd_export_ollama(&v(&["only-one"]), None, None)
            .unwrap_err()
            .contains("usage:"));
        cmd_export_ollama(&v(&[s(&model), "n"]), Some(s(&tool)), None).unwrap();
        // --modelfile-only writes the file and never needs the tool.
        let out = d.join("out.Modelfile");
        cmd_export_ollama(&v(&[s(&model), "n"]), Some("/nonexistent"), Some(s(&out))).unwrap();
        assert!(std::fs::read_to_string(&out)
            .unwrap()
            .contains("num_ctx 16"));
        let e =
            cmd_export_ollama(&v(&[s(&model), "n"]), None, Some("/nonexistent/dir/o")).unwrap_err();
        assert!(e.contains("/nonexistent/dir/o"), "{e}");
        let bad = gguf(&d, "f.gguf", Some("ferrum"), Some(8));
        let e = cmd_export_ollama(&v(&[s(&bad), "n"]), None, Some(s(&out))).unwrap_err();
        assert!(e.contains("not runnable"), "{e}");
    }

    #[test]
    fn cmd_export_lmstudio_paths() {
        let d = tmp("cmd_lm");
        let model = gguf(&d, "m.gguf", Some("gpt2"), Some(16));
        let ls = r#"echo '[{"modelKey":"K","path":"ferrum/m/m.gguf"}]'"#.to_string();
        let tool = lms(&d, "exit 0", &ls);
        assert!(cmd_export_lmstudio(&[], None, None)
            .unwrap_err()
            .contains("usage:"));
        // Default repo is ferrum/<stem>.
        cmd_export_lmstudio(&v(&[s(&model)]), None, Some(s(&tool))).unwrap();
        assert!(std::fs::read_to_string(d.join("args"))
            .unwrap()
            .contains("--user-repo ferrum/m "));
        // An explicit repo wins (and here matches no ls entry).
        let e =
            cmd_export_lmstudio(&v(&[s(&model)]), Some("me/other"), Some(s(&tool))).unwrap_err();
        assert!(e.contains("me/other/m.gguf"), "{e}");
    }
}
