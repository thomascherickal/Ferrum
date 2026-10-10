//! The CLI must reject what it does not understand instead of silently
//! falling back to defaults (review defects #3 and #4).

use std::process::Command;

fn run(args: &[&str]) -> (i32, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_train_transformer"))
        .args(args)
        .output()
        .expect("spawn train_transformer");
    let text =
        String::from_utf8_lossy(&out.stderr).to_string() + &String::from_utf8_lossy(&out.stdout);
    (out.status.code().unwrap_or(-1), text)
}

/// A corpus file private to the calling test (a shared path races: `fs::write` truncates it
/// while a parallel test is reading it, which showed up as "corpus is empty").
fn corpus() -> std::path::PathBuf {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "ferrum_cli_strict_{}_{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join("c.txt");
    std::fs::write(&p, "abc abc abc\n".repeat(50)).unwrap();
    p
}

#[test]
fn unknown_flag_is_an_error() {
    let c = corpus();
    let (code, text) = run(&[
        "train",
        c.to_str().unwrap(),
        "/nonexistent/m.gguf",
        "--contxt",
        "8",
    ]);
    assert_eq!(code, 1, "{text}");
    assert!(text.contains("unknown flag --contxt"), "{text}");
}

#[test]
fn unparsable_value_is_an_error() {
    let c = corpus();
    let (code, text) = run(&[
        "train",
        c.to_str().unwrap(),
        "/nonexistent/m.gguf",
        "--context",
        "12x",
    ]);
    assert_eq!(code, 1, "{text}");
    assert!(text.contains("--context") && text.contains("12x"), "{text}");
}

#[test]
fn missing_value_is_an_error() {
    let (code, text) = run(&["generate", "m.gguf", "hi", "--temp"]);
    assert_eq!(code, 1, "{text}");
    assert!(text.contains("--temp needs a value"), "{text}");
}

#[test]
fn missing_resume_checkpoint_is_an_error() {
    let c = corpus();
    let (code, text) = run(&[
        "train",
        c.to_str().unwrap(),
        "/nonexistent/m.gguf",
        "--resume",
        "/nonexistent/nope.fckp",
    ]);
    assert_eq!(code, 1, "{text}");
    assert!(
        text.contains("checkpoint /nonexistent/nope.fckp does not exist"),
        "{text}"
    );
}

#[test]
fn flag_valid_elsewhere_is_rejected_for_this_command() {
    let (code, text) = run(&["info", "m.gguf", "--epochs", "3"]);
    assert_eq!(code, 1, "{text}");
    assert!(text.contains("unknown flag --epochs for 'info'"), "{text}");
}

#[test]
fn unknown_command_is_an_error_with_usage() {
    let (code, text) = run(&["frobnicate"]);
    assert_eq!(code, 1, "{text}");
    assert!(text.contains("unknown command: frobnicate"), "{text}");
    assert!(text.contains("USAGE"), "{text}");
}

#[test]
fn no_arguments_prints_usage_and_fails() {
    let (code, text) = run(&[]);
    assert_eq!(code, 1, "{text}");
    assert!(text.contains("USAGE"), "{text}");
}

#[test]
fn help_variants_print_usage_and_succeed() {
    for h in ["help", "-h", "--help"] {
        let (code, text) = run(&[h]);
        assert_eq!(code, 0, "{h}: {text}");
        assert!(text.contains("USAGE"), "{h}: {text}");
    }
}

#[test]
fn valid_flags_pass_the_parser_and_reach_the_command() {
    // Parses fine, then fails later on the missing model file (not the parser).
    let (code, text) = run(&[
        "generate",
        "/nonexistent/m.gguf",
        "hi",
        "--temp",
        "-1",
        "-v",
    ]);
    assert_eq!(code, 1, "{text}");
    assert!(!text.contains("unknown flag"), "{text}");
    assert!(!text.contains("needs a value"), "{text}");
}

#[test]
fn aliases_use_their_commands_flag_table() {
    // `gguf` accepts --top-k; `export` accepts --quant; `gen` rejects --top-k.
    let (_, text) = run(&["gguf", "/nonexistent.gguf", "--top-k", "3"]);
    assert!(!text.contains("unknown flag"), "{text}");
    let (code, text) = run(&["gen", "m.gguf", "hi", "--top-k", "3"]);
    assert_eq!(code, 1);
    assert!(text.contains("unknown flag --top-k for 'gen'"), "{text}");
}

#[test]
fn eval_and_info_dispatch_past_the_parser() {
    for args in [
        ["eval", "/nonexistent/m.gguf", "/nonexistent/t.txt"],
        ["info", "/nonexistent/m.gguf", "-v"],
    ] {
        let (code, text) = run(&args);
        assert_eq!(code, 1, "{text}");
        assert!(!text.contains("unknown"), "{text}");
    }
}

#[test]
fn info_reports_gguf_format_and_rejects_non_gguf() {
    let dir = std::env::temp_dir().join(format!("ferrum_cli_info_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let corpus_path = dir.join("c.txt");
    std::fs::write(&corpus_path, "abc abc abc\n".repeat(50)).unwrap();
    let model = dir.join("m.gguf");
    let (code, text) = run(&[
        "train",
        corpus_path.to_str().unwrap(),
        model.to_str().unwrap(),
        "--epochs",
        "1",
        "--context",
        "8",
        "--embed",
        "16",
        "--heads",
        "2",
        "--blocks",
        "1",
        "--hidden",
        "32",
    ]);
    assert_eq!(code, 0, "{text}");
    assert!(text.contains("(int8 GGUF)"), "{text}");
    let (code, text) = run(&["info", model.to_str().unwrap()]);
    assert_eq!(code, 0, "{text}");
    assert!(text.contains("Format    : GGUF v3 (gpt2, int8)"), "{text}");

    let junk = dir.join("junk.gguf");
    std::fs::write(&junk, b"FINF\x05\x00\x00\x00").unwrap();
    let (code, text) = run(&["info", junk.to_str().unwrap()]);
    assert_ne!(code, 0);
    assert!(text.contains("old Ferrum .bin (FINF) model"), "{text}");
    let (code, text) = run(&["info"]);
    assert_ne!(code, 0);
    assert!(
        text.contains("usage: train_transformer info <model.gguf>"),
        "{text}"
    );
}
