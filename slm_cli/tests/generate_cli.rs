//! `generate` / `run` sampling and stop flags (Phase 1A Task 2).
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::OnceLock;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_train_transformer"))
}

fn scratch() -> &'static Path {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let d = std::env::temp_dir().join(format!("ferrum_gen_cli_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    })
}

/// Train a tiny model once per `name`: `corpus` text, `vocab` 0 = char-level.
fn train(name: &str, corpus: &str, vocab: &str, epochs: &str) -> PathBuf {
    let dir = scratch();
    let c = dir.join(format!("{name}.txt"));
    let m = dir.join(format!("{name}.gguf"));
    std::fs::write(&c, corpus).unwrap();
    let st = bin()
        .args(["train", c.to_str().unwrap(), m.to_str().unwrap()])
        .args(["--context", "8", "--embed", "16", "--heads", "2"])
        .args(["--blocks", "1", "--hidden", "32", "--epochs", epochs])
        .args(["--vocab", vocab])
        .output()
        .unwrap();
    assert!(
        st.status.success(),
        "{}",
        String::from_utf8_lossy(&st.stderr)
    );
    m
}

fn tiny_model() -> &'static Path {
    static M: OnceLock<PathBuf> = OnceLock::new();
    M.get_or_init(|| {
        let text = "the cat sat on the mat. the dog sat on the log. ".repeat(40);
        train("bpe", &text, "270", "2")
    })
}

fn out(o: Output) -> String {
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    String::from_utf8(o.stdout).unwrap()
}

fn generate_on(model: &Path, seed: &str, extra: &[&str]) -> Output {
    bin()
        .args(["generate", model.to_str().unwrap(), seed, "--temp", "0"])
        .args(extra)
        .output()
        .unwrap()
}

fn generate(extra: &[&str]) -> String {
    out(generate_on(tiny_model(), "the cat", extra))
}

#[test]
fn cli_generate_stop_list_cuts_output() {
    let free = generate(&["--max-tokens", "30"]);
    let cont = free.trim_end().strip_prefix("the cat").unwrap().to_string();
    // A 3-char stop taken from the middle of the free run spans token boundaries.
    let stop: String = cont.chars().skip(2).take(3).collect();
    assert_eq!(stop.chars().count(), 3, "{free:?}");
    let cut = generate(&["--max-tokens", "30", "--stop", "#none#", "--stop", &stop]);
    // Stops are matched in the continuation only and excluded from the output.
    let at = cont.find(&stop).unwrap();
    assert_eq!(cut.trim_end(), format!("the cat{}", &cont[..at]), "{cut:?}");
}

#[test]
fn cli_generate_stop_alone_uses_the_chars_budget() {
    let free = generate(&["--max-tokens", "40"]);
    // No --max-tokens: the token budget defaults to --chars.
    let capped = generate(&["--stop", "#none#", "--chars", "4"]);
    assert!(
        capped.trim_end().len() < free.trim_end().len(),
        "{capped:?}"
    );
    let cont = capped.trim_end().strip_prefix("the cat").unwrap();
    assert!(cont.chars().count() <= 4 * 4, "{capped:?}");
}

#[test]
fn cli_generate_max_tokens_zero_returns_only_the_seed() {
    assert_eq!(generate(&["--max-tokens", "0"]), "the cat\n");
}

#[test]
fn cli_generate_token_path_prints_once_with_stream() {
    let plain = generate(&["--max-tokens", "12", "--stop", "#none#"]);
    let streamed = generate(&["--max-tokens", "12", "--stop", "#none#", "--stream"]);
    assert_eq!(plain, streamed);
    assert_eq!(streamed.matches("the cat").count(), 1, "{streamed:?}");
}

#[test]
fn cli_generate_accepts_sampling_knobs() {
    let knobs = [
        "--top-k",
        "5",
        "--top-p",
        "0.9",
        "--repetition-penalty",
        "1.2",
        "--gen-seed",
        "3",
        "--temp",
        "0.8",
    ];
    let a = generate(&knobs);
    let b = generate(&knobs);
    assert_eq!(a, b);
    assert!(a.starts_with("the cat"));
    // The knobs reach the sampler on the streaming path as well.
    let mut s = knobs.to_vec();
    s.push("--stream");
    assert_eq!(generate(&s), a);
    // top-k 1 is greedy whatever the temperature.
    let k1 = generate(&[
        "--top-k",
        "1",
        "--temp",
        "1.5",
        "--gen-seed",
        "9",
        "--chars",
        "30",
    ]);
    let greedy = generate(&["--chars", "30"]);
    assert_eq!(k1, greedy);
}

#[test]
fn cli_run_accepts_the_new_flags() {
    let m = tiny_model();
    let c = scratch().join("bpe.txt");
    let o = bin()
        .args(["run", c.to_str().unwrap(), m.to_str().unwrap(), "the cat"])
        .args(["--temp", "0", "--max-tokens", "6", "--stop", "#none#"])
        .output()
        .unwrap();
    let text = out(o);
    assert!(text.contains("\nthe cat"), "{text:?}");
}

#[test]
fn cli_generate_short_seed_on_char_level_needs_the_token_path() {
    let m = train("chars", &"abcabcabc ".repeat(40), "0", "2");
    // Plain path: the seed must fill the context window.
    let o = generate_on(&m, "ab", &["--chars", "5"]);
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("at least 8"));
    // Token path: any prompt length is fine.
    let text = out(generate_on(&m, "ab", &["--max-tokens", "5"]));
    assert!(text.starts_with("ab"), "{text:?}");
    assert!(text.trim_end().chars().count() <= 2 + 5, "{text:?}");
}

#[test]
fn cli_generate_stops_at_end_of_document() {
    let corpus = "the cat sat.<|endoftext|>the dog ran.<|endoftext|>".repeat(60);
    let m = train("docs", &corpus, "280", "40");
    let text = out(generate_on(&m, "the", &["--chars", "3000"]));
    assert!(text.starts_with("the"), "{text:?}");
    assert!(
        text.trim_end().chars().count() < 3000,
        "generation ignored EOS: {} chars",
        text.len()
    );
}

// ── train --quant and info (Phase 1A Task 3) ─────────────────────────────────

fn train_quant(name: &str, quant: &[&str], extra: &[&str]) -> (PathBuf, Output) {
    let dir = scratch();
    let c = dir.join("q.txt");
    std::fs::write(&c, "the cat sat on the mat. ".repeat(60)).unwrap();
    let m = dir.join(format!("{name}.gguf"));
    let o = bin()
        .args(["train", c.to_str().unwrap(), m.to_str().unwrap()])
        .args(["--context", "8", "--embed", "16", "--heads", "2"])
        .args(["--blocks", "1", "--hidden", "32", "--epochs", "1"])
        .args(["--vocab", "270"])
        .args(quant)
        .args(extra)
        .output()
        .unwrap();
    (m, o)
}

fn info(model: &Path) -> String {
    out(bin()
        .args(["info", model.to_str().unwrap()])
        .output()
        .unwrap())
}

fn line<'a>(text: &'a str, key: &str) -> &'a str {
    text.lines()
        .find_map(|l| l.strip_prefix(key))
        .unwrap_or_else(|| panic!("no {key:?} line in {text}"))
        .trim_start_matches([' ', ':'])
}

#[test]
fn info_reports_params_and_int4() {
    let (m, o) = train_quant("q4", &["--quant", "int4"], &[]);
    let stdout = String::from_utf8_lossy(&o.stdout).to_string();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert!(stdout.contains("(int4 GGUF)"), "{stdout}");
    let text = info(&m);
    assert!(text.contains("GGUF v3 (gpt2, int4)"), "{text}");
    // ctx 8, embed 16, hidden 32, 1 block, untied head, no biases on the head:
    // tok V*16 + pos 8*16 + block (4*(16*16+16) + 4*16 + 16*32+32 + 32*16+16) + lnf 2*16 + head 16*V
    let v: usize = line(&text, "Output dim").parse().unwrap();
    let block = 4 * (16 * 16 + 16) + 4 * 16 + (16 * 32 + 32) + (32 * 16 + 16);
    let want = v * 16 + 8 * 16 + block + 2 * 16 + 16 * v;
    assert_eq!(line(&text, "Params    "), want.to_string(), "{text}");
}

#[test]
fn info_counts_a_tied_head_once() {
    // Same tiny config as `info_reports_params_and_int4`, trained with --tie:
    // the file omits output.weight, so `info` must not count the head again.
    let block = 4 * (16 * 16 + 16) + 4 * 16 + (16 * 32 + 32) + (32 * 16 + 16);
    for q in ["f32", "int8", "int4"] {
        let (m, o) = train_quant(&format!("tied_{q}"), &["--quant", q], &["--tie"]);
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
        let text = info(&m);
        let v: usize = line(&text, "Output dim").parse().unwrap();
        let untied = v * 16 + 8 * 16 + block + 2 * 16 + 16 * v;
        assert_eq!(
            line(&text, "Params    "),
            (untied - v * 16).to_string(),
            "{q}: {text}"
        );
    }
}

#[test]
fn train_quant_selects_the_on_disk_precision() {
    let mut sizes = Vec::new();
    for q in ["int8", "int4", "f32"] {
        let (m, o) = train_quant(&format!("sz_{q}"), &["--quant", q], &[]);
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
        assert!(info(&m).contains(&format!("(gpt2, {q})")));
        sizes.push(std::fs::metadata(&m).unwrap().len());
    }
    assert!(sizes[1] < sizes[0] && sizes[0] < sizes[2], "{sizes:?}");
    // The default is int8.
    let (m, o) = train_quant("sz_default", &[], &[]);
    assert!(o.status.success());
    assert!(info(&m).contains("(gpt2, int8)"));
}

#[test]
fn train_rejects_an_unknown_quant_before_training() {
    let (m, o) = train_quant("badq", &["--quant", "int3"], &[]);
    assert!(!o.status.success());
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(
        err.contains("--quant must be int8|int4|f32 (got 'int3')"),
        "{err}"
    );
    assert!(!m.exists(), "a model was written despite the bad flag");
    assert!(
        !String::from_utf8_lossy(&o.stdout).contains("Corpus"),
        "trained first"
    );
}

// ── training banner, -v progress, --debug, tokenizer reuse (Phase 1A Task 4) ──

fn train_with(name: &str, extra: &[&str]) -> (PathBuf, Output) {
    let dir = scratch();
    let c = dir.join("q.txt");
    std::fs::write(
        &c,
        "the cat sat on the mat. the dog sat on the log. ".repeat(60),
    )
    .unwrap();
    let m = dir.join(format!("{name}.gguf"));
    let _ = std::fs::remove_file(&m);
    let o = bin()
        .args(["train", c.to_str().unwrap(), m.to_str().unwrap()])
        .args(["--context", "8", "--embed", "16", "--heads", "2"])
        .args(["--blocks", "1", "--hidden", "32", "--batch", "2"])
        .args(["--vocab", "270"])
        .args(extra)
        .output()
        .unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    (m, o)
}

fn stdout_of(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).to_string()
}

#[test]
fn train_banner_says_budget_instead_of_a_misleading_epoch_count() {
    let (_, o) = train_with("banner_budget", &["--tokens", "2000"]);
    let text = stdout_of(&o);
    assert!(text.contains("Epochs  : budget"), "{text}");
    assert!(!text.contains("Epochs  : 100"), "{text}");
    let (_, o) = train_with("banner_epochs", &["--epochs", "3"]);
    assert!(stdout_of(&o).contains("Epochs  : 3 "), "{}", stdout_of(&o));
}

#[test]
fn verbose_train_prints_progress_and_debug_prints_engine_internals() {
    let (_, o) = train_with("verbose", &["--epochs", "3", "-v"]);
    let text = stdout_of(&o);
    assert!(text.contains("[train] step 10/"), "{text}");
    assert!(text.contains("tokens/s"), "{text}");
    // -v alone no longer dumps engine internals.
    let all = text + &String::from_utf8_lossy(&o.stderr);
    assert!(!all.contains("[slm::"), "{all}");
    let (_, quiet) = train_with("quiet", &["--epochs", "3"]);
    assert!(!stdout_of(&quiet).contains("[train] step"));
    let (_, o) = train_with("debug", &["--epochs", "1", "--debug"]);
    let all = stdout_of(&o) + &String::from_utf8_lossy(&o.stderr);
    assert!(all.contains("[slm::"), "{all}");
}

#[test]
fn val_with_an_existing_tokenizer_file_prints_a_leak_note() {
    let tok = scratch().join("reuse_tok.gguf");
    let tok_s = tok.to_str().unwrap();
    let _ = std::fs::remove_file(&tok);
    // The tokenizer is created by the first run: nothing to warn about yet.
    let (_, o) = train_with(
        "tok1",
        &["--epochs", "1", "--val", "0.1", "--tokenizer", tok_s],
    );
    assert!(
        !stdout_of(&o).contains("note: reusing"),
        "{}",
        stdout_of(&o)
    );
    assert!(tok.exists());
    let (_, o) = train_with(
        "tok2",
        &["--epochs", "1", "--val", "0.1", "--tokenizer", tok_s],
    );
    let text = stdout_of(&o);
    assert!(text.contains(&format!("note: reusing {tok_s}")), "{text}");
    assert!(text.contains("validation text"), "{text}");
    // Without --val there is no leak to warn about.
    let (_, o) = train_with("tok3", &["--epochs", "1", "--tokenizer", tok_s]);
    assert!(!stdout_of(&o).contains("note: reusing"));
}
