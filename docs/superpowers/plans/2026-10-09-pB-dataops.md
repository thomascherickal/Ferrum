# Phase 1B — DataOps (prepare + DVC) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible, DVC-driven data pipeline: `train_transformer prepare` cleans, dedups, and splits raw text by document and writes a hashed manifest. `dvc repro` runs fetch → prepare → train → eval on TinyStories.

**Architecture:** Pure text-cleaning functions live in `ferrum_core::dataset`. Streaming I/O, hashing, dedup, splitting, and the manifest live in a new `slm_cli/src/prepare.rs`, which is wired into `main.rs` as the `prepare` subcommand. DVC stages are plain CLI calls (`cargo run --release -q -p slm_cli -- …`), templated from `params.yaml`, so Rust never parses YAML.

**Tech Stack:** Rust 2021; `unicode-normalization 0.1` (ferrum_core), `sha2 0.11` and `serde_json 1` (slm_cli); DVC `3.67.1` in a repo-local `.venv`; bash + curl + sha256sum.

**Spec:** `docs/superpowers/specs/2026-10-09-dataops-serve-eval-design.md` (§5). Phase 0 plan: `docs/superpowers/plans/2026-10-09-p0-shared-surfaces.md`.

## Global Constraints

- Start from the Phase 0 tip in a worktree: `git worktree add ../Ferrum-B -b feat/dos-B feat/dataops-serve-eval`.
- `#![forbid(unsafe_code)]` stays. `ferrum_core` must build for `wasm32-unknown-unknown`. `unicode-normalization` is pure Rust and wasm-safe. Do not add file-system or hashing crates to `ferrum_core`.
- `prepare` output must be **byte-identical** across runs with the same inputs and flags. That means no timestamps or absolute paths in the manifest, and maps serialized in sorted key order.
- Default corpus (pinned):
  - URL `https://huggingface.co/datasets/roneneldan/TinyStories/resolve/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/TinyStoriesV2-GPT4-valid.txt`
  - size 22,502,601 bytes
  - sha256 `6874bae9a4c1a4e7edcf0e53b86c17817e9cf881fc75ff2368da457b80c0585d`
  - licence CDLA-Sharing-1.0, never committed
- New flags go in `const` tables via the Phase 0 parser (`Kind`, `Spec`, `Args::{value, get, get_all, has}`).
- Do not edit `ReadMe.md`, `Manual.md`, `Status.md` (Phase 2).
- All model and tokenizer files are GGUF (`.gguf`). Every feature lands with its unit tests in the same commit. No `unsafe` code.
- Each commit passes `cargo test --workspace --release`, `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo build -p slm_wasm --target wasm32-unknown-unknown --release`.

## Contract with Phase 1C

`dvc.yaml`'s optional `lighteval` stage calls
`eval/lighteval/run.sh --model models/slm.gguf --test data/prepared/test.txt --out metrics/lighteval.json`.
Phase 1C creates that script. The stage is `frozen: true`, so `dvc repro` skips it until a user unfreezes it.

## Merge note

Phase 1A adds `train --quant`; it is not available in this worktree. Phase 2 adds `quant: int8` to `params.yaml` and `--quant ${train.quant}` to the `train` stage after the merge.

## Planning refinement

`prepare` writes `val.txt`. The trainer's `--val` keeps using its internal tail split of `train.txt` (no library change, so no conflict with Phase 1A). The DVC `eval` stage scores **both** `val.txt` and `test.txt`.

## Review Focus

1. A file with a UTF-8 BOM, zero-width spaces, and CRLF line endings — none survive into the output. → Task 1 test `strip_invisible_removes_bom_zero_width_and_controls`, Task 2 test `prepare_cleans_bom_crlf_and_zero_width`.
2. Appending new documents to an input — every previously assigned document keeps its split. → Task 2 test `splits_are_stable_when_documents_are_appended`.
3. Invalid UTF-8 bytes — clear error with file and byte offset; `--lossy` succeeds. → Task 2 test `invalid_utf8_errors_unless_lossy`.
4. Two Gutenberg books concatenated in one file — both bodies kept, both headers removed. → Task 1 test `gutenberg_bodies_keeps_every_book`.
5. Running `prepare` twice — byte-identical outputs and manifest. → Task 2 test `prepare_is_deterministic`.

---

### Task 1: Cleaning primitives in `ferrum_core::dataset`

**Files:**
- Modify: `ferrum_core/Cargo.toml` (add `unicode-normalization = "0.1"`)
- Modify: `ferrum_core/src/dataset.rs`, `ferrum_core/src/lib.rs` (re-exports)

**Interfaces:**
- Produces:
  ```rust
  /// Remove U+FEFF (BOM), U+200B–U+200D, U+2060, `\r`, and control chars except `\n`/`\t`.
  pub fn strip_invisible(text: &str) -> String;
  /// Bodies of every `*** START OF … ***` / `*** END OF … ***` pair, in order;
  /// `vec![raw]` when there is no complete pair.
  pub fn gutenberg_bodies(raw: &str) -> Vec<&str>;
  /// Collapse runs of spaces/tabs inside each line to one space.
  pub fn squeeze_spaces(text: &str) -> String;
  /// Unicode NFKC normalization.
  pub fn normalize_nfkc(text: &str) -> String;
  ```
  `clean_corpus` keeps its signature. It now uses `gutenberg_bodies` (joined by `"\n\n"`) and `strip_invisible` when `strip_control_chars` is set.

- [ ] **Step 1: Failing tests** (append to `dataset.rs` tests)

```rust
    #[test]
    fn strip_invisible_removes_bom_zero_width_and_controls() {
        let raw = "\u{FEFF}he\u{200B}llo\r\n\u{2060}wor\u{0007}ld\tok\u{200D}";
        assert_eq!(strip_invisible(raw), "hello\nworld\tok");
    }

    #[test]
    fn gutenberg_bodies_keeps_every_book() {
        let raw = "hdr1\n*** START OF BOOK ONE ***\nalpha\n*** END OF BOOK ONE ***\nlicense\n\
                   hdr2\n*** START OF BOOK TWO ***\nbeta\n*** END OF BOOK TWO ***\nfooter";
        let b = gutenberg_bodies(raw);
        assert_eq!(b.len(), 2);
        assert_eq!(b[0].trim(), "alpha");
        assert_eq!(b[1].trim(), "beta");
        assert_eq!(gutenberg_bodies("no markers"), vec!["no markers"]);
        let cleaned = clean_corpus(raw, &CleanOptions::default());
        assert!(cleaned.contains("alpha") && cleaned.contains("beta"));
        assert!(!cleaned.contains("license") && !cleaned.contains("hdr2"));
    }

    #[test]
    fn squeeze_spaces_and_nfkc() {
        assert_eq!(squeeze_spaces("a  b\t\t c\n  d"), "a b c\n d");
        assert_eq!(normalize_nfkc("ﬁ ① Ａ"), "fi 1 A");
    }

    #[test]
    fn clean_corpus_strips_bom_by_default() {
        let out = clean_corpus("\u{FEFF}Hi\u{200B} there", &CleanOptions::default());
        assert_eq!(out.trim_end(), "Hi there");
    }
```

- [ ] **Step 2: Verify failure** — `cargo test -p ferrum_core --release --lib dataset 2>&1 | tail -5`. Expected: compile error (`strip_invisible` not found).

- [ ] **Step 3: Implement**

```rust
/// Characters that render as nothing but break tokenization and dedup.
fn is_invisible(ch: char) -> bool {
    matches!(ch, '\u{FEFF}' | '\u{200B}' | '\u{200C}' | '\u{200D}' | '\u{2060}')
}

/// Remove the BOM, zero-width characters, `\r`, and control characters other
/// than `\n` and `\t`.
pub fn strip_invisible(text: &str) -> String {
    text.chars()
        .filter(|&c| !(is_invisible(c) || c == '\r' || (c.is_control() && c != '\n' && c != '\t')))
        .collect()
}

/// The body of every Project Gutenberg book in `raw`: the text between each
/// `*** START OF … ***` line and the following `*** END OF` marker, in order.
/// Returns `vec![raw]` when no complete START/END pair exists.
pub fn gutenberg_bodies(raw: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut rest = 0usize;
    while let Some(s) = find_marker(&raw[rest..], "START OF") {
        let s_abs = rest + s;
        let Some(body_start) = raw[s_abs..].find('\n').map(|j| s_abs + j + 1) else { break };
        let Some(e) = find_marker(&raw[body_start..], "END OF") else { break };
        let e_abs = body_start + e;
        out.push(&raw[body_start..e_abs]);
        rest = raw[e_abs..].find('\n').map_or(raw.len(), |j| e_abs + j + 1);
    }
    if out.is_empty() {
        vec![raw]
    } else {
        out
    }
}

/// Offset of `*** <what>` or `***<what>` in `s`, whichever comes first.
fn find_marker(s: &str, what: &str) -> Option<usize> {
    let a = s.find(&format!("*** {what}"));
    let b = s.find(&format!("***{what}"));
    match (a, b) {
        (Some(x), Some(y)) => Some(x.min(y)),
        (x, y) => x.or(y),
    }
}

/// Collapse runs of spaces and tabs within each line to a single space.
pub fn squeeze_spaces(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut in_run = false;
    for ch in text.chars() {
        if ch == ' ' || ch == '\t' {
            if !in_run {
                out.push(' ');
            }
            in_run = true;
        } else {
            out.push(ch);
            in_run = false;
        }
    }
    out
}

/// Unicode NFKC normalization (compatibility forms such as ligatures,
/// full-width letters, and circled digits fold to their plain equivalents).
pub fn normalize_nfkc(text: &str) -> String {
    use unicode_normalization::UnicodeNormalization;
    text.nfkc().collect()
}
```

In `clean_corpus`:
- Replace the `strip_gutenberg_boilerplate` call with an owned body: `let joined; let body = if opts.strip_gutenberg { joined = gutenberg_bodies(raw).join("\n\n"); joined.as_str() } else { raw };`.
- In the char loop, extend the control-char condition to also drop `is_invisible(ch)` when `opts.strip_control_chars` is set.
- Delete `strip_gutenberg_boilerplate`.
- Update the module doc's first paragraph: drop the "zero-dependency" sentence and mention NFKC.
- In `lib.rs`, add `gutenberg_bodies, normalize_nfkc, squeeze_spaces, strip_invisible` to `pub use dataset::{…}`.

- [ ] **Step 4: Verify** — `cargo test --workspace --release 2>&1 | grep -E "^test result|FAILED"` and the wasm32 build. Expected: green.

- [ ] **Step 5: Commit**

```bash
cargo fmt --all && cargo clippy --workspace --all-targets -- -D warnings
git add ferrum_core Cargo.lock
git commit -m "feat(dataset): strip_invisible, multi-book gutenberg_bodies, squeeze_spaces, NFKC

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `train_transformer prepare`

**Files:**
- Modify: `slm_cli/Cargo.toml` (add `sha2 = "0.11"`, `serde_json = "1"`)
- Create: `slm_cli/src/prepare.rs`
- Modify: `slm_cli/src/main.rs` (`mod prepare;`, `PREPARE_FLAGS`, `spec_for`, dispatch, usage)
- Create: `slm_cli/tests/prepare_cli.rs`

**Interfaces:**
- Consumes: `ferrum_core::{strip_invisible, gutenberg_bodies, squeeze_spaces, normalize_nfkc, clean_corpus, CleanOptions, DOC_SEP}`.
- Produces: the CLI below, and files `out/{train,val,test}.txt` (documents joined by `"\n<|endoftext|>\n"`, no leading or trailing separator) plus `out/manifest.json`:

```json
{
  "args": ["--input", "...", "..."],
  "ferrum_version": "0.3.0",
  "inputs": [{"path": "data/raw/x.txt", "bytes": 1, "sha256": "…"}],
  "outputs": {"train": {"path": "…/train.txt", "bytes": 1, "sha256": "…", "documents": 1, "chars": 1}, "val": {…}, "test": {…}},
  "rejected": {"duplicate": 0, "empty": 0, "invalid_utf8": 0, "too_long": 0, "too_short": 0},
  "train_stats": {"bytes": 1, "chars": 1, "lines": 1, "unique_chars": 1, "words": 1}
}
```

CLI:
```
train_transformer prepare --input <file|dir> [--input …] --out <dir>
    [--doc-sep S (default "<|endoftext|>")] [--lossy] [--nfkc] [--lowercase]
    [--strip-gutenberg] [--min-chars N (default 1)] [--max-chars N (default 0 = off)]
    [--dedup] [--val-frac F (default 0.05)] [--test-frac F (default 0.05)] [--seed N (default 42)]
```

- [ ] **Step 1: Failing tests** — `slm_cli/tests/prepare_cli.rs`:

```rust
use std::path::{Path, PathBuf};
use std::process::Command;

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ferrum_prep_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn prepare(args: &[&str]) -> (bool, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_train_transformer"))
        .arg("prepare")
        .args(args)
        .output()
        .unwrap();
    (out.status.success(), String::from_utf8_lossy(&out.stderr).to_string() + &String::from_utf8_lossy(&out.stdout))
}

fn stories(n: usize) -> String {
    (0..n).map(|i| format!("Story number {i}. Tom had a red ball.")).collect::<Vec<_>>().join("\n<|endoftext|>\n")
}

fn read(p: &Path) -> String {
    std::fs::read_to_string(p).unwrap()
}

fn docs(p: &Path) -> Vec<String> {
    let t = read(p);
    if t.is_empty() { return vec![]; }
    t.split("\n<|endoftext|>\n").map(String::from).collect()
}

#[test]
fn prepare_writes_splits_and_manifest() {
    let d = tmp("basic");
    let input = d.join("raw.txt");
    std::fs::write(&input, stories(400)).unwrap();
    let out = d.join("out");
    let (ok, log) = prepare(&["--input", input.to_str().unwrap(), "--out", out.to_str().unwrap()]);
    assert!(ok, "{log}");
    let n: usize = ["train", "val", "test"].iter().map(|s| docs(&out.join(format!("{s}.txt"))).len()).sum();
    assert_eq!(n, 400);
    let val = docs(&out.join("val.txt")).len();
    assert!((5..=40).contains(&val), "val docs {val}");
    let m: serde_json::Value = serde_json::from_str(&read(&out.join("manifest.json"))).unwrap();
    let train_bytes = std::fs::read(out.join("train.txt")).unwrap();
    use sha2::Digest;
    let hex = format!("{:x}", sha2::Sha256::digest(&train_bytes));
    assert_eq!(m["outputs"]["train"]["sha256"], hex);
    assert_eq!(m["outputs"]["train"]["bytes"], train_bytes.len());
}

#[test]
fn prepare_is_deterministic() {
    let d = tmp("det");
    let input = d.join("raw.txt");
    std::fs::write(&input, stories(200)).unwrap();
    for o in ["a", "b"] {
        let (ok, log) = prepare(&["--input", input.to_str().unwrap(), "--out", d.join(o).to_str().unwrap(), "--dedup", "--nfkc"]);
        assert!(ok, "{log}");
    }
    for f in ["train.txt", "val.txt", "test.txt"] {
        assert_eq!(std::fs::read(d.join("a").join(f)).unwrap(), std::fs::read(d.join("b").join(f)).unwrap());
    }
    // Manifests differ only in the --out argument.
    let ma = read(&d.join("a/manifest.json")).replace(d.join("a").to_str().unwrap(), "OUT");
    let mb = read(&d.join("b/manifest.json")).replace(d.join("b").to_str().unwrap(), "OUT");
    assert_eq!(ma, mb);
}

#[test]
fn splits_are_stable_when_documents_are_appended() {
    let d = tmp("stable");
    let input = d.join("raw.txt");
    std::fs::write(&input, stories(300)).unwrap();
    prepare(&["--input", input.to_str().unwrap(), "--out", d.join("a").to_str().unwrap()]);
    std::fs::write(&input, stories(300) + "\n<|endoftext|>\n" + &(300..400).map(|i| format!("New story {i}.")).collect::<Vec<_>>().join("\n<|endoftext|>\n")).unwrap();
    prepare(&["--input", input.to_str().unwrap(), "--out", d.join("b").to_str().unwrap()]);
    for s in ["train", "val", "test"] {
        let before = docs(&d.join("a").join(format!("{s}.txt")));
        let after = docs(&d.join("b").join(format!("{s}.txt")));
        assert!(before.iter().all(|x| after.contains(x)), "{s}: a document changed split");
    }
}

#[test]
fn prepare_cleans_bom_crlf_and_zero_width_and_dedups() {
    let d = tmp("clean");
    let input = d.join("raw.txt");
    std::fs::write(&input, "\u{FEFF}Hello\u{200B} world.\r\n<|endoftext|>\r\nHello world.\n<|endoftext|>\nBye  now.").unwrap();
    let out = d.join("out");
    let (ok, log) = prepare(&["--input", input.to_str().unwrap(), "--out", out.to_str().unwrap(), "--dedup", "--val-frac", "0", "--test-frac", "0"]);
    assert!(ok, "{log}");
    let train = docs(&out.join("train.txt"));
    assert_eq!(train, vec!["Hello world.".to_string(), "Bye now.".to_string()]);
    let m: serde_json::Value = serde_json::from_str(&read(&out.join("manifest.json"))).unwrap();
    assert_eq!(m["rejected"]["duplicate"], 1);
}

#[test]
fn invalid_utf8_errors_unless_lossy() {
    let d = tmp("utf8");
    let input = d.join("raw.bin");
    std::fs::write(&input, b"good text\n<|endoftext|>\nbad \xff\xfe bytes").unwrap();
    let out = d.join("out");
    let (ok, log) = prepare(&["--input", input.to_str().unwrap(), "--out", out.to_str().unwrap()]);
    assert!(!ok);
    assert!(log.contains("invalid UTF-8") && log.contains("byte"), "{log}");
    let (ok, log) = prepare(&["--input", input.to_str().unwrap(), "--out", out.to_str().unwrap(), "--lossy", "--val-frac", "0", "--test-frac", "0"]);
    assert!(ok, "{log}");
    assert!(read(&out.join("train.txt")).contains("bad \u{FFFD}"));
}

#[test]
fn length_filters_and_directory_input() {
    let d = tmp("dir");
    let inp = d.join("in");
    std::fs::create_dir_all(&inp).unwrap();
    std::fs::write(inp.join("b.txt"), "tiny").unwrap();
    std::fs::write(inp.join("a.txt"), "a long enough document here").unwrap();
    let out = d.join("out");
    let (ok, log) = prepare(&["--input", inp.to_str().unwrap(), "--out", out.to_str().unwrap(), "--min-chars", "10", "--val-frac", "0", "--test-frac", "0"]);
    assert!(ok, "{log}");
    assert_eq!(docs(&out.join("train.txt")), vec!["a long enough document here".to_string()]);
    let m: serde_json::Value = serde_json::from_str(&read(&out.join("manifest.json"))).unwrap();
    assert_eq!(m["rejected"]["too_short"], 1);
    assert!(m["inputs"][0]["path"].as_str().unwrap().ends_with("a.txt"), "inputs sorted");
}
```

Add `sha2` and `serde_json` under `[dev-dependencies]` in `slm_cli/Cargo.toml` too. They are already normal dependencies after Step 3, so the test can use them; Cargo allows both.

- [ ] **Step 2: Verify failure** — `cargo test -p slm_cli --release --test prepare_cli 2>&1 | grep -E "test |unknown command"`. Expected: FAIL with `unknown command: prepare`.

- [ ] **Step 3: Implement `slm_cli/src/prepare.rs`**

```rust
//! `train_transformer prepare`: stream raw text, clean it per document, drop
//! duplicates and out-of-range lengths, split train/val/test by a seeded
//! content hash, and write a manifest that pins every input and output by
//! sha256 — a deterministic, DVC-friendly data stage.

use ferrum_core::{clean_corpus, gutenberg_bodies, normalize_nfkc, squeeze_spaces, strip_invisible, CleanOptions};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

const JOIN: &str = "\n<|endoftext|>\n";

pub struct PrepareOptions {
    pub inputs: Vec<String>,
    pub out: String,
    pub doc_sep: String,
    pub lossy: bool,
    pub nfkc: bool,
    pub lowercase: bool,
    pub strip_gutenberg: bool,
    pub min_chars: usize,
    pub max_chars: usize,
    pub dedup: bool,
    pub val_frac: f64,
    pub test_frac: f64,
    pub seed: u64,
    /// The raw argument list, recorded in the manifest.
    pub args: Vec<String>,
}

struct Split {
    path: PathBuf,
    w: BufWriter<std::fs::File>,
    hash: Sha256,
    bytes: u64,
    docs: u64,
    chars: u64,
}

impl Split {
    fn create(dir: &Path, name: &str) -> std::io::Result<Self> {
        let path = dir.join(format!("{name}.txt"));
        Ok(Split { w: BufWriter::new(std::fs::File::create(&path)?), path, hash: Sha256::new(), bytes: 0, docs: 0, chars: 0 })
    }
    fn write(&mut self, doc: &str) -> std::io::Result<()> {
        if self.docs > 0 {
            self.put(JOIN.as_bytes())?;
        }
        self.put(doc.as_bytes())?;
        self.docs += 1;
        self.chars += doc.chars().count() as u64;
        Ok(())
    }
    fn put(&mut self, b: &[u8]) -> std::io::Result<()> {
        self.w.write_all(b)?;
        self.hash.update(b);
        self.bytes += b.len() as u64;
        Ok(())
    }
    fn finish(mut self) -> std::io::Result<Value> {
        self.w.flush()?;
        Ok(json!({
            "path": self.path.to_string_lossy(),
            "bytes": self.bytes,
            "sha256": format!("{:x}", self.hash.finalize()),
            "documents": self.docs,
            "chars": self.chars,
        }))
    }
}

/// Every regular, non-hidden file under each input (directories walked
/// recursively), sorted by path.
fn expand_inputs(inputs: &[String]) -> std::io::Result<Vec<PathBuf>> {
    fn walk(p: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
        if p.is_dir() {
            for e in std::fs::read_dir(p)? {
                let e = e?.path();
                if !e.file_name().is_some_and(|n| n.to_string_lossy().starts_with('.')) {
                    walk(&e, out)?;
                }
            }
        } else {
            out.push(p.to_path_buf());
        }
        Ok(())
    }
    let mut out = Vec::new();
    for i in inputs {
        let p = Path::new(i);
        if !p.exists() {
            return Err(std::io::Error::new(std::io::ErrorKind::NotFound, format!("input {i} does not exist")));
        }
        walk(p, &mut out)?;
    }
    out.sort();
    Ok(out)
}

/// Stream `path` as raw documents split on `sep` (byte-exact). An empty
/// separator yields the whole file as one document. Calls `f(bytes, offset)`.
fn for_each_raw_doc(
    path: &Path,
    sep: &str,
    hasher: &mut Sha256,
    total: &mut u64,
    mut f: impl FnMut(&[u8], u64) -> Result<(), String>,
) -> Result<(), String> {
    let file = std::fs::File::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let mut r = BufReader::with_capacity(1 << 20, file);
    let sep = sep.as_bytes();
    let mut doc: Vec<u8> = Vec::new();
    let mut doc_start = 0u64;
    let mut line = Vec::new();
    loop {
        line.clear();
        let n = r.read_until(b'\n', &mut line).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        hasher.update(&line);
        *total += n as u64;
        doc.extend_from_slice(&line);
        if !sep.is_empty() {
            // A separator can only complete inside the bytes just appended.
            let mut from = doc.len().saturating_sub(line.len() + sep.len() - 1);
            while let Some(i) = doc[from..].windows(sep.len()).position(|w| w == sep).map(|i| from + i) {
                f(&doc[..i], doc_start)?;
                doc_start += (i + sep.len()) as u64;
                doc.drain(..i + sep.len());
                from = 0; // what remains is the short tail of this line
            }
        }
    }
    f(&doc, doc_start)
}

pub fn run(o: &PrepareOptions) -> Result<(), Box<dyn std::error::Error>> {
    if !(0.0..1.0).contains(&o.val_frac) || !(0.0..1.0).contains(&o.test_frac) || o.val_frac + o.test_frac >= 1.0 {
        return Err("--val-frac and --test-frac must be in [0, 1) and sum to less than 1".into());
    }
    let out = Path::new(&o.out);
    std::fs::create_dir_all(out)?;
    let mut splits = [Split::create(out, "train")?, Split::create(out, "val")?, Split::create(out, "test")?];
    let val_cut = (o.val_frac * 10_000.0).round() as u64;
    let test_cut = val_cut + (o.test_frac * 10_000.0).round() as u64;
    let clean_opts = CleanOptions {
        strip_gutenberg: false,
        lowercase: o.lowercase,
        collapse_whitespace: true,
        normalize_punctuation: true,
        strip_control_chars: true,
        max_chars: None,
    };
    let mut rejected: BTreeMap<&str, u64> = ["duplicate", "empty", "invalid_utf8", "too_long", "too_short"].map(|k| (k, 0)).into();
    let mut seen: HashSet<[u8; 16]> = HashSet::new();
    let mut uniq: HashSet<char> = HashSet::new();
    let (mut lines, mut words) = (0u64, 0u64);
    let mut inputs = Vec::new();

    for path in expand_inputs(&o.inputs)? {
        let mut hasher = Sha256::new();
        let mut total = 0u64;
        for_each_raw_doc(&path, &o.doc_sep, &mut hasher, &mut total, |raw, offset| {
            let text = match std::str::from_utf8(raw) {
                Ok(s) => std::borrow::Cow::Borrowed(s),
                Err(_) if o.lossy => String::from_utf8_lossy(raw),
                Err(e) => {
                    return Err(format!(
                        "{}: invalid UTF-8 at byte {} (pass --lossy to replace invalid bytes)",
                        path.display(),
                        offset + e.valid_up_to() as u64
                    ))
                }
            };
            let books = if o.strip_gutenberg { gutenberg_bodies(&text) } else { vec![text.as_ref()] };
            for book in books {
                let mut doc = strip_invisible(book);
                if o.nfkc {
                    doc = normalize_nfkc(&doc);
                }
                let doc = squeeze_spaces(&clean_corpus(&doc, &clean_opts));
                let doc = doc.trim();
                let n = doc.chars().count();
                let reason = if n == 0 {
                    Some("empty")
                } else if n < o.min_chars {
                    Some("too_short")
                } else if o.max_chars > 0 && n > o.max_chars {
                    Some("too_long")
                } else {
                    None
                };
                if let Some(r) = reason {
                    *rejected.get_mut(r).unwrap() += 1;
                    continue;
                }
                if o.dedup {
                    let key: [u8; 16] = Sha256::digest(doc.as_bytes())[..16].try_into().unwrap();
                    if !seen.insert(key) {
                        *rejected.get_mut("duplicate").unwrap() += 1;
                        continue;
                    }
                }
                let mut h = Sha256::new();
                h.update(o.seed.to_le_bytes());
                h.update(doc.as_bytes());
                let bucket = u64::from_be_bytes(h.finalize()[..8].try_into().unwrap()) % 10_000;
                let which = if bucket < val_cut { 1 } else if bucket < test_cut { 2 } else { 0 };
                if which == 0 {
                    uniq.extend(doc.chars());
                    lines += doc.lines().count() as u64;
                    words += doc.split_whitespace().count() as u64;
                }
                splits[which].write(doc).map_err(|e| e.to_string())?;
            }
            Ok(())
        })?;
        inputs.push(json!({"path": path.to_string_lossy(), "bytes": total, "sha256": format!("{:x}", hasher.finalize())}));
    }

    let [train, val, test] = splits;
    let train = train.finish()?;
    let train_stats = json!({
        "bytes": train["bytes"], "chars": train["chars"], "lines": lines,
        "unique_chars": uniq.len(), "words": words,
    });
    let mut outputs = Map::new();
    outputs.insert("test".into(), test.finish()?);
    outputs.insert("train".into(), train);
    outputs.insert("val".into(), val.finish()?);
    let manifest = json!({
        "args": o.args,
        "ferrum_version": env!("CARGO_PKG_VERSION"),
        "inputs": inputs,
        "outputs": outputs,
        "rejected": rejected,
        "train_stats": train_stats,
    });
    std::fs::write(out.join("manifest.json"), serde_json::to_string_pretty(&manifest)? + "\n")?;
    let docs: Vec<String> = ["train", "val", "test"].iter().map(|s| format!("{s}={}", manifest["outputs"][s]["documents"])).collect();
    println!("prepared {} → {}  ({})  rejected {:?}", o.inputs.join(" "), o.out, docs.join(" "), manifest["rejected"]);
    Ok(())
}
```

Notes for the implementer:
- `serde_json`'s default `Map` is a `BTreeMap`, so keys serialize sorted and the manifest is deterministic. Do **not** enable serde_json's `preserve_order` feature.
- Remove the unused `Read` import if clippy flags it.
- `clean_corpus` with `collapse_whitespace` trims leading and trailing blank lines and ends every line with `\n`. The `.trim()` afterwards removes the trailing newline, so documents join cleanly.

In `main.rs`:

```rust
mod prepare;

const PREPARE_FLAGS: Spec = &[
    ("input", Kind::StrList), ("out", Kind::Str), ("doc-sep", Kind::Str),
    ("lossy", Kind::Bool), ("nfkc", Kind::Bool), ("lowercase", Kind::Bool),
    ("strip-gutenberg", Kind::Bool), ("min-chars", Kind::Usize), ("max-chars", Kind::Usize),
    ("dedup", Kind::Bool), ("val-frac", Kind::F64), ("test-frac", Kind::F64), ("seed", Kind::U64),
];

fn cmd_prepare(args: &Args, raw: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let inputs = args.get_all("input");
    let out = args.value("out").ok_or("usage: train_transformer prepare --input <file|dir> --out <dir> [options]")?;
    if inputs.is_empty() {
        return Err("prepare needs at least one --input".into());
    }
    prepare::run(&prepare::PrepareOptions {
        inputs,
        out: out.to_string(),
        doc_sep: args.value("doc-sep").unwrap_or(ferrum_core::DOC_SEP).to_string(),
        lossy: args.has("lossy"),
        nfkc: args.has("nfkc"),
        lowercase: args.has("lowercase"),
        strip_gutenberg: args.has("strip-gutenberg"),
        min_chars: args.get("min-chars", 1),
        max_chars: args.get("max-chars", 0),
        dedup: args.has("dedup"),
        val_frac: args.get("val-frac", 0.05),
        test_frac: args.get("test-frac", 0.05),
        seed: args.get("seed", 42),
        args: raw.to_vec(),
    })
}
```

Add `"prepare" => PREPARE_FLAGS` to `spec_for` and `"prepare" => cmd_prepare(&args, &argv[2..])` to the dispatch. Add a `prepare` line and an options block to `print_usage` and the module doc.

- [ ] **Step 4: Verify** — `cargo test -p slm_cli --release 2>&1 | grep -E "^test result|FAILED"`. Expected: all ok, including the 6 new tests.

- [ ] **Step 5: Commit**

```bash
cargo fmt --all && cargo clippy --workspace --all-targets -- -D warnings
git add slm_cli Cargo.lock
git commit -m "feat(cli): prepare — streamed clean/dedup/split with sha256 manifest

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `eval --json`

**Files:**
- Modify: `slm_cli/src/main.rs` (`EVAL_FLAGS`, `cmd_eval`)
- Test: `slm_cli/tests/prepare_cli.rs` (append)

**Interfaces:**
- Produces: `train_transformer eval <model> <text> [--json <path>]`, which writes `{"bits_per_byte","bits_per_token","bytes","cross_entropy_nats","perplexity","seconds","tokens"}`.

- [ ] **Step 1: Failing test**

```rust
#[test]
fn eval_writes_json_metrics() {
    let d = tmp("evaljson");
    let corpus = d.join("c.txt");
    std::fs::write(&corpus, "the cat sat on the mat. ".repeat(60)).unwrap();
    let model = d.join("m.bin");
    let st = Command::new(env!("CARGO_BIN_EXE_train_transformer"))
        .args(["train", corpus.to_str().unwrap(), model.to_str().unwrap(), "--context", "8", "--embed", "16",
               "--heads", "2", "--blocks", "1", "--hidden", "32", "--epochs", "1", "--vocab", "270"])
        .status().unwrap();
    assert!(st.success());
    let json = d.join("m.json");
    let st = Command::new(env!("CARGO_BIN_EXE_train_transformer"))
        .args(["eval", model.to_str().unwrap(), corpus.to_str().unwrap(), "--json", json.to_str().unwrap()])
        .status().unwrap();
    assert!(st.success());
    let v: serde_json::Value = serde_json::from_str(&read(&json)).unwrap();
    for k in ["perplexity", "cross_entropy_nats", "bits_per_token", "bits_per_byte", "tokens", "bytes", "seconds"] {
        assert!(v.get(k).is_some(), "missing {k}");
    }
    let bpb = v["bits_per_byte"].as_f64().unwrap();
    let want = v["cross_entropy_nats"].as_f64().unwrap() * v["tokens"].as_f64().unwrap()
        / std::f64::consts::LN_2 / v["bytes"].as_f64().unwrap();
    assert!((bpb - want).abs() < 1e-6);
}
```

- [ ] **Step 2: Verify failure** — Expected: `unknown flag --json for 'eval'`.

- [ ] **Step 3: Implement**

Change `spec_for`'s `"eval" | "info" => NO_FLAGS` to `"eval" => EVAL_FLAGS, "info" => NO_FLAGS`, with `const EVAL_FLAGS: Spec = &[("json", Kind::Str)];`. In `cmd_eval`, time the evaluation and write the JSON when requested:

```rust
    let t0 = Instant::now();
    let eval = slm.evaluate(&text)?;
    let seconds = t0.elapsed().as_secs_f64();
    let bytes = text.len();
    let bits_per_byte = eval.cross_entropy as f64 * eval.num_predictions as f64 / std::f64::consts::LN_2 / bytes as f64;
    // … existing println! lines, plus:
    println!("Bits/byte    : {bits_per_byte:.4}");
    println!("Seconds      : {seconds:.2}");
    if let Some(path) = args.value("json") {
        let v = serde_json::json!({
            "perplexity": eval.perplexity, "cross_entropy_nats": eval.cross_entropy,
            "bits_per_token": eval.bits_per_token, "bits_per_byte": bits_per_byte,
            "tokens": eval.num_predictions, "bytes": bytes, "seconds": seconds,
        });
        if let Some(parent) = std::path::Path::new(path).parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, serde_json::to_string_pretty(&v)? + "\n")?;
    }
```

Note: `seconds` makes the metrics file differ between runs. That is fine for DVC metrics (`cache: false`); `dvc metrics diff` shows it.

- [ ] **Step 4: Verify and commit**

```bash
cargo test -p slm_cli --release 2>&1 | grep -E "^test result|FAILED"
cargo fmt --all && cargo clippy --workspace --all-targets -- -D warnings
git add slm_cli
git commit -m "feat(cli): eval --json writes DVC-trackable metrics incl. bits/byte

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: DVC pipeline on TinyStories

**Files:**
- Create: `scripts/fetch.sh`, `params.yaml`, `dvc.yaml`, `requirements-dataops.txt`
- Create (by `dvc init`): `.dvc/`, `.dvcignore`; (by `dvc repro`): `dvc.lock`, per-directory `.gitignore` files
- Modify: `.gitignore` (add `.venv/`)

**Interfaces:**
- Produces: `dvc repro`, which yields `data/raw/tinystories.txt`, `data/prepared/{train,val,test}.txt` + `manifest.json`, `models/slm.gguf` + `models/tokenizer.gguf`, and `metrics/{val,test}.json`.

- [ ] **Step 1: Write `scripts/fetch.sh`**

```bash
#!/usr/bin/env bash
# Download <url> to <out> and verify its sha256; fails (and leaves nothing
# behind) on a mismatch. Usage: scripts/fetch.sh <url> <sha256> <out>
set -euo pipefail
url=${1:?url}; want=${2:?sha256}; out=${3:?out}
mkdir -p "$(dirname "$out")"
tmp="$out.part"
curl -fL --retry 3 --retry-delay 2 -o "$tmp" "$url"
got=$(sha256sum "$tmp" | cut -d' ' -f1)
if [ "$got" != "$want" ]; then
  rm -f "$tmp"
  echo "sha256 mismatch for $url: got $got, want $want" >&2
  exit 1
fi
mv "$tmp" "$out"
```

`chmod +x scripts/fetch.sh`.

- [ ] **Step 2: Write `params.yaml`**

```yaml
fetch:
  url: https://huggingface.co/datasets/roneneldan/TinyStories/resolve/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/TinyStoriesV2-GPT4-valid.txt
  sha256: 6874bae9a4c1a4e7edcf0e53b86c17817e9cf881fc75ff2368da457b80c0585d
prepare:
  min_chars: 50
  max_chars: 20000
  val_frac: 0.05
  test_frac: 0.05
  seed: 42
model:            # 507,552 parameters (all dims multiples of 32 → Q8_0-friendly)
  context: 128
  embed: 96
  heads: 4
  blocks: 3
  hidden: 320
  vocab: 1024
train:
  tokens: 2.0e6
  lr: 0.005
  warmup: 50
  clip: 1.0
  val: 0.02
  patience: 3
  seed: 1337
```

- [ ] **Step 3: Write `dvc.yaml`**

```yaml
stages:
  fetch:
    cmd: scripts/fetch.sh ${fetch.url} ${fetch.sha256} data/raw/tinystories.txt
    deps:
      - scripts/fetch.sh
    params:
      - fetch
    outs:
      - data/raw/tinystories.txt
  prepare:
    cmd: >-
      cargo run --release -q -p slm_cli -- prepare
      --input data/raw/tinystories.txt --out data/prepared
      --min-chars ${prepare.min_chars} --max-chars ${prepare.max_chars}
      --val-frac ${prepare.val_frac} --test-frac ${prepare.test_frac}
      --seed ${prepare.seed} --dedup --nfkc
    deps:
      - data/raw/tinystories.txt
      - ferrum_core/src
      - slm_cli/src
    params:
      - prepare
    outs:
      - data/prepared/train.txt
      - data/prepared/val.txt
      - data/prepared/test.txt
      - data/prepared/manifest.json:
          cache: false
  train:
    cmd: >-
      cargo run --release -q -p slm_cli -- train data/prepared/train.txt models/slm.gguf
      --context ${model.context} --embed ${model.embed} --heads ${model.heads}
      --blocks ${model.blocks} --hidden ${model.hidden} --vocab ${model.vocab}
      --tokens ${train.tokens} --lr ${train.lr} --cosine --warmup ${train.warmup}
      --clip ${train.clip} --val ${train.val} --patience ${train.patience}
      --seed ${train.seed}
      --tokenizer models/tokenizer.gguf --force
    deps:
      - data/prepared/train.txt
      - ferrum_core/src
      - slm_cli/src
    params:
      - model
      - train
    outs:
      - models/slm.gguf
      - models/tokenizer.gguf
  eval:
    cmd: >-
      cargo run --release -q -p slm_cli -- eval models/slm.gguf data/prepared/val.txt --json metrics/val.json &&
      cargo run --release -q -p slm_cli -- eval models/slm.gguf data/prepared/test.txt --json metrics/test.json
    deps:
      - models/slm.gguf
      - data/prepared/val.txt
      - data/prepared/test.txt
    metrics:
      - metrics/val.json:
          cache: false
      - metrics/test.json:
          cache: false
  lighteval:
    frozen: true
    cmd: eval/lighteval/run.sh --model models/slm.gguf --test data/prepared/test.txt --out metrics/lighteval.json
    deps:
      - models/slm.gguf
      - data/prepared/test.txt
      - eval/lighteval
    metrics:
      - metrics/lighteval.json:
          cache: false
```

`requirements-dataops.txt`:

```
dvc==3.67.1
```

Append `.venv/` to `.gitignore` under "Build artefacts".

- [ ] **Step 4: Install DVC, initialise, reproduce**

```bash
python3 -m venv .venv && .venv/bin/pip install -q -r requirements-dataops.txt
.venv/bin/dvc init
.venv/bin/dvc repro 2>&1 | tail -30
```

Expected:
- `fetch` downloads 22,502,601 bytes and the checksum passes.
- `prepare` prints the split counts.
- `train` runs to its 2e6-token budget. With the ~10K tok/s review baseline that is about 3–4 minutes; Phase 1A's speed work may shorten it.
- `eval` writes both metrics files.
- `lighteval` is skipped as frozen. If DVC reports a missing dependency `eval/lighteval` for the frozen stage, create `eval/lighteval/.gitkeep` (Phase 1C fills the directory).

Then check reproducibility and the DVC workflow:

```bash
.venv/bin/dvc status                     # "Data and pipelines are up to date."
cat metrics/test.json data/prepared/manifest.json | head -40
sha256sum models/slm.gguf > /tmp/claude-1000/-home-thomas-Projects-ferrum-cpu/88623770-ced2-4a37-be7f-75cb7f78b99e/scratchpad/slm1.sha
.venv/bin/dvc repro --force train
sha256sum -c /tmp/claude-1000/-home-thomas-Projects-ferrum-cpu/88623770-ced2-4a37-be7f-75cb7f78b99e/scratchpad/slm1.sha   # must print OK (deterministic training)
.venv/bin/dvc params diff && .venv/bin/dvc metrics show
```

Expected: `sha256sum -c` prints `models/slm.gguf: OK`. If it does not, training is non-deterministic: stop and report.

- [ ] **Step 5: Commit**

```bash
git add scripts/fetch.sh params.yaml dvc.yaml dvc.lock requirements-dataops.txt .dvc .dvcignore .gitignore \
        data/.gitignore data/raw/.gitignore data/prepared/.gitignore data/prepared/manifest.json models/.gitignore metrics
git status --short   # nothing under data/ or models/ except .gitignore and manifest.json may be staged
git commit -m "feat(dataops): DVC pipeline fetch → prepare → train → eval on TinyStories

<paste the metrics/test.json perplexity and the prepare split counts>

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

(Adjust the `git add` list to the `.gitignore` files DVC actually created: `git status --short` shows them.)

---

### Task 5: Phase 1B gate

- [ ] Full battery from the Phase 0 plan's Task 6 Step 1. All green.
- [ ] `.venv/bin/dvc status` → up to date.
- [ ] Report to the controller:
  - split counts and rejected counts from `manifest.json`
  - `metrics/{val,test}.json`
  - wall time per stage
