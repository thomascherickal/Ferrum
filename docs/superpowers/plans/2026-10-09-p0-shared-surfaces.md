# Phase 0 — Shared Surfaces Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **Runs after `2026-10-09-p0g-gguf-only.md`** (models are GPT-2-compatible GGUF; FFN is GELU; the LM head has no bias).

**Goal:** Land the interfaces every later workstream depends on: a public scoring API, `evaluate()` rebuilt on it, EOS document boundaries, a token-budgeted multi-stop generation API, and a strict CLI parser.

**Architecture:** All new library code lives in `ferrum_core/src/slm.rs` (scoring + generation on `GenerativeSLM`) and `ferrum_core/src/tokenizer.rs` (document separator). Scoring batches fixed-size padded windows through the existing `Sequential::forward`, whose output for a `[B, context_len]` id matrix is `[B·context_len, vocab]` softmax rows. The CLI parser in `slm_cli/src/main.rs` becomes table-driven per subcommand.

**Tech Stack:** Rust 2021, std only for this phase.

**Spec:** `docs/superpowers/specs/2026-10-09-dataops-serve-eval-design.md` (§4.1–§4.3; `generate_tokens` from §4.4 is pulled forward here because `slm_server` needs it).

## Global Constraints

- `#![forbid(unsafe_code)]` stays in `ferrum_core`.
- Results bit-for-bit identical at `FERRUM_NUM_THREADS=1` vs N.
- `ferrum_core` must build for `wasm32-unknown-unknown` (`cargo build -p slm_wasm --target wasm32-unknown-unknown --release`).
- All files are GGUF (Phase 0-G); `TransformerConfig` only grows via defaulted fields.
- Every feature lands with its unit tests in the same commit (test first). No `unsafe` code (`#![forbid(unsafe_code)]` in every crate).
- Every commit builds and passes `cargo test --workspace --release`, `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`.
- `ferrum_gui` is outside the workspace; after touching public `ferrum_core` APIs also run `cd ferrum_gui && cargo check --all-targets`.
- Work on branch `feat/dataops-serve-eval` (already exists).

## Spec refinements made while planning (call out in final report)

- `score_rolling` scores **every** token, conditioned on a leading BOS (EOS id when the tokenizer has one, else id 0). `evaluate()` therefore reports `num_predictions == token count`.
- The new token-budget, multi-stop generator is a **new** method `generate_tokens`; the existing `generate_until(seed, max_chars, stop: &str, …)` keeps its signature because the GUI and tests call it.
- EOS is registered only when the training corpus contains `<|endoftext|>` (keeps every existing training run and test bit-identical).

## Review Focus

1. Context ending in whitespace (`"The cat "` + `"sat"`) — the space must be scored as part of the continuation, not lost or double-counted. → Task 1 test `score_moves_trailing_whitespace_into_continuation`.
2. Text much longer than `context_len` (e.g. 10× window) — every token scored exactly once. → Task 1 test `rolling_scores_each_token_once_across_windows`.
3. Empty context / empty continuation / empty text — no panic; empty continuation scores `0.0` with `n_tokens = 0`. → Task 1 test `score_edge_cases_do_not_panic`.
4. A prompt longer than the context window passed to `generate_tokens` — keep the most recent tokens, never panic. → Task 4 test `generate_tokens_accepts_prompt_longer_than_context`.
5. A flag given twice (`--stop a --stop b`) and a value that looks like a flag (`--temp -1`) — repeated values are kept in order; negative numbers parse as values. → Task 5 tests `repeated_flags_keep_all_values`, `negative_number_is_a_value`.

---

### Task 1: Public scoring API (`Score`, `score`, `score_rolling`)

**Files:**
- Modify: `ferrum_core/src/slm.rs` (add `Score`, `GenerativeSLM::{score, score_rolling, encode_text, bos_id, token_logprobs}`; tests in the existing `#[cfg(test)] mod tests`)
- Modify: `ferrum_core/src/lib.rs` (re-export `Score`)

**Interfaces:**
- Produces:
  ```rust
  #[derive(Debug, Clone, Copy, PartialEq)]
  pub struct Score { pub logprob: f64, pub n_tokens: usize, pub greedy: bool }
  impl GenerativeSLM {
      pub fn score(&self, context: &str, continuation: &str) -> Result<Score>;
      pub fn score_rolling(&self, text: &str) -> Result<Score>;
      /// Token ids for `text` (BPE: document-aware encode; char-level: error on unknown chars).
      pub fn encode_text(&self, text: &str) -> Result<Vec<usize>>;
      /// The conditioning token used before the first scored/generated token.
      pub fn bos_id(&self) -> usize;
  }
  ```

- [ ] **Step 1: Write the failing tests**

Add to `mod tests` in `ferrum_core/src/slm.rs`:

```rust
    /// A tiny trained BPE transformer SLM for scoring tests (context 8).
    fn tiny_bpe_slm() -> GenerativeSLM {
        let corpus = "the cat sat on the mat. the dog sat on the log. ".repeat(20);
        let cfg = TransformerConfig {
            context_len: 8,
            embed_dim: 16,
            num_heads: 2,
            num_blocks: 1,
            hidden_dim: 32,
            epochs: 2,
            vocab_size: 270,
            ..TransformerConfig::default()
        };
        GenerativeSLM::train_transformer_config(&corpus, &cfg, &mut Rng::new(3), |_, _| {}).unwrap()
    }

    /// Reference: score each target with its own full forward over the
    /// preceding (≤ context_len) tokens, right-padded, reading the row of the
    /// last real token. Slow but obviously correct.
    fn naive_logprobs(slm: &GenerativeSLM, ids: &[usize], from: usize) -> Vec<(f64, bool)> {
        let t = slm.meta.input_dim;
        (from..ids.len())
            .map(|i| {
                let start = i.saturating_sub(t);
                let ctx = &ids[start..i];
                let mut row: Vec<f32> = ctx.iter().map(|&x| x as f32).collect();
                row.resize(t, 0.0);
                let out = slm.model.forward(&Tensor::matrix(1, t, row).unwrap()).unwrap();
                let v = out.matrix_dims().unwrap().1;
                let r = &out.data[(ctx.len() - 1) * v..ctx.len() * v];
                let p = r[ids[i]].max(1e-12) as f64;
                let argmax = (0..v).max_by(|&a, &b| r[a].total_cmp(&r[b])).unwrap();
                (p.ln(), argmax == ids[i])
            })
            .collect()
    }

    #[test]
    fn score_matches_naive_reference() {
        let slm = tiny_bpe_slm();
        let ctx = "the cat sat on";
        let cont = " the mat.";
        let s = slm.score(ctx, cont).unwrap();
        let ctx_ids = slm.encode_text(ctx).unwrap();
        let all = slm.encode_text(&format!("{ctx}{cont}")).unwrap();
        let want = naive_logprobs(&slm, &all, ctx_ids.len());
        let want_lp: f64 = want.iter().map(|w| w.0).sum();
        assert_eq!(s.n_tokens, all.len() - ctx_ids.len());
        assert!((s.logprob - want_lp).abs() < 1e-4, "{} vs {want_lp}", s.logprob);
        assert_eq!(s.greedy, want.iter().all(|w| w.1));
        assert!(s.logprob < 0.0);
    }

    #[test]
    fn score_moves_trailing_whitespace_into_continuation() {
        let slm = tiny_bpe_slm();
        let a = slm.score("the cat ", "sat").unwrap();
        let b = slm.score("the cat", " sat").unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn rolling_scores_each_token_once_across_windows() {
        let slm = tiny_bpe_slm();
        let text = "the dog sat on the log. the cat sat on the mat. ".repeat(4);
        let ids = slm.encode_text(&text).unwrap();
        assert!(ids.len() > 5 * slm.meta.input_dim, "text must span many windows");
        let s = slm.score_rolling(&text).unwrap();
        assert_eq!(s.n_tokens, ids.len());
        let mut with_bos = vec![slm.bos_id()];
        with_bos.extend(&ids);
        // The reference sees at most context_len tokens of history, the windowed
        // scorer at least context_len/2, so compare per-token bounds loosely and
        // the first window exactly.
        let t = slm.meta.input_dim;
        let naive_first: f64 = naive_logprobs(&slm, &with_bos[..t + 1], 1)
            .iter()
            .map(|w| w.0)
            .sum();
        let first = slm.score_rolling_prefix_for_test(&with_bos[..t + 1]);
        assert!((first - naive_first).abs() < 1e-4);
        assert!(s.logprob.is_finite() && s.logprob < 0.0);
    }

    #[test]
    fn score_edge_cases_do_not_panic() {
        let slm = tiny_bpe_slm();
        let empty_cont = slm.score("the cat", "").unwrap();
        assert_eq!(empty_cont, Score { logprob: 0.0, n_tokens: 0, greedy: true });
        let empty_ctx = slm.score("", "the cat").unwrap();
        assert!(empty_ctx.n_tokens > 0 && empty_ctx.logprob < 0.0);
        let empty_text = slm.score_rolling("").unwrap();
        assert_eq!(empty_text.n_tokens, 0);
    }

    #[test]
    fn char_level_score_rejects_unknown_chars() {
        let corpus = "abcabcabc abc\n".repeat(20);
        let cfg = TransformerConfig {
            context_len: 8,
            embed_dim: 8,
            num_heads: 2,
            num_blocks: 1,
            hidden_dim: 16,
            epochs: 1,
            vocab_size: 0,
            ..TransformerConfig::default()
        };
        let slm =
            GenerativeSLM::train_transformer_config(&corpus, &cfg, &mut Rng::new(1), |_, _| {})
                .unwrap();
        assert!(slm.score("abc", "abc").is_ok());
        let err = slm.score("abc", "xyz").unwrap_err().to_string();
        assert!(err.contains("not in the model vocabulary"), "{err}");
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p ferrum_core --release --lib slm::tests::score 2>&1 | tail -5`
Expected: compile error — `no method named score`, `cannot find type Score`.

- [ ] **Step 3: Implement**

Add after the `Evaluation` struct in `slm.rs`:

```rust
/// Loglikelihood of a token span under the model ([`GenerativeSLM::score`],
/// [`GenerativeSLM::score_rolling`]).
///
/// Per-token probabilities come from the model's softmax output, clamped at
/// `1e-12`, so one token contributes at least `ln(1e-12) ≈ −27.63`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Score {
    /// Sum of natural-log probabilities of the scored tokens.
    pub logprob: f64,
    /// Number of tokens scored.
    pub n_tokens: usize,
    /// True when every scored token was the model's argmax prediction.
    pub greedy: bool,
}

/// Windows scored per batched forward pass in [`GenerativeSLM::token_logprobs`].
const SCORE_BATCH: usize = 32;
```

Add inside `impl GenerativeSLM` (after `evaluate`):

```rust
    /// The conditioning token placed before the first scored or generated
    /// token: the tokenizer's EOS id when it has one (document boundary), else 0.
    pub fn bos_id(&self) -> usize {
        if self.meta.tokenizer_state.is_empty() {
            return 0;
        }
        ByteBpeTokenizer::from_state(&self.meta.tokenizer_state)
            .ok()
            .and_then(|t| t.eos_id())
            .unwrap_or(0)
    }

    /// Token ids for `text`. BPE models use the document-aware encoder;
    /// character-level models map each char (CR dropped) and error on a char
    /// outside the model vocabulary instead of silently using id 0.
    pub fn encode_text(&self, text: &str) -> Result<Vec<usize>> {
        if !self.meta.tokenizer_state.is_empty() {
            let tok = ByteBpeTokenizer::from_state(&self.meta.tokenizer_state)?;
            return Ok(tok.encode_docs(text));
        }
        self.require_token_id_model()?;
        let index: std::collections::HashMap<&str, usize> = self
            .meta
            .class_names
            .iter()
            .enumerate()
            .map(|(i, s)| (s.as_str(), i))
            .collect();
        text.chars()
            .filter(|&c| c != '\r')
            .map(|ch| {
                index.get(char_to_hex(ch).as_str()).copied().ok_or_else(|| {
                    InferError::DimMismatch(format!(
                        "character {ch:?} is not in the model vocabulary"
                    ))
                })
            })
            .collect()
    }

    /// log p(`continuation` | `context`) using the lm-eval-harness convention:
    /// trailing whitespace of `context` moves to the front of `continuation`;
    /// continuation tokens are the suffix of `encode(context + continuation)`
    /// after `encode(context)` (falling back to separate encodes when the joint
    /// encoding does not start with the context's). An empty context is scored
    /// after [`GenerativeSLM::bos_id`].
    pub fn score(&self, context: &str, continuation: &str) -> Result<Score> {
        let trimmed = context.trim_end();
        let (context, continuation) = if trimmed.len() < context.len() {
            (trimmed, format!("{}{continuation}", &context[trimmed.len()..]))
        } else {
            (context, continuation.to_string())
        };
        if continuation.is_empty() {
            return Ok(Score { logprob: 0.0, n_tokens: 0, greedy: true });
        }
        let ctx_ids = if context.is_empty() {
            vec![self.bos_id()]
        } else {
            self.encode_text(context)?
        };
        let joint = if context.is_empty() {
            None
        } else {
            Some(self.encode_text(&format!("{context}{continuation}"))?)
        };
        let ids = match joint {
            Some(j) if j.len() > ctx_ids.len() && j.starts_with(&ctx_ids) => j,
            _ => {
                let mut ids = ctx_ids.clone();
                ids.extend(self.encode_text(&continuation)?);
                ids
            }
        };
        self.token_logprobs(&ids, ctx_ids.len())
    }

    /// Loglikelihood of all of `text`: every token is scored, the first one
    /// conditioned on [`GenerativeSLM::bos_id`]. Long texts are scored in
    /// sliding windows (see [`GenerativeSLM::token_logprobs`]).
    pub fn score_rolling(&self, text: &str) -> Result<Score> {
        let mut ids = vec![self.bos_id()];
        ids.extend(self.encode_text(text)?);
        self.token_logprobs(&ids, 1)
    }

    /// Sum log p(ids[i] | ids[..i]) for `i in from..ids.len()` (`from >= 1`).
    ///
    /// Transformer models: the targets are covered by windows of `context_len`
    /// tokens. The first window starts at 0; each later window starts
    /// `context_len / 2` tokens before its first new target, so every target is
    /// scored exactly once with at least half a window of history. Windows are
    /// right-padded with id 0 (safe under the causal mask) and batched
    /// [`SCORE_BATCH`] at a time through one forward pass. Models without
    /// transformer blocks fall back to one forward per target.
    fn token_logprobs(&self, ids: &[usize], from: usize) -> Result<Score> {
        let from = from.max(1);
        let mut score = Score { logprob: 0.0, n_tokens: 0, greedy: true };
        if from >= ids.len() {
            return Ok(score);
        }
        let t = self.meta.input_dim;
        let half = (t / 2).max(1);

        // (window start, first target, end target exclusive)
        let mut windows: Vec<(usize, usize, usize)> = Vec::new();
        let mut next = from;
        while next < ids.len() {
            let start = if next <= t { 0 } else { next - half };
            let end = (start + t + 1).min(ids.len());
            windows.push((start, next, end));
            next = end;
        }

        let transformer = CachedTransformer::try_new(&self.model).is_some();
        let mut add = |row: &[f32], target: usize| {
            let p = row[target].max(1e-12) as f64;
            let argmax = (0..row.len()).max_by(|&a, &b| row[a].total_cmp(&row[b])).unwrap_or(0);
            score.logprob += p.ln();
            score.n_tokens += 1;
            score.greedy &= argmax == target;
        };

        if transformer {
            for chunk in windows.chunks(SCORE_BATCH) {
                let mut input = Vec::with_capacity(chunk.len() * t);
                for &(start, _, _) in chunk {
                    let w = &ids[start..(start + t).min(ids.len())];
                    input.extend(w.iter().map(|&x| x as f32));
                    input.resize(input.len() + (t - w.len()), 0.0);
                }
                let out = self.model.forward(&Tensor::matrix(chunk.len(), t, input)?)?;
                let v = out.matrix_dims()?.1;
                for (b, &(start, first, end)) in chunk.iter().enumerate() {
                    for target in first..end {
                        let r = b * t + (target - 1 - start);
                        add(&out.data[r * v..(r + 1) * v], ids[target]);
                    }
                }
            }
        } else {
            for target in from..ids.len() {
                let start = target.saturating_sub(t);
                let mut ctx: Vec<f32> = vec![0.0; t - (target - start)];
                ctx.extend(ids[start..target].iter().map(|&x| x as f32));
                let out = self.model.forward(&Tensor::matrix(1, t, ctx)?)?;
                let (rows, v) = out.matrix_dims()?;
                add(&out.data[(rows - 1) * v..rows * v], ids[target]);
            }
        }
        Ok(score)
    }

    #[cfg(test)]
    fn score_rolling_prefix_for_test(&self, ids: &[usize]) -> f64 {
        self.token_logprobs(ids, 1).unwrap().logprob
    }
```

`encode_docs` does not exist until Task 3; for this task add a temporary forwarding method in `tokenizer.rs` so the crate compiles (Task 3 replaces its body):

```rust
    /// Encode text that may contain document separators (see [`DOC_SEP`]).
    pub fn encode_docs(&self, text: &str) -> Vec<usize> {
        self.encode(text)
    }
```

In `lib.rs` add `Score` to the `pub use slm::{ … }` list.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p ferrum_core --release --lib slm::tests 2>&1 | tail -5`
Expected: all `slm::tests` pass, including the 5 new ones.

- [ ] **Step 5: Commit**

```bash
cargo fmt --all && cargo clippy --workspace --all-targets -- -D warnings
git add ferrum_core/src/slm.rs ferrum_core/src/lib.rs ferrum_core/src/tokenizer.rs
git commit -m "feat(slm): public score / score_rolling loglikelihood API

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Rebuild `evaluate()` (and `--val`) on `score_rolling`

**Files:**
- Modify: `ferrum_core/src/slm.rs:1241-1325` (`evaluate`, `score_token_ids`, `finish_evaluation`, `Evaluation`)
- Test: `ferrum_core/src/slm.rs` tests; existing `tests/tests/test_slm_library.rs` assertions on `num_predictions`

**Interfaces:**
- Consumes: `GenerativeSLM::score_rolling` (Task 1).
- Produces: `Evaluation` unchanged in shape; `num_predictions` now equals the text's token count; texts of any length ≥ 1 token are accepted.

- [ ] **Step 1: Write the failing test**

```rust
    #[test]
    fn evaluate_agrees_with_score_rolling_and_accepts_short_text() {
        let slm = tiny_bpe_slm();
        let text = "the cat sat on the log.";
        let e = slm.evaluate(text).unwrap();
        let s = slm.score_rolling(text).unwrap();
        assert_eq!(e.num_predictions, s.n_tokens);
        let ce = (-s.logprob / s.n_tokens as f64) as f32;
        assert!((e.cross_entropy - ce).abs() < 1e-5);
        assert!((e.perplexity - ce.exp()).abs() < 1e-3);
        assert!(slm.evaluate("").is_err());
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p ferrum_core --release --lib evaluate_agrees 2>&1 | tail -5`
Expected: FAIL — `num_predictions` differs (old path skips the first window) or the short text errors.

- [ ] **Step 3: Implement**

Replace the body of `evaluate` and delete `score_token_ids`:

```rust
    pub fn evaluate(&self, text: &str) -> Result<Evaluation> {
        vprintln!(
            "[slm::evaluate] scoring {} chars of held-out text",
            text.chars().count()
        );
        let s = self.score_rolling(text)?;
        if s.n_tokens == 0 {
            return Err(InferError::DimMismatch(
                "evaluation text must contain at least one token".into(),
            ));
        }
        Ok(finish_evaluation(-s.logprob, s.n_tokens))
    }
```

Update the doc comments: `Evaluation::num_predictions` → "Number of next-token predictions scored (every token of the text, the first conditioned on the BOS token)."; `evaluate` doc → replace the "past the first `context_len` tokens" and "`text` must contain at least `context_len + 1` tokens" sentences with "Every token is scored; long texts use sliding windows with at least `context_len / 2` tokens of history (see [`GenerativeSLM::score_rolling`]). Character-level models return an error for characters outside the vocabulary."

Then fix callers' expectations:

Run: `grep -rn "num_predictions\|context_len + 1\|more than context_len" --include=*.rs ferrum_core tests slm_cli slm_wasm ferrum_gui/src`
For each test asserting `num_predictions == len - context_len`, change the expected value to the token count; for each test asserting `evaluate` errors on short text, change it to assert success on a 1-token text and error on `""`.

- [ ] **Step 4: Run the full suite**

Run: `cargo test --workspace --release 2>&1 | grep -E "^test result|FAILED|panicked" | head -30`
Expected: every `test result: ok`. Then `cd ferrum_gui && cargo test 2>&1 | grep "test result"` → ok.

- [ ] **Step 5: Commit**

```bash
cargo fmt --all && cargo clippy --workspace --all-targets -- -D warnings
git add -A ferrum_core tests ferrum_gui/src
git commit -m "perf(slm): evaluate() and --val on batched score_rolling (~100x faster)

Every token is now scored (previously the first context window was skipped),
so perplexities shift slightly; recorded for Status §7.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: EOS document separator in the tokenizer and trainer

**Files:**
- Modify: `ferrum_core/src/tokenizer.rs` (add `DOC_SEP`, real `encode_docs`, `train_with_docs`)
- Modify: `ferrum_core/src/slm.rs` (`tokenize_for_lm`, `tokenize_for_lm_cfg`, `lm_tokens_from_bpe`, BPE generation's `tok.encode(seed)`)
- Modify: `ferrum_core/src/lib.rs` (re-export `DOC_SEP`)

**Interfaces:**
- Produces:
  ```rust
  pub const DOC_SEP: &str = "<|endoftext|>";
  impl ByteBpeTokenizer {
      /// Like `encode`, but each `DOC_SEP` becomes the EOS id when the tokenizer has one.
      pub fn encode_docs(&self, text: &str) -> Vec<usize>;
      /// Train on `corpus`; when it contains `DOC_SEP`, train merges on the text
      /// without separators to `vocab_size - 1` and register `TOK_EOS`, so the
      /// total vocabulary is still `vocab_size`.
      pub fn train_with_docs(corpus: &str, vocab_size: usize) -> Result<Self>;
  }
  ```

- [ ] **Step 1: Write the failing tests** (in `tokenizer.rs` tests)

```rust
    #[test]
    fn doc_separator_becomes_eos_when_registered() {
        let corpus = "one story.<|endoftext|>two story.<|endoftext|>three".repeat(10);
        let tok = ByteBpeTokenizer::train_with_docs(&corpus, 300).unwrap();
        assert_eq!(tok.vocab_size(), 300);
        let eos = tok.eos_id().expect("EOS registered");
        let ids = tok.encode_docs("a.<|endoftext|>b.");
        assert_eq!(ids.iter().filter(|&&i| i == eos).count(), 1);
        assert_eq!(tok.decode(&ids), "a.b.");
        // No separator in the corpus → identical to plain training.
        let plain = ByteBpeTokenizer::train("abc abc abc", 260).unwrap();
        let docs = ByteBpeTokenizer::train_with_docs("abc abc abc", 260).unwrap();
        assert_eq!(plain.encode_state(), docs.encode_state());
        assert_eq!(docs.encode_docs("x<|endoftext|>y"), docs.encode("x<|endoftext|>y"));
    }
```

And in `slm.rs` tests:

```rust
    #[test]
    fn training_on_docs_registers_eos_and_bos_uses_it() {
        let corpus = "the cat sat.<|endoftext|>the dog ran.<|endoftext|>".repeat(30);
        let cfg = TransformerConfig {
            context_len: 8, embed_dim: 8, num_heads: 2, num_blocks: 1, hidden_dim: 16,
            epochs: 1, vocab_size: 280, ..TransformerConfig::default()
        };
        let slm = GenerativeSLM::train_transformer_config(&corpus, &cfg, &mut Rng::new(2), |_, _| {}).unwrap();
        let tok = ByteBpeTokenizer::from_state(&slm.meta.tokenizer_state).unwrap();
        assert_eq!(slm.bos_id(), tok.eos_id().unwrap());
        assert_eq!(slm.meta.output_dim, 280);
    }
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p ferrum_core --release --lib doc_separator training_on_docs 2>&1 | tail -5`
Expected: compile error — `train_with_docs` not found.

- [ ] **Step 3: Implement**

In `tokenizer.rs`, near the `TOK_*` constants:

```rust
/// Document separator used by prepared corpora (`train_transformer prepare`)
/// and TinyStories. Tokenizers trained with [`ByteBpeTokenizer::train_with_docs`]
/// on text containing it map it to the EOS special token.
pub const DOC_SEP: &str = "<|endoftext|>";
```

Replace the temporary `encode_docs` and add `train_with_docs`:

```rust
    /// Like [`encode`](Self::encode), but every [`DOC_SEP`] becomes the EOS id
    /// when this tokenizer has one. Without EOS it is exactly `encode`.
    pub fn encode_docs(&self, text: &str) -> Vec<usize> {
        let Some(eos) = self.eos_id() else {
            return self.encode(text);
        };
        let mut ids = Vec::new();
        for (i, doc) in text.split(DOC_SEP).enumerate() {
            if i > 0 {
                ids.push(eos);
            }
            ids.extend(self.encode(doc));
        }
        ids
    }

    /// [`train`](Self::train) that understands document separators: when
    /// `corpus` contains [`DOC_SEP`], merges are learned on the text without
    /// separators to `vocab_size - 1` and [`TOK_EOS`] is registered, so the
    /// total vocabulary is still `vocab_size`. Otherwise identical to `train`.
    pub fn train_with_docs(corpus: &str, vocab_size: usize) -> Result<Self> {
        if !corpus.contains(DOC_SEP) {
            return Self::train(corpus, vocab_size);
        }
        let mut tok = Self::train(&corpus.replace(DOC_SEP, ""), vocab_size - 1)?;
        tok.add_special_token(TOK_EOS)?;
        Ok(tok)
    }
```

In `slm.rs`: in `tokenize_for_lm` and `tokenize_for_lm_cfg` replace `ByteBpeTokenizer::train(` with `ByteBpeTokenizer::train_with_docs(`; in `lm_tokens_from_bpe` replace `tok.encode(corpus)` with `tok.encode_docs(corpus)`. Do **not** touch `generate_bpe_stream` (it trims its output by decoding the seed, and EOS decodes to nothing; Phase 1 A rewrites that path). In `lib.rs` add `DOC_SEP` to the `pub use tokenizer::{…}` list.

- [ ] **Step 4: Run the full suite**

Run: `cargo test --workspace --release 2>&1 | grep -E "^test result|FAILED" | head -30`
Expected: all ok (no existing corpus contains `<|endoftext|>`, so existing runs are unchanged).

- [ ] **Step 5: Commit**

```bash
cargo fmt --all && cargo clippy --workspace --all-targets -- -D warnings
git add ferrum_core/src/tokenizer.rs ferrum_core/src/slm.rs ferrum_core/src/lib.rs
git commit -m "feat(tokenizer): <|endoftext|> document separator maps to EOS

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: `generate_tokens` — token budget, multi-stop, EOS, greedy

**Files:**
- Modify: `ferrum_core/src/slm.rs` (add `Generation`, `StopReason`, `generate_tokens`; greedy branch in `sample_with_params`)
- Modify: `ferrum_core/src/lib.rs` (re-export `Generation`, `StopReason`)

**Interfaces:**
- Consumes: `encode_text`, `bos_id` (Task 1), `encode_docs`/`eos_id` (Task 3), private `CachedTransformer::{try_new, prime, feed, capacity}`.
- Produces:
  ```rust
  #[derive(Debug, Clone, Copy, PartialEq, Eq)]
  pub enum StopReason { MaxTokens, Stop, Eos }
  #[derive(Debug, Clone, PartialEq)]
  pub struct Generation { pub text: String, pub n_tokens: usize, pub stop_reason: StopReason }
  impl GenerativeSLM {
      /// Continuation only (prompt excluded). `params.temperature <= 0` = argmax.
      pub fn generate_tokens(&self, prompt: &str, max_tokens: usize, stops: &[String],
                             params: &SamplingParams, rng: &mut Rng) -> Result<Generation>;
  }
  ```

- [ ] **Step 1: Write the failing tests**

```rust
    #[test]
    fn generate_tokens_respects_budget_and_is_deterministic() {
        let slm = tiny_bpe_slm();
        let p = SamplingParams { temperature: 0.0, ..SamplingParams::default() };
        let a = slm.generate_tokens("the cat", 12, &[], &p, &mut Rng::new(1)).unwrap();
        let b = slm.generate_tokens("the cat", 12, &[], &p, &mut Rng::new(99)).unwrap();
        assert_eq!(a, b, "temperature 0 is argmax: the seed must not matter");
        assert_eq!(a.n_tokens, 12);
        assert_eq!(a.stop_reason, StopReason::MaxTokens);
    }

    #[test]
    fn generate_tokens_stops_at_first_stop_and_excludes_it() {
        let slm = tiny_bpe_slm();
        let p = SamplingParams { temperature: 0.0, ..SamplingParams::default() };
        let free = slm.generate_tokens("the cat", 40, &[], &p, &mut Rng::new(1)).unwrap();
        // Pick a stop string that greedy decoding is known to produce.
        let stop = free.text.chars().skip(3).take(2).collect::<String>();
        let g = slm
            .generate_tokens("the cat", 40, &["#never#".into(), stop.clone()], &p, &mut Rng::new(1))
            .unwrap();
        assert_eq!(g.stop_reason, StopReason::Stop);
        assert!(!g.text.contains(&stop), "stop string must be excluded: {:?}", g.text);
        assert!(free.text.starts_with(&g.text));
    }

    #[test]
    fn generate_tokens_accepts_prompt_longer_than_context() {
        let slm = tiny_bpe_slm();
        let long = "the dog sat on the log. ".repeat(10);
        let p = SamplingParams { temperature: 0.0, ..SamplingParams::default() };
        let g = slm.generate_tokens(&long, 30, &[], &p, &mut Rng::new(1)).unwrap();
        assert_eq!(g.n_tokens, 30);
        let empty = slm.generate_tokens("", 5, &[], &p, &mut Rng::new(1)).unwrap();
        assert_eq!(empty.n_tokens, 5);
    }

    #[test]
    fn zero_temperature_is_argmax() {
        let logits = [0.1f32, 2.0, 1.9, -1.0];
        let p = SamplingParams { temperature: 0.0, ..SamplingParams::default() };
        for s in 0..20 {
            assert_eq!(sample_with_params(&logits, &p, &[], &mut Rng::new(s)), 1);
        }
    }
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p ferrum_core --release --lib generate_tokens zero_temperature 2>&1 | tail -5`
Expected: compile error (`generate_tokens`, `StopReason` missing).

- [ ] **Step 3: Implement**

At the top of `sample_with_params`, after the repetition-penalty block moves first, add the greedy branch. Restructure as: compute `work` with the repetition penalty (existing step 1), then:

```rust
    // Temperature ≤ 0 is exact argmax (greedy decoding); no RNG draw.
    if params.temperature <= 0.0 {
        return (0..n).max_by(|&a, &b| work[a].total_cmp(&work[b])).unwrap_or(0);
    }
    let t = params.temperature;
```

(and delete the old `let t = params.temperature.max(0.01);`). Update the `SamplingParams` doc: "`temperature <= 0` selects greedy argmax decoding."

Add types after `SamplingParams`:

```rust
/// Why [`GenerativeSLM::generate_tokens`] stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// `max_tokens` tokens were generated.
    MaxTokens,
    /// A stop string appeared (it is excluded from the text).
    Stop,
    /// The model emitted the EOS token.
    Eos,
}

/// Output of [`GenerativeSLM::generate_tokens`].
#[derive(Debug, Clone, PartialEq)]
pub struct Generation {
    /// The continuation only (prompt excluded, stop string excluded).
    pub text: String,
    /// Tokens generated (including the one that completed a stop string; EOS
    /// itself is not counted).
    pub n_tokens: usize,
    pub stop_reason: StopReason,
}
```

Add the method in `impl GenerativeSLM`:

```rust
    /// Generate up to `max_tokens` tokens after `prompt`, returning only the
    /// continuation. Stops early at the first of `stops` to appear (excluded
    /// from the text) or when the model emits EOS. Any prompt length works: a
    /// prompt longer than the context keeps its most recent tokens; an empty
    /// prompt starts from [`GenerativeSLM::bos_id`].
    ///
    /// When the KV cache fills, it is re-primed from the most recent
    /// `context_len / 2` tokens, so the cost is one re-prime per
    /// `context_len / 2` generated tokens rather than one per token (learned
    /// absolute positions rule out sliding the cache by one).
    pub fn generate_tokens(
        &self,
        prompt: &str,
        max_tokens: usize,
        stops: &[String],
        params: &SamplingParams,
        rng: &mut Rng,
    ) -> Result<Generation> {
        self.require_token_id_model()?;
        let bpe = if self.meta.tokenizer_state.is_empty() {
            None
        } else {
            Some(ByteBpeTokenizer::from_state(&self.meta.tokenizer_state)?)
        };
        let eos = bpe.as_ref().and_then(|t| t.eos_id());
        let mut cached = CachedTransformer::try_new(&self.model).ok_or_else(|| {
            InferError::DimMismatch("generate_tokens needs a transformer model".into())
        })?;
        let cap = cached.capacity().max(2);
        let keep = cap / 2;

        let mut window = self.encode_text(prompt)?;
        if window.is_empty() {
            window.push(self.bos_id());
        }
        if window.len() > cap {
            window.drain(..window.len() - cap);
        }
        let mut dist = cached.prime(&window)?;

        let mut text = String::new();
        let mut pending: Vec<u8> = Vec::new();
        let max_stop = stops.iter().map(|s| s.len()).max().unwrap_or(0);
        let mut n_tokens = 0;
        let mut reason = StopReason::MaxTokens;

        while n_tokens < max_tokens {
            let logits: Vec<f32> = dist.iter().map(|&p| p.max(1e-12).ln()).collect();
            let next = sample_with_params(&logits, params, &window, rng);
            if Some(next) == eos {
                reason = StopReason::Eos;
                break;
            }
            n_tokens += 1;
            let before = text.len();
            match &bpe {
                Some(tok) => {
                    pending.extend_from_slice(tok.token_bytes(next).unwrap_or(&[]));
                    drain_utf8(&mut pending, &mut text);
                }
                None => text.push(hex_to_char(&self.meta.class_names[next])),
            }
            // Only the tail can contain a newly completed stop string.
            let from = before.saturating_sub(max_stop);
            let from = (0..=from).rev().find(|&i| text.is_char_boundary(i)).unwrap_or(0);
            if let Some(cut) = stops
                .iter()
                .filter(|s| !s.is_empty())
                .filter_map(|s| text[from..].find(s.as_str()).map(|i| from + i))
                .min()
            {
                text.truncate(cut);
                reason = StopReason::Stop;
                break;
            }
            window.push(next);
            if window.len() > cap {
                window.drain(..window.len() - keep);
                dist = cached.prime(&window)?;
            } else {
                dist = cached.feed(next)?;
            }
        }
        if reason != StopReason::Stop && !pending.is_empty() {
            text.push_str(&String::from_utf8_lossy(&pending));
        }
        Ok(Generation { text, n_tokens, stop_reason: reason })
    }
```

Add a free function near `char_to_hex`:

```rust
/// Move the longest valid UTF-8 prefix of `pending` onto `out`, replacing
/// invalid sequences with U+FFFD exactly like `String::from_utf8_lossy`, and
/// leave an incomplete trailing sequence in `pending` for the next token.
fn drain_utf8(pending: &mut Vec<u8>, out: &mut String) {
    loop {
        match std::str::from_utf8(pending) {
            Ok(s) => {
                out.push_str(s);
                pending.clear();
                return;
            }
            Err(e) => {
                let valid = e.valid_up_to();
                out.push_str(std::str::from_utf8(&pending[..valid]).unwrap_or_default());
                match e.error_len() {
                    Some(bad) => {
                        out.push('\u{FFFD}');
                        pending.drain(..valid + bad);
                    }
                    None => {
                        pending.drain(..valid);
                        return;
                    }
                }
            }
        }
    }
}
```

Re-export `Generation, StopReason` in `lib.rs`. Note: `token_bytes(id)` returns `Some(&[])` for specials.

- [ ] **Step 4: Run tests**

Run: `cargo test --workspace --release 2>&1 | grep -E "^test result|FAILED" | head -30`
Expected: all ok. If `temperature_only_params_match_legacy_sampler` fails, check it does not use temperature 0 — it should not.

- [ ] **Step 5: Commit**

```bash
cargo fmt --all && cargo clippy --workspace --all-targets -- -D warnings
git add ferrum_core/src/slm.rs ferrum_core/src/lib.rs
git commit -m "feat(slm): generate_tokens with token budget, stop list, EOS, greedy

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Strict, table-driven CLI parser

**Files:**
- Modify: `slm_cli/src/main.rs:56-188` (`Args`), `run()`, `cmd_train`/`cmd_run` (resume check)
- Create: `slm_cli/tests/cli_strict.rs`

**Interfaces:**
- Produces (used by Phase 1 A and B to add flags):
  ```rust
  #[derive(Clone, Copy, PartialEq)]
  enum Kind { Bool, Usize, U64, F32, F64, Str, StrList }
  /// (flag name without "--", kind)
  type Spec = &'static [(&'static str, Kind)];
  fn spec_for(cmd: &str) -> Option<Spec>;
  impl Args {
      fn parse(raw: &[String], spec: Spec) -> Result<Args, String>;
      fn get<T: FromStr>(&self, key: &str, default: T) -> T;   // last value
      fn get_all(&self, key: &str) -> Vec<String>;            // StrList
      fn value(&self, key: &str) -> Option<&str>;              // last raw value
      fn has(&self, key: &str) -> bool;
  }
  ```
  To add a flag to a subcommand later: add `("name", Kind::X)` to that subcommand's `const` table.

- [ ] **Step 1: Write the failing tests** — `slm_cli/tests/cli_strict.rs`:

```rust
//! The CLI must reject what it does not understand instead of silently
//! falling back to defaults (review defects #3 and #4).

use std::process::Command;

fn run(args: &[&str]) -> (i32, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_train_transformer"))
        .args(args)
        .output()
        .expect("spawn train_transformer");
    let text = String::from_utf8_lossy(&out.stderr).to_string()
        + &String::from_utf8_lossy(&out.stdout);
    (out.status.code().unwrap_or(-1), text)
}

fn corpus() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("ferrum_cli_strict_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join("c.txt");
    std::fs::write(&p, "abc abc abc\n".repeat(50)).unwrap();
    p
}

#[test]
fn unknown_flag_is_an_error() {
    let c = corpus();
    let (code, text) = run(&["train", c.to_str().unwrap(), "/nonexistent/m.bin", "--contxt", "8"]);
    assert_eq!(code, 1, "{text}");
    assert!(text.contains("unknown flag --contxt"), "{text}");
}

#[test]
fn unparsable_value_is_an_error() {
    let c = corpus();
    let (code, text) = run(&["train", c.to_str().unwrap(), "/nonexistent/m.bin", "--context", "12x"]);
    assert_eq!(code, 1, "{text}");
    assert!(text.contains("--context") && text.contains("12x"), "{text}");
}

#[test]
fn missing_value_is_an_error() {
    let (code, text) = run(&["generate", "m.bin", "hi", "--temp"]);
    assert_eq!(code, 1, "{text}");
    assert!(text.contains("--temp needs a value"), "{text}");
}

#[test]
fn missing_resume_checkpoint_is_an_error() {
    let c = corpus();
    let (code, text) = run(&[
        "train", c.to_str().unwrap(), "/nonexistent/m.bin", "--resume", "/nonexistent/nope.fckp",
    ]);
    assert_eq!(code, 1, "{text}");
    assert!(text.contains("checkpoint /nonexistent/nope.fckp does not exist"), "{text}");
}

#[test]
fn flag_valid_elsewhere_is_rejected_for_this_command() {
    let (code, text) = run(&["info", "m.bin", "--epochs", "3"]);
    assert_eq!(code, 1, "{text}");
    assert!(text.contains("unknown flag --epochs for 'info'"), "{text}");
}
```

Also add unit tests at the bottom of `main.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }
    const SPEC: Spec = &[("stop", Kind::StrList), ("temp", Kind::F32), ("stream", Kind::Bool)];

    #[test]
    fn repeated_flags_keep_all_values() {
        let a = Args::parse(&s(&["--stop", "a", "--stop", "b"]), SPEC).unwrap();
        assert_eq!(a.get_all("stop"), vec!["a", "b"]);
    }

    #[test]
    fn negative_number_is_a_value() {
        let a = Args::parse(&s(&["--temp", "-1"]), SPEC).unwrap();
        assert_eq!(a.get("temp", 0.0f32), -1.0);
    }

    #[test]
    fn bool_and_positionals_mix() {
        let a = Args::parse(&s(&["m.bin", "--stream", "hello", "-v"]), SPEC).unwrap();
        assert!(a.has("stream") && a.verbose);
        assert_eq!(a.positional, vec!["m.bin", "hello"]);
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p slm_cli --release 2>&1 | grep -E "test |error" | head -20`
Expected: compile errors (`Kind`, `Spec` missing) for unit tests.

- [ ] **Step 3: Implement**

Replace the `Args` struct, `VALUE_FLAGS`, and `Args::parse`/`get`/`has` with:

```rust
/// How a `--flag` is parsed. Values are validated at parse time, so a typo
/// like `--context 12x` is an error instead of a silent default.
#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Bool,
    Usize,
    U64,
    F32,
    /// f64, so budgets like `5e7` parse.
    F64,
    Str,
    /// Repeatable string flag (`--stop a --stop b`).
    StrList,
}

/// A subcommand's accepted flags.
type Spec = &'static [(&'static str, Kind)];

const TRAIN_FLAGS: Spec = &[
    ("context", Kind::Usize), ("embed", Kind::Usize), ("heads", Kind::Usize),
    ("blocks", Kind::Usize), ("hidden", Kind::Usize), ("epochs", Kind::Usize),
    ("lr", Kind::F32), ("batch", Kind::Usize), ("vocab", Kind::Usize),
    ("seed", Kind::U64), ("threads", Kind::Usize), ("weight_decay", Kind::F32),
    ("dropout", Kind::F32), ("stride", Kind::Usize), ("tokens", Kind::F64),
    ("cosine", Kind::Bool), ("warmup", Kind::U64), ("clip", Kind::F32),
    ("tie", Kind::Bool), ("no-qat", Kind::Bool), ("tokenizer", Kind::Str),
    ("checkpoint", Kind::Str), ("checkpoint-every", Kind::U64), ("resume", Kind::Str),
    ("val", Kind::F32), ("patience", Kind::Usize), ("force", Kind::Bool),
    ("sample", Kind::Bool),
];
const GENERATE_FLAGS: Spec = &[
    ("chars", Kind::Usize), ("temp", Kind::F32), ("gen-seed", Kind::U64),
    ("stream", Kind::Bool),
];
const RUN_GGUF_FLAGS: Spec = &[
    ("quant", Kind::Str), ("max", Kind::Usize), ("temp", Kind::F32),
    ("gen-seed", Kind::U64), ("top-k", Kind::Usize), ("top-p", Kind::F32),
    ("rep", Kind::F32), ("ids", Kind::Str), ("raw", Kind::Bool),
    ("force", Kind::Bool), ("resume", Kind::Str),
];
const FINETUNE_FLAGS: Spec = &[
    ("epochs", Kind::Usize), ("lr", Kind::F32), ("batch", Kind::Usize),
    ("seq", Kind::Usize), ("weight_decay", Kind::F32), ("dropout", Kind::F32),
    ("clip", Kind::F32), ("warmup", Kind::U64), ("qat", Kind::Bool),
    ("seed", Kind::U64), ("threads", Kind::Usize), ("resume", Kind::Str),
    ("sample", Kind::Bool), ("temp", Kind::F32), ("max", Kind::Usize),
    ("force", Kind::Bool),
];
const EXPORT_FLAGS: Spec = &[("quant", Kind::Str), ("resume", Kind::Str), ("force", Kind::Bool)];
const NO_FLAGS: Spec = &[];

/// `run` takes the union of train and generate flags.
fn run_flags() -> Vec<(&'static str, Kind)> {
    TRAIN_FLAGS.iter().chain(GENERATE_FLAGS).copied().collect()
}

fn spec_for(cmd: &str) -> Option<Spec> {
    static RUN: std::sync::OnceLock<Vec<(&'static str, Kind)>> = std::sync::OnceLock::new();
    Some(match cmd {
        "train" => TRAIN_FLAGS,
        "run" => RUN.get_or_init(run_flags).as_slice(),
        "generate" | "gen" => GENERATE_FLAGS,
        "run-gguf" | "gguf" => RUN_GGUF_FLAGS,
        "finetune-gguf" | "finetune" => FINETUNE_FLAGS,
        "export-gguf" | "export" => EXPORT_FLAGS,
        "eval" | "info" => NO_FLAGS,
        _ => return None,
    })
}

/// Parsed `positional --flag value` arguments for one subcommand.
struct Args {
    cmd: String,
    positional: Vec<String>,
    verbose: bool,
    flags: std::collections::HashMap<String, Vec<String>>,
    bools: std::collections::HashSet<String>,
}

impl Args {
    fn parse(raw: &[String], spec: Spec) -> Result<Self, String> {
        Self::parse_for("", raw, spec)
    }

    fn parse_for(cmd: &str, raw: &[String], spec: Spec) -> Result<Self, String> {
        let mut a = Args {
            cmd: cmd.to_string(),
            positional: Vec::new(),
            verbose: false,
            flags: Default::default(),
            bools: Default::default(),
        };
        let mut i = 0;
        while i < raw.len() {
            let arg = &raw[i];
            if arg == "--verbose" || arg == "-v" {
                a.verbose = true;
            } else if let Some(name) = arg.strip_prefix("--") {
                let kind = spec
                    .iter()
                    .find(|(n, _)| *n == name)
                    .map(|(_, k)| *k)
                    .ok_or_else(|| {
                        let valid: Vec<String> = spec.iter().map(|(n, _)| format!("--{n}")).collect();
                        let for_cmd = if cmd.is_empty() { String::new() } else { format!(" for '{cmd}'") };
                        format!("unknown flag --{name}{for_cmd} (valid: {})", valid.join(" "))
                    })?;
                if kind == Kind::Bool {
                    a.bools.insert(name.to_string());
                } else {
                    let val = raw.get(i + 1).ok_or_else(|| format!("--{name} needs a value"))?;
                    let ok = match kind {
                        Kind::Usize => val.parse::<usize>().is_ok(),
                        Kind::U64 => val.parse::<u64>().is_ok(),
                        Kind::F32 => val.parse::<f32>().is_ok(),
                        Kind::F64 => val.parse::<f64>().is_ok(),
                        Kind::Str | Kind::StrList | Kind::Bool => true,
                    };
                    if !ok {
                        return Err(format!("--{name}: cannot parse '{val}' as {}", kind_name(kind)));
                    }
                    a.flags.entry(name.to_string()).or_default().push(val.clone());
                    i += 1;
                }
            } else {
                a.positional.push(arg.clone());
            }
            i += 1;
        }
        Ok(a)
    }

    /// Last value of `key` (values were validated at parse time).
    fn get<T: std::str::FromStr>(&self, key: &str, default: T) -> T {
        self.value(key).and_then(|s| s.parse().ok()).unwrap_or(default)
    }
    fn value(&self, key: &str) -> Option<&str> {
        self.flags.get(key).and_then(|v| v.last()).map(String::as_str)
    }
    fn get_all(&self, key: &str) -> Vec<String> {
        self.flags.get(key).cloned().unwrap_or_default()
    }
    fn has(&self, key: &str) -> bool {
        self.bools.contains(key)
    }
}

fn kind_name(k: Kind) -> &'static str {
    match k {
        Kind::Usize | Kind::U64 => "a non-negative integer",
        Kind::F32 | Kind::F64 => "a number",
        _ => "a string",
    }
}
```

Keep `fn config(&self)` as is, except replace `self.flags.get("resume").or_else(|| self.flags.get("checkpoint")).cloned()` with `self.value("resume").or_else(|| self.value("checkpoint")).map(String::from)` and `self.flags.contains_key("resume")` with `self.value("resume").is_some()`.

Elsewhere in `main.rs`, replace every `args.flags.get("X")` with `args.value("X")` (types: `Option<&str>` instead of `Option<&String>`; add `.map(String::from)` where an owned `String` is needed, e.g. `let resume = args.value("resume").map(String::from);`) and every `args.flags.contains_key("X")` with `args.value("X").is_some()`. Find them with `grep -n 'args.flags' slm_cli/src/main.rs`.

In `run()` replace `let args = Args::parse(&argv[2..]);` with:

```rust
    let Some(spec) = spec_for(&cmd) else {
        if matches!(cmd.as_str(), "-h" | "--help" | "help") {
            print_usage();
            return Ok(());
        }
        eprintln!("unknown command: {cmd}\n");
        print_usage();
        std::process::exit(1);
    };
    let args = Args::parse_for(&cmd, &argv[2..], spec)?;
```

and remove the now-unreachable `"-h" | "--help" | "help"` and `other =>` arms (keep a final `_ => unreachable!("spec_for covers every command")`).

Missing resume file — at the top of `train_and_save`:

```rust
    if let Some(ckpt) = args.value("resume") {
        if !std::path::Path::new(ckpt).exists() {
            return Err(format!("checkpoint {ckpt} does not exist (--resume needs an existing file; use --checkpoint to start a new one)").into());
        }
    }
```

Finally confirm each GGUF subcommand's flag table matches what its function reads:
Run: `awk '/^fn cmd_run_gguf/,/^}/' slm_cli/src/main.rs | grep -o 'args\.\(get\|has\|value\|get_all\)("[a-z_-]*"' | sort -u` (repeat for `cmd_finetune_gguf`, `cmd_export_gguf`) and add any missing name to that command's table.

- [ ] **Step 4: Run tests**

Run: `cargo test -p slm_cli --release 2>&1 | grep -E "^test result|FAILED|panicked"`
Expected: all ok, including `cli_strict` (5) and the 3 unit tests, and the existing `export_cli`.

- [ ] **Step 5: Commit**

```bash
cargo fmt --all && cargo clippy --workspace --all-targets -- -D warnings
git add slm_cli/src/main.rs slm_cli/tests/cli_strict.rs
git commit -m "fix(cli): reject unknown flags, bad values, and missing --resume files

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Phase 0 gate

- [ ] **Step 1: Full battery**

```bash
cargo build --workspace && cargo test --workspace --release 2>&1 | grep -E "^test result|FAILED"
cargo fmt --all -- --check && cargo clippy --workspace --all-targets -- -D warnings
cargo build -p slm_wasm --target wasm32-unknown-unknown --release
cargo clippy -p slm_wasm --target wasm32-unknown-unknown -- -D warnings
(cd ferrum_gui && cargo check --all-targets && cargo test 2>&1 | grep "test result")
```
Expected: all green.

- [ ] **Step 2: Measure the eval speed-up on the review model**

```bash
S=/tmp/claude-1000/-home-thomas-Projects-ferrum-cpu/88623770-ced2-4a37-be7f-75cb7f78b99e/scratchpad/review
cargo build --release -p slm_cli
# The review's FINF model no longer loads (Phase 0-G); train the reference config once and reuse it.
[ -f $S/ref.gguf ] || ./target/release/train_transformer train $S/train.txt $S/ref.gguf --context 128 --embed 96 --heads 4 --blocks 3 --hidden 320 --vocab 1024 --tokens 1.2e6 --lr 5e-3 --cosine --warmup 30 --clip 1 --tokenizer $S/ref_tok.gguf
time ./target/release/train_transformer eval $S/ref.gguf $S/heldout.txt
```
Expected: well under the review's 292 s baseline for 60 KB (target ≤ 10 s). Record the time in the Phase 0 summary for Status §9.
