# DataOps, serving, LightEval, speed, and samples — design

_2026-10-09 · branch `feat/dataops-serve-eval` · status: approved; amended 2026-10-09 (GGUF only, GPT-2-compatible, no unsafe)_

## 1. Goal

Make the train → evaluate → run workflow for a ~500K-parameter SLM fast,
reproducible, servable, and testable with a standard harness:

1. Fix every defect found by the 2026-10-09 workflow review (§9).
2. Drop the zero-dependency constraint and use external crates where they give
   a **measured** speed-up.
3. Add a DVC-driven data pipeline: fetch → prepare (clean) → train → eval.
4. Add an HTTP inference endpoint (`slm_server`, axum) and a Hugging Face
   LightEval module that evaluates Ferrum models through it.
5. Write `Samples.md`: 10 detailed, actually-executed walk-throughs.
6. **GGUF only** (amendment): the FINF format is removed. Every file Ferrum writes — models, training
   checkpoints, tokenizer files — is GGUF. Ferrum's own trained SLMs are **GPT-2-architecture GGUFs
   that run unmodified in llama.cpp, Ollama, and LM Studio** (§4.0).
7. **No unsafe code** in any crate (`#![forbid(unsafe_code)]` everywhere, plus a repo-wide test).
8. Unit tests land with every feature (test first).

Out of scope: the imported Llama/GGUF stack (`llm`, `llm_train`, `gguf*`,
`chat_format`) is untouched; streaming the *training* corpus (the trainer still
loads the prepared train split whole); OpenAI-compatible routes; a DVC remote.

## 2. Decisions (user-approved)

| Topic | Decision |
|---|---|
| DataOps | DVC: `dvc.yaml` + `params.yaml`, `dvc repro` reproduces data, model, metrics |
| Endpoint | New workspace crate `slm_server` on axum/tokio/serde |
| Defects | Fix all: #1–#5 Important and all Minor |
| Corpus | TinyStories V2 (GPT-4) validation file, pinned (§5.4) |
| Dependencies | Zero-dependency constraint dropped; crates admitted on measured gain |
| Samples | New top-level `Samples.md`, 10 samples, every command executed |
| File formats | GGUF only; FINF removed (breaking: old `.bin` files stop loading; committed ones converted) |
| Model architecture | Ferrum's own Transformer becomes GPT-2-compatible (GELU FFN, no LM-head bias, GPT-2 pre-tokenizer + byte-to-unicode vocab); saved as `gpt2` GGUF in F32/F16/Q8_0/Q4_0 |
| Compatibility target | llama.cpp, Ollama, LM Studio run Ferrum GGUFs; verified by automated tests against installed binaries |
| Unsafe | `#![forbid(unsafe_code)]` in every crate (`deny` only where a third-party proc macro forces it) |

## 3. Hard invariants (still enforced)

- `#![forbid(unsafe_code)]` in Ferrum's own crates (dependencies may use unsafe internally; we call only safe APIs).
- Bit-for-bit identical results at `FERRUM_NUM_THREADS=1` vs N, enforced by a test. A crate that breaks this is rejected.
- `ferrum_core` and `slm_wasm` build and run on `wasm32-unknown-unknown`.
- All model, checkpoint, and tokenizer files are GGUF v3. FINF is gone (breaking change, recorded in Status §7).
- A model Ferrum saves and the same file run in llama.cpp produce the same greedy continuation for a fixed prompt (tested).
- `TransformerConfig` grows only via fields with defaults.
- Every commit builds and passes its tests.

## 4. Workstream A — core engine

### 4.0 GGUF everywhere, GPT-2-compatible models (Phase 0-G; runs first)

- **Architecture changes to Ferrum's own Transformer**: FFN activation ReLU → GELU (tanh
  approximation, as llama.cpp's `ggml_gelu`); the LM head loses its bias; weight tying exports
  by omitting `output.weight` (llama.cpp falls back to `token_embd`), which also makes `--tie`
  shrink the file. Everything else (learned positions, pre-LN LayerNorm ε=1e-5 with bias, dense MHA
  with biases) already matches GPT-2.
- **Tokenizer**: byte-level BPE pre-splits text with the GPT-2 regex llama.cpp uses for
  `tokenizer.ggml.pre = "gpt-2"`, and writes tokens/merges in GPT-2's bytes-to-unicode form, so
  llama.cpp tokenizes prompts exactly as Ferrum does. EOS is `<|endoftext|>`.
- **Model file** = `general.architecture = "gpt2"` GGUF with llama.cpp's GPT-2 tensor names (fused
  `attn_qkv`), plus a `ferrum.*` metadata namespace (task, class names, normalizer, quant) that
  llama.cpp ignores. Tensor types: F32, F16, Q8_0 (int8), Q4_0 (int4). A tensor whose row length is
  not a multiple of 32 is stored F16 (Q8_0/Q4_0 need 32-element blocks).
- **QAT simulates Q8_0 exactly** (32-element blocks along each weight row, f16-rounded scale), so an
  int8 file reloads bit-identically to the trained model. int4 (Q4_0) export stays a lossy option.
- Loaded Ferrum GGUF weights are dequantized to f32 in memory (small models); the packed
  int8/int4 kernels remain for imported Llama models.
- **Non-model files**: training checkpoints and the reusable tokenizer file are GGUF containers
  with `general.architecture = "ferrum-checkpoint"` / `"ferrum-tokenizer"` (not runnable models).
- **Non-transformer models** (Ferrum-ML tabular MLPs) are GGUF with `general.architecture = "ferrum-mlp"`
  (generic layer list) — GGUF, but not runnable in llama.cpp, which has no MLP-classifier architecture.
- Exact key names, tensor shapes, and the pre-tokenizer regex are pinned from llama.cpp's source in
  plan `2026-10-09-p0g-gguf-only.md`.

### 4.1 Scoring API (Phase 0; the shared interface)

In `ferrum_core::slm`:

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Score {
    /// Sum of natural-log probabilities of the scored tokens.
    pub logprob: f64,
    /// Number of tokens scored.
    pub n_tokens: usize,
    /// True when every scored token was the argmax prediction.
    pub greedy: bool,
}

impl GenerativeSLM {
    /// log p(continuation | context).
    pub fn score(&self, context: &str, continuation: &str) -> Result<Score>;
    /// Loglikelihood of the whole text (every token after the first).
    pub fn score_rolling(&self, text: &str) -> Result<Score>;
}
```

- **Tokenization** (lm-eval-harness convention): trailing whitespace of
  `context` is moved to the front of `continuation`; encode `context` and
  `context + continuation`; continuation tokens are the suffix of the joint
  encoding after the context encoding's length. If the joint encoding does not
  start with the context encoding, fall back to encoding the two separately and
  concatenating. Empty context: score from a BOS position (token 0 / EOS id
  when present) so the first continuation token is scored.
- **Compute**: one padded forward per window (rows = `context_len`; padding is
  safe under the causal mask), reading the log of the target token's
  probability from every row. Inputs longer than `context_len` use sliding
  windows with stride `context_len / 2`; each window scores only tokens not
  scored by a previous window, so every token is scored exactly once with at
  least `context_len / 2` tokens of history (except the first window).
- **Floor**: the model emits clamped probabilities (min 1e-12). `logprob` uses
  that value; document the −27.63 per-token floor.
- `evaluate()` and training `--val` are re-implemented on `score_rolling`
  (defect #1). Perplexity now covers every token after the first instead of
  skipping the first window; record this in `Status.md` §7.
- Character-level models: an unknown character is an error in `score*`
  (not a silent id 0); `evaluate()` reports the count of unknown characters.

### 4.2 EOS as a special token (Phase 0)

- BPE training registers `<|endoftext|>` as a special token (id reserved
  after merges; counted in vocab). `ByteBpeTokenizer::eos_id()` returns it.
- The encoder maps the literal string `<|endoftext|>` in text to the EOS id.
- Older tokenizers (no specials) keep working; `eos_id()` is `None` for them.

### 4.3 Strict CLI parser (Phase 0)

- Per-subcommand allow-list of flags with their kind (bool / value).
- Unknown flag → error naming the subcommand and listing valid flags.
- A value that fails to parse → error (never a silent default).
- `--resume <path>` that does not exist → error (defect #4).
- Existing flags and defaults otherwise unchanged.

### 4.4 Generation (Phase 1, agent A)

- **Re-prime fix (defect #2)**: learned absolute positions forbid sliding the
  KV cache one token at a time. When the cache is full, re-prime from the last
  `context_len / 2` tokens — one re-prime per `context_len / 2` generated
  tokens. Text output is decoded incrementally (no full re-decode per step).
- `generate_until(prompt, stops: &[&str], max_tokens, sampling)`: stops are a
  list; returned text excludes the matched stop; budget is in tokens; stops on
  EOS when the tokenizer has one.
- `temperature == 0.0` is exact argmax (the 0.01 clamp is removed for 0).
- Native CLI `generate` wires `--top-k`, `--top-p`, `--repetition-penalty`,
  `--max-tokens`, `--stop` (repeatable) to the existing `generate_with` path.

### 4.5 Speed through dependencies (Phase 1, agent A)

1. Baseline: `cargo bench --bench gemm` and timed train / eval / generate of
   the 500K config (§5.4 data), recorded in the commit message and Status §9.
2. Matmul: benchmark `gemm` and `matrixmultiply` behind `ops::matmul` and
   `qlinear`; keep the faster one that passes the determinism test and the
   wasm32 build. Reject both if neither beats the current kernel.
3. Threading: replace `parallel.rs`'s hand-written pool with `rayon` if it is
   no slower (less code is a gain in itself). Verify wasm32 behaviour.
4. The `gemm` bench may move to Criterion.
5. `deny.toml` updated so `cargo deny check` passes with the new crates.
6. Docs: ReadMe pitch, Manual §2, Status, `CLAUDE.md` hard-constraints list.

### 4.6 Minor fixes (Phase 1, agent A)

- `info` prints the parameter count and labels int4 files as int4.
- CLI can write int4 models (`--quant int4` on train/save path).
- `--tie` shares the weights: reported parameter count and saved file shrink by `vocab × embed` (file: via §4.0's omitted `output.weight`).
- Resume with `--val` restores the best-so-far validation score from the checkpoint.
- A resumed mid-epoch run reports the correct epoch/step.
- The banner shows the token budget instead of "Epochs: 100" when `--tokens` is set.
- `-v` during training is throttled (per-step summaries, not per-op logs).
- When `--val` is set and a tokenizer is trained, it is trained on the train portion only.
- Doc corrections: QAT is int8 only; Manual §7 per-token cost; `--tie`; `-v` advice in Status §5; `clean_corpus` lowercase default; eval cost.

## 5. Workstream B — DataOps (Phase 1, agent B)

### 5.1 `train_transformer prepare`

```
train_transformer prepare --input <file|dir>... --out <dir>
    [--doc-sep '<|endoftext|>'] [--lossy] [--nfkc] [--lowercase]
    [--strip-gutenberg] [--min-chars N] [--max-chars N] [--dedup]
    [--val-frac 0.05] [--test-frac 0.05] [--seed 42]
```

- Streams each input with `BufRead`; directories are walked in sorted order.
- Per document: UTF-8 decode (`--lossy` replaces invalid bytes, otherwise
  error with file and byte offset); strip BOM, zero-width (U+200B–U+200D,
  U+2060, U+FEFF) and control characters except `\n` and `\t`; NFKC via
  `unicode-normalization` when `--nfkc`; collapse runs of spaces and of 3+
  newlines; optional lowercase; Gutenberg header/footer stripping per book
  (handles concatenated books); length filter; exact dedup on a 128-bit content
  hash (only hashes are held in memory).
- Split: `split = H(seed, doc) mod 10_000` against `val-frac`/`test-frac`
  thresholds — deterministic, leak-free, stable when documents are appended.
- Outputs `out/{train,val,test}.txt` (documents joined by `\n<|endoftext|>\n`)
  and `out/manifest.json`:
  - `ferrum_version`, full argument list
  - per input: path, bytes, sha256
  - per output: path, bytes, sha256, documents, characters
  - rejected counts by reason (`empty`, `too_short`, `too_long`, `duplicate`, `invalid_utf8`)
  - `corpus_stats` of the train split
- Library: the cleaning steps live in `ferrum_core::dataset` as composable
  functions; `clean_corpus` is kept as a wrapper.

### 5.2 Metrics output

`train_transformer eval ... --json <path>` writes
`{"perplexity", "cross_entropy_nats", "bits_per_token", "bits_per_byte", "tokens", "bytes", "seconds"}`.

### 5.3 DVC

- `params.yaml`: `fetch` (url, sha256), `prepare` (flags in §5.1), `model`
  (context, embed, heads, blocks, hidden, vocab — the 486,624-param review
  config as default), `train` (tokens, lr, warmup, clip, val, patience, quant,
  seed).
- `dvc.yaml` stages, each a CLI call templated from `params.yaml`:
  - `fetch`: `scripts/fetch.sh <url> <sha256> data/raw/tinystories.txt` (curl, then a sha256 check that fails the stage on mismatch)
  - `prepare` → `data/prepared/` (`outs`), `data/prepared/manifest.json`
  - `train` → `models/slm.gguf`, `models/tokenizer.gguf`
  - `eval` → `metrics/eval.json` (`metrics`, cache: false)
  - `lighteval` (optional, `frozen: true` by default) → `metrics/lighteval.json`
- `dvc.lock` committed. `data/`, `models/` git-ignored and tracked by DVC.
  `.dvc/` initialised with no remote.

### 5.4 Default corpus

TinyStories V2 (GPT-4) validation file — enough text (~22 MB, millions of
tokens) for a 500K model, and stories are already separated by
`<|endoftext|>`.

- URL: `https://huggingface.co/datasets/roneneldan/TinyStories/resolve/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/TinyStoriesV2-GPT4-valid.txt`
- Size: 22,502,601 bytes
- sha256: `6874bae9a4c1a4e7edcf0e53b86c17817e9cf881fc75ff2368da457b80c0585d`
- Licence: CDLA-Sharing-1.0 (data is downloaded, never committed).

## 6. Workstream C — serving and LightEval (Phase 1, agent C)

### 6.1 `slm_server` crate

Binary `ferrum-serve --model <path.gguf> [--host 127.0.0.1] [--port 8080]
[--max-batch 64] [--max-chars 65536] [--max-body 8MiB]`.

| Route | Request | Response |
|---|---|---|
| `GET /health` | — | `ok` |
| `GET /v1/info` | — | `{params, context_len, vocab, quant, tokenizer, model_sha256}` |
| `POST /v1/loglikelihood` | `{"requests":[{"context","continuation"}]}` | `{"results":[{"logprob","n_tokens","greedy"}]}` |
| `POST /v1/loglikelihood_rolling` | `{"texts":[...]}` | `{"results":[{"logprob","n_tokens"}]}` |
| `POST /v1/generate` | `{"requests":[{"prompt","max_tokens","stop","temperature","top_k","top_p","seed"}]}` | `{"results":[{"text","n_tokens"}]}` |

- Model held in `Arc`; each request's work runs in `spawn_blocking`.
- Limits enforced before any compute; violations → `400`/`413` with
  `{"error": "..."}`. Model errors → `422`. Binds loopback by default.
- `generate` defaults: `temperature 0` (greedy), `max_tokens 64`, `seed 0`.

### 6.2 `eval/lighteval/`

Pinned to **lighteval 0.13.0** (Python ≥ 3.10).

- `ferrum_model.py`: `FerrumModel(LightevalModel)`
  - sets `self.config = config` and `self._cache = SampleCache(config)` directly (no `super().__init__`, which raises in 0.13.0)
  - `tokenizer` → `None`, `add_special_tokens` → `False`, `max_length` → `/v1/info` `context_len`
  - `greedy_until` (`@cached(GENERATIVE)`) → `/v1/generate` with `doc.stop_sequences`, `doc.generation_size`
  - `loglikelihood` (`@cached(LOGPROBS)`) → `/v1/loglikelihood`, one entry per choice; fills `logprobs` and `argmax_logits_eq_gold`
  - `loglikelihood_rolling` (**not** cached: 0.13.0's PERPLEXITY cache drops results) → `/v1/loglikelihood_rolling`
  - server URL from `FERRUM_URL` (default `http://127.0.0.1:8080`); stdlib `urllib`; requests batched
- `ferrum_tasks.py` (`TASKS_TABLE`):
  - `ferrum_heldout`: rolling loglikelihood on `data/prepared/test.txt` documents (local jsonl generated by `run.sh`); metrics word/byte perplexity, bits per byte
  - `tinystories_cloze`: multiple choice — story prefix, gold = true next sentence, 3 distractors = sentences from other stories (fixed seed); metric `loglikelihood_acc`
- `requirements.txt`: `lighteval==0.13.0`, `xxhash<4` (4.x crashes the details logger), CPU torch index.
- `run.sh`: start `ferrum-serve`, poll `/health`, run
  `lighteval custom ferrum ferrum_model.py "<tasks>" --custom-tasks ferrum_tasks.py --output-dir ...`,
  stop the server, write a summary to `metrics/lighteval.json`.
  Default tasks: `ferrum_heldout`, `tinystories_cloze`, a `blimp` subset, `arc:easy`;
  `--wikitext` adds `wikitext:103:document_level`.

## 7. Workstream D — `Samples.md` and docs (Phase 2)

`Samples.md`, 10 samples. Each has: goal, prerequisites, exact commands,
**real** output excerpts captured while writing, and what to look at.

1. Quick start: train a 500K SLM on TinyStories; sizing a config to a parameter target
2. Full DVC pipeline: `dvc repro`, edit a param, `dvc params diff`, `dvc metrics diff`
3. Cleaning your own corpus with `prepare`; reading `manifest.json`
4. Training controls: token budgets, cosine LR, `--val` + early stopping, checkpoint → kill → resume
5. Evaluation: perplexity, bits/byte, int8 vs int4 size and quality, `--json`
6. Generation: sampling knobs, stop sequences, EOS, seed determinism
7. Serving: `ferrum-serve` plus `curl` for every route
8. Testing with LightEval: `run.sh`, custom and built-in tasks, reading results
9. Benchmarking: `cargo bench`, train/eval/generate throughput, thread scaling, before/after numbers for §4.5
10. Library API: train, score, generate from Rust

Also: ReadMe (pitch, layout table with `slm_server`, docs list), Manual
(§2, corpus prep with `prepare`, eval, serving chapter pointer), Status
(snapshot, issues, roadmap, metrics, §7 compat), `CLAUDE.md`/`AGENTS.md`
(constraints, layout, commands, docs list incl. `Samples.md`).

## 8. Testing and CI

| Area | Tests |
|---|---|
| Scoring | `score` equals a naive per-token reference; whitespace and mid-word boundaries; rolling windows score each token exactly once; empty context; char-level unknown char errors |
| Eval | new `evaluate()` agrees with summed `score_rolling` |
| Generation | re-prime path identical to old path before the window fills; long generation runs in linear time (step-count bound, not wall-clock); multi-stop excluded from text; temperature 0 = argmax |
| Determinism | train/eval/generate identical at 1 vs N threads |
| CLI | unknown flag, bad value, missing resume file each error |
| prepare | BOM/zero-width stripped, dedup, split stability on append, multi-book Gutenberg, manifest hashes match files |
| GGUF / compat | f32/int8/int4 round-trips; QAT int8 reload bit-identical; GPT-2 tokenizer matches the regex on edge cases; greedy output equal in Ferrum and llama.cpp; Ollama and LM Studio load and run the file (tests skip when the binary is absent) |
| No unsafe | repo-wide source scan test |
| Server | each route, limits (413/400), error JSON, via `tower::ServiceExt::oneshot` |
| LightEval | stdlib `unittest` of `FerrumModel` against a stub HTTP server; real `lighteval custom` smoke run with `--max-samples 5` at final verification |

CI: `slm_server` in the workspace jobs; `cargo deny` covers new crates; wasm32
job unchanged; Python not in CI (torch is heavy), smoke run is part of the
final verification battery.

## 9. Review findings addressed

Important: #1 slow eval/`--val` (§4.1) · #2 O(n²) generation (§4.4) · #3 silent CLI parsing (§4.3) · #4 missing resume file (§4.3) · #5 `-v` volume (§4.6).
Minor: all listed in §4.6. Data gaps: §5.1, §4.2. Endpoint gaps: §4.1, §4.4, §6.

## 10. Execution

- Branch `feat/dataops-serve-eval`.
- **Phase 0-G** (one agent, sequential, first): §4.0 — GGUF everywhere, GPT-2-compatible models, no unsafe.
- **Phase 0** (one agent, sequential): §4.1, §4.2, §4.3 — the shared surfaces (`slm.rs` API, tokenizer, `main.rs` parser).
- **Phase 1** (three agents in parallel worktrees branched from Phase 0):
  - A: §4.4, §4.5, §4.6
  - B: §5
  - C: §6
  - Merged back in order A, B, C; conflicts resolved at merge.
- **Phase 2**: §7, then the full verification battery: workspace build/test/fmt/clippy, wasm32, `cargo deny`, GUI crate, `dvc repro`, LightEval smoke run.
- Finish: fast-forward `master`, push, delete branch.
