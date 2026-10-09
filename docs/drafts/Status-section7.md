# Status: replacement text (draft)

<!-- REPLACES: Status §7 "Backward compatibility". Keep the existing bullets that are still true (see "Carried over" at the end). -->
<!-- ALSO: Status §5 "Known issues" is replaced by the table at the end of this file. §1 Snapshot, §6 Roadmap, and §9 Metrics need the numbers marked MEASURE. -->

## 7. Backward compatibility

This release has deliberate breaking changes. Each one is listed with what you
must do.

### 7.1 Breaking changes

| Change | What breaks | What to do |
|---|---|---|
| **FINF is removed.** Every file Ferrum writes is GGUF v3: models, training checkpoints, and tokenizer files. | `.bin` model files (FINF v4 and v5) no longer load. Checkpoints written before this change (magic `FCKP`) fail with a "pre-GGUF checkpoint" error. Old tokenizer files in the `256;257...` text form are rejected. Model file names are now `.gguf`. | Retrain. Committed models in this repo and in Ferrum-ML (10 tabular models) were converted before the reader was deleted. |
| **The Transformer is GPT-2-compatible.** The feed-forward activation changes from ReLU to GELU (tanh approximation), and the LM head loses its bias. | A model trained before this change computes a different function. Its weights do not carry over. The parameter count falls by `vocab` (the removed head bias). Tests that pinned exact losses or generated text were re-derived. | Retrain. There is no converter. |
| **The tokenizer changed.** Pre-tokenization now uses the GPT-2 regex llama.cpp uses (`tokenizer.ggml.pre = "gpt-2"`), and vocabulary and merges are stored in GPT-2 bytes-to-unicode form. | Token ids differ from the old tokenizer for the same text. Tokenizers saved before this change cannot be reused with `--tokenizer`. | Let `train` build a new tokenizer file. |
| **`<|endoftext|>` is always reserved.** BPE training registers it as a special token, counted in the vocabulary size, so every model has a real `eos_token_id`. | The reserved id is counted in `--vocab`. The literal text `<|endoftext|>` now encodes to one id. | None for new models. Older tokenizers had no specials, and `eos_id()` was `None` for them; they no longer load as GGUF tokenizer files anyway. |
| **QAT simulates Q8_0 exactly.** Fake-quantization works in 32-element blocks with an f16 scale. | Int8 weights differ from the old per-channel int8 snapping, so a model trained before this change gives different int8 numbers. | Retrain. A new int8 file reloads bit-identically. |
| **The CLI is strict.** Each subcommand accepts only its own flags. | Scripts with a misspelled or unsupported flag, a value that does not parse, or a `--resume` path that does not exist now stop with an error. Before, they ran with a silent default. | Fix the command. The error names the flag and lists the valid ones. |
| **`temperature 0` is exact argmax.** The old code clamped it to 0.01. | Greedy output can differ from runs made with the old clamp. | None. `--temp 0` is now reproducible and matches llama.cpp at `--temp 0`. |
| **Perplexity covers every token.** `evaluate()` and training `--val` are rebuilt on `score_rolling`. The old code skipped the first window. | Perplexity, bits/token, and `num_predictions` are not comparable with numbers from before this change. | Re-evaluate old models if you need a comparison, in a Ferrum version that can load them. For new work, compare bits per byte. |
| **Other behaviour changes.** | `info` prints the parameter count and labels int4 files. `-v` on `train` prints a progress line every 10 steps instead of per-op logs (the old output is behind `--debug`). With `--val`, a tokenizer trained in the same run uses only the train portion. A resumed run restores the best validation score and reports the correct epoch and step. `--tie` shrinks the parameter count and the file by `vocab × embed`. | None. |
| **Character-level models** raise an error for an unknown character in `score` and `score_rolling`, instead of using id 0. `evaluate()` reports the count. | Callers that relied on silent id 0 now get an error. | Clean the text, or use a BPE model. |
| **Dependencies were admitted.** The zero-dependency rule is replaced by "dependencies need a measured reason". | `ferrum_core` now builds with a few crates. `deny.toml` and `cargo deny check` cover them. | None. |
| **`TransformerConfig` gained `log_every_steps`** (default 0 = off). | Struct literals already need `..TransformerConfig::default()`. | None. |

<!-- MEASURE: parameter counts before/after for the review config, to quote in the GELU/no-head-bias row (reference 507,552 is after) -->

### 7.2 What still works

- The imported Llama/Qwen stack (`llm`, `llm_train`, `gguf*`, `chat_format`) is
  untouched, and so are `run-gguf`, `finetune-gguf`, and `export-gguf`.
- The positional `train_transformer*` functions keep stride 1 and their API.
  Their numeric results changed for the reasons above (GELU, tokenizer).
- `TransformerConfig` is extended only by fields with defaults. Its default
  stride is `context_len`. Set `window_stride: 1` (`--stride 1`) for full overlap.
- `generate`, `generate_with`, `generate_until`, and `generate_stream` keep their
  signatures. The new token-budget generator is `generate_tokens`.
- The Ferrum GGUF `ferrum.format` is 1.

### 7.3 Carried over from the old §7 (still true)

- **Removed APIs** from earlier releases stay removed: `GenerativeSLM::train`,
  `train_with_callback`, `train_embedded`, `train_embedded_with_callback`,
  `slm::build_csv_dataset`, `ferrum_core::{CsvDataset, CsvRow,
  fit_normalizer_with_target, train_val_split, Net, EmbedT, train_epoch,
  accuracy}` (the last six live in `ferrum_ml`), and the `ferrum_core::csv`
  module (now `meta`).
- Legacy one-hot MLP model files can no longer be read at all (they were FINF).
- The GUI `train_slm` command ignores the removed `momentum` field. New fields
  are optional and default sensibly.

---

## 5. Known issues (replacement table)

<!-- Items fixed by this release are removed: slow eval, O(n^2) generation, silent CLI parsing, missing resume file, -v flood, no EOS between documents, CLI exposing only temperature, no metrics export. Items kept are still true. -->

| Severity | Issue |
|---|---|
| 🟡 | **The training corpus is read whole.** `train` loads the prepared train split into memory and holds tokens as 8-byte `usize`. `prepare` streams its input, but there is no pre-tokenized file or streamed training. A multi-gigabyte corpus will hit RAM. |
| 🟡 | **The server handles one request at a time.** One dedicated model thread owns the model, because `GenerativeSLM` is not `Send` or `Sync`. Each matmul still uses every core, but a long batch delays other clients. It has no authentication and no TLS, and binds to loopback by default. |
| 🟡 | **LightEval needs two workarounds on 0.13.0.** `FerrumModel` skips `super().__init__()`, and rolling log-likelihood is not cached. The module is pinned to that version, to `xxhash<4`, and to `datasets<5`. The litellm and TGI backends cannot run log-likelihood tasks. Python is not in CI. |
| 🟡 | **Quantized files can differ from llama.cpp late in a generation.** llama.cpp quantizes activations in Q8_0 matmuls. f32 files match exactly; Q8_0 files are compared over the first 8 tokens; Q4_0 files are checked only for loading and generating. |
| 🟡 | **Character-level models and Ferrum-ML tabular models do not run in llama.cpp, Ollama, or LM Studio.** They are GGUF, but use a generic Ferrum layout. The export commands refuse them. |
| 🟡 | **Only the verified tool versions are tested.** llama.cpp b11521, Ollama v0.40.2, LM Studio 0.4.18. Runtime tests skip a missing tool, but fail under `FERRUM_REQUIRE_RUNTIMES=1`; the CI `runtimes` job sets it (llama.cpp and Ollama required, `FERRUM_SKIP_LMSTUDIO=1` as the one exemption; LM Studio is required locally before a release). Newer versions may behave differently. |
| 🟡 | **Models are small and CPU-bound.** The reachable sizes in Status §3 are unchanged until re-measured. <!-- MEASURE: re-run the capability table after the speed work --> |
| 🟡 | **The Capable tab's training estimates are rough.** They ignore activation memory and attention FLOPs, and they scale a single-thread GEMM rate by 8 hyperthreads on 4 physical cores. Treat them as loose upper bounds. |
| 🟡 | **The GUI is not click-tested after the 2026-10-03 changes, or after the GGUF change.** The backend compiles and its tests pass. |
| 🟢 | Resume may not replay the exact remaining batches of the interrupted epoch (marked `ponytail:` in `slm.rs`). <!-- MEASURE: confirm this still holds after the resume-epoch fix; delete this row if fixed --> |
| 🟢 | GGUF import of Llama/Qwen models is lossy (dequantize and re-quantize to Ferrum's per-row grid) and not bit-exact to llama.cpp. RoPE type is hard-wired to `Norm` on import; no RoPE scaling. |
| 🟢 | No DVC remote is configured. `data/` and `models/` live only on the machine that ran `dvc repro`. |
| 🟢 | `--tie` and LM-head sharing apply to Ferrum's own models only. |

<!-- MEASURE: test counts for §1 Snapshot (ferrum_core unit, integration, slm_wasm, slm_server, doc-tests, ferrum_gui, Ferrum-ML) -->
<!-- MEASURE: §9 Metrics before/after: train tokens/s, eval seconds, generate ms/token on the reference model, with the machine name and date -->

---

## 1. Snapshot (replacement)

| | |
|---|---|
| **Scope** | A pure-Rust engine for training, running, serving, and evaluating small transformer SLMs on the CPU. Models are GPT-2-architecture GGUF files that run in llama.cpp, Ollama, and LM Studio. Also GGUF import, fine-tuning, and export for Llama/Qwen |
| **Date and branch** | 2026-10-09 · `feat/dataops-serve-eval` <!-- MEASURE: version and branch at release --> |
| **Build** | <!-- MEASURE: cargo build --workspace result; ferrum_gui build result --> |
| **Tests** | <!-- MEASURE: counts for ferrum_core unit, slm_cli, slm_server, integration, slm_wasm, doc-tests, ferrum_gui, Ferrum-ML; Python unittest count --> |
| **Safety** | `#![forbid(unsafe_code)]` in every crate, plus a repo-wide scan test |
| **Dependencies** | Few, each with a measured reason <!-- MEASURE: list --> |
| **Commits** | <!-- MEASURE: git log --oneline master..HEAD --> |

## 6. Roadmap (replacement)

### 6.1 Done in this release

Scoring API and fast evaluation, EOS as a special token, strict CLI, linear-time
generation, `prepare` with manifests, DVC pipeline, `ferrum-serve`, the LightEval
module, GGUF-only files, GPT-2-compatible models, and the Ollama and LM Studio
export commands.

### 6.2 Still open

<!-- MEASURE: copy the still-open roadmap rows from Status.md §6 at release -->
These are directions, not commitments.

## 9. Metrics (replacement)

_As of commit <!-- MEASURE: sha and date -->, excluding `target/`._

| Metric | Value |
|---|---:|
| Workspace crates | <!-- MEASURE: ferrum_core, slm_cli, slm_server, slm_wasm, tests + ferrum_gui (excluded) --> |
| `ferrum_core` source files / lines | <!-- MEASURE --> |
| External dependencies (`ferrum_core`) | <!-- MEASURE --> |
| `unsafe` blocks | 0 (forbidden) |
| Tests | <!-- MEASURE --> |
| Train tokens/s, reference model, review baseline vs final | <!-- MEASURE: machine, date --> |
| Eval seconds, reference model, before vs after | <!-- MEASURE: machine, date --> |
| Generate ms/token, before vs after | <!-- MEASURE: machine, date --> |
| Markdown documents | 5 (`ReadMe`, `Manual`, `Status`, `Samples`, `glossary`) |
