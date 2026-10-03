# Project Status

_Last updated: 2026-10-03 · version 0.3.0 · branch `feat/eight-upgrades`_

This page is the single place for **where Ferrum stands**: what was just
reorganized, what the engine can realistically do on laptop hardware, what is
implemented and tested, what is broken or missing, what comes next, and how the
project got here. For *how to use* Ferrum, see [Manual.md](Manual.md). For the
overview, see [ReadMe.md](ReadMe.md).

---

## Contents

1. [Snapshot](#1-snapshot)
2. [The 2026-10-03 restructuring](#2-the-2026-10-03-restructuring)
3. [Capability assessment](#3-capability-assessment)
4. [Implemented and tested](#4-implemented-and-tested)
5. [Known issues](#5-known-issues)
6. [Roadmap](#6-roadmap)
7. [Backward compatibility](#7-backward-compatibility)
8. [Development history](#8-development-history)
9. [Metrics](#9-metrics)

---

## 1. Snapshot

| | |
|---|---|
| **Scope** | A zero-dependency Rust engine for training, running, and shipping small transformer SLMs on the CPU, plus GGUF import, fine-tuning, and export for Llama/Qwen |
| **Build** | `cargo build --workspace` clean; `ferrum_gui` builds separately and clean |
| **Tests** | **418 passed, 0 failed** (360 `ferrum_core` unit, 52 integration, 3 `slm_wasm`, 3 doc-tests). **39** `ferrum_gui` backend tests pass. The sibling **Ferrum-ML** project passes **78**. |
| **Safety** | `#![forbid(unsafe_code)]`; zero external dependencies in `ferrum_core` |
| **Commits** | `133cdfe` F8 (Q2_K/Q3_K) · `d7ecb3a` pipeline fixes · `ed672d0` restructuring · `6dcb629` docs; each one builds and passes its tests on its own |

---

## 2. The 2026-10-03 restructuring

An evaluation on 2026-10-03 (summarized in §3) found two kinds of problem: a
training pipeline that could not run at real scale, and a repository carrying a
second product (tabular ML) plus several dead-end features. Both were
addressed the same day.

### 2.1 Training-pipeline fixes

| Defect found | Fix | Evidence |
|---|---|---|
| **Stride-1 windows.** One "epoch" trained on every token about `context_len` times. A 6.9M model on 10M tokens needed about 32 days per epoch. | Configurable window stride, defaulting to `context_len` (non-overlapping) with a random phase each epoch. The legacy positional API keeps stride 1. | A test pins steps per epoch at each stride; on a real corpus one epoch went from about 12,000 steps to 186 |
| **No token or step budget** | `max_tokens` / `--tokens 5e7` stops exactly on budget, mid-epoch if needed, through a new step-capped epoch core (`train_transformer_steps` with a per-step hook) | Tests: budget stops at the exact step across epoch boundaries |
| **BPE trainer was O(merges × corpus)**: 437 KB took 49 s, and 1 GB would take over 30 h | Count each distinct word once and update pair counts incrementally using a lazy max-heap | 437 KB: **49 s → 0.09 s**; 22 MB: 0.33 s; an oracle test proves merges are **identical** to the old algorithm |
| **Tokenizer retrained every run** | `tokenizer_path` / `--tokenizer`: trained and saved once, reused afterwards | Test: second run on different text reuses the saved merges |
| **Schedule, clipping, tying, validation, and checkpoints existed but were unreachable** | All wired into `TransformerConfig`, the CLI (`--cosine --warmup --clip --tie --no-qat --val --patience --checkpoint --checkpoint-every --resume`), and the GUI Train tab | End-to-end: a run killed at 25 s resumed at step 300, reused its tokenizer, stopped exactly at the 1,465-step budget; loss 5.23 → 3.44 |
| **QAT forced on** | `qat` field / `--no-qat` | Test |
| **GUI dropped weight decay and dropout** | Restored, alongside all the new options | GUI tests pass |
| **Checkpoint writes could corrupt on a crash** | Atomic writes (temporary file, then rename) | — |
| **Resume could silently mix tokenizers** | Resume refuses a checkpoint whose vocabulary or context differs | Test |

### 2.2 What moved out

The repository was narrowed to *training and releasing SLMs*. Everything else
went to one of two sibling folders next to the repository.

| Item | Went to | Why |
|---|---|---|
| Tabular MLP stack: `train_cli`, the CSV loader (`CsvDataset`, split helpers), the trainable MLP (`Net`), 10 datasets, 10 pretrained models, download/train scripts, and their tests | **`../Ferrum-ML`** as crate `ferrum_ml` + `tabular_wasm`, depending on `ferrum_core` by path | A working, useful ML demo that has nothing to do with SLMs |
| `TabularModel` browser binding | **Ferrum-ML** `tabular_wasm` | Same |
| One-hot MLP SLM trainer (`GenerativeSLM::train`), embedding-MLP trainer (`train_embedded`), hex-CSV builder, one-hot inference and scoring paths, and their 6 tests | **`../Ferrum-Junk`** (`removed_slm_paths.rs`, `removed_slm_tests.rs`) | Teaching baselines strictly dominated by the transformer, and of no use on their own |
| `web/` launcher (linked to three external repos not present here, ran one-hot models only) | **Ferrum-Junk** `web/` | Broken links, superseded model family |
| GitHub Pages `deploy.yml` (trained tabular models, published `web/`) | **Ferrum-Junk** `github-workflows/` | Deployed only the removed pieces |
| GUI embedded shell (`run_terminal`, `term_cwd`) and the Tabular tab | **Ferrum-Junk** `gui/` (fragments) | Arbitrary command execution from the webview, unrelated to SLMs. The docked panel remains as a read-only engine log. |
| GUI Train-tab method selector (embedded / one-hot) | removed | Only the transformer path remains |
| 26 overlapping Markdown files (root docs, `docs/`, `manual/`, crate READMEs) | Consolidated into **ReadMe.md, Manual.md, Status.md**. The agent design specs, plans, and work logs are archived in **Ferrum-Junk** `docs-archive/`. | One coherent document set instead of 26 drifting ones |

**Renames:** `tabular_wasm` → **`slm_wasm`** (now only `TransformerSLMModel`);
`ferrum_core::csv` → **`ferrum_core::meta`** (now only `TaskType`,
`ModelMetadata`, `Normalizer`, which the model format needs).

**A latent test bug surfaced.** Ferrum-ML's trained-model tests used a path
relative to the working directory that never resolved under `cargo test`, so
they had **always skipped silently**. Once anchored to the crate directory,
two failed. They probed the housing regressor with every feature set to 0.5
(longitude 0.5, latitude 0.5), far outside the data. They now probe at each
feature's training-range midpoint, and all 78 pass. The model itself was fine:
retrained, it predicts $417k against an actual $453k.

---

## 3. Capability assessment

Reference machine: Intel i5-1135G7 (4 cores / 8 threads), 16 GB RAM, GeForce
MX330 (2 GB). Ferrum never uses the GPU, so the MX330 contributes nothing.
Compute is the binding limit; RAM is not.

**Measured kernel rate:** 35–43 GFLOP/s (8 threads). `target-cpu=native` gains
nothing, because the scalar kernels are the bottleneck.

### 3.1 Training from scratch (native transformer)

| Goal | Reachable size |
|---|---|
| Heavily over-trained, minutes to hours | 0.1M–2M parameters |
| Compute-optimal (≈20 tokens/param) in 24 h | ≈5M |
| Compute-optimal in a week | ≈12–15M |
| Compute-optimal in a month | ≈25–30M |
| Fits in RAM but impractical | ≈400M (batch 8, context 256) |

Measured throughput ranges from 24,700 tokens/s at 0.17M parameters to 193
tokens/s at 33.8M (full table in [Manual §6.7](Manual.md#67-sizing-a-run-for-your-machine)).
Activations, not optimizer state, dominate RAM: the 6.9M model peaks at
1.3 GB. **Below 10M parameters is where this engine works best.**

### 3.2 External models (GGUF)

| | Verdict on this machine |
|---|---|
| Load 1–3B (supported quants) | Yes: about 0.5 GB int4 plus the f32 embedding; 2–3.5 GB peak during import |
| Run 1B | Yes, slowly: about 7 tokens/s int4 decode, about 30 s prefill for 512 tokens |
| Fine-tune (full AdamW, f32, 16 B/param) | SmolLM2-135M comfortable (≈2.2 GB, about 4M tokens/day); Qwen2.5-0.5B tight (≈8 GB + activations); **1.5B and up: no** |
| Train 1B | **No.** About 16 GB of optimizer state, and about 200 days per 100M tokens |

### 3.3 Overall verdict

The engineering quality is high. Zero dependencies, no `unsafe`,
gradient-checked backprop, determinism across thread counts, a self-contained
model format, and a byte-exact GGUF round trip are all real and tested. Before
2026-10-03 the training **pipeline** was the weak point. With the §2.1 fixes,
1–10M-parameter models trained on hundreds of millions of tokens are realistic
on a laptop. A TinyStories-class model (1–3M parameters, about 470M tokens) is
a 1–3 day job.

---

## 4. Implemented and tested

| Subsystem | Module(s) | State |
|---|---|---|
| Tensors and kernels | `tensor`, `ops` | ✅ Fused cache-tiled `linear_forward`; packed `qlinear` int8/int4 with column-split decode GEMV |
| Parallelism | `parallel` | ✅ Persistent `std` worker pool; row and column splits; deterministic; serial on wasm32 |
| Layers | `layer`, `model` | ✅ Linear (+ packed `QWeight`), LayerNorm, Embedding, causal MHA, `KvCache` |
| Quantization | `quant` | ✅ int8 per-tensor and per-channel QAT; in-memory int8/int4 (split-half) |
| Transformer training | `train_transformer` | ✅ Full backprop, Adam/AdamW, data-parallel epochs, dropout, weight tying, clipping, LR schedules, **window stride**, **step-capped epochs with hooks**, checkpoint codec |
| SLM pipeline | `slm` | ✅ Config-driven training (budget, validation + early stop, tokenizer reuse, checkpoint/resume); KV-cached generation with top-k/top-p/repetition/stop strings; streaming; evaluation |
| Tokenizer | `tokenizer` | ✅ Byte-level BPE, **incremental trainer**, whitespace pre-tokenization, rank-based encode, special tokens |
| Data tools | `dataset` | ✅ `clean_corpus`, `corpus_stats`, `validate_for_training` |
| Model format | `meta`, `loader` | ✅ FINF v4/v5; per-vector f32/int8/int4; bounds-checked |
| GGUF reader | `gguf` | ✅ F32/F16/Q8_0/Q8_1/Q4_0/Q4_1/Q4_K/Q5_K/Q6_K; Q2_K/Q3_K (read-only); streamed `open`; IQ* rejected |
| GGUF tokenizer | `gguf_tokenizer` | ✅ BPE exact; SPM decode exact, encode greedy |
| Llama/Qwen runner | `llm` | ✅ RMSNorm, RoPE, GQA, SwiGLU; cached decode equals full forward |
| Llama/Qwen fine-tuning | `llm_train` | ✅ Gradient-checked backprop, AdamW, schedules, QAT, dropout, `.flck` checkpoints, threaded epochs |
| GGUF writer | `gguf_write` | ✅ Byte-exact GGUF v3; encoders verified as inverses of the decoders; atomic writes |
| Chat templates | `chat_format` | ✅ ChatML / Llama-3 / Llama-2 / Zephyr detection, rendering, and stop tokens |
| CLI | `slm_cli` | ✅ train · run · generate · eval · info · run-gguf · finetune-gguf · export-gguf |
| Browser | `slm_wasm` | ✅ KV-cached SLM inference, sampling helpers, attention maps; builds for wasm32 |
| GUI | `ferrum_gui` | ✅ Backend tests and builds. Tabs: Datasets, Train, Generate, Evaluate, Models, GGUF, Fine-tune, Export, System, Capable, plus engine log. |
| CI | `.github/workflows/ci.yml` | ✅ Build/test, rustfmt, clippy, JS syntax, cargo-deny, wasm32, GUI fmt/clippy/test |

---

## 5. Known issues

| Severity | Issue |
|---|---|
| 🟡 | **The Capable tab's training estimates are rough.** They ignore activation memory (measured at about 12× the predicted 16 B/param at small sizes) and attention FLOPs, and they scale a single-thread GEMM rate by 8 hyperthreads on 4 physical cores. Treat them as loose upper bounds. |
| 🟡 | **The GUI is not click-tested after the 2026-10-03 changes.** The backend compiles and its 39 tests pass, and the JS passes syntax checks, but the new Train-tab fields have not been exercised in a running window. |
| 🟡 | **No data pipeline for large corpora.** The corpus is read whole with `read_to_string` and tokens are held as 8-byte `usize`. There is no pre-tokenized file or streaming, and no EOS separators between documents. Gigabyte corpora will hit RAM. |
| 🟡 | **Budgeted runs report progress per epoch.** On a huge corpus one epoch can take hours with no output; `--verbose` shows per-epoch steps. There is no tokens/s, ETA, or gradient-norm readout. |
| 🟢 | Resume restarts the interrupted epoch with a fresh shuffle instead of replaying the exact remaining batches (marked `ponytail:` in `slm.rs`). |
| 🟢 | Native `generate` on the CLI exposes temperature only; top-k, top-p, and repetition penalty are library-only (`generate_with`). |
| 🟢 | GGUF import is lossy (dequantize and re-quantize to Ferrum's per-row grid) and not bit-exact to llama.cpp; RoPE type is hard-wired to `Norm` on import; no RoPE scaling. |

---

## 6. Roadmap

### 6.1 In flight: the eight-upgrades branch

Designed and approved 2026-07-03. Done: **F2** (sampling knobs on the GGUF run
path), **F1** (chat templates), and **F8** (Q2_K/Q3_K read).

| ID | Feature | State |
|---|---|---|
| F8 | Q2_K / Q3_K read support | **Done** (`133cdfe`) |
| F3 | `qwen3` architecture (per-head q/k RMSNorm, explicit head_dim) | Not started |
| F7 | Perplexity evaluation for GGUF models (`eval-gguf`, Evaluate tab) | Not started |
| F5 | Streamed GGUF export (two-pass, checkpoint overlay without loading it whole) | Not started |
| F6 | Safe-SIMD decode (autovectorization-friendly `qlinear` loops) | Not started |
| F4 | GUI CI boot smoke test (`xvfb-run`) | The CI GUI job exists; the boot smoke step is outstanding |

### 6.2 Next: from-scratch SLM gaps (from the 2026-10-03 evaluation)

**P0, remaining blockers**

- A pre-tokenized `u16`/`u32` token file, streamed from disk, with EOS markers
  between documents.

**P1, high leverage**

- **Llama architecture from scratch.** `LlamaModel` (RoPE, RMSNorm, SwiGLU,
  GQA, AdamW, GGUF export) can only be loaded from a file today. Adding
  `LlamaModel::init(cfg, rng)` gives a complete path: train from scratch,
  export GGUF, run in llama.cpp. Every other piece exists. Once it lands, the
  native GPT-1-style stack becomes a candidate for retirement.
- **Faster kernels.** Register-blocked micro-kernels with packed B, and no
  full-weight `Arc` copy per parallel matmul (`ops.rs`). Likely 2–5× without
  `unsafe`, which raises every bound in §3.
- **Gradient accumulation** (effective batch not limited by RAM) and
  activation recomputation for longer contexts.
- **Distillation from a GGUF teacher** (logit KD), the best-known quality
  lever below 10M parameters.
- **SFT support**: chat template plus prompt-loss masking for native models.
- **LoRA adapters**, so 0.5B-class fine-tuning fits in 16 GB.
- **Metrics**: tokens/s, ETA, gradient norm, CSV/JSON loss export,
  bits-per-byte.
- **Size-aware defaults.** The CLI defaults (context 16, embed 32, lr 0.01)
  are toy settings.

**Also open:** exact SPM encode (unigram Viterbi), IQ* quants, finer import
quantization (per-block scales), CLI binary smoke tests, and WASM streaming
bindings.

These are directions, not commitments.

---

## 7. Backward compatibility

- FINF **v4** and **v5** files load. Files from before the tokenizer field
  default to character-level.
- `TransformerConfig` gained fields. Struct literals need
  `..TransformerConfig::default()`. Its **default stride is now
  `context_len`**, so config-driven runs take far fewer steps per epoch than
  before. Set `window_stride: 1` (`--stride 1`) to reproduce an old run.
- The positional `train_transformer*` functions are unchanged (stride 1, same
  results).
- **Removed APIs:** `GenerativeSLM::train`, `train_with_callback`,
  `train_embedded`, `train_embedded_with_callback`, `slm::build_csv_dataset`,
  `ferrum_core::{CsvDataset, CsvRow, fit_normalizer_with_target,
  train_val_split, Net, EmbedT, train_epoch, accuracy}` (the last six now live
  in `ferrum_ml`), and the `ferrum_core::csv` module (now `meta`).
- **Legacy one-hot MLP model files** load and inspect, but `generate` and
  `evaluate` return an error. Embedding-MLP files (token-ID input) still
  generate.
- The GUI `train_slm` command ignores the removed `momentum` field. New fields
  are optional and default sensibly, so older front-ends keep working.

---

## 8. Development history

| Date | Milestone |
|---|---|
| 2026-05-30 | Repository converted into an independent edge generative SLM library; installation and deployment docs |
| 2026-06-11 | Trainable transformer, KV cache, Adam, hardened FINF parsing; compile and latent-bug fixes |
| 2026-06-12 | `train_transformer` CLI; **byte-level BPE** integrated end to end; matmul parallelized across threads, then a **persistent worker pool**; first benchmarks |
| 2026-06-21–22 | Major progress: **GGUF compatibility**: reader, k-quants, tokenizer import, Llama/Qwen runner, gradient-checked `llm_train` |
| 2026-06-23 | Deep project review (1B feasibility pushback); documentation brought up to date |
| 2026-06-23–24 | **Capable** module: machine micro-benchmarks, capability report, GGUF over-budget warning |
| 2026-07-01 | **Full AdamW GGUF fine-tuning** (CLI + GUI); lint tooling (clippy, rustfmt, cargo-deny, JS syntax) gating CI; SPM/BPE speedups; **GGUF export**: `GgufBuilder`, all nine encoders, quantization policy, `export-gguf` |
| 2026-07-02 | Export hardening and audit fixes; **GUI Export tab**; **Capable v2** (load/train/fine-tune/run parameter ranges) |
| 2026-07-03 | Eight-upgrades design and plan; **F2** sampling knobs; **F1** chat templates |
| 2026-10-03 | Evaluation for SLM training on 2 GB VRAM / 16 GB RAM hardware; **training-pipeline fixes** (stride, budgets, incremental BPE, tokenizer reuse, schedule/clip/tie/validation/checkpoint wiring); **restructuring** into Ferrum / Ferrum-ML / Ferrum-Junk; documentation consolidated into three files |

**Design decisions worth keeping** (from the archived specs):

- `#![forbid(unsafe_code)]` stays, by user decision, even where `unsafe` would
  buy speed (mmap, intrinsics).
- Zero external dependencies in the engine; optimizations must be achievable
  with safe, autovectorizable Rust.
- Chat templates are fingerprinted, not interpreted: there is no Jinja engine,
  and single-turn only until asked.
- Capability estimates are grounded in a live micro-benchmark, not static
  tables.
- GGUF export uses one quantization policy shared by every writer path, and
  the encoders are tested as exact inverses of the decoders.

---

## 9. Metrics

_As of commit `6dcb629` (2026-10-03), excluding `target/`._

| Metric | Value |
|---|---:|
| Workspace crates | 4 (`ferrum_core`, `slm_cli`, `slm_wasm`, `tests`) + `ferrum_gui` (excluded) |
| `ferrum_core` source files / lines | 25 / 20,022 (including unit tests) |
| `ferrum_core` `pub fn` | 287 |
| External dependencies (`ferrum_core`) | 0 |
| `unsafe` blocks | 0 (forbidden) |
| Tests: core unit / integration / wasm / doc | 360 / 52 / 3 / 3 → **418 pass** |
| Tests: `ferrum_gui` backend | 39 pass |
| Sibling Ferrum-ML | 2,088 lines, 78 tests pass |
| Markdown documents | 3 (from 26) |
