# Ferrum — Project Review

_Date: 2026-06-21 · Reviewer: automated deep review · Target machine: Intel
i5-1135G7 (4 cores / 8 threads), 16 GB RAM._

This document reviews the `ferrum` project as it now stands: a **zero-dependency,
pure-Rust, CPU-only engine** for building, training, and running causal
Transformers, Small Language Models (SLMs), and classical MLPs. It covers the
architecture, the state of each subsystem, what was recently hardened, the gaps
that remain, and a quantitative **Metrics** section including a code-coverage
report.

---

## 1. Bottom line

Ferrum is a genuinely self-contained ML stack — tensors, hand-written backprop,
attention, a byte-level BPE tokenizer, int8 quantization-aware training, a
self-describing model file format (FINF), a multi-threaded matmul, a WASM build,
two CLIs, and a Tauri GUI — built on **`std` only**, with an empty
`[dependencies]` table for the core crate and `#![forbid(unsafe_code)]`.

The engine is **correct, well-tested, and reproducible**. All three SLM training
paths and the tabular MLP path train and reduce loss; quantization round-trips
with small drift; generation (char-level + BPE), streaming, KV-cached inference,
and held-out perplexity evaluation work from the library, the CLIs, the GUI, and
WASM. Multi-threaded training is bit-for-bit reproducible and identical to serial
at one shard.

The binding constraint remains **compute, not features or memory**: this is a
scalar CPU engine, so wall-clock time caps practical training at a few million
parameters even though RAM would allow far more. That is a property of the design
goal (edge / offline / dependency-free), not a defect.

Severity legend: 🔴 will break/block at modest scale · 🟡 limits quality/scale ·
🟢 polish/robustness.

---

## 2. Architecture

```
                ┌──────────────────────────────────────────────┐
                │                 ferrum_core                  │
                │  (zero-dep, std-only, #![forbid(unsafe)])    │
                │                                              │
   tensor ─ ops ─ activation ─ loss ─ rng ─ parallel          │
        │                                                     │
   layer (Linear, LayerNorm, Embedding, TransformerBlock,     │
        │   KvCache, Flatten, Activation)                     │
        │                                                     │
   model (Sequential) ── quant (int8, per-channel) ── tokenizer (BPE)
        │                                                     │
   train (MLP/embedded) ── train_transformer (decoder-only)   │
        │                                                     │
   slm (GenerativeSLM: train / generate / evaluate / sample)  │
        │                                                     │
   csv ── dataset ── loader (FINF v4/v5) ── verbose ── error  │
                └───────────────┬──────────────────────────────┘
        ┌───────────────────────┼───────────────────────┬───────────────┐
   slm_cli                 train_cli              tabular_wasm       ferrum_gui
 (transformer SLM)      (tabular MLP)         (wasm-bindgen)        (Tauri app,
                                                                    excluded crate)
```

- **Data flow for an SLM:** corpus → `tokenize_for_lm` (char or BPE) →
  `TransformerNet` (QAT, Adam, optional LR schedule / grad clip / weight tying) →
  `to_inference()` → `Sequential` → FINF bytes → `GenerativeSLM::generate*`
  (KV-cached, with top-k/top-p/repetition-penalty sampling).
- **Determinism** is a first-class property: a seeded `Rng` (xorshift64\*),
  fixed shard-reduction order, and a stable parameter-ordering convention make
  training and generation reproducible and serial≡threaded-at-one-shard.

---

## 3. What works today (verified)

- **All four training paths** (`GenerativeSLM::train` one-hot MLP,
  `train_embedded`, `train_transformer`, and the multi-threaded transformer
  variant) train and reduce loss.
- **Quantization-aware training** + **per-channel** int8 FINF v5 serialization
  round-trips with small drift.
- **KV-cached generation** for native (CLI + GUI) and WASM, char-level and BPE,
  matching a full forward within the context window.
- **Validation-aware training** with held-out split, early stopping on
  perplexity, and best-epoch checkpointing.
- **Checkpoint / resume** of full training state (weights + Adam moments + step +
  RNG), bit-identical to an uninterrupted run.
- **Byte-level BPE** with whitespace pre-tokenization and rank-based encode,
  round-tripping arbitrary UTF-8.
- **FINF loader** is defensively coded: magic/version checks, bounds-checked
  reads, `checked_mul` against overflow, graceful rejection of corrupt buffers.

---

## 4. Subsystem review

| Subsystem | File(s) | State | Notes |
|---|---|---|---|
| Tensors / ops | `tensor.rs`, `ops.rs` | ✅ Solid | Row-major f32, parallel matmul variants; high coverage. |
| Activations / loss | `activation.rs`, `loss.rs` | ✅ Solid | Softmax-CE with stable log-sum-exp; MSE. |
| Layers | `layer.rs` | ✅ Solid | Linear, LayerNorm, Embedding (+`embed_one`), causal MHA, `KvCache`, `forward_with_cache`. |
| Optimizers | `optim.rs` | ✅ Strong | SGD+momentum, Adam, **global-norm grad clipping**, **warmup+cosine/linear LR schedule**. |
| MLP trainer | `train.rs` | ✅ Solid | One-hot + embedded LM MLP, QAT, grad clip, shuffle-without-replacement. |
| Transformer trainer | `train_transformer.rs` | ✅ Strong | Full backprop; **data-parallel sharing read-only weights** (no per-shard clone); **weight tying**; **checkpoint/resume**; per-channel QAT. Largest, best-covered module. |
| SLM API | `slm.rs` | ✅ Strong | Train/generate/evaluate; **KV-cached** generation; **top-k/top-p/repetition penalty**; **validation+early-stop**. |
| Quantization | `quant.rs` | ✅ Strong | Per-tensor **and per-channel** int8 (outlier isolation). |
| Tokenizer | `tokenizer.rs` | ✅ Strong | Byte-level BPE, **whitespace pre-tokenization**, **rank-based encode**. |
| Serialization | `loader.rs` | ✅ Strong | FINF v4/v5; per-vector encoding markers incl. **per-channel int8**. |
| CSV / dataset | `csv.rs`, `dataset.rs` | ✅ Solid | Parsing, normalization, corpus cleaning. |
| Concurrency | `parallel.rs` | ✅ Solid | std-thread pool; dynamic core detection; deterministic. |
| WASM | `tabular_wasm` | ✅ Works | KV-cached `TransformerSLMModel`; now CI-built for `wasm32`. |
| CLIs | `slm_cli`, `train_cli` | ✅ Works | Not exercised by automated tests (0% coverage — see Metrics). |
| GUI | `ferrum_gui` | ✅ Works | Tauri + vanilla JS; backend command tests + a Node UI test; CI job added. |

---

## 5. Recently hardened (closed gaps)

The following items from the prior review have been implemented and tested:

- **T1 — gradient clipping** (global-norm) in all training loops.
- **T2 — LR schedule** (linear warmup + cosine/linear decay).
- **T3 — data-parallel memory fix:** workers borrow read-only weights and
  allocate only a flat gradient buffer instead of deep-cloning the whole net.
- **T4 — true epochs:** per-epoch Fisher–Yates shuffle, minibatches drawn
  without replacement.
- **T5 — validation split + early stopping + best-epoch checkpoint.**
- **T6 — checkpoint/resume** of full training state (weights, Adam moments, step,
  RNG); bit-identical resume.
- **T9 — weight tying** (embedding ↔ LM head) with a verified folded gradient.
- **I1 — KV cache wired into native generation** (char + BPE), O(context)/token.
- **I2 — sampling controls:** top-k, top-p (nucleus), repetition penalty.
- **§7 — per-channel int8 quantization** end-to-end (QAT + FINF v5 marker).
- **K1 — rank-based BPE encode**; **K2 — whitespace pre-tokenization.**
- **G1 — streamed-generation tail reconciliation; G2 — HTTP download timeouts.**
- **X1 — GUI backend + UI tests and a dedicated CI job; X2 — wasm32 CI build.**

---

## 6. Remaining gaps & recommendations

**Quality / training:**

- 🟡 **T7 — no weight decay (AdamW) / dropout / regularization.** With small
  corpora the model memorizes (train-perplexity ≈ 1.0 vs held-out ≈ 3.7). Adding
  AdamW-style decoupled weight decay and optional dropout would improve
  generalization. _Recommended next._
- 🟢 **T8 — embedded & one-hot paths are single-threaded** (only the transformer
  has the data-parallel epoch).
- 🟢 **T10 — whole corpus tokenized into RAM**; no streaming/chunked dataset.

**Inference:**

- 🟡 **I3 — no EOS / stop criterion** (and **K3 — no BOS/EOS/PAD/UNK special
  tokens**). Generation always runs the full character budget; multi-document
  corpora have no separator. These two are best done together.
- 🟢 **I4 — probabilities are softmaxed in-model then `ln()`-ed back** for
  temperature; exposing raw logits from the head would avoid the `1e-12` floor.
- 🟢 **I5 — no batched generation** (one sequence at a time).

**Tokenization:**

- 🟢 Pre-tokenization splits on whitespace only; GPT-2 additionally splits
  punctuation and attaches leading spaces to words. A small refinement.

**Testing / CI / GUI:**

- 🟢 **X3 — no fuzz/adversarial tests** for the loader & tokenizer `from_state`
  beyond the existing negative cases.
- 🟢 **X4 — integration tests are slow** (the BPE suite trains in debug, ~50 s).
  Consider tinier configs or a `--release` test profile.
- 🟢 **CLIs are untested** (`slm_cli`, `train_cli` at 0% coverage); a couple of
  end-to-end smoke tests would catch argument/wiring regressions.
- 🟢 Minor GUI items remain (process-wide verbose flag, basic embedded shell
  semantics, CSP disabled for local use).

None of these block the current, working workflow; they are the path to larger,
higher-quality local models and tighter CI.

---

## 7. Maximum SLM size on this machine

Unchanged by the recent work (these are compute-bound limits):

| Budget | `P × token-forwards` ≤ | Example |
|-------|-----------------------:|---------|
| ~15 min | ~1 × 10¹² | 1 M params × (small corpus, few epochs) |
| Overnight (~8 h) | ~3 × 10¹³ | ~1–2 M params on a ≤1 MB corpus, tens of epochs |
| Multi-day | ~10¹⁴ | ~5–8 M params, small corpus only |

**Practical answer:** aim for **≤ ~2 M parameters** for comfortable local
training and responsive generation; **~8 M is the upper bound** for patient,
small-corpus runs. The **T3 memory fix** removed the ~8× per-shard blow-up, so
threaded training is no longer the memory ceiling it once was; the **I1 KV
cache** makes generation O(context) per token.

---

## 8. Metrics

_Generated 2026-06-21 from the working tree (excludes `target/`)._

### 8a. Codebase size

| Metric | Value |
|---|---:|
| Workspace crates | **6** (5 in-workspace + `ferrum_gui` excluded) |
| Rust source files | **37** |
| Rust lines of code (incl. tests) | **14,861** |
| └ `ferrum_core` | 11,101 |
| └ `tests` (integration) | 1,858 |
| └ `ferrum_gui` (Rust) | 773 |
| └ `slm_cli` / `train_cli` | 436 / 253 |
| └ `tabular_wasm` | 440 |
| `ferrum_core` library modules | **19** (+ `lib.rs`) |
| JavaScript (GUI frontend) | 5 files, 1,081 lines |
| HTML/CSS (GUI) | 4 files |
| Markdown docs | 17 files |

### 8b. API surface & code health (`ferrum_core`)

| Metric | Value |
|---|---:|
| `pub fn` | **198** |
| `pub struct` / `pub enum` / `pub trait` | 32 / 4 / 1 |
| Total `fn` (workspace) | 764 |
| Doc-comment lines (`///`) | 845 |
| External dependencies | **0** (`std` only) |
| `unsafe` blocks | **0** (`#![forbid(unsafe_code)]`) |
| FINF format versions | v4 (f32), v5 (int8 per-tensor + per-channel) |

### 8c. Tests

| Suite | `#[test]` functions |
|---|---:|
| `ferrum_core` (unit) | **249** |
| `tests` crate (integration) | **94** |
| `tabular_wasm` | 4 |
| `ferrum_gui` (backend) | 4 |
| **Total Rust tests** | **351** |
| Node UI test (`stream.test.js`) | 1 file, 9 assertions |

All 351 Rust tests pass; native workspace build is warning-free.

---

## 9. Code coverage (llvm-cov)

Generated with `cargo llvm-cov --workspace --summary-only`. This is the
Markdown translation of the llvm-cov HTML report. `ferrum_gui` is excluded (it is
not part of the workspace and is covered by its own CI job); the CLI binaries
show 0% because their `main()` entry points are not driven by automated tests.

### 9a. Totals

| Metric | Covered | Total | Coverage |
|---|---:|---:|---:|
| **Lines** | 7,017 | 8,151 | **86.09%** |
| **Regions** | 14,602 | 17,131 | **85.24%** |
| **Functions** | 811 | 943 | **86.00%** |

### 9b. Per-file

| File | Regions | Region cov. | Functions | Function cov. | Lines | Line cov. |
|---|---:|---:|---:|---:|---:|---:|
| `ferrum_core/src/activation.rs` | 124 | 92.74% | 13 | 100.00% | 73 | 94.52% |
| `ferrum_core/src/csv.rs` | 1210 | 92.56% | 82 | 84.15% | 658 | 92.86% |
| `ferrum_core/src/dataset.rs` | 319 | 98.43% | 26 | 100.00% | 201 | 98.01% |
| `ferrum_core/src/error.rs` | 114 | 98.25% | 13 | 100.00% | 61 | 100.00% |
| `ferrum_core/src/layer.rs` | 1861 | 89.90% | 106 | 94.34% | 859 | 92.67% |
| `ferrum_core/src/loader.rs` | 1613 | 91.51% | 67 | 89.55% | 667 | 94.15% |
| `ferrum_core/src/loss.rs` | 327 | 87.77% | 11 | 100.00% | 173 | 85.55% |
| `ferrum_core/src/model.rs` | 182 | 88.46% | 19 | 100.00% | 89 | 95.51% |
| `ferrum_core/src/ops.rs` | 536 | 97.20% | 32 | 100.00% | 236 | 97.03% |
| `ferrum_core/src/optim.rs` | 502 | 95.02% | 25 | 100.00% | 241 | 96.68% |
| `ferrum_core/src/parallel.rs` | 181 | 92.27% | 13 | 100.00% | 100 | 91.00% |
| `ferrum_core/src/quant.rs` | 294 | 99.66% | 26 | 100.00% | 140 | 99.29% |
| `ferrum_core/src/rng.rs` | 191 | 92.67% | 23 | 91.30% | 100 | 91.00% |
| `ferrum_core/src/slm.rs` | 2236 | 88.77% | 107 | 90.65% | 1214 | 89.95% |
| `ferrum_core/src/tensor.rs` | 161 | 96.27% | 20 | 95.00% | 93 | 96.77% |
| `ferrum_core/src/tokenizer.rs` | 562 | 96.44% | 43 | 97.67% | 262 | 96.56% |
| `ferrum_core/src/train.rs` | 1472 | 89.67% | 81 | 95.06% | 691 | 91.75% |
| `ferrum_core/src/train_transformer.rs` | 3538 | 96.50% | 130 | 98.46% | 1485 | 97.98% |
| `ferrum_core/src/verbose.rs` | 83 | 6.02% | 9 | 11.11% | 51 | 5.88% |
| `slm_cli/src/main.rs` | 554 | 0.00% | 28 | 0.00% | 291 | 0.00% |
| `tabular_wasm/src/lib.rs` | 648 | 44.44% | 57 | 33.33% | 297 | 37.71% |
| `train_cli/src/main.rs` | 423 | 0.00% | 12 | 0.00% | 169 | 0.00% |
| **TOTAL** | **17131** | **85.24%** | **943** | **86.00%** | **8151** | **86.09%** |

### 9c. Coverage notes

- The **engine modules** (`ferrum_core`) are well covered: the two largest and
  most safety-critical — `train_transformer.rs` (97.98% lines) and `loader.rs`
  (94.15% lines) — are among the best tested. `quant.rs` is at 99.29%.
- `verbose.rs` (5.88%) is logging plumbing gated behind a runtime flag that tests
  leave off — low value to cover.
- The two **CLI binaries** (0%) and `tabular_wasm` (37.71%) drag the workspace
  total down; their logic is thin wrappers over the well-covered core, but a few
  end-to-end smoke tests (CLIs) and a `wasm-bindgen-test` pass (WASM) would lift
  these and guard the integration seams.

---

## 10. Prioritized recommendations

1. **AdamW weight decay + optional dropout (T7)** — the biggest remaining quality
   lever for small-corpus generalization.
2. **EOS / special tokens (I3 + K3)** — natural stopping and document boundaries.
3. **CLI smoke tests + `wasm-bindgen-test`** — close the two biggest coverage
   gaps and the untested integration seams.
4. **Faster test profile (X4)** and **loader/tokenizer fuzzing (X3)** — CI
   ergonomics and hardening.
5. **Streaming dataset (T10)** and **threaded MLP epoch (T8)** — only if you push
   toward larger corpora/models.

The engine is in good shape: correct, dependency-free, reproducible, and
well-tested. The path forward is generalization (regularization), generation
ergonomics (stop tokens), and lifting coverage on the thin outer layers.
