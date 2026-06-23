# Ferrum — Project Review

_Date: 2026-06-23 · Reviewer: automated deep review · Target ("current") hardware:
Intel i5-1135G7 (4 cores / 8 threads), 16 GB RAM, no GPU._

This reviews `ferrum` as a **zero-dependency, pure-Rust, CPU-only** engine for
building, training, and running causal Transformers, Small Language Models (SLMs),
and classical MLPs — now also with a std-only **GGUF reader** and a **Llama/Qwen
decoder** (`llm.rs`) that imports and runs externally pretrained checkpoints in
int4/int8/f32, plus a gradient-checked backward pass for that architecture
(`llm_train.rs`).

The engine has always been excellent at its core job. The consequential newer
surface is GGUF/LLM, so this review leads with it and — as the project's brief
demands — **pushes back hard on loading, training, or running 1B+ models on this
hardware** (§4). The short version: the GGUF/LLM *primitives* are real, correct,
and well-tested; importing is lossy and not bit-exact to llama.cpp; and the
hardware makes 1B the painful upper edge of "runnable" and flatly outside
"trainable."

Severity legend: 🔴 blocks the stated use · 🟡 limits quality/scale/usability ·
🟢 polish.

---

## 1. Bottom line

- **As an edge SLM engine for models you train yourself (≤ a few M params):**
  correct, reproducible, well-tested, genuinely dependency-free. This is its real
  job and it does it well (§5–6).
- **As a GGUF runner:** the math is right and unit-covered. **Q4_K/Q5_K/Q6_K**
  k-quants load (so the common `_K_M` downloads work), alongside legacy
  `F32/F16/Q8_0/Q8_1/Q4_0/Q4_1`; `Q2_K/Q3_K` and `IQ*` are rejected. The
  checkpoint's **own tokenizer is imported** (BPE exact, SPM encode greedy), so it
  runs on text. `slm_cli run-gguf` and a GUI panel drive it, with a `/proc/meminfo`
  memory guard and a `--quant f32` path that skips re-quantization. Import is still
  **lossy by default** and **not bit-exact to llama.cpp** — the code says so.
- **As a place to run a 1B model:** functional, marginal, and **measured**
  (benchmarks.md §3): after the int4-kernel fix, **~7 tok/s decode at int4
  (~4–5× the f32 rate)** at 8 threads, int8 marginally faster per call, with tens of
  seconds of compute-bound prefill per real prompt. A patient demo, not a chatbot.
- **As a place to train or fine-tune a 1B model:** the architecture is now
  **trainable** — `llm_train.rs` adds a finite-difference-checked backward pass — but
  training a *1B* model here is **still not feasible**: no RAM for the optimizer
  state (~16 GB at f32) and no compute for the FLOPs (§4.3). The missing capability
  (gradients) now exists; the hardware ceiling does not move.

To its credit, the GGUF/LLM module documents its limits in its own doc-comments.
This review's job is to keep them from getting lost behind the word
"compatibility."

---

## 2. Architecture

```
                ┌────────────────────────────────────────────────────┐
                │                    ferrum_core                       │
                │     (zero-dep, std-only, #![forbid(unsafe_code)])    │
                │                                                      │
   tensor ─ ops ─ activation ─ loss ─ rng ─ parallel                  │
        │     (matmul/qlinear; persistent CPU worker pool; run_1d)     │
   quant (int8 + int4 QWeight, per-tensor/per-channel, consumed packed)│
        │                                                              │
   layer (Linear[+QWeight], LayerNorm, Embedding, TransformerBlock,    │
        │   KvCache)                                                   │
        │                                                              │
   ┌── train / train_transformer ──┐     ┌── llm ─────────────────────┐ │
   │  Ferrum's OWN architecture:   │     │  Llama/Qwen architecture:  │ │
   │  learned-pos + LayerNorm +    │     │  RMSNorm + RoPE + GQA +    │ │
   │  ReLU FFN. Trainable (backprop)│     │  SwiGLU. Run + train       │ │
   │                                │     │  (llm_train, grad-checked) │ │
   └───────────────┬───────────────┘     └─────────────┬──────────────┘ │
        │ slm (GenerativeSLM: train/generate/evaluate)  │ gguf:          │
        │ tokenizer (byte-level BPE)                     │ std-only GGUF  │
        │ loader (FINF v4/v5; f32/int8/int4 per-vector)  │ reader →       │
        │                                                │ LlamaModel     │
                └───────────────┬──────────────────────────────────────┘
        ┌───────────────────────┼───────────────────────┬───────────────┐
   slm_cli                 train_cli              tabular_wasm       ferrum_gui
 (transformer SLM        (tabular MLP)         (wasm-bindgen)        (Tauri app,
  + run-gguf)            ── no GGUF ──          ── no GGUF ──        + GGUF panel)
```

There are **two distinct Transformer stacks** in the tree, and conflating them is
the single biggest source of over-claiming:

| | Ferrum's own (`train_transformer.rs`) | Imported (`llm.rs`) |
|---|---|---|
| Positions | learned absolute embedding | **RoPE** (rotary) |
| Norm | LayerNorm (mean+var, bias) | **RMSNorm** (no mean, no bias) |
| FFN | ReLU MLP | **SwiGLU** gated |
| Attention | dense MHA | **grouped-query (GQA)** |
| Training | ✅ full backprop, Adam/AdamW, QAT | ✅ **gradient-checked backprop** (`llm_train.rs`, f32) |
| Source | trained from your text | imported from a GGUF |
| Tokenizer | Ferrum byte-level BPE (in-file) | ✅ imported (`gguf_tokenizer.rs`; BPE exact, SPM approx) |

They share the low-level kernels (`Linear`/`qlinear`, the worker pool, the quant
grid) but nothing above that. You **can** now train an imported model (f32, via
`llm_train.rs`) — though not at 1B scale — and you **can** run a downloaded Llama
with its own tokenizer.

---

## 3. The GGUF / LLM subsystem — detailed

### 3a. What is genuinely there and correct

- **`gguf.rs` (1,526 lines, 26 tests).** A pure-`std`, `unsafe`-free GGUF v2/v3
  reader: magic/version check, the full typed metadata key/value table, the tensor
  directory, alignment handling, and block-dequantizers for **F32, F16, Q8_0, Q8_1,
  Q4_0, Q4_1** plus the **Q4_K/Q5_K/Q6_K** super-block k-quants. Defensively coded —
  checked offset arithmetic, EOF guards, rejects nested arrays and absurd counts.
  `Gguf::from_path` reads the whole file; **`Gguf::open` streams** tensor bytes from a
  `Mutex<File>` on demand (a safe-Rust alternative to `mmap`, which would need
  `unsafe`). A hand-rolled in-memory writer in the tests exercises the reader,
  including a synthetic 2-layer llama/qwen2 file imported and run end-to-end.
- **`gguf_tokenizer.rs` (505 lines, 6 tests).** Reconstructs a checkpoint's own
  tokenizer from `tokenizer.ggml.*`: BPE encode/decode (exact), SPM decode (exact),
  SPM encode (greedy longest-match approximation). This is what lets imported models
  run on **text** rather than raw IDs.
- **`llm.rs` (799 lines, 17 tests).** A real Llama/Qwen2 decoder: `RmsNorm`,
  `apply_rope` (both `Norm` interleaved and `Neox` split-half conventions),
  grouped-query `Attention` with a per-layer KV cache, the `SwiGLU` `FeedForward`,
  the pre-norm `LlamaBlock`, and `LlamaModel` with a full-sequence forward and an
  O(context)/token cached decode plus sampling `generate`. Correctness is checked two
  ways: each primitive against its closed-form definition, and the **cached decode
  path against an independent full-attention implementation, row-for-row**.
- **`llm_train.rs` (821 lines, 6 tests).** A hand-derived backward pass for the
  imported architecture — RMSNorm, RoPE, GQA+softmax, SwiGLU, embedding, LM head —
  each **checked against finite differences**, with next-token cross-entropy and an
  SGD `train_step`. `LlamaTrainer::new` rejects quantized models (there is no f32
  master behind a packed `QWeight`).
- **The int4/int8 path it rides on is real.** Projections are `Linear`s holding an
  `Option<Arc<QWeight>>`; when quantized they dispatch to `ops::qlinear`, which
  consumes packed weights **without expanding to f32**, folds the per-row scale into
  the activations once, and — for single-token decode (`m == 1`) — splits the GEMV
  across the worker pool **by output column** (`run_1d`). Deterministic across thread
  counts, and unit-checked against an f32 reference.

This is careful, honest work. None of the push-back below is about the math.

### 3b. What still limits the importer

- 🟡 **Lossy, coarser quantization.** A `Q4_0`/k-quant GGUF is already quantized;
  Ferrum dequantizes to f32 and **re-quantizes** to its own grid (per-row scale).
  That is a second lossy step *and* a coarser grid than GGML's per-block scales.
  Expect measurably worse output than the same file in llama.cpp; the code does not
  claim bit-exact parity. `--quant f32` avoids the *second* quantization (at full
  RAM) but not the first dequant.
- 🟡 **Remaining quant formats.** `Q2_K`, `Q3_K`, and the `IQ*` families reject.
  Q4_K/Q5_K/Q6_K cover most `_K_M` downloads, but not all.
- 🟡 **f32 token embedding stays resident.** `LlamaModel` keeps `tok_emb: Vec<f32>`,
  so for a large-vocab model it is the single biggest resident array (§4.1) — the
  packed-int4 weight figure does not include it.
- 🟡 **SPM encode is approximate.** SentencePiece `encode` is greedy longest-match
  (decode is exact); a unigram-Viterbi pass would make it token-for-token correct.
  BPE encode is already exact.
- 🟢 RoPE is hard-wired to `Norm` in `load_llama` (correct for llama.cpp's GGUF
  permutation); `rope_type` is plumbed through config but ignored on import — a
  latent foot-gun if a `Neox`-permuted GGUF ever appears.
- 🟢 No RoPE scaling (YaRN / linear / NTK); long-context variants would use the base
  frequency only.

---

## 4. Pushing back hard: 1B+ models on this hardware

> Numbers below derive from the project's **own** measurements in `benchmarks.md §3`
> (scalar kernels, ~36 GFLOP/s peak across 8 cores at 2048², ~6–8 GB/s f32 decode
> bandwidth, ~7 tok/s int4 decode) plus first-principles memory math. The machine is
> 4c/8t, 16 GB, no GPU, no SIMD intrinsics in the kernels.

Useful constants: forward ≈ **2·N FLOPs/token** (N = parameters); decode also
**streams every weight once per token** (the bandwidth wall). A training step ≈
**6·N·T FLOPs** and needs **~16 bytes/param** resident (f32 weight + grad + Adam m +
v) before activations/KV/batch.

### 4.1 Loading a 1B model — feasible at int4, but gated and lossy

Memory is **not** the blocker at int4. A 1B model packs to ~0.5 GB int4; the f32
token embedding adds ~0.26 GB (32k vocab) to ~1 GB (128k vocab); during import you
transiently hold the source file (Q4_0 ≈ 0.55 GB, Q8_0 ≈ 1.05 GB, F16 ≈ 2 GB) plus
per-tensor f32 dequant buffers plus, for a tied head, a transposed f32 copy of the
embedding. Peak ≈ 2–3.5 GB. **That fits in 16 GB.** What gates loading is §3b: the
file must be a supported quant (no `Q2_K`/`Q3_K`/`IQ*`) and a llama/qwen2
architecture, and even then you get a doubly-quantized model.

> The 7B story differs: an F16 source (~14 GB read) plus transients will not fit in
> 16 GB. **1B/3B are the realistic ceiling** for this no-mmap import path.

### 4.2 Running a 1B model — functional, marginal (measured)

On the synthetic ~1B config (benchmarks.md §3, re-measured 2026-06-23):

| Phase | Bound | Measured on this machine (1B) |
|---|---|---|
| **Prefill** a 512-tok prompt | compute, ~2·N·T ≈ 1.0×10¹² FLOP | **~30 s** at ~36 GFLOP/s (8 threads) |
| Prefill a 2048-tok prompt | compute | **~2 min** |
| **Decode** (per token), f32 | bandwidth, serial | ~3.5 GB/tok → **~1.6 tok/s** |
| **Decode** (per token), int4 | bandwidth + nibble unpack, column-split | **~7 tok/s** (8 threads) — ~4–5× f32 |
| **Decode** (per token), int8 | bandwidth, column-split | marginally faster per call, 2× the RAM |

The **split-half int4 repack** restored int4 to ~4–5× the f32 rate, near the ⅛-byte
bandwidth ideal (an interleaved nibble layout had defeated the autovectorizer).
These figures *exclude* attention score·V, layernorm, softmax, RoPE, and sampling,
and the bench replays one layer's weights (cache-warm), so a real model is somewhat
slower; attention is also O(context)/token, so it degrades as the chat grows.

**Verdict:** a 1B model loads and emits text (its tokenizer is imported), but expect
~30 s before the first token and a ~7 tok/s stream after. A patient demo; not a
chatbot.

### 4.3 Training / fine-tuning a 1B model — gradients now exist, the hardware still doesn't

The **backward pass now exists** (`llm_train.rs`): gradients for every primitive,
finite-difference-checked, with an SGD `train_step` that demonstrably reduces loss.
So the capability gap is closed — you can fine-tune a **small** imported model. But
training a **1B** model remains blocked two independent ways, **either fatal**:

1. 🔴 **Optimizer state does not fit.** At ~16 bytes/param, a 1B model needs **~16
   GB** for f32 weights + gradients + Adam moments — the whole RAM budget, before a
   single activation or minibatch. (`LlamaTrainer::new` requires f32 masters, for
   exactly this reason.)
2. 🔴 **Compute is off by orders of magnitude.** A *tiny-by-LLM-standards* 100M-token
   pass over a 1B model is ~6×10¹⁷ FLOPs → **~200 days** at ~36 GFLOP/s. Real
   pretraining (trillions of tokens) is astronomically further out.

For Ferrum's **own** trainable architecture the practical ceiling is far below 1B:

| Budget | `P × token-forwards` ≲ | Practical model |
|---|---:|---|
| ~15 min | ~10¹² | ~1 M params, small corpus |
| Overnight (~8 h) | ~3×10¹³ | ~1–2 M params, ≤1 MB corpus |
| Multi-day | ~10¹⁴ | ~5–8 M params, small corpus |

**Aim for ≤ ~2 M trainable params on this machine; ~8 M is the patient upper bound.**
"Train a 1B model" is outside the design by 3–5 orders of magnitude on every axis.

### 4.4 One-line summary of §4

> **Load:** yes for supported-quant 1–3B, with a lossy double-quant and an f32
> embedding resident. **Run:** yes but slow (~30 s prefill, ~7 tok/s int4 decode).
> **Train/fine-tune:** small models yes; **1B no** — not for RAM, not for compute.

---

## 5. What works today (verified this session)

`cargo build --workspace` is clean and `cargo test --workspace` is **green and
warning-free**: **342** `ferrum_core` unit tests, **95** integration tests, **4**
WASM, **2** doc-tests — **443 tests, 0 failures, 0 warnings** (plus **14**
`ferrum_gui` backend tests in the excluded crate → 457 total).

- **All four training paths** (`train` one-hot MLP, `train_embedded`,
  `train_transformer`, threaded transformer) train and reduce loss; threaded training
  is bit-identical to serial at one shard.
- **QAT + int8 (per-tensor and per-channel) + int4 (split-half)** round-trip through
  FINF v4/v5 with bounded drift.
- **KV-cached generation** (native + WASM, char + BPE) matches a full forward in the
  context window; sampling has top-k / top-p / repetition-penalty controls.
- **Determinism, re-verified:** generation output is byte-identical across thread
  counts; training is byte-identical for a fixed configuration and when only the
  matmul pool varies (the data-parallel shard count, being a different
  floating-point summation grouping, may change the low bits — deterministic per
  shard count, identical to serial at one shard).
- **GGUF import + Llama/Qwen2 decode + a synthetic end-to-end generate** run, with the
  cached path cross-checked against the full forward; the `llm_train` backward pass is
  finite-difference-checked.

---

## 6. Subsystem review

| Subsystem | File(s) | State | Notes |
|---|---|---|---|
| Tensors / ops | `tensor.rs`, `ops.rs` | ✅ Solid | f32 row-major; fused+cache-tiled `linear_forward`; `qlinear` (int4/int8, column-split GEMV). |
| Quantization | `quant.rs` | ✅ Strong | per-tensor & per-channel int8 QAT; `QWeight` in-memory int4/int8 consumed packed; int4 split-half. |
| Parallelism | `parallel.rs` | ✅ Solid | persistent std worker pool; row split + column split (`run_1d`); deterministic; serial on wasm. |
| Layers | `layer.rs` | ✅ Solid | `Linear` carries optional `Arc<QWeight>`; LayerNorm, Embedding, causal MHA, `KvCache`. |
| Optimizers | `optim.rs` | ✅ Strong | SGD+momentum, Adam, **AdamW** decay, global-norm clip, warmup+cosine/linear schedule. |
| MLP / Transformer trainers | `train.rs`, `train_transformer.rs` | ✅ Strong | full backprop; data-parallel epoch; FFN dropout; weight tying; checkpoint/resume; per-channel QAT. |
| SLM API | `slm.rs` | ✅ Strong | train/generate/evaluate; KV-cached; sampling controls; validation+early-stop; streaming. |
| Tokenizer | `tokenizer.rs` | ✅ Strong | byte-level BPE, whitespace pre-tok, rank-based encode, special tokens. |
| Serialization | `loader.rs` | ✅ Strong | FINF v4/v5; per-vector f32 / int8 / int4 (per-tensor & per-channel); bounds-checked. |
| **GGUF reader** | `gguf.rs` | ✅ Real, partial coverage | legacy + **Q4_K/Q5_K/Q6_K**; streamed `open`; tokenizer-adjacent metadata parsed; **no Q2_K/Q3_K/IQ**. (§3) |
| **GGUF tokenizer** | `gguf_tokenizer.rs` | ✅ Solid | BPE exact, SPM decode exact / encode greedy. |
| **Llama/Qwen runner** | `llm.rs` | ✅ Correct | RMSNorm/RoPE/GQA/SwiGLU, cached decode == full forward; lossy import. |
| **LLM training** | `llm_train.rs` | ✅ New | gradient-checked backprop + SGD `train_step`; rejects quantized. |
| CSV / dataset | `csv.rs`, `dataset.rs` | ✅ Solid | parsing, normalization, corpus cleaning. |
| WASM | `tabular_wasm` | ✅ Works | KV-cached SLM; CI-built for wasm32; no GGUF. |
| CLIs | `slm_cli`, `train_cli` | ✅ Works | `slm_cli` has **`run-gguf`** + `--weight_decay`/`--dropout`/`--stream`; binaries not unit-tested. |
| GUI | `ferrum_gui` | ✅ Backend checks | Tauri + vanilla JS; **GGUF panel** (`gguf_info`/`run_gguf`); windowed build needs WebView libs. |

---

## 7. Remaining gaps & recommendations

**GGUF/LLM — finish the runner:**

- 🟡 **Close the last int4↔int8 decode gap.** int4 is now ~4–5× f32 and within ~1.5×
  of int8's per-call time at half the RAM; a SIMD/LUT nibble unpack could make int4
  strictly fastest (the kernels carry no explicit SIMD at all — see §3a / benchmarks
  §3a, the broader headroom).
- 🟡 **Remaining quant formats** (`Q2_K`, `Q3_K`, `IQ*`) and **exact SPM encode**
  (unigram Viterbi) — widen the set of files that load and tokenize correctly.
- 🟡 **Finer import quantization** (per-block scales in `QWeight`) — reduce the
  double-quant quality loss without paying full f32 RAM.
- 🟢 **A training CLI/loop + AdamW for `llm_train`.** The backward pass + SGD step
  exist and are gradient-checked; batching, an AdamW path, and a `fine-tune` command
  would make small-model fine-tuning ergonomic.

**Engine quality:**

- 🟢 **CLI smoke tests + `wasm-bindgen-test`** — the two biggest coverage gaps;
  `run-gguf` has an end-to-end library test but no binary-level test, and the GUI
  GGUF commands have none.
- 🟢 **Explicit SIMD** in the hot kernels would unlock the idle vector units (§3a).

---

## 8. Metrics

_Generated 2026-06-23 from the working tree (excludes `target/`)._

### 8a. Codebase size

| Metric | Value |
|---|---:|
| Workspace crates | **6** (5 in-workspace + `ferrum_gui` excluded) |
| `ferrum_core` source files | **24** |
| `ferrum_core` lines (incl. unit tests) | **16,695** |
| GGUF/LLM modules | `gguf.rs` 1,526 · `llm.rs` 799 · `gguf_tokenizer.rs` 505 · `llm_train.rs` 821 |
| Markdown docs | 26 files |

### 8b. API surface & code health (`ferrum_core`)

| Metric | Value |
|---|---:|
| `pub fn` | **279** |
| `pub struct` / `pub enum` | 45 / 8 |
| External dependencies | **0** (`std` only) |
| `unsafe` blocks | **0** (`#![forbid(unsafe_code)]` at `lib.rs:51`) |
| Model formats | FINF v4/v5 (int8 & int4, per-tensor/per-channel); **reads** GGUF v2/v3 (F32/F16/Q8_0/Q8_1/Q4_0/Q4_1/**Q4_K/Q5_K/Q6_K**) |

### 8c. Tests

| Suite | `#[test]` functions | Result |
|---|---:|---|
| `ferrum_core` (unit) | **342** | ✅ pass |
| └ `gguf` / `llm` / `gguf_tokenizer` / `llm_train` | 26 / 17 / 6 / 6 | ✅ pass |
| `tests` crate (integration) | **95** (incl. `test_gguf_llm`) | ✅ pass |
| `tabular_wasm` | 4 | ✅ pass |
| doc-tests | 2 | ✅ pass |
| **Workspace total** | **443** | **0 failures, 0 warnings** |
| `ferrum_gui` (backend, excluded crate) | 14 | ✅ pass / `cargo check` clean |

---

## 9. Prioritized recommendations

1. **Explicit SIMD / LUT int4 unpack** — the single biggest speed lever left; the
   kernels are scalar, and int4 decode would become strictly fastest.
2. **Q2_K/Q3_K/IQ\* dequantizers** and **exact SPM encode** — widen the files that
   load and tokenize correctly.
3. **Finer import quantization** (per-block scales in `QWeight`) — cut the
   double-quant quality loss.
4. **CLI/WASM smoke tests** — close the remaining integration-seam coverage gap (the
   engine and GGUF primitives are well covered; the binaries are not).

The engine remains what it has always been at its core: correct, dependency-free,
reproducible, and well-tested for **small models you train yourself**. The GGUF/LLM
work is a genuinely usable importer — k-quants load, the tokenizer comes across,
`run-gguf` and a GUI panel drive it, the int4 kernel is fixed (~7 tok/s, ~4–5× f32),
and the architecture is now **trainable** (gradient-checked). On this hardware it is
a few-tok/s inference path for 1–3B models and a fine-tuner for *small* ones — but
training a 1B specifically is still out of reach (RAM + compute, §4.3).
