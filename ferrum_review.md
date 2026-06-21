# Ferrum Review — gaps, limits, and GUI audit

_Date: 2026-06-21 · Reviewer: automated deep review · Target machine: Intel
i5-1135G7 (4 cores / 8 threads), 16 GB RAM._

This document reviews the entire `ferrum` module for gaps that limit **training**,
**inference**, and **testing** of SLMs locally; estimates the **maximum model
size** practical on this machine; and audits the new **Tauri GUI** (`ferrum_gui`)
for errors and mistakes.

Scope reviewed: `ferrum_core` (`tensor, ops, layer, model, loss, optim, rng,
activation, csv, quant, tokenizer, train, train_transformer, slm, loader,
parallel, verbose, dataset`), `slm_cli`, `train_cli`, `tabular_wasm`, the `tests`
crate, and `ferrum_gui` (`src/*.rs`, `ui/*`, config).

---

## 1. Bottom line

**There are no _blocking_ gaps for small SLMs** — training, inference, evaluation,
serialization, and the test suite all work today (201 `ferrum_core` lib tests +
the integration suite pass this session; the CLI and GUI both run). The gaps
below do not stop the happy path; they **limit quality and prevent scaling** to
larger or more serious SLMs, and several will cause real failures
(divergence → `NaN`, out-of-memory, or unusably slow runs) as you scale up.

The single most important practical fact: **this is a scalar, CPU-only engine, so
compute — not features and not RAM — is the binding constraint.** Memory would
allow tens of millions of parameters; wall-clock time makes anything beyond a few
million impractical to train locally (see §9).

Severity legend: 🔴 will break/block at modest scale · 🟡 limits quality/scale ·
🟢 polish/robustness.

---

## 2. What works today (verified)

- All three SLM training paths (`GenerativeSLM::train`, `train_embedded`,
  `train_transformer` + the multi-threaded variant) run and reduce loss.
- Quantization-aware training + int8 FINF v5 serialization round-trips with small
  drift (tested).
- Generation (char-level + byte-level BPE), streaming generation, held-out
  perplexity evaluation, and model inspection work from both the library and GUI.
- FINF loader is defensively coded: magic/version checks, bounds-checked reads,
  `checked_mul` against dimension/length overflow (`loader.rs`).
- Determinism: fixed seeds reproduce results; multi-threaded training is
  reproducible per thread count and bit-identical to serial at one shard.

---

## 3. Gaps that limit TRAINING

| # | Gap | Sev | Where | Impact |
|---|-----|-----|-------|--------|
| T1 | **No gradient clipping / norm clipping** | 🔴 | `optim.rs`, `train_transformer.rs::train_transformer_epoch*`, `train.rs::train_epoch` | The code *detects* `NaN`/`Inf`/explosion and warns (`slm.rs`, `train.rs`) but cannot prevent it. Larger models / higher LR diverge with no recourse. This is the most likely thing to *prevent successful training* as you scale. |
| T2 | **No LR schedule / warmup / decay** | 🔴 | `optim.rs` (fixed `lr`), `slm.rs` | Transformers normally need warmup + decay for stability. Fixed Adam LR makes bigger models fragile and slower to converge. |
| T3 | **Data-parallel training clones the whole net per shard** | 🔴 | `train_transformer.rs::train_transformer_epoch_threaded` (`worker = net_ref.clone()`) | Each shard deep-copies `data+grad+m+v` (16 B/param). At the default `threads = 8` that is **≈8× the model in RAM every step** (≈128 B/param of clones on top of the 20 B/param master). This is the memory ceiling for threaded training (see §9) and adds a per-step clone cost. **Fix:** share read-only weights via `Arc` and give each worker only a gradient accumulator (≈4 B/param), not a full clone. |
| T4 | **Minibatch sampling is with replacement; no shuffle, no true epochs** | 🟡 | `train.rs::train_epoch` ("Random minibatch indices (with replacement)"), `train_transformer.rs` (`rng.next_u64() % num_windows`) | An "epoch" covers ≈63 % of windows in expectation; some never seen in short runs. Hurts sample efficiency and reproducible coverage. **Fix:** shuffle index permutation per epoch, draw without replacement. |
| T5 | **No validation split / early stopping / best-checkpoint during training** | 🟡 | `slm.rs` training fns | `evaluate()` exists but is not wired into the loop; you cannot stop on val-perplexity or keep the best epoch. Overfitting on small corpora (the common local case) is unmanaged. |
| T6 | **No checkpoint/resume mid-training** | 🟡 | `slm.rs::load_or_train` only loads *finished* models | A long local run that is interrupted is lost; cannot resume optimizer state. |
| T7 | **No weight decay (AdamW) / regularization / dropout** | 🟡 | `optim.rs`, `layer.rs` | Limits generalization; with small corpora the model memorizes (we measured train-perplexity ≈ 1.0 vs held-out ≈ 3.7). |
| T8 | **Embedded & one-hot paths are single-threaded** | 🟢 | only `train_transformer` has the threaded epoch | `train`/`train_embedded` don't use the data-parallel path; slower than necessary. |
| T9 | **No weight tying (embedding ↔ LM head)** | 🟢 | `train_transformer.rs` (`tok_emb` and `head_w` are independent) | Wastes ≈`vocab×embed` params and a little quality; tying is a standard, free win. |
| T10 | **Whole corpus tokenized into RAM; no streaming dataset** | 🟢 | `slm.rs::tokenize_for_lm` | Caps trainable corpus size at memory; fine for local scale, but no chunked/streamed ingestion. |

---

## 4. Gaps that limit INFERENCE

| # | Gap | Sev | Where | Impact |
|---|-----|-----|-------|--------|
| I1 | **KV cache exists and is tested but is NOT wired into native generation** | 🔴 (perf) | `layer.rs::TransformerBlock::forward_with_cache` + `KvCache` are used by the **WASM** `TransformerSLMModel` (`tabular_wasm/src/lib.rs`) but **not** by `GenerativeSLM::generate`/`generate_stream` (`slm.rs`), which re-runs a full forward over the whole context window **every token**. | Native (CLI **and** GUI) generation is O(context²) per token instead of O(context). It works but is far slower than the WASM build and slower than necessary — the GUI streams on the slow path. **Fix:** add a cached generate to `GenerativeSLM` reusing the per-block `forward_with_cache` (the WASM model already shows the pattern). |
| I2 | **Sampling is temperature-only** | 🟡 | `slm.rs::sample_from_logits` | No top-k, top-p/nucleus, or repetition penalty. Low-temperature output falls into repetition loops; there is no knob to fix it. A real generation-quality gap. |
| I3 | **No EOS / stop criterion** | 🟡 | `tokenizer.rs` (no special tokens), `slm.rs::generate` | Generation always runs exactly `num_chars`; cannot stop at a natural boundary, and multi-document corpora have no separator token. |
| I4 | **Probabilities are softmaxed in-model, then `ln()`-ed back for temperature** | 🟢 | `slm.rs::generate*` (`p.max(1e-12).ln()`) | The `1e-12` floor distorts the tail slightly and is lossy vs. sampling from raw logits. Cosmetic but avoidable if the inference head exposed logits. |
| I5 | **No batched generation** | 🟢 | `slm.rs` | One sequence at a time; fine for a single user, but no throughput path. |

---

## 5. Gaps in TOKENIZATION

| # | Gap | Sev | Where | Impact |
|---|-----|-----|-------|--------|
| K1 | **`encode` re-applies every merge over the whole text: O(merges × len)** | 🟡 | `tokenizer.rs::encode` (loops `merges`, each `merge_pair` rescans) | For vocab 512 that is 256 full passes per encode. Dominates BPE corpus tokenization and the `generate_bpe` loop (which also `decode`s the full id list every step → O(steps²)). Slow for large corpora/long generations. **Fix:** rank/heap-based BPE encode; cache decode length incrementally. |
| K2 | **No pre-tokenization (word/space boundaries)** | 🟡 | `tokenizer.rs::train` merges across spaces/newlines | Unlike GPT-2 BPE, merges can span whitespace, yielding lower-quality, less reusable tokens on natural language. |
| K3 | **No special tokens (BOS/EOS/PAD/UNK)** | 🟢 | `tokenizer.rs` | Cannot mark document/sequence boundaries; see I3. |

---

## 6. Gaps in TESTING / CI

| # | Gap | Sev | Where | Impact |
|---|-----|-----|-------|--------|
| X1 | **`ferrum_gui` has zero automated tests and is excluded from the workspace** | 🟡 | `ferrum_gui/` (own crate) | The whole GUI backend (`commands.rs`) is untested and not built by `cargo test --workspace`. Regressions there are invisible to CI. **Fix:** unit-test the pure command logic (validation, dataset cleaning paths) and add a CI job that `cargo check`s `ferrum_gui`. |
| X2 | **No CI check that `tabular_wasm` builds for `wasm32`** | 🟡 | `.github/` | The WASM target (which *does* use the KV cache) can silently break. Add `cargo build -p tabular_wasm --target wasm32-unknown-unknown`. |
| X3 | **No adversarial/fuzz tests for the loader & tokenizer `from_state`** | 🟢 | `loader.rs`, `tokenizer.rs` | Loader has some negative tests; add truncated/corrupt-buffer fuzzing and malformed merge-state cases. |
| X4 | **Integration tests are slow (~50 s) because they train in debug** | 🟢 | `tests/` | Friction; consider tiny configs or a `--release` test profile. |

---

## 7. Quantization & numerical notes

- **Per-tensor symmetric int8** (`quant.rs`, `scale = max|w|/127`). A single outlier
  weight inflates the scale and coarsens the whole tensor. 🟡 At larger widths,
  **per-channel** quantization would materially improve fidelity. QAT currently
  hides most of this.
- Biases / LayerNorm (< `QUANT_MIN_LEN = 64`) stay f32 — correct and sensible.
- LayerNorm `eps = 1e-5` is consistent between train and inference (good).

---

## 8. Summary: does anything *prevent* success?

- **Small SLMs (≤ ~1 M params):** nothing prevents training/inference/testing — verified.
- **Scaling up:** T1 (no grad clipping) and T2 (no LR schedule) will *prevent
  successful training* of larger models (divergence); T3 (per-shard clone) will
  *prevent* threaded training of larger models on 16 GB (OOM); I1 makes larger-model
  inference slow but not impossible.

---

## 9. Maximum SLM size on this machine

**Machine:** i5-1135G7, 4 physical / 8 logical cores, 16 GB RAM (only ~2 GB free
right now under your current desktop load), scalar f32 kernels (no SIMD intrinsics,
no GPU), default `num_threads = 8`.

### 9a. Parameter count of a decoder-only SLM

For embedding `C`, vocabulary `V`, context `T`, FFN hidden `H`, `L` blocks:

```
P ≈ 2·V·C        (token embedding + LM head)
  + L·(4·C² + 2·C·H)   (attention QKVO + FFN per block)
  + T·C          (positional embedding)
```

### 9b. Memory ceiling (bytes per parameter)

| Mode | Bytes/param | Why |
|------|------------:|-----|
| Inference (load int8 → f32 in RAM) | ~5 | weights f32 + small activations |
| **Serial** training (`threads = 1`) | ~20 | param+grad+m+v (16) + transient QAT snapshot (4) + activations |
| **Threaded** training (`threads = 8`) | ~150 | master (20) + **8 worker clones × 16** (T3) + activations |

Resulting ceilings (P = budget ÷ bytes/param):

| Usable RAM | Inference | Serial train | Threaded train (×8) |
|-----------:|----------:|-------------:|--------------------:|
| **~2 GB** (free now) | ~400 M | ~100 M | ~13 M |
| **~8 GB** (close other apps) | ~1.6 B | ~400 M | ~54 M |

> Takeaway: memory is **not** the limit for any model you would realistically
> train on a CPU. If you do train multi-million-parameter models, prefer
> `--threads 1` (or fix T3) — it raises the memory ceiling ~7× and avoids the
> per-step clone.

### 9c. Compute ceiling — the real limit

Empirical anchor (this session): a ~37 K-param transformer, ≈144 K token-forwards
(`windows × T × epochs`), trained in **75 s in a debug build** → roughly **3–5 s
in release**. Training cost scales linearly:

```
release_seconds ≈ 8e-10 × P × (windows × T × epochs)
```

Because real training pushes `windows × T × epochs` token-forwards through the
net, this grows fast. Practical wall-clock budgets:

| Budget | `P × token-forwards` ≤ | Example |
|-------|-----------------------:|---------|
| ~15 min | ~1 × 10¹² | 1 M params × (small corpus, few epochs) |
| Overnight (~8 h) | ~3 × 10¹³ | ~1–2 M params on a ≤1 MB corpus, tens of epochs |
| Multi-day | ~10¹⁴ | ~5–8 M params, small corpus only |

So the **practical training ceiling is ~1–5 M parameters**, not the tens of
millions RAM allows. Inference is cheaper but, without the KV cache (I1), costs
~`2·P·context` FLOPs/token → a 10 M-param model is ~0.1–1 s/token (usable);
~100 M is seconds/token (not interactive).

### 9d. Recommended configurations

| Tier | C | heads | L | H | V | T | **≈ Params** | Train feel | Mem (serial / ×8) |
|------|--:|--:|--:|--:|--:|--:|------------:|-----------|------------------:|
| Nano (GUI default ≈) | 32 | 4 | 2 | 64 | 512 | 16 | **~50 K** | seconds | <2 MB / ~8 MB |
| Micro | 64 | 4 | 4 | 256 | 1024 | 64 | **~330 K** | minutes | ~7 MB / ~50 MB |
| **Small (recommended max)** | 128 | 8 | 6 | 512 | 2048 | 128 | **~1.7 M** | hours (small corpus) | ~34 MB / ~255 MB |
| Medium (upper practical) | 256 | 8 | 8 | 1024 | 4096 | 128 | **~8.4 M** | overnight+, tiny corpus | ~170 MB / ~1.2 GB |
| (Memory-bound only) | — | — | — | — | — | — | ~50–300 M | weeks → impractical | 1–6 GB |

**Practical answer:** aim for **≤ ~2 M parameters** (the "Small" row) for
comfortable local training and responsive generation; **~8 M is the upper bound**
for patient, small-corpus, serial runs. Anything larger is RAM-feasible but
compute-impractical on this CPU.

---

## 10. GUI review (`ferrum_gui`) — errors & mistakes

The GUI **compiles cleanly (debug + release, 0 warnings) and runs** (verified:
window renders, backend connected, live CPU/MEM/THREADS monitor, terminal prompt
shows the live cwd). Issues found, by severity:

| # | Issue | Sev | Where | Fix |
|---|-------|-----|-------|-----|
| G1 | **Streamed-generation tail can be dropped** | 🟡 | `ui/app.js` (`genStart` sets `generating=false` in `finally`; `gen-fragment` handler ignores events once false). The `invoke` promise can resolve before the last event arrives over IPC, and stream mode never falls back to the returned text. | On completion in stream mode, set `genOut` from the authoritative returned string (or keep appending and reconcile). The data is correct in the return value; only the display can lose the last fragment. |
| G2 | **`download_text` has no HTTP timeout** | 🟡 | `commands.rs::download_text` (`ureq::get(&url).call()`) | A hung/slow host blocks the task indefinitely. Use an agent with connect/read timeouts (e.g. 30 s). |
| G3 | **Tabular default binary path is wrong relative to the app cwd** | 🟢 | `ui/index.html` `#tbBin = "./target/release/train_cli"`; the app's cwd is its launch dir (e.g. `ferrum_gui/`), where that path does not exist | Default to an absolute path, or resolve against the workspace root, or document that the user must set it. |
| G4 | **Browse buttons depend on the `window.__TAURI__.dialog` global** | 🟢 | `ui/app.js::pickOpen/pickSave` | If the dialog JS global isn't present, Browse silently no-ops (manual path entry still works — graceful). Verify it is exposed under `withGlobalTauri`; if not, add a small Rust dialog command and `invoke` it. |
| G5 | **Process-wide verbose + log sink** | 🟢 | `commands.rs` (`set_verbose` toggled per call) + `lib.rs` (single sink) | Concurrent train+generate interleave logs and flip each other's verbose flag. Fine for one user; document or scope per-operation. |
| G6 | **Embedded shell does not persist non-`cd` state** | 🟢 | `commands.rs::run_shell` (each command is a fresh `sh -c`) | `export FOO=bar` won't be visible to the next command (only `cd` is special-cased). It's a "basic shell," but worth stating in the UI. |
| G7 | **Non-UTF-8 shell output is truncated at the first invalid line** | 🟢 | `commands.rs::run_shell` (`lines().map_while(Result::ok)`) | Binary tool output stops early. Stream raw bytes with `from_utf8_lossy` if needed. |
| G8 | **CSP disabled (`"csp": null`)** | 🟢 | `tauri.conf.json` | Fine for a local app; tighten before distributing. |
| G9 | First `system_stats` sample may read 0 % CPU | 🟢 | `commands.rs::system_stats` | Needs two refreshes spaced by `MINIMUM_CPU_UPDATE_INTERVAL`; self-corrects on the next poll. Cosmetic. |

No correctness bug was found that prevents the GUI from functioning. G1 and G2 are
the two worth fixing soon.

---

## 11. Prioritized recommendations

**Do first (unblock real training/inference):**
1. **Gradient clipping** (global-norm) in the training loops — T1. Cheap, prevents divergence.
2. **Wire the KV cache into `GenerativeSLM::generate`** — I1. The pieces exist; this is the biggest inference speedup.
3. **Fix threaded-training memory** (share weights via `Arc`, per-worker grad-only) — T3. Removes the ~8× memory blow-up.
4. **LR warmup + cosine/linear decay** — T2.

**High value next:**
5. top-k / top-p / repetition-penalty sampling — I2.
6. Validation split + early stopping + best-checkpoint, reusing `evaluate()` — T5.
7. Shuffle-without-replacement minibatching — T4.
8. GUI: fix G1 (stream tail) and G2 (download timeout).

**Quality / hardening:**
9. Per-channel int8 quant (§7); weight tying (T9); BPE pre-tokenization + faster `encode` (K1/K2); checkpoint-resume (T6); GUI tests + wasm-build CI (X1/X2).

None of these are required for the current, working small-SLM workflow; they are
the path to training and serving meaningfully larger models locally.
