# Ferrum — Project Status

_Last updated: 2026-06-12_

Ferrum is a zero-dependency, pure-Rust engine for building, training, and
running causal Transformers, Small Language Models (SLMs), and classical MLPs
on CPU-only / edge / WebAssembly targets. This document records what works
end-to-end, what is partial, and what is not present.

## TL;DR

- **Builds:** `cargo build --workspace` — clean.
- **Tests:** `cargo test --workspace` — **241 tests, all passing** (0 failed).
- **Library (`ferrum_core`):** zero external dependencies, packages cleanly
  (`cargo package`), compiles in isolation. Fully self-contained.
- **End-to-end SLM pipeline (train → save → reload → generate):** works via the
  new `train_transformer` CLI and the `GenerativeSLM` library API.
- **Tabular pipeline (CSV → train → FINF → predict):** works via `train_cli`.
- **WASM:** `tabular_wasm` compiles to `wasm32-unknown-unknown` (245 KB release).
  JS-glue generation needs the external `wasm-bindgen` CLI (not installed here);
  prebuilt glue ships in `web/pkg/`.

## Workspace layout

| Crate / dir     | Purpose                                                        | Status |
|-----------------|----------------------------------------------------------------|--------|
| `ferrum_core`   | The engine: tensors, layers, transformer, SLM, FINF I/O, optim | ✅ Works |
| `train_cli`     | CSV tabular trainer (classification / regression) → FINF        | ✅ Works |
| `slm_cli`       | **NEW** `train_transformer` binary: text → SLM, generate, info  | ✅ Works |
| `tabular_wasm`  | WASM bindings for tabular + transformer SLM inference           | ✅ Compiles |
| `tests`         | Integration / unit test crate                                  | ✅ 241 pass |
| `web/`          | Browser playground (HTML/JS + prebuilt WASM + model.bin files)  | ⚠️ Needs `wasm-bindgen` CLI to rebuild glue |
| `scripts/`      | Dataset download, train-all, WASM build helpers                | ⚠️ Need network / external tools |

## What works

### Core library (`ferrum_core`)
- **Tensors & ops:** matrix creation, matmul, softmax, layernorm, argmax, etc.
- **Layers:** `Linear`, `ActivationLayer`, `LayerNorm`, `Embedding`, `Flatten`,
  `TransformerBlock` (causal multi-head self-attention + FFN), `KvCache`.
- **Models:** `Sequential` inference pipeline.
- **Training:**
  - `Net` trainable MLP with SGD (+ momentum) — `train_epoch`, `accuracy`.
  - `TransformerNet` trainable decoder-only transformer with Adam —
    `train_transformer_epoch`.
  - Embedding-MLP language model (`Net::embedding_mlp`).
- **Losses:** `softmax_cross_entropy`, `mse`.
- **Optimisers:** `Sgd` (optional momentum), `Adam`.
- **Serialisation:** FINF binary format — `to_bytes` / `from_bytes` /
  `save` / `load`, plus int8-quantised variants (`to_bytes_quantized`,
  `save_quantized`) ≈4× smaller. Corrupt-input handling is tested.
- **SLM:** `GenerativeSLM` with three training paths — `train` (one-hot MLP),
  `train_embedded` (token-ID embedding MLP), `train_transformer` (true causal
  transformer) — each with a `*_with_callback` progress variant; plus
  `generate` (temperature sampling), `to_bytes`/`from_bytes` roundtrip.
- **Tokenizer:** byte-level BPE (`ByteBpeTokenizer`) and char/hex helpers.
- **RNG:** deterministic seeded xorshift64* (`Rng`).
- **Verbose mode:** `set_verbose` toggles detailed engine tracing.
- **Zero dependencies:** `cargo tree -p ferrum_core` shows only itself.

### `train_transformer` CLI (new — `slm_cli`)
Closes the gap where `train_cli` told users to "use train_transformer" but no
such binary existed. Verified end-to-end on a sample corpus:

```
train_transformer train    <corpus.txt> <model.bin> [--arch transformer|embedded|mlp] [opts]
train_transformer generate <model.bin>  <seed text> [--chars N] [--temp F] [--seed N]
train_transformer info     <model.bin>
```

- All three architectures train, export FINF (plain or `--quantize`), reload,
  and generate. Example run: transformer, 80 epochs, loss 2.52 → 0.33, produced
  coherent text matching the corpus.
- Friendly errors for empty corpus, short seed (< context window), bad `--arch`.

### `train_cli` (tabular)
- Auto-detects classification vs regression from a CSV, splits, normalises,
  trains an MLP, evaluates, exports FINF, and spot-checks a reload.
- Correctly rejects `TransformerSLM` CSVs and points to `train_transformer`.

### WASM (`tabular_wasm`)
- Compiles to `wasm32-unknown-unknown` (release, 245 KB).
- Exposes `TabularModel` (tabular predict) and `TransformerSLMModel`
  (priming, KV-cached `predict_next_cached`, attention weights, top-k,
  entropy, temperature sampling).
- Prebuilt JS glue + demo models already present under `web/`.

## What does not work / requires external setup

- **`web/` glue rebuild:** `scripts/build_wasm.sh` needs the `wasm-bindgen` CLI
  (`cargo install wasm-bindgen-cli --version 0.2.122`). The Rust→wasm32 compile
  itself works; only JS-binding generation is gated on that tool. Prebuilt
  `web/pkg/` artifacts let the existing demos run without rebuilding.
- **Dataset download / train-all scripts:** `scripts/download_datasets.sh`
  fetches the 10 tabular datasets from the public internet and needs `curl` +
  `python3` + network access; `scripts/train_all.sh` then needs those CSVs.
  No tabular CSVs are committed to the repo (by design).
- **`scripts/build_wasm.sh` auto-distribution:** the tail of the script copies
  artifacts into sibling repos (`../brand_alchemist`, `../ambient_poet`,
  `../shell_oracle`); it skips silently if those directories are absent.
- **`no_std`:** despite earlier metadata, the crate is **not** `no_std` — it uses
  `std` (`std::fs`, `std::time`, `std::collections`) throughout. Metadata and
  docs were corrected to state "zero-dependency, std-only" rather than `no_std`.
  A genuine `no_std` port (feature-gating file/time/IO) remains future work.

## Known limitations (by design, not bugs)

- Pure CPU, single-threaded; training large models is slow (the 80-epoch
  transformer demo above took ~100 s on a tiny corpus).
- The one-hot `mlp` SLM architecture scales poorly (input = context × vocab) and
  produces weaker samples than `transformer` / `embedded`; prefer those.
- Character-level vocabulary only for SLMs (BPE tokenizer exists but the SLM
  training paths use char/hex tokens).

## How to reproduce

```bash
# Build + test everything
cargo build --workspace
cargo test  --workspace          # 241 tests pass

# End-to-end SLM
printf 'the quick brown fox jumps over the lazy dog.\n' > corpus.txt   # use a longer corpus in practice
cargo run -p slm_cli -- train corpus.txt model.bin --arch transformer --context 12 --epochs 80 --sample
cargo run -p slm_cli -- info model.bin
cargo run -p slm_cli -- generate model.bin "the quick brown" --chars 80 --temp 0.6

# Package the library standalone
cargo package -p ferrum_core      # zero deps, verifies isolated compile

# WASM (needs wasm-bindgen CLI for the glue step)
cargo build -p tabular_wasm --target wasm32-unknown-unknown --release
```
