# Project Status

_Last updated: 2026-07-02_

Ferrum is a working, tested, zero-dependency Rust engine for small Transformers,
SLMs, and MLPs — and an importer/exporter/runner for small open-weight
Llama/Qwen GGUF checkpoints. The workspace builds clean and the full test suite
passes (0 failures, 0 warnings).

---

## Implemented

### Engine (`ferrum_core`)

- **Tensors and ops** — matmul, fused/cache-tiled `linear_forward`, packed
  `qlinear` (int8/int4), softmax, row-wise reductions, LayerNorm.
- **Layers** — `Linear` (optionally carrying a packed `Arc<QWeight>`),
  `ActivationLayer`, `LayerNorm`, `Embedding`, `Flatten`, `TransformerBlock`
  (causal MHA + FFN), behind the `Layer` trait, composed with `Sequential`;
  `KvCache` for incremental generation.
- **Training** — hand-written backprop, `Net` (MLP) and `TransformerNet`,
  `train_epoch` / `train_transformer_epoch`, `Adam` (+ **AdamW** decoupled weight
  decay) and `Sgd`, grad-norm clipping, warmup + cosine/linear LR schedules,
  softmax cross-entropy and MSE; optional FFN dropout.
- **Quantization** — symmetric int8 (per-tensor **and** per-channel),
  fake-quantization for QAT, and **int8 + int4** serialization. In-memory packed
  `QWeight` (int8/int4, int4 in a **split-half** layout for a vectorizable decode).
- **CPU parallelism** — matmul kernels split across a **persistent worker pool**
  (threads spawned once and reused; no per-call thread-creation cost). Decode's
  `m = 1` GEMV is parallelized by a **column split** (`run_1d`). Thread count from
  `available_parallelism` (override `FERRUM_NUM_THREADS`). `std`-only, no
  `unsafe`, no GPU; deterministic across thread counts; `wasm32` runs serially.
- **Tokenizer** — `ByteBpeTokenizer`, byte-level BPE that round-trips any UTF-8
  text, with a portable merge-list state and deterministic merge learning.
- **Generative SLM** — `GenerativeSLM` with three training paths (`train`,
  `train_embedded`, `train_transformer`), unified generation, **streaming**
  (`generate_stream`, CLI `--stream`), held-out perplexity (`evaluate` / CLI
  `eval`), seed-stripped `generate_continuation`, and FINF I/O.
- **Multi-threaded training** — data-parallel minibatch training
  (`train_transformer_epoch_threaded`, CLI `--threads`): each minibatch is sharded
  across `std::thread::scope` workers and gradients reduced in a fixed order —
  `unsafe`-free, reproducible for a given shard count, and bit-identical to the
  serial path at one shard.
- **Model format** — FINF v4 (f32) / v5 (**int8 + int4**, per-tensor or
  per-channel via a per-weight-vector marker), self-contained (weights +
  normalizer + metadata JSON + tokenizer state).

### GGUF import & export, Llama/Qwen runner & training

- **GGUF reader** (`gguf`) — pure-`std`, `unsafe`-free GGUF v2/v3 parser:
  metadata, tensor directory, and dequantizers for `F32/F16/Q8_0/Q8_1/Q4_0/Q4_1`
  plus the **Q4_K/Q5_K/Q6_K** super-block k-quants. `Gguf::open` streams tensor
  data from disk (no whole-file resident); `from_path` keeps it in memory.
  `Q2_K`/`Q3_K`/`IQ*` are rejected with a clear error.
- **GGUF writer** (`gguf_write`) — the reverse path: `GgufBuilder` emits
  byte-exact GGUF v3, block *encoders* for all nine supported types are exact
  inverses of the reader's decoders (verified by round-tripping through the
  reader), and `write_llama_gguf` / `LlamaModel::write_gguf` serialize a loaded
  (or fine-tuned) llama/qwen2 model — tokenizer and hyperparameters carried
  forward verbatim — to a file that runs in llama.cpp/ollama. Writes are
  atomic (temp + rename); norms/biases stay f32; non-block-aligned matrices
  fall back to f16 with the `general.file_type` hint reflecting what was
  actually emitted.
- **GGUF tokenizer import** (`gguf_tokenizer`) — reconstructs a checkpoint's own
  tokenizer from `tokenizer.ggml.*`: BPE encode/decode (exact) and SPM decode
  (exact) / encode (greedy), so imported models run on **text**.
- **Llama/Qwen runner** (`llm`) — `LlamaModel` with RMSNorm, RoPE (Norm/Neox),
  grouped-query attention + KV cache, and the SwiGLU FFN; full-sequence forward
  and O(context)/token cached decode, cross-checked against each other.
  `Gguf::load_llama[_prec]` packs weights to int4/int8 (or keeps f32, no
  double-quant). Import is lossy and not bit-exact to llama.cpp.
- **Training the imported architecture** (`llm_train`) — `LlamaTrainer` adds a
  hand-derived, **finite-difference-checked** backward pass (RMSNorm, RoPE,
  GQA+softmax, SwiGLU, embedding, LM head), next-token cross-entropy, and an SGD
  `train_step`. Requires f32 weights (`new` rejects quantized models).

### Tools

- `slm_cli` → `train_transformer` binary: `train` / `run` / `generate` /
  **`run-gguf`** / **`finetune-gguf`** / **`export-gguf`** / `eval` / `info`,
  with on-disk weight caching, int8 QAT, and
  `--weight_decay` / `--dropout` / `--stream` / `--threads`.
- `train_cli` → tabular MLP trainer with CSV auto-detection.
- `tabular_wasm` → `wasm-bindgen` browser bindings.
- `ferrum_gui` → Tauri 2 desktop/mobile app; a tab per task plus **GGUF** (import/run),
  **Fine-tune**, and **Export** panels
  (`gguf_info` / `run_gguf` / `finetune_gguf` / `export_gguf`). The Rust backend type-checks; the windowed build
  needs the system WebView libraries.

### Tests

- Unit tests across `ferrum_core` plus integration tests in the `tests` crate,
  including dedicated coverage for BPE training/generation, save/load round-trips,
  int8/int4 quantization fidelity, metadata serialization, the GGUF k-quant
  decoders *and encoders* (round-tripped through each other), the streamed
  reader, the Llama cached-decode-vs-full-forward equivalence, the
  gradient-checked `llm_train` backward pass, an end-to-end synthetic-GGUF
  import + generate, and export round-trips (bit-exact f32 logits, qwen2-bias
  preservation, k-quant emission at dim 256, CLI binary happy path).

---

## Backward compatibility

- Models trained before the tokenizer field load unchanged and default to
  character-level tokenization; v4 (f32) loads alongside v5 (int8/int4).
- The character-level training paths are preserved exactly (`vocab_size = 0`), and
  the serial training path equals the threaded path at one shard.

---

## Possible future work

- Remaining quant formats (`Q2_K`/`Q3_K`/`IQ*`) and exact SPM encode (Viterbi).
- A SIMD/LUT int4 nibble unpack to make int4 decode strictly fastest (it is
  currently within ~1.5× of int8 at half the RAM — see [benchmarks.md](benchmarks.md)).
- An AdamW path and a `fine-tune` loop for `llm_train`; finer per-block import
  quantization to cut the double-quant quality loss.
- CLI/WASM smoke tests; WASM streaming bindings (the native `generate_stream`
  exists).

These are directions, not commitments. The current release is complete and
self-consistent for the use cases documented in [usecases.md](usecases.md).
