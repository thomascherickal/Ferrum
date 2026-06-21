# Project Status

_Last updated: 2026-06-12_

Ferrum is a working, tested, zero-dependency Rust engine for small Transformers,
SLMs, and MLPs. The workspace builds clean and the full test suite passes.

---

## Implemented

### Engine (`ferrum_core`)

- **Tensors and ops** — matmul, softmax, row-wise reductions, LayerNorm.
- **Layers** — `Linear`, `ActivationLayer`, `LayerNorm`, `Embedding`,
  `Flatten`, `TransformerBlock` (causal multi-head self-attention + FFN), all
  behind the `Layer` trait, composed with `Sequential`.
- **KV cache** — `KvCache` for incremental generation.
- **Training** — hand-written backprop, `Net` (MLP) and `TransformerNet`,
  `train_epoch` / `train_transformer_epoch`, `Adam` and `Sgd` (with momentum),
  softmax cross-entropy and MSE losses.
- **Quantization** — symmetric per-tensor int8, fake-quantization for QAT, and
  int8 serialization (FINF v5).
- **CPU parallelism** — the matmul kernels behind Linear, FFN, attention, and
  the LM head split their rows across a **persistent worker pool** (threads
  spawned once and reused, so generation pays no per-call thread-creation cost).
  The thread count is detected dynamically (`available_parallelism`, override
  `FERRUM_NUM_THREADS`). Built only on `std` (threads + channels + `Arc`) with no
  `unsafe`, no external crates, no GPU; results are deterministic and `wasm32`
  runs serially.
- **Tokenizer** — `ByteBpeTokenizer`, a byte-level BPE tokenizer that
  round-trips any UTF-8 text, with a portable merge-list state.
- **Generative SLM** — `GenerativeSLM` with three training paths (`train`,
  `train_embedded`, `train_transformer`), unified generation, **streaming
  generation** (`generate_stream`, fragment-at-a-time + CLI `--stream`),
  held-out perplexity evaluation (`evaluate` / CLI `eval`), seed-stripped
  `generate_continuation`, and FINF I/O.
- **Multi-threaded training** — data-parallel minibatch training for the
  transformer (`train_transformer_epoch_threaded`, SLM
  `train_transformer_threaded_with_callback`, CLI `--threads`). Each minibatch is
  sharded across `std::thread::scope` workers and gradients are reduced in a
  fixed order, so it is `unsafe`-free, zero-dependency, reproducible for a given
  thread count, and bit-identical to the serial path at one shard. Complements
  the existing per-matmul row parallelism.
- **Model format** — FINF v4 (f32) / v5 (int8), self-contained (weights +
  normalizer + metadata JSON + tokenizer state).

### BPE integration (this release)

- The byte-level BPE tokenizer is wired through the **embedded** and
  **transformer** SLM training paths via a `vocab_size` selector
  (`0` = character-level, `>= 256` = BPE).
- The tokenizer's merge list is serialized in model metadata (`tokenizer_state`)
  and round-trips through save/load.
- Generation dispatches on the stored tokenizer: BPE models encode the seed,
  generate subword tokens, and decode back to text, while `num_chars` still
  counts characters and short prompts are left-padded.
- Training remains fully quantization-aware on BPE token streams.
- The CLI gained a `--vocab` flag and BPE-aware `info` output.

### Tools

- `slm_cli` → `train_transformer` binary: train / run / generate / info, with
  on-disk weight caching and int8 QAT.
- `train_cli` → tabular MLP trainer with CSV auto-detection.
- `tabular_wasm` → `wasm-bindgen` browser bindings.

### Tests

- Unit tests across `ferrum_core` plus integration tests in the `tests` crate,
  including dedicated coverage for BPE training, generation, save/load
  round-trips, quantization fidelity, and metadata serialization.

---

## Backward compatibility

- Models trained before the tokenizer field load unchanged and default to
  character-level tokenization.
- The character-level training paths are preserved exactly (`vocab_size = 0`).

---

## Possible future work

- Larger pre-tokenization vocabularies and merge caching for big corpora.
- Additional activation and normalization variants.
- Streaming generation APIs in the WASM bindings (the native streaming API,
  `generate_stream`, already exists).

These are directions, not commitments. The current release is complete and
self-consistent for the use cases documented in [usecases.md](usecases.md).
