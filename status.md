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
- **Tokenizer** — `ByteBpeTokenizer`, a byte-level BPE tokenizer that
  round-trips any UTF-8 text, with a portable merge-list state.
- **Generative SLM** — `GenerativeSLM` with three training paths (`train`,
  `train_embedded`, `train_transformer`), unified generation, and FINF I/O.
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
- Multi-threaded training for larger models.
- Additional activation and normalization variants.
- Streaming generation APIs in the WASM bindings.

These are directions, not commitments. The current release is complete and
self-consistent for the use cases documented in [usecases.md](usecases.md).
