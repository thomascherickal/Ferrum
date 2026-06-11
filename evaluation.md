# Ferrum SLM Inference Engine — Engineering Evaluation

**Scope:** the `ferrum` workspace only (`ferrum_core`, `tabular_wasm`, `train_cli`, `tests`).
**Date:** 2026-06-11.
**Method:** full source review of all 17 Rust source files (~6,200 lines), build,
complete test-suite run, and an end-to-end CLI smoke test (iris → 98% accuracy →
FINF export → reload → predict).

**Verdict:** Ferrum is a well-architected, genuinely zero-dependency educational/edge
ML engine with unusually good test discipline (200+ tests including analytic gradient
checks). As shipped it **did not compile** (171 errors); after the fixes below the
entire workspace builds warning-free and all 200 tests pass.

---

## 1. Errors Found and Corrected

### 1.1 Compile failure: `use` statement inside a module doc comment — `loader.rs`

```rust
//! FINF binary format v4: weights + normalizer + ModelMetadata JSON.
//!
use crate::verbose;          // ← inserted mid-doc-comment
//! Layout (all integers little-endian):
```

Inner doc comments (`//!`) must precede all items; this produced 14 `E0753` errors.
The import was also unnecessary (the `vprintln!` macro resolves `is_verbose` through
`$crate::verbose::`). **Fix:** removed the stray import, restored the doc block.

### 1.2 Compile failure: `vprintln!` not in scope (157 errors)

`vprintln!` is declared with `#[macro_export]` in `verbose.rs`, which exports it at the
*crate root* — but every module invoked it bare, without importing it. Additionally,
`mod verbose` was declared *last* in `lib.rs`, so even textual `#[macro_use]` scoping
could not work. **Fix:** moved `verbose` to the top of `lib.rs` with `#[macro_use]`,
making the macro textually visible to all subsequent modules.

### 1.3 Stale tests: `tests/test_slm_library.rs` (2 failures)

Two tests asserted the *old* SLM input representation (token-index columns, header
`c0,c1,c2,label`, `input_dim == context_len`). The engine has since moved to one-hot
encoding (`c{pos}_v{vocab}` columns, `input_dim == context_len × vocab_size`); the
implementation is self-consistent and `generate()` depends on the new layout.
**Fix:** updated the assertions to the one-hot contract.

### 1.4 Missing validation: `TransformerBlock::new` head/dimension mismatch

`num_heads = 0` caused a divide-by-zero panic; `embedding_dim % num_heads != 0`
silently **dropped trailing channels** (the integer-truncated `head_dim` left part of
each row's attention output permanently zero) — a silent-wrong-answer bug, the worst
kind. `context_len = 0` would also panic later. **Fix:** constructor now returns
`InferError::DimMismatch` for all three cases (and the loader inherits the check).

### 1.5 Panic on NaN: `ops::argmax_rows` and `tabular_wasm::top_k_indices`

Both used `partial_cmp(..).unwrap()`, which panics on NaN logits — exactly the
situation (diverged training) where you most need a clean answer.
**Fix:** switched to `f32::total_cmp`.

### 1.6 Panic on multi-byte UTF-8: `GenerativeSLM::generate`

`&seed[..50]` byte-slices the seed for a log line; any seed longer than 50 bytes with
a multi-byte character straddling the boundary panics (the vocabulary explicitly
supports such characters — the test suite includes `'🌸'`). **Fix:** char-safe
`seed.chars().take(50)`.

### 1.7 Argument-parsing bug: `train_cli`

The CLI filtered `--verbose`/`-v` into a `positional` vector but then read
`args[1..5]` directly, so `train_cli --verbose data.csv out.bin` treated `--verbose`
as the CSV path. **Fix:** positional arguments are now read from `positional`.

### 1.8 Dead field warning: `TransformerSLMModel.vocab_size`

Stored but never read. **Fix:** exposed as a `vocab_size()` getter — useful to JS
callers anyway.

**Post-fix status:** `cargo build --workspace` → 0 errors, 0 warnings.
`cargo test --workspace` → **200 passed, 0 failed.**

---

## 2. What Is Good

- **True zero-dependency core.** `ferrum_core` uses only `std`. No serde, no ndarray,
  no rand. This makes the engine auditable end-to-end, trivially portable to
  `wasm32-unknown-unknown`, and immune to supply-chain churn.
- **Test discipline far above hobby-project norm.** Analytic-vs-finite-difference
  gradient checks for the entire backprop path *and* the fused loss; causal-mask
  enforcement tests; attention-rows-sum-to-one tests; FINF round-trip tests for every
  layer type; corrupt/truncated/bad-tag file tests; a full train→generate→serialize→
  reload integration test.
- **Correct numerics where it matters.** Max-subtracted softmax everywhere; fused
  softmax-cross-entropy with the clean `(p − onehot)/batch` gradient; `ln(max(p,1e-12))`
  loss clamping; Kaiming-style init scaled for ReLU; Box-Muller normal sampling with a
  `max(1e-7)` guard; LayerNorm with ε.
- **Cache-aware kernels.** `matmul` uses i-k-j loop ordering so the inner loop walks
  contiguous memory in both `b` and the output; the attention helpers use a
  transposed-B kernel for `QKᵀ` for the same reason.
- **Genuinely useful transformer implementation.** Pre-norm decoder block, causal
  masking, multi-head reshaping done explicitly and readably, residual connections,
  and per-head attention maps captured via `RefCell` for browser visualisation —
  a thoughtful teaching/debugging feature.
- **Self-describing model format.** FINF v4 embeds the normalizer and a metadata JSON
  (feature names/ranges, class names, dims, task type) so a UI can configure itself
  from the model file alone. Magic + version + bounds-checked reader = graceful
  failure on corrupt input.
- **Exemplary observability.** The `vprintln!`/`set_verbose` system logs shapes,
  activation statistics, dead-ReLU ratios, NaN/Inf detection at every stage, per-epoch
  loss and ETA — at the cost of one relaxed atomic load when disabled.
- **Determinism.** One seeded xorshift64* PRNG threaded explicitly through every API
  that needs randomness. Same seed → same model, bit-for-bit.

---

## 3. What Is Missing

### Architecture / capability gaps

- ~~**`GenerativeSLM` does not use the transformer.**~~ **Resolved** — see §4
  item 9: `GenerativeSLM::train_transformer` now trains a real causal
  transformer end-to-end with Adam.
- ~~**No KV cache.**~~ **Resolved** — see §4 item 8.
- **Character-level inputs only.** No BPE/byte-level tokenizer. (The transformer
  path now uses compact token-ID inputs, but the MLP path still uses one-hot
  contexts where `input_dim = context_len × vocab_size`.)
- **No quantisation.** Weights are f32 only. Int8 (or even f16) storage would cut
  model size 4× — important for a project that advertises <180 KB binaries.
- ~~**Optimisers: SGD+momentum only.**~~ **Resolved** — Adam added (§4 item 10).
  AdamW (decoupled weight decay) remains a possible refinement.
- **No softmax-free logits path.** The inference `Sequential` bakes Softmax in as the
  last layer, and `GenerativeSLM::generate` then applies temperature to the *probabilities*
  (treating them as logits). This works but double-normalises; a logits-out model with
  sampling applied once would be cleaner and slightly faster.

### Engineering gaps

- **No `rayon`-style or even chunked parallelism** (deliberate, but worth a feature
  flag for non-WASM targets).
- **No SIMD.** `std::simd` or manual 4-wide unrolling would give 2–4× on matmul.
- ~~**FINF parser can attempt huge allocations.**~~ **Resolved** — see §4
  item 7. (Dedicated fuzzing of the parser is still worthwhile.)
- **Vocabulary alignment hack.** `build_csv_dataset` injects one all-zero training row
  per vocab character to force class ordering. These are junk samples that bias the
  model toward uniform predictions on empty contexts; a `class_names` override in the
  dataset/metadata path would be cleaner.
- **`ModelMetadata::from_json`** is a hand-rolled substring scanner — fine for its own
  output, brittle for anything else (no escape handling beyond `\"`, no nesting).
- **No CI config** in the workspace for build+test on the WASM target.
- **No benchmarks** (`criterion` is excluded by the zero-dep rule, but a simple
  `std::time` bench binary would do).

---

## 4. Required / Recommended Changes

### Required (correctness & safety) — ✅ all applied in this review

1. Fix the two compile-blocking defects (doc comment, macro scoping). **Done.**
2. Validate `TransformerBlock` head divisibility — silent wrong answers. **Done.**
3. NaN-safe argmax/top-k. **Done.**
4. UTF-8-safe seed truncation. **Done.**
5. `train_cli` positional-arg parsing. **Done.**
6. Update stale SLM tests to the one-hot contract. **Done.**

### High value — ✅ implemented (2026-06-11, follow-up to the original review)

7. **Bounds-checked FINF parsing** — `Reader::f32_vec` now verifies the byte
   length against the remaining buffer *before* allocating, and all dimension
   products (`in×out`, `vocab×dim`, `c×c`, `c×h`) use checked multiplication,
   so corrupt or malicious files with huge dimension fields fail fast with a
   `Format` error instead of attempting multi-terabyte allocations
   (overflow-safe on 32-bit/wasm targets too). Covered by new tests.
8. **KV cache** — `layer::KvCache` plus `TransformerBlock::forward_with_cache`
   and `Embedding::embed_one` give O(T)-per-token incremental generation;
   `TransformerSLMModel` exposes it to JavaScript as `prime(context)` /
   `predict_next_cached(token_id)` / `cached_len()`. A test proves the cached
   path matches the full forward pass row-for-row.
9. **Trainable transformer** — new `train_transformer` module implements full
   backprop through token+positional embeddings, LayerNorm, causal multi-head
   attention (softmax backward through the mask), and the FFN. `TransformerNet`
   trains with next-token loss at every position and exports via
   `to_inference()` to a FINF-serialisable `Sequential`. The high-level API is
   `GenerativeSLM::train_transformer(corpus, …)`, and `generate()` now handles
   both model families. Verified by finite-difference gradient checks across
   all 18 parameter groups, a loss-halving training test, and an end-to-end
   train→generate→serialize→reload integration test.
10. **Adam optimiser** — `optim::Adam` with bias correction, used by the
    transformer trainer (SGD+momentum remains for the MLP path).

### Nice to have

11. Int8 post-training quantisation in FINF v5 (tag-compatible extension).
12. Replace the one-hot MLP input with token-ID + embedding even for the simple SLM
    path (shrinks models drastically).
13. Byte-pair or byte-level tokenizer module.
14. `#![forbid(unsafe_code)]` in `ferrum_core` (it already contains no unsafe; make it
    a guarantee).
15. CI workflow: `cargo test --workspace` + `cargo build --target wasm32-unknown-unknown`.
16. Remove the vocab-alignment padding rows in favour of explicit class registration.

---

## 5. Test & Verification Summary

| Check | Result |
|---|---|
| `cargo build --workspace` | ✅ 0 errors, 0 warnings (was: 171 errors) |
| `cargo test --workspace` | ✅ 200 passed / 0 failed (was: did not compile; 2 stale failures after compile fix) |
| Gradient checks (analytic vs finite difference) | ✅ pass |
| Causal mask & attention normalisation | ✅ pass |
| FINF round-trip, all 5 layer types | ✅ pass |
| End-to-end CLI: iris train → 98% acc → save → reload → predict | ✅ pass |
| SLM train → generate → serialize → reload (integration) | ✅ pass |
