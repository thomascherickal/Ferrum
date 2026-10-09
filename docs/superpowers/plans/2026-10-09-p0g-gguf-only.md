# Phase 0-G — GGUF Everywhere, FINF Removed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **Run this plan before `2026-10-09-p0-shared-surfaces.md`.**

**Goal:** Every file Ferrum writes is GGUF v3. Ferrum's own BPE transformer SLMs are `gpt2`-architecture GGUFs that run unmodified in llama.cpp, Ollama, and LM Studio, down to ~0.1M parameters. That covers trained models (f32, int8, int4), training checkpoints (`.fckp` and `.flck` contents), and the reusable BPE tokenizer file. The FINF format and its code are deleted. Every crate forbids `unsafe` code.

**Architecture:**
- `ferrum_core/src/loader.rs` keeps its public function names (`to_bytes`, `to_bytes_quantized`, `to_bytes_quantized_int4`, `from_bytes`, `save`, `save_quantized`, `save_quantized_int4`, `load`), so callers do not change. The bodies now write and read a GGUF container, built with the existing `GgufBuilder` and parsed with the existing `Gguf::parse`.
- Quantized matrices are stored **exactly** as Ferrum holds them in memory: per-row int8, or packed int4 bytes as a GGML `I8` tensor plus an F32 per-row `scale` tensor. A reload is bit-identical to the trained (QAT) model.
- Ferrum's architecture is not llama, so `general.architecture = "ferrum"`. llama.cpp will not run these files; Ferrum's own `export-gguf` path for imported Llama/Qwen models is unchanged.

**Tech Stack:** Rust 2021, the in-crate GGUF reader/writer (`gguf.rs`, `gguf_write.rs`).

**Spec:** `docs/superpowers/specs/2026-10-09-dataops-serve-eval-design.md` §4.0 (added for this change).

## Global Constraints

- Task order: GA, GB, GT, G1, G2, G3, G4; then Phase 0 Tasks 1–5; then G5 (needs Phase 0 Task 4's `generate_tokens`).
- Supported size range: from **~0.1M params** (reference tiny config `--context 64 --embed 64 --heads 4 --blocks 2 --hidden 128 --vocab 512 --tie` = 103,936 params) to the 507,552-param reference and beyond. Tensors whose GGML row length is not a multiple of 32 are stored **F32** (exact), never F16.

- Branch `feat/dataops-serve-eval`. No worktree: this runs before the parallel phases.
- No `unsafe` code anywhere: `#![forbid(unsafe_code)]` at every crate root (Task G4).
- Every feature lands with its unit tests in the same commit (test first, then code).
- Each commit passes `cargo test --workspace --release`, `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo build -p slm_wasm --target wasm32-unknown-unknown --release`, and `(cd ferrum_gui && cargo check --all-targets && cargo test)`.
- `../Ferrum-ML` depends on `ferrum_core` by path. It must build and pass its tests after Task G2. It is a separate local git repo with no remote: commit there, never push.
- Old FINF files stop loading. This is a deliberate breaking change, recorded for Status §7 in Phase 2. Before the FINF reader is deleted, convert every FINF file that is committed in either repo (Task G2).

## Ferrum GGUF layout (`ferrum.format = 1`)

| Key / tensor | Type | Meaning |
|---|---|---|
| `general.architecture` | string | `"ferrum"` |
| `general.name` | string | `ModelMetadata.dataset_name` |
| `ferrum.format` | u32 | `1` |
| `ferrum.quant` | string | `"f32"`, `"int8"`, or `"int4"` (the precision used for matrices) |
| `ferrum.metadata` | string | `ModelMetadata::to_json()` (task, dims, class names, tokenizer state, …) |
| `ferrum.normalizer` | string | `Normalizer::encode()` (empty for SLMs) |
| `ferrum.layer_count` | u32 | number of layers |
| `ferrum.layer.{i}.type` | string | `linear`, `activation`, `flatten`, `embedding`, `layernorm`, `transformer_block` |
| `ferrum.layer.{i}.{dim}` | u32 | `linear`: `in`, `out` · `activation`: `kind` (`Activation::tag()`) · `embedding`: `vocab`, `seq`, `dim` · `layernorm`: `dim` · `transformer_block`: `context`, `heads`, `dim`, `hidden` |
| `l{i}.{param}` | F32 | f32 tensor; GGML dims `[cols, rows]` for a row-major `[rows, cols]` matrix, `[n]` for a vector |
| `l{i}.{param}.q8` / `.q4` | I8 | quantized matrix bytes, GGML dims `[row_bytes, rows]` (`row_bytes = cols` for int8, `QWeight::row_bytes()` for int4) |
| `l{i}.{param}.scale` | F32 | per-row scales `[rows]` for the `.q8`/`.q4` tensor |

Param names:
- `linear`: `weight` `[in,out]`, `bias`.
- `embedding`: `token` `[vocab,dim]`, `pos` `[seq,dim]`.
- `layernorm`: `gamma`, `beta`.
- `transformer_block`: `ln1.gamma`, `ln1.beta`, `q.weight`, `q.bias`, `k.weight`, `k.bias`, `v.weight`, `v.bias`, `out.weight`, `out.bias`, `ln2.gamma`, `ln2.beta`, `ffn1.weight` `[dim,hidden]`, `ffn1.bias`, `ffn2.weight` `[hidden,dim]`, `ffn2.bias`.

Quantization policy (same as the old v5 writer):
- Matrices with at least `QUANT_MIN_LEN` elements are quantized at the requested precision. Use the **same scale computation the old FINF writer used** (`int8_scales_per_channel` / `QWeight::from_f32` with per-row channels), so a QAT model reloads bit-identically.
- Vectors (biases, LayerNorm) are always F32.
- Embedding tables may be quantized on disk and are dequantized to f32 on load (as before). Linear and transformer projections stay packed `QWeight` in memory.

## Review Focus

1. Reloading an int8 file produced from a QAT-trained model — `evaluate()` must match the in-memory model exactly. → Task G1 test `qat_model_reloads_bit_identically_from_int8_gguf`.
2. A file that is valid GGUF but not a Ferrum model (e.g. a Llama GGUF), and a truncated file — clear `Format` error, no panic. → Task G1 test `rejects_foreign_and_truncated_gguf`.
3. A tabular (Ferrum-ML style) model with a non-empty `Normalizer` and class names — round-trips through GGUF. → Task G1 test `tabular_mlp_roundtrips_with_normalizer`.
4. Resuming from a checkpoint written before this change (FCKP magic) — clear error telling the user to retrain. → Task G3 test `old_checkpoint_magic_is_a_clear_error`.
5. The browser build loading a GGUF model — `slm_wasm` tests pass with GGUF bytes. → Task G2 Step 4.

---

### Task GA: GPT-2-compatible architecture — GELU FFN, no LM-head bias

**Files:**
- Modify: `ferrum_core/src/train_transformer.rs`:
  - FFN forward at `:1058` and the cache field `h_relu`.
  - Backward at `:1153-1175`.
  - Remove `head_b` everywhere it appears (`grep -n head_b`): the struct, `new`, the param lists, grads, step, `num_params`, `to_inference`, the gradient-check test.
- Modify: `ferrum_core/src/layer.rs:736` and `:938` (the `TransformerBlock` inference and cached paths: `max(0.0)` becomes `gelu`).
- Modify: `ferrum_core/src/activation.rs` (add `pub fn gelu(x: f32) -> f32` and `pub fn gelu_grad(x: f32) -> f32`, the tanh approximation llama.cpp's `ggml_gelu` uses).
- Test: `ferrum_core/src/activation.rs`, `ferrum_core/src/train_transformer.rs` tests.

**Interfaces:**
- Produces:
  - `ferrum_core::activation::{gelu, gelu_grad}`.
  - `TransformerNet` has no `head_b`. The inference `Sequential`'s head `Linear` gets an all-zero bias, so the forward pass is unchanged and the GGUF writer omits it.
  - The FFN is `ffn2(gelu(ffn1(x)))`.

- [ ] **Step 1: Failing tests**

`activation.rs`:

```rust
    #[test]
    fn gelu_matches_tanh_approximation_and_its_derivative() {
        let k = (2.0f32 / std::f32::consts::PI).sqrt();
        for &x in &[-6.0f32, -2.0, -0.5, 0.0, 0.3, 1.0, 4.0] {
            let want = 0.5 * x * (1.0 + (k * (x + 0.044715 * x * x * x)).tanh());
            assert!((gelu(x) - want).abs() < 1e-6, "gelu({x})");
            let h = 1e-3;
            let num = (gelu(x + h) - gelu(x - h)) / (2.0 * h);
            assert!((gelu_grad(x) - num).abs() < 1e-3, "gelu'({x}): {} vs {num}", gelu_grad(x));
        }
        assert_eq!(gelu(0.0), 0.0);
    }
```

`train_transformer.rs`: the existing gradient-check test (find it with `grep -n "fn .*grad.*check\|check!(" ferrum_core/src/train_transformer.rs`) must pass with GELU. Delete its `check!(head_b, …)` line. Add:

```rust
    #[test]
    fn head_has_no_bias_and_inference_matches_training_forward() {
        let mut rng = Rng::new(9);
        let net = TransformerNet::new(40, 8, 16, 2, 32, 2, &mut rng).unwrap();
        assert!(net.param_names_for_test().iter().all(|n| !n.contains("head_b")));
        let tokens: Vec<usize> = (0..8).map(|i| i * 5 % 40).collect();
        let (logits, _) = net.forward(&tokens).unwrap();
        let inf = net.to_inference().unwrap();
        let x = crate::tensor::Tensor::matrix(1, 8, tokens.iter().map(|&t| t as f32).collect()).unwrap();
        let probs = inf.forward(&x).unwrap();
        // softmax(logits) == inference output (which ends in Softmax)
        let v = 40;
        for r in 0..8 {
            let row = &logits.data[r * v..(r + 1) * v];
            let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let z: f32 = row.iter().map(|&l| (l - m).exp()).sum();
            for c in 0..v {
                let p = (row[c] - m).exp() / z;
                assert!((p - probs.data[r * v + c]).abs() < 1e-5);
            }
        }
    }
```

(`to_inference_matches_trainable_forward` already exists and may cover the second half. If so, keep only the no-`head_b` assertion. If `param_names_for_test` does not exist, assert instead that `net.num_params()` equals the hand formula for this config, which has no `+ vocab` term.)

- [ ] **Step 2: Verify failure** — `cargo test -p ferrum_core --release --lib gelu head_has_no_bias 2>&1 | tail`. Expected: compile errors.

- [ ] **Step 3: Implement**

```rust
/// GELU, tanh approximation — identical to llama.cpp's `ggml_gelu`, so Ferrum
/// models run unchanged in llama.cpp's GPT-2 graph.
pub fn gelu(x: f32) -> f32 {
    const K: f32 = 0.797_884_6; // sqrt(2/pi)
    0.5 * x * (1.0 + (K * (x + 0.044715 * x * x * x)).tanh())
}

/// d/dx [`gelu`].
pub fn gelu_grad(x: f32) -> f32 {
    const K: f32 = 0.797_884_6;
    let u = K * (x + 0.044715 * x * x * x);
    let t = u.tanh();
    0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * K * (1.0 + 3.0 * 0.044715 * x * x)
}
```

- Training forward: keep the pre-activation `h_pre` in the block cache (rename the `h_relu` field to `h_act` and add `h_pre: Tensor`). Compute `h_act = h_pre.map(gelu)` (then dropout as before).
- Backward: `dh_pre = dh ⊙ mask ⊙ gelu_grad(h_pre)` (without dropout: `dh ⊙ gelu_grad(h_pre)`).
- Inference: `layer.rs:736` and `:938` use `.map(crate::activation::gelu)`.
- Remove `head_b` (the logits are `linear_fwd(&norm_f, &head_w, &zeros)`, or a bias-free matmul if one exists). `to_inference` builds the head `Linear` with `vec![0.0; vocab]`.
- Update doc comments that say ReLU.

- [ ] **Step 4: Verify** — `cargo test --workspace --release 2>&1 | grep -E "^test result|FAILED"`, the wasm32 build, and `(cd ferrum_gui && cargo test)`.
  - Tests that pinned exact losses or generated text from the ReLU model must be **re-derived** (run, read the new value, update it, and say so in the commit message).
  - Tests asserting a property (loss decreases, gradient check, determinism) must pass unchanged.

- [ ] **Step 5: Commit**

```bash
cargo fmt --all && cargo clippy --workspace --all-targets -- -D warnings
git add -A ferrum_core tests ferrum_gui/src slm_wasm
git commit -m "feat!(transformer): GPT-2-compatible blocks — GELU FFN, no LM-head bias

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task GB: QAT simulates GGUF Q8_0 exactly

**Files:**
- Modify: `ferrum_core/src/quant.rs` (add the Q8_0 block helpers)
- Modify: `ferrum_core/src/gguf_write.rs` (its Q8_0 encoder must call the shared helper; find it with `grep -n "q8_0\|Q8_0" ferrum_core/src/gguf_write.rs`)
- Modify: `ferrum_core/src/train_transformer.rs:957` (`fake_quantize_weights`), plus the end of QAT training so the returned inference model holds the snapped weights
- Test: `ferrum_core/src/quant.rs`, `ferrum_core/src/train_transformer.rs`

**Interfaces:**
- Produces:
  ```rust
  /// Q8_0 block size (values per f16 scale).
  pub const Q8_0_BLOCK: usize = 32;
  /// Quantize one block exactly like llama.cpp's quantize_row_q8_0_ref:
  /// d = amax/127; q = round(x * (1/d)); stored scale = f16(d).
  pub fn q8_0_block(x: &[f32]) -> (u16 /*f16 d*/, [i8; 32]);
  /// Snap a GGML-row-contiguous tensor (rows of length `ne0`, ne0 % 32 == 0) onto its Q8_0 values in place.
  pub fn fake_quantize_q8_0_rows(data: &mut [f32], ne0: usize);
  /// Snap a Ferrum `[in, out]` row-major weight as GGUF stores it (GGML ne0 = in: Q8_0 blocks run down each column).
  pub fn fake_quantize_q8_0_cols(data: &mut [f32], rows_in: usize, cols_out: usize);
  /// True when a matrix with GGML row length `ne0` is stored Q8_0 (else F32).
  pub fn q8_0_eligible(ne0: usize) -> bool { ne0 % Q8_0_BLOCK == 0 }
  ```
  `TransformerNet::fake_quantize_weights` snaps exactly the tensors the GGUF writer (Task G1) quantizes, along the same axis:
  - `token_embd` and `position_embd` rows: `ne0 = embed`, via `_rows`
  - attention q/k/v/out and ffn1/ffn2 weights: `ne0 = in_features`, via `_cols`
  - the untied head `[embed, vocab]`: `ne0 = embed`, via `_cols`
  - nothing 1-D
  - nothing whose `ne0 % 32 != 0` (stored F32)

- [ ] **Step 1: Failing tests** (`quant.rs`)

```rust
    #[test]
    fn q8_0_block_matches_llama_cpp_reference() {
        let x: Vec<f32> = (0..32).map(|i| ((i as f32) - 15.5) * 0.37).collect();
        let (d16, q) = q8_0_block(&x);
        let amax = x.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let d = amax / 127.0;
        assert_eq!(d16, crate::gguf_write::f32_to_f16(d));
        for (i, &v) in x.iter().enumerate() {
            assert_eq!(q[i] as f32, (v * (1.0 / d)).round());
        }
        let zeros = q8_0_block(&[0.0; 32]);
        assert_eq!(zeros.1, [0i8; 32]);
    }

    #[test]
    fn q8_0_snap_is_a_fixed_point_of_encode_decode() {
        let mut w: Vec<f32> = (0..96 * 40).map(|i| ((i * 37 % 101) as f32 - 50.0) / 61.0).collect();
        fake_quantize_q8_0_cols(&mut w, 96, 40);
        let once = w.clone();
        fake_quantize_q8_0_cols(&mut w, 96, 40);
        assert_eq!(w, once, "snapping twice must not move values");
    }
```

`train_transformer.rs`:

```rust
    #[test]
    fn qat_snaps_only_q8_0_eligible_matrices() {
        let mut rng = Rng::new(2);
        // embed 48: ne0 = 48 is not a multiple of 32 → those tensors stay f32;
        // hidden 64: ffn2 has ne0 = 64 → snapped.
        let mut net = TransformerNet::new(40, 8, 48, 2, 64, 1, &mut rng).unwrap();
        let before = net.snapshot_weights();
        net.fake_quantize_weights();
        let after = net.snapshot_weights();
        let changed: Vec<bool> = before.iter().zip(&after).map(|(a, b)| a != b).collect();
        assert!(changed.iter().any(|&c| c), "ffn2 must be snapped");
        // 1-D params never change.
        for (t, (a, b)) in net.param_tensors_for_test().iter().zip(before.iter().zip(&after)) {
            if t.shape.len() == 1 {
                assert_eq!(a, b);
            }
        }
    }
```

(Adapt `param_tensors_for_test` to whatever accessor exposes shapes alongside `snapshot_weights`; add a `#[cfg(test)]` one if none exists.)

- [ ] **Step 2: Verify failure** — Expected: compile errors.

- [ ] **Step 3: Implement**
  - Write the helpers, with `q8_0_block` as the single source of truth. Change `gguf_write.rs`'s Q8_0 encoder to call it, so the writer and QAT cannot drift. Export `f32_to_f16`/`f16_to_f32` if they are not already public in the crate.
  - `fake_quantize_q8_0_cols`: for each output column `j` and each block of 32 consecutive input rows `i`, gather `w[i*cols+j]`, snap the block, and scatter it back.
  - Rewrite `fake_quantize_weights` per the interface rules. It needs to know which tensor is which, so use the named fields (`tok_emb`, `pos_emb`, each block's `q_w`/`k_w`/`v_w`/`o_w`/`f1_w`/`f2_w`, and `head_w` when untied) instead of the anonymous `param_tensors_mut()` loop.
  - At the end of a QAT run (`build_transformer_slm` or the code that turns the trained net into the returned `GenerativeSLM`), build the inference model from snapped weights: snapshot, `fake_quantize_weights`, `to_inference`, restore. Saving to int8 then changes nothing.
  - Update the docs on `TransformerConfig::qat` and `set_qat`: "int8 QAT simulating GGUF Q8_0".

- [ ] **Step 4: Verify** — full battery. The existing `qat_model_survives_int8_export_with_small_drift` test may now be stricter (zero drift). Keep it passing, and tighten it to exact equality once Task G1's writer exists.

- [ ] **Step 5: Commit**

```bash
cargo fmt --all && cargo clippy --workspace --all-targets -- -D warnings
git add -A ferrum_core
git commit -m "feat(qat): simulate GGUF Q8_0 exactly (32-blocks, f16 scale, shared encoder)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task GT: GPT-2-compatible tokenizer (llama.cpp `gpt-2` pre-tokenizer, bytes-to-unicode vocab, EOS)

**Files:**
- Modify: `ferrum_core/Cargo.toml` (add `unicode-general-category = "1"`; check `cargo search unicode-general-category` for the current major version and pin it)
- Modify: `ferrum_core/src/tokenizer.rs` (`pretokenize`, `TOK_EOS`, training always registers EOS, the GGUF vocab export/import)
- Test: `ferrum_core/src/tokenizer.rs` tests

**Interfaces:**
- Produces:
  ```rust
  pub const TOK_EOS: &str = "<|endoftext|>";      // was "<eos>"; equals DOC_SEP (Phase 0 Task 3)
  /// GPT-2 byte → unicode char mapping (llama.cpp unicode_byte_to_utf8_map / OpenAI bytes_to_unicode).
  pub fn byte_to_unicode(b: u8) -> char;
  impl ByteBpeTokenizer {
      /// Training always reserves one id for TOK_EOS: merges are learned to vocab_size - 1, then EOS is added.
      pub fn train(corpus: &str, vocab_size: usize) -> Result<Self>;
      /// tokenizer.ggml.tokens: token i's bytes mapped through byte_to_unicode; specials by name.
      pub fn gguf_tokens(&self) -> Vec<String>;
      /// tokenizer.ggml.merges in rank order: "<left> <right>" in byte_to_unicode space.
      pub fn gguf_merges(&self) -> Vec<String>;
      /// tokenizer.ggml.token_type: 1 = normal, 3 = control (specials).
      pub fn gguf_token_types(&self) -> Vec<i32>;
      /// Rebuild from the three arrays (inverse of the above). Errors on a merge whose parts are not tokens.
      pub fn from_gguf_vocab(tokens: &[String], merges: &[String], token_types: &[i32]) -> Result<Self>;
  }
  ```
- `pretokenize(text)` becomes a port of llama.cpp b11521's `unicode_regex_split_custom_gpt2`, i.e. the regex `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)` (with the implicit `\s+` fallback the C++ splitter applies). It still returns a partition of `text`, so encode/decode round-trips. The reference source is at `/tmp/claude-1000/-home-thomas-Projects-ferrum-cpu/88623770-ced2-4a37-be7f-75cb7f78b99e/scratchpad/tools/llama-src/llama.cpp-b11521/src/unicode.cpp:215` (splitter) and `:148` (byte map); read both before writing the port. `\p{L}` and `\p{N}` come from `unicode_general_category::get_general_category` (letter = Lu/Ll/Lt/Lm/Lo; number = Nd/Nl/No). `\s` is `char::is_whitespace`.

- [ ] **Step 1: Failing tests**

```rust
    #[test]
    fn pretokenize_matches_gpt2_regex() {
        let cases: &[(&str, &[&str])] = &[
            ("Hello world", &["Hello", " world"]),
            ("I'm here, it's 2024!", &["I", "'m", " here", ",", " it", "'s", " ", "2024", "!"]),
            ("a  b", &["a", " ", " b"]),
            ("x\n\ny", &["x", "\n", "\n", "y"]),
            ("end   ", &["end", "   "]),
            ("naïve café", &["naïve", " café"]),
            ("<|endoftext|>Once", &["<|", "endoftext", "|>", "Once"]),
        ];
        for (text, want) in cases {
            let got = pretokenize(text);
            assert_eq!(&got, want, "{text:?}");
            assert_eq!(got.concat(), *text);
        }
    }
```

(The expected splits above follow the GPT-2 regex by hand. If the ported splitter disagrees on a case, check that case against llama.cpp: `/tmp/claude-1000/-home-thomas-Projects-ferrum-cpu/88623770-ced2-4a37-be7f-75cb7f78b99e/scratchpad/tools/llama.cpp/llama-b11521/llama-tokenize -m /tmp/claude-1000/-home-thomas-Projects-ferrum-cpu/88623770-ced2-4a37-be7f-75cb7f78b99e/scratchpad/gpt2gguf/gpt2-f32.gguf -p "<text>" --ids` gives GPT-2's ids. Decode them with the HF tokenizer in `/tmp/claude-1000/-home-thomas-Projects-ferrum-cpu/88623770-ced2-4a37-be7f-75cb7f78b99e/scratchpad/venv` to see the pieces. Then fix whichever side is wrong; llama.cpp is the authority.)

```rust
    #[test]
    fn byte_to_unicode_matches_gpt2() {
        assert_eq!(byte_to_unicode(b'A'), 'A');
        assert_eq!(byte_to_unicode(b' '), 'Ġ');
        assert_eq!(byte_to_unicode(b'\n'), 'Ċ');
        let all: std::collections::HashSet<char> = (0..=255u8).map(byte_to_unicode).collect();
        assert_eq!(all.len(), 256);
    }

    #[test]
    fn training_reserves_eos_and_gguf_vocab_roundtrips() {
        let tok = ByteBpeTokenizer::train("the cat sat on the mat. the cat ran. ".repeat(20).as_str(), 300).unwrap();
        assert_eq!(tok.vocab_size(), 300);
        assert_eq!(tok.eos_id(), Some(299));
        let (t, m, ty) = (tok.gguf_tokens(), tok.gguf_merges(), tok.gguf_token_types());
        assert_eq!(t.len(), 300);
        assert_eq!(t[b' ' as usize], "Ġ");
        assert_eq!(t[299], "<|endoftext|>");
        assert_eq!(ty[299], 3);
        assert!(m.iter().all(|x| x.matches(' ').count() == 1));
        let back = ByteBpeTokenizer::from_gguf_vocab(&t, &m, &ty).unwrap();
        assert_eq!(back.encode_state(), tok.encode_state());
        let text = "the cat sat. ünïcode 123";
        assert_eq!(back.encode(text), tok.encode(text));
    }
```

- [ ] **Step 2: Verify failure.** Expected: compile errors, plus a failure on the current whitespace-only `pretokenize`.

- [ ] **Step 3: Implement** per the interface.
  - `train` reserves the EOS id. Phase 0 Task 3's `train_with_docs` then only has to strip separators before training.
  - Leave a `// ponytail:` note in `train` saying EOS is always reserved so every GGUF has a real `eos_token_id`.
  - Update every test and caller that assumed `vocab_size` merges, or the `"<eos>"` name: `grep -rn "TOK_EOS\|<eos>\|vocab_size() - 256\|merges().len()" ferrum_core tests slm_cli slm_wasm ferrum_gui/src`.
  - Tests that pinned exact token ids or counts from the old pre-tokenizer must be re-derived, and listed in the commit message.

- [ ] **Step 4: Verify** — the full battery from Global Constraints.

- [ ] **Step 5: Commit**

```bash
cargo fmt --all && cargo clippy --workspace --all-targets -- -D warnings
git add -A ferrum_core tests slm_cli slm_wasm ferrum_gui/src Cargo.lock
git commit -m "feat!(tokenizer): llama.cpp gpt-2 pre-tokenizer, bytes-to-unicode GGUF vocab, EOS always reserved

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task G1: GGUF model codecs behind the `loader` API (GPT-2 for BPE transformers, generic for the rest)

**Files:**
- Modify: `ferrum_core/src/gguf.rs` (tolerate the generic layout; nothing new is needed for reading Q8_0/Q4_0/F16/F32, because `dequantize` covers them)
- Modify: `ferrum_core/src/loader.rs` (new `to_gguf`, `from_gguf`, `gguf_quant`; FINF stays until G2)
- Test: `ferrum_core/src/loader.rs` tests

**Facts pinned from llama.cpp b11521** (research 2026-10-09; the toolchain ran a converted GPT-2 in llama.cpp, Ollama and LM Studio with identical greedy output):
- **KV keys.** The loader is type-strict.
  - `general.architecture` (str `"gpt2"`).
  - UINT32: `gpt2.context_length`, `gpt2.embedding_length`, `gpt2.block_count`, `gpt2.feed_forward_length`, `gpt2.attention.head_count`.
  - FLOAT32: `gpt2.attention.layer_norm_epsilon` = 1e-5.
  - `tokenizer.ggml.model` = `"gpt2"`, `tokenizer.ggml.pre` = `"gpt-2"`.
  - `tokenizer.ggml.tokens` [string], `tokenizer.ggml.merges` [string] (required), `tokenizer.ggml.token_type` [INT32].
  - UINT32 `tokenizer.ggml.bos_token_id` and `tokenizer.ggml.eos_token_id`, both = the EOS id.
  - BOOL `tokenizer.ggml.add_bos_token` = false and `tokenizer.ggml.add_eos_token` = false.
  - UINT32 `general.file_type` (0 = all F32, 7 = mostly Q8_0, 2 = mostly Q4_0) and UINT32 `general.quantization_version` = 2.
- **Tensors**, as GGML dims `[ne0, ne1]`:

  | Tensor | dims |
  |---|---|
  | `token_embd.weight` | `[n_embd, n_vocab]` |
  | `position_embd.weight` | `[n_embd, n_ctx]` |
  | `blk.N.attn_norm.{weight,bias}` | `[n_embd]` |
  | `blk.N.attn_qkv.weight` | `[n_embd, 3·n_embd]`, rows Q, then K, then V |
  | `blk.N.attn_qkv.bias` | `[3·n_embd]`, Q\|K\|V |
  | `blk.N.attn_output.{weight,bias}` | `[n_embd, n_embd]` / `[n_embd]` |
  | `blk.N.ffn_norm.{weight,bias}` | `[n_embd]` |
  | `blk.N.ffn_up.weight` | `[n_embd, n_ff]`, bias `[n_ff]` |
  | `blk.N.ffn_down.weight` | `[n_ff, n_embd]`, bias `[n_embd]` |
  | `output_norm.{weight,bias}` | `[n_embd]` |
  | `output.weight` | `[n_embd, n_vocab]`, **omitted when tied**: llama.cpp falls back to `token_embd`, and the head has no bias |

  A weight's `ne0` is its **input** dim. Ferrum's row-major `[in, out]` matrix is therefore written transposed: GGML row `o` holds column `o` of Ferrum's matrix.
- **Quant types:**
  - Q8_0 and Q4_0 need `ne0 % 32 == 0`; otherwise the tensor must be F32 (Ferrum's rule) or F16.
  - 1-D tensors are always F32.
  - Following llama-quantize's convention, `position_embd` stays F32 even when quantizing. **GB's QAT must not snap `position_embd`.** If GB snapped it, change GB's `fake_quantize_weights` in this task and say so in the commit message.
  - `encode_tensor(data, ggml_type)` in `gguf_write.rs` encodes GGML-ordered data. Its Q8_0 path is GB's shared `q8_0_block`.
- **Ferrum extras** go under the `ferrum.` namespace, which llama.cpp ignores:
  - `ferrum.format` u32 = 1
  - `ferrum.quant` str `f32` | `int8` | `int4`
  - `ferrum.metadata` str (the `ModelMetadata` JSON minus the tokenizer state, which lives in `tokenizer.ggml.*`)
  - `ferrum.normalizer` str

**Interfaces:**
- Produces:
  ```rust
  pub use QPrec; // F32 | Int8 (→ Q8_0) | Int4 (→ Q4_0)
  pub fn to_gguf(model: &Sequential, norm: &Normalizer, meta: &ModelMetadata, prec: QPrec) -> Result<Vec<u8>>;
  pub fn from_gguf(bytes: &[u8]) -> Result<(Sequential, Normalizer, ModelMetadata)>;
  pub fn gguf_quant(bytes: &[u8]) -> Result<String>;
  ```
  - `to_gguf` picks the **GPT-2 codec** when the model is a BPE transformer SLM: `meta.task == TaskType::TransformerSLM`, non-empty `meta.tokenizer_state`, and layers `Embedding, TransformerBlock×N, LayerNorm, Linear, Softmax`. Anything else uses the **generic codec**: `general.architecture = "ferrum"`, the `ferrum.layer.{i}.*` keys and `l{i}.*` tensors from the layout table at the top of this plan, with the I8 encoding replaced by standard F32/Q8_0/Q4_0 under the same `ne0 % 32` rule and the same transposed `[in,out]` handling.
  - `from_gguf` dispatches on `general.architecture`:
    - `gpt2` builds `Embedding`, the blocks (via `TransformerBlock::from_parts`, splitting QKV), `LayerNorm`, the head `Linear` (zero bias; from `output.weight`, or the transposed `token_embd` when absent) and `Softmax`. All weights are dequantized to f32.
    - The tokenizer comes from `ByteBpeTokenizer::from_gguf_vocab(tokens, merges, token_type)`. `ModelMetadata` is rebuilt from `ferrum.metadata` when present, otherwise from the `gpt2.*` keys, so a **foreign** GPT-2 GGUF (e.g. `/tmp/claude-1000/-home-thomas-Projects-ferrum-cpu/88623770-ced2-4a37-be7f-75cb7f78b99e/scratchpad/gpt2gguf/gpt2-f32.gguf`) also loads.
    - `ferrum` builds the generic layer list.
    - Anything else returns `Err(Format("not a Ferrum-loadable model: architecture …"))`.
  - `gguf_quant` reads `ferrum.quant`, falling back to `general.file_type`.

- [ ] **Step 1: Failing tests**
  - `f32_gguf_roundtrip_is_exact`: a GPT-2-codec model with a BPE tokenizer, built by training a tiny model as in the test below. It must round-trip `forward` bit-exactly, and the tokenizer state must survive.
  - `qat_model_reloads_bit_identically_from_int8_gguf`: train a QAT model with **embed 64** (Q8_0-eligible), save it as Int8, reload, and require `evaluate()` to be equal.
  - `int4_and_ineligible_dims`: an embed-48 model (no tensor eligible) saved as Int8 has every tensor F32 (check with `Gguf::parse` that the types are `GGML_F32`) and round-trips exactly. An embed-64 model saved as Int4 loads and its forward is within 0.1 of f32.
  - `gpt2_layout_matches_llama_cpp`: parse a written file and assert the exact key and tensor names and dims above, including that `output.weight` is absent for a tied model and present for an untied one.
  - `tabular_mlp_roundtrips_with_normalizer` (generic codec, as in the original G1 text below).
  - `rejects_foreign_and_truncated_gguf`: a llama-architecture GGUF is rejected; every truncation point errors without panicking; `b"FINF…"` errors.
  - `loads_foreign_gpt2_gguf`: if `/tmp/claude-1000/-home-thomas-Projects-ferrum-cpu/88623770-ced2-4a37-be7f-75cb7f78b99e/scratchpad/gpt2gguf/gpt2-f32.gguf` exists, load it and check that `generate_tokens` (or a greedy argmax loop over `forward`) continues `"Hello, my name is"` with `" John. I'm a writer"`, which is what llama.cpp and HF produced. Skip with a printed note when the file is absent.

  Use a 0.1M-class tiny config for speed: `--context 64 --embed 64 --heads 4 --blocks 2 --hidden 128 --vocab 512 --tie`.

- [ ] **Step 2: Verify failure.**

- [ ] **Step 3: Implement.**
  - Write shared helpers `put_matrix(name, ferrum_row_major, rows_in, cols_out, prec)`, which transposes to GGML and picks the type by the `ne0` rule, and `put_rows(name, row_major, rows, ne0, prec)` for embeddings, which needs no transpose. Write matching `get_*` readers using `Gguf::dequantize` and transposing back.
  - Both codecs use the same helpers.
  - Map `meta`'s fields (`input_dim` = context, `output_dim` = vocab, etc.) to and from the `gpt2.*` keys.

- [ ] **Step 4: Verify** — the full battery.

- [ ] **Step 5: Commit**

```bash
cargo fmt --all && cargo clippy --workspace --all-targets -- -D warnings
git add -A ferrum_core
git commit -m "feat(loader): GPT-2 GGUF codec (llama.cpp layout) + generic Ferrum GGUF codec

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

<details><summary>Superseded original G1 (custom I8 codec) — reference only for the generic codec's layer/tensor naming and the tabular test</summary>

#### (superseded) Task G1 original text

**Files:**
- Modify: `ferrum_core/src/gguf.rs` (add `GGML_I8 = 24` to `type_nbytes` (1 byte per element); add `pub(crate) fn raw_tensor(&self, name: &str) -> Result<Vec<u8>>` built on the private `tensor_bytes`)
- Modify: `ferrum_core/src/loader.rs` (new GGUF implementation; the FINF code is deleted in Task G2)
- Test: `ferrum_core/src/loader.rs` `#[cfg(test)]`

**Interfaces:**
- Produces (temporary, Task G2 renames them over the FINF functions):
  ```rust
  pub fn to_gguf(model: &Sequential, norm: &Normalizer, meta: &ModelMetadata, prec: QPrec) -> Result<Vec<u8>>;
  pub fn from_gguf(bytes: &[u8]) -> Result<(Sequential, Normalizer, ModelMetadata)>;
  pub fn gguf_quant(bytes: &[u8]) -> Result<String>; // the "ferrum.quant" value
  ```
  `QPrec` is the existing private enum in `loader.rs` (`F32`, `Int8`, `Int4`). Make it `pub` and re-export it from `lib.rs`.

- [ ] **Step 1: Failing tests** (in `loader.rs` tests; reuse the file's existing test helpers for tiny models where they exist)

```rust
    fn tiny_slm_parts() -> (Sequential, Normalizer, ModelMetadata) {
        let mut rng = crate::rng::Rng::new(7);
        let net = crate::train_transformer::TransformerNet::new(40, 8, 16, 2, 32, 2, &mut rng).unwrap();
        let model = net.to_inference().unwrap();
        let meta = ModelMetadata::transformer_slm_for_test(8, 40); // see note below
        (model, Normalizer { means: vec![], stds: vec![] }, meta)
    }

    fn forward_all(m: &Sequential) -> Vec<f32> {
        let ids: Vec<f32> = (0..8).map(|i| (i * 3 % 40) as f32).collect();
        m.forward(&crate::tensor::Tensor::matrix(1, 8, ids).unwrap()).unwrap().data
    }

    #[test]
    fn f32_gguf_roundtrip_is_exact() {
        let (m, n, meta) = tiny_slm_parts();
        let bytes = to_gguf(&m, &n, &meta, QPrec::F32).unwrap();
        assert_eq!(&bytes[..4], b"GGUF");
        let (m2, _, meta2) = from_gguf(&bytes).unwrap();
        assert_eq!(forward_all(&m), forward_all(&m2));
        assert_eq!(meta.to_json(), meta2.to_json());
        assert_eq!(gguf_quant(&bytes).unwrap(), "f32");
    }

    #[test]
    fn int8_and_int4_gguf_roundtrip_is_idempotent_and_close() {
        let (m, n, meta) = tiny_slm_parts();
        for prec in [QPrec::Int8, QPrec::Int4] {
            let a = to_gguf(&m, &n, &meta, prec).unwrap();
            let (ma, _, _) = from_gguf(&a).unwrap();
            let b = to_gguf(&ma, &n, &meta, prec).unwrap();
            assert_eq!(a, b, "re-encoding a loaded quantized model must be byte-identical");
            let (fa, ff) = (forward_all(&ma), forward_all(&m));
            let err = fa.iter().zip(&ff).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
            assert!(err < 0.05, "{prec:?} max abs err {err}");
        }
        let f = to_gguf(&m, &n, &meta, QPrec::F32).unwrap().len();
        let q8 = to_gguf(&m, &n, &meta, QPrec::Int8).unwrap().len();
        assert!(q8 < f, "int8 file must be smaller than f32");
    }

    #[test]
    fn qat_model_reloads_bit_identically_from_int8_gguf() {
        use crate::slm::{GenerativeSLM, TransformerConfig};
        let corpus = "the cat sat on the mat. ".repeat(40);
        let cfg = TransformerConfig { context_len: 8, embed_dim: 16, num_heads: 2, num_blocks: 1,
            hidden_dim: 32, epochs: 2, vocab_size: 270, ..TransformerConfig::default() };
        let slm = GenerativeSLM::train_transformer_config(&corpus, &cfg, &mut crate::rng::Rng::new(1), |_, _| {}).unwrap();
        let bytes = to_gguf(&slm.model, &slm.norm, &slm.meta, QPrec::Int8).unwrap();
        let (m2, n2, meta2) = from_gguf(&bytes).unwrap();
        let back = GenerativeSLM::new(m2, n2, meta2);
        assert_eq!(slm.evaluate(&corpus).unwrap(), back.evaluate(&corpus).unwrap());
    }

    #[test]
    fn tabular_mlp_roundtrips_with_normalizer() {
        use crate::activation::Activation;
        use crate::layer::{ActivationLayer, Linear};
        let mut rng = crate::rng::Rng::new(3);
        let model = Sequential::new()
            .with(Box::new(Linear::new_random(4, 8, &mut rng)))
            .with(Box::new(ActivationLayer::new(Activation::Relu)))
            .with(Box::new(Linear::new_random(8, 3, &mut rng)))
            .with(Box::new(ActivationLayer::new(Activation::Softmax)));
        let norm = Normalizer { means: vec![1.0, 2.0, 3.0, 4.0], stds: vec![0.5, 1.0, 1.5, 2.0] };
        let meta = ModelMetadata::classifier_for_test(4, &["a", "b", "c"]);
        let bytes = to_gguf(&model, &norm, &meta, QPrec::F32).unwrap();
        let (m2, n2, meta2) = from_gguf(&bytes).unwrap();
        assert_eq!(n2.encode(), norm.encode());
        assert_eq!(meta2.class_names, meta.class_names);
        let x = crate::tensor::Tensor::matrix(1, 4, vec![0.1, -0.2, 0.3, 0.9]).unwrap();
        assert_eq!(model.forward(&x).unwrap().data, m2.forward(&x).unwrap().data);
    }

    #[test]
    fn rejects_foreign_and_truncated_gguf() {
        let mut b = crate::gguf_write::GgufBuilder::new();
        b.meta("general.architecture", crate::gguf::MetaValue::String("llama".into()));
        let foreign = b.into_bytes();
        assert!(from_gguf(&foreign).unwrap_err().to_string().contains("not a Ferrum model"));
        let (m, n, meta) = tiny_slm_parts();
        let good = to_gguf(&m, &n, &meta, QPrec::Int8).unwrap();
        for cut in [3, 20, good.len() / 2, good.len() - 1] {
            assert!(from_gguf(&good[..cut]).is_err(), "truncated at {cut} must error");
        }
        assert!(from_gguf(b"FINF\x05\x00\x00\x00").is_err());
    }
```

Note: the `ModelMetadata::*_for_test` and `Linear::new_random` helper names above are placeholders for **whatever constructors the existing loader/meta tests already use**. Before writing the tests, run `grep -n "fn .*(" ferrum_core/src/meta.rs | head -30` and `grep -n "ModelMetadata {" -A12 ferrum_core/src/loader.rs | head -40` and build the metadata the same way the existing FINF tests do. If no such helper exists, construct `ModelMetadata { … }` literally as the existing tests do.

- [ ] **Step 2: Verify failure** — `cargo test -p ferrum_core --release --lib loader 2>&1 | tail -5`. Expected: compile error (`to_gguf` not found).

- [ ] **Step 3: Implement**

In `gguf.rs`:
- add `pub(crate) const GGML_I8: u32 = 24;`
- add an `GGML_I8 => Ok(n)` arm in `type_nbytes`
- add:

```rust
    /// Raw bytes of tensor `name` (any type), copied out.
    pub(crate) fn raw_tensor(&self, name: &str) -> Result<Vec<u8>> {
        let t = self.tensor(name).ok_or_else(|| fmt(&format!("missing tensor '{name}'")))?;
        Ok(self.tensor_bytes(t)?.into_owned())
    }
```

In `loader.rs`, add the writer:

```rust
use crate::gguf::{Gguf, MetaValue, GGML_F32, GGML_I8};
use crate::gguf_write::GgufBuilder;

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Writes one Ferrum model into a GGUF builder.
struct GgufOut {
    b: GgufBuilder,
    prec: QPrec,
}

impl GgufOut {
    fn u32(&mut self, key: &str, v: usize) {
        self.b.meta(key, MetaValue::U32(v as u32));
    }
    fn vec(&mut self, name: &str, v: &[f32]) {
        self.b.tensor(name, &[v.len() as u64], GGML_F32, f32_bytes(v));
    }
    /// A row-major `[rows, cols]` matrix: quantized per row when large enough.
    fn mat(&mut self, name: &str, data: &[f32], rows: usize, cols: usize) {
        let kind = match self.prec {
            QPrec::Int8 if data.len() >= QUANT_MIN_LEN => Some(QKind::Int8),
            QPrec::Int4 if data.len() >= QUANT_MIN_LEN => Some(QKind::Int4),
            _ => None,
        };
        match kind {
            None => {
                self.b.tensor(name, &[cols as u64, rows as u64], GGML_F32, f32_bytes(data));
            }
            Some(kind) => {
                let q = QWeight::from_f32(data, rows, cols, kind);
                let suffix = if kind == QKind::Int8 { "q8" } else { "q4" };
                let row_bytes = q.q.len() / rows.max(1);
                self.b.tensor(&format!("{name}.{suffix}"), &[row_bytes as u64, rows as u64], GGML_I8, q.q.clone());
                self.b.tensor(&format!("{name}.scale"), &[rows as u64], GGML_F32, f32_bytes(&q.scales));
            }
        }
    }
}

pub fn to_gguf(model: &Sequential, norm: &Normalizer, meta: &ModelMetadata, prec: QPrec) -> Result<Vec<u8>> {
    let mut o = GgufOut { b: GgufBuilder::new(), prec };
    o.b.meta("general.architecture", MetaValue::String("ferrum".into()));
    o.b.meta("general.name", MetaValue::String(meta.dataset_name.clone()));
    o.u32("ferrum.format", 1);
    let q = match prec { QPrec::F32 => "f32", QPrec::Int8 => "int8", QPrec::Int4 => "int4" };
    o.b.meta("ferrum.quant", MetaValue::String(q.into()));
    o.b.meta("ferrum.metadata", MetaValue::String(meta.to_json()));
    o.b.meta("ferrum.normalizer", MetaValue::String(norm.encode()));
    o.u32("ferrum.layer_count", model.len());
    for (i, layer) in model.layers().iter().enumerate() {
        let k = |s: &str| format!("ferrum.layer.{i}.{s}");
        let t = |s: &str| format!("l{i}.{s}");
        let any = layer.as_any();
        let ty = |o: &mut GgufOut, s: &str| o.b.meta(&k("type"), MetaValue::String(s.into()));
        if let Some(lin) = any.downcast_ref::<Linear>() {
            ty(&mut o, "linear");
            o.u32(&k("in"), lin.in_features());
            o.u32(&k("out"), lin.out_features());
            o.mat(&t("weight"), &lin.weight_f32(), lin.in_features(), lin.out_features());
            o.vec(&t("bias"), &lin.bias.data);
        } else if let Some(act) = any.downcast_ref::<ActivationLayer>() {
            ty(&mut o, "activation");
            o.u32(&k("kind"), act.kind.tag() as usize);
        } else if any.downcast_ref::<Flatten>().is_some() {
            ty(&mut o, "flatten");
        } else if let Some(emb) = any.downcast_ref::<Embedding>() {
            ty(&mut o, "embedding");
            o.u32(&k("vocab"), emb.vocab_size());
            o.u32(&k("seq"), emb.max_seq_len());
            o.u32(&k("dim"), emb.embedding_dim());
            o.mat(&t("token"), &emb.token_weight.data, emb.vocab_size(), emb.embedding_dim());
            o.mat(&t("pos"), &emb.pos_weight.data, emb.max_seq_len(), emb.embedding_dim());
        } else if let Some(ln) = any.downcast_ref::<LayerNorm>() {
            ty(&mut o, "layernorm");
            o.u32(&k("dim"), ln.dim());
            o.vec(&t("gamma"), &ln.gamma.data);
            o.vec(&t("beta"), &ln.beta.data);
        } else if let Some(tb) = any.downcast_ref::<TransformerBlock>() {
            ty(&mut o, "transformer_block");
            let (c, h) = (tb.embedding_dim(), tb.hidden_dim());
            o.u32(&k("context"), tb.context_len());
            o.u32(&k("heads"), tb.num_heads());
            o.u32(&k("dim"), c);
            o.u32(&k("hidden"), h);
            o.vec(&t("ln1.gamma"), &tb.ln1.gamma.data);
            o.vec(&t("ln1.beta"), &tb.ln1.beta.data);
            for (name, lin, r, cl) in [("q", &tb.q_proj, c, c), ("k", &tb.k_proj, c, c), ("v", &tb.v_proj, c, c),
                                       ("out", &tb.out_proj, c, c), ("ffn1", &tb.ffn1, c, h), ("ffn2", &tb.ffn2, h, c)] {
                if name == "ffn1" {
                    o.vec(&t("ln2.gamma"), &tb.ln2.gamma.data);
                    o.vec(&t("ln2.beta"), &tb.ln2.beta.data);
                }
                o.mat(&t(&format!("{name}.weight")), &lin.weight_f32(), r, cl);
                o.vec(&t(&format!("{name}.bias")), &lin.bias.data);
            }
        } else {
            return Err(InferError::Format(format!("unknown layer type: {}", layer.name())));
        }
    }
    Ok(o.b.into_bytes())
}
```

Check before relying on it: is `QWeight::from_f32(data, rows, cols, kind)` the same per-row scale computation the FINF v5 writer applied (`push_weights` → `int8_scales_per_channel` with `channels = rows`)?
- If yes, use it as written.
- If not, build the `QWeight` the way `push_weights` did, so QAT models stay bit-identical. The `qat_model_reloads_bit_identically_from_int8_gguf` test enforces this.

Reader:

```rust
struct GgufIn {
    g: Gguf,
}

impl GgufIn {
    fn u32(&self, key: &str) -> Result<usize> {
        self.g.meta(key).and_then(|v| v.as_usize()).ok_or_else(|| InferError::Format(format!("missing or bad key {key}")))
    }
    fn str(&self, key: &str) -> Result<&str> {
        self.g.meta(key).and_then(|v| v.as_str()).ok_or_else(|| InferError::Format(format!("missing key {key}")))
    }
    fn f32s(&self, name: &str, n: usize) -> Result<Vec<f32>> {
        let t = self.g.tensor(name).ok_or_else(|| InferError::Format(format!("missing tensor '{name}'")))?;
        if t.ggml_type != GGML_F32 || t.num_elements() != n {
            return Err(InferError::Format(format!("tensor '{name}': expected {n} f32 values")));
        }
        self.g.dequantize(t)
    }
    fn mat(&self, name: &str, rows: usize, cols: usize) -> Result<LoadedWeights> {
        let n = mul_dims(rows, cols)?;
        for (suffix, kind) in [("q8", QKind::Int8), ("q4", QKind::Int4)] {
            let qn = format!("{name}.{suffix}");
            if self.g.tensor(&qn).is_some() {
                let q = self.g.raw_tensor(&qn)?;
                let scales = self.f32s(&format!("{name}.scale"), rows)?;
                let w = QWeight { rows, cols, kind, scales, q };
                if w.q.len() != rows * w.row_bytes() {
                    return Err(InferError::Format(format!("tensor '{qn}' has the wrong size")));
                }
                return Ok(LoadedWeights::Quant(w));
            }
        }
        Ok(LoadedWeights::F32(self.f32s(name, n)?))
    }
}

pub fn from_gguf(bytes: &[u8]) -> Result<(Sequential, Normalizer, ModelMetadata)> {
    let g = Gguf::parse(bytes.to_vec())?;
    if g.architecture() != Some("ferrum") {
        return Err(InferError::Format(format!(
            "not a Ferrum model (general.architecture = {:?})",
            g.architecture()
        )));
    }
    let r = GgufIn { g };
    if r.u32("ferrum.format")? != 1 {
        return Err(InferError::Format("unsupported ferrum.format".into()));
    }
    let norm_s = r.str("ferrum.normalizer")?;
    let norm = if norm_s.is_empty() { Normalizer { means: vec![], stds: vec![] } } else { Normalizer::decode(norm_s)? };
    let meta = ModelMetadata::from_json(r.str("ferrum.metadata")?)?;
    let mut model = Sequential::new();
    for i in 0..r.u32("ferrum.layer_count")? {
        let k = |s: &str| format!("ferrum.layer.{i}.{s}");
        let t = |s: &str| format!("l{i}.{s}");
        match r.str(&k("type"))? {
            "linear" => {
                let (inf, outf) = (r.u32(&k("in"))?, r.u32(&k("out"))?);
                let w = r.mat(&t("weight"), inf, outf)?;
                model.push(Box::new(linear_from_loaded(inf, outf, w, r.f32s(&t("bias"), outf)?)?));
            }
            "activation" => {
                let tag = r.u32(&k("kind"))? as u8;
                let act = Activation::from_tag(tag).ok_or_else(|| InferError::Format(format!("bad activation {tag}")))?;
                model.push(Box::new(ActivationLayer::new(act)));
            }
            "flatten" => model.push(Box::new(Flatten::new())),
            "embedding" => {
                let (v, s, d) = (r.u32(&k("vocab"))?, r.u32(&k("seq"))?, r.u32(&k("dim"))?);
                let tok = r.mat(&t("token"), v, d)?.into_f32();
                let pos = r.mat(&t("pos"), s, d)?.into_f32();
                model.push(Box::new(Embedding::new(v, s, d, tok, pos)?));
            }
            "layernorm" => {
                let d = r.u32(&k("dim"))?;
                model.push(Box::new(LayerNorm::new(d, r.f32s(&t("gamma"), d)?, r.f32s(&t("beta"), d)?)?));
            }
            "transformer_block" => {
                let (ctx, heads, c, h) = (r.u32(&k("context"))?, r.u32(&k("heads"))?, r.u32(&k("dim"))?, r.u32(&k("hidden"))?);
                let lin = |name: &str, rows: usize, cols: usize| -> Result<Linear> {
                    let w = r.mat(&t(&format!("{name}.weight")), rows, cols)?;
                    linear_from_loaded(rows, cols, w, r.f32s(&t(&format!("{name}.bias")), cols)?)
                };
                let ln = |name: &str| -> Result<LayerNorm> {
                    LayerNorm::new(c, r.f32s(&t(&format!("{name}.gamma")), c)?, r.f32s(&t(&format!("{name}.beta")), c)?)
                };
                model.push(Box::new(TransformerBlock::from_parts(
                    ctx, heads, c, ln("ln1")?, lin("q", c, c)?, lin("k", c, c)?, lin("v", c, c)?,
                    lin("out", c, c)?, ln("ln2")?, lin("ffn1", c, h)?, lin("ffn2", h, c)?,
                )?));
            }
            other => return Err(InferError::Format(format!("unknown layer type '{other}'"))),
        }
    }
    Ok((model, norm, meta))
}

/// The `ferrum.quant` value ("f32" | "int8" | "int4") of a Ferrum GGUF model.
pub fn gguf_quant(bytes: &[u8]) -> Result<String> {
    let g = Gguf::parse(bytes.to_vec())?;
    g.meta("ferrum.quant").and_then(|v| v.as_str()).map(String::from)
        .ok_or_else(|| InferError::Format("not a Ferrum model (no ferrum.quant)".into()))
}
```

`Gguf::parse` is assumed to reject truncated input; the truncation test checks it. If a cut point panics instead of erroring, fix the reader in `gguf.rs` (it must never panic on untrusted input).

- [ ] **Step 4: Verify** — `cargo test -p ferrum_core --release --lib loader 2>&1 | grep -E "test |panicked|result"`. Expected: the 5 new tests pass alongside the existing FINF tests.

- [ ] **Step 5: Commit**

```bash
cargo fmt --all && cargo clippy --workspace --all-targets -- -D warnings
git add ferrum_core
git commit -m "feat(loader): Ferrum GGUF model codec (exact per-row int8/int4 + f32)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---


</details>

---

### Task G2: Switch every model file to GGUF, convert committed models, delete FINF

**Files:**
- Modify: `ferrum_core/src/loader.rs` (public functions now call `to_gguf`/`from_gguf`; FINF code and the module doc deleted)
- Modify: `ferrum_core/src/slm.rs`, `ferrum_core/src/lib.rs`, `slm_cli/src/main.rs` (`info`), `slm_wasm/src/lib.rs`, `ferrum_gui/src/commands.rs`, `ferrum_gui/ui/*` (any `.bin`/FINF strings), the tests in `tests/tests/*` and `slm_cli/tests/*` that mention FINF or `.bin`
- Modify (separate repo): `../Ferrum-ML` (models converted, paths `.bin` → `.gguf`)
- Temporary (not committed): `ferrum_core/examples/finf2gguf.rs`

**Interfaces:**
- Produces:
  - `loader::{to_bytes, to_bytes_quantized, to_bytes_quantized_int4, from_bytes, save, save_quantized, save_quantized_int4, load}`, with the same signatures as before but GGUF content.
  - `pub fn loader::gguf_quant(bytes) -> Result<String>`.
  - The default model extension everywhere in code, tests and usage text becomes `.gguf`.

- [ ] **Step 1: Convert the committed FINF models while the FINF reader still exists**

```bash
grep -rl --include=*.bin -l "FINF" /home/thomas/Projects/ferrum-cpu/Ferrum /home/thomas/Projects/ferrum-cpu/Ferrum-ML 2>/dev/null | grep -v target
```

Create `ferrum_core/examples/finf2gguf.rs`:

```rust
//! One-off: convert FINF model files to Ferrum GGUF, keeping their precision.
fn main() {
    for path in std::env::args().skip(1) {
        let bytes = std::fs::read(&path).unwrap();
        let ver = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let (m, n, meta) = ferrum_core::loader::from_bytes(&bytes).unwrap();
        let prec = if ver == 5 { ferrum_core::loader::QPrec::Int8 } else { ferrum_core::loader::QPrec::F32 };
        // Note: FINF SLMs were trained with ReLU and a head bias, so they cannot be converted faithfully after Task GA.
        // Convert only non-transformer (tabular) models; old SLM .bin files are retrained, not converted.
        let out = path.trim_end_matches(".bin").to_string() + ".gguf";
        std::fs::write(&out, ferrum_core::loader::to_gguf(&m, &n, &meta, prec).unwrap()).unwrap();
        println!("{path} → {out}");
    }
}
```

(If a v5 file holds int4 weights, detect it from the loaded `QWeight` kinds and use `QPrec::Int4`.)

Run it on every file the grep found:

```bash
cargo run --release -p ferrum_core --example finf2gguf -- <files>
```

For each converted file, check that the GGUF model gives the same predictions:
- tabular: run Ferrum-ML's own tests after Step 3
- SLM: `evaluate` on any text, before vs after

Then delete the example file.

- [ ] **Step 2: Swap the implementation and delete FINF**

- In `loader.rs`:
  - `to_bytes` → `to_gguf(.., QPrec::F32)`; `to_bytes_quantized` → `QPrec::Int8`; `to_bytes_quantized_int4` → `QPrec::Int4`; `from_bytes` → `from_gguf`.
  - `save*`/`load` keep writing and reading files through those.
  - Delete `MAGIC`, `VERSION*`, `TAG_*`, `ENC_*`, the `Reader` type, `push_*`, `to_bytes_impl`, the old `from_bytes` body, and the FINF module doc. Write a new module doc describing the GGUF layout table from this plan.
  - Convert the FINF-specific loader tests to the GGUF API or delete them where Task G1's tests already cover the same property.
- `slm.rs`: update the doc comments that say "FINF v4/v5" to "Ferrum GGUF".
- `slm_cli/src/main.rs`:
  - `info` prints `Format    : GGUF v{version} (ferrum, {quant})` using `ferrum_core::Gguf::parse` and `loader::gguf_quant`.
  - Change the save message `(int8-quantized FINF v5)` to `(int8 GGUF)`.
  - Use `.gguf` in usage text and examples.
- `slm_wasm`, `ferrum_gui`: replace `.bin` defaults and FINF strings. Find them with `grep -rn -i "finf\|\.bin\b" slm_wasm/src ferrum_gui/src ferrum_gui/ui slm_cli tests`.
- Tests: rename `*.bin` temp paths to `*.gguf`, and change any assertion on `b"FINF"` to `b"GGUF"`.

- [ ] **Step 3: Ferrum-ML**

```bash
cd /home/thomas/Projects/ferrum-cpu/Ferrum-ML
git rm -q models/*/model.bin
grep -rn -i "model\.bin\|finf" --include=*.rs --include=*.sh --include=*.md --include=*.toml --include=*.js --include=*.html . | grep -v target
# update every hit to model.gguf / GGUF
cargo test --release 2>&1 | grep -E "^test result|FAILED"
git add -A && git commit -m "chore: models are Ferrum GGUF (FINF removed upstream)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
cd ../Ferrum
```

- [ ] **Step 4: Verify** — the full battery from Global Constraints, plus `wasm-pack`-free `cargo test -p slm_wasm --release` (its native tests load GGUF bytes now), plus `grep -rn -i "finf" --include=*.rs . | grep -v target`. That grep must show no hits except the Status/history docs, which Phase 2 rewrites.

- [ ] **Step 5: Commit**

```bash
cargo fmt --all && cargo clippy --workspace --all-targets -- -D warnings
git add -A
git commit -m "refactor!: all model files are GGUF; FINF format removed

BREAKING: FINF (.bin) files no longer load. Committed models were
converted (Ferrum-ML: 10 tabular models).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task G3: Checkpoints and the tokenizer file as GGUF

**Files:**
- Modify: `ferrum_core/src/train_transformer.rs:812-880` (`save_checkpoint` / `load_checkpoint`)
- Modify: `ferrum_core/src/llm_train.rs:1384-1460` (`save_checkpoint` / `load_checkpoint_into`)
- Modify: `ferrum_core/src/tokenizer.rs` (`to_gguf_bytes`, `from_gguf_bytes`), `ferrum_core/src/slm.rs` (`tokenize_for_lm_cfg` reads/writes the tokenizer file through them)
- Modify: `ferrum_gui/src/commands.rs` and `slm_cli/src/main.rs` if they write or read the tokenizer file directly (`grep -rn "encode_state\|from_state" ferrum_gui/src slm_cli/src`)

**Interfaces:**
- Produces: `save_checkpoint(&self, rng) -> Vec<u8>` (GGUF bytes) and `load_checkpoint(bytes)` with unchanged signatures. Layout:
  - `general.architecture = "ferrum-checkpoint"` (or `"ferrum-llama-checkpoint"`)
  - `ferrum.ckpt.version = 1`
  - every scalar the old header carried as a `U32`/`U64`/`Bool` key (`vocab`, `context`, `embed`, `heads`, `hidden`, `blocks`, `qat`, `tie`, `step`, `rng`)
  - one F32 tensor per parameter: `p{i}.data`, `p{i}.m`, `p{i}.v`, in `params_all()` order
  - For `llm_train`, port every field its `save_checkpoint` writes, the same way.
- Produces: `ByteBpeTokenizer::to_gguf_bytes(&self) -> Vec<u8>` (`general.architecture = "ferrum-tokenizer"`, `tokenizer.ferrum.state = encode_state()`) and `ByteBpeTokenizer::from_gguf_bytes(&[u8]) -> Result<Self>`.

- [ ] **Step 1: Failing tests**

In `train_transformer.rs` tests:

```rust
    #[test]
    fn checkpoint_is_gguf_and_roundtrips() {
        let mut rng = Rng::new(5);
        let net = TransformerNet::new(30, 8, 16, 2, 32, 1, &mut rng).unwrap();
        let bytes = net.save_checkpoint(&rng);
        assert_eq!(&bytes[..4], b"GGUF");
        let g = crate::gguf::Gguf::parse(bytes.clone()).unwrap();
        assert_eq!(g.architecture(), Some("ferrum-checkpoint"));
        let (net2, rng2) = TransformerNet::load_checkpoint(&bytes).unwrap();
        assert_eq!(net2.save_checkpoint(&rng2), bytes);
    }

    #[test]
    fn old_checkpoint_magic_is_a_clear_error() {
        let err = TransformerNet::load_checkpoint(b"FCKP\x01\x00\x00\x00").unwrap_err().to_string();
        assert!(err.contains("pre-GGUF checkpoint"), "{err}");
    }
```

An equivalent pair goes in `llm_train.rs`, using its existing tiny-model test helper and `load_checkpoint_into`. In `tokenizer.rs`:

```rust
    #[test]
    fn tokenizer_file_is_gguf_and_roundtrips() {
        let mut tok = ByteBpeTokenizer::train("low lower lowest low low", 300).unwrap();
        tok.add_special_token(TOK_EOS).unwrap();
        let b = tok.to_gguf_bytes();
        assert_eq!(&b[..4], b"GGUF");
        let back = ByteBpeTokenizer::from_gguf_bytes(&b).unwrap();
        assert_eq!(back.encode_state(), tok.encode_state());
        assert!(ByteBpeTokenizer::from_gguf_bytes(b"256;257").is_err());
    }
```

- [ ] **Step 2: Verify failure.** Expected: assertion failures (magic is `FCKP`) and compile errors (`to_gguf_bytes`).

- [ ] **Step 3: Implement** — rewrite the four functions with `GgufBuilder` and `Gguf::parse`:
- Read the existing bodies first and carry **every** field across.
- `load_checkpoint` checks for the `FCKP`/`FLCK` magic first and returns `InferError::Format("pre-GGUF checkpoint (FCKP); it cannot be resumed — retrain or start a new checkpoint")`.
- `tokenize_for_lm_cfg` writes `tok.to_gguf_bytes()` with `std::fs::write` and reads with `from_gguf_bytes(&std::fs::read(path)?)`.

- [ ] **Step 4: Verify** — the full battery. The existing resume tests (`slm.rs`, `test_gguf_finetune.rs`, GUI) must pass unchanged.

- [ ] **Step 5: Commit**

```bash
cargo fmt --all && cargo clippy --workspace --all-targets -- -D warnings
git add -A
git commit -m "refactor!: training checkpoints and tokenizer files are GGUF

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task G4: Forbid `unsafe` in every crate

**Files:**
- Modify: `slm_cli/src/main.rs`, `slm_wasm/src/lib.rs`, `ferrum_gui/src/lib.rs`, `ferrum_gui/src/main.rs`, `tests/` crate root(s), and in `../Ferrum-ML`: `tabular_wasm/src/lib.rs`, `ferrum_ml/src/bin/train_cli.rs`

- [ ] **Step 1:** Add `#![forbid(unsafe_code)]` as the first inner attribute of each crate root listed. Find the remaining roots with `grep -rL "forbid(unsafe_code)" $(git ls-files '*/src/lib.rs' '*/src/main.rs' '*/src/bin/*.rs')`.
- [ ] **Step 2: Build every target**, including `cargo build -p slm_wasm --target wasm32-unknown-unknown --release` and `cd ferrum_gui && cargo check --all-targets`. A proc macro from wasm-bindgen or Tauri may emit `unsafe` blocks that `forbid` rejects. If so, use `#![deny(unsafe_code)]` for that crate only, with the comment `// deny, not forbid: <macro> expands to unsafe internally; this crate contains no unsafe code.` Never add an `unsafe` block or `#[allow(unsafe_code)]` of our own.
- [ ] **Step 3: Add a guard test** — `tests/tests/test_no_unsafe.rs`:

```rust
//! Repository invariant: no `unsafe` code in any Ferrum crate.
#[test]
fn no_unsafe_blocks_or_impls_in_sources() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf();
    let mut bad = Vec::new();
    let mut stack = vec![root.clone()];
    while let Some(dir) = stack.pop() {
        for e in std::fs::read_dir(&dir).unwrap() {
            let p = e.unwrap().path();
            let name = p.file_name().unwrap().to_string_lossy().to_string();
            if p.is_dir() {
                if !["target", ".git", "node_modules", "gen"].contains(&name.as_str()) {
                    stack.push(p);
                }
            } else if name.ends_with(".rs") && name != "test_no_unsafe.rs" {
                let src = std::fs::read_to_string(&p).unwrap();
                for (i, line) in src.lines().enumerate() {
                    let code = line.split("//").next().unwrap_or("");
                    if code.contains("unsafe ") || code.contains("unsafe{") {
                        bad.push(format!("{}:{}: {}", p.display(), i + 1, line.trim()));
                    }
                }
            }
        }
    }
    assert!(bad.is_empty(), "unsafe code found:\n{}", bad.join("\n"));
}
```

- [ ] **Step 4: Verify and commit** — the full battery, then:

```bash
git add -A
git commit -m "chore: forbid unsafe_code in every crate + repo-wide no-unsafe test

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
(cd ../Ferrum-ML && cargo test --release 2>&1 | grep -E "^test result|FAILED" && git add -A && git commit -m "chore: forbid unsafe_code in every crate

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>")
```

---

### Task G5: Run Ferrum GGUFs in llama.cpp, Ollama, and LM Studio (compat tests) — execute after Phase 0 Task 4

**Files:**
- Create: `scripts/verify_runtimes.sh` (based on `/tmp/claude-1000/-home-thomas-Projects-ferrum-cpu/88623770-ced2-4a37-be7f-75cb7f78b99e/scratchpad/gpt2gguf/verify.sh`; tool paths come from env vars `FERRUM_LLAMA_BIN` (the dir holding `llama-completion` and `llama-tokenize`), `FERRUM_OLLAMA` (the `ollama` binary), and `FERRUM_LMS` (default `~/.lmstudio/bin/lms`). Any tool that is not found is skipped with a message.)
- Create: `tests/tests/test_runtime_compat.rs`
- Modify: `.github/workflows/ci.yml` (new job `runtimes`)

**Tool discovery:** env var, then `PATH`, then the default location: `~/.local/opt/llama.cpp-b11521/` (installed on this machine; `llama-completion`, `llama-tokenize`), `~/.local/bin/ollama` (v0.40.2), `~/.lmstudio/bin/lms` (0.4.18). With `FERRUM_REQUIRE_RUNTIMES=1`, a tool that can't be found is a **failure**, not a skip.

**CI job `runtimes`** (ubuntu-latest):
- download the llama.cpp `b11521` Linux x64 CPU release zip and the Ollama `v0.40.2` linux-amd64 tarball, both by pinned URL, verifying their sha256 (record the hashes when writing the job)
- start `ollama serve`
- run `FERRUM_REQUIRE_RUNTIMES=1 FERRUM_SKIP_LMSTUDIO=1 cargo test --release --test test_runtime_compat`
LM Studio can't run in CI (it needs its desktop app), so `FERRUM_SKIP_LMSTUDIO=1` is the one allowed exemption there. The LM Studio test is required in local pre-release runs.

**Pinned commands** (b11521, Ollama v0.40.2, LM Studio 0.4.18):
- llama.cpp: `$FERRUM_LLAMA_BIN/llama-completion -m X.gguf -p "<prompt>" -n 16 --temp 0 -no-cnv --no-warmup </dev/null 2>/dev/null`
- llama.cpp tokenizer: `$FERRUM_LLAMA_BIN/llama-tokenize -m X.gguf -p "<text>" --ids --log-disable`
- Ollama:
  - `OLLAMA_HOST=127.0.0.1:11545 OLLAMA_MODELS=<dir> $FERRUM_OLLAMA serve &`
  - a Modelfile containing `FROM /abs/X.gguf`, then `ollama create NAME -f Modelfile`
  - `curl -s http://127.0.0.1:11545/api/generate -d '{"model":"NAME","prompt":"<p>","raw":true,"stream":false,"options":{"temperature":0,"num_predict":16}}'`
- LM Studio:
  - `lms import -y -l --user-repo ferrum-test/x X.gguf`
  - the key from `lms ls --json`, matched on `path`
  - `lms load KEY --gpu off --identifier ID -y`, then `lms server start -p 1234`
  - `POST /v1/completions` with `max_tokens` 17 (it returns N−1 visible tokens), `temperature` 0 and `repeat_penalty` 1.0
  - unload and delete the import afterwards

- [ ] **Step 1: Write the tests** (each one skips with `eprintln!("skip: …")` and returns when its tool or env var is missing)
  - `tokenizer_matches_llama_cpp`: train a tiny BPE model (the 0.1M config) on a fixed small corpus and save it as F32 GGUF. For strings covering contractions, digits, punctuation runs, multiple spaces, newlines, non-ASCII and the literal `<|endoftext|>`, Ferrum's `encode_docs` ids must equal `llama-tokenize --ids` output.
  - `llama_cpp_greedy_matches_ferrum_f32`: train the **0.1M-param** model (`--context 64 --embed 64 --heads 4 --blocks 2 --hidden 128 --vocab 512 --tie`, 103,936 params) for enough steps that its distribution is peaked (a repetitive corpus, e.g. 200 copies of a short story). Save it as F32. Ferrum's `generate_tokens(prompt, 16, &[], greedy)` text must equal llama.cpp's 16-token output.
  - `llama_cpp_runs_q8_0_and_q4_0`: the same model saved Int8 and Int4. llama.cpp loads both and prints 16 tokens. For Q8_0, also compare to Ferrum's greedy output from the reloaded file and print the first mismatch index. Assert equality only on the first 8 tokens, because llama.cpp quantizes activations for Q8_0 matmuls, which can flip near-ties late in a tiny model.
  - `ollama_and_lm_studio_run_ferrum_gguf`: runs only when `FERRUM_VERIFY_RUNTIMES=1`. Invokes `scripts/verify_runtimes.sh` on the F32 file and asserts that each available tool returned non-empty text equal to llama.cpp's.
  - Also the **507,552-param reference config** (embed 96): train for a few steps, save as Int8, and require that llama.cpp loads it and generates. This proves the larger end too.

- [ ] **Step 2: Run them** with the tools installed in the scratchpad:

```bash
export FERRUM_REQUIRE_RUNTIMES=1 FERRUM_VERIFY_RUNTIMES=1   # tools found at their default locations
cargo test -p integration_tests --release --test test_runtime_compat -- --nocapture --test-threads 1
```

(Use the integration crate's real package name from `tests/Cargo.toml`.) Expected: every test passes, none skipped (the flag makes a skip fail). Also add a unit test showing that a missing tool under `FERRUM_REQUIRE_RUNTIMES=1` fails. Share the helper with plan 1E (`require_or_skip`). A greedy mismatch in the F32 test is a real bug: find the first diverging op (tokenizer ids, then logits of the first step via a single forward) rather than loosening the assertion.

- [ ] **Step 3: Commit**

```bash
git add scripts/verify_runtimes.sh tests/tests/test_runtime_compat.rs
git commit -m "test: Ferrum GGUFs run in llama.cpp, Ollama, LM Studio (0.1M and 0.5M configs)

<paste the three tools' outputs for the 0.1M model>

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
