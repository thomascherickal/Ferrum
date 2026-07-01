# GGUF Export — Write GGUF Files From Ferrum Models

Date: 2026-07-01
Status: Approved (design)

## Purpose

Ferrum can **read** GGUF (import Llama/Qwen checkpoints into an in-memory
`LlamaModel`) but cannot **write** it. This spec adds a GGUF *writer* so a model
that was imported — and optionally fine-tuned with `LlamaTrainer` — can be
serialized back to a GGUF file that runs unchanged in the wider ecosystem
(llama.cpp, ollama, LM Studio).

This closes the round-trip:

```
GGUF (llama/qwen)
   │ import
   ▼
LlamaModel ──finetune (LlamaTrainer)──▶ LlamaModel
   │ EXPORT (this spec)
   ▼
GGUF  ──runs in──▶ llama.cpp / ollama / LM Studio
```

### Scope

**In scope:** exporting an in-memory `LlamaModel` (architecture `llama` or
`qwen2`) to a GGUF v3 file, at any of the on-disk types
`F32, F16, Q8_0, Q8_1, Q4_0, Q4_1, Q4_K, Q5_K, Q6_K`, driven by a source GGUF
whose metadata (hyperparameters + tokenizer) is carried forward verbatim.

**Out of scope (deliberate):**

- Exporting Ferrum's *native* `GenerativeSLM` or tabular `Sequential` models.
  Their architectures (learned positional embeddings, LayerNorm, ReLU-FFN;
  tabular MLPs) are not recognized GGUF `general.architecture` values, so the
  resulting file would run in no external tool and would only duplicate what the
  FINF format already does. If a Ferrum-only interchange is ever wanted, it is a
  separate decision.
- The k-quant sub-types `Q2_K`/`Q3_K` and the `IQ*` formats (the reader does not
  decode them either).
- A per-tensor mixed-precision policy à la llama.cpp's `Q4_K_M` (noted as a
  future refinement below).
- `ferrum_gui` wiring (the crate is excluded from the workspace; a follow-up).

### Why round-trip only

The whole GGUF ecosystem only *executes* recognized architectures. Ferrum's
`LlamaModel` **is** such an architecture (RoPE, RMSNorm, SwiGLU, GQA), so its
GGUF is runnable everywhere. A GGUF of Ferrum's native transformer would not be.
Exporting `LlamaModel` is therefore the only variant that produces a useful file.

## The reader is the specification

Everything the writer emits must be byte-compatible with what
`ferrum_core::gguf` already parses. Two consequences shape the whole design:

1. **The header/metadata/tensor-directory layout is fixed by `parse_header`.**
   The writer is the exact inverse: magic `GGUF`, `version = 3`, `tensor_count`
   (u64), `kv_count` (u64), the typed KV table, then the tensor directory, then
   an aligned data section. `general.alignment` (default 32) governs padding.

2. **Each block encoder is the inverse of the matching `dequant_*` decoder** and
   shares its layout constants (`QK = 32`, `QK_K = 256`, `Q4_K_BLOCK = 144`,
   `Q5_K_BLOCK = 176`, `Q6_K_BLOCK = 210`, and the 6-bit scale packing embodied
   by `get_scale_min_k4`). Because the reader already decodes every target type,
   the encoders are **verified by round-tripping through the reader** within a
   per-format error bound. Writer and reader cannot silently drift.

## Weight orientation (critical detail)

The reader imports a 2-D weight as: GGUF declares `dims = [n_in, n_out]`; the raw
contiguous data is row-major `[n_out, n_in]`; `linear_from` transposes it to
Ferrum's `[n_in, n_out]`. The writer **inverts** this:

- Take Ferrum's weight `[n_in, n_out]` (from `Linear.weight`, or `qweight.to_f32()`
  when the model is quantized in memory).
- Transpose to raw `[n_out, n_in]` and quantize that buffer.
- Declare `dims = [n_in, n_out]` (GGML order, `dims[0]` fastest-varying).

`token_embd.weight` is a special case: Ferrum holds it row-major `[vocab, dim]`,
which is already the correct raw byte order; it is declared `dims = [dim, vocab]`
and written without transpose.

Because export is a **round-trip**, the Q/K RoPE permutation that llama.cpp's
converter applies is already baked into the imported weights, and is written back
as-is (`RopeType::Norm`). No re-permutation is performed.

## Components

All new code is in `ferrum_core`, in a new module `gguf_write.rs`, except a
visibility change in `gguf.rs`.

### 0. Shared constants (visibility change only)

The format constants the writer needs (`GGUF_MAGIC`, `DEFAULT_ALIGNMENT`, the
`GGML_*` type ids, the `VT_*` value-type tags, `QK`, `QK_K`, `Q4_K_BLOCK`,
`Q5_K_BLOCK`, `Q6_K_BLOCK`) are promoted from private to `pub(crate)` in
`gguf.rs`. No reader logic moves — this is a visibility change only, so reader
and writer share one source of truth.

### 1. `GgufQuant` (public enum)

The requested on-disk type.

```rust
pub enum GgufQuant { F32, F16, Q8_0, Q8_1, Q4_0, Q4_1, Q4K, Q5K, Q6K }

impl GgufQuant {
    pub fn from_str(s: &str) -> Option<Self>;   // "f32","f16","q8_0","q4_0",
                                                //  "q4_1","q8_1","q4_k","q5_k","q6_k"
    fn ggml_type(self) -> u32;                  // the GGML_* id
    fn file_type(self) -> u32;                  // general.file_type ftype id
}
```

### 2. `GgufBuilder` (low-level, format-only)

Accumulates metadata and tensors, then emits bytes. Knows nothing about Llama.

```rust
pub struct GgufBuilder { /* metadata: Vec<(String, MetaValue)>, tensors, alignment=32 */ }

impl GgufBuilder {
    pub fn new() -> Self;
    pub fn meta(&mut self, key: &str, val: MetaValue) -> &mut Self;
    pub fn tensor(&mut self, name: &str, dims: &[u64], ggml_type: u32, data: Vec<u8>) -> &mut Self;
    pub fn into_bytes(self) -> Vec<u8>;
    pub fn write(self, path: &str) -> Result<()>;
}
```

Emission (exact inverse of `parse_header`):

1. `GGUF` magic, `version = 3`, `tensor_count` (u64), `kv_count` (u64).
2. KV table in insertion order (keys are `general.*`, `<arch>.*`, `tokenizer.*`,
   …). Every `MetaValue` variant is serialized, including nested-free arrays.
3. Tensor directory: for each tensor `name`, `n_dims` (u32), `dims` (u64 each),
   `ggml_type` (u32), `offset` (u64). `offset` is relative to the data section
   and equals `align_up(previous_end, alignment)`.
4. Pad from the directory end to `align_up(pos, alignment)` — the data-section
   start — then write each tensor's bytes, padding to `alignment` between
   tensors so declared offsets are exact.

### 3. Block encoders + `f32_to_f16`

One `enc_*(&[f32]) -> Vec<u8>` per type, each the inverse of the reader's
`dequant_*`, plus `f32_to_f16(f32) -> u16` (round-to-nearest-even; inf/NaN,
subnormal, and overflow-to-inf handled) sitting next to the reader's
`f16_to_f32`.

- **F32/F16:** direct element encode.
- **Q8_0 / Q8_1 / Q4_0 / Q4_1:** per 32-element block, compute the block scale
  (and min/sum for the `_1` forms), quantize, pack. The chosen scale scheme need
  only satisfy the reader's decode formula; correctness is the round-trip bound,
  not bit-identity with llama.cpp's encoder.
- **Q4_K / Q5_K / Q6_K:** per 256-element super-block, derive the 8 sub-block
  scales (and mins for Q4_K/Q5_K), pack the 6-bit scales exactly as
  `get_scale_min_k4` unpacks them, and pack the 4-bit (+ high-bit) quants exactly
  as the decoder reads them.

### 4. `write_llama_gguf` (high-level mapping)

```rust
pub fn write_llama_gguf(model: &LlamaModel, source: &Gguf, quant: GgufQuant, path: &str) -> Result<()>;
pub fn llama_gguf_bytes(model: &LlamaModel, source: &Gguf, quant: GgufQuant) -> Result<Vec<u8>>;

impl LlamaModel {
    pub fn write_gguf(&self, source: &Gguf, quant: GgufQuant, path: &str) -> Result<()>; // delegates
}
```

Flow:

1. Read `general.architecture` from `source`; error unless `llama`/`qwen2`.
2. **Copy metadata verbatim** from `source`: all `general.*`, all `<arch>.*`
   (the hyperparameters), and the entire `tokenizer.ggml.*` block (the whole
   tokenizer — token list, scores, merges, types, bos/eos). Refresh only
   `general.file_type` and `general.quantization_version` to match `quant`.
3. Emit tensors in a stable order, each quantized per the policy below:
   - `token_embd.weight` — `dims=[dim, vocab]`, no transpose.
   - per block `i`: `blk.i.attn_norm.weight`, `blk.i.attn_q.weight`
     (+`.bias`†), `blk.i.attn_k.weight` (+`.bias`†), `blk.i.attn_v.weight`
     (+`.bias`†), `blk.i.attn_output.weight`, `blk.i.ffn_norm.weight`,
     `blk.i.ffn_gate.weight`, `blk.i.ffn_up.weight`, `blk.i.ffn_down.weight`.
   - `output_norm.weight`, `output.weight`‡.
4. Hand everything to `GgufBuilder` and write.

† **Biases** are emitted only if the source GGUF contained that bias tensor, so
Llama stays bias-free while Qwen2 keeps its q/k/v biases.

‡ **`output.weight`** is emitted iff the source had it; otherwise the
tied-embedding arrangement is preserved (no `output.weight`). *Assumption:* a
fine-tune does not untie a tied head. Documented as a known edge case.

## Quantization policy (`plan_tensor_type`)

- **1-D tensors** (all `*_norm.weight`, all biases) → **F32** always (tiny and
  precision-critical).
- **2-D matrices** → the requested `quant`, guarded by divisibility of `dims[0]`
  (the block axis, i.e. the fastest-varying dimension of the raw data):
  - k-quants require `dims[0] % 256 == 0`;
  - legacy quants require `dims[0] % 32 == 0`.
  - On violation → **fall back to F16** for that tensor and emit a warning. The
    file stays valid; only that tensor is larger.

This uniform policy is intentionally simple and predictable. A llama.cpp-style
mixed policy (bumping `attn_v`/`ffn_down` to a higher type for `_M` variants) is
a future refinement and is not implemented here.

## Error handling

- Source architecture not `llama`/`qwen2` → `InferError::Format`, matching the
  reader's own message style.
- A missing expected tensor → a clear, named `InferError::Format`.
- Divisibility fallback → `eprintln!` warning; export continues.
- I/O failures → wrapped in `InferError`.

## CLI — `slm_cli export-gguf`

```
export-gguf <in.gguf> <out.gguf> [--quant q8_0|q4_0|q4_1|q8_1|q4_k|q5_k|q6_k|f16|f32]
                                 [--resume tuned.flck] [--force]
```

Behavior:

1. Open `in.gguf` **streamed** (`Gguf::open`); print `GGUF vN  architecture = …`.
2. Apply the existing memory guard for an **f32** load
   (`estimate_resident_bytes(&g, None)`); `--force` overrides the 90%-of-RAM
   refusal, mirroring `run-gguf`/`finetune-gguf`.
3. Load the model at full precision (`load_llama_prec(None)`).
4. If `--resume tuned.flck` is given, apply the checkpoint's f32 masters onto the
   loaded model before export.
5. Call `write_llama_gguf(&model, &g, quant, out)`.
6. Print a summary: per-type tensor counts and the output file size.

Default `--quant` is `q8_0` (high quality, universally supported). `--quant` names
match the reader's existing conventions where they overlap.

*Note:* re-quantizing an already-quantized source (e.g. a `Q4_K` download to
`Q4_0`) goes through f32 and is inherently lossy — expected for a converter. The
lossless path is `f32`/`f16`, or exporting a fine-tuned model whose trainer held
f32 masters.

## Testing (TDD)

Written test-first, in this order:

1. **Encoder round-trips** — for each type, `enc_*` a known f32 vector, decode
   with the existing `dequant_*`, and assert max-abs error ≤ the format's bound
   (F32 exact; F16 within half-ULP; Q8_0/Q4_0/… within block quantization error;
   k-quants within their bound). Plus constant-block exactness tests mirroring
   the reader's existing `dequant_q6_k_constant_block` / `dequant_q4_k_constant_block`
   / `dequant_q5_k_low_and_high_bits` tests.
2. **Builder** — build a GGUF with scalar, string, and array metadata plus a
   couple of tensors; parse it back with `Gguf::parse`; assert metadata and the
   tensor directory match (generalizes the minimal writer already in gguf.rs's
   test module).
3. **Full model round-trip** — construct a tiny `LlamaModel` and a minimal source
   `Gguf` (architecture + a small tokenizer); export → reopen (`Gguf::open`) →
   `load_llama_prec(None)`; compare logits on a fixed token sequence (exact for
   `F32`; bounded for `Q8_0`/`Q4_0`/`Q4_K`).
4. **Tokenizer preservation** — export with a source carrying `tokenizer.ggml.*`;
   reopen; `GgufTokenizer::from_gguf` succeeds and encodes a sample string
   identically to the source tokenizer.
5. **CLI smoke** — in `tests/`, export a small fixture and re-open it end to end.

**External validation** against real llama.cpp is documented as a manual check,
never a CI dependency (Ferrum stays zero-dependency).

## Implementation details to confirm during planning

Neither changes the design:

- How to retrieve the underlying `LlamaModel` back out of `LlamaTrainer` after
  applying a `--resume` checkpoint (accessor vs. rebuild).
- The exact `general.file_type` ftype id to stamp per target type.

## Files touched

| File | Change |
|------|--------|
| `ferrum_core/src/gguf.rs` | Promote format constants to `pub(crate)`; add `f32_to_f16` next to `f16_to_f32` (or place in `gguf_write.rs` and share). |
| `ferrum_core/src/gguf_write.rs` | **New.** `GgufBuilder`, `GgufQuant`, block encoders, `write_llama_gguf`/`llama_gguf_bytes`, `plan_tensor_type`; unit tests 1–4. |
| `ferrum_core/src/llm.rs` | `impl LlamaModel { pub fn write_gguf(...) }` (thin delegate). |
| `ferrum_core/src/lib.rs` | Export `gguf_write` items (`GgufBuilder`, `GgufQuant`, `write_llama_gguf`). |
| `slm_cli/src/main.rs` | New `export-gguf` subcommand + help text. |
| `tests/tests/` | CLI/round-trip integration test (test 5). |
| `readme.md` / `docs/how_to_use.md` | Document the new `export-gguf` command and library API. |
