# Capable v2 — Parameter Ranges for Load / Train / Fine-tune / Inference

Date: 2026-07-02
Status: Approved (design)

## Purpose

The Capable tab currently reports six single upper bounds (inference at three
precisions, two training scenarios, one eval bound). This revision reorganizes
it around the four questions a user actually asks about an SLM on this machine
— **can it Load, Train, Fine-tune, and Run Inference** — and answers each with
a **parameter range** across that capability's natural axis:

| Capability | Range axis | Basis |
|---|---|---|
| **Load** | f32 → int4 | weights fit in usable free RAM |
| **Train** (scratch) | Chinchilla → fixed-corpus | 6·N·T FLOPs < 24 h, RAM-capped @ 16 B/param |
| **Fine-tune** | 10M-token → 1M-token corpus | 6·N·T FLOPs < 24 h, RAM-capped @ 16 B/param |
| **Inference** (≥ 3 tok/s) | f32 → int4 | bandwidth-bound decode, **RAM-capped** |

Eval (< 24 h forward pass) remains as a fifth row — it exists, the GGUF budget
dialog uses it, but it is not part of the headline four.

### Bug fix folded in

Today the three inference bounds ignore RAM entirely (`infer_*` is never min'd
with a memory bound, unlike train/eval). A high-bandwidth machine with little
free RAM is told it can decode a model that cannot fit. This revision caps each
inference bound by the load bound at the same precision.

### Scope

**In scope:** `ferrum_gui/src/capable.rs` (constants, pure fns, report,
tests), the Capable tab render + budget checks in `ferrum_gui/ui/app.js`, the
panel hint in `ui/index.html`, docs (`manual/05`, `ferrum_gui/README.md`).

**Out of scope:** new benchmarks (bandwidth + GEMM measurements are unchanged),
`ferrum_core`, the CLI, any other tab.

### Constraints

- `ferrum_gui` is excluded from the workspace — cargo commands run from
  `ferrum_gui/`.
- **Back-compat:** `CapabilityReport` keeps all six existing bound fields
  (`infer_int4/int8/f32`, `train_chinchilla`, `train_fixed1b`, `test_eval`) and
  all existing echo fields; `checkGgufBudget`'s six existing checks keep
  working unmodified (inference values change only by becoming correctly
  RAM-capped — a strict improvement).
- All verification is AI-runnable (cargo/node gates + boot smoke), no human step.

## Backend (`ferrum_gui/src/capable.rs`)

### New constants

```rust
/// Usable share of available RAM for holding model weights (matches the 90%
/// convention every RAM guard in this app uses).
pub const LOAD_FRACTION: f64 = 0.9;
/// Fine-tune corpus range: a large fine-tune (range low end) …
pub const FINETUNE_TOKENS_LO: f64 = 1e7;
/// … and a small one (range high end).
pub const FINETUNE_TOKENS_HI: f64 = 1e6;
```

### Pure functions (all unit-tested)

```rust
/// Max params whose weights fit in usable free RAM at `bytes_per_param`.
/// 0.0 when memory is unknown (rendered as "—"), NOT infinity — this is a
/// display bound, unlike `mem_bound_params` which is a cap.
pub fn load_max_params(mem_avail: u64, bytes_per_param: f64) -> f64;

/// Max params trainable on a `tokens`-token corpus within the 24 h budget
/// (compute only): `flop_budget(gflops) / (6 * tokens)`.
pub fn train_max_on_corpus(gflops: f64, tokens: f64) -> f64;
```

`train_max_fixed(gflops)` becomes `train_max_on_corpus(gflops, FIXED_TRAIN_TOKENS)`
(behavior identical). Fine-tune bounds are
`train_max_on_corpus(gflops, FINETUNE_TOKENS_LO/HI)`, each min'd with
`mem_bound_params(mem_avail, TRAIN_BYTES_PER_PARAM)` — fine-tuning holds the
same four f32 tensors per weight as training (weights + gradient + Adam m/v ≈
16 B/param; the same fact the export RAM guard encodes).

### Inference RAM cap (the fix)

Two sentinel conventions coexist deliberately: `load_max_params` returns
**0.0** when memory is unknown (it is a *display* bound → "—"), while capping
needs **∞** so an unknown leaves the other bound untouched (the existing
`mem_bound_params` convention). So the cap is its own tiny helper:

```rust
/// Load ceiling as a CAP: `LOAD_FRACTION * mem_avail / bpp`, or +∞ when
/// memory is unknown (so `min` leaves the bandwidth bound untouched).
fn load_cap_params(mem_avail: u64, bytes_per_param: f64) -> f64;
```

In `assemble_report`, each inference bound becomes:

```rust
infer_int4: infer_max_params(bw, BPP_INT4).min(load_cap_params(mem_avail, BPP_INT4)),
// …int8, f32 likewise
```

### `CapabilityReport` additions (serde camelCase, flat fields)

```rust
pub load_int4: f64,
pub load_int8: f64,
pub load_f32: f64,
pub finetune_lo: f64,          // 10M-token corpus, RAM-capped
pub finetune_hi: f64,          // 1M-token corpus, RAM-capped
// Assumption echoes for the dialog's Basis column:
pub finetune_tokens_lo: f64,
pub finetune_tokens_hi: f64,
pub load_fraction: f64,
```

All existing fields stay.

### Tests (added to the existing module tests)

- `load_bound_orders_by_precision` — int4 > int8 > f32 for the same RAM;
  `load_max_params(0, _) == 0.0`.
- `load_uses_the_90_percent_fraction` — exact arithmetic check.
- `finetune_range_is_ordered_and_scales` — `finetune_hi ≥ finetune_lo`;
  smaller corpus ⇒ larger bound; scales with gflops.
- `finetune_is_ram_capped_when_ram_is_scarce` — mirrors the existing training
  RAM-cap test.
- `inference_is_ram_capped_when_ram_is_scarce` — huge bandwidth + tiny RAM ⇒
  `infer_f32` equals the f32 load ceiling, not the bandwidth figure (this is
  the regression test for the bug fix).
- `report_keeps_backcompat_fields_and_range_order` — the six legacy fields
  present and consistent; `load_* ≥ infer_*` per precision (a model you can
  decode always fits); `finetune_hi ≥ finetune_lo`.
- `train_max_fixed_unchanged_by_refactor` — equals
  `train_max_on_corpus(g, FIXED_TRAIN_TOKENS)`.

## Frontend

### `ui/app.js`

- `capBounds` gains `loadInt4` and `finetuneHi` (the two used by budget checks).
- `checkGgufBudget` gains a **first, hard** check ahead of the existing six:
  `["loading at int4 (fits in RAM)", capBounds.loadInt4]` — a model that
  cannot even load is a stronger warning than a slow one. Existing six checks
  unchanged.
- `renderCapReport` dialog table becomes the approved four-range layout plus
  the eval row. Range cell format: `fmtParams(lo) + " – " + fmtParams(hi)`
  (single value when lo and hi coincide or a side is "—"):

```
Capability             Range (params)     Basis
Load                   f32 → int4         fits in 90% of free RAM
Train (scratch)        Chinchilla → fixed 6·N·T, < 24 h, RAM-capped @16 B/param
Fine-tune              10M → 1M tokens    6·N·T, < 24 h, RAM-capped @16 B/param
Inference ≥ 3 tok/s    f32 → int4         bandwidth-bound, RAM-capped
Eval pass              single bound       2·N·T over 10M tokens, < 24 h
```

  For Load and Inference the low end is the f32 bound and the high end int4
  (int8 shown in the Basis note); for Train, low = min(chinchilla, fixed) and
  high = max of the two, with the scenario names in the Basis note; for
  Fine-tune, low = `finetuneLo`, high = `finetuneHi`.
- Summary cards: replace the single "Infer @int8 ceiling" card with two cards —
  "Load @int4 ceiling" and "Infer @int8 ceiling" (both `fmtParams`).

### `ui/index.html`

Panel hint updated to name the four capabilities: "…estimates the parameter
range of an SLM this machine can **load**, **train** (< 24 h), **fine-tune**
(< 24 h), and **run** (≥ 3 tok/s), plus a 24 h eval bound."

## Docs

- `manual/05-using-the-gui.md`: the Capable tab description/walkthrough
  reframed around the four ranges (+ eval), keeping the existing voice.
- `ferrum_gui/README.md`: the Capable feature row mentions the four ranges.

## Error handling

Unchanged: `capability_report` already returns `Result<_, String>`; the new
math is total (0.0/∞ sentinels for unknown memory, consistent with the
existing `mem_bound_params` convention).

## Testing — fully AI-verifiable

1. `cd ferrum_gui && cargo test` — all existing + new unit tests.
2. `cd ferrum_gui && cargo fmt -- --check && cargo clippy --all-targets -- -D warnings`.
3. `node --check ui/app.js`; `node --test ui/`.
4. Field cross-check: every `r.<field>`/`capBounds.<field>` the JS reads exists
   on `CapabilityReport` (grep cross-reference).
5. Windowed boot smoke (snap-scrubbed env, 8 s alive, no panics) — proves the
   tab still renders.

## Files touched

| File | Change |
|------|--------|
| `ferrum_gui/src/capable.rs` | constants, `load_max_params`, `train_max_on_corpus` refactor, finetune bounds, inference RAM cap, report fields, tests |
| `ferrum_gui/ui/app.js` | `capBounds` additions, hard load check, four-range dialog, summary cards |
| `ferrum_gui/ui/index.html` | Capable panel hint |
| `manual/05-using-the-gui.md`, `ferrum_gui/README.md` | docs |
