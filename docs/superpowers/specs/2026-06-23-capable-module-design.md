# `capable` — Machine Capability Estimator

Date: 2026-06-23
Status: Approved (design)

## Purpose

Give Ferrum SLM Studio users a grounded answer to "how big a model can *this*
machine realistically handle?" by micro-benchmarking the host and deriving
upper-bound parameter counts for three workloads:

- **Inference** — sustained decode at **≥ 3 tokens/sec**.
- **Training** — full training run completing in **< 24 hours**.
- **Testing / evaluation** — forward-only pass over a held-out corpus in **< 24 hours**.

Results are shown in an HTML `<dialog>` box. GGUF imports that exceed any bound
raise a modal warning with an override.

## Estimation model

The estimates are grounded by a live in-process micro-benchmark (≈1–2 s), not
static guesses, because decode is bandwidth-bound and training is compute-bound
and both vary widely across CPUs.

### Measurements

- `measure_mem_bandwidth() -> f64` (GB/s): stream a ~256 MB buffer with a
  reduction and time it. Anchors the decode (bandwidth-bound) model.
- `measure_gemm_gflops() -> f64` (GFLOP/s): time a moderate square GEMM,
  FLOPs = `2·n³`. Anchors the training/eval (compute-bound) model.

### Bound math (pure, unit-tested)

Let `bw` = measured bytes/sec, `gflops` = measured FLOP/s.

**Inference** (each decoded token reads all weights once):
```
N_infer(bpp) = (bw · η_decode) / (3 · bpp)
```
with bytes-per-param `bpp`: int4 = 0.5, int8 = 1, f32 = 4.

**Training** (`6·N·T` FLOPs; budget `B_train = gflops · 86400 · η_train`):
```
Chinchilla 20×:   T = 20·N   →  N = sqrt(B_train / 120)
Fixed 1B tokens:  T = 1e9    →  N = B_train / 6e9
```

**Testing / eval** (forward-only `2·N·T`; fixed held-out corpus `T_eval`,
default 10M tokens; budget `B_eval = gflops · 86400 · η_train`):
```
N_test = B_eval / (2 · T_eval)
```

### Calibration constants

`η_decode` and `η_train` are named constants in the module, calibrated against
`benchmarks.md` so the estimates track *achieved* throughput rather than raw
peak (e.g. ~1B params at int4 should land near the measured ~7 tok/s, not the
raw stream-bandwidth ideal). They are documented inline where defined.

## Architecture

### Backend — `ferrum_gui/src/capable.rs` (new)

- Pure helpers (no GUI, unit-tested): `infer_max_params`, `train_max_chinchilla`,
  `train_max_fixed`, `test_max_params`, with degenerate-input guards.
- Benchmark helpers: `measure_mem_bandwidth`, `measure_gemm_gflops`.
- `CapabilityReport` (serde, camelCase): CPU model, cores, threads, total/avail
  RAM, measured `memBwGbps` / `gemmGflops`, the six bounds, and the assumptions
  (constants, token budgets, the 3 tok/s and 24h figures) so the dialog can show
  its own workings.
- `#[tauri::command] async fn capability_report(state) -> Result<CapabilityReport, String>`
  runs the benchmark on a blocking task. Registered in `lib.rs`.

### Existing file touch — `ferrum_gui/src/commands.rs`

- Add a `paramCount` field to `GgufInfo` (`sum(num_elements)`), so the frontend
  can compare a GGUF against the cached bounds without another round-trip.

### Frontend — `ui/index.html`, `ui/app.js`, `ui/styles.css`

- New **"Capable"** tab (matches the existing tab/panel pattern) with a
  "Check this machine" button.
- `<dialog id="capDialog">` rendered via `showModal()`: machine summary, measured
  throughput, and a table of the six bounds with assumptions footnoted. This is
  the requested HTML dialog box.
- The returned bounds are cached in a module variable `capBounds`.
- **GGUF gate:** on Inspect and before Run, if `capBounds` is known, compare the
  GGUF `paramCount` against all three categories; if any are crossed, show a
  `<dialog id="ggWarnDialog">` listing every exceeded threshold with
  **Cancel / Proceed anyway**. Run continues only on confirm. If bounds are not
  yet measured, offer to check first. Mirrors the existing `force`/"load anyway"
  override UX.

## Testing

Rust unit tests on the pure bound functions:
- Monotonicity: more bandwidth → larger inference bound; more FLOP/s → larger
  train/test bound.
- Precision ordering: `N_infer(int4) > N_infer(int8) > N_infer(f32)`.
- Benchmark anchor sanity: ~1B int4 falls in a plausible range at calibrated η.
- Degenerate guards: zero/NaN bandwidth or FLOP/s does not panic or divide by
  zero.

## Scope guards (YAGNI)

- No persisted history, charts, or remote calls.
- No new dependencies — reuses `sysinfo` (already present) and pure Rust loops
  for the benchmark.
