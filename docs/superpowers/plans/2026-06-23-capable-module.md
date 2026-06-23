# Capable Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `capable` feature to Ferrum SLM Studio that micro-benchmarks the host machine and reports upper-bound parameter counts for inference (≥3 tok/s), training (<24h), and eval (<24h), shown in an HTML dialog, with a GGUF-import warning modal when a model exceeds any bound.

**Architecture:** A new Rust module `ferrum_gui/src/capable.rs` holds pure bound-math helpers, two micro-benchmark functions (memory bandwidth, GEMM throughput), and a `capability_report` Tauri command. The vanilla-JS frontend gains a "Capable" tab, an HTML `<dialog>` report, and a warning `<dialog>` that gates GGUF Inspect/Run against the cached bounds.

**Tech Stack:** Rust + Tauri 2 (backend), `sysinfo` (already a dependency), vanilla HTML/CSS/JS (frontend). No new dependencies.

## Global Constraints

- Every Tauri command returns `Result<T, String>` (human-readable errors surface in the GUI). — verbatim convention from `commands.rs` header.
- Heavy/blocking work runs inside `spawn_blocking` so the UI thread stays responsive.
- Serde structs crossing to JS use `#[serde(rename_all = "camelCase")]`.
- No new crate dependencies; reuse `sysinfo` and `ferrum_core`.
- Frontend is framework-free vanilla JS in `ui/`; follow the existing `$()`, `toast()`, `invoke()`, tab/panel, and `.card`/`.infocards` patterns.
- The "HTML dialog box" requirement is satisfied with native `<dialog>` elements driven by `showModal()`.

---

### Task 1: Pure bound-math helpers in `capable.rs`

**Files:**
- Create: `ferrum_gui/src/capable.rs`
- Test: same file (`#[cfg(test)] mod tests`)

**Interfaces:**
- Consumes: nothing.
- Produces (all `pub`):
  - Constants: `BPP_INT4: f64 = 0.5`, `BPP_INT8: f64 = 1.0`, `BPP_F32: f64 = 4.0`, `TARGET_TOKS: f64 = 3.0`, `TRAIN_SECS: f64`, `FIXED_TRAIN_TOKENS: f64 = 1e9`, `EVAL_TOKENS: f64 = 1e7`, `CHINCHILLA_RATIO: f64 = 20.0`, `DECODE_EFFICIENCY: f64 = 0.35`, `TRAIN_EFFICIENCY: f64 = 0.5`
  - `infer_max_params(bw_bytes_per_s: f64, bytes_per_param: f64) -> f64`
  - `train_max_chinchilla(gflops: f64) -> f64`
  - `train_max_fixed(gflops: f64) -> f64`
  - `test_max_params(gflops: f64) -> f64`
  - (`gflops` here is the **aggregate** GEMM throughput across all cores.)

- [ ] **Step 1: Write the failing tests**

Create `ferrum_gui/src/capable.rs` with only the test module and the (not yet written) function references:

```rust
//! Machine-capability estimator: micro-benchmarks the host and derives
//! upper-bound parameter counts for inference (>=3 tok/s), training (<24h),
//! and evaluation (<24h). See docs/superpowers/specs/2026-06-23-capable-module-design.md.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn infer_bound_orders_by_precision() {
        // Smaller bytes-per-param => more params fit the same bandwidth budget.
        let bw = 10e9; // 10 GB/s
        let i4 = infer_max_params(bw, BPP_INT4);
        let i8 = infer_max_params(bw, BPP_INT8);
        let f32 = infer_max_params(bw, BPP_F32);
        assert!(i4 > i8 && i8 > f32, "expected int4 > int8 > f32, got {i4} {i8} {f32}");
    }

    #[test]
    fn infer_bound_scales_with_bandwidth() {
        let lo = infer_max_params(5e9, BPP_INT8);
        let hi = infer_max_params(20e9, BPP_INT8);
        assert!(hi > lo * 3.0, "4x bandwidth should give ~4x params: {lo} {hi}");
    }

    #[test]
    fn infer_bound_anchor_is_plausible() {
        // benchmarks.md: ~1B params at int4 decodes ~7 tok/s. At 3 tok/s the
        // ceiling should sit above 1B and below ~10B for a typical ~8 GB/s CPU.
        let n = infer_max_params(8e9, BPP_INT4);
        assert!(n > 1e9 && n < 1e10, "anchor implausible: {n}");
    }

    #[test]
    fn train_bounds_scale_with_flops() {
        assert!(train_max_chinchilla(200.0) > train_max_chinchilla(50.0));
        assert!(train_max_fixed(200.0) > train_max_fixed(50.0));
        assert!(test_max_params(200.0) > test_max_params(50.0));
    }

    #[test]
    fn chinchilla_is_sqrt_shaped() {
        // 4x the FLOP budget should ~double the Chinchilla param bound (sqrt law).
        let a = train_max_chinchilla(50.0);
        let b = train_max_chinchilla(200.0);
        let ratio = b / a;
        assert!((ratio - 2.0).abs() < 0.05, "expected ~2x, got {ratio}");
    }

    #[test]
    fn degenerate_inputs_do_not_panic() {
        for &bad in &[0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(infer_max_params(bad, BPP_INT8), 0.0);
            assert_eq!(train_max_chinchilla(bad), 0.0);
            assert_eq!(train_max_fixed(bad), 0.0);
            assert_eq!(test_max_params(bad), 0.0);
        }
        assert_eq!(infer_max_params(10e9, 0.0), 0.0);
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail to compile**

Run: `cargo test -p ferrum_gui capable:: 2>&1 | tail -20`
Expected: FAIL — `cannot find function infer_max_params` / `cannot find value BPP_INT4`.

- [ ] **Step 3: Write the constants and pure functions**

Insert above the `#[cfg(test)]` module:

```rust
// ── Modeling constants ───────────────────────────────────────────────────────

/// Bytes per stored parameter at each precision (token-embedding nuance ignored
/// at the headline level — see the design doc).
pub const BPP_INT4: f64 = 0.5;
pub const BPP_INT8: f64 = 1.0;
pub const BPP_F32: f64 = 4.0;

/// Sustained decode rate (tokens/sec) the inference bound is solved for.
pub const TARGET_TOKS: f64 = 3.0;
/// Wall-clock budget for the training / eval bounds: 24 hours, in seconds.
pub const TRAIN_SECS: f64 = 24.0 * 3600.0;
/// Fixed-corpus training assumption (tokens).
pub const FIXED_TRAIN_TOKENS: f64 = 1e9;
/// Held-out eval corpus assumption (tokens).
pub const EVAL_TOKENS: f64 = 1e7;
/// Tokens-per-parameter for the compute-optimal (Chinchilla) training bound.
pub const CHINCHILLA_RATIO: f64 = 20.0;

/// Fraction of raw stream bandwidth that weight-streaming (m=1 GEMV) decode
/// actually achieves. Calibrated against benchmarks.md (~7 tok/s @ 1B int4):
/// decode reaches only a portion of peak stream bandwidth.
pub const DECODE_EFFICIENCY: f64 = 0.35;
/// Fraction of aggregate peak GEMM throughput a real training step sustains
/// (imperfect parallel scaling + non-GEMM work).
pub const TRAIN_EFFICIENCY: f64 = 0.5;

// ── Pure bound math ──────────────────────────────────────────────────────────

fn finite_pos(x: f64) -> bool {
    x.is_finite() && x > 0.0
}

/// Max parameters decodable at >= [`TARGET_TOKS`], bandwidth-bound: each decoded
/// token streams every weight once, so `tok/s = bw / (N * bytes_per_param)`.
pub fn infer_max_params(bw_bytes_per_s: f64, bytes_per_param: f64) -> f64 {
    if !finite_pos(bw_bytes_per_s) || !finite_pos(bytes_per_param) {
        return 0.0;
    }
    (bw_bytes_per_s * DECODE_EFFICIENCY) / (TARGET_TOKS * bytes_per_param)
}

/// Usable training/eval compute budget (FLOPs) within the 24h wall clock.
/// `gflops` is the aggregate GEMM throughput across all cores.
fn flop_budget(gflops: f64) -> f64 {
    if !finite_pos(gflops) {
        return 0.0;
    }
    gflops * 1e9 * TRAIN_SECS * TRAIN_EFFICIENCY
}

/// Max trainable params, compute-optimal: training FLOPs ~ `6*N*T` with
/// `T = 20*N`, so `N = sqrt(B / 120)`.
pub fn train_max_chinchilla(gflops: f64) -> f64 {
    let b = flop_budget(gflops);
    (b / (6.0 * CHINCHILLA_RATIO)).sqrt()
}

/// Max trainable params on a fixed [`FIXED_TRAIN_TOKENS`] corpus: `N = B / (6*T)`.
pub fn train_max_fixed(gflops: f64) -> f64 {
    flop_budget(gflops) / (6.0 * FIXED_TRAIN_TOKENS)
}

/// Max params evaluable (forward-only, `2*N*T`) over [`EVAL_TOKENS`] within 24h.
pub fn test_max_params(gflops: f64) -> f64 {
    flop_budget(gflops) / (2.0 * EVAL_TOKENS)
}
```

- [ ] **Step 4: Wire the module in so tests compile**

Modify `ferrum_gui/src/lib.rs` — add `mod capable;` next to the existing module declarations (after `mod datasets;`):

```rust
mod commands;
mod datasets;
mod capable;
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test -p ferrum_gui capable:: 2>&1 | tail -20`
Expected: PASS — all 6 tests green (`test result: ok. 6 passed`).

- [ ] **Step 6: Commit**

```bash
git add ferrum_gui/src/capable.rs ferrum_gui/src/lib.rs
git commit -m "feat(capable): pure bound-math helpers for capability estimates

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Micro-benchmark functions

**Files:**
- Modify: `ferrum_gui/src/capable.rs`
- Test: same file

**Interfaces:**
- Consumes: nothing.
- Produces (all `pub`):
  - `measure_mem_bandwidth() -> f64` — usable memory bandwidth in **bytes/sec**.
  - `measure_gemm_gflops() -> f64` — single-thread GEMM throughput in **GFLOP/s**.

- [ ] **Step 1: Write the failing tests**

Add inside the existing `#[cfg(test)] mod tests`:

```rust
    #[test]
    fn mem_bandwidth_is_positive_and_sane() {
        let bw = measure_mem_bandwidth();
        // Any real machine streams between ~0.5 GB/s and ~2 TB/s.
        assert!(bw > 5e8 && bw < 2e12, "implausible bandwidth: {bw} B/s");
    }

    #[test]
    fn gemm_throughput_is_positive_and_sane() {
        let g = measure_gemm_gflops();
        assert!(g > 0.1 && g < 5000.0, "implausible GFLOP/s: {g}");
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p ferrum_gui capable::tests::mem_bandwidth 2>&1 | tail -20`
Expected: FAIL — `cannot find function measure_mem_bandwidth`.

- [ ] **Step 3: Implement the benchmark functions**

Add to `capable.rs` above the test module (after the pure math):

```rust
// ── Live micro-benchmark ─────────────────────────────────────────────────────

use std::hint::black_box;
use std::time::Instant;

/// Stream a ~256 MB buffer with a reduction to estimate usable memory
/// bandwidth (bytes/sec). Bandwidth-bound CPU decode is governed by this.
pub fn measure_mem_bandwidth() -> f64 {
    const N: usize = 64 * 1024 * 1024; // 64M f32 = 256 MB
    const REPS: usize = 4;
    let buf = vec![1.0f32; N];

    // Warm pages/caches so the timed loop measures steady-state bandwidth.
    let mut warm = 0.0f32;
    for &x in buf.iter().step_by(4096) {
        warm += x;
    }
    black_box(warm);

    let start = Instant::now();
    let mut acc = 0.0f32;
    for _ in 0..REPS {
        let mut s = 0.0f32;
        for &x in &buf {
            s += x;
        }
        acc += s;
    }
    let secs = start.elapsed().as_secs_f64();
    black_box(acc);

    if secs > 0.0 {
        (N * 4 * REPS) as f64 / secs
    } else {
        0.0
    }
}

/// Time a single-threaded square matmul (cache-friendly i-k-j order) to
/// estimate sustained GEMM throughput in GFLOP/s (FLOPs = 2*n^3).
pub fn measure_gemm_gflops() -> f64 {
    const N: usize = 512;
    let a = vec![1.0f32; N * N];
    let b = vec![1.0f32; N * N];
    let mut c = vec![0.0f32; N * N];

    let start = Instant::now();
    for i in 0..N {
        for k in 0..N {
            let aik = a[i * N + k];
            let brow = &b[k * N..k * N + N];
            let crow = &mut c[i * N..i * N + N];
            for j in 0..N {
                crow[j] += aik * brow[j];
            }
        }
    }
    let secs = start.elapsed().as_secs_f64();
    black_box(c[0]);

    if secs > 0.0 {
        (2.0 * (N as f64).powi(3)) / 1e9 / secs
    } else {
        0.0
    }
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p ferrum_gui capable:: 2>&1 | tail -20`
Expected: PASS — 8 tests green.

- [ ] **Step 5: Commit**

```bash
git add ferrum_gui/src/capable.rs
git commit -m "feat(capable): memory-bandwidth and GEMM micro-benchmarks

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: `CapabilityReport` struct + `capability_report` command

**Files:**
- Modify: `ferrum_gui/src/capable.rs`
- Modify: `ferrum_gui/src/lib.rs` (register command)

**Interfaces:**
- Consumes: Task 1 + 2 functions/constants; `crate::AppState` (`sys: Mutex<System>`); `ferrum_core::num_threads()`.
- Produces:
  - `pub struct CapabilityReport` (serde camelCase) — fields below.
  - `#[tauri::command] pub async fn capability_report(state: State<'_, AppState>) -> Result<CapabilityReport, String>`
  - JS field names (camelCase): `cpu, cores, threads, memTotal, memAvail, memBwGbps, gemmGflops, inferInt4, inferInt8, inferF32, trainChinchilla, trainFixed1b, testEval, targetToks, trainHours, evalTokens, fixedTrainTokens, chinchillaRatio`.

- [ ] **Step 1: Write the failing test**

Add inside `#[cfg(test)] mod tests`:

```rust
    #[test]
    fn report_assembles_consistent_bounds() {
        // Bypass the command (needs a Tauri runtime) and check the assembly
        // helper used by it directly.
        let r = assemble_report(
            "Test CPU".into(), 8, 8, 16_000_000_000, 8_000_000_000,
            8e9,   // 8 GB/s
            10.0,  // 10 GFLOP/s single-thread
        );
        assert_eq!(r.cores, 8);
        assert!((r.gemm_gflops - 80.0).abs() < 1e-6, "aggregate = single*cores");
        assert!(r.infer_int4 > r.infer_int8 && r.infer_int8 > r.infer_f32);
        assert!(r.train_chinchilla > 0.0 && r.train_fixed1b > 0.0 && r.test_eval > 0.0);
        assert_eq!(r.target_toks, TARGET_TOKS);
    }
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p ferrum_gui capable::tests::report_assembles 2>&1 | tail -20`
Expected: FAIL — `cannot find function assemble_report`.

- [ ] **Step 3: Implement the struct, assembly helper, and command**

Add to `capable.rs` above the test module:

```rust
// ── Report assembly + Tauri command ──────────────────────────────────────────

use crate::AppState;
use serde::Serialize;
use tauri::State;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CapabilityReport {
    pub cpu: String,
    pub cores: usize,
    pub threads: usize,
    pub mem_total: u64,
    pub mem_avail: u64,
    /// Measured memory bandwidth (GB/s).
    pub mem_bw_gbps: f64,
    /// Aggregate GEMM throughput across all cores (GFLOP/s).
    pub gemm_gflops: f64,
    pub infer_int4: f64,
    pub infer_int8: f64,
    pub infer_f32: f64,
    pub train_chinchilla: f64,
    pub train_fixed1b: f64,
    pub test_eval: f64,
    // Assumptions echoed so the dialog can show its own workings.
    pub target_toks: f64,
    pub train_hours: f64,
    pub eval_tokens: f64,
    pub fixed_train_tokens: f64,
    pub chinchilla_ratio: f64,
}

/// Build a report from measured numbers (pure; no Tauri runtime needed).
/// `gemm_single` is single-thread GFLOP/s; aggregate = single * cores.
fn assemble_report(
    cpu: String,
    cores: usize,
    threads: usize,
    mem_total: u64,
    mem_avail: u64,
    bw_bytes_per_s: f64,
    gemm_single: f64,
) -> CapabilityReport {
    let gflops = gemm_single * cores as f64;
    CapabilityReport {
        cpu,
        cores,
        threads,
        mem_total,
        mem_avail,
        mem_bw_gbps: bw_bytes_per_s / 1e9,
        gemm_gflops: gflops,
        infer_int4: infer_max_params(bw_bytes_per_s, BPP_INT4),
        infer_int8: infer_max_params(bw_bytes_per_s, BPP_INT8),
        infer_f32: infer_max_params(bw_bytes_per_s, BPP_F32),
        train_chinchilla: train_max_chinchilla(gflops),
        train_fixed1b: train_max_fixed(gflops),
        test_eval: test_max_params(gflops),
        target_toks: TARGET_TOKS,
        train_hours: TRAIN_SECS / 3600.0,
        eval_tokens: EVAL_TOKENS,
        fixed_train_tokens: FIXED_TRAIN_TOKENS,
        chinchilla_ratio: CHINCHILLA_RATIO,
    }
}

/// Micro-benchmark the host and return capability bounds. Polled on demand by
/// the "Capable" tab.
#[tauri::command]
pub async fn capability_report(state: State<'_, AppState>) -> Result<CapabilityReport, String> {
    // Snapshot machine facts under the shared sysinfo lock, then release it.
    let (cpu, cores, mem_total, mem_avail) = {
        let mut sys = state.sys.lock().map_err(|e| format!("state lock poisoned: {e}"))?;
        sys.refresh_memory();
        let cpu = sys
            .cpus()
            .first()
            .map(|c| c.brand().trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "unknown CPU".to_string());
        (cpu, sys.cpus().len(), sys.total_memory(), sys.available_memory())
    };
    let threads = ferrum_core::num_threads();

    // Run the blocking benchmark off the async runtime thread.
    let (bw, gemm_single) =
        tauri::async_runtime::spawn_blocking(|| (measure_mem_bandwidth(), measure_gemm_gflops()))
            .await
            .map_err(|e| format!("benchmark task error: {e}"))?;

    Ok(assemble_report(cpu, cores, threads, mem_total, mem_avail, bw, gemm_single))
}
```

Note: `use sysinfo::CpuExt;` is **not** needed on sysinfo 0.33 (the `brand()`/`cpus()` methods are inherent). If the compiler reports a missing trait for `.brand()`, add `use sysinfo::CpuExt;` at the top of the `use` block in this section.

- [ ] **Step 4: Register the command**

Modify `ferrum_gui/src/lib.rs` — add to the `tauri::generate_handler!` list (after `commands::system_stats,`):

```rust
            commands::system_stats,
            capable::capability_report,
```

- [ ] **Step 5: Run tests + a release-mode compile check**

Run: `cargo test -p ferrum_gui capable:: 2>&1 | tail -20`
Expected: PASS — 9 tests green.

Run: `cargo build -p ferrum_gui 2>&1 | tail -15`
Expected: compiles (warnings about unused `assemble_report` should NOT appear — it's used by both the command and the test).

- [ ] **Step 6: Commit**

```bash
git add ferrum_gui/src/capable.rs ferrum_gui/src/lib.rs
git commit -m "feat(capable): capability_report Tauri command

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Expose GGUF parameter count

**Files:**
- Modify: `ferrum_gui/src/commands.rs` (`GgufInfo` struct ~line 550; `gguf_info` body ~line 602)

**Interfaces:**
- Consumes: `Gguf.tensors` (`Vec<TensorInfo>`), `TensorInfo::num_elements() -> usize`.
- Produces: new `GgufInfo` field `param_count: u64` → JS `paramCount`.

- [ ] **Step 1: Add the struct field**

Modify the `GgufInfo` struct in `commands.rs` — add after `pub num_tensors: usize,`:

```rust
    pub num_tensors: usize,
    /// Total parameters across all tensors (sum of element counts).
    pub param_count: u64,
```

- [ ] **Step 2: Populate it in `gguf_info`**

In `gguf_info`, just before the `Ok(GgufInfo {` return, add:

```rust
        let param_count: u64 = g.tensors.iter().map(|t| t.num_elements() as u64).sum();
```

Then add `param_count,` to the struct literal, right after `num_tensors: g.tensors.len(),`:

```rust
            num_tensors: g.tensors.len(),
            param_count,
```

- [ ] **Step 3: Verify it compiles**

Run: `cargo build -p ferrum_gui 2>&1 | tail -15`
Expected: compiles cleanly.

- [ ] **Step 4: Commit**

```bash
git add ferrum_gui/src/commands.rs
git commit -m "feat(gguf): expose total parameter count in GgufInfo

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: "Capable" tab + report dialog (frontend)

**Files:**
- Modify: `ferrum_gui/ui/index.html` (tab nav ~line 33; new panel after `panel-system`; dialogs before `</body>`)
- Modify: `ferrum_gui/ui/app.js` (new section)
- Modify: `ferrum_gui/ui/styles.css` (dialog styling)

**Interfaces:**
- Consumes: `invoke("capability_report")` → `CapabilityReport` (camelCase fields from Task 3).
- Produces: a module-scope `capBounds` object cached for Task 6:
  `{ inferInt4, inferInt8, inferF32, trainChinchilla, trainFixed1b, testEval }` (all numbers), or `null` until first check.
  Exposes `window.capBounds` getter is NOT needed — Task 6 lives in the same file.
- Produces JS helper `fmtParams(n)` returning a human string ("1.5 B", "740 M", "12 K").

- [ ] **Step 1: Add the tab button**

In `index.html`, add to `<nav id="tabs">` after the System tab button (line ~33):

```html
    <button class="tab" data-tab="system">System</button>
    <button class="tab" data-tab="capable">Capable</button>
```

- [ ] **Step 2: Add the panel**

In `index.html`, add immediately after the closing `</section>` of `panel-system` (before `</main>` at line ~353):

```html
    <!-- ── Capable ───────────────────────────────────────────────────────── -->
    <section class="panel" id="panel-capable">
      <h2>Capable — what can this machine run?</h2>
      <p class="hint">Runs a quick (~1–2 s) micro-benchmark of memory bandwidth and
        matmul throughput on <em>this</em> CPU, then estimates the largest model you can
        realistically <strong>infer</strong> (≥ 3 tok/s), <strong>train</strong> (&lt; 24 h),
        and <strong>test</strong> (&lt; 24 h). Estimates are upper bounds — real models
        vary with architecture and context length.</p>
      <div class="row">
        <button id="capCheck" class="primary">Check this machine</button>
        <span id="capStatus" class="muted"></span>
      </div>
      <div id="capCards" class="infocards"></div>
    </section>
```

- [ ] **Step 3: Add the dialogs**

In `index.html`, add just before `<div id="toasts"></div>` (line ~370):

```html
  <!-- ── Capability report dialog ─────────────────────────────────────────── -->
  <dialog id="capDialog" class="capdlg">
    <form method="dialog">
      <h2>Machine capability</h2>
      <div id="capDialogBody"></div>
      <menu class="dlg-actions">
        <button class="primary" value="ok">Close</button>
      </menu>
    </form>
  </dialog>

  <!-- ── GGUF over-budget warning dialog ──────────────────────────────────── -->
  <dialog id="ggWarnDialog" class="capdlg">
    <h2>⚠ Model exceeds this machine's estimated limits</h2>
    <div id="ggWarnBody"></div>
    <menu class="dlg-actions">
      <button id="ggWarnCancel" class="ghost">Cancel</button>
      <button id="ggWarnProceed" class="primary">Proceed anyway</button>
    </menu>
  </dialog>
```

- [ ] **Step 4: Add the CSS**

Append to `ui/styles.css`:

```css
/* Capability dialogs */
dialog.capdlg {
  border: 1px solid var(--line);
  border-radius: 10px;
  background: var(--bg2);
  color: var(--fg);
  max-width: 640px;
  padding: 18px 20px;
}
dialog.capdlg::backdrop { background: rgba(0, 0, 0, 0.55); }
dialog.capdlg h2 { margin-top: 0; }
dialog.capdlg .dlg-actions {
  display: flex; justify-content: flex-end; gap: 8px;
  margin: 16px 0 0; padding: 0;
}
dialog.capdlg .capfoot { color: var(--muted); font-size: 11px; margin-top: 12px; }
dialog.capdlg table.data td.over { color: var(--accent); font-weight: 600; }
```

- [ ] **Step 5: Add the JS (cache + dialog render)**

Append to `ui/app.js` (after the system-monitor section, before any final init):

```javascript
// ── Capable (machine capability) ─────────────────────────────────────────────
let capBounds = null; // {inferInt4, inferInt8, inferF32, trainChinchilla, trainFixed1b, testEval}

function fmtParams(n) {
  if (!Number.isFinite(n) || n <= 0) return "—";
  if (n >= 1e9) return (n / 1e9).toFixed(2) + " B";
  if (n >= 1e6) return (n / 1e6).toFixed(1) + " M";
  if (n >= 1e3) return (n / 1e3).toFixed(0) + " K";
  return n.toFixed(0);
}

function renderCapReport(r) {
  capBounds = {
    inferInt4: r.inferInt4, inferInt8: r.inferInt8, inferF32: r.inferF32,
    trainChinchilla: r.trainChinchilla, trainFixed1b: r.trainFixed1b, testEval: r.testEval,
  };
  // Summary cards on the panel.
  const card = (k, v) => `<div class="card"><div class="k">${k}</div><div class="v">${v}</div></div>`;
  $("capCards").innerHTML =
    card("CPU", r.cpu) +
    card("Cores / threads", `${r.cores} / ${r.threads}`) +
    card("Memory", `${fmtBytes(r.memAvail)} free / ${fmtBytes(r.memTotal)}`) +
    card("Mem bandwidth", r.memBwGbps.toFixed(1) + " GB/s") +
    card("GEMM throughput", r.gemmGflops.toFixed(1) + " GFLOP/s") +
    card("Infer @int8 ceiling", fmtParams(r.inferInt8));
  // Detailed table inside the dialog.
  const row = (label, val, note) =>
    `<tr><td>${label}</td><td>${fmtParams(val)}</td><td class="muted">${note}</td></tr>`;
  $("capDialogBody").innerHTML = `
    <p class="muted">${r.cpu} · ${r.cores} cores · ${r.memBwGbps.toFixed(1)} GB/s · ${r.gemmGflops.toFixed(1)} GFLOP/s</p>
    <table class="data">
      <thead><tr><th>Workload</th><th>Max params</th><th>Basis</th></tr></thead>
      <tbody>
        ${row("Inference · int4", r.inferInt4, `≥ ${r.targetToks} tok/s`)}
        ${row("Inference · int8", r.inferInt8, `≥ ${r.targetToks} tok/s`)}
        ${row("Inference · f32", r.inferF32, `≥ ${r.targetToks} tok/s`)}
        ${row("Train · compute-optimal", r.trainChinchilla, `${r.chinchillaRatio}× tokens, < ${r.trainHours} h`)}
        ${row("Train · fixed corpus", r.trainFixed1b, `${fmtParams(r.fixedTrainTokens)} tokens, < ${r.trainHours} h`)}
        ${row("Test · eval pass", r.testEval, `${fmtParams(r.evalTokens)} tokens, < ${r.trainHours} h`)}
      </tbody>
    </table>
    <p class="capfoot">Upper bounds from a live micro-benchmark. Decode is bandwidth-bound;
      training/eval are compute-bound (≈6·N·T train, 2·N·T eval). Real throughput depends on
      architecture, context length, and other load.</p>`;
}

$("capCheck").addEventListener("click", async () => {
  $("capStatus").textContent = "benchmarking…";
  $("capCheck").disabled = true;
  try {
    const r = await invoke("capability_report");
    renderCapReport(r);
    $("capStatus").textContent = "";
    $("capDialog").showModal();
    toast("Capability check complete", "ok");
  } catch (e) {
    $("capStatus").textContent = "error";
    toast("Capability check failed: " + e, "error");
  } finally { $("capCheck").disabled = false; }
});
```

- [ ] **Step 6: Manual smoke test**

Run: `cargo build -p ferrum_gui 2>&1 | tail -5` (Expected: compiles.)
Then launch and click: open the **Capable** tab → "Check this machine". Verify the `<dialog>` opens modally with the six bounds and closes on "Close". (If a full GUI launch isn't available in the environment, confirm the HTML/JS parse by loading `ui/index.html` in a browser — the tab switch and `fmtParams` should work; `invoke` will toast a runtime error, which is expected outside Tauri.)

- [ ] **Step 7: Commit**

```bash
git add ferrum_gui/ui/index.html ferrum_gui/ui/app.js ferrum_gui/ui/styles.css
git commit -m "feat(capable): Capable tab and HTML report dialog

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: GGUF over-budget warning gate

**Files:**
- Modify: `ferrum_gui/ui/app.js` (GGUF Inspect handler ~line 444; GGUF Run handler ~line 466)

**Interfaces:**
- Consumes: `capBounds` (Task 5), `GgufInfo.paramCount` (Task 4), the `#ggWarnDialog` / `#ggWarnBody` / `#ggWarnCancel` / `#ggWarnProceed` elements (Task 5), `fmtParams` (Task 5).
- Produces: `checkGgufBudget(paramCount) -> string[]` (list of exceeded-bound descriptions) and `confirmGgufWarning(crossed) -> Promise<boolean>`.

- [ ] **Step 1: Add the budget-check + confirm helpers**

In `app.js`, add just above the existing `$("ggInspect").addEventListener(...)` (line ~444):

```javascript
// Compare a GGUF's parameter count against the cached capability bounds.
// Returns a list of human-readable descriptions for every bound it exceeds.
function checkGgufBudget(paramCount) {
  if (!capBounds || !Number.isFinite(paramCount) || paramCount <= 0) return [];
  const checks = [
    ["inference at int4 (≥ 3 tok/s)", capBounds.inferInt4],
    ["inference at int8 (≥ 3 tok/s)", capBounds.inferInt8],
    ["inference at f32 (≥ 3 tok/s)", capBounds.inferF32],
    ["training, compute-optimal (< 24 h)", capBounds.trainChinchilla],
    ["training, fixed corpus (< 24 h)", capBounds.trainFixed1b],
    ["evaluation pass (< 24 h)", capBounds.testEval],
  ];
  return checks
    .filter(([, bound]) => Number.isFinite(bound) && bound > 0 && paramCount > bound)
    .map(([label, bound]) => `${label}: limit ${fmtParams(bound)}`);
}

// Show the warning dialog; resolves true if the user chooses Proceed.
function confirmGgufWarning(paramCount, crossed) {
  return new Promise((resolve) => {
    $("ggWarnBody").innerHTML =
      `<p>This model has <strong>${fmtParams(paramCount)}</strong> parameters, which exceeds
       the estimated limits of this machine:</p><ul>` +
      crossed.map((c) => `<li>${c}</li>`).join("") +
      `</ul><p class="muted">You can still proceed — expect slower than the targets above.</p>`;
    const dlg = $("ggWarnDialog");
    const onProceed = () => { cleanup(); resolve(true); };
    const onCancel = () => { cleanup(); resolve(false); };
    function cleanup() {
      $("ggWarnProceed").removeEventListener("click", onProceed);
      $("ggWarnCancel").removeEventListener("click", onCancel);
      dlg.close();
    }
    $("ggWarnProceed").addEventListener("click", onProceed);
    $("ggWarnCancel").addEventListener("click", onCancel);
    dlg.showModal();
  });
}
```

- [ ] **Step 2: Warn on Inspect**

In the `ggInspect` handler, after `$("ggStatus").textContent = "";` and the existing `toast(...)` call (end of the `try`), add a non-blocking advisory. Replace the existing toast line:

```javascript
    $("ggStatus").textContent = "";
    toast(i.runnable ? "GGUF inspected" : "Inspected — architecture not runnable", i.runnable ? "ok" : "error");
```

with:

```javascript
    $("ggStatus").textContent = "";
    toast(i.runnable ? "GGUF inspected" : "Inspected — architecture not runnable", i.runnable ? "ok" : "error");
    const crossedI = checkGgufBudget(i.paramCount);
    if (crossedI.length) {
      await confirmGgufWarning(i.paramCount, crossedI); // advisory on inspect; result ignored
    } else if (!capBounds) {
      toast("Tip: run the Capable check to flag oversized models.", "info");
    }
```

- [ ] **Step 3: Gate the Run**

In the `ggRun` handler, the run currently begins right after params are built. Add a budget gate before `$("ggOut").textContent = "";`. The param count comes from a quick `gguf_info` lookup (cheap header parse). Replace:

```javascript
  } catch (e) { setErr("errGguf", String(e)); return; }

  $("ggOut").textContent = "";
```

with:

```javascript
  } catch (e) { setErr("errGguf", String(e)); return; }

  // Budget gate: if the model exceeds this machine's estimated limits, confirm.
  if (capBounds) {
    try {
      const info = await invoke("gguf_info", { path: params.modelPath });
      const crossed = checkGgufBudget(info.paramCount);
      if (crossed.length) {
        const ok = await confirmGgufWarning(info.paramCount, crossed);
        if (!ok) { $("ggStatus").textContent = "cancelled"; return; }
      }
    } catch (_) { /* inspection failed; fall through to run, which will error clearly */ }
  }

  $("ggOut").textContent = "";
```

- [ ] **Step 4: Manual smoke test**

Run: `cargo build -p ferrum_gui 2>&1 | tail -5` (Expected: compiles.)
Then: open **Capable** → Check. Open **GGUF**, inspect a model larger than the reported int8 ceiling → the warning `<dialog>` lists every exceeded bound. Click **Run** on that model → the same dialog appears; **Cancel** aborts (status "cancelled"), **Proceed anyway** continues to generation. Inspect a small model → no dialog.

- [ ] **Step 5: Commit**

```bash
git add ferrum_gui/ui/app.js
git commit -m "feat(capable): warn when GGUF imports exceed machine limits

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- Module named `capable` checking architecture → Task 1–3 (`capable.rs` + `capability_report`). ✓
- Highest params at 3 tok/s → `infer_max_params` / `inferInt4|8|f32`. ✓
- Training < 24h → `train_max_chinchilla` + `train_max_fixed`. ✓
- Upper bounds for train/test/infer (param counts) → all six bounds in the dialog table. ✓ (test = `test_max_params`.)
- HTML dialog box → `<dialog id="capDialog">` + `showModal()`. ✓
- Feature in ferrum-gui → Capable tab. ✓
- Modal warning on GGUF import when count exceeded → Task 6, `ggWarnDialog`, warn-with-override, lists all crossed bounds. ✓

**Placeholder scan:** No TBD/TODO; all code blocks complete; no "handle errors appropriately" hand-waves. ✓

**Type consistency:** Rust `assemble_report` field names match `CapabilityReport`; camelCase JS names (`inferInt8`, `paramCount`, `memBwGbps`, etc.) match the serde `rename_all = "camelCase"` structs. `capBounds`, `fmtParams`, `checkGgufBudget`, `confirmGgufWarning` referenced consistently across Tasks 5–6. ✓
