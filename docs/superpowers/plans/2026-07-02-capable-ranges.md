# Capable v2 — Parameter Ranges Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reframe the Capable tab around four capabilities — Load / Train / Fine-tune / Inference — each reporting a parameter **range** across its natural axis, and fix the missing RAM cap on the inference bounds.

**Architecture:** Additive changes to `ferrum_gui/src/capable.rs` (two new pure helpers, three constants, eight new flat report fields, RAM-capped inference; all six legacy fields kept for back-compat) plus a reshaped dialog table and a new hard "fits in RAM" budget check in `ui/app.js`. No new benchmarks; no `ferrum_core` changes.

**Tech Stack:** Rust (Tauri 2), vanilla JS. No new dependencies.

## Global Constraints

- `ferrum_gui` is **excluded from the workspace**: every cargo command runs from the `ferrum_gui/` directory (`cd ferrum_gui && cargo …`), never `-p ferrum_gui` at the root.
- **Back-compat:** `CapabilityReport` keeps `infer_int4/int8/f32`, `train_chinchilla`, `train_fixed1b`, `test_eval` and all existing echo fields; `checkGgufBudget`'s six existing checks stay unmodified.
- Sentinel conventions: `load_max_params` returns **0.0** when memory is unknown (display bound → "—"); `load_cap_params`/`mem_bound_params` return **+∞** when unknown (caps must not bind).
- Constants (exact values): `LOAD_FRACTION = 0.9`, `FINETUNE_TOKENS_LO = 1e7`, `FINETUNE_TOKENS_HI = 1e6`.
- Keep output pristine: `cargo fmt -- --check` and `cargo clippy --all-targets -- -D warnings` clean (from `ferrum_gui/`); `node --check` clean.
- All verification is AI-runnable; the windowed boot smoke uses a snap-scrubbed environment (`env -u LD_LIBRARY_PATH -u GTK_PATH -u GIO_MODULE_DIR -u GTK_EXE_PREFIX -u GDK_PIXBUF_MODULE_FILE -u LOCPATH`).

---

### Task 1: Backend — constants, helpers, RAM-capped inference, report fields, tests

**Files:**
- Modify: `ferrum_gui/src/capable.rs` (module doc ~line 1-3; constants block ~line 41; pure-math block ~line 60-94; `CapabilityReport` ~line 100-122; `assemble_report` ~line 130-162; tests module)

**Interfaces:**
- Consumes: existing `finite_pos`, `flop_budget`, `mem_bound_params`, `infer_max_params`, `assemble_report`, constants (`BPP_*`, `TRAIN_BYTES_PER_PARAM`, `FIXED_TRAIN_TOKENS`).
- Produces: `pub const LOAD_FRACTION/FINETUNE_TOKENS_LO/FINETUNE_TOKENS_HI`, `pub fn load_max_params(u64, f64) -> f64`, `fn load_cap_params(u64, f64) -> f64`, `pub fn train_max_on_corpus(f64, f64) -> f64`, and report fields `load_int4/load_int8/load_f32/finetune_lo/finetune_hi/finetune_tokens_lo/finetune_tokens_hi/load_fraction` (serde camelCase: `loadInt4`, …, `loadFraction`).

- [ ] **Step 1: Write the failing tests**

Append inside `mod tests` in `ferrum_gui/src/capable.rs`:

```rust
    #[test]
    fn load_bound_orders_by_precision() {
        let mem = 8_000_000_000u64;
        let i4 = load_max_params(mem, BPP_INT4);
        let i8 = load_max_params(mem, BPP_INT8);
        let f = load_max_params(mem, BPP_F32);
        assert!(i4 > i8 && i8 > f, "expected int4 > int8 > f32: {i4} {i8} {f}");
        assert_eq!(load_max_params(0, BPP_INT8), 0.0); // unknown memory → display "—"
    }

    #[test]
    fn load_uses_the_90_percent_fraction() {
        // 10 GB free at int8 (1 B/param) → exactly 9e9 params.
        assert!((load_max_params(10_000_000_000, BPP_INT8) - 9e9).abs() < 1.0);
    }

    #[test]
    fn finetune_range_is_ordered_and_scales() {
        let lo = train_max_on_corpus(100.0, FINETUNE_TOKENS_LO);
        let hi = train_max_on_corpus(100.0, FINETUNE_TOKENS_HI);
        assert!(hi > lo, "smaller corpus must allow more params: {lo} {hi}");
        assert!(train_max_on_corpus(200.0, FINETUNE_TOKENS_LO) > lo);
        assert_eq!(train_max_on_corpus(100.0, 0.0), 0.0); // degenerate corpus
    }

    #[test]
    fn train_max_fixed_unchanged_by_refactor() {
        for &g in &[10.0, 50.0, 400.0] {
            assert_eq!(train_max_fixed(g), train_max_on_corpus(g, FIXED_TRAIN_TOKENS));
        }
    }

    #[test]
    fn finetune_is_ram_capped_when_ram_is_scarce() {
        // Huge compute, 1 GB free: both range ends collapse to the 16 B/param cap.
        let r = assemble_report(
            "cpu".into(),
            64,
            64,
            2_000_000_000,
            1_000_000_000,
            50e9,
            200.0,
        );
        let cap = 1_000_000_000.0 / TRAIN_BYTES_PER_PARAM;
        assert!((r.finetune_hi - cap).abs() < 1.0, "hi not capped: {}", r.finetune_hi);
        assert!((r.finetune_lo - cap).abs() < 1.0, "lo not capped: {}", r.finetune_lo);
    }

    #[test]
    fn inference_is_ram_capped_when_ram_is_scarce() {
        // Regression test for the fix: huge bandwidth (100 GB/s), 1 GB free —
        // f32 inference must equal the f32 LOAD ceiling, not the bandwidth figure.
        let r = assemble_report(
            "cpu".into(),
            8,
            8,
            2_000_000_000,
            1_000_000_000,
            100e9,
            10.0,
        );
        let cap = LOAD_FRACTION * 1_000_000_000.0 / BPP_F32;
        assert!((r.infer_f32 - cap).abs() < 1.0, "not RAM-capped: {}", r.infer_f32);
        // Premise check: bandwidth alone would have allowed more.
        assert!(infer_max_params(100e9, BPP_F32) > cap);
    }

    #[test]
    fn report_keeps_backcompat_fields_and_range_order() {
        let r = assemble_report(
            "cpu".into(),
            8,
            4,
            16_000_000_000,
            8_000_000_000,
            8e9,
            10.0,
        );
        // Legacy fields alive and still ordered.
        assert!(r.infer_int4 > r.infer_int8 && r.infer_int8 > r.infer_f32);
        assert!(r.train_chinchilla > 0.0 && r.train_fixed1b > 0.0 && r.test_eval > 0.0);
        // A model you can decode always fits: load ≥ infer per precision.
        assert!(r.load_int4 >= r.infer_int4);
        assert!(r.load_int8 >= r.infer_int8);
        assert!(r.load_f32 >= r.infer_f32);
        // Fine-tune range ordered; echoes present.
        assert!(r.finetune_hi >= r.finetune_lo);
        assert_eq!(r.load_fraction, LOAD_FRACTION);
        assert_eq!(r.finetune_tokens_lo, FINETUNE_TOKENS_LO);
        assert_eq!(r.finetune_tokens_hi, FINETUNE_TOKENS_HI);
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd ferrum_gui && cargo test capable 2>&1 | tail -4`
Expected: FAIL to compile — `LOAD_FRACTION`, `load_max_params`, `train_max_on_corpus`, `finetune_lo` etc. not defined.

- [ ] **Step 3: Implement**

3a. Update the module doc (lines 1-3) to:

```rust
//! Machine-capability estimator: micro-benchmarks the host and derives
//! parameter ranges for four capabilities — load (fits in RAM), train (<24h),
//! fine-tune (<24h), and inference (>=3 tok/s) — plus a 24h eval bound.
//! See docs/superpowers/specs/2026-06-23-capable-module-design.md and
//! docs/superpowers/specs/2026-07-02-capable-ranges-design.md.
```

3b. Add constants after `TRAIN_BYTES_PER_PARAM`/`EVAL_BYTES_PER_PARAM` (~line 43):

```rust
/// Usable share of available RAM for holding model weights (matches the 90%
/// convention every RAM guard in this app uses).
pub const LOAD_FRACTION: f64 = 0.9;
/// Fine-tune corpus range: a large fine-tune (the range's low end)…
pub const FINETUNE_TOKENS_LO: f64 = 1e7;
/// …and a small one (the range's high end).
pub const FINETUNE_TOKENS_HI: f64 = 1e6;
```

3c. Add the helpers after `mem_bound_params` (~line 59):

```rust
/// Max params whose weights fit in usable free RAM at `bytes_per_param`.
/// Returns 0.0 when memory is unknown — this is a *display* bound (shown as
/// "—"), unlike the ∞-sentinel cap conventions used for `min`-ing.
pub fn load_max_params(mem_avail: u64, bytes_per_param: f64) -> f64 {
    if mem_avail == 0 || !finite_pos(bytes_per_param) {
        return 0.0;
    }
    LOAD_FRACTION * mem_avail as f64 / bytes_per_param
}

/// Load ceiling as a CAP: like [`load_max_params`] but +∞ when memory is
/// unknown, so a `min` leaves the other bound untouched.
fn load_cap_params(mem_avail: u64, bytes_per_param: f64) -> f64 {
    if mem_avail == 0 || !finite_pos(bytes_per_param) {
        return f64::INFINITY;
    }
    LOAD_FRACTION * mem_avail as f64 / bytes_per_param
}
```

3d. Add `train_max_on_corpus` and refactor `train_max_fixed` (replacing its body; keep the doc comment style):

```rust
/// Max params trainable on a `tokens`-token corpus within the wall-clock
/// budget (compute only): `N = B / (6·T)`.
pub fn train_max_on_corpus(gflops: f64, tokens: f64) -> f64 {
    if !finite_pos(tokens) {
        return 0.0;
    }
    flop_budget(gflops) / (6.0 * tokens)
}

/// Max trainable params on a fixed [`FIXED_TRAIN_TOKENS`] corpus.
pub fn train_max_fixed(gflops: f64) -> f64 {
    train_max_on_corpus(gflops, FIXED_TRAIN_TOKENS)
}
```

3e. Extend `CapabilityReport` (after `test_eval`):

```rust
    pub load_int4: f64,
    pub load_int8: f64,
    pub load_f32: f64,
    /// Fine-tune range, RAM-capped: `lo` = [`FINETUNE_TOKENS_LO`]-token corpus,
    /// `hi` = [`FINETUNE_TOKENS_HI`]-token corpus.
    pub finetune_lo: f64,
    pub finetune_hi: f64,
```

and after `chinchilla_ratio` (assumption echoes):

```rust
    pub finetune_tokens_lo: f64,
    pub finetune_tokens_hi: f64,
    pub load_fraction: f64,
```

3f. In `assemble_report`, cap the inference bounds and fill the new fields (`train_mem` already exists in scope):

```rust
        infer_int4: infer_max_params(bw_bytes_per_s, BPP_INT4)
            .min(load_cap_params(mem_avail, BPP_INT4)),
        infer_int8: infer_max_params(bw_bytes_per_s, BPP_INT8)
            .min(load_cap_params(mem_avail, BPP_INT8)),
        infer_f32: infer_max_params(bw_bytes_per_s, BPP_F32)
            .min(load_cap_params(mem_avail, BPP_F32)),
        // …existing train/eval fields unchanged…
        load_int4: load_max_params(mem_avail, BPP_INT4),
        load_int8: load_max_params(mem_avail, BPP_INT8),
        load_f32: load_max_params(mem_avail, BPP_F32),
        finetune_lo: train_max_on_corpus(gflops, FINETUNE_TOKENS_LO).min(train_mem),
        finetune_hi: train_max_on_corpus(gflops, FINETUNE_TOKENS_HI).min(train_mem),
        finetune_tokens_lo: FINETUNE_TOKENS_LO,
        finetune_tokens_hi: FINETUNE_TOKENS_HI,
        load_fraction: LOAD_FRACTION,
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd ferrum_gui && cargo test capable 2>&1 | tail -3`
Expected: PASS — all existing capable tests plus the 7 new ones. Then the whole crate: `cargo test 2>&1 | grep "test result" | head -1` (expect ~37 passed, 0 failed).

- [ ] **Step 5: fmt + clippy gates**

Run: `cd ferrum_gui && cargo fmt && cargo fmt -- --check && cargo clippy --all-targets -- -D warnings 2>&1 | tail -2`
Expected: clean. (`load_max_params`/new fields are consumed by the report, so no dead-code accommodation should be needed.)

- [ ] **Step 6: Commit**

```bash
git add ferrum_gui/src/capable.rs
git commit -m "feat(gui): capable v2 backend — load/fine-tune bounds, RAM-capped inference"
```

---

### Task 2: Frontend — four-range dialog, hard load check, panel hint

**Files:**
- Modify: `ferrum_gui/ui/app.js` (`capBounds` decl ~line 759 and assignment inside `renderCapReport` ~line 770; `checkGgufBudget` ~line 448; the dialog/table portion of `renderCapReport` ~line 774-802; summary cards ~line 776-782)
- Modify: `ferrum_gui/ui/index.html` (Capable panel hint, the `<p class="hint">` inside `panel-capable`)

**Interfaces:**
- Consumes: report fields from Task 1 via camelCase (`r.loadInt4`, `r.loadInt8`, `r.loadF32`, `r.finetuneLo`, `r.finetuneHi`, `r.finetuneTokensLo`, `r.finetuneTokensHi`, `r.loadFraction`) plus all existing fields; existing helpers `fmtParams`, `fmtBytes`, `card`.
- Produces: `capBounds.loadInt4` (used by the new budget check) and `capBounds.finetuneHi` (available to future consumers).

- [ ] **Step 1: Extend `capBounds`**

Replace the declaration comment and the assignment in `renderCapReport`:

```javascript
let capBounds = null; // {inferInt4/8/F32, trainChinchilla, trainFixed1b, testEval, loadInt4, finetuneHi}
```

```javascript
  capBounds = {
    inferInt4: r.inferInt4, inferInt8: r.inferInt8, inferF32: r.inferF32,
    trainChinchilla: r.trainChinchilla, trainFixed1b: r.trainFixed1b, testEval: r.testEval,
    loadInt4: r.loadInt4, finetuneHi: r.finetuneHi,
  };
```

- [ ] **Step 2: Add the hard load check to `checkGgufBudget`**

Insert as the FIRST entry of the `checks` array (before the int4-inference line):

```javascript
    ["loading at int4 (fits in RAM)", capBounds.loadInt4],
```

- [ ] **Step 3: Reshape the dialog table and cards in `renderCapReport`**

Replace the summary-cards block's last card and the whole `$("capDialogBody").innerHTML = …` template with:

```javascript
  // Summary cards on the panel.
  const card = (k, v) => `<div class="card"><div class="k">${k}</div><div class="v">${v}</div></div>`;
  $("capCards").innerHTML =
    card("CPU", r.cpu) +
    card("Cores / threads", `${r.cores} / ${r.threads}`) +
    card("Memory", `${fmtBytes(r.memAvail)} free / ${fmtBytes(r.memTotal)}`) +
    card("Mem bandwidth", r.memBwGbps.toFixed(1) + " GB/s") +
    card("GEMM throughput", r.gemmGflops.toFixed(1) + " GFLOP/s") +
    card("Load @int4 ceiling", fmtParams(r.loadInt4)) +
    card("Infer @int8 ceiling", fmtParams(r.inferInt8));
  // Detailed four-range table inside the dialog.
  const rng = (lo, hi) => {
    const l = fmtParams(lo), h = fmtParams(hi);
    return l === h ? l : `${l} – ${h}`;
  };
  const row = (label, val, note) =>
    `<tr><td>${label}</td><td>${val}</td><td class="muted">${note}</td></tr>`;
  const trainLo = Math.min(r.trainChinchilla, r.trainFixed1b);
  const trainHi = Math.max(r.trainChinchilla, r.trainFixed1b);
  $("capDialogBody").innerHTML = `
    <p class="muted">${r.cpu} · ${r.cores} cores · ${r.memBwGbps.toFixed(1)} GB/s · ${r.gemmGflops.toFixed(1)} GFLOP/s</p>
    <table class="data">
      <thead><tr><th>Capability</th><th>Range (params)</th><th>Basis</th></tr></thead>
      <tbody>
        ${row("Load", rng(r.loadF32, r.loadInt4),
              `f32 → int4 (int8: ${fmtParams(r.loadInt8)}); fits in ${Math.round(r.loadFraction * 100)}% of free RAM`)}
        ${row("Train (scratch)", rng(trainLo, trainHi),
              `Chinchilla ${r.chinchillaRatio}× → fixed ${fmtParams(r.fixedTrainTokens)} tokens; < ${r.trainHours} h, RAM-capped @16 B/param`)}
        ${row("Fine-tune", rng(r.finetuneLo, r.finetuneHi),
              `${fmtParams(r.finetuneTokensLo)} → ${fmtParams(r.finetuneTokensHi)} token corpus; < ${r.trainHours} h, RAM-capped @16 B/param`)}
        ${row("Inference", rng(r.inferF32, r.inferInt4),
              `f32 → int4 (int8: ${fmtParams(r.inferInt8)}); ≥ ${r.targetToks} tok/s, bandwidth-bound, RAM-capped`)}
        ${row("Eval pass", fmtParams(r.testEval),
              `${fmtParams(r.evalTokens)} tokens, < ${r.trainHours} h`)}
      </tbody>
    </table>
    <p class="capfoot">Ranges from a live micro-benchmark: each capability spans its natural axis
      (precision f32 → int4, or corpus size). Decode is bandwidth-bound; train/fine-tune/eval are
      compute-bound (≈6·N·T train, 2·N·T eval) and memory-capped. Real throughput varies with
      architecture, context length, and other load.</p>`;
```

- [ ] **Step 4: Update the panel hint (`index.html`)**

Replace the `<p class="hint">` inside `panel-capable` with:

```html
      <p class="hint">Runs a quick (~1–2 s) micro-benchmark of memory bandwidth and
        matmul throughput on <em>this</em> CPU, then estimates the parameter range of an
        SLM this machine can <strong>load</strong>, <strong>train</strong> (&lt; 24 h),
        <strong>fine-tune</strong> (&lt; 24 h), and <strong>run</strong> (≥ 3 tok/s),
        plus a 24 h eval bound. Estimates are upper bounds — real models vary with
        architecture and context length.</p>
```

- [ ] **Step 5: Static gates + field cross-check**

Run, from `ferrum_gui/`:
```bash
node --check ui/app.js && echo "app.js OK"
node --test ui/ 2>&1 | tail -3
for f in load_int4 load_int8 load_f32 finetune_lo finetune_hi finetune_tokens_lo finetune_tokens_hi load_fraction; do
  grep -q "pub $f:" src/capable.rs || echo "MISSING-RUST: $f"
done
for f in loadInt4 loadInt8 loadF32 finetuneLo finetuneHi finetuneTokensLo finetuneTokensHi loadFraction; do
  grep -q "r\.$f" ui/app.js || echo "MISSING-JS: r.$f"
done
```
Expected: `app.js OK`; node tests pass; no `MISSING-*` lines.

- [ ] **Step 6: Commit**

```bash
git add ferrum_gui/ui/app.js ferrum_gui/ui/index.html
git commit -m "feat(gui): capable v2 frontend — four-range table, hard fits-in-RAM budget check"
```

---

### Task 3: Docs

**Files:**
- Modify: `manual/05-using-the-gui.md` (the Capable tab-list line and, if present, any Capable walkthrough section)
- Modify: `ferrum_gui/README.md` (the Capable mention, if a feature row exists)

**Interfaces:** none (docs only).

- [ ] **Step 1: `manual/05-using-the-gui.md`**

Read the file. Update the Capable line in the tab list (currently: "**Capable** | Benchmark this machine and estimate the largest model it can run, train, or evaluate.") to:

```markdown
| **Capable** | Benchmark this machine and see the parameter range of an SLM it can load, train, fine-tune, and run. |
```

If the file contains a longer Capable walkthrough section, reword its capability enumeration the same way (ranges for load/train/fine-tune/run + a 24 h eval bound), keeping its voice; if none exists, add nothing.

- [ ] **Step 2: `ferrum_gui/README.md`**

Read the file. If a requirements/feature row mentions the Capable tab or `capability_report`, update its description to the four ranges (e.g. "parameter ranges: load / train / fine-tune / run (+ eval bound)"). If only the tab list names Capable, no change is needed beyond confirming that.

- [ ] **Step 3: Commit**

```bash
git add manual/05-using-the-gui.md ferrum_gui/README.md
git commit -m "docs: Capable tab now reports load/train/fine-tune/run parameter ranges"
```

(If Step 2 changed nothing, commit only the manual.)

---

### Task 4: Final verification — full gates + boot smoke

**Files:** none (verification only).

- [ ] **Step 1: Backend + frontend gates (from `ferrum_gui/`)**

```bash
cd ferrum_gui && cargo fmt -- --check && cargo clippy --all-targets -- -D warnings 2>&1 | tail -2 && cargo test 2>&1 | grep "test result" | head -1 && node --check ui/app.js && node --test ui/ 2>&1 | tail -2
```
Expected: all clean; ~37 tests passed, 0 failed.

- [ ] **Step 2: Windowed build + boot smoke (snap-scrubbed env)**

```bash
cd ferrum_gui && cargo build 2>&1 | tail -1
rm -f /tmp/ferrum_gui_smoke.log
env -u LD_LIBRARY_PATH -u GTK_PATH -u GIO_MODULE_DIR -u GTK_EXE_PREFIX -u GDK_PIXBUF_MODULE_FILE -u LOCPATH \
  ./target/debug/ferrum_gui >/tmp/ferrum_gui_smoke.log 2>&1 &
PID=$!; sleep 8
if kill -0 $PID 2>/dev/null; then echo "BOOT OK"; kill $PID; else echo "BOOT FAILED"; fi
grep -ci "panic" /tmp/ferrum_gui_smoke.log
```
Expected: `Finished`; `BOOT OK`; panic count `0`. If the launch fails only on a display/GDK error (no display reachable), treat the build + Task 1 tests as the gate and say so honestly.

- [ ] **Step 3: Root workspace untouched**

```bash
cargo test --workspace 2>&1 | grep -c "0 failed"
```
Run from the repo root. Expected: 20 (all suites) — this branch never touches workspace crates.

- [ ] **Step 4: Commit (only if fixups were needed)**

```bash
git add -A && git commit -m "fix(gui): capable v2 verification fixups"  # skip if tree is clean
```

---

## Self-review notes (author)

**Spec coverage:** constants/helpers/report/cap-fix/tests (Task 1 ≙ spec Backend, incl. the `load_cap_params` ∞-sentinel and the `train_max_fixed` refactor-equivalence test); capBounds + hard load check + four-range dialog + cards + hint (Task 2 ≙ spec Frontend, endpoints per spec: Load/Infer f32→int4 with int8 in the note, Train min/max of scenarios, Fine-tune lo→hi); docs (Task 3); AI-only verification incl. field cross-check and boot smoke (Tasks 2/4 ≙ spec Testing).

**Type consistency:** Rust snake_case fields ↔ JS camelCase via the struct's existing `rename_all`; every `r.<field>` in Task 2's template exists in Task 1's struct (checked by the Task 2 Step 5 gate); `capBounds.loadInt4` matches the Task 2 Step 1 assignment.

**Placeholders:** none — all code steps carry complete code; the two docs steps that depend on unread file state give exact target text plus explicit if-absent fallbacks.
