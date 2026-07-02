# GUI Export Tab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an **Export** tab to Ferrum SLM Studio (`ferrum_gui`) that writes a loaded — and optionally fine-tuned — llama/qwen2 GGUF back out at a chosen quantization, wrapping `ferrum_core::write_llama_gguf`.

**Architecture:** One new Tauri command `export_gguf` — a thin `spawn_blocking` wrapper emitting `export-progress` phase events around a **testable core** `do_export(params, &progress_cb)`; one new frontend tab/panel in vanilla HTML/JS mirroring the Fine-tune tab's structure; the existing capability advisory + RAM guard patterns reused verbatim.

**Tech Stack:** Rust (Tauri 2), vanilla HTML/CSS/JS. No new dependencies.

## Global Constraints

- `ferrum_gui` is **excluded from the workspace**: every cargo command in this plan runs **from the `ferrum_gui/` directory** (`cd ferrum_gui && cargo …`), never `-p ferrum_gui` at the root.
- Follow the crate's command conventions exactly: `#[tauri::command]` async fn → `tauri::async_runtime::spawn_blocking` → inner fn; serde `rename_all = "camelCase"` on param/result structs; `Result<_, String>` errors; `ferrum_core::set_verbose` honored and reset.
- Frontend is vanilla JS; `invoke`/`listen`/`dialog`, `$()`, `toast`, `setErr`/`clearErr`, `reqStr` helpers already exist in `ui/app.js` — reuse them; no new CSS classes.
- **No human verification step**: every gate is a command an agent runs (`cargo test`/`check`/`clippy`/`fmt`, `node --check`, `node --test`, scripted ID cross-check, automated boot smoke).
- Progress event name: `export-progress` (string payload). Phase strings begin with `opening`, `loading`, `applying`, `encoding` — tests assert these prefixes.
- UI element ids are prefixed `ex` (`exPath`, `exInspect`, `exInfo`, `exQuant`, `exResume`, `exOut`, `exForce`, `errExport`, `exExport`, `exStatus`, `exResult`); panel id `panel-export`; tab `data-tab="export"`.

---

### Task 1: Backend core — `do_export`, helpers, params/result types, tests

**Files:**
- Modify: `ferrum_gui/src/commands.rs` (new section after the fine-tune section ending near line 1010; tests appended to the existing `#[cfg(test)] mod tests` at line ~1168)

**Interfaces:**
- Consumes (already in scope in `commands.rs`): `ferrum_core::{Gguf, LlamaTrainer}` (imported at top), `gguf_resident_bytes(&Gguf, Option<QKind>) -> usize` (line ~562), `available_memory_bytes() -> Option<usize>` (line ~581).
- Produces:
  - `pub struct ExportParams { model_path, out_path, quant, resume, force, verbose }` (camelCase serde)
  - `pub struct ExportResult { out_path, bytes, seconds, tensor_summary }` (camelCase serde)
  - `pub(crate) fn do_export(p: &ExportParams, progress: &dyn Fn(&str)) -> Result<ExportResult, String>`
  - `fn export_output_estimate(est_model: usize, quant: ferrum_core::GgufQuant) -> usize`
  - `fn ggml_type_name(t: u32) -> String`

- [ ] **Step 1: Write the failing tests**

Append inside the existing `#[cfg(test)] mod tests { … }` block in `ferrum_gui/src/commands.rs`:

```rust
    // ── GGUF export (do_export) ──────────────────────────────────────────────

    use ferrum_core::GgufQuant;

    #[test]
    fn export_output_estimate_matches_cli_heuristic() {
        assert_eq!(export_output_estimate(900, GgufQuant::F32), 900);
        assert_eq!(export_output_estimate(900, GgufQuant::F16), 450);
        assert_eq!(export_output_estimate(900, GgufQuant::Q8_0), 300);
        assert_eq!(export_output_estimate(900, GgufQuant::Q4K), 300);
    }

    #[test]
    fn do_export_rejects_bad_inputs() {
        let base = ExportParams {
            model_path: "/nonexistent.gguf".into(),
            out_path: "/tmp/out.gguf".into(),
            quant: "q8_0".into(),
            resume: None,
            force: false,
            verbose: false,
        };
        let noop = |_: &str| {};

        let mut p = base;
        p.out_path = "  ".into();
        assert!(do_export(&p, &noop).unwrap_err().contains("output"));

        p.out_path = "/tmp/out.gguf".into();
        p.quant = "q9_9".into();
        assert!(do_export(&p, &noop).unwrap_err().contains("q9_9"));
    }

    // Build a minimal valid `llama` GGUF (dim 8, 2 layers, gpt2 tokenizer
    // metadata) with deterministic weights — same shape as ferrum_core's own
    // export-test fixture. Returns the file's bytes.
    fn tiny_llama_gguf() -> Vec<u8> {
        use ferrum_core::{GgufBuilder, MetaValue};
        const GGML_F32: u32 = 0;
        let (dim, n_layers, n_heads, ffn, vocab) = (8usize, 2usize, 2usize, 16usize, 32usize);
        let head_dim = dim / n_heads;
        let gen = |seed: usize, n: usize| -> Vec<f32> {
            (0..n)
                .map(|i| (((seed * 131 + i * 17) % 101) as f32 / 500.0) - 0.1)
                .collect()
        };
        let f32_bytes = |xs: &[f32]| -> Vec<u8> {
            let mut o = Vec::with_capacity(xs.len() * 4);
            for &x in xs {
                o.extend_from_slice(&x.to_le_bytes());
            }
            o
        };
        let mut b = GgufBuilder::new();
        b.meta("general.architecture", MetaValue::String("llama".into()));
        b.meta("llama.embedding_length", MetaValue::U32(dim as u32));
        b.meta("llama.block_count", MetaValue::U32(n_layers as u32));
        b.meta("llama.attention.head_count", MetaValue::U32(n_heads as u32));
        b.meta(
            "llama.attention.head_count_kv",
            MetaValue::U32(n_heads as u32),
        );
        b.meta("llama.feed_forward_length", MetaValue::U32(ffn as u32));
        b.meta("llama.context_length", MetaValue::U32(16));
        b.meta(
            "llama.attention.layer_norm_rms_epsilon",
            MetaValue::F32(1e-5),
        );
        b.meta("tokenizer.ggml.model", MetaValue::String("gpt2".into()));
        b.meta(
            "tokenizer.ggml.tokens",
            MetaValue::Array(
                (0..vocab)
                    .map(|i| MetaValue::String(format!("t{i}")))
                    .collect(),
            ),
        );
        let mut t = |b: &mut GgufBuilder, name: &str, dims: &[u64], seed: usize| {
            let n: usize = dims.iter().product::<u64>() as usize;
            b.tensor(name, dims, GGML_F32, f32_bytes(&gen(seed, n)));
        };
        t(&mut b, "token_embd.weight", &[dim as u64, vocab as u64], 1);
        for i in 0..n_layers {
            let p = format!("blk.{i}");
            t(&mut b, &format!("{p}.attn_norm.weight"), &[dim as u64], 10 + i);
            let qkv = (n_heads * head_dim) as u64;
            t(&mut b, &format!("{p}.attn_q.weight"), &[dim as u64, qkv], 20 + i);
            t(&mut b, &format!("{p}.attn_k.weight"), &[dim as u64, qkv], 30 + i);
            t(&mut b, &format!("{p}.attn_v.weight"), &[dim as u64, qkv], 40 + i);
            t(&mut b, &format!("{p}.attn_output.weight"), &[qkv, dim as u64], 50 + i);
            t(&mut b, &format!("{p}.ffn_norm.weight"), &[dim as u64], 60 + i);
            t(&mut b, &format!("{p}.ffn_gate.weight"), &[dim as u64, ffn as u64], 70 + i);
            t(&mut b, &format!("{p}.ffn_up.weight"), &[dim as u64, ffn as u64], 80 + i);
            t(&mut b, &format!("{p}.ffn_down.weight"), &[ffn as u64, dim as u64], 90 + i);
        }
        t(&mut b, "output_norm.weight", &[dim as u64], 200);
        t(&mut b, "output.weight", &[dim as u64, vocab as u64], 201);
        b.into_bytes()
    }

    #[test]
    fn do_export_roundtrips_a_tiny_model_with_phases() {
        use ferrum_core::Gguf;
        let dir = std::env::temp_dir();
        let pid = std::process::id();
        let src = dir.join(format!("ferrum_gui_export_src_{pid}.gguf"));
        let dst = dir.join(format!("ferrum_gui_export_dst_{pid}.gguf"));
        std::fs::write(&src, tiny_llama_gguf()).unwrap();

        let p = ExportParams {
            model_path: src.to_string_lossy().into_owned(),
            out_path: dst.to_string_lossy().into_owned(),
            quant: "f32".into(),
            resume: None,
            force: false,
            verbose: false,
        };
        let phases = std::cell::RefCell::new(Vec::<String>::new());
        let r = do_export(&p, &|m| phases.borrow_mut().push(m.to_string())).unwrap();

        // Phases arrive in order: opening → loading → encoding (no resume).
        let ph = phases.borrow();
        assert!(ph.len() >= 3, "want ≥3 phases, got {ph:?}");
        assert!(ph[0].starts_with("opening"), "{ph:?}");
        assert!(ph[1].starts_with("loading"), "{ph:?}");
        assert!(ph.last().unwrap().starts_with("encoding"), "{ph:?}");
        assert!(!ph.iter().any(|m| m.starts_with("applying")), "{ph:?}");

        // Result: real file, non-trivial summary naming F32 tensors.
        assert!(r.bytes > 0);
        assert!(r.seconds >= 0.0);
        assert!(r.tensor_summary.contains("F32"), "{}", r.tensor_summary);

        // The output re-opens and re-loads as a runnable llama model.
        let g = Gguf::open(&p.out_path).unwrap();
        assert_eq!(g.architecture(), Some("llama"));
        g.load_llama_prec(None).unwrap();

        let _ = std::fs::remove_file(&src);
        let _ = std::fs::remove_file(&dst);
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd ferrum_gui && cargo test export_ 2>&1 | tail -5` and `cd ferrum_gui && cargo test do_export 2>&1 | tail -5`
Expected: FAIL to compile — `ExportParams`, `do_export`, `export_output_estimate` not defined.

- [ ] **Step 3: Write the implementation**

Add a new section in `ferrum_gui/src/commands.rs` after the fine-tune section (after `finetune_inner`'s closing brace and its trailing helpers, before the `run_terminal` section around line 1011):

```rust
// ─────────────────────────────────────────────────────────────────────────────
// GGUF export (Export tab): write a loaded — optionally fine-tuned — llama/
// qwen2 model back out as a runnable GGUF at a chosen quantization.
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExportParams {
    pub model_path: String,
    pub out_path: String,
    /// Output type name, parsed by `GgufQuant::from_str` ("q8_0", "q4_k", "f16", …).
    pub quant: String,
    /// Optional `.flck` fine-tune checkpoint to overlay before export.
    pub resume: Option<String>,
    /// Bypass the RAM guard.
    pub force: bool,
    pub verbose: bool,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ExportResult {
    pub out_path: String,
    pub bytes: u64,
    pub seconds: f64,
    /// Per-type tensor counts of the written file, e.g. "42× Q6_K, 19× F32".
    pub tensor_summary: String,
}

/// Output-buffer size estimate for the RAM guard: the whole encoded file is
/// built in memory before the atomic write, so a lossless f32 target costs a
/// second model's worth; f16 half; quantized targets ≤ ~a third (conservative).
fn export_output_estimate(est_model: usize, quant: ferrum_core::GgufQuant) -> usize {
    use ferrum_core::GgufQuant as Q;
    match quant {
        Q::F32 => est_model,
        Q::F16 => est_model / 2,
        _ => est_model / 3,
    }
}

/// Human name for the GGML tensor-type ids the exporter can emit.
fn ggml_type_name(t: u32) -> String {
    match t {
        0 => "F32".into(),
        1 => "F16".into(),
        2 => "Q4_0".into(),
        3 => "Q4_1".into(),
        8 => "Q8_0".into(),
        9 => "Q8_1".into(),
        12 => "Q4_K".into(),
        13 => "Q5_K".into(),
        14 => "Q6_K".into(),
        other => format!("type{other}"),
    }
}

/// The testable core of the Export tab: everything `export_gguf` does except
/// event emission. `progress` receives one human-readable line per phase
/// (prefixes: "opening", "loading", "applying", "encoding").
pub(crate) fn do_export(
    p: &ExportParams,
    progress: &dyn Fn(&str),
) -> Result<ExportResult, String> {
    if p.out_path.trim().is_empty() {
        return Err("please provide an output path (.gguf)".into());
    }
    let quant = ferrum_core::GgufQuant::from_str(&p.quant)
        .ok_or_else(|| format!("unknown output type '{}'", p.quant))?;

    progress(&format!("opening {} (streamed)…", p.model_path));
    let g = Gguf::open(&p.model_path).map_err(|e| format!("cannot open {}: {e}", p.model_path))?;

    // Memory guard: f32 model + in-RAM output buffer.
    let est_model = gguf_resident_bytes(&g, None);
    let est = est_model.saturating_add(export_output_estimate(est_model, quant));
    if let Some(avail) = available_memory_bytes() {
        if (est as f64) > 0.9 * avail as f64 && !p.force {
            return Err(format!(
                "estimated peak memory ({:.2} GB: f32 model + output buffer) exceeds 90% of \
                 available ({:.2} GB) — pick a smaller output type or tick 'Export anyway'.",
                est as f64 / 1e9,
                avail as f64 / 1e9
            ));
        }
    }

    let t0 = std::time::Instant::now();
    progress("loading weights (f32)…");
    let mut model = g
        .load_llama_prec(None)
        .map_err(|e| format!("cannot load model: {e}"))?;

    if let Some(ckpt) = &p.resume {
        progress(&format!("applying {ckpt}…"));
        let bytes =
            std::fs::read(ckpt).map_err(|e| format!("cannot read checkpoint {ckpt}: {e}"))?;
        let mut tr = LlamaTrainer::new(model).map_err(|e| format!("cannot wrap model: {e}"))?;
        tr.load_checkpoint_into(&bytes)
            .map_err(|e| format!("cannot apply checkpoint: {e}"))?;
        model = tr.model;
    }

    progress(&format!("encoding + writing ({})…", p.quant));
    ferrum_core::write_llama_gguf(&model, &g, quant, &p.out_path)
        .map_err(|e| format!("export failed: {e}"))?;

    // Per-type summary, read back from the written file (also proves it
    // re-opens as valid GGUF).
    let gout =
        Gguf::open(&p.out_path).map_err(|e| format!("written file failed to re-open: {e}"))?;
    let mut counts: std::collections::BTreeMap<u32, usize> = std::collections::BTreeMap::new();
    for t in &gout.tensors {
        *counts.entry(t.ggml_type).or_default() += 1;
    }
    let tensor_summary = counts
        .iter()
        .map(|(ty, n)| format!("{n}× {}", ggml_type_name(*ty)))
        .collect::<Vec<_>>()
        .join(", ");

    let bytes = std::fs::metadata(&p.out_path).map(|m| m.len()).unwrap_or(0);
    Ok(ExportResult {
        out_path: p.out_path.clone(),
        bytes,
        seconds: t0.elapsed().as_secs_f64(),
        tensor_summary,
    })
}
```

Note: `Gguf` and `LlamaTrainer` are already in the file's top `use ferrum_core::{…}` list — do not re-import them; `GgufQuant`/`GgufBuilder`/`MetaValue` are referenced via full paths / test-local imports, so the top-level import list needs no change.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd ferrum_gui && cargo test export 2>&1 | tail -6`
Expected: PASS — `export_output_estimate_matches_cli_heuristic`, `do_export_rejects_bad_inputs`, `do_export_roundtrips_a_tiny_model_with_phases` (and no other test broken: `cargo test 2>&1 | tail -3` all green).

- [ ] **Step 5: fmt + clippy gates**

Run: `cd ferrum_gui && cargo fmt -- --check && cargo clippy --all-targets -- -D warnings 2>&1 | tail -3`
Expected: fmt clean; clippy `Finished` with no warnings. (If `do_export` triggers `dead_code` because nothing outside tests calls it yet, add `#[allow(dead_code)] // wired to the export_gguf command in the next commit` on `do_export` only, and remove it in Task 2.)

- [ ] **Step 6: Commit**

```bash
git add ferrum_gui/src/commands.rs
git commit -m "feat(gui): add do_export core for the Export tab, with fixture round-trip tests"
```

---

### Task 2: `export_gguf` command wrapper + registration

**Files:**
- Modify: `ferrum_gui/src/commands.rs` (immediately after `do_export`)
- Modify: `ferrum_gui/src/lib.rs` (handler list, after `commands::finetune_gguf,` at line ~64)

**Interfaces:**
- Consumes: `do_export` (Task 1); `AppHandle` + `Emitter` (already imported at top of `commands.rs`).
- Produces: `#[tauri::command] pub async fn export_gguf(app: AppHandle, params: ExportParams) -> Result<ExportResult, String>`, emitting `export-progress` string events; registered in `lib.rs`.

- [ ] **Step 1: Add the command wrapper**

Insert directly after `do_export` in `commands.rs`:

```rust
/// Export a llama/qwen2 GGUF (optionally with a fine-tune checkpoint applied)
/// to a new GGUF at the requested quantization. Phase progress streams as
/// `export-progress` events (see [`do_export`] for the phases).
#[tauri::command]
pub async fn export_gguf(app: AppHandle, params: ExportParams) -> Result<ExportResult, String> {
    tauri::async_runtime::spawn_blocking(move || {
        ferrum_core::set_verbose(params.verbose);
        let emit = |m: &str| {
            let _ = app.emit("export-progress", m.to_string());
        };
        let r = do_export(&params, &emit);
        ferrum_core::set_verbose(false);
        r
    })
    .await
    .map_err(|e| format!("task error: {e}"))?
}
```

If Task 1 added `#[allow(dead_code)]` on `do_export`, remove it now.

- [ ] **Step 2: Register the handler**

In `ferrum_gui/src/lib.rs`, add one line to the `tauri::generate_handler![…]` list, after `commands::finetune_gguf,`:

```rust
            commands::export_gguf,
```

- [ ] **Step 3: Verify compile + full backend gates**

Run: `cd ferrum_gui && cargo check 2>&1 | tail -2 && cargo test 2>&1 | tail -3 && cargo fmt -- --check && cargo clippy --all-targets -- -D warnings 2>&1 | tail -2`
Expected: check `Finished`; all tests pass; fmt clean; clippy clean.

- [ ] **Step 4: Commit**

```bash
git add ferrum_gui/src/commands.rs ferrum_gui/src/lib.rs
git commit -m "feat(gui): export_gguf command with export-progress events"
```

---

### Task 3: Frontend — Export tab, panel, and handlers

**Files:**
- Modify: `ferrum_gui/ui/index.html` (tab button in `#tabs` at line ~31; new panel section after `panel-finetune`, before `panel-tabular` at line ~352)
- Modify: `ferrum_gui/ui/app.js` (handlers after the Fine-tune section ending near line 640; listener in the startup `listen` block near lines 752-772)

**Interfaces:**
- Consumes: `invoke`, `listen`, `$`, `toast`, `setErr`/`clearErr`, `reqStr`, `checkGgufBudget`, `confirmGgufWarning`, `capBounds` — all existing in `app.js`; the generic tab switcher and `.browse`/`.browse-save` bindings (both already bound); backend commands `gguf_info` and `export_gguf` (Task 2).
- Produces: ids `exPath, exInspect, exInfo, exQuant, exResume, exOut, exForce, errExport, exExport, exStatus, exResult`, panel `panel-export`, tab `data-tab="export"`.

- [ ] **Step 1: Add the tab button**

In `ferrum_gui/ui/index.html`, in the `<nav id="tabs">` list, after the Fine-tune button:

```html
    <button class="tab" data-tab="export">Export</button>
```

- [ ] **Step 2: Add the panel**

Insert after the closing `</section>` of `panel-finetune` (before the `<!-- ── Tabular (train_cli) … -->` comment):

```html
    <!-- ── Export (write a GGUF back out) ───────────────────────────────── -->
    <section class="panel" id="panel-export">
      <h2>Export — write a model back to GGUF</h2>
      <p class="hint">Serializes a Llama/Qwen <code>.gguf</code> — optionally with a
        fine-tune checkpoint applied — to a new GGUF that runs in llama.cpp, ollama,
        or LM Studio. <strong>f16/f32</strong> are lossless (f32 masters); re-quantizing an
        already-quantized file is lossy. Norms and biases stay f32; a matrix whose row
        length is not block-aligned for the chosen type falls back to f16 (the summary
        below shows exactly what was written).</p>
      <div class="row">
        <label>Source GGUF <input type="text" id="exPath" placeholder="/path/to/model.gguf" /></label>
        <button class="ghost browse" data-target="exPath">Browse…</button>
        <button id="exInspect" class="ghost">Inspect</button>
      </div>
      <div id="exInfo" class="infocards"></div>
      <div class="grid">
        <label>Output type
          <select id="exQuant">
            <option value="q8_0">q8_0 (default — high quality)</option>
            <option value="q4_k">q4_k (small, modern)</option>
            <option value="q5_k">q5_k</option>
            <option value="q6_k">q6_k (near-lossless)</option>
            <option value="q4_0">q4_0 (legacy)</option>
            <option value="q4_1">q4_1 (legacy)</option>
            <option value="q8_1">q8_1 (legacy)</option>
            <option value="f16">f16 (lossless, 2 B/param)</option>
            <option value="f32">f32 (lossless, 4 B/param)</option>
          </select>
        </label>
      </div>
      <div class="row">
        <label>Fine-tune checkpoint (optional) <input type="text" id="exResume" placeholder="/path/to/finetune.flck — export tuned weights" /></label>
        <button class="ghost browse" data-target="exResume">Browse…</button>
      </div>
      <div class="row">
        <label>Output file (.gguf) <input type="text" id="exOut" placeholder="/path/to/out.gguf" /></label>
        <button class="ghost browse-save" data-target="exOut">Save as…</button>
      </div>
      <label class="chk"><input type="checkbox" id="exForce" /> Export even if the estimate exceeds available RAM</label>
      <span class="err" id="errExport"></span>
      <div class="row">
        <button id="exExport" class="primary">Export</button>
        <span id="exStatus" class="muted"></span>
      </div>
      <div id="exResult" class="result"></div>
    </section>
```

- [ ] **Step 3: Add the JS handlers**

In `ferrum_gui/ui/app.js`, insert after the Fine-tune section (after the `$("ftRun")…` handler's closing `});`, before the `// ── Tabular …` comment):

```javascript
// ── Export (write a GGUF back out) ──────────────────────────────────────────
$("exInspect").addEventListener("click", async () => {
  clearErr("errExport");
  let path;
  try { path = reqStr("exPath", "Source GGUF"); } catch (e) { setErr("errExport", String(e)); return; }
  $("exStatus").textContent = "inspecting…";
  try {
    const i = await invoke("gguf_info", { path });
    const card = (k, v) => `<div class="card"><div class="k">${k}</div><div class="v">${v}</div></div>`;
    const avail = i.availMb == null ? "unknown" : (i.availMb / 1024).toFixed(2) + " GB";
    $("exInfo").innerHTML =
      card("Architecture", i.architecture + (i.runnable ? " ✓" : " ✗ not exportable")) +
      card("GGUF", "v" + i.version + " · " + i.numTensors + " tensors") +
      card("Shape", `dim ${i.modelDim} · ${i.nLayers}L · ${i.nHeads}h/${i.nKvHeads}kv · vocab ${i.vocabSize}`) +
      card("Tokenizer", i.tokenizer) +
      card("Resident f32 (export loads f32)", (i.estF32Mb / 1024).toFixed(2) + " GB") +
      card("Available RAM", avail);
    $("exStatus").textContent = "";
    toast(i.runnable ? "GGUF inspected" : "Inspected — architecture not exportable", i.runnable ? "ok" : "error");
    const crossed = checkGgufBudget(i.paramCount);
    if (crossed.length) await confirmGgufWarning(i.paramCount, crossed); // advisory; result ignored
  } catch (e) { $("exStatus").textContent = "error"; setErr("errExport", String(e)); toast("Inspect failed: " + e, "error"); }
});

$("exExport").addEventListener("click", async () => {
  clearErr("errExport");
  let params;
  try {
    const resume = $("exResume").value.trim();
    params = {
      modelPath: reqStr("exPath", "Source GGUF"),
      outPath: reqStr("exOut", "Output file"),
      quant: $("exQuant").value,
      resume: resume === "" ? null : resume,
      force: $("exForce").checked,
      verbose: false,
    };
  } catch (e) { setErr("errExport", String(e)); return; }

  // Budget gate (advisory + confirm), mirroring the GGUF Run button.
  if (capBounds) {
    try {
      const info = await invoke("gguf_info", { path: params.modelPath });
      const crossed = checkGgufBudget(info.paramCount);
      if (crossed.length) {
        const ok = await confirmGgufWarning(info.paramCount, crossed);
        if (!ok) { $("exStatus").textContent = "cancelled"; return; }
      }
    } catch (_) { /* inspection failed; the export itself will error clearly */ }
  }

  $("exResult").textContent = "";
  $("exStatus").textContent = "starting… (CPU: a large model can take minutes)";
  $("exExport").disabled = true;
  try {
    const r = await invoke("export_gguf", { params });
    $("exStatus").textContent = "done";
    $("exResult").textContent =
      `Wrote ${r.bytes.toLocaleString()} bytes → ${r.outPath} in ${r.seconds.toFixed(1)}s\n` +
      `tensors: ${r.tensorSummary}\n` +
      `The file runs in llama.cpp / ollama / LM Studio as-is.`;
    toast("Export complete", "ok");
  } catch (e) {
    $("exStatus").textContent = "error";
    setErr("errExport", String(e));
    toast("Export failed: " + e, "error");
  } finally { $("exExport").disabled = false; }
});
```

- [ ] **Step 4: Add the progress listener**

In the startup block where the other listeners live (directly after the `finetune-progress` listener around line 763-771), add:

```javascript
  await listen("export-progress", (e) => { $("exStatus").textContent = e.payload; });
```

- [ ] **Step 5: Static frontend gates**

Run, from `ferrum_gui/`:
```bash
node --check ui/app.js && echo "app.js OK"
node --test ui/ 2>&1 | tail -3        # stream.test.js must still pass
# ID cross-check: every $(\"ex…\")/errExport id used in app.js exists in index.html
for id in $(grep -o '\$("e[xr][A-Za-z]*")' ui/app.js | sed 's/\$("//;s/")//' | sort -u); do
  grep -q "id=\"$id\"" ui/index.html || echo "MISSING: $id"
done
grep -q 'data-tab="export"' ui/index.html && grep -q 'id="panel-export"' ui/index.html && echo "tab+panel OK"
```
Expected: `app.js OK`; node --test all pass; no `MISSING:` lines; `tab+panel OK`.

- [ ] **Step 6: Commit**

```bash
git add ferrum_gui/ui/index.html ferrum_gui/ui/app.js
git commit -m "feat(gui): Export tab — panel, handlers, live export-progress status"
```

---

### Task 4: Docs

**Files:**
- Modify: `manual/05-using-the-gui.md` (tab list at line ~67; new walkthrough section)
- Modify: `manual/03-the-ferrum-engine-and-its-capabilities.md` (the "(import/run only for now)" phrase)
- Modify: `status.md` (`ferrum_gui` bullet, line ~76)
- Modify: `ferrum_gui/README.md` (tab table row 4, feature row ~25, feature matrix line ~83)

**Interfaces:** none (docs only). The tab lists in `manual/05` and `ferrum_gui/README.md` are stale beyond this feature (they predate the Fine-tune and Capable tabs) — fix them fully while touching them.

- [ ] **Step 1: `manual/05-using-the-gui.md`**

Read the file first. In the "The tabs, left to right:" list (~line 67), make the list match the real tab bar: `Datasets / Train / Generate / Evaluate / Models / GGUF / Fine-tune / Export / Tabular (CLI) / System / Capable`, adding a one-line description for Export: "**Export** — write a loaded (or fine-tuned) GGUF back out at a chosen quantization." Then add a walkthrough section after the GGUF/fine-tune how-to sections, matching the file's numbered-steps style:

```markdown
## Export a model back to GGUF

1. Go to the **Export** tab. **Browse…** to the source `.gguf` and click
   **Inspect** to confirm the architecture is exportable (llama/qwen2).
2. Pick an **Output type** — `q8_0` is a good default; `f16`/`f32` are
   lossless; the k-quants are smallest. Re-quantizing an already-quantized
   file is lossy.
3. To export a fine-tune, point **Fine-tune checkpoint** at the `.flck` the
   Fine-tune tab produced.
4. **Save as…** the output path and click **Export**. The status line follows
   the phases (opening → loading → writing), and the result shows a per-type
   tensor summary — any matrix that fell back to f16 shows up there.
5. The written file runs unchanged in llama.cpp, ollama, or LM Studio.
```

- [ ] **Step 2: `manual/03-…capabilities.md`**

Change the §3.5b closing line `or the GUI's **GGUF** tab (import/run only for now).` to `or the GUI's **GGUF** and **Export** tabs.`

- [ ] **Step 3: `status.md`**

Update the `ferrum_gui` bullet (currently "…plus a **GGUF** panel (`gguf_info` / `run_gguf`)…") to name all four commands: `(gguf_info / run_gguf / finetune_gguf / export_gguf)` and mention the Export tab.

- [ ] **Step 4: `ferrum_gui/README.md`**

Read the file first. Fix row 4's tab list to the real set (add Fine-tune, **Export**, Capable); extend row 7 (or add a row) for export: "Export / re-quantize a GGUF | **Export** (source, output type, checkpoint overlay, save-as) | `export_gguf`"; update the feature-matrix line "GGUF import & run" to "GGUF import, run & export".

- [ ] **Step 5: Commit**

```bash
git add manual/05-using-the-gui.md manual/03-the-ferrum-engine-and-its-capabilities.md status.md ferrum_gui/README.md
git commit -m "docs: document the GUI Export tab (and repair stale tab lists)"
```

---

### Task 5: Final verification — full gates + automated boot smoke

**Files:** none created (verification only; a scratch log under /tmp).

- [ ] **Step 1: Backend gates (from `ferrum_gui/`)**

```bash
cd ferrum_gui && cargo fmt -- --check && cargo clippy --all-targets -- -D warnings 2>&1 | tail -2 && cargo test 2>&1 | tail -3
```
Expected: all clean/green.

- [ ] **Step 2: Frontend gates**

```bash
cd ferrum_gui && node --check ui/app.js && node --test ui/ 2>&1 | tail -3
```
Expected: clean; stream tests pass.

- [ ] **Step 3: Windowed build**

```bash
cd ferrum_gui && cargo build 2>&1 | tail -2
```
Expected: `Finished` (WebView system libs are present on this machine — `ferrum_gui/target/` exists from prior builds).

- [ ] **Step 4: Automated boot smoke**

```bash
cd ferrum_gui && rm -f /tmp/ferrum_gui_smoke.log && ./target/debug/ferrum_gui >/tmp/ferrum_gui_smoke.log 2>&1 &
PID=$!; sleep 8
if kill -0 $PID 2>/dev/null; then echo "BOOT OK (alive after 8s)"; kill $PID; else echo "BOOT FAILED"; fi
grep -i "panic" /tmp/ferrum_gui_smoke.log && echo "PANIC FOUND" || echo "no panics"
```
Expected: `BOOT OK (alive after 8s)` and `no panics`. If the launch fails **only** because no display server is reachable (log shows a display/GDK error, not a panic), record that verbatim and treat Step 3's successful build + Tasks 1-2's tests as the gate — state this honestly in the report rather than claiming a boot.

- [ ] **Step 5: Root workspace untouched**

```bash
cargo test --workspace 2>&1 | grep -c "FAILED" || echo "workspace green"
```
Run from the repo root. Expected: `workspace green` (this feature never touches workspace crates).

- [ ] **Step 6: Commit (only if any fixups were needed)**

```bash
git add -A && git commit -m "fix(gui): verification fixups for the Export tab" # skip if tree is clean
```

---

## Self-review notes (author)

**Spec coverage:** command + testable core + estimate helper (Task 1-2 ≙ spec Backend); tab/panel/handlers/listener/save-picker (Task 3 ≙ spec Frontend — `.browse-save` binding already exists at app.js:79, so the spec's conditional "add the binding" resolves to no-op); advisory + RAM guard reuse (Tasks 1&3); docs incl. the stale-tab-list repair (Task 4 ≙ spec Docs); AI-only verification incl. fixture e2e, static gates, ID cross-check, boot smoke with honest no-display fallback (Tasks 1,3,5 ≙ spec Testing).

**Type consistency:** `ExportParams`/`ExportResult` field names match the JS `params` object (camelCase serde: `modelPath`/`outPath`/`quant`/`resume`/`force`/`verbose`) and the JS reads `r.outPath`/`r.bytes`/`r.seconds`/`r.tensorSummary`. `do_export(&ExportParams, &dyn Fn(&str))` used identically in wrapper and tests. Phase prefixes consistent between implementation and test assertions.

**Placeholders:** none — every code step carries complete code; doc steps that depend on unread file bodies say "read the file first" and give the exact target text.
