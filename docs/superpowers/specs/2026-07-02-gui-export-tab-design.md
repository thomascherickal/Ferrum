# GUI Export Tab — GGUF Export in Ferrum SLM Studio

Date: 2026-07-02
Status: Approved (design)

## Purpose

`ferrum_core` can export a loaded (and optionally fine-tuned) llama/qwen2 model
back to a runnable GGUF v3 file, and the CLI exposes it as `export-gguf` — but
the GUI (`ferrum_gui`, "Ferrum SLM Studio") only imports/runs/fine-tunes. This
spec adds a dedicated **Export** tab, closing the GUI's feature-parity gap:

```
GGUF tab (run) ── Fine-tune tab (.flck) ── Export tab (this spec)
                                              │
                                              ▼
                              out.gguf → llama.cpp / ollama / LM Studio
```

### Scope

**In scope:** one new Tauri command (`export_gguf`) mirroring the CLI's
`cmd_export_gguf`; one new frontend tab/panel; phase-event progress; docs.

**Out of scope:** any change to `ferrum_core`; exporting native SLM/MLP models;
batch export; post-fine-tune auto-export shortcuts.

### Constraints

- `ferrum_gui` is **excluded from the workspace**: build/test from its own
  directory (`cd ferrum_gui && cargo …`), never `-p ferrum_gui` at the root.
- Follow the crate's established command pattern exactly: `#[tauri::command]`
  `async fn` → `spawn_blocking` → `_inner`, serde `camelCase` param/result
  structs, `Result<_, String>` errors, `set_verbose` honored and reset.
- Frontend is vanilla HTML/CSS/JS; no new dependencies anywhere.
- **Everything is AI-verifiable** — no step of implementation or verification
  is left to a human (see Testing).

## Backend (`ferrum_gui/src/commands.rs`, registered in `src/lib.rs`)

### Types

```rust
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExportParams {
    pub model_path: String,        // source .gguf
    pub out_path: String,          // destination .gguf
    pub quant: String,             // "q8_0" (UI default) … parsed by GgufQuant::from_str
    pub resume: Option<String>,    // optional .flck fine-tune checkpoint
    pub force: bool,               // bypass the RAM guard
    pub verbose: bool,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ExportResult {
    pub out_path: String,
    pub bytes: u64,                // written file size
    pub seconds: f64,              // wall time
    pub tensor_summary: String,    // e.g. "42× Q6_K, 19× F32" (from re-opening the output)
}
```

### Command and testable core

```rust
#[tauri::command]
pub async fn export_gguf(app: AppHandle, params: ExportParams) -> Result<ExportResult, String>;
```

The command is a thin wrapper: it `spawn_blocking`s into
`do_export(params, &progress)` where **`progress: &dyn Fn(&str)`** — the
wrapper's closure emits each phase string as an `export-progress` event
(`app.emit("export-progress", msg)`); tests pass a no-op or collector closure.
This split is what makes the whole backend path machine-testable without a
window.

`do_export` flow (mirrors the CLI):

1. Validate: `out_path` non-empty; `quant` parses (`GgufQuant::from_str`);
   error strings match sibling commands' tone.
2. `progress("opening <file> (streamed)…")` → `Gguf::open`.
3. **RAM guard**: `est_model = gguf_resident_bytes(&g, None)` (existing helper)
   plus an output-buffer estimate from a small pure function
   `export_output_estimate(est_model, quant)` (f32 → `est_model`, f16 → `/2`,
   else `/3` — same heuristic as the CLI). If `est_model + est_out` exceeds 90%
   of available memory and `!force`, return an error naming the **Force**
   checkbox. (The existing `available_memory_bytes` helper supplies the limit.)
4. `progress("loading weights (f32)…")` → `load_llama_prec(None)`.
5. If `resume`: `progress("applying <ckpt>…")` → read bytes →
   `LlamaTrainer::new(model)` → `load_checkpoint_into` → `model = trainer.model`.
6. `progress("encoding + writing (<quant>)…")` → `ferrum_core::write_llama_gguf`.
7. Summary: re-open the written file (`Gguf::open`), tally `ggml_type` counts
   into `"N× TYPE"` strings joined with `", "` (reusing a `ggml_type_name`
   match like the CLI's), stat the size, return `ExportResult`.

`export_gguf` is added to the `generate_handler![…]` list in `lib.rs`.

## Frontend (`ferrum_gui/ui/index.html`, `ui/app.js`)

### Tab + panel (`index.html`)

New tab button after Fine-tune: `<button class="tab" data-tab="export">Export</button>`,
and a `panel-export` section (approved mockup):

- **Source GGUF** text input `exPath` + `Browse…` (`.browse`, `data-target="exPath"`)
  + `Inspect` button `exInspect`; info cards render into `exInfo` (same
  `gguf_info` command and card markup as the GGUF tab).
- **Output type** select `exQuant`, options
  `q8_0 (default) | q4_k | q5_k | q6_k | q4_0 | q4_1 | q8_1 | f16 | f32`, with a
  hint: lossless = `f16`/`f32` or exporting fine-tuned f32 masters; re-quantizing
  a quantized source is lossy; non-block-aligned matrices fall back to f16 and
  show up in the summary.
- **Fine-tune checkpoint (optional)** input `exResume` + `Browse…`.
- **Output file** input `exOut` + **Save as…** button using the existing
  `pickSave` helper (bound via the save-picker class/binding already present in
  `app.js`; if only `.browse` is bound today, add the one-line binding for a
  `.browse-save` class as part of this work).
- **Force** checkbox `exForce` ("Export even if it may exceed available RAM"),
  error span `errExport`, primary button `exExport`, status span `exStatus`,
  result div `exResult`.

### Behavior (`app.js`)

- `listen("export-progress", e => exStatus.textContent = e.payload)` registered
  once at startup alongside the finetune listener.
- `exInspect` click: `invoke("gguf_info", { path })` → render cards; on a
  capability crossing, show the existing `confirmGgufWarning` advisory
  (result ignored — advisory only), exactly like the GGUF tab's Inspect.
- `exExport` click: client-side validation (paths non-empty) → optional
  advisory (`gguf_info` param count vs capability bounds, same as Run; the
  dialog's Proceed/Cancel is honored) → disable button → `invoke("export_gguf",
  { params })` → render `exResult` as `"<tensor_summary> — <size> in <s>s"` +
  toast → re-enable in `finally`. Errors go to `errExport` + toast, matching
  siblings.

No new CSS classes — the panel reuses `row`, `grid`, `infocards`, `chk`, `err`,
`output`, `primary`, `ghost`.

## Docs

- `manual/05-using-the-gui.md`: new "Export" tab section (what it does, the
  lossless-vs-lossy note, the fine-tune → export flow).
- `manual/03-…capabilities.md`: update "(import/run only for now)" to include
  export.
- `status.md`: extend the `ferrum_gui` bullet (`gguf_info` / `run_gguf` /
  `finetune_gguf` / **`export_gguf`**).
- `ferrum_gui/README.md`: add the tab to its tab list if one exists.

## Error handling

All backend errors are `Err(String)` in the crate's existing voice: empty
output path, unknown quant name, guard refusal (names the Force checkbox),
open/load/checkpoint/write failures prefixed with context. The frontend never
leaves the button disabled (re-enable in `finally`) and always surfaces the
message in `errExport` + a toast.

## Testing — fully AI-verifiable, no human step

1. **Backend unit/integration tests** (in `commands.rs`'s test module, run via
   `cd ferrum_gui && cargo test`):
   - `export_output_estimate` heuristic values for f32/f16/quantized.
   - **End-to-end `do_export`**: build a tiny valid llama GGUF fixture with
     `ferrum_core::GgufBuilder` (same shape as `ferrum_core`'s test fixture:
     dim 8, 2 layers, gpt2 tokenizer metadata) into a temp dir; run `do_export`
     with a collector closure; assert the output re-opens
     (`Gguf::open` → `architecture() == Some("llama")` →
     `load_llama_prec(None)` ok), the `tensor_summary` is non-empty and names
     F32/F16, and the progress collector saw the expected phase prefixes in
     order.
   - Validation paths: empty `out_path` errors; bad `quant` errors.
2. **Static gates** (from `ferrum_gui/`): `cargo check`, `cargo clippy
   --all-targets -- -D warnings`, `cargo fmt -- --check` (headless — the
   backend type-checks without a window), plus root-repo gates untouched.
3. **Frontend gates**: `node --check ui/app.js`; `node --test ui/` (the
   existing `stream.test.js` must still pass); a scripted ID-consistency check
   during verification: every `$("ex…")` id referenced in `app.js` exists in
   `index.html` (grep cross-reference, not a committed test).
4. **Windowed boot smoke, automated**: the WebView system libraries are present
   on this machine (`ferrum_gui/target/` exists from prior builds), so:
   `cargo build` must succeed, then launch the built binary with a timeout
   (`timeout 20s`), assert it stays alive for the grace period without
   panicking, then kill it. This proves the app boots with the new tab wired
   in. (DOM click-through automation is explicitly not attempted; the
   command layer it would exercise is covered by the `do_export` test.)

## Files touched

| File | Change |
|------|--------|
| `ferrum_gui/src/commands.rs` | `ExportParams`/`ExportResult`, `export_gguf` command, `do_export` core, `export_output_estimate` + `ggml_type_name` helpers, tests |
| `ferrum_gui/src/lib.rs` | register `commands::export_gguf` |
| `ferrum_gui/ui/index.html` | Export tab button + `panel-export` section |
| `ferrum_gui/ui/app.js` | export handlers, `export-progress` listener, save-picker binding if missing |
| `manual/05-using-the-gui.md`, `manual/03-…`, `status.md`, `ferrum_gui/README.md` | docs |
