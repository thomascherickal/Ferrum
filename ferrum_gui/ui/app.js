// Ferrum SLM Studio — vanilla JS frontend (no frameworks).
"use strict";

// ── Tauri bindings (gracefully degrade if opened outside Tauri) ─────────────
const T = window.__TAURI__;
const hasTauri = !!(T && T.core && typeof T.core.invoke === "function");
const invoke = hasTauri ? T.core.invoke : async () => { throw new Error("Not running inside the Tauri runtime"); };
const listen = hasTauri ? T.event.listen : async () => () => {};
const dialog = hasTauri && T.dialog ? T.dialog : null;

const $ = (id) => document.getElementById(id);
const bySel = (s) => Array.from(document.querySelectorAll(s));

// ── Toasts & errors ─────────────────────────────────────────────────────────
function toast(msg, kind = "info") {
  const el = document.createElement("div");
  el.className = "toast " + (kind === "error" ? "err" : kind === "ok" ? "ok" : "");
  el.textContent = msg;
  $("toasts").appendChild(el);
  setTimeout(() => el.remove(), 5000);
}
function setErr(id, msg) { const e = $(id); if (e) e.textContent = msg || ""; }
function clearErr(id) { setErr(id, ""); }

// ── Number/validation helpers ───────────────────────────────────────────────
function num(id) { return parseFloat($(id).value); }
function int(id) { return parseInt($(id).value, 10); }
function reqInt(id, name, min = 1) {
  const v = int(id);
  if (!Number.isFinite(v) || v < min) throw new Error(`${name} must be an integer ≥ ${min}`);
  return v;
}
function reqNum(id, name, min = 0, allowEq = false) {
  const v = num(id);
  if (!Number.isFinite(v) || (allowEq ? v < min : v <= min)) {
    throw new Error(`${name} must be a number ${allowEq ? "≥" : ">"} ${min}`);
  }
  return v;
}
function reqStr(id, name) {
  const v = $(id).value.trim();
  if (!v) throw new Error(`${name} is required`);
  return v;
}

// ── Tabs ─────────────────────────────────────────────────────────────────────
bySel(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    bySel(".tab").forEach((b) => b.classList.remove("active"));
    bySel(".panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    $("panel-" + btn.dataset.tab).classList.add("active");
  });
});

// ── File dialogs (Browse buttons) ────────────────────────────────────────────
async function pickOpen(targetId) {
  if (!dialog) { toast("File dialog unavailable; type the path manually.", "info"); return; }
  try {
    const file = await dialog.open({ multiple: false, directory: false });
    if (file) $(targetId).value = typeof file === "string" ? file : file.path || file;
  } catch (e) { toast("dialog error: " + e, "error"); }
}
async function pickSave(targetId) {
  if (!dialog) { toast("File dialog unavailable; type the path manually.", "info"); return; }
  try {
    const file = await dialog.save({});
    if (file) $(targetId).value = typeof file === "string" ? file : file.path || file;
  } catch (e) { toast("dialog error: " + e, "error"); }
}
async function pickDir(targetId) {
  if (!dialog) { toast("File dialog unavailable; type the path manually.", "info"); return; }
  try {
    const dir = await dialog.open({ multiple: false, directory: true });
    if (dir) $(targetId).value = typeof dir === "string" ? dir : dir.path || dir;
  } catch (e) { toast("dialog error: " + e, "error"); }
}
bySel(".browse").forEach((b) => b.addEventListener("click", () => pickOpen(b.dataset.target)));
bySel(".browse-save").forEach((b) => b.addEventListener("click", () => pickSave(b.dataset.target)));
bySel(".browse-dir").forEach((b) => b.addEventListener("click", () => pickDir(b.dataset.target)));

// ── Terminal (bottom dock) ───────────────────────────────────────────────────
const termOut = $("termOut");
function termLine(text, cls = "") {
  const div = document.createElement("div");
  div.className = "line " + cls;
  div.textContent = text;
  termOut.appendChild(div);
  // Keep last ~2000 lines.
  while (termOut.childElementCount > 2000) termOut.removeChild(termOut.firstChild);
  termOut.scrollTop = termOut.scrollHeight;
}
$("termClear").addEventListener("click", () => { termOut.innerHTML = ""; });

let termBusy = false;
async function refreshPrompt() {
  if (!hasTauri) { $("termPrompt").textContent = "$"; return; }
  try { $("termPrompt").textContent = (await invoke("term_cwd")) + " $"; } catch { $("termPrompt").textContent = "$"; }
}
async function runTerminal(cmd) {
  if (!cmd.trim()) return;
  termLine($("termPrompt").textContent + " " + cmd, "cmd");
  if (!hasTauri) { termLine("(backend unavailable outside Tauri)", "stderr"); return; }
  termBusy = true; $("termCmd").disabled = true;
  try {
    const code = await invoke("run_terminal", { command: cmd });
    if (code !== 0) termLine(`[exit ${code}]`, "sys");
  } catch (e) {
    termLine(String(e), "stderr");
  } finally {
    termBusy = false; $("termCmd").disabled = false;
    await refreshPrompt(); $("termCmd").focus();
  }
}
$("termCmd").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !termBusy) {
    const v = $("termCmd").value; $("termCmd").value = ""; runTerminal(v);
  }
});

// ── Datasets ─────────────────────────────────────────────────────────────────
let cleanedCorpus = "";

$("dsDownload").addEventListener("click", async () => {
  clearErr("errDsSource");
  try {
    const url = reqStr("dsUrl", "Source URL");
    const maxBytes = Math.max(1, int("dsMaxMb") || 8) * 1024 * 1024;
    $("dsDownload").disabled = true; toast("Downloading…");
    const text = await invoke("download_text", { url, maxBytes });
    $("dsRaw").value = text;
    toast(`Downloaded ${text.length.toLocaleString()} chars`, "ok");
  } catch (e) { setErr("errDsSource", String(e)); }
  finally { $("dsDownload").disabled = false; }
});

$("dsLoadFile").addEventListener("click", async () => {
  clearErr("errDsSource");
  try {
    const path = reqStr("dsFile", "File path");
    const text = await invoke("read_text_file", { path, maxBytes: null });
    $("dsRaw").value = text;
    toast(`Loaded ${text.length.toLocaleString()} chars`, "ok");
  } catch (e) { setErr("errDsSource", String(e)); }
});

// ── Dataset catalog (HuggingFace / Kaggle) ───────────────────────────────────
function destDir() {
  const d = $("dsDestDir").value.trim();
  if (!d) throw new Error("Choose a download folder first");
  return d;
}

// After a dataset lands on disk, point the local-file loader at it so it flows
// straight into the clean pipeline.
function afterDownload(res) {
  const mb = (res.bytes / (1024 * 1024)).toFixed(2);
  $("dsFile").value = res.path;
  toast(`Downloaded ${mb} MB from ${res.source} → ${res.path}. Click “Load file”.`, "ok");
}

function renderCatalog(list) {
  const box = $("dsCatalog");
  box.innerHTML = "";
  if (!list.length) { box.textContent = "No datasets listed."; return; }
  for (const d of list) {
    const card = document.createElement("div");
    card.className = "catalog-item";
    const meta = document.createElement("div");
    meta.innerHTML =
      `<strong>${d.name}</strong> <span class="badge">${d.source}</span> ` +
      `<span class="muted">~${d.approxMb} MB · ${d.format}</span><br>` +
      `<span class="muted">${d.description}</span>`;
    const btn = document.createElement("button");
    btn.textContent = "Download";
    btn.addEventListener("click", async () => {
      clearErr("errDsCatalog");
      try {
        btn.disabled = true; toast(`Downloading ${d.name}…`);
        const res = await invoke("download_dataset", { id: d.id, destDir: destDir() });
        afterDownload(res);
      } catch (e) { setErr("errDsCatalog", String(e)); }
      finally { btn.disabled = false; }
    });
    card.appendChild(meta);
    card.appendChild(btn);
    box.appendChild(card);
  }
}

async function loadCatalog() {
  try {
    renderCatalog(await invoke("list_datasets"));
  } catch (e) { setErr("errDsCatalog", String(e)); }
}

$("dsCatalogLoad").addEventListener("click", loadCatalog);

$("dsHfDownload").addEventListener("click", async () => {
  clearErr("errDsCatalog");
  try {
    const repo = reqStr("dsHfRepo", "HuggingFace repo");
    const file = reqStr("dsHfFile", "File");
    const hfToken = $("dsHfToken").value.trim() || null;
    $("dsHfDownload").disabled = true; toast("Downloading from HuggingFace…");
    const res = await invoke("download_hf_file", { repo, file, revision: null, destDir: destDir(), hfToken });
    afterDownload(res);
  } catch (e) { setErr("errDsCatalog", String(e)); }
  finally { $("dsHfDownload").disabled = false; }
});

$("dsKgDownload").addEventListener("click", async () => {
  clearErr("errDsCatalog");
  try {
    const ownerSlug = reqStr("dsKgRepo", "Kaggle owner/slug");
    const file = reqStr("dsKgFile", "File");
    $("dsKgDownload").disabled = true; toast("Downloading from Kaggle…");
    const res = await invoke("download_kaggle_file", { ownerSlug, file, destDir: destDir() });
    afterDownload(res);
  } catch (e) { setErr("errDsCatalog", String(e)); }
  finally { $("dsKgDownload").disabled = false; }
});

// Populate the catalog on startup (best-effort).
loadCatalog();

$("dsClean").addEventListener("click", async () => {
  clearErr("errDsClean");
  try {
    const raw = $("dsRaw").value;
    if (!raw.trim()) throw new Error("No raw text — download, load, or paste first");
    const opts = {
      stripGutenberg: $("optGutenberg").checked,
      lowercase: $("optLower").checked,
      collapseWhitespace: $("optCollapse").checked,
      normalizePunctuation: $("optPunct").checked,
      stripControlChars: $("optControl").checked,
      maxChars: Math.max(0, int("optMaxChars") || 0),
    };
    const r = await invoke("clean_text", { raw, opts });
    cleanedCorpus = r.cleaned;
    $("dsPreview").value = r.preview;
    $("dsStats").innerHTML =
      `chars <b>${r.chars.toLocaleString()}</b> · words <b>${r.words.toLocaleString()}</b> · ` +
      `lines <b>${r.lines.toLocaleString()}</b> · bytes <b>${r.bytes.toLocaleString()}</b> · ` +
      `char-vocab <b>${r.uniqueChars}</b>`;
    toast("Cleaned. char-level vocab = " + r.uniqueChars, "ok");
  } catch (e) { setErr("errDsClean", String(e)); }
});

$("dsSave").addEventListener("click", async () => {
  clearErr("errDsSave");
  try {
    if (!cleanedCorpus) throw new Error("Clean a corpus first");
    const path = reqStr("dsOut", "Output path");
    const n = await invoke("save_corpus", { text: cleanedCorpus, path });
    toast(`Saved ${n.toLocaleString()} bytes → ${path}`, "ok");
    // Convenience: prefill the Train corpus field.
    if (!$("trCorpus").value) $("trCorpus").value = path;
  } catch (e) { setErr("errDsSave", String(e)); }
});

// ── Train ─────────────────────────────────────────────────────────────────────
function applyMethodVisibility() {
  const m = $("trMethod").value;
  bySel("[data-fields]").forEach((el) => {
    el.style.display = el.dataset.fields.split(" ").includes(m) ? "" : "none";
  });
}
$("trMethod").addEventListener("change", applyMethodVisibility);
applyMethodVisibility();

let lossHistory = [];
function drawChart() {
  const c = $("trChart"), ctx = c.getContext("2d");
  const w = c.width, h = c.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#1e293b"; ctx.fillRect(0, 0, w, h);
  if (lossHistory.length < 2) return;
  const max = Math.max(...lossHistory), min = Math.min(...lossHistory);
  const pad = 8, span = (max - min) || 1;
  ctx.strokeStyle = "#ea580c"; ctx.lineWidth = 2; ctx.beginPath();
  lossHistory.forEach((v, i) => {
    const x = pad + (i / (lossHistory.length - 1)) * (w - 2 * pad);
    const y = pad + (1 - (v - min) / span) * (h - 2 * pad);
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.stroke();
  ctx.fillStyle = "#94a3b8"; ctx.font = "11px monospace";
  ctx.fillText("loss " + max.toFixed(4), pad, 14);
  ctx.fillText(min.toFixed(4), pad, h - 4);
}

$("trStart").addEventListener("click", async () => {
  clearErr("errTrain");
  let params;
  try {
    const method = $("trMethod").value;
    params = {
      method,
      corpusPath: reqStr("trCorpus", "Corpus"),
      modelPath: reqStr("trModel", "Model output"),
      contextLen: reqInt("trContext", "Context"),
      embedDim: reqInt("trEmbed", "Embed dim"),
      numHeads: reqInt("trHeads", "Heads"),
      numBlocks: reqInt("trBlocks", "Blocks"),
      hiddenDim: reqInt("trHidden", "Hidden"),
      epochs: reqInt("trEpochs", "Epochs"),
      lr: reqNum("trLr", "Learning rate", 0),
      momentum: Math.max(0, num("trMomentum") || 0),
      batchSize: reqInt("trBatch", "Batch"),
      vocabSize: Math.max(0, int("trVocab") || 0),
      seed: Math.max(0, int("trSeed") || 0),
      threads: Math.max(0, int("trThreads") || 0),
      verbose: $("trVerbose").checked,
    };
    // Client-side guards mirroring the backend, for instant feedback.
    if (method === "transformer") {
      if (params.embedDim % params.numHeads !== 0)
        throw new Error(`Embed dim (${params.embedDim}) must be divisible by heads (${params.numHeads})`);
      if (params.vocabSize !== 0 && params.vocabSize < 256)
        throw new Error("Vocab must be 0 (char-level) or ≥ 256 (BPE)");
    }
    if (method === "embedded" && params.vocabSize !== 0 && params.vocabSize < 256)
      throw new Error("Vocab must be 0 (char-level) or ≥ 256 (BPE)");
  } catch (e) { setErr("errTrain", String(e)); return; }

  lossHistory = []; drawChart();
  $("trBar").style.width = "0%";
  $("trResult").textContent = "";
  $("trStatus").textContent = "training…";
  $("trStart").disabled = true;
  try {
    const r = await invoke("train_slm", { params });
    $("trStatus").textContent = "done";
    $("trResult").textContent =
      `Saved ${r.bytes.toLocaleString()} bytes → ${r.modelPath}\n` +
      `method=${r.method}  final loss=${r.finalLoss.toFixed(6)}  time=${r.seconds.toFixed(2)}s\n` +
      `input_dim=${r.inputDim}  output_dim=${r.outputDim}  tokenizer=${r.tokenizer}  layers=${r.layers}`;
    toast("Training complete", "ok");
    if (!$("genModel").value) $("genModel").value = r.modelPath;
    if (!$("evModel").value) $("evModel").value = r.modelPath;
    if (!$("miPath").value) $("miPath").value = r.modelPath;
  } catch (e) {
    $("trStatus").textContent = "error";
    setErr("errTrain", String(e));
    toast("Training failed: " + e, "error");
  } finally { $("trStart").disabled = false; }
});

// ── Generate ──────────────────────────────────────────────────────────────────
let generating = false;
$("genStart").addEventListener("click", async () => {
  clearErr("errGen");
  let params;
  try {
    const seedNum = $("genSeedNum").value.trim();
    params = {
      modelPath: reqStr("genModel", "Model"),
      seedText: $("genSeed").value,
      numChars: reqInt("genChars", "Chars"),
      temp: reqNum("genTemp", "Temperature", 0),
      genSeed: seedNum === "" ? null : parseInt(seedNum, 10),
      stream: $("genStream").checked,
      verbose: $("genVerbose").checked,
    };
  } catch (e) { setErr("errGen", String(e)); return; }

  $("genOut").textContent = "";
  $("genStatus").textContent = "generating…";
  $("genStart").disabled = true;
  generating = params.stream;
  try {
    const out = await invoke("generate_slm", { params });
    // Stop accepting late gen-fragment events *before* reconciling, then set the
    // display from the authoritative return value so a dropped/late stream tail
    // can never leave the output truncated (G1).
    generating = false;
    if (params.stream) {
      $("genOut").textContent = streamContinuation(out, params.seedText);
    } else {
      $("genOut").textContent = out;
    }
    $("genStatus").textContent = `done (${out.length} chars)`;
  } catch (e) {
    $("genStatus").textContent = "error";
    setErr("errGen", String(e));
    toast("Generation failed: " + e, "error");
  } finally { generating = false; $("genStart").disabled = false; }
});

// ── Evaluate ──────────────────────────────────────────────────────────────────
$("evRun").addEventListener("click", async () => {
  clearErr("errEval");
  let params;
  try {
    params = {
      modelPath: reqStr("evModel", "Model"),
      text: $("evText").value.trim() || null,
      textPath: $("evFile").value.trim() || null,
    };
    if (!params.text && !params.textPath) throw new Error("Provide held-out text or a file");
  } catch (e) { setErr("errEval", String(e)); return; }

  $("evStatus").textContent = "evaluating…"; $("evRun").disabled = true;
  try {
    const r = await invoke("evaluate_slm", { params });
    const good = r.perplexity < r.uniformBaseline * 0.5;
    const tr = document.createElement("tr");
    tr.innerHTML =
      `<td>${r.model}</td><td>${r.numPredictions.toLocaleString()}</td>` +
      `<td>${r.crossEntropy.toFixed(4)}</td><td>${r.bitsPerToken.toFixed(4)}</td>` +
      `<td><b>${r.perplexity.toFixed(4)}</b></td><td>${r.uniformBaseline.toFixed(0)}</td>` +
      `<td class="${good ? "verdict-good" : "verdict-weak"}">${good ? "learned" : "weak"}</td>`;
    $("evTable").querySelector("tbody").appendChild(tr);
    $("evStatus").textContent = "added";
    toast(`Perplexity ${r.perplexity.toFixed(3)}`, "ok");
  } catch (e) {
    $("evStatus").textContent = "error";
    setErr("errEval", String(e));
    toast("Evaluation failed: " + e, "error");
  } finally { $("evRun").disabled = false; }
});
$("evClear").addEventListener("click", () => { $("evTable").querySelector("tbody").innerHTML = ""; });

// ── Models ────────────────────────────────────────────────────────────────────
$("miInspect").addEventListener("click", async () => {
  clearErr("errModel");
  let path;
  try { path = reqStr("miPath", "Model file"); } catch (e) { setErr("errModel", String(e)); return; }
  try {
    const i = await invoke("model_info", { path });
    const card = (k, v) => `<div class="card"><div class="k">${k}</div><div class="v">${v}</div></div>`;
    $("miInfo").innerHTML =
      card("File", i.path) + card("Size", i.bytes.toLocaleString() + " bytes") +
      card("Format", i.format) + card("Name", i.name) + card("Task", i.task) +
      card("Input dim", i.inputDim) + card("Output dim", i.outputDim) +
      card("Tokenizer", i.tokenizer) + card("BPE merges", i.merges) + card("Layers", i.layers);
    toast("Model loaded", "ok");
  } catch (e) { setErr("errModel", String(e)); toast("Inspect failed: " + e, "error"); }
});

// ── GGUF (import & run Llama/Qwen) ──────────────────────────────────────────────

// Compare a GGUF's parameter count against the cached capability bounds.
// Returns a list of human-readable descriptions for every bound it exceeds.
function checkGgufBudget(paramCount) {
  if (!capBounds || !Number.isFinite(paramCount) || paramCount <= 0) return [];
  const checks = [
    ["loading at int4 (fits in RAM)", capBounds.loadInt4],
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

// Show the warning dialog; resolves true if the user chooses Proceed, and
// false on Cancel or any dismissal (Esc/backdrop close the native <dialog>,
// which fires its "close" event — the single resolution point).
function confirmGgufWarning(paramCount, crossed) {
  return new Promise((resolve) => {
    // Keyed on the load-check label in checkGgufBudget: crossing that line
    // means "may not fit in RAM", which is a different risk than "slow".
    const footer = crossed.some((c) => c.includes("fits in RAM"))
      ? "You can still proceed — but this exceeds the fits-in-RAM ceiling: expect heavy swapping or a failed load."
      : "You can still proceed — expect slower than the targets above.";
    $("ggWarnBody").innerHTML =
      `<p>This model has <strong>${fmtParams(paramCount)}</strong> parameters, which exceeds
       the estimated limits of this machine:</p><ul>` +
      crossed.map((c) => `<li>${c}</li>`).join("") +
      `</ul><p class="muted">${footer}</p>`;
    const dlg = $("ggWarnDialog");
    let proceed = false; // default for Esc / backdrop dismissal
    const onProceed = () => { proceed = true; dlg.close(); };
    const onCancel = () => { proceed = false; dlg.close(); };
    const onClose = () => {
      $("ggWarnProceed").removeEventListener("click", onProceed);
      $("ggWarnCancel").removeEventListener("click", onCancel);
      resolve(proceed);
    };
    $("ggWarnProceed").addEventListener("click", onProceed);
    $("ggWarnCancel").addEventListener("click", onCancel);
    dlg.addEventListener("close", onClose, { once: true });
    dlg.showModal();
  });
}

$("ggInspect").addEventListener("click", async () => {
  clearErr("errGguf");
  let path;
  try { path = reqStr("ggPath", "GGUF file"); } catch (e) { setErr("errGguf", String(e)); return; }
  $("ggStatus").textContent = "inspecting…";
  try {
    const i = await invoke("gguf_info", { path });
    const card = (k, v) => `<div class="card"><div class="k">${k}</div><div class="v">${v}</div></div>`;
    const avail = i.availMb == null ? "unknown" : (i.availMb / 1024).toFixed(2) + " GB";
    $("ggInfo").innerHTML =
      card("Architecture", i.architecture + (i.runnable ? " ✓" : " ✗ not runnable")) +
      card("GGUF", "v" + i.version + " · " + i.numTensors + " tensors") +
      card("Shape", `dim ${i.modelDim} · ${i.nLayers}L · ${i.nHeads}h/${i.nKvHeads}kv · vocab ${i.vocabSize}`) +
      card("Tokenizer", i.tokenizer) +
      card("Resident (int4/int8/f32)", `${(i.estInt4Mb/1024).toFixed(2)} / ${(i.estInt8Mb/1024).toFixed(2)} / ${(i.estF32Mb/1024).toFixed(2)} GB`) +
      card("Available RAM", avail) +
      card("Note", i.note);
    $("ggStatus").textContent = "";
    toast(i.runnable ? "GGUF inspected" : "Inspected — architecture not runnable", i.runnable ? "ok" : "error");
    const crossedI = checkGgufBudget(i.paramCount);
    if (crossedI.length) {
      await confirmGgufWarning(i.paramCount, crossedI); // advisory on inspect; result ignored
    } else if (!capBounds) {
      toast("Tip: run the Capable check to flag oversized models.", "info");
    }
  } catch (e) { $("ggStatus").textContent = "error"; setErr("errGguf", String(e)); toast("Inspect failed: " + e, "error"); }
});

$("ggRun").addEventListener("click", async () => {
  clearErr("errGguf");
  let params;
  try {
    const seed = $("ggSeed").value.trim();
    const ids = $("ggIds").value.trim();
    const resume = $("ggResume").value.trim();
    params = {
      modelPath: reqStr("ggPath", "GGUF file"),
      prompt: $("ggPrompt").value,
      quant: $("ggQuant").value,
      maxNew: reqInt("ggMax", "Max new tokens"),
      temp: reqNum("ggTemp", "Temperature", 0),
      genSeed: seed === "" ? null : parseInt(seed, 10),
      ids: ids === "" ? null : ids,
      force: $("ggForce").checked,
      resume: resume === "" ? null : resume,
    };
    if (!params.prompt.trim() && !params.ids) throw new Error("Enter a prompt (or token IDs)");
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
  $("ggStatus").textContent = "loading & generating… (CPU: this can take a while)";
  $("ggRun").disabled = true;
  try {
    const r = await invoke("run_gguf", { params });
    $("ggOut").textContent = r.text;
    $("ggStatus").textContent =
      `done — ${r.generated} tokens in ${r.seconds.toFixed(1)}s (${r.tokensPerSec.toFixed(1)} tok/s, prompt ${r.promptTokens} tok)`;
  } catch (e) {
    $("ggStatus").textContent = "error";
    setErr("errGguf", String(e));
    toast("Run failed: " + e, "error");
  } finally { $("ggRun").disabled = false; }
});

// ── Fine-tune (GGUF) ────────────────────────────────────────────────────────────
let ftLossHistory = [];
function drawFtChart() {
  const c = $("ftChart"), ctx = c.getContext("2d");
  const w = c.width, h = c.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#1e293b"; ctx.fillRect(0, 0, w, h);
  if (ftLossHistory.length < 2) return;
  const max = Math.max(...ftLossHistory), min = Math.min(...ftLossHistory);
  const pad = 8, span = (max - min) || 1;
  ctx.strokeStyle = "#ea580c"; ctx.lineWidth = 2; ctx.beginPath();
  ftLossHistory.forEach((v, i) => {
    const x = pad + (i / (ftLossHistory.length - 1)) * (w - 2 * pad);
    const y = pad + (1 - (v - min) / span) * (h - 2 * pad);
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.stroke();
  ctx.fillStyle = "#94a3b8"; ctx.font = "11px monospace";
  ctx.fillText("loss " + max.toFixed(4), pad, 14);
  ctx.fillText(min.toFixed(4), pad, h - 4);
}

$("ftRun").addEventListener("click", async () => {
  clearErr("errFt");
  let params;
  try {
    const resume = $("ftResume").value.trim();
    params = {
      modelPath: reqStr("ftPath", "GGUF file"),
      corpusPath: reqStr("ftCorpus", "Training corpus"),
      outPath: reqStr("ftOut", "Output checkpoint"),
      epochs: reqInt("ftEpochs", "Epochs"),
      lr: reqNum("ftLr", "Learning rate", 0),
      batchSize: reqInt("ftBatch", "Batch size"),
      seqLen: reqInt("ftSeq", "Sequence length"),
      warmup: Math.max(0, int("ftWarmup") || 0),
      clip: Math.max(0, num("ftClip") || 0),
      weightDecay: Math.max(0, num("ftWd") || 0),
      dropout: Math.max(0, num("ftDropout") || 0),
      qat: $("ftQat").checked,
      seed: Math.max(0, int("ftSeed") || 0),
      threads: Math.max(0, int("ftThreads") || 0),
      resume: resume === "" ? null : resume,
      force: $("ftForce").checked,
      verbose: false,
    };
  } catch (e) { setErr("errFt", String(e)); return; }

  ftLossHistory = []; drawFtChart();
  $("ftBar").style.width = "0%";
  $("ftResult").textContent = "";
  $("ftStatus").textContent = "fine-tuning… (CPU: this can take a while)";
  $("ftRun").disabled = true;
  try {
    const r = await invoke("finetune_gguf", { params });
    $("ftStatus").textContent = "done";
    $("ftResult").textContent =
      `Saved ${r.bytes.toLocaleString()} bytes → ${r.outPath}\n` +
      `epochs=${r.epochs}  first loss=${r.firstLoss.toFixed(6)}  final loss=${r.finalLoss.toFixed(6)}\n` +
      `tokens=${r.tokens.toLocaleString()}  steps=${r.steps}  time=${r.seconds.toFixed(2)}s\n` +
      `Run it: GGUF tab → set "Fine-tune checkpoint" to ${r.outPath}`;
    toast("Fine-tune complete", "ok");
    if (!$("ggPath").value) $("ggPath").value = params.modelPath;
    if (!$("ggResume").value) $("ggResume").value = r.outPath;
  } catch (e) {
    $("ftStatus").textContent = "error";
    setErr("errFt", String(e));
    toast("Fine-tune failed: " + e, "error");
  } finally { $("ftRun").disabled = false; }
});

// ── Export (write a GGUF back out) ──────────────────────────────────────────────
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

// ── Tabular (train_cli via shell) ──────────────────────────────────────────────
$("tbRun").addEventListener("click", async () => {
  clearErr("errTab");
  let cmd;
  try {
    const bin = reqStr("tbBin", "train_cli binary");
    const csv = reqStr("tbCsv", "CSV");
    const model = reqStr("tbModel", "Model output");
    const name = reqStr("tbName", "Dataset name");
    const hidden = reqInt("tbHidden", "Hidden");
    const epochs = reqInt("tbEpochs", "Epochs");
    const q = (s) => `"${s.replace(/"/g, '\\"')}"`;
    cmd = `${q(bin)} ${q(csv)} ${q(model)} ${q(name)} ${hidden} ${epochs}`;
  } catch (e) { setErr("errTab", String(e)); return; }
  // Make sure the terminal is visible and run there (streams output).
  toast("Running train_cli — see terminal", "info");
  await runTerminal(cmd);
});

// ── System monitor ──────────────────────────────────────────────────────────
let sysTimer = null;
function fmtBytes(b) {
  const u = ["B", "KB", "MB", "GB", "TB"]; let i = 0; let n = b;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return n.toFixed(1) + " " + u[i];
}
async function pollStats() {
  if (!hasTauri) return;
  try {
    const s = await invoke("system_stats");
    $("miniCpu").textContent = s.cpuTotal.toFixed(0) + "%";
    $("miniMem").textContent = s.memPercent.toFixed(0) + "%";
    $("miniThreads").textContent = s.ferrumThreads;
    $("sysCpu").textContent = s.cpuTotal.toFixed(1) + "%";
    $("sysCpuBar").style.width = Math.min(100, s.cpuTotal) + "%";
    $("sysMem").textContent = s.memPercent.toFixed(1) + "%";
    $("sysMemBar").style.width = Math.min(100, s.memPercent) + "%";
    $("sysMemAbs").textContent = `${fmtBytes(s.memUsed)} / ${fmtBytes(s.memTotal)}`;
    $("sysThreads").textContent = s.ferrumThreads;
    $("sysCores").innerHTML = s.perCore.map((v, i) =>
      `<div class="core">core ${i} — ${v.toFixed(0)}%<div class="cbar"><div style="width:${Math.min(100, v)}%"></div></div></div>`
    ).join("");
  } catch (e) { /* transient; ignore */ }
}
function restartMonitor() {
  if (sysTimer) clearInterval(sysTimer);
  if (!$("sysOn").checked) return;
  const sec = Math.max(0.5, num("sysInterval") || 1.5);
  pollStats();
  sysTimer = setInterval(pollStats, sec * 1000);
}
$("sysOn").addEventListener("change", restartMonitor);
$("sysInterval").addEventListener("change", restartMonitor);

// ── Capable (machine capability) ─────────────────────────────────────────────
let capBounds = null; // {inferInt4/8/F32, trainChinchilla, trainFixed1b, testEval, loadInt4, finetuneHi}

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
    loadInt4: r.loadInt4, finetuneHi: r.finetuneHi,
  };
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
              `Chinchilla ${r.chinchillaRatio}× vs fixed ${fmtParams(r.fixedTrainTokens)}-token corpus; < ${r.trainHours} h, RAM-capped @16 B/param`)}
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

// ── Backend event wiring ──────────────────────────────────────────────────────
async function wireEvents() {
  await listen("engine-log", (e) => termLine(e.payload, "log"));
  await listen("term-output", (e) => termLine(e.payload.line, e.payload.stream === "stderr" ? "stderr" : ""));
  await listen("train-progress", (e) => {
    const p = e.payload;
    const pct = Math.round((p.epoch / p.total) * 100);
    $("trBar").style.width = pct + "%";
    $("trStatus").textContent = `epoch ${p.epoch}/${p.total} — loss ${p.loss.toFixed(6)}`;
    lossHistory.push(p.loss);
    if (lossHistory.length > 1000) lossHistory.shift();
    drawChart();
  });
  await listen("finetune-progress", (e) => {
    const p = e.payload;
    const pct = Math.round((p.epoch / p.total) * 100);
    $("ftBar").style.width = pct + "%";
    $("ftStatus").textContent = `epoch ${p.epoch}/${p.total} — loss ${p.loss.toFixed(4)} (ppl ${p.ppl.toFixed(2)})`;
    ftLossHistory.push(p.loss);
    if (ftLossHistory.length > 1000) ftLossHistory.shift();
    drawFtChart();
  });
  await listen("export-progress", (e) => { $("exStatus").textContent = e.payload; });
  await listen("gen-fragment", (e) => { if (generating) $("genOut").textContent += e.payload; });
}

// ── Init ──────────────────────────────────────────────────────────────────────
(async function init() {
  if (!hasTauri) {
    $("webBanner").classList.remove("hidden");
    $("connDot").classList.remove("ok");
    $("connDot").title = "Backend not connected (running outside Tauri)";
    termLine("Ferrum SLM Studio — backend unavailable (open via `cargo tauri dev`).", "sys");
    return;
  }
  $("connDot").classList.add("ok");
  $("connDot").title = "Backend connected";
  await wireEvents();
  await refreshPrompt();
  restartMonitor();
  termLine("Ferrum SLM Studio ready. Verbose engine logs and shell output appear here.", "sys");
})();
