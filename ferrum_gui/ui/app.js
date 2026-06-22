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
bySel(".browse").forEach((b) => b.addEventListener("click", () => pickOpen(b.dataset.target)));
bySel(".browse-save").forEach((b) => b.addEventListener("click", () => pickSave(b.dataset.target)));

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
