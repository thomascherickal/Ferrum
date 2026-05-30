/**
 * engine.js — shared WASM loader, inference, slider builder, and display updater.
 * Import with: import { loadModel, predict, buildSliders, updateDisplay, renderStats } from '../shared/engine.js';
 */

import init, { TabularModel } from '../pkg/tabular_wasm.js';
import { renderStats } from './stats.js';

let wasmReady = false;
let initPromise = null;

async function ensureWasm() {
  if (!wasmReady) {
    if (!initPromise) initPromise = init();
    await initPromise;
    wasmReady = true;
  }
}

/**
 * Load a model from a .bin URL.
 * Returns { model, meta, norm } where norm = { means, stds } extracted from
 * the encoded normaliser string embedded in the metadata JSON.
 */
export async function loadModel(modelUrl) {
  await ensureWasm();
  const resp = await fetch(modelUrl);
  if (!resp.ok) throw new Error(`Failed to fetch ${modelUrl}: ${resp.status}`);
  const bytes = new Uint8Array(await resp.arrayBuffer());
  const model = new TabularModel(bytes);
  const meta  = JSON.parse(model.metadata());
  // The norm object is reconstructed from the FINF-embedded normaliser string
  // which is exposed via model.norm_encoded() — we expose it as a plain object.
  const norm  = decodeNorm(model.norm_encoded());
  return { model, meta, norm };
}

/**
 * Parse the "mean0,std0;mean1,std1;…" normaliser string into { means, stds }.
 */
function decodeNorm(encoded) {
  const means = [], stds = [];
  for (const token of encoded.split(';')) {
    const [m, s] = token.split(',').map(Number);
    means.push(m); stds.push(s);
  }
  return { means, stds };
}

/**
 * Run inference. Returns a parsed result object.
 */
export function predict(model, values) {
  return JSON.parse(model.predict(new Float32Array(values)));
}

/**
 * Build slider UI from metadata. Appends to `container`.
 * Returns a function `getValues()` → number[].
 */
export function buildSliders(meta, container) {
  container.innerHTML = '';
  const sliders = [];

  meta.feature_names.forEach((name, i) => {
    // Defensive: fall back to [0,1] if ranges are missing/short so a malformed
    // model degrades gracefully instead of throwing "undefined is not iterable".
    const range = Array.isArray(meta.feature_ranges?.[i]) ? meta.feature_ranges[i] : [0, 1];
    const [lo, hi] = range[0] === range[1] ? [range[0], range[0] + 1] : range;
    const mid  = (lo + hi) / 2;
    const step = Math.max((hi - lo) / 200, 0.001);

    const group = document.createElement('div');
    group.className = 'slider-group';
    group.innerHTML = `
      <label>
        <span class="feat-name">${name.replace(/_/g, ' ')}</span>
        <span class="feat-range">(${fmt(lo)} – ${fmt(hi)})</span>
      </label>
      <input type="range" min="${lo}" max="${hi}" step="${step}" value="${mid}" />
      <span class="val-label">${fmt(mid)}</span>
    `;
    const input = group.querySelector('input');
    const valEl = group.querySelector('.val-label');
    input.addEventListener('input', () => { valEl.textContent = fmt(parseFloat(input.value)); });
    container.appendChild(group);
    sliders.push(input);
  });

  return () => sliders.map(s => parseFloat(s.value));
}

/**
 * Update the main prediction display.
 */
export function updateDisplay(result, meta, predEl, confEl, barsEl) {
  if (result.type === 'classification') {
    const name = meta.class_names[result.class_index] ?? `Class ${result.class_index}`;
    predEl.textContent = name;
    confEl.textContent = `${(result.confidence * 100).toFixed(1)}% confidence`;

    barsEl.innerHTML = meta.class_names.map((cn, i) => {
      const p = result.probabilities[i] ?? 0;
      return `
        <div class="prob-row">
          <span class="prob-class">${cn}</span>
          <div class="prob-track">
            <div class="prob-fill fill-${i % 6}" style="width:${(p*100).toFixed(1)}%"></div>
          </div>
          <span class="prob-pct pct-${i % 6}">${(p*100).toFixed(0)}%</span>
        </div>`;
    }).join('');
  } else {
    predEl.textContent = fmt(result.value);
    confEl.textContent = meta.target_name || 'predicted value';
    barsEl.innerHTML = `
      <div class="reg-result">
        <div class="reg-bar-wrap">
          <div class="reg-bar" style="width:${rangePercent(result.value, meta.target_range)}%"></div>
        </div>
        <div class="reg-labels">
          <span>${fmt(meta.target_range[0])}</span>
          <span>${fmt(meta.target_range[1])}</span>
        </div>
      </div>`;
  }
}

/**
 * Update both statistical terminal panes.
 * pane1El = Model Statistics, pane2El = Quantitative Report.
 */
export { renderStats };

// ── Formatting helpers ────────────────────────────────────────────────────────
function fmt(v) {
  if (Math.abs(v) >= 100000) return v.toLocaleString(undefined, {maximumFractionDigits: 0});
  if (Math.abs(v) >= 100)   return v.toFixed(1);
  if (Math.abs(v) >= 1)     return v.toFixed(2);
  return v.toFixed(3);
}

function rangePercent(v, [lo, hi]) {
  return Math.min(100, Math.max(0, (v - lo) / (hi - lo) * 100)).toFixed(1);
}
