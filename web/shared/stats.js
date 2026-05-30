/**
 * stats.js — Live statistical terminals for the Tabular ML demo.
 *
 * Exports one function:
 *   renderStats(pane1El, pane2El, meta, norm, rawValues, result)
 *
 * Both panes update every time the user moves a slider.
 * All statistics are computed from data already embedded in the model file
 * (feature_ranges, class_names, normalizer means/stds) — no extra fetch needed.
 */

// ─── Formatting helpers ────────────────────────────────────────────────────────

function fmtNum(v, decimals = 3) {
  if (!isFinite(v)) return '—';
  const abs = Math.abs(v);
  if (abs === 0) return '0';
  if (abs >= 1e6)  return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
  if (abs >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 1 });
  if (abs >= 100)  return v.toFixed(1);
  if (abs >= 10)   return v.toFixed(2);
  return v.toFixed(decimals);
}

function fmtPct(v) { return (v * 100).toFixed(1) + '%'; }
function fmtZ(z)   {
  const s = z >= 0 ? '+' : '';
  return `${s}${z.toFixed(3)}σ`;
}

function badge(text, cls) {
  return `<span class="stat-badge stat-badge--${cls}">${text}</span>`;
}

// ─── Statistical calculations ─────────────────────────────────────────────────

/** Shannon entropy in nats of a probability distribution. */
function entropy(probs) {
  return -probs.reduce((s, p) => s + (p > 1e-12 ? p * Math.log(p) : 0), 0);
}

/** Max possible entropy for C classes (uniform). */
function maxEntropy(c) { return Math.log(c); }

/** "Certainty" ∈ [0,1]: 1 = certain, 0 = maximally confused. */
function certainty(probs) {
  const H  = entropy(probs);
  const Hm = maxEntropy(probs.length);
  return Hm > 0 ? 1 - H / Hm : 1;
}

/** Confidence label from certainty score. */
function confidenceLabel(cert) {
  if (cert > 0.90) return ['Certain',    'green'];
  if (cert > 0.70) return ['Confident',  'blue'];
  if (cert > 0.45) return ['Uncertain',  'yellow'];
  return                  ['Toss-up',    'red'];
}

/** Odds ratio p/(1-p), clamped. */
function oddsRatio(p) {
  const eps = 1e-7;
  return (p + eps) / (1 - p + eps);
}

/** z-score of a raw value given mean and std. */
function zScore(v, mean, std) {
  return std > 0 ? (v - mean) / std : 0;
}

/** Percentile position within [lo, hi], clamped to [0,1]. */
function rangePct(v, lo, hi) {
  if (hi <= lo) return 0.5;
  return Math.min(1, Math.max(0, (v - lo) / (hi - lo)));
}

/** A colour on a green→yellow→red gradient by a certainty value. */
function certColor(cert) {
  // cert: 1=green, 0.5=yellow, 0=red
  const r = Math.round(cert < 0.5 ? 255 : 255 * (1 - cert) * 2);
  const g = Math.round(cert > 0.5 ? 100 : 100 *  cert      * 2);
  return `rgb(${r},${g},60)`;
}

// ─── Spark-bar helper ─────────────────────────────────────────────────────────

/** A slim inline bar, pct 0-100. */
function sparkBar(pct, colorVar = '--accent', width = 80) {
  const filled = Math.round(Math.min(100, Math.max(0, pct)));
  return `<span class="spark-wrap" style="width:${width}px">` +
         `<span class="spark-fill" style="width:${filled}%;background:var(${colorVar})"></span>` +
         `</span>`;
}

/** A centred bar (negative left, positive right). */
function centredBar(z, maxZ = 3, width = 80) {
  const pct = Math.min(1, Math.abs(z) / maxZ) * 50;
  const isPos = z >= 0;
  return `<span class="spark-wrap" style="width:${width}px;position:relative">` +
         `<span class="spark-center-line"></span>` +
         `<span class="spark-fill spark-fill--centred" ` +
         `style="width:${pct.toFixed(1)}%;` +
         `${isPos ? 'left:50%' : `right:50%;left:${(50 - pct).toFixed(1)}%`};` +
         `background:${isPos ? 'var(--c1)' : 'var(--c2)'}"></span>` +
         `</span>`;
}

// ─── Pane 1: Model Statistics (input vector + architecture) ────────────────────

function renderPane1(el, meta, norm, rawValues) {
  const feats = meta.feature_names;
  const nf    = feats.length;

  // Per-feature rows: name | value | z-score bar | range bar
  const rows = feats.map((name, i) => {
    const v   = rawValues[i] ?? 0;
    const lo  = meta.feature_ranges[i]?.[0] ?? 0;
    const hi  = meta.feature_ranges[i]?.[1] ?? 1;
    const μ   = norm.means[i] ?? 0;
    const σ   = norm.stds[i]  ?? 1;
    const z   = zScore(v, μ, σ);
    const rp  = rangePct(v, lo, hi);
    const zClamped = Math.min(3, Math.max(-3, z));

    const zColor = Math.abs(z) > 2 ? 'var(--c2)' : Math.abs(z) > 1 ? 'var(--c3)' : 'var(--c1)';

    return `<tr class="st-row">
      <td class="st-name" title="${name}">${name.replace(/_/g,' ').substring(0,16)}</td>
      <td class="st-val">${fmtNum(v)}</td>
      <td class="st-z" style="color:${zColor}">${fmtZ(z)}</td>
      <td class="st-bar">${sparkBar(rp * 100, '--accent', 64)}</td>
      <td class="st-zbar">${centredBar(zClamped, 3, 64)}</td>
    </tr>`;
  }).join('');

  // Architecture card
  const archRows = [
    ['Input dim',  nf],
    ['Hidden',     '— (32–64 ReLU)'],
    ['Output dim', meta.output_dim],
    ['Task',       meta.task === 'regression' ? 'Regression (MSE)' : `Classification (Softmax)`],
    ['Normaliser', 'Z-score per feature'],
    ['Format',     'FINF v3 binary'],
  ].map(([k,v]) =>
    `<tr><td class="arch-key">${k}</td><td class="arch-val">${v}</td></tr>`
  ).join('');

  el.innerHTML = `
    <div class="term-header">
      <span class="term-dot term-dot--green"></span>
      <span class="term-dot term-dot--yellow"></span>
      <span class="term-dot term-dot--red"></span>
      <span class="term-title">MODEL STATISTICS</span>
    </div>
    <div class="term-body">
      <div class="term-section-label">INPUT VECTOR  (${nf} features)</div>
      <div class="st-scroll">
        <table class="st-table">
          <thead>
            <tr>
              <th>Feature</th><th>Value</th><th>Z-score</th>
              <th>Range%</th><th>Z-bar</th>
            </tr>
          </thead>
          <tbody>${rows}</tbody>
        </table>
      </div>

      <div class="term-section-label" style="margin-top:.75rem">ARCHITECTURE</div>
      <table class="arch-table">
        <tbody>${archRows}</tbody>
      </table>
    </div>`;
}

// ─── Pane 2: Quantitative Report (prediction analysis) ─────────────────────────

function renderPane2Classification(el, meta, result, norm) {
  const probs = result.probabilities;
  const nC    = probs.length;
  const H     = entropy(probs);
  const Hmax  = maxEntropy(nC);
  const cert  = certainty(probs);
  const [confLabel, confCls] = confidenceLabel(cert);

  // Sort classes by probability descending for the table
  const sorted = meta.class_names
    .map((cn, i) => ({ cn, p: probs[i] ?? 0, i }))
    .sort((a, b) => b.p - a.p);

  const winner = sorted[0];
  const margin = sorted.length > 1 ? winner.p - sorted[1].p : winner.p;

  const tableRows = sorted.map(({ cn, p, i }) => {
    const lp   = p > 1e-12 ? Math.log(p) : -Infinity;
    const odds = oddsRatio(p);
    const isTop = i === result.class_index;
    return `<tr class="${isTop ? 'qr-top-row' : ''}">
      <td class="qr-class">${cn}</td>
      <td class="qr-prob">${fmtPct(p)}</td>
      <td class="qr-bar">${sparkBar(p * 100, isTop ? '--c1' : '--muted', 72)}</td>
      <td class="qr-logp">${isFinite(lp) ? lp.toFixed(3) : '−∞'}</td>
      <td class="qr-odds">${odds > 1000 ? '>1000' : odds.toFixed(2)}:1</td>
    </tr>`;
  }).join('');

  // Entropy gauge
  const entropyPct = Hmax > 0 ? (H / Hmax) * 100 : 0;
  const marginPct  = margin * 100;

  // Feature-level contributions (z-score magnitude as proxy for information)
  const contributions = (norm.means || []).slice(0, meta.input_dim).map((_, i) => ({
    name: meta.feature_names[i] || `f${i}`,
    weight: Math.abs(result._z_scores?.[i] ?? 0),
  })).sort((a, b) => b.weight - a.weight).slice(0, 5);

  el.innerHTML = `
    <div class="term-header">
      <span class="term-dot term-dot--green"></span>
      <span class="term-dot term-dot--yellow"></span>
      <span class="term-dot term-dot--red"></span>
      <span class="term-title">CLASSIFICATION REPORT</span>
    </div>
    <div class="term-body">

      <div class="term-section-label">PREDICTION SUMMARY</div>
      <div class="qr-summary-grid">
        <div class="qr-summary-cell">
          <div class="qr-cell-label">Predicted Class</div>
          <div class="qr-cell-value qr-cell-value--big">${winner.cn}</div>
        </div>
        <div class="qr-summary-cell">
          <div class="qr-cell-label">Confidence</div>
          <div class="qr-cell-value">${badge(confLabel, confCls)}</div>
        </div>
        <div class="qr-summary-cell">
          <div class="qr-cell-label">Certainty Score</div>
          <div class="qr-cell-value">${(cert * 100).toFixed(1)}%</div>
        </div>
        <div class="qr-summary-cell">
          <div class="qr-cell-label">Top-2 Margin</div>
          <div class="qr-cell-value">${fmtPct(margin)}</div>
        </div>
      </div>

      <div class="term-section-label" style="margin-top:.65rem">PROBABILITY DISTRIBUTION</div>
      <table class="qr-table">
        <thead>
          <tr><th>Class</th><th>P</th><th></th><th>ln P</th><th>Odds</th></tr>
        </thead>
        <tbody>${tableRows}</tbody>
      </table>

      <div class="term-section-label" style="margin-top:.65rem">ENTROPY ANALYSIS</div>
      <div class="entropy-row">
        <span class="entropy-label">H(p)</span>
        <span class="entropy-val">${H.toFixed(4)} nats</span>
        <div class="entropy-bar-wrap">
          <div class="entropy-bar-fill" style="width:${entropyPct.toFixed(1)}%;background:${certColor(cert)}"></div>
        </div>
        <span class="entropy-max">/ ${Hmax.toFixed(3)}</span>
      </div>
      <div class="entropy-row" style="margin-top:.3rem">
        <span class="entropy-label">Margin</span>
        <span class="entropy-val">${fmtPct(margin)}</span>
        <div class="entropy-bar-wrap">
          <div class="entropy-bar-fill" style="width:${marginPct.toFixed(1)}%;background:var(--c1)"></div>
        </div>
        <span class="entropy-max">/ 100%</span>
      </div>

      <div class="qr-note">
        H=0 nats: model is certain. H=${Hmax.toFixed(2)} nats: model is maximally uncertain (uniform over ${nC} classes).
      </div>
    </div>`;
}

function renderPane2Regression(el, meta, result, norm) {
  const predRaw  = result.value;
  const predNorm = result.value_norm;
  const [tLo, tHi] = Array.isArray(meta.target_range) ? meta.target_range : [0, 1];
  const tMean = norm.means[norm.means.length - 1] ?? ((tLo + tHi) / 2);
  const tStd  = norm.stds[norm.stds.length - 1]  ?? ((tHi - tLo) / 4);

  const tZ       = zScore(predRaw, tMean, tStd);
  const tPct     = rangePct(predRaw, tLo, tHi) * 100;
  const abovePct = ((predRaw - tMean) / tMean) * 100;
  const ci68Lo   = predRaw - tStd;
  const ci68Hi   = predRaw + tStd;

  // Quartile estimate
  let quartile;
  const q1 = tLo + (tHi - tLo) * 0.25;
  const q3 = tLo + (tHi - tLo) * 0.75;
  const median = (tLo + tHi) / 2;
  if      (predRaw < q1)     quartile = 'Q1 (low quarter)';
  else if (predRaw < median) quartile = 'Q2 (lower-mid)';
  else if (predRaw < q3)     quartile = 'Q3 (upper-mid)';
  else                       quartile = 'Q4 (high quarter)';

  const zColor = Math.abs(tZ) > 2 ? 'var(--c2)' : Math.abs(tZ) > 1 ? 'var(--c3)' : 'var(--c1)';
  const aboveColor = abovePct >= 0 ? 'var(--c1)' : 'var(--c2)';

  const statsRows = [
    ['Prediction',        `<strong>${fmtNum(predRaw)}</strong>`],
    ['Range position',    `${tPct.toFixed(1)}% of [${fmtNum(tLo)} – ${fmtNum(tHi)}]`],
    ['Z-score (targets)', `<span style="color:${zColor}">${fmtZ(tZ)}</span>`],
    ['vs. dataset mean',  `<span style="color:${aboveColor}">${abovePct >= 0 ? '+' : ''}${abovePct.toFixed(1)}% (mean: ${fmtNum(tMean)})</span>`],
    ['68% ref interval',  `${fmtNum(ci68Lo)} – ${fmtNum(ci68Hi)} (±1σ)`],
    ['Approx. quartile',  quartile],
    ['Normalised output', predNorm.toFixed(5)],
  ].map(([k, v]) =>
    `<tr><td class="arch-key">${k}</td><td class="arch-val">${v}</td></tr>`
  ).join('');

  el.innerHTML = `
    <div class="term-header">
      <span class="term-dot term-dot--green"></span>
      <span class="term-dot term-dot--yellow"></span>
      <span class="term-dot term-dot--red"></span>
      <span class="term-title">REGRESSION REPORT</span>
    </div>
    <div class="term-body">

      <div class="term-section-label">PREDICTION SUMMARY</div>
      <div class="qr-summary-grid">
        <div class="qr-summary-cell">
          <div class="qr-cell-label">Predicted Value</div>
          <div class="qr-cell-value qr-cell-value--big">${fmtNum(predRaw)}</div>
        </div>
        <div class="qr-summary-cell">
          <div class="qr-cell-label">Z-score</div>
          <div class="qr-cell-value" style="color:${zColor}">${fmtZ(tZ)}</div>
        </div>
        <div class="qr-summary-cell">
          <div class="qr-cell-label">Quartile</div>
          <div class="qr-cell-value">${quartile.split(' ')[0]}</div>
        </div>
        <div class="qr-summary-cell">
          <div class="qr-cell-label">vs. Mean</div>
          <div class="qr-cell-value" style="color:${aboveColor}">${abovePct >= 0 ? '+' : ''}${abovePct.toFixed(1)}%</div>
        </div>
      </div>

      <div class="term-section-label" style="margin-top:.65rem">TARGET SCALE POSITION</div>
      <div class="scale-row">
        <span class="scale-label">${fmtNum(tLo)}</span>
        <div class="scale-track">
          <div class="scale-mean-line" style="left:${rangePct(tMean, tLo, tHi)*100}%"
               title="Dataset mean: ${fmtNum(tMean)}"></div>
          <div class="scale-pointer" style="left:${tPct}%"></div>
        </div>
        <span class="scale-label">${fmtNum(tHi)}</span>
      </div>
      <div class="scale-legend">
        <span style="color:var(--muted);font-size:.68rem">
          ▲ = prediction&nbsp;&nbsp; | = dataset mean (${fmtNum(tMean)})
        </span>
      </div>

      <div class="term-section-label" style="margin-top:.65rem">DETAILED STATISTICS</div>
      <table class="arch-table">
        <tbody>${statsRows}</tbody>
      </table>

      <div class="qr-note">
        Z-score measures how many standard deviations the prediction lies from the
        training-set target mean. |Z| &gt; 2 indicates an unusual prediction.
      </div>
    </div>`;
}

// ─── Public API ────────────────────────────────────────────────────────────────

/**
 * Render both terminal panes.
 * @param {HTMLElement} pane1El - container for the Model Statistics terminal
 * @param {HTMLElement} pane2El - container for the Quantitative Report terminal
 * @param {object}      meta    - parsed ModelMetadata JSON
 * @param {object}      norm    - { means: number[], stds: number[] }
 * @param {number[]}    rawValues - current slider values (un-normalised)
 * @param {object}      result  - output of predict() — already parsed JSON
 */
export function renderStats(pane1El, pane2El, meta, norm, rawValues, result) {
  renderPane1(pane1El, meta, norm, rawValues);

  if (meta.task === 'classification') {
    renderPane2Classification(pane2El, meta, result, norm);
  } else {
    renderPane2Regression(pane2El, meta, result, norm);
  }
}
