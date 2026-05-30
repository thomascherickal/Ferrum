# Deployment Guide — Tabular ML WASM Demo

Complete instructions for building, testing, and deploying the ten-dataset,
pure-Rust WebAssembly ML demo — from a local preview on your laptop to a
permanent HTTPS URL on GitHub Pages, Cloudflare Pages, or Netlify.

---

## Architecture overview

```
One 128 KB WASM binary  +  ten tiny model files  +  three shared JS/CSS files
         ↓                         ↓                          ↓
  tabular_wasm.wasm         datasets/*/model.bin        shared/engine.js
  (ferrum_core inside)      (FINF v3 binary format)      shared/stats.js
                            weights + normalizer          shared/style.css
                            + metadata JSON embedded
```

The browser loads `tabular_wasm.wasm` once. Each dataset page then fetches
its own `model.bin` (~1–10 KB). Both the main prediction UI and the two live
statistical terminals are built dynamically from metadata embedded in the model
file — no per-dataset JavaScript is needed.

### File map

```
tabular_ml/
├── Cargo.toml                  Cargo workspace (ferrum_core, tabular_wasm, train_cli, tests)
├── Cargo.lock
├── .github/workflows/deploy.yml  CI/CD: test → train → WASM → Pages
├── .gitignore
├── scripts/
│   ├── download_datasets.sh    Fetch + clean all 10 source CSVs
│   ├── train_all.sh            Train all 10 models → web/datasets/*/model.bin
│   └── build_wasm.sh           Compile to WASM → web/pkg/
├── ferrum_core/                Zero-dependency ML engine (inference + training)
│   └── src/  (12 modules)
├── tabular_wasm/               WASM bindings — TabularModel { predict, metadata, norm_encoded }
├── train_cli/                  Generic CSV trainer (auto-detects classification/regression)
├── tests/                      131 tests: unit + integration across all 10 datasets
├── *.csv / iris.data           10 cleaned training datasets
└── web/                        Static site — serve this directory
    ├── index.html              Landing page (10 dataset cards, 2 sections)
    ├── shared/
    │   ├── engine.js           WASM loader, predict(), buildSliders(), updateDisplay()
    │   ├── stats.js            Live statistical terminals (392 lines of analysis)
    │   └── style.css           Dark-theme CSS (terminal chrome, spark bars, badges)
    ├── pkg/
    │   ├── tabular_wasm_bg.wasm  Compiled ML engine (~128 KB)
    │   └── tabular_wasm.js       wasm-bindgen JS glue (~8 KB)
    └── datasets/
        ├── iris/      index.html + model.bin (1.5 KB)
        ├── wine/      index.html + model.bin (5.5 KB)
        ├── diabetes/  index.html + model.bin (2.8 KB)
        ├── titanic/   index.html + model.bin (1.7 KB)
        ├── housing/   index.html + model.bin (3.3 KB)   ← regression
        ├── heart/     index.html + model.bin (3.0 KB)
        ├── cancer/    index.html + model.bin (10.5 KB)
        ├── penguins/  index.html + model.bin (1.5 KB)
        ├── mpg/       index.html + model.bin (2.2 KB)   ← regression
        └── seeds/     index.html + model.bin (1.7 KB)
```

---

## Prerequisites

### Required

| Tool | Version | Install |
|------|---------|---------|
| Rust (stable) | 1.75+ | `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh` |
| wasm32 target | — | `rustup target add wasm32-unknown-unknown` |
| wasm-bindgen-cli | **0.2.122** | `cargo install wasm-bindgen-cli --version 0.2.122 --locked` |

> **Version lock:** `wasm-bindgen-cli` must exactly match the `wasm-bindgen`
> crate version in `tabular_wasm/Cargo.toml`. Check with
> `wasm-bindgen --version` vs `grep wasm-bindgen tabular_wasm/Cargo.toml`.

### Optional (for dataset download script only)

- `python3` (3.7+) — used by `scripts/download_datasets.sh` for CSV cleaning
- `curl` — HTTP downloader

---

## Quick start (local preview)

If you have the pre-built `web/` directory from the zip:

```bash
cd tabular_ml/web
python3 -m http.server 8080
# Open http://localhost:8080
```

That's it — the WASM binary and all model files are already in the zip.

---

## Full build from source

### Step 1 — Download datasets (skip if CSVs already present)

```bash
bash scripts/download_datasets.sh
```

Downloads, cleans, and writes all 10 CSVs to the project root. Takes ~30 seconds
on a normal connection. The raw intermediate files are not committed (see
`.gitignore`).

### Step 2 — Run tests

```bash
cargo test --workspace
# 131 tests, 0 failures across ferrum_core, tabular_wasm, and integration suite
```

The integration test suite loads every dataset CSV and every trained model file
(if present), so it doubles as a deployment verification step.

### Step 3 — Train all models

```bash
bash scripts/train_all.sh
# Or train a single dataset:
cargo run -p train_cli --release -- heart.csv web/datasets/heart/model.bin "Heart Disease" 32 600
```

| Dataset | CSV | Model | Accuracy / RMSE |
|---------|-----|-------|----------------|
| Iris | `iris.data` | `iris/model.bin` | 98.7% |
| Wine Quality | `wine.csv` | `wine/model.bin` | 80.9% |
| Pima Diabetes | `diabetes.csv` | `diabetes/model.bin` | 93.0% |
| Titanic | `titanic.csv` | `titanic/model.bin` | 86.9% |
| California Housing | `housing.csv` | `housing/model.bin` | RMSE ~$52k |
| Heart Disease | `heart.csv` | `heart/model.bin` | 96.3% |
| Breast Cancer | `cancer.csv` | `cancer/model.bin` | 99.3% |
| Palmer Penguins | `penguins.csv` | `penguins/model.bin` | 99.4% |
| Auto MPG | `mpg.csv` | `mpg/model.bin` | RMSE 1.95 mpg |
| Wheat Seeds | `seeds.csv` | `seeds/model.bin` | 99.5% |

Training all 10 takes ~60 seconds total, single-threaded on a laptop.

### Step 4 — Compile to WebAssembly

```bash
bash scripts/build_wasm.sh
# Or manually:
cargo build -p tabular_wasm --target wasm32-unknown-unknown --release
wasm-bindgen \
  target/wasm32-unknown-unknown/release/tabular_wasm.wasm \
  --out-dir web/pkg \
  --target web \
  --no-typescript
```

Output: `web/pkg/tabular_wasm_bg.wasm` (~128 KB) and `web/pkg/tabular_wasm.js` (~8 KB).

### Step 5 — Serve locally

```bash
# Python (built into every machine with Python 3)
python3 -m http.server 8080 --directory web

# Node.js
npx serve web

# Rust (cargo install basic-http-server)
basic-http-server web
```

Open `http://localhost:8080`. The landing page shows all 10 dataset cards.
Click any card to open its dataset page with sliders and live statistical terminals.

---

## The statistical terminals

Every dataset page contains two live terminals below the main prediction UI,
updating on every slider movement:

### Terminal 1 — Model Statistics (left)

Displays the **input vector analysis**:
- Feature name, current raw value, z-score (colour-coded: green < 1σ, yellow 1–2σ, red > 2σ)
- Mini bar showing the value's position within the dataset's feature range
- Centred z-bar showing direction and distance from the training mean
- Architecture summary: input/hidden/output dimensions, task type, normaliser, file format

### Terminal 2 — Quantitative Report (right)

**Classification datasets** show:
- Confidence badge: Certain / Confident / Uncertain / Toss-up
- Full probability table: P, bar, ln P, odds ratio for every class
- Shannon entropy H(p) in nats (0 = model certain, ln C = maximally confused)
- Top-2 margin: P(winner) − P(runner-up)

**Regression datasets** show:
- Predicted value on the full target range scale with a sliding pointer
- Dataset mean marker on the same scale
- Z-score of the prediction relative to the training target distribution
- Percentage above/below the dataset mean
- ±1σ reference interval from the training targets
- Approximate quartile (Q1/Q2/Q3/Q4)

All statistics are computed in JavaScript from data embedded in the model file —
no additional server calls or configuration files needed.

---

## Deployment targets

### GitHub Pages (recommended — free, permanent HTTPS)

#### Manual deploy

```bash
# 1. Create repo and push
git init && git add . && git commit -m "initial"
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main

# 2. Enable Pages: Settings → Pages → Source → Deploy from branch
#    Branch: main  |  Folder: /web  |  Save

# Site available at: https://<you>.github.io/<repo>/
```

#### Automated deploy with GitHub Actions

The included `.github/workflows/deploy.yml` runs on every push to `main`:

1. Installs Rust + wasm32 target + wasm-bindgen-cli
2. Checks formatting, runs clippy, runs all 131 tests
3. Trains all 10 models fresh from source
4. Compiles to WASM
5. Verifies all 10 model files have the correct FINF magic bytes
6. Deploys `web/` to GitHub Pages

Enable it by pushing `.github/workflows/deploy.yml` and activating GitHub Pages
under **Settings → Pages → Source → GitHub Actions**.


### Cloudflare Pages

```bash
npm install -g wrangler 
wrangler login
wrangler pages deploy web --project-name tabular-ml
```

For continuous deployment, connect the GitHub repo in the Cloudflare dashboard.
Set the build command to:

```bash
bash scripts/download_datasets.sh && \
bash scripts/train_all.sh && \
bash scripts/build_wasm.sh
```

Output directory: `web`

Build environment variables to add:

```
RUSTUP_TOOLCHAIN = stable
```

Add a build script that installs Rust on Cloudflare's build VMs:

```bash
# build.sh (Cloudflare build command)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
export PATH="$HOME/.cargo/bin:$PATH"
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.122 --locked
bash scripts/download_datasets.sh
bash scripts/train_all.sh
bash scripts/build_wasm.sh
```

### Netlify

**Drop-folder deploy** (instant, no build):
1. Drag `web/` folder onto https://app.netlify.com/drop
2. Get a `*.netlify.app` URL immediately

**Continuous deployment**: connect the GitHub repo and use the same build
command as Cloudflare above.

### Custom domain (GitHub Pages)

1. Add a `CNAME` file inside `web/`:
   ```
   ml.yourdomain.com
   ```
2. DNS: add `CNAME  ml.yourdomain.com  →  <you>.github.io`
3. GitHub Pages settings: enter the custom domain
4. GitHub provisions Let's Encrypt automatically (~15 minutes)

---

## Adding a new dataset

The engine is fully generic — adding an eleventh dataset requires:

1. **Prepare the CSV**: numeric feature columns, string or integer label in
   the last column, optional header row.

2. **Train the model**:
   ```bash
   cargo run -p train_cli --release -- my_data.csv web/datasets/myds/model.bin \
     "My Dataset Name" 48 500
   ```
   The trainer auto-detects classification vs regression. Hidden layer size
   and epochs are the last two arguments.

3. **Create the HTML page**: copy any existing `web/datasets/*/index.html`
   and change:
   - `<title>` and `<h1>` to the new dataset name
   - `<p class="subtitle">` description
   - The `.preset-btn` values (raw feature values for a few interesting examples)
   - The `<footer>` source URL

   Everything else — slider labels, ranges, the two statistical terminals —
   builds itself from the embedded metadata. No JavaScript changes needed.

4. **Add a card** to `web/index.html` in the appropriate section grid.

5. **Add to `scripts/train_all.sh`** and `.github/workflows/deploy.yml`.

---

## Troubleshooting

### "Failed to fetch model.bin" / 404

- Browsers block `fetch()` on `file://` URLs. Always serve via HTTP.
- Confirm `web/datasets/<slug>/model.bin` exists.
- On GitHub Pages, check the deploy log — sometimes Pages serves a cached
  version for a few minutes after a new push.

### "bad FINF magic" or "unsupported FINF version"

The model file was built with an older version of `ferrum_core`. Retrain:

```bash
bash scripts/train_all.sh
```

Then rebuild the WASM (the reader and writer must use the same version):

```bash
bash scripts/build_wasm.sh
```

### "wasm-bindgen version mismatch"

```
error: wasm-bindgen crate version X, CLI version Y — must match exactly
```

Fix: `cargo install -f wasm-bindgen-cli --version <crate-version> --locked`
The crate version is in `tabular_wasm/Cargo.toml`.

### Statistical terminals blank / show "—"

- Open DevTools → Console for JavaScript errors.
- Check that `model.norm_encoded()` is available — it requires the WASM build
  from this version of `tabular_wasm/src/lib.rs`.
- Ensure `web/shared/stats.js` is being served (check Network tab for 404s).

### NaN predictions from the housing or MPG model

These regression models use normalised targets. If you see NaN, the model
may have diverged during training. Retrain with a lower learning rate:

```bash
cargo run -p train_cli --release -- housing.csv web/datasets/housing/model.bin \
  "California Housing" 64 400
# The CLI automatically uses lr=0.01 for regression vs lr=0.05 for classification
```

### Blank page — ES module error in old Safari

The site uses `<script type="module">` and named ES module imports, which
require Safari 13.1+, Chrome 80+, or Firefox 72+. There is no fallback
for older browsers — the engine uses WASM, which those browsers cannot run.

---

## Size budget

| File | Size | Notes |
|------|------|-------|
| `tabular_wasm_bg.wasm` | ~128 KB | Entire ML engine, compiled |
| `tabular_wasm.js` | ~8 KB | wasm-bindgen glue |
| `shared/stats.js` | ~14 KB | Statistical terminal logic |
| `shared/style.css` | ~13 KB | Dark theme + terminal CSS |
| `shared/engine.js` | ~5 KB | Loader + inference helpers |
| `index.html` | ~7 KB | Landing page |
| Per dataset `index.html` | ~3 KB | Dataset page template |
| Per dataset `model.bin` | 1.5–10.5 KB | Weights + normalizer + metadata |
| **Total (all 10 datasets)** | **~230 KB** | Less than a medium JPEG |

---

## Running the full test suite

```bash
cargo test --workspace
```

131 tests across three suites:

| Suite | Count | Covers |
|-------|-------|--------|
| `ferrum_core` unit | 86 | Tensor math, ops, loss (finite-diff gradient check), normalizer, CSV parser, metadata JSON roundtrip, model serialisation |
| Integration | 39 | Full pipeline for all 10 datasets: parse → normalise → train → serialise → reload → infer on known samples; all models produce valid probability distributions |
| `tabular_wasm` unit | 6 | WASM glue: load from bytes, classification + regression inference, metadata JSON fields, corrupt-byte rejection, batch vs individual agreement |

The integration tests are also the deployment verification checklist: if
`cargo test --workspace` passes with the live model files, the site is working.
