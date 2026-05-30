# Tabular ML — Pure Rust → WebAssembly

Ten real datasets. Ten trained neural networks. Two live statistical terminals
per page. All running in your browser from a single **128 KB WebAssembly binary**
— compiled from hand-written Rust with **zero external dependencies**.

No server. No Python. No cloud. Move a slider and the prediction updates in
microseconds, directly inside the browser tab.

---

## Live demo

```
https://<your-github-username>.github.io/tabular-ml/
```

Landing page shows all 10 dataset cards. Each card opens an interactive page with:
- **Sliders** built dynamically from embedded feature metadata
- **Prediction display** with probability bars (classification) or a scaled value gauge (regression)
- **Terminal 1 — Model Statistics**: input vector z-scores, feature range positions, architecture summary
- **Terminal 2 — Quantitative Report**: entropy analysis, log-probabilities, odds ratios (classification) or z-score, quartile, reference intervals (regression)

---

## Datasets

| | Dataset | Task | Features | Rows | Result |
|---|---------|------|----------|------|--------|
| 🌸 | Iris Species | 3-class | 4 | 150 | 98.7% acc |
| 🐧 | Palmer Penguins | 3-class | 4 | 342 | 99.4% acc |
| 🌾 | Wheat Seeds | 3-class | 7 | 210 | 99.5% acc |
| 🍷 | Wine Quality | 3-class | 11 | 1,599 | 80.9% acc |
| 🩺 | Pima Diabetes | binary | 8 | 768 | 93.0% acc |
| ❤️ | Heart Disease | binary | 13 | 297 | 96.3% acc |
| 🔬 | Breast Cancer | binary | 30 | 569 | 99.3% acc |
| 🚢 | Titanic Survival | binary | 6 | 891 | 86.9% acc |
| 🚗 | Auto MPG | regression | 6 | 392 | RMSE 1.95 mpg |
| 🏠 | California Housing | regression | 8 | 20,433 | RMSE ~$52k |

---

## Quick start

```bash
# Prerequisites (one-time)
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.122 --locked

# 1  Run all 131 tests
cargo test --workspace

# 2  Train all 10 models (~60 seconds)
bash scripts/train_all.sh

# 3  Compile to WASM
bash scripts/build_wasm.sh

# 4  Serve
python3 -m http.server 8080 --directory web
#     → open http://localhost:8080
```

For a fresh clone without pre-downloaded CSVs, also run:

```bash
bash scripts/download_datasets.sh   # fetches + cleans all 10 source files
```

---

## Repository layout

```
tabular_ml/
├── .github/workflows/deploy.yml  # CI: test → train → WASM → GitHub Pages
├── scripts/
│   ├── download_datasets.sh      # fetch + clean all source CSVs
│   ├── train_all.sh              # train all 10 models
│   └── build_wasm.sh             # compile to WASM
│
├── ferrum_core/                  # ML engine — pure Rust, std only
│   └── src/
│       ├── error.rs              # InferError, Result<T>
│       ├── tensor.rs             # Tensor: Vec<f32> + row-major shape
│       ├── ops.rs                # matmul, bias-add, transpose, argmax, softmax
│       ├── activation.rs         # ReLU, Sigmoid, Tanh, Softmax, Identity
│       ├── layer.rs              # Layer trait, Linear, ActivationLayer
│       ├── model.rs              # Sequential pipeline
│       ├── rng.rs                # xorshift64* PRNG
│       ├── loss.rs               # softmax cross-entropy + MSE (both with gradients)
│       ├── optim.rs              # SGD with momentum
│       ├── csv.rs                # CSV parser, Normalizer, ModelMetadata, TaskType
│       ├── train.rs              # DenseT, ReluT, Net, backprop, train_epoch
│       └── loader.rs             # FINF v3 binary format
│
├── tabular_wasm/src/lib.rs       # WASM bindings: TabularModel { predict, metadata, norm_encoded }
├── train_cli/src/main.rs         # Generic CSV trainer (auto-detects task type)
├── tests/integration_test.rs     # 39 end-to-end integration tests
│
├── *.csv / iris.data             # 10 cleaned training datasets
│
└── web/                          # ← serve this directory
    ├── index.html                # Landing page (10 dataset cards)
    ├── shared/
    │   ├── engine.js             # WASM loader + inference + slider builder
    │   ├── stats.js              # Live statistical terminals (392 lines)
    │   └── style.css             # Dark theme + terminal CSS
    ├── pkg/
    │   ├── tabular_wasm_bg.wasm  # Compiled ML engine (~128 KB)
    │   └── tabular_wasm.js       # wasm-bindgen glue (~8 KB)
    └── datasets/
        └── <slug>/
            ├── index.html        # Dataset page (identical structure, metadata-driven)
            └── model.bin         # FINF v3: weights + normalizer + metadata JSON
```

---

## The FINF v3 model format

Each `model.bin` is a self-contained binary file in the **FINF v3** format:

```
4 bytes  b"FINF"                      magic
u32      version = 3
u32      normalizer_byte_length
[bytes]  "mean0,std0;mean1,std1;…"    z-score stats (features + optional target)
u32      metadata_byte_length
[bytes]  { JSON }                     ModelMetadata (feature names, ranges,
                                      class names, task type, input/output dims)
u32      num_layers
[layers] u8 tag, then layer bytes
```

The embedded metadata is what allows the browser to build sliders, label
probability bars, and power both statistical terminals — without any per-dataset
JavaScript or additional configuration files.

---

## The statistical terminals

Every dataset page updates two terminals on every slider drag:

**Terminal 1 — Model Statistics**

| Column | What it shows |
|--------|--------------|
| Feature | Name from CSV header |
| Value | Current raw slider value |
| Z-score | (value − μ) / σ from training data; colour-coded by magnitude |
| Range% | Mini bar: position within [dataset min, dataset max] |
| Z-bar | Centred bar: direction and distance from the training mean |

Plus a static architecture card showing layer dimensions, task, and file format.

**Terminal 2 — Quantitative Report (classification)**

- Confidence badge: Certain / Confident / Uncertain / Toss-up (from Shannon entropy)
- Full probability table: P, log P, odds ratio for every class
- Shannon entropy H(p) gauge: 0 nats = model certain, ln(C) = maximally confused
- Top-2 margin (P(winner) − P(runner-up))

**Terminal 2 — Quantitative Report (regression)**

- Prediction on a range scale with dataset mean marker
- Z-score of the prediction relative to training targets
- Percentage above/below the dataset mean
- ±1σ reference interval; approximate quartile

---

## Test coverage

```
cargo test --workspace    →   131 tests, 0 failures
```

| Suite | Tests | What it verifies |
|-------|-------|-----------------|
| `ferrum_core` unit | 86 | Every arithmetic kernel, loss (finite-diff gradient check), normalizer, CSV parser, metadata JSON roundtrip, FINF v3 serialisation |
| Integration | 39 | All 10 datasets: parse → train → serialise → reload → infer; classification outputs sum to 1; regression predictions in plausible ranges |
| `tabular_wasm` unit | 6 | WASM glue: load, infer, metadata fields, norm_encoded, batch vs individual agreement, corrupt-byte rejection |

---

## Deployment

See **[DEPLOYMENT.md](DEPLOYMENT.md)** for:

- Full build-from-source walkthrough (download → test → train → WASM → serve)
- GitHub Pages (manual + GitHub Actions CI/CD)
- Cloudflare Pages and Netlify
- Custom domain + HTTPS
- Adding a new dataset (3-step process)
- Troubleshooting common issues

---

## Why zero dependencies?

- The `wasm32-unknown-unknown` WASM target has no OS, no file system, no libc.
  A `std`-only crate compiles to it cleanly; any crate with OS dependencies won't.
- The binary contains exactly what inference needs — no Python interpreter,
  no framework, no runtime overhead.
- Every line of the ML pipeline is in this repository and auditable.
- The deployment story *is* the architecture story: the same constraint that
  requires zero dependencies is what makes a 128 KB browser-side ML engine possible.
