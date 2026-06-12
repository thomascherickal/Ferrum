# Ferrum — Edge SLM Inference & Training Engine

Ferrum is a **zero-dependency, pure-Rust** workspace for building, training, and running
small machine-learning models — feedforward MLPs and decoder-only causal Transformers —
on CPU-only, edge, and WebAssembly targets.

- **No GPU.** Single-threaded CPU kernels, cache-friendly loop ordering.
- **No external crates** in the core engine (`ferrum_core` uses `std` only).
- **Deterministic.** Seeded xorshift64* PRNG; the same seed reproduces the same model.
- **Self-contained models.** One FINF v4 binary holds weights + normalizer + metadata.
- **Browser-ready.** `tabular_wasm` exposes the engine to JavaScript via `wasm-bindgen`.

---

## Workspace Layout

```text
ferrum/
├── Cargo.toml             # Workspace: ferrum_core, tabular_wasm, train_cli, tests
│
├── ferrum_core/           # The engine (pure Rust, std only)
│   └── src/
│       ├── tensor.rs      # Flat row-major f32 tensor with shape checking
│       ├── ops.rs         # matmul, add_bias, transpose, softmax_rows, argmax_rows…
│       ├── activation.rs  # Identity / ReLU / Sigmoid / Tanh / Softmax (serialisable)
│       ├── layer.rs       # Layer trait: Linear, ActivationLayer, LayerNorm,
│       │                  #   Embedding (token+positional), TransformerBlock (causal MHA)
│       ├── model.rs       # Sequential — ordered pipeline of layers
│       ├── loss.rs        # Fused softmax cross-entropy + MSE, both with gradients
│       ├── optim.rs       # SGD with momentum + Adam (bias-corrected)
│       ├── train.rs       # Trainable MLP (DenseT/ReluT/Net), backprop, train_epoch
│       ├── train_transformer.rs # Trainable causal Transformer (full backprop + Adam)
│       ├── csv.rs         # CSV parser, Normalizer, task auto-detection, ModelMetadata
│       ├── slm.rs         # GenerativeSLM — character-level causal language model
│       ├── loader.rs      # FINF v4 binary save/load (all 5 layer types)
│       ├── rng.rs         # Seeded xorshift64* PRNG
│       └── verbose.rs     # Opt-in tracing (set_verbose) + vprintln! macro
│
├── tabular_wasm/          # WASM bindings: TabularModel + TransformerSLMModel
├── train_cli/             # CLI: train any CSV (classification/regression) → FINF
├── tests/                 # Integration test crate (200+ tests in the workspace)
└── web/                   # Browser playgrounds consuming the WASM package
```

---

## Architecture at a Glance

```text
Tensor ──► ops (matmul, softmax, …)
     │
     └──► Layer trait
            ├── Linear           (y = xW + b, W is [in, out] — no transpose at inference)
            ├── ActivationLayer  (ReLU / Softmax / …)
            ├── LayerNorm        (per-row normalisation, ε = 1e-5)
            ├── Embedding        (token lookup + learned positional encoding)
            └── TransformerBlock (pre-norm causal multi-head self-attention + FFN,
                                  residual connections, attention maps exposed)

Sequential ──► ordered pipeline of Layers
loader     ──► FINF v4 binary format (save / load / to_bytes / from_bytes)
slm        ──► GenerativeSLM: corpus → train → generate (char-level, temperature sampling)
train      ──► Net (trainable MLP), train_epoch, accuracy
```

---

## Quick Start

### Build & test

```bash
cargo build --workspace
cargo test  --workspace      # 217 tests, all passing
```

### Train a tabular model from any CSV

```bash
cargo run -p train_cli -- tests/fixtures/iris.data /tmp/iris.bin "Iris" 32 200
# Auto-detects classification vs regression, trains, validates, exports FINF.
# Add --verbose to trace every kernel call.
```

### Train and run a character-level SLM (library API)

```rust
use ferrum_core::{GenerativeSLM, Rng};

let corpus = std::fs::read_to_string("corpus.txt")?;
let mut rng = Rng::new(42);

// corpus, context_len, hidden_size, epochs, lr, momentum, batch_size, rng
let slm = GenerativeSLM::train(&corpus, 8, 64, 200, 0.05, 0.9, 16, &mut rng)?;

let text = slm.generate("once upo", 100, 0.8, &mut rng)?;   // seed, n_chars, temperature
std::fs::write("model.bin", slm.to_bytes()?)?;              // self-contained binary
let reloaded = GenerativeSLM::from_bytes(&std::fs::read("model.bin")?)?;
```

Or train a **real decoder-only causal Transformer** (embeddings + multi-head
attention + FFN, trained end-to-end with Adam):

```rust
// corpus, context_len, embed_dim, num_heads, num_blocks, hidden_dim,
// epochs, lr, batch_size, rng
let slm = GenerativeSLM::train_transformer(&corpus, 16, 32, 4, 2, 64,
                                           100, 0.003, 16, &mut rng)?;
let text = slm.generate("once upon a time", 200, 0.8, &mut rng)?;
```

See **[example.md](example.md)** for a complete walkthrough, including hand-building a
true Transformer SLM with `Embedding` + `TransformerBlock`.

### Run a pre-trained model (inference only)

```rust
use ferrum_core::{from_bytes, Tensor};

let bytes = std::fs::read("model.bin")?;
let (model, norm, meta) = from_bytes(&bytes)?;
let input = norm.transform(&Tensor::row(vec![5.1, 3.5, 1.4, 0.2])?)?;
let probs = model.forward(&input)?;          // softmax already applied
```

### WebAssembly

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.122 --locked
bash scripts/build_wasm.sh
python3 -m http.server 8080 --directory web   # open http://localhost:8080
```

JavaScript side:

```js
const bytes = new Uint8Array(await (await fetch('model.bin')).arrayBuffer());
const slm   = new TransformerSLMModel(bytes);
const probs = slm.predict_next(new Float32Array([12, 5, 3, 7]));  // token IDs
const attn  = slm.get_last_attention_weights();  // [heads × T × T] for visualisation
```

---

## The FINF v4 Model Format

One little-endian binary blob, fully self-describing:

| Field        | Size | Contents                                         |
|--------------|------|--------------------------------------------------|
| magic        | 4 B  | `"FINF"`                                         |
| version      | u32  | `4`                                              |
| norm_len     | u32  | length of normalizer string (empty for SLMs)     |
| norm         | …    | `mean,std;mean,std;…` per feature                |
| meta_len     | u32  | length of metadata JSON                          |
| meta         | …    | dataset name, task, feature/class names, dims    |
| num_layers   | u32  |                                                  |
| layers       | …    | per layer: `u8` tag + raw f32 weights            |

Layer tags: `0` Linear, `1` Activation, `2` Embedding, `3` LayerNorm, `4` TransformerBlock.

The embedded metadata means a browser UI can build itself from the model file alone — no
separate config request.

---

## Verbose Diagnostics

Every module is instrumented. Enable tracing once at startup:

```rust
ferrum_core::set_verbose(true);
```

You get per-call shape logs, min/max/mean activation stats, dead-ReLU percentages,
NaN/Inf detection, per-epoch loss/ETA, and attention-weight statistics. Overhead when
off is a single atomic load per call site.

---

## Testing

```bash
cargo test --workspace
```

The suite covers tensor/shape validation, every kernel against hand-computed values,
analytic-vs-finite-difference gradient checks for the full backprop path, causal-mask
enforcement, attention rows summing to 1, FINF round-trips for all layer types,
corrupt/truncated-file handling, and an end-to-end train→generate→serialize→reload SLM
pipeline.

---

## Documentation

- **[evaluation.md](evaluation.md)** — engineering evaluation: strengths, fixed defects, gaps, roadmap.
- **[example.md](example.md)** — build your own SLM with Ferrum, step by step.
- **[INSTALLATION.md](INSTALLATION.md)** / **[DEPLOYMENT.md](DEPLOYMENT.md)** — setup and hosting.
- `docs/` — user guide, manual, FAQs.

## License

MIT OR Apache-2.0
