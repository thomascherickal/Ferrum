# 🧬 Ferrum — Hand-Crafted Edge ML & Causal Transformer Engine

Ferrum is a zero-dependency, pure-Rust workspace designed to compile and run hand-crafted machine learning models—specifically Feedforward MLPs and Decoder-Only Causal Transformers—directly at the edge, on the CPU, and in the browser via WebAssembly. 

No GPU required. No PyTorch or Python runtimes. No cloud latency or API keys. Single-threaded CPU execution compiled directly to highly compact binaries under 180 KB.

---

## Workspace Layout

```text
ferrum_lib/
├── shell_oracle/              # Parent Crate 1: CLI commands weaver
├── ambient_poet/              # Parent Crate 2: System state Zen haiku poet
├── brand_alchemist/           # Parent Crate 3: Startup brand & slogan copywriter
│
└── ferrum/                    # Core Workspace Crate
    ├── Cargo.toml             # Workspace definition (members: ferrum_core, tabular_wasm, train_cli, tests)
    │
    ├── ferrum_core/           # ML Engine (Pure Rust, std only)
    │   └── src/
    │       ├── slm.rs         # [NEW] Causal Small Language Model (SLM) library module
    │       ├── layer.rs       # Layer trait: Linear, LayerNorm, Embedding, TransformerBlock
    │       ├── model.rs       # Sequential network pipeline
    │       ├── tensor.rs      # High-performance row-major float arrays
    │       ├── ops.rs         # Mathematical operations: matmul, bias, argmax, softmax
    │       ├── rng.rs         # Seeded xorshift64* pseudo-random generator
    │       ├── loss.rs        # Loss kernels (Softmax Cross-Entropy & MSE with gradients)
    │       ├── optim.rs       # SGD optimizer with momentum
    │       ├── csv.rs         # Robust CSV dataset parser, normalizer, and metadata
    │       ├── train.rs       # Trainable layer wrappers (DenseT, ReluT, Net)
    │       └── loader.rs      # Serializer/deserializer for self-contained FINF v4 binaries
    │
    ├── tabular_wasm/          # WebAssembly Bindings (wasm-bindgen)
    │   └── src/lib.rs         # TabularModel and TransformerSLMModel WASM bindings
    │
    ├── train_cli/             # Generic CSV tabular model trainer
    ├── tests/                 # 195+ automated unit and integration tests
    │
    └── web/                   # WASM Web Playgrounds
        ├── index.html         # Suite gateway portal page
        ├── shared/            # Common assets: style.css & engine.js WASM interface
        ├── shell_oracle/      # Cyberpunk terminal autocompleter playground
        ├── ambient_poet/      # Calming Zen telemetry composer playground
        └── brand_alchemist/   # Modern gradient startup weaver playground
```

---

## Key Features

1. **`ferrum_core` as an Independent Library**: A lightweight, auditable Rust engine that compiles cleanly to OS-less targets (like `wasm32-unknown-unknown`) because it relies strictly on `std` and zero external crates.
2. **Generic `slm` Causal Module**: An out-of-the-box library engine inside `ferrum_core` designed to train small next-character language models from scratch on custom raw text corpora, utilizing hex-encoded vocabulary mappings to maintain 100% data integrity within CSV rows.
3. **Decoder-Only Causal Transformer Blocks**: Includes `Embedding` (token + positional), `LayerNorm`, and `TransformerBlock` (Causal Multi-Head Self-Attention + FFN) layers for running complex Small Language Models (SLMs) in WASM.
4. **Stunning Web Playgrounds**: Three highly optimized, beautiful, responsive interactive browser applications that stream characters, output Shannon entropy, and render probability distributions dynamically.

---

## Quick Start

### 1. Pre-requisites (One-time)
```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.122 --locked
```

### 2. Run the 196 Automated Tests
```bash
cargo test --workspace
```

### 3. Compile the WebAssembly Package
```bash
bash scripts/build_wasm.sh
```

### 4. Run the Standalone Parent Applications
To train, export, and run interactive generation loops:
```bash
# Shell Oracle
cd ../shell_oracle && cargo run --release

# Ambient Poet
cd ../ambient_poet && cargo run --release

# Brand Alchemist
cd ../brand_alchemist && cargo run --release
```

### 5. Launch the Web Playgrounds
Copy the trained models to the web folder and host a local server:
```bash
# From workspace root (ferrum/)
mkdir -p web/datasets/shell_oracle web/datasets/ambient_poet web/datasets/brand_alchemist
cp ../shell_oracle/shell_oracle.bin web/datasets/shell_oracle/model.bin
cp ../ambient_poet/ambient_poet.bin web/datasets/ambient_poet/model.bin
cp ../brand_alchemist/brand_alchemist.bin web/datasets/brand_alchemist/model.bin

# Serve
python3 -m http.server 8080 --directory web
# Open http://localhost:8080
```
