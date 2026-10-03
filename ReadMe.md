# Ferrum

**A zero-dependency, pure-Rust engine for training, running, and shipping small
causal Transformer language models (SLMs) on the CPU, with no GPU and no
external crates.**

Ferrum is `std`-only and `#![forbid(unsafe_code)]` throughout. The project
contains all of it: tensors, hand-written and gradient-checked backpropagation,
a byte-level BPE tokenizer, int8/int4 quantization-aware training, a
self-contained model file, a multi-core matmul engine, and a GGUF
reader/writer for Llama/Qwen checkpoints. There is no NumPy, no BLAS, and no
CUDA. The same code compiles to a native binary and to WebAssembly, so a model
you train runs unchanged on a server, a laptop, a Raspberry Pi, or in a browser
tab.

The guiding idea is that **every part is visible and auditable**. That is also
the main limit: Ferrum writes its own kernels instead of calling a tuned BLAS,
so it gives up peak throughput in exchange for transparency and portability. It
is a precise tool for small models. It is not a runtime for frontier LLMs.

| Property | What it means |
|---|---|
| **No dependencies** | `ferrum_core`'s `[dependencies]` table is empty. Only the WASM crate pulls in `wasm-bindgen`. |
| **CPU only, deterministic** | Every matmul splits across a persistent worker pool. Results are bit-for-bit identical at any thread count (override with `FERRUM_NUM_THREADS`). |
| **Self-contained models** | Weights, metadata, and tokenizer travel in one `.bin` file (the FINF format), with no sidecar vocab or config. |
| **Quantization-aware** | Train against int8-snapped weights and ship models 4× smaller (int8) or 8× smaller (int4) that behave like what you trained. |
| **Byte-level BPE** | Any UTF-8 text round-trips, with no unknown-token fallback. Tokenizers train quickly and can be saved and reused across runs. |
| **A real training pipeline** | Non-overlapping windows, token budgets, warmup plus cosine LR, gradient clipping, weight tying, validation with early stopping, and checkpoint/resume. |
| **GGUF in and out** | Import Llama/Qwen checkpoints (F32/F16/Q8_0/Q8_1/Q4_0/Q4_1, Q2_K–Q6_K k-quants) with their own tokenizer. Run them, fine-tune them with AdamW, and export them back to GGUF for llama.cpp, ollama, or LM Studio. |

---

## Contents

1. [Workspace layout](#workspace-layout)
2. [Installation](#installation)
3. [Quick start](#quick-start)
4. [How big a model can I train?](#how-big-a-model-can-i-train)
5. [Documentation](#documentation)
6. [Building and testing](#building-and-testing)
7. [License](#license)

---

## Workspace layout

| Crate | Kind | What it is |
|---|---|---|
| `ferrum_core` | library | The engine: tensors, layers, the transformer trainer, tokenizer, quantization, FINF I/O, and the GGUF reader/writer plus Llama/Qwen runner and trainer |
| `slm_cli` | binary `train_transformer` | Train, generate, evaluate, and inspect SLMs; `run-gguf`, `finetune-gguf`, and `export-gguf` for external checkpoints |
| `slm_wasm` | cdylib + rlib | `wasm-bindgen` bindings that run a trained SLM in the browser (`TransformerSLMModel`) |
| `tests` | integration tests | Cross-crate integration and regression tests |
| `ferrum_gui` | Tauri app (built separately) | **Ferrum SLM Studio**, a desktop GUI for the whole pipeline; see [Manual §10](Manual.md#10-the-gui-ferrum-slm-studio) |

`ferrum_gui` is deliberately **outside** the Cargo workspace because it needs
heavy system WebView libraries. Build it from its own directory.

**Sibling folders.** Two folders next to this repository hold code that used
to live here (see [Status §2](Status.md#2-the-2026-10-03-restructuring)):

- `../Ferrum-ML` contains the working machine-learning demos that are not
  about SLMs: a tabular MLP classifier/regressor (`train_cli`), its CSV loader,
  ten datasets, pretrained models, and a browser binding. It builds on
  `ferrum_core` by path.
- `../Ferrum-Junk` is an archive of removed features that have no use on their
  own: the one-hot and embedding-MLP SLM trainers, the old web launcher, the
  GUI's embedded shell, and the agent planning documents.

---

## Installation

### Requirements

| Requirement | Version | Notes |
|---|---|---|
| Rust + Cargo | 1.74+ stable, edition 2021 | Install via [rustup](https://rustup.rs) |
| OS | Linux, macOS, or Windows | Pure CPU code with no platform-specific paths |
| `wasm32-unknown-unknown` | optional | Only for browser builds (`slm_wasm`) |
| WebView system libraries | optional | Only for the desktop GUI; see [Manual §10.2](Manual.md#102-building-and-launching) |

A clean checkout pulls in **zero** third-party crates for `ferrum_core` and the
CLI, so builds are fast and reproducible and there is no supply chain to audit.

### Build and verify

```bash
git clone https://github.com/thomascherickal/Ferrum
cd Ferrum/ferrum
cargo build --workspace --release   # LTO + 1 codegen unit; use it for real training
cargo test  --workspace --release   # all unit + integration tests should pass
```

Debug builds are several times slower. Use `--release` for any real training
or for running GGUF models.

### Install the CLI, or use the library

```bash
cargo install --path slm_cli        # installs the `train_transformer` binary
```

```toml
[dependencies]
ferrum_core = { git = "https://github.com/thomascherickal/Ferrum" }
```

Because `ferrum_core` is `std`-only and `#![forbid(unsafe_code)]`, it adds no
transitive dependencies to your project. That makes it easy to use in audited,
embedded, or air-gapped builds.

---

## Quick start

### Train, generate, evaluate

```bash
# A ~1.3M-parameter BPE model on a few MB of text, budgeted at 50M tokens.
train_transformer train corpus.txt model.bin \
    --context 128 --embed 128 --heads 4 --blocks 4 --hidden 512 \
    --vocab 2048 --lr 0.003 --tokens 5e7 --cosine --warmup 200 --clip 1 --tie \
    --tokenizer tok.bpe --checkpoint run.fckp --checkpoint-every 500 --val 0.05

# Interrupted? Pick up exactly where the checkpoint left off.
train_transformer train corpus.txt model.bin ...same flags... --resume run.fckp

train_transformer generate model.bin "Once upon a time" --chars 300 --temp 0.7 --stream
train_transformer eval     model.bin heldout.txt     # perplexity, bits/token
train_transformer info     model.bin                 # format, tokenizer, layers
```

For a 30-second smoke test, `train_transformer train corpus.txt model.bin
--epochs 20` uses the toy defaults (context 16, embedding 32, 2 blocks).
[Manual §6](Manual.md#6-training-an-slm) explains every flag.

### Use the library

```rust
use ferrum_core::{GenerativeSLM, Rng, TransformerConfig};

let corpus = std::fs::read_to_string("corpus.txt")?;
let cfg = TransformerConfig {
    context_len: 64, embed_dim: 64, num_heads: 4, num_blocks: 2, hidden_dim: 256,
    vocab_size: 1024, lr: 0.003, batch_size: 16,
    max_tokens: 5_000_000, cosine_lr: true, warmup_steps: 100, grad_clip: 1.0,
    ..TransformerConfig::default()
};
let mut rng = Rng::new(1337);
let slm = GenerativeSLM::train_transformer_config_threaded(&corpus, &cfg, 0, &mut rng, |ep, loss| {
    println!("epoch {ep}: loss {loss:.4}");
})?;
slm.save("model.bin")?;                                    // int8 FINF v5
println!("{}", slm.generate("Once upon a time", 200, 0.7, &mut rng)?);
```

### Run, fine-tune, and export a GGUF model

```bash
train_transformer run-gguf      qwen2-0_5b.gguf "Explain rust ownership" --quant int4 --max 64
train_transformer finetune-gguf smollm2-135m.gguf corpus.txt tuned.flck --epochs 1 --lr 1e-4
train_transformer export-gguf   smollm2-135m.gguf tuned.gguf --resume tuned.flck --quant q8_0
```

Supported architectures are `llama` and `qwen2`. The imported model's tokenizer
comes from the file, and instruct models get their chat template applied
automatically. See [Manual §9](Manual.md#9-external-models-gguf-run-fine-tune-export).

---

## How big a model can I train?

Measured on an Intel i5-1135G7 (4 cores / 8 threads, 16 GB RAM) with int8 QAT,
batch 16, and 8 threads. Compute is the binding limit. RAM is not, and a 2 GB
GPU makes no difference because Ferrum never uses it.

| Goal | Reachable size |
|---|---|
| Heavily over-trained, minutes to hours | 0.1M–2M parameters |
| Compute-optimal (≈20 tokens/param) in 24 h | ≈5M |
| Compute-optimal in a week | ≈12–15M |
| Compute-optimal in a month | ≈25–30M |
| Fine-tune an imported GGUF (full AdamW, f32) | ≈135M comfortably; 0.5B is tight |

A TinyStories-class model (1–3M parameters on about 470M tokens) is a 1–3 day
job. [Manual §6.7](Manual.md#67-sizing-a-run-for-your-machine) has the measured
throughput table, and [Status §3](Status.md#3-capability-assessment) has the
full assessment.

---

## Documentation

| Document | Read it for |
|---|---|
| **[Manual.md](Manual.md)** | Everything about *using* Ferrum, from first principles (what an SLM is, why Rust, why data matters) through training, generation, evaluation, GGUF, the GUI, and deployment, to the full API and file-format reference, benchmarks, use cases, FAQ, and an honest critique |
| **[Status.md](Status.md)** | Where the project stands: what is implemented and tested, the 2026-10-03 restructuring, the capability assessment, known issues, the roadmap, and the development history |
| `cargo doc -p ferrum_core --open` | Rustdoc for every public item |

---

## Building and testing

```bash
cargo build --workspace                       # everything except the GUI
cargo test  --workspace --release             # unit + integration tests
cargo bench --bench gemm                      # std-only matmul/decode microbenchmark
cargo clippy --workspace --all-targets -- -D warnings
bash scripts/build_wasm.sh                    # slm_wasm → slm_wasm/pkg/ (needs wasm-bindgen)
(cd ferrum_gui && cargo tauri dev)            # the desktop app
```

CI (`.github/workflows/ci.yml`) runs build and test, rustfmt, clippy, JS syntax
checks, cargo-deny, a wasm32 build of `slm_wasm`, and the GUI crate's own
format, lint, and test jobs.

---

## License

MIT. See [LICENSE](LICENSE).
