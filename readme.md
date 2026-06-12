# Ferrum

**Zero-dependency, pure-Rust engine for building, training, and running causal
Transformers, Small Language Models (SLMs), and classical MLPs — entirely on the
CPU, with no GPU and no external crates.**

Ferrum is `std`-only and `#![forbid(unsafe_code)]`. The whole stack — tensors,
hand-written backprop, a byte-level BPE tokenizer, int8 quantization-aware
training, and a self-contained model file format — compiles to native binaries
and to WebAssembly, so the same trained model runs on a server, a laptop, a
Raspberry Pi, or in a browser tab.

- **No dependencies.** `ferrum_core` has an empty `[dependencies]` table.
- **No GPU.** Everything runs on the CPU; models are small by design.
- **Multi-threaded.** The matmul kernels behind every Linear, FFN, attention,
  and LM-head step run in parallel across all CPU cores — detected dynamically
  via `std::thread::available_parallelism()`, with zero external crates and
  bit-for-bit deterministic results. Override with `FERRUM_NUM_THREADS`.
- **Self-contained models.** Weights, normalizer, metadata, and the tokenizer
  travel together in one `.bin` file (the FINF format).
- **Quantization-aware.** Train against int8-snapped weights and ship 4×-smaller
  models that behave like the model you trained.
- **Byte-level BPE.** A subword tokenizer is integrated end-to-end through
  training, generation, and serialization — and it round-trips any UTF-8 text.

---

## Workspace layout

| Crate          | Kind                         | What it is                                                          |
|----------------|------------------------------|---------------------------------------------------------------------|
| `ferrum_core`  | library                      | The engine: tensors, layers, training, quantization, tokenizer, SLM, FINF I/O |
| `slm_cli`      | binary (`train_transformer`) | Train and generate from causal-transformer SLMs                     |
| `train_cli`    | binary (`train_cli`)         | Train tabular MLP classifiers/regressors from any CSV               |
| `tabular_wasm` | cdylib + rlib                | `wasm-bindgen` bindings for running models in the browser           |
| `tests`        | integration                  | Cross-crate integration and regression tests                        |

---

## Quick start

### Train a Small Language Model (byte-level BPE)

```bash
# Train a causal transformer SLM with a 512-token BPE vocabulary (the default).
cargo run -p slm_cli -- train corpus.txt model.bin --epochs 200 --context 16

# Continue a prompt.
cargo run -p slm_cli -- generate model.bin "Once upon a time" --chars 300 --temp 0.7

# Inspect a trained model.
cargo run -p slm_cli -- info model.bin
```

`--vocab 0` selects the legacy character-level tokenizer; any value `>= 256`
trains a byte-level BPE tokenizer of that size and stores its merge list inside
the model file.

### Use the library directly

```rust
use ferrum_core::{GenerativeSLM, Rng};

let corpus = std::fs::read_to_string("corpus.txt").unwrap();
let mut rng = Rng::new(1337);

// Train a BPE transformer SLM: context 16, embed 32, 4 heads, 2 blocks,
// FFN 64, 200 epochs, Adam lr 0.01, batch 16, BPE vocab 512.
let slm = GenerativeSLM::train_transformer(
    &corpus, 16, 32, 4, 2, 64, 200, 0.01, 16, 512, &mut rng,
).unwrap();

slm.save("model.bin").unwrap();                // int8-quantized FINF v5
let text = slm.generate("Once upon a time", 200, 0.7, &mut rng).unwrap();
println!("{text}");
```

### Train a tabular model

```bash
cargo run -p train_cli -- iris.csv model.bin "Iris" 32 500
```

---

## Architecture at a glance

```
Tensor ──► ops (matmul, softmax, layernorm, …)
     │
     └──► Layer trait
            ├── Linear            (y = xW + b)
            ├── ActivationLayer   (ReLU / Softmax / …)
            ├── LayerNorm         (per-row normalization)
            ├── Embedding         (token + positional lookup)
            ├── Flatten           (sequence → row)
            └── TransformerBlock  (causal multi-head self-attention + FFN)

Sequential ──► ordered pipeline of Layers   (+ KvCache for fast generation)

tokenizer  ──► ByteBpeTokenizer  (byte-level BPE; char-level fallback)
quant      ──► int8 fake-quantization for QAT and serialization
train      ──► Net (MLP), train_epoch, Adam / Sgd
train_transformer ──► TransformerNet, train_transformer_epoch
slm        ──► GenerativeSLM: train / train_embedded / train_transformer / generate
loader     ──► FINF v4 (f32) / v5 (int8) self-contained model files
```

See [docs/manual.md](docs/manual.md) for the full reference.

---

## The three SLM training paths

Ferrum offers three ways to train a generative model from raw text. All three
are quantization-aware and all three share the same generation and file format.

| Method                          | Architecture                         | Tokenizer            | Best for                                  |
|---------------------------------|--------------------------------------|----------------------|-------------------------------------------|
| `GenerativeSLM::train`          | flat one-hot MLP                     | character-level      | the simplest possible baseline            |
| `GenerativeSLM::train_embedded` | embedding + MLP                      | char or **BPE**      | small, fast models that beat one-hot size |
| `GenerativeSLM::train_transformer` | causal multi-head Transformer     | char or **BPE**      | the highest quality on real text          |

The `vocab_size` argument selects the tokenizer for the embedded and transformer
paths: `0` is character-level, any value `>= 256` trains a byte-level BPE
tokenizer of that size.

---

## Documentation

| Document                          | Contents                                            |
|-----------------------------------|-----------------------------------------------------|
| [installation.md](installation.md)| Build, install, and toolchain requirements          |
| [howtouse.md](howtouse.md)        | CLI and library usage guide                          |
| [example.md](example.md)          | End-to-end worked examples                           |
| [usecases.md](usecases.md)        | Ten scenarios where Ferrum is a good fit            |
| [evaluation.md](evaluation.md)    | How to measure quality, size, and speed             |
| [deployment.md](deployment.md)    | Shipping models to edge, embedded, and WASM targets |
| [docs/manual.md](docs/manual.md)  | Complete API and format reference                   |
| [docs/user_guide.md](docs/user_guide.md) | Task-oriented walkthroughs                   |
| [docs/how_to_use.md](docs/how_to_use.md) | Browser/WASM playground tutorial             |
| [docs/FAQs.md](docs/FAQs.md)      | Frequently asked questions                          |
| [status.md](status.md)            | Project status and roadmap                          |

---

## Building and testing

```bash
cargo build --workspace            # build everything
cargo test  --workspace            # run all unit + integration tests
cargo doc   -p ferrum_core --open  # browse the API docs
```

---

## License

Licensed under either of **MIT** or **Apache-2.0** at your option.
