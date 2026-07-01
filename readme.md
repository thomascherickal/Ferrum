# Ferrum

**A zero-dependency, pure-Rust engine for building, training, and running causal
Transformers, Small Language Models (SLMs), and classical MLPs — entirely on the
CPU, with no GPU and no external crates.**

Ferrum is `std`-only and `#![forbid(unsafe_code)]` from top to bottom. Tensors,
hand-written backpropagation, a byte-level BPE tokenizer, int8/int4
quantization, a self-contained model file, a GGUF importer for external
Llama/Qwen checkpoints, and a multi-core matmul engine are all written out in the
project itself — no NumPy, no BLAS, no CUDA. The same code compiles to a native
binary and to WebAssembly, so a model you train runs unchanged on a server, a
laptop, a Raspberry Pi, or a browser tab.

The guiding idea: **every gear is visible and auditable.** That is also the
honest limit — by hand-writing its own kernels instead of calling a tuned BLAS,
Ferrum trades peak throughput for transparency and portability. It is a precision
hand tool for small models, not a frontier-LLM runtime.

- **No dependencies.** `ferrum_core`'s `[dependencies]` table is empty; only the
  WASM crate pulls `wasm-bindgen`.
- **No GPU, ever.** Everything is CPU; models are small by design.
- **Multi-threaded, deterministically.** Every matmul (Linear, FFN, attention,
  LM head) splits its rows across a **persistent worker pool** spawned once and
  reused — so autoregressive decoding pays no per-token thread-spawn cost. The
  split never changes the arithmetic, so results are **bit-for-bit identical at
  any thread count**. Detected via `std::thread::available_parallelism()`;
  override with `FERRUM_NUM_THREADS`.
- **Self-contained models.** Weights, normalizer, metadata, and tokenizer travel
  in one `.bin` file (the FINF format) — no sidecar vocab or config to lose.
- **Quantization-aware.** Train against int8-snapped weights and ship 4×-smaller
  models that behave like what you trained; serialize to int8 *or* int4.
- **Byte-level BPE.** A subword tokenizer integrated end-to-end through training,
  generation, and serialization; its 256-byte base means *any* UTF-8 text
  round-trips with no unknown-token escape hatch.
- **Run external models (GGUF).** Import quantized **Llama/Qwen** checkpoints —
  GGUF `F32/F16/Q8_0/Q8_1/Q4_0/Q4_1` and the **Q4_K/Q5_K/Q6_K** k-quants — *with
  their own tokenizer*, and decode them on the CPU (RMSNorm, RoPE, grouped-query
  attention, SwiGLU, KV cache) in int4/int8/f32. A streamed reader avoids holding
  the whole file in RAM. You can also **write** them back: export a llama/qwen2 model
  (imported or fine-tuned) to GGUF at f16/int8/int4/k-quants with `export-gguf`.
- **Train the imported architecture too.** A finite-difference-checked backward
  pass (`llm_train`) makes the Llama/Qwen stack trainable, not just runnable.

---

## Workspace layout

| Crate          | Kind                         | What it is                                                          |
|----------------|------------------------------|---------------------------------------------------------------------|
| `ferrum_core`  | library                      | The engine: tensors, layers, training, quantization, tokenizer, SLM, GGUF importer, FINF I/O |
| `slm_cli`      | binary (`train_transformer`) | Train/generate causal-transformer SLMs; **`run-gguf`** imports & runs Llama/Qwen GGUFs |
| `train_cli`    | binary (`train_cli`)         | Train tabular MLP classifiers/regressors from any CSV               |
| `tabular_wasm` | cdylib + rlib                | `wasm-bindgen` bindings for running models in the browser           |
| `tests`        | integration                  | Cross-crate integration and regression tests                        |
| `ferrum_gui`   | Tauri app (excluded crate)   | Cross-platform GUI (HTML/CSS/vanilla JS) for the whole project — see [ferrum_gui/README.md](ferrum_gui/README.md) |

`ferrum_gui` is deliberately **outside** the workspace (it pulls heavy system
WebView libraries); build it from its own directory.

---

## Quick start

### Train a Small Language Model (byte-level BPE)

```bash
# Train a causal transformer SLM with a 512-token BPE vocabulary (the default).
cargo run -p slm_cli -- train corpus.txt model.bin --epochs 200 --context 16

# Continue a prompt (add --stream to watch it type live).
cargo run -p slm_cli -- generate model.bin "Once upon a time" --chars 300 --temp 0.7

# Score held-out text, and inspect the file.
cargo run -p slm_cli -- eval model.bin heldout.txt
cargo run -p slm_cli -- info model.bin
```

`--vocab 0` selects the character-level tokenizer; any value `>= 256` trains a
byte-level BPE tokenizer of that size and stores its merge list inside the model.
`--weight_decay` and `--dropout` enable AdamW decoupled decay and FFN dropout.

### Use the library directly

```rust
use ferrum_core::{GenerativeSLM, Rng};

let corpus = std::fs::read_to_string("corpus.txt").unwrap();
let mut rng = Rng::new(1337);

// context 16, embed 32, 4 heads, 2 blocks, FFN 64, 200 epochs,
// Adam lr 0.01, batch 16, BPE vocab 512.
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

### Run an external GGUF model (Llama/Qwen)

```bash
# Imports the checkpoint + its own tokenizer and generates.
# --quant int4|int8|f32 picks the in-memory precision (default int4):
#   int4 = smallest RAM, int8 = fastest decode, f32 = no second quantization.
# A /proc/meminfo guard warns before loading a model that won't fit.
cargo run -p slm_cli -- run-gguf model.gguf "Once upon a time" --quant int4 --max 64
```

Only `llama`/`qwen2` architectures load, and `Q2_K`/`Q3_K`/`IQ*` files are still
rejected. On this class of CPU a 1B model decodes at only **a few tokens per
second** with tens of seconds of prefill — see [ferrum_review.md](ferrum_review.md)
§4 and [benchmarks.md](benchmarks.md) §4 for the measured ceiling and the math
behind it.

### Export a model back to GGUF

```bash
# Re-quantize a stock GGUF (e.g. Q4_K download → Q8_0).
cargo run -p slm_cli -- export-gguf in.gguf out.gguf --quant q8_0

# Export a fine-tuned model: weights come from the checkpoint, the tokenizer
# and hyperparameters are copied from the source GGUF.
cargo run -p slm_cli -- export-gguf base.gguf tuned.gguf --resume tuned.flck --quant q6_k
```

Ferrum writes GGUF v3 at `f32/f16/q8_0/q8_1/q4_0/q4_1/q4_k/q5_k/q6_k`. Norms and
biases stay f32; a weight matrix whose row length is not block-aligned for the
chosen quant is stored f16 (with a note). Only `llama`/`qwen2` models export
(the only architectures that run in the GGUF ecosystem).

---

## Architecture at a glance

```
Tensor ──► ops (matmul, qlinear, softmax, layernorm, …)  ──► parallel (worker pool)
     │
     └──► Layer trait
            ├── Linear            (y = xW + b; optional packed int8/int4 QWeight)
            ├── ActivationLayer   (ReLU / Softmax / …)
            ├── LayerNorm         (per-row normalization)
            ├── Embedding         (token + positional lookup)
            ├── Flatten           (sequence → row)
            └── TransformerBlock  (causal multi-head self-attention + FFN)

Sequential ──► ordered pipeline of Layers   (+ KvCache for fast generation)

tokenizer  ──► ByteBpeTokenizer  (byte-level BPE; char-level fallback)
quant      ──► int8 / int4 (split-half) fake-quant for QAT + in-memory QWeight
train / train_transformer ──► Net (MLP), TransformerNet, Adam(+AdamW)/Sgd
slm        ──► GenerativeSLM: train / train_embedded / train_transformer / generate
loader     ──► FINF v4 (f32) / v5 (int8 + int4, per-tensor or per-channel)

── imported architecture (a second, distinct Transformer stack) ──
gguf       ──► std-only GGUF reader (incl. Q4_K/Q5_K/Q6_K) + tokenizer import
llm        ──► LlamaModel: RMSNorm, RoPE, GQA, SwiGLU, KV-cached decode
llm_train  ──► gradient-checked backprop + SGD train_step for the Llama stack
```

**Why two Transformer stacks?** Ferrum's own (`train_transformer`: learned
positions, LayerNorm, ReLU FFN, dense attention) is what you *train from your
text*; the imported one (`llm`: RoPE, RMSNorm, SwiGLU, grouped-query attention)
is what a downloaded Llama/Qwen GGUF *actually is*. They share the low-level
kernels and quant grid but nothing above that — conflating them is the easiest
way to over-claim. See [docs/manual.md](docs/manual.md) for the full reference.

---

## The three SLM training paths

All three are quantization-aware and share one generation API and one file
format, so you can compare them on the same corpus.

| Method                          | Architecture                         | Tokenizer            | Best for                                  |
|---------------------------------|--------------------------------------|----------------------|-------------------------------------------|
| `GenerativeSLM::train`          | flat one-hot MLP                     | character-level      | the simplest, most transparent baseline   |
| `GenerativeSLM::train_embedded` | embedding + MLP                      | char or **BPE**      | small, fast models that beat one-hot size |
| `GenerativeSLM::train_transformer` | causal multi-head Transformer     | char or **BPE**      | the highest quality on real text          |

The `vocab_size` argument selects the tokenizer (`0` = character-level,
`>= 256` = byte-level BPE). Values in `1..256` are rejected: the 256-byte base
is irreducible.

---

## Documentation

| Document                          | Contents                                            |
|-----------------------------------|-----------------------------------------------------|
| [installation.md](installation.md)| Build, install, and toolchain requirements          |
| [instructions.md](instructions.md)| End-to-end SLM build walkthrough (train → eval → ship) |
| [howtouse.md](howtouse.md)        | CLI and library usage guide (incl. `run-gguf`)       |
| [example.md](example.md)          | End-to-end worked examples                           |
| [usecases.md](usecases.md)        | Ten scenarios where Ferrum is a good fit            |
| [evaluation.md](evaluation.md)    | How to measure quality, size, and speed             |
| [benchmarks.md](benchmarks.md)    | Measured CPU parallelism + quantized-decode benchmarks |
| [ferrum_review.md](ferrum_review.md) | Deep project review incl. the 1B+ feasibility analysis |
| [deployment.md](deployment.md)    | Shipping models to edge, embedded, and WASM targets |
| [docs/manual.md](docs/manual.md)  | Complete API and format reference                   |
| [docs/user_guide.md](docs/user_guide.md) | Task-oriented walkthroughs                   |
| [docs/how_to_use.md](docs/how_to_use.md) | Browser/WASM playground tutorial             |
| [docs/FAQs.md](docs/FAQs.md)      | Frequently asked questions                          |
| [manual/](manual/README.md)       | Beginner's manual — AI/Rust/Ferrum from zero, honestly |
| [status.md](status.md)            | Project status and roadmap                          |

---

## Building and testing

```bash
cargo build --workspace            # build everything
cargo test  --workspace            # run all unit + integration tests
cargo bench --bench gemm           # the matmul/decode microbenchmark (std-only)
cargo doc   -p ferrum_core --open  # browse the API docs
```

---

## License

Licensed under the **MIT** license. See [LICENSE](LICENSE).
