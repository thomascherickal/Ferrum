# How to Use Ferrum

This guide covers the two command-line tools, the GGUF runner/exporter, and the
`ferrum_core` library API. For a narrated end-to-end walkthrough see
[example.md](example.md); for the complete reference see
[docs/manual.md](docs/manual.md); for the shorter library-level walkthrough see
[docs/how_to_use.md](docs/how_to_use.md).

---

## 1. The SLM trainer (`train_transformer`)

The `slm_cli` crate builds the `train_transformer` binary: a trainer and text
generator for causal-transformer Small Language Models, plus a `run-gguf`
subcommand for external checkpoints. Training is int8 quantization-aware (QAT)
and trained weights are cached on disk, so re-running a `train`/`run` command
loads the saved model instead of retraining (use `--force` to retrain).

### Commands

```text
train_transformer train    <corpus.txt> <model.bin> [options]
train_transformer run      <corpus.txt> <model.bin> <seed text> [options]
train_transformer generate <model.bin>  <seed text> [options]
train_transformer run-gguf      <model.gguf> [prompt] [options]
train_transformer finetune-gguf <model.gguf> <corpus.txt> <out.flck> [options]
train_transformer export-gguf   <in.gguf> <out.gguf> [options]
train_transformer eval     <model.bin>  <heldout.txt>
train_transformer info     <model.bin>
```

### Train / run options

| Flag             | Default | Meaning                                                |
|------------------|---------|--------------------------------------------------------|
| `--context N`    | 16      | Context window length                                  |
| `--embed N`      | 32      | Embedding dimension (must divide evenly by `--heads`)  |
| `--heads N`      | 4       | Attention heads                                        |
| `--blocks N`     | 2       | Transformer blocks                                     |
| `--hidden N`     | 64      | Feed-forward hidden width                              |
| `--epochs N`     | 100     | Training epochs                                        |
| `--lr F`         | 0.01    | Adam learning rate                                     |
| `--batch N`      | 16      | Minibatch size (sequences per step)                    |
| `--vocab N`      | 512     | BPE vocab size. `0` = character-level; `>= 256` = byte-level BPE |
| `--seed N`       | 1337    | RNG seed for deterministic training                    |
| `--weight_decay F` | 0     | **AdamW** decoupled weight decay (0 = plain Adam)      |
| `--dropout F`    | 0       | FFN-hidden dropout probability during training         |
| `--threads N`    | 0       | Data-parallel worker threads (`0` = auto, `1` = serial)|
| `--force`        | —       | Retrain even if the model file exists                  |
| `--sample`       | —       | Print a short sample after training                    |
| `--verbose`/`-v` | —       | Stream the engine's internal trace                     |

### Generate / run options

| Flag           | Default     | Meaning                                          |
|----------------|-------------|--------------------------------------------------|
| `--chars N`    | 200         | Characters to generate (counts characters even for BPE) |
| `--temp F`     | 0.8         | Sampling temperature (lower = greedier)          |
| `--gen-seed N` | time-based  | RNG seed for generation                          |
| `--stream`     | —           | Print the completion live, fragment by fragment  |

### Examples

```bash
# Train with the default 512-token BPE vocabulary, AdamW decay + dropout.
train_transformer train corpus.txt model.bin --epochs 200 --context 16 \
    --weight_decay 0.01 --dropout 0.1

# Train a character-level model instead.
train_transformer train corpus.txt model.bin --vocab 0

# Generate a continuation, streaming it live.
train_transformer generate model.bin "Once upon a time" --chars 300 --temp 0.7 --stream

# Train (if needed) and immediately generate.
train_transformer run corpus.txt model.bin "Once upon a time" --chars 300

# Score held-out text, and inspect the model.
train_transformer eval model.bin heldout.txt
train_transformer info model.bin
```

`info` reports whether the model is character-level or byte-level BPE, including
the BPE merge count:

```text
Tokenizer : byte-level BPE (298 tokens, 42 merges)
```

`--threads` controls **training** parallelism (data-parallel minibatches);
`FERRUM_NUM_THREADS` controls the **matmul** worker pool used by both training
and generation. They are independent knobs.

---

## 2. Running an external GGUF model (`run-gguf`)

`run-gguf` imports a quantized **Llama/Qwen** checkpoint *and its own tokenizer*,
then generates text on the CPU. Only `llama`/`qwen2` architectures load; the
supported quant formats are `F32/F16/Q8_0/Q8_1/Q4_0/Q4_1` and the
`Q4_K/Q5_K/Q6_K` k-quants (`Q2_K`/`Q3_K`/`IQ*` are rejected with a clear error).

```text
train_transformer run-gguf <model.gguf> [prompt] [options]
```

| Flag           | Default | Meaning                                                       |
|----------------|---------|---------------------------------------------------------------|
| `--quant Q`    | int4    | In-memory precision: `int4` (smallest RAM), `int8` (fastest decode), `f32` (no second quantization) |
| `--max N`      | —       | Maximum new tokens to generate                                |
| `--temp F`     | 0.8     | Sampling temperature                                          |
| `--gen-seed N` | —       | RNG seed for generation                                       |
| `--ids`        | —       | Print raw token IDs alongside the decoded text                |

```bash
train_transformer run-gguf model.gguf "Once upon a time" --quant int4 --max 64
```

**What to expect.** Before loading, a `/proc/meminfo`-based guard estimates the
resident footprint and warns if it will not fit. Decode is **bandwidth-bound**:
each token streams every weight once, so a 1B model runs at only a few tokens per
second and a real prompt takes tens of seconds to prefill (compute-bound). `int8`
gives the fastest per-call decode; `int4` halves the RAM at a modest speed cost;
`f32` avoids re-quantizing an already-quantized file but is the largest and is the
precision required for training (see §3.4). The import is lossy by default
(dequantize → re-quantize to Ferrum's per-row grid) and is **not** bit-exact to
llama.cpp. See [benchmarks.md](benchmarks.md) §4 and [ferrum_review.md](ferrum_review.md) §4.

### Fine-tuning (`finetune-gguf`)

`finetune-gguf` loads the checkpoint at f32 (training needs full-precision
masters), fine-tunes on a text corpus, and writes a `.flck` checkpoint that
`run-gguf --resume` and `export-gguf --resume` can apply:

```bash
train_transformer finetune-gguf model.gguf corpus.txt tuned.flck --epochs 3
```

See the [readme](readme.md) for the full flag list (`--lr`, `--batch`, `--seq`,
`--warmup`, `--clip`, `--weight_decay`, `--dropout`, `--qat`, `--sample`, …).

### Exporting back to GGUF (`export-gguf`)

`export-gguf` (alias: `export`) writes a loaded — and optionally fine-tuned —
model back out as a **GGUF v3 file that runs in llama.cpp / ollama / LM Studio**,
carrying the source's hyperparameters and tokenizer forward verbatim:

```bash
# Re-quantize a stock GGUF (e.g. a Q4_K download → Q8_0).
train_transformer export-gguf in.gguf out.gguf --quant q8_0

# Export a fine-tune: weights from the checkpoint, metadata from the source.
train_transformer export-gguf base.gguf tuned.gguf --resume tuned.flck --quant q6_k
```

Output types: `f32 | f16 | q8_0 | q8_1 | q4_0 | q4_1 | q4_k | q5_k | q6_k`
(default `q8_0`). Norms and biases always stay f32. A weight matrix whose row
length is not block-aligned for the chosen quant (32 for the legacy formats,
256 for k-quants) is stored **f16** instead — the per-type tensor summary
printed after export shows exactly what was emitted. Re-quantizing an
already-quantized source goes through f32 and is inherently lossy; the lossless
paths are `f32`/`f16` or exporting fine-tuned f32 masters.

---

## 3. The tabular trainer (`train_cli`)

`train_cli` trains a classical MLP classifier or regressor from any CSV file. It
auto-detects classification vs. regression, normalizes features, trains, and
exports a self-contained FINF model that embeds the feature names, ranges, and
class labels.

```text
train_cli <csv_path> <model_output.bin> [dataset_name] [hidden_size] [epochs] [--verbose]
```

```bash
cargo run -p train_cli -- datasets/tabular/iris.data    model.bin "Iris"    32 500
cargo run -p train_cli -- datasets/tabular/housing.csv  model.bin "Housing" 64 400
```

---

## 4. The library API

### 4.1 Training an SLM

`GenerativeSLM` exposes three training paths. All are quantization-aware and
share one generation API and one file format.

```rust
use ferrum_core::{GenerativeSLM, Rng};
let mut rng = Rng::new(1337);
let corpus = "…your text…";

// (a) One-hot MLP — character-level, simplest baseline.
let a = GenerativeSLM::train(corpus, 8, 64, 100, 0.05, 0.9, 16, &mut rng);

// (b) Embedding MLP — char (vocab 0) or BPE (vocab >= 256).
let b = GenerativeSLM::train_embedded(corpus, 8, 16, 64, 100, 0.05, 0.9, 16, 512, &mut rng);

// (c) Causal Transformer — char (vocab 0) or BPE (vocab >= 256).
let c = GenerativeSLM::train_transformer(corpus, 16, 32, 4, 2, 64, 200, 0.01, 16, 512, &mut rng);
```

`vocab_size` (second-to-last on the embedded and transformer paths) selects the
tokenizer: `0` → character-level; `>= 256` → a byte-level BPE tokenizer of that
size, trained on the corpus and stored inside the model. Values in `1..256` are
rejected because the 256-byte base vocabulary is irreducible.

### 4.2 Configuration object

```rust
use ferrum_core::{GenerativeSLM, Rng, TransformerConfig};

let cfg = TransformerConfig {
    context_len: 16, embed_dim: 32, num_heads: 4, num_blocks: 2,
    hidden_dim: 64, epochs: 200, lr: 0.01, batch_size: 16,
    vocab_size: 512,   // 0 = character-level, >= 256 = BPE
};
let mut rng = Rng::new(1337);
let slm = GenerativeSLM::train_transformer_config("…corpus…", &cfg, &mut rng, |ep, loss| {
    println!("epoch {ep}: loss {loss:.4}");
}).unwrap();
```

Threaded variants (`train_transformer_threaded_with_callback`,
`train_transformer_config_threaded`) take a thread count (`0` = auto).

### 4.3 Generating, saving, loading

```rust
let text = slm.generate("Once upon a time", 200, 0.7, &mut rng).unwrap();

slm.save("model.bin").unwrap();               // int8-quantized FINF v5 (≈4× smaller)
let slm = GenerativeSLM::load("model.bin").unwrap();

// Train-once / load-from-disk cache:
let (slm, was_loaded) =
    GenerativeSLM::load_or_train("model.bin", corpus, &cfg, &mut rng, |_, _| {}).unwrap();
```

`num_chars` always counts **characters**, even for BPE models that emit one
subword token at a time; short prompts are left-padded for BPE. `to_bytes()`
produces an f32 v4/v5 buffer, `to_bytes_quantized()` the int8 v5 buffer, and
`to_bytes_quantized_int4()` an int4 v5 buffer (≈8× smaller). Streaming
(`generate_stream`) and continuation-only (`generate_continuation`) variants
exist too.

### 4.4 Importing and training a GGUF model

```rust
use ferrum_core::{Gguf, GgufTokenizer, LlamaTrainer, QKind, Rng};

let gguf  = Gguf::open("model.gguf")?;            // streamed reader
let tok   = GgufTokenizer::from_gguf(&gguf)?;     // the checkpoint's own tokenizer
let model = gguf.load_llama(QKind::Int4)?;        // int4/int8 packed, ready to run
let out   = model.generate(&tok.encode("Hello"), 32, 0.7, &mut Rng::new(1))?;
println!("{}", tok.decode(&out));

// To fine-tune, import at f32 (training needs f32 masters) and step SGD:
let f32_model = Gguf::open("model.gguf")?.load_llama_prec(None)?;
let mut trainer = LlamaTrainer::new(f32_model)?;  // errs if any weight is quantized
let loss = trainer.train_step(&tok.encode("training text"), 1e-3)?;
```

The backward pass is finite-difference-checked per primitive (RMSNorm, RoPE,
GQA+softmax, SwiGLU, embedding, LM head). It makes *small* imported models
trainable; a 1B model is out of reach on a single CPU (RAM + compute — see
[ferrum_review.md](ferrum_review.md) §4.3).

### 4.5 Using the tokenizer directly

```rust
use ferrum_core::ByteBpeTokenizer;

let tok = ByteBpeTokenizer::train("low lower lowest low low", 300).unwrap();
let ids = tok.encode("lowest");
assert_eq!(tok.decode(&ids), "lowest");
let restored = ByteBpeTokenizer::from_state(&tok.encode_state()).unwrap();
```

---

## 5. Quantization-aware training

Quantization is symmetric int8 (`value ≈ i8 × scale`, `scale = max|value| / 127`),
available per-tensor and per-channel. During QAT, weight tensors are snapped onto
the int8 grid each step while full-precision masters are kept (a straight-through
estimator), so the int8 file you ship behaves like the model you trained. Tensors
shorter than 64 values (biases, LayerNorm) stay f32. The BPE tokenizer is
orthogonal to QAT — it changes only the token stream and vocabulary size. Models
can be serialized int8 (≈4×) or int4 (≈8×); int4 uses a **split-half** packing so
its decode kernel vectorizes (see §6 and [benchmarks.md](benchmarks.md) §4d).

---

## 6. CPU parallelism

Training and inference are multi-threaded on the CPU — no GPU. The matmul kernels
behind every Linear, FFN, attention, and LM-head step split across cores.

- **Dynamic detection.** Worker count comes from
  `std::thread::available_parallelism()`, resolved once and cached.
- **Override.** `FERRUM_NUM_THREADS=N` pins it (`=1` forces serial).
- **Persistent pool.** Threads are spawned once and reused, so autoregressive
  generation pays no per-call thread-creation cost.
- **Column split for decode.** A single-token decode is `m = 1`, which plain
  row-splitting cannot parallelize; the quantized path splits that GEMV across
  cores **by output column** instead.
- **Deterministic.** The split never changes per-element arithmetic, so output is
  bit-for-bit identical at any thread count. Small workloads and `wasm32` run
  serially.

```bash
FERRUM_NUM_THREADS=4 train_transformer train corpus.txt model.bin --epochs 200
```

```rust
println!("using {} CPU threads", ferrum_core::num_threads());
```

## 7. Determinism

Training and generation are deterministic for a fixed RNG seed, **independent of
thread count**. BPE merge learning is deterministic too: ties between equally
frequent pairs break by pair ordering, so the same corpus and `vocab_size` always
yield the same merges.
