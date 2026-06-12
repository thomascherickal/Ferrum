# How to Use Ferrum

This guide covers the two command-line tools and the `ferrum_core` library API.
For a narrated, end-to-end walkthrough see [example.md](example.md); for the
complete reference see [docs/manual.md](docs/manual.md).

---

## 1. The SLM trainer (`train_transformer`)

The `slm_cli` crate builds the `train_transformer` binary: a trainer and text
generator for causal-transformer Small Language Models. Training is int8
quantization-aware (QAT) and trained weights are cached on disk, so re-running a
`train`/`run` command loads the saved model instead of retraining (use
`--force` to retrain).

### Commands

```text
train_transformer train    <corpus.txt> <model.bin> [options]
train_transformer run      <corpus.txt> <model.bin> <seed text> [options]
train_transformer generate <model.bin>  <seed text> [options]
train_transformer info     <model.bin>
```

### Train / run options

| Flag         | Default | Meaning                                            |
|--------------|---------|----------------------------------------------------|
| `--context N`| 16      | Context window length                              |
| `--embed N`  | 32      | Embedding dimension (must divide evenly by `--heads`) |
| `--heads N`  | 4       | Attention heads                                    |
| `--blocks N` | 2       | Transformer blocks                                 |
| `--hidden N` | 64      | Feed-forward hidden width                          |
| `--epochs N` | 100     | Training epochs                                    |
| `--lr F`     | 0.01    | Adam learning rate                                 |
| `--batch N`  | 16      | Minibatch size (sequences per step)               |
| `--vocab N`  | 512     | **BPE vocabulary size.** `0` = character-level; `>= 256` = byte-level BPE |
| `--seed N`   | 1337    | RNG seed for deterministic training                |
| `--force`    | —       | Retrain even if the model file exists              |
| `--sample`   | —       | Print a short sample after training                |
| `--verbose`/`-v` | —   | Print all engine internals                         |

### Generate / run options

| Flag           | Default     | Meaning                                          |
|----------------|-------------|--------------------------------------------------|
| `--chars N`    | 200         | Characters to generate (counts characters even for BPE models) |
| `--temp F`     | 0.8         | Sampling temperature (lower = greedier)          |
| `--gen-seed N` | time-based  | RNG seed for generation                          |

### Examples

```bash
# Train with the default 512-token BPE vocabulary.
train_transformer train corpus.txt model.bin --epochs 200 --context 16

# Train a character-level model instead.
train_transformer train corpus.txt model.bin --vocab 0

# Generate a continuation.
train_transformer generate model.bin "Once upon a time" --chars 300 --temp 0.7

# Train (if needed) and immediately generate.
train_transformer run corpus.txt model.bin "Once upon a time" --chars 300

# Inspect the model (tokenizer type, vocab, layers).
train_transformer info model.bin
```

`info` reports whether the model is character-level or byte-level BPE, including
the BPE merge count:

```text
Tokenizer : byte-level BPE (298 tokens, 42 merges)
```

---

## 2. The tabular trainer (`train_cli`)

The `train_cli` binary trains a classical MLP classifier or regressor from any
CSV file. It auto-detects whether the task is classification or regression,
normalizes features, trains, and exports a self-contained FINF model.

```text
train_cli <csv_path> <model_output.bin> [dataset_name] [hidden_size] [epochs] [--verbose]
```

```bash
cargo run -p train_cli -- iris.csv   model.bin "Iris"     32 500
cargo run -p train_cli -- housing.csv model.bin "Housing" 64 400
```

---

## 3. The library API

### Training an SLM

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

The `vocab_size` argument (the second-to-last on the embedded and transformer
paths) is the tokenizer selector:

- `0` → character-level tokenization (the corpus's sorted character set).
- `>= 256` → a byte-level BPE tokenizer of that target size, trained on the
  corpus and stored inside the model. Values in `1..256` are rejected because
  the 256-byte base vocabulary is irreducible.

### Configuration object

For the transformer path you can pass a `TransformerConfig` instead of a long
argument list:

```rust
use ferrum_core::{GenerativeSLM, Rng, TransformerConfig};

let cfg = TransformerConfig {
    context_len: 16,
    embed_dim: 32,
    num_heads: 4,
    num_blocks: 2,
    hidden_dim: 64,
    epochs: 200,
    lr: 0.01,
    batch_size: 16,
    vocab_size: 512,   // 0 = character-level, >= 256 = BPE
};

let mut rng = Rng::new(1337);
let slm = GenerativeSLM::train_transformer_config("…corpus…", &cfg, &mut rng, |ep, loss| {
    println!("epoch {ep}: loss {loss:.4}");
}).unwrap();
```

### Generating text

```rust
let text = slm.generate("Once upon a time", 200, 0.7, &mut rng).unwrap();
```

`num_chars` always counts **characters**, even for BPE models that generate one
subword token at a time. A seed shorter than the context window is left-padded
automatically for BPE models, so short prompts still work.

### Saving and loading

```rust
slm.save("model.bin").unwrap();               // int8-quantized FINF v5 (≈4× smaller)
let slm = GenerativeSLM::load("model.bin").unwrap();

// Train-once / load-from-disk cache:
let (slm, was_loaded) =
    GenerativeSLM::load_or_train("model.bin", corpus, &cfg, &mut rng, |_, _| {}).unwrap();
```

`save` writes int8-quantized FINF v5; `to_bytes()` produces a full-precision
v4/v5 buffer, and `to_bytes_quantized()` the int8 v5 buffer. The tokenizer's
merge list is serialized in the model metadata, so a loaded BPE model
tokenizes and generates exactly as it did before saving.

### Using the tokenizer directly

```rust
use ferrum_core::ByteBpeTokenizer;

let tok = ByteBpeTokenizer::train("low lower lowest low low", 300).unwrap();
let ids = tok.encode("lowest");
assert_eq!(tok.decode(&ids), "lowest");

// The merge list is the full serializable state.
let state = tok.encode_state();
let restored = ByteBpeTokenizer::from_state(&state).unwrap();
```

---

## 4. Quantization-aware training

Quantization is symmetric per-tensor int8 (`value ≈ i8 × scale`, with
`scale = max|value| / 127`). During QAT, weight tensors are snapped onto the
int8 grid on every forward/backward pass (a straight-through estimator keeps
full-precision master weights), so the int8 file you ship behaves like the model
you trained. Tensors shorter than 64 values (biases, LayerNorm parameters) stay
f32. All three SLM training paths enable QAT automatically; the BPE tokenizer is
orthogonal to it and changes only the token stream and vocabulary size.

---

## 5. Determinism

Training and generation are deterministic for a fixed RNG seed. BPE training is
also deterministic: ties between equally frequent pairs are broken by pair
ordering, so the same corpus and `vocab_size` always yield the same merges.
