# Building an SLM with Ferrum — A Walkthrough

This document builds a Small Language Model (SLM) end-to-end: preparing a corpus,
training a causal Transformer, inspecting it, generating text, **evaluating
held-out quality**, and shipping the int8 model. It uses only `ferrum_core` and
the `slm_cli` binary — no GPU, no external crates.

The commands and numbers below come from actually running the steps on a small
corpus; treat the exact figures as **representative** — yours vary with corpus,
hyperparameters, and seed. (To instead *run a downloaded* Llama/Qwen model rather
than train your own, see [howtouse.md](howtouse.md) §2.)

---

## 0. Prerequisites

```bash
cargo build -p slm_cli            # debug build (fast to compile, slow to train)
cargo build -p slm_cli --release  # release build — use this for real training
```

The binary is named `train_transformer`. Examples below invoke
`./target/debug/train_transformer`; swap in `target/release/` for real runs —
training is several times faster with `--release` (LTO + `opt-level = 3`).

---

## 1. Prepare a corpus

An SLM learns from raw UTF-8 text. Any plain-text file works; bigger and more
representative is better. For this walkthrough we use a short paragraph:

```text
the quick brown fox jumps over the lazy dog. the calm river flows past green
hills and quiet villages. travelers walk along the winding road, telling stories
of distant lands, bright stars, and the slow turning of the seasons. the old
town wakes at dawn as merchants open their shops and children run to school. by
evening the streets grow quiet again and the lamps glow warm against the dark.
```

Save it as `corpus.txt`. The corpus must be longer than the context window; a few
hundred characters demonstrates the pipeline, though real models want far more —
the data is the dominant lever (see the manual's
[GIGO page](manual/06-data-gigo-and-why-good-data-wins.md)).

**Hold some text back.** To measure generalization you need text the model was
*not* trained on. Keep a second file, `heldout.txt`, in the same style.

---

## 2. Choose an architecture and tokenizer

| Flag         | Meaning                               | Walkthrough value |
|--------------|---------------------------------------|-------------------|
| `--context`  | context window (tokens)               | 12                |
| `--embed`    | embedding dimension (÷ heads)         | 32                |
| `--heads`    | attention heads per block             | 4                 |
| `--blocks`   | transformer blocks                    | 2                 |
| `--hidden`   | FFN hidden width                      | 64                |
| `--epochs`   | training epochs                       | 60                |
| `--lr`       | Adam learning rate                    | 0.01              |
| `--batch`    | minibatch size                        | 16                |
| `--vocab`    | BPE vocab (`0` = char-level, ≥256 BPE)| 300               |
| `--seed`     | RNG seed (determinism)                | 7                 |

`--embed` must be divisible by `--heads`. `--vocab 0` uses character-level
tokenization; any value `>= 256` trains a byte-level BPE tokenizer of that size
and stores its merge list inside the model file. To regularize a small corpus,
add `--weight_decay` (AdamW decoupled decay) and `--dropout` (FFN-hidden dropout).

---

## 3. Train

```bash
./target/debug/train_transformer train corpus.txt model.bin \
    --epochs 60 --context 12 --embed 32 --heads 4 --blocks 2 \
    --hidden 64 --vocab 300 --seed 7
```

Output (abridged):

```text
  epoch     3/60   loss = 2.596811
  epoch     9/60   loss = 0.333096
  epoch    30/60   loss = 0.189039
  epoch    60/60   loss = 0.152412

Trained in 75.18s.
  vocab = 300 BPE tokens   context = 12   output_dim = 300
Saved 40055 bytes → model.bin (int8-quantized FINF v5)
Reload check: OK (6 layers).
```

Loss falls steadily — a healthy run. Training is int8 quantization-aware (QAT) and
the model is saved as an int8-quantized FINF v5 file (~4× smaller than f32). If
`model.bin` already exists, `train` loads it instead of retraining; pass `--force`
to retrain.

### Library equivalent

```rust
use ferrum_core::{GenerativeSLM, Rng, TransformerConfig};

let corpus = std::fs::read_to_string("corpus.txt")?;
let mut rng = Rng::new(7);
let cfg = TransformerConfig {
    context_len: 12, embed_dim: 32, num_heads: 4, num_blocks: 2,
    hidden_dim: 64, epochs: 60, lr: 0.01, batch_size: 16, vocab_size: 300,
};
let slm = GenerativeSLM::train_transformer_config(&corpus, &cfg, &mut rng, |ep, loss| {
    if ep % 10 == 0 { println!("epoch {ep}: loss {loss:.4}"); }
})?;
slm.save("model.bin")?;
```

---

## 4. Inspect the model

```bash
./target/debug/train_transformer info model.bin
```

```text
Model     : model.bin  (40055 bytes)
Format    : FINF v5 (int8-quantized)
Task      : TransformerSLM
Input dim : 12
Output dim: 300
Tokenizer : byte-level BPE (300 tokens, 44 merges)
Layers    : 6
```

`44 merges` confirms BPE training found real subword structure. `Input dim` is the
context window; `Output dim` is the vocabulary size.

---

## 5. Generate text

```bash
./target/debug/train_transformer generate model.bin "the quick brown" \
    --chars 60 --temp 0.2
```

```text
the quick brown to school. by evening the streets grow quiet again and the
```

`generate` returns the **seed followed by** the new text; `--chars` counts
characters (both tokenizers). Lower `--temp` (0.1–0.3) reproduces learned
patterns; higher adds variety. Add `--stream` to print the completion live.

### Continuation-only and streaming (library)

```rust
// Just the model's new text, without the seed prefix:
let reply = slm.generate_continuation("the quick brown", 60, 0.2, &mut Rng::new(1))?;

// Each fragment as it lands (UTF-8-safe for BPE; never emits a half-character):
use std::io::Write;
let full = slm.generate_stream("the quick brown", 50, 0.2, &mut Rng::new(1), |frag| {
    print!("{frag}"); let _ = std::io::stdout().flush();
})?;
```

Concatenating the streamed fragments equals the continuation, and
`format!("{seed}{continuation}")` equals `generate(seed, …)`.

---

## 6. Evaluate held-out quality

Watching training loss tells you the model is *fitting*; it does not tell you
whether it *generalizes*. Score the model on text it never saw:

```bash
./target/debug/train_transformer eval model.bin heldout.txt
```

```text
Predictions  : 72
Cross-entropy: 1.3112 nats/token
Bits/token   : 1.8917
Perplexity   : 3.7107
```

Compare against the training corpus (which the model has memorized):

```bash
./target/debug/train_transformer eval model.bin corpus.txt
# Perplexity : 1.0036   ← near-perfect on seen text
```

**Reading the numbers.** Perplexity is the model's effective branching factor;
lower is better, `1.0` is perfect, and a model that learned *nothing* scores the
vocabulary size (~300 here). Held-out 3.7 is far below that, so the model learned
real structure. The large gap between training (~1.0) and held-out (~3.7)
perplexity is the textbook signature of a tiny corpus: it memorizes. Closing that
gap is what more data, regularization (`--weight_decay`/`--dropout`), and
right-sizing buy you.

### Library equivalent

```rust
let eval = slm.evaluate(&std::fs::read_to_string("heldout.txt")?)?;
println!("perplexity {:.3}, bits/token {:.3}, scored {}",
    eval.perplexity, eval.bits_per_token, eval.num_predictions);
```

`evaluate` works for all model families, uses the trained int8-aware weights, and
requires text longer than the context window.

---

## 7. Train faster with multiple threads

Training is data-parallel: each minibatch is split across worker threads, their
gradients are summed in a fixed order, and one optimizer step is applied — built
only on `std::thread::scope`, no `unsafe`, complementing the per-matmul row
parallelism inside each forward/backward pass.

```bash
# Auto-detect cores (default). --threads 1 forces serial training.
./target/debug/train_transformer train corpus.txt model.bin \
    --epochs 60 --context 12 --vocab 300 --threads 4
```

- **Reproducible.** For a fixed `--threads` value the result is deterministic
  (fixed shard-reduction order).
- **Serial-identical at one shard.** `--threads 1` is bit-for-bit identical to the
  serial trainer.
- **QAT-preserving.** Weights are int8-snapped on the shared master before the
  shards fork.

Speedups show up on larger models/batches, where forward+backward dominates the
per-step cost of cloning the network for each shard; for tiny models that
per-step overhead can dominate, so threads help less. See
[benchmarks.md](benchmarks.md) for measured scaling.

```rust
let slm = GenerativeSLM::train_transformer_threaded_with_callback(
    &corpus, 12, 32, 4, 2, 64, 60, 0.01, 16, 300,
    0,            // 0 = auto-detect, 1 = serial, N = N workers
    &mut rng, |_, _| {},
)?;
```

---

## 8. Ship the model

`model.bin` is self-contained: weights, normalizer, metadata, and the tokenizer
merge list travel in one int8-quantized FINF v5 file. Load it anywhere
`ferrum_core` runs — server, laptop, Raspberry Pi, or a WASM tab:

```rust
let slm = GenerativeSLM::load("model.bin")?;
let text = slm.generate("the quick brown", 200, 0.7, &mut Rng::new(42))?;
```

A reloaded model evaluates and generates within a tiny tolerance of the in-memory
one (int8 drift is bounded by ~half a quantization step, kept small by QAT). For
an even smaller artifact, serialize int4 with `to_bytes_quantized_int4()` (≈8×).

---

## 9. Iterate

To improve held-out perplexity:

1. **More and more-varied data** — the single biggest lever.
2. **Right-size the model** — too large memorizes; too small underfits. Watch the
   train-vs-held-out gap via `eval`.
3. **Regularize** — `--weight_decay` / `--dropout` when the gap is wide.
4. **Tune `--vocab`** — larger BPE vocabularies pack more text into the same
   context window but need more data to learn.
5. **Train longer / tune `--lr`** — until held-out perplexity stops improving.

Re-run `eval` after each change; it is the objective signal that the model is
getting better rather than memorizing harder.

---

## Capabilities used here

This walkthrough exercises features that each carry tests in
`tests/tests/test_slm_library.rs`: held-out evaluation
(`GenerativeSLM::evaluate` + the `eval` subcommand), seed-stripped output
(`generate_continuation`), streaming generation (`generate_stream` + `--stream`,
UTF-8-safe for BPE), and data-parallel threaded training
(`train_transformer_threaded_with_callback`, `--threads`). All are pure additions
over the original serial char-level paths, which are preserved exactly.

### Documented future work

- Larger pre-tokenization vocabularies and merge caching for big corpora.
- WASM streaming bindings (the native `generate_stream` exists).
- Additional activation and normalization variants.
