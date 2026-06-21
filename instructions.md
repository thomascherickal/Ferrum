# Building an SLM with Ferrum — A Walkthrough

This document walks through building a Small Language Model (SLM) end-to-end with
Ferrum: preparing a corpus, training a causal Transformer, inspecting it,
generating text, **evaluating held-out quality**, and shipping the int8 model.
It uses only `ferrum_core` and the `slm_cli` binary — no GPU, no external crates.

Every command and number below was produced by actually running the steps on a
small corpus; your numbers will vary with corpus, hyperparameters, and seed.

> **New in this walkthrough.** Two gaps surfaced while building a model the
> "normal" way and were filled in (with tests):
> - `GenerativeSLM::evaluate` + the `eval` CLI subcommand — measure held-out
>   **perplexity / cross-entropy**. Previously you could only watch *training*
>   loss; there was no way to score a finished model on unseen text.
> - `GenerativeSLM::generate_continuation` — return **only the newly generated
>   text**, without the seed prefix glued on.
>
> See [Flagged missing features](#flagged-missing-features-added-here) at the end.

---

## 0. Prerequisites

```bash
cargo build -p slm_cli            # debug build (fast to compile, slow to train)
cargo build -p slm_cli --release  # release build — use this for real training
```

The binary is named `train_transformer`. In the examples below it is invoked as
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

Save it as `corpus.txt`. The corpus must be longer than the context window
(below); a few hundred characters is enough to demonstrate the pipeline, though
real models want far more.

**Hold some text back.** To measure generalization you need text the model was
*not* trained on. Keep a second file, `heldout.txt`, in the same style.

---

## 2. Choose an architecture and tokenizer

The transformer path is selected by `train`. The key knobs:

| Flag        | Meaning                              | Walkthrough value |
|-------------|--------------------------------------|-------------------|
| `--context` | context window (tokens)              | 12                |
| `--embed`   | embedding dimension (÷ heads)        | 32                |
| `--heads`   | attention heads per block            | 4                 |
| `--blocks`  | transformer blocks                   | 2                 |
| `--hidden`  | FFN hidden width                     | 64                |
| `--epochs`  | training epochs                      | 60                |
| `--lr`      | Adam learning rate                   | 0.01              |
| `--batch`   | minibatch size                       | 16                |
| `--vocab`   | BPE vocab (`0` = char-level, ≥256 BPE)| 300              |
| `--seed`    | RNG seed (determinism)               | 7                 |

`--embed` must be divisible by `--heads`. `--vocab 0` uses the character-level
tokenizer; any value `>= 256` trains a byte-level BPE tokenizer of that size and
stores its merge list inside the model file.

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

Loss falls steadily — a healthy run. Training is int8 quantization-aware (QAT),
and the model is saved as an int8-quantized FINF v5 file (~4× smaller than f32).
If `model.bin` already exists, `train` loads it instead of retraining; pass
`--force` to retrain.

> Training is cached on disk. Re-running `train` with the same `model.bin`
> reloads the saved weights and skips training.

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

`44 merges` confirms BPE training found real subword structure. `Input dim`
is the context window; `Output dim` is the vocabulary size.

---

## 5. Generate text

```bash
./target/debug/train_transformer generate model.bin "the quick brown" \
    --chars 60 --temp 0.2
```

```text
the quick brown to school. by evening the streets grow quiet again and the
```

`generate` returns the **seed followed by** the new text. `--chars` counts
characters (true for both char-level and BPE models). Lower `--temp` (0.1–0.3)
reproduces learned patterns; higher temperature adds variety.

### Continuation-only output (library)

When you want just the model's output — a chat reply, an autocomplete
suggestion — use `generate_continuation`, which strips the seed:

```rust
let mut rng = Rng::new(1);
let reply = slm.generate_continuation("the quick brown", 60, 0.2, &mut rng)?;
// reply == " to school. by evening the streets grow quiet again and the"
//   (the seed "the quick brown" is NOT included)
```

`format!("{seed}{}", continuation)` is exactly equal to `generate(seed, …)`.

---

## 6. Evaluate held-out quality

Watching training loss tells you the model is *fitting*; it does not tell you
whether it *generalizes*. Use `eval` to score the model on text it never saw:

```bash
./target/debug/train_transformer eval model.bin heldout.txt
```

```text
Model        : model.bin
Held-out text: heldout.txt  (145 chars)
Predictions  : 72
Cross-entropy: 1.3112 nats/token
Bits/token   : 1.8917
Perplexity   : 3.7107
```

Compare against the training corpus, which the model has memorized:

```bash
./target/debug/train_transformer eval model.bin corpus.txt
# Perplexity : 1.0036   ← near-perfect on seen text
```

**Reading the numbers**

- **Perplexity** — the effective branching factor of the model's predictions.
  Lower is better; the theoretical best is `1.0`. A *uniform* model that learned
  nothing scores the vocabulary size (here ~300). Our held-out 3.7 is far below
  that, so the model learned real structure.
- **Cross-entropy** — mean next-token negative log-likelihood in nats.
- **Bits/token** — the same quantity in base 2 (compression view).

A large gap between training perplexity (~1.0) and held-out perplexity (~3.7) is
the expected signature of a tiny corpus: the model memorizes. Closing that gap is
what more data, regularization, and a smaller model buy you.

### Library equivalent

```rust
let eval = slm.evaluate(&std::fs::read_to_string("heldout.txt")?)?;
println!("perplexity = {:.3}", eval.perplexity);
println!("bits/token = {:.3}", eval.bits_per_token);
println!("scored {} predictions", eval.num_predictions);
```

`evaluate` works for **all** model families (one-hot MLP, embedded, transformer,
and BPE), uses the trained int8-aware weights, and requires the text to be longer
than the context window.

---

## 6a. Train faster with multiple threads

Training is data-parallel: each minibatch is split across worker threads, their
gradients are summed, and one optimizer step is applied. This is built only on
`std::thread::scope` — no external crates, no `unsafe` — and complements the
per-matmul row parallelism already used inside each forward/backward pass.

```bash
# Auto-detect cores (default). Use --threads 1 to force serial training.
./target/debug/train_transformer train corpus.txt model.bin \
    --epochs 60 --context 12 --vocab 300 --threads 4
```

```text
  Threads : 4 (data-parallel minibatch training)
  epoch     1/60   loss = 2.727268
  ...
```

Properties:

- **Reproducible.** For a fixed `--threads` value the result is deterministic —
  gradients are reduced in a fixed shard order, independent of how the OS
  schedules the workers.
- **Serial-identical at one shard.** `--threads 1` is bit-for-bit identical to
  the old serial trainer; the RNG draws the same training windows regardless of
  thread count.
- **QAT-preserving.** Weights are int8-snapped on the shared master before the
  shards fork, exactly as in serial QAT.

Speedups show up on larger models / batches, where forward+backward dominates the
per-step cost of cloning the network for each shard.

### Library equivalent

```rust
let slm = GenerativeSLM::train_transformer_threaded_with_callback(
    &corpus, 12, 32, 4, 2, 64, 60, 0.01, 16, 300,
    0,            // threads: 0 = auto-detect, 1 = serial, N = N workers
    &mut rng, |_, _| {},
)?;
// or, config-driven:
let slm = GenerativeSLM::train_transformer_config_threaded(&corpus, &cfg, 0, &mut rng, cb)?;
```

---

## 6b. Stream generation live

`generate` returns the whole completion at once. For a REPL, a server stream, or
a TUI, use `generate_stream` to receive each fragment as it is produced:

```bash
./target/debug/train_transformer generate model.bin "the quick brown" \
    --chars 50 --temp 0.2 --stream      # prints the text as it appears
```

```rust
use std::io::Write;
let mut rng = Rng::new(1);
let full = slm.generate_stream("the quick brown", 50, 0.2, &mut rng, |frag| {
    print!("{frag}");
    let _ = std::io::stdout().flush();
})?;
```

Guarantees: concatenating every fragment yields exactly the continuation
(`full` minus the seed); the returned `String` equals `generate`'s output for the
same RNG; and the seed is never re-emitted. Character-level models emit one
character per step; BPE models emit decoded text as tokens land, holding back a
partial trailing multi-byte character until it completes — so a `U+FFFD`
placeholder is never streamed and then revised.

---

## 7. Ship the model

`model.bin` is self-contained: weights, normalizer, metadata, and the tokenizer
merge list all travel in one int8-quantized FINF v5 file. Load it anywhere
`ferrum_core` runs — server, laptop, Raspberry Pi, or a WASM tab:

```rust
let slm = GenerativeSLM::load("model.bin")?;
let text = slm.generate("the quick brown", 200, 0.7, &mut Rng::new(42))?;
```

A reloaded model evaluates and generates within a tiny tolerance of the
in-memory one (int8 drift is bounded by half a quantization step, kept small by
QAT).

---

## 8. Iterate

To improve held-out perplexity:

1. **More and more-varied data** — the single biggest lever.
2. **Right-size the model** — too large memorizes; too small underfits. Watch the
   train-vs-held-out perplexity gap via `eval`.
3. **Tune `--vocab`** — larger BPE vocabularies pack more text into the same
   context window (fewer steps per character) but need more data to learn.
4. **Train longer / tune `--lr`** — until held-out perplexity stops improving.

Re-run `eval` after each change; it is the objective signal that the model is
getting better rather than just memorizing harder.

---

## Flagged missing features (added here)

While doing this walkthrough the following gaps were identified and implemented,
each with tests in `tests/tests/test_slm_library.rs`:

| Gap                                            | Added                                                       | Tests |
|------------------------------------------------|-------------------------------------------------------------|-------|
| No way to measure held-out quality (only training loss was observable) | `GenerativeSLM::evaluate` → `Evaluation { num_predictions, cross_entropy, bits_per_token, perplexity }`, plus the `eval` CLI subcommand. Dispatches over all four model families using the shipped int8-aware weights. | `test_evaluate_*` (memorization, beats-uniform baseline, all-paths, FINF roundtrip, short-text error) |
| `generate` always glues the seed to the output | `GenerativeSLM::generate_continuation` returns only the newly generated characters. | `test_generate_continuation_*` |
| No streaming generation (whole completion returned at once) | `GenerativeSLM::generate_stream(seed, n, temp, rng, on_text)` emits each fragment as it is produced (`generate` now delegates to it); CLI `--stream`. UTF-8-safe for BPE. | `test_generate_stream_*` |
| Single-threaded training (only matmuls were parallel) | Data-parallel minibatch training: `train_transformer_epoch_threaded`, `GenerativeSLM::train_transformer_threaded_with_callback` / `..._config_threaded`, CLI `--threads`. `std::thread::scope`, no `unsafe`, deterministic. | `threaded_*` (unit) + `test_threaded_training_*` (integration) |

All four are pure additions — existing APIs and the FINF format are unchanged,
the serial paths are preserved exactly, and the full pre-existing test suite
still passes.

### Still open (documented as future work)

- Larger pre-tokenization vocabularies and merge caching for big corpora.
- WASM streaming bindings (the native `generate_stream` API exists; the browser
  bindings could expose it).
- Additional activation and normalization variants.
