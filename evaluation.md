# Evaluation Guide

How to measure a Ferrum model along the three axes that matter at the edge:
**quality**, **size**, and **speed**. Everything here uses only the library and
the CLIs — no external benchmarking tools.

---

## 1. Training loss

Every training path reports per-epoch loss through a callback. For the SLM paths,
loss is next-token cross-entropy averaged over all positions:

```rust
let mut losses = Vec::new();
let slm = GenerativeSLM::train_transformer_with_callback(
    corpus, 16, 32, 4, 2, 64, 200, 0.01, 16, 512, &mut rng,
    |_, loss| losses.push(loss),
).unwrap();
println!("start {:.4} → end {:.4}", losses[0], losses.last().unwrap());
```

A healthy run shows loss dropping steadily. On a repetitive corpus it should fall
by at least half within a few dozen epochs; the integration tests assert exactly
this as a regression guard. But falling loss only proves the model is *fitting* —
the more important question is §1a.

---

## 1a. Held-out perplexity (the honesty check)

Training loss measures fit; perplexity on text the model never saw measures
*generalization*. `GenerativeSLM::evaluate` slides the context window across
held-out text and scores the probability assigned to each actual next token,
using the trained int8-aware weights:

```rust
let eval = slm.evaluate(&held_out_text)?;
println!("perplexity   = {:.3}", eval.perplexity);     // lower is better; 1.0 is ideal
println!("cross-entropy= {:.3} nats", eval.cross_entropy);
println!("bits/token   = {:.3}", eval.bits_per_token);
println!("scored {} predictions", eval.num_predictions);
```

The number to compare against is the **learned-nothing baseline**: a uniform
model scores a perplexity equal to the vocabulary size. A trained model far below
that has captured real structure. The CLI exposes the same metric:

```bash
train_transformer eval model.bin heldout.txt
```

```text
Predictions  : 72
Cross-entropy: 1.3112 nats/token
Bits/token   : 1.8917
Perplexity   : 3.7107
```

A large gap between training perplexity (near 1.0 on a small corpus) and held-out
perplexity is the textbook signature of **memorization** — the engine telling you
the data is too small or too repetitive, not that the model is too small.
`evaluate` works for all model families (one-hot MLP, embedded, transformer, BPE)
and requires text longer than the context window.

---

## 2. Generation quality

There is no GPU-scale benchmark here; evaluate generation qualitatively and with
simple invariants:

- **Seed fidelity.** `generate(seed, n, …)` always begins with `seed` (the BPE
  tokenizer round-trips the prompt exactly). If it does not, the file is corrupt.
- **Continuation coherence.** On a periodic or templated corpus, low-temperature
  generation (`--temp 0.1`–`0.3`) should reproduce the pattern; raising the
  temperature increases variety.
- **Character budget.** `--chars N` yields the seed plus exactly `N` new
  characters when generation is not cut short — true for both tokenizers.

```bash
train_transformer generate model.bin "the quick brown" --chars 40 --temp 0.2
```

---

## 3. Tokenizer effectiveness

For BPE models, measure how compactly the tokenizer encodes held-out text — this
is *why* BPE often beats character-level on the same network: more text per step.

```rust
use ferrum_core::ByteBpeTokenizer;
let tok = ByteBpeTokenizer::train(&corpus, 512).unwrap();
let sample = "some held-out sentence";
let ratio = sample.len() as f32 / tok.encode(sample).len() as f32;
println!("{ratio:.2} bytes per token"); // higher = better compression
```

`info` reports the learned merge count — a quick check that training found
structure:

```text
Tokenizer : byte-level BPE (298 tokens, 42 merges)
```

---

## 4. Model size

Quantization-aware training lets you ship int8 models ~4× smaller (or int4 ~8×)
than f32 while behaving almost identically.

```rust
let full  = slm.to_bytes()?.len();             // FINF v4/v5, f32 weights
let int8  = slm.to_bytes_quantized()?.len();   // FINF v5, int8 (≈4×)
let int4  = slm.to_bytes_quantized_int4()?.len(); // FINF v5, int4 (≈8×)
println!("{full} → {int8} (int8) → {int4} (int4) bytes");
```

`save()` writes the int8 v5 file. Only tensors of at least 64 values are
quantized; biases and LayerNorm parameters stay f32, which is why the ratio is a
little under the theoretical 4×/8×.

---

## 5. Quantization fidelity

Confirm the quantized file behaves like the trained model by comparing
forward-pass outputs:

```rust
let x = /* a [1, context_len] tensor of token IDs */;
let a = slm.model.forward(&x)?;
let loaded = GenerativeSLM::load("model.bin")?;
let b = loaded.model.forward(&x)?;
for (p, q) in a.data.iter().zip(&b.data) {
    assert!((p - q).abs() < 0.05); // int8 error is bounded by ~half a scale step
}
```

Because training is quantization-aware, this drift stays small. The integration
tests assert it for character-level and BPE models, and for int8 and int4
round-trips. (int4's grid is coarser, so its tolerance is correspondingly wider —
the trade you accept for the extra 2× size win.)

---

## 6. Speed

All work is CPU-only. The matmul kernels run in parallel across cores, so training
and inference speed up on multi-core machines — but only the parallelizable parts.

```bash
time train_transformer train corpus.txt model.bin --epochs 200
time train_transformer generate model.bin "seed text" --chars 500
```

To measure the parallel speedup, pin the thread count with `FERRUM_NUM_THREADS`
and compare — the *results* are identical, only the wall-clock changes:

```bash
FERRUM_NUM_THREADS=1 time train_transformer train corpus.txt m1.bin --epochs 100
time                  train_transformer train corpus.txt mN.bin --epochs 100
# m1.bin and mN.bin are byte-for-byte identical.
```

Two realities worth internalizing (both quantified in
[benchmarks.md](benchmarks.md)): **training** scales with cores but sub-linearly
(the SGD loop, softmax, and tiny per-head matmuls stay serial); **generation**
barely scales, because producing each token is a chain of small, dependent
matmuls — its win from threads is modest. To make generation faster, prefer a BPE
vocabulary (fewer steps per character) and a smaller network rather than more
cores.

---

## 7. Reproducibility checks

Determinism is part of correctness. The same seed must produce identical output:

```rust
let a = slm.generate("seed", 32, 0.7, &mut Rng::new(9))?;
let b = slm.generate("seed", 32, 0.7, &mut Rng::new(9))?;
assert_eq!(a, b);
```

A reloaded model must match the in-memory one for the same generation seed; BPE
training must produce the same merges for the same corpus and `vocab_size`; and
output must be identical across thread counts (the kernels are split, never
re-ordered arithmetically).
