# Evaluation Guide

How to measure a Ferrum model along the three axes that matter at the edge:
**quality**, **size**, and **speed**. Everything here uses only the library and
the CLIs — no external benchmarking tools.

---

## 1. Training loss

Every training path reports per-epoch loss through a callback. For the SLM
paths, loss is next-token cross-entropy averaged over all positions:

```rust
let mut losses = Vec::new();
let slm = GenerativeSLM::train_transformer_with_callback(
    corpus, 16, 32, 4, 2, 64, 200, 0.01, 16, 512, &mut rng,
    |_, loss| losses.push(loss),
).unwrap();

println!("start {:.4} → end {:.4}", losses[0], losses.last().unwrap());
```

A healthy run shows loss dropping steadily. On a repetitive corpus it should
fall by at least half within a few dozen epochs; the integration tests assert
exactly this as a regression guard.

The CLI prints loss at regular intervals during `train`:

```text
  epoch     1/200   loss = 3.214887
  epoch    20/200   loss = 1.002145
  …
```

---

## 2. Generation quality

There is no GPU-scale benchmark here; evaluate generation qualitatively and with
simple invariants:

- **Seed fidelity.** `generate(seed, n, …)` always begins with `seed` (the BPE
  tokenizer round-trips the prompt exactly). If it does not, the model file is
  corrupt.
- **Continuation coherence.** On a periodic or templated corpus, low-temperature
  generation (`--temp 0.1`–`0.3`) should reproduce the pattern; raising the
  temperature increases variety.
- **Character budget.** `--chars N` yields the seed plus exactly `N` new
  characters when generation is not cut short — true for both character-level
  and BPE models.

```bash
train_transformer generate model.bin "the quick brown" --chars 40 --temp 0.2
```

---

## 3. Tokenizer effectiveness

For BPE models, compare how compactly the tokenizer encodes held-out text:

```rust
use ferrum_core::ByteBpeTokenizer;

let tok = ByteBpeTokenizer::train(&corpus, 512).unwrap();
let sample = "some held-out sentence";
let ratio = sample.len() as f32 / tok.encode(sample).len() as f32;
println!("{ratio:.2} bytes per token"); // higher = better compression
```

A good BPE vocabulary on natural-language text compresses several bytes into
each token, which lets a fixed context window cover more text than a
character-level model would.

`info` reports the learned merge count, a quick sanity check that training
actually found structure:

```text
Tokenizer : byte-level BPE (298 tokens, 42 merges)
```

---

## 4. Model size

Quantization-aware training lets you ship int8 models that are roughly 4×
smaller than f32 while behaving almost identically.

```rust
let full  = slm.to_bytes()?.len();            // FINF v4/v5, f32 weights
let quant = slm.to_bytes_quantized()?.len();  // FINF v5, int8 weights
println!("{full} → {quant} bytes ({:.1}×)", full as f32 / quant as f32);
```

`save()` always writes the int8 v5 file. Only tensors of at least 64 values are
quantized; biases and LayerNorm parameters stay f32.

---

## 5. Quantization fidelity

Confirm the int8 file behaves like the trained model by comparing forward-pass
outputs:

```rust
let x = /* a [1, context_len] tensor of token IDs */;
let a = slm.model.forward(&x)?;
let loaded = GenerativeSLM::load("model.bin")?;
let b = loaded.model.forward(&x)?;
for (p, q) in a.data.iter().zip(&b.data) {
    assert!((p - q).abs() < 0.05); // int8 error is bounded by half a scale step
}
```

Because training is quantization-aware, this drift stays small. The integration
tests assert it for both character-level and BPE models.

---

## 6. Speed

All work is single-threaded CPU. Time training and generation with the shell or
`std::time::Instant`:

```bash
time train_transformer train corpus.txt model.bin --epochs 200
time train_transformer generate model.bin "seed text" --chars 500
```

Generation cost scales with context length, embedding dimension, number of
blocks, and (for BPE) how many tokens are needed to reach the requested
character count. To speed up generation, prefer a BPE vocabulary (fewer steps
per character) and a smaller network.

---

## 7. Reproducibility checks

Determinism is part of correctness. The same seed must produce identical output:

```rust
let a = slm.generate("seed", 32, 0.7, &mut Rng::new(9))?;
let b = slm.generate("seed", 32, 0.7, &mut Rng::new(9))?;
assert_eq!(a, b);
```

A reloaded model must match the in-memory one for the same generation seed, and
BPE training must produce the same merges for the same corpus and `vocab_size`.
