# Build Your Own SLM with Ferrum

This walkthrough builds a small language model two ways:

- **Path A — the 10-minute SLM**: `GenerativeSLM::train`, a character-level causal
  MLP trained with one function call. Best for autocomplete-style models on small,
  domain-specific corpora (commands, poetry templates, slogans, logs).
- **Path B — a real Transformer**: `GenerativeSLM::train_transformer` trains a
  decoder-only causal transformer (embeddings, multi-head attention, FFN) end-to-end
  with Adam, then serializes it to FINF and runs it natively or in the browser with
  KV-cached generation and attention visualisation.

Everything below uses only `ferrum_core` — no external crates.

---

## Setup

```toml
# Cargo.toml
[dependencies]
ferrum_core = { path = "ferrum/ferrum_core" }   # adjust to your layout
```

```rust
use ferrum_core::{GenerativeSLM, Rng};
```

---

## Path A — Train a Generative SLM in One Call

### 1. Get a corpus

Any plain text works. Small and domain-specific is the sweet spot — the model learns
character-level patterns, so repetitive structure (commands, templates, verse) trains
fast and generates convincingly.

```rust
let corpus = "\
git status\n\
git commit -m 'update'\n\
git push origin main\n\
git pull --rebase\n\
git checkout -b feature\n\
cargo build --release\n\
cargo test --workspace\n\
cargo run -p train_cli\n\
";
```

### 2. Train

```rust
let mut rng = Rng::new(42);          // deterministic: same seed → same model

let slm = GenerativeSLM::train(
    corpus,
    8,      // context_len  — characters of context the model sees
    64,     // hidden_size  — MLP hidden width
    300,    // epochs
    0.05,   // learning rate
    0.9,    // momentum
    16,     // batch size
    &mut rng,
)?;
```

What happens inside: the corpus is windowed into (8-char context → next char) pairs,
each character is one-hot encoded against the discovered vocabulary
(`input_dim = context_len × vocab_size`), and a `input → hidden(ReLU) → vocab` MLP is
trained with fused softmax cross-entropy and SGD+momentum.

Want progress reporting? Use the callback variant:

```rust
let slm = GenerativeSLM::train_with_callback(
    corpus, 8, 64, 300, 0.05, 0.9, 16, &mut rng,
    |epoch, loss| {
        if epoch % 50 == 0 { println!("epoch {epoch:4}  loss {loss:.4}"); }
    },
)?;
```

And for deep diagnostics (shapes, activation stats, NaN detection at every layer):

```rust
ferrum_core::set_verbose(true);
```

### 3. Generate

```rust
// seed text, number of chars to generate, temperature, rng
let out = slm.generate("git c", 40, 0.5, &mut rng)?;
println!("{out}");   // e.g. "git checkout -b feature\ngit commit -m '…"
```

Temperature guide:

| temp | behaviour |
|------|-----------|
| 0.1–0.3 | nearly deterministic — picks the most likely continuation |
| 0.7–1.0 | balanced sampling |
| > 1.5   | chaotic; useful for testing vocabulary coverage |

Note: the seed must be at least `context_len` characters, otherwise generation stops
immediately (the model needs a full context window).

### 4. Save and reload

```rust
std::fs::write("my_slm.bin", slm.to_bytes()?)?;            // FINF v4, self-contained

let bytes = std::fs::read("my_slm.bin")?;
let slm2 = GenerativeSLM::from_bytes(&bytes)?;
let again = slm2.generate("cargo t", 30, 0.3, &mut rng)?;
```

The binary embeds the weights, the (identity) normalizer, and metadata including the
vocabulary (`meta.class_names`, hex-encoded chars), so nothing else needs to ship.

### Sizing cheat sheet

`params ≈ (context_len × vocab × hidden) + (hidden × vocab)`.
A 40-char vocabulary with context 8 and hidden 64 is ~23k params ≈ 92 KB of f32 —
small enough for instant load in a browser.

| corpus | context_len | hidden | epochs |
|---|---|---|---|
| < 5 KB, very repetitive | 6–8 | 32–64 | 200–500 |
| 5–50 KB | 8–16 | 64–128 | 300–800 |
| > 50 KB | 16–32 | 128–256 | 500+ (be patient: CPU-only) |

---

## Path B — Train a Causal Transformer

### The one-call API

```rust
use ferrum_core::{GenerativeSLM, Rng};

let corpus = std::fs::read_to_string("corpus.txt")?;
let mut rng = Rng::new(42);

let slm = GenerativeSLM::train_transformer_with_callback(
    &corpus,
    16,    // context_len  — tokens per window
    32,    // embed_dim    — must be divisible by num_heads
    4,     // num_heads
    2,     // num_blocks   — transformer depth
    64,    // hidden_dim   — FFN inner width (2–4 × embed_dim)
    100,   // epochs
    0.003, // learning rate (Adam)
    16,    // batch_size
    &mut rng,
    |ep, loss| { if ep % 10 == 0 { println!("epoch {ep:4}  loss {loss:.4}"); } },
)?;

let text = slm.generate("once upon a time", 200, 0.8, &mut rng)?;
std::fs::write("transformer.bin", slm.to_bytes()?)?;   // FINF v4, WASM-ready
```

This trains token + positional embeddings, every attention/FFN/LayerNorm parameter,
and the LM head end-to-end with next-token loss at all positions — verified by
finite-difference gradient checks in the test suite. Unlike Path A, inputs are
compact token IDs (`input_dim = context_len`), so models stay small as the
vocabulary grows.

For lower-level control (custom training loops, your own data pipeline), use
`ferrum_core::TransformerNet` directly: `forward` → `softmax_cross_entropy` →
`backward` → `step(&Adam)`, then `to_inference()` to export.

### Hand-assembling instead (optional)

If you want to port weights from elsewhere or experiment with custom
architectures, you can also build the same model layer by layer — everything
below serializes to FINF and runs in WASM.

### 1. Build the model

```rust
use ferrum_core::{
    Embedding, LayerNorm, Linear, ModelMetadata, Normalizer, Rng,
    Sequential, TaskType, Tensor, TransformerBlock, save,
};

let vocab_size  = 40;   // your character set
let context_len = 16;   // T — tokens per forward pass
let embed_dim   = 32;   // C — must be divisible by num_heads
let num_heads   = 4;
let hidden_dim  = 64;   // FFN inner width (usually 2–4 × embed_dim)

let mut rng = Rng::new(7);
let scale = (1.0 / embed_dim as f32).sqrt();
let mut randn = |n: usize| -> Vec<f32> {
    (0..n).map(|_| rng.next_normal() * scale).collect()
};

// Token + learned positional embeddings
let emb = Embedding::new(
    vocab_size, context_len, embed_dim,
    randn(vocab_size * embed_dim),     // token table   [vocab, C]
    randn(context_len * embed_dim),    // position table [T, C]
)?;

// One pre-norm causal block: LN → MHA → residual → LN → FFN → residual
let block = TransformerBlock::new(
    context_len, num_heads, embed_dim,
    vec![1.0; embed_dim], vec![0.0; embed_dim],            // ln1 γ, β
    randn(embed_dim * embed_dim), vec![0.0; embed_dim],    // Q
    randn(embed_dim * embed_dim), vec![0.0; embed_dim],    // K
    randn(embed_dim * embed_dim), vec![0.0; embed_dim],    // V
    randn(embed_dim * embed_dim), vec![0.0; embed_dim],    // out proj
    vec![1.0; embed_dim], vec![0.0; embed_dim],            // ln2 γ, β
    randn(embed_dim * hidden_dim), vec![0.0; hidden_dim],  // FFN up
    randn(hidden_dim * embed_dim), vec![0.0; embed_dim],   // FFN down
)?;
// Note: the constructor rejects num_heads = 0 and embed_dim % num_heads != 0.

let ln_f    = LayerNorm::new(embed_dim, vec![1.0; embed_dim], vec![0.0; embed_dim])?;
let lm_head = Linear::new(embed_dim, vocab_size,
                          randn(embed_dim * vocab_size), vec![0.0; vocab_size])?;

let model = Sequential::new()
    .with(Box::new(emb))
    .with(Box::new(block))      // stack more blocks here for depth
    .with(Box::new(ln_f))
    .with(Box::new(lm_head));

println!("{}", model.summary());
```

### 2. Run a forward pass

The embedding takes a row of **token IDs** (as f32) and emits `[T, C]`; the LM head
emits `[T, vocab]` — one next-token distribution per position. The **last row** is
your next-token prediction:

```rust
let context: Vec<f32> = vec![3.0, 17.0, 5.0, 1.0, 9.0, 2.0, 8.0, 4.0,
                             3.0, 17.0, 5.0, 1.0, 9.0, 2.0, 8.0, 4.0]; // T ids
let x = Tensor::matrix(1, context_len, context)?;
let logits = model.forward(&x)?;                 // [T, vocab]
let (rows, cols) = logits.matrix_dims()?;
let next = &logits.data[(rows - 1) * cols..];    // last position's logits
```

After any forward pass you can read the attention maps for visualisation:

```rust
use ferrum_core::Layer;
// block.last_attention: RefCell<Vec<f32>> with shape [batch × heads × T × T]
```

### 3. Save it (FINF v4 handles all transformer layers)

```rust
let vocab: Vec<String> = ('a'..='z').chain([' ', '\n']).take(vocab_size)
    .map(|c| c.to_string()).collect();

let meta = ModelMetadata {
    dataset_name: "my_transformer".into(),
    task: TaskType::TransformerSLM,
    feature_names: (0..context_len).map(|i| format!("c_{i}")).collect(),
    feature_ranges: vec![[0.0, vocab_size as f32]; context_len],
    class_names: vocab,                        // idx → token mapping
    target_name: "next_char".into(),
    target_range: [0.0, vocab_size as f32],
    input_dim: context_len,
    output_dim: vocab_size,
};
let norm = Normalizer { means: vec![], stds: vec![] };  // identity for SLMs

save(&model, &norm, &meta, "transformer.bin")?;
```

> **Tip:** for training from scratch, prefer `GenerativeSLM::train_transformer`
> (above) — hand-assembly is for weight porting and architecture experiments.

### 4. Run it in the browser

```bash
bash scripts/build_wasm.sh
```

```js
const bytes = new Uint8Array(await (await fetch('transformer.bin')).arrayBuffer());
const slm   = new TransformerSLMModel(bytes);

const meta  = JSON.parse(slm.metadata());
const vocab = meta.class_names;                       // idx → token

const ctx   = new Float32Array(slm.context_len());    // fill with token ids
const probs = slm.predict_next(ctx);                  // Float32Array [vocab]

const idx   = slm.sample_from_probs(probs, 0.8, Math.random());
const tok   = vocab[idx];

// Extras for UIs:
const H     = slm.entropy(probs);                     // uncertainty meter
const top5  = slm.top_k_indices(probs, 5);            // ranked candidates
const attn  = slm.get_last_attention_weights();       // heads × T × T heatmap
```

### Fast generation with the KV cache

`predict_next` re-runs the whole context every call (O(T²) per token). For
interactive streaming, prime the per-block KV caches once and feed one token
at a time at O(T):

```js
let probs = slm.prime(seedIds);              // seedIds: 1..contextLen token ids
const out = [...seedIds];
while (out.length < slm.context_len()) {     // cache holds contextLen positions
  const next = slm.sample_from_probs(probs, 0.8, Math.random());
  out.push(next);
  probs = slm.predict_next_cached(next);
}
// Window full? Re-prime with the most recent tokens and keep going:
// probs = slm.prime(new Float32Array(out.slice(-slm.context_len() + 1)));
```

Both paths produce identical distributions — the cached path is verified
against the full forward pass in the test suite.

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `Corpus length shorter than context window` | corpus has fewer chars than `context_len` |
| Generation returns only the seed | seed shorter than `context_len` chars |
| Loss is NaN / exploding | lower `lr` (try 0.01), enable `set_verbose(true)` to see where |
| `embedding_dim must be divisible by num_heads` | pick `embed_dim` as a multiple of `num_heads` |
| Output looks uniform-random | undertrained — more epochs, or corpus too diverse for model size |
| Model file huge | `input_dim = context_len × vocab_size`; shrink context or vocabulary |

## Reproducibility

All randomness flows through the `Rng` you pass in. Fix the seed and the trained
weights, the generated text, and the saved bytes are identical across runs and
platforms.
