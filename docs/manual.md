# Ferrum Reference Manual

The complete technical reference for `ferrum_core`: the public API, the tokenizer,
quantization, the FINF model format, and the GGUF importer. For task-oriented
guides see [user_guide.md](user_guide.md) and the top-level
[howtouse.md](../howtouse.md).

The crate holds **two distinct Transformer stacks**. Sections 1–7 describe
Ferrum's *own* architecture (what you train from your text); section 8 describes
the *imported* Llama/Qwen stack (what a downloaded GGUF actually is). They share
the kernels in §1 but nothing above them.

---

## 1. Tensors and ops

`Tensor` is a flat `Vec<f32>` plus a shape. Construct rows and matrices with
`Tensor::row`, `Tensor::matrix`; query a 2-D shape with `matrix_dims()`. The
`ops` module provides matmul, a fused/cache-tiled `linear_forward`, a packed
`qlinear` (int8/int4), `softmax_rows`, and `argmax_rows`.

### CPU parallelism

The matmul kernels split their output rows across a **persistent worker pool**
(the `parallel` module), built only on `std` (threads, channels, `Arc`) with no
`unsafe`. Worker threads are spawned once and reused, so autoregressive
generation — thousands of small matmuls — pays no per-call thread-creation cost.
Because safe Rust cannot hand a borrowed closure to threads that outlive the
call, each kernel shares its read-only inputs via `Arc` (one clone per matmul,
not per worker) and every worker returns an owned output block the caller
stitches together.

A single-token decode is `m = 1` (one activation row), which a row-split cannot
parallelize. The quantized path therefore also offers a **column split**
(`run_1d`): it divides the output columns across workers instead, so the
bandwidth-bound decode GEMV uses every core.

Worker count is detected once from `std::thread::available_parallelism()`,
overridable with `FERRUM_NUM_THREADS`; query it with `ferrum_core::num_threads()`.
Workloads below an internal work threshold, and the `wasm32` target, run
serially. The split never changes per-element arithmetic, so output is
bit-for-bit identical regardless of thread count — training and inference both
stay deterministic. No GPU is ever used.

---

## 2. Layers

All layers implement the `Layer` trait (`forward`, `name`, `as_any`) and compose
into a `Sequential` pipeline.

| Layer              | Forward                                              |
|--------------------|-----------------------------------------------------|
| `Linear`           | `y = xW + b` (may carry a packed `Arc<QWeight>`)     |
| `ActivationLayer`  | element-wise `Activation` (ReLU, Softmax, …)         |
| `LayerNorm`        | per-row normalization with learned gamma/beta        |
| `Embedding`        | token + positional lookup                            |
| `Flatten`          | `[T, D]` sequence → `[1, T·D]` row                   |
| `TransformerBlock` | causal multi-head self-attention + feed-forward      |

A `Linear` whose weights came from a quantized FINF file (or were quantized in
memory) holds an `Option<Arc<QWeight>>` and dispatches to `ops::qlinear`, which
consumes the packed bytes **without expanding to f32**. `KvCache` stores past
keys/values so generation extends a sequence one token at a time without
recomputing the whole context.

---

## 3. Training

- `Net` — a trainable MLP (`Net::mlp`, `Net::embedding_mlp`) with optional QAT via
  `set_qat(true)`; `train_epoch` runs one shuffled pass.
- `TransformerNet` — a trainable causal transformer; `train_transformer_epoch`
  applies next-token loss at every position. `train_transformer_epoch_threaded`
  shards each minibatch across `std::thread::scope` workers and reduces gradients
  in a fixed order (bit-identical to serial at one shard). FFN-hidden dropout is
  available during training.
- Optimizers — `Sgd` (optional momentum) and `Adam`; `Adam::with_weight_decay`
  adds **AdamW** decoupled decay. `clip_grad_norm`, and `LrSchedule`/`LrDecay`
  (warmup + cosine/linear) round out the optimizer module.
- Losses — `softmax_cross_entropy`, `mse`.
- `to_inference()` / `to_inference_task()` convert a trained net into a
  `Sequential` inference model.

Training is deterministic for a fixed `Rng` seed, independent of thread count.

---

## 4. The generative SLM API

`GenerativeSLM` wraps a `Sequential` model, a `Normalizer`, and `ModelMetadata`.

### Training paths

```rust
// One-hot MLP (character-level only):
GenerativeSLM::train(corpus, context_len, hidden, epochs, lr, momentum, batch, rng);

// Embedding MLP (char if vocab_size == 0, BPE if >= 256):
GenerativeSLM::train_embedded(corpus, context_len, embed, hidden,
    epochs, lr, momentum, batch, vocab_size, rng);

// Causal transformer (char if vocab_size == 0, BPE if >= 256):
GenerativeSLM::train_transformer(corpus, context_len, embed, heads, blocks,
    hidden, epochs, lr, batch, vocab_size, rng);
```

Each has a `*_with_callback` variant taking `FnMut(epoch, loss)` and a
`*_threaded_with_callback` variant taking a thread count. The transformer path
also accepts a `TransformerConfig` via `train_transformer_config[_threaded]`.

### `TransformerConfig`

```rust
pub struct TransformerConfig {
    pub context_len: usize,
    pub embed_dim: usize,
    pub num_heads: usize,
    pub num_blocks: usize,
    pub hidden_dim: usize,
    pub epochs: usize,
    pub lr: f32,
    pub batch_size: usize,
    pub vocab_size: usize, // 0 = character-level, >= 256 = byte-level BPE
}
```

`Default` uses `vocab_size: 512` (BPE).

### Generation and evaluation

```rust
slm.generate(seed, num_chars, temperature, rng) -> Result<String>
slm.generate_continuation(seed, num_chars, temperature, rng) -> Result<String> // no seed prefix
slm.generate_stream(seed, num_chars, temperature, rng, on_text) -> Result<String>
slm.evaluate(text) -> Result<Evaluation> // perplexity, cross-entropy, bits/token
```

`num_chars` counts characters for both tokenizers. BPE models encode the seed,
sample subword tokens (always feeding the last `context_len` tokens, left-padded
when the prompt is short), decode the stream, and trim to `seed + num_chars`
characters. `generate_stream` is UTF-8-safe: it holds back a partial trailing
multi-byte character until it completes, so a `U+FFFD` placeholder is never
emitted and then revised.

### Persistence

```rust
slm.to_bytes()                 // FINF v4/v5, f32 weights
slm.to_bytes_quantized()       // FINF v5, int8 weights (≈4× smaller)
slm.to_bytes_quantized_int4()  // FINF v5, int4 weights (≈8× smaller)
slm.save(path)                 // int8 v5 to disk
GenerativeSLM::load(path)
GenerativeSLM::from_bytes(bytes)
GenerativeSLM::load_or_train(path, corpus, cfg, rng, cb) // train-once cache
```

---

## 5. The BPE tokenizer

`ByteBpeTokenizer` is a byte-level Byte-Pair Encoding tokenizer. Its base
vocabulary is the 256 single bytes, so any UTF-8 text round-trips with no
unknown-token escape hatch.

```rust
ByteBpeTokenizer::byte_level()         // 256 byte tokens, no merges
ByteBpeTokenizer::train(corpus, vocab) // learn up to vocab-256 merges
tok.encode(text) -> Vec<usize>
tok.decode(&ids) -> String
tok.vocab_size() -> usize
tok.encode_state() -> String           // serializable merge list "a,b;c,d;…"
ByteBpeTokenizer::from_state(state)     // rebuild from a merge list
```

Training greedily merges the most frequent adjacent pair until the target vocab
size is reached or no pair repeats. Ties break deterministically, so the same
corpus and target size always yield the same merges. `train` requires
`vocab_size >= 256`. Special tokens `TOK_BOS/TOK_EOS/TOK_PAD/TOK_UNK` are
exposed for callers that need them.

---

## 6. Quantization

Symmetric int8: `value ≈ i8 × scale`, `scale = max|value| / 127`. Available
**per-tensor** and **per-channel** (one scale per output row — better when a
single channel carries an outlier).

- `fake_quantize_int8(&mut data)` / `fake_quantize_int8_per_channel` snap a tensor
  onto the int8 grid in place (used during QAT).
- `QUANT_MIN_LEN = 64`: tensors shorter than this stay f32 (biases, LayerNorm).
- Non-finite tensors are left untouched.

QAT keeps full-precision master weights and snaps the working copy each step
(straight-through estimator), so the int8 file matches the trained model.

### In-memory packed weights (`QWeight`)

`QWeight` holds weights packed at `QKind::Int8` or `QKind::Int4` for direct
consumption by `ops::qlinear` (no f32 expansion). **int4 uses a split-half
layout**: byte `b`'s low nibble is column `b`, its high nibble is column
`half + b`. That makes each nibble lane a contiguous, unit-stride column range, so
the decode kernel vectorizes the same `out[c] += a · sext(nibble)` loop it uses
for int8 — the alternative interleaved packing defeats the autovectorizer and is
several times slower (see [../benchmarks.md](../benchmarks.md) §4d).

---

## 7. The FINF model format

Little-endian binary:

```text
4 bytes  "FINF"
u32      version (4 = f32, 5 = int8/int4-capable)
u32      norm_len;  [bytes] normalizer string  (empty for SLM)
u32      meta_len;  [bytes] ModelMetadata JSON
u32      num_layers
per layer: u8 tag, then layer payload
```

Layer tags: `0` Linear, `1` ActivationLayer, `2` Embedding, `3` LayerNorm,
`4` TransformerBlock, `5` Flatten. In **v5** each weight *vector* carries a
one-byte encoding marker, which is what lets a single file mix precisions
(small bias vectors stay f32 while large matrices go int8/int4):

| Marker | Encoding                                             |
|:------:|------------------------------------------------------|
| 0 | raw f32                                                    |
| 1 | int8 symmetric per-tensor (f32 scale, then one i8/value)  |
| 2 | int8 symmetric per-channel (one f32 scale per input row)  |
| 3 | int4 symmetric per-tensor (f32 scale, then packed nibbles)|
| 4 | int4 symmetric per-channel (the default for `to_bytes_quantized_int4`) |

The loader reads v4 and v5 transparently, rejects unknown markers rather than
misreading, and bounds-checks every dimension field before allocating. Matrices
from an int8/int4 file are loaded **packed** (as `QWeight`) so they stay
quantized in memory.

### `ModelMetadata`

Serialized as JSON inside the file. Fields include `task`, `feature_names`,
`feature_ranges`, `class_names`, `input_dim`, `output_dim`, and
`tokenizer_state` — the BPE merge list (empty for character-level models). Older
files without `tokenizer_state` parse fine and default to character-level.

---

## 8. The GGUF importer and Llama/Qwen runner

This is the second Transformer stack: import and run externally pretrained
checkpoints.

### 8.1 Reading a GGUF (`gguf`)

```rust
Gguf::from_path(path)  // reads the whole file into memory
Gguf::open(path)       // streamed: parses the header, reads tensor bytes on demand
```

A pure-`std`, `unsafe`-free GGUF v2/v3 reader: magic/version check, the full
typed metadata key/value table, the tensor directory, alignment handling, and
dequantizers for **F32, F16, Q8_0, Q8_1, Q4_0, Q4_1** plus the **Q4_K, Q5_K,
Q6_K** super-block k-quants. It is defensively coded (checked offset arithmetic,
EOF guards, rejects nested arrays and absurd counts). `Q2_K`/`Q3_K` and the `IQ*`
families return a clear "needs its own decoder" error. `Gguf::open` avoids holding
the whole (multi-GB) file resident — it keeps a `Mutex<File>` and seeks to each
tensor — without any `unsafe` (so it is a streamed read, not an `mmap`).

### 8.2 Importing the tokenizer (`gguf_tokenizer`)

```rust
let tok = GgufTokenizer::from_gguf(&gguf)?;
tok.encode(text) -> Vec<usize>;
tok.decode(&ids) -> String;
```

Reconstructs the checkpoint's own tokenizer from `tokenizer.ggml.*`. BPE
encode/decode is exact; SentencePiece (SPM) decode is exact and encode is a greedy
longest-match approximation. Without this, you could only feed raw token IDs —
with it, imported models run on **text**.

### 8.3 Running (`llm`)

```rust
let model = gguf.load_llama(QKind::Int4)?;       // or Int8
let model = gguf.load_llama_prec(None)?;          // keep f32 (no second quantization)
let out = model.generate(&ids, max_new, temperature, rng)?;
```

`LlamaModel` implements RMSNorm, RoPE (both `Norm` interleaved and `Neox`
split-half conventions), grouped-query attention with a per-layer KV cache, and
the SwiGLU FFN. It offers a full-sequence forward and an O(context)/token cached
decode; the tests cross-check the cached path against the full forward row for
row. Import is **lossy by default** (dequantize → re-quantize to Ferrum's per-row
grid) and is **not** bit-exact to llama.cpp.

### 8.4 Training the imported architecture (`llm_train`)

```rust
let mut trainer = LlamaTrainer::new(f32_model)?;  // requires f32 weights
let pre_step_loss = trainer.train_step(&tokens, lr)?;
```

`LlamaTrainer` adds a hand-derived, **finite-difference-checked** backward pass
(RMSNorm, RoPE, GQA+softmax, SwiGLU, embedding, LM head), next-token
cross-entropy, and an SGD `train_step`. `new` rejects quantized weights — there is
no f32 master to update behind a packed `QWeight` — so import at f32
(`load_llama_prec(None)`) before training. This makes *small* imported models
fine-tunable; a 1B model is blocked by RAM (~16 bytes/param for the optimizer
state) and compute, not by missing gradients (see
[../ferrum_review.md](../ferrum_review.md) §4.3).
