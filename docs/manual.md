# Ferrum Reference Manual

The complete technical reference for `ferrum_core`: the public API, the tokenizer,
quantization, and the FINF model format. For task-oriented guides see
[user_guide.md](user_guide.md) and the top-level [howtouse.md](../howtouse.md).

---

## 1. Tensors and ops

`Tensor` is a flat `Vec<f32>` plus a shape. Construct rows and matrices with
`Tensor::row`, `Tensor::matrix`; query a 2-D shape with `matrix_dims()`. The
`ops` module provides matmul, `softmax_rows`, `argmax_rows`, and the kernels the
layers are built from.

### CPU parallelism

The matmul kernels (in `ops` and the attention helpers) split their output rows
across CPU threads using the `parallel` module, which is built only on `std`
(`thread::scope`). The worker count is detected once from
`std::thread::available_parallelism()` and can be overridden with the
`FERRUM_NUM_THREADS` environment variable; query it with
`ferrum_core::num_threads()`. Workloads below an internal scalar-work threshold,
and the `wasm32` target, run serially. The row split does not change per-element
arithmetic, so output is bit-for-bit identical regardless of thread count — both
training and inference stay deterministic. No GPU is ever used.

---

## 2. Layers

All layers implement the `Layer` trait (`forward`, `name`, `as_any`) and compose
into a `Sequential` pipeline.

| Layer              | Forward                                              |
|--------------------|-----------------------------------------------------|
| `Linear`           | `y = xW + b`                                         |
| `ActivationLayer`  | element-wise `Activation` (ReLU, Softmax, …)         |
| `LayerNorm`        | per-row normalization with learned gamma/beta        |
| `Embedding`        | token + positional lookup                            |
| `Flatten`          | `[T, D]` sequence → `[1, T·D]` row                   |
| `TransformerBlock` | causal multi-head self-attention + feed-forward      |

`KvCache` stores past keys/values so generation extends a sequence one token at
a time without recomputing the whole context.

---

## 3. Training

- `Net` — a trainable MLP (`Net::mlp`, `Net::embedding_mlp`) with optional QAT
  via `set_qat(true)`; `train_epoch` runs one shuffled pass.
- `TransformerNet` — a trainable causal transformer; `train_transformer_epoch`
  applies next-token loss at every position.
- Optimizers — `Sgd` (optionally with momentum) and `Adam`.
- Losses — `softmax_cross_entropy`, `mse`.
- `to_inference()` / `to_inference_task()` convert a trained net into a
  `Sequential` inference model.

Training is deterministic for a fixed `Rng` seed.

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

Each has a `*_with_callback` variant taking an `FnMut(epoch, loss)`. The
transformer path also accepts a `TransformerConfig` via
`train_transformer_config`.

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

### Generation

```rust
slm.generate(seed, num_chars, temperature, rng) -> Result<String>
```

`num_chars` counts characters for both tokenizers. BPE models encode the seed,
sample subword tokens (always feeding the last `context_len` tokens, left-padded
when the prompt is short), decode the whole stream, and trim to `seed + num_chars`
characters.

### Persistence

```rust
slm.to_bytes()            // FINF v4/v5, f32 weights
slm.to_bytes_quantized()  // FINF v5, int8 weights
slm.save(path)            // int8 v5 to disk
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
size is reached or no pair repeats. Ties are broken deterministically, so the
same corpus and target size always yield the same merges. `train` requires
`vocab_size >= 256`.

---

## 6. Quantization

Symmetric per-tensor int8: `value ≈ i8 × scale`, `scale = max|value| / 127`.

- `fake_quantize_int8(&mut data)` snaps a tensor onto the int8 grid in place
  (used during QAT).
- `QUANT_MIN_LEN = 64`: tensors shorter than this stay f32 (biases, LayerNorm).
- Non-finite tensors are left untouched.

QAT keeps full-precision master weights and snaps the working copy each step
(straight-through estimator), so the int8 file matches the trained model.

---

## 7. The FINF model format

Little-endian binary:

```text
4 bytes  "FINF"
u32      version (4 = f32, 5 = int8-capable)
u32      norm_len;  [bytes] normalizer string  (empty for SLM)
u32      meta_len;  [bytes] ModelMetadata JSON
u32      num_layers
per layer: u8 tag, then layer payload
```

Layer tags: `0` Linear, `1` ActivationLayer, `2` Embedding, `3` LayerNorm,
`4` TransformerBlock, `5` Flatten. In v5 each weight vector carries a one-byte
encoding marker (`0` raw f32, `1` int8 symmetric). The loader reads v4 and v5
transparently and rejects corrupt dimension fields before allocating.

### `ModelMetadata`

Serialized as JSON inside the file. Fields include `task`, `feature_names`,
`feature_ranges`, `class_names`, `input_dim`, `output_dim`, and
`tokenizer_state` — the BPE merge list (empty for character-level models). Older
files without `tokenizer_state` parse fine and default to character-level.
