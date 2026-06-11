# ferrum_core

The ML engine at the heart of the Ferrum workspace. Pure Rust, zero dependencies
(`std` only), compiles to `wasm32-unknown-unknown` without modification.

---

## Design principle

Each module depends only on the ones above it in this list. There are no cycles
and no forward references. Read them top-to-bottom and every type is defined
before it is used.

```
verbose    ←  opt-in tracing: set_verbose() + the vprintln! macro
error      ←  every other module uses Result<T, InferError>
tensor     ←  the only data structure; a flat Vec<f32> + shape
ops        ←  all raw arithmetic: matmul, bias-add, transpose, argmax, softmax
activation ←  ReLU, Sigmoid, Tanh, Softmax, Identity
layer      ←  Layer trait: Linear, ActivationLayer, LayerNorm,
              Embedding (token + positional), TransformerBlock (causal MHA),
              KvCache for O(T)-per-token incremental generation
model      ←  Sequential: Vec<Box<dyn Layer>>, forward()
rng        ←  seeded xorshift64* (weight init + minibatch sampling)
loss       ←  fused softmax cross-entropy + gradient, MSE
optim      ←  SGD with momentum + Adam (bias-corrected)
csv        ←  CSV parser, Normalizer, task auto-detection, ModelMetadata
train      ←  DenseT, ReluT, Net (trainable MLP), backprop, train_epoch
train_transformer ← TransformerNet: full backprop through the causal transformer
loader     ←  FINF v4 binary format: weights + normalizer + metadata in one file
slm        ←  GenerativeSLM: corpus → train → generate (char-level)
```

---

## Module details

### `verbose.rs`
`set_verbose(true)` enables the `vprintln!` macro everywhere — per-call shape
logs, activation statistics, NaN/Inf detection, per-epoch loss/ETA. Overhead
when off is a single relaxed atomic load per call site.

### `error.rs`
Defines `InferError` (an enum covering ShapeMismatch, DimMismatch, NotAMatrix,
Io, Format, ParseError) and the `Result<T>` alias used everywhere. `From` impls
for `std::io::Error` and `std::num::ParseFloatError` make `?` work at I/O and
parse sites.

### `tensor.rs`
`Tensor { shape: Vec<usize>, data: Vec<f32> }` in row-major (C) order.
`new()` validates that `shape.product() == data.len()`. Key methods:
- `matrix(r, c, data)` / `vector(data)` / `row(data)` — typed constructors.
- `matrix_dims()` — returns `(rows, cols)` or `NotAMatrix` error.
- `at(r, c)` — index into a matrix without slicing.
- `map(f)` — elementwise transform, returns a new tensor.
- `reshape()` — reinterpret shape without copying data.

### `ops.rs`
All raw `f32` arithmetic. The `matmul` implementation uses `i-k-j` loop order
(rather than the textbook `i-j-k`) for cache friendliness: the innermost loop
walks contiguous memory in both `b` and the output buffer. `argmax_rows` uses
`f32::total_cmp`, so NaN logits give a deterministic answer instead of a panic.

Backprop kernels (`transpose`, `sum_axis0`, `mul`) live here so all arithmetic
is in one auditable place.

### `activation.rs`
`Activation` is a `u8`-tagged enum so it can be serialised as a single byte by
the loader. Softmax is implemented row-wise with a max-subtraction stability
trick. Every variant is tested, including a `tag` round-trip test.

### `layer.rs`
The `Layer` trait has three methods: `forward`, `name`, and `as_any`. The last
one — upcasting to `&dyn Any` — is the standard Rust idiom for recovering a
concrete type from a trait object, which the loader needs for serialisation.

Five layer types:
- `Linear` stores weight as `[in_features, out_features]` so the forward pass is
  `matmul(input, weight)` with no transpose.
- `ActivationLayer` wraps an `Activation`.
- `LayerNorm` — per-row normalisation with ε = 1e-5 and learned γ/β.
- `Embedding` — token lookup + learned positional encoding; `embed_one` embeds
  a single (token, position) pair for cached generation.
- `TransformerBlock` — pre-norm causal multi-head self-attention + FFN with
  residual connections. The constructor rejects `num_heads = 0`,
  `embedding_dim % num_heads != 0`, and `context_len = 0`. Per-head attention
  maps are captured via `RefCell` for visualisation.

`KvCache` plus `TransformerBlock::forward_with_cache` give O(T)-per-token
incremental generation; a test proves the cached path matches the full forward
pass row-for-row.

### `model.rs`
`Sequential` holds `Vec<Box<dyn Layer>>` and runs `forward()` by threading the
input through each layer in order. `summary()` prints the architecture for
debugging.

### `rng.rs`
Xorshift64* with a Box-Muller `next_normal()` for Kaiming weight initialisation.
A zero seed is remapped to a nonzero constant. All tests use fixed seeds so they
are deterministic and reproducible.

### `loss.rs`
`softmax_cross_entropy(logits, targets) -> (f32, Tensor)` fuses softmax and
cross-entropy into one numerically stable pass. The returned gradient is
`(p - onehot(t)) / batch_size`, which is the exact expression the chain rule
gives. This is verified by a finite-difference gradient check in the tests.
`mse` provides the regression loss, also with its gradient.

### `optim.rs`
Two optimizers, both stateless — callers own the moment buffers, which keeps
Rust's borrow checker happy:
- `Sgd { lr, momentum }`: `v ← m·v + g; p ← p − lr·v`. Used by the MLP trainer.
- `Adam`: bias-corrected first/second moments. Used by the transformer trainer.

### `csv.rs`
- `CsvDataset::from_str(text)` — parses a CSV, auto-detects a header row and
  classification vs regression, assigns integer labels to string classes in
  order-of-first-appearance.
- `Normalizer` — fits per-column mean/std on training data, transforms any
  matrix, and serialises to `"mean0,std0;mean1,std1;…"` for embedding in the
  model file. Constant columns get `std = 1.0` rather than dividing by zero.
- `ModelMetadata` — dataset name, `TaskType` (Classification / Regression /
  TransformerSLM), feature names/ranges, class names, dims — serialised as JSON
  inside the model file so a UI can configure itself from the file alone.
- `train_val_split` — Fisher-Yates shuffle + split, preserving class metadata.

### `train.rs`
`DenseT` and `ReluT` mirror their inference counterparts but cache activations
for the backward pass. `Net::mlp(in, hidden, out, rng)` uses Kaiming
initialisation. `train_epoch` runs one pass of random-minibatch SGD. The
gradient check in the test suite perturbs individual weights by ε and confirms
the analytic gradient matches `(L(w+ε) - L(w-ε)) / 2ε` to within 1e-2.

`Net::to_inference()` converts the trainable network back to a `Sequential`
(appending `Softmax`) ready for the inference engine and loader.

### `train_transformer.rs`
`TransformerNet` implements full backprop through token + positional
embeddings, LayerNorm, causal multi-head attention (softmax backward through
the mask), and the FFN, trained with next-token loss at every position via
`forward` → `softmax_cross_entropy` → `backward` → `step(&Adam)`.
`to_inference()` exports to a FINF-serialisable `Sequential`. Verified by
finite-difference gradient checks across all 18 parameter groups.

### `slm.rs`
`GenerativeSLM` — the high-level character-level language-model API:
- `train(corpus, context_len, hidden, epochs, lr, momentum, batch, rng)` — the
  one-hot MLP path (`input_dim = context_len × vocab_size`).
- `train_transformer(corpus, context_len, embed_dim, heads, blocks, hidden,
  epochs, lr, batch, rng)` — a real decoder-only causal transformer trained
  end-to-end with Adam (compact token-ID inputs).
- `generate(seed, n_chars, temperature, rng)` — handles both model families;
  char-safe seed handling for multi-byte UTF-8.
- `to_bytes` / `from_bytes` — self-contained FINF v4 round-trip; the vocabulary
  is hex-encoded into `meta.class_names`.
- Both training paths have `_with_callback` variants for progress reporting.

### `loader.rs`
**FINF v4** binary format (all little-endian):

```
4 bytes  b"FINF"
u32      version = 4
u32      normalizer_byte_length
[bytes]  normalizer string ("mean0,std0;mean1,std1;…", empty for SLMs)
u32      metadata_byte_length
[bytes]  ModelMetadata JSON
u32      num_layers
per layer:
  u8     tag: 0=Linear, 1=Activation, 2=Embedding, 3=LayerNorm, 4=TransformerBlock
  followed by that layer's dims and raw f32 weights
```

The reader is a forward-only bounds-checked cursor that returns a `Format`
error on any truncation rather than panicking. Byte lengths are verified
against the remaining buffer *before* allocating, and all dimension products
use checked multiplication, so corrupt or malicious files fail fast instead of
attempting huge allocations.

---

## Running the tests

```bash
# Unit tests only (fast)
cargo test -p ferrum_core

# With output for failing tests
cargo test -p ferrum_core -- --nocapture

# A specific test
cargo test -p ferrum_core loss::tests::gradient_finite_difference
```

The gradient check and training convergence tests take ~0.1s each. The full
149-test unit suite runs in about a second (217 tests across the workspace).
