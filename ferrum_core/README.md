# ferrum_core

The ML engine that powers the demo. Pure Rust, zero dependencies (`std` only),
compiles to `wasm32-unknown-unknown` without modification.

---

## Design principle

Each module depends only on the ones above it in this list. There are no cycles
and no forward references. Read them top-to-bottom and every type is defined
before it is used.

```
error      ←  every other module uses Result<T, InferError>
tensor     ←  the only data structure; a flat Vec<f32> + shape
ops        ←  all raw arithmetic: matmul, bias-add, transpose, argmax, softmax
activation ←  ReLU, Sigmoid, Tanh, Softmax, Identity
layer      ←  Layer trait, Linear (y = xW + b), ActivationLayer
model      ←  Sequential: Vec<Box<dyn Layer>>, forward()
rng        ←  seeded xorshift64* (weight init + minibatch sampling)
loss       ←  fused softmax cross-entropy + gradient (dL/dz = p − onehot(t))
optim      ←  SGD with momentum
csv        ←  numeric CSV parser + Normalizer (z-score, encode/decode)
train      ←  DenseT, ReluT, Net (trainable MLP), backprop, train_epoch
loader     ←  FINF v2 binary format: weights + normalizer in one file
```

---

## Module details

### `error.rs`
Defines `InferError` (an enum covering ShapeMismatch, DimMismatch, Io, Format,
ParseError) and the `Result<T>` alias used everywhere. `From` impls for
`std::io::Error` and `std::num::ParseFloatError` make `?` work at I/O and
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
walks contiguous memory in both `b` and the output buffer.

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

`Linear` stores weight as `[in_features, out_features]` so the forward pass is
`matmul(input, weight)` with no transpose.

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

### `optim.rs`
`Sgd { lr, momentum }` updates one parameter tensor at a time:
`v ← m·v + g; p ← p − lr·v`.
Callers own the velocity buffers (one per parameter), which keeps Rust's borrow
checker happy and the optimizer itself stateless.

### `csv.rs`
Two types:
- `CsvDataset::from_str(text)` — parses a CSV, auto-detects a header row,
  assigns integer labels to string classes in order-of-first-appearance.
- `Normalizer` — fits per-column mean/std on training data, transforms any
  matrix, and serialises to `"mean0,std0;mean1,std1;…"` for embedding in the
  model file. Constant columns get `std = 1.0` rather than dividing by zero.
- `train_val_split` — Fisher-Yates shuffle + split, preserving class metadata.

### `train.rs`
`DenseT` and `ReluT` mirror their inference counterparts but cache activations
for the backward pass. `Net::mlp(in, hidden, out, rng)` uses Kaiming
initialisation. `train_epoch` runs one pass of random-minibatch SGD. The
gradient check in the test suite perturbs individual weights by ε and confirms
the analytic gradient matches `(L(w+ε) - L(w-ε)) / 2ε` to within 1e-2.

`Net::to_inference()` converts the trainable network back to a `Sequential`
(appending `Softmax`) ready for the inference engine and loader.

### `loader.rs`
**FINF v2** binary format (all little-endian):

```
4 bytes  b"FINF"
u32      version = 2
u32      normalizer_byte_length
[bytes]  normalizer string ("mean0,std0;mean1,std1;…")
u32      num_layers
per layer:
  u8     tag: 0=Linear, 1=Activation
  Linear:     u32 in_f, u32 out_f, f32[in_f*out_f] weights, f32[out_f] bias
  Activation: u8 activation_tag
```

The reader is a forward-only bounds-checked cursor that returns a `Format`
error on any truncation rather than panicking.

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
84-test suite runs in under a second.
