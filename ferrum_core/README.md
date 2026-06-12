# ferrum_core

The engine at the heart of the [Ferrum](https://github.com/thomascherickal/Ferrum)
workspace. Pure Rust, zero dependencies, `std`-only, `#![forbid(unsafe_code)]`.

`ferrum_core` lets you hand-build, train, and run causal Transformers, Small
Language Models, and classical MLPs entirely on the CPU — and serialize them to
a single self-contained file that also runs in WebAssembly.

## Features

- **Tensors and ops** — matmul, softmax, LayerNorm, row reductions.
- **Layers** — `Linear`, `ActivationLayer`, `LayerNorm`, `Embedding`,
  `Flatten`, `TransformerBlock`, composed via `Sequential`; `KvCache` for fast
  generation.
- **Training** — `Net` (MLP) and `TransformerNet`, `train_epoch` /
  `train_transformer_epoch`, `Adam` and `Sgd`, cross-entropy and MSE losses.
- **Quantization-aware training** — symmetric int8 fake-quantization with a
  straight-through estimator; ship int8 models that behave like what you trained.
- **CPU parallelism** — matmul kernels split across all cores via
  `std::thread::scope`, with dynamic thread detection (`FERRUM_NUM_THREADS`
  override). No GPU, no external crates, deterministic results, serial on wasm.
- **Byte-level BPE tokenizer** — `ByteBpeTokenizer` round-trips any UTF-8 text
  and serializes to a compact merge list.
- **Generative SLM** — `GenerativeSLM` with three training paths and one
  generation API.
- **FINF model format** — v4 (f32) / v5 (int8), self-contained: weights,
  normalizer, metadata, and tokenizer in one buffer.

## Quick start

```rust
use ferrum_core::{GenerativeSLM, Rng};

let mut rng = Rng::new(1337);
let corpus = "the quick brown fox jumps over the lazy dog. ".repeat(20);

// Causal transformer SLM with a 512-token byte-level BPE vocabulary.
let slm = GenerativeSLM::train_transformer(
    &corpus, 16, 32, 4, 2, 64, 200, 0.01, 16, 512, &mut rng,
).unwrap();

slm.save("model.bin").unwrap();
let text = slm.generate("the quick", 100, 0.7, &mut rng).unwrap();
println!("{text}");
```

The `vocab_size` argument (`512` above) selects the tokenizer: `0` is
character-level, any value `>= 256` trains a byte-level BPE tokenizer of that
size and stores it inside the model.

## Tokenizer

```rust
use ferrum_core::ByteBpeTokenizer;

let tok = ByteBpeTokenizer::train("low lower lowest low low", 300).unwrap();
let ids = tok.encode("lowest");
assert_eq!(tok.decode(&ids), "lowest");
let restored = ByteBpeTokenizer::from_state(&tok.encode_state()).unwrap();
```

## Module map

| Module              | Responsibility                                            |
|---------------------|-----------------------------------------------------------|
| `tensor`, `ops`     | Tensor type and numeric kernels                           |
| `parallel`          | std-only CPU thread pool for matmul (`num_threads`)       |
| `layer`             | `Layer` trait and all layer implementations               |
| `model`             | `Sequential` pipeline                                     |
| `train`             | `Net`, `train_epoch`, accuracy                            |
| `train_transformer` | `TransformerNet`, `train_transformer_epoch`              |
| `optim`             | `Adam`, `Sgd`                                             |
| `loss`              | softmax cross-entropy, MSE                                |
| `quant`             | int8 fake-quantization for QAT and serialization          |
| `tokenizer`         | `ByteBpeTokenizer`                                        |
| `slm`               | `GenerativeSLM`, `TransformerConfig`                     |
| `csv`               | CSV dataset, normalizer, `ModelMetadata`                 |
| `loader`            | FINF v4/v5 read/write                                     |
| `rng`               | seeded deterministic PRNG                                 |

## License

MIT OR Apache-2.0.
