# ferrum_core

The engine at the heart of the [Ferrum](https://github.com/thomascherickal/Ferrum)
workspace. Pure Rust, zero dependencies, `std`-only, `#![forbid(unsafe_code)]`.

`ferrum_core` lets you hand-build, train, and run causal Transformers, Small
Language Models, and classical MLPs entirely on the CPU; serialize them to a
single self-contained file that also runs in WebAssembly; and import and run
external **Llama/Qwen GGUF** checkpoints. Every kernel — matmul, attention,
quantized GEMV, the BPE merges, the dequantizers — is written in the crate
itself, so the whole forward *and* backward pass is readable with no hidden
library underneath.

## Features

- **Tensors and ops** — matmul, fused/cache-tiled `linear_forward`, packed
  `qlinear` (int8/int4), softmax, LayerNorm, row reductions.
- **Layers** — `Linear` (optionally carrying a packed `Arc<QWeight>`),
  `ActivationLayer`, `LayerNorm`, `Embedding`, `Flatten`, `TransformerBlock`,
  composed via `Sequential`; `KvCache` for O(context)/token generation.
- **Training** — `Net` (MLP) and `TransformerNet`, `train_epoch` /
  `train_transformer_epoch`, data-parallel threaded epochs, `Adam` (with optional
  **AdamW** decoupled weight decay) and `Sgd`, grad-norm clipping, warmup +
  cosine/linear LR schedules, cross-entropy and MSE.
- **Quantization-aware training** — symmetric int8 fake-quantization (per-tensor
  *and* per-channel) with a straight-through estimator; ship int8 models that
  behave like what you trained.
- **CPU parallelism** — matmul kernels split across a **persistent worker pool**
  (threads spawned once and reused), with dynamic thread detection
  (`FERRUM_NUM_THREADS` override). A `run_1d` **column split** parallelizes the
  `m = 1` decode GEMV that plain row-splitting cannot. `std`-only, no `unsafe`,
  no GPU; deterministic across thread counts, serial on wasm.
- **Byte-level BPE tokenizer** — `ByteBpeTokenizer` round-trips any UTF-8 text
  and serializes to a compact merge list; deterministic merge learning.
- **Generative SLM** — `GenerativeSLM` with three training paths and one
  generation API (plus streaming and held-out perplexity); QAT with optional
  AdamW weight decay and dropout.
- **GGUF import & Llama/Qwen runner** — read GGUF checkpoints
  (`F32/F16/Q8_0/Q8_1/Q4_0/Q4_1` and the **Q4_K/Q5_K/Q6_K** k-quants) and their
  own tokenizer; run `LlamaModel` (RMSNorm, RoPE, grouped-query attention,
  SwiGLU, KV cache) in int4/int8/f32; **train it** via the gradient-checked
  `LlamaTrainer`. A streamed reader (`Gguf::open`) avoids holding the whole file
  resident.
- **FINF model format** — v4 (f32) / v5 (**int8 + int4**, per-tensor or
  per-channel), self-contained: weights, normalizer, metadata, and tokenizer in
  one buffer.

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

## Importing a GGUF

```rust
use ferrum_core::{Gguf, GgufTokenizer, QKind, Rng};

let gguf = Gguf::open("model.gguf")?;                 // streamed; not fully resident
let tok  = GgufTokenizer::from_gguf(&gguf)?;          // the checkpoint's own vocab
let model = gguf.load_llama(QKind::Int4)?;            // or load_llama_prec(None) for f32
let ids  = tok.encode("Once upon a time");
let out  = model.generate(&ids, 64, 0.7, &mut Rng::new(1))?;
println!("{}", tok.decode(&out));
```

`load_llama` packs weights to int4/int8 (per-row scales); `load_llama_prec(None)`
keeps them f32 (no second quantization, the path `LlamaTrainer` requires).

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
| `tensor`, `ops`     | Tensor type and numeric kernels (matmul, `qlinear`, softmax) |
| `parallel`          | std-only persistent CPU worker pool; row split + `run_1d` column split |
| `layer`             | `Layer` trait and all layer implementations               |
| `model`             | `Sequential` pipeline                                     |
| `train`             | `Net`, `train_epoch`, accuracy                            |
| `train_transformer` | `TransformerNet`, threaded epochs, FFN dropout            |
| `optim`             | `Adam` (+ AdamW decay), `Sgd`, grad clip, LR schedule    |
| `loss`              | softmax cross-entropy, MSE                                |
| `quant`             | int8 + int4 (split-half), per-tensor/per-channel, `QWeight` |
| `tokenizer`         | `ByteBpeTokenizer`                                        |
| `slm`               | `GenerativeSLM`, `TransformerConfig`                     |
| `gguf`              | GGUF reader (legacy + Q4_K/Q5_K/Q6_K), streamed `open` → `load_llama` |
| `gguf_tokenizer`    | import a GGUF's own BPE/SPM tokenizer (`GgufTokenizer`)   |
| `llm`               | `LlamaModel`: RMSNorm, RoPE, GQA, SwiGLU, KV-cached decode |
| `llm_train`         | `LlamaTrainer`: gradient-checked backprop + SGD `train_step` |
| `csv`, `dataset`    | CSV dataset, normalizer, `ModelMetadata`, corpus cleaning |
| `loader`            | FINF v4/v5 read/write (f32 / int8 / int4, per-tensor/per-channel) |
| `rng`               | seeded deterministic PRNG (`xorshift64*`)                |

## License

MIT OR Apache-2.0.
