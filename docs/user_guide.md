# Ferrum Developer User Guide

Task-oriented walkthroughs for common goals. Assumes you have built the
workspace (see [installation.md](../installation.md)).

---

## Choosing a training path

| You want…                                  | Use                               |
|--------------------------------------------|-----------------------------------|
| The simplest transparent baseline          | `GenerativeSLM::train` (one-hot)  |
| A small, fast model that beats one-hot size | `GenerativeSLM::train_embedded`   |
| The best quality on real text              | `GenerativeSLM::train_transformer`|
| Tabular classification/regression          | the `train_cli` binary            |

For the embedded and transformer paths, set `vocab_size = 0` for character-level
or `>= 256` for byte-level BPE.

---

## Train a BPE SLM and generate

```rust
use ferrum_core::{GenerativeSLM, Rng};

let corpus = std::fs::read_to_string("corpus.txt").unwrap();
let mut rng = Rng::new(1337);

let slm = GenerativeSLM::train_transformer(
    &corpus, 16, 32, 4, 2, 64, 200, 0.01, 16, 512, &mut rng,
).unwrap();

slm.save("model.bin").unwrap();
println!("{}", slm.generate("Once upon a time", 200, 0.7, &mut rng).unwrap());
```

---

## Watch training progress

```rust
let slm = GenerativeSLM::train_transformer_with_callback(
    &corpus, 16, 32, 4, 2, 64, 200, 0.01, 16, 512, &mut rng,
    |epoch, loss| if epoch % 10 == 0 { println!("epoch {epoch}: {loss:.4}") },
).unwrap();
```

---

## Cache a model on disk

`load_or_train` trains and saves the first time, then loads on subsequent runs:

```rust
use ferrum_core::{GenerativeSLM, Rng, TransformerConfig};

let cfg = TransformerConfig::default();      // BPE vocab 512
let mut rng = Rng::new(1337);
let (slm, was_loaded) =
    GenerativeSLM::load_or_train("model.bin", &corpus, &cfg, &mut rng, |_, _| {}).unwrap();
println!("loaded from disk: {was_loaded}");
```

---

## Shrink a model with int8

```rust
let full  = slm.to_bytes().unwrap().len();
let quant = slm.to_bytes_quantized().unwrap().len();
println!("{full} → {quant} bytes");   // roughly 4× smaller
```

`save()` already writes the int8 file. Quantization-aware training keeps the int8
model close to the f32 model.

---

## Work with the tokenizer directly

```rust
use ferrum_core::ByteBpeTokenizer;

let tok = ByteBpeTokenizer::train(&corpus, 512).unwrap();
let ids = tok.encode("some text");
let text = tok.decode(&ids);
let state = tok.encode_state();            // store this string
let same = ByteBpeTokenizer::from_state(&state).unwrap();
```

---

## Common pitfalls

- **Corpus too short for the context window.** Both tokenizers require more than
  `context_len` tokens. BPE compresses text, so a heavily repetitive corpus can
  fall below the threshold even when it has many characters — use a longer or
  more varied corpus, a smaller `context_len`, or a smaller `vocab_size`.
- **`vocab_size` between 1 and 255.** Rejected: the 256-byte base vocabulary is
  irreducible. Use `0` (character-level) or `>= 256` (BPE).
- **`embed_dim` not divisible by `num_heads`.** Multi-head attention requires it.
- **Seed shorter than the context window.** Fine for BPE (left-padded), but
  character-level generation needs at least `context_len` characters of seed.
