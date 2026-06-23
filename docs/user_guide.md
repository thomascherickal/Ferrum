# Ferrum Developer User Guide

Task-oriented walkthroughs for common goals. Assumes you have built the workspace
(see [installation.md](../installation.md)).

---

## Choosing a path

| You want…                                   | Use                                |
|---------------------------------------------|------------------------------------|
| The simplest transparent baseline           | `GenerativeSLM::train` (one-hot)   |
| A small, fast model that beats one-hot size  | `GenerativeSLM::train_embedded`    |
| The best quality on real text               | `GenerativeSLM::train_transformer` |
| Tabular classification/regression           | the `train_cli` binary             |
| Run a downloaded Llama/Qwen model            | `Gguf` + `LlamaModel` (or `run-gguf`) |
| Fine-tune an imported model                  | `LlamaTrainer`                     |

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

## Regularize with AdamW decay and dropout

Plain Adam can over-fit a small corpus. `Adam::with_weight_decay` adds decoupled
(AdamW) decay, and `TransformerNet` supports FFN-hidden dropout during training.
From the CLI these are `--weight_decay` and `--dropout`; from the library they are
configured on the optimizer / network. Watch the train-vs-held-out perplexity gap
(below) to see whether they help — regularization trades a little training fit for
generalization.

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

## Shrink a model with int8 or int4

```rust
let full  = slm.to_bytes().unwrap().len();
let int8  = slm.to_bytes_quantized().unwrap().len();        // ≈4× smaller
let int4  = slm.to_bytes_quantized_int4().unwrap().len();   // ≈8× smaller
println!("{full} → {int8} → {int4} bytes");
```

`save()` already writes int8. Quantization-aware training keeps the quantized
model close to the f32 model; int4's grid is coarser, so expect slightly more
drift for the extra size win.

---

## Run a downloaded Llama/Qwen GGUF

```rust
use ferrum_core::{Gguf, GgufTokenizer, QKind, Rng};

let gguf  = Gguf::open("model.gguf").unwrap();      // streamed; not fully resident
let tok   = GgufTokenizer::from_gguf(&gguf).unwrap();
let model = gguf.load_llama(QKind::Int4).unwrap();  // int4 RAM, or QKind::Int8 for speed
let ids   = tok.encode("Once upon a time");
let out   = model.generate(&ids, 64, 0.7, &mut Rng::new(1)).unwrap();
println!("{}", tok.decode(&out));
```

Only `llama`/`qwen2` architectures load; `Q2_K`/`Q3_K`/`IQ*` are rejected. Decode
is bandwidth-bound (a few tok/s for ~1B on a CPU). The CLI equivalent is
`train_transformer run-gguf model.gguf "prompt" --quant int4 --max 64`.

---

## Fine-tune an imported model

```rust
use ferrum_core::{Gguf, GgufTokenizer, LlamaTrainer};

let model = Gguf::open("model.gguf").unwrap().load_llama_prec(None).unwrap(); // f32!
let tok   = GgufTokenizer::from_gguf(&Gguf::open("model.gguf").unwrap()).unwrap();
let mut trainer = LlamaTrainer::new(model).unwrap();   // errs if weights are quantized
for _ in 0..steps {
    let loss = trainer.train_step(&tok.encode("training text"), 1e-3).unwrap();
}
```

Import at **f32** (`load_llama_prec(None)`) — training needs f32 masters, so
`LlamaTrainer::new` rejects quantized weights. The backward pass is
finite-difference-checked, but the optimizer-state RAM (~16 bytes/param) and
compute make this practical only for *small* models, not 1B.

---

## Common pitfalls

- **Corpus too short for the context window.** Both tokenizers require more than
  `context_len` tokens. BPE compresses text, so a repetitive corpus can fall below
  the threshold even with many characters — use a longer/more varied corpus, a
  smaller `context_len`, or a smaller `vocab_size`.
- **`vocab_size` between 1 and 255.** Rejected: the 256-byte base is irreducible.
  Use `0` (character-level) or `>= 256` (BPE).
- **`embed_dim` not divisible by `num_heads`.** Multi-head attention requires it.
- **Seed shorter than the context window.** Fine for BPE (left-padded), but
  character-level generation needs at least `context_len` characters of seed.
- **`LlamaTrainer::new` returns an error.** The model is quantized; re-import with
  `load_llama_prec(None)` for f32 weights.
