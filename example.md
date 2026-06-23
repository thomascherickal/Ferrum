# Worked Examples

End-to-end examples: a BPE transformer SLM, the same model character-level, a
tabular classifier, the tokenizer on its own, and importing an external GGUF.
Every example is self-contained and uses only `cargo`. Numbers shown are
illustrative — yours vary with corpus, hyperparameters, and seed.

---

## Example 1 — A byte-level BPE Transformer SLM

### Step 1: prepare a corpus

Any UTF-8 text file works:

```bash
cat > corpus.txt <<'EOF'
the quick brown fox jumps over the lazy dog while the calm river flows past
green hills and quiet villages. travelers walk along the winding road, telling
stories of distant lands, bright stars, and the slow turning of the seasons.
morning light spills over the valley as birds begin their song.
EOF
```

### Step 2: train

```bash
cargo run -p slm_cli -- train corpus.txt model.bin \
    --vocab 320 --context 8 --embed 16 --heads 2 --blocks 1 --epochs 30
```

```text
╔══════════════════════════════════════════════════════════╗
  ferrum transformer SLM trainer (int8 QAT)
  Corpus  : corpus.txt  (293 chars)
  Context : 8   Embed: 16   Hidden: 64
  Heads   : 2   Blocks: 1
  Tokenizer: byte-level BPE (vocab 320)
  Epochs  : 30   LR: 0.01   Batch: 16   Seed: 1337
╚══════════════════════════════════════════════════════════╝

  epoch     1/30   loss = …
  …
Trained in …s.
  vocab = 298 BPE tokens   context = 8   output_dim = 298
Saved … bytes → model.bin (int8-quantized FINF v5)
Reload check: OK (6 layers).
```

### Step 3: inspect

```bash
cargo run -p slm_cli -- info model.bin
```

```text
Format    : FINF v5 (int8-quantized)
Task      : TransformerSLM
Input dim : 8
Output dim: 298
Tokenizer : byte-level BPE (298 tokens, 42 merges)
Layers    : 6
```

### Step 4: generate

```bash
cargo run -p slm_cli -- generate model.bin "the quick brown" --chars 40 --temp 0.6
```

```text
the quick brown fox jumps over the lazy dog while the c
```

The seed round-trips exactly through the tokenizer, and `--chars` counts
characters even though the model generates BPE tokens internally. Add `--stream`
to watch the completion appear fragment by fragment.

---

## Example 2 — The same model, character-level

To compare tokenization strategies, train an identical network with
character-level tokenization (`--vocab 0`):

```bash
cargo run -p slm_cli -- train corpus.txt char_model.bin \
    --vocab 0 --context 8 --embed 16 --heads 2 --blocks 1 --epochs 30
cargo run -p slm_cli -- info char_model.bin
```

```text
Tokenizer : character-level (… chars)
```

The character model's vocabulary is one entry per distinct character; the BPE
model has at least 256 token IDs. On longer, varied corpora the BPE model packs
recurring multi-character patterns into single tokens, so a fixed context window
covers more text — often the difference between a model that captures structure
and one that runs out of view.

---

## Example 3 — From the library

```rust
use ferrum_core::{GenerativeSLM, Rng, TransformerConfig};

fn main() -> Result<(), ferrum_core::InferError> {
    let corpus = std::fs::read_to_string("corpus.txt").expect("read corpus");
    let mut rng = Rng::new(1337);

    let cfg = TransformerConfig {
        context_len: 16, embed_dim: 32, num_heads: 4, num_blocks: 2,
        hidden_dim: 64, epochs: 200, lr: 0.01, batch_size: 16,
        vocab_size: 512, // byte-level BPE
    };

    let slm = GenerativeSLM::train_transformer_config(&corpus, &cfg, &mut rng, |ep, loss| {
        if ep % 20 == 0 { println!("epoch {ep}: loss {loss:.4}"); }
    })?;

    slm.save("model.bin")?; // int8-quantized FINF v5
    println!("{}", slm.generate("Once upon a time", 200, 0.7, &mut rng)?);
    Ok(())
}
```

Reloading and continuing later:

```rust
use ferrum_core::{GenerativeSLM, Rng};
let slm = GenerativeSLM::load("model.bin").unwrap();
let mut rng = Rng::new(42);
println!("{}", slm.generate("In the beginning", 120, 0.8, &mut rng).unwrap());
```

---

## Example 4 — A tabular classifier

```bash
cargo run -p train_cli -- iris.csv iris_model.bin "Iris" 32 500
```

`train_cli` auto-detects classification vs. regression from the CSV, fits a
feature normalizer, trains an MLP, and writes a self-contained FINF model that
embeds the feature names, ranges, and class labels — everything a UI needs to
present the model with no sidecar files.

---

## Example 5 — Using the BPE tokenizer on its own

```rust
use ferrum_core::ByteBpeTokenizer;

let tok = ByteBpeTokenizer::train("low lower lowest low low newer newest", 300).unwrap();

let ids = tok.encode("lowest news");
assert_eq!(tok.decode(&ids), "lowest news");

// Non-ASCII round-trips losslessly thanks to the byte base vocabulary.
assert_eq!(tok.decode(&tok.encode("café 🌸 мир")), "café 🌸 мир");

// The merge list is the complete, portable state.
let restored = ByteBpeTokenizer::from_state(&tok.encode_state()).unwrap();
assert_eq!(restored.encode("lowest"), tok.encode("lowest"));
```

---

## Example 6 — Importing and running an external GGUF (Llama/Qwen)

From the command line, with the checkpoint's own tokenizer:

```bash
# int4 (smallest RAM, default), int8 (fastest decode), or f32 (no re-quantization).
cargo run -p slm_cli -- run-gguf qwen2-0_5b.gguf "Once upon a time" \
    --quant int4 --max 64 --temp 0.7
```

From the library:

```rust
use ferrum_core::{Gguf, GgufTokenizer, QKind, Rng};

let gguf  = Gguf::open("qwen2-0_5b.gguf")?;     // streamed; not fully resident
let tok   = GgufTokenizer::from_gguf(&gguf)?;
let model = gguf.load_llama(QKind::Int4)?;      // RMSNorm/RoPE/GQA/SwiGLU, KV-cached
let ids   = tok.encode("Once upon a time");
let out   = model.generate(&ids, 64, 0.7, &mut Rng::new(1))?;
println!("{}", tok.decode(&out));
```

Expect a few tokens per second for a ~1B model on a CPU, and tens of seconds to
prefill a long prompt — decode streams every weight once per token, so it is
bandwidth-bound (see [benchmarks.md](benchmarks.md) §4). Only `llama`/`qwen2`
load; `Q2_K`/`Q3_K`/`IQ*` files are rejected. To fine-tune instead of run, import
at f32 and wrap in `LlamaTrainer` (see [howtouse.md](howtouse.md) §4.4).
