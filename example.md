# Worked Examples

Three end-to-end examples: a BPE transformer SLM, a character-level model for
comparison, and a tabular classifier. Every example is self-contained and uses
only `cargo`.

---

## Example 1 — A byte-level BPE Transformer SLM

### Step 1: prepare a corpus

Any UTF-8 text file works. For a quick demo:

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

Output:

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
characters even though the model generates BPE tokens internally.

---

## Example 2 — The same model, character-level

To compare tokenization strategies, train an identical network with
character-level tokenization by setting `--vocab 0`:

```bash
cargo run -p slm_cli -- train corpus.txt char_model.bin \
    --vocab 0 --context 8 --embed 16 --heads 2 --blocks 1 --epochs 30
cargo run -p slm_cli -- info char_model.bin
```

```text
Tokenizer : character-level (… chars)
```

The character model has a small vocabulary (one entry per distinct character)
while the BPE model has at least 256 token IDs. On longer, more varied corpora
the BPE model captures recurring multi-character patterns and typically needs a
shorter context window to model the same span of text.

---

## Example 3 — From the library

```rust
use ferrum_core::{GenerativeSLM, Rng, TransformerConfig};

fn main() -> Result<(), ferrum_core::InferError> {
    let corpus = std::fs::read_to_string("corpus.txt")
        .expect("read corpus");
    let mut rng = Rng::new(1337);

    let cfg = TransformerConfig {
        context_len: 16,
        embed_dim: 32,
        num_heads: 4,
        num_blocks: 2,
        hidden_dim: 64,
        epochs: 200,
        lr: 0.01,
        batch_size: 16,
        vocab_size: 512, // byte-level BPE
    };

    let slm = GenerativeSLM::train_transformer_config(&corpus, &cfg, &mut rng, |ep, loss| {
        if ep % 20 == 0 {
            println!("epoch {ep}: loss {loss:.4}");
        }
    })?;

    slm.save("model.bin")?; // int8-quantized FINF v5

    let text = slm.generate("Once upon a time", 200, 0.7, &mut rng)?;
    println!("{text}");
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
present the model.

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
