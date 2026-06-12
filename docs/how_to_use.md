# Tutorial: Building a Custom Edge SLM

This tutorial trains a custom causal Small Language Model on your own text,
exports it to a standalone `.bin` file, and shows how to run it natively or in
the browser. It complements the concise reference in
[../howtouse.md](../howtouse.md).

---

## Step 1 — Prepare a corpus

Any UTF-8 text file works — documentation, logs, dialogue, code. The richer and
longer the corpus, the better the model. Save it as `corpus.txt`.

```bash
cat > corpus.txt <<'EOF'
the quick brown fox jumps over the lazy dog while the calm river flows past
green hills and quiet villages. travelers walk along the winding road, telling
stories of distant lands and the slow turning of the seasons.
EOF
```

---

## Step 2 — Train with byte-level BPE

```bash
cargo run -p slm_cli -- train corpus.txt model.bin \
    --vocab 320 --context 12 --embed 32 --heads 4 --blocks 2 --epochs 200
```

- `--vocab 320` trains a byte-level BPE tokenizer (256 byte tokens + up to 64
  merges) and embeds it in the model. Use `--vocab 0` for character-level.
- Training is int8 quantization-aware, so the saved file is small and faithful.
- The model file is cached: re-running `train` loads it instead of retraining
  unless you pass `--force`.

---

## Step 3 — Inspect the model

```bash
cargo run -p slm_cli -- info model.bin
```

```text
Format    : FINF v5 (int8-quantized)
Task      : TransformerSLM
Input dim : 12
Output dim: …
Tokenizer : byte-level BPE (… tokens, … merges)
Layers    : …
```

---

## Step 4 — Generate text

```bash
cargo run -p slm_cli -- generate model.bin "the quick brown" --chars 200 --temp 0.7
```

Lower `--temp` makes generation greedier and more repetitive; higher values add
variety. The output always begins with your seed and adds exactly `--chars`
characters (unless generation is cut short).

---

## Step 5 — Run it from Rust

```rust
use ferrum_core::{GenerativeSLM, Rng};

let slm = GenerativeSLM::load("model.bin").unwrap();
let mut rng = Rng::new(7);
println!("{}", slm.generate("the quick brown", 200, 0.7, &mut rng).unwrap());
```

The loaded model carries its own tokenizer, so no separate vocabulary file is
needed.

---

## Step 6 — Run it in the browser (WASM)

Build the `tabular_wasm` bindings and host the `.wasm`, JS glue, and `model.bin`
on any static server:

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
cd tabular_wasm
wasm-pack build --release --target web
```

See [../deployment.md](../deployment.md) for hosting details. Because the model
file embeds its own metadata and tokenizer, the page needs nothing else to run
inference entirely client-side.

---

## Next steps

- Compare BPE against character-level (`--vocab 0`) on the same corpus.
- Try the smaller `train_embedded` path from the library for tighter models.
- Measure size and quality with the [evaluation guide](../evaluation.md).
