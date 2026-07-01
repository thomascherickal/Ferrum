# Tutorial: Building a Custom Edge SLM

This tutorial trains a custom causal Small Language Model on your own text,
exports it to a standalone `.bin` file, and runs it natively and in the browser.
It complements the concise reference in [../howtouse.md](../howtouse.md).

> Scope note: this tutorial is about a model **you train**. Running a *downloaded*
> Llama/Qwen GGUF is a separate, native-only path (`run-gguf`); see
> [../howtouse.md](../howtouse.md) §2. The WASM build below is for your own FINF
> models.

---

## Step 1 — Prepare a corpus

Any UTF-8 text file works — documentation, logs, dialogue, code. The richer and
longer the corpus, the better the model (the single biggest lever — see the
manual's [data page](../manual/06-data-gigo-and-why-good-data-wins.md)). Save it
as `corpus.txt`.

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
- Add `--weight_decay 0.01 --dropout 0.1` if a small corpus is being memorized.

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
cargo run -p slm_cli -- generate model.bin "the quick brown" --chars 200 --temp 0.7 --stream
```

Lower `--temp` makes generation greedier and more repetitive; higher values add
variety. The output always begins with your seed and adds exactly `--chars`
characters (unless cut short). `--stream` prints it live as it is produced.

---

## Step 5 — Run it from Rust

```rust
use ferrum_core::{GenerativeSLM, Rng};

let slm = GenerativeSLM::load("model.bin").unwrap();
let mut rng = Rng::new(7);
println!("{}", slm.generate("the quick brown", 200, 0.7, &mut rng).unwrap());
```

The loaded model carries its own tokenizer, so no separate vocabulary file is
needed — this is what makes a single `.bin` enough to ship anywhere.

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

Because the model file embeds its own metadata and tokenizer, the page needs
nothing else to run inference entirely client-side — no backend, no API key, and
the visitor's input never leaves their machine. The WASM build runs serially
(`wasm32` has no threads) but is otherwise the same engine. See
[../deployment.md](../deployment.md) for hosting details.

---

## Export to GGUF (library)

```rust
use ferrum_core::{Gguf, GgufQuant};

let g = Gguf::open("base.gguf")?;          // source: metadata + tokenizer
let model = g.load_llama_prec(None)?;      // f32 model (fine-tune here if desired)
model.write_gguf(&g, GgufQuant::Q4K, "out.gguf")?;
```

`write_gguf` carries the source's hyperparameters and tokenizer forward
verbatim, so the exported file runs in llama.cpp / ollama unchanged.

---

## Next steps

- Compare BPE against character-level (`--vocab 0`) on the same corpus.
- Try the smaller `train_embedded` path from the library for tighter models.
- Measure size and quality with the [evaluation guide](../evaluation.md), paying
  attention to the train-vs-held-out perplexity gap.
