# Deployment Guide

A Ferrum model is a single self-contained `.bin` file (the FINF format): it
carries its weights, normalizer, metadata, and — for BPE models — the tokenizer
merge list. There is nothing else to ship. This guide covers the common targets.

---

## What you deploy

| Artifact            | Produced by                          | Contains                                  |
|---------------------|--------------------------------------|-------------------------------------------|
| `model.bin`         | `slm_cli` / `train_cli` / `save()`   | Weights + normalizer + metadata + tokenizer |
| A host binary       | `cargo build --release`              | Your app linked against `ferrum_core`     |
| A WASM bundle       | `wasm-pack build` on `tabular_wasm`  | Browser-runnable model + bindings         |

Because `ferrum_core` is `std`-only with zero dependencies, the host binary has
nothing to vet or update at runtime.

---

## 1. Native CPU deployment (server, desktop, edge box)

Build a release binary and copy it plus the model file to the target:

```bash
cargo build --release -p slm_cli
scp target/release/train_transformer model.bin user@host:/opt/ferrum/
```

On the target, generation needs only the two files:

```bash
/opt/ferrum/train_transformer generate /opt/ferrum/model.bin "prompt" --chars 200
```

To embed inference in your own service, depend on `ferrum_core` and call
`GenerativeSLM::load` / `generate` directly — no model server required.

---

## 2. Embedded and resource-constrained targets

Int8-quantized models are typically tens of kilobytes. For the smallest
footprint:

- Train with `save()` (int8 v5) rather than full-precision `to_bytes()`.
- Prefer the `train_embedded` path with a modest BPE vocabulary.
- Build with the workspace `release` profile (LTO, one codegen unit).

The engine is allocation-light with no driver or GPU dependency. By default it
parallelizes matmul across all CPU cores; on single-core or timing-sensitive
targets set `FERRUM_NUM_THREADS=1` to force fully serial, predictable execution.
On `wasm32` it runs serially automatically.

---

## 3. WebAssembly (in-browser inference)

Build the `tabular_wasm` crate to `wasm32`:

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
cd tabular_wasm
wasm-pack build --release --target web
```

This produces a `pkg/` directory with the `.wasm` module and JS glue. Serve the
`.wasm`, the JS bindings, and your `model.bin` from any **static** host — GitHub
Pages, Netlify, S3, or a plain web server. No backend is needed: the model loads
and runs entirely in the browser, and its embedded metadata lets the page build
its own UI.

A minimal load looks like:

```js
import init, { /* exported bindings */ } from "./pkg/tabular_wasm.js";
await init();
const bytes = new Uint8Array(await (await fetch("model.bin")).arrayBuffer());
// …construct the model from bytes and run inference…
```

---

## 4. Air-gapped deployment

Train on a connected machine, then transfer the single `model.bin` and the
static host binary to the isolated environment by physical media. Nothing is
fetched at runtime, so there are no network prerequisites and no dependency
updates to manage in the field.

---

## 5. Versioning and compatibility

- FINF **v4** holds full-precision f32 weights; **v5** adds per-tensor int8
  quantization. The loader reads both transparently.
- Models written before the tokenizer field still load: they default to
  character-level tokenization (empty tokenizer state).
- Pin a model to your app by shipping them together; the model file fully
  determines the architecture, vocabulary, and tokenizer.
