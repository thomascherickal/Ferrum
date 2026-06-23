# Deployment Guide

A Ferrum model is a single self-contained `.bin` file (the FINF format): it
carries its weights, normalizer, metadata, and — for BPE models — the tokenizer
merge list. There is nothing else to ship: no sidecar vocabulary, no config, no
runtime to install. This guide covers the common targets.

---

## What you deploy

| Artifact            | Produced by                          | Contains                                  |
|---------------------|--------------------------------------|-------------------------------------------|
| `model.bin`         | `slm_cli` / `train_cli` / `save()`   | Weights + normalizer + metadata + tokenizer |
| A host binary       | `cargo build --release`              | Your app linked against `ferrum_core`     |
| A WASM bundle       | `wasm-pack build` on `tabular_wasm`  | Browser-runnable model + bindings         |

Because `ferrum_core` is `std`-only with zero dependencies, the host binary has
nothing to vet or update at runtime — the property that makes it viable in
audited, embedded, and air-gapped environments.

> **GGUF is an import path, not a deployment format.** `run-gguf` loads someone
> *else's* Llama/Qwen checkpoint at runtime; it is not how you ship a model you
> trained. Deploy your own models as FINF.

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
`GenerativeSLM::load` / `generate` directly — no model server required. Set
`FERRUM_NUM_THREADS` to bound CPU use per process when colocating many services.

---

## 2. Embedded and resource-constrained targets

Int8 models are typically tens of kilobytes; int4 halves that again. For the
smallest footprint:

- Save int8 (`save()`, the default) or int4 (`to_bytes_quantized_int4()`, ≈8×).
- Prefer the `train_embedded` path with a modest BPE vocabulary.
- Build with the workspace `release` profile (LTO, one codegen unit).

The engine is allocation-light with no driver or GPU dependency. By default it
parallelizes matmul across all cores; on single-core or timing-sensitive targets
set `FERRUM_NUM_THREADS=1` for fully serial, predictable execution. On `wasm32` it
runs serially automatically.

---

## 3. WebAssembly (in-browser inference)

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

```js
import init, { /* exported bindings */ } from "./pkg/tabular_wasm.js";
await init();
const bytes = new Uint8Array(await (await fetch("model.bin")).arrayBuffer());
// …construct the model from bytes and run inference…
```

---

## 4. Air-gapped deployment

Train on a connected machine, then transfer the single `model.bin` and the static
host binary to the isolated environment by physical media. Nothing is fetched at
runtime, so there are no network prerequisites and no dependency updates to manage
in the field — the zero-dependency design is what makes this trivial rather than a
security review.

---

## 5. Versioning and compatibility

- FINF **v4** holds full-precision f32 weights; **v5** adds int8 *and* int4
  (per-tensor or per-channel, selected per weight vector). The loader reads both
  transparently and rejects unknown encoding markers rather than misreading.
- Models written before the tokenizer field still load: they default to
  character-level tokenization (empty tokenizer state).
- Pin a model to your app by shipping them together; the file fully determines the
  architecture, vocabulary, and tokenizer, so there is no version skew between
  "the model" and "its config."
