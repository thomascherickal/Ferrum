# Use Cases for Ferrum

Ferrum is a zero-dependency, CPU-only, pure-Rust engine for small Transformers,
SLMs, and MLPs — and an importer for small open-weight Llama/Qwen checkpoints.
Its sweet spot is anywhere a small, self-contained, predictable model has to run
without a GPU, a Python runtime, or a network connection. Here are eleven
scenarios where that combination is a genuine advantage.

---

## 1. Offline text generation on edge devices

Train a causal-transformer SLM on a domain corpus, quantize it to int8, and ship
a single `.bin` file to a Raspberry Pi, an industrial gateway, or a kiosk. The
model generates text with no GPU, no Python, and no internet. The byte-level BPE
tokenizer means the same model handles any UTF-8 input — accented and non-Latin
scripts included — without an unknown-token escape hatch.

**Why Ferrum:** one self-contained file, CPU-only inference, 4× size reduction
from QAT, no runtime dependencies.

---

## 2. Running small open-weight models offline (GGUF import)

Import a small open-weight **Llama/Qwen** checkpoint in GGUF — including the
common `Q4_K`/`Q5_K`/`Q6_K` k-quants — along with its own tokenizer, and run it on
the CPU in int4/int8/f32. A streamed reader avoids holding the whole file in RAM,
and a memory guard warns before a model that won't fit. Be realistic: a ~1B model
decodes at only a few tokens per second here — fine for a patient, private,
offline demo; not an interactive chatbot.

**Why Ferrum:** run a downloaded model with zero Python and zero network, fully
auditable, with no `unsafe`; `--quant` trades RAM for speed as your hardware
demands.

---

## 3. Autocomplete and suggestion for niche domains

Train a small subword model on logs, command histories, code snippets, or chat
templates to power next-token suggestions inside a CLI, an editor plugin, or an
embedded form. A BPE vocabulary captures recurring multi-character motifs (flags,
identifiers, file paths) far more compactly than a character model.

**Why Ferrum:** BPE subword tokens, deterministic output for reproducible
suggestions, microsecond-to-millisecond latency on a CPU.

---

## 4. Privacy-preserving, on-device modeling

Because training and inference run locally with no external calls, sensitive
data — medical notes, financial records, internal documents — never leaves the
machine. Train on a private corpus and keep both the data and the model in your
own environment.

**Why Ferrum:** no telemetry, no network, no third-party crates that could
exfiltrate data; `#![forbid(unsafe_code)]` for auditability.

---

## 5. In-browser AI playgrounds (WebAssembly)

Compile a trained model and the `tabular_wasm` bindings to `wasm32` and run
inference entirely in the browser. Users interact with a live model on a static
page — no backend, no API keys, no per-request cost. Ideal for teaching demos,
interactive documentation, and "try it yourself" widgets.

**Why Ferrum:** pure Rust compiles cleanly to WASM; models embed their own
metadata so the UI can build itself from the file.

---

## 6. Reproducible research and teaching

Every layer, the optimizer, the tokenizer, the quantizer, and the file format are
written in readable Rust with no hidden CUDA kernels or opaque dependencies.
Students and researchers can read the entire forward *and* backward pass, set a
seed, and get bit-for-bit reproducible results — a transparent reference
implementation of attention, BPE, and quantization-aware training.

**Why Ferrum:** the whole stack is inspectable, deterministic, and
dependency-free.

---

## 7. Tabular classification and regression at the edge

Beyond language, `train_cli` trains MLP classifiers and regressors from any CSV,
auto-detecting the task, normalizing features, and exporting a self-contained
model. Deploy fraud scores, quality predictions, or sensor classifications to
devices that can't host a Python ML stack.

**Why Ferrum:** one command from CSV to a deployable model; the same FINF format
and WASM runtime as the SLMs.

---

## 8. Embedded and resource-constrained systems

With no allocator surprises from heavy dependencies and int8/int4 models a few
tens of kilobytes in size, Ferrum fits microcontroller-class budgets and firmware
images. The CPU-only design means no driver stack and predictable, single-threaded
execution (`FERRUM_NUM_THREADS=1`).

**Why Ferrum:** tiny quantized models, zero dependencies, no GPU/driver
requirements, deterministic timing.

---

## 9. Air-gapped and field deployments

Defense, scientific instruments, remote sensors, and secure facilities often
forbid network access entirely. A model trained elsewhere can be carried in as a
single file and run with a static binary that has no dependencies to vet or
update.

**Why Ferrum:** fully self-contained models and binaries; nothing to fetch at
runtime.

---

## 10. Cost-free, scalable CPU inference services

Running thousands of tiny CPU inferences is far cheaper than provisioning GPUs.
Embed `ferrum_core` directly in a Rust service to serve completions or
classifications at high throughput with no model server, no GPU scheduler, and no
inference framework to operate.

**Why Ferrum:** library-first design embeds in any Rust service; CPU inference
scales horizontally on commodity hardware.

---

## 11. Rapid prototyping of model ideas

Because there is no setup beyond `cargo run`, Ferrum is a fast sketchpad for
trying tokenization strategies, context lengths, and architectures. Compare a
one-hot MLP, an embedding MLP, and a transformer — and character-level versus
BPE — on the same corpus in minutes, all with identical APIs.

**Why Ferrum:** three interchangeable training paths, a single tokenizer knob
(`vocab_size`), instant builds, and deterministic comparisons.

---

## Choosing a configuration

| Goal                                   | Suggested setup                                              |
|----------------------------------------|-------------------------------------------------------------|
| Smallest possible model                | `train_embedded`, BPE vocab 256–512, int8 (or int4) save     |
| Highest quality on real text           | `train_transformer`, BPE vocab 512–2000, more blocks/heads   |
| Maximum transparency / teaching        | `train` (one-hot) or character-level transformer             |
| Multilingual / emoji-heavy text        | any path with BPE (`vocab_size >= 256`)                     |
| Run a small open model offline         | `run-gguf` (Llama/Qwen), `--quant int4` for RAM or `int8` for speed |
| Tabular data                           | `train_cli`                                                  |

See these in context with the worked [examples](example.md) and the
[evaluation guide](evaluation.md) to measure quality, size, and speed.
