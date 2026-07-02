# 3. The Ferrum Engine — What It Is, What It Can Do, and Where It Falls Short

> **Who this is for:** anyone who has read pages 1–2 (or already knows the basics)
> and now wants the full, honest tour of the actual project. We'll cover what Ferrum
> is made of, what each part does, the impressive things it can do, and — just as
> importantly — the things it can't.

---

## 3.1 What is Ferrum, in one sentence?

> **Ferrum is a zero-dependency, pure-Rust engine for building, training, and running
> causal Transformers, Small Language Models (SLMs), and classical neural networks —
> entirely on the CPU, with no GPU and no external libraries — and an importer that
> can run small open-weight Llama/Qwen models too.**

The name is a small joke: *ferrum* is Latin for **iron** (chemical symbol **Fe**),
and the project is the "iron" — the solid, no-frills metal — under your AI. It is
`std`-only and `#![forbid(unsafe_code)]` (see
[the Rust doc](02-rust-and-why-it-matters.md) for what those mean).

A trained Ferrum model is a **single self-contained `.bin` file** you can copy to a
server, a laptop, a Raspberry Pi, or a browser tab and run with nothing else
installed.

---

## 3.2 The shape of the project (the "workspace")

Ferrum is organised into several cooperating parts:

| Part | What it is | Think of it as… |
|------|-----------|-----------------|
| **`ferrum_core`** | The engine library — all the maths, layers, training, tokenizer, file format, and the GGUF importer/exporter | The brain. Everything else calls into this. |
| **`slm_cli`** | A command-line tool (binary `train_transformer`): train / generate / evaluate / inspect, **plus `run-gguf` / `finetune-gguf` / `export-gguf`** | The text-model workbench. |
| **`train_cli`** | A command-line tool for **tabular** models (spreadsheets/CSV) | The "predict-from-a-table" tool (not text). |
| **`tabular_wasm`** | Bindings that let models run in a **web browser** | The bridge to the web. |
| **`ferrum_gui`** | A point-and-click desktop app ("Ferrum SLM Studio") | The friendly face — see [the GUI guide](05-using-the-gui.md). |
| **`tests`** | Automated checks that everything still works | The safety net. |

The first three are the ones a beginner meets first; the GUI wraps all of them in
buttons.

---

## 3.3 What's inside the brain (`ferrum_core`)

Knowing the vocabulary helps you read the menus and docs. The building blocks,
assembled like LEGO:

- **Tensors** — the basic data containers (grids of numbers) that everything flows
  through.
- **Layers** — the reusable transformations from
  [page 1](01-generative-ai-slms-and-transformers.md): `Linear`, `Embedding`,
  `LayerNorm`, `Activation`, and the star, `TransformerBlock` (self-attention +
  feed-forward).
- **The tokenizer** — `ByteBpeTokenizer`, the byte-level BPE tool that can encode
  *any* text.
- **Training machinery** — hand-written backpropagation, plus the well-known `SGD`
  and `Adam` optimisers (with an AdamW weight-decay option and optional dropout).
- **Quantization** — the int8/int4 compression trick explained in §3.6.
- **The file format (FINF)** — how a finished model is saved into one file (§3.7).
- **A KV-cache** — a speed optimisation that remembers past work during generation.
- **A GGUF importer & exporter** — a from-scratch reader *and writer* for the
  standard open-model file format, plus a Llama/Qwen runner (§3.5b).

Everything is built only from Rust's standard library — there is genuinely no
NumPy-equivalent under the hood; the matrix multiplication, and even the GGUF
parser, are written out in the project itself.

---

## 3.4 The three ways to build a model

One of Ferrum's nicest teaching features: **three model recipes** that all share the
same commands and file format, so you can compare them on the same text and *see* the
trade-offs. From simplest to most powerful:

| Recipe | Architecture | Tokenizer | Best for |
|--------|-------------|-----------|----------|
| `train` | A flat "one-hot" MLP (the simplest neural net) | character-level | The absolute baseline; maximum transparency. |
| `train_embedded` | An embedding layer + MLP | character **or** BPE | Small, fast models that beat the baseline on size. |
| `train_transformer` | A real causal multi-head **Transformer** | character **or** BPE | The highest quality on real text. |

You choose your tokenizer with a single knob, `vocab_size` (the `--vocab` flag): `0`
→ **character-level**; any value `≥ 256` → a **byte-level BPE** tokenizer of that
size; values `1`–`255` are rejected (the 256-byte base alphabet is the irreducible
minimum).

> **Beginner tip:** start with `train_transformer` and `--vocab 512`.

---

## 3.5 What Ferrum can actually do — the capability list

Every feature here is implemented and tested in the project:

**Building & training**
- Train all three model families from any plain-text (UTF-8) file.
- Train tabular **classifiers and regressors** from any CSV with `train_cli`
  (auto-detecting which).
- **Watch training live** — loss is reported every epoch.
- **Multi-core training** — work is split across CPU cores (see §3.8).
- **Regularize** with AdamW weight decay (`--weight_decay`) and FFN dropout
  (`--dropout`).

**Generating**
- **Generate text** from a prompt, with a `temperature` creativity dial.
- **Stream** generation fragment-by-fragment (`--stream`), careful never to emit a
  broken half-character for multi-byte scripts.
- **Continuation-only** output — just the model's new text, without your prompt.

**Measuring quality**
- **Evaluate** on held-out text, reporting **perplexity**, cross-entropy, and
  bits-per-token (see §3.9).

**Running other people's models**
- **Import and run a small open-weight Llama/Qwen GGUF** with its own tokenizer
  (§3.5b).
- **Export back to GGUF**: write a loaded (and optionally fine-tuned) model out
  as a standard GGUF file that runs in llama.cpp / ollama / LM Studio (§3.5b).

**Shipping**
- Save a model as **one self-contained file**; load it anywhere.
- **Quantize** to int8 (~4× smaller) or int4 (~8×) (§3.6).
- Compile to **WebAssembly** to run in a browser.
- **Deterministic** results: same seed + settings → bit-for-bit identical output,
  on any number of CPU cores.

---

## 3.5b Running, fine-tuning, and exporting external models — the GGUF importer

Beyond models you train, Ferrum can **import a small open-weight model** that someone
else trained, in the standard **GGUF** format used across the open-model world. The
reader — written from scratch in safe Rust — understands the common quantized layouts
(including the `Q4_K`/`Q5_K`/`Q6_K` "k-quants" most downloads use) and reconstructs
the model's **own tokenizer** from the file, so the imported model runs on real text.

What's honest about it:

- It runs **Llama/Qwen-family** models (the architecture Ferrum implements:
  RMSNorm, RoPE, grouped-query attention, SwiGLU). A few exotic quant formats and
  non-Llama architectures are politely refused.
- It is **slow**, by physics, not by bug: a ~1B model decodes at only a few tokens
  per second on a CPU, with tens of seconds to digest a long prompt (§3.8 explains
  why). Great for a patient, private, offline demo; not a chatbot.
- The import is **lossy** (it re-packs the weights onto Ferrum's own grid) and is
  *not* bit-identical to a dedicated runner like llama.cpp.
- The architecture is even **trainable**: a finite-difference-checked backward pass
  lets you fine-tune a *small* imported model. A 1B model, though, is out of reach
  on one CPU — its optimizer state alone would need ~16 GB of RAM (§3.10).
- And the road runs **both ways**: `export-gguf` writes the loaded (or fine-tuned)
  model back out as a standard GGUF v3 file — re-quantized to your chosen level
  (`q8_0`, `q4_k`, …, or lossless `f16`/`f32`), with the source's tokenizer and
  settings carried along — so the result runs unchanged in llama.cpp, ollama, or
  LM Studio.

Use it from the command line (`train_transformer run-gguf model.gguf "prompt"
--quant int4`, `train_transformer export-gguf in.gguf out.gguf --quant q8_0`)
or the GUI's **GGUF** tab (import/run only for now).

---

## 3.6 Quantization: making models 4×–8× smaller without breaking them

A model's knowledge lives in many numbers. Stored at full precision, each takes 32
bits (4 bytes). **Quantization** stores each using just **8 bits (1 byte)** — an
immediate **~4× reduction** — or even **4 bits** (~8×).[^int8]

The catch is that squishing loses precision. Ferrum uses the better remedy,
**Quantization-Aware Training (QAT)**: the model is made *aware*, during training,
that it will be squished, so it learns weights that survive it. QAT is "the de facto
approach towards designing robust quantized models with low error," and "usually
yields higher accuracy than" quantizing only after training.[^qat]

Beyond the size win, 8-bit integer maths can run **2×–3× faster on a CPU** than
full-precision on hardware with the right instructions.[^speed] Ferrum's models are
tens of kilobytes as a result — small enough to embed almost anywhere. (The same
int4/int8 packing is what makes the imported GGUF runner from §3.5b feasible on a
CPU — see [the benchmarks](../benchmarks.md) §3d.)

[^int8]: "INT8 quantization offers a 4× model compression compared to FP32 models while reaching comparable accuracy levels." APXML, *Model Quantization Techniques: INT8 and FP8* https://apxml.com/courses/advanced-ai-infrastructure-design-optimization/chapter-4-high-performance-model-inference/model-quantization-techniques
[^qat]: NVIDIA Technical Blog, *Improving INT8 Accuracy Using Quantization Aware Training* https://developer.nvidia.com/blog/improving-int8-accuracy-using-quantization-aware-training-and-tao-toolkit/
[^speed]: "Quantized inference at 8-bits can provide 2x-3x speed-up on a CPU." Krishnamoorthi, *Quantizing deep convolutional networks for efficient inference* (arXiv 1806.08342) https://arxiv.org/pdf/1806.08342

---

## 3.7 One file to rule them all: the FINF format

When you save a model, Ferrum writes a single **FINF** file bundling *everything the
model needs to run*: the weights, the metadata (architecture, vocabulary size, task
type), the normalizer (for tabular models), **and** the tokenizer's learned merge
list (for BPE text models).

There is **no separate vocabulary file, config file, or tokenizer file** to keep
track of — a frequent source of pain in other ML stacks. You ship one `.bin` and it
works. Two versions exist, and the loader reads both automatically: **FINF v4** stores
full-precision (f32) weights; **FINF v5** adds int8 *and* int4 (chosen per weight, so
small bias vectors can stay f32 while big matrices shrink). Older files made before
the tokenizer feature still load — they simply default to character-level. That
backward-compatibility care is a sign of a mature, honestly-maintained format.

---

## 3.8 Running fast on a CPU with no GPU and no libraries

Ferrum never uses a GPU. To stay fast it spreads the heaviest maths (the matrix
multiplications behind every layer) across **all your CPU cores** at once, using a
**persistent pool of worker threads** — spawned once and reused — so generating a long
passage doesn't pay a fresh "start up threads" cost on every step.[^pool] This is all
built using **only Rust's standard library**, with **no `unsafe` and no external
crates**, and the results are **bit-for-bit identical no matter how many cores you
use**. You can control the core count with `FERRUM_NUM_THREADS` (set it to `1` for
fully serial, predictable execution on a single-core chip).

**What the real benchmarks show** (measured, reproducible from `benchmarks.md`):

- **The matrix-multiply kernel itself scales well** — about **2.4× to 4.0×** across 8
  cores for large multiplies. That is the engine's true parallel ceiling.
- **End-to-end training scales less, and it depends on model size.** A small model
  saw only about **1.2×**, because much of a training step is inherently sequential
  (the loop over examples, softmax, the optimizer update) and the multi-core
  training also pays a per-worker setup cost. Bigger models, where the big
  matmuls dominate each step, get closer to the kernel's ceiling.
- **Generation barely parallelises.** Producing text is a long chain of *small,
  dependent* steps, so a small model's generation was essentially **flat** across
  thread counts. The pool's real value during generation is removing thread-creation
  overhead; for speed, prefer a BPE vocabulary and a smaller network over more cores.

Why so slow for big models, fundamentally? Generating each token streams *every
weight* through the processor once — it's **memory-bandwidth-bound**, and a CPU's
bandwidth is far below a GPU's. Quantization helps a lot here: Ferrum's int4 decode
path runs about **4–5× faster than f32** for a 1B-class model (it moves ⅛ the bytes
and uses a multi-core column split), but "faster" still means *a few tokens per
second*, not interactive speed.

> **Honest takeaway:** Ferrum makes good, deterministic use of multiple cores, but the
> laws of physics still apply — CPU generation of larger models is *slow* compared to
> GPU systems. This is a feature-trade, not a flaw, covered candidly in
> [`07-critique.md`](07-critique.md) and [`08-applications.md`](08-applications.md).

[^pool]: Project documentation, `benchmarks.md` and `docs/FAQs.md` — persistent worker pool, `std`-only, deterministic across thread counts.

---

## 3.9 How you know a model is any good: perplexity

A beginner's trap is to watch the training **loss** fall and conclude "great, it
learned!" But a model can get a low *training* loss simply by **memorising** the
training text — like a student who memorises the exam answers without understanding.

The real test is performance on **held-out text** the model never saw. Ferrum's
`evaluate` function (and the GUI's **Evaluate** tab) measures this with
**perplexity**:

- **Perplexity** is, intuitively, "how surprised the model is by real text." Lower is
  better; a perfect model scores `1.0`. A model that learned *nothing* scores roughly
  the vocabulary size.
- A big gap between low *training* perplexity (~1.0) and higher *held-out* perplexity
  is the tell-tale sign of memorisation on a too-small corpus.

This built-in honesty check is one of Ferrum's best teaching features: it actively
shows you when your model is fooling you.

---

## 3.10 The shortcomings — read this before you build anything serious

Ferrum is excellent at what it's for, and genuinely limited outside that. (The
deeper, more pointed version is in [`07-critique.md`](07-critique.md); here are the
practical shortcomings.)

1. **The models are *tiny*.** Trained on a small corpus, they **memorise more than
   they generalise**, and they will not write coherent long-form prose, hold a
   conversation, or answer general-knowledge questions. *Not* a local ChatGPT.

2. **CPU-only means slow at scale.** No GPU, ever. Perfect for tiny models and edge
   devices, but a hard ceiling on how big a model you can realistically train or run.

3. **"Zero dependencies" costs peak speed.** By writing its own maths instead of using
   hyper-optimised libraries (OpenBLAS, Intel MKL), Ferrum trades some raw performance
   for purity and readability. It is not the fastest CPU engine, and doesn't try to
   be.

4. **You train your own models (mostly).** Ferrum can *run* a small downloaded model,
   but it ships no pretrained weights of its own and is built to train from your text.
   Quality is bounded by *your* data and *your* patience — which is why
   [`06-data-gigo-and-why-good-data-wins.md`](06-data-gigo-and-why-good-data-wins.md)
   may be the most important page in this manual.

5. **The GGUF runner is real but limited.** It runs *small* Llama/Qwen models slowly
   (a few tok/s for ~1B), re-quantizes them lossily, refuses a few exotic formats, and
   cannot train a 1B model (the optimizer state alone would exceed this machine's
   RAM). It's a private, offline runner for small open models — not a llama.cpp
   replacement.

6. **It inherits every limitation of language models in general.** Like all such
   models it predicts plausible text — it does not *understand*, *reason reliably*, or
   *know facts*. Smaller models are *more* prone to errors and "hallucination" than
   large ones.[^slmlimit] See [`07-critique.md`](07-critique.md).

7. **The GUI's heavy build.** The desktop app's Rust backend type-checks, but the
   full windowed app depends on system WebView libraries and may need its
   prerequisites installed and verified on your machine (the project says so openly).

[^slmlimit]: "Smaller models may exhibit fewer reasoning capabilities and more hallucinated behaviors… likely due to the lack of scale." *On the Fundamental Limits of LLMs at Scale* (arXiv 2511.12869) https://arxiv.org/pdf/2511.12869

---

## 3.11 So who is Ferrum *for*?

Putting the capabilities and the limits together, Ferrum is an outstanding choice when
you want **a small, transparent, self-contained model you fully control** — for
**learning** how AI really works, for **privacy/offline** use, for **embedded and
edge** tasks, for **narrow, repetitive** text or tabular jobs, and for **running a
small open model privately** when slow is acceptable. It is the wrong choice when you
want broad, general, human-like intelligence — that still needs the giant cloud
models.

The next two pages get concrete: [how Ferrum handles the world's many languages, plus
a catalogue of real use cases](04-non-english-text-and-practical-uses.md), and [the
friendly GUI](05-using-the-gui.md).

---

## Sources

- APXML, *Model Quantization Techniques: INT8 and FP8* — https://apxml.com/courses/advanced-ai-infrastructure-design-optimization/chapter-4-high-performance-model-inference/model-quantization-techniques
- NVIDIA Technical Blog, *Improving INT8 Accuracy Using Quantization Aware Training* — https://developer.nvidia.com/blog/improving-int8-accuracy-using-quantization-aware-training-and-tao-toolkit/
- Krishnamoorthi, *Quantizing deep convolutional networks for efficient inference* (arXiv 1806.08342) — https://arxiv.org/pdf/1806.08342
- *On the Fundamental Limits of LLMs at Scale* (arXiv 2511.12869) — https://arxiv.org/pdf/2511.12869
- Ferrum project docs: `readme.md`, `benchmarks.md`, `status.md`, `docs/FAQs.md` (in this repository)
