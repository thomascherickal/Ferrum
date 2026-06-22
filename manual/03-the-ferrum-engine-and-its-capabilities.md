# 3. The Ferrum Engine — What It Is, What It Can Do, and Where It Falls Short

> **Who this is for:** anyone who has read pages 1–2 (or already knows the basics)
> and now wants the full, honest tour of the actual project. We'll cover what
> Ferrum is made of, what each part does, the impressive things it can do, and —
> just as importantly — the things it can't.

---

## 3.1 What is Ferrum, in one sentence?

> **Ferrum is a zero-dependency, pure-Rust engine for building, training, and
> running causal Transformers, Small Language Models (SLMs), and classical neural
> networks — entirely on the CPU, with no GPU and no external libraries.**

The name is a small joke: *ferrum* is Latin for **iron** (chemical symbol **Fe**),
and the project is the "iron" — the solid, no-frills metal — under your AI. It is
`std`-only and `#![forbid(unsafe_code)]` (see [the Rust doc](02-rust-and-why-it-matters.md)
for what those mean).

A trained Ferrum model is a **single self-contained `.bin` file** that you can
copy to a server, a laptop, a Raspberry Pi, or a browser tab and run with nothing
else installed.

---

## 3.2 The shape of the project (the "workspace")

Ferrum is organised into several cooperating parts. You don't need to use all of
them; here's what each is for, in plain terms:

| Part | What it is | Think of it as… |
|------|-----------|-----------------|
| **`ferrum_core`** | The engine library — all the maths, layers, training, tokenizer, and file format | The brain. Everything else calls into this. |
| **`slm_cli`** | A command-line tool (its binary is named `train_transformer`) | The text-model workbench: train / generate / evaluate / inspect. |
| **`train_cli`** | A command-line tool for **tabular** models (spreadsheets/CSV) | The "predict-from-a-table" tool (not text). |
| **`tabular_wasm`** | Bindings that let models run in a **web browser** | The bridge to the web. |
| **`ferrum_gui`** | A point-and-click desktop app ("Ferrum SLM Studio") | The friendly face — see [the GUI guide](05-using-the-gui.md). |
| **`tests`** | Automated checks that everything still works | The safety net. |

The first three are the ones a beginner meets first; the GUI wraps all of them in
buttons.

---

## 3.3 What's inside the brain (`ferrum_core`)

You never have to read this code, but knowing the vocabulary helps you understand
the menus and the docs. These are the building blocks, assembled like LEGO:

- **Tensors** — the basic data containers (grids of numbers) that everything
  flows through.
- **Layers** — the reusable transformations from [page 1](01-generative-ai-slms-and-transformers.md):
  `Linear` (a basic maths step), `Embedding` (ID → numbers), `LayerNorm`
  (stabiliser), `Activation` (adds non-linearity), and the star,
  `TransformerBlock` (self-attention + feed-forward).
- **The tokenizer** — `ByteBpeTokenizer`, the byte-level BPE tool from
  [page 1](01-generative-ai-slms-and-transformers.md) that can encode *any* text.
- **Training machinery** — hand-written backpropagation, plus two well-known
  optimisers (`SGD` and `Adam`) that decide *how* to nudge the weights.
- **Quantization** — the int8 compression trick explained in §3.6.
- **The file format (FINF)** — how a finished model is saved into one file (§3.7).
- **A KV-cache** — a speed optimisation that remembers past work during
  generation so each new token is cheaper to produce.

Everything is built only from Rust's standard library — there is genuinely no
NumPy-equivalent under the hood; the matrix multiplication is written out in the
project itself.

---

## 3.4 The three ways to build a model

This is one of Ferrum's nicest teaching features: it offers **three different
model recipes** that all share the same commands and file format, so you can
compare them on the same text and *see* the trade-offs. From simplest to most
powerful:

| Recipe | Architecture | Tokenizer | Best for |
|--------|-------------|-----------|----------|
| `train` | A flat "one-hot" MLP (the simplest possible neural net) | character-level | The absolute baseline; maximum transparency. |
| `train_embedded` | An embedding layer + MLP | character **or** BPE | Small, fast models that beat the baseline on size. |
| `train_transformer` | A real causal multi-head **Transformer** | character **or** BPE | The highest quality on real text. |

You choose your tokenizer with a single knob, `vocab_size` (the `--vocab` flag):

- `0` → simple **character-level** tokenization.
- any value `≥ 256` → trains a **byte-level BPE** tokenizer of that size.
- values `1`–`255` are rejected, because the 256-byte base alphabet is the
  irreducible minimum (you can't have fewer tokens than there are byte values).

> **Beginner tip:** start with `train_transformer` and `--vocab 512`. It's the
> path the documentation's walkthrough uses and it gives the best results on
> ordinary text.

---

## 3.5 What Ferrum can actually do — the capability list

Here is the honest inventory of features, all of which are implemented and tested
in the project:

**Building & training**
- Train all three model families from any plain-text (UTF-8) file.
- Train tabular **classifiers and regressors** from any CSV with `train_cli` —
  it even auto-detects whether your task is classification or regression.
- **Watch training live** — loss is reported every epoch.
- **Multi-core training** — work is split across all your CPU cores
  automatically (see §3.8).

**Generating**
- **Generate text** from a prompt, with a `temperature` creativity dial.
- **Stream** generation — receive the text fragment-by-fragment as it's produced
  (great for a live, "typing" feel), and it's careful never to emit a broken
  half-character for multi-byte scripts.
- **Continuation-only** output — get just the model's new text without your
  prompt glued on the front (handy for chat-style replies and autocomplete).

**Measuring quality**
- **Evaluate** a finished model on held-out text it never saw, reporting
  **perplexity**, cross-entropy, and bits-per-token — the standard, objective way
  to tell whether a model truly *learned* versus merely *memorised* (see §3.9).

**Shipping**
- Save a model as **one self-contained file** and load it anywhere.
- **Quantize** to int8 for ~4× smaller files (§3.6).
- Compile to **WebAssembly** to run in a browser.
- **Deterministic** results: same seed + same settings → bit-for-bit identical
  output, every time, on any number of CPU cores.

---

## 3.6 Quantization: making models 4× smaller without breaking them

A model's knowledge lives in millions of numbers. Stored at full precision, each
number takes 32 bits (4 bytes). **Quantization** is the trick of storing each
number using just **8 bits (1 byte)** instead — an immediate **~4× reduction in
file size**.[^int8]

The catch is that squishing numbers loses precision, which can hurt accuracy.
Ferrum uses the better of the two standard remedies, **Quantization-Aware
Training (QAT)**: the model is made *aware*, during training, that it will
ultimately be squished, so it learns weights that survive the squishing. QAT is
"the de facto approach towards designing robust quantized models with low
error," and "usually yields higher accuracy than" quantizing only after
training.[^qat]

Beyond the 4× size win, 8-bit integer maths can also run **2×–3× faster on a
CPU** than full-precision maths on hardware with the right instructions.[^speed]
Ferrum's models are tens of kilobytes as a result — small enough to embed almost
anywhere.

[^int8]: "INT8 quantization offers a 4× model compression compared to FP32 models while reaching comparable accuracy levels." APXML, *Model Quantization Techniques: INT8 and FP8* https://apxml.com/courses/advanced-ai-infrastructure-design-optimization/chapter-4-high-performance-model-inference/model-quantization-techniques
[^qat]: NVIDIA Technical Blog, *Improving INT8 Accuracy Using Quantization Aware Training* https://developer.nvidia.com/blog/improving-int8-accuracy-using-quantization-aware-training-and-tao-toolkit/
[^speed]: "Quantized inference at 8-bits can provide 2x-3x speed-up on a CPU." Krishnamoorthi, *Quantizing deep convolutional networks for efficient inference* (arXiv 1806.08342) https://arxiv.org/pdf/1806.08342

---

## 3.7 One file to rule them all: the FINF format

When you save a model, Ferrum writes a single file in its own **FINF** format
that bundles *everything the model needs to run*:

- the weights,
- the metadata (architecture, vocabulary size, task type),
- the normalizer (for tabular models), **and**
- the tokenizer's learned merge list (for BPE text models).

There is **no separate vocabulary file, config file, or tokenizer file** to keep
track of — a frequent source of pain in other ML stacks. You ship one `.bin` and
it works.

There are two versions, and the loader reads both automatically:
- **FINF v4** stores full-precision (f32) weights.
- **FINF v5** adds int8 quantization (the smaller, default save).

Older files made before the tokenizer feature existed still load fine — they
simply default to character-level tokenization. That backward-compatibility care
is a sign of a mature, honestly-maintained format.

---

## 3.8 Running fast on a CPU with no GPU and no libraries

Ferrum never uses a GPU. To stay fast it spreads the heaviest maths (the matrix
multiplications behind every layer) across **all your CPU cores** at once. It does
this with a **persistent pool of worker threads** — spawned once and reused — so
that generating a long passage doesn't pay a fresh "start up threads" cost on
every single step.[^pool]

Crucially, this is all built using **only Rust's standard library** (threads and
safe shared references), with **no `unsafe` code and no external crates**, and the
results are **bit-for-bit identical no matter how many cores you use**. You can
control the core count with the `FERRUM_NUM_THREADS` setting (or force fully
serial, predictable execution by setting it to `1`, which is what you'd want on a
single-core embedded chip).

**What the real benchmarks show** (measured on an 8-core machine, and reproducible
from the project's `benchmarks.md`):

- **Training** sped up about **2.3×** on 8 cores versus 1 core. (It's not 8×
  because some steps are inherently sequential — a normal, honest result.)
- Switching from "spawn threads per operation" to the **persistent pool** roughly
  **halved** training time again for one configuration (2.24× faster).
- **Generation** parallelises only modestly — best around **1.25× at 4 threads** —
  because producing text is a long chain of small, dependent steps. The pool's
  main value during generation is removing thread-creation overhead, not big
  multi-core scaling. The sweet spot for generating is 2–4 threads.

> **Honest takeaway:** Ferrum makes good, deterministic use of multiple cores,
> but the laws of physics still apply — CPU generation of larger models is *slow*
> compared to GPU systems. This is a feature-trade, not a flaw, and it's covered
> candidly in [`07-critique.md`](07-critique.md) and [`08-applications.md`](08-applications.md).

[^pool]: Project documentation, `benchmarks.md` and `docs/FAQs.md` — persistent worker pool, `std`-only, deterministic across thread counts.

---

## 3.9 How you know a model is any good: perplexity

A beginner's trap is to watch the training **loss** fall, see it hit a low
number, and conclude "great, it learned!" But a model can get a low *training*
loss simply by **memorising** the training text — like a student who memorises the
exam answers without understanding the subject.

The real test is performance on **held-out text** the model never saw. Ferrum's
`evaluate` function (and the GUI's **Evaluate** tab) measures this with
**perplexity**:

- **Perplexity** is, intuitively, "how surprised the model is by real text." Lower
  is better; a perfect model scores `1.0`. A model that learned *nothing* scores
  roughly the vocabulary size.
- A big gap between low *training* perplexity (~1.0) and higher *held-out*
  perplexity is the tell-tale sign of memorisation on a too-small corpus.

This built-in honesty check is one of Ferrum's best teaching features: it
actively shows you when your model is fooling you.

---

## 3.10 The shortcomings — read this before you build anything serious

Ferrum is excellent at what it's for, and genuinely limited outside that. Being
clear about both is the whole point of this manual. (The deeper, more pointed
version of this discussion is in [`07-critique.md`](07-critique.md); here are the
practical shortcomings.)

1. **The models are *tiny*.** Ferrum builds models that are a fraction of the size
   of even a commercial "small" model. Trained on a small corpus, they will
   **memorise more than they generalise**, and they will not write coherent
   long-form prose, hold a conversation, or answer general-knowledge questions.
   They are *not* a local ChatGPT.

2. **CPU-only means slow at scale.** No GPU is used, ever. That's perfect for tiny
   models and edge devices, but it puts a hard ceiling on how big a model you can
   realistically train or run.

3. **"Zero dependencies" costs peak speed.** By writing its own maths instead of
   using hyper-optimised libraries (OpenBLAS, Intel MKL), Ferrum trades some raw
   performance for purity and readability. It is not the fastest CPU engine in
   existence, and doesn't try to be.

4. **You train your own models.** Ferrum does not download giant pre-trained
   models. You bring a text corpus and train from scratch. That's a feature for
   learning and privacy, but it means quality is bounded by *your* data and *your*
   patience — which is exactly why
   [`06-data-gigo-and-why-good-data-wins.md`](06-data-gigo-and-why-good-data-wins.md)
   may be the most important page in this manual.

5. **It inherits every limitation of language models in general.** Like all such
   models, it predicts plausible text — it does not *understand*, *reason
   reliably*, or *know facts*. Smaller models are *more* prone to errors and
   "hallucination" than large ones.[^slmlimit] See [`07-critique.md`](07-critique.md).

6. **The GUI's heavy build.** The desktop app depends on system WebView libraries
   and was authored in an environment that couldn't compile it; the *engine* it
   wraps is fully tested, but the GUI itself may need its prerequisites installed
   and verified on your machine (the project says so openly).

[^slmlimit]: "Smaller models may exhibit fewer reasoning capabilities and more hallucinated behaviors… likely due to the lack of scale." *On the Fundamental Limits of LLMs at Scale* (arXiv 2511.12869) https://arxiv.org/pdf/2511.12869

---

## 3.11 So who is Ferrum *for*?

Putting the capabilities and the limits together, Ferrum is an outstanding choice
when you want **a small, transparent, self-contained model you fully control** —
for **learning** how AI really works, for **privacy/offline** use, for **embedded
and edge** tasks, and for **narrow, repetitive** text or tabular jobs. It is the
wrong choice when you want broad, general, human-like intelligence — that still
needs the giant cloud models.

The next two pages get concrete: [how Ferrum handles the world's many languages,
plus a catalogue of real use cases](04-non-english-text-and-practical-uses.md),
and [the friendly GUI](05-using-the-gui.md).

---

## Sources

- APXML, *Model Quantization Techniques: INT8 and FP8* — https://apxml.com/courses/advanced-ai-infrastructure-design-optimization/chapter-4-high-performance-model-inference/model-quantization-techniques
- NVIDIA Technical Blog, *Improving INT8 Accuracy Using Quantization Aware Training* — https://developer.nvidia.com/blog/improving-int8-accuracy-using-quantization-aware-training-and-tao-toolkit/
- Krishnamoorthi, *Quantizing deep convolutional networks for efficient inference* (arXiv 1806.08342) — https://arxiv.org/pdf/1806.08342
- *On the Fundamental Limits of LLMs at Scale* (arXiv 2511.12869) — https://arxiv.org/pdf/2511.12869
- Ferrum project docs: `readme.md`, `benchmarks.md`, `status.md`, `docs/FAQs.md` (in this repository)
