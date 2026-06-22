# 4. Non-English Text, and a Catalogue of Practical Uses

> **Who this is for:** anyone wondering "will this work on *my* language or *my*
> data?" and "what would I actually *do* with it?" This page answers both —
> first the multilingual question, then a dedicated section of concrete,
> grounded use cases.

---

## Part A — Will it work on non-English text?

### 4.1 The short answer: yes, including non-Latin scripts and emoji

Many older text systems quietly assume English (or at least the Latin alphabet).
The moment you feed them Chinese characters, Arabic script, Cyrillic, Indian
scripts, or an emoji, they choke — producing a dreaded "unknown character"
placeholder (the `<UNK>` token) and losing information.

Ferrum is built so this **cannot happen**, and the reason is the **byte-level
BPE tokenizer** introduced on [page 1](01-generative-ai-slms-and-transformers.md).

### 4.2 Why byte-level tokenization is the key

Every piece of digital text — in any language on Earth — is ultimately stored as
a sequence of **bytes** (values from 0 to 255), via an encoding called **UTF-8**.
A Latin "a", a Chinese "好", an Arabic "م", and a 😀 emoji are all just specific
patterns of bytes.

Ferrum's tokenizer starts its vocabulary from **all 256 possible byte values**.
Because *every* possible file is made of those 256 values, **there is no input it
cannot represent.** As the project's own documentation puts it, the base
vocabulary is the full byte range, so "any UTF-8 text — emoji, Cyrillic, CJK,
control characters — round-trips with no unknown-token escape hatch."

This mirrors exactly the design GPT-2 pioneered and GPT-4 still uses: starting
from 256 raw byte tokens "guarantees that literally any byte sequence has a valid
tokenization — no `<UNK>` token needed, ever."[^bytelevel] Ferrum is a small,
readable implementation of the same proven idea.

> **"Round-trips" means:** if you encode text into tokens and then decode it back,
> you get *exactly* the original bytes — nothing is lost or corrupted, in any
> language.

[^bytelevel]: Byte-level BPE starts from 256 byte tokens so any byte sequence has a valid tokenization with no unknown token. Hugging Face course, *Byte-Pair Encoding* https://huggingface.co/learn/llm-course/en/chapter6/5 ; A. Karpathy, *minbpe* https://github.com/karpathy/minbpe

### 4.3 What this means in practice

- **Train on any language.** Point Ferrum at a French, Hindi, Japanese, Swahili,
  or mixed-language corpus and it trains the same way it does on English. Just use
  a BPE vocabulary (`--vocab 256` or higher) rather than character-level.
- **Mixed and messy text is fine.** Source code, chat logs full of emoji,
  documents that switch between scripts mid-sentence — all encode cleanly.
- **Streaming stays correct.** When generating multi-byte characters one fragment
  at a time, Ferrum deliberately *holds back* a partial trailing character until
  it's complete, so you never see a momentary "�" placeholder that then gets
  fixed. That's careful, multilingual-aware engineering.

### 4.4 The honest caveats for non-English use

The *tokenizer* handles any language flawlessly. But **quality still depends on
your data and on a few realities**:

1. **The model only learns what's in your corpus.** A model trained on English
   won't speak French. Train on the language(s) you actually need.
2. **BPE compresses, which can surprise you.** BPE packs frequent byte-patterns
   into single tokens. For some scripts, a single visible character may be several
   bytes and may tokenize into multiple tokens — so a short-looking text can have
   fewer *tokens* than you'd expect. If your corpus is short or very repetitive,
   you may hit the "corpus must be longer than the context window" message; the
   fix is more text, a smaller context, or a smaller vocabulary.
3. **Languages without spaces** (e.g. Chinese, Japanese) work fine at the byte
   level, but you may want a larger vocabulary so the model can learn meaningful
   multi-byte units.
4. **Right-to-left scripts** (Arabic, Hebrew) are stored and generated correctly
   byte-for-byte; how they *display* is up to your terminal or app, not Ferrum.

> **Bottom line:** Ferrum is genuinely script-agnostic — a real strength for
> multilingual and non-English work — provided you train it on representative
> text in the language you care about.

---

## Part B — Practical use cases

> Ferrum's sweet spot is **anywhere a small, self-contained, predictable model
> must run without a GPU, a Python runtime, or a network connection.** The
> scenarios below are drawn from the project's own documented use cases and
> grounded in where the wider industry actually deploys small, on-device models.
> Be realistic: these play to Ferrum's strengths (small, private, offline,
> embeddable) — not to writing essays.

### 4.5 The ten documented scenarios

| # | Use case | Why Ferrum fits |
|---|----------|-----------------|
| 1 | **Offline text generation on edge devices** | One self-contained `.bin`, CPU-only, ~4× smaller via int8, no runtime to install. Ship it to a Raspberry Pi, kiosk, or gateway. |
| 2 | **In-browser AI playgrounds (WebAssembly)** | Compiles to WASM; runs entirely in the visitor's browser — no backend, no API keys, no per-request cost. Ideal for teaching demos. |
| 3 | **Autocomplete / suggestion for niche domains** | Train a small BPE model on logs, command histories, or code snippets to power next-token suggestions in a CLI or editor, with microsecond-to-millisecond latency. |
| 4 | **Privacy-preserving, on-device modeling** | Training and inference are local; sensitive data (medical notes, financial records) never leaves the machine. No telemetry, no network, auditable code. |
| 5 | **Reproducible research & teaching** | The entire forward/backward pass, tokenizer, and file format are readable Rust with no hidden CUDA. Set a seed → bit-for-bit reproducible results. |
| 6 | **Tabular classification/regression at the edge** | `train_cli` turns any CSV into a deployable model — fraud scores, quality predictions, sensor classifications — on devices that can't host a Python ML stack. |
| 7 | **Embedded / resource-constrained systems** | Int8 models are tens of kilobytes with no driver or GPU dependency and predictable, serial timing (`FERRUM_NUM_THREADS=1`). |
| 8 | **Air-gapped & field deployments** | A model trained elsewhere is carried in as one file and run by a static binary with nothing to fetch or update — defense, instruments, secure facilities. |
| 9 | **Cost-free, scalable CPU inference services** | Embed `ferrum_core` directly in a Rust service; thousands of tiny CPU inferences are far cheaper than provisioning GPUs. |
| 10 | **Rapid prototyping of model ideas** | Compare one-hot vs. embedding vs. transformer, and character vs. BPE, on the same corpus in minutes with identical APIs. |

These are not hypothetical marketing categories — they map directly to the
project's `usecases.md`, and each plays to a *real* property of the engine
(self-contained file, CPU-only, deterministic, dependency-free, browser-capable).

### 4.6 These align with where the industry really uses small models

The same strengths are exactly why **TinyML and on-device "edge AI"** are a fast-
growing field: running inference *locally* on microcontrollers and small CPUs
gives **instant responses, multi-month battery life, data privacy, and offline
operation** — "no need to send data to remote servers."[^tinyml] Documented edge
deployments run on chips like the ESP32 and ARM Cortex-M series with sub-second
latency and milliamp power draws.[^edge] Ferrum is a CPU/edge-oriented engine in
the same spirit (it targets small CPUs and microcontroller-class budgets rather
than sub-milliwatt chips specifically). For a careful breakdown of what CPU-bound
inference is and isn't good for, see [`08-applications.md`](08-applications.md).

[^tinyml]: "TinyML performs inference locally with no need to send data to remote servers… instant responses… devices can run for months or even years on small batteries… sensitive data never leaves the device." Talent500, *What Is TinyML?* https://talent500.com/blog/what-is-tinyml-introduction/
[^edge]: A trained model deployed on an ESP32-S3 achieved ~113.6 ms inference latency at ~5.78 mA, enabling up to 7 days of continuous operation. *Deploying TinyML for energy-efficient object detection* (Nature Scientific Reports, 2025) https://www.nature.com/articles/s41598-025-27818-9

### 4.7 A concrete starter idea you can build today

A realistic first project that genuinely suits Ferrum:

> **A private, offline command-line autocomplete.** Collect your own shell
> command history (a plain text file). Train a small BPE transformer on it. Now
> you have a tiny model that suggests the next command — running locally,
> instantly, with your history never leaving your laptop. It's useful, it's
> private, and it teaches you the whole pipeline.

That single example exercises every strength on this page: any-text tokenization,
CPU-only training, a self-contained model, privacy, and determinism.

---

## What you now know

- Ferrum handles **any language and any script** because it tokenizes at the byte
  level — the same proven design as GPT-2/GPT-4 — so there are no "unknown
  character" failures.
- Quality still depends on training with **representative data** in your target
  language, and BPE's compression has a few practical quirks to watch.
- Ferrum's **practical use cases** cluster around small, private, offline,
  embeddable, and educational tasks — the same niche where on-device "edge AI" is
  booming.

Next: [the friendly point-and-click app](05-using-the-gui.md). For an unflinching
look at the limits, see [`07-critique.md`](07-critique.md), and for the
applications-vs-no-applications verdict, [`08-applications.md`](08-applications.md).

---

## Sources

- Hugging Face LLM Course, *Byte-Pair Encoding tokenization* — https://huggingface.co/learn/llm-course/en/chapter6/5
- A. Karpathy, *minbpe* — https://github.com/karpathy/minbpe
- Talent500, *What Is TinyML? A Guide to Tiny Machine Learning on Edge Devices* — https://talent500.com/blog/what-is-tinyml-introduction/
- *Deploying TinyML for energy-efficient object detection and communication in low-power edge AI systems*, Nature Scientific Reports (2025) — https://www.nature.com/articles/s41598-025-27818-9
- Ferrum project docs: `usecases.md`, `docs/FAQs.md`, `ferrum_core/src/tokenizer.rs` (in this repository)
