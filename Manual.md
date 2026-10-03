# The Ferrum Manual

This manual covers everything about *using* Ferrum. It starts from first
principles for readers who have never trained a model. It then walks through
preparing data, training, generating, evaluating, running external models, the
GUI, and deployment. It ends with the complete reference.

For a one-page overview and installation, see [ReadMe.md](ReadMe.md). For what
is implemented, tested, and planned, see [Status.md](Status.md).

**How to read it.**

- New to all of this? Read Part I in order, then §6.10, a complete worked
  walkthrough.
- Know the basics already? Go to Part II and read only the chapter for the
  task in front of you.
- Integrating the library? Part III is the reference.

The manual separates *excitement* from *honesty* on purpose. Several chapters
explain what Ferrum and small models can do. [§17](#17-limits-an-honest-critique)
is about what they cannot do, and never will. [§3](#3-data-decides-garbage-in-garbage-out)
argues the point beginners miss most often: the **data**, not the model,
usually decides whether a project succeeds.

---

## Contents

**Part I — Foundations**
1. [Generative AI, SLMs, and Transformers from zero](#1-generative-ai-slms-and-transformers-from-zero)
2. [Why Rust, and what "zero dependencies" buys](#2-why-rust-and-what-zero-dependencies-buys)
3. [Data decides: garbage in, garbage out](#3-data-decides-garbage-in-garbage-out)

**Part II — Using Ferrum**

4. [How Ferrum is built](#4-how-ferrum-is-built)
5. [Preparing a corpus](#5-preparing-a-corpus)
6. [Training an SLM](#6-training-an-slm)
7. [Generating text](#7-generating-text)
8. [Evaluating a model](#8-evaluating-a-model)
9. [External models (GGUF): run, fine-tune, export](#9-external-models-gguf-run-fine-tune-export)
10. [The GUI (Ferrum SLM Studio)](#10-the-gui-ferrum-slm-studio)
11. [Deploying models](#11-deploying-models)

**Part III — Reference**

12. [Library API reference](#12-library-api-reference)
13. [Performance and benchmarks](#13-performance-and-benchmarks)
14. [Working with any language](#14-working-with-any-language)
15. [Use cases and applications](#15-use-cases-and-applications)
16. [FAQ](#16-faq)
17. [Limits: an honest critique](#17-limits-an-honest-critique)

[Sources](#sources)

---

# Part I — Foundations

## 1. Generative AI, SLMs, and Transformers from zero

> **Who this is for:** someone who has heard "AI", "language model", or
> "transformer" but could not confidently explain them. No maths required.

### 1.1 The single idea: guess the next piece

Shown *"The cat sat on the ____"*, you think "mat". You are not remembering
that sentence. You are predicting what is likely to come next. A **language
model** is a program trained on large amounts of text to do exactly this: given
some text, predict the next small piece. Append the prediction, predict again,
and repeat, and the machine *generates* new text one piece at a time. This loop
is **autoregressive generation**.

Because each piece depends on the one before, generation is *inherently
sequential*. That one fact explains why text generation is hard to speed up
with more CPU cores ([§13](#13-performance-and-benchmarks)).

**Generative AI** is the umbrella term for systems that *produce* content, as
opposed to systems that *classify* it.

### 1.2 Training

A fresh model's internal numbers (**parameters**, or **weights**) are random,
so its guesses are gibberish. **Training** repeatedly does the following:

1. Show the model real text and hide what comes next.
2. Let it guess.
3. Score how wrong it was. This score is the **loss**.
4. Nudge every weight slightly in the direction that would have reduced the
   loss (**backpropagation** and **gradient descent**, which Ferrum writes out
   by hand).

One pass over the training text is an **epoch**. Falling loss means the model
is getting less wrong.

> **Key intuition:** a model is not a database that looks up answers. It is a
> large adjustable function tuned so that plausible text comes out. That is
> also why it can be confidently wrong ([§17](#17-limits-an-honest-critique)).

### 1.3 Tokens and byte-level BPE

Computers see numbers, not letters. A **tokenizer** chops text into
**tokens** and gives each one an ID. A token can be a word, part of a word, a
character, or a byte.

- One token per character is simple but wasteful: the model takes many tiny
  steps to produce one word.
- One token per word is compact but breaks on words it has never seen.
- **Byte-Pair Encoding (BPE)** is the compromise used by GPT-2, GPT-4, and
  most modern models. It starts from a small alphabet and *learns* to merge the
  most frequent adjacent pairs into new tokens.

Ferrum uses **byte-level BPE**, which starts from the 256 possible byte values.
Every file in every language is made of those 256 values, so **any text can be
tokenized** (English, emoji, Chinese, Arabic, code) and there is no "unknown
character" error.

### 1.4 The Transformer

The 2017 paper *Attention Is All You Need* introduced the **Transformer**, the
architecture behind nearly every modern language model. Its key mechanism is
**self-attention**: while processing one token, the model looks at every other
token in its context and weights the relevant ones. In *"The animal didn't
cross the street because **it** was too tired"*, attention lets "it" draw on
"animal".

| Term | Plain-language meaning |
|---|---|
| Self-attention | Every token looks at every other token and decides what is relevant |
| Multi-head attention | Several attention "viewpoints" in parallel, each catching a different relationship |
| Embedding | A lookup table that turns token IDs into lists of numbers |
| Positional encoding | Tells the model the *order* of tokens, since attention sees them all at once |
| Feed-forward network (FFN) | A small neural network applied after attention |
| Layer normalization | Keeps numbers in a healthy range so training stays stable |
| Block / layer | One round of attention + FFN + normalization; stacking blocks learns deeper patterns |
| Context window | How many tokens the model can see at once |

Ferrum implements two Transformer flavours by hand. One is its own design,
which you train from your text. The other is the Llama/Qwen design it imports
from downloaded files ([§4](#4-how-ferrum-is-built)).

### 1.5 LLMs and SLMs

**Large language models** (ChatGPT, Claude, Gemini, Llama) have billions to
trillions of parameters, cost millions to train, and usually run in the cloud.
**Small language models** use the same architecture kept deliberately small:
generally under about 10 billion parameters, often far smaller. That lets them
run on laptops, phones, and microcontrollers. Recent industry research argues
SLMs are "sufficiently powerful, inherently more suitable, and necessarily more
economical" for the many narrow, repetitive jobs that make up most real AI
pipelines. The advantages: lower latency, less memory, lower cost, easier
deployment, and private offline operation.

> **Where Ferrum sits:** at the *very small* end. Ferrum trains models from
> about 0.1M to about 30M parameters on a laptop CPU, and runs or fine-tunes
> small open models that others trained. That makes it excellent for learning,
> experimentation, and narrow tasks, and unsuitable as a ChatGPT replacement.

### 1.6 Training and inference

**Training** teaches the model: slow, done once, compute-heavy. **Inference**
uses the trained model to generate or predict: fast and repeated. Ferrum does
both on the CPU in one tool. It can also run and fine-tune models someone else
trained.

### 1.7 The journey of one sentence

```text
"the quick brown fox"
   │ (1) TOKENIZER  → [412, 88, 1290, 77]
   │ (2) EMBEDDING  → a vector of numbers per token
   │ (3) TRANSFORMER BLOCKS → attention + FFN, several times
   │ (4) OUTPUT     → a probability for every possible next token
   │                  " jumps" 71%, " ran" 9%, " sat" 4%, …
   │ (5) SAMPLING   → pick one (temperature = how adventurous)
   └──► append " jumps" and repeat from (1)
```

**Temperature** controls how adventurous sampling is. Low values (0.1–0.3)
play it safe and repeat learned patterns. High values (0.8 and up) give more
variety and more risk.

---

## 2. Why Rust, and what "zero dependencies" buys

### 2.1 Rust in one minute

**Rust** compiles to native machine code like C and C++, but its compiler
enforces ownership rules that make whole categories of memory bugs impossible
to compile. You get C-level speed with guarantees C cannot offer, and no
garbage collector pausing the program.

### 2.2 The three guarantees

- **Memory safety without a garbage collector.** There are no surprise pauses
  while tokens are streaming.
- **`#![forbid(unsafe_code)]`.** The compiler rejects the entire engine if any
  code uses Rust's `unsafe` escape hatch, so every memory operation is
  compiler-checked. The cost is concrete: reading a downloaded model cannot use
  memory-mapping (which needs `unsafe`), so Ferrum streams files with ordinary
  safe reads instead.
- **Zero dependencies (`std`-only).** `ferrum_core`'s dependency list is empty.
  The project writes its own matrix maths, neural-network layers, tokenizer,
  file formats, multi-core parallelism, and GGUF parser.

| Benefit | Why it matters |
|---|---|
| Tiny, fast builds | Nothing to download; one small binary |
| Nothing to vet or update | No supply-chain surface; the whole engine is readable |
| Runs anywhere Rust runs | Servers, laptops, Raspberry Pi, browsers |
| Reproducibility | No moving external parts, so results do not drift |
| A teaching artifact | You can read the entire forward and backward pass |

### 2.3 Why Rust fits CPU inference

Rust has no global interpreter lock, so it can safely spread work across every
core, which Ferrum does for every matrix multiply. Its efficiency suits edge
and resource-constrained systems. The wider industry is moving the same way:
Burn, Candle, mistral.rs, and MicroFlow all serve or run models in Rust.

### 2.4 One codebase, many destinations

Because the engine is pure Rust, it compiles to **WebAssembly**. A model
trained on your laptop can run entirely in a visitor's browser, with no server,
no API key, and no data leaving their machine ([§11.3](#113-in-the-browser-webassembly)).

### 2.5 The trade-offs, honestly

- Rust is harder to learn than Python. *Using* Ferrum requires no Rust.
- Writing its own maths instead of using OpenBLAS or MKL means Ferrum is **not
  the fastest possible CPU engine**. It trades peak speed for transparency.
- The AI ecosystem still lives mostly in Python, so ready-made Rust resources
  are fewer.

---

## 3. Data decides: garbage in, garbage out

> If you read only one chapter before training anything, read this one.

### 3.1 The model is a mirror

A language model sees nothing but its training text. It cannot learn what is
not there, and it cannot un-learn what is. **Garbage In, Garbage Out (GIGO)**
was first recorded in 1957 and is usually credited to IBM's George Fuechsel.
Machine learning amplifies it. A buggy program fails predictably. A model
trained on bad data fails in ways baked invisibly into millions of numbers, and
it fails confidently.

### 3.2 Why data matters even more for small models

A giant LLM can average away noise across trillions of tokens. A small model
cannot. Every token must earn its place, and junk consumes scarce capacity.
Microsoft's *Textbooks Are All You Need* showed a 1.3B model trained on curated
"textbook-quality" data beating models about 10× larger trained on about 100×
more data. **Quality beats volume**, and with Ferrum's tiny models quality is
the only lever you have.

### 3.3 What good data is

| Quality | What goes wrong without it |
|---|---|
| **Relevant** to your task and domain | Train on Shakespeare, get Shakespeare |
| **Representative**, including the right *language* | An English corpus cannot speak French |
| **Clean**: no boilerplate, markup, OCR errors | The model learns `<div>` tags and page numbers |
| **Correct** | Errors are learned and repeated confidently |
| **Consistent** in format | Capacity is wasted on pointless variation |
| **Sufficient and diverse** | The model memorizes instead of generalizing |
| **De-duplicated** | Repeats are over-weighted and encourage memorization |
| **Balanced** | Biased data in, biased output out |
| **Valid UTF-8** | Mojibake in, mojibake out |
| **Legally and ethically yours** | Legal and privacy risk |

A useful test: *if a careful human read this corpus, would they learn the right
thing from it?*

### 3.4 The asymmetry

Good data cannot turn a tiny model into a genius. But bad data guarantees
failure at *any* size. Great data is necessary but not sufficient, and bad data
is sufficient on its own to ruin a model. Spend your effort accordingly.
[§5](#5-preparing-a-corpus) shows the tools Ferrum gives you.

---

# Part II — Using Ferrum

## 4. How Ferrum is built

### 4.1 The workspace

| Part | What it is |
|---|---|
| `ferrum_core` | The engine library. Everything else calls into it. |
| `slm_cli` | The `train_transformer` command-line tool: train, generate, eval, info, run-gguf, finetune-gguf, export-gguf |
| `slm_wasm` | Browser bindings for trained SLMs |
| `ferrum_gui` | **Ferrum SLM Studio**, a point-and-click desktop app ([§10](#10-the-gui-ferrum-slm-studio)) |
| `tests` | Integration and regression tests |

### 4.2 Two Transformer stacks

The single most common source of over-claiming is mixing these two up:

| | Ferrum's own (`train_transformer`) | Imported (`llm`) |
|---|---|---|
| Positions | learned absolute embedding | RoPE (rotary) |
| Norm | LayerNorm | RMSNorm |
| FFN | ReLU MLP | SwiGLU |
| Attention | dense multi-head | grouped-query (GQA) |
| Training | full backprop, Adam/AdamW, QAT, from scratch | gradient-checked backprop, AdamW fine-tuning (f32) |
| Source | trained from your text | imported from a GGUF file |
| Tokenizer | Ferrum byte-level BPE (stored in the model) | the checkpoint's own (BPE exact, SPM approximate) |
| Ships as | FINF `.bin` (native, WASM) | GGUF (llama.cpp, ollama, LM Studio) |

They share the low-level kernels, worker pool, and quantization grid, and
nothing above that.

### 4.3 Architecture at a glance

```text
Tensor ──► ops (matmul, qlinear, softmax, layernorm)  ──► parallel (persistent worker pool)
   │
   └──► Layer trait: Linear[+packed QWeight] · ActivationLayer · LayerNorm · Embedding
                     · Flatten · TransformerBlock (causal MHA + FFN) · KvCache
Sequential ──► ordered pipeline of layers

tokenizer          ByteBpeTokenizer (byte-level BPE; char-level fallback)
quant              int8/int4 fake-quant for QAT; packed in-memory QWeight
train_transformer  TransformerNet: backprop, data-parallel epochs, stride, step budgets
slm                GenerativeSLM: config-driven training pipeline, generate, evaluate
meta / loader      ModelMetadata + FINF v4 (f32) / v5 (int8, int4) model files
dataset            clean_corpus, corpus_stats, validate_for_training

── imported architecture ──
gguf / gguf_tokenizer   std-only GGUF v2/v3 reader + the checkpoint's own tokenizer
llm                     LlamaModel: RMSNorm, RoPE, GQA, SwiGLU, KV-cached decode
llm_train               LlamaTrainer: gradient-checked backprop, AdamW, .flck checkpoints
gguf_write              GgufBuilder + block encoders → GGUF v3 export
chat_format             chat-template detection and rendering for instruct models
```

---

## 5. Preparing a corpus

Any UTF-8 text file is a corpus. Three engine functions help you make it a
*good* one. They are available from the library and from the GUI's
**Datasets** tab ([§10.3](#103-the-tabs)):

1. **`clean_corpus`**: strip Project Gutenberg boilerplate, lowercase, collapse
   whitespace, normalize punctuation, strip control characters, and cap the
   size for quick experiments.
2. **`corpus_stats`**: characters, bytes, lines, words, and unique characters.
   A suspiciously low unique-character count points to a corrupt or one-note
   file.
3. **`validate_for_training`**: fails fast, with a reason, if the corpus cannot
   train (for example, it is shorter than the context window).

The GUI can also download corpora directly from a URL, from Hugging Face
(including gated files with a token), or from Kaggle. It ships a small catalog:
TinyStories, TinyStories V2 (GPT-4), WikiText-2, Shakespeare, and Poetry
Foundation.

**Hold text back.** Keep a separate `heldout.txt` in the same style that the
model never trains on. It is how you measure whether the model generalizes
([§8](#8-evaluating-a-model)). Alternatively, `--val F` splits off the last
fraction automatically ([§6.6](#66-validation-checkpoints-and-resuming)).

**Checklist before you train**

- [ ] The right content, in the domain and language you want
- [ ] Cleaned of boilerplate, markup, and control characters
- [ ] No large passages repeated many times
- [ ] Comfortably longer than the context window, with real variety
- [ ] Inspected: you looked at `corpus_stats` and *read a sample*
- [ ] Held-out text set aside

---

## 6. Training an SLM

### 6.1 How training works

Ferrum trains a causal Transformer by cutting the token stream into
**windows** of `context_len` tokens. In each window the model predicts the next
token at *every* position, so one window yields `context_len` training targets.
Minibatches of windows are drawn from a fresh random shuffle each epoch.

- **Window stride** is the distance between window starts. The default is
  `stride = context_len`: windows do not overlap, and **one epoch sees each
  token once**. With `--stride 1` every token starts a window (the legacy
  behaviour), so an epoch trains on each token about `context_len` times and
  costs that much more. With a stride above 1, the window grid gets a random
  offset each epoch, so boundaries differ between epochs.
- **Epochs or a token budget.** Train for `--epochs N` passes, or set
  `--tokens N` (for example `5e7`) to stop after exactly that many training
  tokens (`steps × batch × context`), even partway through an epoch. A budget
  is the right unit for real runs. "Compute-optimal" training is about 20
  tokens per parameter.
- **Quantization-aware training (QAT)** is on by default. Each step runs
  forward and backward against int8-snapped weights while Adam updates
  full-precision masters (a straight-through estimator), so the int8 file you
  ship behaves like the model you trained.

Training is int8 QAT, multi-threaded, and deterministic for a fixed seed and
configuration. The model is saved as an int8-quantized FINF v5 file. If the
model file already exists, `train` loads it instead of retraining; pass
`--force` to retrain.

### 6.2 Command-line reference

```text
train_transformer train <corpus.txt> <model.bin> [options]
train_transformer run   <corpus.txt> <model.bin> <seed text> [options]   # train if needed, then generate
```

**Architecture**

| Flag | Default | Meaning |
|---|---|---|
| `--context N` | 16 | Context window in tokens |
| `--embed N` | 32 | Embedding width (must be divisible by `--heads`) |
| `--heads N` | 4 | Attention heads per block |
| `--blocks N` | 2 | Transformer blocks |
| `--hidden N` | 64 | FFN hidden width (typically 4× `--embed`) |
| `--vocab N` | 512 | `0` = character-level; `≥ 256` = byte-level BPE of that size |

**Schedule and data**

| Flag | Default | Meaning |
|---|---|---|
| `--epochs N` | 100 | Training epochs (ignored when `--tokens` is set) |
| `--tokens N` | 0 (off) | Token budget, e.g. `5e7`; stops mid-epoch if need be |
| `--stride N` | 0 (= context) | Window stride; `1` = legacy full overlap |
| `--batch N` | 16 | Sequences per optimizer step |
| `--tokenizer PATH` | — | Reuse a saved BPE tokenizer; trained and saved there if missing |
| `--val F` | — | Hold out the last fraction `F` for validation; keep the best epoch |
| `--patience N` | 0 | With `--val`: stop after `N` epochs without improvement (0 = never) |

**Optimization**

| Flag | Default | Meaning |
|---|---|---|
| `--lr F` | 0.01 | Adam learning rate (peak rate when `--cosine` is on) |
| `--cosine` | off | Linear warmup, then cosine decay to 0 over the whole run |
| `--warmup N` | 0 | Warmup steps for `--cosine` |
| `--clip F` | 0 (off) | Global gradient-norm clipping threshold |
| `--weight_decay F` | 0 | AdamW decoupled weight decay |
| `--dropout F` | 0 | FFN-hidden dropout during training |
| `--tie` | off | Tie the LM head to the token embedding (saves `vocab × embed` parameters) |
| `--no-qat` | off | Plain fp32 training instead of int8 QAT |

**Run control**

| Flag | Default | Meaning |
|---|---|---|
| `--checkpoint PATH` | — | Write training state (weights + Adam moments + RNG) |
| `--checkpoint-every N` | 0 | Checkpoint every `N` optimizer steps (always at the end) |
| `--resume PATH` | — | Resume from that checkpoint and keep writing to it |
| `--threads N` | 0 (auto) | Data-parallel training workers; `1` = serial |
| `--seed N` | 1337 | RNG seed |
| `--force` | — | Retrain even if the model file exists |
| `--sample` | — | Print a short sample after training |
| `--verbose`, `-v` | — | Stream the engine's internal trace |

### 6.3 Choosing a tokenizer

- **`--vocab 0`** (character-level): one token per distinct character. This is
  the simplest and most transparent option, and works well for tiny corpora.
  Character-level generation needs a seed at least `context_len` characters
  long.
- **`--vocab ≥ 256`** (byte-level BPE): subword tokens pack more text into the
  same context window, handle any language, and usually win on real text. Start
  with 512–4096. Larger vocabularies need more data to learn. Values from 1 to
  255 are rejected because the 256-byte base is irreducible.
- **Train once, reuse.** `--tokenizer tok.bpe` writes the learned merge list on
  the first run and loads it on every later run, so tokenizer training is
  skipped and the vocabulary stays stable across experiments and resumes. The
  trainer counts each distinct word once and updates pair counts
  incrementally, so it is fast: 22 MB trains at vocab 4096 in about 0.3 s.
- Merge learning is deterministic. Ties break by pair order, so the same corpus
  and vocab size always give the same tokenizer.

### 6.4 Epochs, stride, and budgets in practice

| You want | Use |
|---|---|
| A quick experiment on a small file | `--epochs 20` (stride = context) |
| A real run sized to compute | `--tokens <20 × params>`, e.g. `--tokens 3e7` for 1.5M params |
| To reproduce a pre-2026-10 run exactly | `--stride 1` (legacy overlapping windows) |

With the default stride, `--epochs` means what it says: passes over the
corpus. Example CLI commands from older docs that used `--epochs 200` with
stride 1 trained on each token thousands of times. Do not copy them.

### 6.5 Optimization and regularization

- **Learning rate.** Small models tolerate `3e-3`–`1e-2`. Above about 5M
  parameters, start near `1e-3`–`3e-3` and use `--cosine --warmup 100–500`.
- **`--clip 1`** prevents rare exploding steps; it is cheap insurance on
  longer runs.
- **`--weight_decay 0.01–0.1` and `--dropout 0.1`** help when held-out
  perplexity lags training perplexity, which is the signature of
  memorization.
- **`--tie`** shares the embedding and output matrices. It saves parameters
  and usually helps small models.
- **`--no-qat`** trains in plain fp32. QAT costs a little speed. Keep it on if
  you will ship int8.

### 6.6 Validation, checkpoints, and resuming

**Validation.** `--val 0.05` holds out the last 5% of the corpus (the
tokenizer is fit on the training part only). After each epoch the model is
scored on it, and the weights with the **lowest validation loss are the ones
saved**, which are not necessarily the last epoch's. `--patience N` stops
early.

**Checkpoints.** `--checkpoint run.fckp --checkpoint-every 500` writes the
full training state every 500 steps and at the end: weights, Adam moments,
step count, and RNG. Writes are atomic (temporary file, then rename), so a
crash mid-write never corrupts the checkpoint.

**Resume.** Re-run the *same* command with `--resume run.fckp`. Training
continues from the saved step toward the same budget or epoch count, and the
LR schedule picks up where it stopped. The checkpoint fixes the architecture.
If its vocabulary or context does not match the run (for example, a different
tokenizer), training refuses rather than silently mixing them. Use
`--tokenizer` so the vocabulary is guaranteed identical. One detail: a resumed
run restarts its partially-finished epoch with a fresh shuffle instead of
replaying the exact remaining batches.

```bash
train_transformer train big.txt model.bin --context 128 --embed 128 --heads 4 --blocks 4 \
    --hidden 512 --vocab 2048 --lr 0.003 --tokens 5e7 --cosine --warmup 200 --clip 1 --tie \
    --tokenizer tok.bpe --checkpoint run.fckp --checkpoint-every 500
# …power cut at step 9,000…
train_transformer train big.txt model.bin <same flags> --resume run.fckp
```

### 6.7 Sizing a run for your machine

Measured on an i5-1135G7 (4 cores / 8 threads, 16 GB) with int8 QAT, batch 16,
and 8 threads:

| Params | Config (T = context, C = width, L = layers) | Sec/step | Tokens/s | Peak RAM | Tokens/day |
|---|---|---|---|---|---|
| 0.17M | T64, C64, L2 | 0.04 | 24,700 | 36 MB | 2.1B |
| 1.34M | T128, C128, L4 | 0.45 | 4,520 | 245 MB | 391M |
| 6.9M | T256, C256, L6 | 4.5 | 913 | 1.3 GB | 79M |
| 20.6M | T256, C384, L8 | 11.9 | 345 | 2.5 GB | 30M |
| 33.8M | T256, C512, L8 | 21.3 | 193 | 3.2 GB | 17M |

**RAM is dominated by activations**, not by the 16 bytes per parameter of
weights and optimizer state. RAM is rarely the limit; compute is.

| Goal | Reachable size | Example: time to 20 tokens/param |
|---|---|---|
| Heavily over-trained, minutes to hours | 0.1M–2M | 1.3M model: about 1.7 h |
| Compute-optimal in 24 h | ≈5M | 6.9M model: about 1.75 days |
| Compute-optimal in a week | ≈12–15M | 20.6M model: about 14 days |
| Compute-optimal in a month | ≈25–30M | 33.8M model: about 40 days |

**Starting points by size**

| Target | `--context` | `--embed` | `--heads` | `--blocks` | `--hidden` | `--vocab` | `--lr` |
|---|---|---|---|---|---|---|---|
| ~0.2M (minutes) | 64 | 64 | 4 | 2 | 256 | 512 | 1e-2 |
| ~1.5M (hours) | 128 | 128 | 4 | 4 | 512 | 2048 | 3e-3 |
| ~7M (a day or two) | 256 | 256 | 8 | 6 | 1024 | 4096 | 1e-3 |

Throughput is measured; the presets are reasonable defaults, not tuned
optima. A GPU is never used, so a small discrete GPU (for example, a 2 GB
MX330) adds nothing.

### 6.8 Multi-threading and determinism

There are two independent knobs:

- **`--threads N`** splits each minibatch across worker threads (data-parallel
  training). Gradients are reduced in a fixed order, so a given thread count is
  reproducible, and `--threads 1` is bit-identical to the serial trainer.
  Changing the shard count can change the low bits of the weights, because
  floating-point addition is not associative.
- **`FERRUM_NUM_THREADS=N`** sizes the matmul worker pool used by both
  training and generation. It never changes results.

### 6.9 Library equivalents

Every CLI flag maps onto a field of `TransformerConfig`:

```rust
pub struct TransformerConfig {
    pub context_len: usize,    // 16
    pub embed_dim: usize,      // 32
    pub num_heads: usize,      // 4
    pub num_blocks: usize,     // 2
    pub hidden_dim: usize,     // 64
    pub epochs: usize,         // 100
    pub lr: f32,               // 0.01
    pub batch_size: usize,     // 16
    pub vocab_size: usize,     // 512   (0 = char-level, >= 256 = BPE)
    pub weight_decay: f32,     // 0.0
    pub dropout: f32,          // 0.0
    pub window_stride: usize,  // 0     (= context_len)
    pub max_tokens: u64,       // 0     (train for `epochs`)
    pub qat: bool,             // true
    pub grad_clip: f32,        // 0.0   (off)
    pub cosine_lr: bool,       // false
    pub warmup_steps: u64,     // 0
    pub weight_tying: bool,    // false
    pub tokenizer_path: String,   // ""  (train in memory)
    pub checkpoint_path: String,  // ""  (no checkpoints)
    pub checkpoint_every: u64,    // 0   (only at the end)
    pub resume: bool,             // false
}
```

```rust
use ferrum_core::{GenerativeSLM, Rng, TransformerConfig, ValidationConfig};

let cfg = TransformerConfig { max_tokens: 2_000_000, cosine_lr: true, ..TransformerConfig::default() };
let mut rng = Rng::new(7);

// Plain (threads: 0 = auto, 1 = serial):
let slm = GenerativeSLM::train_transformer_config_threaded(&corpus, &cfg, 0, &mut rng, |ep, loss| {})?;

// With validation + early stopping (returns the best-by-validation model):
let val = ValidationConfig { val_fraction: 0.05, patience: 3 };
let slm = GenerativeSLM::train_transformer_config_validated(&corpus, &cfg, 0, &val, &mut rng, |p| {
    println!("epoch {} train {:.3} val ppl {:.2}{}", p.epoch, p.train_loss, p.val.perplexity,
             if p.is_best { " *" } else { "" });
})?;

// Train once, then load from disk on later runs:
let (slm, was_loaded) = GenerativeSLM::load_or_train("model.bin", &corpus, &cfg, &mut rng, |_, _| {})?;
```

The older positional functions `GenerativeSLM::train_transformer(corpus,
context, embed, heads, blocks, hidden, epochs, lr, batch, vocab, rng)` and the
`*_with_callback` / `*_threaded_with_callback` variants still work. They keep
**stride 1** so existing code reproduces exactly.

For full control, drive `TransformerNet` directly ([§12.3](#123-training-primitives)).

### 6.10 A worked walkthrough

This uses a short paragraph as the corpus, which is enough to see the pipeline
end to end. Real models want megabytes.

```bash
cargo build -p slm_cli --release
BIN=./target/release/train_transformer

$BIN train corpus.txt model.bin --epochs 60 --context 12 --embed 32 --heads 4 \
    --blocks 2 --hidden 64 --vocab 300 --seed 7 --stride 1
```

```text
  epoch     3/60   loss = 2.596811
  epoch    30/60   loss = 0.189039
  epoch    60/60   loss = 0.152412
Saved 40055 bytes → model.bin (int8-quantized FINF v5)
Reload check: OK (6 layers).
```

(`--stride 1` is used here only because the corpus is a few hundred
characters. On real data, leave the default.)

```bash
$BIN info model.bin
```

```text
Format    : FINF v5 (int8-quantized)
Task      : TransformerSLM
Input dim : 12
Output dim: 300
Tokenizer : byte-level BPE (300 tokens, 44 merges)
Layers    : 6
```

```bash
$BIN generate model.bin "the quick brown" --chars 60 --temp 0.2
# the quick brown to school. by evening the streets grow quiet again and the

$BIN eval model.bin heldout.txt
# Predictions  : 72
# Cross-entropy: 1.3112 nats/token
# Bits/token   : 1.8917
# Perplexity   : 3.7107

$BIN eval model.bin corpus.txt
# Perplexity   : 1.0036   ← near-perfect on text it has memorized
```

A model that learned nothing scores a perplexity equal to its vocabulary size
(300 here), so held-out 3.7 means it learned real structure. The gap between
1.0 on seen text and 3.7 on unseen text is the textbook sign of memorization on
a tiny corpus. More data, regularization, and a right-sized model close it.
Iterate in this order: **more and better data → right-size the model →
regularize → tune `--vocab` → tune `--lr` and the budget**. Re-run `eval` after
each change.

---

## 7. Generating text

```text
train_transformer generate <model.bin> <seed text> [--chars N] [--temp F] [--gen-seed N] [--stream]
```

| Flag | Default | Meaning |
|---|---|---|
| `--chars N` | 200 | Characters to generate (characters even for BPE models) |
| `--temp F` | 0.8 | Sampling temperature; 0.1–0.3 for faithful, 0.7+ for varied |
| `--gen-seed N` | time-based | Fix it for reproducible output |
| `--stream` | — | Print the completion live, fragment by fragment |

- The output is the **seed followed by** exactly `--chars` new characters,
  unless generation is cut short.
- BPE models encode the seed, left-pad short prompts, and sample subword
  tokens. Character-level models need a seed of at least `context_len`
  characters.
- Streaming is UTF-8-safe: a partial multi-byte character is held back until
  it completes, so a `U+FFFD` placeholder is never shown.
- Generation uses a per-block **KV cache**, so each new token costs
  O(context) instead of recomputing the whole window.

The library adds top-k, top-p (nucleus), repetition penalty, and stop strings:

```rust
use ferrum_core::{Rng, SamplingParams};

let p = SamplingParams { temperature: 0.7, top_k: 40, top_p: 0.9, repetition_penalty: 1.1 };
let text  = slm.generate_with("Once upon a time", 200, &p, &mut Rng::new(1))?;
let reply = slm.generate_continuation("Q: hi\nA:", 80, 0.5, &mut Rng::new(1))?; // without the seed
let line  = slm.generate_until("Q: hi\nA:", 200, "\n", &p, &mut Rng::new(1))?;  // stop at newline
let full  = slm.generate_stream("the", 100, 0.7, &mut Rng::new(1), |frag| print!("{frag}"))?;
```

`format!("{seed}{continuation}") == generate(seed, …)`, and the concatenated
stream equals the continuation.

---

## 8. Evaluating a model

### 8.1 Held-out perplexity: the honesty check

Training loss shows the model is *fitting*. Perplexity on unseen text shows
whether it *generalizes*.

```bash
train_transformer eval model.bin heldout.txt
```

```rust
let e = slm.evaluate(&std::fs::read_to_string("heldout.txt")?)?;
println!("ppl {:.3}  bits/token {:.3}  CE {:.3} nats  ({} predictions)",
         e.perplexity, e.bits_per_token, e.cross_entropy, e.num_predictions);
```

- Lower is better, and **1.0 is perfect**.
- A model that learned nothing scores the **vocabulary size**.
- A large train-versus-held-out gap means **memorization**. That points to a
  data problem (too small or too repetitive), not a model that is too small.
- Evaluation uses the shipped int8-aware weights and the same forward path as
  generation. The text must be longer than the context window.
- To compare models with **different tokenizers**, compare bits per *byte*
  (bits/token × tokens ÷ bytes). Per-token numbers are not comparable across
  vocabularies.

### 8.2 Other checks

| Axis | How |
|---|---|
| **Seed fidelity** | `generate(seed, …)` always starts with `seed`. If not, the file is corrupt. |
| **Tokenizer compression** | `text.len() / tok.encode(text).len()` bytes per token; higher packs more text per step |
| **Size** | `to_bytes()` (f32) vs `to_bytes_quantized()` (int8, ≈4×) vs `to_bytes_quantized_int4()` (≈8×). Tensors under 64 values stay f32, so ratios are slightly below 4×/8×. |
| **Quantization fidelity** | Compare `slm.model.forward(x)` before and after a save/load round trip; int8 drift stays below about half a quantization step thanks to QAT. int4's grid is coarser. |
| **Speed** | `time train_transformer generate …`; compare `FERRUM_NUM_THREADS=1` vs auto. Outputs are identical, only wall-clock changes. |
| **Reproducibility** | Same seed, same output; reloaded model matches in-memory model; same corpus and vocab give the same BPE merges. |

---

## 9. External models (GGUF): run, fine-tune, export

Ferrum imports small open-weight **Llama/Qwen** checkpoints in GGUF, the
standard open-model format. It runs them, fine-tunes them, and writes them back
out.

**Formats read:** `F32, F16, Q8_0, Q8_1, Q4_0, Q4_1` and the k-quants
`Q2_K, Q3_K, Q4_K, Q5_K, Q6_K`. The `IQ*` families are rejected with a clear
message. **Architectures:** `llama`, `qwen2`.

### 9.1 Running (`run-gguf`)

```text
train_transformer run-gguf <model.gguf> [prompt] [options]
```

| Flag | Default | Meaning |
|---|---|---|
| `--quant int4\|int8\|f32` | int4 | In-memory precision: int4 = least RAM, int8 = fastest decode, f32 = no second quantization |
| `--max N` | — | Maximum new tokens |
| `--temp F` | 0.8 | Temperature |
| `--top-k N` / `--top-p F` / `--rep F` | 0 / 1.0 / 1.0 | Sampling knobs (defaults = off) |
| `--gen-seed N` | — | Generation seed |
| `--raw` | — | Disable automatic chat-template formatting |
| `--ids "1 2 3"` | — | Raw prompt token IDs (only if the file has no tokenizer) |
| `--resume ckpt.flck` | — | Overlay fine-tuned weights (forces f32) |
| `--force` | — | Load even if the memory estimate exceeds available RAM |

- **Chat templates.** For instruct models, Ferrum detects the template from
  `tokenizer.chat_template` (ChatML, Llama-3, Llama-2 `[INST]`, Zephyr) or
  falls back by architecture. It wraps your prompt and stops at the
  template's end-of-turn token. It handles single turns only. Use `--raw` for
  base models.
- **Memory guard.** Before loading, Ferrum estimates the resident footprint
  and refuses with a warning if it will not fit (override with `--force`).
- **Expectations.** Decode is memory-bandwidth-bound: each token streams every
  weight once. A ~1B model decodes at about **7 tokens/s at int4** after about
  **30 s of prefill** for a 512-token prompt. That is a patient demo, not a
  chatbot ([§13.4](#134-quantized-decode-and-the-gguf-runner)).
- **Fidelity.** Import dequantizes and re-quantizes onto Ferrum's per-row
  grid. That is lossy and **not bit-exact to llama.cpp**. `--quant f32` avoids
  the second quantization.

### 9.2 Fine-tuning (`finetune-gguf`)

```text
train_transformer finetune-gguf <model.gguf> <corpus.txt> <out.flck> [options]
```

| Flag | Default | Meaning |
|---|---|---|
| `--epochs N` | 3 | Passes over the corpus |
| `--lr F` | 1e-4 | AdamW learning rate |
| `--batch N` | 8 | Sequences per step |
| `--seq N` | 64 | Window length (capped at the model's context) |
| `--warmup N` | 0 | Warmup steps (cosine schedule) |
| `--clip F` | 1.0 | Gradient clipping |
| `--weight_decay F`, `--dropout F` | 0 | Regularization |
| `--qat` | off | int8 quantization-aware fine-tuning |
| `--threads N`, `--seed N` | auto, — | Parallelism and determinism |
| `--resume ckpt.flck` | — | Continue a previous fine-tune |
| `--sample` | — | Print a sample afterwards |

The model loads at f32, because training needs full-precision masters.
Full-parameter AdamW needs about **16 bytes per parameter**, so SmolLM2-135M
(about 2.2 GB) is comfortable on 16 GB, Qwen2.5-0.5B (about 8 GB plus
activations) is tight, and 1.5B and above do not fit. The output `.flck`
checkpoint is applied with `run-gguf --resume` or exported with `export-gguf
--resume`.

### 9.3 Exporting (`export-gguf`)

```bash
train_transformer export-gguf in.gguf out.gguf --quant q8_0                       # re-quantize
train_transformer export-gguf base.gguf tuned.gguf --resume tuned.flck --quant q6_k  # ship a fine-tune
```

- Output types are `f32 | f16 | q8_0 | q8_1 | q4_0 | q4_1 | q4_k | q5_k | q6_k`
  (default `q8_0`). Q2_K and Q3_K are read-only.
- The source's hyperparameters and tokenizer carry over verbatim, so the file
  runs unchanged in llama.cpp, ollama, and LM Studio.
- Norms and biases stay f32. A matrix whose rows are not block-aligned for
  the chosen quant (32 for legacy formats, 256 for k-quants) falls back to
  f16. The per-type summary printed after export shows what was emitted.
- Writes are atomic. Re-quantizing an already-quantized source is lossy. The
  lossless paths are `f16`/`f32`, or exporting fine-tuned f32 masters.

### 9.4 Library API

```rust
use ferrum_core::{Gguf, GgufQuant, GgufTokenizer, LlamaTrainer, QKind, Rng};

let gguf  = Gguf::open("model.gguf")?;           // streamed: tensors read on demand
let tok   = GgufTokenizer::from_gguf(&gguf)?;    // the checkpoint's own tokenizer
let model = gguf.load_llama(QKind::Int4)?;       // or Int8; load_llama_prec(None) = f32
let out   = model.generate(&tok.encode("Hello"), 32, 0.7, &mut Rng::new(1))?;
println!("{}", tok.decode(&out));

let f32m = Gguf::open("model.gguf")?.load_llama_prec(None)?;
let mut trainer = LlamaTrainer::new(f32m)?;      // AdamW, lr 1e-4; errors if any weight is quantized
trainer.set_lr(3e-4);
trainer.set_grad_clip(Some(1.0));
let loss = trainer.train_step(&tok.encode("training text"))?;   // one AdamW step, returns the loss

trainer.model.write_gguf(&gguf, GgufQuant::Q4K, "out.gguf")?;  // export the fine-tuned model
```

---

## 10. The GUI (Ferrum SLM Studio)

### 10.1 What it is

Ferrum SLM Studio is a cross-platform **Tauri 2** desktop (and mobile) app.
Its interface is plain HTML, CSS, and vanilla JavaScript with no framework and
no Node build step, and every button calls straight into `ferrum_core`. It puts
the whole pipeline in one window: data preparation, training, streaming
generation, evaluation, model inspection, GGUF run, fine-tune, and export, plus
a system monitor and a live engine log.

### 10.2 Building and launching

**Prerequisites**

1. Rust and Cargo, plus the Tauri CLI: `cargo install tauri-cli --version "^2"`.
2. System WebView libraries:
   - **Linux (Debian/Ubuntu):** `libwebkit2gtk-4.1-dev libgtk-3-dev libayatana-appindicator3-dev librsvg2-dev build-essential curl wget file libssl-dev`
   - **macOS:** Xcode command-line tools.
   - **Windows:** WebView2 runtime (preinstalled on Windows 11) and MSVC build tools.

```bash
cd ferrum_gui
cargo tauri dev      # development window
cargo tauri build    # installers / binaries for this OS
```

For real release artifacts, replace the placeholder icons with `cargo tauri
icon path/to/logo.png`.

**Snap-polluted environments.** Launching from a terminal inside a snap app
(for example, the VS Code snap) can crash with `undefined symbol:
__libc_pthread_init`, because the snap's GTK paths leak in. Strip them:

```bash
env -u GTK_PATH -u GTK_EXE_PREFIX -u LOCPATH -u GIO_MODULE_DIR \
    -u GTK_IM_MODULE_FILE -u GSETTINGS_SCHEMA_DIR -u XDG_DATA_HOME \
    XDG_DATA_DIRS=/usr/local/share:/usr/share ./target/release/ferrum_gui
```

**Mobile:** `cargo tauri android init && cargo tauri android dev`, or `cargo
tauri ios init && cargo tauri ios dev` (macOS only).

### 10.3 The tabs

| Tab | What you do there |
|---|---|
| **Datasets** | Download from a URL, Hugging Face, or Kaggle (or pick a catalog corpus), clean with `clean_corpus`, preview statistics, save |
| **Train** | Train a transformer SLM: architecture, epochs or a **token budget**, stride, LR/cosine/warmup, clip, weight decay, dropout, tie, QAT, tokenizer file, checkpoint/resume, validation fraction and patience; live loss chart |
| **Generate** | Prompt, characters, temperature, optional streaming |
| **Evaluate** | Score a model on held-out text; each run adds a table row (perplexity, CE, bits/token, uniform baseline) |
| **Models** | Inspect or reload a `.bin`: format, task, dimensions, tokenizer, layers |
| **GGUF** | Inspect and run a Llama/Qwen checkpoint: precision, top-k/top-p/repetition penalty, chat format, optional fine-tune overlay |
| **Fine-tune** | AdamW fine-tune an imported GGUF on a corpus into a `.flck` checkpoint |
| **Export** | Re-quantize or export a (fine-tuned) GGUF with phase-by-phase progress and a per-type summary |
| **System** | Per-core CPU, memory, and Ferrum's thread count (also as top-bar gauges) |
| **Capable** | A 1–2 s micro-benchmark that estimates the parameter ranges this machine can load, train, fine-tune, and run. Treat the numbers as rough upper bounds ([Status §5](Status.md#5-known-issues)). |
| **Engine log** (docked) | The engine's `--verbose` trace, streamed live when *Verbose* is ticked |

Every form validates input and shows plain-language errors. If the Rust
backend is unavailable (for example, in a plain web preview), a banner says so.

### 10.4 A first run, click by click

1. **Datasets:** pick *TinyStories (validation)* or load a local file, tick the
   cleaning options, click **Clean & preview**, then **Save corpus**.
2. **Train:** browse to the corpus and an output path. For a first model use
   context 64, embed 64, heads 4, blocks 2, hidden 256, vocab 1024, LR 0.003,
   and a token budget of `2e6`. Tick **Cosine LR** and set a checkpoint file.
   Click **Train** and watch the loss chart.
3. **Generate:** browse to the model, type a prompt, set temperature 0.7, tick
   **stream**.
4. **Evaluate:** add held-out text and compare the perplexity row against the
   baseline column.
5. **Models:** inspect the file you just made.
6. **GGUF → Fine-tune → Export:** the separate "someone else's model" path
   ([§9](#9-external-models-gguf-run-fine-tune-export)).

### 10.5 Platform support

| Feature | Desktop | Android / iOS | Web |
|---|:-:|:-:|:-:|
| Datasets, Train, Generate, Evaluate, Models | ✅ | ✅ | ❌¹ |
| GGUF run, fine-tune, export | ✅ | ✅² | ❌¹ |
| Engine log | ✅ | ✅ | ❌¹ |
| System monitor | ✅ | partial³ | ❌¹ |

¹ Tauri's Rust backend does not run in a plain browser. To run a *model* in a
browser, use `slm_wasm` ([§11.3](#113-in-the-browser-webassembly)).
² Limited by device RAM. ³ Limited by `sysinfo`'s mobile backends.

### 10.6 Architecture

```text
ui/               index.html, styles.css, app.js (vanilla), stream.js (+ stream.test.js)
src/lib.rs        Tauri builder, shared state, verbose → event sink, command registry
src/commands.rs   #[tauri::command]s calling ferrum_core directly
src/capable.rs    machine micro-benchmark + capability bounds
src/datasets.rs   corpus catalog + HTTP / Hugging Face / Kaggle downloads
tauri.conf.json   window + bundle config (frontendDist = "ui")
capabilities/     Tauri v2 permissions (core + dialog)
```

Heavy commands run on blocking tasks and stream events back to the UI
(`engine-log`, `train-progress`, `train-done`, `gen-fragment`,
`finetune-progress`, `export-progress`), so the window stays responsive. Under
a token budget, `train-progress` sends `total: null` because the epoch count is
not known in advance.

---

## 11. Deploying models

A trained model is one self-contained `.bin` (FINF). It carries weights,
metadata, and, for BPE models, the tokenizer. There is nothing else to ship.

> **GGUF is an import/export path, not Ferrum's deployment format.** Deploy
> your own models as FINF. Export Llama/Qwen fine-tunes as GGUF for the wider
> ecosystem.

### 11.1 Native (server, desktop, edge box)

```bash
cargo build --release -p slm_cli
scp target/release/train_transformer model.bin user@host:/opt/ferrum/
ssh user@host /opt/ferrum/train_transformer generate /opt/ferrum/model.bin "prompt" --chars 200
```

To embed inference in your own service, depend on `ferrum_core` and call
`GenerativeSLM::load` and `generate`, with no model server. Set
`FERRUM_NUM_THREADS` to bound CPU use per process.

### 11.2 Embedded and constrained targets

- int8 models are typically tens of kilobytes for sub-megabyte parameter
  counts. Use `to_bytes_quantized_int4()` to halve that again.
- Build with the workspace `release` profile (LTO, one codegen unit).
- On single-core or timing-sensitive targets, set `FERRUM_NUM_THREADS=1` for
  fully serial, predictable execution.

### 11.3 In the browser (WebAssembly)

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.122
bash scripts/build_wasm.sh        # → slm_wasm/pkg/{slm_wasm.js, slm_wasm_bg.wasm}
```

Serve `pkg/` and your `model.bin` from any static host. There is no backend,
and the visitor's input never leaves their machine.

```js
import init, { TransformerSLMModel } from "./pkg/slm_wasm.js";
await init();
const bytes = new Uint8Array(await (await fetch("model.bin")).arrayBuffer());
const slm = new TransformerSLMModel(bytes);
const meta = JSON.parse(slm.metadata());          // class_names, input_dim, tokenizer_state, …
let probs = slm.prime(new Float32Array(contextIds)); // KV-cached: prime once…
const next = slm.sample_from_probs(probs, 0.7, Math.random());
probs = slm.predict_next_cached(next);            // …then one token at a time
const attn = slm.get_last_attention_weights();    // [heads × T × T] for visualization
```

Other methods: `predict_next` (full forward), `num_heads`, `vocab_size`,
`context_len`, `num_layers`, `cached_len`, `entropy`, and `top_k_indices`.
WASM runs single-threaded but is otherwise the same engine.

### 11.4 Air-gapped deployment

Train on a connected machine, then carry the single `model.bin` and the static
binary in on physical media. Nothing is fetched at runtime, so there is nothing
to update in the field.

### 11.5 Versioning and compatibility

- FINF **v4** stores f32 weights. **v5** adds int8 and int4, chosen per weight
  vector. The loader reads both and rejects unknown encodings rather than
  misreading them.
- Models from before the tokenizer field load as character-level.
- Legacy **one-hot MLP** SLM files still load and inspect, but generation and
  evaluation reject them. Their trainer was retired ([Status §2](Status.md#2-the-2026-10-03-restructuring)).
- The file fully determines architecture, vocabulary, and tokenizer, so there
  is no version skew between a model and its config.

---

# Part III — Reference

## 12. Library API reference

### 12.1 Tensors, ops, and parallelism

`Tensor` is a flat `Vec<f32>` plus a shape (`Tensor::row`, `Tensor::matrix`,
`matrix_dims()`). The `ops` module provides matmul, a fused cache-tiled
`linear_forward`, packed `qlinear` (int8/int4), `softmax_rows`, and
`argmax_rows`.

The matmul kernels split output rows across a **persistent worker pool**
(`parallel`). It is built only on `std` threads, channels, and `Arc`, with no
`unsafe`. Workers are spawned once and reused, so generation pays no per-call
thread-creation cost. Single-token decode (`m = 1`) cannot be split by rows,
so the quantized path splits the GEMV **by output column** (`run_1d`). The
worker count comes from `std::thread::available_parallelism()`; override it
with `FERRUM_NUM_THREADS` and query it with `ferrum_core::num_threads()`. Small
workloads and `wasm32` run serially. Splitting never changes per-element
arithmetic, so results are bit-identical at any thread count.

### 12.2 Layers

All layers implement `Layer` (`forward`, `name`, `as_any`) and compose into
`Sequential`.

| Layer | Forward |
|---|---|
| `Linear` | `y = xW + b`; may hold a packed `Arc<QWeight>` consumed by `qlinear` without f32 expansion |
| `ActivationLayer` | element-wise ReLU, Softmax, … |
| `LayerNorm` | per-row normalization with learned γ/β |
| `Embedding` | token + positional lookup |
| `Flatten` | `[T, D]` → `[1, T·D]` |
| `TransformerBlock` | causal multi-head self-attention + FFN |

`KvCache` stores past keys and values per block for O(context)-per-token
generation.

### 12.3 Training primitives

- **`TransformerNet`**: the trainable causal transformer.
  - Setters: `set_qat`, `set_weight_decay`, `set_dropout`, `set_grad_clip`,
    `set_lr_schedule`, `set_weight_tying`, `set_window_stride`.
  - `save_checkpoint(&rng)` and `load_checkpoint(bytes)` serialize the full
    training state.
  - `to_inference()` returns a `Sequential`.
- **Epoch drivers**:
  - `train_transformer_epoch(net, tokens, batch, &adam, rng)` runs serially.
  - `train_transformer_epoch_threaded(…, threads)` shards each minibatch
    across threads.
  - `train_transformer_steps(…, threads, max_steps, &mut on_step)` runs one
    epoch capped at `max_steps` optimizer steps and calls
    `on_step(&net, &rng)` after each step (the hook used for budgets and
    checkpoints).
- **Optimizers**: `Sgd` (optional momentum), `Adam`, AdamW decay
  (`Adam::with_weight_decay`), `clip_grad_norm`, and `LrSchedule::{warmup_cosine,
  warmup_linear}` with `LrDecay`.
- **Losses**: `softmax_cross_entropy`, `mse`.

### 12.4 `GenerativeSLM`

`GenerativeSLM` wraps a `Sequential`, a `Normalizer`, and `ModelMetadata`.

| Group | Methods |
|---|---|
| Train | `train_transformer_config`, `train_transformer_config_threaded`, `train_transformer_config_validated`, `load_or_train`; legacy positional `train_transformer[_with_callback / _threaded_with_callback]` |
| Generate | `generate`, `generate_with`, `generate_continuation[_with]`, `generate_stream[_with]`, `generate_until`, `generate_stream_until` |
| Evaluate | `evaluate(text) -> Evaluation { num_predictions, cross_entropy, bits_per_token, perplexity }` |
| Persist | `to_bytes` (f32), `to_bytes_quantized` (int8), `to_bytes_quantized_int4`, `save` (int8), `save_int4`, `load`, `from_bytes` |

`SamplingParams { temperature, top_k, top_p, repetition_penalty }`;
`SamplingParams::with_temperature(t)` reproduces temperature-only sampling.

### 12.5 The BPE tokenizer

```rust
use ferrum_core::ByteBpeTokenizer;

let mut tok = ByteBpeTokenizer::train(corpus, 4096)?;   // learns up to vocab-256 merges
let ids  = tok.encode("lowest news");
assert_eq!(tok.decode(&ids), "lowest news");
let state = tok.encode_state();                          // "a,b;c,d;…" (+ "\n" + specials)
let tok2  = ByteBpeTokenizer::from_state(&state)?;
let [bos, eos, pad, unk] = tok.add_standard_special_tokens();
let framed = tok.encode_framed("text", true, true);      // <bos> … <eos>
```

Text is pre-split at whitespace boundaries, so merges never span a space.
Encoding is rank-based, applying the earliest-learned merge first. Special
tokens get ids above the byte and merge range and decode to nothing.

### 12.6 Quantization

Symmetric int8: `value ≈ i8 × scale`, `scale = max|value| / 127`, per tensor
or **per channel** (one scale per output row).

- `fake_quantize_int8` and `fake_quantize_int8_per_channel` snap values onto
  the grid in place (used by QAT).
- Tensors shorter than `QUANT_MIN_LEN = 64` stay f32, and non-finite tensors
  are left untouched.
- `QWeight` holds int8 or int4 weights packed in memory. int4 uses a
  **split-half** layout: byte `b`'s low nibble is column `b` and its high
  nibble is column `half + b`. That makes each nibble lane unit-stride, so the
  decode loop autovectorizes like int8. An interleaved layout ran several
  times slower.

### 12.7 The FINF model format

```text
4 bytes  "FINF"
u32      version (4 = f32, 5 = int8/int4-capable)
u32      norm_len;  [bytes] normalizer (empty for SLMs)
u32      meta_len;  [bytes] ModelMetadata JSON
u32      num_layers
per layer: u8 tag (0 Linear, 1 Activation, 2 Embedding, 3 LayerNorm, 4 TransformerBlock, 5 Flatten) + payload
```

In v5, each weight *vector* carries a one-byte encoding marker, so one file can
mix precisions:

| Marker | Encoding |
|:-:|---|
| 0 | raw f32 |
| 1 | int8, per tensor (f32 scale + i8 values) |
| 2 | int8, per channel (one f32 scale per row) |
| 3 | int4, per tensor |
| 4 | int4, per channel (the `to_bytes_quantized_int4` default) |

The loader bounds-checks every dimension before allocating. Quantized matrices
load **packed** and stay quantized in memory. `ModelMetadata` (JSON) holds
`task`, `feature_names`, `feature_ranges`, `class_names`, `input_dim`,
`output_dim`, and `tokenizer_state` (the BPE merge list; empty for
character-level models).

**Training checkpoints** are a separate format:
- `.fckp` (magic `FCKP`) for native transformers: architecture, QAT/tying
  flags, step, RNG, then weights and Adam moments.
- `.flck` for Llama fine-tunes: shape header, then weights, m, and v.

### 12.8 GGUF reader, tokenizer, runner, trainer, writer

| Module | Key API | Notes |
|---|---|---|
| `gguf` | `Gguf::from_path` (in memory), `Gguf::open` (streamed via `Mutex<File>`, no mmap), `load_llama(QKind)`, `load_llama_prec(None)` | GGUF v2/v3; checked offsets, EOF guards, rejects nested arrays and absurd counts |
| `gguf_tokenizer` | `GgufTokenizer::from_gguf`, `encode`, `decode` | BPE exact; SPM decode exact, encode greedy longest-match |
| `llm` | `LlamaModel::{forward, generate}`, `LlamaConfig`, `RopeType` | RMSNorm, RoPE (Norm and Neox), GQA + KV cache, SwiGLU; cached decode verified row-for-row against the full forward |
| `llm_train` | `LlamaTrainer::{new, set_lr, set_lr_schedule, set_grad_clip, train_step, train_batch, finetune_epoch_threaded, save_checkpoint, load_checkpoint_into}` | finite-difference-checked backward pass; AdamW; `.flck` checkpoints; rejects quantized weights |
| `gguf_write` | `GgufBuilder`, `write_llama_gguf`, `LlamaModel::write_gguf`, `GgufQuant` | byte-exact GGUF v3; encoders are exact inverses of the decoders (verified round trip) |
| `chat_format` | `ChatFormat::{ChatMl, Llama3, Llama2Inst, Zephyr}`, `detect`, `render`, `stop_token` | single-turn instruct formatting |

---

## 13. Performance and benchmarks

All numbers were measured on this project on an i5-1135G7 (4c/8t, 16 GB),
with release builds (LTO, 1 codegen unit), CPU only. Treat them as indicative
and run on an idle machine. Background load roughly halved one figure in an
earlier run.

### 13.1 Determinism (verified)

- Generating 2,000 characters at `FERRUM_NUM_THREADS=1,2,4,8` produced
  byte-identical files.
- The same training config produced byte-identical models twice, including
  when only the matmul pool size changed. Changing the data-parallel shard
  count can change low bits ([§6.8](#68-multi-threading-and-determinism)).

### 13.2 End-to-end scaling

Measured on a small model (context 16, embedding 64, 2 blocks, character-level):

| | 1 thread | 8 threads | Speedup |
|---|---|---|---|
| Training, 10 epochs | 44.65 s | 36.50 s | 1.22× |
| Generating 2,000 chars | 0.73 s | 0.67 s | 1.09× |

Small models scale poorly. Much of a training step is serial (softmax, small
per-head matmuls, the optimizer), and generation is a chain of small dependent
matmuls. Larger models, where big matmuls dominate, approach the kernel
ceiling below. For faster generation, prefer a BPE vocabulary (fewer steps per
character) and a smaller network over more cores.

### 13.3 Kernel throughput (synthetic ~1B-class shapes)

| Square GEMM | 8 threads | 1 thread | Speedup |
|---|---|---|---|
| 256² | 68.0 GFLOP/s | 17.2 | 4.0× |
| 512² | 52.9 | 16.1 | 3.3× |
| 1024² | 42.3 | 13.5 | 3.1× |
| 2048² | 35.7 | 14.6 | 2.4× |

The cache-tiled kernel holds about 36 GFLOP/s at 2048². The kernels use no
explicit SIMD, so the vector units are mostly idle. That is the main remaining
headroom: `target-cpu=native` gains nothing, and the fix has to be
register-blocked micro-kernels.

**f32 decode GEMV** (`m = 1`) runs serially at any thread count, at 5.5–8.2 GB/s
of weights streamed. A synthesized f32 1B decode step reaches **1.4–1.6
tokens/s**.

### 13.4 Quantized decode and the GGUF runner

| Synthesized 1B decode step | f32 (8 threads) | int4 (8 threads) | int4 (1 thread) |
|---|---|---|---|
| ms/token | 632.8 | **144.1** | 256.1 |
| tokens/s | 1.58 | **6.94** | 3.90 |
| Bytes streamed per token | 3.48 GB | 0.44 GB | 0.44 GB |

- **int4** is the right default for the GGUF runner: about 4–5× f32, at half
  int8's RAM.
- **int8** is slightly faster per call (no nibble unpack). Choose it when RAM
  is ample.
- The column split spreads the `m = 1` GEMV across cores for both.
- Net result for a 1B model: about 7 tokens/s decode, and about 30 s (512
  tokens) to 2 min (2,048 tokens) of compute-bound prefill.

### 13.5 Feasibility at 1B and beyond

Two rules of thumb set the limits. A forward pass costs about 2·N FLOPs per
token, and decode streams every weight once per token. A training step costs
about 6·N·T FLOPs and needs about 16 bytes per parameter resident.

| | 1B model on this machine |
|---|---|
| **Load** | Yes for supported quants: about 0.5 GB int4 plus the f32 token embedding; 2–3.5 GB peak during import. 7B F16 sources do not fit the no-mmap path. |
| **Run** | Yes, slowly (above) |
| **Fine-tune / train** | **No.** Optimizer state alone is about 16 GB, and 100M tokens through 1B parameters would take about 200 days at about 36 GFLOP/s. |

### 13.6 Reproducing

```bash
cargo build --release -p slm_cli
cargo bench --bench gemm                          # auto threads
FERRUM_NUM_THREADS=1 cargo bench --bench gemm     # serial
cargo bench --bench gemm -- 512 1024 4096         # custom GEMM sizes
for t in 1 2 4 8; do FERRUM_NUM_THREADS=$t target/release/train_transformer \
  generate model.bin "a long enough seed text here" --chars 2000 --gen-seed 42 > out_$t.txt; done
md5sum out_*.txt                                  # identical
```

---

## 14. Working with any language

The byte-level tokenizer starts from all 256 byte values, and every UTF-8 text
is made of them. So **any language, script, or emoji round-trips exactly**,
with no `<UNK>` token. This is the same design GPT-2 introduced and GPT-4 still
uses.

- Train on any language. Use BPE (`--vocab ≥ 256`) rather than
  character-level for scripts with large alphabets.
- Mixed scripts, code, and emoji-heavy chat logs all encode cleanly.
  Streaming never shows a half-character.
- Imported GGUF models bring *their own* tokenizer. Their language coverage is
  whatever they were trained on.

**Caveats.** The model learns only what is in the corpus: English text does
not teach French. In some scripts one visible character is several bytes, so a
short corpus may have fewer tokens than expected and hit the "corpus must be
longer than the context window" error. The fix is more text, a smaller
context, or a smaller vocabulary. Languages without spaces benefit from a
larger vocabulary. Right-to-left scripts are stored correctly; how they
display is up to your terminal or app.

---

## 15. Use cases and applications

Ferrum fits wherever a **small, self-contained, predictable model must run
without a GPU, a Python runtime, or a network connection**.

| # | Use case | Why Ferrum fits |
|---|---|---|
| 1 | Offline text generation on edge devices | One `.bin`, CPU only, 4× smaller via int8, nothing to install |
| 2 | Running and fine-tuning small open models offline | GGUF import with its own tokenizer, AdamW fine-tune, export back to GGUF; private but slow (a few tokens/s at 1B) |
| 3 | Domain autocomplete (shell history, logs, code, templates) | BPE captures flags, paths, and identifiers; deterministic suggestions; millisecond latency |
| 4 | Privacy-preserving on-device modeling | Nothing leaves the machine; no telemetry; auditable code |
| 5 | In-browser demos (WASM) | `slm_wasm` runs models client-side with no backend or API key |
| 6 | Reproducible research and teaching | Readable forward and backward pass; bit-for-bit reproducible |
| 7 | Embedded and resource-constrained systems | Kilobyte-scale int8/int4 models; serial, predictable timing |
| 8 | Air-gapped and field deployments | One file plus a static binary; nothing to fetch or update |
| 9 | Cheap, horizontally scaled CPU inference | Embed `ferrum_core` in a Rust service with no model server |
| 10 | Rapid prototyping | Compare tokenizers, contexts, and sizes in minutes with identical APIs |

(Tabular classification and regression now live in the sibling **Ferrum-ML**
project.)

**A starter project.** Train a small BPE transformer on your own shell history
and you have a private, offline command autocomplete. That exercises every
strength above.

**The decision rule:** use CPU-bound inference when the model is small, the
task is narrow, and the value lies in being local, private, offline, cheap,
deterministic, or embeddable. Do not use it when the value depends on a large
model's breadth, real-time responses from a big model, or reliable open-ended
reasoning.

| Situation | CPU-bound inference? |
|---|:-:|
| Keyword spotting, sensor anomaly detection on a device | ✅ Excellent |
| Private, offline autocomplete or classification | ✅ Excellent |
| Teaching how models work, reproducible research | ✅ Excellent |
| A small open model offline for a private demo | ✅ Workable (slow) |
| Browser "try it yourself" demo | ✅ Good |
| Interactive chat with a ChatGPT-class model | ❌ Use a GPU or cloud |
| High-throughput serving of a large model | ❌ Use accelerators |
| Reliable facts or logic from a tiny model | ❌ Wrong tool entirely |

This niche is real and growing under the names *edge AI* and *TinyML*. One
documented ESP32-S3 deployment ran inference in about 113 ms at about 5.8 mA,
enough for a week on battery.

---

## 16. FAQ

**Does it need a GPU?** No, and it never uses one, even if one is present.

**What dependencies does it have?** None for `ferrum_core` and `slm_cli`.
`slm_wasm` uses `wasm-bindgen`, and the GUI uses Tauri.

**Character-level or BPE?** BPE (`--vocab 512–4096`) for real text,
multilingual text, or anything long. Character-level (`--vocab 0`) for tiny
corpora and maximum transparency.

**Where does the tokenizer live?** Inside the model file
(`tokenizer_state`). With `--tokenizer`, it is also saved to disk so later
runs reuse it.

**Is BPE compatible with QAT?** Yes. Tokenization changes only the token
stream. QAT acts on the weights.

**How big are the models?** int8 files are about one byte per parameter plus
small f32 tensors, so a 1M-parameter model is about 1 MB. int4 is about half
that.

**Why does one epoch take so long?** If you passed `--stride 1`, each epoch
trains on every token about `context_len` times. Use the default stride, and
prefer `--tokens` to `--epochs` for real runs.

**Training crashed after hours. Did I lose everything?** Not if you used
`--checkpoint`. Re-run with `--resume <file>`.

**Can it run models I downloaded?** Llama/Qwen GGUFs, yes
([§9](#9-external-models-gguf-run-fine-tune-export)). Other architectures are
rejected.

**Can I fine-tune them?** Yes, with full AdamW at f32. That works up to about
135M parameters comfortably on 16 GB. 1B is out of reach (about 16 GB of
optimizer state alone).

**Can it write GGUF?** Yes: `export-gguf` produces files that run in
llama.cpp, ollama, and LM Studio.

**Does `--chars` count characters or tokens?** Characters, for both
tokenizers.

**"Corpus must be longer than the context window"?** BPE compresses, so a
short or repetitive corpus may have too few tokens. Use more text, a smaller
`--context`, or a smaller `--vocab`.

**Is it deterministic?** Yes. Training, tokenizer learning, and generation are
deterministic for a fixed seed and configuration, and identical across matmul
thread counts.

**Will my old models load?** v4 and v5 FINF files load. Pre-tokenizer files
default to character-level. Legacy one-hot MLP models load but cannot
generate.

**Where did the tabular trainer and the web playground go?** The tabular
trainer is in `../Ferrum-ML`. The old one-hot web playground is archived in
`../Ferrum-Junk`.

**What can it not do?** See the next chapter.

---

## 17. Limits: an honest critique

> This is deliberately the harshest chapter. A tool whose limits you
> understand is one you can trust. Limits are labelled **(Design)** for
> deliberate trade-offs and **(Fundamental)** for what no engineering on a
> project like this can remove.

### 17.1 A teaching-and-edge engine, not an intelligence

Ferrum is not a drop-in replacement for large GPU-trained LLMs. Even when it
runs a downloaded billion-parameter model, it does so at a few tokens per
second. That proves capability; it is not a usable chatbot.

### 17.2 Limits chosen on purpose (Design)

- **The models are small.** Ferrum models range from kilobytes to tens of
  megabytes. Commercial models are gigabytes to terabytes trained on much of
  the internet. A small corpus is memorized more than generalized (seen-text
  perplexity of about 1.0 against about 3.7 unseen in §6.10). Do not expect
  long-form writing, factual Q&A, conversation, or complex
  instruction-following.
- **CPU only is a hard ceiling.** Compute-optimal training tops out around
  5M parameters per day on a laptop, and generation barely speeds up with more
  cores.
- **Zero dependencies costs peak speed.** Without explicit SIMD or a tuned
  BLAS, much of a modern CPU's vector throughput goes unused.
- **You supply the intelligence ceiling.** Ferrum ships no pretrained weights
  of its own. Quality is bounded by your data and patience
  ([§3](#3-data-decides-garbage-in-garbage-out)).
- **The GUI is verified by compilation and backend tests, not on every
  machine.** Build it yourself before relying on it.

### 17.3 Limits shared by all small models

- **More hallucination, less reasoning.** Smaller models show fewer reasoning
  capabilities and more hallucination, due to scale. Never wire a small
  generative model to anything consequential without a human or a hard rule
  checking its output.
- **Brittle logic.** Rule-following, multi-step reasoning, and cross-step
  consistency are weak. Do not expect arithmetic or long constraint chains.

### 17.4 What language models will never achieve (Fundamental)

- **They predict; they do not understand.** You can read every line of Ferrum
  and confirm there is no understanding module, only weighted guesses about
  the next token. It can find something *probable*; it cannot *know* that it
  is true.
- **Hallucination can be reduced, not eliminated.** Treat every
  factual-sounding output as unverified.
- **No grounding, memory, or goals.** There are no senses, no live
  information, no memory beyond frozen weights and a short context window,
  and no ability to explain *why*.
- **No substitute for thinking.** The fluency is real; the reliability is not.

### 17.5 The fair counterpoint

Most Design limits are the *reasons* to choose Ferrum:

- "Too small to be ChatGPT" is the same property as "small enough to audit
  and run on a Raspberry Pi".
- "CPU-only" is the same property as "no driver stack, runs offline
  anywhere".
- "You train it yourself" is the same property as "your data never leaves
  your machine".
- "Writes its own maths" is the same property as "nothing to vet".

> **Verdict:** Ferrum will never be intelligent, a knowledge source, or
> trustworthy without verification, and that is fine. It was built to be
> small, transparent, private, and yours. Judge it as a precise tool, not a
> substitute brain, and it excels.

---

## Sources

- Vaswani et al., *Attention Is All You Need* (2017) — https://arxiv.org/abs/1706.03762
- NVIDIA Research, *Small Language Models Are the Future of Agentic AI* — https://research.nvidia.com/labs/lpr/slm-agents/
- *Small Language Models for Agentic Systems: A Survey* (arXiv 2510.03847) — https://arxiv.org/pdf/2510.03847
- Hugging Face LLM Course, *Byte-Pair Encoding tokenization* — https://huggingface.co/learn/llm-course/en/chapter6/5
- A. Karpathy, *minbpe* — https://github.com/karpathy/minbpe
- Gunasekar et al., *Textbooks Are All You Need* (arXiv 2306.11644) — https://arxiv.org/abs/2306.11644
- Xu et al., *Data-Centric AI in the Age of Large Language Models* (arXiv 2406.14473) — https://arxiv.org/abs/2406.14473
- TechTarget, *What is garbage in, garbage out (GIGO)?* — https://www.techtarget.com/searchsoftwarequality/definition/garbage-in-garbage-out
- Nagle, Redman & Sammon, *Only 3% of Companies' Data Meets Basic Quality Standards*, HBR (2017) — https://hbr.org/2017/09/only-3-of-companies-data-meets-basic-quality-standards
- Krishnamoorthi, *Quantizing deep convolutional networks for efficient inference* (arXiv 1806.08342) — https://arxiv.org/pdf/1806.08342
- NVIDIA Technical Blog, *Improving INT8 Accuracy Using Quantization Aware Training* — https://developer.nvidia.com/blog/improving-int8-accuracy-using-quantization-aware-training-and-tao-toolkit/
- Markaicode, *Rust for ML: Building High-Performance Inference Engines in 2025* — https://markaicode.com/rust-ml-Building-high-performance-inference-engines-2025/
- Carnelos et al., *MicroFlow: An Efficient Rust-Based Inference Engine for TinyML* (arXiv 2409.19432) — https://arxiv.org/abs/2409.19432
- Talent500, *What Is TinyML?* — https://talent500.com/blog/what-is-tinyml-introduction/
- *Deploying TinyML for energy-efficient object detection…*, Nature Scientific Reports (2025) — https://www.nature.com/articles/s41598-025-27818-9
- *On the Fundamental Limits of LLMs at Scale* (arXiv 2511.12869) — https://arxiv.org/pdf/2511.12869
- Huang et al., *A Survey on Hallucination in Large Language Models* (arXiv 2311.05232) — https://arxiv.org/pdf/2311.05232
- Tauri — https://tauri.app/
