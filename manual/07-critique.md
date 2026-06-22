# A Critique of Ferrum — and What Models Like It Will Never Achieve

> This document is deliberately the harshest in the manual. The rest of the
> manual explains what Ferrum *is* and *can do*; this one is about its ceilings —
> some self-imposed, some shared by all small models, and some fundamental to
> language models as a category. None of this is meant to dismiss the project. A
> tool you understand the limits of is a tool you can trust. A tool you've only
> heard the marketing for is a liability.
>
> Two kinds of limit are mixed here on purpose, and they're labelled:
> **(Design)** limits that are deliberate trade-offs Ferrum chose, and **(Fundamental)**
> limits that no amount of engineering on a project like this can remove.

---

## 1. The headline: this is a teaching-and-edge engine, not an intelligence

Ferrum's own FAQ says it plainly: *"It is not a drop-in replacement for large
GPU-trained LLMs."* Everything below is really a series of footnotes to that one
sentence. The danger is not that Ferrum is weak — it's excellent at its job — but
that a beginner, fresh off [page 1](01-generative-ai-slms-and-transformers.md),
might expect a pocket ChatGPT. It is not that, will not be that, and was never
trying to be that.

---

## 2. Limits Ferrum chose on purpose (Design)

These are trade-offs, not failures. But trade-offs have a downside, and honesty
means naming it.

### 2.1 The models are extraordinarily small

Ferrum builds models measured in **tens of kilobytes**, trained on corpora a
single person can prepare on a laptop. Commercial models are measured in
**gigabytes to terabytes**, trained on a meaningful fraction of the public
internet. This is a difference of *six to nine orders of magnitude*. The
consequence is blunt: a Ferrum model trained on a small corpus will **memorise
far more than it generalises**. The project's own walkthrough shows this candidly
— a tiny corpus yields ~1.0 perplexity on *seen* text and ~3.7 on *unseen* text,
the textbook signature of memorisation.

> **What this means you cannot expect:** coherent long-form writing, factual
> question-answering, holding a conversation, following complex instructions, or
> "knowing" anything beyond the patterns in the small text you fed it.

### 2.2 CPU-only is a hard ceiling, not just "slower"

No GPU is ever used. For tiny models on edge devices, that's the whole point. But
it means there is a practical wall on model size and training data: the project's
own benchmarks show a *small* transformer taking **over two minutes to train even
across 8 cores**, and generation scaling only ~1.25× with parallelism because
text production is an inherently sequential chain. You will not train a large,
capable model on a CPU in a reasonable time, and Ferrum doesn't pretend
otherwise.

### 2.3 "Zero dependencies" sacrifices peak performance

By writing its own matrix maths instead of using decades-tuned libraries
(OpenBLAS, Intel MKL, oneDNN), Ferrum buys purity, auditability, and portability
— and gives up raw speed. It is **not the fastest CPU inference engine**, and by
design it never will be. That's a reasonable price for a teaching/edge tool, but
if your goal is maximum throughput on a server, a dependency-rich engine will
beat it.

### 2.4 You supply the intelligence ceiling

Ferrum ships no pretrained weights. Quality is bounded entirely by *your* corpus
and *your* patience. This is a privacy and learning feature — but it means there
is no shortcut to a capable model. Garbage in, garbage out, with no giant
pretraining to fall back on — a principle important enough to get its own page:
[`06-data-gigo-and-why-good-data-wins.md`](06-data-gigo-and-why-good-data-wins.md).

### 2.5 The GUI is not yet proven on every machine

The project openly states the desktop GUI was authored in an environment that
couldn't compile it, depends on heavy system WebView libraries, and "has not been
compiled here." The engine underneath is tested; the GUI shell is a documented
question mark until you build it yourself. Stated honestly by the project — and
worth repeating here.

---

## 3. Limits shared by all *small* models (Fundamental-ish)

Making the model bigger would *reduce* these, but Ferrum is small by definition,
so it lives with them more acutely than a frontier model does.

### 3.1 More hallucination, less reasoning

This is not a Ferrum bug; it is a measured property of scale. Research finds that
**smaller models "exhibit fewer reasoning capabilities and more hallucinated
behaviors… likely due to the lack of scale,"** and that even models *trained* to
avoid hallucination still hallucinate more when small.[^limits] A small model
will confidently produce text that is fluent and wrong, and it will do so *more
often* than a large one. Never wire a small generative model to anything
consequential without a human or a hard rule checking its output.

### 3.2 Brittle logic and multi-step tasks

The same body of research notes that "adherence to rules/logic, exploitation of
reasoning patterns, and cross-step consistency remain brittle in language
models," and that they "struggle with multi-step reasoning" and can "deviate from
the original context and instructions."[^limits] Small models feel this hardest.
Don't expect Ferrum to do arithmetic, follow a long recipe of constraints, or
chain several reasoning steps reliably.

---

## 4. What language models — Ferrum included — will *never* achieve (Fundamental)

These are not "wait for the next version" problems. They follow from *what a
next-token predictor is*. Scaling up reduces the symptoms; it does not change the
nature of the thing.

### 4.1 It does not understand; it predicts

A language model "relies on patterns in datasets rather than in-depth content
understanding," which can "produce structurally coherent but content-flawed
outputs," and these limitations are "fundamental to how language models operate,
not merely engineering problems to be solved by scaling alone."[^limits] Ferrum,
being a small and transparent instance of exactly this mechanism, makes the point
vivid: you can read every line and confirm there is no "understanding" module
anywhere — only weighted guesses about the next token. **It will never *know* that
something is true. It can only find it *probable*.**

### 4.2 Hallucination cannot be fully eliminated, only reduced

A survey of hallucination in language models frames it as a deep, open research
problem with a whole taxonomy of failure modes — not a solved one.[^hallu] Some
researchers argue a non-zero rate of confident fabrication is *inherent* to
models that must always produce *an* answer from a probability distribution. For
a tiny model with little to ground itself in, the rate is simply higher. Treat
**every** factual-sounding output as unverified.

### 4.3 No grounding in the world, no genuine memory, no goals

Ferrum has no senses, no access to live information, no persistent memory beyond
its frozen weights and its short context window, and no goals of its own. It
cannot learn from a conversation after training, cannot look something up, and
cannot tell you *why* it produced an answer. These aren't missing features to be
added later in a project of this kind — they're outside the category of "a
function that maps text to a next-token guess."

### 4.4 It will never be a substitute for thinking

The most important critique is the human one. A small model that produces fluent
text is *especially* seductive precisely because it's easy to run and feels
authoritative. The fluency is real; the reliability is not. Ferrum is a wonderful
way to *learn how the machinery works* and to *do narrow, checkable jobs* — and a
terrible thing to outsource judgment to.

---

## 5. The fair counter-point (so this critique stays honest)

It would be dishonest to end on only the negatives, because most of §2 is the
*reason to choose Ferrum*, not a reason to avoid it:

- "Too small to be ChatGPT" is the same property as "small enough to run on a
  Raspberry Pi, audit completely, and reproduce bit-for-bit."
- "CPU-only and slow at scale" is the same property as "no GPU, no driver stack,
  runs offline anywhere."
- "You must train it yourself" is the same property as "your data never leaves
  your machine."
- "Writes its own slow maths" is the same property as "zero dependencies, nothing
  to vet, no supply-chain risk."

A growing industry literature even argues that, for the *narrow, repetitive*
tasks that dominate real systems, small models are not a sad compromise but the
*correct* choice — "sufficiently powerful, inherently more suitable, and
necessarily more economical."[^nvidia] Ferrum is a faithful, transparent member
of that small-model world.

---

## 6. The one-line verdict

> **Ferrum will never be intelligent, never be a knowledge source, never be
> trustworthy without verification, and never rival a frontier LLM's breadth —
> and that is fine, because it was built to be small, transparent, private, and
> yours.** Judge it as a precision hand tool, not as a substitute brain, and it
> excels. Judge it as a pocket ChatGPT, and it will disappoint you exactly as much
> as you misunderstood it.

---

## Sources

- *On the Fundamental Limits of LLMs at Scale* (arXiv 2511.12869) — https://arxiv.org/pdf/2511.12869
- Huang et al., *A Survey on Hallucination in Large Language Models* (arXiv 2311.05232) — https://arxiv.org/pdf/2311.05232
- NVIDIA Research, *Small Language Models Are the Future of Agentic AI* — https://research.nvidia.com/labs/lpr/slm-agents/
- Ferrum project docs in this repository: `docs/FAQs.md`, `instructions.md`, `benchmarks.md`, `ferrum_gui/README.md`

[^limits]: *On the Fundamental Limits of LLMs at Scale* (arXiv 2511.12869) — smaller models show fewer reasoning capabilities and more hallucination; rule/logic adherence and multi-step consistency are brittle; models rely on dataset patterns rather than understanding; these are framed as fundamental, not merely engineering, limitations. https://arxiv.org/pdf/2511.12869
[^hallu]: Huang et al., *A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions* (arXiv 2311.05232). https://arxiv.org/pdf/2311.05232
[^nvidia]: NVIDIA Research, *Small Language Models Are the Future of Agentic AI*. https://research.nvidia.com/labs/lpr/slm-agents/
