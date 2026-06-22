# 1. Generative AI, SLMs, and Transformers — From Absolute Zero

> **Who this is for:** someone who has heard the words "AI", "ChatGPT",
> "language model", or "transformer" but could not confidently explain any of
> them. By the end of this page you will understand all of them well enough to
> use Ferrum and to follow the rest of this manual. No maths required — just
> curiosity.

---

## 1.1 The single idea behind it all: "guess the next bit"

Imagine a game. I show you a sentence with the last word missing:

> "The cat sat on the ____."

You instantly think **"mat"** (or "sofa", "windowsill", "keyboard"). You are not
remembering this exact sentence — you have *never seen it before*. You are using
a lifetime of reading and listening to **predict what is likely to come next**.

That is the entire secret of modern "generative AI". A **language model** is a
computer program that has been shown enormous amounts of text and trained to do
exactly this: given some text, predict the next small piece. Do that prediction
once, glue the predicted piece onto the end, and predict again — and again, and
again — and the machine *generates* brand-new text one piece at a time. This
loop is called **autoregressive generation** ("auto" = self, "regressive" =
feeding its own output back in).

That's it. Everything else — ChatGPT, the model in your phone's keyboard, and
Ferrum — is engineering built on top of "guess the next bit, then repeat."

**"Generative AI"** is simply the umbrella term for AI systems that *produce* new
content (text, images, audio, code) rather than just *classifying* existing
content ("is this email spam: yes/no"). This manual is about the text kind.

---

## 1.2 What does it mean to "train" a model?

A freshly created model knows nothing — its internal numbers (called
**parameters** or **weights**) start out random, so its guesses are gibberish.

**Training** is the process of slowly nudging those millions of numbers until the
guesses get good. The recipe is a loop:

1. Show the model a real chunk of text and hide the next piece.
2. Let the model guess.
3. Measure how wrong it was. This wrongness score is called the **loss**.
4. Nudge every weight a tiny amount in the direction that would have reduced the
   loss. (The maths for "which direction" is called **backpropagation** and
   **gradient descent** — Ferrum writes this out by hand so you can read it.)
5. Repeat millions of times.

One full pass over your training text is called an **epoch**. As training
proceeds, the loss falls — that's the model getting less wrong. You will watch
this number drop live when you train a model in Ferrum.

> **Key intuition:** the model is not a database that "looks up" answers. It is a
> giant adjustable function whose knobs have been tuned so that plausible text
> comes out. This is also *why it can be confidently wrong* — see
> [`07-critique.md`](07-critique.md).

---

## 1.3 Tokens: how a computer reads text

Computers don't see letters; they see numbers. So before any of this works, text
must be chopped into pieces and each piece given an ID number. The pieces are
called **tokens**, and the chopping tool is the **tokenizer**.

A token can be a whole word, a piece of a word ("sub-word"), a single character,
or even a single **byte**. For example, the word `lower` might become two tokens:
`low` + `er`. The model then works purely with the ID numbers; at the very end it
translates the numbers back into text.

### Byte-Pair Encoding (BPE) — the clever middle ground

Two extremes are possible. One token per **character** is simple but wasteful —
the model has to take many tiny steps to produce one word. One token per **word**
is compact but breaks the moment it meets a word it never saw during training.

**Byte-Pair Encoding (BPE)** is the popular compromise used by GPT-2, GPT-4, and
most modern models. It starts from the smallest possible alphabet and then
*learns* which pairs of symbols occur together most often, merging them into new
tokens. So a corpus full of the word "lower" will learn a merge for `low`, then
`lower`, automatically — without anyone listing the words in advance.[^bpe]

Ferrum uses a specific, powerful variant called **byte-level BPE**. Instead of
starting from characters, it starts from the **256 possible byte values** that
every digital file is made of. This has a magical consequence: *literally any
text that can be stored on a computer can be tokenized* — English, emoji, Chinese,
Arabic, Cyrillic, mathematical symbols — with no possibility of an "unknown
character" error.[^gpt2bpe] (More on this in
[`04-non-english-text-and-practical-uses.md`](04-non-english-text-and-practical-uses.md).)

[^bpe]: Hugging Face, *Byte-Pair Encoding tokenization*. https://huggingface.co/learn/llm-course/en/chapter6/5
[^gpt2bpe]: GPT-2 introduced byte-level BPE starting from 256 raw byte tokens, "guaranteeing that literally any byte sequence has a valid tokenization — no `<UNK>` token needed, ever." Summary via Hugging Face course and minbpe by Andrej Karpathy: https://github.com/karpathy/minbpe

---

## 1.4 The Transformer: the engine inside the engine

For decades, models read text strictly left-to-right, one word at a time, trying
to hold the whole sentence in a small "memory." They were slow and forgetful.

In 2017, a Google research paper with the cheeky title **"Attention Is All You
Need"** introduced the **Transformer** architecture, and it changed
everything.[^attention] Almost every famous AI system today — GPT, Claude,
Gemini, Llama — is a Transformer. Ferrum implements one too, by hand, in plain
Rust.

The Transformer's breakthrough is a mechanism called **self-attention**. Here is
the intuition without any maths:

> When the model processes a word, self-attention lets it **look at every other
> word in the context at the same time** and decide which ones matter for
> understanding this word.

Take the sentence *"The animal didn't cross the street because **it** was too
tired."* What does "it" refer to — the animal or the street? A human knows it's
the animal. Self-attention is the machinery that lets the model *attend* to the
word "animal" when interpreting "it", weighting it more heavily than "street".

A few supporting parts make this work; you'll see all of them named in Ferrum's
code and menus, so here is a plain-language glossary:

| Term | Plain-language meaning |
|------|------------------------|
| **Self-attention** | Every word looks at every other word and decides what's relevant. |
| **Multi-head attention** | Several attention "viewpoints" run in parallel, each spotting a different kind of relationship (grammar, topic, position…). |
| **Embedding** | A lookup table that turns each token ID into a list of numbers the model can do maths on. |
| **Positional encoding** | Extra information that tells the model the *order* of the words (since attention looks at all of them at once). |
| **Feed-forward network (FFN)** | A small classic neural network applied after attention to "think" about what attention gathered. |
| **Layer normalization** | A stabilizer that keeps the numbers in a healthy range so training doesn't blow up. |
| **Block / layer** | One full round of (attention + FFN + normalization). Stacking several blocks lets the model learn deeper patterns. |
| **Context window** | How many tokens the model can look at at once — its "field of view". |

You do **not** need to memorise these. You just need to know that a Transformer
is a stack of these blocks, and that **attention** is the famous trick. Ferrum
lets you set the number of heads, blocks, embedding size, and context window
yourself.

[^attention]: Vaswani et al., *Attention Is All You Need*, NeurIPS 2017. https://arxiv.org/abs/1706.03762 — introduced the Transformer and self-attention; one of the most-cited computer-science papers in history.

---

## 1.5 LLMs vs. SLMs: the big and the small

You'll constantly see two acronyms. They describe the *same kind of thing* at two
*very different sizes*.

### LLM — Large Language Model

The giants: ChatGPT, Claude, Gemini, Llama. "Large" refers to the number of
parameters (those tunable knobs from §1.2) — **billions to trillions** of them.
Training one can cost millions of dollars and requires data centres full of
specialised graphics cards (GPUs). Running ("inference") usually happens in the
cloud, on someone else's expensive hardware, over the internet.

### SLM — Small Language Model

The same architecture, deliberately kept small — typically **under ~10 billion
parameters**, and often *far* smaller — so it can run on everyday hardware: a
laptop, a phone, a Raspberry Pi, even a microcontroller.[^slm-def]

The trade is obvious: a smaller model knows less and reasons less broadly than a
giant. But a wave of recent industry research argues this trade is *worth it* for
a huge fraction of real tasks. NVIDIA's 2025 position paper, *Small Language
Models Are the Future of Agentic AI*, argues that SLMs are "sufficiently
powerful, inherently more suitable, and necessarily more economical" for the
many small, repetitive jobs that make up most real AI pipelines.[^nvidia] The
advantages cited across the field are consistent: **lower latency, smaller memory
footprint, lower cost, easier deployment, and the ability to run privately and
offline.**[^slm-adv]

> **Where Ferrum sits:** Ferrum builds models at the *very small* end — small
> enough to read, audit, and run anywhere, and small enough that you can train
> one yourself in seconds to minutes on a laptop CPU. They are a fraction of the
> size of even a "small" commercial SLM. That makes Ferrum superb for learning,
> experimentation, and narrow tasks — and unsuitable as a ChatGPT replacement.
> This manual is careful never to blur that line; see
> [`03-the-ferrum-engine-and-its-capabilities.md`](03-the-ferrum-engine-and-its-capabilities.md)
> and [`07-critique.md`](07-critique.md).

[^slm-def]: NVIDIA researchers define SLMs as models compact enough to run on everyday devices while serving a single user with low latency — as of 2025, generally under ~10 billion parameters. https://research.nvidia.com/labs/lpr/slm-agents/
[^nvidia]: NVIDIA Research, *Small Language Models Are the Future of Agentic AI*. https://research.nvidia.com/labs/lpr/slm-agents/
[^slm-adv]: Survey discussion of SLM advantages (latency, memory, cost, deployability): *Small Language Models for Agentic Systems: A Survey* (arXiv 2510.03847) https://arxiv.org/pdf/2510.03847 ; Analytics Vidhya, *SLMs for Agentic AI* https://www.analyticsvidhya.com/blog/2025/08/slms-for-agentic-ai/

---

## 1.6 "Inference" and "training" — the two phases

Two more words you'll see everywhere:

- **Training** = teaching the model (the slow, one-time, compute-heavy phase from
  §1.2).
- **Inference** = *using* the already-trained model to generate text or make a
  prediction. This is the fast, repeatable phase that happens every time you ask
  the model something.

Ferrum is described as an **"inference engine"** because running models is its
core job — but, unusually, it *also* lets you do the training, all on the CPU,
all in one self-contained tool.

---

## 1.7 Putting it together: the journey of one sentence

Here is the whole pipeline, start to finish, using everything above:

```
Your text:   "the quick brown fox"
     │
     ▼  (1) TOKENIZER chops it into tokens and gives IDs
   [the][ quick][ brown][ fox]  →  [412, 88, 1290, 77]
     │
     ▼  (2) EMBEDDING turns each ID into a list of numbers
     │
     ▼  (3) TRANSFORMER BLOCKS apply self-attention + FFN, several times,
     │       so each token "understands" its context
     │
     ▼  (4) The model outputs a probability for every possible next token
     │       e.g. " jumps" 71%, " ran" 9%, " sat" 4%, …
     │
     ▼  (5) SAMPLING picks one (temperature controls how adventurous)
   " jumps"
     │
     └──►  glue it on, and repeat from step 1 →  "the quick brown fox jumps …"
```

The word **temperature** in step 5 is your creativity dial: low temperature
(0.1–0.3) makes the model play it safe and repeat learned patterns; high
temperature (0.8+) makes it take risks and produce more varied — but riskier —
text. You set this every time you generate with Ferrum.

---

## 1.8 What you now know

- Generative AI = "predict the next piece, then repeat."
- Models learn by **training** (reducing **loss** over **epochs**) and are used
  via **inference**.
- **Tokenizers** chop text into **tokens**; **byte-level BPE** can encode *any*
  text.
- The **Transformer** with **self-attention** is the architecture behind modern
  models, including Ferrum's.
- **LLMs** are giant and cloud-bound; **SLMs** are small and run anywhere —
  and Ferrum builds models at the small, transparent, do-it-yourself end.

Next: [why this engine is written in **Rust**](02-rust-and-why-it-matters.md),
and why that choice matters more than it might first appear.

---

## Sources

- Vaswani et al., *Attention Is All You Need* (2017) — https://arxiv.org/abs/1706.03762
- NVIDIA Research, *Small Language Models Are the Future of Agentic AI* — https://research.nvidia.com/labs/lpr/slm-agents/
- *Small Language Models for Agentic Systems: A Survey* (arXiv 2510.03847) — https://arxiv.org/pdf/2510.03847
- Hugging Face LLM Course, *Byte-Pair Encoding tokenization* — https://huggingface.co/learn/llm-course/en/chapter6/5
- A. Karpathy, *minbpe* (minimal byte-level BPE) — https://github.com/karpathy/minbpe
- Analytics Vidhya, *SLMs for Agentic AI* — https://www.analyticsvidhya.com/blog/2025/08/slms-for-agentic-ai/
