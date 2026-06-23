# 6. Data — "Garbage In, Garbage Out", and Why Good Data Makes or Breaks an SLM

> **Who this is for:** anyone about to train a model in Ferrum (or anywhere). If you
> only read one page of this manual before building something, **read this one.**
> Newcomers obsess over the model — the architecture, the number of heads, the clever
> settings. Practitioners know a quieter truth: **the data decides almost
> everything.** A mediocre model on great data beats a great model on bad data,
> almost every time.

---

## 6.1 The uncomfortable headline

A language model has no senses, no experience, and no access to the world. The
*only* thing it ever sees is the text you train it on. It cannot learn what isn't
there, and it cannot un-learn what is. So a model is, quite literally, a compressed
reflection of its training data.

> **The model is a mirror. The data is the face.** If the data is clean, relevant,
> and varied, the model reflects something useful. If the data is junk, the model
> faithfully reflects junk — fluently and confidently.

This is why experienced teams now talk about **"data-centric AI"**: the idea that the
biggest gains come not from tweaking the algorithm but from improving the *data*. As
the research literature puts it, model performance is "heavily based on the quality
and characteristics of the data used for training, rather than solely focusing on
algorithmic improvements" — and yet data "receives disproportionally low
attention."[^datacentric] This page gives it the attention it deserves.

[^datacentric]: Xu et al., *Data-Centric AI in the Age of Large Language Models* (arXiv 2406.14473). https://arxiv.org/abs/2406.14473

---

## 6.2 GIGO: Garbage In, Garbage Out

There is a phrase older than almost everyone reading this, and it has never been more
relevant:

> **Garbage In, Garbage Out (GIGO):** the quality of a system's output is directly
> determined by the quality of its input. Feed a computer flawed input and it
> produces flawed output — every time.

The phrase was first recorded in **1957** and is usually credited to **George
Fuechsel**, an IBM programmer and instructor in the early 1960s, who used it to
explain — memorably — that a program "will produce erroneous output if given
erroneous input."[^gigo] It was a truth about ordinary computer programs decades
before anyone trained a neural network. Machine learning didn't repeal it; it
**amplified** it.

Here's the crucial difference. A traditional program with a bug does the *wrong thing
in a predictable way* you can find and fix. A model trained on bad data does the
*wrong thing in a way that's baked into millions of numbers*, invisibly, and it keeps
doing it confidently wherever you apply it. As one summary puts it: "if a machine
learning model is not given correct training data, the model will learn incorrectly
and produce incorrect output wherever its knowledge is applied."[^gigo]

And the scale of the problem is sobering: a widely-cited Harvard Business Review
analysis found that **only about 3% of companies' data meets basic quality
standards.**[^hbr] Most data, in other words, is *not* ready to be learned from
as-is.

[^gigo]: TechTarget, *What is garbage in, garbage out (GIGO)?* — origin (1957), attribution to George Fuechsel, and its meaning for ML. https://www.techtarget.com/searchsoftwarequality/definition/garbage-in-garbage-out
[^hbr]: Nagle, Redman & Sammon, *Only 3% of Companies' Data Meets Basic Quality Standards*, Harvard Business Review (2017). https://hbr.org/2017/09/only-3-of-companies-data-meets-basic-quality-standards

---

## 6.3 Why data matters *even more* for a Small Language Model

You might think GIGO is a bigger problem for the giant models, since they eat more
data. The opposite is true in an important way — and it's the heart of this page.

A giant LLM trained on a sizeable slice of the internet has enough capacity to
**average out** a lot of noise: a few thousand bad documents drown in trillions of
good tokens. A **Small Language Model has no such luxury.** With far fewer
parameters, it can't memorise everything and hope the good outweighs the bad. Every
token you give it has to *earn its place*. Junk doesn't get averaged away — it takes
up scarce capacity that should have gone to signal.

The most striking evidence comes from Microsoft's **Phi** project and its
famously-titled paper, **"Textbooks Are All You Need."** The team trained a *small*
1.3-billion-parameter model (`phi-1`) on a carefully curated set of "textbook-quality"
data instead of the usual giant, messy web scrape. The result stunned the field:
`phi-1` **beat models roughly 10× larger that were trained on 100× more data** on
standard coding benchmarks.[^phi] Their central finding, stated plainly: **the
quality of the data matters more than its sheer volume.**[^phidecoder]

> **The lesson for you:** with Ferrum's *tiny* models, you can't win on quantity —
> you can only win on **quality**. Curating a small, clean, relevant corpus is not a
> chore to rush through; it is the single most powerful lever you have.

[^phi]: Gunasekar et al., *Textbooks Are All You Need* (arXiv 2306.11644), Microsoft Research — `phi-1` (1.3B), trained on ~7B tokens of "textbook quality" data, outperformed models ~10× larger trained on ~100× more data on HumanEval/MBPP. https://arxiv.org/abs/2306.11644
[^phidecoder]: "The central hypothesis behind the Phi project is that the quality of data is more important than its volume." The Decoder, *Microsoft's tiny Phi-1 shows how important data quality is*. https://the-decoder.com/microsofts-tiny-phi-1-language-model-shows-the-importance-of-data-quality-in-ai-training/

---

## 6.4 So what *is* "good data", concretely?

"Good data" is not a vibe — it's a checklist of measurable qualities. Here are the
dimensions that matter, in plain language, with what each means for a Ferrum corpus:

| Quality | What it means | What goes wrong without it |
|---------|---------------|----------------------------|
| **Relevant** | The text matches the task and domain you care about. | Train on Shakespeare, get Shakespeare — not the customer-support replies you wanted. |
| **Representative** | It covers the real range of inputs the model will face, including the *language* you need. | The model fails on the cases you forgot; an English corpus can't speak French. |
| **Clean** | Free of boilerplate, markup, broken encoding, headers/footers, navigation junk, OCR errors. | The model learns to reproduce `<div>` tags, page numbers, and "Chapter copyright ©…". |
| **Correct** | Factually and grammatically sound; well-formed examples. | The model confidently learns and repeats the errors. GIGO in its purest form. |
| **Consistent** | Uniform format and style where it matters (date formats, casing). | The model wastes capacity modelling pointless variation instead of real signal. |
| **Sufficient & diverse** | Enough volume, and varied enough that the model *generalises* instead of *memorising*. | Too little or too repetitive → memorisation (high held-out perplexity; see [page 3, §3.9](03-the-ferrum-engine-and-its-capabilities.md)). |
| **De-duplicated** | Repeated passages removed. | Duplicates over-weight some text, encourage memorisation, and waste scarce capacity. |
| **Balanced / unbiased** | No accidental skew toward one group, topic, or viewpoint. | "If a system is trained on biased data, it produces biased results."[^gigo] The model amplifies the skew. |
| **Properly encoded** | Valid UTF-8 text. | Mojibake in, mojibake out (Ferrum's byte-level tokenizer survives it, but it still pollutes what's learned — see [page 4](04-non-english-text-and-practical-uses.md)). |
| **Legally & ethically yours** | You have the right to use it; sensitive data is handled appropriately. | Legal and privacy risk — one of the very reasons to train locally with Ferrum. |

A useful one-sentence test: **"If a careful human read this corpus, would they learn
the right thing from it?"** If yes, it's probably good data. If they'd come away
confused, misled, or bored by repetition — so will the model.

---

## 6.5 What *bad* data looks like in practice

Concrete villains you'll actually meet:

- **Boilerplate:** Project Gutenberg license headers/footers, website navigation
  menus, cookie banners, "Page 12 of 340", email signatures.
- **Markup and code noise:** stray HTML/XML tags, Markdown artifacts, escape
  sequences left in from scraping.
- **OCR and conversion errors:** "rn" read as "m", garbled characters from a bad PDF.
- **Duplication:** the same paragraph copied across dozens of files; a FAQ pasted on
  every page.
- **Topic soup:** an unfocused mix of unrelated content when you wanted a domain.
- **Too small / too repetitive:** a corpus so short or repetitive that the model just
  memorises it (the classic giveaway: near-perfect *training* perplexity but poor
  *held-out* perplexity).

Every one of these teaches the model something you didn't want it to learn.

---

## 6.6 How Ferrum actually helps you get good data

Ferrum takes data seriously enough to build cleaning and inspection *into the engine*
— not bolted on the side. Three tools do the work (all available from the GUI's
**Datasets** tab, see [page 5](05-using-the-gui.md), and from the library):

1. **`clean_corpus` — automated cleanup.** A configurable cleaner whose options map
   directly to the villains above: *strip Project Gutenberg boilerplate*,
   *lowercase*, *collapse whitespace*, *normalize punctuation*, *strip control
   characters*, and *max characters* (cap size for quick experiments).

2. **`corpus_stats` — know what you have.** Reports characters, bytes, lines, words,
   and **unique characters**, so you can sanity-check a corpus *before* spending time
   training. (A suspiciously low unique-character count hints at a corrupt or
   one-note file.)

3. **`validate_for_training` — fail fast, with a reason.** Checks the corpus is
   actually usable (e.g. longer than the context window) and returns a clear message
   instead of a confusing crash mid-training.

And then the all-important **feedback loop**: after training, use Ferrum's
**`evaluate`** / **Evaluate** tab to measure **held-out perplexity**
([page 3, §3.9](03-the-ferrum-engine-and-its-capabilities.md)). A big gap between
training and held-out perplexity is the engine *telling you* your data was too small
or too repetitive — a data problem, not a model problem. Most beginners reach for
bigger models when the fix is better data.

---

## 6.7 A practical checklist before you hit "Train"

- [ ] **Right content?** The text is in the domain *and language* you want the model
      to produce.
- [ ] **Cleaned?** Boilerplate, markup, and control characters stripped
      (`clean_corpus`).
- [ ] **De-duplicated?** No large passages repeated many times.
- [ ] **Big and varied enough?** Comfortably longer than the context window, with real
      variety — not the same sentence 500 times.
- [ ] **Inspected?** You looked at `corpus_stats` and *actually read a sample* with
      your own eyes.
- [ ] **Validated?** `validate_for_training` passes.
- [ ] **Plan to verify?** You set aside held-out text to measure perplexity afterward.

That five-minute discipline will do more for your results than any hyperparameter you
can tune.

---

## 6.8 The honest caveat (and the symmetry)

Two balancing truths:

1. **Good data cannot make a tiny model into a genius.** A clean, perfect corpus
   still can't give a kilobyte-scale Ferrum model the breadth of a frontier LLM — the
   size ceiling from [`07-critique.md`](07-critique.md) still applies. Good data
   raises the ceiling you *can* reach; it doesn't remove the ceiling.

2. **But bad data guarantees failure regardless of model size.** This is the
   asymmetry that makes data the priority: great data is *necessary but not
   sufficient* for a good model, while bad data is *sufficient on its own* to ruin
   one. You can't tune your way out of garbage.

So: you cannot win on data alone — but you can *definitely lose* on it. Spend your
effort accordingly.

---

## What you now know

- A model is a **mirror of its training data**; it can only learn what's there.
- **GIGO** (Garbage In, Garbage Out), a truth since 1957, hits machine learning
  *harder* than ordinary software, because the errors get baked invisibly into the
  weights.
- Data matters **even more for SLMs**: with little capacity, they can't average out
  noise — Microsoft's Phi showed curated, "textbook-quality" data letting a small
  model beat far larger ones.
- **Good data** is relevant, representative, clean, correct, consistent,
  sufficient-and-diverse, de-duplicated, unbiased, well-encoded, and rightfully yours.
- Ferrum builds in **`clean_corpus`**, **`corpus_stats`**, and
  **`validate_for_training`**, and gives you **perplexity** as a feedback loop to
  catch data problems.
- Good data raises the ceiling; **bad data guarantees the floor.**

Next, the limits no data can fix: [`07-critique.md`](07-critique.md). And to decide if
CPU-bound inference suits your task at all: [`08-applications.md`](08-applications.md).

---

## Sources

- Xu et al., *Data-Centric AI in the Age of Large Language Models* (arXiv 2406.14473) — https://arxiv.org/abs/2406.14473
- Gunasekar et al., *Textbooks Are All You Need* (arXiv 2306.11644), Microsoft Research — https://arxiv.org/abs/2306.11644
- The Decoder, *Microsoft's tiny Phi-1 shows how important data quality is for AI training* — https://the-decoder.com/microsofts-tiny-phi-1-language-model-shows-the-importance-of-data-quality-in-ai-training/
- TechTarget, *What is garbage in, garbage out (GIGO)?* — https://www.techtarget.com/searchsoftwarequality/definition/garbage-in-garbage-out
- Nagle, Redman & Sammon, *Only 3% of Companies' Data Meets Basic Quality Standards*, Harvard Business Review (2017) — https://hbr.org/2017/09/only-3-of-companies-data-meets-basic-quality-standards
- Ferrum project docs in this repository: `ferrum_core/src/dataset.rs`, `ferrum_gui/src/commands.rs`, `instructions.md`
