# The Ferrum Manual — A Beginner's Guide

Welcome. This `manual/` folder explains the **Ferrum** project from the ground up,
for people who have *never* touched generative AI, Rust, or machine learning. You
do not need a maths degree, a GPU, or any prior experience. If you can read a
recipe, you can read this manual.

Ferrum is a small, self-contained engine — written entirely in the Rust
programming language — that lets you **build, train, and run your own tiny
"language model" on an ordinary computer's CPU**, with no graphics card, no
internet connection, and no giant downloads. It can also **import and run small
open-weight models** (Llama/Qwen checkpoints) on the same CPU. Think of it as a
"model from scratch" kit where every gear is visible and hackable.

This manual deliberately separates the *excitement* from the *honesty*. Several
documents explain what Ferrum and small models like it **can** do; others
([`07-critique.md`](07-critique.md) and the limits sections) are dedicated to what
they **cannot** do, and never will. One file
([`06-data-gigo-and-why-good-data-wins.md`](06-data-gigo-and-why-good-data-wins.md))
argues the thing beginners most often miss: that the **data**, not the model,
usually decides success. All of it is true at the same time, and reading the whole
is the point.

---

## How to read this manual

Read the numbered files in order if you are starting from zero. Jump straight to
the topic you need if you are not.

| # | File | What it covers | Read this if… |
|---|------|----------------|---------------|
| 1 | [`01-generative-ai-slms-and-transformers.md`](01-generative-ai-slms-and-transformers.md) | Generative AI, LLMs, **SLMs**, Transformers, and tokenization — for total beginners | You don't yet know what any of these words mean |
| 2 | [`02-rust-and-why-it-matters.md`](02-rust-and-why-it-matters.md) | What the **Rust** language is, and why this engine is written in it | You want to understand the "engine room" |
| 3 | [`03-the-ferrum-engine-and-its-capabilities.md`](03-the-ferrum-engine-and-its-capabilities.md) | What **Ferrum** actually is, how it's built, what it can do (including importing GGUF models), and its shortcomings | You want the full tour of the project |
| 4 | [`04-non-english-text-and-practical-uses.md`](04-non-english-text-and-practical-uses.md) | How Ferrum handles **non-English / non-Latin text**, plus a section of **practical use cases** | You work with multilingual text or want real-world ideas |
| 5 | [`05-using-the-gui.md`](05-using-the-gui.md) | Step-by-step guide to **Ferrum SLM Studio**, the point-and-click app | You'd rather click buttons than type commands |
| 6 | [`06-data-gigo-and-why-good-data-wins.md`](06-data-gigo-and-why-good-data-wins.md) | The importance of **data**, the **GIGO** principle, and what good data actually is | You're about to train *anything* (read this first) |
| 7 | [`07-critique.md`](07-critique.md) | An honest, pointed **critique** — what these models will *never* achieve | You want the unvarnished truth before you rely on it |
| 8 | [`08-applications.md`](08-applications.md) | What **CPU-bound inference** is genuinely good for (and where it isn't) | You're deciding whether to use this for a real task |

---

## The one-paragraph summary

A *language model* is a program that has read a lot of text and learned to guess
the next chunk of text. *Large* language models (LLMs) like ChatGPT need
warehouses of expensive graphics cards. *Small* language models (SLMs) are
shrunk-down versions that can run on a phone or a laptop — and a growing body of
industry research argues they are "sufficiently powerful, inherently more
suitable, and necessarily more economical" for many real jobs.[^nvidia] Ferrum is
a teaching-grade, zero-dependency Rust engine for building such small models **by
hand**, running them on a plain CPU, and even importing small open-weight models
to run offline. It is brilliant for learning, privacy, offline use, and tiny
embedded tasks — and it is *not* a replacement for the giant cloud models, by
design.

[^nvidia]: NVIDIA Research, *Small Language Models Are the Future of Agentic AI*. https://research.nvidia.com/labs/lpr/slm-agents/

---

*This manual was written to be honest. Where a claim rests on outside research, it
links to the source. Where Ferrum has limits, it says so plainly.*
