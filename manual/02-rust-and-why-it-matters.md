# 2. Rust, and Why It Matters Here

> **Who this is for:** someone who has never written a line of Rust (or any
> systems language) and wants to understand *why this particular AI engine is
> built in it*. You will not be asked to write Rust. You will come away
> understanding the engineering choices that make Ferrum unusual.

---

## 2.1 What is Rust, in one minute?

**Rust** is a programming language, first released in 2015, designed for writing
software that is **fast and safe at the same time** — two goals that used to be in
tension.

To see why that's a big deal, here's the lay of the land:

- **Python** is the language most AI is written in. It's easy and friendly, but
  it is *interpreted* and relatively slow, and it leans on a mountain of
  pre-compiled libraries (written in C/C++) to do the heavy lifting.
- **C and C++** are blazingly fast and run close to the hardware, which is why
  those AI libraries are written in them. But they are *dangerous*: a small
  mistake with computer memory causes crashes, corruption, and the majority of
  the world's serious security vulnerabilities.
- **Rust** aims to give you C/C++ speed **without** the danger. Its compiler
  enforces a set of rules (the famous "borrow checker") that make whole
  categories of memory bugs *impossible to compile*. You get the performance of
  C with guarantees that C can't offer.[^rustsafe]

The slogan people use is "**fearless**" programming: Rust catches the scary
mistakes before your program ever runs.

[^rustsafe]: "Rust guarantees memory safety at compile time with zero-cost abstractions — no runtime overhead, no dangling pointers, no null dereferences." Summary of Rust's value for ML inference: Markaicode, *Rust for ML* https://markaicode.com/rust-ml-Building-high-performance-inference-engines-2025/

---

## 2.2 The three Rust ideas you'll hear Ferrum brag about

Ferrum's documentation repeats three phrases. Now you'll know exactly what they
mean and why they're impressive.

### a) "Memory-safe, no garbage collector"

Most safe, friendly languages (Python, Java, Go) achieve safety using a
**garbage collector** — a background janitor that periodically pauses your
program to clean up unused memory. Convenient, but it costs speed and causes
unpredictable little freezes.

Rust achieves safety a *different* way: at **compile time**, through ownership
rules, so there is **no janitor and no pauses**. For an AI engine that must
produce a steady stream of tokens, "no surprise pauses" is genuinely valuable.

### b) `#![forbid(unsafe_code)]`

Rust has an escape hatch called `unsafe` that lets expert programmers bypass the
safety rules for tricky low-level work. Ferrum's core puts a single line at the
top — `#![forbid(unsafe_code)]` — which tells the compiler: **"reject this entire
program if anyone, anywhere, tries to use that escape hatch."**

In plain terms: Ferrum's engine promises that *every* memory operation is checked
by the compiler. There are no hand-waved "trust me" sections. For anyone who
needs to *audit* what an AI tool does — in regulated industries, security, or
teaching — this is a strong, machine-verified guarantee.

### c) "Zero dependencies" / "`std`-only"

This is the most unusual claim of all, and it deserves a section of its own.

---

## 2.3 The "zero dependencies" superpower

Modern software is built by stacking other people's code. A typical AI project in
Python pulls in *hundreds* of third-party packages (NumPy, PyTorch, CUDA
libraries, tokenizer libraries…), which in turn pull in hundreds more. This is
convenient, but it has real costs:

- **Bloat:** installations balloon to gigabytes.
- **Fragility:** one updated package can break everything ("dependency hell").
- **Security risk:** every outside package is code you must *trust*. A single
  compromised package can poison the whole stack (this is the famous "supply
  chain attack").
- **Auditability:** you can't realistically read all of it.

Ferrum's engine (`ferrum_core`) takes the opposite path: its list of outside
dependencies is **empty**. It uses only Rust's **standard library** (`std`) — the
batteries that ship with the language itself. Everything else — the maths, the
neural-network layers, the tokenizer, the file format, even the multi-core
parallelism — is written from scratch in the project.

What this buys you:

| Benefit | Why it matters |
|---------|----------------|
| **Tiny, fast builds** | Nothing to download; compiles to a single small binary. |
| **Nothing to vet or update** | No supply-chain surface; auditors can read the whole thing. |
| **Runs anywhere Rust runs** | Servers, laptops, Raspberry Pi, even inside a web browser (see §2.5). |
| **Total reproducibility** | No moving external parts means results don't drift over time. |
| **A real teaching artifact** | You can read the *entire* forward and backward pass with no hidden libraries. |

> This is rare. Most "from scratch" AI tutorials still import NumPy for the maths.
> Ferrum implements even the matrix multiplication itself. That's what "zero
> dependency" really means here.

---

## 2.4 Why Rust is a genuinely good fit for AI inference

This isn't just ideology — there's a practical case, backed by the wider
industry's move toward Rust for the *inference* (model-running) side of AI:

- **Speed without a GPU.** Rust compiles to native machine code, so CPU-only
  inference is as fast as the hardware allows. In a 2025 benchmark, the Rust
  `Burn` framework hit ~97% of PyTorch's GPU performance with lower memory
  overhead and no garbage-collection pauses.[^burn]
- **True parallelism.** Python has a notorious "Global Interpreter Lock" (GIL)
  that hampers using many CPU cores at once. Rust has no such lock, so it can
  safely spread work across every core — which is exactly how Ferrum speeds up
  its maths (see [the engine doc](03-the-ferrum-engine-and-its-capabilities.md)).[^gil]
- **Ideal for the edge.** Rust's efficiency and safety make it "particularly
  well-suited for edge computing and resource-constrained environments."[^edge]
  There is even a published Rust inference engine for microcontrollers,
  *MicroFlow*, demonstrating the language reaching all the way down to tiny
  chips.[^microflow]
- **Production-grade serving.** Rust's reliability for "long-running services"
  is why real inference servers (e.g. `mistral.rs`, built on the `Candle`
  framework) are written in it.[^mistral]

Ferrum is a small, educational member of this family — but it sits squarely in
the same trend: **Rust for safe, fast, dependency-light model running.**

[^burn]: "In a 2025 benchmark on the Phi3 model, Burn+CUDA achieved 97% of PyTorch+CUDA's performance with lower memory overhead and no runtime garbage collection." Markaicode, *Rust for ML* https://markaicode.com/rust-ml-Building-high-performance-inference-engines-2025/
[^gil]: "Rust allows for true, safe parallelism across multiple CPU cores without the performance limitations of Python's Global Interpreter Lock (GIL)." Medium (S. Swain), *Rust: The Performance Edge for LLM Inference* https://medium.com/@soumyajit.swain/rust-the-performance-edge-for-large-language-model-inference-59528a66ec68
[^edge]: Markaicode, *Rust for ML: Building High-Performance Inference Engines in 2025* https://markaicode.com/rust-ml-Building-high-performance-inference-engines-2025/
[^microflow]: Carnelos et al., *MicroFlow: An Efficient Rust-Based Inference Engine for TinyML* (arXiv 2409.19432) https://arxiv.org/abs/2409.19432
[^mistral]: "Mistral-rs is a high-performance inference engine built upon the Candle machine learning framework, ensuring a pure Rust implementation with minimal dependencies." (industry summary) https://markaicode.com/rust-ml-Building-high-performance-inference-engines-2025/

---

## 2.5 One language, many destinations: WebAssembly

Here's a payoff that surprises newcomers. Because Ferrum is pure Rust, it can be
compiled to **WebAssembly (WASM)** — a portable format that runs inside any
modern **web browser** at near-native speed.

The practical magic: a model you trained on your laptop can be put on a plain web
page and run **entirely in the visitor's browser** — no server, no API key, no
per-request cost, and the visitor's data never leaves their machine. The *same*
Rust code becomes a command-line tool, a desktop app, *and* a web page. (Ferrum's
`tabular_wasm` crate provides these browser bindings.)

This "write once, run on server / laptop / Pi / browser" portability is a direct
consequence of choosing Rust with no dependencies.

---

## 2.6 The honest trade-offs of choosing Rust

To keep this manual trustworthy, the costs deserve mention too:

- **Rust is harder to learn than Python.** The borrow checker that prevents bugs
  also frustrates beginners. (Good news: *using* Ferrum requires zero Rust.)
- **The "no dependencies" rule means re-inventing wheels.** Ferrum writes its own
  matrix maths instead of using a hyper-optimised library like Intel's MKL or
  OpenBLAS. That keeps it pure and readable but means it is **not the fastest
  possible** CPU engine — battle-tested numeric libraries would beat it on raw
  speed. Ferrum trades a bit of peak performance for total transparency and
  portability. (This trade is examined further in [`07-critique.md`](07-critique.md).)
- **The AI ecosystem still lives in Python.** Most pretrained models, datasets,
  and tutorials assume Python. Rust is growing fast for inference, but you'll
  find fewer ready-made resources.

These are deliberate, reasonable trade-offs for a teaching-and-edge engine — but
they are real, and you should know them.

---

## 2.7 What you now know

- **Rust** gives C-level speed with compiler-enforced memory safety and no
  garbage-collector pauses.
- Ferrum leans on three Rust strengths: **safety** (`#![forbid(unsafe_code)]`),
  **no garbage collector**, and **zero outside dependencies** (`std`-only).
- Rust is a genuinely good fit for **CPU inference at the edge**, matching a
  broader industry trend.
- The cost is a steeper learning curve and giving up the very fastest hand-tuned
  numeric libraries — a deliberate trade for transparency and portability.

Next: the full tour of [**the Ferrum engine itself** — what it's made of, what it
can do, and where it falls short](03-the-ferrum-engine-and-its-capabilities.md).

---

## Sources

- Markaicode, *Rust for ML: Building High-Performance Inference Engines in 2025* — https://markaicode.com/rust-ml-Building-high-performance-inference-engines-2025/
- S. Swain, *Rust: The Performance Edge for LLM Inference* — https://medium.com/@soumyajit.swain/rust-the-performance-edge-for-large-language-model-inference-59528a66ec68
- Carnelos et al., *MicroFlow: An Efficient Rust-Based Inference Engine for TinyML* (arXiv 2409.19432) — https://arxiv.org/abs/2409.19432
- Rust Foundation, *Rust and AI: Position Statement* — https://rustfoundation.org/resource/rust-and-ai-position-statement/
