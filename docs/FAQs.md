# Ferrum FAQs

Honest answers to common questions about scope, constraints, and design.

### What is Ferrum?

A zero-dependency, pure-Rust engine for building, training, and running small
causal Transformers, Small Language Models, and classical MLPs on the CPU. It is
`std`-only and `#![forbid(unsafe_code)]`, and it compiles to native binaries and
to WebAssembly.

### Does it need a GPU?

No — and it never uses one. Everything runs on the CPU, parallelized across all
available cores.

### How does it use multiple cores without extra dependencies?

The matmul kernels (the dominant cost of every Linear, FFN, attention, and
LM-head step) split their output rows across CPU threads using only
`std::thread::scope`. The thread count is detected dynamically from
`std::thread::available_parallelism()` and can be overridden with the
`FERRUM_NUM_THREADS` environment variable. Small workloads and the `wasm32`
target run serially. Results are bit-for-bit identical regardless of thread
count, so training and inference stay deterministic.

### What external dependencies does it have?

The engine (`ferrum_core`) and both CLIs have **none**. Only `tabular_wasm`
depends on `wasm-bindgen` for browser bindings.

### What is the byte-level BPE tokenizer, and why does it matter?

`ByteBpeTokenizer` learns subword merges over the 256 base byte values. Because
the base vocabulary is the full byte range, any UTF-8 text — emoji, Cyrillic,
CJK, control characters — round-trips with no unknown-token escape hatch. Subword
tokens also let a fixed context window cover more text than character tokens.

### How do I choose between character-level and BPE?

Set `vocab_size` (the `--vocab` flag): `0` is character-level, any value `>= 256`
trains a BPE tokenizer of that size. BPE usually wins on longer, varied,
multilingual text; character-level is simplest and most transparent. Values
between 1 and 255 are rejected because the byte base is irreducible.

### Is BPE compatible with quantization-aware training?

Yes. Tokenization only changes the token stream and vocabulary size; QAT operates
on the network weights and is unaffected. BPE models are trained int8-aware and
saved as int8 FINF v5 exactly like character-level models.

### Where does the tokenizer live after training?

Inside the model file. Its merge list is serialized into the FINF metadata
(`tokenizer_state`), so a loaded model tokenizes and generates exactly as it did
before saving. No separate vocabulary file is needed.

### Will my old models still load?

Yes. Files written before the tokenizer field load unchanged and default to
character-level tokenization.

### How big are the models?

Int8-quantized models are typically tens of kilobytes. QAT makes them roughly 4×
smaller than full-precision while behaving almost identically.

### Does `--chars` count characters or tokens for BPE models?

Characters. Generation runs over subword tokens internally but the output is the
seed plus exactly `--chars` new characters (unless cut short).

### Why did training fail with "corpus must be longer than the context window"?

The tokenized corpus had too few tokens. BPE compresses text, so a short or
highly repetitive corpus can fall below `context_len + 1` tokens even with many
characters. Use a longer/more varied corpus, a smaller `--context`, or a smaller
`--vocab`.

### Is it deterministic?

Yes — training, BPE merge learning, and generation are all deterministic for a
fixed seed and configuration.

### What can it not do?

It is not a drop-in replacement for large GPU-trained LLMs. It targets small,
self-contained, CPU-friendly models for edge, embedded, browser, and offline use.
