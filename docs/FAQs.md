# Ferrum FAQs

Honest answers to common questions about scope, constraints, and design.

### What is Ferrum?

A zero-dependency, pure-Rust engine for building, training, and running small
causal Transformers, Small Language Models, and classical MLPs on the CPU — and
for importing and running small open-weight **Llama/Qwen GGUF** checkpoints. It is
`std`-only and `#![forbid(unsafe_code)]`, and it compiles to native binaries and
to WebAssembly.

### Does it need a GPU?

No — and it never uses one. Everything runs on the CPU, parallelized across all
available cores.

### How does it use multiple cores without extra dependencies?

The matmul kernels (the dominant cost of every Linear, FFN, attention, and
LM-head step) split their output rows across a **persistent pool** of worker
threads — spawned once and reused, so autoregressive generation does not pay
per-call thread-creation cost. It is built only on `std` (threads, channels,
`Arc`) with no `unsafe`: kernels share read-only inputs through `Arc` and each
worker returns an owned output block. For single-token decode (`m = 1`, which a
row-split cannot parallelize) the quantized path splits the GEMV across cores **by
output column** instead. Thread count comes from
`std::thread::available_parallelism()` (override `FERRUM_NUM_THREADS`); small
workloads and `wasm32` run serially. Results are bit-for-bit identical regardless
of thread count, so training and inference stay deterministic.

### What external dependencies does it have?

The engine (`ferrum_core`) and both CLIs have **none**. Only `tabular_wasm`
depends on `wasm-bindgen`, and the separate `ferrum_gui` app depends on Tauri.

### What is the byte-level BPE tokenizer, and why does it matter?

`ByteBpeTokenizer` learns subword merges over the 256 base byte values. Because
the base vocabulary is the full byte range, any UTF-8 text — emoji, Cyrillic, CJK,
control characters — round-trips with no unknown-token escape hatch. Subword
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

### How big are the models, and can I go smaller than int8?

Int8-quantized models are typically tens of kilobytes (~4× smaller than f32). You
can also serialize **int4** (`to_bytes_quantized_int4`, ≈8× smaller); its grid is
coarser, so expect a little more drift. Only tensors of ≥64 values are quantized —
biases and LayerNorm parameters stay f32.

### Where does the tokenizer live after training?

Inside the model file. Its merge list is serialized into the FINF metadata
(`tokenizer_state`), so a loaded model tokenizes and generates exactly as before
saving. No separate vocabulary file is needed.

### Will my old models still load?

Yes. Files written before the tokenizer field load unchanged and default to
character-level tokenization, and v4 (f32) files load alongside v5 (int8/int4).

### Can Ferrum run models I downloaded (e.g. from Hugging Face)?

Small **Llama/Qwen** checkpoints in GGUF, yes — via `run-gguf` or the `Gguf` API.
It reads `F32/F16/Q8_0/Q8_1/Q4_0/Q4_1` and the `Q4_K/Q5_K/Q6_K` k-quants and
imports the checkpoint's own tokenizer. `Q2_K`/`Q3_K`/`IQ*` and non-Llama
architectures are rejected. Import is lossy (it re-quantizes to Ferrum's per-row
grid) and not bit-exact to llama.cpp, and a ~1B model decodes at only a few
tokens/sec on a CPU. Treat it as a private, offline runner for *small* open
models, not a llama.cpp replacement.

### Can I train or fine-tune an imported GGUF model?

The architecture is **trainable**: `LlamaTrainer` adds a finite-difference-checked
backward pass and an SGD step. Import at f32 first (`load_llama_prec(None)`), since
training needs f32 master weights. This works for *small* models; a 1B model is
out of reach on a single CPU — the optimizer state alone is ~16 bytes/param
(~16 GB for 1B), before any compute considerations.

### Can Ferrum write GGUF files, or only read them?

Both. `export-gguf` (CLI) / `LlamaModel::write_gguf` (library) serialize a
loaded — and optionally fine-tuned — llama/qwen2 model back to a GGUF v3 file
that runs in llama.cpp / ollama / LM Studio, at any of
`f32/f16/q8_0/q8_1/q4_0/q4_1/q4_k/q5_k/q6_k`. The source file's hyperparameters
and tokenizer are carried forward verbatim, so the output is self-contained.
Typical uses: re-quantizing a download (e.g. `Q4_K` → `Q8_0`) and sharing a
fine-tune as a standard GGUF.

### What are `--weight_decay` and `--dropout`?

Regularizers for training. `--weight_decay` enables **AdamW** decoupled weight
decay; `--dropout` applies dropout to the FFN hidden activations. Both default to
0 (plain Adam, no dropout). Use them when a small corpus is being memorized.

### Does `--chars` count characters or tokens for BPE models?

Characters. Generation runs over subword tokens internally, but the output is the
seed plus exactly `--chars` new characters (unless cut short).

### Why did training fail with "corpus must be longer than the context window"?

The tokenized corpus had too few tokens. BPE compresses text, so a short or highly
repetitive corpus can fall below `context_len + 1` tokens even with many
characters. Use a longer/more varied corpus, a smaller `--context`, or a smaller
`--vocab`.

### Is it deterministic?

Yes — training, BPE merge learning, and generation are all deterministic for a
fixed seed and configuration, and identical across thread counts.

### What can it not do?

It is not a drop-in replacement for large GPU-trained LLMs. It targets small,
self-contained, CPU-friendly models for edge, embedded, browser, and offline use,
and it can run *small* open-weight models slowly. It will not train a large model,
serve a frontier LLM interactively, or reason reliably — see the
[critique](../manual/07-critique.md).
