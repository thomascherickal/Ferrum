# 🧬 Ferrum Developer User Guide

This guide is designed to help developers understand the core design, constraints, and architecture of **Ferrum** as a zero-dependency Edge AI library. 

---

## 1. Zero-Dependency & Clean-Room Philosophy

Ferrum is built strictly with Rust’s standard library (`std`). It uses:
- **No external BLAS/LAPACK** for matrix operations.
- **No PyTorch/ndarray bindings**.
- **No OS-level bindings** in the core layer (allowing compilation to target environments like WebAssembly `wasm32-unknown-unknown` where libc, threads, or filesystem elements are absent).

This design ensures that the entire code stack is:
- **Fully Auditable**: Every mathematical kernel is hand-crafted and contained within a few files.
- **Ultra-Lightweight**: Binary models load instantly and consume minimal memory, unlike massive neural network runtimes.
- **Off-Grid Capable**: Perfect for systems where internet connections, API keys, or bulky LLM weights are unavailable or unacceptable.

---

## 2. Hand-Crafted MLP vs. Causal Transformer

`ferrum_core` supports two distinct ML architectural formats:
1. **Feedforward MLP (Multi-Layer Perceptron)**:
   - High training efficiency and fast convergence on tabular and next-character sequence predicting tasks.
   - Core of our standard training pipeline (DenseT layers, ReLU, Backpropagation).
2. **Causal Transformer Block (Inference Only)**:
   - Contains token + positional `Embedding`, `LayerNorm`, and `TransformerBlock` layers.
   - Implements **Decoder-Only Causal Multi-Head Self-Attention** with future-attention masking (`-1e9` for keys at indices larger than query indices).
   - Allows loading and executing small language models directly in the browser via WebAssembly, with full access to live attention weights for visualizations.

---

## 3. The Hex-Encoded Vocabulary Strategy

When building generative Small Language Models (SLMs) from custom text datasets, using traditional comma-separated token representations breaks CSV parser integrity (special characters like commas `,`, quotes `"`, or newlines `\n` clash with file structures).

To solve this, Ferrum implements a **Hex-Encoded Vocabulary** strategy:
1. **Hexadecimal Translation**: Every character in the generative training text corpus is converted to its unique hexadecimal string representation (e.g. `'a'` -> `"61"`, `' '` -> `"20"`, `'\n'` -> `"0a"`).
2. **Safe CSV Generation**: The training target inputs and labels are stored purely as clean hexadecimal alphanumeric strings that never break CSV formats.
3. **Baked-In Metadata**: The unique vocabulary strings are baked directly into the `class_names` metadata array of the FINF v4 model binary file.
4. **Self-Containment**: When the model is reloaded, it parses the vocabulary indices directly from the metadata back to their character representations, rendering the resulting `.bin` file **100% self-contained**!

---

## 4. Key Performance Guidelines

To maintain responsive, crash-free execution, verify your generative setups align with these guidelines:
- **Vocabulary Size**: Keep the character vocabulary small ($\le 80$ characters). Larger vocabularies bloat the linear projection dimensions.
- **Context Window**: A sliding context window of $4 - 6$ characters is highly optimal. Larger contexts require wider linear inputs.
- **Corpus Dimension**: Keep raw text corpora small ($\le 50$ KB) to ensure training converges cleanly on the CPU in under a minute.
