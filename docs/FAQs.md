# 🧬 Ferrum Edge AI FAQs

Clear, honest answers to common technical questions, constraints, and architecture considerations when using the Ferrum library.

---

## 1. What are the context length constraints of this model?

Because Ferrum trains next-character prediction models using standard MLPs (Multi-Layer Perceptrons) in the core backpropagation trainer:
- **Optimal Context Length**: $4 - 6$ characters.
- **Why?**: The inputs are flat-concatenated. The input dimension of the first linear layer equals $N \times d$ (context length times embedding dimension). Increasing the context to 100+ characters with even a modest embedding dimension causes a parameter explosion in the first layer, slowing CPU training and causing model convergence to fail.
- **For larger contexts**: Use the causal Transformer path (`GenerativeSLM::train_transformer`), which trains end-to-end with Adam and uses compact token-ID inputs (`input_dim = context_len`), scaling much better for sequence context lengths of $16 - 128$ tokens.

---

## 2. Can I run Gemma, Llama, or GPT-4 weights with this library?

**No.**
- **Architectural Constraints**: Modern Large Language Models (LLMs) require billions of parameters, Rotary Positional Embeddings (RoPE), Multi-Query/Grouped-Query Attention (MQA/GQA), and custom KV-caches. Ferrum is optimized for ultra-lightweight, zero-dependency small models ($15,000 - 40,000$ parameters) running directly on the client's CPU thread.
- **Compute Constraints**: A billion-parameter model requires gigabytes of memory bandwidth. Running a massive model on standard single-threaded CPU WebAssembly will freeze the browser tab and trigger out-of-memory errors instantly.

---

## 3. Why is my generated text repeating or degenerating into character soup?

This is a physical limitation of small models and naive character-level sequence modeling. If your model gets stuck in repetitive loops:
- **Adjust the Temperature**: A temperature value that is too low ($T < 0.05$) makes predictions completely deterministic, leading the model to get stuck in repeating cycles. Increase the temperature to $0.15 - 0.30$ to add random variance.
- **Verify Loss Convergence**: Make sure you trained the model for enough epochs. If training loss remains high ($> 2.0$), the model has not converged and will output random character sequences.

---

## 4. Does the WASM compiler support SIMD or multi-threading?

Currently, the WebAssembly bindings compile to `wasm32-unknown-unknown` without WASM-SIMD autovectorization or threads:
- **Single-Threaded**: This keeps execution completely robust and portable across all modern browsers (including legacy mobile browsers) without requiring HTTP headers for SharedArrayBuffer (Cross-Origin Opener Policy).
- **Fast Execution**: Because the models are tiny ($~25$ KB), a single-threaded CPU WASM pass takes less than **2 milliseconds**, making SIMD acceleration unnecessary.

---

## 5. How do I add support for a completely new language or special symbols?

The **Hex-Encoded Vocabulary** strategy handles all characters automatically:
- Simply include your symbols, emojis, or Cyrillic/Asian characters directly in the text corpus.
- The `GenerativeSLM` dataset builder translates every unique character to its matching hexadecimal string representation.
- The model will learn and predict these symbols cleanly, and the WASM/JS engine will decode the hex back to standard Unicode characters.
