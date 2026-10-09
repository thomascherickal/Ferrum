# Manual: new and rewritten sections (draft)

Each section below names the Manual section it replaces or the place it is
inserted. Section numbers follow the current `Manual.md`. New sub-sections get
numbers in the order shown. Cross-references in `ReadMe.md` use these numbers.

---

<!-- REPLACES: Manual §2 "Why Rust, and what zero dependencies buys" (heading, §2.2 third bullet and table, §2.5). Keep §2.1, §2.3, §2.4. -->

## 2. Why Rust, and what few dependencies buy

### 2.2 The three guarantees

- **Memory safety without a garbage collector.** There are no surprise pauses
  while tokens are streaming.
- **`#![forbid(unsafe_code)]` in every crate.** The compiler rejects the code if
  it uses Rust's `unsafe` escape hatch. A repo-wide test scans the sources for
  `unsafe` as well. Where a third-party proc macro forces it, the crate uses
  `deny` instead of `forbid`. Dependencies may use `unsafe` inside themselves;
  Ferrum calls only their safe APIs. This rules out crate functions that are
  `unsafe fn` (for example `matrixmultiply::sgemm` and the `gemm` crate when
  called directly). Ferrum's crates forbid `unsafe` code
  (`#![forbid(unsafe_code)]`); third-party crates may use it internally.
- **Dependencies need a measured reason.** Earlier versions had an empty
  `[dependencies]` table. That rule is dropped. A crate is admitted only when:
  1. a benchmark shows a gain (or the crate replaces more code than it adds),
  2. results stay bit-for-bit identical at 1 thread and at N threads (a test
     enforces this), and
  3. `ferrum_core` and `slm_wasm` still build and run on
     `wasm32-unknown-unknown`.

  A crate that fails any of the three is rejected. `deny.toml` lists the
  allowed licences, and `cargo deny check` runs in CI.

### 2.2.1 What was kept, and why

| Crate | Used for | Why it is kept |
|---|---|---|
| `rayon` | Worker pool behind the matmul row split (not on wasm32) | Replaces the hand-written pool. Kept only if no slower than it. <!-- MEASURE: rayon vs hand-written pool, train tokens/s and generate ms/token --> |
| A matmul crate used through a safe API (candidates: `ndarray` with its `matrixmultiply` backend, and `faer`) | `ops::matmul`, `qlinear` | Kept only if at least 10% faster on train tokens/s than the current kernel, and deterministic and wasm32-clean. <!-- MEASURE: which candidate won, or "none beat the current kernel", with GFLOP/s from `cargo bench --bench gemm` --> |
| `sha2` | sha256 in `prepare` manifests and the server's `model_sha256` | Hashes are correctness-critical; a hand-written one is more code to audit. |
| `serde_json` | `manifest.json`, `eval --json`, the server's JSON | Standard JSON. |
| `unicode-normalization` | `prepare --nfkc` | NFKC needs the Unicode tables. |
| `axum`, `tokio`, `serde` | `slm_server` only | The HTTP server. The engine does not depend on them. |
| `wasm-bindgen` | `slm_wasm` only | Browser bindings. |
| `tauri` | `ferrum_gui` only | Desktop GUI, outside the workspace. |

<!-- MEASURE: the final dependency list must be checked against each Cargo.toml before release -->

| Benefit | Why it matters |
|---|---|
| Small, readable engine | The forward and backward pass, the tokenizer, and the GGUF codecs are all in the repo |
| Few crates to vet | `cargo deny check` covers the whole tree; the engine crate has a short list |
| Runs anywhere Rust runs | Servers, laptops, Raspberry Pi, browsers |
| Reproducibility | A determinism test blocks any crate that changes results with the thread count |

### 2.5 The trade-offs, honestly

- Rust is harder to learn than Python. *Using* Ferrum requires no Rust.
- Admitting crates costs something: more code you did not write, a larger
  build, and a licence and advisory list to watch. The rule above keeps that
  cost to crates that pay for themselves.
- Ferrum does not call OpenBLAS or MKL, so it is **not the fastest possible CPU
  engine**. It trades peak speed for transparency, safety, and portability.
- The AI ecosystem still lives mostly in Python, so ready-made Rust resources
  are fewer. (Ferrum's LightEval module and DVC pipeline use Python tools from
  outside the engine.)

---

<!-- REPLACES: Manual §5 "Preparing a corpus" (keep the opening paragraph, the "Hold text back" paragraph, and the checklist; replace the list of three functions and add this) -->

## 5. Preparing a corpus

Any UTF-8 text file is a corpus. For a good one, use `prepare`. It is a
repeatable, streaming cleaner and splitter, and it records exactly what it did.

### 5.1 `train_transformer prepare`

```text
train_transformer prepare --input <file|dir> [--input ...] --out <dir>
    [--doc-sep S] [--lossy] [--nfkc] [--lowercase]
    [--strip-gutenberg] [--min-chars N] [--max-chars N] [--dedup]
    [--val-frac F] [--test-frac F] [--seed N]
```

| Flag | Default | Meaning |
|---|---|---|
| `--input PATH` | required, repeatable | A file or a directory. Directories are walked in sorted order. |
| `--out DIR` | required | Output directory. |
| `--doc-sep S` | `<|endoftext|>` | The string that separates documents in the input. |
| `--lossy` | off | Replace invalid UTF-8 bytes. Without it, invalid UTF-8 is an error that names the file and byte offset. |
| `--nfkc` | off | Unicode NFKC normalization. |
| `--lowercase` | off | Lowercase every document. |
| `--strip-gutenberg` | off | Remove the Project Gutenberg header and footer of each book. Handles several books in one file. |
| `--min-chars N` | 1 | Drop documents shorter than `N` characters. |
| `--max-chars N` | 0 (off) | Drop documents longer than `N` characters. |
| `--dedup` | off | Drop exact duplicate documents (128-bit content hash; only hashes are held in memory). |
| `--val-frac F` | 0.05 | Share of documents for the validation split. |
| `--test-frac F` | 0.05 | Share of documents for the test split. |
| `--seed N` | 42 | Seed for the split. |

What it does to each document, in order:

1. Decode UTF-8 (error, or replace with `--lossy`).
2. Strip the byte-order mark, zero-width characters (U+200B to U+200D, U+2060,
   U+FEFF), and control characters except newline and tab.
3. Apply NFKC if asked.
4. Normalize punctuation, then collapse runs of spaces and runs of three or
   more newlines.
5. Lowercase if asked.
6. Strip Gutenberg boilerplate if asked.
7. Apply the length filter, then the duplicate filter.

Each input is streamed, so a large file is not loaded whole.

**The split is by document, not by line.** A document goes to `val`, `test`, or
`train` by `H(seed, document) mod 10,000` against the fractions. Three
consequences: the same document always lands in the same split, no document
appears in two splits, and appending new documents later does not move the old
ones.

**Outputs.** `out/train.txt`, `out/val.txt`, `out/test.txt` (documents joined by
`\n<|endoftext|>\n`) and `out/manifest.json`.

### 5.2 The manifest

`manifest.json` makes a run auditable.

| Field | Content |
|---|---|
| `ferrum_version` | The version that ran `prepare` |
| `args` | The full argument list |
| `inputs` | For each input: `path`, `bytes`, `sha256` |
| `outputs` | For `train`, `val`, `test`: `path`, `bytes`, `sha256`, `documents`, `chars` |
| `rejected` | Counts by reason: `empty`, `too_short`, `too_long`, `duplicate`, `invalid_utf8` |
| `train_stats` | `corpus_stats` of the train split (`bytes`, `chars`, `lines`, `unique_chars`, `words`) |

If the hashes in two manifests match, the inputs and outputs are byte-identical.
DVC tracks `manifest.json` in git (not in the DVC cache), so a change in the
data shows up in a `git diff`.

### 5.3 Library helpers

The cleaning steps live in `ferrum_core::dataset` as small functions you can
combine: `strip_invisible`, `gutenberg_bodies`, `squeeze_spaces`,
`normalize_nfkc`, `clean_corpus`, `corpus_stats`, and `validate_for_training`.
`clean_corpus` is kept as a wrapper. **It lowercases by default**, which
`prepare` does not (add `--lowercase`). The GUI **Datasets** tab uses the same
functions.

<!-- MEASURE: confirm the public names of the dataset functions and that clean_corpus still lowercases by default, against the finished ferrum_core::dataset -->

### 5.4 The default corpus

The DVC pipeline uses the TinyStories V2 (GPT-4) validation file, about 22 MB.
Stories are already separated by `<|endoftext|>`.

| | |
|---|---|
| URL | `https://huggingface.co/datasets/roneneldan/TinyStories/resolve/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/TinyStoriesV2-GPT4-valid.txt` |
| Size | 22,502,601 bytes |
| sha256 | `6874bae9a4c1a4e7edcf0e53b86c17817e9cf881fc75ff2368da457b80c0585d` |
| Licence | CDLA-Sharing-1.0 (downloaded, never committed) |

`scripts/fetch.sh <url> <sha256> <out>` downloads with `curl` and fails if the
checksum does not match.

**Hold text back.** Keep a separate held-out file in the same style that the
model never trains on. `prepare` makes `test.txt` for this. Alternatively,
`--val F` on `train` splits off the last fraction automatically (§6.6).

---

<!-- REPLACES: Manual §6.2 "Command-line reference" — update the tables and add the sub-sections below. Keep the Architecture and Schedule tables. -->

## 6.2 Command-line reference (changes)

All subcommands use a strict parser. Each subcommand has an allow-list of flags.

| Mistake | What happens |
|---|---|
| Unknown flag | Error naming the subcommand and listing valid flags, for example `unknown flag --epochs for 'info'` |
| Value that does not parse (`--context 12x`) | Error naming the flag and the value; never a silent default |
| Flag with no value (`--temp` at the end) | `--temp needs a value` |
| `--resume` path that does not exist | Error; use `--checkpoint` to start a new run |

A flag may repeat where it is documented as repeatable (`--stop`), and a
negative number is read as a value (`--temp -1`).

**Model file name.** Model files are GGUF. Use `.gguf`: `model.gguf`.

**Optimization and output (new or changed rows)**

| Flag | Default | Meaning |
|---|---|---|
| `--quant f32\|int8\|int4` | `int8` | Precision of the saved model. `int8` is GGUF Q8_0, `int4` is Q4_0. |
| `--tie` | off | Tie the LM head to the token embedding. The parameter count drops by `vocab × embed`, and the file shrinks by the same amount, because a tied model has no `output.weight` tensor. |
| `--no-qat` | off | Plain f32 training. **QAT is int8 only** (Q8_0). There is no int4 QAT; `--quant int4` converts after training. |
| `--tokenizer PATH` | none | Reuse a saved BPE tokenizer (a GGUF file); trained and saved there if missing. |
| `--val F` | none | Hold out the last fraction `F`. When a tokenizer is trained, it is trained on the train portion only. |
| `--verbose`, `-v` | off | Prints a progress line every 10 optimizer steps (`[train] step S/T tokens/s R elapsed Xs`). It no longer prints per-op logs. |
| `--debug` | off | `train` and `run` only. Streams the engine's internal trace. Very verbose; use it to debug, not to watch progress. |

<!-- MEASURE: confirm -v progress interval (10 steps) and line format in the finished CLI -->

With `--tokens N` set, the banner shows `Epochs  : budget` instead of an epoch
count. A resumed run reports the correct epoch and step, and with `--val` it
restores the best validation score from the checkpoint.

**`prepare`** — see §5.1.

**`generate` and `run` (new flags)**

| Flag | Default | Meaning |
|---|---|---|
| `--top-k N` | 0 (off) | Sample only from the `N` most likely tokens |
| `--top-p F` | 1.0 (off) | Nucleus sampling |
| `--repetition-penalty F` | 1.0 (off) | Penalize tokens already generated |
| `--max-tokens N` | 64 | Token budget for the continuation |
| `--stop S` | none | Stop when `S` appears. Repeatable. The text excludes the stop string. |

<!-- MEASURE: defaults of --top-k, --top-p, --repetition-penalty in the finished CLI -->

**`eval`**

```text
train_transformer eval <model.gguf> <text.txt> [--json <path>]
```

`--json` writes `{"perplexity", "cross_entropy_nats", "bits_per_token", "bits_per_byte", "tokens", "bytes", "seconds"}`.
DVC tracks this file as a metric.

**Export commands**

```text
train_transformer export-ollama   <model.gguf> <name> [--bin P] [--modelfile-only out.Modelfile]
train_transformer export-lmstudio <model.gguf> [--repo ferrum/<stem>] [--bin P]
```

See §9.6.

---

<!-- REPLACES: Manual §7 "Generating text" — replace the command block, flag table, and the KV-cache bullet; keep the rest -->

## 7. Generating text

```text
train_transformer generate <model.gguf> <seed text> [--chars N] [--max-tokens N] [--temp F] [--gen-seed N]
                           [--top-k N] [--top-p F] [--repetition-penalty F] [--stop S]... [--stream]
```

| Flag | Default | Meaning |
|---|---|---|
| `--chars N` | 200 | Characters to generate (characters even for BPE models) |
| `--max-tokens N` | 64 | Token budget instead of a character count |
| `--temp F` | 0.8 | Sampling temperature. **`0` is exact argmax (greedy).** 0.1–0.3 for faithful, 0.7+ for varied. |
| `--gen-seed N` | time-based | Fix it for reproducible output |
| `--top-k`, `--top-p`, `--repetition-penalty` | see §6.2 | Sampling controls |
| `--stop S` | none | Repeatable. The output ends before the first match. |
| `--stream` | off | Print the completion live |

- Temperature `0` is exact argmax. The old behaviour clamped `0` to `0.01`;
  that clamp is gone for `0`.
- If the tokenizer has an end-of-text token (every newly trained model does),
  generation stops when the model emits it.
- With the same `--gen-seed`, the output is identical at any thread count.

### 7.1 Generation cost

A per-block **KV cache** means each new token attends to the cached keys and
values instead of recomputing the window. Learned absolute positions forbid
sliding the cache one token at a time. When the cache is full, Ferrum
**re-primes** from the last `context_len / 2` tokens. That is one re-prime per
`context_len / 2` generated tokens. The cost per generated token is therefore
bounded, and generating many times the window length takes linear time. Output
text is decoded incrementally, not re-decoded in full on every step.

The library adds `generate_tokens`, which takes a token budget and a list of
stop strings and returns the continuation only:

```rust
use ferrum_core::{Rng, SamplingParams};

let p = SamplingParams { temperature: 0.7, top_k: 40, top_p: 0.9, repetition_penalty: 1.1 };
let g = slm.generate_tokens("Once upon a time", 100, &["\n\n".to_string()], &p, &mut Rng::new(1))?;
println!("{} ({} tokens, {:?})", g.text, g.n_tokens, g.stop_reason);
```

The older `generate`, `generate_with`, `generate_until`, and `generate_stream`
are unchanged.

---

<!-- REPLACES: Manual §8.1 "Held-out perplexity" bullets and adds §8.3 (keep the rest of §8) -->

## 8. Evaluating a model

### 8.1 Held-out perplexity: the honesty check

```bash
train_transformer eval model.gguf heldout.txt --json metrics/heldout.json
```

```rust
let e = slm.evaluate(&std::fs::read_to_string("heldout.txt")?)?;
println!("ppl {:.3}  bits/token {:.3}  CE {:.3} nats  ({} predictions)",
         e.perplexity, e.bits_per_token, e.cross_entropy, e.num_predictions);
```

- **Every token is scored.** The first token is conditioned on a start
  position (BOS, which is the EOS id), so `num_predictions` equals the token
  count. Earlier
  versions skipped the first window, so perplexities from before this change are
  not comparable (Status §7).
- Texts longer than the context window use sliding windows with stride
  `context_len / 2`. Each token is scored exactly once, with at least
  `context_len / 2` tokens of history (except in the first window).
- Lower is better, and **1.0 is perfect**. A model that learned nothing scores
  about the vocabulary size.
- Each token's probability is floored at 1e-12, which is a per-token floor of
  −27.63 nats.
- A large train-versus-held-out gap means **memorization**, which points to a
  data problem rather than a model that is too small.
- Evaluation uses the shipped weights and the same forward path as generation.
- For a character-level model, an unknown character is an error in the scoring
  API, and `evaluate()` reports how many it met.
- To compare models with **different tokenizers**, compare **bits per byte**.
  Per-token numbers are not comparable across vocabularies.

**Speed.** Evaluation runs one padded forward pass per window, which is much faster
than the old per-window loop.
<!-- MEASURE: eval seconds before/after on the reference model and corpus (Samples §9) -->

### 8.3 The scoring API

```rust
let s = slm.score("The cat sat on the", " mat")?;   // log p(continuation | context)
let r = slm.score_rolling(&text)?;                   // every token, the first conditioned on BOS
// Score { logprob: f64, n_tokens: usize, greedy: bool }
```

`score` follows the lm-eval-harness convention. Trailing whitespace in the
context moves to the front of the continuation. The continuation tokens are the
suffix of the joint encoding after the context's length. An empty continuation
scores `0.0` over `0` tokens. `greedy` is true when every scored token was the
argmax. The server and LightEval use this API.

---

<!-- INSERTS AFTER: Manual §9.4 "Library API". New sections 9.5–9.8; "9. External models (GGUF)" is retitled "GGUF: Ferrum models, other runtimes, serving, and external models". The Ferrum GGUF format below also REPLACES §12.7 "The FINF model format". -->

## 9.5 Running Ferrum models in llama.cpp, Ollama, and LM Studio

A Ferrum BPE transformer SLM is a `gpt2`-architecture GGUF. Other runtimes load
it without changes. The same f32 file gives the same greedy continuation in
Ferrum and in llama.cpp; the test `llama_cpp_greedy_matches_ferrum_f32` checks
this. Verified so far:

| Tool | Version | Verified how |
|---|---|---|
| llama.cpp | b11521 | `llama-completion`, `llama-tokenize` |
| Ollama | v0.40.2 (bundled llama-server) | `ollama create`, `/api/generate` |
| LM Studio | 0.4.18 (llama.cpp-linux-x86_64-avx2 2.23.1) | `lms import`, `lms load`, `/v1/completions` |

A stock GPT-2 124M Q8_0 GGUF gave identical 16-token greedy output in all three
tools, which fixed the method.

**llama.cpp**

```bash
llama-completion -m models/slm.gguf -p "Once upon a time" -n 16 --temp 0 -no-cnv --no-warmup </dev/null 2>/dev/null
llama-tokenize   -m models/slm.gguf -p "Once upon a time" --ids --log-disable
```

**Ollama**

```bash
OLLAMA_HOST=127.0.0.1:11545 OLLAMA_MODELS=./ollama-models ollama serve &
printf 'FROM "%s"\nTEMPLATE "{{ .Prompt }}"\nPARAMETER num_predict 16\n' "$PWD/models/slm.gguf" > Modelfile
OLLAMA_HOST=127.0.0.1:11545 OLLAMA_MODELS=./ollama-models ollama create ferrum-tiny -f Modelfile
curl -s http://127.0.0.1:11545/api/generate \
  -d '{"model":"ferrum-tiny","prompt":"Once upon a time","raw":true,"stream":false,"options":{"temperature":0,"num_predict":16}}'
```

Use `raw: true`. A Ferrum model has no chat template.

**LM Studio**

```bash
lms import -y -l --user-repo ferrum-test/slm models/slm.gguf
lms ls --json                       # find the entry whose path matches; its modelKey is KEY
lms load KEY --gpu off --identifier ferrum-tiny -y
lms server start -p 1234
curl -s http://127.0.0.1:1234/v1/completions \
  -d '{"model":"ferrum-tiny","prompt":"Once upon a time","max_tokens":17,"temperature":0,"repeat_penalty":1.0}'
```

Two LM Studio details. `max_tokens` N returns N−1 visible tokens, so ask for 17
to see 16. And its default repeat penalty (1.1) changes the output, so set
`repeat_penalty` to 1.0 when you compare with Ferrum.

**What to expect**

- **f32:** greedy output equals Ferrum's.
- **Q8_0:** llama.cpp quantizes activations in its Q8_0 matmuls, which can flip
  near-ties late in a generation. The compatibility test compares Ferrum's
  output with llama.cpp's and asserts equality only over the first 8 tokens.
- **Q4_0:** the test only checks that llama.cpp loads the file and generates.
- **Prompts** are tokenized the same way as in Ferrum, because the file declares
  `tokenizer.ggml.pre = "gpt-2"` and Ferrum uses the same pre-tokenizer.

**Finding the tools.** Ferrum looks for each tool in this order: the `--bin`
flag (export commands), the environment variable (`FERRUM_LLAMA_BIN` for the
llama.cpp directory, `FERRUM_OLLAMA`, `FERRUM_LMS`), `PATH`, then the default
install location (`~/.local/opt/llama.cpp-b11521`, `~/.local/bin/ollama`,
`~/.lmstudio/bin/lms`). If none has it, the error lists all four places.
Compatibility tests skip a missing tool, but fail instead when
`FERRUM_REQUIRE_RUNTIMES=1` is set.

**What does not run there.** Character-level models (no BPE vocabulary) and
Ferrum-ML tabular MLPs are valid GGUF files, but llama.cpp has no architecture
for them. Use `train_transformer` or the Ferrum library for those.

## 9.6 One-command exports

```text
train_transformer export-ollama   <model.gguf> <name> [--bin P] [--modelfile-only out.Modelfile]
train_transformer export-lmstudio <model.gguf> [--repo ferrum/<stem>] [--bin P]
```

| Command | What it does |
|---|---|
| `export-ollama` | Writes a Modelfile and runs `ollama create <name> -f Modelfile`. `--modelfile-only out.Modelfile` writes the Modelfile and stops. `--bin` names the `ollama` binary. |
| `export-lmstudio` | Runs `lms import -y -l` on the file. `-l` links the file instead of copying it. `--repo` sets the `publisher/name` (default `ferrum/<file stem>`). `--bin` names `lms`. |

The Modelfile contains:

```text
FROM "<absolute path to the GGUF>"
TEMPLATE "{{ .Prompt }}"
PARAMETER stop "<|endoftext|>"
PARAMETER temperature 0.8
PARAMETER num_ctx <context_length of the model>
```

Rules for both commands:

- A GGUF that llama.cpp cannot run (generic layout, character-level, tabular) is
  **refused before any tool runs**. The error says why.
- A missing tool gives an error that lists the four places searched (§9.5).
- If the Ollama server is not running, `ollama create` fails and the error says
  to start it with `ollama serve`.
- After export you can use `ollama run <name>` or load the model in LM Studio.

<!-- MEASURE: confirm the --repo default and the printed messages against the finished commands -->

## 9.7 Serving: `ferrum-serve`

```text
ferrum-serve --model <path.gguf> [--host 127.0.0.1] [--port 8080]
             [--max-batch 64] [--max-chars 65536] [--max-body 8388608]
```

The server binds to loopback by default. Unknown flags are errors. It holds one
loaded model on **one dedicated model thread** and handles **one request at a
time**, because `GenerativeSLM` is not `Send` or `Sync`. Each matmul still uses
every core.

| Route | Request | Response |
|---|---|---|
| `GET /health` | none | `ok` |
| `GET /v1/info` | none | `{params, context_len, vocab, quant, tokenizer, model_sha256}` |
| `POST /v1/loglikelihood` | `{"requests":[{"context","continuation"}]}` | `{"results":[{"logprob","n_tokens","greedy"}]}` |
| `POST /v1/loglikelihood_rolling` | `{"texts":[...]}` | `{"results":[{"logprob","n_tokens"}]}` |
| `POST /v1/generate` | `{"requests":[{"prompt","max_tokens","stop","temperature","top_k","top_p","seed"}]}` | `{"results":[{"text","n_tokens"}]}` |

`generate` defaults: `temperature` 0 (greedy), `max_tokens` 64, `seed` 0.

```bash
curl -s http://127.0.0.1:8080/v1/loglikelihood -d '{"requests":[{"context":"The cat sat on the","continuation":" mat"}]}'
```

**Limits.** They are checked before any compute.

| Limit | Default | Error |
|---|---|---|
| Body size (`--max-body`) | 8 MiB | `413` |
| Batch length (`--max-batch`) | 64 | `400` |
| Characters per text (`--max-chars`) | 65,536 | `400` |
| Malformed JSON or a missing field | | `400` |
| Model error (for example an unknown character in a character-level model) | | `422` |

Errors have the form `{"error": "..."}`. An empty `requests` array returns `200`
with `{"results": []}`. A model error does not stop the server.

There is no authentication and no TLS. Do not bind it to a public address. The
routes are Ferrum's own; there are no OpenAI-compatible routes.

## 9.8 Evaluating with LightEval

`eval/lighteval/` lets Hugging Face LightEval **0.13.0** (Python 3.10 or newer)
evaluate a Ferrum model through `ferrum-serve`.

```bash
python3 -m venv eval/lighteval/.venv
eval/lighteval/.venv/bin/pip install -r eval/lighteval/requirements.txt
eval/lighteval/run.sh --model models/slm.gguf --test data/prepared/test.txt \
    --out metrics/lighteval.json --max-samples 200
```

`run.sh` starts `ferrum-serve`, waits for `/health`, runs
`lighteval custom ferrum ferrum_model.py "<tasks>" --custom-tasks ferrum_tasks.py`,
stops the server, and writes a summary:
`{"tasks": {"<task>|0": {"<metric>": value}}, "lighteval_version": "0.13.0", "model": ..., "max_samples": ...}`.

| Option | Meaning |
|---|---|
| `--model`, `--test`, `--out` | Required: model file, test split, summary path |
| `--max-samples N` | Cap samples per task |
| `--wikitext` | Add `wikitext:103:document_level` |
| `--port P` | Server port |

**Tasks.**

| Task | What it measures | Metrics |
|---|---|---|
| `ferrum_heldout` | Rolling log-likelihood of the test documents | word and byte perplexity, bits per byte |
| `tinystories_cloze` | Multiple choice: the true next sentence versus 3 sentences from other stories (fixed seed) | `loglikelihood_acc` (reported as `acc`) |
| `blimp:anaphor_gender_agreement`, `blimp:anaphor_number_agreement`, `blimp:regular_plural_subject_verb_agreement_1` | Grammatical versus ungrammatical sentence pairs | accuracy |
| `arc:easy` | Science questions | accuracy |

A 500K-parameter model will score near chance on `arc:easy` (25% for four
choices). That is expected. <!-- MEASURE: results table from a real run -->

`FerrumModel` reads the server address from `FERRUM_URL` (default
`http://127.0.0.1:8080`) and uses only the Python standard library to talk to it.

**Two LightEval 0.13.0 workarounds.**

1. `FerrumModel` does not call `super().__init__()` (it raises in 0.13.0). It
   sets `self.config` and the sample cache directly.
2. `loglikelihood_rolling` is not cached, because 0.13.0's perplexity cache
   drops results. `requirements.txt` also pins `xxhash<4`, because xxhash 4.x
   crashes the details logger, and `datasets<5`.

LightEval's litellm and TGI backends cannot run log-likelihood tasks, which is
why Ferrum plugs in as a custom model class.

In DVC, the `lighteval` stage is frozen by default. Run
`dvc unfreeze lighteval && dvc repro lighteval` to include it.

## 9.9 DVC pipeline

`dvc repro` reproduces data, model, and metrics from `params.yaml`.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dataops.txt
.venv/bin/dvc repro
.venv/bin/dvc dag
.venv/bin/dvc metrics show
```

| Stage | Command (from `dvc.yaml`) | Outputs |
|---|---|---|
| `fetch` | `scripts/fetch.sh <url> <sha256> data/raw/tinystories.txt` | `data/raw/tinystories.txt` |
| `prepare` | `train_transformer prepare ... --dedup --nfkc` | `data/prepared/{train,val,test}.txt`, `manifest.json` |
| `train` | `train_transformer train data/prepared/train.txt models/slm.gguf ... --force` | `models/slm.gguf`, `models/tokenizer.gguf` |
| `eval` | `train_transformer eval ... --json` on `val.txt` and `test.txt` | `metrics/val.json`, `metrics/test.json` |
| `lighteval` (frozen) | `eval/lighteval/run.sh ...` | `metrics/lighteval.json` |

`params.yaml` holds the knobs:

| Section | Keys |
|---|---|
| `fetch` | `url`, `sha256` |
| `prepare` | `min_chars`, `max_chars`, `val_frac`, `test_frac`, `seed` |
| `model` | `context`, `embed`, `heads`, `blocks`, `hidden`, `vocab` (default: the 507,552-parameter reference) |
| `train` | `tokens`, `lr`, `warmup`, `clip`, `val`, `patience`, `quant`, `seed` |

Edit a value, run `dvc repro`, then `dvc params diff` and `dvc metrics diff` to
see what changed. Stages whose inputs did not change are skipped.

Notes:

- `data/` and `models/` are git-ignored and tracked by DVC. `dvc.lock` is
  committed. `.dvc/` is initialised with **no remote**; a remote is out of scope.
- Training is deterministic, so `dvc repro --force train` gives the same
  `models/slm.gguf` sha256.
- `train` always reads the whole prepared train split into memory.
- Metrics files use `cache: false`, so they live in git.

---

<!-- REPLACES: Manual §12.7 "The FINF model format". Also update §11.5 "Versioning and compatibility", §12.5 (tokenizer file is GGUF), and every ".bin" in the Manual to ".gguf". -->

### 12.7 The Ferrum GGUF format

Every file Ferrum writes is GGUF v3. There are four kinds.

| Kind | `general.architecture` | Runs in llama.cpp |
|---|---|---|
| BPE transformer SLM (what `train` writes) | `gpt2` | Yes |
| Generic Ferrum model (character-level SLMs, Ferrum-ML tabular MLPs) | `ferrum` | No |
| Training checkpoint | `ferrum-checkpoint` (`ferrum-llama-checkpoint` for Llama fine-tunes) | No |
| Reusable tokenizer file | `ferrum-tokenizer` | No |

#### The GPT-2 layout

Ferrum's own transformer matches GPT-2: learned absolute positions, pre-LayerNorm
(epsilon 1e-5, with bias), dense multi-head attention with biases, a **GELU**
feed-forward block (the tanh approximation llama.cpp uses as `ggml_gelu`), and
an LM head **without a bias**.

Metadata that llama.cpp reads:

| Key | Type | Value |
|---|---|---|
| `general.architecture` | string | `gpt2` |
| `gpt2.context_length`, `gpt2.embedding_length`, `gpt2.block_count`, `gpt2.feed_forward_length`, `gpt2.attention.head_count` | u32 | model dimensions |
| `gpt2.attention.layer_norm_epsilon` | f32 | 1e-5 |
| `tokenizer.ggml.model` | string | `gpt2` |
| `tokenizer.ggml.pre` | string | `gpt-2` |
| `tokenizer.ggml.tokens`, `.merges`, `.token_type` | arrays | the vocabulary in GPT-2 bytes-to-unicode form |
| `tokenizer.ggml.bos_token_id`, `.eos_token_id` | u32 | both the id of `<|endoftext|>` |
| `tokenizer.ggml.add_bos_token`, `.add_eos_token` | bool | false |
| `general.file_type` | u32 | 0 all F32, 7 mostly Q8_0, 2 mostly Q4_0 |
| `general.quantization_version` | u32 | 2 |

Tensors (GGML dims `[ne0, ne1]`):

| Tensor | Dims |
|---|---|
| `token_embd.weight` | `[n_embd, n_vocab]` |
| `position_embd.weight` | `[n_embd, n_ctx]` |
| `blk.N.attn_norm.{weight,bias}` | `[n_embd]` |
| `blk.N.attn_qkv.{weight,bias}` | `[n_embd, 3·n_embd]` and `[3·n_embd]`, ordered Q, K, V |
| `blk.N.attn_output.{weight,bias}` | `[n_embd, n_embd]` and `[n_embd]` |
| `blk.N.ffn_norm.{weight,bias}` | `[n_embd]` |
| `blk.N.ffn_up.{weight,bias}` | `[n_embd, n_ff]` and `[n_ff]` |
| `blk.N.ffn_down.{weight,bias}` | `[n_ff, n_embd]` and `[n_embd]` |
| `output_norm.{weight,bias}` | `[n_embd]` |
| `output.weight` | `[n_embd, n_vocab]`; **omitted when the head is tied** (llama.cpp falls back to `token_embd`) |

Ferrum stores weights as row-major `[in, out]`, so they are written transposed.
This is why `--tie` shrinks the file as well as the parameter count.

#### Ferrum extras

Keys in the `ferrum.` namespace are ignored by llama.cpp.

| Key | Type | Meaning |
|---|---|---|
| `ferrum.format` | u32 | 1 |
| `ferrum.quant` | string | `f32`, `int8`, or `int4` |
| `ferrum.metadata` | string | `ModelMetadata` JSON: task, class names, dimensions |
| `ferrum.normalizer` | string | Normalizer state (empty for SLMs) |

`train_transformer info` reads these. A GPT-2 GGUF that Ferrum did not write
also loads, with its metadata rebuilt from the `gpt2.*` keys.

#### Quantization rules

| Type | Used for |
|---|---|
| F32 | `quant = f32`; all 1-D tensors (biases, LayerNorm); `position_embd`; and any matrix whose row length (`ne0`) is not a multiple of 32 (stored exactly, never as F16) |
| Q8_0 | `int8`: blocks of 32 values along each weight row, one f16 scale per block |
| Q4_0 | `int4`: same block size, 4 bits per value; lossy |

- Q8_0 and Q4_0 need the row length (`ne0`, the input width) to be a multiple of
  32. Keep `--embed`, `--hidden`, and `--vocab` multiples of 32 so every matrix
  quantizes. The 507,552-parameter reference is built that way.
- `position_embd` stays F32 when quantizing, following llama-quantize.
- Because ineligible tensors are F32, an int8 file reloads bit-identically for
  **every** tensor.
- Loaded Ferrum GGUF weights are dequantized to f32 in memory. The packed
  int8/int4 kernels are for imported Llama models.

#### QAT equals Q8_0

Quantization-aware training simulates **Q8_0 exactly**: 32-element blocks along
each weight row, the scale computed as `max|x| / 127` and rounded to f16, and
values rounded to int8. Because the training step and the file writer share one
encoder, an int8 file reloads **bit-identically** to the model you trained.
`evaluate()` returns the same numbers before and after saving. **QAT is int8
only.** int4 export is lossy and is applied after training.

#### Tokenizer

The tokenizer is byte-level BPE. Text is pre-split with the GPT-2 regex that
llama.cpp uses for `tokenizer.ggml.pre = "gpt-2"`, and tokens and merges are
stored in the GPT-2 bytes-to-unicode form. `<|endoftext|>` is registered as a
special token during BPE training and **is always reserved**, so every model has
a real `eos_token_id`. The literal string `<|endoftext|>` in text encodes to
that one id.

#### The generic codec (not runnable in llama.cpp)

Character-level SLMs and Ferrum-ML tabular classifiers do not fit the GPT-2
layout. They use a generic layer list: `ferrum.layer_count`,
`ferrum.layer.{i}.type` (`linear`, `activation`, `flatten`, `embedding`,
`layernorm`, `transformer_block`), per-layer dimension keys, and `l{i}.*`
tensors, with matrices in F32, Q8_0, or Q4_0 under the same 32-wide rule. These
are valid GGUF, so general GGUF tools can read their metadata, but llama.cpp
cannot run them: it has no MLP-classifier architecture and no character-level
tokenizer.

#### Checkpoints and the tokenizer file

A training checkpoint is a GGUF container with `ferrum.ckpt.version = 1`, the
scalar settings (`vocab`, `context`, `embed`, `heads`, `hidden`, `blocks`,
`qat`, `tie`, `step`, `rng`), and three f32 tensors per parameter (`p{i}.data`,
`p{i}.m`, `p{i}.v`: the weights and the two Adam moments). A tokenizer file holds
`tokenizer.ferrum.state`. Both are for Ferrum only. Checkpoints written before
this change (magic `FCKP`) do not load; the error tells you to retrain. Old
tokenizer files in the `256;257...` text form are also rejected.

Files are written atomically. The loader bounds-checks every dimension before it
allocates, and a file that is valid GGUF but not a Ferrum model (for example a
Llama GGUF), or a truncated file, gives a `Format` error rather than a panic.

### 11.5 Versioning and compatibility (replacement text)

- FINF `.bin` files do not load. Retrain, or convert with an older Ferrum
  release. Status §7 has the full list of breaking changes.
- Models are GGUF v3 with `ferrum.format = 1`. A future format change will bump
  that number.
- `TransformerConfig` grows only by fields with defaults. Use
  `..TransformerConfig::default()`.
