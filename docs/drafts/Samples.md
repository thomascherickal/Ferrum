# Ferrum Samples

<!-- DRAFT RULES: every OUTPUT marker below is replaced by a block pasted from a real run on the machine named in the header. Do not write any output by hand. MEASURE markers mark a number to fill from a run. -->

Ten walk-throughs. Each one has a goal, what you need, numbered steps with exact
commands, the output from a real run, and what to look at. Run every command
from the repository root. Scratch files go in `samples/` (git-ignored).

Machine for all outputs: <!-- OUTPUT: contents of `lscpu | grep -E "Model name|^CPU\(s\)"` and `date -I` -->

Throughput numbers differ on other machines. Compare ratios, not absolute speeds.

All samples use the release build and put it on your `PATH` for the shell session:

```bash
cargo build --release --workspace
export PATH="$PWD/target/release:$PATH"
mkdir -p samples
```

## Contents

1. [Quick start: a 500K SLM on TinyStories](#1-quick-start-a-500k-slm-on-tinystories)
2. [The full DVC pipeline](#2-the-full-dvc-pipeline)
3. [Cleaning your own corpus](#3-cleaning-your-own-corpus)
4. [Training controls](#4-training-controls)
5. [Evaluation](#5-evaluation)
6. [Generation](#6-generation)
7. [Serving, and running your model in llama.cpp, Ollama, and LM Studio](#7-serving-and-running-your-model-in-llamacpp-ollama-and-lm-studio)
8. [Testing with LightEval](#8-testing-with-lighteval)
9. [Benchmarking](#9-benchmarking)
10. [The Rust library API](#10-the-rust-library-api)

---

## 1. Quick start: a 500K SLM on TinyStories

**Goal.** Go from nothing to a trained 507,552-parameter model that writes short stories.

**You need.** Rust, `curl`, and about 25 MB of disk. No Python and no DVC.

**Steps.**

1. Download the pinned TinyStories V2 file. The script checks the sha256 and fails on a mismatch.

```bash
scripts/fetch.sh \
  https://huggingface.co/datasets/roneneldan/TinyStories/resolve/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/TinyStoriesV2-GPT4-valid.txt \
  6874bae9a4c1a4e7edcf0e53b86c17817e9cf881fc75ff2368da457b80c0585d \
  data/raw/tinystories.txt
```

<!-- OUTPUT: fetch.sh result, showing the size 22502601 and the checksum check passing -->

2. Clean and split it.

```bash
train_transformer prepare --input data/raw/tinystories.txt --out data/prepared \
    --dedup --nfkc --min-chars 50 --max-chars 20000
```

<!-- OUTPUT: prepare summary: documents kept per split, rejected counts -->

3. Train the reference config.

```bash
train_transformer train data/prepared/train.txt models/slm.gguf \
    --context 128 --embed 96 --heads 4 --blocks 3 --hidden 320 --vocab 1024 \
    --tokens 2e6 --lr 5e-3 --cosine --warmup 50 --clip 1 \
    --val 0.02 --patience 3 --tokenizer models/tokenizer.gguf
```

<!-- OUTPUT: training banner (Epochs: budget), loss lines, final line, elapsed time -->

4. Inspect the model.

```bash
train_transformer info models/slm.gguf
```

<!-- OUTPUT: info output; parameter count must read 507,552 and quantization int8 -->

5. Generate a story.

```bash
train_transformer generate models/slm.gguf "Once upon a time" --max-tokens 80 --temp 0.7 --gen-seed 1
```

<!-- OUTPUT: generated text -->

**Sizing a config to a parameter target.** Parameter counts for three sizes. Each
row was trained for one step (`--tokens 1`) and read from `info`.

```bash
for cfg in "64 4 2 128 512" "96 4 3 320 1024" "192 6 4 768 2048"; do
  set -- $cfg
  train_transformer train data/prepared/train.txt samples/size.gguf \
      --context 128 --embed $1 --heads $2 --blocks $3 --hidden $4 --vocab $5 \
      --tokens 1 --force >/dev/null
  train_transformer info samples/size.gguf | grep -i param
done
```

<!-- OUTPUT: three info lines. The sizing table below is filled from them. -->

| Target | `--embed` | `--heads` | `--blocks` | `--hidden` | `--vocab` | Parameters |
|---|---|---|---|---|---|---|
| about 100K | 64 | 4 | 2 | 128 | 512 | <!-- MEASURE: from info; the `--tie` variant of this config is 103,936 --> |
| about 500K | 96 | 4 | 3 | 320 | 1024 | 507,552 |
| about 2M | 192 | 6 | 4 | 768 | 2048 | <!-- MEASURE: from info --> |

Per block the parameters are `4·e² + 4·e + 2·e·h + h + e + 4·e` (attention,
feed-forward, and two LayerNorms; `e` = embed, `h` = hidden). Add the token and
position embeddings, the final LayerNorm, and the head (`e × vocab`, no bias, and
none when `--tie` is on). <!-- MEASURE: verify this formula against info before keeping it; e.g. it must reproduce 507,552 -->

**What to look at.**

- The banner shows the token budget, not an epoch count.
- `info` reports the parameter count and the quantization (int8).
- Keep `--embed`, `--hidden`, and `--vocab` multiples of 32 so every matrix can be stored as Q8_0.
- The file is `models/slm.gguf`, a `gpt2` GGUF. Sample 7 runs it in other tools.

---

## 2. The full DVC pipeline

**Goal.** Reproduce data, model, and metrics with one command, change a parameter, and see exactly what reruns.

**You need.** Sample 1's prerequisites, plus Python 3.10 or newer.

**Steps.**

1. Install DVC into a local virtual environment.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dataops.txt
```

2. Show the stage graph and run the pipeline.

```bash
.venv/bin/dvc dag
.venv/bin/dvc repro
```

<!-- OUTPUT: dag drawing; repro log with fetch, prepare, train, eval running and lighteval skipped as frozen -->

3. Read the metrics.

```bash
.venv/bin/dvc metrics show
cat data/prepared/manifest.json
```

<!-- OUTPUT: metrics table (val.json, test.json) and the manifest -->

4. Change a parameter and rerun. Only `train` and `eval` run, because `fetch` and `prepare` do not depend on `train.tokens`.

```bash
.venv/bin/dvc exp run -S train.tokens=4e6
.venv/bin/dvc params diff
.venv/bin/dvc metrics diff
```

<!-- OUTPUT: exp run log, params diff, metrics diff. If `dvc exp run` is not used, edit train.tokens in params.yaml and run `dvc repro` instead; show the same two diffs. -->

5. Check that nothing is stale.

```bash
.venv/bin/dvc status
```

<!-- OUTPUT: "Data and pipelines are up to date." -->

6. Prove training is deterministic: force the train stage again and compare hashes.

```bash
sha256sum models/slm.gguf > samples/slm1.sha
.venv/bin/dvc repro --force train
sha256sum -c samples/slm1.sha
```

<!-- OUTPUT: "models/slm.gguf: OK" -->

**What to look at.**

- Stages whose inputs did not change are skipped.
- `dvc.lock` records the hash of every input and output. It is committed; `data/` and `models/` are not.
- There is no DVC remote. The data stays on this machine.
- The same hash after `--force` means the whole run is reproducible.

---

## 3. Cleaning your own corpus

**Goal.** Turn messy text files into clean train/val/test splits and read the manifest that records what happened.

**You need.** Sample 1's prerequisites.

**Steps.**

1. Build a messy input. It has a byte-order mark, Windows line endings, zero-width characters, a duplicated paragraph, a two-book Gutenberg file, and a file with an invalid byte.

```bash
mkdir -p samples/raw
printf '\xef\xbb\xbfThe sun rose over the quiet town and the baker opened his shop.\n' > samples/raw/bom.txt
printf 'The river ran down to the sea.\r\nIt carried leaves and small boats.\r\n' > samples/raw/crlf.txt
printf 'The mountain stood tall\xe2\x80\x8b and silent\xe2\x80\x8c in the cold morning air.\n' > samples/raw/zerowidth.txt
printf 'A fox came out of the wood at night.\n\nA fox came out of the wood at night.\n' > samples/raw/dup.txt
printf 'Preface text.\n*** START OF THE PROJECT GUTENBERG EBOOK ONE ***\nBook one is about a lighthouse keeper who counted the waves each night.\n*** END OF THE PROJECT GUTENBERG EBOOK ONE ***\nlicence text\n*** START OF THE PROJECT GUTENBERG EBOOK TWO ***\nBook two is about a girl who found a map inside an old clock.\n*** END OF THE PROJECT GUTENBERG EBOOK TWO ***\n' > samples/raw/books.txt
printf 'This line has a bad byte \xff in the middle of the sentence.\n' > samples/raw/badbyte.txt
```

2. Run `prepare` without `--lossy`. It stops and names the file and byte offset.

```bash
train_transformer prepare --input samples/raw --out samples/prepared
```

<!-- OUTPUT: the invalid UTF-8 error with file name and byte offset -->

3. Run it with the cleaning options.

```bash
train_transformer prepare --input samples/raw --out samples/prepared \
    --lossy --dedup --nfkc --strip-gutenberg --min-chars 20 --val-frac 0.2 --test-frac 0.2
cat samples/prepared/manifest.json
```

<!-- OUTPUT: prepare summary and manifest.json -->

**What to look at.**

- `rejected` has one counter per reason: `empty`, `too_short` (under `--min-chars`), `too_long`, `duplicate`, and `invalid_utf8`.
- `inputs` and `outputs` list the sha256 of every file. If two manifests carry the same hashes, the runs are byte-identical.
- `train_stats` is `corpus_stats` of the train split.
- The split is by document and stable: rerun after appending a file and the old documents stay in their old splits.

---

## 4. Training controls

**Goal.** Use a token budget, cosine learning rate, validation with early stopping, and checkpoints; interrupt a run and resume it; see the strict CLI reject mistakes.

**You need.** Sample 1's data.

**Steps.**

1. Train with a validation split, progress lines, and a checkpoint. `timeout -s INT 30` interrupts it after 30 seconds.

```bash
timeout -s INT 30 train_transformer train data/prepared/train.txt samples/run.gguf \
    --context 128 --embed 96 --heads 4 --blocks 3 --hidden 320 --vocab 1024 \
    --tokens 2e6 --lr 5e-3 --cosine --warmup 50 --clip 1 \
    --val 0.02 --patience 3 --tokenizer samples/tok.gguf \
    --checkpoint samples/run.fckp --checkpoint-every 100 -v
```

<!-- OUTPUT: banner and `[train] step S/T tokens/s R elapsed Xs` lines up to the interrupt -->

2. Resume with the same flags plus `--resume`.

```bash
train_transformer train data/prepared/train.txt samples/run.gguf \
    --context 128 --embed 96 --heads 4 --blocks 3 --hidden 320 --vocab 1024 \
    --tokens 2e6 --lr 5e-3 --cosine --warmup 50 --clip 1 \
    --val 0.02 --patience 3 --tokenizer samples/tok.gguf \
    --checkpoint samples/run.fckp --checkpoint-every 100 -v --resume samples/run.fckp
```

<!-- OUTPUT: resume banner, with step and epoch numbers continuing from where step 1 stopped -->

3. Try two mistakes.

```bash
train_transformer train data/prepared/train.txt samples/x.gguf --contxt 8
train_transformer train data/prepared/train.txt samples/x.gguf --resume samples/does-not-exist.fckp
```

<!-- OUTPUT: "unknown flag --contxt for 'train' (valid: ...)" and the missing-checkpoint error -->

**What to look at.**

- The step and epoch counters after `--resume` continue from the checkpoint. A mid-epoch interrupt finishes the partial epoch.
- With `--val`, the best validation score is restored on resume.
- `-v` prints one line per 10 steps. `--debug` is the old per-op trace.
- The CLI never falls back to a default silently.

---

## 5. Evaluation

**Goal.** Measure perplexity and bits per byte, write them as JSON, and compare f32, int8, and int4 files.

**You need.** Samples 1 and 3 done; `data/prepared/val.txt` and `test.txt` exist.

**Steps.**

1. Evaluate the Sample 1 model on both splits.

```bash
mkdir -p metrics
train_transformer eval models/slm.gguf data/prepared/val.txt  --json metrics/val.json
train_transformer eval models/slm.gguf data/prepared/test.txt --json metrics/test.json
cat metrics/test.json
```

<!-- OUTPUT: eval printout and the JSON with perplexity, cross_entropy_nats, bits_per_token, bits_per_byte, tokens, bytes, seconds -->

2. Train the same config at three precisions.

```bash
for q in f32 int8 int4; do
  train_transformer train data/prepared/train.txt samples/slm-$q.gguf \
      --context 128 --embed 96 --heads 4 --blocks 3 --hidden 320 --vocab 1024 \
      --tokens 2e6 --lr 5e-3 --cosine --warmup 50 --clip 1 --tokenizer samples/tok-$q.gguf \
      --quant $q --force
  train_transformer eval samples/slm-$q.gguf data/prepared/test.txt --json samples/test-$q.json
done
ls -l samples/slm-*.gguf
```

<!-- OUTPUT: file sizes and perplexities -->

| Precision | File size (bytes) | Test perplexity | Bits per byte |
|---|---:|---:|---:|
| f32 | <!-- MEASURE --> | <!-- MEASURE --> | <!-- MEASURE --> |
| int8 (Q8_0) | <!-- MEASURE --> | <!-- MEASURE --> | <!-- MEASURE --> |
| int4 (Q4_0) | <!-- MEASURE --> | <!-- MEASURE --> | <!-- MEASURE --> |

**What to look at.**

- Every token after the first is scored, so `tokens` in the JSON is the token count of the file.
- **Perplexity** is per token. **Bits per byte** divides the total loss by the file's bytes, so it compares models with different tokenizers. Per-token numbers do not.
- QAT is int8 only. The int8 file reloads bit-identically to the model that trained, so its numbers match the f32-QAT model closely. int4 is converted after training and is lossy.
- Matrices whose row length is not a multiple of 32 are not quantized (they stay in a wider type). That is why the reference config uses multiples of 32.

---

## 6. Generation

**Goal.** Use the sampling controls, stop sequences, the end-of-text token, and a fixed seed.

**You need.** `models/slm.gguf` from Sample 1.

**Steps.**

1. Greedy output (`--temp 0` is exact argmax), then sampled output.

```bash
train_transformer generate models/slm.gguf "Once upon a time" --max-tokens 60 --temp 0
train_transformer generate models/slm.gguf "Once upon a time" --max-tokens 60 \
    --temp 0.8 --top-k 40 --top-p 0.9 --repetition-penalty 1.2 --gen-seed 7
```

<!-- OUTPUT: both completions -->

2. Stop sequences. `--stop` is repeatable. The output ends before the first match.

```bash
train_transformer generate models/slm.gguf "Once upon a time" --max-tokens 200 --temp 0.7 --gen-seed 3 \
    --stop "." --stop "!"
```

<!-- OUTPUT: a completion that ends before the first "." or "!" -->

3. End of text. The model emits `<|endoftext|>` at the end of a story, and generation stops there.

```bash
train_transformer generate models/slm.gguf "Once upon a time" --max-tokens 400 --temp 0.7 --gen-seed 5
```

<!-- OUTPUT: a story shorter than 400 tokens; note the stop reason if the CLI prints it -->

4. The same seed gives the same text at any thread count.

```bash
FERRUM_NUM_THREADS=1 train_transformer generate models/slm.gguf "Once upon a time" --max-tokens 60 --temp 0.8 --gen-seed 9 | md5sum
FERRUM_NUM_THREADS=8 train_transformer generate models/slm.gguf "Once upon a time" --max-tokens 60 --temp 0.8 --gen-seed 9 | md5sum
```

<!-- OUTPUT: two identical md5sum lines -->

**What to look at.**

- `--temp 0` always gives the same text and matches llama.cpp at `--temp 0` (Sample 7).
- `--max-tokens` is a token budget. `--chars` is a character count.
- Generation past the context window works: the cache is rebuilt from the last half window once per half window of new tokens.
- Identical hashes show that thread count does not change results.

---

## 7. Serving, and running your model in llama.cpp, Ollama, and LM Studio

**Goal.** Run the model from three other tools, check they agree with Ferrum, then serve it over HTTP and call every route.

**You need.** `models/slm.gguf` from Sample 1. The tools are optional; skip any you do not have. The commands were verified with llama.cpp b11521, Ollama v0.40.2, and LM Studio 0.4.18.

Set these to your install paths:

```bash
export FERRUM_LLAMA_BIN=/path/to/llama.cpp        # the directory holding llama-completion and llama-tokenize
export FERRUM_OLLAMA=/path/to/ollama
export FERRUM_LMS="$HOME/.lmstudio/bin/lms"
PROMPT="Once upon a time"
```

**Part A. Ferrum's own greedy output (the reference).**

```bash
train_transformer generate models/slm.gguf "$PROMPT" --max-tokens 16 --temp 0
```

<!-- OUTPUT: 16-token greedy continuation -->

**Part B. llama.cpp.**

```bash
"$FERRUM_LLAMA_BIN/llama-completion" -m models/slm.gguf -p "$PROMPT" -n 16 --temp 0 -no-cnv --no-warmup </dev/null 2>/dev/null
"$FERRUM_LLAMA_BIN/llama-tokenize" -m models/slm.gguf -p "$PROMPT" --ids --log-disable
```

<!-- OUTPUT: llama.cpp continuation and token ids; compare with Part A -->

**Part C. Ollama.** Run a private server on its own port and model directory.

```bash
export OLLAMA_HOST=127.0.0.1:11545 OLLAMA_MODELS="$PWD/samples/ollama-models"
"$FERRUM_OLLAMA" serve >samples/ollama.log 2>&1 &
sleep 3
printf 'FROM %s\nTEMPLATE "{{ .Prompt }}"\nPARAMETER num_predict 16\n' "$PWD/models/slm.gguf" > samples/Modelfile
"$FERRUM_OLLAMA" create ferrum-tiny -f samples/Modelfile
curl -s http://127.0.0.1:11545/api/generate \
  -d '{"model":"ferrum-tiny","prompt":"Once upon a time","raw":true,"stream":false,"options":{"temperature":0,"num_predict":16}}'
```

<!-- OUTPUT: the JSON response, with the "response" field to compare with Part A -->

Stop that server when done: `kill %1`.

**Part D. LM Studio.**

```bash
"$FERRUM_LMS" import -y -l --user-repo ferrum-test/slm models/slm.gguf
KEY=$("$FERRUM_LMS" ls --json | python3 -I -c 'import json,sys; print([m["modelKey"] for m in json.load(sys.stdin) if "ferrum-test" in m.get("path","")][0])')
"$FERRUM_LMS" load "$KEY" --gpu off --identifier ferrum-tiny -y
"$FERRUM_LMS" server start -p 1234
curl -s http://127.0.0.1:1234/v1/completions \
  -d '{"model":"ferrum-tiny","prompt":"Once upon a time","max_tokens":17,"temperature":0,"repeat_penalty":1.0}'
"$FERRUM_LMS" server stop
"$FERRUM_LMS" unload ferrum-tiny
```

<!-- OUTPUT: the completion JSON. Note: max_tokens 17 shows 16 visible tokens; repeat_penalty 1.0 matters because the default 1.1 changes the text. Check the `lms ls --json` field names and the `lms server stop` / `lms unload` subcommands against the installed version. -->

**Part E. One-command exports.**

```bash
train_transformer export-ollama   models/slm.gguf ferrum-tiny
train_transformer export-lmstudio models/slm.gguf
train_transformer export-unsloth  models/slm.gguf
```

<!-- OUTPUT: export-ollama (Modelfile written, ollama create result), then `ollama run ferrum-tiny "Once upon a time"` -->
<!-- OUTPUT: export-lmstudio result and the model listed by `lms ls` -->
<!-- OUTPUT: export-unsloth: PENDING. The location where Unsloth Studio finds local models is still being verified. Paste the real result once the command is final. -->

**Part F. Serve over HTTP.**

```bash
cargo run --release -p slm_server -- --model models/slm.gguf &
sleep 2
```

Every route:

```bash
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/v1/info
curl -s http://127.0.0.1:8080/v1/loglikelihood -d '{"requests":[{"context":"The cat sat on the","continuation":" mat"},{"context":"The cat sat on the","continuation":" moon"}]}'
curl -s http://127.0.0.1:8080/v1/loglikelihood_rolling -d '{"texts":["Tom had a red ball. He liked to play with it."]}'
curl -s http://127.0.0.1:8080/v1/generate -d '{"requests":[{"prompt":"Once upon a time","max_tokens":40,"stop":["."],"temperature":0}]}'
```

<!-- OUTPUT: "ok"; the info JSON (params must equal the `info` count; model_sha256 must equal `sha256sum models/slm.gguf`); the two scores with logprob, n_tokens, greedy; the rolling score; the generated text -->

An oversized batch (65 items, over the default limit of 64) is rejected before any compute:

```bash
python3 -I -c 'import json; print(json.dumps({"requests":[{"context":"a","continuation":"b"}]*65}))' \
  | curl -s -i http://127.0.0.1:8080/v1/loglikelihood -d @-
kill %1
```

<!-- OUTPUT: HTTP 400 and {"error": "..."} -->

**What to look at.**

- Greedy outputs from Ferrum and llama.cpp should be the same text for an f32 file. For int8 or int4 files, tiny numeric differences in llama.cpp can change tokens late in a run.
- `llama-tokenize` ids equal Ferrum's ids for the same text, because both use the GPT-2 pre-tokenizer.
- LightEval and your own scripts use `/v1/loglikelihood`. A higher `logprob` is a more likely continuation. `greedy` is true when every token was the model's top choice.
- The server binds to `127.0.0.1` by default and has no authentication. Limits (body 8 MiB, batch 64, 65,536 characters per text) are checked first: 413 for a large body, 400 for the others, 422 for a model error.
- The server handles requests one at a time on a model thread. <!-- MEASURE: confirm (plan) vs spawn_blocking (spec) -->

---

## 8. Testing with LightEval

**Goal.** Score the model with Hugging Face LightEval and read the results.

**You need.** Python 3.10 or newer. `models/slm.gguf` and `data/prepared/test.txt` from Sample 1. About 2 GB of disk for CPU PyTorch.

**Steps.**

1. Install the pinned versions in their own environment.

```bash
python3 -m venv eval/lighteval/.venv
eval/lighteval/.venv/bin/pip install -r eval/lighteval/requirements.txt
```

`requirements.txt` pins `lighteval==0.13.0` and `xxhash<4` (xxhash 4.x crashes the details logger), and uses the CPU PyTorch index.

2. Run it. `run.sh` starts `ferrum-serve`, waits for `/health`, runs LightEval, and stops the server.

```bash
eval/lighteval/run.sh --model models/slm.gguf --test data/prepared/test.txt \
    --out metrics/lighteval.json --max-samples 200
cat metrics/lighteval.json
```

<!-- OUTPUT: LightEval progress summary and metrics/lighteval.json -->

3. Or run it through DVC. The stage is frozen by default.

```bash
.venv/bin/dvc unfreeze lighteval && .venv/bin/dvc repro lighteval
```

<!-- OUTPUT: dvc repro log for the lighteval stage -->

| Task | Metrics | Value |
|---|---|---|
| `ferrum_heldout` | word perplexity, byte perplexity, bits per byte | <!-- MEASURE --> |
| `tinystories_cloze` | `loglikelihood_acc` | <!-- MEASURE --> |
| `blimp` subset | accuracy | <!-- MEASURE --> |
| `arc:easy` | accuracy | <!-- MEASURE --> |

**What to look at.**

- `ferrum_heldout` is rolling log-likelihood of whole test documents.
- `tinystories_cloze` gives the model a story prefix and four sentences: the real next sentence and three from other stories (fixed seed). It must choose the real one. Chance is 25%.
- `blimp` compares a grammatical sentence with a minimally different ungrammatical one. Chance is 50%.
- `arc:easy` is a science question set. **A 500K model scores near chance (25%). That is expected.** Do not read it as a failure.
- Two LightEval 0.13.0 workarounds are built in: `FerrumModel` does not call `super().__init__()` (it raises in 0.13.0), and rolling log-likelihood is not cached (0.13.0's perplexity cache drops results).
- LightEval's litellm and TGI backends cannot run log-likelihood tasks, which is why Ferrum is a custom model class.

---

## 9. Benchmarking

**Goal.** Measure training, evaluation, and generation speed, thread scaling, and the before/after of this release.

**You need.** Sample 1's data. Report the machine with every table.

**Steps.**

1. The end-to-end benchmark script, at auto threads and at one thread.

```bash
scripts/bench_slm.sh data/prepared/train.txt
FERRUM_NUM_THREADS=1 scripts/bench_slm.sh data/prepared/train.txt
```

<!-- OUTPUT: both script outputs -->

2. The matmul microbenchmark.

```bash
cargo bench --bench gemm
```

<!-- OUTPUT: the tail of the gemm bench -->

3. Thread scaling.

```bash
for t in 1 2 4 8; do
  echo "threads $t"
  FERRUM_NUM_THREADS=$t scripts/bench_slm.sh data/prepared/train.txt
done
```

| Threads | Train tokens/s | Eval seconds | Generate ms/token |
|---:|---:|---:|---:|
| 1 | <!-- MEASURE --> | <!-- MEASURE --> | <!-- MEASURE --> |
| 2 | <!-- MEASURE --> | <!-- MEASURE --> | <!-- MEASURE --> |
| 4 | <!-- MEASURE --> | <!-- MEASURE --> | <!-- MEASURE --> |
| 8 | <!-- MEASURE --> | <!-- MEASURE --> | <!-- MEASURE --> |

4. Before and after this release. The "before" numbers come from the Phase 1A commit messages (the review baseline); the "after" numbers from the runs above.

| Measure | Review baseline | Final |
|---|---:|---:|
| Train tokens/s | <!-- MEASURE --> | <!-- MEASURE --> |
| Eval seconds | <!-- MEASURE --> | <!-- MEASURE --> |
| Generate ms/token | <!-- MEASURE --> | <!-- MEASURE --> |
| Matmul kernel | hand-written pool | <!-- MEASURE: which crate won, or none --> |

**What to look at.**

- Output text and trained weights are identical at every thread count; only time changes.
- Eval is much faster than before because it scores whole windows in one forward pass.
- Generation time per token stays flat as the output grows past the context window.

---

## 10. The Rust library API

**Goal.** Train, score, generate, save, and load from Rust.

**You need.** Rust only. The program lives in `ferrum_core/examples/library_tour.rs`.

**The program.**

```rust
use ferrum_core::{GenerativeSLM, Rng, SamplingParams, TransformerConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A small repetitive corpus; documents are separated by <|endoftext|>.
    let story = "Tom had a red ball. He kicked it to the park. His dog ran after it.";
    let corpus = vec![story; 200].join("\n<|endoftext|>\n");

    let cfg = TransformerConfig {
        context_len: 64, embed_dim: 64, num_heads: 4, num_blocks: 2, hidden_dim: 128,
        vocab_size: 512, lr: 0.01, batch_size: 8, max_tokens: 200_000,
        ..TransformerConfig::default()
    };
    let mut rng = Rng::new(1);
    let slm = GenerativeSLM::train_transformer_config_threaded(&corpus, &cfg, 0, &mut rng, |ep, loss| {
        println!("epoch {ep}: loss {loss:.4}");
    })?;

    // Score a continuation, and a whole text.
    let s = slm.score("Tom had a red", " ball")?;
    println!("score: logprob {:.3}, {} tokens, greedy {}", s.logprob, s.n_tokens, s.greedy);
    let r = slm.score_rolling(story)?;
    println!("rolling: logprob {:.3} over {} tokens", r.logprob, r.n_tokens);

    // Generate: greedy, with a stop string.
    let greedy = SamplingParams { temperature: 0.0, ..SamplingParams::default() };
    let g = slm.generate_tokens("Tom had", 30, &[".".to_string()], &greedy, &mut Rng::new(2))?;
    println!("generated: {:?} ({} tokens, {:?})", g.text, g.n_tokens, g.stop_reason);

    // Save as GGUF and load it back.
    slm.save("samples/library_tour.gguf")?;
    let back = GenerativeSLM::load("samples/library_tour.gguf")?;
    let s2 = back.score("Tom had a red", " ball")?;
    println!("reloaded score: logprob {:.3}", s2.logprob);
    Ok(())
}
```

<!-- MEASURE: the program above is a draft. Compile it against the finished API (field names, load/save signatures, SamplingParams::default, StopReason Debug) and replace it with the file as committed. -->

```bash
mkdir -p samples
cargo run --release -p ferrum_core --example library_tour
```

<!-- OUTPUT: the program's output -->

**What to look at.**

- `score` returns the log-probability of the continuation given the context, the number of tokens scored, and whether all of them were the model's top choice.
- `score_rolling` scores every token after the first; long texts use sliding windows.
- `generate_tokens` returns only the continuation, without the stop string.
- A model saved with `save` is a `gpt2` GGUF. The reloaded model gives the same score, because the int8 file reloads bit-identically to a QAT model.
