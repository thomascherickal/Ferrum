# Phase 2 — Merge, Samples.md, Docs, Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge workstreams A, B and C, apply the post-merge fix-ups, write `Samples.md` (10 executed walk-throughs), bring every doc in line with the new behaviour, run the full verification battery, and finish the branch.

**Architecture:** Docs only, plus three small post-merge code fix-ups (`--quant` in DVC, the real param count in `slm_server`, and the docs list in `CLAUDE.md`/`AGENTS.md`).

**Tech Stack:** Markdown; the CLI, `ferrum-serve`, DVC (`.venv`), LightEval (`eval/lighteval/.venv`).

**Spec:** `docs/superpowers/specs/2026-10-09-dataops-serve-eval-design.md` (§7, §8, §10). Plans: `2026-10-09-p0-shared-surfaces.md`, `-pA-engine.md`, `-pB-dataops.md`, `-pC-serve-lighteval.md`.

## Global Constraints

- Every command shown in `Samples.md` is executed while writing it, and the output excerpts are pasted from that run. No invented numbers. Where output is long, show the meaningful lines and mark cuts with `…`.
- Docs style: plain English and short sentences, matching `ReadMe.md` and `Manual.md`. Tables for reference material. No marketing adjectives.
- Keep the docs set to `ReadMe.md`, `Manual.md`, `Status.md`, `Samples.md`, and `glossary.md`. Update `CLAUDE.md`'s docs rule to list `Samples.md`.
- `CLAUDE.md` and `AGENTS.md` stay identical (`cp CLAUDE.md AGENTS.md`).
- Each commit passes the full battery (Task 5 Step 1).

## Review Focus

1. A reader on a fresh clone with no DVC or Python installed — Sample 1 must work with only Rust. → Task 3 Sample 1 uses only `cargo` plus a `curl` of the pinned file.
2. A reader on another machine — throughput numbers will differ. → every benchmark table states the machine (`lscpu | grep "Model name"`, core count) and date.
3. Copy-paste of multi-line commands — no shell prompts (`$`) in command blocks, line continuations with `\`. → Task 3 checklist.
4. Version drift in LightEval — `Samples.md` pins `lighteval==0.13.0` and names the two workarounds. → Sample 8.
5. Stale claims in the old docs (zero dependencies, QAT int4, `--tie` saving, `-v` advice, O(context) per token, `clean_corpus` lowercase, eval cost) — every one corrected. → Task 4 grep checklist.

---

### Task 1: Merge the three workstreams

- [ ] **Step 1: Integrate A, B, C onto `feat/dataops-serve-eval` with linear history**

The user's repo keeps history linear (no merge commits), so rebase each workstream onto the integration branch and fast-forward:

```bash
git switch feat/dataops-serve-eval && git merge --ff-only feat/dos-A
git rebase feat/dataops-serve-eval feat/dos-B   # resolve conflicts commit by commit
git switch feat/dataops-serve-eval && git merge --ff-only feat/dos-B
git rebase feat/dataops-serve-eval feat/dos-C
git switch feat/dataops-serve-eval && git merge --ff-only feat/dos-C
cargo test --workspace --release 2>&1 | grep -E "^test result|FAILED"
```

After each rebase, every rebased commit must still build. Spot-check with `git rebase --exec "cargo build --workspace -q" feat/dataops-serve-eval~N` if a conflict touched code.

Conflicts are expected during the rebases in `slm_cli/src/main.rs`:
- the `*_FLAGS` tables
- `spec_for`
- usage text

Resolve by keeping **all** flags from both sides. Use the `superpowers:resolving-merge-conflicts` skill if a conflict is not a pure union.

Expect `Cargo.lock` conflicts too. Resolve with `git checkout --theirs Cargo.lock && cargo generate-lockfile` only if needed, then re-run the tests.

- [ ] **Step 2: Post-merge fix-ups**

1. `params.yaml`: add `quant: int8` under `train:`. In `dvc.yaml`'s `train` cmd, append `--quant ${train.quant}`.
2. `slm_server/src/lib.rs`: replace the `ponytail:` placeholder param count with `model.model.num_params()`. In `slm_server/tests/api.rs`, restore `assert!(v["params"].as_u64().unwrap() > 0)`.
3. `cargo test --workspace --release`, `cargo test -p slm_cli --release --test determinism`, `.venv/bin/dvc repro` (the `train` stage re-runs because its command changed).

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "chore: post-merge fix-ups (--quant in DVC, server param count)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git worktree remove ../Ferrum-A; git worktree remove ../Ferrum-B; git worktree remove ../Ferrum-C
```

---

### Task 2: Gather the facts the docs will cite

- [ ] Run and save each output to `$S/facts/` (`S=/tmp/claude-1000/-home-thomas-Projects-ferrum-cpu/88623770-ced2-4a37-be7f-75cb7f78b99e/scratchpad`):

```bash
mkdir -p $S/facts
lscpu | grep -E "Model name|^CPU\(s\)" > $S/facts/machine.txt; date -I >> $S/facts/machine.txt
cargo test --workspace --release 2>&1 | grep "^test result" > $S/facts/tests.txt
(cd ferrum_gui && cargo test 2>&1 | grep "^test result") >> $S/facts/tests.txt
cat metrics/val.json metrics/test.json data/prepared/manifest.json > $S/facts/dvc.txt
./target/release/train_transformer info models/slm.gguf > $S/facts/info.txt
scripts/bench_slm.sh data/prepared/train.txt > $S/facts/bench_auto.txt
FERRUM_NUM_THREADS=1 scripts/bench_slm.sh data/prepared/train.txt > $S/facts/bench_1t.txt
cargo bench --bench gemm 2>&1 | tail -30 > $S/facts/gemm.txt
eval/lighteval/run.sh --model models/slm.gguf --test data/prepared/test.txt --out metrics/lighteval.json --max-samples 200
cp metrics/lighteval.json $S/facts/
git log --oneline master..HEAD > $S/facts/commits.txt
```

Also collect the baseline-vs-final speed numbers from the Phase 1A commit messages: `git log --format=%B master..HEAD | grep -A8 -E "baseline|ndarray|faer|rayon"`.

---

### Task 3: Write `Samples.md`

**Files:**
- Create: `Samples.md`

Structure: a title, a one-paragraph intro, a contents list, then ten `## N. Title` sections. Each section has these sub-parts in this order:
- **Goal** (one sentence)
- **You need** (prerequisites)
- **Steps**: numbered, each with a `bash` block, followed by an **Output** block pasted from the run
- **What to look at** (2–4 bullets interpreting the output)

Run every block from the repo root on the machine in `$S/facts/machine.txt`, and keep the samples' scratch files in `samples/` (git-ignored: add `samples/` to `.gitignore`).

- [ ] **Sample 1 — Quick start: a 500K SLM on TinyStories.**
  - Fetch the file with `scripts/fetch.sh` (the pinned URL and sha256 from `params.yaml`) and run `prepare`.
  - Train the reference config (`--context 128 --embed 96 --heads 4 --blocks 3 --hidden 320 --vocab 1024 --tokens 2e6 --lr 5e-3 --cosine --warmup 50 --clip 1`, 507,552 params), then `info`, then a `generate`.
  - Include a short "sizing a config" table for ~100K, ~500K and ~2M params. Get the counts from `info` on models trained for 1 step each, using `--tokens 1`. Add the per-block formula `4·e² + 4·e + 2·e·h + h + e + 4·e` plus the embeddings and head; verify it against `info` before printing it.
- [ ] **Sample 2 — Full DVC pipeline.**
  - Commands: `python3 -m venv .venv && .venv/bin/pip install -r requirements-dataops.txt`, `.venv/bin/dvc repro`, `.venv/bin/dvc dag`.
  - Edit `params.yaml` (`train.tokens: 4.0e6`) with `.venv/bin/dvc exp run -S train.tokens=4e6` or a plain edit plus `dvc repro`, then show `dvc params diff` and `dvc metrics diff`.
  - Show that `dvc status` is clean afterwards.
  - Explain that a second `dvc repro --force train` reproduces the same `models/slm.gguf` sha256, because training is deterministic.
- [ ] **Sample 3 — Cleaning your own corpus.**
  - Build a messy input in `samples/raw/`: a BOM file, CRLF endings, zero-width characters, a duplicated paragraph, a two-book Gutenberg file, and a file with an invalid byte.
  - Run `prepare` without `--lossy` to show the error with its byte offset, then run with `--lossy --dedup --nfkc --strip-gutenberg --min-chars 20`.
  - Show `manifest.json` and explain each rejected-reason counter and how the hashes pin inputs and outputs.
- [ ] **Sample 4 — Training controls.**
  - Run a token budget with `--val 0.02 --patience 3`, `-v` progress lines and a `--checkpoint`, and interrupt it with `timeout -s INT 30`.
  - Resume with `--resume`, and show the epoch and step numbers lining up.
  - Show the strict CLI: `--contxt 8` produces an error, and so does a missing `--resume` file.
- [ ] **Sample 5 — Evaluation.**
  - `eval` on `val.txt` and `test.txt` with `--json`.
  - Train the same config with `--quant int4` and `--quant f32`, then build a table of size and perplexity for f32, int8 and int4.
  - Explain bits/byte vs perplexity, and why bits/byte compares models with different tokenizers.
- [ ] **Sample 6 — Generation.**
  - Greedy (`--temp 0`), sampled (`--temp 0.8 --top-k 40 --top-p 0.9 --repetition-penalty 1.2`), `--stop` (repeatable), `--max-tokens`, and EOS stopping.
  - Show the same `--gen-seed` giving the same output at `FERRUM_NUM_THREADS=1` and `=8` (compare with `md5sum`).
- [ ] **Sample 7 — Serving, and running your model in llama.cpp, Ollama, and LM Studio.**
  - The trained `models/slm.gguf` is a GPT-2-architecture GGUF. Run it with each tool, using the exact commands pinned in the Phase 0-G plan:
    - `llama-cli` (or `llama-completion`) greedy
    - `ollama create ferrum-tiny -f Modelfile` plus `ollama run`
    - `lms import` plus a completion
  - Paste each output. Show that the greedy continuation matches `train_transformer generate --temp 0`.
  - `cargo run --release -p slm_server -- --model models/slm.gguf &`
  - `curl` every route: `/health`, `/v1/info`, `/v1/loglikelihood` with two continuations, `/v1/loglikelihood_rolling`, `/v1/generate` with `stop`.
  - Show one 400 (an oversized batch) and its error JSON.
  - Note that it binds to loopback and processes one request at a time on a model thread.
- [ ] **Sample 8 — Testing with LightEval.**
  - Install from `eval/lighteval/requirements.txt` (pinned `lighteval==0.13.0`, `xxhash<4`).
  - `run.sh` with `--max-samples 200`, plus the DVC route: `.venv/bin/dvc unfreeze lighteval && .venv/bin/dvc repro lighteval`.
  - Show the results table from `metrics/lighteval.json`.
  - Explain each task: ferrum_heldout, tinystories_cloze, BLiMP pairs, and ARC-easy. Say plainly that ARC-easy near chance (25%) is expected for a 500K model.
  - Explain the two 0.13.0 workarounds and why the litellm/TGI backends can't run log-likelihood tasks.
- [ ] **Sample 9 — Benchmarking.**
  - `scripts/bench_slm.sh` at auto and 1 thread, `cargo bench --bench gemm`, and a thread-scaling table (1/2/4/8 threads via `FERRUM_NUM_THREADS`).
  - The before/after table for this branch: review baseline vs final for train tokens/s, eval seconds and generate ms/token. Source: the Phase 1A commit messages plus `$S/facts`.
  - Name the machine.
- [ ] **Sample 10 — The Rust library API.**
  - A complete `examples/`-style program shown inline: train from a string with `TransformerConfig`, `score`, `score_rolling`, `generate_tokens`, then save and load.
  - Put it in `ferrum_core/examples/library_tour.rs` so `cargo run --release -p ferrum_core --example library_tour` runs it. Paste that output.

- [ ] **Commit**

```bash
git add Samples.md ferrum_core/examples/library_tour.rs .gitignore
git commit -m "docs: Samples.md — 10 executed walk-throughs (train, eval, serve, LightEval, bench)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Update ReadMe, Manual, Status, CLAUDE.md/AGENTS.md

- [ ] **FINF → GGUF everywhere:** every mention of FINF, `.bin` model files, the `.fckp`/`.flck` binary formats, and `tok.bpe` becomes GGUF (`.gguf`). Status §7 records the breaking change. ReadMe's pitch says Ferrum's own models run in llama.cpp, Ollama, and LM Studio. It also says Ferrum-ML's tabular models are GGUF but not runnable there, because llama.cpp has no MLP-classifier architecture.
- [ ] **ReadMe.md:**
  - Rewrite the pitch: Ferrum is pure Rust and CPU-only; replace the "No dependencies" row with one naming the crates actually kept and why.
  - Add `slm_server` and `eval/lighteval` to the layout table, and `Samples.md` to Documentation.
  - Add `prepare`, DVC and `ferrum-serve` lines to "Building and testing".
- [ ] **Manual.md:**
  - §2: rewrite "what zero dependencies buys" as "what few dependencies buy", covering the trade-off and the kept crates.
  - §5 corpus prep: document `prepare` and the manifest.
  - §6.2 CLI reference: add the new flags (`--quant`, `--debug`, `-v` meaning, generate flags, `eval --json`).
  - §7 generation cost: one re-prime per `context_len/2` tokens.
  - Eval section: every token is scored, plus speed.
  - Add a short serving and LightEval chapter that points to `Samples.md` §7–8.
  - Fix the `--tie` text: shared weights during training, and a tied GGUF omits `output.weight` (llama.cpp reuses `token_embd`), so the file shrinks by `vocab × embed` weights.
  - Fix the `clean_corpus` lowercase default.
- [ ] **Status.md:**
  - Snapshot: date 2026-10-09, branch, test counts from `$S/facts/tests.txt`.
  - Known issues: remove the fixed items; keep the honest remaining ones (whole-file training corpus, `--tie` on-disk size, server serializes requests, LightEval 0.13.0 workarounds).
  - Roadmap.
  - §7 backward compatibility:
    - perplexity now covers every token
    - `TransformerConfig.log_every_steps`
    - `temperature 0` is now exact argmax
    - `-v` on train no longer floods
    - CLI rejects unknown flags
  - §9 metrics: the before/after numbers.
- [ ] **CLAUDE.md:**
  - Hard constraints: replace the zero-dependency bullet with "dependencies need a measured reason" and the kept crates.
  - Layout: add `slm_server`, `eval/lighteval`, DVC files.
  - Commands: `prepare`, `dvc repro`, `ferrum-serve`, the Python unittest.
  - Docs rule: include `Samples.md`.
  - Then run `cp CLAUDE.md AGENTS.md`.
- [ ] **Stale-claim sweep.** This must return nothing still false:

```bash
grep -n -i -E "finf|\.bin\b|relu|zero[- ]dep|no external|std[- ]only|int8/int4 quantization-aware|O\(context\) per token|lowercases|--verbose.*per-epoch|saves vocab" ReadMe.md Manual.md Status.md CLAUDE.md ferrum_core/src/lib.rs ferrum_core/Cargo.toml slm_cli/src/main.rs
```

Review each hit and fix it or confirm it is still true. In particular, `ferrum_core/Cargo.toml`'s description says "Zero-dependency" and `lib.rs`/`main.rs` module docs say "zero external dependencies".

- [ ] **Commit**

```bash
git add ReadMe.md Manual.md Status.md CLAUDE.md AGENTS.md ferrum_core slm_cli
git commit -m "docs: ReadMe/Manual/Status/CLAUDE.md for dataops, serving, LightEval, deps

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Final verification and finish

- [ ] **Step 1: Battery**

```bash
cargo build --workspace && cargo test --workspace --release 2>&1 | grep -E "^test result|FAILED"
cargo fmt --all -- --check && cargo clippy --workspace --all-targets -- -D warnings
cargo build -p slm_wasm --target wasm32-unknown-unknown --release
cargo clippy -p slm_wasm --target wasm32-unknown-unknown -- -D warnings
cargo deny check
(cd ferrum_gui && cargo fmt -- --check && cargo clippy --all-targets -- -D warnings && cargo test 2>&1 | grep "test result")
node --check ferrum_gui/ui/app.js && node --check ferrum_gui/ui/stream.js && node ferrum_gui/ui/stream.test.js
cargo deny --manifest-path ferrum_gui/Cargo.toml check --config deny.toml
.venv/bin/dvc status
eval/lighteval/.venv/bin/python -m unittest eval/lighteval/test_ferrum_model.py
```

Expected: everything green. `dvc status` is up to date.

- [ ] **Step 2: Whole-branch review.** Invoke `superpowers:requesting-code-review` for `master..feat/dataops-serve-eval`. Fix Critical/Important findings, and fix Minor findings in one polish commit (the user's standing preference). Re-run Step 1.

- [ ] **Step 3: Finish** (the user's pattern: fast-forward, push, delete branch, no PR). Confirm with the user before pushing.

```bash
git switch master && git merge --ff-only feat/dataops-serve-eval
git push origin master && git branch -d feat/dataops-serve-eval feat/dos-A feat/dos-B feat/dos-C
```
