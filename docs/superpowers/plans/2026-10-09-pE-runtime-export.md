# Phase 1E — One-Command Export to Ollama and LM Studio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **Runs after Phase 0** (needs the strict CLI parser from Phase 0 Task 5 and the GPT-2 GGUF codec from Phase 0-G Task G1). It can run in parallel with Phases 1A–1C in its own worktree.

**Goal:** Add two `train_transformer` subcommands that put a trained Ferrum GGUF into a local runtime in one step:
- `export-ollama`
- `export-lmstudio`

(Unsloth Studio was dropped at the user's request.)

**Architecture:** One module, `slm_cli/src/runtime_export.rs`. Each command validates that the file is a runnable Ferrum GGUF (`general.architecture == "gpt2"`, read with `ferrum_core::Gguf::open`), then drives the tool's own CLI with `std::process::Command`. There is no HTTP client, and no new dependency.

**Tech Stack:** Rust std; the Ollama and LM Studio (`lms`) CLIs.

**Spec:** `docs/superpowers/specs/2026-10-09-dataops-serve-eval-design.md` §4.0, plus the amendment "one-command export".

## Global Constraints

- **Full unit-test coverage (user requirement, 2026-10-09):** every function you add or change gets unit tests written with it, in the same commit. Cover every branch, error path and edge case (empty input, boundaries, invalid input). Measure with `cargo llvm-cov --release -p <crate> --summary-only` (and `--html` for detail). Every new or changed line must be covered; justify any line that cannot be (e.g. `main()`'s process exit) in the report. Reviewers check this per function.

- Worktree: `git worktree add ../Ferrum-E -b feat/dos-E feat/dataops-serve-eval`, taken after Phase 0 is complete.
- No `unsafe`. Unit tests come first and land in the same commit as the feature.
- Each subcommand's flags live in its own `const *_FLAGS: Spec` table (Phase 0 parser).
- Tool binaries are found in this order: the flag `--bin <path>`, then the env var (`FERRUM_OLLAMA`, `FERRUM_LMS`), then `PATH`, then the default install location (`~/.local/bin/ollama`, `~/.lmstudio/bin/lms`). A missing tool gives an error naming all four places checked.
- Refuse a non-runnable GGUF: `ferrum` (generic) architecture, character-level, or tabular. The error says why (llama.cpp has no such architecture).
- Pinned behaviour (verified 2026-10-09, Ollama v0.40.2, LM Studio 0.4.18):
  - Ollama: `ollama create NAME -f Modelfile`. The Modelfile is `FROM <abs path>`, `TEMPLATE "{{ .Prompt }}"`, `PARAMETER stop "<|endoftext|>"`, `PARAMETER temperature 0.8`, and `PARAMETER num_ctx <gpt2.context_length>`.
  - LM Studio: `lms import -y -l --user-repo <publisher>/<name> <file>` (`-l` links instead of copying). Then confirm with `lms ls --json` that an entry's `path` ends with `<publisher>/<name>/<file name>`.
- Tools are installed on this machine at their default locations: Ollama v0.40.2 at `~/.local/bin/ollama`, with `OLLAMA_MODELS` defaulting to `~/.ollama/models`; LM Studio 0.4.18 at `~/.lmstudio/bin/lms`. Tool discovery works without env vars.
- A test that needs a tool **skips** with `eprintln!("skip: …")` when it can't find the tool, **unless `FERRUM_REQUIRE_RUNTIMES=1`**, in which case a missing tool is a test **failure**. CI and pre-release runs set it.

## Review Focus

1. A model path with spaces or a relative path — the Modelfile must use the absolute, quoted path. → Task E1 test `modelfile_uses_absolute_quoted_path`.
2. Re-exporting under an existing name — Ollama overwrites (that is its `create` semantics). LM Studio reports "already imported" with exit 0, or re-links. Both must count as success, not as an error. → Task E1 test `export_is_idempotent` (tool-gated).
3. A generic or char-level Ferrum GGUF — refused before any tool runs. → Task E1 test `refuses_non_runnable_gguf`.
4. Ollama server not running — `ollama create` fails. The error must say "start it with `ollama serve`". → Task E1 test `ollama_down_gives_actionable_error` (uses a bogus `OLLAMA_HOST`).
5. `FERRUM_REQUIRE_RUNTIMES=1` with a tool missing — the test fails, it does not skip. → Task E1 test `require_runtimes_turns_skip_into_failure`.

---

### Task E1: `export-ollama` and `export-lmstudio`

**Files:**
- Create: `slm_cli/src/runtime_export.rs`, `slm_cli/tests/runtime_export_cli.rs`
- Modify: `slm_cli/src/main.rs` (`mod runtime_export;`, flag tables, `spec_for`, dispatch, usage)

**Interfaces:**
- Produces:
  ```rust
  pub struct GgufSummary { pub arch: String, pub context_len: u32, pub name: String }
  /// Errors unless `path` is a runnable (gpt2-architecture) Ferrum/GGUF file.
  pub fn runnable_summary(path: &Path) -> Result<GgufSummary, String>;
  pub fn ollama_modelfile(abs_model: &Path, s: &GgufSummary) -> String;
  pub fn export_ollama(model: &Path, name: &str, bin: Option<&str>) -> Result<(), String>;
  pub fn export_lmstudio(model: &Path, repo: &str, bin: Option<&str>) -> Result<String /*modelKey*/, String>;
  pub fn find_tool(flag: Option<&str>, env: &str, name: &str, fallback: Option<PathBuf>) -> Result<PathBuf, String>;
  ```
- CLI:
  - `train_transformer export-ollama <model.gguf> <name> [--bin P] [--modelfile-only out.Modelfile]`
  - `train_transformer export-lmstudio <model.gguf> [--repo ferrum/<stem>] [--bin P]`

- [ ] **Step 1: Failing unit tests** (in `runtime_export.rs`):
  - `modelfile_uses_absolute_quoted_path`: `ollama_modelfile(Path::new("/tmp/my model.gguf"), &s)` contains `FROM "/tmp/my model.gguf"`, the TEMPLATE line, the stop line, and `num_ctx 64` for `context_len 64`.
  - `find_tool_order`: given a temp dir with a fake executable, check the precedence flag > env > PATH > fallback, and that a missing tool lists all four places.
  - `refuses_non_runnable_gguf`: write a generic-codec GGUF (a tabular model through `loader::to_bytes`) and assert `runnable_summary` errors with "not runnable in llama.cpp".

- [ ] **Step 2: Failing integration tests** (`slm_cli/tests/runtime_export_cli.rs`, tool-gated):
  - Train the 0.1M reference model (`--context 64 --embed 64 --heads 4 --blocks 2 --hidden 128 --vocab 512 --tie`, small budget) to a temp `.gguf`.
  - `export_ollama_then_generate`: run `export-ollama`, then `POST /api/generate` (raw, temperature 0, num_predict 8) and require a non-empty response that equals llama.cpp's output for the same file and prompt. Clean up with `ollama rm`.
  - `export_lmstudio_then_generate`: run `export-lmstudio`. Check that `lms ls --json` lists it, then `lms load KEY --gpu off --identifier ID -y`, `lms server start -p 1234`, and `POST /v1/completions` (`max_tokens` 9, `temperature` 0, `repeat_penalty` 1.0). Require non-empty text equal to llama.cpp's output. Then unload and remove the import (`rm -rf ~/.lmstudio/models/<repo>`).
  - `export_is_idempotent`: export twice; both succeed.
  - `ollama_down_gives_actionable_error`: `OLLAMA_HOST=127.0.0.1:1` makes it fail with the message containing `ollama serve`.

  - `require_runtimes_turns_skip_into_failure`: a unit test of the shared `require_or_skip(tool_result)` helper. With the env flag set, a missing tool makes it panic; without the flag it returns a skip. The helper lives in `slm_cli/tests/common/mod.rs` and is shared with Phase 0-G G5.
  - `export_ollama_then_generate` uses the installed Ollama. If no server answers on `OLLAMA_HOST` (default `127.0.0.1:11434`), the test spawns `ollama serve` on `127.0.0.1:11545` itself and stops it afterwards.

- [ ] **Step 3: Implement.**
  - Write the Modelfile to a temp dir. Run the tool with captured stderr, and include it in errors.
  - For LM Studio, treat "already exists" as success, then resolve the `modelKey` via `lms ls --json`. Parse it with a small string scan for `"path"` and `"modelKey"`. No JSON crate in slm_cli unless Phase 1B already added `serde_json`; if it did, use it.

- [ ] **Step 4: Verify.** Run the full battery, then the integration tests with the scratchpad tools:

```bash
FERRUM_REQUIRE_RUNTIMES=1 cargo test -p slm_cli --release --test runtime_export_cli -- --nocapture --test-threads 1
```

Expected: everything passes with nothing skipped (the flag makes a skip fail).

- [ ] **Step 5: Commit**

```bash
git add slm_cli
git commit -m "feat(cli): export-ollama / export-lmstudio — one command into the local runtime

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
