# Ferrum

A zero-dependency, pure-Rust engine for training, running, and shipping small
causal Transformer language models (SLMs) on the CPU, plus GGUF import,
fine-tuning, and export for Llama/Qwen checkpoints. Overview in `ReadMe.md`,
usage in `Manual.md`, current state / known issues / roadmap in `Status.md`.

> `CLAUDE.md` and `AGENTS.md` are identical copies. Edit both together.

## Hard constraints

Do not break these; they are the product's selling points.

- **`ferrum_core` has zero external dependencies.** Its `[dependencies]` table
  stays empty — std only. Benchmarks are std-only too (`harness = false`, no
  Criterion). Only `slm_wasm` (wasm-bindgen) and `ferrum_gui` (Tauri) pull crates.
- **`#![forbid(unsafe_code)]`** in `ferrum_core`. No `unsafe`, no SIMD intrinsics.
- **Deterministic at any thread count.** Matmuls split across the persistent
  worker pool in `parallel.rs`; results must stay bit-for-bit identical
  regardless of `FERRUM_NUM_THREADS`.
- **WASM-portable.** `ferrum_core` must compile for `wasm32-unknown-unknown`
  (no threads/fs assumptions on that path). CI builds `slm_wasm` for wasm32.
- **Backward compatibility.** FINF v4 (f32) and v5 (int8/int4) model files must
  keep loading. `TransformerConfig` is extended with new fields + `Default`;
  callers use `..TransformerConfig::default()`. The positional
  `train_transformer*` functions keep stride 1 and their exact results. Record
  any compat change in `Status.md` §7.

## Layout

| Path | What |
|---|---|
| `ferrum_core/` | The engine library; everything calls into it |
| `slm_cli/` | Binary `train_transformer`: train, generate, eval, info, run-gguf, finetune-gguf, export-gguf |
| `slm_wasm/` | wasm-bindgen browser bindings (`TransformerSLMModel`) |
| `tests/` | Cross-crate integration/regression tests (`tests/tests/*.rs`) |
| `ferrum_gui/` | Tauri app "Ferrum SLM Studio" — **excluded from the workspace** |

Sibling folders `../Ferrum-ML` (tabular MLP demos, depends on `ferrum_core` by
path) and `../Ferrum-Junk` (archive of removed features) are separate repos.
Changing a `ferrum_core` public API can break `Ferrum-ML`.

## Two Transformer stacks — don't mix them up

They share only the low-level kernels (`ops`, `parallel`, `quant`).

| | Ferrum's own | Imported |
|---|---|---|
| Modules | `train_transformer`, `slm`, `tokenizer`, `meta`, `loader` | `gguf`, `gguf_tokenizer`, `llm`, `llm_train`, `gguf_write`, `chat_format` |
| Architecture | learned positions, LayerNorm, ReLU MLP, dense MHA | RoPE, RMSNorm, SwiGLU, GQA |
| Training | from scratch, Adam/AdamW, QAT | AdamW fine-tuning (f32) |
| Ships as | FINF `.bin` (weights + metadata + tokenizer in one file) | GGUF v3 |

## Commands

Run from the repo root unless noted. This mirrors CI (`.github/workflows/ci.yml`).

```bash
cargo build --workspace
cargo test --workspace --release          # as documented in ReadMe; CI runs it without --release
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy -p slm_wasm --target wasm32-unknown-unknown -- -D warnings
cargo deny check
cargo bench --bench gemm                  # std-only matmul/decode microbenchmark
bash scripts/build_wasm.sh                # slm_wasm → slm_wasm/pkg/ (needs wasm-bindgen 0.2.122)
```

**`ferrum_gui`** is outside the workspace, so `cargo … -p ferrum_gui` from the
root fails. Work inside the crate:

```bash
cd ferrum_gui
cargo fmt -- --check && cargo clippy --all-targets -- -D warnings && cargo test
cargo tauri dev                           # run the app
node --check ui/app.js && node --check ui/stream.js && node ui/stream.test.js
```

The UI (`ui/index.html`, `app.js`, `styles.css`) is static — `cargo build`
won't catch JS errors; `node --check` is the gate. Tabs are generic: a
`<button class="tab" data-tab="X">` auto-wires to `<section class="panel" id="panel-X">`.

## Conventions

- Rust 2021, rustfmt defaults (`rustfmt.toml`, Unix newlines); clippy clean with `-D warnings`.
- Deliberate simplifications are marked with a `// ponytail:` comment naming the
  ceiling and the upgrade path.
- Docs are consolidated into three files: `ReadMe.md`, `Manual.md`, `Status.md`.
  Don't add new top-level docs; update `Status.md` (snapshot, test counts,
  known issues, roadmap) when shipping a feature. `glossary.md` is a
  beginner-facing guide to LLM terms, not a domain glossary for agents.
- Git: solo repo, linear history. Feature branches are fast-forward merged into
  `master`, pushed, then deleted — no PRs, no merge commits. Each commit should
  build and pass its tests on its own. Conventional-style subjects
  (`feat(gguf): …`, `fix(gui): …`, `docs(status): …`).

## Agent skills

### Issue tracker

GitHub Issues via the `gh` CLI; external PRs are not a triage surface. See `docs/agents/issue-tracker.md`.

### Triage labels

Default vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` + `docs/adr/` at the repo root (created lazily). See `docs/agents/domain.md`.
