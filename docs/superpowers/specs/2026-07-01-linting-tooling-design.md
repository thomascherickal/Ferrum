# Linting & Error-Catching Tooling — Design

**Date:** 2026-07-01
**Status:** Approved (design)

## Goal

Add automated linting and error-catching to the project and gate CI on it, so
regressions in code quality, formatting, syntax, and dependency health are caught
before merge. Four tools, all enforced (CI fails on violations):

1. **Clippy** — the Rust linter (bug/anti-pattern detection).
2. **rustfmt** — consistent Rust formatting.
3. **`node --check`** — JavaScript syntax gate for the hand-written frontend.
4. **cargo-deny** — dependency vulnerability / license / source auditing.

The project already has CI (`.github/workflows/ci.yml`) covering workspace
build+test, a wasm build, the excluded `ferrum_gui` Tauri crate, and a JS
`stream.test.js` run. This work adds the missing *quality* gates on top.

## Context / constraints

- Rust **workspace**: `ferrum_core` (zero-dependency, std-only), `slm_cli`,
  `train_cli`, `tabular_wasm`, `tests` (50 packages in the lockfile).
- `ferrum_gui` is a **separate crate excluded from the workspace** (heavy Tauri
  deps, 457 packages, its own `Cargo.lock`). Every tool must be run twice: once
  at the workspace root and once inside `ferrum_gui/`.
- Frontend is **vanilla JS, no npm/package.json**. Keep it that way — no ESLint.
- `web/pkg/tabular_wasm.js` is **wasm-bindgen generated** and must be excluded
  from linting.

## Decisions

### 1. Clippy — fix-all, then enforce `-D warnings`

- Fix every existing warning (~42 in `ferrum_core` plus a handful elsewhere).
  - `cargo clippy --fix` for the auto-fixable majority (`is_multiple_of`,
    `manual div_ceil`, `repeat().take()`, `manual RangeInclusive::contains`, …).
  - For lints that would hurt readability to "fix", a **targeted
    `#[allow(clippy::…)]` with a one-line justification** — not a blanket
    crate-level allow. Expected cases: `needless_range_loop` where one index
    drives several parallel arrays, and `too_many_arguments` on the RoPE helper.
- CI enforcement (`-D warnings`):
  - Workspace: `cargo clippy --workspace --all-targets -- -D warnings`
  - wasm: `cargo clippy -p tabular_wasm --target wasm32-unknown-unknown -- -D warnings`
  - GUI: `cargo clippy --all-targets -- -D warnings` inside `ferrum_gui/`

### 2. rustfmt — isolated reformat + blame-ignore

Enforcing standard rustfmt reformats **~9.7k lines** (≈9,160 workspace + ≈540
GUI) and erases the current hand-alignment. Introduced the standard way for an
unformatted repo:

1. Add `rustfmt.toml` (stable-safe: `edition = "2021"`, otherwise defaults).
2. **One isolated commit** `style: apply rustfmt across the tree` containing
   only mechanical `cargo fmt` output (workspace + `ferrum_gui`).
3. Record that commit SHA in **`.git-blame-ignore-revs`** so `git blame` skips
   the reformat (and note the `git config blame.ignoreRevsFile` one-liner in the
   file header).
4. CI: `cargo fmt --all -- --check` (workspace) + `cargo fmt -- --check` (GUI).

### 3. JavaScript `node --check`

CI syntax-gates the hand-written files only:
`ferrum_gui/ui/app.js`, `ferrum_gui/ui/stream.js`, `ferrum_gui/ui/stream.test.js`,
`web/shared/engine.js`. **Excludes** `web/pkg/**` (generated). Zero new
dependencies.

### 4. cargo-deny + project license

**Project license = MIT.** The repo currently has *no* `LICENSE` file,
`Cargo.toml` declares `license = "MIT"`, but `readme.md` says "either MIT or
Apache-2.0 at your option" — a contradiction. Standardize on MIT:
- Add a top-level `ferrum/LICENSE` with the MIT text
  (`Copyright (c) 2026 Thomas Cherickal`).
- Fix `readme.md` to state MIT only (remove the dual-license line).
- `Cargo.toml` already says `MIT` — no change.

**Dependency policy is separate from the project license.** cargo-deny audits
the licenses of *dependencies*, which must stay a permissive multi-license
allow-list — the 457-package Tauri tree pulls Apache-2.0/BSD/ISC/Unicode/etc.,
so an MIT-only *dependency* rule would fail CI immediately and is not the intent.

Add `deny.toml`:
- **advisories**: deny RUSTSEC vulnerabilities and unmaintained crates.
- **licenses**: permissive allow-list derived from the actual trees
  (MIT, Apache-2.0, Apache-2.0-WITH-LLVM-exception, BSD-2/3-Clause, ISC,
  Unicode-3.0, Zlib, MPL-2.0, CC0-1.0, …); extend to whatever the trees actually
  contain so the initial run is green.
- **bans**: `multiple-versions = "warn"` (informational, non-blocking).
- **sources**: allow only crates.io.

CI runs `cargo deny check` at the workspace root **and** inside `ferrum_gui/`
(separate lockfile). Uses the maintained `EmbarkStudios/cargo-deny-action`.

## CI structure

- New **`lint`** job (ubuntu-latest, no system deps): toolchain with
  `clippy,rustfmt` components → `cargo fmt --all -- --check` →
  `cargo clippy --workspace --all-targets -- -D warnings` → `node --check` of
  the four JS files.
- New **`deny`** job(s) via the cargo-deny action for root + `ferrum_gui`.
- Extend the existing **`gui`** job with `cargo fmt -- --check` and
  `cargo clippy --all-targets -- -D warnings`.
- Extend the existing **`wasm`** job with the wasm-target clippy step.
- Toolchain: add `components: clippy, rustfmt` to the relevant
  `dtolnay/rust-toolchain@stable` steps.

## Rollout (commit sequence, on `feature/linting-tooling`)

1. **clippy fixes** — code changes + minimal justified `#[allow]`s. Verified:
   `cargo clippy … -D warnings` clean (workspace + gui + wasm); full test suite
   still green (fixes must be behavior-preserving).
2. **rustfmt** — `rustfmt.toml`, the isolated reformat, `.git-blame-ignore-revs`.
   Verified: `cargo fmt … --check` clean; tests green.
3. **cargo-deny + license** — `deny.toml`, the `LICENSE` file, and the
   `readme.md` license fix. Verified: `cargo deny check` passes (root + gui).
4. **CI + toolchain** — the workflow edits above.

## Success criteria

Locally green before the CI commit lands:
- `cargo clippy --workspace --all-targets -- -D warnings` (+ gui, + wasm) — clean.
- `cargo fmt --all -- --check` (+ gui) — clean.
- `node --check` on each of the four JS files — clean.
- `cargo deny check` (root + gui) — passes.
- `cargo test` workspace + `ferrum_gui` — still all green.

## Non-goals

- No ESLint / npm / `package.json` (keeps the frontend dependency-free).
- No pre-commit / husky hooks (CI is the gate; can be added later if wanted).
- No functional/behavioral changes — clippy fixes and rustfmt are
  behavior-preserving; the test suites must stay green throughout.
