# Eight upgrades — design spec

**Date:** 2026-07-03 · **Branch:** `feat/eight-upgrades` · **Status:** approved (user: "Looks good, proceed")

Eight features from the Capable-v2 wrap-up, each with unit tests. Constraints that bind every
feature: zero external dependencies, `#![forbid(unsafe_code)]` stays (user decision), one commit
per feature, whole-branch review before merge.

Feature order (dependency + shared-file serialization): F2 → F1 → F8 → F3 → F7 → F5 → F6 → F4,
then docs, then verify/review/finish.

## F2 — Sampling knobs on the GGUF run path

**Correction found while planning:** `llm.rs::generate` (llm.rs:518) already calls the full
sampler `sample_with_params(logits, params, recent, rng)` (slm.rs:2058) with `recent` = prompt +
generated ids — the engine honors all four knobs today. Only the surfaces don't: both CLI
(main.rs:690) and GUI (commands.rs:770) construct `SamplingParams::with_temperature(temp)`,
discarding the other three. F2 is therefore pure plumbing plus fixing the stale
"Greedy/temperature generation" doc comment.
- CLI `run-gguf`: `--top-k N`, `--top-p F`, `--rep F` flags. GUI: three inputs on the GGUF tab → `GgufRunParams { top_k, top_p, rep_penalty }` (serde camelCase) → decode loop.
- Defaults 0 / 1.0 / 1.0 = exactly current behavior.

**Tests:** the sampler itself is already covered in slm.rs; F2 adds a generate-level test on a
tiny model proving non-default knobs reach the sampler (top_k=1 makes generation deterministic
across two different seeds), plus CLI/GUI mapping tests (flag/param values land in
`SamplingParams` fields; defaults produce `SamplingParams::default()`-equivalent values).

## F1 — Chat templates for instruct GGUFs

New module `ferrum_core/src/chat_format.rs`. No Jinja engine — fingerprint + fallback:

- `pub enum ChatFormat { ChatMl, Llama3, Llama2Inst, Zephyr }`.
- `pub fn detect(gguf: &Gguf) -> Option<ChatFormat>`: if `tokenizer.chat_template` metadata exists, fingerprint it (`<|im_start|>`→ChatMl, `<|start_header_id|>`→Llama3, `[INST]`→Llama2Inst, `<|user|>`→Zephyr). Else fall back by architecture: qwen2/qwen3→ChatMl; llama→Llama3 if `<|start_header_id|>` is in the vocab, else Llama2Inst. None if nothing matches.
- `pub fn render(fmt: ChatFormat, user_prompt: &str) -> Vec<Segment>` where `Segment { Text(String), Special(&'static str) }` — we generate the template, so we know each special token verbatim; the runner encodes Text segments normally and maps each Special to its exact vocab id (`token_to_id`); a Special missing from vocab falls back to text-encoding it (degraded but functional).
- `pub fn stop_token(fmt: ChatFormat) -> &'static str` (`<|im_end|>`, `<|eot_id|>`, `</s>`, `<|endoftext|>` per format); runner adds its id to the stop set alongside eos.
- Single-turn only (CLI/GUI take one prompt). No system-prompt parameter — YAGNI until asked.
- Plumbing: on by default when `detect` returns Some; CLI `--raw` disables; GUI checkbox "Chat format (auto)" default-checked → `GgufRunParams { raw: bool }`.

**Tests:** golden rendered sequences per format; detection matrix (each fingerprint, arch
fallbacks incl. llama-with/without header token, none-case); specials map to atomic ids against a
tiny synthetic vocab and missing-special falls back to text; stop-token lookup per format; raw
passthrough leaves the prompt untouched.

## F8 — Q2_K / Q3_K read support

gguf.rs currently refuses them (gguf.rs:570). Add ggml-exact decoders:

- Constants `GGML_Q2_K = 10`, `GGML_Q3_K = 11`; block sizes: Q2_K = 84 B (16 B packed 4-bit scale/min pairs, 64 B 2-bit quants, f16 d + f16 dmin), Q3_K = 110 B (32 B high-bit mask, 64 B 2-bit low quants, 12 B packed 6-bit scales, f16 d) per 256-element super-block; add to `type_nbytes`.
- `dequant_q2_k`, `dequant_q3_k` following ggml's reference layout exactly (Q3_K scales: 6-bit values split across the 12-byte array in ggml's split encoding; Q3_K quants: `q = (low2 | high1<<2) - 4`).
- Refusal message narrows to IQ* only. Writer intentionally unchanged (read-only support; export re-quantizes to a supported output type, which is already the flow).

**Tests:** hand-crafted single blocks with scales/quants chosen so expected f32 values are
computed independently in the test (bit-layout oracle, several patterns incl. negative dmin and
high-bit toggles); `type_nbytes` entries incl. non-multiple-of-256 error; a synthetic fixture
with a Q2_K tensor parses and dequantizes; IQ* still refused with the clear message.

## F3 — qwen3 architecture

- Arch gate (gguf.rs:677) accepts `"qwen3"`. Metadata keys already read via the `{arch}.*` pattern; qwen3 supplies explicit head_dim (`qwen3.attention.key_length`) — honor it when present rather than deriving dim/heads.
- Per-layer tensors `blk.N.attn_q_norm.weight`, `blk.N.attn_k_norm.weight` (RMS-norm over head_dim) load into `Attention` as `Option<RmsNorm>` pairs; when present, apply per-head RMS-norm to q and k **before** RoPE (both `forward_one` and `forward_full`). qwen3 has no qkv bias — absent biases already handled.
- Export: the per-tensor list in gguf_write includes the qk-norm tensors so qwen3 round-trips. Fine-tune: qk-norm weights load but stay frozen (not in the trainable set).

**Tests:** synthetic qwen3 fixture (extend the existing synth builder) parses with norms wired
and explicit head_dim honored; tiny `forward_one` where q/k norm output matches a hand-computed
RMS-norm; with-vs-without norms differ; export round-trip preserves `attn_q_norm`/`attn_k_norm`
byte-exactly; unknown arch (`"gemma2"`) still rejected.

## F7 — Perplexity eval for GGUF models

- `ferrum_core`: `pub fn eval_perplexity(model: &LlamaModel, ids: &[u32]) -> PplStats { cross_entropy, bits_per_token, perplexity, n_predictions }` — teacher-forced `forward_full` over consecutive non-overlapping windows of `context_len`, CE summed over positions predicting `ids[1..]` within each window (log-softmax from existing ops; no window overlap — simple and honest, noted in docs).
- CLI: `eval-gguf <model.gguf> <heldout.txt> [--quant int4|int8|f32]`. GUI: Evaluate tab detects `.gguf` extension and routes to a new `evaluate_gguf` command; same results-table row shape (uniform baseline = vocab_size).

**Tests:** exact oracle — synthetic model with zeroed lm_head ⇒ uniform logits ⇒ perplexity ==
vocab_size (to float tolerance) and CE == ln(V); biasing lm_head toward the true next token
strictly lowers ppl; n_predictions arithmetic across window boundaries (len == k·ctx+1, len <
ctx, len == ctx); empty/1-token input errors cleanly.

## F5 — Streamed export

New `ferrum_core/src/gguf_stream_write.rs`:

- `pub fn stream_export(source: &str, out: &str, quant: GgufQuant, checkpoint: Option<&str>) -> Result<ExportSummary>` (paths are `&str` crate-wide).
- Pass 1: walk the source tensor table (existing streamed reader mode), compute each output tensor's ggml type via the **same per-tensor policy function** used by `llama_gguf_bytes` (factor that policy out of gguf_write.rs — one source of truth), derive sizes/offsets, write header + KVs + tensor table.
- Pass 2: per tensor — read raw, dequant to f32 (existing), overlay the checkpoint's weights for that tensor if given, encode to target type (existing encoders), write, 32-byte-align. Atomic temp-file + rename like the current writer.
- Checkpoint overlay without loading it whole: FLCK v1 (llm_train.rs:1384) is shape header + qat + step + rng, then **three flat f32 blobs — weights, Adam m, Adam v — in canonical `param_data_ref` order**. Validate the shape header against the GGUF config, compute the weights-section file offset, and for each streamed tensor derive its span in canonical order and read only that slice; Adam m/v are never read. Any order-mapping mistake is caught by the byte-identity oracle below.
- CLI `export-gguf` and GUI export switch to it. RAM guard shrinks from ~4× model to max(largest tensor f32 × small constant); GUI hint text updated.

**Tests:** the byte-identity oracle — `stream_export` output must equal `llama_gguf_bytes`
byte-for-byte on the synthetic fixture for every supported output quant, and again with a
checkpoint applied; re-parse output with the reader (offsets/alignment/summary counts);
oversized-tensor guard math unit-tested.

## F6 — Safe-SIMD decode (autovectorization)

`forbid(unsafe_code)` stays. Restructure `ops.rs::qlinear` int4/int8 inner loops:

- `chunks_exact` + exact-size slices to kill bounds checks; 4 independent f32 accumulators; branch-free nibble decode (the split-half QWeight layout already makes int4 unit-stride).
- Scalar reference implementations stay in the test module as the correctness oracle.
- `#[test] #[ignore] bench_qlinear` prints GFLOP/s before/after style numbers (run manually with `--ignored --nocapture`); acceptance = no correctness change, measured speedup reported honestly whatever it is.

**Tests:** property equality (|Δ| < 1e-4 relative) vs scalar reference for int4/int8 across
shapes incl. cols not divisible by chunk width, odd cols (split-half padding edge), rows 1 and
many; the fixed-seed generation pin from F2 must still pass (end-to-end unchanged).

## F4 — CI for ferrum_gui

ci.yml today never builds ferrum_gui (workspace-excluded). Add job `gui`:

- ubuntu-latest; apt-get the documented webkit deps (`libwebkit2gtk-4.1-dev libgtk-3-dev libayatana-appindicator3-dev librsvg2-dev build-essential libssl-dev`); separate cargo cache key.
- Steps from `ferrum_gui/`: `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test`; `node ui/stream.test.js` (execute, not just `--check`); `xvfb-run -a timeout 20 ./target/debug/ferrum_gui` treating exit 124 as pass (boot smoke).

**Verification:** YAML parses locally (python3 yaml.safe_load); real proof = watch the live run
via `gh run watch` after the branch is pushed/merged.

## Docs task

Update manual/05 (GGUF tab knobs + chat checkbox, Evaluate accepts .gguf), howtouse/README/FAQs
where they enumerate flags, quant support (Q2_K/Q3_K read), qwen3 in supported archs, export RAM
note. The zero-unsafe claim stays true — verify no doc needs weakening.

## Out of scope

IQ* quants, Jinja templating, multi-turn chat state, mmap, Q2_K/Q3_K **write** support,
gemma/phi architectures, GPU anything.

## Verification (branch-final)

Workspace suites + ferrum_gui suite + node gates + snap-scrubbed boot smoke + fixture round
trips; whole-branch opus review; merge/push per standing pattern; post-push `gh run watch` for
the new CI job.
