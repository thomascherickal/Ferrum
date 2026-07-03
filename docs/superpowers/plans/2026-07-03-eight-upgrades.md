# Eight Upgrades Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the eight improvement areas from the Capable-v2 wrap-up — GGUF sampling knobs, chat templates, Q2_K/Q3_K read, qwen3, GGUF perplexity eval, streamed export, safe-SIMD decode, ferrum_gui CI — each with unit tests, on one branch.

**Architecture:** All engine work lands in `ferrum_core` behind existing seams (`SamplingParams` plumbing, a new `chat_format` module, new dequant arms in `gguf.rs`, optional qk-norm in `Attention`, `eval_perplexity` on `LlamaModel`, a two-pass streamed writer beside `gguf_write`, loop restructuring in the `qlinear` kernel). CLI (`slm_cli`) and GUI (`ferrum_gui`) expose each feature. CI gains a `gui` job.

**Tech Stack:** Pure Rust (std only), Tauri 2 GUI shell, plain JS UI, GitHub Actions.

## Global Constraints

- Zero external dependencies in `ferrum_core`/`slm_cli`. `#![forbid(unsafe_code)]` (ferrum_core/src/lib.rs:51) must stay — user decision.
- `ferrum_gui` is EXCLUDED from the cargo workspace: run all its cargo commands from `ferrum_gui/`, never `-p ferrum_gui` at root. rust-analyzer "unlinked-file"/phantom errors on its files are noise; cargo output is authoritative.
- NEVER delete or modify anything under `web/datasets/` (a past subagent repeatedly deleted tracked `model.bin` files there). Do not touch files outside your task's list.
- serde structs crossing to JS use `#[serde(rename_all = "camelCase")]`; JS sends camelCase keys.
- Rust→JS field additions must default sensibly (`#[serde(default…)]`) so older payloads still deserialize.
- Every task: `cargo test --workspace` at repo root green, plus `ferrum_gui/` `cargo test` + `node --check ferrum_gui/ui/app.js` when GUI files changed, `cargo fmt` + clippy clean for touched crates, then one commit ending with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- If a numbered requirement here conflicts with what you find in the code, STOP and report back instead of improvising (plan-mandated decisions are the controller's).

Feature order is dependency-driven and serializes shared files: T1(F2) → T2(F1) → T3(F8) → T4(F3) → T5(F7) → T6(F5) → T7(F6) → T8(F4) ∥ T9(docs) → T10(verify/review/finish).

---

### Task 1: F2 — expose sampling knobs on the GGUF run path

`LlamaModel::generate` (ferrum_core/src/llm.rs:518) already calls the full sampler
`sample_with_params` (ferrum_core/src/slm.rs:2058) with `recent` = prompt+generated. Only the
surfaces discard the knobs.

**Files:**
- Modify: `ferrum_core/src/llm.rs` (doc comment ~515; new test in its test mod)
- Modify: `slm_cli/src/main.rs` (known-flags registry ~line 86; `cmd_run_gguf` lines 571–698; usage strings ~205 and ~575)
- Modify: `ferrum_gui/src/commands.rs` (`GgufRunParams` ~line 675; `gguf_run_inner` ~line 770)
- Modify: `ferrum_gui/ui/index.html` (GGUF tab), `ferrum_gui/ui/app.js` (`ggRun` handler payload)

**Interfaces:**
- Consumes: `SamplingParams { temperature, top_k, top_p, repetition_penalty }` (slm.rs:103).
- Produces: CLI flags `--top-k N --top-p F --rep F` on `run-gguf`; `GgufRunParams` fields `top_k: usize`, `top_p: f32`, `rep_penalty: f32` (JS: `topK`, `topP`, `repPenalty`).

- [ ] **Step 1: llm.rs** — fix the stale doc comment on `generate` ("Greedy/temperature generation…" → say it performs full `SamplingParams` sampling: repetition penalty, temperature, top-k, top-p — see `sample_with_params`). Add to llm.rs's test mod (reuse its existing tiny-model builder used by `generate_is_deterministic_and_respects_eos`, llm.rs:841):

```rust
#[test]
fn generate_topk1_is_greedy_regardless_of_seed() {
    // top_k = 1 collapses sampling to argmax, so two different seeds must
    // produce identical output — proof the knob reaches the sampler.
    let m = /* same builder the determinism test uses */;
    let params = SamplingParams { temperature: 1.0, top_k: 1, top_p: 1.0, repetition_penalty: 1.0 };
    let a = m.generate(&[1, 2], 8, &params, None, &mut Rng::new(7)).unwrap();
    let b = m.generate(&[1, 2], 8, &params, None, &mut Rng::new(999)).unwrap();
    assert_eq!(a, b);
}
```

(`eos` argument shape may be `None` today; Task 2 changes it to `&[]` — write it to match the code as it exists NOW.)

- [ ] **Step 2: CLI** — add `"top-k"`, `"top-p"`, `"rep"` to the known-flags list (main.rs ~86, alongside `"quant"`, `"max"`). In `cmd_run_gguf` replace line 690:

```rust
let params = SamplingParams {
    temperature: temp,
    top_k: args.get("top-k", 0usize),
    top_p: args.get("top-p", 1.0f32),
    repetition_penalty: args.get("rep", 1.0f32),
};
```

Extend both usage strings mentioning run-gguf options with `[--top-k N] [--top-p F] [--rep F]`.

- [ ] **Step 3: GUI backend** — extend `GgufRunParams`:

```rust
#[serde(default)]
pub top_k: usize,
#[serde(default = "one_f32")]
pub top_p: f32,
#[serde(default = "one_f32")]
pub rep_penalty: f32,
```

with `fn one_f32() -> f32 { 1.0 }` beside the struct. Validate in `gguf_run_inner` next to the temp check: `top_p` must be finite and in `(0.0, 1.0]`, `rep_penalty` finite and `> 0.0` (clear error strings). Replace line 770 with the four-field `SamplingParams` construction. Add a serde test (serde_json is already in ferrum_gui's dep tree via Tauri):

```rust
#[test]
fn gguf_run_params_sampling_defaults_off() {
    let p: GgufRunParams = serde_json::from_str(
        r#"{"modelPath":"m","prompt":"p","quant":"int4","maxNew":8,"temp":0.8,"force":false}"#,
    ).unwrap();
    assert_eq!((p.top_k, p.top_p, p.rep_penalty), (0, 1.0, 1.0));
}
```

(If commands.rs has no test mod, create one at the bottom.)

- [ ] **Step 4: GUI frontend** — in the GGUF tab (index.html), after the temperature input add three numeric inputs in the tab's existing field markup style: ids `ggTopK` (default `0`, hint "0 = off"), `ggTopP` (default `1.0`, hint "1.0 = off"), `ggRep` (default `1.0`, hint "1.0 = off"). In app.js's `ggRun` handler include `topK`, `topP`, `repPenalty` in the invoke payload using the tab's existing numeric parsing pattern.

- [ ] **Step 5: Verify + commit** — root `cargo test --workspace`; from `ferrum_gui/`: `cargo test`, `cargo clippy --all-targets -- -D warnings`; `node --check ferrum_gui/ui/app.js`. Commit: `feat(gguf): expose top-k/top-p/repetition penalty on the run path`.

---

### Task 2: F1 — chat templates for instruct GGUFs

**Files:**
- Create: `ferrum_core/src/chat_format.rs`
- Modify: `ferrum_core/src/lib.rs` (declare `pub mod chat_format;` + re-export `ChatFormat`), `ferrum_core/src/gguf_tokenizer.rs` (add `token_id`), `ferrum_core/src/llm.rs` (`generate` stop-set signature + its tests), `slm_cli/src/main.rs` (`cmd_run_gguf`, `--raw` flag, usage), `ferrum_gui/src/commands.rs` (`GgufRunParams.raw`, wiring), `ferrum_gui/ui/index.html` + `app.js` (checkbox `ggChatFmt`, default checked, payload `raw: !checked`)
- Also: `rg -n "\.generate\(" --type rust` and update EVERY call site of `LlamaModel::generate` to the new signature (finetune sample paths, tests).

**Interfaces:**
- Consumes: `Gguf::meta(key) -> Option<&MetaValue>` (gguf.rs:505), `Gguf::architecture()`, `GgufTokenizer::{encode, bos, eos}`.
- Produces: `GgufTokenizer::token_id(&self, piece: &str) -> Option<usize>`; `chat_format::{ChatFormat, Segment, detect, render, stop_token, encode_segments}`; **changed signature** `LlamaModel::generate(prompt, max_new, params, stops: &[usize], rng)` (was `eos: Option<usize>`) — stop token is pushed to the output then generation breaks, exactly like today's eos.

- [ ] **Step 1: tokenizer lookup** — in gguf_tokenizer.rs beside `bos()`/`eos()`:

```rust
/// Exact-piece vocabulary lookup (used to map chat special tokens to ids).
pub fn token_id(&self, piece: &str) -> Option<usize> {
    self.token_to_id.get(piece).map(|&id| id as usize)
}
```

- [ ] **Step 2: the module** — create `ferrum_core/src/chat_format.rs`:

```rust
//! Chat-prompt formatting for instruct GGUFs.
//!
//! No Jinja: the `tokenizer.chat_template` string is *fingerprinted* against
//! the four formats small llama/qwen models actually use, with a vocab-based
//! fallback when the key is absent. `render` returns text/special segments so
//! the runner can map each special token to its exact vocab id instead of
//! letting byte-BPE shred it. Single-turn only, by design.

use crate::gguf_tokenizer::GgufTokenizer;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChatFormat {
    ChatMl,     // Qwen2/Qwen3 (<|im_start|> …)
    Llama3,     // Llama-3.x header tags
    Llama2Inst, // [INST] … [/INST]
    Zephyr,     // TinyLlama-chat style <|user|>/<|assistant|>
}

pub enum Segment {
    Text(String),
    Special(&'static str),
}

/// Decide the chat format. `template` is `tokenizer.chat_template` if present;
/// `has_token` reports whether a piece exists verbatim in the vocab (the
/// instruct signal for template-less files — base models must NOT be wrapped,
/// so arch alone is never enough).
pub fn detect(
    template: Option<&str>,
    arch: Option<&str>,
    has_token: impl Fn(&str) -> bool,
) -> Option<ChatFormat> {
    if let Some(t) = template {
        if t.contains("<|im_start|>") {
            return Some(ChatFormat::ChatMl);
        }
        if t.contains("<|start_header_id|>") {
            return Some(ChatFormat::Llama3);
        }
        if t.contains("[INST]") {
            return Some(ChatFormat::Llama2Inst);
        }
        if t.contains("<|user|>") {
            return Some(ChatFormat::Zephyr);
        }
    }
    match arch {
        Some("qwen2") | Some("qwen3") if has_token("<|im_start|>") => Some(ChatFormat::ChatMl),
        Some("llama") if has_token("<|start_header_id|>") => Some(ChatFormat::Llama3),
        _ => None,
    }
}

/// Single-turn prompt in `fmt`, ending where the assistant should speak.
pub fn render(fmt: ChatFormat, user: &str) -> Vec<Segment> {
    use Segment::*;
    match fmt {
        ChatFormat::ChatMl => vec![
            Special("<|im_start|>"),
            Text(format!("user\n{user}")),
            Special("<|im_end|>"),
            Text("\n".into()),
            Special("<|im_start|>"),
            Text("assistant\n".into()),
        ],
        ChatFormat::Llama3 => vec![
            Special("<|start_header_id|>"),
            Text("user".into()),
            Special("<|end_header_id|>"),
            Text(format!("\n\n{user}")),
            Special("<|eot_id|>"),
            Special("<|start_header_id|>"),
            Text("assistant".into()),
            Special("<|end_header_id|>"),
            Text("\n\n".into()),
        ],
        ChatFormat::Llama2Inst => vec![Text(format!("[INST] {user} [/INST]"))],
        ChatFormat::Zephyr => vec![
            Special("<|user|>"),
            Text(format!("\n{user}")),
            Special("</s>"),
            Text("\n".into()),
            Special("<|assistant|>"),
            Text("\n".into()),
        ],
    }
}

/// The token that ends the assistant turn in `fmt`.
pub fn stop_token(fmt: ChatFormat) -> &'static str {
    match fmt {
        ChatFormat::ChatMl => "<|im_end|>",
        ChatFormat::Llama3 => "<|eot_id|>",
        ChatFormat::Llama2Inst | ChatFormat::Zephyr => "</s>",
    }
}

/// Encode segments: specials map to their atomic vocab id when present
/// (falling back to plain text encoding — degraded but functional).
pub fn encode_segments(tok: &GgufTokenizer, segs: &[Segment]) -> Vec<usize> {
    let mut ids = Vec::new();
    for s in segs {
        match s {
            Segment::Special(p) => match tok.token_id(p) {
                Some(id) => ids.push(id),
                None => ids.extend(tok.encode(p)),
            },
            Segment::Text(t) => ids.extend(tok.encode(t)),
        }
    }
    ids
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(segs: &[Segment]) -> String {
        segs.iter()
            .map(|s| match s {
                Segment::Text(t) => t.as_str(),
                Segment::Special(p) => p,
            })
            .collect()
    }

    #[test]
    fn detect_fingerprints_template_string() {
        let t = |s: &str| detect(Some(s), None, |_| false);
        assert_eq!(t("{% if %}<|im_start|>…"), Some(ChatFormat::ChatMl));
        assert_eq!(t("…<|start_header_id|>…"), Some(ChatFormat::Llama3));
        assert_eq!(t("…[INST]…"), Some(ChatFormat::Llama2Inst));
        assert_eq!(t("…<|user|>…"), Some(ChatFormat::Zephyr));
        assert_eq!(t("unrecognized"), None);
    }

    #[test]
    fn detect_falls_back_by_arch_only_with_vocab_evidence() {
        assert_eq!(
            detect(None, Some("qwen2"), |p| p == "<|im_start|>"),
            Some(ChatFormat::ChatMl)
        );
        assert_eq!(detect(None, Some("qwen2"), |_| false), None); // base model: never wrap
        assert_eq!(
            detect(None, Some("llama"), |p| p == "<|start_header_id|>"),
            Some(ChatFormat::Llama3)
        );
        assert_eq!(detect(None, Some("llama"), |_| false), None);
        assert_eq!(detect(None, None, |_| true), None);
    }

    #[test]
    fn render_golden_strings() {
        assert_eq!(
            flat(&render(ChatFormat::ChatMl, "hi")),
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
        assert_eq!(
            flat(&render(ChatFormat::Llama3, "hi")),
            "<|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|>\
             <|start_header_id|>assistant<|end_header_id|>\n\n"
        );
        assert_eq!(flat(&render(ChatFormat::Llama2Inst, "hi")), "[INST] hi [/INST]");
        assert_eq!(
            flat(&render(ChatFormat::Zephyr, "hi")),
            "<|user|>\nhi</s>\n<|assistant|>\n"
        );
    }

    #[test]
    fn stop_tokens_per_format() {
        assert_eq!(stop_token(ChatFormat::ChatMl), "<|im_end|>");
        assert_eq!(stop_token(ChatFormat::Llama3), "<|eot_id|>");
        assert_eq!(stop_token(ChatFormat::Llama2Inst), "</s>");
        assert_eq!(stop_token(ChatFormat::Zephyr), "</s>");
    }
}
```

`encode_segments` needs a real `GgufTokenizer`: add its test next to the existing tokenizer-test
fixtures in gguf_tokenizer.rs's test mod (they already synthesize a Gguf) — build a vocab that
contains `<|im_start|>` as one piece plus ordinary tokens; assert the special encodes to exactly
one id, and that a special absent from vocab falls back to multi-id text encoding.

- [ ] **Step 3: multi-stop generate** — in llm.rs change `eos: Option<usize>` to `stops: &[usize]` in `generate`; body change: `if Some(next) == eos` → `if stops.contains(&next)`. Doc comment updated. Update every call site found by `rg -n "\.generate\(" --type rust` (CLI run-gguf, GUI gguf_run_inner, finetune sample paths if any, llm.rs tests): plain-eos callers pass `&eos.into_iter().collect::<Vec<_>>()` (bind the Vec to a local first).

- [ ] **Step 4: CLI wiring** — in `cmd_run_gguf`, add `"raw"` to the flag registry. After the tokenizer is built and when `--ids` was NOT used and a tokenizer exists:

```rust
let fmt = if args.has("raw") { None } else {
    ferrum_core::chat_format::detect(
        g.meta("tokenizer.chat_template").and_then(|v| v.as_str()),
        g.architecture(),
        |p| tok.as_ref().and_then(|t| t.token_id(p)).is_some(),
    )
};
```

When `fmt` is Some: `prompt_ids` = bos (as today) + `encode_segments(tok, &render(fmt, &prompt_text))`; print `  chat format = ChatML (pass --raw to disable)` (Debug name is fine). Build the stop set: eos id plus `tok.token_id(stop_token(fmt))` if present; pass to `generate`. When None: behavior identical to today. Usage strings gain `[--raw]`.

- [ ] **Step 5: GUI wiring** — `GgufRunParams` gains `#[serde(default)] pub raw: bool`. `gguf_run_inner` mirrors Step 4 exactly (same detect/render/stops logic; skipped when token-ids input was used). index.html: checkbox `ggChatFmt`, label "Chat format (auto)", checked by default, in the GGUF tab options row; app.js payload: `raw: !$("ggChatFmt").checked`.

- [ ] **Step 6: Verify + commit** — same battery as Task 1 (root + ferrum_gui + node). Commit: `feat(gguf): auto chat-template formatting for instruct models`.

---

### Task 3: F8 — Q2_K / Q3_K read support

**Files:**
- Modify: `ferrum_core/src/gguf.rs` only. Constants near line 49; `type_nbytes` arms near 334; `dequantize` dispatch + refusal near 547–571; tests near the other `dequant_*` block tests (~1534+). Also update the existing test asserting Q2_K/Q3_K are refused (search `"need their own decoders"`), and the GUI/CLI docs strings if they enumerate refused quants (leave docs to Task 9).

**Interfaces:**
- Produces: `GGML_Q2_K: u32 = 10`, `GGML_Q3_K: u32 = 11` handled by `type_nbytes` and `dequantize`. Refusal narrows to IQ*.

- [ ] **Step 1: constants + sizes** — beside the existing k-quant consts:

```rust
pub(crate) const GGML_Q2_K: u32 = 10;
pub(crate) const GGML_Q3_K: u32 = 11;
pub(crate) const Q2_K_BLOCK: usize = 16 + QK_K / 4 + 2 + 2; // scales, 2-bit qs, d, dmin = 84
pub(crate) const Q3_K_BLOCK: usize = QK_K / 8 + QK_K / 4 + 12 + 2; // hmask, qs, scales, d = 110
```

`type_nbytes` gains `GGML_Q2_K => mul(kblocks()?, Q2_K_BLOCK)` and `GGML_Q3_K => mul(kblocks()?, Q3_K_BLOCK)` arms (mirroring Q4_K's).

- [ ] **Step 2: decoders** — ggml-exact, next to `dequant_q4_k`:

```rust
/// Q2_K super-block: 16 packed 4-bit (scale, min) pairs, 64 bytes of 2-bit
/// quants, f16 d + f16 dmin. y = d·sc·q − dmin·m, in ggml's 128-element-half
/// scan order (shift 0,2,4,6 over two 16-element runs per shift).
fn dequant_q2_k(raw: &[u8], n: usize) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity(n);
    for block in raw.chunks_exact(Q2_K_BLOCK) {
        let scales = &block[..16];
        let qs = &block[16..16 + 64];
        let d = f16_to_f32(u16::from_le_bytes([block[80], block[81]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[82], block[83]]));
        let mut is = 0usize;
        for half in 0..2 {
            let q = &qs[32 * half..32 * half + 32];
            for shift in [0u8, 2, 4, 6] {
                for run in 0..2 {
                    let sc = scales[is];
                    is += 1;
                    let dl = d * (sc & 0xF) as f32;
                    let ml = dmin * (sc >> 4) as f32;
                    for l in 16 * run..16 * run + 16 {
                        out.push(dl * ((q[l] >> shift) & 3) as f32 - ml);
                    }
                }
            }
        }
    }
    out.truncate(n);
    if out.len() != n {
        return Err(fmt("Q2_K data shorter than expected"));
    }
    Ok(out)
}

/// Q3_K super-block: 32-byte high-bit mask, 64 bytes of 2-bit low quants,
/// 12 bytes of packed 6-bit scales, f16 d. q = low2 | (¬hbit → −4 offset),
/// y = d·(scale−32)·q, same scan order as Q2_K; the high-bit selector `m`
/// doubles after each shift group.
fn dequant_q3_k(raw: &[u8], n: usize) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity(n);
    for block in raw.chunks_exact(Q3_K_BLOCK) {
        let hmask = &block[..32];
        let qs = &block[32..32 + 64];
        let sc = &block[96..96 + 12];
        let d = f16_to_f32(u16::from_le_bytes([block[108], block[109]]));
        // Unpack 16 six-bit scales from ggml's split encoding.
        let mut scales = [0i8; 16];
        for j in 0..8 {
            scales[j] = (sc[j] & 0xF) as i8;
            scales[j + 8] = (sc[j] >> 4) as i8;
        }
        for j in 0..16 {
            let hi = (sc[8 + j / 4] >> (2 * (j % 4))) & 3;
            scales[j] |= (hi << 4) as i8;
        }
        let mut is = 0usize;
        let mut m: u8 = 1;
        for half in 0..2 {
            let q = &qs[32 * half..32 * half + 32];
            for shift in [0u8, 2, 4, 6] {
                for run in 0..2 {
                    let dl = d * (scales[is] as i32 - 32) as f32;
                    is += 1;
                    for l in 16 * run..16 * run + 16 {
                        let low = ((q[l] >> shift) & 3) as i32;
                        let qv = low - if hmask[l] & m != 0 { 0 } else { 4 };
                        out.push(dl * qv as f32);
                    }
                }
                m <<= 1;
            }
        }
    }
    out.truncate(n);
    if out.len() != n {
        return Err(fmt("Q3_K data shorter than expected"));
    }
    Ok(out)
}
```

**Layout warning (why the tests are the oracle):** the Q3_K scale unpack above follows ggml's
`kmask` scheme re-expressed per-nibble: scales j∈0..8 low nibbles of bytes 0–7, j∈8..16 high
nibbles, plus 2 top bits each from bytes 8–11 (j-th pair of bits of byte `8 + j/4`). If the
hand-crafted block tests disagree with the implementation, TRUST THE TEST VALUES (they are
computed from the ggml definition independently) and fix the decoder, not the tests. The Q3_K
high-bit convention: hbit **set** means NO −4 offset.

- [ ] **Step 3: dispatch + refusal** — `dequantize` gains `GGML_Q2_K => dequant_q2_k(raw, n)` / `GGML_Q3_K => dequant_q3_k(raw, n)` arms; the catch-all message becomes "(the IQ* formats need their own decoders)". Update the old refusal test to expect Q2_K acceptance + IQ refusal.

- [ ] **Step 4: tests** — hand-crafted blocks with exact expectations:

```rust
#[test]
fn q2k_q3k_block_sizes() {
    assert_eq!(type_nbytes(GGML_Q2_K, QK_K).unwrap(), Q2_K_BLOCK); // 84
    assert_eq!(type_nbytes(GGML_Q3_K, QK_K).unwrap(), Q3_K_BLOCK); // 110
    assert!(type_nbytes(GGML_Q2_K, 200).is_err());
}

#[test]
fn dequant_q2_k_scales_and_mins() {
    // qs = 0x55 → every 2-bit q = 1. d = 1.0, dmin = 0.5 (f16-exact).
    // scales[i] = i | ((15 - i) << 4) → y over group i = 1·i − 0.5·(15 − i).
    let mut b = vec![0u8; Q2_K_BLOCK];
    for i in 0..16 { b[i] = (i as u8) | (((15 - i) as u8) << 4); }
    for q in &mut b[16..80] { *q = 0x55; }
    b[80..82].copy_from_slice(&f32_to_f16_bits(1.0).to_le_bytes());
    b[82..84].copy_from_slice(&f32_to_f16_bits(0.5).to_le_bytes());
    let y = dequant_q2_k(&b, 256).unwrap();
    for (g, chunk) in y.chunks_exact(16).enumerate() {
        let expect = 1.0 * g as f32 - 0.5 * (15 - g) as f32;
        for v in chunk { assert!((v - expect).abs() < 1e-3, "group {g}: {v} vs {expect}"); }
    }
}

#[test]
fn dequant_q3_k_high_bit_offsets() {
    // qs = 0x55 (low bits = 1). All 16 scales = 34 → dl = d·2. d = 1.0.
    // hmask all 0xFF → q = 1 → y = 2. hmask all 0 → q = 1−4 = −3 → y = −6.
    let mut b = vec![0u8; Q3_K_BLOCK];
    for q in &mut b[32..96] { *q = 0x55; }
    // scales all = 34 = 0b100010: low nibble 2 in bytes 0–7 (both nibbles),
    // top bits 0b10 for every j in bytes 8–11 (0b10101010 = 0xAA).
    for j in 0..8 { b[96 + j] = 0x22; }
    for j in 8..12 { b[96 + j] = 0xAA; }
    b[108..110].copy_from_slice(&f32_to_f16_bits(1.0).to_le_bytes());
    for h in &mut b[..32] { *h = 0xFF; }
    let y = dequant_q3_k(&b, 256).unwrap();
    for v in &y { assert!((v - 2.0).abs() < 1e-3, "set-bit case: {v}"); }
    for h in &mut b[..32] { *h = 0; }
    let y = dequant_q3_k(&b, 256).unwrap();
    for v in &y { assert!((v + 6.0).abs() < 1e-3, "clear-bit case: {v}"); }
}
```

Use the file's existing f32→f16 helper if one exists in test scope (`gguf_write::f32_to_f16` is
pub — import it as `f32_to_f16_bits`); otherwise inline the standard conversion in the test mod.

- [ ] **Step 5: Verify + commit** — root `cargo test --workspace`. Commit: `feat(gguf): Q2_K and Q3_K dequantization (read support)`.

---

### Task 4: F3 — qwen3 architecture

**Files:**
- Modify: `ferrum_core/src/gguf.rs` (arch gates 677 + error strings; qk-norm tensor loading in `load_llama_prec` ~731; qwen3 synth fixture + tests ~1281+)
- Modify: `ferrum_core/src/llm.rs` (`Attention` fields + per-head norm application in `forward_one`:279 and `forward_full`:296; small pub(crate) helper + tests)
- Modify: `ferrum_core/src/gguf_write.rs` (arch gate 795; emit qk-norm tensors; round-trip test)

**Interfaces:**
- Consumes: `RmsNorm` (llm.rs:39, has `forward_row(&self, x, out)`), `Attention::new(wq,wk,wv,wo,n_heads,n_kv,head_dim,rope_dim,rope_base,rope_type)` (llm.rs:168).
- Produces: `Attention::set_qk_norm(&mut self, q: RmsNorm, k: RmsNorm)`; `Attention.q_norm/k_norm: Option<RmsNorm>` (pub(crate) or accessor for the writer); arch gates accept `"qwen3"`.

- [ ] **Step 1: Attention** — add `q_norm: Option<RmsNorm>, k_norm: Option<RmsNorm>` fields (init `None` in `new` — do NOT widen `new`'s signature) plus:

```rust
/// Qwen3-style per-head RMS norms applied to q/k before RoPE.
pub fn set_qk_norm(&mut self, q: RmsNorm, k: RmsNorm) {
    self.q_norm = Some(q);
    self.k_norm = Some(k);
}
```

Add a free helper + apply it in BOTH `forward_one` and `forward_full` at the point where the raw
q and k projections exist but BEFORE `apply_rope` (read the two functions; q is `n_heads ×
head_dim` contiguous, k is `n_kv × head_dim`):

```rust
/// Apply `norm` (weight length = head_dim) to each `head_dim` chunk in place.
pub(crate) fn norm_per_head(x: &mut [f32], head_dim: usize, norm: &RmsNorm) {
    let mut tmp = vec![0.0f32; head_dim];
    for chunk in x.chunks_exact_mut(head_dim) {
        norm.forward_row(chunk, &mut tmp);
        chunk.copy_from_slice(&tmp);
    }
}
```

In the forward paths: `if let Some(n) = &self.q_norm { norm_per_head(&mut q, self.head_dim, n); }`
(same for k with `k_norm`) — adapt variable names to the actual code. Unit tests in llm.rs:

```rust
#[test]
fn norm_per_head_matches_rmsnorm_per_chunk() {
    let norm = RmsNorm::new(vec![2.0, 0.5], 1e-6);
    let mut x = vec![1.0, -2.0, 3.0, 0.5]; // two heads of dim 2
    let expected: Vec<f32> = {
        let mut e = Vec::new();
        for c in x.chunks_exact(2) {
            let mut o = vec![0.0; 2];
            norm.forward_row(c, &mut o);
            e.extend(o);
        }
        e
    };
    norm_per_head(&mut x, 2, &norm);
    assert_eq!(x, expected.as_slice());
}

#[test]
fn qk_norm_changes_attention_output() {
    // Build any small Attention (reuse the test-mod builders), run forward_one
    // on a fixed input twice — with and without a non-unit qk norm — and
    // assert the outputs differ (proves the hook is live in the graph).
}
```

- [ ] **Step 2: loader** — in gguf.rs change both `load_llama_prec` gate + message to accept `llama | qwen2 | qwen3`. After `let mut attn = Attention::new(...)` (make it `mut`), load the norms presence-driven:

```rust
let qn = format!("{p}.attn_q_norm.weight");
let kn = format!("{p}.attn_k_norm.weight");
if self.tensor(&qn).is_some() && self.tensor(&kn).is_some() {
    attn.set_qk_norm(
        RmsNorm::new(self.dequant_named(&qn)?, eps),
        RmsNorm::new(self.dequant_named(&kn)?, eps),
    );
}
```

(head_dim from `{arch}.attention.key_length` is ALREADY honored at gguf.rs:689 — no change.
RoPE stays `RopeType::Norm`, mirroring the qwen2 path.) Note: `param_data_ref`
(llm_train.rs:261) does not include the norm weights, so fine-tuning leaves them frozen — add
one comment line there saying so.

- [ ] **Step 3: fixture + loader tests** — in gguf.rs tests add `synth_qwen3()`: copy the
`synth_llama_cfg` pattern with arch `"qwen3"`, per-layer extra tensors
`blk.{i}.attn_q_norm.weight` / `attn_k_norm.weight` = `[head_dim]` of value `2.0` (a non-unit
weight so the norm is observable), plus TWO extra metadata keys `qwen3.attention.key_length` and
`qwen3.attention.value_length` = head_dim (bump the hardcoded `n_meta` count accordingly).
Tests: parses + `load_llama_prec(None)` succeeds; `cfg.head_dim` equals the explicit key;
`generate` on it runs deterministically; `"gemma2"` arch still rejected (adapt the existing
unsupported-arch test to also try gemma2).

- [ ] **Step 4: writer** — gguf_write.rs gate accepts qwen3 (message text updated). In the
per-block loop of `llama_gguf_bytes` (gguf_write.rs:837), right after the `attn_norm` emission,
emit the norms when present:

```rust
if let (Some(qn), Some(kn)) = (&blk.attn.q_norm, &blk.attn.k_norm) {
    add_norm(&mut b, &format!("{p}.attn_q_norm.weight"), &qn.weight);
    add_norm(&mut b, &format!("{p}.attn_k_norm.weight"), &kn.weight);
}
```

(Expose the fields pub(crate) or add accessors — match the file's style.) Round-trip test in
gguf_write.rs: load `synth_qwen3` f32 → `llama_gguf_bytes(F32)` → reparse → both norm tensors
present per layer with values == 2.0.

- [ ] **Step 5: Verify + commit** — root battery. Commit: `feat(gguf): qwen3 architecture (qk-norm) across import, run, export`.

---

### Task 5: F7 — perplexity eval for GGUF models

**Files:**
- Modify: `ferrum_core/src/llm.rs` (add `PplStats` + `LlamaModel::eval_perplexity`), `ferrum_core/src/gguf.rs` (tests — the fixture lives there)
- Modify: `slm_cli/src/main.rs` (new `eval-gguf` command + dispatch + usage)
- Modify: `ferrum_gui/src/commands.rs` (new `evaluate_gguf` command), `ferrum_gui/src/lib.rs` or wherever `tauri::generate_handler![...]` lists commands (add it), `ferrum_gui/ui/app.js` (route `.gguf` model paths in the Evaluate handler), `ferrum_gui/ui/index.html` (Evaluate hint text mentions .gguf)

**Interfaces:**
- Consumes: `LlamaModel::forward_tokens(&self, tokens: &[usize]) -> Result<Tensor>` (llm.rs:500, `[seq, vocab]` logits), `GgufTokenizer::encode`, existing `EvalRow` struct in commands.rs (~455 region).
- Produces: `pub struct PplStats { pub cross_entropy: f64, pub bits_per_token: f64, pub perplexity: f64, pub n_predictions: usize }`; `LlamaModel::eval_perplexity(&self, ids: &[usize]) -> Result<PplStats>`; CLI `eval-gguf <model.gguf> <heldout.txt> [--quant int4|int8|f32]`; Tauri command `evaluate_gguf`.

- [ ] **Step 1: engine** — in llm.rs:

```rust
/// Teacher-forced evaluation statistics from [`LlamaModel::eval_perplexity`].
pub struct PplStats {
    pub cross_entropy: f64,
    pub bits_per_token: f64,
    pub perplexity: f64,
    pub n_predictions: usize,
}

impl LlamaModel {
    /// Perplexity of `ids` under this model: consecutive `context_len` windows
    /// (each overlapping the previous by one token, so every next-token in
    /// `ids` is predicted exactly once — window-start predictions see short
    /// context; simple and stated honestly in the docs).
    pub fn eval_perplexity(&self, ids: &[usize]) -> Result<PplStats> {
        if ids.len() < 2 {
            return Err(InferError::DimMismatch(
                "need at least 2 tokens to evaluate".into(),
            ));
        }
        let v = self.cfg.vocab_size;
        if let Some(&bad) = ids.iter().find(|&&t| t >= v) {
            return Err(InferError::DimMismatch(format!(
                "token {bad} ≥ vocab_size {v}"
            )));
        }
        let ctx = self.cfg.context_len.max(2);
        let (mut nll, mut n) = (0.0f64, 0usize);
        let mut start = 0usize;
        while start + 1 < ids.len() {
            let end = (start + ctx).min(ids.len());
            let window = &ids[start..end];
            let logits = self.forward_tokens(window)?;
            for t in 0..window.len() - 1 {
                let row = &logits.data[t * v..(t + 1) * v];
                let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let lse = max as f64
                    + row.iter().map(|&l| ((l - max) as f64).exp()).sum::<f64>().ln();
                nll += lse - row[window[t + 1]] as f64;
                n += 1;
            }
            start = end - 1;
        }
        let ce = nll / n as f64;
        Ok(PplStats {
            cross_entropy: ce,
            bits_per_token: ce / std::f64::consts::LN_2,
            perplexity: ce.exp(),
            n_predictions: n,
        })
    }
}
```

- [ ] **Step 2: engine tests** — in gguf.rs's test mod (fixture home):

```rust
#[test]
fn eval_perplexity_uniform_oracle_and_window_count() {
    let g = Gguf::parse(synth_llama()).unwrap();
    let mut m = g.load_llama_prec(None).unwrap();
    // Zero LM head ⇒ logits all zero ⇒ exactly uniform ⇒ ppl == vocab (10).
    for w in &mut m.lm_head.weight.data { *w = 0.0; }
    let ids = vec![1usize; 40]; // > ctx? fixture ctx = 32 → two windows
    let s = m.eval_perplexity(&ids).unwrap();
    assert_eq!(s.n_predictions, 39); // every next-token predicted exactly once
    assert!((s.perplexity - 10.0).abs() < 1e-3, "ppl {}", s.perplexity);
    assert!((s.cross_entropy - (10.0f64).ln()).abs() < 1e-6);
    // Biasing the true next token must strictly lower perplexity.
    m.lm_head.bias.data[1] = 5.0;
    assert!(m.eval_perplexity(&ids).unwrap().perplexity < 10.0);
    // Degenerate input errors.
    assert!(m.eval_perplexity(&[1]).is_err());
}
```

(If `lm_head.bias` is not directly settable, use any equivalent field access — the struct
fields are pub per llm.rs:454.)

- [ ] **Step 3: CLI** — new `cmd_eval_gguf` mirroring `cmd_run_gguf`'s skeleton (open streamed,
memory guard with the same 90% rule, `--quant` default int4, `--force`): read the held-out file
(`read_corpus` like `cmd_eval`), require a tokenizer in the file (error otherwise), `ids =
tok.encode(&text)` (no bos), call `eval_perplexity`, print perplexity / cross-entropy /
bits-per-token / n_predictions and `uniform baseline = vocab_size`. Register `"eval-gguf"` in
the command dispatch next to `"eval"` and extend the usage/help text.

- [ ] **Step 4: GUI** — new command in commands.rs:

```rust
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GgufEvalParams {
    pub model_path: String,
    pub text: Option<String>,
    pub text_path: Option<String>,
    pub quant: String,
    #[serde(default)]
    pub force: bool,
}
```

`evaluate_gguf(params) -> Result<EvalRow, String>` via `spawn_blocking` like `run_gguf`: resolve
text (inline wins, else read `text_path`, error if both empty — same rule as `evaluate_slm`),
open + guard + `load_llama_prec(parse_quant(&params.quant)?)`, tokenizer required, encode, eval,
map into the existing `EvalRow` (model = file name, uniform_baseline = vocab_size as f64).
Register in the `generate_handler![]` list. app.js: in the Evaluate run handler, if the model
path ends with `.gguf` (case-insensitive) invoke `evaluate_gguf` with `{modelPath, text,
textPath, quant: "int4", force: false}` and reuse the same row-append code; index.html Evaluate
hint text: "…works with .bin models and imported .gguf checkpoints (int4)".

- [ ] **Step 5: Verify + commit** — full battery incl. ferrum_gui + node. Commit: `feat(eval): perplexity evaluation for GGUF models (engine, CLI, GUI)`.

---

### Task 6: F5 — streamed export

**Files:**
- Create: `ferrum_core/src/gguf_stream_write.rs` (+ `pub mod` in lib.rs)
- Modify: `ferrum_core/src/gguf_write.rs` (make `put_u32/put_u64/put_str/put_value` and the alignment const pub(crate) if not already; factor the per-tensor type decision into a pub(crate) `planned_type(dims: &[u64], quant: GgufQuant) -> u32` used by BOTH paths — read `add_tensor`/`add_linear`/`add_weight_2d` (~700–790) first and extract exactly the decision they make, including the k-quant block-alignment fallback to F16 and norms/biases staying F32)
- Modify: `slm_cli/src/main.rs` (`cmd_export_gguf` switches to `stream_export`; guard formula), `ferrum_gui/src/commands.rs` (`do_export`/`export_gguf` likewise; keep the phase events, mapping opening→planning→writing)

**Interfaces:**
- Consumes: `Gguf::open` (streamed header parse, gguf.rs:441 region), `Gguf::dequantize`, `encode_tensor(data, ggml_type)` (gguf_write.rs:143), `transpose_2d` (gguf.rs, used by `linear_from`:640), the FLCK v1 layout (llm_train.rs:1384: magic "FLCK", u32 version 1, seven u32 shape fields `[vocab, dim, n_layers, n_heads, n_kv_heads, head_dim, ffn_dim]`, u8 qat, u64 step, u64 rng, then three flat f32 sections — weights, m, v — in `param_data_ref` order, llm_train.rs:261).
- Produces: `pub fn stream_export(source_path: &str, out_path: &str, quant: GgufQuant, checkpoint: Option<&str>) -> Result<ExportSummary>` where `ExportSummary` is whatever per-type summary type the CLI/GUI already consume from the in-memory path (reuse it; if it is currently computed CLI-side from builder output, return `Vec<(u32 /*ggml_type*/, usize /*count*/)>` and adapt both consumers).

**The invariant that makes this reviewable:** for every supported output quant, `stream_export`'s
output file must be **byte-identical** to `llama_gguf_bytes(load_llama_prec(None)?, source,
quant)` — with and without a checkpoint. Any deviation is a bug in the streamed path.

- [ ] **Step 1: read first** — `GgufBuilder::into_bytes` (gguf_write.rs:614–660) for the exact
header/meta/table/alignment layout to replicate; `add_tensor`/`add_linear`/`add_norm`/
`add_weight_2d` for the emission ORDER, names, dims, bias-presence rule (bias tensors are
emitted only when the source has them), and per-tensor type policy; `llama_gguf_bytes` (790–886)
for the metadata copy rules (skip `general.file_type`, `general.quantization_version`,
`general.alignment`; append refreshed file_type + quantization_version LAST, where file_type
falls back to F16's when no tensor of the requested type was emitted).

- [ ] **Step 2: pass 1 (plan)** — from the source's tensor table + config, build
`Vec<Planned { name: String, dims: Vec<u64>, gtype: u32, src: SrcKind }>` in EXACTLY
`llama_gguf_bytes`' emission order (token_embd; per block: attn_norm, [attn_q_norm +
attn_k_norm when present — Task 4's position], attn_q.weight (+`.bias` if source has),
attn_k…, attn_v…, attn_output, ffn_norm, ffn_gate, ffn_up, ffn_down; output_norm;
output.weight only if the source has it). `gtype` via the factored `planned_type`. Serialize
header (magic, v3, counts), metadata (source order, skip-3, append-2 — computing
`used_requested` from the planned types), tensor infos with running 32-aligned offsets — all
with the same put_* helpers the builder uses. Write to `{out}.tmp`.

- [ ] **Step 3: pass 2 (stream)** — per planned tensor: read the source tensor's raw bytes
(seek + read, the streamed reader path), `dequantize` to f32, optionally overlay checkpoint
weights (Step 4), apply the SAME data transform the in-memory path applies (2-D weights:
`linear_from` transposed `[n_out,n_in] → [n_in,n_out]` on load, and `add_linear`/`add_weight_2d`
transpose back on write — so for a **non-checkpoint** export the net transform is identity:
encode the dequantized source data as-is; for checkpoint data, which lives in the in-memory
`[n_in,n_out]` layout, transpose to `[n_out,n_in]` first; token_embd and norms have no
transpose either way), `encode_tensor(&f32, gtype)`, write + zero-pad to 32. Then
`std::fs::rename(tmp, out)` (mirror `atomic_write`'s cleanup-on-error).

- [ ] **Step 4: checkpoint overlay** — parse the FLCK header; verify magic/version and the
seven shape fields against the source's config (error on mismatch, same message spirit as
`load_checkpoint_into`); weights section starts at byte 4+4+28+1+8+8 = 53. Build the canonical
span table (offset, len in f32s) walking `param_data_ref` order with lengths derived from
config — per block: attn_norm `dim`; wq `dim·qd` + bias `qd`; wk `dim·kvd` + bias `kvd`; wv
`dim·kvd` + bias `kvd`; wo `qd·dim` + bias `dim`; ffn_norm `dim`; gate `dim·ffn` + bias `ffn`;
up `dim·ffn` + bias `ffn`; down `ffn·dim` + bias `dim`; prefixed by tok_emb `vocab·dim` and
suffixed by final_norm `dim`, lm_head `dim·vocab` + bias `vocab` (qd = n_heads·head_dim, kvd =
n_kv·head_dim; bias spans exist even for absent biases — they are zero-filled in memory,
llm_train.rs:266ff + gguf.rs:643). For each emitted tensor read ONLY its span (seek), transpose
2-D weights to GGUF layout, encode. The Adam m/v sections are never read. Tied-head sources
(no `output.weight`): the lm_head span exists in the checkpoint but is simply never emitted —
token_embd gets ITS OWN span's data, exactly like the in-memory path (`load_checkpoint_into`
then export reads `model.tok_emb`).

- [ ] **Step 5: oracle tests** (in gguf_stream_write.rs, reusing gguf_write.rs's test fixture
machinery — it already synthesizes llama GGUFs for its round-trip tests; if it's file-private,
lift the fixture fn to `pub(crate)` under `#[cfg(test)]`):

```rust
#[test]
fn stream_export_is_byte_identical_to_in_memory_path() {
    for quant in [GgufQuant::F32, GgufQuant::F16, GgufQuant::Q8_0, GgufQuant::Q4_0, GgufQuant::Q4K, GgufQuant::Q6K] {
        // (use the enum's real variant names)
        let src_bytes = /* fixture */;
        let dir = std::env::temp_dir().join(format!("ferrum_stream_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("src.gguf"); std::fs::write(&src, &src_bytes).unwrap();
        let g = Gguf::parse(src_bytes.clone()).unwrap();
        let expect = llama_gguf_bytes(&g.load_llama_prec(None).unwrap(), &g, quant).unwrap();
        let out = dir.join("out.gguf");
        stream_export(src.to_str().unwrap(), out.to_str().unwrap(), quant, None).unwrap();
        assert_eq!(std::fs::read(&out).unwrap(), expect, "quant {:?}", quant);
    }
}

#[test]
fn stream_export_applies_checkpoint_byte_identically() {
    // Build a synthetic FLCK: header from the fixture's config, weights section
    // = in-memory param flats + 1.0 elementwise, m/v sections zeroed.
    // In-memory: load f32 → LlamaTrainer::new → load_checkpoint_into → export.
    // Streamed: stream_export(…, Some(ckpt_path)).
    // Assert byte equality AND that the output differs from the no-checkpoint export.
}
```

Plus: reparse the streamed output with `Gguf::parse` (offsets/alignment valid); shape-mismatch
checkpoint rejected; summary counts equal between paths.

- [ ] **Step 6: switch CLI + GUI** — `cmd_export_gguf` and the GUI `do_export` call
`stream_export` instead of load+`write_llama_gguf`. Replace the RAM-guard estimate with the
streamed peak: `3 × 4 × (largest tensor numel)` bytes (source f32 + encode buffer + slack),
same 90%/force rule; put the formula in one small shared-in-crate helper if both binaries can
reach it, else duplicate the one-line computation with a comment naming the other copy. Keep
CLI output and GUI phase events working (phases: opening → writing; drop the "applying" phase
or emit it when a checkpoint is set before pass 2 — match what app.js listens for and keep it
listening-compatible).

- [ ] **Step 7: Verify + commit** — full battery. Commit: `feat(export): streamed GGUF export — RAM peak drops to one tensor`.

---

### Task 7: F6 — safe-SIMD decode restructure

**Files:**
- Modify: `ferrum_core/src/quant.rs` and/or `ferrum_core/src/ops.rs` — wherever `qaccum_cols` lives (`rg -n "fn qaccum_cols"`). Tests + `#[ignore]` bench in the same file's test mod.

**Interfaces:**
- Consumes: `QWeight { rows, cols, kind, scales /*per row*/, packed }` (quant.rs:130; int8 = `rows×cols` bytes row-major; int4 = split-half nibble layout documented at quant.rs:136–140), `qlinear` (ops.rs:207) which pre-folds `scales[p]` into the activations and calls `qaccum_cols(a, w, j0, j1, out)`.
- Produces: same `qaccum_cols` signature, restructured. **Bit-identical results required.**

**The hard constraint:** each output element's reduction order over `p` (the k dimension) must
stay ascending-sequential — vectorize ACROSS OUTPUT COLUMNS `j` (lane per column), never split
the p-reduction into partial accumulators. That keeps f32 results bit-identical to today (and to
the reference below), so the fixed-seed generation tests elsewhere in the suite cannot flip.

- [ ] **Step 1: read** the current `qaccum_cols` and the int4/int8 decode it does.

- [ ] **Step 2: restructure** for autovectorization, keeping semantics:
  - int8: for each `p`, the row slice `&qs[p*cols + j0 .. p*cols + j1]` is unit-stride; write the inner loop as `for (o, &q) in out.iter_mut().zip(row) { *o += a_p * (q as i8 as f32); }` over exact-size slices (no indexed access, no bounds checks in the loop).
  - int4: within a row, the split-half layout gives two unit-stride column ranges (low nibbles → cols `0..half`, high nibbles → cols `half..cols`); process each range as its own `zip`ped unit-stride pass with branch-free nibble extraction (`(b & 0xF)` / `(b >> 4)`, then `- 8`).
  - Hoist per-row constants; use `chunks_exact`/`split_at` so LLVM sees exact trip counts. No `unsafe`, no feature detection.

- [ ] **Step 3: reference + tests** in the test mod:

```rust
/// Straightforward per-element reference with the same p-ascending reduction
/// order — results must be BIT-identical to qaccum_cols.
fn qaccum_ref(a: &[f32], w: &QWeight, j0: usize, j1: usize, out: &mut [f32]) {
    for (idx, j) in (j0..j1).enumerate() {
        let mut acc = out[idx];
        for p in 0..w.rows {
            let q = match w.kind {
                QKind::Int8 => w.packed[p * w.cols + j] as i8 as f32,
                QKind::Int4 => {
                    let half = w.cols.div_ceil(2);
                    let b = w.packed[p * half + (j % half)];
                    let nib = if j < half { b & 0xF } else { b >> 4 };
                    (nib as i32 - 8) as f32
                }
            };
            acc += a[p] * q;
        }
        out[idx] = acc;
    }
}
```

(Adapt field/`QKind` names and the packed-byte indexing to the real `QWeight` — the reference
MUST match the documented layout at quant.rs:136–140; if the doc and code disagree, stop and
report.) Tests: for shapes `(rows, cols)` in `[(8, 8), (17, 33), (64, 1), (33, 64), (1, 7)]`,
both kinds, xorshift-random weights/activations: quantize via the crate's own QWeight
constructor, run old-path (`qaccum_ref`) vs `qaccum_cols` over full and partial `[j0, j1)`
ranges, `assert_eq!` on the f32 **bit patterns** (`to_bits()`), not approximate equality. Plus
`#[test] #[ignore] fn bench_qaccum()` timing 512×512 and 1024×1024 int4+int8 single-token
qlinear loops, printing GFLOP/s (run manually via `cargo test --release bench_qaccum -- --ignored --nocapture`).

- [ ] **Step 4: measure + report honestly** — run the ignored bench before and after the
restructure (before = git stash of the change or a first run on the parent commit) and record
both numbers in the commit message. If the gain is under 5%, still commit the cleanup IF results
are bit-identical and the code is no less clear; report the number to the controller either way.

- [ ] **Step 5: Verify + commit** — root battery (all fixed-seed tests must pass unchanged). Commit: `perf(quant): autovectorization-friendly qaccum_cols (safe Rust, bit-identical)` including the measured before/after GFLOP/s.

---

### Task 8: F4 — CI job for ferrum_gui

**Files:**
- Modify: `.github/workflows/ci.yml` (append job `gui` after `lint`)

- [ ] **Step 1: append the job** (verify first whether `ferrum_gui/Cargo.lock` exists — if not, key the cache on `ferrum_gui/Cargo.toml`):

```yaml
  gui:
    name: GUI crate (ferrum_gui)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: clippy, rustfmt
      - name: System WebView dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y libwebkit2gtk-4.1-dev libgtk-3-dev \
            libayatana-appindicator3-dev librsvg2-dev build-essential \
            libssl-dev xvfb
      - uses: actions/cache@v4
        with:
          path: |
            ~/.cargo/registry
            ~/.cargo/git
            ferrum_gui/target
          key: ${{ runner.os }}-cargo-gui-${{ hashFiles('ferrum_gui/Cargo.lock') }}
      - name: Format, lint, test
        working-directory: ferrum_gui
        run: |
          cargo fmt --check
          cargo clippy --all-targets -- -D warnings
          cargo test
      - name: JS unit test
        run: node ferrum_gui/ui/stream.test.js
      - name: Boot smoke (xvfb, 20s survival = pass)
        working-directory: ferrum_gui
        run: |
          cargo build
          set +e
          xvfb-run -a timeout 20 ./target/debug/ferrum_gui
          code=$?
          set -e
          if [ "$code" -ne 124 ]; then
            echo "boot smoke failed: exit $code (wanted 124 = alive at timeout)"
            exit 1
          fi
```

- [ ] **Step 2: verify + commit** — `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"` parses; `git diff` shows only the appended job. Commit: `ci: build, lint, test, and boot-smoke ferrum_gui`. (The live run is watched in Task 10 after push.)

---

### Task 9: Docs

**Files:**
- Modify: `manual/05-using-the-gui.md` (GGUF tab: three sampling knobs + "Chat format (auto)" checkbox and what it does incl. `--raw` equivalence; Evaluate tab: accepts `.gguf`; Export: RAM note now "streams tensor-by-tensor"; §5.2/CI note if it mentions tests), `howtouse.md` / `readme.md` / `FAQs.md` / `usecases.md` / `manual/*` wherever they: enumerate `run-gguf` options (add `--top-k/--top-p/--rep/--raw`), list supported architectures (add qwen3), list refused quants (Q2_K/Q3_K move to supported-read; IQ* stays refused), describe export RAM behavior, or enumerate CLI commands (add `eval-gguf`). Search terms: `rg -n "Q3_K|qwen2|run-gguf|IQ\*" *.md manual/ docs/ --glob '!docs/superpowers/**'`.
- Confirm no doc claims are weakened: the zero-unsafe claim REMAINS TRUE (grep `unsafe` in the md set to confirm wording still holds).

- [ ] **Step 1:** make the edits; keep each doc's voice (the manual is beginner-warm, README is terse).
- [ ] **Step 2:** `rg` the stale strings again to prove zero leftovers; commit: `docs: eight-upgrades — new knobs, qwen3, Q2/Q3_K read, streamed export, eval-gguf`.

---

### Task 10: Verify, review, finish (controller-run)

- [ ] Full battery: root `cargo test --workspace` (expect 20+ suites, 0 failures); `ferrum_gui/`: fmt + clippy + `cargo test`; `node --check` app.js/stream.js + `node ui/stream.test.js`; snap-scrubbed windowed boot smoke (env -u LD_LIBRARY_PATH -u GTK_PATH -u GIO_MODULE_DIR -u GTK_EXE_PREFIX -u GDK_PIXBUF_MODULE_FILE -u LOCPATH, timeout 20, exit 124 + 0 panics); `--ignored` bench run for the Task 7 numbers.
- [ ] Whole-branch opus review (review-package master..HEAD), adjudicate findings (Critical/Important fixed before merge; plan-mandated findings go to the human).
- [ ] Finish per the standing pattern: fast-forward merge to master, push, delete branch, then `gh run watch` the CI run to see the new `gui` job pass live.
