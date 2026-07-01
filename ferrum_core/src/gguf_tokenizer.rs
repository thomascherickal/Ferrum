//! Import a GGUF checkpoint's **own** tokenizer (G-T) so an imported
//! [`crate::llm::LlamaModel`] can be driven with *text* rather than raw token
//! IDs. A GGUF file stores its vocabulary, merges, and special-token IDs in the
//! `tokenizer.ggml.*` metadata; this module parses that table with zero
//! dependencies and offers `encode` / `decode`.
//!
//! Two tokenizer families cover essentially all current GGUF SLMs:
//!
//! * **BPE / `gpt2`** (Llama-3, Qwen2, GPT-2, Mistral-bpe): byte-level
//!   byte-pair encoding. `encode` is the real merge-rank BPE; `decode` is exact.
//! * **SPM / `llama`** (Llama-1/2, many SentencePiece models): `decode` is exact
//!   (handles the `▁` space marker and `<0xXX>` byte-fallback tokens); `encode`
//!   is a **greedy longest-match approximation** of SentencePiece's unigram
//!   Viterbi search — good enough to prompt a model, but **not guaranteed
//!   token-for-token identical** to the reference. The byte fallback keeps it
//!   total (any input encodes to *something* valid).
//!
//! The BPE pre-tokenizer here is a dependency-free approximation of GPT-2's
//! regex (it has no `regex` crate): it groups letters, digits, and symbol runs
//! and attaches a single leading space to the following word. For typical
//! prompts this matches; for exotic whitespace it can differ. Where exactness
//! matters, the CLI also accepts explicit token IDs (`--ids`).

use crate::error::{InferError, Result};
use crate::gguf::{Gguf, MetaValue};
use std::collections::HashMap;

/// SentencePiece's visible space marker (U+2581, "▁").
const SPM_SPACE: char = '\u{2581}';

fn fmt(msg: &str) -> InferError {
    InferError::Format(msg.to_string())
}

/// Which tokenizer family a GGUF declares in `tokenizer.ggml.model`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokModel {
    /// Byte-level BPE (`gpt2` / `bpe`): Llama-3, Qwen2, GPT-2, …
    Bpe,
    /// SentencePiece unigram (`llama` / `spm`): Llama-1/2, …
    Spm,
}

/// A tokenizer reconstructed from a GGUF metadata table.
pub struct GgufTokenizer {
    model: TokModel,
    /// id → token piece, exactly as stored in the GGUF.
    tokens: Vec<String>,
    token_to_id: HashMap<String, u32>,
    /// BPE merge ranks keyed by the `left\0right` piece pair (lower = earlier).
    /// A single joined `String` key (with a NUL separator, which never occurs in
    /// a GPT-2 byte-encoded piece) lets the hot merge scan look up pairs through
    /// a reused buffer with **zero per-pair allocation** (`get(&str)` via
    /// `Borrow`), instead of cloning both pieces into a `(String, String)` key.
    merges: HashMap<String, u32>,
    /// Longest vocabulary token in **chars** — bounds the SPM longest-match
    /// window so `encode_spm` is O(n·max_token_len), not O(n³).
    max_token_len: usize,
    /// SPM `<0xXX>` byte-fallback tokens: raw byte → token id.
    byte_to_id: HashMap<u8, u32>,
    /// GPT-2 byte ↔ printable-unicode tables (BPE only).
    byte_encoder: [char; 256],
    byte_decoder: HashMap<char, u8>,
    bos: Option<u32>,
    eos: Option<u32>,
}

/// Build GPT-2's byte→unicode table (and its inverse). Printable byte ranges map
/// to themselves; the rest map to code points ≥ 256, so every byte is a
/// printable, single-`char` token — the trick that makes byte-level BPE total.
fn gpt2_byte_tables() -> ([char; 256], HashMap<char, u8>) {
    let mut bs: Vec<u8> = Vec::new();
    for b in b'!'..=b'~' {
        bs.push(b);
    }
    for b in 0xA1u8..=0xAC {
        bs.push(b);
    }
    for b in 0xAEu8..=0xFF {
        bs.push(b);
    }
    let mut cs: Vec<u32> = bs.iter().map(|&b| b as u32).collect();
    let mut n: u32 = 0;
    for b in 0u16..256 {
        let b = b as u8;
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut enc = ['\0'; 256];
    for (&b, &c) in bs.iter().zip(cs.iter()) {
        enc[b as usize] = char::from_u32(c).unwrap_or('\u{FFFD}');
    }
    let mut dec = HashMap::with_capacity(256);
    for (b, &c) in enc.iter().enumerate() {
        dec.insert(c, b as u8);
    }
    (enc, dec)
}

/// Parse a SentencePiece byte-fallback token `<0xHH>` into its raw byte.
fn parse_byte_token(tok: &str) -> Option<u8> {
    let hex = tok.strip_prefix("<0x")?.strip_suffix('>')?;
    u8::from_str_radix(hex, 16).ok()
}

fn string_array(g: &Gguf, key: &str) -> Option<Vec<String>> {
    match g.meta(key) {
        Some(MetaValue::Array(items)) => Some(
            items
                .iter()
                .map(|v| v.as_str().unwrap_or("").to_string())
                .collect(),
        ),
        _ => None,
    }
}

impl GgufTokenizer {
    /// Reconstruct the tokenizer from a parsed GGUF's metadata. Errors if the
    /// file declares no tokenizer or an unsupported family.
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let model_str = g
            .meta("tokenizer.ggml.model")
            .and_then(|v| v.as_str())
            .ok_or_else(|| fmt("GGUF has no tokenizer.ggml.model metadata"))?;
        let model = match model_str {
            "gpt2" | "bpe" => TokModel::Bpe,
            "llama" | "spm" => TokModel::Spm,
            other => {
                return Err(fmt(&format!(
                    "unsupported GGUF tokenizer model '{other}' (have: gpt2/bpe, llama/spm)"
                )))
            }
        };

        let tokens = string_array(g, "tokenizer.ggml.tokens")
            .ok_or_else(|| fmt("GGUF missing tokenizer.ggml.tokens"))?;
        if tokens.is_empty() {
            return Err(fmt("GGUF tokenizer.ggml.tokens is empty"));
        }
        let mut token_to_id = HashMap::with_capacity(tokens.len());
        let mut byte_to_id = HashMap::new();
        for (id, tok) in tokens.iter().enumerate() {
            token_to_id.entry(tok.clone()).or_insert(id as u32);
            if let Some(b) = parse_byte_token(tok) {
                byte_to_id.insert(b, id as u32);
            }
        }

        // BPE merges: "left right" piece pairs, in priority order. Stored under a
        // `left\0right` joined key (see the `merges` field docs).
        let mut merges = HashMap::new();
        if let Some(pairs) = string_array(g, "tokenizer.ggml.merges") {
            for (rank, m) in pairs.iter().enumerate() {
                if let Some(sp) = m.find(' ') {
                    let key = format!("{}\u{0}{}", &m[..sp], &m[sp + 1..]);
                    merges.entry(key).or_insert(rank as u32);
                }
            }
        }

        // Longest token (in chars) for the SPM longest-match bound.
        let max_token_len = tokens
            .iter()
            .map(|t| t.chars().count())
            .max()
            .unwrap_or(1)
            .max(1);

        let bos = g
            .meta("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.as_usize())
            .map(|u| u as u32);
        let eos = g
            .meta("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.as_usize())
            .map(|u| u as u32);

        let (byte_encoder, byte_decoder) = gpt2_byte_tables();
        Ok(Self {
            model,
            tokens,
            token_to_id,
            merges,
            max_token_len,
            byte_to_id,
            byte_encoder,
            byte_decoder,
            bos,
            eos,
        })
    }

    pub fn model(&self) -> TokModel {
        self.model
    }
    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }
    pub fn bos(&self) -> Option<usize> {
        self.bos.map(|x| x as usize)
    }
    pub fn eos(&self) -> Option<usize> {
        self.eos.map(|x| x as usize)
    }

    /// Encode text to token IDs. BPE is exact (given the approximate
    /// pre-tokenizer); SPM is a greedy longest-match approximation with byte
    /// fallback. See the module docs.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        match self.model {
            TokModel::Bpe => self.encode_bpe(text),
            TokModel::Spm => self.encode_spm(text),
        }
    }

    /// Decode token IDs back to text. Exact for both families.
    pub fn decode(&self, ids: &[usize]) -> String {
        match self.model {
            TokModel::Bpe => self.decode_bpe(ids),
            TokModel::Spm => self.decode_spm(ids),
        }
    }

    // ── BPE ───────────────────────────────────────────────────────────────────

    fn encode_bpe(&self, text: &str) -> Vec<usize> {
        let mut out = Vec::new();
        for chunk in pretokenize(text) {
            // Byte-level map: each byte → one printable GPT-2 char (a "piece").
            let mut pieces: Vec<String> = chunk
                .bytes()
                .map(|b| self.byte_encoder[b as usize].to_string())
                .collect();
            if pieces.is_empty() {
                continue;
            }
            // Greedy lowest-rank merges until none apply (standard BPE). `key` is
            // reused across every pair lookup so the scan allocates nothing.
            let mut key = String::new();
            while pieces.len() > 1 {
                let mut best: Option<(usize, u32)> = None;
                for i in 0..pieces.len() - 1 {
                    key.clear();
                    key.push_str(&pieces[i]);
                    key.push('\u{0}');
                    key.push_str(&pieces[i + 1]);
                    if let Some(&r) = self.merges.get(key.as_str()) {
                        let better = match best {
                            None => true,
                            Some((_, br)) => r < br,
                        };
                        if better {
                            best = Some((i, r));
                        }
                    }
                }
                let Some((i, _)) = best else { break };
                let merged = format!("{}{}", pieces[i], pieces[i + 1]);
                pieces[i] = merged;
                pieces.remove(i + 1);
            }
            for p in pieces {
                if let Some(&id) = self.token_to_id.get(&p) {
                    out.push(id as usize);
                } else {
                    // Unmergeable leftover: emit its constituent byte tokens,
                    // which always exist in a byte-level vocab.
                    for ch in p.chars() {
                        if let Some(&id) = self.token_to_id.get(&ch.to_string()) {
                            out.push(id as usize);
                        }
                    }
                }
            }
        }
        out
    }

    fn decode_bpe(&self, ids: &[usize]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            let Some(tok) = self.tokens.get(id) else {
                continue;
            };
            for ch in tok.chars() {
                match self.byte_decoder.get(&ch) {
                    Some(&b) => bytes.push(b),
                    // A non-byte char (e.g. an added special token's literal
                    // text) — emit its own UTF-8.
                    None => {
                        let mut buf = [0u8; 4];
                        bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                    }
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    // ── SPM ───────────────────────────────────────────────────────────────────

    fn encode_spm(&self, text: &str) -> Vec<usize> {
        // SentencePiece prepends a dummy space and renders spaces as ▁.
        let normalized: Vec<char> = format!(" {text}")
            .replace(' ', &SPM_SPACE.to_string())
            .chars()
            .collect();
        let mut out = Vec::new();
        let mut i = 0;
        while i < normalized.len() {
            // Longest token in the vocab starting at i. The window is capped at
            // the longest vocab token, so this is O(max_token_len) per position
            // (not O(remaining-length)) — without the cap, encoding a long corpus
            // is O(n³) and effectively hangs.
            let mut hit = None;
            let mut j = (i + self.max_token_len).min(normalized.len());
            while j > i {
                let cand: String = normalized[i..j].iter().collect();
                if let Some(&id) = self.token_to_id.get(&cand) {
                    hit = Some((id, j));
                    break;
                }
                j -= 1;
            }
            match hit {
                Some((id, j)) => {
                    out.push(id as usize);
                    i = j;
                }
                None => {
                    // Byte fallback: emit each UTF-8 byte's <0xXX> token.
                    let mut buf = [0u8; 4];
                    for &b in normalized[i].encode_utf8(&mut buf).as_bytes() {
                        if let Some(&id) = self.byte_to_id.get(&b) {
                            out.push(id as usize);
                        }
                    }
                    i += 1;
                }
            }
        }
        out
    }

    fn decode_spm(&self, ids: &[usize]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            let Some(tok) = self.tokens.get(id) else {
                continue;
            };
            if let Some(b) = parse_byte_token(tok) {
                bytes.push(b);
                continue;
            }
            for ch in tok.chars() {
                if ch == SPM_SPACE {
                    bytes.push(b' ');
                } else {
                    let mut buf = [0u8; 4];
                    bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                }
            }
        }
        let s = String::from_utf8_lossy(&bytes).into_owned();
        // Drop the leading space SentencePiece's dummy prefix introduced.
        s.strip_prefix(' ').map(str::to_string).unwrap_or(s)
    }
}

/// Dependency-free approximation of GPT-2 pre-tokenization: split into runs of
/// letters / digits / symbols, with a single leading space attached to the
/// following run (so the space becomes the BPE `Ġ` marker). Whitespace runs are
/// emitted as their own chunks.
fn pretokenize(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut chunks = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let start = i;
        if chars[i] == ' ' {
            i += 1; // attach one leading space to the following run
        }
        if i >= chars.len() {
            chunks.push(chars[start..].iter().collect());
            break;
        }
        let c = chars[i];
        if c.is_alphabetic() {
            while i < chars.len() && chars[i].is_alphabetic() {
                i += 1;
            }
        } else if c.is_numeric() {
            while i < chars.len() && chars[i].is_numeric() {
                i += 1;
            }
        } else if c.is_whitespace() {
            while i < chars.len() && chars[i].is_whitespace() {
                i += 1;
            }
        } else {
            while i < chars.len() && !chars[i].is_alphanumeric() && !chars[i].is_whitespace() {
                i += 1;
            }
        }
        chunks.push(chars[start..i].iter().collect());
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── In-memory GGUF builders (reuse the writer pattern from gguf.rs tests) ──

    fn put_u32(o: &mut Vec<u8>, v: u32) {
        o.extend_from_slice(&v.to_le_bytes());
    }
    fn put_u64(o: &mut Vec<u8>, v: u64) {
        o.extend_from_slice(&v.to_le_bytes());
    }
    fn put_str(o: &mut Vec<u8>, s: &str) {
        put_u64(o, s.len() as u64);
        o.extend_from_slice(s.as_bytes());
    }
    fn kv_str(o: &mut Vec<u8>, key: &str, v: &str) {
        put_str(o, key);
        put_u32(o, 8); // VT_STRING
        put_str(o, v);
    }
    fn kv_u32(o: &mut Vec<u8>, key: &str, v: u32) {
        put_str(o, key);
        put_u32(o, 4); // VT_U32
        put_u32(o, v);
    }
    fn kv_str_array(o: &mut Vec<u8>, key: &str, items: &[&str]) {
        put_str(o, key);
        put_u32(o, 9); // VT_ARRAY
        put_u32(o, 8); // elem type STRING
        put_u64(o, items.len() as u64);
        for s in items {
            put_str(o, s);
        }
    }

    /// A GGUF carrying only a tokenizer metadata table (no tensors).
    fn tok_gguf(kvs: &[u8], n_kv: u64) -> Gguf {
        let mut bytes = Vec::new();
        put_u32(&mut bytes, 0x4655_4747); // GGUF magic
        put_u32(&mut bytes, 3); // version
        put_u64(&mut bytes, 0); // tensor_count
        put_u64(&mut bytes, n_kv);
        bytes.extend_from_slice(kvs);
        // Pad to alignment 32 so the (empty) data section is in-bounds.
        let pad = bytes.len().div_ceil(32) * 32 - bytes.len();
        bytes.extend(std::iter::repeat_n(0u8, pad));
        Gguf::parse(bytes).unwrap()
    }

    #[test]
    fn bpe_roundtrips_via_merges() {
        // Vocab of single-byte GPT-2 chars plus a few merges so "ab"+"c" forms.
        // bytes: 'a','b','c',' ' → GPT-2 chars 'a','b','c','Ġ'.
        let toks = ["a", "b", "c", "\u{0120}", "ab", "abc", "\u{0120}a"];
        let merges = ["a b", "ab c", "\u{0120} a"];
        let mut kvs = Vec::new();
        kv_str(&mut kvs, "tokenizer.ggml.model", "gpt2");
        kv_str_array(&mut kvs, "tokenizer.ggml.tokens", &toks);
        kv_str_array(&mut kvs, "tokenizer.ggml.merges", &merges);
        kv_u32(&mut kvs, "tokenizer.ggml.bos_token_id", 0);
        let g = tok_gguf(&kvs, 4);

        let tk = GgufTokenizer::from_gguf(&g).unwrap();
        assert_eq!(tk.model(), TokModel::Bpe);
        // "abc" merges a+b→ab, ab+c→abc ⇒ single token id 5.
        assert_eq!(tk.encode("abc"), vec![5]);
        // decode is the exact inverse.
        assert_eq!(tk.decode(&[5]), "abc");
        // " a" → leading space attaches: 'Ġ'+'a' merges to "Ġa" (id 6).
        assert_eq!(tk.encode(" a"), vec![6]);
        assert_eq!(tk.decode(&tk.encode("abc")), "abc");
    }

    #[test]
    fn spm_decode_handles_space_marker_and_byte_fallback() {
        let toks = ["<unk>", "\u{2581}hi", "\u{2581}", "<0x41>"]; // ▁hi, ▁, byte 'A'
        let mut kvs = Vec::new();
        kv_str(&mut kvs, "tokenizer.ggml.model", "llama");
        kv_str_array(&mut kvs, "tokenizer.ggml.tokens", &toks);
        kv_u32(&mut kvs, "tokenizer.ggml.eos_token_id", 0);
        let g = tok_gguf(&kvs, 3);

        let tk = GgufTokenizer::from_gguf(&g).unwrap();
        assert_eq!(tk.model(), TokModel::Spm);
        // ▁hi → "hi" (leading dummy space stripped).
        assert_eq!(tk.decode(&[1]), "hi");
        // <0x41> → 'A'.
        assert_eq!(tk.decode(&[3]), "A");
        assert_eq!(tk.eos(), Some(0));
    }

    #[test]
    fn spm_encode_greedy_longest_match() {
        // "▁he", "▁", "h", "e" — greedy picks the longest match "▁he".
        let toks = ["<unk>", "\u{2581}he", "\u{2581}", "h", "e"];
        let mut kvs = Vec::new();
        kv_str(&mut kvs, "tokenizer.ggml.model", "spm");
        kv_str_array(&mut kvs, "tokenizer.ggml.tokens", &toks);
        let g = tok_gguf(&kvs, 2);
        let tk = GgufTokenizer::from_gguf(&g).unwrap();
        // "he" → dummy space → "▁he" (id 1).
        assert_eq!(tk.encode("he"), vec![1]);
    }

    #[test]
    fn spm_encode_is_bounded_and_roundtrips_on_long_input() {
        // A long input must encode quickly (the longest-match window is capped at
        // the longest vocab token) and decode back exactly. Before the cap this
        // was O(n³) and effectively hung on any real corpus.
        let toks = ["<unk>", "\u{2581}", "a", "b", "ab"];
        let mut kvs = Vec::new();
        kv_str(&mut kvs, "tokenizer.ggml.model", "spm");
        kv_str_array(&mut kvs, "tokenizer.ggml.tokens", &toks);
        let g = tok_gguf(&kvs, 2);
        let tk = GgufTokenizer::from_gguf(&g).unwrap();
        assert_eq!(tk.max_token_len, 5); // "<unk>"
        let input = "ab".repeat(10_000); // 20k chars — O(n³) would never finish
        let ids = tk.encode(&input);
        assert!(!ids.is_empty());
        assert_eq!(tk.decode(&ids), input, "SPM long-input round-trip");
    }

    #[test]
    fn bpe_roundtrips_long_input_with_reused_key_buffer() {
        // Exercises the reused merge-key buffer across many pairs; byte-level BPE
        // round-trips exactly for in-vocab bytes.
        let toks = ["a", "b", "c", "\u{0120}", "ab", "abc", "\u{0120}a"];
        let merges = ["a b", "ab c", "\u{0120} a"];
        let mut kvs = Vec::new();
        kv_str(&mut kvs, "tokenizer.ggml.model", "gpt2");
        kv_str_array(&mut kvs, "tokenizer.ggml.tokens", &toks);
        kv_str_array(&mut kvs, "tokenizer.ggml.merges", &merges);
        let g = tok_gguf(&kvs, 3);
        let tk = GgufTokenizer::from_gguf(&g).unwrap();
        let input = "abc abc abc abc abc";
        assert_eq!(tk.decode(&tk.encode(input)), input, "BPE round-trip");
    }

    #[test]
    fn unsupported_or_missing_tokenizer_errors() {
        let mut kvs = Vec::new();
        kv_str(&mut kvs, "tokenizer.ggml.model", "wordpiece");
        let g = tok_gguf(&kvs, 1);
        assert!(GgufTokenizer::from_gguf(&g).is_err());

        // No tokenizer metadata at all.
        let empty = tok_gguf(&[], 0);
        assert!(GgufTokenizer::from_gguf(&empty).is_err());
    }

    #[test]
    fn gpt2_byte_tables_are_a_bijection() {
        let (enc, dec) = gpt2_byte_tables();
        for b in 0u16..256 {
            let b = b as u8;
            assert_eq!(
                dec.get(&enc[b as usize]),
                Some(&b),
                "byte {b} not invertible"
            );
        }
        assert_eq!(dec.len(), 256, "encoder is not injective");
        // Space maps to the BPE marker 'Ġ' (U+0120).
        assert_eq!(enc[b' ' as usize], '\u{0120}');
    }

    #[test]
    fn pretokenize_attaches_leading_space() {
        assert_eq!(
            pretokenize("hi there"),
            vec!["hi".to_string(), " there".to_string()]
        );
        assert_eq!(
            pretokenize("a1!"),
            vec!["a".to_string(), "1".to_string(), "!".to_string()]
        );
    }
}
