//! Byte-level Byte-Pair Encoding (BPE) tokenizer. Zero dependencies.
//!
//! The base vocabulary is the 256 single bytes, so *any* text — emoji,
//! Cyrillic, CJK, control characters — round-trips without an unknown-token
//! escape hatch. Training greedily merges the most frequent adjacent token
//! pair until the requested vocabulary size is reached (or no pair repeats).
//!
//! ```rust
//! use ferrum_core::ByteBpeTokenizer;
//!
//! let corpus = "low lower lowest low low";
//! let tok = ByteBpeTokenizer::train(corpus, 300).unwrap();
//! let ids = tok.encode("lowest");
//! assert_eq!(tok.decode(&ids), "lowest");
//! ```
use crate::error::{InferError, Result};
use std::collections::HashMap;

/// Number of base tokens (one per byte value).
const BASE_VOCAB: usize = 256;

/// A trained byte-level BPE tokenizer.
///
/// The full state is the ordered merge list: token `256 + i` is defined as
/// the concatenation of merge `i`'s pair, so [`encode_state`](Self::encode_state)
/// serialises merges only and [`from_state`](Self::from_state) rebuilds the
/// vocabulary deterministically.
pub struct ByteBpeTokenizer {
    /// id → byte string. ids 0..256 are the single bytes; special tokens (if
    /// any) occupy the highest ids and map to empty byte strings.
    vocab: Vec<Vec<u8>>,
    /// Learned merges in order; merge `i` creates token id `256 + i`.
    merges: Vec<(u32, u32)>,
    /// `pair → merge rank` (= index in `merges`), for the O(applied) rank-based
    /// encoder (K1). Rebuilt whenever `merges` changes.
    ranks: HashMap<(u32, u32), usize>,
    /// Named special tokens (BOS/EOS/PAD/UNK, …) in id order (K3). Their ids are
    /// reserved **above** the byte + merge range, so adding them never collides
    /// with bytes or merges. They never arise from encoding ordinary text and
    /// decode to the empty string.
    special: Vec<String>,
}

/// Conventional names for the four standard special tokens (K3).
pub const TOK_BOS: &str = "<bos>";
pub const TOK_EOS: &str = "<eos>";
pub const TOK_PAD: &str = "<pad>";
pub const TOK_UNK: &str = "<unk>";

/// Replace every non-overlapping occurrence of `pair` in `ids` with `new_id`,
/// scanning left to right.
fn merge_pair(ids: &[u32], pair: (u32, u32), new_id: u32) -> Vec<u32> {
    let mut out = Vec::with_capacity(ids.len());
    let mut i = 0;
    while i < ids.len() {
        if i + 1 < ids.len() && ids[i] == pair.0 && ids[i + 1] == pair.1 {
            out.push(new_id);
            i += 2;
        } else {
            out.push(ids[i]);
            i += 1;
        }
    }
    out
}

/// Pre-tokenize `text` into chunks at whitespace↔non-whitespace boundaries (K2),
/// so learned merges never span a space/newline (as GPT-2 BPE avoids). The
/// chunks are a partition of `text` — concatenating them reproduces it exactly,
/// so encode/decode still round-trips.
fn pretokenize(text: &str) -> Vec<&str> {
    let mut chunks = Vec::new();
    let mut start = 0;
    let mut prev_ws: Option<bool> = None;
    for (i, ch) in text.char_indices() {
        let ws = ch.is_whitespace();
        if prev_ws.is_some_and(|p| p != ws) {
            chunks.push(&text[start..i]);
            start = i;
        }
        prev_ws = Some(ws);
    }
    if start < text.len() {
        chunks.push(&text[start..]);
    }
    chunks
}

impl ByteBpeTokenizer {
    /// A tokenizer with no merges: pure byte-level (vocab = 256).
    pub fn byte_level() -> Self {
        Self {
            vocab: (0..=255u8).map(|b| vec![b]).collect(),
            merges: Vec::new(),
            ranks: HashMap::new(),
            special: Vec::new(),
        }
    }

    /// Rebuild the `pair → rank` map from the current merge list.
    fn rebuild_ranks(&mut self) {
        self.ranks = self
            .merges
            .iter()
            .enumerate()
            .map(|(i, &p)| (p, i))
            .collect();
    }

    /// Learn up to `vocab_size − 256` merges from `corpus`.
    ///
    /// The corpus is pre-tokenized at whitespace boundaries (K2) so merges stay
    /// within words; training then greedily merges the most frequent adjacent
    /// pair. Stops early when no adjacent pair occurs at least twice.
    /// `vocab_size` below 256 is an error (the byte base is irreducible).
    pub fn train(corpus: &str, vocab_size: usize) -> Result<Self> {
        if vocab_size < BASE_VOCAB {
            return Err(InferError::DimMismatch(format!(
                "vocab_size {vocab_size} < {BASE_VOCAB} (byte base vocabulary)"
            )));
        }
        let mut tok = Self::byte_level();
        // Pre-tokenized chunks as independent id sequences (no cross-chunk merges).
        let mut chunks: Vec<Vec<u32>> = pretokenize(corpus)
            .iter()
            .map(|c| c.bytes().map(u32::from).collect())
            .collect();
        vprintln!(
            "[tokenizer::train] corpus={} bytes, {} chunks, target vocab={}",
            corpus.len(),
            chunks.len(),
            vocab_size
        );

        while tok.vocab.len() < vocab_size {
            // Count adjacent pairs within each chunk.
            let mut counts: HashMap<(u32, u32), usize> = HashMap::new();
            for chunk in &chunks {
                for w in chunk.windows(2) {
                    *counts.entry((w[0], w[1])).or_insert(0) += 1;
                }
            }
            // Most frequent pair; ties broken by smallest pair so training is
            // deterministic regardless of HashMap iteration order.
            let best = counts
                .iter()
                .max_by(|a, b| a.1.cmp(b.1).then_with(|| b.0.cmp(a.0)))
                .map(|(&pair, &count)| (pair, count));
            let Some((pair, count)) = best else { break };
            if count < 2 {
                break;
            }
            let new_id = tok.vocab.len() as u32;
            let merged = [
                tok.vocab[pair.0 as usize].clone(),
                tok.vocab[pair.1 as usize].clone(),
            ]
            .concat();
            vprintln!(
                "[tokenizer::train] merge {}: {:?}+{:?} → id {} (count {})",
                tok.merges.len(),
                pair.0,
                pair.1,
                new_id,
                count
            );
            tok.vocab.push(merged);
            tok.merges.push(pair);
            for chunk in &mut chunks {
                *chunk = merge_pair(chunk, pair, new_id);
            }
        }
        tok.rebuild_ranks();
        vprintln!(
            "[tokenizer::train] done: vocab={}, merges={}",
            tok.vocab.len(),
            tok.merges.len()
        );
        Ok(tok)
    }

    /// Rank-based BPE encode of one pre-tokenized chunk (K1): repeatedly apply
    /// the lowest-rank (earliest-learned) applicable merge until none applies —
    /// `O(applied merges × len)` instead of rescanning the whole text once per
    /// learned merge.
    fn encode_chunk(&self, bytes: &[u8], out: &mut Vec<usize>) {
        let mut ids: Vec<u32> = bytes.iter().map(|&b| u32::from(b)).collect();
        while ids.len() >= 2 {
            // Lowest-rank adjacent pair present in the sequence.
            let mut best: Option<(usize, (u32, u32))> = None;
            for w in ids.windows(2) {
                let pair = (w[0], w[1]);
                if let Some(&rank) = self.ranks.get(&pair) {
                    if best.is_none_or(|(br, _)| rank < br) {
                        best = Some((rank, pair));
                    }
                }
            }
            let Some((rank, pair)) = best else { break };
            ids = merge_pair(&ids, pair, (BASE_VOCAB + rank) as u32);
        }
        out.extend(ids.into_iter().map(|i| i as usize));
    }

    /// Encode text to token IDs: pre-tokenize at whitespace boundaries (K2),
    /// then rank-based BPE-encode each chunk (K1) and concatenate.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut out = Vec::new();
        for chunk in pretokenize(text) {
            self.encode_chunk(chunk.as_bytes(), &mut out);
        }
        out
    }

    // ── Special tokens (K3) ───────────────────────────────────────────────────

    /// Append a named special token, returning its reserved id (above the byte +
    /// merge range). Idempotent: re-adding an existing name returns its id.
    /// Names may not contain `,`, `;`, or newlines (the state delimiters).
    pub fn add_special_token(&mut self, name: &str) -> Result<usize> {
        if name.is_empty() || name.contains(',') || name.contains(';') || name.contains('\n') {
            return Err(InferError::Format(
                "special-token name must be non-empty and free of ',', ';', and newlines".into(),
            ));
        }
        if let Some(id) = self.special_id(name) {
            return Ok(id);
        }
        let id = self.vocab.len();
        self.vocab.push(Vec::new()); // decodes to nothing
        self.special.push(name.to_string());
        Ok(id)
    }

    /// Register the four conventional special tokens ([`TOK_BOS`], [`TOK_EOS`],
    /// [`TOK_PAD`], [`TOK_UNK`]) and return their ids in that order.
    pub fn add_standard_special_tokens(&mut self) -> [usize; 4] {
        [TOK_BOS, TOK_EOS, TOK_PAD, TOK_UNK].map(|n| self.add_special_token(n).unwrap())
    }

    /// The id of a registered special token, if present.
    pub fn special_id(&self, name: &str) -> Option<usize> {
        self.special
            .iter()
            .position(|n| n == name)
            .map(|i| BASE_VOCAB + self.merges.len() + i)
    }

    /// Convenience accessors for the conventional special tokens.
    pub fn bos_id(&self) -> Option<usize> {
        self.special_id(TOK_BOS)
    }
    pub fn eos_id(&self) -> Option<usize> {
        self.special_id(TOK_EOS)
    }
    pub fn pad_id(&self) -> Option<usize> {
        self.special_id(TOK_PAD)
    }
    pub fn unk_id(&self) -> Option<usize> {
        self.special_id(TOK_UNK)
    }

    /// Number of registered special tokens.
    pub fn num_special(&self) -> usize {
        self.special.len()
    }

    /// Encode `text`, optionally framed with the registered BOS/EOS tokens
    /// (I3/K3). Framing is skipped silently for any token not registered.
    pub fn encode_framed(&self, text: &str, bos: bool, eos: bool) -> Vec<usize> {
        let mut ids = Vec::new();
        if bos {
            if let Some(b) = self.bos_id() {
                ids.push(b);
            }
        }
        ids.extend(self.encode(text));
        if eos {
            if let Some(e) = self.eos_id() {
                ids.push(e);
            }
        }
        ids
    }

    /// Decode token IDs back to text. Unknown IDs are skipped; byte sequences
    /// that are not valid UTF-8 decode lossily (U+FFFD). Special tokens map to
    /// empty byte strings, so they vanish from decoded text.
    pub fn decode(&self, ids: &[usize]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if let Some(tok) = self.vocab.get(id) {
                bytes.extend_from_slice(tok);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// The raw bytes of one token, if the ID is in range.
    pub fn token_bytes(&self, id: usize) -> Option<&[u8]> {
        self.vocab.get(id).map(Vec::as_slice)
    }

    /// Serialise the merge list as `"a,b;c,d;…"`, optionally followed by a
    /// newline and the comma-separated special-token names (K3). An empty string
    /// is byte-level with no specials. Compact enough to embed in FINF metadata.
    pub fn encode_state(&self) -> String {
        let merges = self
            .merges
            .iter()
            .map(|(a, b)| format!("{a},{b}"))
            .collect::<Vec<_>>()
            .join(";");
        if self.special.is_empty() {
            merges
        } else {
            format!("{merges}\n{}", self.special.join(","))
        }
    }

    /// Rebuild a tokenizer from [`encode_state`](Self::encode_state) output,
    /// including any special tokens.
    pub fn from_state(state: &str) -> Result<Self> {
        let mut tok = Self::byte_level();
        // Split the optional special-token section off the merge list.
        let (merge_part, special_part) = match state.split_once('\n') {
            Some((m, s)) => (m, Some(s)),
            None => (state, None),
        };
        if !merge_part.is_empty() {
            for entry in merge_part.split(';') {
                let (a, b) = entry
                    .split_once(',')
                    .ok_or_else(|| InferError::Format(format!("bad merge entry {entry:?}")))?;
                let a: u32 = a
                    .trim()
                    .parse()
                    .map_err(|_| InferError::ParseError(format!("bad merge id {a:?}")))?;
                let b: u32 = b
                    .trim()
                    .parse()
                    .map_err(|_| InferError::ParseError(format!("bad merge id {b:?}")))?;
                if a as usize >= tok.vocab.len() || b as usize >= tok.vocab.len() {
                    return Err(InferError::Format(format!(
                        "merge ({a},{b}) references undefined token (vocab {})",
                        tok.vocab.len()
                    )));
                }
                let merged =
                    [tok.vocab[a as usize].clone(), tok.vocab[b as usize].clone()].concat();
                tok.vocab.push(merged);
                tok.merges.push((a, b));
            }
        }
        if let Some(specials) = special_part {
            for name in specials.split(',').filter(|n| !n.is_empty()) {
                tok.add_special_token(name)?;
            }
        }
        tok.rebuild_ranks();
        Ok(tok)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    const CORPUS: &str = "low lower lowest low low newer newest new news";

    #[test]
    fn byte_level_roundtrip() {
        let tok = ByteBpeTokenizer::byte_level();
        let text = "hello 🌸 мир\n";
        let ids = tok.encode(text);
        assert_eq!(ids.len(), text.len()); // one token per byte
        assert_eq!(tok.decode(&ids), text);
    }

    #[test]
    fn trained_roundtrip_exact() {
        let tok = ByteBpeTokenizer::train(CORPUS, 300).unwrap();
        for text in [CORPUS, "lowest news", "unseen zzz 🌸 text"] {
            assert_eq!(
                tok.decode(&tok.encode(text)),
                text,
                "roundtrip failed for {text:?}"
            );
        }
    }

    #[test]
    fn merges_compress_the_corpus() {
        let tok = ByteBpeTokenizer::train(CORPUS, 300).unwrap();
        assert!(!tok.encode_state().is_empty(), "no merges learned");
        let ids = tok.encode(CORPUS);
        assert!(
            ids.len() < CORPUS.len(),
            "encoding not shorter: {} tokens for {} bytes",
            ids.len(),
            CORPUS.len()
        );
    }

    #[test]
    fn vocab_size_is_respected() {
        let tok = ByteBpeTokenizer::train(CORPUS, 260).unwrap();
        assert!(tok.vocab_size() <= 260);
        assert_eq!(tok.vocab_size(), 260); // corpus has ≥4 repeating pairs
    }

    #[test]
    fn training_stops_when_nothing_repeats() {
        // All-distinct bytes: no pair occurs twice, so no merges happen.
        let tok = ByteBpeTokenizer::train("abcdefg", 1000).unwrap();
        assert_eq!(tok.vocab_size(), 256);
        assert!(tok.encode_state().is_empty());
    }

    #[test]
    fn vocab_below_256_errors() {
        assert!(ByteBpeTokenizer::train(CORPUS, 100).is_err());
    }

    #[test]
    fn training_is_deterministic() {
        let a = ByteBpeTokenizer::train(CORPUS, 300).unwrap();
        let b = ByteBpeTokenizer::train(CORPUS, 300).unwrap();
        assert_eq!(a.encode_state(), b.encode_state());
        assert_eq!(a.encode(CORPUS), b.encode(CORPUS));
    }

    #[test]
    fn state_roundtrip_preserves_encoding() {
        let tok = ByteBpeTokenizer::train(CORPUS, 300).unwrap();
        let restored = ByteBpeTokenizer::from_state(&tok.encode_state()).unwrap();
        assert_eq!(restored.vocab_size(), tok.vocab_size());
        assert_eq!(restored.encode(CORPUS), tok.encode(CORPUS));
        assert_eq!(restored.decode(&restored.encode("lowest")), "lowest");
    }

    #[test]
    fn bad_state_errors() {
        assert!(ByteBpeTokenizer::from_state("not-a-pair").is_err());
        assert!(ByteBpeTokenizer::from_state("1,2;9999,3").is_err()); // undefined id
        assert!(ByteBpeTokenizer::from_state("x,y").is_err());
    }

    #[test]
    fn token_bytes_lookup() {
        let tok = ByteBpeTokenizer::byte_level();
        assert_eq!(tok.token_bytes(b'a' as usize), Some(&b"a"[..]));
        assert_eq!(tok.token_bytes(999), None);
    }

    #[test]
    fn multibyte_utf8_merges_safely() {
        // Repeated multi-byte chars: merges may span partial code points
        // internally, but decode must still reassemble valid UTF-8.
        let corpus = "🌸🌸🌸🌸🌸 zen 🌸🌸🌸";
        let tok = ByteBpeTokenizer::train(corpus, 280).unwrap();
        assert_eq!(tok.decode(&tok.encode(corpus)), corpus);
    }

    // ── Pre-tokenization (K2) ─────────────────────────────────────────────────

    #[test]
    fn pretokenize_partitions_on_whitespace_boundaries() {
        assert_eq!(pretokenize("low lower"), vec!["low", " ", "lower"]);
        assert_eq!(pretokenize("a\n b"), vec!["a", "\n ", "b"]);
        assert_eq!(pretokenize(""), Vec::<&str>::new());
        assert_eq!(pretokenize("solid"), vec!["solid"]);
        // The chunks always reconstruct the original text.
        let s = "the  quick\tbrown\nfox";
        assert_eq!(pretokenize(s).concat(), s);
    }

    #[test]
    fn no_merge_spans_whitespace() {
        // "ab ab ab" — without pre-tokenization BPE could learn a token that
        // straddles the space (e.g. "b a"); pre-tokenization forbids it.
        let tok = ByteBpeTokenizer::train("ab ab ab ab ab", 300).unwrap();
        for (id, bytes) in tok.vocab.iter().enumerate() {
            if id >= BASE_VOCAB {
                let has_inner_space = bytes.iter().any(|&b| (b as char).is_whitespace());
                assert!(
                    !has_inner_space,
                    "merged token {id} ({bytes:?}) spans whitespace"
                );
            }
        }
    }

    // ── Rank-based encode (K1) ────────────────────────────────────────────────

    #[test]
    fn rank_based_encode_is_greedy_lowest_rank_first() {
        // Train so "lo","ow" (hence "low") are learned; encoding "low" must use
        // the merged token, i.e. fewer tokens than its 3 bytes.
        let tok = ByteBpeTokenizer::train("low low low low low", 300).unwrap();
        let ids = tok.encode("low");
        assert!(ids.len() < 3, "expected a merged 'low' token, got {ids:?}");
        assert_eq!(tok.decode(&ids), "low");
    }

    #[test]
    fn encode_uses_learned_merges_across_words_independently() {
        let tok = ByteBpeTokenizer::train("hello hello world world hello world", 320).unwrap();
        // Two words → each encodes via its own within-word merges; concatenation
        // round-trips and the space is its own (unmerged) token between them.
        let ids = tok.encode("hello world");
        assert_eq!(tok.decode(&ids), "hello world");
        assert!(ids.len() <= "hello world".len());
    }

    #[test]
    fn encode_matches_after_state_roundtrip_with_ranks() {
        // The rank map must be rebuilt by from_state so encoding is identical.
        let tok = ByteBpeTokenizer::train(CORPUS, 320).unwrap();
        let restored = ByteBpeTokenizer::from_state(&tok.encode_state()).unwrap();
        for text in [CORPUS, "lowest newer news", "🌸 unseen"] {
            assert_eq!(
                restored.encode(text),
                tok.encode(text),
                "mismatch for {text:?}"
            );
        }
    }

    // ── Special tokens (K3) ───────────────────────────────────────────────────

    #[test]
    fn special_tokens_get_reserved_ids_above_vocab() {
        let mut tok = ByteBpeTokenizer::train(CORPUS, 300).unwrap();
        let base = tok.vocab_size();
        let [bos, eos, pad, unk] = tok.add_standard_special_tokens();
        assert_eq!([bos, eos, pad, unk], [base, base + 1, base + 2, base + 3]);
        assert_eq!(tok.vocab_size(), base + 4);
        assert_eq!(tok.eos_id(), Some(eos));
        assert_eq!(tok.bos_id(), Some(bos));
        assert_eq!(tok.num_special(), 4);
        // Adding an existing special is idempotent.
        assert_eq!(tok.add_special_token(TOK_EOS).unwrap(), eos);
        assert_eq!(tok.vocab_size(), base + 4);
    }

    #[test]
    fn special_ids_are_disjoint_from_text_and_decode_to_empty() {
        let mut tok = ByteBpeTokenizer::train(CORPUS, 300).unwrap();
        tok.add_standard_special_tokens();
        // Ordinary text never encodes to a special id.
        let ids = tok.encode("low lowest newer");
        let max_text_id = tok.vocab_size() - tok.num_special();
        assert!(
            ids.iter().all(|&i| i < max_text_id),
            "text encoded to a special id"
        );
        // Special tokens decode to nothing, so they vanish from text.
        let eos = tok.eos_id().unwrap();
        assert_eq!(tok.decode(&[eos]), "");
        let mixed = tok.encode_framed("low", true, true);
        assert_eq!(
            tok.decode(&mixed),
            "low",
            "framing must not alter decoded text"
        );
    }

    #[test]
    fn encode_framed_adds_bos_and_eos() {
        let mut tok = ByteBpeTokenizer::train(CORPUS, 300).unwrap();
        tok.add_standard_special_tokens();
        let bare = tok.encode("lower");
        let framed = tok.encode_framed("lower", true, true);
        assert_eq!(framed.first(), Some(&tok.bos_id().unwrap()));
        assert_eq!(framed.last(), Some(&tok.eos_id().unwrap()));
        assert_eq!(&framed[1..framed.len() - 1], bare.as_slice());
        // Without registered specials, framing is a no-op.
        let plain = ByteBpeTokenizer::train(CORPUS, 300).unwrap();
        assert_eq!(
            plain.encode_framed("lower", true, true),
            plain.encode("lower")
        );
    }

    #[test]
    fn special_tokens_survive_state_roundtrip() {
        let mut tok = ByteBpeTokenizer::train(CORPUS, 300).unwrap();
        tok.add_standard_special_tokens();
        let restored = ByteBpeTokenizer::from_state(&tok.encode_state()).unwrap();
        assert_eq!(restored.vocab_size(), tok.vocab_size());
        assert_eq!(restored.eos_id(), tok.eos_id());
        assert_eq!(restored.num_special(), 4);
        assert_eq!(restored.encode(CORPUS), tok.encode(CORPUS));

        // Specials also round-trip on a byte-level (no-merge) tokenizer.
        let mut bl = ByteBpeTokenizer::byte_level();
        bl.add_special_token(TOK_EOS).unwrap();
        let r = ByteBpeTokenizer::from_state(&bl.encode_state()).unwrap();
        assert_eq!(r.eos_id(), bl.eos_id());
        assert_eq!(r.vocab_size(), 256 + 1);
    }

    #[test]
    fn add_special_token_rejects_delimiter_names() {
        let mut tok = ByteBpeTokenizer::byte_level();
        assert!(tok.add_special_token("").is_err());
        assert!(tok.add_special_token("a,b").is_err());
        assert!(tok.add_special_token("a;b").is_err());
        assert!(tok.add_special_token("a\nb").is_err());
        assert!(tok.add_special_token(TOK_EOS).is_ok());
    }
}
