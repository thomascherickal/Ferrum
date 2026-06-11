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
    /// id → byte string. ids 0..256 are the single bytes.
    vocab: Vec<Vec<u8>>,
    /// Learned merges in order; merge `i` creates token id `256 + i`.
    merges: Vec<(u32, u32)>,
}

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

impl ByteBpeTokenizer {
    /// A tokenizer with no merges: pure byte-level (vocab = 256).
    pub fn byte_level() -> Self {
        Self {
            vocab: (0..=255u8).map(|b| vec![b]).collect(),
            merges: Vec::new(),
        }
    }

    /// Learn up to `vocab_size − 256` merges from `corpus`.
    ///
    /// Stops early when no adjacent pair occurs at least twice. `vocab_size`
    /// below 256 is an error (the byte base is irreducible).
    pub fn train(corpus: &str, vocab_size: usize) -> Result<Self> {
        if vocab_size < BASE_VOCAB {
            return Err(InferError::DimMismatch(format!(
                "vocab_size {vocab_size} < {BASE_VOCAB} (byte base vocabulary)"
            )));
        }
        let mut tok = Self::byte_level();
        let mut ids: Vec<u32> = corpus.bytes().map(u32::from).collect();
        vprintln!("[tokenizer::train] corpus={} bytes, target vocab={}", ids.len(), vocab_size);

        while tok.vocab.len() < vocab_size {
            // Count adjacent pairs.
            let mut counts: HashMap<(u32, u32), usize> = HashMap::new();
            for w in ids.windows(2) {
                *counts.entry((w[0], w[1])).or_insert(0) += 1;
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
            let merged = [tok.vocab[pair.0 as usize].clone(), tok.vocab[pair.1 as usize].clone()].concat();
            vprintln!("[tokenizer::train] merge {}: {:?}+{:?} → id {} (count {})",
                tok.merges.len(), pair.0, pair.1, new_id, count);
            tok.vocab.push(merged);
            tok.merges.push(pair);
            ids = merge_pair(&ids, pair, new_id);
        }
        vprintln!("[tokenizer::train] done: vocab={}, merges={}", tok.vocab.len(), tok.merges.len());
        Ok(tok)
    }

    /// Encode text to token IDs by applying the learned merges in order.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut ids: Vec<u32> = text.bytes().map(u32::from).collect();
        for (i, &pair) in self.merges.iter().enumerate() {
            ids = merge_pair(&ids, pair, (BASE_VOCAB + i) as u32);
        }
        ids.into_iter().map(|i| i as usize).collect()
    }

    /// Decode token IDs back to text. Unknown IDs are skipped; byte sequences
    /// that are not valid UTF-8 decode lossily (U+FFFD).
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

    /// Serialise the merge list as `"a,b;c,d;…"` (empty string = byte-level).
    /// Compact enough to embed in FINF metadata.
    pub fn encode_state(&self) -> String {
        self.merges
            .iter()
            .map(|(a, b)| format!("{a},{b}"))
            .collect::<Vec<_>>()
            .join(";")
    }

    /// Rebuild a tokenizer from [`encode_state`](Self::encode_state) output.
    pub fn from_state(state: &str) -> Result<Self> {
        let mut tok = Self::byte_level();
        if state.is_empty() {
            return Ok(tok);
        }
        for entry in state.split(';') {
            let (a, b) = entry
                .split_once(',')
                .ok_or_else(|| InferError::Format(format!("bad merge entry {entry:?}")))?;
            let a: u32 = a.trim().parse().map_err(|_| {
                InferError::ParseError(format!("bad merge id {a:?}"))
            })?;
            let b: u32 = b.trim().parse().map_err(|_| {
                InferError::ParseError(format!("bad merge id {b:?}"))
            })?;
            if a as usize >= tok.vocab.len() || b as usize >= tok.vocab.len() {
                return Err(InferError::Format(format!(
                    "merge ({a},{b}) references undefined token (vocab {})",
                    tok.vocab.len()
                )));
            }
            let merged = [tok.vocab[a as usize].clone(), tok.vocab[b as usize].clone()].concat();
            tok.vocab.push(merged);
            tok.merges.push((a, b));
        }
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
            assert_eq!(tok.decode(&tok.encode(text)), text, "roundtrip failed for {text:?}");
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
}
