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
use crate::gguf::{bounded, Gguf, MetaValue};
use crate::gguf_write::GgufBuilder;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

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
pub const TOK_EOS: &str = "<|endoftext|>";
pub const TOK_PAD: &str = "<pad>";
pub const TOK_UNK: &str = "<unk>";

/// Document separator used by prepared corpora (`train_transformer prepare`)
/// and TinyStories. [`ByteBpeTokenizer::encode_docs`] maps it to the EOS special
/// token, and [`ByteBpeTokenizer::train_with_docs`] keeps it out of the merges.
pub const DOC_SEP: &str = TOK_EOS; // "<|endoftext|>" since Phase 0-G GT

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

/// GPT-2 byte → unicode char mapping (OpenAI `bytes_to_unicode`, llama.cpp
/// `unicode_byte_to_utf8_map`): printable Latin-1 bytes map to themselves, the
/// other 68 bytes to U+0100.. in byte order.
pub fn byte_to_unicode(b: u8) -> char {
    let printable = |x: u32| (0x21..=0x7E).contains(&x) || (0xA1..=0xAC).contains(&x) || x >= 0xAE;
    let b32 = u32::from(b);
    if printable(b32) {
        return char::from(b);
    }
    let n = (0..b32).filter(|&x| !printable(x)).count() as u32;
    char::from_u32(256 + n).expect("256 + n < 324 is a valid char")
}

#[derive(Clone, Copy, PartialEq)]
enum CharClass {
    Letter,
    Number,
    Space,
    Other,
}

fn char_class(c: char) -> CharClass {
    use unicode_general_category::{get_general_category as cat, GeneralCategory as G};
    if c.is_whitespace() {
        return CharClass::Space;
    }
    match cat(c) {
        G::UppercaseLetter
        | G::LowercaseLetter
        | G::TitlecaseLetter
        | G::ModifierLetter
        | G::OtherLetter => CharClass::Letter,
        G::DecimalNumber | G::LetterNumber | G::OtherNumber => CharClass::Number,
        _ => CharClass::Other,
    }
}

/// Pre-tokenize `text` exactly like llama.cpp's `gpt-2` pre-tokenizer
/// (`unicode_regex_split_custom_gpt2`, b11521):
/// `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`.
/// The chunks are a partition of `text`, so encode/decode still round-trips and
/// learned merges never span chunks.
fn pretokenize(text: &str) -> Vec<&str> {
    let cs: Vec<(usize, char)> = text.char_indices().collect();
    let n = cs.len();
    let at = |p: usize| cs.get(p).map(|&(_, c)| c);
    let class = |p: usize| at(p).map(char_class);
    let mut chunks = Vec::new();
    let mut prev = 0; // char index where the current chunk starts
    let mut pos = 0;
    while pos < n {
        let c = cs[pos].1;
        let mut end = None;
        // Contractions.
        if c == '\'' && pos + 1 < n {
            let (c1, c2) = (cs[pos + 1].1, at(pos + 2));
            if matches!(c1, 's' | 't' | 'm' | 'd') {
                end = Some(pos + 2);
            } else if matches!(
                (c1, c2),
                ('r', Some('e')) | ('v', Some('e')) | ('l', Some('l'))
            ) {
                end = Some(pos + 3);
            }
        }
        if end.is_none() {
            // ` ?` then a run of letters / numbers / other.
            let lead = usize::from(c == ' ');
            let cl = class(pos + lead);
            if matches!(
                cl,
                Some(CharClass::Letter | CharClass::Number | CharClass::Other)
            ) {
                let mut e = pos + lead;
                while class(e) == cl {
                    e += 1;
                }
                end = Some(e);
            }
        }
        let end = end.unwrap_or_else(|| {
            // Whitespace run: leave the last one for the next word (`\s+(?!\S)`)
            // unless the run reaches the end of the text.
            let mut e = pos;
            while class(e) == Some(CharClass::Space) {
                e += 1;
            }
            if e - pos > 1 && e < n {
                e - 1
            } else {
                e.max(pos + 1)
            }
        });
        let byte = |i: usize| cs.get(i).map_or(text.len(), |&(b, _)| b);
        chunks.push(&text[byte(prev)..byte(end)]);
        prev = end;
        pos = end;
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

    /// Learn up to `vocab_size − 257` merges from `corpus`, then reserve one id
    /// for [`TOK_EOS`] (so the result has at most `vocab_size` tokens).
    ///
    /// The corpus is pre-tokenized like llama.cpp's `gpt-2` splitter so merges
    /// stay within words; training then greedily merges the most frequent
    /// adjacent pair. Stops early when no adjacent pair occurs at least twice.
    /// `vocab_size` below 257 is an error (256 bytes + EOS are irreducible).
    pub fn train(corpus: &str, vocab_size: usize) -> Result<Self> {
        if vocab_size <= BASE_VOCAB {
            return Err(InferError::DimMismatch(format!(
                "vocab_size {vocab_size} < {} (byte base vocabulary + EOS)",
                BASE_VOCAB + 1
            )));
        }
        let mut tok = Self::byte_level();
        // Pre-tokenized chunks never merge across each other, so identical
        // chunks behave identically: train on each distinct chunk once,
        // weighted by its frequency.
        let mut freq: HashMap<&str, i64> = HashMap::new();
        for c in pretokenize(corpus) {
            *freq.entry(c).or_insert(0) += 1;
        }
        let mut words: Vec<(Vec<u32>, i64)> = freq
            .into_iter()
            .map(|(c, f)| (c.bytes().map(u32::from).collect(), f))
            .collect();
        vprintln!(
            "[tokenizer::train] corpus={} bytes, {} distinct chunks, target vocab={}",
            corpus.len(),
            words.len(),
            vocab_size
        );

        // Pair counts are computed once, then updated incrementally per merge
        // (only the words containing the merged pair change), instead of
        // recounting the whole corpus on every merge.
        let mut counts: HashMap<(u32, u32), i64> = HashMap::new();
        let mut where_: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
        for (wi, (ids, f)) in words.iter().enumerate() {
            for w in ids.windows(2) {
                *counts.entry((w[0], w[1])).or_insert(0) += f;
                where_.entry((w[0], w[1])).or_default().push(wi);
            }
        }
        // Max-heap on (count, smallest pair) with lazy invalidation: an entry is
        // live only while it still matches `counts`. Ties break to the smallest
        // pair, so training is deterministic regardless of HashMap order.
        let mut heap: BinaryHeap<(i64, Reverse<(u32, u32)>)> =
            counts.iter().map(|(&p, &c)| (c, Reverse(p))).collect();

        while tok.vocab.len() < vocab_size - 1 {
            let Some((count, Reverse(pair))) = heap.pop() else {
                break;
            };
            if counts.get(&pair) != Some(&count) {
                continue; // stale entry
            }
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

            let mut touched: HashMap<(u32, u32), i64> = HashMap::new();
            let mut wis = where_.remove(&pair).unwrap_or_default();
            wis.sort_unstable();
            wis.dedup();
            for wi in wis {
                let (ids, f) = &mut words[wi];
                if !ids.windows(2).any(|w| (w[0], w[1]) == pair) {
                    continue; // stale index entry
                }
                for w in ids.windows(2) {
                    *touched.entry((w[0], w[1])).or_insert(0) -= *f;
                }
                *ids = merge_pair(ids, pair, new_id);
                for w in ids.windows(2) {
                    *touched.entry((w[0], w[1])).or_insert(0) += *f;
                    // Only pairs involving the new token are new to this word.
                    if w[0] == new_id || w[1] == new_id {
                        where_.entry((w[0], w[1])).or_default().push(wi);
                    }
                }
            }
            for (p, delta) in touched {
                if delta == 0 {
                    continue;
                }
                let c = counts.entry(p).or_insert(0);
                *c += delta;
                if *c > 0 {
                    heap.push((*c, Reverse(p)));
                } else {
                    counts.remove(&p);
                }
            }
        }
        tok.rebuild_ranks();
        // ponytail: EOS is always reserved so every exported GGUF has a real
        // `eos_token_id`; make it optional only if a no-EOS model is ever needed.
        tok.add_special_token(TOK_EOS)?;
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

    /// Like [`encode`](Self::encode), but every [`DOC_SEP`] becomes the EOS id
    /// when this tokenizer has one. Without EOS it is exactly `encode`.
    pub fn encode_docs(&self, text: &str) -> Vec<usize> {
        let Some(eos) = self.eos_id() else {
            return self.encode(text);
        };
        let mut ids = Vec::new();
        for (i, doc) in text.split(DOC_SEP).enumerate() {
            if i > 0 {
                ids.push(eos);
            }
            ids.extend(self.encode(doc));
        }
        ids
    }

    /// [`train`](Self::train) that understands document separators: they are
    /// removed before learning merges, and `train` already reserves [`TOK_EOS`]
    /// (the last id), so the total vocabulary is still `vocab_size`. Identical
    /// to `train` when the corpus has no separator.
    pub fn train_with_docs(corpus: &str, vocab_size: usize) -> Result<Self> {
        Self::train(&corpus.replace(DOC_SEP, ""), vocab_size)
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

    // ── GGUF vocab (GPT-2 bytes-to-unicode space) ─────────────────────────────

    /// `tokenizer.ggml.tokens`: each token's bytes mapped through
    /// [`byte_to_unicode`]; special tokens by name.
    pub fn gguf_tokens(&self) -> Vec<String> {
        let n_text = BASE_VOCAB + self.merges.len();
        (0..self.vocab.len())
            .map(|i| match i.checked_sub(n_text) {
                Some(s) => self.special[s].clone(),
                None => self.vocab[i].iter().map(|&b| byte_to_unicode(b)).collect(),
            })
            .collect()
    }

    /// `tokenizer.ggml.merges` in rank order: `"<left> <right>"`.
    pub fn gguf_merges(&self) -> Vec<String> {
        let toks = self.gguf_tokens();
        self.merges
            .iter()
            .map(|&(a, b)| format!("{} {}", toks[a as usize], toks[b as usize]))
            .collect()
    }

    /// `tokenizer.ggml.token_type`: 1 = normal, 3 = control (specials).
    pub fn gguf_token_types(&self) -> Vec<i32> {
        let n_text = BASE_VOCAB + self.merges.len();
        (0..self.vocab.len())
            .map(|i| if i < n_text { 1 } else { 3 })
            .collect()
    }

    /// Rebuild from the three GGUF arrays (inverse of the exporters). Expects
    /// Ferrum's layout: 256 byte tokens, then one token per merge in rank
    /// order, then control tokens. Errors on any deviation.
    pub fn from_gguf_vocab(
        tokens: &[String],
        merges: &[String],
        token_types: &[i32],
    ) -> Result<Self> {
        let bad = |m: String| InferError::Format(format!("gguf vocab: {m}"));
        let n_text = BASE_VOCAB + merges.len();
        if tokens.len() < n_text || token_types.len() != tokens.len() {
            return Err(bad(format!(
                "{} tokens / {} types for {} merges",
                tokens.len(),
                token_types.len(),
                merges.len()
            )));
        }
        let ids: HashMap<&str, u32> = tokens[..n_text]
            .iter()
            .enumerate()
            .map(|(i, t)| (t.as_str(), i as u32))
            .collect();
        let mut tok = Self::byte_level();
        for (b, t) in tokens[..BASE_VOCAB].iter().enumerate() {
            if *t != byte_to_unicode(b as u8).to_string() {
                return Err(bad(format!("token {b} is not byte {b}")));
            }
        }
        for (i, m) in merges.iter().enumerate() {
            // llama.cpp: split at the first space searching from index 1.
            let from = m.chars().next().map_or(0, char::len_utf8);
            let (l, r) = m[from..]
                .find(' ')
                .map(|j| (&m[..from + j], &m[from + j + 1..]))
                .ok_or_else(|| bad(format!("bad merge {m:?}")))?;
            let (Some(&a), Some(&b)) = (ids.get(l), ids.get(r)) else {
                return Err(bad(format!("merge {m:?} has a part that is not a token")));
            };
            if a as usize >= BASE_VOCAB + i || b as usize >= BASE_VOCAB + i {
                return Err(bad(format!("merge {m:?} uses a later token")));
            }
            let merged = [tok.vocab[a as usize].clone(), tok.vocab[b as usize].clone()].concat();
            let want: String = merged.iter().map(|&x| byte_to_unicode(x)).collect();
            if tokens[BASE_VOCAB + i] != want {
                return Err(bad(format!("token {} is not merge {m:?}", BASE_VOCAB + i)));
            }
            tok.vocab.push(merged);
            tok.merges.push((a, b));
        }
        for (k, (name, ty)) in tokens[n_text..]
            .iter()
            .zip(&token_types[n_text..])
            .enumerate()
        {
            if *ty != 3 {
                return Err(bad(format!("token {name:?} after merges is not control")));
            }
            if tok.add_special_token(name)? != n_text + k {
                return Err(bad(format!("duplicate special token {name:?}")));
            }
        }
        tok.rebuild_ranks();
        Ok(tok)
    }

    /// The reusable tokenizer file: a vocab-only GGUF (`ferrum-tokenizer`) whose
    /// `tokenizer.ggml.{model,tokens,merges,token_type}` arrays are also a valid
    /// llama.cpp vocab, plus `tokenizer.ferrum.state` = [`encode_state`](Self::encode_state)
    /// as the lossless source of truth.
    pub fn to_gguf_bytes(&self) -> Vec<u8> {
        let strings =
            |v: Vec<String>| MetaValue::Array(v.into_iter().map(MetaValue::String).collect());
        let types = self.gguf_token_types().into_iter().map(MetaValue::I32);
        let mut b = GgufBuilder::new();
        b.meta(
            "general.architecture",
            MetaValue::String("ferrum-tokenizer".into()),
        )
        .meta("tokenizer.ggml.model", MetaValue::String("gpt2".into()))
        .meta("tokenizer.ggml.tokens", strings(self.gguf_tokens()))
        .meta("tokenizer.ggml.merges", strings(self.gguf_merges()))
        .meta(
            "tokenizer.ggml.token_type",
            MetaValue::Array(types.collect()),
        )
        .meta(
            "tokenizer.ferrum.state",
            MetaValue::String(self.encode_state()),
        );
        b.into_bytes()
    }

    /// Read a [`to_gguf_bytes`](Self::to_gguf_bytes) file. The state string is
    /// authoritative and is cross-checked against the vocab arrays; a file
    /// without it is rebuilt from the arrays alone. The pre-GGUF text form is
    /// rejected with a pointed error.
    pub fn from_gguf_bytes(bytes: &[u8]) -> Result<Self> {
        let bad = |m: String| InferError::Format(m);
        if bytes.len() < 4 {
            return Err(bad("empty or truncated tokenizer file".into()));
        }
        if !bytes.starts_with(b"GGUF") {
            return Err(bad(
                "old tokenizer file (pre-GGUF); delete it to retrain the tokenizer".into(),
            ));
        }
        let g = Gguf::parse(bytes.to_vec())?;
        if g.architecture() != Some("ferrum-tokenizer") {
            return Err(bad(format!(
                "not a ferrum-tokenizer file (general.architecture = {:?})",
                g.architecture()
            )));
        }
        let array = |key: &str| match g.meta(key) {
            Some(MetaValue::Array(a)) => Ok(a),
            _ => Err(bad(format!(
                "tokenizer file key '{key}' missing or not an array"
            ))),
        };
        let strings = |key: &str| -> Result<Vec<String>> {
            array(key)?
                .iter()
                .map(|v| v.as_str().map(str::to_string))
                .collect::<Option<_>>()
                .ok_or_else(|| bad(format!("tokenizer file key '{key}' is not a string array")))
        };
        let tokens = strings("tokenizer.ggml.tokens")?;
        let merges = strings("tokenizer.ggml.merges")?;
        let types: Vec<i32> = array("tokenizer.ggml.token_type")?
            .iter()
            .map(|v| match v {
                MetaValue::I32(x) => Some(*x),
                _ => None,
            })
            .collect::<Option<_>>()
            .ok_or_else(|| bad("tokenizer file token_type is not an i32 array".into()))?;
        let Some(state) = g.meta("tokenizer.ferrum.state").and_then(MetaValue::as_str) else {
            return Self::from_gguf_vocab(&tokens, &merges, &types);
        };
        let tok = Self::from_state(state)?;
        if tok.gguf_tokens() != tokens
            || tok.gguf_merges() != merges
            || tok.gguf_token_types() != types
        {
            return Err(bad(
                "tokenizer file is corrupt: tokenizer.ferrum.state and the tokenizer.ggml.* arrays disagree"
                    .into(),
            ));
        }
        Ok(tok)
    }

    /// Serialise the merge list as `"a,b;c,d;…"`, optionally followed by a
    /// newline and the comma-separated special-token names (K3). An empty string
    /// is byte-level with no specials. Compact enough to embed in GGUF metadata.
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
        // Each merge concatenates two earlier tokens, so a chain of merges can
        // double a token's length per entry: 40 entries (~400 bytes of state)
        // would build a 1 TiB vocab. Real vocabularies hold well under a byte
        // of token text per state byte, so cap the total.
        // ponytail: 64 B per state byte + 64 MiB; a tokenizer whose tokens
        // really exceed that (trained on one giant word) would need a lazier
        // vocab representation.
        let max_bytes = state.len().saturating_mul(64).saturating_add(64 << 20);
        let mut vocab_bytes = 0usize;
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
                let len = tok.vocab[a as usize].len() + tok.vocab[b as usize].len();
                vocab_bytes = bounded(
                    vocab_bytes.saturating_add(len),
                    max_bytes,
                    "tokenizer state vocab bytes",
                )?;
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
        assert_eq!(tok.vocab_size(), 257); // 256 bytes + reserved EOS
        assert!(tok.merges.is_empty());
    }

    #[test]
    fn vocab_below_256_errors() {
        assert!(ByteBpeTokenizer::train(CORPUS, 100).is_err());
    }

    /// The pre-incremental trainer (full recount per merge), kept as an oracle.
    fn naive_merges(corpus: &str, vocab_size: usize) -> Vec<(u32, u32)> {
        let mut chunks: Vec<Vec<u32>> = pretokenize(corpus)
            .iter()
            .map(|c| c.bytes().map(u32::from).collect())
            .collect();
        let mut merges = Vec::new();
        while BASE_VOCAB + merges.len() < vocab_size - 1 {
            let mut counts: HashMap<(u32, u32), usize> = HashMap::new();
            for chunk in &chunks {
                for w in chunk.windows(2) {
                    *counts.entry((w[0], w[1])).or_insert(0) += 1;
                }
            }
            let Some((pair, count)) = counts
                .iter()
                .max_by(|a, b| a.1.cmp(b.1).then_with(|| b.0.cmp(a.0)))
                .map(|(&p, &c)| (p, c))
            else {
                break;
            };
            if count < 2 {
                break;
            }
            let new_id = (BASE_VOCAB + merges.len()) as u32;
            merges.push(pair);
            for chunk in &mut chunks {
                *chunk = merge_pair(chunk, pair, new_id);
            }
        }
        merges
    }

    #[test]
    fn incremental_training_matches_naive_oracle() {
        // Repeats, overlapping runs ("aaaa"), ties, and multibyte text.
        let corpus = format!(
            "{CORPUS} aaaa aaa abab ababab mississippi banana bandana 🌸🌸 мир мира \n\t {}",
            "the cat sat on the mat with the hat ".repeat(7)
        );
        for vocab in [257, 270, 300, 400] {
            let fast = ByteBpeTokenizer::train(&corpus, vocab).unwrap();
            assert_eq!(fast.merges, naive_merges(&corpus, vocab), "vocab {vocab}");
        }
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
    fn doubling_merge_chain_in_state_is_capped() {
        // "97,97;256,256;257,257;…": token 256+i is 2^(i+1) bytes, so 40
        // entries (~400 bytes) would build a 1 TiB vocab. Rejected near the
        // 64 MiB floor of the cap instead, fast.
        let chain = |n: u32| {
            std::iter::once("97,97".to_string())
                .chain((0..n - 1).map(|i| format!("{0},{0}", 256 + i)))
                .collect::<Vec<_>>()
                .join(";")
        };
        let t = std::time::Instant::now();
        let e = ByteBpeTokenizer::from_state(&chain(40)).err().unwrap();
        assert!(e.to_string().contains("tokenizer state vocab bytes"), "{e}");
        assert!(t.elapsed().as_secs() < 5, "took {:?}", t.elapsed());
        // A 20-long chain (1 MiB token, 2 MiB total) is within the cap.
        let tok = ByteBpeTokenizer::from_state(&chain(20)).unwrap();
        assert_eq!(tok.vocab.last().unwrap().len(), 1 << 20);
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
    fn pretokenize_partitions_text() {
        assert_eq!(pretokenize("low lower"), vec!["low", " lower"]);
        assert_eq!(pretokenize("a\n b"), vec!["a", "\n", " b"]);
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
                // GPT-2 chunks may carry one leading space, never an inner one.
                let has_inner_space = bytes.iter().skip(1).any(|&b| (b as char).is_whitespace());
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
        // train() already reserved EOS as the last id; the rest are appended.
        let [bos, eos, pad, unk] = tok.add_standard_special_tokens();
        assert_eq!([bos, eos, pad, unk], [base, base - 1, base + 1, base + 2]);
        assert_eq!(tok.vocab_size(), base + 3);
        assert_eq!(tok.eos_id(), Some(eos));
        assert_eq!(tok.bos_id(), Some(bos));
        assert_eq!(tok.num_special(), 4);
        // Adding an existing special is idempotent.
        assert_eq!(tok.add_special_token(TOK_EOS).unwrap(), eos);
        assert_eq!(tok.vocab_size(), base + 3);
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
        let plain = ByteBpeTokenizer::byte_level();
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

    // ── GPT-2 / llama.cpp compatibility (Task GT) ─────────────────────────────

    #[test]
    fn pretokenize_matches_gpt2_regex() {
        let cases: &[(&str, &[&str])] = &[
            ("Hello world", &["Hello", " world"]),
            (
                "I'm here, it's 2024!",
                &["I", "'m", " here", ",", " it", "'s", " 2024", "!"],
            ),
            ("a  b", &["a", " ", " b"]),
            ("x\n\ny", &["x", "\n", "\n", "y"]),
            ("end   ", &["end", "   "]),
            ("naïve café", &["naïve", " café"]),
            ("<|endoftext|>Once", &["<|", "endoftext", "|>", "Once"]),
        ];
        for (text, want) in cases {
            let got = pretokenize(text);
            assert_eq!(&got, want, "{text:?}");
            assert_eq!(got.concat(), *text);
        }
    }

    #[test]
    fn byte_to_unicode_matches_gpt2() {
        assert_eq!(byte_to_unicode(b'A'), 'A');
        assert_eq!(byte_to_unicode(b' '), 'Ġ');
        assert_eq!(byte_to_unicode(b'\n'), 'Ċ');
        let all: std::collections::HashSet<char> = (0..=255u8).map(byte_to_unicode).collect();
        assert_eq!(all.len(), 256);
    }

    #[test]
    fn training_reserves_eos_and_gguf_vocab_roundtrips() {
        // Needs enough distinct words that 43 merges exist (re-derived: the
        // brief's 4-word corpus runs out of repeated pairs at 272 tokens).
        let corpus: String = (0..200)
            .map(|i| format!("the cat{i} sat{} on mat{}. ", i % 13, i % 7))
            .collect::<String>()
            .repeat(2);
        let tok = ByteBpeTokenizer::train(&corpus, 300).unwrap();
        assert_eq!(tok.vocab_size(), 300);
        assert_eq!(tok.eos_id(), Some(299));
        let (t, m, ty) = (tok.gguf_tokens(), tok.gguf_merges(), tok.gguf_token_types());
        assert_eq!(t.len(), 300);
        assert_eq!(t[b' ' as usize], "Ġ");
        assert_eq!(t[299], "<|endoftext|>");
        assert_eq!(ty[299], 3);
        assert!(m.iter().all(|x| x.matches(' ').count() == 1));
        let back = ByteBpeTokenizer::from_gguf_vocab(&t, &m, &ty).unwrap();
        assert_eq!(back.encode_state(), tok.encode_state());
        let text = "the cat sat. ünïcode 123";
        assert_eq!(back.encode(text), tok.encode(text));
    }

    /// A corpus with enough distinct words to always reach the requested vocab.
    fn docs_corpus() -> String {
        (0..200)
            .map(|i| format!("the cat{i} sat{} on mat{}.<|endoftext|>", i % 13, i % 7))
            .collect::<String>()
            .repeat(2)
    }

    #[test]
    fn doc_separator_becomes_eos_when_registered() {
        let corpus = docs_corpus();
        let tok = ByteBpeTokenizer::train_with_docs(&corpus, 300).unwrap();
        assert_eq!(tok.vocab_size(), 300);
        let eos = tok.eos_id().expect("EOS registered");
        assert_eq!(DOC_SEP, TOK_EOS);
        let ids = tok.encode_docs("a.<|endoftext|>b.");
        assert_eq!(ids.iter().filter(|&&i| i == eos).count(), 1);
        assert_eq!(tok.decode(&ids), "a.b.");
        // The separator never leaks into the learned merges.
        assert!(!tok.encode(&corpus).contains(&eos));
        assert!(tok.encode_docs(&corpus).contains(&eos));
    }

    #[test]
    fn train_with_docs_without_separator_equals_plain_training() {
        let corpus = "abc abc abc abd abd abd ".repeat(10);
        let plain = ByteBpeTokenizer::train(&corpus, 262).unwrap();
        let docs = ByteBpeTokenizer::train_with_docs(&corpus, 262).unwrap();
        assert_eq!(plain.encode_state(), docs.encode_state());
        // Vocab below the irreducible minimum still errors through the wrapper.
        assert!(ByteBpeTokenizer::train_with_docs("<|endoftext|>", 256).is_err());
    }

    #[test]
    fn encode_docs_separator_positions_and_missing_eos() {
        let tok = ByteBpeTokenizer::train(&"ab ab ab ".repeat(10), 270).unwrap();
        let eos = tok.eos_id().unwrap();
        // Leading, trailing, adjacent and absent separators.
        assert_eq!(tok.encode_docs("<|endoftext|>"), vec![eos]);
        assert_eq!(
            tok.encode_docs("<|endoftext|><|endoftext|>"),
            vec![eos, eos]
        );
        let mixed = tok.encode_docs("<|endoftext|>ab<|endoftext|>");
        assert_eq!(mixed.first(), Some(&eos));
        assert_eq!(mixed.last(), Some(&eos));
        assert_eq!(tok.encode_docs("ab ab"), tok.encode("ab ab"));
        assert!(tok.encode_docs("").is_empty());
        // A tokenizer without EOS keeps the separator as literal text.
        let bare = ByteBpeTokenizer::byte_level();
        assert!(bare.eos_id().is_none());
        assert_eq!(
            bare.encode_docs("x<|endoftext|>y"),
            bare.encode("x<|endoftext|>y")
        );
        assert_eq!(
            bare.decode(&bare.encode_docs("x<|endoftext|>y")),
            "x<|endoftext|>y"
        );
    }

    #[test]
    fn from_gguf_vocab_rejects_bad_merges() {
        let tok = ByteBpeTokenizer::train(&"abab cdcd ".repeat(10), 280).unwrap();
        let (t, mut m, ty) = (tok.gguf_tokens(), tok.gguf_merges(), tok.gguf_token_types());
        m[0] = "zz\u{1F338} yy".into();
        assert!(ByteBpeTokenizer::from_gguf_vocab(&t, &m, &ty).is_err());
    }
    fn pt(text: &str) -> Vec<&str> {
        let got = pretokenize(text);
        assert_eq!(got.concat(), text, "chunks must partition {text:?}");
        assert!(got.iter().all(|c| !c.is_empty()), "empty chunk in {got:?}");
        got
    }

    #[test]
    fn byte_to_unicode_full_table_matches_openai_algorithm() {
        // OpenAI gpt-2 `bytes_to_unicode()`, computed here independently.
        let mut bs: Vec<u32> = (0x21..=0x7E)
            .chain(0xA1..=0xAC)
            .chain(0xAE..=0xFF)
            .collect();
        let mut cs = bs.clone();
        let mut n = 0;
        for b in 0..256u32 {
            if !bs.contains(&b) {
                bs.push(b);
                cs.push(256 + n);
                n += 1;
            }
        }
        assert_eq!((bs.len(), n), (256, 68));
        for (b, c) in bs.iter().zip(&cs) {
            assert_eq!(byte_to_unicode(*b as u8) as u32, *c, "byte {b:#04x}");
        }
        // spot checks of the published table
        assert_eq!(byte_to_unicode(0x00), '\u{100}');
        assert_eq!(byte_to_unicode(0x20), '\u{120}'); // Ġ
        assert_eq!(byte_to_unicode(0x7F), '\u{121}');
        assert_eq!(byte_to_unicode(0xA0), '\u{142}');
        assert_eq!(byte_to_unicode(0xAD), '\u{143}');
        assert_eq!(byte_to_unicode(0xFF), 'ÿ');
        let uniq: std::collections::HashSet<char> = (0..=255u8).map(byte_to_unicode).collect();
        assert_eq!(uniq.len(), 256, "table is injective");
    }

    #[test]
    fn pretokenize_contractions() {
        assert_eq!(pt("don't"), ["don", "'t"]);
        assert_eq!(
            pt("I've we're they'll he'd she's I'm"),
            ["I", "'ve", " we", "'re", " they", "'ll", " he", "'d", " she", "'s", " I", "'m"]
        );
        // case-sensitive, and incomplete suffixes are plain punctuation
        assert_eq!(pt("'S"), ["'", "S"]);
        assert_eq!(pt("'RE"), ["'", "RE"]);
        assert_eq!(pt("'r"), ["'", "r"]);
        assert_eq!(pt("'l"), ["'", "l"]);
        assert_eq!(pt("'v"), ["'", "v"]);
        assert_eq!(pt("it'"), ["it", "'"]);
        assert_eq!(pt("'re"), ["'re"]);
        assert_eq!(pt("'ll"), ["'ll"]);
        assert_eq!(pt("x'rex"), ["x", "'re", "x"]);
        assert_eq!(pt("''s"), ["''", "s"]); // contraction only at the start of a token
        assert_eq!(pt("'t'd"), ["'t", "'d"]);
        assert_eq!(pt("a 's"), ["a", " '", "s"]); // space-led: ` ?[^\s\p{L}\p{N}]+` wins
    }

    #[test]
    fn pretokenize_whitespace_rules() {
        assert_eq!(pt(""), Vec::<&str>::new());
        assert_eq!(pt(" "), [" "]);
        assert_eq!(pt("   "), ["   "]);
        assert_eq!(pt("\n\n\n"), ["\n\n\n"]);
        assert_eq!(pt("  x"), [" ", " x"]);
        assert_eq!(pt("x \n"), ["x", " \n"]);
        assert_eq!(pt("a\n\n b"), ["a", "\n\n", " b"]);
        assert_eq!(pt("a\tb"), ["a", "\t", "b"]);
        assert_eq!(pt("a \tb"), ["a", " ", "\t", "b"]);
        assert_eq!(pt("a  !!"), ["a", " ", " !!"]);
        assert_eq!(pt("\n word"), ["\n", " word"]);
        assert_eq!(pt("\r\n\r\nx"), ["\r\n\r", "\n", "x"]);
        // Unicode White_Space is whitespace; a lone leading ' ' only attaches to the next word
        assert_eq!(pt("a\u{a0}b"), ["a", "\u{a0}", "b"]);
        assert_eq!(pt("\u{2003}x"), ["\u{2003}", "x"]);
        assert_eq!(pt(" \u{a0}b"), [" ", "\u{a0}", "b"]);
        assert_eq!(pt("a\u{3000}\u{3000}"), ["a", "\u{3000}\u{3000}"]);
    }

    #[test]
    fn pretokenize_char_classes() {
        assert_eq!(pt("a1b2"), ["a", "1", "b", "2"]);
        assert_eq!(pt(" 12ab"), [" 12", "ab"]);
        assert_eq!(pt("x 3.14"), ["x", " 3", ".", "14"]);
        assert_eq!(pt(" !"), [" !"]);
        assert_eq!(pt("?!.. x"), ["?!..", " x"]);
        // No (superscript), Nl (roman numeral) are numbers; Nd in other scripts too
        assert_eq!(pt("a\u{b2}b"), ["a", "\u{b2}", "b"]);
        assert_eq!(pt("x\u{2167}y"), ["x", "\u{2167}", "y"]);
        assert_eq!(pt(" \u{663}\u{664}z"), [" \u{663}\u{664}", "z"]);
        // letters: Lo (CJK/kana), Ll/Lu (Greek), Lm (modifier), Lt (titlecase)
        assert_eq!(pt("日本語 テスト"), ["日本語", " テスト"]);
        assert_eq!(pt("αβγ Δ"), ["αβγ", " Δ"]);
        assert_eq!(pt("a\u{2b0}b"), ["a\u{2b0}b"]);
        assert_eq!(pt("\u{1c5}x"), ["\u{1c5}x"]);
        // combining marks are not letters: they split off as punctuation-like runs
        assert_eq!(pt("e\u{301}x"), ["e", "\u{301}", "x"]);
        // symbols, emoji, ZWJ (Cf), controls and private use are all `[^\s\p{L}\p{N}]`
        assert_eq!(pt("😀😀 😀"), ["😀😀", " 😀"]);
        assert_eq!(pt("a\u{200d}b"), ["a", "\u{200d}", "b"]);
        assert_eq!(pt("\u{1}\u{2}x"), ["\u{1}\u{2}", "x"]);
        assert_eq!(pt("\u{e000}\u{e001}"), ["\u{e000}\u{e001}"]);
    }

    #[test]
    fn pretokenize_unassigned_codepoints_lump_with_other_like_llama_cpp() {
        // llama.cpp b11521's table marks Cn as UNDEFINED (a nonzero flag), so its
        // `[^\s\p{L}\p{N}]+` branch accepts them: they join the punctuation run
        // rather than becoming one chunk each.
        assert!(char::from_u32(0x378).is_some());
        assert_eq!(pt("a\u{378}\u{379}b"), ["a", "\u{378}\u{379}", "b"]);
        assert_eq!(pt(" \u{378}z"), [" \u{378}", "z"]);
        assert_eq!(pt("!\u{378}?"), ["!\u{378}?"]);
    }

    #[test]
    fn pretokenize_is_a_partition_for_mixed_text() {
        let text = "Héllo, wörld!! It's 3.14 — ok?\n\n\tTab\u{a0}end  \r\n日本 😀 'x'  ";
        let chunks = pt(text);
        assert!(chunks.len() > 15);
        let tok = ByteBpeTokenizer::train(&text.repeat(5), 300).unwrap();
        assert_eq!(tok.decode(&tok.encode(text)), text);
    }

    #[test]
    fn merges_never_cross_pretokenized_chunks() {
        // "ab" appears only across the chunk boundary of "a" + " b"-style splits
        let tok = ByteBpeTokenizer::train(&"a,b ".repeat(30), 300).unwrap();
        let ids = tok.encode("a,b");
        for id in ids {
            if let Some(bytes) = tok.token_bytes(id) {
                assert!(
                    !(bytes.len() > 1 && bytes.contains(&b'a') && bytes.contains(&b',')),
                    "token {id} spans letter and punctuation: {bytes:?}"
                );
            }
        }
    }

    fn small_tok() -> ByteBpeTokenizer {
        ByteBpeTokenizer::train(&"abab cdcd ünï ".repeat(12), 290).unwrap()
    }

    #[test]
    fn eos_reservation_edge_cases() {
        // 257 = 256 bytes + EOS and nothing else
        let t = ByteBpeTokenizer::train("aaaa bbbb", 257).unwrap();
        assert_eq!(
            (t.vocab_size(), t.eos_id(), t.num_special()),
            (257, Some(256), 1)
        );
        assert_eq!(t.gguf_merges(), Vec::<String>::new());
        assert_eq!(t.gguf_token_types().last(), Some(&3));
        // EOS lands directly after the last merge, wherever training stopped
        let t = ByteBpeTokenizer::train("ab", 400).unwrap();
        assert_eq!(
            t.vocab_size(),
            257,
            "no repeated pair -> no merges, still EOS"
        );
        assert_eq!(t.eos_id(), Some(256));
        let t = small_tok();
        assert_eq!(t.eos_id(), Some(t.vocab_size() - 1));
        assert!(t.vocab_size() <= 290);
        assert_eq!(
            t.decode(&[t.eos_id().unwrap()]),
            "",
            "EOS decodes to nothing"
        );
        assert_eq!(t.pad_id(), None);
        assert_eq!(t.unk_id(), None);
        assert_eq!(t.bos_id(), None);
        assert!(!t.encode("<|endoftext|>").contains(&t.eos_id().unwrap()));
    }

    #[test]
    fn standard_special_tokens_ids_and_gguf_types() {
        let mut t = small_tok();
        let n_text = t.vocab_size() - 1;
        let ids = t.add_standard_special_tokens();
        assert_eq!(ids, [n_text + 1, n_text, n_text + 2, n_text + 3]);
        assert_eq!(t.pad_id(), Some(n_text + 2));
        assert_eq!(t.unk_id(), Some(n_text + 3));
        let types = t.gguf_token_types();
        assert_eq!(types.len(), t.vocab_size());
        assert!(types[..n_text].iter().all(|&x| x == 1));
        assert!(types[n_text..].iter().all(|&x| x == 3));
        let toks = t.gguf_tokens();
        assert_eq!(
            &toks[n_text..],
            ["<|endoftext|>", "<bos>", "<pad>", "<unk>"]
        );
        let back = ByteBpeTokenizer::from_gguf_vocab(&toks, &t.gguf_merges(), &types).unwrap();
        assert_eq!(back.encode_state(), t.encode_state());
        assert_eq!(back.unk_id(), t.unk_id());
    }

    #[test]
    fn gguf_tokens_map_every_byte_through_the_table() {
        let t = small_tok();
        let toks = t.gguf_tokens();
        for b in 0..=255u8 {
            assert_eq!(toks[b as usize], byte_to_unicode(b).to_string(), "byte {b}");
        }
        // a merged token is the concatenation of its parts' mapped chars
        let m = t.gguf_merges();
        for (i, line) in m.iter().enumerate() {
            let (l, r) = line.split_once(' ').unwrap();
            assert_eq!(format!("{l}{r}"), toks[256 + i]);
        }
        // multi-byte UTF-8 (ü = C3 BC) is two mapped chars (Ã¼), not one raw ü
        assert_eq!((toks[0xC3].as_str(), toks[0xBC].as_str()), ("Ã", "¼"));
    }

    fn gguf_parts(t: &ByteBpeTokenizer) -> (Vec<String>, Vec<String>, Vec<i32>) {
        (t.gguf_tokens(), t.gguf_merges(), t.gguf_token_types())
    }

    #[test]
    fn from_gguf_vocab_error_paths() {
        let t = small_tok();
        let (tk, mg, ty) = gguf_parts(&t);
        assert!(mg.len() >= 3);
        let err = |tk: &[String], mg: &[String], ty: &[i32]| {
            ByteBpeTokenizer::from_gguf_vocab(tk, mg, ty)
                .err()
                .map(|e| e.to_string())
        };
        // lengths
        assert!(err(&tk[..tk.len() - 3], &mg, &ty[..ty.len() - 3])
            .unwrap()
            .contains("tokens"));
        assert!(err(&tk, &mg, &ty[1..]).unwrap().contains("types"));
        // byte table
        let mut bad = tk.clone();
        bad.swap(65, 66);
        assert!(err(&bad, &mg, &ty).unwrap().contains("is not byte"));
        // merge line without a separator / with unknown parts / forward reference
        let mut m = mg.clone();
        m[1] = "nospace".into();
        assert!(err(&tk, &m, &ty).unwrap().contains("bad merge"));
        let mut m = mg.clone();
        m[1] = "ZZ YY".into();
        assert!(err(&tk, &m, &ty).unwrap().contains("not a token"));
        let mut m = mg.clone();
        let j = t
            .merges
            .iter()
            .position(|&(a, b)| a >= 256 || b >= 256)
            .unwrap();
        m[0] = mg[j].clone(); // refers to a merged token that does not exist yet
        assert!(err(&tk, &m, &ty).unwrap().contains("later token"));
        // token that is not the concatenation of its merge
        let mut bad = tk.clone();
        bad.swap(256, 257);
        assert!(err(&bad, &mg, &ty).unwrap().contains("is not merge"));
        // non-control token after the merges
        let mut ty2 = ty.clone();
        *ty2.last_mut().unwrap() = 1;
        assert!(err(&tk, &mg, &ty2).unwrap().contains("not control"));
        // invalid special name (contains ',')
        let mut bad = tk.clone();
        *bad.last_mut().unwrap() = "a,b".into();
        assert!(err(&bad, &mg, &ty).is_some());
        // empty merge list / no specials is fine
        let byte_only = ByteBpeTokenizer::from_gguf_vocab(&tk[..256], &[], &ty[..256]).unwrap();
        assert_eq!((byte_only.vocab_size(), byte_only.num_special()), (256, 0));
    }

    #[test]
    fn from_gguf_vocab_rejects_duplicate_special_names() {
        let mut t = small_tok();
        t.add_special_token("<pad>").unwrap();
        let (mut tk, mg, mut ty) = gguf_parts(&t);
        tk.push("<pad>".into()); // same name twice: ids after it would silently shift
        ty.push(3);
        assert!(ByteBpeTokenizer::from_gguf_vocab(&tk, &mg, &ty).is_err());
    }

    #[test]
    fn from_gguf_vocab_multibyte_first_char_split_follows_llama_cpp() {
        // Merge lines split at the first space after index 1, so a left token whose
        // first char is multi-byte (e.g. 'Ġ') still parses.
        let t = ByteBpeTokenizer::train(&"the the the the ".repeat(10), 300).unwrap();
        let (tk, mg, ty) = gguf_parts(&t);
        assert!(mg.iter().any(|m| m.starts_with('Ġ')), "{mg:?}");
        let back = ByteBpeTokenizer::from_gguf_vocab(&tk, &mg, &ty).unwrap();
        assert_eq!(back.encode(" the the"), t.encode(" the the"));
    }
    fn trained_tok() -> ByteBpeTokenizer {
        let mut tok = ByteBpeTokenizer::train("low lower lowest low low", 300).unwrap();
        tok.add_special_token(TOK_EOS).unwrap();
        tok
    }

    #[test]
    fn tokenizer_file_is_gguf_and_roundtrips() {
        let tok = trained_tok();
        let b = tok.to_gguf_bytes();
        assert_eq!(&b[..4], b"GGUF");
        let back = ByteBpeTokenizer::from_gguf_bytes(&b).unwrap();
        assert_eq!(back.encode_state(), tok.encode_state());
        assert_eq!(back.gguf_tokens(), tok.gguf_tokens());
        assert!(ByteBpeTokenizer::from_gguf_bytes(b"256;257").is_err());
        let g = crate::gguf::Gguf::parse(b).unwrap();
        assert_eq!(g.architecture(), Some("ferrum-tokenizer"));
        assert_eq!(
            g.meta("tokenizer.ggml.model").and_then(|v| v.as_str()),
            Some("gpt2")
        );
        for k in ["tokens", "merges", "token_type"] {
            assert!(g.meta(&format!("tokenizer.ggml.{k}")).is_some(), "{k}");
        }
        assert_eq!(
            g.meta("tokenizer.ferrum.state").and_then(|v| v.as_str()),
            Some(tok.encode_state().as_str())
        );
    }

    #[test]
    fn tokenizer_file_roundtrips_byte_level_and_no_specials() {
        for tok in [
            ByteBpeTokenizer::byte_level(),
            ByteBpeTokenizer::train("low lower lowest low low", 300).unwrap(),
        ] {
            let back = ByteBpeTokenizer::from_gguf_bytes(&tok.to_gguf_bytes()).unwrap();
            assert_eq!(back.encode_state(), tok.encode_state());
            assert_eq!(back.encode("low lowest"), tok.encode("low lowest"));
        }
    }

    #[test]
    fn tokenizer_file_old_text_form_is_a_clear_error() {
        let old = trained_tok().encode_state();
        for bytes in [old.as_bytes(), b"256;257"] {
            let e = ByteBpeTokenizer::from_gguf_bytes(bytes)
                .err()
                .unwrap()
                .to_string();
            assert!(e.contains("old tokenizer file (pre-GGUF)"), "{e}");
            assert!(e.contains("delete it to retrain the tokenizer"), "{e}");
        }
    }

    #[test]
    fn tokenizer_file_empty_or_tiny_is_truncated_not_old() {
        for bytes in [&b""[..], b"G", b"256"] {
            let e = ByteBpeTokenizer::from_gguf_bytes(bytes)
                .err()
                .unwrap()
                .to_string();
            assert!(e.contains("empty or truncated tokenizer file"), "{e}");
            assert!(!e.contains("pre-GGUF"), "{e}");
        }
    }

    /// An array count far beyond the file size is rejected by the container
    /// parser before anything is reserved or built from it.
    #[test]
    fn tokenizer_file_array_count_is_bounded_by_file_size() {
        let mut f = Vec::new();
        f.extend_from_slice(b"GGUF");
        f.extend_from_slice(&3u32.to_le_bytes());
        f.extend_from_slice(&0u64.to_le_bytes()); // tensors
        f.extend_from_slice(&2u64.to_le_bytes()); // kv
        let key = |f: &mut Vec<u8>, k: &str| {
            f.extend_from_slice(&(k.len() as u64).to_le_bytes());
            f.extend_from_slice(k.as_bytes());
        };
        key(&mut f, "general.architecture");
        f.extend_from_slice(&8u32.to_le_bytes());
        f.extend_from_slice(&16u64.to_le_bytes());
        f.extend_from_slice(b"ferrum-tokenizer");
        key(&mut f, "tokenizer.ggml.tokens");
        f.extend_from_slice(&9u32.to_le_bytes()); // array
        f.extend_from_slice(&8u32.to_le_bytes()); // of strings
        f.extend_from_slice(&(u64::MAX / 2).to_le_bytes());
        let e = ByteBpeTokenizer::from_gguf_bytes(&f).err().unwrap();
        assert!(e.to_string().contains("array count exceeds"), "{e}");
    }

    #[test]
    fn tokenizer_file_rejects_truncation_wrong_architecture_and_bad_state() {
        use crate::gguf::MetaValue;
        let b = trained_tok().to_gguf_bytes();
        assert!(ByteBpeTokenizer::from_gguf_bytes(&b[..b.len() / 2]).is_err());
        assert!(ByteBpeTokenizer::from_gguf_bytes(b"GGUF").is_err());
        let set = |k: &str, v: MetaValue| {
            let k = k.to_string();
            crate::ckpt::rewrite(
                &b,
                move |m| {
                    m.insert(k, v);
                },
                None,
            )
        };
        let e = ByteBpeTokenizer::from_gguf_bytes(&set(
            "general.architecture",
            MetaValue::String("gpt2".into()),
        ))
        .err()
        .unwrap();
        assert!(e.to_string().contains("not a ferrum-tokenizer"), "{e}");
        let mut no_arch = b.clone();
        no_arch = crate::ckpt::rewrite(&no_arch, |m| drop(m.remove("general.architecture")), None);
        assert!(ByteBpeTokenizer::from_gguf_bytes(&no_arch).is_err());
        // A state string that does not parse.
        let e = ByteBpeTokenizer::from_gguf_bytes(&set(
            "tokenizer.ferrum.state",
            MetaValue::String("not-a-pair".into()),
        ));
        assert!(e.is_err());
        // A state that disagrees with the arrays (different merge count).
        let other = ByteBpeTokenizer::train("abab abab abab", 270)
            .unwrap()
            .encode_state();
        let e = ByteBpeTokenizer::from_gguf_bytes(&set(
            "tokenizer.ferrum.state",
            MetaValue::String(other),
        ))
        .err()
        .unwrap();
        assert!(e.to_string().contains("disagree"), "{e}");
        // Arrays edited while the state is intact.
        for key in ["tokens", "merges", "token_type"] {
            let key = format!("tokenizer.ggml.{key}");
            let bad = crate::ckpt::rewrite(
                &b,
                |m| {
                    if let Some(MetaValue::Array(a)) = m.get_mut(&key) {
                        a.pop();
                    }
                },
                None,
            );
            let e = ByteBpeTokenizer::from_gguf_bytes(&bad).err().unwrap();
            assert!(e.to_string().contains("disagree"), "{key}: {e}");
        }
        // Arrays removed or mistyped.
        for key in ["tokens", "merges", "token_type"] {
            let key = format!("tokenizer.ggml.{key}");
            let gone = crate::ckpt::rewrite(&b, |m| drop(m.remove(&key)), None);
            assert!(
                ByteBpeTokenizer::from_gguf_bytes(&gone).is_err(),
                "{key} gone"
            );
            let wrong = crate::ckpt::rewrite(
                &b,
                |m| {
                    m.insert(key.clone(), MetaValue::Array(vec![MetaValue::U32(1)]));
                },
                None,
            );
            assert!(
                ByteBpeTokenizer::from_gguf_bytes(&wrong).is_err(),
                "{key} type"
            );
        }
    }

    #[test]
    fn tokenizer_file_without_state_falls_back_to_the_arrays() {
        let tok = trained_tok();
        let b = crate::ckpt::rewrite(
            &tok.to_gguf_bytes(),
            |m| drop(m.remove("tokenizer.ferrum.state")),
            None,
        );
        let back = ByteBpeTokenizer::from_gguf_bytes(&b).unwrap();
        assert_eq!(back.encode_state(), tok.encode_state());
        // Without state, broken arrays are rejected by the vocab parser.
        let broken = crate::ckpt::rewrite(
            &b,
            |m| {
                if let Some(crate::gguf::MetaValue::Array(a)) = m.get_mut("tokenizer.ggml.merges") {
                    a[0] = crate::gguf::MetaValue::String("zz zz".into());
                }
            },
            None,
        );
        assert!(ByteBpeTokenizer::from_gguf_bytes(&broken).is_err());
    }

    /// A tokenizer file normally has no tensors; one that carries two
    /// overlapping ones is rejected by the shared GGUF extent check.
    #[test]
    fn tokenizer_file_rejects_overlapping_tensors() {
        let good = trained_tok().to_gguf_bytes();
        let mut b = GgufBuilder::new();
        for (k, v) in &crate::gguf::Gguf::parse(good).unwrap().metadata {
            b.meta(k, v.clone());
        }
        b.tensor("a", &[8], crate::gguf::GGML_F32, vec![0; 32]);
        b.tensor("b", &[8], crate::gguf::GGML_F32, vec![0; 32]);
        let mut bytes = b.into_bytes();
        ByteBpeTokenizer::from_gguf_bytes(&bytes).unwrap(); // control
        crate::gguf::patch_tensor_offset(&mut bytes, 1, 16);
        let e = ByteBpeTokenizer::from_gguf_bytes(&bytes).err().unwrap();
        assert!(e.to_string().contains("overlap"), "{e}");
    }
}
