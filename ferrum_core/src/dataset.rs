//! Corpus acquisition helpers: cleaning raw text into a shape the SLM training
//! paths expect, and summarising it.
//!
//! Unicode NFKC normalization is provided by the `unicode-normalization` crate.
//! Downloading is intentionally **not** here — `ferrum_core` has no HTTP client. An embedding application (the GUI,
//! a script) fetches bytes however it likes and then calls [`clean_corpus`] /
//! [`corpus_stats`] / [`validate_for_training`], all of which are pure string
//! functions and fully tested.

use crate::error::{InferError, Result};

/// Knobs controlling [`clean_corpus`]. `Default` is a sensible "tidy a plain
/// English text file" preset: CRLF stripped, Project-Gutenberg boilerplate
/// removed, blank-line runs collapsed, smart punctuation folded to ASCII, and
/// the text left in its original case with no length cap.
#[derive(Clone, Debug, PartialEq)]
pub struct CleanOptions {
    /// Remove the Project Gutenberg header/footer, keeping only the body between
    /// the `*** START OF ... ***` and `*** END OF ... ***` markers.
    pub strip_gutenberg: bool,
    /// Lowercase the whole corpus (shrinks the character/BPE vocabulary).
    pub lowercase: bool,
    /// Collapse runs of 3+ newlines into a single blank line and trim trailing
    /// spaces on each line.
    pub collapse_whitespace: bool,
    /// Fold curly quotes, en/em dashes, and the ellipsis character to ASCII.
    pub normalize_punctuation: bool,
    /// Drop ASCII control characters other than `\n` and `\t`.
    pub strip_control_chars: bool,
    /// Truncate the cleaned corpus to at most this many characters (`None` = no
    /// cap). Applied last.
    pub max_chars: Option<usize>,
}

impl Default for CleanOptions {
    fn default() -> Self {
        Self {
            strip_gutenberg: true,
            lowercase: false,
            collapse_whitespace: true,
            normalize_punctuation: true,
            strip_control_chars: true,
            max_chars: None,
        }
    }
}

/// Summary statistics of a corpus, used to sanity-check a dataset before
/// training (e.g. the character-level vocabulary size is `unique_chars`).
#[derive(Clone, Debug, PartialEq)]
pub struct CorpusStats {
    /// Total Unicode scalar values.
    pub chars: usize,
    /// Total bytes (UTF-8).
    pub bytes: usize,
    /// Number of `\n`-separated lines.
    pub lines: usize,
    /// Whitespace-separated word count.
    pub words: usize,
    /// Distinct characters — the char-level model's vocabulary size.
    pub unique_chars: usize,
}

/// Clean `raw` text into a training-ready corpus according to `opts`.
///
/// The steps run in a fixed order: Gutenberg stripping → CRLF/`\r` removal →
/// control-char filtering → punctuation folding → lowercasing → whitespace
/// collapsing → length cap. Every step is optional via [`CleanOptions`]; with
/// all flags off the only transformation is `\r` removal (the engine never wants
/// carriage returns, matching `tokenize_corpus`).
pub fn clean_corpus(raw: &str, opts: &CleanOptions) -> String {
    let joined;
    let body = if opts.strip_gutenberg {
        joined = gutenberg_bodies(raw).join("\n\n");
        joined.as_str()
    } else {
        raw
    };

    let mut out = String::with_capacity(body.len());
    for ch in body.chars() {
        // `\r` is always dropped — the tokenizers filter it out anyway, so
        // removing it here keeps stats and the saved file consistent.
        if ch == '\r' {
            continue;
        }
        if opts.strip_control_chars
            && ((ch.is_control() && ch != '\n' && ch != '\t') || is_invisible(ch))
        {
            continue;
        }
        let mapped = if opts.normalize_punctuation {
            fold_punctuation(ch)
        } else {
            None
        };
        match mapped {
            Some(s) => out.push_str(s),
            None => {
                if opts.lowercase {
                    out.extend(ch.to_lowercase());
                } else {
                    out.push(ch);
                }
            }
        }
    }

    if opts.collapse_whitespace {
        out = collapse_whitespace(&out);
    }

    if let Some(max) = opts.max_chars {
        if out.chars().count() > max {
            out = out.chars().take(max).collect();
        }
    }

    out
}

/// One curly-quote / dash / ellipsis → ASCII replacement, if `ch` is one of the
/// folded characters. Multi-character outputs (the ellipsis) return a `&str`.
fn fold_punctuation(ch: char) -> Option<&'static str> {
    Some(match ch {
        '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{2032}' => "'", // ‘ ’ ‚ ′
        '\u{201C}' | '\u{201D}' | '\u{201E}' | '\u{2033}' => "\"", // “ ” „ ″
        '\u{2013}' | '\u{2014}' | '\u{2015}' => "-",              // – — ―
        '\u{2026}' => "...",                                      // …
        '\u{00A0}' | '\u{2007}' | '\u{202F}' => " ",              // nbsp variants
        '\u{2022}' => "*",                                        // •
        _ => return None,
    })
}

/// Collapse trailing line whitespace and runs of 3+ blank lines, and trim the
/// document's leading/trailing blank lines.
fn collapse_whitespace(text: &str) -> String {
    let mut lines: Vec<&str> = text.lines().map(|l| l.trim_end()).collect();
    // Trim leading and trailing empty lines.
    while lines.first().is_some_and(|l| l.is_empty()) {
        lines.remove(0);
    }
    while lines.last().is_some_and(|l| l.is_empty()) {
        lines.pop();
    }

    let mut out = String::with_capacity(text.len());
    let mut blank_run = 0usize;
    for line in lines {
        if line.is_empty() {
            blank_run += 1;
            // Allow at most one consecutive blank line.
            if blank_run >= 2 {
                continue;
            }
        } else {
            blank_run = 0;
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

/// Characters that render as nothing but break tokenization and dedup.
fn is_invisible(ch: char) -> bool {
    matches!(
        ch,
        '\u{FEFF}' | '\u{200B}' | '\u{200C}' | '\u{200D}' | '\u{2060}'
    )
}

/// Remove the BOM, zero-width characters, `\r`, and control characters other
/// than `\n` and `\t`.
pub fn strip_invisible(text: &str) -> String {
    text.chars()
        .filter(|&c| !(is_invisible(c) || c == '\r' || (c.is_control() && c != '\n' && c != '\t')))
        .collect()
}

/// The body of every Project Gutenberg book in `raw`: the text between each
/// `*** START OF … ***` line and the following `*** END OF` marker, in order.
/// Returns `vec![raw]` when no complete START/END pair exists.
pub fn gutenberg_bodies(raw: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut rest = 0usize;
    while let Some(s) = find_marker(&raw[rest..], "START OF") {
        let s_abs = rest + s;
        let Some(body_start) = raw[s_abs..].find('\n').map(|j| s_abs + j + 1) else {
            break;
        };
        let Some(e) = find_marker(&raw[body_start..], "END OF") else {
            break;
        };
        let e_abs = body_start + e;
        out.push(&raw[body_start..e_abs]);
        rest = raw[e_abs..].find('\n').map_or(raw.len(), |j| e_abs + j + 1);
    }
    if out.is_empty() {
        vec![raw]
    } else {
        out
    }
}

/// Offset of `*** <what>` or `***<what>` in `s`, whichever comes first.
fn find_marker(s: &str, what: &str) -> Option<usize> {
    let a = s.find(&format!("*** {what}"));
    let b = s.find(&format!("***{what}"));
    match (a, b) {
        (Some(x), Some(y)) => Some(x.min(y)),
        (x, y) => x.or(y),
    }
}

/// Collapse runs of spaces and tabs within each line to a single space.
pub fn squeeze_spaces(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut in_run = false;
    for ch in text.chars() {
        if ch == ' ' || ch == '\t' {
            if !in_run {
                out.push(' ');
            }
            in_run = true;
        } else {
            out.push(ch);
            in_run = false;
        }
    }
    out
}

/// Unicode NFKC normalization (compatibility forms such as ligatures,
/// full-width letters, and circled digits fold to their plain equivalents).
pub fn normalize_nfkc(text: &str) -> String {
    use unicode_normalization::UnicodeNormalization;
    text.nfkc().collect()
}

/// Compute [`CorpusStats`] for `text`.
pub fn corpus_stats(text: &str) -> CorpusStats {
    use std::collections::HashSet;
    let mut seen: HashSet<char> = HashSet::new();
    let mut chars = 0usize;
    for c in text.chars() {
        chars += 1;
        seen.insert(c);
    }
    CorpusStats {
        chars,
        bytes: text.len(),
        lines: text.lines().count(),
        words: text.split_whitespace().count(),
        unique_chars: seen.len(),
    }
}

/// Validate that a cleaned corpus can train a model with the given
/// `context_len`. Returns a descriptive error if it is empty or too short to
/// produce even one `(context, next)` training window.
pub fn validate_for_training(text: &str, context_len: usize) -> Result<()> {
    if context_len == 0 {
        return Err(InferError::DimMismatch("context_len must be > 0".into()));
    }
    let chars = text.chars().filter(|&c| c != '\r').count();
    if chars == 0 {
        return Err(InferError::DimMismatch(
            "corpus is empty after cleaning".into(),
        ));
    }
    if chars < context_len + 1 {
        return Err(InferError::DimMismatch(format!(
            "corpus has {chars} characters but needs at least {} (context_len + 1)",
            context_len + 1
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_removes_carriage_returns() {
        let cleaned = clean_corpus("a\r\nb\r\n", &CleanOptions::default());
        assert!(!cleaned.contains('\r'));
        assert!(cleaned.contains("a\nb"));
    }

    #[test]
    fn lowercase_option_folds_case() {
        let opts = CleanOptions {
            lowercase: true,
            ..CleanOptions::default()
        };
        assert_eq!(clean_corpus("ABCdef", &opts).trim(), "abcdef");
    }

    #[test]
    fn punctuation_is_folded_to_ascii() {
        let raw = "\u{201C}quote\u{201D} \u{2018}q\u{2019} dash\u{2014}dash ellipsis\u{2026}";
        let cleaned = clean_corpus(raw, &CleanOptions::default());
        assert!(cleaned.contains("\"quote\""));
        assert!(cleaned.contains("'q'"));
        assert!(cleaned.contains("dash-dash"));
        assert!(cleaned.contains("ellipsis..."));
        // No smart punctuation should remain.
        for bad in [
            '\u{201C}', '\u{201D}', '\u{2018}', '\u{2019}', '\u{2014}', '\u{2026}',
        ] {
            assert!(!cleaned.contains(bad), "found {bad:?}");
        }
    }

    #[test]
    fn control_chars_stripped_but_tabs_and_newlines_kept() {
        let raw = "a\u{0007}b\tc\nd"; // bell char removed; tab/newline kept
        let cleaned = clean_corpus(raw, &CleanOptions::default());
        assert!(!cleaned.contains('\u{0007}'));
        assert!(cleaned.contains('\t'));
        assert!(cleaned.contains('\n'));
    }

    #[test]
    fn whitespace_collapsing_limits_blank_runs_and_trims() {
        let raw = "\n\n\nhello   \n\n\n\nworld\n\n\n";
        let cleaned = clean_corpus(raw, &CleanOptions::default());
        // Leading/trailing blank lines trimmed.
        assert!(cleaned.starts_with("hello"));
        assert!(cleaned.trim_end().ends_with("world"));
        // No run of 2+ consecutive blank lines (i.e. no triple newline).
        assert!(!cleaned.contains("\n\n\n"));
        // Trailing spaces on the "hello" line are trimmed.
        assert!(cleaned.contains("hello\n"));
    }

    #[test]
    fn gutenberg_boilerplate_is_stripped() {
        let raw = "The Project Gutenberg eBook ...\nlicense junk\n\
            *** START OF THE PROJECT GUTENBERG EBOOK FOO ***\n\
            real body text here\n\
            *** END OF THE PROJECT GUTENBERG EBOOK FOO ***\n\
            footer license junk\n";
        let cleaned = clean_corpus(raw, &CleanOptions::default());
        assert!(cleaned.contains("real body text here"));
        assert!(!cleaned.contains("license junk"));
        assert!(!cleaned.contains("START OF"));
    }

    #[test]
    fn gutenberg_passthrough_when_markers_absent() {
        let raw = "just a normal corpus with no markers\n";
        let opts = CleanOptions {
            collapse_whitespace: false,
            ..CleanOptions::default()
        };
        assert_eq!(
            clean_corpus(raw, &opts),
            "just a normal corpus with no markers\n"
        );
    }

    #[test]
    fn max_chars_truncates() {
        let opts = CleanOptions {
            max_chars: Some(5),
            ..CleanOptions::default()
        };
        let cleaned = clean_corpus("abcdefghij", &opts);
        assert_eq!(cleaned.chars().count(), 5);
    }

    #[test]
    fn stats_count_correctly() {
        let s = corpus_stats("ab ab\ncd");
        assert_eq!(s.chars, 8);
        assert_eq!(s.lines, 2);
        assert_eq!(s.words, 3);
        // distinct: a b space \n c d = 6
        assert_eq!(s.unique_chars, 6);
        assert_eq!(s.bytes, 8);
    }

    #[test]
    fn validate_rejects_empty_and_short() {
        assert!(validate_for_training("", 4).is_err());
        assert!(validate_for_training("abc", 4).is_err()); // 3 < 5
        assert!(validate_for_training("abcde", 4).is_ok()); // 5 >= 5
        assert!(validate_for_training("abcde", 0).is_err()); // bad context
    }

    #[test]
    fn cleaning_then_validation_is_consistent() {
        let raw = "café résumé naïve façade — a short but valid corpus.\r\n";
        let cleaned = clean_corpus(raw, &CleanOptions::default());
        assert!(validate_for_training(&cleaned, 8).is_ok());
        // Non-ASCII survives (byte-level BPE handles it).
        assert!(cleaned.contains("café"));
    }

    #[test]
    fn strip_invisible_removes_bom_zero_width_and_controls() {
        let raw = "\u{FEFF}he\u{200B}llo\r\n\u{2060}wor\u{0007}ld\tok\u{200D}";
        assert_eq!(strip_invisible(raw), "hello\nworld\tok");
    }

    #[test]
    fn gutenberg_bodies_keeps_every_book() {
        let raw = "hdr1\n*** START OF BOOK ONE ***\nalpha\n*** END OF BOOK ONE ***\nlicense\n\
                   hdr2\n*** START OF BOOK TWO ***\nbeta\n*** END OF BOOK TWO ***\nfooter";
        let b = gutenberg_bodies(raw);
        assert_eq!(b.len(), 2);
        assert_eq!(b[0].trim(), "alpha");
        assert_eq!(b[1].trim(), "beta");
        assert_eq!(gutenberg_bodies("no markers"), vec!["no markers"]);
        let cleaned = clean_corpus(raw, &CleanOptions::default());
        assert!(cleaned.contains("alpha") && cleaned.contains("beta"));
        assert!(!cleaned.contains("license") && !cleaned.contains("hdr2"));
    }

    #[test]
    fn squeeze_spaces_and_nfkc() {
        assert_eq!(squeeze_spaces("a  b\t\t c\n  d"), "a b c\n d");
        assert_eq!(normalize_nfkc("ﬁ ① Ａ"), "fi 1 A");
    }

    #[test]
    fn clean_corpus_strips_bom_by_default() {
        let out = clean_corpus("\u{FEFF}Hi\u{200B} there", &CleanOptions::default());
        assert_eq!(out.trim_end(), "Hi there");
    }
    #[test]
    fn nbsp_and_bullet_fold_and_folding_can_be_disabled() {
        let raw = "a\u{00A0}b\u{2007}c\u{202F}d \u{2022} e\u{2014}f";
        let on = clean_corpus(raw, &CleanOptions::default());
        assert_eq!(on.trim_end(), "a b c d * e-f");
        let off = clean_corpus(
            raw,
            &CleanOptions {
                normalize_punctuation: false,
                ..CleanOptions::default()
            },
        );
        assert!(off.contains('\u{2022}') && off.contains('\u{2014}'));
    }

    #[test]
    fn gutenberg_marker_forms_and_truncated_books() {
        // no-space marker form, and the earlier of two forms wins
        let a = "x\n***START OF A***\nbody1\n*** END OF A ***\ny";
        assert_eq!(gutenberg_bodies(a), vec!["body1\n"]);
        // both forms of the START marker present: the earlier one wins
        let e = "***START OF A***\nb1\n*** END OF A ***\n*** START OF B ***\nb2\n***END OF B***";
        assert_eq!(gutenberg_bodies(e), vec!["b1\n", "b2\n"]);
        // START with no newline after it, or no END: whole text passes through
        let b = "x\n*** START OF A ***";
        assert_eq!(gutenberg_bodies(b), vec![b]);
        let c = "*** START OF A ***\nbody with no end";
        assert_eq!(gutenberg_bodies(c), vec![c]);
        // a complete book followed by a truncated one keeps only the complete one
        let d = "*** START OF A ***\nb1\n*** END OF A ***\n*** START OF B ***\nb2";
        assert_eq!(gutenberg_bodies(d), vec!["b1\n"]);
    }
}
