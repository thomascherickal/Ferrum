//! Corpus acquisition helpers: cleaning raw text into a shape the SLM training
//! paths expect, and summarising it.
//!
//! Downloading is intentionally **not** here — `ferrum_core` is zero-dependency
//! and `std`-only, so it has no HTTP client. An embedding application (the GUI,
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
    let body = if opts.strip_gutenberg {
        strip_gutenberg_boilerplate(raw)
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
        if opts.strip_control_chars && ch.is_control() && ch != '\n' && ch != '\t' {
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

/// Keep only the body of a Project Gutenberg text, dropping the license header
/// and footer. If the markers are not both present, the input is returned
/// unchanged.
fn strip_gutenberg_boilerplate(raw: &str) -> &str {
    let start = raw
        .find("*** START OF")
        .or_else(|| raw.find("***START OF"))
        .and_then(|i| raw[i..].find('\n').map(|j| i + j + 1));
    let end = raw.find("*** END OF").or_else(|| raw.find("***END OF"));
    match (start, end) {
        (Some(s), Some(e)) if s < e => &raw[s..e],
        _ => raw,
    }
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
}
