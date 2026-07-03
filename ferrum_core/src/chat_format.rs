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
        assert_eq!(
            flat(&render(ChatFormat::Llama2Inst, "hi")),
            "[INST] hi [/INST]"
        );
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
