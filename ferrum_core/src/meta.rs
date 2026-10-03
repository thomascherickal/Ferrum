//! Model metadata, task type, and input normalizer — the descriptive half of
//! every FINF model file.
//!
//! `ModelMetadata` embeds everything a UI needs to build itself dynamically:
//! feature names, ranges, class names, task type, and (for BPE models) the
//! tokenizer state. It serialises to compact JSON and is stored inside the FINF
//! model file, so a model never needs a separate config file.

use crate::error::{InferError, Result};
use crate::tensor::Tensor;
use crate::verbose;

// ─────────────────────────────────────────────────────────────────────────────
// Task type
// ─────────────────────────────────────────────────────────────────────────────

/// Whether the network predicts a class label, a continuous value, or is a generative small language model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TaskType {
    Classification,
    Regression,
    TransformerSLM,
}

impl TaskType {
    pub fn as_str(&self) -> &'static str {
        match self {
            TaskType::Classification => "classification",
            TaskType::Regression => "regression",
            TaskType::TransformerSLM => "transformer_slm",
        }
    }
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "classification" => Some(Self::Classification),
            "regression" => Some(Self::Regression),
            "transformer_slm" => Some(Self::TransformerSLM),
            _ => None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Model metadata  (serialised to JSON, embedded in the FINF file)
// ─────────────────────────────────────────────────────────────────────────────

/// Everything the browser UI needs to build itself for an arbitrary dataset.
#[derive(Clone, Debug)]
pub struct ModelMetadata {
    pub dataset_name: String,
    pub task: TaskType,
    pub feature_names: Vec<String>,
    /// Per-feature [min, max] in the *raw* (un-normalised) dataset.
    pub feature_ranges: Vec<[f32; 2]>,
    /// Class names in label order (empty for regression).
    pub class_names: Vec<String>,
    /// For regression: the target variable name.
    pub target_name: String,
    /// For regression: [min, max] of raw target values.
    pub target_range: [f32; 2],
    pub input_dim: usize,
    pub output_dim: usize,
    /// Serialized BPE tokenizer state (the merge list from
    /// [`ByteBpeTokenizer::encode_state`](crate::ByteBpeTokenizer::encode_state)).
    /// Empty when the model uses character-level tokenization, in which case
    /// `class_names` carries the per-character vocabulary instead. When set,
    /// generation encodes and decodes text through the tokenizer rather than
    /// through `class_names`.
    pub tokenizer_state: String,
}

impl Default for ModelMetadata {
    fn default() -> Self {
        Self {
            dataset_name: String::new(),
            task: TaskType::Classification,
            feature_names: Vec::new(),
            feature_ranges: Vec::new(),
            class_names: Vec::new(),
            target_name: String::new(),
            target_range: [0.0, 1.0],
            input_dim: 0,
            output_dim: 1,
            tokenizer_state: String::new(),
        }
    }
}

impl ModelMetadata {
    /// Serialise to a compact JSON string (no external deps).
    pub fn to_json(&self) -> String {
        let feat_names = self
            .feature_names
            .iter()
            .map(|n| format!("\"{}\"", n.replace('"', "\\\"")))
            .collect::<Vec<_>>()
            .join(",");
        let feat_ranges = self
            .feature_ranges
            .iter()
            .map(|[lo, hi]| format!("[{lo:.6},{hi:.6}]"))
            .collect::<Vec<_>>()
            .join(",");
        let class_names = self
            .class_names
            .iter()
            .map(|n| format!("\"{}\"", n.replace('"', "\\\"")))
            .collect::<Vec<_>>()
            .join(",");
        let tok_state = self
            .tokenizer_state
            .replace('\\', "\\\\")
            .replace('"', "\\\"");
        format!(
            concat!(
                r#"{{"dataset_name":"{dn}","task":"{task}","feature_names":[{fn_}],"#,
                r#""feature_ranges":[{fr}],"class_names":[{cn}],"#,
                r#""target_name":"{tn}","target_range":[{tlo:.6},{thi:.6}],"#,
                r#""input_dim":{id},"output_dim":{od},"tokenizer_state":"{ts}"}}"#
            ),
            dn = self.dataset_name.replace('"', "\\\""),
            task = self.task.as_str(),
            fn_ = feat_names,
            fr = feat_ranges,
            cn = class_names,
            tn = self.target_name.replace('"', "\\\""),
            tlo = self.target_range[0],
            thi = self.target_range[1],
            id = self.input_dim,
            od = self.output_dim,
            ts = tok_state,
        )
    }

    /// Parse the JSON produced by `to_json`.
    pub fn from_json(s: &str) -> Result<Self> {
        fn extract<'a>(s: &'a str, key: &str) -> Option<&'a str> {
            let needle = format!("\"{}\":", key);
            let start = s.find(&needle)? + needle.len();
            Some(s[start..].trim_start())
        }
        fn str_val(s: &str) -> Option<String> {
            let s = s.trim_start_matches('"');
            let end = s.find('"')?;
            Some(s[..end].to_string())
        }
        fn usize_val(s: &str) -> Option<usize> {
            s.split([',', '}']).next()?.trim().parse().ok()
        }
        fn _f32_val(s: &str) -> Option<f32> {
            s.split([',', ']', '}']).next()?.trim().parse().ok()
        }

        fn str_arr(s: &str) -> Vec<String> {
            let inner = s.trim_start_matches('[');
            let end = inner.find(']').unwrap_or(inner.len());
            inner[..end]
                .split(',')
                .map(|t| t.trim().trim_matches('"').to_string())
                .filter(|t| !t.is_empty())
                .collect()
        }
        fn f32_pair(s: &str) -> [f32; 2] {
            let inner = s.trim_start_matches('[');
            let end = inner.find(']').unwrap_or(inner.len());
            let parts: Vec<f32> = inner[..end]
                .split(',')
                .filter_map(|t| t.trim().parse().ok())
                .collect();
            [
                parts.first().copied().unwrap_or(0.0),
                parts.get(1).copied().unwrap_or(1.0),
            ]
        }
        fn f32_pairs(s: &str) -> Vec<[f32; 2]> {
            // Parse an outer array of pairs `[[a,b],[c,d],...]` that may be
            // followed by more JSON. Scan from the first `[` (the outer bracket)
            // and stop at its matching close bracket, collecting each inner
            // `[...]` pair. Robust to whitespace and to later arrays in the doc.
            let bytes = s.as_bytes();
            let Some(outer_start) = bytes.iter().position(|&b| b == b'[') else {
                return Vec::new();
            };
            let mut out = Vec::new();
            let mut depth = 0i32;
            let mut pair_start = 0usize;
            for (i, &b) in bytes.iter().enumerate().skip(outer_start) {
                match b {
                    b'[' => {
                        depth += 1;
                        if depth == 2 {
                            pair_start = i; // start of an inner pair
                        }
                    }
                    b']' => {
                        if depth == 2 {
                            out.push(f32_pair(&s[pair_start..=i])); // end of pair
                        }
                        depth -= 1;
                        if depth == 0 {
                            break; // outer array closed
                        }
                    }
                    _ => {}
                }
            }
            out
        }

        let task_str = extract(s, "task")
            .and_then(str_val)
            .ok_or_else(|| InferError::Format("missing task".into()))?;
        let task = TaskType::from_str(&task_str)
            .ok_or_else(|| InferError::Format(format!("unknown task: {task_str}")))?;

        let dataset_name = extract(s, "dataset_name")
            .and_then(str_val)
            .unwrap_or_default();
        let target_name = extract(s, "target_name")
            .and_then(str_val)
            .unwrap_or_default();
        let feat_names = extract(s, "feature_names").map(str_arr).unwrap_or_default();
        let feat_ranges = extract(s, "feature_ranges")
            .map(f32_pairs)
            .unwrap_or_default();
        let class_names = extract(s, "class_names").map(str_arr).unwrap_or_default();
        let target_range = extract(s, "target_range")
            .map(f32_pair)
            .unwrap_or([0.0, 1.0]);
        let input_dim = extract(s, "input_dim").and_then(usize_val).unwrap_or(0);
        let output_dim = extract(s, "output_dim").and_then(usize_val).unwrap_or(1);
        // Absent in pre-BPE models → empty (character-level tokenization).
        let tokenizer_state = extract(s, "tokenizer_state")
            .and_then(str_val)
            .unwrap_or_default();

        Ok(Self {
            dataset_name,
            task,
            feature_names: feat_names,
            feature_ranges: feat_ranges,
            class_names,
            target_name,
            target_range,
            input_dim,
            output_dim,
            tokenizer_state,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Normalizer
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct Normalizer {
    pub means: Vec<f32>,
    pub stds: Vec<f32>,
}

impl Normalizer {
    pub fn fit(x: &Tensor) -> Result<Self> {
        let (rows, cols) = x.matrix_dims()?;
        vprintln!(
            "[csv::Normalizer::fit] Fitting on [{},{}] matrix",
            rows,
            cols
        );
        if rows == 0 {
            return Err(InferError::Format("cannot fit on 0 rows".into()));
        }
        let n = rows as f32;
        let mut means = vec![0.0f32; cols];
        for r in 0..rows {
            #[allow(clippy::needless_range_loop)]
            for c in 0..cols {
                means[c] += x.at(r, c);
            }
        }
        for m in &mut means {
            *m /= n;
        }
        let mut stds = vec![0.0f32; cols];
        for r in 0..rows {
            for c in 0..cols {
                stds[c] += (x.at(r, c) - means[c]).powi(2);
            }
        }
        for s in &mut stds {
            *s = (*s / n).sqrt();
            if *s < 1e-8 {
                *s = 1.0;
            }
        }
        vprintln!("[csv::Normalizer::fit] Computed {} column stats", cols);
        if verbose::is_verbose() {
            for c in 0..cols.min(8) {
                vprintln!(
                    "[csv::Normalizer::fit]   col[{}]: mean={:.6}, std={:.6}",
                    c,
                    means[c],
                    stds[c]
                );
            }
            if cols > 8 {
                vprintln!("[csv::Normalizer::fit]   ... ({} more columns)", cols - 8);
            }
        }
        Ok(Self { means, stds })
    }

    pub fn transform(&self, x: &Tensor) -> Result<Tensor> {
        let (rows, cols) = x.matrix_dims()?;
        vprintln!("[csv::Normalizer::transform] [{},{}]", rows, cols);
        // Allow normalizer to have one extra column for the target (regression).
        if cols != self.means.len() && cols != self.means.len().saturating_sub(1) {
            return Err(InferError::DimMismatch(format!(
                "normalizer has {} cols, input has {cols}",
                self.means.len()
            )));
        }
        let mut out = x.data.clone();
        for r in 0..rows {
            for c in 0..cols {
                out[r * cols + c] = (x.at(r, c) - self.means[c]) / self.stds[c];
            }
        }
        Tensor::matrix(rows, cols, out)
    }

    pub fn transform_row(&self, features: &[f32]) -> Result<Tensor> {
        self.transform(&Tensor::row(features.to_vec())?)
    }

    /// Normalise a single target value for regression.
    pub fn normalise_target(&self, v: f32) -> f32 {
        let c = self.means.len() - 1; // last entry is target stats
        (v - self.means[c]) / self.stds[c]
    }

    /// Denormalise a predicted value back to the original scale.
    pub fn denormalise_target(&self, v: f32) -> f32 {
        let c = self.means.len() - 1;
        v * self.stds[c] + self.means[c]
    }

    pub fn encode(&self) -> String {
        self.means
            .iter()
            .zip(&self.stds)
            .map(|(m, s)| format!("{m:.8},{s:.8}"))
            .collect::<Vec<_>>()
            .join(";")
    }
    pub fn decode(s: &str) -> Result<Self> {
        let mut means = Vec::new();
        let mut stds = Vec::new();
        for token in s.split(';') {
            let p: Vec<&str> = token.split(',').collect();
            if p.len() != 2 {
                return Err(InferError::Format(format!("bad token: {token}")));
            }
            means.push(
                p[0].parse::<f32>()
                    .map_err(|e| InferError::ParseError(e.to_string()))?,
            );
            stds.push(
                p[1].parse::<f32>()
                    .map_err(|e| InferError::ParseError(e.to_string()))?,
            );
        }
        Ok(Self { means, stds })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tokenizer_state_survives_json_roundtrip() {
        // BPE merge lists ("a,b;c,d;…") must round-trip through the metadata
        // JSON unchanged; a model with no tokenizer keeps an empty string.
        let mut meta = ModelMetadata {
            tokenizer_state: "256,257;258,32;100,200".into(),
            output_dim: 300,
            ..Default::default()
        };
        let back = ModelMetadata::from_json(&meta.to_json()).unwrap();
        assert_eq!(back.tokenizer_state, meta.tokenizer_state);
        assert_eq!(back.output_dim, 300);

        meta.tokenizer_state = String::new();
        let back2 = ModelMetadata::from_json(&meta.to_json()).unwrap();
        assert!(back2.tokenizer_state.is_empty());
    }

    #[test]
    fn pre_bpe_metadata_json_defaults_tokenizer_state_empty() {
        // Older models serialized before the tokenizer_state field must still
        // parse, defaulting to character-level (empty tokenizer state).
        let json = r#"{"dataset_name":"old","task":"transformer_slm","feature_names":[],"feature_ranges":[],"class_names":["61","62"],"target_name":"next_char","target_range":[0.000000,2.000000],"input_dim":4,"output_dim":2}"#;
        let meta = ModelMetadata::from_json(json).unwrap();
        assert!(meta.tokenizer_state.is_empty());
        assert_eq!(meta.output_dim, 2);
        assert_eq!(meta.class_names, vec!["61", "62"]);
    }
}
