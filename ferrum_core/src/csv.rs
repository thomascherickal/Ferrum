//! Numeric CSV loader, normalizer, task detection, and model metadata.
//!
//! Three steps turn a CSV into network inputs:
//!   1. `CsvDataset::from_str`  — parse raw text; auto-detect header & task type.
//!   2. `Normalizer::fit`       — per-column mean/std from training split only.
//!   3. `Normalizer::transform` — apply to any split (train, val, live inference).
//!
//! `ModelMetadata` embeds everything the UI needs to build itself dynamically:
//! feature names, ranges, class names, task type. It serialises to compact JSON
//! and is stored inside the FINF model file so the browser never needs a
//! separate config request.

use crate::error::{InferError, Result};
use crate::tensor::Tensor;

// ─────────────────────────────────────────────────────────────────────────────
// Task type
// ─────────────────────────────────────────────────────────────────────────────

/// Whether the network predicts a class label or a continuous value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TaskType {
    Classification,
    Regression,
}

impl TaskType {
    pub fn as_str(&self) -> &'static str {
        match self {
            TaskType::Classification => "classification",
            TaskType::Regression => "regression",
        }
    }
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "classification" => Some(Self::Classification),
            "regression" => Some(Self::Regression),
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
        format!(
            concat!(
                r#"{{"dataset_name":"{dn}","task":"{task}","feature_names":[{fn_}],"#,
                r#""feature_ranges":[{fr}],"class_names":[{cn}],"#,
                r#""target_name":"{tn}","target_range":[{tlo:.6},{thi:.6}],"#,
                r#""input_dim":{id},"output_dim":{od}}}"#
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
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CSV parsing
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq)]
pub struct CsvRow {
    pub features: Vec<f32>,
    pub label: usize, // class index (classification) or 0 (regression)
    pub target: f32,  // raw target value (regression) or label as f32
}

pub struct CsvDataset {
    pub rows: Vec<CsvRow>,
    pub num_features: usize,
    pub num_classes: usize,
    pub class_names: Vec<String>,
    pub feature_names: Vec<String>,
    pub task: TaskType,
    /// Per-feature [min, max] computed during parsing.
    pub feature_ranges: Vec<[f32; 2]>,
    pub target_range: [f32; 2],
}

impl CsvDataset {
    /// Parse a CSV. Auto-detects:
    ///  - header row (if last column is non-numeric → skip as header)
    ///  - task type: if the target column has >20 distinct numeric values → regression
    ///
    /// For regression the target column is a raw f32; for classification it is
    /// a string (or integer) class label.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(text: &str) -> Result<Self> {
        Self::from_str_with_task(text, None)
    }

    /// Like `from_str` but lets the caller override task detection.
    pub fn from_str_with_task(text: &str, force_task: Option<TaskType>) -> Result<Self> {
        let mut lines = text.lines().peekable();
        while lines.peek().map(|l| l.trim().is_empty()) == Some(true) {
            lines.next();
        }

        let first = lines
            .peek()
            .ok_or_else(|| InferError::Format("empty CSV".into()))?;
        let cols: Vec<&str> = first.split(',').map(str::trim).collect();
        let has_header = cols
            .last()
            .map(|c| c.parse::<f64>().is_err())
            .unwrap_or(false);

        let mut feature_names: Vec<String> = Vec::new();
        if has_header {
            let header: Vec<&str> = lines.next().unwrap().split(',').map(str::trim).collect();
            let n = header.len();
            feature_names = header[..n - 1]
                .iter()
                .map(|s| s.trim_matches('"').to_string())
                .collect();
            // target column name is informational only; stored in ModelMetadata by the caller
        }

        // First pass: collect all rows as raw strings
        let mut raw_rows: Vec<(Vec<f32>, String)> = Vec::new();
        let mut num_features: Option<usize> = None;

        for (ln, line) in lines.enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let fields: Vec<&str> = line.split(',').map(str::trim).collect();
            if fields.len() < 2 {
                continue;
            }
            let nf = fields.len() - 1;
            if let Some(expected) = num_features {
                if nf != expected {
                    return Err(InferError::Format(format!(
                        "line {ln}: {nf} features, expected {expected}"
                    )));
                }
            } else {
                num_features = Some(nf);
                if feature_names.is_empty() {
                    feature_names = (0..nf).map(|i| format!("feature_{i}")).collect();
                }
            }
            let features: Vec<f32> = fields[..nf]
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    s.parse::<f32>()
                        .map_err(|_| InferError::ParseError(format!("line {ln}, col {i}: {:?}", s)))
                })
                .collect::<Result<_>>()?;
            raw_rows.push((
                features,
                fields.last().unwrap().trim_matches('"').to_string(),
            ));
        }

        let num_features = num_features.ok_or_else(|| InferError::Format("no data rows".into()))?;

        // Determine task type
        let distinct_targets: std::collections::HashSet<String> =
            raw_rows.iter().map(|(_, t)| t.clone()).collect();
        let all_numeric = distinct_targets.iter().all(|t| t.parse::<f64>().is_ok());
        let task = force_task.unwrap_or_else(|| {
            let reg_threshold = if raw_rows.len() > 50 {
                15
            } else {
                raw_rows.len() / 3
            };
            if all_numeric && distinct_targets.len() > reg_threshold {
                TaskType::Regression
            } else {
                TaskType::Classification
            }
        });

        // Build feature ranges
        let mut feat_min = vec![f32::MAX; num_features];
        let mut feat_max = vec![f32::MIN; num_features];
        for (feats, _) in &raw_rows {
            for (i, &v) in feats.iter().enumerate() {
                if v < feat_min[i] {
                    feat_min[i] = v;
                }
                if v > feat_max[i] {
                    feat_max[i] = v;
                }
            }
        }
        let feature_ranges: Vec<[f32; 2]> = feat_min
            .iter()
            .zip(&feat_max)
            .map(|(&lo, &hi)| [lo, hi])
            .collect();

        // Build rows
        let mut class_index: std::collections::HashMap<String, usize> = Default::default();
        let mut class_names: Vec<String> = Vec::new();
        let mut rows: Vec<CsvRow> = Vec::new();
        let mut tgt_min = f32::MAX;
        let mut tgt_max = f32::MIN;

        for (features, label_str) in raw_rows {
            let (label, target) = match task {
                TaskType::Regression => {
                    let v: f32 = label_str.parse().map_err(|_| {
                        InferError::ParseError(format!("regression target: {label_str}"))
                    })?;
                    if v < tgt_min {
                        tgt_min = v;
                    }
                    if v > tgt_max {
                        tgt_max = v;
                    }
                    (0, v)
                }
                TaskType::Classification => {
                    let idx = *class_index.entry(label_str.clone()).or_insert_with(|| {
                        let i = class_names.len();
                        class_names.push(label_str);
                        i
                    });
                    (idx, idx as f32)
                }
            };
            rows.push(CsvRow {
                features,
                label,
                target,
            });
        }

        let (num_classes, target_range) = match task {
            TaskType::Classification => (class_names.len(), [0.0, (class_names.len() - 1) as f32]),
            TaskType::Regression => (1, [tgt_min, tgt_max]),
        };

        Ok(Self {
            rows,
            num_features,
            num_classes,
            class_names,
            feature_names,
            task,
            feature_ranges,
            target_range,
        })
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Build input matrix + target vector. For regression, targets are raw f32 values.
    pub fn to_tensors(&self) -> Result<(Tensor, Vec<usize>, Vec<f32>)> {
        let n = self.rows.len();
        let dim = self.num_features;
        let mut data = Vec::with_capacity(n * dim);
        let mut labels = Vec::with_capacity(n);
        let mut targets = Vec::with_capacity(n);
        for row in &self.rows {
            data.extend_from_slice(&row.features);
            labels.push(row.label);
            targets.push(row.target);
        }
        Ok((Tensor::matrix(n, dim, data)?, labels, targets))
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
        Ok(Self { means, stds })
    }

    pub fn transform(&self, x: &Tensor) -> Result<Tensor> {
        let (rows, cols) = x.matrix_dims()?;
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

/// Fit a Normalizer that also covers the target column (for regression).
/// Feature columns first, then one more pair for the target.
pub fn fit_normalizer_with_target(x: &Tensor, targets: &[f32]) -> Result<Normalizer> {
    let mut norm = Normalizer::fit(x)?;
    let n = targets.len() as f32;
    let mean = targets.iter().sum::<f32>() / n;
    let std = (targets.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n)
        .sqrt()
        .max(1e-8);
    norm.means.push(mean);
    norm.stds.push(std);
    Ok(norm)
}

// ─────────────────────────────────────────────────────────────────────────────
// Train / val split
// ─────────────────────────────────────────────────────────────────────────────

pub fn train_val_split(
    ds: &CsvDataset,
    val_fraction: f32,
    rng: &mut crate::rng::Rng,
) -> (CsvDataset, CsvDataset) {
    let n = ds.rows.len();
    let mut idx: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = (rng.next_u64() as usize) % (i + 1);
        idx.swap(i, j);
    }
    let val_n = ((n as f32 * val_fraction).round() as usize).max(1);
    let (val_idx, train_idx) = idx.split_at(val_n);
    let make = |indices: &[usize]| CsvDataset {
        rows: indices.iter().map(|&i| ds.rows[i].clone()).collect(),
        num_features: ds.num_features,
        num_classes: ds.num_classes,
        class_names: ds.class_names.clone(),
        feature_names: ds.feature_names.clone(),
        task: ds.task,
        feature_ranges: ds.feature_ranges.clone(),
        target_range: ds.target_range,
    };
    (make(train_idx), make(val_idx))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    const SNIPPET: &str = "\
sepal_length,sepal_width,petal_length,petal_width,species
5.1,3.5,1.4,0.2,Iris-setosa
4.9,3.0,1.4,0.2,Iris-setosa
6.3,3.3,6.0,2.5,Iris-virginica
5.8,2.7,5.1,1.9,Iris-virginica
7.0,3.2,4.7,1.4,Iris-versicolor
6.4,3.2,4.5,1.5,Iris-versicolor
";

    const REGRESSION: &str = "\
x1,x2,price
1.0,2.0,100000.0
2.0,3.0,200000.0
3.0,4.0,300000.0
4.0,5.0,400000.0
5.0,6.0,500000.0
6.0,7.0,600000.0
";

    #[test]
    fn classification_parses_header_and_rows() {
        let ds = CsvDataset::from_str(SNIPPET).unwrap();
        assert_eq!(ds.rows.len(), 6);
        assert_eq!(ds.num_features, 4);
        assert_eq!(ds.num_classes, 3);
        assert_eq!(ds.task, TaskType::Classification);
    }

    #[test]
    fn feature_names_read_from_header() {
        let ds = CsvDataset::from_str(SNIPPET).unwrap();
        assert_eq!(ds.feature_names[0], "sepal_length");
        assert_eq!(ds.feature_names[3], "petal_width");
    }

    #[test]
    fn regression_detected_automatically() {
        let ds = CsvDataset::from_str(REGRESSION).unwrap();
        assert_eq!(ds.task, TaskType::Regression);
        assert_eq!(ds.num_features, 2);
    }

    #[test]
    fn regression_targets_correct() {
        let ds = CsvDataset::from_str(REGRESSION).unwrap();
        assert_eq!(ds.rows[0].target, 100000.0);
        assert_eq!(ds.rows[5].target, 600000.0);
    }

    #[test]
    fn feature_ranges_computed() {
        let ds = CsvDataset::from_str(SNIPPET).unwrap();
        // sepal_length ranges from 4.9 to 7.0
        assert!((ds.feature_ranges[0][0] - 4.9).abs() < 0.01);
        assert!((ds.feature_ranges[0][1] - 7.0).abs() < 0.01);
    }

    #[test]
    fn to_tensors_shape() {
        let ds = CsvDataset::from_str(SNIPPET).unwrap();
        let (x, labels, _) = ds.to_tensors().unwrap();
        assert_eq!(x.shape, vec![6, 4]);
        assert_eq!(labels.len(), 6);
    }

    #[test]
    fn normalizer_zero_mean() {
        let ds = CsvDataset::from_str(SNIPPET).unwrap();
        let (x, _, _) = ds.to_tensors().unwrap();
        let norm = Normalizer::fit(&x).unwrap();
        let z = norm.transform(&x).unwrap();
        let (n, cols) = z.matrix_dims().unwrap();
        for c in 0..cols {
            let mean: f32 = (0..n).map(|r| z.at(r, c)).sum::<f32>() / n as f32;
            assert!(mean.abs() < 1e-4, "col {c} mean = {mean}");
        }
    }

    #[test]
    fn normalizer_encode_decode_roundtrip() {
        let ds = CsvDataset::from_str(SNIPPET).unwrap();
        let (x, _, _) = ds.to_tensors().unwrap();
        let n1 = Normalizer::fit(&x).unwrap();
        let n2 = Normalizer::decode(&n1.encode()).unwrap();
        for (a, b) in n1.means.iter().zip(&n2.means) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn metadata_json_roundtrip() {
        let ds = CsvDataset::from_str(SNIPPET).unwrap();
        let meta = ModelMetadata {
            dataset_name: "test".into(),
            task: TaskType::Classification,
            feature_names: ds.feature_names.clone(),
            feature_ranges: ds.feature_ranges.clone(),
            class_names: ds.class_names.clone(),
            target_name: "".into(),
            target_range: [0.0, 1.0],
            input_dim: 4,
            output_dim: 3,
        };
        let json = meta.to_json();
        let meta2 = ModelMetadata::from_json(&json).unwrap();
        assert_eq!(meta2.task, TaskType::Classification);
        assert_eq!(meta2.feature_names, ds.feature_names);
        assert_eq!(meta2.class_names, ds.class_names);
        assert_eq!(meta2.input_dim, 4);
        // Regression guard: feature_ranges must survive the JSON round-trip
        // (a prior parser bug silently dropped them, producing an empty Vec
        // that broke the browser's slider builder).
        assert_eq!(
            meta2.feature_ranges.len(),
            ds.feature_ranges.len(),
            "feature_ranges count must survive round-trip"
        );
        assert_eq!(meta2.feature_ranges.len(), 4);
        for (a, b) in meta2.feature_ranges.iter().zip(&ds.feature_ranges) {
            assert!((a[0] - b[0]).abs() < 1e-3, "range lo mismatch");
            assert!((a[1] - b[1]).abs() < 1e-3, "range hi mismatch");
        }
    }

    #[test]
    fn regression_metadata_roundtrip() {
        let ds = CsvDataset::from_str(REGRESSION).unwrap();
        let meta = ModelMetadata {
            dataset_name: "reg".into(),
            task: TaskType::Regression,
            feature_names: ds.feature_names.clone(),
            feature_ranges: ds.feature_ranges.clone(),
            class_names: vec![],
            target_name: "price".into(),
            target_range: ds.target_range,
            input_dim: 2,
            output_dim: 1,
        };
        let json = meta.to_json();
        let meta2 = ModelMetadata::from_json(&json).unwrap();
        assert_eq!(meta2.task, TaskType::Regression);
        assert_eq!(meta2.target_name, "price");
        assert!((meta2.target_range[0] - 100000.0).abs() < 1.0);
    }

    #[test]
    fn train_val_split_preserves_total() {
        let ds = CsvDataset::from_str(SNIPPET).unwrap();
        let mut rng = crate::rng::Rng::new(42);
        let (tr, va) = train_val_split(&ds, 0.33, &mut rng);
        assert_eq!(tr.rows.len() + va.rows.len(), 6);
    }

    #[test]
    fn constant_column_doesnt_divide_by_zero() {
        let csv = "a,b,label\n1.0,0.0,x\n2.0,0.0,y\n3.0,0.0,x\n";
        let ds = CsvDataset::from_str(csv).unwrap();
        let (x, _, _) = ds.to_tensors().unwrap();
        let norm = Normalizer::fit(&x).unwrap();
        assert_eq!(norm.stds[1], 1.0);
    }

    #[test]
    fn fit_with_target_normalizer() {
        let ds = CsvDataset::from_str(REGRESSION).unwrap();
        let (x, _, targets) = ds.to_tensors().unwrap();
        let norm = fit_normalizer_with_target(&x, &targets).unwrap();
        // Should have 3 pairs: 2 features + 1 target
        assert_eq!(norm.means.len(), 3);
        let back = norm.denormalise_target(norm.normalise_target(300000.0));
        assert!((back - 300000.0).abs() < 1.0);
    }
}
