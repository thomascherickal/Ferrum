//! Generic offline Edge Generative SLM Library Engine.
use crate::error::{InferError, Result};
use crate::csv::{CsvDataset, ModelMetadata, Normalizer, TaskType};
use crate::rng::Rng;
use crate::optim::Sgd;
use crate::train::{Net, train_epoch};
use crate::model::Sequential;
use crate::loader::{to_bytes, from_bytes};
use crate::tensor::Tensor;
use std::collections::HashSet;

/// Unified Causal Small Language Model (SLM) for off-grid edge generative AI.
pub struct GenerativeSLM {
    pub model: Sequential,
    pub norm: Normalizer,
    pub meta: ModelMetadata,
}

impl GenerativeSLM {
    /// Construct a new instance manually from sequential layers, normalizer, and metadata.
    pub fn new(model: Sequential, norm: Normalizer, meta: ModelMetadata) -> Self {
        Self { model, norm, meta }
    }

    /// Train a hand-crafted edge Generative SLM (MLP Causal model) on any customized raw text corpus.
    pub fn train(
        corpus: &str,
        context_len: usize,
        hidden_size: usize,
        epochs: usize,
        lr: f32,
        momentum: f32,
        batch_size: usize,
        rng: &mut Rng,
    ) -> Result<Self> {
        Self::train_with_callback(
            corpus,
            context_len,
            hidden_size,
            epochs,
            lr,
            momentum,
            batch_size,
            rng,
            |_, _| {},
        )
    }

    /// Train a hand-crafted edge Generative SLM with a callback invoked at each epoch with loss.
    pub fn train_with_callback<F>(
        corpus: &str,
        context_len: usize,
        hidden_size: usize,
        epochs: usize,
        lr: f32,
        momentum: f32,
        batch_size: usize,
        rng: &mut Rng,
        mut progress_callback: F,
    ) -> Result<Self>
    where
        F: FnMut(usize, f32),
    {
        let csv_data = build_csv_dataset(corpus, context_len)?;
        let ds = CsvDataset::from_str(&csv_data)?;

        let (x_raw, y_cls, _) = ds.to_tensors()?;
        let mut norm = Normalizer::fit(&x_raw)?;
        for m in &mut norm.means { *m = 0.0; }
        for s in &mut norm.stds { *s = 1.0; }
        let x_train = norm.transform(&x_raw)?;

        let mut net = Net::mlp(context_len, hidden_size, ds.num_classes, rng);
        let opt = Sgd::with_momentum(lr, momentum);

        for ep in 1..=epochs {
            let loss = train_epoch(&mut net, &x_train, &y_cls, batch_size, &opt, rng)?;
            progress_callback(ep, loss);
        }

        let model = net.to_inference_task(TaskType::Classification)?;
        let meta = ModelMetadata {
            dataset_name: "GenerativeSLM Model".into(),
            task: TaskType::Classification,
            feature_names: ds.feature_names.clone(),
            feature_ranges: ds.feature_ranges.clone(),
            class_names: ds.class_names.clone(),
            target_name: "next_char".into(),
            target_range: ds.target_range,
            input_dim: context_len,
            output_dim: ds.num_classes,
        };

        Ok(Self { model, norm, meta })
    }

    /// Serialize the trained Generative SLM model to self-contained FINF v4 bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        to_bytes(&self.model, &self.norm, &self.meta)
    }

    /// Load a trained Generative SLM model from self-contained FINF v4 bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let (model, norm, meta) = from_bytes(bytes)?;
        Ok(Self { model, norm, meta })
    }

    /// Autoregressively generate next-character sequence completions from a seed text.
    pub fn generate(&self, seed: &str, num_chars: usize, temp: f32, rng: &mut Rng) -> Result<String> {
        let mut generated = seed.to_string();
        let context_len = self.meta.input_dim;

        for _ in 0..num_chars {
            let current_len = generated.chars().count();
            if current_len < context_len {
                break;
            }
            let context_chars: Vec<char> = generated.chars().skip(current_len - context_len).collect();

            // Convert to indices using class_names (vocabulary)
            let mut input_data = Vec::with_capacity(context_len);
            for &ch in &context_chars {
                let hex = char_to_hex(ch);
                let idx = self.meta
                    .class_names
                    .iter()
                    .position(|s| s == &hex)
                    .unwrap_or(0);
                input_data.push(idx as f32);
            }

            let input_tensor = Tensor::row(input_data)?;
            let transformed_input = self.norm.transform(&input_tensor)?;
            
            // Forward pass
            let logits = self.model.forward(&transformed_input)?;
            
            // Sample with temperature
            let next_idx = sample_from_logits(&logits.data, temp, rng);

            // Decode prediction
            let predicted_hex = &self.meta.class_names[next_idx];
            let next_char = hex_to_char(predicted_hex);

            generated.push(next_char);
        }

        Ok(generated)
    }
}

/// Translates a character into its exact lowercase hex-string representation.
pub fn char_to_hex(ch: char) -> String {
    format!("{:x}", ch as u32)
}

/// Decodes an exact hex-string representation back into its character representation.
pub fn hex_to_char(hex: &str) -> char {
    let code = u32::from_str_radix(hex, 16).unwrap_or(32); // Fallback to space on parse error
    std::char::from_u32(code).unwrap_or(' ')
}

/// Builds a clean hex-encoded sliding window CSV dataset for causal character sequence training.
pub fn build_csv_dataset(corpus: &str, context_len: usize) -> Result<String> {
    let chars: Vec<char> = corpus.chars().filter(|&c| c != '\r').collect();
    if chars.len() < context_len {
        return Err(InferError::DimMismatch("Corpus length shorter than context window".into()));
    }
    
    let mut vocab: HashSet<char> = chars.iter().copied().collect();
    vocab.insert(' ');
    vocab.insert('\n');

    let mut vocab_vec: Vec<char> = vocab.into_iter().collect();
    vocab_vec.sort();

    let mut csv = String::new();
    // Header row
    for i in 0..context_len {
        csv.push_str(&format!("c{},", i));
    }
    csv.push_str("label\n");

    // Vocabulary alignment padding to force class_names to cover all characters in exact sorted order at the beginning
    for &ch in &vocab_vec {
        for _ in 0..context_len {
            csv.push_str("0.0,");
        }
        csv.push_str(&format!("{}\n", char_to_hex(ch)));
    }

    // Sliding windows
    for i in 0..chars.len().saturating_sub(context_len) {
        let context = &chars[i..i + context_len];
        let target = chars[i + context_len];

        for &ch in context {
            let idx = vocab_vec.iter().position(|&c| c == ch).unwrap_or(0);
            csv.push_str(&format!("{}.0,", idx));
        }
        csv.push_str(&format!("{}\n", char_to_hex(target)));
    }

    Ok(csv)
}

/// Core logits sampler scaled by temperature.
fn sample_from_logits(logits: &[f32], temp: f32, rng: &mut Rng) -> usize {
    let t = temp.max(0.01);
    
    // Apply softmax with temperature
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exp_logits: Vec<f32> = logits.iter().map(|&l| ((l - max) / t).exp()).collect();
    let sum: f32 = exp_logits.iter().sum();
    
    if sum <= 1e-10 {
        return rng.next_u64() as usize % logits.len();
    }
    
    for v in &mut exp_logits {
        *v /= sum;
    }
    
    let r = rng.next_f32();
    let mut cumsum = 0.0f32;
    for (i, &p) in exp_logits.iter().enumerate() {
        cumsum += p;
        if r <= cumsum {
            return i;
        }
    }
    logits.len() - 1
}
