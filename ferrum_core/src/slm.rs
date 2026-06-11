//! Generic offline Edge Generative SLM Library Engine.
use crate::error::{InferError, Result};
use crate::csv::{CsvDataset, ModelMetadata, Normalizer, TaskType};
use crate::rng::Rng;
use crate::optim::{Adam, Sgd};
use crate::train::{Net, train_epoch};
use crate::train_transformer::{train_transformer_epoch, TransformerNet};
use crate::model::Sequential;
use crate::loader::{to_bytes, from_bytes};
use crate::tensor::Tensor;
use crate::verbose;
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
        vprintln!("[slm::GenerativeSLM::new] Creating SLM with {} layers, input_dim={}, output_dim={}",
            model.len(), meta.input_dim, meta.output_dim);
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
        vprintln!("[slm::GenerativeSLM::train_with_callback] ═══════════════════════════════════════");
        vprintln!("[slm::GenerativeSLM::train_with_callback] Starting SLM training:");
        vprintln!("[slm::GenerativeSLM::train_with_callback]   corpus length:  {} chars", corpus.len());
        vprintln!("[slm::GenerativeSLM::train_with_callback]   context_len:    {}", context_len);
        vprintln!("[slm::GenerativeSLM::train_with_callback]   hidden_size:    {}", hidden_size);
        vprintln!("[slm::GenerativeSLM::train_with_callback]   epochs:         {}", epochs);
        vprintln!("[slm::GenerativeSLM::train_with_callback]   lr:             {}", lr);
        vprintln!("[slm::GenerativeSLM::train_with_callback]   momentum:       {}", momentum);
        vprintln!("[slm::GenerativeSLM::train_with_callback]   batch_size:     {}", batch_size);
        vprintln!("[slm::GenerativeSLM::train_with_callback] ═══════════════════════════════════════");

        vprintln!("[slm::train] Building CSV dataset from corpus...");
        let csv_build_start = std::time::Instant::now();
        let csv_data = build_csv_dataset(corpus, context_len)?;
        vprintln!("[slm::train] CSV dataset built in {:.1}ms, size={} bytes",
            csv_build_start.elapsed().as_secs_f64() * 1000.0, csv_data.len());

        vprintln!("[slm::train] Parsing CSV dataset...");
        let parse_start = std::time::Instant::now();
        // Register the full sorted vocabulary explicitly so class indices
        // cover every character (even ones never appearing as a target) in
        // exact sorted order — no padding rows needed.
        let class_names: Vec<String> = corpus_vocab(corpus).iter().map(|&ch| char_to_hex(ch)).collect();
        let ds = CsvDataset::from_str_with_classes(&csv_data, &class_names)?;
        vprintln!("[slm::train] Parsed in {:.1}ms: rows={}, features={}, classes={}",
            parse_start.elapsed().as_secs_f64() * 1000.0, ds.len(), ds.num_features, ds.num_classes);

        vprintln!("[slm::train] Converting to tensors...");
        let (x_raw, y_cls, _) = ds.to_tensors()?;
        vprintln!("[slm::train] Tensor shapes: x={:?}, y_len={}", x_raw.shape, y_cls.len());

        vprintln!("[slm::train] Fitting normalizer (identity for SLM)...");
        let mut norm = Normalizer::fit(&x_raw)?;
        for m in &mut norm.means { *m = 0.0; }
        for s in &mut norm.stds { *s = 1.0; }
        let x_train = norm.transform(&x_raw)?;
        vprintln!("[slm::train] Normalizer applied, x_train shape={:?}", x_train.shape);

        vprintln!("[slm::train] Creating trainable MLP...");
        let mut net = Net::mlp(ds.num_features, hidden_size, ds.num_classes, rng);
        let opt = Sgd::with_momentum(lr, momentum);
        vprintln!("[slm::train] Network: {} params, optimizer: lr={}, momentum={}",
            net.num_params(), lr, momentum);

        vprintln!("[slm::train] ── Beginning training loop ({} epochs) ──", epochs);
        let train_start = std::time::Instant::now();

        for ep in 1..=epochs {
            let ep_start = std::time::Instant::now();
            vprintln!("[slm::train] ── Epoch {}/{} ──", ep, epochs);

            let loss = train_epoch(&mut net, &x_train, &y_cls, batch_size, &opt, rng)?;

            let ep_ms = ep_start.elapsed().as_secs_f64() * 1000.0;
            let total_elapsed = train_start.elapsed().as_secs_f64();
            let eta_secs = if ep > 0 {
                (total_elapsed / ep as f64) * (epochs - ep) as f64
            } else {
                0.0
            };

            vprintln!("[slm::train] Epoch {}/{}: loss={:.6}, time={:.1}ms, ETA={:.1}s",
                ep, epochs, loss, ep_ms, eta_secs);

            if verbose::is_verbose() {
                if loss.is_nan() {
                    println!("[ferrum_core::WARN] ⚠️  NaN loss at epoch {}! Training is diverging!", ep);
                }
                if loss.is_infinite() {
                    println!("[ferrum_core::WARN] ⚠️  Infinite loss at epoch {}! Training is diverging!", ep);
                }
                if loss > 1e6 {
                    println!("[ferrum_core::WARN] ⚠️  Very large loss ({:.2}) at epoch {} — possible explosion!", loss, ep);
                }
            }

            progress_callback(ep, loss);
        }

        let total_train_time = train_start.elapsed().as_secs_f64();
        vprintln!("[slm::train] ── Training complete in {:.2}s ──", total_train_time);

        vprintln!("[slm::train] Converting to inference model...");
        let model = net.to_inference_task(TaskType::Classification)?;
        vprintln!("[slm::train] Inference model has {} layers", model.len());

        let meta = ModelMetadata {
            dataset_name: "GenerativeSLM Model".into(),
            task: TaskType::Classification,
            feature_names: ds.feature_names.clone(),
            feature_ranges: ds.feature_ranges.clone(),
            class_names: ds.class_names.clone(),
            target_name: "next_char".into(),
            target_range: ds.target_range,
            input_dim: ds.num_features,
            output_dim: ds.num_classes,
        };
        vprintln!("[slm::train] Metadata: input_dim={}, output_dim={}, vocab={}",
            meta.input_dim, meta.output_dim, meta.class_names.len());

        Ok(Self { model, norm, meta })
    }

    /// Train a true decoder-only causal Transformer SLM on a raw text corpus.
    ///
    /// Unlike [`GenerativeSLM::train`] (a flat one-hot MLP), this trains
    /// token + positional embeddings, `num_blocks` causal multi-head attention
    /// blocks, and an LM head end-to-end with Adam, using next-token loss at
    /// every position. The exported model serializes to FINF v4 and runs in
    /// WASM via `TransformerSLMModel` (including KV-cached generation).
    #[allow(clippy::too_many_arguments)]
    pub fn train_transformer(
        corpus: &str,
        context_len: usize,
        embed_dim: usize,
        num_heads: usize,
        num_blocks: usize,
        hidden_dim: usize,
        epochs: usize,
        lr: f32,
        batch_size: usize,
        rng: &mut Rng,
    ) -> Result<Self> {
        Self::train_transformer_with_callback(
            corpus, context_len, embed_dim, num_heads, num_blocks, hidden_dim,
            epochs, lr, batch_size, rng, |_, _| {},
        )
    }

    /// [`GenerativeSLM::train_transformer`] with an `(epoch, loss)` callback.
    #[allow(clippy::too_many_arguments)]
    pub fn train_transformer_with_callback<F>(
        corpus: &str,
        context_len: usize,
        embed_dim: usize,
        num_heads: usize,
        num_blocks: usize,
        hidden_dim: usize,
        epochs: usize,
        lr: f32,
        batch_size: usize,
        rng: &mut Rng,
        mut progress_callback: F,
    ) -> Result<Self>
    where
        F: FnMut(usize, f32),
    {
        let (vocab_vec, tokens) = tokenize_corpus(corpus, context_len)?;
        let vocab_size = vocab_vec.len();
        vprintln!("[slm::train_transformer] corpus={} chars, vocab={}, ctx={}, dim={}, heads={}, blocks={}, hidden={}",
            tokens.len(), vocab_size, context_len, embed_dim, num_heads, num_blocks, hidden_dim);

        let mut net = TransformerNet::new(
            vocab_size, context_len, embed_dim, num_heads, hidden_dim, num_blocks, rng,
        )?;
        vprintln!("[slm::train_transformer] {} params, Adam lr={}", net.num_params(), lr);

        let adam = Adam::new(lr);
        for ep in 1..=epochs {
            let loss = train_transformer_epoch(&mut net, &tokens, batch_size, &adam, rng)?;
            vprintln!("[slm::train_transformer] epoch {}/{}: loss={:.6}", ep, epochs, loss);
            progress_callback(ep, loss);
        }

        let model = net.to_inference()?;
        let class_names: Vec<String> = vocab_vec.iter().map(|&ch| char_to_hex(ch)).collect();
        let meta = ModelMetadata {
            dataset_name: "GenerativeSLM Transformer".into(),
            task: TaskType::TransformerSLM,
            feature_names: (0..context_len).map(|i| format!("c_{i}")).collect(),
            feature_ranges: vec![[0.0, vocab_size as f32]; context_len],
            class_names,
            target_name: "next_char".into(),
            target_range: [0.0, vocab_size as f32],
            input_dim: context_len,
            output_dim: vocab_size,
        };
        let norm = Normalizer { means: vec![], stds: vec![] };
        Ok(Self { model, norm, meta })
    }

    /// Train a compact token-ID + embedding MLP language model.
    ///
    /// The recommended simple path: like [`GenerativeSLM::train`] but the
    /// flat one-hot input (`context_len × vocab_size` wide) is replaced by a
    /// learned embedding table, so model size no longer scales with the
    /// vocabulary squared. Inputs at inference are token IDs
    /// (`input_dim = context_len`), the same contract as the transformer path.
    #[allow(clippy::too_many_arguments)]
    pub fn train_embedded(
        corpus: &str,
        context_len: usize,
        embed_dim: usize,
        hidden_size: usize,
        epochs: usize,
        lr: f32,
        momentum: f32,
        batch_size: usize,
        rng: &mut Rng,
    ) -> Result<Self> {
        Self::train_embedded_with_callback(
            corpus, context_len, embed_dim, hidden_size, epochs, lr, momentum,
            batch_size, rng, |_, _| {},
        )
    }

    /// [`GenerativeSLM::train_embedded`] with an `(epoch, loss)` callback.
    #[allow(clippy::too_many_arguments)]
    pub fn train_embedded_with_callback<F>(
        corpus: &str,
        context_len: usize,
        embed_dim: usize,
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
        let (vocab_vec, tokens) = tokenize_corpus(corpus, context_len)?;
        let vocab_size = vocab_vec.len();
        let n_windows = tokens.len() - context_len;
        vprintln!("[slm::train_embedded] corpus={} chars, vocab={}, ctx={}, E={}, hidden={}, windows={}",
            tokens.len(), vocab_size, context_len, embed_dim, hidden_size, n_windows);

        // Sliding windows of token IDs — no CSV round-trip needed.
        let mut x_data = Vec::with_capacity(n_windows * context_len);
        let mut y = Vec::with_capacity(n_windows);
        for i in 0..n_windows {
            x_data.extend(tokens[i..i + context_len].iter().map(|&t| t as f32));
            y.push(tokens[i + context_len]);
        }
        let x = Tensor::matrix(n_windows, context_len, x_data)?;

        let mut net = Net::embedding_mlp(
            vocab_size, context_len, embed_dim, hidden_size, vocab_size, rng,
        );
        let opt = Sgd::with_momentum(lr, momentum);
        vprintln!("[slm::train_embedded] {} params, SGD lr={}, momentum={}",
            net.num_params(), lr, momentum);

        for ep in 1..=epochs {
            let loss = train_epoch(&mut net, &x, &y, batch_size, &opt, rng)?;
            vprintln!("[slm::train_embedded] epoch {}/{}: loss={:.6}", ep, epochs, loss);
            progress_callback(ep, loss);
        }

        let model = net.to_inference_task(TaskType::Classification)?;
        let class_names: Vec<String> = vocab_vec.iter().map(|&ch| char_to_hex(ch)).collect();
        // task = TransformerSLM marks the token-ID input contract (the family
        // flag `generate` keys on), independent of the internal architecture.
        let meta = ModelMetadata {
            dataset_name: "GenerativeSLM Embedded".into(),
            task: TaskType::TransformerSLM,
            feature_names: (0..context_len).map(|i| format!("c_{i}")).collect(),
            feature_ranges: vec![[0.0, vocab_size as f32]; context_len],
            class_names,
            target_name: "next_char".into(),
            target_range: [0.0, vocab_size as f32],
            input_dim: context_len,
            output_dim: vocab_size,
        };
        let norm = Normalizer { means: vec![], stds: vec![] };
        Ok(Self { model, norm, meta })
    }

    /// Serialize the trained Generative SLM model to self-contained FINF v4 bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        vprintln!("[slm::GenerativeSLM::to_bytes] Serializing model...");
        let bytes = to_bytes(&self.model, &self.norm, &self.meta)?;
        vprintln!("[slm::GenerativeSLM::to_bytes] Serialized to {} bytes", bytes.len());
        Ok(bytes)
    }

    /// Serialize to FINF v5 with int8-quantised weights (≈4× smaller files).
    pub fn to_bytes_quantized(&self) -> Result<Vec<u8>> {
        vprintln!("[slm::GenerativeSLM::to_bytes_quantized] Serializing quantized model...");
        let bytes = crate::loader::to_bytes_quantized(&self.model, &self.norm, &self.meta)?;
        vprintln!("[slm::GenerativeSLM::to_bytes_quantized] Serialized to {} bytes", bytes.len());
        Ok(bytes)
    }

    /// Load a trained Generative SLM model from self-contained FINF v4 bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        vprintln!("[slm::GenerativeSLM::from_bytes] Deserializing from {} bytes...", bytes.len());
        let (model, norm, meta) = from_bytes(bytes)?;
        vprintln!("[slm::GenerativeSLM::from_bytes] Loaded: {} layers, input_dim={}, output_dim={}",
            model.len(), meta.input_dim, meta.output_dim);
        Ok(Self { model, norm, meta })
    }

    /// Autoregressively generate next-character sequence completions from a seed text.
    pub fn generate(&self, seed: &str, num_chars: usize, temp: f32, rng: &mut Rng) -> Result<String> {
        vprintln!("[slm::GenerativeSLM::generate] seed=\"{}\", num_chars={}, temp={:.2}",
            seed.chars().take(50).collect::<String>(), num_chars, temp);

        let mut generated = seed.to_string();
        let vocab_size = self.meta.output_dim;
        let input_dim = self.meta.input_dim;
        // Transformer models take context_len token IDs; the MLP takes a
        // flattened one-hot context of context_len × vocab_size values.
        let is_transformer = self.meta.task == TaskType::TransformerSLM;
        let context_len = if is_transformer { input_dim } else { input_dim / vocab_size };

        vprintln!("[slm::generate] vocab_size={}, input_dim={}, context_len={}, transformer={}",
            vocab_size, input_dim, context_len, is_transformer);

        for step in 0..num_chars {
            let current_len = generated.chars().count();
            if current_len < context_len {
                vprintln!("[slm::generate] Step {}: generated length {} < context_len {}, stopping", step, current_len, context_len);
                break;
            }
            let context_chars: Vec<char> = generated.chars().skip(current_len - context_len).collect();

            vprintln!("[slm::generate] Step {}: context=\"{}\"",
                step, context_chars.iter().collect::<String>());

            let char_idx = |ch: char| -> usize {
                let hex = char_to_hex(ch);
                self.meta.class_names.iter().position(|s| s == &hex).unwrap_or(0)
            };

            let next_dist: Vec<f32> = if is_transformer {
                // Token-ID input → [T, vocab] probabilities; keep the last row.
                let ids: Vec<f32> = context_chars.iter().map(|&ch| char_idx(ch) as f32).collect();
                let input = Tensor::matrix(1, context_len, ids)?;
                let out = self.model.forward(&input)?;
                let (rows, cols) = out.matrix_dims()?;
                // The model ends in Softmax; convert probabilities back to
                // log-space so temperature scaling behaves correctly.
                out.data[(rows - 1) * cols..]
                    .iter()
                    .map(|&p| p.max(1e-12).ln())
                    .collect()
            } else {
                // One-hot context for the MLP path.
                let mut input_data = Vec::with_capacity(input_dim);
                for &ch in &context_chars {
                    let idx = char_idx(ch);
                    for j in 0..vocab_size {
                        input_data.push(if j == idx { 1.0 } else { 0.0 });
                    }
                }
                let input_tensor = Tensor::row(input_data)?;
                let transformed_input = self.norm.transform(&input_tensor)?;
                self.model.forward(&transformed_input)?.data
            };

            if verbose::is_verbose() {
                let (lmin, lmax, lmean) = verbose::stats(&next_dist);
                vprintln!("[slm::generate] Step {}: logits stats: min={:.4}, max={:.4}, mean={:.4}",
                    step, lmin, lmax, lmean);
            }

            // Sample with temperature
            let next_idx = sample_from_logits(&next_dist, temp, rng);

            // Decode prediction
            let predicted_hex = &self.meta.class_names[next_idx];
            let next_char = hex_to_char(predicted_hex);

            vprintln!("[slm::generate] Step {}: sampled idx={}, hex=\"{}\", char='{}'",
                step, next_idx, predicted_hex, next_char);

            generated.push(next_char);
        }

        vprintln!("[slm::generate] Generated {} total chars", generated.len());
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

/// The sorted character vocabulary of a corpus (always includes ' ' and '\n',
/// never '\r').
pub fn corpus_vocab(corpus: &str) -> Vec<char> {
    let mut vocab: HashSet<char> = corpus.chars().filter(|&c| c != '\r').collect();
    vocab.insert(' ');
    vocab.insert('\n');
    let mut vocab_vec: Vec<char> = vocab.into_iter().collect();
    vocab_vec.sort();
    vocab_vec
}

/// Tokenize a corpus into (sorted vocabulary, token-ID stream) for
/// transformer training. The vocabulary always includes ' ' and '\n'.
pub fn tokenize_corpus(corpus: &str, context_len: usize) -> Result<(Vec<char>, Vec<usize>)> {
    let chars: Vec<char> = corpus.chars().filter(|&c| c != '\r').collect();
    if chars.len() < context_len + 1 {
        return Err(InferError::DimMismatch(
            "Corpus must be longer than the context window".into(),
        ));
    }
    let vocab_vec = corpus_vocab(corpus);
    let tokens: Vec<usize> = chars
        .iter()
        .map(|c| vocab_vec.binary_search(c).unwrap_or(0))
        .collect();
    Ok((vocab_vec, tokens))
}

/// Builds a clean hex-encoded sliding window CSV dataset for causal character sequence training with one-hot encoded inputs.
pub fn build_csv_dataset(corpus: &str, context_len: usize) -> Result<String> {
    vprintln!("[slm::build_csv_dataset] corpus_len={}, context_len={}", corpus.len(), context_len);

    let chars: Vec<char> = corpus.chars().filter(|&c| c != '\r').collect();
    if chars.len() < context_len {
        return Err(InferError::DimMismatch("Corpus length shorter than context window".into()));
    }
    
    let vocab_vec = corpus_vocab(corpus);
    let v_size = vocab_vec.len();

    vprintln!("[slm::build_csv_dataset] chars={}, vocab_size={}, sliding_windows={}",
        chars.len(), v_size, chars.len().saturating_sub(context_len));
    vprintln!("[slm::build_csv_dataset] input_dim={} (context_len × vocab_size)",
        context_len * v_size);

    let mut csv = String::new();
    // Header row
    for i in 0..context_len {
        for j in 0..v_size {
            csv.push_str(&format!("c{}_v{},", i, j));
        }
    }
    csv.push_str("label\n");

    // Class coverage and ordering come from explicit registration
    // (`CsvDataset::from_str_with_classes` with the sorted hex vocabulary) —
    // no all-zero alignment rows are injected.

    // Sliding windows
    let window_count = chars.len().saturating_sub(context_len);
    vprintln!("[slm::build_csv_dataset] Writing {} sliding window rows", window_count);
    for i in 0..window_count {
        let context = &chars[i..i + context_len];
        let target = chars[i + context_len];

        for &ch in context {
            let idx = vocab_vec.iter().position(|&c| c == ch).unwrap_or(0);
            for j in 0..v_size {
                if j == idx {
                    csv.push_str("1.0,");
                } else {
                    csv.push_str("0.0,");
                }
            }
        }
        csv.push_str(&format!("{}\n", char_to_hex(target)));
    }

    vprintln!("[slm::build_csv_dataset] CSV built: {} bytes, {} total rows",
        csv.len(), window_count);

    Ok(csv)
}

/// Core logits sampler scaled by temperature.
fn sample_from_logits(logits: &[f32], temp: f32, rng: &mut Rng) -> usize {
    let t = temp.max(0.01);
    
    // Apply softmax with temperature
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exp_logits: Vec<f32> = logits.iter().map(|&l| ((l - max) / t).exp()).collect();
    let sum: f32 = exp_logits.iter().sum();

    vprintln!("[slm::sample_from_logits] temp={:.3}, max_logit={:.4}, softmax_sum={:.6}, vocab_size={}",
        t, max, sum, logits.len());
    
    if sum <= 1e-10 {
        let fallback = rng.next_u64() as usize % logits.len();
        vprintln!("[slm::sample_from_logits] ⚠️  Near-zero softmax sum, using random fallback idx={}", fallback);
        return fallback;
    }
    
    for v in &mut exp_logits {
        *v /= sum;
    }
    
    let r = rng.next_f32();
    let mut cumsum = 0.0f32;
    for (i, &p) in exp_logits.iter().enumerate() {
        cumsum += p;
        if r <= cumsum {
            vprintln!("[slm::sample_from_logits] random={:.4}, selected idx={}, prob={:.4}", r, i, p);
            return i;
        }
    }
    let last = logits.len() - 1;
    vprintln!("[slm::sample_from_logits] Fell through, returning last idx={}", last);
    last
}
