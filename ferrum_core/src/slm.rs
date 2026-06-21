//! Generic offline Edge Generative SLM Library Engine.
use crate::error::{InferError, Result};
use crate::csv::{CsvDataset, ModelMetadata, Normalizer, TaskType};
use crate::rng::Rng;
use crate::optim::{Adam, Sgd};
use crate::train::{Net, train_epoch};
use crate::train_transformer::{train_transformer_epoch_threaded, TransformerNet};
use crate::model::Sequential;
use crate::loader::{to_bytes, from_bytes};
use crate::tensor::Tensor;
use crate::tokenizer::ByteBpeTokenizer;
use crate::verbose;
use std::collections::HashSet;

/// Hyperparameters for training a causal transformer SLM.
///
/// Used by [`GenerativeSLM::train_transformer_config`] and
/// [`GenerativeSLM::load_or_train`]. `Default` matches the CLI defaults.
#[derive(Clone, Debug, PartialEq)]
pub struct TransformerConfig {
    /// Context window in characters.
    pub context_len: usize,
    /// Embedding dimension (must be divisible by `num_heads`).
    pub embed_dim: usize,
    /// Attention heads per block.
    pub num_heads: usize,
    /// Number of transformer blocks.
    pub num_blocks: usize,
    /// FFN hidden width.
    pub hidden_dim: usize,
    /// Training epochs.
    pub epochs: usize,
    /// Adam learning rate.
    pub lr: f32,
    /// Minibatch size (sequences per step).
    pub batch_size: usize,
    /// Byte-level BPE target vocabulary size. `0` selects character-level
    /// tokenization (legacy behaviour); any value `>= 256` trains a BPE
    /// tokenizer of that size and stores it inside the model.
    pub vocab_size: usize,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            context_len: 16,
            embed_dim: 32,
            num_heads: 4,
            num_blocks: 2,
            hidden_dim: 64,
            epochs: 100,
            lr: 0.01,
            batch_size: 16,
            vocab_size: 512,
        }
    }
}

/// Result of scoring a trained model against held-out text with
/// [`GenerativeSLM::evaluate`].
///
/// All three fields describe the model's next-token predictive quality on the
/// supplied text, using exactly the same forward path that generation uses, so
/// the numbers reflect the shipped (quantization-aware) model.
#[derive(Clone, Debug, PartialEq)]
pub struct Evaluation {
    /// Number of next-token predictions scored (positions past the first
    /// `context_len` tokens of the held-out text).
    pub num_predictions: usize,
    /// Mean next-token negative log-likelihood in **nats** (natural-log
    /// cross-entropy). Lower is better.
    pub cross_entropy: f32,
    /// Mean next-token cross-entropy in **bits** (`cross_entropy / ln 2`).
    pub bits_per_token: f32,
    /// Perplexity, `exp(cross_entropy)`. Lower is better; a uniform model over a
    /// vocabulary of `V` tokens scores `V`.
    pub perplexity: f32,
}

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

        vprintln!("[slm::train] Creating trainable MLP (QAT enabled)...");
        let mut net = Net::mlp(ds.num_features, hidden_size, ds.num_classes, rng);
        net.set_qat(true);
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
                    crate::verbose::log_line(&format!("[ferrum_core::WARN] ⚠️  NaN loss at epoch {}! Training is diverging!", ep));
                }
                if loss.is_infinite() {
                    crate::verbose::log_line(&format!("[ferrum_core::WARN] ⚠️  Infinite loss at epoch {}! Training is diverging!", ep));
                }
                if loss > 1e6 {
                    crate::verbose::log_line(&format!("[ferrum_core::WARN] ⚠️  Very large loss ({:.2}) at epoch {} — possible explosion!", loss, ep));
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
            // One-hot MLP path is always character-level (no BPE tokenizer).
            tokenizer_state: String::new(),
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
    ///
    /// `vocab_size` selects the tokenizer: `0` is character-level (the corpus's
    /// sorted character set), and any value `>= 256` trains a byte-level BPE
    /// tokenizer of that size whose merge list is stored inside the model.
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
        vocab_size: usize,
        rng: &mut Rng,
    ) -> Result<Self> {
        Self::train_transformer_with_callback(
            corpus, context_len, embed_dim, num_heads, num_blocks, hidden_dim,
            epochs, lr, batch_size, vocab_size, rng, |_, _| {},
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
        vocab_size: usize,
        rng: &mut Rng,
        progress_callback: F,
    ) -> Result<Self>
    where
        F: FnMut(usize, f32),
    {
        // `threads = 1` is the serial path, bit-for-bit identical to before.
        Self::train_transformer_inner(
            corpus, context_len, embed_dim, num_heads, num_blocks, hidden_dim,
            epochs, lr, batch_size, vocab_size, 1, rng, progress_callback,
        )
    }

    /// Data-parallel [`GenerativeSLM::train_transformer_with_callback`]:
    /// `threads` worker threads split each minibatch and their gradients are
    /// reduced before each optimizer step (see
    /// [`crate::train_transformer::train_transformer_epoch_threaded`]).
    ///
    /// Pass `0` to auto-detect the machine's parallelism via
    /// [`crate::num_threads`]. `1` forces the serial path. Results are
    /// reproducible for a fixed `threads` value; the model is identical to the
    /// serial one when `threads` resolves to `1`.
    #[allow(clippy::too_many_arguments)]
    pub fn train_transformer_threaded_with_callback<F>(
        corpus: &str,
        context_len: usize,
        embed_dim: usize,
        num_heads: usize,
        num_blocks: usize,
        hidden_dim: usize,
        epochs: usize,
        lr: f32,
        batch_size: usize,
        vocab_size: usize,
        threads: usize,
        rng: &mut Rng,
        progress_callback: F,
    ) -> Result<Self>
    where
        F: FnMut(usize, f32),
    {
        let threads = if threads == 0 { crate::parallel::num_threads() } else { threads };
        Self::train_transformer_inner(
            corpus, context_len, embed_dim, num_heads, num_blocks, hidden_dim,
            epochs, lr, batch_size, vocab_size, threads, rng, progress_callback,
        )
    }

    /// Shared implementation behind the serial and threaded transformer trainers.
    #[allow(clippy::too_many_arguments)]
    fn train_transformer_inner<F>(
        corpus: &str,
        context_len: usize,
        embed_dim: usize,
        num_heads: usize,
        num_blocks: usize,
        hidden_dim: usize,
        epochs: usize,
        lr: f32,
        batch_size: usize,
        vocab_size: usize,
        threads: usize,
        rng: &mut Rng,
        mut progress_callback: F,
    ) -> Result<Self>
    where
        F: FnMut(usize, f32),
    {
        let tc = tokenize_for_lm(corpus, context_len, vocab_size)?;
        let model_vocab = tc.vocab_size;
        vprintln!("[slm::train_transformer] corpus={} chars, vocab={}, bpe={}, ctx={}, dim={}, heads={}, blocks={}, hidden={}, threads={}",
            tc.tokens.len(), model_vocab, !tc.tokenizer_state.is_empty(),
            context_len, embed_dim, num_heads, num_blocks, hidden_dim, threads);

        let mut net = TransformerNet::new(
            model_vocab, context_len, embed_dim, num_heads, hidden_dim, num_blocks, rng,
        )?;
        net.set_qat(true);
        vprintln!("[slm::train_transformer] {} params, Adam lr={}, QAT=int8, threads={}",
            net.num_params(), lr, threads);

        let adam = Adam::new(lr);
        for ep in 1..=epochs {
            let loss = train_transformer_epoch_threaded(
                &mut net, &tc.tokens, batch_size, &adam, rng, threads,
            )?;
            vprintln!("[slm::train_transformer] epoch {}/{}: loss={:.6}", ep, epochs, loss);
            progress_callback(ep, loss);
        }

        let model = net.to_inference()?;
        let meta = ModelMetadata {
            dataset_name: "GenerativeSLM Transformer".into(),
            task: TaskType::TransformerSLM,
            feature_names: (0..context_len).map(|i| format!("c_{i}")).collect(),
            feature_ranges: vec![[0.0, model_vocab as f32]; context_len],
            class_names: tc.class_names,
            target_name: "next_char".into(),
            target_range: [0.0, model_vocab as f32],
            input_dim: context_len,
            output_dim: model_vocab,
            tokenizer_state: tc.tokenizer_state,
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
    ///
    /// `vocab_size` selects the tokenizer exactly as in
    /// [`GenerativeSLM::train_transformer`]: `0` is character-level, `>= 256`
    /// trains and embeds a byte-level BPE tokenizer.
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
        vocab_size: usize,
        rng: &mut Rng,
    ) -> Result<Self> {
        Self::train_embedded_with_callback(
            corpus, context_len, embed_dim, hidden_size, epochs, lr, momentum,
            batch_size, vocab_size, rng, |_, _| {},
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
        vocab_size: usize,
        rng: &mut Rng,
        mut progress_callback: F,
    ) -> Result<Self>
    where
        F: FnMut(usize, f32),
    {
        let tc = tokenize_for_lm(corpus, context_len, vocab_size)?;
        let model_vocab = tc.vocab_size;
        let n_windows = tc.tokens.len() - context_len;
        vprintln!("[slm::train_embedded] corpus={} chars, vocab={}, bpe={}, ctx={}, E={}, hidden={}, windows={}",
            tc.tokens.len(), model_vocab, !tc.tokenizer_state.is_empty(),
            context_len, embed_dim, hidden_size, n_windows);

        // Sliding windows of token IDs — no CSV round-trip needed.
        let mut x_data = Vec::with_capacity(n_windows * context_len);
        let mut y = Vec::with_capacity(n_windows);
        for i in 0..n_windows {
            x_data.extend(tc.tokens[i..i + context_len].iter().map(|&t| t as f32));
            y.push(tc.tokens[i + context_len]);
        }
        let x = Tensor::matrix(n_windows, context_len, x_data)?;

        let mut net = Net::embedding_mlp(
            model_vocab, context_len, embed_dim, hidden_size, model_vocab, rng,
        );
        net.set_qat(true);
        let opt = Sgd::with_momentum(lr, momentum);
        vprintln!("[slm::train_embedded] {} params, SGD lr={}, momentum={}",
            net.num_params(), lr, momentum);

        for ep in 1..=epochs {
            let loss = train_epoch(&mut net, &x, &y, batch_size, &opt, rng)?;
            vprintln!("[slm::train_embedded] epoch {}/{}: loss={:.6}", ep, epochs, loss);
            progress_callback(ep, loss);
        }

        let model = net.to_inference_task(TaskType::Classification)?;
        // task = TransformerSLM marks the token-ID input contract (the family
        // flag `generate` keys on), independent of the internal architecture.
        let meta = ModelMetadata {
            dataset_name: "GenerativeSLM Embedded".into(),
            task: TaskType::TransformerSLM,
            feature_names: (0..context_len).map(|i| format!("c_{i}")).collect(),
            feature_ranges: vec![[0.0, model_vocab as f32]; context_len],
            class_names: tc.class_names,
            target_name: "next_char".into(),
            target_range: [0.0, model_vocab as f32],
            input_dim: context_len,
            output_dim: model_vocab,
            tokenizer_state: tc.tokenizer_state,
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

    /// [`GenerativeSLM::train_transformer_with_callback`] driven by a
    /// [`TransformerConfig`]. Training is int8 quantization-aware (QAT).
    pub fn train_transformer_config<F>(
        corpus: &str,
        cfg: &TransformerConfig,
        rng: &mut Rng,
        progress_callback: F,
    ) -> Result<Self>
    where
        F: FnMut(usize, f32),
    {
        Self::train_transformer_with_callback(
            corpus,
            cfg.context_len,
            cfg.embed_dim,
            cfg.num_heads,
            cfg.num_blocks,
            cfg.hidden_dim,
            cfg.epochs,
            cfg.lr,
            cfg.batch_size,
            cfg.vocab_size,
            rng,
            progress_callback,
        )
    }

    /// Data-parallel [`GenerativeSLM::train_transformer_config`]: trains with
    /// `threads` worker threads (`0` = auto-detect via [`crate::num_threads`],
    /// `1` = serial). Training stays int8 quantization-aware (QAT) and the
    /// result is reproducible for a fixed `threads` value.
    pub fn train_transformer_config_threaded<F>(
        corpus: &str,
        cfg: &TransformerConfig,
        threads: usize,
        rng: &mut Rng,
        progress_callback: F,
    ) -> Result<Self>
    where
        F: FnMut(usize, f32),
    {
        Self::train_transformer_threaded_with_callback(
            corpus,
            cfg.context_len,
            cfg.embed_dim,
            cfg.num_heads,
            cfg.num_blocks,
            cfg.hidden_dim,
            cfg.epochs,
            cfg.lr,
            cfg.batch_size,
            cfg.vocab_size,
            threads,
            rng,
            progress_callback,
        )
    }

    /// Save the trained model to `model_path` as int8-quantized FINF v5
    /// (the default on-disk format — ≈4× smaller than f32). Parent
    /// directories are created as needed.
    pub fn save(&self, model_path: &str) -> Result<()> {
        let bytes = self.to_bytes_quantized()?;
        if let Some(parent) = std::path::Path::new(model_path).parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        std::fs::write(model_path, &bytes)?;
        vprintln!("[slm::GenerativeSLM::save] Wrote {} bytes → {}", bytes.len(), model_path);
        Ok(())
    }

    /// Load a previously saved model (FINF v4 or v5) from `model_path`.
    pub fn load(model_path: &str) -> Result<Self> {
        let bytes = std::fs::read(model_path)?;
        Self::from_bytes(&bytes)
    }

    /// Load the model from `model_path` if it exists; otherwise train a
    /// transformer SLM on `corpus` with `cfg` (QAT, int8), save it to
    /// `model_path`, and return it.
    ///
    /// The returned `bool` is `true` when the model was loaded from disk
    /// (no training happened) and `false` when it was freshly trained.
    pub fn load_or_train<F>(
        model_path: &str,
        corpus: &str,
        cfg: &TransformerConfig,
        rng: &mut Rng,
        progress_callback: F,
    ) -> Result<(Self, bool)>
    where
        F: FnMut(usize, f32),
    {
        if std::path::Path::new(model_path).exists() {
            vprintln!("[slm::load_or_train] Found {} — loading instead of retraining", model_path);
            return Ok((Self::load(model_path)?, true));
        }
        let slm = Self::train_transformer_config(corpus, cfg, rng, progress_callback)?;
        slm.save(model_path)?;
        Ok((slm, false))
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
    ///
    /// For BPE models (`meta.tokenizer_state` non-empty) generation runs over
    /// subword tokens but `num_chars` still counts **characters**: the output is
    /// `seed` followed by exactly `num_chars` newly generated characters
    /// (fewer only if generation is cut short). Character-level models keep the
    /// original per-character behaviour.
    pub fn generate(&self, seed: &str, num_chars: usize, temp: f32, rng: &mut Rng) -> Result<String> {
        // The full-string API is the streaming API with a no-op sink.
        self.generate_stream(seed, num_chars, temp, rng, |_| {})
    }

    /// Streaming counterpart of [`GenerativeSLM::generate`]: identical sampling
    /// and identical return value, but `on_text` is invoked with each newly
    /// generated **fragment** as soon as it is produced, so callers can print a
    /// completion token-by-token (a REPL, a server's SSE stream, a TUI) instead
    /// of waiting for the whole result.
    ///
    /// Guarantees:
    /// - Concatenating every fragment passed to `on_text` yields exactly the
    ///   generated continuation — i.e. the return value with the `seed` prefix
    ///   removed (equivalently, [`GenerativeSLM::generate_continuation`]'s
    ///   output).
    /// - The seed itself is never emitted.
    /// - For a fixed RNG the output and the fragment sequence are deterministic,
    ///   and the returned `String` equals what [`GenerativeSLM::generate`] would
    ///   return for the same arguments.
    ///
    /// Character-level models emit one character per step. BPE models emit
    /// decoded text as whole tokens land, holding back a partial trailing
    /// multi-byte character until it completes (so no `U+FFFD` placeholder is
    /// ever streamed and then revised).
    pub fn generate_stream<F>(
        &self,
        seed: &str,
        num_chars: usize,
        temp: f32,
        rng: &mut Rng,
        on_text: F,
    ) -> Result<String>
    where
        F: FnMut(&str),
    {
        vprintln!("[slm::GenerativeSLM::generate_stream] seed=\"{}\", num_chars={}, temp={:.2}",
            seed.chars().take(50).collect::<String>(), num_chars, temp);

        if !self.meta.tokenizer_state.is_empty() {
            return self.generate_bpe_stream(seed, num_chars, temp, rng, on_text);
        }

        let mut on_text = on_text;
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
            // Stream exactly the new character (never the seed).
            let mut buf = [0u8; 4];
            on_text(next_char.encode_utf8(&mut buf));
        }

        vprintln!("[slm::generate] Generated {} total chars", generated.len());
        Ok(generated)
    }

    /// Streaming BPE generation path. Rebuilds the stored [`ByteBpeTokenizer`],
    /// encodes the seed to token IDs, samples one token at a time from the model
    /// (always feeding the last `context_len` tokens, left-padded with token 0
    /// when the prompt is shorter than the window), and decodes the whole token
    /// stream back to text. Generation continues until at least `num_chars`
    /// characters have been added past the seed, then the result is trimmed to
    /// exactly that length so the character contract matches the char-level path.
    ///
    /// As tokens land, the newly decoded continuation characters are streamed to
    /// `on_text`, holding back the final (possibly partial) character until the
    /// next token completes it so a `U+FFFD` placeholder is never emitted and
    /// then revised.
    fn generate_bpe_stream<F>(
        &self,
        seed: &str,
        num_chars: usize,
        temp: f32,
        rng: &mut Rng,
        mut on_text: F,
    ) -> Result<String>
    where
        F: FnMut(&str),
    {
        let tok = ByteBpeTokenizer::from_state(&self.meta.tokenizer_state)?;
        let context_len = self.meta.input_dim;
        let seed_chars = seed.chars().count();
        let target_chars = seed_chars + num_chars;

        let mut ids: Vec<usize> = tok.encode(seed);
        vprintln!("[slm::generate_bpe] seed encoded to {} tokens, ctx={}, target_chars={}",
            ids.len(), context_len, target_chars);

        // Number of continuation characters already streamed (never includes the
        // seed). Emit a delta after every token so output is incremental.
        let mut emitted_cont = 0usize;
        let mut flush = |full: &str, hold_back_partial: bool, on_text: &mut F| {
            let cont_chars = full.chars().count().saturating_sub(seed_chars);
            // While generating, withhold the last char (it may be an incomplete
            // multi-byte sequence that the next token will complete); on the
            // final flush emit everything up to the character budget.
            let avail = if hold_back_partial {
                cont_chars.saturating_sub(1).min(num_chars)
            } else {
                cont_chars.min(num_chars)
            };
            if avail > emitted_cont {
                let delta: String = full
                    .chars()
                    .skip(seed_chars + emitted_cont)
                    .take(avail - emitted_cont)
                    .collect();
                on_text(&delta);
                emitted_cont = avail;
            }
        };

        // Bound the loop: every step adds ≥1 byte, but guard against a model
        // that keeps emitting zero-width or repeated control tokens.
        let max_steps = num_chars * 8 + 64;
        let mut steps = 0;
        while tok.decode(&ids).chars().count() < target_chars && steps < max_steps {
            // Last `context_len` tokens, left-padded with token 0 if too short.
            let mut ctx: Vec<f32> = Vec::with_capacity(context_len);
            if ids.len() < context_len {
                ctx.resize(context_len - ids.len(), 0.0);
                ctx.extend(ids.iter().map(|&t| t as f32));
            } else {
                ctx.extend(ids[ids.len() - context_len..].iter().map(|&t| t as f32));
            }
            let input = Tensor::matrix(1, context_len, ctx)?;
            let out = self.model.forward(&input)?;
            let (rows, cols) = out.matrix_dims()?;
            // The model ends in Softmax; convert the last row's probabilities
            // back to log-space so temperature scaling behaves correctly.
            let logits: Vec<f32> = out.data[(rows - 1) * cols..]
                .iter()
                .map(|&p| p.max(1e-12).ln())
                .collect();
            let next = sample_from_logits(&logits, temp, rng);
            ids.push(next);
            steps += 1;
            // Stream what is safely decodable so far.
            flush(&tok.decode(&ids), true, &mut on_text);
        }

        // Decode the full token stream, then keep seed + num_chars characters.
        let full = tok.decode(&ids);
        // Final flush emits any remaining (now-complete) continuation chars.
        flush(&full, false, &mut on_text);
        let trimmed: String = full.chars().take(target_chars).collect();
        vprintln!("[slm::generate_bpe] generated {} tokens → {} chars (trimmed to {})",
            ids.len(), full.chars().count(), trimmed.chars().count());
        Ok(trimmed)
    }

    /// Generate a completion from `seed` and return **only the newly generated
    /// text**, excluding the seed prefix.
    ///
    /// [`GenerativeSLM::generate`] returns `seed` followed by the continuation
    /// (the natural shape for printing a running document); this is the
    /// ergonomic counterpart for callers that only want the model's own output
    /// — chat replies, autocomplete suggestions, code completion. The character
    /// budget (`num_chars`) and sampling semantics are identical to
    /// [`GenerativeSLM::generate`]; only the returned slice differs.
    pub fn generate_continuation(
        &self,
        seed: &str,
        num_chars: usize,
        temp: f32,
        rng: &mut Rng,
    ) -> Result<String> {
        let full = self.generate(seed, num_chars, temp, rng)?;
        // `generate` always returns the seed verbatim as a prefix, so skipping
        // the seed's character count yields exactly the continuation.
        let seed_chars = seed.chars().count();
        Ok(full.chars().skip(seed_chars).collect())
    }

    /// Score the model's next-token predictions against held-out `text`,
    /// returning cross-entropy, bits-per-token, and perplexity.
    ///
    /// This slides the model's context window across `text` and, at every
    /// position past the first `context_len` tokens, measures the probability
    /// the model assigns to the token that actually follows. It dispatches over
    /// the same three families [`GenerativeSLM::generate`] handles — byte-level
    /// BPE, char/token-ID transformer & embedded, and the one-hot MLP — using
    /// the trained (quantization-aware) weights, so the reported perplexity is
    /// the quality of the model you will ship, not a separate f32 reference.
    ///
    /// `text` must contain at least `context_len + 1` tokens; otherwise no
    /// prediction can be scored and a [`InferError::DimMismatch`] is returned.
    /// Use a corpus the model was **not** trained on to measure generalization.
    pub fn evaluate(&self, text: &str) -> Result<Evaluation> {
        vprintln!("[slm::evaluate] scoring {} chars of held-out text", text.chars().count());

        if !self.meta.tokenizer_state.is_empty() {
            let tok = ByteBpeTokenizer::from_state(&self.meta.tokenizer_state)?;
            let ids = tok.encode(text);
            return self.score_token_ids(&ids, self.meta.input_dim);
        }

        let vocab_size = self.meta.output_dim;
        let input_dim = self.meta.input_dim;
        let is_transformer = self.meta.task == TaskType::TransformerSLM;
        let context_len = if is_transformer { input_dim } else { input_dim / vocab_size };

        let chars: Vec<char> = text.chars().filter(|&c| c != '\r').collect();
        let ids: Vec<usize> = chars
            .iter()
            .map(|&ch| {
                let hex = char_to_hex(ch);
                self.meta.class_names.iter().position(|s| s == &hex).unwrap_or(0)
            })
            .collect();

        if is_transformer {
            self.score_token_ids(&ids, context_len)
        } else {
            self.score_onehot(&ids, context_len, vocab_size)
        }
    }

    /// Accumulate held-out cross-entropy for the token-ID families (BPE,
    /// char-level transformer, and embedded MLP). The model is fed the last
    /// `context_len` token IDs and its final-row softmax row gives the predicted
    /// next-token distribution.
    fn score_token_ids(&self, ids: &[usize], context_len: usize) -> Result<Evaluation> {
        if ids.len() <= context_len {
            return Err(InferError::DimMismatch(
                "evaluation text must contain more than context_len tokens".into(),
            ));
        }
        let mut total_nll = 0.0f64;
        let mut count = 0usize;
        for i in context_len..ids.len() {
            let ctx: Vec<f32> = ids[i - context_len..i].iter().map(|&t| t as f32).collect();
            let input = Tensor::matrix(1, context_len, ctx)?;
            let out = self.model.forward(&input)?;
            let (rows, cols) = out.matrix_dims()?;
            // The model ends in Softmax, so the last row holds probabilities.
            let p = out.data[(rows - 1) * cols + ids[i]].max(1e-12);
            total_nll += -(p as f64).ln();
            count += 1;
        }
        Ok(finish_evaluation(total_nll, count))
    }

    /// Accumulate held-out cross-entropy for the one-hot MLP path: the context
    /// is encoded as a flattened `context_len × vocab_size` one-hot row and the
    /// single output row holds the next-token probabilities.
    fn score_onehot(&self, ids: &[usize], context_len: usize, vocab_size: usize) -> Result<Evaluation> {
        if ids.len() <= context_len {
            return Err(InferError::DimMismatch(
                "evaluation text must contain more than context_len tokens".into(),
            ));
        }
        let mut total_nll = 0.0f64;
        let mut count = 0usize;
        for i in context_len..ids.len() {
            let mut input_data = Vec::with_capacity(context_len * vocab_size);
            for &idx in &ids[i - context_len..i] {
                for j in 0..vocab_size {
                    input_data.push(if j == idx { 1.0 } else { 0.0 });
                }
            }
            let input = self.norm.transform(&Tensor::row(input_data)?)?;
            let out = self.model.forward(&input)?;
            let p = out.data.get(ids[i]).copied().unwrap_or(1e-12).max(1e-12);
            total_nll += -(p as f64).ln();
            count += 1;
        }
        Ok(finish_evaluation(total_nll, count))
    }
}

/// Turn an accumulated negative-log-likelihood sum into an [`Evaluation`].
fn finish_evaluation(total_nll: f64, count: usize) -> Evaluation {
    let ce = if count == 0 { 0.0 } else { (total_nll / count as f64) as f32 };
    Evaluation {
        num_predictions: count,
        cross_entropy: ce,
        bits_per_token: ce / std::f32::consts::LN_2,
        perplexity: ce.exp(),
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

/// Everything the token-ID language-model paths need to size a network and
/// describe it in metadata, produced by [`tokenize_for_lm`].
struct LmTokens {
    /// Token-ID stream over the whole corpus.
    tokens: Vec<usize>,
    /// Number of distinct token IDs to embed and predict (the model's
    /// embedding rows and LM-head width).
    vocab_size: usize,
    /// Per-class hex code-point names for character-level models; empty for BPE
    /// (where decoding goes through the stored tokenizer instead).
    class_names: Vec<String>,
    /// Serialized BPE merge list; empty for character-level models.
    tokenizer_state: String,
}

/// Tokenize `corpus` for the transformer / embedded language-model paths.
///
/// `vocab_size == 0` selects character-level tokenization (the sorted corpus
/// vocabulary, identical to [`tokenize_corpus`]). `vocab_size >= 256` trains a
/// byte-level [`ByteBpeTokenizer`] of that target size and encodes the corpus
/// with it, so the same trainer, QAT, and FINF serialization work unchanged on
/// subword tokens. Values in `1..256` are rejected (the 256-byte base
/// vocabulary is irreducible).
fn tokenize_for_lm(corpus: &str, context_len: usize, vocab_size: usize) -> Result<LmTokens> {
    if vocab_size == 0 {
        let (vocab_vec, tokens) = tokenize_corpus(corpus, context_len)?;
        let class_names = vocab_vec.iter().map(|&ch| char_to_hex(ch)).collect();
        return Ok(LmTokens {
            vocab_size: vocab_vec.len(),
            tokens,
            class_names,
            tokenizer_state: String::new(),
        });
    }
    let tok = ByteBpeTokenizer::train(corpus, vocab_size)?;
    let tokens = tok.encode(corpus);
    if tokens.len() < context_len + 1 {
        return Err(InferError::DimMismatch(
            "BPE-tokenized corpus must be longer than the context window".into(),
        ));
    }
    vprintln!("[slm::tokenize_for_lm] BPE: target vocab={}, actual vocab={}, {} tokens",
        vocab_size, tok.vocab_size(), tokens.len());
    Ok(LmTokens {
        vocab_size: tok.vocab_size(),
        tokens,
        class_names: Vec::new(),
        tokenizer_state: tok.encode_state(),
    })
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
