//! Generic offline Edge Generative SLM Library Engine.
use crate::csv::{CsvDataset, ModelMetadata, Normalizer, TaskType};
use crate::error::{InferError, Result};
use crate::layer::{Embedding, KvCache, TransformerBlock};
use crate::loader::{from_bytes, to_bytes};
use crate::model::Sequential;
use crate::optim::{Adam, Sgd};
use crate::rng::Rng;
use crate::tensor::Tensor;
use crate::tokenizer::ByteBpeTokenizer;
use crate::train::{train_epoch, Net};
use crate::train_transformer::{train_transformer_epoch_threaded, TransformerNet};
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
    /// Decoupled (AdamW) weight-decay coefficient applied to weight matrices
    /// (T7). `0.0` disables it; typical values are 0.01–0.1.
    pub weight_decay: f32,
    /// FFN-hidden dropout probability used during training (T7), in `[0, 1)`.
    /// `0.0` disables it; inference is always dropout-free.
    pub dropout: f32,
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
            weight_decay: 0.0,
            dropout: 0.0,
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

/// Decoding controls for generation (I2). Beyond temperature, these add the
/// standard knobs that tame the repetition loops a temperature-only sampler
/// falls into at low temperature:
///
/// - `top_k`: keep only the `k` highest-probability tokens (`0` disables).
/// - `top_p`: nucleus sampling — keep the smallest set of tokens whose
///   cumulative probability reaches `top_p` (`1.0` disables).
/// - `repetition_penalty`: down-weight tokens already present in the recent
///   context (`1.0` disables; typical values 1.05–1.3).
///
/// [`SamplingParams::with_temperature`] reproduces the previous temperature-only
/// behaviour exactly (all other knobs disabled), so `generate(.., temp, ..)` is
/// unchanged.
#[derive(Clone, Debug, PartialEq)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
        }
    }
}

impl SamplingParams {
    /// Temperature-only sampling (top-k / top-p / repetition penalty disabled) —
    /// identical to the pre-I2 sampler.
    pub fn with_temperature(temperature: f32) -> Self {
        Self {
            temperature,
            ..Self::default()
        }
    }
}

/// Validation-aware training controls (T5): an internal held-out split, early
/// stopping on validation perplexity, and best-epoch checkpointing.
#[derive(Clone, Debug)]
pub struct ValidationConfig {
    /// Fraction of the corpus (in `0.0..1.0`) held out at the **tail** for
    /// validation. The tokenizer/vocabulary is fit on the training portion only.
    pub val_fraction: f32,
    /// Early stopping: stop once validation cross-entropy fails to improve for
    /// this many consecutive epochs. `0` disables early stopping (all epochs run)
    /// but the best-by-validation checkpoint is still returned.
    pub patience: usize,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            val_fraction: 0.1,
            patience: 0,
        }
    }
}

/// Per-epoch report from validation-aware training (T5).
#[derive(Clone, Debug)]
pub struct ValidationProgress {
    pub epoch: usize,
    pub train_loss: f32,
    /// Held-out evaluation for this epoch.
    pub val: Evaluation,
    /// Whether this epoch set a new best validation cross-entropy (and is the
    /// checkpoint that would be returned if training stopped now).
    pub is_best: bool,
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
        vprintln!(
            "[slm::GenerativeSLM::new] Creating SLM with {} layers, input_dim={}, output_dim={}",
            model.len(),
            meta.input_dim,
            meta.output_dim
        );
        Self { model, norm, meta }
    }

    /// Train a hand-crafted edge Generative SLM (MLP Causal model) on any customized raw text corpus.
    #[allow(clippy::too_many_arguments)] // stable public training API; kept flat rather than a config struct
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
    #[allow(clippy::too_many_arguments)] // stable public training API; kept flat rather than a config struct
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
        vprintln!(
            "[slm::GenerativeSLM::train_with_callback] ═══════════════════════════════════════"
        );
        vprintln!("[slm::GenerativeSLM::train_with_callback] Starting SLM training:");
        vprintln!(
            "[slm::GenerativeSLM::train_with_callback]   corpus length:  {} chars",
            corpus.len()
        );
        vprintln!(
            "[slm::GenerativeSLM::train_with_callback]   context_len:    {}",
            context_len
        );
        vprintln!(
            "[slm::GenerativeSLM::train_with_callback]   hidden_size:    {}",
            hidden_size
        );
        vprintln!(
            "[slm::GenerativeSLM::train_with_callback]   epochs:         {}",
            epochs
        );
        vprintln!(
            "[slm::GenerativeSLM::train_with_callback]   lr:             {}",
            lr
        );
        vprintln!(
            "[slm::GenerativeSLM::train_with_callback]   momentum:       {}",
            momentum
        );
        vprintln!(
            "[slm::GenerativeSLM::train_with_callback]   batch_size:     {}",
            batch_size
        );
        vprintln!(
            "[slm::GenerativeSLM::train_with_callback] ═══════════════════════════════════════"
        );

        vprintln!("[slm::train] Building CSV dataset from corpus...");
        let csv_build_start = std::time::Instant::now();
        let csv_data = build_csv_dataset(corpus, context_len)?;
        vprintln!(
            "[slm::train] CSV dataset built in {:.1}ms, size={} bytes",
            csv_build_start.elapsed().as_secs_f64() * 1000.0,
            csv_data.len()
        );

        vprintln!("[slm::train] Parsing CSV dataset...");
        let parse_start = std::time::Instant::now();
        // Register the full sorted vocabulary explicitly so class indices
        // cover every character (even ones never appearing as a target) in
        // exact sorted order — no padding rows needed.
        let class_names: Vec<String> = corpus_vocab(corpus)
            .iter()
            .map(|&ch| char_to_hex(ch))
            .collect();
        let ds = CsvDataset::from_str_with_classes(&csv_data, &class_names)?;
        vprintln!(
            "[slm::train] Parsed in {:.1}ms: rows={}, features={}, classes={}",
            parse_start.elapsed().as_secs_f64() * 1000.0,
            ds.len(),
            ds.num_features,
            ds.num_classes
        );

        vprintln!("[slm::train] Converting to tensors...");
        let (x_raw, y_cls, _) = ds.to_tensors()?;
        vprintln!(
            "[slm::train] Tensor shapes: x={:?}, y_len={}",
            x_raw.shape,
            y_cls.len()
        );

        vprintln!("[slm::train] Fitting normalizer (identity for SLM)...");
        let mut norm = Normalizer::fit(&x_raw)?;
        for m in &mut norm.means {
            *m = 0.0;
        }
        for s in &mut norm.stds {
            *s = 1.0;
        }
        let x_train = norm.transform(&x_raw)?;
        vprintln!(
            "[slm::train] Normalizer applied, x_train shape={:?}",
            x_train.shape
        );

        vprintln!("[slm::train] Creating trainable MLP (QAT enabled)...");
        let mut net = Net::mlp(ds.num_features, hidden_size, ds.num_classes, rng);
        net.set_qat(true);
        let opt = Sgd::with_momentum(lr, momentum);
        vprintln!(
            "[slm::train] Network: {} params, optimizer: lr={}, momentum={}",
            net.num_params(),
            lr,
            momentum
        );

        vprintln!(
            "[slm::train] ── Beginning training loop ({} epochs) ──",
            epochs
        );
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

            vprintln!(
                "[slm::train] Epoch {}/{}: loss={:.6}, time={:.1}ms, ETA={:.1}s",
                ep,
                epochs,
                loss,
                ep_ms,
                eta_secs
            );

            if verbose::is_verbose() {
                if loss.is_nan() {
                    crate::verbose::log_line(&format!(
                        "[ferrum_core::WARN] ⚠️  NaN loss at epoch {}! Training is diverging!",
                        ep
                    ));
                }
                if loss.is_infinite() {
                    crate::verbose::log_line(&format!(
                        "[ferrum_core::WARN] ⚠️  Infinite loss at epoch {}! Training is diverging!",
                        ep
                    ));
                }
                if loss > 1e6 {
                    crate::verbose::log_line(&format!("[ferrum_core::WARN] ⚠️  Very large loss ({:.2}) at epoch {} — possible explosion!", loss, ep));
                }
            }

            progress_callback(ep, loss);
        }

        let total_train_time = train_start.elapsed().as_secs_f64();
        vprintln!(
            "[slm::train] ── Training complete in {:.2}s ──",
            total_train_time
        );

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
        vprintln!(
            "[slm::train] Metadata: input_dim={}, output_dim={}, vocab={}",
            meta.input_dim,
            meta.output_dim,
            meta.class_names.len()
        );

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
            corpus,
            context_len,
            embed_dim,
            num_heads,
            num_blocks,
            hidden_dim,
            epochs,
            lr,
            batch_size,
            vocab_size,
            rng,
            |_, _| {},
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
            corpus,
            context_len,
            embed_dim,
            num_heads,
            num_blocks,
            hidden_dim,
            epochs,
            lr,
            batch_size,
            vocab_size,
            0.0,
            0.0,
            1,
            rng,
            progress_callback,
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
        let threads = if threads == 0 {
            crate::parallel::num_threads()
        } else {
            threads
        };
        Self::train_transformer_inner(
            corpus,
            context_len,
            embed_dim,
            num_heads,
            num_blocks,
            hidden_dim,
            epochs,
            lr,
            batch_size,
            vocab_size,
            0.0,
            0.0,
            threads,
            rng,
            progress_callback,
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
        weight_decay: f32,
        dropout: f32,
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
            model_vocab,
            context_len,
            embed_dim,
            num_heads,
            hidden_dim,
            num_blocks,
            rng,
        )?;
        net.set_qat(true);
        net.set_weight_decay(weight_decay);
        net.set_dropout(dropout);
        vprintln!("[slm::train_transformer] {} params, Adam lr={}, wd={}, dropout={}, QAT=int8, threads={}",
            net.num_params(), lr, weight_decay, dropout, threads);

        let adam = Adam::new(lr);
        for ep in 1..=epochs {
            let loss = train_transformer_epoch_threaded(
                &mut net, &tc.tokens, batch_size, &adam, rng, threads,
            )?;
            vprintln!(
                "[slm::train_transformer] epoch {}/{}: loss={:.6}",
                ep,
                epochs,
                loss
            );
            progress_callback(ep, loss);
        }

        Self::build_transformer_slm(&net, &tc, context_len)
    }

    /// Construct an inference [`GenerativeSLM`] from a trained transformer net
    /// and its tokenization (shared by the plain and validation-aware trainers).
    fn build_transformer_slm(
        net: &TransformerNet,
        tc: &LmTokens,
        context_len: usize,
    ) -> Result<Self> {
        let model_vocab = tc.vocab_size;
        let model = net.to_inference()?;
        let meta = ModelMetadata {
            dataset_name: "GenerativeSLM Transformer".into(),
            task: TaskType::TransformerSLM,
            feature_names: (0..context_len).map(|i| format!("c_{i}")).collect(),
            feature_ranges: vec![[0.0, model_vocab as f32]; context_len],
            class_names: tc.class_names.clone(),
            target_name: "next_char".into(),
            target_range: [0.0, model_vocab as f32],
            input_dim: context_len,
            output_dim: model_vocab,
            tokenizer_state: tc.tokenizer_state.clone(),
        };
        let norm = Normalizer {
            means: vec![],
            stds: vec![],
        };
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
            corpus,
            context_len,
            embed_dim,
            hidden_size,
            epochs,
            lr,
            momentum,
            batch_size,
            vocab_size,
            rng,
            |_, _| {},
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
            model_vocab,
            context_len,
            embed_dim,
            hidden_size,
            model_vocab,
            rng,
        );
        net.set_qat(true);
        let opt = Sgd::with_momentum(lr, momentum);
        vprintln!(
            "[slm::train_embedded] {} params, SGD lr={}, momentum={}",
            net.num_params(),
            lr,
            momentum
        );

        for ep in 1..=epochs {
            let loss = train_epoch(&mut net, &x, &y, batch_size, &opt, rng)?;
            vprintln!(
                "[slm::train_embedded] epoch {}/{}: loss={:.6}",
                ep,
                epochs,
                loss
            );
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
        let norm = Normalizer {
            means: vec![],
            stds: vec![],
        };
        Ok(Self { model, norm, meta })
    }

    /// Serialize the trained Generative SLM model to self-contained FINF v4 bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        vprintln!("[slm::GenerativeSLM::to_bytes] Serializing model...");
        let bytes = to_bytes(&self.model, &self.norm, &self.meta)?;
        vprintln!(
            "[slm::GenerativeSLM::to_bytes] Serialized to {} bytes",
            bytes.len()
        );
        Ok(bytes)
    }

    /// Serialize to FINF v5 with int8-quantised weights (≈4× smaller files).
    pub fn to_bytes_quantized(&self) -> Result<Vec<u8>> {
        vprintln!("[slm::GenerativeSLM::to_bytes_quantized] Serializing quantized model...");
        let bytes = crate::loader::to_bytes_quantized(&self.model, &self.norm, &self.meta)?;
        vprintln!(
            "[slm::GenerativeSLM::to_bytes_quantized] Serialized to {} bytes",
            bytes.len()
        );
        Ok(bytes)
    }

    /// Serialize to FINF v5 with **int4** weights (~8× smaller than f32). The
    /// recommended on-disk format for large (≥1B) models: the loader keeps the
    /// weight matrices packed in memory (int4), so a model both *fits* and
    /// *streams ⅛ the bytes per token* on the decode hot path.
    pub fn to_bytes_quantized_int4(&self) -> Result<Vec<u8>> {
        vprintln!("[slm::GenerativeSLM::to_bytes_quantized_int4] Serializing int4 model...");
        let bytes = crate::loader::to_bytes_quantized_int4(&self.model, &self.norm, &self.meta)?;
        vprintln!(
            "[slm::GenerativeSLM::to_bytes_quantized_int4] Serialized to {} bytes",
            bytes.len()
        );
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
        Self::train_transformer_inner(
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
            cfg.weight_decay,
            cfg.dropout,
            1,
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
        let threads = if threads == 0 {
            crate::parallel::num_threads()
        } else {
            threads
        };
        Self::train_transformer_inner(
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
            cfg.weight_decay,
            cfg.dropout,
            threads,
            rng,
            progress_callback,
        )
    }

    /// Train a transformer SLM with an internal validation split, early stopping
    /// on validation perplexity, and best-epoch checkpointing (T5).
    ///
    /// The corpus is split by character count: the final `val.val_fraction` is
    /// held out for validation and the rest is used for training (the
    /// tokenizer/vocabulary is fit on the training portion only, as it must be).
    /// After every epoch the in-progress model is scored on the held-out text
    /// with [`GenerativeSLM::evaluate`]; the weights with the **lowest validation
    /// cross-entropy** are retained and returned — not necessarily the final
    /// epoch's. When `val.patience > 0`, training stops early once validation
    /// fails to improve for that many consecutive epochs.
    ///
    /// `threads` follows the same convention as
    /// [`GenerativeSLM::train_transformer_config_threaded`] (`0` = auto, `1` =
    /// serial). The callback receives a [`ValidationProgress`] each epoch.
    pub fn train_transformer_config_validated<F>(
        corpus: &str,
        cfg: &TransformerConfig,
        threads: usize,
        val: &ValidationConfig,
        rng: &mut Rng,
        mut progress_callback: F,
    ) -> Result<Self>
    where
        F: FnMut(&ValidationProgress),
    {
        if !(val.val_fraction > 0.0 && val.val_fraction < 1.0) {
            return Err(InferError::DimMismatch(
                "val_fraction must be in the open interval (0, 1)".into(),
            ));
        }
        let threads = if threads == 0 {
            crate::parallel::num_threads()
        } else {
            threads
        };

        // Split the corpus by character: head = train, tail = validation.
        let chars: Vec<char> = corpus.chars().collect();
        let total = chars.len();
        let val_chars = ((total as f32) * val.val_fraction).round() as usize;
        let val_chars = val_chars.clamp(1, total.saturating_sub(1));
        let split = total - val_chars;
        let train_text: String = chars[..split].iter().collect();
        let val_text: String = chars[split..].iter().collect();

        let tc = tokenize_for_lm(&train_text, cfg.context_len, cfg.vocab_size)?;
        let model_vocab = tc.vocab_size;
        vprintln!(
            "[slm::train_validated] train={} chars, val={} chars, vocab={}, patience={}",
            train_text.chars().count(),
            val_text.chars().count(),
            model_vocab,
            val.patience
        );

        let mut net = TransformerNet::new(
            model_vocab,
            cfg.context_len,
            cfg.embed_dim,
            cfg.num_heads,
            cfg.hidden_dim,
            cfg.num_blocks,
            rng,
        )?;
        net.set_qat(true);
        net.set_weight_decay(cfg.weight_decay);
        net.set_dropout(cfg.dropout);
        let adam = Adam::new(cfg.lr);

        let mut best: Option<Self> = None;
        let mut best_ce = f32::INFINITY;
        let mut stale = 0usize;

        for ep in 1..=cfg.epochs {
            let train_loss = train_transformer_epoch_threaded(
                &mut net,
                &tc.tokens,
                cfg.batch_size,
                &adam,
                rng,
                threads,
            )?;

            // Score the in-progress model on the held-out split.
            let candidate = Self::build_transformer_slm(&net, &tc, cfg.context_len)?;
            let eval = candidate.evaluate(&val_text)?;

            let is_best = eval.cross_entropy < best_ce;
            if is_best {
                best_ce = eval.cross_entropy;
                best = Some(candidate);
                stale = 0;
            } else {
                stale += 1;
            }

            progress_callback(&ValidationProgress {
                epoch: ep,
                train_loss,
                val: eval,
                is_best,
            });

            if val.patience > 0 && stale >= val.patience {
                vprintln!("[slm::train_validated] early stop at epoch {ep} (no val improvement for {} epochs)", stale);
                break;
            }
        }

        // `best` is always set: epoch 1 improves on +inf.
        best.ok_or_else(|| InferError::DimMismatch("training ran zero epochs".into()))
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
        vprintln!(
            "[slm::GenerativeSLM::save] Wrote {} bytes → {}",
            bytes.len(),
            model_path
        );
        Ok(())
    }

    /// Save as **int4** FINF v5 (~8× smaller than f32; the matrices load back
    /// packed in memory). Recommended for large or GGUF-imported models. Note:
    /// for the small QAT-trained SLMs produced here, weights are int8-snapped
    /// during training, so [`GenerativeSLM::save`] (int8) is bit-faithful to the
    /// in-memory model whereas int4 adds extra quantization drift — int4 pays off
    /// at scale, where fitting in RAM and streaming fewer bytes per token matter
    /// more than the last bit of small-model accuracy.
    pub fn save_int4(&self, model_path: &str) -> Result<()> {
        let bytes = self.to_bytes_quantized_int4()?;
        if let Some(parent) = std::path::Path::new(model_path).parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        std::fs::write(model_path, &bytes)?;
        vprintln!(
            "[slm::GenerativeSLM::save_int4] Wrote {} bytes → {}",
            bytes.len(),
            model_path
        );
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
            vprintln!(
                "[slm::load_or_train] Found {} — loading instead of retraining",
                model_path
            );
            return Ok((Self::load(model_path)?, true));
        }
        let slm = Self::train_transformer_config(corpus, cfg, rng, progress_callback)?;
        slm.save(model_path)?;
        Ok((slm, false))
    }

    /// Load a trained Generative SLM model from self-contained FINF v4 bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        vprintln!(
            "[slm::GenerativeSLM::from_bytes] Deserializing from {} bytes...",
            bytes.len()
        );
        let (model, norm, meta) = from_bytes(bytes)?;
        vprintln!(
            "[slm::GenerativeSLM::from_bytes] Loaded: {} layers, input_dim={}, output_dim={}",
            model.len(),
            meta.input_dim,
            meta.output_dim
        );
        Ok(Self { model, norm, meta })
    }

    /// Autoregressively generate next-character sequence completions from a seed text.
    ///
    /// For BPE models (`meta.tokenizer_state` non-empty) generation runs over
    /// subword tokens but `num_chars` still counts **characters**: the output is
    /// `seed` followed by exactly `num_chars` newly generated characters
    /// (fewer only if generation is cut short). Character-level models keep the
    /// original per-character behaviour.
    pub fn generate(
        &self,
        seed: &str,
        num_chars: usize,
        temp: f32,
        rng: &mut Rng,
    ) -> Result<String> {
        // The full-string API is the streaming API with a no-op sink.
        self.generate_stream(seed, num_chars, temp, rng, |_| {})
    }

    /// Like [`GenerativeSLM::generate`] but with full decoding control (I2):
    /// temperature plus top-k, top-p (nucleus), and repetition penalty. See
    /// [`SamplingParams`].
    pub fn generate_with(
        &self,
        seed: &str,
        num_chars: usize,
        params: &SamplingParams,
        rng: &mut Rng,
    ) -> Result<String> {
        self.generate_stream_with(seed, num_chars, params, rng, |_| {})
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
        self.generate_stream_with(
            seed,
            num_chars,
            &SamplingParams::with_temperature(temp),
            rng,
            on_text,
        )
    }

    /// Streaming generation with full decoding control (I2) — the implementation
    /// behind [`GenerativeSLM::generate_stream`] (which passes a
    /// temperature-only [`SamplingParams`]). See [`SamplingParams`] for the
    /// top-k / top-p / repetition-penalty knobs.
    pub fn generate_stream_with<F>(
        &self,
        seed: &str,
        num_chars: usize,
        params: &SamplingParams,
        rng: &mut Rng,
        on_text: F,
    ) -> Result<String>
    where
        F: FnMut(&str),
    {
        self.generate_stream_core(seed, num_chars, params, None, rng, on_text)
    }

    /// [`GenerativeSLM::generate_with`] that **stops early** at a natural
    /// boundary (I3): generation halts as soon as the `stop` string appears in
    /// the generated continuation (and the result ends right after it), or after
    /// `max_chars` characters — whichever comes first. Useful with a sentence
    /// terminator or a document separator (see the tokenizer's special tokens,
    /// K3). An empty `stop` disables early stopping.
    pub fn generate_until(
        &self,
        seed: &str,
        max_chars: usize,
        stop: &str,
        params: &SamplingParams,
        rng: &mut Rng,
    ) -> Result<String> {
        self.generate_stream_until(seed, max_chars, stop, params, rng, |_| {})
    }

    /// Streaming counterpart of [`GenerativeSLM::generate_until`] (I3).
    pub fn generate_stream_until<F>(
        &self,
        seed: &str,
        max_chars: usize,
        stop: &str,
        params: &SamplingParams,
        rng: &mut Rng,
        on_text: F,
    ) -> Result<String>
    where
        F: FnMut(&str),
    {
        let stop = if stop.is_empty() { None } else { Some(stop) };
        self.generate_stream_core(seed, max_chars, params, stop, rng, on_text)
    }

    /// Core streaming generation shared by the plain and stop-criterion APIs.
    /// When `stop` is `Some`, generation halts once the stop string appears in
    /// the continuation (I3).
    fn generate_stream_core<F>(
        &self,
        seed: &str,
        num_chars: usize,
        params: &SamplingParams,
        stop: Option<&str>,
        rng: &mut Rng,
        on_text: F,
    ) -> Result<String>
    where
        F: FnMut(&str),
    {
        vprintln!("[slm::GenerativeSLM::generate_stream] seed=\"{}\", num_chars={}, temp={:.2}, top_k={}, top_p={:.2}, rep={:.2}, stop={:?}",
            seed.chars().take(50).collect::<String>(), num_chars,
            params.temperature, params.top_k, params.top_p, params.repetition_penalty, stop);

        if !self.meta.tokenizer_state.is_empty() {
            return self.generate_bpe_stream(seed, num_chars, params, stop, rng, on_text);
        }

        let vocab_size = self.meta.output_dim;
        let input_dim = self.meta.input_dim;
        // Transformer models take context_len token IDs; the MLP takes a
        // flattened one-hot context of context_len × vocab_size values.
        let is_transformer = self.meta.task == TaskType::TransformerSLM;
        let context_len = if is_transformer {
            input_dim
        } else {
            input_dim / vocab_size
        };

        vprintln!(
            "[slm::generate] vocab_size={}, input_dim={}, context_len={}, transformer={}",
            vocab_size,
            input_dim,
            context_len,
            is_transformer
        );

        // Fast path: genuine transformer models generate token-at-a-time with a
        // per-block KV cache (O(context) per token). Models without transformer
        // blocks (the embedded-MLP family) fall through to the full-forward loop.
        if is_transformer {
            if let Some(cached) = CachedTransformer::try_new(&self.model) {
                return self.generate_char_cached_stream(
                    seed,
                    num_chars,
                    params,
                    stop,
                    rng,
                    on_text,
                    cached,
                    context_len,
                );
            }
        }

        let mut on_text = on_text;
        let mut generated = seed.to_string();
        let seed_bytes = seed.len();

        for step in 0..num_chars {
            let current_len = generated.chars().count();
            if current_len < context_len {
                vprintln!(
                    "[slm::generate] Step {}: generated length {} < context_len {}, stopping",
                    step,
                    current_len,
                    context_len
                );
                break;
            }
            let context_chars: Vec<char> =
                generated.chars().skip(current_len - context_len).collect();

            vprintln!(
                "[slm::generate] Step {}: context=\"{}\"",
                step,
                context_chars.iter().collect::<String>()
            );

            let char_idx = |ch: char| -> usize {
                let hex = char_to_hex(ch);
                self.meta
                    .class_names
                    .iter()
                    .position(|s| s == &hex)
                    .unwrap_or(0)
            };

            let next_dist: Vec<f32> = if is_transformer {
                // Token-ID input → [T, vocab] probabilities; keep the last row.
                let ids: Vec<f32> = context_chars
                    .iter()
                    .map(|&ch| char_idx(ch) as f32)
                    .collect();
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
                vprintln!(
                    "[slm::generate] Step {}: logits stats: min={:.4}, max={:.4}, mean={:.4}",
                    step,
                    lmin,
                    lmax,
                    lmean
                );
            }

            // Sample with the full decoding controls; the repetition penalty
            // sees the tokens currently in the context window.
            let recent: Vec<usize> = context_chars.iter().map(|&ch| char_idx(ch)).collect();
            let next_idx = sample_with_params(&next_dist, params, &recent, rng);

            // Decode prediction
            let predicted_hex = &self.meta.class_names[next_idx];
            let next_char = hex_to_char(predicted_hex);

            vprintln!(
                "[slm::generate] Step {}: sampled idx={}, hex=\"{}\", char='{}'",
                step,
                next_idx,
                predicted_hex,
                next_char
            );

            generated.push(next_char);
            // Stop early at a natural boundary (I3): the newest char may have
            // completed the stop string in the continuation.
            let hit_stop = stop.is_some_and(|s| generated[seed_bytes..].ends_with(s));
            // Stream exactly the new character (never the seed).
            let mut buf = [0u8; 4];
            on_text(next_char.encode_utf8(&mut buf));
            if hit_stop {
                break;
            }
        }

        vprintln!("[slm::generate] Generated {} total chars", generated.len());
        Ok(generated)
    }

    /// KV-cached char-level generation for genuine transformer models (I1).
    ///
    /// Maintains a rolling window of the most recent ≤ `context_len` token IDs
    /// in a per-block [`KvCache`] and samples one character per step. While the
    /// window is below capacity each step is a single O(context) cached `feed`;
    /// once the window is full the cache is re-primed with the most recent
    /// `context_len` tokens before the next prediction. Because that re-prime
    /// reconstructs exactly the same `context_len`-token window at positions
    /// `0..context_len-1` that the previous (full-forward) path fed, this
    /// produces **bit-identical** output to the old path for any seed at least
    /// `context_len` characters long, while being strictly faster whenever the
    /// generated sequence fits within the context window.
    #[allow(clippy::too_many_arguments)]
    fn generate_char_cached_stream<F>(
        &self,
        seed: &str,
        num_chars: usize,
        params: &SamplingParams,
        stop: Option<&str>,
        rng: &mut Rng,
        mut on_text: F,
        mut cached: CachedTransformer,
        context_len: usize,
    ) -> Result<String>
    where
        F: FnMut(&str),
    {
        let mut generated = seed.to_string();
        let seed_bytes = seed.len();
        let cap = cached.capacity().max(1);

        let char_idx = |ch: char| -> usize {
            let hex = char_to_hex(ch);
            self.meta
                .class_names
                .iter()
                .position(|s| s == &hex)
                .unwrap_or(0)
        };

        // The window holds the token IDs currently represented in the cache.
        let mut window: Vec<usize> = seed.chars().map(char_idx).collect();
        // Preserve the old contract: a seed shorter than one context window
        // cannot start char-level generation.
        if window.len() < context_len || num_chars == 0 {
            return Ok(generated);
        }
        // Prime with the most recent `cap` tokens (positions 0..cap-1).
        if window.len() > cap {
            window.drain(..window.len() - cap);
        }
        let mut dist = cached.prime(&window)?;

        for step in 0..num_chars {
            // The model ends in Softmax; convert probabilities back to log-space
            // so temperature scaling behaves exactly as on the full-forward path.
            let logits: Vec<f32> = dist.iter().map(|&p| p.max(1e-12).ln()).collect();
            if verbose::is_verbose() {
                let (lmin, lmax, lmean) = verbose::stats(&logits);
                vprintln!("[slm::generate_cached] Step {}: logits stats: min={:.4}, max={:.4}, mean={:.4}",
                    step, lmin, lmax, lmean);
            }
            // Repetition penalty sees the tokens currently in the cache window.
            let next_idx = sample_with_params(&logits, params, &window, rng);
            let next_char = hex_to_char(&self.meta.class_names[next_idx]);
            generated.push(next_char);
            let hit_stop = stop.is_some_and(|s| generated[seed_bytes..].ends_with(s));
            let mut buf = [0u8; 4];
            on_text(next_char.encode_utf8(&mut buf));
            if hit_stop {
                break; // I3: halt at the natural boundary
            }

            // Advance the window by the freshly sampled token. If it would
            // overflow the context, slide and re-prime; otherwise extend the
            // cache incrementally with a single cheap feed.
            window.push(next_idx);
            if window.len() > cap {
                window.remove(0);
                dist = cached.prime(&window)?;
            } else {
                dist = cached.feed(next_idx)?;
            }
        }

        vprintln!(
            "[slm::generate_cached] Generated {} total chars",
            generated.chars().count()
        );
        Ok(generated)
    }

    /// Streaming BPE generation path. Rebuilds the stored [`ByteBpeTokenizer`],
    /// encodes the seed to token IDs, samples one token at a time from the model,
    /// and decodes the whole token stream back to text. Generation continues
    /// until at least `num_chars` characters have been added past the seed, then
    /// the result is trimmed to exactly that length so the character contract
    /// matches the char-level path.
    ///
    /// Genuine transformer models drive a per-block [`KvCache`] (O(context) per
    /// token, the same fast path as the char-level and WASM builds); models
    /// without transformer blocks (the embedded-MLP BPE family) fall back to a
    /// full forward over the last `context_len` tokens, left-padded with token 0
    /// when the prompt is shorter than the window.
    ///
    /// As tokens land, the newly decoded continuation characters are streamed to
    /// `on_text`, holding back the final (possibly partial) character until the
    /// next token completes it so a `U+FFFD` placeholder is never emitted and
    /// then revised.
    #[allow(clippy::too_many_arguments)]
    fn generate_bpe_stream<F>(
        &self,
        seed: &str,
        num_chars: usize,
        params: &SamplingParams,
        stop: Option<&str>,
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
        vprintln!(
            "[slm::generate_bpe] seed encoded to {} tokens, ctx={}, target_chars={}",
            ids.len(),
            context_len,
            target_chars
        );

        // Number of continuation characters already streamed (never includes the
        // seed). Emit a delta after every token so output is incremental, capped
        // at `budget` continuation chars (which tightens to the stop position).
        let mut emitted_cont = 0usize;
        let mut flush = |full: &str, hold_back_partial: bool, budget: usize, on_text: &mut F| {
            let cont_chars = full.chars().count().saturating_sub(seed_chars);
            let avail = if hold_back_partial {
                cont_chars.saturating_sub(1).min(budget)
            } else {
                cont_chars.min(budget)
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

        // Continuation-char count up to and including the first `stop` match
        // (I3), if present in the decoded continuation.
        let stop_cut = |full: &str| -> Option<usize> {
            let s = stop?;
            let cont: String = full.chars().skip(seed_chars).collect();
            cont.find(s).map(|b| cont[..b + s.len()].chars().count())
        };

        // Bound the loop: every step adds ≥1 byte, but guard against a model
        // that keeps emitting zero-width or repeated control tokens.
        let max_steps = num_chars * 8 + 64;
        // The continuation-char budget that finally bounds the output: `num_chars`
        // normally, tightened to the stop position when one is hit.
        let mut budget = num_chars;

        if let Some(mut cached) = CachedTransformer::try_new(&self.model) {
            // Fast path: KV-cached incremental decoding. A rolling window of the
            // most recent ≤ capacity tokens lives in the cache; each step is a
            // single O(context) `feed` until the window fills, then a re-prime.
            let cap = cached.capacity().max(1);
            let mut window: Vec<usize> = if ids.is_empty() {
                vec![0]
            } else {
                ids[ids.len().saturating_sub(cap)..].to_vec()
            };
            let mut dist = cached.prime(&window)?;

            let mut steps = 0;
            while tok.decode(&ids).chars().count() < target_chars && steps < max_steps {
                let logits: Vec<f32> = dist.iter().map(|&p| p.max(1e-12).ln()).collect();
                let next = sample_with_params(&logits, params, &window, rng);
                ids.push(next);
                steps += 1;
                let full = tok.decode(&ids);
                // I3: stop at a natural boundary if the stop string appeared.
                if let Some(cut) = stop_cut(&full) {
                    budget = cut.min(num_chars);
                    break;
                }
                flush(&full, true, budget, &mut on_text);

                window.push(next);
                if window.len() > cap {
                    window.remove(0);
                    dist = cached.prime(&window)?;
                } else {
                    dist = cached.feed(next)?;
                }
            }
        } else {
            // Fallback (no transformer blocks): full forward over the last
            // `context_len` tokens, left-padded with token 0 when too short.
            let mut steps = 0;
            while tok.decode(&ids).chars().count() < target_chars && steps < max_steps {
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
                let logits: Vec<f32> = out.data[(rows - 1) * cols..]
                    .iter()
                    .map(|&p| p.max(1e-12).ln())
                    .collect();
                let recent_start = ids.len().saturating_sub(context_len);
                let recent = ids[recent_start..].to_vec();
                let next = sample_with_params(&logits, params, &recent, rng);
                ids.push(next);
                steps += 1;
                let full = tok.decode(&ids);
                if let Some(cut) = stop_cut(&full) {
                    budget = cut.min(num_chars);
                    break;
                }
                flush(&full, true, budget, &mut on_text);
            }
        }

        // Decode the full token stream, emit any remaining continuation chars up
        // to the final budget, and trim to exactly that many.
        let full = tok.decode(&ids);
        flush(&full, false, budget, &mut on_text);
        let trimmed: String = full.chars().take(seed_chars + budget).collect();
        vprintln!(
            "[slm::generate_bpe] generated {} tokens → {} chars (trimmed to {})",
            ids.len(),
            full.chars().count(),
            trimmed.chars().count()
        );
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

    /// [`GenerativeSLM::generate_continuation`] with full decoding control (I2):
    /// returns only the newly generated text (seed prefix removed).
    pub fn generate_continuation_with(
        &self,
        seed: &str,
        num_chars: usize,
        params: &SamplingParams,
        rng: &mut Rng,
    ) -> Result<String> {
        let full = self.generate_with(seed, num_chars, params, rng)?;
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
        vprintln!(
            "[slm::evaluate] scoring {} chars of held-out text",
            text.chars().count()
        );

        if !self.meta.tokenizer_state.is_empty() {
            let tok = ByteBpeTokenizer::from_state(&self.meta.tokenizer_state)?;
            let ids = tok.encode(text);
            return self.score_token_ids(&ids, self.meta.input_dim);
        }

        let vocab_size = self.meta.output_dim;
        let input_dim = self.meta.input_dim;
        let is_transformer = self.meta.task == TaskType::TransformerSLM;
        let context_len = if is_transformer {
            input_dim
        } else {
            input_dim / vocab_size
        };

        let chars: Vec<char> = text.chars().filter(|&c| c != '\r').collect();
        let ids: Vec<usize> = chars
            .iter()
            .map(|&ch| {
                let hex = char_to_hex(ch);
                self.meta
                    .class_names
                    .iter()
                    .position(|s| s == &hex)
                    .unwrap_or(0)
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
    fn score_onehot(
        &self,
        ids: &[usize],
        context_len: usize,
        vocab_size: usize,
    ) -> Result<Evaluation> {
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
    let ce = if count == 0 {
        0.0
    } else {
        (total_nll / count as f64) as f32
    };
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
    vprintln!(
        "[slm::tokenize_for_lm] BPE: target vocab={}, actual vocab={}, {} tokens",
        vocab_size,
        tok.vocab_size(),
        tokens.len()
    );
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
    vprintln!(
        "[slm::build_csv_dataset] corpus_len={}, context_len={}",
        corpus.len(),
        context_len
    );

    let chars: Vec<char> = corpus.chars().filter(|&c| c != '\r').collect();
    if chars.len() < context_len {
        return Err(InferError::DimMismatch(
            "Corpus length shorter than context window".into(),
        ));
    }

    let vocab_vec = corpus_vocab(corpus);
    let v_size = vocab_vec.len();

    vprintln!(
        "[slm::build_csv_dataset] chars={}, vocab_size={}, sliding_windows={}",
        chars.len(),
        v_size,
        chars.len().saturating_sub(context_len)
    );
    vprintln!(
        "[slm::build_csv_dataset] input_dim={} (context_len × vocab_size)",
        context_len * v_size
    );

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
    vprintln!(
        "[slm::build_csv_dataset] Writing {} sliding window rows",
        window_count
    );
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

    vprintln!(
        "[slm::build_csv_dataset] CSV built: {} bytes, {} total rows",
        csv.len(),
        window_count
    );

    Ok(csv)
}

// ─────────────────────────────────────────────────────────────────────────────
// KV-cached incremental transformer driver (I1)
// ─────────────────────────────────────────────────────────────────────────────

/// Incremental, KV-cached driver over an inference `Sequential` whose layers are
/// `Embedding → TransformerBlock × N → LayerNorm → Linear → Softmax`.
///
/// Feeding tokens one at a time with a per-block [`KvCache`] makes each new-token
/// step O(context) instead of the O(context²) of re-running a full forward over
/// the whole window every token (which `generate` did before). This mirrors the
/// WASM `TransformerSLMModel` path so native (CLI/GUI) generation runs on the
/// same fast inference engine as the browser build.
///
/// [`CachedTransformer::try_new`] returns `None` for models without
/// `TransformerBlock`s (e.g. the embedded-MLP family), so callers transparently
/// fall back to the full-forward loop.
struct CachedTransformer<'a> {
    model: &'a Sequential,
    /// One cache per `TransformerBlock`, in layer order.
    caches: Vec<KvCache>,
    /// Absolute sequence position of the next token to be fed.
    pos: usize,
    /// Block context length = cache capacity = positional-table size.
    capacity: usize,
}

impl<'a> CachedTransformer<'a> {
    /// Build a driver if `model` contains at least one `TransformerBlock`.
    fn try_new(model: &'a Sequential) -> Option<Self> {
        let mut caches = Vec::new();
        let mut capacity = 0;
        for layer in model.layers() {
            if let Some(tb) = layer.as_any().downcast_ref::<TransformerBlock>() {
                capacity = tb.context_len();
                caches.push(KvCache::new(tb.context_len(), tb.embedding_dim()));
            }
        }
        if caches.is_empty() {
            None
        } else {
            Some(Self {
                model,
                caches,
                pos: 0,
                capacity,
            })
        }
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    /// Drop all cached positions (start a fresh window at position 0).
    fn reset(&mut self) {
        for c in &mut self.caches {
            c.clear();
        }
        self.pos = 0;
    }

    /// Feed one token through the cached path, returning the model's final
    /// softmax row — the next-token probability distribution. Walks the layer
    /// list exactly like a full forward, but routes the embedding through
    /// `embed_one(token, pos)` and each block through `forward_with_cache`.
    fn feed(&mut self, token: usize) -> Result<Vec<f32>> {
        let mut x: Option<Tensor> = None;
        let mut block_idx = 0usize;
        for layer in self.model.layers() {
            let any = layer.as_any();
            if let Some(emb) = any.downcast_ref::<Embedding>() {
                x = Some(emb.embed_one(token, self.pos)?);
            } else if let Some(tb) = any.downcast_ref::<TransformerBlock>() {
                let cur = x.ok_or_else(|| {
                    InferError::DimMismatch("Embedding must precede TransformerBlock".into())
                })?;
                x = Some(tb.forward_with_cache(&cur, &mut self.caches[block_idx])?);
                block_idx += 1;
            } else {
                let cur = x.ok_or_else(|| {
                    InferError::DimMismatch("model must start with an Embedding layer".into())
                })?;
                x = Some(layer.forward(&cur)?);
            }
        }
        self.pos += 1;
        x.map(|t| t.data)
            .ok_or_else(|| InferError::DimMismatch("empty model".into()))
    }

    /// Reset and feed `ids` (positions 0..ids.len()), returning the
    /// distribution after the last token. `ids.len()` must be ≤ `capacity`.
    fn prime(&mut self, ids: &[usize]) -> Result<Vec<f32>> {
        self.reset();
        let mut dist = Vec::new();
        for &t in ids {
            dist = self.feed(t)?;
        }
        Ok(dist)
    }
}

/// Core logits sampler (I2): repetition penalty → temperature softmax → top-k →
/// top-p (nucleus) → renormalise → sample.
///
/// `recent` lists token IDs in the current context; the repetition penalty
/// down-weights their logits (dividing positive logits / multiplying negative
/// ones by `repetition_penalty`, the standard CTRL scheme). With
/// `SamplingParams::with_temperature` (top_k=0, top_p=1, penalty=1, recent
/// ignored) this draws exactly one `rng.next_f32()` from the temperature softmax
/// — bit-identical to the pre-I2 sampler.
pub(crate) fn sample_with_params(
    logits: &[f32],
    params: &SamplingParams,
    recent: &[usize],
    rng: &mut Rng,
) -> usize {
    let n = logits.len();
    let t = params.temperature.max(0.01);

    // 1. Repetition penalty on the raw logits of already-seen tokens.
    let mut work: Vec<f32> = logits.to_vec();
    if (params.repetition_penalty - 1.0).abs() > f32::EPSILON && params.repetition_penalty > 0.0 {
        for &tok in recent {
            if let Some(l) = work.get_mut(tok) {
                *l = if *l > 0.0 {
                    *l / params.repetition_penalty
                } else {
                    *l * params.repetition_penalty
                };
            }
        }
    }

    // 2. Temperature softmax (numerically stable).
    let max = work.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = work.iter().map(|&l| ((l - max) / t).exp()).collect();
    let sum: f32 = probs.iter().sum();
    // `sum <= threshold || NaN` — a degenerate/non-finite softmax mass triggers
    // the random fallback (equivalent to the previous `!(sum > 1e-10)`).
    if sum.is_nan() || sum <= 1e-10 {
        let fallback = rng.next_u64() as usize % n;
        vprintln!(
            "[slm::sample] ⚠️  Near-zero softmax sum, random fallback idx={}",
            fallback
        );
        return fallback;
    }
    for p in &mut probs {
        *p /= sum;
    }

    // 3. top-k: zero out everything below the k-th largest probability.
    if params.top_k > 0 && params.top_k < n {
        let mut sorted = probs.clone();
        sorted.sort_unstable_by(|a, b| b.total_cmp(a));
        let threshold = sorted[params.top_k - 1];
        for p in &mut probs {
            if *p < threshold {
                *p = 0.0;
            }
        }
    }

    // 4. top-p (nucleus): keep the smallest high-prob set reaching `top_p`.
    if params.top_p < 1.0 {
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_unstable_by(|&a, &b| probs[b].total_cmp(&probs[a]));
        let mut cum = 0.0f32;
        let mut keep = vec![false; n];
        for &i in &order {
            keep[i] = true; // always keep at least the top token
            cum += probs[i];
            if cum >= params.top_p {
                break;
            }
        }
        for (i, p) in probs.iter_mut().enumerate() {
            if !keep[i] {
                *p = 0.0;
            }
        }
    }

    // 5. Renormalise the surviving mass and sample.
    let sum2: f32 = probs.iter().sum();
    if sum2.is_nan() || sum2 <= 1e-10 {
        return rng.next_u64() as usize % n;
    }
    for p in &mut probs {
        *p /= sum2;
    }
    let r = rng.next_f32();
    let mut cumsum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r <= cumsum {
            return i;
        }
    }
    n - 1
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::Net;
    use crate::train_transformer::TransformerNet;

    /// A small, genuine decoder-only transformer as an inference `Sequential`
    /// (`Embedding → TransformerBlock×2 → LayerNorm → Linear → Softmax`).
    fn tiny_transformer_model() -> Sequential {
        let mut rng = Rng::new(7);
        // vocab=6, context=8, embed=8, heads=2, hidden=16, 2 blocks
        let net = TransformerNet::new(6, 8, 8, 2, 16, 2, &mut rng).unwrap();
        net.to_inference().unwrap()
    }

    /// I1 core guarantee: priming the KV cache with a full `context_len` window
    /// yields the same next-token distribution as a single full forward over
    /// that window (its last row). This is what makes the cached generation a
    /// drop-in, faster replacement for the O(context²) per-token full forward.
    #[test]
    fn cached_prime_matches_full_forward_last_row() {
        let model = tiny_transformer_model();
        let context_len = 8;
        let ids = [1usize, 0, 3, 5, 2, 4, 1, 3]; // exactly one window

        let x = Tensor::matrix(1, context_len, ids.iter().map(|&t| t as f32).collect()).unwrap();
        let full = model.forward(&x).unwrap();
        let (rows, cols) = full.matrix_dims().unwrap();
        let full_last = &full.data[(rows - 1) * cols..];

        let mut cached = CachedTransformer::try_new(&model).expect("model has transformer blocks");
        assert_eq!(cached.capacity(), context_len);
        let dist = cached.prime(&ids).unwrap();

        assert_eq!(dist.len(), full_last.len());
        for (a, b) in dist.iter().zip(full_last) {
            assert!((a - b).abs() < 1e-5, "cached {a} vs full-forward {b}");
        }
        // Both are proper probability distributions (model ends in Softmax).
        assert!((dist.iter().sum::<f32>() - 1.0).abs() < 1e-4);
    }

    /// Feeding a partial prefix (window below capacity) matches a full forward
    /// over a window whose first `prefix_len` rows are those tokens: the cached
    /// path's distribution after the last fed token equals that window's row at
    /// the same position. Confirms the incremental O(context) feed is exact, not
    /// just the full-window prime.
    #[test]
    fn cached_partial_prefix_matches_full_forward_row() {
        let model = tiny_transformer_model();
        let context_len = 8;
        let prefix = [4usize, 1, 5, 2]; // 4 of 8 positions
        let p = prefix.len() - 1; // position of the last fed token

        // Full forward over a window holding `prefix` at positions 0..p (the
        // remaining slots are arbitrary; causal masking makes row p depend only
        // on positions 0..=p).
        let mut window = prefix.to_vec();
        window.resize(context_len, 0);
        let x = Tensor::matrix(1, context_len, window.iter().map(|&t| t as f32).collect()).unwrap();
        let full = model.forward(&x).unwrap();
        let (_rows, cols) = full.matrix_dims().unwrap();
        let row_p = &full.data[p * cols..(p + 1) * cols];

        let mut cached = CachedTransformer::try_new(&model).unwrap();
        let dist = cached.prime(&prefix).unwrap();
        for (a, b) in dist.iter().zip(row_p) {
            assert!((a - b).abs() < 1e-5, "cached prefix {a} vs full row {b}");
        }
    }

    /// The cached path is correct for BPE-sized vocabularies and large token
    /// IDs — not just the tiny char vocab. Drives a window of high-valued tokens
    /// through the cache and checks it tracks a full forward, confirming the BPE
    /// generation path feeds the cache valid positions/ids.
    #[test]
    fn cached_prime_matches_full_forward_bpe_vocab() {
        let mut rng = Rng::new(19);
        // vocab=300 (BPE base 256 + merges), context=6, embed=8, heads=2, 2 blocks
        let model = TransformerNet::new(300, 6, 8, 2, 16, 2, &mut rng)
            .unwrap()
            .to_inference()
            .unwrap();
        let context_len = 6;
        let ids = [257usize, 12, 299, 0, 130, 256]; // spans the byte base and merges

        let x = Tensor::matrix(1, context_len, ids.iter().map(|&t| t as f32).collect()).unwrap();
        let full = model.forward(&x).unwrap();
        let (rows, cols) = full.matrix_dims().unwrap();
        let full_last = &full.data[(rows - 1) * cols..];

        let mut cached = CachedTransformer::try_new(&model).unwrap();
        let dist = cached.prime(&ids).unwrap();
        assert_eq!(dist.len(), full_last.len());
        for (a, b) in dist.iter().zip(full_last) {
            assert!((a - b).abs() < 1e-5, "cached {a} vs full-forward {b}");
        }
    }

    // ── Sampling controls (I2) ────────────────────────────────────────────────

    #[test]
    fn temperature_only_params_match_legacy_sampler() {
        // The pre-I2 sampler: temperature softmax + one next_f32 draw. The new
        // sampler with temperature-only params must reproduce it bit-for-bit.
        let logits = [0.5f32, -1.2, 2.3, 0.0, 1.1, -0.4];
        let temp = 0.8f32;
        let legacy = |logits: &[f32], rng: &mut Rng| -> usize {
            let t = temp.max(0.01);
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut p: Vec<f32> = logits.iter().map(|&l| ((l - max) / t).exp()).collect();
            let sum: f32 = p.iter().sum();
            for v in &mut p {
                *v /= sum;
            }
            let r = rng.next_f32();
            let mut c = 0.0;
            for (i, &pi) in p.iter().enumerate() {
                c += pi;
                if r <= c {
                    return i;
                }
            }
            p.len() - 1
        };
        let params = SamplingParams::with_temperature(temp);
        let mut r1 = Rng::new(42);
        let mut r2 = Rng::new(42);
        for _ in 0..200 {
            assert_eq!(
                legacy(&logits, &mut r1),
                sample_with_params(&logits, &params, &[], &mut r2)
            );
        }
    }

    #[test]
    fn top_k_restricts_support_to_k_tokens() {
        // One clearly-dominant pair; top_k=2 must never sample outside them.
        let logits = [5.0f32, 4.0, -2.0, -3.0, -5.0];
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 2,
            top_p: 1.0,
            repetition_penalty: 1.0,
        };
        let mut rng = Rng::new(7);
        for _ in 0..500 {
            let idx = sample_with_params(&logits, &params, &[], &mut rng);
            assert!(
                idx == 0 || idx == 1,
                "top_k=2 sampled out-of-support idx {idx}"
            );
        }
    }

    #[test]
    fn top_p_nucleus_keeps_minimal_high_prob_set() {
        // Token 0 alone exceeds p=0.5, so nucleus sampling must always pick it.
        let logits = [10.0f32, 1.0, 0.5, 0.0];
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.5,
            repetition_penalty: 1.0,
        };
        let mut rng = Rng::new(3);
        for _ in 0..500 {
            assert_eq!(sample_with_params(&logits, &params, &[], &mut rng), 0);
        }
    }

    #[test]
    fn repetition_penalty_suppresses_recent_tokens() {
        // Token 0 dominates; penalising it (present in `recent`) shifts mass to 1.
        let logits = [4.0f32, 3.0, -2.0];
        let recent = [0usize];
        let penalised = SamplingParams {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 100.0,
        };
        let mut rng = Rng::new(11);
        let mut ones = 0;
        for _ in 0..1000 {
            if sample_with_params(&logits, &penalised, &recent, &mut rng) == 1 {
                ones += 1;
            }
        }
        // Without the penalty token 0 wins ~73% of the time; with a heavy penalty
        // token 1 should dominate.
        assert!(
            ones > 800,
            "repetition penalty did not redirect mass: {ones}/1000 → token 1"
        );
    }

    #[test]
    fn generation_with_params_is_deterministic_and_preserves_seed() {
        // End-to-end: the *_with API runs the cached transformer path with the
        // extra knobs and stays deterministic / seed-preserving.
        let mut rng = Rng::new(7);
        let slm = GenerativeSLM::train_transformer(
            "abcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabc",
            4,
            8,
            2,
            1,
            16,
            30,
            0.01,
            8,
            0,
            &mut rng,
        )
        .unwrap();
        let params = SamplingParams {
            temperature: 0.7,
            top_k: 3,
            top_p: 0.9,
            repetition_penalty: 1.2,
        };
        let a = slm
            .generate_with("abca", 10, &params, &mut Rng::new(99))
            .unwrap();
        let b = slm
            .generate_with("abca", 10, &params, &mut Rng::new(99))
            .unwrap();
        assert_eq!(a, b, "generation must be deterministic for a fixed RNG");
        assert!(a.starts_with("abca"));
        assert_eq!(a.chars().count(), 4 + 10);
    }

    /// End-to-end int4 (Opt#1): a trained transformer serialized to int4 and
    /// reloaded must keep its projection weights packed in memory, generate
    /// deterministically, and stay roughly faithful to the in-memory model.
    #[test]
    fn int4_serialize_load_keeps_weights_packed_and_generates() {
        let corpus: String = "the quick brown fox jumps over the lazy dog. ".repeat(8);
        let mut rng = Rng::new(7);
        // dim 32 / 1 block keeps the test cheap while the [32,32] projections are
        // still well above QUANT_MIN_LEN, so int4 packing genuinely engages.
        let slm =
            GenerativeSLM::train_transformer(&corpus, 8, 32, 4, 1, 64, 8, 0.01, 8, 0, &mut rng)
                .unwrap();

        let bytes = slm.to_bytes_quantized_int4().unwrap();
        // FINF v5.
        assert_eq!(&bytes[4..8], &5u32.to_le_bytes());
        let loaded = GenerativeSLM::from_bytes(&bytes).unwrap();

        // A transformer block projection must be quantized in memory (int4).
        let tb = loaded
            .model
            .layers()
            .iter()
            .find_map(|l| l.as_any().downcast_ref::<TransformerBlock>())
            .expect("loaded model has a transformer block");
        let qw = tb.q_proj.qweight().expect("q_proj kept packed in memory");
        assert_eq!(qw.kind, crate::quant::QKind::Int4);

        // Generation runs and is deterministic for a fixed RNG.
        let a = loaded
            .generate("the quick", 20, 0.7, &mut Rng::new(99))
            .unwrap();
        let b = loaded
            .generate("the quick", 20, 0.7, &mut Rng::new(99))
            .unwrap();
        assert_eq!(a, b);
        assert!(a.starts_with("the quick"));
        assert_eq!(a.chars().count(), "the quick".chars().count() + 20);
    }

    // ── Validation split / early stopping / best checkpoint (T5) ──────────────

    fn t5_cfg(epochs: usize) -> TransformerConfig {
        TransformerConfig {
            context_len: 4,
            embed_dim: 8,
            num_heads: 2,
            num_blocks: 1,
            hidden_dim: 16,
            epochs,
            lr: 0.02,
            batch_size: 8,
            vocab_size: 0,
            weight_decay: 0.0,
            dropout: 0.0,
        }
    }

    /// The returned model is the best-by-validation checkpoint, not the last
    /// epoch's. Re-scoring the returned model on the reconstructed validation
    /// split must reproduce the lowest cross-entropy seen during training.
    #[test]
    fn validated_training_returns_best_checkpoint() {
        let corpus: String = "abcd".repeat(80); // 320 chars, periodic
        let frac = 0.2f32;
        let vcfg = ValidationConfig {
            val_fraction: frac,
            patience: 0,
        };
        let mut evals: Vec<(usize, f32, bool)> = Vec::new();
        let mut rng = Rng::new(7);
        let slm = GenerativeSLM::train_transformer_config_validated(
            &corpus,
            &t5_cfg(12),
            1,
            &vcfg,
            &mut rng,
            |p| evals.push((p.epoch, p.val.cross_entropy, p.is_best)),
        )
        .unwrap();

        assert_eq!(evals.len(), 12, "patience=0 runs every epoch");
        assert!(
            evals.iter().any(|&(_, _, best)| best),
            "no epoch was ever best"
        );
        let min_ce = evals
            .iter()
            .map(|&(_, ce, _)| ce)
            .fold(f32::INFINITY, f32::min);

        // Reconstruct the validation split exactly as the trainer does.
        let chars: Vec<char> = corpus.chars().collect();
        let total = chars.len();
        let val_chars = ((total as f32) * frac).round() as usize;
        let val_chars = val_chars.clamp(1, total - 1);
        let val_text: String = chars[total - val_chars..].iter().collect();

        let returned_ce = slm.evaluate(&val_text).unwrap().cross_entropy;
        assert!(
            (returned_ce - min_ce).abs() < 1e-4,
            "returned model CE {returned_ce} is not the best {min_ce}"
        );
    }

    /// Early stopping halts training once validation stops improving for
    /// `patience` epochs (here the periodic corpus's val loss converges), so the
    /// run ends well before the generous epoch budget.
    #[test]
    fn validated_training_early_stops_on_plateau() {
        let corpus: String = "abcd".repeat(80);
        let vcfg = ValidationConfig {
            val_fraction: 0.2,
            patience: 3,
        };
        let mut count = 0usize;
        let mut rng = Rng::new(3);
        let _ = GenerativeSLM::train_transformer_config_validated(
            &corpus,
            &t5_cfg(200),
            1,
            &vcfg,
            &mut rng,
            |_| count += 1,
        )
        .unwrap();
        assert!(
            count < 200,
            "early stopping never triggered (ran {count} epochs)"
        );
        assert!(
            count >= 4,
            "must run at least patience+1 epochs (ran {count})"
        );
    }

    #[test]
    fn validated_training_rejects_bad_fraction() {
        let mut rng = Rng::new(1);
        for f in [0.0f32, 1.0, -0.1, 1.5] {
            let vcfg = ValidationConfig {
                val_fraction: f,
                patience: 0,
            };
            let r = GenerativeSLM::train_transformer_config_validated(
                "abcdabcdabcdabcd",
                &t5_cfg(2),
                1,
                &vcfg,
                &mut rng,
                |_| {},
            );
            assert!(r.is_err(), "val_fraction {f} should be rejected");
        }
    }

    // ── EOS / stop criterion (I3) ─────────────────────────────────────────────

    #[test]
    fn generate_until_stops_at_boundary_char_level() {
        let corpus: String = "abc".repeat(40);
        let mut rng = Rng::new(7);
        let slm =
            GenerativeSLM::train_transformer(&corpus, 4, 8, 2, 1, 16, 40, 0.01, 8, 0, &mut rng)
                .unwrap();
        let params = SamplingParams::with_temperature(0.1); // near-greedy

        let out = slm
            .generate_until("abca", 30, "c", &params, &mut Rng::new(2))
            .unwrap();
        assert!(out.starts_with("abca"));
        assert!(out.ends_with('c'), "should stop right after a 'c': {out:?}");
        assert!(out.chars().count() < 4 + 30, "did not stop early: {out:?}");

        // Deterministic for a fixed RNG.
        let a = slm
            .generate_until("abca", 30, "c", &params, &mut Rng::new(5))
            .unwrap();
        let b = slm
            .generate_until("abca", 30, "c", &params, &mut Rng::new(5))
            .unwrap();
        assert_eq!(a, b);

        // Empty stop disables early stopping → exactly the full budget.
        let full = slm
            .generate_until("abca", 12, "", &params, &mut Rng::new(5))
            .unwrap();
        assert_eq!(full.chars().count(), 4 + 12);
    }

    #[test]
    fn generate_stream_until_fragments_match_continuation() {
        let corpus: String = "abc".repeat(40);
        let mut rng = Rng::new(7);
        let slm =
            GenerativeSLM::train_transformer(&corpus, 4, 8, 2, 1, 16, 40, 0.01, 8, 0, &mut rng)
                .unwrap();
        let params = SamplingParams::with_temperature(0.1);

        let mut streamed = String::new();
        let full = slm
            .generate_stream_until("abca", 30, "c", &params, &mut Rng::new(9), |f| {
                streamed.push_str(f)
            })
            .unwrap();
        let cont: String = full.chars().skip("abca".chars().count()).collect();
        assert_eq!(
            streamed, cont,
            "streamed fragments must equal the stopped continuation"
        );
        assert!(full.ends_with('c'));
    }

    #[test]
    fn generate_until_stops_at_boundary_bpe() {
        let corpus = "the quick brown fox jumps over the lazy dog. the quick brown fox \
            jumps over the lazy dog. the quick brown fox jumps over the lazy dog. ";
        let mut rng = Rng::new(11);
        let slm =
            GenerativeSLM::train_transformer(corpus, 8, 16, 2, 1, 32, 30, 0.01, 8, 300, &mut rng)
                .unwrap();
        let params = SamplingParams::with_temperature(0.5);
        let seed = "the quick";

        let out = slm
            .generate_until(seed, 40, " ", &params, &mut Rng::new(3))
            .unwrap();
        assert!(out.starts_with(seed));
        let n = out.chars().count();
        // If it stopped before the budget, it must have stopped right after a space.
        if n < seed.chars().count() + 40 {
            assert!(
                out.ends_with(' '),
                "stopped but not at the boundary: {out:?}"
            );
        }
        // Deterministic.
        let a = slm
            .generate_until(seed, 40, " ", &params, &mut Rng::new(8))
            .unwrap();
        let b = slm
            .generate_until(seed, 40, " ", &params, &mut Rng::new(8))
            .unwrap();
        assert_eq!(a, b);
    }

    /// Models without transformer blocks (the embedded-MLP family) have no KV
    /// cache to drive, so the driver declines and the caller falls back to the
    /// full-forward loop.
    #[test]
    fn try_new_declines_non_transformer_models() {
        let mut rng = Rng::new(1);
        let net = Net::embedding_mlp(6, 4, 8, 16, 6, &mut rng);
        let model = net
            .to_inference_task(crate::csv::TaskType::TransformerSLM)
            .unwrap();
        assert!(CachedTransformer::try_new(&model).is_none());
    }
}
