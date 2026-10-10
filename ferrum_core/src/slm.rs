//! Generic offline Edge Generative SLM Library Engine.
use crate::error::{InferError, Result};
use crate::layer::{Embedding, KvCache, TransformerBlock};
use crate::loader::{from_bytes, to_bytes};
use crate::meta::{ModelMetadata, Normalizer, TaskType};
use crate::model::Sequential;
use crate::optim::Adam;
use crate::rng::Rng;
use crate::tensor::Tensor;
use crate::tokenizer::ByteBpeTokenizer;
use crate::train_transformer::{train_transformer_steps, TransformerNet};
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
    /// Tokens between training-window starts. `0` = `context_len`
    /// (non-overlapping: one epoch sees each token once). `1` = the legacy
    /// fully-overlapping window, where an epoch costs ≈`context_len`× more.
    pub window_stride: usize,
    /// Training budget in tokens (`steps × batch_size × context_len`). `0`
    /// trains for `epochs`; otherwise `epochs` is ignored and training stops
    /// once the budget is spent, mid-epoch if need be.
    pub max_tokens: u64,
    /// Int8 quantization-aware training.
    pub qat: bool,
    /// Global-norm gradient clipping threshold; `0.0` disables it.
    pub grad_clip: f32,
    /// Linear warmup then cosine decay to 0 over the whole run (`false` =
    /// fixed `lr`).
    pub cosine_lr: bool,
    /// Warmup steps for `cosine_lr`.
    pub warmup_steps: u64,
    /// Tie the LM head to the token embedding (saves `vocab × embed` params).
    pub weight_tying: bool,
    /// BPE tokenizer file: loaded when it exists, otherwise trained and
    /// written there, so later runs skip tokenizer training. Empty = train in
    /// memory every run. Ignored for character-level (`vocab_size == 0`).
    pub tokenizer_path: String,
    /// Training-checkpoint file (weights + Adam state + RNG). Empty disables.
    pub checkpoint_path: String,
    /// Checkpoint every this many optimizer steps (and always at the end);
    /// `0` = only at the end.
    pub checkpoint_every: u64,
    /// Resume from `checkpoint_path` when it exists. The checkpoint fixes the
    /// architecture; its vocab and context must match this run's.
    pub resume: bool,
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
            window_stride: 0,
            max_tokens: 0,
            qat: true,
            grad_clip: 0.0,
            cosine_lr: false,
            warmup_steps: 0,
            weight_tying: false,
            tokenizer_path: String::new(),
            checkpoint_path: String::new(),
            checkpoint_every: 0,
            resume: false,
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
    /// Number of next-token predictions scored (every token of the text, the
    /// first conditioned on the BOS token).
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

/// Loglikelihood of a token span under the model ([`GenerativeSLM::score`],
/// [`GenerativeSLM::score_rolling`]).
///
/// Per-token probabilities come from the model's softmax output, clamped at
/// `1e-12`, so one token contributes at least `ln(1e-12) ≈ −27.63`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Score {
    /// Sum of natural-log probabilities of the scored tokens.
    pub logprob: f64,
    /// Number of tokens scored.
    pub n_tokens: usize,
    /// True when every scored token was the model's argmax prediction.
    pub greedy: bool,
}

/// Windows scored per batched forward pass in [`GenerativeSLM::token_logprobs`].
const SCORE_BATCH: usize = 32;

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
    /// Softmax temperature; `<= 0` selects greedy argmax decoding.
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

/// Why [`GenerativeSLM::generate_tokens`] stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// `max_tokens` tokens were generated.
    MaxTokens,
    /// A stop string appeared (it is excluded from the text).
    Stop,
    /// The model emitted the EOS token.
    Eos,
}

/// Output of [`GenerativeSLM::generate_tokens`].
#[derive(Debug, Clone, PartialEq)]
pub struct Generation {
    /// The continuation only (prompt excluded, stop string excluded).
    pub text: String,
    /// Tokens generated (including the one that completed a stop string; EOS
    /// itself is not counted).
    pub n_tokens: usize,
    pub stop_reason: StopReason,
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
        let cfg = legacy_config(
            context_len,
            embed_dim,
            num_heads,
            num_blocks,
            hidden_dim,
            epochs,
            lr,
            batch_size,
            vocab_size,
        );
        Self::train_transformer_inner(corpus, &cfg, 1, rng, progress_callback)
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
        let cfg = legacy_config(
            context_len,
            embed_dim,
            num_heads,
            num_blocks,
            hidden_dim,
            epochs,
            lr,
            batch_size,
            vocab_size,
        );
        Self::train_transformer_inner(corpus, &cfg, threads, rng, progress_callback)
    }

    /// Shared implementation behind the serial and threaded transformer trainers.
    fn train_transformer_inner<F>(
        corpus: &str,
        cfg: &TransformerConfig,
        threads: usize,
        rng: &mut Rng,
        mut progress_callback: F,
    ) -> Result<Self>
    where
        F: FnMut(usize, f32),
    {
        let tc = tokenize_for_lm_cfg(corpus, cfg)?;
        vprintln!("[slm::train_transformer] corpus={} tokens, vocab={}, bpe={}, ctx={}, dim={}, heads={}, blocks={}, hidden={}, threads={}",
            tc.tokens.len(), tc.vocab_size, !tc.tokenizer_state.is_empty(),
            cfg.context_len, cfg.embed_dim, cfg.num_heads, cfg.num_blocks, cfg.hidden_dim, threads);
        let mut net = prepare_net(cfg, tc.vocab_size, rng)?;
        run_training(&mut net, &tc.tokens, cfg, threads, rng, |_, ep, loss| {
            progress_callback(ep, loss);
            Ok(true)
        })?;
        Self::build_transformer_slm(&net, &tc)
    }

    /// Construct an inference [`GenerativeSLM`] from a trained transformer net
    /// and its tokenization (shared by the plain and validation-aware trainers).
    fn build_transformer_slm(net: &TransformerNet, tc: &LmTokens) -> Result<Self> {
        let model_vocab = tc.vocab_size;
        let context_len = net.context_len();
        // QAT: ship exactly the Q8_0 values the training saw, so an int8 export
        // changes nothing.
        let snapped;
        let net = if net.qat_enabled() {
            let mut n = net.clone();
            n.fake_quantize_weights();
            snapped = n;
            &snapped
        } else {
            net
        };
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
        Self::train_transformer_inner(corpus, cfg, 1, rng, progress_callback)
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
        Self::train_transformer_inner(corpus, cfg, threads, rng, progress_callback)
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

        let tc = tokenize_for_lm_cfg(&train_text, cfg)?;
        vprintln!(
            "[slm::train_validated] train={} chars, val={} chars, vocab={}, patience={}",
            train_text.chars().count(),
            val_text.chars().count(),
            tc.vocab_size,
            val.patience
        );

        let mut net = prepare_net(cfg, tc.vocab_size, rng)?;
        let mut best: Option<Self> = None;
        let mut best_ce = f32::INFINITY;
        let mut stale = 0usize;

        run_training(
            &mut net,
            &tc.tokens,
            cfg,
            threads,
            rng,
            |net, ep, train_loss| {
                // Score the in-progress model on the held-out split.
                let candidate = Self::build_transformer_slm(net, &tc)?;
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
                    return Ok(false);
                }
                Ok(true)
            },
        )?;

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

        self.require_token_id_model()?;
        let context_len = self.meta.input_dim;
        vprintln!(
            "[slm::generate] vocab_size={}, context_len={}",
            self.meta.output_dim,
            context_len
        );

        // Fast path: genuine transformer models generate token-at-a-time with a
        // per-block KV cache (O(context) per token). Token-ID models without
        // transformer blocks fall through to the full-forward loop.
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

        let mut on_text = on_text;
        let mut generated = seed.to_string();
        let seed_bytes = seed.len();

        // Map each class name (a char's hex) → its index once, so per-step token
        // lookup is O(1) instead of an O(vocab) linear scan per character.
        let hex_to_idx: std::collections::HashMap<&str, usize> = self
            .meta
            .class_names
            .iter()
            .enumerate()
            .map(|(i, s)| (s.as_str(), i))
            .collect();
        let char_idx = |ch: char| -> usize {
            hex_to_idx
                .get(char_to_hex(ch).as_str())
                .copied()
                .unwrap_or(0)
        };

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

            // Token-ID input → [T, vocab] probabilities; keep the last row.
            let next_dist: Vec<f32> = {
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

        // O(1) class lookup (see `generate_stream_core`): built once, reused for
        // every seed and generated character.
        let hex_to_idx: std::collections::HashMap<&str, usize> = self
            .meta
            .class_names
            .iter()
            .enumerate()
            .map(|(i, s)| (s.as_str(), i))
            .collect();
        let char_idx = |ch: char| -> usize {
            hex_to_idx
                .get(char_to_hex(ch).as_str())
                .copied()
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
    /// Every token is scored (the first conditioned on
    /// [`GenerativeSLM::bos_id`]); long texts use sliding windows with at least
    /// `context_len / 2` tokens of history (see [`GenerativeSLM::score_rolling`]).
    /// The trained (quantization-aware) weights are used, so the reported
    /// perplexity is the quality of the model you will ship. Character-level
    /// models return an error for characters outside the vocabulary.
    ///
    /// `text` must contain at least one token, else [`InferError::DimMismatch`]
    /// is returned. Use a corpus the model was **not** trained on to measure
    /// generalization.
    pub fn evaluate(&self, text: &str) -> Result<Evaluation> {
        vprintln!(
            "[slm::evaluate] scoring {} chars of held-out text",
            text.chars().count()
        );
        let s = self.score_rolling(text)?;
        if s.n_tokens == 0 {
            return Err(InferError::DimMismatch(
                "evaluation text must contain at least one token".into(),
            ));
        }
        Ok(finish_evaluation(-s.logprob, s.n_tokens))
    }

    /// Generate up to `max_tokens` tokens after `prompt`, returning only the
    /// continuation. Stops early at the first of `stops` to appear (excluded
    /// from the text) or when the model emits EOS. Any prompt length works: a
    /// prompt longer than the context keeps its most recent tokens; an empty
    /// prompt starts from [`GenerativeSLM::bos_id`].
    ///
    /// When the KV cache fills, it is re-primed from the most recent
    /// `context_len / 2` tokens, so the cost is one re-prime per
    /// `context_len / 2` generated tokens rather than one per token (learned
    /// absolute positions rule out sliding the cache by one).
    pub fn generate_tokens(
        &self,
        prompt: &str,
        max_tokens: usize,
        stops: &[String],
        params: &SamplingParams,
        rng: &mut Rng,
    ) -> Result<Generation> {
        self.require_token_id_model()?;
        let bpe = if self.meta.tokenizer_state.is_empty() {
            None
        } else {
            Some(ByteBpeTokenizer::from_state(&self.meta.tokenizer_state)?)
        };
        let eos = bpe.as_ref().and_then(|t| t.eos_id());
        let mut cached = CachedTransformer::try_new(&self.model).ok_or_else(|| {
            InferError::DimMismatch("generate_tokens needs a transformer model".into())
        })?;
        let cap = cached.capacity().max(2);
        let keep = cap / 2;

        let mut window = self.encode_text(prompt)?;
        if window.is_empty() {
            window.push(self.bos_id());
        }
        if window.len() > cap {
            window.drain(..window.len() - cap);
        }
        let mut dist = cached.prime(&window)?;

        let mut text = String::new();
        let mut pending: Vec<u8> = Vec::new();
        let max_stop = stops.iter().map(|s| s.len()).max().unwrap_or(0);
        let mut n_tokens = 0;
        let mut reason = StopReason::MaxTokens;

        while n_tokens < max_tokens {
            let logits: Vec<f32> = dist.iter().map(|&p| p.max(1e-12).ln()).collect();
            let next = sample_with_params(&logits, params, &window, rng);
            if Some(next) == eos {
                reason = StopReason::Eos;
                break;
            }
            n_tokens += 1;
            let before = text.len();
            match &bpe {
                Some(tok) => {
                    pending.extend_from_slice(tok.token_bytes(next).unwrap_or(&[]));
                    drain_utf8(&mut pending, &mut text);
                }
                None => text.push(hex_to_char(&self.meta.class_names[next])),
            }
            // Only the tail can contain a newly completed stop string.
            let from = before.saturating_sub(max_stop);
            let from = (0..=from)
                .rev()
                .find(|&i| text.is_char_boundary(i))
                .unwrap_or(0);
            if let Some(cut) = stops
                .iter()
                .filter(|s| !s.is_empty())
                .filter_map(|s| text[from..].find(s.as_str()).map(|i| from + i))
                .min()
            {
                text.truncate(cut);
                reason = StopReason::Stop;
                break;
            }
            window.push(next);
            if window.len() > cap {
                window.drain(..window.len() - keep);
                dist = cached.prime(&window)?;
            } else {
                dist = cached.feed(next)?;
            }
        }
        if reason != StopReason::Stop && !pending.is_empty() {
            text.push_str(&String::from_utf8_lossy(&pending));
        }
        Ok(Generation {
            text,
            n_tokens,
            stop_reason: reason,
        })
    }

    /// The conditioning token placed before the first scored or generated
    /// token: the tokenizer's EOS id when it has one (document boundary), else 0.
    pub fn bos_id(&self) -> usize {
        if self.meta.tokenizer_state.is_empty() {
            return 0;
        }
        ByteBpeTokenizer::from_state(&self.meta.tokenizer_state)
            .ok()
            .and_then(|t| t.eos_id())
            .unwrap_or(0)
    }

    /// Token ids for `text`. BPE models use the document-aware encoder;
    /// character-level models map each char (CR dropped) and error on a char
    /// outside the model vocabulary instead of silently using id 0.
    pub fn encode_text(&self, text: &str) -> Result<Vec<usize>> {
        if !self.meta.tokenizer_state.is_empty() {
            let tok = ByteBpeTokenizer::from_state(&self.meta.tokenizer_state)?;
            return Ok(tok.encode_docs(text));
        }
        self.require_token_id_model()?;
        let index: std::collections::HashMap<&str, usize> = self
            .meta
            .class_names
            .iter()
            .enumerate()
            .map(|(i, s)| (s.as_str(), i))
            .collect();
        text.chars()
            .filter(|&c| c != '\r')
            .map(|ch| {
                index.get(char_to_hex(ch).as_str()).copied().ok_or_else(|| {
                    InferError::DimMismatch(format!(
                        "character {ch:?} is not in the model vocabulary"
                    ))
                })
            })
            .collect()
    }

    /// log p(`continuation` | `context`) using the lm-eval-harness convention:
    /// trailing whitespace of `context` moves to the front of `continuation`;
    /// continuation tokens are the suffix of `encode(context + continuation)`
    /// after `encode(context)` (falling back to separate encodes when the joint
    /// encoding does not start with the context's). An empty context is scored
    /// after [`GenerativeSLM::bos_id`].
    pub fn score(&self, context: &str, continuation: &str) -> Result<Score> {
        let trimmed = context.trim_end();
        let (context, continuation) = if trimmed.len() < context.len() {
            (
                trimmed,
                format!("{}{continuation}", &context[trimmed.len()..]),
            )
        } else {
            (context, continuation.to_string())
        };
        if continuation.is_empty() {
            return Ok(Score {
                logprob: 0.0,
                n_tokens: 0,
                greedy: true,
            });
        }
        let ctx_ids = if context.is_empty() {
            vec![self.bos_id()]
        } else {
            self.encode_text(context)?
        };
        let joint = if context.is_empty() {
            None
        } else {
            Some(self.encode_text(&format!("{context}{continuation}"))?)
        };
        let ids = match joint {
            Some(j) if j.len() > ctx_ids.len() && j.starts_with(&ctx_ids) => j,
            _ => {
                let mut ids = ctx_ids.clone();
                ids.extend(self.encode_text(&continuation)?);
                ids
            }
        };
        self.token_logprobs(&ids, ctx_ids.len())
    }

    /// Loglikelihood of all of `text`: every token is scored, the first one
    /// conditioned on [`GenerativeSLM::bos_id`]. Long texts are scored in
    /// sliding windows (see [`GenerativeSLM::token_logprobs`]).
    pub fn score_rolling(&self, text: &str) -> Result<Score> {
        let mut ids = vec![self.bos_id()];
        ids.extend(self.encode_text(text)?);
        self.token_logprobs(&ids, 1)
    }

    /// Sum log p(ids[i] | ids[..i]) for `i in from..ids.len()` (`from >= 1`).
    ///
    /// Transformer models: the targets are covered by windows of `context_len`
    /// tokens. The first window starts at 0; each later window starts
    /// `context_len / 2` tokens before its first new target, so every target is
    /// scored exactly once with at least half a window of history. Windows are
    /// right-padded with id 0 (safe under the causal mask) and batched
    /// [`SCORE_BATCH`] at a time through one forward pass. Models without
    /// transformer blocks fall back to one forward per target.
    fn token_logprobs(&self, ids: &[usize], from: usize) -> Result<Score> {
        let from = from.max(1);
        let mut score = Score {
            logprob: 0.0,
            n_tokens: 0,
            greedy: true,
        };
        if from >= ids.len() {
            return Ok(score);
        }
        let t = self.meta.input_dim;
        let half = (t / 2).max(1);

        // (window start, first target, end target exclusive)
        let mut windows: Vec<(usize, usize, usize)> = Vec::new();
        let mut next = from;
        while next < ids.len() {
            let start = if next <= t { 0 } else { next - half };
            let end = (start + t + 1).min(ids.len());
            windows.push((start, next, end));
            next = end;
        }

        let transformer = CachedTransformer::try_new(&self.model).is_some();
        let mut add = |row: &[f32], target: usize| {
            let p = row[target].max(1e-12) as f64;
            let argmax = (0..row.len())
                .max_by(|&a, &b| row[a].total_cmp(&row[b]))
                .unwrap_or(0);
            score.logprob += p.ln();
            score.n_tokens += 1;
            score.greedy &= argmax == target;
        };

        if transformer {
            for chunk in windows.chunks(SCORE_BATCH) {
                let mut input = Vec::with_capacity(chunk.len() * t);
                for &(start, _, _) in chunk {
                    let w = &ids[start..(start + t).min(ids.len())];
                    input.extend(w.iter().map(|&x| x as f32));
                    input.resize(input.len() + (t - w.len()), 0.0);
                }
                let out = self
                    .model
                    .forward(&Tensor::matrix(chunk.len(), t, input)?)?;
                let v = out.matrix_dims()?.1;
                for (b, &(start, first, end)) in chunk.iter().enumerate() {
                    for (target, &id) in ids.iter().enumerate().take(end).skip(first) {
                        let r = b * t + (target - 1 - start);
                        add(&out.data[r * v..(r + 1) * v], id);
                    }
                }
            }
        } else {
            for target in from..ids.len() {
                let start = target.saturating_sub(t);
                let mut ctx: Vec<f32> = vec![0.0; t - (target - start)];
                ctx.extend(ids[start..target].iter().map(|&x| x as f32));
                let out = self.model.forward(&Tensor::matrix(1, t, ctx)?)?;
                let (rows, v) = out.matrix_dims()?;
                add(&out.data[(rows - 1) * v..rows * v], ids[target]);
            }
        }
        Ok(score)
    }

    #[cfg(test)]
    fn score_rolling_prefix_for_test(&self, ids: &[usize]) -> f64 {
        self.token_logprobs(ids, 1).unwrap().logprob
    }

    /// Error unless this is a token-ID model (`TaskType::TransformerSLM`).
    /// Legacy one-hot MLP SLM files (flattened one-hot input) are no longer
    /// supported for generation or evaluation.
    fn require_token_id_model(&self) -> Result<()> {
        if self.meta.task == TaskType::TransformerSLM {
            Ok(())
        } else {
            Err(InferError::DimMismatch(format!(
                "not a token-ID SLM (task {:?}); legacy one-hot MLP models are not supported",
                self.meta.task
            )))
        }
    }
}

/// Turn an accumulated negative-log-likelihood sum into an [`Evaluation`]
/// (`count` must be non-zero).
fn finish_evaluation(total_nll: f64, count: usize) -> Evaluation {
    let ce = (total_nll / count as f64) as f32;
    Evaluation {
        num_predictions: count,
        cross_entropy: ce,
        bits_per_token: ce / std::f32::consts::LN_2,
        perplexity: ce.exp(),
    }
}

/// Move the longest valid UTF-8 prefix of `pending` onto `out`, replacing
/// invalid sequences with U+FFFD exactly like `String::from_utf8_lossy`, and
/// leave an incomplete trailing sequence in `pending` for the next token.
fn drain_utf8(pending: &mut Vec<u8>, out: &mut String) {
    loop {
        match std::str::from_utf8(pending) {
            Ok(s) => {
                out.push_str(s);
                pending.clear();
                return;
            }
            Err(e) => {
                let valid = e.valid_up_to();
                out.push_str(&String::from_utf8_lossy(&pending[..valid]));
                match e.error_len() {
                    Some(bad) => {
                        out.push('\u{FFFD}');
                        pending.drain(..valid + bad);
                    }
                    None => {
                        pending.drain(..valid);
                        return;
                    }
                }
            }
        }
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

/// [`TransformerConfig`] for the positional (legacy) training API: stride 1,
/// so those entry points keep their original per-epoch semantics.
#[allow(clippy::too_many_arguments)]
fn legacy_config(
    context_len: usize,
    embed_dim: usize,
    num_heads: usize,
    num_blocks: usize,
    hidden_dim: usize,
    epochs: usize,
    lr: f32,
    batch_size: usize,
    vocab_size: usize,
) -> TransformerConfig {
    TransformerConfig {
        context_len,
        embed_dim,
        num_heads,
        num_blocks,
        hidden_dim,
        epochs,
        lr,
        batch_size,
        vocab_size,
        window_stride: 1,
        ..TransformerConfig::default()
    }
}

/// [`tokenize_for_lm`], reusing (or creating) `cfg.tokenizer_path` for BPE.
fn tokenize_for_lm_cfg(corpus: &str, cfg: &TransformerConfig) -> Result<LmTokens> {
    let path = std::path::Path::new(&cfg.tokenizer_path);
    if cfg.vocab_size == 0 || cfg.tokenizer_path.is_empty() {
        return tokenize_for_lm(corpus, cfg.context_len, cfg.vocab_size);
    }
    let tok = if path.exists() {
        vprintln!("[slm::tokenize] reusing tokenizer {}", path.display());
        ByteBpeTokenizer::from_state(&std::fs::read_to_string(path)?)?
    } else {
        let tok = ByteBpeTokenizer::train_with_docs(corpus, cfg.vocab_size)?;
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, tok.encode_state())?;
        tok
    };
    lm_tokens_from_bpe(&tok, corpus, cfg.context_len)
}

/// A fresh (or checkpoint-resumed) training net configured from `cfg`.
fn prepare_net(cfg: &TransformerConfig, vocab: usize, rng: &mut Rng) -> Result<TransformerNet> {
    let ckpt = std::path::Path::new(&cfg.checkpoint_path);
    let mut net = if cfg.resume && !cfg.checkpoint_path.is_empty() && ckpt.exists() {
        let (net, ckpt_rng) = TransformerNet::load_checkpoint(&std::fs::read(ckpt)?)?;
        if net.vocab_size() != vocab || net.context_len() != cfg.context_len {
            return Err(InferError::DimMismatch(format!(
                "checkpoint has vocab {} / context {}, this run needs vocab {vocab} / context {}",
                net.vocab_size(),
                net.context_len(),
                cfg.context_len
            )));
        }
        *rng = ckpt_rng;
        vprintln!(
            "[slm::train] resumed {} at step {}",
            ckpt.display(),
            net.step_count()
        );
        net
    } else {
        TransformerNet::new(
            vocab,
            cfg.context_len,
            cfg.embed_dim,
            cfg.num_heads,
            cfg.hidden_dim,
            cfg.num_blocks,
            rng,
        )?
    };
    net.set_qat(cfg.qat);
    net.set_weight_decay(cfg.weight_decay);
    net.set_dropout(cfg.dropout);
    net.set_weight_tying(cfg.weight_tying);
    net.set_grad_clip((cfg.grad_clip > 0.0).then_some(cfg.grad_clip));
    net.set_window_stride(if cfg.window_stride == 0 {
        cfg.context_len
    } else {
        cfg.window_stride
    });
    vprintln!(
        "[slm::train] {} params, lr={}, wd={}, dropout={}, qat={}, stride={}, tied={}",
        net.num_params(),
        cfg.lr,
        cfg.weight_decay,
        cfg.dropout,
        cfg.qat,
        net.window_stride(),
        cfg.weight_tying
    );
    Ok(net)
}

/// Write a training checkpoint atomically (temp file + rename), so a crash
/// mid-write never leaves a truncated checkpoint behind.
fn write_checkpoint(path: &str, net: &TransformerNet, rng: &Rng) -> Result<()> {
    let tmp = format!("{path}.tmp");
    std::fs::write(&tmp, net.save_checkpoint(rng))?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// The training loop shared by every config-driven trainer: runs `cfg.epochs`
/// epochs, or until `cfg.max_tokens` is spent, checkpointing as configured.
/// `on_epoch(net, epoch, mean_loss)` runs after each (possibly partial) epoch
/// and returns `false` to stop early.
fn run_training<F>(
    net: &mut TransformerNet,
    tokens: &[usize],
    cfg: &TransformerConfig,
    threads: usize,
    rng: &mut Rng,
    mut on_epoch: F,
) -> Result<()>
where
    F: FnMut(&TransformerNet, usize, f32) -> Result<bool>,
{
    let t = net.context_len();
    let batch = cfg.batch_size.max(1);
    let windows = tokens.len().saturating_sub(t + 1) / net.window_stride() + 1;
    let steps_per_epoch = windows.div_ceil(batch).max(1) as u64;
    let budgeted = cfg.max_tokens > 0;
    let total_steps = if budgeted {
        cfg.max_tokens.div_ceil((batch * t) as u64)
    } else {
        cfg.epochs as u64 * steps_per_epoch
    };
    if cfg.cosine_lr {
        net.set_lr_schedule(Some(crate::optim::LrSchedule::warmup_cosine(
            cfg.lr,
            cfg.warmup_steps,
            total_steps,
        )));
    }
    let adam = Adam::new(cfg.lr);
    let ckpt = &cfg.checkpoint_path;
    let every = cfg.checkpoint_every;
    let mut on_step = |n: &TransformerNet, r: &Rng| -> Result<()> {
        if !ckpt.is_empty() && every > 0 && n.step_count().is_multiple_of(every) {
            write_checkpoint(ckpt, n, r)?;
        }
        Ok(())
    };
    // ponytail: a resumed run restarts its partially-done epoch with a fresh
    // shuffle; exact mid-epoch replay would need the permutation in the checkpoint.
    let mut ep = (net.step_count() / steps_per_epoch) as usize;
    loop {
        let remaining = total_steps.saturating_sub(net.step_count());
        if (budgeted && remaining == 0) || (!budgeted && ep >= cfg.epochs) {
            break;
        }
        ep += 1;
        let max_steps = if budgeted {
            remaining as usize
        } else {
            usize::MAX
        };
        let loss = train_transformer_steps(
            net,
            tokens,
            batch,
            &adam,
            rng,
            threads,
            max_steps,
            &mut on_step,
        )?;
        vprintln!(
            "[slm::train] epoch {ep}: step {}/{total_steps}, loss={loss:.6}",
            net.step_count()
        );
        if !on_epoch(net, ep, loss)? {
            break;
        }
    }
    if !ckpt.is_empty() {
        write_checkpoint(ckpt, net, rng)?;
    }
    Ok(())
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
    let tok = ByteBpeTokenizer::train_with_docs(corpus, vocab_size)?;
    lm_tokens_from_bpe(&tok, corpus, context_len)
}

fn lm_tokens_from_bpe(
    tok: &ByteBpeTokenizer,
    corpus: &str,
    context_len: usize,
) -> Result<LmTokens> {
    let tokens = tok.encode_docs(corpus);
    if tokens.len() < context_len + 1 {
        return Err(InferError::DimMismatch(
            "BPE-tokenized corpus must be longer than the context window".into(),
        ));
    }
    vprintln!(
        "[slm::tokenize_for_lm] BPE: vocab={}, {} tokens",
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

    // Temperature <= 0 is exact argmax (greedy decoding); no RNG draw.
    if params.temperature <= 0.0 {
        return (0..n)
            .max_by(|&a, &b| work[a].total_cmp(&work[b]))
            .unwrap_or(0);
    }
    let t = params.temperature.max(0.01);

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
            ..TransformerConfig::default()
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
            &t5_cfg(3000),
            1,
            &vcfg,
            &mut rng,
            |_| count += 1,
        )
        .unwrap();
        assert!(
            count < 3000,
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

    /// Models without transformer blocks have no KV cache to drive, so the
    /// driver declines and the caller falls back to the full-forward loop.
    #[test]
    fn try_new_declines_non_transformer_models() {
        let linear = crate::layer::Linear::new(4, 6, vec![0.0; 24], vec![0.0; 6]).unwrap();
        let model = Sequential::new().with(Box::new(linear));
        assert!(CachedTransformer::try_new(&model).is_none());
    }

    // ── Scoring API (Phase 0 Task 1) ──────────────────────────────────────────

    /// A tiny trained BPE transformer SLM for scoring tests (context 8).
    fn tiny_bpe_slm() -> GenerativeSLM {
        let corpus = "the cat sat on the mat. the dog sat on the log. ".repeat(20);
        let cfg = TransformerConfig {
            context_len: 8,
            embed_dim: 16,
            num_heads: 2,
            num_blocks: 1,
            hidden_dim: 32,
            epochs: 2,
            vocab_size: 270,
            ..TransformerConfig::default()
        };
        GenerativeSLM::train_transformer_config(&corpus, &cfg, &mut Rng::new(3), |_, _| {}).unwrap()
    }

    fn tiny_char_slm() -> GenerativeSLM {
        let corpus = "abcabcabc abc\n".repeat(20);
        let cfg = TransformerConfig {
            context_len: 8,
            embed_dim: 8,
            num_heads: 2,
            num_blocks: 1,
            hidden_dim: 16,
            epochs: 1,
            vocab_size: 0,
            ..TransformerConfig::default()
        };
        GenerativeSLM::train_transformer_config(&corpus, &cfg, &mut Rng::new(1), |_, _| {}).unwrap()
    }

    /// Reference: score each target with its own full forward over the
    /// preceding (<= context_len) tokens, right-padded, reading the row of the
    /// last real token. Slow but obviously correct.
    fn naive_logprobs(slm: &GenerativeSLM, ids: &[usize], from: usize) -> Vec<(f64, bool)> {
        let t = slm.meta.input_dim;
        (from..ids.len())
            .map(|i| {
                let start = i.saturating_sub(t);
                let ctx = &ids[start..i];
                let mut row: Vec<f32> = ctx.iter().map(|&x| x as f32).collect();
                row.resize(t, 0.0);
                let out = slm
                    .model
                    .forward(&Tensor::matrix(1, t, row).unwrap())
                    .unwrap();
                let v = out.matrix_dims().unwrap().1;
                let r = &out.data[(ctx.len() - 1) * v..ctx.len() * v];
                let p = r[ids[i]].max(1e-12) as f64;
                let argmax = (0..v).max_by(|&a, &b| r[a].total_cmp(&r[b])).unwrap();
                (p.ln(), argmax == ids[i])
            })
            .collect()
    }

    #[test]
    fn score_matches_naive_reference() {
        let slm = tiny_bpe_slm();
        let ctx = "the cat sat on";
        let cont = " the mat.";
        let s = slm.score(ctx, cont).unwrap();
        let ctx_ids = slm.encode_text(ctx).unwrap();
        let all = slm.encode_text(&format!("{ctx}{cont}")).unwrap();
        let want = naive_logprobs(&slm, &all, ctx_ids.len());
        let want_lp: f64 = want.iter().map(|w| w.0).sum();
        assert_eq!(s.n_tokens, all.len() - ctx_ids.len());
        assert!(
            (s.logprob - want_lp).abs() < 1e-4,
            "{} vs {want_lp}",
            s.logprob
        );
        assert_eq!(s.greedy, want.iter().all(|w| w.1));
        assert!(s.logprob < 0.0);
    }

    #[test]
    fn score_moves_trailing_whitespace_into_continuation() {
        let slm = tiny_bpe_slm();
        let a = slm.score("the cat ", "sat").unwrap();
        let b = slm.score("the cat", " sat").unwrap();
        assert_eq!(a, b);
        // Whitespace-only context trims to empty: scored after BOS.
        let ws = slm.score("  \n", "the").unwrap();
        let empty = slm.score("", "  \nthe").unwrap();
        assert_eq!(ws, empty);
        // Whitespace-only continuation is still scored.
        assert!(slm.score("the cat", " ").unwrap().n_tokens >= 1);
    }

    #[test]
    fn score_falls_back_to_separate_encodes_when_joint_differs() {
        let slm = tiny_bpe_slm();
        // "the ca" + "t" merges differently jointly than the context alone, so
        // the joint encoding does not start with the context's ids.
        let ctx_ids = slm.encode_text("the ca").unwrap();
        let joint = slm.encode_text("the cat").unwrap();
        assert!(!joint.starts_with(&ctx_ids), "premise of this test");
        let s = slm.score("the ca", "t").unwrap();
        let mut ids = ctx_ids.clone();
        ids.extend(slm.encode_text("t").unwrap());
        let want = naive_logprobs(&slm, &ids, ctx_ids.len());
        assert_eq!(s.n_tokens, want.len());
        let want_lp: f64 = want.iter().map(|w| w.0).sum();
        assert!((s.logprob - want_lp).abs() < 1e-4);
    }

    #[test]
    fn rolling_scores_each_token_once_across_windows() {
        let slm = tiny_bpe_slm();
        let text = "the dog sat on the log. the cat sat on the mat. ".repeat(4);
        let ids = slm.encode_text(&text).unwrap();
        assert!(
            ids.len() > 5 * slm.meta.input_dim,
            "text must span many windows"
        );
        let s = slm.score_rolling(&text).unwrap();
        assert_eq!(s.n_tokens, ids.len());
        let mut with_bos = vec![slm.bos_id()];
        with_bos.extend(&ids);
        // The reference sees at most context_len tokens of history, the windowed
        // scorer at least context_len/2, so only the first window is exact.
        let t = slm.meta.input_dim;
        let naive_first: f64 = naive_logprobs(&slm, &with_bos[..t + 1], 1)
            .iter()
            .map(|w| w.0)
            .sum();
        let first = slm.score_rolling_prefix_for_test(&with_bos[..t + 1]);
        assert!((first - naive_first).abs() < 1e-4);
        assert!(s.logprob.is_finite() && s.logprob < 0.0);
        // Mean per-token loss is in a sane range (not the 1e-12 clamp).
        assert!(-s.logprob / (s.n_tokens as f64) < 20.0);
    }

    #[test]
    fn rolling_handles_text_spanning_more_than_one_batch_of_windows() {
        let slm = tiny_bpe_slm();
        // > SCORE_BATCH windows of ~context_len/2 new tokens each.
        let text = "the dog sat on the log. the cat sat on the mat. ".repeat(40);
        let ids = slm.encode_text(&text).unwrap();
        assert!(ids.len() > (SCORE_BATCH + 2) * slm.meta.input_dim / 2 + slm.meta.input_dim);
        let s = slm.score_rolling(&text).unwrap();
        assert_eq!(s.n_tokens, ids.len());
        assert!(s.logprob.is_finite());
        // A prefix scored alone agrees with the same prefix inside the batch
        // for the first window (exactness of windowing start).
        let short = slm.score_rolling("the dog").unwrap();
        assert_eq!(short.n_tokens, slm.encode_text("the dog").unwrap().len());
    }

    #[test]
    fn rolling_window_boundaries_cover_exact_lengths() {
        let slm = tiny_bpe_slm();
        let t = slm.meta.input_dim;
        // Byte-level pieces: lengths straddling one window and 1.5 windows.
        for n in [1, t - 1, t, t + 1, t + 2, 2 * t, 2 * t + 1, 3 * t + 3] {
            let ids = vec![b'a' as usize; n];
            let mut with_bos = vec![slm.bos_id()];
            with_bos.extend(&ids);
            let s = slm.token_logprobs(&with_bos, 1).unwrap();
            assert_eq!(s.n_tokens, n, "n={n}");
            assert!(s.logprob.is_finite());
        }
    }

    #[test]
    fn score_edge_cases_do_not_panic() {
        let slm = tiny_bpe_slm();
        let empty_cont = slm.score("the cat", "").unwrap();
        assert_eq!(
            empty_cont,
            Score {
                logprob: 0.0,
                n_tokens: 0,
                greedy: true
            }
        );
        let empty_ctx = slm.score("", "the cat").unwrap();
        assert!(empty_ctx.n_tokens > 0 && empty_ctx.logprob < 0.0);
        let empty_text = slm.score_rolling("").unwrap();
        assert_eq!(empty_text.n_tokens, 0);
        assert_eq!(empty_text.logprob, 0.0);
        assert!(empty_text.greedy);
        let both_empty = slm.score("", "").unwrap();
        assert_eq!(both_empty.n_tokens, 0);
        // `from` past the end (or 0) is clamped, never panics.
        assert_eq!(slm.token_logprobs(&[5], 1).unwrap().n_tokens, 0);
        assert_eq!(slm.token_logprobs(&[5, 6], 0).unwrap().n_tokens, 1);
        assert_eq!(slm.token_logprobs(&[5, 6], 9).unwrap().n_tokens, 0);
    }

    #[test]
    fn greedy_flag_is_false_for_an_unlikely_continuation() {
        let slm = tiny_bpe_slm();
        let s = slm
            .score("the cat sat", "\u{1F600}\u{1F600}\u{1F600}")
            .unwrap();
        assert!(!s.greedy);
        assert!(s.n_tokens >= 3);
    }

    #[test]
    fn char_level_score_rejects_unknown_chars() {
        let slm = tiny_char_slm();
        assert!(slm.score("abc", "abc").is_ok());
        let err = slm.score("abc", "xyz").unwrap_err().to_string();
        assert!(err.contains("not in the model vocabulary"), "{err}");
        assert!(slm.score_rolling("abcx").is_err());
        assert!(slm.encode_text("zzz").is_err());
    }

    #[test]
    fn char_level_encode_drops_cr_and_bos_is_zero() {
        let slm = tiny_char_slm();
        assert_eq!(slm.bos_id(), 0);
        assert_eq!(
            slm.encode_text("ab\r\nc").unwrap(),
            slm.encode_text("ab\nc").unwrap()
        );
        assert!(slm.encode_text("").unwrap().is_empty());
        assert_eq!(slm.score_rolling("abc").unwrap().n_tokens, 3);
    }

    #[test]
    fn bpe_bos_is_eos_and_encode_text_matches_tokenizer() {
        let slm = tiny_bpe_slm();
        let tok = ByteBpeTokenizer::from_state(&slm.meta.tokenizer_state).unwrap();
        assert_eq!(slm.bos_id(), tok.eos_id().unwrap());
        assert_eq!(slm.encode_text("the cat").unwrap(), tok.encode("the cat"));
        assert!(slm.encode_text("").unwrap().is_empty());
    }

    #[test]
    fn bos_id_is_zero_when_tokenizer_state_is_unusable() {
        let mut slm = tiny_bpe_slm();
        slm.meta.tokenizer_state = "not,a,state".into();
        assert_eq!(slm.bos_id(), 0);
    }

    /// A model with no transformer blocks (just Linear + Softmax) exercises the
    /// one-forward-per-target fallback; constant bias gives a closed-form oracle.
    fn bias_only_slm() -> GenerativeSLM {
        let bias = vec![0.0f32, 1.0, 2.0, 3.0, 4.0];
        let lin = crate::layer::Linear::new(4, 5, vec![0.0; 20], bias.clone()).unwrap();
        let model = Sequential::new().with(Box::new(lin)).with(Box::new(
            crate::layer::ActivationLayer::new(crate::activation::Activation::Softmax),
        ));
        let meta = ModelMetadata {
            task: TaskType::TransformerSLM,
            class_names: "abcde".chars().map(char_to_hex).collect(),
            input_dim: 4,
            output_dim: 5,
            ..ModelMetadata::default()
        };
        GenerativeSLM::new(
            model,
            Normalizer {
                means: vec![],
                stds: vec![],
            },
            meta,
        )
    }

    #[test]
    fn evaluate_agrees_with_score_rolling_and_accepts_short_text() {
        let slm = tiny_bpe_slm();
        let text = "the cat sat on the log.";
        let e = slm.evaluate(text).unwrap();
        let s = slm.score_rolling(text).unwrap();
        assert_eq!(e.num_predictions, s.n_tokens);
        let ce = (-s.logprob / s.n_tokens as f64) as f32;
        assert!((e.cross_entropy - ce).abs() < 1e-5);
        assert!((e.perplexity - ce.exp()).abs() < 1e-3);
        assert!((e.bits_per_token - ce / std::f32::consts::LN_2).abs() < 1e-5);
        assert!(slm.evaluate("").is_err());
        // One token is enough.
        assert_eq!(slm.evaluate("a").unwrap().num_predictions, 1);
    }

    #[test]
    fn evaluate_char_level_errors_on_unknown_chars_and_counts_every_char() {
        let slm = tiny_char_slm();
        let e = slm.evaluate("abcabc").unwrap();
        assert_eq!(e.num_predictions, 6);
        assert!(slm.evaluate("abcz").is_err());
        assert!(slm.evaluate("").is_err());
        // Longer than one window: every char still counted exactly once.
        let long = "abc".repeat(30);
        assert_eq!(slm.evaluate(&long).unwrap().num_predictions, 90);
    }

    #[test]
    fn training_on_docs_registers_eos_and_bos_uses_it() {
        let corpus = "the cat sat.<|endoftext|>the dog ran.<|endoftext|>".repeat(30);
        let cfg = TransformerConfig {
            context_len: 8,
            embed_dim: 8,
            num_heads: 2,
            num_blocks: 1,
            hidden_dim: 16,
            epochs: 1,
            vocab_size: 280,
            ..TransformerConfig::default()
        };
        let slm =
            GenerativeSLM::train_transformer_config(&corpus, &cfg, &mut Rng::new(2), |_, _| {})
                .unwrap();
        let tok = ByteBpeTokenizer::from_state(&slm.meta.tokenizer_state).unwrap();
        assert_eq!(slm.bos_id(), tok.eos_id().unwrap());
        assert_eq!(tok.vocab_size(), slm.meta.output_dim);
        // Separators in text are scored as the EOS token, not as literal bytes.
        let doc = slm.encode_text("the cat sat.<|endoftext|>").unwrap();
        assert_eq!(doc.last(), tok.eos_id().as_ref());
        assert_eq!(doc.len(), tok.encode("the cat sat.").len() + 1);
    }

    #[test]
    fn legacy_positional_trainer_also_uses_doc_aware_tokenization() {
        // tokenize_for_lm (positional train_transformer API) path.
        let corpus = "the cat sat.<|endoftext|>the dog ran.<|endoftext|>".repeat(30);
        let tc = tokenize_for_lm(&corpus, 8, 280).unwrap();
        let tok = ByteBpeTokenizer::from_state(&tc.tokenizer_state).unwrap();
        let eos = tok.eos_id().unwrap();
        assert_eq!(tc.tokens.iter().filter(|&&t| t == eos).count(), 60);
        // Char-level corpora are untouched by document handling.
        let c = tokenize_for_lm("abcabcabcabc", 4, 0).unwrap();
        assert!(c.tokenizer_state.is_empty());
    }

    #[test]
    fn encode_text_rejects_legacy_non_token_models() {
        let mut slm = tiny_char_slm();
        slm.meta.task = TaskType::Classification;
        let err = slm.encode_text("abc").unwrap_err().to_string();
        assert!(err.contains("not a token-ID SLM"), "{err}");
    }

    #[test]
    fn tokenizer_path_is_created_then_reused_and_short_corpus_errors() {
        let dir = std::env::temp_dir().join(format!("ferrum_tokpath_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("nested").join("tok.txt");
        let corpus = "the cat sat.<|endoftext|>the dog ran.<|endoftext|>".repeat(30);
        let cfg = TransformerConfig {
            context_len: 8,
            vocab_size: 280,
            tokenizer_path: path.to_str().unwrap().into(),
            ..TransformerConfig::default()
        };
        let first = tokenize_for_lm_cfg(&corpus, &cfg).unwrap();
        assert!(path.exists(), "tokenizer file created (with parent dirs)");
        // Second call reuses the saved tokenizer: identical result.
        let again = tokenize_for_lm_cfg(&corpus, &cfg).unwrap();
        assert_eq!(first.tokens, again.tokens);
        assert_eq!(first.tokenizer_state, again.tokenizer_state);
        // A bare file name has no parent component to create.
        let bare_cfg = TransformerConfig {
            tokenizer_path: dir.join("bare.txt").to_str().unwrap().into(),
            ..cfg.clone()
        };
        assert!(tokenize_for_lm_cfg(&corpus, &bare_cfg).is_ok());
        // Corpus no longer than the context window is rejected.
        let tiny = TransformerConfig {
            context_len: 10_000,
            ..cfg.clone()
        };
        let err = match tokenize_for_lm_cfg(&corpus, &tiny) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("short corpus must be rejected"),
        };
        assert!(err.contains("longer than the context window"), "{err}");
        // Char-level / no path falls through to tokenize_for_lm.
        let no_path = TransformerConfig {
            tokenizer_path: String::new(),
            ..cfg
        };
        assert!(!tokenize_for_lm_cfg(&corpus, &no_path)
            .unwrap()
            .tokenizer_state
            .is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── generate_tokens (Phase 0 Task 4) ──────────────────────────────────────

    fn greedy() -> SamplingParams {
        SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        }
    }

    /// Trained on short documents so greedy decoding reaches EOS.
    fn tiny_eos_slm() -> GenerativeSLM {
        let corpus = "the cat sat.<|endoftext|>".repeat(60);
        let cfg = TransformerConfig {
            context_len: 16,
            embed_dim: 16,
            num_heads: 2,
            num_blocks: 1,
            hidden_dim: 32,
            epochs: 30,
            vocab_size: 270,
            ..TransformerConfig::default()
        };
        GenerativeSLM::train_transformer_config(&corpus, &cfg, &mut Rng::new(5), |_, _| {}).unwrap()
    }

    /// Byte-level model (no merges) trained on a two-byte UTF-8 character, so
    /// generation emits partial UTF-8 sequences token by token.
    fn tiny_utf8_slm() -> GenerativeSLM {
        let corpus = "\u{e9}".repeat(200);
        let cfg = TransformerConfig {
            context_len: 8,
            embed_dim: 16,
            num_heads: 2,
            num_blocks: 1,
            hidden_dim: 32,
            epochs: 30,
            vocab_size: 257,
            ..TransformerConfig::default()
        };
        GenerativeSLM::train_transformer_config(&corpus, &cfg, &mut Rng::new(5), |_, _| {}).unwrap()
    }

    #[test]
    fn generate_tokens_respects_budget_and_is_deterministic() {
        let slm = tiny_bpe_slm();
        let p = greedy();
        let a = slm
            .generate_tokens("the cat", 12, &[], &p, &mut Rng::new(1))
            .unwrap();
        let b = slm
            .generate_tokens("the cat", 12, &[], &p, &mut Rng::new(99))
            .unwrap();
        assert_eq!(a, b, "temperature 0 is argmax: the seed must not matter");
        assert_eq!(a.n_tokens, 12);
        assert_eq!(a.stop_reason, StopReason::MaxTokens);
        // Sampling at temperature 1 is deterministic for a fixed seed.
        let sp = SamplingParams::default();
        let s1 = slm
            .generate_tokens("the cat", 12, &[], &sp, &mut Rng::new(4))
            .unwrap();
        let s2 = slm
            .generate_tokens("the cat", 12, &[], &sp, &mut Rng::new(4))
            .unwrap();
        assert_eq!(s1, s2);
    }

    #[test]
    fn generate_tokens_zero_budget_returns_empty() {
        let slm = tiny_bpe_slm();
        let g = slm
            .generate_tokens("the cat", 0, &[], &greedy(), &mut Rng::new(1))
            .unwrap();
        assert_eq!(
            g,
            Generation {
                text: String::new(),
                n_tokens: 0,
                stop_reason: StopReason::MaxTokens
            }
        );
    }

    #[test]
    fn generate_tokens_stops_at_first_stop_and_excludes_it() {
        let slm = tiny_bpe_slm();
        let p = greedy();
        let free = slm
            .generate_tokens("the cat", 40, &[], &p, &mut Rng::new(1))
            .unwrap();
        // Pick a stop string that greedy decoding is known to produce.
        let stop = free.text.chars().skip(3).take(2).collect::<String>();
        let g = slm
            .generate_tokens(
                "the cat",
                40,
                &["#never#".into(), String::new(), stop.clone()],
                &p,
                &mut Rng::new(1),
            )
            .unwrap();
        assert_eq!(g.stop_reason, StopReason::Stop);
        assert!(
            !g.text.contains(&stop),
            "stop string must be excluded: {:?}",
            g.text
        );
        assert!(free.text.starts_with(&g.text));
        // Only empty stops: they never fire.
        let none = slm
            .generate_tokens("the cat", 10, &[String::new()], &p, &mut Rng::new(1))
            .unwrap();
        assert_eq!(none.stop_reason, StopReason::MaxTokens);
    }

    #[test]
    fn generate_tokens_stop_spanning_tokens_and_earliest_wins() {
        let slm = tiny_char_slm(); // one token per char
        let p = greedy();
        let free = slm
            .generate_tokens("ab", 12, &[], &p, &mut Rng::new(1))
            .unwrap();
        // One token per char, so a 2-char stop always spans two tokens.
        let early: String = free.text.chars().skip(1).take(2).collect();
        let late: String = free.text.chars().skip(6).take(2).collect();
        let at = free.text.find(&early).unwrap();
        let g = slm
            .generate_tokens("ab", 12, std::slice::from_ref(&early), &p, &mut Rng::new(1))
            .unwrap();
        assert_eq!(g.stop_reason, StopReason::Stop);
        assert_eq!(g.text, &free.text[..at]);
        assert_eq!(
            g.n_tokens,
            at + 2,
            "the token completing the stop is counted"
        );
        // Several stops: the earliest position in the text wins, whatever the list order.
        let cut = free
            .text
            .find(&early)
            .unwrap()
            .min(free.text.find(&late).unwrap());
        let g2 = slm
            .generate_tokens(
                "ab",
                12,
                &[late.clone(), early.clone()],
                &p,
                &mut Rng::new(1),
            )
            .unwrap();
        assert_eq!(g2.text, &free.text[..cut]);
        let g3 = slm
            .generate_tokens("ab", 12, &[early, late], &p, &mut Rng::new(1))
            .unwrap();
        assert_eq!(g3, g2);
    }

    #[test]
    fn generate_tokens_char_level_paths() {
        let slm = tiny_char_slm();
        let p = greedy();
        // Empty prompt starts from BOS.
        let g = slm
            .generate_tokens("", 5, &[], &p, &mut Rng::new(1))
            .unwrap();
        assert_eq!(g.n_tokens, 5);
        assert_eq!(g.text.chars().count(), 5);
        // Unknown prompt chars are an error, not id 0.
        assert!(slm
            .generate_tokens("xyz", 5, &[], &p, &mut Rng::new(1))
            .is_err());
    }

    #[test]
    fn generate_tokens_stops_at_eos() {
        let slm = tiny_eos_slm();
        let g = slm
            .generate_tokens("the cat", 60, &[], &greedy(), &mut Rng::new(1))
            .unwrap();
        assert_eq!(g.stop_reason, StopReason::Eos, "{g:?}");
        assert!(g.n_tokens < 60);
        assert!(!g.text.contains("endoftext"));
    }

    #[test]
    fn generate_tokens_accepts_prompt_longer_than_context() {
        let slm = tiny_bpe_slm();
        let long = "the dog sat on the log. ".repeat(10);
        let p = greedy();
        let g = slm
            .generate_tokens(&long, 30, &[], &p, &mut Rng::new(1))
            .unwrap();
        assert_eq!(g.n_tokens, 30);
        let empty = slm
            .generate_tokens("", 5, &[], &p, &mut Rng::new(1))
            .unwrap();
        assert_eq!(empty.n_tokens, 5);
        // Generation beyond several cache windows re-primes without error.
        let far = slm
            .generate_tokens("the", 100, &[], &p, &mut Rng::new(1))
            .unwrap();
        assert_eq!(far.n_tokens, 100);
    }

    #[test]
    fn generate_tokens_flushes_incomplete_utf8_at_budget() {
        let slm = tiny_utf8_slm();
        let p = greedy();
        let full = slm
            .generate_tokens("\u{e9}", 4, &[], &p, &mut Rng::new(1))
            .unwrap();
        assert_eq!(full.text, "\u{e9}\u{e9}", "premise: model learned the char");
        // An odd budget ends mid-character: the dangling lead byte is flushed
        // lossily instead of being dropped.
        let odd = slm
            .generate_tokens("\u{e9}", 3, &[], &p, &mut Rng::new(1))
            .unwrap();
        assert_eq!(odd.text, "\u{e9}\u{fffd}");
        // A multi-byte stop string straddles two byte tokens.
        let stopped = slm
            .generate_tokens("\u{e9}", 8, &["\u{e9}".into()], &p, &mut Rng::new(1))
            .unwrap();
        assert_eq!(stopped.stop_reason, StopReason::Stop);
        assert_eq!(stopped.text, "");
        let late = slm
            .generate_tokens("\u{e9}", 8, &["\u{e9}\u{e9}".into()], &p, &mut Rng::new(1))
            .unwrap();
        assert_eq!(late.stop_reason, StopReason::Stop);
        assert_eq!(late.text, "");
    }

    #[test]
    fn generate_tokens_rejects_non_transformer_and_legacy_models() {
        let slm = bias_only_slm();
        let err = slm
            .generate_tokens("ab", 3, &[], &greedy(), &mut Rng::new(1))
            .unwrap_err()
            .to_string();
        assert!(err.contains("needs a transformer model"), "{err}");
        let mut legacy = tiny_char_slm();
        legacy.meta.task = TaskType::Classification;
        assert!(legacy
            .generate_tokens("ab", 3, &[], &greedy(), &mut Rng::new(1))
            .is_err());
        let mut bad = tiny_bpe_slm();
        bad.meta.tokenizer_state = "1,2,3".into();
        assert!(bad
            .generate_tokens("ab", 3, &[], &greedy(), &mut Rng::new(1))
            .is_err());
    }

    #[test]
    fn zero_temperature_is_argmax() {
        let logits = [0.1f32, 2.0, 1.9, -1.0];
        for t in [0.0, -1.0] {
            let p = SamplingParams {
                temperature: t,
                ..SamplingParams::default()
            };
            for s in 0..20 {
                assert_eq!(sample_with_params(&logits, &p, &[], &mut Rng::new(s)), 1);
            }
        }
        // Greedy still honours the repetition penalty (applied to raw logits).
        let p = SamplingParams {
            temperature: 0.0,
            repetition_penalty: 2.0,
            ..SamplingParams::default()
        };
        assert_eq!(sample_with_params(&logits, &p, &[1], &mut Rng::new(0)), 2);
        // Greedy draws nothing from the RNG.
        let mut r = Rng::new(3);
        let before = r.next_u64();
        let mut r2 = Rng::new(3);
        sample_with_params(&logits, &greedy(), &[], &mut r2);
        assert_eq!(r2.next_u64(), before);
    }

    #[test]
    fn drain_utf8_handles_valid_invalid_and_incomplete() {
        let drain = |bytes: &[u8]| {
            let mut pending = bytes.to_vec();
            let mut out = String::new();
            drain_utf8(&mut pending, &mut out);
            (out, pending)
        };
        assert_eq!(drain(b""), (String::new(), vec![]));
        assert_eq!(drain(b"abc"), ("abc".into(), vec![]));
        assert_eq!(
            drain("h\u{e9}llo".as_bytes()),
            ("h\u{e9}llo".into(), vec![])
        );
        // Incomplete trailing sequence waits for the next token.
        assert_eq!(drain(&[0xC3]), (String::new(), vec![0xC3]));
        assert_eq!(drain(b"a\xE2\x82"), ("a".into(), vec![0xE2, 0x82]));
        // Invalid bytes become U+FFFD like from_utf8_lossy.
        assert_eq!(drain(&[0xFF, b'a']), ("\u{fffd}a".into(), vec![]));
        assert_eq!(
            drain(&[b'a', 0xFF, 0xFF, b'b']),
            ("a\u{fffd}\u{fffd}b".into(), vec![])
        );
        // Invalid followed by an incomplete tail: replace, keep the tail.
        assert_eq!(drain(&[0xFF, 0xC3]), ("\u{fffd}".into(), vec![0xC3]));
        // A lead byte followed by a non-continuation is invalid, the next char is kept.
        assert_eq!(drain(&[0xC3, b'a']), ("\u{fffd}a".into(), vec![]));
        // Appending: existing output is preserved.
        let mut pending = "\u{e9}".as_bytes()[1..].to_vec();
        let mut out = String::from("x");
        drain_utf8(&mut pending, &mut out);
        assert_eq!(out, "x\u{fffd}");
        // Equivalent to lossy decoding for complete input.
        let bytes = [b'a', 0xF0, 0x9F, 0x98, 0x80, 0xC0, b'z'];
        assert_eq!(drain(&bytes).0, String::from_utf8_lossy(&bytes));
    }

    #[test]
    fn non_transformer_fallback_scores_every_target() {
        let slm = bias_only_slm();
        let bias = [0.0f64, 1.0, 2.0, 3.0, 4.0];
        let z: f64 = bias.iter().map(|b| b.exp()).sum();
        let lp = |i: usize| bias[i] - z.ln();
        // 6 targets > context 4: covers both the padded and the full-history path.
        let s = slm.score_rolling("eeabcd").unwrap();
        assert_eq!(s.n_tokens, 6);
        let want = lp(4) + lp(4) + lp(0) + lp(1) + lp(2) + lp(3);
        assert!((s.logprob - want).abs() < 1e-4, "{} vs {want}", s.logprob);
        assert!(!s.greedy);
        let all_e = slm.score_rolling("eee").unwrap();
        assert!(all_e.greedy);
        assert_eq!(all_e.n_tokens, 3);
        assert_eq!(slm.score("ab", "e").unwrap().n_tokens, 1);
    }
}
