//! Integration tests for int8 QAT training, quantized save/load, and the
//! train-once / load-from-disk weight cache.

use ferrum_core::{GenerativeSLM, Rng, TaskType, TransformerConfig};

const CORPUS: &str = "the quick brown fox jumps over the lazy dog. \
the quick brown fox jumps over the lazy dog. \
the quick brown fox jumps over the lazy dog. ";

/// Longer, more varied text for BPE tests: byte-pair merges compress repetitive
/// corpora aggressively, so the token stream must stay comfortably longer than
/// the context window after merging.
const BPE_CORPUS: &str = "the quick brown fox jumps over the lazy dog while the \
calm river flows past green hills and quiet villages. travelers walk along the \
winding road, telling stories of distant lands, bright stars, and the slow turning \
of the seasons. morning light spills over the valley as birds begin their song, \
and the old mill wheel creaks against the steady current of the stream. ";

fn tiny_config() -> TransformerConfig {
    TransformerConfig {
        context_len: 8,
        embed_dim: 16,
        num_heads: 2,
        num_blocks: 1,
        hidden_dim: 32,
        epochs: 3,
        lr: 0.01,
        batch_size: 8,
        vocab_size: 0, // character-level baseline
        weight_decay: 0.0,
        dropout: 0.0,
    }
}

/// Same tiny network, but tokenized with a byte-level BPE vocabulary so the
/// QAT / save-load / generation path is exercised on subword tokens.
fn tiny_bpe_config() -> TransformerConfig {
    TransformerConfig {
        vocab_size: 300, // 256 byte base + up to 44 merges
        ..tiny_config()
    }
}

/// Unique temp path per test so parallel test runs don't collide.
fn temp_model_path(name: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("ferrum_test_{}_{}.bin", name, std::process::id()));
    let _ = std::fs::remove_file(&p);
    p
}

#[test]
fn default_config_is_valid() {
    let d = TransformerConfig::default();
    assert_eq!(d.embed_dim % d.num_heads, 0);
    assert!(d.context_len > 0 && d.num_blocks > 0 && d.epochs > 0);
}

#[test]
fn train_transformer_config_trains_a_generating_model() {
    let mut rng = Rng::new(1);
    let mut epochs_seen = 0usize;
    let slm =
        GenerativeSLM::train_transformer_config(CORPUS, &tiny_config(), &mut rng, |_, _| {
            epochs_seen += 1;
        })
        .unwrap();
    assert_eq!(epochs_seen, tiny_config().epochs);
    assert_eq!(slm.meta.task, TaskType::TransformerSLM);
    let out = slm.generate("the quick brown fox", 20, 0.8, &mut Rng::new(2)).unwrap();
    assert!(out.chars().count() >= "the quick brown fox".chars().count() + 20);
}

#[test]
fn save_writes_int8_finf_v5_and_load_roundtrips() {
    let path = temp_model_path("save_load");
    let mut rng = Rng::new(3);
    let slm =
        GenerativeSLM::train_transformer_config(CORPUS, &tiny_config(), &mut rng, |_, _| {})
            .unwrap();
    slm.save(path.to_str().unwrap()).unwrap();

    // Saved file must be int8-quantized FINF v5.
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(&bytes[0..4], b"FINF");
    assert_eq!(&bytes[4..8], &5u32.to_le_bytes());

    // Loading restores a working model with the same metadata, and generation
    // from the loaded model is deterministic given the same RNG seed.
    let loaded = GenerativeSLM::load(path.to_str().unwrap()).unwrap();
    assert_eq!(loaded.meta.task, TaskType::TransformerSLM);
    assert_eq!(loaded.meta.class_names, slm.meta.class_names);
    assert_eq!(loaded.meta.input_dim, slm.meta.input_dim);
    let a = loaded.generate("the quick", 15, 0.8, &mut Rng::new(9)).unwrap();
    let b = loaded.generate("the quick", 15, 0.8, &mut Rng::new(9)).unwrap();
    assert_eq!(a, b);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn save_creates_parent_directories() {
    let mut dir = std::env::temp_dir();
    dir.push(format!("ferrum_test_nested_{}", std::process::id()));
    let path = dir.join("deep").join("model.bin");
    let mut rng = Rng::new(4);
    let slm =
        GenerativeSLM::train_transformer_config(CORPUS, &tiny_config(), &mut rng, |_, _| {})
            .unwrap();
    slm.save(path.to_str().unwrap()).unwrap();
    assert!(path.exists());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn load_or_train_trains_once_then_loads_from_disk() {
    let path = temp_model_path("load_or_train");
    let path_str = path.to_str().unwrap();
    let cfg = tiny_config();

    // First call: no file on disk → trains and saves.
    let mut rng = Rng::new(5);
    let mut first_epochs = 0usize;
    let (_slm, loaded) =
        GenerativeSLM::load_or_train(path_str, CORPUS, &cfg, &mut rng, |_, _| {
            first_epochs += 1;
        })
        .unwrap();
    assert!(!loaded, "first call must train, not load");
    assert_eq!(first_epochs, cfg.epochs);
    assert!(path.exists(), "model file must be written after training");

    // Second call: file exists → loads, never invokes the training callback.
    let mut rng2 = Rng::new(6);
    let mut second_epochs = 0usize;
    let (reloaded, loaded2) =
        GenerativeSLM::load_or_train(path_str, CORPUS, &cfg, &mut rng2, |_, _| {
            second_epochs += 1;
        })
        .unwrap();
    assert!(loaded2, "second call must load from disk");
    assert_eq!(second_epochs, 0, "loading must not retrain");
    assert_eq!(reloaded.meta.task, TaskType::TransformerSLM);

    // The cached model still generates.
    let out = reloaded.generate("the quick", 10, 0.8, &mut Rng::new(7)).unwrap();
    assert!(out.starts_with("the quick"));

    let _ = std::fs::remove_file(&path);
}

// ─────────────────────────────────────────────────────────────────────────────
// Byte-level BPE tokenizer integration
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn bpe_config_trains_qat_model_and_embeds_tokenizer() {
    // BPE training must produce a QAT model whose metadata carries the merge
    // list, a ≥256 token vocabulary, and an empty character class list.
    let mut rng = Rng::new(101);
    let cfg = tiny_bpe_config();
    let slm = GenerativeSLM::train_transformer_config(BPE_CORPUS, &cfg, &mut rng, |_, _| {}).unwrap();

    assert_eq!(slm.meta.task, TaskType::TransformerSLM);
    assert!(!slm.meta.tokenizer_state.is_empty(), "BPE merge list must be stored");
    assert!(slm.meta.output_dim >= 256, "BPE vocab is at least the 256-byte base");
    assert!(slm.meta.output_dim <= cfg.vocab_size, "vocab must not exceed the requested size");
    assert!(slm.meta.class_names.is_empty(), "BPE models decode via the tokenizer, not class_names");
}

#[test]
fn bpe_model_generates_and_preserves_seed() {
    let mut rng = Rng::new(202);
    let slm =
        GenerativeSLM::train_transformer_config(BPE_CORPUS, &tiny_bpe_config(), &mut rng, |_, _| {})
            .unwrap();

    // num_chars counts characters even though generation runs over tokens.
    let seed = "the quick brown fox";
    let out = slm.generate(seed, 20, 0.8, &mut Rng::new(7)).unwrap();
    assert!(out.starts_with(seed), "seed must round-trip through the tokenizer: {out:?}");
    assert!(
        out.chars().count() >= seed.chars().count() + 20,
        "expected at least {} chars, got {}", seed.chars().count() + 20, out.chars().count()
    );
}

#[test]
fn bpe_short_seed_is_left_padded_not_rejected() {
    // A seed shorter than the context window still generates: the KV-cached
    // transformer path primes a partial window (positions 0..len), and the
    // embedded-MLP fallback left-pads the token context. Either way a short
    // prompt is accepted rather than rejected.
    let mut rng = Rng::new(303);
    let slm =
        GenerativeSLM::train_transformer_config(BPE_CORPUS, &tiny_bpe_config(), &mut rng, |_, _| {})
            .unwrap();
    let out = slm.generate("the", 8, 0.5, &mut Rng::new(1)).unwrap();
    assert!(out.starts_with("the"));
    assert!(out.chars().count() >= "the".chars().count());
}

#[test]
fn bpe_save_load_roundtrips_and_is_deterministic() {
    let path = temp_model_path("bpe_save_load");
    let mut rng = Rng::new(404);
    let slm =
        GenerativeSLM::train_transformer_config(BPE_CORPUS, &tiny_bpe_config(), &mut rng, |_, _| {})
            .unwrap();
    slm.save(path.to_str().unwrap()).unwrap();

    // Saved file is int8-quantized FINF v5 (QAT path), exactly like char models.
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(&bytes[0..4], b"FINF");
    assert_eq!(&bytes[4..8], &5u32.to_le_bytes());

    let loaded = GenerativeSLM::load(path.to_str().unwrap()).unwrap();
    // The tokenizer survives serialization.
    assert_eq!(loaded.meta.tokenizer_state, slm.meta.tokenizer_state);
    assert_eq!(loaded.meta.output_dim, slm.meta.output_dim);

    // Generation from the reloaded model is deterministic for a fixed seed and
    // matches the in-memory model (int8 QAT means the file behaves identically).
    let a = slm.generate("the quick brown", 12, 0.7, &mut Rng::new(9)).unwrap();
    let b = loaded.generate("the quick brown", 12, 0.7, &mut Rng::new(9)).unwrap();
    assert_eq!(a, b, "reloaded BPE model must generate identically");

    let _ = std::fs::remove_file(&path);
}

#[test]
fn bpe_quantized_outputs_track_in_memory_model() {
    use ferrum_core::Tensor;
    let path = temp_model_path("bpe_quant_drift");
    let mut rng = Rng::new(505);
    let slm =
        GenerativeSLM::train_transformer_config(BPE_CORPUS, &tiny_bpe_config(), &mut rng, |_, _| {})
            .unwrap();
    slm.save(path.to_str().unwrap()).unwrap();
    let loaded = GenerativeSLM::load(path.to_str().unwrap()).unwrap();

    let ctx = slm.meta.input_dim;
    let ids: Vec<f32> = (0..ctx).map(|i| (i % slm.meta.output_dim) as f32).collect();
    let x = Tensor::matrix(1, ctx, ids).unwrap();
    let a = slm.model.forward(&x).unwrap();
    let b = loaded.model.forward(&x).unwrap();
    assert_eq!(a.shape, b.shape);
    for (p, q) in a.data.iter().zip(&b.data) {
        assert!((p - q).abs() < 0.05, "int8 BPE file drifted: {p} vs {q}");
    }
    let _ = std::fs::remove_file(&path);
}

#[test]
fn bpe_vocab_below_256_is_rejected() {
    // 1..256 is invalid: the byte base vocabulary is irreducible.
    let mut rng = Rng::new(606);
    let cfg = TransformerConfig { vocab_size: 100, ..tiny_config() };
    assert!(GenerativeSLM::train_transformer_config(CORPUS, &cfg, &mut rng, |_, _| {}).is_err());
}

#[test]
fn quantized_save_outputs_stay_close_to_in_memory_model() {
    // QAT trains against int8-snapped weights, so the int8 file must behave
    // like the in-memory model: compare next-char probability rows.
    use ferrum_core::Tensor;
    let path = temp_model_path("quant_drift");
    let mut rng = Rng::new(8);
    let slm =
        GenerativeSLM::train_transformer_config(CORPUS, &tiny_config(), &mut rng, |_, _| {})
            .unwrap();
    slm.save(path.to_str().unwrap()).unwrap();
    let loaded = GenerativeSLM::load(path.to_str().unwrap()).unwrap();

    let ctx = slm.meta.input_dim;
    let ids: Vec<f32> = (0..ctx).map(|i| (i % slm.meta.output_dim) as f32).collect();
    let x = Tensor::matrix(1, ctx, ids).unwrap();
    let a = slm.model.forward(&x).unwrap();
    let b = loaded.model.forward(&x).unwrap();
    assert_eq!(a.shape, b.shape);
    for (p, q) in a.data.iter().zip(&b.data) {
        assert!((p - q).abs() < 0.05, "int8 file drifted from trained model: {p} vs {q}");
    }
    let _ = std::fs::remove_file(&path);
}
