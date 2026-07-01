//! Unit and integration tests for the GenerativeSLM edge library module.
use ferrum_core::{
    slm::{build_csv_dataset, char_to_hex, corpus_vocab, hex_to_char, GenerativeSLM},
    Rng, TaskType,
};

#[test]
fn test_hex_conversion_roundtrip() {
    let test_chars = vec!['a', 'z', ' ', '\n', '🌸', '1', '$'];
    for ch in test_chars {
        let hex = char_to_hex(ch);
        let decoded = hex_to_char(&hex);
        assert_eq!(ch, decoded, "Hex roundtrip failed for '{}' ({})", ch, hex);
    }
}

#[test]
fn test_hex_conversion_invalid_fallback() {
    // Should fallback gracefully to a space or a standard character
    let ch = hex_to_char("invalid_hex_string_123");
    assert_eq!(ch, ' ');
}

#[test]
fn test_build_csv_dataset_sliding_windows() {
    let corpus = "abcdefg";
    let context_len = 3;
    let csv = build_csv_dataset(corpus, context_len).unwrap();

    // Verify CSV header: one-hot columns c{pos}_v{vocab_idx}, vocab = {a..g, ' ', '\n'} = 9 chars
    let header = csv.lines().next().unwrap();
    assert!(header.starts_with("c0_v0,"));
    assert!(header.ends_with("label"));
    assert_eq!(header.split(',').count(), 3 * 9 + 1);

    // Verify sliding windows:
    // abc -> d
    // bcd -> e
    // cde -> f
    // def -> g
    // Exactly 4 sliding-window rows — class coverage now comes from explicit
    // registration (CsvDataset::from_str_with_classes), not padding rows.
    let lines: Vec<&str> = csv.trim().lines().collect();
    assert_eq!(lines.len(), 1 + 4);
}

#[test]
fn test_trained_slm_covers_full_vocab_in_sorted_order() {
    // Characters that never appear as a target (here: nothing after the last
    // 'g'... vocab includes ' ' and '\n' which never appear at all) must still
    // be registered classes, in sorted order — the guarantee the padding rows
    // used to provide.
    let corpus = "abcdefg";
    let mut rng = Rng::new(5);
    let slm = GenerativeSLM::train(corpus, 3, 8, 5, 0.05, 0.9, 4, &mut rng).unwrap();
    let expected: Vec<String> = corpus_vocab(corpus)
        .iter()
        .map(|&c| char_to_hex(c))
        .collect();
    assert_eq!(slm.meta.class_names, expected);
    assert_eq!(slm.meta.output_dim, expected.len());
}

#[test]
fn test_build_csv_dataset_short_corpus_errors() {
    let corpus = "ab";
    let context_len = 3; // Context larger than corpus
    let result = build_csv_dataset(corpus, context_len);
    assert!(result.is_err());
}

#[test]
fn test_transformer_slm_training_and_generation_roundtrip() {
    let corpus = "abcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabc";
    let context_len = 4;
    let mut rng = Rng::new(7);

    // Train a real causal transformer end-to-end with Adam.
    let mut losses: Vec<f32> = Vec::new();
    let slm = GenerativeSLM::train_transformer_with_callback(
        corpus,
        context_len,
        8,    // embed_dim
        2,    // num_heads
        1,    // num_blocks
        16,   // hidden_dim
        40,   // epochs
        0.01, // lr (Adam)
        8,    // batch_size
        0,    // vocab_size (0 = character-level)
        &mut rng,
        |_, loss| losses.push(loss),
    )
    .unwrap();

    // Loss must drop substantially on this trivially periodic corpus.
    assert!(
        losses.last().unwrap() < &(losses[0] * 0.5),
        "loss did not halve: {} → {}",
        losses[0],
        losses.last().unwrap()
    );

    // Metadata reflects the token-ID input contract.
    assert_eq!(slm.meta.task, TaskType::TransformerSLM);
    assert_eq!(slm.meta.input_dim, context_len);
    assert_eq!(slm.meta.output_dim, slm.meta.class_names.len());

    // Greedy-ish generation continues the abc pattern.
    let generated = slm.generate("abca", 12, 0.1, &mut rng).unwrap();
    assert!(generated.starts_with("abca"));
    assert!(generated.chars().count() == 4 + 12);
    // The trained model should keep cycling a→b→c.
    assert!(
        generated.contains("bcabc"),
        "unexpected generation: {generated:?}"
    );

    // Serialization roundtrip preserves behaviour.
    let bytes = slm.to_bytes().unwrap();
    let reloaded = GenerativeSLM::from_bytes(&bytes).unwrap();
    assert_eq!(reloaded.meta.task, TaskType::TransformerSLM);
    let mut rng2 = Rng::new(123);
    let mut rng3 = Rng::new(123);
    let a = slm.generate("abca", 8, 0.1, &mut rng2).unwrap();
    let b = reloaded.generate("abca", 8, 0.1, &mut rng3).unwrap();
    assert_eq!(a, b, "reloaded model generates differently");
}

#[test]
fn test_embedded_slm_training_and_generation_roundtrip() {
    let corpus = "abcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabc";
    let context_len = 4;
    let mut rng = Rng::new(11);

    let mut losses: Vec<f32> = Vec::new();
    let slm = GenerativeSLM::train_embedded_with_callback(
        corpus,
        context_len,
        8,    // embed_dim
        32,   // hidden_size
        60,   // epochs
        0.05, // lr
        0.9,  // momentum
        8,    // batch_size
        0,    // vocab_size (0 = character-level)
        &mut rng,
        |_, loss| losses.push(loss),
    )
    .unwrap();

    assert!(
        losses.last().unwrap() < &(losses[0] * 0.5),
        "loss did not halve: {} → {}",
        losses[0],
        losses.last().unwrap()
    );

    // Token-ID input contract: input_dim = context_len, NOT context_len × vocab.
    assert_eq!(slm.meta.input_dim, context_len);
    assert_eq!(slm.meta.output_dim, slm.meta.class_names.len());

    let generated = slm.generate("abca", 12, 0.1, &mut rng).unwrap();
    assert!(generated.starts_with("abca"));
    assert_eq!(generated.chars().count(), 4 + 12);
    assert!(
        generated.contains("bcabc"),
        "unexpected generation: {generated:?}"
    );

    // Roundtrip through FINF v5 (Flatten layer forces v5) preserves behaviour.
    let bytes = slm.to_bytes().unwrap();
    let reloaded = GenerativeSLM::from_bytes(&bytes).unwrap();
    let mut rng2 = Rng::new(123);
    let mut rng3 = Rng::new(123);
    let a = slm.generate("abca", 8, 0.1, &mut rng2).unwrap();
    let b = reloaded.generate("abca", 8, 0.1, &mut rng3).unwrap();
    assert_eq!(a, b, "reloaded model generates differently");

    // Quantized serialization also reloads and generates.
    let qbytes = slm.to_bytes_quantized().unwrap();
    assert!(qbytes.len() <= bytes.len());
    let qmodel = GenerativeSLM::from_bytes(&qbytes).unwrap();
    let q = qmodel.generate("abca", 8, 0.1, &mut Rng::new(123)).unwrap();
    assert!(q.starts_with("abca"));
}

#[test]
fn test_embedded_slm_is_smaller_than_one_hot() {
    // Same corpus, context, and hidden width: the embedded model file must be
    // much smaller because the first layer no longer scales with the one-hot
    // width (context_len × vocab_size).
    let corpus = "the quick brown fox jumps over the lazy dog 0123456789\n";
    let mut rng = Rng::new(2);
    let onehot = GenerativeSLM::train(corpus, 6, 64, 2, 0.05, 0.9, 8, &mut rng).unwrap();
    let embedded =
        GenerativeSLM::train_embedded(corpus, 6, 16, 64, 2, 0.05, 0.9, 8, 0, &mut rng).unwrap();
    let onehot_len = onehot.to_bytes().unwrap().len();
    let embedded_len = embedded.to_bytes().unwrap().len();
    assert!(
        (embedded_len as f32) < (onehot_len as f32) * 0.5,
        "embedded model not smaller: {embedded_len} vs {onehot_len} bytes"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Byte-level BPE tokenizer integration (embedded + transformer paths)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_embedded_bpe_training_stores_tokenizer_and_generates() {
    let corpus = "the quick brown fox jumps over the lazy dog while the calm river \
        flows past green hills and quiet villages. travelers walk along the winding road, \
        telling stories of distant lands, bright stars, and the slow turning of the seasons. ";
    let mut rng = Rng::new(71);
    let slm =
        GenerativeSLM::train_embedded(corpus, 8, 16, 32, 30, 0.05, 0.9, 8, 300, &mut rng).unwrap();

    // Token-ID contract unchanged; tokenizer state now carries the merges.
    assert_eq!(slm.meta.task, TaskType::TransformerSLM);
    assert_eq!(slm.meta.input_dim, 8);
    assert!(!slm.meta.tokenizer_state.is_empty());
    assert!(slm.meta.output_dim >= 256);

    let out = slm
        .generate("the quick brown", 10, 0.5, &mut Rng::new(3))
        .unwrap();
    assert!(out.starts_with("the quick brown"));

    // FINF roundtrip preserves the tokenizer and behaviour.
    let bytes = slm.to_bytes_quantized().unwrap();
    let reloaded = GenerativeSLM::from_bytes(&bytes).unwrap();
    assert_eq!(reloaded.meta.tokenizer_state, slm.meta.tokenizer_state);
    let a = slm
        .generate("the quick brown", 8, 0.4, &mut Rng::new(5))
        .unwrap();
    let b = reloaded
        .generate("the quick brown", 8, 0.4, &mut Rng::new(5))
        .unwrap();
    assert_eq!(a, b);
}

#[test]
fn test_bpe_handles_non_ascii_corpus() {
    // The byte-level base vocabulary means non-ASCII text round-trips even
    // though no merge ever spans a whole multi-byte character cleanly.
    let corpus = "café au lait et résumé naïve façade. la fête commence à Genève où \
        les enfants jouent près de la rivière. élégance, créativité, persévérance — \
        des mots français avec des accents variés résonnent dans le café résumé. ";
    let mut rng = Rng::new(91);
    let slm = GenerativeSLM::train_transformer(corpus, 6, 16, 2, 1, 32, 25, 0.01, 8, 300, &mut rng)
        .unwrap();
    assert!(!slm.meta.tokenizer_state.is_empty());

    // Decoding must always yield valid UTF-8 (String guarantees it) and keep
    // the seed prefix intact.
    let out = slm
        .generate("café résumé", 12, 0.6, &mut Rng::new(2))
        .unwrap();
    assert!(
        out.starts_with("café résumé"),
        "non-ASCII seed lost: {out:?}"
    );
}

#[test]
fn test_char_level_path_still_default_when_vocab_zero() {
    // Regression guard: vocab_size 0 keeps the legacy character-level tokenizer
    // (hex class_names, empty tokenizer_state).
    let corpus = "abcabcabcabcabcabcabcabcabcabcabcabc";
    let mut rng = Rng::new(13);
    let slm =
        GenerativeSLM::train_transformer(corpus, 4, 8, 2, 1, 16, 20, 0.01, 8, 0, &mut rng).unwrap();
    assert!(slm.meta.tokenizer_state.is_empty());
    assert_eq!(slm.meta.output_dim, slm.meta.class_names.len());
}

// ─────────────────────────────────────────────────────────────────────────────
// generate_continuation: returns only the newly generated text
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_generate_continuation_excludes_seed_and_matches_generate() {
    let corpus = "abcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabc";
    let mut rng = Rng::new(7);
    let slm =
        GenerativeSLM::train_transformer(corpus, 4, 8, 2, 1, 16, 40, 0.01, 8, 0, &mut rng).unwrap();

    let seed = "abca";
    // Same RNG seed → generate() and generate_continuation() must agree, with
    // the continuation being exactly generate()'s output minus the seed prefix.
    let full = slm.generate(seed, 12, 0.1, &mut Rng::new(99)).unwrap();
    let cont = slm
        .generate_continuation(seed, 12, 0.1, &mut Rng::new(99))
        .unwrap();

    assert!(
        !cont.starts_with(seed),
        "continuation should not repeat the seed"
    );
    assert_eq!(
        cont.chars().count(),
        12,
        "continuation must honour the char budget"
    );
    assert_eq!(
        format!("{seed}{cont}"),
        full,
        "seed + continuation must equal generate()"
    );
}

#[test]
fn test_generate_continuation_bpe_path() {
    let corpus = "the quick brown fox jumps over the lazy dog while the calm river \
        flows past green hills and quiet villages. travelers walk along the winding road. ";
    let mut rng = Rng::new(71);
    let slm =
        GenerativeSLM::train_embedded(corpus, 8, 16, 32, 30, 0.05, 0.9, 8, 300, &mut rng).unwrap();

    let seed = "the quick brown";
    let full = slm.generate(seed, 10, 0.4, &mut Rng::new(5)).unwrap();
    let cont = slm
        .generate_continuation(seed, 10, 0.4, &mut Rng::new(5))
        .unwrap();
    assert_eq!(format!("{seed}{cont}"), full);
}

// ─────────────────────────────────────────────────────────────────────────────
// generate_stream: incremental generation
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_generate_stream_concatenates_to_continuation_char_level() {
    let corpus = "abcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabc";
    let mut rng = Rng::new(7);
    let slm =
        GenerativeSLM::train_transformer(corpus, 4, 8, 2, 1, 16, 40, 0.01, 8, 0, &mut rng).unwrap();

    let seed = "abca";
    // Drive the streaming API, collecting every fragment.
    let mut streamed = String::new();
    let mut frags = 0usize;
    let full = slm
        .generate_stream(seed, 12, 0.1, &mut Rng::new(55), |frag| {
            streamed.push_str(frag);
            frags += 1;
        })
        .unwrap();

    // The streamed fragments concatenate to exactly the continuation, and the
    // returned string equals generate()'s output for the same RNG seed.
    let reference = slm.generate(seed, 12, 0.1, &mut Rng::new(55)).unwrap();
    assert_eq!(full, reference, "stream return differs from generate()");
    assert_eq!(
        format!("{seed}{streamed}"),
        full,
        "fragments ≠ continuation"
    );
    // Char-level models emit one character per step.
    assert_eq!(streamed.chars().count(), 12);
    assert_eq!(frags, 12, "expected one fragment per generated character");
}

#[test]
fn test_generate_stream_matches_generate_bpe() {
    let corpus = "the quick brown fox jumps over the lazy dog while the calm river \
        flows past green hills and quiet villages. travelers walk along the winding road, \
        telling stories of distant lands, bright stars, and the slow turning of the seasons. ";
    let mut rng = Rng::new(71);
    let slm = GenerativeSLM::train_transformer(corpus, 8, 16, 2, 1, 32, 30, 0.01, 8, 300, &mut rng)
        .unwrap();

    let seed = "the quick brown";
    let mut streamed = String::new();
    let full = slm
        .generate_stream(seed, 24, 0.5, &mut Rng::new(9), |frag| {
            streamed.push_str(frag)
        })
        .unwrap();

    let reference = slm.generate(seed, 24, 0.5, &mut Rng::new(9)).unwrap();
    assert_eq!(
        full, reference,
        "stream return differs from generate() (BPE)"
    );
    // Concatenated fragments equal the continuation (return minus the seed).
    let continuation: String = full.chars().skip(seed.chars().count()).collect();
    assert_eq!(streamed, continuation, "BPE fragments ≠ continuation");
    // No replacement char should ever be streamed for this valid corpus.
    assert!(
        !streamed.contains('\u{FFFD}'),
        "streamed a U+FFFD placeholder"
    );
}

#[test]
fn test_generate_stream_handles_non_ascii_without_replacement_chars() {
    // Byte-level BPE splits multi-byte characters across tokens; streaming must
    // never emit a partial-character placeholder that later changes.
    let corpus = "café au lait et résumé naïve façade. la fête commence à Genève où \
        les enfants jouent près de la rivière. élégance, créativité, persévérance — \
        des mots français avec des accents variés résonnent dans le café résumé. ";
    let mut rng = Rng::new(91);
    let slm = GenerativeSLM::train_transformer(corpus, 6, 16, 2, 1, 32, 25, 0.01, 8, 300, &mut rng)
        .unwrap();

    let seed = "café résumé";
    let mut streamed = String::new();
    let full = slm
        .generate_stream(seed, 20, 0.6, &mut Rng::new(2), |frag| {
            streamed.push_str(frag)
        })
        .unwrap();
    let continuation: String = full.chars().skip(seed.chars().count()).collect();
    assert_eq!(streamed, continuation);
    assert!(
        !streamed.contains('\u{FFFD}'),
        "streamed a U+FFFD placeholder"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Multi-threaded (data-parallel) training via the SLM API
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_threaded_training_single_thread_matches_serial_end_to_end() {
    // threads = 1 must reproduce the serial trainer exactly, so a model trained
    // each way generates identical text.
    let corpus = "the calm river flows past green hills and quiet villages near \
        the winding road at dawn while travelers tell stories of distant lands. ";

    let serial =
        GenerativeSLM::train_transformer(corpus, 8, 16, 2, 1, 32, 40, 0.01, 8, 0, &mut Rng::new(7))
            .unwrap();
    let threaded = GenerativeSLM::train_transformer_threaded_with_callback(
        corpus,
        8,
        16,
        2,
        1,
        32,
        40,
        0.01,
        8,
        0,
        1,
        &mut Rng::new(7),
        |_, _| {},
    )
    .unwrap();

    let seed = "the calm river flows past green hills and quiet villages";
    let a = serial.generate(seed, 30, 0.3, &mut Rng::new(42)).unwrap();
    let b = threaded.generate(seed, 30, 0.3, &mut Rng::new(42)).unwrap();
    assert_eq!(a, b, "threads=1 model diverged from serial model");
}

#[test]
fn test_threaded_training_multi_thread_learns() {
    // 4-way data-parallel training must drive the loss down just like serial.
    let corpus = "abcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabc";
    let mut losses: Vec<f32> = Vec::new();
    let slm = GenerativeSLM::train_transformer_threaded_with_callback(
        corpus,
        4,
        16,
        2,
        1,
        32,
        60,
        0.01,
        16,
        0,
        4,
        &mut Rng::new(7),
        |_, loss| losses.push(loss),
    )
    .unwrap();

    assert!(
        losses.last().unwrap() < &(losses[0] * 0.5),
        "threaded loss did not halve: {} → {}",
        losses[0],
        losses.last().unwrap()
    );
    // The trained model still continues the periodic pattern.
    let out = slm.generate("abca", 12, 0.1, &mut Rng::new(3)).unwrap();
    assert!(out.starts_with("abca"));
    assert!(out.contains("bcabc"), "unexpected generation: {out:?}");
}

#[test]
fn test_threaded_training_reproducible_across_runs() {
    let corpus = "the quick brown fox jumps over the lazy dog near the calm river. ";
    let train = || {
        GenerativeSLM::train_transformer_threaded_with_callback(
            corpus,
            6,
            16,
            2,
            1,
            32,
            30,
            0.01,
            8,
            300,
            4,
            &mut Rng::new(7),
            |_, _| {},
        )
        .unwrap()
        .to_bytes()
        .unwrap()
    };
    assert_eq!(
        train(),
        train(),
        "threaded training is not reproducible run-to-run"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// evaluate: held-out cross-entropy / perplexity
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_evaluate_transformer_memorizes_training_corpus() {
    // A trivially periodic corpus is memorized, so perplexity on it approaches
    // the ideal 1.0 and cross-entropy approaches 0.
    let corpus = "abcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabc";
    let mut rng = Rng::new(7);
    let slm = GenerativeSLM::train_transformer(corpus, 4, 16, 2, 1, 32, 120, 0.01, 8, 0, &mut rng)
        .unwrap();

    let eval = slm.evaluate(corpus).unwrap();
    assert_eq!(eval.num_predictions, corpus.chars().count() - 4);
    assert!(
        eval.perplexity >= 1.0,
        "perplexity cannot be below 1.0: {}",
        eval.perplexity
    );
    assert!(
        eval.perplexity < 1.5,
        "memorized perplexity too high: {}",
        eval.perplexity
    );
    assert!(
        eval.cross_entropy < 0.4,
        "memorized cross-entropy too high: {}",
        eval.cross_entropy
    );
    // bits/token is just cross_entropy re-expressed in base 2.
    assert!((eval.bits_per_token - eval.cross_entropy / std::f32::consts::LN_2).abs() < 1e-4);
}

#[test]
fn test_evaluate_perplexity_beats_uniform_baseline() {
    // After training, held-out perplexity must be well below the uniform-model
    // baseline (= vocabulary size), proving the model learned real structure.
    let corpus = "the calm river flows past green hills and quiet villages. \
        travelers walk along the winding road, telling stories of distant lands. \
        the old town wakes at dawn as merchants open their shops. ";
    let heldout = "the river flows past the green hills and the quiet town wakes at dawn. ";
    let mut rng = Rng::new(3);
    let slm = GenerativeSLM::train_transformer(corpus, 8, 24, 4, 2, 48, 80, 0.01, 8, 0, &mut rng)
        .unwrap();

    let eval = slm.evaluate(heldout).unwrap();
    let uniform = slm.meta.output_dim as f32; // perplexity of a uniform model
    assert!(eval.num_predictions > 0);
    assert!(
        eval.perplexity < uniform * 0.5,
        "held-out perplexity {} not below half the uniform baseline {}",
        eval.perplexity,
        uniform
    );
}

#[test]
fn test_evaluate_works_for_all_three_paths_and_bpe() {
    // evaluate() must dispatch correctly over one-hot MLP, embedded, transformer,
    // and BPE models — every path returns a finite, ≥1.0 perplexity.
    let corpus = "the quick brown fox jumps over the lazy dog. the calm river flows \
        past green hills and quiet villages near the winding road at dawn. ";

    let mut rng = Rng::new(13);
    let onehot = GenerativeSLM::train(corpus, 6, 48, 40, 0.05, 0.9, 8, &mut rng).unwrap();
    let embedded =
        GenerativeSLM::train_embedded(corpus, 6, 16, 48, 40, 0.05, 0.9, 8, 0, &mut rng).unwrap();
    let transformer =
        GenerativeSLM::train_transformer(corpus, 6, 16, 2, 1, 32, 40, 0.01, 8, 0, &mut rng)
            .unwrap();
    let bpe = GenerativeSLM::train_transformer(corpus, 6, 16, 2, 1, 32, 40, 0.01, 8, 300, &mut rng)
        .unwrap();

    for (name, slm) in [
        ("onehot", &onehot),
        ("embedded", &embedded),
        ("transformer", &transformer),
        ("bpe", &bpe),
    ] {
        let eval = slm.evaluate(corpus).unwrap();
        assert!(eval.num_predictions > 0, "{name}: no predictions scored");
        assert!(eval.perplexity.is_finite(), "{name}: non-finite perplexity");
        assert!(
            eval.perplexity >= 1.0,
            "{name}: perplexity {} below 1.0",
            eval.perplexity
        );
        assert!(eval.cross_entropy >= 0.0, "{name}: negative cross-entropy");
    }
}

#[test]
fn test_evaluate_survives_finf_roundtrip() {
    // A reloaded model must score identically to the in-memory one.
    let corpus = "abcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabc";
    let mut rng = Rng::new(7);
    let slm = GenerativeSLM::train_transformer(corpus, 4, 16, 2, 1, 32, 60, 0.01, 8, 0, &mut rng)
        .unwrap();

    let before = slm.evaluate(corpus).unwrap();
    let reloaded = GenerativeSLM::from_bytes(&slm.to_bytes_quantized().unwrap()).unwrap();
    let after = reloaded.evaluate(corpus).unwrap();
    assert_eq!(before.num_predictions, after.num_predictions);
    // QAT keeps int8 drift tiny, so perplexity is preserved to within a small
    // tolerance across the f32 → int8 serialization boundary.
    assert!(
        (before.perplexity - after.perplexity).abs() < 0.01,
        "evaluation drifted across save/load: {} → {}",
        before.perplexity,
        after.perplexity
    );
}

#[test]
fn test_evaluate_rejects_text_shorter_than_context() {
    let corpus = "abcabcabcabcabcabcabcabcabcabcabcabc";
    let mut rng = Rng::new(7);
    let slm =
        GenerativeSLM::train_transformer(corpus, 8, 8, 2, 1, 16, 10, 0.01, 8, 0, &mut rng).unwrap();
    // Fewer than context_len + 1 characters → nothing to predict.
    assert!(slm.evaluate("abc").is_err());
}

#[test]
fn test_slm_training_and_generation_roundtrip() {
    let corpus = "stripe payments\nvercel develop\nstripe payouts\n";
    let context_len = 4;
    let mut rng = Rng::new(42);

    // Train the causal model
    let slm = GenerativeSLM::train(
        corpus,
        context_len,
        32,   // Hidden size
        50,   // Epochs (small for fast testing)
        0.05, // Learning rate
        0.9,  // Momentum
        8,    // Batch size
        &mut rng,
    )
    .unwrap();

    // Verify Model Metadata: inputs are one-hot, so input_dim = context_len × vocab_size
    assert_eq!(slm.meta.output_dim, slm.meta.class_names.len());
    assert_eq!(slm.meta.input_dim, context_len * slm.meta.output_dim);

    // Autoregressively generate text from seed
    let seed = "stri";
    let generated = slm.generate(seed, 20, 0.1, &mut rng).unwrap();
    assert!(generated.starts_with(seed));
    assert!(generated.chars().count() > seed.chars().count());

    // Verify serialization roundtrip
    let bytes = slm.to_bytes().unwrap();
    assert!(!bytes.is_empty());

    let reloaded = GenerativeSLM::from_bytes(&bytes).unwrap();
    assert_eq!(reloaded.meta.input_dim, slm.meta.input_dim);
    assert_eq!(reloaded.meta.output_dim, slm.meta.output_dim);
    assert_eq!(reloaded.meta.class_names, slm.meta.class_names);
}
