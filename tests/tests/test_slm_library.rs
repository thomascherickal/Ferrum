//! Unit and integration tests for the GenerativeSLM edge library module.
use ferrum_core::{
    slm::{GenerativeSLM, char_to_hex, hex_to_char, build_csv_dataset},
    Rng,
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

    // Verify CSV header
    assert!(csv.starts_with("c0,c1,c2,label\n"));

    // Verify sliding windows:
    // abc -> d
    // bcd -> e
    // cde -> f
    // def -> g
    // Total of 4 sliding window rows, plus 9 vocabulary padding rows (a,b,c,d,e,f,g,' ','\n')
    let lines: Vec<&str> = csv.trim().lines().collect();
    assert_eq!(lines.len(), 1 + 4 + 9);
}

#[test]
fn test_build_csv_dataset_short_corpus_errors() {
    let corpus = "ab";
    let context_len = 3; // Context larger than corpus
    let result = build_csv_dataset(corpus, context_len);
    assert!(result.is_err());
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
        32,    // Hidden size
        50,    // Epochs (small for fast testing)
        0.05,  // Learning rate
        0.9,   // Momentum
        8,     // Batch size
        &mut rng,
    ).unwrap();

    // Verify Model Metadata
    assert_eq!(slm.meta.input_dim, context_len);
    assert_eq!(slm.meta.output_dim, slm.meta.class_names.len());

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
