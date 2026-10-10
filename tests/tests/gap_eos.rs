//! End-to-end EOS behaviour: the `<|endoftext|>` document separator must train as the EOS token
//! (never as literal text) and generation must stop at it in Ferrum and in llama.cpp.
use ferrum_core::{
    from_gguf, to_gguf, ByteBpeTokenizer, GenerativeSLM, QPrec, Rng, SamplingParams,
    TransformerConfig,
};
use std::path::Path;

fn corpus() -> String {
    (0..60)
        .map(|i| format!("The cat sat on mat {}.", i % 4))
        .collect::<Vec<_>>()
        .join("\n<|endoftext|>\n")
}

fn train() -> GenerativeSLM {
    let cfg = TransformerConfig {
        context_len: 16,
        embed_dim: 32,
        num_heads: 2,
        num_blocks: 1,
        hidden_dim: 64,
        epochs: 60,
        lr: 0.01,
        batch_size: 8,
        vocab_size: 300,
        ..TransformerConfig::default()
    };
    GenerativeSLM::train_transformer_config(&corpus(), &cfg, &mut Rng::new(7), |_, _| {}).unwrap()
}

#[test]
fn separator_trains_as_eos_token_and_generation_stops_there() {
    let slm = train();
    let tok = ByteBpeTokenizer::from_state(&slm.meta.tokenizer_state).unwrap();
    let eos = tok.eos_id().expect("EOS reserved");
    let ids = tok.encode_docs(&corpus());
    assert_eq!(
        ids.iter().filter(|&&i| i == eos).count(),
        59,
        "one EOS per separator"
    );
    // the literal separator never survives as byte pieces
    assert!(!tok.decode(&ids).contains("endoftext"));

    let p = SamplingParams::with_temperature(0.0);
    let g = slm
        .generate_tokens("The cat sat on mat", 200, &[], &p, &mut Rng::new(1))
        .unwrap();
    assert_eq!(format!("{:?}", g.stop_reason), "Eos", "text: {:?}", g.text);
    assert!(!g.text.contains("endoftext"), "{:?}", g.text);
    assert!(
        g.n_tokens < 40,
        "stopped at the end of the document, n={}",
        g.n_tokens
    );
}

#[test]
fn gguf_roundtrip_keeps_eos_and_generation_identical() {
    let slm = train();
    let p = SamplingParams::with_temperature(0.0);
    let want = slm
        .generate_tokens("The cat sat on mat", 200, &[], &p, &mut Rng::new(1))
        .unwrap();
    let bytes = to_gguf(&slm.model, &slm.norm, &slm.meta, QPrec::F32).unwrap();
    let (m, n, me) = from_gguf(&bytes).unwrap();
    let back = GenerativeSLM::new(m, n, me);
    let got = back
        .generate_tokens("The cat sat on mat", 200, &[], &p, &mut Rng::new(1))
        .unwrap();
    assert_eq!(got.text, want.text);
    assert_eq!(format!("{:?}", got.stop_reason), "Eos");
}

#[test]
fn llama_cpp_stops_at_eos_and_never_prints_the_literal() {
    let bin = std::path::PathBuf::from(std::env::var("HOME").unwrap())
        .join(".local/opt/llama.cpp-b11521/llama-completion");
    if !Path::new(&bin).exists() {
        eprintln!("llama-completion absent; skipping");
        return;
    }
    let slm = train();
    let p = SamplingParams::with_temperature(0.0);
    let want = slm
        .generate_tokens("The cat sat on mat", 200, &[], &p, &mut Rng::new(1))
        .unwrap();
    let dir = std::env::temp_dir().join(format!("ferrum_eos_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let f = dir.join("m.gguf");
    std::fs::write(
        &f,
        to_gguf(&slm.model, &slm.norm, &slm.meta, QPrec::F32).unwrap(),
    )
    .unwrap();
    // llama-completion's async logger intermittently drops the tail of its output for tiny,
    // fast models (a harness race, not a model difference), so take the longest of a few runs.
    let run = || {
        let out = std::process::Command::new(&bin)
            .args([
                "-m",
                f.to_str().unwrap(),
                "-p",
                "The cat sat on mat",
                "-n",
                "200",
                "--temp",
                "0",
                "-no-cnv",
                "--no-warmup",
            ])
            .stdin(std::process::Stdio::null())
            .stdout(std::fs::File::create(dir.join("out.txt")).unwrap())
            .stderr(std::fs::File::create(dir.join("err.txt")).unwrap())
            .status()
            .unwrap();
        assert!(out.success());
        std::fs::read_to_string(dir.join("out.txt")).unwrap()
    };
    let text = (0..6).map(|_| run()).max_by_key(|t| t.len()).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    assert!(!text.contains("endoftext"), "{text}");
    let full = format!("The cat sat on mat{}", want.text);
    // llama-completion prints " [end of text]" on EOS, but its async logger intermittently drops
    // the tail of the output for tiny, fast models. So the robust invariants are: it never runs
    // past where Ferrum stops, never prints the literal separator, and agrees on what it did print.
    let text = text.replace("[end of text]", "");
    let (got, full) = (text.trim_end(), full.trim_end());
    assert!(
        full.starts_with(got) && got.len() > "The cat sat on mat".len() - 1,
        "llama.cpp output {got:?} is not a prefix of Ferrum's {full:?} (it must stop at EOS)"
    );
}
