//! Pins for CLI gaps found while exploring the pipeline end to end. Ignored tests record a
//! known gap (see .superpowers/sdd/gap-ledger.md); un-ignore when fixed.
use std::path::PathBuf;
use std::process::Command;

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ferrum_gap_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn tt(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_train_transformer"))
        .args(args)
        .output()
        .unwrap()
}

fn tiny_train(dir: &std::path::Path, extra: &[&str]) -> std::process::Output {
    let corpus = dir.join("c.txt");
    std::fs::write(&corpus, "the cat sat on the mat. ".repeat(40)).unwrap();
    let model = dir.join("m.bin");
    let mut a = vec![
        "train",
        corpus.to_str().unwrap(),
        model.to_str().unwrap(),
        "--epochs",
        "1",
        "--context",
        "8",
        "--embed",
        "16",
        "--heads",
        "2",
        "--blocks",
        "1",
        "--hidden",
        "32",
    ];
    a.extend_from_slice(extra);
    tt(&a)
}

#[test]
fn train_rejects_nonpositive_or_nonfinite_lr() {
    for lr in ["nan", "-1", "0", "inf"] {
        let d = tmp("lr");
        let out = tiny_train(&d, &["--vocab", "300", "--lr", lr]);
        assert!(!out.status.success(), "--lr {lr} should be rejected");
        assert!(!d.join("m.bin").exists(), "--lr {lr} left a model behind");
    }
}

#[test]
#[ignore = "gap: char-level generate demands seed >= context chars; BPE models accept short seeds"]
fn char_level_generate_accepts_short_seed_like_bpe() {
    let d = tmp("charseed");
    assert!(tiny_train(&d, &["--vocab", "0"]).status.success());
    let out = tt(&[
        "generate",
        d.join("m.bin").to_str().unwrap(),
        "the",
        "--temp",
        "0",
        "--chars",
        "5",
    ]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn bpe_generate_accepts_short_seed_and_is_deterministic() {
    let d = tmp("bpeseed");
    assert!(tiny_train(&d, &["--vocab", "300"]).status.success());
    let run = || {
        tt(&[
            "generate",
            d.join("m.bin").to_str().unwrap(),
            "th",
            "--temp",
            "0",
            "--chars",
            "12",
        ])
    };
    let (a, b) = (run(), run());
    assert!(a.status.success());
    assert_eq!(a.stdout, b.stdout);
    assert!(String::from_utf8_lossy(&a.stdout).starts_with("th"));
}

#[test]
fn cli_generate_stops_at_eos() {
    let d = tmp("eosgen");
    let corpus = d.join("c.txt");
    let docs: Vec<String> = (0..60)
        .map(|i| format!("The cat sat on mat {}.", i % 4))
        .collect();
    std::fs::write(&corpus, docs.join("\n<|endoftext|>\n")).unwrap();
    let model = d.join("m.gguf");
    let out = tt(&[
        "train",
        corpus.to_str().unwrap(),
        model.to_str().unwrap(),
        "--epochs",
        "60",
        "--context",
        "16",
        "--embed",
        "32",
        "--heads",
        "2",
        "--blocks",
        "1",
        "--hidden",
        "64",
        "--vocab",
        "300",
        "--lr",
        "0.01",
        "--batch",
        "8",
    ]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let g = tt(&[
        "generate",
        model.to_str().unwrap(),
        "The cat sat on mat",
        "--temp",
        "0",
        "--chars",
        "400",
    ]);
    assert!(g.status.success());
    let text = String::from_utf8_lossy(&g.stdout);
    assert!(!text.contains("endoftext"), "{text}");
    assert!(
        text.trim().len() < 60,
        "generation should stop at the document end, got {} chars: {text}",
        text.trim().len()
    );
}
