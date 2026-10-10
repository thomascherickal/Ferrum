//! Exploration helper (ignored): export a Ferrum .bin to GGUF at every precision and print
//! Ferrum's own greedy continuation from each, so the shell can diff it against llama.cpp,
//! Ollama and LM Studio.
//! FERRUM_PROBE_BIN=m.bin FERRUM_PROBE_OUT=/dir/m FERRUM_PROBE_PROMPT="Once" FERRUM_PROBE_CHARS=40
//!   cargo test --release -p integration_tests --test probe_gguf -- --ignored --nocapture
use ferrum_core::{from_gguf, to_gguf, GenerativeSLM, QPrec, Rng, SamplingParams};

#[test]
#[ignore = "exploration helper; needs FERRUM_PROBE_* env"]
fn probe_export() {
    let bin = std::env::var("FERRUM_PROBE_BIN").unwrap();
    let out = std::env::var("FERRUM_PROBE_OUT").unwrap();
    let prompt = std::env::var("FERRUM_PROBE_PROMPT").unwrap_or("Once upon a time".into());
    let n: usize = std::env::var("FERRUM_PROBE_CHARS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(40);
    let slm = GenerativeSLM::load(&bin).unwrap();
    for (tag, prec) in [
        ("f32", QPrec::F32),
        ("q8", QPrec::Int8),
        ("q4", QPrec::Int4),
    ] {
        let bytes = to_gguf(&slm.model, &slm.norm, &slm.meta, prec).unwrap();
        std::fs::write(format!("{out}.{tag}.gguf"), &bytes).unwrap();
        let (m, nm, me) = from_gguf(&bytes).unwrap();
        let back = GenerativeSLM::new(m, nm, me);
        let p = SamplingParams::with_temperature(0.0);
        let g = back
            .generate_with(&prompt, n, &p, &mut Rng::new(1))
            .unwrap();
        println!("FERRUM[{tag}] {:?}", g);
    }
    let p = SamplingParams::with_temperature(0.0);
    let g = slm
        .generate_tokens(&prompt, 400, &[], &p_greedy(), &mut Rng::new(1))
        .unwrap();
    println!(
        "FERRUM_TOKENS {:?} n={} reason={:?}",
        g.text, g.n_tokens, g.stop_reason
    );
    println!(
        "FERRUM[bin] {:?}",
        slm.generate_with(&prompt, n, &p, &mut Rng::new(1)).unwrap()
    );
}

/// FERRUM_PROBE_LOAD=file.gguf (e.g. written by llama-quantize): load it with Ferrum and greedy-generate.
#[test]
#[ignore = "exploration helper; needs FERRUM_PROBE_LOAD env"]
fn probe_load() {
    let f = std::env::var("FERRUM_PROBE_LOAD").unwrap();
    let prompt = std::env::var("FERRUM_PROBE_PROMPT").unwrap_or("Once upon a time".into());
    let n: usize = std::env::var("FERRUM_PROBE_CHARS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(40);
    let bytes = std::fs::read(&f).unwrap();
    match from_gguf(&bytes) {
        Ok((m, nm, me)) => {
            let slm = GenerativeSLM::new(m, nm, me);
            let p = SamplingParams::with_temperature(0.0);
            println!(
                "LOADED {:?}",
                slm.generate_with(&prompt, n, &p, &mut Rng::new(1)).unwrap()
            );
        }
        Err(e) => println!("LOAD-ERROR {e}"),
    }
}

fn p_greedy() -> SamplingParams {
    SamplingParams::with_temperature(0.0)
}
