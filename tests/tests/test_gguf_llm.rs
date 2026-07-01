//! End-to-end test of the GGUF import → run pipeline that `slm_cli run-gguf`
//! drives: write a synthetic `llama`-architecture GGUF (with a tokenizer table)
//! to a temp file, then `Gguf::open` it (streamed), build its `GgufTokenizer`,
//! `load_llama_prec`, encode a prompt, generate, and decode — the exact sequence
//! the CLI performs. Validates G-K/G-T/G-Q/G-mmap wiring as one flow.

use ferrum_core::{Gguf, GgufTokenizer, QKind, Rng, SamplingParams};

const MAGIC: u32 = 0x4655_4747;

fn pu32(o: &mut Vec<u8>, v: u32) {
    o.extend_from_slice(&v.to_le_bytes());
}
fn pu64(o: &mut Vec<u8>, v: u64) {
    o.extend_from_slice(&v.to_le_bytes());
}
fn pstr(o: &mut Vec<u8>, s: &str) {
    pu64(o, s.len() as u64);
    o.extend_from_slice(s.as_bytes());
}
fn pf32(o: &mut Vec<u8>, v: f32) {
    o.extend_from_slice(&v.to_bits().to_le_bytes());
}
fn kv_u32(o: &mut Vec<u8>, k: &str, v: u32) {
    pstr(o, k);
    pu32(o, 4);
    pu32(o, v);
}
fn kv_f32(o: &mut Vec<u8>, k: &str, v: f32) {
    pstr(o, k);
    pu32(o, 6);
    pf32(o, v);
}
fn kv_str(o: &mut Vec<u8>, k: &str, v: &str) {
    pstr(o, k);
    pu32(o, 8);
    pstr(o, v);
}
fn kv_strs(o: &mut Vec<u8>, k: &str, items: &[&str]) {
    pstr(o, k);
    pu32(o, 9); // array
    pu32(o, 8); // of strings
    pu64(o, items.len() as u64);
    for s in items {
        pstr(o, s);
    }
}

/// Build a tiny but complete `llama` GGUF with an F32 weight set and a gpt2-style
/// tokenizer of 10 single-char tokens.
fn synth_llama_with_tokenizer() -> Vec<u8> {
    let (vocab, dim, n_heads, n_layers, ffn) = (10usize, 8usize, 2usize, 2usize, 16usize);
    let head_dim = dim / n_heads;
    let qd = n_heads * head_dim;

    let mut seed = 0x1234_5678u64;
    let mut rnd = |n: usize| -> Vec<f32> {
        (0..n)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (seed >> 40) as f32 / (1u64 << 23) as f32 - 1.0
            })
            .collect()
    };

    // Tensor directory + data section (all F32, type tag 0).
    let mut infos = Vec::new();
    let mut data = Vec::new();
    let mut count = 0u32;
    let add = |infos: &mut Vec<u8>, data: &mut Vec<u8>, count: &mut u32, name: &str, ne: &[u64], vals: &[f32]| {
        let off = data.len() as u64;
        for &v in vals {
            pf32(data, v);
        }
        pstr(infos, name);
        pu32(infos, ne.len() as u32);
        for &d in ne {
            pu64(infos, d);
        }
        pu32(infos, 0); // GGML_F32
        pu64(infos, off);
        *count += 1;
    };

    add(&mut infos, &mut data, &mut count, "token_embd.weight", &[dim as u64, vocab as u64], &rnd(vocab * dim));
    for i in 0..n_layers {
        let p = format!("blk.{i}");
        add(&mut infos, &mut data, &mut count, &format!("{p}.attn_norm.weight"), &[dim as u64], &vec![1.0; dim]);
        add(&mut infos, &mut data, &mut count, &format!("{p}.attn_q.weight"), &[dim as u64, qd as u64], &rnd(dim * qd));
        add(&mut infos, &mut data, &mut count, &format!("{p}.attn_k.weight"), &[dim as u64, qd as u64], &rnd(dim * qd));
        add(&mut infos, &mut data, &mut count, &format!("{p}.attn_v.weight"), &[dim as u64, qd as u64], &rnd(dim * qd));
        add(&mut infos, &mut data, &mut count, &format!("{p}.attn_output.weight"), &[qd as u64, dim as u64], &rnd(qd * dim));
        add(&mut infos, &mut data, &mut count, &format!("{p}.ffn_norm.weight"), &[dim as u64], &vec![1.0; dim]);
        add(&mut infos, &mut data, &mut count, &format!("{p}.ffn_gate.weight"), &[dim as u64, ffn as u64], &rnd(dim * ffn));
        add(&mut infos, &mut data, &mut count, &format!("{p}.ffn_up.weight"), &[dim as u64, ffn as u64], &rnd(dim * ffn));
        add(&mut infos, &mut data, &mut count, &format!("{p}.ffn_down.weight"), &[ffn as u64, dim as u64], &rnd(ffn * dim));
    }
    add(&mut infos, &mut data, &mut count, "output_norm.weight", &[dim as u64], &vec![1.0; dim]);
    add(&mut infos, &mut data, &mut count, "output.weight", &[dim as u64, vocab as u64], &rnd(dim * vocab));

    // Metadata table (architecture + tokenizer).
    let toks = ["h", "i", " ", "a", "b", "c", "d", "e", "f", "g"];
    let mut meta = Vec::new();
    kv_str(&mut meta, "general.architecture", "llama");
    kv_u32(&mut meta, "llama.embedding_length", dim as u32);
    kv_u32(&mut meta, "llama.block_count", n_layers as u32);
    kv_u32(&mut meta, "llama.attention.head_count", n_heads as u32);
    kv_u32(&mut meta, "llama.attention.head_count_kv", n_heads as u32);
    kv_u32(&mut meta, "llama.feed_forward_length", ffn as u32);
    kv_u32(&mut meta, "llama.context_length", 32);
    kv_f32(&mut meta, "llama.attention.layer_norm_rms_epsilon", 1e-5);
    kv_f32(&mut meta, "llama.rope.freq_base", 10000.0);
    kv_str(&mut meta, "tokenizer.ggml.model", "gpt2");
    kv_strs(&mut meta, "tokenizer.ggml.tokens", &toks);
    kv_strs(&mut meta, "tokenizer.ggml.merges", &[]);
    kv_u32(&mut meta, "tokenizer.ggml.bos_token_id", 0);
    kv_u32(&mut meta, "tokenizer.ggml.eos_token_id", 9);
    let n_meta = 14u64;

    let mut bytes = Vec::new();
    pu32(&mut bytes, MAGIC);
    pu32(&mut bytes, 3);
    pu64(&mut bytes, count as u64);
    pu64(&mut bytes, n_meta);
    bytes.extend_from_slice(&meta);
    bytes.extend_from_slice(&infos);
    let pad = bytes.len().div_ceil(32) * 32 - bytes.len();
    bytes.extend(std::iter::repeat_n(0u8, pad));
    bytes.extend_from_slice(&data);
    bytes
}

#[test]
fn gguf_run_pipeline_end_to_end() {
    let bytes = synth_llama_with_tokenizer();
    let path = std::env::temp_dir().join(format!("ferrum_run_gguf_{}.gguf", std::process::id()));
    std::fs::write(&path, &bytes).unwrap();

    // 1. Streamed open (G-mmap).
    let g = Gguf::open(path.to_str().unwrap()).unwrap();
    assert_eq!(g.architecture(), Some("llama"));

    // 2. Import the model's own tokenizer (G-T).
    let tok = GgufTokenizer::from_gguf(&g).unwrap();
    assert_eq!(tok.vocab_size(), 10);
    assert_eq!(tok.bos(), Some(0));
    assert_eq!(tok.eos(), Some(9));

    // 3. Load at int8 (and confirm f32 import works too — G-Q).
    let model = g.load_llama_prec(Some(QKind::Int8)).unwrap();
    assert_eq!(model.cfg.vocab_size, 10);
    assert_eq!(model.cfg.n_layers, 2);
    let f32_model = g.load_llama_prec(None).unwrap();
    assert_eq!(f32_model.cfg.vocab_size, 10);

    // 4. Encode a prompt with the imported tokenizer; ids must be in-vocab.
    let mut prompt = vec![tok.bos().unwrap()];
    prompt.extend(tok.encode("hi"));
    assert!(prompt.iter().all(|&t| t < model.cfg.vocab_size), "prompt {prompt:?} out of vocab");
    assert_eq!(tok.decode(&tok.encode("hi")), "hi", "tokenizer must round-trip");

    // 5. Generate (the CLI's exact call) and decode.
    let params = SamplingParams::with_temperature(0.8);
    let out = model.generate(&prompt, 8, &params, tok.eos(), &mut Rng::new(7)).unwrap();
    assert!(out.iter().all(|&t| t < model.cfg.vocab_size));
    let _text = tok.decode(&out); // must not panic on any produced ids

    // Deterministic for a fixed seed.
    let again = model.generate(&prompt, 8, &params, tok.eos(), &mut Rng::new(7)).unwrap();
    assert_eq!(out, again);

    let _ = std::fs::remove_file(&path);
}
