//! End-to-end test of the GGUF import → **fine-tune** → checkpoint → resume →
//! run pipeline that `slm_cli finetune-gguf` / `run-gguf --resume` drive: write a
//! synthetic `llama` GGUF, `load_llama_prec(None)`, wrap it in a `LlamaTrainer`,
//! run a few AdamW epochs (loss must fall), save a checkpoint, reload it over a
//! fresh base model, and confirm the resumed weights match and still generate.

use ferrum_core::{Adam, Gguf, LlamaTrainer, Rng, SamplingParams};

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
    pu32(o, 9);
    pu32(o, 8);
    pu64(o, items.len() as u64);
    for s in items {
        pstr(o, s);
    }
}

/// A tiny complete `llama` GGUF (F32 weights, 10-token tokenizer).
fn synth_llama() -> Vec<u8> {
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

    let mut infos = Vec::new();
    let mut data = Vec::new();
    let mut count = 0u32;
    let add = |infos: &mut Vec<u8>,
               data: &mut Vec<u8>,
               count: &mut u32,
               name: &str,
               ne: &[u64],
               vals: &[f32]| {
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

    add(
        &mut infos,
        &mut data,
        &mut count,
        "token_embd.weight",
        &[dim as u64, vocab as u64],
        &rnd(vocab * dim),
    );
    for i in 0..n_layers {
        let p = format!("blk.{i}");
        add(
            &mut infos,
            &mut data,
            &mut count,
            &format!("{p}.attn_norm.weight"),
            &[dim as u64],
            &vec![1.0; dim],
        );
        add(
            &mut infos,
            &mut data,
            &mut count,
            &format!("{p}.attn_q.weight"),
            &[dim as u64, qd as u64],
            &rnd(dim * qd),
        );
        add(
            &mut infos,
            &mut data,
            &mut count,
            &format!("{p}.attn_k.weight"),
            &[dim as u64, qd as u64],
            &rnd(dim * qd),
        );
        add(
            &mut infos,
            &mut data,
            &mut count,
            &format!("{p}.attn_v.weight"),
            &[dim as u64, qd as u64],
            &rnd(dim * qd),
        );
        add(
            &mut infos,
            &mut data,
            &mut count,
            &format!("{p}.attn_output.weight"),
            &[qd as u64, dim as u64],
            &rnd(qd * dim),
        );
        add(
            &mut infos,
            &mut data,
            &mut count,
            &format!("{p}.ffn_norm.weight"),
            &[dim as u64],
            &vec![1.0; dim],
        );
        add(
            &mut infos,
            &mut data,
            &mut count,
            &format!("{p}.ffn_gate.weight"),
            &[dim as u64, ffn as u64],
            &rnd(dim * ffn),
        );
        add(
            &mut infos,
            &mut data,
            &mut count,
            &format!("{p}.ffn_up.weight"),
            &[dim as u64, ffn as u64],
            &rnd(dim * ffn),
        );
        add(
            &mut infos,
            &mut data,
            &mut count,
            &format!("{p}.ffn_down.weight"),
            &[ffn as u64, dim as u64],
            &rnd(ffn * dim),
        );
    }
    add(
        &mut infos,
        &mut data,
        &mut count,
        "output_norm.weight",
        &[dim as u64],
        &vec![1.0; dim],
    );
    add(
        &mut infos,
        &mut data,
        &mut count,
        "output.weight",
        &[dim as u64, vocab as u64],
        &rnd(dim * vocab),
    );

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
fn gguf_finetune_checkpoint_resume_pipeline() {
    let gguf = synth_llama();
    let gpath = std::env::temp_dir().join(format!("ferrum_ft_{}.gguf", std::process::id()));
    let cpath = std::env::temp_dir().join(format!("ferrum_ft_{}.flck", std::process::id()));
    std::fs::write(&gpath, &gguf).unwrap();

    // A learnable periodic token stream over the in-vocab ids (matches how the
    // CLI tokenizes a corpus, minus the tokenizer step).
    let tokens: Vec<usize> = (0..120).map(|i| 1 + (i % 6)).collect();
    let (seq, batch) = (8usize, 8usize);

    // 1. Import f32 and wrap in a trainer (the finetune-gguf load path).
    let g = Gguf::open(gpath.to_str().unwrap()).unwrap();
    let model = g.load_llama_prec(None).unwrap();
    let mut tr = LlamaTrainer::new(model).unwrap();
    tr.set_optimizer(Adam::new(0.01));
    tr.set_grad_clip(Some(1.0));

    // 2. Fine-tune: loss must fall.
    let mut rng = Rng::new(1337);
    let first = tr.finetune_epoch(&tokens, seq, batch, &mut rng).unwrap();
    let mut last = first;
    for _ in 0..12 {
        last = tr.finetune_epoch(&tokens, seq, batch, &mut rng).unwrap();
    }
    assert!(
        last < first * 0.8,
        "fine-tune did not reduce loss: {first:.4} → {last:.4}"
    );
    let trained = tr.model_snapshot();
    let trained_steps = tr.step_count();

    // 3. Save a checkpoint (the artifact finetune-gguf writes).
    let ckpt = tr.save_checkpoint(&rng);
    std::fs::write(&cpath, &ckpt).unwrap();

    // 4. Resume over a *fresh* base model (the run-gguf --resume path): reload the
    //    base GGUF f32, apply the checkpoint, and confirm the weights + step count
    //    match the trained model bit-for-bit.
    let base2 = g.load_llama_prec(None).unwrap();
    let mut tr2 = LlamaTrainer::new(base2).unwrap();
    let bytes = std::fs::read(&cpath).unwrap();
    let _resumed_rng = tr2.load_checkpoint_into(&bytes).unwrap();
    assert_eq!(
        tr2.step_count(),
        trained_steps,
        "resumed step counter mismatch"
    );
    let resumed = tr2.model_snapshot();
    assert_eq!(resumed.len(), trained.len());
    for (a, b) in trained.iter().zip(&resumed) {
        assert_eq!(a, b, "resumed weights differ from the trained model");
    }

    // 5. The resumed model generates in-vocab, deterministic tokens.
    let params = SamplingParams::with_temperature(0.8);
    let prompt = [1usize, 2, 3];
    let out = tr2
        .model
        .generate(&prompt, 8, &params, Some(9), &mut Rng::new(7))
        .unwrap();
    assert!(out.iter().all(|&t| t < tr2.model.cfg.vocab_size));
    let again = tr2
        .model
        .generate(&prompt, 8, &params, Some(9), &mut Rng::new(7))
        .unwrap();
    assert_eq!(
        out, again,
        "generation must be deterministic for a fixed seed"
    );

    let _ = std::fs::remove_file(&gpath);
    let _ = std::fs::remove_file(&cpath);
}
