//! End-to-end: build a tiny llama GGUF, export it via the library at several
//! quant levels, and confirm every output re-opens and re-loads as a runnable
//! model. Mirrors what the `export-gguf` CLI does internally.

use ferrum_core::{Gguf, GgufBuilder, GgufQuant, MetaValue};

fn f32_bytes(xs: &[f32]) -> Vec<u8> {
    let mut o = Vec::new();
    for &x in xs {
        o.extend_from_slice(&x.to_le_bytes());
    }
    o
}

fn tiny_llama() -> Vec<u8> {
    const GGML_F32: u32 = 0;
    let (dim, n_layers, n_heads, ffn, vocab) = (8usize, 2usize, 2usize, 16usize, 32usize);
    let head_dim = dim / n_heads;
    let gen = |seed: usize, n: usize| -> Vec<f32> {
        (0..n)
            .map(|i| (((seed * 131 + i * 17) % 101) as f32 / 500.0) - 0.1)
            .collect()
    };
    let mut b = GgufBuilder::new();
    b.meta("general.architecture", MetaValue::String("llama".into()));
    b.meta("llama.embedding_length", MetaValue::U32(dim as u32));
    b.meta("llama.block_count", MetaValue::U32(n_layers as u32));
    b.meta("llama.attention.head_count", MetaValue::U32(n_heads as u32));
    b.meta(
        "llama.attention.head_count_kv",
        MetaValue::U32(n_heads as u32),
    );
    b.meta("llama.feed_forward_length", MetaValue::U32(ffn as u32));
    b.meta("llama.context_length", MetaValue::U32(16));
    b.meta(
        "llama.attention.layer_norm_rms_epsilon",
        MetaValue::F32(1e-5),
    );
    b.meta("tokenizer.ggml.model", MetaValue::String("gpt2".into()));
    b.meta(
        "tokenizer.ggml.tokens",
        MetaValue::Array(
            (0..vocab)
                .map(|i| MetaValue::String(format!("t{i}")))
                .collect(),
        ),
    );
    let t = |b: &mut GgufBuilder, name: &str, dims: &[u64], seed: usize| {
        let n: usize = dims.iter().product::<u64>() as usize;
        b.tensor(name, dims, GGML_F32, f32_bytes(&gen(seed, n)));
    };
    t(&mut b, "token_embd.weight", &[dim as u64, vocab as u64], 1);
    for i in 0..n_layers {
        let p = format!("blk.{i}");
        t(
            &mut b,
            &format!("{p}.attn_norm.weight"),
            &[dim as u64],
            10 + i,
        );
        t(
            &mut b,
            &format!("{p}.attn_q.weight"),
            &[dim as u64, (n_heads * head_dim) as u64],
            20 + i,
        );
        t(
            &mut b,
            &format!("{p}.attn_k.weight"),
            &[dim as u64, (n_heads * head_dim) as u64],
            30 + i,
        );
        t(
            &mut b,
            &format!("{p}.attn_v.weight"),
            &[dim as u64, (n_heads * head_dim) as u64],
            40 + i,
        );
        t(
            &mut b,
            &format!("{p}.attn_output.weight"),
            &[(n_heads * head_dim) as u64, dim as u64],
            50 + i,
        );
        t(
            &mut b,
            &format!("{p}.ffn_norm.weight"),
            &[dim as u64],
            60 + i,
        );
        t(
            &mut b,
            &format!("{p}.ffn_gate.weight"),
            &[dim as u64, ffn as u64],
            70 + i,
        );
        t(
            &mut b,
            &format!("{p}.ffn_up.weight"),
            &[dim as u64, ffn as u64],
            80 + i,
        );
        t(
            &mut b,
            &format!("{p}.ffn_down.weight"),
            &[ffn as u64, dim as u64],
            90 + i,
        );
    }
    t(&mut b, "output_norm.weight", &[dim as u64], 200);
    t(&mut b, "output.weight", &[dim as u64, vocab as u64], 201);
    b.into_bytes()
}

#[test]
fn export_roundtrips_at_multiple_quants() {
    let g0 = Gguf::parse(tiny_llama()).unwrap();
    let model = g0.load_llama_prec(None).unwrap();
    // Small dims fall back to F16 for k-quants; that is fine — the file must
    // still re-open and re-load as a runnable model at every requested level.
    for q in [
        GgufQuant::F32,
        GgufQuant::F16,
        GgufQuant::Q8_0,
        GgufQuant::Q4_0,
        GgufQuant::Q4K,
    ] {
        let bytes = ferrum_core::llama_gguf_bytes(&model, &g0, q).unwrap();
        let g1 = Gguf::parse(bytes).unwrap();
        assert_eq!(g1.architecture(), Some("llama"));
        let m2 = g1.load_llama_prec(None).unwrap();
        let logits = m2.forward_tokens(&[1usize, 3, 5]).unwrap();
        assert!(
            logits.data.iter().all(|v| v.is_finite()),
            "non-finite logits for {q:?}"
        );
    }
}
