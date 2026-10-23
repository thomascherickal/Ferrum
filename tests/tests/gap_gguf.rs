//! Pins for GGUF interop gaps (see .superpowers/sdd/gap-ledger.md).
use ferrum_core::{Gguf, GgufBuilder, MetaValue};

fn one_tensor(ggml_type: u32, block: &[u8]) -> Result<Vec<f32>, String> {
    let mut b = GgufBuilder::new();
    b.meta("general.architecture", MetaValue::String("llama".into()));
    b.tensor("t", &[32], ggml_type, block.to_vec());
    let g = Gguf::parse(b.into_bytes()).map_err(|e| e.to_string())?;
    g.dequantize(g.tensor("t").ok_or("no tensor")?)
        .map_err(|e| e.to_string())
}

#[test]
fn q5_0_block_dequantizes() {
    // d = 1.0 (f16 0x3C00), qh = 0, qs = 0x32 -> low nibble 2 -> -14 (first 16), high nibble 3 -> -13 (last 16)
    let mut blk = vec![0x00, 0x3C, 0, 0, 0, 0];
    blk.extend([0x32u8; 16]);
    let v = one_tensor(6, &blk).unwrap();
    assert_eq!(&v[..16], &[-14.0; 16]);
    assert_eq!(&v[16..], &[-13.0; 16]);
}

#[test]
fn q5_1_block_dequantizes() {
    // d = 1.0, m = 0.5, qh = 0, qs = 0x21 -> low 1 -> 1.5 ; high 2 -> 2.5
    let mut blk = vec![0x00, 0x3C, 0x00, 0x38, 0, 0, 0, 0];
    blk.extend([0x21u8; 16]);
    let v = one_tensor(7, &blk).unwrap();
    assert_eq!(&v[..16], &[1.5; 16]);
    assert_eq!(&v[16..], &[2.5; 16]);
}
