//! Shared GGUF plumbing for the two training-checkpoint codecs
//! ([`TransformerNet::save_checkpoint`](crate::train_transformer::TransformerNet::save_checkpoint)
//! and [`LlamaTrainer::save_checkpoint`](crate::llm_train::LlamaTrainer::save_checkpoint)).
//!
//! A checkpoint is a GGUF v3 container: `general.architecture` names the
//! codec, `ferrum.ckpt.*` scalars carry the shape / flags / step / RNG state,
//! and every parameter contributes three F32 tensors (`p{i}.data`, `p{i}.m`,
//! `p{i}.v`). Tensors are always F32: training state must be exact, never
//! quantized.

use crate::error::{InferError, Result};
use crate::gguf::{Gguf, MetaValue, GGML_F32};
use crate::gguf_write::GgufBuilder;

/// Checkpoint layout version written under `ferrum.ckpt.version`.
const VERSION: u32 = 1;

fn bad(msg: String) -> InferError {
    InferError::Format(msg)
}

/// A builder primed with the architecture and layout version.
pub(crate) fn begin(arch: &str) -> GgufBuilder {
    let mut b = GgufBuilder::new();
    b.meta("general.architecture", MetaValue::String(arch.into()));
    b.meta("ferrum.ckpt.version", MetaValue::U32(VERSION));
    b
}

pub(crate) fn put_u32(b: &mut GgufBuilder, name: &str, v: usize) {
    b.meta(&format!("ferrum.ckpt.{name}"), MetaValue::U32(v as u32));
}

pub(crate) fn put_u64(b: &mut GgufBuilder, name: &str, v: u64) {
    b.meta(&format!("ferrum.ckpt.{name}"), MetaValue::U64(v));
}

pub(crate) fn put_bool(b: &mut GgufBuilder, name: &str, v: bool) {
    b.meta(&format!("ferrum.ckpt.{name}"), MetaValue::Bool(v));
}

/// One parameter's weights and both Adam moments as three F32 tensors.
pub(crate) fn put_param(b: &mut GgufBuilder, i: usize, data: &[f32], m: &[f32], v: &[f32]) {
    for (suffix, xs) in [("data", data), ("m", m), ("v", v)] {
        let bytes = xs.iter().flat_map(|x| x.to_le_bytes()).collect();
        b.tensor(
            &format!("p{i}.{suffix}"),
            &[xs.len() as u64],
            GGML_F32,
            bytes,
        );
    }
}

/// Parse checkpoint bytes and validate the container: old-format magic ->
/// pointed error, GGUF parse, architecture, layout version.
pub(crate) fn open(bytes: &[u8], old_magic: &[u8; 4], arch: &str) -> Result<Gguf> {
    if bytes.starts_with(old_magic) {
        return Err(bad(format!(
            "pre-GGUF checkpoint ({}); it cannot be resumed — retrain or start a new checkpoint",
            String::from_utf8_lossy(old_magic)
        )));
    }
    // ponytail: parse() copies the bytes (peak memory ~3x the file); stream the
    // tensors via Gguf::open if checkpoints ever outgrow RAM.
    let g = Gguf::parse(bytes.to_vec())?;
    match g.architecture() {
        Some(a) if a == arch => {}
        other => {
            return Err(bad(format!(
                "not a {arch} checkpoint (general.architecture = {other:?})"
            )))
        }
    }
    let version = get_usize(&g, "version")?;
    if version != VERSION as usize {
        return Err(bad(format!("unsupported checkpoint version {version}")));
    }
    // Every tensor must lie inside the file, so file size bounds all later
    // sizing (checked arithmetic inside `tensor_bytes`).
    for t in &g.tensors {
        g.tensor_bytes(t)?;
    }
    Ok(g)
}

fn missing(key: &str) -> InferError {
    bad(format!(
        "checkpoint key 'ferrum.ckpt.{key}' missing or of the wrong type"
    ))
}

pub(crate) fn get_usize(g: &Gguf, name: &str) -> Result<usize> {
    g.meta(&format!("ferrum.ckpt.{name}"))
        .and_then(MetaValue::as_usize)
        .ok_or_else(|| missing(name))
}

pub(crate) fn get_u64(g: &Gguf, name: &str) -> Result<u64> {
    match g.meta(&format!("ferrum.ckpt.{name}")) {
        Some(MetaValue::U64(v)) => Ok(*v),
        _ => Err(missing(name)),
    }
}

pub(crate) fn get_bool(g: &Gguf, name: &str) -> Result<bool> {
    match g.meta(&format!("ferrum.ckpt.{name}")) {
        Some(MetaValue::Bool(v)) => Ok(*v),
        _ => Err(missing(name)),
    }
}

/// Tensor `name` as exactly `n` F32 values (checks type and shape).
pub(crate) fn get_f32(g: &Gguf, name: &str, n: usize) -> Result<Vec<f32>> {
    let t = g
        .tensor(name)
        .ok_or_else(|| bad(format!("checkpoint is missing tensor '{name}'")))?;
    if t.ggml_type != GGML_F32 || t.dims != [n as u64] {
        return Err(bad(format!(
            "checkpoint tensor '{name}' has dims {:?} (type {}), expected {n} F32 values",
            t.dims, t.ggml_type
        )));
    }
    g.dequantize(t)
}

/// Test helper: re-emit a GGUF container after `edit` has mutated its
/// metadata, optionally dropping one tensor. Tensors are copied as F32.
#[cfg(test)]
pub(crate) fn rewrite(
    bytes: &[u8],
    edit: impl FnOnce(&mut std::collections::BTreeMap<String, MetaValue>),
    drop_tensor: Option<&str>,
) -> Vec<u8> {
    let mut g = Gguf::parse(bytes.to_vec()).unwrap();
    edit(&mut g.metadata);
    let mut b = GgufBuilder::new();
    for (k, v) in &g.metadata {
        b.meta(k, v.clone());
    }
    for t in g
        .tensors
        .iter()
        .filter(|t| Some(t.name.as_str()) != drop_tensor)
    {
        let data = g.dequantize(t).unwrap();
        let raw = data.iter().flat_map(|x| x.to_le_bytes()).collect();
        b.tensor(&t.name, &t.dims, GGML_F32, raw);
    }
    b.into_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    const ARCH: &str = "ferrum-test-checkpoint";

    fn sample() -> Vec<u8> {
        let mut b = begin(ARCH);
        put_u32(&mut b, "dim", 7);
        put_u64(&mut b, "step", u64::MAX - 1);
        put_bool(&mut b, "flag", true);
        put_param(
            &mut b,
            0,
            &[1.0, -2.5],
            &[f32::MIN_POSITIVE, 0.0],
            &[3.0, 4.0],
        );
        b.into_bytes()
    }

    #[test]
    fn roundtrip_scalars_and_bit_exact_tensors() {
        let g = open(&sample(), b"OLDM", ARCH).unwrap();
        assert_eq!(g.architecture(), Some(ARCH));
        assert_eq!(get_usize(&g, "dim").unwrap(), 7);
        assert_eq!(get_u64(&g, "step").unwrap(), u64::MAX - 1);
        assert!(get_bool(&g, "flag").unwrap());
        assert_eq!(get_f32(&g, "p0.data", 2).unwrap(), vec![1.0, -2.5]);
        let m = get_f32(&g, "p0.m", 2).unwrap();
        assert_eq!(m[0].to_bits(), f32::MIN_POSITIVE.to_bits());
        assert_eq!(get_f32(&g, "p0.v", 2).unwrap(), vec![3.0, 4.0]);
    }

    #[test]
    fn old_magic_is_a_pointed_error() {
        let e = open(b"OLDM\x01\x00\x00\x00", b"OLDM", ARCH).err().unwrap();
        assert!(e.to_string().contains("pre-GGUF checkpoint (OLDM)"), "{e}");
        assert!(e.to_string().contains("retrain"), "{e}");
    }

    #[test]
    fn foreign_and_truncated_input_error() {
        assert!(open(b"", b"OLDM", ARCH).is_err());
        assert!(open(b"not a checkpoint at all", b"OLDM", ARCH).is_err());
        let bytes = sample();
        assert!(open(&bytes[..bytes.len() / 2], b"OLDM", ARCH).is_err());
    }

    #[test]
    fn wrong_architecture_and_missing_architecture() {
        let e = open(&sample(), b"OLDM", "other").err().unwrap();
        assert!(e.to_string().contains("not a other checkpoint"), "{e}");
        let mut b = GgufBuilder::new();
        b.meta("ferrum.ckpt.version", MetaValue::U32(1));
        let e = open(&b.into_bytes(), b"OLDM", ARCH).err().unwrap();
        assert!(e.to_string().contains("None"), "{e}");
    }

    #[test]
    fn version_is_checked() {
        let mut b = GgufBuilder::new();
        b.meta("general.architecture", MetaValue::String(ARCH.into()));
        b.meta("ferrum.ckpt.version", MetaValue::U32(2));
        let e = open(&b.into_bytes(), b"OLDM", ARCH).err().unwrap();
        assert!(
            e.to_string().contains("unsupported checkpoint version 2"),
            "{e}"
        );
        let mut b = GgufBuilder::new();
        b.meta("general.architecture", MetaValue::String(ARCH.into()));
        let e = open(&b.into_bytes(), b"OLDM", ARCH).err().unwrap();
        assert!(e.to_string().contains("ferrum.ckpt.version"), "{e}");
    }

    #[test]
    fn scalar_getters_reject_missing_and_wrongly_typed_keys() {
        let g = open(&sample(), b"OLDM", ARCH).unwrap();
        assert!(get_usize(&g, "absent").is_err());
        assert!(get_u64(&g, "absent").is_err());
        assert!(get_u64(&g, "dim").is_err()); // U32, not U64
        assert!(get_bool(&g, "absent").is_err());
        assert!(get_bool(&g, "dim").is_err());
        assert!(get_usize(&g, "flag").is_err()); // Bool is not an integer
    }

    #[test]
    fn tensor_getter_checks_presence_shape_and_type() {
        let g = open(&sample(), b"OLDM", ARCH).unwrap();
        let e = get_f32(&g, "p9.data", 2).unwrap_err();
        assert!(e.to_string().contains("missing tensor 'p9.data'"), "{e}");
        let e = get_f32(&g, "p0.data", 3).unwrap_err();
        assert!(e.to_string().contains("expected 3 F32"), "{e}");
        // Right element count, wrong type (Q8_0 = 8): rejected, not dequantized.
        let mut b = begin(ARCH);
        b.tensor("q", &[32], 8, vec![0u8; 34]);
        let g = open(&b.into_bytes(), b"OLDM", ARCH).unwrap();
        assert!(get_f32(&g, "q", 32).is_err());
    }

    #[test]
    fn rewrite_edits_metadata_and_drops_tensors() {
        let out = rewrite(
            &sample(),
            |m| {
                m.insert("ferrum.ckpt.dim".into(), MetaValue::U32(9));
            },
            Some("p0.v"),
        );
        let g = open(&out, b"OLDM", ARCH).unwrap();
        assert_eq!(get_usize(&g, "dim").unwrap(), 9);
        assert_eq!(get_f32(&g, "p0.data", 2).unwrap(), vec![1.0, -2.5]);
        assert!(get_f32(&g, "p0.v", 2).is_err());
    }

    #[test]
    fn open_rejects_tensors_past_eof_without_allocating() {
        // A tiny file whose directory claims enormous tensors.
        for dims in [vec![1u64 << 40], vec![u64::MAX], vec![1 << 33, 1 << 33]] {
            let mut b = begin(ARCH);
            b.tensor("p0.data", &dims, GGML_F32, vec![0u8; 4]);
            assert!(open(&b.into_bytes(), b"OLDM", ARCH).is_err(), "{dims:?}");
        }
        // Dropping the tail of a valid file also puts a tensor past EOF.
        let bytes = sample();
        assert!(open(&bytes[..bytes.len() - 4], b"OLDM", ARCH).is_err());
    }

    #[test]
    fn empty_tensor_roundtrips() {
        let mut b = begin(ARCH);
        put_param(&mut b, 0, &[], &[], &[]);
        let g = open(&b.into_bytes(), b"OLDM", ARCH).unwrap();
        assert!(get_f32(&g, "p0.data", 0).unwrap().is_empty());
    }
}
