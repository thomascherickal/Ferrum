//! FINF binary format v3: weights + normalizer + ModelMetadata JSON.
//!
//! Layout (all integers little-endian):
//!   4 bytes  b"FINF"
//!   u32      version = 3
//!   u32      norm_len;    [bytes] normalizer string
//!   u32      meta_len;    [bytes] ModelMetadata JSON
//!   u32      num_layers
//!   per layer: u8 tag, then layer bytes
use crate::activation::Activation;
use crate::csv::{ModelMetadata, Normalizer};
use crate::error::{InferError, Result};
use crate::layer::{ActivationLayer, Linear};
use crate::model::Sequential;

const MAGIC: &[u8; 4] = b"FINF";
const VERSION: u32 = 3;
const TAG_LINEAR: u8 = 0;
const TAG_ACTIVATION: u8 = 1;

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}
impl<'a> Reader<'a> {
    fn new(b: &'a [u8]) -> Self {
        Self { bytes: b, pos: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos + n;
        if end > self.bytes.len() {
            return Err(InferError::Format(format!(
                "EOF at +{}: need {n}",
                self.pos
            )));
        }
        let s = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn f32(&mut self) -> Result<f32> {
        let b = self.take(4)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn f32_vec(&mut self, n: usize) -> Result<Vec<f32>> {
        (0..n).map(|_| self.f32()).collect()
    }
    fn utf8(&mut self, n: usize) -> Result<&'a str> {
        std::str::from_utf8(self.take(n)?)
            .map_err(|_| InferError::Format("invalid UTF-8 in blob".into()))
    }
}

pub fn to_bytes(model: &Sequential, norm: &Normalizer, meta: &ModelMetadata) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());

    let norm_s = norm.encode();
    out.extend_from_slice(&(norm_s.len() as u32).to_le_bytes());
    out.extend_from_slice(norm_s.as_bytes());

    let meta_s = meta.to_json();
    out.extend_from_slice(&(meta_s.len() as u32).to_le_bytes());
    out.extend_from_slice(meta_s.as_bytes());

    out.extend_from_slice(&(model.len() as u32).to_le_bytes());
    for layer in model.layers() {
        let any = layer.as_any();
        if let Some(lin) = any.downcast_ref::<Linear>() {
            out.push(TAG_LINEAR);
            out.extend_from_slice(&(lin.in_features() as u32).to_le_bytes());
            out.extend_from_slice(&(lin.out_features() as u32).to_le_bytes());
            for x in &lin.weight.data {
                out.extend_from_slice(&x.to_le_bytes());
            }
            for x in &lin.bias.data {
                out.extend_from_slice(&x.to_le_bytes());
            }
        } else if let Some(act) = any.downcast_ref::<ActivationLayer>() {
            out.push(TAG_ACTIVATION);
            out.push(act.kind.tag());
        } else {
            return Err(InferError::Format(format!(
                "unknown layer: {}",
                layer.name()
            )));
        }
    }
    Ok(out)
}

pub fn from_bytes(bytes: &[u8]) -> Result<(Sequential, Normalizer, ModelMetadata)> {
    let mut r = Reader::new(bytes);
    if r.take(4)? != MAGIC {
        return Err(InferError::Format("bad FINF magic".into()));
    }
    let ver = r.u32()?;
    if ver != VERSION {
        return Err(InferError::Format(format!(
            "unsupported FINF v{ver} (need v{VERSION})"
        )));
    }
    let norm_len = r.u32()? as usize;
    let norm = Normalizer::decode(r.utf8(norm_len)?)?;
    let meta_len = r.u32()? as usize;
    let meta = ModelMetadata::from_json(r.utf8(meta_len)?)?;

    let num_layers = r.u32()? as usize;
    let mut model = Sequential::new();
    for _ in 0..num_layers {
        match r.u8()? {
            TAG_LINEAR => {
                let in_f = r.u32()? as usize;
                let out_f = r.u32()? as usize;
                model.push(Box::new(Linear::new(
                    in_f,
                    out_f,
                    r.f32_vec(in_f * out_f)?,
                    r.f32_vec(out_f)?,
                )?));
            }
            TAG_ACTIVATION => {
                let t = r.u8()?;
                model.push(Box::new(ActivationLayer::new(
                    Activation::from_tag(t)
                        .ok_or_else(|| InferError::Format(format!("bad act tag {t}")))?,
                )));
            }
            t => return Err(InferError::Format(format!("bad layer tag {t}"))),
        }
    }
    Ok((model, norm, meta))
}

pub fn save(model: &Sequential, norm: &Normalizer, meta: &ModelMetadata, path: &str) -> Result<()> {
    std::fs::write(path, to_bytes(model, norm, meta)?)?;
    Ok(())
}
pub fn load(path: &str) -> Result<(Sequential, Normalizer, ModelMetadata)> {
    from_bytes(&std::fs::read(path)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::Activation;
    use crate::csv::TaskType;
    use crate::layer::ActivationLayer;
    use crate::tensor::Tensor;

    fn make_bundle() -> (Sequential, Normalizer, ModelMetadata) {
        let l1 = Linear::new(
            4,
            8,
            (0..32).map(|i| i as f32 * 0.01).collect(),
            vec![0.0; 8],
        )
        .unwrap();
        let l2 = Linear::new(
            8,
            3,
            (0..24).map(|i| i as f32 * -0.01).collect(),
            vec![0.1; 3],
        )
        .unwrap();
        let model = Sequential::new()
            .with(Box::new(l1))
            .with(Box::new(ActivationLayer::new(Activation::ReLU)))
            .with(Box::new(l2))
            .with(Box::new(ActivationLayer::new(Activation::Softmax)));
        let norm = Normalizer {
            means: vec![5.8, 3.1, 3.7, 1.2],
            stds: vec![0.8, 0.4, 1.7, 0.8],
        };
        let meta = ModelMetadata {
            dataset_name: "test".into(),
            task: TaskType::Classification,
            feature_names: vec!["a".into(), "b".into(), "c".into(), "d".into()],
            feature_ranges: vec![[0.0, 10.0]; 4],
            class_names: vec!["X".into(), "Y".into(), "Z".into()],
            target_name: "".into(),
            target_range: [0.0, 2.0],
            input_dim: 4,
            output_dim: 3,
        };
        (model, norm, meta)
    }

    #[test]
    fn roundtrip_preserves_outputs() {
        let (model, norm, meta) = make_bundle();
        let raw = Tensor::row(vec![5.1f32, 3.5, 1.4, 0.2]).unwrap();
        let before = model.forward(&norm.transform(&raw).unwrap()).unwrap();
        let bytes = to_bytes(&model, &norm, &meta).unwrap();
        let (m2, n2, _) = from_bytes(&bytes).unwrap();
        let after = m2.forward(&n2.transform(&raw).unwrap()).unwrap();
        for (a, b) in before.data.iter().zip(&after.data) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn metadata_survives_roundtrip() {
        let (model, norm, meta) = make_bundle();
        let bytes = to_bytes(&model, &norm, &meta).unwrap();
        let (_, _, m2) = from_bytes(&bytes).unwrap();
        assert_eq!(m2.task, TaskType::Classification);
        assert_eq!(m2.class_names, vec!["X", "Y", "Z"]);
        assert_eq!(m2.feature_names, vec!["a", "b", "c", "d"]);
    }

    #[test]
    fn bad_magic_errors() {
        assert!(matches!(
            from_bytes(b"XXXX\x03\x00\x00\x00"),
            Err(InferError::Format(_))
        ));
    }

    #[test]
    fn wrong_version_errors() {
        let (m, n, meta) = make_bundle();
        let mut bytes = to_bytes(&m, &n, &meta).unwrap();
        bytes[4] = 99;
        bytes[5] = 0;
        bytes[6] = 0;
        bytes[7] = 0;
        assert!(matches!(from_bytes(&bytes), Err(InferError::Format(_))));
    }

    #[test]
    fn truncation_errors() {
        let (m, n, meta) = make_bundle();
        let mut bytes = to_bytes(&m, &n, &meta).unwrap();
        bytes.truncate(bytes.len() - 8);
        assert!(from_bytes(&bytes).is_err());
    }
}
