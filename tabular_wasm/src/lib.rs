//! Generic tabular WASM bindings. One model file handles any CSV dataset.
//! JavaScript calls `new TabularModel(bytes)` then `predict(f32array)`.

use ferrum_core::{argmax_rows, from_bytes, TaskType};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct TabularModel {
    model: ferrum_core::Sequential,
    norm: ferrum_core::Normalizer,
    meta_json: String,
    task: TaskType,
}

#[wasm_bindgen]
impl TabularModel {
    /// Construct from FINF v3 bytes (call once after fetch).
    #[wasm_bindgen(constructor)]
    pub fn new(bytes: &[u8]) -> Result<TabularModel, JsValue> {
        let (model, norm, meta) =
            from_bytes(bytes).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let task = meta.task;
        let meta_json = meta.to_json();
        Ok(Self {
            model,
            norm,
            meta_json,
            task,
        })
    }

    /// Return the ModelMetadata as a JSON string so JS can build the UI.
    pub fn metadata(&self) -> String {
        self.meta_json.clone()
    }

    /// Return the normaliser encoded string ("mean0,std0;mean1,std1;…") so that
    /// JavaScript can reconstruct per-feature z-scores for the statistics panes.
    pub fn norm_encoded(&self) -> String {
        self.norm.encode()
    }

    /// Run inference on one row. `values` must have length == input_dim.
    /// Returns JSON: { "prediction": ..., "confidence": ..., "probabilities": [...] }
    pub fn predict(&self, values: &[f32]) -> Result<String, JsValue> {
        let raw = ferrum_core::Tensor::row(values.to_vec())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let input = self
            .norm
            .transform(&raw)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let out = self
            .model
            .forward(&input)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let json = match self.task {
            TaskType::Classification => {
                let class_idx =
                    argmax_rows(&out).map_err(|e| JsValue::from_str(&e.to_string()))?[0];
                let confidence = out.data[class_idx];
                let probs = out
                    .data
                    .iter()
                    .map(|p| format!("{p:.6}"))
                    .collect::<Vec<_>>()
                    .join(",");
                format!(
                    r#"{{"type":"classification","class_index":{class_idx},"confidence":{confidence:.6},"probabilities":[{probs}]}}"#
                )
            }
            TaskType::Regression => {
                let pred_norm = out.data[0];
                let pred_raw = self.norm.denormalise_target(pred_norm);
                format!(
                    r#"{{"type":"regression","value":{pred_raw:.4},"value_norm":{pred_norm:.6}}}"#
                )
            }
        };
        Ok(json)
    }
}

// ── Pure-Rust tests (run with `cargo test -p tabular_wasm`) ──────────────────
#[cfg(test)]
mod tests {
    use ferrum_core::{
        from_bytes, to_bytes, ModelMetadata, Net, Normalizer, Rng, TaskType, Tensor,
    };

    fn clf_bytes() -> Vec<u8> {
        let mut rng = Rng::new(1);
        let net = Net::mlp(4, 8, 3, &mut rng);
        let model = net.to_inference().unwrap();
        let norm = Normalizer {
            means: vec![5.8, 3.0, 3.7, 1.2],
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
        to_bytes(&model, &norm, &meta).unwrap()
    }

    fn reg_bytes() -> Vec<u8> {
        let mut rng = Rng::new(2);
        let net = Net::mlp(2, 4, 1, &mut rng);
        let model = net.to_inference_task(TaskType::Regression).unwrap();
        // 2 features + 1 target = 3 entries in normalizer
        let norm = Normalizer {
            means: vec![3.0, 4.0, 300000.0],
            stds: vec![1.5, 2.0, 100000.0],
        };
        let meta = ModelMetadata {
            dataset_name: "reg_test".into(),
            task: TaskType::Regression,
            feature_names: vec!["x1".into(), "x2".into()],
            feature_ranges: vec![[0.0, 10.0], [0.0, 10.0]],
            class_names: vec![],
            target_name: "price".into(),
            target_range: [100000.0, 500000.0],
            input_dim: 2,
            output_dim: 1,
        };
        to_bytes(&model, &norm, &meta).unwrap()
    }

    #[test]
    fn classification_loads_and_predicts() {
        let bytes = clf_bytes();
        let (model, norm, _meta) = from_bytes(&bytes).unwrap();
        let raw = Tensor::row(vec![5.1f32, 3.5, 1.4, 0.2]).unwrap();
        let input = norm.transform(&raw).unwrap();
        let out = model.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
        let sum: f32 = out.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax sum={sum}");
    }

    #[test]
    fn regression_loads_and_predicts() {
        let bytes = reg_bytes();
        let (model, norm, meta) = from_bytes(&bytes).unwrap();
        assert_eq!(meta.task, TaskType::Regression);
        let raw = Tensor::row(vec![3.0f32, 4.0]).unwrap();
        let input = norm.transform(&raw).unwrap();
        let out = model.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 1]);
        // Denormalise should return a finite number in a plausible range
        let pred = norm.denormalise_target(out.data[0]);
        assert!(pred.is_finite(), "prediction is NaN or Inf");
    }

    #[test]
    fn metadata_json_has_correct_fields() {
        let bytes = clf_bytes();
        let (_, _, meta) = from_bytes(&bytes).unwrap();
        let json = meta.to_json();
        assert!(json.contains("\"task\":\"classification\""));
        assert!(json.contains("\"dataset_name\":\"test\""));
        assert!(json.contains("\"input_dim\":4"));
        assert!(json.contains("\"output_dim\":3"));
    }

    #[test]
    fn corrupt_bytes_error_gracefully() {
        assert!(from_bytes(b"CORRUPT").is_err());
        assert!(from_bytes(&[]).is_err());
    }

    #[test]
    fn batch_and_single_agree() {
        let bytes = clf_bytes();
        let (model, norm, _) = from_bytes(&bytes).unwrap();
        let samples = vec![
            vec![5.1f32, 3.5, 1.4, 0.2],
            vec![6.4, 3.2, 4.5, 1.5],
            vec![6.3, 3.3, 6.0, 2.5],
        ];
        // individual
        let individual: Vec<usize> = samples
            .iter()
            .map(|f| {
                let p = model
                    .forward(&norm.transform(&Tensor::row(f.clone()).unwrap()).unwrap())
                    .unwrap();
                ferrum_core::argmax_rows(&p).unwrap()[0]
            })
            .collect();
        // batch
        let mut all = Vec::new();
        for s in &samples {
            all.extend_from_slice(s);
        }
        let batch_in = norm.transform(&Tensor::matrix(3, 4, all).unwrap()).unwrap();
        let batch_out = model.forward(&batch_in).unwrap();
        let batch_pred = ferrum_core::argmax_rows(&batch_out).unwrap();
        assert_eq!(individual, batch_pred);
    }

    #[test]
    fn all_five_model_files_load() {
        // Load each trained model if it exists (skips in CI if not trained yet)
        let paths = [
            "../web/datasets/iris/model.bin",
            "../web/datasets/wine/model.bin",
            "../web/datasets/diabetes/model.bin",
            "../web/datasets/titanic/model.bin",
            "../web/datasets/housing/model.bin",
        ];
        for path in &paths {
            if let Ok(bytes) = std::fs::read(path) {
                let result = from_bytes(&bytes);
                assert!(result.is_ok(), "failed to load {path}: {:?}", result.err());
                let (_, _, meta) = result.unwrap();
                println!("{path}: {} / {:?}", meta.dataset_name, meta.task);
            }
        }
    }
}
