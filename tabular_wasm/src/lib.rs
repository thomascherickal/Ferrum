//! WASM bindings for Ferrum — tabular ML + Transformer SLM.
//!
//! Exposes two structs to JavaScript:
//!   `TabularModel`       — original tabular inference (classification / regression)
//!   `TransformerSLMModel`— character-level SLM with attention map access

use ferrum_core::{argmax_rows, from_bytes, TaskType};
use ferrum_core::layer::TransformerBlock;
use wasm_bindgen::prelude::*;

// ─────────────────────────────────────────────────────────────────────────────
// TabularModel (unchanged from original)
// ─────────────────────────────────────────────────────────────────────────────

#[wasm_bindgen]
pub struct TabularModel {
    model: ferrum_core::Sequential,
    norm: ferrum_core::Normalizer,
    meta_json: String,
    task: TaskType,
}

#[wasm_bindgen]
impl TabularModel {
    #[wasm_bindgen(constructor)]
    pub fn new(bytes: &[u8]) -> Result<TabularModel, JsValue> {
        let (model, norm, meta) =
            from_bytes(bytes).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let task = meta.task;
        let meta_json = meta.to_json();
        Ok(Self { model, norm, meta_json, task })
    }

    pub fn metadata(&self) -> String {
        self.meta_json.clone()
    }

    pub fn norm_encoded(&self) -> String {
        self.norm.encode()
    }

    pub fn predict(&self, values: &[f32]) -> Result<String, JsValue> {
        let raw = ferrum_core::Tensor::row(values.to_vec())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let input = self.norm.transform(&raw)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let out = self.model.forward(&input)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let json = match self.task {
            TaskType::Classification => {
                let class_idx =
                    argmax_rows(&out).map_err(|e| JsValue::from_str(&e.to_string()))?[0];
                let confidence = out.data[class_idx];
                let probs = out.data.iter().map(|p| format!("{p:.6}")).collect::<Vec<_>>().join(",");
                format!(r#"{{"type":"classification","class_index":{class_idx},"confidence":{confidence:.6},"probabilities":[{probs}]}}"#)
            }
            TaskType::Regression => {
                let pred_norm = out.data[0];
                let pred_raw = self.norm.denormalise_target(pred_norm);
                format!(r#"{{"type":"regression","value":{pred_raw:.4},"value_norm":{pred_norm:.6}}}"#)
            }
            TaskType::TransformerSLM => {
                format!(r#"{{"type":"error","message":"Use TransformerSLMModel for SLM inference"}}"#)
            }
        };
        Ok(json)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TransformerSLMModel
// ─────────────────────────────────────────────────────────────────────────────

/// Edge Small Language Model — causal Transformer running in WASM.
///
/// JavaScript usage:
/// ```js
/// const resp = await fetch('model.bin');
/// const bytes = new Uint8Array(await resp.arrayBuffer());
/// const slm = new TransformerSLMModel(bytes);
///
/// const meta = JSON.parse(slm.metadata());
/// const vocab = meta.class_names;           // idx → char mapping
/// const contextLen = meta.input_dim;
///
/// // Feed a context of integer token IDs
/// const context = new Float32Array([12, 5, 3, ...]);
/// const probs = slm.predict_next(context);  // Float32Array length vocab_size
///
/// // After inference, read attention weights for visualization
/// const attn = slm.get_last_attention_weights(); // Float32Array [heads × T × T]
/// const numHeads = slm.num_heads();
/// const contextLength = slm.context_len();
/// ```
#[wasm_bindgen]
pub struct TransformerSLMModel {
    model: ferrum_core::Sequential,
    meta_json: String,
    vocab_size: usize,
    context_len: usize,
    num_heads: usize,
    num_blocks: usize,
}

#[wasm_bindgen]
impl TransformerSLMModel {
    /// Construct from FINF v4 bytes.
    #[wasm_bindgen(constructor)]
    pub fn new(bytes: &[u8]) -> Result<TransformerSLMModel, JsValue> {
        let (model, _norm, meta) =
            from_bytes(bytes).map_err(|e| JsValue::from_str(&e.to_string()))?;

        if meta.task != TaskType::TransformerSLM {
            return Err(JsValue::from_str("Expected TransformerSLM task type"));
        }

        let vocab_size = meta.output_dim;
        let context_len = meta.input_dim;

        // Count transformer blocks and extract num_heads from first found block
        let mut num_blocks = 0usize;
        let mut num_heads = 4usize; // default
        for layer in model.layers() {
            if let Some(tb) = layer.as_any().downcast_ref::<TransformerBlock>() {
                if num_blocks == 0 {
                    num_heads = tb.num_heads();
                }
                num_blocks += 1;
            }
        }

        let meta_json = meta.to_json();
        Ok(Self { model, meta_json, vocab_size, context_len, num_heads, num_blocks })
    }

    /// Return model metadata JSON (vocab, context length, architecture details).
    pub fn metadata(&self) -> String {
        self.meta_json.clone()
    }

    /// Number of attention heads.
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Vocabulary size (number of distinct tokens).
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Context length (number of characters the model sees at once).
    pub fn context_len(&self) -> usize {
        self.context_len
    }

    /// Number of Transformer blocks.
    pub fn num_layers(&self) -> usize {
        self.num_blocks
    }

    /// Run inference on a context of token indices.
    ///
    /// `context` must have exactly `context_len` elements (Float32Array of usize-as-f32).
    /// Returns a Float32Array of length `vocab_size` with next-token probabilities.
    pub fn predict_next(&self, context: &[f32]) -> Result<Vec<f32>, JsValue> {
        if context.len() != self.context_len {
            return Err(JsValue::from_str(&format!(
                "Context must have exactly {} elements, got {}",
                self.context_len,
                context.len()
            )));
        }

        let input = ferrum_core::Tensor::matrix(1, self.context_len, context.to_vec())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let out = self.model.forward(&input)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        // out is [1 * T, vocab] from the embedding chain, then [1, vocab] after lm_head.
        // We want the last row (the next-token prediction).
        let (rows, cols) = out.matrix_dims()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let last_row_start = (rows - 1) * cols;
        Ok(out.data[last_row_start..].to_vec())
    }

    /// Return the self-attention weights from the LAST Transformer block's last forward pass.
    ///
    /// Returns a flat Float32Array of shape [num_heads × context_len × context_len].
    /// For `num_heads=4` and `context_len=32`, this is 4096 floats.
    ///
    /// In JavaScript, index as: `attn[h * T * T + i * T + j]` where h=head, i=query pos, j=key pos.
    pub fn get_last_attention_weights(&self) -> Vec<f32> {
        // Walk the layer list to find the last TransformerBlock
        for layer in self.model.layers().iter().rev() {
            if let Some(tb) = layer.as_any().downcast_ref::<TransformerBlock>() {
                return tb.last_attention.borrow().clone();
            }
        }
        vec![]
    }

    /// Sample a next token index from probabilities using temperature scaling.
    /// `probs` = output of `predict_next`. `temperature` = 1.0 is neutral.
    /// Lower temperature → more deterministic. Higher → more random.
    pub fn sample_from_probs(&self, probs: &[f32], temperature: f32, random_value: f32) -> usize {
        let scaled: Vec<f32> = probs.iter()
            .map(|&p| (p.ln() / temperature.max(0.01)).exp())
            .collect();
        let sum: f32 = scaled.iter().sum();
        let normed: Vec<f32> = scaled.iter().map(|&v| v / sum).collect();
        let mut cumsum = 0.0f32;
        let r = random_value.clamp(0.0, 1.0 - 1e-7);
        for (i, &p) in normed.iter().enumerate() {
            cumsum += p;
            if r <= cumsum { return i; }
        }
        normed.len() - 1
    }

    /// Compute Shannon entropy of a probability distribution (0 = certain, ln(V) = uniform).
    pub fn entropy(&self, probs: &[f32]) -> f32 {
        probs.iter()
            .filter(|&&p| p > 1e-10)
            .map(|&p| -p * p.ln())
            .sum()
    }

    /// Return the top-k token indices sorted by probability (descending).
    pub fn top_k_indices(&self, probs: &[f32], k: usize) -> Vec<usize> {
        let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
        indexed.iter().take(k).map(|&(i, _)| i).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use ferrum_core::{
        from_bytes, to_bytes, Embedding, LayerNorm, Linear, ModelMetadata, Net, Normalizer,
        Rng, TaskType, Tensor, TransformerBlock,
    };
    use ferrum_core::activation::{Activation};
    use ferrum_core::layer::ActivationLayer;
    use ferrum_core::model::Sequential;

    fn clf_bytes() -> Vec<u8> {
        let mut rng = Rng::new(1);
        let net = Net::mlp(4, 8, 3, &mut rng);
        let model = net.to_inference().unwrap();
        let norm = Normalizer { means: vec![5.8, 3.0, 3.7, 1.2], stds: vec![0.8, 0.4, 1.7, 0.8] };
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

    fn slm_bytes() -> Vec<u8> {
        let vocab_size = 10;
        let context_len = 8;
        let embed_dim = 16;
        let num_heads = 2;
        let hidden_dim = 32;
        let mut rng = Rng::new(99);
        let scale = (1.0 / embed_dim as f32).sqrt();

        let emb = Embedding::new(
            vocab_size, context_len, embed_dim,
            (0..vocab_size * embed_dim).map(|_| rng.next_normal() * scale).collect(),
            (0..context_len * embed_dim).map(|_| rng.next_normal() * scale).collect(),
        ).unwrap();

        let tb = TransformerBlock::new(
            context_len, num_heads, embed_dim,
            vec![1.0; embed_dim], vec![0.0; embed_dim],
            (0..embed_dim*embed_dim).map(|_| rng.next_normal() * scale).collect(), vec![0.0; embed_dim],
            (0..embed_dim*embed_dim).map(|_| rng.next_normal() * scale).collect(), vec![0.0; embed_dim],
            (0..embed_dim*embed_dim).map(|_| rng.next_normal() * scale).collect(), vec![0.0; embed_dim],
            (0..embed_dim*embed_dim).map(|_| rng.next_normal() * scale).collect(), vec![0.0; embed_dim],
            vec![1.0; embed_dim], vec![0.0; embed_dim],
            (0..embed_dim*hidden_dim).map(|_| rng.next_normal() * scale).collect(), vec![0.0; hidden_dim],
            (0..hidden_dim*embed_dim).map(|_| rng.next_normal() * scale).collect(), vec![0.0; embed_dim],
        ).unwrap();

        let ln_final = LayerNorm::new(embed_dim, vec![1.0; embed_dim], vec![0.0; embed_dim]).unwrap();
        let lm_head = Linear::new(embed_dim, vocab_size,
            (0..embed_dim * vocab_size).map(|_| rng.next_normal() * scale).collect(),
            vec![0.0; vocab_size],
        ).unwrap();

        let model = Sequential::new()
            .with(Box::new(emb))
            .with(Box::new(tb))
            .with(Box::new(ln_final))
            .with(Box::new(lm_head));

        let vocab_strs: Vec<String> = (0..vocab_size).map(|i| ((b'a' + i as u8) as char).to_string()).collect();
        let norm = Normalizer { means: vec![], stds: vec![] };
        let meta = ModelMetadata {
            dataset_name: "slm_test".into(),
            task: TaskType::TransformerSLM,
            feature_names: (0..context_len).map(|i| format!("c_{i}")).collect(),
            feature_ranges: vec![[0.0, vocab_size as f32]; context_len],
            class_names: vocab_strs,
            target_name: "next_char".into(),
            target_range: [0.0, vocab_size as f32],
            input_dim: context_len,
            output_dim: vocab_size,
        };
        to_bytes(&model, &norm, &meta).unwrap()
    }

    #[test]
    fn slm_loads_and_predicts() {
        let bytes = slm_bytes();
        let (model, _norm, meta) = from_bytes(&bytes).unwrap();
        assert_eq!(meta.task, TaskType::TransformerSLM);
        let context: Vec<f32> = (0..8).map(|i| (i % 10) as f32).collect();
        let x = Tensor::matrix(1, 8, context).unwrap();
        let out = model.forward(&x).unwrap();
        // Should produce [1, vocab_size] or [T, vocab] — check it's finite
        assert!(out.data.iter().all(|v| v.is_finite()));
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
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn slm_metadata_has_vocab() {
        let bytes = slm_bytes();
        let (_model, _norm, meta) = from_bytes(&bytes).unwrap();
        assert_eq!(meta.class_names.len(), 10);
        assert_eq!(meta.class_names[0], "a");
        assert_eq!(meta.input_dim, 8);
        assert_eq!(meta.output_dim, 10);
    }

    #[test]
    fn corrupt_bytes_error_gracefully() {
        assert!(from_bytes(b"CORRUPT").is_err());
        assert!(from_bytes(&[]).is_err());
    }
}
