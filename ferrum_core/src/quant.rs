//! Int8 quantization primitives shared by FINF serialization (post-training
//! quantization on save) and Quantization Aware Training (QAT).
//!
//! Quantization is symmetric per-tensor: `value ≈ i8 × scale` with
//! `scale = max|value| / 127`. During QAT, [`fake_quantize_int8`] snaps a
//! weight tensor onto that int8 grid in place (quantize → dequantize), so the
//! forward and backward passes see exactly the weights an int8-serialized
//! model will run with, while the optimizer keeps updating full-precision
//! master weights (straight-through estimator).

/// Tensors shorter than this stay f32 both in quantized FINF files and during
/// QAT: biases and LayerNorm parameters are small (no size win) and
/// accuracy-sensitive.
pub const QUANT_MIN_LEN: usize = 64;

/// The symmetric int8 scale for a tensor: `max|value| / 127`.
pub fn int8_scale(data: &[f32]) -> f32 {
    data.iter().fold(0.0f32, |m, &v| m.max(v.abs())) / 127.0
}

/// Snap `data` onto the symmetric int8 grid in place (quantize → dequantize).
///
/// Mirrors the FINF v5 serialization rules: tensors shorter than
/// [`QUANT_MIN_LEN`] or containing non-finite values are left untouched.
/// Returns `true` if the tensor was quantized.
pub fn fake_quantize_int8(data: &mut [f32]) -> bool {
    if data.len() < QUANT_MIN_LEN || !data.iter().all(|v| v.is_finite()) {
        return false;
    }
    let scale = int8_scale(data);
    if scale == 0.0 {
        return true; // all zeros: already on the grid
    }
    for v in data.iter_mut() {
        *v = (*v / scale).round().clamp(-127.0, 127.0) * scale;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fake_quantize_snaps_to_int8_grid() {
        let mut data: Vec<f32> = (0..128).map(|i| (i as f32 * 0.13).sin()).collect();
        assert!(fake_quantize_int8(&mut data));
        let scale = int8_scale(&data);
        for &v in &data {
            let q = v / scale;
            assert!(
                (q - q.round()).abs() < 1e-4,
                "{v} is not an int8 multiple of scale {scale}"
            );
        }
    }

    #[test]
    fn fake_quantize_is_idempotent() {
        let mut data: Vec<f32> = (0..100).map(|i| (i as f32 * 0.37).cos() * 2.0).collect();
        fake_quantize_int8(&mut data);
        let once = data.clone();
        fake_quantize_int8(&mut data);
        for (a, b) in once.iter().zip(&data) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn fake_quantize_bounds_error_by_half_scale() {
        let original: Vec<f32> = (0..256).map(|i| (i as f32 * 0.211).sin() * 0.5).collect();
        let mut data = original.clone();
        fake_quantize_int8(&mut data);
        let scale = int8_scale(&original);
        for (o, q) in original.iter().zip(&data) {
            assert!((o - q).abs() <= scale * 0.5 + 1e-6);
        }
    }

    #[test]
    fn short_and_nonfinite_tensors_untouched() {
        let mut short = vec![0.3f32; QUANT_MIN_LEN - 1];
        assert!(!fake_quantize_int8(&mut short));
        assert!(short.iter().all(|&v| v == 0.3));

        let mut bad = vec![1.0f32; QUANT_MIN_LEN];
        bad[7] = f32::NAN;
        assert!(!fake_quantize_int8(&mut bad));
    }

    #[test]
    fn all_zero_tensor_is_fine() {
        let mut zeros = vec![0.0f32; QUANT_MIN_LEN];
        assert!(fake_quantize_int8(&mut zeros));
        assert!(zeros.iter().all(|&v| v == 0.0));
    }
}
