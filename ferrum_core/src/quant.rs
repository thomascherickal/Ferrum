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

// ─────────────────────────────────────────────────────────────────────────────
// Per-channel quantization (§7)
// ─────────────────────────────────────────────────────────────────────────────
//
// Per-tensor quantization uses a single scale for the whole tensor, so one
// outlier weight inflates the scale and coarsens *every* value. Per-channel
// quantization splits the tensor into `channels` contiguous rows (for a
// row-major `[channels, row_len]` weight matrix this is one scale per row) so an
// outlier only coarsens its own channel. Storage cost is `channels` extra f32
// scales — negligible against `channels × row_len` int8 weights.

/// One symmetric int8 scale per contiguous channel (row) of `data`. `channels`
/// must be non-zero and divide `data.len()`.
pub fn int8_scales_per_channel(data: &[f32], channels: usize) -> Vec<f32> {
    debug_assert!(channels != 0 && data.len() % channels == 0);
    let row = data.len() / channels.max(1);
    data.chunks(row.max(1)).map(int8_scale).collect()
}

/// Per-channel counterpart of [`fake_quantize_int8`]: snap each of the
/// `channels` contiguous rows of `data` onto its own int8 grid, in place.
///
/// Whole-tensor guards match [`fake_quantize_int8`]: tensors shorter than
/// [`QUANT_MIN_LEN`], non-finite tensors, or a `channels` that does not divide
/// the length are left untouched. With `channels == 1` this is exactly the
/// per-tensor behaviour. Returns `true` if the tensor was quantized.
pub fn fake_quantize_int8_per_channel(data: &mut [f32], channels: usize) -> bool {
    if channels == 0 || data.is_empty() || data.len() % channels != 0 {
        return false;
    }
    if data.len() < QUANT_MIN_LEN || !data.iter().all(|v| v.is_finite()) {
        return false;
    }
    let row = data.len() / channels;
    for chunk in data.chunks_mut(row) {
        let scale = int8_scale(chunk);
        if scale > 0.0 {
            for v in chunk.iter_mut() {
                *v = (*v / scale).round().clamp(-127.0, 127.0) * scale;
            }
        }
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

    // ── Per-channel quantization (§7) ─────────────────────────────────────────

    #[test]
    fn per_channel_each_row_on_its_own_grid() {
        let channels = 4;
        let row = 32; // 4×32 = 128 ≥ QUANT_MIN_LEN
        let mut data: Vec<f32> = (0..channels * row).map(|i| (i as f32 * 0.13).sin()).collect();
        // Give one row a large outlier so its scale differs from the others.
        data[2 * row + 5] = 50.0;
        assert!(fake_quantize_int8_per_channel(&mut data, channels));

        let scales = int8_scales_per_channel(&data, channels);
        for (c, chunk) in data.chunks(row).enumerate() {
            for &v in chunk {
                let q = v / scales[c];
                assert!((q - q.round()).abs() < 1e-3, "row {c}: {v} off its grid");
            }
        }
        // The outlier row's scale is much larger than a neighbour's.
        assert!(scales[2] > scales[0] * 5.0, "outlier did not localise to its row");
    }

    #[test]
    fn per_channel_beats_per_tensor_with_a_localized_outlier() {
        // A weight matrix where one channel holds a large outlier. Per-tensor
        // quantization inflates the global scale and coarsens every channel;
        // per-channel confines the coarseness to the outlier's channel, so the
        // overall reconstruction error is much smaller.
        let channels = 8;
        let row = 32;
        let original: Vec<f32> = (0..channels * row)
            .map(|i| ((i as f32 * 0.07).sin()) * 0.5)
            .collect();
        let mut with_outlier = original.clone();
        with_outlier[3] = 40.0; // outlier in channel 0

        let mut per_tensor = with_outlier.clone();
        fake_quantize_int8(&mut per_tensor);
        let mut per_channel = with_outlier.clone();
        fake_quantize_int8_per_channel(&mut per_channel, channels);

        // Error measured over the clean channels (1..) where the two schemes
        // differ: per-tensor still carries the outlier-inflated scale there.
        let err = |q: &[f32]| -> f32 {
            with_outlier[row..]
                .iter()
                .zip(&q[row..])
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>()
        };
        let e_tensor = err(&per_tensor);
        let e_channel = err(&per_channel);
        assert!(e_channel * 5.0 < e_tensor,
            "per-channel ({e_channel:.4}) should be far below per-tensor ({e_tensor:.4})");
    }

    #[test]
    fn per_channel_single_channel_equals_per_tensor() {
        let mut a: Vec<f32> = (0..128).map(|i| (i as f32 * 0.21).cos()).collect();
        let mut b = a.clone();
        fake_quantize_int8(&mut a);
        fake_quantize_int8_per_channel(&mut b, 1);
        assert_eq!(a, b);
    }

    #[test]
    fn per_channel_guards_reject_bad_inputs() {
        let mut short = vec![0.5f32; QUANT_MIN_LEN - 1];
        assert!(!fake_quantize_int8_per_channel(&mut short, 1));
        let mut indivisible = vec![0.5f32; 100];
        assert!(!fake_quantize_int8_per_channel(&mut indivisible, 7)); // 100 % 7 ≠ 0
        let mut zero_ch = vec![0.5f32; 100];
        assert!(!fake_quantize_int8_per_channel(&mut zero_ch, 0));
    }
}
