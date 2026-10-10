//! Int8 quantization primitives shared by GGUF serialization (post-training
//! quantization on save) and Quantization Aware Training (QAT).
//!
//! Quantization is symmetric per-tensor: `value ≈ i8 × scale` with
//! `scale = max|value| / 127`. During QAT, [`fake_quantize_int8`] snaps a
//! weight tensor onto that int8 grid in place (quantize → dequantize), so the
//! forward and backward passes see exactly the weights an int8-serialized
//! model will run with, while the optimizer keeps updating full-precision
//! master weights (straight-through estimator).

/// Tensors shorter than this stay f32 both in quantized GGUF files and during
/// QAT: biases and LayerNorm parameters are small (no size win) and
/// accuracy-sensitive.
pub const QUANT_MIN_LEN: usize = 64;

/// The symmetric int8 scale for a tensor: `max|value| / 127`.
pub fn int8_scale(data: &[f32]) -> f32 {
    data.iter().fold(0.0f32, |m, &v| m.max(v.abs())) / 127.0
}

/// Snap `data` onto the symmetric int8 grid in place (quantize → dequantize).
///
/// Mirrors the GGUF serialization rules: tensors shorter than
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
    debug_assert!(channels != 0 && data.len().is_multiple_of(channels));
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
    if channels == 0 || data.is_empty() || !data.len().is_multiple_of(channels) {
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

// ─────────────────────────────────────────────────────────────────────────────
// In-memory quantized weights (§Opt#1): keep big weight matrices packed in RAM
// and feed them to the kernels *without* expanding to f32.
// ─────────────────────────────────────────────────────────────────────────────
//
// The decode hot path is bandwidth-bound: each generated token streams every
// weight once. Storing weights as int8 (¼ the bytes) or int4 (⅛) cuts both the
// resident footprint *and* the bytes streamed per token, which is the only lever
// that raises the bandwidth-limited token rate. `QWeight` is the in-memory
// counterpart used for imported Llama models and `qlinear`; it is *not* the
// GGUF on-disk codec (the loader writes/reads GGUF Q8_0/Q4_0 blocks). Symmetric,
// one scale per **input row** of the `[in_features, out_features]` matrix, so
// `value ≈ q × scale[row]`.

/// The 4-bit symmetric grid uses 15 levels (−7..=7); `scale = max|w| / 7`.
pub const INT4_MAX: f32 = 7.0;

/// Bit width of an in-memory quantized weight matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QKind {
    /// One signed byte per weight (`value = i8 × scale`).
    Int8,
    /// Two signed nibbles per byte (`value = nibble × scale`, nibble ∈ −7..=7).
    Int4,
}

impl QKind {
    /// Resident bytes for a `rows × cols` matrix in this representation
    /// (excludes the per-row `f32` scales).
    pub fn packed_len(self, rows: usize, cols: usize) -> usize {
        match self {
            QKind::Int8 => rows * cols,
            QKind::Int4 => rows * cols.div_ceil(2),
        }
    }
}

/// A weight matrix kept quantized in memory: `[rows, cols]` row-major with one
/// symmetric scale per row. `rows = in_features`, `cols = out_features`.
///
/// This is what makes a 1B/3B model *fit* and *stream less* on CPU: at int4 a
/// 1B model is ~0.5 GB resident (vs ~3.7 GB as f32) and streams ⅛ the bytes per
/// token. The kernels in [`crate::ops`] consume it directly; it is never
/// expanded back to a full f32 buffer.
#[derive(Clone, Debug, PartialEq)]
pub struct QWeight {
    pub rows: usize,
    pub cols: usize,
    pub kind: QKind,
    /// One scale per row (length `rows`).
    pub scales: Vec<f32>,
    /// Packed quantized values. Int8: `rows × cols` bytes (each an `i8`). Int4:
    /// `rows × ceil(cols/2)` bytes in a **split-half** layout — within each row,
    /// byte `b`'s low nibble holds column `b` and its high nibble holds column
    /// `half + b` (`half = ceil(cols/2)`). This maps the two nibble lanes onto two
    /// *contiguous* column ranges, so int4 decode is unit-stride and vectorises
    /// like int8 (the interleaved "even/odd in one byte" layout did not).
    pub q: Vec<u8>,
}

impl QWeight {
    /// Number of bytes packed per row (int4 rounds `cols` up to a whole byte).
    #[inline]
    pub fn row_bytes(&self) -> usize {
        match self.kind {
            QKind::Int8 => self.cols,
            QKind::Int4 => self.cols.div_ceil(2),
        }
    }

    /// Total resident bytes (packed weights + `f32` scales) — the quantity that
    /// bounds decode bandwidth.
    pub fn resident_bytes(&self) -> usize {
        self.q.len() + self.scales.len() * 4
    }

    /// Quantize a row-major `[rows, cols]` f32 matrix per row.
    pub fn from_f32(data: &[f32], rows: usize, cols: usize, kind: QKind) -> Self {
        debug_assert_eq!(data.len(), rows * cols, "QWeight::from_f32 shape mismatch");
        let mut scales = Vec::with_capacity(rows);
        let q = match kind {
            QKind::Int8 => {
                let mut q = vec![0u8; rows * cols];
                for r in 0..rows {
                    let row = &data[r * cols..(r + 1) * cols];
                    let scale = int8_scale(row);
                    scales.push(scale);
                    let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
                    let out = &mut q[r * cols..(r + 1) * cols];
                    for (o, &v) in out.iter_mut().zip(row) {
                        *o = ((v * inv).round().clamp(-127.0, 127.0) as i8) as u8;
                    }
                }
                q
            }
            QKind::Int4 => {
                let half = cols.div_ceil(2);
                let row_bytes = half;
                let mut q = vec![0u8; rows * row_bytes];
                for r in 0..rows {
                    let row = &data[r * cols..(r + 1) * cols];
                    let amax = row.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
                    let scale = amax / INT4_MAX;
                    scales.push(scale);
                    let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
                    let dst = &mut q[r * row_bytes..(r + 1) * row_bytes];
                    // Split-half: column c<half → byte c low nibble; column
                    // c≥half → byte (c-half) high nibble. Keeps each nibble lane
                    // on a contiguous column range (see the `q` field docs).
                    for (c, &v) in row.iter().enumerate() {
                        let qi = (v * inv).round().clamp(-INT4_MAX, INT4_MAX) as i32;
                        let nib = (qi & 0x0F) as u8;
                        if c < half {
                            dst[c] = (dst[c] & 0xF0) | nib;
                        } else {
                            dst[c - half] = (dst[c - half] & 0x0F) | (nib << 4);
                        }
                    }
                }
                q
            }
        };
        Self {
            rows,
            cols,
            kind,
            scales,
            q,
        }
    }

    /// Sign-extend a 4-bit nibble (`0..=15`) to `-8..=7`.
    #[inline(always)]
    pub fn nibble_to_i8(nib: u8) -> i8 {
        ((nib as i8) << 4) >> 4
    }

    /// Dequantize one row into `out` (length `cols`). Used by tests and the
    /// f32 fallback; the hot kernels read `q`/`scales` directly.
    pub fn dequant_row(&self, r: usize, out: &mut [f32]) {
        let scale = self.scales[r];
        match self.kind {
            QKind::Int8 => {
                let src = &self.q[r * self.cols..(r + 1) * self.cols];
                for (o, &b) in out.iter_mut().zip(src) {
                    *o = (b as i8) as f32 * scale;
                }
            }
            QKind::Int4 => {
                let half = self.cols.div_ceil(2);
                let rb = self.row_bytes();
                let src = &self.q[r * rb..(r + 1) * rb];
                // Split-half layout (see the `q` field docs).
                for c in 0..self.cols {
                    let nib = if c < half {
                        src[c] & 0x0F
                    } else {
                        src[c - half] >> 4
                    };
                    out[c] = Self::nibble_to_i8(nib) as f32 * scale;
                }
            }
        }
    }

    /// Materialize the whole matrix back to a row-major f32 `Vec` (for
    /// serialization fallback or debugging — not the hot path).
    pub fn to_f32(&self) -> Vec<f32> {
        let mut out = vec![0.0f32; self.rows * self.cols];
        for r in 0..self.rows {
            let s = r * self.cols;
            self.dequant_row(r, &mut out[s..s + self.cols]);
        }
        out
    }
}

/// Q8_0 block size (values per f16 scale).
pub const Q8_0_BLOCK: usize = 32;

/// Quantize one block exactly like llama.cpp's `quantize_row_q8_0_ref`:
/// `d = amax/127`; `q = round(x * (1/d))`; stored scale = `f16(d)`.
/// Single source of truth for both the GGUF writer and QAT.
pub fn q8_0_block(x: &[f32]) -> (u16, [i8; 32]) {
    let amax = x.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    let d = amax / 127.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    let mut q = [0i8; 32];
    for (qi, &v) in q.iter_mut().zip(x) {
        *qi = (v * id).round() as i8; // saturating cast; |v*id| <= 127 (+eps)
    }
    (crate::gguf_write::f32_to_f16(d), q)
}

/// True when a matrix with GGML row length `ne0` is stored Q8_0 (else F32).
pub fn q8_0_eligible(ne0: usize) -> bool {
    ne0.is_multiple_of(Q8_0_BLOCK)
}

/// Snap one block onto the values llama.cpp reads back (`q * f16(d)`).
fn snap_q8_0_block(x: &mut [f32]) {
    let (d16, q) = q8_0_block(x);
    let d = crate::gguf::f16_to_f32(d16);
    for (v, &qi) in x.iter_mut().zip(&q) {
        *v = qi as f32 * d;
    }
}

/// Snap a GGML-row-contiguous tensor (rows of length `ne0`, `ne0 % 32 == 0`)
/// onto its Q8_0 values in place. Ineligible `ne0` leaves the data untouched.
pub fn fake_quantize_q8_0_rows(data: &mut [f32], ne0: usize) {
    if !q8_0_eligible(ne0) || ne0 == 0 {
        return;
    }
    for blk in data.chunks_exact_mut(Q8_0_BLOCK) {
        snap_q8_0_block(blk);
    }
}

/// Snap a Ferrum `[in, out]` row-major weight as GGUF stores it (GGML
/// `ne0 = in`: Q8_0 blocks run down each column). Ineligible `rows_in` is a no-op.
pub fn fake_quantize_q8_0_cols(data: &mut [f32], rows_in: usize, cols_out: usize) {
    if !q8_0_eligible(rows_in) || rows_in == 0 {
        return;
    }
    let mut buf = [0.0f32; Q8_0_BLOCK];
    for j in 0..cols_out {
        for i0 in (0..rows_in).step_by(Q8_0_BLOCK) {
            for (k, b) in buf.iter_mut().enumerate() {
                *b = data[(i0 + k) * cols_out + j];
            }
            snap_q8_0_block(&mut buf);
            for (k, b) in buf.iter().enumerate() {
                data[(i0 + k) * cols_out + j] = *b;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_0_block_matches_llama_cpp_reference() {
        let x: Vec<f32> = (0..32).map(|i| ((i as f32) - 15.5) * 0.37).collect();
        let (d16, q) = q8_0_block(&x);
        let amax = x.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let d = amax / 127.0;
        assert_eq!(d16, crate::gguf_write::f32_to_f16(d));
        for (i, &v) in x.iter().enumerate() {
            assert_eq!(q[i] as f32, (v * (1.0 / d)).round());
        }
        let zeros = q8_0_block(&[0.0; 32]);
        assert_eq!(zeros.1, [0i8; 32]);
    }

    #[test]
    fn q8_0_snap_is_a_fixed_point_of_encode_decode() {
        let mut w: Vec<f32> = (0..96 * 40)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 61.0)
            .collect();
        fake_quantize_q8_0_cols(&mut w, 96, 40);
        let once = w.clone();
        fake_quantize_q8_0_cols(&mut w, 96, 40);
        assert_eq!(w, once, "snapping twice must not move values");
    }

    #[test]
    fn q8_0_block_rounds_half_away_from_zero_and_pins_amax_to_127() {
        // amax = 127 -> d = 1 exactly, so q = round(x) with ties away from zero (C roundf).
        let mut x = [0.0f32; 32];
        x[..8].copy_from_slice(&[127.0, 0.5, 1.5, -0.5, -2.5, 2.4999, -126.5, 0.49]);
        let (d16, q) = q8_0_block(&x);
        assert_eq!(crate::gguf::f16_to_f32(d16), 1.0);
        assert_eq!(&q[..8], &[127, 1, 2, -1, -3, 2, -127, 0]);
        assert!(q[8..].iter().all(|&v| v == 0));
    }

    #[test]
    fn q8_0_block_negative_extreme_all_negative_and_single_outlier() {
        let mut neg = [-0.01f32; 32];
        neg[7] = -3.0;
        let (d16, q) = q8_0_block(&neg);
        assert_eq!(
            q[7], -127,
            "the largest-magnitude (negative) value maps to -127"
        );
        assert!(q.iter().all(|&v| v <= 0));
        let d = crate::gguf::f16_to_f32(d16);
        assert!(
            (d - 3.0 / 127.0).abs() < 3.0 / 127.0 * 1e-3,
            "f16 scale ~ amax/127: {d}"
        );
        // one outlier crushes the small values to the 0/-1 bins
        assert_eq!(q[0], (-0.01f32 * (1.0 / (3.0 / 127.0))).round() as i8);
        let mut pos = [0.0f32; 32];
        pos[31] = 5.0;
        let (_, q) = q8_0_block(&pos);
        assert_eq!((q[31], q[..31].iter().all(|&v| v == 0)), (127, true));
    }

    #[test]
    fn q8_0_all_zero_block_has_zero_scale_and_snaps_to_zero_without_nan() {
        let (d16, q) = q8_0_block(&[0.0; 32]);
        assert_eq!((d16, q), (0, [0i8; 32]));
        let mut w = vec![0.0f32; 64];
        fake_quantize_q8_0_rows(&mut w, 32);
        assert!(w.iter().all(|&v| v == 0.0 && v.is_finite()));
        // sub-f16-range block: the stored scale underflows to 0 -> snaps to zero, never NaN
        let mut tiny = vec![1e-30f32; 32];
        fake_quantize_q8_0_rows(&mut tiny, 32);
        assert!(tiny.iter().all(|v| v.is_finite()), "{tiny:?}");
    }

    #[test]
    fn q8_0_snap_equals_q_times_f16_scale_and_blocks_are_independent() {
        // two blocks with very different magnitudes keep their own scales
        let mut w: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.001).collect();
        w.extend((0..32).map(|i| (i as f32 - 16.0) * 100.0));
        let orig = w.clone();
        fake_quantize_q8_0_rows(&mut w, 64);
        for b in 0..2 {
            let src = &orig[b * 32..(b + 1) * 32];
            let (d16, q) = q8_0_block(src);
            let d = crate::gguf::f16_to_f32(d16);
            for k in 0..32 {
                assert_eq!(w[b * 32 + k], q[k] as f32 * d, "block {b} elem {k}");
            }
            let amax = src.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
            for k in 0..32 {
                assert!((w[b * 32 + k] - src[k]).abs() <= amax / 127.0 * 0.51 + amax * 1e-3);
            }
        }
    }

    #[test]
    fn q8_0_cols_blocks_run_down_each_column_and_match_transposed_rows() {
        let (rows, cols) = (64usize, 5usize);
        let orig: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 29 % 97) as f32 - 48.0) * (1.0 + (i % cols) as f32))
            .collect();
        let mut w = orig.clone();
        fake_quantize_q8_0_cols(&mut w, rows, cols);
        // reference: transpose to [cols][rows] (GGML ne0 = rows), snap rows, transpose back
        let mut t: Vec<f32> = (0..cols)
            .flat_map(|j| (0..rows).map(move |i| (i, j)))
            .map(|(i, j)| orig[i * cols + j])
            .collect();
        fake_quantize_q8_0_rows(&mut t, rows);
        for i in 0..rows {
            for j in 0..cols {
                assert_eq!(w[i * cols + j], t[j * rows + i], "({i},{j})");
            }
        }
    }

    #[test]
    fn q8_0_degenerate_shapes_are_noops() {
        let orig = vec![1.5f32; 64];
        let mut w = orig.clone();
        fake_quantize_q8_0_rows(&mut w, 0);
        fake_quantize_q8_0_cols(&mut w, 0, 4);
        fake_quantize_q8_0_cols(&mut w, 63, 1);
        assert_eq!(w, orig);
        assert!(q8_0_eligible(32) && q8_0_eligible(0) && q8_0_eligible(4096));
        assert!(!q8_0_eligible(31) && !q8_0_eligible(33) && !q8_0_eligible(48));
    }

    #[test]
    fn q8_0_ineligible_is_untouched_and_rows_snap() {
        let orig: Vec<f32> = (0..48 * 3).map(|i| i as f32 * 0.013).collect();
        let mut w = orig.clone();
        fake_quantize_q8_0_cols(&mut w, 48, 3);
        fake_quantize_q8_0_rows(&mut w, 48);
        assert_eq!(w, orig);
        let mut r: Vec<f32> = (0..64).map(|i| (i as f32).sin()).collect();
        fake_quantize_q8_0_rows(&mut r, 32);
        assert_ne!(r, (0..64).map(|i| (i as f32).sin()).collect::<Vec<_>>());
    }

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
        let mut data: Vec<f32> = (0..channels * row)
            .map(|i| (i as f32 * 0.13).sin())
            .collect();
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
        assert!(
            scales[2] > scales[0] * 5.0,
            "outlier did not localise to its row"
        );
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
        assert!(
            e_channel * 5.0 < e_tensor,
            "per-channel ({e_channel:.4}) should be far below per-tensor ({e_tensor:.4})"
        );
    }

    #[test]
    fn per_channel_single_channel_equals_per_tensor() {
        let mut a: Vec<f32> = (0..128).map(|i| (i as f32 * 0.21).cos()).collect();
        let mut b = a.clone();
        fake_quantize_int8(&mut a);
        fake_quantize_int8_per_channel(&mut b, 1);
        assert_eq!(a, b);
    }

    // ── In-memory QWeight (§Opt#1) ────────────────────────────────────────────

    #[test]
    fn qweight_int8_roundtrip_bounded_by_half_scale() {
        let (rows, cols) = (5, 40);
        let data: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32 * 0.017).sin() * 1.3)
            .collect();
        let qw = QWeight::from_f32(&data, rows, cols, QKind::Int8);
        assert_eq!(qw.kind, QKind::Int8);
        assert_eq!(qw.resident_bytes(), rows * cols + rows * 4);
        let back = qw.to_f32();
        for r in 0..rows {
            let row = &data[r * cols..(r + 1) * cols];
            let scale = int8_scale(row);
            for c in 0..cols {
                assert!((row[c] - back[r * cols + c]).abs() <= scale * 0.5 + 1e-6);
            }
        }
    }

    #[test]
    fn qweight_int4_roundtrip_bounded_by_half_scale() {
        // 4-bit: each value within half its row's (max/7) step. Odd `cols`
        // exercises the per-row nibble padding.
        let (rows, cols) = (4, 31);
        let data: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.07).cos()).collect();
        let qw = QWeight::from_f32(&data, rows, cols, QKind::Int4);
        assert_eq!(qw.kind, QKind::Int4);
        // ~⅛ the f32 footprint: ceil(31/2)=16 bytes/row.
        assert_eq!(qw.q.len(), rows * 16);
        let back = qw.to_f32();
        for r in 0..rows {
            let row = &data[r * cols..(r + 1) * cols];
            let amax = row.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            let step = amax / INT4_MAX;
            for c in 0..cols {
                assert!(
                    (row[c] - back[r * cols + c]).abs() <= step * 0.5 + 1e-6,
                    "int4 r{r} c{c}: {} vs {}",
                    row[c],
                    back[r * cols + c]
                );
            }
        }
    }

    #[test]
    fn qweight_int4_is_eighth_size_of_f32() {
        let (rows, cols) = (64, 64);
        let data = vec![0.5f32; rows * cols];
        let qw = QWeight::from_f32(&data, rows, cols, QKind::Int4);
        // 4 bits/weight vs 32: packed bytes are 1/8 the f32 byte count.
        assert_eq!(qw.q.len(), rows * cols / 2);
        assert_eq!(QKind::Int4.packed_len(rows, cols), rows * cols / 2);
        assert_eq!(QKind::Int8.packed_len(rows, cols), rows * cols);
    }

    #[test]
    fn nibble_sign_extension() {
        assert_eq!(QWeight::nibble_to_i8(0x0), 0);
        assert_eq!(QWeight::nibble_to_i8(0x7), 7);
        assert_eq!(QWeight::nibble_to_i8(0xF), -1);
        assert_eq!(QWeight::nibble_to_i8(0x9), -7);
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
