//! Primitive math kernels. All raw arithmetic lives here — nothing above
//! this module does floating-point arithmetic directly.
use crate::error::{InferError, Result};
use crate::quant::{QKind, QWeight};
use crate::tensor::Tensor;
use crate::verbose;
use std::sync::Arc;

/// Cache-tiling panel sizes for [`matmul_block`]. A `KC × NC` panel of `B`
/// (256×256 f32 = 256 KB) is reused across every row of the current block before
/// the next panel is streamed, which removes the ~2× "cache cliff" the untiled
/// i-k-j kernel hit once `B` overflowed cache at ≥2048² (see `benchmarks.md §4a`).
const KC: usize = 256;
const NC: usize = 256;

/// Accumulate rows `r0..r1` of an `[m, n]` matmul `A·B` (A is `[m, k]`, B is
/// `[k, n]`) into `out`, which holds those rows indexed locally (row `i` at
/// `(i - r0) * n`) and **must be pre-initialised** (to zero, or to a bias row so
/// the epilogue is fused — see [`linear_forward`]). Shared by the serial and
/// pooled paths so the arithmetic is written once.
///
/// The loop is tiled over the `k` and `n` dimensions so a `KC × NC` block of `B`
/// stays hot while it is reused across the block's rows. For any fixed `(i, j)`
/// the contraction still sums `p` in ascending order, so the result is
/// **bit-for-bit identical** to the untiled kernel and to the serial reference.
fn matmul_block(a: &[f32], b: &[f32], k: usize, n: usize, r0: usize, r1: usize, out: &mut [f32]) {
    let mut jj = 0;
    while jj < n {
        let j1 = (jj + NC).min(n);
        let mut pp = 0;
        while pp < k {
            let p1 = (pp + KC).min(k);
            for i in r0..r1 {
                let a_row = i * k;
                let o_base = (i - r0) * n;
                for p in pp..p1 {
                    let a_ip = a[a_row + p];
                    let b_row = p * n;
                    let o = &mut out[o_base + jj..o_base + j1];
                    let bs = &b[b_row + jj..b_row + j1];
                    for (ov, &bv) in o.iter_mut().zip(bs) {
                        *ov += a_ip * bv;
                    }
                }
            }
            pp = p1;
        }
        jj = j1;
    }
}

/// Matrix multiply: [m,k] × [k,n] → [m,n]. Uses i-k-j loop order for cache
/// friendliness: the innermost loop walks contiguous memory in b and the output.
pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let (m, ka) = a.matrix_dims()?;
    let (kb, n) = b.matrix_dims()?;
    if ka != kb {
        return Err(InferError::DimMismatch(format!(
            "matmul: [{m},{ka}] × [{kb},{n}] — inner dims disagree"
        )));
    }
    vprintln!("[ops::matmul] [{},{}] × [{},{}] → [{},{}]", m, ka, kb, n, m, n);
    // Split the output rows across the persistent CPU worker pool when the
    // workload is large enough; otherwise compute serially with no Arc clone.
    // Each output row depends only on `a`'s matching row and all of `b`, so the
    // per-element arithmetic is identical regardless of the split.
    let cost = m.saturating_mul(ka).saturating_mul(n);
    let out = if crate::parallel::should_parallelize(m, cost) {
        let a_arc = Arc::<[f32]>::from(a.data.as_slice());
        let b_arc = Arc::<[f32]>::from(b.data.as_slice());
        crate::parallel::run(m, n, move |r0, r1, block| {
            matmul_block(&a_arc, &b_arc, ka, n, r0, r1, block);
        })
    } else {
        let mut out = vec![0.0f32; m * n];
        matmul_block(&a.data, &b.data, ka, n, 0, m, &mut out);
        out
    };
    if verbose::is_verbose() {
        verbose::check_nan_inf(&out, &format!("ops::matmul result [{m},{n}]"));
    }
    Tensor::matrix(m, n, out)
}

/// Affine layer `y = x·W + b` with the bias **fused into the matmul epilogue**
/// (Opt#3): the output is allocated once and initialised to the bias row, then
/// the products accumulate onto it — no separate `add_bias` pass and no
/// intermediate clone (the old `add_bias(matmul(..), b)` did two allocations and
/// a full copy per call, hundreds of times per generated token).
///
/// `weight` is `[k, n]` (`k = in_features`, `n = out_features`), `bias` length
/// `n`. Bit-identical to `add_bias(&matmul(input, weight)?, bias)`.
pub fn linear_forward(input: &Tensor, weight: &Tensor, bias: &[f32]) -> Result<Tensor> {
    let (m, k) = input.matrix_dims()?;
    let (kb, n) = weight.matrix_dims()?;
    if k != kb {
        return Err(InferError::DimMismatch(format!(
            "linear_forward: input width {k} ≠ weight rows {kb}"
        )));
    }
    if bias.len() != n {
        return Err(InferError::DimMismatch(format!(
            "linear_forward: bias length {} ≠ out_features {n}",
            bias.len()
        )));
    }
    // Accumulate the products from zero, then add the bias as an epilogue over
    // the *same* buffer. This drops the intermediate tensor + clone that the old
    // `add_bias(matmul(..))` allocated (the Opt#3 win) while keeping the bias
    // added last — so the result is bit-for-bit identical, not merely close.
    let cost = m.saturating_mul(k).saturating_mul(n);
    let out = if crate::parallel::should_parallelize(m, cost) {
        let a_arc = Arc::<[f32]>::from(input.data.as_slice());
        let b_arc = Arc::<[f32]>::from(weight.data.as_slice());
        let bias_arc = Arc::<[f32]>::from(bias);
        crate::parallel::run(m, n, move |r0, r1, block| {
            matmul_block(&a_arc, &b_arc, k, n, r0, r1, block);
            for i in r0..r1 {
                let o = &mut block[(i - r0) * n..(i - r0) * n + n];
                for (ov, &bv) in o.iter_mut().zip(bias_arc.iter()) {
                    *ov += bv;
                }
            }
        })
    } else {
        let mut out = vec![0.0f32; m * n];
        matmul_block(&input.data, &weight.data, k, n, 0, m, &mut out);
        for i in 0..m {
            for (ov, &bv) in out[i * n..i * n + n].iter_mut().zip(bias) {
                *ov += bv;
            }
        }
        out
    };
    if verbose::is_verbose() {
        verbose::check_nan_inf(&out, &format!("ops::linear_forward result [{m},{n}]"));
    }
    Tensor::matrix(m, n, out)
}

/// Accumulate columns `j0..j1` of one output row onto `out` (length `j1 - j0`,
/// pre-initialised to the bias): `out[j-j0] += Σ_p a_row[p] · dequant(W[p, j])`.
///
/// `a_row` is the activation row **already pre-multiplied by the per-row weight
/// scale** (so the kernel only ever multiplies by raw integer levels). For int8
/// each weight row is a contiguous `i8` slice (autovectorises cleanly); for int4
/// two nibbles are unpacked per byte.
fn qaccum_cols(a_row: &[f32], w: &QWeight, j0: usize, j1: usize, out: &mut [f32]) {
    let k = a_row.len();
    match w.kind {
        QKind::Int8 => {
            let cols = w.cols;
            for (p, &ap) in a_row.iter().enumerate().take(k) {
                let row = &w.q[p * cols + j0..p * cols + j1];
                for (o, &qb) in out.iter_mut().zip(row) {
                    *o += ap * ((qb as i8) as f32);
                }
            }
        }
        QKind::Int4 => {
            let rb = w.row_bytes();
            for (p, &ap) in a_row.iter().enumerate().take(k) {
                let base = p * rb;
                let mut j = j0;
                while j < j1 {
                    let byte = w.q[base + (j >> 1)];
                    if j & 1 == 0 {
                        out[j - j0] += ap * (QWeight::nibble_to_i8(byte & 0x0F) as f32);
                        if j + 1 < j1 {
                            out[j + 1 - j0] += ap * (QWeight::nibble_to_i8(byte >> 4) as f32);
                        }
                        j += 2;
                    } else {
                        out[j - j0] += ap * (QWeight::nibble_to_i8(byte >> 4) as f32);
                        j += 1;
                    }
                }
            }
        }
    }
}

/// Affine layer `y = x·W + b` where `W` is an **in-memory quantized** weight
/// (int8 / int4) consumed directly — never expanded to f32 (Opt#1). This is the
/// kernel that makes 1B/3B fit and stream less: a 1B int4 layer reads ⅛ the
/// bytes of f32. Bias is fused like [`linear_forward`].
///
/// The single-token decode case (`m == 1`) is split across the worker pool by
/// **output column** (Opt#2), so the autoregressive hot path — which `matmul`'s
/// row split leaves on one core — finally uses every core. The split is
/// deterministic: each output element is reduced entirely within one worker, so
/// the result does not depend on the thread count.
pub fn qlinear(input: &Tensor, w: &Arc<QWeight>, bias: &[f32]) -> Result<Tensor> {
    let (m, k) = input.matrix_dims()?;
    if k != w.rows {
        return Err(InferError::DimMismatch(format!(
            "qlinear: input width {k} ≠ weight rows {}",
            w.rows
        )));
    }
    let n = w.cols;
    if bias.len() != n {
        return Err(InferError::DimMismatch(format!(
            "qlinear: bias length {} ≠ out_features {n}",
            bias.len()
        )));
    }
    // Fold the per-row weight scale into the activations once: a'[i,p] =
    // x[i,p]·scale[p]. Cost m·k — negligible against the m·k·n contraction.
    let mut a = input.data.clone();
    for i in 0..m {
        let base = i * k;
        for (p, s) in w.scales.iter().enumerate().take(k) {
            a[base + p] *= *s;
        }
    }

    let out = if m == 1 && crate::parallel::should_parallelize_1d(n, k) {
        let a_arc = Arc::<[f32]>::from(a.as_slice());
        let bias_arc = Arc::<[f32]>::from(bias);
        let w = Arc::clone(w);
        crate::parallel::run_1d(n, 64, move |j0, j1, block| {
            for (idx, j) in (j0..j1).enumerate() {
                block[idx] = bias_arc[j];
            }
            qaccum_cols(&a_arc, &w, j0, j1, block);
        })
    } else if m >= 2 && crate::parallel::should_parallelize(m, m.saturating_mul(k).saturating_mul(n)) {
        let a_arc = Arc::<[f32]>::from(a.as_slice());
        let bias_arc = Arc::<[f32]>::from(bias);
        let w = Arc::clone(w);
        crate::parallel::run(m, n, move |r0, r1, block| {
            for i in r0..r1 {
                let o = &mut block[(i - r0) * n..(i - r0) * n + n];
                o.copy_from_slice(&bias_arc);
                qaccum_cols(&a_arc[i * k..i * k + k], &w, 0, n, o);
            }
        })
    } else {
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            let o = &mut out[i * n..i * n + n];
            o.copy_from_slice(bias);
            qaccum_cols(&a[i * k..i * k + k], w, 0, n, o);
        }
        out
    };
    if verbose::is_verbose() {
        verbose::check_nan_inf(&out, &format!("ops::qlinear result [{m},{n}]"));
    }
    Tensor::matrix(m, n, out)
}

/// Add a bias vector of length `cols` to every row of a [rows, cols] matrix.
pub fn add_bias(matrix: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let (rows, cols) = matrix.matrix_dims()?;
    if bias.numel() != cols {
        return Err(InferError::DimMismatch(format!(
            "bias length {} ≠ matrix cols {cols}",
            bias.numel()
        )));
    }
    vprintln!("[ops::add_bias] [{},{}] + bias[{}]", rows, cols, cols);
    let mut out = matrix.data.clone();
    for r in 0..rows {
        let base = r * cols;
        for c in 0..cols {
            out[base + c] += bias.data[c];
        }
    }
    if verbose::is_verbose() {
        verbose::check_nan_inf(&out, "ops::add_bias result");
    }
    Tensor::matrix(rows, cols, out)
}

/// Element-wise sum of two identically-shaped tensors.
pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.shape != b.shape {
        return Err(InferError::DimMismatch(format!(
            "add: shapes {:?} and {:?} differ",
            a.shape, b.shape
        )));
    }
    vprintln!("[ops::add] shapes {:?} + {:?}", a.shape, b.shape);
    let data = a.data.iter().zip(&b.data).map(|(x, y)| x + y).collect();
    Tensor::new(a.shape.clone(), data)
}

/// Transpose a rank-2 tensor: [r,c] → [c,r].
pub fn transpose(m: &Tensor) -> Result<Tensor> {
    let (r, c) = m.matrix_dims()?;
    vprintln!("[ops::transpose] [{},{}] → [{},{}]", r, c, c, r);
    let mut out = vec![0.0f32; r * c];
    for i in 0..r {
        for j in 0..c {
            out[j * r + i] = m.data[i * c + j];
        }
    }
    Tensor::matrix(c, r, out)
}

/// Sum a [rows, cols] matrix along axis-0 → a length-cols vector.
pub fn sum_axis0(m: &Tensor) -> Result<Tensor> {
    let (rows, cols) = m.matrix_dims()?;
    vprintln!("[ops::sum_axis0] [{},{}] → [{}]", rows, cols, cols);
    let mut out = vec![0.0f32; cols];
    for i in 0..rows {
        let base = i * cols;
        #[allow(clippy::needless_range_loop)]
        for j in 0..cols {
            out[j] += m.data[base + j];
        }
    }
    Ok(Tensor::vector(out))
}

/// Scale all elements by a scalar.
pub fn scale(t: &Tensor, s: f32) -> Tensor {
    vprintln!("[ops::scale] shape={:?}, scalar={:.6}", t.shape, s);
    t.map(|x| x * s)
}

/// Element-wise (Hadamard) product of two identically-shaped tensors.
pub fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.shape != b.shape {
        return Err(InferError::DimMismatch(format!(
            "mul: shapes {:?} and {:?} differ",
            a.shape, b.shape
        )));
    }
    vprintln!("[ops::mul] shapes {:?} ⊙ {:?}", a.shape, b.shape);
    let data = a.data.iter().zip(&b.data).map(|(x, y)| x * y).collect();
    Tensor::new(a.shape.clone(), data)
}

/// Index of the largest value in each row → one class label per example.
pub fn argmax_rows(matrix: &Tensor) -> Result<Vec<usize>> {
    let (rows, cols) = matrix.matrix_dims()?;
    vprintln!("[ops::argmax_rows] [{},{}]", rows, cols);
    let mut out = Vec::with_capacity(rows);
    for r in 0..rows {
        let base = r * cols;
        let best = (0..cols)
            .max_by(|&a, &b| matrix.data[base + a].total_cmp(&matrix.data[base + b]))
            .unwrap_or(0);
        out.push(best);
    }
    Ok(out)
}

/// Row-wise softmax → probability distribution per row.
pub fn softmax_rows(m: &Tensor) -> Result<Tensor> {
    let (rows, cols) = m.matrix_dims()?;
    vprintln!("[ops::softmax_rows] [{},{}]", rows, cols);
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let base = r * cols;
        let max = m.data[base..base + cols]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for c in 0..cols {
            let e = (m.data[base + c] - max).exp();
            out[base + c] = e;
            sum += e;
        }
        for c in 0..cols {
            out[base + c] /= sum;
        }
    }
    if verbose::is_verbose() {
        verbose::check_nan_inf(&out, "ops::softmax_rows result");
    }
    Tensor::matrix(rows, cols, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_1x3_by_3x1() {
        let a = Tensor::matrix(1, 3, vec![1.0, 2.0, 3.0]).unwrap();
        let b = Tensor::matrix(3, 1, vec![1.0, 0.0, -1.0]).unwrap();
        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.shape, vec![1, 1]);
        assert_eq!(c.data[0], -2.0);
    }

    #[test]
    fn matmul_2x2() {
        let a = Tensor::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Tensor::matrix(2, 2, vec![5.0, 6.0, 7.0, 8.0]).unwrap();
        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.data, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn matmul_dim_mismatch_errors() {
        let a = Tensor::matrix(2, 3, vec![0.0; 6]).unwrap();
        let b = Tensor::matrix(2, 2, vec![0.0; 4]).unwrap();
        assert!(matches!(matmul(&a, &b), Err(InferError::DimMismatch(_))));
    }

    #[test]
    fn add_bias_per_row() {
        let m = Tensor::matrix(2, 2, vec![1.0, 1.0, 1.0, 1.0]).unwrap();
        let b = Tensor::vector(vec![10.0, 20.0]);
        let out = add_bias(&m, &b).unwrap();
        assert_eq!(out.data, vec![11.0, 21.0, 11.0, 21.0]);
    }

    #[test]
    fn add_bias_wrong_length() {
        let m = Tensor::matrix(2, 2, vec![0.0; 4]).unwrap();
        let b = Tensor::vector(vec![1.0]);
        assert!(add_bias(&m, &b).is_err());
    }

    #[test]
    fn add_elementwise() {
        let a = Tensor::vector(vec![1.0, 2.0]);
        let b = Tensor::vector(vec![3.0, 4.0]);
        assert_eq!(add(&a, &b).unwrap().data, vec![4.0, 6.0]);
    }

    #[test]
    fn add_shape_mismatch() {
        let a = Tensor::vector(vec![1.0]);
        let b = Tensor::vector(vec![1.0, 2.0]);
        assert!(add(&a, &b).is_err());
    }

    #[test]
    fn transpose_swaps_dims() {
        let m = Tensor::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let t = transpose(&m).unwrap();
        assert_eq!(t.shape, vec![3, 2]);
        assert_eq!(t.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn sum_axis0_adds_down_columns() {
        let m = Tensor::matrix(2, 3, vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0]).unwrap();
        assert_eq!(sum_axis0(&m).unwrap().data, vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn scale_multiplies_all() {
        let t = Tensor::vector(vec![1.0, 2.0, 3.0]);
        assert_eq!(scale(&t, 2.0).data, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn mul_hadamard() {
        let a = Tensor::vector(vec![2.0, 3.0]);
        let b = Tensor::vector(vec![4.0, 5.0]);
        assert_eq!(mul(&a, &b).unwrap().data, vec![8.0, 15.0]);
    }

    #[test]
    fn argmax_rows_picks_max_per_row() {
        let m = Tensor::matrix(2, 3, vec![0.1, 0.7, 0.2, 0.9, 0.05, 0.05]).unwrap();
        assert_eq!(argmax_rows(&m).unwrap(), vec![1, 0]);
    }

    #[test]
    fn matmul_large_matches_serial_reference() {
        // A matmul big enough to cross the parallel threshold must equal a
        // straightforward serial reference bit-for-bit, regardless of how many
        // threads the row split used.
        let (m, k, n) = (200usize, 80usize, 160usize);
        let a_data: Vec<f32> = (0..m * k).map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.1).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| ((i * 5 % 11) as f32 - 5.0) * 0.1).collect();
        let a = Tensor::matrix(m, k, a_data.clone()).unwrap();
        let b = Tensor::matrix(k, n, b_data.clone()).unwrap();

        let got = matmul(&a, &b).unwrap();
        assert_eq!(got.shape, vec![m, n]);

        let mut want = vec![0.0f32; m * n];
        for i in 0..m {
            for p in 0..k {
                let a_ip = a_data[i * k + p];
                for j in 0..n {
                    want[i * n + j] += a_ip * b_data[p * n + j];
                }
            }
        }
        assert_eq!(got.data, want, "parallel matmul diverged from serial reference");
    }

    // ── Fused Linear + quantized kernels (Opt#1/#2/#3) ────────────────────────

    fn det_data(n: usize, seed: u64) -> Vec<f32> {
        let mut x = seed | 1;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x >> 40) as f32 / (1u64 << 23) as f32 - 1.0
            })
            .collect()
    }

    #[test]
    fn linear_forward_matches_add_bias_matmul_bitwise() {
        // Large enough to cross the parallel threshold; the fused, tiled path
        // must equal the old add_bias(matmul(..)) byte-for-byte.
        let (m, k, n) = (40usize, 70usize, 130usize);
        let input = Tensor::matrix(m, k, det_data(m * k, 1)).unwrap();
        let weight = Tensor::matrix(k, n, det_data(k * n, 2)).unwrap();
        let bias = det_data(n, 3);
        let want = add_bias(&matmul(&input, &weight).unwrap(), &Tensor::vector(bias.clone())).unwrap();
        let got = linear_forward(&input, &weight, &bias).unwrap();
        assert_eq!(got.shape, want.shape);
        assert_eq!(got.data, want.data, "fused linear diverged from add_bias(matmul)");
    }

    #[test]
    fn qlinear_int8_close_to_f32_linear() {
        let (k, n) = (96usize, 200usize);
        let wdata = det_data(k * n, 7);
        let weight = Tensor::matrix(k, n, wdata.clone()).unwrap();
        let bias = det_data(n, 8);
        let input = Tensor::matrix(1, k, det_data(k, 9)).unwrap();
        let reference = linear_forward(&input, &weight, &bias).unwrap();

        let qw = Arc::new(QWeight::from_f32(&wdata, k, n, QKind::Int8));
        let got = qlinear(&input, &qw, &bias).unwrap();
        assert_eq!(got.shape, vec![1, n]);
        // int8 per-row error accumulates over k terms but stays small.
        for (a, b) in reference.data.iter().zip(&got.data) {
            assert!((a - b).abs() < 0.05, "int8 qlinear drift: {a} vs {b}");
        }
    }

    #[test]
    fn qlinear_int4_tracks_f32_linear() {
        let (k, n) = (96usize, 200usize);
        let wdata = det_data(k * n, 11);
        let weight = Tensor::matrix(k, n, wdata.clone()).unwrap();
        let bias = det_data(n, 12);
        let input = Tensor::matrix(1, k, det_data(k, 13)).unwrap();
        let reference = linear_forward(&input, &weight, &bias).unwrap();

        let qw = Arc::new(QWeight::from_f32(&wdata, k, n, QKind::Int4));
        let got = qlinear(&input, &qw, &bias).unwrap();
        // int4 is coarse, but the output must correlate strongly with f32 (the
        // sign and rough magnitude survive) — checked as bounded mean abs error.
        let mae: f32 = reference
            .data
            .iter()
            .zip(&got.data)
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / n as f32;
        assert!(mae < 0.5, "int4 qlinear mean abs error too high: {mae}");
    }

    #[test]
    fn qlinear_decode_gemv_is_deterministic_and_correct() {
        // m == 1 crosses the 1-D (column-split) threshold, so this runs across
        // the worker pool. It must equal a straight serial reference exactly,
        // proving the column split is bit-stable regardless of thread count.
        let (k, n) = (64usize, 2048usize); // k*n = 131072 ≥ PARALLEL_THRESHOLD
        let wdata = det_data(k * n, 21);
        let bias = det_data(n, 22);
        let input = Tensor::matrix(1, k, det_data(k, 23)).unwrap();
        let qw = Arc::new(QWeight::from_f32(&wdata, k, n, QKind::Int8));

        let got = qlinear(&input, &qw, &bias).unwrap();

        // Serial reference: dequantize and do the dot products directly.
        let wf = qw.to_f32();
        let mut want = bias.clone();
        for (j, wj) in want.iter_mut().enumerate() {
            for p in 0..k {
                *wj += input.data[p] * wf[p * n + j];
            }
        }
        for (a, b) in want.iter().zip(&got.data) {
            assert!((a - b).abs() < 1e-3, "decode GEMV mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn qlinear_prefill_m_ge_2_matches_f32_int8_and_int4() {
        // m >= 2 with a big enough product takes the row-split parallel quant
        // path. Check both int8 (tight) and int4 (loose) against f32 linear.
        let (m, k, n) = (4usize, 64usize, 300usize); // m*k*n = 76800 ≥ threshold
        let wdata = det_data(k * n, 31);
        let weight = Tensor::matrix(k, n, wdata.clone()).unwrap();
        let bias = det_data(n, 32);
        let input = Tensor::matrix(m, k, det_data(m * k, 33)).unwrap();
        let reference = linear_forward(&input, &weight, &bias).unwrap();

        let q8 = Arc::new(QWeight::from_f32(&wdata, k, n, QKind::Int8));
        let got8 = qlinear(&input, &q8, &bias).unwrap();
        assert_eq!(got8.shape, vec![m, n]);
        for (a, b) in reference.data.iter().zip(&got8.data) {
            assert!((a - b).abs() < 0.06, "int8 prefill drift {a} vs {b}");
        }
        let q4 = Arc::new(QWeight::from_f32(&wdata, k, n, QKind::Int4));
        let got4 = qlinear(&input, &q4, &bias).unwrap();
        let mae: f32 = reference.data.iter().zip(&got4.data).map(|(a, b)| (a - b).abs()).sum::<f32>()
            / (m * n) as f32;
        assert!(mae < 0.6, "int4 prefill mae {mae}");
    }

    #[test]
    fn qlinear_serial_small_path_is_correct() {
        // Below the parallel threshold → serial path (no Arc).
        let (k, n) = (8usize, 8usize);
        let wdata = det_data(k * n, 41);
        let bias = det_data(n, 42);
        let input = Tensor::matrix(1, k, det_data(k, 43)).unwrap();
        let qw = Arc::new(QWeight::from_f32(&wdata, k, n, QKind::Int8));
        let got = qlinear(&input, &qw, &bias).unwrap();
        let wf = qw.to_f32();
        for j in 0..n {
            let mut want = bias[j];
            for p in 0..k {
                want += input.data[p] * wf[p * n + j];
            }
            assert!((got.data[j] - want).abs() < 1e-3);
        }
    }

    #[test]
    fn qlinear_and_linear_forward_reject_bad_shapes() {
        let bias = vec![0.0; 4];
        let qw = Arc::new(QWeight::from_f32(&det_data(3 * 4, 1), 3, 4, QKind::Int8));
        // input width 5 ≠ weight rows 3
        let bad_in = Tensor::matrix(1, 5, vec![0.0; 5]).unwrap();
        assert!(qlinear(&bad_in, &qw, &bias).is_err());
        // bias length mismatch
        let good_in = Tensor::matrix(1, 3, vec![0.0; 3]).unwrap();
        assert!(qlinear(&good_in, &qw, &[0.0; 2]).is_err());

        let w = Tensor::matrix(3, 4, vec![0.0; 12]).unwrap();
        assert!(linear_forward(&bad_in, &w, &bias).is_err()); // width mismatch
        assert!(linear_forward(&good_in, &w, &[0.0; 2]).is_err()); // bias mismatch
    }

    #[test]
    fn softmax_rows_sums_to_one() {
        let m = Tensor::matrix(2, 3, vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0]).unwrap();
        let p = softmax_rows(&m).unwrap();
        for r in 0..2 {
            let s: f32 = p.data[r * 3..(r + 1) * 3].iter().sum();
            assert!((s - 1.0).abs() < 1e-6, "row {r} sum = {s}");
        }
    }

    #[test]
    fn softmax_monotone_within_row() {
        let m = Tensor::matrix(1, 3, vec![1.0, 2.0, 3.0]).unwrap();
        let p = softmax_rows(&m).unwrap();
        assert!(p.data[0] < p.data[1] && p.data[1] < p.data[2]);
    }
}
