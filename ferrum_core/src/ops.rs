//! Primitive math kernels. All raw arithmetic lives here — nothing above
//! this module does floating-point arithmetic directly.
use crate::error::{InferError, Result};
use crate::tensor::Tensor;
use crate::verbose;

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
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        let a_row = i * ka;
        let o_row = i * n;
        for k in 0..ka {
            let a_ik = a.data[a_row + k];
            let b_row = k * n;
            for j in 0..n {
                out[o_row + j] += a_ik * b.data[b_row + j];
            }
        }
    }
    if verbose::is_verbose() {
        verbose::check_nan_inf(&out, &format!("ops::matmul result [{m},{n}]"));
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
