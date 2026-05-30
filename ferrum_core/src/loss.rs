//! Fused softmax cross-entropy loss and its gradient.
//!
//! The gradient of cross-entropy(softmax(z)) w.r.t. z is simply (p - onehot(t))
//! divided by batch size. This clean form is why we fuse the two operations.
use crate::error::{InferError, Result};
use crate::tensor::Tensor;

/// Returns (mean_loss, dL/d_logits) for a batch.
/// `logits` shape: [batch, num_classes]. `targets`: one class index per row.
#[allow(clippy::needless_range_loop)]
pub fn softmax_cross_entropy(logits: &Tensor, targets: &[usize]) -> Result<(f32, Tensor)> {
    let (batch, vocab) = logits.matrix_dims()?;
    if targets.len() != batch {
        return Err(InferError::DimMismatch(format!(
            "{} targets for batch of {batch}",
            targets.len()
        )));
    }
    let mut grad = vec![0.0f32; batch * vocab];
    let mut total_loss = 0.0f32;
    let inv_batch = 1.0 / batch as f32;

    for i in 0..batch {
        let base = i * vocab;
        // Numerically stable softmax.
        let max = logits.data[base..base + vocab]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut probs = vec![0.0f32; vocab];
        let mut sum = 0.0f32;
        for j in 0..vocab {
            let e = (logits.data[base + j] - max).exp();
            probs[j] = e;
            sum += e;
        }
        for j in 0..vocab {
            probs[j] /= sum;
        }

        let t = targets[i];
        if t >= vocab {
            return Err(InferError::DimMismatch(format!(
                "target index {t} ≥ num_classes {vocab}"
            )));
        }
        total_loss += -(probs[t].max(1e-12)).ln();
        for j in 0..vocab {
            grad[base + j] = (probs[j] - if j == t { 1.0 } else { 0.0 }) * inv_batch;
        }
    }
    Ok((total_loss * inv_batch, Tensor::matrix(batch, vocab, grad)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_prediction_near_zero_loss() {
        let logits = Tensor::matrix(1, 3, vec![100.0, 0.0, 0.0]).unwrap();
        let (loss, _) = softmax_cross_entropy(&logits, &[0]).unwrap();
        assert!(loss < 1e-4, "loss = {loss}");
    }

    #[test]
    fn uniform_logits_give_log_vocab_loss() {
        let logits = Tensor::matrix(1, 4, vec![0.0; 4]).unwrap();
        let (loss, _) = softmax_cross_entropy(&logits, &[0]).unwrap();
        assert!((loss - (4.0f32).ln()).abs() < 1e-5, "loss = {loss}");
    }

    #[test]
    fn target_out_of_range_errors() {
        let logits = Tensor::matrix(1, 3, vec![0.0; 3]).unwrap();
        assert!(softmax_cross_entropy(&logits, &[5]).is_err());
    }

    #[test]
    fn target_count_mismatch_errors() {
        let logits = Tensor::matrix(2, 3, vec![0.0; 6]).unwrap();
        assert!(softmax_cross_entropy(&logits, &[0]).is_err()); // 1 target for batch of 2
    }

    #[test]
    fn gradient_finite_difference() {
        let base = vec![0.3f32, -1.2, 0.7, 2.1, -0.5, 0.1];
        let targets = [2usize, 0];
        let logits = Tensor::matrix(2, 3, base.clone()).unwrap();
        let (_, grad) = softmax_cross_entropy(&logits, &targets).unwrap();

        let eps = 1e-3f32;
        for k in 0..base.len() {
            let mut plus = base.clone();
            plus[k] += eps;
            let mut minus = base.clone();
            minus[k] -= eps;
            let lp = softmax_cross_entropy(&Tensor::matrix(2, 3, plus).unwrap(), &targets)
                .unwrap()
                .0;
            let lm = softmax_cross_entropy(&Tensor::matrix(2, 3, minus).unwrap(), &targets)
                .unwrap()
                .0;
            let numeric = (lp - lm) / (2.0 * eps);
            assert!(
                (numeric - grad.data[k]).abs() < 5e-4,
                "k={k}: analytic={} numeric={numeric}",
                grad.data[k]
            );
        }
    }

    #[test]
    fn gradient_shape_matches_logits() {
        let logits = Tensor::matrix(3, 5, vec![0.0; 15]).unwrap();
        let (_, grad) = softmax_cross_entropy(&logits, &[0, 1, 2]).unwrap();
        assert_eq!(grad.shape, logits.shape);
    }

    #[test]
    fn loss_decreases_after_gradient_step() {
        let mut logits_data = vec![0.0f32; 3];
        let targets = [1usize];
        let lr = 1.0f32;
        let (loss0, grad) = softmax_cross_entropy(
            &Tensor::matrix(1, 3, logits_data.clone()).unwrap(),
            &targets,
        )
        .unwrap();
        for i in 0..3 {
            logits_data[i] -= lr * grad.data[i];
        }
        let (loss1, _) =
            softmax_cross_entropy(&Tensor::matrix(1, 3, logits_data).unwrap(), &targets).unwrap();
        assert!(loss1 < loss0, "loss did not decrease: {loss0} → {loss1}");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Mean-squared-error loss for regression
// ─────────────────────────────────────────────────────────────────────────────

/// MSE loss + gradient for regression.
/// `preds` shape [batch, 1], `targets` raw normalised values.
/// Returns (mean_loss, dL/d_preds) also shape [batch, 1].
pub fn mse(preds: &crate::tensor::Tensor, targets: &[f32]) -> Result<(f32, crate::tensor::Tensor)> {
    let (batch, cols) = preds.matrix_dims()?;
    if cols != 1 {
        return Err(InferError::DimMismatch(format!(
            "MSE expects 1 output col, got {cols}"
        )));
    }
    if targets.len() != batch {
        return Err(InferError::DimMismatch(format!(
            "{} targets for batch {batch}",
            targets.len()
        )));
    }
    let inv = 1.0 / batch as f32;
    let mut loss = 0.0f32;
    let mut grad = vec![0.0f32; batch];
    for i in 0..batch {
        let diff = preds.data[i] - targets[i];
        loss += diff * diff;
        grad[i] = 2.0 * diff * inv;
    }
    Ok((loss * inv, crate::tensor::Tensor::matrix(batch, 1, grad)?))
}

#[cfg(test)]
mod mse_tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn mse_perfect_prediction_zero_loss() {
        let p = Tensor::matrix(2, 1, vec![1.0, 2.0]).unwrap();
        let (loss, _) = mse(&p, &[1.0, 2.0]).unwrap();
        assert!(loss < 1e-7, "{loss}");
    }
    #[test]
    fn mse_gradient_finite_difference() {
        let targets = [0.5f32, -0.3];
        let base = vec![0.7f32, -0.1];
        let (_, grad) = mse(&Tensor::matrix(2, 1, base.clone()).unwrap(), &targets).unwrap();
        let eps = 1e-3f32;
        for k in 0..2 {
            let mut p = base.clone();
            p[k] += eps;
            let (lp, _) = mse(&Tensor::matrix(2, 1, p).unwrap(), &targets).unwrap();
            let mut m = base.clone();
            m[k] -= eps;
            let (lm, _) = mse(&Tensor::matrix(2, 1, m).unwrap(), &targets).unwrap();
            let numeric = (lp - lm) / (2.0 * eps);
            assert!((numeric - grad.data[k]).abs() < 1e-3, "k={k}");
        }
    }
}
