//! Machine-capability estimator: micro-benchmarks the host and derives
//! upper-bound parameter counts for inference (>=3 tok/s), training (<24h),
//! and evaluation (<24h). See docs/superpowers/specs/2026-06-23-capable-module-design.md.

// ── Modeling constants ───────────────────────────────────────────────────────

/// Bytes per stored parameter at each precision (token-embedding nuance ignored
/// at the headline level — see the design doc).
pub const BPP_INT4: f64 = 0.5;
pub const BPP_INT8: f64 = 1.0;
pub const BPP_F32: f64 = 4.0;

/// Sustained decode rate (tokens/sec) the inference bound is solved for.
pub const TARGET_TOKS: f64 = 3.0;
/// Wall-clock budget for the training / eval bounds: 24 hours, in seconds.
pub const TRAIN_SECS: f64 = 24.0 * 3600.0;
/// Fixed-corpus training assumption (tokens).
pub const FIXED_TRAIN_TOKENS: f64 = 1e9;
/// Held-out eval corpus assumption (tokens).
pub const EVAL_TOKENS: f64 = 1e7;
/// Tokens-per-parameter for the compute-optimal (Chinchilla) training bound.
pub const CHINCHILLA_RATIO: f64 = 20.0;

/// Fraction of raw stream bandwidth that weight-streaming (m=1 GEMV) decode
/// actually achieves. Calibrated against benchmarks.md (~7 tok/s @ 1B int4):
/// decode reaches only a portion of peak stream bandwidth.
pub const DECODE_EFFICIENCY: f64 = 0.35;
/// Fraction of aggregate peak GEMM throughput a real training step sustains
/// (imperfect parallel scaling + non-GEMM work).
pub const TRAIN_EFFICIENCY: f64 = 0.5;

// ── Pure bound math ──────────────────────────────────────────────────────────

fn finite_pos(x: f64) -> bool {
    x.is_finite() && x > 0.0
}

/// Max parameters decodable at >= [`TARGET_TOKS`], bandwidth-bound: each decoded
/// token streams every weight once, so `tok/s = bw / (N * bytes_per_param)`.
pub fn infer_max_params(bw_bytes_per_s: f64, bytes_per_param: f64) -> f64 {
    if !finite_pos(bw_bytes_per_s) || !finite_pos(bytes_per_param) {
        return 0.0;
    }
    (bw_bytes_per_s * DECODE_EFFICIENCY) / (TARGET_TOKS * bytes_per_param)
}

/// Usable training/eval compute budget (FLOPs) within the 24h wall clock.
/// `gflops` is the aggregate GEMM throughput across all cores.
fn flop_budget(gflops: f64) -> f64 {
    if !finite_pos(gflops) {
        return 0.0;
    }
    gflops * 1e9 * TRAIN_SECS * TRAIN_EFFICIENCY
}

/// Max trainable params, compute-optimal: training FLOPs ~ `6*N*T` with
/// `T = 20*N`, so `N = sqrt(B / 120)`.
pub fn train_max_chinchilla(gflops: f64) -> f64 {
    let b = flop_budget(gflops);
    (b / (6.0 * CHINCHILLA_RATIO)).sqrt()
}

/// Max trainable params on a fixed [`FIXED_TRAIN_TOKENS`] corpus: `N = B / (6*T)`.
pub fn train_max_fixed(gflops: f64) -> f64 {
    flop_budget(gflops) / (6.0 * FIXED_TRAIN_TOKENS)
}

/// Max params evaluable (forward-only, `2*N*T`) over [`EVAL_TOKENS`] within 24h.
pub fn test_max_params(gflops: f64) -> f64 {
    flop_budget(gflops) / (2.0 * EVAL_TOKENS)
}

// ── Live micro-benchmark ─────────────────────────────────────────────────────

use std::hint::black_box;
use std::time::Instant;

/// Stream a ~256 MB buffer with a reduction to estimate usable memory
/// bandwidth (bytes/sec). Bandwidth-bound CPU decode is governed by this.
pub fn measure_mem_bandwidth() -> f64 {
    const N: usize = 64 * 1024 * 1024; // 64M f32 = 256 MB
    const REPS: usize = 4;
    let buf = vec![1.0f32; N];

    // Warm pages/caches so the timed loop measures steady-state bandwidth.
    let mut warm = 0.0f32;
    for &x in buf.iter().step_by(4096) {
        warm += x;
    }
    black_box(warm);

    let start = Instant::now();
    let mut acc = 0.0f32;
    for _ in 0..REPS {
        let mut s = 0.0f32;
        for &x in &buf {
            s += x;
        }
        acc += s;
    }
    let secs = start.elapsed().as_secs_f64();
    black_box(acc);

    if secs > 0.0 {
        (N * 4 * REPS) as f64 / secs
    } else {
        0.0
    }
}

/// Time a single-threaded square matmul (cache-friendly i-k-j order) to
/// estimate sustained GEMM throughput in GFLOP/s (FLOPs = 2*n^3).
pub fn measure_gemm_gflops() -> f64 {
    const N: usize = 512;
    let a = vec![1.0f32; N * N];
    let b = vec![1.0f32; N * N];
    let mut c = vec![0.0f32; N * N];

    let start = Instant::now();
    for i in 0..N {
        for k in 0..N {
            let aik = a[i * N + k];
            let brow = &b[k * N..k * N + N];
            let crow = &mut c[i * N..i * N + N];
            for j in 0..N {
                crow[j] += aik * brow[j];
            }
        }
    }
    let secs = start.elapsed().as_secs_f64();
    black_box(&c);

    if secs > 0.0 {
        (2.0 * (N as f64).powi(3)) / 1e9 / secs
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn infer_bound_orders_by_precision() {
        // Smaller bytes-per-param => more params fit the same bandwidth budget.
        let bw = 10e9; // 10 GB/s
        let i4 = infer_max_params(bw, BPP_INT4);
        let i8 = infer_max_params(bw, BPP_INT8);
        let f32 = infer_max_params(bw, BPP_F32);
        assert!(i4 > i8 && i8 > f32, "expected int4 > int8 > f32, got {i4} {i8} {f32}");
    }

    #[test]
    fn infer_bound_scales_with_bandwidth() {
        let lo = infer_max_params(5e9, BPP_INT8);
        let hi = infer_max_params(20e9, BPP_INT8);
        assert!(hi > lo * 3.0, "4x bandwidth should give ~4x params: {lo} {hi}");
    }

    #[test]
    fn infer_bound_anchor_is_plausible() {
        // benchmarks.md: ~1B params at int4 decodes ~7 tok/s. At 3 tok/s the
        // ceiling should sit above 1B and below ~10B for a typical ~8 GB/s CPU.
        let n = infer_max_params(8e9, BPP_INT4);
        assert!(n > 1e9 && n < 1e10, "anchor implausible: {n}");
    }

    #[test]
    fn train_bounds_scale_with_flops() {
        assert!(train_max_chinchilla(200.0) > train_max_chinchilla(50.0));
        assert!(train_max_fixed(200.0) > train_max_fixed(50.0));
        assert!(test_max_params(200.0) > test_max_params(50.0));
    }

    #[test]
    fn chinchilla_is_sqrt_shaped() {
        // 4x the FLOP budget should ~double the Chinchilla param bound (sqrt law).
        let a = train_max_chinchilla(50.0);
        let b = train_max_chinchilla(200.0);
        let ratio = b / a;
        assert!((ratio - 2.0).abs() < 0.05, "expected ~2x, got {ratio}");
    }

    #[test]
    fn degenerate_inputs_do_not_panic() {
        for &bad in &[0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(infer_max_params(bad, BPP_INT8), 0.0);
            assert_eq!(train_max_chinchilla(bad), 0.0);
            assert_eq!(train_max_fixed(bad), 0.0);
            assert_eq!(test_max_params(bad), 0.0);
        }
        assert_eq!(infer_max_params(10e9, 0.0), 0.0);
    }

    #[test]
    fn mem_bandwidth_is_positive_and_sane() {
        let bw = measure_mem_bandwidth();
        // Any real machine streams between ~0.5 GB/s and ~2 TB/s.
        assert!(bw > 5e8 && bw < 2e12, "implausible bandwidth: {bw} B/s");
    }

    #[test]
    fn gemm_throughput_is_positive_and_sane() {
        let g = measure_gemm_gflops();
        assert!(g > 0.1 && g < 5000.0, "implausible GFLOP/s: {g}");
    }
}
