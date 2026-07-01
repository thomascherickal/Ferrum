//! Machine-capability estimator: micro-benchmarks the host and derives
//! upper-bound parameter counts for inference (>=3 tok/s), training (<24h),
//! and evaluation (<24h). See docs/superpowers/specs/2026-06-23-capable-module-design.md.

use crate::AppState;
use serde::Serialize;
use std::hint::black_box;
use std::time::Instant;
use tauri::State;

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

/// Resident bytes per parameter **during training**: this engine's optimizer
/// keeps four f32 tensors per weight — the master, the gradient, and Adam's two
/// moments — so a trainable model needs ≈16 B/param in RAM (activations extra).
pub const TRAIN_BYTES_PER_PARAM: f64 = 16.0;
/// Resident bytes per parameter for forward-only **eval** (fp32 weights).
pub const EVAL_BYTES_PER_PARAM: f64 = 4.0;

// ── Pure bound math ──────────────────────────────────────────────────────────

fn finite_pos(x: f64) -> bool {
    x.is_finite() && x > 0.0
}

/// Max parameters that fit in `mem_avail` bytes at `bytes_per_param`. Returns
/// `+∞` when memory is unknown (`0`), so a later `min` leaves the compute bound
/// untouched rather than collapsing it to zero.
fn mem_bound_params(mem_avail: u64, bytes_per_param: f64) -> f64 {
    if mem_avail == 0 || !finite_pos(bytes_per_param) {
        return f64::INFINITY;
    }
    mem_avail as f64 / bytes_per_param
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
/// `gflops` is the aggregate GEMM throughput across all worker threads.
fn flop_budget(gflops: f64) -> f64 {
    if !finite_pos(gflops) {
        return 0.0;
    }
    gflops * 1e9 * TRAIN_SECS * TRAIN_EFFICIENCY
}

/// Max trainable params, compute-optimal: training FLOPs ~ `6*N*T` with
/// `T = 20*N`, so `N = sqrt(B / (6*ratio))`.
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

// ── Report assembly + Tauri command ──────────────────────────────────────────

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CapabilityReport {
    pub cpu: String,
    pub cores: usize,
    pub threads: usize,
    pub mem_total: u64,
    pub mem_avail: u64,
    /// Measured memory bandwidth (GB/s).
    pub mem_bw_gbps: f64,
    /// Aggregate GEMM throughput across all worker threads (GFLOP/s).
    pub gemm_gflops: f64,
    pub infer_int4: f64,
    pub infer_int8: f64,
    pub infer_f32: f64,
    pub train_chinchilla: f64,
    pub train_fixed1b: f64,
    pub test_eval: f64,
    // Assumptions echoed so the dialog can show its own workings.
    pub target_toks: f64,
    pub train_hours: f64,
    pub eval_tokens: f64,
    pub fixed_train_tokens: f64,
    pub chinchilla_ratio: f64,
}

/// Build a report from measured numbers (pure; no Tauri runtime needed).
/// `gemm_single` is single-thread GFLOP/s; aggregate throughput uses `threads`
/// (the engine's worker-pool size — what actually runs), not the hardware core
/// count. Training/eval param bounds are the **min of the compute bound and the
/// memory bound**, so a machine with plenty of FLOPs but little RAM is not told
/// it can train a model that won't fit.
fn assemble_report(
    cpu: String,
    cores: usize,
    threads: usize,
    mem_total: u64,
    mem_avail: u64,
    bw_bytes_per_s: f64,
    gemm_single: f64,
) -> CapabilityReport {
    let gflops = gemm_single * threads.max(1) as f64;
    let train_mem = mem_bound_params(mem_avail, TRAIN_BYTES_PER_PARAM);
    let eval_mem = mem_bound_params(mem_avail, EVAL_BYTES_PER_PARAM);
    CapabilityReport {
        cpu,
        cores,
        threads,
        mem_total,
        mem_avail,
        mem_bw_gbps: bw_bytes_per_s / 1e9,
        gemm_gflops: gflops,
        infer_int4: infer_max_params(bw_bytes_per_s, BPP_INT4),
        infer_int8: infer_max_params(bw_bytes_per_s, BPP_INT8),
        infer_f32: infer_max_params(bw_bytes_per_s, BPP_F32),
        train_chinchilla: train_max_chinchilla(gflops).min(train_mem),
        train_fixed1b: train_max_fixed(gflops).min(train_mem),
        test_eval: test_max_params(gflops).min(eval_mem),
        target_toks: TARGET_TOKS,
        train_hours: TRAIN_SECS / 3600.0,
        eval_tokens: EVAL_TOKENS,
        fixed_train_tokens: FIXED_TRAIN_TOKENS,
        chinchilla_ratio: CHINCHILLA_RATIO,
    }
}

/// Micro-benchmark the host and return capability bounds. Polled on demand by
/// the "Capable" tab.
#[tauri::command]
pub async fn capability_report(state: State<'_, AppState>) -> Result<CapabilityReport, String> {
    // Snapshot machine facts under the shared sysinfo lock, then release it.
    let (cpu, cores, mem_total, mem_avail) = {
        let mut sys = state
            .sys
            .lock()
            .map_err(|e| format!("state lock poisoned: {e}"))?;
        // Ensure the CPU list is populated before reading it; otherwise `cpus()`
        // can be empty (→ cores = 0 → all GFLOP-derived bounds collapse to 0).
        sys.refresh_cpu_all();
        sys.refresh_memory();
        let cpu = sys
            .cpus()
            .first()
            .map(|c| c.brand().trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "unknown CPU".to_string());
        (
            cpu,
            sys.cpus().len(),
            sys.total_memory(),
            sys.available_memory(),
        )
    };
    // `cores` is the live sysinfo hardware CPU count; `threads` is the engine's
    // parallel pool size (`ferrum_core::num_threads()`) — the two can differ.
    let threads = ferrum_core::num_threads();

    // Run the blocking benchmark off the async runtime thread.
    let (bw, gemm_single) =
        tauri::async_runtime::spawn_blocking(|| (measure_mem_bandwidth(), measure_gemm_gflops()))
            .await
            .map_err(|e| format!("benchmark task error: {e}"))?;

    Ok(assemble_report(
        cpu,
        cores,
        threads,
        mem_total,
        mem_avail,
        bw,
        gemm_single,
    ))
}

// ── Live micro-benchmark ─────────────────────────────────────────────────────

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
    // Non-uniform, deterministic inputs: all-ones data lets the optimizer
    // simplify the products and overstate throughput, so seed varied values
    // (and black_box them) to measure a realistic mixed-data matmul.
    let a: Vec<f32> = (0..N * N).map(|i| (i % 7) as f32 * 0.5 + 0.1).collect();
    let b: Vec<f32> = (0..N * N).map(|i| (i % 13) as f32 * 0.25 + 0.2).collect();
    let mut c = vec![0.0f32; N * N];
    black_box(&a);
    black_box(&b);

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
        assert!(
            i4 > i8 && i8 > f32,
            "expected int4 > int8 > f32, got {i4} {i8} {f32}"
        );
    }

    #[test]
    fn infer_bound_scales_with_bandwidth() {
        let lo = infer_max_params(5e9, BPP_INT8);
        let hi = infer_max_params(20e9, BPP_INT8);
        assert!(
            hi > lo * 3.0,
            "4x bandwidth should give ~4x params: {lo} {hi}"
        );
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

    #[test]
    fn report_assembles_consistent_bounds() {
        // Bypass the command (needs a Tauri runtime) and check the assembly
        // helper used by it directly. cores ≠ threads to prove the aggregate uses
        // the worker-pool size (threads), not the hardware core count.
        let r = assemble_report(
            "Test CPU".into(),
            8, // cores
            4, // threads (engine worker pool)
            16_000_000_000,
            8_000_000_000,
            8e9,  // 8 GB/s
            10.0, // 10 GFLOP/s single-thread
        );
        assert_eq!(r.cores, 8);
        assert_eq!(r.threads, 4);
        assert!(
            (r.gemm_gflops - 40.0).abs() < 1e-6,
            "aggregate = single*threads (got {})",
            r.gemm_gflops
        );
        assert!(r.infer_int4 > r.infer_int8 && r.infer_int8 > r.infer_f32);
        assert!(r.train_chinchilla > 0.0 && r.train_fixed1b > 0.0 && r.test_eval > 0.0);
        assert_eq!(r.target_toks, TARGET_TOKS);
    }

    #[test]
    fn mem_bound_is_infinite_when_memory_unknown() {
        assert_eq!(mem_bound_params(0, TRAIN_BYTES_PER_PARAM), f64::INFINITY);
        assert_eq!(mem_bound_params(16_000_000_000, 16.0), 1e9);
    }

    #[test]
    fn training_bound_is_memory_capped_when_ram_is_scarce() {
        // Huge compute, tiny RAM (1 GB): the training bound must collapse to the
        // memory ceiling (1e9 / 16 B ≈ 62.5M params), not the compute figure.
        let r = assemble_report(
            "cpu".into(),
            64,
            64,
            2_000_000_000,
            1_000_000_000,
            50e9,
            200.0,
        );
        let mem_cap = 1_000_000_000.0 / TRAIN_BYTES_PER_PARAM;
        assert!(
            (r.train_chinchilla - mem_cap).abs() < 1.0,
            "chinchilla not RAM-capped: {}",
            r.train_chinchilla
        );
        assert!(
            r.train_fixed1b <= mem_cap + 1.0,
            "fixed not RAM-capped: {}",
            r.train_fixed1b
        );
        // Compute alone (200 GFLOP/s × 64 threads) would have allowed more.
        assert!(train_max_chinchilla(200.0 * 64.0) > mem_cap);
    }

    #[test]
    fn training_bound_is_compute_limited_when_ram_is_ample() {
        // Modest compute, huge RAM: the memory cap must NOT bind — the reported
        // bound equals the pure compute figure.
        let r = assemble_report(
            "cpu".into(),
            4,
            4,
            u64::MAX,
            1_000_000_000_000_000,
            8e9,
            10.0,
        );
        let compute = train_max_chinchilla(10.0 * 4.0);
        assert!(
            (r.train_chinchilla - compute).abs() < 1.0,
            "should be compute-limited"
        );
    }
}
