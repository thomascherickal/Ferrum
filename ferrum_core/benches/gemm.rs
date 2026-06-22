//! Matrix-multiply benchmarks for Ferrum — the kernel that dominates both
//! training and inference. Std-only and `harness = false`, so it pulls in **no**
//! external dependency (no Criterion): it just times [`ferrum_core::ops::matmul`]
//! with [`std::time::Instant`], matching the measured-on-this-project style of
//! `benchmarks.md`.
//!
//! Run:
//!   cargo bench --bench gemm
//!   FERRUM_NUM_THREADS=1 cargo bench --bench gemm   # force serial, for scaling
//!   cargo bench --bench gemm -- 512 1024 4096        # custom square GEMM sizes
//!
//! Three sections, each illustrating a point from the 1B-SLM analysis:
//!   1. Square GEMM   `C[m×n] = A[m×k]·B[k×n]` — compute-bound, crosses the
//!      parallel threshold, so it exercises the persistent worker pool. Reported
//!      as GFLOP/s.
//!   2. Decode GEMV   `c[1×n] = a[1×k]·W[k×n]` — the autoregressive hot path.
//!      Because `m == 1`, `should_parallelize` returns `false`, so this runs
//!      **serial on one core regardless of `FERRUM_NUM_THREADS`**. It is
//!      bandwidth-bound, so it is reported as GB/s of weight streamed.
//!   3. Synthesized decode step — replays the per-layer GEMVs of a ~1B-class
//!      config to get a measured ms/token and tokens/sec (an estimate; see the
//!      caveats it prints).

use std::hint::black_box;
use std::time::{Duration, Instant};

use ferrum_core::ops;
use ferrum_core::Tensor;

/// Fill an `rows × cols` tensor with deterministic pseudo-random values in
/// `[-1, 1)` (xorshift64 — no `rng` coupling, no all-equal data that could let
/// the optimizer or cache behave unrealistically).
fn rand_matrix(rows: usize, cols: usize, seed: &mut u64) -> Tensor {
    let n = rows * cols;
    let mut data = Vec::with_capacity(n);
    for _ in 0..n {
        let mut x = *seed;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *seed = x;
        // top 24 bits → [0, 2) → [-1, 1)
        data.push((x >> 40) as f32 / (1u64 << 23) as f32 - 1.0);
    }
    Tensor::matrix(rows, cols, data).expect("matrix dims")
}

/// Run `f` once to warm up, then repeatedly until `budget` elapses (always at
/// least one timed iteration). Returns the best (minimum) observed duration —
/// the standard low-noise estimate for a throughput micro-benchmark — and the
/// iteration count.
fn time_best<F: FnMut()>(budget: Duration, mut f: F) -> (Duration, u64) {
    f(); // warm up: allocator, caches, lazy worker-pool spawn
    let mut best = Duration::MAX;
    let mut iters = 0u64;
    let start = Instant::now();
    loop {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed());
        iters += 1;
        if start.elapsed() >= budget {
            break;
        }
    }
    (best, iters)
}

fn gemm_row(m: usize, k: usize, n: usize, best: Duration, iters: u64) {
    let s = best.as_secs_f64();
    let gflops = 2.0 * m as f64 * k as f64 * n as f64 / s / 1e9;
    println!(
        "  [{m:>5}×{k:<5}]·[{k:>5}×{n:<5}]  {:>9.3} ms  {:>7.1} GFLOP/s   ({iters} iters)",
        s * 1e3,
        gflops,
    );
}

fn gemv_row(label: &str, k: usize, n: usize, best: Duration, iters: u64) {
    let s = best.as_secs_f64();
    // Decode is dominated by streaming the weight matrix W[k×n] once per token.
    let bytes = k as f64 * n as f64 * 4.0;
    let gbps = bytes / s / 1e9;
    println!(
        "  {label:<20} W[{k:>5}×{n:<5}]  {:>9.1} µs/call  {:>6.1} GB/s   ({iters} iters)",
        s * 1e6,
        gbps,
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let budget = Duration::from_millis(1200);
    let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;

    println!("Ferrum matmul benchmark — std-only, zero external deps");
    println!(
        "threads = {}   (FERRUM_NUM_THREADS = {})",
        ferrum_core::num_threads(),
        std::env::var("FERRUM_NUM_THREADS").unwrap_or_else(|_| "unset".into()),
    );
    println!();

    // ── 1. Square GEMM: compute-bound, uses the worker pool ──────────────────
    println!("== Square GEMM   C[m×n] = A[m×k]·B[k×n]   (compute-bound; m≥2 ⇒ parallel) ==");
    // Only positive integers are size overrides; ignore anything else (cargo
    // passes harness flags like `--bench` to this binary). Fall back to defaults
    // when no numeric size was given.
    let parsed: Vec<usize> = args
        .iter()
        .filter_map(|a| a.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .collect();
    let sizes: Vec<usize> = if parsed.is_empty() {
        vec![256, 512, 1024, 2048]
    } else {
        parsed
    };
    for &s in &sizes {
        let a = rand_matrix(s, s, &mut seed);
        let b = rand_matrix(s, s, &mut seed);
        let (best, iters) = time_best(budget, || {
            let c = ops::matmul(black_box(&a), black_box(&b)).unwrap();
            black_box(c);
        });
        gemm_row(s, s, s, best, iters);
    }
    println!();

    // ── 2. Decode GEMV: m=1 ⇒ serial, bandwidth-bound ────────────────────────
    println!("== Decode GEMV   c[1×n] = a[1×k]·W[k×n]   (m=1 ⇒ SERIAL on one core; bandwidth-bound) ==");
    let shapes = [
        (2048usize, 2048usize, "attn q/k/v/o proj"),
        (2048, 8192, "ffn up"),
        (8192, 2048, "ffn down"),
        (2048, 32000, "output logits"),
    ];
    for &(k, n, label) in &shapes {
        let a = rand_matrix(1, k, &mut seed);
        let w = rand_matrix(k, n, &mut seed);
        let (best, iters) = time_best(budget, || {
            let c = ops::matmul(black_box(&a), black_box(&w)).unwrap();
            black_box(c);
        });
        gemv_row(label, k, n, best, iters);
    }
    println!();

    // ── 3. Synthesized ~1B decode step (estimate) ────────────────────────────
    // d_model=2048, d_ff=8192, layers=16, vocab=32000. Per token per layer:
    // 4 attention projections + FFN up + FFN down, then one output projection.
    // Weights from a single layer are replayed across all layers (keeps the
    // benchmark's resident set ~0.4 GB instead of ~3 GB); see caveats below.
    const D_MODEL: usize = 2048;
    const D_FF: usize = 8192;
    const LAYERS: usize = 16;
    const VOCAB: usize = 32000;

    let a_dmodel = rand_matrix(1, D_MODEL, &mut seed);
    let a_dff = rand_matrix(1, D_FF, &mut seed);
    let w_proj = rand_matrix(D_MODEL, D_MODEL, &mut seed);
    let w_up = rand_matrix(D_MODEL, D_FF, &mut seed);
    let w_down = rand_matrix(D_FF, D_MODEL, &mut seed);
    let w_logits = rand_matrix(D_MODEL, VOCAB, &mut seed);

    println!("== Synthesized decode step   d_model={D_MODEL}, d_ff={D_FF}, layers={LAYERS}, vocab={VOCAB} ==");
    let (best, iters) = time_best(Duration::from_millis(2000), || {
        for _ in 0..LAYERS {
            for _ in 0..4 {
                black_box(ops::matmul(black_box(&a_dmodel), black_box(&w_proj)).unwrap());
            }
            black_box(ops::matmul(black_box(&a_dmodel), black_box(&w_up)).unwrap());
            black_box(ops::matmul(black_box(&a_dff), black_box(&w_down)).unwrap());
        }
        black_box(ops::matmul(black_box(&a_dmodel), black_box(&w_logits)).unwrap());
    });
    let s = best.as_secs_f64();
    // Weight bytes streamed per token (the lower bound that sets the decode ceiling).
    let weight_bytes = ((4 * D_MODEL * D_MODEL + D_MODEL * D_FF + D_FF * D_MODEL) * LAYERS
        + D_MODEL * VOCAB) as f64
        * 4.0;
    println!(
        "  {:>8.1} ms/token   {:>6.2} tok/s   {:>6.1} GB/s effective   ({iters} iters)",
        s * 1e3,
        1.0 / s,
        weight_bytes / s / 1e9,
    );
    println!(
        "  weights streamed/token ≈ {:.2} GB  (this is the f32 footprint; int8 ⇒ ¼, int4 ⇒ ⅛)",
        weight_bytes / 1e9,
    );
    println!();
    println!("  NOTE: estimate. Excludes attention score·V matmuls, layernorm, softmax, and");
    println!("  sampling; one layer's weights are replayed, so they may stay warmer in cache");
    println!("  than a true {LAYERS}-layer model. Re-run with FERRUM_NUM_THREADS=1 — the GEMV");
    println!("  sections will not change, because m=1 never parallelizes.");
}
