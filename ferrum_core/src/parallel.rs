//! Dynamic CPU parallelism for the hot numeric kernels — built only on `std`.
//!
//! Ferrum has zero external dependencies, so there is no `rayon` or `num_cpus`.
//! This module detects the machine's parallelism once via
//! [`std::thread::available_parallelism`] (overridable with the
//! `FERRUM_NUM_THREADS` environment variable) and splits row-major matrix
//! outputs into contiguous row blocks computed on scoped threads.
//!
//! The design is opt-out-safe:
//!
//! * Workloads below a scalar-work threshold run serially (thread spawn cost
//!   would dominate a small matmul).
//! * The `wasm32` target, which has no thread support, always runs serially.
//! * Splitting the output by rows never changes the per-element arithmetic, so
//!   results are **bit-for-bit identical** regardless of the thread count and
//!   training/inference stay deterministic.
//!
//! GPUs are never used: all parallelism is plain CPU threads.

use std::sync::OnceLock;

static NUM_THREADS: OnceLock<usize> = OnceLock::new();

/// The worker-thread count Ferrum uses for parallel kernels, detected once and
/// cached for the lifetime of the process.
///
/// Resolution order:
/// 1. `FERRUM_NUM_THREADS` environment variable, if set to a positive integer.
/// 2. [`std::thread::available_parallelism`] (the CPU's reported parallelism).
/// 3. `1` if neither is available.
///
/// Always `1` on `wasm32`, which has no threads.
pub fn num_threads() -> usize {
    *NUM_THREADS.get_or_init(detect)
}

#[cfg(target_arch = "wasm32")]
fn detect() -> usize {
    1
}

#[cfg(not(target_arch = "wasm32"))]
fn detect() -> usize {
    if let Ok(v) = std::env::var("FERRUM_NUM_THREADS") {
        if let Ok(n) = v.parse::<usize>() {
            if n >= 1 {
                return n;
            }
        }
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// Total scalar multiply–add count below which threading is not worth its spawn
/// cost; smaller outputs are computed serially.
const PARALLEL_THRESHOLD: usize = 1 << 16;

/// Fill a row-major `[m, n]` output by running `f(first_row, block)` over
/// contiguous blocks of rows, in parallel across CPU threads when the workload
/// justifies it.
///
/// `cost` is an estimate of the total scalar work (e.g. `m * k * n` for a
/// matmul); `out` must contain exactly `m * n` elements. `f` receives the
/// global index of a block's first row and that block's mutable slice of `out`
/// (whose length is a whole number of `n`-wide rows). Every output element is
/// written by exactly one invocation of `f`, so the result does not depend on
/// the number of threads.
pub fn for_row_blocks<F>(m: usize, n: usize, cost: usize, out: &mut [f32], f: F)
where
    F: Fn(usize, &mut [f32]) + Sync,
{
    debug_assert_eq!(out.len(), m * n, "out must hold exactly m*n elements");

    let threads = num_threads().min(m.max(1));
    let parallel = threads > 1 && n > 0 && cost >= PARALLEL_THRESHOLD;

    #[cfg(not(target_arch = "wasm32"))]
    if parallel {
        let rows_per = m.div_ceil(threads);
        let block_len = rows_per * n;
        std::thread::scope(|s| {
            let f = &f;
            let mut row0 = 0usize;
            for block in out.chunks_mut(block_len) {
                let start = row0;
                s.spawn(move || f(start, block));
                row0 += rows_per;
            }
        });
        return;
    }

    let _ = parallel; // used only on native; silences the wasm warning
    f(0, out);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn num_threads_is_at_least_one() {
        assert!(num_threads() >= 1);
    }

    #[test]
    fn for_row_blocks_covers_every_row_once() {
        // A tiny serial workload (below threshold) and a large one must both
        // fill the whole output exactly once.
        for &(m, n) in &[(3usize, 4usize), (257, 129)] {
            let mut out = vec![0.0f32; m * n];
            let cost = m * n * 64; // large enough to cross the threshold for the big case
            for_row_blocks(m, n, cost, &mut out, |row0, block| {
                let rows = block.len() / n;
                for li in 0..rows {
                    let i = row0 + li;
                    for j in 0..n {
                        block[li * n + j] = (i * n + j) as f32;
                    }
                }
            });
            for (idx, &v) in out.iter().enumerate() {
                assert_eq!(v, idx as f32, "element {idx} not written exactly once");
            }
        }
    }
}
