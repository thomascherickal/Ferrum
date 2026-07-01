//! Dynamic CPU parallelism for the hot numeric kernels — built only on `std`.
//!
//! Ferrum has zero external dependencies and is `#![forbid(unsafe_code)]`, so
//! there is no `rayon`/`num_cpus` and no `unsafe` lifetime tricks. Instead this
//! module runs a **persistent worker pool**: a fixed set of `'static` threads is
//! spawned once (lazily, on first use) and reused for every matmul, rather than
//! creating fresh OS threads per call. This matters most for autoregressive
//! generation, which issues thousands of small matmuls whose per-call
//! thread-creation cost would otherwise dominate (and even regress at high
//! thread counts).
//!
//! Because safe Rust cannot hand a borrowed closure to threads that outlive the
//! call, kernels share their read-only inputs through [`std::sync::Arc`] (one
//! cheap clone per matmul, not per worker) and each worker computes and returns
//! an **owned** output block; the caller stitches the blocks back together.
//!
//! The design is opt-out-safe:
//!
//! * Workloads below a scalar-work threshold run serially on the calling thread
//!   ([`should_parallelize`] returns `false`), so small matmuls pay nothing.
//! * The `wasm32` target, which has no threads, always runs serially and never
//!   spawns the pool.
//! * Splitting the output by rows never changes the per-element arithmetic, so
//!   results are **bit-for-bit identical** regardless of the thread count and
//!   training/inference stay deterministic.
//!
//! GPUs are never used: all parallelism is plain CPU threads.

#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc::{channel, Sender};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc;
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

/// Total scalar multiply–add count below which threading is not worth its
/// dispatch cost; smaller outputs are computed serially.
const PARALLEL_THRESHOLD: usize = 1 << 16;

/// Whether a row-major output with `rows` rows and an estimated total scalar
/// `cost` (e.g. `m * k * n` for a matmul) is worth splitting across the worker
/// pool. Callers should run the serial path directly when this is `false` to
/// avoid the input `Arc` clone the parallel path needs.
pub fn should_parallelize(rows: usize, cost: usize) -> bool {
    rows >= 2 && cost >= PARALLEL_THRESHOLD && num_threads() > 1
}

// ─────────────────────────────────────────────────────────────────────────────
// Persistent worker pool
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(not(target_arch = "wasm32"))]
type Job = Box<dyn FnOnce() + Send + 'static>;

/// A fixed set of long-lived worker threads, each owning one job channel.
#[cfg(not(target_arch = "wasm32"))]
struct Pool {
    senders: Vec<Sender<Job>>,
}

#[cfg(not(target_arch = "wasm32"))]
static POOL: OnceLock<Pool> = OnceLock::new();

#[cfg(not(target_arch = "wasm32"))]
fn pool() -> &'static Pool {
    POOL.get_or_init(|| {
        let n = num_threads().max(1);
        let mut senders = Vec::with_capacity(n);
        for id in 0..n {
            let (tx, rx) = channel::<Job>();
            std::thread::Builder::new()
                .name(format!("ferrum-worker-{id}"))
                .spawn(move || {
                    // Park on the channel; run jobs until the sender is dropped
                    // (only at process shutdown, since `POOL` is 'static).
                    while let Ok(job) = rx.recv() {
                        job();
                    }
                })
                .expect("ferrum: failed to spawn worker thread");
            senders.push(tx);
        }
        Pool { senders }
    })
}

/// Fill a row-major `[m, n]` output by running `kernel(r0, r1, block)` over
/// contiguous row blocks on the persistent worker pool.
///
/// `kernel` must fill rows `r0..r1` into `block`, a fresh buffer of exactly
/// `(r1 - r0) * n` elements indexed locally (row `i` lives at `(i - r0) * n`).
/// Every row is produced by exactly one call, so the result does not depend on
/// the number of threads. `kernel` is shared across workers via `Arc`, so it
/// must own its inputs (typically `Arc<[f32]>` clones) — that ownership is what
/// lets the work run on threads outliving this call without any `unsafe`.
///
/// Intended to be called only when [`should_parallelize`] is `true`; it is
/// always correct, and on `wasm32` simply runs the kernel serially.
pub fn run<K>(m: usize, n: usize, kernel: K) -> Vec<f32>
where
    K: Fn(usize, usize, &mut [f32]) + Send + Sync + 'static,
{
    #[cfg(target_arch = "wasm32")]
    {
        let mut out = vec![0.0f32; m * n];
        kernel(0, m, &mut out);
        out
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let threads = num_threads().max(1).min(m.max(1));
        if threads <= 1 {
            let mut out = vec![0.0f32; m * n];
            kernel(0, m, &mut out);
            return out;
        }

        let rows_per = m.div_ceil(threads);
        let kernel = Arc::new(kernel);
        let pool = pool();
        let (tx, rx) = channel::<(usize, Vec<f32>)>();

        let mut blocks = 0usize;
        let mut r0 = 0usize;
        while r0 < m {
            let r1 = (r0 + rows_per).min(m);
            let kernel = Arc::clone(&kernel);
            let tx = tx.clone();
            let job: Job = Box::new(move || {
                let mut block = vec![0.0f32; (r1 - r0) * n];
                kernel(r0, r1, &mut block);
                let _ = tx.send((r0, block));
            });
            // One block per worker: `blocks` never exceeds `threads`.
            let _ = pool.senders[blocks % pool.senders.len()].send(job);
            blocks += 1;
            r0 = r1;
        }
        drop(tx);

        let mut out = vec![0.0f32; m * n];
        for _ in 0..blocks {
            let (r0, block) = rx.recv().expect("ferrum: worker thread disconnected");
            let start = r0 * n;
            out[start..start + block.len()].copy_from_slice(&block);
        }
        out
    }
}

/// Fill a length-`total` 1-D output by running `kernel(j0, j1, block)` over
/// contiguous column ranges on the persistent worker pool, where `block` is a
/// fresh buffer of `j1 - j0` elements indexed locally (`j` at `j - j0`).
///
/// This is the column-split counterpart of [`run`]: where `run` parallelizes a
/// `[m, n]` output by **rows** (needs `m ≥ 2`, useless for single-token decode),
/// this parallelizes a single output vector by **columns**. It is what lets the
/// autoregressive GEMV — every decode matmul has `m = 1` — use more than one
/// core. Because each output index is produced by exactly one worker and the
/// reduction over the contraction dimension happens entirely inside that worker,
/// the result is **bit-for-bit identical** regardless of the thread count.
///
/// `min_chunk` keeps workers from being handed trivially small slices; the whole
/// thing runs serially below it or when threading is unavailable.
pub fn run_1d<K>(total: usize, min_chunk: usize, kernel: K) -> Vec<f32>
where
    K: Fn(usize, usize, &mut [f32]) + Send + Sync + 'static,
{
    #[cfg(target_arch = "wasm32")]
    {
        let _ = min_chunk;
        let mut out = vec![0.0f32; total];
        kernel(0, total, &mut out);
        out
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let threads = num_threads()
            .max(1)
            .min(total.div_ceil(min_chunk.max(1)).max(1));
        if threads <= 1 {
            let mut out = vec![0.0f32; total];
            kernel(0, total, &mut out);
            return out;
        }

        let cols_per = total.div_ceil(threads);
        let kernel = Arc::new(kernel);
        let pool = pool();
        let (tx, rx) = channel::<(usize, Vec<f32>)>();

        let mut blocks = 0usize;
        let mut j0 = 0usize;
        while j0 < total {
            let j1 = (j0 + cols_per).min(total);
            let kernel = Arc::clone(&kernel);
            let tx = tx.clone();
            let job: Job = Box::new(move || {
                let mut block = vec![0.0f32; j1 - j0];
                kernel(j0, j1, &mut block);
                let _ = tx.send((j0, block));
            });
            let _ = pool.senders[blocks % pool.senders.len()].send(job);
            blocks += 1;
            j0 = j1;
        }
        drop(tx);

        let mut out = vec![0.0f32; total];
        for _ in 0..blocks {
            let (j0, block) = rx.recv().expect("ferrum: worker thread disconnected");
            out[j0..j0 + block.len()].copy_from_slice(&block);
        }
        out
    }
}

/// Whether a 1-D output of `total` elements with `per_element` scalar work each
/// is worth splitting across the pool (the GEMV/decode gate).
pub fn should_parallelize_1d(total: usize, per_element: usize) -> bool {
    total >= 2 && total.saturating_mul(per_element) >= PARALLEL_THRESHOLD && num_threads() > 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn num_threads_is_at_least_one() {
        assert!(num_threads() >= 1);
    }

    #[test]
    fn run_covers_every_row_exactly_once() {
        // A large output (above the threshold) split across the pool must fill
        // every element exactly once, regardless of how many workers ran it.
        let (m, n) = (257usize, 129usize);
        let out = run(m, n, move |r0, r1, block| {
            for i in r0..r1 {
                let o = (i - r0) * n;
                for j in 0..n {
                    block[o + j] = (i * n + j) as f32;
                }
            }
        });
        assert_eq!(out.len(), m * n);
        for (idx, &v) in out.iter().enumerate() {
            assert_eq!(v, idx as f32, "element {idx} not written exactly once");
        }
    }

    #[test]
    fn should_parallelize_gates_on_size() {
        assert!(
            !should_parallelize(1, usize::MAX),
            "single row stays serial"
        );
        assert!(!should_parallelize(1000, 0), "trivial work stays serial");
        if num_threads() > 1 {
            assert!(should_parallelize(1000, PARALLEL_THRESHOLD));
        }
    }

    #[test]
    fn run_1d_covers_every_column_exactly_once() {
        let total = 1000usize;
        let out = run_1d(total, 64, move |j0, j1, block| {
            for (idx, j) in (j0..j1).enumerate() {
                block[idx] = (j * 2) as f32;
            }
        });
        assert_eq!(out.len(), total);
        for (j, &v) in out.iter().enumerate() {
            assert_eq!(v, (j * 2) as f32, "column {j} not written exactly once");
        }
    }

    #[test]
    fn run_1d_serial_when_min_chunk_exceeds_total() {
        // min_chunk ≥ total collapses to a single serial invocation.
        let out = run_1d(50, 1_000_000, move |j0, j1, block| {
            for (idx, j) in (j0..j1).enumerate() {
                block[idx] = j as f32;
            }
        });
        assert_eq!(out.len(), 50);
        assert!(out.iter().enumerate().all(|(j, &v)| v == j as f32));
    }

    #[test]
    fn should_parallelize_1d_gates() {
        assert!(
            !should_parallelize_1d(1, usize::MAX),
            "single element stays serial"
        );
        assert!(!should_parallelize_1d(1000, 0), "no work stays serial");
        if num_threads() > 1 {
            assert!(should_parallelize_1d(PARALLEL_THRESHOLD, 1));
        }
    }
}
