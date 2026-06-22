# Benchmarks

CPU parallelism benchmarks for Ferrum's training and inference. All numbers are
**measured on this project**, CPU-only (no GPU), with the zero-dependency
`std`-only engine.

## Environment

| Item        | Value                                              |
|-------------|----------------------------------------------------|
| Cores       | 8 (logical, via `nproc`)                           |
| Build       | `cargo build --release` (LTO, 1 codegen unit)      |
| Binary      | `slm_cli` → `train_transformer`                    |
| Tokenizer   | byte-level BPE                                      |
| Thread knob | `FERRUM_NUM_THREADS` (0/unset = auto-detect all cores) |

Thread count is detected dynamically from `std::thread::available_parallelism()`
and overridden per-run with `FERRUM_NUM_THREADS`. Two parallelism implementations
are compared:

- **Per-call spawn** — the initial design, which spawned fresh threads
  (`std::thread::scope`) for every matmul.
- **Persistent pool** — the current design, a fixed set of worker threads spawned
  once and reused for every matmul (`ferrum_core/src/parallel.rs`).

All results are **deterministic**: model files and generated text are
**byte-for-byte identical across every thread count** (verified with `md5sum` and
`cmp`). Parallelism never changes the numerical result.

---

## 1. Training — thread scaling (per-call spawn)

First parallelization benchmark. Model: `context 24, embed 96, heads 4,
blocks 3, hidden 192, BPE vocab 400, 40 epochs, seed 7`. Wall-clock for a full
training run.

| `FERRUM_NUM_THREADS` | wall time | speedup |
|----------------------|-----------|---------|
| 1                    | 156.84 s  | 1.00×   |
| 8 (all cores)        | 67.89 s   | **2.31×** |

The 1-thread and 8-thread model files were byte-for-byte identical.

The speedup is well below 8× because the sequential SGD batch loop, the small
per-head attention matmuls, softmax, and optimizer updates remain serial — only
the dominant Linear/FFN/LM-head matmuls parallelize.

---

## 2. Training — per-call spawn vs. persistent pool

Same model in both runs: `context 32, embed 128, heads 8, blocks 4, hidden 256,
BPE vocab 512 (322 learned tokens), 25 epochs, seed 7`, default thread count
(all 8 cores).

| Implementation      | wall time | speedup vs per-call |
|---------------------|-----------|---------------------|
| Per-call spawn      | 114.72 s  | 1.00×               |
| Persistent pool     | 51.25 s   | **2.24×**           |

Reusing threads instead of re-spawning them per matmul roughly halved training
time for this configuration.

---

## 3. Generation — thread scaling, per-call spawn vs. persistent pool

Model: `8 layers, embed 128, 4 blocks, 8 heads, context 32, BPE vocab 322`.
Workload: generate **4000 characters**, `--temp 0.7 --gen-seed 42` (a fixed
generation seed, so every run does identical work). Best of 2 runs per cell.

| `FERRUM_NUM_THREADS` | per-call spawn | persistent pool | pool speedup vs 1-thread pool |
|----------------------|----------------|-----------------|-------------------------------|
| 1                    | 24.49 s        | 7.48 s          | 1.00×                          |
| 2                    | 17.41 s        | 6.66 s          | 1.12×                          |
| 4                    | 16.81 s        | **5.98 s**      | **1.25×** (best)               |
| 8                    | 20.42 s (regressed) | 7.22 s     | 1.03× (no regression)          |

Output was byte-for-byte identical across all thread counts and both
implementations.

### What this shows

- **The high-thread-count regression is fixed.** With per-call spawning, 8
  threads (20.42 s) was *slower* than 1 thread (24.49 s would have been the
  serial baseline, and 8 threads regressed against the 4-thread result of
  16.81 s) — thousands of generated tokens × ~25 matmuls each × 8
  `pthread_create`s per matmul dominated. The persistent pool removes that
  per-call thread-creation cost, so 8 threads (7.22 s) no longer regresses.
- **Generation parallelizes modestly.** Best is ~1.25× at 4 threads. Generation
  is autoregressive — a long sequence of small, latency-bound matmuls
  (`context_len = 32` rows each) — so matmul-level parallelism has a low ceiling.
  The pool's main value for generation is eliminating the spawn overhead, not
  large multi-core scaling.
- **Sweet spot:** `FERRUM_NUM_THREADS=2`–`4` for this inference workload.

---

## 4. Matmul kernel — GEMM throughput & decode bandwidth

The sections above time end-to-end training and generation of small models. This
one isolates the kernel that dominates both — `ops::matmul` — at **synthetic
~1B-class shapes**, to gauge how the current scalar kernels would behave for a
much larger SLM. It uses a std-only microbenchmark
(`ferrum_core/benches/gemm.rs`, `harness = false`, no Criterion), reporting the
best of many timed iterations on random data. These are raw-kernel numbers, **not
a trained Ferrum model**; the shapes (`d_model = 2048`, `d_ff = 8192`,
`vocab = 32000`, 16 layers) stand in for a ~1B-parameter model.

### 4a. Square GEMM — `C[m×n] = A[m×k]·B[k×n]` (compute-bound)

Large square multiplies cross the internal work threshold, so they run on the
persistent pool. Reported as GFLOP/s (`2·m·k·n / time`).

| Size  | 8 threads        | 1 thread        | thread speedup |
|-------|------------------|-----------------|----------------|
| 256²  | 44.9 GFLOP/s     | 17.4 GFLOP/s    | 2.6×           |
| 512²  | 45.8 GFLOP/s     | 15.9 GFLOP/s    | 2.9×           |
| 1024² | 47.9 GFLOP/s     | 13.4 GFLOP/s    | 3.6×           |
| 2048² | **23.5 GFLOP/s** | **6.1 GFLOP/s** | 3.9×           |

- **Cache cliff at 2048².** Throughput holds ~46–48 GFLOP/s through 1024², then
  roughly halves at 2048², where each matrix (16 MB in f32) overflows cache and
  the untiled i-k-j kernel re-streams `B` from memory. A cache-tiled kernel would
  not fall off here.
- **Vector units sit idle.** Peak ~48 GFLOP/s across 8 cores is ~6 GFLOP/s/core —
  about what scalar f32 yields; a single AVX2+FMA core can do several times this.
  The kernels carry no explicit SIMD.

### 4b. Decode GEMV — `c[1×n] = a[1×k]·W[k×n]` (the autoregressive hot path)

With a KV cache, each generated token feeds **one row** through the network, so
every decode matmul has `m = 1`. `should_parallelize` requires `rows ≥ 2`, so
these run **serial on one core regardless of thread count**. Reported as GB/s of
weight `W` streamed (the quantity that bounds bandwidth-limited decode).

| Weight shape          | 8 threads  | 1 thread   |
|-----------------------|------------|------------|
| attn proj `2048×2048` | 17.9 GB/s  | 17.8 GB/s  |
| ffn up `2048×8192`    | 15.4 GB/s  | 16.7 GB/s  |
| ffn down `8192×2048`  | 16.3 GB/s  | 15.5 GB/s  |
| logits `2048×32000`   | 15.0 GB/s  | 14.6 GB/s  |

The two columns are identical within noise — **direct confirmation that
single-token decode never uses more than one core**, and that it is bound by
memory bandwidth (~15–18 GB/s, one core's share) rather than compute.

### 4c. Synthesized decode step (~1B-class)

One token through 16 layers (4 attention projections + FFN up/down each) plus the
output projection. **Estimate** — excludes attention score·V matmuls, layernorm,
softmax, and sampling; one layer's weights are replayed (so they may stay warmer
in cache than a true 16-layer model).

| Metric                 | 8 threads | 1 thread  |
|------------------------|-----------|-----------|
| ms/token               | 268       | 250       |
| tokens/sec             | 3.7       | 4.0       |
| weights streamed/token | 3.48 GB   | 3.48 GB   |
| effective bandwidth    | 13.0 GB/s | 13.9 GB/s |

The 1-thread run is marginally **faster** (no pool-dispatch overhead), since the
`m = 1` GEMVs cannot use the extra cores anyway.

### What this shows

- **Decode is bandwidth-bound and single-threaded.** The 3.48 GB streamed per
  token at f32 sets the ceiling; at ~14 GB/s that is ~4 tok/s. Quantizing weights
  and consuming them in the kernel cuts the bytes — int8 ≈ ¼, int4 ≈ ⅛ — and
  lifts the ceiling proportionally (int4 ≈ ~30 tok/s here, from bandwidth alone).
- **The pool helps prefill/training, not single-token decode.** Row-split
  parallelism needs `m ≥ 2`; splitting GEMV along columns (`n`) or `k` would put
  the idle cores to work on the decode path.
- **The scalar kernel leaves a large SIMD/tiling factor on the table**, most
  visible as the 2048² cache cliff in §4a.

This sharpens §3: the "low ceiling" for generation is, for single-token steps
specifically, no parallelism at all.

### 4d. What changed (the three bottlenecks above are now addressed in code)

The three levers this section identified are now implemented (see the `Opt#1/2/3`
references in the source). These are **structural** changes, not yet re-measured
on this machine — re-run `cargo bench --bench gemm` and a real int4 decode to put
numbers here:

- **int4/int8 weights are kept packed in memory and consumed directly**
  (`quant::QWeight`, `ops::qlinear`, and the loader, which no longer expands
  quantized matrices to f32). This cuts both the resident footprint *and* the
  bytes streamed per token — the only lever that raises the bandwidth-bound
  ceiling. A ~1B model is ~0.5 GB resident at int4 vs ~3.7 GB at f32.
- **Single-token decode now uses every core.** `ops::qlinear` splits the `m = 1`
  GEMV across the worker pool by output column (`parallel::run_1d`), so the
  decode path is no longer pinned to one core. The split is deterministic
  (bit-stable across thread counts).
- **The Linear epilogue is fused and the GEMM is cache-tiled** (`ops::linear_forward`,
  `matmul_block`): no `add_bias` clone per call, and a `KC×NC` panel of `B` is
  reused across rows to remove the 2048² cliff in §4a. The fused path is
  bit-identical to `add_bias(matmul(..))`.

GGUF weights can be imported into this int4/int8 path with the std-only reader in
`gguf.rs`. Llama/Qwen-family checkpoints are **runnable**: `llm.rs` implements
RMSNorm, RoPE, grouped-query attention (KV-cached), and the SwiGLU FFN, and
`Gguf::load_llama` maps a `llama`/`qwen2` GGUF onto a runnable `LlamaModel`
(weights packed to int4/int8). Bit-exact parity with llama.cpp on a real
checkpoint is not asserted (it needs the actual multi-GB file), but every
primitive is unit-covered and the imported model's KV-cached decode matches its
own full forward.

---

## Reproducing

```bash
cargo build --release -p slm_cli
BIN=target/release/train_transformer

# Prepare a corpus (any UTF-8 text), then train.
$BIN train corpus.txt model.bin --vocab 512 --context 32 \
    --embed 128 --heads 8 --blocks 4 --hidden 256 --epochs 25 --seed 7

# Time generation at a fixed thread count (identical work via --gen-seed).
FERRUM_NUM_THREADS=4 time \
    $BIN generate model.bin "the quick brown fox jumps over the lazy dog while" \
    --chars 4000 --temp 0.7 --gen-seed 42 > out.txt

# Verify determinism: the output is identical regardless of thread count.
for t in 1 2 4 8; do
  FERRUM_NUM_THREADS=$t $BIN generate model.bin "seed text here long enough" \
      --chars 2000 --temp 0.7 --gen-seed 42 > out_$t.txt
done
md5sum out_*.txt   # all hashes identical
```

Matmul kernel microbenchmark (§4), std-only, no Criterion:

```bash
cargo bench --bench gemm                        # auto-detected threads
FERRUM_NUM_THREADS=1 cargo bench --bench gemm    # force serial
cargo bench --bench gemm -- 512 1024 4096        # custom square GEMM sizes
```

---

## Caveats

- Timings are single-machine, wall-clock, best-of-two; treat them as indicative,
  not precise. The controlled comparisons are within a single session (same
  build, same model, same conditions): §2 (training, per-call vs pool) and §3
  (generation thread scaling).
- Absolute times depend heavily on model size, context length, vocabulary, and
  corpus. Larger Linear/FFN dimensions parallelize better; tiny models stay
  serial by design (below the internal work threshold).
- No GPU is used anywhere. All parallelism is plain CPU threads built on `std`
  only, with no `unsafe`.
- §4 measures the raw `ops::matmul` kernel on **synthetic ~1B-class shapes with
  random data**, not a trained model; its decode-step figure is an estimate (see
  §4c). The §4 GFLOP/s and GB/s numbers are best-of-many-iterations, lower
  variance than the wall-clock best-of-two above, but still single-machine.
