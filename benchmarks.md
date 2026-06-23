# Benchmarks

CPU parallelism and quantized-decode benchmarks for Ferrum. All numbers are
**measured on this project**, CPU-only (no GPU), with the zero-dependency
`std`-only engine. **Re-measured 2026-06-23** on a clean machine state.

## Environment

| Item        | Value                                                  |
|-------------|--------------------------------------------------------|
| CPU         | 11th Gen Intel i5-1135G7 @ 2.40 GHz (4 cores / 8 threads) |
| RAM         | 16 GB                                                  |
| Build       | `cargo build --release` (LTO, 1 codegen unit), rustc 1.96 |
| Binary      | `slm_cli` → `train_transformer`                        |
| Tokenizer   | byte-level BPE / character-level as noted              |
| Thread knob | `FERRUM_NUM_THREADS` (0/unset = auto-detect all cores) |

Two parallelism knobs are independent and easy to confuse:

- **Matmul worker pool** (`FERRUM_NUM_THREADS`) — a fixed set of threads spawned
  once and reused for every matmul (Linear/FFN/attention/LM head). Splits output
  **rows**, and for single-token decode (`m = 1`) splits output **columns**
  (`run_1d`). Used by both training and generation.
- **Data-parallel training shards** (CLI `--threads`) — splits each minibatch
  across `std::thread::scope` workers, reducing gradients in a fixed order.

> Historical note: the matmul pool replaced an earlier design that spawned fresh
> threads per matmul. That per-call-spawn implementation no longer exists in the
> tree, so the numbers below are all the **current persistent-pool** design.

### Determinism (verified this run)

- **Generation output is byte-for-byte identical across thread counts.** Generating
  2000 characters at `FERRUM_NUM_THREADS=1,2,4,8` produced four identical files
  (same `md5sum`). Generation uses only the matmul pool, whose row/column split
  never reorders arithmetic.
- **Training is deterministic per configuration.** The same config trained twice
  (`--threads 4`) produced byte-identical model files. Varying *only* the matmul
  pool (`FERRUM_NUM_THREADS` 8 vs 1) with the data-parallel shard count held fixed
  also produced byte-identical models.
- **Caveat, stated precisely:** changing the *data-parallel shard count* can change
  the low bits of the trained weights, because summing gradients in a different
  number of groups is floating-point non-associative. Each shard count is itself
  deterministic, and one shard is bit-identical to the serial trainer — but 8
  shards need not equal 1 shard bit-for-bit. Generation has no such caveat.

---

## 1. Training — thread scaling (persistent pool)

End-to-end wall-clock for a full training run. Model: `context 24, embed 96,
heads 4, blocks 3, hidden 192` was attempted but is impractically slow on this
loaded machine; the figures below use a deliberately **small, fast** config —
`context 16, embed 64, heads 4, blocks 2, hidden 128, char-level, 10 epochs` on a
~5 KB corpus — so the run completes in under a minute.

| Threads (`--threads` + pool) | wall time | speedup |
|------------------------------|-----------|---------|
| 1 (serial)                   | 44.65 s   | 1.00×   |
| 8 (all cores)                | 36.50 s   | **1.22×** |

**Read this with §4a in mind.** For a *small* model the end-to-end speedup is
modest, because much of a training step is serial (the SGD loop, softmax, the tiny
per-head attention matmuls, the optimizer update) and the data-parallel path also
pays a per-shard cost of cloning the network. The *kernel* that actually
parallelizes — large matmul — scales 2.4–4.0× on its own (§4a); bigger models,
where those matmuls dominate each step, approach that ceiling. So treat 1.22× as a
floor for tiny models, not the ceiling for realistic ones.

---

## 2. Generation — thread scaling (persistent pool)

Workload: generate **2000 characters** from the model above, `--temp 0.7
--gen-seed 42` (fixed seed, so every run does identical work).

| `FERRUM_NUM_THREADS` | wall time | speedup |
|----------------------|-----------|---------|
| 1                    | 0.73 s    | 1.00×   |
| 2                    | 0.71 s    | 1.03×   |
| 4                    | 0.80 s    | 0.91×   |
| 8                    | 0.67 s    | 1.09×   |

Output was byte-for-byte identical across all four thread counts.

**What this shows.** Generation of a small model **barely parallelizes** — it is a
long chain of small, dependent matmuls (each only `context_len` rows wide), most
below the internal work threshold, so adding cores does little and can even cost a
little (the 4-thread dip is dispatch overhead). For small SLMs, the way to faster
generation is a BPE vocabulary (fewer steps per character) and a smaller network,
not more threads. Where thread-level parallelism *does* help decode is the
quantized `m = 1` column split at large (1B-class) widths — see §4d.

---

## 3. Matmul kernel — GEMM throughput & decode bandwidth

The sections above time end-to-end small models. This one isolates the kernel that
dominates both — `ops::matmul` / `ops::qlinear` — at **synthetic ~1B-class shapes**
(`d_model = 2048`, `d_ff = 8192`, `vocab = 32000`, 16 layers), to gauge how the
scalar kernels would behave for a much larger model. Std-only microbenchmark
(`ferrum_core/benches/gemm.rs`, no Criterion), best of many timed iterations on
random data. These are **raw-kernel** numbers, not a trained model. Run with
`cargo bench --bench gemm`.

### 3a. Square GEMM — `C[m×n] = A[m×k]·B[k×n]` (compute-bound)

| Size  | 8 threads        | 1 thread        | thread speedup |
|-------|------------------|-----------------|----------------|
| 256²  | **68.0 GFLOP/s** | 17.2 GFLOP/s    | 4.0×           |
| 512²  | 52.9 GFLOP/s     | 16.1 GFLOP/s    | 3.3×           |
| 1024² | 42.3 GFLOP/s     | 13.5 GFLOP/s    | 3.1×           |
| 2048² | 35.7 GFLOP/s     | 14.6 GFLOP/s    | 2.4×           |

- **The 2048² cache cliff is gone.** The cache-tiled kernel holds ~36 GFLOP/s at
  2048² — in line with smaller sizes — instead of collapsing when each 16 MB matrix
  overflows cache. (An earlier *contended* run, with the test suite hogging the
  CPU, briefly showed 13.7 here; on a clean machine it is 35.7.)
- **Vector units still sit idle.** ~4–8 GFLOP/s/core is about what scalar f32
  yields; an AVX2+FMA core could do several times this. The kernels carry no
  explicit SIMD — that is the remaining headroom.

### 3b. Decode GEMV (f32) — `c[1×n] = a[1×k]·W[k×n]` (the autoregressive hot path)

With a KV cache each generated token feeds **one row** through the network, so every
decode matmul has `m = 1`. The plain f32 `matmul` requires `rows ≥ 2` to
parallelize, so f32 decode runs **serial regardless of thread count**. GB/s of f32
weight `W` streamed:

| Weight shape          | 8 threads  | 1 thread   |
|-----------------------|------------|------------|
| attn proj `2048×2048` | 8.2 GB/s   | 8.0 GB/s   |
| ffn up `2048×8192`    | 7.4 GB/s   | 6.2 GB/s   |
| ffn down `8192×2048`  | 6.0 GB/s   | 5.8 GB/s   |
| logits `2048×32000`   | 5.5 GB/s   | 5.9 GB/s   |

The two columns match within noise — **f32 single-token decode never uses more than
one core** (the quantized path below fixes this).

### 3c. Synthesized f32 decode step (~1B-class)

One token through 16 layers (4 attention projections + FFN up/down each) plus the
output projection. **Estimate** — excludes attention score·V, layernorm, softmax,
RoPE, and sampling; one layer's weights are replayed (so they may stay warmer in
cache than a true 16-layer model).

| Metric                 | 8 threads | 1 thread  |
|------------------------|-----------|-----------|
| ms/token               | 632.8     | 713.1     |
| tokens/sec             | **1.58**  | **1.40**  |
| weights streamed/token | 3.48 GB   | 3.48 GB   |

About **1.4–1.6 tok/s** at f32 — bandwidth-bound, with the `m = 1` GEMVs unable to
use extra cores (the small 8-vs-1 difference is the column-split helping the few
wide projections, not the f32 row-split).

### 3d. Quantized decode (packed weights + column-split GEMV)

`ops::qlinear` keeps weights packed (int8/int4) and splits the `m = 1` GEMV across
the pool by output column. GB/s is over the **packed** bytes.

| Weight shape (8 threads) | int8       | int4       |
|--------------------------|------------|------------|
| attn proj `2048×2048`    | 12.2 GB/s  | 6.0 GB/s   |
| ffn up `2048×8192`       | 17.0 GB/s  | 6.4 GB/s   |
| ffn down `8192×2048`     | 7.4 GB/s   | 5.6 GB/s   |
| logits `2048×32000`      | 21.1 GB/s  | 6.5 GB/s   |

Synthesized ~1B decode step:

| Metric      | f32 (8t) | int4 (8t)  | int4 (1t)  |
|-------------|----------|------------|------------|
| ms/token    | 632.8    | **144.1**  | 256.1      |
| tokens/sec  | 1.58     | **6.94**   | 3.90       |
| streamed/tok| 3.48 GB  | 0.44 GB    | 0.44 GB    |

**The int4 split-half fix.** int4 weights are packed so that byte `b`'s low nibble
is column `b` and its high nibble is column `half + b`. That makes each nibble lane
a **contiguous, unit-stride** column range, so `qaccum_cols` decodes int4 with the
same vectorizable `out[c] += a·sext(nibble)` loop as int8 — the alternative
interleaved packing defeats the autovectorizer and runs several times slower. With
the fix the synthesized **int4 step reaches ~6.94 tok/s, ~4.4× the f32 rate**, near
the ⅛-byte bandwidth ideal.

Guidance the numbers support:

- **int4 is the right default for the GGUF runner** (`--quant int4`): ~4–5× faster
  than f32 while using **half int8's RAM**.
- **int8 is marginally faster *per call*** (a raw `i8` load beats a nibble unpack:
  note int8's higher GB/s and that int8 and int4 have similar per-call *time*
  despite int4 moving half the bytes). Pick `--quant int8` when RAM is ample. A
  SIMD/LUT nibble unpack could make int4 strictly fastest.
- **The column split works:** it spreads the `m = 1` GEMV across cores for both
  int8 and int4 (unlike the serial f32 path in §3b).
- **Net for a 1B model on this CPU:** ~7 tok/s decode at int4 and tens of seconds of
  compute-bound prefill per real prompt. Usable, not interactive — see
  [ferrum_review.md](ferrum_review.md) §4.

GGUF weights feed this path through the std-only reader in `gguf.rs` (now including
the **Q4_K/Q5_K/Q6_K** super-block formats) and the model's own tokenizer via
`gguf_tokenizer.rs`; `Gguf::load_llama` builds a runnable `LlamaModel`
(RMSNorm/RoPE/GQA/SwiGLU). Bit-exact parity with llama.cpp is not asserted, but
every primitive is unit-covered and the imported model's KV-cached decode matches
its own full forward. Try it:
`train_transformer run-gguf model.gguf "..." --quant int4`.

---

## Reproducing

```bash
cargo build --release -p slm_cli
BIN=target/release/train_transformer

# Train a small model, then time generation at a fixed thread count.
$BIN train corpus.txt model.bin --vocab 0 --context 16 --embed 64 \
    --heads 4 --blocks 2 --hidden 128 --epochs 10 --seed 7 --force

# Determinism: identical output regardless of thread count.
for t in 1 2 4 8; do
  FERRUM_NUM_THREADS=$t $BIN generate model.bin "a long enough seed text here" \
      --chars 2000 --temp 0.7 --gen-seed 42 > out_$t.txt
done
md5sum out_*.txt   # all hashes identical
```

Matmul kernel microbenchmark (§3), std-only, no Criterion:

```bash
cargo bench --bench gemm                         # auto-detected threads
FERRUM_NUM_THREADS=1 cargo bench --bench gemm     # force serial
cargo bench --bench gemm -- 512 1024 4096         # custom square GEMM sizes
```

---

## Caveats

- Timings are single-machine, wall-clock; treat them as indicative, not precise.
  The §1/§2 end-to-end figures used a deliberately small model so they finish
  quickly on a loaded laptop — absolute scaling grows with model size.
- Background load skews results: an earlier §3a run contended with the test suite
  and showed 2048² at 13.7 GFLOP/s; the clean number is 35.7. Run benchmarks on an
  idle machine.
- §3 measures the raw `ops::matmul`/`qlinear` kernels on **synthetic ~1B-class
  shapes with random data**, not a trained model; its decode-step figure is an
  estimate (see §3c). These best-of-many-iterations numbers have lower variance
  than the §1/§2 wall-clock, but are still single-machine.
- No GPU is used anywhere. All parallelism is plain CPU threads on `std` only, with
  no `unsafe`.
