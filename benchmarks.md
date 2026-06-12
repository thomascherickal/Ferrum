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
