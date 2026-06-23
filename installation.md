# Installation

Ferrum is a Cargo workspace of pure-Rust crates with **no external dependencies**
in the core engine. If you can run `cargo`, you can build Ferrum — there is no
Python, no CUDA toolkit, no BLAS, and no system ML library to install. The
practical upshot: a clean checkout resolves zero third-party crates for
`ferrum_core` and both CLIs, so builds are fast and reproducible and there is
nothing to vet for supply-chain risk.

## Requirements

| Requirement | Version                  | Notes                                                  |
|-------------|--------------------------|--------------------------------------------------------|
| Rust        | 1.74+ (stable)           | Edition 2021; install via [rustup](https://rustup.rs)  |
| Cargo       | ships with Rust          | Used for all builds, tests, and benchmarks             |
| OS          | Linux / macOS / Windows  | Pure CPU; no platform-specific code                    |
| WASM target | `wasm32-unknown-unknown` | Only for the browser playground (`tabular_wasm`)       |

The only crate that pulls a third-party dependency is `tabular_wasm`
(`wasm-bindgen`). The engine (`ferrum_core`) and both command-line tools build
with zero external crates. The desktop GUI (`ferrum_gui`) is a separate Tauri app
with its own, heavier prerequisites — see [ferrum_gui/README.md](ferrum_gui/README.md).

## Clone and build

```bash
git clone https://github.com/thomascherickal/Ferrum
cd Ferrum/ferrum
cargo build --workspace --release
```

The release profile enables LTO and a single codegen unit for the smallest,
fastest binaries. Use it for any real training or for running GGUF models —
debug builds are several times slower.

## Verify the install

```bash
cargo test --workspace
```

All unit and integration tests should pass. To try the SLM trainer immediately:

```bash
printf 'the quick brown fox jumps over the lazy dog. ' > /tmp/corpus.txt
cargo run -p slm_cli -- train /tmp/corpus.txt /tmp/model.bin --epochs 50 --vocab 0
cargo run -p slm_cli -- info /tmp/model.bin
```

You can also run the std-only matmul/decode microbenchmark (no Criterion, no
external deps):

```bash
cargo bench --bench gemm                       # auto-detected threads
FERRUM_NUM_THREADS=1 cargo bench --bench gemm   # force serial
```

## Using `ferrum_core` as a dependency

```toml
[dependencies]
ferrum_core = { git = "https://github.com/thomascherickal/Ferrum" }
```

`ferrum_core` is `#![forbid(unsafe_code)]` and `std`-only, so it adds **no**
transitive dependencies to your project — a rare property for an ML crate, and
the main reason it drops cleanly into audited, embedded, or air-gapped builds.

## Building the WebAssembly playground

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
cd tabular_wasm
wasm-pack build --release --target web
```

This emits a `pkg/` directory (the `.wasm` module + JS glue) that runs entirely
client-side. See [deployment.md](deployment.md) for hosting it. (The WASM build
runs serially — `wasm32` has no threads — but is otherwise the same engine.)

## Installing the CLIs

```bash
cargo install --path slm_cli    # installs the `train_transformer` binary
cargo install --path train_cli  # installs the `train_cli` binary
```

The `slm_cli` binary is named `train_transformer`; it also hosts the `run-gguf`
subcommand for importing external Llama/Qwen GGUF checkpoints.
