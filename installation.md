# Installation

Ferrum is a Cargo workspace of pure-Rust crates with **no external
dependencies** in the core engine. If you can run `cargo`, you can build Ferrum.

## Requirements

| Requirement | Version                | Notes                                            |
|-------------|------------------------|--------------------------------------------------|
| Rust        | 1.74+ (stable)         | Edition 2021; install via [rustup](https://rustup.rs) |
| Cargo       | ships with Rust        | Used for all builds and tests                    |
| OS          | Linux / macOS / Windows | Pure CPU; no platform-specific code             |
| WASM target | `wasm32-unknown-unknown` | Only needed for the browser playground (`tabular_wasm`) |

The only crate that pulls a third-party dependency is `tabular_wasm`, which uses
`wasm-bindgen` for browser bindings. The engine itself (`ferrum_core`) and both
command-line tools build with zero external crates.

## Clone and build

```bash
git clone https://github.com/thomascherickal/Ferrum
cd Ferrum/ferrum
cargo build --workspace --release
```

The release profile enables LTO and a single codegen unit for the smallest,
fastest binaries.

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

## Using `ferrum_core` as a dependency

To use the engine from your own crate, add a path or git dependency:

```toml
[dependencies]
ferrum_core = { git = "https://github.com/thomascherickal/Ferrum" }
```

`ferrum_core` is `#![forbid(unsafe_code)]` and `std`-only, so it imposes no
transitive dependencies on your project.

## Building the WebAssembly playground

Install the WASM target and `wasm-pack` (or use `cargo build --target`):

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
cd tabular_wasm
wasm-pack build --release --target web
```

See [deployment.md](deployment.md) for how to host the resulting bundle.

## Installing the CLIs

```bash
cargo install --path slm_cli    # installs the `train_transformer` binary
cargo install --path train_cli  # installs the `train_cli` binary
```
