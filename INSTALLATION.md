# 🧬 Ferrum Workspace Installation Guide

This document describes how to install, set up, and verify the core **Ferrum** workspace.

---

## 1. System Pre-requisites

Ferrum requires a modern Rust toolchain. Ensure you have Rust installed (v1.70.0 or later recommended).

### Install Rust (if not present)
On Linux/macOS:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

---

## 2. WebAssembly Tooling (For Browser Playgrounds)

To compile the WASM bindings (`tabular_wasm`) and generate the Javascript interface, install the standard Rust target and the matching `wasm-bindgen-cli` tool:

```bash
# Add the WebAssembly target
rustup target add wasm32-unknown-unknown

# Install the exact wasm-bindgen compiler (v0.2.122)
cargo install wasm-bindgen-cli --version 0.2.122 --locked
```

---

## 3. Clone & Build the Workspace

1. **Clone the repository**:
   ```bash
   git clone https://github.com/thomascherickal/Ferrum.git
   cd Ferrum
   ```

2. **Verify workspace components compile**:
   ```bash
   cargo check --workspace
   ```

---

## 4. Run the Automated Tests

Ferrum features a comprehensive test suite of 196 unit and integration tests verifying all operations:

```bash
cargo test --workspace
```

---

## 5. Compile WebAssembly Bindings

Execute the built-in shell script to compile the Rust bindings to a small WASM binary and bundle the JS interface into the `web/pkg` folder:

```bash
bash scripts/build_wasm.sh
```
