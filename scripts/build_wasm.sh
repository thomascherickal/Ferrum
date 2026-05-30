#!/usr/bin/env bash
# scripts/build_wasm.sh
# Compile tabular_wasm to WebAssembly and generate JS bindings.
# Usage: bash scripts/build_wasm.sh
set -euo pipefail
cd "$(dirname "$0")/.."

need() { command -v "$1" &>/dev/null || { echo "ERROR: $1 not found. Install with: cargo install $1 --version 0.2.122"; exit 1; }; }
need wasm-bindgen
command -v rustup &>/dev/null && rustup target add wasm32-unknown-unknown &>/dev/null

echo "=== Compiling tabular_wasm to WASM ==="
cargo build -p tabular_wasm --target wasm32-unknown-unknown --release

echo "=== Generating JS bindings ==="
wasm-bindgen \
  target/wasm32-unknown-unknown/release/tabular_wasm.wasm \
  --out-dir web/pkg \
  --target web \
  --no-typescript

echo ""
echo "WASM binary : $(du -sh web/pkg/tabular_wasm_bg.wasm | cut -f1)"
echo "JS glue     : $(du -sh web/pkg/tabular_wasm.js      | cut -f1)"
echo ""
echo "=== Done. Serve with: python3 -m http.server 8080 --directory web ==="
