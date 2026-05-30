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
mkdir -p web/pkg
wasm-bindgen \
  target/wasm32-unknown-unknown/release/tabular_wasm.wasm \
  --out-dir web/pkg \
  --target web \
  --no-typescript

echo ""
echo "WASM binary : $(du -sh web/pkg/tabular_wasm_bg.wasm | cut -f1)"
echo "JS glue     : $(du -sh web/pkg/tabular_wasm.js      | cut -f1)"
echo ""

# Distribute playgrounds directly to decoupled repositories
echo "=== Decoupled Playgrounds Auto-Distribution ==="
for repo in brand_alchemist ambient_poet shell_oracle; do
  target_dir="../$repo/web"
  if [ -d "../$repo" ]; then
    echo "Distributing to decoupled repository: $repo"
    mkdir -p "$target_dir/pkg" "$target_dir/shared"
    
    # Copy shared styles and engine
    cp -r web/shared/* "$target_dir/shared/"
    
    # Copy compiled WASM package
    cp -r web/pkg/* "$target_dir/pkg/"
    
    # Copy compiled model if it exists in the use-case repo root
    if [ -f "../$repo/$repo.bin" ]; then
      cp "../$repo/$repo.bin" "$target_dir/model.bin"
      echo "  -> Copied $repo.bin to $target_dir/model.bin"
    fi
  else
    echo "Decoupled repository not found at ../$repo (skipping distribution)"
  fi
done

echo ""
echo "=== Done. Decoupled playgrounds built and distributed successfully. ==="
