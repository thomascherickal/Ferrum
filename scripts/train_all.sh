#!/usr/bin/env bash
# scripts/train_all.sh
# Train all 10 dataset models and place them in web/datasets/*/model.bin.
# Requires: cargo build -p train_cli --release (done automatically).
# Usage: bash scripts/train_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Building train_cli ==="
cargo build --release -p train_cli

echo ""
echo "=== Training all 10 datasets ==="

run() {
  local csv="$1" dst="$2" name="$3" hidden="$4" epochs="$5"
  mkdir -p "$(dirname "$dst")"
  echo "--- $name ---"
  cargo run -p train_cli --release -- "$csv" "$dst" "$name" "$hidden" "$epochs" \
    2>&1 | grep -E "accuracy|RMSE|Saved|error" | head -4
}

run iris.data    web/datasets/iris/model.bin      "Iris"               32 500
run wine.csv     web/datasets/wine/model.bin      "Wine Quality"       64 600
run diabetes.csv web/datasets/diabetes/model.bin  "Pima Diabetes"      48 600
run titanic.csv  web/datasets/titanic/model.bin   "Titanic"            32 500
run housing.csv  web/datasets/housing/model.bin   "California Housing" 64 400
run heart.csv    web/datasets/heart/model.bin     "Heart Disease"      32 600
run cancer.csv   web/datasets/cancer/model.bin    "Breast Cancer"      64 600
run penguins.csv web/datasets/penguins/model.bin  "Palmer Penguins"    32 500
run mpg.csv      web/datasets/mpg/model.bin       "Auto MPG"           48 500
run seeds.csv    web/datasets/seeds/model.bin     "Wheat Seeds"        24 500

echo ""
echo "=== All models trained. Run 'bash scripts/build_wasm.sh' next. ==="
