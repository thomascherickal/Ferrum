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

run datasets/tabular/iris.data    web/datasets/iris/model.bin      "Iris"               32 500
run datasets/tabular/wine.csv     web/datasets/wine/model.bin      "Wine Quality"       64 600
run datasets/tabular/diabetes.csv web/datasets/diabetes/model.bin  "Pima Diabetes"      48 600
run datasets/tabular/titanic.csv  web/datasets/titanic/model.bin   "Titanic"            32 500
run datasets/tabular/housing.csv  web/datasets/housing/model.bin   "California Housing" 64 400
run datasets/tabular/heart.csv    web/datasets/heart/model.bin     "Heart Disease"      32 600
run datasets/tabular/cancer.csv   web/datasets/cancer/model.bin    "Breast Cancer"      64 600
run datasets/tabular/penguins.csv web/datasets/penguins/model.bin  "Palmer Penguins"    32 500
run datasets/tabular/mpg.csv      web/datasets/mpg/model.bin       "Auto MPG"           48 500
run datasets/tabular/seeds.csv    web/datasets/seeds/model.bin     "Wheat Seeds"        24 500

echo ""
echo "=== All models trained. Run 'bash scripts/build_wasm.sh' next. ==="
