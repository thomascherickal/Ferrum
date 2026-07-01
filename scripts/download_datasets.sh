#!/usr/bin/env bash
# scripts/download_datasets.sh
# Download and clean all 10 source datasets into datasets/tabular/.
#
# Idempotent: every dataset is skipped if its cleaned file already exists, so
# this is safe to re-run and safe to invoke from CI when the datasets are already
# committed (it then makes no network calls). Delete a file to force a refresh.
#
# Usage: bash scripts/download_datasets.sh
set -euo pipefail
cd "$(dirname "$0")/.."

need() { command -v "$1" &>/dev/null || { echo "ERROR: $1 not found"; exit 1; }; }
need curl; need python3

DIR="datasets/tabular"
mkdir -p "$DIR"

echo "=== Preparing datasets in $DIR (existing files are kept) ==="

# ── 1. Iris ───────────────────────────────────────────────────────────────────
if [ -f "$DIR/iris.data" ]; then echo "  iris.data — present"; else
  curl -sL "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data" \
    | { echo "sepal_length,sepal_width,petal_length,petal_width,species"; cat; } > "$DIR/iris.data"
  echo "  iris.data ($(wc -l < "$DIR/iris.data") rows)"
fi

# ── 2. Wine Quality ─────────────────────────────────────────────────────────
if [ -f "$DIR/wine.csv" ]; then echo "  wine.csv — present"; else
  curl -sL "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv" \
    | sed 's/;/,/g; s/"//g' > "$DIR/wine.csv"
  echo "  wine.csv ($(wc -l < "$DIR/wine.csv") rows)"
fi

# ── 3. Pima Diabetes ────────────────────────────────────────────────────────
if [ -f "$DIR/diabetes.csv" ]; then echo "  diabetes.csv — present"; else
  curl -sL "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv" \
    | tr -d '\r' > "$DIR/diabetes.csv"
  echo "  diabetes.csv ($(wc -l < "$DIR/diabetes.csv") rows)"
fi

# ── 4. Titanic ──────────────────────────────────────────────────────────────
if [ -f "$DIR/titanic.csv" ]; then echo "  titanic.csv — present"; else
python3 - << 'PYEOF'
import csv
import urllib.request
rows = []
data = urllib.request.urlopen(
  "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
).read().decode()
reader = csv.DictReader(data.splitlines())
for r in reader:
    try:
        rows.append([
            r['Pclass'], '1' if r['Sex']=='female' else '0',
            r['Age'] if r['Age'] else '29.0',
            r['SibSp'], r['Parch'],
            r['Fare'] if r['Fare'] else '32.0',
            r['Survived']
        ])
    except: pass
with open('datasets/tabular/titanic.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['pclass','sex','age','sibsp','parch','fare','survived'])
    w.writerows(rows)
PYEOF
  echo "  titanic.csv ($(wc -l < "$DIR/titanic.csv") rows)"
fi

# ── 5. California Housing ───────────────────────────────────────────────────
if [ -f "$DIR/housing.csv" ]; then echo "  housing.csv — present"; else
python3 - << 'PYEOF'
import csv, urllib.request
data = urllib.request.urlopen(
  "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
).read().decode()
reader = csv.DictReader(data.splitlines())
with open('datasets/tabular/housing.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['longitude','latitude','housing_median_age','total_rooms',
                'total_bedrooms','population','households','median_income','median_house_value'])
    for r in reader:
        if r['total_bedrooms']:
            w.writerow([r['longitude'],r['latitude'],r['housing_median_age'],
                        r['total_rooms'],r['total_bedrooms'],r['population'],
                        r['households'],r['median_income'],r['median_house_value']])
PYEOF
  echo "  housing.csv ($(wc -l < "$DIR/housing.csv") rows)"
fi

# ── 6. Heart Disease ────────────────────────────────────────────────────────
if [ -f "$DIR/heart.csv" ]; then echo "  heart.csv — present"; else
python3 - << 'PYEOF'
import urllib.request, csv
data = urllib.request.urlopen(
  "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
).read().decode()
with open('datasets/tabular/heart.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['age','sex','cp','trestbps','chol','fbs','restecg',
                'thalach','exang','oldpeak','slope','ca','thal','target'])
    for line in data.splitlines():
        v = line.strip().split(',')
        if '?' not in v and len(v)==14:
            v[-1] = '1' if int(float(v[-1])) > 0 else '0'
            w.writerow(v)
PYEOF
  echo "  heart.csv ($(wc -l < "$DIR/heart.csv") rows)"
fi

# ── 7. Breast Cancer Wisconsin ──────────────────────────────────────────────
if [ -f "$DIR/cancer.csv" ]; then echo "  cancer.csv — present"; else
python3 - << 'PYEOF'
import urllib.request, csv
features = ['radius','texture','perimeter','area','smoothness',
            'compactness','concavity','concave_pts','symmetry','fractal_dim']
cols = [f+'_mean' for f in features]+[f+'_se' for f in features]+[f+'_worst' for f in features]
data = urllib.request.urlopen(
  "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
).read().decode()
with open('datasets/tabular/cancer.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(cols+['diagnosis'])
    for line in data.splitlines():
        v = line.strip().split(',')
        if len(v)==32: w.writerow(v[2:]+[v[1]])
PYEOF
  echo "  cancer.csv ($(wc -l < "$DIR/cancer.csv") rows)"
fi

# ── 8. Palmer Penguins ──────────────────────────────────────────────────────
if [ -f "$DIR/penguins.csv" ]; then echo "  penguins.csv — present"; else
python3 - << 'PYEOF'
import urllib.request, csv
data = urllib.request.urlopen(
  "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv"
).read().decode()
reader = csv.DictReader(data.splitlines())
with open('datasets/tabular/penguins.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['bill_length','bill_depth','flipper_length','body_mass','species'])
    for r in reader:
        try:
            w.writerow([float(r['bill_length_mm']),float(r['bill_depth_mm']),
                        float(r['flipper_length_mm']),float(r['body_mass_g']),r['species'].strip()])
        except: pass
PYEOF
  echo "  penguins.csv ($(wc -l < "$DIR/penguins.csv") rows)"
fi

# ── 9. Auto MPG ─────────────────────────────────────────────────────────────
if [ -f "$DIR/mpg.csv" ]; then echo "  mpg.csv — present"; else
python3 - << 'PYEOF'
import urllib.request, re
data = urllib.request.urlopen(
  "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
).read().decode()
with open('datasets/tabular/mpg.csv', 'w') as f:
    f.write('cylinders,displacement,horsepower,weight,acceleration,model_year,mpg\n')
    for line in data.splitlines():
        line = re.sub(r'\s+".*"$','',line.strip())
        parts = line.split()
        if len(parts)>=7 and '?' not in parts:
            f.write(','.join([parts[1],parts[2],parts[3],parts[4],parts[5],parts[6],parts[0]])+'\n')
PYEOF
  echo "  mpg.csv ($(wc -l < "$DIR/mpg.csv") rows)"
fi

# ── 10. Wheat Seeds ─────────────────────────────────────────────────────────
if [ -f "$DIR/seeds.csv" ]; then echo "  seeds.csv — present"; else
python3 - << 'PYEOF'
import urllib.request
names = {1:'Kama', 2:'Rosa', 3:'Canadian'}
data = urllib.request.urlopen(
  "https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt"
).read().decode()
with open('datasets/tabular/seeds.csv', 'w') as f:
    f.write('area,perimeter,compactness,kernel_length,kernel_width,asymmetry,groove_length,variety\n')
    for line in data.splitlines():
        v = line.strip().split()
        if len(v)==8:
            f.write(','.join(v[:7])+','+names.get(int(v[7]),'Unknown')+'\n')
PYEOF
  echo "  seeds.csv ($(wc -l < "$DIR/seeds.csv") rows)"
fi

echo ""
echo "=== All datasets ready in $DIR. Run 'bash scripts/train_all.sh' next. ==="
