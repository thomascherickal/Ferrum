#!/usr/bin/env bash
# scripts/download_datasets.sh
# Download and clean all 10 source datasets.
# Run this once from the project root before training.
# Usage: bash scripts/download_datasets.sh
set -euo pipefail
cd "$(dirname "$0")/.."

need() { command -v "$1" &>/dev/null || { echo "ERROR: $1 not found"; exit 1; }; }
need curl; need python3

echo "=== Downloading datasets ==="

# ── 1. Iris (already ships with the repo) ─────────────────────────────────────
[ -f iris.data ] && echo "  iris.data — already present" || \
  curl -sL "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data" \
    | { echo "sepal_length,sepal_width,petal_length,petal_width,species"; cat; } > iris.data
echo "  iris.data ($(wc -l < iris.data) rows)"

# ── 2. Wine Quality ─────────────────────────────────────────────────────────
curl -sL "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv" \
  | sed 's/;/,/g; s/"//g' > wine.csv
echo "  wine.csv ($(wc -l < wine.csv) rows)"

# ── 3. Pima Diabetes ────────────────────────────────────────────────────────
curl -sL "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv" \
  | tr -d '\r' > diabetes.csv
echo "  diabetes.csv ($(wc -l < diabetes.csv) rows)"

# ── 4. Titanic ──────────────────────────────────────────────────────────────
python3 - << 'PYEOF'
import csv, re
rows = []
import urllib.request
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
with open('titanic.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['pclass','sex','age','sibsp','parch','fare','survived'])
    w.writerows(rows)
PYEOF
echo "  titanic.csv ($(wc -l < titanic.csv) rows)"

# ── 5. California Housing ───────────────────────────────────────────────────
python3 - << 'PYEOF'
import csv, urllib.request
data = urllib.request.urlopen(
  "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
).read().decode()
reader = csv.DictReader(data.splitlines())
with open('housing.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['longitude','latitude','housing_median_age','total_rooms',
                'total_bedrooms','population','households','median_income','median_house_value'])
    for r in reader:
        if r['total_bedrooms']:
            w.writerow([r['longitude'],r['latitude'],r['housing_median_age'],
                        r['total_rooms'],r['total_bedrooms'],r['population'],
                        r['households'],r['median_income'],r['median_house_value']])
PYEOF
echo "  housing.csv ($(wc -l < housing.csv) rows)"

# ── 6. Heart Disease ────────────────────────────────────────────────────────
python3 - << 'PYEOF'
import urllib.request, csv
data = urllib.request.urlopen(
  "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
).read().decode()
with open('heart.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['age','sex','cp','trestbps','chol','fbs','restecg',
                'thalach','exang','oldpeak','slope','ca','thal','target'])
    for line in data.splitlines():
        v = line.strip().split(',')
        if '?' not in v and len(v)==14:
            v[-1] = '1' if int(float(v[-1])) > 0 else '0'
            w.writerow(v)
PYEOF
echo "  heart.csv ($(wc -l < heart.csv) rows)"

# ── 7. Breast Cancer Wisconsin ──────────────────────────────────────────────
python3 - << 'PYEOF'
import urllib.request, csv
features = ['radius','texture','perimeter','area','smoothness',
            'compactness','concavity','concave_pts','symmetry','fractal_dim']
cols = [f+'_mean' for f in features]+[f+'_se' for f in features]+[f+'_worst' for f in features]
data = urllib.request.urlopen(
  "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
).read().decode()
with open('cancer.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(cols+['diagnosis'])
    for line in data.splitlines():
        v = line.strip().split(',')
        if len(v)==32: w.writerow(v[2:]+[v[1]])
PYEOF
echo "  cancer.csv ($(wc -l < cancer.csv) rows)"

# ── 8. Palmer Penguins ──────────────────────────────────────────────────────
python3 - << 'PYEOF'
import urllib.request, csv
data = urllib.request.urlopen(
  "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv"
).read().decode()
reader = csv.DictReader(data.splitlines())
with open('penguins.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['bill_length','bill_depth','flipper_length','body_mass','species'])
    for r in reader:
        try:
            w.writerow([float(r['bill_length_mm']),float(r['bill_depth_mm']),
                        float(r['flipper_length_mm']),float(r['body_mass_g']),r['species'].strip()])
        except: pass
PYEOF
echo "  penguins.csv ($(wc -l < penguins.csv) rows)"

# ── 9. Auto MPG ─────────────────────────────────────────────────────────────
python3 - << 'PYEOF'
import urllib.request, re
data = urllib.request.urlopen(
  "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
).read().decode()
with open('mpg.csv', 'w') as f:
    f.write('cylinders,displacement,horsepower,weight,acceleration,model_year,mpg\n')
    for line in data.splitlines():
        line = re.sub(r'\s+".*"$','',line.strip())
        parts = line.split()
        if len(parts)>=7 and '?' not in parts:
            f.write(','.join([parts[1],parts[2],parts[3],parts[4],parts[5],parts[6],parts[0]])+'\n')
PYEOF
echo "  mpg.csv ($(wc -l < mpg.csv) rows)"

# ── 10. Wheat Seeds ─────────────────────────────────────────────────────────
python3 - << 'PYEOF'
import urllib.request
names = {1:'Kama', 2:'Rosa', 3:'Canadian'}
data = urllib.request.urlopen(
  "https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt"
).read().decode()
with open('seeds.csv', 'w') as f:
    f.write('area,perimeter,compactness,kernel_length,kernel_width,asymmetry,groove_length,variety\n')
    for line in data.splitlines():
        v = line.strip().split()
        if len(v)==8:
            f.write(','.join(v[:7])+','+names.get(int(v[7]),'Unknown')+'\n')
PYEOF
echo "  seeds.csv ($(wc -l < seeds.csv) rows)"

echo ""
echo "=== All datasets ready. Run 'bash scripts/train_all.sh' next. ==="
