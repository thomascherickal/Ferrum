#!/usr/bin/env bash
# Evaluate a Ferrum SLM with LightEval 0.13.0 through ferrum-serve.
# Usage: eval/lighteval/run.sh --model M --test T --out O [--max-samples N] [--wikitext] [--port P]
# Env: FERRUM_LIGHTEVAL_VENV  venv holding python + lighteval (default: eval/lighteval/.venv)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
model= test= out= max= port=8765 wikitext=0
while [ $# -gt 0 ]; do
  case $1 in
    --model) model=$2; shift 2;;
    --test) test=$2; shift 2;;
    --out) out=$2; shift 2;;
    --max-samples) max=$2; shift 2;;
    --port) port=$2; shift 2;;
    --wikitext) wikitext=1; shift;;
    *) echo "unknown argument $1" >&2; exit 2;;
  esac
done
: "${model:?--model required}" "${test:?--test required}" "${out:?--out required}"
venv="${FERRUM_LIGHTEVAL_VENV:-$here/.venv}"
py="$venv/bin/python"; le="$venv/bin/lighteval"
[ -x "$le" ] || { echo "run: python3 -m venv $venv && $venv/bin/pip install -r $here/requirements.txt" >&2; exit 1; }

work="$here/out"; rm -rf "$work"; mkdir -p "$work"
"$py" "$here/make_task_data.py" "$test" "$work/data"

(cd "$root" && cargo build --release -q -p slm_server)
"$root/target/release/ferrum-serve" --model "$model" --port "$port" 2>"$work/server.log" &
srv=$!
trap 'kill $srv 2>/dev/null || true' EXIT
for _ in $(seq 1 100); do curl -sf "http://127.0.0.1:$port/health" >/dev/null && break; sleep 0.1; done
curl -sf "http://127.0.0.1:$port/health" >/dev/null || { cat "$work/server.log" >&2; exit 1; }

tasks="ferrum_heldout|0,tinystories_cloze|0,blimp:anaphor_gender_agreement|0,blimp:anaphor_number_agreement|0,blimp:regular_plural_subject_verb_agreement_1|0,arc:easy|0"
[ "$wikitext" = 1 ] && tasks="$tasks,wikitext:103:document_level|0"
extra=(); [ -n "$max" ] && extra=(--max-samples "$max")

FERRUM_URL="http://127.0.0.1:$port" FERRUM_TASK_DATA="$work/data" \
  "$le" custom ferrum "$here/ferrum_model.py" "$tasks" \
  --custom-tasks "$here/ferrum_tasks.py" --output-dir "$work/results" "${extra[@]}"

"$py" - "$work/results" "$out" "$model" "${max:-}" <<'EOF'
import glob, json, os, sys
res_dir, out, model, maxs = sys.argv[1:5]
f = max(glob.glob(os.path.join(res_dir, "**", "results_*.json"), recursive=True), key=os.path.getmtime)
r = json.load(open(f))["results"]
tasks = {k: v for k, v in r.items() if k != "all"}
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
json.dump({"lighteval_version": "0.13.0", "model": model, "max_samples": int(maxs) if maxs else None, "tasks": tasks},
          open(out, "w"), indent=2, sort_keys=True)
print(json.dumps(tasks, indent=2, sort_keys=True))
EOF
