"""Build LightEval task data from a Ferrum test split.

    python make_task_data.py <test.txt> <out_dir>

Writes <out_dir>/heldout/test.jsonl ({"text"}) and
<out_dir>/cloze/test.jsonl ({"question","choices","gold"}): story prefix →
true next sentence vs 3 sentences from other stories."""
import json
import random
import re
import sys
from pathlib import Path

SEP = "<|endoftext|>"


def sentences(doc):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", doc) if s.strip()]


def main(test_path, out_dir):
    docs = [d.strip() for d in Path(test_path).read_text(encoding="utf-8").split(SEP) if d.strip()]
    out = Path(out_dir)
    (out / "heldout").mkdir(parents=True, exist_ok=True)
    (out / "cloze").mkdir(parents=True, exist_ok=True)
    with open(out / "heldout" / "test.jsonl", "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps({"text": d}) + "\n")
    rng = random.Random(1234)
    pool = [s for d in docs for s in sentences(d)]
    with open(out / "cloze" / "test.jsonl", "w", encoding="utf-8") as f:
        for d in docs:
            ss = sentences(d)
            if len(ss) < 3:
                continue
            k = rng.randrange(2, len(ss))
            gold = ss[k]
            distractors = []
            while len(distractors) < 3:
                c = rng.choice(pool)
                if c != gold and c not in distractors:
                    distractors.append(c)
            choices = distractors + [gold]
            rng.shuffle(choices)
            f.write(json.dumps({"question": " ".join(ss[:k]), "choices": choices, "gold": choices.index(gold)}) + "\n")
    print(f"{len(docs)} heldout docs → {out}/heldout, cloze → {out}/cloze")


if __name__ == "__main__":
    main(*sys.argv[1:3])
