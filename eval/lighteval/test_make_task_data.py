"""make_task_data.py: heldout/cloze data from a test split. Pure stdlib."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_task_data as mtd  # noqa: E402

SEP = "<|endoftext|>"


def story(i, n=5):
    return " ".join(f"Story {i} sentence {j} is here." for j in range(n))


def run(docs, sep=SEP):
    tmp = Path(tempfile.mkdtemp())
    src = tmp / "test.txt"
    src.write_text(sep.join(docs), encoding="utf-8")
    mtd.main(str(src), str(tmp / "out"))
    rd = lambda p: [json.loads(l) for l in (tmp / "out" / p / "test.jsonl").read_text(encoding="utf-8").splitlines()]  # noqa: E731
    return rd("heldout"), rd("cloze"), tmp


class SentencesTest(unittest.TestCase):
    def test_splits_on_terminal_punctuation_and_strips(self):
        self.assertEqual(mtd.sentences("Hi there.  How are you? Fine! ok"), ["Hi there.", "How are you?", "Fine!", "ok"])
        self.assertEqual(mtd.sentences("no break 3.5 here"), ["no break 3.5 here"])
        self.assertEqual(mtd.sentences("   "), [])


class MainTest(unittest.TestCase):
    def test_heldout_one_doc_per_separator_blank_skipped(self):
        held, _, _ = run([f" {story(0)}\n", "  ", story(1), ""])
        self.assertEqual([h["text"] for h in held], [story(0), story(1)])

    def test_cloze_gold_index_points_at_true_next_sentence(self):
        docs = [story(i) for i in range(6)]
        _, cloze, _ = run(docs)
        self.assertEqual(len(cloze), 6)
        for row, doc in zip(cloze, docs):
            ss = mtd.sentences(doc)
            self.assertEqual(len(row["choices"]), 4)
            self.assertEqual(len(set(row["choices"])), 4)
            gold = row["choices"][row["gold"]]
            k = ss.index(gold)
            self.assertGreaterEqual(k, 2)
            self.assertEqual(row["question"], " ".join(ss[:k]))
            for j, c in enumerate(row["choices"]):
                if j != row["gold"]:
                    self.assertNotEqual(c, gold)

    def test_docs_with_fewer_than_three_sentences_have_no_cloze(self):
        docs = [story(0), "One. Two.", story(1), "Single sentence only", story(2)]
        held, cloze, _ = run(docs)
        self.assertEqual(len(held), 5)
        self.assertEqual(len(cloze), 3)
        self.assertTrue(all("Story" in r["question"] for r in cloze))

    def test_deterministic_with_fixed_seed(self):
        docs = [story(i, 4 + i % 3) for i in range(10)]
        _, c1, t1 = run(docs)
        _, c2, t2 = run(docs)
        self.assertEqual(c1, c2)
        self.assertEqual((t1 / "out/cloze/test.jsonl").read_bytes(), (t2 / "out/cloze/test.jsonl").read_bytes())

    def test_too_few_distinct_sentences_skips_instead_of_hanging(self):
        # a lone 3-sentence doc has only 2 possible distractors
        code = (
            "import sys,tempfile,os;sys.path.insert(0,%r);import make_task_data as m;"
            "t=tempfile.mkdtemp();p=os.path.join(t,'x.txt');open(p,'w').write('A a. B b. C c.');"
            "m.main(p,os.path.join(t,'o'));print(open(os.path.join(t,'o','cloze','test.jsonl')).read()=='')" % HERE
        )
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=20)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("True", r.stdout)
        # same guard in-process (the subprocess above is the hang detector)
        held, cloze, _ = run(["A a. B b. C c."])
        self.assertEqual((len(held), cloze), (1, []))

    def test_cli_entry_point(self):
        tmp = Path(tempfile.mkdtemp())
        (tmp / "t.txt").write_text(story(0) + SEP + story(1), encoding="utf-8")
        r = subprocess.run([sys.executable, os.path.join(HERE, "make_task_data.py"), str(tmp / "t.txt"), str(tmp / "o")], capture_output=True, text=True, timeout=20)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("2 heldout docs", r.stdout)
        self.assertTrue((tmp / "o/cloze/test.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
