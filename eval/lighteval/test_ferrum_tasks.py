"""ferrum_tasks.TASKS_TABLE: prompt functions build correct Docs. Skips without lighteval."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import ferrum_tasks as ft
except ImportError:  # pragma: no cover
    ft = None


@unittest.skipIf(ft is None, "lighteval not installed")
class TasksTest(unittest.TestCase):
    def test_heldout_doc_is_plain_text_for_rolling_perplexity(self):
        d = ft.heldout_prompt({"text": "Once upon a time."}, "ferrum_heldout")
        self.assertEqual((d.query, d.task_name), ("Once upon a time.", "ferrum_heldout"))
        self.assertEqual(d.original_query, "Once upon a time.")
        self.assertFalse(d.choices)
        self.assertIsNone(d.gold_index)

    def test_cloze_doc_prefixes_choices_with_space_and_keeps_gold(self):
        line = {"question": "Tom ran.", "choices": ["He fell.", "A dog.", "The sun.", "Yes."], "gold": 2}
        d = ft.cloze_prompt(line, "tinystories_cloze")
        self.assertEqual(d.query, "Tom ran.")
        self.assertEqual(d.choices, [" He fell.", " A dog.", " The sun.", " Yes."])
        self.assertEqual(d.gold_index, 2)
        self.assertEqual(d.choices[d.gold_index], " The sun.")

    def test_tasks_table_wires_prompt_functions_and_local_data(self):
        t = {c.name: c for c in ft.TASKS_TABLE}
        self.assertEqual(set(t), {"ferrum_heldout", "tinystories_cloze"})
        self.assertIs(t["ferrum_heldout"].prompt_function, ft.heldout_prompt)
        self.assertIs(t["tinystories_cloze"].prompt_function, ft.cloze_prompt)
        for name, sub in [("ferrum_heldout", "heldout"), ("tinystories_cloze", "cloze")]:
            c = t[name]
            self.assertEqual(c.hf_repo, os.path.join(ft.DATA, sub))
            self.assertEqual((list(c.evaluation_splits), c.generation_size), (["test"], -1))
        self.assertEqual(len(t["ferrum_heldout"].metrics), 3)
        self.assertEqual(len(t["tinystories_cloze"].metrics), 1)

    def test_data_dir_follows_environment(self):
        import importlib
        old = os.environ.get("FERRUM_TASK_DATA")
        os.environ["FERRUM_TASK_DATA"] = "/somewhere/else"
        try:
            self.assertEqual(importlib.reload(ft).DATA, "/somewhere/else")
        finally:
            if old is None:
                del os.environ["FERRUM_TASK_DATA"]
            else:
                os.environ["FERRUM_TASK_DATA"] = old
            importlib.reload(ft)


if __name__ == "__main__":
    unittest.main()
