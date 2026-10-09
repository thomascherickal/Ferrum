"""FerrumModel against a stub server. Run:
eval/lighteval/.venv/bin/python -m unittest eval/lighteval/test_ferrum_model.py
Skips cleanly when lighteval is not installed."""
import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from lighteval.tasks.requests import Doc
    import ferrum_model
except ImportError:  # pragma: no cover
    ferrum_model = None

CALLS = []


class Stub(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        CALLS.append((self.path, None))
        self._send(200, {"context_len": 128, "vocab": 1024, "params": 486624})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        CALLS.append((self.path, body))
        if self.path == "/v1/loglikelihood":
            res = [{"logprob": -float(len(r["continuation"])), "n_tokens": 1, "greedy": r["continuation"] == " a"} for r in body["requests"]]
        elif self.path == "/v1/loglikelihood_rolling":
            res = [{"logprob": -2.0 * len(t), "n_tokens": len(t)} for t in body["texts"]]
        elif self.path == "/v1/generate":
            if body["requests"][0]["prompt"] == "boom":
                return self._send(422, {"error": "kaput"})
            res = [{"text": "out:" + r["prompt"], "n_tokens": 3} for r in body["requests"]]
        else:
            return self._send(404, {"error": "no"})
        self._send(200, {"results": res})


@unittest.skipIf(ferrum_model is None, "lighteval not installed")
class FerrumModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = HTTPServer(("127.0.0.1", 0), Stub)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        os.environ["FERRUM_URL"] = f"http://127.0.0.1:{cls.srv.server_port}"
        os.environ["FERRUM_BATCH"] = "2"
        from lighteval.models.custom.custom_model import CustomModelConfig
        cfg = CustomModelConfig(model_name="ferrum", model_definition_file_path=ferrum_model.__file__,
                                 cache_dir=tempfile.mkdtemp())  # fresh cache: lighteval keys cached samples by task, not text
        cls.m = ferrum_model.FerrumModel(cfg)

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def setUp(self):
        CALLS.clear()

    def test_max_length_from_info(self):
        self.assertEqual(self.m.max_length, 128)

    def test_loglikelihood_one_logprob_per_choice_batched(self):
        docs = [Doc(query="q", choices=[" a", " bb", " ccc"], gold_index=0, task_name="t")]
        out = self.m.loglikelihood(docs)
        self.assertEqual(out[0].logprobs, [-2.0, -3.0, -4.0])
        self.assertEqual(out[0].argmax_logits_eq_gold, [True, False, False])
        self.assertEqual([len(b["requests"]) for _, b in CALLS], [2, 1])  # FERRUM_BATCH=2

    def test_rolling(self):
        docs = [Doc(query="hello", choices=[], gold_index=0, task_name="t")]
        out = self.m.loglikelihood_rolling(docs)
        self.assertEqual(out[0].logprobs, [-10.0])

    def test_greedy_until_passes_stops_and_budget(self):
        docs = [Doc(query="hi", choices=[], gold_index=0, task_name="t", generation_size=7, stop_sequences=["\n"])]
        out = self.m.greedy_until(docs)
        self.assertEqual(out[0].text, ["out:hi"])
        req = CALLS[0][1]["requests"][0]
        self.assertEqual((req["max_tokens"], req["stop"], req["temperature"]), (7, ["\n"], 0.0))

    def test_http_error_raises(self):
        docs = [Doc(query="boom", choices=[], gold_index=0, task_name="t_err")]
        with self.assertRaises(RuntimeError) as ctx:
            self.m.greedy_until(docs)
        self.assertIn("422", str(ctx.exception))
        self.assertIn("kaput", str(ctx.exception))

    def test_generation_size_unset_zero_or_negative_means_64(self):
        for i, gs in enumerate([None, 0, -1, -100]):
            CALLS.clear()
            self.m.greedy_until([Doc(query=f"p{i}", choices=[], gold_index=0, task_name=f"t_gs{i}", generation_size=gs)])
            self.assertEqual(CALLS[0][1]["requests"][0]["max_tokens"], 64, gs)
        CALLS.clear()
        self.m.greedy_until([Doc(query="p", choices=[], gold_index=0, task_name="t_gs_pos", generation_size=1)])
        self.assertEqual(CALLS[0][1]["requests"][0]["max_tokens"], 1)

    def test_greedy_without_stop_sequences_sends_empty_list(self):
        self.m.greedy_until([Doc(query="nostop", choices=[], gold_index=0, task_name="t_nostop")])
        self.assertEqual(CALLS[0][1]["requests"][0]["stop"], [])

    def test_generate_output_tokens_match_n_tokens(self):
        out = self.m.greedy_until([Doc(query="ntok", choices=[], gold_index=0, task_name="t_ntok")])
        self.assertEqual(len(out[0].output_tokens[0]), 3)

    def test_unreachable_server_raises_runtime_error(self):
        old = self.m.url
        self.m.url = "http://127.0.0.1:9"
        try:
            with self.assertRaises(RuntimeError) as ctx:
                self.m.loglikelihood_rolling([Doc(query="x", choices=[], gold_index=0, task_name="t_down")])
        finally:
            self.m.url = old
        self.assertIn("cannot reach ferrum-serve", str(ctx.exception))

    def test_static_properties_and_batches(self):
        self.assertIsNone(self.m.tokenizer)
        self.assertFalse(self.m.add_special_tokens)
        self.assertEqual(list(ferrum_model._batches([1, 2, 3, 4, 5], 2)), [[1, 2], [3, 4], [5]])
        self.assertEqual(list(ferrum_model._batches([], 2)), [])

    def test_max_length_fetches_info_once(self):
        self.m._info = None
        self.assertEqual((self.m.max_length, self.m.max_length), (128, 128))
        self.assertEqual(CALLS, [("/v1/info", None)])


if __name__ == "__main__":
    unittest.main()
