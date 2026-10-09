"""LightEval 0.13.0 custom model that evaluates a Ferrum SLM served by
`ferrum-serve` (slm_server). Usage:

    FERRUM_URL=http://127.0.0.1:8080 lighteval custom ferrum \
        eval/lighteval/ferrum_model.py "ferrum_heldout|0" \
        --custom-tasks eval/lighteval/ferrum_tasks.py

LightEval 0.13.0 quirks handled here:
- LightevalModel has no __init__: calling super().__init__() raises TypeError.
- The pipeline calls model._cache._init_registry(...) unconditionally, so
  self._cache must be a SampleCache.
- @cached(SamplingMethod.PERPLEXITY) drops every result, so
  loglikelihood_rolling is deliberately not cached.
"""
import json
import os
import urllib.error
import urllib.request

from lighteval.models.abstract_model import LightevalModel
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.utils.cache_management import SampleCache, cached

URL = os.environ.get("FERRUM_URL", "http://127.0.0.1:8080").rstrip("/")


def _batches(items, n):
    for i in range(0, len(items), n):
        yield items[i : i + n]


class FerrumModel(LightevalModel):
    def __init__(self, config):
        self.config = config
        self._cache = SampleCache(config)
        self.url = os.environ.get("FERRUM_URL", URL).rstrip("/")
        self.batch = int(os.environ.get("FERRUM_BATCH", "32"))
        self._info = None

    # ── HTTP ────────────────────────────────────────────────────────────
    def _call(self, path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(
            self.url + path, data=data, headers={"content-type": "application/json"}, method="GET" if body is None else "POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            raise RuntimeError(f"ferrum-serve {self.url}{path} returned {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"cannot reach ferrum-serve at {self.url}{path}: {e.reason}") from e

    def _results(self, path, key, items):
        out = []
        for b in _batches(items, self.batch):
            out.extend(self._call(path, {key: b})["results"])
        return out

    # ── LightevalModel interface ────────────────────────────────────────
    @property
    def tokenizer(self):
        return None

    @property
    def add_special_tokens(self):
        return False

    @property
    def max_length(self):
        if self._info is None:
            self._info = self._call("/v1/info")
        return int(self._info["context_len"])

    @cached(SamplingMethod.GENERATIVE)
    def greedy_until(self, docs: list[Doc]) -> list[ModelResponse]:
        reqs = [
            {
                "prompt": d.query,
                "max_tokens": int(d.generation_size or 64),
                "stop": list(d.stop_sequences or []),
                "temperature": 0.0,
            }
            for d in docs
        ]
        res = self._results("/v1/generate", "requests", reqs)
        return [ModelResponse(text=[r["text"]], output_tokens=[[0] * r["n_tokens"]]) for r in res]

    @cached(SamplingMethod.LOGPROBS)
    def loglikelihood(self, docs: list[Doc]) -> list[ModelResponse]:
        flat = [{"context": d.query, "continuation": c} for d in docs for c in d.choices]
        res = iter(self._results("/v1/loglikelihood", "requests", flat))
        out = []
        for d in docs:
            rs = [next(res) for _ in d.choices]
            out.append(ModelResponse(logprobs=[r["logprob"] for r in rs], argmax_logits_eq_gold=[r["greedy"] for r in rs]))
        return out

    # Not cached: lighteval 0.13.0's PERPLEXITY cache drops all results.
    def loglikelihood_rolling(self, docs: list[Doc]) -> list[ModelResponse]:
        res = self._results("/v1/loglikelihood_rolling", "texts", [d.query for d in docs])
        return [ModelResponse(logprobs=[r["logprob"]]) for r in res]
