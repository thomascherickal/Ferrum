"""run.sh with fakes for cargo, ferrum-serve and lighteval (no network, no model)."""
import json
import os
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE / "run.sh"


def free_port():
    with socket.socket() as so:
        so.bind(("127.0.0.1", 0))
        return so.getsockname()[1]


def exe(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


class Fixture:
    """A fake repo: <root>/eval/lighteval/{run.sh,*.py}, fake server, venv and cargo."""

    def __init__(self, server_body):
        self.root = Path(tempfile.mkdtemp())
        lt = self.root / "eval/lighteval"
        lt.mkdir(parents=True)
        for f in ("run.sh", "make_task_data.py"):
            shutil.copy(HERE / f, lt / f)
        (lt / "ferrum_model.py").write_text("")
        (lt / "ferrum_tasks.py").write_text("")
        exe(self.root / "target/release/ferrum-serve", server_body)
        venv = self.root / "venv"
        exe(venv / "bin/python", f'#!/bin/bash\nexec {sys.executable} "$@"\n')
        self.argv = self.root / "lighteval.argv"
        exe(
            venv / "bin/lighteval",
            "#!/bin/bash\nprintf '%s\\n' \"$@\" > " + str(self.argv) + "\n"
            'while [ $# -gt 0 ]; do [ "$1" = --output-dir ] && d=$2; shift; done\n'
            'mkdir -p "$d/x"; echo \'{"results": {"all": {"z": 0}, "t|0": {"acc": 0.5}}}\' > "$d/x/results_1.json"\n',
        )
        exe(self.root / "bin/cargo", "#!/bin/bash\nexit 0\n")
        self.test = self.root / "test.txt"
        self.test.write_text("A a. B b. C c. D d. E e.<|endoftext|>F f. G g. H h. I i. J j.")
        self.out = self.root / "res/out.json"

    def run(self, *extra, timeout=60, env_extra=None):
        env = {k: v for k, v in os.environ.items() if k != "CARGO_TARGET_DIR"}
        env.update(env_extra or {})
        env.update( FERRUM_LIGHTEVAL_VENV=str(self.root / "venv"), PATH=f"{self.root/'bin'}:{os.environ['PATH']}")
        return subprocess.run(
            ["bash", str(self.root / "eval/lighteval/run.sh"), "--model", "m.bin", "--test", str(self.test), "--out", str(self.out), "--port", str(free_port()), *extra],
            capture_output=True, text=True, env=env, timeout=timeout,
        )


# fake server: answers /health 200 on --port
GOOD_SERVER = (
    "#!/bin/bash\nport=8765\nwhile [ $# -gt 0 ]; do [ \"$1\" = --port ] && port=$2; shift; done\n"
    f"exec {sys.executable} -c \"import http.server as h,sys\n"
    "class H(h.BaseHTTPRequestHandler):\n"
    "  def do_GET(s): s.send_response(200); s.send_header('content-length','2'); s.end_headers(); s.wfile.write(b'ok')\n"
    "  def log_message(s,*a): pass\n"
    "h.HTTPServer(('127.0.0.1',int(sys.argv[1])),H).serve_forever()\" $port\n"
)
DEAD_SERVER = "#!/bin/bash\necho 'fatal: cannot load model' >&2\nexit 3\n"


class RunShTest(unittest.TestCase):
    def test_syntax(self):
        self.assertEqual(subprocess.run(["bash", "-n", str(RUN)]).returncode, 0)

    def test_dead_server_fails_fast_with_server_log(self):
        fx = Fixture(DEAD_SERVER)
        t = time.time()
        r = fx.run()
        self.assertNotEqual(r.returncode, 0)
        self.assertLess(time.time() - t, 20, "must not wait out the health cap")
        self.assertIn("cannot load model", r.stderr)
        self.assertFalse(fx.out.exists())

    def test_happy_path_writes_summary_and_passes_max_samples(self):
        fx = Fixture(GOOD_SERVER)
        r = fx.run("--max-samples", "7")
        self.assertEqual(r.returncode, 0, r.stderr)
        summary = json.loads(fx.out.read_text())
        self.assertEqual(summary["tasks"], {"t|0": {"acc": 0.5}})
        self.assertEqual((summary["model"], summary["max_samples"], summary["lighteval_version"]), ("m.bin", 7, "0.13.0"))
        argv = fx.argv.read_text().split("\n")
        self.assertEqual(argv[argv.index("--max-samples") + 1], "7")
        self.assertTrue((fx.root / "eval/lighteval/out/data/cloze/test.jsonl").exists())

    def test_no_max_samples_and_wikitext_flag(self):
        fx = Fixture(GOOD_SERVER)
        r = fx.run("--wikitext")
        self.assertEqual(r.returncode, 0, r.stderr)
        argv = fx.argv.read_text()
        self.assertNotIn("--max-samples", argv)
        self.assertIn("wikitext:103:document_level|0", argv)
        self.assertIsNone(json.loads(fx.out.read_text())["max_samples"])

    def test_honours_cargo_target_dir(self):
        fx = Fixture(DEAD_SERVER)  # the default location holds a server that dies
        alt = fx.root / "alt-target"
        exe(alt / "release/ferrum-serve", GOOD_SERVER)
        r = fx.run(env_extra={"CARGO_TARGET_DIR": str(alt)})
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(fx.out.exists())

    def test_bad_arguments(self):
        fx = Fixture(GOOD_SERVER)
        r = fx.run("--bogus")
        self.assertEqual(r.returncode, 2)
        self.assertIn("unknown argument", r.stderr)
        r = subprocess.run(["bash", str(fx.root / "eval/lighteval/run.sh"), "--model", "m"], capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("--test required", r.stderr)

    def test_missing_venv_hint(self):
        fx = Fixture(GOOD_SERVER)
        env = dict(os.environ, FERRUM_LIGHTEVAL_VENV=str(fx.root / "nope"))
        r = subprocess.run(["bash", str(fx.root / "eval/lighteval/run.sh"), "--model", "m", "--test", "t", "--out", "o"], capture_output=True, text=True, env=env)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("python3 -m venv", r.stderr)


if __name__ == "__main__":
    unittest.main()
