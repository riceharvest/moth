"""moth/config.py and `python -m moth config`, plus the browser opt-in of `python -m moth` (serve faked out).

  python3 tests/test_config.py      (from ~/moth; about a second, temp dirs only, no model, no server)

Covers: show_in_harness off by default; on/off written as a JSON bool, atomically (no temp file left), keeping other
keys; bad keys / values refused with the file unchanged; a corrupt file reads as the default and is replaced on the
next write; MOTH_SHOW_IN_HARNESS beating the file (and the CLI saying so); and that `python -m moth` opens the browser
only with --open / --browser, accepts --no-browser as a no-op, and never passes those flags on to serve.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
import unittest.mock
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from moth import config as C  # noqa: E402


class ConfigCli(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name) / "dotmoth"   # doesn't exist yet: the first write makes it
        self.file = self.dir / "config.json"
        self.env = {**os.environ, "MOTH_CONFIG_FILE": str(self.file)}
        self.env.pop("MOTH_SHOW_IN_HARNESS", None)
        self.patch = unittest.mock.patch.dict(os.environ, self.env, clear=True)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def cli(self, *args, env=None):
        """The real entry point, as a user types it (a subprocess: proves nothing torch-y is imported)."""
        p = subprocess.run([sys.executable, "-m", "moth", "config", *args], cwd=ROOT, capture_output=True, text=True,
                           timeout=20, env={**self.env, **(env or {})})
        return p.returncode, p.stdout, p.stderr

    def test_default_is_off(self):
        self.assertFalse(C.show_in_harness())
        self.assertEqual(C.lookup("show_in_harness"), (False, "default"))
        code, out, err = self.cli()
        self.assertEqual((code, err), (0, ""))
        self.assertIn(str(self.file), out)
        self.assertIn("show_in_harness = off   (default)", out)
        self.assertFalse(self.dir.exists())   # reading never creates anything

    def test_on_off_writes_bool_and_keeps_other_keys(self):
        self.dir.mkdir()
        self.file.write_text(json.dumps({"something_else": [1, 2]}))
        code, out, err = self.cli("show_in_harness", "on")
        self.assertEqual((code, err), (0, ""))
        self.assertIn("show_in_harness = on", out)
        self.assertEqual(json.loads(self.file.read_text()), {"something_else": [1, 2], "show_in_harness": True})
        self.assertTrue(C.show_in_harness())
        self.assertIn("show_in_harness = on   (from the file)", self.cli()[1])
        for word, want in (("OFF", False), ("1", True), ("false", False), ("yes", True), ("0", False)):
            with self.subTest(word):
                self.assertEqual(self.cli("show_in_harness", word)[0], 0)
                self.assertIs(json.loads(self.file.read_text())["show_in_harness"], want)
        self.assertEqual(sorted(os.listdir(self.dir)), ["config.json"])   # no temp files left behind

    def test_first_write_makes_the_folder(self):
        self.assertEqual(self.cli("show_in_harness", "on")[0], 0)
        self.assertEqual(json.loads(self.file.read_text()), {"show_in_harness": True})

    def test_bad_input_leaves_file_alone(self):
        self.dir.mkdir()
        self.file.write_text('{"show_in_harness": true}')
        before = self.file.read_bytes()
        for args in (("show_in_harness", "maybe"), ("show_in_harness",), ("nope", "on"),
                     ("show_in_harness", "on", "extra")):
            with self.subTest(args):
                code, out, err = self.cli(*args)
                self.assertEqual(code, 2)
                self.assertTrue(err)
                self.assertEqual(self.file.read_bytes(), before)
        code, out, _ = self.cli("--help")
        self.assertEqual(code, 0)
        self.assertIn("show_in_harness", out)

    def test_corrupt_file_reads_as_default_and_is_replaced(self):
        self.dir.mkdir()
        for junk in ("{not json", "[1, 2]", '"on"', ""):
            with self.subTest(junk):
                self.file.write_text(junk)
                self.assertEqual(C.lookup("show_in_harness"), (False, "default"))
                self.assertEqual(self.cli()[0], 0)
        self.file.write_text('{"show_in_harness": "sometimes"}')   # an unknown value: the default too
        self.assertEqual(C.lookup("show_in_harness"), (False, "default"))
        self.file.write_text("{not json")
        self.assertEqual(self.cli("show_in_harness", "on")[0], 0)
        self.assertEqual(json.loads(self.file.read_text()), {"show_in_harness": True})

    def test_env_beats_file(self):
        self.assertEqual(self.cli("show_in_harness", "on")[0], 0)
        with unittest.mock.patch.dict(os.environ, {"MOTH_SHOW_IN_HARNESS": "0"}):
            self.assertEqual(C.lookup("show_in_harness"), (False, "env"))
        with unittest.mock.patch.dict(os.environ, {"MOTH_SHOW_IN_HARNESS": "garbage"}):   # ignored: the file wins
            self.assertEqual(C.lookup("show_in_harness"), (True, "file"))
        code, out, _ = self.cli(env={"MOTH_SHOW_IN_HARNESS": "off"})
        self.assertIn("show_in_harness = off   (from MOTH_SHOW_IN_HARNESS)", out)
        code, out, _ = self.cli("show_in_harness", "on", env={"MOTH_SHOW_IN_HARNESS": "0"})
        self.assertEqual(code, 0)
        self.assertIn("MOTH_SHOW_IN_HARNESS=0 is set in this shell and wins", out)
        self.assertTrue(json.loads(self.file.read_text())["show_in_harness"])

    def test_failed_write_keeps_old_file_and_no_temp(self):
        self.dir.mkdir()
        self.file.write_text('{"show_in_harness": false}')
        with unittest.mock.patch("moth.config.os.replace", side_effect=OSError("disk full")), \
                contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(C.main(["show_in_harness", "on"]), 1)
        self.assertIn("disk full", err.getvalue())
        self.assertEqual(json.loads(self.file.read_text()), {"show_in_harness": False})
        self.assertEqual(os.listdir(self.dir), ["config.json"])


class MainBrowser(unittest.TestCase):
    """`python -m moth`: the page is opened only on request; serve gets argv without the browser flags."""

    def start(self, argv):
        from moth import __main__ as M
        seen = {}
        fake = types.ModuleType("moth.serve")

        def serve_main(argv, on_ready=None):
            seen["argv"] = list(argv)
            on_ready("http://127.0.0.1:4747")
        fake.main = serve_main
        out = io.StringIO()
        with contextlib.redirect_stdout(out), unittest.mock.patch.dict(sys.modules, {"moth.serve": fake}), \
                unittest.mock.patch("moth.serve", fake, create=True), \
                unittest.mock.patch("threading.Timer") as timer:
            M.main(argv)
        opened = [c.args[2] for c in timer.call_args_list]   # Timer(0.3, webbrowser.open, [url])
        return seen["argv"], opened, out.getvalue()

    def test_default_does_not_open(self):
        argv, opened, out = self.start(["--port", "4799"])
        self.assertEqual((argv, opened), (["--port", "4799"], []))
        self.assertIn("chat page: http://127.0.0.1:4747  (not opened; python -m moth --open opens it)", out)
        self.assertIn("show_in_harness on", out)   # the snippets say how to see moth inside Pi / OpenCode

    def test_open_and_browser_open(self):
        for flag in ("--open", "--browser"):
            with self.subTest(flag):
                argv, opened, out = self.start([flag, "--react-every", "0"])
                self.assertEqual((argv, opened), (["--react-every", "0"], [["http://127.0.0.1:4747"]]))
                self.assertNotIn("not opened", out)

    def test_no_browser_is_a_harmless_no_op(self):
        argv, opened, _ = self.start(["--no-browser"])
        self.assertEqual((argv, opened), ([], []))

    def test_config_never_starts_serve(self):
        from moth import __main__ as M
        with tempfile.TemporaryDirectory() as d, \
                unittest.mock.patch.dict(os.environ, {"MOTH_CONFIG_FILE": os.path.join(d, "c.json")}), \
                unittest.mock.patch.dict(sys.modules, {"moth.serve": None}), \
                contextlib.redirect_stdout(io.StringIO()) as out, self.assertRaises(SystemExit) as ex:
            M.main(["config", "--help"])   # "--help" here is config's, not serve's
        self.assertEqual(ex.exception.code, 0)
        self.assertIn("show_in_harness on|off", out.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
