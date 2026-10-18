"""moth setup, in temporary HOME / XDG folders with fake harness installs (as tests/test_cli_tools.py).

  python3 -m pytest -q tests/test_setup.py      (from ~/moth; a few seconds: no torch probe, no Qt, no network)

Every probe that would reach outside is replaced: torch / PySide6 (child processes in real life), the weights
download, free RAM, the display, /api/health and the pet launcher (no pet or server is ever started here). The weights
are a tiny but well-formed safetensors file in a temp MOTH_CACHE (pip mode) or a temp Hugging Face layout.
Covers: the dry run (and that it writes nothing), --yes with everything mocked, a second run changing nothing and
starting nothing, no TTY = dry run unless --yes, interactive answers (EOF = no, a pip install only on an explicit yes),
a partial Hugging Face download, the pet dying, a taken port and low memory.
"""
from __future__ import annotations

import json
import os
import struct
import sys
import unittest
import unittest.mock
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import test_cli_tools as T  # noqa: E402  (TempHome, fake_bin, FAKE_HERMES, the real-files guard)
from moth import setup as S  # noqa: E402

BEFORE = T._fingerprint()


def tearDownModule():
    after = T._fingerprint()
    changed = {k for k in set(BEFORE) | set(after) if BEFORE.get(k) != after.get(k)
               and not k.endswith(".log") and "serve-" not in str(k)}
    if changed:
        raise AssertionError(f"tests touched real files: {sorted(changed)}")


def make_weights(folder: Path):
    """model.safetensors (one tiny tensor, header and size consistent), config.json, tokenizer.json."""
    folder.mkdir(parents=True, exist_ok=True)
    h = json.dumps({"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}).encode()
    h += b" " * (-len(h) % 8)
    (folder / "model.safetensors").write_bytes(struct.pack("<Q", len(h)) + h + b"\0" * 8)
    (folder / "config.json").write_text("{}")
    (folder / "tokenizer.json").write_text("{}")


class FakeProc:
    def __init__(self, pid=424242, code=None):
        self.pid, self.code = pid, code

    def poll(self):
        return self.code


class Base(T.TempHome):
    def setUp(self):
        super().setUp()
        self.cache = self.tmp / "cache"
        os.environ["MOTH_CACHE"] = str(self.cache)
        os.environ.pop("MOTH_URL", None)
        self.url_state = ["none"]   # what /api/health says: "none" | "moth" | "other"
        self.launched: list = []
        self.proc = FakeProc()
        self.fetched: list = []

        def health(url, timeout=2.0):
            k = self.url_state[0]
            if len(self.url_state) > 1:
                self.url_state.pop(0)
            return (k, {"ok": True, "pid": 777}) if k == "moth" else (k, "nothing listens" if k == "none" else
                                                                     "it answers HTTP 404")

        def launch(url, log):
            self.launched.append((url, log))
            self.url_state[:] = ["none", "moth"]   # the server answers on the second poll
            return self.proc

        def fetch(folder):
            self.fetched.append(folder)
            make_weights(Path(folder))
            return folder

        self.pyside = [None]
        self.patches = [
            unittest.mock.patch("moth.weights.mode", return_value="pip"),
            unittest.mock.patch.object(S, "probe_torch", return_value={"version": "2.9.0", "cuda": False}),
            unittest.mock.patch.object(S, "probe_pyside", side_effect=lambda: self.pyside[0] if len(self.pyside) == 1
                                       else self.pyside.pop(0)),
            unittest.mock.patch.object(S, "ram_gb", return_value=12.0),
            unittest.mock.patch.object(S, "has_display", return_value=True),
            unittest.mock.patch.object(S, "health", side_effect=health),
            unittest.mock.patch.object(S, "launch_pet", side_effect=launch),
            unittest.mock.patch.object(S, "fetch_weights", side_effect=fetch),
            unittest.mock.patch.object(S, "alive", side_effect=lambda pid: pid == self.proc.pid and
                                       self.proc.code is None),
            unittest.mock.patch.object(S.time, "sleep"),
            unittest.mock.patch.object(S, "_tty", return_value=True),
            unittest.mock.patch("importlib.util.find_spec", side_effect=self._find_spec),
            unittest.mock.patch.object(S.Setup, "_pet_pkg", return_value="PySide6-Essentials"),   # HF layout
        ]
        for p in self.patches:
            p.start()

    _real_find_spec = staticmethod(__import__("importlib.util").util.find_spec)

    def _find_spec(self, name, *a):
        if name in ("tokenizers", "safetensors", "numpy", "huggingface_hub"):
            return object()   # present, whatever this test Python has
        return Base._real_find_spec(name, *a)

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        super().tearDown()

    def harnesses(self):
        """Claude Code (binary + ~/.claude), Pi (~/.pi), Codex (binary only)."""
        self.fake_bin("claude")
        self.write(".claude/settings.json", json.dumps({"model": "opus"}))
        (self.home / ".pi" / "agent").mkdir(parents=True)
        self.fake_bin("codex")

    def setup_run(self, *args, answers=None):
        side = answers if answers is not None else EOFError
        out = T.io.StringIO()
        with T.contextlib.redirect_stdout(out), T.contextlib.redirect_stderr(out), \
                unittest.mock.patch("builtins.input", side_effect=side):
            rc = S.main(list(args))
        return rc, out.getvalue()

    def tree(self) -> dict:
        return {str(p.relative_to(self.home)): (p.stat().st_size, p.stat().st_mtime_ns) if p.is_file() else None
                for p in sorted(self.home.rglob("*"))}

    def autostart_file(self) -> Path:
        return self.home / ".config" / "autostart" / "moth-pet.desktop"


class DryRun(Base):
    def test_dry_run_prints_the_plan_and_writes_nothing(self):
        self.harnesses()
        make_weights(Path(S.weights.cache_dir()))
        before = self.tree()
        rc, out = self.setup_run("--dry-run")
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.tree(), before)
        for want in ("dry run", "1 packages", "2 weights", "3 memory", "4 coding agents", "5 autostart", "6 the pet",
                     "hook moth into claude-code? yes", "hook moth into pi? yes", "hook moth into codex? yes",
                     "would install:", "claude-code: add moth's Stop hook", "(dry run: nothing changed)",
                     "show_in_harness)? no", f"would write {self.autostart_file()}", "would run:", "moth.pet --url",
                     "not found: hermes, opencode, dsh", "uninstall: python -m moth unhook --all"):
            self.assertIn(want, out)
        self.assertNotIn("[Y/n]", out)
        self.assertEqual(self.launched, [])

    def test_missing_weights_dry_run_doesnt_download(self):
        rc, out = self.setup_run("--dry-run")
        self.assertEqual(rc, 0, out)
        self.assertIn("not downloaded yet: about 860 MB", out)
        self.assertIn(str(self.cache), out)
        self.assertIn("download them now? yes", out)
        self.assertEqual(self.fetched, [])
        self.assertFalse(self.cache.exists())


class Yes(Base):
    def test_yes_does_everything_then_a_second_run_changes_nothing(self):
        self.harnesses()
        rc, out = self.setup_run("--yes")
        self.assertEqual(rc, 0, out)
        # weights: downloaded into the pip cache, once
        self.assertEqual(self.fetched, [str(self.cache / "darioooooo0o" / "moth-112m")])
        # hooks: installed, with the backup shown and Codex's approval note
        settings = json.loads((self.home / ".claude" / "settings.json").read_text())
        self.assertEqual(settings["model"], "opus")
        self.assertTrue(any(T.runs(h["command"], T.INTEG / "claude_code.py")
                            for e in settings["hooks"]["Stop"] for h in e["hooks"]))
        self.assertTrue((self.home / ".pi" / "agent" / "extensions" / "moth.ts").is_file())
        self.assertTrue((self.home / ".codex" / "hooks.json").is_file())
        self.assertIn("backup: ", out)
        self.assertIn("approve it once", out)
        # show_in_harness: default off, nothing written
        self.assertFalse((self.home / ".moth" / "config.json").exists())
        # autostart on, the pet launched once and moth answered
        self.assertTrue(self.autostart_file().is_file())
        self.assertEqual(len(self.launched), 1)
        self.assertEqual(self.launched[0][0], "http://127.0.0.1:4747")
        self.assertEqual(self.launched[0][1], str(self.home / ".moth" / "pet.log"))
        self.assertIn("moth is awake at http://127.0.0.1:4747 (pet pid 424242, server pid 777)", out)
        self.assertIn("all set.", out)
        self.assertFalse((self.home / ".moth" / "setup-pet.pid").exists())   # only kept while moth wakes up

        # second run: moth answers, everything's in place -> no question, no change, no launch
        before = self.tree()
        backups = sorted(self.home.rglob("*.bak-moth-*"))
        self.url_state[:] = ["moth"]
        rc, out2 = self.setup_run("--yes")
        self.assertEqual(rc, 0, out2)
        self.assertEqual(self.tree(), before)
        self.assertEqual(sorted(self.home.rglob("*.bak-moth-*")), backups)
        self.assertEqual(len(self.launched), 1)
        self.assertEqual(len(self.fetched), 1)
        self.assertNotRegex(out2, r"(?m)\? (yes|no)$")   # not a single question
        self.assertIn("show_in_harness: off (default)", out2)
        self.assertIn("moth is already awake", out2)
        self.assertIn("no pet on your screen? python -m moth.pet", out2)
        self.assertIn("all set.", out2)
        self.assertNotIn("fluttering", out2)   # only said when this run started the pet
        self.assertIn("already there: packages, weights, claude-code hook, codex hook, pi hook, autostart, "
                      "moth running", out2)
        self.assertIn("Codex runs it only once approved: /hooks in codex", out2)

    def test_rerun_while_the_pet_is_still_waking_moth_doesnt_start_a_second(self):
        make_weights(Path(S.weights.cache_dir()))
        self.url_state[:] = ["none"]
        self.launched_pet_before = None
        S.os.makedirs(self.home / ".moth", exist_ok=True)
        (self.home / ".moth" / "setup-pet.pid").write_text(json.dumps({"pid": self.proc.pid,
                                                                        "url": "http://127.0.0.1:4747"}))
        self.url_state[:] = ["none", "none", "none", "moth"]
        rc, out = self.setup_run("--yes")
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.launched, [])
        self.assertIn("still waking moth up (pid 424242)", out)
        self.assertIn("moth is awake", out)

    def test_custom_url_and_moth_url(self):
        make_weights(Path(S.weights.cache_dir()))
        rc, out = self.setup_run("--yes", "--url", "http://127.0.0.1:47999/")
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.launched[0][0], "http://127.0.0.1:47999")
        os.environ["MOTH_URL"] = "http://127.0.0.1:47998"
        self.url_state[:] = ["none"]
        rc, out = self.setup_run("--yes")
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.launched[1][0], "http://127.0.0.1:47998")


class NoTTY(Base):
    def test_no_tty_is_a_dry_run(self):
        self.harnesses()
        before = self.tree()
        with unittest.mock.patch.object(S, "_tty", return_value=False):
            rc, out = self.setup_run(answers=AssertionError("must not ask"))
        self.assertEqual(rc, 0, out)
        self.assertIn("no terminal to ask in, so this is a dry run", out)
        self.assertEqual(self.tree(), before)
        self.assertEqual((self.launched, self.fetched), ([], []))

    def test_no_tty_with_yes_acts(self):
        self.harnesses()
        with unittest.mock.patch.object(S, "_tty", return_value=False):
            rc, out = self.setup_run("--yes", answers=AssertionError("must not ask"))
        self.assertEqual(rc, 0, out)
        self.assertNotIn("dry run", out)
        self.assertTrue(self.autostart_file().is_file())
        self.assertEqual(len(self.launched), 1)


class Interactive(Base):
    def test_answers_and_eof(self):
        """pip prompt: Enter = no (default); claude-code y, codex n, pi y, show_in_harness y; then EOF: autostart no.
        PySide6 is missing, so the pet isn't offered at all."""
        self.harnesses()
        make_weights(Path(S.weights.cache_dir()))
        self.pyside = ["not installed"]
        with unittest.mock.patch.object(S.subprocess, "call") as call:
            rc, out = self.setup_run(answers=["", "y", "n", "y", "y", EOFError])
        call.assert_not_called()
        self.assertEqual(rc, 0, out)
        self.assertIn("pip install PySide6-Essentials", out)
        self.assertTrue("claude_code.py" in (self.home / ".claude" / "settings.json").read_text())
        self.assertFalse((self.home / ".codex" / "hooks.json").exists())
        self.assertTrue((self.home / ".pi" / "agent" / "extensions" / "moth.ts").is_file())
        self.assertTrue(json.loads((self.home / ".moth" / "config.json").read_text())["show_in_harness"])
        self.assertFalse(self.autostart_file().exists())   # skipped: needs PySide6 (never even asked)
        self.assertIn("skipped: the pet needs PySide6", out)
        self.assertEqual(self.launched, [])
        self.assertIn("left:", out)
        self.assertIn("codex hook", out)

    def test_eof_means_no(self):
        self.harnesses()
        make_weights(Path(S.weights.cache_dir()))
        before = self.tree()
        rc, out = self.setup_run(answers=EOFError)
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.tree(), before)
        self.assertEqual(self.launched, [])

    def test_pip_only_on_an_explicit_yes(self):
        make_weights(Path(S.weights.cache_dir()))
        self.pyside = ["not installed", None]   # the first check, then fine after pip
        with unittest.mock.patch.object(S.subprocess, "call", return_value=0) as call:
            rc, out = self.setup_run(answers=["y", EOFError, EOFError])   # pip yes; autostart, pet: EOF
        call.assert_called_once_with([sys.executable, "-m", "pip", "install", "PySide6-Essentials"])
        self.assertIn("installed packages", out)
        self.assertNotIn("sudo ", out.replace("no sudo", ""))

    def test_yes_never_runs_pip(self):
        make_weights(Path(S.weights.cache_dir()))
        self.pyside = ["not installed"]
        with unittest.mock.patch.object(S.subprocess, "call") as call:
            rc, out = self.setup_run("--yes")
        call.assert_not_called()
        self.assertIn("run it now (pip", out)
        self.assertIn("? no", out)
        self.assertEqual(self.launched, [])


class Problems(Base):
    def test_partial_hugging_face_download(self):
        repo = self.tmp / "moth-112m"
        (repo / "code" / "moth").mkdir(parents=True)
        (repo / "config.json").write_text("{}")
        (repo / "tokenizer.json").write_text("{}")
        with unittest.mock.patch("moth.weights.mode", return_value="hf"), \
                unittest.mock.patch.object(S.weights, "ROOT", str(repo / "code")):
            rc, out = self.setup_run("--yes")
        self.assertEqual(rc, 1, out)   # moth can't run yet
        self.assertIn("missing model files: model.safetensors", out)
        self.assertIn(f"cd {repo / 'code'} && python -m moth update", out)
        self.assertIn("hf download darioooooo0o/moth-112m --local-dir", out)
        self.assertIn("no weights yet (step 2)", out)
        self.assertEqual((self.launched, self.fetched), ([], []))

    def test_pet_dies_shows_the_log(self):
        make_weights(Path(S.weights.cache_dir()))
        self.proc = FakeProc(code=2)
        log = self.home / ".moth" / "pet.log"
        log.parent.mkdir(parents=True)
        log.write_text("moth pet: not enough memory to wake moth\n")
        rc, out = self.setup_run("--yes")
        self.assertEqual(rc, 1, out)
        self.assertIn("the pet stopped before moth woke up (exit 2)", out)
        self.assertIn("not enough memory to wake moth", out)
        self.assertFalse((self.home / ".moth" / "setup-pet.pid").exists())

    def test_port_taken(self):
        make_weights(Path(S.weights.cache_dir()))
        self.url_state[:] = ["other"]
        rc, out = self.setup_run("--yes")
        self.assertIn("something else answers at http://127.0.0.1:4747", out)
        self.assertEqual(self.launched, [])

    def test_low_memory_warns_but_starts(self):
        make_weights(Path(S.weights.cache_dir()))
        with unittest.mock.patch.object(S, "ram_gb", return_value=0.5):   # estimates run low (macOS): never a block
            rc, out = self.setup_run("--yes")
        self.assertIn("0.5 GB available by this system's count", out)
        self.assertEqual(len(self.launched), 1, out)

    def test_download_failure(self):
        with unittest.mock.patch.object(S, "fetch_weights", side_effect=SystemExit("moth: download failed (boom)")):
            rc, out = self.setup_run("--yes")
        self.assertEqual(rc, 1, out)
        self.assertIn("download failed (boom)", out)
        self.assertEqual(self.launched, [])


class Wiring(unittest.TestCase):
    def test_usage_and_dispatch(self):
        from moth import __main__ as M
        loaded = "moth.serve" in sys.modules   # another test module in this process may have imported it
        self.assertIn("python -m moth setup", M.USAGE)
        with unittest.mock.patch("moth.setup.main", return_value=0) as sm:
            with self.assertRaises(SystemExit) as cm:
                M.main(["setup", "--dry-run"])
        self.assertEqual(cm.exception.code, 0)
        sm.assert_called_once_with(["--dry-run"])
        if not loaded:   # setup's dispatch doesn't pull in the server (and torch)
            self.assertNotIn("moth.serve", sys.modules)

    def test_shipped_in_the_hub_build(self):
        self.assertRegex((ROOT / "release" / "build.sh").read_text(), r'MOTH_FILES="[^"]*\bsetup\b')


if __name__ == "__main__":
    unittest.main()
