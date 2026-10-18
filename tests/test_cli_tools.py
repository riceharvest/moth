"""moth hooks / hook / unhook, moth doctor, moth autostart and moth forget, in temporary HOME / XDG folders.

  python3 tests/test_cli_tools.py      (from ~/moth; a few seconds, temp dirs only, no model, no server)

Every test runs with HOME, XDG_CONFIG_HOME, APPDATA and PATH pointing into a temp folder (CLAUDE_CONFIG_DIR,
CODEX_HOME, HERMES_HOME, PI_CODING_AGENT_DIR unset), fake harness configs and a fake `hermes` on PATH, and ports
nobody uses. A module-level guard checks that the real ~/.claude, ~/.codex, ~/.hermes, ~/.pi, ~/.config/opencode,
~/.config/autostart, ~/.moth and the checkout's live/ folder are byte-for-byte the same after the run.

Covers: hook/unhook roundtrips per harness (other hooks and keys kept, backups made, idempotent, outdated copies
detected and replaced, Codex never both variants, foreign files left alone, dry run and cancel change nothing);
doctor with torch missing / broken (a fake torch on PYTHONPATH, through the real subprocess probe) and present, the
port states and that it never imports serve; autostart files written / refused / removed for Linux, macOS and
Windows; forget refusing while a moth answers, deleting only marked homes, and the token only with --all --token.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import plistlib
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
import unittest.mock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from moth import autostart as A  # noqa: E402
from moth import doctor as D  # noqa: E402
from moth import forget as F  # noqa: E402
from moth import hooks as H  # noqa: E402

INTEG = ROOT / "moth" / "integrations"
REAL_HOME = Path(os.path.expanduser("~"))


# --- guard: the user's real configs must not change --------------------------------------------------------------
def _fingerprint() -> dict:
    out = {}
    for base in [REAL_HOME / ".claude" / "settings.json", REAL_HOME / ".codex" / "hooks.json",
                 REAL_HOME / ".codex" / "config.toml", REAL_HOME / ".hermes" / "config.yaml",
                 REAL_HOME / ".hermes" / "plugins", REAL_HOME / ".pi" / "agent" / "extensions",
                 REAL_HOME / ".config" / "opencode" / "plugins", REAL_HOME / ".config" / "autostart",
                 REAL_HOME / ".dsh", REAL_HOME / ".moth", ROOT / "live" / "home",
                 REAL_HOME / ".config" / "systemd" / "user"] + sorted((ROOT / "live").glob("*.log")):
        if base.is_file():
            out[str(base)] = (base.stat().st_size, base.stat().st_mtime_ns)
        elif base.is_dir():
            out[str(base)] = sorted(p.name for p in base.iterdir())
    return out


BEFORE = _fingerprint()


def tearDownModule():
    after = _fingerprint()
    changed = {k for k in set(BEFORE) | set(after) if BEFORE.get(k) != after.get(k)
               and not k.endswith(".log") and "serve-" not in str(k)}   # the live moth may log / rewrite its pidfile
    if changed:
        raise AssertionError(f"tests touched real files: {sorted(changed)}")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def fwd(path) -> str:
    """A path as moth writes it into a hook command (forward slashes on Windows, see hooks.hook_command)."""
    return str(path).replace("\\", "/") if os.name == "nt" else str(path)


def runs(cmd: str, script) -> bool:
    """Does hook command `cmd` run this script? (Windows: moth writes forward slashes, see hooks.hook_command.)"""
    found = H.moth_script_in(cmd, os.path.basename(str(script)))
    norm = lambda p: os.path.normcase(os.path.normpath(str(p)))   # noqa: E731
    return found is not None and norm(found) == norm(script)


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


FAKE_HERMES = r'''import os, re, sys
cfg = os.path.join(os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes"), "config.yaml")
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "hermes.log"), "a") as f:
    f.write(" ".join(sys.argv[1:]) + "\n")
text = open(cfg).read() if os.path.exists(cfg) else "plugins:\n  enabled: []\n"


def terminal():
    if sys.stdin is None or sys.stdin.closed or not sys.stdin.isatty():
        return False
    if os.name != "nt":
        return True
    import ctypes, msvcrt   # Windows: NUL is a character device, so isatty() says yes; a console has a mode
    mode = ctypes.c_ulong()
    return bool(ctypes.windll.kernel32.GetConsoleMode(msvcrt.get_osfhandle(sys.stdin.fileno()), ctypes.byref(mode)))


if terminal():
    sys.exit("a prompt would hang here: moth must run hermes without a terminal on stdin")
if sys.argv[1:3] == ["plugins", "enable"]:
    text = text.replace("  enabled: []\n", "  enabled:\n").replace("  enabled:\n", "  enabled:\n    - " + sys.argv[-1] + "\n", 1)
elif sys.argv[1:3] == ["plugins", "disable"]:
    text = re.sub(r"(?m)^    - " + re.escape(sys.argv[-1]) + r"\n", "", text)
open(cfg, "w").write(text)
'''


class TempHome(unittest.TestCase):
    """A fresh HOME / XDG / PATH per test."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="moth-cli-"))
        self.home = self.tmp / "home"
        self.bin = self.tmp / "bin"
        self.home.mkdir()
        self.bin.mkdir()
        env = {"HOME": str(self.home), "USERPROFILE": str(self.home), "XDG_CONFIG_HOME": str(self.home / ".config"),
               "APPDATA": str(self.home / "AppData" / "Roaming"), "PATH": str(self.bin),
               "MOTH_CONFIG_FILE": str(self.home / ".moth" / "config.json"),
               "MOTH_TOKEN_FILE": str(self.home / ".moth" / "token"), "XDG_CACHE_HOME": str(self.home / ".cache")}
        self.envp = unittest.mock.patch.dict(os.environ, env)
        self.envp.start()
        for k in ("CLAUDE_CONFIG_DIR", "CODEX_HOME", "HERMES_HOME", "PI_CODING_AGENT_DIR", "MOTH_CACHE",
                  "MOTH_SHOW_IN_HARNESS"):
            os.environ.pop(k, None)
        self.assertEqual(os.path.expanduser("~"), str(self.home))

    def tearDown(self):
        self.envp.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write(self, rel: str, text: str) -> Path:
        p = self.home / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        return p

    def fake_bin(self, name: str, body: str | None = None):
        """A program `name` on PATH running the Python `body` (default: exit 0). POSIX: a script with a shebang to
        this interpreter. Windows: name.cmd (found through PATHEXT, as npm's shims are) calling _name.py."""
        body = body or "raise SystemExit(0)\n"
        if os.name == "nt":
            (self.bin / f"_{name}.py").write_text(body)
            (self.bin / f"{name}.cmd").write_text(f'@"{sys.executable}" "%~dp0_{name}.py" %*\r\n')
        else:
            p = self.bin / name
            p.write_text(f"#!{sys.executable}\n{body}")
            p.chmod(0o755)

    def remove_bin(self, name: str):
        for p in (self.bin / name, self.bin / f"{name}.cmd", self.bin / f"_{name}.py"):
            p.unlink(missing_ok=True)
        self.assertIsNone(shutil.which(name, path=str(self.bin)))

    def run_cli(self, mod, *args, stdin_eof: bool = True):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out), \
                unittest.mock.patch("builtins.input", side_effect=EOFError):
            rc = mod(*args)
        return rc, out.getvalue()

    def hook(self, *args):
        return self.run_cli(H.main, "hook", list(args))

    def unhook(self, *args):
        return self.run_cli(H.main, "unhook", list(args))

    def backups(self, path: Path) -> list[Path]:
        return sorted(path.parent.glob(path.name + ".bak-moth-*"))


# --- hooks ------------------------------------------------------------------------------------------------------
class ClaudeCode(TempHome):
    def settings(self, data) -> Path:
        return self.write(".claude/settings.json", json.dumps(data, indent=2))

    def test_roundtrip_keeps_other_hooks_and_keys(self):
        orig = {"model": "opus", "permissions": {"allow": ["Bash(ls)"]},
                "hooks": {"SessionStart": [{"hooks": [{"type": "command", "command": "echo hi"}]}],
                          "Stop": [{"matcher": "", "hooks": [{"type": "command", "command": "notify-send done"}]}]}}
        p = self.settings(orig)
        rc, out = self.hook("claude-code", "--yes")
        self.assertEqual(rc, 0, out)
        self.assertIn("done: add moth's Stop hook", out)
        d = json.loads(p.read_text())
        self.assertEqual(d["model"], "opus")
        self.assertEqual(d["permissions"], orig["permissions"])
        self.assertEqual(d["hooks"]["SessionStart"], orig["hooks"]["SessionStart"])
        self.assertEqual(d["hooks"]["Stop"][0], orig["hooks"]["Stop"][0])   # the other Stop hook keeps index 0
        mine = d["hooks"]["Stop"][1]["hooks"][0]
        # foreground (no "async"): a one-shot `claude -p` kills a background hook before it posts
        self.assertEqual((mine.get("async"), mine["timeout"], mine["type"]), (None, 3, "command"))
        self.assertTrue(runs(mine["command"], INTEG / "claude_code.py"), mine["command"])
        self.assertEqual(len(self.backups(p)), 1)
        self.assertEqual(json.loads(self.backups(p)[0].read_text()), orig)
        self.assertEqual(H.status("claude-code")["state"], "installed")

        before = sha(p)   # idempotent: nothing written, no new backup
        rc, out = self.hook("claude-code", "--yes")
        self.assertEqual(rc, 0)
        self.assertIn("already installed", out)
        self.assertEqual(sha(p), before)
        self.assertEqual(len(self.backups(p)), 1)

        rc, out = self.unhook("claude-code", "--yes")
        self.assertEqual(rc, 0, out)
        self.assertEqual(json.loads(p.read_text()), orig)
        self.assertEqual(len(self.backups(p)), 2)
        rc, out = self.unhook("claude-code", "--yes")
        self.assertIn("not installed, nothing to remove", out)

    def test_new_file_and_project_settings(self):
        proj = self.tmp / "proj"
        proj.mkdir()
        rc, out = self.hook("claude-code", "--project", str(proj), "--yes")
        self.assertEqual(rc, 0, out)
        p = proj / ".claude" / "settings.json"
        self.assertEqual(len(json.loads(p.read_text())["hooks"]["Stop"]), 1)
        self.assertFalse((self.home / ".claude" / "settings.json").exists())   # global untouched
        self.assertEqual(self.backups(p), [])                                  # nothing to back up
        self.assertEqual(H.status("claude-code", str(proj))["state"], "installed")
        self.assertEqual(H.status("claude-code")["state"], "absent")
        rc, _ = self.unhook("claude-code", "--project", str(proj), "--yes")
        self.assertNotIn("hooks", json.loads(p.read_text()))

    def test_hand_install_with_other_python_counts(self):
        cmd = f"/opt/some/venv/bin/python {fwd(INTEG / 'claude_code.py')}"
        p = self.settings({"hooks": {"Stop": [{"hooks": [{"type": "command", "command": cmd, "timeout": 10}]}]}})
        self.assertEqual(H.status("claude-code")["state"], "installed")
        before = sha(p)
        rc, out = self.hook("claude-code", "--yes")
        self.assertIn("already installed", out)
        self.assertEqual(sha(p), before)

    def test_background_hook_is_outdated_and_fixed_in_place(self):
        cmd = f"/opt/some/venv/bin/python {fwd(INTEG / 'claude_code.py')}"
        other = {"type": "command", "command": "notify-send done", "async": True}
        p = self.settings({"hooks": {"Stop": [{"hooks": [other]},
                                              {"hooks": [{"type": "command", "command": cmd, "async": True,
                                                          "timeout": 5}]}]}})
        st = H.status("claude-code")
        self.assertEqual(st["state"], "outdated")
        self.assertIn("claude -p", st["detail"])
        rc, out = self.hook("claude-code", "--yes")
        self.assertEqual(rc, 0, out)
        self.assertIn("foreground", out)
        stop = json.loads(p.read_text())["hooks"]["Stop"]
        self.assertEqual(stop[0]["hooks"][0], other)   # someone else's async hook is theirs: untouched
        self.assertEqual(stop[1]["hooks"], [{"type": "command", "command": cmd, "timeout": 3}])   # same python
        self.assertEqual(len(self.backups(p)), 1)
        self.assertEqual(H.status("claude-code")["state"], "installed")
        rc, out = self.hook("claude-code", "--yes")
        self.assertIn("already installed", out)

    def test_moved_install_is_updated_in_place(self):
        cmd = "MOTH_URL=http://127.0.0.1:4750 python3 '/gone/moth-112m/code/moth/integrations/claude_code.py'"
        p = self.settings({"hooks": {"Stop": [{"hooks": [{"type": "command", "command": cmd}]}]}})
        self.assertEqual(H.status("claude-code")["state"], "broken")
        rc, out = self.hook("claude-code", "--yes")
        self.assertEqual(rc, 0, out)
        stop = json.loads(p.read_text())["hooks"]["Stop"]
        self.assertEqual(len(stop), 1)   # replaced, not duplicated
        self.assertTrue(runs(stop[0]["hooks"][0]["command"], INTEG / "claude_code.py"))

    def test_symlinked_settings_keep_their_link(self):
        real = self.tmp / "dotfiles" / "settings.json"
        real.parent.mkdir()
        real.write_text('{"model": "x"}')
        (self.home / ".claude").mkdir()
        link = self.home / ".claude" / "settings.json"
        link.symlink_to(real)
        rc, out = self.hook("claude-code", "--yes")
        self.assertEqual(rc, 0, out)
        self.assertTrue(link.is_symlink())
        self.assertEqual(json.loads(real.read_text())["model"], "x")
        self.assertIn("Stop", json.loads(real.read_text())["hooks"])

    def test_invalid_json_is_left_alone(self):
        p = self.write(".claude/settings.json", "{ not json")
        rc, out = self.hook("claude-code", "--yes")
        self.assertEqual(rc, 1)
        self.assertIn("isn't valid JSON", out)
        self.assertEqual(p.read_text(), "{ not json")
        self.assertEqual(self.backups(p), [])

    def test_dry_run_and_cancel_change_nothing(self):
        p = self.settings({"model": "x"})
        before = sha(p)
        rc, out = self.hook("claude-code", "--dry-run")
        self.assertIn("would install", out)
        self.assertIn("dry run: nothing changed", out)
        rc, out = self.hook("claude-code")   # no --yes, stdin closed: cancelled
        self.assertEqual(rc, 1)
        self.assertIn("cancelled", out)
        self.assertEqual(sha(p), before)
        self.assertEqual(self.backups(p), [])


class Codex(TempHome):
    HERDR = {"hooks": {"SessionStart": [{"hooks": [{"command": "bash herdr.sh session", "timeout": 10,
                                                    "type": "command"}]}]}}
    TOML = '# my codex\nmodel = "gpt-5"\n\n[projects."/x"]\ntrust_level = "trusted"\n'

    def setUp(self):
        super().setUp()
        self.hooks_json = self.write(".codex/hooks.json", json.dumps(self.HERDR, indent=2))
        self.toml = self.write(".codex/config.toml", self.TOML)

    def test_hooks_json_by_default_and_never_both(self):
        rc, out = self.hook("codex", "--yes")
        self.assertEqual(rc, 0, out)
        self.assertIn("/hooks", out)   # the one-time approval reminder
        d = json.loads(self.hooks_json.read_text())
        self.assertEqual(d["hooks"]["SessionStart"], self.HERDR["hooks"]["SessionStart"])
        self.assertTrue(runs(d["hooks"]["Stop"][0]["hooks"][0]["command"], INTEG / "codex.py"))
        self.assertEqual(self.toml.read_text(), self.TOML)   # legacy variant not touched
        self.assertEqual(H.status("codex")["state"], "installed")
        rc, out = self.hook("codex", "--legacy-notify", "--yes")
        self.assertEqual(rc, 1)
        self.assertIn("never installs both", out)
        self.assertEqual(self.toml.read_text(), self.TOML)

        rc, out = self.unhook("codex", "--yes")
        self.assertEqual(json.loads(self.hooks_json.read_text()), self.HERDR)

    def test_background_hook_is_rewritten_and_needs_approval_again(self):
        cmd = f"python3 {fwd(INTEG / 'codex.py')}"
        d = json.loads(self.hooks_json.read_text())
        d["hooks"]["Stop"] = [{"hooks": [{"type": "command", "command": cmd, "async": True, "timeout": 5}]}]
        self.hooks_json.write_text(json.dumps(d, indent=2))
        self.assertEqual(H.status("codex")["state"], "outdated")
        rc, out = self.hook("codex", "--yes")
        self.assertEqual(rc, 0, out)
        self.assertIn("approve it once", out)   # a changed entry is untrusted until approved again in codex
        stop = json.loads(self.hooks_json.read_text())["hooks"]["Stop"]
        self.assertEqual(stop, [{"hooks": [{"type": "command", "command": cmd, "timeout": 3}]}])
        self.assertEqual(json.loads(self.hooks_json.read_text())["hooks"]["SessionStart"],
                         self.HERDR["hooks"]["SessionStart"])
        self.assertEqual(H.status("codex")["state"], "installed")

    def test_legacy_notify_variant(self):
        rc, out = self.hook("codex", "--legacy-notify", "--yes")
        self.assertEqual(rc, 0, out)
        text = self.toml.read_text()
        self.assertTrue(text.startswith("# my codex\nnotify = ["), text)   # top level: before any [table]
        self.assertEqual(H.codex_notify(text)[1], [sys.executable, str(INTEG / "codex.py")])   # escaped as TOML
        self.assertIn('[projects."/x"]\ntrust_level = "trusted"\n', text)
        if sys.version_info >= (3, 11):
            import tomllib
            self.assertEqual(tomllib.loads(text)["notify"][1], str(INTEG / "codex.py"))
        self.assertEqual(len(self.backups(self.toml)), 1)
        self.assertEqual(H.status("codex")["state"], "installed")
        before = sha(self.hooks_json)
        rc, out = self.hook("codex", "--yes")   # the default variant is not added on top
        self.assertIn("already installed as legacy notify", out)
        self.assertEqual(sha(self.hooks_json), before)
        rc, out = self.unhook("codex", "--yes")
        self.assertEqual(self.toml.read_text(), self.TOML)

    def test_both_present_is_flagged_and_unhook_removes_both(self):
        self.hook("codex", "--yes")
        self.toml.write_text(f'notify = {json.dumps(["python3", str(INTEG / "codex.py")])}\n' + self.TOML)   # valid TOML
        st = H.status("codex")
        self.assertEqual(st["state"], "error")
        self.assertIn("BOTH", st["detail"])
        rc, out = self.hook("codex", "--yes")
        self.assertEqual(rc, 1)
        rc, out = self.unhook("codex", "--yes")
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.toml.read_text(), self.TOML)
        self.assertEqual(json.loads(self.hooks_json.read_text()), self.HERDR)

    def test_foreign_notify_is_kept(self):
        self.toml.write_text('notify = ["notify-send", "codex"]\n' + self.TOML)
        rc, out = self.hook("codex", "--legacy-notify", "--yes")
        self.assertEqual(rc, 1)
        self.assertIn("one notify slot", out)
        rc, out = self.unhook("codex", "--yes")
        self.assertIn('notify = ["notify-send", "codex"]', self.toml.read_text())


class WindowsPathsInConfigs(TempHome):
    """Every OS: an install under a Windows path (backslashes, a space) is written into JSON / TOML escaped, read back
    exactly, and removed again. A hand-written TOML with raw backslashes (invalid escapes) doesn't crash `moth hooks`."""

    PY = r"C:\Program Files\Python312\python.exe"
    INTEG = r"C:\Users\Me Q\moth\integrations"
    TOML = '# my codex\nmodel = "gpt-5"\n'

    def setUp(self):
        super().setUp()
        self.patches = [unittest.mock.patch.object(sys, "executable", self.PY),
                        unittest.mock.patch.object(H, "INTEGRATIONS", self.INTEG)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        super().tearDown()

    def script(self, name: str) -> str:
        return os.path.join(self.INTEG, name)

    def test_claude_code_settings_json(self):
        orig = {"model": "opus"}
        p = self.write(".claude/settings.json", json.dumps(orig))
        rc, out = self.hook("claude-code", "--yes")
        self.assertEqual(rc, 0, out)
        cmd = json.loads(p.read_text())["hooks"]["Stop"][0]["hooks"][0]["command"]   # valid JSON, path escaped
        self.assertTrue(runs(cmd, self.script("claude_code.py")), cmd)
        if os.name != "nt":
            import shlex
            self.assertEqual(shlex.split(cmd), [self.PY, self.script("claude_code.py")])
        rc, out = self.unhook("claude-code", "--yes")
        self.assertEqual(rc, 0, out)
        self.assertEqual(json.loads(p.read_text()), orig)

    def test_codex_legacy_notify_toml(self):
        self.write(".codex/hooks.json", "{}")
        toml = self.write(".codex/config.toml", self.TOML)
        rc, out = self.hook("codex", "--legacy-notify", "--yes")
        self.assertEqual(rc, 0, out)
        text = toml.read_text()
        want = [self.PY, self.script("codex.py")]
        self.assertIn(json.dumps(self.PY), text)   # backslashes doubled: a TOML basic string
        self.assertEqual(H.codex_notify(text)[1], want)
        if sys.version_info >= (3, 11):
            import tomllib
            self.assertEqual(tomllib.loads(text)["notify"], want)
        self.assertIn(H.status("codex")["state"], ("installed", "broken", "other"))
        rc, out = self.unhook("codex", "--yes")
        self.assertEqual(rc, 0, out)
        self.assertEqual(toml.read_text(), self.TOML)

    def test_hand_written_raw_backslashes(self):
        self.write(".codex/hooks.json", "{}")
        raw = 'notify = ["%s", "%s"]\n' % (self.PY, self.script("codex.py"))   # invalid TOML escapes (\P, \U...)
        toml = self.write(".codex/config.toml", raw + self.TOML)
        self.assertEqual(H.codex_notify(toml.read_text())[1], [self.PY, self.script("codex.py")])
        H.status("codex")   # no traceback
        rc, out = self.run_cli(H.main, "hooks", [])
        self.assertEqual(rc, 0, out)
        rc, out = self.unhook("codex", "--yes")
        self.assertEqual(rc, 0, out)
        self.assertEqual(toml.read_text(), self.TOML)

    def test_windows_hook_command_uses_forward_slashes(self):
        with unittest.mock.patch.object(H, "_WIN", True):
            cmd = H.hook_command(self.PY, self.script("claude_code.py"))
            self.assertNotIn("\\", cmd)   # Git Bash would eat unquoted backslashes
            self.assertEqual(H.moth_script_in(cmd, "claude_code.py").replace("/", "\\"),
                             self.script("claude_code.py").replace("/", "\\"))
            # an older moth's backslash hook is outdated there, and `moth hook` rewrites it
            p = self.write(".claude/settings.json", json.dumps({"hooks": {"Stop": [{"hooks": [H.hook_entry(
                H.command(self.PY, self.script("claude_code.py")).replace("/", "\\"))]}]}}))
            with unittest.mock.patch.object(H, "_where", return_value="this"):
                self.assertEqual(H.status("claude-code")["state"], "outdated")
                rc, out = self.hook("claude-code", "--yes")
                self.assertEqual(rc, 0, out)
                self.assertIn("forward slashes", out)
                fixed = json.loads(p.read_text())["hooks"]["Stop"][0]["hooks"][0]["command"]
                self.assertNotIn("\\", fixed)
                self.assertTrue(runs(fixed, self.script("claude_code.py").replace("\\", "/")), fixed)
                self.assertEqual(H.status("claude-code")["state"], "installed")

    def test_toml_escapes(self):
        for body, want in ((r"a\\b", "a\\b"), (r"tab\there", "tab\there"), (r"\u00e9\U0001F98B", "\u00e9\U0001F98B"),
                           (r"\e[0m", "\x1b[0m"), (r'say \"hi\"', 'say "hi"'), (r"C:\Users\x", r"C:\Users\x"),
                           (r"bad \q", r"bad \q"), (r"short \u12", r"short \u12")):
            with self.subTest(body=body):
                self.assertEqual(H._toml_basic(body), want)


class HookCommandRunsInShells(TempHome):
    """The hook command moth writes, run the way harnesses run it: sh on POSIX; cmd.exe and Git Bash on Windows (Claude
    Code can use either there). The script sits in a folder with a space, as under C:\\Users\\First Last."""

    def test_runs(self):
        integ = self.tmp / "Me Q" / "integrations"
        integ.mkdir(parents=True)
        out = self.tmp / "ran.txt"
        (integ / "claude_code.py").write_text("import os, sys\nopen(os.environ['RAN'], 'a').write(sys.argv[0] + '\\n')\n")
        with unittest.mock.patch.object(H, "INTEGRATIONS", str(integ)):
            cmd = H.hook_command(sys.executable, H.script("claude_code.py"))
        shells = []
        if os.name == "nt":
            shells.append(("cmd.exe", lambda: subprocess.run(cmd, shell=True, env=env, capture_output=True, text=True)))
            git_bash = os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "Git", "bin", "bash.exe")
            if os.path.isfile(git_bash):
                shells.append(("git bash", lambda: subprocess.run([git_bash, "-c", cmd], env=env, capture_output=True,
                                                                  text=True)))
        else:
            shells.append(("sh", lambda: subprocess.run(["/bin/sh", "-c", cmd], env=env, capture_output=True, text=True)))
        env = dict(os.environ, RAN=str(out), PATH=os.environ.get("PATH", "") + os.pathsep + os.path.dirname(sys.executable))
        for name, run in shells:
            with self.subTest(shell=name):
                out.unlink(missing_ok=True)
                r = run()
                self.assertEqual(r.returncode, 0, (cmd, r.stdout, r.stderr))
                self.assertTrue(out.exists(), (cmd, r.stdout, r.stderr))
                self.assertEqual(Path(out.read_text().strip()).resolve(), (integ / "claude_code.py").resolve())


class CopiedAdapters(TempHome):
    def check(self, name: str, dst: Path):
        rc, out = self.hook(name, "--yes")
        self.assertEqual(rc, 0, out)
        self.assertEqual(dst.read_bytes(), Path(H.copy_source(name)).read_bytes())
        self.assertEqual(H.status(name)["state"], "installed")
        rc, out = self.hook(name, "--yes")
        self.assertIn("already installed", out)

        dst.write_text(dst.read_text() + "\n// an older version\n")   # e.g. after `moth update` changed the source
        self.assertEqual(H.status(name)["state"], "outdated")
        rc, out = self.run_cli(H.main, "hooks", [])
        self.assertIn("outdated", out)
        old = dst.read_text()
        rc, out = self.hook(name, "--yes")
        self.assertIn("update", out)
        self.assertEqual(dst.read_bytes(), Path(H.copy_source(name)).read_bytes())
        self.assertEqual([b.read_text() for b in self.backups(dst)], [old])

        rc, out = self.unhook(name, "--yes")
        self.assertEqual(rc, 0, out)
        self.assertFalse(dst.exists())
        self.assertEqual(len(self.backups(dst)), 1)   # an identical copy needs no backup

        dst.write_text("// someone else's moth.ts\n")   # not moth's: never overwritten or deleted
        rc, out = self.hook(name, "--yes")
        self.assertEqual(rc, 1)
        rc, out = self.unhook(name, "--yes")
        self.assertEqual(rc, 1)
        self.assertEqual(dst.read_text(), "// someone else's moth.ts\n")

    def test_pi(self):
        self.check("pi", self.home / ".pi" / "agent" / "extensions" / "moth.ts")

    def test_opencode(self):
        (self.home / ".config" / "opencode" / "plugins").mkdir(parents=True)
        (self.home / ".config" / "opencode" / "plugins" / "other.js").write_text("// other\n")
        self.check("opencode", self.home / ".config" / "opencode" / "plugins" / "moth.js")
        self.assertEqual((self.home / ".config" / "opencode" / "plugins" / "other.js").read_text(), "// other\n")


class Hermes(TempHome):
    CFG = "model: x\nplugins:\n  enabled:\n    - web/local-web\n  disabled: []\ndisplay:\n  skin: default\n"

    def setUp(self):
        super().setUp()
        self.fake_bin("hermes", FAKE_HERMES)
        self.cfg = self.write(".hermes/config.yaml", self.CFG)
        self.dst = self.home / ".hermes" / "plugins" / "moth"

    def test_enabled_reader_without_pyyaml(self):
        with unittest.mock.patch.dict(sys.modules, {"yaml": None}):   # import yaml -> ImportError
            self.assertEqual(H.yaml_enabled_plugins(self.CFG), ["web/local-web"])
            self.assertEqual(H.yaml_enabled_plugins("plugins:\n  enabled: [a, 'moth']\n"), ["a", "moth"])
            self.assertEqual(H.yaml_enabled_plugins("other:\n  enabled:\n    - moth\nplugins:\n  enabled: []\n"), [])
            self.assertEqual(H.yaml_enabled_plugins(
                "plugins:\n  enabled:\n    - x  # note\n    - moth\n  entries:\n    - nope\n"), ["x", "moth"])

    def test_roundtrip(self):
        self.assertEqual(H.yaml_enabled_plugins(self.CFG), ["web/local-web"])
        rc, out = self.hook("hermes", "--yes")
        self.assertEqual(rc, 0, out)
        self.assertTrue(H.same_tree(str(self.dst), H.copy_source("hermes")))
        self.assertIn("plugins enable --no-allow-tool-override moth", (self.bin / "hermes.log").read_text())
        self.assertEqual(H.yaml_enabled_plugins(self.cfg.read_text()), ["moth", "web/local-web"])
        self.assertEqual([b.read_text() for b in self.backups(self.cfg)], [self.CFG])
        self.assertEqual(H.status("hermes")["state"], "installed")
        rc, out = self.hook("hermes", "--yes")
        self.assertIn("already installed", out)

        (self.dst / "plugin.yaml").write_text((self.dst / "plugin.yaml").read_text() + "# old\n")
        self.assertEqual(H.status("hermes")["state"], "outdated")
        rc, out = self.hook("hermes", "--yes")
        self.assertEqual(rc, 0, out)
        self.assertTrue(H.same_tree(str(self.dst), H.copy_source("hermes")))
        self.assertEqual(sorted(os.listdir(self.dst.parent)), ["moth"])   # no backup where Hermes would load it
        self.assertEqual(len(list((self.home / ".moth" / "backups").glob("moth.bak-moth-*"))), 1)

        rc, out = self.unhook("hermes", "--yes")
        self.assertEqual(rc, 0, out)
        self.assertFalse(self.dst.exists())
        self.assertEqual(H.yaml_enabled_plugins(self.cfg.read_text()), ["web/local-web"])
        self.assertIn("display:\n  skin: default", self.cfg.read_text())

    def test_without_hermes_binary_prints_the_step(self):
        self.remove_bin("hermes")
        rc, out = self.hook("hermes", "--yes")
        self.assertEqual(rc, 0, out)
        self.assertIn("hermes plugins enable moth", out)
        self.assertEqual(self.cfg.read_text(), self.CFG)
        self.assertEqual(H.status("hermes")["state"], "partial")

    def test_foreign_plugin_folder(self):
        self.dst.mkdir(parents=True)
        (self.dst / "plugin.yaml").write_text("name: somebody-else\n")
        rc, out = self.hook("hermes", "--yes")
        self.assertEqual(rc, 1)
        self.assertEqual((self.dst / "plugin.yaml").read_text(), "name: somebody-else\n")


class Dsh(TempHome):
    OTHER = "- insert: [{ id: pet, name: '/opt/pet.mjs' }]\n"

    def test_roundtrip(self):
        p = self.write(".dsh/profiles/main/cordis.patch.yml", self.OTHER)
        rc, out = self.hook("dsh", "--yes")
        self.assertEqual(rc, 0, out)
        text = p.read_text()
        self.assertTrue(text.startswith(self.OTHER))
        self.assertIn(f"id: moth, name: '{INTEG / 'dsh' / 'moth.mjs'}'", text)
        self.assertEqual(H.status("dsh")["state"], "installed")
        before = sha(p)
        rc, out = self.hook("dsh", "--yes")
        self.assertIn("already installed", out)
        self.assertEqual(sha(p), before)
        rc, out = self.unhook("dsh", "--yes")
        self.assertEqual(p.read_text(), self.OTHER)

    def test_several_profiles_need_a_choice(self):
        self.write(".dsh/profiles/a/cordis.patch.yml", "")
        self.write(".dsh/profiles/b/cordis.patch.yml", "")
        rc, out = self.hook("dsh", "--yes")
        self.assertEqual(rc, 1)
        self.assertIn("--profile", out)
        rc, out = self.hook("dsh", "--profile", "b", "--yes")
        self.assertEqual(rc, 0, out)
        self.assertEqual((self.home / ".dsh/profiles/a/cordis.patch.yml").read_text(), "")


class HooksList(TempHome):
    def test_lists_detected_and_all_only_touches_detected(self):
        self.write(".claude/settings.json", "{}")
        (self.home / ".config" / "opencode").mkdir(parents=True)
        self.fake_bin("pi")
        rc, out = self.run_cli(H.main, "hooks", [])
        self.assertEqual(rc, 0)
        self.assertRegex(out, r"claude-code\s+not installed")
        self.assertRegex(out, r"pi\s+not installed.*`pi` on PATH")
        self.assertRegex(out, r"codex\s+not found on this machine")
        rc, out = self.hook("--all", "--dry-run")
        self.assertIn("claude-code: add moth's Stop hook", out)
        self.assertIn("opencode: copy", out)
        self.assertIn("pi: copy", out)
        self.assertNotIn("codex:", out)
        self.assertNotIn("hermes:", out)
        self.assertEqual((self.home / ".claude" / "settings.json").read_text(), "{}")
        self.assertFalse((self.home / ".pi").exists())
        rc, out = self.hook("bogus")
        self.assertEqual(rc, 2)


# --- doctor -------------------------------------------------------------------------------------------------------
class HealthServer:
    """A fake moth: /api/health with a pid; or anything else on a port."""

    def __init__(self, pid: int):
        body = json.dumps({"ok": True, "pid": pid, "install_root": "/somewhere", "version": "9.9"}).encode()

        class Hd(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200 if self.path == "/api/health" else 404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), Hd)
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


class Doctor(TempHome):
    def fake_torch(self, body: str) -> dict:
        d = self.tmp / "faketorch"
        d.mkdir(exist_ok=True)
        (d / "torch.py").write_text(body)
        return dict(os.environ, PYTHONPATH=str(d))

    def test_torch_missing_and_the_rest_still_runs(self):
        env = self.fake_torch("raise ModuleNotFoundError(\"No module named 'torch'\")\n")
        d = D.probe_torch(env=env)
        self.assertIn("No module named 'torch'", d["error"])
        r = D.run(free_port(), probe=lambda: D.probe_torch(env=env))
        text = r.render()
        self.assertRegex(text, r"✗ torch\s+not installed")
        self.assertIn("fix: pip install torch", text)
        for name in ("version", "python", "packages", "weights", "tokenizer", "token", "port", "hooks",
                     "pet (PySide6)", "display", "RAM", "disk", "show_in_harness"):
            self.assertTrue(any(n == name for _, n, *_ in r.lines), name)
        self.assertTrue(r.failed)

    def test_torch_broken(self):
        env = self.fake_torch("raise RuntimeError('CUDA driver too old')\n")
        r = D.Report()
        D.check_torch(r, lambda: D.probe_torch(env=env))
        self.assertIn("broken: RuntimeError: CUDA driver too old", r.render())

    def test_torch_present(self):
        r = D.Report()
        D.check_torch(r, lambda: {"version": "2.9.0+cu128", "cuda": True, "xpu": False})
        self.assertRegex(r.render(), r"✓ torch\s+2\.9\.0\+cu128  \(devices: cpu, cuda")
        import importlib.util
        if importlib.util.find_spec("torch") is not None and D.probe_torch().get("version"):   # the real one
            r = D.Report()
            D.check_torch(r)
            self.assertEqual(r.lines[0][0], "✓")

    def test_port_states(self):
        port = free_port()
        r = D.Report()
        D.check_port(r, port)
        self.assertEqual(r.lines[-1][0], "·")
        with socket.socket() as s:   # something else holds the port
            s.bind(("127.0.0.1", 0))
            s.listen()
            D.check_port(r, s.getsockname()[1])
        self.assertEqual(r.lines[-1][0], "✗")
        self.assertIn("another program", r.lines[-1][2])
        fake = HealthServer(pid=4242)
        try:
            D.check_port(r, fake.port)
        finally:
            fake.close()
        self.assertEqual(r.lines[-1][0], "✓")
        self.assertIn("moth 9.9 is running (pid 4242", r.lines[-1][2])

    def test_display_ram_token(self):
        r = D.Report()
        D.check_display(r, env={"WAYLAND_DISPLAY": "wayland-0", "DISPLAY": ":0"}, platform="linux")
        D.check_display(r, env={"WAYLAND_DISPLAY": "wayland-0"}, platform="linux")
        D.check_display(r, env={}, platform="linux")
        D.check_display(r, env={}, platform="darwin")
        self.assertEqual([m for m, *_ in r.lines], ["✓", "!", "!", "✓"])
        D.check_ram(r, gb=2.0)
        self.assertEqual(r.lines[-1][0], "!")
        D.check_ram(r, gb=8.0)
        self.assertEqual(r.lines[-1][0], "✓")
        D.check_token(r)
        self.assertEqual(r.lines[-1][0], "!")   # not made yet
        self.write(".moth/token", "abc\n").chmod(0o600)
        D.check_token(r)
        self.assertEqual(r.lines[-1][0], "✓")

    def test_never_imports_serve_and_cli(self):
        sys.modules.pop("moth.serve", None)
        rc, out = self.run_cli(D.main, ["--port", str(free_port())])
        self.assertIn("python", out)
        self.assertIn("show_in_harness", out)
        self.assertNotIn("moth.serve", sys.modules)
        self.assertLess(len(out.splitlines()), 40)   # one screen


# --- autostart ----------------------------------------------------------------------------------------------------
class Autostart(TempHome):
    def cycle(self, platform: str, path: Path):
        self.assertEqual(Path(A.entry_path(platform)), path)
        rc, out = self.run_cli(A.main, ["status"], platform)
        self.assertIn("autostart is off", out)
        rc, out = self.run_cli(A.main, ["on"], platform)
        self.assertEqual(rc, 0, out)
        self.assertTrue(path.is_file())
        self.assertEqual(A.state(platform)[0], "on")
        before = sha(path)
        rc, out = self.run_cli(A.main, ["on"], platform)
        self.assertIn("already on", out)
        self.assertEqual(sha(path), before)
        rc, out = self.run_cli(A.main, ["status"], platform)
        self.assertIn("autostart is on", out)
        return path

    def off(self, platform: str, path: Path):
        rc, out = self.run_cli(A.main, ["off"], platform)
        self.assertEqual(rc, 0, out)
        self.assertFalse(path.exists())
        rc, out = self.run_cli(A.main, ["off"], platform)
        self.assertIn("already off", out)

    def test_linux_desktop_entry(self):
        p = self.cycle("linux", self.home / ".config" / "autostart" / "moth-pet.desktop")
        text = p.read_text()
        self.assertIn("[Desktop Entry]", text)
        exec_line = next(x for x in text.splitlines() if x.startswith("Exec="))
        py = A._desktop_quote(sys.executable)   # as Exec= spells it (a Windows python here: backslashes escaped)
        self.assertIn(py, exec_line)
        self.assertTrue(exec_line.endswith("-m moth.pet"))
        self.assertIn(f"Path={A.CODE}", text)
        p.write_text(text.replace(py, "/old/python"))   # another interpreter: stale, `on` fixes it
        self.assertEqual(A.state("linux")[0], "stale")
        self.run_cli(A.main, ["on"], "linux")
        self.assertEqual(A.state("linux")[0], "on")
        self.off("linux", p)

    def test_desktop_quoting(self):
        self.assertEqual(A._desktop_quote("/usr/bin/python3"), "/usr/bin/python3")
        self.assertEqual(A._desktop_quote("/my dir/py$"), '"/my dir/py\\$"')
        self.assertEqual(A._desktop_quote("50%"), "50%%")

    def test_macos_launch_agent(self):
        p = self.cycle("darwin", self.home / "Library" / "LaunchAgents" / "local.moth.pet.plist")
        d = plistlib.loads(p.read_bytes())
        self.assertEqual(d["ProgramArguments"], [sys.executable, "-m", "moth.pet"])
        self.assertTrue(d["RunAtLoad"])
        self.assertEqual(d["WorkingDirectory"], A.CODE)
        self.off("darwin", p)

    def test_windows_startup_bat(self):
        p = self.cycle("win32", self.home / "AppData" / "Roaming" / "Microsoft" / "Windows" / "Start Menu" /
                       "Programs" / "Startup" / "moth-pet.bat")
        text = p.read_text()
        self.assertIn("-m moth.pet", text)
        self.assertIn(f'cd /d "{A.CODE}"', text)
        self.off("win32", p)

    def test_foreign_entry_is_left_alone(self):
        p = self.write(".config/autostart/moth-pet.desktop", "[Desktop Entry]\nExec=something-else\n")
        rc, out = self.run_cli(A.main, ["on"], "linux")
        self.assertEqual(rc, 1)
        rc, out = self.run_cli(A.main, ["off"], "linux")
        self.assertEqual(rc, 1)
        self.assertEqual(p.read_text(), "[Desktop Entry]\nExec=something-else\n")


# --- forget -------------------------------------------------------------------------------------------------------
class Forget(TempHome):
    def setUp(self):
        super().setUp()
        self.mode = unittest.mock.patch("moth.weights.mode", return_value="pip")   # never the checkout's live/
        self.mode.start()
        self.port = free_port()
        self.token = self.write(".moth/token", "tok\n")
        self.config = self.write(".moth/config.json", "{}\n")
        self.log = self.write(".moth/pet.log", "pet says hi\n")
        self.mhome = self.home / ".moth" / "home"
        self.write(".moth/home/diary/tuesday.txt", "ate a crumb\n")
        self.write(".moth/home/saved/name.txt", "the human is called sam\n")

    def tearDown(self):
        self.mode.stop()
        super().tearDown()

    def forget(self, *args):
        return self.run_cli(F.main, ["--port", str(self.port), "--home", str(self.mhome), *args])

    def test_refuses_while_running(self):
        fake = HealthServer(pid=os.getpid())   # a live pid, its pidfile, /api/health agreeing
        self.write(f".moth/serve-{fake.port}.pid", json.dumps({"pid": os.getpid(), "port": fake.port}))
        (self.mhome / ".moth-home").write_text("moth made this\n")
        try:
            rc, out = self.forget("--all", "--token", "--yes")
        finally:
            fake.close()
        self.assertEqual(rc, 1)
        self.assertIn("moth is running", out)
        self.assertIn("stop it first", out)
        for p in (self.token, self.config, self.log, self.mhome / "diary" / "tuesday.txt"):
            self.assertTrue(p.exists(), p)

    def test_refuses_when_health_answers_without_pidfile(self):
        fake = HealthServer(pid=1)
        try:
            rc, out = self.run_cli(F.main, ["--port", str(fake.port), "--home", str(self.mhome), "--yes"])
        finally:
            fake.close()
        self.assertEqual(rc, 1)
        self.assertTrue(self.log.exists())

    def test_logs_only_by_default(self):
        stale = self.write(".moth/serve-1.pid", json.dumps({"pid": 2**22 + 12345, "port": 1}))
        rc, out = self.forget("--yes")
        self.assertEqual(rc, 0, out)
        self.assertFalse(self.log.exists())
        self.assertFalse(stale.exists())
        self.assertTrue(self.mhome.exists())
        self.assertTrue(self.token.exists() and self.config.exists())
        self.assertIn("nothing on disk", out)

    def test_all_needs_the_marker(self):
        rc, out = self.forget("--all", "--yes")
        self.assertEqual(rc, 1)
        self.assertIn("won't delete it", out)
        self.assertIn("Nothing was deleted", out)
        self.assertTrue((self.mhome / "diary" / "tuesday.txt").exists())
        self.assertTrue(self.log.exists())
        (self.mhome / ".moth-home").symlink_to(self.token)   # a symlinked marker doesn't count either
        rc, out = self.run_cli(F.main, ["--port", str(self.port), "--home", str(self.mhome), "--all", "--yes"])
        self.assertEqual(rc, 1)
        self.assertTrue(self.mhome.exists())

    def test_all_deletes_marked_home_keeps_token(self):
        (self.mhome / ".moth-home").write_text("moth made this\n")
        rc, out = self.forget("--all")   # stdin closed: cancelled
        self.assertEqual(rc, 1)
        self.assertTrue(self.mhome.exists())
        rc, out = self.forget("--all", "--yes")
        self.assertEqual(rc, 0, out)
        self.assertFalse(self.mhome.exists())
        self.assertFalse(self.log.exists())
        self.assertTrue(self.token.exists() and self.config.exists())

    def test_all_token(self):
        (self.mhome / ".moth-home").write_text("moth made this\n")
        backups = self.write(".moth/backups/settings.json.bak-moth-1", "{}")
        rc, out = self.forget("--token", "--yes")
        self.assertEqual(rc, 2)
        rc, out = self.forget("--all", "--token", "--yes")
        self.assertEqual(rc, 0, out)
        self.assertFalse(self.token.exists() or self.config.exists() or self.mhome.exists())
        self.assertTrue(backups.exists())   # hook backups of the user's configs are never moth's to delete


class Main(unittest.TestCase):
    def test_help_lists_new_commands(self):
        from moth import __main__ as M
        for word in ("moth hooks", "moth hook <harness", "moth unhook", "moth doctor", "moth autostart on|off|status",
                     "moth forget"):
            self.assertIn(word, M.USAGE)

    def test_dispatch_never_reaches_serve(self):
        from moth import __main__ as M
        with unittest.mock.patch("moth.hooks.main", return_value=0) as hm, \
                unittest.mock.patch("moth.doctor.main", return_value=0) as dm, \
                unittest.mock.patch("moth.forget.main", return_value=0) as fm, \
                unittest.mock.patch("moth.autostart.main", return_value=0) as am:
            for argv in (["hook", "--help"], ["hooks"], ["unhook", "pi"], ["doctor"], ["forget", "--all"],
                         ["autostart", "status"]):
                with self.assertRaises(SystemExit) as cm:
                    M.main(argv)
                self.assertEqual(cm.exception.code, 0)
        hm.assert_any_call("hook", ["--help"])
        hm.assert_any_call("unhook", ["pi"])
        fm.assert_called_once_with(["--all"])
        am.assert_called_once_with(["status"])
        dm.assert_called_once_with([])
        self.assertNotIn("moth.serve", sys.modules)


if __name__ == "__main__":
    unittest.main()
