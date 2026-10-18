"""The installers' app (PyInstaller, sys.frozen): hooks and autostart run the bundled executables, not a Python.

moth/hooks.py frozen_exe / hook_argv: `moth hook` writes `<moth-hook> hook-run <integrations/x.py>` (an AppImage:
`<the .AppImage> hook-run integrations/x.py`), and status / unhook still recognise it as this install's hook.
moth/autostart.py pet_python: mothw.exe on Windows, the .AppImage file itself. Stdlib only; no real freezing.
"""
from __future__ import annotations

import json
import os
import shlex
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from moth import autostart, hooks  # noqa: E402


class Frozen(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="moth frozen "))
        self.app = self.tmp / "app"
        self.app.mkdir()
        for f in ("moth", "moth-hook", "moth.exe", "mothw.exe", "moth-hook.exe"):
            (self.app / f).write_text("")
        self.home = self.tmp / "home"
        (self.home / ".claude").mkdir(parents=True)
        env = {k: v for k, v in os.environ.items() if k not in ("APPIMAGE", "APPDIR", "CLAUDE_CONFIG_DIR", "CODEX_HOME")}
        env.update(HOME=str(self.home), USERPROFILE=str(self.home))
        self.patches = [mock.patch.dict(os.environ, env, clear=True),
                        mock.patch.object(sys, "frozen", True, create=True),
                        mock.patch.object(sys, "executable", str(self.app / "moth"))]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_not_frozen_keeps_python(self):
        with mock.patch.object(sys, "frozen", False):
            self.assertEqual(hooks.hook_argv("claude_code.py"), [sys.executable, hooks.script("claude_code.py")])

    def test_command_runs_moth_hook(self):
        cmd = hooks.harness_command("claude_code.py")
        if os.name == "nt":
            self.skipTest("POSIX quoting")
        self.assertEqual(shlex.split(cmd), [str(self.app / "moth-hook"), "hook-run", hooks.script("claude_code.py")])
        self.assertNotIn("python", cmd.lower())
        self.assertEqual(hooks.moth_script_in(cmd, "claude_code.py"), hooks.script("claude_code.py"))
        self.assertEqual(hooks._where(hooks.script("claude_code.py"), "claude_code.py"), "this")

    def test_py_argument_wins(self):
        self.assertEqual(hooks.hook_argv("codex.py", "/usr/bin/python3"), ["/usr/bin/python3", hooks.script("codex.py")])

    def test_hook_status_unhook_cycle(self):
        plan = hooks.plan_hook("claude-code")
        self.assertFalse(plan.error)
        self.assertEqual(hooks.run_plans([plan], True, False, "install"), 0)
        settings = json.loads((self.home / ".claude" / "settings.json").read_text())
        cmd = settings["hooks"]["Stop"][0]["hooks"][0]["command"]
        self.assertIn("hook-run", cmd)
        self.assertEqual(hooks.status("claude-code")["state"], "installed")
        self.assertTrue(hooks.plan_hook("claude-code").nothing)   # installing twice changes nothing
        self.assertEqual(hooks.run_plans([hooks.plan_unhook("claude-code")], True, False, "remove"), 0)
        self.assertEqual(hooks.status("claude-code")["state"], "absent")

    def test_python_hook_from_another_install_is_rewritten(self):
        other = self.tmp / "old" / "moth" / "integrations"
        other.mkdir(parents=True)
        (other / "claude_code.py").write_text("")
        old = {"hooks": {"Stop": [{"hooks": [hooks.hook_entry(hooks.hook_command("python3", str(other / "claude_code.py")))]}]}}
        (self.home / ".claude" / "settings.json").write_text(json.dumps(old))
        self.assertEqual(hooks.status("claude-code")["state"], "other")
        hooks.run_plans([hooks.plan_hook("claude-code")], True, False, "install")
        self.assertEqual(hooks.status("claude-code")["state"], "installed")
        self.assertIn("moth-hook", (self.home / ".claude" / "settings.json").read_text())

    def test_another_appimage_is_not_moth(self):
        """APPIMAGE inherited from another AppImage (an editor's terminal): the tarball's moth-hook still."""
        other = {"APPIMAGE": str(self.tmp / "Editor.AppImage"), "APPDIR": str(self.tmp / ".mount_Editor")}
        with mock.patch.dict(os.environ, other):
            runner = "moth-hook.exe" if hooks._WIN else "moth-hook"
            self.assertEqual(hooks.hook_argv("claude_code.py")[0], str(self.app / runner))
            self.assertEqual(hooks._where("integrations/claude_code.py", "claude_code.py"), "missing")
            with mock.patch.object(autostart, "FROZEN", True):
                self.assertEqual(autostart.pet_python("linux"), str(self.app / "moth"))

    def test_appimage(self):
        image = str(self.tmp / "Apps" / "moth.AppImage")
        with mock.patch.dict(os.environ, {"APPIMAGE": image, "APPDIR": str(self.tmp)}):
            self.assertEqual(hooks.hook_argv("claude_code.py"), [image, "hook-run", "integrations/claude_code.py"])
            self.assertEqual(hooks._where("integrations/claude_code.py", "claude_code.py"), "this")
            hooks.run_plans([hooks.plan_hook("claude-code")], True, False, "install")
            self.assertEqual(hooks.status("claude-code")["state"], "installed")
            with mock.patch.object(autostart, "FROZEN", True):
                self.assertEqual(autostart.pet_python("linux"), image)
        with mock.patch.object(sys, "frozen", False):   # a Python install doesn't claim an AppImage's relative path
            self.assertEqual(hooks._where("integrations/claude_code.py", "claude_code.py"), "missing")

    def test_windows_codex_gets_call_operator(self):
        exe = "C:\\Program Files\\moth\\moth-hook.exe"
        with mock.patch.object(hooks, "_WIN", True), mock.patch.object(hooks, "frozen_exe", lambda: exe):
            cmd = hooks.harness_command("codex.py")
            self.assertTrue(cmd.startswith('& "C:/Program Files/moth/moth-hook.exe" hook-run '), cmd)
            self.assertNotIn("\\", cmd)
            claude = hooks.harness_command("claude_code.py")
            self.assertTrue(claude.startswith('"C:/Program Files/moth/moth-hook.exe" hook-run '), claude)

    def test_autostart_windows_uses_mothw(self):
        with mock.patch.object(autostart, "FROZEN", True), \
                mock.patch.object(sys, "executable", str(self.app / "moth.exe")):
            self.assertEqual(autostart.pet_python("win32"), str(self.app / "mothw.exe"))
        with mock.patch.object(autostart, "FROZEN", False), \
                mock.patch.object(sys, "executable", str(self.app / "moth.exe")):
            self.assertEqual(autostart.pet_python("win32"), str(self.app / "moth.exe"))   # no pythonw.exe there


if __name__ == "__main__":
    unittest.main()
