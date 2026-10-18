"""Smoke-test an installed moth app (CI, .github/workflows/installers.yml; locally against a dist/ folder). Stdlib only.

  python packaging/smoke.py --app EXE [--cli EXE] --model DIR --out DIR [--port 4791] [--uninstall CMD]

  --app        what the Start menu / Dock / desktop entry opens (mothw.exe, moth.app/Contents/MacOS/moth, the
               AppImage, dist/moth/moth)
  --cli        the console program for `moth ...` commands (moth.exe on Windows; default --app)
  --model      a weights folder for MOTH_MODEL_DIR (packaging/make_fixture.py: a tiny random model)
  --uninstall  Windows: the uninstaller (unins000.exe), run silently at the end; elsewhere `moth unhook --all` and
               `moth autostart off` are what uninstalling means (the app is then deleted by the workflow)

Everything runs in a scratch home (HOME / USERPROFILE / APPDATA / XDG_*; Windows: a path with a space) with ~/.claude
and ~/.codex in it, so Claude Code and Codex count as found, and moth on --port, never 4747:
  1 the CLI answers (--version), the bundled huggingface_hub reaches the Hub (a missing repo fails with moth's message)
  2 first-run window, download pressed with no weights published: the friendly message (screenshot firstrun-download)
  3 first-run window with MOTH_MODEL_DIR (screenshot firstrun), Start pressed: setup.py hooks Claude Code + Codex,
    turns autostart on and starts the pet, which wakes the real frozen server (torch) -> /api/health
  4 the written hook commands and autostart entry run the bundled executable, not a Python
  5 the Claude Code hook as written, through bash (Git Bash on Windows), and the Codex one through PowerShell on
    Windows: the server saw both sessions; timings (cold first run, then the median of 10)
  6 screenshot of the pet with its reaction (pet.png)
  7 uninstall / unhook: the hooks and the autostart entry are gone
Writes OUT/smoke.json and OUT/*.png; exit 1 when a check failed.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

WIN, MAC = sys.platform == "win32", sys.platform == "darwin"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
PS_SHOT = r"""
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
$b = [System.Windows.Forms.SystemInformation]::VirtualScreen
$bmp = New-Object System.Drawing.Bitmap $b.Width, $b.Height
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.CopyFromScreen($b.Left, $b.Top, 0, 0, $bmp.Size)
$bmp.Save($env:SHOT_PATH, [System.Drawing.Imaging.ImageFormat]::Png)
"""
QT_SHOT = ("import sys; from PySide6.QtGui import QGuiApplication; app = QGuiApplication(sys.argv); "
           "s = app.primaryScreen(); print(s.geometry()); s.grabWindow(0).save(sys.argv[1])")
STOP = {"last_assistant_message": "Fixed the login bug. All tests pass.", "hook_event_name": "Stop"}


class Smoke:
    def __init__(self, a):
        self.a = a
        self.out = Path(a.out).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.url = f"http://127.0.0.1:{a.port}"
        self.home = Path(tempfile.mkdtemp(prefix="moth smoke home " if WIN else "moth-smoke-home-"))
        for d in (".claude", ".codex"):
            (self.home / d).mkdir()
        env = {k: v for k, v in os.environ.items() if not k.startswith("MOTH_") and k != "QT_QPA_PLATFORM"}
        env.update(HOME=str(self.home), XDG_CONFIG_HOME=str(self.home / ".config"),
                   XDG_CACHE_HOME=str(self.home / ".cache"), MOTH_URL=self.url, PYTHONIOENCODING="utf-8")
        if WIN:
            env.update(USERPROFILE=str(self.home), APPDATA=str(self.home / "AppData" / "Roaming"))
        elif not MAC:   # the virtual X screen (xvfb-run), never a Wayland socket found by default (wayland-0)
            env.pop("WAYLAND_DISPLAY", None)
            env["QT_QPA_PLATFORM"] = "xcb"
        self.env = env
        self.results: dict = {"os": sys.platform, "checks": {}, "timings_ms": {}, "notes": []}
        self.procs: list = []

    # --- helpers -----------------------------------------------------------------------------------------------
    def check(self, name: str, ok: bool, detail=""):
        self.results["checks"][name] = {"ok": bool(ok), "detail": str(detail)[:2000]}
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {str(detail)[:600]}", flush=True)
        return ok

    def note(self, text: str):
        self.results["notes"].append(text)
        print(f"  note: {text}", flush=True)

    def run(self, args, timeout=120, env=None, **kw):
        t = time.perf_counter()
        p = subprocess.run(args, env=env or self.env, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout, **kw)
        return p, (time.perf_counter() - t) * 1000

    def launch(self, args, log: str, **extra_env):
        f = open(self.out / log, "w")
        kw = {}
        if not WIN:
            kw["start_new_session"] = True
        p = subprocess.Popen(args, env={**self.env, **extra_env}, stdout=f, stderr=subprocess.STDOUT, **kw)
        self.procs.append(p)
        return p

    def shot(self, name: str) -> None:
        path = self.out / name
        try:
            if WIN:
                r, _ = self.run(["powershell", "-NoProfile", "-Command", PS_SHOT], env={**self.env, "SHOT_PATH": str(path)})
            elif MAC:
                r, _ = self.run(["screencapture", "-x", str(path)])
            else:
                r, _ = self.run([self.a.cli, "-c", QT_SHOT, str(path)], env={**self.env, "QT_QPA_PLATFORM": "xcb"})
            print(f"  screenshot {name}: exit {r.returncode} {r.stdout.strip()[:200]} {r.stderr.strip()[-300:]}",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  screenshot {name} failed: {e}", flush=True)

    def health(self):
        try:
            with OPENER.open(self.url + "/api/health", timeout=2) as r:
                return json.loads(r.read())
        except Exception:  # noqa: BLE001
            return None

    def token(self) -> str:
        try:
            return (self.home / ".moth" / "token").read_text().strip()
        except OSError:
            return ""

    def agents(self) -> list:
        req = urllib.request.Request(self.url + "/api/log?since=0", headers={"X-Moth-Token": self.token()})
        with OPENER.open(req, timeout=5) as r:
            return (json.loads(r.read()).get("agents") or {}).get("sessions") or []

    def tail(self, path: Path, n=25) -> str:
        try:
            return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
        except OSError:
            return "(no file)"

    # --- steps -------------------------------------------------------------------------------------------------
    def cli(self):
        p, ms = self.run([self.a.cli, "--version"])
        self.check("cli --version", p.returncode == 0 and "moth" in p.stdout, f"{p.stdout.strip()} ({ms:.0f} ms)")
        code = "from moth import weights; weights.ensure(None, 'darioooooo0o/moth-installer-smoke-missing')"
        p, ms = self.run([self.a.cli, "-c", code], timeout=180)
        msg = (p.stdout + p.stderr).strip()
        self.check("bundled huggingface_hub: a missing repo fails with moth's message",
                   p.returncode != 0 and "can't get darioooooo0o/moth-installer-smoke-missing" in msg, msg[-400:])

    def firstrun_download(self):
        p = self.launch([self.a.app], "firstrun-download.log", MOTH_APP_SMOKE="download",
                        MOTH_REPO="darioooooo0o/moth-installer-smoke-missing")
        time.sleep(25)
        self.shot("firstrun-download.png")
        alive = p.poll() is None
        self.stop(p)
        self.check("first-run window: download without published weights keeps the window up", alive,
                   self.tail(self.out / "firstrun-download.log", 8))
        state = self.home / ".moth" / "app.json"
        if state.exists():
            state.unlink()

    def firstrun_start(self) -> bool:
        p = self.launch([self.a.app], "firstrun.log", MOTH_APP_SMOKE="start", MOTH_MODEL_DIR=str(Path(self.a.model)))
        time.sleep(5)
        self.shot("firstrun.png")
        t0, h = time.monotonic(), None
        while time.monotonic() - t0 < 300:
            h = self.health()
            if h:
                break
            time.sleep(1)
        ok = self.check("first-run Start -> the pet woke the frozen server: /api/health", bool(h),
                        f"{h} after {time.monotonic() - t0:.0f} s" if h else self.tail(self.home / ".moth" / "pet.log"))
        if ok:
            self.results["timings_ms"]["firstrun_start_to_health"] = round((time.monotonic() - t0) * 1000)
        try:
            p.wait(30)
            self.check("first-run window closed itself after moth woke up", p.returncode == 0, f"exit {p.returncode}")
        except subprocess.TimeoutExpired:
            self.check("first-run window closed itself after moth woke up", False, "still open after 30 s")
            self.stop(p)
        st = {}
        try:
            st = json.loads((self.home / ".moth" / "app.json").read_text())
        except (OSError, ValueError):
            pass
        self.check("first run recorded (~/.moth/app.json)", st.get("first_run_done") is True, st)
        return ok

    def hook_commands(self) -> dict:
        cmds = {}
        for name, path in (("claude-code", self.home / ".claude" / "settings.json"),
                           ("codex", self.home / ".codex" / "hooks.json")):
            try:
                d = json.loads(path.read_text())
                cmd = d["hooks"]["Stop"][-1]["hooks"][0]["command"]
            except (OSError, ValueError, KeyError, IndexError, TypeError) as e:
                self.check(f"{name} hook written", False, f"{path}: {e}")
                continue
            ok = "python" not in cmd.lower() and "hook-run" in cmd and ("\\" not in cmd)
            self.check(f"{name} hook runs the bundled executable (no python, forward slashes)", ok, cmd)
            cmds[name] = cmd
        entry = {"win32": self.home / "AppData" / "Roaming" / "Microsoft" / "Windows" / "Start Menu" / "Programs"
                 / "Startup" / "moth-pet.bat",
                 "darwin": self.home / "Library" / "LaunchAgents" / "local.moth.pet.plist"}.get(
            sys.platform, self.home / ".config" / "autostart" / "moth-pet.desktop")
        try:
            text = entry.read_text()
            want = "mothw.exe" if WIN else (os.environ.get("SMOKE_APPIMAGE") or "moth")
            self.check("autostart entry starts the bundled app (no python)",
                       "python" not in text.lower() and want in text, text)
        except OSError as e:
            self.check("autostart entry written", False, e)
        return cmds

    def time_cmd(self, label: str, argv: list, stdin: str, n: int = 10):
        runs = []
        for i in range(n + 1):
            p, ms = self.run(argv, input=stdin, timeout=60)
            if p.returncode != 0:
                self.check(f"{label} exits 0", False, (p.stdout + p.stderr)[-400:])
                return
            runs.append(ms)
        cold, warm = runs[0], statistics.median(runs[1:])
        self.results["timings_ms"][label] = {"cold": round(cold), "median_warm": round(warm),
                                             "max_warm": round(max(runs[1:]))}
        print(f"  {label}: cold {cold:.0f} ms, warm median {warm:.0f} ms (max {max(runs[1:]):.0f})", flush=True)

    def hooks(self, cmds: dict):
        if "claude-code" in cmds:
            bash = shutil.which("bash")
            if WIN:
                git = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git" / "bin" / "bash.exe"
                bash = str(git) if git.exists() else bash
            stop = json.dumps({**STOP, "session_id": "smoke-claude", "cwd": "/work/smoke-claude"})
            self.time_cmd("claude-code hook via bash (as written)", [bash, "-c", cmds["claude-code"]], stop)
        if "codex" in cmds:
            stop = json.dumps({**STOP, "session_id": "smoke-codex", "cwd": "/work/smoke-codex"})
            shell = (["powershell", "-NoProfile", "-Command"] if WIN else ["sh", "-c"])
            self.time_cmd("codex hook via " + ("powershell" if WIN else "sh") + " (as written)",
                          shell + [cmds["codex"]], stop)
        raw = self.raw_runner()
        if raw:
            stop = json.dumps({**STOP, "session_id": "smoke-raw", "cwd": "/work/smoke-raw"})
            self.time_cmd("moth-hook start + post (direct, no shell)", raw + ["hook-run", "claude_code.py"], stop)
            self.time_cmd("big moth executable hook-run (for comparison)", [self.a.cli, "hook-run", "claude_code.py"],
                          stop, n=5)
        time.sleep(1.5)
        try:
            labels = {s.get("label") for s in self.agents()}
        except Exception as e:  # noqa: BLE001
            labels = set()
            self.note(f"/api/log failed: {e}")
        for name in cmds:
            want = "smoke-claude" if name == "claude-code" else "smoke-codex"
            self.check(f"{name} hook reached the server", want in labels, sorted(x for x in labels if x))

    def raw_runner(self):
        if os.environ.get("SMOKE_APPIMAGE"):
            return [os.environ["SMOKE_APPIMAGE"]]
        d = Path(self.a.cli).parent
        r = d / ("moth-hook.exe" if WIN else "moth-hook")
        return [str(r)] if r.exists() else None

    def uninstall(self):
        if WIN and self.a.uninstall:
            self.kill_all()
            p, ms = self.run([self.a.uninstall, "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART"], timeout=300)
            gone = Path(self.a.cli)
            t0 = time.monotonic()
            while gone.exists() and time.monotonic() - t0 < 180:   # the uninstaller re-runs itself from %TEMP%
                time.sleep(1)
            self.check("uninstaller removed the app", not gone.exists(), f"exit {p.returncode}")
        else:
            for args in (["unhook", "--all", "--yes"], ["autostart", "off"]):
                p, _ = self.run([self.a.cli] + args)
                print(f"  moth {' '.join(args)}: exit {p.returncode}\n{p.stdout[-800:]}", flush=True)
            self.kill_all()
        for name, path in (("claude-code", self.home / ".claude" / "settings.json"),
                           ("codex", self.home / ".codex" / "hooks.json")):
            text = path.read_text() if path.exists() else ""
            self.check(f"{name} hook removed", "hook-run" not in text and "integrations" not in text, text[:400])
        left = [str(p) for p in (self.home / "AppData", self.home / "Library" / "LaunchAgents",
                                 self.home / ".config" / "autostart") if p.exists() for p in p.rglob("*moth-pet*")]
        left += [str(p) for p in (self.home / "Library" / "LaunchAgents").glob("local.moth.pet.plist")]
        self.check("autostart entry removed", not left, left)

    # --- processes ---------------------------------------------------------------------------------------------
    def stop(self, p):
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                p.kill()

    def kill_all(self):
        """The pet setup started (detached) and the server it started: by executable, never by command line."""
        for p in self.procs:
            self.stop(p)
        if WIN:
            for exe in ("mothw.exe", "moth.exe"):
                subprocess.run(["taskkill", "/F", "/T", "/IM", exe], capture_output=True)
            time.sleep(2)
            return
        roots = [str(Path(self.a.app).resolve().parent), str(Path(self.a.cli).resolve().parent)]
        if os.environ.get("SMOKE_APPIMAGE"):
            roots += [os.environ["SMOKE_APPIMAGE"], "/tmp/.mount_moth"]
        mine = {os.getpid(), os.getppid()}
        pids = []
        if MAC:
            out = subprocess.run(["ps", "-axo", "pid=,comm="], capture_output=True, text=True).stdout
            for line in out.splitlines():
                pid, _, comm = line.strip().partition(" ")
                if pid.isdigit() and any(comm.strip().startswith(r) for r in roots):
                    pids.append(int(pid))
        else:
            for d in Path("/proc").iterdir():
                if d.name.isdigit():
                    try:
                        exe = os.readlink(d / "exe")
                    except OSError:
                        continue
                    if any(exe.startswith(r) for r in roots):
                        pids.append(int(d.name))
        for pid in set(pids) - mine:
            try:
                os.kill(pid, 15)
            except OSError:
                pass
        time.sleep(3)

    def main(self) -> int:
        try:
            self.cli()
            self.firstrun_download()
            if self.firstrun_start():
                cmds = self.hook_commands()
                self.hooks(cmds)
                time.sleep(5)
                self.shot("pet.png")
            self.uninstall()
        finally:
            self.kill_all()
            for f in ("pet.log", "serve.log", "app.log"):
                src = self.home / ".moth" / f
                if src.exists():
                    shutil.copyfile(src, self.out / f)
            (self.out / "smoke.json").write_text(json.dumps(self.results, indent=1))
            shutil.rmtree(self.home, ignore_errors=True)
        failed = [k for k, v in self.results["checks"].items() if not v["ok"]]
        print(f"\n{len(self.results['checks']) - len(failed)} passed, {len(failed)} failed: {failed}", flush=True)
        print(json.dumps(self.results["timings_ms"], indent=1), flush=True)
        return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", required=True)
    ap.add_argument("--cli", default=None)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=4791)
    ap.add_argument("--uninstall", default=None)
    a = ap.parse_args()
    a.cli = a.cli or a.app
    return Smoke(a).main()


if __name__ == "__main__":
    sys.exit(main())
