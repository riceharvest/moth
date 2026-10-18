"""moth setup : from install to a running moth, in one go.

  moth setup [--yes] [--dry-run] [--url http://127.0.0.1:4747] [--timeout 300]

  1 packages   python, torch, tokenizers / safetensors / numpy, and PySide6 for the desktop pet. Anything missing:
               the exact pip command, run only when you say yes at the prompt (never by --yes, never with sudo).
  2 weights    pip install: offers to download them (about 860 MB, into ~/.cache/moth/<repo>). Hugging Face download:
               checks they're complete (`moth update` or `hf download` finish a partial one).
  3 memory     moth needs about 1 GB; the pet won't wake it with less than ~3 GB free.
  4 hooks      the coding agents found here and whether moth is hooked in; one question per harness, installed
               by moth/hooks.py (backs up and merges configs). Then show_in_harness (Pi / OpenCode, default off).
  5 autostart  start the pet at login (moth/autostart.py).
  6 the pet    start it now, detached, and wait until moth answers on /api/health. Logs: ~/.moth/pet.log and the
               server's log (~/.moth/serve.log).

Each step says what's already done and only offers what's missing, so running it again is safe. --yes takes every
default; --dry-run only prints. Without a terminal to ask in (CI, a pipe) it behaves like --dry-run unless --yes is
given. Exit 0 when moth can run here and nothing failed (a declined hook or autostart is fine), 1 otherwise; a dry run
exits 0. Stdlib only at import (torch and PySide6 are probed in child processes).
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import time

from moth import autostart, config, doctor, hooks, launch, weights

DEFAULT_URL = "http://127.0.0.1:4747"
MIN_RUN_GB = 1.0                       # moth itself, once loaded (~0.8 GB)
PET_LOG = ("~", ".moth", "pet.log")    # same log as the macOS autostart entry (moth forget wipes ~/.moth/*.log)
PIDFILE = ("~", ".moth", "setup-pet.pid")
TORCH_INDEX = "https://download.pytorch.org/whl/cpu"
PYSIDE_PROBE = "import PySide6.QtWidgets, PySide6.QtSvg\n"
FIXABLE = ("absent", "outdated", "other", "broken", "partial")   # hook states `moth hook` fixes


def prog() -> str:
    return "moth" if os.path.basename(sys.argv[0]) in ("moth", "moth.exe") else "python -m moth"


def _path(parts) -> str:
    return os.path.join(os.path.expanduser(parts[0]), *parts[1:])


def _tty() -> bool:
    try:
        return bool(sys.stdin and not sys.stdin.closed and sys.stdin.isatty())
    except (ValueError, OSError):
        return False


# --- probes (module level, so the tests can replace them) --------------------------------------------------------
def probe_torch() -> dict:
    return doctor.probe_torch()


def probe_pyside() -> str | None:
    """None when the pet's Qt modules import (in a child process: a broken Qt can't take setup down), else why not."""
    if importlib.util.find_spec("PySide6") is None:
        return "not installed"
    try:
        p = subprocess.run([sys.executable, "-c", PYSIDE_PROBE], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as e:
        return str(e)
    if p.returncode == 0:
        return None
    err = [x for x in p.stderr.splitlines() if x.strip()]
    return f"installed but won't load: {err[-1][:140] if err else 'exit ' + str(p.returncode)}"


def ram_gb() -> float | None:
    return doctor.available_ram_gb()


def health(url: str, timeout: float = 2.0):
    return launch.probe(url, timeout)


def fetch_weights(folder: str) -> str:
    return weights.ensure(folder)


def pet_command(url: str) -> list[str]:
    return [autostart.pet_python(), "-m", "moth.pet", "--url", url]


def launch_pet(url: str, log: str) -> subprocess.Popen:
    """The pet, detached (its own session / process group): setup returns while it keeps flying. Output to log."""
    os.makedirs(os.path.dirname(log), exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env["PYTHONPATH"] = autostart.CODE + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    kw: dict = {"cwd": autostart.CODE, "env": env, "stdin": subprocess.DEVNULL, "stderr": subprocess.STDOUT,
                "close_fds": True}
    if os.name == "nt":
        kw["creationflags"] = (getattr(subprocess, "DETACHED_PROCESS", 0x8)
                               | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200))
    else:
        kw["start_new_session"] = True
    with open(log, "ab") as f:
        f.write(f"\n--- moth setup started the pet, {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n".encode())
        f.flush()
        return subprocess.Popen(pet_command(url), stdout=f, **kw)


def alive(pid: int) -> bool:
    from moth import update   # stdlib only
    return update._alive(pid)


def has_display(env=None, platform: str | None = None) -> bool:
    env = os.environ if env is None else env
    platform = platform or sys.platform
    if platform == "darwin" or platform.startswith("win"):
        return True
    return bool(env.get("DISPLAY") or env.get("WAYLAND_DISPLAY") or env.get("QT_QPA_PLATFORM"))


def _tail(path: str, n: int = 8) -> list[str]:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return [x.rstrip() for x in f.readlines()[-n:] if x.strip()]
    except OSError:
        return []


def _fmt_cmd(cmd: list[str]) -> str:
    return hooks.command(*cmd)


# --- the run -----------------------------------------------------------------------------------------------------
class Setup:
    def __init__(self, yes: bool = False, dry: bool = False, url: str | None = None, timeout: float = 300.0,
                 note: str | None = None):
        self.yes, self.dry, self.timeout, self.note = yes, dry, timeout, note
        self.url = (url or os.environ.get("MOTH_URL") or DEFAULT_URL).rstrip("/")
        self.did: list[str] = []      # changed this run
        self.had: list[str] = []      # already in place
        self.left: list[str] = []     # still missing (declined, blocked or failed)
        self.failed = False
        self.serve_ok = self.weights_ok = self.pet_ok = self.ram_ok = True
        self.mode = weights.mode()
        self.started = self.running = False

    # output --------------------------------------------------------------------------------------------------
    @staticmethod
    def say(text: str = ""):
        print(text, flush=True)

    def line(self, mark: str, text: str):
        self.say(f"  {doctor._mark(mark)} {text}")

    def head(self, n: int, title: str):
        self.say(f"\n{n} {title}")

    def ask(self, question: str, default: bool) -> bool:
        """The one place setup asks. --yes / dry run: the default (printed); a closed input (EOF): no."""
        if self.yes or self.dry:
            self.say(f"    {question} {'yes' if default else 'no'}")
            return default
        try:
            a = input(f"    {question} {'[Y/n]' if default else '[y/N]'} ").strip().lower()
        except EOFError:
            self.say("")
            return False
        return default if not a else a in ("y", "yes")

    def run_quiet(self, fn, *args) -> tuple[int, str]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = fn(*args)
        return rc or 0, out.getvalue()

    def echo(self, text: str, indent: int = 4):
        for x in text.rstrip().splitlines():
            self.say(" " * indent + x)

    # 1 -------------------------------------------------------------------------------------------------------
    def packages(self):
        self.head(1, "packages")
        v = sys.version_info
        if v < (3, 10):
            self.line("✗", f"python {v.major}.{v.minor}: moth needs 3.10 or newer")
            self.serve_ok = self.pet_ok = False
            self.left.append("python 3.10+")
            return
        self.line("✓", f"python {v.major}.{v.minor}.{v.micro}  ({sys.executable})")
        cmds = self._package_check()
        if not cmds:
            return
        for c in cmds:
            self.say(f"      {_fmt_cmd(c)}")
        many = len(cmds) > 1
        if not self.ask(f"run {'these' if many else 'it'} now (pip, into {sys.prefix}; no sudo)?", False):
            self.left.append("packages (the pip command" + ("s" if many else "") + " above)")
            return
        if self.dry:
            self.say("    (dry run: not installing)")
            return
        for c in cmds:
            self.say(f"    running: {_fmt_cmd(c)}")
            try:
                rc = subprocess.call(c)
            except OSError as e:
                rc = f"can't run it: {e}"
            if rc != 0:
                self.line("✗", f"pip failed ({rc}); run it yourself and then {prog()} setup again")
                self.failed = True
                self.left.append("packages")
                return
        self.say("    checking again...")
        if self._package_check(again=True):
            self.left.append("packages")
            self.failed = True
        else:
            self.did.append("installed packages")

    def _package_check(self, again: bool = False) -> list[list[str]]:
        py = sys.executable
        cmds = []
        t = probe_torch()
        if "error" in t:
            missing = "No module named 'torch'" in t["error"]
            self.line("✗", "torch: " + ("not installed" if missing else f"broken ({t['error'][:120]})"))
            cmds.append([py, "-m", "pip", "install", "torch", "--index-url", TORCH_INDEX])
            self.serve_ok = False
        else:
            devs = [d for d in ("cuda", "xpu") if t.get(d)]
            self.line("✓", f"torch {t.get('version')}  (cpu{', ' + ', '.join(devs) if devs else ''})")
            self.serve_ok = True
        need = ["tokenizers", "safetensors", "numpy"] + (["huggingface_hub"] if self.mode == "pip" else [])
        gone = [m for m in need if importlib.util.find_spec(m) is None]
        if gone:
            self.line("✗", f"missing: {', '.join(gone)}")
            cmds.append([py, "-m", "pip", "install", *gone])
            if set(gone) - {"huggingface_hub"}:
                self.serve_ok = False
        else:
            self.line("✓", ", ".join(need))
        why = probe_pyside()
        if why:
            self.line("✗", f"PySide6 (the desktop pet): {why}")
            cmds.append([py, "-m", "pip", "install", self._pet_pkg()] if why == "not installed" else
                        [py, "-m", "pip", "install", "--force-reinstall", "PySide6-Essentials"])   # broken: just Qt
            self.pet_ok = False
        else:
            self.line("✓", "PySide6 (the desktop pet)")
            self.pet_ok = True
        if not cmds and not again:
            self.had.append("packages")
        return cmds

    @staticmethod
    def _pet_pkg() -> str:
        from importlib.metadata import PackageNotFoundError, distribution
        try:
            distribution("moth-companion")
            return "moth-companion[pet]"   # the pip install's own extra (PySide6-Essentials)
        except PackageNotFoundError:
            return "PySide6-Essentials"

    # 2 -------------------------------------------------------------------------------------------------------
    def weights_step(self):
        self.head(2, "weights")
        if config.model():   # another model behind moth (`moth config model`): moth's own weights aren't needed
            note = config._model_note(config.model()) or config.model()
            self.line("·", f"model: {note.splitlines()[0]}  (moth config model; moth doctor checks it)")
            for extra in note.splitlines()[1:]:
                self.say(f"      {extra}")
            return
        if self.mode == "hf":
            folder = os.path.dirname(weights.ROOT)
            if weights.complete(folder) and os.path.isfile(os.path.join(folder, "tokenizer.json")):
                self._have(folder, "hugging face download")
            else:
                self.weights_ok = False
                self.line("✗", f"{folder} is missing model files: "
                               + (", ".join(weights.missing(folder)) or "model.safetensors is cut off"))
                self.say(f"      finish it: cd {hooks.command(weights.ROOT)} && {prog()} update")
                self.say(f"      (or: hf download {weights.repo()} --local-dir {hooks.command(folder)}, which resumes)")
                self.left.append("weights (finish the download)")
            return
        if self.mode == "dev":
            try:
                p = launch.server_paths()
                self._have(p["model"], "dev checkout")
            except launch.LaunchError as e:
                self.weights_ok = False
                self.line("✗", str(e))
                self.left.append("weights")
            return
        folder = weights.cache_dir()
        if weights.present(folder) and weights.complete(folder):
            self._have(folder, "pip install")
            return
        if weights.present(folder):
            self.weights_ok = False
            self.line("✗", f"{os.path.join(folder, 'model.safetensors')} looks cut off")
            self.say(f"      repair it in place (with a backup): {prog()} update")
            self.left.append(f"weights ({prog()} update)")
            return
        self.weights_ok = False
        size = weights.SIZE_HINT if weights.repo() == weights.REPO_ID else "size unknown"
        self.line("·", f"not downloaded yet: {size}, from huggingface.co/{weights.repo()}")
        self.say(f"      into {folder}")
        if config.offline():
            self.line("✗", "offline mode is on (HF_HUB_OFFLINE / MOTH_OFFLINE): unset it to download")
            self.left.append("weights (offline)")
            return
        if importlib.util.find_spec("huggingface_hub") is None:
            self.line("✗", f"the download needs huggingface_hub: {sys.executable} -m pip install -U huggingface_hub")
            self.left.append("weights (needs huggingface_hub)")
            return
        free = self._free_gb(folder)
        if free is not None and free < 1.0:
            self.line("✗", f"only {free:.1f} GB free there; the weights need about 0.9 GB")
            self.left.append("weights (disk space)")
            return
        if not self.ask("download them now?", True):
            self.left.append("weights (the pet downloads them on its first start)")
            return
        if self.dry:
            self.say("    (dry run: not downloading)")
            self.weights_ok = True   # for the rest of the dry run: as if they were there
            return
        try:
            fetch_weights(folder)
        except SystemExit as e:   # weights.ensure says why (offline, no repo, cut off)
            self.line("✗", str(e.code if e.code is not None else "download failed"))
            self.failed = True
            self.left.append("weights (run setup again to resume)")
            return
        self.weights_ok = True
        self.line("✓", f"downloaded into {folder}")
        self.did.append("downloaded the weights")

    def _have(self, folder: str, what: str):
        try:
            mb = os.path.getsize(os.path.join(folder, "model.safetensors")) / 1e6
            self.line("✓", f"{mb:.0f} MB in {folder}  ({what})")
        except OSError:
            self.line("✓", f"{folder}  ({what})")
        self.had.append("weights")

    @staticmethod
    def _free_gb(folder: str) -> float | None:
        p = folder
        while p and not os.path.isdir(p) and os.path.dirname(p) != p:
            p = os.path.dirname(p)
        try:
            return shutil.disk_usage(p).free / 2**30
        except OSError:
            return None

    # 3 -------------------------------------------------------------------------------------------------------
    def memory(self):
        self.head(3, "memory")
        gb = ram_gb()
        if self.running:
            self.line("✓", (f"{gb:.1f} GB available" if gb is not None else "fine") + " (moth is already loaded)")
        elif gb is None:
            self.line("·", "couldn't read free memory here; moth needs about 1 GB")
        elif gb < MIN_RUN_GB:
            self.ram_ok = False
            self.line("✗", f"only {gb:.1f} GB available: moth needs about 1 GB (and ~3 GB free to start)")
            self.left.append("free memory")
        elif gb < launch.MIN_FREE_GB:
            self.ram_ok = False
            self.line("!", f"{gb:.1f} GB available: enough for moth once it runs (~1 GB), but the pet won't wake it "
                           f"below ~{launch.MIN_FREE_GB:g} GB free (loading takes more). close something first")
            self.left.append("free memory (~3 GB to start)")
        else:
            self.line("✓", f"{gb:.1f} GB available (moth needs about 1 GB)")

    # 4 -------------------------------------------------------------------------------------------------------
    def hooks_step(self):
        self.head(4, "coding agents")
        sts = [hooks.status(n) for n in hooks.HARNESSES]
        shown = [s for s in sts if s["detected"] or s["state"] != "absent"]
        if not shown:
            self.line("·", f"none found here (moth chats fine without one; later: {prog()} hook <harness>)")
            return
        for s in shown:
            self.line(hooks.MARK[s["state"]], f"{s['name']:<12} {hooks.WORD[s['state']]}: {s['detail']}")
        gone = [s["name"] for s in sts if s not in shown]
        if gone:
            self.say(f"    not found: {', '.join(gone)}")
        chosen = []
        for s in shown:
            if s["state"] == "installed":
                self.had.append(f"{s['name']} hook")
            elif s["state"] == "error":
                self.left.append(f"{s['name']} hook (see above)")
            elif s["state"] in FIXABLE:
                verb = "hook moth into" if s["state"] == "absent" else "fix moth's hook in"
                if self.ask(f"{verb} {s['name']}?", True):
                    chosen.append(s["name"])
                else:
                    self.left.append(f"{s['name']} hook")
        if chosen:
            plans = [hooks.plan_hook(n) for n in chosen]
            rc, out = self.run_quiet(hooks.run_plans, plans, True, self.dry, "install")
            self.echo(out)
            if self.dry:   # run_plans prints the notes (Codex's one approval, ...) only after doing it
                self.echo("\n".join(f"{p.harness}: note: {n}" for p in plans if p.actions for n in p.notes), 6)
            self.failed |= rc != 0 and not self.dry
            for p in plans:
                if p.error:
                    self.left.append(f"{p.harness} hook")
                elif not self.dry and p.actions:
                    ok = hooks.status(p.harness)["state"] == "installed"
                    (self.did if ok else self.left).append(f"{p.harness} hook" + ("" if ok else " (see the note)"))
        self.show_in_harness([s["name"] for s in shown], {"pi", "opencode"} & set(chosen))

    def show_in_harness(self, names: list[str], new: set):
        """Asked when a Pi / OpenCode hook goes in this run and the setting was never set; otherwise one line."""
        if not {"pi", "opencode"} & set(names):
            return
        value, source = config.lookup("show_in_harness")
        if source != "default" or not new:
            where = {"env": config.ENV["show_in_harness"], "file": "config"}.get(source, "default")
            self.line("✓" if source != "default" else "·",
                      f"show_in_harness: {'on' if value else 'off'} ({where}): moth's replies "
                      f"{'also' if value else 'not'} shown inside pi / opencode  ({prog()} config show_in_harness "
                      f"{'off' if value else 'on'})")
            return
        want = self.ask("show moth's replies inside pi / opencode too (show_in_harness)?", False)
        if self.dry or not want:
            return   # off is the default: nothing written (asked again next time, like any step left out)
        try:
            config.write("show_in_harness", want)
        except OSError as e:
            self.line("✗", f"couldn't write {config.path()}: {e}")
            self.failed = True
            return
        self.say(f"    show_in_harness = {'on' if want else 'off'}  (saved in {config.path()})")
        if want:
            self.did.append("show_in_harness on")

    # 5 -------------------------------------------------------------------------------------------------------
    def autostart_step(self):
        self.head(5, "autostart at login")
        st, p = autostart.state()
        if st == "on":
            self.line("✓", f"on ({p})")
            self.had.append("autostart")
            return
        if st == "foreign":
            self.line("!", f"{p} exists but isn't moth's; leaving it alone")
            self.left.append("autostart")
            return
        if not self.pet_ok:
            self.line("·", "skipped: the pet needs PySide6 (step 1)")
            self.left.append("autostart (needs PySide6)")
            return
        if st == "stale":
            self.line("!", f"{p} starts another python or folder than this one")
            q = "point it at this install?"
        else:
            self.line("·", "off")
            q = "start the pet when you log in?"
        if not self.ask(q, True):
            self.left.append(f"autostart ({prog()} autostart on)")
            return
        if self.dry:
            self.say(f"    would write {p}")
            return
        rc, out = self.run_quiet(autostart.main, ["on"])
        self.echo(out)
        if rc == 0:
            self.did.append("autostart on")
        else:
            self.failed = True
            self.left.append("autostart")

    # 6 -------------------------------------------------------------------------------------------------------
    def pet_step(self):
        self.head(6, "the pet")
        kind, info = health(self.url)
        if kind == "moth":
            pid = info.get("pid") if isinstance(info, dict) else None
            self.line("✓", f"moth is already awake at {self.url}" + (f" (pid {pid})" if pid else ""))
            self.say(f"    no pet on your screen? {'moth-pet' if prog() == 'moth' else 'python -m moth.pet'} "
                     "(it uses the moth that's running)")
            self.had.append("moth running")
            return
        if kind == "other":
            self.line("✗", f"something else answers at {self.url} ({info}); moth won't start there")
            self.say(f"      pick another port: {prog()} setup --url http://127.0.0.1:<port>  (hooks then need "
                     "MOTH_URL set to it)")
            self.left.append("start moth (port taken)")
            return
        if launch.local_port(self.url) is None:
            self.line("·", f"{self.url} isn't on this machine: start moth there")
            return
        log = _path(PET_LOG)
        pid = self._setup_pid()
        if pid:
            self.line("·", f"a pet setup started earlier is still waking moth up (pid {pid})")
            self._wait(lambda: alive(pid), lambda: None, pid, log)
            return
        why = []
        if not self.serve_ok:
            why.append("torch / tokenizers missing (step 1)")
        if not self.weights_ok:
            why.append("no weights yet (step 2)")
        if not self.pet_ok:
            why.append("PySide6 missing (step 1)")
        if not self.ram_ok:
            why.append("not enough free memory (step 3)")
        if not has_display():
            why.append("no graphical session here (moth without the pet: " + prog() + ")")
        if why:
            self.line("·", "not starting it: " + "; ".join(why))
            self.left.append("start the pet")
            return
        self.line("·", f"moth isn't running at {self.url}")
        if not self.ask("start the pet now?", True):
            self.left.append(f"start the pet ({'moth-pet' if prog() == 'moth' else 'python -m moth.pet'})")
            return
        if self.dry:
            self.say(f"    would run: {_fmt_cmd(pet_command(self.url))}  (detached, log {log})")
            self.say(f"    and wait up to {self.timeout:.0f} s for {self.url}/api/health")
            return
        try:
            proc = launch_pet(self.url, log)
        except OSError as e:
            self.line("✗", f"couldn't start the pet: {e}")
            self.failed = True
            self.left.append("start the pet")
            return
        self._write_pid(proc.pid)
        self._wait(lambda: proc.poll() is None, proc.poll, proc.pid, log)

    def _wait(self, is_alive, code, pid: int, log: str):
        try:
            serve_log = launch.server_paths()["log"]
        except launch.LaunchError:
            serve_log = None
        logs = f"{log}" + (f", {serve_log}" if serve_log else "")
        self.say("    waiting for moth to wake up (the model loads in well under a minute on most machines)...")
        deadline = time.monotonic() + self.timeout
        while True:
            kind, info = health(self.url)
            if kind == "moth":
                spid = info.get("pid") if isinstance(info, dict) else None
                self.line("✓", f"moth is awake at {self.url} (pet pid {pid}" + (f", server pid {spid}" if spid else "")
                          + ")")
                self.say(f"    logs: {logs}")
                self.did.append("moth started")
                self.started = True
                self._drop_pid()
                return
            if kind == "other" or not is_alive():
                if kind == "other":
                    self.line("✗", f"something else answers at {self.url} ({info})")
                else:
                    c = code()
                    self.line("✗", "the pet stopped before moth woke up" + (f" (exit {c})" if c is not None else ""))
                tail = _tail(log)
                if tail:
                    self.say(f"    last lines of {log}:")
                    self.echo("\n".join(tail), 6)
                self.failed = True
                self.left.append("start the pet")
                self._drop_pid()
                return
            if time.monotonic() >= deadline:
                self.line("!", f"no answer from {self.url} after {self.timeout:.0f} s; moth may still be loading "
                               f"(the pet keeps going, pid {pid})")
                self.say(f"    logs: {logs}")
                self.left.append(f"moth answering (check {prog()} doctor in a minute)")
                self.failed = True
                return
            time.sleep(1.0)

    def _setup_pid(self) -> int | None:
        try:
            with open(_path(PIDFILE)) as f:
                d = json.load(f)
            pid = int(d["pid"])
        except (OSError, ValueError, KeyError, TypeError):
            return None
        if d.get("url") == self.url and alive(pid):
            return pid
        self._drop_pid()
        return None

    def _write_pid(self, pid: int):
        """Only while the pet wakes moth up: a re-run in that window waits for it instead of starting a second."""
        try:
            os.makedirs(os.path.dirname(_path(PIDFILE)), exist_ok=True)
            with open(_path(PIDFILE), "w") as f:
                json.dump({"pid": pid, "url": self.url}, f)
        except OSError:
            pass

    @staticmethod
    def _drop_pid():
        with contextlib.suppress(OSError):
            os.remove(_path(PIDFILE))

    # all of it ----------------------------------------------------------------------------------------------
    def where(self) -> str:
        if self.mode == "hf":
            return f"hugging face download, {os.path.dirname(weights.ROOT)}"
        if self.mode == "pip":
            return "pip install"
        return f"dev checkout, {weights.ROOT}"

    def run(self) -> int:
        self.say(f"moth setup ~ let's get you a moth.  ({self.where()})")
        if self.dry:
            self.say(self.note or "dry run: showing what setup would do with the defaults; nothing is changed or "
                                  "started.")
        kind, _ = health(self.url)
        self.running = kind == "moth"
        self.packages()
        self.weights_step()
        self.memory()
        self.hooks_step()
        self.autostart_step()
        self.pet_step()
        self.summary()
        if self.dry:
            return 0
        return 1 if self.failed or not (self.serve_ok and self.weights_ok) else 0

    def summary(self):
        p = prog()
        self.say("")
        if self.dry:
            self.say("that was a dry run. " + ("run `" + p + " setup` in a terminal to answer the questions, or "
                                               "`" + p + " setup --yes` for the defaults."))
        elif not self.left:
            self.say("all set." + (" moth is fluttering around your screen somewhere, probably near a light."
                                   if self.started else ""))
        else:
            self.say(f"almost: {len(self.left)} thing{'s' if len(self.left) > 1 else ''} left.")
        if self.did:
            self.say("  done now:      " + ", ".join(self.did))
        if self.had:
            self.say("  already there: " + ", ".join(self.had))
        if self.left:
            self.say("  left:          " + ", ".join(self.left))
        where = f" (from {weights.ROOT})" if self.mode == "hf" and p != "moth" else ""
        self.say(f"\nnext{where}:")
        for cmd, what in (("doctor", "health check"), ("config", "moth's settings"),
                          ("forget", "wipe what moth remembers about you"), ("update", "get a newer moth"),
                          ("setup", "again any time: it only offers what's missing")):
            self.say(f"  {p} {cmd:<8} {what}")
        self.say(f"uninstall: {p} unhook --all, then {p} autostart off")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog=f"{prog()} setup", description="from install to a running moth: checks the "
                                 "packages and weights, hooks your coding agents, autostart, and starts the pet")
    ap.add_argument("--yes", "-y", action="store_true",
                    help="take every default without asking (never runs pip; downloads the weights)")
    ap.add_argument("--dry-run", action="store_true", help="only print what would be done")
    ap.add_argument("--url", default=None, help=f"where moth answers (default $MOTH_URL or {DEFAULT_URL})")
    ap.add_argument("--timeout", type=float, default=300.0,
                    help="seconds to wait for moth to answer after starting the pet (default 300)")
    ap.add_argument("--model", default=None,
                    help="the model behind moth: a weights folder, an HF repo id, openai:<url>#<model> or default "
                         "(saved as `moth config model`; moth/docs/MODEL_BACKENDS.md)")
    a = ap.parse_args(argv)
    dry, note = a.dry_run, None
    if not dry and not a.yes and not _tty():
        dry, note = True, "no terminal to ask in, so this is a dry run: nothing is changed or started (--yes does " \
                          "it with the defaults)."
    if a.model is not None:
        value = config._model_value(a.model.strip())
        if value != "default":
            from moth import backends
            try:
                backends.resolve(value, network=False)
            except backends.ManifestError as e:
                print(f"moth setup: --model {a.model}: {e}", file=sys.stderr)
                return 2
        if dry:
            print(f"would set model = {value}  ({prog()} config model)", flush=True)
        else:
            config.write("model", value)
            print(f"model = {value}  (saved in {config.path()})", flush=True)
    s = Setup(yes=a.yes, dry=dry, url=a.url, timeout=a.timeout, note=note)
    try:
        return s.run()
    except KeyboardInterrupt:
        print("\nstopped. what's done stays done; run setup again to pick up from here.", flush=True)
        return 130


if __name__ == "__main__":
    sys.exit(main())
