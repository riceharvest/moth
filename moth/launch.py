"""Waking moth's server for the desktop pet (stdlib only: never imports torch, so the pet starts fast).

python -m moth.pet first asks GET /api/health on its URL (moth.serve answers it without a token):
  a moth answers          -> use it; the pet never stops a server it didn't start
  nothing listens         -> start `python -m moth.serve` as a child (same interpreter, CPU, the weights below) and
                             stop it again when the pet quits
  something else answers  -> start nothing; the pet says so

Weights, tokenizer, home and log:
  Hugging Face download (model.safetensors next to code/): that model, ~/.moth/home, log ~/.moth/serve.log
  pip install (moth/weights.py): the model files in weights.cache_dir() (~/.cache/moth/<repo>), downloaded on the
  first run (weights.ensure; the pet says "downloading moth…" meanwhile), ~/.moth/home, log ~/.moth/serve.log
  this dev checkout: the first complete one of runs/final2/ckpt_100, runs/final/ckpt_100, runs/v2/model (as
  try_moth.sh), data/tiny-agent-112m/tokenizer.json, live/home, log live/serve.log
  --model DIR (pet flag) picks the weights; MOTH_SERVE_LOG moves the log.

The child runs in its own process group (POSIX: its own session) so it can be stopped as a whole, and on Linux it
gets SIGTERM if the pet dies without cleaning up (PR_SET_PDEATHSIG). Windows: its own process group (Ctrl-Break)
inside a Job Object that kills every process in it when stopped or when the pet dies (the job's last handle closes),
so the server's own children go too, even after the server itself has exited; `taskkill /T /F` is the fallback
when no job could be made. Everywhere (and the only guard on macOS, which has neither): the pet starts the server
with --exit-with-parent and a pipe on its stdin that the pet holds and never writes to; when the pet dies, however
it dies, the OS closes that pipe and the server sees EOF and stops cleanly (exit_with_parent below). A server run on
its own (`python -m moth`, `python -m moth.launch`, try_moth.sh --no-pet) never gets that flag.
MOTH_SERVE_CMD (a shell-quoted command)
replaces `python -m moth.serve` (the tests start a stub server that way).

  python -m moth.launch [--model DIR] [--port N] [-- moth.serve flags]
      run moth.serve in this terminal with the weights picked above (try_moth.sh --no-pet uses it)
"""
from __future__ import annotations

import atexit
import json
import os
import shlex
import signal
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit

# stdlib only at import: where a pip install keeps (and downloads) its weights
from moth import weights

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # moth-112m/code, or this checkout
DEV_WEIGHTS = ("runs/final2/ckpt_100", "runs/final/ckpt_100", "runs/v2/model")
MIN_FREE_GB = 3.0        # moth peaks ~3.5 GB while loading; below ~3 GB free the OOM killers start
LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")
READY_TIMEOUT = 300.0    # s for the model to load (on CPU usually well under a minute)

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # moth is local: never via a proxy


class LaunchError(Exception):
    """Why moth can't be started (shown in the pet's bubble)."""


# --- weights and paths ----------------------------------------------------------------------------------------
def complete(folder: str) -> bool:
    """config.json plus a model.safetensors whose size matches its header (a synced folder may be mid-copy)."""
    st = os.path.join(folder, "model.safetensors")
    if not (os.path.isfile(os.path.join(folder, "config.json")) and os.path.isfile(st)):
        return False
    try:
        with open(st, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            head = json.loads(f.read(n))
        end = max(v["data_offsets"][1] for k, v in head.items() if k != "__metadata__")
        return os.path.getsize(st) == 8 + n + end
    except Exception:  # noqa: BLE001  short, garbage, no tensors: not complete
        return False


def hf_repo(code: str = CODE) -> str | None:
    """The download folder when this is the Hugging Face layout (weights next to code/), else None."""
    repo = os.path.dirname(code)
    return repo if os.path.exists(os.path.join(repo, "model.safetensors")) else None


def install_mode(code: str = CODE) -> str:
    """"hf", "dev" or "pip" (weights.mode() for this install; for another folder (tests): hf or dev)."""
    if os.path.realpath(code) == os.path.realpath(weights.ROOT):
        return weights.mode()
    return "hf" if hf_repo(code) else "dev"


def server_paths(model: str | None = None, code: str = CODE, mode: str | None = None) -> dict:
    """{"model", "tokenizer", "home", "log", "download"} for this install; "download": the folder weights.ensure()
    has to fill before the server can start (pip install, first run), else None. Raises LaunchError without usable
    weights."""
    home = os.path.expanduser("~")
    mode = mode or install_mode(code)
    repo = hf_repo(code)
    d = {"download": None}
    if mode == "pip":   # only the code is installed: never put home or the log under site-packages
        cache = weights.cache_dir()
        d.update(model=cache, tokenizer=os.path.join(cache, "tokenizer.json"),
                 home=os.path.join(home, ".moth", "home"), log=os.path.join(home, ".moth", "serve.log"))
        if model and os.path.isfile(os.path.join(model, "tokenizer.json")):
            d["tokenizer"] = os.path.join(os.path.abspath(os.path.expanduser(model)), "tokenizer.json")
        elif not weights.present(cache) and not (model and os.path.isfile(d["tokenizer"])):
            d["download"] = cache   # (with --model too when the tokenizer only comes with the download)
    elif repo:
        d.update(model=repo, tokenizer=os.path.join(repo, "tokenizer.json"),
                 home=os.path.join(home, ".moth", "home"), log=os.path.join(home, ".moth", "serve.log"))
    else:
        d.update(model=None, tokenizer=os.path.join(code, "data", "tiny-agent-112m", "tokenizer.json"),
                 home=os.path.join(code, "live", "home"), log=os.path.join(code, "live", "serve.log"))
        if not model:
            d["model"] = next((os.path.join(code, c) for c in DEV_WEIGHTS if complete(os.path.join(code, c))), None)
            if not d["model"]:
                raise LaunchError("no complete weights in " + ", ".join(c.rsplit("/", 1)[0] for c in DEV_WEIGHTS))
    if model:
        model = os.path.abspath(os.path.expanduser(model))
        if not complete(model):
            raise LaunchError(f"{model} has no complete config.json + model.safetensors")
        d["model"] = model
    d["log"] = os.environ.get("MOTH_SERVE_LOG") or d["log"]
    return d


def server_command(port: int, paths: dict, extra=(), token_file: str | None = None,
                   exit_with_parent: bool = False) -> list[str]:
    """moth.serve on CPU with these paths; `extra` goes last, so the user's flags win (argparse: last one counts).
    exit_with_parent: only for a server the pet starts (Server.start pipes its stdin), never for one run on its own."""
    base = shlex.split(os.environ.get("MOTH_SERVE_CMD", "")) or [sys.executable, "-m", "moth.serve"]
    cmd = base + ["--device", "cpu", "--model", paths["model"], "--tokenizer", paths["tokenizer"],
                  "--home", paths["home"], "--port", str(port)]
    if token_file:
        cmd += ["--token-file", token_file]
    if exit_with_parent:
        cmd.append(EXIT_WITH_PARENT)
    return cmd + list(extra)


# --- what's on the port ---------------------------------------------------------------------------------------
def mem_available_gb(path: str = "/proc/meminfo") -> float | None:
    """MemAvailable in GB, or None where there's no /proc/meminfo (macOS, Windows: no check there)."""
    try:
        with open(path) as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1048576
    except (OSError, ValueError, IndexError):
        pass
    return None


def _get(url: str, timeout: float):
    with _OPENER.open(urllib.request.Request(url), timeout=timeout) as r:
        return json.loads(r.read(1 << 20))


def probe(url: str, timeout: float = 2.0) -> tuple[str, dict | str]:
    """("moth", health) / ("none", why) / ("other", what answered). "none" when nothing accepts the connection: refused,
    or the connect itself timed out (Windows retries a refused loopback connect for ~2 s, so a free port can look
    like a connect timeout there). "other" when something accepted it and then answered wrong, or not at all."""
    url = url.rstrip("/")
    try:
        d = _get(f"{url}/api/health", timeout)
    except urllib.error.HTTPError as e:
        if e.code == 404:   # an older moth without /api/health: its token-free /api/says has a known shape
            try:
                d = _get(f"{url}/api/says?since=1000000000", timeout)
                if isinstance(d, dict) and isinstance(d.get("events"), list) and isinstance(d.get("epoch"), str):
                    return "moth", {"ok": True, "pid": None, "old": True}
            except Exception:  # noqa: BLE001, S110  not that either: "other" below
                pass
        return "other", f"it answers HTTP {e.code}"
    except urllib.error.URLError as e:
        if isinstance(e.reason, ConnectionRefusedError):
            return "none", "nothing listens"
        if isinstance(e.reason, OSError) and getattr(e.reason, "errno", None) in (111, 61, 10061):
            return "none", "nothing listens"
        if isinstance(e.reason, TimeoutError):   # urllib wraps only connect/send errors: nothing accepted in time
            return "none", "nothing answers"
        return "other", f"it doesn't answer like moth ({e.reason})"
    except TimeoutError:   # accepted, then said nothing within the timeout: something is there, not moth
        return "other", "it doesn't answer like moth (timed out)"
    except ConnectionRefusedError:
        return "none", "nothing listens"
    except Exception as e:  # noqa: BLE001  reset, timeout, garbage
        return "other", f"it doesn't answer like moth ({type(e).__name__})"
    if isinstance(d, dict) and d.get("ok") is True and isinstance(d.get("pid"), int):
        return "moth", d
    return "other", "it doesn't answer like moth"


def local_port(url: str) -> int | None:
    """The port when url is on this machine (the pet only starts servers here), else None."""
    u = urlsplit(url)
    if (u.hostname or "") not in LOCAL_HOSTS:
        return None
    return u.port or (443 if u.scheme == "https" else 80)


# --- the child ------------------------------------------------------------------------------------------------
def _deathsig():
    """Linux: a preexec_fn that makes the child get SIGTERM when the pet dies, even by SIGKILL (no atexit then).
    libc is looked up here, in the parent; the child only makes the call."""
    try:
        import ctypes
        prctl = ctypes.CDLL(None, use_errno=True).prctl
    except Exception:  # noqa: BLE001  no prctl: the pet's quit/atexit paths still stop the server
        return None
    term = int(signal.SIGTERM)
    return lambda: prctl(1, term)   # PR_SET_PDEATHSIG


class _WinJob:
    """Windows: a Job Object with KILL_ON_JOB_CLOSE holding the server; processes it starts join it too (the
    default). kill() ends them all, and so does the pet dying (the handle closes). ctypes only; any failure: None
    from make(), and stop() falls back to taskkill /T."""

    @staticmethod
    def make(proc):
        try:
            import ctypes
            from ctypes import wintypes

            class Basic(ctypes.Structure):
                _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                            ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                            ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                            ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                            ("SchedulingClass", wintypes.DWORD)]

            class Io(ctypes.Structure):
                _fields_ = [(n, ctypes.c_uint64) for n in ("ReadOperationCount", "WriteOperationCount",
                                                           "OtherOperationCount", "ReadTransferCount",
                                                           "WriteTransferCount", "OtherTransferCount")]

            class Extended(ctypes.Structure):
                _fields_ = [("BasicLimitInformation", Basic), ("IoInfo", Io), ("ProcessMemoryLimit", ctypes.c_size_t),
                            ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                            ("PeakJobMemoryUsed", ctypes.c_size_t)]

            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.CreateJobObjectW.restype = wintypes.HANDLE
            k32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
            k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
            k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
            k32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
            k32.CloseHandle.argtypes = [wintypes.HANDLE]
            job = k32.CreateJobObjectW(None, None)
            if not job:
                return None
            info = Extended()
            info.BasicLimitInformation.LimitFlags = 0x2000   # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            ok = k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))   # ExtendedLimit
            if not ok or not k32.AssignProcessToJobObject(job, int(proc._handle)):
                k32.CloseHandle(job)
                return None
            return _WinJob(k32, job)
        except Exception:  # noqa: BLE001  no job: taskkill /T is the fallback
            return None

    def __init__(self, k32, handle):
        self.k32, self.handle = k32, handle

    def kill(self):
        """End every process still in the job, then let the handle go."""
        if self.handle:
            self.k32.TerminateJobObject(self.handle, 1)
            self.k32.CloseHandle(self.handle)
            self.handle = None


def _taskkill_tree(pid: int):
    """Windows fallback: the process and everything it started (while it lives: the tree is found by parent pid)."""
    try:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], stdin=subprocess.DEVNULL,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
    except (OSError, subprocess.SubprocessError):
        pass


EXIT_WITH_PARENT = "--exit-with-parent"


def exit_with_parent(stop=None, grace: float = 10.0):
    """The server side of the lifeline (moth.serve --exit-with-parent; tests/fake_serve.py too): a daemon thread reads
    stdin, which is a pipe whose write end only the pet holds and never writes to. EOF (or a broken-pipe error,
    Windows) means the pet is gone, even after SIGKILL: then stop() runs (default: KeyboardInterrupt in the main
    thread, the same clean exit as Ctrl-C), and if the process is still there `grace` s later it exits hard.
    Returns the thread, or None when there's no stdin to watch."""
    import _thread
    import threading
    try:
        fd = sys.stdin.fileno()
    except (AttributeError, OSError, ValueError):   # pythonw: no stdin at all
        print("moth: --exit-with-parent but no stdin to watch; ignored", file=sys.stderr, flush=True)
        return None

    def watch():
        try:
            while os.read(fd, 4096):   # the pet never writes; anything that does arrive is ignored
                pass
        except OSError:
            pass
        print("moth: the program that started me is gone; stopping", file=sys.stderr, flush=True)
        try:
            (stop or _thread.interrupt_main)()
        finally:
            time.sleep(grace)
            os._exit(0)

    t = threading.Thread(target=watch, daemon=True, name="moth-lifeline")
    t.start()
    return t


class Server:
    """A moth.serve the pet started. start() in the main thread (PR_SET_PDEATHSIG follows the spawning thread);
    stop() is idempotent and safe from atexit, aboutToQuit and signal-driven quits alike."""

    def __init__(self, cmd: list[str], url: str, log: str):
        self.cmd, self.url, self.log = cmd, url.rstrip("/"), log
        self.proc: subprocess.Popen | None = None
        self.stopped = False
        self.job = None   # Windows: the _WinJob holding the server and its children

    def start(self):
        os.makedirs(os.path.dirname(self.log) or ".", exist_ok=True)
        env = dict(os.environ)
        env["PYTHONPATH"] = CODE + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env["PYTHONUNBUFFERED"] = "1"
        kw: dict = {}
        if os.name == "nt":
            kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kw["start_new_session"] = True   # own session + process group: Ctrl-C in the terminal reaches only the
            if sys.platform.startswith("linux"):   # pet, which then stops the group
                kw["preexec_fn"] = _deathsig()
        with open(self.log, "a", encoding="utf-8") as f:
            f.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} moth pet started: {shlex.join(self.cmd)}\n")
            f.flush()
            # stdin: the lifeline (exit_with_parent); the pet keeps the write end open, unwritten, until stop()
            self.proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=f, stderr=subprocess.STDOUT,
                                         cwd=CODE, env=env, **kw)
        if os.name == "nt":   # right away, before the server (a Python start-up: tens of ms) can start children
            self.job = _WinJob.make(self.proc)
        atexit.register(self.stop)
        return self

    @property
    def pid(self) -> int | None:
        return self.proc.pid if self.proc else None

    def exited(self) -> bool:
        return self.proc is not None and self.proc.poll() is not None

    def wait_ready(self, stop=None, timeout: float = READY_TIMEOUT, every: float = 0.1) -> tuple[bool, str]:
        """Until /api/health answers: (True, "") or (False, why). stop: a threading.Event that gives up early.
        every: s between probes (a refused local connect costs next to nothing; 0.5 s added ~0.25 s to the wake-up)."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if stop is not None and stop.is_set():
                return False, "stopped"
            if self.exited():
                if probe(self.url, timeout=1.0)[0] == "moth":   # another moth won the port while ours loaded:
                    return True, "another moth woke up first; using that one"   # (not ours: never stopped)
                return False, f"moth's server stopped while waking up (exit {self.proc.returncode}); see {self.log}"
            if probe(self.url, timeout=1.0)[0] == "moth":
                return True, ""
            time.sleep(every)
        self.stop()
        return False, f"moth didn't wake up within {int(timeout)} s; see {self.log}"

    def _signal_group(self, sig) -> bool:
        try:
            os.killpg(self.proc.pid, sig)
            return True
        except (ProcessLookupError, PermissionError, OSError):
            return False

    def stop(self, grace: float = 5.0):
        """SIGINT to the group (serve.py exits cleanly and removes its pidfile), then SIGTERM, then SIGKILL; reaps."""
        if self.proc is None or self.stopped:
            return
        self.stopped = True
        p = self.proc
        if os.name == "nt":   # Ctrl-Break to its group (a clean exit), then the whole tree goes
            try:
                if p.poll() is None:
                    try:
                        p.send_signal(signal.CTRL_BREAK_EVENT)
                        p.wait(grace)
                    except (OSError, subprocess.TimeoutExpired):
                        pass
            finally:
                if p.poll() is None and self.job is None:
                    _taskkill_tree(p.pid)   # no job: get the tree while the server still holds it together
                if self.job is not None:
                    self.job.kill()         # the server's children too, even if the server already exited
                if p.poll() is None:
                    p.kill()
                try:
                    p.wait(5.0)
                except subprocess.TimeoutExpired:
                    pass
                self._close_lifeline()
            return
        try:
            for sig, wait in ((signal.SIGINT, grace), (signal.SIGTERM, 3.0), (signal.SIGKILL, 5.0)):
                if p.poll() is not None:
                    break
                self._signal_group(sig)
                try:
                    p.wait(wait)
                except subprocess.TimeoutExpired:
                    continue
        finally:   # also when a second Ctrl-C interrupts the waiting: never leave the group behind
            self._signal_group(signal.SIGKILL)   # (and anything the server left in its group)
            try:
                p.wait(1.0)
            except subprocess.TimeoutExpired:
                pass
            self._close_lifeline()

    def _close_lifeline(self):
        """Our end of the server's stdin pipe (a server still running sees EOF and stops: exit_with_parent)."""
        try:
            pipe = getattr(self.proc, "stdin", None)
            if pipe is not None:
                pipe.close()
        except OSError:
            pass


# --- what the pet does at start -------------------------------------------------------------------------------
class Boot:
    """state: "ready" (a moth answers), "starting" (server = our child, loading), "downloading" (a pip install's first
    run: fetch() the weights into `download`, then server.start() from the main thread), "error" (message says why;
    nothing started), "wait" (just poll: --no-server, or a URL on another machine)."""

    def fetch(self) -> str | None:
        """weights.ensure() (no time limit: ~860 MB): None when done, else why not. Any thread."""
        try:
            weights.ensure(self.download)
            return None
        except SystemExit as e:   # ensure() exits with a message (no huggingface_hub, offline, no such repo)
            return str(e.code if e.code is not None else "") or "moth's weights couldn't be downloaded"
        except Exception as e:  # noqa: BLE001
            return f"moth's weights couldn't be downloaded ({type(e).__name__}: {e})"

    def __init__(self, state: str, message: str = "", server: Server | None = None, download: str | None = None):
        self.state, self.message, self.server, self.download = state, message, server, download

    def __repr__(self):
        return f"Boot({self.state!r}, {self.message!r}, server={self.server and self.server.pid})"


def prepare(url: str, no_server: bool = False, model: str | None = None, extra=(), token_file: str | None = None,
            mem=None, min_free_gb: float = MIN_FREE_GB) -> Boot:
    """Decide (and, if needed, start the server) before the pet's windows exist. Call from the main thread."""
    kind, info = probe(url)
    if kind == "moth":
        return Boot("ready")
    port = local_port(url)
    if port is None:   # another machine: never started from here; just poll it (slow or down looks the same)
        return Boot("wait", "" if no_server else f"no moth at {url} yet, and i only start moth on this machine")
    if kind == "other":
        return Boot("error", f"something else is on port {port} ({info}), so i won't wake moth there")
    if no_server:
        return Boot("wait")
    free = (mem or mem_available_gb)()
    if free is not None and free < min_free_gb:
        return Boot("error", f"not enough memory to wake moth ({free:.1f} GB free, needs ~{min_free_gb:g} GB)")
    try:
        paths = server_paths(model)
    except LaunchError as e:
        return Boot("error", f"can't wake moth: {e}")
    srv = Server(server_command(port, paths, extra, token_file, exit_with_parent=True), url, paths["log"])
    if paths["download"]:   # first run of a pip install: the pet downloads (in a thread), then starts srv
        return Boot("downloading", f"downloading moth's weights ({weights.SIZE_HINT}) into {paths['download']}",
                    srv, download=paths["download"])
    try:
        srv.start()
    except OSError as e:
        return Boot("error", f"can't wake moth: {e}")
    return Boot("starting", f"waking moth (weights {paths['model']}, log {paths['log']})", srv)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    import argparse
    ap = argparse.ArgumentParser(description="run moth.serve with this install's weights (picked as the pet does)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--port", type=int, default=4747)
    a = ap.parse_args(argv)
    try:
        paths = server_paths(a.model)
    except LaunchError as e:
        print(e, file=sys.stderr)
        return 1
    if paths["download"]:
        weights.ensure(paths["download"])
    print(f"moth weights: {paths['model']}", flush=True)
    cmd = server_command(a.port, paths, extra)
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env["PYTHONPATH"] = CODE + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if os.name == "nt":
        return subprocess.call(cmd, cwd=CODE, env=env)
    os.chdir(CODE)
    os.execvpe(cmd[0], cmd, env)   # this terminal's Ctrl-C goes straight to moth.serve


if __name__ == "__main__":
    sys.exit(main())
