"""moth doctor : a one-screen health check, a fix hint on every line that needs one.

  moth doctor [--port 4747]

Checks Python, torch (in a subprocess with a timeout, so a missing or broken torch can't take this down), the other
packages, the model (`moth config model`: backend, model, character; a privacy line when an openai endpoint
isn't on this machine) and, for moth's own model, the weights and tokenizer, the token file, the port, the
coding-agent hooks, PySide6 and the display for the pet, free RAM and disk, the show_in_harness setting and the
version. No model is loaded; it takes a second or
two (most of it torch's import). Exit 1 if any line is ✗.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import shutil
import socket
import struct
import subprocess
import sys

TORCH_PROBE = ("import json, torch\n"
               "d = {'version': torch.__version__, 'cuda': bool(torch.cuda.is_available())}\n"
               "x = getattr(torch, 'xpu', None)\n"
               "try:\n    d['xpu'] = bool(x is not None and x.is_available())\n"
               "except Exception:\n    d['xpu'] = False\n"
               "print('MOTH-DOCTOR ' + json.dumps(d))\n")
TORCH_HINT = "pip install torch --index-url https://download.pytorch.org/whl/cpu   (CPU build; CUDA/XPU builds work too)"
MIN_RAM_GB = 3.0


def prog() -> str:
    return "moth" if os.path.basename(sys.argv[0]) in ("moth", "moth.exe") else "python -m moth"


class Report:
    def __init__(self):
        self.lines: list[tuple[str, str, str, str | None]] = []   # mark, name, text, fix

    def add(self, mark: str, name: str, text: str, fix: str | None = None):
        self.lines.append((mark, name, text, fix))

    @property
    def failed(self) -> bool:
        return any(m == "✗" for m, *_ in self.lines)

    def render(self, mark=lambda m: m) -> str:
        out = []
        for m, name, text, fix in self.lines:
            out.append(f"{mark(m)} {name:<16}{text}")
            if fix and m != "✓":
                out.append(f"  {'':<16}fix: {fix}")
        return "\n".join(out)


# --- the checks (each one catches its own failures) -------------------------------------------------------------
def check_python(r: Report):
    v = sys.version_info
    text = f"{v.major}.{v.minor}.{v.micro}  ({sys.executable})"
    if v >= (3, 10):
        r.add("✓", "python", text)
    else:
        r.add("✗", "python", text + ": moth needs 3.10+", "install Python 3.10 or newer")


def probe_torch(python: str = sys.executable, env: dict | None = None, timeout: float = 90) -> dict:
    """{"version", "cuda", "xpu"} or {"error": "..."}: torch imported in a child process."""
    try:
        p = subprocess.run([python, "-c", TORCH_PROBE], capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return {"error": f"importing torch took over {timeout:.0f} s"}
    except OSError as e:
        return {"error": str(e)}
    for line in p.stdout.splitlines():
        if line.startswith("MOTH-DOCTOR "):
            try:
                return json.loads(line[len("MOTH-DOCTOR "):])
            except ValueError:
                break
    err = [x for x in p.stderr.strip().splitlines() if x.strip()]
    return {"error": err[-1] if err else f"python exited with {p.returncode}"}


def check_torch(r: Report, probe=probe_torch):
    d = probe()
    if "error" in d:
        missing = "No module named 'torch'" in d["error"]
        r.add("✗", "torch", "not installed" if missing else f"broken: {d['error'][:120]}",
              TORCH_HINT if missing else "reinstall torch:  " + TORCH_HINT)
        return
    devs = ["cpu"] + (["cuda"] if d.get("cuda") else []) + (["xpu"] if d.get("xpu") else [])
    r.add("✓", "torch", f"{d.get('version')}  (devices: {', '.join(devs)}; moth runs fine on cpu, --device "
                        f"{devs[-1]} to use {devs[-1]})" if len(devs) > 1 else f"{d.get('version')}  (cpu)")


def check_packages(r: Report):
    from importlib.metadata import PackageNotFoundError, version
    bad, ok = [], []
    for mod in ("tokenizers", "safetensors", "numpy"):
        try:
            importlib.import_module(mod)
            try:
                ok.append(f"{mod} {version(mod)}")
            except PackageNotFoundError:
                ok.append(mod)
        except Exception as e:   # missing, or broken (a wheel for another Python ...)
            bad.append(f"{mod} ({'missing' if isinstance(e, ImportError) and 'No module' in str(e) else e})")
    if bad:
        r.add("✗", "packages", ", ".join(bad) + (f"; ok: {', '.join(ok)}" if ok else ""),
              "pip install tokenizers safetensors numpy   (or pip install -r requirements.txt)")
    else:
        r.add("✓", "packages", ", ".join(ok))


def weights_paths() -> tuple[str, str, str]:
    """(mode, model folder, tokenizer file) as serve would pick them by default (serve itself imports torch)."""
    from moth import weights
    mode = weights.mode()
    if mode == "hf":
        folder = os.path.dirname(weights.ROOT)
        return mode, folder, os.path.join(folder, "tokenizer.json")
    if mode == "dev":
        return mode, weights.DEV_MODEL, os.path.join(weights.DEV_MODEL, "tokenizer.json")
    folder = weights.cache_dir()
    return mode, folder, os.path.join(folder, "tokenizer.json")


def safetensors_complete(path: str) -> bool:
    """The header's last tensor ends exactly at the end of the file (a cut-off download doesn't)."""
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            if n > 100_000_000:
                return False
            h = json.loads(f.read(n))
        end = max(v["data_offsets"][1] for k, v in h.items() if k != "__metadata__")
        return os.path.getsize(path) == 8 + n + end
    except (OSError, ValueError, KeyError, TypeError, struct.error):
        return False


def check_weights(r: Report) -> str | None:
    from moth import weights
    try:
        mode, folder, _ = weights_paths()
    except Exception as e:
        r.add("✗", "weights", f"can't work out where they are: {e}")
        return None
    st = os.path.join(folder, "model.safetensors")
    if os.path.isfile(st):
        size = os.path.getsize(st) / 1e6
        if not safetensors_complete(st):
            r.add("✗", "weights", f"{st} looks cut off ({size:.0f} MB)",
                  f"{prog()} update (or delete the file and run moth again to re-download)")
        elif not os.path.isfile(os.path.join(folder, "config.json")):
            r.add("✗", "weights", f"{folder} has no config.json", f"{prog()} update")
        else:
            r.add("✓", "weights", f"{size:.0f} MB in {folder}  ({mode} install)")
    elif mode == "pip" and not weights.released(weights.repo()):
        r.add("✗", "weights", f"no model yet: the XS release isn't published, so there is nothing to download into "
                              f"{folder}", f"{prog()} config model <a model folder> (or set MOTH_REPO)")
    elif mode == "pip":
        r.add("!", "weights", f"not downloaded yet: moth fetches them ({weights.SIZE_HINT}) into {folder} on first run",
              f"start moth once with a network connection ({prog()})")
    elif mode == "dev" and dev_alternative():
        alt = dev_alternative()
        size = os.path.getsize(os.path.join(alt, "model.safetensors")) / 1e6
        r.add("!", "weights", f"serve's default {folder} is missing; found {alt} ({size:.0f} MB)",
              f"pass --model {alt} (try_moth.sh picks it by itself)")
        return alt
    else:
        r.add("✗", "weights", f"no model.safetensors in {folder}",
              "pass --model DIR to moth" if mode == "dev" else
              f"download again: hf download {weights.repo()} --local-dir <this folder>")
    return folder


def check_model(r: Report, spec: str | None = None, timeout: float = 2.0):
    """The model behind moth (`moth config model`, moth/backends): backend, model, character. Returns its manifest when
    it isn't this install's own moth (then the weights/tokenizer lines are skipped), else None. Never downloads; an
    openai endpoint is asked GET /v1/models only when it is on this machine (doctor never contacts another host)."""
    try:
        from moth import backends, config
        spec = config.model() if spec is None else spec
    except Exception as e:  # noqa: BLE001
        r.add("!", "model", f"couldn't read the model setting: {e}")
        return None
    if not spec:
        r.add("✓", "model", f"moth's own (XS on tiny-moe; the card says who it is)  ({prog()} config model ... swaps "
                            "it)")
        return None
    try:
        m = backends.resolve(spec, network=False)
    except Exception as e:  # noqa: BLE001  ManifestError, an unreadable folder
        r.add("✗", "model", f"{spec}: {e}", f"fix its moth.json, or {prog()} config model default")
        return None
    for n in m.notes:
        r.add("!", "model", f"moth.json: {n}")
    what = m.describe()
    if m.backend in ("tiny-agent", "tiny-moe"):
        st = os.path.join(m.folder or "", "model.safetensors")
        if m.folder and os.path.isfile(st) and safetensors_complete(st):
            r.add("✓", "model", what)
        elif m.repo:
            r.add("!", "model", f"{what}: not downloaded yet; moth fetches it from {m.repo} on its next start",
                  f"start moth once with a network connection ({prog()})")
        else:
            r.add("✗", "model", f"{what}: no complete model.safetensors in {m.folder}",
                  f"{prog()} config model <a weights folder> (or default)")
    elif m.backend == "transformers":
        if importlib.util.find_spec("transformers") is None:
            r.add("✗", "model", f"{what}: transformers isn't installed",
                  'pip install "moth-companion[transformers]"   (or pip install transformers)')
        else:
            r.add("✓", "model", what)
    else:
        note = backends.privacy_note(m)
        if note:   # never contacted from here
            r.add("!", "model", what)
            r.add("!", "privacy", note, f"run the server on this machine, or {prog()} config model default")
        else:
            from moth.backends.openai_compat import chat_url
            models = chat_url(m.base_url)[:-len("/chat/completions")] + "/models"
            try:
                import urllib.request
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(models, timeout=timeout) as resp:
                    resp.read(1 << 16)
                r.add("✓", "model", f"{what}  (answers at {models}; moth's text stays on this machine)")
            except Exception as e:  # noqa: BLE001
                r.add("!", "model", f"{what}: nothing answers at {models} yet ({type(e).__name__})",
                      "start the server (llama-server / ollama serve / vllm serve) before moth")
    return m


def dev_alternative() -> str | None:
    """The dev checkout's newest complete weights, as try_moth.sh picks them."""
    from moth import weights
    for c in ("runs/final2/ckpt_100", "runs/final/ckpt_100", "runs/v2/model"):
        d = os.path.join(weights.ROOT, c)
        if os.path.isfile(os.path.join(d, "config.json")) and safetensors_complete(os.path.join(d, "model.safetensors")):
            return d
    return None


def check_tokenizer(r: Report):
    try:
        mode, _, tok = weights_paths()
    except Exception as e:
        r.add("✗", "tokenizer", f"can't work out where it is: {e}")
        return
    if not os.path.isfile(tok):
        if mode == "pip":
            r.add("!", "tokenizer", f"not downloaded yet ({tok}); comes with the weights on first run")
        else:
            r.add("✗", "tokenizer", f"missing: {tok}", "pass --tokenizer FILE" if mode == "dev" else f"{prog()} update")
        return
    try:
        from tokenizers import Tokenizer
        n = Tokenizer.from_file(tok).get_vocab_size()
        r.add("✓", "tokenizer", f"{n} tokens  ({tok})")
    except ImportError:
        r.add("!", "tokenizer", f"{tok} (not loaded: tokenizers isn't installed)")
    except Exception as e:
        r.add("✗", "tokenizer", f"{tok} won't load: {str(e)[:100]}", f"{prog()} update")


def token_path() -> str:   # same rule as serve.token_path(), without importing serve (torch)
    return os.environ.get("MOTH_TOKEN_FILE") or os.path.join(os.path.expanduser("~"), ".moth", "token")


def check_token(r: Report):
    p = token_path()
    try:
        with open(p) as f:
            tok = f.read().strip()
    except FileNotFoundError:
        r.add("!", "token", f"no {p} yet: moth makes it on its first run (the hooks read it)",
              f"start moth once ({prog()})")
        return
    except OSError as e:
        r.add("✗", "token", f"can't read {p}: {e}", f"check the file's permissions (ls -l {p})")
        return
    if not tok:
        r.add("✗", "token", f"{p} is empty", f"delete it; moth makes a new one on its next start")
        return
    if os.name != "nt" and os.stat(p).st_mode & 0o077:
        r.add("!", "token", f"{p} is readable by other users", f"chmod 600 {p}")
        return
    r.add("✓", "token", p)


def _port_owner(port: int) -> str:
    """'pid 123 (name)' via ss on Linux, else ''."""
    if not sys.platform.startswith("linux") or not shutil.which("ss"):
        return ""
    try:
        out = subprocess.run(["ss", "-Hltnp", f"sport = :{port}"], capture_output=True, text=True, timeout=3).stdout
    except (OSError, subprocess.SubprocessError):
        return ""
    import re
    m = re.search(r'\("([^"]+)",pid=(\d+)', out)
    return f"pid {m.group(2)} ({m.group(1)})" if m else ""


def check_port(r: Report, port: int):
    from moth import update
    h = update._health(port)
    if isinstance(h, dict) and h.get("ok"):
        here = os.path.realpath(str(h.get("install_root", ""))) == os.path.realpath(update.ROOT)
        r.add("✓", "port", f"{port}: moth {h.get('version', '?')} is running (pid {h.get('pid')}"
                           f"{'' if here else ', from ' + str(h.get('install_root'))})")
        return
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.5)
    try:
        busy = s.connect_ex(("127.0.0.1", port)) == 0
    finally:
        s.close()
    if busy:
        who = _port_owner(port)
        r.add("✗", "port", f"{port} is held by another program{' ' + who if who else ''}, not moth",
              f"stop it, or run moth with --port N and point the hooks at it (MOTH_URL=http://127.0.0.1:N)")
    else:
        r.add("·", "port", f"{port} is free: moth isn't running (start it with {prog()})")


def check_hooks(r: Report):
    try:
        from moth import hooks
        lines = hooks.status_lines(only_detected=True)
    except Exception as e:
        r.add("✗", "hooks", f"couldn't check: {e}")
        return
    if not lines:
        r.add("·", "hooks", "no coding agents found on this machine (moth chats fine without one)")
        return
    for i, (mark, text) in enumerate(lines):
        name, _, rest = text.partition(" ")
        r.add(mark if mark != "·" else "·", "hooks" if i == 0 else "", f"{name} {rest.strip()}")
    r.add("·", "", f"details: {prog()} hooks")


def check_pyside(r: Report):
    if importlib.util.find_spec("PySide6") is None:
        r.add("!", "pet (PySide6)", "not installed: the desktop pet needs it (the chat page and moth tail don't)",
              "pip install PySide6")
        return
    try:
        from importlib.metadata import PackageNotFoundError, version
        try:
            v = version("PySide6")
        except PackageNotFoundError:   # the pet extra installs PySide6-Essentials alone
            v = version("PySide6-Essentials")
        r.add("✓", "pet (PySide6)", v)
    except Exception:
        r.add("✓", "pet (PySide6)", "installed")


def check_display(r: Report, env=None, platform: str | None = None):
    env = os.environ if env is None else env
    platform = platform or sys.platform
    if platform == "darwin":
        r.add("✓", "display", "macOS")
        return
    if platform.startswith("win"):
        r.add("✓", "display", "Windows")
        return
    wayland = env.get("WAYLAND_DISPLAY") or env.get("XDG_SESSION_TYPE") == "wayland"
    x11 = env.get("DISPLAY")
    if wayland and x11:
        r.add("✓", "display", f"Wayland, with XWayland (DISPLAY={x11}): the pet can place itself")
    elif wayland:
        r.add("!", "display", "Wayland without XWayland: the pet runs, but can't place or move its own window",
              "install / enable XWayland (most desktops have it)")
    elif x11:
        r.add("✓", "display", f"X11 (DISPLAY={x11})")
    else:
        r.add("!", "display", "no graphical session here: the pet needs one (the server, chat page and moth tail "
                              "work without)")


def check_fullscreen(r: Report, env=None, platform: str | None = None, on: bool | None = None):
    """Whether the pet can tell that another app is fullscreen (config hide_in_fullscreen; moth/pet.py probes)."""
    env = os.environ if env is None else env
    platform = platform or sys.platform
    if on is None:
        try:
            from moth import config
            on = config.hide_in_fullscreen()
        except Exception:  # noqa: BLE001
            on = True
    if not on:
        r.add("·", "fullscreen", f"off: the pet stays up over fullscreen apps  ({prog()} config hide_in_fullscreen on)")
        return
    wayland = env.get("WAYLAND_DISPLAY") or env.get("XDG_SESSION_TYPE") == "wayland"
    if platform == "darwin" or platform.startswith("win"):
        r.add("✓", "fullscreen", "the pet hides while another app is fullscreen on its screen")
    elif wayland and env.get("DISPLAY"):
        r.add("·", "fullscreen", "XWayland: the pet hides for fullscreen X11 apps (Wine / Proton games), not for "
                                 "native-Wayland ones (can't be seen: it stays up)")
    elif wayland:
        r.add("·", "fullscreen", "native Wayland can't tell the pet what's fullscreen, so it never hides (fails open)")
    elif env.get("DISPLAY"):
        r.add("✓", "fullscreen", "X11: the pet hides while another app is fullscreen on its screen")


def available_ram_gb() -> float | None:
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) / 1048576
        if sys.platform == "darwin":
            out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=3).stdout
            import re
            page = int(re.search(r"page size of (\d+)", out).group(1))
            pages = sum(int(re.search(rf"{k}:\s+(\d+)", out).group(1))
                        for k in ("Pages free", "Pages inactive", "Pages speculative"))
            return pages * page / 2**30
        if os.name == "nt":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MS()
            m.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.ullAvailPhys / 2**30
    except Exception:
        return None
    return None


def check_ram(r: Report, gb: float | None = None, running: bool = False):
    gb = available_ram_gb() if gb is None else gb
    if gb is None:
        r.add("·", "RAM", "couldn't read free memory on this system")
    elif gb < MIN_RAM_GB and not running:
        r.add("!", "RAM", f"{gb:.1f} GB available: moth needs about 3 GB while it loads",
              "close some programs before starting moth")
    else:
        r.add("✓", "RAM", f"{gb:.1f} GB available" + (" (moth already loaded)" if running else ""))


def check_disk(r: Report, folder: str | None, have_weights: bool):
    p = folder or os.path.expanduser("~")
    while p and not os.path.isdir(p) and os.path.dirname(p) != p:
        p = os.path.dirname(p)
    try:
        free = shutil.disk_usage(p).free / 2**30
    except OSError as e:
        r.add("·", "disk", f"couldn't check {p}: {e}")
        return
    need = 2.0 if have_weights else 1.0   # an update keeps a backup of the replaced weights; a download needs ~0.9
    if free < need:
        r.add("✗" if not have_weights else "!", "disk", f"{free:.1f} GB free at {p}",
              f"free up space: {'an update keeps a backup of the old weights (~0.9 GB)' if have_weights else 'the weights need about 0.9 GB'}")
    else:
        r.add("✓", "disk", f"{free:.1f} GB free at {p}")


def check_setting(r: Report):
    try:
        from moth import config
        value, source = config.lookup("show_in_harness")
        where = {"env": f"from {config.ENV['show_in_harness']}", "file": f"in {config.path()}",
                 "default": "default"}[source]
        r.add("✓", "show_in_harness", f"{'on' if value else 'off'} ({where}): moth's replies "
                                      f"{'also show' if value else 'show only in the pet / moth tail, not'} inside "
                                      f"Pi / OpenCode / Codex" + ("" if value else f"  ({prog()} config show_in_harness on)"))
    except Exception as e:
        r.add("✗", "show_in_harness", f"couldn't read the setting: {e}", f"{prog()} config")


def check_version(r: Report):
    try:
        from moth import update
        r.add("✓", "version", update.version_line(), None)
    except Exception as e:
        r.add("!", "version", f"unknown ({e})")


def run(port: int = 4747, probe=probe_torch) -> Report:
    r = Report()
    check_version(r)
    check_python(r)
    check_torch(r, probe)
    check_packages(r)
    custom = check_model(r)
    if custom is None:   # this install's own moth: its weights and tokenizer
        folder = check_weights(r)
        check_tokenizer(r)
    else:
        folder = custom.folder if custom.backend == "tiny-agent" else None
    check_token(r)
    check_port(r, port)
    running = any(name == "port" and m == "✓" for m, name, *_ in r.lines)
    check_hooks(r)
    check_pyside(r)
    check_display(r)
    check_fullscreen(r)
    check_ram(r, running=running)
    check_disk(r, folder, bool(folder) and os.path.isfile(os.path.join(folder, "model.safetensors")))
    check_setting(r)
    return r


def _mark(m: str) -> str:
    try:
        m.encode(getattr(sys.stdout, "encoding", None) or "ascii")
        return m
    except (UnicodeEncodeError, LookupError):
        return {"✓": "ok", "✗": "XX", "!": "!!", "·": "..", "-": "--"}.get(m, m)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog=f"{prog()} doctor", description="check that moth can run here (no model load)")
    ap.add_argument("--port", type=int, default=4747, help="the port moth serves on (default 4747)")
    a = ap.parse_args(argv)
    r = run(a.port)
    print(r.render(_mark))
    print(f"\n{'something needs fixing (✗ above)' if r.failed else 'all good'}".replace("✗", _mark("✗")))
    return 1 if r.failed else 0


if __name__ == "__main__":
    sys.exit(main())
