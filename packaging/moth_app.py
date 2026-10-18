"""The installers' entry point: PyInstaller freezes this file into moth / moth.exe + mothw.exe / moth.app.

  moth                          the app: the first-run window (weights, coding-agent hooks, autostart) until that is
                                done, then the desktop pet (which wakes moth's server up)
  moth --first-run              the first-run window again
  moth hook-run <x.py> [args]   a coding-agent hook: runs the bundled moth/integrations/x.py. `moth hook` writes the
                                small moth-hook executable instead (moth_hook.py: starts several times faster)
  moth -m <module> [args]       like `python -m` (moth/launch.py starts `<exe> -m moth.serve`, autostart `-m moth.pet`)
  moth -c <code> [args]         like `python -c` (moth setup / moth doctor probe torch and Qt that way)
  moth <script.py> [args]       like `python script.py`
  moth <anything else>          the `moth` command: setup, hooks, hook, unhook, doctor, autostart, config, forget,
                                update, --version, or serve flags (no command: the server in this terminal)

Local weights (while the model isn't on the Hub, or offline): $MOTH_MODEL_DIR, or the folder picked in the first-run
window (saved in ~/.moth/app.json), a folder with model.safetensors + config.json + tokenizer.json. It goes to the pet
and the server as --model (moth/launch.py takes the tokenizer from that folder too).
Windows: moth.exe is the console program (CLI and hooks), mothw.exe the windowed one (Start menu, autostart, the pet
and the server it starts).
"""
import os
import sys

APP_JSON = os.path.join(os.path.expanduser("~"), ".moth", "app.json")
GUI_FLAGS = ("--first-run",)


def bundle() -> str:
    """Where PyInstaller put moth/ (sys._MEIPASS); the checkout when run unfrozen (tests, local tries)."""
    return getattr(sys, "_MEIPASS", None) or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --- hooks: moth_hook.py (stdlib only) ------------------------------------------------------------------------
def hook_run(argv: list) -> int:
    from moth_hook import hook_run as run
    return run(argv)


# --- local weights -------------------------------------------------------------------------------------------------
def app_state() -> dict:
    import json
    try:
        with open(APP_JSON, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def save_state(**kv) -> None:
    import json
    d = app_state()
    d.update(kv)
    os.makedirs(os.path.dirname(APP_JSON), exist_ok=True)
    tmp = APP_JSON + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=1)
        f.write("\n")
    os.replace(tmp, APP_JSON)


def model_dir() -> str | None:
    d = os.environ.get("MOTH_MODEL_DIR") or app_state().get("model_dir")
    return os.path.abspath(os.path.expanduser(d)) if d else None


def usable(folder: str | None) -> bool:
    from moth import weights   # stdlib only
    return bool(folder) and weights.complete(folder) and os.path.isfile(os.path.join(folder, "tokenizer.json"))


def with_model(args: list, serve: bool = False) -> list:
    """args plus --model (and the server's --tokenizer) for the local weights folder, unless args name one."""
    d = model_dir()
    if not d or "--model" in args or any(a.startswith("--model=") for a in args):
        return args
    extra = ["--model", d] + (["--tokenizer", os.path.join(d, "tokenizer.json")] if serve else [])
    if "--" in args:   # the pet: flags after a bare -- go to its server
        i = args.index("--")
        return args[:i] + extra + args[i:]
    return extra + args


def ready() -> bool:
    """Weights the pet can use: the local folder, else a finished download in the cache."""
    from moth import weights
    d = model_dir()
    if d:
        return usable(d)
    return weights.present(weights.cache_dir()) and weights.complete(weights.cache_dir())


# --- moth setup in the app -----------------------------------------------------------------------------------------
def patch_setup():
    """moth/setup.py, adapted to the app without changing it: packages are bundled (no pip hints), weights may be a
    local folder, and every hint says `moth`."""
    from moth import setup
    setup.prog = lambda: "moth"

    def packages(self):
        self.head(1, "packages")
        v = sys.version_info
        self.line("✓", f"bundled in this app: python {v.major}.{v.minor}.{v.micro}, torch, tokenizers, PySide6  "
                       f"({sys.executable})")
        self.had.append("packages")

    plain = setup.Setup.weights_step

    def weights_step(self):
        d = model_dir()
        if not d:
            return plain(self)
        self.head(2, "weights")
        if usable(d):
            self.weights_ok = True
            self._have(d, "your folder" + (", MOTH_MODEL_DIR" if os.environ.get("MOTH_MODEL_DIR") else ""))
        else:
            self.weights_ok = False
            self.line("✗", f"{d} needs model.safetensors (complete), config.json and tokenizer.json")
            self.left.append("weights (pick another folder)")

    setup.Setup.packages, setup.Setup.weights_step = packages, weights_step
    return setup


# --- dispatch ------------------------------------------------------------------------------------------------------
def run_module(mod: str, args: list) -> None:
    import runpy
    if mod == "moth.pet":
        args = with_model(args)
    elif mod == "moth" and args[:1] == ["setup"]:
        patch_setup()
    sys.argv = [mod] + list(args)
    runpy.run_module(mod, run_name="__main__", alter_sys=True)


def _quiet_streams():
    """The windowed exe (Windows), or an app opened from Finder: no stdout. Log to ~/.moth/app.log instead of None.
    A log file or pipe gets UTF-8 (a frozen app ignores PYTHONIOENCODING, and Windows would write cp1252, which has
    no ✓); a console keeps its encoding, never failing on a character it lacks."""
    for s in (sys.stdout, sys.stderr):
        try:
            if s is not None:
                s.reconfigure(errors="replace") if s.isatty() else s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            pass
    if sys.stdout is not None and sys.stderr is not None:
        return
    try:
        os.makedirs(os.path.dirname(APP_JSON), exist_ok=True)
        f = open(os.path.join(os.path.dirname(APP_JSON), "app.log"), "a", buffering=1, encoding="utf-8",
                 errors="replace")
    except OSError:
        f = open(os.devnull, "w")
    sys.stdout = sys.stdout or f
    sys.stderr = sys.stderr or f


def main() -> int:
    argv = [a for a in sys.argv[1:] if not a.startswith("-psn_")]   # old macOS Finder adds -psn_<pid>
    if argv[:1] == ["hook-run"]:
        return hook_run(argv[1:])
    _quiet_streams()
    if os.environ.get("MOTH_MODEL_DIR"):   # absolute, for the pet and server started from other folders
        os.environ["MOTH_MODEL_DIR"] = os.path.abspath(os.path.expanduser(os.environ["MOTH_MODEL_DIR"]))
    if argv[:1] == ["-m"] and len(argv) > 1:
        run_module(argv[1], argv[2:])
        return 0
    if argv[:1] == ["-c"] and len(argv) > 1:
        sys.argv = ["-c"] + argv[2:]
        exec(compile(argv[1], "<string>", "exec"), {"__name__": "__main__"})
        return 0
    if argv and argv[0].endswith(".py") and os.path.isfile(argv[0]):
        import runpy
        sys.argv = list(argv)
        sys.path.insert(0, os.path.dirname(os.path.abspath(argv[0])))
        runpy.run_path(argv[0], run_name="__main__")
        return 0
    sys.argv[0] = "moth"   # moth's own hints (prog()) then say `moth ...`
    if not argv or argv[0] in GUI_FLAGS:
        if argv[:1] == ["--first-run"] or not (ready() and app_state().get("first_run_done")):
            import moth_firstrun
            return moth_firstrun.main()
        run_module("moth.pet", [])
        return 0
    if argv[0] == "setup":
        return patch_setup().main(argv[1:])
    from moth import __main__ as cli
    known = ("config", "update", "hooks", "hook", "unhook", "doctor", "autostart", "forget")
    if argv[0] == "serve":
        argv = argv[1:]
    cli.main(argv if argv[0] in known or "--version" in argv else with_model(argv, serve=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
