"""moth autostart on|off|status : start the desktop pet at login (the pet wakes the server up itself).

  Linux    an XDG autostart entry, $XDG_CONFIG_HOME/autostart/moth-pet.desktop (~/.config/autostart/). KDE, GNOME,
           XFCE, Cinnamon, MATE, LXQt all start these with the full graphical session environment (DISPLAY,
           WAYLAND_DISPLAY, the D-Bus address), which a systemd user unit would have to import first, so it is the
           more reliable choice for a GUI app.
  macOS    a LaunchAgent, ~/Library/LaunchAgents/local.moth.pet.plist (RunAtLoad, output to ~/.moth/pet.log).
  Windows  a moth-pet.bat in the Startup folder (%APPDATA%\\Microsoft\\Windows\\Start Menu\\Programs\\Startup),
           starting pythonw (no console window) when it exists.

The entry runs this Python (sys.executable) with -m moth.pet, from the folder that holds the moth package, so it
works for a pip install, the Hugging Face download (code/) and a checkout alike. `status` also says when the entry
points at another Python or folder than this one (`on` again rewrites it). `off` deletes only moth's own entry.
"""
from __future__ import annotations

import argparse
import os
import plistlib
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)   # the folder that holds the moth package
ICON = os.path.join(HERE, "assets", "silkmoth_adult_female_icon.svg")
LABEL = "local.moth.pet"
MARKER = "X-Moth-Autostart=1"
# the installers' app (PyInstaller, sys.frozen): moth is inside the executable, so the entry runs from the home folder
# (an AppImage's mount, where CODE and ICON are, moves on every launch) and names the icon by its theme name
FROZEN = bool(getattr(sys, "frozen", False))
RUN_DIR = os.path.expanduser("~") if FROZEN else CODE
if FROZEN:
    from moth.hooks import appimage   # stdlib only
    if appimage():
        ICON = "moth"


def prog() -> str:
    return "moth" if os.path.basename(sys.argv[0]) in ("moth", "moth.exe") else "python -m moth"


def kind(platform: str | None = None) -> str:
    platform = platform or sys.platform
    return "mac" if platform == "darwin" else "windows" if platform.startswith("win") or platform == "cygwin" \
        else "linux"


def entry_path(platform: str | None = None) -> str:
    k, home = kind(platform), os.path.expanduser("~")
    if k == "mac":
        return os.path.join(home, "Library", "LaunchAgents", LABEL + ".plist")
    if k == "windows":
        appdata = os.environ.get("APPDATA") or os.path.join(home, "AppData", "Roaming")
        return os.path.join(appdata, "Microsoft", "Windows", "Start Menu", "Programs", "Startup", "moth-pet.bat")
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(home, ".config")
    return os.path.join(base, "autostart", "moth-pet.desktop")


def pet_python(platform: str | None = None) -> str:
    """sys.executable; on Windows its pythonw.exe sibling (no console window) if there is one. The installers' app:
    its windowed mothw.exe on Windows, the .AppImage file itself ($APPIMAGE) rather than its per-launch mount."""
    py = sys.executable
    if FROZEN:
        from moth.hooks import appimage
        if appimage():
            return appimage()
    if kind(platform) == "windows":
        for name in ("mothw.exe",) if FROZEN else ("pythonw.exe",):
            w = os.path.join(os.path.dirname(py), name)
            if os.path.isfile(w):
                return w
    return py


def _desktop_quote(arg: str) -> str:
    """One Exec= argument (Desktop Entry spec: quote if needed; \\ " ` $ escaped inside quotes; % doubled)."""
    arg = arg.replace("%", "%%")
    if arg and not any(c in arg for c in ' \t\n"\'\\><~|&;$*?#()`'):
        return arg
    return '"' + "".join("\\" + c if c in '"`$\\' else c for c in arg) + '"'


def render(platform: str | None = None) -> str:
    k, py = kind(platform), pet_python(platform)
    if k == "mac":
        log = os.path.join(os.path.expanduser("~"), ".moth", "pet.log")
        return plistlib.dumps({"Label": LABEL, "ProgramArguments": [py, "-m", "moth.pet"],
                               "WorkingDirectory": RUN_DIR, "RunAtLoad": True, "KeepAlive": False,
                               "ProcessType": "Interactive", "StandardOutPath": log,
                               "StandardErrorPath": log}).decode()
    if k == "windows":
        return ("@echo off\r\nrem moth: start the desktop pet at login (made by `moth autostart on`; "
                "`moth autostart off` removes it)\r\n"
                f'cd /d "{RUN_DIR}"\r\nstart "" "{py}" -m moth.pet\r\n')
    return "\n".join([
        "[Desktop Entry]", "Type=Application", "Name=moth",
        "Comment=moth, a tiny companion, flying around your screen (desktop pet)",
        "Exec=" + " ".join(_desktop_quote(a) for a in (py, "-m", "moth.pet")),
        "Path=" + RUN_DIR, "Icon=" + ICON, "Terminal=false", "X-GNOME-Autostart-enabled=true",
        "X-GNOME-Autostart-Delay=5", "X-KDE-autostart-after=panel", MARKER, ""])


def is_moth(path: str, platform: str | None = None) -> bool:
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return False
    if kind(platform) == "mac":
        try:
            return plistlib.loads(data).get("Label") == LABEL
        except Exception:
            return False
    text = data.decode("utf-8", "replace")
    return MARKER in text if kind(platform) == "linux" else "moth autostart" in text and "moth.pet" in text


def state(platform: str | None = None) -> tuple[str, str]:
    """('off' | 'on' | 'stale' | 'foreign', path)."""
    p = entry_path(platform)
    if not os.path.exists(p):
        return "off", p
    if not is_moth(p, platform):
        return "foreign", p
    with open(p, encoding="utf-8", errors="replace") as f:
        current = f.read()
    return ("on" if current.replace("\r\n", "\n") == render(platform).replace("\r\n", "\n") else "stale"), p


def write(path: str, text: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".moth-tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    os.replace(tmp, path)


def main(argv=None, platform: str | None = None) -> int:
    ap = argparse.ArgumentParser(prog=f"{prog()} autostart",
                                 description="start moth's desktop pet when you log in (it wakes the server up)")
    ap.add_argument("action", choices=("on", "off", "status"), nargs="?", default="status")
    a = ap.parse_args(argv)
    st, p = state(platform)
    if a.action == "status":
        print({"off": f"autostart is off (no {p})",
               "on": f"autostart is on: {p}\n  starts: {pet_python(platform)} -m moth.pet  (in {RUN_DIR})",
               "stale": f"autostart is on, but {p} starts another Python or folder than this one "
                        f"(`{prog()} autostart on` points it here)",
               "foreign": f"{p} exists but wasn't made by moth"}[st])
        return 0
    if a.action == "on":
        if st == "foreign":
            print(f"{p} exists and wasn't made by moth; not touching it")
            return 1
        if st == "on":
            print(f"autostart is already on ({p})")
            return 0
        if kind(platform) == "mac":   # launchd won't create the log's folder
            os.makedirs(os.path.join(os.path.expanduser("~"), ".moth"), exist_ok=True)
        write(p, render(platform))
        print(f"{'updated' if st == 'stale' else 'autostart on'}: {p}\n  at login: {pet_python(platform)} -m moth.pet "
              f"(the pet starts moth's server if it isn't running)")
        if kind(platform) == "mac":
            print(f"  takes effect at your next login (or now: launchctl load {p})")
        return 0
    if st == "off":
        print(f"autostart is already off (no {p})")
        return 0
    if st == "foreign":
        print(f"{p} wasn't made by moth; not deleting it")
        return 1
    os.remove(p)
    print(f"autostart off: removed {p}" + (f"\n  (a pet already running keeps running; to unload the agent now: "
                                           f"launchctl remove {LABEL})" if kind(platform) == "mac" else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
