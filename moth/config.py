"""moth's own settings, in ~/.moth/config.json (MOTH_CONFIG_FILE overrides the path). Stdlib only: the harness
adapters and `python -m moth config` read it without torch.

  python -m moth config                          print the current settings and where each comes from
  python -m moth config show_in_harness on|off   change one (written atomically; other keys are kept)
  python -m moth config pet_skin <petdex id | folder | moth>

Settings:
  show_in_harness  (default off)  show moth's replies INSIDE the coding harness: the Pi widget above the editor and
                   the OpenCode toast. Off, the adapters still send each agent reply to moth (the desktop pet and
                   `moth tail` react as usual); they just don't draw anything in the harness. The adapters read the
                   file at every turn, so a change applies from the next turn, no restart.
                   MOTH_SHOW_IN_HARNESS=1/0 in the harness's environment beats the file (fixed for that process).
  pet_skin         (default moth)  what the desktop pet looks like: moth, the id of an installed Petdex pet
                   (~/.petdex/pets/<id>, $CODEX_HOME/pets/<id>, <hermes home>/pets/<id>) or a pet folder. Read when
                   the pet starts; its right-click menu "skin" changes it live. `python -m moth.pet --skin X` beats it.
  update_check     (default on)  the desktop pet checks for a new moth once a day and says so in a bubble (it never
                   updates anything itself). Skipped when HF_HUB_OFFLINE=1 or MOTH_OFFLINE=1.

Values: on/off, 1/0, true/false, yes/no (any case), or a JSON true/false in the file. The Pi and OpenCode adapters
use the same rule (their `enabled()`). pet_skin takes any text ("default" resets it to moth).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

DEFAULTS = {"show_in_harness": False, "pet_skin": "moth", "update_check": True}
TEXT = {"pet_skin"}   # text settings; the rest are on/off
ENV = {"show_in_harness": "MOTH_SHOW_IN_HARNESS"}
TRUE = {"1", "on", "true", "yes"}
FALSE = {"0", "off", "false", "no"}
LAST_UPDATE_CHECK = "last_update_check"   # the pet's bookkeeping (unix time), not a setting: not settable here

USAGE = """usage: python -m moth config                          print the current settings
       python -m moth config show_in_harness on|off   show moth's replies inside Pi / OpenCode (default off)
       python -m moth config pet_skin <id|folder|moth> the desktop pet's look: an installed Petdex pet (default moth)
       python -m moth config update_check on|off      the pet's daily "new moth out" bubble (default on)
"""


def path() -> str:
    return os.environ.get("MOTH_CONFIG_FILE") or os.path.join(os.path.expanduser("~"), ".moth", "config.json")


def parse_bool(value) -> bool | None:
    """True / False for a recognised spelling, None for anything else."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in TRUE:
            return True
        if v in FALSE:
            return False
    return None


def read_file() -> dict:
    """The file's JSON object; {} when it is missing, unreadable or not an object."""
    try:
        with open(path(), encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def parse_text(value) -> str | None:
    """A text setting: stripped, one line, not empty; None otherwise."""
    if isinstance(value, str) and value.strip() and "\n" not in value and len(value) < 1024:
        return value.strip()
    return None


def lookup(key: str):
    """(value, source) with source "env", "file" or "default"."""
    parse = parse_text if key in TEXT else parse_bool
    env = parse(os.environ.get(ENV[key], "")) if key in ENV else None
    if env is not None:
        return env, "env"
    v = parse(read_file().get(key))
    if v is not None:
        return v, "file"
    return DEFAULTS[key], "default"


def show_in_harness() -> bool:
    return lookup("show_in_harness")[0]


def pet_skin() -> str:
    return lookup("pet_skin")[0]


def update_check() -> bool:
    return lookup("update_check")[0]


def offline() -> bool:
    """HF_HUB_OFFLINE=1 (what moth's weights download honours) or MOTH_OFFLINE=1: nothing goes to the network."""
    return any(parse_bool(os.environ.get(k, "")) for k in ("HF_HUB_OFFLINE", "MOTH_OFFLINE"))


def last_update_check() -> float:
    v = read_file().get(LAST_UPDATE_CHECK)
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0.0


def write(key: str, value) -> None:
    """Set one key, keeping the others: temp file in the same folder, fsync, rename over the old one."""
    p = path()
    data = read_file()
    data[key] = value
    folder = os.path.dirname(os.path.abspath(p))
    os.makedirs(folder, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".config-", suffix=".tmp", dir=folder)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def describe() -> str:
    lines = [f"moth settings ({path()}):"]
    for key in DEFAULTS:
        value, source = lookup(key)
        where = {"env": f"from {ENV.get(key)}", "file": "from the file", "default": "default"}[source]
        lines.append(f"  {key} = {value if key in TEXT else ('on' if value else 'off')}   ({where})")
    return "\n".join(lines)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if any(a in ("-h", "--help") for a in argv):
        print(USAGE + "\n" + __doc__.split("Settings:", 1)[1].split("Values:", 1)[0].rstrip())
        return 0
    if not argv:
        print(describe())
        return 0
    if len(argv) != 2 or argv[0] not in DEFAULTS:
        print(USAGE, end="", file=sys.stderr)
        return 2
    key, raw = argv
    if key in TEXT:
        value = parse_text(raw)
        if value is None:
            print(f"{key} takes a name or a folder, not {raw!r}", file=sys.stderr)
            return 2
        if key == "pet_skin" and value.lower() == "default":
            value = DEFAULTS[key]
        elif key == "pet_skin" and ("/" in value or os.sep in value or value.startswith(("~", "."))):
            value = os.path.abspath(os.path.expanduser(value))   # a folder: the pet may start from another cwd
    else:
        value = parse_bool(raw)
        if value is None:
            print(f"{key} takes on or off, not {raw!r}", file=sys.stderr)
            return 2
    try:
        write(key, value)
    except OSError as e:
        print(f"couldn't write {path()}: {e}", file=sys.stderr)
        return 1
    if key == "pet_skin":
        print(f"pet_skin = {value}   (saved in {path()}; the desktop pet uses it from its next start)")
        note = _skin_note(value)
        if note:
            print(note)
        return 0
    if key == "update_check":
        print(f"update_check = {'on' if value else 'off'}   (saved in {path()})")
        return 0
    print(f"{key} = {'on' if value else 'off'}   (saved in {path()}; harnesses pick it up from the next turn)")
    env = os.environ.get(ENV[key], "")
    if parse_bool(env) is not None and parse_bool(env) != value:
        print(f"note: {ENV[key]}={env} is set in this shell and wins over the file in harnesses started from it")
    return 0


def _skin_note(value: str) -> str | None:
    """A warning when the pet won't find that skin (it would fall back to moth); None when it will."""
    try:
        from moth import pet   # stdlib at import time (PySide6 only when drawing)
        if value == "moth" or pet.find_skin(value):
            return None
        names = ", ".join(i for i, _ in pet.installed_skins()) or "none installed"
    except Exception:  # noqa: BLE001  a note, never a failure
        return None
    return f"note: no Petdex pet {value!r} found (installed: {names}); the pet will be moth until it is installed"


if __name__ == "__main__":
    sys.exit(main())
