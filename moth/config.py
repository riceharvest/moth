"""moth's own settings, in ~/.moth/config.json (MOTH_CONFIG_FILE overrides the path). Stdlib only: the harness
adapters and `python -m moth config` read it without torch.

  python -m moth config                          print the current settings and where each comes from
  python -m moth config show_in_harness on|off   change one (written atomically; other keys are kept)
  python -m moth config pet_skin <petdex id | folder | moth>

Settings:
  show_in_harness  (default off)  show moth's replies INSIDE the coding harness: the Pi widget above the editor,
                   the OpenCode toast and a "moth: ..." line under the turn in the Codex TUI. Off, the adapters still send each agent reply to moth (the desktop pet and
                   `moth tail` react as usual); they just don't draw anything in the harness. The adapters read the
                   file at every turn, so a change applies from the next turn, no restart.
                   MOTH_SHOW_IN_HARNESS=1/0 in the harness's environment beats the file (fixed for that process).
  pet_skin         (default moth)  what the desktop pet looks like: moth, the id of an installed Petdex pet
                   (~/.petdex/pets/<id>, $CODEX_HOME/pets/<id>, <hermes home>/pets/<id>) or a pet folder. Read when
                   the pet starts; its right-click menu "skin" changes it live. `python -m moth.pet --skin X` beats it.
  update_check     (default on)  the desktop pet checks for a new moth once a day and says so in a bubble (it never
                   updates anything itself). Skipped when HF_HUB_OFFLINE=1 or MOTH_OFFLINE=1.
  hide_in_fullscreen (default on)  the pet hides while another app is fullscreen on its screen (a game, a video)
                   and pops no bubbles; what moth says meanwhile waits in the menu's "what did i miss".
  chirp            (default off)  a soft sound with the pet's "your agent is done" hop.
  muted_sources    (default none)  harnesses whose reactions pop no bubble and no hop (they still land in "what
                   did i miss"): a comma list of claude-code, codex, hermes, pi, opencode, dsh; "none" clears it.
  pet_screen       (default any)  the screen the pet stays on (its name, as the pet's menu "stay on this screen"
                   saves it); "any" = wherever it is put.
  quiet_hours      (default off)  e.g. 23:00-08:00 (may cross midnight): no bubbles, no chirp, the moth dozes.
  model            (default default = this install's moth)  the model behind moth: a weights folder (its moth.json
                   says how to run it), an HF repo id, or openai:<base_url>#<model> for a local llama.cpp / Ollama /
                   vLLM server. Used from moth's next start; `--model` beats it. moth/docs/MODEL_BACKENDS.md.

Values: on/off, 1/0, true/false, yes/no (any case), or a JSON true/false in the file. The Pi, OpenCode and Codex adapters
use the same rule (their `enabled()`). pet_skin takes any text ("default" resets it to moth).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

DEFAULTS = {"show_in_harness": False, "pet_skin": "moth", "update_check": True, "hide_in_fullscreen": True,
            "chirp": False, "muted_sources": [], "pet_screen": "any", "quiet_hours": "off", "model": "default"}
TEXT = {"pet_skin", "pet_screen", "model"}   # text settings; muted_sources is a list, quiet_hours a time span, the rest on/off
ENV = {"show_in_harness": "MOTH_SHOW_IN_HARNESS"}
TRUE = {"1", "on", "true", "yes"}
FALSE = {"0", "off", "false", "no"}
LAST_UPDATE_CHECK = "last_update_check"   # the pet's bookkeeping (unix time), not a setting: not settable here
SOURCES = ("claude-code", "codex", "hermes", "pi", "opencode", "dsh")   # the harnesses' "source" (moth/tail.py NAMES)

USAGE = """usage: python -m moth config                          print the current settings
       python -m moth config show_in_harness on|off   show moth's replies inside Pi / OpenCode / Codex (default off)
       python -m moth config pet_skin <id|folder|moth> the desktop pet's look: an installed Petdex pet (default moth)
       python -m moth config update_check on|off      the pet's daily "new moth out" bubble (default on)
       python -m moth config hide_in_fullscreen on|off  the pet hides while another app is fullscreen (default on)
       python -m moth config chirp on|off             a soft sound when an agent is done (default off)
       python -m moth config muted_sources codex,pi|none  no bubble / hop for these harnesses (default none)
       python -m moth config pet_screen <name>|any    keep the pet on one screen (default any)
       python -m moth config quiet_hours 23:00-08:00|off  no bubbles or chirp then; the moth dozes (default off)
       python -m moth config model <folder|repo id|openai:URL#model|default>  the model behind moth (default: its own)
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


def parse_sources(value) -> list | None:
    """muted_sources: a JSON list or a comma list of known harness sources ("none" / "" = []); None if any is
    unknown. Sorted, no repeats."""
    if isinstance(value, str):
        v = value.strip().lower()
        value = [] if v in ("", "none", "off") else [x.strip() for x in v.split(",") if x.strip()]
    if not isinstance(value, list) or not all(isinstance(x, str) and x.lower() in SOURCES for x in value):
        return None
    return sorted({x.lower() for x in value})


def quiet_span(value) -> tuple[int, int] | None:
    """'23:00-08:00' -> (1380, 480): minutes of the day it starts and ends (it may cross midnight); None for "off"
    or anything else (bad times, start == end)."""
    if not isinstance(value, str):
        return None
    parts = value.strip().replace(" ", "").split("-")
    if len(parts) != 2:
        return None
    out = []
    for p in parts:
        hm = p.split(":")
        if len(hm) != 2 or not all(x.isdigit() and 1 <= len(x) <= 2 for x in hm):
            return None
        h, m = int(hm[0]), int(hm[1])
        if h > 23 or m > 59:
            return None
        out.append(h * 60 + m)
    return (out[0], out[1]) if out[0] != out[1] else None


def parse_quiet(value) -> str | None:
    """quiet_hours: "off" (also off/none/false in any case) or a span, written back as HH:MM-HH:MM; None if bad."""
    if value is False or (isinstance(value, str) and value.strip().lower() in ("off", "none", "no", "false", "0")):
        return "off"
    span = quiet_span(value)
    if span is None:
        return None
    return "-".join(f"{m // 60:02d}:{m % 60:02d}" for m in span)


PARSE = {"muted_sources": parse_sources, "quiet_hours": parse_quiet}


def parser(key: str):
    return PARSE.get(key) or (parse_text if key in TEXT else parse_bool)


def show(key: str, value) -> str:
    """A value as `config` prints it."""
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, list):
        return ",".join(value) or "none"
    return str(value)


def lookup(key: str):
    """(value, source) with source "env", "file" or "default"."""
    parse = parser(key)
    env = parse(os.environ.get(ENV[key], "")) if key in ENV else None
    if env is not None:
        return env, "env"
    v = parse(read_file().get(key))
    if v is not None:
        return v, "file"
    d = DEFAULTS[key]
    return (list(d) if isinstance(d, list) else d), "default"


def show_in_harness() -> bool:
    return lookup("show_in_harness")[0]


def pet_skin() -> str:
    return lookup("pet_skin")[0]


def update_check() -> bool:
    return lookup("update_check")[0]


def hide_in_fullscreen() -> bool:
    return lookup("hide_in_fullscreen")[0]


def chirp() -> bool:
    return lookup("chirp")[0]


def muted_sources() -> list:
    return lookup("muted_sources")[0]


def pet_screen() -> str:
    return lookup("pet_screen")[0]


def model() -> str | None:
    """The model spec to run (moth/backends), or None for this install's own moth."""
    v = lookup("model")[0]
    return None if v.lower() in ("default", "moth") else v


def quiet_hours() -> str:
    return lookup("quiet_hours")[0]


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
        lines.append(f"  {key} = {show(key, value)}   ({where})")
    return "\n".join(lines)


PET_LIVE = ("hide_in_fullscreen", "chirp", "muted_sources", "pet_screen", "quiet_hours")   # the pet rereads these
HINTS = {"muted_sources": f"a comma list of {', '.join(SOURCES)}, or none",
         "quiet_hours": "a time span like 23:00-08:00, or off"}


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
    if key in PARSE:
        value = PARSE[key](raw)
        if value is None:
            print(f"{key} takes {HINTS[key]}, not {raw!r}", file=sys.stderr)
            return 2
    elif key in TEXT:
        value = parse_text(raw)
        if value is None:
            print(f"{key} takes a name or a folder, not {raw!r}", file=sys.stderr)
            return 2
        if key == "pet_skin" and value.lower() == "default":
            value = DEFAULTS[key]
        elif key == "pet_skin" and ("/" in value or os.sep in value or value.startswith(("~", "."))):
            value = os.path.abspath(os.path.expanduser(value))   # a folder: the pet may start from another cwd
        elif key == "model":
            value = _model_value(value)
        elif key == "pet_screen" and value.lower() in ("any", "default", "off", "none"):
            value = DEFAULTS[key]
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
    if key == "model":
        print(f"model = {value}   (saved in {path()}; used from moth's next start)")
        note = _model_note(value)
        if note:
            print(note)
        return 0
    if key == "update_check":
        print(f"update_check = {'on' if value else 'off'}   (saved in {path()})")
        return 0
    if key in PET_LIVE:
        print(f"{key} = {show(key, value)}   (saved in {path()}; a running desktop pet picks it up within 30 s)")
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


def _model_value(value: str) -> str:
    """default / moth -> "default"; a folder (it exists, or the text starts like a path) -> its absolute path (moth
    may start from another cwd); an HF repo id or openai:... stays as it is."""
    if value.lower() in ("default", "moth"):
        return "default"
    if value.startswith("openai:"):
        return value
    p = os.path.expanduser(value)
    if os.path.exists(p) or value.startswith((".", "~", "/", "\\")):
        return os.path.abspath(p)
    return value


def _model_note(value: str) -> str | None:
    """What that model is (offline: no download, no network), or why moth can't use it."""
    if value == "default":
        return "moth runs its own model again"
    try:
        from moth import backends   # stdlib only
        m = backends.resolve(value, network=False)
    except Exception as e:  # noqa: BLE001
        return f"note: moth can't use that yet: {e}"
    out = [f"{m.describe()}"]
    if m.backend == "tiny-agent" and m.source == "default" and m.folder and \
            not os.path.isfile(os.path.join(m.folder, "config.json")) and not m.repo:
        out.append(f"note: {m.folder} has no config.json + model.safetensors yet")
    if m.source == "spec" and backends.parse_spec(value)[0] == "hub":
        out[0] = f"{value}: an HF repo id; moth checks it for a moth.json on its next start (else: transformers)"
    if m.backend == "transformers":
        out.append('needs: pip install "moth-companion[transformers]"')
    note = backends.privacy_note(m)
    if note:
        out.append(f"privacy: {note}")
    return "\n".join(out)


if __name__ == "__main__":
    sys.exit(main())
