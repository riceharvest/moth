"""moth hooks / moth hook / moth unhook : see and change which coding agents tell moth about their turns.

  moth hooks [--project DIR]                 harnesses found on this machine, and whether moth is hooked into each
  moth hook <harness ...> | --all [flags]    install moth's hook / plugin (asks first; --yes skips, --dry-run shows)
  moth unhook <harness ...> | --all [flags]  remove exactly what `moth hook` adds, nothing else

Harnesses: claude-code, codex, hermes, pi, opencode, dsh (the adapters in moth/integrations/, README there).
Rules every change follows:
  - a config file it edits is backed up first, next to it: <file>.bak-moth-<timestamp>;
  - JSON / YAML / TOML configs are merged: other hooks, plugins and keys stay as they were;
  - installing twice changes nothing; a copied adapter (Pi, OpenCode, Hermes) that differs from this install's copy
    (say after `moth update`) is replaced, the old copy backed up (Hermes: in ~/.moth/backups/, so Hermes doesn't load
    it as a second plugin);
  - unhook removes only moth's entries, and deletes a copied file only if it is moth's.
Stdlib only (no torch, no PyYAML): safe to run anywhere. Paths are read from the environment on every call
(HOME, XDG_CONFIG_HOME, CLAUDE_CONFIG_DIR, CODEX_HOME, HERMES_HOME, PI_CODING_AGENT_DIR).
"""
from __future__ import annotations

import argparse
import filecmp
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
INTEGRATIONS = os.path.join(HERE, "integrations")
HARNESSES = ("claude-code", "codex", "hermes", "pi", "opencode", "dsh")
ALIASES = {"claude": "claude-code", "claude_code": "claude-code", "claudecode": "claude-code", "codex-cli": "codex",
           "hermes-agent": "hermes", "open-code": "opencode", "deepseek": "dsh"}
BINARIES = {"claude-code": "claude", "codex": "codex", "hermes": "hermes", "pi": "pi", "opencode": "opencode",
            "dsh": "dsh"}
SIGNATURE = {"pi": "// moth extension for Pi", "opencode": "// moth plugin for OpenCode"}
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")


def prog() -> str:
    return "moth" if os.path.basename(sys.argv[0]) in ("moth", "moth.exe") else "python -m moth"


# --- paths (from the environment on every call, so tests and other HOMEs work) ----------------------------------
def home() -> str:
    return os.path.expanduser("~")


def xdg_config() -> str:
    return os.environ.get("XDG_CONFIG_HOME") or os.path.join(home(), ".config")


def claude_dir() -> str:
    return os.environ.get("CLAUDE_CONFIG_DIR") or os.path.join(home(), ".claude")


def claude_settings(project: str | None = None) -> str:
    if project:
        return os.path.join(os.path.abspath(project), ".claude", "settings.json")
    return os.path.join(claude_dir(), "settings.json")


def codex_dir() -> str:
    return os.environ.get("CODEX_HOME") or os.path.join(home(), ".codex")


def hermes_dir() -> str:
    return os.environ.get("HERMES_HOME") or os.path.join(home(), ".hermes")


def pi_dir() -> str:
    return os.environ.get("PI_CODING_AGENT_DIR") or os.path.join(home(), ".pi", "agent")


def opencode_dir() -> str:
    return os.path.join(xdg_config(), "opencode")


def dsh_dir() -> str:
    return os.path.join(home(), ".dsh")


def config_dir(name: str) -> str:
    return {"claude-code": claude_dir(), "codex": codex_dir(), "hermes": hermes_dir(),
            "pi": os.path.dirname(pi_dir()), "opencode": opencode_dir(), "dsh": dsh_dir()}[name]


def copy_target(name: str) -> str:
    return {"hermes": os.path.join(hermes_dir(), "plugins", "moth"),
            "pi": os.path.join(pi_dir(), "extensions", "moth.ts"),
            "opencode": os.path.join(opencode_dir(), "plugins", "moth.js")}[name]


def copy_source(name: str) -> str:
    return {"hermes": os.path.join(INTEGRATIONS, "hermes", "moth"), "pi": os.path.join(INTEGRATIONS, "pi", "moth.ts"),
            "opencode": os.path.join(INTEGRATIONS, "opencode", "moth.js")}[name]


def script(name: str) -> str:
    return os.path.join(INTEGRATIONS, name)


def command(*parts: str) -> str:
    return subprocess.list2cmdline(parts) if os.name == "nt" else shlex.join(parts)


def short(path: str) -> str:
    h = home()
    return "~" + path[len(h):] if path == h or path.startswith(h + os.sep) else path


# --- small file helpers ---------------------------------------------------------------------------------------
def _read(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return None


def _stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def backup_path(path: str, where: str | None = None) -> str:
    """<file>.bak-moth-<timestamp> (or in `where`), never overwriting an earlier backup."""
    base = os.path.join(where, os.path.basename(path)) if where else path
    out, n = f"{base}.bak-moth-{_stamp()}", 1
    while os.path.lexists(out):
        out, n = f"{base}.bak-moth-{_stamp()}.{n}", n + 1
    return out


def write_atomic(path: str, text: str) -> None:
    path = os.path.realpath(path)   # a symlinked config (dotfiles) keeps its link; the target is written
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    mode = os.stat(path).st_mode & 0o7777 if os.path.exists(path) else None
    fd, tmp = tempfile.mkstemp(prefix=".moth-", suffix=".tmp", dir=folder)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def same_tree(a: str, b: str) -> bool:
    """Two folders with the same files (ignoring __pycache__ / .pyc), byte for byte."""
    def files(root):
        out = set()
        for d, dirs, fs in os.walk(root):
            dirs[:] = [x for x in dirs if x != "__pycache__"]
            out |= {os.path.relpath(os.path.join(d, f), root) for f in fs if not f.endswith(".pyc") and f != ".DS_Store"}
        return out
    fa, fb = files(a), files(b)
    return fa == fb and all(filecmp.cmp(os.path.join(a, f), os.path.join(b, f), shallow=False) for f in fa)


def same_file(a: str, b: str) -> bool:
    return os.path.isfile(a) and os.path.isfile(b) and filecmp.cmp(a, b, shallow=False)


# --- a plan: what one hook / unhook will do, printed before it is done -----------------------------------------
class Action:
    """kind: write (path, text) | copy (src -> path, a file or a folder) | remove (path, a file or a folder) |
    run (cmd). backup: back the old path up first (backup_dir: where, for folders a harness would load)."""

    def __init__(self, kind: str, desc: str, path: str | None = None, text: str | None = None,
                 src: str | None = None, cmd: list[str] | None = None, backup: bool = True,
                 backup_dir: str | None = None, backup_also: tuple[str, ...] = (), fallback: list[str] | None = None):
        self.kind, self.desc, self.path, self.text, self.src, self.cmd = kind, desc, path, text, src, cmd
        self.fallback = fallback   # run: tried instead when cmd's flags are unknown to an older CLI
        self.backup, self.backup_dir, self.backup_also = backup, backup_dir, backup_also

    def apply(self) -> list[str]:
        """Do it; returns the backups made."""
        made = []
        for p in ((self.path,) if self.backup and self.path else ()) + self.backup_also:
            if p and os.path.lexists(p):
                if self.backup_dir:
                    os.makedirs(self.backup_dir, exist_ok=True)
                dst = backup_path(p, self.backup_dir)
                (shutil.copytree(p, dst, ignore=IGNORE, symlinks=True) if os.path.isdir(p) and not os.path.islink(p)
                 else shutil.copy2(p, dst))
                made.append(dst)
        if self.kind == "write":
            write_atomic(self.path, self.text)
        elif self.kind == "copy":
            if os.path.isdir(self.src):
                tmp = self.path + ".moth-new"
                shutil.rmtree(tmp, ignore_errors=True)
                os.makedirs(os.path.dirname(self.path), exist_ok=True)
                shutil.copytree(self.src, tmp, ignore=IGNORE)
                if os.path.lexists(self.path):
                    shutil.rmtree(self.path) if os.path.isdir(self.path) and not os.path.islink(self.path) \
                        else os.remove(self.path)
                os.replace(tmp, self.path)
            else:
                with open(self.src, encoding="utf-8") as f:
                    write_atomic(self.path, f.read())
        elif self.kind == "remove":
            if os.path.isdir(self.path) and not os.path.islink(self.path):
                shutil.rmtree(self.path)
            elif os.path.lexists(self.path):
                os.remove(self.path)
        elif self.kind == "run":
            r = subprocess.run(self.cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, timeout=120)
            if r.returncode != 0 and self.fallback and "unrecognized arguments" in r.stdout:
                r = subprocess.run(self.fallback, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, timeout=120)
            if r.returncode != 0:
                raise RuntimeError(f"{command(*self.cmd)} failed (exit {r.returncode}): {r.stdout.strip()[-300:]}")
        return made


class Plan:
    def __init__(self, harness: str):
        self.harness = harness
        self.actions: list[Action] = []
        self.notes: list[str] = []     # printed after (approval reminders, manual steps)
        self.nothing: str | None = None  # "already installed ..." when there is nothing to do
        self.error: str | None = None


# --- hook commands in JSON settings (Claude Code, Codex hooks.json) -------------------------------------------
# The hook runs in the foreground (no "async"): one-shot runs (`claude -p`, `codex exec`) kill a background hook when
# they exit, before it has posted (seen live, 2026-10-18). It takes a moment: the adapter gives up on moth after 1 s
# and moth only queues the text. HOOK_TIMEOUT is the harness's own hard stop, with room for a slow interpreter start.
HOOK_TIMEOUT = 3


def hook_entry(cmd: str) -> dict:
    """moth's command hook, as `moth hook` writes it and `python -m moth` prints it."""
    return {"type": "command", "command": cmd, "timeout": HOOK_TIMEOUT}


def moth_script_in(cmd, name: str) -> str | None:
    """The path of moth's integrations/<name> that a hook command runs (any interpreter, MOTH_URL=... prefixes and
    quoting allowed), else None. A copy of moth elsewhere counts too: it's still moth's hook."""
    if not isinstance(cmd, str):
        return None
    try:
        parts = shlex.split(cmd, posix=os.name != "nt")
    except ValueError:
        parts = cmd.split()
    for p in parts:
        p = p.strip("\"'")
        norm = p.replace("\\", "/")
        if norm.endswith("/integrations/" + name) or norm == "integrations/" + name:
            return p
    return None


def _load_json(path: str) -> tuple[dict | None, str | None]:
    text = _read(path)
    if text is None or not text.strip():
        return {}, None
    try:
        data = json.loads(text)
    except ValueError as e:
        return None, f"{path} isn't valid JSON ({e}); fix it by hand first, moth won't touch it"
    if not isinstance(data, dict):
        return None, f"{path} isn't a JSON object; moth won't touch it"
    return data, None


def _dump_json(data: dict, old: str | None) -> str:
    indent = 2
    m = re.search(r"\n( +)\S", old or "")
    if m:
        indent = len(m.group(1))
    return json.dumps(data, indent=indent, ensure_ascii=False) + "\n"


def _stop_entries(data: dict, name: str) -> list[tuple[dict, dict, str]]:
    """(group, handler, script path) for each of moth's handlers under hooks.Stop."""
    out = []
    hooks = data.get("hooks")
    stop = hooks.get("Stop") if isinstance(hooks, dict) else None
    for group in stop if isinstance(stop, list) else ():
        for h in (group.get("hooks") or ()) if isinstance(group, dict) else ():
            if isinstance(h, dict):
                p = moth_script_in(h.get("command"), name)
                if p:
                    out.append((group, h, p))
    return out


def _where(p: str, name: str) -> str:
    """'this' (this install's script), 'other' (another moth that exists) or 'missing'."""
    if os.path.realpath(os.path.expanduser(p)) == os.path.realpath(script(name)):
        return "this"
    return "other" if os.path.isfile(os.path.expanduser(p)) else "missing"


def json_hook_status(path: str, name: str) -> tuple[str, str]:
    data, err = _load_json(path)
    if err:
        return "error", err
    entries = _stop_entries(data, name)
    if not entries:
        return "absent", f"no moth hook in {short(path)}"
    kinds = [_where(p, name) for _, _, p in entries]
    if "this" in kinds:
        if any(h.get("async") for _, h, _ in entries):
            return "outdated", (f"Stop hook in {short(path)} runs in the background (\"async\": true), which one-shot "
                                f"runs (claude -p, codex exec) kill before it posts")
        extra = f" ({len(entries)} moth hooks: duplicates)" if len(entries) > 1 else ""
        return "installed", f"Stop hook in {short(path)}{extra}"
    if "other" in kinds:
        return "other", f"Stop hook in {short(path)} runs another moth copy: {entries[0][2]}"
    return "broken", f"Stop hook in {short(path)} points to a missing file: {entries[0][2]}"


def plan_json_hook(plan: Plan, path: str, name: str, what: str) -> None:
    data, err = _load_json(path)
    if err:
        plan.error = err
        return
    old = _read(path)
    want = command(sys.executable, script(name))
    entries = _stop_entries(data, name)
    here = any(_where(p, name) == "this" for _, _, p in entries)
    background = [h for _, h, _ in entries if h.get("async")]
    if here and not background:
        plan.nothing = f"already installed ({what} in {short(path)})"
        return
    if entries:   # moth's hook from a moved / older install, or an older background one: fix it in place
        changes = []
        if not here:
            for _, h, p in entries:
                h["command"] = want
            changes.append(f"{entries[0][2]} -> {script(name)}")
        for h in background:
            del h["async"]
            h["timeout"] = HOOK_TIMEOUT
        if background:
            changes.append("run it in the foreground (a background hook is killed by claude -p / codex exec before "
                           "it posts)")
        plan.actions.append(Action("write", f"update moth's {what} in {short(path)}: " + "; ".join(changes), path,
                                   _dump_json(data, old)))
        return
    hooks = data.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        plan.error = f"{path}: \"hooks\" isn't an object; moth won't touch it"
        return
    stop = hooks.setdefault("Stop", [])
    if not isinstance(stop, list):
        plan.error = f"{path}: \"hooks.Stop\" isn't a list; moth won't touch it"
        return
    stop.append({"hooks": [hook_entry(want)]})   # appended: other
    verb = "add" if old is not None else "create"   # entries keep their index (Codex trust is keyed by it)
    plan.actions.append(Action("write", f"{verb} {short(path)} with moth's {what}: {want}" if verb == "create" else
                               f"add moth's {what} to {short(path)}: {want}", path, _dump_json(data, old)))


def plan_json_unhook(plan: Plan, path: str, name: str, what: str) -> None:
    data, err = _load_json(path)
    if err:
        plan.error = err
        return
    old = _read(path)
    entries = _stop_entries(data, name)
    if not entries:
        return
    stop = data["hooks"]["Stop"]
    ids = {id(h) for _, h, _ in entries}
    for group in stop:
        if isinstance(group, dict) and isinstance(group.get("hooks"), list):
            group["hooks"] = [h for h in group["hooks"] if id(h) not in ids]
    data["hooks"]["Stop"] = [g for g in stop if not (isinstance(g, dict) and g.get("hooks") == []
                                                      and any(id(g) == id(eg) for eg, _, _ in entries))]
    if not data["hooks"]["Stop"]:
        del data["hooks"]["Stop"]
    if not data["hooks"]:
        del data["hooks"]
    plan.actions.append(Action("write", f"remove moth's {what} from {short(path)} "
                                        f"({len(entries)} entr{'y' if len(entries) == 1 else 'ies'})",
                               path, _dump_json(data, old)))


# --- Codex legacy notify (config.toml), read and written as text: no TOML library needed -----------------------
NOTIFY = re.compile(r"^notify\s*=\s*\[(?P<body>[^\]]*)\][^\n]*\n?", re.M)


def _toml_top(text: str) -> str:
    """The part of a TOML file before its first [table] header (where a top-level key like notify lives)."""
    m = re.search(r"^\s*\[", text, re.M)
    return text[:m.start()] if m else text


def codex_notify(text: str | None) -> tuple[re.Match | None, list[str]]:
    if not text:
        return None, []
    m = NOTIFY.search(_toml_top(text))
    if not m:
        return None, []
    items = [json.loads(s) if s.startswith('"') else s[1:-1]
             for s in re.findall(r'"(?:[^"\\]|\\.)*"|\'[^\']*\'', m.group("body"))]
    return m, items


def codex_config() -> str:
    return os.path.join(codex_dir(), "config.toml")


def codex_hooks() -> str:
    return os.path.join(codex_dir(), "hooks.json")


def notify_is_moth(items: list[str]) -> str | None:
    for it in items:
        if moth_script_in(command(it), "codex.py") or it.replace("\\", "/").endswith("/integrations/codex.py"):
            return it
    return None


# --- Hermes plugins.enabled, read without PyYAML ---------------------------------------------------------------
def yaml_enabled_plugins(text: str | None) -> list[str]:
    """plugins.enabled of a Hermes config.yaml (block or flow list). PyYAML if it's there, else a small reader."""
    if not text:
        return []
    try:
        import yaml   # optional
        d = yaml.safe_load(text)
        en = (d or {}).get("plugins", {}).get("enabled") if isinstance(d, dict) else None
        return [str(x) for x in en] if isinstance(en, list) else []
    except ImportError:
        pass
    except Exception:
        return []
    lines = text.splitlines()
    out, in_plugins, en_indent = [], False, None
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if indent == 0:
            in_plugins = line.split("#")[0].strip() == "plugins:"
            en_indent = None
            continue
        if not in_plugins:
            continue
        if en_indent is None:
            m = re.match(r"\s*enabled:\s*(.*)$", line)
            if m:
                rest = m.group(1).split("#")[0].strip()
                if rest.startswith("["):
                    return [x.strip().strip("'\"") for x in rest.strip("[]").split(",") if x.strip()]
                en_indent = indent
            continue
        m = re.match(r"\s*-\s*(.+)$", line)
        if indent >= en_indent and m:
            out.append(m.group(1).split(" #")[0].strip().strip("'\""))
        elif indent <= en_indent:
            break
    return out


def hermes_config() -> str:
    return os.path.join(hermes_dir(), "config.yaml")


def hermes_is_moth(folder: str) -> bool:
    y = _read(os.path.join(folder, "plugin.yaml")) or ""
    return bool(re.search(r"^name:\s*['\"]?moth['\"]?\s*$", y, re.M)) and \
        os.path.isfile(os.path.join(folder, "__init__.py"))


# --- copied single files (Pi, OpenCode) -------------------------------------------------------------------------
def file_is_moth(path: str, name: str) -> bool:
    if same_file(path, copy_source(name)):
        return True
    try:
        with open(path, encoding="utf-8") as f:
            return f.readline().startswith(SIGNATURE[name])
    except (OSError, UnicodeDecodeError):
        return False


# --- dsh: one line in a YAML list (cordis.patch.yml), as text --------------------------------------------------
DSH_LINE = re.compile(r"^[ \t]*-[ \t]*insert:[ \t]*\[[ \t]*\{[ \t]*id:[ \t]*['\"]?moth['\"]?[ \t]*,[ \t]*name:[ \t]*"
                      r"(?P<name>'(?:[^']|'')*'|\"(?:[^\"\\]|\\.)*\"|[^}\s]+)[ \t]*\}[ \t]*\][ \t]*(?:#[^\n]*)?\n?",
                      re.M)


def dsh_profiles() -> list[str]:
    base = os.path.join(dsh_dir(), "profiles")
    return sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))) if os.path.isdir(base) else []


def dsh_patch(profile: str) -> str:
    return os.path.join(dsh_dir(), "profiles", profile, "cordis.patch.yml")


def dsh_line() -> str:
    return "- insert: [{ id: moth, name: '" + script(os.path.join("dsh", "moth.mjs")).replace("'", "''") + "' }]\n"


def _dsh_name(m: re.Match) -> str:
    n = m.group("name")
    if n.startswith("'"):
        return n[1:-1].replace("''", "'")
    if n.startswith('"'):
        return json.loads(n)
    return n


# --- status ----------------------------------------------------------------------------------------------------
def detected(name: str) -> tuple[bool, str]:
    why = []
    b = shutil.which(BINARIES[name])
    if b:
        why.append(f"`{BINARIES[name]}` on PATH")
    if os.path.isdir(config_dir(name)):
        why.append(short(config_dir(name)))
    return bool(why), ", ".join(why)


def status(name: str, project: str | None = None) -> dict:
    """{"name", "detected", "why", "state", "detail"}; state: installed | outdated | other | broken | partial |
    absent | error. outdated / other / broken / partial are fixed by `moth hook <name>`."""
    found, why = detected(name)
    st = {"name": name, "detected": found, "why": why}
    if name == "claude-code":
        st["state"], st["detail"] = json_hook_status(claude_settings(project), "claude_code.py")
    elif name == "codex":
        s, d = json_hook_status(codex_hooks(), "codex.py")
        _, items = codex_notify(_read(codex_config()))
        legacy = notify_is_moth(items)
        if legacy and s != "absent" and s != "error":
            s, d = "error", f"BOTH variants installed (hooks.json and legacy notify in {short(codex_config())}): " \
                            f"every reply posts twice; `{prog()} unhook codex` then `{prog()} hook codex`"
        elif legacy:
            s = "installed" if _where(legacy, "codex.py") == "this" else _where(legacy, "codex.py").replace(
                "missing", "broken")
            d = f"legacy notify in {short(codex_config())}"
        elif s == "installed":
            d += " (Codex runs it only once approved: /hooks in codex)"
        st["state"], st["detail"] = s, d
    elif name in ("pi", "opencode"):
        dst = copy_target(name)
        if not os.path.exists(dst):
            st["state"], st["detail"] = "absent", f"no {short(dst)}"
        elif same_file(dst, copy_source(name)):
            st["state"], st["detail"] = "installed", short(dst)
        elif file_is_moth(dst, name):
            st["state"], st["detail"] = "outdated", f"{short(dst)} differs from this install's copy"
        else:
            st["state"], st["detail"] = "error", f"{short(dst)} exists but isn't moth's adapter"
    elif name == "hermes":
        dst = copy_target(name)
        on = "moth" in yaml_enabled_plugins(_read(hermes_config()))
        if not os.path.exists(dst):
            st["state"], st["detail"] = ("broken", f"enabled in config.yaml but {short(dst)} is missing") if on else \
                ("absent", f"no {short(dst)}")
        elif not hermes_is_moth(dst):
            st["state"], st["detail"] = "error", f"{short(dst)} exists but isn't moth's plugin"
        elif not same_tree(dst, copy_source(name)):
            st["state"], st["detail"] = "outdated", f"{short(dst)} differs from this install's copy" + \
                ("" if on else ", and not enabled")
        elif not on:
            st["state"], st["detail"] = "partial", f"{short(dst)} copied but not enabled (hermes plugins enable moth)"
        else:
            st["state"], st["detail"] = "installed", f"{short(dst)}, enabled"
    elif name == "dsh":
        found_in = []
        for p in dsh_profiles():
            for m in DSH_LINE.finditer(_read(dsh_patch(p)) or ""):
                found_in.append((p, _dsh_name(m)))
        if not found_in:
            st["state"], st["detail"] = "absent", "no moth entry in any ~/.dsh profile"
        else:
            kinds = [_where(n, os.path.join("dsh", "moth.mjs")) for _, n in found_in]
            st["state"] = "installed" if all(k == "this" for k in kinds) else \
                "other" if "missing" not in kinds else "broken"
            st["detail"] = "profile " + ", ".join(sorted({p for p, _ in found_in}))
    return st


MARK = {"installed": "✓", "outdated": "!", "other": "!", "broken": "✗", "partial": "!", "error": "✗", "absent": "·"}
WORD = {"installed": "installed", "outdated": "outdated", "other": "another moth", "broken": "broken",
        "partial": "half-installed", "error": "problem", "absent": "not installed"}


def status_lines(project: str | None = None, only_detected: bool = False) -> list[tuple[str, str]]:
    """(mark, text) per harness, for `moth hooks` and `moth doctor`."""
    out = []
    for name in HARNESSES:
        st = status(name, project)
        if not st["detected"] and st["state"] == "absent":
            if not only_detected:
                out.append(("-", f"{name:<12} not found on this machine"))
            continue
        hint = ""
        if st["state"] in ("outdated", "other", "broken", "partial", "absent"):
            hint = f"  -> {prog()} hook {name}"
        where = f" [{st['why']}]" if st["why"] else " [not detected, but configured]"
        out.append((MARK[st["state"]], f"{name:<12} {WORD[st['state']]}: {st['detail']}{hint}{where}"))
    return out


def ascii_safe(mark: str) -> str:
    enc = (getattr(sys.stdout, "encoding", None) or "ascii")
    try:
        mark.encode(enc)
        return mark
    except (UnicodeEncodeError, LookupError):
        return {"✓": "ok", "✗": "XX", "!": "!!", "·": "..", "-": "--"}.get(mark, mark)


def show(project: str | None = None) -> int:
    print("coding agents moth can watch (✓ hooked up):".replace("✓", ascii_safe("✓")))
    for mark, text in status_lines(project):
        print(f"  {ascii_safe(mark)} {text}")
    print(f"\n{prog()} hook <name> installs one ({prog()} hook --all: every harness found), {prog()} unhook <name> "
          "removes it.\nThe Claude Code band above the prompt is a separate mod: claude --plugin-dir "
          f"{command(os.path.join(INTEGRATIONS, 'claude_code_mod'))}")
    return 0


# --- plans per harness -------------------------------------------------------------------------------------------
def plan_hook(name: str, project: str | None = None, profile: str | None = None, legacy_notify: bool = False) -> Plan:
    plan = Plan(name)
    if name == "claude-code":
        plan_json_hook(plan, claude_settings(project), "claude_code.py", "Stop hook")
        if plan.actions:
            plan.notes.append("Claude Code picks the hook up in new sessions (or after /hooks). The band above the "
                              f"prompt is separate: claude --plugin-dir "
                              f"{command(os.path.join(INTEGRATIONS, 'claude_code_mod'))}")
    elif name == "codex":
        text = _read(codex_config())
        m, items = codex_notify(text)
        legacy = notify_is_moth(items)
        hs, _ = json_hook_status(codex_hooks(), "codex.py")
        if legacy and hs not in ("absent", "error"):
            plan.error = (f"both Codex variants are installed (hooks.json and notify in {short(codex_config())}), so "
                          f"every reply posts twice: run `{prog()} unhook codex`, then `{prog()} hook codex`")
        elif legacy_notify:
            if hs not in ("absent", "error"):
                plan.error = (f"moth's hooks.json hook is installed; moth never installs both variants. To switch: "
                              f"`{prog()} unhook codex`, then `{prog()} hook codex --legacy-notify`")
            elif legacy and _where(legacy, "codex.py") == "this":
                plan.nothing = f"already installed (legacy notify in {short(codex_config())})"
            elif m and not legacy:
                plan.error = (f"{short(codex_config())} already has a notify program ({' '.join(items)}); Codex has "
                              f"one notify slot, so moth won't replace it. Use the hooks.json variant (without "
                              f"--legacy-notify) instead")
            else:
                want = f"notify = {json.dumps([sys.executable, script('codex.py')], ensure_ascii=False)}\n"
                if m:   # moth's notify from another install: point it here
                    new = text[:m.start()] + want + text[m.end():]
                    desc = f"update moth's legacy notify in {short(codex_config())}"
                else:   # top-level key: before any [table]; after the file's leading comments
                    lead = re.match(r"(?:[ \t]*(?:#[^\n]*)?\n)*", text or "").end()
                    new = (text or "")[:lead] + want + (text or "")[lead:]
                    desc = f"add legacy notify to {short(codex_config())}: {want.strip()}"
                plan.actions.append(Action("write", desc, codex_config(), new))
        elif legacy:
            plan.nothing = (f"already installed as legacy notify in {short(codex_config())}; moth never installs both "
                            f"variants (to switch to hooks.json: `{prog()} unhook codex`, then `{prog()} hook codex`)")
            if _where(legacy, "codex.py") != "this":
                plan.nothing = None
                want = f"notify = {json.dumps([sys.executable, script('codex.py')], ensure_ascii=False)}\n"
                plan.actions.append(Action("write", f"update moth's legacy notify in {short(codex_config())}",
                                           codex_config(), text[:m.start()] + want + text[m.end():]))
        else:
            plan_json_hook(plan, codex_hooks(), "codex.py", "Stop hook")
        if plan.actions and not legacy_notify and not legacy:
            plan.notes.append("Codex runs a new or changed hook only after you approve it once: start `codex` and "
                              "accept it in the hook review (or later with /hooks). Until then it is skipped silently.")
    elif name in ("pi", "opencode"):
        src, dst = copy_source(name), copy_target(name)
        if not os.path.exists(dst):
            plan.actions.append(Action("copy", f"copy {short(src)} -> {short(dst)}", dst, src=src, backup=False))
        elif same_file(dst, src):
            plan.nothing = f"already installed ({short(dst)})"
        elif not file_is_moth(dst, name):
            plan.error = f"{short(dst)} exists and isn't moth's adapter; moth won't overwrite it"
        else:
            plan.actions.append(Action("copy", f"update {short(dst)} (differed from this install's copy)", dst,
                                       src=src))
        if plan.actions:
            plan.notes.append(f"{'Pi' if name == 'pi' else 'OpenCode'} loads it at its next start.")
    elif name == "hermes":
        src, dst = copy_source(name), copy_target(name)
        backups = os.path.join(home(), ".moth", "backups")
        if not os.path.exists(dst):
            plan.actions.append(Action("copy", f"copy {short(src)}/ -> {short(dst)}/", dst, src=src, backup=False))
        elif not hermes_is_moth(dst):
            plan.error = f"{short(dst)} exists and isn't moth's plugin; moth won't overwrite it"
            return plan
        elif not same_tree(dst, src):
            plan.actions.append(Action("copy", f"update {short(dst)}/ (differed from this install's copy; old one "
                                               f"kept in {short(backups)})", dst, src=src, backup_dir=backups))
        if "moth" not in yaml_enabled_plugins(_read(hermes_config())):
            hermes = shutil.which("hermes")
            if hermes:
                plan.actions.append(Action("run", f"run `hermes plugins enable moth` (adds moth to plugins.enabled in "
                                                  f"{short(hermes_config())})",
                                           # moth replaces no built-in tools: say so, which also skips Hermes's prompt
                                           cmd=[hermes, "plugins", "enable", "--no-allow-tool-override", "moth"],
                                           fallback=[hermes, "plugins", "enable", "moth"],
                                           backup=False, backup_also=(hermes_config(),)))
            else:
                plan.notes.append(f"`hermes` isn't on PATH: enable the plugin with `hermes plugins enable moth`, or add "
                                  f"moth under plugins.enabled in {short(hermes_config())}")
        if not plan.actions and not plan.notes:
            plan.nothing = f"already installed ({short(dst)}, enabled)"
    elif name == "dsh":
        profiles = dsh_profiles()
        if not profile:
            if len(profiles) == 1:
                profile = profiles[0]
            else:
                plan.error = ("pick a dsh profile with --profile NAME (" + (
                    f"found: {', '.join(profiles)}" if profiles else f"none in {short(os.path.join(dsh_dir(), 'profiles'))}")
                    + ")")
                return plan
        path = dsh_patch(profile)
        text = _read(path)
        want = dsh_line()
        ms = list(DSH_LINE.finditer(text or ""))
        mjs = os.path.join("dsh", "moth.mjs")
        if any(_where(_dsh_name(m), mjs) == "this" for m in ms):
            plan.nothing = f"already installed ({short(path)})"
        elif ms:
            new = text
            for m in reversed(ms):
                new = new[:m.start()] + want + new[m.end():]
            plan.actions.append(Action("write", f"update moth's entry in {short(path)} to {script(mjs)}", path, new))
        else:
            body = text or ""
            if body.strip() in ("", "[]"):
                body = ""
            elif not body.endswith("\n"):
                body += "\n"
            plan.actions.append(Action("write", f"{'add moth to' if text is not None else 'create'} {short(path)}: "
                                                f"{want.strip()}", path, body + want))
    return plan


def plan_unhook(name: str, project: str | None = None, profile: str | None = None) -> Plan:
    plan = Plan(name)
    if name == "claude-code":
        plan_json_unhook(plan, claude_settings(project), "claude_code.py", "Stop hook")
    elif name == "codex":
        plan_json_unhook(plan, codex_hooks(), "codex.py", "Stop hook")
        text = _read(codex_config())
        m, items = codex_notify(text)
        if m and notify_is_moth(items):
            plan.actions.append(Action("write", f"remove moth's legacy notify from {short(codex_config())}",
                                       codex_config(), text[:m.start()] + text[m.end():]))
    elif name in ("pi", "opencode"):
        dst = copy_target(name)
        if os.path.exists(dst):
            if not file_is_moth(dst, name):
                plan.error = f"{short(dst)} isn't moth's adapter; moth won't delete it"
            else:   # a byte-identical copy can be re-made any time; anything else is backed up first
                plan.actions.append(Action("remove", f"delete {short(dst)}", dst,
                                           backup=not same_file(dst, copy_source(name))))
    elif name == "hermes":
        dst = copy_target(name)
        if "moth" in yaml_enabled_plugins(_read(hermes_config())):
            hermes = shutil.which("hermes")
            if hermes:
                plan.actions.append(Action("run", f"run `hermes plugins disable moth` ({short(hermes_config())})",
                                           cmd=[hermes, "plugins", "disable", "moth"], backup=False,
                                           backup_also=(hermes_config(),)))
            else:
                plan.notes.append(f"`hermes` isn't on PATH: remove moth from plugins.enabled in "
                                  f"{short(hermes_config())} by hand")
        if os.path.exists(dst):
            if not hermes_is_moth(dst):
                plan.error = f"{short(dst)} isn't moth's plugin; moth won't delete it"
            else:
                plan.actions.append(Action("remove", f"delete {short(dst)}/", dst,
                                           backup=not same_tree(dst, copy_source(name)),
                                           backup_dir=os.path.join(home(), ".moth", "backups")))
    elif name == "dsh":
        for p in ([profile] if profile else dsh_profiles()):
            path = dsh_patch(p)
            text = _read(path)
            ms = list(DSH_LINE.finditer(text or ""))
            if ms:
                new = text
                for m in reversed(ms):
                    new = new[:m.start()] + new[m.end():]
                plan.actions.append(Action("write", f"remove moth's entry from {short(path)}", path, new))
    if not plan.actions and not plan.error and not plan.notes:
        plan.nothing = "not installed, nothing to remove"
    return plan


# --- running plans ------------------------------------------------------------------------------------------------
def _confirm(question: str) -> bool:
    try:
        return input(question + " [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def run_plans(plans: list[Plan], yes: bool, dry_run: bool, verb: str) -> int:
    rc = 0
    todo = [p for p in plans if p.actions and not p.error]
    for p in plans:
        if p.error:
            print(f"{p.harness}: {p.error}")
            rc = 1
        elif not p.actions:
            if p.nothing:
                print(f"{p.harness}: {p.nothing}")
            for n in p.notes:
                print(f"{p.harness}: note: {n}")
    if not todo:
        return rc
    print(("would " if dry_run else "moth will ") + verb + ":")
    for p in todo:
        for a in p.actions:
            print(f"  {p.harness}: {a.desc}")
    if dry_run:
        print("(dry run: nothing changed)")
        return rc
    if not yes and not _confirm("go ahead?"):
        print("cancelled, nothing changed")
        return 1
    for p in todo:
        for a in p.actions:
            try:
                made = a.apply()
            except Exception as e:   # one harness failing doesn't stop the others
                print(f"  {p.harness}: FAILED: {a.desc}: {e}")
                rc = 1
                break
            print(f"  {p.harness}: done: {a.desc}")
            for b in made:
                print(f"    backup: {b}")
        for n in p.notes:
            print(f"  {p.harness}: note: {n}")
    return rc


def _names(raw: list[str], everything: bool, pick) -> list[str] | str:
    if everything:
        return [n for n in HARNESSES if pick(n)]
    out = []
    for r in raw:
        n = ALIASES.get(r.lower(), r.lower())
        if n not in HARNESSES:
            return f"unknown harness {r!r}; one of: {', '.join(HARNESSES)}"
        if n not in out:
            out.append(n)
    return out


def _parser(cmd: str) -> argparse.ArgumentParser:
    what = {"hooks": "list the coding agents found here and whether moth is hooked into each",
            "hook": "install moth's hook / plugin into coding agents (backs up and merges configs)",
            "unhook": "remove moth's hook / plugin from coding agents (only moth's entries)"}[cmd]
    ap = argparse.ArgumentParser(prog=f"{prog()} {cmd}", description=what,
                                 epilog=f"harnesses: {', '.join(HARNESSES)}")
    if cmd != "hooks":
        ap.add_argument("harness", nargs="*", help="one or more harness names")
        ap.add_argument("--all", action="store_true",
                        help="every harness found on this machine" if cmd == "hook" else "every harness moth is in")
        ap.add_argument("--yes", "-y", action="store_true", help="don't ask before changing files")
        ap.add_argument("--dry-run", action="store_true", help="only show what would change")
        ap.add_argument("--profile", help="dsh profile (default: the only one in ~/.dsh/profiles)")
    if cmd == "hook":
        ap.add_argument("--legacy-notify", action="store_true",
                        help="Codex: use the legacy notify line in config.toml instead of hooks.json")
    ap.add_argument("--project", metavar="DIR", help="Claude Code: DIR/.claude/settings.json instead of ~/.claude")
    return ap


def main(cmd: str, argv=None) -> int:
    a = _parser(cmd).parse_args(argv)
    if cmd == "hooks":
        return show(a.project)
    if a.all and a.harness:
        print("give harness names or --all, not both", file=sys.stderr)
        return 2
    if cmd == "hook":
        names = _names(a.harness, a.all, lambda n: detected(n)[0])
    else:
        names = _names(a.harness, a.all, lambda n: status(n, a.project)["state"] != "absent")
    if isinstance(names, str):
        print(names, file=sys.stderr)
        return 2
    if not names:
        if a.all:
            print("no coding agents found on this machine" if cmd == "hook" else "moth isn't hooked into anything")
            return 0
        _parser(cmd).print_usage(sys.stderr)
        return 2
    if cmd == "hook":
        plans = [plan_hook(n, a.project, a.profile, a.legacy_notify) for n in names]
        return run_plans(plans, a.yes, a.dry_run, "install")
    plans = [plan_unhook(n, a.project, a.profile) for n in names]
    return run_plans(plans, a.yes, a.dry_run, "remove")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "hooks", sys.argv[2:]))
