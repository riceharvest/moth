"""moth forget : wipe what moth remembers about you.

  moth forget [--yes]                 its log files (and stale pidfiles). What moth remembers while it runs (the
                                      chat, and the token ids of its own last 16 reactions, which its phrase memory
                                      keeps so it doesn't repeat a joke) lives only in RAM: stopping moth wipes it.
                                      The chat page's "new chat" button wipes the chat but keeps those reactions.
  moth forget --all [--yes]           also the home folder an older moth (0.3, which had files and a diary) left
                                      behind; only a folder moth made itself (.moth-home marker). moth 0.4 is pure
                                      chat and keeps no files.
  moth forget --all --token [--yes]   also the install token and moth's settings (~/.moth/token, config.json).

The token and settings are never removed otherwise; hook backups in ~/.moth/backups are kept. Deletes no files while a
moth is running (pidfiles in ~/.moth and /api/health): it wipes that moth's memory in RAM instead, the chat and its
last reactions (POST /api/reset {"everything": true}, with the token), and asks you to stop moth and run it again for
the files. Asks before deleting or wiping anything (--yes skips).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

MARKER = ".moth-home"   # what moth <= 0.3 wrote into the home folder it made (its home was dropped in 0.4)


def prog() -> str:
    return "moth" if os.path.basename(sys.argv[0]) in ("moth", "moth.exe") else "python -m moth"


def moth_dir() -> str:
    return os.path.join(os.path.expanduser("~"), ".moth")


def default_home() -> str:
    """Where moth <= 0.3 kept its home: ~/.moth/home, or live/home in this project's dev checkout."""
    from moth import weights
    if weights.mode() == "dev":
        return os.path.join(weights.ROOT, "live", "home")
    return os.path.join(moth_dir(), "home")


def token_path() -> str:   # serve.token_path(), without importing serve (torch)
    return os.environ.get("MOTH_TOKEN_FILE") or os.path.join(moth_dir(), "token")


def running(port: int = 4747) -> tuple[int, int | None] | None:
    """(port, pid) of any live moth: a pidfile whose pid is alive and whose port answers /api/health with that pid,
    or a moth answering /api/health on `port`. From any install (they all share ~/.moth)."""
    from moth import update
    for path in sorted(glob.glob(os.path.join(moth_dir(), "serve-*.pid"))):
        try:
            with open(path) as f:
                d = json.load(f)
            pid, p = int(d["pid"]), int(d["port"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if not update._alive(pid):
            continue
        h = update._health(p)
        if isinstance(h, dict) and h.get("ok") and h.get("pid") == pid:
            return p, pid
    h = update._health(port)
    if isinstance(h, dict) and h.get("ok"):
        return port, h.get("pid")
    return None


def wipe_running(port: int, timeout: float = 30.0) -> str | None:
    """Make the moth on `port` forget everything it holds in RAM: the chat and its phrase memory (serve.Moth.forget
    with everything=True). None when done, else why not. Never via a proxy (moth is local)."""
    import urllib.request
    try:
        with open(token_path(), encoding="utf-8") as f:
            token = f.read().strip()
    except OSError as e:
        return f"can't read the token ({e.strerror or e})"
    req = urllib.request.Request(f"http://127.0.0.1:{port}/api/reset", data=json.dumps({"everything": True}).encode(),
                                 headers={"Content-Type": "application/json", "X-Moth-Token": token})
    try:   # timeout: a reply moth is writing finishes first (forget waits for the model's lock)
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=timeout) as r:
            ok = json.load(r).get("ok") is True
    except Exception as e:  # noqa: BLE001  refused, wrong token, an older moth, not moth: say so
        return f"{type(e).__name__}: {e}"
    return None if ok else "it didn't say ok"


def stale_pidfiles() -> list[str]:
    from moth import update
    out = []
    for path in sorted(glob.glob(os.path.join(moth_dir(), "serve-*.pid"))):
        try:
            with open(path) as f:
                pid = int(json.load(f)["pid"])
        except (OSError, ValueError, KeyError, TypeError):
            out.append(path)
            continue
        if not update._alive(pid):
            out.append(path)
    return out


def log_files() -> list[str]:
    out = sorted(glob.glob(os.path.join(moth_dir(), "*.log")))
    if os.path.isdir(os.path.join(moth_dir(), "logs")):
        out.append(os.path.join(moth_dir(), "logs"))
    try:
        from moth import weights
        if weights.mode() == "dev":   # try_moth.sh's live/serve.log, live/pet.log
            out += sorted(glob.glob(os.path.join(weights.ROOT, "live", "*.log")))
    except Exception:
        pass
    return out


def home_owned(home: str) -> bool:
    """serve.Home.owned: moth made this folder itself (the marker is a real file, not a symlink)."""
    m = os.path.join(home, MARKER)
    return os.path.isfile(m) and not os.path.islink(m)


def _confirm(question: str) -> bool:
    try:
        return input(question + " [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog=f"{prog()} forget", description="wipe moth's memory of you: its logs; with "
                                 "--all also an old home folder (moth <= 0.3); the token and settings only with "
                                 "--all --token")
    ap.add_argument("--all", action="store_true", help="also delete the home folder an older moth (<= 0.3) made")
    ap.add_argument("--token", action="store_true", help="with --all: also delete the token and config.json")
    ap.add_argument("--home", default=None, help="moth's home folder, if you start moth with --home (default: "
                                                 "serve's default)")
    ap.add_argument("--port", type=int, default=4747, help="the port moth serves on (default 4747)")
    ap.add_argument("--yes", "-y", action="store_true", help="don't ask before deleting")
    a = ap.parse_args(argv)
    if a.token and not a.all:
        print("--token only goes with --all (moth forget --all --token)", file=sys.stderr)
        return 2
    live = running(a.port)
    if live:
        print(f"moth is running (port {live[0]}{f', pid {live[1]}' if live[1] else ''}), so its files can't go yet.")
        if a.yes or _confirm("wipe what it remembers right now (this chat and its last reactions, all in RAM)?"):
            why = wipe_running(live[0])
            print("wiped: the running moth forgot this chat and its last reactions." if why is None else
                  f"couldn't wipe its memory ({why}); stopping moth wipes it.")
        print(f"To delete its logs (and with --all its home), stop it first, then run `{prog()} forget` again; "
              "stopping it also wipes everything it holds in RAM.\nIf the desktop pet is running, quit it as well, or "
              f"it may wake moth up again (and `{prog()} autostart off` if you turned autostart on).")
        return 1
    targets: list[tuple[str, str]] = [(p, "log") for p in log_files()] + [(p, "stale pidfile") for p in stale_pidfiles()]
    if a.all:
        home = os.path.abspath(os.path.expanduser(a.home or default_home()))
        if os.path.isdir(home):
            danger = {os.path.realpath(os.path.expanduser("~")), os.path.realpath(os.sep), os.path.realpath(moth_dir())}
            if os.path.islink(home) or os.path.realpath(home) in danger or not home_owned(home):
                print(f"moth didn't make {home} itself (no {MARKER} marker), so it won't delete it. Move it away by "
                      f"hand (say, mv {home} {home}.old); moth 0.4 doesn't use it. Nothing was "
                      "deleted.")
                return 1
            targets.append((home, "old home folder (moth <= 0.3's files and diary)"))
        if a.token:
            from moth import config
            targets += [(p, w) for p, w in ((token_path(), "token (hooks pick up the new one on moth's next start)"),
                                            (config.path(), "settings")) if os.path.exists(p)]
    print("conversation and reaction memory: nothing on disk (they live only in RAM while moth runs; moth isn't "
          "running now)")
    if not targets:
        print("nothing else to forget" + ("" if a.all else f" (`{prog()} forget --all` also deletes an old home folder)"))
        return 0
    print("moth forget will delete:")
    for p, what in targets:
        print(f"  {p}  ({what})")
    if not a.yes and not _confirm("delete these?"):
        print("cancelled, nothing deleted")
        return 1
    rc = 0
    for p, _ in targets:
        try:
            if os.path.isdir(p) and not os.path.islink(p):
                shutil.rmtree(p)
            else:
                os.remove(p)
            print(f"  deleted {p}")
        except OSError as e:
            print(f"  couldn't delete {p}: {e}")
            rc = 1
    if a.all:
        print("the old home folder is gone (moth 0.4 keeps no files).")
    return rc


if __name__ == "__main__":
    sys.exit(main())
