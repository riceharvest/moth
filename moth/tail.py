"""Watch moth and talk to it from a side terminal:  python -m moth.tail      (python -m moth.tail --watch: no prompt)

moth (python -m moth.serve) reacts to every coding-agent turn, but the harness adapters are one-way and never see
the reply. This is the display that works next to any harness (Claude Code, Codex, Hermes, Pi, OpenCode, dsh): put
it in a tmux split or a side terminal. It polls /api/log and prints moth's new lines, with a dim one-liner for each
agent turn moth heard; lines you type go to /api/chat and the reply shows up through the same poll. It never posts
to /api/agent (the adapters do that), so moth never hears a turn twice. When moth is down it says so once and
keeps retrying, and starts over when moth restarts or the chat is reset (a new log epoch). Stdlib only, no torch:
safe to run anywhere. MOTH_URL overrides http://127.0.0.1:4747; the token is read from ~/.moth/token
(MOTH_TOKEN_FILE overrides), since the full log holds every agent reply. Terminal escape codes in agent and tool
text are stripped before printing. Several agent sessions: each agent note says which one ("claude · moth-repo
said: ..."), and --session <label> shows only that session's agent notes (moth's lines and yours always show).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import textwrap
import threading
import time
import urllib.error
import urllib.request
from collections import deque

try:
    import readline   # line editing, and lets a printed line redraw what you were typing
except ImportError:   # pragma: no cover  (Windows)
    readline = None
try:   # readline's own redraw of prompt + typed text (GNU readline and libedit both export it)
    import ctypes
    _redraw = ctypes.CDLL(readline.__file__).rl_forced_update_display
except Exception:  # noqa: BLE001  static build, Windows, no readline: redraw by hand
    _redraw = None

COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
DIM, LAMP, OFF = ("\033[2m", "\033[33m", "\033[0m") if COLOR else ("", "", "")   # same palette as moth.cli
POLL, MAX_BACKOFF = 1.0, 10.0
TOKEN_FILE = os.environ.get("MOTH_TOKEN_FILE") or os.path.join(os.path.expanduser("~"), ".moth", "token")
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)?|\x1b[@-_]|[\x00-\x08\x0b-\x1f\x7f-\x9f]")
NAMES = {"claude-code": "claude", "codex": "codex", "hermes": "hermes", "pi": "pi", "opencode": "opencode",
         "dsh": "deepseek"}
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # moth is local: never via a proxy


def clean(text) -> str:
    """No terminal escape codes from agent / tool text (they could move the cursor or retitle the terminal)."""
    return ANSI.sub("", str(text))


def token() -> str:
    try:
        with open(TOKEN_FILE) as f:
            return f.read().strip()
    except OSError:
        return ""


class Tail:
    def __init__(self, url: str, watch: bool, history: int, session: str | None = None):
        self.url, self.watch, self.history, self.only = url.rstrip("/"), watch, history, session
        self.cursor = None          # index into moth's log; None = not connected yet
        self.epoch = None           # moth's log epoch: a new one = restarted or reset, start over
        self.asleep = None          # None until the first poll answers or fails
        self.sent = deque(maxlen=20)   # our own lines, so their {"you"} echo isn't printed again
        self.prompting = False      # a "you> " prompt is on screen: print above it
        self.lock = threading.Lock()

    # --- printing -------------------------------------------------------------------------------------------
    def show(self, line: str):
        """Print above the prompt, then redraw the prompt and whatever was half typed."""
        with self.lock:
            if not self.prompting:
                print(line, flush=True)
                return
            sys.stdout.write(f"\r\033[2K{line}\n")
            if _redraw:
                sys.stdout.flush()
                _redraw()
            else:
                sys.stdout.write("you> " + (readline.get_line_buffer() if readline else ""))
            sys.stdout.flush()

    def wrap(self, prefix: str, text: str, color: str) -> str:
        width = max(20, shutil.get_terminal_size().columns - 1)
        lines = []
        for para in clean(text).splitlines() or [""]:
            lines += textwrap.wrap(para, width - len(prefix)) or [""]
        pad = " " * len(prefix)
        return "\n".join((f"{color}{prefix}{OFF}" if i == 0 else pad) + ln for i, ln in enumerate(lines))

    def one_line(self, label: str, text: str) -> str:
        width = max(20, shutil.get_terminal_size().columns - 1)
        flat = " ".join(clean(text).split())
        line = label + flat
        return DIM + (line if len(line) <= width else line[:width - 1] + "…") + OFF

    def mine(self, ev: dict) -> bool:
        """--session: the agent card's label (also without the " (ab12)" moth adds when two sessions share a folder),
        session id, or harness."""
        label, src = ev.get("label") or "", ev.get("source")
        return not self.only or self.only in (label, re.sub(r" \([^)]*\)$", "", label), ev.get("session"), src,
                                              NAMES.get(src))

    def render(self, ev: dict) -> str | None:
        if "say" in ev:
            return self.wrap("moth> ", ev["say"], LAMP)
        if "agent" in ev:
            if not self.mine(ev):
                return None
            who = NAMES.get(ev.get("source"), ev.get("source") or "agent")
            if ev.get("label"):
                who += f" · {ev['label']}"
            return self.one_line(f"{who} said{' (moth let it pass)' if ev.get('skipped') else ''}: ", ev["agent"])
        if "you" in ev:
            if ev["you"] in self.sent:
                self.sent.remove(ev["you"])
                return None
            return self.one_line("you> ", ev["you"])
        return None

    # --- talking to moth ------------------------------------------------------------------------------------
    def get_log(self, since: int) -> dict:
        q = f"since={since}" + (f"&epoch={self.epoch}" if self.epoch else "")
        req = urllib.request.Request(f"{self.url}/api/log?{q}", headers={"X-Moth-Token": token()})
        try:
            with _OPENER.open(req, timeout=3) as r:
                d = json.load(r)
        except urllib.error.HTTPError as e:
            e.close()
            if e.code == 401:
                raise PermissionError("moth wants its token") from None
            raise
        if not isinstance(d.get("events"), list) or not isinstance(d.get("next"), int):
            raise ValueError("not moth")
        return d

    def chat(self, text: str):
        """POST on a thread; the reply is printed by the poller, not from this response."""
        try:
            req = urllib.request.Request(f"{self.url}/api/chat", data=json.dumps({"text": text}).encode(),
                                         headers={"Content-Type": "application/json", "X-Moth-Token": token()})
            _OPENER.open(req, timeout=600).close()
        except Exception as e:  # noqa: BLE001  moth down: say so, never crash the prompt
            if text in self.sent:
                self.sent.remove(text)
            if isinstance(e, urllib.error.HTTPError):
                e.close()
            if isinstance(e, urllib.error.HTTPError) and e.code == 401:
                self.show(f"{DIM}(moth didn't take that: no token in {TOKEN_FILE}){OFF}")
            else:
                self.show(f"{DIM}(moth is asleep and didn't hear that){OFF}")

    def poll_once(self):
        if self.cursor is None:   # first contact: show a little recent history, then follow
            d = self.get_log(0)
            evs = d["events"][-self.history:] if self.history > 0 else []
        else:
            d = self.get_log(self.cursor)
            if d.get("epoch") != self.epoch or d["next"] < self.cursor:
                # moth restarted or the chat was reset: the server sent the new log from its start
                evs = d["events"] if d.get("epoch") != self.epoch else self.get_log(0)["events"]
            else:
                evs = d["events"]
        self.epoch = d.get("epoch")
        if self.asleep is not False:
            self.show(f"{DIM}moth is awake at {self.url}{'' if self.watch else '  (type to talk to it, /quit)'}{OFF}")
            self.asleep = False
        self.cursor = d["next"]
        for ev in evs:
            line = self.render(ev) if isinstance(ev, dict) else None
            if line is not None:
                self.show(line)

    def follow(self, stop: threading.Event):
        delay = POLL
        while not stop.is_set():
            try:
                self.poll_once()
                delay = POLL
            except PermissionError:
                if self.asleep is not True:
                    self.show(f"{DIM}moth won't show its log without its token ({TOKEN_FILE}). retrying...{OFF}")
                    self.asleep = True
                delay = min(delay * 2, MAX_BACKOFF)
            except Exception:  # noqa: BLE001  down, restarting, or not moth at all
                if self.asleep is not True:
                    self.show(f"{DIM}moth is asleep (start it: python -m moth.serve). retrying...{OFF}")
                    self.asleep = True
                delay = min(delay * 2, MAX_BACKOFF)
            stop.wait(delay)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Show moth's reactions and chat with it, next to any coding agent.")
    ap.add_argument("--url", default=os.environ.get("MOTH_URL", "http://127.0.0.1:4747"))
    ap.add_argument("--watch", action="store_true", help="status pane only: no input prompt")
    ap.add_argument("--history", type=int, default=3, help="recent events to show on connect (default 3)")
    ap.add_argument("--session", default=None, metavar="LABEL",
                    help="only this agent session's notes (its label, e.g. the project folder, or its session id)")
    a = ap.parse_args(argv)
    try:   # a pipe on Windows is cp1252: an unencodable "…" or emoji must print as "?", not read as "moth is asleep"
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass
    t, stop = Tail(a.url, a.watch, a.history, a.session), threading.Event()
    try:
        if a.watch:
            t.follow(stop)
            return
        try:
            t.poll_once()   # first contact before the prompt is drawn, so the two never interleave
        except Exception:  # noqa: BLE001  the follower reports "asleep"
            pass
        t.prompting = sys.stdin.isatty() and sys.stdout.isatty()
        threading.Thread(target=t.follow, args=(stop,), daemon=True).start()
        while True:
            text = input("you> ").strip()
            if text in ("/quit", "/exit"):
                break
            if text:
                t.sent.append(text[:500])   # serve.py keeps the first 500 chars
                threading.Thread(target=t.chat, args=(text[:500],), daemon=True).start()
    except (EOFError, KeyboardInterrupt):
        print()
    finally:
        stop.set()


if __name__ == "__main__":
    main()
