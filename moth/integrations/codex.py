"""Codex CLI adapter: send Codex's final message of each turn to moth (one-way; moth never talks back).

Two ways in; install ONE of them (both would post every reply twice). See README.md.
  Stop hook (preferred): ~/.codex/hooks.json, "Stop" event, JSON on stdin with `last_assistant_message`. Run it in
      the foreground (no "async"), "timeout": 3: `codex exec` drops a background hook when it exits, before the POST
      (seen live 2026-10-18; the interactive TUI posted either way). We post before exiting, giving up after 1 s.
      Codex runs a hooks.json hook only once it is approved in the interactive `codex` (/hooks), and again after
      every edit of the file (the trust is a hash of the entry); unapproved, `codex exec` skips it silently.
      Codex runs user Stop hooks for root turns only: thread-spawned subagents fire `SubagentStop`, internal
      subagents and memory consolidation skip user hooks (codex-rs/core/src/hook_runtime.rs run_turn_stop_hooks,
      codex-rs/hooks/src/events/stop.rs select_handlers; rust-v0.163.0).
  notify (legacy): `notify = [...]` in ~/.codex/config.toml. Codex appends one JSON argument
      {"type": "agent-turn-complete", ..., "last-assistant-message": ...} (codex-rs/hooks/src/legacy_notify.rs)
      and spawns us without waiting. It is not gated, so subagent turns post too.

Session and label, so moth can tell several sessions apart: the hook's `session_id` / notify's `thread-id`, and the
basename of `cwd` (the project folder), present in both (codex-rs/hooks/src/events/stop.rs StopRequest,
legacy_notify.rs). Always exits 0, gives up after 1 s. MOTH_URL overrides http://127.0.0.1:4747; the token comes from
~/.moth/token (MOTH_TOKEN_FILE overrides).

Display (moth's setting `show_in_harness`, off by default: MOTH_SHOW_IN_HARNESS, else "show_in_harness" in
~/.moth/config.json, MOTH_CONFIG_FILE overrides, read every turn). Off, the hook prints nothing. On, the Stop hook
prints {"systemMessage": "moth: <line>"} with moth's newest reaction to THIS session, which the Codex TUI draws under
the turn as "↳ Hook · moth: ...". A Stop hook's systemMessage becomes a Warning entry of the hook run
(codex-rs/hooks/src/events/stop.rs) that only the UI renders (tui/src/history_cell/hook_cell.rs); it is not added
to the conversation, so the model never sees it (checked with real `codex` turns against a fake model API:
tests/e2e/run_e2e.py, codex_display). `codex exec` doesn't print it. The line comes from GET /api/says?source=codex
&session=<id> (token-free, moth's own lines only), with a cursor per session in ~/.moth/codex-display.json:
  - before posting, any reaction moth made since this session's last turn (moth's default pacing reacts a few
    seconds after a turn, in a batch window, so that one shows at the end of the next turn);
  - after posting, when moth queued this turn right away (`"status": "queued"`, e.g. --react-every 0), up to
    WAIT s for its reaction.
Every request is bounded and the whole display stays within BUDGET s: a slow or down moth costs Codex no more
than the plain post does (a failed first request skips the display). The notify path never prints (Codex doesn't
read its output).
"""
import json
import os
import sys

URL = os.environ.get("MOTH_URL", "http://127.0.0.1:4747").rstrip("/") + "/api/agent"
TOKEN_FILE = os.environ.get("MOTH_TOKEN_FILE") or os.path.join(os.path.expanduser("~"), ".moth", "token")
SAYS = URL[: -len("/api/agent")] + "/api/says"
STATE = os.path.join(os.path.dirname(TOKEN_FILE), "codex-display.json")   # display cursors, next to the token
WAIT, BUDGET = 1.2, 2.0   # s: for a queued reaction; for the whole display (Codex's hook timeout is 3 s)


def _token() -> str:
    """moth's per-install token (made by moth.serve on first run); read on every post, it may be newer than us."""
    try:
        with open(TOKEN_FILE) as f:
            return f.read().strip()
    except OSError:
        return ""


def label_of(cwd, session) -> str:
    """The project folder's name; without a usable cwd, a short session id."""
    name = cwd.rstrip("/\\").replace("\\", "/").rsplit("/", 1)[-1] if isinstance(cwd, str) else ""
    return name or (session or "")[:8]


def final_turn(argv: list[str], stdin) -> tuple[str, str | None, str]:
    """(final text, session id, label)."""
    if len(argv) > 1 and argv[-1].lstrip().startswith("{"):        # notify: payload is the last argument
        d = json.loads(argv[-1])
        text = (d.get("last-assistant-message") or "") if d.get("type") == "agent-turn-complete" else ""
        session = d.get("thread-id")
    else:
        d = json.load(stdin)                                         # hook: payload on stdin
        text = (d.get("last_assistant_message") or "") if d.get("hook_event_name", "Stop") == "Stop" else ""
        session = d.get("session_id")
    session = session if isinstance(session, str) else None
    return text, session, label_of(d.get("cwd"), session)


def http_post(url: str, data: bytes, headers: dict, timeout: float = 1.0, method: str = "POST") -> bytes:
    """POST (or `method`) and wait for moth's answer, returned as the body bytes (never via a proxy: moth is local).
    Plain http:// goes over a bare socket: this runs on every agent turn, and importing urllib.request alone takes
    longer than the whole post.
    (The same function as claude_code.py's: each adapter is one standalone file.)"""
    if not url.lower().startswith("http://"):   # https (a moth behind something): the standard client
        import urllib.request
        req = urllib.request.Request(url, data=data or None, headers=headers, method=method)
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=timeout) as r:
            return r.read()
    import socket
    import time
    netloc, _, path = url[7:].partition("/")
    host, _, port = netloc.rpartition(":") if netloc.rpartition(":")[2].isdigit() else (netloc, "", "80")
    if any("\r" in str(v) or "\n" in str(v) for v in headers.values()):
        raise ValueError("bad header")   # a token with a line break in it: refused, as http.client does
    head = "".join(f"{k}: {v}\r\n" for k, v in {**headers, "Content-Length": len(data)}.items())
    req = f"{method} /{path} HTTP/1.1\r\nHost: {netloc}\r\n{head}Connection: close\r\n\r\n".encode("latin-1") + data
    got = []
    with socket.create_connection((host.strip("[]"), int(port)), timeout=timeout) as s:
        end = time.monotonic() + timeout   # as urllib: up to `timeout` to connect, then up to `timeout` for the answer
        s.sendall(req)
        while True:   # moth's answer, to the end (it has the turn before we go)
            s.settimeout(max(0.001, end - time.monotonic()))
            chunk = s.recv(4096)
            if not chunk:
                break
            got.append(chunk)
    return b"".join(got).partition(b"\r\n\r\n")[2]


def show_in_harness() -> bool:
    """moth's `show_in_harness` setting, the same spellings as moth/config.py; anything unreadable = off."""
    flag = lambda v: {"1": True, "on": True, "true": True, "yes": True, "0": False, "off": False, "false": False,
                      "no": False}.get(v.strip().lower()) if isinstance(v, str) else v if isinstance(v, bool) else None
    env = flag(os.environ.get("MOTH_SHOW_IN_HARNESS", ""))
    if env is not None:
        return env
    try:
        path = os.environ.get("MOTH_CONFIG_FILE") or os.path.join(os.path.expanduser("~"), ".moth", "config.json")
        with open(path, encoding="utf-8") as f:
            return flag(json.load(f).get("show_in_harness")) is True
    except (OSError, ValueError, AttributeError):
        return False


def clean(say: str) -> str:
    """One plain line: no terminal escapes or control characters (the TUI would draw them), whitespace collapsed."""
    import re
    say = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)?|\x1b[@-_]", "", say)
    return " ".join("".join(c if c.isprintable() else " " for c in say).split())[:300]


class Display:
    """moth's lines for one Codex session: a cursor into moth's log (index + epoch), kept between turns."""

    def __init__(self, session: str, deadline: float):
        self.session, self.deadline = session, deadline
        try:
            with open(STATE, encoding="utf-8") as f:
                self.all = json.load(f)
            cur = self.all.get(session)
            self.next, self.epoch = (int(cur["next"]), str(cur["epoch"])) if isinstance(cur, dict) else (None, None)
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            self.all, self.next, self.epoch = {}, None, None
        if not isinstance(self.all, dict):
            self.all = {}

    def left(self) -> float:
        import time
        return self.deadline - time.monotonic()

    def poll(self) -> list[str]:
        """moth's new lines for this session since the cursor (raises when moth doesn't answer in time)."""
        from urllib.parse import quote
        q = f"source=codex&session={quote(self.session, safe='')}&since={self.next if self.next is not None else 10**15}"
        if self.epoch:
            q += f"&epoch={quote(self.epoch, safe='')}"
        d = json.loads(http_post(f"{SAYS}?{q}", b"", {}, timeout=max(0.05, min(0.5, self.left())), method="GET"))
        if not isinstance(d.get("next"), int) or not isinstance(d.get("events"), list):
            raise ValueError("not moth")
        fresh = self.next is not None and d.get("epoch") == self.epoch   # else (first turn, moth restarted): baseline
        self.next, self.epoch = d["next"], str(d.get("epoch") or "")
        return [e["say"] for e in d["events"] if fresh and isinstance(e, dict) and isinstance(e.get("say"), str)]

    def save(self):
        self.all.pop(self.session, None)
        self.all[self.session] = {"next": self.next, "epoch": self.epoch}
        keep = dict(list(self.all.items())[-64:])   # the newest sessions only
        tmp = f"{STATE}.{os.getpid()}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(keep, f)
            os.replace(tmp, STATE)
        except OSError:
            pass


def post_and_show(text: str, session: str | None, label: str, hook: bool) -> str | None:
    """Post the turn; with the display on (a Stop hook with a session id), moth's newest line for it, or None."""
    import time
    body = json.dumps({k: v for k, v in {"text": text, "source": "codex", "session": session,
                                          "label": label}.items() if v}).encode()
    headers = {"Content-Type": "application/json", "X-Moth-Token": _token()}
    if not (hook and session and show_in_harness()):
        http_post(URL, body, headers)
        return None
    disp = Display(session, time.monotonic() + BUDGET)
    try:
        says = disp.poll()   # what moth said about this session's earlier turns since the last one we showed
    except Exception:  # noqa: BLE001  moth down or slow: no display this turn, just the post
        http_post(URL, body, headers)
        return None
    try:
        status = json.loads(http_post(URL, body, headers, timeout=max(0.05, min(1.0, disp.left())))).get("status")
        if status == "queued":   # moth reacts to this turn right away: wait a little for it
            until = time.monotonic() + WAIT
            while disp.left() > 0.15 and time.monotonic() < until:
                time.sleep(0.1)
                new = disp.poll()
                if new:
                    says += new
                    break
    except Exception:  # noqa: BLE001  keep what we have
        pass
    disp.save()
    return clean(says[-1]) if says else None


def main() -> None:
    try:
        is_hook = not (len(sys.argv) > 1 and sys.argv[-1].lstrip().startswith("{"))
        text, session, label = final_turn(sys.argv, sys.stdin.buffer)   # bytes: UTF-8 whatever the locale
        if text.strip():
            say = post_and_show(text, session, label, is_hook)
            if say:   # only the Codex UI draws it; it never reaches the model (see the module docstring)
                sys.stdout.write(json.dumps({"systemMessage": f"moth: {say}"}) + "\n")
                sys.stdout.flush()
    except BaseException:  # noqa: BLE001  moth down, timeout, odd input: never bother Codex
        pass


if __name__ == "__main__":
    main()
    sys.exit(0)
