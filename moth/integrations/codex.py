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
legacy_notify.rs). Prints nothing, always exits 0, gives up after 1 s. MOTH_URL overrides http://127.0.0.1:4747; the token comes from
~/.moth/token (MOTH_TOKEN_FILE overrides).
"""
import json
import os
import sys
import urllib.request

URL = os.environ.get("MOTH_URL", "http://127.0.0.1:4747").rstrip("/") + "/api/agent"
TOKEN_FILE = os.environ.get("MOTH_TOKEN_FILE") or os.path.join(os.path.expanduser("~"), ".moth", "token")
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # moth is local: never via a proxy


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


def main() -> None:
    try:
        text, session, label = final_turn(sys.argv, sys.stdin.buffer)   # bytes: UTF-8 whatever the locale
        if text.strip():
            body = {"text": text, "source": "codex", "session": session, "label": label}
            req = urllib.request.Request(URL, data=json.dumps({k: v for k, v in body.items() if v}).encode(),
                                         headers={"Content-Type": "application/json", "X-Moth-Token": _token()})
            _OPENER.open(req, timeout=1).close()
    except BaseException:  # noqa: BLE001  moth down, timeout, odd input: never bother Codex
        pass


if __name__ == "__main__":
    main()
    sys.exit(0)
