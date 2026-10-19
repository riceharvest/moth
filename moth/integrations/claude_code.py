"""Claude Code `Stop` hook: send Claude's final message of each turn to moth (one-way; moth never talks back).

Install (see README.md): a `Stop` command hook in the foreground (no "async"), "timeout": 3. It posts before it exits:
a one-shot `claude -p` kills a background ("async": true) hook when it exits, before the POST (seen live
2026-10-18), and the wait is short: moth only queues the text, and we give up on it after 1 s.
Stop fires only for the main agent; subagents fire `SubagentStop` instead, which we don't hook.

The text comes from `last_assistant_message` in the Stop input (docs: code.claude.com/docs/en/hooks#stop-input;
the transcript "isn't guaranteed to include the final message at Stop time"). Only when an older Claude Code
sends no such key do we read `transcript_path`, and then only the current turn, so an earlier turn's reply is
never re-sent. The session is the Stop input's `session_id`, its label the basename of `cwd` (the project folder),
so moth can tell several sessions apart. Prints nothing, always exits 0, gives up after 1 s. MOTH_URL overrides http://127.0.0.1:4747;
the token comes from ~/.moth/token (MOTH_TOKEN_FILE overrides).
"""
import json
import os
import sys

URL = os.environ.get("MOTH_URL", "http://127.0.0.1:4747").rstrip("/") + "/api/agent"
TOKEN_FILE = os.environ.get("MOTH_TOKEN_FILE") or os.path.join(os.path.expanduser("~"), ".moth", "token")


def _token() -> str:
    """moth's per-install token (made by moth.serve on first run); read on every post, it may be newer than us."""
    try:
        with open(TOKEN_FILE) as f:
            return f.read().strip()
    except OSError:
        return ""


def _is_prompt(d: dict) -> bool:
    """A real user prompt (turn boundary), not a tool_result line, which the transcript also logs as "user"."""
    content = (d.get("message") or {}).get("content")
    if isinstance(content, str):
        return True
    return isinstance(content, list) and any(isinstance(b, dict) and b.get("type") != "tool_result" for b in content)


def last_text(transcript: str) -> str:
    text = ""
    with open(os.path.expanduser(transcript), encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if not isinstance(d, dict) or d.get("isSidechain"):
                continue
            if d.get("type") == "user" and _is_prompt(d):
                text = ""                       # new turn: forget the previous turn's reply
            elif d.get("type") == "assistant":
                blocks = (d.get("message") or {}).get("content") or []
                parts = [b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text"]
                if any(p.strip() for p in parts):
                    text = "\n".join(parts)     # keep the newest assistant text of this turn
    return text


def label_of(cwd, session) -> str:
    """The project folder's name; without a usable cwd, a short session id."""
    name = cwd.rstrip("/\\").replace("\\", "/").rsplit("/", 1)[-1] if isinstance(cwd, str) else ""
    return name or (session or "")[:8]


def http_post(url: str, data: bytes, headers: dict, timeout: float = 1.0, method: str = "POST") -> bytes:
    """POST (or `method`) and wait for moth's answer, returned as the body bytes (never via a proxy: moth is local).
    Plain http:// goes over a bare socket: this runs on every agent turn, and importing urllib.request alone takes
    longer than the whole post."""
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


def post(text: str, session=None, label=None) -> None:
    body = {"text": text, "source": "claude-code", "session": session, "label": label}
    http_post(URL, json.dumps({k: v for k, v in body.items() if v}).encode(),
              {"Content-Type": "application/json", "X-Moth-Token": _token()})


def main() -> None:
    try:
        hook = json.load(sys.stdin.buffer)   # bytes: json detects UTF-8 (the locale may be cp1252 on Windows)
        if "last_assistant_message" in hook:
            text = hook["last_assistant_message"] or ""
        else:
            text = last_text(hook["transcript_path"])
        if text.strip():
            session = hook.get("session_id") if isinstance(hook.get("session_id"), str) else None
            post(text, session, label_of(hook.get("cwd"), session))
    except BaseException:  # noqa: BLE001  moth down, timeout, odd input: never bother Claude
        pass


if __name__ == "__main__":
    main()
    sys.exit(0)
