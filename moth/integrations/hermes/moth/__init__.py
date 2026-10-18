"""moth plugin for Hermes Agent: send the agent's final reply of each turn to moth (one-way; moth's reply is shown
by desktop/plugin.js on the Hermes Desktop pet, or by `python -m moth.tail`).

Install + enable (see ../../README.md):
  cp -r moth/integrations/hermes/moth ~/.hermes/plugins/moth
  hermes plugins enable moth          # or add `moth` to plugins.enabled in ~/.hermes/config.yaml

Hook: post_llm_call(session_id, user_message, assistant_response, conversation_history, model, platform, **kw)
fires once per turn after the tool loop, only for turns that produced a final response and were not interrupted
(website/docs/user-guide/features/hooks.md#post_llm_call; fired in agent/turn_finalizer.py _apply_output_hooks).
delegate_task children run with platform="subagent" (tools/delegate_tool.py) and are skipped.
Session and label, so moth can tell several sessions apart: the hook's `session_id`, and the basename of the turn's
working directory (agent/runtime_cwd.py: the gateway's per-session cwd, else TERMINAL_CWD, else the launch dir),
read in the hook itself, since the per-session cwd is a ContextVar the posting thread wouldn't see.
The POST runs on a daemon thread with a 1 s timeout, so the agent never waits on moth. MOTH_URL overrides
http://127.0.0.1:4747; the token comes from ~/.moth/token (MOTH_TOKEN_FILE overrides).

No /moth slash command, on purpose: talking to moth must never reach the agent, and in the messaging gateway a
plugin command sent while a turn runs is not dispatched to the plugin. It falls through to the busy lane and is
handed to the running agent as redirect / steer / interrupt text (gateway/run_inbound.py _hm_busy_slash_or_photo
-> _hm_busy_steer / _hm_busy_interrupt; resolve_command only knows built-ins). Chat with moth in
`python -m moth.tail` instead. desktop/plugin.js is the Hermes Desktop half: moth's lines on the desktop pet.
"""
import json
import os
import threading
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


def _cwd() -> str:
    """The turn's working directory as Hermes itself resolves it; any trouble: TERMINAL_CWD or the launch dir."""
    try:
        from agent.runtime_cwd import scope_terminal_cwd, scoped_session_cwd
        return scoped_session_cwd() or scope_terminal_cwd() or os.getcwd()
    except Exception:  # noqa: BLE001  older Hermes, or a refusing terminal scope
        return os.environ.get("TERMINAL_CWD") or os.getcwd()


def label_of(cwd, session) -> str:
    """The project folder's name; without a usable cwd, a short session id."""
    name = cwd.rstrip("/\\").replace("\\", "/").rsplit("/", 1)[-1] if isinstance(cwd, str) else ""
    return name or (session or "")[:8]


def _post(text, session=None, label=None):
    try:
        body = {"text": text, "source": "hermes", "session": session, "label": label}
        req = urllib.request.Request(URL, data=json.dumps({k: v for k, v in body.items() if v}).encode(),
                                     headers={"Content-Type": "application/json", "X-Moth-Token": _token()})
        _OPENER.open(req, timeout=1).close()
    except Exception:  # noqa: BLE001  moth down or slow: drop it
        pass


def on_post_llm_call(assistant_response=None, platform="", session_id="", **kwargs):
    try:
        if platform == "subagent" or not isinstance(assistant_response, str) or not assistant_response.strip():
            return None
        session = session_id if isinstance(session_id, str) and session_id else None
        try:
            label = label_of(_cwd(), session)
        except Exception:  # noqa: BLE001
            label = (session or "")[:8]
        threading.Thread(target=_post, args=(assistant_response, session, label), daemon=True,
                         name="moth-post").start()
    except Exception:  # noqa: BLE001
        pass
    return None


def register(ctx):
    ctx.register_hook("post_llm_call", on_post_llm_call)
