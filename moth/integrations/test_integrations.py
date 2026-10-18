"""Offline checks for the moth harness adapters: fake harness payloads against a throwaway stdlib stub of moth.

Each adapter must post exactly the turn's final assistant text, skip subagent / aborted / text-less turns, print
nothing, exit 0, and give up within ~1 s when moth is slow (stub sleeps 2 s) or down (closed port).
The JS adapters run under `node` with fake `pi` / `ctx` / plugin inputs; a stray promise rejection would crash node.
`Display` covers the read-back half against a stub with a real log (LogStub): moth tail, the Hermes desktop pet
(and that Hermes registers no /moth), the Pi widget and /moth, the OpenCode toast, and the `show_in_harness` setting
(moth/config.py): off by default (still posted, nothing polled or drawn), env beats file, and a live toggle of the
file applies from the next turn without a restart. `Serve` checks serve.py's real
HTTP handler (torch faked out): Host allow-list, CORS, the per-install token, POST hygiene, /api/says, X-Frame-Options,
and that "new home" only deletes a folder moth made. `AdapterToLog` runs every adapter against the real handler and
the real Moth.hear_agent: the session ids and labels from each harness's payload end up on the log's cards.
Every adapter must send the token (from MOTH_TOKEN_FILE, set to a
throwaway file here) and its harness name as `source`, and post past an HTTP proxy set in the environment.

  python3 moth/integrations/test_integrations.py      (from ~/moth; about 50 s, mostly deliberate waits; only 127.0.0.1)
"""
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOKEN = "test-token-not-secret"
_tf = tempfile.NamedTemporaryFile("w", suffix=".token", delete=False)
_tf.write(TOKEN + "\n")
_tf.close()
os.environ["MOTH_TOKEN_FILE"] = _tf.name   # every adapter and display started from here reads this one
# moth's show_in_harness setting: never the user's real ~/.moth/config.json or shell; a file that doesn't exist = off
os.environ["MOTH_CONFIG_FILE"] = os.path.join(tempfile.mkdtemp(prefix="moth-cfg-"), "absent.json")
os.environ.pop("MOTH_SHOW_IN_HARNESS", None)
# "gives up within ~1 s": GIVE_UP bounds an adapter run (1 s timeout + interpreter start), SLOW is the slow stub's delay.
# Cold starts and refused loopback connects are slower on Windows / macOS runners, so both grow there, SLOW > GIVE_UP.
GIVE_UP, SLOW = (1.8, 2.0) if sys.platform.startswith("linux") else (3.0, 4.0)


class Stub:
    """Fake moth: records every POSTed {"text", "source"[, "session", "label"]} that carries the token; optionally
    answers only after `delay` seconds."""

    def __init__(self, delay=0.0):
        self.texts, self.sources, self.metas, stub = [], [], [], self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                assert self.path == "/api/agent" and {"text", "source"} <= set(body) <= {"text", "source", "session",
                                                                                         "label"}, (self.path, body)
                assert all(isinstance(v, str) and v for v in body.values()), body
                assert self.headers["X-Moth-Token"] == TOKEN and self.headers["Content-Type"] == "application/json"
                stub.texts.append(body["text"])
                stub.sources.append(body["source"])
                stub.metas.append((body.get("session"), body.get("label")))
                time.sleep(delay)
                try:
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"{}")
                except OSError:
                    pass

            def log_message(self, *a):
                pass

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.srv.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.srv.server_port}"
        threading.Thread(target=self.srv.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


def closed_url():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return f"http://127.0.0.1:{port}"


def run(cmd, url, stdin=""):
    t = time.monotonic()
    p = subprocess.run(cmd, input=stdin, capture_output=True, text=True, encoding="utf-8", timeout=10,
                       env={**os.environ, "MOTH_URL": url})
    return p, time.monotonic() - t


class Base(unittest.TestCase):
    def setUp(self):
        self.ok, self.slow = Stub(), Stub(delay=SLOW)

    def tearDown(self):
        self.ok.close()
        self.slow.close()

    def quiet(self, p, took):
        self.assertEqual((p.returncode, p.stdout, p.stderr), (0, "", ""))
        self.assertLess(took, GIVE_UP)


class ClaudeCode(Base):
    HOOK = [sys.executable, str(HERE / "claude_code.py")]

    def test_last_assistant_message(self):
        p, took = run(self.HOOK, self.ok.url, json.dumps({"hook_event_name": "Stop", "transcript_path": "/nope",
                                                          "session_id": "0f3c9a2e-aaaa", "cwd": "/home/u/moth-repo/",
                                                          "last_assistant_message": "## Done\nfixed **it**"}))
        self.quiet(p, took)
        self.assertEqual(self.ok.texts, ["## Done\nfixed **it**"])
        self.assertEqual(self.ok.sources, ["claude-code"])
        self.assertEqual(self.ok.metas, [("0f3c9a2e-aaaa", "moth-repo")])

    def test_session_label_fallbacks(self):
        for hook, meta in (({"session_id": "0f3c9a2e-aaaa", "cwd": "/"}, ("0f3c9a2e-aaaa", "0f3c9a2e")),
                           ({"session_id": 5, "cwd": "C:\\Users\\u\\proj"}, (None, "proj")),
                           ({}, (None, None))):
            with self.subTest(hook=hook):
                self.ok.metas.clear()
                self.quiet(*run(self.HOOK, self.ok.url, json.dumps({**hook, "last_assistant_message": "hi"})))
                self.assertEqual(self.ok.metas, [meta])

    def test_ignores_proxy_env(self):
        """A corporate proxy in the environment must not swallow a post to 127.0.0.1."""
        proxy = closed_url()
        p = subprocess.run(self.HOOK, input=json.dumps({"last_assistant_message": "via no proxy"}), text=True,
                           capture_output=True, timeout=10, env={**os.environ, "MOTH_URL": self.ok.url,
                           "http_proxy": proxy, "HTTP_PROXY": proxy, "no_proxy": "", "NO_PROXY": ""})
        self.assertEqual((p.returncode, self.ok.texts), (0, ["via no proxy"]))

    def test_empty_message_posts_nothing(self):
        p, took = run(self.HOOK, self.ok.url, json.dumps({"last_assistant_message": "", "transcript_path": "/nope"}))
        self.quiet(p, took)
        self.assertEqual(self.ok.texts, [])

    def test_utf8_payloads(self):
        """Harnesses write raw UTF-8 JSON; the hooks must not decode it with the locale's code page (cp1252 on
        Windows), for the stdin payload (Claude Code, Codex) and Claude Code's transcript fallback alike."""
        text = "it’s fixed — naïve café, 🦋 ok"
        with tempfile.NamedTemporaryFile("wb", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": text}]}},
                               ensure_ascii=False).encode("utf-8") + b"\n")
        try:
            for hook, payload in ((self.HOOK, {"last_assistant_message": text}), (self.HOOK, {"transcript_path": f.name}),
                                  (Codex.HOOK, {"hook_event_name": "Stop", "last_assistant_message": text})):
                with self.subTest(hook=hook[-1], keys=sorted(payload)):
                    self.ok.texts.clear()
                    p = subprocess.run(hook, input=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                                       capture_output=True, timeout=10, env={**os.environ, "MOTH_URL": self.ok.url})
                    self.assertEqual((p.returncode, p.stdout, p.stderr), (0, b"", b""))
                    self.assertEqual(self.ok.texts, [text])
        finally:
            os.unlink(f.name)

    def test_transcript_fallback_current_turn_only(self):
        lines = [
            {"type": "user", "message": {"content": "first prompt"}},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "old reply"}]}},
            {"type": "user", "message": {"content": "second prompt"}},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "looking"},
                                                          {"type": "tool_use", "id": "t1"}]}},
            {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "t1"}]}},
            {"type": "assistant", "isSidechain": True, "message": {"content": [{"type": "text", "text": "sub"}]}},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "new reply"}]}},
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            f.write("\n".join(json.dumps(x) for x in lines) + "\nnot json\n")
        p, took = run(self.HOOK, self.ok.url, json.dumps({"transcript_path": f.name}))
        self.quiet(p, took)
        self.assertEqual(self.ok.texts, ["new reply"])
        with open(f.name, "a") as g:                      # a turn with no text yet: nothing, not "new reply" again
            g.write(json.dumps({"type": "user", "message": {"content": "third"}}) + "\n")
        p, took = run(self.HOOK, self.ok.url, json.dumps({"transcript_path": f.name}))
        self.quiet(p, took)
        self.assertEqual(self.ok.texts, ["new reply"])
        os.unlink(f.name)

    def test_moth_slow_down_or_garbage(self):
        stdin = json.dumps({"last_assistant_message": "hi"})
        p, took = run(self.HOOK, self.slow.url, stdin)
        self.quiet(p, took)
        self.assertEqual(self.slow.texts, ["hi"])
        self.quiet(*run(self.HOOK, closed_url(), stdin))
        self.quiet(*run(self.HOOK, self.ok.url, "not json"))


class Codex(Base):
    HOOK = [sys.executable, str(HERE / "codex.py")]

    def test_stop_hook(self):
        p, took = run(self.HOOK, self.ok.url, json.dumps({"hook_event_name": "Stop", "turn_id": "1",
                                                          "session_id": "cx-thread-1", "cwd": "/srv/api",
                                                          "last_assistant_message": "codex done"}))
        self.quiet(p, took)
        self.assertEqual(self.ok.texts, ["codex done"])
        self.assertEqual(self.ok.sources, ["codex"])
        self.assertEqual(self.ok.metas, [("cx-thread-1", "api")])

    def test_subagent_stop_and_null_skipped(self):
        self.quiet(*run(self.HOOK, self.ok.url, json.dumps({"hook_event_name": "SubagentStop",
                                                            "last_assistant_message": "sub"})))
        self.quiet(*run(self.HOOK, self.ok.url, json.dumps({"hook_event_name": "Stop",
                                                            "last_assistant_message": None})))
        self.assertEqual(self.ok.texts, [])

    def test_legacy_notify_argv(self):
        payload = {"type": "agent-turn-complete", "thread-id": "t", "turn-id": "1", "cwd": "/x",
                   "input-messages": ["do it"], "last-assistant-message": "notify done"}
        p, took = run(self.HOOK + [json.dumps(payload)], self.ok.url)
        self.quiet(p, took)
        self.assertEqual(self.ok.texts, ["notify done"])
        self.assertEqual(self.ok.metas, [("t", "x")])

    def test_moth_slow_or_down(self):
        stdin = json.dumps({"hook_event_name": "Stop", "last_assistant_message": "hi"})
        self.quiet(*run(self.HOOK, self.slow.url, stdin))
        self.assertEqual(self.slow.texts, ["hi"])
        self.quiet(*run(self.HOOK, closed_url(), stdin))


class Hermes(Base):
    def load(self, url):
        spec = importlib.util.spec_from_file_location("moth_hermes", HERE / "hermes" / "moth" / "__init__.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.URL = url + "/api/agent"
        return mod

    def wait(self, stub, n):
        for _ in range(100):
            if len(stub.texts) >= n:
                return
            time.sleep(0.01)

    def test_posts_final_skips_subagent(self):
        mod = self.load(self.ok.url)
        hooks = []
        mod.register(type("Ctx", (), {"register_hook": lambda self, n, f: hooks.append(n)})())
        self.assertEqual(hooks, ["post_llm_call"])
        kw = dict(session_id="s", user_message="hi", conversation_history=[], model="m")
        mod.on_post_llm_call(assistant_response="sub reply", platform="subagent", **kw)
        mod.on_post_llm_call(assistant_response="   ", platform="cli", **kw)
        mod.on_post_llm_call(assistant_response="hermes reply", platform="cli", **kw)
        self.wait(self.ok, 1)
        time.sleep(0.05)
        self.assertEqual(self.ok.texts, ["hermes reply"])
        self.assertEqual(self.ok.sources, ["hermes"])
        self.assertEqual(self.ok.metas, [("s", os.path.basename(os.environ.get("TERMINAL_CWD") or os.getcwd()))])

    def test_label_from_hermes_cwd(self):
        """Hermes' own per-session cwd (agent.runtime_cwd, a ContextVar) is read in the hook, before the thread."""
        import types
        from unittest import mock
        fake = types.ModuleType("agent.runtime_cwd")
        fake.scoped_session_cwd, fake.scope_terminal_cwd = (lambda: "/gw/sess-proj/"), (lambda: "/term/other")
        mod = self.load(self.ok.url)
        with mock.patch.dict(sys.modules, {"agent": types.ModuleType("agent"), "agent.runtime_cwd": fake}):
            mod.on_post_llm_call(assistant_response="one", platform="telegram", session_id="20261018_1200_ab")
            fake.scoped_session_cwd = lambda: ""
            mod.on_post_llm_call(assistant_response="two", platform="cli", session_id="")

            def refuse():
                raise RuntimeError("terminal scope refused")
            fake.scope_terminal_cwd = refuse
            with mock.patch.dict(os.environ, {"TERMINAL_CWD": "/env/fallback"}):
                mod.on_post_llm_call(assistant_response="three", platform="cli", session_id="s3")
        self.wait(self.ok, 3)
        got = dict(zip(self.ok.texts, self.ok.metas))
        self.assertEqual(got, {"one": ("20261018_1200_ab", "sess-proj"), "two": (None, "other"),
                               "three": ("s3", "fallback")})

    def test_never_blocks(self):
        for url in (self.slow.url, closed_url()):
            mod = self.load(url)
            t = time.monotonic()
            self.assertIsNone(mod.on_post_llm_call(assistant_response="hi", platform="cli"))
            self.assertLess(time.monotonic() - t, 0.1)
        self.wait(self.slow, 1)
        self.assertEqual(self.slow.texts, ["hi"])


# --- JS adapters: one node driver per harness, fed fake events ------------------------------------------------
PI = r"""
import ext from "./adapter.mjs";
const h = {}; ext({ on: (n, f) => { (h[n] ||= []).push(f); } });
const ctx = { cwd: "/home/u/pi-proj", sessionManager: { getSessionId: () => "pi-sess-1" } };
const fire = (n, e) => h[n].forEach((f) => f(e, ctx));
const A = (text, stopReason = "stop") => ({ role: "assistant", stopReason,
  content: [{ type: "thinking", thinking: "hmm" }, { type: "text", text }] });
fire("agent_start", {});
fire("agent_end", { messages: [{ role: "user", content: "hi" },
  { role: "assistant", stopReason: "toolUse", content: [{ type: "text", text: "pi-step" }, { type: "toolCall" }] },
  { role: "toolResult", content: [] }, A("pi-final")] });
fire("agent_settled", { aborted: false });
fire("agent_start", {}); fire("agent_end", { messages: [A("pi-aborted")] }); fire("agent_settled", { aborted: true });
fire("agent_start", {}); fire("agent_end", { messages: [A("pi-error", "error")] }); fire("agent_settled", { aborted: false });
fire("agent_start", {}); fire("agent_end", { messages: [A("x", "error")] });
fire("agent_start", {}); fire("agent_end", { messages: [A("pi-retried")] }); fire("agent_settled", { aborted: false });
fire("agent_end", null); fire("agent_settled", undefined);
"""
PI_EXPECT = ["pi-final", "pi-retried"]

OPENCODE = r"""
import { Moth } from "./adapter.mjs";
const hooks = await Moth({ client: {}, project: {}, directory: "/x/oc-plugin-dir", worktree: "/x", $: null });
const ev = (type, properties) => hooks.event({ event: { type, properties } });
const msg = (id, sid, role, extra = {}) => ev("message.updated", { info: { id, sessionID: sid, role, ...extra } });
const txt = (id, mid, sid, text, extra = {}) => ev("message.part.updated", { part: { id, messageID: mid, sessionID: sid, type: "text", text, ...extra } });
const idle = (sid) => ev("session.status", { sessionID: sid, status: { type: "idle" } });
await ev("session.created", { info: { id: "s1", title: "t", directory: "/home/u/oc-proj" } });
await msg("u1", "s1", "user"); await txt("p0", "u1", "s1", "user prompt");
await msg("a1", "s1", "assistant"); await txt("p1", "a1", "s1", "oc-step");
await ev("message.part.updated", { part: { id: "t1", messageID: "a1", sessionID: "s1", type: "tool" } });
await msg("a2", "s1", "assistant"); await txt("p2", "a2", "s1", "oc-fi");
await ev("message.part.delta", { sessionID: "s1", messageID: "a2", partID: "p2", field: "text", delta: "nal" });
await txt("p2", "a2", "s1", "oc-final", { time: { start: 1, end: 2 } });
await txt("p3", "a2", "s1", "reminder", { synthetic: true });
await ev("session.status", { sessionID: "s1", status: { type: "busy" } });
await idle("s1"); await ev("session.idle", { sessionID: "s1" });
await ev("session.created", { info: { id: "c1", parentID: "s1" } });
await msg("c1a", "c1", "assistant"); await txt("cp", "c1a", "c1", "oc-child"); await idle("c1");
await msg("a3", "s1", "assistant"); await idle("s1");
await msg("a4", "s1", "assistant"); await txt("p4", "a4", "s1", "oc-aborted");
await msg("a4", "s1", "assistant", { error: { name: "MessageAbortedError", data: {} } }); await idle("s1");
await msg("a5", "s1", "assistant", { summary: true }); await txt("p5", "a5", "s1", "oc-summary"); await idle("s1");
await hooks.event({}); await hooks.event({ event: null });
"""
OPENCODE_EXPECT = ["oc-final"]

DSH = r"""
import { apply, name } from "./adapter.mjs";
if (name !== "moth") throw new Error("name");
const h = {}; apply({ on: (n, f) => { (h[n] ||= []).push(f); } });
const se = (sid, type, data) => h["session/event"].forEach((f) => f({ id: sid }, { seq: 0, time: 0, type, data }));
const am = (turn, text, extra = {}) => ({ turn, step: 0, stream: [], ...extra, message: { id: "m", role: "assistant",
  source: { kind: "model" }, content: [{ type: "reasoning", text: "think" }, { type: "text", text }] } });
se("s1", "turn/start", { turn: 1 }); se("s1", "assistant/message", am(1, "dsh-step"));
se("s1", "tool/call", { turn: 1, step: 0, callId: "c", name: "bash", arguments: "{}" });
se("s1", "assistant/message", am(1, "dsh-final")); se("s1", "turn/end", { turn: 1, reason: { kind: "completed" } });
se("s1", "turn/start", { turn: 2 }); se("s1", "assistant/message", am(2, "dsh-aborted"));
se("s1", "turn/end", { turn: 2, reason: { kind: "aborted", reason: { kind: "user" } } });
se("s1", "turn/start", { turn: 3 }); se("s1", "assistant/message", am(3, "dsh-cut", { interrupted: true }));
se("s1", "turn/end", { turn: 3, reason: { kind: "completed" } });
se("s1", "turn/start", { turn: 4 }); se("s1", "turn/end", { turn: 4, reason: { kind: "completed" } });
h["subagent/start"].forEach((f) => f({ runId: "r", provider: "local", id: "c1", local: true }));
se("c1", "turn/start", { turn: 1 }); se("c1", "assistant/message", am(1, "dsh-child"));
se("c1", "turn/end", { turn: 1, reason: { kind: "completed" } });
se("s1", "turn/start", { turn: 5 }); se("s1", "assistant/message", am(5, "dsh-max"));
se("s1", "turn/end", { turn: 5, reason: { kind: "max-tokens" } });
se("s1", "turn/end", { turn: 6, reason: { kind: "completed" } }); h["session/event"].forEach((f) => f(null, null));
"""
DSH_EXPECT = ["dsh-final", "dsh-max"]


@unittest.skipUnless(shutil.which("node"), "node not installed")
class JsAdapters(Base):
    CASES = {"pi": (HERE / "pi" / "moth.ts", PI, PI_EXPECT),
             "opencode": (HERE / "opencode" / "moth.js", OPENCODE, OPENCODE_EXPECT),
             "dsh": (HERE / "dsh" / "moth.mjs", DSH, DSH_EXPECT)}
    SOURCE = {"pi": "pi", "opencode": "opencode", "dsh": "dsh"}
    META = {"pi": ("pi-sess-1", "pi-proj"), "opencode": ("s1", "oc-proj"), "dsh": ("s1", Path.cwd().name)}

    def drive(self, harness, url):
        src, driver, _ = self.CASES[harness]
        with tempfile.TemporaryDirectory() as d:
            shutil.copy(src, Path(d) / "adapter.mjs")   # .ts / .js are plain ESM JS; .mjs makes node load them as ESM
            (Path(d) / "driver.mjs").write_text(driver, encoding="utf-8")
            return run(["node", str(Path(d) / "driver.mjs")], url)

    def test_posts_only_final_replies(self):
        for show in ("", "1"):   # posting never depends on the show_in_harness setting (default off, on)
            for harness, (_, _, expect) in self.CASES.items():
                with self.subTest(harness, show=show):
                    self.ok.texts.clear()
                    self.ok.sources.clear()
                    self.ok.metas.clear()
                    os.environ["MOTH_SHOW_IN_HARNESS"] = show
                    try:
                        self.quiet(*self.drive(harness, self.ok.url))
                    finally:
                        os.environ.pop("MOTH_SHOW_IN_HARNESS", None)
                    self.assertEqual(self.ok.texts, expect)
                    self.assertEqual(self.ok.sources, [self.SOURCE[harness]] * len(expect))
                    self.assertEqual(self.ok.metas, [self.META[harness]] * len(expect))

    def test_moth_slow_or_down_never_crashes(self):
        for harness, (_, _, expect) in self.CASES.items():
            with self.subTest(harness):
                self.slow.texts.clear()
                self.quiet(*self.drive(harness, self.slow.url))   # fetch aborted at 1 s: no unhandled rejection
                self.assertEqual(self.slow.texts, expect)
                self.quiet(*self.drive(harness, closed_url()))
                self.quiet(*self.drive(harness, "not a url"))


# --- display layer: moth tail, Hermes /moth + desktop pet, Pi widget + /moth, OpenCode toast, serve.py CORS ----------
class LogStub:
    """Fake moth with a real log: /api/agent and /api/chat append events like serve.py; GET /api/log slices it
    (token only), GET /api/says gives moth's lines only (`?source=&session=`: only its reactions to those, like
    serve.py, which tags a reaction with "re": {"source", "session"} of the turn it answers). `epoch` changes when moth
    "restarts"; a poll with an old epoch gets the whole new log. A reaction to an agent turn lands `react` seconds
    after the post (moth thinking); chat answers after `delay`."""

    def __init__(self, delay=0.0, react=0.2):
        self.log, self.paths, self.epoch, stub = [], [], "e1", self

        class H(BaseHTTPRequestHandler):
            def _json(self, obj, code=200):
                b = json.dumps(obj).encode()
                try:
                    self.send_response(code)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(b)))
                    self.end_headers()
                    self.wfile.write(b)
                except OSError:
                    pass

            def do_GET(self):
                from urllib.parse import parse_qs, urlsplit
                stub.paths.append(("GET", self.path))
                url = urlsplit(self.path)
                q = parse_qs(url.query)
                if url.path == "/api/log" and self.headers.get("X-Moth-Token") != TOKEN:
                    return self._json({"error": "token"}, 401)
                n = min(int(q.get("since", ["0"])[0]), len(stub.log))
                if q.get("epoch", [stub.epoch])[0] != stub.epoch:
                    n = 0
                evs = stub.log[n:]
                if url.path == "/api/says":
                    want = {k: q[k][0] for k in ("source", "session") if k in q}
                    evs = [{"say": e["say"]} for e in evs if "say" in e and
                           all((e.get("re") or {}).get(k) == v for k, v in want.items())]
                self._json({"events": evs, "next": len(stub.log), "epoch": stub.epoch, "tree": []})

            def do_POST(self):
                stub.paths.append(("POST", self.path))
                if self.headers.get("X-Moth-Token") != TOKEN:
                    return self._json({"error": "token"}, 401)
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                text = body["text"]
                if self.path == "/api/agent":
                    stub.log.append({"agent": text})
                    re_ = {k: body[k] for k in ("source", "session") if body.get(k)}
                    threading.Timer(react, stub.log.append, [{"say": f"ooh, {text}", "re": re_}]).start()
                    return self._json({"events": []})
                stub.log.append({"you": text})
                time.sleep(delay)
                evs = [{"tool": "bash", "args": {"command": "ls"}, "result": "notes"}, {"say": f"hi back: {text}"}]
                stub.log.extend(evs)
                self._json({"events": evs})

            def log_message(self, *a):
                pass

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.srv.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.srv.server_port}"
        threading.Thread(target=self.srv.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


class Display(unittest.TestCase):
    ON = {"MOTH_SHOW_IN_HARNESS": "1"}

    def setUp(self):
        self.moth = LogStub()
        self.tmp = tempfile.mkdtemp()
        self.cfg = os.path.join(self.tmp, "config.json")   # per test; absent = the default (off)

    def tearDown(self):
        self.moth.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def says_polls(self):
        return [p for m, p in self.moth.paths if m == "GET" and p.startswith("/api/says")]

    def test_tail_watch_prints_new_lines_and_survives_moth_down(self):
        self.moth.log += [{"agent": "old turn"}, {"say": "old line"}]
        p = subprocess.Popen([sys.executable, "-m", "moth.tail", "--watch", "--history", "1", "--url", self.moth.url],
                             cwd=HERE.parents[1], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                             env={**os.environ, "COLUMNS": "40"})
        time.sleep(0.6)
        self.moth.log += [{"agent": "fixed the parser\nand ran the tests"}, {"say": "a parser! i ate a word once. " * 3}]
        time.sleep(1.4)
        p.terminate()
        out, err = p.communicate(timeout=5)
        self.assertEqual(err, "")
        self.assertNotIn("old turn", out)                       # --history 1: only the newest old event
        self.assertIn("moth> old line", out)
        self.assertIn("agent said: fixed the parser and ran", out)
        self.assertIn("moth> a parser!", out)
        self.assertTrue(all(len(ln) <= 39 for ln in out.splitlines()), out)   # wrapped to the pane
        self.assertFalse(any(m == "POST" for m, _ in self.moth.paths))   # display never posts
        down = subprocess.Popen([sys.executable, "-m", "moth.tail", "--watch", "--url", closed_url()],
                                cwd=HERE.parents[1], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        time.sleep(1.0 if sys.platform.startswith("linux") else 3.5)   # Windows: a refused connect takes ~2 s
        down.terminate()
        out, err = down.communicate(timeout=5)
        self.assertEqual((out.count("moth is asleep"), err), (1, ""))   # said once, not once per retry

    def test_tail_chat_echo_and_restart(self):
        sys.path.insert(0, str(HERE.parents[1]))
        from moth.tail import Tail
        t, shown = Tail(self.moth.url, watch=True, history=0), []
        t.show = shown.append
        t.poll_once()
        t.sent.append("hello moth")
        t.chat("hello moth")                       # the reply comes through the poll, our own line is not echoed
        t.poll_once()
        self.assertEqual([s for s in shown if "awake" not in s], ["  $ ls", "moth> hi back: hello moth"])
        self.moth.log[:] = [{"say": "fresh start"}]    # moth restarted: a new epoch, its log starts over
        self.moth.epoch = "e2"
        t.poll_once()
        self.assertEqual(shown[-1], "moth> fresh start")
        self.moth.log += [{"say": "one"}, {"say": "two"}, {"say": "three"}]   # restarted again, already past our cursor
        self.moth.log[:] = self.moth.log[-3:]
        self.moth.epoch = "e3"
        t.poll_once()
        self.assertEqual(shown[-3:], ["moth> one", "moth> two", "moth> three"])
        self.moth.log += [{"agent": "\x1b[31mred\x1b[0m done", "source": "claude-code"},
                          {"agent": "meh", "source": "codex", "skipped": True}, {"say": "\x1b]0;pwned\x07hi"}]
        t.poll_once()
        self.assertEqual(shown[-3:], ["claude said: red done", "codex said (moth let it pass): meh", "moth> hi"])
        t.url = closed_url()
        t.chat("anyone?")
        self.assertIn("didn't hear that", shown[-1])
        import moth.tail as tail_mod
        t.url, good = self.moth.url, tail_mod.TOKEN_FILE
        tail_mod.TOKEN_FILE = "/nonexistent/token"
        try:
            t.chat("no token")
            self.assertIn("no token in", shown[-1])
            with self.assertRaises(PermissionError):
                t.poll_once()
        finally:
            tail_mod.TOKEN_FILE = good

    def test_tail_labels_and_session_filter(self):
        sys.path.insert(0, str(HERE.parents[1]))
        from moth.tail import Tail
        self.moth.log += [{"agent": "fixed login", "source": "claude-code", "session": "a1", "label": "web"},
                          {"agent": "migrated db", "source": "codex", "session": "b2", "label": "api", "skipped": True},
                          {"agent": "two in web", "source": "codex", "session": "c3", "label": "web (c3c3)"},
                          {"agent": "old adapter", "source": "pi"}, {"say": "ooh"}, {"you": "nice"}]
        every = ["claude · web said: fixed login", "codex · api said (moth let it pass): migrated db",
                 "codex · web (c3c3) said: two in web", "pi said: old adapter", "moth> ooh", "you> nice"]
        for only, keep in ((None, every), ("web", [every[0], every[2]]), ("api", [every[1]]), ("b2", [every[1]]),
                           ("pi", [every[3]]), ("claude", [every[0]]), ("nope", [])):
            with self.subTest(session=only):
                t, shown = Tail(self.moth.url, watch=True, history=10, session=only), []
                t.show = shown.append
                t.poll_once()
                rest = [ln for ln in every[4:]]   # moth's lines and yours always show
                self.assertEqual([s for s in shown if "awake" not in s], keep + rest if only else every)
        p = subprocess.Popen([sys.executable, "-m", "moth.tail", "--watch", "--history", "10", "--session", "api",
                              "--url", self.moth.url], cwd=HERE.parents[1], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, env={**os.environ, "COLUMNS": "100"})
        time.sleep(0.8)
        p.terminate()
        out, err = p.communicate(timeout=5)
        self.assertEqual(err, "")
        self.assertIn("codex · api said (moth let it pass): migrated db", out)
        self.assertNotIn("fixed login", out)
        self.assertIn("moth> ooh", out)

    def test_hermes_has_no_moth_command(self):
        """/moth was removed from Hermes: mid-turn, the gateway hands unknown slash commands to the agent."""
        spec = importlib.util.spec_from_file_location("moth_hermes_cmd", HERE / "hermes" / "moth" / "__init__.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        calls = []
        mod.register(type("Ctx", (), {"register_hook": lambda self, n, f: calls.append(("hook", n)),
                                      "register_command": lambda self, n, f, **kw: calls.append(("command", n))})())
        self.assertEqual(calls, [("hook", "post_llm_call")])

    def node(self, driver, src, url, name="adapter.mjs", env=None):
        with tempfile.TemporaryDirectory() as d:
            shutil.copy(src, Path(d) / name)
            (Path(d) / "driver.mjs").write_text(driver, encoding="utf-8")
            p = subprocess.run(["node", str(Path(d) / "driver.mjs")], capture_output=True, text=True, encoding="utf-8",
                               timeout=30,
                               env={**os.environ, "MOTH_URL": url, "MOTH_CONFIG_FILE": self.cfg, **(env or {})})
        self.assertEqual((p.returncode, p.stderr), (0, ""), p.stdout)
        return json.loads(p.stdout.strip().splitlines()[-1])

    @unittest.skipUnless(shutil.which("node"), "node not installed")
    def test_hermes_desktop_pet(self):
        driver = r"""
import plugin from "./plugin.mjs";
const said = [], timers = [];
let tick = null;
const ctx = { pet: { say: (text, o) => { said.push([text, o.id, o.ttlMs]); return () => {}; } },
  setInterval: (fn, ms) => { tick = fn; return () => {}; },
  setTimeout: (fn, ms) => { timers.push([fn, ms]); return () => {}; } };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const post = (path, body) => fetch(`${process.env.MOTH_URL}${path}`, { method: "POST", body: JSON.stringify(body),
  headers: { "X-Moth-Token": "test-token-not-secret" } });
const add = (text, source = "hermes") => post("/api/agent", { text, source, session: "20261018_120000_ab12" });
if (plugin.id !== "moth" || typeof plugin.register !== "function") throw new Error("shape");
plugin.register(ctx); await sleep(200);                 // first poll: cursor at the end, old lines skipped
await add("one"); await sleep(300); tick(); await sleep(200);     // moth's reaction to a Hermes turn
await add("claude's turn", "claude-code"); await post("/api/chat", { text: "hi" });   // not Hermes: never voiced
await sleep(300); tick(); await sleep(200);
await add("x ".repeat(150) + "end"); await sleep(300); tick(); await sleep(200);   // a long line: cut into pieces
timers.forEach(([fn]) => fn());
plugin.register({}); plugin.register({ pet: {} });    // older desktops: no pet, no crash
console.log(JSON.stringify({ said, delays: timers.map(([, ms]) => ms) }));
"""
        self.moth.log += [{"say": "old line"}]
        with tempfile.TemporaryDirectory() as d:   # the renderer has no env: point the plugin's URL at the stub
            src = Path(d) / "plugin.js"
            src.write_text((HERE / "hermes" / "moth" / "desktop" / "plugin.js").read_text(encoding="utf-8")
                           .replace("http://127.0.0.1:4747", self.moth.url), encoding="utf-8")
            r = self.node(driver, src, self.moth.url, "plugin.mjs")
        texts = [s[0] for s in r["said"]]
        self.assertEqual(texts[0], "ooh, one")
        self.assertNotIn("old line", texts)
        self.assertFalse(any("claude" in t or "hi back" in t for t in texts), texts)
        self.assertTrue(all("source=hermes" in p for p in self.says_polls()))
        self.assertTrue(all(s[1] == "moth" and 1000 <= s[2] <= 30000 for s in r["said"]))
        self.assertTrue(all(len(t) <= 120 for t in texts), texts)
        self.assertEqual(len(texts), 1 + 3)            # the long line became 3 pieces, shown one after another
        self.assertEqual(len(r["delays"]), 2)
        self.assertEqual({p.split("?")[0] for m, p in self.moth.paths if m == "GET"}, {"/api/says"})   # no token

    @unittest.skipUnless(shutil.which("node"), "node not installed")
    def test_hermes_desktop_pet_moth_down(self):
        driver = r"""
import plugin from "./plugin.mjs";
let tick = null; const said = [];
plugin.register({ pet: { say: (t) => said.push(t) }, setInterval: (fn) => { tick = fn; }, setTimeout: () => () => {} });
await new Promise((r) => setTimeout(r, 300)); tick(); await new Promise((r) => setTimeout(r, 300));
console.log(JSON.stringify({ said }));
"""
        with tempfile.TemporaryDirectory() as d:
            src = (HERE / "hermes" / "moth" / "desktop" / "plugin.js").read_text(encoding="utf-8")
            (Path(d) / "plugin.mjs").write_text(src.replace("http://127.0.0.1:4747", closed_url()), encoding="utf-8")
            (Path(d) / "driver.mjs").write_text(driver, encoding="utf-8")
            p = subprocess.run(["node", str(Path(d) / "driver.mjs")], capture_output=True, text=True, encoding="utf-8",
                               timeout=10)
        self.assertEqual((p.returncode, p.stderr, json.loads(p.stdout)), (0, "", {"said": []}))

    PI_DRIVER = r"""
import ext from "./adapter.mjs";
const h = {}, cmds = {}, widgets = [], notes = [];
ext({ on: (n, f) => { (h[n] ||= []).push(f); }, registerCommand: (n, o) => { cmds[n] = o; } });
const ctx = { hasUI: true, ui: { setWidget: (k, lines) => widgets.push([k, lines]), notify: (m) => notes.push(m) },
  sessionManager: { getSessionId: () => "pi-s1" }, cwd: "/w/shop" };
const fire = (n, e) => h[n].forEach((f) => f(e, ctx));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
fire("agent_start", {}); await sleep(200);
fire("agent_end", { messages: [{ role: "assistant", stopReason: "stop", content: [{ type: "text", text: "pi done" }] }] });
fire("agent_settled", { aborted: false }); await sleep(1500);
await cmds.moth.handler("hey", ctx); await cmds.moth.handler("", ctx);
const bare = { hasUI: false, ui: { notify: (m) => notes.push(m) } };   // no widget (print mode): the reply is a notice
await cmds.moth.handler("plain", bare);
console.log(JSON.stringify({ widgets, notes, desc: cmds.moth.description }));
"""

    @unittest.skipUnless(shutil.which("node"), "node not installed")
    def test_pi_widget_and_command(self):
        r = self.node(self.PI_DRIVER, HERE / "pi" / "moth.ts", self.moth.url, env=self.ON)
        lines = [w[1][0] for w in r["widgets"]]
        self.assertTrue(all(w[0] == "moth" for w in r["widgets"]))
        self.assertIn("\x1b[33mmoth>\x1b[0m ooh, pi done", lines)
        self.assertEqual(lines[-1], "\x1b[33mmoth>\x1b[0m hi back: hey")
        self.assertIn("usage", r["notes"][0])
        self.assertEqual(r["notes"][1], "moth> hi back: plain")
        self.assertEqual({p.split("?")[0] for m, p in self.moth.paths if m == "GET"}, {"/api/says"})
        r = self.node(self.PI_DRIVER, HERE / "pi" / "moth.ts", closed_url(), env=self.ON)
        self.assertEqual(r["widgets"], [])
        self.assertIn("asleep", r["notes"][0])

    @unittest.skipUnless(shutil.which("node"), "node not installed")
    def test_pi_off_by_default_posts_but_draws_nothing(self):
        """No env, no config file: the turn is still posted (the pet reacts), but nothing is polled or drawn.
        /moth is registered anyway and answers with a one-off notice, never the widget."""
        r = self.node(self.PI_DRIVER, HERE / "pi" / "moth.ts", self.moth.url)
        self.assertIn({"agent": "pi done"}, self.moth.log)   # the stub's card keeps only the text
        self.assertEqual(r["widgets"], [])
        self.assertEqual(r["notes"][0], "moth> hi back: hey")
        self.assertIn("usage", r["notes"][1])
        self.assertEqual(r["notes"][2], "moth> hi back: plain")
        self.assertEqual(self.says_polls(), [])
        Path(self.cfg).write_text('{"show_in_harness": true}')   # the env beats the file, both ways
        r = self.node(self.PI_DRIVER, HERE / "pi" / "moth.ts", self.moth.url, env={"MOTH_SHOW_IN_HARNESS": "0"})
        self.assertEqual((r["widgets"], self.says_polls()), ([], []))
        Path(self.cfg).write_text('{"show_in_harness": "off"}')
        r = self.node(self.PI_DRIVER, HERE / "pi" / "moth.ts", self.moth.url, env={"MOTH_SHOW_IN_HARNESS": "yes"})
        self.assertIn("\x1b[33mmoth>\x1b[0m ooh, pi done", [w[1][0] for w in r["widgets"]])

    @unittest.skipUnless(shutil.which("node"), "node not installed")
    def test_pi_live_toggle_needs_no_restart(self):
        """One Pi process: on, then off (the widget is taken down), then on again. The reaction lands 1 s after the
        post, so a cursor kept from before the off turn would show the off turn's line instead of the new one."""
        self.moth.close()
        self.moth = LogStub(react=1.0)
        driver = r"""
import ext from "./adapter.mjs";
import { writeFileSync } from "node:fs";
const h = {}, widgets = [];
ext({ on: (n, f) => { (h[n] ||= []).push(f); }, registerCommand: () => {} });
const ctx = { hasUI: true, ui: { setWidget: (k, lines) => widgets.push(lines === undefined ? null : lines[0]),
  notify: () => {} }, sessionManager: { getSessionId: () => "pi-s1" } };
const fire = (n, e) => h[n].forEach((f) => f(e, ctx));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const set = (v) => writeFileSync(process.env.MOTH_CONFIG_FILE, JSON.stringify({ other: 1, show_in_harness: v }));
const turn = async (text) => {
  fire("agent_start", {}); await sleep(200);
  fire("agent_end", { messages: [{ role: "assistant", stopReason: "stop", content: [{ type: "text", text }] }] });
  fire("agent_settled", { aborted: false }); await sleep(1900);
};
set("on"); await turn("one"); set(false); await turn("two"); set(true); await turn("three");
console.log(JSON.stringify({ widgets }));
"""
        r = self.node(driver, HERE / "pi" / "moth.ts", self.moth.url)
        y = "\x1b[33mmoth>\x1b[0m "
        self.assertEqual(r["widgets"], [y + "ooh, one", None, y + "ooh, three"])
        self.assertEqual([e["agent"] for e in self.moth.log if "agent" in e], ["one", "two", "three"])

    @unittest.skipUnless(shutil.which("node"), "node not installed")
    def test_pi_widget_shows_only_its_own_session(self):
        """Another agent finishes right after Pi: moth's reply to it lands last, but Pi's widget shows the reply to
        Pi's own turn (live 2026-10-18: Claude Code's band showed moth's reply to Hermes). No session id: nothing."""
        self.moth.close()
        self.moth = LogStub(react=0.6)
        driver = r"""
import ext from "./adapter.mjs";
const h = {}, widgets = [];
ext({ on: (n, f) => { (h[n] ||= []).push(f); }, registerCommand: () => {} });
const mk = (sid) => ({ hasUI: true, ui: { setWidget: (k, l) => widgets.push(l[0]), notify: () => {} },
  sessionManager: { getSessionId: () => sid } });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const other = (text) => fetch(`${process.env.MOTH_URL}/api/agent`, { method: "POST",
  headers: { "X-Moth-Token": "test-token-not-secret" }, body: JSON.stringify({ text, source: "claude-code", session: "cc1" }) });
const turn = async (ctx, text) => {
  h.agent_start.forEach((f) => f({}, ctx)); await sleep(200);
  h.agent_end.forEach((f) => f({ messages: [{ role: "assistant", stopReason: "stop", content: [{ type: "text", text }] }] }, ctx));
  h.agent_settled.forEach((f) => f({ aborted: false }, ctx));
  await sleep(100); await other("claude " + text); await sleep(1800);
};
await turn(mk("pi-s1"), "mine");
await turn(mk(""), "no id");   // an older Pi without a session id: no way to tell, so no widget
console.log(JSON.stringify({ widgets }));
"""
        r = self.node(driver, HERE / "pi" / "moth.ts", self.moth.url, env=self.ON)
        self.assertEqual(r["widgets"], ["\x1b[33mmoth>\x1b[0m ooh, mine"])
        self.assertTrue(any("source=pi&session=pi-s1" in p for p in self.says_polls()), self.says_polls())
        self.assertEqual(len([e for e in self.moth.log if "say" in e]), 4)   # the other reactions did happen

    @unittest.skipUnless(shutil.which("node"), "node not installed")
    def test_setting_spellings_match_python(self):
        """Pi's enabled() and moth/config.py read the same env / file values the same way."""
        sys.path.insert(0, str(HERE.parents[1]))
        from moth import config
        cases = [("", None), ("", '{"show_in_harness": true}'), ("", '{"show_in_harness": "On"}'),
                 ("", '{"show_in_harness": "nah"}'), ("", '{"show_in_harness": 1}'), ("", "not json"), ("", "[]"),
                 ("0", '{"show_in_harness": true}'), ("yes", '{"show_in_harness": false}'), ("ON", None),
                 ("off", None), ("maybe", '{"show_in_harness": true}'), (" true ", None), ("no", None)]
        driver = r"""
import { enabled } from "./adapter.mjs";
import { writeFileSync, rmSync } from "node:fs";
const out = [];
for (const [env, file] of JSON.parse(process.env.CASES)) {
  process.env.MOTH_SHOW_IN_HARNESS = env;
  if (file === null) rmSync(process.env.MOTH_CONFIG_FILE, { force: true });
  else writeFileSync(process.env.MOTH_CONFIG_FILE, file);
  out.push(enabled());
}
console.log(JSON.stringify(out));
"""
        js = self.node(driver, HERE / "pi" / "moth.ts", self.moth.url, env={"CASES": json.dumps(cases)})
        py = []
        old = os.environ.get("MOTH_CONFIG_FILE")
        try:
            os.environ["MOTH_CONFIG_FILE"] = self.cfg
            for env, file in cases:
                os.environ["MOTH_SHOW_IN_HARNESS"] = env
                if file is None:
                    Path(self.cfg).unlink(missing_ok=True)
                else:
                    Path(self.cfg).write_text(file)
                py.append(config.show_in_harness())
        finally:
            os.environ.pop("MOTH_SHOW_IN_HARNESS", None)
            os.environ["MOTH_CONFIG_FILE"] = old
        self.assertEqual(js, py)
        self.assertEqual(py, [False, True, True, False, False, False, False, False, True, True, False, True, True,
                              False])

    OC_DRIVER = r"""
import { Moth } from "./adapter.mjs";
const toasts = [];
const client = { tui: { showToast: async (o) => { toasts.push(o.body); return {}; } } };
const hooks = await Moth({ client, project: {}, directory: "/x", worktree: "/x", $: null });
const ev = (type, properties) => hooks.event({ event: { type, properties } });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
await ev("session.status", { sessionID: "s1", status: { type: "busy" } }); await sleep(200);
await ev("message.updated", { info: { id: "a1", sessionID: "s1", role: "assistant" } });
await ev("message.part.updated", { part: { id: "p1", messageID: "a1", sessionID: "s1", type: "text", text: "oc done" } });
const t = Date.now(); await ev("session.status", { sessionID: "s1", status: { type: "idle" } });
const waited = Date.now() - t; await sleep(1500);
console.log(JSON.stringify({ toasts, waited }));
"""

    @unittest.skipUnless(shutil.which("node"), "node not installed")
    def test_opencode_toast(self):
        r = self.node(self.OC_DRIVER, HERE / "opencode" / "moth.js", self.moth.url, env=self.ON)
        self.assertEqual(r["toasts"], [{"title": "moth", "message": "ooh, oc done", "variant": "info", "duration": 8000}])
        self.assertLess(r["waited"], 50)             # the event hook never waits on the poll
        self.assertEqual(self.node(self.OC_DRIVER, HERE / "opencode" / "moth.js", closed_url(), env=self.ON)["toasts"],
                         [])

    @unittest.skipUnless(shutil.which("node"), "node not installed")
    def test_opencode_off_by_default_posts_but_no_toast(self):
        r = self.node(self.OC_DRIVER, HERE / "opencode" / "moth.js", self.moth.url)
        self.assertIn({"agent": "oc done"}, self.moth.log)
        self.assertEqual((r["toasts"], self.says_polls()), ([], []))
        Path(self.cfg).write_text('{"show_in_harness": "on"}')   # the file alone turns it on
        r = self.node(self.OC_DRIVER, HERE / "opencode" / "moth.js", self.moth.url)
        self.assertEqual([t["message"] for t in r["toasts"]], ["ooh, oc done"])
        r = self.node(self.OC_DRIVER, HERE / "opencode" / "moth.js", self.moth.url, env={"MOTH_SHOW_IN_HARNESS": "0"})
        self.assertEqual(r["toasts"], [])                           # ...and the env beats it

    @unittest.skipUnless(shutil.which("node"), "node not installed")
    def test_opencode_live_toggle_needs_no_restart(self):
        self.moth.close()
        self.moth = LogStub(react=1.0)
        driver = r"""
import { Moth } from "./adapter.mjs";
import { writeFileSync } from "node:fs";
const toasts = [];
const client = { tui: { showToast: async (o) => { toasts.push(o.body.message); return {}; } } };
const hooks = await Moth({ client, project: {}, directory: "/x", worktree: "/x", $: null });
const ev = (type, properties) => hooks.event({ event: { type, properties } });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const set = (v) => writeFileSync(process.env.MOTH_CONFIG_FILE, JSON.stringify({ show_in_harness: v }));
let n = 0;
const turn = async (text) => {
  n++;
  await ev("session.status", { sessionID: "s1", status: { type: "busy" } }); await sleep(200);
  await ev("message.updated", { info: { id: "a" + n, sessionID: "s1", role: "assistant" } });
  await ev("message.part.updated", { part: { id: "p" + n, messageID: "a" + n, sessionID: "s1", type: "text", text } });
  await ev("session.status", { sessionID: "s1", status: { type: "idle" } }); await sleep(1900);
};
set(true); await turn("one"); set("0"); await turn("two"); set("1"); await turn("three");
console.log(JSON.stringify({ toasts }));
"""
        r = self.node(driver, HERE / "opencode" / "moth.js", self.moth.url)
        self.assertEqual(r["toasts"], ["ooh, one", "ooh, three"])
        self.assertEqual([e["agent"] for e in self.moth.log if "agent" in e], ["one", "two", "three"])


    @unittest.skipUnless(shutil.which("node"), "node not installed")
    def test_opencode_two_sessions_each_get_their_own_toast(self):
        """Two OpenCode sessions finish 0.1 s apart and another harness's turn lands too: each session's poll asks
        for its own reactions only and keeps its own cursor, so both toasts show, once each, and nothing else."""
        self.moth.close()
        self.moth = LogStub(react=0.6)
        driver = r"""
import { Moth } from "./adapter.mjs";
const toasts = [];
const client = { tui: { showToast: async (o) => { toasts.push(o.body.message); return {}; } } };
const hooks = await Moth({ client, project: {}, directory: "/x", worktree: "/x", $: null });
const ev = (type, properties) => hooks.event({ event: { type, properties } });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
for (const s of ["s1", "s2"]) await ev("session.status", { sessionID: s, status: { type: "busy" } });
await sleep(200);
for (const [s, text] of [["s1", "one"], ["s2", "two"]]) {
  await ev("message.updated", { info: { id: "a" + s, sessionID: s, role: "assistant" } });
  await ev("message.part.updated", { part: { id: "p" + s, messageID: "a" + s, sessionID: s, type: "text", text } });
}
await ev("session.status", { sessionID: "s1", status: { type: "idle" } }); await sleep(100);
await ev("session.status", { sessionID: "s2", status: { type: "idle" } });
await fetch(`${process.env.MOTH_URL}/api/agent`, { method: "POST", headers: { "X-Moth-Token": "test-token-not-secret" },
  body: JSON.stringify({ text: "hermes", source: "hermes", session: "h1" }) });
await sleep(2000);
console.log(JSON.stringify({ toasts }));
"""
        r = self.node(driver, HERE / "opencode" / "moth.js", self.moth.url, env=self.ON)
        self.assertEqual(sorted(r["toasts"]), ["ooh, one", "ooh, two"])

def load_serve():
    """The real serve.py, with torch and the model code faked out (no model is loaded)."""
    import types
    from unittest import mock
    fakes = {n: types.ModuleType(n) for n in ("torch", "tokenizers", "tiny_agent.checkpoint", "tiny_agent.generate")}
    fakes["torch"].no_grad = lambda: (lambda f: f)
    fakes["tokenizers"].Tokenizer = object
    fakes["tiny_agent.checkpoint"].load_model = None
    fakes["tiny_agent.generate"].KVCache = fakes["tiny_agent.generate"].append = None
    with mock.patch.dict(sys.modules, fakes):
        spec = importlib.util.spec_from_file_location("moth_serve_http", HERE.parent / "serve.py")
        serve = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(serve)
    return serve


class Serve(unittest.TestCase):
    """serve.py's real HTTP handler in front of a fake engine."""

    @classmethod
    def setUpClass(cls):
        cls.serve = load_serve()

    def setUp(self):
        import types
        serve, self.tmp = self.serve, tempfile.mkdtemp()
        moth = types.SimpleNamespace(log=serve.Log(), lock=threading.RLock(), heard=[], chats=[],
                                     home=serve.Home(os.path.join(self.tmp, "home"), 1, None),
                                     a=types.SimpleNamespace(home=os.path.join(self.tmp, "home"), pool_file=None))
        moth.log.add({"you": "my boss is awful"}, {"agent": "Fixed the bug."}, {"say": "hi"},
                     {"tool": "bash", "args": {"command": "ls"}, "result": "diary"})
        moth.hear_agent = lambda text, source, session=None, label=None: (
            moth.heard.append((text, source, session, label)) or "queued")
        moth.agents = lambda: {"active": 1, "skipped": 0,
                               "sessions": [{"source": "codex", "session": "s9", "label": "secret-proj"}]}
        moth.say_to = lambda text: moth.chats.append(text) or [{"say": "ok"}]
        moth.forget = lambda: moth.log.clear()
        self.moth = moth
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), serve.make_handler(moth, TOKEN))
        self.srv.daemon_threads = True
        self.port = self.srv.server_port
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def req(self, method, path, headers=None, body=None):
        """-> (status, lower-cased headers, body). A header set to None is left out (Host included)."""
        import http.client
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        c.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
        h = {"Host": f"127.0.0.1:{self.port}"}
        if body is not None:
            h["Content-Length"] = str(len(body))
        h.update(headers or {})
        for k, v in h.items():
            if v is not None:
                c.putheader(k, v)
        c.endheaders(body)
        r = c.getresponse()
        data = r.read()
        c.close()
        return r.status, {k.lower(): v for k, v in r.getheaders()}, data

    def post(self, path, obj=None, raw=None, **headers):
        h = {"Content-Type": "application/json", "X-Moth-Token": TOKEN, **headers}
        return self.req("POST", path, h, raw if raw is not None else json.dumps(obj or {}).encode())

    def test_host_allow_list(self):
        for host in ("evil.example:%d", "127.0.0.1", "localhost", "127.0.0.1:1", "localhost.evil.example:%d",
                     "192.168.1.5:%d", None):
            h = host % self.port if host and "%d" in host else host
            with self.subTest(host=h):
                for method, path in (("GET", "/"), ("GET", "/api/says"), ("OPTIONS", "/api/chat")):
                    self.assertEqual(self.req(method, path, {"Host": h})[0], 403)
                self.assertEqual(self.post("/api/chat", {"text": "hi"}, Host=h)[0], 403)
        for h in (f"127.0.0.1:{self.port}", f"localhost:{self.port}", f"LOCALHOST:{self.port}"):
            self.assertEqual(self.req("GET", "/api/says", {"Host": h})[0], 200)
        self.assertEqual(self.moth.chats, [])

    def test_page_carries_token_and_no_cors(self):
        code, h, body = self.req("GET", "/", {"Origin": "http://localhost:5173"})
        self.assertEqual(code, 200)
        self.assertIn(f'content="{TOKEN}"'.encode(), body)
        self.assertNotIn(b"__MOTH_TOKEN__", body)
        self.assertEqual((h["x-frame-options"], h["cache-control"]), ("DENY", "no-store"))
        self.assertFalse([k for k in h if k.startswith("access-control")])

    def test_full_log_needs_token(self):
        for tok in (None, "", "wrong", TOKEN + "x"):
            with self.subTest(token=tok):
                self.assertEqual(self.req("GET", "/api/log?since=0", {"X-Moth-Token": tok})[0], 401)
                self.assertEqual(self.req("GET", "/api/home", {"X-Moth-Token": tok})[0], 401)
        code, h, body = self.req("GET", "/api/log?since=0", {"X-Moth-Token": TOKEN})
        d = json.loads(body)
        self.assertEqual((code, len(d["events"]), d["next"], d["epoch"]), (200, 4, 4, self.moth.log.epoch))
        self.assertEqual(h["x-frame-options"], "DENY")
        self.assertEqual(json.loads(self.req("GET", "/api/log?since=3", {"X-Moth-Token": TOKEN})[2])["events"][0]["tool"],
                         "bash")

    def test_says_view(self):
        code, h, body = self.req("GET", "/api/says?since=0", {"Origin": "null"})
        d = json.loads(body)
        self.assertEqual((code, d["events"], d["next"]), (200, [{"say": "hi"}], 4))   # no vent, no agent reply
        self.assertNotIn(b"boss", body)
        self.assertEqual(json.loads(self.req("GET", "/api/says?since=1000000000")[2])["events"], [])
        for bad in ("abc", "-1", "1e9", "1.5", "99999999999999999999"):
            with self.subTest(since=bad):
                self.assertEqual(self.req("GET", f"/api/says?since={bad}")[0], 400)
                self.assertEqual(self.req("GET", f"/api/log?since={bad}", {"X-Moth-Token": TOKEN})[0], 400)

    def test_says_per_session(self):
        """?source=&session=: only moth's reactions to that harness / session (tagged "re" by Moth.react), never chat
        or another session's line, and never the tag itself; no params: every line, as before (pet, page)."""
        self.moth.log.add({"agent": "a", "source": "claude-code", "session": "cc-1", "label": "secret-proj"},
                          {"agent": "b", "source": "hermes", "session": "h-1"},
                          {"say": "to hermes", "re": {"source": "hermes", "session": "h-1"}},
                          {"say": "to claude", "re": {"source": "claude-code", "session": "cc-1"}},
                          {"you": "hey"}, {"say": "chat reply"})
        says = lambda q: json.loads(self.req("GET", f"/api/says?since=4&{q}")[2])   # noqa: E731
        for q, want in (("", ["to hermes", "to claude", "chat reply"]),
                        ("source=claude-code&session=cc-1", ["to claude"]), ("session=cc-1", ["to claude"]),
                        ("source=Hermes", ["to hermes"]), ("source=pi", []), ("session=cc-2", []),
                        ("source=hermes&session=cc-1", []), ("session=%21%21", [])):   # cleans to nothing: no match
            with self.subTest(q=q):
                d = says(q)
                self.assertEqual(d["events"], [{"say": t} for t in want])   # stripped to {"say"}: no ids, no labels
                self.assertEqual(d["next"], 10)                               # the cursor still counts every event
        self.assertNotIn(b"cc-1", self.req("GET", "/api/says?since=0&session=cc-1")[2])
        self.assertNotIn(b"secret", self.req("GET", "/api/says?since=0&source=claude-code")[2])
        log = json.loads(self.req("GET", "/api/log?since=6&session=cc-1", {"X-Moth-Token": TOKEN})[2])
        self.assertEqual(len(log["events"]), 4)   # the full log ignores the filter

    def test_cors(self):
        local = ["http://localhost:5173", "https://localhost", "http://127.0.0.1:5174"]
        pets = ["app://hermes", "app://-", "file://", "null"]
        foreign = ["https://evil.example", "http://localhost.evil.example", "http://127.0.0.1.evil.example:80",
                   "http://192.168.1.5:4747", "https://localhost@evil.example", "app://x/../y", "NULL"]

        def acao(path, origin, **h):
            return self.req("GET", path, {"Origin": origin, **h})[1].get("access-control-allow-origin")

        for o in local:
            with self.subTest(origin=o):
                self.assertEqual(acao("/api/log?since=0", o, **{"X-Moth-Token": TOKEN}), o)
                self.assertEqual(acao("/api/says", o), o)
                code, h, _ = self.req("OPTIONS", "/api/chat", {"Origin": o, "Access-Control-Request-Method": "POST"})
                self.assertEqual((code, h["access-control-allow-origin"], h["vary"]), (204, o, "Origin"))
                self.assertEqual(h["access-control-allow-headers"], "Content-Type, X-Moth-Token")
        for o in pets:   # desktop shells: moth's lines only
            with self.subTest(origin=o):
                self.assertEqual(acao("/api/says", o), o)
                self.assertIsNone(acao("/api/log?since=0", o, **{"X-Moth-Token": TOKEN}))
                self.assertFalse([k for k in self.req("OPTIONS", "/api/chat", {"Origin": o})[1]
                                  if k.startswith("access-control")])
        for o in foreign:
            with self.subTest(origin=o):
                self.assertIsNone(acao("/api/says", o))
                self.assertIsNone(acao("/api/log?since=0", o, **{"X-Moth-Token": TOKEN}))

    def test_post_hygiene(self):
        P = "/api/chat"
        self.assertEqual(self.post(P, {"text": "hi"}, **{"X-Moth-Token": None})[0], 401)
        self.assertEqual(self.post(P, {"text": "hi"}, **{"X-Moth-Token": "nope"})[0], 401)
        for o in ("https://evil.example", "null", "file://", "app://hermes"):
            self.assertEqual(self.post(P, {"text": "hi"}, Origin=o)[0], 403)
        self.assertEqual(self.post(P, {"text": "hi"}, **{"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.post(P, {"text": "hi"}, **{"Content-Type": None})[0], 415)
        self.assertEqual(self.post(P, raw=b"x" * (64 * 1024 + 1))[0], 413)
        self.assertEqual(self.req("POST", P, {"Content-Type": "application/json", "X-Moth-Token": TOKEN})[0], 411)
        for raw in (b"{nope", b"[1, 2]", b'"text"', b"\xff\xfe", b'{"text": 5}', b'{"text": "   "}', b"null"):
            with self.subTest(raw=raw):
                self.assertEqual(self.post(P, raw=raw)[0], 400)
        self.assertEqual(self.post("/api/nope", {"text": "hi"})[0], 404)
        self.assertEqual(self.moth.chats, [])
        code, h, body = self.post(P, {"text": " hi moth "}, Origin=f"http://127.0.0.1:{self.port}",
                                  **{"Content-Type": "application/json; charset=utf-8"})
        self.assertEqual((code, json.loads(body)["events"], self.moth.chats), (200, [{"say": "ok"}], ["hi moth"]))
        self.assertEqual(h["x-frame-options"], "DENY")

    def test_agent_source(self):
        for src, want in (("claude-code", "claude-code"), ("Codex", "codex"), ("evil <b>", None), (5, None), (None, None)):
            code, _, body = self.post("/api/agent", {"text": "Done.", "source": src})
            self.assertEqual((code, json.loads(body)["status"]), (200, "queued"))
            self.assertEqual(self.moth.heard[-1][:2], ("Done.", want))
        self.assertEqual(self.post("/api/agent", {"text": ""})[0], 400)

    def test_agent_session_label(self):
        for session, label, want in (("0f3c9a2e-11", "moth-repo", ("0f3c9a2e-11", "moth-repo")),
                                     ("ses_ab:12.x", "\x1b[31mmy\n  proj\x1b[0m\x07", ("ses_ab:12.x", "my proj")),
                                     ("../evil id<b>" + "x" * 99, "L" * 99, ("..evilidbx" + "x" * 70, "L" * 40)),
                                     (5, ["x"], (None, None)), ("", "   ", (None, None)), (None, None, (None, None))):
            with self.subTest(session=session, label=label):
                code = self.post("/api/agent", {"text": "Done.", "source": "codex", "session": session,
                                                "label": label})[0]
                self.assertEqual((code, self.moth.heard[-1]), (200, ("Done.", "codex") + want))
        log = json.loads(self.req("GET", "/api/log?since=0", {"X-Moth-Token": TOKEN})[2])
        self.assertEqual(log["agents"]["sessions"][0]["label"], "secret-proj")
        says = self.req("GET", "/api/says?since=0")[2]
        self.assertNotIn(b"secret-proj", says)                  # folder names only behind the token
        self.assertNotIn(b"agents", says)

    def test_reset_only_deletes_moths_own_home(self):
        home = self.moth.home.root
        e0 = self.moth.log.epoch
        code, _, body = self.post("/api/reset", {})
        self.assertEqual((code, json.loads(body)["epoch"] != e0, len(self.moth.log)), (200, True, 0))
        with open(os.path.join(home, "diary", "mine.txt"), "w") as f:
            f.write("keep me")
        self.assertEqual(self.post("/api/reset", {"new_home": True})[0], 200)   # moth made it: replaced
        self.assertFalse(os.path.exists(os.path.join(home, "diary", "mine.txt")))
        self.assertTrue(self.moth.home.owned)
        os.remove(os.path.join(home, self.serve.MARKER))   # e.g. an older home, or --home pointed at your own folder
        with open(os.path.join(home, "precious.txt"), "w") as f:
            f.write("mine")
        code, _, body = self.post("/api/reset", {"new_home": True})
        self.assertEqual(code, 409)
        self.assertIn("won't delete", json.loads(body)["error"])
        self.assertTrue(os.path.exists(os.path.join(home, "precious.txt")))
        self.assertEqual(self.post("/api/reset", {"new_home": True}, **{"X-Moth-Token": None})[0], 401)


class AdapterToLog(unittest.TestCase):
    """Labels end to end: every adapter, fed its harness's payload, against serve.py's real handler and real
    Moth.hear_agent (a Moth without weights or tokenizer: only the agent-message half, react_every 0, no window)."""

    @classmethod
    def setUpClass(cls):
        cls.serve = load_serve()

    def setUp(self):
        serve, self.tmp = self.serve, tempfile.mkdtemp()
        m = serve.Moth.__new__(serve.Moth)
        m.log, m.lock, m.wake, m.clock = serve.Log(), threading.RLock(), threading.Condition(threading.RLock()), time.monotonic
        m.sessions, m.pending, m.batch, m.batch_until = {}, None, None, 0.0
        m.react_every, m.batch_window, m.last_react = 0, 0, float("-inf")
        m.home = serve.Home(os.path.join(self.tmp, "home"), 1, None)
        self.m = m
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), serve.make_handler(m, TOKEN))
        self.srv.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.srv.server_port}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def cards(self, n):
        for _ in range(200):
            cards = [e for e in self.m.log.since(0)["events"] if "agent" in e]
            if len(cards) >= n:
                return cards
            time.sleep(0.01)
        return cards

    def test_labels_flow_to_the_log(self):
        want = []
        run([sys.executable, str(HERE / "claude_code.py")], self.url,
            json.dumps({"session_id": "cc-1", "cwd": "/home/u/moth-repo", "last_assistant_message": "claude done"}))
        want.append(("claude done", "claude-code", "cc-1", "moth-repo"))
        run([sys.executable, str(HERE / "codex.py")], self.url,
            json.dumps({"hook_event_name": "Stop", "session_id": "cx-1", "cwd": "/srv/api", "last_assistant_message": "codex done"}))
        want.append(("codex done", "codex", "cx-1", "api"))
        spec = importlib.util.spec_from_file_location("moth_hermes_e2e", HERE / "hermes" / "moth" / "__init__.py")
        hermes = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hermes)
        hermes.URL, hermes._cwd = self.url + "/api/agent", lambda: "/home/u/notes"
        hermes.on_post_llm_call(assistant_response="hermes done", platform="cli", session_id="hm-1")
        want.append(("hermes done", "hermes", "hm-1", "notes"))
        self.cards(3)
        if shutil.which("node"):
            for harness, (src, driver, expect) in JsAdapters.CASES.items():
                with tempfile.TemporaryDirectory() as d:
                    shutil.copy(src, Path(d) / "adapter.mjs")
                    (Path(d) / "driver.mjs").write_text(driver, encoding="utf-8")
                    run(["node", str(Path(d) / "driver.mjs")], self.url)
                want += [(t, JsAdapters.SOURCE[harness]) + JsAdapters.META[harness] for t in expect]
                self.cards(len(want))
        got = [(c["agent"], c.get("source"), c.get("session"), c.get("label")) for c in self.cards(len(want))]
        self.assertEqual(sorted(got), sorted(want))
        self.assertEqual(self.m.agents()["active"], len({w[1:3] for w in want}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
