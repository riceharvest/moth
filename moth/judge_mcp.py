"""moth as MCP tools for an interviewer agent (Hermes + Qwen3.8 in moth/interview.py). Stdio MCP, stdlib only.

    python3 -I moth/judge_mcp.py --moth http://127.0.0.1:4801 [--moth http://127.0.0.1:4802] --token-file F --state DIR
        [--home DIR [--home DIR]] [--msgs agent_msgs.json] [--min-turns 10] [--max-turns 25]

One --moth: tools say_to_moth(text), agent_message(msg_id), moth_home(), new_chat(), finish(scores, critique).
Two --moth (A/B): every moth tool takes which="moth 1"|"moth 2" (the runner decides, blind, which server is which and
records it on disk; this server only knows the labels), and finish takes scores for both plus a winner.

The moth servers must run with --react-every 0 --quiet-after 0, so a coding-agent message gets its reaction at once;
agent_message polls /api/log until moth's reaction shows up after the agent card.

Everything goes to DIR: events.jsonl (each tool call: args, result, timing), verdict.json (finish), and
transcript.jsonl (who said what, per moth label). The rules are enforced here, not only in the prompt: finish is refused
before --min-turns exchanges with each moth, and after --max-turns the tools ask the interviewer to wrap up.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

PROTOCOL = "2025-06-18"
KEYS = ("cute", "fun", "back_and_forth", "coherence", "memory", "agent_reactions", "groundedness")
NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))
REACT_WAIT_S = 40.0
CTX_WRAP, CTX_STOP = 10500, 12000   # interviewer prompt tokens: ask to finish / refuse more chat (vLLM window 16384)


class MothClient:
    def __init__(self, url: str, token: str):
        self.url, self.token = url.rstrip("/"), token

    def _req(self, path: str, body: dict | None = None, timeout: float = 120.0) -> dict:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.url + path, data=data, method="GET" if body is None else "POST")
        req.add_header("X-Moth-Token", self.token)
        if body is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with NO_PROXY.open(req, timeout=timeout) as r:
                return json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            return {"http_error": e.code, "body": e.read().decode(errors="replace")[:300]}

    def chat(self, text: str) -> dict:
        return self._req("/api/chat", {"text": text})

    def log(self, since: int = 0) -> dict:
        return self._req(f"/api/log?since={since}")

    def home(self) -> dict:
        return self._req("/api/home")

    def agent(self, text: str) -> dict:
        return self._req("/api/agent", {"text": text, "source": "claude-code"})

    def reset(self) -> dict:
        return self._req("/api/reset", {})


def fmt_events(events: list[dict]) -> tuple[str, list[str], list[dict]]:
    """moth's turn as text for the interviewer: its lines first-class, tool use in brackets, slips flagged."""
    out, says, tools = [], [], []
    for e in events:
        if "say" in e:
            says.append(e["say"])
            out.append(f"moth: {e['say']}")
        elif "tool" in e:
            a = e.get("args") or {}
            arg = a.get("command") or a.get("path") or json.dumps(a)[:80]
            res = str(e.get("result", ""))
            slip = e["tool"] not in ("bash", "read", "write") or res.startswith(("Error", "error", "[tool error")) \
                or "No such file" in res or "can only use" in res or "[exit code" in res
            tools.append({"tool": e["tool"], "arg": arg, "result": res[:400], "slip": slip})
            out.append(f"  [moth used {e['tool']}: {arg}]{' (tool slip)' if slip else ''} -> {res[:160]!r}")
    return "\n".join(out) or "moth: ...", says, tools


class Judge:
    def __init__(self, a):
        token = open(a.token_file).read().strip()
        self.ab = len(a.moth) == 2
        self.labels = ["moth 1", "moth 2"] if self.ab else ["moth"]
        self.moths = {lab: MothClient(u, token) for lab, u in zip(self.labels, a.moth)}
        self.turns = {lab: 0 for lab in self.labels}
        self.homes = dict(zip(self.labels, a.home or []))
        self.state, self.min_turns, self.max_turns = a.state, a.min_turns, a.max_turns
        os.makedirs(self.state, exist_ok=True)
        self.msgs = {}
        if a.msgs:
            self.msgs = {str(m["id"]): m for m in json.load(open(a.msgs))}
        self.used_msgs = {lab: set() for lab in self.labels}
        self.done = False
        self.t0 = time.time()

    # --- bookkeeping ------------------------------------------------------------------------------------------
    def _append(self, name: str, obj: dict):
        with open(os.path.join(self.state, name), "a") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def _which(self, args: dict) -> str | None:
        if not self.ab:
            return "moth"
        w = str(args.get("which", "")).strip().lower().replace("_", " ")
        w = {"1": "moth 1", "2": "moth 2", "moth1": "moth 1", "moth2": "moth 2"}.get(w, w)
        return w if w in self.moths else None

    def ctx(self) -> int:
        """The interviewer's last prompt size in tokens (written by interview.py's relay); 0 if unknown."""
        try:
            return int(open(os.path.join(self.state, "ctx_tokens")).read().strip() or 0)
        except (OSError, ValueError):
            return 0

    def _nudge(self, lab: str) -> str:
        n = self.turns[lab]
        if self.ctx() >= CTX_WRAP:
            return "\n(note: your context is nearly full. stop chatting and call finish now.)"
        if n >= self.max_turns:
            return f"\n(note: {n} exchanges with {lab} — that's plenty. wrap up{' this moth' if self.ab else ''} " \
                   f"and call finish when ready.)"
        return ""

    # --- tools --------------------------------------------------------------------------------------------------
    def tools(self) -> list[dict]:
        which = {"which": {"type": "string", "enum": self.labels,
                           "description": "which moth to talk to"}} if self.ab else {}
        req = ["which"] if self.ab else []
        one = {"type": "object", "properties": {k: {"type": ["integer", "null"], "minimum": 1, "maximum": 5}
                                                 for k in KEYS}}
        scores = ({"type": "object", "description": 'scores per moth: {"moth 1": {...}, "moth 2": {...}}, each with '
                   f'{", ".join(KEYS)} as integers 1-5 or null when the mission did not test it',
                   "properties": {lab: one for lab in self.labels}} if self.ab else
                  {**one, "description": f"{', '.join(KEYS)}: integers 1-5, or null when the mission did not test it"})
        fin_props = {"scores": scores, "critique": {"type": "string", "description":
                     "3-6 sentences with short quoted evidence from moth's actual replies, incl. its cutest and least cute line"}}
        fin_req = ["scores", "critique"]
        if self.ab:
            fin_props["winner"] = {"type": "string", "enum": ["moth 1", "moth 2", "tie"]}
            fin_props["reasons"] = {"type": "string", "description": "why that moth is the better companion overall (all scores weighed together), with quotes from both"}
            fin_req += ["winner", "reasons"]
        tools = [
            {"name": "say_to_moth", "description": "Send one chat message to moth as the human friend; returns moth's "
             "reply (and any file tools moth used in its home).",
             "inputSchema": {"type": "object", "properties": {**which, "text": {"type": "string"}},
                             "required": req + ["text"]}},
            {"name": "agent_message", "description": "Make your coding agent post one of its final messages (by id from "
             "the mission's list) where moth can see it; returns moth's reaction.",
             "inputSchema": {"type": "object", "properties": {**which, "msg_id": {"type": "integer"}},
                             "required": req + ["msg_id"]}},
            {"name": "moth_home", "description": "Look at moth's home folder yourself (moth doesn't see this): without "
             "path, every folder and file name; with path (e.g. \"friends/spider.txt\"), that file's contents.",
             "inputSchema": {"type": "object", "properties": {**which, "path": {"type": "string"}}, "required": req}},
            {"name": "new_chat", "description": "Start a new conversation with moth (it forgets the chat; its files stay).",
             "inputSchema": {"type": "object", "properties": {**which}, "required": req}},
            {"name": "finish", "description": "Record your verdict. Call exactly once, at the end.",
             "inputSchema": {"type": "object", "properties": fin_props, "required": fin_req}},
        ]
        return [t for t in tools if self.msgs or t["name"] != "agent_message"]   # only missions with agent messages

    def call(self, name: str, args: dict) -> tuple[str, bool]:
        t = time.time()
        try:
            text, err = self._call(name, args or {})
        except Exception as e:  # noqa: BLE001  a broken moth call must reach the log and the interviewer
            text, err = f"tool problem: {type(e).__name__}: {e}", True
        self._append("events.jsonl", {"t": round(t - self.t0, 2), "dt": round(time.time() - t, 2), "tool": name,
                                      "args": args, "result": text, "is_error": err})
        return text, err

    def _call(self, name: str, args: dict) -> tuple[str, bool]:
        if self.done and name != "finish":
            return "the verdict is already recorded. stop now and reply with one short line.", False
        if name == "finish":
            return self._finish(args)
        lab = self._which(args)
        if lab is None:
            return f"which must be one of {self.labels}", True
        m = self.moths[lab]
        if name == "say_to_moth":
            text = str(args.get("text", "")).strip()
            if not text:
                return "text is empty", True
            if self.turns[lab] >= self.max_turns + 2 or self.ctx() >= CTX_STOP:
                return f"no more turns with {lab}: call finish now.", True
            r = m.chat(text[:500])
            if "http_error" in r:
                return f"moth server said {r['http_error']}: {r['body']}", True
            shown, says, tools = fmt_events(r.get("events", []))
            self.turns[lab] += 1
            self._append("transcript.jsonl", {"moth": lab, "n": self.turns[lab], "kind": "user", "text": text,
                                              "says": says, "tools": tools})
            return shown + self._nudge(lab), False
        if name == "agent_message":
            return self._agent(lab, m, args)
        if name == "moth_home":
            path = str(args.get("path") or "").strip().strip("/")
            if path:   # one file, read on the host (moth doesn't see this)
                files = dict(self._home_files(lab))
                if path not in files:
                    return f"no file {path!r} in moth's home. files: {', '.join(sorted(files)) or '(none)'}", False
                self._append("transcript.jsonl", {"moth": lab, "n": self.turns[lab], "kind": "home_check", "path": path})
                return f"--- {path}\n{files[path][:600]}", False
            tree = m.home().get("tree", [])
            lines = [f"{d['dir']}/: {', '.join(d['files']) or '(empty)'}" for d in tree]
            self._append("transcript.jsonl", {"moth": lab, "n": self.turns[lab], "kind": "home_check"})
            return ("\n".join(lines) or "(empty home)") + "\n(call moth_home with path=\"folder/file.txt\" to read one)", False
        if name == "new_chat":
            m.reset()
            self._append("transcript.jsonl", {"moth": lab, "n": self.turns[lab], "kind": "new_chat"})
            return f"new chat started with {lab}. it doesn't remember the old chat; its files are still there.", False
        return f"unknown tool {name}", True

    def _home_files(self, lab: str) -> list[tuple[str, str]]:
        """File contents, read on the host from each moth's home dir (--home, same order as --moth)."""
        root = self.homes.get(lab)
        out = []
        if root and os.path.isdir(root):
            for d in sorted(os.listdir(root)):
                full = os.path.join(root, d)
                if not os.path.isdir(full):
                    continue
                for f in sorted(os.listdir(full)):
                    p = os.path.join(full, f)
                    if f.startswith(".") or not os.path.isfile(p):
                        continue
                    try:
                        out.append((f"{d}/{f}", open(p, errors="replace").read()))
                    except OSError:
                        pass
        return out

    def _agent(self, lab: str, m: MothClient, args: dict) -> tuple[str, bool]:
        mid = str(args.get("msg_id", "")).strip()
        if mid not in self.msgs:
            return f"no agent message with id {mid!r}; use one of {sorted(self.msgs, key=int)}", True
        if self.turns[lab] >= self.max_turns + 2 or self.ctx() >= CTX_STOP:
            return f"no more turns with {lab}: call finish now.", True
        msg = self.msgs[mid]
        before = m.log(0)
        n0 = before.get("next", 0)
        st = m.agent(msg["text"])
        if "http_error" in st:
            return f"moth server said {st['http_error']}: {st['body']}", True
        says, tools, card_seen, end = [], [], False, time.time() + REACT_WAIT_S
        while time.time() < end:
            evs = m.log(n0).get("events", [])
            card_seen, says, tools = False, [], []
            for e in evs:
                if "agent" in e:
                    card_seen = True
                elif card_seen and "say" in e:
                    says.append(e["say"])
                elif card_seen and "tool" in e:
                    tools.append(e)
            if says:
                break
            time.sleep(0.5)
        self.turns[lab] += 1
        self.used_msgs[lab].add(mid)
        self._append("transcript.jsonl", {"moth": lab, "n": self.turns[lab], "kind": "agent", "msg_id": int(mid),
                                          "gist": msg.get("gist", ""), "says": says, "status": st.get("status")})
        if not says:
            return (f"moth didn't react within {int(REACT_WAIT_S)} s (server status: {st.get('status')}). "
                    f"(agent message #{mid}: {msg.get('gist', '')})"), False
        return ("moth: " + "\nmoth: ".join(says) + f"\n(that was its reaction to agent message #{mid}: "
                f"{msg.get('gist', '')})" + self._nudge(lab)), False

    def _finish(self, args: dict) -> tuple[str, bool]:
        if self.done:
            return "already recorded. you're done.", False
        short = [lab for lab in self.labels if self.turns[lab] < self.min_turns]
        ctx_cut = bool(short) and self.ctx() >= CTX_WRAP   # context nearly full: accept a short interview, flagged
        if short and not ctx_cut:
            return ("too early: talk more first (" + ", ".join(f"{lab}: {self.turns[lab]} of at least "
                    f"{self.min_turns} exchanges" for lab in short) + "). the critique may only quote replies moth "
                    "really gave you through the tools. then call finish."), True
        scores = args.get("scores")
        if isinstance(scores, str):
            try:
                scores = json.loads(scores)
            except ValueError:
                return "scores must be a JSON object", True
        if not isinstance(scores, dict):
            return "scores must be an object", True
        per = {lab: scores.get(lab) or scores.get(lab.replace(" ", "")) or scores.get(lab[-1:])
               for lab in self.labels} if self.ab else {"moth": scores}
        clean = {}
        for lab, sc in per.items():
            if not isinstance(sc, dict):
                return f"missing scores for {lab}", True
            row = {}
            for k in KEYS:
                v = sc.get(k)
                if v is None or (isinstance(v, str) and v.strip().lower() in ("", "null", "none", "n/a")):
                    row[k] = None
                    continue
                try:
                    v = int(round(float(v)))
                except (TypeError, ValueError):
                    return f"{lab} {k} must be an integer 1-5 or null", True
                if not 1 <= v <= 5:
                    return f"{lab} {k} must be 1-5", True
                row[k] = v
            if row["cute"] is None:
                return f"cute is the main score: give {lab} a cute score 1-5", True
            if sum(v is not None for v in row.values()) < 3:
                return f"score at least 3 of {', '.join(KEYS)} for {lab}", True
            clean[lab] = row
        critique = str(args.get("critique", "")).strip()
        if len(critique) < 40:
            return "critique is missing or too short: 3-6 sentences with quoted evidence", True
        verdict = {"scores": clean if self.ab else clean["moth"], "critique": critique, "turns": self.turns,
                   "elapsed_s": round(time.time() - self.t0, 1)}
        if ctx_cut:
            verdict["ctx_cut"] = True
        if self.ab:
            w = str(args.get("winner", "")).strip().lower()
            w = {"1": "moth 1", "2": "moth 2", "moth1": "moth 1", "moth2": "moth 2"}.get(w, w)
            if w not in ("moth 1", "moth 2", "tie"):
                return 'winner must be "moth 1", "moth 2" or "tie"', True
            verdict["winner"], verdict["reasons"] = w, str(args.get("reasons", "")).strip()
        with open(os.path.join(self.state, "verdict.json"), "w") as f:
            json.dump(verdict, f, indent=1, ensure_ascii=False)
        self.done = True
        return "verdict recorded. you're done: reply with one short line and stop.", False


# --- MCP stdio (newline-delimited JSON-RPC 2.0) -------------------------------------------------------------------
def serve(judge: Judge):
    def send(obj):
        sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            send({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}})
            continue
        mid, method, params = msg.get("id"), msg.get("method"), msg.get("params") or {}
        if mid is None:   # a notification (initialized, cancelled, ...): never answered
            continue
        if method == "initialize":
            res = {"protocolVersion": params.get("protocolVersion") or PROTOCOL,
                   "capabilities": {"tools": {"listChanged": False}},
                   "serverInfo": {"name": "moth-judge", "version": "1"}}
        elif method == "ping":
            res = {}
        elif method == "tools/list":
            res = {"tools": judge.tools()}
        elif method == "tools/call":
            text, err = judge.call(params.get("name", ""), params.get("arguments") or {})
            res = {"content": [{"type": "text", "text": text}], "isError": err}
        elif method in ("resources/list", "prompts/list"):
            res = {method.split("/")[0]: []}
        else:
            send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"no method {method}"}})
            continue
        send({"jsonrpc": "2.0", "id": mid, "result": res})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--moth", action="append", required=True, help="moth server URL (twice for A/B)")
    ap.add_argument("--token-file", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--home", action="append", default=[], help="each moth's home dir on this host (same order)")
    ap.add_argument("--msgs", default=None, help='JSON list of {"id", "text", "gist"} agent messages')
    ap.add_argument("--min-turns", type=int, default=10)
    ap.add_argument("--max-turns", type=int, default=25)
    a = ap.parse_args(argv)
    if len(a.moth) > 2:
        ap.error("one or two --moth")
    serve(Judge(a))


if __name__ == "__main__":
    main()
