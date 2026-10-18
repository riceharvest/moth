"""moth's real HTTP handler (moth/serve.py make_handler) in front of a recording stub engine: no torch, no model.

  python tests/e2e/fake_moth.py --port 0 --info FILE --token-file FILE --record FILE

What the hooks see is the real thing: the Host check, the X-Moth-Token check, /api/agent's cleaning of source / session /
label, /api/says with its ?source=&session= filter. Only the engine is a stub:
  hear_agent  appends {"t", "text", "source", "session", "label"} to --record (one JSON line per post, t = time.time()
              when it arrived), then logs moth's "reaction" the way Moth.react does: a say event tagged
              re={"source", "session"}, saying "heard <label or session>: <first words>".
  say_to      (the chat field, /api/chat) answers "you said: <text>".
<--record>.access gets one JSON line per request (t, method, path with query): which session a display polled for.
The token is made here and written to --token-file (where the hooks read it: ~/.moth/token of the scratch home).
torch, tokenizers and tiny_agent's model modules are faked before serve.py is imported, as in
moth/integrations/test_integrations.py load_serve().
"""
import argparse
import importlib.util
import json
import os
import secrets
import sys
import threading
import time
import types
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]


def load_serve():
    sys.path.insert(0, str(ROOT))
    fakes = {n: types.ModuleType(n) for n in ("torch", "tokenizers", "tiny_agent.checkpoint", "tiny_agent.generate")}
    fakes["torch"].no_grad = lambda: (lambda f: f)
    fakes["tokenizers"].Tokenizer = object
    fakes["tiny_agent.checkpoint"].load_model = None
    fakes["tiny_agent.generate"].KVCache = fakes["tiny_agent.generate"].append = None
    with mock.patch.dict(sys.modules, fakes):
        spec = importlib.util.spec_from_file_location("moth_serve_e2e", ROOT / "moth" / "serve.py")
        serve = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(serve)
    return serve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--info", required=True)
    ap.add_argument("--token-file", required=True)
    ap.add_argument("--record", required=True)
    a = ap.parse_args()
    serve = load_serve()
    token = secrets.token_hex(16)
    os.makedirs(os.path.dirname(os.path.abspath(a.token_file)), exist_ok=True)
    with open(a.token_file, "w") as f:
        f.write(token + "\n")
    lock = threading.Lock()
    tmp = os.path.join(os.path.dirname(os.path.abspath(a.record)), "fake-moth-home")

    def hear_agent(text, source, session=None, label=None):
        with lock, open(a.record, "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.time(), "text": text, "source": source, "session": session,
                                "label": label}) + "\n")
        words = " ".join(text.split()[:4])
        tag = {k: v for k, v in (("source", source), ("session", session)) if v}
        card = {"agent": text, **tag, **({"label": label} if label else {})}   # as Moth.card: the pet's caption
        moth.log.add(card, {"say": f"heard {label or session}: {words}", "re": tag})
        return "queued"

    moth = types.SimpleNamespace(log=serve.Log(), lock=threading.RLock(), hear_agent=hear_agent,
                                 home=serve.Home(tmp, 1, None), a=types.SimpleNamespace(home=tmp, pool_file=None))
    moth.agents = lambda: {"active": 0, "skipped": 0, "sessions": []}
    moth.say_to = lambda text: [{"say": f"you said: {text}"}]
    moth.forget = lambda: moth.log.clear()

    base = serve.make_handler(moth, token)
    access = a.record + ".access"   # every request's time, method and path (with query): who polled what, when

    class H(base):
        def _access(self):
            with lock, open(access, "a", encoding="utf-8") as f:
                f.write(json.dumps({"t": time.time(), "method": self.command, "path": self.path}) + "\n")

        def do_GET(self):
            self._access()
            super().do_GET()

        def do_POST(self):
            self._access()
            super().do_POST()

    srv = serve.Server(("127.0.0.1", a.port), H)
    srv.daemon_threads = True
    with open(a.info + ".tmp", "w") as f:
        json.dump({"port": srv.server_address[1], "pid": os.getpid()}, f)
    os.replace(a.info + ".tmp", a.info)
    print(f"fake moth on {srv.server_address[1]}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
