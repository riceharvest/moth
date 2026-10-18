"""Scripted OpenAI-compatible server for plumbing tests (python tests/fake_oai.py PORT)."""
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, obj):
        b = json.dumps(obj).encode(); self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self): self._send({"object": "list", "data": [{"id": "fake", "object": "model"}]})
    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        msgs = req["messages"]
        if req.get("tools"):
            if msgs[-1]["role"] == "user":
                msg = {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function",
                       "function": {"name": "bash", "arguments": json.dumps({"command": "ls ~"})}}]}
            else:
                msg = {"role": "assistant", "content": "six little rooms. i like the lamps one."}
        elif "JSON list" in msgs[0]["content"]:
            msg = {"role": "assistant", "content": json.dumps([{"name": "Test Lamp.txt", "content": "bright"}])}
        else:
            n = msgs[0]["content"].count("\nyou: ")
            msg = {"role": "assistant", "content": "[END]" if n >= 2 else "what's in your home?"}
        self._send({"id": "x", "object": "chat.completion", "created": 0, "model": "fake",
                    "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}]})

HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
