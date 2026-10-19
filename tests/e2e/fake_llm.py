"""A fake LLM API for driving the real coding-agent CLIs without a model: every request gets one fixed final reply.

  python tests/e2e/fake_llm.py --port 0 --info FILE [--reply TEXT] [--log FILE]

Speaks, on one port:
  Anthropic Messages  POST /v1/messages[?beta=true]  (streaming SSE when "stream": true, else JSON), as Claude Code calls
                      it with ANTHROPIC_BASE_URL; POST /v1/messages/count_tokens.
  OpenAI Responses    POST /v1/responses  (SSE), as Codex calls a custom provider with wire_api = "responses". The event
                      shapes are those of Codex's own test mocks (codex-rs/core/tests/common/responses.rs, rust-v0.163.0:
                      response.created, output_item.added, output_text.delta, output_item.done, response.completed).
  OpenAI Chat         POST /v1/chat/completions  (SSE chunks + [DONE] when "stream": true, else JSON), for Pi / OpenCode.
  GET anything        /v1/models (a one-model list) or {}.
The reply never contains a tool call, so each agent turn ends at once with that text as its final message.
--info FILE gets {"port", "pid"} once it listens (--port 0 picks a free port); --log FILE gets one JSON line per request
(method, path, a head of the body), for debugging a CLI that calls something unexpected.
"""
import argparse
import json
import os
import socketserver
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPLY = "Fixed the login bug. All tests pass."


def anthropic_events(model: str, text: str):
    mid = "msg_" + uuid.uuid4().hex[:24]
    usage = {"input_tokens": 12, "output_tokens": 1, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    yield "message_start", {"type": "message_start", "message": {
        "id": mid, "type": "message", "role": "assistant", "model": model, "content": [],
        "stop_reason": None, "stop_sequence": None, "usage": usage}}
    yield "content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}
    for piece in _pieces(text):
        yield "content_block_delta", {"type": "content_block_delta", "index": 0,
                                      "delta": {"type": "text_delta", "text": piece}}
    yield "content_block_stop", {"type": "content_block_stop", "index": 0}
    yield "message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                            "usage": {"output_tokens": 9}}
    yield "message_stop", {"type": "message_stop"}


def anthropic_json(model: str, text: str) -> dict:
    return {"id": "msg_" + uuid.uuid4().hex[:24], "type": "message", "role": "assistant", "model": model,
            "content": [{"type": "text", "text": text}], "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 12, "output_tokens": 9}}


def responses_events(model: str, text: str):
    rid, iid = "resp_" + uuid.uuid4().hex[:16], "msg_" + uuid.uuid4().hex[:16]
    usage = {"input_tokens": 12, "input_tokens_details": None, "output_tokens": 9, "output_tokens_details": None,
             "total_tokens": 21}
    yield "response.created", {"type": "response.created", "response": {"id": rid, "model": model}}
    yield "response.output_item.added", {"type": "response.output_item.added", "output_index": 0, "item": {
        "type": "message", "role": "assistant", "id": iid, "content": []}}
    for piece in _pieces(text):
        yield "response.output_text.delta", {"type": "response.output_text.delta", "item_id": iid, "output_index": 0,
                                             "content_index": 0, "delta": piece}
    yield "response.output_item.done", {"type": "response.output_item.done", "output_index": 0, "item": {
        "type": "message", "role": "assistant", "id": iid, "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}]}}
    yield "response.completed", {"type": "response.completed", "response": {"id": rid, "model": model,
                                                                           "status": "completed", "usage": usage}}


def chat_chunks(model: str, text: str):
    cid, now = "chatcmpl-" + uuid.uuid4().hex[:16], int(time.time())
    base = {"id": cid, "object": "chat.completion.chunk", "created": now, "model": model}
    yield {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]}
    for piece in _pieces(text):
        yield {**base, "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]}
    yield {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
           "usage": {"prompt_tokens": 12, "completion_tokens": 9, "total_tokens": 21}}


def chat_json(model: str, text: str) -> dict:
    return {"id": "chatcmpl-" + uuid.uuid4().hex[:16], "object": "chat.completion", "created": int(time.time()),
            "model": model, "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 9, "total_tokens": 21}}


def _pieces(text: str):
    words = text.split(" ")
    return [w + (" " if i < len(words) - 1 else "") for i, w in enumerate(words)]


def make_handler(reply: str, log_path: str | None, bodies_path: str | None = None):
    lock = threading.Lock()

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _log(self, body: bytes):
            if bodies_path and body:   # every request body in full: what the model was really sent
                with lock, open(bodies_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"path": self.path, "body": body.decode("utf-8", "replace")}) + "\n")
            if not log_path:
                return
            rec = {"t": time.time(), "method": self.command, "path": self.path,
                   "ua": self.headers.get("User-Agent", ""), "body": body[:400].decode("utf-8", "replace")}
            with lock, open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

        def _json(self, obj, code=200):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def _sse(self, events, named: bool):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            for ev in events:
                if named:
                    name, data = ev
                    chunk = f"event: {name}\ndata: {json.dumps(data)}\n\n"
                else:
                    chunk = f"data: {json.dumps(ev) if not isinstance(ev, str) else ev}\n\n"
                self.wfile.write(chunk.encode())
                self.wfile.flush()
            self.close_connection = True

        def do_GET(self):
            self._log(b"")
            if self.path.rstrip("/").endswith("/models"):
                return self._json({"object": "list", "data": [{"id": "fake-model", "object": "model", "created": 0,
                                                                "owned_by": "moth-e2e", "type": "model",
                                                                "display_name": "fake"}]})
            self._json({})

        def do_HEAD(self):
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else b""
            self._log(raw)
            try:
                req = json.loads(raw or b"{}")
            except ValueError:
                req = {}
            model = str(req.get("model") or "fake-model")
            path = self.path.split("?", 1)[0].rstrip("/")
            if path.endswith("/messages/count_tokens"):
                return self._json({"input_tokens": 12})
            if path.endswith("/messages"):
                if req.get("stream"):
                    return self._sse(anthropic_events(model, reply), named=True)
                return self._json(anthropic_json(model, reply))
            if path.endswith("/responses"):
                return self._sse(responses_events(model, reply), named=True)
            if path.endswith("/chat/completions"):
                if req.get("stream"):
                    return self._sse(list(chat_chunks(model, reply)) + ["[DONE]"], named=False)
                return self._json(chat_json(model, reply))
            self._json({})

    return H


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def server_bind(self):   # no reverse-DNS lookup (slow on some runners)
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = "127.0.0.1", self.server_address[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--reply", default=REPLY)
    ap.add_argument("--info")
    ap.add_argument("--log")
    ap.add_argument("--bodies", help="FILE: one JSON line per request with its whole body")
    a = ap.parse_args()
    srv = Server(("127.0.0.1", a.port), make_handler(a.reply, a.log, a.bodies))
    if a.info:
        with open(a.info + ".tmp", "w") as f:
            json.dump({"port": srv.server_port, "pid": os.getpid()}, f)
        os.replace(a.info + ".tmp", a.info)
    print(f"fake llm on {srv.server_port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
