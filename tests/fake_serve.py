"""A stand-in for `python -m moth.serve` in tests/test_pet.py (MOTH_SERVE_CMD points here): no torch, no model.

Takes moth.serve's flags (only --port and --exit-with-parent matter; the rest is ignored), sleeps FAKE_SERVE_DELAY s before it listens
(the model loading), starts a grandchild in its own process group (so the tests can see the whole group stopped),
writes {"pid", "child", "argv"} to FAKE_SERVE_INFO, then answers /api/health, /api/says and /api/log like moth.
FAKE_SERVE_EXIT=N: exit with N instead of listening (a server that dies while waking up).
"""
import json
import os
import socketserver
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def main():
    argv = sys.argv[1:]
    port = int(argv[argv.index("--port") + 1])
    if "--exit-with-parent" in argv:   # the real lifeline, as moth.serve runs it (stdlib only)
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from moth.launch import exit_with_parent
        exit_with_parent()
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    info = os.environ.get("FAKE_SERVE_INFO")
    if info:
        with open(info + ".tmp", "w") as f:
            json.dump({"pid": os.getpid(), "child": child.pid, "argv": argv}, f)
        os.replace(info + ".tmp", info)
    time.sleep(float(os.environ.get("FAKE_SERVE_DELAY", "0")))
    if os.environ.get("FAKE_SERVE_EXIT"):   # the grandchild stays: whoever stops the group must get it too
        sys.exit(int(os.environ["FAKE_SERVE_EXIT"]))

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path.startswith("/api/health"):
                d = {"ok": True, "pid": os.getpid(), "install_root": "fake", "version": "test"}
            elif self.path.startswith(("/api/says", "/api/log")):
                d = {"events": [], "next": 0, "epoch": "fake"}
            else:
                self.send_error(404)
                return
            b = json.dumps(d).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

    class Server(ThreadingHTTPServer):   # as moth.serve.Server: no reverse-DNS lookup between bind() and listen()
        daemon_threads = True

        def server_bind(self):
            socketserver.TCPServer.server_bind(self)
            self.server_name, self.server_port = "127.0.0.1", self.server_address[1]

    srv = Server(("127.0.0.1", port), H)
    print(f"fake moth is awake on {port}", flush=True)
    srv.serve_forever()   # never cleans up the grandchild itself: stopping the process group must


if __name__ == "__main__":
    main()
