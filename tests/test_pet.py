"""moth/pet.py: the flight logic, the bubble queue, the poller against serve.py's real HTTP handler (torch faked out,
no model), the platform choice, and (with PySide6) the Petdex sheet and an offscreen run of the real windows.
Plus the one-command start (moth/launch.py + the pet): a server is started when none answers and stopped with the
pet (process group, no orphans, also on SIGTERM/SIGINT/SIGHUP and SIGKILL of the pet: the stdin lifeline on every
OS, plus PR_SET_PDEATHSIG on Linux and the Job Object on Windows); a running moth is
reused and left alone; something else on the port and low memory start nothing. The "server" is tests/fake_serve.py
(MOTH_SERVE_CMD), on a spare port; nothing touches 4747, live/ or ~/.moth.

  python3 tests/test_pet.py      (from ~/moth; a few seconds, only 127.0.0.1; the Qt tests skip without PySide6)

Never opens a window on the desktop: QT_QPA_PLATFORM is forced to "offscreen" before anything imports PySide6.
"""
from __future__ import annotations

import importlib.util
import json
import os
import random
import shlex
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"   # before PySide6 can be imported anywhere
# never the user's ~/.moth/config.json, petdex pets or network: a throwaway config with the daily update check off
# (the tests that want it turn it on), and empty pet folders (child processes inherit all of this)
_SANDBOX = tempfile.mkdtemp(prefix="moth-pet-test-")
os.environ["MOTH_CONFIG_FILE"] = os.path.join(_SANDBOX, "config.json")
os.environ["CODEX_HOME"] = os.path.join(_SANDBOX, "codex")
os.environ["HERMES_HOME"] = os.path.join(_SANDBOX, "hermes")
with open(os.environ["MOTH_CONFIG_FILE"], "w") as _f:
    json.dump({"update_check": False}, _f)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from moth import launch as L
from moth import pet as P

TOKEN = "test-token-not-secret"
try:   # QtGui / QtWidgets, not just PySide6: they load the native libs (libEGL ...) and fail without them
    from PySide6 import QtGui, QtWidgets  # noqa: F401
    HAVE_QT = True
except ImportError:
    HAVE_QT = False


def load_serve():
    spec = importlib.util.spec_from_file_location("moth_itest", ROOT / "moth" / "integrations" / "test_integrations.py")
    itest = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(itest)
    return itest.load_serve()


def closed_url():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return f"http://127.0.0.1:{port}"


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def alive(pid: int) -> bool:
    if os.name == "nt":   # os.kill(pid, 0) is no probe there: signal 0 = CTRL_C_EVENT, sent to a console group
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
        k32.CloseHandle(h)
        return bool(ok) and code.value == 259   # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"   # a zombie is gone, only not reaped yet
    except OSError:
        return True


def wait_until(cond, timeout=10.0, every=0.05) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(every)
    return cond()


def fake_weights(folder: str, complete: bool = True) -> str:
    """config.json + a tiny model.safetensors (truncated when not complete)."""
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "config.json"), "w") as f:
        f.write("{}")
    head = json.dumps({"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}).encode()
    with open(os.path.join(folder, "model.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(head)) + head + (b"\0" * (8 if complete else 3)))
    return folder


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


# --- flight -------------------------------------------------------------------------------------------------------
class WanderTest(unittest.TestCase):
    B = (0, 0, 1920, 1050)

    def sim(self, w, seconds, t0=0.0, cursor=None, dt=1 / 30):
        t = t0
        for _ in range(int(seconds / dt)):
            t += dt
            w.step(dt, t, cursor=cursor(t) if callable(cursor) else cursor)
        return t

    def test_stays_on_screen_and_lands_and_rests(self):
        w = P.Wander(self.B, 56, rng=random.Random(1))
        w.take_off(0.0)
        states, t = set(), 0.0
        for _ in range(600):   # 10 min in 1 s chunks
            t = self.sim(w, 1.0, t)
            states.add(w.state)
            w.idle_since = t   # keep it from dozing: this test is about flying and landing
            self.assertTrue(0 <= w.x <= 1920 and 0 <= w.y <= 1050, (w.x, w.y))
        self.assertIn(P.FLY, states)
        self.assertIn(P.REST, states)

    def test_rest_spots_are_on_edges(self):
        w = P.Wander(self.B, 56, rng=random.Random(2))
        for _ in range(200):
            x, y = w.landing_spot()
            edge = min(x, 1920 - x, y, 1050 - y)
            self.assertLess(edge, 56, (x, y))

    def test_dozes_after_idle_and_excite_wakes(self):
        w = P.Wander(self.B, 56, rng=random.Random(3))
        w.take_off(0.0)
        t = self.sim(w, P.DOZE_AFTER + 60)
        self.assertEqual(w.state, P.DOZE)
        w.excite(t)
        self.assertEqual(w.state, P.FLY)
        self.assertTrue(w.excited)

    def test_sleep_holds_until_woken(self):
        w = P.Wander(self.B, 56, rng=random.Random(4))
        w.take_off(0.0)
        self.sim(w, 2)
        w.sleep(2.0)
        t = self.sim(w, 60, 2.0)
        self.assertEqual(w.state, P.DOZE)
        w.excite(t)                 # a new line doesn't wake a moth put to sleep
        self.assertEqual(w.state, P.DOZE)
        w.sleep(t, False)
        self.assertEqual(w.state, P.REST)

    def test_drawn_to_a_moving_cursor(self):
        def mean_dist(cur):
            w = P.Wander(self.B, 56, rng=random.Random(5))
            w.x, w.y = 200, 200
            w.take_off(0.0)
            w.fly_for = 1e9
            ds, t = [], 0.0
            for i in range(120):
                pos = (1500 + (i % 2), 800)  # wiggles: a cursor that moves
                t = self.sim(w, 0.5, t, cursor=pos if cur else None)
                w.idle_since = t
                ds.append(((w.x - 1500) ** 2 + (w.y - 800) ** 2) ** .5)
            return sum(ds[40:]) / len(ds[40:])
        near, far = mean_dist(True), mean_dist(False)
        self.assertLess(near, far * 0.75, (near, far))
        self.assertGreater(near, 60)   # circles the light rather than sitting on it

    def test_stale_cursor_fades(self):
        w = P.Wander(self.B, 56, rng=random.Random(6))
        w.see_cursor(0.0, (10, 10))
        w.see_cursor(100.0, (10, 10))   # same spot: not "seen" again
        self.assertEqual(w.cursor_seen, 0.0)

    def test_no_wander_sits_in_the_corner(self):
        w = P.Wander(self.B, 56, wander=False, rng=random.Random(7))
        home = w.home()
        w.excite(0.0)
        self.sim(w, 30)
        self.assertEqual((w.x, w.y), home)
        self.assertIn(w.state, (P.REST, P.DOZE))
        self.assertGreater(home[0], 1700)
        self.assertGreater(home[1], 1000)

    def test_drop_near_edge_rests(self):
        w = P.Wander(self.B, 56, rng=random.Random(8))
        w.grab(0.0)
        w.step(0.03, 0.03)
        self.assertEqual(w.state, P.HELD)
        w.drop(1.0, 1900, 500)
        self.assertEqual(w.state, P.REST)
        w.grab(2.0)
        w.drop(3.0, 900, 500)
        self.assertEqual(w.state, P.FLY)

    def test_fps(self):
        w = P.Wander(self.B, 56, rng=random.Random(9))
        w.state = P.DOZE
        self.assertLessEqual(w.fps(), 10)
        w.take_off(0)
        self.assertEqual(w.fps(), 30)


# --- bubbles ------------------------------------------------------------------------------------------------------
class BubbleTest(unittest.TestCase):
    def test_wrap_lowercase_and_ttl(self):
        lines = P.wrap_text("I Think\x1b[31m the  LAMP is warm and i would like to sit on it for a while please")
        self.assertTrue(all(len(x) <= P.WRAP for x in lines))
        self.assertEqual(lines[0], "i think the lamp is warm and i")
        self.assertEqual(P.bubble_ttl(["hi"]), 6.0)
        self.assertEqual(P.bubble_ttl(["x" * 32] * 5), 10.0)

    def test_queue_one_at_a_time_then_fades(self):
        c = Clock()
        q = P.BubbleQueue(clock=c)
        q.push("hello", "claude · repo")
        q.push("second")
        b, _ = q.current()
        self.assertEqual((b.lines, b.caption), (["hello"], "claude · repo"))
        c.t += 3
        self.assertEqual(q.current()[1], 1.0)
        c.t += b.ttl - 3 - 0.4
        self.assertLess(q.current()[1], 1.0)   # fading out
        c.t += 0.5
        b2, _ = q.current()
        self.assertEqual(b2.lines, ["second"])
        c.t += 20
        self.assertEqual(q.current(), (None, 0.0))

    def test_long_line_pages_and_burst_cap(self):
        q = P.BubbleQueue(clock=Clock())
        q.push("word " * 300)
        self.assertEqual(len(q.waiting), P.MAX_PAGES)
        self.assertTrue(q.waiting[-1].lines[-1].endswith("…"))
        self.assertIsNone(q.waiting[1].caption)
        for i in range(10):
            q.push(f"line {i}")
        self.assertEqual([b.lines[0] for b in q.waiting], ["line 6", "line 7", "line 8", "line 9"])


# --- the poller against serve.py's real handler ------------------------------------------------------------------
class PollTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.serve = load_serve()

    def setUp(self):
        serve = self.serve
        self.tmp = tempfile.mkdtemp()
        self.tok = os.path.join(self.tmp, "token")
        with open(self.tok, "w") as f:
            f.write(TOKEN + "\n")
        m = types.SimpleNamespace(log=serve.Log(), lock=threading.RLock(), chats=[],
                                  home=types.SimpleNamespace(tree=list, owned=True),
                                  agents=lambda: {"active": 0, "skipped": 0, "sessions": []})
        m.log.add({"say": "old line from before the pet"})

        def say_to(text):
            m.chats.append(text)
            evs = [{"say": f"you said {text}"}]
            m.log.add({"you": text}, *evs)
            return evs
        m.say_to = say_to
        self.m = m
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), serve.make_handler(m, TOKEN))
        self.srv.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.srv.server_port}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()

    def test_says_without_token(self):
        p = P.Poller(self.url, os.path.join(self.tmp, "missing"))
        self.assertEqual(p.poll(), ([], "up"))           # history skipped
        self.m.log.add({"agent": "Fixed it.", "source": "claude-code", "label": "repo"}, {"say": "yay"})
        self.assertEqual(p.poll(), ([("yay", None)], None))   # /api/says: no captions
        self.assertEqual(p.poll(), ([], None))

    def test_log_with_token_gives_captions(self):
        p = P.Poller(self.url, self.tok)
        p.poll()
        self.m.log.add({"agent": "a", "source": "codex", "label": "api", "skipped": True},
                       {"agent": "b", "source": "claude-code", "label": "moth-repo"}, {"say": "ooh"},
                       {"you": "hi"}, {"say": "hello you"}, {"tool": "bash", "args": {}, "result": "x"})
        self.assertEqual(p.poll()[0], [("ooh", "claude · moth-repo"), ("hello you", None)])

    def test_lines_carry_source_label_reply_and_index(self):
        """What the pet's filters and "what did i miss" need: the harness a reaction answered (the say's own re tag
        beats the card before it: a batch's last card can be another session's), its project, and replies to you."""
        p = P.Poller(self.url, self.tok)
        p.poll()
        start = p.cursor
        self.m.log.add({"agent": "b", "source": "codex", "label": "api"}, {"say": "ooh", "re": {"source": "codex"}},
                       {"you": "hi"}, {"say": "hello you"},
                       {"agent": "c", "source": "claude-code", "label": "moth-repo"},
                       {"say": "a batch", "re": {"source": "hermes", "session": "s1"}})
        got = p.poll()[0]
        self.assertEqual(got, [("ooh", "codex · api"), ("hello you", None), ("a batch", "hermes")])
        self.assertEqual([(x.source, x.label, x.reply) for x in got],
                         [("codex", "api", False), (None, None, True), ("hermes", None, False)])
        self.assertEqual([x.index for x in got], [start + 1, start + 3, start + 5])
        # /api/says (no token): text only, nothing to filter on, and never a "reply"
        q = P.Poller(self.url, os.path.join(self.tmp, "missing"))
        q.poll()
        self.m.log.add({"agent": "d", "source": "pi"}, {"say": "plain"})
        (line,), _ = q.poll()
        self.assertEqual((line.source, line.label, line.reply), (None, None, False))

    def test_history_is_what_the_log_held_at_start(self):
        self.m.log.add({"agent": "x", "source": "pi", "label": "web"}, {"say": "earlier one"},
                       {"you": "q"}, {"say": "an answer, not a reaction"})
        p = P.Poller(self.url, self.tok)
        self.assertEqual(p.history(), [])   # not polled yet
        p.poll()
        self.m.log.add({"say": "after the start"})
        h = p.history()
        self.assertEqual([x.text for x in h], ["old line from before the pet", "earlier one"])
        self.assertEqual((h[1].source, h[1].label), ("pi", "web"))
        self.assertEqual(p.history(keep=1)[0].text, "earlier one")
        self.assertEqual(p.poll()[0], [("after the start", None)])   # the cursor and card state are untouched
        self.assertEqual(P.Poller(closed_url(), self.tok).history(), [])

    def test_wrong_token_falls_back_to_says(self):
        with open(self.tok, "w") as f:
            f.write("nope")
        p = P.Poller(self.url, self.tok)
        self.assertEqual(p.poll(), ([], "up"))
        self.m.log.add({"say": "still here"})
        self.assertEqual(p.poll()[0], [("still here", None)])
        self.assertEqual(p.bad_token, "nope")

    def test_new_epoch_starts_over(self):
        p = P.Poller(self.url, self.tok)
        p.poll()
        self.m.log.clear()   # moth restarted / chat reset
        self.m.log.add({"say": "one"}, {"say": "two"}, {"say": "three"})
        self.assertEqual([t for t, _ in p.poll()[0]], ["two", "three"])

    def test_chat_goes_through_and_reply_comes_by_poll(self):
        p = P.Poller(self.url, self.tok)
        p.poll()
        self.assertIsNone(P.send_chat(self.url, self.tok, "  hi moth  "))
        self.assertEqual(self.m.chats, ["hi moth"])
        self.assertEqual(p.poll()[0], [("you said hi moth", None)])

    def test_chat_errors(self):
        self.assertIn("token", P.send_chat(self.url, os.path.join(self.tmp, "missing"), "hi"))
        bad = os.path.join(self.tmp, "bad")
        with open(bad, "w") as f:
            f.write("wrong")
        self.assertIn("token", P.send_chat(self.url, bad, "hi"))
        self.assertIn("reach", P.send_chat(closed_url(), self.tok, "hi"))

    def test_down_backoff_and_recovery(self):
        p = P.Poller(closed_url(), self.tok)
        self.assertEqual(p.poll(), ([], "down"))
        self.assertEqual(p.poll(), ([], None))           # said once per outage
        p.poll()
        self.assertEqual(p.interval(), P.POLL * 8)
        for _ in range(10):
            p.poll()
        self.assertEqual(p.interval(), P.MAX_BACKOFF)
        p.url = self.url
        self.assertEqual(p.poll(), ([], "up"))
        self.assertEqual(p.interval(), P.POLL)


class Garbage(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        body = b"<html>not moth</html>" if "says" in self.path else b'{"events": "nope"}'
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class GarbageTest(unittest.TestCase):
    def test_not_moth_counts_as_asleep(self):
        srv = ThreadingHTTPServer(("127.0.0.1", 0), Garbage)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            for tok in ("/nonexistent", None):
                p = P.Poller(f"http://127.0.0.1:{srv.server_port}", tok or os.devnull)
                self.assertEqual(p.poll(), ([], "down"))
        finally:
            srv.shutdown()
            srv.server_close()


class PlatformTest(unittest.TestCase):
    def test_choose(self):
        env = {"WAYLAND_DISPLAY": "wayland-0", "DISPLAY": ":0"}
        self.assertEqual(P.choose_platform(env, "linux"), "xwayland")
        self.assertEqual(env["QT_QPA_PLATFORM"], "xcb;wayland")
        self.assertEqual(P.choose_platform({"QT_QPA_PLATFORM": "offscreen", "WAYLAND_DISPLAY": "w"}, "linux"), "set")
        self.assertEqual(P.choose_platform({"XDG_SESSION_TYPE": "wayland"}, "linux"), "wayland")
        self.assertEqual(P.choose_platform({"DISPLAY": ":0"}, "linux"), "native")
        self.assertEqual(P.choose_platform({"WAYLAND_DISPLAY": "w", "DISPLAY": ":0"}, "darwin"), "native")

    def test_caption(self):
        self.assertEqual(P.caption_for({"source": "dsh", "label": "x"}), "deepseek · x")
        self.assertEqual(P.caption_for({}), "agent")
        self.assertEqual(len(P.caption_for({"source": "pi", "label": "y" * 80})), P.WRAP)


# --- the one-command start: moth/launch.py ---------------------------------------------------------------------------
FAKE = str(ROOT / "tests" / "fake_serve.py")


class LaunchBase(unittest.TestCase):
    """Every server is tests/fake_serve.py on a spare port; its pids land in self.info. tearDown kills leftovers."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.info = os.path.join(self.tmp, "fake.json")
        self.model = fake_weights(os.path.join(self.tmp, "weights"))
        self.env_before = {k: os.environ.get(k) for k in
                           ("MOTH_SERVE_CMD", "MOTH_SERVE_LOG", "FAKE_SERVE_INFO", "FAKE_SERVE_DELAY", "FAKE_SERVE_EXIT",
                            "FAKE_SERVE_GATE", "MOTH_CACHE")}
        os.environ["MOTH_CACHE"] = os.path.join(self.tmp, "cache")   # a pip install's weights: never ~/.cache
        os.environ.update(MOTH_SERVE_CMD=shlex.join([sys.executable, FAKE]), FAKE_SERVE_INFO=self.info,
                          MOTH_SERVE_LOG=os.path.join(self.tmp, "serve.log"), FAKE_SERVE_DELAY="0")
        os.environ.pop("FAKE_SERVE_EXIT", None)
        os.environ.pop("FAKE_SERVE_GATE", None)
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.procs: list[subprocess.Popen] = []

    def tearDown(self):
        for k, v in self.env_before.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        for pid in self.pids() + self.pids(self.info + ".ext"):
            if alive(pid):
                try:
                    os.kill(pid, getattr(signal, "SIGKILL", signal.SIGTERM))   # Windows: SIGTERM = TerminateProcess
                except OSError:
                    pass
        for p in self.procs:
            if p.poll() is None:
                p.kill()
            p.wait()

    def pids(self, info: str | None = None) -> list[int]:
        try:
            with open(info or self.info) as f:
                d = json.load(f)
            return [d["pid"], d["child"]]
        except (OSError, ValueError):
            return []

    def fake_info(self) -> dict:
        with open(self.info) as f:
            return json.load(f)

    def external_moth(self) -> subprocess.Popen:
        """A moth someone else started (not the pet): the fake server, in its own session."""
        env = dict(os.environ, FAKE_SERVE_INFO=self.info + ".ext")   # its pids apart from a server the pet starts
        out = os.path.join(self.tmp, "external.log")
        with open(out, "w") as f:
            p = subprocess.Popen([sys.executable, FAKE, "--port", str(self.port)], start_new_session=True, env=env,
                                 stdout=f, stderr=subprocess.STDOUT)
        self.procs.append(p)
        self.assertTrue(wait_until(lambda: L.probe(self.url)[0] == "moth"),
                        f"the external fake moth never woke: {self.read(out)!r}")
        return p

    @staticmethod
    def read(path: str) -> str:
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                return f.read()[-2000:]
        except OSError as e:
            return str(e)

    def serve_log(self) -> str:
        return self.read(os.environ["MOTH_SERVE_LOG"])

    def garbage_server(self):
        srv = ThreadingHTTPServer(("127.0.0.1", self.port), Garbage)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        return srv


class LaunchTest(LaunchBase):
    def test_weights_and_paths(self):
        self.assertTrue(L.complete(self.model))
        self.assertFalse(L.complete(fake_weights(os.path.join(self.tmp, "half"), complete=False)))
        self.assertFalse(L.complete(os.path.join(self.tmp, "nothing")))
        self.assertEqual(L.server_paths(self.model)["model"], self.model)
        with self.assertRaises(L.LaunchError):
            L.server_paths(os.path.join(self.tmp, "half"))
        dev = os.path.join(self.tmp, "dev")             # the dev checkout: try_moth.sh's order, first complete wins
        os.makedirs(dev)
        with self.assertRaises(L.LaunchError):
            L.server_paths(code=dev)
        fake_weights(os.path.join(dev, "runs/final2/ckpt_100"), complete=False)
        fake_weights(os.path.join(dev, "runs/final/ckpt_100"))
        fake_weights(os.path.join(dev, "runs/v2/model"))
        d = L.server_paths(code=dev)
        self.assertEqual(d["model"], os.path.join(dev, "runs/final/ckpt_100"))
        self.assertEqual(d["home"], os.path.join(dev, "live", "home"))
        self.assertEqual(d["log"], os.environ["MOTH_SERVE_LOG"])
        del os.environ["MOTH_SERVE_LOG"]
        self.assertEqual(L.server_paths(code=dev)["log"], os.path.join(dev, "live", "serve.log"))
        repo = os.path.join(self.tmp, "moth-112m")      # the Hugging Face layout: weights next to code/
        fake_weights(repo)
        os.makedirs(os.path.join(repo, "code"))
        d = L.server_paths(code=os.path.join(repo, "code"))
        self.assertEqual(d["model"], repo)
        self.assertEqual(d["log"], os.path.join(os.path.expanduser("~"), ".moth", "serve.log"))

    def test_probe(self):
        self.assertEqual(L.probe(self.url)[0], "none")
        self.garbage_server()
        kind, why = L.probe(self.url)
        self.assertEqual(kind, "other")
        self.assertIn("moth", why)

    def test_probe_kinds_of_failure(self):
        """Every OS: how each failure is classified. A connect that times out is a free port (Windows retries a refused
        loopback connect for ~2 s, longer than probe's timeout), a listener that accepts and stays silent is not."""
        import urllib.error
        from unittest import mock
        cases = [(urllib.error.URLError(ConnectionRefusedError(10061, "refused")), "none"),
                 (urllib.error.URLError(OSError(10061, "WSAECONNREFUSED")), "none"),
                 (urllib.error.URLError(TimeoutError("timed out")), "none"),   # connect timed out: nothing accepted
                 (TimeoutError("timed out"), "other"),                          # accepted, then silent: not moth
                 (ConnectionResetError(104, "reset"), "other"),
                 (urllib.error.URLError(OSError(113, "no route")), "other")]
        for exc, want in cases:
            with self.subTest(exc=repr(exc)), mock.patch.object(L._OPENER, "open", side_effect=exc):
                self.assertEqual(L.probe(self.url)[0], want)

    def test_probe_silent_listener_is_other(self):
        srv = socket.socket()
        srv.bind(("127.0.0.1", self.port))
        srv.listen(4)   # accepts (the kernel does) but never answers
        self.addCleanup(srv.close)
        kind, why = L.probe(self.url, timeout=0.5)
        self.assertEqual(kind, "other", why)

    def test_probe_old_moth_without_health(self):
        class Old(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if not self.path.startswith("/api/says"):
                    return self.send_error(404)
                b = json.dumps({"events": [], "next": 3, "epoch": "e"}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)
        srv = ThreadingHTTPServer(("127.0.0.1", self.port), Old)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            self.assertEqual(L.probe(self.url)[0], "moth")
        finally:
            srv.shutdown()
            srv.server_close()

    def test_meminfo(self):
        mi = os.path.join(self.tmp, "meminfo")
        with open(mi, "w") as f:
            f.write("MemTotal:       65000000 kB\nMemFree:  100 kB\nMemAvailable:    2097152 kB\n")
        self.assertEqual(L.mem_available_gb(mi), 2.0)
        self.assertIsNone(L.mem_available_gb(os.path.join(self.tmp, "no-meminfo")))   # macOS/Windows: no check

    def test_windows_stop_takes_the_whole_tree(self):
        """Server.stop()'s Windows branch, driven with fakes (no Windows here): Ctrl-Break first, then the Job Object
        ends what is left (also after the server itself exited: its children are in the job); without a job,
        taskkill /T /F while the server still holds its tree."""
        from unittest import mock

        class Proc:
            def __init__(self, exits_on_break):
                self.pid, self.rc, self.exits_on_break, self.calls = 4242, None, exits_on_break, []

            def poll(self):
                return self.rc

            def send_signal(self, sig):
                self.calls.append(("signal", sig))
                if self.exits_on_break:
                    self.rc = 0

            def wait(self, timeout=None):
                if self.rc is None:
                    raise subprocess.TimeoutExpired("x", timeout)
                return self.rc

            def kill(self):
                self.calls.append("kill")
                self.rc = 1

        for exits, with_job in ((True, True), (False, True), (True, False), (False, False)):
            with self.subTest(exits_on_break=exits, job=with_job):
                srv = L.Server(["x"], "http://127.0.0.1:1", os.path.join(self.tmp, "log"))
                srv.proc = proc = Proc(exits)
                job = mock.Mock() if with_job else None
                srv.job = job
                with mock.patch.object(L.os, "name", "nt"), \
                        mock.patch.object(L.signal, "CTRL_BREAK_EVENT", 1, create=True), \
                        mock.patch.object(L, "_taskkill_tree") as tk:
                    srv.stop(grace=0.01)
                    srv.stop()   # once only
                self.assertEqual(proc.calls[0], ("signal", 1))
                if with_job:
                    job.kill.assert_called_once()   # even when the server exited on its own
                    tk.assert_not_called()
                else:
                    self.assertEqual(tk.call_args_list, [] if exits else [mock.call(4242)])
                self.assertIsNotNone(proc.rc)

    def test_autostart_then_stop_leaves_no_orphan(self):
        gate = os.path.join(self.tmp, "loaded")   # the fake server "loads" until this file exists (a 0.6 s delay
        os.environ["FAKE_SERVE_GATE"] = gate      # was outlasted by prepare + probe on a busy Windows runner)
        boot = L.prepare(self.url, model=self.model, extra=["--react-every", "0"], mem=lambda: 32.0)
        self.assertEqual(boot.state, "starting", boot)
        self.assertEqual(L.probe(self.url, timeout=0.3)[0], "none")   # still "loading" (Windows: connect times out)
        open(gate, "w").close()
        ok, why = boot.server.wait_ready(timeout=20)
        self.assertTrue(ok, f"{why}: {self.serve_log()!r}")
        argv = self.fake_info()["argv"]
        self.assertEqual(argv[-2:], ["--react-every", "0"])   # passthrough last: the user's flags win
        self.assertEqual(argv[argv.index("--device") + 1], "cpu")
        self.assertEqual(argv[argv.index("--model") + 1], self.model)
        pid, child = self.pids()
        self.assertTrue(alive(pid) and alive(child))
        boot.server.stop()
        boot.server.stop()   # idempotent (aboutToQuit, run()'s finally and atexit all call it)
        self.assertTrue(wait_until(lambda: not alive(pid) and not alive(child), 5), "server or its child survived")
        self.assertEqual(L.probe(self.url)[0], "none")
        with open(os.environ["MOTH_SERVE_LOG"]) as f:
            self.assertIn("fake moth is awake", f.read())

    def test_reuses_a_running_moth(self):
        ext = self.external_moth()
        boot = L.prepare(self.url, model=self.model)
        self.assertEqual(boot.state, "ready")
        self.assertIsNone(boot.server)
        self.assertFalse(os.path.exists(self.info))   # nothing started
        self.assertIsNone(ext.poll())

    def test_not_moth_on_port_starts_nothing(self):
        self.garbage_server()
        boot = L.prepare(self.url, model=self.model, mem=lambda: 32.0)
        self.assertEqual(boot.state, "error")
        self.assertIn("something else is on port", boot.message)
        self.assertIn(str(self.port), boot.message)
        self.assertIsNone(boot.server)
        self.assertFalse(os.path.exists(self.info))

    def test_low_memory_refuses(self):
        boot = L.prepare(self.url, model=self.model, mem=lambda: 1.4)
        self.assertEqual((boot.state, boot.server), ("error", None))
        self.assertIn("not enough memory to wake moth", boot.message)
        self.assertFalse(os.path.exists(self.info))

    def test_no_server_and_remote_url_only_wait(self):
        self.assertEqual(L.prepare(self.url, no_server=True, model=self.model).state, "wait")
        boot = L.prepare(f"http://127.0.0.2:{self.port}", model=self.model)   # not 127.0.0.1/localhost: "remote"
        self.assertEqual(boot.state, "wait")
        self.assertIn("only start moth on this machine", boot.message)
        self.assertFalse(os.path.exists(self.info))

    def test_pip_install_paths(self):
        cache = L.weights.cache_dir()
        self.assertTrue(cache.startswith(os.environ["MOTH_CACHE"]))
        d = L.server_paths(mode="pip")
        self.assertEqual((d["model"], d["download"]), (cache, cache))   # first run: download before starting
        self.assertEqual(d["tokenizer"], os.path.join(cache, "tokenizer.json"))
        self.assertEqual(d["home"], os.path.join(os.path.expanduser("~"), ".moth", "home"))   # not site-packages
        del os.environ["MOTH_SERVE_LOG"]
        self.assertEqual(L.server_paths(mode="pip")["log"], os.path.join(os.path.expanduser("~"), ".moth", "serve.log"))
        fake_weights(cache)
        with open(os.path.join(cache, "tokenizer.json"), "w") as f:
            f.write("{}")
        self.assertIsNone(L.server_paths(mode="pip")["download"])   # downloaded once: used as it is

    def test_pip_first_run_downloads_before_starting(self):
        from unittest import mock
        with mock.patch.object(L, "install_mode", return_value="pip"):
            boot = L.prepare(self.url, mem=lambda: 32.0)
        self.assertEqual(boot.state, "downloading", boot)
        self.assertEqual(boot.download, L.weights.cache_dir())
        self.assertIsNone(boot.server.proc)   # nothing runs until the weights are there
        self.assertFalse(os.path.exists(self.info))
        got = []
        with mock.patch.object(L.weights, "ensure", side_effect=lambda folder: got.append(folder) or folder):
            self.assertIsNone(boot.fetch())
        self.assertEqual(got, [boot.download])
        with mock.patch.object(L.weights, "ensure", side_effect=SystemExit("moth: can't get it (offline)")):
            self.assertEqual(boot.fetch(), "moth: can't get it (offline)")
        boot.server.stop()   # never started: nothing to do, no error

    def test_server_that_dies_while_waking(self):
        os.environ["FAKE_SERVE_EXIT"] = "3"
        boot = L.prepare(self.url, model=self.model, mem=lambda: 32.0)
        ok, why = boot.server.wait_ready(timeout=20)
        self.assertFalse(ok)
        self.assertIn("stopped while waking up", why)
        self.assertIn(os.environ["MOTH_SERVE_LOG"], why)
        _, child = self.pids()
        self.assertTrue(alive(child))   # its leftover grandchild ...
        boot.server.stop()
        self.assertTrue(wait_until(lambda: not alive(child), 5))   # ... goes with the group

    def test_lifeline_flag_only_for_a_server_the_pet_starts(self):
        paths = {"model": "m", "tokenizer": "t", "home": "h"}
        self.assertNotIn(L.EXIT_WITH_PARENT, L.server_command(1, paths, ["--x"]))   # python -m moth.launch, ...
        cmd = L.server_command(1, paths, ["--react-every", "0"], exit_with_parent=True)
        self.assertEqual(cmd[-3:], [L.EXIT_WITH_PARENT, "--react-every", "0"])     # the user's flags stay last

    def _orphan_parent(self) -> tuple[subprocess.Popen, int]:
        """A stand-in pet: a process that starts the server through launch.prepare with the parent-death signal and
        the Job Object switched off, so the stdin lifeline is the only guard left (macOS's situation, on any OS)."""
        code = (f"import sys, time; sys.path.insert(0, {str(ROOT)!r}); from moth import launch as L\n"
                "L._deathsig = lambda: None\nL._WinJob.make = staticmethod(lambda proc: None)\n"
                f"boot = L.prepare({self.url!r}, model={self.model!r}, mem=lambda: 32.0)\n"
                "assert boot.state == 'starting', boot\nok, why = boot.server.wait_ready(timeout=20)\n"
                "print('ready' if ok else why, flush=True)\ntime.sleep(600)\n")
        p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
        self.procs.append(p)
        self.assertEqual(p.stdout.readline().strip(), "ready", self.serve_log())
        return p, self.pids()[0]

    def test_lifeline_stops_the_server_when_its_launcher_is_killed(self):
        parent, pid = self._orphan_parent()
        self.assertIn(L.EXIT_WITH_PARENT, self.fake_info()["argv"])
        parent.kill()   # SIGKILL / TerminateProcess: no cleanup in the launcher at all
        parent.wait()
        self.assertTrue(wait_until(lambda: not alive(pid), 10), f"orphaned server: {self.serve_log()!r}")
        self.assertIn("the program that started me is gone", self.serve_log())

    def test_without_the_flag_stdin_eof_changes_nothing(self):
        """A server run on its own (python -m moth) never watches stdin: EOF there (e.g. </dev/null) is harmless."""
        p = subprocess.Popen([sys.executable, FAKE, "--port", str(self.port)], stdin=subprocess.PIPE,
                             stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, start_new_session=os.name != "nt")
        self.procs.append(p)
        self.assertTrue(wait_until(lambda: L.probe(self.url)[0] == "moth"))
        p.stdin.close()
        time.sleep(1.0)
        self.assertIsNone(p.poll())
        self.assertEqual(L.probe(self.url)[0], "moth")   # (tearDown kills it and its child)


@unittest.skipUnless(HAVE_QT, "PySide6 not installed")
class PetStartsServerTest(LaunchBase):
    """The real windows (offscreen) deciding at start: wake a server, reuse one, or say why not."""

    def run_pet(self, check, args=(), timeout=20.0, model=True) -> dict:
        """P.run with check(pet, seen) on a 50 ms timer until it returns True; seen["texts"]: every bubble shown."""
        seen: dict = {"texts": []}

        def ready(pet):
            from PySide6.QtCore import QTimer
            deadline = time.monotonic() + timeout
            seen["pet"] = pet

            def tick():
                pet.tick()
                for b in [pet.bubbles.now_showing] + pet.bubbles.waiting:
                    if b is not None and " ".join(b.lines) not in seen["texts"]:
                        seen["texts"].append(" ".join(b.lines))
                if check(pet, seen) or time.monotonic() > deadline:
                    seen["timed_out"] = time.monotonic() > deadline
                    pet.quit()
            t = QTimer(pet.win)
            t.timeout.connect(tick)
            t.start(50)
        a = P.parse_args(["--url", self.url, "--token-file", os.path.join(self.tmp, "no-token"),
                          *(["--model", self.model] if model else []), *args])
        seen["rc"] = P.run(a, on_ready=ready)
        return seen

    def test_wakes_a_server_and_stops_it_on_quit(self):
        gate = os.path.join(self.tmp, "loaded")   # the server "loads" until the pet has been seen booting
        os.environ["FAKE_SERVE_GATE"] = gate      # (not a fixed delay a slow runner's Qt start-up can outlast)

        def check(pet, seen):
            seen.setdefault("first", (pet.booting, pet.w.asleep, list(seen["texts"])))
            if not os.path.exists(gate):
                open(gate, "w").close()
            return not pet.booting and pet.poller.up and not pet.server_down
        seen = self.run_pet(check, ["--server-arg=--quiet-after", "--server-arg=0", "--", "--react-every", "0"])
        self.assertFalse(seen["timed_out"], (seen["texts"], self.serve_log()))
        booting, asleep, texts = seen["first"]
        self.assertTrue(booting and asleep)
        self.assertIn(P.WAKING, texts)
        self.assertNotIn(P.wrap_text(P.ASLEEP)[0], " ".join(seen["texts"]))   # not "server asleep" while it loads
        pet = seen["pet"]
        self.assertEqual(pet.boot.state, "starting")
        self.assertFalse(pet.w.asleep)   # woke up when moth answered
        self.assertEqual(self.fake_info()["argv"][-4:], ["--quiet-after", "0", "--react-every", "0"])
        pid, child = self.pids()
        self.assertTrue(wait_until(lambda: not alive(pid) and not alive(child), 5), "the pet left its server behind")

    def test_reuses_a_running_moth_and_leaves_it_running(self):
        ext = self.external_moth()
        seen = self.run_pet(lambda pet, seen: pet.poller.up)
        self.assertFalse(seen["timed_out"], (seen["texts"], self.serve_log()))
        self.assertEqual(seen["pet"].boot.state, "ready")
        self.assertIsNone(seen["pet"].boot.server)
        self.assertFalse(os.path.exists(self.info))   # the pet started nothing
        time.sleep(0.3)
        self.assertIsNone(ext.poll(), "the pet stopped a moth it didn't start")
        self.assertEqual(L.probe(self.url)[0], "moth")

    def test_not_moth_on_the_port(self):
        self.garbage_server()
        t0 = time.monotonic()
        seen = self.run_pet(lambda pet, seen: time.monotonic() - t0 > 2.5)   # long enough for polls to fail
        self.assertEqual(seen["pet"].boot.state, "error")
        self.assertTrue(any("something else is on port" in t for t in seen["texts"]), seen["texts"])
        self.assertFalse(any("asleep" in t for t in seen["texts"]), seen["texts"])   # the reason stays the message
        self.assertTrue(seen["pet"].w.asleep)
        self.assertFalse(os.path.exists(self.info))

    def test_low_memory_refuses_to_wake_moth(self):
        from unittest import mock
        with mock.patch.object(L, "mem_available_gb", return_value=1.2):
            seen = self.run_pet(lambda pet, seen: bool(seen["texts"]))
        self.assertEqual(seen["pet"].boot.state, "error")
        self.assertIn("not enough memory to wake moth", " ".join(seen["texts"]))
        self.assertFalse(os.path.exists(self.info))
        self.assertEqual(L.probe(self.url)[0], "none")

    def test_pip_first_run_downloads_then_wakes(self):
        from unittest import mock

        def ensure(folder):   # the ~860 MB download, faked: a second, then the files are there
            time.sleep(1.0)
            fake_weights(folder)
            with open(os.path.join(folder, "tokenizer.json"), "w") as f:
                f.write("{}")
            return folder

        def check(pet, seen):
            seen.setdefault("first", (pet.downloading, pet.booting, list(seen["texts"]), os.path.exists(self.info)))
            return pet.poller.up and not pet.server_down
        with mock.patch.object(L, "install_mode", return_value="pip"), \
                mock.patch.object(L.weights, "ensure", side_effect=ensure):
            seen = self.run_pet(check, model=False)
        self.assertFalse(seen["timed_out"], (seen["texts"], self.serve_log()))
        downloading, booting, texts, started = seen["first"]
        self.assertTrue(downloading and not booting and not started)   # the server waits for the weights
        self.assertEqual(texts, [P.DOWNLOADING])
        self.assertIn(P.WAKING, seen["texts"])                          # then the usual waking up
        argv = self.fake_info()["argv"]
        self.assertEqual(argv[argv.index("--model") + 1], L.weights.cache_dir())
        pid, child = self.pids()
        self.assertTrue(wait_until(lambda: not alive(pid) and not alive(child), 5))

    def test_pip_download_failure_says_why(self):
        from unittest import mock
        with mock.patch.object(L, "install_mode", return_value="pip"), \
                mock.patch.object(L.weights, "ensure", side_effect=SystemExit("moth: can't reach the hub")):
            seen = self.run_pet(lambda pet, seen: any("hub" in t for t in seen["texts"]), model=False)
        self.assertFalse(seen["timed_out"], (seen["texts"], self.serve_log()))
        self.assertFalse(seen["pet"].downloading)
        self.assertFalse(os.path.exists(self.info))   # nothing started

    def test_no_server_flag_only_waits(self):
        seen = self.run_pet(lambda pet, seen: pet.poller.up is False, ["--no-server"])
        self.assertEqual(seen["pet"].boot.state, "wait")
        self.assertIn(P.wrap_text(P.ASLEEP)[0], " ".join(seen["texts"]))   # today's behaviour
        self.assertFalse(os.path.exists(self.info))


@unittest.skipUnless(HAVE_QT, "PySide6 not installed")
class PetProcessTest(LaunchBase):
    """`python -m moth.pet` as a real process (offscreen): signals and a hard kill must not orphan its server."""

    def start_pet(self) -> subprocess.Popen:
        for f in (self.info,):
            if os.path.exists(f):
                os.remove(f)
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        log = open(os.path.join(self.tmp, f"pet-{self.port}.log"), "w")   # noqa: SIM115  open while it runs
        self.addCleanup(log.close)
        p = subprocess.Popen([sys.executable, "-m", "moth.pet", "--url", self.url, "--model", self.model,
                              "--token-file", os.path.join(self.tmp, "no-token"), "--", "--react-every", "0"],
                             cwd=str(ROOT), env=dict(os.environ, QT_QPA_PLATFORM="offscreen"),
                             stdout=log, stderr=subprocess.STDOUT)
        self.procs.append(p)
        self.assertTrue(wait_until(lambda: len(self.pids()) == 2 and L.probe(self.url)[0] == "moth", 30),
                        f"the pet never woke its server: {self.serve_log()!r}")
        return p

    @unittest.skipUnless(os.name == "posix", "SIGTERM/SIGINT/SIGHUP to a process are POSIX (Windows has none of them)")
    def test_signals_stop_the_server(self):
        """Right after the server answers (the windows may not exist yet: the early path, exit 130) and once the pet
        is settled (quits through Qt, exit 0). Either way the server and its child go."""
        for sig, settle in ((signal.SIGTERM, 0.0), (signal.SIGTERM, 1.5), (signal.SIGINT, 1.5), (signal.SIGHUP, 1.5)):
            with self.subTest(sig=sig.name, settle=settle):
                pet = self.start_pet()
                pid, child = self.pids()
                time.sleep(settle)
                pet.send_signal(sig)
                rc = pet.wait(20)
                self.assertIn(rc, (0, 130) if not settle else (0,))
                gone = wait_until(lambda p=pid, c=child: not alive(p) and not alive(c), 5)
                self.assertTrue(gone, f"{sig.name}: the server outlived the pet")
                self.assertEqual(L.probe(self.url)[0], "none")

    def test_hard_kill_of_the_pet_takes_the_server(self):
        pet = self.start_pet()
        pid, child = self.pids()
        self.assertIn(L.EXIT_WITH_PARENT, self.fake_info()["argv"])
        pet.kill()   # no atexit, no Qt quit: only the kernel is left (everywhere: the stdin lifeline sees EOF; Linux
        pet.wait()   # also the parent-death signal; Windows also the Job Object's KILL_ON_JOB_CLOSE)
        self.assertTrue(wait_until(lambda: not alive(pid), 5), "the server outlived a hard-killed pet")
        if os.name == "nt":   # the job holds the server's children too
            self.assertTrue(wait_until(lambda: not alive(child), 5), "the server's child outlived a hard-killed pet")


# --- Qt, offscreen --------------------------------------------------------------------------------------------------
@unittest.skipUnless(HAVE_QT, "PySide6 not installed")
class QtTest(unittest.TestCase):
    def setUp(self):
        self.assertEqual(os.environ["QT_QPA_PLATFORM"], "offscreen")

    def test_petdex_sheet(self):
        from PySide6.QtGui import QImage
        d = tempfile.mkdtemp()
        pj, sheet = P.export_petdex(d)
        with open(pj) as f:
            meta = json.load(f)
        self.assertEqual(meta["spritesheetPath"], "spritesheet.png")
        self.assertEqual(set(meta), {"id", "displayName", "description", "spritesheetPath"})
        img = QImage(sheet)
        self.assertEqual((img.width(), img.height()), (1536, 1872))   # Codex rejects any other size

        def alpha_box(col, row):
            xs, ys = [], []
            for y in range(row * 208, row * 208 + 208, 2):
                for x in range(col * 192, col * 192 + 192, 2):
                    if img.pixelColor(x, y).alpha() > 16:
                        xs.append(x - col * 192)
                        ys.append(y - row * 208)
            return (min(xs), min(ys), max(xs), max(ys)) if xs else None
        for row, (_state, n) in enumerate(P.PETDEX_ROWS):
            for col in range(8):
                box = alpha_box(col, row)
                if col >= n:
                    self.assertIsNone(box, (row, col))
                else:
                    self.assertIsNotNone(box, (row, col))
                    self.assertGreater(box[2] - box[0], 60)   # fills the cell, not a speck

    def test_cached_z_glyphs_draw_the_same_pixels(self):
        """paint_z draws cached glyph outlines: pixel for pixel what building them per frame (the old code) drew."""
        from PySide6.QtCore import QPointF, Qt
        from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen
        from PySide6.QtWidgets import QApplication
        QApplication.instance() or QApplication([])

        def old_paint_z(p, cx, cy, w, h, phase):
            p.save()
            p.setRenderHint(p.RenderHint.Antialiasing, True)
            f = QFont()
            f.setFamilies(["Fraunces", "Georgia", "DejaVu Serif", "serif"])
            f.setItalic(True)
            f.setWeight(QFont.Weight.DemiBold)
            for i in range(3):
                ph = (phase * 0.45 + i / 3) % 1.0
                f.setPixelSize(max(9, int(h * (0.24 + 0.18 * ph))))
                al = max(0.0, min(1.0, ph * 5)) * min(1.0, (1 - ph) * 2.2)
                zp = QPainterPath()
                zp.addText(QPointF(cx + w * (0.06 + 0.22 * ph), cy - h * (0.36 + 0.55 * ph)), f, "z")
                c, e = QColor(P.LAMP), QColor(P.INK)
                c.setAlphaF(al)
                e.setAlphaF(al * 0.8)
                p.setPen(QPen(e, 0.8))
                p.setBrush(c)
                p.drawPath(zp)
            p.restore()

        def draw(fn, phase, w, h):
            img = QImage(160, 160, QImage.Format.Format_ARGB32_Premultiplied)
            img.fill(Qt.GlobalColor.transparent)
            p = QPainter(img)
            fn(p, 61.3, 97.7, w, h, phase)
            p.end()
            return bytes(img.constBits())
        for phase in (0.0, 0.37, 1.1, 2.9, 1234.5678):
            for w, h in ((64, 44.8), (112, 78.4), (40.5, 28.35)):
                self.assertEqual(draw(old_paint_z, phase, w, h), draw(P.paint_z, phase, w, h), (phase, w, h))

    def test_still_moth_ticks_slowly_and_skips_repaints(self):
        """Idle cost: a resting moth ticks STILL_MS apart and repaints nothing new, yet is on time for its wing flick
        (12 fps through it); dozing, the z's move at DOZE_FPS; flying is 30 fps; a click ticks at once."""
        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                b = json.dumps({"ok": True, "pid": os.getpid(), "install_root": "x", "version": "t"} if
                               self.path.startswith("/api/health") else {"events": [], "next": 1, "epoch": "e"}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)
        srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        seen = {}

        def ready(pet):
            from PySide6.QtCore import QTimer
            QTimer.singleShot(0, lambda: checks(pet))   # in the running loop: quit() before exec() would be lost

        def checks(pet):
            paints = []
            pet.win.update = lambda: paints.append(1)
            w, now = pet.w, time.monotonic()
            pet.bubbles.clear()
            w.wander, w.state, w.idle_since = False, P.REST, now
            pet.anim["shiver_next"] = now + 5
            pet._tick()
            paints.clear()
            for _ in range(3):
                pet._tick()
            seen["still"] = (pet.timer.interval(), len(paints))
            pet.anim["shiver_next"] = time.monotonic() + 0.1
            pet._tick()
            seen["before_flick"] = pet.timer.interval()
            pet.anim["shiver_next"] = time.monotonic() - 0.001
            pet._tick()
            seen["flick"] = (pet.win.pose.frame, pet.timer.interval(), len(paints))
            w.sleep(time.monotonic(), True)
            pet._tick()
            n = len(paints)
            time.sleep(0.02)
            pet._tick()
            seen["doze"] = (pet.timer.interval(), len(paints) > n)
            pet.say("zzz", ttl=5)
            b, _ = pet.bubbles.current()
            b.shown, b.ttl = time.monotonic() - 1.0, 1.0 + pet.bubbles.FADE_OUT + 0.1   # fully shown, fade in 0.1 s
            pet._tick()
            seen["doze_fade"] = pet.timer.interval()
            pet.bubbles.clear()
            w.sleep(time.monotonic(), False)
            pet.clicked()
            seen["kick"] = pet.timer.interval()
            pet.inp.hide()
            w.wander = True
            w.excite(time.monotonic(), 30)
            pet._tick()
            seen["fly"] = pet.timer.interval()
            seen["complained"] = pet.complained
            from PySide6.QtWidgets import QApplication
            seen["name"] = (QApplication.instance().applicationName(), QApplication.instance().applicationDisplayName())
            pet.quit()
        a = P.build_parser().parse_args(["--url", f"http://127.0.0.1:{srv.server_port}", "--token-file",
                                         "/nonexistent/token"])
        try:
            P.run(a, on_ready=ready)
        finally:
            srv.shutdown()
            srv.server_close()
        self.assertEqual(seen["still"], (P.STILL_MS, 0))
        self.assertTrue(int(1000 / 12) <= seen["before_flick"] <= 101, seen)
        self.assertEqual(seen["flick"][:2], (2, int(1000 / 12)))   # the flick's first frame, at the resting rate
        self.assertEqual(seen["flick"][2], 1)
        self.assertEqual(seen["doze"], (1000 // P.DOZE_FPS, True))
        self.assertEqual(seen["doze_fade"], int(1000 / 8))   # the fade-out is no later than at the old 8 fps
        self.assertEqual(seen["kick"], 0)
        self.assertEqual(seen["fly"], int(1000 / 30))
        self.assertFalse(seen["complained"])
        self.assertEqual(seen["name"], ("moth", "moth"))   # set_app_name: the Dock / menus say moth, not Python

    def test_app_name_helpers_never_raise(self):
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication([])
        got = P.macos_bundle_name("moth")   # the CFBundleName trick through ctypes; not macOS: does nothing
        self.assertIsInstance(got, bool)
        if sys.platform != "darwin":
            self.assertFalse(got)
        P.set_app_name(app, "moth")
        self.assertEqual((app.applicationName(), app.applicationDisplayName()), ("moth", "moth"))
        P.set_app_name(object())   # not a Qt app at all: silently nothing

    @unittest.skipUnless(os.name == "posix", "SIGTERM to itself: POSIX")
    def test_signal_while_qt_starts_quits_through_qt(self):
        """A SIGTERM while QApplication and the windows are being built (here: during load_art) is noted, and the
        pet quits through Qt once it exists (exit 0). It used to raise KeyboardInterrupt through half-made Qt
        objects (exit 130), and PySide6 could abort at exit over those (rc -6 on macOS)."""
        from unittest import mock
        real, seen = P.load_art, {}

        def load_art_then_sigterm():
            os.kill(os.getpid(), signal.SIGTERM)
            return real()

        def ready(pet):
            from PySide6.QtCore import QTimer

            def fallback():
                seen["fallback"] = True
                pet.quit()
            QTimer.singleShot(5000, fallback)   # never hang the suite if the deferred quit is lost
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            url = f"http://127.0.0.1:{s.getsockname()[1]}"   # nothing listens: --no-server just waits
        a = P.parse_args(["--url", url, "--no-server", "--token-file", "/nonexistent/token"])
        with mock.patch.object(P, "load_art", load_art_then_sigterm):
            rc = P.run(a, on_ready=ready)
        self.assertEqual(rc, 0)
        self.assertNotIn("fallback", seen)

    def test_windows_run_offscreen(self):
        """The real windows against a stub moth: a new line shows in the bubble and the moth gets excited."""
        lines = [[], [{"say": "Hello From The Lamp"}]]

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path.startswith("/api/health"):   # a moth answers: the pet uses it, starts nothing
                    b = json.dumps({"ok": True, "pid": os.getpid(), "install_root": "x", "version": "t"}).encode()
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(b)))
                    self.end_headers()
                    self.wfile.write(b)
                    return
                # the pet's one "what did i miss" history request (since=0) mustn't eat a scripted answer
                evs = [] if "since=0" in self.path else (lines.pop(0) if lines else [])
                b = json.dumps({"events": evs, "next": 10 + len(evs), "epoch": "e"}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)
        srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        seen = {}

        def ready(pet):
            from PySide6.QtCore import QTimer

            def check():
                pet.tick()
                b, _ = pet.bubbles.current()
                if b and not seen:
                    seen.update(lines=b.lines, excited=pet.w.excited, visible=pet.bub.isVisible() or None)
                    exercise()
                if seen or time.monotonic() > deadline:
                    t.stop()     # once: a second app.quit() stays queued and ends the next test's exec() at once
                    pet.quit()

            def exercise():
                """The other branches of the tick (it swallows errors, so check it never had to)."""
                pet.grab()
                pet.win.move(pet.win.pos().x() + 40, pet.win.pos().y())
                for _ in range(3):
                    pet._tick()
                pet.drop()
                pet.events.put(("status", "down"))
                pet._tick()
                seen["asleep"] = pet.w.asleep
                pet.events.put(("status", "up"))
                pet._tick()
                seen["woke"] = not pet.w.asleep
                pet.menu_sleep = True
                pet.w.sleep(time.monotonic(), True)
                pet.events.put(("sent", "i couldn't reach moth"))
                for _ in range(3):
                    pet._tick()
                pet.muted = True
                pet.bubbles.push("hidden")
                pet.clicked()
                pet._tick()
                pet.clicked()   # second click closes the box
                seen["box_closed"] = not pet.inp.isVisible()
                for _ in range(3):
                    pet.tick()
                seen["complained"] = pet.complained
            deadline = time.monotonic() + 8
            t = QTimer(pet.win)
            t.timeout.connect(check)
            t.start(50)
        a = P.build_parser().parse_args(["--url", f"http://127.0.0.1:{srv.server_port}", "--token-file",
                                         "/nonexistent/token"])
        try:
            P.run(a, on_ready=ready)
        finally:
            srv.shutdown()
            srv.server_close()
        self.assertEqual(seen.get("lines"), ["hello from the lamp"])
        self.assertTrue(seen.get("excited"))
        self.assertTrue(seen.get("asleep"))
        self.assertTrue(seen.get("woke"))
        self.assertTrue(seen.get("box_closed"))
        self.assertIs(seen.get("complained"), False)


# --- skins, tray menu, update bubble -------------------------------------------------------------------------------
ROW_COLORS = ["#d04040", "#40d040", "#4040d0", "#d0d040", "#d040d0", "#40d0d0", "#806040", "#408060", "#604080"]


def make_pet(folder: str, pet_id: str = "testpet", size=(1536, 1872), empty=(), sheet_name="spritesheet.png",
             meta: dict | None = None) -> str:
    """A synthetic Petdex pet (no real art): every frame of row r is a ROW_COLORS[r] block, with a white marker on
    its left side (to see mirroring). Rows in `empty` are left transparent. Needs QtGui."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor, QImage, QPainter
    os.makedirs(folder, exist_ok=True)
    img = QImage(size[0], size[1], QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    for r, (row, n) in enumerate(P.PETDEX_ROWS):
        if row in empty:
            continue
        for i in range(n):
            x0, y0 = i * 192, r * 208
            p.fillRect(x0 + 40, y0 + 30, 112, 170, QColor(ROW_COLORS[r]))
            p.fillRect(x0 + 44, y0 + 100, 20, 20, QColor("#ffffff"))   # left-side marker
    p.end()
    assert img.save(os.path.join(folder, sheet_name))
    m = {"id": pet_id, "displayName": pet_id.title(), "description": "test", "spritesheetPath": sheet_name}
    m.update(meta or {})
    with open(os.path.join(folder, "pet.json"), "w") as f:
        json.dump(m, f)
    return folder


class SkinLogicTest(unittest.TestCase):
    """Stdlib parts: where pets are looked up, pet.json checks, the row each state uses, the daily-check rule."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.dirs = [os.path.join(self.tmp, d) for d in ("petdex", "codex", "hermes")]

    def pet(self, d: int, pid: str, meta=None, sheet=True) -> str:
        f = os.path.join(self.dirs[d], pid)
        os.makedirs(f)
        with open(os.path.join(f, "pet.json"), "w") as fh:
            json.dump(meta if meta is not None else {"id": pid, "displayName": pid.upper(),
                                                     "spritesheetPath": "spritesheet.png"}, fh)
        if sheet:
            open(os.path.join(f, "spritesheet.png"), "wb").close()
        return f

    def test_pet_dirs_follow_codex_and_hermes_homes(self):
        from unittest import mock
        with mock.patch.dict(os.environ, {"CODEX_HOME": "/x/codex", "HERMES_HOME": "/y/hermes"}):
            d = P.pet_dirs()
        self.assertEqual(d[0], os.path.join(os.path.expanduser("~"), ".petdex", "pets"))
        self.assertEqual(d[1:], [os.path.join("/x/codex", "pets"), os.path.join("/y/hermes", "pets")])

    def test_lookup_order_and_folders(self):
        from unittest import mock
        a = self.pet(1, "cat")    # codex
        self.pet(2, "cat")        # hermes: codex wins
        b = self.pet(2, "dog")
        self.pet(0, "moth")       # an exported moth: the built-in one wins, and it isn't listed twice
        c = self.pet(0, "bird")
        with mock.patch.object(P, "pet_dirs", return_value=self.dirs):
            self.assertEqual(P.find_skin("cat"), a)
            self.assertEqual(P.find_skin("dog"), b)
            self.assertEqual(P.find_skin("bird"), c)
            self.assertIsNone(P.find_skin("moth"))
            self.assertIsNone(P.find_skin("nope"))
            self.assertIsNone(P.find_skin("../petdex/bird"))   # a path: only taken as a folder, and it isn't one here
            self.assertEqual(P.find_skin(b), b)                # a folder path
            self.assertIsNone(P.find_skin(self.tmp))           # a folder without pet.json
            self.assertEqual([(i, n) for i, _, n in P.installed_skins()], [("bird", "BIRD"), ("cat", "CAT"),
                                                                           ("dog", "DOG")])

    def test_read_pet_refuses_bad_folders(self):
        ok = self.pet(0, "ok")
        self.assertEqual(P.read_pet(ok), ("OK", os.path.join(os.path.realpath(ok), "spritesheet.png")))
        cases = {
            "no pet.json": None,
            "can't be read": "{not json",
            "isn't a JSON object": "[1]",
            "outside the pet's folder": {"spritesheetPath": "../ok/spritesheet.png"},
            "isn't a file in the pet's folder": {"spritesheetPath": "/etc/hostname"},
            "missing": {"spritesheetPath": "nothere.png"},
        }
        for want, meta in cases.items():
            with self.subTest(want):
                f = tempfile.mkdtemp(dir=self.tmp)
                if meta is not None:
                    with open(os.path.join(f, "pet.json"), "w") as fh:
                        fh.write(meta if isinstance(meta, str) else json.dumps(meta))
                with self.assertRaises(P.SkinError) as e:
                    P.read_pet(f)
                self.assertIn(want, str(e.exception))
        f = tempfile.mkdtemp(dir=self.tmp)   # no spritesheetPath: spritesheet.webp is found
        with open(os.path.join(f, "pet.json"), "w") as fh:
            json.dump({"id": "w"}, fh)
        open(os.path.join(f, "spritesheet.webp"), "wb").close()
        self.assertEqual(P.read_pet(f), ("w", os.path.join(os.path.realpath(f), "spritesheet.webp")))

    def test_row_for_each_state(self):
        all_rows = {r for r, _ in P.PETDEX_ROWS}
        row = P.skin_row
        self.assertEqual(row(P.FLY, True, False, 30, False, all_rows)[:2], ("running-right", False))
        self.assertEqual(row(P.FLY, True, False, -30, False, all_rows)[:2], ("running-left", False))
        self.assertEqual(row(P.LAND, True, False, -1, False, all_rows)[:2], ("running-left", False))
        self.assertEqual(row(P.REST, False, False, 0, False, all_rows)[:2], ("idle", False))
        self.assertEqual(row(P.DOZE, False, False, 0, False, all_rows), ("waiting", False, P.SKIN_FPS["waiting"]))
        self.assertEqual(row(P.FLY, True, True, 30, False, all_rows)[:2], ("jumping", False))   # excited, in the air
        self.assertEqual(row(P.REST, False, True, 0, False, all_rows)[:2], ("waving", False))   # excited, sitting
        self.assertEqual(row(P.REST, False, False, 0, True, all_rows)[:2], ("review", False))   # waiting on a reply
        self.assertEqual(row(P.HELD, True, False, 0, False, all_rows)[:2], ("waving", False))
        # a sheet with only one running direction: the other is mirrored; with neither, "running"; then idle
        self.assertEqual(row(P.FLY, True, False, -5, False, all_rows - {"running-left"})[:2], ("running-right", True))
        self.assertEqual(row(P.FLY, True, False, 5, False, all_rows - {"running-right"})[:2], ("running-left", True))
        self.assertEqual(row(P.FLY, True, False, 5, False, {"idle", "running"})[:2], ("running", False))
        self.assertEqual(row(P.FLY, True, False, 5, False, {"idle"})[:2], ("idle", False))
        # dozing without a waiting row: idle, slowed down
        self.assertEqual(row(P.DOZE, False, False, 0, False, {"idle"}), ("idle", False, P.SKIN_FPS["idle"] / 3))
        self.assertEqual(row(P.REST, False, True, 0, False, {"idle"})[:2], ("idle", False))

    def test_update_due(self):
        day = P.UPDATE_EVERY
        self.assertTrue(P.update_due(True, False, 0.0, 1e9))
        self.assertFalse(P.update_due(True, False, 1e9 - day / 2, 1e9))
        self.assertTrue(P.update_due(True, False, 1e9 - day, 1e9))
        self.assertFalse(P.update_due(False, False, 0.0, 1e9))     # update_check off
        self.assertFalse(P.update_due(True, True, 0.0, 1e9))       # offline
        self.assertTrue(P.update_due(True, False, 1e9 + 10 * day, 1e9))   # the clock went back: check anyway

    def test_check_for_update_reads_the_exit_code(self):
        from unittest import mock
        def fake(rc, out="", err=""):
            return mock.patch.object(P.subprocess, "run", return_value=subprocess.CompletedProcess([], rc, out, err))
        with fake(10, "newer code: moth 0.3.0 -> 0.4.0:  pip install -U moth-companion"):
            self.assertEqual(P.check_for_update(), ("update", P.PIP_HINT))
        from moth.backends import compat
        with fake(10, "newer code: moth 0.3.0 -> 0.4.0:  install the newest moth from ..."), \
                mock.patch.object(sys, "frozen", True, create=True):   # the installers' app: a new installer
            self.assertEqual(P.check_for_update(), ("update", compat.RELEASES))
        self.assertEqual(P.update_line(compat.RELEASES), f"there's a new me! get it at {compat.RELEASES}")
        with fake(10, "update available: moth 0.3.0 -> 0.4.0"), mock.patch.object(L.weights, "mode",
                                                                                 return_value="hub"):
            self.assertEqual(P.check_for_update(), ("update", "python -m moth update"))
        with fake(10, "newer weights on the Hub"), mock.patch.object(L.weights, "mode", return_value="pip"):
            self.assertEqual(P.check_for_update(), ("update", "moth update"))
        with fake(0, "up to date"):
            self.assertEqual(P.check_for_update(), ("none", ""))
        with fake(1, "can't reach the Hugging Face Hub (ConnectError)\n"):
            self.assertEqual(P.check_for_update(), ("error", "can't reach the Hugging Face Hub (ConnectError)"))
        with mock.patch.object(P.subprocess, "run", side_effect=subprocess.TimeoutExpired("x", 1)):
            self.assertEqual(P.check_for_update()[0], "error")
        with mock.patch.object(P.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "", "")) as run:
            P.check_for_update()
        argv = run.call_args.args[0]
        self.assertEqual(argv[1:], ["-m", "moth", "update", "--check"])   # a check only, never an update
        self.assertEqual(run.call_args.kwargs["cwd"], str(ROOT))
        self.assertIn("pip install -U moth-companion", P.update_line(P.PIP_HINT))
        q = P.BubbleQueue(clock=lambda: 0.0)   # the bubble lowercases moth's lines, never a `command`
        q.push(P.update_line(P.PIP_HINT))
        self.assertIn("`pip install -U moth-companion`", " ".join(q.waiting[0].lines))
        self.assertTrue(" ".join(q.waiting[0].lines).startswith("there's a new me!"))
        self.assertEqual(P.wrap_text("Set `HF_HUB_OFFLINE` Now"), ["set `HF_HUB_OFFLINE` now"])
        self.assertEqual(P.wrap_text("A ` Lone Tick"), ["a ` lone tick"])
        self.assertIn("run `python -m moth update`", P.update_line("python -m moth update"))


class SkinConfigTest(unittest.TestCase):
    """pet_skin / update_check in config.json: defaults, the CLI, persistence."""

    def setUp(self):
        from unittest import mock
        from moth import config as C
        self.C = C
        self.file = os.path.join(tempfile.mkdtemp(), "config.json")
        self.patch = mock.patch.dict(os.environ, {"MOTH_CONFIG_FILE": self.file})
        self.patch.start()

    def tearDown(self):
        self.patch.stop()

    def run_cli(self, *args):
        p = subprocess.run([sys.executable, "-m", "moth", "config", *args], cwd=ROOT, capture_output=True, text=True,
                           timeout=30, env=dict(os.environ))
        return p.returncode, p.stdout, p.stderr

    def test_defaults(self):
        self.assertEqual(self.C.pet_skin(), "moth")
        self.assertTrue(self.C.update_check())
        self.assertEqual(self.C.last_update_check(), 0.0)
        out = self.C.describe()
        self.assertIn("pet_skin = moth   (default)", out)
        self.assertIn("update_check = on   (default)", out)

    def test_cli_persists_pet_skin_and_update_check(self):
        code, out, err = self.run_cli("pet_skin", "nobody-installed-this")
        self.assertEqual(code, 0, err)
        self.assertIn("pet_skin = nobody-installed-this", out)
        self.assertIn("no Petdex pet 'nobody-installed-this' found", out)   # a note, still saved
        self.assertEqual(self.C.pet_skin(), "nobody-installed-this")
        self.assertEqual(self.C.lookup("pet_skin"), ("nobody-installed-this", "file"))
        self.assertEqual(self.run_cli("pet_skin", "default")[0], 0)
        self.assertEqual(self.C.pet_skin(), "moth")
        self.assertEqual(self.run_cli("update_check", "off")[0], 0)
        self.assertFalse(self.C.update_check())
        with open(self.file) as f:
            self.assertEqual(json.load(f), {"pet_skin": "moth", "update_check": False})
        folder = tempfile.mkdtemp()   # a relative folder is saved absolute: the pet may run from another cwd
        p = subprocess.run([sys.executable, "-m", "moth", "config", "pet_skin", "./mypet"], cwd=folder,
                           capture_output=True, text=True, timeout=30,
                           env=dict(os.environ, PYTHONPATH=str(ROOT)))
        self.assertEqual(p.returncode, 0, p.stderr)
        saved = self.C.pet_skin()   # absolute; Windows may spell the temp folder short (RUNNER~1): compare resolved
        self.assertTrue(os.path.isabs(saved), saved)
        self.assertEqual((os.path.realpath(os.path.dirname(saved)), os.path.basename(saved)),
                         (os.path.realpath(folder), "mypet"))
        for bad in (("pet_skin", " "), ("update_check", "maybe"), ("last_update_check", "5")):
            with self.subTest(bad):
                self.assertEqual(self.run_cli(*bad)[0], 2)

    def test_offline_and_last_check(self):
        from unittest import mock
        self.C.write(self.C.LAST_UPDATE_CHECK, 1234)
        self.assertEqual(self.C.last_update_check(), 1234.0)
        self.C.write(self.C.LAST_UPDATE_CHECK, "junk")
        self.assertEqual(self.C.last_update_check(), 0.0)
        with mock.patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}):
            self.assertTrue(self.C.offline())
        with mock.patch.dict(os.environ, {"MOTH_OFFLINE": "yes", "HF_HUB_OFFLINE": ""}):
            self.assertTrue(self.C.offline())
        with mock.patch.dict(os.environ, {"MOTH_OFFLINE": "0", "HF_HUB_OFFLINE": ""}):
            self.assertFalse(self.C.offline())


@unittest.skipUnless(HAVE_QT, "PySide6 not installed")
class SkinQtTest(unittest.TestCase):
    """Loading a synthetic Petdex sheet: frames cut per row, mirroring, drawing, and refusing bad sheets."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_loads_rows_and_draws_them(self):
        from PySide6.QtCore import QPointF  # noqa: F401
        from PySide6.QtGui import QColor, QImage, QPainter
        folder = make_pet(os.path.join(self.tmp, "testpet"), empty=("running-left",))
        skin = P.load_skin(folder)
        self.assertEqual(skin.name, "Testpet")
        self.assertEqual(skin.id, "testpet")
        self.assertEqual({r: len(f) for r, f in skin.rows.items()},
                         {r: n for r, n in P.PETDEX_ROWS if r != "running-left"})
        self.assertEqual((skin.top, skin.bottom), (30, 199))
        # each state's frame comes from its row: the colour at the cell's centre
        for state, flying, excited, vx, row_i in ((P.REST, False, False, 0, 0), (P.FLY, True, False, 10, 1),
                                                  (P.DOZE, False, False, 0, 6), (P.FLY, True, True, 10, 4),
                                                  (P.REST, False, True, 0, 3)):
            row, mirrored, _ = P.skin_row(state, flying, excited, vx, False, skin.rows)
            img = QImage(200, 200, QImage.Format.Format_ARGB32_Premultiplied)
            img.fill(0)
            p = QPainter(img)
            P.paint_skin(p, skin, 100, 100, 100, row, mirrored, 0)
            p.end()
            self.assertEqual(img.pixelColor(100, 120).name(), QColor(ROW_COLORS[row_i]).name(), (state, row))
        # running-left is empty: flying left mirrors running-right (the marker moves to the right side)
        row, mirrored, _ = P.skin_row(P.FLY, True, False, -10, False, skin.rows)
        self.assertEqual((row, mirrored), ("running-right", True))
        plain, flipped = skin.frames(row, False, 208)[0], skin.frames(row, True, 208)[0]
        self.assertEqual(plain.pixelColor(54, 110).name(), "#ffffff")
        self.assertEqual(flipped.pixelColor(191 - 54, 110).name(), "#ffffff")
        self.assertFalse(skin.icon_image(32).isNull())

    def test_bad_sheets_fall_back_to_moth(self):
        from PySide6.QtGui import QImage
        cases = {
            "1535x1872": make_pet(os.path.join(self.tmp, "narrow"), size=(1535, 1872)),
            "1536x1873": make_pet(os.path.join(self.tmp, "tall"), size=(1536, 1873)),
            "idle row is empty": make_pet(os.path.join(self.tmp, "noidle"), empty=("idle",)),
        }
        garbage = os.path.join(self.tmp, "garbage")
        os.makedirs(garbage)
        with open(os.path.join(garbage, "spritesheet.png"), "wb") as f:
            f.write(b"not a png at all" * 10)
        with open(os.path.join(garbage, "pet.json"), "w") as f:
            json.dump({"id": "garbage", "spritesheetPath": "spritesheet.png"}, f)
        cases["can't be read"] = garbage
        blank = os.path.join(self.tmp, "blank")
        os.makedirs(blank)
        img = QImage(1536, 1872, QImage.Format.Format_ARGB32_Premultiplied)
        img.fill(0)
        img.save(os.path.join(blank, "spritesheet.png"))
        with open(os.path.join(blank, "pet.json"), "w") as f:
            json.dump({"id": "blank", "spritesheetPath": "spritesheet.png"}, f)
        cases["idle row is empty "] = blank
        cases["no petdex pet 'not-installed'"] = "not-installed"
        for want, spec in cases.items():
            with self.subTest(want):
                skin, why = P.wear(spec)
                self.assertIsNone(skin)
                self.assertIn(want.strip(), why)
        self.assertEqual(P.wear("moth"), (None, None))
        self.assertEqual(P.wear(None), (None, None))
        ok = make_pet(os.path.join(self.tmp, "webp"), sheet_name="sheet.webp")   # webp, as Petdex ships them
        self.assertIsNotNone(P.wear(ok)[0])


class StubMoth:
    """A moth that answers /api/health and has nothing to say (so the pet starts no server)."""

    def __enter__(self):
        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path.startswith("/api/health"):
                    d = {"ok": True, "pid": os.getpid(), "install_root": "x", "version": "t"}
                else:
                    d = {"events": [], "next": 10, "epoch": "e"}
                b = json.dumps(d).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_port}"
        return self

    def __exit__(self, *exc):
        self.srv.shutdown()
        self.srv.server_close()


@unittest.skipUnless(HAVE_QT, "PySide6 not installed")
class PetSkinTrayUpdateTest(unittest.TestCase):
    """The real windows (offscreen) wearing a skin, the tray menu, and the update bubble (the check mocked)."""

    def setUp(self):
        from unittest import mock
        self.tmp = tempfile.mkdtemp()
        self.cfg = os.path.join(self.tmp, "config.json")
        self.patches = [mock.patch.dict(os.environ, {"MOTH_CONFIG_FILE": self.cfg}),
                        mock.patch.object(P, "pet_dirs", return_value=[os.path.join(self.tmp, "pets")])]
        for p in self.patches:
            p.start()
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("MOTH_OFFLINE", None)
        self.write_cfg({"update_check": False})

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()

    def write_cfg(self, d):
        with open(self.cfg, "w") as f:
            json.dump(d, f)

    def read_cfg(self):
        with open(self.cfg) as f:
            return json.load(f)

    def run_pet(self, script, args=(), timeout=10.0) -> dict:
        """P.run against a stub moth; script(pet, seen) is a generator stepped every 30 ms (yield = wait a step)."""
        seen: dict = {"texts": []}

        def ready(pet):
            from PySide6.QtCore import QTimer
            gen = script(pet, seen)
            deadline = time.monotonic() + timeout

            def collect():
                for b in [pet.bubbles.now_showing] + pet.bubbles.waiting:
                    if b is not None and " ".join(b.lines) not in seen["texts"]:
                        seen["texts"].append(" ".join(b.lines))

            def step():
                seen["steps"] = seen.get("steps", 0) + 1
                collect()
                try:
                    if time.monotonic() > deadline:
                        seen["timed_out"] = True
                        raise StopIteration
                    next(gen)
                    collect()
                except StopIteration:
                    collect()
                    t.stop()
                    pet.quit()
                except BaseException as e:  # noqa: BLE001  re-raised after the loop
                    seen["error"] = e
                    t.stop()
                    pet.quit()
            t = QTimer(pet.win)
            t.timeout.connect(step)
            t.start(30)
        with StubMoth() as moth:
            a = P.parse_args(["--url", moth.url, "--token-file", os.path.join(self.tmp, "no-token"), "--no-server",
                              *args])
            P.run(a, on_ready=ready)
        if "error" in seen:
            raise seen["error"]
        self.assertTrue(seen.get("steps"), "the Qt loop ended before the script ran (a stray queued app.quit()?)")
        self.assertNotIn("timed_out", seen)
        return seen

    def test_wears_a_skin_and_the_menu_switches_it(self):
        pets = os.path.join(self.tmp, "pets")
        make_pet(os.path.join(pets, "testpet"))
        make_pet(os.path.join(pets, "other"), meta={"displayName": "Another One"})

        def script(pet, seen):
            yield
            pet._tick()
            seen["skin"] = pet.skin and pet.skin.id
            seen["look"] = pet.win.look and pet.win.look[1]
            sw, sh = pet.skin.size(64)
            seen["mask"] = pet.win.mask().boundingRect()
            seen["drawn"] = (int((pet.win.width() - sw) / 2), int((pet.win.height() - sh) / 2), int(sw), int(sh))
            seen["gaps"] = pet.bubble_gaps()
            # the tray's menu (built even where there's no tray, e.g. offscreen)
            m = pet.tray_menu
            seen["tray"] = pet.tray
            seen["items"] = [x.text() for x in m.actions()]
            sub = next(x.menu() for x in m.actions() if x.text() == "skin")
            seen["skins"] = [(x.text(), x.isChecked()) for x in sub.actions()]
            # the harness switch writes config.json
            next(x for x in m.actions() if x.text() == "show moth in harnesses").trigger()
            seen["harness"] = self.read_cfg().get("show_in_harness")
            next(x for x in m.actions() if x.text() == "mute bubbles").trigger()
            pet.fill_menu(m)   # aboutToShow does this before the tray menu opens
            seen["muted"] = (pet.muted, "unmute bubbles" in [x.text() for x in m.actions()])
            next(x for x in m.actions() if x.text() == "unmute bubbles").trigger()
            # pick "Another One" from the skin menu: worn and saved
            sub = next(x.menu() for x in m.actions() if x.text() == "skin")
            next(x for x in sub.actions() if x.text().startswith("Another One")).trigger()
            seen["switched"] = (pet.skin.id, self.read_cfg().get("pet_skin"))
            # a broken choice keeps the current skin and says why
            pet.choose_skin(os.path.join(self.tmp, "nowhere"))
            seen["kept"] = pet.skin.id
            # back to moth
            pet.fill_menu(m)
            sub = next(x.menu() for x in m.actions() if x.text() == "skin")
            next(x for x in sub.actions() if x.text() == "moth").trigger()
            seen["moth"] = (pet.skin, self.read_cfg().get("pet_skin"))
            pet._tick()
            seen["moth_look"] = pet.win.look
            seen["icon"] = pet.icon().isNull()
            for state in (P.DOZE, P.FLY):   # the other branches of the tick with a skin on
                pet.set_skin(P.load_skin("testpet"))
                pet.w.state = state
                pet._tick()
                pet.win.repaint()
            seen["complained"] = pet.complained
        seen = self.run_pet(script, ["--skin", "testpet"])
        self.assertEqual(seen["skin"], "testpet")
        self.assertIn(seen["look"], ("idle", "running-right", "running-left", "jumping", "waving"))
        x, y, w, h = seen["drawn"]
        r = seen["mask"]
        self.assertTrue(r.left() <= x and r.top() <= y and r.right() >= x + w - 1 and r.bottom() >= y + h - 1,
                        (r, seen["drawn"]))   # the mask covers the taller skin cell
        up, down = seen["gaps"]
        self.assertAlmostEqual(up, 64 * P.SKIN_H / 2 - 30 * 64 * P.SKIN_H / 208 + 4, places=3)
        self.assertGreater(down, 0)
        from PySide6.QtWidgets import QSystemTrayIcon
        if not QSystemTrayIcon.isSystemTrayAvailable():
            self.assertIsNone(seen["tray"])   # no tray offscreen: skipped silently
        self.assertEqual([t for t in seen["items"] if t],
                         ["open chat page", "what did i miss", "mute bubbles", "mute harnesses", "quiet hours",
                          "sleep", "show moth in harnesses", "skin", "stay on this screen", "check for updates",
                          "quit"])
        self.assertEqual(seen["skins"], [("moth", False), ("Another One (other)", False), ("Testpet", True)])
        self.assertIs(seen["harness"], True)
        self.assertEqual(seen["muted"], (True, True))
        self.assertEqual(seen["switched"], ("other", "other"))
        self.assertEqual(seen["kept"], "other")
        self.assertTrue(any("i can't wear" in t and "nowhere" in t for t in seen["texts"]), seen["texts"])
        self.assertEqual(seen["moth"], (None, "moth"))
        self.assertIsNone(seen["moth_look"])
        self.assertFalse(seen["icon"])
        self.assertIs(seen["complained"], False)

    def test_bad_skin_in_config_means_moth_with_a_note(self):
        self.write_cfg({"update_check": False, "pet_skin": "not-installed"})

        def script(pet, seen):
            yield
            seen["skin"] = pet.skin
        seen = self.run_pet(script)
        self.assertIsNone(seen["skin"])
        self.assertTrue(any("i can't wear 'not-installed'" in t for t in seen["texts"]), seen["texts"])

    def test_update_bubble(self):
        from unittest import mock
        self.write_cfg({"update_check": True, "last_update_check": 0})
        calls = []

        def fake_check(timeout=None):
            calls.append(threading.current_thread().name)
            return "update", "python -m moth update"

        def script(pet, seen):
            yield
            pet.next_update_eval = 0.0   # don't wait the first 30 s
            while not any("new me" in t for t in seen["texts"]):
                pet._tick()
                yield
            seen["last"] = self.read_cfg().get("last_update_check")
            # a day hasn't passed: no second automatic check
            pet.next_update_eval = 0.0
            pet._tick()
            seen["calls_after"] = len(calls)
            # "check for updates" from the menu always asks, and says when there is nothing new
            P.check_for_update = lambda timeout=None: ("none", "")   # restored by the patch below
            next(x for x in pet.tray_menu.actions() if x.text() == "check for updates").trigger()
            while not any("newest me" in t for t in seen["texts"]):
                pet._tick()
                yield
            os.environ["HF_HUB_OFFLINE"] = "1"
            pet.check_updates(manual=True)
            del os.environ["HF_HUB_OFFLINE"]
        with mock.patch.object(P, "check_for_update", side_effect=fake_check):
            before = time.time()
            seen = self.run_pet(script)
        self.assertEqual(calls, ["moth-pet-update"])   # off the UI thread, once
        self.assertIn("there's a new me! run `python -m moth update`", seen["texts"])
        self.assertGreaterEqual(seen["last"], int(before))
        self.assertEqual(seen["calls_after"], 1)
        self.assertIn("i'm the newest me already", seen["texts"])
        self.assertTrue(any("offline" in t for t in seen["texts"]), seen["texts"])

    def test_a_finished_pet_never_acts_again(self):
        """Regression (test_update_bubble, flaky on Windows): a pet's QTimer outlived app.exec(), so an earlier test's
        pet kept ticking in later tests' event loops, and its own first daily check (30 s after it started) wrote
        last_update_check into the running test's config: that test's pet then saw no check due."""
        from unittest import mock
        pets = []

        def script(pet, seen):
            pets.append(pet)
            yield
        self.run_pet(script)
        old = pets[0]
        self.assertFalse(old.timer.isActive())
        self.write_cfg({"update_check": True, "last_update_check": 0})
        old.next_update_eval = 0.0
        with mock.patch.object(P, "check_for_update", side_effect=AssertionError("an old pet checked")) as chk:
            old.tick()
        chk.assert_not_called()
        self.assertEqual(self.read_cfg(), {"update_check": True, "last_update_check": 0})

    def test_no_check_when_off_or_offline(self):
        from unittest import mock
        for cfg, env in (({"update_check": False}, {}), ({"update_check": True}, {"HF_HUB_OFFLINE": "1"}),
                         ({"update_check": True}, {"MOTH_OFFLINE": "1"})):
            with self.subTest(cfg=cfg, env=env):
                self.write_cfg(cfg)

                def script(pet, seen):
                    yield
                    pet.next_update_eval = 0.0
                    pet._tick()
                    for _ in range(5):
                        yield
                    seen["running"] = pet.update_running
                with mock.patch.object(P, "check_for_update", side_effect=AssertionError("checked")) as chk, \
                        mock.patch.dict(os.environ, env):
                    seen = self.run_pet(script)
                chk.assert_not_called()
                self.assertFalse(seen["running"])
                self.assertNotIn("last_update_check", self.read_cfg())


# --- quality of life: fullscreen, done-signal, missed list, mutes, quiet hours, screens -----------------------------
class QolLogicTest(unittest.TestCase):
    """The stdlib parts: quiet-hours parsing, the missed list, mute filtering, screen picking, fullscreen decisions
    from (mocked) window-system snapshots, and the done-hop's animation."""

    def test_quiet_hours(self):
        from moth import config as C
        self.assertEqual(C.quiet_span("23:00-08:00"), (23 * 60, 8 * 60))
        self.assertEqual(C.quiet_span(" 9:05 - 17:30 "), (9 * 60 + 5, 17 * 60 + 30))
        for bad in ("off", "", "23:00", "24:00-08:00", "23:60-08:00", "10:00-10:00", "a:b-c:d", "1:2:3-4:5", None, 7):
            self.assertIsNone(C.quiet_span(bad), bad)
        self.assertEqual(C.parse_quiet("23:00-8:00"), "23:00-08:00")
        self.assertEqual(C.parse_quiet("OFF"), "off")
        self.assertIsNone(C.parse_quiet("whenever"))
        night = "23:00-08:00"   # over midnight: 23:00 up to, not including, 08:00
        for minute, want in ((22 * 60 + 59, False), (23 * 60, True), (0, True), (3 * 60, True), (7 * 60 + 59, True),
                             (8 * 60, False), (12 * 60, False)):
            self.assertEqual(P.in_quiet(night, minute), want, minute)
        day = "09:00-17:30"
        self.assertEqual([P.in_quiet(day, m) for m in (8 * 60 + 59, 9 * 60, 17 * 60 + 29, 17 * 60 + 30)],
                         [False, True, True, False])
        self.assertFalse(P.in_quiet("off", 0))
        self.assertFalse(P.in_quiet("nonsense", 0))
        self.assertIsInstance(P.in_quiet(night), bool)   # now, local time

    def test_new_config_keys(self):
        from unittest import mock

        from moth import config as C
        tmp = tempfile.mkdtemp()
        with mock.patch.dict(os.environ, {"MOTH_CONFIG_FILE": os.path.join(tmp, "c.json")}):
            self.assertEqual([C.lookup(k)[0] for k in C.PET_LIVE], [True, False, [], "any", "off"])
            C.muted_sources().append("x")   # a copy: the default stays empty
            self.assertEqual(C.muted_sources(), [])
            self.assertEqual(C.parse_sources("Codex, pi,codex"), ["codex", "pi"])
            self.assertEqual(C.parse_sources("none"), [])
            self.assertIsNone(C.parse_sources("codex,nope"))
            for argv, rc in ((["muted_sources", "pi,dsh"], 0), (["muted_sources", "bogus"], 2),
                             (["quiet_hours", "22:30-6:00"], 0), (["quiet_hours", "late"], 2),
                             (["pet_screen", "HDMI-1"], 0), (["chirp", "on"], 0), (["hide_in_fullscreen", "nah"], 2)):
                with mock.patch("sys.stdout"), mock.patch("sys.stderr"):
                    self.assertEqual(C.main(argv), rc, argv)
            self.assertEqual((C.muted_sources(), C.quiet_hours(), C.pet_screen(), C.chirp(), C.hide_in_fullscreen()),
                             (["dsh", "pi"], "22:30-06:00", "HDMI-1", True, True))
            with mock.patch("sys.stdout"):
                C.main(["pet_screen", "any"])
            self.assertEqual(C.pet_screen(), "any")
            text = C.describe()
            self.assertIn("muted_sources = dsh,pi", text)
            self.assertIn("quiet_hours = 22:30-06:00", text)

    def test_mute_filter(self):
        self.assertTrue(P.source_muted("codex", ["codex", "pi"]))
        self.assertFalse(P.source_muted("hermes", ["codex"]))
        self.assertFalse(P.source_muted(None, ["codex"]))   # unknown source (no token): never muted
        self.assertFalse(P.source_muted("codex", None))

    def test_missed_list(self):
        clock = Clock()
        clock.t = 10_000.0
        m = P.Missed(keep=3, clock=clock)
        self.assertEqual(m.rows(), [])
        m.add("one", "codex", "api")
        clock.t += 60
        m.add("two", "claude-code", "moth-repo", why="hidden")
        clock.t += 120
        m.add("three", None, None, why="muted")
        m.add("four", "hermes", None, why="asleep")
        clock.t += 240
        self.assertEqual(m.rows(), [("hermes · 4 min ago", "four", "while asleep"),
                                    ("4 min ago", "three", "while muted"),
                                    ("claude · moth-repo · 6 min ago", "two", "while hidden")])   # "one" fell off
        m.seed([P.Line("from the log", None, "pi", "web")])
        self.assertEqual([r[1] for r in m.rows()], ["four", "three", "two"])   # full: the history doesn't fit
        m2 = P.Missed(clock=clock)
        m2.add("new")
        m2.seed([P.Line("old a", None, "pi", "web"), P.Line("old b")])
        self.assertEqual([r[:2] for r in m2.rows()], [("just now", "new"), ("earlier", "old b"),
                                                       ("pi · web · earlier", "old a")])
        self.assertEqual([P.ago(s) for s in (None, 0, 44, 50, 61, 3599, 3600, 7300, 90000, -5)],
                         ["earlier", "just now", "just now", "1 min ago", "1 min ago", "60 min ago", "1 h ago",
                          "2 h ago", "1 d ago", "just now"])

    def test_pick_bounds_and_staying_on_a_screen(self):
        A, B = (0, 0, 1920, 1040), (1920, 0, 1280, 984)   # side by side, B shorter: a gap under it
        screens = [("A", A), ("B", B)]
        self.assertEqual(P.pick_bounds("B", screens, 100, 100), (B, "B"))        # pinned
        self.assertEqual(P.pick_bounds("gone", screens, 100, 100), (A, "A"))     # unplugged: where it is
        self.assertEqual(P.pick_bounds("any", screens, 2000, 50), (B, "B"))
        self.assertEqual(P.pick_bounds(None, screens, 2500, 1010), (B, "B"))     # in the gap: nearest, not primary
        self.assertEqual(P.pick_bounds(None, [], 0, 0), (None, None))
        w = P.Wander(A, size=56, rng=random.Random(4))
        w.move_to(B, 0.0)
        self.assertEqual(w.state, P.REST)
        self.assertTrue(B[0] <= w.x < B[0] + B[2] and B[1] <= w.y < B[1] + B[3])
        w.cursor_seen = -1e9
        t = 0.0
        for _ in range(int(120 * 30)):   # two minutes of flight with a cursor on screen A pulling at it
            t += 1 / 30
            w.step(1 / 30, t, cursor=(300, 300) if int(t) % 3 else (310, 300), bounds=B)
            self.assertTrue(B[0] <= w.x <= B[0] + B[2] and B[1] <= w.y <= B[1] + B[3], (t, w.x, w.y))

    def test_fullscreen_decision(self):
        on = {"rect": (0, 0, 1920, 1080), "states": {"fullscreen"}, "types": {"normal"}, "managed": True,
              "pt": (1800, 1000), "screen": (1920, 1080)}
        self.assertTrue(P.fullscreen_on(on))
        self.assertFalse(P.fullscreen_on({**on, "pt": (2500, 500)}))       # fullscreen on the other screen
        self.assertFalse(P.fullscreen_on({**on, "types": {"desktop"}}))     # the desktop window is screen-sized
        self.assertFalse(P.fullscreen_on({**on, "types": {"dock"}}))
        self.assertFalse(P.fullscreen_on(None))
        self.assertFalse(P.fullscreen_on({}))
        self.assertFalse(P.fullscreen_on({**on, "pt": None}))
        plain = {**on, "states": set()}
        self.assertTrue(P.fullscreen_on(plain))                              # borderless, covers the screen
        self.assertFalse(P.fullscreen_on({**plain, "managed": False}))       # XWayland placeholder / not an app
        self.assertFalse(P.fullscreen_on({**plain, "states": {"maximized_vert", "maximized_horz"}}))
        self.assertFalse(P.fullscreen_on({**plain, "rect": (0, 30, 1920, 1050)}))   # a big window, not the screen
        self.assertFalse(P.fullscreen_on({**on, "rect": (10, 10, 800, 600), "pt": (1800, 1000)}))
        # HiDPI: the same window in device pixels, the pet's point and screen scaled to match
        self.assertTrue(P.fullscreen_on({"rect": (0, 0, 3840, 2160), "states": set(), "managed": True,
                                         "pt": (3600, 2000), "screen": (3840, 2160)}))

    def test_probe_choice(self):
        self.assertIsNone(P.make_probe("offscreen"))
        self.assertIsNone(P.make_probe("wayland"))   # native Wayland: can't ask, never hides
        from unittest import mock
        with mock.patch.object(P, "X11Probe", side_effect=OSError("no display")):
            self.assertIsNone(P.make_probe("xcb"))   # a broken X: fails open too

    def test_doctor_says_where_fullscreen_hiding_works(self):
        from moth import doctor as D
        r = D.Report()
        for env, plat, on in (({"DISPLAY": ":0"}, "linux", True), ({"WAYLAND_DISPLAY": "w", "DISPLAY": ":0"}, "linux", True),
                              ({"WAYLAND_DISPLAY": "w"}, "linux", True), ({}, "win32", True), ({}, "darwin", True),
                              ({"DISPLAY": ":0"}, "linux", False)):
            D.check_fullscreen(r, env, plat, on)
        self.assertEqual([m for m, *_ in r.lines], ["✓", "·", "·", "✓", "✓", "·"])
        self.assertIn("never hides", r.lines[2][2])
        self.assertIn("not for native-Wayland", r.lines[1][2])
        self.assertIn("hide_in_fullscreen on", r.lines[5][2])

    def test_done_hop(self):
        self.assertIsNone(P.hop(-0.01))
        self.assertIsNone(P.hop(P.HOP_S))
        self.assertEqual(P.hop(0.0), (0.0, P.OPEN))
        lifts = [P.hop(i / 100)[0] for i in range(int(P.HOP_S * 100))]
        first, second = max(lifts[:50]), max(lifts[50:])
        self.assertAlmostEqual(first, 0.34, places=2)
        self.assertTrue(0.08 < second < first, (first, second))   # a big hop, then a small bounce
        self.assertLess(lifts[50], 0.01)                           # touches down in between
        air = {P.hop(i / 100)[1] for i in range(5, 45)}
        self.assertGreaterEqual(len(air), 3)                       # the wings flutter in the air
        self.assertTrue(air <= set(P.FLAP))
        self.assertLess(first + 0.035 + P.ART_ASPECT / 2, 1.75 / 2)   # stays inside the 1.75-size window

    def test_chirp_never_raises(self):
        from unittest import mock
        self.assertTrue(os.path.isfile(P.CHIRP_WAV))
        import wave
        with wave.open(P.CHIRP_WAV) as wv:
            self.assertLess(wv.getnframes() / wv.getframerate(), 1.0)   # a short chirp
        c = P.Chirp(os.path.join(tempfile.mkdtemp(), "missing.wav"))
        self.assertFalse(c.play())
        c = P.Chirp()
        c.how = ""   # no player found
        self.assertFalse(c.play())
        c.how = "/bin/true"
        with mock.patch.object(P.subprocess, "Popen") as po:
            po.return_value.poll.return_value = None   # still playing
            self.assertTrue(c.play())
            self.assertFalse(c.play())                 # no pile-up
        self.assertEqual(po.call_count, 1)
        self.assertEqual(po.call_args[0][0], ["/bin/true", P.CHIRP_WAV])
        c.how, c.proc = "/usr/bin/aplay", None
        with mock.patch.object(P.subprocess, "Popen", side_effect=OSError("gone")):
            self.assertFalse(c.play())


class FakeProbe:
    """A window system whose active window is fullscreen (over the pet's point) or not; counts its snapshots."""

    def __init__(self):
        self.full, self.calls = False, 0

    def snapshot(self, ctx):
        self.calls += 1
        if not self.full:
            return None
        x, y = ctx["pt"]
        return {"rect": (x - 50, y - 50, 100, 100), "states": {"fullscreen"}, "types": set(), "managed": True,
                "pt": ctx["pt"], "screen": ctx["screen"]}


@unittest.skipUnless(HAVE_QT, "PySide6 not installed")
class PetQolTest(unittest.TestCase):
    """The real windows (offscreen): the done-signal, mutes, quiet hours, fullscreen hiding, the missed popup and the
    new menu entries."""

    setUp = PetSkinTrayUpdateTest.setUp
    tearDown = PetSkinTrayUpdateTest.tearDown
    write_cfg = PetSkinTrayUpdateTest.write_cfg
    read_cfg = PetSkinTrayUpdateTest.read_cfg
    run_pet = PetSkinTrayUpdateTest.run_pet

    @staticmethod
    def line(text, source="claude-code", label="moth-repo", reply=False):
        return P.Line(text, P.caption_for({"source": source, "label": label}) if source else None, source, label,
                      reply)

    def test_done_signal_mutes_quiet_and_fullscreen(self):
        from unittest import mock
        chirps = []

        def script(pet, seen):
            yield
            pet.chirper.play = lambda: chirps.append(1) or True
            w = pet.w
            w.wander, w.state, w.idle_since = False, P.REST, time.monotonic()
            pet.bubbles.clear()
            # 1. a reaction: bubble, hop (wings fluttering, lifted, 30 fps), no chirp (off by default)
            pet.events.put(("lines", [self.line("fixed it!")]))
            pet._tick()
            seen["bubble"] = [b.lines for b in [pet.bubbles.now_showing] + pet.bubbles.waiting if b]
            t0 = time.monotonic()
            frames, dys = set(), []
            while time.monotonic() - t0 < 0.45:
                pet._tick()
                frames.add(pet.win.pose.frame)
                dys.append(pet.win.dy)
                yield
            seen["hop"] = (len(frames), min(dys), pet.timer.interval())
            while time.monotonic() - pet.hop_at < P.HOP_S + 0.05:
                yield
            pet._tick()
            seen["after_hop"] = (pet.hop_dy, pet.timer.interval() != int(1000 / 30) or w.flying())
            seen["chirps_off"] = len(chirps)
            # 2. chirp on, and a muted harness: no bubble, no hop, no chirp, still in the list
            pet.save_setting("chirp", True)
            pet.save_setting("muted_sources", ["codex"])
            pet.bubbles.clear()
            hop_at = pet.hop_at
            pet.events.put(("lines", [self.line("codex did it", "codex", "api")]))
            pet._tick()
            seen["muted_src"] = (pet.bubbles.now_showing, pet.hop_at == hop_at, len(chirps))
            pet.events.put(("lines", [self.line("claude again")]))
            pet._tick()
            seen["chirp_on"] = len(chirps)
            # 3. quiet hours: dozes, no bubble, no chirp; your own chat still answered; ends: wakes
            pet.bubbles.clear()
            pet.set_quiet(True, time.monotonic())
            seen["quiet_asleep"] = w.asleep
            pet.events.put(("lines", [self.line("night owl"), self.line("re: you", None, None, reply=True)]))
            pet._tick()
            seen["quiet"] = ([b.lines for b in [pet.bubbles.now_showing] + pet.bubbles.waiting if b], len(chirps))
            pet.set_quiet(False, time.monotonic())
            seen["quiet_over"] = w.asleep
            pet.menu_sleep = True                      # quiet hours ending never wakes a menu-slept moth
            pet.set_quiet(True, time.monotonic())
            pet.set_quiet(False, time.monotonic())
            seen["menu_sleep_kept"] = w.asleep
            pet.menu_sleep = False
            pet.apply_sleep(time.monotonic())
            # 4. fullscreen: hidden, slow ticks, no bubbles; asked once per FS_EVERY; back when it ends
            pet.bubbles.clear()
            fake = pet.probe = FakeProbe()
            fake.full, pet.next_fs = True, 0.0
            pet._tick()
            seen["hidden"] = (pet.hidden, pet.win.isVisible(), pet.timer.interval())
            for _ in range(5):
                pet._tick()
            seen["probe_calls"] = fake.calls
            pet.events.put(("lines", [self.line("while you game")]))
            pet._tick()
            seen["hidden_bubble"] = pet.bubbles.now_showing
            pet.say("zzz… (moth server asleep)")          # not saved up for later either
            seen["hidden_queue"] = len(pet.bubbles.waiting)
            fake.full, pet.next_fs = False, 0.0
            pet._tick()
            seen["back"] = (pet.hidden, pet.win.isVisible())
            pet.save_setting("hide_in_fullscreen", False)   # off: never hides
            fake.full, pet.next_fs = True, 0.0
            pet._tick()
            seen["off"] = pet.hidden
            seen["rows"] = pet.missed.rows()
            seen["complained"] = pet.complained
        with mock.patch.object(P, "make_probe", return_value=None):
            seen = self.run_pet(script, timeout=15)
        self.assertEqual(seen["bubble"], [["fixed it!"]])
        n, lowest, iv = seen["hop"]
        self.assertGreaterEqual(n, 3)                       # wing frames change through the hop
        self.assertLess(lowest, -0.2 * 64)                  # lifted (size 64)
        self.assertEqual(iv, int(1000 / 30))
        self.assertEqual(seen["after_hop"][0], 0.0)
        self.assertEqual(seen["chirps_off"], 0)
        self.assertEqual(seen["muted_src"], (None, True, 0))
        self.assertEqual(seen["chirp_on"], 1)
        self.assertTrue(seen["quiet_asleep"])
        self.assertEqual(seen["quiet"], ([["re: you"]], 1))
        self.assertFalse(seen["quiet_over"])
        self.assertTrue(seen["menu_sleep_kept"])
        self.assertEqual(seen["hidden"], (True, False, 1000))   # Pet.HIDDEN_MS
        self.assertEqual(seen["probe_calls"], 1)
        self.assertIsNone(seen["hidden_bubble"])
        self.assertEqual(seen["hidden_queue"], 0)
        self.assertEqual(seen["back"], (False, True))
        self.assertFalse(seen["off"])
        self.assertFalse(seen["complained"])
        rows = seen["rows"]
        self.assertEqual([(r[1], r[2]) for r in rows[:5]],
                         [("while you game", "while hidden"), ("night owl", "while asleep"), ("claude again", None),
                          ("codex did it", "while muted"), ("fixed it!", None)])
        self.assertEqual(rows[3][0], "codex · api · just now")
        self.assertNotIn("re: you", [r[1] for r in rows])   # answers to you aren't "missed"

    def test_menu_entries_and_missed_popup(self):
        def script(pet, seen):
            yield
            m = pet.tray_menu
            pet.fill_menu(m)
            texts = [a.text() for a in m.actions()]
            seen["texts"] = texts
            mutes = next(a.menu() for a in m.actions() if a.text() == "mute harnesses")
            seen["mutes"] = [(a.text(), a.isCheckable(), a.isChecked()) for a in mutes.actions()]
            next(a for a in mutes.actions() if a.text() == "codex").trigger()
            seen["cfg_mute"] = self.read_cfg().get("muted_sources")
            pet.fill_menu(m)
            mutes = next(a.menu() for a in m.actions() if a.text() == "mute harnesses")
            seen["codex_checked"] = next(a for a in mutes.actions() if a.text() == "codex").isChecked()
            quiet = next(a.menu() for a in m.actions() if a.text() == "quiet hours")
            seen["presets"] = [a.text() for a in quiet.actions() if a.isCheckable()]
            next(a for a in quiet.actions() if a.text() == "23:00 – 08:00").trigger()
            seen["cfg_quiet"] = self.read_cfg().get("quiet_hours")
            pin = next(a for a in m.actions() if a.text() == "stay on this screen")
            pin.trigger()
            seen["cfg_pin"] = self.read_cfg().get("pet_screen")
            pet.fill_menu(m)
            seen["pin_checked"] = next(a for a in m.actions() if a.text() == "stay on this screen").isChecked()
            next(a for a in m.actions() if a.text() == "stay on this screen").trigger()
            seen["cfg_unpin"] = self.read_cfg().get("pet_screen")
            pet.missed.add("ooh, a parser!", "claude-code", "moth-repo")
            pet.missed.add("tests pass", "codex", "api", why="hidden")
            next(a for a in m.actions() if a.text() == "what did i miss").trigger()
            for _ in range(3):
                yield
            mw = pet.missed_win
            seen["popup"] = (mw is not None and mw.isVisible(), [e[0] for e in mw.entries], mw.entries[0][1])
            g, sg = mw.geometry(), pet.win.screen().availableGeometry()
            seen["on_screen"] = sg.contains(g)
            mw.hide()
            pet.set_hidden(True, time.monotonic())   # from the tray while hidden: at the cursor, still on screen
            pet.show_missed()
            seen["popup_hidden"] = mw.isVisible() and sg.contains(mw.geometry())
            mw.hide()
            pet.set_hidden(False, time.monotonic())
            seen["complained"] = pet.complained
        seen = self.run_pet(script)
        t = seen["texts"]
        for name in ("open chat page", "what did i miss", "mute bubbles", "mute harnesses", "quiet hours", "sleep",
                     "show moth in harnesses", "skin", "stay on this screen", "check for updates", "quit"):
            self.assertIn(name, t)
        self.assertEqual([x[0] for x in seen["mutes"]], ["claude", "codex", "hermes", "pi", "opencode",
                                                         "deepseek (dsh)"])
        self.assertTrue(all(c and not on for _, c, on in seen["mutes"]))
        self.assertEqual(seen["cfg_mute"], ["codex"])
        self.assertTrue(seen["codex_checked"])
        self.assertEqual(seen["presets"], ["off", "22:00 – 07:00", "23:00 – 08:00", "00:00 – 09:00"])
        self.assertEqual(seen["cfg_quiet"], "23:00-08:00")
        self.assertNotIn(seen["cfg_pin"], (None, "any"))   # the screen's name (offscreen has none: "0,0 800x600")
        self.assertTrue(seen["pin_checked"])
        self.assertEqual(seen["cfg_unpin"], "any")
        self.assertEqual(seen["popup"], (True, ["codex · api · just now", "claude · moth-repo · just now"],
                                         "while hidden"))
        self.assertTrue(seen["on_screen"])
        self.assertTrue(seen["popup_hidden"])
        self.assertFalse(seen["complained"])

    def test_skin_hops_with_its_jumping_row(self):
        pets = os.path.join(self.tmp, "pets")
        make_pet(os.path.join(pets, "jumper"))
        make_pet(os.path.join(pets, "nojump"), empty=("jumping", "waving"))

        def script(pet, seen):
            yield
            for sid in ("jumper", "nojump"):
                pet.choose_skin(sid)
                pet.w.wander, pet.w.state, pet.w.excited_until = False, P.REST, -1.0   # sitting, calm
                pet.done_signal(time.monotonic())
                time.sleep(0.2)
                pet._tick()
                seen[sid] = (pet.win.look[1], pet.win.dy)
                while time.monotonic() - pet.hop_at < P.HOP_S + 0.05:
                    yield
        seen = self.run_pet(script)
        self.assertEqual(seen["jumper"], ("jumping", 0.0))   # its own jump: no extra lift
        self.assertEqual(seen["nojump"][0], "idle")
        self.assertLess(seen["nojump"][1], 0)                # no jump / wave row: lifted instead


if __name__ == "__main__":
    unittest.main()
