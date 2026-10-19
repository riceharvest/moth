"""`moth update` end to end against a fake Hugging Face Hub (tests/fake_hub.py), in real subprocesses.

  python -m pytest -q tests/test_update_e2e.py     (from ~/moth; about a minute, temp dirs only, never the real Hub)

huggingface_hub runs for real (HF_ENDPOINT points at the fake): model_info, the tree listing, HEAD/GET resolve, the
LFS redirect to a "CDN", resumed ranges. Two releases of the repo are built from this checkout: 0.2.9 (an extra file
that 0.3.0 drops, an older persona.py) and 0.3.0 (a changed code file, an added doc, changed weights, a changed
integration file deep in the tree). The weights are a tiny random model (a few MB) when torch and the tokenizer are
here, so a real `python -m moth` serves /api/chat and /api/agent from the updated folder; otherwise a few hundred
random bytes, and the server steps skip.

Covers both layouts: a Hugging Face download (installed with snapshot_download like `hf download`, run as
`cd code && python -m moth update`) in a deep folder (Windows long paths), and a pip install (the package on
PYTHONPATH, weights in $MOTH_CACHE). Scenarios: up to date, --check, refusing while a real server from that folder
runs, interrupted downloads (killed mid-transfer, the Hub dropping the connection), a process killed halfway through
the swap, a file that can't be replaced (open on Windows, read-only elsewhere), your added and edited files, offline
switches and an unreachable Hub, --rollback (twice), --revision, the pet's daily check and --version.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
from fake_hub import FakeHub  # noqa: E402

try:
    import huggingface_hub  # noqa: F401
    HAVE_HUB = True
except ImportError:
    HAVE_HUB = False

REPO = "darioooooo0o/moth-112m"
TOKENIZER = ROOT / "data" / "tiny-agent-112m" / "tokenizer.json"
OLD, NEW = "0.2.9", "0.3.0"
DEEP = "code/moth/integrations/claude_code_mod/.claude-plugin/plugin.json"   # the longest path in the repo
SKIP_DIRS = {"__pycache__", "candidates", "node_modules", ".pytest_cache"}
SKIP_FILES = {"agent_pool.json", "bible_lines_r3.json", ".DS_Store"}
CRASH_HOOK = '''import os
_real, _n = os.replace, [0]
def _replace(src, dst, *a, **k):
    if ".moth-swap" in str(src) or ".moth-swap" in str(dst):
        _n[0] += 1
        if _n[0] > int(os.environ["MOTH_TEST_CRASH_AFTER"]):
            os._exit(9)   # dies mid-swap: no finally, no undo (like a kill or a power cut)
    return _real(src, dst, *a, **k)
os.replace = _replace
'''


def package_files() -> dict:
    """{repo path under code/: bytes} for moth/ and tiny_agent/ as this checkout has them (what code/ ships)."""
    out = {}
    for pkg in ("moth", "tiny_agent"):
        for d, dirs, files in os.walk(ROOT / pkg):
            dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS)
            for f in sorted(files):
                if f in SKIP_FILES or f.endswith(".pyc"):
                    continue
                p = Path(d, f)
                out[p.relative_to(ROOT).as_posix()] = p.read_bytes()
    return out


def can_serve() -> bool:
    if not TOKENIZER.is_file():
        return False
    try:
        import safetensors  # noqa: F401
        import tokenizers  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def tiny_model(seed: int) -> tuple[bytes, bytes]:
    """A real (random, 4 MB) model in moth's format: model.safetensors + config.json."""
    import torch
    from safetensors.torch import save
    sys.path.insert(0, str(ROOT))
    from tiny_agent.model import ModelConfig, TinyAgentLM
    cfg = dict(vocab_size=32768, d_model=32, n_layers=2, n_heads=2, head_dim=16, n_kv_heads=1, kv_group=2,
               swa_window=128, ffn_mult=2.0, rope_base=10000.0, max_seq_len=8192, logit_softcap=15.0, engram_layers=[],
               engram_orders=[2, 3], engram_heads=1, engram_head_dim=16, engram_rows_per_head=64)
    torch.manual_seed(seed)
    m = TinyAgentLM(ModelConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg.items()}))
    sd = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v).contiguous() for k, v in m.state_dict().items()}
    return save(sd), json.dumps(cfg).encode()


def releases(serve: bool) -> tuple[dict, dict]:
    code = {"code/" + k: v for k, v in package_files().items()}
    if serve:
        (w_old, cfg), (w_new, _) = tiny_model(1), tiny_model(2)
        tok = TOKENIZER.read_bytes()
    else:
        rnd = random.Random(1)
        w_old, w_new = bytes(rnd.getrandbits(8) for _ in range(300_000)), bytes(rnd.getrandbits(8) for _ in range(300_000))
        cfg, tok = b'{"vocab_size": 64}', b"{}"   # (a config moth/backends/compat.py accepts)
    base = {"config.json": cfg, "tokenizer.json": tok, "LICENSE": b"Apache-2.0\n",
            "code/requirements.txt": b"torch>=2.5\ntokenizers\nsafetensors\nnumpy\nhuggingface_hub\n", **code}
    old = dict(base, **{"model.safetensors": w_old, "README.md": b"# moth\n", "code/moth/VERSION": OLD.encode() + b"\n",
                        "code/moth/old_helper.py": b'"""a helper 0.3.0 drops"""\n',
                        "code/moth/persona.py": code["code/moth/persona.py"] + b"\n# 0.2.9\n"})
    new = dict(base, **{"model.safetensors": w_new, "code/moth/VERSION": NEW.encode() + b"\n",
                        "README.md": b"# moth\n\n## Changelog\n- 0.3.0: kinder replies\n\n## Files\nx\n",
                        "code/moth/docs/NEW_IN_030.md": b"new in 0.3.0\n", DEEP: code[DEEP] + b"\n"})
    return old, new


def later_releases(old: dict) -> tuple[dict, dict]:
    """(docs, future) on top of the old release: one that changes only the card and a doc (no new moth), and one
    whose model this code can't run (a config setting it doesn't know, a moth.json asking for moth 9.0.0)."""
    docs = dict(old, **{"README.md": b"# moth\n\nnow with a better card\n", "LICENSE": b"Apache-2.0 (reworded)\n",
                        "code/moth/docs/MODEL_BACKENDS.md": old["code/moth/docs/MODEL_BACKENDS.md"] + b"\nmore\n"})
    cfg = dict(json.loads(old["config.json"]), n_experts=8)
    future = dict(old, **{"config.json": json.dumps(cfg).encode(), "moth.json": b'{"needs_moth": "9.0.0"}',
                          "code/moth/VERSION": b"9.0.0\n"})
    return docs, future


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def digest(files: dict) -> dict:
    return {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}


def tree(root: Path) -> dict:
    """{repo path: sha256} of the install, without moth update's and huggingface_hub's own files."""
    out = {}
    for d, dirs, files in os.walk(root):
        rel_d = Path(d).relative_to(root).as_posix()
        dirs[:] = [x for x in dirs if x != "__pycache__" and not (rel_d == "." and x.startswith((".moth", ".cache")))]
        for f in files:
            rel = (Path(d, f).relative_to(root)).as_posix()
            if not rel.startswith((".moth", ".cache")):
                out[rel] = hashlib.sha256(Path(d, f).read_bytes()).hexdigest()
    return out


@unittest.skipUnless(HAVE_HUB, "needs huggingface_hub")
class E2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.serve = can_serve()
        cls.old, cls.new = releases(cls.serve)
        cls.hub = FakeHub(REPO)
        cls.sha_old = cls.hub.add_commit("old", cls.old)
        cls.sha_new = cls.hub.add_commit("new", cls.new)
        cls.docs, cls.future = later_releases(cls.old)
        cls.sha_docs = cls.hub.add_commit("docs", cls.docs)
        cls.sha_future = cls.hub.add_commit("future", cls.future)
        cls.hub.set_ref("v" + OLD, "old")
        cls.hub.set_ref("v" + NEW, "new")
        cls.hub.start()

    @classmethod
    def tearDownClass(cls):
        cls.hub.stop()

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="moth-e2e-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.home = self.tmp / "home"
        self.home.mkdir()
        self.hub.set_ref("main", "old")
        self.hub.drop.clear()
        self.hub.bps.clear()
        self.hub.fail.clear()
        self.procs = []
        self.addCleanup(self.stop_all)

    # -- helpers ------------------------------------------------------------------------------------------------
    def env(self, **extra) -> dict:
        e = {k: v for k, v in os.environ.items()
             if not k.startswith(("HF_", "HUGGING", "MOTH_", "PYTHON")) and k.upper() not in ("HTTP_PROXY", "HTTPS_PROXY")}
        e.update(HOME=str(self.home), USERPROFILE=str(self.home), HF_HOME=str(self.tmp / "hf-home"),
                 MOTH_CACHE=str(self.tmp / "cache"), HF_ENDPOINT=self.hub.endpoint, HF_HUB_DISABLE_TELEMETRY="1",
                 HF_HUB_DISABLE_PROGRESS_BARS="1", HF_HUB_ETAG_TIMEOUT="5", PYTHONIOENCODING="utf-8",
                 PYTHONDONTWRITEBYTECODE="1", NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost",
                 QT_QPA_PLATFORM="offscreen")
        e.update({k: v for k, v in extra.items() if v is not None})
        for k, v in extra.items():
            if v is None:
                e.pop(k, None)
        return e

    def moth(self, *args, cwd, timeout=180, **env) -> tuple[int, str]:
        p = subprocess.run([sys.executable, "-m", "moth", *args], cwd=cwd, env=self.env(**env), capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=timeout, stdin=subprocess.DEVNULL)
        return p.returncode, p.stdout + p.stderr

    def py(self, code: str, cwd, **env) -> str:
        p = subprocess.run([sys.executable, "-c", code], cwd=cwd, env=self.env(**env), capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=180, stdin=subprocess.DEVNULL)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        return p.stdout

    def expect(self, root: Path, release: dict, extra: dict | None = None, pip: bool = False):
        want = digest({k: v for k, v in release.items() if not pip or k in ("model.safetensors", "config.json",
                                                                            "tokenizer.json")})
        want.update(digest(extra or {}))
        got = tree(root)
        self.assertEqual(sorted(got), sorted(want))
        self.assertEqual(got, want)

    def start_server(self, cwd, **env) -> tuple[int, subprocess.Popen]:
        port = free_port()
        log = open(self.tmp / f"serve-{port}.log", "w")
        self.addCleanup(log.close)
        p = subprocess.Popen([sys.executable, "-m", "moth", "--port", str(port), "--threads", "1"], cwd=cwd,
                             env=self.env(**env), stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        self.procs.append(p)
        for _ in range(240):
            if p.poll() is not None:
                break
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=1) as r:
                    if json.loads(r.read()).get("ok"):
                        return port, p
            except OSError:
                time.sleep(0.5)
        log.flush()
        self.fail("moth didn't start:\n" + Path(log.name).read_text(errors="replace")[-3000:])

    def stop_server(self, p: subprocess.Popen):
        p.terminate()   # Windows: TerminateProcess, the pidfile stays behind (stale): update must not mind
        try:
            p.wait(30)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()

    def stop_all(self):
        for p in self.procs:
            if p.poll() is None:
                self.stop_server(p)

    def talk(self, port: int, version: str):
        token = (self.home / ".moth" / "token").read_text().strip()
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=5) as r:
            self.assertEqual(json.loads(r.read())["version"], version)
        for path, body in (("/api/chat", {"text": "hi moth"}), ("/api/agent", {"text": "All tests pass.",
                                                                               "source": "claude"})):
            req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json", "X-Moth-Token": token})
            with urllib.request.urlopen(req, timeout=120) as r:
                self.assertEqual(r.status, 200, path)
                d = json.loads(r.read())
            self.assertTrue(d.get("events") is not None if path == "/api/chat" else d.get("ok"), d)

    def kill_mid_transfer(self, args, cwd, path="model.safetensors", **env):
        """Start `python -m moth <args>` (or `python <args>` when args start with "-c"), kill it while `path` is
        downloading (throttled), return when it's dead."""
        size = len(self.new[path])
        self.hub.bps[path] = max(size // 20, 1000)   # ~20 s for the file: plenty of time to kill it halfway
        before = len(self.hub.requests("GET", "/cdn/"))
        argv = [sys.executable, *args] if args[0] == "-c" else [sys.executable, "-m", "moth", *args]
        p = subprocess.Popen(argv, cwd=cwd, env=self.env(**env),
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
        self.procs.append(p)
        for _ in range(600):
            if len(self.hub.requests("GET", "/cdn/")) > before or p.poll() is not None:
                break
            time.sleep(0.05)
        time.sleep(0.5)
        self.assertIsNone(p.poll(), "the update finished before it could be killed")
        p.kill()
        p.wait()
        self.hub.bps.clear()

    def crash_hook(self) -> str:
        d = self.tmp / "crash-hook"
        d.mkdir(exist_ok=True)
        (d / "sitecustomize.py").write_text(CRASH_HOOK)
        return str(d)

    # -- the Hugging Face download layout -------------------------------------------------------------------------
    def hf_install(self) -> Path:
        """`hf download` of 0.2.9 into a folder deep enough that the backup's deepest file passes Windows' 260."""
        pad = max(20, 175 - len(str(self.tmp)) - len("/moth-112m"))
        base = self.tmp
        while pad > 0:
            part = min(pad, 60)
            base = base / ("d" * part)
            pad -= part + 1
        root = base / "moth-112m"
        self.py(f"from huggingface_hub import snapshot_download; snapshot_download({REPO!r}, local_dir={str(root)!r})",
                cwd=self.tmp)
        self.expect(root, self.old)
        return root

    def test_hf_download_layout(self):
        root = self.hf_install()
        code = root / "code"
        self.assertGreater(len(str(root / ".moth-backup" / OLD / "files" / DEEP)), 260)

        rc, out = self.moth("--version", cwd=code)
        self.assertEqual(out.strip(), f"moth {OLD} (revision {self.sha_old[:12]})")
        rc, out = self.moth("update", "--check", cwd=code)
        self.assertEqual((rc, out.strip()), (0, f"up to date: moth {OLD} (revision {self.sha_old[:12]})"))

        # the pet's daily check, before and after a release; it never installs anything
        pet = "from moth import pet; print(pet.check_for_update())"
        self.assertIn("('none', '')", self.py(pet, cwd=code))
        self.hub.set_ref("main", "new")
        self.assertIn("('update', 'python -m moth update')", self.py(pet, cwd=code))
        self.assertIn("there's a new me! run `python -m moth update`",
                      self.py("from moth import pet; print(pet.update_line('python -m moth update'))", cwd=code))
        self.expect(root, self.old)

        # your files: added ones are never touched; an edit of a file upstream didn't change is kept; an edit of one
        # it changed is replaced (and named, and kept in the backup)
        mine = {"notes.md": b"my notes\n", "code/moth/my_plugin.py": b"# mine\n"}
        tail = self.old["code/moth/tail.py"] + b"\n# my tweak\n"
        persona = self.old["code/moth/persona.py"] + b"\n# my tweak\n"
        for k, v in dict(mine, **{"code/moth/tail.py": tail, "code/moth/persona.py": persona}).items():
            (root / k).write_bytes(v)
        edited_old = dict(self.old, **mine, **{"code/moth/tail.py": tail, "code/moth/persona.py": persona})

        rc, out = self.moth("update", "--check", cwd=code)
        self.assertEqual(rc, 10, out)
        self.assertIn(f"update available: moth {OLD} -> {NEW}  (revision {self.sha_old[:12]} -> {self.sha_new[:12]})",
                      out)
        self.assertIn("removed: code/moth/old_helper.py", out)
        self.assertIn("kept your changes to code/moth/tail.py (not changed upstream)", out)
        self.assertIn("you changed code/moth/persona.py: the update replaces it", out)
        self.assertIn("- 0.3.0: kinder replies", out)
        self.expect(root, edited_old)

        # killed while the weights download: nothing changed, and the old version still runs
        self.kill_mid_transfer(["update", "--yes"], cwd=code)
        self.expect(root, edited_old)
        self.assertEqual(self.moth("--version", cwd=code)[1].strip(), f"moth {OLD} (revision {self.sha_old[:12]})")

        # the Hub drops the connection halfway, every time (the client retries, then gives up)
        self.hub.drop["model.safetensors"] = len(self.new["model.safetensors"]) // 2
        rc, out = self.moth("update", "--yes", cwd=code)
        self.assertEqual(rc, 1, out)
        self.assertIn("nothing was changed", out)
        self.hub.drop.clear()
        self.expect(root, edited_old)

        # a real moth from this folder: the update refuses while it runs and changes nothing
        if self.serve:
            port, server = self.start_server(code)
            self.talk(port, OLD)
            rc, out = self.moth("update", "--yes", cwd=code)
            self.assertEqual(rc, 1, out)
            self.assertIn(f"moth is running from this folder (port {port}", out)
            self.assertEqual(self.moth("update", "--rollback", "--yes", cwd=code)[0], 1)
            self.assertEqual(self.moth("update", "--check", cwd=code)[0], 10)   # read-only: fine while it runs
            self.stop_server(server)
            self.expect(root, edited_old)

        # killed halfway through swapping files in: the next run undoes it first, then updates
        rc, out = self.moth("update", "--yes", cwd=code, PYTHONPATH=self.crash_hook(), MOTH_TEST_CRASH_AFTER="3")
        self.assertNotEqual(rc, 0, out)
        self.assertTrue((root / ".moth-swap").is_dir())
        self.assertNotEqual(tree(root), digest(edited_old))   # half swapped
        rc, out = self.moth("update", "--yes", cwd=code)
        self.assertEqual(rc, 0, out)
        self.assertIn(f"undid an update that stopped halfway: back on moth {OLD}", out)
        self.assertIn(f"updated: moth {OLD} -> {NEW} (revision {self.sha_new[:12]})", out)
        self.assertIn("integration files changed", out)
        edited_new = dict(self.new, **mine, **{"code/moth/tail.py": tail})
        self.expect(root, edited_new)
        self.assertEqual((root / ".moth-backup" / OLD / "files" / "code/moth/persona.py").read_bytes(), persona)
        self.assertEqual(self.moth("--version", cwd=code)[1].strip(), f"moth {NEW} (revision {self.sha_new[:12]})")
        rc, out = self.moth("update", "--check", cwd=code)
        self.assertEqual(rc, 0, out)
        self.assertIn("up to date", out)
        self.assertIn("('none', '')", self.py(pet, cwd=code))
        if self.serve:
            port, server = self.start_server(code)
            self.talk(port, NEW)
            self.stop_server(server)

        # offline switches and an unreachable Hub: a clear message, nothing changes
        self.hub.set_ref("main", "old")
        for env, msg in (({"MOTH_OFFLINE": "1"}, "offline mode is on (MOTH_OFFLINE)"),
                         ({"HF_HUB_OFFLINE": "1"}, "offline mode is on (HF_HUB_OFFLINE)"),
                         ({"HF_ENDPOINT": f"http://127.0.0.1:{free_port()}"}, "can't reach the Hugging Face Hub")):
            for args in (("update", "--check"), ("update", "--yes")):
                rc, out = self.moth(*args, cwd=code, **env)
                self.assertEqual(rc, 1, out)
                self.assertIn(msg, out)
        self.expect(root, edited_new)

        # rollback (works offline), then a second one: nothing left to roll back to
        rc, out = self.moth("update", "--rollback", "--yes", cwd=code, MOTH_OFFLINE="1")
        self.assertEqual(rc, 0, out)
        self.assertIn(f"rolled back to moth {OLD}", out)
        self.expect(root, edited_old)
        self.assertEqual(self.moth("--version", cwd=code)[1].strip(), f"moth {OLD} (revision {self.sha_old[:12]})")
        rc, out = self.moth("update", "--rollback", "--yes", cwd=code)
        self.assertEqual(rc, 1, out)
        self.assertIn("no backup to roll back to", out)
        if self.serve:
            port, server = self.start_server(code)
            self.talk(port, OLD)
            self.stop_server(server)

        # --revision: pin a tag or a commit, either way
        rc, out = self.moth("update", "--yes", "--revision", "v" + NEW, cwd=code)
        self.assertEqual(rc, 0, out)
        self.expect(root, edited_new)
        rc, out = self.moth("update", "--check", "--revision", self.sha_old, cwd=code)
        self.assertEqual(rc, 10, out)
        self.assertIn(f"moth {NEW} -> {OLD}", out)
        rc, out = self.moth("update", "--yes", "--revision", "nope", cwd=code)
        self.assertEqual(rc, 1, out)
        self.assertIn("not found on the Hugging Face Hub", out)
        rc, out = self.moth("update", "--yes", cwd=code, MOTH_REVISION="v" + OLD)
        self.assertEqual(rc, 0, out)
        self.expect(root, dict(edited_old, **{"code/moth/persona.py": self.old["code/moth/persona.py"]}))
        self.assertEqual(self.moth("--version", cwd=code)[1].strip(), f"moth {OLD} (revision {self.sha_old[:12]})")
        if self.serve:
            port, server = self.start_server(code)
            self.talk(port, OLD)
            self.stop_server(server)
        leftovers = [p.name for p in root.iterdir() if p.name.startswith(".moth") and
                     p.name not in (".moth-backup", ".moth-version")]
        self.assertEqual(leftovers, [])   # no staging, swap plan or lock left behind

    def test_file_in_use_or_read_only(self):
        """A file the update must replace can't be moved: the swap is undone and a rerun finishes."""
        root = self.hf_install()
        code = root / "code"
        self.hub.set_ref("main", "new")
        target = code / "moth" / "persona.py"
        if os.name == "nt":
            blocker = open(target, "rb")   # Windows can't rename a file another program has open
            release = blocker.close
        else:
            if hasattr(os, "geteuid") and os.geteuid() == 0:
                self.skipTest("root ignores permissions")
            os.chmod(code / "moth", 0o555)
            release = lambda: os.chmod(code / "moth", 0o755)   # noqa: E731
        try:
            rc, out = self.moth("update", "--yes", cwd=code)
        finally:
            release()
        self.assertEqual(rc, 1, out)
        self.assertIn("replacing the files failed", out)
        self.assertIn(f"nothing was changed, still on moth {OLD}", out)
        self.expect(root, self.old)
        rc, out = self.moth("update", "--yes", cwd=code)
        self.assertEqual(rc, 0, out)
        self.expect(root, self.new)

    # -- pip install ------------------------------------------------------------------------------------------------
    def test_pip_layout(self):
        """moth and tiny_agent on PYTHONPATH (as site-packages has them, no data/ next to them): pip mode, weights in
        $MOTH_CACHE/<repo>/; update refreshes only them, and --check says when newer code is out."""
        site = self.tmp / "site"
        for k, v in package_files().items():
            (site / k).parent.mkdir(parents=True, exist_ok=True)
            (site / k).write_bytes(v)
        (site / "moth" / "VERSION").write_text(OLD + "\n")
        cache = self.tmp / "cache" / Path(*REPO.split("/"))
        cwd = self.tmp / "elsewhere"
        cwd.mkdir()
        run = lambda *a, **e: self.moth(*a, cwd=cwd, PYTHONPATH=str(site), **e)   # noqa: E731

        self.assertEqual(run("--version")[1].strip(), f"moth {OLD} (revision unknown)")
        rc, out = run("update", "--yes")   # before the first run: fetches the weights
        self.assertEqual(rc, 0, out)
        self.expect(cache, self.old, pip=True)
        self.assertEqual(run("--version")[1].strip(), f"moth {OLD} (revision {self.sha_old[:12]})")
        rc, out = run("update", "--check")
        self.assertEqual(rc, 0, out)
        self.assertIn(f"weights up to date: revision {self.sha_old[:12]}", out)

        self.hub.set_ref("main", "new")
        rc, out = run("update", "--check")
        self.assertEqual(rc, 10, out)
        self.assertIn(f"newer weights on the Hub: revision {self.sha_old[:12]} -> {self.sha_new[:12]}", out)
        self.assertIn(f"newer code: moth {OLD} -> {NEW}:  pip install -U moth-companion", out)
        out = self.py("from moth import pet; print(pet.check_for_update())", cwd=cwd, PYTHONPATH=str(site))
        self.assertIn("('update', 'pip install -U moth-companion')", out)

        self.kill_mid_transfer(["update", "--yes"], cwd=cwd, PYTHONPATH=str(site))
        self.expect(cache, self.old, pip=True)
        if self.serve:   # the old weights still serve; and the update waits for that server
            port, server = self.start_server(cwd, PYTHONPATH=str(site))
            self.talk(port, OLD)
            rc, out = run("update", "--yes")
            self.assertEqual(rc, 1, out)
            self.assertIn("stop moth first", out)
            self.stop_server(server)

        rc, out = run("update", "--yes")
        self.assertEqual(rc, 0, out)
        self.assertIn(f"updated the weights in {cache} to revision {self.sha_new[:12]}", out)
        self.expect(cache, self.new, pip=True)
        self.assertEqual(run("--version")[1].strip(), f"moth {OLD} (revision {self.sha_new[:12]})")
        if self.serve:
            port, server = self.start_server(cwd, PYTHONPATH=str(site))
            self.talk(port, OLD)   # the code version: pip's
            self.stop_server(server)

        self.assertEqual(run("update", "--rollback", "--yes")[0], 0)
        self.expect(cache, self.old, pip=True)
        rc, out = run("update", "--rollback", "--yes")
        self.assertEqual(rc, 1, out)
        self.assertIn("no backup to roll back to", out)
        rc, out = run("update", "--yes", MOTH_OFFLINE="1")
        self.assertEqual(rc, 1, out)
        self.assertIn("offline mode is on", out)
        self.expect(cache, self.old, pip=True)

    def test_docs_only_change_is_no_update(self):
        """A card / docs edit: --check stays "up to date" (exit 0) and the pet stays quiet; update still fetches it."""
        root = self.hf_install()
        code = root / "code"
        self.hub.set_ref("main", "docs")
        rc, out = self.moth("update", "--check", cwd=code)
        self.assertEqual(rc, 0, out)
        self.assertIn(f"up to date: moth {OLD} (revision {self.sha_old[:12]})  (only the docs changed upstream", out)
        self.assertIn("('none', '')", self.py("from moth import pet; print(pet.check_for_update())", cwd=code))
        rc, out = self.moth("update", "--yes", cwd=code)
        self.assertEqual(rc, 0, out)
        self.expect(root, self.docs)

    def test_pip_never_fetches_a_model_its_code_cant_run(self):
        """pip: a Hub model that needs newer code is not fetched (check says how to upgrade; the pet says pip); the
        first run falls back to the model tagged for this version, or says how to upgrade when one was pinned."""
        site = self.tmp / "site"
        for k, v in package_files().items():
            (site / k).parent.mkdir(parents=True, exist_ok=True)
            (site / k).write_bytes(v)
        (site / "moth" / "VERSION").write_text(OLD + "\n")
        cache = self.tmp / "cache" / Path(*REPO.split("/"))
        cwd = self.tmp / "elsewhere"
        cwd.mkdir()
        run = lambda *a, **e: self.moth(*a, cwd=cwd, PYTHONPATH=str(site), **e)   # noqa: E731
        self.assertEqual(run("update", "--yes")[0], 0)   # the old weights
        self.expect(cache, self.old, pip=True)

        self.hub.set_ref("main", "future")
        rc, out = run("update", "--check")
        self.assertEqual(rc, 10, out)
        self.assertIn("needs newer moth code: it needs moth 9.0.0 or newer", out)
        self.assertIn("newer code: pip install -U moth-companion", out)
        out = self.py("from moth import pet; print(pet.check_for_update())", cwd=cwd, PYTHONPATH=str(site))
        self.assertIn("('update', 'pip install -U moth-companion')", out)
        rc, out = run("update", "--yes")
        self.assertEqual(rc, 1, out)
        self.assertIn("needs newer moth code", out)
        self.assertIn("Nothing was changed", out)
        self.expect(cache, self.old, pip=True)

        shutil.rmtree(cache)   # a first run now: falls back to the model tagged v<this version>
        ensure = "from moth import weights; print(weights.ensure())"
        p = subprocess.run([sys.executable, "-c", ensure], cwd=self.tmp, env=self.env(PYTHONPATH=str(site)),
                           capture_output=True, text=True, timeout=180)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn(f"using the one made for moth {OLD} (v{OLD})", p.stderr)
        self.expect(cache, self.old, pip=True)
        shutil.rmtree(cache)   # pinned to the new one: no fallback, just how to upgrade
        p = subprocess.run([sys.executable, "-c", ensure], cwd=self.tmp, env=self.env(
            PYTHONPATH=str(site), MOTH_REVISION=self.sha_future), capture_output=True, text=True, timeout=180)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn(f"is newer than this moth ({OLD}): it needs moth 9.0.0 or newer", p.stderr)
        self.assertIn("pip install -U moth-companion", p.stderr)
        self.assertFalse((cache / "model.safetensors").exists())

    def test_pip_first_run_offline_and_interrupted(self):
        """The first run's download (weights.ensure): refused offline, resumable after a kill."""
        site = self.tmp / "site"
        for k, v in package_files().items():
            (site / k).parent.mkdir(parents=True, exist_ok=True)
            (site / k).write_bytes(v)
        cache = self.tmp / "cache" / Path(*REPO.split("/"))
        ensure = "from moth import weights; print(weights.ensure())"
        p = subprocess.run([sys.executable, "-c", ensure], cwd=self.tmp, env=self.env(PYTHONPATH=str(site),
                           MOTH_OFFLINE="1"), capture_output=True, text=True, timeout=120)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("offline mode is on", p.stderr)
        self.assertFalse(cache.exists())
        self.hub.set_ref("main", "new")
        self.kill_mid_transfer(["-c", ensure], cwd=self.tmp, PYTHONPATH=str(site))
        self.assertFalse((cache / "model.safetensors").exists())
        self.assertFalse((cache / ".moth-version").exists())
        out = self.py(ensure, cwd=self.tmp, PYTHONPATH=str(site))
        self.assertEqual(Path(out.strip()).resolve(), cache.resolve())
        self.expect(cache, self.new, pip=True)


if __name__ == "__main__":
    unittest.main()
