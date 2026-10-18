"""moth/update.py against a fake Hub: huggingface_hub is replaced in sys.modules, so nothing can reach the network.

  python3 tests/test_update.py      (from ~/moth; under a second, temp dirs only, no model, no server)

Covers --check (up to date / update available), an in-place update that leaves ~/.moth alone, backup + rollback,
files the new version dropped (removed, restored on rollback; user files kept), the refusal only while a moth from
this install runs (serve.py's real /api/health handler and pidfile, torch faked out), the offline path, the
missing-huggingface_hub hint and `python -m moth --help`.
"""
from __future__ import annotations

import contextlib
import fnmatch
import hashlib
import http.client
import importlib.util
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import types
import unittest
import unittest.mock
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from moth import update as U  # noqa: E402
from moth import weights as W  # noqa: E402

LFS = {"model.safetensors"}
HOME_VARS = ("HOME", "USERPROFILE")   # expanduser("~") reads USERPROFILE on Windows, HOME elsewhere


def blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


class FakeHub:
    """A repo with commits {sha: {path: bytes}}; model_info / snapshot_download / hf_hub_download like the real one."""

    def __init__(self, commits: dict, head: str):
        self.commits, self.head, self.offline, self.fetched = commits, head, False, []
        hub = self

        class HfApi:
            def model_info(self, repo, revision=None, files_metadata=False):
                if hub.offline:
                    raise ConnectionError("Failed to resolve 'huggingface.co'")
                sha = revision or hub.head
                sibs = []
                for path, data in hub.commits[sha].items():
                    lfs = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()} if path in LFS else None
                    sibs.append(types.SimpleNamespace(rfilename=path, size=len(data), lfs=lfs,
                                                      blob_id=hashlib.sha1(data).hexdigest() if lfs else blob_id(data)))
                return types.SimpleNamespace(sha=sha, siblings=sibs)

        self.module = types.ModuleType("huggingface_hub")
        self.module.HfApi = HfApi
        self.module.snapshot_download = self.snapshot_download
        self.module.hf_hub_download = self.hf_hub_download

    def snapshot_download(self, repo_id, local_dir, revision=None, allow_patterns=None):
        for path, data in self.commits[revision].items():
            if allow_patterns is not None and not any(fnmatch.fnmatch(path, p) for p in allow_patterns):
                continue
            dest = Path(local_dir, path)
            if dest.exists() and dest.read_bytes() == data:
                continue   # the real one skips current files too
            self.fetched.append(path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
        return local_dir

    def hf_hub_download(self, repo_id, filename, revision=None, cache_dir=None):
        path = Path(cache_dir, filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.commits[revision][filename])
        return str(path)


V1 = {"model.safetensors": b"weights-v1" * 100, "config.json": b'{"n": 1}', "tokenizer.json": b"{}",
      "README.md": b"# moth\n", "code/moth/VERSION": b"0.3.0\n", "code/moth/serve.py": b"# serve v1\n",
      "code/moth/integrations/pi/moth.ts": b"// pi v1\n", "code/moth/old/gone.py": b"# dropped in v2\n"}
V2 = {k: v for k, v in V1.items() if k != "code/moth/old/gone.py"}
V2.update(**{"README.md": b"# moth\n\n## Changelog\n- 0.3.1: kinder replies\n\n## License\nx\n",
                 "code/moth/VERSION": b"0.3.1\n", "code/moth/serve.py": b"# serve v2\n",
                 "code/moth/integrations/pi/moth.ts": b"// pi v2\n", "code/moth/new.py": b"# new\n"})
TOKEN = "test-token-not-secret"


def load_serve():
    """serve.py with torch faked out, through the integrations test's loader (as tests/test_pet.py does)."""
    spec = importlib.util.spec_from_file_location("moth_itest", ROOT / "moth" / "integrations" / "test_integrations.py")
    itest = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(itest)
    return itest.load_serve()
V3 = dict(V2, **{"model.safetensors": b"weights-v3" * 100, "code/moth/VERSION": b"0.4.0\n"})


def http_server(handler):
    from http.server import ThreadingHTTPServer
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


class UpdateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.root = self.tmp / "moth-112m"
        for path, data in V1.items():
            (self.root / path).parent.mkdir(parents=True, exist_ok=True)
            (self.root / path).write_bytes(data)
        U._write_state(str(self.root), {"revision": "sha1", "files": sorted(V1)})
        self.home = self.tmp / "home"
        (self.home / ".moth" / "home").mkdir(parents=True)
        (self.home / ".moth" / "token").write_text("secret-token")
        (self.home / ".moth" / "home" / "notes.txt").write_text("moth's notes")
        self.hub = FakeHub({"sha1": V1, "sha2": V2, "sha3": V3}, head="sha2")
        self.env = {k: os.environ.get(k) for k in HOME_VARS}
        os.environ.update(dict.fromkeys(HOME_VARS, str(self.home)))   # serve's pidfiles: ~/.moth/serve-<port>.pid
        self.saved = sys.modules.get("huggingface_hub")
        sys.modules["huggingface_hub"] = self.hub.module

    def tearDown(self):
        for k, v in self.env.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        sys.modules.pop("huggingface_hub", None)
        if self.saved is not None:
            sys.modules["huggingface_hub"] = self.saved
        shutil.rmtree(self.tmp)

    def run_cli(self, *args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = U.main(list(args), root=str(self.root))
        return code, out.getvalue()

    def tree(self) -> dict:
        return {p.relative_to(self.root).as_posix(): p.read_bytes() for p in self.root.rglob("*")
                if p.is_file() and not p.relative_to(self.root).as_posix().startswith((".moth", ".cache"))}

    def moth_state(self) -> dict:
        return {str(p): p.read_bytes() for p in (self.home / ".moth").rglob("*") if p.is_file()}

    def test_check_up_to_date(self):
        self.hub.head = "sha1"
        code, out = self.run_cli("--check")
        self.assertEqual(code, 0, out)
        self.assertIn("up to date: moth 0.3.0 (revision sha1)", out)

    def test_bare_sha_revision_file(self):   # an older .moth-version: no manifest, so nothing gets removed
        (self.root / U.REVISION_FILE).write_text("sha1\n")
        self.assertEqual(U.revision(str(self.root)), "sha1")
        self.assertEqual(self.run_cli("--yes")[0], 0)
        self.assertEqual(self.tree(), dict(V2, **{"code/moth/old/gone.py": V1["code/moth/old/gone.py"]}))

    def test_check_up_to_date_without_revision_file(self):   # plain `hf download`, no .moth-version, same files
        (self.root / U.REVISION_FILE).unlink()
        self.hub.head = "sha1"
        self.assertEqual(self.run_cli("--check")[0], 0)

    def test_check_update_available(self):
        before = self.tree()
        code, out = self.run_cli("--check")
        self.assertEqual(code, 10, out)
        self.assertIn("moth 0.3.0 -> 0.3.1", out)
        self.assertIn("changes: code (5 files", out)
        self.assertIn("removed: code/moth/old/gone.py", out)
        self.assertIn("- 0.3.1: kinder replies", out)
        self.assertNotIn("License", out)
        self.assertEqual(self.tree(), before)   # --check changes nothing
        self.assertFalse((self.root / U.BACKUP).exists())
        self.hub.head = "sha3"
        self.assertIn("changes: weights and code", self.run_cli("--check")[1])

    def test_update_in_place_and_rollback(self):
        moth_before = self.moth_state()
        code, out = self.run_cli("--yes")
        self.assertEqual(code, 0, out)
        self.assertEqual(self.tree(), V2)
        self.assertEqual(sorted(self.hub.fetched), sorted(["README.md", "code/moth/VERSION", "code/moth/serve.py",
                                                           "code/moth/integrations/pi/moth.ts", "code/moth/new.py"]))
        self.assertEqual(U.state(str(self.root)), {"revision": "sha2", "files": sorted(V2)})
        self.assertFalse((self.root / "code/moth/old").exists())   # dropped file and its emptied folder
        self.assertIn("updated: moth 0.3.0 -> 0.3.1", out)
        self.assertIn("integration files changed", out)
        self.assertTrue((self.root / U.BACKUP / "0.3.0" / "files" / "code/moth/serve.py").is_file())
        self.assertFalse((self.root / U.BACKUP / "0.3.0" / "files" / "model.safetensors").exists())   # unchanged
        self.assertEqual(self.moth_state(), moth_before)
        self.assertEqual(U.version_line(str(self.root)), "moth 0.3.1 (revision sha2)")

        # a second update keeps only the newest backup
        self.hub.head = "sha3"
        self.assertEqual(self.run_cli("--yes")[0], 0)
        self.assertEqual(self.tree(), V3)
        self.assertEqual(os.listdir(self.root / U.BACKUP), ["0.3.1"])

        code, out = self.run_cli("--rollback", "--yes")
        self.assertEqual(code, 0, out)
        self.assertEqual(self.tree(), V2)
        self.assertEqual(U.state(str(self.root)), {"revision": "sha2", "files": sorted(V2)})
        self.assertFalse((self.root / U.BACKUP).exists())
        self.assertEqual(self.run_cli("--rollback", "--yes")[0], 1)   # nothing left to roll back to
        self.assertEqual(self.moth_state(), moth_before)

    def test_rollback_removes_added_files_and_hf_metadata(self):
        self.assertEqual(self.run_cli("--yes")[0], 0)
        meta = self.root / ".cache/huggingface/download/code/moth/serve.py.metadata"
        meta.parent.mkdir(parents=True, exist_ok=True)
        meta.write_text("sha2\netag\n0\n")
        self.assertEqual(self.run_cli("--rollback", "--yes")[0], 0)
        self.assertEqual(self.tree(), V1)   # incl. the removed code/moth/old/gone.py
        self.assertEqual(U.state(str(self.root)), {"revision": "sha1", "files": sorted(V1)})
        self.assertFalse(meta.exists())

    def test_plain_hf_download_manifest(self):   # no .moth-version: the file list from hf's .metadata files
        (self.root / U.REVISION_FILE).unlink()
        for path in V1:
            meta = self.root / ".cache/huggingface/download" / (path + ".metadata")
            meta.parent.mkdir(parents=True, exist_ok=True)
            meta.write_text("sha1\netag\n0\n")
        (self.root / "code/moth/old/mine.txt").write_text("my notes")   # no metadata: the user's
        self.assertEqual(U.state(str(self.root)), {"revision": "sha1", "files": sorted(V1)})
        self.assertIn("removed: code/moth/old/gone.py", self.run_cli("--check")[1])
        self.assertEqual(self.run_cli("--yes")[0], 0)
        self.assertFalse((self.root / "code/moth/old/gone.py").exists())
        self.assertEqual((self.root / "code/moth/old/mine.txt").read_text(), "my notes")
        self.assertEqual(self.run_cli("--rollback", "--yes")[0], 0)
        self.assertEqual(self.tree(), dict(V1, **{"code/moth/old/mine.txt": b"my notes"}))

    def test_user_files_are_never_removed(self):
        (self.root / "code/moth/old/mine.txt").write_text("my notes")   # next to a dropped file, not in any manifest
        (self.root / "notes.md").write_text("mine")
        (self.root / U.REVISION_FILE).write_text(json.dumps(   # a tampered manifest can't reach outside the download
            {"revision": "sha1", "files": sorted(V1) + ["../outside.txt", str(self.tmp / "abs.txt")]}))
        (self.tmp / "outside.txt").write_text("x")
        (self.tmp / "abs.txt").write_text("x")
        self.assertEqual(self.run_cli("--yes")[0], 0)
        self.assertFalse((self.root / "code/moth/old/gone.py").exists())
        self.assertEqual((self.root / "code/moth/old/mine.txt").read_text(), "my notes")
        self.assertEqual((self.root / "notes.md").read_text(), "mine")
        self.assertTrue((self.tmp / "outside.txt").exists() and (self.tmp / "abs.txt").exists())

    def test_update_up_to_date_writes_revision(self):
        (self.root / U.REVISION_FILE).unlink()
        self.hub.head = "sha1"
        code, out = self.run_cli("--yes")
        self.assertEqual(code, 0, out)
        self.assertEqual(U.state(str(self.root)), {"revision": "sha1", "files": sorted(V1)})
        self.assertEqual(self.hub.fetched, [])

    def serve_moth(self, install_root):
        """serve.py's real handler on a free port with its real pidfile, claiming install_root."""
        serve = load_serve()
        serve.INSTALL_ROOT = str(install_root)
        srv = http_server(serve.make_handler(types.SimpleNamespace(), TOKEN))
        pidfile = serve.write_pidfile(srv.server_address[1])
        self.addCleanup(lambda: (srv.shutdown(), srv.server_close(), serve.remove_pidfile(pidfile)))
        return serve, srv, pidfile

    def test_health_and_pidfile(self):
        serve, srv, pidfile = self.serve_moth(self.root)
        port = srv.server_address[1]
        self.assertEqual(pidfile, str(self.home / ".moth" / f"serve-{port}.pid"))
        self.assertEqual(json.loads(Path(pidfile).read_text()),
                         {"pid": os.getpid(), "port": port, "install_root": str(self.root)})
        for host, code in ((f"127.0.0.1:{port}", 200), (f"localhost:{port}", 200), (f"evil.example:{port}", 403)):
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            c.request("GET", "/api/health", headers={"Host": host})   # no token
            r = c.getresponse()
            self.assertEqual(r.status, code, host)
            body = json.loads(r.read())
            if code == 200:
                self.assertEqual(body, {"ok": True, "pid": os.getpid(), "install_root": str(self.root),
                                        "version": serve.version()})
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        c.request("GET", "/api/log")   # everything else still needs the token
        self.assertEqual(c.getresponse().status, 401)
        serve.remove_pidfile(pidfile)
        self.assertFalse(Path(pidfile).exists())

    def test_alive_on_every_os(self):
        """The pid check behind stale pidfiles: a real check on Windows too (not os.kill(pid, 0), which signals there)."""
        self.assertTrue(U._alive(os.getpid()))
        p = subprocess.Popen([sys.executable, "-c", "pass"])
        p.wait()
        self.assertFalse(U._alive(p.pid))
        self.assertFalse(U._alive(2 ** 40))   # a garbled pidfile

    def test_refuses_while_moth_from_this_install_runs(self):
        _, srv, _ = self.serve_moth(self.root)
        port = srv.server_address[1]
        for args in (("--yes",), ("--rollback", "--yes")):
            code, out = self.run_cli(*args)
            self.assertEqual(code, 1)
            self.assertIn(f"port {port}, pid {os.getpid()}): stop moth first", out)
        self.assertEqual(self.run_cli("--check")[0], 10)   # read-only, fine while running
        self.assertEqual(self.tree(), V1)

    def test_other_listeners_dont_block(self):
        moth = self.home / ".moth"
        with socket.socket() as other:   # not moth: no health route, and no pidfile of its own
            other.bind(("127.0.0.1", 0))
            other.listen()
            port = other.getsockname()[1]
            # a pidfile pointing at it (moth crashed, something else took the port)
            (moth / f"serve-{port}.pid").write_text(json.dumps({"pid": os.getpid(), "port": port,
                                                                "install_root": str(self.root)}))
            dead = subprocess.Popen([sys.executable, "-c", "pass"])
            dead.wait()
            (moth / "serve-1.pid").write_text(json.dumps({"pid": dead.pid, "port": 1, "install_root": str(self.root)}))
            (moth / "serve-2.pid").write_text("not json")
            self.serve_moth(self.tmp / "another-install")   # a live moth, but from another download
            code, out = self.run_cli("--yes")
        self.assertEqual(code, 0, out)
        self.assertEqual(self.tree(), V2)

    def test_offline(self):
        self.hub.offline = True
        for args in (("--check",), ("--yes",)):
            code, out = self.run_cli(*args)
            self.assertEqual(code, 1)
            self.assertIn("can't reach the Hugging Face Hub", out)
        self.assertEqual(self.tree(), V1)

    def test_download_failure_restores(self):
        def broken(repo_id, local_dir, revision=None):
            (Path(local_dir) / "code/moth/serve.py").write_bytes(b"# half\n")
            (Path(local_dir) / "code/moth/new.py").write_bytes(b"# half\n")
            raise ConnectionError("connection reset")
        self.hub.module.snapshot_download = broken
        code, out = self.run_cli("--yes")
        self.assertEqual(code, 1)
        self.assertIn("download failed", out)
        self.assertEqual(self.tree(), V1)
        self.assertEqual(U.state(str(self.root)), {"revision": "sha1", "files": sorted(V1)})

    def test_no_confirmation_no_change(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), unittest.mock.patch("builtins.input", return_value="n"):
            self.assertEqual(U.main([], root=str(self.root)), 1)
        self.assertEqual(self.tree(), V1)

    def test_missing_huggingface_hub(self):
        sys.modules["huggingface_hub"] = None
        code, out = self.run_cli("--check")
        self.assertEqual(code, 1)
        self.assertIn("pip install -U huggingface_hub", out)

    def test_not_a_download(self):
        (self.root / "model.safetensors").unlink()
        code, out = self.run_cli("--yes")
        self.assertEqual(code, 1)
        self.assertIn("isn't a Hugging Face download", out)

    def test_main_version_flag(self):
        from moth import __main__ as M
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            M.main(["--version"])
        self.assertRegex(out.getvalue(), r"^moth \S+ \(revision \S+\)")
        self.assertNotIn("moth.serve", sys.modules)   # no torch / server import for --version

    def test_main_help_lists_subcommands(self):
        from moth import __main__ as M
        fake = types.ModuleType("moth.serve")
        fake.main = lambda argv, on_ready=None: print("SERVE HELP", argv)   # serve's argparse, without torch
        out = io.StringIO()
        with contextlib.redirect_stdout(out), unittest.mock.patch.dict(sys.modules, {"moth.serve": fake}), \
                unittest.mock.patch("moth.serve", fake, create=True):
            M.main(["--help"])
        text = out.getvalue()
        for line in ("python -m moth [serve options]", "python -m moth update", "python -m moth --version"):
            self.assertIn(line, text)
        self.assertIn("SERVE HELP ['--help']", text)
        self.assertNotIn("Paste what fits your harness", text)   # no snippets on --help
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit):
            U.main(["--help"], root=str(self.root))
        self.assertIn("--rollback", out.getvalue())
        self.assertIn("10 = update available", out.getvalue())


class PipModeTest(unittest.TestCase):
    """pip install moth-companion: weights fetched into the cache on first run, `moth update` refreshes only them."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.home = self.tmp / "home"
        (self.home / ".moth").mkdir(parents=True)
        self.hub = FakeHub({"sha1": V1, "sha2": V2, "sha3": V3}, head="sha1")
        self.env = {k: os.environ.get(k) for k in (*HOME_VARS, "MOTH_CACHE", "MOTH_REPO", "MOTH_REVISION")}
        os.environ.update(dict.fromkeys(HOME_VARS, str(self.home)), MOTH_CACHE=str(self.tmp / "cache"))
        os.environ.pop("MOTH_REPO", None)
        os.environ.pop("MOTH_REVISION", None)
        self.cache = Path(W.cache_dir())
        self.saved = sys.modules.get("huggingface_hub")
        sys.modules["huggingface_hub"] = self.hub.module
        self.patches = [unittest.mock.patch.object(W, "mode", return_value="pip"),
                        unittest.mock.patch.object(U, "version", lambda root=None: "0.3.0")]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        for k, v in self.env.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        sys.modules.pop("huggingface_hub", None)
        if self.saved is not None:
            sys.modules["huggingface_hub"] = self.saved
        shutil.rmtree(self.tmp)

    def ensure(self, *a, **kw):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            return W.ensure(*a, **kw), err.getvalue()

    def run_cli(self, *args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = U.main(list(args))   # the real install root: pip mode decides by weights.mode()
        return code, out.getvalue()

    def files(self) -> dict:
        return {p.relative_to(self.cache).as_posix(): p.read_bytes() for p in self.cache.rglob("*")
                if p.is_file() and not p.relative_to(self.cache).as_posix().startswith((".moth", ".cache"))}

    def test_cache_dir(self):
        self.assertEqual(self.cache, self.tmp / "cache" / "darioooooo0o" / "moth-112m")
        os.environ["MOTH_REPO"] = "someone/other-moth"
        self.assertEqual(Path(W.cache_dir()), self.tmp / "cache" / "someone" / "other-moth")

    def test_first_run_downloads_only_weights(self):
        folder, err = self.ensure()
        self.assertEqual(Path(folder), self.cache)
        self.assertEqual(self.files(), {f: V1[f] for f in W.FILES})   # no README, no code/
        self.assertIn("downloading the model", err)
        self.assertIn("revision sha1", err)
        self.assertEqual(U.state(str(self.cache)), {"revision": "sha1", "files": sorted(W.FILES)})
        self.hub.offline, self.hub.fetched = True, []
        self.assertEqual(self.ensure(), (folder, ""))   # present: no network, silent
        self.assertEqual(self.hub.fetched, [])

    def test_revision_pin_and_partial_download(self):
        os.environ["MOTH_REVISION"] = "sha3"
        self.hub.head = "sha1"
        self.ensure()
        self.assertEqual(self.files()["model.safetensors"], V3["model.safetensors"])
        (self.cache / "config.json").unlink()   # interrupted download: all three files are needed
        self.assertFalse(W.present(str(self.cache)))
        self.ensure()
        self.assertTrue(W.present(str(self.cache)))

    def test_offline_and_missing_hub(self):
        self.hub.offline = True
        with self.assertRaises(SystemExit) as cm:
            self.ensure()
        self.assertIn("can't get darioooooo0o/moth-112m", str(cm.exception))
        sys.modules["huggingface_hub"] = None   # import fails
        with self.assertRaises(SystemExit) as cm:
            self.ensure()
        self.assertIn("pip install -U huggingface_hub", str(cm.exception))
        self.assertFalse(self.cache.exists())

    def test_ensure_default_only_for_the_default_folder(self):
        W.ensure_default(str(self.tmp / "my-own-weights"))   # --model elsewhere: never downloads
        self.assertFalse(self.cache.exists())
        with unittest.mock.patch.object(W, "mode", return_value="dev"):
            W.ensure_default(str(self.cache))
        self.assertFalse(self.cache.exists())
        with contextlib.redirect_stderr(io.StringIO()):
            W.ensure_default(str(self.cache))
        self.assertTrue(W.present(str(self.cache)))

    def test_default_paths_pip_mode_does_not_download(self):
        serve = load_serve()
        d = serve.default_paths()
        self.assertEqual(Path(d["model"]), self.cache)
        self.assertEqual(Path(d["tokenizer"]), self.cache / "tokenizer.json")
        self.assertEqual(Path(d["home"]), self.home / ".moth" / "home")
        self.assertTrue(d["pool"].endswith("file_pool.json"))
        self.assertFalse(self.cache.exists())   # building the parser (moth --help) fetches nothing

    def test_check(self):
        self.ensure()
        self.hub.head = "sha2"   # same weights, newer code (0.3.1)
        code, out = self.run_cli("--check")
        self.assertEqual(code, 10, out)
        self.assertIn("weights up to date", out)
        self.assertIn("newer code: moth 0.3.0 -> 0.3.1:  pip install -U moth-companion", out)
        self.hub.head = "sha3"   # new weights
        code, out = self.run_cli("--check")
        self.assertEqual(code, 10, out)
        self.assertIn("newer weights on the Hub: revision sha1 -> sha3", out)
        self.assertIn("changes: weights (1 file: model.safetensors)", out)
        self.assertEqual(self.files(), {f: V1[f] for f in W.FILES})   # --check changes nothing
        self.hub.commits["sha4"] = dict(V1, **{"code/moth/VERSION": b"0.3.0\n"})
        self.hub.head = "sha4"
        code, out = self.run_cli("--check")
        self.assertEqual(code, 0, out)
        self.assertIn("pip install -U moth-companion gets the newest release", out)

    def test_update_and_rollback(self):
        self.ensure()
        self.hub.head, self.hub.fetched = "sha3", []
        code, out = self.run_cli("--yes")
        self.assertEqual(code, 0, out)
        self.assertEqual(self.hub.fetched, ["model.safetensors"])   # never code/ or README into the cache
        self.assertEqual(self.files(), {f: V3[f] for f in W.FILES})
        self.assertEqual(U.state(str(self.cache))["revision"], "sha3")
        self.assertIn("pip install -U moth-companion", out)
        self.assertEqual(U.version_line(), "moth 0.3.0 (revision sha3)")
        self.assertEqual(self.run_cli("--yes")[0], 0)   # up to date now
        code, out = self.run_cli("--rollback", "--yes")
        self.assertEqual(code, 0, out)
        self.assertEqual(self.files(), {f: V1[f] for f in W.FILES})
        self.assertEqual(U.state(str(self.cache))["revision"], "sha1")

    def test_update_before_first_run_downloads(self):
        self.assertEqual(self.run_cli("--yes")[0], 0)
        self.assertTrue(W.present(str(self.cache)))

    def test_refuses_while_moth_from_this_install_runs(self):
        self.ensure()
        self.hub.head = "sha3"
        with unittest.mock.patch.object(U, "server_running", return_value=(4747, 123)) as running:
            code, out = self.run_cli("--yes")
        self.assertEqual(code, 1)
        self.assertIn("stop moth first", out)
        running.assert_called_with(U.ROOT)   # serve's pidfile names the install, not the cache
        self.assertEqual(self.files(), {f: V1[f] for f in W.FILES})


if __name__ == "__main__":
    unittest.main()
