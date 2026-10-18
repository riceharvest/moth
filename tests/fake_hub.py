"""A local stand-in for the Hugging Face Hub, for tests/test_update_e2e.py and manual `moth update` runs.

huggingface_hub talks to it through HF_ENDPOINT=http://127.0.0.1:<port>: the real model_info / list_repo_tree /
snapshot_download / hf_hub_download code paths run, only the server is fake. Nothing here can reach the real Hub.

  python tests/fake_hub.py --port 8790 --repo darioooooo0o/moth-112m --commit old=DIR --commit new=DIR --main old
      serves the folders as commits (sha = sha1 of the file list and contents); control endpoints:
      GET /_fake/main?to=new            point main (and the default revision) at another commit
      GET /_fake/fault?path=P&drop=N    drop the connection after N bytes of P's content (until cleared: drop=)
      GET /_fake/fault?path=P&bps=N     throttle P to N bytes/s (bps= clears)
      GET /_fake/fail?what=tree&code=500  answer that endpoint kind (info, tree, head, get, cdn) with an error
      GET /_fake/log                    the requests so far (JSON)

What it implements (what huggingface_hub 1.x/2.x calls): GET /api/models/<repo>[/revision/<rev>] (siblings with
size / blobId / lfs when ?blobs=true), GET /api/models/<repo>/tree/<rev>[/<path>] (recursive), HEAD/GET
/<repo>/resolve/<rev>/<path> with X-Repo-Commit and ETag (git blob id; for LFS files a 302 to the "CDN" with
X-Linked-Etag = sha256 and X-Linked-Size, like the real Hub), and the CDN at localhost:<another port> (another host
name, so the client doesn't follow the HEAD redirect: the real CDN is another host) with Range support for resumed
downloads.
No xet headers: downloads take the plain HTTP path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, unquote, urlsplit

LFS_PATTERNS = (".safetensors", ".bin", ".pt", ".gguf")
CHUNK = 64 * 1024


def _lfs(path: str) -> bool:
    return path.endswith(LFS_PATTERNS)


class Blob:
    """One file of a commit: bytes, or a path on disk (streamed; the real 860 MB weights)."""

    def __init__(self, path: str, data: bytes | None = None, src: str | None = None):
        self.path, self.data, self.src = path, data, src
        self.size = len(data) if data is not None else os.path.getsize(src)
        sha1, sha256 = hashlib.sha1(b"blob %d\0" % self.size), hashlib.sha256()
        for chunk in self.chunks():
            sha1.update(chunk)
            sha256.update(chunk)
        self.sha256 = sha256.hexdigest()
        self.lfs = _lfs(path)
        if self.lfs:   # the git blob of an LFS file is its pointer
            self.pointer = (f"version https://git-lfs.github.com/spec/v1\noid sha256:{self.sha256}\n"
                            f"size {self.size}\n").encode()
            self.blob_id = hashlib.sha1(b"blob %d\0" % len(self.pointer) + self.pointer).hexdigest()
        else:
            self.blob_id = sha1.hexdigest()

    def chunks(self, start: int = 0, end: int | None = None):
        end = self.size if end is None else end
        if self.data is not None:
            for i in range(start, end, CHUNK):
                yield self.data[i:min(i + CHUNK, end)]
            return
        with open(self.src, "rb") as f:
            f.seek(start)
            left = end - start
            while left > 0:
                b = f.read(min(CHUNK, left))
                if not b:
                    return
                left -= len(b)
                yield b


def commit_sha(files: dict) -> str:
    h = hashlib.sha1()
    for p in sorted(files):
        h.update(p.encode() + b"\0" + files[p].blob_id.encode() + files[p].sha256.encode())
    return h.hexdigest()


class FakeHub:
    def __init__(self, repo_id: str):
        self.repo_id = repo_id
        self.commits: dict[str, dict[str, Blob]] = {}
        self.names: dict[str, str] = {}    # friendly name -> sha
        self.refs: dict[str, str] = {}     # branch / tag -> sha
        self.drop: dict[str, int] = {}     # path -> drop the connection after this many bytes of content
        self.bps: dict[str, int] = {}      # path -> throttle
        self.fail: dict[str, int] = {}     # endpoint kind -> HTTP status
        self.log: list[dict] = []
        self.lock = threading.Lock()
        self.servers: list[ThreadingHTTPServer] = []

    # -- content ---------------------------------------------------------------------------------------------
    def add_commit(self, name: str, files: dict) -> str:
        """files: {repo path: bytes | Path-like/str (a file on disk)}. Returns the 40-hex commit sha."""
        blobs = {}
        for p, v in files.items():
            blobs[p] = Blob(p, data=v) if isinstance(v, (bytes, bytearray)) else Blob(p, src=os.fspath(v))
        sha = commit_sha(blobs)
        self.commits[sha] = blobs
        self.names[name] = sha
        return sha

    @staticmethod
    def folder(path: str) -> dict:
        """{repo path: file path} for every file under a folder (a built release/hf), skipping caches."""
        out = {}
        for d, dirs, files in os.walk(path):
            dirs[:] = [x for x in dirs if x not in (".cache", "__pycache__", ".git")]
            for f in files:
                full = os.path.join(d, f)
                out[os.path.relpath(full, path).replace(os.sep, "/")] = full
        return out

    def set_ref(self, ref: str, name_or_sha: str):
        self.refs[ref] = self.names.get(name_or_sha, name_or_sha)

    def resolve(self, rev: str | None) -> str | None:
        rev = rev or "main"
        if rev in self.commits:
            return rev
        return self.refs.get(rev)

    # -- serving ---------------------------------------------------------------------------------------------
    def start(self, port: int = 0, cdn_port: int = 0, host: str = "127.0.0.1"):
        """The Hub on host:port; the "CDN" on localhost:cdn_port, another host name (huggingface_hub 1.x follows a
        redirect to the same host name, any port), listening on 127.0.0.1 and, when there is one, ::1."""
        handler = type("H", (_Handler,), {"hub": self})
        hub_srv = ThreadingHTTPServer((host, port), handler)
        cdn = [ThreadingHTTPServer(("127.0.0.1", cdn_port), handler)]
        try:
            v6 = type("V6", (ThreadingHTTPServer,), {"address_family": socket.AF_INET6})
            cdn.append(v6(("::1", cdn[0].server_address[1]), handler))
        except OSError:
            pass   # no IPv6 loopback: localhost is 127.0.0.1 only
        for srv in [hub_srv] + cdn:
            srv.daemon_threads = True
            threading.Thread(target=srv.serve_forever, daemon=True, name="fake-hub").start()
            self.servers.append(srv)
        self.endpoint = f"http://{host}:{hub_srv.server_address[1]}"
        self.cdn = f"http://localhost:{cdn[0].server_address[1]}"
        return self

    def stop(self):
        for s in self.servers:
            s.shutdown()
            s.server_close()
        self.servers = []

    def requests(self, method: str | None = None, contains: str = "") -> list[dict]:
        with self.lock:
            return [r for r in self.log if (method is None or r["method"] == method) and contains in r["path"]]


class _Handler(BaseHTTPRequestHandler):
    hub: FakeHub
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    # -- helpers -----------------------------------------------------------------------------------------------
    def _record(self):
        with self.hub.lock:
            self.hub.log.append({"method": self.command, "path": unquote(urlsplit(self.path).path),
                                 "range": self.headers.get("Range"), "t": time.time()})

    def _json(self, obj, code: int = 200, headers: dict | None = None):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _error(self, code: int, error_code: str | None = None, msg: str = ""):
        h = {"X-Error-Code": error_code, "X-Error-Message": msg or error_code} if error_code else {}
        self._json({"error": msg or error_code or "error"}, code, h)

    def _failing(self, kind: str) -> bool:
        code = self.hub.fail.get(kind)
        if code:
            self._error(code, None, f"fake hub: {kind} fails with {code}")
            return True
        return False

    # -- routes ------------------------------------------------------------------------------------------------
    def do_HEAD(self):
        self._route()

    def do_GET(self):
        self._route()

    def _route(self):
        self._record()
        url = urlsplit(self.path)
        path, q = unquote(url.path), parse_qs(url.query, keep_blank_values=True)
        hub = self.hub
        if path.startswith("/_fake/"):
            return self._control(path[len("/_fake/"):], q)
        if path.startswith("/cdn/"):
            return self._cdn(path[len("/cdn/"):])
        repo = hub.repo_id
        api = f"/api/models/{repo}"
        if path == api or path.startswith(api + "/revision/"):
            if self._failing("info"):
                return
            rev = path[len(api + "/revision/"):] if path != api else None
            return self._info(rev, blobs=q.get("blobs", ["false"])[0].lower() == "true")
        if path.startswith(api + "/tree/"):
            if self._failing("tree"):
                return
            rest = path[len(api + "/tree/"):]
            return self._tree(rest, recursive=q.get("recursive", ["false"])[0].lower() == "true")
        if path.startswith(api) or path.startswith("/api/"):
            return self._error(404, "RepoNotFound", "Repository not found")
        m = re.fullmatch(rf"/{re.escape(repo)}/resolve/([^/]+)/(.+)", url.path)   # rev and file still quoted
        if m:
            return self._resolve(unquote(m.group(1)), unquote(m.group(2)))
        return self._error(404, "RepoNotFound", "Repository not found")

    def _control(self, what: str, q):
        hub = self.hub
        get = lambda k: q.get(k, [""])[0]   # noqa: E731
        if what == "main":
            hub.set_ref("main", get("to"))
            return self._json({"main": hub.refs["main"]})
        if what == "fault":
            p = get("path")
            for key, table in (("drop", hub.drop), ("bps", hub.bps)):
                if key in q:
                    v = get(key)
                    table.pop(p, None) if not v else table.__setitem__(p, int(v))
            return self._json({"drop": hub.drop, "bps": hub.bps})
        if what == "fail":
            w, c = get("what"), get("code")
            hub.fail.pop(w, None) if not c else hub.fail.__setitem__(w, int(c))
            return self._json({"fail": hub.fail})
        if what == "log":
            return self._json(hub.requests())
        if what == "commits":
            return self._json({"names": hub.names, "refs": hub.refs})
        return self._error(404)

    def _info(self, rev, blobs: bool):
        sha = self.hub.resolve(rev)
        if sha is None:
            return self._error(404, "RevisionNotFound", f"Invalid rev id: {rev}")
        sibs = []
        for p, b in sorted(self.hub.commits[sha].items()):
            s = {"rfilename": p}
            if blobs:
                s.update(size=b.size, blobId=b.blob_id)
                if b.lfs:
                    s["lfs"] = {"size": b.size, "sha256": b.sha256, "pointerSize": len(b.pointer)}
            sibs.append(s)
        self._json({"_id": "fake", "id": self.hub.repo_id, "modelId": self.hub.repo_id, "author":
                    self.hub.repo_id.split("/")[0], "sha": sha, "lastModified": "2026-10-18T00:00:00.000Z",
                    "private": False, "disabled": False, "gated": False, "downloads": 0, "likes": 0,
                    "tags": [], "library_name": "moth", "siblings": sibs, "createdAt": "2026-10-01T00:00:00.000Z"})

    def _tree(self, rest: str, recursive: bool):
        rev, _, sub = rest.partition("/")
        sha = self.hub.resolve(rev)
        if sha is None:
            return self._error(404, "RevisionNotFound", f"Invalid rev id: {rev}")
        files = self.hub.commits[sha]
        prefix = sub.rstrip("/") + "/" if sub else ""
        out, dirs = [], set()
        for p, b in sorted(files.items()):
            if not p.startswith(prefix):
                continue
            rel = p[len(prefix):]
            parts = rel.split("/")
            for i in range(1, len(parts)):
                d = prefix + "/".join(parts[:i])
                if (recursive or i == 1) and d not in dirs:
                    dirs.add(d)
                    out.append({"type": "directory", "oid": hashlib.sha1(d.encode()).hexdigest(), "size": 0,
                                "path": d})
            if not recursive and len(parts) > 1:
                continue
            e = {"type": "file", "oid": b.blob_id, "size": b.size, "path": p}
            if b.lfs:
                e["lfs"] = {"oid": b.sha256, "size": b.size, "pointerSize": len(b.pointer)}
            out.append(e)
        self._json(out)

    def _resolve(self, rev: str, path: str):
        if self._failing("head" if self.command == "HEAD" else "get"):
            return
        sha = self.hub.resolve(rev)
        if sha is None:
            return self._error(404, "RevisionNotFound", f"Invalid rev id: {rev}")
        b = self.hub.commits[sha].get(path)
        if b is None:
            return self._error(404, "EntryNotFound", f"{path} does not exist on {rev}")
        common = {"X-Repo-Commit": sha, "Accept-Ranges": "bytes"}
        if b.lfs:   # the real Hub: 302 to the CDN, the metadata on the redirect itself
            self.send_response(302)
            self.send_header("Location", f"{self.hub.cdn}/cdn/{b.sha256}?path={quote(path)}")
            self.send_header("ETag", f'"{b.blob_id}"')
            self.send_header("X-Linked-Etag", f'"{b.sha256}"')
            self.send_header("X-Linked-Size", str(b.size))
            self.send_header("Content-Length", "0")
            for k, v in common.items():
                self.send_header(k, v)
            self.end_headers()
            return
        self._content(b, {"ETag": f'"{b.blob_id}"', **common})

    def _cdn(self, sha256: str):
        if self._failing("cdn"):
            return
        for blobs in self.hub.commits.values():
            for b in blobs.values():
                if b.sha256 == sha256 and b.lfs:
                    return self._content(b, {"ETag": f'"{b.sha256}"'})
        self._error(404, None, "no such object")

    def _content(self, b: Blob, headers: dict):
        start, end = 0, b.size
        rng = self.headers.get("Range")
        m = re.fullmatch(r"bytes=(\d+)-(\d*)", rng or "")
        code = 200
        if m and int(m.group(1)) < b.size:
            start, end = int(m.group(1)), (int(m.group(2)) + 1 if m.group(2) else b.size)
            code = 206
        self.send_response(code)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(end - start))
        if code == 206:
            self.send_header("Content-Range", f"bytes {start}-{end - 1}/{b.size}")
        for k, v in headers.items():
            self.send_header(k, v)
        self.end_headers()
        if self.command == "HEAD":
            return
        drop, bps = self.hub.drop.get(b.path), self.hub.bps.get(b.path)
        sent, t0 = 0, time.monotonic()
        try:
            for chunk in b.chunks(start, end):
                if drop is not None and start + sent + len(chunk) > drop:
                    chunk = chunk[:max(0, drop - start - sent)]
                    self.wfile.write(chunk)
                    self.wfile.flush()
                    self.close_connection = True
                    self.connection.shutdown(2)   # mid-transfer: the client sees a short read
                    return
                self.wfile.write(chunk)
                sent += len(chunk)
                if bps:
                    ahead = sent / bps - (time.monotonic() - t0)
                    if ahead > 0:
                        time.sleep(ahead)
        except (BrokenPipeError, ConnectionResetError, OSError):
            self.close_connection = True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--cdn-port", type=int, default=0)
    ap.add_argument("--repo", default="darioooooo0o/moth-112m")
    ap.add_argument("--commit", action="append", default=[], help="NAME=FOLDER (a built Hub folder)")
    ap.add_argument("--main", help="the commit name main points at (default: the last --commit)")
    ap.add_argument("--tag", action="append", default=[], help="TAG=NAME")
    a = ap.parse_args()
    hub = FakeHub(a.repo)
    for c in a.commit:
        name, _, folder = c.partition("=")
        print(f"{name}: {hub.add_commit(name, FakeHub.folder(folder))}  ({folder})", flush=True)
    hub.set_ref("main", a.main or a.commit[-1].partition("=")[0])
    for t in a.tag:
        tag, _, name = t.partition("=")
        hub.set_ref(tag, name)
    hub.start(a.port, a.cdn_port)
    print(f"fake hub for {a.repo} at {hub.endpoint} (cdn {hub.cdn}); main -> {hub.refs['main']}", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        hub.stop()


if __name__ == "__main__":
    main()
