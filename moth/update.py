"""python -m moth update : update a Hugging Face download of moth in place.

  python -m moth update --check     what would change (exit 0 = up to date, 10 = update available, 1 = error)
  python -m moth update [--yes]     fetch the changed files into this same folder (hook paths stay valid)
  python -m moth update --rollback  put back the version before the last update (works offline)

pip install (moth-companion): the code comes from pip, so this only refreshes the model files in
~/.cache/moth/<repo>/ (moth/weights.py), with the same backup and --rollback; --check also says whether newer code
is out (pip install -U moth-companion).

Works on the download folder two levels above this file (moth-112m/, next to code/). It never touches ~/.moth/
(home, token) beyond reading serve's pidfiles, and refuses while a moth from this folder is running.

How an update changes files, so that an interrupted one (network drop, Ctrl-C, a killed process, a full disk) never
leaves a mix of two versions:
  1. the changed files download into <download>/.moth-update/ (kept if it fails: the next run reuses the finished
     ones), and each one is checked against the Hub's size and hash; until then nothing in the install has changed;
  2. a short local swap: each old file moves into <download>/.moth-swap/ (with a plan written first) and the new
     one moves in. If a move fails (a file in use on Windows, a permission error) or the process dies, the swap is
     undone, right away or at the start of the next update / rollback;
  3. the swapped-out files become <download>/.moth-backup/<old version>/ (only the last update's) and
     <download>/.moth-version records the Hub revision you are on and its file list, so files the new version
     dropped can be removed (and nothing else: files you added are never touched).
Shipped files you edited: kept as you have them when the new version didn't change them upstream; replaced when it
did (the update names them, and your copies stay in the backup).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
import urllib.request

from moth import weights

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))   # moth-112m/ in the download: code/moth/update.py
REPO_ID = weights.REPO_ID   # one place for the repo id (moth/weights.py)
WEIGHTS = {"model.safetensors", "config.json", "tokenizer.json", "moth.json"}   # what a pip install fetches
INTEGRATIONS = "code/moth/integrations/"
BACKUP, REVISION_FILE = ".moth-backup", ".moth-version"
STAGING, JOURNAL, LOCK = ".moth-update", ".moth-swap", ".moth-update.lock"
PIP_HINT = "moth update needs huggingface_hub:  pip install -U huggingface_hub"
OFFLINE = "offline mode is on ({}): moth update needs the Hugging Face Hub. Unset it and try again " \
          "(--rollback works offline)"
EXIT_HELP = "exit codes: 0 = done / up to date, 10 = update available (--check), 1 = error or cancelled"
SPARE = 64 << 20   # free space to leave on the disk beyond the download


def version(root: str = ROOT) -> str:
    for path in (os.path.join(root, "code", "moth", "VERSION"), os.path.join(HERE, "VERSION")):
        if os.path.isfile(path):
            with open(path) as f:
                return f.read().strip()
    return "unknown"


def state(root: str = ROOT) -> dict:
    """{"revision", "files"} from .moth-version (files None if unknown: an update then removes nothing); for a plain
    `hf download`, the commit and the files huggingface_hub noted in .cache/huggingface/download/."""
    try:
        with open(os.path.join(root, REVISION_FILE)) as f:
            text = f.read().strip()
        try:
            d = json.loads(text)
            return {"revision": d.get("revision"), "files": d.get("files")}
        except ValueError:
            return {"revision": text or None, "files": None}   # a bare sha
        except AttributeError:   # valid JSON but not an object
            return {"revision": None, "files": None}
    except OSError:
        pass
    meta = os.path.join(root, ".cache", "huggingface", "download")
    files = sorted(os.path.relpath(p, meta).replace(os.sep, "/")[:-len(".metadata")]   # what `hf download` fetched
                   for p in glob.glob(os.path.join(meta, "**", "*.metadata"), recursive=True))
    for name in ("code/moth/VERSION", "config.json", "README.md"):
        try:
            with open(os.path.join(meta, name + ".metadata")) as f:
                return {"revision": f.readline().strip() or None, "files": files or None}
        except OSError:
            continue
    return {"revision": None, "files": None}


def revision(root: str = ROOT) -> str | None:
    return state(root)["revision"]


def version_line(root: str = ROOT) -> str:
    rev = revision(weights.cache_dir() if _pip(root) else root)
    return f"moth {version(root)} (revision {rev[:12] if rev else 'unknown'})"


def _pip(root: str) -> bool:
    """A pip install (weights.mode()): only for the real install, never a folder a caller (or test) passes, and never
    a Hugging Face download whose model.safetensors is still missing (that one gets completed in place)."""
    return root == ROOT and weights.mode() == "pip" and not is_download(root)


def weight_siblings(siblings) -> list:
    return [s for s in siblings if s.rfilename in WEIGHTS]


def is_download(root: str) -> bool:
    """A Hugging Face download of moth: code/moth/ plus the weights next to it, or the marks of one whose weights
    never finished downloading (weights.hf_layout, the same test serve.default_paths() and the pet use, so they
    send such a folder here instead of downloading elsewhere). Keeps the dev checkout (and ~) safe."""
    return weights.hf_layout(root)


def _same_dir(a, b) -> bool:
    return isinstance(a, str) and os.path.realpath(a) == os.path.realpath(b)


def _alive(pid: int) -> bool:
    if not 0 < pid < 2 ** 31:   # a garbled pidfile (and os.kill would overflow)
        return False
    if os.name == "nt":   # os.kill(pid, 0) is no probe there (signal 0 = CTRL_C_EVENT): ask the kernel instead
        return _alive_nt(pid)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except OSError:
        return True   # exists, owned by someone else


def _alive_nt(pid: int) -> bool:
    """Windows: is there a running process with this pid? Unsure (no ctypes, an odd error): True, and /api/health
    decides, as before."""
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        k32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        k32.CloseHandle.argtypes = (wintypes.HANDLE,)
    except (ImportError, OSError, AttributeError):
        return True
    h = k32.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return ctypes.get_last_error() != 87   # ERROR_INVALID_PARAMETER: no such pid; access denied: it exists
    try:
        code = wintypes.DWORD()
        if not k32.GetExitCodeProcess(h, ctypes.byref(code)):
            return True
        return code.value == 259   # STILL_ACTIVE
    finally:
        k32.CloseHandle(h)


def _health(port: int) -> dict | None:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # never through an HTTP proxy
    try:
        with opener.open(f"http://127.0.0.1:{port}/api/health", timeout=1) as r:
            return json.loads(r.read())
    except (OSError, ValueError):
        return None


def server_running(root: str) -> tuple[int, int] | None:
    """(port, pid) of a live moth serving from this install: its ~/.moth/serve-<port>.pid, and /api/health on that
    port answering with the same pid and folder. Stale pidfiles and other programs on the port don't count."""
    for path in sorted(glob.glob(os.path.join(os.path.expanduser("~"), ".moth", "serve-*.pid"))):
        try:
            with open(path) as f:
                d = json.load(f)
            pid, port = int(d["pid"]), int(d["port"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if not _same_dir(d.get("install_root"), root) or not _alive(pid):
            continue
        h = _health(port)
        if isinstance(h, dict) and h.get("ok") and h.get("pid") == pid and _same_dir(h.get("install_root"), root):
            return port, pid
    return None


# --- files --------------------------------------------------------------------------------------------------------
def _x(path: str) -> str:
    """Windows: the extended-length form (\\\\?\\C:\\...) once a path nears MAX_PATH, so a deep file in a deep folder
    works without the LongPathsEnabled switch. Elsewhere, and for short paths: unchanged."""
    if os.name != "nt" or len(path) < 240 or path.startswith("\\\\?\\"):
        return path
    path = os.path.abspath(path)
    return "\\\\?\\UNC\\" + path[2:] if path.startswith("\\\\") else "\\\\?\\" + path


def _join(root: str, rel: str) -> str:
    """root + a repo path ("code/moth/x.py", always "/"): native separators throughout (\\\\?\\ needs them)."""
    return _x(os.path.join(root, *rel.split("/")))


_HASHES: dict = {}


def _file_hash(path: str, lfs: bool) -> str:
    st = os.stat(path)
    key = (os.path.abspath(path), lfs, st.st_size, st.st_mtime_ns)
    if key not in _HASHES:
        h = hashlib.sha256() if lfs else hashlib.sha1(b"blob %d\0" % st.st_size)   # LFS: sha256; else git blob id
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        _HASHES[key] = h.hexdigest()
    return _HASHES[key]


def _matches(path: str, s) -> bool:
    """The file at path is the Hub's file s (size, then LFS sha256 or git blob id)."""
    if not os.path.isfile(path):
        return False
    want = s.lfs["sha256"] if s.lfs else s.blob_id
    size = s.lfs["size"] if s.lfs else s.size
    if size is not None and os.path.getsize(path) != size:
        return False
    return not want or _file_hash(path, bool(s.lfs)) == want


def _safe(rel) -> bool:
    """A repo-relative path that stays inside the download (manifests are data; never trust them blindly) and isn't
    one of moth update's own files or huggingface_hub's cache."""
    return isinstance(rel, str) and bool(rel) and not os.path.isabs(rel) and "\\" not in rel and \
        ":" not in rel and not os.path.normpath(rel).startswith("..") and not rel.startswith((".moth", ".cache"))


def changed_files(root: str, siblings) -> list[str]:
    """Repo files that are missing here or differ (LFS files by content sha256, others by git blob id)."""
    return sorted(s.rfilename for s in siblings if _safe(s.rfilename) and not _matches(_join(root, s.rfilename), s))


def removed_files(root: str, siblings) -> list[str]:
    """Files of the installed revision's manifest that the new revision dropped and that are still here."""
    old = state(root)["files"] or []
    new = {s.rfilename for s in siblings}
    return sorted(f for f in old if _safe(f) and f not in new and os.path.isfile(_join(root, f)))


def _same_blob(a, b) -> bool:
    """Two Hub files (siblings) with the same content."""
    if a.lfs or b.lfs:
        return bool(a.lfs and b.lfs) and a.lfs["sha256"] == b.lfs["sha256"]
    return a.blob_id is not None and a.blob_id == b.blob_id


def _old_siblings(hub, repo: str, mine: str | None, sha: str) -> dict | None:
    """{path: sibling} of the installed revision, to tell your edits from upstream changes; None if unknown."""
    if not mine or mine == sha:
        return None
    try:
        return {s.rfilename: s for s in hub.HfApi().model_info(repo, revision=mine, files_metadata=True).siblings}
    except Exception:   # gone from the Hub, offline halfway ...
        return None


def _sort_edits(root: str, files: list[str], new: dict, old: dict | None) -> tuple[list[str], list[str], list[str]]:
    """(files to fetch, kept, edited). A shipped file you changed that the new revision didn't change upstream is kept
    as you have it; one that it did change (or a file of yours where the new revision adds one) is replaced, and named
    so you know (your copy goes to the backup). Without the installed revision's file list everything is replaced."""
    if old is None:
        return files, [], []
    fetch, kept, edited = [], [], []
    for f in files:
        here = os.path.isfile(_join(root, f))
        if here and f in old and _same_blob(old[f], new[f]):
            kept.append(f)
            continue
        fetch.append(f)
        if here and (f not in old or not _matches(_join(root, f), old[f])):
            edited.append(f)
    return fetch, kept, edited


def _say_edits(kept: list[str], edited: list[str]):
    if kept:
        print(f"kept your changes to {', '.join(kept)} (not changed upstream)")
    if edited:
        print(f"you changed {', '.join(edited)}: the update replaces {'them' if len(edited) > 1 else 'it'} "
              f"(your copy is kept in {BACKUP}/)")


def describe(files: list[str], removed: list[str] = ()) -> str:
    allf = list(files) + list(removed)
    weights = any(f in WEIGHTS for f in allf)
    code = any(f.startswith("code/") for f in allf)
    what = "weights and code" if weights and code else "weights" if weights else "code" if code else "docs"
    out = f"{what} ({len(files)} file{'s' * (len(files) != 1)}: {', '.join(files[:8])}{' ...' * (len(files) > 8)})"
    return out + (f"; removed: {', '.join(removed)}" if removed else "")


def changelog(readme: str, max_lines: int = 25) -> str:
    """The model card's '## Changelog' section, if it has one."""
    m = re.search(r"^(#+)\s*change\s*log\b.*$", readme, re.I | re.M)
    if not m:
        return ""
    rest = readme[m.end():]
    end = re.search(rf"^#{{1,{len(m.group(1))}}}\s", rest, re.M)
    lines = rest[:end.start() if end else None].strip().splitlines()
    return "\n".join(lines[:max_lines] + (["..."] if len(lines) > max_lines else []))


def _remote_text(hub, repo: str, sha: str, filename: str) -> str:
    """A small file at that revision, into a throwaway cache (never into the install)."""
    try:
        with tempfile.TemporaryDirectory() as tmp:
            with open(hub.hf_hub_download(repo, filename, revision=sha, cache_dir=tmp), encoding="utf-8") as f:
                return f.read()
    except Exception:
        return ""


def _hub():
    try:
        import huggingface_hub
        return huggingface_hub
    except ImportError:
        return None


def _offline() -> str | None:
    """Which offline switch is on (HF_HUB_OFFLINE, MOTH_OFFLINE; the same ones moth.config.offline() reads)."""
    from moth import config
    if config.offline():
        return " / ".join(k for k in ("HF_HUB_OFFLINE", "MOTH_OFFLINE") if config.parse_bool(os.environ.get(k, "")))
    return None


def _first_line(e: BaseException) -> str:
    return str(e).splitlines()[0][:160] if str(e) else ""


def _latest(hub, repo: str, rev: str | None):
    try:
        return hub.HfApi().model_info(repo, revision=rev, files_metadata=True), None
    except Exception as e:   # offline, DNS, timeout, 404 ...
        name = type(e).__name__
        if "NotFound" in name:
            return None, f"{repo}{'@' + rev if rev else ''} not found on the Hugging Face Hub ({name})"
        return None, f"can't reach the Hugging Face Hub ({name}: {_first_line(e)}); check your connection and try " \
                     "again. Nothing was changed."


def check(root: str, repo: str, rev: str | None, pip: bool = False) -> int:
    if not pip and not is_download(root):
        print(f"{root} isn't a Hugging Face download of moth (no model.safetensors next to code/); nothing to check")
        return 1
    off = _offline()
    if off:
        print(OFFLINE.format(off))
        return 1
    hub = _hub()
    if hub is None:
        print(PIP_HINT)
        return 1
    info, err = _latest(hub, repo, rev)
    if err:
        print(err)
        return 1
    sibs = weight_siblings(info.siblings) if pip else info.siblings
    mine = revision(root)
    files, removed, kept, edited = _plan(hub, repo, root, sibs, mine, info.sha)
    if os.path.isdir(os.path.join(root, JOURNAL)):
        print("an earlier update or rollback stopped halfway: the next  python -m moth update  undoes it first")
    if pip:
        return _pip_check(root, hub, repo, info, files, removed)
    if not files and not removed:
        print(f"up to date: {version_line(root)}")
        _say_edits(kept, [])
        return 0
    if mine == info.sha:   # same revision, files gone (an interrupted `hf download`, a deleted file)
        print(f"files missing from this download of {version_line(root)}: {describe(files)}")
        print("\nrun  python -m moth update  to fetch them")
        return 10
    if docs_only(files, removed):   # a card edit is no new moth: no "update available" (nor the pet's bubble)
        print(f"up to date: {version_line(root)}  (only the docs changed upstream: {', '.join((files + removed)[:4])}"
              f"{' ...' * (len(files) + len(removed) > 4)}; python -m moth update fetches them)")
        return 0
    new = _remote_text(hub, repo, info.sha, "code/moth/VERSION").strip() or "?"
    print(f"update available: moth {version(root)} -> {new}  (revision {(mine or 'unknown')[:12]} -> {info.sha[:12]})")
    print(f"changes: {describe(files, removed)}")
    _say_edits(kept, edited)
    log = changelog(_remote_text(hub, repo, info.sha, "README.md"))
    if log:
        print("\nchangelog:\n" + log)
    print("\nrun  python -m moth update  to install it")
    return 10


def _plan(hub, repo: str, root: str, sibs, mine: str | None, sha: str):
    """(files to fetch, files to remove, kept, edited). Already on that revision: only files that went missing (your
    edits of shipped files stay); otherwise what differs from the Hub (minus your edits of files upstream didn't
    change, see _sort_edits) and what the new revision dropped."""
    if mine == sha:
        missing = [s.rfilename for s in sibs if _safe(s.rfilename) and not os.path.isfile(_join(root, s.rfilename))]
        return sorted(missing), [], [], []
    new = {s.rfilename: s for s in sibs}
    files, kept, edited = _sort_edits(root, changed_files(root, sibs), new, _old_siblings(hub, repo, mine, sha))
    return files, removed_files(root, sibs), kept, edited


def _newer(a: str, b: str) -> bool:
    from moth.backends import compat   # stdlib only
    return compat.newer(a, b)


def docs_only(files, removed=()) -> bool:
    """Only the model card / docs differ (README.md, LICENSE, NOTICE, *.md under code/): nothing moth runs."""
    allf = list(files) + list(removed)
    return bool(allf) and all(f not in WEIGHTS and (not f.startswith("code/") or f.endswith(".md")) for f in allf)


def _too_new(hub, repo: str, info) -> str | None:
    """A pip / app install fetches only the model files: why this code can't run the Hub's model, if it can't."""
    from moth.backends import compat
    return compat.remote_problem(hub, repo, info.sha, {s.rfilename for s in info.siblings})


def _pip_check(cache: str, hub, repo: str, info, files: list[str], removed: list[str]) -> int:
    from moth.backends import compat
    rc, mine = 0, revision(cache)
    why = _too_new(hub, repo, info) if files or removed else None
    if why:
        print(f"a newer model is on the Hub (revision {info.sha[:12]}), but it needs newer moth code: {why}")
        print(f"newer code: {compat.upgrade_hint()}, then  moth update  fetches the model")
        return 10
    if files or removed:
        rc = 10
        print(f"newer weights on the Hub: revision {(mine or 'none')[:12]} -> {info.sha[:12]}  ({repo})")
        print(f"changes: {describe(files, removed)}")
        log = changelog(_remote_text(hub, repo, info.sha, "README.md"))
        if log:
            print("\nchangelog:\n" + log)
        print("\nrun  moth update  to fetch them (into " + cache + ")")
    else:
        print(f"weights up to date: revision {info.sha[:12]} ({repo})")
    new = _remote_text(hub, repo, info.sha, "code/moth/VERSION").strip()
    if _newer(version(), new):
        rc = 10
        print(f"newer code: moth {version()} -> {new}:  {compat.upgrade_hint()}")
    else:
        print(f"code: moth {version()}; {compat.upgrade_hint(short=True)} gets the newest release")
    return rc


# --- backup, swap, undo -------------------------------------------------------------------------------------------
def _backup_dir(root: str) -> str | None:
    base = os.path.join(root, BACKUP)
    dirs = [d for d in os.listdir(base) if os.path.isfile(os.path.join(base, d, "manifest.json"))] \
        if os.path.isdir(base) else []
    return os.path.join(base, dirs[0]) if dirs else None


def _write_json(path: str, obj):
    """Write, flush to disk, then rename over: a crash leaves the old file or the new one, never half of one."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _write_state(root: str, st: dict | None):
    path = os.path.join(root, REVISION_FILE)
    if st and st.get("revision"):
        _write_json(path, {"revision": st["revision"], "files": st.get("files")})
    elif os.path.exists(path):
        os.remove(path)


def _drop_hf_metadata(root: str, files):
    """hf's local-dir metadata for files moth update changed would claim the wrong commit."""
    for f in files:
        meta = _join(root, ".cache/huggingface/download/" + f + ".metadata")
        if os.path.exists(meta):
            os.remove(meta)


def _norm(path: str) -> str:
    return os.path.normcase(os.path.abspath(path[4:] if path.startswith("\\\\?\\") else path))


def _prune(root: str, path: str):
    """Remove the folders a moved-away file left empty, up to (never including) root."""
    top, d = _norm(root), os.path.dirname(path)
    while _norm(d) != top and _norm(d).startswith(top + os.sep) and os.path.isdir(d) and not os.listdir(d):
        os.rmdir(d)
        d = os.path.dirname(d)


def _move(src: str, dst: str):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for i in range(3 if os.name == "nt" else 1):
        try:
            return os.replace(src, dst)
        except PermissionError:   # Windows: a virus scanner may hold a just-written file for a moment
            if i == 2 or os.name != "nt":
                raise
            time.sleep(0.3)


def _rmtree(path: str):
    if os.path.isdir(path):
        shutil.rmtree(_x(path), ignore_errors=True)


def _swap(root: str, kind: str, put: list[str], drop: list[str], incoming: str, ver: str, st: dict) -> dict:
    """Move put (from root/incoming) into root and drop out of it; every file it replaces or drops goes to
    root/.moth-swap/files/. The plan is on disk before the first move, so _undo can put everything back after a
    failure here or a crash (the next run calls _recover). Returns the plan (the backup manifest)."""
    j = os.path.join(root, JOURNAL)
    _rmtree(j)
    replaced = [f for f in put if os.path.isfile(_join(root, f))]
    plan = {"kind": kind, "version": ver, "state": st, "replaced": replaced,
            "added": [f for f in put if f not in replaced], "removed": list(drop), "incoming": incoming}
    os.makedirs(j)
    _write_json(os.path.join(j, "manifest.json"), plan)
    src = os.path.join(root, *incoming.split("/"))
    try:
        for f in drop:
            _move(_join(root, f), _join(j, "files/" + f))
            _prune(root, _join(root, f))
        for f in put:
            if f in replaced:
                _move(_join(root, f), _join(j, "files/" + f))
            _move(_join(src, f), _join(root, f))
    except BaseException:
        _undo(root)
        raise
    return plan


def _undo(root: str) -> dict | None:
    """Put back what a _swap moved (the new files back where they came from). Raises if a file can't be moved; the
    plan then stays on disk for the next try."""
    j = os.path.join(root, JOURNAL)
    try:
        with open(os.path.join(j, "manifest.json")) as f:
            plan = json.load(f)
    except (OSError, ValueError):
        _rmtree(j)   # died before the plan was written: nothing was moved yet
        return None
    src = os.path.join(root, *str(plan.get("incoming", STAGING)).split("/"))
    replaced = [f for f in plan.get("replaced", []) if _safe(f)]
    for f in [f for f in plan.get("added", []) if _safe(f)] + replaced:
        moved_out = f not in replaced or os.path.isfile(_join(j, "files/" + f))
        if moved_out and os.path.isfile(_join(root, f)):   # the incoming copy: back where it came from
            _move(_join(root, f), _join(src, f))
            _prune(root, _join(root, f))
    for f in replaced + [f for f in plan.get("removed", []) if _safe(f)]:
        if os.path.isfile(_join(j, "files/" + f)):
            _move(_join(j, "files/" + f), _join(root, f))
    _rmtree(j)
    return plan


def _recover(root: str):
    """Start of an update / rollback: undo a swap that a crash (or a failed undo) left halfway."""
    if not os.path.isdir(os.path.join(root, JOURNAL)):
        return
    plan = _undo(root)
    if plan:
        print(f"undid an {plan.get('kind', 'update')} that stopped halfway: back on moth {version(root)}")


class _Lock:
    """<root>/.moth-update.lock while an update or rollback runs: a second one at the same time would undo the
    first one's swap. A lock whose process is gone is taken over."""

    def __init__(self, root: str):
        self.path = os.path.join(root, LOCK)

    def __enter__(self):
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w") as f:
                    f.write(str(os.getpid()))
                return self
            except FileExistsError:
                try:
                    with open(self.path) as f:
                        pid = int(f.read().strip() or 0)
                except (OSError, ValueError):
                    pid = 0
                if pid and pid != os.getpid() and _alive(pid):
                    raise RuntimeError(f"another moth update is running (pid {pid}); wait for it to finish")
                os.remove(self.path)
        raise RuntimeError(f"can't take {self.path}")

    def __exit__(self, *exc):
        try:
            os.remove(self.path)
        except OSError:
            pass


def _forget_journal(root: str):
    """The swap is done for good: the plan goes first, so whatever rmtree leaves can never be "undone"."""
    try:
        os.remove(os.path.join(root, JOURNAL, "manifest.json"))
    except OSError:
        pass
    _rmtree(os.path.join(root, JOURNAL))


def _commit_backup(root: str, plan: dict) -> bool:
    """The swapped-out files become the backup (replacing the previous update's). False if that failed (then there
    is no backup, but the update stands)."""
    base = os.path.join(root, BACKUP)
    try:
        if os.path.isdir(base):
            old = base + ".old"
            _rmtree(old)
            os.replace(base, old)
            _rmtree(old)
        os.makedirs(base)
        os.replace(os.path.join(root, JOURNAL), os.path.join(base, re.sub(r"[^\w.+-]", "_", plan["version"]) or "_"))
        return True
    except OSError:
        _forget_journal(root)
        return False


def _not_safe(root: str, pip: bool = False) -> str | None:
    if not pip and not is_download(root):
        return f"{root} isn't a Hugging Face download of moth (no model.safetensors next to code/); nothing to update"
    running = server_running(ROOT if pip else root)   # pip: serve's pidfile names the install, not the cache
    if running:
        return f"moth is running from this {'install' if pip else 'folder'} (port {running[0]}, pid {running[1]}): " \
               "stop moth first, then run the update again"
    return None


def _confirm(question: str) -> bool:
    try:
        return input(question + " [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def _os_error(e: OSError, what: str) -> str:
    where = e.filename or ""
    hint = " (a full disk: free some space)" if getattr(e, "errno", None) == 28 else \
        " (is the file open in another program, or read-only? close moth, the pet and editors)" \
        if isinstance(e, PermissionError) else ""
    return f"{what} failed: {e.strerror or type(e).__name__}{': ' + str(where) if where else ''}{hint}"


def rollback(root: str, yes: bool, pip: bool = False) -> int:
    err = _not_safe(root, pip)
    if err:
        print(err)
        return 1
    if not os.path.isdir(root):
        print("no backup to roll back to (nothing was updated yet)")
        return 1
    try:
        with _Lock(root):
            _recover(root)
            return _rollback(root, yes, pip)
    except RuntimeError as e:
        print(e)
        return 1
    except KeyboardInterrupt:
        print(f"\ninterrupted: still on moth {version(root)}")
        return 1
    except OSError as e:
        print(_os_error(e, "rollback") + f"; still on moth {version(root)}")
        return 1


def _rollback(root: str, yes: bool, pip: bool = False) -> int:
    dest = _backup_dir(root)
    if not dest:
        print(f"no backup to roll back to (only the last update keeps one, and a rollback uses it up); you're on "
              f"{version_line(root)}. {'moth update' if pip else 'python -m moth update'} gets the newest again")
        return 1
    with open(os.path.join(dest, "manifest.json")) as f:
        m = json.load(f)
    if not yes and not _confirm(f"roll moth {version(root)} back to {m['version']}?"):
        print("cancelled")
        return 1
    files = os.path.relpath(os.path.join(dest, "files"), root).replace(os.sep, "/")
    put = [f for f in m["replaced"] + m.get("removed", []) if _safe(f) and os.path.isfile(_join(dest, "files/" + f))]
    drop = [f for f in m["added"] if _safe(f) and os.path.isfile(_join(root, f))]
    _swap(root, "rollback", put, drop, files, version(root), state(root))
    _write_state(root, m.get("state"))
    _forget_journal(root)
    try:
        os.remove(os.path.join(dest, "manifest.json"))   # first: a half-deleted backup can't be used again
    except OSError:
        pass
    _rmtree(os.path.join(root, BACKUP))
    _drop_hf_metadata(root, put + drop)
    print(f"rolled back to moth {m['version']}: restored {len(put)} file(s), removed {len(drop)}")
    return 0


def _glob_escape(path: str) -> str:
    """An allow_patterns entry (fnmatch) that matches exactly this path."""
    return re.sub(r"([*?\[])", r"[\1]", path)


def _stage(hub, repo: str, sha: str, root: str, files: list[str], sibs: dict) -> list[str]:
    """Download files into root/.moth-update/ (resuming what an earlier try left there); the ones that didn't arrive
    intact. huggingface_hub can return without an error when the Hub stops answering halfway, so check each."""
    stage = os.path.join(root, STAGING)
    for f in files:   # a leftover that isn't this file: fetch it again
        p = _join(stage, f)
        if os.path.exists(p) and not _matches(p, sibs[f]):
            os.remove(p)
            _drop_hf_metadata(stage, [f])
    os.makedirs(stage, exist_ok=True)
    weights.drop_orphans(stage, min_age=0)   # partial files killed tries left (we hold the lock: none is live)
    hub.snapshot_download(repo_id=repo, local_dir=stage, revision=sha, allow_patterns=[_glob_escape(f) for f in files])
    return [f for f in files if not _matches(_join(stage, f), sibs[f])]


def _need(root: str, files: list[str], sibs: dict) -> int:
    stage = os.path.join(root, STAGING)
    total = 0
    for f in files:
        s = sibs[f]
        size = (s.lfs["size"] if s.lfs else s.size) or 0
        p = _join(stage, f)
        total += 0 if os.path.isfile(p) and os.path.getsize(p) == size else size
    return total


def update(root: str, repo: str, rev: str | None, yes: bool, pip: bool = False) -> int:
    err = _not_safe(root, pip)
    if err:
        print(err)
        return 1
    off = _offline()
    if off:
        print(OFFLINE.format(off))
        return 1
    hub = _hub()
    if hub is None:
        print(PIP_HINT)
        return 1
    os.makedirs(root, exist_ok=True)
    try:
        with _Lock(root):
            _recover(root)
            return _update(hub, root, repo, rev, yes, pip)
    except RuntimeError as e:
        print(e)
        return 1
    except KeyboardInterrupt:
        print(f"\ninterrupted: nothing was changed, still on moth {version(root)}; run the update again to resume")
        return 1
    except OSError as e:
        print(_os_error(e, "update") + f"; nothing was changed, still on moth {version(root)}")
        return 1


def _update(hub, root: str, repo: str, rev: str | None, yes: bool, pip: bool) -> int:
    info, err = _latest(hub, repo, rev)
    if err:
        print(err)
        return 1
    sibs = weight_siblings(info.siblings) if pip else info.siblings   # pip: only the model files, never code/
    if pip and not sibs:
        print(f"{repo} has none of {', '.join(sorted(WEIGHTS))}; is it a moth repo?")
        return 1
    by_name = {s.rfilename: s for s in sibs}
    manifest = sorted(by_name)
    mine = revision(root)
    files, removed, kept, edited = _plan(hub, repo, root, sibs, mine, info.sha)
    if not files and not removed:
        _write_state(root, {"revision": info.sha, "files": manifest})
        _rmtree(os.path.join(root, STAGING))
        print(f"up to date: {version_line(root)}")
        _say_edits(kept, [])
        return 0
    why = _too_new(hub, repo, info) if pip else None
    if why:   # the model files alone would leave this code unable to start moth
        from moth.backends import compat
        print(f"the model at revision {info.sha[:12]} needs newer moth code ({why}): {compat.upgrade_hint()}, then "
              "run  moth update  again. Nothing was changed.")
        return 1
    print(f"update {root}: {describe(files, removed)}  (revision {(mine or 'unknown')[:12]} -> {info.sha[:12]})")
    _say_edits(kept, edited)
    try:
        free = shutil.disk_usage(root).free
    except OSError:
        free = None
    need = _need(root, files, by_name)
    if free is not None and free < need + SPARE:
        print(f"not enough disk space in {root}: the download needs {(need + SPARE) >> 20} MB, {free >> 20} MB free. "
              "Nothing was changed.")
        return 1
    if not yes and not _confirm("download it now?"):
        print("cancelled")
        return 1
    old = version(root)
    try:   # pinned to the sha that was compared
        bad = _stage(hub, repo, info.sha, root, files, by_name)
    except Exception as e:
        print(f"download failed ({type(e).__name__}: {_first_line(e)}); nothing was changed. Run the update again "
              "to resume")
        return 1
    if bad:
        print(f"download incomplete ({', '.join(bad[:4])}{' ...' * (len(bad) > 4)} didn't arrive intact; the Hub may "
              "have stopped answering); nothing was changed. Run the update again to resume")
        return 1
    try:
        plan = _swap(root, "update", files, removed, STAGING, old, state(root))
    except OSError as e:
        print(_os_error(e, "replacing the files") + f"; nothing was changed, still on moth {old}")
        return 1
    if mine == info.sha or (mine is None and not plan["replaced"] and not plan["removed"]):
        # a repair (missing files of this same revision) or the first download: nothing to go back to, and the last
        # real update's backup stays the one --rollback uses
        _forget_journal(root)
        backed_up = None
    else:
        backed_up = _commit_backup(root, plan)
    _write_state(root, {"revision": info.sha, "files": manifest})
    _drop_hf_metadata(root, files + removed)
    _rmtree(os.path.join(root, STAGING))
    if pip:
        print(f"updated the weights in {root} to revision {info.sha[:12]}: {describe(files, removed)}")
        if backed_up is not None:
            print(f"previous weights kept in {os.path.join(root, BACKUP)}  (moth update --rollback)" if backed_up
                  else "couldn't keep a backup of the previous weights this time (no --rollback)")
        from moth.backends import compat
        print(f"newer code, when there is some:  {compat.upgrade_hint(short=True)}")
        return 0
    print(f"updated: moth {old} -> {version(root)} (revision {info.sha[:12]})")
    print(f"changed: {describe(files, removed)}")
    if backed_up is not None:
        print(f"previous version kept in {os.path.join(root, BACKUP)}  (python -m moth update --rollback)"
              if backed_up else "couldn't keep a backup of the previous version this time (no --rollback)")
    if any(f.startswith(INTEGRATIONS) for f in files + removed):
        print("integration files changed: re-copy any adapter you copied elsewhere (Pi moth.ts, OpenCode moth.js, "
              "the Hermes plugin folder). Hooks that point into code/moth/integrations keep working.")
    return 0


def model_target():
    """What `moth config model` means for moth update: None = this install's own moth (as always); (folder, repo) for
    a tiny-agent model moth downloaded from a Hub repo (its manifest's repo id; updated like a pip install's weights:
    model files only, with a backup); a str = why moth update leaves it alone."""
    from moth import backends, config
    spec = config.model()
    if not spec:
        return None
    try:
        m = backends.resolve(spec, network=False)
    except backends.ManifestError as e:
        return f"can't read the configured model ({e})"
    if m.backend == "openai":
        return (f"your model is {spec}, an OpenAI-compatible server, not a moth model from the Hugging Face Hub: "
                "update it where it runs (e.g. `ollama pull`, a newer GGUF). `moth config model default` goes back "
                "to moth's own model.")
    if m.backend == "transformers":
        src = m.repo or m.model or m.folder
        how = f"`hf download {src}` refreshes it" if src and backends.HUB_ID.fullmatch(src) else "update it by hand"
        return f"your model {src} runs on transformers, which keeps it itself: {how}."
    repo = m.repo
    if repo and m.folder and (_same_dir(m.folder, weights.cache_dir(repo))
                              or os.path.isfile(os.path.join(m.folder, REVISION_FILE))):
        return m.folder, repo
    return (f"your model {m.folder} is a folder moth didn't download from the Hub (no repo id in its moth.json, or no "
            f"{REVISION_FILE}): update it the way you made it.")


def main(argv=None, root: str = ROOT) -> int:
    ap = argparse.ArgumentParser(prog="python -m moth update", epilog=EXIT_HELP,
                                 description="Update this Hugging Face download of moth in place (stop moth first). "
                                             "~/.moth (home, token) is never touched.")
    ap.add_argument("--check", action="store_true", help="only show what would change (exit 10 = update available)")
    ap.add_argument("--rollback", action="store_true", help="restore the version before the last update")
    ap.add_argument("--repo", default=weights.repo(), help=f"default {REPO_ID} (or $MOTH_REPO)")
    ap.add_argument("--revision", default=weights.pinned_revision(),
                    help="branch, tag or commit (default: $MOTH_REVISION, else main)")
    ap.add_argument("--yes", "-y", action="store_true", help="don't ask before changing files")
    args = list(sys.argv[1:] if argv is None else argv)
    a = ap.parse_args(args)
    pip = _pip(root)
    target = None if any(x == "--repo" or x.startswith("--repo=") for x in args) else model_target()
    if isinstance(target, str):   # `moth config model` is something moth update doesn't manage: say so, then do
        # this install's own update as always (its code, and moth's own model), so code updates and the pet's
        # daily "new moth out" check keep working
        print(f"moth update: {target} (this updates moth's own install, as usual.)", file=sys.stderr, flush=True)
        target = None
    if target:   # a tiny-agent model from another Hub repo, in the folder moth downloaded it to
        root, a.repo, pip = target[0], target[1], True
    elif pip:   # pip install: the weights cache is what gets updated
        root = weights.cache_dir(a.repo)
    if a.rollback:
        return rollback(root, a.yes, pip)
    if a.check:
        return check(root, a.repo, a.revision, pip)
    return update(root, a.repo, a.revision, a.yes, pip)


if __name__ == "__main__":
    sys.exit(main())
