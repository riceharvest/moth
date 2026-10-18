"""python -m moth update : update a Hugging Face download of moth in place.

  python -m moth update --check     what would change (exit 0 = up to date, 10 = update available, 1 = error)
  python -m moth update [--yes]     fetch the changed files into this same folder (hook paths stay valid)
  python -m moth update --rollback  put back the version before the last update

pip install (moth-companion): the code comes from pip, so this only refreshes the model files in
~/.cache/moth/<repo>/ (moth/weights.py), with the same backup and --rollback; --check also says whether newer code
is out (pip install -U moth-companion).

Works on the download folder two levels above this file (moth-112m/, next to code/). It never touches ~/.moth/
(home, token) beyond reading serve's pidfiles. Replaced and removed files are kept in
<download>/.moth-backup/<old version>/ (only the last update's). <download>/.moth-version holds the Hub revision
you are on and its file list, so files the new version dropped can be removed (and nothing else).
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
import urllib.request

from moth import weights

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))   # moth-112m/ in the download: code/moth/update.py
REPO_ID = weights.REPO_ID   # one place for the repo id (moth/weights.py)
WEIGHTS = {"model.safetensors", "config.json", "tokenizer.json"}
INTEGRATIONS = "code/moth/integrations/"
BACKUP, REVISION_FILE = ".moth-backup", ".moth-version"
PIP_HINT = "moth update needs huggingface_hub:  pip install -U huggingface_hub"
EXIT_HELP = "exit codes: 0 = done / up to date, 10 = update available (--check), 1 = error or cancelled"


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
    """A pip install (weights.mode()): only for the real install, never a folder a caller (or test) passes."""
    return root == ROOT and weights.mode() == "pip"


def weight_siblings(siblings) -> list:
    return [s for s in siblings if s.rfilename in WEIGHTS]


def is_download(root: str) -> bool:
    """Same test as serve.default_paths(): weights next to code/. Keeps the dev checkout (and ~) safe."""
    return os.path.isfile(os.path.join(root, "model.safetensors")) and os.path.isdir(os.path.join(root, "code", "moth"))


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


def _file_hash(path: str, lfs: bool) -> str:
    h = hashlib.sha256() if lfs else hashlib.sha1(b"blob %d\0" % os.path.getsize(path))   # git blob id
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe(rel) -> bool:
    """A repo-relative path that stays inside the download (manifests are data; never trust them blindly)."""
    return isinstance(rel, str) and rel and not os.path.isabs(rel) and \
        not os.path.normpath(rel).startswith("..") and not rel.startswith((BACKUP, REVISION_FILE, ".cache"))


def changed_files(root: str, siblings) -> list[str]:
    """Repo files that are missing here or differ (LFS files by content sha256, others by git blob id)."""
    out = []
    for s in siblings:
        if not _safe(s.rfilename):
            continue
        path = os.path.join(root, s.rfilename)
        if not os.path.isfile(path):
            out.append(s.rfilename)
            continue
        want = s.lfs["sha256"] if s.lfs else s.blob_id
        size = s.lfs["size"] if s.lfs else s.size
        if size is not None and os.path.getsize(path) != size:
            out.append(s.rfilename)
        elif want and _file_hash(path, bool(s.lfs)) != want:
            out.append(s.rfilename)
    return sorted(out)


def removed_files(root: str, siblings) -> list[str]:
    """Files of the installed revision's manifest that the new revision dropped and that are still here."""
    old = state(root)["files"] or []
    new = {s.rfilename for s in siblings}
    return sorted(f for f in old if _safe(f) and f not in new and os.path.isfile(os.path.join(root, f)))


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
            with open(hub.hf_hub_download(repo, filename, revision=sha, cache_dir=tmp)) as f:
                return f.read()
    except Exception:
        return ""


def _hub():
    try:
        import huggingface_hub
        return huggingface_hub
    except ImportError:
        return None


def _latest(hub, repo: str, rev: str | None):
    try:
        return hub.HfApi().model_info(repo, revision=rev, files_metadata=True), None
    except Exception as e:   # offline, DNS, timeout, 404, HF_HUB_OFFLINE=1 ...
        name = type(e).__name__
        if "NotFound" in name:
            return None, f"{repo}{'@' + rev if rev else ''} not found on the Hugging Face Hub ({name})"
        return None, f"can't reach the Hugging Face Hub ({name}: {str(e).splitlines()[0][:120] if str(e) else ''}); " \
                     "check your connection and try again"


def check(root: str, repo: str, rev: str | None, pip: bool = False) -> int:
    if not pip and not is_download(root):
        print(f"{root} isn't a Hugging Face download of moth (no model.safetensors next to code/); nothing to check")
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
    files, removed = ([], []) if mine == info.sha else (changed_files(root, sibs), removed_files(root, sibs))
    if pip:
        return _pip_check(root, hub, repo, info, files, removed)
    if not files and not removed:
        print(f"up to date: {version_line(root)}")
        return 0
    new = _remote_text(hub, repo, info.sha, "code/moth/VERSION").strip() or "?"
    print(f"update available: moth {version(root)} -> {new}  (revision {(mine or 'unknown')[:12]} -> {info.sha[:12]})")
    print(f"changes: {describe(files, removed)}")
    log = changelog(_remote_text(hub, repo, info.sha, "README.md"))
    if log:
        print("\nchangelog:\n" + log)
    print("\nrun  python -m moth update  to install it")
    return 10


def _newer(a: str, b: str) -> bool:
    """Version b is newer than a (dotted numbers; anything else: just different)."""
    try:
        return tuple(map(int, b.split("."))) > tuple(map(int, a.split(".")))
    except ValueError:
        return bool(b) and b != a


def _pip_check(cache: str, hub, repo: str, info, files: list[str], removed: list[str]) -> int:
    rc, mine = 0, revision(cache)
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
        print(f"newer code: moth {version()} -> {new}:  pip install -U moth-companion  "
              "(uv tool upgrade moth-companion / pipx upgrade moth-companion)")
    else:
        print(f"code: moth {version()}; pip install -U moth-companion gets the newest release")
    return rc


def _backup_dir(root: str) -> str | None:
    base = os.path.join(root, BACKUP)
    dirs = [d for d in os.listdir(base) if os.path.isfile(os.path.join(base, d, "manifest.json"))] \
        if os.path.isdir(base) else []
    return os.path.join(base, dirs[0]) if dirs else None


def _write_state(root: str, st: dict | None):
    path = os.path.join(root, REVISION_FILE)
    if st and st.get("revision"):
        with open(path, "w") as f:
            json.dump({"revision": st["revision"], "files": st.get("files")}, f, indent=1)
            f.write("\n")
    elif os.path.exists(path):
        os.remove(path)


def _drop_hf_metadata(root: str, files):
    """hf's local-dir metadata would claim the new commit and make the next update skip these files."""
    for f in files:
        meta = os.path.join(root, ".cache", "huggingface", "download", f + ".metadata")
        if os.path.exists(meta):
            os.remove(meta)


def _remove(root: str, rel: str):
    path = os.path.join(root, rel)
    if os.path.isfile(path):
        os.remove(path)
    d = os.path.dirname(path)
    while os.path.realpath(d) != os.path.realpath(root) and os.path.isdir(d) and not os.listdir(d):
        os.rmdir(d)   # folders the file leaves empty, never the download itself
        d = os.path.dirname(d)


def backup(root: str, files: list[str], removed: list[str] = ()) -> str:
    """Copy what an update replaces or removes to .moth-backup/<old version>/ (dropping any older backup)."""
    shutil.rmtree(os.path.join(root, BACKUP), ignore_errors=True)
    old = version(root)
    dest = os.path.join(root, BACKUP, re.sub(r"[^\w.+-]", "_", old))
    replaced = [f for f in files if os.path.isfile(os.path.join(root, f))]
    for f in replaced + list(removed):
        os.makedirs(os.path.dirname(os.path.join(dest, "files", f)), exist_ok=True)
        shutil.copy2(os.path.join(root, f), os.path.join(dest, "files", f))
    manifest = {"version": old, "state": state(root), "replaced": replaced,
                "added": [f for f in files if f not in replaced], "removed": list(removed)}
    os.makedirs(dest, exist_ok=True)
    with open(os.path.join(dest, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    return dest


def restore(root: str) -> dict | None:
    """Undo the last update from its backup; the backup is used up."""
    dest = _backup_dir(root)
    if not dest:
        return None
    with open(os.path.join(dest, "manifest.json")) as f:
        m = json.load(f)
    back = [f for f in m["replaced"] + m.get("removed", []) if _safe(f)]
    _drop_hf_metadata(root, back + [f for f in m["added"] if _safe(f)])
    for f in back:
        os.makedirs(os.path.dirname(os.path.join(root, f)), exist_ok=True)
        shutil.copy(os.path.join(dest, "files", f), os.path.join(root, f))   # copy, not copy2: a fresh mtime
    for f in m["added"]:
        if _safe(f):
            _remove(root, f)
    _write_state(root, m.get("state"))
    shutil.rmtree(os.path.join(root, BACKUP), ignore_errors=True)
    return m


def _not_safe(root: str, pip: bool = False) -> str | None:
    if not pip and not is_download(root):
        return f"{root} isn't a Hugging Face download of moth (no model.safetensors next to code/); nothing to update"
    running = server_running(ROOT if pip else root)   # pip: serve's pidfile names the install, not the cache
    if running:
        return f"moth is running from this folder (port {running[0]}, pid {running[1]}): stop moth first, " \
               "then run the update again"
    return None


def _confirm(question: str) -> bool:
    try:
        return input(question + " [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def rollback(root: str, yes: bool, pip: bool = False) -> int:
    err = _not_safe(root, pip)
    if err:
        print(err)
        return 1
    dest = _backup_dir(root)
    if not dest:
        print("no backup to roll back to (only the last update keeps one)")
        return 1
    with open(os.path.join(dest, "manifest.json")) as f:
        old = json.load(f)["version"]
    if not yes and not _confirm(f"roll moth {version(root)} back to {old}?"):
        print("cancelled")
        return 1
    m = restore(root)
    print(f"rolled back to moth {m['version']}: restored {len(m['replaced']) + len(m.get('removed', []))} file(s), "
          f"removed {len(m['added'])}")
    return 0


def update(root: str, repo: str, rev: str | None, yes: bool, pip: bool = False) -> int:
    err = _not_safe(root, pip)
    if err:
        print(err)
        return 1
    hub = _hub()
    if hub is None:
        print(PIP_HINT)
        return 1
    info, err = _latest(hub, repo, rev)
    if err:
        print(err)
        return 1
    sibs = weight_siblings(info.siblings) if pip else info.siblings   # pip: only the model files, never code/
    if pip and not sibs:
        print(f"{repo} has none of {', '.join(sorted(WEIGHTS))}; is it a moth repo?")
        return 1
    manifest = sorted(s.rfilename for s in sibs)
    files, removed = changed_files(root, sibs), removed_files(root, sibs)
    if not files and not removed:
        _write_state(root, {"revision": info.sha, "files": manifest})
        print(f"up to date: {version_line(root)}")
        return 0
    print(f"update {root}: {describe(files, removed)}  (revision {(revision(root) or 'unknown')[:12]} -> "
          f"{info.sha[:12]})")
    if not yes and not _confirm("download it now?"):
        print("cancelled")
        return 1
    old = version(root)
    os.makedirs(root, exist_ok=True)
    backup(root, files, removed)
    try:   # pinned to the sha that was compared; huggingface_hub skips files that are already current
        hub.snapshot_download(repo_id=repo, local_dir=root, revision=info.sha,
                              **({"allow_patterns": sorted(WEIGHTS)} if pip else {}))
        for f in removed:   # only files the old version shipped and the new one dropped
            _remove(root, f)
        _drop_hf_metadata(root, removed)
    except Exception as e:
        restore(root)
        print(f"download failed ({type(e).__name__}: {str(e).splitlines()[0][:120] if str(e) else ''}); "
              "nothing was changed, try again later")
        return 1
    _write_state(root, {"revision": info.sha, "files": manifest})
    if pip:
        print(f"updated the weights in {root} to revision {info.sha[:12]}: {describe(files, removed)}")
        print(f"previous weights kept in {os.path.join(root, BACKUP)}  (moth update --rollback)")
        print("code comes from pip:  pip install -U moth-companion")
        return 0
    print(f"updated: moth {old} -> {version(root)} (revision {info.sha[:12]})")
    print(f"changed: {describe(files, removed)}")
    print(f"previous version kept in {os.path.join(root, BACKUP)}  (python -m moth update --rollback)")
    if any(f.startswith(INTEGRATIONS) for f in files + removed):
        print("integration files changed: re-copy any adapter you copied elsewhere (Pi moth.ts, OpenCode moth.js, "
              "the Hermes plugin folder). Hooks that point into code/moth/integrations keep working.")
    return 0


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
    a = ap.parse_args(argv)
    pip = _pip(root)
    if pip:   # pip install: the weights cache is what gets updated
        root = weights.cache_dir(a.repo)
    if a.rollback:
        return rollback(root, a.yes, pip)
    if a.check:
        return check(root, a.repo, a.revision, pip)
    return update(root, a.repo, a.revision, a.yes, pip)


if __name__ == "__main__":
    sys.exit(main())
