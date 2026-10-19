"""Where moth's weights come from, and fetching them on first run of a pip install.

Three ways moth can be installed (mode()):
  hf   a Hugging Face download: model.safetensors next to code/ (code/moth/weights.py), or one whose weights
       haven't finished downloading (hf_layout; serve then says how to finish it)
  dev  this project's checkout: runs/ and data/ next to the package
  pip  `pip install moth-companion` (or uv tool / pipx): only the code is installed. The model files
       (model.safetensors, config.json, tokenizer.json) are downloaded from the Hub on first run into
       ~/.cache/moth/<repo>/  ($MOTH_CACHE, else $XDG_CACHE_HOME/moth, overrides the base folder).

$MOTH_REPO picks the Hub repo (default below), $MOTH_REVISION pins the first download to a branch, tag or commit.
After that the folder is used as it is (no network); `moth update` refreshes it in place, with a backup.
Stdlib only at import time: huggingface_hub is imported when a download is actually needed.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # code/ (Hub download), the project (dev checkout) or site-packages (pip)
REPO_ID = "darioooooo0o/moth-112m"
FILES = ("model.safetensors", "config.json", "tokenizer.json")
OPTIONAL = ("moth.json",)   # a model's manifest (moth/backends/manifest.py), when the repo has one
DEV_MODEL = os.path.join(ROOT, "runs", "sft1", "model")
DEV_TOKENIZER = os.path.join(ROOT, "data", "tiny-agent-112m", "tokenizer.json")
REVISION_FILE = ".moth-version"   # same file and format as moth/update.py
SIZE_HINT = "about 860 MB"


# what a Hugging Face download of moth has next to code/ besides the weights: an interrupted `hf download` leaves
# these behind (huggingface_hub moves model.safetensors into place only once it is whole; the small files land first)
HF_MARKS = ("model.safetensors", REVISION_FILE, ".cache/huggingface/download/code/moth/VERSION.metadata",
            ("config.json", "tokenizer.json"))


def hf_layout(repo: str) -> bool:
    """repo is a Hugging Face download of moth: code/moth/ plus model.safetensors next to code/, or the marks of a
    download whose weights never finished (HF_MARKS; a tuple: all of those). moth/update.py's is_download is this
    test. A dev checkout or the home folder has no code/moth/."""
    if not os.path.isdir(os.path.join(repo, "code", "moth")):
        return False
    return any(all(os.path.isfile(os.path.join(repo, *f.split("/"))) for f in (m if isinstance(m, tuple) else (m,)))
               for m in HF_MARKS)


def complete(folder: str) -> bool:
    """config.json plus a model.safetensors whose size matches its header (a synced folder may be mid-copy, an
    interrupted download cut off)."""
    import struct
    st = os.path.join(folder, "model.safetensors")
    if not (os.path.isfile(os.path.join(folder, "config.json")) and os.path.isfile(st)):
        return False
    try:
        with open(st, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            if not 2 <= n <= 100 << 20:   # a header, not a garbage length
                return False
            head = json.loads(f.read(n))
        end = max(v["data_offsets"][1] for k, v in head.items() if k != "__metadata__")
        return os.path.getsize(st) == 8 + n + end
    except Exception:  # noqa: BLE001  short, garbage, no tensors: not complete
        return False


def incomplete_message(folder: str) -> str:
    """What to tell a user whose Hugging Face download of moth (folder) has no usable weights yet."""
    gone = missing(folder)
    what = (f"{', '.join(gone)} missing" if gone else "model.safetensors is cut off")
    return (f"{folder} is a Hugging Face download of moth whose weights didn't finish downloading ({what}). Finish "
            f"it (hf download {repo()} --local-dir \"{folder}\" resumes it), or run `python -m moth update` from its "
            "code/ folder, which completes it in place. Nothing was downloaded anywhere else.")


def mode() -> str:
    repo_dir = os.path.dirname(ROOT)
    if os.path.exists(os.path.join(repo_dir, "model.safetensors")) or hf_layout(repo_dir):
        return "hf"   # (also a download still missing its weights: never "pip", which would fetch them elsewhere)
    if os.path.isfile(DEV_TOKENIZER):   # a positive marker: an editable install of the checkout stays dev
        return "dev"
    return "pip"


def repo() -> str:
    return os.environ.get("MOTH_REPO") or REPO_ID


def pinned_revision() -> str | None:
    return os.environ.get("MOTH_REVISION") or None


def cache_dir(repo_id: str | None = None) -> str:
    base = os.environ.get("MOTH_CACHE") or os.path.join(
        os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache"), "moth")
    return os.path.join(base, *(repo_id or repo()).split("/"))


def missing(folder: str) -> list[str]:
    return [f for f in FILES if not os.path.isfile(os.path.join(folder, f))]


def present(folder: str) -> bool:
    return not missing(folder)


def ensure(folder: str | None = None, repo_id: str | None = None, revision: str | None = None) -> str:
    """The weights folder, downloading the model files into it first if any is missing. Exits with a message on
    failure (no huggingface_hub, offline, repo not found)."""
    repo_id = repo_id or repo()
    folder = folder or cache_dir(repo_id)
    if present(folder):
        return folder
    revision = revision or pinned_revision()
    from moth import config   # stdlib only
    if config.offline():   # MOTH_OFFLINE / HF_HUB_OFFLINE: nothing goes to the network, not even the first download
        raise SystemExit(f"moth: the model isn't downloaded yet ({', '.join(missing(folder))} missing in {folder}) and "
                         "offline mode is on (HF_HUB_OFFLINE / MOTH_OFFLINE). Unset it for the first run, or pass "
                         "--model and --tokenizer.")
    try:
        import huggingface_hub as hub
    except ImportError:
        raise SystemExit("moth needs huggingface_hub to download its weights:  pip install -U huggingface_hub")
    try:
        info = hub.HfApi().model_info(repo_id, revision=revision)
    except Exception as e:
        raise SystemExit(f"moth: can't get {repo_id}{'@' + revision if revision else ''} from the Hugging Face Hub "
                         f"({type(e).__name__}: {_first_line(e)}). Check your connection, or set MOTH_REPO / "
                         "MOTH_REVISION, or pass --model and --tokenizer.")
    info = compatible(hub, repo_id, info, pinned=bool(revision))
    sha = info.sha   # pin the download to one commit
    size = f" ({SIZE_HINT})" if repo_id == REPO_ID else ""
    endpoint = (os.environ.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")
    print(f"moth: first run, downloading the model{size} from {endpoint}/{repo_id} "
          f"(revision {sha[:12]})\n      into {folder}  (only once; `moth update` refreshes it)", file=sys.stderr,
          flush=True)
    os.makedirs(folder, exist_ok=True)
    drop_orphans(folder)   # what killed first runs left
    try:
        hub.snapshot_download(repo_id=repo_id, revision=sha, local_dir=folder, allow_patterns=list(FILES + OPTIONAL))
    except Exception as e:
        raise SystemExit(f"moth: download failed ({type(e).__name__}: {_first_line(e)}); run moth again to resume")
    gone = missing(folder)
    if gone:   # huggingface_hub can return quietly when the Hub stops answering halfway
        on_hub = {s.rfilename for s in (info.siblings or [])}
        if any(f not in on_hub for f in gone) and on_hub:
            raise SystemExit(f"moth: {repo_id}@{sha[:12]} has no {', '.join(f for f in gone if f not in on_hub)}; "
                             "is MOTH_REPO a moth repo?")
        raise SystemExit(f"moth: download incomplete ({', '.join(gone)} missing; the Hub stopped answering?); "
                         "run moth again to resume")
    with open(os.path.join(folder, REVISION_FILE), "w") as f:
        json.dump({"revision": sha, "files": sorted(FILES + tuple(x for x in OPTIONAL if os.path.isfile(
            os.path.join(folder, x))))}, f, indent=1)
        f.write("\n")
    print("moth: model downloaded", file=sys.stderr, flush=True)
    return folder


def compatible(hub, repo_id: str, info, pinned: bool = False):
    """info, unless the model at that revision is too new for this code (moth/backends/compat.py): then, when no
    revision was asked for, the one tagged v<this moth's version> if the repo has it; else exit saying how to get
    newer code. Never downloads more than the model's config.json / moth.json to decide."""
    from moth.backends import compat
    names = {s.rfilename for s in (info.siblings or [])}
    why = compat.remote_problem(hub, repo_id, info.sha, names)
    if not why:
        return info
    tag = "v" + compat.version()
    if not pinned:
        try:
            old = hub.HfApi().model_info(repo_id, revision=tag)
        except Exception:  # noqa: BLE001  no such tag (or no network): nothing to fall back to
            old = None
        if old is not None and not compat.remote_problem(hub, repo_id, old.sha, {s.rfilename for s in
                                                                              (old.siblings or [])}):
            print(f"moth: the newest model on the Hub is too new for this moth ({why}); using the one made for moth "
                  f"{compat.version()} ({tag}). For the new one: {compat.upgrade_hint()}", file=sys.stderr, flush=True)
            return old
    raise SystemExit(compat.message(f"{repo_id}@{info.sha[:12]}", why))


def drop_orphans(folder: str, min_age: float = 300.0) -> int:
    """Delete huggingface_hub's leftover partial downloads under folder/.cache/huggingface/download/: since hub 1.x
    names them <file>.<etag>.<random>.incomplete, a killed download is never resumed and its part (up to the whole
    model) would stay forever. Only ones untouched for min_age seconds (a live download writes all the time); the
    older <file>.<etag>.incomplete form is left alone (huggingface_hub resumes it). Returns how many went."""
    import re
    import time
    base, n = os.path.join(folder, ".cache", "huggingface", "download"), 0
    for d, _, files in os.walk(base):
        for f in files:
            if re.fullmatch(r"[\w=-]+\.[0-9a-f]+\.[0-9a-f]{8}\.incomplete", f):
                p = os.path.join(d, f)
                try:
                    if time.time() - os.path.getmtime(p) >= min_age:
                        os.remove(p)
                        n += 1
                except OSError:
                    pass
    return n


def ensure_default(model_dir: str) -> None:
    """serve / cli after parsing: fetch the weights only in pip mode and only when --model is the default folder.
    A Hugging Face download whose weights didn't finish (--model left at the download folder) exits with how to
    finish it, instead of failing to load or fetching a second copy elsewhere."""
    m = mode()
    if m == "hf":
        repo_dir = os.path.dirname(ROOT)
        if os.path.realpath(model_dir) == os.path.realpath(repo_dir) and not complete(repo_dir):
            raise SystemExit("moth: " + incomplete_message(repo_dir))
    elif m == "pip" and os.path.realpath(model_dir) == os.path.realpath(cache_dir()):
        ensure(model_dir)


def _first_line(e: Exception) -> str:
    return str(e).splitlines()[0][:160] if str(e) else ""
