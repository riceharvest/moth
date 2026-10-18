"""Where moth's weights come from, and fetching them on first run of a pip install.

Three ways moth can be installed (mode()):
  hf   a Hugging Face download: model.safetensors next to code/ (code/moth/weights.py)
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
DEV_MODEL = os.path.join(ROOT, "runs", "sft1", "model")
DEV_TOKENIZER = os.path.join(ROOT, "data", "tiny-agent-112m", "tokenizer.json")
REVISION_FILE = ".moth-version"   # same file and format as moth/update.py
SIZE_HINT = "about 860 MB"


def mode() -> str:
    if os.path.exists(os.path.join(os.path.dirname(ROOT), "model.safetensors")):
        return "hf"
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


def present(folder: str) -> bool:
    return all(os.path.isfile(os.path.join(folder, f)) for f in FILES)


def ensure(folder: str | None = None, repo_id: str | None = None, revision: str | None = None) -> str:
    """The weights folder, downloading the model files into it first if any is missing. Exits with a message on
    failure (no huggingface_hub, offline, repo not found)."""
    repo_id = repo_id or repo()
    folder = folder or cache_dir(repo_id)
    if present(folder):
        return folder
    revision = revision or pinned_revision()
    try:
        import huggingface_hub as hub
    except ImportError:
        raise SystemExit("moth needs huggingface_hub to download its weights:  pip install -U huggingface_hub")
    try:
        sha = hub.HfApi().model_info(repo_id, revision=revision).sha   # pin the download to one commit
    except Exception as e:
        raise SystemExit(f"moth: can't get {repo_id}{'@' + revision if revision else ''} from the Hugging Face Hub "
                         f"({type(e).__name__}: {_first_line(e)}). Check your connection, or set MOTH_REPO / "
                         "MOTH_REVISION, or pass --model and --tokenizer.")
    size = f" ({SIZE_HINT})" if repo_id == REPO_ID else ""
    print(f"moth: first run, downloading the model{size} from https://huggingface.co/{repo_id} "
          f"(revision {sha[:12]})\n      into {folder}  (only once; `moth update` refreshes it)", file=sys.stderr,
          flush=True)
    os.makedirs(folder, exist_ok=True)
    try:
        hub.snapshot_download(repo_id=repo_id, revision=sha, local_dir=folder, allow_patterns=list(FILES))
    except Exception as e:
        raise SystemExit(f"moth: download failed ({type(e).__name__}: {_first_line(e)}); run moth again to resume")
    if not present(folder):
        missing = [f for f in FILES if not os.path.isfile(os.path.join(folder, f))]
        raise SystemExit(f"moth: {repo_id}@{sha[:12]} has no {', '.join(missing)}; is MOTH_REPO a moth repo?")
    with open(os.path.join(folder, REVISION_FILE), "w") as f:
        json.dump({"revision": sha, "files": sorted(FILES)}, f, indent=1)
        f.write("\n")
    print("moth: model downloaded", file=sys.stderr, flush=True)
    return folder


def ensure_default(model_dir: str) -> None:
    """serve / cli after parsing: fetch the weights only in pip mode and only when --model is the default folder."""
    if mode() == "pip" and os.path.realpath(model_dir) == os.path.realpath(cache_dir()):
        ensure(model_dir)


def _first_line(e: Exception) -> str:
    return str(e).splitlines()[0][:160] if str(e) else ""
