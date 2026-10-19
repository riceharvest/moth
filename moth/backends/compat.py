"""Can this moth's code load that model? Stdlib only (update --check, the first-run download and serve ask it before
torch loads anything), so an older install says what to do instead of failing deep inside the model code.

A model folder is too new for this code when:
  - its moth.json names a backend this moth doesn't have, or says "needs_moth": "<version>" newer than this one;
  - (tiny-agent) its config.json has settings tiny_agent.model.ModelConfig doesn't know: a newer architecture.
Then the user gets upgrade_hint(): what brings newer code for the way moth was installed.
"""
from __future__ import annotations

import json
import os
import sys

from moth.backends.manifest import BACKENDS, FILE

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # moth/
RELEASES = "https://github.com/riceharvest/moth/releases/latest"
# tiny_agent.model.ModelConfig's fields (tests/test_backends.py checks they match; importing it here would load torch)
TINY_AGENT_KEYS = frozenset({
    "vocab_size", "d_model", "n_layers", "n_heads", "head_dim", "n_kv_heads", "kv_group", "swa_window", "ffn_mult",
    "rope_base", "max_seq_len", "logit_softcap", "engram_layers", "engram_orders", "engram_heads", "engram_head_dim",
    "engram_rows_per_head"})


def version() -> str:
    try:
        with open(os.path.join(HERE, "VERSION")) as f:
            return f.read().strip()
    except OSError:
        return "unknown"


def newer(a: str, b: str) -> bool:
    """Version b is newer than a (dotted numbers; anything else: just different)."""
    try:
        return tuple(map(int, b.split("."))) > tuple(map(int, a.split(".")))
    except ValueError:
        return bool(b) and b != a


def _json(path: str):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def problem(folder: str) -> str | None:
    """Why this moth can't load the model in folder (None: it can, or there's nothing there to judge yet)."""
    backend = "tiny-agent"
    man = _json(os.path.join(folder, FILE))
    if isinstance(man, dict):
        backend = man.get("backend", "tiny-agent")
        if backend not in BACKENDS:
            return f"it runs on a {backend!r} backend this moth doesn't have"
        need = man.get("needs_moth")
        if isinstance(need, str) and need.strip() and newer(version(), need.strip()):
            return f"it needs moth {need.strip()} or newer"
    if backend == "tiny-agent":
        cfg = _json(os.path.join(folder, "config.json"))
        if isinstance(cfg, dict) and not cfg.get("model_type"):   # (model_type: a transformers save, not ours)
            extra = sorted(set(cfg) - TINY_AGENT_KEYS)
            if extra:
                return (f"its config.json has settings this moth doesn't know ({', '.join(extra[:4])}"
                        f"{' ...' * (len(extra) > 4)}): a newer model architecture")
    return None


def install_kind() -> str:
    """app (the installers' bundle), hf (a Hugging Face download), pip, or dev (this project's checkout)."""
    if getattr(sys, "frozen", False):
        return "app"
    from moth import weights   # stdlib only
    return weights.mode()


def upgrade_hint(kind: str | None = None, short: bool = False) -> str:
    """What gets newer moth code for this install (short: without the pip alternatives)."""
    pip = "pip install -U moth-companion" + ("" if short else "  (uv tool upgrade / pipx upgrade moth-companion)")
    return {"app": f"install the newest moth from {RELEASES}",
            "hf": "run  python -m moth update  (it brings the new code with the model)",
            "dev": "git pull",
            }.get(kind or install_kind(), pip)


def message(folder: str, why: str) -> str:
    return (f"moth: the model in {folder} is newer than this moth ({version()}): {why}. To get a moth that can run "
            f"it: {upgrade_hint()}")


def require(folder: str) -> None:
    """Exit with message() when this code can't load the model in folder."""
    why = problem(folder)
    if why:
        raise SystemExit(message(folder, why))


def remote_problem(hub, repo: str, sha: str, names) -> str | None:
    """problem() for a Hub revision: its config.json and moth.json (whichever of names it has; a few KB) into a
    throwaway folder. Can't fetch them: None (the download or update itself then says what went wrong)."""
    import tempfile
    try:
        with tempfile.TemporaryDirectory() as tmp:
            for f in ("config.json", FILE):
                if f in names:
                    hub.hf_hub_download(repo, f, revision=sha, local_dir=tmp)
            return problem(tmp)
    except Exception:  # noqa: BLE001  offline, a dropped connection: not this check's call
        return None
