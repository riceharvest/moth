"""Swappable model backends: what runs behind moth's pet, hooks and chat page.

  tiny-agent    moth-112m and its kin: tiny_agent's own architecture, KV cache kept across turns, device graphs,
                int8 on the CPU, the phrase ban, home tools. The default; a folder without moth.json is this.
  transformers  any causal LM transformers can load (a folder or an HF repo id) with a chat template, CPU or GPU.
                Optional: pip install "moth-companion[transformers]".
  openai        any OpenAI-compatible /v1/chat/completions server (llama.cpp server, Ollama, vLLM, LM Studio...),
                on this machine by default.

Pick one with a model spec (serve --model, `moth config model ...`, `moth setup --model ...`):
  a folder                    its moth.json (moth/backends/manifest.py), else tiny-agent defaults; a plain
                              transformers folder (config.json with model_type) without moth.json: transformers
  an HF repo id (owner/name)  its moth.json if it has one (fetched alone, a few hundred bytes), else transformers
  openai:<base_url>#<model>   e.g. openai:http://127.0.0.1:11434/v1#qwen3:0.6b

Stdlib only at import: torch / transformers load inside the backend that needs them. The format and examples:
moth/docs/MODEL_BACKENDS.md.
"""
from __future__ import annotations

import json
import os
import re
import sys
from urllib.parse import urlsplit

from moth.backends.manifest import FILE, Manifest, ManifestError, load, parse, with_updates  # noqa: F401

HUB_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")
OPENAI = "openai:"
LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1", "0.0.0.0")


def is_local_url(url: str) -> bool:
    """The endpoint is on this machine (loopback). A LAN address is not: moth's text would leave the machine."""
    try:
        host = (urlsplit(url).hostname or "").lower()
    except ValueError:
        return False
    return host in LOCAL_HOSTS or host.startswith("127.") or host.endswith(".localhost")


def parse_spec(spec: str) -> tuple[str, object]:
    """("openai", (base_url, model)) | ("folder", abspath) | ("hub", repo_id)."""
    spec = (spec or "").strip()
    if spec.startswith(OPENAI):
        rest = spec[len(OPENAI):]
        url, sep, model = rest.rpartition("#")
        if not sep or not url or not model:
            raise ManifestError(f"model {spec!r}: write openai:<base_url>#<model>, "
                                "e.g. openai:http://127.0.0.1:11434/v1#qwen3:0.6b")
        if not url.startswith(("http://", "https://")):
            raise ManifestError(f"model {spec!r}: the base url must start with http:// or https://")
        return "openai", (url, model)
    path = os.path.expanduser(spec)
    if os.path.exists(path) or spec.startswith((".", "/", "~", "\\")) or "\\" in spec or not HUB_ID.fullmatch(spec):
        return "folder", os.path.abspath(path)
    return "hub", spec


def configured() -> str | None:
    """The `model` setting of ~/.moth/config.json, or None (the install's own moth)."""
    try:
        from moth import config
        return config.model()
    except Exception:  # noqa: BLE001  an unreadable config is the default
        return None


def _transformers_folder(folder: str) -> bool:
    """A plain transformers save (config.json with model_type), not a tiny-agent checkpoint (whose config has none)."""
    try:
        with open(os.path.join(folder, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        return False
    return isinstance(cfg, dict) and bool(cfg.get("model_type"))


def resolve(spec: str | None, network: bool = True) -> Manifest:
    """The Manifest for a model spec (None or "": the install's default folder is the caller's business). network:
    an HF repo id moth hasn't seen yet may fetch its moth.json (only that file); False (doctor, the pet): never."""
    kind, val = parse_spec(spec or "")
    if kind == "openai":
        url, model = val
        m = parse({"backend": "openai", "base_url": url, "model": model}, None, f"model {spec}")
        return with_updates(m, source="spec")
    if kind == "folder":
        m = load(val)
        if m.source == "default" and _transformers_folder(val):
            return with_updates(parse({"backend": "transformers", "model": val}, val), source="inferred")
        return m
    return _hub(val, network)


def _hub(repo: str, network: bool) -> Manifest:
    from moth import weights   # stdlib only
    folder = weights.cache_dir(repo)
    if repo == weights.REPO_ID:   # moth-112m itself: no need to ask the Hub
        local = os.path.join(folder, FILE)
        return load(folder) if os.path.isfile(local) else Manifest(folder=folder, repo=repo)
    local = os.path.join(folder, FILE)
    if not os.path.isfile(local) and network and not _offline():
        print(f"moth: {repo} isn't a folder here, so it's a Hugging Face repo id: checking it for a moth.json",
              file=sys.stderr, flush=True)
        _fetch_manifest(repo, folder)
    if os.path.isfile(local):
        m = load(folder)
        if m.backend == "openai":   # a downloaded file must never send moth's text somewhere: endpoints are yours
            raise ManifestError(f"{local}: a moth.json from the Hub ({repo}) can't name an openai endpoint; "
                                "to use a server, write openai:<url>#<model> or a moth.json of your own")
        upd = {"repo": m.repo or repo}
        if m.backend == "transformers" and not m.model:
            upd["model"] = repo
        return with_updates(m, **upd)
    # no moth.json: a general chat model, loaded by transformers from the HF cache
    return with_updates(parse({"backend": "transformers", "model": repo, "repo": repo}, None, f"model {repo}"),
                        source="spec")


def _offline() -> bool:
    try:
        from moth import config
        return config.offline()
    except Exception:  # noqa: BLE001
        return False


def _fetch_manifest(repo: str, folder: str) -> None:
    """repo's moth.json into folder (nothing else). No such file, no network or no huggingface_hub: nothing."""
    try:
        import huggingface_hub as hub
        os.makedirs(folder, exist_ok=True)
        hub.hf_hub_download(repo_id=repo, filename=FILE, local_dir=folder)
    except Exception as e:  # noqa: BLE001  most often: the repo simply has no moth.json
        name = type(e).__name__
        if "NotFound" not in name and "Gated" not in name:
            print(f"moth: couldn't check {repo} for a moth.json ({name}); treating it as a transformers model",
                  file=sys.stderr, flush=True)


def prepare(m: Manifest) -> None:
    """Before loading: a tiny-agent model from the Hub gets its weights downloaded (moth/weights.py, with the same
    checks and messages as moth-112m's first run). Others: nothing (transformers uses its own HF cache)."""
    if m.backend == "tiny-agent" and m.repo and m.folder:
        from moth import weights
        if os.path.realpath(m.folder) == os.path.realpath(weights.cache_dir(m.repo)):
            weights.ensure(m.folder, m.repo)


def privacy_note(m: Manifest) -> str | None:
    """What doctor and serve say when moth's text leaves this machine (an openai endpoint that isn't loopback)."""
    if m.backend == "openai" and not is_local_url(m.base_url or ""):
        return (f"the model endpoint {m.base_url} is not on this machine: every agent message moth reacts to and "
                "everything you type to moth is sent there")
    return None


def load_backend(m: Manifest, a):
    """The backend object for m (imports torch / transformers only now)."""
    if m.backend == "tiny-agent":
        from moth.backends.tiny_agent import TinyAgentBackend
        return TinyAgentBackend(a)
    if m.backend == "transformers":
        from moth.backends.hf_transformers import TransformersBackend
        return TransformersBackend(m, a)
    from moth.backends.openai_compat import OpenAIBackend
    return OpenAIBackend(m, a)
