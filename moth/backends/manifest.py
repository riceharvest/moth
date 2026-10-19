"""moth.json: what model runs behind moth, and how moth talks to it. Stdlib only (the pet, doctor, launch and
`moth config` read it without torch). The format is documented in moth/docs/MODEL_BACKENDS.md.

A weights folder without moth.json: an XS folder (config.json "arch": "tiny-moe") runs on the tiny-moe backend, any
other one is the old moth-112m on the tiny-agent backend, with every default below, so existing folders keep working. Unknown keys are ignored (with a note), so an older moth can read a newer
manifest; a key of the wrong type is an error that names the key.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields, replace

FILE = "moth.json"
BACKENDS = ("tiny-moe", "tiny-agent", "transformers", "openai")
FORMATS = ("chatml", "template", "xs")   # template: the tokenizer's own (transformers) / the server's (openai); xs: tiny-moe
AGENT_CARD = "[the agent said]\n{message}"   # = moth.agent.agent_turn(); the format moth-112m was trained on
SAMPLING = {"temperature": 0.8, "top_p": 0.95, "rep_penalty": 1.3, "max_tokens": 200}   # the old model's (tiny-agent)
# XS serves with the sampler its RL trains with: plain softmax at the RL's serving temperature, no top-p, no repetition
# penalty (moth.norepeat's phrase ban stays as a hard filter on reactions; serve --phrase-memory 0 turns it off)
XS_SAMPLING = {"temperature": 0.7, "top_p": 1.0, "rep_penalty": 1.0, "max_tokens": 200}
API_KEY_ENV = "MOTH_OPENAI_API_KEY"

# For a general chat model that was never trained as moth (an HF repo or an endpoint without its own moth.json):
# a short character sheet and a per-reaction nudge. {character} is the manifest's character name.
GENERIC_IDENTITY = "you are {character}, a small, sweet, slightly silly moth who lives inside your friend's computer. "
GENERIC_STYLE = (
    "talk with your friend in ONE short lowercase line, usually under 15 words: simple words, no lists, no markdown, "
    "no emojis, never like an assistant. you take things literally and get facts sweetly wrong, but you are warm and "
    "kind. messages from the big coding agent your friend works with show up marked [the agent said].")
GENERIC_SYSTEM = GENERIC_IDENTITY + GENERIC_STYLE
GENERIC_REACTION = (
    "(react to that in ONE short silly lowercase line, under 15 words: grab one everyday word from it and confidently "
    "misunderstand it in moth logic (lamps, dust, crumbs, rain, bugs, snacks). don't explain or summarize it, don't "
    "give advice, no code words.)")


def generic_system(card: dict | None, character: str = "moth") -> str:
    """A general model's system prompt: the generic sheet for the moth card, else the card's identity lines + the
    generic style rules (who it is still comes only from the card)."""
    from moth import cards
    if card is None or card.get("id") == "moth":
        return GENERIC_SYSTEM.replace("{character}", character)
    lines = cards._ident(card)
    return (" ".join(lines) + " " if lines else "") + GENERIC_STYLE


class ManifestError(ValueError):
    """A moth.json moth can't use (the message names the file and the key)."""


@dataclass(frozen=True)
class Manifest:
    backend: str = "tiny-agent"         # a folder's default; XS folders: "tiny-moe" (backends.resolve infers it)
    character: str = "moth"
    chat_format: str = "chatml"
    system: str | None = None          # None: moth.persona.SYSTEM_STUDENT (tiny-agent) / GENERIC_SYSTEM (others)
    agent_card: str = AGENT_CARD
    reaction: str | None = None        # appended to the agent card for that one turn only (never kept in history)
    sampling: dict = field(default_factory=lambda: dict(SAMPLING))
    tools: bool = False                # ignored: moth has no tools any more (pure chat); kept so old files parse
    lowercase: bool = True             # moth only speaks lowercase
    context: int | None = None         # tokens of chat kept in mind (default: moth.serve.CTX_BUDGET)
    stop: tuple = ()                   # extra stop strings
    model: str | None = None           # transformers: folder or HF repo id; openai: the server's model name
    base_url: str | None = None        # openai: e.g. http://127.0.0.1:8080/v1
    api_key_env: str | None = None     # openai: the env var holding the key (default MOTH_OPENAI_API_KEY)
    tokenizer: str | None = None       # tiny-agent / tiny-moe: tokenizer.json (relative to the folder)
    repo: str | None = None            # HF repo id `moth update` refreshes this model from
    version: str | None = None         # the model's own version (shown by doctor)
    needs_moth: str | None = None      # the oldest moth code that can run it (moth/backends/compat.py checks it)
    # where it came from (not keys of the file)
    folder: str | None = None          # the folder moth.json (or the weights) is in
    source: str = "default"            # "file", "default" (no moth.json), "spec" (openai:..., a bare HF repo id)
    notes: tuple = ()                  # ignored keys etc., for doctor

    # --- derived -------------------------------------------------------------------------------------------
    @property
    def is_default(self) -> bool:
        """A folder of moth's own models without a moth.json (XS or the old moth-112m), run with the defaults."""
        return self.backend in ("tiny-agent", "tiny-moe") and self.source == "default"

    def system_prompt(self, default: str | None = None) -> str:
        """The system prompt to serve with: the manifest's, else (tiny-agent) default, else GENERIC_SYSTEM."""
        if self.system:
            return self.system
        if self.backend in ("tiny-agent", "tiny-moe") and default is not None:
            return default
        return GENERIC_SYSTEM.replace("{character}", self.character)

    def sample(self, key: str):
        return self.sampling.get(key, SAMPLING[key])

    def card(self, message: str) -> str:
        """How the coding agent's (already cleaned) final message is shown to the model."""
        return self.agent_card.replace("{message}", message)

    def card_parts(self) -> tuple[str, str]:
        """(text before {message}, text after it) of the agent card."""
        pre, _, post = self.agent_card.partition("{message}")
        return pre, post

    def describe(self) -> str:
        """One line for doctor / the serve log: backend, model, character."""
        what = {"tiny-agent": self.folder or "?", "tiny-moe": self.folder or "?", "transformers": self.model or self.folder or "?",
                "openai": f"{self.model or '?'} at {self.base_url or '?'}"}[self.backend]
        ver = f" v{self.version}" if self.version else ""
        return f"{self.backend}: {what}{ver}, character {self.character!r}"


def _norm_sampling(v, where: str, base: dict = SAMPLING) -> dict:
    if not isinstance(v, dict):
        raise ManifestError(f"{where}: sampling must be an object like {json.dumps(base)}")
    out = dict(base)
    for k, x in v.items():
        if k not in SAMPLING:
            continue
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise ManifestError(f"{where}: sampling.{k} must be a number")
        out[k] = int(x) if k == "max_tokens" else float(x)
    if out["max_tokens"] < 1 or not 0 < out["top_p"] <= 1 or out["temperature"] < 0 or out["rep_penalty"] <= 0:
        raise ManifestError(f"{where}: sampling out of range (temperature >= 0, 0 < top_p <= 1, rep_penalty > 0, "
                            "max_tokens >= 1)")
    return out


TYPES = {"backend": str, "character": str, "chat_format": str, "system": str, "agent_card": str, "reaction": str,
         "tools": bool, "lowercase": bool, "context": int, "model": str, "base_url": str, "api_key_env": str,
         "tokenizer": str, "repo": str, "version": str, "needs_moth": str}


def parse(data: dict, folder: str | None = None, where: str = FILE) -> Manifest:
    """A Manifest from moth.json's object. Defaults depend on the backend: general models (transformers, openai)
    default to the tokenizer's / server's chat template, no tools and the generic reaction nudge."""
    if not isinstance(data, dict):
        raise ManifestError(f"{where}: expected a JSON object")
    backend = data.get("backend", "tiny-agent")
    if backend not in BACKENDS:
        raise ManifestError(f"{where}: backend must be one of {', '.join(BACKENDS)}, not {backend!r}")
    general = backend not in ("tiny-agent", "tiny-moe")
    kw: dict = {"backend": backend, "folder": folder, "source": "file"}
    if general:
        kw.update(chat_format="template", lowercase=True, reaction=GENERIC_REACTION)
    if backend == "tiny-moe":
        kw.update(chat_format="xs", sampling=dict(XS_SAMPLING))
    notes = []
    for k, v in data.items():
        if k in ("backend", "$schema", "comment", "moth_manifest"):
            continue
        if k == "sampling":
            kw["sampling"] = _norm_sampling(v, where, XS_SAMPLING if backend == "tiny-moe" else SAMPLING)
        elif k == "tools":
            if not isinstance(v, bool):
                raise ManifestError(f"{where}: tools must be true/false")
            if v:
                notes.append("tools: moth has no tools any more (pure chat); ignored")
        elif k == "stop":
            if not (isinstance(v, list) and all(isinstance(s, str) and s for s in v)):
                raise ManifestError(f"{where}: stop must be a list of strings")
            kw["stop"] = tuple(v)
        elif k in TYPES:
            if v is None and k not in ("backend", "character", "agent_card", "chat_format", "tools", "lowercase"):
                kw[k] = None
                continue
            t = TYPES[k]
            if not isinstance(v, t) or (t is int and isinstance(v, bool)):
                raise ManifestError(f"{where}: {k} must be {'a string' if t is str else 'true/false' if t is bool else 'a number'}")
            kw[k] = v
        else:
            notes.append(f"unknown key {k!r} ignored")
    if kw.get("chat_format", "chatml") not in FORMATS:
        raise ManifestError(f"{where}: chat_format must be one of {', '.join(FORMATS)}")
    if "{message}" not in kw.get("agent_card", AGENT_CARD):
        raise ManifestError(f"{where}: agent_card must contain {{message}}")
    if not str(kw.get("character", "moth")).strip() or len(kw.get("character", "moth")) > 40:
        raise ManifestError(f"{where}: character must be a short name")
    if kw.get("context") is not None and kw["context"] < 256:
        raise ManifestError(f"{where}: context must be at least 256 tokens")
    if kw.get("api_key_env") and not kw["api_key_env"].startswith("MOTH_"):   # never hand e.g. GITHUB_TOKEN to a server
        raise ManifestError(f"{where}: api_key_env must name a MOTH_... variable (e.g. MOTH_OPENAI_API_KEY)")
    if backend == "openai" and not (kw.get("base_url") and kw.get("model")):
        raise ManifestError(f"{where}: the openai backend needs base_url and model")
    if backend == "tiny-agent" and kw.get("chat_format", "chatml") != "chatml":
        raise ManifestError(f"{where}: the tiny-agent backend only speaks chatml")
    if backend == "tiny-moe" and kw.get("chat_format") != "xs":
        raise ManifestError(f"{where}: the tiny-moe backend only speaks its own template (chat_format xs)")
    kw["notes"] = tuple(notes)
    return Manifest(**kw)


def default_for(folder: str) -> Manifest:
    """A folder without moth.json: an XS folder (config.json "arch": "tiny-moe") -> tiny-moe with XS's defaults; any
    other -> tiny-agent (the old moth-112m)."""
    try:
        with open(os.path.join(folder, "config.json"), encoding="utf-8") as f:
            arch = json.load(f).get("arch")
    except (OSError, ValueError, AttributeError):
        arch = None
    if arch == "tiny-moe":
        return Manifest(backend="tiny-moe", chat_format="xs", sampling=dict(XS_SAMPLING), folder=folder)
    return Manifest(folder=folder)


def load(folder: str) -> Manifest:
    """folder/moth.json, or the defaults of the model in the folder when there is none (default_for)."""
    path = os.path.join(folder, FILE)
    if not os.path.isfile(path):
        return default_for(folder)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        raise ManifestError(f"{path}: can't read it ({e})") from e
    return parse(data, folder, path)


def with_updates(m: Manifest, **kw) -> Manifest:
    return replace(m, **kw)


def apply_args(a, m: Manifest):
    """The serve options a manifest implies, on a copy of the caller's namespace (unchanged when there are none): its
    context budget unless --ctx-budget was given."""
    upd = {}
    if m.context and not getattr(a, "ctx_budget", None):
        upd["ctx_budget"] = m.context
    if not upd:
        return a
    import argparse
    return argparse.Namespace(**{**vars(a), **upd})


def as_dict(m: Manifest) -> dict:
    """The file's keys (for docs/tests: what a moth.json spelling out m would hold)."""
    skip = {"folder", "source", "notes"}
    d = {f.name: getattr(m, f.name) for f in fields(m) if f.name not in skip}
    d["stop"] = list(d["stop"])
    return d
