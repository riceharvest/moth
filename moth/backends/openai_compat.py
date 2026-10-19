"""The openai backend: moth's character on any OpenAI-compatible chat server (llama.cpp's llama-server, Ollama, vLLM,
LM Studio, ...): POST <base_url>/chat/completions, stdlib only (urllib), no streaming.

Privacy: moth's promise is that nothing leaves your machine. With this backend every agent message moth reacts to
and everything you type to moth goes to that server. On localhost / 127.x that stays on this machine; for any other
host moth.serve warns at startup and `moth doctor` says so on every run.

The API key (if the server wants one) comes only from an environment variable: MOTH_OPENAI_API_KEY, or the one the
manifest names in api_key_env. It is never written to moth's config or logs. No token-level sampling here, so no
phrase ban; the repetition penalty is passed on as llama.cpp's repeat_penalty / vLLM's repetition_penalty (servers
that refuse unknown fields get the request again without them).
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from moth.backends import is_local_url
from moth.backends.base import Backend, PseudoTokenizer, approx_tokens, chat_messages, cut_at_stops
from moth.backends.manifest import API_KEY_ENV

TIMEOUT = 120.0   # s for one reply (a cold Ollama model loads first)


def chat_url(base_url: str) -> str:
    """http://host:port[/v1] -> .../v1/chat/completions; a full .../chat/completions URL stays as it is."""
    u = base_url.rstrip("/")
    if u.endswith("/chat/completions"):
        return u
    return (u if u.endswith("/v1") else u + "/v1") + "/chat/completions"


class OpenAIBackend(Backend):
    name = "openai"

    def __init__(self, m, a=None):
        self.m = m
        self.url = chat_url(m.base_url)
        self.model = m.model
        self.local = is_local_url(m.base_url)
        self.key = os.environ.get(m.api_key_env or API_KEY_ENV) or None
        # loopback never goes through a proxy (as moth's own clients); another host uses the environment's proxies
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if self.local \
            else urllib.request.build_opener()
        self.extras = True
        self.tok = PseudoTokenizer()

    def tokenizer(self):
        return self.tok

    def count(self, messages):
        return approx_tokens(messages)

    def _post(self, body: dict) -> dict:
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json",
                                              **({"Authorization": f"Bearer {self.key}"} if self.key else {})})
        with self.opener.open(req, timeout=TIMEOUT) as r:
            return json.loads(r.read() or b"{}")

    def generate(self, messages, *, max_tokens, temperature, top_p, rep_penalty=1.0, pen_ids=(), ban_tools=False,
                 logits_hook=None, stop=()):
        body = {"model": self.model, "messages": chat_messages(messages), "max_tokens": int(max_tokens),
                "temperature": float(temperature), "top_p": float(top_p), "stream": False}
        if stop:
            body["stop"] = list(stop)[:4]
        extra = {"repeat_penalty": rep_penalty, "repetition_penalty": rep_penalty} if rep_penalty != 1.0 else {}
        try:
            d = self._post({**body, **(extra if self.extras else {})})
        except urllib.error.HTTPError as e:
            if not (extra and self.extras and e.code in (400, 422)):
                raise RuntimeError(f"{self.url} answered HTTP {e.code}") from e
            self.extras = False   # a strict server: no penalty fields from now on
            d = self._post(body)
        except urllib.error.URLError as e:
            raise RuntimeError(f"can't reach {self.url} ({e.reason})") from e
        try:
            msg = d["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"{self.url} sent no choices") from e
        return cut_at_stops(msg.get("content") or "", stop).strip()
