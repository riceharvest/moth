"""OpenAI-compatible chat client (stdlib urllib, no `openai` package) and the cost meter.

CostMeter counts calls, prompt/completion tokens and wall seconds per role ("sim", "judge", "agent_msg", "policy").
Tokens come from the server's `usage`; mock components (and servers without usage) estimate ~4 chars per token and
the meter marks those roles "est". box_hours() turns a per-episode meter into box-hours and dollars, given measured
throughput (tok/s: a pilot measures it; nothing here guesses it) and a box price ($/h; runbook/cr.sh CR_PRICE when the
worktree has it).
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CR_SH = os.path.join(ROOT, "runbook", "cr.sh")
DEFAULT_PRICE = 1.30   # = cr.sh's fallback for an unknown box type


def write_json(path: str, obj, **kw):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, **kw)


def read_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_text(path: str, text: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def est_tokens(text: str) -> int:
    return max(1, round(len(text or "") / 4))


class CostMeter:
    """Per-role counters. Thread-safe; merge() adds another meter (per-episode meters -> a run total)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.roles: dict[str, dict] = {}
        self.episodes = 0

    def add(self, role: str, prompt_tokens: int, completion_tokens: int, seconds: float = 0.0, est: bool = False,
            cached: bool = False):
        with self.lock:
            r = self.roles.setdefault(role, {"calls": 0, "cached": 0, "prompt_tokens": 0, "completion_tokens": 0,
                                             "seconds": 0.0, "est": False})
            if cached:   # a judge-cache hit costs nothing
                r["cached"] += 1
                return
            r["calls"] += 1
            r["prompt_tokens"] += int(prompt_tokens)
            r["completion_tokens"] += int(completion_tokens)
            r["seconds"] += float(seconds)
            r["est"] = r["est"] or est

    def merge(self, other: "CostMeter | dict"):
        roles = other.roles if isinstance(other, CostMeter) else other.get("roles", {})
        with self.lock:
            self.episodes += other.episodes if isinstance(other, CostMeter) else other.get("episodes", 1)
            for k, v in roles.items():
                r = self.roles.setdefault(k, {"calls": 0, "cached": 0, "prompt_tokens": 0, "completion_tokens": 0,
                                              "seconds": 0.0, "est": False})
                for f in ("calls", "cached", "prompt_tokens", "completion_tokens", "seconds"):
                    r[f] += v.get(f, 0)
                r["est"] = r["est"] or v.get("est", False)

    def summary(self) -> dict:
        with self.lock:
            return {"episodes": self.episodes, "roles": {k: dict(v) for k, v in self.roles.items()}}

    def per_episode(self) -> dict:
        """Mean calls/tokens per episode for each role."""
        s = self.summary()
        n = max(1, s["episodes"])
        return {k: {f: round(v[f] / n, 2) for f in ("calls", "cached", "prompt_tokens", "completion_tokens", "seconds")}
                | {"est": v["est"]} for k, v in s["roles"].items()}


def cr_prices(path: str = CR_SH) -> dict[str, float]:
    """{box type: $/h} from runbook/cr.sh's CR_PRICE (gitignored; {} when the file isn't there)."""
    if not os.path.exists(path):
        return {}
    text = open(path, encoding="utf-8", errors="replace").read()
    m = re.search(r"declare -A CR_PRICE=\((.*?)\)\s*\n", text, re.S)
    return {k: float(v) for k, v in re.findall(r"\[([\w.-]+)\]=([\d.]+)", m.group(1))} if m else {}


def box_hours(per_episode: dict, episodes: int, tps: dict[str, dict], price: float | None = None,
              box: str | None = None) -> dict:
    """Box-hours (and $) for `episodes` episodes.

    per_episode: CostMeter.per_episode(). tps: {role: {"prefill": tok/s, "decode": tok/s}} measured on the box for
    the whole server at the pilot's concurrency (aggregate throughput, not one stream). Roles without tps are skipped
    and listed. Prefix-cache hits make the real prefill cheaper than this (the label prompt and the sim's system
    prompt are shared prefixes), so the estimate errs high."""
    if price is None:
        price = cr_prices().get(box or "", DEFAULT_PRICE)
    out, total, skipped = {}, 0.0, []
    for role, v in per_episode.items():
        t = tps.get(role)
        if not t:
            skipped.append(role)
            continue
        secs = v["prompt_tokens"] / t["prefill"] + v["completion_tokens"] / t["decode"]
        h = secs * episodes / 3600
        out[role] = round(h, 3)
        total += h
    return {"episodes": episodes, "box_hours": out, "total_box_hours": round(total, 3), "price_per_hour": price,
            "dollars": round(total * price, 2), "skipped_roles": skipped}


class OAIClient:
    """Minimal /v1/chat/completions client. model=None: the first id from /v1/models."""

    def __init__(self, base_url: str, model: str | None = None, api_key: str = "x", timeout: float = 300,
                 retries: int = 3):
        self.base = base_url.rstrip("/")
        self.api_key, self.timeout, self.retries = api_key, timeout, retries
        self._model = model

    def _req(self, path: str, body: dict | None = None) -> dict:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data, method="GET" if body is None else "POST",
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    @property
    def model(self) -> str:
        if self._model is None:
            self._model = self._req("/models")["data"][0]["id"]
        return self._model

    def chat(self, messages: list[dict], meter: CostMeter | None = None, role: str = "llm", **kw) -> dict:
        """-> {"text", "finish_reason", "usage"}. kw goes into the request body (temperature, max_tokens, seed,
        chat_template_kwargs, ...). Server errors are retried with a short backoff."""
        body = {"model": self.model, "messages": messages, **kw}
        err = None
        for attempt in range(self.retries + 1):
            t0 = time.time()
            try:
                r = self._req("/chat/completions", body)
                break
            except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
                err = e
                if attempt == self.retries:
                    raise
                time.sleep(min(10, 2 * (attempt + 1)))
        else:   # pragma: no cover
            raise err  # type: ignore[misc]
        ch = r["choices"][0]
        text = ch["message"].get("content") or ""
        text = text.split("</think>")[-1]
        u = r.get("usage") or {}
        est = not u
        pt = u.get("prompt_tokens") or sum(est_tokens(str(m.get("content"))) for m in messages)
        ct = u.get("completion_tokens") or est_tokens(text)
        if meter is not None:
            meter.add(role, pt, ct, time.time() - t0, est=est)
        return {"text": text, "finish_reason": ch.get("finish_reason"), "usage": {"prompt_tokens": pt,
                                                                                    "completion_tokens": ct}}
