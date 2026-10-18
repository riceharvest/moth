"""Per-reply labels for an episode: ONE judge call labels every reply of a chat (moth.gist.LABEL_PROMPT, the same
prompt build_sft filters with and moth.eval reports on_gist % with), cached by hash.

label(card, turns) -> [{"on", "self", "why"}, ...] (one per reply, in order), or None when the judge failed three
times: the reward then masks those components instead of scoring judge noise as off-gist.
Cache key: sha256(judge model id + sha256(LABEL_PROMPT) + rendered prompt), so a changed prompt or judge never hits
an old entry. Default judge model: Qwen3.8-27B, thinking off, temperature 0.

MockJudge is a deterministic heuristic (content-word overlap, agent outcome cues, gist.identity_ref for "self") that
answers in the judge's JSON format, so tests run the real parse path. It is for plumbing only: identity_ref has a
~2% false-positive floor and the overlap rule is crude; never train on it.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading

from moth.agent import AGENT_TAG
from moth.gist import LABEL_PROMPT, chat_for_label_turns, identity_ref, label_prompt, max_tokens_for, parse_labels
from moth.metrics import ADVICE, STOP, words
from moth.rl.llm import CostMeter, OAIClient, est_tokens

JUDGE_MODEL = "Qwen3.8-27B"
PROMPT_HASH = hashlib.sha256(LABEL_PROMPT.encode()).hexdigest()[:12]


class JudgeCache:
    """In-memory dict, optionally backed by an append-only JSON-lines file (resumable across runs)."""

    def __init__(self, path: str | None = None):
        self.path, self.lock, self.d = path, threading.Lock(), {}
        if path and os.path.exists(path):
            for line in open(path, encoding="utf-8"):
                try:
                    x = json.loads(line)
                    self.d[x["k"]] = x["v"]
                except (ValueError, KeyError):
                    pass

    def get(self, k: str):
        with self.lock:
            return self.d.get(k)

    def put(self, k: str, v):
        with self.lock:
            self.d[k] = v
            if self.path:
                os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"k": k, "v": v}, ensure_ascii=False) + "\n")

    def __len__(self):
        return len(self.d)


class Judge:
    model_id = "judge"
    tries = 3

    def __init__(self, cache: JudgeCache | None = None):
        self.cache = cache if cache is not None else JudgeCache()

    def key(self, prompt: str) -> str:
        return hashlib.sha256(f"{self.model_id}\n{PROMPT_HASH}\n{prompt}".encode()).hexdigest()

    def label(self, card: dict, turns: list[dict], meter: CostMeter | None = None) -> list[dict] | None:
        chat, n = chat_for_label_turns(turns)
        if n == 0:
            return []
        prompt = label_prompt(chat, n, card)
        k = self.key(prompt)
        hit = self.cache.get(k)
        if hit is not None:
            if meter is not None:
                meter.add("judge", 0, 0, cached=True)
            return hit
        for attempt in range(self.tries):
            try:
                labs = parse_labels(self._ask(prompt, n, meter, card, turns, attempt), n)
            except (ValueError, KeyError, TypeError):   # malformed / short JSON: ask again
                continue
            self.cache.put(k, labs)
            return labs
        return None

    def _ask(self, prompt, n, meter, card, turns, attempt) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class LLMJudge(Judge):
    def __init__(self, client: OAIClient, cache: JudgeCache | None = None):
        super().__init__(cache)
        self.client = client
        self.model_id = client._model or "auto"

    def _ask(self, prompt, n, meter, card, turns, attempt) -> str:
        return self.client.chat([{"role": "user", "content": prompt}], meter=meter, role="judge", temperature=0,
                                max_tokens=max_tokens_for(n), chat_template_kwargs={"enable_thinking": False})["text"]


# --- the mock ---------------------------------------------------------------------------------------------------
OUTCOMES = {
    "good": r"\b(fixed|done|pass(es|ed)?|green|works?|implemented|finished|in\b)",
    "bad": r"\b(fail(ed|s)?|broke|broken|error|crash|couldn'?t|out of memory|can'?t)",
    "ask": r"\?\s*$|\b(should i|want me to|ok to|which)\b",
    "sorry": r"\b(sorry|apolog)",
    "wait": r"\b(still|booting|waiting|in a minute|later)\b",
}
REACT = {
    "good": r"\b(yay|fixed|did it|done|high five|hooray|it works|finished|proud)",
    "bad": r"\b(oh no|broke|broken|sad|snack break|uh oh|ouch)",
    "ask": r"\b(asking|wants|pick|yes|no)\b|\?",
    "sorry": r"\b(sorry|forgive|it's ok|oops)",
    "wait": r"\b(still|waiting|wake|sleep|minute|later|mornings)",
}
FEEL = [(r"\b(sad|bad|mean|tired|lonely|ugh|rough)\b", r"\b(oh no|sorry|hug|poor|ouch|that's hard|mean)\b"),
        (r"\b(got the job|happy|yay|!!+)", r"\b(yay|wow|proud|congrat|hooray)")]


def _stems(t: str) -> set[str]:
    return {re.sub(r"(es|s|ed|ing)$", "", w) for w in words(t) if len(w) >= 3 and w not in STOP}


def mock_on(user: str, reply: str) -> tuple[bool, str]:
    r = reply.lower()
    if not r.strip() or ADVICE.search(r):
        return False, "mock: empty or advice"
    if user.startswith(AGENT_TAG):
        msg = user[len(AGENT_TAG):]
        want = {k for k, p in OUTCOMES.items() if re.search(p, msg, re.I | re.M)}
        got = {k for k, p in REACT.items() if re.search(p, r)}
        return bool(want & got), f"mock: agent {sorted(want)} vs reply {sorted(got)}"
    if _stems(user) & _stems(reply):
        return True, "mock: shares a content word"
    for pu, pr in FEEL:
        if re.search(pu, user.lower()) and re.search(pr, r):
            return True, "mock: reacts to the feeling"
    if len(user.split()) <= 3 and len(reply.split()) <= 12:
        return True, "mock: short line, short answer"
    return False, "mock: no overlap with the message"


class MockJudge(Judge):
    model_id = "mock-judge"

    def __init__(self, cache: JudgeCache | None = None, fail: bool = False):
        super().__init__(cache)
        self.fail = fail   # tests: always answer garbage

    def _ask(self, prompt, n, meter, card, turns, attempt) -> str:
        labs = []
        for t in turns:
            for e in t["events"]:
                if e.get("say") is None or "tool" in e:
                    continue
                me = identity_ref(e["say"], card, t["user"])
                on, why = mock_on(t["user"], e["say"])
                labs.append({"n": len(labs) + 1, "why": why + ("; talks about itself" if me else ""),
                             "on": on and not me, "self": me})
        out = "sorry, no json" if self.fail else json.dumps({"labels": labs})
        if meter is not None:
            meter.add("judge", est_tokens(prompt), est_tokens(out), est=True)
        return out
