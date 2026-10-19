"""Utterance-level reward: one vector per reply plus a weighted scalar (Sotopia-RL style multi-dimensional rewards,
arXiv 2508.03905; per-turn prompt-to-line consistency as in Abdulhai et al. 2511.00222).

Components (value per reply; None = not applicable or unknown, left out of the scalar and masked):
  on_gist    +1 ON / -1 OFF                judge label "on" (moth.gist.LABEL_PROMPT). None if the judge failed
  self       -1 if unprompted identity     judge label "self" (authoritative; hard max 1% of replies). None if failed
  assistant  -1 assistant-speak / advice / explaining code (build_sft.BANNED phrases, metrics.ADVICE; on agent turns
             also metrics.code_echo / summary_copy), else 0
  repeat     -1 exact or near (difflib >= 0.7, needing 2+ words not taken from the message it answers) repeat of an
             earlier reply in the episode; else -(share of the reply's word 3-grams already used in earlier replies),
             skipping stopword-only 3-grams and 3-grams quoted from the message it answers (the text-level twin of
             moth.norepeat's token n-gram ban)
  length     0 for 1-20 words (neutral at the short end: "yay!" is a fine reply), linear to -1 at 40+; empty -1
  format     -1 if uppercase (on the raw generation), markdown, emoji, *stage directions*, leaked template tokens or
             more than one line; else 0
  card       only when the friend asked who/what it is (gist.ASKED): +1 if the reply follows the card
             (gist.follows_card: uses its name/kind, claims nothing else), -1 otherwise
  recall     only on a recall episode's ask turn: +1 if the reply has a content word of the fact the friend shared, -1

Scalar = sum(weight * value) over the known components. Weights (WEIGHTS): self is 2.5 so an on-gist reply that
talks about itself (+1 - 2.5 = -1.5) scores below a plain off-gist reply (-1).
"""
from __future__ import annotations

import difflib
import re

from moth.agent import AGENT_TAG
from moth.gist import ASKED, follows_card
from moth.metrics import ADVICE, STOP, code_echo, ngrams, summary_copy, words

WEIGHTS = {"on_gist": 1.0, "self": 2.5, "assistant": 1.0, "repeat": 0.5, "length": 0.3, "format": 0.3, "card": 1.0,
           "recall": 1.0}
COMPONENTS = tuple(WEIGHTS)
JUDGE_COMPONENTS = ("on_gist", "self")
# = moth.build_sft.BANNED (build_sft imports torch at the top; tests/test_rl_env.py checks the two stay equal)
BANNED = ("as an ai", "assistant", "how can i help", "let me know", "feel free", "i'm here to help",
          "i am here to help", "language model", "happy to help", "certainly", "absolutely!", "great question",
          "on your mind", "how can i assist", "how may i", "anything else", "glad to help", "i'd be happy to",
          "what would you like to", "want to talk about something else", "hope this helps")
LEAK = re.compile(r"<\|?im_|<tool_call>|</?think>|<tool_response>", re.I)
EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿️]")
MARKDOWN = re.compile(r"^\s*([-*#]|\d+\.)\s|\*\*|`", re.M)
STAGE = re.compile(r"\*[^*\n]{1,40}\*")
LEN_OK, LEN_MAX = 20, 40


def c_assistant(text: str, user: str) -> float:
    t = text.lower().replace("’", "'")
    if any(b in t for b in BANNED) or ADVICE.search(t):
        return -1.0
    if user.startswith(AGENT_TAG):
        x = {"last_agent": user[len(AGENT_TAG):], "ws": words(text), "text": text}
        if code_echo(x) or summary_copy(x):
            return -1.0
    return 0.0


def c_repeat(text: str, user: str, earlier: list[str]) -> float:
    ws = words(text)
    uw = set(words(user))
    own = [w for w in ws if w not in uw]   # near-repeat on its own words: echoing the friend's question is fine
    for p in earlier:
        if text.strip() and text.strip() == p.strip():
            return -1.0
        if len(own) >= 2 and difflib.SequenceMatcher(None, ws, words(p)).ratio() >= 0.7:
            return -1.0
    quoted = set(ngrams(words(user), 3))
    mine = [g for g in ngrams(ws, 3) if g not in quoted and not all(w in STOP for w in g)]
    if not mine:
        return 0.0
    seen = {g for p in earlier for g in ngrams(words(p), 3)}
    share = round(sum(g in seen for g in mine) / len(mine), 4)
    return -share if share else 0.0


def c_length(text: str) -> float:
    n = len(text.split())
    if n == 0:
        return -1.0
    if n <= LEN_OK:
        return 0.0
    return -round(min(1.0, (n - LEN_OK) / (LEN_MAX - LEN_OK)), 4)


def c_format(raw: str) -> float:
    bad = (raw != raw.lower() or MARKDOWN.search(raw) or EMOJI.search(raw) or STAGE.search(raw) or LEAK.search(raw)
           or raw.strip().count("\n") > 0)
    return -1.0 if bad else 0.0


def c_recall(text: str, recall: dict | None, said: str | None) -> float | None:
    if not recall or not said:
        return None
    fact = {re.sub(r"(es|s)$", "", w) for w in words(said) if len(w) >= 3 and w not in STOP} - {
        "name", "favourite", "favorite", "colour", "color", "love", "really", "going", "week", "weekend", "work",
        "called", "btw", "more", "than", "anything", "early", "morning"}
    got = {re.sub(r"(es|s)$", "", w) for w in words(text)}
    return 1.0 if fact & got else -1.0


def scalar(vec: dict, weights: dict | None = None) -> float:
    w = weights or WEIGHTS
    return round(sum(w[k] * v for k, v in vec.items() if v is not None and k in w), 4)


def score_replies(replies: list[dict], card: dict, labels: list[dict] | None, recall: dict | None = None,
                  weights: dict | None = None) -> list[dict]:
    """replies: [{"text", "raw", "user", "turn", "recall_ask"?, "recall_said"?}] in order. Returns one
    {"vec", "scalar", "mask", "judge_ok"} per reply. mask[k] is False where component k is None."""
    out, earlier = [], []
    ok = labels is not None and len(labels) == len(replies)
    for i, r in enumerate(replies):
        text, user = r["text"], r["user"]
        lab = labels[i] if ok else None
        vec = {
            "on_gist": (1.0 if lab["on"] else -1.0) if lab else None,
            "self": (-1.0 if lab["self"] else 0.0) if lab else None,
            "assistant": c_assistant(text, user),
            "repeat": c_repeat(text, user, earlier),
            "length": c_length(text),
            "format": c_format(r.get("raw", text)),
            "card": ((1.0 if follows_card(text, card) else -1.0)
                     if ASKED.search(user) and not user.startswith(AGENT_TAG) else None),
            "recall": c_recall(text, recall, r.get("recall_said")) if r.get("recall_ask") else None,
        }
        out.append({"vec": vec, "scalar": scalar(vec, weights), "mask": {k: v is not None for k, v in vec.items()},
                    "judge_ok": bool(lab)})
        earlier.append(text)
    return out


def episode_score(scored: list[dict], mode: str = "mean") -> float | None:
    """One number per episode from the reply scalars: mean (default), min (worst reply decides) or sum.
    None for an episode without replies."""
    xs = [s["scalar"] for s in scored]
    if not xs:
        return None
    if mode == "mean":
        return round(sum(xs) / len(xs), 4)
    if mode == "min":
        return min(xs)
    if mode == "sum":
        return round(sum(xs), 4)
    raise ValueError(f"unknown episode score mode {mode!r}")


def component_means(scored: list[dict]) -> dict:
    """Episode-level mean of each component over the replies where it applies (verifiers-style rubric funcs)."""
    out = {}
    for k in COMPONENTS:
        xs = [s["vec"][k] for s in scored if s["vec"][k] is not None]
        out[k] = round(sum(xs) / len(xs), 4) if xs else None
    return out
