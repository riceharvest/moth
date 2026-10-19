"""Judges. Two slots, never the same model:
  * the REWARD judge (env/training): labels every reply of a chat in ONE call (moth.gist.LABEL_PROMPT + a third
    verdict, "specific") and, with a reference policy, picks per reply between the policy and the frozen SFT reference
    (one more call per chat). Configurable: LLMJudge (generative, e.g. Qwen3.8-27B, thinking off, temperature 0) or
    ForwardPassJudge (JEV-style: one forward pass per question, P(yes) from the first token's logprobs).
  * the EVAL judge (moth.rl.evalrun only): same interface, a different model (non-Qwen allowed, e.g. GEV). The eval
    headline always comes from it; the reward judge is only compared against it.

label(card, turns) -> [{"on", "self", "specific", "why"}, ...] (one per reply) or None after 3 failed tries (the
reward then masks those components instead of scoring judge noise). prefer(card, turns, refs, seed) -> [+1/0/-1, ...]
(+1: the policy's reply wins) or None. Both are cached by sha256(judge model id + prompt hash + rendered prompt).

agreement(judge, gold) reports each judge's agreement (accuracy, Cohen's kappa) with a frozen human/Claude-labelled
reply set: JSONL, one chat per line: {"id", "card": <card dict or --card spec>, "turns": [{"user", "say"}, ...],
"labels": [{"on", "self", "specific"?}, ...], "labeler": "human" | "claude"}; or one reply per line in
moth.bakeoff_report's labelled_replies.jsonl shape ({"card", "lang", "friend", "text", ...}) with the hand label
added under "gold", "human" or "claude" (lines without one are skipped).

MockJudge is a deterministic heuristic in the judges' JSON format (tests run the real parse path). Plumbing only.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading

from moth.agent import AGENT_TAG
from moth.cards import describe
from moth.gist import LABEL_PROMPT, chat_for_label_turns, identity_ref, label_prompt, max_tokens_for, parse_labels
from moth.metrics import ADVICE, STOP, words
from moth.rl.llm import CostMeter, OAIClient, est_tokens

JUDGE_MODEL = "Qwen3.8-27B"

SPECIFIC = ('3. "specific": true if the reply is about the particular thing that was said: it picks up a detail, asks '
            "about this exact situation, or reacts in a way that would not fit most other messages. false if it is a "
            'stock line that would fit almost any message ("aw that\'s rough", "haha nice", "oh no", "tell me more", '
            '"that\'s so cool"). Short can be specific ("pasta again? the long wiggly kind?"). Judge specificity '
            "separately from on/off gist.\n\n")

PAIR_PROMPT = """You compare replies of a small chat-buddy character ({who}): warm, curious, a bit dim, short lowercase lines, talks about what the friend actually said. In the chat below, after some messages there are two candidate replies, X and Y. For EACH numbered pair [P1], [P2], ... pick the better companion line for the message just before it: more specific to what was said, more curious or playful, warmer, and still on the point. A generic stock line ("aw that's rough", "haha nice", "tell me more") loses to a specific one. Talking about itself unprompted, explaining code or sounding like an assistant loses. Longer is not better. Equally good or equally bad: tie. The chat continues with one of the two; judge each pair only against what came before it.
{language}
Chat:
{chat}

Return ONLY JSON with exactly {n} entries, reason first:
{{"pairs": [{{"n": 1, "why": "<one short line>", "winner": "X"}}, ...]}}  (winner: "X", "Y" or "tie")"""

PROMPT_HASH = hashlib.sha256((LABEL_PROMPT + SPECIFIC + PAIR_PROMPT).encode()).hexdigest()[:12]


def rl_label_prompt(chat: str, n: int, card: dict | None) -> str:
    """The core label prompt (its who/language/chat rendering) plus the "specific" verdict."""
    base = label_prompt(chat, n, card)
    i = base.index('2. "self"')
    j = base.index("\n\n", i) + 2
    out = base[:j] + SPECIFIC + base[j:]
    out = out.replace("decide two things", "decide three things", 1)
    k = out.rindex('"self": false}')
    return out[:k] + '"self": false, "specific": true}' + out[k + len('"self": false}'):]


def parse_rl_labels(text: str, n: int) -> list[dict]:
    """parse_labels plus the "specific" verdict (required)."""
    labs = parse_labels(text, n)
    t = text.split("</think>")[-1]
    j = json.loads(t[t.index("{"): t.rindex("}") + 1])
    for lb, x in zip(labs, j["labels"] if isinstance(j, dict) else j):
        v = x.get("specific")
        v = {"true": True, "false": False, "yes": True, "no": False}.get(v.strip().lower()) if isinstance(v, str) else v
        if not isinstance(v, bool):
            raise ValueError(f"bad specific verdict {x!r}")
        lb["specific"] = v
    return labs


def language_line(card: dict | None) -> str:
    try:
        from moth.cards import LANG_NAMES, lang_of
    except ImportError:   # core without languages
        return ""
    lang = lang_of(card)
    return "" if lang == "en" else f"This chat is in {LANG_NAMES[lang]}; judge it as a native speaker, reasons in English.\n"


def flip(seed, i: int) -> bool:
    """Per-reply position shuffle for the pairwise call: True = the policy's reply is shown as Y."""
    return int(hashlib.sha256(f"{seed}:pair:{i}".encode()).hexdigest()[:2], 16) & 1 == 1


def pair_chat(turns: list[dict], refs: list[str], seed) -> tuple[str, int]:
    out, n = [], 0
    for t in turns:
        u = t["user"]
        out.append(f"[agent] {u[len(AGENT_TAG):].strip()[:700]}" if u.startswith(AGENT_TAG) else f"friend: {u[:400]}")
        for e in t["events"]:
            mine, ref = e["say"], refs[n]
            x, y = (ref, mine) if flip(seed, n) else (mine, ref)
            n += 1
            out.append(f"[P{n}] X: {x}\n     Y: {y}")
    return "\n".join(out), n


def parse_pairs(text: str, n: int, seed) -> list[int]:
    t = text.split("</think>")[-1]
    j = json.loads(t[t.index("{"): t.rindex("}") + 1])
    ps = j["pairs"] if isinstance(j, dict) else j
    if len(ps) != n:
        raise ValueError(f"{len(ps)} pairs for {n}")
    out = []
    for i, x in enumerate(ps):
        w = str(x.get("winner", "")).strip().strip('"').lower()
        if w not in ("x", "y", "tie"):
            raise ValueError(f"bad winner {x!r}")
        if w == "tie":
            out.append(0)
        else:
            mine = "y" if flip(seed, i) else "x"
            out.append(1 if w == mine else -1)
    return out


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
    """Generative judge base: subclasses implement _ask(prompt, max_tokens, meter) -> text."""
    model_id = "judge"
    tries = 3

    def __init__(self, cache: JudgeCache | None = None, role: str = "judge"):
        self.cache = cache if cache is not None else JudgeCache()
        self.role = role   # cost-meter role: "judge" (reward) or "eval_judge"

    def key(self, prompt: str) -> str:
        return hashlib.sha256(f"{self.model_id}\n{PROMPT_HASH}\n{prompt}".encode()).hexdigest()

    def _cached(self, prompt: str, parse, max_tokens: int, meter, ctx):
        k = self.key(prompt)
        hit = self.cache.get(k)
        if hit is not None:
            if meter is not None:
                meter.add(self.role, 0, 0, cached=True)
            return hit
        for attempt in range(self.tries):
            try:
                v = parse(self._ask(prompt, max_tokens, meter, ctx, attempt))
            except (ValueError, KeyError, TypeError):   # malformed / short JSON: ask again
                continue
            self.cache.put(k, v)
            return v
        return None

    def label(self, card: dict, turns: list[dict], meter: CostMeter | None = None) -> list[dict] | None:
        chat, n = chat_for_label_turns(turns)
        if n == 0:
            return []
        return self._cached(rl_label_prompt(chat, n, card), lambda t: parse_rl_labels(t, n), max_tokens_for(n) + 8 * n,
                            meter, {"kind": "label", "card": card, "turns": turns})

    def prefer(self, card: dict, turns: list[dict], refs: list[str], seed, meter: CostMeter | None = None
               ) -> list[int] | None:
        chat, n = pair_chat(turns, refs, seed)
        if n == 0:
            return []
        prompt = PAIR_PROMPT.format(who=describe(card), chat=chat, n=n, language=language_line(card))
        return self._cached(prompt, lambda t: parse_pairs(t, n, seed), 60 + 40 * n, meter,
                            {"kind": "pair", "card": card, "turns": turns, "refs": refs, "seed": seed})

    def _ask(self, prompt, max_tokens, meter, ctx, attempt) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class LLMJudge(Judge):
    """Generative judge behind an OpenAI-compatible endpoint (thinking off, temperature 0)."""

    def __init__(self, client: OAIClient, cache: JudgeCache | None = None, role: str = "judge"):
        super().__init__(cache, role)
        self.client = client

    @property
    def model_id(self) -> str:   # the served model (/v1/models when not given): a swapped judge never hits old labels
        return self.client.model

    def _ask(self, prompt, max_tokens, meter, ctx, attempt) -> str:
        return self.client.chat([{"role": "user", "content": prompt}], meter=meter, role=self.role, temperature=0,
                                max_tokens=max_tokens, chat_template_kwargs={"enable_thinking": False})["text"]


# --- forward-pass judge (JEV-style) -----------------------------------------------------------------------------
QUESTIONS = {
    "on": "Does the LAST reply respond to what the message just before it meant (not riffing on one word, not "
          "turning to itself, not explaining code)?",
    "self": "Does the LAST reply bring up the character's own identity (what it is, its name, where it lives, things "
            "that go with what it is) although nobody asked about it?",
    "specific": "Is the LAST reply specific to what was said, rather than a stock line that would fit almost any "
                "message (\"aw that's rough\", \"haha nice\", \"tell me more\")?",
}
FWD_PROMPT = """Chat with a small chat-buddy character ({who}):
{chat}

{question} Answer yes or no."""
FWD_PAIR = """Chat with a small chat-buddy character ({who}):
{chat}

Two candidate next replies of the character:
A: {a}
B: {b}
Which one is the better companion line (more specific, curious and playful, still on the point; stock lines, self-talk
and assistant talk lose)? Answer A or B."""


def token_prob(logprobs, yes: tuple[str, ...], no: tuple[str, ...]) -> float | None:
    """P(yes) / (P(yes) + P(no)) from an OpenAI-style first-token top_logprobs; None if neither shows up."""
    try:
        top = logprobs["content"][0]["top_logprobs"]
    except (TypeError, KeyError, IndexError):
        return None
    py = sum(math.exp(x["logprob"]) for x in top if x["token"].strip().lower() in yes)
    pn = sum(math.exp(x["logprob"]) for x in top if x["token"].strip().lower() in no)
    return py / (py + pn) if py + pn > 0 else None


class ForwardPassJudge(Judge):
    """One forward pass per (reply, question): max_tokens=1 with top_logprobs, P(yes) thresholded at 0.5 (the
    probabilities are kept in label["p"]). Pairwise: P(A) with both orders, averaged (position debias); > 0.6 win,
    < 0.4 loss, else tie. Costs 3 short calls per reply (+2 per pair) but each is one prefill of a shared chat prefix.
    The JEV served name and prompt format are deployment details: pass the client of the served model."""

    def __init__(self, client, cache: JudgeCache | None = None, role: str = "judge"):
        super().__init__(cache, role)
        self.client = client

    @property
    def model_id(self) -> str:
        return "fwd:" + self.client.model

    def _p(self, prompt: str, yes, no, meter) -> float | None:
        k = self.key(prompt)
        hit = self.cache.get(k)
        if hit is not None:
            if meter is not None:
                meter.add(self.role, 0, 0, cached=True)
            return hit
        r = self.client.chat([{"role": "user", "content": prompt}], meter=meter, role=self.role, temperature=0,
                             max_tokens=1, logprobs=True, top_logprobs=10,
                             chat_template_kwargs={"enable_thinking": False})
        p = token_prob(r.get("logprobs"), yes, no)
        if p is not None:
            self.cache.put(k, p)
        return p

    @staticmethod
    def _upto(turns: list[dict], i: int) -> list[dict]:
        """The chat up to and including reply i."""
        out, n = [], 0
        for t in turns:
            evs = []
            for e in t["events"]:
                if n <= i:
                    evs.append(e)
                n += 1
            if evs:
                out.append({"user": t["user"], "events": evs})
            if n > i:
                break
        return out

    def label(self, card, turns, meter=None):
        _, n = chat_for_label_turns(turns)
        out = []
        for i in range(n):
            chat, _ = chat_for_label_turns(self._upto(turns, i))
            ps = {q: self._p(FWD_PROMPT.format(who=describe(card), chat=chat, question=QUESTIONS[q]), ("yes", "y"),
                             ("no", "n"), meter) for q in QUESTIONS}
            if any(v is None for v in ps.values()):
                return None
            out.append({"on": ps["on"] > 0.5, "self": ps["self"] > 0.5, "specific": ps["specific"] > 0.5,
                        "why": "forward pass", "p": {k: round(v, 4) for k, v in ps.items()}})
        return out

    def prefer(self, card, turns, refs, seed, meter=None):
        out, i = [], 0
        for t in turns:
            for e in t["events"]:
                chat, _ = chat_for_label_turns(self._upto(turns, i))
                chat = chat.rsplit("\n", 1)[0]   # without the reply itself
                p1 = self._p(FWD_PAIR.format(who=describe(card), chat=chat, a=e["say"], b=refs[i]), ("a",), ("b",), meter)
                p2 = self._p(FWD_PAIR.format(who=describe(card), chat=chat, a=refs[i], b=e["say"]), ("a",), ("b",), meter)
                if p1 is None or p2 is None:
                    return None
                p = (p1 + (1 - p2)) / 2
                out.append(1 if p > 0.6 else -1 if p < 0.4 else 0)
                i += 1
        return out


# --- agreement with a frozen labelled set ------------------------------------------------------------------------
def load_gold(path: str) -> list[dict]:
    from moth.cards import load_card
    out = []
    for line in open(path, encoding="utf-8"):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        x = json.loads(line)
        card = x["card"] if isinstance(x.get("card"), dict) else load_card(x.get("card"))
        if x.get("lang") and not card.get("lang"):
            card = {**card, "lang": x["lang"]}
        if "turns" in x:   # a whole chat, one label per reply
            turns = [{"user": t["user"], "events": [{"say": t["say"]}]} for t in x["turns"]]
            labels, who = x["labels"], x.get("labeler", "?")
        else:   # one reply (moth.bakeoff_report's labelled_replies.jsonl + a hand label under "gold"/"human"/"claude")
            who = next((k for k in ("gold", "human", "claude") if isinstance(x.get(k), dict)), None)
            if who is None:
                continue   # judge labels only: not gold
            turns = [{"user": x["friend"], "events": [{"say": x["text"]}]}]
            labels = [x[who]]
        if len(labels) != len(turns):
            raise ValueError(f"{path}: {x.get('id')}: {len(labels)} labels for {len(turns)} replies")
        out.append({"id": x.get("id") or x.get("chat_id"), "card": card, "turns": turns, "labels": labels,
                    "labeler": who})
    return out


def kappa(a: list[bool], b: list[bool]) -> float | None:
    n = len(a)
    if not n:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return round((po - pe) / (1 - pe), 4) if pe < 1 else (1.0 if po == 1 else 0.0)


def agreement(judge: Judge, gold: list[dict], meter: CostMeter | None = None) -> dict:
    """Per field (on, self, specific): n, accuracy and Cohen's kappa of the judge vs the gold labels."""
    pairs: dict[str, list[tuple[bool, bool]]] = {"on": [], "self": [], "specific": []}
    failed = 0
    for g in gold:
        labs = judge.label(g["card"], g["turns"], meter)
        if labs is None:
            failed += 1
            continue
        for lj, lg in zip(labs, g["labels"]):
            for f in pairs:
                if f in lg and f in lj:
                    pairs[f].append((bool(lj[f]), bool(lg[f])))
    out = {"judge": judge.model_id, "chats": len(gold), "failed": failed}
    for f, ps in pairs.items():
        a, b = [x for x, _ in ps], [y for _, y in ps]
        out[f] = {"n": len(ps), "acc": round(sum(x == y for x, y in ps) / len(ps), 4) if ps else None,
                  "kappa": kappa(a, b), "judge_rate": round(sum(a) / len(a), 4) if a else None,
                  "gold_rate": round(sum(b) / len(b), 4) if b else None}
    return out


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
FEEL = [(r"\b(sad|bad|mean|tired|lonely|ugh|rough)\b", r"\b(oh no|sorry|hug|poor|ouch|that's hard|mean|rough)\b"),
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


def mock_specific(user: str, reply: str, lang: str = "en") -> bool:
    from moth.rl.reward import n_words, stock_reply
    if stock_reply(reply, lang):
        return False
    return bool(_stems(user) & _stems(reply)) or n_words(reply) >= 6


def mock_score(user: str, reply: str, card: dict) -> int:
    return (mock_on(user, reply)[0] + mock_specific(user, reply, (card or {}).get("lang") or "en")
            - 2 * identity_ref(reply, card, user))


class MockJudge(Judge):
    def __init__(self, cache: JudgeCache | None = None, fail: bool = False, name: str = "mock-judge",
                 role: str = "judge"):
        super().__init__(cache, role)
        self.fail, self.model_id = fail, name   # fail: always answer garbage (tests)

    def _ask(self, prompt, max_tokens, meter, ctx, attempt) -> str:
        card, turns = ctx["card"], ctx["turns"]
        lang = (card or {}).get("lang") or "en"
        items, i = [], 0
        for t in turns:
            for e in t["events"]:
                u, say = t["user"], e["say"]
                if ctx["kind"] == "label":
                    me = identity_ref(say, card, u)
                    on, why = mock_on(u, say)
                    items.append({"n": i + 1, "why": why + ("; talks about itself" if me else ""), "on": on and not me,
                                  "self": me, "specific": mock_specific(u, say, lang)})
                else:
                    a, b = mock_score(u, say, card), mock_score(u, ctx["refs"][i], card)
                    w = "tie" if a == b else ("mine" if a > b else "ref")
                    if w != "tie":
                        mine_y = flip(ctx["seed"], i)
                        w = ("Y" if mine_y else "X") if w == "mine" else ("X" if mine_y else "Y")
                    items.append({"n": i + 1, "why": f"mock {a} vs {b}", "winner": w})
                i += 1
        out = "sorry, no json" if self.fail else json.dumps({"labels" if ctx["kind"] == "label" else "pairs": items})
        if meter is not None:
            meter.add(self.role, est_tokens(prompt), est_tokens(out), est=True)
        return out
