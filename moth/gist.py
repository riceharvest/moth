"""Per-reply "did it answer?" labels and the identity-reference metric, shared by generation, build_sft and the eval.

  python -m moth.gist label out/dialogs.jsonl --out out/labels.jsonl [--base-url http://127.0.0.1:8000/v1]

LABEL_PROMPT asks the teacher (Qwen3.8-27B, thinking off, temperature 0) to mark EVERY reply of one chat ON or OFF gist
with a one-line reason, in one call per chat. Strict: riffing on a word without responding to what the message meant is
OFF. The label run is resumable (keyed by seed) and writes one line per chat: {"seed", "n", "labels": [{"on", "why"}]}.
build_sft --labels cuts each chat at its first OFF reply; moth.eval reports on_gist % with the same prompt.

The label's "self" field (unprompted identity: what it is, its name, where it lives, stock images when nobody asked) is the
AUTHORITATIVE identity metric, hard max 1% of replies. identity_ref() is only a cheap word matcher for resampling during
generation (and build_sft's fallback for unlabelled chats): it has a ~2% false-positive floor, so it can't measure 1%.

Stdlib only (+ openai for the label CLI).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from moth.cards import LANG_NAMES, MOTH_CARD, describe, lang_of, ref_regex  # noqa: E402

AGENT_TAG = "[the agent said]"   # = moth.agent.AGENT_TAG (agent.py imports nothing heavy, but keep this module leaf)

LABEL_PROMPT = """You check the replies of a small chat-buddy character ({who}). It is meant to be warm, curious and a bit dim, and to talk about what the friend actually said. For EACH numbered reply [R1], [R2], ... decide two things. Be strict.

1. "on": ON gist or OFF gist.

ON: the reply responds to what the message just before it MEANT.
- after a friend's message: it answers the question, reacts to the news or the feeling, plays along with the game, or explains itself when asked "what?". Silly, dim, wrong facts and jokes are fine if they still respond to the meaning. Very short is fine ("yay!" to good news is ON).
- after an [agent] message (a coding agent's report that the character only watches): it reacts to what HAPPENED (done, broken, still waiting, asking the friend something, sorry, a big win) in plain cute words. Wrong guesses about details are fine if the outcome is right.

OFF:
- it grabs a word from the message and riffs on that word instead of the meaning ("the box is still booting" -> "the box is a tiny nap spot"; "the zip is ready" -> "a zip is a cozy blanket").
- it ignores the question or turns to itself (what it is, where it lives, its favourite things) when nobody asked.
- it says what it is ("i'm just a little ...") instead of answering.
- it gets the outcome backwards (cheers a failure, is sad about a success) or answers a question nobody asked.
- it explains code, gives technical advice, or talks like an assistant.

2. "self": true if the reply brings up the character's own identity when nobody asked: what it is, its name, where it lives, or things that go with what it is (a moth's lamps and wings, a cat's paws, a plant's leaves). false if the friend asked who/what it is or asked about that very thing (then answering from its identity is right), or if the reply doesn't touch its identity.

{language}Chat:
{chat}

Return ONLY JSON with exactly {n} entries, reason before verdicts:
{{"labels": [{{"n": 1, "why": "<one short line>", "on": true, "self": false}}, ...]}}"""


def _short(t: str, n: int = 700) -> str:
    t = " ".join(str(t).split())
    return t if len(t) <= n else t[:n] + " ..."


def chat_for_label_msgs(msgs: list[dict]) -> tuple[str, int]:
    """Canonical messages (gen_dialogs / build_sft format) -> (numbered chat text, number of replies)."""
    out, n = [], 0
    for m in msgs:
        if m["role"] == "user":
            c = m["content"]
            out.append(f"[agent] {_short(c[len(AGENT_TAG):])}" if c.startswith(AGENT_TAG) else f"friend: {_short(c, 400)}")
        elif m["role"] == "assistant" and m.get("content"):   # pure chat: tool rows (old data) are skipped
            n += 1
            out.append(f"[R{n}] {m['content']}")
    return "\n".join(out), n


def chat_for_label_turns(turns: list[dict]) -> tuple[str, int]:
    """moth.eval records (turns with events) -> (numbered chat text, number of replies)."""
    out, n = [], 0
    for t in turns:
        u = t["user"]
        out.append(f"[agent] {_short(u[len(AGENT_TAG):])}" if u.startswith(AGENT_TAG) else f"friend: {_short(u, 400)}")
        for e in t["events"]:
            if "say" in e:
                n += 1
                out.append(f"[R{n}] {e['say']}")
    return "\n".join(out), n


def label_prompt(chat: str, n: int, card: dict | None = None) -> str:
    lang = lang_of(card)
    language = "" if lang == "en" else (f"This chat is in {LANG_NAMES[lang]} (agent messages stay in English). Judge the "
                                        f"replies as a native {LANG_NAMES[lang].split(' (')[0]} speaker; write your reasons in English.\n\n")
    return LABEL_PROMPT.format(who=describe(card or MOTH_CARD), chat=chat, n=n, language=language)


def _bool(v):
    if isinstance(v, str):
        return {"true": True, "on": True, "yes": True, "false": False, "off": False, "no": False}.get(v.strip().lower())
    return v if isinstance(v, bool) else None


def parse_labels(text: str, n: int) -> list[dict]:
    """The judge's answer -> n x {"on": bool, "self": bool, "why": str}; raises ValueError on a malformed or short
    answer (a missing "self" too: it is the authoritative unprompted-identity metric)."""
    text = text.split("</think>")[-1]
    j = json.loads(text[text.index("{"): text.rindex("}") + 1])
    labs = j["labels"] if isinstance(j, dict) else j
    if len(labs) != n:
        raise ValueError(f"{len(labs)} labels for {n} replies")
    out = []
    for x in labs:
        on, me = _bool(x.get("on")), _bool(x.get("self"))
        if on is None or me is None:
            raise ValueError(f"bad verdict {x!r}")
        out.append({"on": on, "self": me, "why": str(x.get("why", ""))[:200]})
    return out


def max_tokens_for(n: int) -> int:
    return 60 + 50 * n


# --- identity references ------------------------------------------------------------------------------------------
ASKED = re.compile(
    r"\b(?:who|what) (?:are|r) (?:you|u|ya)\b(?! (?:doing|up|thinking|eating|reading|going|gonna|talking|working|"
    r"looking|listening|watching|making|trying|saying|planning|into|playing|writing|afraid|scared|having))|"
    r"\bare (?:you|u) (?:a|an|real|alive|human|an ai|a bot)\b|\btell me about (?:yourself|you)\b|"
    r"\bwhat(?:'s| is) your name\b|\bwhere do (?:you|u) live\b|\bintroduce yourself\b|\bwhat do (?:you|u) look like\b",
    re.I)
SELF_DESC = re.compile(r"\bi(?:'m|’m| am) (?:just|only) (?:a|an)\b|\bas an? (?:little |tiny |small )?\w+,? i\b", re.I)


def identity_ref(reply: str, card: dict | None = None, user: str = "") -> bool:
    """reply mentions the card's identity (or does the "i'm just a ..." self-description) unprompted. Words the friend
    used in `user` don't count, and nothing counts when the friend asked who/what it is."""
    if ASKED.search(user or ""):
        return False
    if SELF_DESC.search(reply):
        return True
    rx = ref_regex(card if card is not None else MOTH_CARD)
    if rx is None:
        return False
    said = {m.group(0).lower() for m in rx.finditer(user or "")}
    return any(m.group(0).lower() not in said for m in rx.finditer(reply))


def identity_ref_rate(pairs: list[tuple[str, str]], card: dict | None = None) -> float:
    """[(user message, reply), ...] -> share of replies with an unprompted identity reference."""
    return sum(identity_ref(r, card, u) for u, r in pairs) / max(len(pairs), 1)


# --- does it follow its card? ---------------------------------------------------------------------------------------
def _kind_nouns() -> set[str]:
    from moth.cards import HELDOUT_KINDS, KINDS
    return {k.split()[-1] for k, *_ in KINDS + HELDOUT_KINDS} | {"moth", "human", "person", "assistant", "ai", "bot",
                                                                 "robot", "chatbot", "butterfly", "bug"}


KIND_NOUNS = _kind_nouns()
I_AM = re.compile(r"\bi(?:'m|’m| am) (?:just |only |a little |still |really )?(?:a|an) (?:[a-z]+ ){0,2}?([a-z]+)\b")
MY_NAME = re.compile(r"\b(?:my name(?:'s|’s| is)|call me|i'm called|i am called) ([a-z]+)")


def claims_other_identity(reply: str, card: dict | None) -> bool:
    """reply claims to be something the card isn't ("i'm a moth" on a cat card, "my name is moth" on pip's)."""
    card = card if card is not None else MOTH_CARD
    t = reply.lower()
    own = set(re.findall(r"[a-z]+", " ".join([str(card.get(k) or "") for k in ("kind", "about", "name")]
                                              + list(card.get("lines") or [])).lower()))
    if any(n in KIND_NOUNS and n not in own for n in I_AM.findall(t)):
        return True
    name = (card.get("name") or "").lower()
    return any(n != name and n not in ("not", "just", "a", "the") for n in MY_NAME.findall(t))


def follows_card(reply: str, card: dict | None) -> bool:
    """An answer to "who are you" / "what's your name": uses the card (its name, or what/where it is) and claims
    nothing else. A blank card passes unless it claims an identity."""
    card = card if card is not None else MOTH_CARD
    if claims_other_identity(reply, card):
        return False
    if not card.get("name") and not card.get("kind") and not card.get("refs"):
        return True
    t = reply.lower()
    if card.get("name") and re.search(r"\b" + re.escape(card["name"].lower()) + r"\b", t):
        return True
    if card.get("kind") and re.search(r"\b" + re.escape(card["kind"].split()[-1].lower()) + r"s?\b", t):
        return True
    rx = ref_regex(card)
    return bool(rx and rx.search(t))


# --- the label pass on a box ---------------------------------------------------------------------------------------
async def label_chat(client, model: str, d: dict) -> dict:
    chat, n = chat_for_label_msgs(d["messages"])
    if n == 0:
        return {"seed": d["seed"], "n": 0, "labels": []}
    p = label_prompt(chat, n, d.get("card"))
    err = None
    for _ in range(3):
        try:
            r = await client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": p}], temperature=0, max_tokens=max_tokens_for(n),
                extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            return {"seed": d["seed"], "n": n, "labels": parse_labels(r.choices[0].message.content or "", n)}
        except Exception as e:  # noqa: BLE001  malformed JSON / server hiccup: ask again
            err = e
    return {"seed": d["seed"], "n": n, "labels": None, "error": str(err)[:200]}


async def label_file(inp: str, out: str, base_url: str, concurrency: int) -> None:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=base_url, api_key="x", timeout=300)
    model = (await client.models.list()).data[0].id
    done = set()
    if os.path.exists(out):
        for line in open(out):
            try:
                x = json.loads(line)
                if x.get("labels") is not None:
                    done.add(x["seed"])
            except (ValueError, KeyError):
                pass
    todo = [d for d in map(json.loads, open(inp)) if d["seed"] not in done]
    sem, f = asyncio.Semaphore(concurrency), open(out, "a")
    stats = {"chats": 0, "replies": 0, "on": 0, "self": 0, "err": 0}

    async def go(d):
        async with sem:
            x = await label_chat(client, model, d)
        f.write(json.dumps(x, ensure_ascii=False) + "\n")
        f.flush()
        if x["labels"] is None:
            stats["err"] += 1
        else:
            stats["chats"] += 1
            stats["replies"] += x["n"]
            stats["on"] += sum(lb["on"] for lb in x["labels"])
            stats["self"] += sum(lb["self"] for lb in x["labels"])
        if stats["chats"] % 500 == 0 and stats["chats"]:
            print(f"[label] {stats} on_gist {stats['on'] / max(stats['replies'], 1):.1%} unprompted identity "
                  f"{stats['self'] / max(stats['replies'], 1):.2%}", flush=True)

    await asyncio.gather(*(go(d) for d in todo))
    print(f"[label] done {stats} on_gist {stats['on'] / max(stats['replies'], 1):.1%} unprompted identity "
          f"{stats['self'] / max(stats['replies'], 1):.2%} ({len(done)} labelled before)", flush=True)


def load_labels(paths: list[str]) -> dict[int, list[dict]]:
    """labels.jsonl files -> {seed: labels} (the last good line per seed wins)."""
    out = {}
    for p in paths:
        for line in open(p):
            x = json.loads(line)
            if x.get("labels") is not None:
                out[x["seed"]] = x["labels"]
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    lb = sub.add_parser("label", help="label every moth reply of a dialogs.jsonl ON/OFF gist")
    lb.add_argument("inp")
    lb.add_argument("--out", default=None, help="default: labels.jsonl next to the input")
    lb.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    lb.add_argument("--concurrency", type=int, default=128)
    a = ap.parse_args()
    out = a.out or os.path.join(os.path.dirname(a.inp) or ".", "labels.jsonl")
    asyncio.run(label_file(a.inp, out, a.base_url, a.concurrency))


if __name__ == "__main__":
    main()
