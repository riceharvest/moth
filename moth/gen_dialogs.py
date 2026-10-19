"""Write moth training chats with a teacher behind an OpenAI-compatible server (vLLM).

  python -m moth.gen_dialogs --dialogs 12000 --start 0           # chats -> out/dialogs.jsonl (resumable)

A PURE chatbot (2026-10-19): no tools, no home folder. One chat: sample an identity card (cards.py) and a (persona,
scenario, length); the teacher plays the human (USER_SIM_PROMPT) and the chat buddy (teacher_system(card) = the
shared TEACHER_CORE + the card). Chat kinds: talk (some long, some with a recall check) and agent reactions. Records
use tiny_agent's canonical message format (assistant tool_calls always empty) plus "card"; filtering and lowercasing
happen later in build_sft.py (with the per-reply labels of moth.gist, if given).
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import random
import re
import sys
import time

from openai import AsyncOpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from moth.agent import (AGENT_CHAT_REMIND, AGENT_LENGTHS, AGENT_PROMPT, AGENT_REMIND,  # noqa: E402
                        AGENT_RULES, AGENT_SCENARIOS, AGENT_TAG, AGENT_TASKS, AGENT_USER_PERSONAS, HARNESS_STYLES,
                        OUTCOMES, agent_turn)
from moth.cards import BLANK_CARD, LANG_NAMES, describe, display_name, is_moth, lang_of, sample_card  # noqa: E402
from moth.gist import ASKED, identity_ref  # noqa: E402
from moth.persona import (SCENARIOS, TEACHER_BIBLE, TOPICS, USER_PERSONAS, USER_SIM_PROMPT,  # noqa: E402
                          teacher_system)

SIM_SWITCH = ("[DIRECTOR NOTE: for this message only, you are NOT the chat-buddy character. You write the HUMAN side of "
              "the chat. Output only the human's next message.]\n\n")
SIM_SYSTEM = ("You write one side of a chat for a story: the HUMAN who is chatting with a tiny cute chatbot. "
              "You are never the chatbot. Type like a real person in a chat app, in the voice of the person you play: a grandparent "
              "or formal person writes proper sentences with capitals and punctuation, a kid or teen types lowercase with slang, "
              "a sloppy typer makes typos. Short (often 1-10 words), no greeting after the first message, rarely emojis, never "
              "the same message twice. Reply with only the human's next chat message: no preface, no quotes, no notes.")
SIM_SHORT = ("\n\nIMPORTANT: this time write a tiny lazy reply of 1 to 5 words to the bot's last message, typed your "
             "way, that sounds {mood}. if it names something, use a word the bot really said.")   # r3: literal examples got copied
SHORT_MOODS = ["amused", "doubtful", "impressed", "teasing", "agreeing", "curious about one word the bot used", "bored",
               "surprised", "touched", "a bit confused", "playfully annoyed", "sleepy", "delighted", "unconvinced"]
# r3: "what lamp" x1346, mostly about words moth never said: a share of the confusion turns is built in code from
# moth's real last reply (no teacher call)
WHAT_X = ["what {x}?", "what {x}", "wait, {x}?", "{x}?", "what do you mean {x}?", "huh, {x}?", "What {x}?", "a {x}?",
          "{x}? what", "why {x}?", "{x}?? lol", "what's {x}"]
WHAT_GENERIC = ["what?", "huh?", "what do you mean?", "wdym", "What do you mean?", "Pardon?", "wait what", "??",
                "i don't get it", "what does that mean", "hm?", "Sorry, what?"]
WORD_STOP = set("that this with have from they them then than there their what when where which would could should "
                "about maybe think really very just like little tiny small some much many more here know your yours "
                "mine been being sure okay also still only once ever never always good nice thing things something "
                "sorry thank thanks wait well kind sort feel feels want need make made said says look tell told it's "
                "i'm don't that's what's you're can't didn't isn't doesn't i'll i've let's every other same into over "
                "back down come came goes going gone does done want these those because while again even most "
                "moth friend idea".split())
DET = set("the a an my your his her our some this that these those its their any no".split())
ADJ = set("little big tiny small soft warm old new nice good long short whole same other only first last best very cosy "
          "cozy fuzzy dusty bright dark sleepy happy sweet great real funny tall round".split())
WHATX_RE = re.compile(r"\b(?:what|which|why)(?: is| a| an| the)? ([a-z']+)\W*$", re.I)
RECALL_FACTS = [("your name (make one up)", "your name"), ("your pet's name and what animal it is (make it up)", "your pet"),
                ("your favourite food", "your favourite food"), ("one small thing that happened to you today",
                "what happened to you today"), ("your favourite colour", "your favourite colour"),
                ("where you're going this weekend", "your weekend plan"), ("what your job is", "your job"),
                ("your best friend's name (make one up)", "your best friend")]
FACT_WORDS = {"name", "favourite", "favorite", "food", "colour", "color", "pet", "today", "friend", "best", "job", "weekend",
              "work", "love", "really", "go", "call", "happen", "moth", "thing", "day"}
RECALL_SHARE = {"p": 0.15}   # --recall-share: talk chats where the human shares a fact early and asks for it 5+ turns later
LONG_SHARE = {"p": 0.15}     # --long-share: talk chats with 16-24 user turns
SIM_NO_REPEAT = "\n\nYou already sent: {prev}. Write something new, not any of these."
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}
# teacher view only (appended to each human turn, never in the training text). TALK_REMIND_ID when the friend asked
# who/what it is, else TALK_REMIND: unprompted identity talk is capped at 1% of replies (label pass "self")
TALK_REMIND = ("\n\n(answer what they just said or meant first, in one short lowercase line, usually under 15 words. "
               "no talk about what you are, your name or where you live: they didn't ask.)")
TALK_REMIND_ID = ("\n\n(answer what they just said first, one short lowercase line. they asked who or what you are: "
                  "answer from WHO YOU ARE in a few plain words, then back to them.)")
MOTH_REMIND = TALK_REMIND   # old name
QBACK = ("\n(this time hand the turn back, on the same line: end with one small question about them, or bring back "
         "something they told you earlier.)")
NO_QBACK = "\n(this time just answer: no question at the end, unless a game needs one.)"   # r4 test: 61% asked back
RECALL_REMIND = "\n(they told you this earlier in this chat: answer from the chat.)"
LEAKS = ("<think>", "</think>", "<|tool_call", "<tool_call>", "<function", "<|im_")   # reasoning or raw tool-call markup in the reply = broken call
TURNS = [3, 4, 5, 6, 6, 7, 8, 8, 9, 10, 12, 14]   # back-and-forth: the thing round 2 couldn't do
THINK = {"on": False, "budget": 256}   # --think: teacher always reasons first (LFM2.5), capped at --think-budget tokens
TURN_CHOICES: list[int] = []   # --turns: override the chat-length distribution (e.g. long chats)
SCENARIO_IDS: list[int] = []   # --scenario-ids: restrict a run to some scenarios (e.g. small talk only)


def _sft_keep():
    """build_sft.keep (the build-time filter), so moth_turn resamples replies it would cut. build_sft imports torch,
    tokenizers and tiny_agent at the top for the token rows only: stub whichever this env lacks."""
    import importlib.util
    import types
    try:
        for name, attrs in (("torch", {}), ("tokenizers", {"Tokenizer": object})):
            if name not in sys.modules and importlib.util.find_spec(name) is None:
                sys.modules[name] = types.ModuleType(name)
                sys.modules[name].__dict__.update(attrs)
        if "tiny_agent.chat" not in sys.modules:
            sys.modules.setdefault("tiny_agent", types.ModuleType("tiny_agent"))
            sys.modules["tiny_agent.chat"] = types.ModuleType("tiny_agent.chat")
            sys.modules["tiny_agent.chat"].render = None
        from moth.build_sft import keep
        return keep
    except Exception as e:  # noqa: BLE001  build_sft changed shape: generate without the check rather than crash
        print(f"WARNING: build_sft.keep not importable ({type(e).__name__}: {e}); no build-time checks", flush=True)
        return None


SFT_KEEP = _sft_keep()


def sft_problem(meta: dict, msgs: list[dict], text: str) -> str | None:
    """Why build_sft would cut the chat at this new reply (None if it keeps it, or the chat is already cut earlier)."""
    if SFT_KEEP is None:
        return None

    def cut(t):
        why = collections.Counter()
        m = msgs + [{"role": "assistant", "think": None, "content": t, "tool_calls": []}]
        out = SFT_KEEP({**meta, "messages": m}, why)
        return None if out is not None and len(out) == len(m) else (next(iter(why), "cut").split(":")[0])
    try:
        r = cut(text)
        return r if r and not cut("ok.") else None   # "ok." cut too: the problem is earlier (a user turn), not here
    except Exception:  # noqa: BLE001
        return None


def transcript(msgs: list[dict], name: str = "the bot") -> str:
    out = []
    for m in msgs:
        if m["role"] == "user" and m["content"].startswith(AGENT_TAG):
            body = m["content"][len(AGENT_TAG):].strip()
            out.append(f"[your AI agent's latest message to you, which {name} can also see]\n" + body[:600])
        elif m["role"] == "user":
            out.append(f"you: {m['content']}")
        elif m["role"] == "assistant" and m.get("content"):
            out.append(f"{name}: {m['content']}")
    return "\n".join(out) or "(nothing yet: write your first message)"


async def chat(client, model, messages, budget=None, **kw):
    # --think: LFM2.5's template always opens <think>. Never send enable_thinking=False then: vLLM's qwen3 reasoning parser
    # would leave the whole reasoning in .content. vLLM's thinking_token_budget forces </think> after `budget` tokens.
    extra = {"thinking_token_budget": THINK["budget"] if budget is None else budget} if THINK["on"] else NO_THINK
    for attempt in range(4):
        try:
            return await client.chat.completions.create(model=model, messages=messages, extra_body=extra, **kw)
        except Exception as e:  # noqa: BLE001  server restarts / timeouts: back off and retry
            if attempt == 3:
                raise
            await asyncio.sleep(5 * (attempt + 1))
            last = e  # noqa: F841


META = re.compile(r"\bthe user\b|\bin character\b|\bmoth (should|would|is asked)\b|\blet me (think|check|re)|"
                  r"\bno markdown\b|\bkeep it (short|simple)\b|\b\d+ words\b|\blet me try\b|\bmisunderstand", re.I)
# agent chats: build_sft's TECHY (copied, it imports torch) + over 25 words + the "i'm just a moth / can't understand" dodge
AGENT_BAD = re.compile(r"\b[a-z]\w*_\w+\b|\w\(\)|\b\w+\.(?:py|js|ts|tsx|json|csv|md|ya?ml|toml|sh|rs|go|html|css|sql|cfg|ini)\b|"
                       r"\b(?:ruff|mypy|pytest|npm|pip|git|github|docker|eslint|webpack|cargo|kubectl|ci|lint|linter|"
                       r"endpoints?|commits?|regex|api|json|csv|yaml|http|https|redis|schema|parser|tuple|repo)\b|"
                       r"\btests? (?:all )?(?:pass|passed|passes|passing)\b|\b(?:all )?\d+ tests\b|^(?:\S+\s+){25}\S|"
                       r"\bi(?:['’]?m| am) (?:just |only )?a (?:little |tiny |small )?moth\b|"
                       r"\bcan(?:['’]?t|not) (?:understand|read|follow)\b|\bdon['’]?t (?:understand|get it)\b")
BUG = re.compile(r"\bbugs?\b|\bgrand ?ma\b", re.I)
MOTIF = re.compile(r"\bfans?\b", re.I)


def grams(t: str, n=5) -> set:
    w = re.sub(r"[^\w\s']", " ", t.lower()).split()
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


EXAMPLE_GRAMS = set().union(*map(grams, re.findall(r'-> "([^"]+)"', AGENT_RULES) +
                                 re.findall(r"^you: (.+)$", TEACHER_BIBLE, re.M) +
                                 ["ohhh it fixed a bug! i'm a bug too. i hope it was nice to it."]))   # r3 bible line


STOP = set("the and for that this with you your are was were has have had not but all can its into from what when then "
           "than they them there their our out just one two too very like little tiny she her his who how why yes now "
           "new agent said".split())
VIBES = [(r"\bsorry|\bapolog", lambda m: re.search(r"sorry|apolog", m)), (r"\bask|\bquestion", lambda m: "?" in m),
         (r"\bwords\b|\blong\b|\bbusy\b|\btired\b", lambda m: len(m.split()) > 80),
         (r"\bproud\b|\bexcited\b|\bhappy\b|\bpolite\b", lambda m: True)]   # vibe it really has, not a made-up "sorry"


def stems(t: str) -> set:
    return {re.sub(r"(es|s|ed|ing)$", "", w) for w in re.findall(r"[a-z]{3,}", t.lower()) if w not in STOP}


def agent_stale(text: str, msgs: list[dict]) -> bool:
    """r3 agent chats: LFM copied the few-shot lines, repeated itself and said "bug" to 1 in 3 agent messages that never
    mentioned one (the grandma lore). (r3 also demanded a word of the message in the reply: that made the word-grab
    template, so gist-first reactions no longer need one.)"""
    mine = set().union(*(grams(m["content"]) for m in msgs if m["role"] == "assistant" and m.get("content")))
    if grams(text) & (EXAMPLE_GRAMS | mine):
        return True
    if MOTIF.search(text) and any(MOTIF.search(m["content"] or "") for m in msgs if m["role"] == "assistant"):
        return True   # build_sft drops chats with the fan twice
    last = msgs[-1]["content"]
    if not last.startswith(AGENT_TAG):
        return False
    return bool(BUG.search(text) and not re.search(r"\bbugs?\b", last, re.I))


def id_allowed(u: str, card: dict | None) -> bool:
    """The friend asked who/what it is: a short identity answer fits. (A card word in the friend's line only makes
    that word "prompted" for identity_ref; it does not invite identity talk.)"""
    return bool(ASKED.search(u))


def lang_remind(card: dict | None) -> str:
    lang = lang_of(card)
    return "" if lang == "en" else f"\n(reply in {LANG_NAMES[lang]} only.)"


def teacher_view(u: str, remind: str | None = None, extra: str = "", card: dict | None = None) -> str:
    if remind:   # agent chats: their own reminder, always
        return u + remind + lang_remind(card)
    return u + (TALK_REMIND_ID if id_allowed(u, card) else TALK_REMIND) + extra + lang_remind(card)


def reply_text(r, cut_ok=False) -> str | None:
    """The reply, or None when the call broke: cut off by max_tokens, or reasoning/tool markup leaked into the text."""
    c = r.choices[0]
    t = (c.message.content or "").strip()
    if (c.finish_reason == "length" and not cut_ok) or any(x in t for x in LEAKS):
        return None
    return t


def last_moth(msgs: list[dict]) -> str:
    return next((m["content"] for m in reversed(msgs) if m["role"] == "assistant" and m.get("content")), "")


def confused(last: str, mine: list[str]) -> str:
    """A "what X?" turn about a word moth really said (or a plain "what?"), built in code."""
    toks, words = [w.strip("'") for w in re.findall(r"[a-z']+", last.lower())], []
    for j, w in enumerate(toks):   # nouns, roughly: the first real word after "the/a/my/..." (skipping adjectives)
        if w in DET:
            x = next((v for v in toks[j + 1:j + 4] if v not in ADJ), "")
            if len(x) >= 3 and x not in WORD_STOP and x not in DET:
                words.append(x)
    for _ in range(4):
        t = random.choice(WHAT_X).format(x=random.choice(words)) if words and random.random() < 0.75 \
            else random.choice(WHAT_GENERIC)
        if t not in mine:
            return t
    return t


def sim_prompt(persona, scenario, turns, i, msgs, topic="anything", card=None) -> str:
    card = card if card is not None else BLANK_CARD   # never moth: moth is held-out (the app's default card only)
    name = display_name(card)
    return USER_SIM_PROMPT.format(name=name, who=describe(card),
                                  persona=persona, scenario=scenario, topic=topic, turns=turns, i=i,
                                  transcript=transcript(msgs, name), language=sim_language(card))


def sim_language(card: dict | None) -> str:
    lang = lang_of(card)
    if lang == "en":
        return "Write in English."
    return (f"Write in {LANG_NAMES[lang]} only, the natural casual way native speakers type it in a chat app (your "
            "persona's age and habits still show). Never switch to English, even though these instructions are English.")


async def user_turn(client, model, persona, scenario, turns, i, msgs, topic="anything", directive="", card=None,
                    system=None) -> str | None:
    p = sim_prompt(persona, scenario, turns, i, msgs, topic, card)
    mine = [m["content"] for m in msgs if m["role"] == "user" and not m["content"].startswith(AGENT_TAG)]
    last = last_moth(msgs)
    short = not directive and msgs and random.random() < 0.4   # without reasoning LFM ignores "one in four very short"
    en = lang_of(card) == "en"   # the built-in "what X?" turns are English
    if short and last and en and random.random() < 0.3:
        return confused(last, mine)
    if mine:   # r3: lazy short lines ("ok and?") came back word for word
        p += SIM_NO_REPEAT.format(prev=" / ".join(f'"{x[:100]}"' for x in mine[-8:]))
    p += directive
    if THINK["on"]:   # LFM2.5: under the character sheet it plays moth (even emits tool calls); its prefix cache
        # works in 16-token blocks, so sharing the sheet buys nothing. No thinking: capped reasoning leaks into the message
        if short:
            p += SIM_SHORT.format(mood=random.choice(SHORT_MOODS))
        r = await chat(client, model, [{"role": "system", "content": SIM_SYSTEM + " " + sim_language(card)},
                                       {"role": "user", "content": p}],
                       budget=0, temperature=1.0, top_p=0.95, max_tokens=100)
    else:   # same system prompt as moth's calls: Qwen3.x's hybrid prefix cache reuses only whole ~1.5k-token blocks,
        # so both roles share one identical leading block (the character sheet) and only the tail is prefilled
        r = await chat(client, model, [{"role": "system", "content": system or TEACHER_BIBLE},
                                       {"role": "user", "content": SIM_SWITCH + p}],
                       temperature=1.0, top_p=0.95, max_tokens=80)
    t = reply_text(r, cut_ok=True)
    if not t or "[END]" in t:
        return None
    lines = [x.strip() for x in t.split("\n") if x.strip()]
    if len(lines) > 1 and lines[0].endswith(":"):   # "Sure! Here's the human's next message:"
        lines = lines[1:]
    t = lines[0].strip('"\u201c\u201d').strip()
    t = re.sub(r"^(you|me|user|friend|human)\s*:\s*", "", t, flags=re.I).strip()
    if not t or re.match(r"moth\s*:", t, re.I) or (card and card.get("name") and t.lower().startswith(card["name"] + ":")):
        return None
    w = WHATX_RE.search(t) if len(t.split()) <= 5 and last and en else None
    if w and w.group(1).lower() not in WORD_STOP | {"do", "does", "is", "are", "you", "u", "mean", "happened", "now", "else",
                                                   "about", "it", "that", "this", "then", "not", "tho", "so"} \
            and stems(w.group(1)).isdisjoint(stems(last)) and w.group(1).lower() not in last.lower():
        return confused(last, mine)   # asked about a word moth never said
    return t[:300]


def turn_context(msgs: list[dict]) -> str:
    """The last human/agent message: words from there are "prompted"."""
    return next(m["content"] for m in reversed(msgs) if m["role"] == "user")


async def moth_turn(client, model, oai: list[dict], msgs: list[dict], stats, talk_only=False, must=None,
                    meta=None, card=None) -> bool | str:
    """Extend oai (teacher view) and msgs (canonical) with the buddy's reply to the last user message (no tools: a
    reply with tool-call or reasoning markup counts as broken and is resampled)."""
    avoid: list[str] = []   # rejected lines, shown back to the teacher (else it resamples the same one)
    for k in range(8 if talk_only else 5):
        view = oai if not avoid else oai[:-1] + [{**oai[-1], "content": oai[-1]["content"] + "\n(not these, they are "
                                                  "stale, off the point, about yourself or too techy: "
                                                  + " / ".join(avoid[-3:]) + ")"}]
        r = await chat(client, model, view, temperature=1.0 if avoid else 0.85, top_p=0.95,
                       max_tokens=THINK["budget"] + 300 if THINK["on"] else 400)
        if r.choices[0].finish_reason == "length" or r.choices[0].message.tool_calls:
            stats["broken_resample"] += 1
            continue
        text = re.sub(r"^moth\s*:\s*", "", reply_text(r) or "", flags=re.I)
        if not text or META.search(text):   # empty, cut off, or the teacher's reasoning spilled into the reply
            continue
        if must and k < 4 and stems(text).isdisjoint(must):
            stats["recall_resample"] += 1   # recall turn: the answer must use a word of the fact the friend shared
            continue
        why = sft_problem(meta, msgs, text) if meta else None   # identity, motif cap, explains, bug, style...
        if why and (talk_only or k < 4):
            stats[f"sft_{why}"] += 1
            avoid.append(text)
            continue
        if talk_only and AGENT_BAD.search(text.lower()):
            stats["agent_resample"] += 1   # dodged or explained the code (build_sft would drop the chat): try again
            avoid.append(text)
            continue
        if talk_only and agent_stale(text, msgs):
            stats["agent_stale"] += 1
            avoid.append(text)
            continue
        ctx = turn_context(msgs)
        if card is not None and k < 4 and not id_allowed(ctx, card) and identity_ref(text, card, ctx):
            stats["id_resample"] += 1   # brought up what it is / where it lives unprompted: identity stays rare
            avoid.append(text)
            continue
        oai.append({"role": "assistant", "content": text})
        msgs.append({"role": "assistant", "think": None, "content": text, "tool_calls": []})
        if why:   # last try still bad: build_sft cuts the chat here, so stop writing it
            stats[f"sft_stop_{why}"] += 1
            return "stop"
        return True
    return False


AGENT_POOL: list[dict] = []   # real agent messages from public traces (moth/agent_pool.json)
AGENT_SHARE = {"p": 0.25}     # --agent-share: fraction of chats where moth watches a big agent's messages
MID_RUN = re.compile(r":\s*\Z|\b(?:Now )?[Ll]et me (?!know)|\bPhase \d|\bI'll now\b|\bLet's\b|^(?:Now|Next),? |"
                     r"^I (?:see|notice)\b|^(?:Excellent|Perfect|Great)! Now\b", re.M)


async def agent_message(client, model, rng) -> tuple[str, str]:
    """(raw message, source): a real one from the pool most of the time, else one written by the teacher."""
    if AGENT_POOL and rng.random() < 0.75:
        m = rng.choice(AGENT_POOL)
        return m["text"], m["source"]
    p = AGENT_PROMPT.format(style=rng.choice(HARNESS_STYLES), task=rng.choice(AGENT_TASKS), outcome=rng.choice(OUTCOMES),
                            length=rng.choice(AGENT_LENGTHS))
    r = await chat(client, model, [{"role": "user", "content": p}], temperature=0.9, top_p=0.95,
                   max_tokens=THINK["budget"] + 900 if THINK["on"] else 700)
    return reply_text(r) or "", "teacher"


async def agent_dialog(client, model, seed, rng, stats, card=None) -> dict | None:
    """The buddy watches a big coding agent: agent message -> its gist-first take -> the human vents or chats -> repeat."""
    persona, scenario = rng.choice(AGENT_USER_PERSONAS), rng.choice(AGENT_SCENARIOS)
    msgs: list[dict] = []
    system = teacher_system(card)
    oai = [{"role": "system", "content": system}]
    sources = []

    async def say(u: str, remind: str) -> bool:
        msgs.append({"role": "user", "content": u})
        oai.append({"role": "user", "content": teacher_view(u, remind, card=card)})
        if await moth_turn(client, model, oai, msgs, stats, talk_only=True,
                           meta={"type": "agent", "topic": "", "card": card}, card=card):
            return True
        msgs.pop()   # no fresh line after all retries: end the chat before this message
        return False

    ok = True
    for k in range(rng.randint(2, 4)):
        if k == 0 and rng.random() < 0.4:   # sometimes the human says something first
            u = await user_turn(client, model, persona, scenario, 6, 1, msgs, "the AI agent", card=card, system=system)
            if u and not (ok := await say(u, AGENT_CHAT_REMIND)):
                break
        raw, src = await agent_message(client, model, rng)
        if not raw:
            break
        if not (ok := await say(agent_turn(raw), AGENT_REMIND)):
            break
        sources.append(src)
        for j in range(rng.choice([0, 1, 1, 2, 3])):
            u = await user_turn(client, model, persona, scenario, 6, j + 2, msgs, "the AI agent", card=card,
                                system=system)
            if u is None or not (ok := await say(u, AGENT_CHAT_REMIND)):
                break
        if not ok:
            break
    if not sources:   # moth never reacted to an agent message
        stats["bad"] += 1
        return None
    stats["agent_cut"] += not ok
    return {"seed": seed, "type": "agent", "persona": persona, "scenario": scenario, "agent_sources": sources,
            "turns": len(msgs), "card": card, "lang": lang_of(card), "messages": msgs}


async def one_dialog(client, model, seed, stats) -> dict | None:
    rng = random.Random(seed)
    card = sample_card(rng)
    if is_moth(card):   # moth is held-out: no training chat may ever use it (cards.is_moth)
        raise ValueError(f"seed {seed}: moth card sampled for a training chat")
    if not SCENARIO_IDS and rng.random() < AGENT_SHARE["p"]:
        return await agent_dialog(client, model, seed, rng, stats, card)
    scen = [SCENARIOS[i] for i in SCENARIO_IDS] if SCENARIO_IDS else SCENARIOS
    persona, scenario, turns = rng.choice(USER_PERSONAS), rng.choice(scen), rng.choice(TURN_CHOICES or TURNS)
    topic = rng.choice(TOPICS)
    if not TURN_CHOICES and rng.random() < LONG_SHARE["p"]:
        turns = rng.randint(16, 24)   # r3 had no chat over 14 turns
    recall = None
    if rng.random() < RECALL_SHARE["p"]:   # r3: ~50 of 21k chats asked for something back
        fact, short = rng.choice(RECALL_FACTS)
        fi = rng.choice([1, 2])
        recall = {"fact": short, "share_turn": fi, "ask_turn": fi + rng.randint(5, 8)}
        turns = max(turns, recall["ask_turn"] + rng.randint(1, 3))
    msgs: list[dict] = []
    system = teacher_system(card)
    oai = [{"role": "system", "content": system}]
    for i in range(1, turns + 1):
        directive, extra = "", QBACK if rng.random() < 0.2 else NO_QBACK
        if recall and i == recall["share_turn"]:
            directive = f"\n\nIn THIS message, casually tell {display_name(card)} {fact}."
        elif recall and i == recall["ask_turn"]:
            directive = (f"\n\nIn THIS message, ask {display_name(card)} if it remembers {short} (you told it "
                         "earlier). Ask in your own words and don't give the answer.")
            extra = RECALL_REMIND
        if (recall and i < recall["ask_turn"]) or (turns >= 16 and i < turns - 3):
            directive += "\n\nThe chat is not finished: do not write [END] yet."
        u = await user_turn(client, model, persona, scenario, turns, i, msgs, topic, directive, card, system)
        for _ in range(3 if "[END]" in directive else 0):   # long / recall chat: broken sim calls (leaked
            if u is not None:                                 # reasoning) ended 13 of 20 long chats early
                break
            u = await user_turn(client, model, persona, scenario, turns, i, msgs, topic, directive, card, system)
        if u is None:
            break
        if recall and i == recall["share_turn"]:
            recall["said"] = u
        must = None
        if recall and i == recall["ask_turn"] and recall.get("said"):
            recall["asked"] = u
            must = stems(recall["said"]) - FACT_WORDS or None
        msgs.append({"role": "user", "content": u})
        oai.append({"role": "user", "content": teacher_view(u, extra=extra, card=card)})
        r = await moth_turn(client, model, oai, msgs, stats, must=must, card=card,
                            meta={"type": "talk", "topic": topic, "card": card})
        if not r:
            stats["bad"] += 1
            return None
        if r == "stop":
            break
    if not any(m["role"] == "assistant" for m in msgs):
        return None
    return {"seed": seed, "type": "talk", "persona": persona, "scenario": scenario, "topic": topic, "turns": turns,
            "recall": recall, "card": card, "lang": lang_of(card), "messages": msgs}


async def run_dialogs(a, client, model):
    path = os.path.join(a.out, "dialogs.jsonl")
    done = set()
    if os.path.exists(path):
        for line in open(path):
            try:
                done.add(json.loads(line)["seed"])
            except (ValueError, KeyError):
                pass
    todo = [s for s in range(a.start, a.start + a.dialogs) if s not in done]
    stats = collections.Counter(ok=0, bad=0, err=0, broken_resample=0, agent_resample=0, agent_stale=0, agent_cut=0,
                                recall_resample=0, id_resample=0)
    sem = asyncio.Semaphore(a.concurrency)
    f = open(path, "a")
    t0 = time.time()

    async def go(seed):
        async with sem:
            try:
                d = await one_dialog(client, model, seed, stats)
            except Exception as e:  # noqa: BLE001
                stats["err"] += 1
                print(f"seed {seed}: {type(e).__name__}: {e}", flush=True)
                return
            if d:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
                f.flush()
                stats["ok"] += 1
                if stats["ok"] % 100 == 0:
                    el = time.time() - t0
                    print(f"{stats} {stats['ok'] / el * 60:.0f} dialogs/min, {len(todo) - stats['ok']} left", flush=True)

    await asyncio.gather(*(go(s) for s in todo))
    print(f"done {stats} in {(time.time() - t0) / 60:.1f} min", flush=True)


SAMPLE_OPENERS = ["hey, i burned my toast this morning", "ok and?", "hiii what are you up to", "i'm so tired today",
                  "my cat keeps sitting on my keyboard", "Good evening. How was your day?", "what are you?",
                  "can you remember my name? it's lena"]
SAMPLE_AGENT = ("I've updated the build config and restarted the staging box. It's still booting; I'll check the health "
                "endpoint again in a minute. Want me to run the smoke tests once it's up?")


def first_teacher_view(seed: int) -> dict:
    """What the teacher sees at the first buddy turn of chat `seed`, offline (tests/render_prompts.py): the same card,
    chat type and reminder logic as one_dialog / agent_dialog; the human's line is a stand-in for the simulator's."""
    rng = random.Random(seed)
    card = sample_card(rng)
    if rng.random() < AGENT_SHARE["p"]:
        u = agent_turn(SAMPLE_AGENT)
        return {"type": "agent", "card": card, "system": teacher_system(card), "user": teacher_view(u, AGENT_REMIND, card=card)}
    persona, scenario = rng.choice(USER_PERSONAS), rng.choice(SCENARIOS)
    u = SAMPLE_OPENERS[seed % len(SAMPLE_OPENERS)]
    return {"type": "talk", "card": card, "system": teacher_system(card),
            "user": teacher_view(u, extra=NO_QBACK, card=card),
            "sim": SIM_SWITCH + sim_prompt(persona, scenario, 6, 1, [], rng.choice(TOPICS), card)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default="out")
    ap.add_argument("--dialogs", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=96)
    ap.add_argument("--agent-share", type=float, default=0.25, help="fraction of chats watching a big agent")
    ap.add_argument("--agent-pool", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_pool.json"))
    ap.add_argument("--eval-agent-msgs", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                               "data", "eval_agent_msgs.json"),
                    help="held-out eval agent messages: dropped from the agent pool")
    ap.add_argument("--recall-share", type=float, default=0.15, help="talk chats: share a fact early, ask 5+ turns later")
    ap.add_argument("--long-share", type=float, default=0.15, help="talk chats with 16-24 user turns")
    ap.add_argument("--think", action="store_true", help="teacher always reasons first (LFM2.5): cap it, split it off")
    ap.add_argument("--think-budget", type=int, default=256, help="--think: max reasoning tokens per moth/agent call")
    ap.add_argument("--turns", default="", help="comma-separated user-turn counts to sample from")
    ap.add_argument("--scenario-ids", default="", help="comma-separated indexes into persona.SCENARIOS")
    a = ap.parse_args()
    SCENARIO_IDS.extend(int(x) for x in a.scenario_ids.split(",") if x)
    TURN_CHOICES.extend(int(x) for x in a.turns.split(",") if x)
    THINK["on"], THINK["budget"] = a.think, a.think_budget
    RECALL_SHARE["p"], LONG_SHARE["p"] = a.recall_share, a.long_share
    AGENT_SHARE["p"] = a.agent_share
    if os.path.exists(a.agent_pool):   # final messages only: drop mid-run narration ("Now let me...", "Phase 2", "...:")
        AGENT_POOL.extend(m for m in json.load(open(a.agent_pool)) if not MID_RUN.search(m["text"]))
        n = len(AGENT_POOL)
        if os.path.exists(a.eval_agent_msgs):   # held-out eval messages never go into training chats
            norm = lambda t: " ".join(str(t).split())   # noqa: E731
            ev = [m["text"] if isinstance(m, dict) else m for m in json.load(open(a.eval_agent_msgs))]
            exact, heads = {norm(t) for t in ev}, {norm(t)[:200] for t in ev}
            AGENT_POOL[:] = [m for m in AGENT_POOL if norm(m["text"]) not in exact and norm(m["text"])[:200] not in heads]
            print(f"agent pool: removed {n - len(AGENT_POOL)} eval messages ({len(ev)} held out)", flush=True)
        else:
            print(f"agent pool: WARNING no eval file at {a.eval_agent_msgs}, nothing held out", flush=True)
        print(f"agent pool: {len(AGENT_POOL)} final messages", flush=True)
    os.makedirs(a.out, exist_ok=True)
    client = AsyncOpenAI(base_url=a.base_url, api_key="x", timeout=300)

    async def run():
        model = a.model or (await client.models.list()).data[0].id
        if a.dialogs:
            await run_dialogs(a, client, model)
    asyncio.run(run())


if __name__ == "__main__":
    main()
