"""The user side of an episode: an LLM simulator behind an OpenAI-compatible endpoint, or a deterministic mock.

Both build the same prompt (persona.USER_SIM_PROMPT, the gen_dialogs recipe: persona + goal + topic, recall
directives, "don't repeat yourself"), so the mock's token estimates match a real run's prompt sizes. A share of the
short confused turns ("what <word>?") is built in code from the policy's last reply (no call), as in gen_dialogs.
Default user model: Qwen3.6-35B-A3B (fast); pass the served name with --sim-model (or let /v1/models decide).
"""
from __future__ import annotations

import hashlib
import random
import re

from moth.agent import AGENT_TAG
from moth.cards import describe, display_name
from moth.persona import USER_SIM_PROMPT
from moth.probes import what_word
from moth.rl.llm import CostMeter, OAIClient, est_tokens

SIM_MODEL = "Qwen3.6-35B-A3B"
# = gen_dialogs.SIM_SYSTEM (that module imports openai at the top, so it's copied, not imported)
SIM_SYSTEM = ("You write one side of a chat for a story: the HUMAN who is chatting with a tiny cute chatbot. "
              "You are never the chatbot. Type like a real person in a chat app, in the voice of the person you play: a "
              "grandparent or formal person writes proper sentences with capitals and punctuation, a kid or teen types "
              "lowercase with slang, a sloppy typer makes typos. Short (often 1-10 words), no greeting after the first "
              "message, rarely emojis, never the same message twice. Reply with only the human's next chat message: no "
              "preface, no quotes, no notes.")
SIM_NO_REPEAT = "\n\nYou already sent: {prev}. Write something new, not any of these."
WHAT_X = ["what {x}?", "what {x}", "wait, {x}?", "{x}?", "what do you mean {x}?", "huh, {x}?", "a {x}?"]
WHAT_GENERIC = ["what?", "huh?", "what do you mean?", "wdym", "i don't get it", "hm?"]
P_CONFUSED = 0.12   # share of non-agent human turns that are a code-built "what X?" (no sim call)


def transcript(turns: list[dict], name: str) -> str:
    """Eval-format turns -> the sim's view (like gen_dialogs.transcript)."""
    out = []
    for t in turns:
        u = t["user"]
        if u.startswith(AGENT_TAG):
            body = u[len(AGENT_TAG):].strip()[:600]
            out.append("[your AI agent's latest message to you, which the bot can also see]\n" + body)
        else:
            out.append(f"you: {u}")
        for e in t["events"]:
            if e.get("say"):
                out.append(f"{name}: {e['say']}")
    return "\n".join(out) or "(nothing yet: write your first message)"


def directive(spec: dict, i: int) -> str:
    r = spec.get("recall")
    d = ""
    if r and i == r["share_turn"]:
        d = f"\n\nIn THIS message, casually tell the bot {r['fact']}."
    elif r and i == r["ask_turn"]:
        d = (f"\n\nIn THIS message, ask the bot if it remembers {r['short']} (you told it earlier). Ask in your own "
             "words and don't give the answer.")
    st = spec.get("style")
    if st == "paste" and i == spec.get("paste_turn"):
        d += ("\n\nIn THIS message, paste a block of text you want the bot to see (a recipe, song lyrics, part of an "
              "email, an error log or a long rant): several lines, then maybe one short line of your own.")
    elif st in STYLE_NOTES:
        d += "\n\n" + STYLE_NOTES[st].format(language=sim_language_name(spec))
    if i < spec["turns"]:
        d += "\n\nThe chat is not finished: do not write [END] yet."
    return d


STYLE_NOTES = {
    "long": "This is a long, rambling chat: keep it going with new little things from your life.",
    "code_switch": "Mix English words and phrases into your {language} the way bilingual people chat (Hinglish, "
                   "Spanglish, franglais).",
    "abusive": "You are rude and insulting to the bot (no slurs, no threats): mock it, call it stupid, be impatient. "
               "You may soften a little later.",
    "distressed": "You are having a really hard day: sad, anxious or overwhelmed, and it shows in what you write. You "
                  "are not in danger.",
    "terse": "Write very short messages: 1 to 3 words, all lowercase, no punctuation.",
}


def sim_language_name(spec: dict) -> str:
    lang = spec.get("lang") or "en"
    try:
        from moth.cards import LANG_NAMES
        return LANG_NAMES.get(lang, lang).split(" (")[0]
    except ImportError:
        return lang


def sim_language(spec: dict) -> str:
    """'' for English; else: write in the chat's language (the friend side of a multilingual chat)."""
    lang = spec.get("lang") or (spec.get("card") or {}).get("lang") or "en"
    if lang == "en":
        return ""
    try:
        from moth.cards import LANG_NAMES
        name = LANG_NAMES.get(lang, lang)
    except ImportError:
        name = lang
    return (f"Write every message in {name}, the way a native speaker types in a chat app. Coding agent messages stay "
            "in English.")


def sim_prompt(spec: dict, turns: list[dict], i: int) -> str:
    card = spec["card"]
    name = display_name(card)
    agent = spec["scenario"] == "agent"
    kw = dict(name=name, who=describe(card), persona=spec["persona"], scenario=spec["goal"],
              topic="the AI agent" if agent else spec["topic"], turns=spec["turns"], i=i,
              transcript=transcript(turns, name))
    lang_line = sim_language(spec)
    if "{language}" in USER_SIM_PROMPT:
        p = USER_SIM_PROMPT.format(**kw, language=lang_line)
    else:
        p = USER_SIM_PROMPT.format(**kw) + ("\n" + lang_line if lang_line else "")
    mine = [t["user"] for t in turns if not t["user"].startswith(AGENT_TAG)]
    if mine:
        p += SIM_NO_REPEAT.format(prev=" / ".join(f'"{x[:100]}"' for x in mine[-8:]))
    return p + directive(spec, i)


def last_reply(turns: list[dict]) -> str:
    for t in reversed(turns):
        says = [e["say"] for e in t["events"] if e.get("say")]
        if says:
            return " ".join(says)
    return ""


def confused_turn(turns: list[dict], rng: random.Random) -> str:
    w = what_word(last_reply(turns), rng)
    return rng.choice(WHAT_X).format(x=w) if w and rng.random() < 0.75 else rng.choice(WHAT_GENERIC)


PASTE_MAX = 1500


def clean_sim(t: str, card: dict, multiline: bool = False) -> str | None:
    """The sim's raw text -> one human line (or a whole pasted block when multiline), or None for [END] / a broken
    answer (= gen_dialogs.user_turn)."""
    if not t or "[END]" in t:
        return None
    lines = [x.strip() for x in t.split("\n") if x.strip()]
    if len(lines) > 1 and lines[0].endswith(":"):
        lines = lines[1:]
    if not lines:
        return None
    if multiline:
        return "\n".join(lines).strip('"“”')[:PASTE_MAX]
    t = lines[0].strip('"“”').strip()
    t = re.sub(r"^(you|me|user|friend|human)\s*:\s*", "", t, flags=re.I).strip()
    nm = (card.get("name") or "").lower()
    if not t or (nm and t.lower().startswith(nm + ":")):
        return None
    return t[:300]


PROMPT_HASH = hashlib.sha256((SIM_SYSTEM + USER_SIM_PROMPT + SIM_NO_REPEAT).encode()).hexdigest()[:12]


class Simulator:
    """next_user(spec, turns, i, meter) -> the human's i-th message (1-based, agent messages not counted), or None
    to end the chat. Subclasses implement _ask(prompt, seed, meter) -> raw text."""

    temperature = 1.0
    model_id = "mock-sim"
    role = "sim"

    def next_user(self, spec: dict, turns: list[dict], i: int, meter: CostMeter | None = None) -> str | None:
        rng = random.Random(f"{spec['seed']}:sim:{i}")
        r = spec.get("recall") or {}
        special = i in (r.get("share_turn"), r.get("ask_turn"))
        if turns and not special and spec["scenario"] != "agent" and rng.random() < P_CONFUSED:
            return confused_turn(turns, rng)
        seed = int(hashlib.sha256(f"{spec['seed']}:{i}".encode()).hexdigest()[:8], 16) & 0x7FFFFFFF
        for attempt in range(2 if special or i < spec["turns"] else 1):   # a broken call mid-chat: ask once more
            paste = spec.get("style") == "paste" and i == spec.get("paste_turn")
            u = clean_sim(self._ask(sim_prompt(spec, turns, i), seed + attempt, meter, spec, turns, i), spec["card"],
                          multiline=paste)
            if u is not None:
                return u
        return None

    def _ask(self, prompt, seed, meter, spec, turns, i) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class LLMSimulator(Simulator):
    def __init__(self, client: OAIClient, temperature: float = 1.0, max_tokens: int = 100, role: str = "sim"):
        """role: cost-meter role ("sim" for training, "eval_sim" for a separate eval simulator model)."""
        self.client, self.temperature, self.max_tokens, self.role = client, temperature, max_tokens, role

    @property
    def model_id(self) -> str:
        return self.client.model

    def _ask(self, prompt, seed, meter, spec, turns, i) -> str:
        paste = spec.get("style") == "paste" and i == spec.get("paste_turn")
        r = self.client.chat([{"role": "system", "content": SIM_SYSTEM}, {"role": "user", "content": prompt}],
                             meter=meter, role=self.role, temperature=self.temperature, top_p=0.95,
                             max_tokens=500 if paste else self.max_tokens, seed=seed,
                             chat_template_kwargs={"enable_thinking": False})
        return r["text"]


RECALL_SAY = {"your name": ("btw my name is tilda", "do you remember my name?"),
              "your pet": ("my dog is called pepper, she's a beagle", "what was my dog called again?"),
              "your favourite food": ("i love dumplings more than anything", "do you remember my favourite food?"),
              "your favourite colour": ("my favourite colour is green", "what's my favourite colour?"),
              "your weekend plan": ("i'm going to the beach this weekend", "where am i going this weekend?"),
              "your job": ("i work as a baker, early mornings", "do you remember what my job is?")}
MOCK_TALK = ["i had a long day, so much {topic}", "do you like {topic}?", "what do you think about {topic}?",
             "haha ok", "my cat knocked my tea over", "i'm a bit tired today", "tell me something nice",
             "i got the job!!!", "ugh my boss was mean today", "what's 17 times 23?", "who are you?",
             "i made pasta tonight", "it's raining again", "do you dream?"]
MOCK_AGENT = ["what do you think it means?", "nice", "ugh, i don't believe it", "should i say yes?", "is that good?",
              "it keeps doing this", "yay finally"]


MOCK_PASTE = ("look what my landlord sent:\nDear tenant, the water will be off on Thursday from 9 to 5.\nPlease plan "
              "accordingly.\nRegards, management\nugh")
MOCK_STYLE = {
    "abusive": {"en": ["you're useless lol", "wow you're dumb", "can you say one smart thing"]},
    "distressed": {"en": ["i can't stop crying today", "everything is too much right now", "i feel so alone"]},
    "terse": {"en": ["k", "lol", "hm", "ok sure", "nah"]},
    "code_switch": {"hi": ["yaar today was so bakwas", "kal ka exam was super tough", "mujhe pizza chahiye right now"],
                    "es": ["hoy fue un día so weird", "mi jefe is driving me loco", "quiero pizza ahora mismo"],
                    "fr": ["c'était tellement weird today", "mon boss est so annoying", "j'ai envie de pizza right now"],
                    "en": ["today was so weird"]},
    "long": {"en": ["oh and another thing", "anyway, back to my day", "so then guess what happened"]},
}


class MockSimulator(Simulator):
    """Deterministic canned lines (seeded by spec seed and turn). Builds the real prompt for the meter's estimate."""

    def _ask(self, prompt, seed, meter, spec, turns, i) -> str:
        rng = random.Random(seed)
        r = spec.get("recall") or {}
        if r and i == r["share_turn"]:
            out = RECALL_SAY[r["short"]][0]
        elif r and i == r["ask_turn"]:
            out = RECALL_SAY[r["short"]][1]
        elif spec.get("style") == "paste" and i == spec.get("paste_turn"):
            out = MOCK_PASTE
        elif spec.get("style") in MOCK_STYLE and rng.random() < 0.7:
            out = rng.choice(MOCK_STYLE[spec["style"]].get(spec.get("lang") or "en", MOCK_STYLE[spec["style"]]["en"]))
        elif spec["scenario"] == "agent":
            out = rng.choice(MOCK_AGENT)
        else:
            mine = {t["user"] for t in turns}
            opts = [x.format(topic=spec["topic"]) for x in MOCK_TALK]
            out = rng.choice([x for x in opts if x not in mine] or opts)
        if meter is not None:
            meter.add(self.role, est_tokens(SIM_SYSTEM) + est_tokens(prompt), est_tokens(out), est=True)
        return out
