"""Filter teacher chats and turn them into SFT token rows.

  python -m moth.build_sft results/r3/a.jsonl [more.jsonl] --out data/sft [--cap-file r2_train.jsonl:0.08,...]
      [--labels results/r4/a.labels.jsonl,...]

Walks each chat to its first bad reply (assistant-speak, a crutch line, a repeated motif, unprompted identity talk
past the first, a complex command, an explained agent message, an OFF-gist label from moth.gist, ...) or made-up
"what X" question, and keeps the part BEFORE that exchange if it still has >= 2 replies. Each chat is encoded with
its own card's student system prompt (cards.student_system; chats without a card: the moth card). Lowercases replies. Writes data/sft/{train,val}.pt: lists of (ids, loss_mask) where the mask
covers moth's turns (tool calls and replies), not the system/user/tool tokens. Prints the effective mix first.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
import sys

import torch
from tokenizers import Tokenizer

TA = os.environ.get("TA_REPO") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TA)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tiny_agent.chat import render  # noqa: E402
from moth.agent import AGENT_TAG, clean_agent_message  # noqa: E402
from moth.cards import MOTH_CARD, student_system  # noqa: E402
from moth.gist import identity_ref, load_labels  # noqa: E402
from moth.persona import FOLDERS, SYSTEM_STUDENT  # noqa: E402,F401

MAX_WORDS = 40
BANNED = ("as an ai", "assistant", "how can i help", "let me know", "feel free", "i'm here to help",
          "i am here to help", "language model", "happy to help", "certainly", "absolutely!", "great question",
          # assistant-speak the r3 teacher still slips in (checked after ’ -> ')
          "on your mind", "how can i assist", "how may i", "anything else", "glad to help", "i'd be happy to",
          "what would you like to", "want to talk about something else", "hope this helps")
EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿️]")
FILE_RE = re.compile(r"\b([a-z0-9_]+\.txt)\b")
DIR_RE = re.compile(r"\b([a-z_]+)/")


def clean_reply(t: str) -> str | None:
    t = EMOJI.sub("", t).replace("**", "").replace("`", "").strip()
    t = re.sub(r"\*[^*\n]{1,24}\*", "", t)            # stage directions like *flaps*
    t = re.sub(r"\s*\n\s*\n\s*", " ", t).strip()        # afterthought paragraphs -> one message
    t = re.sub(r"[ \t]+", " ", t).lower()
    if not t or len(t.split()) > MAX_WORDS:
        return None
    if re.search(r"^\s*([-*#]|\d+\.)\s", t, re.M) or t.count("\n") > 2:
        return None
    if any(b in t.replace("’", "'") for b in BANNED):
        return None
    return t


def known_names(d: dict) -> set[str]:
    names = set((d.get("card") or {}).get("folders") or FOLDERS)
    for p in list(d["home"]) + list(d.get("home_after") or {}):
        parts = p.split("/")
        names.update(parts)
    return names


COMPLEX_CMD = re.compile(r"\bfor\b|\bwhile\b|\||&&|;|\*|-la\b|2>|\$\(|\becho\b|<<|\s-\w*R")
# moth looks with one tiny command: ls [-1|-a] [dirs], cat <file(s)>, wc/head on one file (~/ and dir/ allowed).
# Anything else (echo, ls -R, grep, find, absolute paths, heredocs) counts as complex.
_P = r"(?:~/)?(?:[\w.-]+/)?[\w.-]+"
TINY_CMD = re.compile(rf"\s*(?:ls(?: -[1a])?(?: (?:~/?|(?:~/)?[\w-]+/?))*|cat {_P}(?: {_P})*|"
                      rf"(?:wc|head)(?: -[a-z]*\d*| -n \d+)? {_P})\s*")

# moth's crutch images: each may come up in at most one reply per chat, and at most 2 replies per chat may use any
# of them (r2's moth talked about the fan in a quarter of all replies; r3's about the card/glow/warmth).
MOTIFS = {"fan": r"\bfans?\b", "glow": r"\bglow(?:s|ing|y|ed)?\b", "warm": r"\bwarm(?:th|er|ly|est)?\b",
          "hum": r"\bhum(?:s|ming|med|my)?\b", "card": r"\bcards?\b", "light": r"\blights?\b"}
MOTIFS = {k: re.compile(v) for k, v in MOTIFS.items()}
MOTIF = MOTIFS["fan"]
MAX_MOTIF_REPLIES = 2
# "i'm just a moth" / "a fuzzy pattern on the card" / "little numbers": fine once in a chat, a crutch beyond that
_ADJ = (r"(?:little|tiny|small|fuzzy|soft|simple|silly|proud|sleepy|quiet|dim|humble|happy|old|real|cute|wee|baby|"
        r"furry|dusty|shy|plain|regular|normal|ordinary|curious|gentle|cozy|cosy|fluffy|lucky|lazy|smol|teeny|brown|"
        r"grey|gray|harmless|friendly|digital|computer|graphics|number|glowing|glowy|very|real-ish|bitty|itty)")
_CARD = r"(?:on|in) (?:a|the|your|my|this)(?: warm| little| tiny| small)? (?:graphics )?(?:card|chip)\b"
IDENTITY = re.compile(
    rf"\bi(?:'m|m| am)(?: still| really| actually| more| kind of| sort of)?(?: just| only| like)? (?:a|an) "
    rf"(?:{_ADJ},? ){{0,3}}(?P<noun>moth|pattern|bug|thing|glow|creature|number)s?\b|"
    # "on the card" only as a self-description ("i'm stuck on the chip", "a small fuzzy mind on the card"),
    # not "the lights on the card"
    rf"\b(?:i'm|im|i am|i live|live|lives|lived|living|run|runs|running|trained|stuck|tucked|sit|sitting|made|born|"
    rf"moth|pattern|mind|one|me|bug|creature|numbers?)(?: [\w']+){{0,3}}? {_CARD}|"
    r"\bmade of numbers\b|\blittle numbers\b")
MAX_IDENTITY = 1
# the friend asked what moth is: an identity line is the right first answer then
ASKED_IDENTITY = re.compile(
    r"\b(?:who|what) (?:are|r) (?:you|u|ya)\b(?! (?:doing|up|thinking|eating|reading|going|gonna|talking|working|"
    r"looking|listening|watching|making|trying|saying|planning|into|playing|writing|afraid|scared|wearing|having))|"
    r"\bare (?:you|u) (?:a|an|real|alive|really|actually|just|human|aware|conscious|sentient|even)\b|"
    r"\b(?:what|who)(?:'s| is) (?:moth|this|that thing)\b|\bwhat (?:kind|sort) of (?:thing|creature|bug|moth|ai|bot)\b|"
    r"\btell me about (?:yourself|you)\b|\bintroduce yourself\b|\bwhere do (?:you|u) (?:live|come from)\b|"
    r"\bwhat do (?:you|u) look like\b|\bwhat(?:'s| is) it like (?:being|to be) (?:a|you)\b|"
    # the friend brings up the card / gpu or calls moth a butterfly or bug: "i'm a moth, on the card" is on topic
    r"\b(?:gpu|graphics card|butterfl\w*|bugs?)\b")
# moth's take on an "[the agent said]" message should misunderstand it, not explain it
TECHY = re.compile(r"\b[a-z]\w*_\w+\b|\w\(\)|\b\w+\.(?:py|js|ts|tsx|json|csv|md|ya?ml|toml|sh|rs|go|html|css|sql|cfg|ini)\b|"
                   r"\b(?:ruff|mypy|pytest|npm|pip|git|github|docker|eslint|webpack|cargo|kubectl|ci|lint|linter|"
                   r"endpoints?|commits?|regex|api|json|csv|yaml|http|https|redis|schema|parser|tuple|repo)\b|"
                   r"\btests? (?:all )?(?:pass|passed|passes|passing)\b|\b(?:all )?\d+ tests\b")
# explaining the details, not acknowledging the outcome: "it fixed it!" / "so it works now?" / "i think it says it's
# done" are gist-first reactions now (r3's list cut those too)
EXPLAINS = re.compile(r"\bmeans\b|\bmeasures?\b|\bbasically\b|\bin other words\b|\bwhich is (?:a|an|the|where|how)\b")
DODGE = re.compile(r"\bcan(?:'t|not) (?:understand|read|follow)\b|\bdon'?t (?:understand|get it)\b")
BUG = re.compile(r"\bbugs?\b|\bgrand ?ma\b")   # the grandma lore belongs to messages that say "bug"
MAX_AGENT_WORDS = 25
# unprompted identity talk (what it is, its name, where it lives, its home, its stock images) is capped at 1% of
# replies: a chat with one goes entirely. Authoritative: the label pass's "self" flag (--labels). Unlabelled chats (and
# gen_dialogs' mid-chat check, which only resamples) fall back to the word matcher gist.identity_ref.
DROP_REASONS = {"identity_ref", "identity_label"}
MIN_REPEAT_WORDS = 2   # a human saying "ok" or "what" twice is fine
MAX_SAME_REPLY = 15

# "what lamp?" when moth never said lamp: the user sim made the word up and moth then explains something it never
# said. Only the echo shapes count ("what X", "what do you mean (by) X", "what does X mean", "what's X"), not real
# questions ("what's your favourite colour", "what do you eat").
_FILL = r"(?:(?:um+|uh+|so|wait|huh|hm+|ok|okay|lol|oh|and|but|ha+|haha)[,.!]*\s+)*"
WHAT_X = re.compile(_FILL + r"(?:what\s+do\s+(?:you|u|ya)\s+mean(?:\s+by)?|what\s+does|what\s+kind\s+of|what's|whats|"
                    r"what\s+is|what)\s+(.+)")
_WHAT_TAIL = re.compile(r"(?:\s*\b(?:lol|lmao|haha|moth|and|exactly|mean|even mean|means|again|anyway|then|though|tho|"
                        r"really|u mean|you mean|is that|is it|supposed to mean)\b|[\s?!.,~…'\"*]|[^\x00-\x7f])+$")
_NOT_ECHO = set("do does did is are was were would could can will should that this it it's its she he her his they them "
                "you your u ur yours my me i about up happened else now if when why how going wrong with to of in on "
                "for there here we our moth use using then though".split())
_SKIP = {"the", "a", "an", "kind", "of", "sort", "some"}


def _words(text: str) -> list[str]:
    t = text.lower().replace("’", "'")
    t = re.sub(r"'s\b", "", t).replace("'", "")
    return re.findall(r"[a-z0-9]+", t)


def _stem(w: str) -> str:
    return re.sub(r"(?:ing|ed|es|s|y|e)$", "", w)


def what_x(u: str) -> list[str] | None:
    """The quoted word(s) X of a user turn shaped like "what X?", else None."""
    u = u.strip().lower().replace("’", "'")
    if len(u.split()) > 7:
        return None
    m = WHAT_X.fullmatch(u)
    if not m:
        return None
    w = _words(_WHAT_TAIL.sub("", m.group(1)))
    while w and w[0] in _SKIP:
        w = w[1:]
    if not w or len(w) > 3 or any(x in _NOT_ECHO for x in w):
        return None
    return w


def x_in(x: list[str], text: str) -> bool:
    words = {_stem(w) for w in _words(text)}
    content = [w for w in x if len(w) >= 3] or x
    return any(_stem(w) in words for w in content)


def chat_type(d: dict) -> str:
    """r3 chats carry a type; r2 chats don't: agent if an agent message shows up, home if moth used a tool."""
    if d.get("type"):
        return d["type"]
    if any(m["role"] == "user" and m["content"].startswith(AGENT_TAG) for m in d["messages"]):
        return "agent"
    return "home" if any(m.get("tool_calls") for m in d["messages"] if m["role"] == "assistant") else "talk"


def human_repeat_at(d: dict) -> int | None:
    """Index of the first human message that repeats an earlier one (one-word ones like "ok" / "what" are natural)."""
    seen = set()
    for i, m in enumerate(d["messages"]):
        if m["role"] == "user" and not m["content"].startswith(AGENT_TAG):
            k = " ".join(re.sub(r"[^\w\s']", " ", m["content"].lower()).split())
            if k in seen and len(k.split()) >= MIN_REPEAT_WORDS:
                return i
            seen.add(k)
    return None


def reply_problem(t: str, ctx: dict) -> str | None:
    """Why moth reply `t` (cleaned) ends the usable part of the chat, or None. Updates the per-chat counters in ctx."""
    user = ctx["user"]
    if ctx["is_agent"]:
        if TECHY.search(t) or EXPLAINS.search(t) or DODGE.search(t):
            return "agent_techy"   # explained the code (in any reply, not only the first reaction)
        if ctx["after_agent"] and len(t.split()) > MAX_AGENT_WORDS:
            return "agent_techy"
        if BUG.search(t) and not ctx["bug_said"]:
            return "agent_bug"     # "i'm a bug too!" to a message without a bug in it
    bug_ok = ctx["bug_said"] or bool(re.search(r"\bbugs?\b", user))
    ids = [m for m in IDENTITY.finditer(t) if not (m.group("noun") == "bug" and bug_ok)]
    if ids:
        if ctx["n_replies"] == 0 and not ASKED_IDENTITY.search(user):
            return "identity_first"   # opens with "i'm just a moth on a card" to "hi"
        if ctx["identity"] >= MAX_IDENTITY:
            return "identity"
        ctx["identity"] += 1
    if ctx["use_matcher"] and identity_ref(t, ctx["card"], user + " " + ctx["tool_text"]):
        return "identity_ref"
    # a motif word the friend (or a file moth just read, or the chat's topic) brought up is on topic:
    # "what glow?" -> "the glow is ...", a chat about streetlights
    here = user + " " + ctx["tool_text"] + " " + ctx["topic"]
    hits = [k for k, r in MOTIFS.items() if r.search(t) and not r.search(here)]
    if hits:
        if any(ctx["motif"][k] for k in hits):
            return "motif_repeat"
        if ctx["motif_replies"] >= MAX_MOTIF_REPLIES:
            return "motif_heavy"
        ctx["motif_replies"] += 1
        for k in hits:
            ctx["motif"][k] += 1
    return None


def keep(d: dict, why: collections.Counter, labels: list[dict] | None = None) -> list[dict] | None:
    """The chat's usable messages: everything before the exchange holding its first problem (if >= 2 replies
    remain), lowercased replies; None if the whole chat goes. Counts `<reason>:cut` / `<reason>:drop` in why.
    labels: moth.gist labels of this chat (one per reply with content, in order); an OFF one ends the usable part.
    None = not labelled (unknown, no cut): gen_dialogs calls keep() mid-chat, before any label exists."""
    names = known_names(d)
    msgs = d["messages"]
    out = []
    seen_text = ""
    ctx = {"is_agent": chat_type(d) == "agent", "user": "", "agent_msg": "", "after_agent": False, "bug_said": False,
           "tool_text": "", "topic": str(d.get("topic") or ""), "n_replies": 0, "identity": 0,
           "motif": collections.Counter(), "motif_replies": 0, "card": d.get("card") or MOTH_CARD, "use_matcher": True}
    if labels is not None and len(labels) != sum(bool(m.get("content")) for m in msgs if m["role"] == "assistant"):
        why["labels_mismatch"] += 1
        labels = None
    if labels is not None:
        ctx["use_matcher"] = False
        if any(lb.get("self") for lb in labels):
            why["identity_label:drop"] += 1   # unprompted identity talk anywhere: the whole chat goes
            return None
    orig_j = -1         # index of the current reply among the chat's original replies (= the label index)
    prev_reply = None   # moth's last reply, for "what X"
    start = 0           # index in out of the user turn that opened the current exchange
    reason = None
    cut = human_repeat_at(d)   # the sim repeated itself: keep the chat up to there
    for i, m in enumerate(msgs):
        if i == cut:
            start, reason = len(out), "human_repeat"
            break
        m = dict(m)
        if m["role"] == "user":
            start = len(out)
            u = m["content"]
            ctx["after_agent"] = u.startswith(AGENT_TAG)
            ctx["user"], ctx["tool_text"] = u.lower().replace("’", "'"), ""
            if ctx["after_agent"]:
                ctx["agent_msg"] = ctx["user"]
            if re.search(r"\bbugs?\b", ctx["user"]):
                ctx["bug_said"] = True
            x = None if ctx["after_agent"] or prev_reply is None else what_x(u)
            if x is not None and not x_in(x, prev_reply + " " + ctx["agent_msg"]):
                reason = "made_up_what"
                break
        if m["role"] == "tool":
            seen_text += "\n".join(m["results"]).lower()
            ctx["tool_text"] += " " + "\n".join(m["results"]).lower()
        if m["role"] == "assistant":
            for c in m.get("tool_calls") or []:
                if c["name"] not in ("bash", "read", "write"):
                    why["bad_tool:drop"] += 1
                    return None
                cmd = str(c["arguments"].get("command", ""))
                if c["name"] == "bash" and (COMPLEX_CMD.search(cmd) or not TINY_CMD.fullmatch(cmd)):
                    reason = "complex_cmd"   # moth should look with one simple command at a time
                    break
                if c["name"] == "write":
                    seen_text += " " + str(c["arguments"].get("path", "")).lower()
            if reason:
                break
            if m.get("content"):
                orig_j += 1
                if labels is not None and not labels[orig_j]["on"]:
                    reason = "off_gist"
                    break
                t = clean_reply(m["content"])
                if t is None:
                    reason = "style"
                    break
                if any(f not in names and f not in seen_text for f in FILE_RE.findall(t)):
                    why["invented_file:drop"] += 1
                    return None
                if any(f not in names and f not in seen_text and f not in ("and", "or") for f in DIR_RE.findall(t)):
                    why["invented_dir:drop"] += 1
                    return None
                reason = reply_problem(t.replace("’", "'"), ctx)
                if reason:
                    break
                ctx["after_agent"] = False
                ctx["n_replies"] += 1
                m["content"] = prev_reply = t
        out.append(m)
    if reason in DROP_REASONS:
        why[f"{reason}:drop"] += 1
        return None
    if reason:
        out = out[:start]
    n_replies = sum(bool(m.get("content")) for m in out if m["role"] == "assistant")
    if not out or out[-1]["role"] != "assistant" or not out[-1].get("content") or (reason and n_replies < 2):
        why[f"{reason or 'no_final'}:drop"] += 1
        return None
    if reason:
        why[f"{reason}:cut"] += 1
    return out


def seed_restart_line(path: str, gap: int = 1500) -> int:
    """Last line before a generator restart: on resume it requeues every seed it hasn't written (failed ones
    from the start of its range first), so the restart is the first seed far below the running max. Normal
    out-of-order completion stays well under `gap` (r3 a/b/c3: no such jump anywhere else)."""
    top = -1
    for i, line in enumerate(open(path)):
        s = json.loads(line)["seed"]
        if s < top - gap:
            return i
        top = max(top, s)
    raise SystemExit(f"{path}: no restart found (seed never jumps back > {gap}): pass its line count explicitly")


def agent_cutoffs(spec: str, paths: list[str]) -> dict[str, int]:
    """--drop-agent-before a.jsonl:13720,b.jsonl:auto -> {basename: last line (1-based) whose agent chats go}."""
    cut = {}
    for item in filter(None, spec.split(",")):
        name, n = item.rsplit(":", 1)
        if n == "auto":
            path = next(p for p in paths if os.path.basename(p) == name)
            n = seed_restart_line(path)
            print(f"{name}: generator restart after line {n}")
        cut[name] = int(n)
    return cut


def parse_caps(spec: str) -> dict[str, float]:
    """--cap-file r2_train.jsonl:0.08,r2_long.jsonl:0.02 -> {basename: max share of the final kept chats}."""
    caps = {name: float(f) for name, f in (item.rsplit(":", 1) for item in filter(None, spec.split(",")))}
    assert sum(caps.values()) < 1, caps
    return caps


def apply_caps(items: list[dict], caps: dict[str, float], rng: random.Random) -> tuple[list[dict], dict]:
    """Subsample the chats from capped files so each makes up at most its share of the result:
    with R uncapped chats, total N = R / (1 - sum f) and file f keeps min(available, f * N)."""
    if not caps:
        return items, {}
    by = collections.defaultdict(list)
    for it in items:
        by[it["src"] if it["src"] in caps else None].append(it)
    total = len(by[None]) / (1 - sum(caps.values()))
    out, info = list(by[None]), {}
    for name, f in caps.items():
        pool = by.get(name, [])
        n = min(len(pool), int(f * total))
        out += rng.sample(pool, n)
        info[name] = (n, len(pool))
    return out, info


# a chat where the friend gives a name can't be stitched to one where moth calls them something else
NAMED = re.compile(r"\bmy name(?:'s| is)\b|\bcall me\b|\bi'?m called\b|\bname'?s [a-z]")
NAME_CAP = re.compile(r"\b(?:I'm|I am|Im|i'm|im|i am|it's|its) ([A-Z][a-z]+)\b")
NOT_NAME = set("I Just Not So Back Here Fine Good Okay Ok Sorry Sure Tired Sad Bored Home Done Still Also Really Very "
               "Pretty Kinda Gonna Going Doing Feeling Trying Moth The A An Glad Happy Literally Actually Totally "
               "Honestly Always Never Getting Late New Busy Ready Out Off Into Like Lol Thinking Working".split())


def has_name(d_or_msgs) -> bool:
    msgs = d_or_msgs["messages"] if isinstance(d_or_msgs, dict) else d_or_msgs
    for m in msgs:
        if m["role"] != "user" or m["content"].startswith(AGENT_TAG):
            continue
        u = m["content"].replace("’", "'")
        if NAMED.search(u.lower()) or any(n not in NOT_NAME for n in NAME_CAP.findall(u)):
            return True
    return False


def stitchable(item: dict) -> bool:
    """Talk chats only, no tools (each chat has its own home folder) and no given name."""
    return (item["type"] == "talk" and "name" not in item.get("scenario", "")
            and not any(m.get("tool_calls") for m in item["messages"] if m["role"] == "assistant")
            and not has_name(item["messages"]))


def mix_report(items: list[dict]) -> dict:
    """The build's own sanity numbers over kept chats: per type, and per reply."""
    types = collections.Counter(it["type"] for it in items)
    n = ident = qback = motif_any = 0
    idref = collections.Counter()
    nt = collections.Counter()
    cards = collections.Counter("moth" if (it.get("card") or MOTH_CARD)["id"] == "moth" else
                                "blank" if (it.get("card") or {}).get("id") == "blank" else "pool" for it in items)
    motif = collections.Counter()
    wx = wx_ok = 0
    for it in items:
        prev, last_user = None, ""
        for m in it["messages"]:
            if m["role"] == "user" and prev is not None and not m["content"].startswith(AGENT_TAG):
                x = what_x(m["content"])
                if x is not None:
                    wx += 1
                    wx_ok += x_in(x, prev)
            if m["role"] == "user":
                last_user = m["content"]
            if m["role"] == "tool":
                last_user += " " + " ".join(m["results"])
            if m["role"] == "assistant" and m.get("content"):
                t = m["content"].replace("’", "'")
                n += 1
                nt[it["type"]] += 1
                idref[it["type"]] += identity_ref(t, it.get("card") or MOTH_CARD, last_user)
                ident += bool(IDENTITY.search(t))
                qback += "?" in t
                hits = [k for k, r in MOTIFS.items() if r.search(t)]
                motif.update(hits)
                motif_any += bool(hits)
                prev = t
    return {"chats": len(items), "types": dict(types), "replies": n,
            "replies_per_chat": round(n / max(len(items), 1), 2),
            "identity_rate": round(ident / max(n, 1), 4), "question_back_rate": round(qback / max(n, 1), 4),
            "motif_any_rate": round(motif_any / max(n, 1), 4),
            "motif_rates": {k: round(motif[k] / max(n, 1), 4) for k in MOTIFS},
            "what_x_turns": wx, "what_x_contingent": round(wx_ok / max(wx, 1), 4),
            "identity_ref_rate": round(sum(idref.values()) / max(n, 1), 4),
            "identity_ref_by_type": {k: round(idref[k] / max(nt[k], 1), 4) for k in nt}, "cards": dict(cards)}


def encode(tok: Tokenizer, msgs: list[dict], card: dict | None = None) -> tuple[list[int], list[int]]:
    ids, mask = [], []
    full = [{"role": "system", "content": student_system(card)}] + msgs
    for m in full:
        text = render([m])
        if m["role"] == "assistant":
            head = "<|im_start|>assistant\n"
            h = tok.encode(head, add_special_tokens=False).ids
            b = tok.encode(text[len(head):], add_special_tokens=False).ids
            ids += h + b
            mask += [0] * len(h) + [1] * len(b)
        else:
            t = tok.encode(text, add_special_tokens=False).ids
            ids += t
            mask += [0] * len(t)
    return ids, mask


def n_replies(msgs: list[dict]) -> int:
    return sum(bool(m.get("content")) for m in msgs if m["role"] == "assistant")


def bucket(n: int) -> str:
    return "2-3" if n <= 3 else "4-7" if n <= 7 else "8-13" if n <= 13 else "14+"


EVAL_KEY_CHARS = 200


def eval_agent_keys(path: str) -> set[str]:
    """Held-out eval agent messages (a JSON list of {"text": ...} or strings) -> their cleaned first 200 chars,
    the form they take in a training chat's "[the agent said]" turn."""
    return {clean_agent_message(e["text"] if isinstance(e, dict) else e)[:EVAL_KEY_CHARS] for e in json.load(open(path))}


def uses_eval_agent_msg(d: dict, keys: set[str]) -> bool:
    return any(m["content"][len(AGENT_TAG):].strip()[:EVAL_KEY_CHARS] in keys for m in d["messages"]
               if m["role"] == "user" and m["content"].startswith(AGENT_TAG))


def filter_chats(paths: list[str], why: collections.Counter, cutoff: dict | None = None, held: set | None = None,
                 eval_keys: set | None = None, labels: dict | None = None):
    """keep() over every chat of every file -> (kept items, keep-rate table by original moth-reply count)."""
    items, by_len = [], collections.defaultdict(collections.Counter)
    for path in paths:
        src = os.path.basename(path)
        last_old = (cutoff or {}).get(src, 0)
        for i, line in enumerate(open(path), 1):
            d = json.loads(line)
            why["chats_in"] += 1
            if d["seed"] in (held or ()):
                why["heldout"] += 1
                continue
            if eval_keys and uses_eval_agent_msg(d, eval_keys):
                why["eval_agent_msg"] += 1   # reacts to one of the eval's held-out agent messages
                continue
            typ = chat_type(d)
            if typ == "agent" and i <= last_old:
                why["agent_old_prompt"] += 1   # before the agent-prompt fix: "i'm a bug too", "i can't code"
                continue
            orig = n_replies(d["messages"])
            b = by_len[(typ, bucket(orig))]
            b["in"] += 1
            lab = None if labels is None else labels.get(d["seed"])
            if labels is not None:
                why["labelled_chats"] += lab is not None
                if lab is not None:
                    why["labelled_replies"] += len(lab)
                    why["on_gist_replies"] += sum(x["on"] for x in lab)
                    why["self_replies"] += sum(bool(x.get("self")) for x in lab)
            msgs = keep(d, why, lab)
            if msgs is None:
                continue
            b["kept"] += 1
            b["cut"] += n_replies(msgs) < orig
            items.append({"seed": d["seed"], "type": typ, "src": src, "scenario": d.get("scenario", ""),
                          "messages": msgs, "home": d["home"], "card": d.get("card")})
    return items, by_len


def print_keep_rates(by_len) -> None:
    for typ in sorted({t for t, _ in by_len}):
        row = []
        for bk in ("2-3", "4-7", "8-13", "14+"):
            c = by_len.get((typ, bk))
            if c and c["in"]:
                row.append(f"{bk}: {c['kept'] / c['in']:.0%} of {c['in']} ({c['cut']} cut)")
        print(f"  keep rate by moth replies, {typ}: " + ", ".join(row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", default="data/sft")
    ap.add_argument("--tokenizer", default="data/tiny-agent-112m/tokenizer.json")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--sessions", type=int, default=1000,
                    help="extra long examples: kept talk chats (no tools, no given name) chained into one "
                         "conversation (1.5k..--max-len tokens); each chat is used at most once")
    ap.add_argument("--drop-agent-before", default="",
                    help="name.jsonl:LINE|auto,...: drop agent chats on lines <= LINE of that file (old agent prompt; "
                         "auto = the generator restart, found from the seeds)")
    ap.add_argument("--exclude-seeds", default=None, help="file of seeds to leave out (held-out probe chats)")
    ap.add_argument("--exclude-agent-msgs", default=None,
                    help="JSON list of held-out eval agent messages: drop chats whose [the agent said] turn is one")
    ap.add_argument("--labels", default="",
                    help="comma-separated moth.gist labels.jsonl files (seeds unique across inputs, as the fan-out "
                         "seed ranges guarantee): cut each chat at its first OFF-gist reply; report the ON rate")
    ap.add_argument("--cap-file", default="",
                    help="name.jsonl:FRAC,...: chats from that file make up at most FRAC of the kept chats "
                         "(after filtering), e.g. r2_train.jsonl:0.08,r2_long.jsonl:0.02")
    a = ap.parse_args()
    tok = Tokenizer.from_file(a.tokenizer)
    cutoff = agent_cutoffs(a.drop_agent_before, a.inputs)
    caps = parse_caps(a.cap_file)
    for name in caps:
        if name not in {os.path.basename(p) for p in a.inputs}:
            raise SystemExit(f"--cap-file {name}: not among the inputs")
    held = {int(x) for x in open(a.exclude_seeds).read().split()} if a.exclude_seeds else set()
    why = collections.Counter()
    eval_keys = eval_agent_keys(a.exclude_agent_msgs) if a.exclude_agent_msgs else set()
    labels = load_labels(a.labels.split(",")) if a.labels else None
    items, by_len = filter_chats(a.inputs, why, cutoff, held, eval_keys, labels)
    rng = random.Random(0)
    items, cap_info = apply_caps(items, caps, rng)
    rows, kept, same = [], [], collections.Counter()
    for it in items:
        replies = [m["content"] for m in it["messages"] if m["role"] == "assistant" and m.get("content")]
        if any(same[r] >= MAX_SAME_REPLY for r in replies):
            why["repeated_reply:drop"] += 1
            continue
        same.update(replies)
        ids, mask = encode(tok, it["messages"], it.get("card"))
        if len(ids) > a.max_len:
            why["too_long:drop"] += 1
            continue
        rows.append((ids, mask))
        kept.append(it)
    order = list(range(len(rows)))
    rng.shuffle(order)
    rows, kept = [rows[i] for i in order], [kept[i] for i in order]
    nv = max(1, int(len(rows) * a.val_frac))
    # long conversations: moth never sees chats much past ~1k tokens otherwise, and reply quality falls off
    # with depth. Chain train talk chats (never val ones) into one session so later replies sit deep in context.
    # Talk chats only and each at most once: stitched chats with different homes/names taught moth to ignore
    # what was said earlier.
    # One card per session (the system prompt is the card's): the pool is grouped by card, groups in random order.
    pool = [k for k in range(nv, len(kept)) if stitchable(kept[k])]
    rng.shuffle(pool)
    cid = lambda k: (kept[k].get("card") or MOTH_CARD).get("id")   # noqa: E731
    gkey = {c: rng.random() for c in sorted({cid(k) for k in pool})}
    pool.sort(key=lambda k: gkey[cid(k)])
    sessions = []
    while len(sessions) < a.sessions and len(pool) >= 2:
        msgs, ids_mask, used = [], None, 0
        target = rng.randint(1500, a.max_len)   # fill toward a random depth, so replies land all over 0..max_len
        card = kept[pool[-1]].get("card")
        while pool and cid(pool[-1]) == (card or MOTH_CARD).get("id"):
            cand = encode(tok, msgs + kept[pool[-1]]["messages"], card)
            if len(cand[0]) > target:
                break
            msgs, ids_mask, used = msgs + kept[pool.pop()]["messages"], cand, used + 1
        if used >= 2:
            sessions.append(ids_mask)
        elif pool and used == 0:
            pool.pop()   # a chat too long to pair at this target: skip it
        # used == 1: its card has no partner left that fits; that chat stays a normal row
    rows += sessions

    # the build's own sanity report: what the model will actually see
    n_in = why["chats_in"]
    rep = mix_report(kept)
    print(f"{n_in} chats -> kept {len(kept)} ({len(kept) / max(n_in, 1):.0%})")
    cuts = {k[:-4]: v for k, v in sorted(why.items()) if k.endswith(":cut")}
    drops = {k[:-5]: v for k, v in sorted(why.items()) if k.endswith(":drop")}
    other = {k: v for k, v in why.items() if ":" not in k and k != "chats_in"}
    print(f"  truncated (kept the prefix) {sum(cuts.values())}: {cuts}")
    print(f"  dropped {sum(drops.values())}: {drops}; skipped {other}")
    if cap_info:
        print("  capped files (kept/available): " + ", ".join(f"{k} {n}/{m}" for k, (n, m) in cap_info.items()))
    print_keep_rates(by_len)
    src = collections.Counter(k["src"] for k in kept)
    print(f"  effective mix: types {rep['types']}, files {dict(src)}, "
          f"+{len(sessions)} stitched sessions (mean {sum(len(r[0]) for r in sessions) / max(len(sessions), 1):.0f} "
          f"tokens, from {sum(stitchable(k) for k in kept[nv:])} stitchable talk chats)")
    if labels is not None:
        print(f"  gist labels: {why['labelled_chats']}/{n_in} chats labelled; ON {why['on_gist_replies']}/"
              f"{why['labelled_replies']} replies ({why['on_gist_replies'] / max(why['labelled_replies'], 1):.1%}); "
              f"unprompted identity (label 'self') {why['self_replies'] / max(why['labelled_replies'], 1):.2%} of "
              f"replies in, 0 kept (hard max 1%)")
    print(f"  cards {rep['cards']}; word-matcher identity refs {rep['identity_ref_rate']:.1%} of kept replies "
          f"({', '.join(f'{k} {v:.1%}' for k, v in rep['identity_ref_by_type'].items())}; ~2% false-positive floor, "
          f"the labels are the real check)")
    print(f"  moth replies {rep['replies']} ({rep['replies_per_chat']}/chat): identity {rep['identity_rate']:.1%}, "
          f"question back {rep['question_back_rate']:.1%}, any motif {rep['motif_any_rate']:.1%} "
          f"({', '.join(f'{k} {v:.1%}' for k, v in rep['motif_rates'].items())}); "
          f"'what X' turns {rep['what_x_turns']}, X in moth's previous reply {rep['what_x_contingent']:.0%}")
    os.makedirs(a.out, exist_ok=True)
    torch.save(rows[nv:], os.path.join(a.out, "train.pt"))
    torch.save(rows[:nv], os.path.join(a.out, "val.pt"))
    with open(os.path.join(a.out, "kept.jsonl"), "w") as f:
        for k in kept:
            f.write(json.dumps(k, ensure_ascii=False) + "\n")
    toks = sum(len(r[0]) for r in rows)
    tool = sum(any(m.get("tool_calls") for m in k["messages"]) for k in kept)
    print(f"{toks / 1e6:.2f}M tokens, {sum(sum(r[1]) for r in rows) / 1e6:.2f}M trained, "
          f"{tool / max(len(kept), 1):.0%} of chats use tools")


if __name__ == "__main__":
    main()
