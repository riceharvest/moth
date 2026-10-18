"""Deterministic (no judge) metrics over eval results: what the judge kept missing, measured with regexes and counts.

  python -m moth.metrics evals/v2/results.json [evals/ck1/results.json ...] [--cat style]

With several files it prints one column per file, then the same table on the probes all files share (probe sets
differ between rounds, and the extra probes are the hard ones). A "reply" is one {"say": ...} event.

  identity crutch   "i'm (just) a ... moth/pattern/bug/thing", "on the (graphics) card", "made of numbers", "112 million numbers";
                    replies to "who/what are you", "define moth", "are you a <x>", "tell me about yourself" are skipped
  bible copy        reply shares a word 4-gram with a moth example line in persona.TEACHER_BIBLE / agent.py
                    (the "moth:" lines, the right side of "->" examples, quoted lowercase examples of 5+ words), live
                    plus the round-3 lines frozen in moth/bible_lines_r3.json; stopword-only 4-grams skipped
  dodge             "i can't/don't understand", or "i'm just/only a ... <moth|bug|thing|pattern>" followed in the same sentence
                    by "i can't/don't" or ", not a/the ..." ("i'm just a moth on a card, not a big computer brain")
  question back     reply contains "?"
  repeat exact      reply equals an earlier reply in the same chat
  repeat near       ... or is >= 0.7 similar to one (difflib ratio on words)
  top-20 3-gram     share of replies containing at least one of the 20 3-grams found in the most replies (across all chats)
  distinct-2/3      unique / total word n-grams over all replies (depends on the number of replies: compare like with like)
  motif X           share of replies with the word (fan(s), glow*, warm*, hum/hums/humming, light(s), card(s))
  identity (labels) share of replies the judge flagged "self": unprompted identity talk. AUTHORITATIVE; hard max 1%
  identity_ref      the cheap word matcher (moth.gist.identity_ref: kind/home words, moth's stock images, "i'm just a
                    ...") when the friend didn't bring it up; ~2% false-positive floor, a pre-filter only
  claims other id   reply claims to be something its card isn't ("i'm a moth" on a cat card)
  on_gist           share of replies the judge labelled ON gist (moth.eval --judge-url; moth.gist's label prompt)
  agent chats (categories starting with "agent"):
  bug on non-bug    reply says bug/grandma/1947 though no message so far (agent or friend) said "bug"
  explain/advice    advice ("you should", "make sure", "try running"...), explanation ("it means", "basically", "in other
                    words"...) or a summary (>= 3 content words of 4+ letters copied from the agent message). Saying
                    what happened ("it fixed it!", "it broke") is the gist, not an explanation.
  code echo         reply repeats an identifier from the agent message (file name, path, snake_case, dotted, camelCase)
"""
from __future__ import annotations

import argparse
import collections
import difflib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

AGENT_TAG = "[the agent said]"
STOP = set("""i i'm im me my mine you you're your yours we our us he she it it's its they them their this that these those
a an the and or but so if then than to of in on at by for with from up down out over under into onto about as is am are was
were be been being do does did don't doesn't didn't can can't could would should will won't just very really too also not no
yes oh ok okay what what's who why how when where which there here all some any one more most much many little bit like
have has had get got go goes going make made know think say said see look want need let lets well still even only maybe
thing things way now today again back off lot lots always never every each
""".split())

IDENTITY = re.compile(
    r"\bi(?:'m| am) (?:just |only |still |really |a little )?(?:a|an|the|one|some) (?:[a-z',]+ ){0,4}?(?:moth|pattern|bug|thing)s?\b"
    r"|\bon (?:the|a|my|this) (?:[a-z0-9]+ ){0,2}?card\b"
    r"|\bmade (?:of|from) (?:[a-z]+ ){0,2}?numbers\b"
    r"|\b(?:112|million) (?:little |tiny |small )?numbers\b")
DODGE = re.compile(
    r"\bi (?:can't|cannot|can not|don't|do not) (?:really |quite )?understand\b"
    r"|\bi(?:'m| am) (?:just|only) (?:a|an) (?:[a-z']+ ){0,3}?(?:moth|bug|thing|pattern)\b[^.!?]{0,40}?"
    r"(?:\bi (?:can't|cannot|don't)\b|, not (?:a|an|the)\b)")
# the friend asked what moth is: "i'm a moth" is then an answer, not a crutch
ASKS_IDENTITY = re.compile(r"\b(?:who|what) (?:are|r) (?:you|u)\b|\bdefine moth\b|\bare (?:you|u) (?:a|an) [a-z]+|"
                           r"\btell me about (?:yourself|you)\b|\bwhat(?:'s| is) a moth\b|\bwho is moth\b", re.I)
MOTIFS = {"fan": r"\bfans?\b", "glow": r"\bglow", "warm": r"\bwarm", "hum": r"\bhum(?:s|ming|med)?\b",
          "light": r"\blights?\b", "card": r"\bcards?\b"}
BUGGY = re.compile(r"\bbugs?\b|\bgrandma|\b1947\b")
# gist-first (2026-10-19): "it fixed it!", "it's asking you something", "i'd pick yes. yes is friendly." are the target
# behaviour now, so only real advice and explanation count
ADVICE = re.compile(
    r"\byou should\b|\bi(?:'d| would) (?:recommend|suggest)\b|\bmake sure\b|\btry (?:running|using|adding|checking|restarting)\b"
    r"|\bgo ahead and\b|\bpromote it\b|\b(?:it|that|this) means\b|\bbasically\b|\bin other words\b"
    r"|\b[a-z]+ is (?:better|safer|faster|the best (?:one|choice|option))\b|\bbecause (?:it|the) (?:is|will|would) (?:more|less|faster|safer)\b")
IDENT = re.compile(r"[A-Za-z0-9_./-]*(?:[._/][A-Za-z0-9]|[a-z][A-Z])[A-Za-z0-9_./-]*")


def norm(text: str) -> str:
    return (text or "").lower().replace("’", "'").replace("‘", "'")


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", norm(text))


def ngrams(ws: list[str], n: int) -> list[tuple]:
    return [tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)]


def bible_lines() -> list[str]:
    """moth's own example lines: the live teacher bible (which includes agent.py's rules) plus the round-3 lines frozen
    in bible_lines_r3.json (the bible keeps changing; checkpoints trained on older lines still copy those)."""
    from moth.persona import TEACHER_BIBLE
    lines = [ln.split(":", 1)[1].strip() for ln in TEACHER_BIBLE.splitlines()
             if ln.strip().startswith(("moth:", "you:"))]
    lines += re.findall(r'->\s*"([^"]+)"', TEACHER_BIBLE)
    lefts = set(re.findall(r'"([^"]+)"\)?\s*->', TEACHER_BIBLE))   # the agent's / friend's side of an example
    lines += [q for q in re.findall(r'"([^"]+)"', TEACHER_BIBLE)
              if q == q.lower() and q not in lefts and len(q.split()) >= 5 and "let me know" not in q]
    frozen = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bible_lines_r3.json")
    if os.path.exists(frozen):
        lines += json.load(open(frozen))["lines"]
    return list(dict.fromkeys(lines))


_BIBLE: set | None = None


def bible_4grams() -> set:
    global _BIBLE
    if _BIBLE is None:
        _BIBLE = {g for ln in bible_lines() for g in ngrams(words(ln), 4) if not all(w in STOP for w in g)}
    return _BIBLE


def replies_of(r: dict) -> list[dict]:
    """Every reply of a chat with its context."""
    out, said_bug = [], False
    for ti, t in enumerate(r["turns"]):
        u = t["user"]
        said_bug = said_bug or re.search(r"\bbugs?\b", u, re.I) is not None
        agent_msg = u[len(AGENT_TAG):] if u.startswith(AGENT_TAG) else None
        seen = u   # the friend's line plus what moth's tools showed this turn: words from there are "prompted"
        for e in t["events"]:
            if "tool" in e:
                seen += "\n" + str(e.get("result", ""))
            if "say" in e:
                out.append({"text": e["say"], "turn": ti, "user": u, "said_bug": said_bug, "agent_msg": agent_msg,
                            "card": r.get("card"), "seen": seen})
    return out


def compute(results: list[dict]) -> dict:
    """metric -> (value, n): value is a share (0-1) or None when nothing to measure."""
    big = bible_4grams()
    allr, agent_r = [], []
    rep_exact = rep_near = 0
    for r in results:
        rs = replies_of(r)
        prev: list[list[str]] = []
        last_agent = None
        for x in rs:
            ws = words(x["text"])
            x["ws"] = ws
            if any(x["text"] == p["text"] for p in allr[len(allr) - len(prev):]):
                rep_exact += 1
                rep_near += 1
            elif any(difflib.SequenceMatcher(None, ws, p).ratio() >= 0.7 for p in prev if p):
                rep_near += 1
            prev.append(ws)
            allr.append(x)
            if r["cat"].startswith("agent"):
                last_agent = x["agent_msg"] if x["agent_msg"] is not None else last_agent
                x["last_agent"] = last_agent
                agent_r.append(x)
    n = len(allr)

    def share(pred, xs):
        return (sum(1 for x in xs if pred(x)) / len(xs) if xs else None, len(xs))

    M = {"replies": (n, n), "chats": (len(results), len(results))}
    M["identity crutch"] = share(lambda x: IDENTITY.search(norm(x["text"])), [x for x in allr if not ASKS_IDENTITY.search(x["user"])])
    M["bible copy (4-gram)"] = share(lambda x: any(g in big for g in ngrams(x["ws"], 4)), allr)
    M["dodge"] = share(lambda x: DODGE.search(norm(x["text"])), allr)
    M["question back"] = share(lambda x: "?" in x["text"], allr)
    M["repeat in chat (exact)"] = (rep_exact / n if n else None, n)
    M["repeat in chat (near)"] = (rep_near / n if n else None, n)
    df = collections.Counter(g for x in allr for g in set(ngrams(x["ws"], 3)))
    top = {g for g, _ in df.most_common(20)}
    M["top-20 3-gram share"] = share(lambda x: any(g in top for g in ngrams(x["ws"], 3)), allr)
    M["top 3-grams"] = (", ".join(" ".join(g) for g, _ in df.most_common(5)), n)
    for k in (2, 3):
        grams = [g for x in allr for g in ngrams(x["ws"], k)]
        M[f"distinct-{k}"] = (len(set(grams)) / len(grams) if grams else None, len(grams))
    for name, pat in MOTIFS.items():
        M[f"motif {name}"] = share(lambda x, p=re.compile(pat): p.search(norm(x["text"])), allr)
    from moth.gist import claims_other_identity, identity_ref
    M["identity_ref (matcher, ~2% floor)"] = share(lambda x: identity_ref(x["text"], x.get("card"), x["seen"]), allr)
    M["claims other identity"] = share(lambda x: claims_other_identity(x["text"], x.get("card")), allr)
    labs = [lb for r in results if isinstance(r.get("gist"), list) for lb in r["gist"]]
    M["on_gist (judge labels)"] = (sum(lb["on"] for lb in labs) / len(labs) if labs else None, len(labs))
    M["unprompted identity (labels, max 1%)"] = (sum(bool(lb.get("self")) for lb in labs) / len(labs) if labs else None,
                                                 len(labs))
    M["agent: bug on non-bug"] = share(lambda x: BUGGY.search(norm(x["text"])), [x for x in agent_r if not x["said_bug"]])
    M["agent: explain/advice"] = share(lambda x: ADVICE.search(norm(x["text"])) or summary_copy(x), agent_r)
    M["agent: code echo"] = share(code_echo, agent_r)
    for kind, oks in checks_by_kind(results).items():
        M[f"check {kind}"] = (sum(oks) / len(oks), len(oks))
    return M


def summary_copy(x) -> bool:
    if not x.get("last_agent"):
        return False
    a = {w for w in words(x["last_agent"]) if len(w) >= 4 and w not in STOP}
    return len(a & {w for w in x["ws"] if len(w) >= 4}) >= 3


def code_echo(x) -> bool:
    if not x.get("last_agent"):
        return False
    ids = {m.lower().strip("./-") for m in IDENT.findall(x["last_agent"])}
    ids = {i for i in ids if len(i) >= 4 and not re.fullmatch(r"[\d.]+", i)}
    t = norm(x["text"])
    return any(i in t for i in ids)


def checks_by_kind(results: list[dict]) -> dict[str, list[bool]]:
    """Automatic per-chat checks; "what_x on-word" counts every "what X?" turn."""
    out: dict[str, list[bool]] = collections.defaultdict(list)
    for r in results:
        for w in r.get("what_x", []):
            if w.get("on") is not None:
                out["what_x on-word"].append(bool(w["on"]))
        if r.get("check") is not None and not r.get("what_x"):
            out[r.get("check_kind") or "write/recall"].append(bool(r["check"]))
    return dict(sorted(out.items()))


def fmt(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, str):
        return v
    if isinstance(v, int):
        return str(v)
    return f"{100 * v:.1f}%"


def table(cols: dict[str, list[dict]]) -> str:
    """Markdown table, one column per label."""
    ms = {lab: compute(rs) for lab, rs in cols.items()}
    keys = []
    for m in ms.values():
        keys += [k for k in m if k not in keys]
    L = ["| metric | " + " | ".join(ms) + " |", "|---|" + "---|" * len(ms)]
    for k in keys:
        cells = []
        for m in ms.values():
            v, n = m.get(k, (None, 0))
            cells.append(fmt(v) + (f" (n={n})" if k.startswith(("agent:", "check")) and v is not None else ""))
        L.append(f"| {k} | " + " | ".join(cells) + " |")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="eval results.json files (one column each)")
    ap.add_argument("--cat", default=None, help="only these categories (comma-separated)")
    a = ap.parse_args()
    cols = {}
    for f in a.files:
        rs = json.load(open(f))
        if a.cat:
            rs = [r for r in rs if r["cat"] in a.cat.split(",")]
        lab = os.path.basename(os.path.dirname(os.path.abspath(f))) or f
        while lab in cols:
            lab += "'"
        cols[lab] = rs
    print("## all probes\n\n" + table(cols))
    names = [{r["name"] for r in rs} for rs in cols.values()]
    if len(cols) > 1 and len({frozenset(n) for n in names}) > 1:
        common = set.intersection(*names)
        print(f"\n## shared probes only ({len(common)} probe names)\n")
        print(table({lab: [r for r in rs if r["name"] in common] for lab, rs in cols.items()}))
    if not a.cat and any(r["cat"] == "style" for rs in cols.values() for r in rs):
        print("\n## user-style probes\n")
        print(table({lab: [r for r in rs if r["cat"] == "style"] for lab, rs in cols.items()}))


if __name__ == "__main__":
    main()
