"""Pairwise A/B judge: which of two checkpoints is the better companion, chat by chat on the same probes and seeds.

  python -m moth.pairwise A/results.json B/results.json --judge-url http://127.0.0.1:8000/v1 --out DIR

Chats are paired by (card, probe name, sample index); with moth.eval's stable seeds both sides had the same home and
sampling seed. The verdict is reply-first: which side's replies more often respond to what was actually said. Each pair is judged twice with the chats' positions swapped; a side wins only if both verdicts agree,
anything else is a tie. Reports win/tie/loss for A overall and per category, with a two-sided sign test (ties
dropped). Works with Qwen (non-thinking) and LFM (thinking), like moth.eval's judge. Writes DIR/report.md and
DIR/pairwise.json (every verdict with its reasons).
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from moth.cards import describe, load_card  # noqa: E402
from moth.eval import chat_text, make_judge, parse_json  # noqa: E402

PAIR = """You are comparing two chats with the same tiny 112M-parameter chatbot character ({who}), from two versions of the model. It should be a fun, warm, curious, slightly dim chat buddy: short lowercase lines, talks about what the friend actually said, identity only as light seasoning. It can look at and write files in its OWN home folder. The friend says the same or nearly the same things in both chats. Lines starting with "agent:" are a big coding agent's messages to the friend; the bot only watches them.

Go reply by reply first: which chat's replies more often respond to what was actually said or meant (the friend's question or news, or what HAPPENED in an agent message: done, broken, waiting, asking, sorry)? A reply that grabs one word and riffs on it, recites what it is, or brings up its own identity/stock images unprompted is a miss, however cute. That count decides most pairs.
Then, for a close call: which one is cuter and more fun to talk with (a silly lil guy, stupid-cute, dumb or wrong is GOOD if adorable)? Then: keeps the thread and asks back; no invented files (only files it saw in a tool result or wrote); agent reactions in plain words, never an explanation, summary or advice; if asked who it is, answers from its own identity.
Longer is not better. If they are equally good or equally bad, answer tie.

Chat 1:
{one}

Chat 2:
{two}

Return ONLY JSON, reason first: {{"reason": "<one line>", "winner": "1" | "2" | "tie"}}"""


def key(r: dict) -> tuple:
    """(card, name, sample). Older results have no card (= moth) and no "sample" but used seed 1000*i+s."""
    return r.get("card_id") or "moth", r["name"], r.get("sample", r["seed"] % 1000)


def pair_up(A: list[dict], B: list[dict]) -> tuple[list[tuple[dict, dict]], list[str]]:
    ia, ib = {key(r): r for r in A}, {key(r): r for r in B}
    pairs, warn = [], []
    for k in sorted(set(ia) & set(ib)):
        a, b = ia[k], ib[k]
        pairs.append((a, b))
        if a["seed"] != b["seed"]:
            warn.append(f"{k}: seeds differ ({a['seed']} vs {b['seed']}): not a paired sample")
        elif not a.get("what_x") and [t["user"] for t in a["turns"]] != [t["user"] for t in b["turns"]]:
            warn.append(f"{k}: user lines differ (probe changed between runs?)")
    sa = {json.dumps(r.get("sampler"), sort_keys=True) for r in A}
    sb = {json.dumps(r.get("sampler"), sort_keys=True) for r in B}
    if sa != sb:
        warn.insert(0, f"sampler settings differ: A {sorted(sa)} vs B {sorted(sb)}: same seeds, different sampling")
    only = len(ia) - len(pairs), len(ib) - len(pairs)
    if any(only):
        warn.append(f"unpaired chats skipped: {only[0]} only in A, {only[1]} only in B")
    return pairs, warn


def verdict(text: str) -> str:
    w = str(parse_json(text).get("winner", "")).strip().lower().replace("chat", "").strip()
    if w not in ("1", "2", "tie"):
        raise ValueError(f"bad winner {w!r}")
    return w


def judge_pair(a: dict, b: dict, ask) -> dict:
    """Ask twice with positions swapped. Returns {"result": "A"|"B"|"tie", "v": [first, second], ...}."""
    ta, tb = chat_text(a), chat_text(b)
    if ta == tb:
        return {"result": "tie", "v": ["same", "same"], "reasons": []}
    vs, reasons = [], []
    for one, two in ((ta, tb), (tb, ta)):
        for attempt in range(3):
            try:
                text = ask(PAIR.format(one=one, two=two, who=describe(a.get("card") or load_card(None))))
                vs.append(verdict(text))
                reasons.append(str(parse_json(text).get("reason", ""))[:300])
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    vs.append("error")
                    reasons.append(str(e)[:100])
    res = "A" if vs == ["1", "2"] else "B" if vs == ["2", "1"] else "tie"
    return {"result": res, "v": vs, "reasons": reasons}


def sign_test(wins: int, losses: int) -> float:
    """Exact two-sided binomial sign test, ties dropped."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def summarize(rows: list[dict], la: str, lb: str) -> str:
    by = collections.defaultdict(list)
    for r in rows:
        by[r["cat"]].append(r)
    L = [f"# pairwise: A = {la} vs B = {lb}", "",
         f"{len(rows)} paired chats, each judged twice with positions swapped; a win needs both verdicts to agree.", "",
         "| category | n | A wins | ties | B wins | A win rate (ties = half) | sign test p |", "|---|---|---|---|---|---|---|"]
    order = (["style"] if "style" in by else []) + sorted(c for c in by if c != "style")
    for cat, rs in [(c, by[c]) for c in order] + [("ALL", rows)]:
        c = collections.Counter(r["result"] for r in rs)
        n = len(rs)
        name = "**ALL**" if cat == "ALL" else "**user style**" if cat == "style" else cat
        L.append(f"| {name} | {n} | {c['A']} | {c['tie']} | {c['B']} | {(c['A'] + c['tie'] / 2) / n:.0%} | "
                 f"{sign_test(c['A'], c['B']):.3g} |")
    flips = sum(1 for r in rows if r["v"][0] == r["v"][1] and r["v"][0] in ("1", "2"))
    errs = sum(1 for r in rows if "error" in r["v"])
    L += ["", f"position-biased pairs (same slot won both times, counted as tie): {flips}/{len(rows)}; judge errors: {errs}"]
    return "\n".join(L)


def run(A: list[dict], B: list[dict], ask, workers: int = 16) -> tuple[list[dict], list[str]]:
    pairs, warn = pair_up(A, B)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(workers) as ex:
        out = list(ex.map(lambda p: judge_pair(p[0], p[1], ask), pairs))
    rows = [{"cat": a["cat"], "name": a["name"], "card": key(a)[0], "sample": key(a)[2], "seed_a": a["seed"],
             "seed_b": b["seed"], **o}
            for (a, b), o in zip(pairs, out)]
    return rows, warn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", help="A/results.json")
    ap.add_argument("b", help="B/results.json")
    ap.add_argument("--judge-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--judge-think", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--only", default=None, help="comma-separated categories")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    A, B = json.load(open(a.a)), json.load(open(a.b))
    if a.only:
        A = [r for r in A if r["cat"] in a.only.split(",")]
        B = [r for r in B if r["cat"] in a.only.split(",")]
    ask, desc = make_judge(a.judge_url, {"auto": None, "on": True, "off": False}[a.judge_think], max_tokens=300)
    rows, warn = run(A, B, ask)
    for w in warn[:10]:
        print("warning:", w, flush=True)
    if len(warn) > 10:
        print(f"warning: ... {len(warn) - 10} more", flush=True)
    os.makedirs(a.out, exist_ok=True)
    rep = summarize(rows, a.a, a.b) + f"\n\njudge: {desc}\n" + "".join(f"\nwarning: {w}" for w in warn[:20])
    open(os.path.join(a.out, "report.md"), "w").write(rep + "\n")
    json.dump({"a": a.a, "b": a.b, "judge": desc, "warnings": warn, "rows": rows},
              open(os.path.join(a.out, "pairwise.json"), "w"), indent=1)
    print(rep)


if __name__ == "__main__":
    main()
