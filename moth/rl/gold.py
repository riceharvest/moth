"""The hand-labelled gold set: a 600-reply labelling sheet, and each judge's agreement with it per stratum.

  python -m moth.rl gold-sheet --inputs evals/rl_v5_x/episodes.json out/bakeoff/labelled_replies.jsonl \
      --out gold_sheet.jsonl [--n 600] [--seed 0]

The sheet = n/2 RANDOM replies (uniform over everything given: the unbiased estimate of each judge's accuracy) +
n/2 STRATIFIED replies (where judges fail: identity-suspect replies flagged by the identity_ref matcher or by any
judge's "self", borderline gist (judges disagree on "on", or a forward-pass P in 0.35-0.65), and at least 60 per
language). Each line: {"id", "stratum", "reasons", "card", "lang", "context", "friend", "text", "judge_labels",
"human": null}. A labeller fills "human": {"on", "self", "specific"}; moth.rl.judge.load_gold reads the filled sheet
(lines with "human" left null are skipped), and agreement_by_stratum reports each judge's accuracy with a 95%
interval on the random and the stratified half separately (never pooled: the stratified half is biased on purpose).
A .csv twin of the sheet is written for spreadsheet labelling.

Inputs: moth.rl eval/run episodes.json (episodes with "turns" and the eval judge's "labels") and
moth.bakeoff_report's labelled_replies.jsonl ({"card", "lang", "friend", "text", "labels": {judge: {...}}}).
"""
from __future__ import annotations

import collections
import csv
import json
import random

from moth.gist import identity_ref
from moth.rl.llm import read_json

PER_LANG_MIN = 60


def _items_from_episodes(eps: list[dict], src: str) -> list[dict]:
    out = []
    for e in eps:
        card, lang = e.get("card") or {}, e.get("lang") or (e.get("card") or {}).get("lang") or "en"
        ctx: list[dict] = []
        i = 0
        for t in e["turns"]:
            for ev in t["events"]:
                lab = (e.get("labels") or [None] * 999)[i] if e.get("labels") else None
                out.append({"id": f"{src}:{e.get('spec_id', e.get('name'))}:{e.get('sample', 0)}:{i + 1}", "card": card,
                            "lang": lang, "context": list(ctx), "friend": t["user"], "text": ev["say"],
                            "judge_labels": {"eval_judge": lab} if lab else {}})
                ctx.append({"user": t["user"], "say": ev["say"]})
                i += 1
    return out


def _items_from_labelled(rows: list[dict], src: str) -> list[dict]:
    return [{"id": f"{src}:{r.get('teacher')}:{r.get('chat_id')}:{r.get('reply')}", "card": r.get("card") or {},
             "lang": r.get("lang") or "en", "context": [], "friend": r["friend"], "text": r["text"],
             "judge_labels": {k: v for k, v in (r.get("labels") or {}).items() if v}} for r in rows]


def load_items(paths: list[str]) -> list[dict]:
    items = []
    for p in paths:
        if p.endswith(".jsonl"):
            rows = [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]
            items += _items_from_labelled(rows, p)
        else:
            items += _items_from_episodes(read_json(p), p)
    seen, out = set(), []
    for it in items:   # the same reply from two sources counts once
        k = (it["friend"], it["text"], it.get("card", {}).get("id"))
        if k not in seen:
            seen.add(k)
            out.append(it)
    return out


def flags(it: dict) -> list[str]:
    """Why a reply is worth a hand label: identity suspicion or a borderline gist call."""
    why = []
    if identity_ref(it["text"], it["card"] or None, it["friend"]):
        why.append("identity_matcher")
    labs = [x for x in it["judge_labels"].values() if isinstance(x, dict)]
    if any(x.get("self") for x in labs):
        why.append("judge_self")
    ons = {bool(x.get("on")) for x in labs if "on" in x}
    if len(ons) > 1:
        why.append("judges_disagree_on")
    if any(0.35 <= (x.get("p") or {}).get("on", 0.0) <= 0.65 for x in labs):
        why.append("borderline_p_on")
    return why


def build_sheet(items: list[dict], n: int = 600, seed: int = 0, per_lang: int = PER_LANG_MIN) -> list[dict]:
    rng = random.Random(seed)
    pool = list(items)
    rng.shuffle(pool)
    half = n // 2
    rand = pool[:half]
    taken = {it["id"] for it in rand}
    rest = [it for it in pool if it["id"] not in taken]
    for it in rest:
        it["_why"] = flags(it)
    strat: list[dict] = []
    by_lang = collections.defaultdict(list)
    for it in rest:
        by_lang[it["lang"]].append(it)
    for lg, xs in sorted(by_lang.items()):   # >= per_lang per language, flagged ones first
        xs.sort(key=lambda x: -len(x["_why"]))
        strat += xs[:per_lang]
    chosen = {it["id"] for it in strat}
    flagged = sorted((it for it in rest if it["id"] not in chosen and it["_why"]), key=lambda x: -len(x["_why"]))
    strat += flagged[:max(0, half - len(strat))]
    if len(strat) < half:   # not enough flags: fill with more random ones
        chosen = {it["id"] for it in strat}
        strat += [it for it in rest if it["id"] not in chosen][:half - len(strat)]
    out = []
    for stratum, xs in (("random", rand), ("stratified", strat[:max(half, len(strat))])):
        for it in xs:
            out.append({"id": it["id"], "stratum": stratum, "reasons": it.get("_why") or flags(it), "card": it["card"],
                        "lang": it["lang"], "context": it["context"], "friend": it["friend"], "text": it["text"],
                        "judge_labels": it["judge_labels"], "human": None})
    return out


def write_sheet(rows: list[dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(path.rsplit(".", 1)[0] + ".csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "stratum", "lang", "card", "context", "friend", "reply", "human_on", "human_self",
                    "human_specific"])
        for r in rows:
            ctx = " | ".join(f"F: {c['user']} / B: {c['say']}" for c in r["context"][-3:])
            w.writerow([r["id"], r["stratum"], r["lang"], (r["card"] or {}).get("id"), ctx, r["friend"], r["text"],
                        "", "", ""])


def agreement_by_stratum(judge, gold: list[dict]) -> dict:
    """moth.rl.judge.agreement per stratum ("random", "stratified", or "all" for gold without strata), with 95%
    intervals on the accuracies."""
    from moth.rl.evalrun import wilson
    from moth.rl.judge import agreement
    groups = collections.defaultdict(list)
    for g in gold:
        groups[g.get("stratum") or "all"].append(g)
    out = {}
    for st, gs in sorted(groups.items()):
        a = agreement(judge, gs)
        for f in ("on", "self", "specific"):
            n, acc = a[f]["n"], a[f]["acc"]
            a[f]["acc"] = wilson(round(acc * n), n) if n else None
        out[st] = a
    return out
