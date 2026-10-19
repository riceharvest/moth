"""Tables for the teacher bake-off (moth.bakeoff), run locally on the pulled box output.

  python -m moth.bakeoff report results/bakeoff/rented/bk-1/bakeoff [--out results/bakeoff]

Writes <out>/report.md (speed, per-judge quality, judge agreement, pairwise), <out>/read_<teacher>.md (30 sampled
replies with context and both judges' labels, for reading by eye) and <out>/label_audit.md (self=true on turns where
the friend asked about identity or files, and replies the judges disagree on).
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FILES_ASK = re.compile(r"\b(?:folders?|files?|home|diary|ls|look in|what's in|whats in|saved|notes?|room|write|read)\b",
                       re.IGNORECASE)


def metric_sum(path: str) -> dict:
    out = collections.Counter()
    if not os.path.exists(path):
        return out
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        name = line.split("{")[0].split(" ")[0]
        try:
            out[name] += float(line.rsplit(" ", 1)[1])
        except ValueError:
            pass
    return out


def tstamp(path: str) -> float | None:
    try:
        return float(open(path + ".t").read())
    except (OSError, ValueError):
        return None


def last_counter(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    last = None
    for line in open(path):
        m = re.search(r"Counter\((\{.*?\})\)", line)
        if m:
            last = m.group(1)
    if not last:
        return {}
    return {k: int(v) for k, v in re.findall(r"'([^']+)': (\d+)", last)}


def speed(T: str) -> dict:
    sj = json.load(open(os.path.join(T, "serve.json"))) if os.path.exists(os.path.join(T, "serve.json")) else {}
    t0, t1 = tstamp(os.path.join(T, "metrics_0.txt")), tstamp(os.path.join(T, "metrics_1.txt"))
    D = [json.loads(x) for x in open(os.path.join(T, "dialogs.jsonl"))] if os.path.exists(os.path.join(T, "dialogs.jsonl")) else []
    out = {"model": sj.get("model"), "max_num_seqs": sj.get("max_num_seqs"), "concurrency": sj.get("concurrency"),
           "download_s": sj.get("download_s"), "load_s": sj.get("load_s"), "attempt": sj.get("attempt")}
    if t0 and t1:
        ts = sorted(d["bk"]["t"] for d in D if t0 <= d["bk"]["t"] <= t1 + 5)
        win = t1 - t0
        out["pass_s"] = round(win)
        out["chats_pass"] = len(ts)
        out["chats_min"] = len(ts) / win * 60 if win else None
        steady = [t for t in ts if t >= t0 + 120]
        out["chats_min_steady"] = len(steady) / (min(t1, ts[-1]) - (t0 + 120)) * 60 if len(steady) > 10 else None
        m0, m1 = metric_sum(os.path.join(T, "metrics_0.txt")), metric_sum(os.path.join(T, "metrics_1.txt"))
        gen = m1["vllm:generation_tokens_total"] - m0["vllm:generation_tokens_total"]
        pr = m1["vllm:prompt_tokens_total"] - m0["vllm:prompt_tokens_total"]
        out["gen_tok_s"] = gen / win if win else None
        out["prompt_tok_s"] = pr / win if win else None
        hits = next((m1[k] - m0[k] for k in m1 if "prefix_cache_hits" in k), None)
        qs = next((m1[k] - m0[k] for k in m1 if "prefix_cache_queries" in k), None)
        out["prefix_hit"] = hits / qs if hits is not None and qs else None
        rate = out["chats_min_steady"] or out["chats_min"]
        out["boxh_per_100k"] = 100000 / (rate * 60) if rate else None
    c = last_counter(os.path.join(T, "gen.log"))
    ok = max(c.get("ok", 0), 1)
    out["resample_per_chat"] = {k: round(v / ok, 2) for k, v in c.items() if k not in ("ok", "tool_calls") and v}
    out["fail_per_chat"] = round((c.get("bad", 0) + c.get("err", 0)) / ok, 3)
    q = [d for d in D if d["bk"]["quality"]]
    out["quality_chats"] = len(q)
    talk = [d for d in q if d["type"] in ("talk", "home")]
    if talk:
        done = [sum(m["role"] == "user" for m in d["messages"]) / max(d["turns"], 1) for d in talk]
        out["turns_done_frac"] = sum(done) / len(done)
        out["reached_planned"] = sum(x >= 1 for x in done) / len(done)
    rs = os.path.join(T, "replay", "replay_stats.json")
    if os.path.exists(rs):
        r = json.load(open(rs))
        out["replay_s"], out["replay_chats"], out["replay_errors"] = r.get("s"), r.get("chats"), len(r.get("errors", []))
    return out


def as_eval(d: dict) -> dict:
    from moth.bakeoff import to_turns
    return {"cat": "agent" if d["type"] == "agent" else d["type"], "name": str(d["seed"]), "seed": d["seed"],
            "turns": to_turns(d["messages"]), "card": d.get("card"), "card_id": (d.get("card") or {}).get("id")}


def kappa(a: list[bool], b: list[bool]) -> float | None:
    n = len(a)
    if not n:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else None


def pct(x, d=1):
    return "-" if x is None else f"{100 * x:.{d}f}%"


def num(x, d=0):
    return "-" if x is None else f"{x:.{d}f}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("bdir")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    try:
        import moth.gen_dialogs  # noqa: F401  torch/tokenizers stubs for build_sft when they are missing
    except ImportError:   # no openai here: build_sft then needs the real torch/tokenizers
        pass
    from moth.bakeoff import load_label_file
    from moth.build_sft import BANNED, keep
    from moth.metrics import ADVICE, compute
    out = a.out or os.path.dirname(a.bdir.rstrip("/"))
    os.makedirs(out, exist_ok=True)
    tdirs = sorted(d for d in glob.glob(os.path.join(a.bdir, "*")) if os.path.exists(os.path.join(d, "DONE")))
    tags = [os.path.basename(d) for d in tdirs]
    judges = sorted(os.path.basename(p)[len("labels_"):-len(".jsonl")] for p in glob.glob(os.path.join(a.bdir, "labels_*.jsonl")))
    L = {j: load_label_file(os.path.join(a.bdir, f"labels_{j}.jsonl")) for j in judges}
    R = ["# Teacher bake-off", "", f"teachers: {', '.join(tags)}; judges: {', '.join(judges)}", ""]

    # speed
    R += ["## Speed (gen pass on the box)", "",
          "| teacher | model | seqs/conc | download s | load s | chats/min (pass) | chats/min (steady) | gen tok/s | prompt tok/s | prefix hit | box-h / 100k chats | resamples / chat | failed / chat | talk turns done | replay s |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    SP = {}
    for t, T in zip(tags, tdirs):
        s = SP[t] = speed(T)
        res = ", ".join(f"{k} {v}" for k, v in sorted(s["resample_per_chat"].items(), key=lambda kv: -kv[1])[:4])
        R.append(f"| {t} | {s['model']} | {s['max_num_seqs']}/{s['concurrency']} | {s['download_s']} | {s['load_s']} | "
                 f"{num(s.get('chats_min'), 1)} | {num(s.get('chats_min_steady'), 1)} | {num(s.get('gen_tok_s'))} | "
                 f"{num(s.get('prompt_tok_s'))} | {pct(s.get('prefix_hit'), 0)} | {num(s.get('boxh_per_100k'), 1)} | {res} | "
                 f"{s['fail_per_chat']} | {pct(s.get('turns_done_frac'), 0)} | {s.get('replay_s')} |")

    # quality
    Q = collections.defaultdict(dict)
    dialogs = {}
    replay = {}
    for t, T in zip(tags, tdirs):
        dialogs[t] = [d for d in map(json.loads, open(os.path.join(T, "dialogs.jsonl"))) if d["bk"]["quality"]]
        p = os.path.join(T, "replay", "results.json")
        replay[t] = json.load(open(p)) if os.path.exists(p) else []
        gen_eval = [as_eval(d) for d in dialogs[t]]
        for src, recs in (("gen", gen_eval), ("replay", replay[t])):
            M = compute([dict(r) for r in recs])
            texts = [e["say"] for r in recs for tt in r["turns"] for e in tt["events"] if "say" in e]
            asst = [x for x in texts if any(b in x.lower().replace("’", "'") for b in BANNED) or ADVICE.search(x.lower())]
            Q[t][src] = {"replies": len(texts), "assistant": len(asst) / len(texts) if texts else None,
                         "repeat": M["repeat in chat (near)"][0], "crutch": M["identity crutch"][0],
                         "explain": M["agent: explain/advice"][0], "words": (sum(len(x.split()) for x in texts) / len(texts))
                         if texts else None, "question": M["question back"][0], "distinct2": M["distinct-2"][0]}
            for j in judges:
                ids = [(t, src, str(d["seed"] if src == "gen" else f"{d['card_id']}:{d['name']}")) for d in
                       (dialogs[t] if src == "gen" else replay[t])]
                labs = [L[j].get(k) for k in ids]
                flat = [lb for x in labs if x for lb in x]
                Q[t][src][j] = {"coverage": sum(x is not None for x in labs) / len(labs) if labs else None,
                                "on": sum(lb["on"] for lb in flat) / len(flat) if flat else None,
                                "self": sum(lb["self"] for lb in flat) / len(flat) if flat else None}
                if src == "gen":
                    why = collections.Counter()
                    kept_r = tot_r = kept_c = 0
                    for d, lb in zip(dialogs[t], labs):
                        n = sum(bool(m.get("content")) for m in d["messages"] if m["role"] == "assistant")
                        tot_r += n
                        k = keep(d, why, lb) if lb is not None else None
                        if k:
                            kept_c += 1
                            kept_r += sum(bool(m.get("content")) for m in k if m["role"] == "assistant")
                    Q[t][src][j]["keep_replies"] = kept_r / tot_r if tot_r else None
                    Q[t][src][j]["keep_chats"] = kept_c / len(dialogs[t]) if dialogs[t] else None
                    Q[t][src][j]["cut_reasons"] = dict(why.most_common(5))

    for src, title in (("gen", "own chats (150 quality seeds; the teacher also plays the human)"),
                       ("replay", "scripted replay (moth.probes: identical user lines for every teacher)")):
        R += ["", f"## Quality: {title}", ""]
        hdr = "| teacher | replies | words/reply | " + " | ".join(f"on_gist {j} | self {j}" for j in judges)
        hdr += (" | " + " | ".join(f"keep replies {j} | keep chats {j}" for j in judges)) if src == "gen" else ""
        hdr += " | assistant-speak | repeat (near) | identity crutch | agent explain | asks back | label coverage |"
        R += [hdr, "|" + "---|" * (hdr.count("|") - 1)]
        for t in tags:
            q = Q[t][src]
            row = f"| {t} | {q['replies']} | {num(q['words'], 1)} | " + " | ".join(
                f"{pct(q[j]['on'])} | {pct(q[j]['self'], 2)}" for j in judges)
            if src == "gen":
                row += " | " + " | ".join(f"{pct(q[j]['keep_replies'], 0)} | {pct(q[j]['keep_chats'], 0)}" for j in judges)
            row += (f" | {pct(q['assistant'])} | {pct(q['repeat'])} | {pct(q['crutch'])} | {pct(q['explain'])} | "
                    f"{pct(q['question'], 0)} | " + "/".join(pct(q[j]['coverage'], 0) for j in judges) + " |")
            R.append(row)
        if src == "gen":
            R += ["", "build_sft cut reasons (top 5, per judge's labels):"]
            for t in tags:
                R.append(f"- {t}: " + "; ".join(f"{j}: {Q[t]['gen'][j]['cut_reasons']}" for j in judges))

    LANG = {(t, "gen", str(d["seed"])): (d.get("lang") or (d.get("card") or {}).get("lang") or "en")
            for t in tags for d in dialogs[t]}

    # judge agreement: every pair of judges, per reply, both sources
    def aligned(ja, jb, t="ALL", lang=None):
        A, Bv = [], []
        for k, la in L[ja].items():
            if (t != "ALL" and k[0] != t) or k not in L[jb] or len(L[jb][k]) != len(la):
                continue
            if lang is not None and LANG.get(k, "en") != lang:
                continue
            A += la
            Bv += L[jb][k]
        return A, Bv
    if len(judges) >= 2:
        R += ["", "## Judge agreement (per reply, all teachers, both sources)", "",
              "| judge A | judge B | replies | on agree | on kappa | self agree | self kappa | on: A only / B only | self: A only / B only |",
              "|---|---|---|---|---|---|---|---|---|"]
        for x in range(len(judges)):
            for y in range(x + 1, len(judges)):
                A, Bv = aligned(judges[x], judges[y])
                n = len(A)
                if not n:
                    continue
                ao, bo = [u["on"] for u in A], [u["on"] for u in Bv]
                asf, bsf = [u["self"] for u in A], [u["self"] for u in Bv]
                R.append(f"| {judges[x]} | {judges[y]} | {n} | {pct(sum(p == q for p, q in zip(ao, bo)) / n)} | "
                         f"{num(kappa(ao, bo), 2)} | {pct(sum(p == q for p, q in zip(asf, bsf)) / n)} | {num(kappa(asf, bsf), 2)} | "
                         f"{sum(p and not q for p, q in zip(ao, bo))}/{sum(q and not p for p, q in zip(ao, bo))} | "
                         f"{sum(p and not q for p, q in zip(asf, bsf))}/{sum(q and not p for p, q in zip(asf, bsf))} |")

    # decision judges: calibration of p_on / p_self against each generative judge's labels
    deciders = [j for j in judges if any("p_on" in (lb[0] if lb else {}) for lb in L[j].values())]
    gens = [j for j in judges if j not in deciders]
    if deciders:
        R += ["", "## Decision judges: calibration against the generative labels", "",
              "Brier score and expected calibration error (10 bins) of the decision probability, taking a generative judge's "
              "label as the truth; AUC = how well the probability ranks that judge's yes above its no.", "",
              "| decider | reference | field | n | base rate (ref) | mean p | Brier | ECE | AUC |", "|---|---|---|---|---|---|---|---|---|"]
        for dj in deciders:
            for gj in gens:
                A, Bv = aligned(dj, gj)
                for fld, pk in (("on", "p_on"), ("self", "p_self")):
                    ps = [u[pk] for u in A]
                    ys = [1.0 if v[fld] else 0.0 for v in Bv]
                    if not ps:
                        continue
                    n = len(ps)
                    brier = sum((p - y) ** 2 for p, y in zip(ps, ys)) / n
                    ece = 0.0
                    for b in range(10):
                        idx = [i for i, p in enumerate(ps) if b / 10 <= p < (b + 1) / 10 or (b == 9 and p == 1.0)]
                        if idx:
                            ece += len(idx) / n * abs(sum(ps[i] for i in idx) / len(idx) - sum(ys[i] for i in idx) / len(idx))
                    pos = [p for p, y in zip(ps, ys) if y]
                    neg = [p for p, y in zip(ps, ys) if not y]
                    auc = None
                    if pos and neg:
                        allp = sorted((p, y) for p, y in zip(ps, ys))
                        rank_sum, r = 0.0, 0
                        i = 0
                        while i < len(allp):   # mid-ranks for ties
                            jj = i
                            while jj < len(allp) and allp[jj][0] == allp[i][0]:
                                jj += 1
                            mid = (i + 1 + jj) / 2
                            rank_sum += mid * sum(1 for q in allp[i:jj] if q[1])
                            i = jj
                        auc = (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
                    R.append(f"| {dj} | {gj} | {fld} | {n} | {pct(sum(ys) / n)} | {pct(sum(ps) / n)} | {brier:.3f} | {ece:.3f} | "
                             f"{num(auc, 3)} |")
        R += ["", "Reliability (decider p_on in bins vs the reference's ON rate):", ""]
        for dj in deciders:
            for gj in gens:
                A, Bv = aligned(dj, gj)
                cells = []
                for b in range(10):
                    idx = [i for i, u in enumerate(A) if b / 10 <= u["p_on"] < (b + 1) / 10 or (b == 9 and u["p_on"] == 1.0)]
                    if idx:
                        cells.append(f"{b / 10:.1f}-{(b + 1) / 10:.1f}: n={len(idx)} ref ON {pct(sum(Bv[i]['on'] for i in idx) / len(idx), 0)}")
                R.append(f"- {dj} vs {gj}: " + "; ".join(cells))

    # label speed (generative label pass vs decision judges)
    R += ["", "## Label speed on the box (same chats)", "", "| judge | chats | replies labelled | seconds | reply-labels/s | "
          "reply-labels per box-hour | decisions/s | concurrency |", "|---|---|---|---|---|---|---|---|"]
    for j in judges:
        sp = os.path.join(a.bdir, f"labels_{j}.jsonl.speed.json")
        if os.path.exists(sp):
            x = json.load(open(sp))
            rl = x.get("reply_labels_per_s")
            R.append(f"| {j} | {x.get('chats')} | {x.get('replies', (x.get('decisions') or 0) // 2)} | {x.get('seconds')} | {rl} | "
                     f"{num(rl * 3600 if rl else None)} | {x.get('decisions_per_s', '-')} | {x.get('concurrency')} |")
    for sp in sorted(glob.glob(os.path.join(a.bdir, "bench_*.jsonl.speed.json"))):
        x = json.load(open(sp))
        R.append(f"| {x['judge']} (bench, old chats) | {x.get('chats')} | {(x.get('decisions') or 0) // 2} | {x.get('seconds')} | "
                 f"{x.get('reply_labels_per_s')} | {num((x.get('reply_labels_per_s') or 0) * 3600)} | {x.get('decisions_per_s')} | "
                 f"{x.get('concurrency')} |")
    if any(j.startswith("gev") for j in judges):
        R += ["", "GEV-26B (Gemma-4 base; its card links the Gemma 4 license): eval / agreement only, never a filter or a "
              "reward (project rule)."]

    # per language (own chats; replay is English only)
    R += ["", "## Per language (own chats, quality seeds; every 4th chat is zh/es/fr/hi)", "",
          "| teacher | lang | chats | replies | " + " | ".join(f"on {j} | self {j}" for j in judges) + " |",
          "|" + "---|" * (4 + 2 * len(judges))]
    for t in tags:
        for lg in ("en", "zh", "es", "fr", "hi"):
            ds = [d for d in dialogs[t] if (d.get("lang") or (d.get("card") or {}).get("lang") or "en") == lg]
            if not ds:
                continue
            cells = []
            for j in judges:
                flat = [lb for d in ds for lb in (L[j].get((t, "gen", str(d["seed"]))) or [])]
                cells.append(f"{pct(sum(x['on'] for x in flat) / len(flat)) if flat else '-'} | "
                             f"{pct(sum(x['self'] for x in flat) / len(flat), 2) if flat else '-'}")
            nrep = sum(sum(bool(m.get("content")) for m in d["messages"] if m["role"] == "assistant") for d in ds)
            R.append(f"| {t} | {lg} | {len(ds)} | {nrep} | " + " | ".join(cells) + " |")

    # pairwise
    R += ["", "## Blind pairwise on the scripted replay (each pair judged twice, positions swapped; a win needs both)", "",
          "| judge | A | B | n | A wins | ties | B wins | A win rate (ties half) | sign p |", "|---|---|---|---|---|---|---|---|---|"]
    from moth.pairwise import sign_test
    for j in judges:
        for p in sorted(glob.glob(os.path.join(a.bdir, f"pairwise_{j}", "*.json"))):
            x, y = os.path.basename(p)[:-5].split("__")
            rows = json.load(open(p))
            c = collections.Counter(r["result"] for r in rows)
            n = len(rows)
            R.append(f"| {j} | {x} | {y} | {n} | {c['A']} | {c['tie']} | {c['B']} | {pct((c['A'] + c['tie'] / 2) / n if n else None, 0)} | "
                     f"{sign_test(c['A'], c['B']):.3g} |")
    open(os.path.join(out, "report.md"), "w").write("\n".join(R) + "\n")
    json.dump({"speed": SP, "quality": Q}, open(os.path.join(out, "report.json"), "w"), indent=1, default=str)

    # reading samples + label audit
    audit = ["# Label audit", "", "## self=true where the friend asked about identity or files/home", ""]
    dis = ["", "## judges disagree (sample)", ""]
    from moth.cards import STUDENT_TAIL, student_system
    from moth.gist import ASKED
    allf = open(os.path.join(out, "labelled_replies.jsonl"), "w")   # every labelled reply, for a hand-labelled eval set
    for t in tags:
        rng = random.Random(7)
        items = []
        for d in dialogs[t]:
            k = (t, "gen", str(d["seed"]))
            ri = -1
            last_user = ""
            ctx = []
            for m in d["messages"]:
                if m["role"] == "user":
                    last_user = m["content"]
                if m["role"] == "assistant" and m.get("content"):
                    ri += 1
                    items.append((k, ri, last_user, m["content"], d.get("card") or {}, d["type"], list(ctx[-8:])))
                if m.get("content"):
                    c = m["content"]
                    ctx.append(("[R] " if m["role"] == "assistant" else "[agent] " if c.startswith("[the agent said]")
                                else "friend: ") + " ".join(c.replace("[the agent said]", "").split())[:400])
        for r in replay[t]:
            k = (t, "replay", f"{r['card_id']}:{r['name']}")
            ri = -1
            ctx = []
            for tt in r["turns"]:
                u = tt["user"]
                ctx.append(("[agent] " if u.startswith("[the agent said]") else "friend: ")
                           + " ".join(u.replace("[the agent said]", "").split())[:400])
                for e in tt["events"]:
                    if "say" in e:
                        ri += 1
                        items.append((k, ri, tt["user"], e["say"], r.get("card") or {}, r["cat"], list(ctx[-8:])))
                        ctx.append("[R] " + e["say"])

        def lab(k, ri, j):
            x = L[j].get(k)
            return x[ri] if x and ri < len(x) else None
        for k, ri, u, say, card, typ, ctx in items:
            labs = {j: lab(k, ri, j) for j in judges}
            if any(labs.values()):
                allf.write(json.dumps({"teacher": t, "src": k[1], "chat_id": k[2], "reply": ri + 1, "type": typ,
                                       "card_id": card.get("id"), "lang": LANG.get(k, "en"),
                                       "card_text": student_system(card).replace(STUDENT_TAIL, "").strip(),
                                       "card": card, "context": ctx, "friend": u, "text": say, "labels": labs},
                                      ensure_ascii=False) + "\n")
        S = [f"# {t}: 30 sampled replies (gen + replay), with both judges' labels", ""]
        for k, ri, u, say, card, typ, _ in rng.sample(items, min(30, len(items))):
            ls = "; ".join(f"{j}: {'ON' if (lb := lab(k, ri, j)) and lb['on'] else 'OFF' if lb else '?'}"
                           f"{' SELF' if lb and lb['self'] else ''} ({lb['why'] if lb else ''})" for j in judges)
            S += [f"- [{k[1]} {typ} card={card.get('id')}] **friend:** {u[:300]!r}", f"  **reply:** {say!r}", f"  {ls}", ""]
        open(os.path.join(out, f"read_{t}.md"), "w").write("\n".join(S))
        for k, ri, u, say, card, typ, _ in items:
            labs = {j: lab(k, ri, j) for j in judges}
            selfs = [j for j, lb in labs.items() if lb and lb["self"]]
            if selfs and (ASKED.search(u) or FILES_ASK.search(u)):
                audit.append(f"- {t} {k[1]}:{k[2]} R{ri + 1} [{','.join(selfs)}] friend: {u[:200]!r} -> {say!r} "
                             f"({'; '.join(labs[j]['why'] for j in selfs)})")
            if len(judges) >= 2 and all(labs.values()) and (labs[judges[0]]["on"] != labs[judges[1]]["on"]
                                                            or labs[judges[0]]["self"] != labs[judges[1]]["self"]):
                if rng.random() < 0.08:
                    dis.append(f"- {t} {k[1]} R{ri + 1} friend: {u[:200]!r} -> {say!r} | " + " | ".join(
                        f"{j}: on={labs[j]['on']} self={labs[j]['self']} ({labs[j]['why']})" for j in judges))
    open(os.path.join(out, "label_audit.md"), "w").write("\n".join(audit + dis) + "\n")
    allf.close()
    print("\n".join(R))


if __name__ == "__main__":
    main()
