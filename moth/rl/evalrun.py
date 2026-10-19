"""Eval on a frozen held-out set (pool/eval_<version>.json), scored by a SEPARATE eval judge.

Judges: the eval judge (eval_judge, e.g. a non-Qwen model such as GEV) labels every chat and gives the headline. The
reward judge (the one training optimises, e.g. Qwen3.8-27B or the JEV forward pass) is never the headline: when
given, it labels the same chats and the report shows how well it agrees with the eval judge. The two must be
different models (run_eval raises otherwise). With gold= (a frozen human/Claude-labelled JSONL, see moth.rl.judge)
each judge's agreement with the gold labels is reported too.

Size: the frozen specs are run in passes (pass s > 0 re-seeds every spec: new simulator draws, new sampling) until at
least `replies` replies (default 3000: a 1% identity rate is then ~30 events, not 10). Every rate comes with a 95%
Wilson interval over replies (chats cluster, so read the intervals as a little optimistic); the reward mean with a
normal 95% interval over episodes.

Headline (per reply unless noted), from the eval judge:
  on_gist, self (target <= 1%), specific (not a stock line), follows_card (per who-are-you probe, gist.follows_card),
  assistant (component == -1), repeat (component == -1), reward (mean episode score under the eval judge)
Slices: held-out cards only, scenario, card group, language (en/zh/es/fr/hi; card "lang"). results.json keeps
moth.eval's record shape (moth.pairwise / moth.metrics work on it); pairwise vs a reference run via ref=.
"""
from __future__ import annotations

import collections
import json
import math
import os
import re
import statistics

from moth.rl.env import run_episodes, to_eval_record
from moth.rl.judge import Judge, mock_on
from moth.rl.llm import CostMeter, read_json, write_json, write_text
from moth.rl.registry import POOL_DIR, load_eval
from moth.rl.sim import Simulator


def wilson(k: int, n: int, z: float = 1.96) -> dict | None:
    """Rate in % with its 95% Wilson interval."""
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"pct": round(100 * p, 2), "lo": round(100 * max(0.0, c - h), 2), "hi": round(100 * min(1.0, c + h), 2),
            "k": k, "n": n}


def mean_ci(xs: list[float]) -> dict | None:
    if not xs:
        return None
    m = sum(xs) / len(xs)
    sd = statistics.pstdev(xs) if len(xs) > 1 else 0.0
    h = 1.96 * sd / math.sqrt(len(xs))
    return {"mean": round(m, 4), "lo": round(m - h, 4), "hi": round(m + h, 4), "n": len(xs)}


def card_group(card_id: str) -> str:
    if card_id in ("moth", "blank"):
        return card_id
    return "heldout_phrased" if "~" in card_id else "heldout_template"


def block(es: list[dict]) -> dict:
    labs = [lb for e in es if e["labels"] for lb in e["labels"]]
    reps = [r for e in es for r in e["replies"]]
    fc = [to_eval_record(e)["check"] for e in es if (e.get("check") or [None])[0] == "card"]
    return {"episodes": len(es), "replies": len(reps),
            "on_gist": wilson(sum(x["on"] for x in labs), len(labs)),
            "self": wilson(sum(x["self"] for x in labs), len(labs)),
            "specific": wilson(sum(x.get("specific", True) for x in labs), len(labs)),
            "follows_card": wilson(sum(bool(x) for x in fc), len(fc)),
            "assistant": wilson(sum(r["vec"]["assistant"] == -1 for r in reps), len(reps)),
            "repeat": wilson(sum(r["vec"]["repeat"] == -1 for r in reps), len(reps)),
            "reward": mean_ci([e["reward"] for e in es if e["reward"] is not None]),
            "judge_failed": sum(e["labels"] is None for e in es)}


def summarize(eps: list[dict]) -> dict:
    by = {"scenario": collections.defaultdict(list), "card": collections.defaultdict(list),
          "lang": collections.defaultdict(list)}
    for e in eps:
        by["scenario"][e["scenario"]].append(e)
        by["card"][card_group(e["card_id"])].append(e)
        by["lang"][e.get("lang") or "en"].append(e)
    held = [e for e in eps if card_group(e["card_id"]).startswith("heldout")]
    return {"all": block(eps), "heldout": block(held),
            **{f"by_{k}": {g: block(v) for g, v in sorted(d.items())} for k, d in by.items()}}


KEYS = ("episodes", "replies", "on_gist", "self", "specific", "follows_card", "assistant", "repeat", "reward",
        "judge_failed")


def cell(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, dict) and "pct" in v:
        return f"{v['pct']} [{v['lo']}, {v['hi']}]"
    if isinstance(v, dict) and "mean" in v:
        return f"{v['mean']} [{v['lo']}, {v['hi']}]"
    return str(v)


def report_md(s: dict, title: str) -> str:
    L = [f"# {title}", "", f"headline judge: **{s.get('eval_judge', '?')}** (eval-only); rates in % with 95% "
         "intervals", "", "| group | " + " | ".join(KEYS) + " |", "|---|" + "---|" * len(KEYS)]
    rows = ([("**ALL**", s["all"]), ("**held-out cards only**", s["heldout"])]
            + [(f"{k} {g}", v) for k in ("scenario", "card", "lang") for g, v in s[f"by_{k}"].items()])
    for name, b in rows:
        L.append(f"| {name} | " + " | ".join(cell(b[k]) for k in KEYS) + " |")
    for w in s.get("warnings", []):
        L += ["", f"WARNING: {w}"]
    return "\n".join(L)


def judge_vs_judge(eps: list[dict], other: Judge) -> dict:
    """The reward judge's labels on the eval chats vs the eval judge's: accuracy, kappa, rates (never a headline)."""
    from moth.rl.judge import kappa
    pairs = {"on": [], "self": [], "specific": []}
    failed = 0
    for e in eps:
        if not e["labels"]:
            continue
        labs = other.label(e["card"], [{"user": t["user"], "events": t["events"]} for t in e["turns"]])
        if labs is None:
            failed += 1
            continue
        for a, b in zip(labs, e["labels"]):
            for f in pairs:
                if f in a and f in b:
                    pairs[f].append((bool(a[f]), bool(b[f])))
    out = {"judge": other.model_id, "failed": failed}
    for f, ps in pairs.items():
        a, b = [x for x, _ in ps], [y for _, y in ps]
        out[f] = {"n": len(ps), "agree": wilson(sum(x == y for x, y in ps), len(ps)), "kappa": kappa(a, b),
                  "reward_judge_rate": wilson(sum(a), len(a)), "eval_judge_rate": wilson(sum(b), len(b))}
    return out


def resample(spec: dict, s: int) -> dict:
    """Pass s of a frozen spec: s = 0 as frozen, else re-seeded (the same card, scenario, persona, topic)."""
    if s == 0:
        return {**spec, "sample": 0}
    from moth.probes import stable_seed
    return {**spec, "seed": stable_seed(f"rl-pass:{spec['id']}", s), "sample": s}


def mock_pair_ask(prompt: str) -> str:
    """A deterministic pairwise verdict for mock runs: more replies that pass mock_on and fewer self-descriptions."""
    def score(chat: str) -> int:
        lines = chat.strip().splitlines()
        sc, user = 0, ""
        for ln in lines:
            if ln.startswith(("user: ", "agent: ")):
                user = ("[the agent said]\n" + ln[7:]) if ln.startswith("agent: ") else ln[6:]
            elif ln.startswith("moth: "):
                rep = ln[6:]
                sc += 1 if mock_on(user, rep)[0] else -1
                sc -= 2 if re.search(r"\bi'm just a\b", rep) else 0
        return sc
    m = re.search(r"Chat 1:\n(.*?)\n\nChat 2:\n(.*?)\n\nReturn", prompt, re.S)
    a, b = (score(m.group(1)), score(m.group(2))) if m else (0, 0)
    w = "1" if a > b else "2" if b > a else "tie"
    return json.dumps({"reason": f"mock {a} vs {b}", "winner": w})


def settings(sim: Simulator, judge: Judge) -> dict:
    """What has to match for two eval runs to be comparable (besides the frozen set's sha256)."""
    from moth.rl import judge as J
    from moth.rl import sim as S
    return {"sim_model": sim.model_id, "sim_temperature": sim.temperature, "sim_prompt": S.PROMPT_HASH,
            "eval_judge": judge.model_id, "label_prompt": J.PROMPT_HASH}


SELF_MIN_REPLIES = 3000


def run_eval(version: str, policy_factory, sim: Simulator, eval_judge: Judge, out: str, ref: str | None = None,
             pair_ask=None, workers: int = 1, root: str = POOL_DIR, limit: int | None = None,
             label: str = "policy", replies: int = SELF_MIN_REPLIES, reward_judge: Judge | None = None,
             gold: str | None = None, max_passes: int = 50, ref_policy=None) -> dict:
    """Legacy sets (v1/v2) still contain "home" specs from the tools era: those are skipped (pure chatbot)."""
    if reward_judge is not None and reward_judge.model_id == eval_judge.model_id:
        raise ValueError(f"eval judge and reward judge are the same model ({eval_judge.model_id}): the headline "
                         "must come from a judge the policy was not trained against")
    doc = load_eval(version, root)
    specs = doc["specs"][:limit] if limit else doc["specs"]
    skipped = sum(s["scenario"] == "home" for s in specs)
    specs = [s for s in specs if s["scenario"] != "home"]
    meter = CostMeter()
    eps: list[dict] = []
    passes = 0
    while passes < max_passes and (passes == 0 or sum(len(e["replies"]) for e in eps) < replies):
        eps += run_episodes([resample(sp, passes) for sp in specs], policy_factory, sim, eval_judge, workers,
                            total=meter, ref_policy=ref_policy)
        passes += 1
    recs = [to_eval_record(e, e.get("sample", 0)) for e in eps]
    s = summarize(eps)
    n_rep = s["all"]["replies"]
    warnings = []
    if n_rep < SELF_MIN_REPLIES:
        warnings.append(f"{n_rep} replies < {SELF_MIN_REPLIES}: the 1% identity metric is too noisy to read")
    s.update({"version": version, "sha256": doc["sha256"], "label": label, "n_specs": len(specs), "passes": passes,
              "replies_target": replies, "skipped_home_specs": skipped, "eval_judge": eval_judge.model_id,
              "settings": settings(sim, eval_judge), "cost_per_episode": meter.per_episode(), "warnings": warnings})
    if reward_judge is not None:
        s["reward_judge_vs_eval_judge"] = judge_vs_judge(eps, reward_judge)
    if gold:
        from moth.rl.judge import agreement, load_gold
        g = load_gold(gold)
        s["gold"] = {"path": gold, "chats": len(g), "eval_judge": agreement(eval_judge, g)}
        if reward_judge is not None:
            s["gold"]["reward_judge"] = agreement(reward_judge, g)
    os.makedirs(out, exist_ok=True)
    write_json(os.path.join(out, "results.json"), recs)
    write_json(os.path.join(out, "episodes.json"), eps)
    md = [report_md(s, f"moth RL eval {version} ({doc['sha256'][:12]}): {label}, {passes} pass(es), {n_rep} replies")]
    if reward_judge is not None:
        rj = s["reward_judge_vs_eval_judge"]
        md += ["", f"## reward judge {rj['judge']} vs eval judge (agreement, not a headline)", "",
               "| field | n | agree % | kappa | reward judge rate | eval judge rate |", "|---|---|---|---|---|---|"]
        md += [f"| {f} | {rj[f]['n']} | {cell(rj[f]['agree'])} | {rj[f]['kappa']} | {cell(rj[f]['reward_judge_rate'])} "
               f"| {cell(rj[f]['eval_judge_rate'])} |" for f in ("on", "self", "specific")]
    if gold:
        md += ["", f"## agreement with the gold labels ({s['gold']['chats']} chats)", "",
               "| judge | field | n | acc | kappa |", "|---|---|---|---|---|"]
        for who in ("eval_judge", "reward_judge"):
            a = s["gold"].get(who)
            if a:
                md += [f"| {who} {a['judge']} | {f} | {a[f]['n']} | {a[f]['acc']} | {a[f]['kappa']} |"
                       for f in ("on", "self", "specific")]
    try:
        from moth.metrics import table
        md += ["", "## moth.metrics (deterministic, same records)", "", table({label: recs})]
    except Exception as e:  # noqa: BLE001  metrics' bible lines need persona; never fail the eval over it
        md += ["", f"(moth.metrics table skipped: {type(e).__name__}: {e})"]
    if ref:
        from moth import pairwise  # imports moth.eval -> build_sft -> torch: only when asked
        rows, warn = pairwise.run(recs, read_json(ref), pair_ask or mock_pair_ask, workers=max(1, workers))
        ref_sum = os.path.join(os.path.dirname(os.path.abspath(ref)), "summary.json")
        if os.path.exists(ref_sum):
            rs = read_json(ref_sum)
            for k in ("version", "sha256"):
                if rs.get(k) != s[k]:
                    warn.insert(0, f"ref {k} {rs.get(k)} != {s[k]}: not the same frozen eval set")
            for k, v in (rs.get("settings") or {}).items():
                if s["settings"].get(k) != v:
                    warn.insert(0, f"ref {k}={v!r} but this run {s['settings'].get(k)!r}: scores not comparable")
        # simulated chats diverge after the first reply (the sim answers each policy), so "user lines differ" is
        # expected there; scripted probes stay paired line for line
        warn = [w for w in warn if "user lines differ" not in w] + (
            [f"{sum('user lines differ' in w for w in warn)} simulated chats diverge after turn 1 (expected)"]
            if any("user lines differ" in w for w in warn) else [])
        for w in warn[:5]:
            print("pairwise warning:", w, flush=True)
        c = collections.Counter(r["result"] for r in rows)
        s["pairwise"] = {"ref": ref, "pairs": len(rows), "A_wins": c["A"], "ties": c["tie"], "B_wins": c["B"],
                         "A_win_rate": round((c["A"] + c["tie"] / 2) / len(rows), 4) if rows else None,
                         "sign_p": pairwise.sign_test(c["A"], c["B"]), "warnings": warn[:5]}
        write_json(os.path.join(out, "pairwise.json"), rows)
        md += ["", pairwise.summarize(rows, label, ref)]
    write_json(os.path.join(out, "summary.json"), s, indent=1)
    write_text(os.path.join(out, "report.md"), "\n".join(md) + "\n")
    return s
