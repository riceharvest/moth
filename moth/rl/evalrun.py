"""Eval on a frozen held-out set (pool/eval_<version>.json), scored by a SEPARATE eval judge, in two tiers.

Tiers (TIERS):
  quick   checkpoint picking during RL: ~2k replies; the moth card, all English specs and a quarter of the rest
          (en-heavy is fine); run every N steps, with the reward-hacking monitor (moth.rl.monitor).
  full    milestones and release: >= 10k replies AND >= 2k replies in every language (passes go on until both hold;
          eval_v5 is balanced per language). Targets: MDE <= 3 points on on_gist overall, <= 5 per language.
          Release gates: pairwise vs final2 and vs the SFT checkpoint, >= 1000 pairs each, win-rate interval above 50%.

Judges: the eval judge (eval_judge; e.g. a non-Qwen model such as GEV) labels every chat and gives every headline.
The reward judge (the one training optimises) is never the headline: when given it labels the same chats and the
report shows its agreement with the eval judge and its own reward on them (the monitor compares the two). run_eval
refuses the same model in both slots. gold= adds each judge's agreement with a hand-labelled set (moth.rl.gold).

Statistics (moth.rl.stats): a chat is the unit. Every rate has a design-effect-adjusted Wilson 95% interval, its
deff and its MDE (smallest difference vs another eval of the same size detectable at 80% power); bootstrap=B adds a
chat-level cluster-bootstrap interval for the headline rows.

Headline (per reply unless noted), from the eval judge: on_gist, self (target <= 1%), specific, follows_card (per
who-are-you probe), assistant, repeat (components == -1), reward (mean episode score). Rows: ALL, held-out cards,
the moth card (the shipped default) and its agent reactions (the headline slice), scenario, card group, language.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import re

from moth.rl import reward as R
from moth.rl.env import run_episodes, to_eval_record
from moth.rl.judge import Judge, mock_on
from moth.rl.llm import CostMeter, read_json, write_json, write_text
from moth.rl.registry import POOL_DIR, load_eval
from moth.rl.sim import Simulator
from moth.rl.stats import cluster_bootstrap, cluster_rate, mean_ci, win_rate

TIERS = {
    "quick": {"replies": 2000, "per_lang": 0, "subset": "quick", "min_pairs": 0},
    "full": {"replies": 10000, "per_lang": 2000, "subset": "all", "min_pairs": 1000},
}
TARGET_MDE = {"all": 3.0, "lang": 5.0}   # on_gist, points (full tier)
SELF_MIN_REPLIES = 3000
RATES = ("on_gist", "self", "specific", "follows_card", "assistant", "repeat")


def wilson(k: int, n: int) -> dict | None:
    """A plain (independent-replies) Wilson interval, as cluster_rate with one reply per chat."""
    return cluster_rate([1] * k + [0] * (n - k), [1] * n) if n else None


def card_group(card_id: str) -> str:
    if card_id in ("moth", "blank") or card_id.startswith("moth~"):
        return "moth" if card_id.startswith("moth") else card_id
    return "heldout_phrased" if "~" in card_id else "heldout_template"


def _per_chat(es: list[dict], rate: str) -> tuple[list[int], list[int]]:
    ks, ns = [], []
    for e in es:
        if rate == "follows_card":
            if (e.get("check") or [None])[0] == "card":
                ks.append(int(bool(to_eval_record(e)["check"])))
                ns.append(1)
            continue
        if rate in ("on_gist", "self", "specific"):
            labs = e["labels"] or []
            f = {"on_gist": "on", "self": "self", "specific": "specific"}[rate]
            ks.append(sum(bool(x.get(f, True if f == "specific" else False)) for x in labs))
            ns.append(len(labs))
        else:
            ks.append(sum(r["vec"][rate] == -1 for r in e["replies"]))
            ns.append(len(e["replies"]))
    return ks, ns


def block(es: list[dict], bootstrap: int = 0) -> dict:
    out = {"episodes": len(es), "replies": sum(len(e["replies"]) for e in es)}
    for rate in RATES:
        ks, ns = _per_chat(es, rate)
        out[rate] = cluster_rate(ks, ns)
        if bootstrap and out[rate]:
            out[rate]["boot"] = cluster_bootstrap(ks, ns, bootstrap)
    out["reward"] = mean_ci([e["reward"] for e in es if e["reward"] is not None])
    out["judge_failed"] = sum(e["labels"] is None for e in es)
    return out


def summarize(eps: list[dict], bootstrap: int = 0) -> dict:
    by = {"scenario": collections.defaultdict(list), "card": collections.defaultdict(list),
          "lang": collections.defaultdict(list)}
    for e in eps:
        by["scenario"][e["scenario"]].append(e)
        by["card"][card_group(e["card_id"])].append(e)
        by["lang"][e.get("lang") or "en"].append(e)
    held = [e for e in eps if card_group(e["card_id"]).startswith("heldout")]
    moth = by["card"].get("moth", [])
    return {"all": block(eps, bootstrap), "heldout": block(held), "moth": block(moth, bootstrap),
            "moth_agent": block([e for e in moth if e["scenario"] == "agent"], bootstrap),
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


def rows_of(s: dict) -> list[tuple[str, dict]]:
    return ([("**ALL**", s["all"]), ("**held-out cards**", s["heldout"]), ("**moth card (shipped)**", s["moth"]),
             ("**moth card, agent reactions**", s["moth_agent"])]
            + [(f"{k} {g}", v) for k, g_v in (("scenario", s["by_scenario"]), ("card", s["by_card"]),
                                              ("lang", s["by_lang"])) for g, v in g_v.items()])


def report_md(s: dict, title: str) -> str:
    L = [f"# {title}", "", f"headline judge: **{s.get('eval_judge', '?')}** (eval-only). Rates in % with design-effect-"
         "adjusted 95% intervals (a chat is the unit).", "", "| group | " + " | ".join(KEYS) + " |",
         "|---|" + "---|" * len(KEYS)]
    for name, b in rows_of(s):
        L.append(f"| {name} | " + " | ".join(cell(b[k]) for k in KEYS) + " |")
    L += ["", "## MDE (points; vs another eval of this size, alpha 0.05, power 0.8) and design effect", "",
          "| group | replies | " + " | ".join(f"{r} MDE (deff)" for r in RATES) + " |", "|---|---|" + "---|" * len(RATES)]
    for name, b in rows_of(s):
        L.append(f"| {name} | {b['replies']} | " + " | ".join(
            f"{b[r]['mde']} ({b[r]['deff']})" if b[r] else "n/a" for r in RATES) + " |")
    if s["all"]["on_gist"] and s["all"]["on_gist"].get("boot"):
        L += ["", "cluster bootstrap (ALL): " + ", ".join(f"{r} {s['all'][r]['boot']}" for r in RATES
                                                         if s["all"][r] and s["all"][r].get("boot"))]
    for g in s.get("gates", []):
        L += ["", f"GATE {g['name']}: {'PASS' if g['pass'] else 'FAIL'} ({g['why']})"]
    for w in s.get("warnings", []):
        L += ["", f"WARNING: {w}"]
    return "\n".join(L)


def judge_vs_judge(eps: list[dict], other: Judge) -> dict:
    """The reward judge on the eval chats: agreement with the eval judge (accuracy, kappa, rates) and its own mean
    reward (what training sees). Never a headline."""
    from moth.rl.judge import kappa
    pairs = {"on": [], "self": [], "specific": []}
    failed, rewards = 0, []
    for e in eps:
        if not e["labels"]:
            continue
        tv = [{"user": t["user"], "events": t["events"]} for t in e["turns"]]
        labs = other.label(e["card"], tv)
        if labs is None:
            failed += 1
            continue
        reps = [{**r, "recall_said": None} for r in e["replies"]]
        sc = R.score_replies(reps, e["card"], labs, None, None, e.get("prefs"), e.get("lang") or "en")
        rewards.append(R.episode_score(sc))
        for a, b in zip(labs, e["labels"]):
            for f in pairs:
                if f in a and f in b:
                    pairs[f].append((bool(a[f]), bool(b[f])))
    out = {"judge": other.model_id, "failed": failed, "reward": mean_ci([x for x in rewards if x is not None])}
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


def tier_specs(specs: list[dict], subset: str) -> list[dict]:
    """quick: the moth card, every English spec and a fixed quarter of the others; all: everything."""
    if subset == "all":
        return specs

    def keep(s):
        if s["card_id"].split("~")[0] == "moth" or (s.get("lang") or "en") == "en":
            return True
        return int(hashlib.sha256(s["id"].encode()).hexdigest()[:4], 16) % 4 == 0
    return [s for s in specs if keep(s)]


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


def pairwise_vs(recs: list[dict], s: dict, name: str, ref: str, pair_ask, workers: int, min_pairs: int) -> tuple:
    """Pairwise vs one reference run (results.json of the same eval version): the win rate with its interval, and
    a release gate (>= min_pairs pairs and the interval's lower end above 50%)."""
    from moth import pairwise  # imports moth.eval -> build_sft -> torch: only when asked
    rows, warn = pairwise.run(recs, read_json(ref), pair_ask or mock_pair_ask, workers=max(1, workers))
    ref_sum = os.path.join(os.path.dirname(os.path.abspath(ref)), "summary.json")
    if os.path.exists(ref_sum):
        rs = read_json(ref_sum)
        for k in ("version", "sha256", "tier"):
            if rs.get(k) != s.get(k):
                warn.insert(0, f"ref {k} {rs.get(k)} != {s.get(k)}: not the same frozen eval set / tier")
        for k, v in (rs.get("settings") or {}).items():
            if s["settings"].get(k) != v:
                warn.insert(0, f"ref {k}={v!r} but this run {s['settings'].get(k)!r}: scores not comparable")
    warn = [w for w in warn if "user lines differ" not in w] + (
        [f"{sum('user lines differ' in w for w in warn)} simulated chats diverge after turn 1 (expected)"]
        if any("user lines differ" in w for w in warn) else [])
    c = collections.Counter(r["result"] for r in rows)
    wr = win_rate(c["A"], c["tie"], c["B"])
    res = {"name": name, "ref": ref, "pairs": len(rows), "A_wins": c["A"], "ties": c["tie"], "B_wins": c["B"],
           "win_rate": wr, "sign_p": pairwise.sign_test(c["A"], c["B"]), "warnings": warn[:5]}
    gate = None
    if min_pairs:
        ok = len(rows) >= min_pairs and wr is not None and wr["lo"] > 50
        gate = {"name": f"pairwise vs {name}", "pass": ok,
                "why": f"{len(rows)} pairs (need {min_pairs}), win rate {cell(wr)} (need the interval above 50%)"}
    return res, rows, gate, pairwise.summarize(rows, s["label"], f"{name} ({ref})")


def run_eval(version: str, policy_factory, sim: Simulator, eval_judge: Judge, out: str, ref: str | None = None,
             pair_ask=None, workers: int = 1, root: str = POOL_DIR, limit: int | None = None,
             label: str = "policy", replies: int | None = None, reward_judge: Judge | None = None,
             gold: str | None = None, max_passes: int = 50, ref_policy=None, tier: str | None = None,
             per_lang: int | None = None, refs: dict[str, str] | None = None, bootstrap: int = 0) -> dict:
    """tier: "quick" / "full" (TIERS) sets replies, per-language minimum, spec subset and the pairwise gates;
    explicit replies / per_lang override it. refs: {"final2": results.json, "sft": results.json, ...} (ref= is one
    unnamed reference). Legacy sets (v1/v2) still contain "home" specs from the tools era: those are skipped."""
    if reward_judge is not None and reward_judge.model_id == eval_judge.model_id:
        raise ValueError(f"eval judge and reward judge are the same model ({eval_judge.model_id}): the headline "
                         "must come from a judge the policy was not trained against")
    t = TIERS.get(tier or "", {"replies": SELF_MIN_REPLIES, "per_lang": 0, "subset": "all", "min_pairs": 0})
    replies = t["replies"] if replies is None else replies
    per_lang = t["per_lang"] if per_lang is None else per_lang
    doc = load_eval(version, root)
    specs = tier_specs(doc["specs"], t["subset"])
    specs = specs[:limit] if limit else specs
    skipped = sum(s["scenario"] == "home" for s in specs)
    specs = [s for s in specs if s["scenario"] != "home"]
    langs = {s.get("lang") or "en" for s in specs}
    meter = CostMeter()
    eps: list[dict] = []
    passes = 0

    def short():
        tot = sum(len(e["replies"]) for e in eps)
        by = collections.Counter()
        for e in eps:
            by[e.get("lang") or "en"] += len(e["replies"])
        return tot < replies or (per_lang and any(by[lg] < per_lang for lg in langs))
    while passes < max_passes and (passes == 0 or short()):
        eps += run_episodes([resample(sp, passes) for sp in specs], policy_factory, sim, eval_judge, workers,
                            total=meter, ref_policy=ref_policy)
        passes += 1
    recs = [to_eval_record(e, e.get("sample", 0)) for e in eps]
    s = summarize(eps, bootstrap)
    n_rep = s["all"]["replies"]
    warnings, gates = [], []
    if n_rep < SELF_MIN_REPLIES:
        warnings.append(f"{n_rep} replies < {SELF_MIN_REPLIES}: the 1% identity metric is too noisy to read")
    if short():
        warnings.append(f"stopped at {max_passes} passes short of the size target ({replies} replies, {per_lang} per "
                        "language)")
    if tier == "full":
        og = s["all"]["on_gist"]
        gates.append({"name": "MDE on_gist overall", "pass": bool(og and og["mde"] <= TARGET_MDE["all"]),
                      "why": f"{og and og['mde']} points (target <= {TARGET_MDE['all']})"})
        worst = max(((lg, b["on_gist"]["mde"]) for lg, b in s["by_lang"].items() if b["on_gist"]),
                    key=lambda x: x[1], default=(None, None))
        gates.append({"name": "MDE on_gist per language", "pass": worst[1] is not None and worst[1] <= TARGET_MDE["lang"],
                      "why": f"worst {worst[0]}: {worst[1]} points (target <= {TARGET_MDE['lang']})"})
    s.update({"version": version, "sha256": doc["sha256"], "tier": tier, "label": label, "n_specs": len(specs),
              "passes": passes, "replies_target": replies, "per_lang_target": per_lang, "skipped_home_specs": skipped,
              "eval_judge": eval_judge.model_id, "settings": settings(sim, eval_judge),
              "cost_per_episode": meter.per_episode(), "warnings": warnings})
    if reward_judge is not None:
        s["reward_judge_vs_eval_judge"] = judge_vs_judge(eps, reward_judge)
    if gold:
        from moth.rl.gold import agreement_by_stratum
        from moth.rl.judge import load_gold
        g = load_gold(gold)
        s["gold"] = {"path": gold, "chats": len(g), "eval_judge": agreement_by_stratum(eval_judge, g)}
        if reward_judge is not None:
            s["gold"]["reward_judge"] = agreement_by_stratum(reward_judge, g)
    os.makedirs(out, exist_ok=True)
    write_json(os.path.join(out, "results.json"), recs)
    write_json(os.path.join(out, "episodes.json"), eps)
    md_tail = []
    if reward_judge is not None:
        rj = s["reward_judge_vs_eval_judge"]
        md_tail += ["", f"## reward judge {rj['judge']} vs eval judge (agreement, not a headline)", "",
                    f"reward judge's mean reward on these chats: {cell(rj['reward'])}", "",
                    "| field | n | agree % | kappa | reward judge rate | eval judge rate |", "|---|---|---|---|---|---|"]
        md_tail += [f"| {f} | {rj[f]['n']} | {cell(rj[f]['agree'])} | {rj[f]['kappa']} | "
                    f"{cell(rj[f]['reward_judge_rate'])} | {cell(rj[f]['eval_judge_rate'])} |" for f in ("on", "self", "specific")]
    if gold:
        md_tail += ["", f"## agreement with the gold labels ({s['gold']['chats']} items)", "",
                    "| judge | stratum | field | n | acc | kappa |", "|---|---|---|---|---|---|"]
        for who in ("eval_judge", "reward_judge"):
            for stratum, a in (s["gold"].get(who) or {}).items():
                md_tail += [f"| {who} {a['judge']} | {stratum} | {f} | {a[f]['n']} | {cell(a[f]['acc'])} | "
                            f"{a[f]['kappa']} |" for f in ("on", "self", "specific")]
    try:
        from moth.metrics import table
        md_tail += ["", "## moth.metrics (deterministic, same records)", "", table({label: recs})]
    except Exception as e:  # noqa: BLE001  metrics' bible lines need persona; never fail the eval over it
        md_tail += ["", f"(moth.metrics table skipped: {type(e).__name__}: {e})"]
    all_refs = dict(refs or {})
    if ref:
        all_refs.setdefault("ref", ref)
    s["pairwise"] = {}
    for name, path in all_refs.items():
        res, rows, gate, md = pairwise_vs(recs, s, name, path, pair_ask, workers, t["min_pairs"])
        for w in res["warnings"]:
            print(f"pairwise vs {name} warning:", w, flush=True)
        s["pairwise"][name] = res
        write_json(os.path.join(out, f"pairwise_{name}.json"), rows)
        md_tail += ["", md]
        if gate:
            gates.append(gate)
    s["gates"] = gates
    title = f"moth RL eval {version} ({doc['sha256'][:12]}) {tier or 'custom'}: {label}, {passes} pass(es), {n_rep} replies"
    write_json(os.path.join(out, "summary.json"), s, indent=1)
    write_text(os.path.join(out, "report.md"), report_md(s, title) + "\n" + "\n".join(md_tail) + "\n")
    return s
