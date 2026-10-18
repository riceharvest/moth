"""Eval on a frozen held-out set (pool/eval_<version>.json): every spec run through the policy, one judge call per
chat, then the headline numbers and, optionally, pairwise against a reference checkpoint's results.

Headline (per reply unless noted):
  on_gist %        judge label "on"                           (episodes whose judge failed are left out)
  self %           judge label "self": unprompted identity    (target <= 1%)
  follows_card %   scripted who-are-you probes (card_who / card_name / card_about): the last reply uses the card and
                   claims nothing else (gist.follows_card); per probe
  assistant %      reward component assistant == -1
  repeat %         reward component repeat == -1 (exact or near repeat of an earlier reply)
  reward           mean episode score
Plus the same split by scenario and by card group (held-out template / held-out phrasing / moth / blank), and
moth.metrics' deterministic table on the same records. results.json uses moth.eval's record shape, so
`python -m moth.pairwise A/results.json B/results.json` and `python -m moth.metrics` work on it too.
Pairwise (ref=results.json of a reference run of the same eval version): moth.pairwise (positions swapped, a win
needs both verdicts) with the judge, or a mock verdict in mock mode.
"""
from __future__ import annotations

import collections
import json
import os
import re

from moth.rl.env import run_episodes, to_eval_record
from moth.rl.judge import Judge, mock_on
from moth.rl.llm import CostMeter, read_json, write_json, write_text
from moth.rl.registry import POOL_DIR, load_eval
from moth.rl.sim import Simulator


def _pct(a, n):
    return None if not n else round(100 * a / n, 2)


def card_group(card_id: str) -> str:
    if card_id in ("moth", "blank"):
        return card_id
    return "heldout_phrased" if "~" in card_id else "heldout_template"


def summarize(eps: list[dict]) -> dict:
    def block(es):
        labs = [lb for e in es if e["labels"] for lb in e["labels"]]
        reps = [r for e in es for r in e["replies"]]
        probes = [e for e in es if (e.get("check") or [None])[0] == "card"]
        fc = [to_eval_record(e)["check"] for e in probes]
        rw = [e["reward"] for e in es if e["reward"] is not None]
        return {"episodes": len(es), "replies": len(reps), "on_gist": _pct(sum(x["on"] for x in labs), len(labs)),
                "self": _pct(sum(x["self"] for x in labs), len(labs)),
                "follows_card": _pct(sum(bool(x) for x in fc), len(fc)), "card_probes": len(fc),
                "assistant": _pct(sum(r["vec"]["assistant"] == -1 for r in reps), len(reps)),
                "repeat": _pct(sum(r["vec"]["repeat"] == -1 for r in reps), len(reps)),
                "reward": round(sum(rw) / len(rw), 4) if rw else None,
                "judge_failed": sum(e["labels"] is None for e in es)}
    by_s, by_c = collections.defaultdict(list), collections.defaultdict(list)
    for e in eps:
        by_s[e["scenario"]].append(e)
        by_c[card_group(e["card_id"])].append(e)
    held = [e for e in eps if card_group(e["card_id"]).startswith("heldout")]
    return {"all": block(eps), "heldout": block(held), "by_scenario": {k: block(v) for k, v in sorted(by_s.items())},
            "by_card": {k: block(v) for k, v in sorted(by_c.items())}}


KEYS = ("episodes", "replies", "on_gist", "self", "follows_card", "assistant", "repeat", "reward", "judge_failed")


def report_md(s: dict, title: str) -> str:
    L = [f"# {title}", "", "| group | " + " | ".join(KEYS) + " |", "|---|" + "---|" * len(KEYS)]
    rows = ([("**ALL**", s["all"]), ("**held-out cards only**", s["heldout"])]
            + [(f"scenario {k}", v) for k, v in s["by_scenario"].items()]
            + [(f"card {k}", v) for k, v in s["by_card"].items()])
    for name, b in rows:
        L.append(f"| {name} | " + " | ".join("n/a" if b[k] is None else str(b[k]) for k in KEYS) + " |")
    return "\n".join(L)


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
            "judge_model": judge.model_id, "label_prompt": J.PROMPT_HASH}


def run_eval(version: str, policy_factory, sim: Simulator, judge: Judge, out: str, ref: str | None = None,
             pair_ask=None, workers: int = 1, root: str = POOL_DIR, limit: int | None = None,
             label: str = "policy") -> dict:
    doc = load_eval(version, root)
    specs = doc["specs"][:limit] if limit else doc["specs"]
    meter = CostMeter()
    eps = run_episodes(specs, policy_factory, sim, judge, workers, total=meter)
    recs = [to_eval_record(e) for e in eps]
    s = summarize(eps)
    s.update({"version": version, "sha256": doc["sha256"], "label": label, "n_specs": len(specs),
              "settings": settings(sim, judge), "cost_per_episode": meter.per_episode()})
    os.makedirs(out, exist_ok=True)
    write_json(os.path.join(out, "results.json"), recs)
    write_json(os.path.join(out, "episodes.json"), eps)
    md = [report_md(s, f"moth RL eval {version} ({doc['sha256'][:12]}): {label}")]
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
