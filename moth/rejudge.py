"""Re-score saved chats with the judge, without regenerating them (e.g. after a teacher/judge change).

  python -m moth.rejudge evals/v2/results.json --out evals/v2_lfm --judge-url http://127.0.0.1:8000/v1
  python -m moth.rejudge data_in/a.jsonl:20:talk out/agentmix/dialogs.jsonl:15:agent --out evals/teacher_lfm ...

An input is an eval results.json (its chats, old judge scores dropped) or teacher dialogs as path:N[:type], which
samples N chats that pass build_sft's filter (lowercased, as trained on) and turns them into eval records (cat =
chat type): a ceiling for the student's scores. Writes report.md / transcripts.md / results.json to --out.
Scores with moth.eval's current judge prompt (v2: strict character, grounding = moth's own home only, back_and_forth)
and adds the deterministic metrics (moth.metrics). To compare two checkpoints, prefer python -m moth.pairwise.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from moth.eval import judge_all, report  # noqa: E402


def dialog_record(d: dict, msgs: list[dict]) -> dict:
    turns, calls = [], []
    for m in msgs:
        if m["role"] == "user":
            turns.append({"user": m["content"], "events": [], "s": 0.0})
        elif m["role"] == "assistant":
            calls = m.get("tool_calls") or []
            if m.get("content"):
                turns[-1]["events"].append({"say": m["content"]})
        elif m["role"] == "tool":
            turns[-1]["events"] += [{"tool": c["name"], "args": c["arguments"], "result": res}
                                    for c, res in zip(calls, m["results"])]
    ev = [e for t in turns for e in t["events"]]
    return {"cat": d["type"], "name": f"teacher_{d['type']}", "seed": d["seed"], "turns": turns, "flags": {},
            "check": None, "tools": sum("tool" in e for e in ev), "n_replies": sum("say" in e for e in ev)}


def from_dialogs(spec: str, rng: random.Random) -> list[dict]:
    from moth.build_sft import keep
    path, n, typ = (spec.split(":") + [""])[:3]
    out, why = [], collections.Counter()
    for line in open(path):
        d = json.loads(line)
        if typ and d.get("type") != typ:
            continue
        msgs = keep(d, why)
        if msgs:
            out.append(dialog_record(d, msgs))
    return rng.sample(out, min(int(n), len(out)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="results.json, or teacher dialogs as path.jsonl:N[:type]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--judge-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--judge-think", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--label", default=None)
    a = ap.parse_args()
    rng = random.Random(0)
    results = []
    for src in a.inputs:
        if src.endswith(".json"):
            assert os.path.abspath(os.path.dirname(src)) != os.path.abspath(a.out), "--out would overwrite the input"
            rs = json.load(open(src))
            for r in rs:
                r.pop("judge", None)
            results += rs
        else:
            results += from_dialogs(src, rng)
    os.makedirs(a.out, exist_ok=True)
    judge_all(results, a.judge_url, {"auto": None, "on": True, "off": False}[a.judge_think])
    print(report(results, a.out, a.label or "rejudge " + " ".join(a.inputs)))


if __name__ == "__main__":
    main()
