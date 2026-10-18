"""CLI for the RL environment.

  python -m moth.rl mock --episodes 3 [--policy good|selfy|assistant|parrot] [--json out.json]
  python -m moth.rl run  --sim-url URL --judge-url URL (--policy-url URL | --moth-model PATH) --episodes 20 --out DIR
                         [--agent-pool ~/moth/moth/agent_pool.json]   (read-only; the eval's agent messages are skipped)
  python -m moth.rl eval --version v1 (--mock | --policy-url URL | --moth-model PATH) [--sim-url --judge-url]
                         [--ref OTHER/results.json] --out DIR
  python -m moth.rl cost --meter DIR/meter.json --episodes 100000 --tps sim=20000:2500 --tps judge=15000:1500
                         [--box rtx59-7-24-800-ig.1 | --price 0.65]

run/eval write meter.json (calls and tokens per role, per episode). cost turns it into box-hours with throughput you
measured on the box (aggregate tok/s at the pilot's concurrency; prefill:decode) and the box price (runbook/cr.sh).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from moth.rl.env import MockPolicy, MothPolicy, OpenAIPolicy, run_episodes  # noqa: E402
from moth.rl.judge import JUDGE_MODEL, JudgeCache, LLMJudge, MockJudge  # noqa: E402
from moth.rl.llm import CostMeter, OAIClient, box_hours, read_json, write_json  # noqa: E402
from moth.rl.registry import Registry, sample_spec, train_agent_msgs  # noqa: E402
from moth.rl.sim import SIM_MODEL, LLMSimulator, MockSimulator  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def show(ep: dict) -> str:
    L = [f"== {ep['spec_id']}  scenario={ep['scenario']}  reward={ep['reward']}  judge_ok={ep['judge_ok']}",
         f"   system: {ep['system'][:110]}..."]
    reps = iter(ep["replies"])
    for t in ep["turns"]:
        u = t["user"].replace("\n", " ")
        L.append(f"   friend: {u[:110]}")
        for e in t["events"]:
            if "tool" in e:
                L.append(f"     (tool {e['tool']})")
                continue
            r = next(reps)
            v = {k: x for k, x in r["vec"].items() if x is not None}
            L.append(f"   bot: {e['say']}")
            L.append(f"        r={r['scalar']:+.2f} {v}")
    c = ep["cost"]["roles"]
    L.append("   cost: " + ", ".join(f"{k} {v['calls']} calls{' (+%d cached)' % v['cached'] if v['cached'] else ''} "
                                     f"{v['prompt_tokens']}+{v['completion_tokens']} tok{' est' if v['est'] else ''}"
                                     for k, v in c.items()))
    return "\n".join(L)


def build_moth(a):
    from moth.serve import Moth
    ns = argparse.Namespace(model=a.moth_model, tokenizer=os.path.join(ROOT, "data/tiny-agent-112m/tokenizer.json"),
                            pool_file=os.path.join(ROOT, "moth/file_pool.json"), device=a.device, temperature=0.7,
                            top_p=0.95, rep_penalty=1.3, ctx_budget=None, max_turn_tokens=200, max_tool_rounds=3,
                            home=os.path.join(a.out or tempfile.mkdtemp(prefix="moth-rl-"), "home"), seed=0)
    return MothPolicy(Moth(ns), home_root=ns.home)


def components(a, mock: bool):
    cache = JudgeCache(os.path.join(a.out, "judge_cache.jsonl") if a.out else None)
    if mock:
        return MockSimulator(), MockJudge(cache)
    return LLMSimulator(OAIClient(a.sim_url, a.sim_model)), LLMJudge(OAIClient(a.judge_url, a.judge_model), cache)


def policy_factory(a):
    if getattr(a, "mock", False) or a.cmd == "mock":
        return lambda: MockPolicy(a.policy)
    if a.policy_url:   # one per worker thread: run_episode sets policy.meter per episode
        return lambda: OpenAIPolicy(OAIClient(a.policy_url, a.policy_model))
    if a.moth_model:
        if a.workers > 1:
            raise SystemExit("--moth-model runs one chat engine: use --workers 1")
        p = build_moth(a)
        return lambda: p
    raise SystemExit("need --policy-url or --moth-model (or --mock)")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m moth.rl")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, endpoints=True):
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--out", default=None)
        p.add_argument("--workers", type=int, default=1)
        if endpoints:
            p.add_argument("--sim-url", default="http://127.0.0.1:8001/v1")
            p.add_argument("--sim-model", default=None, help=f"served name (plan: {SIM_MODEL}); default /v1/models")
            p.add_argument("--judge-url", default="http://127.0.0.1:8000/v1")
            p.add_argument("--judge-model", default=None, help=f"served name (plan: {JUDGE_MODEL})")
            p.add_argument("--policy-url", default=None)
            p.add_argument("--policy-model", default=None)
            p.add_argument("--moth-model", default=None)
            p.add_argument("--device", default="cpu")
            p.add_argument("--agent-pool", default=os.path.join(ROOT, "moth/agent_pool.json"))
            p.add_argument("--policy", default="good", help="mock policy mode")
    m = sub.add_parser("mock")
    common(m, endpoints=False)
    m.add_argument("--episodes", type=int, default=3)
    m.add_argument("--policy", default="good", choices=["good", "selfy", "assistant", "parrot"])
    m.add_argument("--split", default="train", choices=["train", "heldout"])
    m.add_argument("--json", default=None)
    r = sub.add_parser("run")
    common(r)
    r.add_argument("--episodes", type=int, default=20)
    r.add_argument("--split", default="train", choices=["train", "heldout"])
    r.add_argument("--mock", action="store_true")
    e = sub.add_parser("eval")
    common(e)
    e.add_argument("--version", default="v1")
    e.add_argument("--mock", action="store_true")
    e.add_argument("--ref", default=None)
    e.add_argument("--limit", type=int, default=None)
    e.add_argument("--label", default=None)
    c = sub.add_parser("cost")
    c.add_argument("--meter", required=True)
    c.add_argument("--episodes", type=int, required=True)
    c.add_argument("--tps", action="append", default=[], help="role=prefill:decode (aggregate tok/s on the box)")
    c.add_argument("--box", default=None)
    c.add_argument("--price", type=float, default=None)
    a = ap.parse_args(argv)

    if a.cmd == "cost":
        per = read_json(a.meter)["per_episode"]
        tps = {}
        for x in a.tps:
            role, v = x.split("=")
            pf, dc = v.split(":")
            tps[role] = {"prefill": float(pf), "decode": float(dc)}
        print(json.dumps(box_hours(per, a.episodes, tps, a.price, a.box), indent=1))
        return

    if a.cmd == "eval":
        from moth.rl.evalrun import run_eval
        label = a.label or a.policy_model or a.moth_model or ("mock-" + a.policy)
        a.out = a.out or os.path.join(ROOT, "evals", f"rl_{a.version}_" + re.sub(r"[^\w.-]+", "_", label).strip("_"))
        sim, judge = components(a, a.mock)
        pair_ask = None
        if a.ref and not a.mock:
            cl = OAIClient(a.judge_url, a.judge_model)
            pair_ask = lambda p: cl.chat([{"role": "user", "content": p}], temperature=0, max_tokens=300,  # noqa: E731
                                         chat_template_kwargs={"enable_thinking": False})["text"]
        s = run_eval(a.version, policy_factory(a), sim, judge, a.out, a.ref, pair_ask, a.workers, limit=a.limit,
                     label=label)
        with open(os.path.join(a.out, "report.md"), encoding="utf-8") as f:
            print(f.read())
        write_json(os.path.join(a.out, "meter.json"), {"per_episode": s["cost_per_episode"]}, indent=1)
        return

    mock = a.cmd == "mock" or a.mock
    reg = Registry()
    rng = random.Random(a.seed)
    agent = train_agent_msgs(getattr(a, "agent_pool", None)) or None
    if agent is None and not mock:
        print(f"WARNING: no agent message pool at {a.agent_pool}: agent episodes use the 8 built-in stand-ins. "
              "Pass --agent-pool (e.g. the main tree's moth/agent_pool.json, read-only).", file=sys.stderr, flush=True)
    specs = [sample_spec(reg, rng, a.split, agent_msgs=agent) for _ in range(a.episodes)]
    if a.out:
        os.makedirs(a.out, exist_ok=True)
    sim, judge = components(a, mock)
    total = CostMeter()
    eps = run_episodes(specs, policy_factory(a), sim, judge, a.workers, total=total)
    for ep in eps:
        print(show(ep), flush=True)
    per = total.per_episode()
    print("\nper episode:", json.dumps(per))
    rw = [e["reward"] for e in eps if e["reward"] is not None]
    print(f"mean reward {sum(rw) / max(len(rw), 1):+.3f} over {len(eps)} episodes, "
          f"{sum(len(e['replies']) for e in eps)} replies")
    if a.out:
        write_json(os.path.join(a.out, "episodes.json"), eps)
        write_json(os.path.join(a.out, "meter.json"), {"per_episode": per, "total": total.summary()}, indent=1)
    if getattr(a, "json", None):
        write_json(a.json, eps)
    return eps, total


if __name__ == "__main__":
    main()
