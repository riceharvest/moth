"""Thin adapters from the framework-agnostic core (moth.rl.env) to RL trainers. No trainer is a dependency.

1. MothMultiTurnEnv: shaped like verifiers' MultiTurnEnv (dataset rows with "prompt"/"info", env_response,
   is_completed, a rubric of reward funcs + weights). Pure Python and synchronous; make_verifiers_env() wraps it in a
   real verifiers.MultiTurnEnv subclass when `verifiers` is importable. That wrapper is UNTESTED here (verifiers is not
   installed and its signatures change between versions): check env_response / is_completed / Rubric against the
   installed version before the first run.

2. grpo_rollouts(): a plain function for a GRPO loop such as tiny-agent's scripts/grpo.py: n_groups specs x G
   rollouts, scored, with per-reply advantages, and turn_advantages(ep) aligned with the episode's assistant turns.
   tiny-agent's token_weights() weights a whole episode by one advantage; with this env the trainer's policy wrapper
   returns {"text", "tokens", "logp", ...} from generate() (kept verbatim in ep["turns"][k]["extra"]) and weights the
   tokens of assistant turn k by turn_advantages(ep)[k] instead (utterance-level credit), None = weight 0.
"""
from __future__ import annotations

import random

from moth.agent import AGENT_TAG
from moth.rl import reward as R
from moth.rl.env import _plan, group_advantages, rollout_group, score_episode
from moth.rl.judge import Judge
from moth.rl.llm import CostMeter
from moth.rl.registry import Registry, sample_spec
from moth.rl.sim import Simulator


class RubricShape:
    """verifiers.Rubric-shaped: funcs(prompt, completion, answer, state, **kw) -> float, and weights."""

    def __init__(self, funcs, weights):
        self.funcs, self.weights = list(funcs), list(weights)

    def score(self, prompt, completion, answer, state, **kw) -> dict:
        vals = {f.__name__: f(prompt, completion, answer, state, **kw) for f in self.funcs}
        vals["reward"] = sum(w * v for w, v in zip(self.weights, vals.values()))
        return vals


def _turns_from_messages(messages: list[dict]) -> list[dict]:
    turns = []
    for m in messages:
        if m["role"] == "user":
            turns.append({"user": m["content"], "events": []})
        elif m["role"] == "assistant" and turns and m.get("content") is not None:
            turns[-1]["events"].append({"say": (m["content"] or "").strip()})
    return [t for t in turns if t["events"]]


class MothMultiTurnEnv:
    def __init__(self, sim: Simulator, judge: Judge, registry: Registry | None = None, split: str = "train",
                 weights: dict | None = None, agent_msgs: list[str] | None = None, mix: dict | None = None):
        self.sim, self.judge, self.reg = sim, judge, registry or Registry()
        self.split, self.weights, self.agent_msgs, self.mix = split, weights, agent_msgs, mix
        self.meter = CostMeter()
        names = ["reward"] + list(R.COMPONENTS)
        self.rubric = RubricShape([self._func(n) for n in names], [1.0] + [0.0] * len(R.COMPONENTS))

    # --- data ------------------------------------------------------------------------------------------------------
    def first_user(self, spec: dict) -> str | None:
        kind, text = _plan(spec)[0]
        if kind == "agent":
            return f"{AGENT_TAG}\n{text}"
        if kind == "script":
            return text
        return self.sim.next_user(spec, [], 1, self.meter)

    def dataset(self, n: int, seed: int = 0, specs: list[dict] | None = None) -> list[dict]:
        """Rows {"prompt": [system, first user message], "answer": "", "info": {"spec": spec}} (one sim call each)."""
        rng = random.Random(seed)
        rows = []
        for spec in specs or [sample_spec(self.reg, rng, self.split, self.mix, self.agent_msgs) for _ in range(n)]:
            u = self.first_user(spec)
            if u is None:
                continue
            rows.append({"prompt": [{"role": "system", "content": spec["system"]}, {"role": "user", "content": u}],
                         "answer": "", "info": {"spec": spec}})
        return rows

    # --- the multi-turn protocol ------------------------------------------------------------------------------------
    @staticmethod
    def _spec(state: dict) -> dict:
        return state.get("spec") or state["info"]["spec"]

    def setup_state(self, state: dict, **kw) -> dict:
        state["spec"] = self._spec(state)
        state["done"] = False
        return state

    def is_completed(self, messages: list[dict], state: dict, **kw) -> bool:
        n_user = sum(m["role"] == "user" for m in messages)
        last_is_reply = bool(messages) and messages[-1]["role"] == "assistant"
        return bool(state.get("done")) or (last_is_reply and n_user >= len(_plan(self._spec(state))))

    def env_response(self, messages: list[dict], state: dict, **kw) -> tuple[list[dict], dict]:
        """The next user message (agent message, script line or simulated human) after the policy's reply."""
        spec = self._spec(state)
        plan = _plan(spec)
        n_user = sum(m["role"] == "user" for m in messages)
        if n_user >= len(plan):
            state["done"] = True
            return [], state
        kind, text = plan[n_user]
        if kind == "agent":
            return [{"role": "user", "content": f"{AGENT_TAG}\n{text}"}], state
        if kind == "script":
            return [{"role": "user", "content": text}], state
        i = sum(k != "agent" for k, _ in plan[:n_user + 1])
        u = self.sim.next_user(spec, _turns_from_messages(messages), i, self.meter)
        if u is None:
            state["done"] = True
            return [], state
        return [{"role": "user", "content": u}], state

    # --- the rubric ------------------------------------------------------------------------------------------------
    def score(self, prompt: list[dict], completion: list[dict], state: dict) -> dict:
        """The episode record for a finished rollout (one judge call, cached in state)."""
        if "moth_episode" not in state:
            spec = self._spec(state)
            msgs = [m for m in list(prompt) + list(completion) if m["role"] != "system"]
            state["moth_episode"] = score_episode(spec, _turns_from_messages(msgs), self.judge, self.meter,
                                                  self.weights)
        return state["moth_episode"]

    def _func(self, name: str):
        def f(prompt, completion, answer, state, **kw) -> float:
            ep = self.score(prompt, completion, state)
            v = ep["reward"] if name == "reward" else ep["components"][name]
            return 0.0 if v is None else float(v)
        f.__name__ = name
        return f


def make_verifiers_env(core: MothMultiTurnEnv, n: int, seed: int = 0, max_turns: int = 12, **vf_kwargs):
    """A real verifiers.MultiTurnEnv around `core` (needs `pip install verifiers datasets`). Untested here."""
    import asyncio

    import verifiers as vf  # noqa: F401  optional dependency
    from datasets import Dataset

    class MothVerifiersEnv(vf.MultiTurnEnv):
        async def setup_state(self, state, **kw):
            return core.setup_state(state, **kw)

        async def is_completed(self, messages, state, **kw):
            return core.is_completed(messages, state)

        async def env_response(self, messages, state, **kw):
            return await asyncio.to_thread(core.env_response, messages, state)

    rubric = vf.Rubric(funcs=core.rubric.funcs, weights=core.rubric.weights)
    return MothVerifiersEnv(dataset=Dataset.from_list(core.dataset(n, seed)), rubric=rubric, max_turns=max_turns,
                            **vf_kwargs)


# --- plain GRPO rollouts (tiny-agent style) ---------------------------------------------------------------------
def grpo_rollouts(policy, n_groups: int, G: int, sim: Simulator, judge: Judge, registry: Registry | None = None,
                  split: str = "train", rng: random.Random | None = None, level: str = "utterance",
                  agent_msgs: list[str] | None = None, mix: dict | None = None, meter: CostMeter | None = None,
                  drop_flat: bool = True, **kw) -> list[list[dict]]:
    """n_groups groups of G scored episodes with reply["advantage"] filled (group_advantages). drop_flat skips groups
    whose advantages are all zero/None (no signal), like tiny-agent's has_signal()."""
    reg = registry or Registry()
    rng = rng or random.Random(0)
    groups = []
    for _ in range(n_groups):
        spec = sample_spec(reg, rng, split, mix, agent_msgs)
        g = rollout_group(spec, policy, G, sim, judge, meter, **kw)
        adv = group_advantages(g, level)
        if drop_flat and not any(a for row in adv for a in row if a is not None):
            continue
        groups.append(g)
    return groups


def turn_advantages(ep: dict) -> list[float | None]:
    """One advantage per assistant message of ep["messages"] (mean over its replies; None if all are None)."""
    n = sum(m["role"] == "assistant" for m in ep["messages"])
    acc: list[list[float]] = [[] for _ in range(n)]
    for r in ep["replies"]:
        if r.get("advantage") is not None:
            acc[r["assistant_index"]].append(r["advantage"])
    return [sum(x) / len(x) if x else None for x in acc]
