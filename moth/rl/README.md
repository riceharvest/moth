# moth.rl: multi-turn RL environment for the chat buddy

The buddy is a pure chatbot: no tools, no home folder, no files. An episode is a card (identity from the system
prompt only) plus a scenario (talk, agent reaction, recall, or a scripted probe). A user simulator chats with the policy for N turns. One judge call then labels every reply, and each reply gets
a reward vector plus a weighted scalar. Module map: `__init__.py`. Reward definition: `reward.py`.

```
python -m moth.rl mock --episodes 3                         # the whole loop on CPU, no endpoint
python -m moth.rl.registry add --line "you are a cat named bo"   # grow the card pool; freeze v5 when it has grown
python -m moth.rl eval --version v4 --moth-model runs/x/model --sim-url URL --eval-judge-url EVAL_URL [--judge-url REWARD_URL] [--ref A/results.json]
python -m pytest -q tests/test_rl_env.py
```

## Reward, judges, eval

- **Reward per reply** (`reward.py` has the table): on_gist, self, specific, vs_ref, group_dup, assistant, repeat,
  length, format, card, recall. On-gist alone no longer wins. A stock "aw that's rough" scores
  1 − 1 (not specific) − 1 (loses to the frozen SFT reference) − 1 (all G rollouts say it) = −2. A specific on-gist
  reply that beats the reference scores +2. Unprompted identity (weight 4) is still the worst move.
- **Reference:** `--ref-policy-url` / `run_episode(ref_policy=...)` takes the frozen SFT checkpoint (stateless;
  `TinyAgentPolicy(stateless=True)` on CPU). Its reply on the same context is asked once per context (cached across a
  group). The judge picks per reply with shuffled positions: one extra judge call per chat.
- **Two judge slots, never the same model:**
  - The reward judge (training) is `--judge-kind generative` (Qwen JSON labels) or `forward` (JEV-style P(yes)
    from one forward pass per question).
  - The eval judge (`--eval-judge-url`, e.g. GEV, non-Qwen allowed) gives every headline number. The reward judge
    only appears as an agreement table against it; `run_eval` refuses the same model in both slots.
  - `--gold labels.jsonl` (hand/Claude labels, chat- or reply-level, see `judge.py`) reports each judge's accuracy
    and kappa.
- **Eval size:** `--replies 3000` (the default) runs passes over the frozen set, re-seeding each pass, until there are
  that many replies; a 1% identity rate is then about 30 events. Every rate has a 95% Wilson interval, and the reward
  mean has a normal interval. The simulator runs at temperature 0 by default (training: 1.0).
- **Languages:** cards and specs carry `lang` (core `cards.LANGS`: en/zh/es/fr/hi).
  - The simulator writes in it.
  - The format check forbids capitals only in cased scripts.
  - The assistant phrases are per language.
  - Length counts CJK by characters.
  - The eval report has a slice per language.
- **Frozen sets:**
  - `eval_v4` is current: 216 specs, each held-out card chatting once in every language; the who-are-you probes
    stay English.
  - `eval_v3` is English only.
  - `eval_v1`/`v2` are legacy: frozen under the old tools/home spec, kept untouched, and their home specs are skipped.

## Plugging into tiny-agent GRPO

The trainer is tiny-agent's own `scripts/grpo.py`, with its custom architecture and continuous batching. The env is
plain Python: it only calls `policy.generate(messages)`, and it returns scored episodes whose per-turn extras carry the
policy's token ids and log-probs. There are three pieces.

### 1. A policy that keeps exact tokens

`moth.rl.tiny_agent_bridge.TinyAgentPolicy(model, tok, device, max_turn_tokens, temperature, max_len, seed)` is the
reference implementation. It decodes one sequence and builds a fresh `KVCache` per episode.

- **First turn:** it feeds `render(messages, add_generation_prompt=True)`.
- **Later turns:** it feeds only `"\n" + render([new user message], add_generation_prompt=True)` after the ids it
  generated itself, plus `<|im_end|>` if a turn hit the token limit. Its own text is never re-encoded, so the recorded
  `logp` matches teacher forcing exactly. `tests/rl_tiny_agent_check.py` asserts |Δlogp| < 1e-3 and measures 0.0.
- **Return value:** `generate()` returns `{"text", "raw", "feed_ids", "gen_ids", "logp"}`. `run_episode` stores
  everything except text and raw in `ep["turns"][k]["extra"]`.

For training, put the same feed rule behind the Engine instead of a private cache. The decode loop is the only part
that moves:

- An `EnginePolicy.generate(messages)` call enqueues `(rollout_id, feed_ids)` and blocks on a `Future`.
- One driver thread runs `engine.tick()`. When a slot emits `<|im_end|>` (or reaches `max_turn_tokens`), it resolves
  that future with `{"gen_ids", "logp", "version": engine.version}`.
- The slot keeps its KV and waits in a new state, `"env"` (in place of the task engine's `"tools"` state), until the
  next `feed_ids` for that rollout arrive.
- `start(spec)` takes a slot; the episode ending frees it. Slots therefore hold KV during simulator calls of roughly
  1–3 s. With 256 slots and `workers` ≈ slots, that is fine.

### 2. Rollouts (one call per training step)

```python
from moth.rl.adapters import grpo_rollouts
from moth.rl.judge import JudgeCache, LLMJudge
from moth.rl.llm import CostMeter, OAIClient
from moth.rl.registry import Registry, train_agent_msgs
from moth.rl.sim import LLMSimulator

sim = LLMSimulator(OAIClient(SIM_URL, "Qwen3.6-35B-A3B"), temperature=1.0)
judge = LLMJudge(OAIClient(JUDGE_URL, "Qwen3.8-27B"), JudgeCache("out/rl/judge_cache.jsonl"))
reg, agent = Registry(), train_agent_msgs("moth/agent_pool.json")      # eval agent messages already removed
meter = CostMeter()

groups = grpo_rollouts(
    None, n_groups=a.n_tasks, G=a.G, sim=sim, judge=judge, registry=reg, split="train",
    rng=rng, level="utterance", agent_msgs=agent, meter=meter,
    policy_factory=lambda: EnginePolicy(engine),     # or TinyAgentPolicy(model, tok, ...) on CPU
    workers=a.slots)                                 # rollouts in flight: the Engine sees parallel requests
# -> list[list[ep]]; groups without signal are already dropped (like has_signal); each group's G rollouts share
#    their first simulated message (asked once, seeded by the spec), so reply 1 is a clean in-group comparison.
```

Each episode `ep` has:

- `ep["reward"]`: episode score, the mean of the reply scalars.
- `ep["replies"][j]["vec" | "scalar" | "mask" | "advantage" | "assistant_index"]`.
- `ep["turns"][k]["extra"]`: whatever `generate()` returned beyond text and raw, i.e. the token ids, log-probs and
  version.
- `ep["judge_ok"]`: False if the judge failed. Advantages are then None, so give those tokens weight 0.

### 3. Token weights for `policy_loss`

```python
from types import SimpleNamespace
from moth.rl.tiny_agent_bridge import episode_tokens, token_advantages

items = []
for g in groups:
    gtok = sum(sum(episode_tokens(e)["gen_mask"]) for e in g)
    for e in g:
        t = episode_tokens(e)                 # {"tokens", "gen_mask", "logp", "turn_spans"}  (tiny-agent Episode fields)
        w = token_advantages(e, t)            # per token: turn k's advantage on turn_spans[k], 0 elsewhere
        norm = 1.0 / (len(groups) * max(1, gtok))     # grpo.token_weights' prompt-mean
        ep = SimpleNamespace(tokens=t["tokens"], logp=t["logp"], gen_mask=t["gen_mask"])
        items.append((ep, [x * norm for x in w][1:]))   # weights for predicting tokens[1:]
st = policy_loss(cmodel, items, a.micro, dev, a.max_len, a.clip_lo, a.clip_hi)   # unchanged
```

Differences from the task-GRPO path:

- **Advantages:** `token_weights` multiplies one advantage per episode. Here each assistant turn has its own:
  utterance level, with the in-group baseline taken per reply position.
- **Not used:**
  - `LengthPenalty`: length is a reward component.
  - `signed_rebalance` and segment flags: a pure chatbot makes no tool calls.
  - `base_reward`.
- **Staleness:** keep `roller.version` in the policy's extras and drop groups older than `--max_stale`, as before.
- **Eval every N steps:** `moth.rl.evalrun.run_eval("v4", lambda: TinyAgentPolicy(model, tok), sim0, eval_judge, out, reward_judge=judge)`,
  where `sim0` is an `LLMSimulator` at temperature 0. Report `summary["heldout"]`; pairwise against the previous
  checkpoint goes through `ref=`.
- **Cost:** `meter.per_episode()`, then `python -m moth.rl cost` with the measured tok/s and the box price.

`tests/test_rl_env.py::test_tiny_agent_grpo_group` runs points 1–3 for one group of 4 rollouts. It uses a tiny
random-init `TinyAgentLM` on CPU, built in tiny-agent's Python with the GPU hidden. It checks:

- the token streams start with the rendered prompt;
- each turn span equals the generated ids;
- the weights are the turn advantage on generated tokens and 0 elsewhere;
- teacher forcing reproduces the logp.
