# moth.rl: multi-turn RL environment for the chat buddy

The buddy is a pure chatbot: no tools, no home folder, no files. An episode is a card (identity from the system
prompt only) plus a scenario (talk, agent reaction, recall, or a scripted probe). A user simulator chats with the policy for N turns. One judge call then labels every reply, and each reply gets
a reward vector plus a weighted scalar. Module map: `__init__.py`. Reward definition: `reward.py`.

```
python -m moth.rl mock --episodes 3                         # the whole loop on CPU, no endpoint
python -m moth.rl.registry add --line "you are a cat named bo"   # grow the card pool; freeze v6 when it has grown
python -m moth.rl eval --tier quick --moth-model runs/x/model --sim-url URL --eval-judge-url EVAL_URL [--judge-url REWARD_URL] [--ref A/results.json]
python -m pytest -q tests/test_rl_env.py
```

## Reward, judges, eval

- **Reward per reply** (`reward.py` has the table and the weights):
  - Judge-based: on_gist, self, specific, sycophant, vs_ref.
  - Group-based: group_dup.
  - Deterministic: echo, q_rate, lang, assistant, repeat, length, format, card, recall.
- **On-gist alone never wins.**
  - A stock "aw that's rough" scores −2.
  - The echo-question exploit ("oh, pasta? what kind of pasta?" every turn) is closed four ways:
    - the judge rubric says echoing the friend's word back is NOT specific;
    - the deterministic `echo` component costs −1.5;
    - `q_rate` penalises asking back in more than 40% of replies;
    - the pairwise prompt rewards fun / specific / responsive, not "curious" in itself.
  - Unprompted identity (weight 4) stays the worst move.
  - A reply in the wrong language costs −2 (script check for zh/hi, stopword vote for en/es/fr; English is fine in a
    code-switching chat).
- **Adversarial reply suite** (`adversarial.py`, `python -m moth.rl adversarial --judge-url URL`):
  - 8 fixed contexts (en/es/fr/zh/hi, including an agent message) × good, bland, echo, selfy, wrong_lang, sycophant,
    judge-injection, ultra_short and assistant replies, with the expected ordering (good above everything; mild
    failures above severe ones).
  - **Pilot gate: the real reward judge reproduces it.** `--mock` is the CI version.
- **Judge prompts quote** every reply and message in ⟦ ⟧ with delimiters and braces neutralised. A reply that talks to
  the judge is OFF and loses (the suite's injection variant).
- **Reference (required for training):** `grpo_rollouts(..., ref_policy=... | ref_policy_factory=...)` raises
  without one (`allow_no_ref=True` for plumbing only), and `run` warns.
  - The frozen SFT checkpoint's reply to every context is compared by the judge (shuffled positions).
  - Turn-1 references are computed once per group before the fan-out.
  - The reference is per thread (factory) or locked; `TinyAgentPolicy(stateless=True)` uses a private cache per call.
- **Clean group comparisons:**
  - If one rollout lacks a judge component (a failed pairwise call), that component is dropped for the whole group
    (`reward.harmonize_group`).
  - Rollouts whose labels failed are out of the baseline.
  - `env.run_tree` branches G candidates from the shared prefix at EVERY turn, so every reply position has a clean
    group baseline. It costs one judge call per candidate.
- **Two judges, never the same model family** (`judge.model_family`: served names, `fwd:` prefixes and quant suffixes
  stripped; `--judge-family` / `--eval-judge-family` to state it explicitly):
  - The reward judge (training) is `--judge-kind generative` (Qwen JSON labels) or `forward` (JEV-style P(yes) per
    question, with the chat's language line).
  - The eval judge (`--eval-judge-url`, non-Qwen allowed, e.g. GEV) uses its OWN holistic rubric (`EVAL_PROMPT`:
    responds / unasked identity / stock / flattery / quality 1–5) and gives every headline. The reward judge only
    shows as an agreement table.
- **Gold-gated headline:**
  - The report marks the headline trusted only when the eval judge's kappa vs the gold RANDOM half is ≥ 0.6 on on and
    self (a full-tier gate).
  - `python -m moth.rl gold-sheet` writes 300 random + 300 stratified replies, with a CSV carrying the card text and
    language. Fill `human`, then `--gold`.
  - `gold-sheet --refresh` samples RL-time replies where the reward and eval judges disagree.
- **Eval tiers** (`--tier`), on the frozen `eval_v5`:
  - `quick` (~2k replies, en-heavy) is for checkpoint picking, together with the reward-hacking monitor
    (`--monitor log --step N`, `monitor.py`).
  - `full` needs ≥ 10k replies and ≥ 2k per language. Its gates: on_gist MDE ≤ 3 overall / ≤ 5 per language, the
    gold-agreement gate, and pairwise vs final2 (`--ref-final2`, run with `--legacy-prompt`) and vs SFT (`--ref-sft`)
    with ≥ 1000 pairs each and a win-rate interval above 50%.
- **Eval simulator:**
  - Pass `--eval-sim-url` / `--eval-sim-model` (a different chat model, e.g. non-Qwen) so the policy can't overfit
    the training simulator.
  - It runs at temperature 0.7 with fixed seeds per pass, so passes differ but are reproducible.
- **Statistics** (`stats.py`):
  - Clusters are identities (the base card) where a slice has ≥ 20 of them, else chats.
  - Every rate has a design-effect-adjusted Wilson interval, its deff and its MDE.
  - One extra row counts judge failures as OFF.
  - `--bootstrap B` adds a cluster bootstrap; `python -m moth.rl plan` prints the MDE table for planned sizes.
- **Report rows:**
  - **HEADLINE = all held-out identities**, balanced per language. moth is one of them, nothing special: it is never
    trained on, and the registry refuses training cards named or kind moth.
  - Then ALL (+ the blank card), scenario (agent reactions across all identities), card group and language.
- **Leakage controls:**
  - No name or kind on both sides of the split, in either direction: a training-hashed card colliding with a
    held-out one goes held-out; a held-out card colliding with a training card is quarantined (`Registry.leaks()`).
  - User behaviour is held out too: 20% of personas, goals and topics (`registry.behaviour`) only ever play eval
    chats.
  - Agent messages near-duplicating a frozen eval message (Jaccard ≥ 0.5 on word 3-shingles) never train.
  - The SFT data (`gen_dialogs`) already used every persona; only RL is kept off the held-out ones.
- **Simulated users** (`registry.STYLES`, `sim.STYLE_NOTES`):
  - long chats (14–20 turns), pasted text blocks;
  - code-switching (Hinglish, Spanglish, franglais), abusive users, distressed users, terse lowercase users;
  - plus the persona pool.
- **Frozen sets:**
  - `eval_v5` is current: 797 specs. Every one of the 132 held-out identities (moth included; 2 quarantined for
    leaks) chats once per language, half of them in one of 22 held-out phrasings, with held-out user behaviour, plus
    one English probe. The blank card has 1 chat per language.
  - `eval_v4` / `v3` are the earlier small sets; `eval_v1`/`v2` are legacy.
  - `eval_v5` was refrozen (replaced) before its first use, to apply these leakage fixes.

## KL anchor to the SFT model (proposal for tiny-agent `scripts/grpo.py`, not applied here)

`grpo.py` has no KL term. The vs_ref reward pulls replies towards being better than the SFT reference, but nothing
stops the policy drifting into a degenerate style that the judge happens to like. The proposal adds a per-token KL to
the frozen SFT model on the generated tokens. It uses the k3 estimator, exp(d) − d − 1 with d = log π_sft − log π,
which is unbiased and ≥ 0. Mean KL per generated token is logged; start with `--kl_coef 0.02` and watch `kl` (worry
above ~0.5 nats/token).

```diff
@@ def policy_loss(model, items, micro, device, T, clip_lo, clip_hi):
-def policy_loss(model, items, micro, device, T, clip_lo, clip_hi):
+def policy_loss(model, items, micro, device, T, clip_lo, clip_hi, ref_model=None, kl_coef=0.0):
     items = sorted(items, key=lambda it: len(it[0].tokens), reverse=True)
     st = defaultdict(float)
+    st_gen_total = sum(sum(e.gen_mask) for e, _ in items)
@@
         logp = -F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), tgt.reshape(-1).clamp_min(0),
                                 reduction="none").view_as(w)
+        loss_kl = 0.0
+        if ref_model is not None and kl_coef > 0:
+            with torch.no_grad(), torch.autocast("xpu", dtype=torch.bfloat16):
+                rlogits = ref_model(x, doc, make_block_mask(doc, ref_model.cfg.swa_window))
+            ref_lp = -F.cross_entropy(rlogits.reshape(-1, rlogits.size(-1)).float(), tgt.reshape(-1).clamp_min(0),
+                                      reduction="none").view_as(w)
+            d = (ref_lp - logp) * g                  # log(pi_sft / pi) on generated tokens only
+            kl = (torch.exp(d) - d - 1) * g          # k3 estimator: unbiased, >= 0
+            loss_kl = kl_coef * kl.sum() / max(1.0, st_gen_total)   # same prompt-mean scale as the PG term
+            st["kl"] += kl.sum().item()
         diff = (logp.detach() - mu) * g
         ratio = torch.exp(diff.clamp(-20, 20))
         keep = ((ratio >= clip_lo) & (ratio <= clip_hi)).float()
-        loss = -(w * ratio * keep * logp).sum()
+        loss = -(w * ratio * keep * logp).sum() + loss_kl
         loss.backward()
@@
-    return {"loss": round(st["loss"], 6), "mismatch": round(st["abs_logratio"] / gen, 5),
+    return {"loss": round(st["loss"], 6), "kl": round(st["kl"] / gen, 5), "mismatch": round(st["abs_logratio"] / gen, 5),
             "clip_frac": round(st["clipped"] / gen, 5)}
@@ def main():
+    ap.add_argument("--kl_coef", type=float, default=0.02, help="KL(pi || pi_sft) per generated token; 0 = off")
+    ap.add_argument("--ref_ckpt", default=None, help="frozen SFT checkpoint for the KL anchor (default: --ckpt)")
@@
     model = load_model(a.ckpt)
+    ref_model = None
+    if a.kl_coef > 0:   # the SFT weights, frozen (one more forward per micro-batch, no backward)
+        ref_model = load_model(a.ref_ckpt or a.ckpt).eval()
+        ref_model.requires_grad_(False)
@@
-        st = policy_loss(cmodel, items, a.micro, dev, a.max_len, a.clip_lo, a.clip_hi)
+        st = policy_loss(cmodel, items, a.micro, dev, a.max_len, a.clip_lo, a.clip_hi, ref_model, a.kl_coef)
```

`st_gen_total` (the batch's generated-token count, computed once before the loop) keeps the KL on the same scale in
every micro-batch. The same `ref_model` can serve as the env's reference policy
on the trainer side: `TinyAgentPolicy(ref_model, tok, stateless=True)` gives vs_ref with no extra endpoint.

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
    ref_policy_factory=lambda: TinyAgentPolicy(ref_model, tok, stateless=True),   # REQUIRED: the frozen SFT model
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
- **Eval every N steps:** `s = moth.rl.evalrun.run_eval("v5", lambda: TinyAgentPolicy(model, tok), sim0, eval_judge,
  out, reward_judge=judge, tier="quick", refs={"final2": F2})`, then `HackMonitor(log).update(step, s)`,
  where `sim0` is an `LLMSimulator` at temperature 0. Report `summary["heldout"]`; pairwise against the previous
  checkpoint goes through `ref=`.
- **Cost:** `meter.per_episode()`, then `python -m moth.rl cost` with the measured tok/s and the box price.

`tests/test_rl_env.py::test_tiny_agent_grpo_group` runs points 1–3 for one group of 4 rollouts. It uses a tiny
random-init `TinyAgentLM` on CPU, built in tiny-agent's Python with the GPU hidden. It checks:

- the token streams start with the rendered prompt;
- each turn span equals the generated ids;
- the weights are the turn advantage on generated tokens and 0 elsewhere;
- teacher forcing reproduces the logp.
