"""Multi-turn RL environment for the chat buddy (framework-agnostic core + thin adapters).

  registry.py  card/phrasing registry that grows (append-only pool/*.ndjson, hash split), frozen eval sets
  sim.py       user simulator (OpenAI-compatible endpoint) + deterministic mock
  judge.py     one call labels every reply of a chat (gist.LABEL_PROMPT), cached; + mock judge
  reward.py    utterance-level reward vector + weighted scalar, episode score
  env.py       episodes, the policy interface (generate(messages) -> reply), MothPolicy, groups + advantages
  adapters.py  verifiers-shaped MultiTurnEnv (optional import) and a plain GRPO rollout function
  evalrun.py   eval on a frozen held-out set (+ pairwise vs a reference)
  llm.py       stdlib OpenAI-compatible client, cost meter, box-hours
  tiny_agent_bridge.py  TinyAgentPolicy (exact tokens + logp per turn), episode -> tiny-agent tokens/weights
  README.md    "plugging into tiny-agent GRPO": the exact call shapes

  python -m moth.rl mock --episodes 3                 # whole loop on CPU, no endpoint
  python -m moth.rl.registry --help                   # add cards / phrasings, freeze an eval version
Dev/research only: not shipped in the wheel (pyproject excludes moth/rl).
"""
