"""One GRPO group of the RL env through a tiny random-init TinyAgentLM on CPU; checks that the token-level view lines
up (run by tests/test_rl_env.py::test_tiny_agent_grpo_group in tiny-agent's Python, with the GPU hidden).

  ONEAPI_DEVICE_SELECTOR=opencl:cpu $TA_PY tests/rl_tiny_agent_check.py "<tiny-agent repo>"
Prints "OK <json stats>" on success; any failed check raises.
"""
import itertools
import json
import os
import random
import sys

MOTH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [sys.argv[1], MOTH]   # the real tiny_agent package first (moth vendors an older subset)

import torch  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from tiny_agent.chat import render  # noqa: E402
from tiny_agent.generate import KVCache, append  # noqa: E402
from tiny_agent.model import ModelConfig, TinyAgentLM  # noqa: E402

from moth import cards as C  # noqa: E402
from moth.rl import adapters, env, registry  # noqa: E402
from moth.rl.judge import MockJudge  # noqa: E402
from moth.rl.sim import MockSimulator  # noqa: E402
from moth.rl.tiny_agent_bridge import TinyAgentPolicy, episode_tokens, token_advantages  # noqa: E402

torch.set_num_threads(2)
torch.manual_seed(0)
tok = Tokenizer.from_file(os.path.join(MOTH, "data", "tiny-agent-112m", "tokenizer.json"))
cfg = ModelConfig(vocab_size=tok.get_vocab_size(), d_model=64, n_layers=2, n_heads=2, head_dim=32, n_kv_heads=1,
                  kv_group=2, swa_window=32, max_seq_len=4096)
model = TinyAgentLM(cfg).eval()
torch.nn.init.normal_(model.head.weight, std=0.05)   # the init zeroes the head: logits would all be equal

seeds = itertools.count(100)
G = 4
spec = registry.make_spec({**C.heldout_cards()[0], "split": "heldout"}, "talk", random.Random(3), 3, turns=3)
group = env.rollout_group(spec, None, G, MockSimulator(), MockJudge(), workers=2,
                          policy_factory=lambda: TinyAgentPolicy(model, tok, max_turn_tokens=10, seed=next(seeds)))
env.group_advantages(group, "utterance")
assert len(group) == G and len({ep["turns"][0]["user"] for ep in group}) == 1   # shared first sim message

nonzero, max_dlp = 0, 0.0
for ep in group:
    t = episode_tokens(ep)
    w = token_advantages(ep, t)
    n = len(t["tokens"])
    assert n == len(t["gen_mask"]) == len(t["logp"]) == len(w)
    assert len(t["turn_spans"]) == len(ep["turns"]) == len(ep["replies"])
    # the sequence starts exactly with the rendered prompt of the first turn
    first = [{"role": "system", "content": spec["system"]}, {"role": "user", "content": ep["turns"][0]["user"]}]
    p0 = tok.encode(render(first, add_generation_prompt=True), add_special_tokens=False).ids
    assert t["tokens"][:len(p0)] == p0
    # weights: the turn's advantage on its generated tokens, 0 elsewhere
    adv = adapters.turn_advantages(ep)
    for k, (s, e) in enumerate(t["turn_spans"]):
        assert all(t["gen_mask"][i] == 1 and w[i] == (adv[k] or 0.0) for i in range(s, e))
        assert t["tokens"][s:e] == ep["turns"][k]["extra"]["gen_ids"]
        assert ep["replies"][k]["advantage"] == adv[k]
    assert all(w[i] == 0.0 for i in range(n) if t["gen_mask"][i] == 0)
    nonzero += sum(1 for x in w if x)
    # teacher forcing over the final sequence reproduces the recorded sampling log-probs (no token got lost or
    # re-encoded: the importance ratio grpo forms starts at 1)
    cache = KVCache(model, 1, 2048, "cpu", dtype=torch.float32)
    pos = 0
    for s, e in t["turn_spans"]:
        logits = append(model, cache, torch.tensor([t["tokens"][pos:s]]), [s - pos])
        for i in range(s, e):
            lp = float(torch.log_softmax(logits.float(), -1)[0, t["tokens"][i]])
            max_dlp = max(max_dlp, abs(lp - t["logp"][i]))
            logits = append(model, cache, torch.tensor([[t["tokens"][i]]]), [1])
        pos = e
assert nonzero > 0, "no token got a non-zero advantage (the group had no signal)"
assert max_dlp < 1e-3, f"recorded logp differs from teacher forcing by {max_dlp}"
print("OK " + json.dumps({"episodes": G, "tokens": [len(episode_tokens(e)["tokens"]) for e in group],
                          "nonzero_weights": nonzero, "max_abs_dlogp": max_dlp,
                          "rewards": [e["reward"] for e in group]}))
