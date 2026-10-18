"""tiny-agent's GRPO on this env: a reference policy over a TinyAgentLM and the token-level view of an episode.

TinyAgentPolicy.generate(messages) keeps the episode's token sequence the way tiny_agent.rollout's Engine and
moth.serve do: the first call encodes render(messages, add_generation_prompt=True); every later call appends only
"\n" + render(<the new user message>, add_generation_prompt=True) after the tokens it generated itself (never a
re-encoding of its own text, so the sampled ids and their log-probs stay exact). Each call returns
{"text", "raw", "feed_ids", "gen_ids", "logp"}; run_episode stores everything but text/raw in turn["extra"].

episode_tokens(ep) concatenates those into tiny-agent's Episode fields (tokens, gen_mask, logp, turn_spans), and
token_advantages(ep) gives one weight per token: the advantage of assistant turn k on the tokens of turn_spans[k]
(adapters.turn_advantages, utterance-level credit), 0 on prompt/user tokens and on turns without an advantage.
grpo.token_weights then needs w[1:] (weights for predicting tokens[1:]) times its prompt-mean norm.

This policy decodes one sequence at a time with a fresh KV cache per episode (fine for tests and a CPU smoke run);
for training, put the same calls behind the Engine's slots (README: "plugging into tiny-agent GRPO").
Needs torch + tiny_agent (the tiny-agent repo on sys.path); imported lazily.
"""
from __future__ import annotations


class TinyAgentPolicy:
    uses_tools = False   # text turns only (the env's tool-less scenarios); see README for tools via the Engine

    def __init__(self, model, tok, device: str = "cpu", max_turn_tokens: int = 64, temperature: float = 1.0,
                 max_len: int = 2048, seed: int | None = None):
        import torch
        self.torch = torch
        self.model, self.tok, self.device = model, tok, device
        self.max_turn_tokens, self.temperature, self.max_len = max_turn_tokens, temperature, max_len
        self.im_end = tok.token_to_id("<|im_end|>")
        self.gen = torch.Generator(device="cpu")
        if seed is not None:
            self.gen.manual_seed(seed)
        self.cache = None
        self.n_msgs = 0          # messages of the episode already fed (incl. its own replies)
        self.ended = True        # the last turn ended with <|im_end|>

    def start(self, spec):
        from tiny_agent.generate import KVCache
        self.cache = KVCache(self.model, 1, self.max_len, self.device, dtype=self.torch.float32)
        self.n_msgs, self.ended = 0, True

    def enc(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    def generate(self, messages):
        import torch
        import torch.nn.functional as F

        from tiny_agent.chat import render
        from tiny_agent.generate import append
        if self.cache is None:
            self.start(None)
        if self.n_msgs == 0:
            feed = self.enc(render(messages, add_generation_prompt=True))
        else:   # skip messages[n_msgs - 1]... : the assistant reply it produced itself is already in the cache
            new = [m for m in messages[self.n_msgs:] if m["role"] != "assistant"]
            feed = ([] if self.ended else [self.im_end]) + self.enc("\n" + render(new, add_generation_prompt=True))
        logits = append(self.model, self.cache, torch.tensor([feed], device=self.device), [len(feed)])
        gen, logp = [], []
        for _ in range(self.max_turn_tokens):
            lg = logits.float() / max(self.temperature, 1e-6)
            p = torch.softmax(lg, -1)
            t = int(torch.multinomial(p.cpu(), 1, generator=self.gen)[0, 0])
            gen.append(t)
            logp.append(float(F.log_softmax(lg, -1)[0, t]))
            if t == self.im_end:
                break
            logits = append(self.model, self.cache, torch.tensor([[t]], device=self.device), [1])
        self.ended = bool(gen) and gen[-1] == self.im_end
        self.n_msgs = len(messages) + 1   # + the reply about to be appended by the env
        text = self.tok.decode([t for t in gen if t != self.im_end], skip_special_tokens=False)
        return {"text": text, "raw": text, "feed_ids": feed, "gen_ids": gen, "logp": logp}


def episode_tokens(ep: dict) -> dict:
    """tiny-agent Episode fields from the per-turn extras: tokens, gen_mask, logp, turn_spans ([start, end) of each
    assistant turn's generated ids, including its <|im_end|>)."""
    tokens, mask, logp, spans = [], [], [], []
    for t in ep["turns"]:
        x = t["extra"]
        tokens += x["feed_ids"]
        mask += [0] * len(x["feed_ids"])
        logp += [0.0] * len(x["feed_ids"])
        s = len(tokens)
        tokens += x["gen_ids"]
        mask += [1] * len(x["gen_ids"])
        logp += list(x["logp"])
        spans.append((s, len(tokens)))
    return {"tokens": tokens, "gen_mask": mask, "logp": logp, "turn_spans": spans}


def token_advantages(ep: dict, toks: dict | None = None) -> list[float]:
    """One weight per token of episode_tokens(ep): turn k's advantage on turn_spans[k], else 0. Needs
    reply["advantage"] (env.group_advantages) and one assistant message per turn (text-only policy)."""
    from moth.rl.adapters import turn_advantages
    toks = toks or episode_tokens(ep)
    adv = turn_advantages(ep)
    if len(adv) != len(toks["turn_spans"]):
        raise ValueError(f"{len(adv)} assistant turns vs {len(toks['turn_spans'])} generated spans (tools?)")
    w = [0.0] * len(toks["tokens"])
    for (s, e), a in zip(toks["turn_spans"], adv):
        for i in range(s, e):
            w[i] = a or 0.0
    return w
