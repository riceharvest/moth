"""Token generation for XS with the interface moth.serve uses (the same as tiny_agent.generate's for the old model):

  append(model, cache, ids)        feed tokens, -> logits (1, V) after the last one
  Decoder(model, cache).generate(logits, max_new, stop, temperature, top_p, pen, pen_ids, ban_ids, generator,
                                 logits_hook) -> sampled ids (stop included when sampled; a turn cut off at max_new
                                 gets stop appended to the cache so it stays well-formed)

Sampling: plain softmax at the temperature (what XS's RL trains with) unless top_p < 1 or a repetition penalty asks for
moth's general sampler. logits_hook is moth.norepeat's phrase ban (a hard filter, host side). Eager: one model.step
per token (a CPU has nothing to gain from graphs; the GPU path runs the same eager loop).
"""
from __future__ import annotations

import torch

from moth.xs.model import XSCache, XSModel


def sample_top_p(logits, temperature, top_p, pen=1.0, pen_mask=None, ban_mask=None, generator=None):
    """XS's default sampler is the one its RL trains with: plain softmax at the temperature (no top-p, no repetition
    penalty; moth.json "sampling" sets them, default top_p 1.0 / rep_penalty 1.0), plus the ban mask (specials). With
    a top_p < 1 or a penalty it is moth's general sampler (tiny_agent.generate.sample_top_p)."""
    if top_p < 1.0 or (pen != 1.0 and pen_mask is not None):
        from tiny_agent.generate import sample_top_p as f
        return f(logits, temperature, top_p, pen, pen_mask, ban_mask, generator)
    lg = logits.float()
    if ban_mask is not None:
        lg = lg.masked_fill(ban_mask, float("-inf"))
    if temperature <= 0:
        return lg.argmax(-1)
    p = torch.softmax(lg / temperature, dim=-1)
    return torch.multinomial(p[0], 1, generator=generator)


@torch.no_grad()
def append(model: XSModel, cache: XSCache, ids) -> torch.Tensor:
    ids = list(ids)
    if len(ids) == 1:
        return model.step(int(ids[0]), cache)
    return model.forward(ids, cache)


class Decoder:
    """Batch-1 sampling loop on an XSCache (moth.serve's turn loop)."""

    mode = "eager"
    fallback = None

    def __init__(self, model: XSModel, cache: XSCache, **_):
        self.model, self.cache = model, cache

    def warmup(self, *a, **kw):
        return None

    def clear_turn_state(self):
        return None

    def step(self, tok: int) -> torch.Tensor:
        return self.model.step(int(tok), self.cache)

    def _masks(self, V, pen_ids, ban_ids):
        pm = bm = None
        if pen_ids:
            pm = torch.zeros(V, dtype=torch.bool)
            pm[torch.as_tensor(sorted(pen_ids), dtype=torch.long)] = True
        if ban_ids:
            bm = torch.zeros(V, dtype=torch.bool)
            bm[torch.as_tensor(sorted(ban_ids), dtype=torch.long)] = True
        dev = self.cache.device
        return (pm.to(dev) if pm is not None else None), (bm.to(dev) if bm is not None else None)

    @torch.no_grad()
    def generate(self, logits: torch.Tensor, max_new: int, stop: int, temperature: float = 0.8, top_p: float = 0.95,
                 pen: float = 1.0, pen_ids=(), ban_ids=(), generator=None, logits_hook=None) -> list[int]:
        V = logits.size(-1)
        pm, bm = self._masks(V, pen_ids if pen != 1.0 else (), ban_ids)
        out: list[int] = []
        room = self.cache.max_len - self.cache.len - 1   # always leave room for the closing stop
        for _ in range(max(0, min(max_new, room))):
            if logits_hook is not None:
                logits = logits_hook(out, logits)
            t = int(sample_top_p(logits, temperature, top_p, pen, pm, bm, generator))
            out.append(t)
            logits = self.step(t)
            if t == stop:
                break
        if not out or out[-1] != stop:
            self.step(stop)
        return out
