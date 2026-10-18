"""Incremental decoding with the small KV cache, for batched multi-turn rollouts.

Cache layout (what makes it small):
  * global K/V: one per layer GROUP (written by the group's Full layer), grows with context;
  * local K/V: one ring buffer of `swa_window` slots per layer, constant size.
Each row has its own length, so episodes can be at different points: `append` takes a
right-padded chunk of new tokens per row (prefill, tool results, or one decoded token).

No host syncs inside a step: padded tokens are written to a scratch slot (last index) instead of
being filtered out with boolean masks, lengths are mirrored on the CPU, and attention is computed
directly against the cache with grouped queries (no K/V concatenation or head repetition).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from tiny_agent.model import TinyAgentLM, rms_norm

CHUNK = 64  # max new tokens per row processed at once (bounds the score tensor during prefill)


class KVCache:
    def __init__(self, model: TinyAgentLM, B: int, max_len: int, device, dtype=torch.bfloat16):
        c = model.cfg
        self.cfg, self.B, self.max_len, self.device = c, B, max_len, device
        G = math.ceil(c.n_layers / c.kv_group)
        Hk, D, W = c.n_kv_heads, c.head_dim, c.swa_window
        self.gk = torch.zeros(G, B, Hk, max_len + 1, D, device=device, dtype=dtype)   # +1 scratch slot
        self.gv = torch.zeros_like(self.gk)
        self.lk = torch.zeros(c.n_layers, B, Hk, W + 1, D, device=device, dtype=dtype)  # +1 scratch slot
        self.lv = torch.zeros_like(self.lk)
        self.lpos = torch.full((B, W + 1), -1, device=device, dtype=torch.long)
        self.lens = torch.zeros(B, device=device, dtype=torch.long)
        self.lens_cpu = torch.zeros(B, dtype=torch.long)
        self.tokens = torch.zeros(B, max_len + 1, device=device, dtype=torch.long)    # history (Engram)
        self.static_feed = None   # set by a GPU Decoder: append() replays captured graphs for short batch-1 chunks

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.gk, self.gv, self.lk, self.lv))

    def reset(self, rows) -> None:
        """Empty these rows so a new sequence can start there (old K/V is masked by length)."""
        rows = torch.as_tensor(rows, dtype=torch.long)
        self.lens_cpu[rows] = 0
        rows = rows.to(self.device)
        self.lens[rows] = 0
        self.lpos[rows] = -1


def _rope_cs(x, cos, sin):
    """_rope_pos with the cos/sin rows already gathered (and cast to x's dtype): same values, fewer kernels."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


def _rope_pos(rope, x, pos):
    """x: (B, H, C, D); pos: (B, C) per-row absolute positions."""
    cos = rope.cos[pos][:, None].to(x.dtype)
    sin = rope.sin[pos][:, None].to(x.dtype)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


# Queries per attention block in append(): bounds the mask / key copies of a long prefill. Attention itself is fused
# (scaled_dot_product_attention), so this is about speed: CPU likes small blocks, GPUs fewer launches.
QUERY_BLOCK = {"cpu": 256}
QUERY_BLOCK_DEFAULT = 1024
PAD_SHAPES = None      # None: pad on GPUs only (True: always, for tests)
KEY_ROUND = 256        # GPUs: global key ranges rounded up to this (stable shapes; the extra keys are masked)


def _mlp(mlp, x):
    """MLP.forward (down(relu(up(x))^2)) with the activation done in place: same values, two fewer big buffers."""
    u = mlp.up(x)
    torch.relu_(u)
    return mlp.down(u.mul_(u))


EXPLICIT_ATTN_MAX = 256


def _attn_explicit(q, K, V, m, Hk):
    """scaled_dot_product_attention(q, K, V, attn_mask=m, enable_gqa=True) as matmul + softmax (fp32 softmax).
    q (B, H, n, D), K/V (B, Hk, L, D), m (B, 1, n, L) bool."""
    B, H, n, D = q.shape
    rep = H // Hk
    s = (q.reshape(B, Hk, rep * n, D) @ K.transpose(-1, -2)).float().view(B, Hk, rep, n, -1)
    s = (s * (1.0 / math.sqrt(D))).masked_fill_(~m[:, :, None], float("-inf"))
    p = torch.softmax(s, dim=-1).to(V.dtype).view(B, Hk, rep * n, -1)
    return (p @ V).view(B, H, n, D)


def _block_size(dev, C: int, block=None) -> int:
    qb = int(block) if block else QUERY_BLOCK.get(torch.device(dev).type, QUERY_BLOCK_DEFAULT)
    return max(1, min(qb, C))


@torch.no_grad()
def append(model: TinyAgentLM, cache: KVCache, idx: torch.Tensor, n_new, rows=None, block=None) -> torch.Tensor:
    """Feed a right-padded chunk idx (B, C); row b has n_new[b] real tokens (0 = untouched).
    n_new may be a CPU tensor/list (preferred: avoids a device sync). Returns logits (B, V) at
    each row's last real token (unchanged garbage for rows with n_new == 0).
    rows: optional cache rows that idx refers to (a sub-batch, e.g. only rows that need prefill);
    by default idx covers all cache rows in order.

    One pass over all new tokens, like the training forward: every projection and MLP runs once over the whole
    chunk; attention is fused (scaled_dot_product_attention) and split into query blocks (block: queries per block,
    default QUERY_BLOCK), each attending to just the keys it can see: global keys up to its last position and the
    band of local keys of its windows. Same results as append_chunked up to float rounding (softmax/reduction
    order); same cache contents and layout."""
    n_cpu = torch.as_tensor(n_new, dtype=torch.long).cpu()
    B, C = idx.shape
    if rows is not None:
        rows = torch.as_tensor(rows, dtype=torch.long).cpu()
    lens_cpu = cache.lens_cpu if rows is None else cache.lens_cpu[rows]
    if int((lens_cpu + n_cpu).max()) > cache.max_len:
        raise RuntimeError("KV cache full")
    width = int(n_cpu.max()) if n_cpu.numel() else 0
    if width == 0:
        return None
    idx = idx[:, :width]
    sf = cache.static_feed
    if sf is not None and rows is None and B == 1 and width <= sf.max_c:
        return sf.feed(idx, width)
    key_round = 1
    if PAD_SHAPES if PAD_SHAPES is not None else idx.device.type != "cpu":
        # GPUs: pad the chunk to a power of two and round the global key range up, so attention / GEMM shapes repeat
        # (XPU builds an SDPA kernel per new shape: ~6 ms each, every turn). Padding is masked out and written to the
        # scratch slots, like any right padding.
        Cp = max(16, 1 << (width - 1).bit_length())
        if Cp > width:
            idx = torch.cat([idx, idx.new_zeros(B, Cp - width)], dim=1)
        key_round = KEY_ROUND
    return _append_all(model, cache, idx, n_cpu, rows, block, key_round)


def _append_all(model, cache: KVCache, idx, n_cpu, rows=None, block=None, key_round=1):
    c = model.cfg
    B, C = idx.shape
    dev = idx.device
    H, Hk, D, W = c.n_heads, c.n_kv_heads, c.head_dim, c.swa_window
    dt = cache.gk.dtype
    n = n_cpu.to(dev, non_blocking=True)
    if rows is None:
        rdev = torch.arange(B, device=dev)
        sel = lambda t: t                                                       # noqa: E731
        lens, lens_cpu = cache.lens, cache.lens_cpu
    else:
        rdev = rows.to(dev, non_blocking=True)
        sel = lambda t: t[rdev]                                                 # noqa: E731
        lens, lens_cpu = cache.lens[rdev], cache.lens_cpu[rows]
    ar = torch.arange(C, device=dev)
    valid = ar[None, :] < n[:, None]                                            # (B, C)
    pos = lens[:, None] + ar[None, :]                                           # (B, C)
    new_lens = lens + n
    new_lens_cpu = lens_cpu + n_cpu
    L0, Lg = int(lens_cpu.max()), int(new_lens_cpu.max())
    bidx = rdev[:, None].expand(B, C)
    gpos = torch.where(valid, pos, torch.full_like(pos, cache.max_len))        # scratch for padding
    keep = valid & (pos >= (new_lens[:, None] - W))                            # last W real tokens
    slot = torch.where(keep, pos % W, torch.full_like(pos, W))                 # scratch slot W

    addrs = {}
    if model.engram_modules():
        max_n = max(c.engram_orders)
        back = torch.arange(-(max_n - 1), 0, device=dev)
        prev_pos = lens[:, None] + back[None, :]
        prev = torch.where(prev_pos >= 0, sel(cache.tokens).gather(1, prev_pos.clamp_min(0)), torch.zeros_like(prev_pos))
        cids = model.cid_map[torch.cat([prev, idx], dim=1)]
        for i, b in enumerate(model.blocks):
            if b.engram is not None:
                addrs[i] = b.engram.addresses(cids)[:, max_n - 1:]
    cache.tokens[bidx, gpos] = idx

    # Local keys in position order: seq[j] = position lens - W + j, i.e. the ring's last W entries (j < W) then the new
    # tokens (j >= W). Query i (position lens + i) sees seq[i+1 .. i+W]: a band, so each query block only needs a
    # slice of it. Masks are the same in every layer: build them once.
    qb = _block_size(dev, C, block)
    wpos = lens[:, None] - W + torch.arange(W, device=dev)[None, :]           # (B, W) positions of the ring part
    ring_idx = wpos % W
    ring_ok = (wpos >= 0) & (sel(cache.lpos).gather(1, ring_idx) == wpos)      # slot really holds that position
    key_ok = torch.cat([ring_ok, valid], dim=1)                                 # (B, W + C)
    ring_gather = ring_idx[:, None, :, None].expand(B, Hk, W, D)
    blocks = []
    for s in range(0, C, qb):
        e = min(C, s + qb)
        lg = min(Lg, L0 + e)              # the last global key any query of this block can see
        lg = min(-(-lg // key_round) * key_round, cache.max_len)   # masked beyond the length
        pq = pos[:, s:e, None]
        kpos = torch.arange(lg, device=dev)
        gm = (kpos[None, None, :] <= pq) & (kpos[None, None, :] < new_lens[:, None, None])
        qi = torch.arange(s, e, device=dev)[:, None]
        j = torch.arange(s + 1, e + W, device=dev)[None, :]
        lm = ((j > qi) & (j <= qi + W))[None] & key_ok[:, None, s + 1:e + W]
        m = torch.cat([gm, lm], dim=2)
        m[:, :, 0] |= ~valid[:, s:e]       # padding queries: never an empty row (SDPA would return NaN)
        blocks.append((s, e, lg, m[:, None]))                                   # B,1,nq,L

    # short chunks on GPUs (a new turn, a tool result): explicit attention. XPU builds an SDPA kernel per shape.
    manual = dev.type != "cpu" and C <= EXPLICIT_ATTN_MAX
    x = model.embed(idx)
    rope = model.rope
    group = -1
    for li, blk in enumerate(model.blocks):
        if blk.engram is not None:
            x = x + blk.engram(x, addrs[li])
        a = blk.attn
        h = rms_norm(x)
        q = _rope_pos(rope, rms_norm(a.w_q(h).view(B, C, H, D).transpose(1, 2)), pos).to(dt)
        if a.mode == "full":
            group += 1
            k, v = a.w_kv_global(h).view(B, C, 2, Hk, D).permute(2, 0, 3, 1, 4)
            k = _rope_pos(rope, rms_norm(k), pos)
            cache.gk[group][bidx, :, gpos] = k.transpose(1, 2).to(dt)
            cache.gv[group][bidx, :, gpos] = v.transpose(1, 2).to(dt)
            if rows is None:
                Kg_all, Vg_all = cache.gk[group], cache.gv[group]
            else:
                lgmax = blocks[-1][2]
                Kg_all, Vg_all = cache.gk[group][rdev, :, :lgmax], cache.gv[group][rdev, :, :lgmax]
        kl, vl = a.w_kv_local(h).view(B, C, 2, Hk, D).permute(2, 0, 3, 1, 4)
        kl = _rope_pos(rope, rms_norm(kl), pos).to(cache.lk.dtype)
        vl = vl.to(cache.lv.dtype)
        seq_k = torch.cat([sel(cache.lk[li]).gather(2, ring_gather), kl], dim=2)   # (B, Hk, W + C, D)
        seq_v = torch.cat([sel(cache.lv[li]).gather(2, ring_gather), vl], dim=2)
        ys = []
        for s, e, lg, m in blocks:
            Kb = torch.cat([Kg_all[:, :, :lg], seq_k[:, :, s + 1:e + W]], dim=2)
            Vb = torch.cat([Vg_all[:, :, :lg], seq_v[:, :, s + 1:e + W]], dim=2)
            if manual:
                ys.append(_attn_explicit(q[:, :, s:e], Kb, Vb, m, Hk))
            else:
                ys.append(F.scaled_dot_product_attention(q[:, :, s:e], Kb, Vb, attn_mask=m, enable_gqa=True))
        y = ys[0] if len(ys) == 1 else torch.cat(ys, dim=2)
        y = y.transpose(1, 2).reshape(B, C, H * D)
        x = x + a.w_o(y.to(x.dtype))
        x = x + _mlp(blk.mlp, rms_norm(x))
        cache.lk[li][bidx, :, slot] = kl.transpose(1, 2)
        cache.lv[li][bidx, :, slot] = vl.transpose(1, 2)
    cache.lpos[bidx, slot] = torch.where(keep, pos, torch.full_like(pos, -1))
    if rows is None:            # in place: a captured decode graph holds these tensors
        cache.lens.copy_(new_lens)
        cache.lens_cpu.copy_(new_lens_cpu)
    else:
        cache.lens[rdev] = new_lens
        cache.lens_cpu[rows] = new_lens_cpu
    last = (n - 1).clamp_min(0)
    xl = x[torch.arange(B, device=dev), last]
    logits = model.head(rms_norm(xl)).float()
    sc = c.logit_softcap
    return sc * torch.tanh(logits / sc)


@torch.no_grad()
def append_chunked(model: TinyAgentLM, cache: KVCache, idx: torch.Tensor, n_new, rows=None) -> torch.Tensor:
    """Reference implementation (the pre-2026-10-18 serving path, kept for equivalence tests and benches):
    processes the new tokens in CHUNK-token pieces, every piece through all layers.
    Feed a right-padded chunk idx (B, C); row b has n_new[b] real tokens (0 = untouched).
    n_new may be a CPU tensor/list (preferred: avoids a device sync). Returns logits (B, V) at
    each row's last real token (unchanged garbage for rows with n_new == 0).
    rows: optional cache rows that idx refers to (a sub-batch, e.g. only rows that need prefill);
    by default idx covers all cache rows in order."""
    n_cpu = torch.as_tensor(n_new, dtype=torch.long).cpu()
    B, C = idx.shape
    if rows is not None:
        rows = torch.as_tensor(rows, dtype=torch.long).cpu()
    lens_cpu = cache.lens_cpu if rows is None else cache.lens_cpu[rows]
    if int((lens_cpu + n_cpu).max()) > cache.max_len:
        raise RuntimeError("KV cache full")
    out = None
    for s in range(0, C, CHUNK):
        n_sub = (n_cpu - s).clamp(0, CHUNK)
        if int(n_sub.max()) == 0:
            break
        width = int(n_sub.max())
        logits = _append_chunk(model, cache, idx[:, s:s + width], n_sub, rows)
        if out is None:
            out = logits
        else:
            out = torch.where((n_sub > 0).to(out.device)[:, None], logits, out)
    return out


def _append_chunk(model, cache: KVCache, idx, n_cpu, rows=None):
    c = model.cfg
    B, C = idx.shape
    dev = idx.device
    H, Hk, D, W = c.n_heads, c.n_kv_heads, c.head_dim, c.swa_window
    rep = H // Hk
    n = n_cpu.to(dev, non_blocking=True)
    # full batch: views of the cache; sub-batch: gathered copies of just those rows
    if rows is None:
        rdev = torch.arange(B, device=dev)
        sel = lambda t: t                                                       # noqa: E731
        lens, lens_cpu = cache.lens, cache.lens_cpu
    else:
        rdev = rows.to(dev, non_blocking=True)
        sel = lambda t: t[rdev]                                                 # noqa: E731
        lens, lens_cpu = cache.lens[rdev], cache.lens_cpu[rows]
    ar = torch.arange(C, device=dev)
    valid = ar[None, :] < n[:, None]                                            # (B, C)
    pos = lens[:, None] + ar[None, :]                                           # (B, C)
    new_lens = lens + n
    new_lens_cpu = lens_cpu + n_cpu
    Lg = int(new_lens_cpu.max())
    bidx = rdev[:, None].expand(B, C)
    gpos = torch.where(valid, pos, torch.full_like(pos, cache.max_len))        # scratch for padding
    keep = valid & (pos >= (new_lens[:, None] - W))                            # last W real tokens
    slot = torch.where(keep, pos % W, torch.full_like(pos, W))                 # scratch slot W

    # Engram addresses need the previous n-1 tokens of each row
    addrs = {}
    if model.engram_modules():
        max_n = max(c.engram_orders)
        back = torch.arange(-(max_n - 1), 0, device=dev)
        prev_pos = lens[:, None] + back[None, :]
        prev = torch.where(prev_pos >= 0, sel(cache.tokens).gather(1, prev_pos.clamp_min(0)), torch.zeros_like(prev_pos))
        cids = model.cid_map[torch.cat([prev, idx], dim=1)]
        for i, b in enumerate(model.blocks):
            if b.engram is not None:
                addrs[i] = b.engram.addresses(cids)[:, max_n - 1:]
    cache.tokens[bidx, gpos] = idx

    kpos = torch.arange(Lg, device=dev)
    gmask = (kpos[None, None, :] <= pos[:, :, None]) & (kpos[None, None, :] < new_lens[:, None, None])  # B,C,Lg
    lpos_all = torch.cat([sel(cache.lpos), torch.where(valid, pos, torch.full_like(pos, -1))], dim=1)  # B,W+1+C
    lmask = (lpos_all[:, None, :] >= 0) & (lpos_all[:, None, :] <= pos[:, :, None]) & \
            (pos[:, :, None] - lpos_all[:, None, :] < W)
    mask = torch.cat([gmask, lmask], dim=2)[:, None, None]                      # B,1,1,C,L
    neg = torch.finfo(torch.float32).min / 2
    scale = 1.0 / math.sqrt(D)

    x = model.embed(idx)
    rope = model.rope
    group = -1
    for li, blk in enumerate(model.blocks):
        if blk.engram is not None:
            x = x + blk.engram(x, addrs[li])
        a = blk.attn
        h = rms_norm(x)
        q = _rope_pos(rope, rms_norm(a.w_q(h).view(B, C, H, D).transpose(1, 2)), pos)
        if a.mode == "full":
            group += 1
            k, v = a.w_kv_global(h).view(B, C, 2, Hk, D).permute(2, 0, 3, 1, 4)
            k = _rope_pos(rope, rms_norm(k), pos)
            cache.gk[group][bidx, :, gpos] = k.transpose(1, 2).to(cache.gk.dtype)
            cache.gv[group][bidx, :, gpos] = v.transpose(1, 2).to(cache.gv.dtype)
        kl, vl = a.w_kv_local(h).view(B, C, 2, Hk, D).permute(2, 0, 3, 1, 4)
        kl = _rope_pos(rope, rms_norm(kl), pos).to(cache.lk.dtype)
        vl = vl.to(cache.lv.dtype)
        # grouped-query attention straight against the cache: queries (B, Hk, rep*C, D)
        q2 = q.to(cache.gk.dtype).reshape(B, Hk, rep * C, D)
        if rows is None:
            Kg, Vg = cache.gk[group][:, :, :Lg], cache.gv[group][:, :, :Lg]
        else:
            Kg, Vg = cache.gk[group][rdev, :, :Lg], cache.gv[group][rdev, :, :Lg]
        Kl = torch.cat([sel(cache.lk[li]), kl], dim=2)                          # B,Hk,W+1+C,D (small)
        Vl = torch.cat([sel(cache.lv[li]), vl], dim=2)
        s = torch.cat([q2 @ Kg.transpose(-1, -2), q2 @ Kl.transpose(-1, -2)], dim=-1).float() * scale
        s = s.view(B, Hk, rep, C, -1).masked_fill(~mask, neg)
        p = torch.softmax(s, dim=-1).to(Vg.dtype).view(B, Hk, rep * C, -1)
        y = p[..., :Lg] @ Vg + p[..., Lg:] @ Vl                                 # B,Hk,rep*C,D
        y = y.view(B, H, C, D).transpose(1, 2).reshape(B, C, H * D)
        x = x + a.w_o(y.to(x.dtype))
        x = x + blk.mlp(rms_norm(x))
        cache.lk[li][bidx, :, slot] = kl.transpose(1, 2)
        cache.lv[li][bidx, :, slot] = vl.transpose(1, 2)
    cache.lpos[bidx, slot] = torch.where(keep, pos, torch.full_like(pos, -1))
    if rows is None:
        cache.lens, cache.lens_cpu = new_lens, new_lens_cpu
    else:
        cache.lens[rdev] = new_lens
        cache.lens_cpu[rows] = new_lens_cpu
    last = (n - 1).clamp_min(0)
    xl = x[torch.arange(B, device=dev), last]
    logits = model.head(rms_norm(xl)).float()
    sc = c.logit_softcap
    return sc * torch.tanh(logits / sc)


def sample(logits: torch.Tensor, temperature: float = 1.0, generator=None) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(-1)
    p = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(p, 1, generator=generator).squeeze(-1)


def sample_top_p(logits: torch.Tensor, temperature: float, top_p: float, pen: float = 1.0, pen_mask=None,
                 ban_mask=None, generator=None) -> torch.Tensor:
    """moth's sampler, on the logits' device, no host sync: ban_mask -> -inf, repetition penalty on pen_mask (positive
    logits / pen, negative * pen), temperature, nucleus top_p, one multinomial draw. logits (1, V); masks (V,) bool or
    None. Returns a (1,) token tensor. Same values as moth.serve's per-token version; for vocabularies up to
    TOPK_SAMPLE_MIN also the same draws (bigger CPU vocabularies: _nucleus_draw_topk, same distribution).
    temperature <= 0: greedy (after the ban and the penalty)."""
    src = logits
    logits = logits.float()
    if ban_mask is not None:
        logits = logits.masked_fill(ban_mask, float("-inf"))
    if pen != 1.0 and pen_mask is not None:
        if logits.device.type == "cpu":   # only the penalised ids (a few hundred), not four passes over V
            ix = pen_mask.nonzero().squeeze(-1)
            if logits is src:
                logits = logits.clone()
            v = logits[..., ix]
            logits[..., ix] = torch.where(v > 0, v / pen, v * pen)
        else:
            logits = torch.where(pen_mask, torch.where(logits > 0, logits / pen, logits * pen), logits)
    if temperature <= 0:
        return logits.argmax(-1)
    p = torch.softmax(logits / temperature, dim=-1)
    if p.device.type == "cpu" and p.size(-1) > TOPK_SAMPLE_MIN:
        r = _nucleus_draw_topk(p, top_p, generator)
        if r is not None:
            return r
    sp, si = p.sort(dim=-1, descending=True)
    keep = sp.cumsum(-1) - sp < top_p
    sp = sp * keep
    return si[0, torch.multinomial(sp[0] / sp[0].sum(), 1, generator=generator)]


TOPK_SAMPLE_MIN = 4096   # CPU vocabularies above this: nucleus via topk (sorting 32k probs costs ~2 ms per token)


def _nucleus_draw_topk(p, top_p, generator):
    """sample_top_p's sort + multinomial draw without the full sort: the nucleus is the top-k prefix (k grown until the
    first dropped token is inside it), then one multinomial draw over those k. Same nucleus, same distribution; a
    seeded draw differs from the sorted path's (multinomial draws one exponential per entry: k instead of V, which
    also saves ~0.5 ms per token). None: the nucleus is most of the vocabulary (the caller sorts)."""
    V = p.size(-1)
    for k in (64, 1024):                                 # a flat distribution: two tries, then the sort
        if k * 4 > V:
            break
        sp, si = p.topk(k, dim=-1)                       # sorted, descending
        keep = sp.cumsum(-1) - sp < top_p
        if not bool(keep[0, -1]):
            sp = sp * keep
            return si[0, torch.multinomial(sp[0] / sp[0].sum(), 1, generator=generator)]
    return None


def _sample_icdf(logits, temperature, top_p, pen, pen_mask, ban_mask, u):
    """sample_top_p with the draw done by inverse CDF from a given uniform u (1,): graph-capturable (no RNG op
    inside the graph). Same distribution as sample_top_p."""
    logits = logits.float().masked_fill(ban_mask, float("-inf"))
    if pen != 1.0:
        logits = torch.where(pen_mask, torch.where(logits > 0, logits / pen, logits * pen), logits)
    if temperature <= 0:
        return logits.argmax(-1)
    p = torch.softmax(logits / temperature, dim=-1)
    sp, si = p.sort(dim=-1, descending=True)
    keep = sp.cumsum(-1) - sp < top_p
    cdf = (sp * keep).cumsum(-1)
    j = torch.searchsorted(cdf, (u * cdf[:, -1])[:, None], right=True).clamp_max(cdf.size(-1) - 1)
    return si.gather(-1, j).view(1)


def _phrase_ban_mask(out, i, ban_ctx, ban_tok, scratch, ar):
    """moth.norepeat.BanHook on the device, graph-capturable (fixed shapes, no host values): (1, V) bool, True for the
    ids that would complete a banned n-gram after the last k = ar.numel() sampled tokens out[i-k:i] (nothing before k
    tokens are sampled, as the hook). ban_ctx (cap, k) context rows (-1 = unused row), ban_tok (cap,) the banned id per
    row (V = unused, lands in scratch's extra slot), scratch (V + 1,) float work buffer, i (1,) long."""
    k = ar.numel()
    ctx = out.index_select(0, (i - k + ar).clamp_min(0))                      # (k,)
    hit = (ban_ctx == ctx[None]).all(-1) & (i >= k)                           # (cap,)
    scratch.zero_().index_add_(0, ban_tok, hit.to(scratch.dtype))
    return (scratch[:-1] > 0)[None]


def _decode_one(model, cache: KVCache, idx, L: int):
    """Append one token (idx (1, 1), device) to row 0 at host-known position L; returns softcapped logits (1, V).
    Exactly the ops and shapes append_chunked runs for a single token (bit-identical logits), minus its general
    batching/masking work: slices instead of scatter writes, the step mask built once."""
    c = model.cfg
    dev = idx.device
    H, Hk, D, W = c.n_heads, c.n_kv_heads, c.head_dim, c.swa_window
    rep = H // Hk
    dt = cache.gk.dtype
    rope = model.rope
    p2 = torch.full((1, 1), L, device=dev, dtype=torch.long)
    addrs = {}
    if model.engram_modules():
        max_n = max(c.engram_orders)
        prev = cache.tokens[:1, max(0, L - (max_n - 1)):L]
        if prev.size(1) < max_n - 1:
            prev = torch.cat([prev.new_zeros(1, max_n - 1 - prev.size(1)), prev], dim=1)
        cids = model.cid_map[torch.cat([prev, idx], dim=1)]
        for i, b in enumerate(model.blocks):
            if b.engram is not None:
                addrs[i] = b.engram.addresses(cids)[:, max_n - 1:]
    cache.tokens[0, L] = idx[0, 0]
    slot = L % W
    lp = torch.cat([cache.lpos[0], p2[0]])                                     # (W+2,)
    mask = torch.cat([torch.ones(L + 1, dtype=torch.bool, device=dev), (lp >= 0) & (lp <= L) & (L - lp < W)])
    inv = ~mask.view(1, 1, 1, 1, -1)
    neg = torch.finfo(torch.float32).min / 2
    scale = 1.0 / math.sqrt(D)
    x = model.embed(idx)
    group = -1
    for li, blk in enumerate(model.blocks):
        if blk.engram is not None:
            x = x + blk.engram(x, addrs[li])
        a = blk.attn
        h = rms_norm(x)
        q = _rope_pos(rope, rms_norm(a.w_q(h).view(1, 1, H, D).transpose(1, 2)), p2)
        if a.mode == "full":
            group += 1
            k, v = a.w_kv_global(h).view(1, 1, 2, Hk, D).permute(2, 0, 3, 1, 4)
            k = _rope_pos(rope, rms_norm(k), p2)
            cache.gk[group][:, :, L] = k[:, :, 0].to(dt)
            cache.gv[group][:, :, L] = v[:, :, 0].to(dt)
            Kg, Vg = cache.gk[group][:, :, :L + 1], cache.gv[group][:, :, :L + 1]
        kl, vl = a.w_kv_local(h).view(1, 1, 2, Hk, D).permute(2, 0, 3, 1, 4)
        kl = _rope_pos(rope, rms_norm(kl), p2).to(cache.lk.dtype)
        vl = vl.to(cache.lv.dtype)
        q2 = q.to(dt).reshape(1, Hk, rep, D)
        Kl = torch.cat([cache.lk[li][:1], kl], dim=2)
        Vl = torch.cat([cache.lv[li][:1], vl], dim=2)
        s = torch.cat([q2 @ Kg.transpose(-1, -2), q2 @ Kl.transpose(-1, -2)], dim=-1).float() * scale
        s = s.view(1, Hk, rep, 1, -1).masked_fill(inv, neg)
        p = torch.softmax(s, dim=-1).to(Vg.dtype).view(1, Hk, rep, -1)
        y = p[..., :L + 1] @ Vg + p[..., L + 1:] @ Vl
        y = y.view(1, H, 1, D).transpose(1, 2).reshape(1, 1, H * D)
        x = x + a.w_o(y.to(x.dtype))
        x = x + blk.mlp(rms_norm(x))
        cache.lk[li][:1, :, slot] = kl[:, :, 0]
        cache.lv[li][:1, :, slot] = vl[:, :, 0]
    cache.lpos[0, slot] = L
    cache.lens[0] += 1
    cache.lens_cpu[0] += 1
    logits = model.head(rms_norm(x[:, 0])).float()
    sc = c.logit_softcap
    return sc * torch.tanh(logits / sc)


class _StaticAux:
    """Per-model tensors for the fused graph step: per layer one weight for [q | k_local | k_global | v_local |
    v_global] (one GEMM instead of 2-3, and the q/k heads contiguous so one norm + one RoPE matmul covers them),
    plus the RoPE helper matrices. Built once per Decoder (an extra ~36 MB in bf16 for this model)."""

    def __init__(self, model: TinyAgentLM, cache: KVCache):
        c = model.cfg
        Hk, D = c.n_kv_heads, c.head_dim
        dev = cache.gk.device
        self.wf, self.nqk = [], []
        for blk in model.blocks:
            a = blk.attn
            wl = a.w_kv_local.weight
            parts_k, parts_v = [wl[:Hk * D]], [wl[Hk * D:]]
            if a.mode == "full":
                wg = a.w_kv_global.weight
                parts_k.append(wg[:Hk * D])
                parts_v.append(wg[Hk * D:])
            self.wf.append(torch.cat([a.w_q.weight] + parts_k + parts_v, dim=0).contiguous())
            self.nqk.append(c.n_heads + Hk * len(parts_k))
        self.kpos = torch.arange(cache.max_len, device=dev)
        self.ar = torch.arange(max(_StaticFeed.BUCKETS), device=dev)
        max_n = max(c.engram_orders) if model.engram_modules() else 1
        self.back = torch.arange(-(max_n - 1), 0, device=dev)
        half = D // 2
        dt = model.embed.weight.dtype
        self.eye = torch.eye(D, device=dev, dtype=dt)
        P = torch.zeros(D, D, device=dev, dtype=dt)
        j = torch.arange(half, device=dev)
        P[j + half, j] = -1
        P[j, j + half] = 1
        self.P = P


def _step_static(model, cache: KVCache, idx, n, aux: _StaticAux):
    """Append the first n of the C tokens idx (1, C) to row 0 with static shapes and no host values (CUDA/XPU graph
    capture). n (1,) long on the device; n == 0 is a no-op on the cache (writes go to the scratch slots, the length
    stays), so a captured decode loop can run past the end of a turn without touching the window. The position is
    cache.lens on the device; global attention runs over all max_len slots with a length mask. Returns softcapped
    logits (1, V) at the last real token.
    Fused for few kernels (launch overhead is the cost at batch 1): one QKV GEMM per layer, one norm and one RoPE
    over the q/k heads (a matmul when C == 1), baddbmm with an additive mask, residual adds inside addmm. Same math
    as append; results differ by rounding (GEMM shapes, RoPE as a matmul)."""
    c = model.cfg
    H, Hk, D, W = c.n_heads, c.n_kv_heads, c.head_dim, c.swa_window
    rep = H // Hk
    M = cache.max_len
    C = idx.size(1)
    dt = cache.gk.dtype
    rope = model.rope
    lens = cache.lens[:1]                                                        # (1,)
    ar = aux.ar[:C]
    pos = lens + ar                                                              # (C,)
    valid = ar < n
    new_len = lens + n
    gpos = torch.where(valid, pos, torch.full_like(pos, M))
    keep = valid & (pos >= new_len - W)
    slot = torch.where(keep, pos % W, torch.full_like(pos, W))
    lnew = torch.where(valid, pos, torch.full_like(pos, -1))
    addrs = {}
    if model.engram_modules():
        max_n = max(c.engram_orders)
        prev_pos = lens + aux.back                                               # (max_n - 1,)
        # index_select, not gather: XPU's gather kernel trips an index assert under graph replay
        prev = cache.tokens[0].index_select(0, prev_pos.clamp_min(0))
        prev = torch.where(prev_pos >= 0, prev, torch.zeros_like(prev))
        cids = model.cid_map[torch.cat([prev, idx[0]])][None]
        for i, b in enumerate(model.blocks):
            if b.engram is not None:
                addrs[i] = b.engram.addresses(cids)[:, max_n - 1:]
    cache.tokens[0].index_copy_(0, gpos, idx[0])
    lp = torch.cat([cache.lpos[0], lnew])                                       # (W+1+C,) local key positions
    pq = pos[:, None]
    mask = torch.cat([(aux.kpos[None, :] <= pq) & (aux.kpos[None, :] < new_len),
                      (lp[None, :] >= 0) & (lp[None, :] <= pq) & (pq - lp[None, :] < W)], dim=1)   # (C, L)
    mask[:, 0] |= ~valid                 # padding / no-op queries: never an all-masked row (NaN)
    x = model.embed(idx[0])                                                      # (C, d)
    xdt = x.dtype
    bias = torch.zeros(mask.shape, device=x.device, dtype=dt).masked_fill_(~mask, float("-inf"))
    bias = bias.repeat(rep, 1)[None]                                             # (1, rep*C, L), rows r*C + i
    if C == 1:
        cs, sn = rope.cos[pos].to(xdt), rope.sin[pos].to(xdt)                    # (1, D/2)
        R = aux.eye * torch.cat([cs, cs], dim=-1) + aux.P * torch.cat([sn, sn], dim=-1)   # x @ R == RoPE(x)
    else:
        cs, sn = rope.cos[pos][:, None].to(xdt), rope.sin[pos][:, None].to(xdt)  # (C, 1, D/2)
    scale = 1.0 / math.sqrt(D)
    group = -1
    for li, blk in enumerate(model.blocks):
        if blk.engram is not None:
            x = x + blk.engram(x[None], addrs[li])[0]
        a = blk.attn
        nqk = aux.nqk[li]
        heads = torch.nn.functional.linear(rms_norm(x), aux.wf[li]).view(C, -1, D)   # (C, nqk + nv, D)
        qk = rms_norm(heads[:, :nqk])
        qk = qk @ R if C == 1 else _rope_cs(qk, cs, sn)                          # normed + rotated q / k heads
        q = qk[:, :H].reshape(C, Hk, rep, D).permute(1, 2, 0, 3).reshape(Hk, rep * C, D)
        kl = qk[:, H:H + Hk].transpose(0, 1).to(cache.lk.dtype)                  # (Hk, C, D)
        if a.mode == "full":
            group += 1
            vl, vg = heads[:, nqk:nqk + Hk], heads[:, nqk + Hk:]
            cache.gk[group][0].index_copy_(1, gpos, qk[:, H + Hk:].transpose(0, 1).to(dt))
            cache.gv[group][0].index_copy_(1, gpos, vg.transpose(0, 1).to(dt))
            Kg, Vg = cache.gk[group][0, :, :M], cache.gv[group][0, :, :M]
        else:
            vl = heads[:, nqk:]
        vl = vl.transpose(0, 1).to(cache.lv.dtype)                               # (Hk, C, D)
        Kb = torch.cat([Kg, cache.lk[li][0], kl], dim=1)                         # (Hk, M + W + 1 + C, D)
        Vb = torch.cat([Vg, cache.lv[li][0], vl], dim=1)
        sc = torch.baddbmm(bias, q.to(dt), Kb.transpose(1, 2), alpha=scale)
        p = torch.softmax(sc, dim=-1, dtype=torch.float32).to(dt)
        y = torch.bmm(p, Vb).view(Hk, rep, C, D).permute(2, 0, 1, 3).reshape(C, H * D).to(xdt)
        x = torch.addmm(x, y, a.w_o.weight.t())
        u = torch.nn.functional.linear(rms_norm(x), blk.mlp.up.weight)
        torch.relu_(u)
        x = torch.addmm(x, u.mul_(u), blk.mlp.down.weight.t())
        cache.lk[li][0].index_copy_(1, slot, kl)
        cache.lv[li][0].index_copy_(1, slot, vl)
    cache.lpos[0].index_copy_(0, slot, torch.where(keep, pos, torch.full_like(pos, -1)))
    cache.lens.add_(n)
    xl = x.index_select(0, (n - 1).clamp_min(0))
    logits = model.head(rms_norm(xl)).float()
    sc = c.logit_softcap
    return sc * torch.tanh(logits / sc)


class _StaticFeed:
    """Graph-captured append() for one row-0 chunk of up to max(BUCKETS) tokens (a new turn, a tool result) on a
    CUDA/XPU cache: the chunk is padded to the next bucket, one graph per bucket, captured on first use. Attached to
    the cache by the Decoder (cache.static_feed); append() uses it for batch-1 chunks that fit."""
    BUCKETS = (16, 32, 64, 128, 256, 512)

    def __init__(self, model, cache: KVCache, aux: _StaticAux, capture: bool = True):
        self.model, self.cache, self.aux, self.capture = model, cache, aux, capture
        self.max_c = self.BUCKETS[-1]
        self.n = torch.zeros(1, dtype=torch.long, device=cache.gk.device)
        self.inp, self.out, self.g, self.fns = {}, {}, {}, {}

    def _graph(self, C):
        dev = self.cache.gk.device
        self.inp[C] = torch.zeros(1, C, dtype=torch.long, device=dev)
        model, cache, aux, inp, n = self.model, self.cache, self.aux, self.inp[C], self.n

        def fn():
            self.out[C] = _step_static(model, cache, inp, n, aux)

        self.fns[C] = fn
        if not self.capture:
            self.g[C] = fn
            return
        Graph, graph_ctx, Stream, stream_ctx, current = _graph_api(dev)
        n.zero_()                                    # warm-up / capture as no-ops: writes land in scratch slots
        s = Stream()
        s.wait_stream(current())
        with stream_ctx(s):
            for _ in range(2):
                fn()
        current().wait_stream(s)
        g = Graph()
        with graph_ctx(g):
            fn()
        self.g[C] = g.replay

    def feed(self, idx, w: int):
        C = next(b for b in self.BUCKETS if b >= w)
        if C not in self.g:
            self._graph(C)
        inp = self.inp[C]
        inp[:, :w].copy_(idx[:1, :w])
        if w < C:
            inp[:, w:].zero_()
        self.n.fill_(w)
        self.g[C]()
        self.cache.lens_cpu[0] += w
        return self.out[C].clone()


def _graph_api(dev: torch.device):
    if dev.type == "cuda":
        return torch.cuda.CUDAGraph, torch.cuda.graph, torch.cuda.Stream, torch.cuda.stream, torch.cuda.current_stream
    if dev.type == "xpu" and hasattr(torch.xpu, "XPUGraph"):
        return torch.xpu.XPUGraph, torch.xpu.graph, torch.xpu.Stream, torch.xpu.stream, torch.xpu.current_stream
    return None


class Decoder:
    """Batch-1 token generation on row 0 of a KVCache (moth.serve's turn loop).

    CPU (and graph=False): an eager loop, one host-sampled token per step; each step runs the exact op sequence of
    append_chunked for one token (bit-identical logits) and sample_top_p. For vocabularies up to TOPK_SAMPLE_MIN a
    seeded turn comes out token-for-token the same as the old per-token append loop; bigger CPU vocabularies (moth's
    32k) draw from the same distribution with a topk nucleus, so seeded draws differ from the old loop's. On the CPU a
    host read costs nothing. A quantize_cpu model (the CPU speed modes) steps through _decode_fast instead
    (Decoder.fast = False: _decode_one).
    CUDA/XPU (graph=True, the default there; graph="static" runs the same step uncaptured): one captured device
    graph per step = append the current token, sample the next one on the device (inverse-CDF draw from a pre-drawn
    uniform), store it; the host only reads the tokens back every sync_every steps. The decoder also attaches
    captured-graph feeds to the cache (feed=True), which append() then uses for short batch-1 chunks. After <|im_end|> the steps are no-ops on the cache (see _step_static), so the
    cache ends exactly as the eager loop leaves it. compile=True: torch.compile the step before capture (fewer,
    fused kernels).
    Keep one Decoder per KVCache: the graph is tied to that cache's tensors (reset() it instead of making a new one)."""

    def __init__(self, model: TinyAgentLM, cache: KVCache, graph=None, compile: bool = False, sync_every: int = 8,
                 feed: bool = True):
        assert cache.B == 1, "Decoder is batch 1"
        self.model, self.cache = model, cache
        self.dev = cache.gk.device
        api = _graph_api(self.dev)
        if graph == "static":           # the device-loop path without capture (tests on CPU; capture-free fallback)
            self.graph, self.capture = True, False
        else:
            self.graph = (api is not None) if graph is None else (bool(graph) and api is not None)
            self.capture = self.graph
        self.compile, self.sync_every = compile, max(1, int(sync_every))
        self._g = None
        self._key = None
        self.fallback = None
        self._aux = None
        self._tok = torch.zeros(1, 1, dtype=torch.long, device=self.dev)
        if self.graph and feed:          # append() of short chunks on this cache: captured graphs too
            cache.static_feed = _StaticFeed(model, cache, self._static_aux(), capture=self.capture)

    @torch.no_grad()
    def warmup(self, max_new: int, stop: int, temperature: float = 0.8, top_p: float = 0.95, pen: float = 1.0):
        """Capture the decode graph and every feed bucket now (about 0.15 s each on the B70), so the first turns
        don't pay for it. Safe on any cache state: warm-up and capture are no-ops on the cache."""
        if not self.graph:
            return
        sf = self.cache.static_feed
        if sf is not None:
            for C in sf.BUCKETS:
                if C not in sf.g:
                    sf._graph(C)
        if self._g is None or self._key[:3] != (max_new, (temperature, top_p, pen), stop):
            try:
                self._setup(max_new, temperature, top_p, pen, stop)
            except Exception as e:  # noqa: BLE001  the phrase-ban ops: retry with the plain step (as generate does)
                if not self.ban_graph:
                    raise
                import warnings
                warnings.warn(f"Decoder: device graph setup with the phrase ban failed ({type(e).__name__}: {e}); "
                              "phrase-ban turns decode eagerly")
                self.ban_graph, self._g = False, None
                self._setup(max_new, temperature, top_p, pen, stop)

    def _static_aux(self):
        if self._aux is None:
            self._aux = _StaticAux(self.model, self.cache)
        return self._aux

    fast = True   # CPU speed modes (quantize_cpu): step() runs the lean _decode_fast; False: _decode_one

    @torch.no_grad()
    def step(self, tok: int) -> torch.Tensor:
        """Append one token (host int); returns the logits (1, V) after it."""
        L = int(self.cache.lens_cpu[0])
        if L + 1 > self.cache.max_len:
            raise RuntimeError("KV cache full")
        self._tok.fill_(int(tok))
        fs = getattr(self.model, "_cpu_fast", None)
        if fs is not None and self.fast and self.dev.type == "cpu":
            return _decode_fast(self.model, self.cache, self._tok, L, fs)
        return _decode_one(self.model, self.cache, self._tok, L)

    def _masks(self, V, pen_ids, ban_ids):
        pm = torch.zeros(V, dtype=torch.bool)
        bm = torch.zeros(V, dtype=torch.bool)
        if pen_ids:
            pm[torch.as_tensor(list(pen_ids), dtype=torch.long)] = True
        if ban_ids:
            bm[torch.as_tensor(list(ban_ids), dtype=torch.long)] = True
        return pm.to(self.dev), bm.to(self.dev)

    @torch.no_grad()
    def generate(self, logits: torch.Tensor, max_new: int, stop: int, temperature: float = 0.8, top_p: float = 0.95,
                 pen: float = 1.0, pen_ids=(), ban_ids=(), generator=None, logits_hook=None) -> list[int]:
        """Sample up to max_new tokens starting from logits (1, V) (the prefill's), appending each one to the cache;
        stop (<|im_end|>) ends the turn and is appended too. A turn cut off at max_new gets stop appended so the cache
        stays well-formed. Returns the sampled tokens (stop included when sampled).
        logits_hook(out, logits) -> logits: called before every draw with the tokens sampled so far (moth.norepeat's
        phrase ban). It runs on the host: the eager loop calls it. A hook with a `table` ({context tuple of k ids:
        [banned next ids]}) and `k` (moth.norepeat.BanHook) also works in the device-graph path, as fixed-size device
        tensors matched on the device (_phrase_ban_mask), so it keeps the graphs; any other hook means the eager loop."""
        V = logits.size(-1)
        pm, bm = self._masks(V, pen_ids if pen != 1.0 else (), ban_ids)
        L = int(self.cache.lens_cpu[0])
        table = getattr(logits_hook, "table", None)
        graph_ok = logits_hook is None or (self.ban_graph and isinstance(table, dict)
                                           and isinstance(getattr(logits_hook, "k", None), int))
        if self.graph and graph_ok and L + max_new + 1 <= self.cache.max_len and max_new > 0:
            return self._generate_graph(logits, max_new, stop, temperature, top_p, pen, pm, bm, generator, logits_hook)
        pm_ = pm if pen_ids and pen != 1.0 else None
        bm_ = bm if ban_ids else None
        out = []
        for _ in range(max_new):
            if logits_hook is not None:
                logits = logits_hook(out, logits)
            t = int(sample_top_p(logits, temperature, top_p, pen, pm_, bm_, generator))
            out.append(t)
            logits = self.step(t)
            if t == stop:
                break
        if not out or out[-1] != stop:
            self.step(stop)
        return out

    # --- device graph path --------------------------------------------------------------------------------------
    BAN_CAP = 1024  # phrase-ban entries the captured graph holds (grown, with a re-capture, when a turn needs more)
    BAN_K = 2       # context length of the phrase ban (moth.norepeat.N - 1)
    ban_graph = True   # False after a capture with the ban ops failed: graphs without them, ban hooks run eagerly

    def _setup(self, max_new, temperature, top_p, pen, stop, ban_cap=None, ban_k=None):
        """ban_cap 0: the step without the phrase-ban ops (the plain body)."""
        dev, cache = self.dev, self.cache
        n = max_new + 2
        V = self.model.cfg.vocab_size
        ban_cap = (self.BAN_CAP if self.ban_graph else 0) if ban_cap is None else ban_cap
        ban_k = ban_k or self.BAN_K
        self._done = torch.ones(1, dtype=torch.bool, device=dev)
        self._i = torch.zeros(1, dtype=torch.long, device=dev)
        self._out = torch.zeros(n, dtype=torch.long, device=dev)
        self._u = torch.zeros(n, device=dev)
        self._pm = torch.zeros(1, V, dtype=torch.bool, device=dev)
        self._bm = torch.zeros_like(self._pm)
        # the phrase ban (a logits_hook with a table): rows of k context ids (-1: unused) -> banned id (V: unused)
        self._ban_ctx = torch.full((ban_cap, ban_k), -1, dtype=torch.long, device=dev)
        self._ban_tok = torch.full((ban_cap,), V, dtype=torch.long, device=dev)
        self._ban_scratch = torch.zeros(V + 1, device=dev)
        self._ban_ar = torch.arange(ban_k, device=dev)
        aux = self._static_aux()     # the graph reads these: kept alive on the decoder
        self._stop = stop
        samp = (temperature, top_p, pen)

        def body():
            live = ~self._done
            lg = _step_static(self.model, cache, self._tok, live.long(), aux)
            self._done |= self._tok[0] == stop
            u = self._u.index_select(0, self._i)
            bm = self._bm
            if ban_cap:
                bm = bm | _phrase_ban_mask(self._out, self._i, self._ban_ctx, self._ban_tok, self._ban_scratch,
                                           self._ban_ar)
            nxt = _sample_icdf(lg, *samp, self._pm, bm, u)
            self._out.index_copy_(0, self._i, nxt)
            self._i += 1
            self._tok.copy_(nxt.view(1, 1))

        fn = self._fn = torch.compile(body, fullgraph=False, dynamic=False) if self.compile else body
        self._key = (max_new, samp, stop, ban_cap, ban_k)
        if not self.capture:
            self._g = fn
            return
        Graph, graph_ctx, Stream, stream_ctx, current = _graph_api(dev)
        s = Stream()
        s.wait_stream(current())
        with stream_ctx(s):          # warm-up with done=True: every write lands in a scratch slot
            for _ in range(3):
                fn()
                self._i.zero_()
        current().wait_stream(s)
        g = Graph()
        with graph_ctx(g):
            fn()
        self._i.zero_()
        self._g = g.replay

    @property
    def mode(self) -> str:
        """What generate() runs: "graph" (captured), "static" (uncaptured device loop) or "eager"."""
        if not self.graph:
            return "eager"
        return ("graph" if self.capture else "static") + ("+compile" if self.compile else "")

    def _generate_graph(self, logits, max_new, stop, temperature, top_p, pen, pm, bm, generator, hook=None):
        table = getattr(hook, "table", None) or {}
        rows = [(ctx, t) for ctx, toks in table.items() for t in toks]
        k = hook.k if table else (self._key[4] if self._key else self.BAN_K)
        if not self.ban_graph:
            cap = 0
        else:
            cap = (self._key[3] if self._key else 0) or self.BAN_CAP
            while cap < len(rows):
                cap *= 2
        if self._g is None or self._key != (max_new, (temperature, top_p, pen), stop, cap, k):
            ids = (lambda m: [i for i in range(m.numel()) if m[i]])   # noqa: E731
            try:
                self._setup(max_new, temperature, top_p, pen, stop, cap, k)
            except Exception as e:  # noqa: BLE001  e.g. an op this backend can't capture
                import warnings
                if cap:   # first without the phrase-ban ops: plain turns keep their graphs, ban hooks run eagerly
                    warnings.warn(f"Decoder: device graph setup with the phrase ban failed ({type(e).__name__}: {e}); "
                                  "phrase-ban turns decode eagerly")
                    self.ban_graph, self._g = False, None
                    return self.generate(logits, max_new, stop, temperature, top_p, pen,
                                         ids(pm) if pen != 1.0 else (), ids(bm), generator, hook)
                warnings.warn(f"Decoder: device graph setup failed ({type(e).__name__}: {e}); decoding eagerly")
                self.graph, self.fallback = False, f"{type(e).__name__}: {e}"[:300]
                self._g = None
                self._done = None
                return self.generate(logits, max_new, stop, temperature, top_p, pen,
                                     ids(pm) if pen != 1.0 else (), ids(bm), generator, hook)
        if cap:
            self._ban_ctx.fill_(-1)      # this turn's phrase ban, outside the graph (the graph reads these tensors)
            self._ban_tok.fill_(self.model.cfg.vocab_size)
        if cap and rows:
            self._ban_ctx[:len(rows)].copy_(torch.tensor([c for c, _ in rows], dtype=torch.long))
            self._ban_tok[:len(rows)].copy_(torch.tensor([t for _, t in rows], dtype=torch.long))
        self._pm.copy_(pm.view(1, -1))
        self._bm.copy_(bm.view(1, -1))
        self._u.copy_(torch.rand(self._u.shape, device=self.dev, generator=generator))
        self._i.zero_()
        first = _sample_icdf(logits, temperature, top_p, pen, self._pm, self._bm, self._u[:1])
        self._out[:1].copy_(first)
        self._tok.copy_(first.view(1, 1))
        self._i.fill_(1)
        self._done.zero_()
        n = 0
        toks = None
        while n < max_new:
            k = min(self.sync_every, max_new - n)
            for _ in range(k):
                self._g()
            n += k
            toks = self._out[:n].tolist()
            if stop in toks:
                break
        cache = self.cache
        if stop in toks:
            toks = toks[:toks.index(stop) + 1]
        else:                       # cut off: append stop (its sampled successor is ignored)
            self._tok.fill_(stop)
            self._g()
        self._done.fill_(True)
        cache.lens_cpu.copy_(cache.lens.cpu())
        return toks


# --- optional CPU speed modes (serving flag --quant); they change the logits slightly -----------------------------
# Batch-1 decode on a CPU is memory-bound: every token streams all Linear weights (136M params) once. Storing them in
# bf16 halves the bytes, int8 quarters them. The kernels take bf16 activations (fp32 accumulation inside); the
# residual stream, norms, attention, softmax and the KV cache stay fp32. Every quant Linear has mm(xb): bf16 (rows, in)
# -> bf16 (rows, out), the call _decode_fast makes.
QUANT_GEMV_ROWS = 8    # BF16Linear: up to this many rows use the bf16 kernel (on AVX2 bf16 GEMM is ~3x slower than fp32)
INT8_GEMV_ROWS = 128   # Int8Linear: up to this many rows use the int8 kernel; beyond, dequantize + fp32 GEMM (measured
                       # crossover on a Ryzen 5950X: the kernel costs per row, the GEMM per weight byte)
_SCRATCH_MAX = 16 << 20


def _scratch(shape, dtype):
    """A reused buffer for prefill's dequantized weights (no 10 MB malloc + page faults per call). Per thread."""
    import threading
    if not hasattr(_scratch, "tls"):
        _scratch.tls = threading.local()
    d = getattr(_scratch.tls, "d", None)
    if d is None:
        d = _scratch.tls.d = {}
    key = (tuple(shape), dtype)
    t = d.get(key)
    if t is None:
        t = torch.empty(shape, dtype=dtype)
        if t.numel() * t.element_size() <= _SCRATCH_MAX:
            d[key] = t
    return t


class BF16Linear(torch.nn.Module):
    """A Linear whose weight is stored in bf16: half the bytes to stream per token. The matmul runs on bf16 inputs
    (fp32 accumulation inside oneDNN), the output is returned in the input dtype."""

    def __init__(self, lin: torch.nn.Linear):
        super().__init__()
        self.in_features, self.out_features = lin.in_features, lin.out_features
        self.weight = torch.nn.Parameter(lin.weight.detach().to(torch.bfloat16), requires_grad=False)
        self.bias = None if lin.bias is None else torch.nn.Parameter(lin.bias.detach().to(torch.bfloat16),
                                                                      requires_grad=False)

    def mm(self, xb):
        return torch.nn.functional.linear(xb, self.weight, self.bias)

    def forward(self, x):
        if x.numel() // x.size(-1) > QUANT_GEMV_ROWS:   # prefill: fp32 GEMM on the upcast weight (no bf16 FMA on
            b = None if self.bias is None else self.bias.to(x.dtype)   # AVX2 CPUs: bf16 GEMM is ~3x slower)
            w = self.weight
            return torch.nn.functional.linear(x, _scratch(w.shape, x.dtype).copy_(w), b)
        return torch.nn.functional.linear(x.to(torch.bfloat16), self.weight, self.bias).to(x.dtype)


class Int8Linear(torch.nn.Module):
    """Weight-only int8 Linear: int8 weights with one scale per output channel (symmetric, max-abs; the scale is
    stored in bf16, the kernel's dtype, and the weights are rounded against that stored value), bf16 activations into
    aten._weight_int8pack_mm (dequantizes on the fly, fp32 accumulation). A quarter of the fp32 bytes per token.
    Activations are not quantized (dynamic activation quantization is lossy for this model's ReLU^2 MLP).
    More than INT8_GEMV_ROWS rows (long prefill): dequantize into a scratch buffer + fp32 GEMM."""

    def __init__(self, lin: torch.nn.Linear):
        super().__init__()
        w = lin.weight.detach().float()
        scale = (w.abs().amax(dim=1) / 127.0).clamp_min(1e-12).to(torch.bfloat16)
        q = torch.round(w / scale.float()[:, None]).clamp_(-127, 127).to(torch.int8)
        self.in_features, self.out_features = lin.in_features, lin.out_features
        self.register_buffer("qweight", q.contiguous())
        self.register_buffer("scale", scale.contiguous())
        self.bias = None if lin.bias is None else torch.nn.Parameter(lin.bias.detach().float(), requires_grad=False)

    @property
    def weight(self):   # dequantized copy, for code that reads .weight (the fused GPU step; not used on CPU)
        return self.qweight.float() * self.scale.float()[:, None]

    def mm(self, xb):
        y = torch.ops.aten._weight_int8pack_mm(xb, self.qweight, self.scale)
        return y if self.bias is None else y + self.bias.to(y.dtype)

    def forward(self, x):
        shape = x.shape
        if x.numel() // shape[-1] > INT8_GEMV_ROWS:
            w = torch.mul(self.qweight, self.scale.float()[:, None],
                          out=_scratch(self.qweight.shape, torch.float32)).to(x.dtype)
            return torch.nn.functional.linear(x, w, None if self.bias is None else self.bias.to(x.dtype))
        y = self.mm(x.reshape(-1, shape[-1]).to(torch.bfloat16).contiguous())
        return y.view(*shape[:-1], self.out_features).to(x.dtype)


class _FullMM(torch.nn.Module):
    """mm() for a Linear left in full precision (quantize_cpu(head=False))."""

    def __init__(self, lin):
        super().__init__()
        self.lin = lin

    def mm(self, xb):
        return torch.nn.functional.linear(xb.to(self.lin.weight.dtype), self.lin.weight, self.lin.bias)


QUANT_CLASSES = {"int8": Int8Linear, "bf16": BF16Linear}


def int8_kernel_ok() -> bool:
    """Does this torch build run aten._weight_int8pack_mm on bf16 inputs (and get it right)? Checked once."""
    if not hasattr(int8_kernel_ok, "ok"):
        try:
            g = torch.Generator().manual_seed(0)
            w = torch.randint(-127, 128, (64, 64), dtype=torch.int8, generator=g)
            s = torch.rand(64, generator=g).add_(0.5).to(torch.bfloat16)
            x = torch.randn(2, 64, generator=g).to(torch.bfloat16)
            y = torch.ops.aten._weight_int8pack_mm(x, w, s).float()
            ref = x.float() @ (w.float() * s.float()[:, None]).t()
            int8_kernel_ok.ok = bool(torch.allclose(y, ref, rtol=0.02, atol=0.02 * float(ref.abs().max())))
        except Exception:  # noqa: BLE001  op missing / unsupported dtype on this build
            int8_kernel_ok.ok = False
    return int8_kernel_ok.ok


def quantize_cpu(model: TinyAgentLM, mode: str, head: bool = True) -> TinyAgentLM:
    """In place. mode "int8": weight-only int8 Linear layers (Int8Linear, per-channel scales, bf16 activations).
    "bf16": Linear weights stored in bf16 (BF16Linear). Embeddings, Engram tables, norms, RoPE and the KV cache stay
    as they are. head=False keeps the output head in full precision. Use an fp32 KV cache with these.
    Also builds model._cpu_fast, the packed weights of the lean CPU decode step (_decode_fast, used by Decoder.step)."""
    if mode in (None, "", "none", "fp32"):
        return model
    cls = QUANT_CLASSES.get(mode)
    if cls is None:
        raise ValueError(f"unknown quant mode {mode!r} (none, {', '.join(QUANT_CLASSES)})")
    if cls is Int8Linear and not int8_kernel_ok():
        import warnings
        warnings.warn("this PyTorch build has no CPU int8 kernel for bf16 inputs (aten._weight_int8pack_mm); "
                      "using --quant bf16 instead")
        mode, cls = "bf16", BF16Linear
    fused = []
    for blk in model.blocks:   # [q | k_local | k_global | v_local | v_global]: per-row formats, so fusing is exact
        a = blk.attn
        Hk, D = model.cfg.n_kv_heads, model.cfg.head_dim
        wl = a.w_kv_local.weight.detach()
        ks, vs = [wl[:Hk * D]], [wl[Hk * D:]]
        if a.mode == "full":
            wg = a.w_kv_global.weight.detach()
            ks.append(wg[:Hk * D])
            vs.append(wg[Hk * D:])
        lin = torch.nn.Linear(model.cfg.d_model, 1, bias=False)
        lin.weight = torch.nn.Parameter(torch.cat([a.w_q.weight.detach()] + ks + vs).float(), requires_grad=False)
        fused.append(cls(lin))
    targets = [(name, mod) for name, mod in model.named_modules() if isinstance(mod, torch.nn.Linear)
               and (head or name != "head")]
    for name, mod in targets:
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        setattr(parent, name.rsplit(".", 1)[-1], cls(mod))
    model._cpu_fast = _CpuFast(model, fused)
    model._quant = mode
    return model


class _CpuFast:
    """What _decode_fast needs besides the model: the fused QKV weights (one weight stream and one kernel call per
    layer instead of 2-3; the q/k heads contiguous, so one norm and one RoPE matmul cover them), per-head norm
    divisors, the RoPE helper matrices and a bf16 activation buffer. The fused weights duplicate w_q/w_kv_* (prefill
    keeps using those): +12 MB in int8."""

    def __init__(self, model: TinyAgentLM, fused):
        c = model.cfg
        H, Hk, D = c.n_heads, c.n_kv_heads, c.head_dim
        self.qkv = fused
        self.nqk, self.ndiv = [], []
        for blk in model.blocks:
            nk = 2 if blk.attn.mode == "full" else 1
            self.nqk.append(H + Hk * nk)
            # q heads: divide by |q| (= rms_norm(q) / sqrt(D): the attention's 1/sqrt(D) folded in);
            # k heads: by |k| / sqrt(D) (= rms_norm(k), what the cache holds)
            self.ndiv.append(torch.cat([torch.ones(H, 1), torch.full((Hk * nk, 1), D ** -0.5)]))
        self.head = model.head if hasattr(model.head, "mm") else _FullMM(model.head)
        self.inv_sqrt_d = c.d_model ** -0.5
        half = D // 2
        self.eye = torch.eye(D)
        P = torch.zeros(D, D)
        j = torch.arange(half)
        P[j + half, j] = -1
        P[j, j + half] = 1
        self.P = P
        self.act = getattr(type(fused[0]), "ACT_DTYPE", torch.bfloat16)   # what mm() takes (tests: float64)
        self.hb = torch.empty(1, c.d_model, dtype=self.act)


def _decode_fast(model, cache: KVCache, idx, L: int, fs: _CpuFast):
    """_decode_one for a quantize_cpu model, with far fewer ops (at batch 1 on a CPU every op costs a few us of
    dispatch, about as much as the weights take to stream): fused QKV, norms as x / |x| (one vector_norm), RoPE as one
    64x64 matmul for all q/k heads, the token's K/V written into the cache first so attention reads the cache directly
    (no K/V concatenation), each Linear fed from one bf16 buffer and added into the fp32 residual in place.
    Same math as _decode_one up to float rounding (norm epsilon, summation order)."""
    c = model.cfg
    H, Hk, D, W = c.n_heads, c.n_kv_heads, c.head_dim, c.swa_window
    rep = H // Hk
    bf = fs.act
    vnorm = torch.linalg.vector_norm
    addrs = {}
    if model.engram_modules():
        max_n = max(c.engram_orders)
        prev = cache.tokens[:1, max(0, L - (max_n - 1)):L]
        if prev.size(1) < max_n - 1:
            prev = torch.cat([prev.new_zeros(1, max_n - 1 - prev.size(1)), prev], dim=1)
        cids = model.cid_map[torch.cat([prev, idx], dim=1)]
        for i, b in enumerate(model.blocks):
            if b.engram is not None:
                addrs[i] = b.engram.addresses(cids)[:, max_n - 1:]
    cache.tokens[0, L] = idx[0, 0]
    slot = L % W
    cache.lpos[0, slot] = L
    lp = cache.lpos[0, :W]
    ok = (lp >= 0) & (L - lp < W)                     # ring slots holding a position of this token's window
    lbias = None if bool(ok.all()) else torch.zeros(W).masked_fill_(~ok, float("-inf"))
    cs, sn = model.rope.cos[L], model.rope.sin[L]
    R = fs.eye * torch.cat([cs, cs]) + fs.P * torch.cat([sn, sn])   # x @ R == RoPE(x) at position L
    lk, lv, gk, gv = cache.lk[:, 0], cache.lv[:, 0], cache.gk[:, 0], cache.gv[:, 0]
    lk_new, lv_new, gk_new, gv_new = lk[:, :, slot], lv[:, :, slot], gk[:, :, L], gv[:, :, L]
    Vl, Vg_all = lv[:, :, :W], gv[:, :, :L + 1]
    KlT, KgT_all = lk[:, :, :W].transpose(-1, -2), gk[:, :, :L + 1].transpose(-1, -2)
    kdt = cache.gk.dtype
    hb, inv = fs.hb, fs.inv_sqrt_d
    x = model.embed.weight[idx[0]]                                               # (1, d), a copy
    xdt = x.dtype
    if R.dtype != xdt:
        R = R.to(xdt)
    group = -1
    for li, blk in enumerate(model.blocks):
        if blk.engram is not None:
            x = x + blk.engram(x[None], addrs[li])[0]
        torch.div(x, vnorm(x, dim=-1, keepdim=True).mul_(inv), out=hb)           # rms_norm(x), in bf16
        heads = fs.qkv[li].mm(hb).view(-1, D)                                    # (nqk + nv, D), bf16
        nqk = fs.nqk[li]
        qk = heads[:nqk]                                                         # / an fp32 norm: fp32 result
        qk = (qk / vnorm(qk, dim=-1, keepdim=True, dtype=xdt).mul_(fs.ndiv[li])) @ R   # normed + rotated q / k
        q = qk[:H].view(Hk, rep, D)
        lk_new[li].copy_(qk[H:H + Hk])
        lv_new[li].copy_(heads[nqk:nqk + Hk])                                    # copy_ casts to the cache dtype
        if blk.attn.mode == "full":
            group += 1
            gk_new[group].copy_(qk[H + Hk:nqk])
            gv_new[group].copy_(heads[nqk + Hk:])
            KgT, Vg = KgT_all[group], Vg_all[group]
        if q.dtype != kdt:
            q = q.to(kdt)
        sl = torch.bmm(q, KlT[li])                                               # (Hk, rep, W) local
        if lbias is not None:
            sl.add_(lbias)
        p = torch.softmax(torch.cat([torch.bmm(q, KgT), sl], dim=-1), dim=-1, dtype=torch.float32)
        p = p.to(kdt)
        y = torch.bmm(p[..., :L + 1], Vg).baddbmm_(p[..., L + 1:], Vl[li])        # (Hk, rep, D)
        x.add_(blk.attn.w_o.mm(y.view(1, H * D).to(bf)))
        torch.div(x, vnorm(x, dim=-1, keepdim=True).mul_(inv), out=hb)
        u = blk.mlp.up.mm(hb)
        torch.relu_(u)
        x.add_(blk.mlp.down.mm(u.mul_(u)))
    cache.lens[0] += 1
    cache.lens_cpu[0] += 1
    torch.div(x, vnorm(x, dim=-1, keepdim=True).mul_(inv), out=hb)
    logits = fs.head.mm(hb).float()
    sc = c.logit_softcap
    return sc * torch.tanh(logits / sc)
