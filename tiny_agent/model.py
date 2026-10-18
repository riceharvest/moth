"""Tiny dense decoder with V4.1-style shared global KV + per-layer sliding window, and Engram.

Attention (simplified CSA2, compression ratio 1, no indexer):
  * Layers are grouped (``kv_group`` layers per group). The first layer of a group is in
    "Full" mode: it projects the global K/V for the whole group. The other layers are in
    "Reuse" mode: they only compute their own queries and reuse the group's global K/V.
  * Every layer also projects its own local K/V, visible only inside a ``swa_window`` window.
  * Each query does ONE softmax over the union of global (causal) and local (windowed) keys,
    as in V4.1. Implemented with FlexAttention over the concatenated [global | local] keys.
  So the decode KV cache is: one global K/V per group (grows with context) + one local K/V
  ring buffer of ``swa_window`` entries per layer (constant size).

Engram (Cheng et al. 2026, as used in V4.1): hashed n-gram embeddings over a compressed
token id space, fused into the residual stream with a context-aware sigmoid gate. The short
causal conv is omitted, as in V4.1.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

# FlexAttention (the training / full-forward path) is imported on first use: importing it pulls in sympy (~0.4 s and
# ~30 MB at every start), and serving (tiny_agent.generate) never runs it.
flex_attention = create_block_mask = None


def _load_flex():
    global flex_attention, create_block_mask
    if flex_attention is None:
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    return flex_attention


# B70-tuned FlexAttention backward tiles (measured in tiny_moe/bench.py: dense L 50.8k -> 60.7k tok/s);
# forward keeps the defaults. Only speed changes, not results.
FLEX_OPTS = dict(bwd_num_warps=16, bwd_BLOCK_M1=32, bwd_BLOCK_N1=128, bwd_BLOCK_M2=128, bwd_BLOCK_N2=32)


@dataclass
class ModelConfig:
    vocab_size: int = 32768
    d_model: int = 640
    n_layers: int = 16
    n_heads: int = 10
    head_dim: int = 64
    n_kv_heads: int = 2          # heads of both the shared global KV and the per-layer local KV
    kv_group: int = 4            # layers per shared-global-KV group (1 Full + kv_group-1 Reuse)
    swa_window: int = 128
    ffn_mult: float = 4.0        # ReLU^2 MLP hidden = ffn_mult * d_model
    rope_base: float = 10000.0
    max_seq_len: int = 8192
    logit_softcap: float = 15.0
    # Engram
    engram_layers: tuple[int, ...] = ()      # e.g. (1,) ; empty disables Engram
    engram_orders: tuple[int, ...] = (2, 3)
    engram_heads: int = 8
    engram_head_dim: int = 64
    engram_rows_per_head: int = 262144       # rounded up to distinct primes per (order, head)

    def layer_mode(self, i: int) -> str:
        return "full" if i % self.kv_group == 0 else "reuse"

    @property
    def ffn_hidden(self) -> int:
        return int(self.ffn_mult * self.d_model) // 64 * 64


def rms_norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


class Rotary(nn.Module):
    def __init__(self, dim: int, max_len: int, base: float):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_len, dtype=torch.float32)
        f = torch.outer(t, inv)
        self.register_buffer("cos", f.cos(), persistent=False)
        self.register_buffer("sin", f.sin(), persistent=False)

    def forward(self, x: torch.Tensor, pos: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, H, T, D). pos: optional (T,) absolute positions (decode).
        T = x.size(-2)
        cos = self.cos[:T] if pos is None else self.cos[pos]
        sin = self.sin[:T] if pos is None else self.sin[pos]
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


def next_prime(n: int) -> int:
    def is_prime(k: int) -> bool:
        if k < 2:
            return False
        if k % 2 == 0:
            return k == 2
        r = int(k**0.5)
        return all(k % d for d in range(3, r + 1, 2))

    while not is_prime(n):
        n += 1
    return n


class Engram(nn.Module):
    """Hashed n-gram memory with context-aware gating (one module = one insertion layer)."""

    def __init__(self, cfg: ModelConfig, seed: int):
        super().__init__()
        self.cfg = cfg
        g = torch.Generator().manual_seed(seed)
        primes, p = [], cfg.engram_rows_per_head
        for _ in cfg.engram_orders:
            for _ in range(cfg.engram_heads):
                p = next_prime(p + 1)
                primes.append(p)
        self.register_buffer("primes", torch.tensor(primes, dtype=torch.int64), persistent=False)
        offs = torch.tensor([0] + primes[:-1], dtype=torch.int64).cumsum(0)
        self.register_buffer("offsets", offs, persistent=False)
        max_n = max(cfg.engram_orders)
        # odd multipliers per (order, head, position-in-ngram)
        mult = torch.randint(1, 2**30, (len(cfg.engram_orders), cfg.engram_heads, max_n), generator=g) * 2 + 1
        self.register_buffer("mult", mult.to(torch.int64), persistent=False)
        self.table = nn.Embedding(sum(primes), cfg.engram_head_dim)
        mem_dim = len(cfg.engram_orders) * cfg.engram_heads * cfg.engram_head_dim
        self.w_k = nn.Linear(mem_dim, cfg.d_model, bias=False)
        self.w_v = nn.Linear(mem_dim, cfg.d_model, bias=False)
        nn.init.normal_(self.table.weight, std=0.02)
        nn.init.zeros_(self.w_v.weight)

    @torch.no_grad()
    def addresses(self, cids: torch.Tensor) -> torch.Tensor:
        """cids: (B, T) compressed token ids -> (B, T, n_orders*heads) global row indices."""
        B, T = cids.shape
        max_n = max(self.cfg.engram_orders)
        pad = cids.new_full((B, max_n - 1), 0)
        ext = torch.cat([pad, cids], dim=1).long()
        # shifted[k] = token at t-k
        shifted = torch.stack([ext[:, max_n - 1 - k: max_n - 1 - k + T] for k in range(max_n)], dim=-1)  # B,T,max_n
        out = []
        for oi, n in enumerate(self.cfg.engram_orders):
            m = self.mult[oi, :, :n]                          # heads, n
            mix = shifted[..., None, 0] * m[:, 0]             # B,T,heads
            for k in range(1, n):
                mix = mix ^ (shifted[..., None, k] * m[:, k])
            out.append(mix)
        mix = torch.cat(out, dim=-1)                          # B,T,n_orders*heads
        return torch.remainder(mix, self.primes) + self.offsets

    def forward(self, h: torch.Tensor, addr: torch.Tensor) -> torch.Tensor:
        mem = self.table(addr).flatten(-2)                    # B,T,mem_dim
        mem = mem.to(h.dtype)
        k = self.w_k(mem)
        v = self.w_v(mem)
        g = (rms_norm(h) * rms_norm(k)).sum(-1, keepdim=True) / math.sqrt(h.size(-1))
        g = torch.sigmoid(g.abs().clamp_min(1e-6).sqrt() * g.sign())
        return g * v


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, mode: str):
        super().__init__()
        self.cfg, self.mode = cfg, mode
        H, Hk, D = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.w_q = nn.Linear(cfg.d_model, H * D, bias=False)
        self.w_q._muon_heads = H  # head-wise Muon (V4.1)
        self.w_kv_local = nn.Linear(cfg.d_model, 2 * Hk * D, bias=False)
        if mode == "full":
            self.w_kv_global = nn.Linear(cfg.d_model, 2 * Hk * D, bias=False)
        self.w_o = nn.Linear(H * D, cfg.d_model, bias=False)
        nn.init.zeros_(self.w_o.weight)

    def _kv(self, lin: nn.Linear, x: torch.Tensor, rope: Rotary, pos=None):
        B, T, _ = x.shape
        Hk, D = self.cfg.n_kv_heads, self.cfg.head_dim
        k, v = lin(x).view(B, T, 2, Hk, D).permute(2, 0, 3, 1, 4)
        return rope(rms_norm(k), pos), v

    def forward(self, x, rope: Rotary, shared: dict, block_mask):
        B, T, _ = x.shape
        H, D = self.cfg.n_heads, self.cfg.head_dim
        q = self.w_q(x).view(B, T, H, D).transpose(1, 2)
        q = rope(rms_norm(q))
        if self.mode == "full":
            shared["k"], shared["v"] = self._kv(self.w_kv_global, x, rope)
        kl, vl = self._kv(self.w_kv_local, x, rope)
        k = torch.cat([shared["k"], kl], dim=2)
        v = torch.cat([shared["v"], vl], dim=2)
        dt = v.dtype  # rms_norm autocasts to fp32; flex backward needs one dtype
        y = (flex_attention or _load_flex())(q.to(dt), k.to(dt), v, block_mask=block_mask, enable_gqa=True,
                                             kernel_options=FLEX_OPTS if q.device.type == "xpu" else None)
        return self.w_o(y.transpose(1, 2).reshape(B, T, H * D))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.up = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.down = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)
        nn.init.zeros_(self.down.weight)

    def forward(self, x):
        return self.down(F.relu(self.up(x)).square())


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, i: int):
        super().__init__()
        self.attn = Attention(cfg, cfg.layer_mode(i))
        self.mlp = MLP(cfg)
        self.engram = Engram(cfg, seed=1000 + i) if i in cfg.engram_layers else None

    def forward(self, x, rope, shared, block_mask, addr):
        if self.engram is not None:
            x = x + self.engram(x, addr)
        x = x + self.attn(rms_norm(x), rope, shared, block_mask)
        return x + self.mlp(rms_norm(x))


def make_block_mask(doc: torch.Tensor, window: int):
    """doc: (B, T) int document ids. Keys are [global(T) | local(T)]."""
    B, T = doc.shape

    def mask_mod(b, h, q, kv):
        is_local = kv >= T
        j = torch.where(is_local, kv - T, kv)
        same = doc[b, q] == doc[b, j]
        causal = q >= j
        in_win = (q - j) < window
        return same & causal & (~is_local | in_win)

    _load_flex()
    return create_block_mask(mask_mod, B, None, T, 2 * T, device=doc.device)


class TinyAgentLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layers)])
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.rope = Rotary(cfg.head_dim, cfg.max_seq_len, cfg.rope_base)
        # token id -> compressed id for Engram (identity until a tokenizer map is loaded)
        self.register_buffer("cid_map", torch.arange(cfg.vocab_size), persistent=True)
        nn.init.normal_(self.embed.weight, std=1.0)
        nn.init.zeros_(self.head.weight)

    def engram_modules(self):
        return [b.engram for b in self.blocks if b.engram is not None]

    def forward(self, idx: torch.Tensor, doc: torch.Tensor, block_mask=None, targets=None):
        """idx, doc: (B, T). Returns softcapped float32 logits (B, T, V), or the mean CE loss
        over targets != -1 when targets is given (lets torch.compile fuse head + softcap + CE)."""
        if block_mask is None:
            block_mask = make_block_mask(doc, self.cfg.swa_window)
        addr = None
        ems = self.engram_modules()
        if ems:
            cids = self.cid_map[idx]
        x = self.embed(idx)
        shared: dict = {}
        for b in self.blocks:
            a = b.engram.addresses(cids) if b.engram is not None else None
            x = b(x, self.rope, shared, block_mask, a)
        logits = self.head(rms_norm(x)).float()
        c = self.cfg.logit_softcap
        logits = c * torch.tanh(logits / c)
        if targets is None:
            return logits
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)

    def param_counts(self) -> dict:
        emb = self.embed.weight.numel() + self.head.weight.numel()
        eng = sum(p.numel() for m in self.engram_modules() for p in [m.table.weight])
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "embedding": emb, "engram_tables": eng, "backbone": total - emb - eng}

    def kv_bytes_per_token(self, context: int, bytes_per_value: float = 2.0) -> float:
        """Decode KV cache per token of context, amortized (global grows, local is a fixed ring)."""
        c = self.cfg
        groups = math.ceil(c.n_layers / c.kv_group)
        glob = groups * 2 * c.n_kv_heads * c.head_dim
        local = c.n_layers * 2 * c.n_kv_heads * c.head_dim * min(c.swa_window, context) / max(context, 1)
        return (glob + local) * bytes_per_value


def dense_reference_attention_mask(doc: torch.Tensor, window: int) -> torch.Tensor:
    """Boolean (B, 1, T, 2T) mask equal to make_block_mask's mask_mod, for tests."""
    B, T = doc.shape
    q = torch.arange(T, device=doc.device)[:, None]
    kv = torch.arange(2 * T, device=doc.device)[None, :]
    is_local = kv >= T
    j = torch.where(is_local, kv - T, kv)                       # (1, 2T)
    same = doc[:, :, None] == doc[:, j[0]][:, None, :]          # (B, T, 2T)
    m = same & (q >= j) & (~is_local | ((q - j) < window))
    return m[:, None]
