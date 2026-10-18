"""Optimizers, following DeepSeek V4.1 §2.5:
  * Muon (Nesterov, decoupled WD) for backbone matrices; head-wise for query projections.
  * Momentum + Sinkhorn balancing (Algorithm 1) for token embedding, LM head and Engram tables.
  * AdamW for anything else (none by default: norms are parameter-free).
Update RMS for Muon is matched to AdamW via 0.2*sqrt(max(m, n)) (Moonlight), and the Sinkhorn
update uses gamma=0.18, so one learning rate works across groups.
"""
from __future__ import annotations

import math

import torch


@torch.no_grad()
def newton_schulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Quintic NS orthogonalization; works on (..., m, n)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    transpose = X.size(-2) > X.size(-1)
    if transpose:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transpose:
        X = X.mT
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=3e-3, momentum=0.95, weight_decay=0.0, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(p, dtype=torch.float32)
                buf = st["buf"]
                grad = p.grad.float()
                buf.lerp_(grad, 1 - g["momentum"])
                upd = grad.lerp(buf, g["momentum"])  # Nesterov
                heads = getattr(p, "_muon_heads", None)
                if heads:  # head-wise: orthogonalize each head's (head_dim, d_model) slice separately
                    u = upd.view(heads, -1, upd.size(-1))
                    u = newton_schulz(u, g["ns_steps"]).view_as(upd)
                    m, n = u.size(0) // heads, u.size(1)
                else:
                    u = newton_schulz(upd, g["ns_steps"])
                    m, n = u.shape
                scale = 0.2 * math.sqrt(max(m, n))
                if g["weight_decay"]:
                    p.mul_(1 - g["lr"] * g["weight_decay"])
                p.add_(u.to(p.dtype), alpha=-g["lr"] * scale)


class SinkhornMomentum(torch.optim.Optimizer):
    """V4.1 Algorithm 1: Nesterov momentum, mask near-zero rows, K (odd) alternating row/col L2
    normalizations ending on rows, scale by sqrt(n) for unit row RMS, lr correction gamma."""

    def __init__(self, params, lr=3e-3, momentum=0.95, K=5, gamma=0.18, tau=0.1, eps=1e-8):
        assert K % 2 == 1
        super().__init__(params, dict(lr=lr, momentum=momentum, K=K, gamma=gamma, tau=tau, eps=eps))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(p, dtype=torch.float32)
                buf = st["buf"]
                grad = p.grad.float()
                buf.lerp_(grad, 1 - g["momentum"])
                U = grad.lerp(buf, g["momentum"])
                rho = U.norm(dim=1)
                # Mean over rows that carry signal at all: huge hash tables are mostly untouched,
                # and a mean over all rows would let stale decayed momentum through the mask.
                nz = rho > 0
                rbar = rho[nz].mean() if nz.any() else rho.new_zeros(())
                U = U * (rho > g["tau"] * rbar)[:, None]
                for k in range(1, g["K"] + 1):
                    if k % 2 == 1:
                        U = U / (U.norm(dim=1, keepdim=True) + g["eps"])
                    else:
                        U = U / (U.norm(dim=0, keepdim=True) + g["eps"])
                U = U * math.sqrt(U.size(1))
                p.add_(U.to(p.dtype), alpha=-g["lr"] * g["gamma"])


def build_optimizers(model, lr=3e-3, weight_decay=0.0):
    sink, muon, adam = [], [], []
    sink_ids = {id(model.embed.weight), id(model.head.weight)}
    for m in model.engram_modules():
        sink_ids.add(id(m.table.weight))
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if id(p) in sink_ids:
            sink.append(p)
        elif p.ndim == 2:
            muon.append(p)
        else:
            adam.append(p)
    opts = [Muon(muon, lr=lr, weight_decay=weight_decay), SinkhornMomentum(sink, lr=lr)]
    if adam:
        opts.append(torch.optim.AdamW(adam, lr=lr, betas=(0.9, 0.95), weight_decay=0.0))
    for o in opts:
        for g in o.param_groups:
            g["base_lr"] = g["lr"]
    return opts


def wsd_lr(step: int, total: int, warmup: int = 200, decay_frac: float = 0.2) -> float:
    """Warmup-stable-decay multiplier. Decay can also be triggered early by passing a smaller total."""
    if step < warmup:
        return (step + 1) / warmup
    decay_start = int(total * (1 - decay_frac))
    if step < decay_start:
        return 1.0
    return max(0.0, (total - step) / max(1, total - decay_start))
