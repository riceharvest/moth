"""XS inference: tiny-moe's forward pass for batch-1 chat, against a CSA2 KV cache, on the CPU first.

Same math as tiny_moe/model.py + tiny_moe/generate.py (tests/test_xs_engine.py checks it against both), restricted
to what the served model uses: one conversation (batch 1), lean config by default (attnres / encdec / indexer /
tied or FP4 KV refused at load by moth.xs.config), gated_norm and gated_attn supported.

Per layer: RMSNorm (gain) [-> GatedNorm gate] -> CSA2 attention -> + residual -> RMSNorm -> LatentMoE -> + residual.
  attention   q (QK-norm, partial RoPE) attends ONE softmax over [shared global KV of the layer's group (causal) |
              its own local KV in a swa_window ring]; the first swa_only layers have the ring only. PKO layers see
              keys whose stationary (unroped) dims come from the previous position; attention sink (softmax
              denominator + exp(sink_h)) and XSA (the output's component along the token's own value removed).
  MoE         hash-routed first hash_layers layers (fixed token -> experts table, uniform weights), then a
              sqrt-softplus router (fp32; the selection bias e_bias only steers which experts, the weights come from
              the unbiased scores), top-k renormalized * routed_scale; SwiGLU experts with V4 clamping; one shared
              expert.

The cache (XSCache): global K/V per layer group (written by the group's Full layer) for the whole context, plus a
shifted copy of the global keys for PKO readers (written incrementally, so a PKO layer never re-shifts the whole
cache per token); per layer a ring of swa_window local K/V slots (PKO layers store keys already shifted, lstat keeps
each slot's unshifted stationary part for the next position).

Two paths, same numbers (up to float rounding):
  forward(ids, cache)   C new tokens at once (prefill, a new message); tokens are batched per expert (a loop over
                        the experts that got tokens, never over tokens).
  step(tok, cache)      one token, lean: one fused q|k|v GEMV per layer, RoPE as one small matmul, the token's K/V
                        written into the cache first so attention reads the cache in place, the top-k experts as
                        k GEMV pairs on per-expert weight views and one elementwise SwiGLU over all k.
                        step() runs _step_fast for the XS shape: the same kernels in the same order as step_ref (bitwise
                        equal logits) with less dispatch around them (~1.1k ops per token instead of ~1.8k on XS);
                        step_ref is the readable version and the fallback for other configs.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from moth.xs.config import XSConfig
from moth.xs.quant import Experts, QLinear, act_dtype, gemv_fn

EPS = 1e-6
NEG = torch.finfo(torch.float32).min / 2
CHUNK = 256   # prefill tokens per forward pass: bounds the (C, context) score tensors


def bigram_row(prev: int, tok: int, rows: int) -> int:
    """tiny_moe.model.TinyMoE.bigram_ids for one pair (int64 math in Python ints: same result)."""
    return ((prev * 1000003) ^ (tok * 2654435761)) % rows


def hash_table(vocab: int, experts: int, k: int, layer: int) -> torch.Tensor:
    """tiny_moe.moe.LatentMoE's fixed token -> experts table (a non-persistent buffer there). moth.xs.weights.export
    stores it in the safetensors; this rebuild is only for checkpoints without it (torch's CPU RNG, seed 7919 + layer)."""
    g = torch.Generator().manual_seed(7919 + layer)
    return torch.rand(vocab, experts, generator=g).argsort(1)[:, :k]


def swiglu(h: torch.Tensor, f: int, clamp: float) -> torch.Tensor:
    g, u = h[..., :f], h[..., f:]
    if clamp:
        g, u = g.clamp(max=clamp), u.clamp(-clamp, clamp)
    return F.silu(g) * u


class XSCache:
    """KV cache of one conversation (batch 1). lens_cpu[0] mirrors len (moth.serve reads it)."""

    def __init__(self, model: "XSModel", max_len: int, device="cpu", dtype=torch.float32):
        c = model.cfg
        Hk, D, W, r = c.n_kv_heads, c.head_dim, c.swa_window, c.rope_dims
        G = max(1, c.n_groups)
        self.cfg, self.max_len, self.device, self.dtype = c, max_len, torch.device(device), dtype
        self.gk = torch.zeros(G, Hk, max_len, D, device=device, dtype=dtype)
        self.gv = torch.zeros_like(self.gk)
        self.gks = torch.zeros_like(self.gk) if model.pko_groups else None
        self.lk = torch.zeros(c.n_layers, Hk, W, D, device=device, dtype=dtype)
        self.lv = torch.zeros_like(self.lk)
        self.lstat = torch.zeros(c.n_layers, Hk, W, D - r, device=device, dtype=dtype)
        self.lpos = torch.full((W,), -1, dtype=torch.long, device=device)
        self.tokens = torch.zeros(max_len, dtype=torch.long)
        # per-layer / per-group views made once (each tensor[i] in the decode loop is an op, ~1 us at batch 1)
        self.lk_l, self.lv_l, self.lstat_l = list(self.lk), list(self.lv), list(self.lstat)
        self.gk_g, self.gv_g = list(self.gk), list(self.gv)
        self.gks_g = list(self.gks) if self.gks is not None else None
        self.pko_layers = [i for i in range(c.n_layers) if c.pko_layer(i)]
        self._fv = None
        self.len = 0
        self.last = c.vocab_size        # the previous token (bigram embedding); vocab_size = document start
        self.lens_cpu = torch.zeros(1, dtype=torch.long)

    def fast_views(self) -> "_FastViews":
        """Views XSModel.step writes and reads through, made once per cache (a few ms; reset() keeps them valid)."""
        if self._fv is None:
            self._fv = _FastViews(self)
        return self._fv

    def reset(self, rows=None) -> None:
        self.len = 0
        self.last = self.cfg.vocab_size
        self.lpos.fill_(-1)
        self.lens_cpu[0] = 0

    def nbytes(self) -> int:
        ts = [self.gk, self.gv, self.lk, self.lv, self.lstat] + ([self.gks] if self.gks is not None else [])
        return sum(t.numel() * t.element_size() for t in ts)


class _FastViews:
    """Per layer and ring slot: the (Hk, D) K / V rows (PKO layers: also the roped / stationary halves of K and the
    lstat rows); keys transposed once (Hk, D, len) so scores need no transpose op."""

    def __init__(self, cache: XSCache):
        r = cache.cfg.rope_dims
        self.lk = [list(t.unbind(1)) for t in cache.lk_l]
        self.lv = [list(t.unbind(1)) for t in cache.lv_l]
        self.lkT = [t.transpose(-1, -2) for t in cache.lk_l]
        self.lk_r, self.lk_s, self.lst = {}, {}, {}
        for li in cache.pko_layers:
            self.lk_r[li] = [t[:, :r] for t in self.lk[li]]
            self.lk_s[li] = [t[:, r:] for t in self.lk[li]]
            self.lst[li] = list(cache.lstat_l[li].unbind(1))
        self.gkT = [t.transpose(-1, -2) for t in cache.gk_g]
        self.gksT = [t.transpose(-1, -2) for t in cache.gks_g] if cache.gks_g is not None else None


class _Layer:
    pass


class XSModel:
    """Weights of an XS checkpoint in one weight format (moth.xs.quant.MODES) plus the forward passes.
    sd: tiny_moe.model.TinyMoE's state dict names (any float dtype; tensors may be memory-mapped)."""

    def __init__(self, cfg: XSConfig, sd: dict, mode: str = "fp32", device="cpu"):
        self.cfg, self.mode, self.device = cfg, mode, torch.device(device)
        c = cfg
        H, Hk, D = c.n_heads, c.n_kv_heads, c.head_dim
        self.act = act_dtype(mode)
        dev = self.device
        lookup_dt = torch.float32 if mode == "fp32" else torch.bfloat16
        emb = sd["embed"]
        self.head = QLinear(emb, mode, dev)                 # tied LM head (int8: its own int8 copy, ~27% of a token)
        # input lookup (bf16 in the fast modes: rows are upcast); fp32 / bf16 share the head's tensor
        self.embed = self.head.w if mode != "int8" else emb.to(device=dev, dtype=lookup_dt)
        self.bigram = sd["bigram"].to(device=dev, dtype=lookup_dt) if c.bigram_rows else None
        self.final_norm = self._norm(sd, "final_norm")
        r = c.rope_dims
        inv = 1.0 / (c.rope_base ** (torch.arange(0, r, 2, dtype=torch.float32) / r))
        f = torch.outer(torch.arange(c.max_seq_len, dtype=torch.float32), inv)
        self.cos, self.sin = f.cos().to(dev), f.sin().to(dev)
        self._ar = torch.arange(r // 2, device=dev)
        self.layers, self.pko_groups = [], set()
        g = -1
        for i in range(c.n_layers):
            p = f"layers.{i}."
            L = _Layer()
            L.mode, L.pko = c.layer_mode(i), c.pko_layer(i)
            L.owns = c.owns_global_kv(i)
            if L.owns:
                g += 1
            L.group = g if L.mode != "swa" else -1
            if L.pko and L.group >= 0:
                self.pko_groups.add(L.group)
            L.attn_norm, L.mlp_norm = self._norm(sd, p + "attn_norm"), self._norm(sd, p + "mlp_norm")
            wqg = sd[p + "attn.w_qg.weight"]
            q, gate = (wqg[:H * D], wqg[H * D:]) if c.gated_attn else (wqg, None)
            wl = sd[p + "attn.w_kv_local.weight"]
            parts_k, parts_v = [wl[:Hk * D]], [wl[Hk * D:]]
            if L.owns:
                wg = sd[p + "attn.w_kv_global.weight"]
                parts_k.append(wg[:Hk * D])
                parts_v.append(wg[Hk * D:])
            # fused rows: q | k_local | k_global | v_local | v_global | gate  (per-row formats: fusing is exact)
            rows = [q] + parts_k + parts_v + ([gate] if gate is not None else [])
            L.qkv = QLinear(torch.cat([t.float() for t in rows]), mode, dev)
            L.nq = H + Hk * len(parts_k)                    # q and k heads (normed + roped together)
            L.nkv = Hk * len(parts_v)
            L.wo = QLinear(sd[p + "attn.w_o.weight"], mode, dev)
            if c.qk_norm:
                gains = [sd[p + "attn.q_norm.weight"].float()] * H + [sd[p + "attn.kl_norm.weight"].float()] * Hk
                if L.owns:
                    gains += [sd[p + "attn.kg_norm.weight"].float()] * Hk
                L.gain = torch.stack(gains)
                L.qk_gain = L.gain.clone()
            else:
                L.gain = None
                L.qk_gain = torch.ones(L.nq, D)
            L.qk_gain[:H] *= D ** -0.5                      # step(): the attention's 1/sqrt(D) folded into q
            L.gain = None if L.gain is None else L.gain.to(dev)
            L.qk_gain = L.qk_gain.to(dev)
            L.sink = sd[p + "attn.sink"].float().to(dev) if c.attn_sink else None
            L.sinkcol = L.sink.view(Hk, H // Hk, 1) if L.sink is not None else None   # step(): one extra logit
            m = p + "moe."
            L.hashed = c.is_hash_layer(i)
            if L.hashed:
                t = sd.get(m + "hash_idx")
                L.hash_idx = (t if t is not None else hash_table(c.vocab_size, c.n_routed, c.top_k_eff, i)).long()
                L.router = None
            else:
                L.router = sd[m + "router"].float().to(dev).contiguous()
                L.hash_idx = None
            if L.hashed:
                L.e_bias = None
            elif (m + "e_bias") in sd:
                L.e_bias = sd[m + "e_bias"].float().to(dev)
            else:   # an EMA file without router buffers: export it with moth.xs.weights (takes them from final.pt)
                raise KeyError(f"{m}e_bias missing: the router's selection bias (an EMA-only checkpoint?)")
            L.experts = Experts(sd[m + "w1"], sd[m + "w2"], mode, dev)
            L.down = QLinear(sd[m + "down.weight"], mode, dev) if c.latent_alpha > 1 else None
            L.up = QLinear(sd[m + "up.weight"], mode, dev) if c.latent_alpha > 1 else None
            L.shared = [(QLinear(sd[f"{m}shared.{j}.w_in.weight"], mode, dev),
                         QLinear(sd[f"{m}shared.{j}.w_out.weight"], mode, dev)) for j in range(c.n_shared)]
            # step(): one shared expert of width sf = n * expert_dff runs as n pseudo-experts next to the routed
            # ones (its w_in rows reordered to [gate_0 | up_0 | gate_1 | up_1 ...] blocks of expert_dff), so all of
            # them share one SwiGLU and one weighted sum. Same math; only for this (XS) shape.
            sf, f = c.shared_ffn, c.expert_dff
            L.fused_shared = None
            if c.n_shared == 1 and c.latent_alpha == 1 and sf % f == 0:
                w = sd[f"{m}shared.0.w_in.weight"].float()
                g_, u_ = w[:sf].view(sf // f, f, -1), w[sf:].view(sf // f, f, -1)
                L.fused_shared = (QLinear(torch.cat([g_, u_], dim=1).reshape(2 * sf, -1), mode, dev), sf // f)
            self.layers.append(L)
        self._prepare_fast()

    def _norm(self, sd, name):
        """RMSNorm gain (+ GatedNorm's low-rank gate when the config has it): (w, down, up)."""
        w = sd[name + ".norm.weight"].float().to(self.device)
        if self.cfg.gated_norm:
            return (w, sd[name + ".down"].float().to(self.device), sd[name + ".up"].float().to(self.device))
        return (w, None, None)

    @staticmethod
    def norm(x, nw):
        w, down, up = nw
        y = F.rms_norm(x, (x.size(-1),), w, EPS)
        if down is not None:
            y = y * (2.0 * torch.sigmoid(F.linear(F.silu(F.linear(y, down)), up)))
        return y

    def new_cache(self, max_len: int, dtype=torch.float32) -> XSCache:
        return XSCache(self, max_len, self.device, dtype)

    def nbytes(self) -> int:
        n = self.head.nbytes() + self.embed.numel() * self.embed.element_size()
        if self.bigram is not None:
            n += self.bigram.numel() * self.bigram.element_size()
        for L in self.layers:
            n += L.qkv.nbytes() + L.wo.nbytes() + L.experts.nbytes() + sum(a.nbytes() + b.nbytes() for a, b in L.shared)
        return n

    # ------------------------------------------------------------------------------------------ embeddings
    def _embed(self, ids: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        x = F.embedding(ids, self.embed).float()
        if self.bigram is not None:
            rows = ((prev * 1000003) ^ (ids * 2654435761)) % c.bigram_rows
            x = x + F.embedding(rows, self.bigram).float()
        return x * c.emb_scale if c.emb_scale != 1.0 else x

    def _rope(self, x, cos, sin):
        """Partial RoPE on (..., C, nh, D) with cos/sin (C, r/2): first and second halves of the rotated dims."""
        r = self.cfg.rope_dims
        h = r // 2
        x1, x2 = x[..., :h], x[..., h:r]
        c, s = cos[:, None, :], sin[:, None, :]
        return torch.cat((x1 * c - x2 * s, x1 * s + x2 * c, x[..., r:]), dim=-1)

    # ------------------------------------------------------------------------------------------ chunk forward
    @torch.no_grad()
    def forward(self, ids, cache: XSCache, all_logits: bool = False) -> torch.Tensor:
        """Feed ids (list or 1-D tensor) after what the cache holds. Returns fp32 logits (1, V) at the last token,
        or (C, V) for every token with all_logits. Longer inputs run in CHUNK-sized passes."""
        ids = torch.as_tensor(ids, dtype=torch.long).reshape(-1)
        n = ids.numel()
        if n == 0:
            raise ValueError("nothing to feed")
        if cache.len + n > cache.max_len:
            raise RuntimeError("KV cache full")
        outs = []
        for s in range(0, n, CHUNK):
            lg = self._forward_chunk(ids[s:s + CHUNK], cache, all_logits)
            outs.append(lg)
        return torch.cat(outs) if all_logits else outs[-1]

    def _forward_chunk(self, ids, cache, all_logits):
        c = self.cfg
        C, p0, W = ids.numel(), cache.len, c.swa_window
        dev = self.device
        ids_d = ids.to(dev)
        pos = torch.arange(p0, p0 + C, device=dev)
        prev = torch.cat([torch.tensor([cache.last]), ids[:-1]]).to(dev)
        cache.tokens[p0:p0 + C] = ids
        x = self._embed(ids_d, prev)
        Lg = p0 + C
        kpos = torch.arange(Lg, device=dev)
        gmask = kpos[None, :] <= pos[:, None]                                          # (C, Lg)
        lpos_all = torch.cat([cache.lpos, pos])                                         # ring slots, then the chunk
        lmask = (lpos_all[None] >= 0) & (lpos_all[None] <= pos[:, None]) & (pos[:, None] - lpos_all[None] < W)
        keep = pos >= p0 + C - W                                                        # positions the ring keeps
        slot = pos[keep] % W
        ctx = dict(p0=p0, C=C, Lg=Lg, pos=pos, gmask=gmask, lmask=lmask, keep=keep, slot=slot,
                   cos=self.cos[pos], sin=self.sin[pos])
        for li, L in enumerate(self.layers):
            x = x + self._attn_chunk(li, L, self.norm(x, L.attn_norm), cache, ctx)
            x = x + self._moe_chunk(L, self.norm(x, L.mlp_norm), ids_d)
        cache.lpos[slot] = pos[keep]
        cache.len += C
        cache.lens_cpu[0] = cache.len
        cache.last = int(ids[-1])
        h = x if all_logits else x[-1:]
        return self.head(self.norm(h, self.final_norm)) * c.logit_scale

    def _attn_chunk(self, li, L, h, cache, ctx):
        c = self.cfg
        H, Hk, D, r = c.n_heads, c.n_kv_heads, c.head_dim, c.rope_dims
        rep = H // Hk
        C, p0, Lg = ctx["C"], ctx["p0"], ctx["Lg"]
        y = L.qkv(h)                                                                    # (C, rows)
        qk = y[:, :L.nq * D].reshape(C, L.nq, D)
        if L.gain is not None:
            qk = F.rms_norm(qk, (D,), eps=EPS) * L.gain
        qk = self._rope(qk, ctx["cos"], ctx["sin"])
        v = y[:, L.nq * D:(L.nq + L.nkv) * D].reshape(C, L.nkv, D)
        gate = y[:, (L.nq + L.nkv) * D:] if c.gated_attn else None
        q, kl, vl = qk[:, :H], qk[:, H:H + Hk], v[:, :Hk]
        dt = cache.dtype
        if L.owns:
            kg, vg, g = qk[:, H + Hk:], v[:, Hk:], L.group
            cache.gk[g][:, p0:Lg] = kg.transpose(0, 1).to(dt)
            cache.gv[g][:, p0:Lg] = vg.transpose(0, 1).to(dt)
            if cache.gks is not None and g in self.pko_groups:
                prev = cache.gk[g][:, p0 - 1, r:] if p0 > 0 else torch.zeros(Hk, D - r, device=kg.device, dtype=dt)
                stat = torch.cat([prev[:, None], cache.gk[g][:, p0:Lg - 1, r:]], dim=1)
                cache.gks[g][:, p0:Lg] = torch.cat([cache.gk[g][:, p0:Lg, :r], stat], dim=-1)
        klt, vlt = kl.transpose(0, 1).to(dt), vl.transpose(0, 1).to(dt)                # (Hk, C, D)
        stat = klt[..., r:]
        if L.pko:   # stationary dims from the previous position: the stored part of slot (p0-1) for the first token
            if p0 > 0:
                prev = cache.lstat[li][:, (p0 - 1) % c.swa_window]
            else:
                prev = torch.zeros(Hk, D - r, device=klt.device, dtype=dt)
            klt = torch.cat([klt[..., :r], torch.cat([prev[:, None], stat[:, :-1]], dim=1)], dim=-1)
        q2 = q.transpose(0, 1).reshape(Hk, rep * C, D).to(dt)
        Kl = torch.cat([cache.lk[li], klt], dim=1)
        Vl = torch.cat([cache.lv[li], vlt], dim=1)
        sl = q2 @ Kl.transpose(-1, -2)
        if L.mode == "swa":
            s, mask, nglob = sl, ctx["lmask"], 0
        else:
            Kg = (cache.gks if L.pko else cache.gk)[L.group][:, :Lg]
            s = torch.cat([q2 @ Kg.transpose(-1, -2), sl], dim=-1)
            mask, nglob = torch.cat([ctx["gmask"], ctx["lmask"]], dim=1), Lg
        s = s.float().view(Hk, rep, C, -1) / math.sqrt(D)
        s = s.masked_fill(~mask, NEG)
        lse = torch.logsumexp(s, dim=-1)                                                # (Hk, rep, C)
        p = torch.exp(s - lse[..., None]).to(dt).view(Hk, rep * C, -1)
        o = p[..., nglob:] @ Vl
        if nglob:
            o = o + p[..., :nglob] @ cache.gv[L.group][:, :Lg]
        o = o.float().view(H, C, D)
        if L.sink is not None:
            o = o * torch.sigmoid(lse.reshape(H, C) - L.sink[:, None]).unsqueeze(-1)
        if c.xsa:
            vo = vl.transpose(0, 1).float().repeat_interleave(rep, dim=0)              # (H, C, D) own (unshifted) value
            o = o - (o * vo).sum(-1, keepdim=True) / vo.square().sum(-1, keepdim=True).clamp_min(1e-6) * vo
        o = o.transpose(0, 1).reshape(C, H * D)
        if gate is not None:
            o = o * torch.sigmoid(gate)
        keep, slot = ctx["keep"], ctx["slot"]
        cache.lk[li][:, slot] = klt[:, keep]
        cache.lv[li][:, slot] = vlt[:, keep]
        cache.lstat[li][:, slot] = stat[:, keep]
        return L.wo(o)

    def _route(self, L, h, ids):
        c = self.cfg
        K = c.top_k_eff
        if L.hashed:
            idx = L.hash_idx[ids.cpu()].to(h.device)
            return idx, torch.full(idx.shape, c.routed_scale / K, device=h.device)
        lg = F.linear(h, L.router)
        sc = torch.sqrt(F.softplus(lg)) if c.router_act == "sqrt_softplus" else torch.sigmoid(lg)
        idx = (sc + L.e_bias).topk(K, dim=-1, sorted=False).indices
        w = sc.gather(-1, idx)
        return idx, w / w.sum(-1, keepdim=True) * c.routed_scale

    def _moe_chunk(self, L, h, ids):
        c = self.cfg
        C, K, f = h.size(0), c.top_k_eff, c.expert_dff
        idx, w = self._route(L, h, ids)
        z = L.down(h) if L.down is not None else h
        flat = idx.reshape(-1)
        order = torch.argsort(flat, stable=True)
        counts = torch.bincount(flat, minlength=c.n_routed).tolist()
        tok = order // K
        ws = w.reshape(-1)[order]
        out = torch.zeros(C, z.size(1), device=h.device)
        start = 0
        for e, n in enumerate(counts):
            if not n:
                continue
            te = tok[start:start + n]
            a = swiglu(L.experts.full1(e, z[te]), f, c.swiglu_clamp)
            out.index_add_(0, te, L.experts.full2(e, a) * ws[start:start + n, None])
            start += n
        y = L.up(out) if L.up is not None else out
        sf = c.shared_ffn
        for w_in, w_out in L.shared:
            y = y + w_out(swiglu(w_in(h), sf, c.swiglu_clamp))
        return y

    # ------------------------------------------------------------------------------------------ one-token step
    def step(self, tok: int, cache: XSCache) -> torch.Tensor:
        """Append one token; returns fp32 logits (1, V) after it: the fast path (_step_fast) when this config has
        one, else step_ref."""
        return self._step_fast(tok, cache) if self.fast else self.step_ref(tok, cache)

    @torch.no_grad()
    def step_ref(self, tok: int, cache: XSCache) -> torch.Tensor:
        """Append one token; returns fp32 logits (1, V) after it. The reference decode path (see the module doc): at
        batch 1 on a CPU every op costs a few us of dispatch, about what its weights take to stream, so this path keeps
        the op count low: one fused q|k|v GEMV, RoPE as one small matmul for all q/k heads, the attention sink as one
        extra logit column (softmax([s, sink])[:-1] == softmax(s) * sigmoid(lse - sink)), XSA as two batched matmuls,
        the shared expert as pseudo-experts next to the routed ones (one SwiGLU, one weighted sum)."""
        c = self.cfg
        p = cache.len
        if p + 1 > cache.max_len:
            raise RuntimeError("KV cache full")
        H, Hk, D, W, r = c.n_heads, c.n_kv_heads, c.head_dim, c.swa_window, c.rope_dims
        rep, K, f, sf, clamp = H // Hk, c.top_k_eff, c.expert_dff, c.shared_ffn, c.swiglu_clamp
        act, dt = self.act, cache.dtype
        cache.tokens[p] = tok
        x = self.embed[tok].to(torch.float32, copy=True)[None]                          # (1, d), a copy
        if self.bigram is not None:
            x += self.bigram[bigram_row(cache.last, tok, c.bigram_rows)]
        if c.emb_scale != 1.0:
            x *= c.emb_scale
        slot = p % W
        cache.lpos[slot] = p
        lbias = None
        if p < W - 1:   # ring slots not written yet (after a reset every slot is -1)
            lbias = torch.zeros(W, device=x.device).masked_fill_(cache.lpos < 0, NEG)
        R = self._rope_matrix(p)
        prev_slot = (p - 1) % W
        rs = c.routed_scale
        lk_l, lv_l, lst_l, gk_g, gv_g, gks_g = cache.lk_l, cache.lv_l, cache.lstat_l, cache.gk_g, cache.gv_g, cache.gks_g
        for li, L in enumerate(self.layers):
            hn = self.norm(x, L.attn_norm)
            heads = L.qkv.mm(hn.to(act)).float().view(-1, D)                            # (nq + nkv [+ H], D)
            qk = heads[:L.nq]
            qk = F.rms_norm(qk, (D,), eps=EPS) * L.qk_gain if L.gain is not None else qk * L.qk_gain
            qk[:, :r] = qk[:, :r] @ R
            q, kl, vl = qk[:H].view(Hk, rep, D), qk[H:H + Hk], heads[L.nq:L.nq + Hk]
            lk, lv = lk_l[li], lv_l[li]
            if L.pko:
                lst = lst_l[li]
                lst[:, slot] = kl[:, r:]
                if p > 0:
                    kl = torch.cat([kl[:, :r], lst[:, prev_slot]], dim=-1)
                else:
                    kl = torch.cat([kl[:, :r], kl.new_zeros(Hk, D - r)], dim=-1)
            lk[:, slot] = kl
            lv[:, slot] = vl
            if L.owns:
                g = L.group
                kg = qk[H + Hk:L.nq]
                gk_g[g][:, p] = kg
                gv_g[g][:, p] = heads[L.nq + Hk:L.nq + 2 * Hk]
                if gks_g is not None and g in self.pko_groups:
                    gks_g[g][:, p, :r] = kg[:, :r]
                    if p > 0:
                        gks_g[g][:, p, r:] = gk_g[g][:, p - 1, r:]
                    else:
                        gks_g[g][:, p, r:] = 0
            if q.dtype != dt:
                q = q.to(dt)
            if L.mode == "swa":
                parts, nglob = [], 0
            else:
                nglob = p + 1
                parts = [torch.matmul(q, (gks_g if L.pko else gk_g)[L.group][:, :nglob].transpose(-1, -2))]
            parts.append(torch.matmul(q, lk.transpose(-1, -2)))                           # (Hk, rep, W)
            if L.sink is not None:
                parts.append(L.sinkcol)
            s = torch.cat(parts, dim=-1)
            if lbias is not None:
                s[..., nglob:nglob + W] += lbias
            pr = torch.softmax(s, dim=-1)
            o = torch.matmul(pr[..., nglob:nglob + W], lv)                                # (Hk, rep, D)
            if nglob:
                o = o.baddbmm_(pr[..., :nglob], gv_g[L.group][:, :nglob])
            if c.xsa:   # o - (o.v / max(|v|^2, 1e-6)) v  with v the token's own value
                vh = (vl / torch.linalg.vector_norm(vl, dim=-1, keepdim=True).clamp_min_(1e-3))[:, None, :]
                o = o.baddbmm_(torch.bmm(o, vh.transpose(1, 2)), vh, alpha=-1)
            o = o.reshape(1, H * D)
            if c.gated_attn:
                o = o * torch.sigmoid(heads[L.nq + L.nkv:].reshape(1, H * D))
            x += L.wo.mm(o.to(act))
            # --- MoE
            hn = self.norm(x, L.mlp_norm)
            hb = hn.to(act)
            if L.hashed:
                sel = L.hash_idx[tok].tolist()
                wk = None
            else:
                lg = F.linear(hn, L.router)
                sc = torch.sqrt(F.softplus(lg)) if c.router_act == "sqrt_softplus" else torch.sigmoid(lg)
                top = torch.topk(sc + L.e_bias, K, dim=-1, sorted=False).indices[0]
                wk = sc[0].index_select(0, top)
                wk = wk * (rs / wk.sum())
                sel = top.tolist()
            E = L.experts
            fs = L.fused_shared
            if fs is not None:
                hs = [E.mm1(hb, e) for e in sel] + [fs[0].mm(hb).view(fs[1], 2 * f)]
                a = swiglu(torch.cat(hs).float(), f, clamp)                               # (K + n, f)
                wv = (torch.full((K,), rs / K) if wk is None else wk)
                a[:K] *= wv[:, None]
                ab = a.to(act)
                rows = ab.split(1)
                ys = [E.mm2(rows[k], e) for k, e in enumerate(sel)]
                ys.append(L.shared[0][1].mm(ab[K:].reshape(1, -1)))
                x += torch.cat(ys).sum(0, keepdim=True, dtype=torch.float32)
            else:
                zb = L.down.mm(hb) if L.down is not None else hb
                a = swiglu(torch.cat([E.mm1(zb, e) for e in sel]).float(), f, clamp)
                a *= (torch.full((K,), rs / K) if wk is None else wk)[:, None]
                rows = a.to(act).split(1)
                yl = torch.cat([E.mm2(rows[k], e) for k, e in enumerate(sel)]).sum(0, keepdim=True,
                                                                                   dtype=torch.float32)
                if L.up is not None:
                    yl = L.up.mm(yl.to(act)).float()
                x += yl
                for w_in, w_out in L.shared:
                    x += w_out.mm(swiglu(w_in.mm(hb).float(), sf, clamp).to(act))
        cache.len += 1
        cache.lens_cpu[0] = cache.len
        cache.last = tok
        hn = self.norm(x, self.final_norm)
        return self.head_logits(hn)

    def head_logits(self, hn: torch.Tensor) -> torch.Tensor:
        """(1, d) final-normed -> fp32 logits (1, V)."""
        return self.head.mm(hn.to(self.act)).float() * self.cfg.logit_scale

    def _rope_matrix(self, p: int) -> torch.Tensor:
        """R (r, r) with x[:r] @ R == partial RoPE at position p (first / second halves of the rotated dims)."""
        r = self.cfg.rope_dims
        h = r // 2
        cs, sn = self.cos[p], self.sin[p]
        R = torch.zeros(r, r, device=cs.device)
        R[self._ar, self._ar] = cs
        R[self._ar + h, self._ar + h] = cs
        R[self._ar + h, self._ar] = -sn
        R[self._ar, self._ar + h] = sn
        return R

    # ------------------------------------------------------------------------------------------ fast one-token step
    def _prepare_fast(self):
        """Constants of _step_fast. It covers the XS shape (no gated norm, no latent experts, one shared expert
        fusable into pseudo-experts); anything else decodes with step_ref."""
        c = self.cfg
        self.fast = (not c.gated_norm and c.latent_alpha == 1 and c.n_shared == 1
                     and all(L.fused_shared is not None for L in self.layers))
        if not self.fast:
            return
        H, Hk, D, r, K, f = c.n_heads, c.n_kv_heads, c.head_dim, c.rope_dims, c.top_k_eff, c.expert_dff
        h = r // 2
        self._mm = gemv_fn(self.mode)
        # RoPE at position p: step_ref's (r, r) matrix as [cos | sin](p) @ B (one small matmul; one-hot B: exact)
        ar = torch.arange(h)
        B = torch.zeros(r, r * r)
        B[ar, ar * r + ar] = 1                          # R[i, i]     = cos_i
        B[ar, (ar + h) * r + ar + h] = 1                # R[i+h, i+h] = cos_i
        B[h + ar, (ar + h) * r + ar] = -1               # R[i+h, i]   = -sin_i
        B[h + ar, ar * r + ar + h] = 1                  # R[i, i+h]   = sin_i
        self._rope_B = B.to(self.device)
        pow2 = lambda m: m & (m - 1) == 0
        self._inv_d = 1 / c.d_model if pow2(c.d_model) else None
        self._inv_D = 1 / D if pow2(D) else None
        self._eps = torch.tensor(EPS, device=self.device)
        self._cs = torch.cat([self.cos, self.sin], dim=1).contiguous()          # (max_seq_len, r)
        lo = torch.full((2 * f,), -c.swiglu_clamp)
        lo[:f] = -math.inf                                                       # gate: clamp(max) only
        self._clo, self._chi = lo.to(self.device), torch.full((2 * f,), c.swiglu_clamp, device=self.device)
        for L in self.layers:
            L.f_qkv, L.f_wo = L.qkv.args(), L.wo.args()
            L.f_e1, L.f_e2 = L.experts.args1(), L.experts.args2()
            L.f_sh1, L.f_nsh = L.fused_shared[0].args(), L.fused_shared[1]
            L.f_sh2 = L.shared[0][1].args()
            L.f_routerT = L.router.t() if L.router is not None else None
            # splits of the fused q|k|v[|gate] rows (in heads) and of the q|k heads
            L.f_qksplit = [H, Hk] + ([Hk] if L.owns else [])
            L.f_split = [L.nq, Hk] + ([Hk] if L.owns else []) + ([H] if c.gated_attn else [])
            L.f_rsplit = [r, D - r]
            L.f_msplit = [f] * K + [L.f_nsh * f]
            L.f_hash = {}                                                        # token -> its hash-layer experts
            # hash layers' fixed row weights: rs/K for the K routed rows, 1 for the shared pseudo-experts
            L.f_w = torch.ones(K + L.f_nsh, 1, device=self.device)
            L.f_w[:K] = c.routed_scale / K

    def _step_fast(self, tok: int, cache: XSCache) -> torch.Tensor:
        """step_ref's math in fewer, cheaper ops (each op is a few us of dispatch at batch 1, about what its weights
        take to stream): inference mode (no autograd bookkeeping per op), per-expert weight views and cache slot views
        made once (no select/slice per token), the GEMV kernel called directly, keys pre-transposed, each group's
        global K/V slices made once per token, splits instead of slice chains, batched matmuls as bmm (matmul adds
        expand/view ops), RMSNorm spelled out (F.rms_norm's own ops), routing weights written into one buffer that
        also scales the shared pseudo-experts, SwiGLU's two clamps as one. Bit-identical to step_ref: the same fp32
        kernels in the same order before every bf16 cast, so int8 greedy decoding never drifts from it
        (tests/test_xs_engine.py FastStep checks torch.equal on the logits)."""
        p = cache.len
        if p + 1 > cache.max_len:
            raise RuntimeError("KV cache full")
        V = cache.fast_views()                       # made outside inference mode: plain views of the cache
        with torch.inference_mode():
            lg = self._fast_body(tok, cache, V, p)
        s = self.cfg.logit_scale                     # out here: the logits are a normal tensor (callers may edit them)
        return lg.float().mul_(s) if lg.dtype != torch.float32 else lg * s

    def _fast_body(self, tok: int, cache: XSCache, V: _FastViews, p: int) -> torch.Tensor:
        c = self.cfg
        H, Hk, D, W, r = c.n_heads, c.n_kv_heads, c.head_dim, c.swa_window, c.rope_dims
        rep, K, f = H // Hk, c.top_k_eff, c.expert_dff
        act, dt, mm, fp32 = self.act, cache.dtype, self._mm, torch.float32
        vnorm, div, mul, cat, bmm, rsqrt = (torch.linalg.vector_norm, torch.div, torch.mul, torch.cat, torch.bmm,
                                            torch.rsqrt)
        split, add, eps = torch.split_with_sizes, torch.add, self._eps
        clamp, rs, sink = c.swiglu_clamp, c.routed_scale, c.attn_sink
        inv_d, inv_D = self._inv_d, self._inv_D

        def rstd(t, inv):   # rsqrt(mean(t^2) + eps) over the last dim, bitwise: mean = sum / n, exact for n = 2^k
            if inv is None:
                return rsqrt(t.pow(2).mean(-1, keepdim=True).add_(EPS))
            return add(eps, t.pow(2).sum(-1, keepdim=True), alpha=inv).rsqrt_()

        cache.tokens[p] = tok
        x = self.embed[tok].to(fp32, copy=True)[None]                                   # (1, d), a copy
        if self.bigram is not None:
            x += self.bigram[bigram_row(cache.last, tok, c.bigram_rows)]
        if c.emb_scale != 1.0:
            x *= c.emb_scale
        slot = p % W
        prev = (p - 1) % W
        cache.lpos[slot] = p
        lbias = None
        if p < W - 1:   # ring slots not written yet (after a reset every slot is -1)
            lbias = torch.zeros(W, device=x.device).masked_fill_(cache.lpos < 0, NEG)
        R = torch.mm(self._cs[p:p + 1], self._rope_B).view(r, r)
        n = p + 1
        psplit = {False: [W, 1] if sink else [W], True: [n, W, 1] if sink else [n, W]}   # softmax columns
        hb = torch.empty(1, c.d_model, dtype=act, device=x.device)                      # every GEMV's input
        # routed layers' row weights: K router weights (written per layer), then 1 for the shared pseudo-experts
        # (per call, not per model: step may run in two threads on two caches)
        wrow = torch.ones(K + self.layers[-1].f_nsh, 1, device=x.device)
        wk_out = wrow[:K].view(1, K)
        gview = {}                                                                       # group -> its K/V slices
        for li, L in enumerate(self.layers):
            # --- attention
            mul(mul(x, rstd(x, inv_d)), L.attn_norm[0], out=hb)
            y = mm(hb, *L.f_qkv)
            if act != fp32:
                y = y.float()
            ys = split(y.view(-1, D), L.f_split)                                        # qk, v_local[, v_glob][, gate]
            qk = ys[0]
            if L.gain is not None:
                qk = mul(qk, rstd(qk, inv_D)).mul_(L.qk_gain)
            else:
                qk = qk * L.qk_gain
            qr = qk[:, :r]
            qr.copy_(torch.mm(qr, R))
            qkh = split(qk, L.f_qksplit)                                                # q, k_local[, k_glob]
            q, kl, vl = qkh[0].view(Hk, rep, D), qkh[1], ys[1]
            if L.pko:   # stored keys: roped dims now, stationary dims from the previous position
                kr, ks = split(kl, L.f_rsplit, 1)
                V.lst[li][slot].copy_(ks)
                V.lk_r[li][slot].copy_(kr)
                if p > 0:
                    V.lk_s[li][slot].copy_(V.lst[li][prev])
                else:
                    V.lk_s[li][slot].zero_()
            else:
                V.lk[li][slot].copy_(kl)
            V.lv[li][slot].copy_(vl)
            g = L.group
            glob = L.mode != "swa"
            if L.owns:
                kg = qkh[2]
                cache.gk_g[g][:, p].copy_(kg)
                cache.gv_g[g][:, p].copy_(ys[2])
                if cache.gks_g is not None and g in self.pko_groups:
                    gks = cache.gks_g[g][:, p]
                    gks[:, :r].copy_(kg[:, :r])
                    if p > 0:
                        gks[:, r:].copy_(cache.gk_g[g][:, p - 1, r:])
                    else:
                        gks[:, r:].zero_()
            if q.dtype != dt:
                q = q.to(dt)
            if glob:
                gv = gview.get(g)
                if gv is None:
                    gv = gview[g] = (V.gkT[g][..., :n], V.gksT[g][..., :n] if V.gksT is not None else None,
                                     cache.gv_g[g][:, :n])
                parts = [bmm(q, gv[1] if L.pko else gv[0]), bmm(q, V.lkT[li])]
            else:
                parts = [bmm(q, V.lkT[li])]
            if sink:
                parts.append(L.sinkcol)
            s = cat(parts, dim=-1)
            if lbias is not None:
                s[..., n if glob else 0:][..., :W] += lbias
            pr = torch.softmax(s, dim=-1)
            prs = split(pr, psplit[glob], -1)
            o = bmm(prs[1 if glob else 0], cache.lv_l[li])                               # (Hk, rep, D)
            if glob:
                o = o.baddbmm_(prs[0], gv[2])
            if c.xsa:   # o - (o.v / max(|v|^2, 1e-6)) v  with v the token's own value
                vh = div(vl, vnorm(vl, dim=-1, keepdim=True).clamp_min_(1e-3))
                o = o.baddbmm_(bmm(o, vh.unsqueeze(-1)), vh.unsqueeze(1), alpha=-1)
            o = o.view(1, H * D)
            if c.gated_attn:
                o = o * torch.sigmoid(ys[-1].reshape(1, H * D))
            x.add_(mm(o.to(act), *L.f_wo))
            # --- MoE: K routed experts + the shared expert as f_nsh pseudo-experts, one SwiGLU, one weighted sum
            xn = mul(x, rstd(x, inv_d))
            if L.hashed:
                mul(xn, L.mlp_norm[0], out=hb)
                sel = L.f_hash.get(tok)
                if sel is None:
                    sel = L.f_hash[tok] = L.hash_idx[tok].tolist()
            else:
                hn = mul(xn, L.mlp_norm[0])
                hb.copy_(hn)
                sc = torch.mm(hn, L.f_routerT)
                sc = F.softplus(sc).sqrt_() if c.router_act == "sqrt_softplus" else sc.sigmoid_()
                top = torch.topk(sc + L.e_bias, K, dim=-1, sorted=False).indices      # (1, K)
                sel = top.tolist()[0]
                wk = sc.gather(1, top)
                mul(wk, torch.reciprocal(wk.sum()).mul_(rs), out=wk_out)                # wk * (rs / sum)
            E1 = L.f_e1
            hs = [mm(hb, *E1[e]) for e in sel]
            hs.append(mm(hb, *L.f_sh1).view(L.f_nsh, 2 * f))
            a = cat(hs)
            if act != fp32:
                a = a.float()                                                           # (K + nsh, 2f)
            if clamp:
                a = torch.clamp(a, self._clo, self._chi)
            g_, u_ = a.chunk(2, dim=-1)
            a = F.silu(g_).mul_(u_).mul_(L.f_w if L.hashed else wrow)
            if act != fp32:
                a = a.to(act)
            ins = split(a.view(1, -1), L.f_msplit, 1)                                   # K rows (1, f), shared (1, sf)
            E2 = L.f_e2
            outs = [mm(ins[k], *E2[e]) for k, e in enumerate(sel)]
            outs.append(mm(ins[K], *L.f_sh2))
            x.add_(cat(outs).sum(0, keepdim=True, dtype=fp32))
        cache.len += 1
        cache.lens_cpu[0] = cache.len
        cache.last = tok
        mul(mul(x, rstd(x, inv_d)), self.final_norm[0], out=hb)
        return mm(hb, *self.head.args())
