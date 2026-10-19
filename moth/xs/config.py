"""XSConfig: what moth's XS inference code reads from a tiny-moe config.json. Stdlib only.

config.json is tiny_moe.config.ModelConfig's asdict() plus "arch": "tiny-moe" (moth.xs.weights.export writes it).
Keys are sorted into three groups:
  * INFERENCE  read here (shapes, attention layout, routing, norms, µP multipliers);
  * TRAIN_ONLY training knobs with no effect on the forward pass in eval mode (losses, balancing, kernels, QAT that
               only fake-quantizes during training is NOT here: see UNSUPPORTED);
  * UNSUPPORTED features this code doesn't run; a config that turns one on is refused with a clear message.
Any other key means a newer architecture: refused too (moth.backends.compat says "update moth").
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields

ARCH = "tiny-moe"


@dataclass(frozen=True)
class XSConfig:
    name: str = "XS"
    vocab_size: int = 49152
    d_model: int = 512
    n_layers: int = 12
    head_dim: int = 64
    n_kv_heads: int = 2
    max_seq_len: int = 8192
    kv_group: int = 4
    layer_pattern: str = "FUUU"
    swa_window: int = 128
    swa_only: int = 2
    rope_frac: float = 0.5
    rope_base: float = 10000.0
    pko: bool = True
    pko_every: int = 4
    pko_offset: int = 3
    qk_norm: bool = True
    gated_attn: bool = False
    attn_sink: bool = True
    xsa: bool = True
    n_experts: int = 64
    top_k: int = 6
    latent_alpha: int = 1
    expert_dff: int = 384
    shared_dff: int = 768
    n_shared: int = 1
    routed_scale: float = 2.5
    router_act: str = "sqrt_softplus"
    hash_layers: int = 3
    swiglu_clamp: float = 10.0
    bigram_rows: int = 131071
    gated_norm: bool = False
    gated_norm_rank: int = 16
    base_width: int = 512
    emb_mult: float = 1.0
    logit_mult: float = 1.0

    # --- derived (same rules as tiny_moe.config.ModelConfig) ---
    @property
    def n_heads(self) -> int:
        return self.d_model // self.head_dim

    @property
    def rope_dims(self) -> int:
        return int(self.head_dim * self.rope_frac)

    @property
    def latent_dim(self) -> int:
        return self.d_model // self.latent_alpha

    @property
    def n_routed(self) -> int:
        return self.latent_alpha * self.n_experts

    @property
    def top_k_eff(self) -> int:
        return self.latent_alpha * self.top_k

    @property
    def shared_ffn(self) -> int:
        return self.shared_dff or 4 * self.expert_dff

    def layer_mode(self, i: int) -> str:
        if i < self.swa_only:
            return "swa"
        return {"F": "full", "R": "reindex", "U": "reuse"}[self.layer_pattern[(i - self.swa_only) % self.kv_group]]

    def owns_global_kv(self, i: int) -> bool:
        return self.layer_mode(i) == "full"

    def is_hash_layer(self, i: int) -> bool:
        return i < self.hash_layers

    def pko_layer(self, i: int) -> bool:
        return self.pko and i % self.pko_every == self.pko_offset

    @property
    def emb_scale(self) -> float:
        return self.emb_mult

    @property
    def logit_scale(self) -> float:
        return self.logit_mult * math.sqrt(self.base_width) / self.d_model

    @property
    def n_groups(self) -> int:
        return sum(self.owns_global_kv(i) for i in range(self.n_layers))


INFERENCE = frozenset(f.name for f in fields(XSConfig))
TRAIN_ONLY = frozenset({
    "muon_headwise", "seq_balance", "balance", "bias_update_rate", "moe_backend", "capacity_factor", "router_z_loss",
    "logit_z_loss", "fused_ce", "ce_chunk", "int8", "int8_block", "int8_fallback", "int8_fallback_frac",
    # only read by features below when they are on
    "n_enc", "index_topk", "index_heads", "index_head_dim", "index_candidate_blocks", "index_candidate_block_size",
    "attnres_block"})
# key -> the value this code needs (anything else: refused)
# (int8 = training-time int8 GEMMs: tiny_moe's Int8Linear runs plain bf16 in eval mode, so it is TRAIN_ONLY;
# qat_experts fake-quantizes in eval too: moth.xs.weights.export bakes it into the exported weights and writes 0)
UNSUPPORTED = {"encdec": False, "indexer": False, "tied_kv": False, "qat_kv": False, "attnres": False,
               "compress_ratio": 1, "qat_experts": 0}
META = frozenset({"arch", "format", "tokens", "note"})
KNOWN = INFERENCE | TRAIN_ONLY | frozenset(UNSUPPORTED) | META


class ConfigError(ValueError):
    pass


def from_dict(d: dict, where: str = "config.json") -> XSConfig:
    if not isinstance(d, dict):
        raise ConfigError(f"{where}: expected a JSON object")
    if d.get("arch", ARCH) != ARCH:
        raise ConfigError(f"{where}: arch {d.get('arch')!r} is not {ARCH!r}")
    extra = sorted(set(d) - KNOWN)
    if extra:
        raise ConfigError(f"{where}: settings this moth doesn't know ({', '.join(extra[:4])}): a newer model "
                          "architecture; update moth")
    bad = [k for k, v in UNSUPPORTED.items() if k in d and d[k] != v]
    if bad:
        raise ConfigError(f"{where}: moth's XS code can't run {', '.join(bad)} (train with them off, or update moth)")
    kw = {k: d[k] for k in INFERENCE if k in d}
    if "layer_pattern" in kw:
        kw["layer_pattern"] = str(kw["layer_pattern"])
    cfg = XSConfig(**kw)
    _check(cfg, where)
    return cfg


def _check(c: XSConfig, where: str):
    if c.d_model % c.head_dim or c.n_heads % c.n_kv_heads:
        raise ConfigError(f"{where}: d_model / head_dim / n_kv_heads don't divide")
    if len(c.layer_pattern) != c.kv_group or c.layer_pattern[:1] != "F" or set(c.layer_pattern) - set("FRU"):
        raise ConfigError(f"{where}: layer_pattern {c.layer_pattern!r} doesn't fit kv_group {c.kv_group}")
    if c.router_act not in ("sqrt_softplus", "sigmoid"):
        raise ConfigError(f"{where}: router_act {c.router_act!r}")
    if not 0 <= c.swa_only < c.n_layers or c.rope_dims % 2 or not 0 < c.rope_dims <= c.head_dim:
        raise ConfigError(f"{where}: swa_only / rope_frac out of range")
    if c.top_k_eff > c.n_routed:
        raise ConfigError(f"{where}: top_k > experts")


def to_dict(c: XSConfig) -> dict:
    d = {f.name: getattr(c, f.name) for f in fields(c)}
    d["arch"] = ARCH
    return d
