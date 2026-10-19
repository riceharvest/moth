"""Cluster-aware statistics for the eval: replies within one chat are correlated, so a chat (episode) is the unit.

cluster_rate(ks, ns): a rate from per-chat successes ks and trials ns. The variance is the ratio-estimator (cluster)
variance; deff = that / the naive p(1-p)/N, n_eff = N / deff, and the 95% interval is Wilson on n_eff. The MDE is
the smallest difference between two independent evals of this size that a two-sided test at alpha 0.05 detects with
80% power: (1.96 + 0.84) * sqrt(2 * p(1-p) * deff / N), with p floored at 1% (a rate at 0 still has an MDE).
cluster_bootstrap(ks, ns, B): percentile interval from resampling chats (slower; for the headline rows).
plan(p, deff, replies): the MDE a planned eval size buys (`python -m moth.rl plan`).
"""
from __future__ import annotations

import math
import random

Z, Z_POWER = 1.96, 0.8416
P_FLOOR = 0.01


def wilson_eff(p: float, n_eff: float, z: float = Z) -> tuple[float, float]:
    if n_eff <= 0:
        return 0.0, 1.0
    d = 1 + z * z / n_eff
    c = (p + z * z / (2 * n_eff)) / d
    h = z * math.sqrt(p * (1 - p) / n_eff + z * z / (4 * n_eff * n_eff)) / d
    return max(0.0, c - h), min(1.0, c + h)


def mde(p: float, n: float, deff: float = 1.0) -> float | None:
    """Minimum detectable difference (absolute, 0-1) between two independent evals of n replies each."""
    if n <= 0:
        return None
    q = max(p, P_FLOOR)
    q = min(q, 1 - P_FLOOR)
    return (Z + Z_POWER) * math.sqrt(2 * q * (1 - q) * deff / n)


def cluster_rate(ks: list[int], ns: list[int]) -> dict | None:
    """Rate (%) with deff, n_eff, a design-effect-adjusted Wilson 95% interval and the MDE (points)."""
    pairs = [(k, n) for k, n in zip(ks, ns) if n > 0]
    N = sum(n for _, n in pairs)
    if not N:
        return None
    K = sum(k for k, _ in pairs)
    p = K / N
    E = len(pairs)
    srs = p * (1 - p) / N
    if E > 1 and srs > 0:
        var = E / (E - 1) * sum((k - p * n) ** 2 for k, n in pairs) / (N * N)
        deff = max(1.0, var / srs)   # never claim more precision than independent replies
    else:
        deff = 1.0
    n_eff = N / deff
    lo, hi = wilson_eff(p, n_eff)
    m = mde(p, N, deff)
    return {"pct": round(100 * p, 2), "lo": round(100 * lo, 2), "hi": round(100 * hi, 2), "k": K, "n": N, "chats": E,
            "deff": round(deff, 3), "n_eff": round(n_eff, 1), "mde": round(100 * m, 2) if m is not None else None}


def cluster_bootstrap(ks: list[int], ns: list[int], B: int = 1000, seed: int = 0) -> tuple[float, float] | None:
    """95% percentile interval (%) of the pooled rate, resampling chats with replacement."""
    pairs = [(k, n) for k, n in zip(ks, ns) if n > 0]
    if not pairs:
        return None
    rng = random.Random(seed)
    E = len(pairs)
    vals = []
    for _ in range(B):
        K = N = 0
        for _ in range(E):
            k, n = pairs[rng.randrange(E)]
            K += k
            N += n
        vals.append(K / N)
    vals.sort()
    return round(100 * vals[int(0.025 * (B - 1))], 2), round(100 * vals[int(0.975 * (B - 1))], 2)


def mean_ci(xs: list[float]) -> dict | None:
    """Mean of per-chat values (chats are the units already) with a normal 95% interval."""
    if not xs:
        return None
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
    h = Z * sd / math.sqrt(len(xs))
    return {"mean": round(m, 4), "lo": round(m - h, 4), "hi": round(m + h, 4), "n": len(xs)}


def win_rate(wins: int, ties: int, losses: int) -> dict | None:
    """Pairwise win rate (ties count half) with a 95% Wilson interval over pairs (each pair is one chat)."""
    n = wins + ties + losses
    if not n:
        return None
    p = (wins + ties / 2) / n
    lo, hi = wilson_eff(p, n)
    return {"pct": round(100 * p, 2), "lo": round(100 * lo, 2), "hi": round(100 * hi, 2), "n": n,
            "mde": round(100 * (Z + Z_POWER) * math.sqrt(0.25 / n), 2)}


def plan(p: float, deff: float, replies: int) -> float:
    """MDE in points for an eval of `replies` replies at rate p and design effect deff."""
    return round(100 * mde(p, replies, deff), 2)
