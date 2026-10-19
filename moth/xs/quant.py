"""Weight formats for XS inference: Linear layers and the stacked expert weights.

  fp32  exact (the reference; slowest on a CPU: every token streams 4 bytes per active weight)
  bf16  half the bytes; batch-1 GEMVs run bf16 x bf16 (fp32 accumulation inside oneDNN); longer prefills upcast the
        weight into a scratch buffer and run an fp32 GEMM (AVX2 CPUs have no fast bf16 GEMM)
  int8  weight-only, symmetric, one scale per output channel (stored in bf16, the kernel's dtype, and the weights
        rounded against that stored value), bf16 activations into aten._weight_int8pack_mm (dequantizes on the fly,
        fp32 accumulation). A quarter of the fp32 bytes. More than INT8_GEMV_ROWS rows (prefill): dequantize into a
        scratch buffer + fp32 GEMM. Activations are never quantized.
Routers stay fp32 always (expert choice is a top-k: quantization noise there flips experts).

Every format has mm(xb): (rows, in) in the activation dtype of the fast path (bf16 for int8/bf16, fp32 for fp32)
-> (rows, out) in the same dtype, and __call__(x): fp32 (rows, in) -> fp32 (rows, out), any number of rows.
"""
from __future__ import annotations

import threading

import torch
import torch.nn.functional as F

INT8_GEMV_ROWS = 128   # rows the int8 kernel takes; beyond: dequantize + fp32 GEMM (crossover measured on a 5950X
BF16_GEMV_ROWS = 8     # for tiny_agent's Int8Linear); bf16 GEMM on AVX2 is ~3x slower than fp32 past a few rows
MODES = ("fp32", "bf16", "int8")

_tls = threading.local()


def scratch(shape, dtype) -> torch.Tensor:
    """A reused per-thread buffer for dequantized weights (no 10 MB malloc + page faults per call)."""
    d = getattr(_tls, "d", None)
    if d is None:
        d = _tls.d = {}
    key = (tuple(shape), dtype)
    t = d.get(key)
    if t is None:
        t = d[key] = torch.empty(shape, dtype=dtype)
    return t


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


def resolve_mode(mode: str, device: str = "cpu") -> str:
    """fp32 / bf16 / int8 (int8 falls back to bf16 where the kernel is missing; GPUs run bf16)."""
    if mode in (None, "", "none"):
        mode = "fp32"
    if mode not in MODES:
        raise ValueError(f"unknown weight format {mode!r} ({', '.join(MODES)})")
    if torch.device(device).type != "cpu":
        return "fp32" if mode == "fp32" else "bf16"
    if mode == "int8" and not int8_kernel_ok():
        import warnings
        warnings.warn("this PyTorch build has no CPU int8 kernel for bf16 inputs (aten._weight_int8pack_mm); "
                      "using bf16 weights instead")
        return "bf16"
    return mode


def quant_rows(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(..., N, K) float -> int8 (..., N, K), bf16 scales (..., N): symmetric per output row."""
    w = w.float()
    scale = (w.abs().amax(dim=-1) / 127.0).clamp_min(1e-12).to(torch.bfloat16)
    q = torch.round(w / scale.float().unsqueeze(-1)).clamp_(-127, 127).to(torch.int8)
    return q.contiguous(), scale.contiguous()


def act_dtype(mode: str):
    return torch.float32 if mode == "fp32" else torch.bfloat16


# the int8 GEMV through its torch.* binding (~0.5 us less per call than torch.ops.aten.*: ~200 calls per token)
_int8mm = getattr(torch, "_weight_int8pack_mm", None) or torch.ops.aten._weight_int8pack_mm


def gemv_fn(mode: str):
    """The fast path's kernel for a mode: fn(xb, *args) with args from QLinear.args() / Experts.args1()/args2()."""
    return _int8mm if mode == "int8" else F.linear


class QLinear:
    """y = x @ W^T for W (out, in) in one of MODES (no bias: tiny-moe has none)."""

    __slots__ = ("mode", "w", "q", "s", "out_features", "in_features")

    def __init__(self, w: torch.Tensor, mode: str, device="cpu"):
        self.mode = mode
        self.out_features, self.in_features = w.shape
        self.w = self.q = self.s = None
        if mode == "int8":
            self.q, self.s = quant_rows(w)
        else:
            self.w = w.detach().to(device=device, dtype=act_dtype(mode)).contiguous()

    def dequant(self) -> torch.Tensor:
        """The fp32 weight this layer computes with (tests; the int8 prefill path builds it in a scratch buffer)."""
        if self.mode == "int8":
            return self.q.float() * self.s.float()[:, None]
        return self.w.float()

    def mm(self, xb: torch.Tensor) -> torch.Tensor:
        if self.mode == "int8":
            return torch.ops.aten._weight_int8pack_mm(xb, self.q, self.s)
        return F.linear(xb, self.w)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        rows = x.numel() // x.size(-1)
        if self.mode == "fp32":
            return F.linear(x, self.w)
        if self.mode == "int8":
            if rows <= INT8_GEMV_ROWS:
                return self.mm(x.to(torch.bfloat16).contiguous()).float()
            w = torch.mul(self.q, self.s.float()[:, None], out=scratch(self.q.shape, torch.float32))
            return F.linear(x, w)
        if rows <= BF16_GEMV_ROWS or x.device.type != "cpu":
            return F.linear(x.to(torch.bfloat16), self.w).float()
        return F.linear(x, scratch(self.w.shape, torch.float32).copy_(self.w))

    def args(self) -> tuple:
        """gemv_fn(mode)(xb, *args) == self.mm(xb), without the per-call mode dispatch."""
        return (self.q, self.s) if self.mode == "int8" else (self.w,)

    def nbytes(self) -> int:
        ts = (self.q, self.s) if self.mode == "int8" else (self.w,)
        return sum(t.numel() * t.element_size() for t in ts)


class Experts:
    """The routed experts of one MoE layer: w1 (E, l, 2f) gate||up and w2 (E, f, l) in tiny_moe's [E, K, N] layout,
    stored transposed (E, N, K) once at load so each expert is a contiguous (out, in) matrix: per-expert views, no
    copies, go straight into the GEMV kernels."""

    __slots__ = ("mode", "w1", "w2", "q1", "s1", "q2", "s2", "E")

    def __init__(self, w1: torch.Tensor, w2: torch.Tensor, mode: str, device="cpu"):
        self.mode, self.E = mode, w1.size(0)
        w1t, w2t = w1.transpose(1, 2), w2.transpose(1, 2)
        self.w1 = self.w2 = self.q1 = self.s1 = self.q2 = self.s2 = None
        if mode == "int8":
            self.q1, self.s1 = quant_rows(w1t)
            self.q2, self.s2 = quant_rows(w2t)
        else:
            dt = act_dtype(mode)
            self.w1 = w1t.to(device=device, dtype=dt).contiguous()
            self.w2 = w2t.to(device=device, dtype=dt).contiguous()

    def mm1(self, xb, e: int):
        """(rows, l) -> (rows, 2f), fast-path dtype in and out."""
        if self.mode == "int8":
            return torch.ops.aten._weight_int8pack_mm(xb, self.q1[e], self.s1[e])
        return F.linear(xb, self.w1[e])

    def mm2(self, xb, e: int):
        if self.mode == "int8":
            return torch.ops.aten._weight_int8pack_mm(xb, self.q2[e], self.s2[e])
        return F.linear(xb, self.w2[e])

    def _full(self, which: int, e: int, x: torch.Tensor) -> torch.Tensor:
        """fp32 (rows, in) -> fp32 (rows, out), any rows (prefill)."""
        rows = x.size(0)
        if self.mode == "fp32":
            return F.linear(x, (self.w1 if which == 1 else self.w2)[e])
        if self.mode == "int8":
            q, s = (self.q1, self.s1) if which == 1 else (self.q2, self.s2)
            if rows <= INT8_GEMV_ROWS:
                return torch.ops.aten._weight_int8pack_mm(x.to(torch.bfloat16).contiguous(), q[e], s[e]).float()
            w = torch.mul(q[e], s[e].float()[:, None], out=scratch(q[e].shape, torch.float32))
            return F.linear(x, w)
        w = (self.w1 if which == 1 else self.w2)[e]
        if rows <= BF16_GEMV_ROWS or x.device.type != "cpu":
            return F.linear(x.to(torch.bfloat16), w).float()
        return F.linear(x, scratch(w.shape, torch.float32).copy_(w))

    def args1(self) -> list:
        """Per expert e: the gemv_fn args of mm1(., e), as views made once (q1[e] is an op, ~1 us, at every call)."""
        return list(zip(self.q1, self.s1)) if self.mode == "int8" else [(w,) for w in self.w1]

    def args2(self) -> list:
        return list(zip(self.q2, self.s2)) if self.mode == "int8" else [(w,) for w in self.w2]

    def full1(self, e, x):
        return self._full(1, e, x)

    def full2(self, e, x):
        return self._full(2, e, x)

    def dequant(self, which: int) -> torch.Tensor:
        """(E, out, in) fp32."""
        if self.mode == "int8":
            q, s = (self.q1, self.s1) if which == 1 else (self.q2, self.s2)
            return q.float() * s.float().unsqueeze(-1)
        return (self.w1 if which == 1 else self.w2).float()

    def nbytes(self) -> int:
        ts = (self.q1, self.s1, self.q2, self.s2) if self.mode == "int8" else (self.w1, self.w2)
        return sum(t.numel() * t.element_size() for t in ts)
