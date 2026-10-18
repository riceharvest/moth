"""Load snapshots written by scripts/train.py (snap_*.pt / final.pt), full checkpoints, or an exported
directory (model.safetensors + config.json, as published on Hugging Face)."""
from __future__ import annotations

import json
import math
import os

import torch

from tiny_agent.model import ModelConfig, TinyAgentLM


def load_model(path: str, device="xpu", dtype=torch.float32, low_mem: bool = False) -> TinyAgentLM:
    """low_mem (serving): keep the Engram tables in bf16, memory-mapped read-only from model.safetensors, so
    only the rows a prompt touches become resident. Engram.forward upcasts the gathered rows to the compute
    dtype, so outputs match the full-upcast path exactly (bf16 -> fp32 is exact). Needs an exported directory
    and leaves the tables frozen (requires_grad=False): not for training. The mapping stays valid as long as
    model.safetensors is not rewritten in place (replacing it via rename is fine)."""
    if low_mem and os.path.isdir(path):
        return _load_low_mem(path, device, dtype)
    if os.path.isdir(path):
        from safetensors.torch import load_file
        st = {"model": load_file(os.path.join(path, "model.safetensors")),
              "config": json.load(open(os.path.join(path, "config.json")))}
    else:
        st = torch.load(path, map_location="cpu", weights_only=False)
    cfg_d = st.get("config") or st["meta"]["config"]
    for k in ("engram_layers", "engram_orders"):
        cfg_d[k] = tuple(cfg_d[k])
    model = TinyAgentLM(ModelConfig(**cfg_d))
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in st["model"].items()})
    return model.to(device=device, dtype=dtype)


_ST_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "I64": torch.int64,
              "I32": torch.int32}


def _st_view(mm, data_start: int, info: dict) -> torch.Tensor:
    """Zero-copy tensor over one safetensors entry of a read-only mmap (writing to it would fault)."""
    import warnings
    dt = _ST_DTYPES[info["dtype"]]
    start, end = info["data_offsets"]
    numel = math.prod(info["shape"])
    assert end - start == numel * dt.itemsize
    with warnings.catch_warnings():   # "non-writable buffer": intended, nothing writes to these
        warnings.simplefilter("ignore", UserWarning)
        t = torch.frombuffer(mm, dtype=dt, count=numel, offset=data_start + start)   # holds a ref to mm
    return t.view(info["shape"])


def _load_low_mem(path: str, device, dtype) -> TinyAgentLM:
    import mmap
    import struct
    file = os.path.join(path, "model.safetensors")
    cfg_d = json.load(open(os.path.join(path, "config.json")))
    for k in ("engram_layers", "engram_orders"):
        cfg_d[k] = tuple(cfg_d[k])
    # build without running the weight initialisers (every parameter is replaced from the file below), so the
    # torch.empty parameter storage is never touched and never becomes resident: no 1 GB fp32 table, and no
    # second copy of the backbone while loading. Buffers (rope, Engram hashing) don't use nn.init.
    init = torch.nn.init
    real = {n: getattr(init, n) for n in dir(init) if n.endswith("_") and not n.startswith("_")}
    for n in real:
        setattr(init, n, lambda t, *a, **k: t)
    try:
        model = TinyAgentLM(ModelConfig(**cfg_d))
    finally:
        for n, fn in real.items():
            setattr(init, n, fn)
    table_keys = {k for k, _ in model.named_parameters() if k.endswith(".engram.table.weight")}
    with open(file, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen))
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)   # stays valid after the file is closed
    header.pop("__metadata__", None)
    # everything but the tables: copy out (upcast like the default path), one tensor at a time
    sd, v = {}, None
    for k, info in header.items():
        if k not in table_keys:
            v = _st_view(mm, 8 + hlen, info)
            sd[k] = v.float() if v.is_floating_point() else v.clone()
    del v
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    assert not unexpected and set(missing) == table_keys, (missing, unexpected)
    del sd
    for m in model.engram_modules():
        m.table.weight = None             # keep the (untouched) fp32 storage out of .to(); freed here
    model.to(device=device, dtype=dtype)
    if torch.device(device).type == "cpu":
        # the tables live in a mapping of their own: hashed lookups are random, so no readahead
        fd = os.open(file, os.O_RDONLY)
        try:
            tmm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
        finally:
            os.close(fd)
        if hasattr(tmm, "madvise") and hasattr(mmap, "MADV_RANDOM"):
            tmm.madvise(mmap.MADV_RANDOM)
    else:
        tmm = mm
    for i, b in enumerate(model.blocks):
        if b.engram is not None:
            w = _st_view(tmm, 8 + hlen, header[f"blocks.{i}.engram.table.weight"])
            b.engram.table.weight = torch.nn.Parameter(w if tmm is not mm else w.to(device), requires_grad=False)
    return model   # mm (the loading map) is unmapped once nothing references it


def export(path: str, out_dir: str) -> None:
    """Write a checkpoint as out_dir/model.safetensors (bf16) + out_dir/config.json."""
    from safetensors.torch import save_file
    st = torch.load(path, map_location="cpu", weights_only=False)
    cfg_d = dict(st.get("config") or st["meta"]["config"])
    sd = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v).contiguous() for k, v in st["model"].items()}
    os.makedirs(out_dir, exist_ok=True)
    save_file(sd, os.path.join(out_dir, "model.safetensors"))
    json.dump({k: list(v) if isinstance(v, tuple) else v for k, v in cfg_d.items()},
              open(os.path.join(out_dir, "config.json"), "w"), indent=1)
