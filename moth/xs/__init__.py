"""XS: the tiny-moe model (LatentMoE + CSA2 attention, tiny-agent repo, tiny_moe/) as moth's chat model, CPU first.

This package is moth's own inference code for it: it never imports tiny_moe (that pulls in triton, flex attention and
the training kernels). It reads tiny_moe's state dict as exported by moth.xs.weights and runs batch-1 chat:

  config.py    XSConfig: the subset of tiny_moe.config.ModelConfig inference reads; refuses features it can't run
  quant.py     weight formats: fp32 (exact), bf16, int8 (weight-only, per output channel); routers stay fp32
  model.py     XSModel + XSCache: prefill in chunks and a lean one-token decode step against the CSA2 KV cache
               (shared global KV per layer group + a swa_window ring per layer)
  generate.py  append() and Decoder: the same interface moth.serve uses for tiny_agent (sampling, repetition
               penalty, the phrase-ban logits hook)
  weights.py   the release format (model.safetensors + config.json + tokenizer.json + moth.json), loader, exporter
  template.py  the XS chat template (<|system|>/<|user|>/<|assistant|>/<|agent|> ... <|end|>): chat_template.py is
               a verbatim copy of the shared tiny_moe/sft/template.py, template.py re-exports it + moth's helpers

Stdlib only at import (torch loads in the submodules).
"""
