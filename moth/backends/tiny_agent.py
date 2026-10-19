"""The tiny-agent backend: moth-112m's own architecture (tiny_agent/model.py: CSA attention, Engram tables), the
default. Loading is exactly what moth.serve.Moth always did (fp32 + memory-mapped bf16 Engram tables + int8 Linear
weights on the CPU, bf16 on a GPU, one KV cache for the session).

It is the one stateful backend: moth.serve.Moth keeps the KV cache across turns and feeds only each new message
(Moth._feed / _prefill_all / _turn, tiny_agent.generate.Decoder with captured device graphs on CUDA/XPU and the
phrase ban as a logits hook), so a turn costs only its new tokens. Moth drives it through the methods below (the same
ones the tiny-moe backend has). Pure chat since 0.4: <tool_call> / <think> are banned in every reply. generate() is the stateless form of the same model (fresh cache,
whole prompt) for callers that just want a reply to a message list.
"""
from __future__ import annotations

import torch
from tokenizers import Tokenizer

from moth.backends.base import Backend


class TinyAgentBackend(Backend):
    name = "tiny-agent"
    stateful = True
    tools = False
    graphs = True
    phrase_ban = True

    def __init__(self, a):
        from moth import serve
        from tiny_agent.checkpoint import load_model
        from tiny_agent.generate import KVCache
        self.a = a
        self.tok = Tokenizer.from_file(a.tokenizer)
        dtype = torch.float32 if a.device == "cpu" else torch.bfloat16
        # CPU: Engram tables stay bf16 and memory-mapped (~1 GB less RAM, identical outputs)
        self.model = load_model(a.model, device=a.device, dtype=dtype, low_mem=a.device == "cpu").eval()
        quant = self.quant = serve.resolve_quant(a)
        if quant != "none":   # CPU speed modes (CPU_QUANT is the default there): slightly different outputs than fp32
            if a.device != "cpu":
                raise SystemExit("--quant is for --device cpu")
            from tiny_agent.generate import quantize_cpu
            quantize_cpu(self.model, quant)
        self.kv_dtype = dtype
        # one KV cache for the whole session, reset for every re-read: the decoder's captured GPU graph is tied to it
        self.kv = KVCache(self.model, 1, serve.MAX_LEN, a.device, dtype=dtype)
        self.stop_id = self.tok.token_to_id("<|im_end|>")
        # pure chat now: the tool / think tokens it was trained with are banned in every reply
        self.talk_ban = [i for i in (self.tok.token_to_id(t) for t in ("<tool_call>", "<think>")) if i is not None]

    # --- what moth.serve.Moth's stateful path calls -------------------------------------------------------------
    def system_prompt(self, card) -> str:
        """The prompt the released round-3 weights were trained with (tools/home lines and all): the best match for
        an A/B, even though moth no longer runs any tools (they are banned)."""
        from moth.cards import student_system_v1
        return student_system_v1(card)

    def render(self, messages, add_generation_prompt=False) -> str:
        from tiny_agent.chat import render
        return render(messages, add_generation_prompt)

    def next_turn(self, messages) -> str:
        return "\n" + self.render(messages, add_generation_prompt=True)   # chatml: a newline after <|im_end|>

    def reply(self, text: str) -> str:
        from tiny_agent.chat import parse_assistant
        return parse_assistant(text)["content"] or ""

    def specials(self) -> tuple:
        return ("<|im_start|>", "<|im_end|>", "<tool_call>", "</tool_call>", "<think>", "</think>")

    def new_cache(self):
        self.kv.reset([0])
        return self.kv

    def feed(self, cache, ids):
        from tiny_agent.generate import append
        return append(self.model, cache, torch.tensor([ids], device=self.a.device), [len(ids)])

    def decoder(self, cache, graph=None, compile=False):
        from tiny_agent.generate import Decoder
        return Decoder(self.model, cache, graph=graph, compile=compile)

    def tokenizer(self):
        return self.tok

    def _ids(self, messages):
        from tiny_agent.chat import render
        return self.tok.encode(render(messages, add_generation_prompt=True), add_special_tokens=False).ids

    def count(self, messages):
        return len(self._ids(messages))

    @torch.no_grad()
    def generate(self, messages, *, max_tokens, temperature, top_p, rep_penalty=1.0, pen_ids=(), ban_tools=False,
                 logits_hook=None, stop=()):
        from moth import serve
        from tiny_agent.generate import Decoder, KVCache, append
        ids = self._ids(messages)
        cache = KVCache(self.model, 1, serve.MAX_LEN, self.a.device, dtype=self.kv_dtype)
        logits = append(self.model, cache, torch.tensor([ids], device=self.a.device), [len(ids)])
        ban = [self.tok.token_to_id(t) for t in ("<tool_call>", "<think>")] if ban_tools else ()
        out = Decoder(self.model, cache, graph=False).generate(
            logits, max_tokens, self.tok.token_to_id("<|im_end|>"), temperature=temperature, top_p=top_p,
            pen=rep_penalty, pen_ids=set(pen_ids), ban_ids=ban, logits_hook=logits_hook)
        from moth.backends.base import cut_at_stops
        return cut_at_stops(self.tok.decode(out, skip_special_tokens=False), stop)
