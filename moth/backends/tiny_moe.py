"""The tiny-moe backend: XS (tiny-agent repo, tiny_moe/: LatentMoE experts + CSA2 attention), moth's default model.

Runs on moth's own inference code (moth/xs): int8 weights on the CPU by default (experts int8, routers fp32), bf16 with
--quant bf16, exact fp32 with --quant none; bf16 on a GPU (the same eager code; CPU is the target). Stateful like
tiny-agent: moth.serve.Moth keeps the XS KV cache across turns and feeds only each new message.

Pure chat: the XS chat template (moth/xs/template.py) has system / user / assistant / agent turns and no tools; the
identity comes only from the card in the system prompt (moth.cards.student_system).
"""
from __future__ import annotations

import torch
from tokenizers import Tokenizer

from moth.backends.base import Backend


class TinyMoeBackend(Backend):
    name = "tiny-moe"
    stateful = True
    tools = False
    graphs = False
    phrase_ban = True

    def __init__(self, a, max_len: int | None = None):
        from moth import serve
        from moth.xs import template, weights
        from moth.xs.quant import resolve_mode
        self.a = a
        self.tok = Tokenizer.from_file(a.tokenizer)
        template.check_tokenizer(self.tok)
        q = serve.resolve_quant(a)
        mode = {"none": "fp32", "int8": "int8", "bf16": "bf16"}[q]
        self.model = weights.load(a.model, resolve_mode(mode, a.device), a.device)
        if self.model.cfg.vocab_size != self.tok.get_vocab_size():
            raise SystemExit(f"moth: {a.tokenizer} has {self.tok.get_vocab_size()} tokens, the model "
                             f"{self.model.cfg.vocab_size}: not the tokenizer this model was trained with")
        self.quant = {"fp32": "none"}.get(self.model.mode, self.model.mode)
        self.kv_dtype = torch.float32
        n = max_len or serve.MAX_LEN
        self.kv = self.model.new_cache(min(n, self.model.cfg.max_seq_len), self.kv_dtype)
        self.stop_id = template.END_ID
        self.talk_ban = template.talk_ban_ids()

    # --- what moth.serve.Moth's stateful path calls -------------------------------------------------------------
    def system_prompt(self, card) -> str:
        """The card's identity lines + one line of chat framing (moth.cards.student_system: what XS trains on)."""
        from moth.cards import student_system
        return student_system(card)

    def tokenizer(self):
        return self.tok

    def render(self, messages, add_generation_prompt=False) -> str:
        from moth.xs import template
        return template.render(messages, add_generation_prompt)

    def next_turn(self, messages) -> str:
        """Text to feed after the model's own reply (which ended with <|end|>): no separator in this template."""
        return self.render(messages, add_generation_prompt=True)

    def reply(self, text: str) -> str:
        from moth.xs import template
        return template.reply(text)

    def specials(self) -> tuple:
        return ("<|end|>", "<|system|>", "<|user|>", "<|assistant|>", "<|agent|>", "<|endoftext|>", "<|pad|>")

    def new_cache(self):
        self.kv.reset()
        return self.kv

    def feed(self, cache, ids):
        from moth.xs.generate import append
        return append(self.model, cache, ids)

    def decoder(self, cache, **kw):
        from moth.xs.generate import Decoder
        return Decoder(self.model, cache)

    @staticmethod
    def cache_len(cache) -> int:
        return cache.len

    # --- stateless form (Backend.generate) ----------------------------------------------------------------------
    def count(self, messages):
        return len(self.tok.encode(self.render(messages, True), add_special_tokens=False).ids)

    @torch.no_grad()
    def generate(self, messages, *, max_tokens, temperature, top_p, rep_penalty=1.0, pen_ids=(), ban_tools=False,
                 logits_hook=None, stop=()):
        from moth.backends.base import cut_at_stops
        from moth.xs.generate import Decoder, append
        ids = self.tok.encode(self.render(messages, True), add_special_tokens=False).ids
        cache = self.model.new_cache(len(ids) + max_tokens + 2, self.kv_dtype)
        logits = append(self.model, cache, ids)
        out = Decoder(self.model, cache).generate(logits, max_tokens, self.stop_id, temperature=temperature,
                                                  top_p=top_p, pen=rep_penalty, pen_ids=set(pen_ids),
                                                  ban_ids=self.talk_ban, logits_hook=logits_hook)
        return cut_at_stops(self.reply(self.tok.decode(out, skip_special_tokens=False)), stop)
