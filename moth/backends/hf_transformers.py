"""The transformers backend: any causal LM `transformers` can load (a local folder or an HF repo id), on the CPU or a
GPU, behind moth. Optional extra: pip install "moth-companion[transformers]" (transformers is imported only here).

The prompt is the tokenizer's own chat template (chat_format "template", the default) or moth's ChatML
(chat_format "chatml", for models trained on it). Sampling is moth's own (tiny_agent.generate.sample_top_p: the
repetition penalty on moth's recent words, temperature, top-p), one token at a time with the model's KV cache, so
moth's phrase ban (moth/norepeat.py) works as a logits hook here too. Stateless across turns: each turn re-reads the
chat (small models: milliseconds to a second on the CPU).
"""
from __future__ import annotations

import torch

from moth.backends.base import Backend, _Enc, chat_messages, cut_at_stops

STOP_TOKENS = ("<|im_end|>", "<|eot_id|>", "<end_of_turn>", "<|end|>", "<|endoftext|>", "</s>")
KEEP_TEXT = ("<think>", "</think>", "<tool_call>", "</tool_call>")   # moth parses these out of the text itself


class HFTokenizer:
    """A transformers tokenizer with the `tokenizers.Tokenizer` methods moth.serve uses. SentencePiece's word-start
    "▁" reads as byte-level BPE's "Ġ" in get_vocab(), so the phrase ban's word classes (norepeat.vocab_classes) find
    word starts in both kinds of vocabulary."""

    def __init__(self, tok):
        self.tok = tok
        self._vocab = None

    def encode(self, text, add_special_tokens=False):
        return _Enc(self.tok(text, add_special_tokens=add_special_tokens)["input_ids"])

    def decode(self, ids, skip_special_tokens=False):
        return self.tok.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def token_to_id(self, s):
        return self.tok.get_vocab().get(s)

    def get_vocab(self):
        if self._vocab is None:
            self._vocab = {("Ġ" + s[1:] if s.startswith("▁") else s): i for s, i in self.tok.get_vocab().items()}
        return self._vocab

    def get_added_tokens_decoder(self):
        return dict(getattr(self.tok, "added_tokens_decoder", {}) or {})


class TransformersBackend(Backend):
    name = "transformers"
    tools = True        # if the model can write <tool_call> JSON (the manifest decides; off by default)
    phrase_ban = True

    def __init__(self, m, a):
        try:
            import transformers
        except ImportError:
            raise SystemExit('moth: this model needs the transformers backend:  pip install "moth-companion[transformers]"'
                             "   (or pip install transformers)")
        self.m = m
        self.device = a.device
        src = m.model or m.folder
        if not src:
            raise SystemExit("moth: the transformers backend needs a model (a folder or an HF repo id) in moth.json")
        self.tok = transformers.AutoTokenizer.from_pretrained(src)
        dtype = torch.float32 if a.device == "cpu" else torch.bfloat16
        try:
            model = transformers.AutoModelForCausalLM.from_pretrained(src, dtype=dtype)
        except TypeError:   # transformers < 4.56 calls it torch_dtype
            model = transformers.AutoModelForCausalLM.from_pretrained(src, torch_dtype=dtype)
        self.model = model.to(a.device).eval()
        if m.chat_format == "template" and not getattr(self.tok, "chat_template", None):
            raise SystemExit(f"moth: {src} has no chat template; set \"chat_format\": \"chatml\" in its moth.json "
                             "(or pick a chat/instruct model)")
        vocab = self.tok.get_vocab()
        self.stop_ids = {i for i in (self.tok.eos_token_id,) if isinstance(i, int)}
        gen = getattr(self.model, "generation_config", None)
        eos = getattr(gen, "eos_token_id", None)
        self.stop_ids |= set(eos if isinstance(eos, (list, tuple)) else [eos] if isinstance(eos, int) else [])
        self.stop_ids |= {vocab[t] for t in STOP_TOKENS if t in vocab}
        self.tool_ids = [vocab[t] for t in ("<tool_call>", "<think>") if t in vocab]
        self.drop = [t for t in (self.tok.all_special_tokens or []) if t not in KEEP_TEXT]
        self.adapter = HFTokenizer(self.tok)

    def tokenizer(self):
        return self.adapter

    def prompt(self, messages) -> str:
        if self.m.chat_format == "chatml":
            from tiny_agent.chat import render
            return render(messages, add_generation_prompt=True)
        return self.tok.apply_chat_template(chat_messages(messages), tokenize=False, add_generation_prompt=True)

    def _ids(self, messages):
        return self.tok(self.prompt(messages), add_special_tokens=False)["input_ids"]

    def count(self, messages):
        return len(self._ids(messages))

    @torch.no_grad()
    def generate(self, messages, *, max_tokens, temperature, top_p, rep_penalty=1.0, pen_ids=(), ban_tools=False,
                 logits_hook=None, stop=()):
        from tiny_agent.generate import sample_top_p
        ids = self._ids(messages)
        out = self.model(input_ids=torch.tensor([ids], device=self.device), use_cache=True)
        past, logits = out.past_key_values, out.logits[:, -1, :]
        V = logits.size(-1)
        pen_list = [i for i in pen_ids if 0 <= i < V]
        pm = None
        if pen_list and rep_penalty != 1.0:
            pm = torch.zeros(V, dtype=torch.bool, device=logits.device)
            pm[pen_list] = True
        bm = None
        if ban_tools and self.tool_ids:
            bm = torch.zeros(V, dtype=torch.bool, device=logits.device)
            bm[self.tool_ids] = True
        gen: list[int] = []
        for _ in range(max_tokens):
            if logits_hook is not None:
                logits = logits_hook(gen, logits)
            t = int(sample_top_p(logits, temperature, top_p, rep_penalty, pm, bm))
            if t in self.stop_ids:
                break
            gen.append(t)
            if stop and any(s in self.tok.decode(gen[-16:]) for s in stop):
                break
            out = self.model(input_ids=torch.tensor([[t]], device=self.device), past_key_values=past, use_cache=True)
            past, logits = out.past_key_values, out.logits[:, -1, :]
        text = self.tok.decode(gen, skip_special_tokens=False)
        for s in self.drop:
            text = text.replace(s, "")
        return cut_at_stops(text, stop).strip()
