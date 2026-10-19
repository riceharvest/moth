"""moth's conversation engine (moth.serve.Moth: history, reactions, sessions, the identity card, the phrase memory) on a
stateless backend (transformers, openai). Moth keeps tiny-agent's KV cache across turns and feeds only new text;
here every turn renders the whole (trimmed) chat and asks the backend for the next assistant turn. Everything above
the model is Moth's own code, unchanged: the same rollback, budget trimming, agent cards, one turn per session.
"""
from __future__ import annotations

from moth.backends.manifest import apply_args as apply_manifest
from moth.norepeat import PhraseMemory
from moth.serve import Moth

READY = "ready"   # stands in for the logits / cache of the stateful path: there is nothing to keep between turns


class GenericMoth(Moth):
    def __init__(self, a, manifest=None, backend=None):
        from moth.backends import load_backend, resolve
        self.manifest = manifest or getattr(a, "manifest", None) or resolve(a.model)
        a = apply_manifest(a, self.manifest)
        self.backend = backend or load_backend(self.manifest, a)
        self.tok = self.backend.tokenizer()
        self.model = self._kv = None
        self.quant = "none"
        self._reacting = False
        self.setup(a)
        if not self.backend.phrase_ban:   # no token-level sampling (openai): no phrase ban, no carried-over words
            self.phrases = PhraseMemory(0)

    # --- what the stateful path does with a KV cache ------------------------------------------------------------
    def _new_cache(self):
        return READY

    def _feed(self, ids):
        return READY   # the next _turn renders the whole chat; the budget is checked there

    def _prefill_all(self):
        """Drop the oldest turns until the chat fits CTX_KEEP-worth of tokens (as Moth._prefill_all)."""
        while len(self.messages) > 2 and self.backend.count(self._view()) >= self.keep:
            i = next((k for k in range(2, len(self.messages)) if self.messages[k]["role"] == "user"), None)
            if i is None:
                break
            self.messages = self.messages[:1] + self.messages[i:]
        self.cache = READY
        return READY

    def _turn(self, logits) -> str:
        if self.backend.count(self._view()) + self.a.max_turn_tokens > self.budget:
            self._prefill_all()
        return self.backend.generate(
            self._view(), max_tokens=self.a.max_turn_tokens, temperature=self.a.temperature, top_p=self.a.top_p,
            rep_penalty=getattr(self.a, "rep_penalty", 1.0), pen_ids=self._pen_ids(), ban_tools=True,
            logits_hook=self._hook, stop=self.manifest.stop)

    def _view(self) -> list[dict]:
        """The chat as the model sees it this turn: moth's bookkeeping keys out, and on a reaction the manifest's
        reaction nudge after the agent card (this turn only; history keeps the plain card)."""
        msgs = [{k: v for k, v in m.items() if k != "session"} for m in self.messages]
        if self._reacting and self.manifest.reaction:
            for m in reversed(msgs):
                if m["role"] == "user":
                    m["content"] = f"{m['content']}\n\n{self.manifest.reaction}"
                    break
        return msgs

    def react(self, text, hold_for_human=False, item=None):
        with self.lock:
            self._reacting = True
            try:
                return super().react(text, hold_for_human=hold_for_human, item=item)
            finally:
                self._reacting = False

