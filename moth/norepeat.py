"""moth's phrase memory: don't tell the same joke twice in a row.

The 112M model has favourite lines ("my great great great grandma was the first bug" came up in 5 of 17 live
reactions), and every reaction starts from a fresh-ish context (one agent turn per session; the live demo even reset
moth between takes), so nothing in the prompt tells it that it just said that. PhraseMemory keeps the token ids of
moth's last `keep` reactions. While moth writes a new one, a word that would complete an n-gram (n tokens) from one
of them is banned: at each step one dict lookup on the last n-1 sampled tokens gives the index tensor of words to
set to -inf (tiny_agent.generate.Decoder.generate's logits_hook). No per-step loop over the history. On a GPU the
same table goes into the captured decode graph as fixed-size device tensors (generate._phrase_ban_mask, filled once
per turn), so reactions keep the device-graph fast path.

Only word-start tokens ("Ġbug", not "ma" after "Ġgrand") are banned, so a phrase is rerouted at a word boundary
instead of being broken mid-word; punctuation and <|im_end|> are never banned (replies don't run on). n-grams of only
common words and punctuation ("oh. i'm a") are never banned, and neither are n-grams that the message moth is
answering contains (reactions quote the agent: "3 tests pass").

A ban only stops the exact wording: the model then tells the joke another way ("my grandma got taped into a
notebook", "great-great-great grandma" with hyphens). So for reactions moth.serve also widens its soft repetition
penalty (--rep-penalty, on moth's last 3 replies in this chat) to the words of the last 3 remembered reactions
(recent()): that is what makes the motif itself come back less often, also after a "new chat".

The memory is moth's own recent lines, in RAM only. It deliberately survives "new chat" (moth.serve.Moth.forget):
the same human sees every reaction. Chat replies are never stored (they can hold the human's name or facts), and
chat turns don't use the ban unless asked (moth.serve --phrase-ban-chat), so recall ("what's my name?") is untouched.
"""
from __future__ import annotations

from collections import deque

KEEP = 16   # reactions remembered
N = 3       # n-gram length: 3 tokens = a 2-token context + the banned word
CARRY = 3   # reactions whose words the soft repetition penalty also covers (recent())


class PhraseMemory:
    def __init__(self, keep: int = KEEP, n: int = N):
        self.n = max(2, int(n))
        self.replies: deque[list[int]] = deque(maxlen=max(1, int(keep)))
        self.on = keep > 0

    def __len__(self):
        return len(self.replies)

    def add(self, ids):
        if self.on and ids:
            self.replies.append(list(ids))

    def clear(self):
        self.replies.clear()

    def recent(self, k: int = CARRY) -> set[int]:
        """The ids of the last k remembered reactions: moth.serve adds them to the repetition penalty's words for a
        reaction, so that soft penalty reaches back over a "new chat" too (Moth.said doesn't)."""
        return {i for r in list(self.replies)[-k:] for i in r} if self.on and k > 0 else set()

    def table(self, word: set[int], boring: set[int], quote=()) -> dict[tuple, list[int]]:
        """{(n-1 context ids): [banned next ids]} for one turn. word: ids that may be banned (word starts);
        boring: common words + punctuation (an n-gram of only these is never banned); quote: the ids of the message
        being answered (its own n-grams stay allowed)."""
        n = self.n
        quote = list(quote)
        quoted = {tuple(quote[j - n + 1:j + 1]) for j in range(n - 1, len(quote))}
        tbl: dict[tuple, set[int]] = {}
        for r in self.replies:
            for j in range(n - 1, len(r)):
                if r[j] not in word:
                    continue
                g = tuple(r[j - n + 1:j + 1])
                if not (g in quoted or all(t in boring for t in g)):
                    tbl.setdefault(g[:-1], set()).add(r[j])
        return {k: sorted(v) for k, v in tbl.items()}

    def hook(self, word: set[int], boring: set[int], quote=()):
        """A Decoder.generate logits_hook for one turn, or None when nothing is banned."""
        tbl = self.table(word, boring, quote) if self.on else {}
        return BanHook(tbl, self.n) if tbl else None


class BanHook:
    """logits_hook(out, logits): -inf on the words that would complete a remembered n-gram after out[-(n-1):]."""

    def __init__(self, table: dict[tuple, list[int]], n: int):
        self.table, self.k = table, n - 1
        self._idx = None
        self.hits = 0   # how many steps had something banned (tests, measurements)

    def __call__(self, out, logits):
        key = tuple(out[-self.k:]) if len(out) >= self.k else None
        if key not in self.table:
            return logits
        if self._idx is None:   # index tensors once per turn, on the logits' device
            import torch
            self._idx = {k: torch.tensor(v, dtype=torch.long, device=logits.device) for k, v in self.table.items()}
        self.hits += 1
        return logits.index_fill(-1, self._idx[key], float("-inf"))


def vocab_classes(tok, common: set[int]) -> tuple[set[int], set[int]]:
    """(word, boring) for a byte-level BPE tokenizer: word = ids of tokens that start a new word with a letter or
    digit ("Ġbug"); boring = common-word ids + every id without a letter or digit (punctuation, whitespace,
    special tokens)."""
    word, boring = set(), set(common)
    for s, i in tok.get_vocab().items():
        if s.startswith("Ġ") and len(s) > 1 and s[1].isalnum():
            word.add(i)
        if not any(c.isalnum() for c in s.replace("Ġ", "").replace("Ċ", "")) or s.startswith("<|"):
            boring.add(i)
    return word, boring
