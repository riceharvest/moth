"""moth/norepeat.py (the phrase ban on moth's recent reactions) and Decoder.generate's logits_hook.

  python3 tests/test_norepeat.py      (from ~/moth; a second; the tokenizer tests need data/tiny-agent-112m)

The engine side (reactions stored, chat untouched, kept across "new chat") is in tests/test_engine.py.
"""
from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
from moth.norepeat import BanHook, PhraseMemory, vocab_classes  # noqa: E402

try:
    import torch
    if not hasattr(torch, "zeros"):   # an empty namespace package, not torch
        torch = None
except ImportError:   # pragma: no cover
    torch = None

TOKENIZER = os.path.join(ROOT, "data", "tiny-agent-112m", "tokenizer.json")
WORD = set(range(100, 200))          # toy vocab: 100-199 start a word, 200+ are subword pieces, 1-9 punctuation
BORING = set(range(1, 10)) | {100, 101}   # punctuation + two "common words"


class TableTest(unittest.TestCase):
    def test_bans_the_word_that_completes_a_remembered_ngram(self):
        mem = PhraseMemory(keep=4, n=3)
        mem.add([150, 151, 152, 153])
        self.assertEqual(mem.table(WORD, BORING), {(150, 151): [152], (151, 152): [153]})

    def test_never_bans_subwords_punctuation_or_all_common_ngrams(self):
        mem = PhraseMemory(keep=4, n=3)
        mem.add([150, 151, 250, 5, 100, 101, 100, 160])
        tbl = mem.table(WORD, BORING)
        self.assertNotIn((150, 151), tbl)   # 250: a subword piece ("ma" after "Ġgrand")
        self.assertNotIn((151, 250), tbl)   # 5: punctuation
        self.assertNotIn((5, 100), tbl)     # 5 100 101: punctuation + common words only
        self.assertNotIn((100, 101), tbl)   # 100 101 100: common words only
        self.assertEqual(tbl[(101, 100)], [160])   # ... but a common context before a real word counts

    def test_ngrams_the_answered_message_contains_stay_allowed(self):
        mem = PhraseMemory(keep=4, n=3)
        mem.add([150, 151, 152, 153])
        self.assertEqual(mem.table(WORD, BORING, quote=[9, 150, 151, 152]), {(151, 152): [153]})

    def test_keeps_only_the_last_replies_and_off_switch(self):
        mem = PhraseMemory(keep=2, n=3)
        for r in ([150, 151, 152], [160, 161, 162], [170, 171, 172]):
            mem.add(r)
        self.assertEqual(set(mem.table(WORD, BORING)), {(160, 161), (170, 171)})
        off = PhraseMemory(keep=0)
        off.add([150, 151, 152])
        self.assertEqual(len(off), 0)
        self.assertIsNone(off.hook(WORD, BORING))
        self.assertIsNone(PhraseMemory().hook(WORD, BORING))   # nothing remembered yet: no hook, no eager fallback

    def test_recent_words_for_the_soft_penalty(self):
        mem = PhraseMemory(keep=8)
        for r in ([150], [160], [170], [180]):
            mem.add(r)
        self.assertEqual(mem.recent(), {160, 170, 180})
        self.assertEqual(mem.recent(1), {180})
        self.assertEqual(PhraseMemory(keep=0).recent(), set())

    def test_longer_n(self):
        mem = PhraseMemory(keep=2, n=4)
        mem.add([150, 151, 152, 153])
        self.assertEqual(mem.table(WORD, BORING), {(150, 151, 152): [153]})


@unittest.skipIf(torch is None, "needs torch")
class HookTest(unittest.TestCase):
    def test_hook_sets_banned_words_to_minus_inf_only_after_the_context(self):
        h = BanHook({(150, 151): [152, 160]}, 3)
        lg = torch.zeros(1, 300)
        self.assertIs(h([], lg), lg)
        self.assertIs(h([150], lg), lg)
        self.assertIs(h([151, 150], lg), lg)
        out = h([7, 150, 151], lg)
        self.assertTrue(torch.isinf(out[0, 152]) and torch.isinf(out[0, 160]))
        self.assertEqual(int(torch.isinf(out).sum()), 2)
        self.assertEqual(float(lg.abs().sum()), 0.0)   # the input is not modified
        self.assertEqual(h.hits, 1)

    def test_decoder_generate_with_a_hook(self):
        """The hook sees the tokens sampled so far and its ban holds; graph="static" (the device loop) falls back to
        the eager loop when a hook is given; without a hook nothing changes."""
        import copy

        from test_fast_infer import tiny_model
        from tiny_agent import generate as G
        m = tiny_model(torch.float32, seed=3)
        c = G.KVCache(m, 1, 256, "cpu", dtype=torch.float32)
        logits = G.append(m, c, torch.tensor([[5, 6, 7, 8, 9, 10]]), [6])
        stop = m.cfg.vocab_size - 1
        ref = copy.deepcopy(c)
        greedy = G.Decoder(m, ref, graph=False).generate(logits.clone(), 12, stop, 0.0, 0.95)
        seen = []

        def ban_greedy(out, lg):
            seen.append(list(out))
            return lg.index_fill(-1, torch.tensor([greedy[len(out)]]), float("-inf"))

        for graph in (False, "static"):
            cc = copy.deepcopy(c)
            seen.clear()
            got = G.Decoder(m, cc, graph=graph).generate(logits.clone(), 12, stop, 0.0, 0.95, logits_hook=ban_greedy)
            self.assertNotEqual(got[0], greedy[0])
            self.assertEqual(seen[0], [])
            self.assertEqual(seen[1], got[:1])
            self.assertEqual(int(cc.lens_cpu[0]), 6 + len(got) + (got[-1] != stop))
        cc = copy.deepcopy(c)
        self.assertEqual(G.Decoder(m, cc, graph=False).generate(logits.clone(), 12, stop, 0.0, 0.95), greedy)

    def test_device_mask_equals_the_hook(self):
        """_phrase_ban_mask (the graph path's on-device ban) bans exactly what BanHook bans, for every step."""
        from tiny_agent import generate as G
        g = torch.Generator().manual_seed(0)
        V = 40
        for n in (2, 3, 4):
            k = n - 1
            for trial in range(20):
                table = {}
                for _ in range(int(torch.randint(1, 12, (1,), generator=g))):
                    ctx = tuple(torch.randint(0, 6, (k,), generator=g).tolist())
                    table.setdefault(ctx, set()).add(int(torch.randint(0, V, (1,), generator=g)))
                table = {c: sorted(v) for c, v in table.items()}
                rows = [(c, t) for c, ts in table.items() for t in ts]
                cap = 32
                ban_ctx = torch.full((cap, k), -1, dtype=torch.long)
                ban_tok = torch.full((cap,), V, dtype=torch.long)
                ban_ctx[:len(rows)] = torch.tensor([c for c, _ in rows])
                ban_tok[:len(rows)] = torch.tensor([t for _, t in rows])
                seq = torch.randint(0, 6, (15,), generator=g).tolist()
                out = torch.tensor(seq + [0] * 5)
                hook = BanHook(table, n)
                for i in range(len(seq) + 1):
                    want = torch.isinf(hook(seq[:i], torch.zeros(1, V)))
                    got = G._phrase_ban_mask(out, torch.tensor([i]), ban_ctx, ban_tok, torch.zeros(V + 1),
                                             torch.arange(k))
                    self.assertTrue(torch.equal(got, want), (n, trial, i))

    def test_ban_table_hook_keeps_the_device_loop(self):
        """A BanHook (it has a table) runs in the device-graph path (graph="static": the captured step, uncaptured on
        the CPU) and gives the same tokens as the eager loop with the hook; capacity grows when a table is big."""
        import copy

        from test_fast_infer import tiny_model
        from tiny_agent import generate as G
        m = tiny_model(torch.float64, seed=3)
        c = G.KVCache(m, 1, 256, "cpu", dtype=torch.float64)
        logits = G.append(m, c, torch.tensor([[5, 6, 7, 8, 9, 10]]), [6])
        never = m.cfg.vocab_size - 1
        greedy = G.Decoder(m, copy.deepcopy(c), graph=False).generate(logits.clone(), 20, never, 0.0, 0.95)
        table = {(greedy[2], greedy[3]): [greedy[4]], (greedy[8], greedy[9]): [greedy[10], 11]}
        for extra in (0, 600):        # 600 dummy rows: more than Decoder.BAN_CAP, so the graph is rebuilt bigger
            tbl = dict(table)
            tbl.update({(1000 + j, 1000 + j): [1] for j in range(extra)})   # contexts that never occur
            eager = G.Decoder(m, copy.deepcopy(c), graph=False).generate(
                logits.clone(), 20, never, 0.0, 0.95, logits_hook=BanHook(tbl, 3))
            self.assertNotEqual(eager, greedy)
            self.assertEqual(eager[:4], greedy[:4])
            self.assertNotEqual(eager[4], greedy[4])
            dec = G.Decoder(m, copy.deepcopy(c), graph="static")
            got = dec.generate(logits.clone(), 20, never, 0.0, 0.95, logits_hook=BanHook(tbl, 3))
            self.assertEqual(got, eager)
            self.assertGreaterEqual(dec._ban_ctx.size(0), len(tbl) + 1)
            # the next turn on the same decoder without a hook: the table is cleared (same tokens as the eager loop)
            lg = dec.step(7)
            sf, dec.cache.static_feed = dec.cache.static_feed, None   # (holds closures: not deep-copyable)
            ref = copy.deepcopy(dec.cache)
            dec.cache.static_feed = sf
            plain = G.Decoder(m, ref, graph=False).generate(lg.clone(), 12, never, 0.0, 0.95)
            self.assertEqual(dec.generate(lg.clone(), 12, never, 0.0, 0.95), plain)
        dec = G.Decoder(m, copy.deepcopy(c), graph="static")
        self.assertEqual(dec.generate(logits.clone(), 20, never, 0.0, 0.95), greedy)

        # a backend that can't capture the ban ops: ban turns go eager (same tokens), plain turns keep the graph path
        real = G.Decoder._setup

        def setup_without_ban(self, *a, **kw):
            cap = kw.get("ban_cap", a[5] if len(a) > 5 else None)
            if cap is None:
                cap = self.BAN_CAP if self.ban_graph else 0
            if cap:
                raise RuntimeError("cannot capture index_add_")
            return real(self, *a, **kw)

        G.Decoder._setup = setup_without_ban
        try:
            import warnings
            dec = G.Decoder(m, copy.deepcopy(c), graph="static")
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                got = dec.generate(logits.clone(), 20, never, 0.0, 0.95, logits_hook=BanHook(table, 3))
            eager = G.Decoder(m, copy.deepcopy(c), graph=False).generate(
                logits.clone(), 20, never, 0.0, 0.95, logits_hook=BanHook(table, 3))
            self.assertEqual(got, eager)
            self.assertFalse(dec.ban_graph)
            self.assertTrue(dec.graph)
            dec2 = G.Decoder(m, copy.deepcopy(c), graph="static")
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                self.assertEqual(dec2.generate(logits.clone(), 20, never, 0.0, 0.95), greedy)
            self.assertEqual(dec2.mode, "static")
            self.assertEqual(dec2._key[3], 0)      # captured without the ban ops
        finally:
            G.Decoder._setup = real


@unittest.skipUnless(os.path.isfile(TOKENIZER), f"needs {TOKENIZER}")
class TokenizerTest(unittest.TestCase):
    def test_the_grandma_joke_is_rerouted_at_word_starts(self):
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(TOKENIZER)
        enc = lambda s: tok.encode(s, add_special_tokens=False).ids   # noqa: E731
        common = {i for w in ("i", "a", "the", "was", "my") for v in (w, " " + w) for i in enc(v)}
        word, boring = vocab_classes(tok, common)
        tid = tok.token_to_id
        self.assertIn(tid("Ġbug"), word)
        self.assertNotIn(tid("ma"), word)           # the piece after "Ġgrand"
        self.assertNotIn(tid("<|im_end|>"), word)
        self.assertIn(tid("<|im_end|>"), boring)
        self.assertIn(tid("."), boring)
        mem = PhraseMemory(keep=4, n=3)
        mem.add(enc("my great great great grandma was the first bug."))
        tbl = mem.table(word, boring)
        g = lambda s: tuple(enc(s))   # noqa: E731
        self.assertIn(tid("Ġgrand"), tbl[g(" great great")])
        self.assertIn(tid("Ġfirst"), tbl[g(" was the")])
        self.assertFalse(any(tid("ma") in v or tid(".") in v for v in tbl.values()))


if __name__ == "__main__":
    unittest.main()
