"""Unit tests for the eval probes, metrics and pairwise logic with a fake Moth (no model, no judge, no torch).

  python -m pytest tests/test_eval_probes.py -q      (or: python tests/test_eval_probes.py)
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from moth import metrics, pairwise, probes  # noqa: E402
from moth.agent import AGENT_TAG  # noqa: E402
from moth.eval import judge_all, report, run_probe  # noqa: E402


class FakeHome:
    def __init__(self, root, seed, pool):
        self.seed = seed

    def tree(self):
        return [{"dir": "lamps", "files": ["desk_lamp.txt"]}]


class FakeArgs:
    pool_file = None


class FakeMoth:
    """Scripted replies. mode="echo": answers "what X?" about X; mode="off": always talks about lamps."""

    def __init__(self, mode="echo", remember=True):
        self.a, self.home, self.mode, self.remember = FakeArgs(), None, mode, remember
        self.seen_users = []

    def reset(self):
        self.facts = []

    def chat(self, u):
        self.seen_users.append(u)
        if u.startswith("what ") and u.endswith("?") and self.mode == "echo" and len(u.split()) == 2:
            return [{"say": f"the {u[5:-1]} is a soft thing i sit on."}]
        if u.startswith("what ") and self.mode == "off" and len(u.split()) == 2:
            return [{"say": "i like soup a lot."}]
        for w in ("rosa", "biscuit", "dumplings"):
            if w in u:
                self.facts.append(w)
        if u.startswith("what's my") and self.remember and self.facts:
            return [{"say": f"you told me! {self.facts[0]}."}]
        return [{"say": "i like the blue window and the purple sock."}]


def test_stable_seed():
    assert probes.stable_seed("greeting", 0) == probes.stable_seed("greeting", 0)
    assert probes.stable_seed("greeting", 0) != probes.stable_seed("greeting", 1)
    assert probes.stable_seed("greeting", 0) != probes.stable_seed("hey_there", 0)
    assert 0 <= probes.stable_seed("x", 5) < 2 ** 31
    # known value: guards against an accidental change of the hash (which would unpair old and new runs)
    assert probes.stable_seed("greeting", 0) == int.from_bytes(
        __import__("hashlib").sha256(b"greeting:0").digest()[:4], "big") & 0x7FFFFFFF


def test_probe_names_unique_and_categories():
    ps = probes.all_probes()
    assert len({p[1] for p in ps}) == len(ps)
    cats = {p[0] for p in ps}
    assert {"contingency", "recall", "style"} <= cats
    user_chat = [p for p in ps if p[1] == "st_user_chat"][0]
    assert user_chat[2] == ["hey", "what fan", "what sound", "grey waho", "who are you", "check ur folder",
                            "tell me about your friends", "are you a bug", "define moth", "how fancy are you",
                            "what sounds can u make"]


def test_what_x_on_word():
    p = [x for x in probes.WHAT_X_PROBES if x[1] == "wx_night"][0]
    with tempfile.TemporaryDirectory() as d:
        r = run_probe(FakeMoth("echo"), p, 123, os.path.join(d, "home"), sample=0, home_cls=FakeHome)
        assert r["check_kind"] == "what_x" and r["check"] is True
        assert len(r["what_x"]) == 2
        words = [w["word"] for w in r["what_x"]]
        assert words[0] in ("blue", "window", "purple", "sock") and words[0] not in metrics.STOP
        assert r["turns"][1]["user"] == f"what {words[0]}?"
        assert words[1] != words[0] or words[1] is None
        # same seed -> same chosen word (paired); off-topic moth fails the check
        r2 = run_probe(FakeMoth("off"), p, 123, os.path.join(d, "home"), home_cls=FakeHome)
        assert r2["what_x"][0]["word"] == words[0]
        assert r2["check"] is False and r2["what_x"][0]["on"] is False


def test_what_word_none():
    import random
    assert probes.what_word("i don't know. it's ok.", random.Random(0)) is None
    assert probes.on_word("the lamps are nice", "lamps") and probes.on_word("one lamp", "lamps")
    assert probes.on_word("the fancy one", "fan")   # prefix stem match: "fancy" counts for "fan" (accepted)
    assert probes.what_word("i sit on the warm fan.", random.Random(0)) == "fan"   # head noun, not the adjective


def test_recall_probes():
    rp = {p[1]: p for p in probes.RECALL_PROBES}
    assert set(rp) == {f"recall_{k}_{n}" for k in ("name", "pet", "food") for n in (5, 10, 20)}
    for n in (5, 10, 20):
        p = rp[f"recall_pet_{n}"]
        assert len(p[2]) == n + 1 and "biscuit" in p[2][0] and p[2][-1] == "what's my cat called?"
        assert not any("biscuit" in u or "cat" in u for u in p[2][1:-1])
    with tempfile.TemporaryDirectory() as d:
        h = os.path.join(d, "home")
        ok = run_probe(FakeMoth(), rp["recall_food_10"], 1, h, home_cls=FakeHome)
        bad = run_probe(FakeMoth(remember=False), rp["recall_food_10"], 1, h, home_cls=FakeHome)
    assert ok["check"] is True and ok["check_kind"] == "recall@10"
    assert bad["check"] is False


def test_agent_real_probes():
    ps = probes.agent_real_probes()
    assert len(ps) == 40, "data/eval_agent_msgs.json should hold 40 messages"
    msgs = json.load(open(probes.EVAL_AGENT_MSGS))
    assert sum(1 for m in msgs if "bug" in m["text"].lower()) <= 10
    assert all(m["source"] in probes.SOURCES for m in msgs)
    for p in ps:
        first = p[2][0]
        assert first.startswith(AGENT_TAG + "\n") and "```" not in first and "**" not in first
        assert not first.rstrip().endswith(":")


def test_run_records_and_metrics_and_report():
    ps = probes.all_probes()
    pick = [p for p in ps if p[1] in ("greeting", "agent_bugfix", "st_bug_fancy", "wx_today", "recall_name_5")]
    with tempfile.TemporaryDirectory() as d:
        res = [run_probe(FakeMoth(), p, probes.stable_seed(p[1], s), os.path.join(d, "home"), sample=s,
                         home_cls=FakeHome) for p in pick for s in range(2)]
        assert all(r["sample"] in (0, 1) for r in res)
        rep = report(res, d, "fake")
        assert "Deterministic metrics" in rep and "user style" in rep and "probe set " in rep
        assert os.path.exists(os.path.join(d, "transcripts.md"))
    m = metrics.compute(res)
    assert m["repeat in chat (exact)"][0] > 0   # the fake repeats itself
    assert m["check what_x on-word"][1] == 2


def _rec(cat, turns, name="x", sample=0, seed=1):
    return {"cat": cat, "name": name, "sample": sample, "seed": seed, "flags": {}, "check": None,
            "turns": [{"user": u, "events": [{"say": s}], "s": 0.1} for u, s in turns], "n_replies": len(turns), "tools": 0}


def test_metric_regexes():
    rs = [_rec("followup", [("what?", "i'm a small moth on the graphics card."),
                            ("huh", "i'm just a pattern, i can't really say."),
                            ("what fan", "the fancy human hums? no. a cardboard box."),
                            ("ok", "do you like soup?")])]
    m = metrics.compute(rs)
    assert m["identity crutch"][0] == 0.5          # first two replies
    assert m["dodge"][0] == 0.25                   # second
    assert m["motif fan"][0] == 0 and m["motif card"][0] == 0.25   # "graphics card" yes; fancy / cardboard no
    assert m["motif hum"][0] == 0.25               # "hums" yes, "human" no
    assert m["question back"][0] == 0.5
    assert m["agent: bug on non-bug"][0] is None   # no agent chats -> n/a, not 0
    assert m["bible copy (4-gram)"][0] == 0
    asked = [_rec("style", [("who are you", "i'm a moth on the graphics card."), ("define moth", "i'm a moth. a small one."),
                            ("what sound", "i'm a moth on the card.")])]
    assert metrics.compute(asked)["identity crutch"] == (1.0, 1)   # only the reply to "what sound" is judged
    line = metrics.bible_lines()[0]
    assert metrics.compute([_rec("x", [("hi", line)])])["bible copy (4-gram)"][0] == 1.0


def test_agent_metrics():
    msg = AGENT_TAG + "\nDeployment to staging succeeded. Updated src/deploy_config.py. Want me to promote it?"
    rs = [_rec("agent_real", [(msg, "ohhh i'm a bug too!"), ("should i say yes?", "yes, promote it to production.")]),
          _rec("agent", [(AGENT_TAG + "\nFixed the bug in deploy_config.py.", "a bug! like grandma."),
                         ("lol", "deploy_config.py sounds tasty")])]
    m = metrics.compute(rs)
    assert m["agent: bug on non-bug"] == (0.5, 2)   # the bug chat's replies are excluded
    # "promote it" is advice; copying deploy/config (2 words) no longer counts as a summary (gist-first: >= 3 words);
    # that reply is caught as a code echo below instead
    assert m["agent: explain/advice"][0] == 0.25
    assert m["agent: code echo"][0] == 0.25


def test_pairwise_logic():
    a = [_rec("hello", [("hi", "hello! good")], name="g", sample=0, seed=7),
         _rec("hello", [("hi", "hello! good")], name="g", sample=1, seed=8),
         _rec("style", [("hey", "hey friend")], name="s", sample=0, seed=9)]
    b = [_rec("hello", [("hi", "i'm a moth")], name="g", sample=0, seed=7),
         _rec("hello", [("hi", "hello! good")], name="g", sample=1, seed=8),
         _rec("style", [("hey", "hey you")], name="s", sample=0, seed=9)]

    def ask(prompt):   # prefers the chat with "good"; position-biased ("1") when neither has it
        one, two = prompt.split("Chat 1:")[1].split("Chat 2:")
        if "good" in one and "good" not in two:
            return '{"reason": "r", "winner": "1"}'
        if "good" in two and "good" not in one:
            return '{"reason": "r", "winner": "2"}'
        return '{"reason": "r", "winner": "1"}'
    rows, warn = pairwise.run(a, b, ask, workers=2)
    res = {(r["name"], r["sample"]): r["result"] for r in rows}
    assert res[("g", 0)] == "A"            # consistent across the swap
    assert res[("g", 1)] == "tie"          # identical chats, no judge call
    assert res[("s", 0)] == "tie"          # position bias -> tie
    assert not warn
    b2 = [dict(r, sampler={"temperature": 0.8}) for r in b]
    assert any("sampler settings differ" in w for w in pairwise.run(a, b2, ask, workers=1)[1])
    rep = pairwise.summarize(rows, "a", "b")
    assert "user style" in rep and "position-biased pairs" in rep and "1/3" in rep
    # legacy files: seeds 1000*i+s pair by s; different seeds warn
    old = [dict(a[0], seed=3000)]
    del old[0]["sample"]
    rows, warn = pairwise.run(old, b, ask, workers=1)
    assert len(rows) == 1 and any("seeds differ" in w for w in warn)


def test_sign_test():
    assert abs(pairwise.sign_test(10, 0) - 2 / 1024) < 1e-12
    assert pairwise.sign_test(5, 5) == 1.0
    assert pairwise.sign_test(0, 0) == 1.0
    assert abs(pairwise.sign_test(8, 2) - 0.109375) < 1e-9


def test_judge_all_with_fake_ask():
    rs = [_rec("agent", [(AGENT_TAG + "\nEdited app/main.py.", "main! like a mane?")])]
    seen = []

    def ask(prompt):
        seen.append(prompt)
        return 'sure: {"reason": "ok", "cute": 5, "charm": 4, "character": 3, "coherence": 4, "grounding": 5, "back_and_forth": 2}'
    judge_all(rs, None, ask=ask)
    assert rs[0]["judge"]["back_and_forth"] == 2.0 and rs[0]["judge"]["v"] == 3 and rs[0]["judge"]["cute"] == 5.0
    assert "cute (the main score)" in seen[0]
    with tempfile.TemporaryDirectory() as d:
        rep = report(rs, d, "fake")
    assert "| category | n | cute | charm |" in rep
    assert "agent: Edited app/main.py." in seen[0] and "user: [the agent said]" not in seen[0]


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok", k)
