"""The multi-turn RL environment (moth.rl) in mock mode: registry + split + frozen evals, simulator, judge, rewards,
episodes, groups, adapters, eval runner, cost meter. CPU only, no endpoint (one test talks to tests/fake_oai.py).

  python -m pytest tests/test_rl_env.py -q      (or: python tests/test_rl_env.py)
"""
import collections
import contextlib
import io
import json
import math
import os
import random
import socket
import subprocess
import sys
import tempfile
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from moth import cards as C  # noqa: E402
from moth.agent import AGENT_TAG  # noqa: E402
from moth.rl import adapters, env, evalrun, llm, registry  # noqa: E402
from moth.rl import reward as R  # noqa: E402
from moth.rl.judge import JudgeCache, LLMJudge, MockJudge  # noqa: E402
from moth.rl.sim import LLMSimulator, MockSimulator  # noqa: E402


def tmp_reg():
    return registry.Registry(tempfile.mkdtemp(prefix="moth-rl-"))


def mock_parts():
    return MockSimulator(), MockJudge()


def spec_of(scenario, card=None, seed=7, **kw):
    card = card or {**C.heldout_cards()[0], "split": "heldout"}
    return registry.make_spec(card, scenario, random.Random(seed), seed, **kw)


# --- registry -------------------------------------------------------------------------------------------------------
def test_builtins_are_pinned_and_bucket_is_stable():
    reg = tmp_reg()
    pool_ids = {c["id"] for c in C.build_pool()}
    assert {c["id"] for c in reg.cards("train")} >= pool_ids | {"moth", "blank"}
    assert {c["id"] for c in reg.cards("heldout")} == {c["id"] for c in C.heldout_cards()}
    assert {p["tmpl"] for p in reg.phrasings("train")} == set(C.PHRASINGS)
    assert {p["tmpl"] for p in reg.phrasings("heldout")} == set(C.HELDOUT_PHRASINGS)
    assert registry.bucket("card:x") == registry.bucket("card:x") == 76   # sha256, not the salted hash()
    share = sum(registry.bucket(f"card:u{i}") < registry.HELDOUT_PCT for i in range(4000)) / 4000
    assert 0.08 < share < 0.12


def test_appended_cards_split_by_hash_and_never_move():
    reg = tmp_reg()
    nm = lambda i, p: p + "abcdefghijklmnopqrstuvwxyz"[i % 26] + "aeiou"[i // 26]  # noqa: E731  distinct letter names
    first = [reg.add_line(f"you are a frog named {nm(i, 'zz')}") for i in range(40)]
    assert all(c["split"] == registry.card_split({k: v for k, v in c.items() if k != "split"}) for c in first)
    before = {c["id"]: c["split"] for c in first}
    for i in range(40):
        reg.add_line(f"you're a teapot called {nm(i, 'qq')}")
    reg2 = registry.Registry(reg.root)   # a fresh load of the same files
    assert {c["id"]: reg2.card(c["id"])["split"] for c in first} == before
    assert reg.add_line(f"you are a frog named {nm(0, 'zz')}") is None   # same line = same id: not added twice
    assert sum(1 for line in open(reg.cards_path, encoding="utf-8") if line.strip()) == 80


def test_heldout_identities_cannot_leak_into_train():
    reg = tmp_reg()
    for line in ["you are a hedgehog named spike", "you are a weather vane", "you are a koala called kim",
                 "you are a dog named wren", "Your name is Hazel. You are a duck."]:
        assert reg.add_line(line)["split"] == "heldout", line
    assert reg.add_line("you are a cat named bo")["split"] in ("train", "heldout")
    assert not registry.collides_heldout(registry.card_from_line("you are a lamp named lumi"))   # lamp != lava lamp
    with pytest.raises(ValueError):
        reg.add_line("you are a cat named luna")   # an eval friend's name


def test_card_from_line_tells_the_judge_what_it_is():
    c = registry.card_from_line("you are a cat named bo.")
    assert c["kind"] == "a small cat" and c["name"] == "bo" and C.describe(c) == "bo, a small cat"
    assert C.student_system(c).startswith("you are a cat named bo.")
    assert registry.kind_of("You are a rubber duck.") == "a rubber duck"
    assert registry.kind_of("you are a lighthouse keeper named ottilie") is None
    v = registry.card_from_line("be a sleepy volcano.")
    assert "sleepy volcano" in C.describe(v)   # not "no particular identity"


def test_phrasings_and_sampling_respect_the_split():
    reg = tmp_reg()
    p = reg.add_phrasing("you go by {name}; you're {kind}.")
    assert p["id"] == registry.phrasing_id("you go by {name}; you're {kind}.")
    assert reg.add_phrasing("you go by {name}; you're {kind}.") is None
    with pytest.raises(ValueError):
        reg.add_phrasing("you are {species}")
    held_card_ids = {c["id"] for c in reg.cards("heldout")}
    held_ph = {p["id"] for p in reg.phrasings("heldout")}
    rng = random.Random(0)
    for _ in range(400):
        c = reg.sample_card(rng, "train")
        assert c.get("split", "train") == "train"
        assert c.get("base_id", c["id"]) not in held_card_ids and c.get("phrasing_id") not in held_ph
    hc = [reg.sample_card(rng, "heldout") for _ in range(50)]
    assert all(c["split"] == "heldout" for c in hc) and any("phrasing_id" in c for c in hc)


def test_freeze_snapshots_refuses_overwrite_and_checks_hash():
    reg = tmp_reg()
    path = registry.freeze("t1", reg, reg.root, per_card=1)
    doc = registry.load_eval("t1", reg.root)
    s = doc["specs"][0]
    assert s["system"] == C.student_system(s["card"]) and s["card"]["id"] == s["card_id"]
    assert {x["card_id"].split("~")[0] for x in doc["specs"]} >= {c["id"] for c in reg.cards("heldout")}
    with pytest.raises(FileExistsError):
        registry.freeze("t1", reg, reg.root)
    reg.add_line("you are a hedgehog named spike")   # the pool grows; the frozen set does not
    assert registry.load_eval("t1", reg.root)["sha256"] == doc["sha256"]
    d = llm.read_json(path)
    d["specs"][0]["system"] = "you are someone else"
    llm.write_json(path, d)
    with pytest.raises(ValueError):
        registry.load_eval("t1", reg.root)


def test_committed_evals_v3_current_v1_v2_legacy_and_training_agent_messages():
    doc = registry.load_eval("v3")
    assert doc["n"] == len(doc["specs"]) and {s["scenario"] for s in doc["specs"]} == {"talk", "agent", "recall",
                                                                                         "script"}
    assert not any("home" in s and isinstance(s["home"], dict) for s in doc["specs"])
    assert all(s["system"] == C.student_system(s["card"]) for s in doc["specs"])   # frozen with the current spec
    d4 = registry.load_eval("v4")   # the multilingual set: every language sliced evenly, probes in English
    assert {s["lang"] for s in d4["specs"] if s["scenario"] != "script"} == {"en", "zh", "es", "fr", "hi"}
    for v in sorted(registry.LEGACY_EVALS):   # kept untouched (hash still checks), marked legacy
        assert registry.load_eval(v)["sha256"]
    reg = registry.Registry()
    train = {c["id"] for c in reg.cards("train")}
    assert not {s["card_id"] for s in doc["specs"] if s["card_id"] not in ("moth", "blank")} & train
    held = registry.eval_agent_texts()
    if held:
        from moth.agent import clean_agent_message
        cleaned = {clean_agent_message(t) for t in held}
        assert all(m in cleaned for s in doc["specs"] for m in s["agent_msgs"])
        pool = os.path.join(tempfile.mkdtemp(), "pool.json")
        llm.write_json(pool, [{"text": next(iter(held))}, {"text": "Done, all green."}])
        assert registry.train_agent_msgs(pool) == ["Done, all green."]


def test_registry_cli():
    root = tempfile.mkdtemp()
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        registry.main(["--root", root, "add", "--line", "you are a cat named bo"])
        f = os.path.join(root, "c.json")
        llm.write_json(f, [{"name": "pim", "kind": "a kettle", "home": "in a cafe"}])
        registry.main(["--root", root, "add", "--file", f])
        registry.main(["--root", root, "add-phrasing", "Character: {Name} ({noun})."])
        registry.main(["--root", root, "freeze", "v9", "--per-card", "1"])
        registry.main(["--root", root, "list"])
    o = out.getvalue()
    assert "bo, a small cat" in o and "pim, a kettle" in o and "eval_v9.json" in o


def test_teacher_batch_parsing():
    reg = tmp_reg()
    ask = lambda p: 'sure: ["you are a moose named ola.", "You are Bram, a toaster.", 7, "Name: Zed. You are a bat."]'  # noqa: E731
    added = registry.add_teacher_batch(reg, ask, 4)
    assert [c["name"] for c in added] == ["ola", "zed"]   # bram is an eval friend's name; 7 is not a line
    assert registry.teacher_lines(lambda p: "no json here", 3, set()) == []


# --- reward ---------------------------------------------------------------------------------------------------------
def test_banned_list_matches_build_sft():
    try:
        from moth.build_sft import BANNED
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"build_sft not importable here ({e})")
    assert R.BANNED == BANNED


def test_reward_components():
    assert R.c_length("yay!") == 0.0 and R.c_length("") == -1.0 and R.c_length(" ".join(["w"] * 40)) == -1.0
    assert -1 < R.c_length(" ".join(["w"] * 30)) < 0
    assert R.c_format("hi there") == 0 and R.c_format("Hi there") == -1 and R.c_format("- a list") == -1
    assert R.c_format("hi *flaps wings*") == -1 and R.c_format("line one\nline two") == -1
    assert R.c_assistant("great question! let me know", "hi") == -1
    assert R.c_assistant("you should restart it", "x") == -1
    assert R.c_assistant("it fixed it! yay", AGENT_TAG + "\nFixed auth/session.py") == 0
    assert R.c_assistant("yay, auth/session.py is fixed", AGENT_TAG + "\nFixed auth/session.py") == -1
    assert R.c_repeat("the soup is warm", "x", ["the soup is warm"]) == -1
    assert R.c_repeat("i love warm soup today", "x", []) == 0
    assert -1 < R.c_repeat("i love warm soup and bread", "x", ["we both love warm soup a lot"]) < 0
    assert R.c_repeat("you love warm soup!", "do you love warm soup?", ["i love warm soup"]) == 0   # quoted
    card = C.heldout_cards()[0]   # hazel, a hedgehog
    sc = R.score_replies([{"text": "i'm hazel, a hedgehog.", "user": "who are you?"},
                          {"text": "i'm a moth.", "user": "what are you?"},
                          {"text": "ok", "user": "nice"}], card, None)
    assert [s["vec"]["card"] for s in sc] == [1.0, -1.0, None]


def test_on_gist_self_beats_nothing_and_judge_failure_masks():
    reps = [{"text": "oh no! was the cat sorry?", "user": "my cat knocked my tea over"}]
    on_self = R.score_replies(reps, C.MOTH_CARD, [{"on": True, "self": True, "why": ""}])[0]["scalar"]
    off = R.score_replies(reps, C.MOTH_CARD, [{"on": False, "self": False, "why": ""}])[0]["scalar"]
    on = R.score_replies(reps, C.MOTH_CARD, [{"on": True, "self": False, "why": ""}])[0]["scalar"]
    assert on_self < off < on
    failed = R.score_replies(reps, C.MOTH_CARD, None)[0]
    assert failed["vec"]["on_gist"] is None and failed["vec"]["self"] is None and not failed["judge_ok"]
    assert failed["mask"]["on_gist"] is False and failed["scalar"] == 0.0
    assert R.episode_score([{"scalar": 1.0}, {"scalar": -1.0}], "min") == -1.0
    assert R.episode_score([], "mean") is None


# --- judge ----------------------------------------------------------------------------------------------------------
def test_judge_one_call_per_chat_cached_and_failure():
    spec = spec_of("talk")
    sim, judge = MockSimulator(), MockJudge()
    m = llm.CostMeter()
    ep = env.run_episode(spec, env.MockPolicy(), sim, judge, m)
    assert m.roles["judge"]["calls"] == 1 and len(ep["labels"]) == len(ep["replies"]) > 0
    m2 = llm.CostMeter()
    ep2 = env.run_episode(spec, env.MockPolicy(), sim, judge, m2)   # same chat again: a cache hit
    assert ep2["labels"] == ep["labels"] and m2.roles["judge"]["calls"] == 0 and m2.roles["judge"]["cached"] == 1
    path = os.path.join(tempfile.mkdtemp(), "cache.jsonl")
    j = MockJudge(JudgeCache(path))
    env.run_episode(spec, env.MockPolicy(), sim, j)
    assert len(JudgeCache(path)) == 1   # persisted, resumable
    m3 = llm.CostMeter()
    bad = env.run_episode(spec, env.MockPolicy(), sim, MockJudge(fail=True), m3)
    assert bad["labels"] is None and not bad["judge_ok"] and m3.roles["judge"]["calls"] == 3
    assert all(r["vec"]["on_gist"] is None for r in bad["replies"])


# --- episodes -------------------------------------------------------------------------------------------------------
def test_mock_episodes_all_scenarios():
    sim, judge = mock_parts()
    reg = tmp_reg()
    for scen in ("talk", "recall", "agent"):
        spec = spec_of(scen, seed=11)
        assert "home" not in spec   # a pure chatbot: no home folder
        m = llm.CostMeter()
        ep = env.run_episode(spec, env.MockPolicy(), sim, judge, m)
        assert ep["replies"] and ep["reward"] is not None and ep["judge_ok"]
        assert ep["turns"][0]["user"] and len(ep["turns"]) <= len(env._plan(spec))
        assert all(set(r["vec"]) == set(R.COMPONENTS) for r in ep["replies"])
        assert ep["cost"]["roles"]["judge"]["calls"] == 1
        if scen == "agent":
            assert ep["turns"][0]["user"].startswith(AGENT_TAG)
            assert m.roles.get("sim", {}).get("calls", 0) == sum(spec["follow"])
        if scen == "recall":
            ask = [r for r in ep["replies"] if r["vec"]["recall"] is not None]
            assert len(ask) == 1 and ask[0]["vec"]["recall"] == 1.0   # MockPolicy("good") recalls
    sel = env.run_episode(spec_of("talk", seed=11), env.MockPolicy("selfy"), sim, judge)
    from moth.gist import ASKED
    assert all(r["vec"]["self"] == -1.0 for r in sel["replies"] if not ASKED.search(r["user"]))
    asst = env.run_episode(spec_of("talk", seed=11), env.MockPolicy("assistant"), sim, judge)
    assert all(r["vec"]["assistant"] == -1 and r["vec"]["format"] == -1 for r in asst["replies"])
    eps = env.run_episodes([registry.sample_spec(reg, random.Random(i)) for i in range(4)],
                           lambda: env.MockPolicy(), sim, judge, workers=2)
    assert len(eps) == 4


def test_policy_extras_and_canonical_messages():
    class FakeMoth:   # moth.serve.Moth's surface: system, reset(), chat() -> [{"say": ...}]
        def __init__(self):
            self.system, self.users = None, []

        def reset(self):
            self.users = []

        def chat(self, u, talk_only=False):
            self.users.append(u)
            return [{"say": "oh! tell me more?"}]

    fm = FakeMoth()
    spec = spec_of("talk", seed=3)
    spec["script"], spec["turns"] = ["hi", "what's up?", "nice"], 3
    ep = env.run_episode(spec, env.MothPolicy(fm), *mock_parts())
    assert fm.system == spec["system"] and fm.users == spec["script"]
    assert len(ep["replies"]) == 3 == len(ep["labels"])
    assert [m["role"] for m in ep["messages"]] == ["user", "assistant"] * 3
    assert [r["assistant_index"] for r in ep["replies"]] == [0, 1, 2]
    from moth.gist import chat_for_label_msgs
    assert chat_for_label_msgs(ep["messages"])[1] == 3
    # a trainer's policy returns extras (token ids, logprobs): stored verbatim per turn
    tp = env.FnPolicy(lambda msgs: {"text": "Oh, Nice!", "raw": "Oh, Nice!", "tokens": [1, 2], "logp": [-0.1, -0.2]})
    ep2 = env.run_episode(spec, tp, *mock_parts())
    assert ep2["turns"][0]["extra"] == {"tokens": [1, 2], "logp": [-0.1, -0.2]}
    assert ep2["replies"][0]["vec"]["format"] == -1   # raw had capitals
    assert env.run_episode(spec, lambda msgs: "ok", *mock_parts())["replies"][0]["text"] == "ok"


def test_group_rollout_shares_the_first_message_and_advantages():
    sim, judge = mock_parts()
    spec = spec_of("talk", seed=5)
    m = llm.CostMeter()
    k = {"i": 0}

    def varied(msgs):   # different replies per rollout so the group has signal
        k["i"] += 1
        return ["oh no, the tea! was it hot?", "i'm just a little hedgehog. leaves!", "Great question!"][k["i"] % 3]
    g = env.rollout_group(spec, varied, 3, sim, judge, m)
    assert len({ep["turns"][0]["user"] for ep in g}) == 1
    sim_calls = m.roles["sim"]["calls"]
    assert sim_calls <= 1 + 3 * (spec["turns"] - 1)
    adv = env.group_advantages(g, "utterance")
    assert [len(a) for a in adv] == [len(ep["replies"]) for ep in g]
    first = [a[0] for a in adv]
    assert abs(sum(first)) < 1e-6 and max(first) > 0 > min(first)   # reply 1: same context, mean-zero baseline
    eadv = env.group_advantages(g, "episode")
    assert all(len(set(row)) == 1 for row in eadv)
    g[0]["replies"][0]["judge_ok"] = False
    assert env.group_advantages(g, "utterance")[0][0] is None


# --- anti-blandness -------------------------------------------------------------------------------------------------
def test_bland_policy_cannot_win_on_gist_alone():
    sim, judge = mock_parts()
    spec = spec_of("talk", seed=21)
    def sft_ref(msgs):   # a frozen reference that picks up a word of the message
        w = [x for x in msgs[-1]["content"].lower().replace("?", "").split() if len(x) > 3] or ["day"]
        return f"oh, {w[-1]}? and then what?"
    ref = env.FnPolicy(sft_ref)
    k = {"i": 0}

    def mixed(msgs):   # half the group bland, half specific
        k["i"] += 1
        u = msgs[-1]["content"]
        w = [x for x in u.lower().replace("?", "").split() if len(x) > 3] or ["day"]
        return "aw that's rough" if k["i"] % 2 else f"wait, {w[0]}? was the {w[-1]} a good one or a weird one?"
    g = env.rollout_group(spec, mixed, 4, sim, judge, ref_policy=ref)
    bland = [r for e in g for r in e["replies"] if r["text"] == "aw that's rough"]
    spec_r = [r for e in g for r in e["replies"] if r["text"] != "aw that's rough"]
    assert bland and spec_r
    for r in bland:
        assert r["vec"]["specific"] == -1 and r["vec"]["vs_ref"] is not None and r["vec"]["group_dup"] < 0
    assert sum(r["vec"]["vs_ref"] for r in bland) < 0   # mostly loses to the reference (the mock is crude)
    mean = lambda rs: sum(r["scalar"] for r in rs) / len(rs)  # noqa: E731
    assert mean(bland) <= -2 < mean(spec_r) - 1
    # on-gist alone does not win: a bland reply the judge calls ON still scores below a specific one
    reps = [{"text": "aw that's rough", "user": "my boss yelled at me"},
            {"text": "he yelled? at work? did you yell back or hide in the loo?", "user": "my boss yelled at me"}]
    on_bland = R.score_replies(reps[:1], C.MOTH_CARD, [{"on": True, "self": False, "specific": False}], prefs=[-1])
    on_spec = R.score_replies(reps[1:], C.MOTH_CARD, [{"on": True, "self": False, "specific": True}], prefs=[1])
    assert on_bland[0]["scalar"] == -1.0 and on_spec[0]["scalar"] == 2.0
    off = R.score_replies(reps[1:], C.MOTH_CARD, [{"on": False, "self": False, "specific": True}])[0]["scalar"]
    self_best = R.score_replies(reps[1:], C.MOTH_CARD, [{"on": True, "self": True, "specific": True}], prefs=[1])
    assert self_best[0]["scalar"] < off   # unprompted identity stays the worst move
    # the reference reply for the shared first message is asked once for the whole group
    assert len({e["turns"][0]["ref"] for e in g}) == 1


def test_group_dup_and_stock_lines():
    assert R.stock_reply("aw, that's rough!") and R.stock_reply("haha nice") and not R.stock_reply("pasta again?")
    assert R.stock_reply("哈哈", "zh") and R.near_dup("oh no", "tell me more")   # two stock lines are the same move
    assert not R.near_dup("the tea! was the cat sorry?", "a new desk? do you get a plant?")
    sim, judge = mock_parts()
    same = env.rollout_group(spec_of("talk", seed=5), lambda m: "oh! tell me more about it?", 3, sim, judge)
    assert all(r["vec"]["group_dup"] == -1.0 for e in same for r in e["replies"])
    solo = env.run_episode(spec_of("talk", seed=5), lambda m: "oh!", sim, judge)
    assert all(r["vec"]["group_dup"] is None and r["vec"]["vs_ref"] is None for r in solo["replies"])


def test_pairwise_prefs_positions_and_parse():
    from moth.rl import judge as J
    turns = [{"user": "i made pasta", "events": [{"say": "the long wiggly kind? did you eat it all?"}]},
             {"user": "yes", "events": [{"say": "haha nice"}]}]
    refs = ["haha nice", "wow, all of it? are you a pasta champion now?"]
    prefs = MockJudge().prefer(C.MOTH_CARD, turns, refs, seed=3)
    assert prefs == [1, -1]
    chat, n = J.pair_chat(turns, refs, 3)
    assert n == 2 and "[P1] X:" in chat and "Y:" in chat
    seen = {J.flip(s, 0) for s in range(20)}
    assert seen == {True, False}   # positions really are shuffled
    mine_is_y = J.flip(3, 0)
    txt = json.dumps({"pairs": [{"n": 1, "winner": "Y" if mine_is_y else "X"}, {"n": 2, "winner": "tie"}]})
    assert J.parse_pairs(txt, 2, 3) == [1, 0]
    with pytest.raises(ValueError):
        J.parse_pairs('{"pairs": [{"n": 1, "winner": "maybe"}]}', 1, 3)
    p = J.rl_label_prompt("friend: hi\n[R1] hi!", 1, C.MOTH_CARD)
    assert '"specific"' in p and "three things" in p
    with pytest.raises(ValueError):
        J.parse_rl_labels('{"labels": [{"n": 1, "on": true, "self": false}]}', 1)   # specific is required


def test_forward_pass_judge():
    from moth.rl import judge as J

    class FakeClient:   # P(yes) from the question; P(A) for pairs: A wins when A is the longer reply
        model = "jev-fake"

        def chat(self, messages, **kw):
            p = messages[0]["content"]
            if "Answer A or B" in p:
                a = p.split("\nA: ")[1].split("\nB: ")[0]
                b = p.split("\nB: ")[1].split("\nWhich")[0]
                ya = 0.9 if len(a) > len(b) else 0.1
                top = [{"token": "A", "logprob": math.log(ya)}, {"token": "B", "logprob": math.log(1 - ya)}]
            else:
                yes = 0.2 if "own identity" in p else 0.8
                top = [{"token": " yes", "logprob": math.log(yes)}, {"token": " no", "logprob": math.log(1 - yes)}]
            return {"text": "", "logprobs": {"content": [{"top_logprobs": top}]}, "usage": {}}
    j = J.ForwardPassJudge(FakeClient())
    turns = [{"user": "i made pasta", "events": [{"say": "the long wiggly kind?"}]}]
    labs = j.label(C.MOTH_CARD, turns)
    assert labs[0]["on"] and labs[0]["specific"] and not labs[0]["self"] and labs[0]["p"]["on"] == 0.8
    assert j.prefer(C.MOTH_CARD, turns, ["ok"], seed=1) == [1]
    assert j.model_id == "fwd:jev-fake"
    assert J.token_prob({"content": [{"top_logprobs": [{"token": "maybe", "logprob": 0.0}]}]}, ("yes",), ("no",)) is None


# --- multilingual -----------------------------------------------------------------------------------------------------
def test_language_aware_checks():
    assert R.c_format("我今天很开心 OK", "zh") == 0 and R.c_format("Hola, ¿qué tal?", "es") == -1
    assert R.c_format("नमस्ते दोस्त", "hi") == 0 and R.c_format("salut, ça va ?", "fr") == 0
    assert R.c_assistant("como ia, no puedo", "hola", "es") == -1 and R.c_assistant("作为一个人工智能，我不能", "你好", "zh") == -1
    assert R.c_assistant("मैं आपकी कैसे मदद कर सकता हूँ", "नमस्ते", "hi") == -1
    assert R.c_assistant("¡qué rico! ¿con queso?", "hice pasta", "es") == 0
    assert R.c_length("我" * 30) == 0 and R.c_length("我" * 80) == -1   # CJK counted by characters
    assert R.c_repeat("今天天气很好啊", "x", ["今天天气很好啊"]) == -1
    sim, judge = mock_parts()
    spec = spec_of("talk", seed=6)
    spec["lang"] = "zh"
    ep = env.run_episode(spec, lambda m: "哈哈 OK", sim, judge)
    assert ep["lang"] == "zh" and all(r["vec"]["format"] == 0 for r in ep["replies"])
    assert "Chinese" in __import__("moth.rl.sim", fromlist=["x"]).sim_language(spec) or not hasattr(C, "LANG_NAMES")


def test_eval_specs_have_language_slices():
    if not hasattr(C, "LANGS"):
        pytest.skip("core without chat languages")
    reg = tmp_reg()
    specs = registry.build_eval_specs(reg, per_card=5)
    langs = {s["lang"] for s in specs if s["scenario"] != "script"}
    assert langs == set(C.LANGS) and all(s["lang"] == "en" for s in specs if s["scenario"] == "script")
    tr = [registry.sample_spec(reg, random.Random(i)) for i in range(300)]
    assert {s["lang"] for s in tr} == set(C.LANGS)


# --- concurrency ----------------------------------------------------------------------------------------------------
def test_groups_run_concurrently_and_deterministically():
    import threading
    sim, judge = mock_parts()
    reg = tmp_reg()
    specs = [registry.sample_spec(reg, random.Random(i)) for i in range(3)]
    seq = env.run_groups(specs, 3, sim, judge, policy_factory=lambda: env.MockPolicy(), workers=1)
    par = env.run_groups(specs, 3, sim, judge, policy_factory=lambda: env.MockPolicy(), workers=6)
    strip = [[[t["user"] for t in ep["turns"]] for ep in g] for g in seq]
    assert strip == [[[t["user"] for t in ep["turns"]] for ep in g] for g in par]
    assert [[ep["reward"] for ep in g] for g in seq] == [[ep["reward"] for ep in g] for g in par]
    for g, spec in zip(par, specs):
        if spec["scenario"] != "agent":
            assert len({ep["turns"][0]["user"] for ep in g}) == 1
    live, peak, lock = [0], [0], threading.Lock()

    def slow(msgs):   # a batching engine behind generate() sees parallel requests
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        time.sleep(0.02)
        with lock:
            live[0] -= 1
        return "oh! tell me more?"
    env.rollout_group(spec_of("talk", seed=2), slow, 4, sim, judge, workers=4)
    assert peak[0] > 1
    with pytest.raises(ValueError):
        env.rollout_group(spec_of("talk"), env.MockPolicy(), 2, sim, judge, workers=2)   # stateful, shared


# --- pure chatbot: no home, no tools --------------------------------------------------------------------------------
def test_no_home_scenarios_and_legacy_home_specs_skipped():
    assert "home" not in registry.SCENARIOS and "home" not in registry.SCENARIO_MIX
    assert "invented" not in R.WEIGHTS
    reg = tmp_reg()
    specs = [registry.sample_spec(reg, random.Random(i)) for i in range(200)]
    assert not any(s["scenario"] == "home" or "home" in s for s in specs)
    registry.freeze("h", reg, reg.root, per_card=1)
    path = registry.eval_path("h", reg.root)
    d = llm.read_json(path)   # a legacy-style set with a tools-era home spec in it
    d["specs"].insert(0, {**d["specs"][0], "id": "old:home", "scenario": "home"})
    d["sha256"] = registry._digest(d["specs"])
    llm.write_json(path, d)
    sim, judge = mock_parts()
    s = evalrun.run_eval("h", lambda: env.MockPolicy(), sim, judge, tempfile.mkdtemp(), root=reg.root, limit=10,
                         replies=0)
    assert s["skipped_home_specs"] == 1 and "home" not in s["by_scenario"] and s["n_specs"] == 9


def test_eval_sim_temperature_defaults():
    import argparse

    from moth.rl.__main__ import components
    for cmd, want in (("eval", 0.0), ("run", 1.0)):
        a = argparse.Namespace(cmd=cmd, out=None, sim_url="http://127.0.0.1:9/v1", sim_model="m", judge_url="x",
                               judge_model="j", sim_temperature=None, judge_kind="generative")
        assert components(a, False)[0].temperature == want


# --- adapters -------------------------------------------------------------------------------------------------------
def test_verifiers_shaped_env_loop():
    sim, judge = mock_parts()
    core = adapters.MothMultiTurnEnv(sim, judge, tmp_reg())
    rows = core.dataset(4, seed=1)
    assert rows and all(r["prompt"][0]["role"] == "system" and r["prompt"][-1]["role"] == "user" for r in rows)
    for row in rows:
        state = core.setup_state({"info": row["info"]})
        pol = env.MockPolicy()
        pol.start(state["spec"])
        msgs = list(row["prompt"])
        for _ in range(30):
            msgs.append({"role": "assistant", "content": pol.generate(msgs)})
            if core.is_completed(msgs, state):
                break
            new, state = core.env_response(msgs, state)
            msgs += new
            if core.is_completed(msgs, state):
                break
        prompt, completion = msgs[:2], msgs[2:]
        vals = core.rubric.score(prompt, completion, "", state)
        ep = state["moth_episode"]
        assert vals["reward"] == pytest.approx(ep["reward"]) and ep["judge_ok"]
        assert len(ep["replies"]) == sum(m["role"] == "assistant" for m in msgs)
    try:
        import verifiers  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError):
            adapters.make_verifiers_env(core, 2)


def test_grpo_rollouts_and_turn_advantages():
    sim, judge = mock_parts()
    k = {"i": 0}

    def varied(msgs):
        k["i"] += 1
        return ["yay! tell me more?", "i'm just a little moth. lamps!", "how can i help you today?"][k["i"] % 3]
    groups = adapters.grpo_rollouts(varied, 3, 4, sim, judge, tmp_reg(), rng=random.Random(2))
    assert groups and all(len(g) == 4 for g in groups)
    for g in groups:
        for ep in g:
            ta = adapters.turn_advantages(ep)
            assert len(ta) == sum(m["role"] == "assistant" for m in ep["messages"])
    flat = adapters.grpo_rollouts(lambda m: "ok", 2, 3, sim, judge, tmp_reg(), rng=random.Random(2))
    assert flat == []   # identical rollouts: no signal, dropped


# --- eval runner ----------------------------------------------------------------------------------------------------
def eval_judge():
    return MockJudge(name="mock-eval-judge", role="eval_judge")


def test_eval_runner_judges_cis_passes_and_pairwise():
    reg = tmp_reg()
    registry.freeze("t", reg, reg.root, per_card=1)
    sim = MockSimulator()
    out_s, out_g = tempfile.mkdtemp(), tempfile.mkdtemp()
    s_bad = evalrun.run_eval("t", lambda: env.MockPolicy("selfy"), sim, eval_judge(), out_s, root=reg.root, limit=30,
                             replies=0)
    s = evalrun.run_eval("t", lambda: env.MockPolicy(), sim, eval_judge(), out_g, root=reg.root, limit=30,
                         replies=300, reward_judge=MockJudge(), ref=os.path.join(out_s, "results.json"))
    a = s["all"]
    assert s["passes"] > 1 and a["replies"] >= 300 and s["eval_judge"] == "mock-eval-judge"
    for k in ("on_gist", "self", "specific", "follows_card", "assistant", "repeat"):
        assert a[k]["lo"] <= a[k]["pct"] <= a[k]["hi"] and a[k]["n"] > 0, k
    assert a["reward"]["lo"] <= a["reward"]["mean"] <= a["reward"]["hi"]
    assert a["self"]["pct"] < s_bad["all"]["self"]["pct"]
    assert any("too noisy" in w for w in s["warnings"])   # < 3000 replies
    rj = s["reward_judge_vs_eval_judge"]
    assert rj["judge"] == "mock-judge" and rj["on"]["n"] > 0 and rj["on"]["kappa"] == 1.0   # same heuristic
    assert {p["sample"] for p in llm.read_json(os.path.join(out_g, "pairwise_ref.json"))} == {0}   # ref ran 1 pass
    recs = llm.read_json(os.path.join(out_g, "results.json"))
    assert {"cat", "name", "seed", "sample", "card_id", "turns"} <= set(recs[0])
    assert len({r["sample"] for r in recs}) == s["passes"]
    rep_md = open(os.path.join(out_g, "report.md"), encoding="utf-8").read()
    assert "headline judge: **mock-eval-judge**" in rep_md and "[" in rep_md and "lang en" in rep_md
    cj = s["cost_per_episode"]["eval_judge"]   # identical scripted chats across passes hit the cache
    assert cj["calls"] + cj["cached"] == pytest.approx(1.0, abs=0.02) and "judge" not in s["cost_per_episode"]
    with pytest.raises(ValueError):   # the headline judge must not be the reward judge
        evalrun.run_eval("t", lambda: env.MockPolicy(), sim, MockJudge(), out_g, root=reg.root, limit=3,
                         reward_judge=MockJudge(), replies=0)


def test_wilson_interval():
    w = evalrun.wilson(10, 1000)
    assert w["pct"] == 1.0 and 0.5 < w["lo"] < 1.0 < w["hi"] < 1.9
    assert evalrun.wilson(0, 0) is None and evalrun.wilson(0, 50)["lo"] == 0.0


def test_gold_agreement():
    from moth.rl import judge as J
    gold = os.path.join(tempfile.mkdtemp(), "gold.jsonl")
    rows = [{"id": "g1", "card": "moth", "labeler": "human",
             "turns": [{"user": "my cat knocked my tea over", "say": "oh no, the tea! was the cat sorry?"},
                       {"user": "i got the job!!!", "say": "aw that's rough"}],
             "labels": [{"on": True, "self": False, "specific": True}, {"on": False, "self": False, "specific": False}]},
            {"id": "g2", "card": {"id": "x", "name": "bo", "kind": "a small cat", "refs": ["cat", "paw"]},
             "labeler": "claude", "turns": [{"user": "how was your day", "say": "i'm just a little cat. paws!"}],
             "labels": [{"on": False, "self": True, "specific": False}]}]
    llm.write_text(gold, "\n".join(json.dumps(r) for r in rows) + "\n")
    g = J.load_gold(gold)
    a = J.agreement(MockJudge(), g)
    assert a["chats"] == 2 and a["on"]["n"] == 3 and a["self"]["acc"] == 1.0 and a["specific"]["n"] == 3
    assert J.kappa([True, False, True, False], [True, False, False, False]) == 0.5
    # bake-off labelled_replies.jsonl lines: only the ones with a hand label count
    llm.write_text(gold, json.dumps({"card": {"id": "moth"}, "lang": "es", "friend": "hice pasta", "text": "¿con queso?",
                                     "labels": {"qwen": {"on": True}}, "human": {"on": True, "self": False}}) + "\n"
                   + json.dumps({"card": {}, "friend": "x", "text": "y", "labels": {}}) + "\n")
    g2 = J.load_gold(gold)
    assert len(g2) == 1 and g2[0]["labeler"] == "human" and g2[0]["card"]["lang"] == "es"


# --- eval sizing: cluster stats, tiers, gates, monitor, gold sheet ---------------------------------------------------
def test_cluster_stats_and_mde():
    from moth.rl import stats
    flat = stats.cluster_rate([1, 0] * 500, [1] * 1000)   # 1000 independent replies
    clumped = stats.cluster_rate([5, 0] * 100, [5] * 200)   # same rate, all-or-nothing chats
    assert flat["pct"] == clumped["pct"] == 50.0
    assert flat["deff"] == pytest.approx(1.0, abs=0.01) and clumped["deff"] > 4 and clumped["hi"] - clumped["lo"] > 2 * (flat["hi"] - flat["lo"])
    assert clumped["mde"] > flat["mde"] and clumped["n_eff"] < 250
    lo, hi = stats.cluster_bootstrap([5, 0] * 100, [5] * 200, B=300)
    assert lo < 50 < hi and hi - lo > 10
    assert stats.plan(0.85, 1.5, 10000) <= 3.0 and stats.plan(0.85, 1.5, 2000) <= 5.0   # the full-tier targets
    assert stats.cluster_rate([0] * 50, [3] * 50)["mde"] > 0   # a rate at 0 still has an MDE (1% floor)
    w = stats.win_rate(600, 100, 300)
    assert w["pct"] == 65.0 and w["lo"] > 50 and w["n"] == 1000


def test_eval_tiers_and_release_gates():
    reg = tmp_reg()
    for i in range(6):   # a few more held-out cards so languages have something to slice
        reg.add_card({"name": f"q{'abcdef'[i]}x", "kind": "a moon rock"}, heldout=True)
    registry.freeze("t5", reg, reg.root)
    doc = registry.load_eval("t5", reg.root)
    quick = evalrun.tier_specs(doc["specs"], "quick")
    qids = {s["id"] for s in quick}
    assert all(s["id"] in qids for s in doc["specs"] if s["lang"] == "en" or s["card_id"] == "moth")
    other = [s for s in doc["specs"] if s["lang"] != "en" and s["card_id"] != "moth"]
    assert 0 < sum(s["id"] in qids for s in other) < len(other) / 2   # en-heavy: a quarter of the rest
    sim = MockSimulator()
    out_ref = tempfile.mkdtemp()
    evalrun.run_eval("t5", lambda: env.MockPolicy("bland"), sim, eval_judge(), out_ref, root=reg.root, tier="full",
                     replies=400, per_lang=60)
    s = evalrun.run_eval("t5", lambda: env.MockPolicy(), sim, eval_judge(), tempfile.mkdtemp(), root=reg.root,
                         tier="full", replies=400, per_lang=60, refs={"final2": os.path.join(out_ref, "results.json")})
    langs = {lg: b["replies"] for lg, b in s["by_lang"].items()}
    assert set(langs) == {"en", "zh", "es", "fr", "hi"} and min(langs.values()) >= 60
    assert s["moth"]["episodes"] > 0 and s["moth_agent"]["episodes"] > 0
    names = {g["name"]: g for g in s["gates"]}
    assert set(names) == {"MDE on_gist overall", "MDE on_gist per language", "pairwise vs final2"}
    assert not names["pairwise vs final2"]["pass"] and "need 1000" in names["pairwise vs final2"]["why"]   # too few
    assert s["pairwise"]["final2"]["win_rate"]["pct"] > 50   # the specific policy beats the bland one
    rep_md = open(os.path.join(out_ref, "report.md"), encoding="utf-8").read()
    assert "MDE (points" in rep_md and "moth card, agent reactions" in rep_md


def test_reward_hacking_monitor():
    from moth.rl.monitor import HackMonitor

    def summ(rew, og, vf=None):
        s = {"reward_judge_vs_eval_judge": {"reward": {"mean": rew}}, "all": {"on_gist": {"pct": og, "mde": 2.0},
                                                                                "reward": {"mean": rew}}}
        if vf is not None:
            s["pairwise"] = {"final2": {"win_rate": {"pct": vf, "mde": 4.0}}}
        return s
    path = os.path.join(tempfile.mkdtemp(), "mon.jsonl")
    m = HackMonitor(path, rise=0.15, window=3)
    assert not m.update(0, summ(0.10, 80.0, 50.0))["flag"]
    assert not m.update(1, summ(0.20, 83.0, 55.0))["flag"]   # both rise: fine
    r = m.update(2, summ(0.40, 80.5, 49.0))   # reward +0.30 over the window, on_gist flat, vs final2 down
    assert r["flag"] and len(r["why"]) == 2
    assert len(HackMonitor(path).hist) == 3   # append-only log, reloads


def test_gold_sheet_stratified_and_per_stratum_agreement():
    from moth.rl import gold as G
    from moth.rl import judge as J
    reg = tmp_reg()
    registry.freeze("g", reg, reg.root)
    out = tempfile.mkdtemp()
    k = {"i": 0}

    def mixed(msgs):
        k["i"] += 1
        return ["i'm just a little moth. lamps!", "oh no, the tea! was the cat sorry?", "haha nice"][k["i"] % 3]
    evalrun.run_eval("g", lambda: env.FnPolicy(mixed), MockSimulator(), eval_judge(), out, root=reg.root, replies=1500)
    items = G.load_items([os.path.join(out, "episodes.json")])
    rows = G.build_sheet(items, n=600, seed=1)
    assert len(rows) == 600 and sum(r["stratum"] == "random" for r in rows) == 300
    strat = [r for r in rows if r["stratum"] == "stratified"]
    by_lang = collections.Counter(r["lang"] for r in strat)
    assert all(by_lang[lg] >= 60 for lg in ("en", "zh", "es", "fr", "hi"))
    assert sum(bool(r["reasons"]) for r in strat) > sum(bool(r["reasons"]) for r in rows if r["stratum"] == "random")
    path = os.path.join(out, "sheet.jsonl")
    G.write_sheet(rows, path)
    assert os.path.exists(path[:-6] + ".csv")
    filled = []   # a "labeller" fills the human column for 40 random + 40 stratified rows
    for r in [x for x in rows if x["stratum"] == "random"][:40] + strat[:40]:
        r = dict(r)
        r["human"] = {"on": "lamps" not in r["text"], "self": "moth" in r["text"], "specific": "tea" in r["text"]}
        filled.append(r)
    llm.write_text(path, "\n".join(json.dumps(x) for x in filled + rows[100:110]) + "\n")   # unfilled rows skipped
    g = J.load_gold(path)
    assert len(g) == 80 and {x["stratum"] for x in g} == {"random", "stratified"}
    a = G.agreement_by_stratum(MockJudge(), g)
    assert set(a) == {"random", "stratified"}
    for st in a.values():
        assert st["on"]["n"] == 40 and st["on"]["acc"]["lo"] <= st["on"]["acc"]["pct"] <= st["on"]["acc"]["hi"]


def test_heldout_growth_pins_and_v5_breadth():
    reg = tmp_reg()
    c = reg.add_line("you are a velvet armchair named sofonisba", heldout=True)
    assert c["split"] == "heldout"
    assert registry.Registry(reg.root).card(c["id"])["split"] == "heldout"   # the pin survives a reload
    with pytest.raises(ValueError):
        reg.add_line("you are a teapot named miso", heldout=True)   # a training card's name
    p = reg.add_phrasing("hey {name}, you're {kind} today.", heldout=True)
    assert p["split"] == "heldout"
    later = reg.add_line("you are a bus named sofonisba")   # a held-out name: pinned held-out, never trains
    assert later["split"] == "heldout"
    ask = lambda prompt: '["you are a grumpy lamp post named lucius.", "You are Zelda, a clockwork owl."]'  # noqa: E731
    got = registry.add_teacher_batch(reg, ask, 2, heldout=True)
    assert [x["split"] for x in got] == ["heldout", "heldout"]
    doc = registry.load_eval("v5")   # the committed release-size set
    base = collections.defaultdict(set)
    for sp in doc["specs"]:
        base[sp["card_id"].split("~")[0]].add(sp.get("lang"))
    held = [b for b in base if b not in ("moth", "blank")]
    assert len(held) >= 100 and all(len(base[b]) == 5 for b in held)
    assert len({sp["card"].get("phrasing_id") for sp in doc["specs"] if sp["card"].get("phrasing_id")}) >= 20
    moth_chats = [sp for sp in doc["specs"] if sp["card_id"] == "moth" and sp["scenario"] != "script"]
    assert len(moth_chats) >= 40 and sum(sp["scenario"] == "agent" for sp in moth_chats) >= 20


def test_moth_policy_legacy_prompt_for_final2():
    class FakeMoth:
        def __init__(self):
            self.system = None

        def reset(self):
            pass
    fm = FakeMoth()
    spec = spec_of("talk", seed=1)
    env.MothPolicy(fm, legacy_prompt=True).start(spec)
    assert fm.system == C.student_system_v1(spec["card"]) and "Tools:" in fm.system


# --- endpoints, cost# --- endpoints, cost ------------------------------------------------------------------------------------------------
def _free_port():
    with socket.socket() as so:
        so.bind(("127.0.0.1", 0))
        return so.getsockname()[1]


def test_llm_components_against_fake_server():
    port = _free_port()
    srv = subprocess.Popen([sys.executable, os.path.join(ROOT, "tests", "fake_oai.py"), str(port)])
    try:
        url = f"http://127.0.0.1:{port}/v1"
        for _ in range(100):
            try:
                assert llm.OAIClient(url).model
                break
            except OSError:
                time.sleep(0.05)
        cl = llm.OAIClient(url, retries=0)
        assert cl.model == "fake"
        sim = LLMSimulator(cl)
        judge = LLMJudge(llm.OAIClient(url, retries=0))
        pol = env.OpenAIPolicy(llm.OAIClient(url, retries=0))
        spec = spec_of("talk", seed=9)
        spec["turns"] = 2
        m = llm.CostMeter()
        ep = env.run_episode(spec, pol, sim, judge, m)
        assert ep["turns"][0]["user"] == "what's in your home?"
        assert ep["labels"] is None   # the fake server can't label: masked, not scored as off-gist
        r = m.roles
        assert r["judge"]["calls"] == 3 and r["policy"]["calls"] == len(ep["turns"]) and r["sim"]["est"]
        # the pilot CLI: no --out (nothing written), 2 worker threads; every policy call is metered once
        from moth.rl.__main__ import main
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            eps, total = main(["run", "--sim-url", url, "--judge-url", url, "--policy-url", url, "--episodes", "4",
                               "--workers", "2", "--agent-pool", os.path.join(tempfile.mkdtemp(), "none.json")])
        assert total.roles["policy"]["calls"] == sum(len(e["turns"]) for e in eps) > 0
        assert all(e["labels"] is None for e in eps)
    finally:
        srv.kill()
        srv.wait()


def test_cost_meter_and_box_hours():
    m = llm.CostMeter()
    for _ in range(2):
        e = llm.CostMeter()
        e.add("sim", 1000, 20)
        e.add("judge", 600, 100)
        e.add("judge", 0, 0, cached=True)
        e.episodes = 1
        m.merge(e)
    per = m.per_episode()
    assert per["sim"] == {"calls": 1.0, "cached": 0.0, "prompt_tokens": 1000.0, "completion_tokens": 20.0,
                          "seconds": 0.0, "est": False}
    assert per["judge"]["cached"] == 1.0
    bh = llm.box_hours(per, 3600, {"sim": {"prefill": 1000, "decode": 20}}, price=2.0)
    assert bh["box_hours"] == {"sim": 2.0} and bh["dollars"] == 4.0 and bh["skipped_roles"] == ["judge"]
    f = os.path.join(tempfile.mkdtemp(), "cr.sh")
    llm.write_text(f, "x=1\ndeclare -A CR_PRICE=([a.1]=0.65 [b-2.1]=1.80\n   [c.1]=0.39)\nmore\n")
    assert llm.cr_prices(f) == {"a.1": 0.65, "b-2.1": 1.8, "c.1": 0.39}
    assert llm.cr_prices(os.path.join(tempfile.mkdtemp(), "none.sh")) == {}


def test_cli_mock_run():
    from moth.rl.__main__ import main
    out = io.StringIO()
    d = tempfile.mkdtemp()
    with contextlib.redirect_stdout(out):
        main(["mock", "--episodes", "2", "--out", d])
        main(["cost", "--meter", os.path.join(d, "meter.json"), "--episodes", "1000", "--tps", "sim=10000:1000",
              "--tps", "judge=10000:1000", "--price", "1.0"])
    o = out.getvalue()
    assert "per episode" in o and "mean reward" in o and '"total_box_hours"' in o


# --- tiny-agent GRPO (real TinyAgentLM, random init, CPU) ------------------------------------------------------------
TA_REPO = os.environ.get("TA_REPO", "/mnt/ssd/dev workspace/tiny-agent")
TA_PY = os.environ.get("TA_PY", "/mnt/ssd/b70-venv/bin/python")


def test_tiny_agent_grpo_group():
    """tests/rl_tiny_agent_check.py in tiny-agent's Python (needs torch + the tiny-agent repo; skipped without them).
    The GPU is hidden (ONEAPI_DEVICE_SELECTOR=opencl:cpu), so this never touches the desktop GPU."""
    if not (os.path.isdir(os.path.join(TA_REPO, "tiny_agent")) and os.path.exists(TA_PY)
            and os.path.exists(os.path.join(ROOT, "data", "tiny-agent-112m", "tokenizer.json"))):
        pytest.skip("tiny-agent repo, its Python or the tokenizer is not here")
    e = {**os.environ, "ONEAPI_DEVICE_SELECTOR": "opencl:cpu", "PYTHONPATH": ""}
    r = subprocess.run([TA_PY, os.path.join(ROOT, "tests", "rl_tiny_agent_check.py"), TA_REPO], env=e,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0 and "\nOK " in "\n" + r.stdout, r.stdout[-2000:] + r.stderr[-3000:]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
