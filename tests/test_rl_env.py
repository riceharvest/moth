"""The multi-turn RL environment (moth.rl) in mock mode: registry + split + frozen evals, simulator, judge, rewards,
episodes, groups, adapters, eval runner, cost meter. CPU only, no endpoint (one test talks to tests/fake_oai.py).

  python -m pytest tests/test_rl_env.py -q      (or: python tests/test_rl_env.py)
"""
import contextlib
import io
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
    first = [reg.add_line(f"you are a frog named zz{i}") for i in range(40)]
    assert all(c["split"] == registry.card_split({k: v for k, v in c.items() if k != "split"}) for c in first)
    before = {c["id"]: c["split"] for c in first}
    for i in range(40):
        reg.add_line(f"you're a teapot called qq{i}")
    reg2 = registry.Registry(reg.root)   # a fresh load of the same files
    assert {c["id"]: reg2.card(c["id"])["split"] for c in first} == before
    assert reg.add_line("you are a frog named zz0") is None   # same line = same id: not added twice
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
    assert C.student_system(c).startswith("you are a cat named bo. your home folder is ~.")
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


def test_committed_eval_v1_and_training_agent_messages():
    doc = registry.load_eval("v1")
    assert doc["n"] == len(doc["specs"]) and {s["scenario"] for s in doc["specs"]} == {
        "talk", "agent", "home", "recall", "script"}
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
    for scen in ("talk", "home", "recall", "agent"):
        spec = spec_of(scen, seed=11)
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


def test_policy_extras_tools_and_canonical_messages():
    class FakeMoth:   # moth.serve.Moth's surface: system, reset(), chat() -> events (with a tool call)
        def __init__(self):
            self.system, self.a, self.users = None, None, []

        def reset(self):
            self.users = []

        def chat(self, u, talk_only=False):
            self.users.append(u)
            if "home" in u and not talk_only:
                return [{"tool": "bash", "args": {"command": "ls"}, "result": "diary\nfriends"},
                        {"say": "i have a diary and some friends."}]
            return [{"say": "oh! tell me more?"}]

    fm = FakeMoth()
    p = env.MothPolicy(fm)
    spec = spec_of("home", seed=3)
    spec["script"], spec["turns"] = ["hi", "what's in your home folder?", "nice"], 3
    ep = env.run_episode(spec, p, *mock_parts())
    assert fm.system == spec["system"] and fm.users == spec["script"]
    assert len(ep["replies"]) == 3 == len(ep["labels"])
    roles = [m["role"] for m in ep["messages"]]
    assert roles == ["user", "assistant", "user", "assistant", "tool", "assistant", "user", "assistant"]
    assert ep["messages"][3]["tool_calls"][0]["name"] == "bash"
    assert [r["assistant_index"] for r in ep["replies"]] == [0, 2, 3]
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
def test_eval_runner_and_pairwise():
    reg = tmp_reg()
    registry.freeze("t", reg, reg.root, per_card=1)
    sim, judge = mock_parts()
    out_s, out_g = tempfile.mkdtemp(), tempfile.mkdtemp()
    s_bad = evalrun.run_eval("t", lambda: env.MockPolicy("selfy"), sim, judge, out_s, root=reg.root, limit=30)
    s = evalrun.run_eval("t", lambda: env.MockPolicy(), sim, judge, out_g, root=reg.root, limit=30,
                         ref=os.path.join(out_s, "results.json"))
    a = s["all"]
    for k in ("on_gist", "self", "follows_card", "assistant", "repeat", "reward"):
        assert a[k] is not None, k
    assert a["self"] < s_bad["all"]["self"] and a["follows_card"] >= s_bad["all"]["follows_card"]
    assert s["pairwise"]["pairs"] == 30 and s["pairwise"]["A_wins"] > s["pairwise"]["B_wins"]
    recs = llm.read_json(os.path.join(out_g, "results.json"))
    assert {"cat", "name", "seed", "sample", "card_id", "turns"} <= set(recs[0])
    assert "on_gist" in open(os.path.join(out_g, "report.md"), encoding="utf-8").read()
    assert s["cost_per_episode"]["judge"]["calls"] == 1


# --- endpoints, cost ------------------------------------------------------------------------------------------------
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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
