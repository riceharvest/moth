"""Gist-first spec, identity cards, the per-reply label pass and the fan-out merge (CPU only, no server).

  python -m pytest tests/test_gist_spec.py -q
"""
import collections
import glob
import json
import os
import random
import re
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from moth import agent, cards, gist, persona, probes  # noqa: E402
from moth.home import load_pool, sample_home  # noqa: E402

OLD_STUDENT = (
    "you are moth. you live on a graphics card. your home folder is ~. talk with your friend. use tools to look at or "
    "change your files. messages from the big agent your friend works with show up as [the agent said].\nTools:\n"
    "bash(command): Run a shell command in your home ~ (ls, cat, wc, grep...).\n"
    "read(path, offset?, limit?): Read a file with line numbers.\n"
    "write(path, content): Create or overwrite a file.")


def few_shot_replies():
    """The buddy's lines in the teacher sheet: "you:" examples and the right side of agent "->" examples."""
    lines = re.findall(r"^you: (.+)$", persona.TEACHER_CORE, re.M)
    pairs = re.findall(r'^\s*"([^"]+)" -> "([^"]+)"', agent.AGENT_RULES, re.M)
    return lines, pairs


# --- cards ----------------------------------------------------------------------------------------------------------
def test_moth_card_keeps_the_old_student_prompt():
    assert cards.student_system(cards.MOTH_CARD) == OLD_STUDENT == persona.SYSTEM_STUDENT
    assert cards.student_system(None) == OLD_STUDENT


def test_student_prompt_other_cards():
    c = {"id": "x", "name": "pip", "kind": "a houseplant", "home": "on a kitchen windowsill", "traits": ["you love puns"]}
    s = cards.student_system(c)
    assert s.startswith("you are pip, a houseplant. you live on a kitchen windowsill. you love puns. your home folder is ~.")
    assert cards.student_system(cards.BLANK_CARD).startswith("your home folder is ~.")
    assert "moth" not in cards.student_system(c)


def test_pool_deterministic_diverse_and_heldout_disjoint():
    a, b = cards.build_pool(), cards.build_pool()
    assert a == b and len(a) >= 500
    kinds = {c["kind"] for c in a}
    assert len(kinds) >= 30 and None in kinds   # plain humans too
    held = cards.heldout_cards()
    assert len(held) >= 8
    held_kinds = {c["kind"] for c in held if c["kind"]}
    held_names = {c["name"] for c in held}
    assert not held_kinds & kinds
    assert not held_names & {c["name"] for c in a}


def test_sample_card_shares():
    rng = random.Random(0)
    ids = collections.Counter()
    for _ in range(4000):
        c = cards.sample_card(rng)
        ids["moth" if c["id"] == "moth" else "blank" if c["id"] == "blank" else "pool"] += 1
    assert ids["moth"] / 4000 <= 0.10 and 0.07 <= ids["blank"] / 4000 <= 0.13


def test_load_card_file_and_specs():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "pip.json")
        json.dump({"name": "pip", "kind": "a houseplant"}, open(p, "w"))
        c = cards.load_card(p)
    assert c["id"] == "pip" and "houseplant" in c["refs"] and c["folders"] == cards.GENERIC_FOLDERS
    assert cards.load_card("moth")["id"] == "moth" and cards.load_card("heldout:0")["id"] == "h000"
    with pytest.raises(SystemExit):
        cards.load_card("nope-not-a-card")


def test_homes():
    pool = load_pool(None)
    assert sample_home(random.Random(5), pool) == sample_home(random.Random(5), pool, cards.MOTH_CARD)
    h = sample_home(random.Random(5), pool, cards.build_pool()[1])
    assert {p.split("/")[0] for p in h} == set(cards.GENERIC_FOLDERS)
    assert not any(w in " ".join(h.values()) for w in ("lamp", "graphics card", "moth", "fan"))
    own = {"c0001": {"diary/rainy.txt": "it rained on my leaves."}}
    h2 = sample_home(random.Random(5), pool, cards.build_pool()[1], own)
    assert h2["diary/rainy.txt"].startswith("it rained")


# --- the teacher sheet ----------------------------------------------------------------------------------------------
def test_core_is_identity_free_and_lenses_gone():
    assert not hasattr(agent, "AGENT_LENSES")
    core = persona.TEACHER_CORE.lower()
    assert "moth" not in core and "graphics card" not in core and "lens" not in core
    assert "pick one plain" not in core and "vary your imagery" not in core
    assert "{lens}" not in agent.AGENT_REMIND


def test_few_shots_have_at_most_one_identity_touch():
    lines, pairs = few_shot_replies()
    assert len(lines) >= 8 and len(pairs) >= 5
    refs = [x for x in lines + [r for _, r in pairs] if gist.identity_ref(x, cards.MOTH_CARD)]
    assert len(refs) <= 1, refs


def test_few_shot_agent_reactions_pass_the_filters():
    """The spec's own examples must survive generation (AGENT_BAD), build_sft (reply_problem) and the eval metrics."""
    pytest.importorskip("torch")
    pytest.importorskip("openai")
    from moth import build_sft, gen_dialogs, metrics
    _, pairs = few_shot_replies()
    for msg, r in pairs:
        ctx = {"is_agent": True, "user": msg.lower(), "after_agent": True, "bug_said": "bug" in msg.lower(),
               "tool_text": "", "topic": "", "n_replies": 1, "identity": 0, "motif": collections.Counter(),
               "motif_replies": 0, "card": cards.MOTH_CARD, "id_refs": 0}
        assert build_sft.reply_problem(r, ctx) is None, r
        assert not gen_dialogs.AGENT_BAD.search(r), r
        assert not metrics.ADVICE.search(r), r
        x = {"last_agent": msg, "ws": metrics.words(r)}
        assert not metrics.summary_copy(x), r


def test_agent_stale_no_longer_demands_a_word_of_the_message():
    pytest.importorskip("openai")
    from moth import gen_dialogs
    msgs = [{"role": "user", "content": agent.agent_turn("The staging box is still booting; I'll check again soon.")}]
    assert not gen_dialogs.agent_stale("still starting? i hope it wakes up soon.", msgs)
    assert gen_dialogs.agent_stale("still waking up? me too, mornings are hard.", msgs)   # the few-shot line: a copy


def test_reminders_gate_identity_talk():
    pytest.importorskip("openai")
    from moth import gen_dialogs as G
    c = cards.build_pool()[3]
    assert G.teacher_view("i burned my toast", card=c).endswith(G.TALK_REMIND)
    assert G.TALK_REMIND_ID in G.teacher_view("who are you?", card=c)
    v = G.first_teacher_view(3)
    assert v["system"].startswith(persona.TEACHER_CORE) and "WHO YOU ARE IN THIS CHAT" in v["system"]


def test_shared_prefix_fills_one_cache_block():
    """tools + TEACHER_CORE just over one 1568-token block (Qwen3.8's hybrid prefix cache), when the tokenizer is here."""
    toks = sorted(glob.glob("/mnt/ssd/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/*"))
    if not toks:
        pytest.skip("no Qwen3.8-27B tokenizer on disk")
    pytest.importorskip("transformers")
    pytest.importorskip("openai")
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    import render_prompts
    n, cached = render_prompts.prefix_tokens(toks[0])
    assert 1568 < n < 1568 + 250, n


# --- labels ---------------------------------------------------------------------------------------------------------
CHAT = [{"role": "user", "content": "i burned my toast"},
        {"role": "assistant", "content": "oh no! was it very black?", "tool_calls": []},
        {"role": "user", "content": "check your diary"},
        {"role": "assistant", "content": "", "tool_calls": [{"name": "bash", "arguments": {"command": "ls diary"}}]},
        {"role": "tool", "results": ["tuesday.txt"]},
        {"role": "assistant", "content": "just tuesday in there.", "tool_calls": []},
        {"role": "user", "content": agent.agent_turn("The box is still booting.")},
        {"role": "assistant", "content": "a box is a tiny nap spot.", "tool_calls": []}]


def test_label_prompt_numbers_only_replies():
    text, n = gist.chat_for_label_msgs(CHAT)
    assert n == 3 and "[R3] a box is a tiny nap spot." in text and "(it used bash: ls diary)" in text
    p = gist.label_prompt(text, n, cards.MOTH_CARD)
    assert "exactly 3 entries" in p and "the box is still booting" in p.lower()


def test_parse_labels():
    good = 'ok {"labels": [{"n": 1, "why": "asks about the toast", "on": true}, {"n": 2, "why": "x", "on": "true"},' \
           ' {"n": 3, "why": "word grab", "on": false}]}'
    labs = gist.parse_labels(good, 3)
    assert [x["on"] for x in labs] == [True, True, False]
    with pytest.raises(ValueError):
        gist.parse_labels(good, 2)
    with pytest.raises(ValueError):
        gist.parse_labels('{"labels": [{"n": 1, "on": "maybe"}]}', 1)


def test_build_sft_cuts_at_first_off_reply():
    pytest.importorskip("torch")
    from moth import build_sft
    d = {"seed": 1, "type": "talk", "home": {"diary/tuesday.txt": "x"}, "card": cards.MOTH_CARD,
         "messages": [m for m in CHAT[:6]] + [{"role": "user", "content": "lol"},
                                              {"role": "assistant", "content": "you laughed!", "tool_calls": []}]}
    on = [{"on": True, "why": ""}] * 3
    why = collections.Counter()
    assert len(build_sft.keep(d, why, on)) == len(d["messages"])
    out = build_sft.keep(d, why, on[:2] + [{"on": False, "why": "x"}])
    assert out[-1]["content"] == "just tuesday in there." and why["off_gist:cut"] == 1
    assert len(build_sft.keep(d, why, None)) == len(d["messages"])        # unlabelled: no cut
    assert len(build_sft.keep(d, why, on[:1])) == len(d["messages"]) and why["labels_mismatch"] == 1


def test_build_sft_caps_unprompted_identity_talk():
    pytest.importorskip("torch")
    from moth import build_sft
    msgs = []
    for u, r in [("hi", "hi! i was sitting by my lamp."), ("how are you", "good. tired."),
                 ("what did you eat", "soup. and some dust off the lamp.")]:
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": r, "tool_calls": []}]
    why = collections.Counter()
    out = build_sft.keep({"seed": 2, "type": "talk", "home": {}, "card": cards.MOTH_CARD, "messages": msgs}, why)
    assert len(out) == 4 and why["identity_heavy:cut"] == 1


def test_identity_ref():
    m = cards.MOTH_CARD
    assert gist.identity_ref("i sat by the lamp all day", m, "how was your day")
    assert not gist.identity_ref("the lamp? i like it", m, "do you like my lamp")     # the friend brought it up
    assert not gist.identity_ref("i'm moth, i live on a graphics card", m, "who are you?")
    assert gist.identity_ref("i'm just a little guy, i don't know", m, "what's 2+2")
    plant = cards.load_card({"id": "p", "name": "fern", "kind": "a houseplant", "refs": ["leaf", "leaves", "pot", "soil"]})
    assert gist.identity_ref("my leaves are droopy today", plant, "how are you")
    assert not gist.identity_ref("my lamp is on", plant, "hi")   # moth words don't count for a plant
    assert gist.identity_ref_rate([("hi", "hi!"), ("yo", "my leaves hurt")], plant) == 0.5


def test_follows_card_and_probes_for_card():
    h = cards.heldout_cards()[0]
    assert gist.follows_card(f"i'm {h['name']}! hi.", h)
    assert not gist.follows_card("i'm a moth on a graphics card", h)
    assert gist.claims_other_identity("my name is moth", h)
    ps = probes.for_card(probes.all_probes(), h)
    names = {p[1] for p in ps}
    assert not names & probes.MOTH_ONLY and {"card_who", "card_name", "greeting"} <= names
    greet = [p for p in ps if p[1] == "greeting"][0]
    assert greet[2][0] == f"hi {h['name']}"
    assert probes.for_card(probes.all_probes(), cards.MOTH_CARD) == probes.all_probes()


# --- fan-out merge --------------------------------------------------------------------------------------------------
def test_merge_gen():
    from moth import merge_gen
    with tempfile.TemporaryDirectory() as d:
        for box, seeds in (("a", [10, 11]), ("b", [11, 20])):
            os.makedirs(os.path.join(d, box))
            with open(os.path.join(d, box, "dialogs.jsonl"), "w") as f:
                for s in seeds:
                    f.write(json.dumps({"seed": s, "type": "talk", "card": {"id": "moth"}, "messages": []}) + "\n")
            with open(os.path.join(d, box, "labels.jsonl"), "w") as f:
                for s in seeds:
                    f.write(json.dumps({"seed": s, "n": 2, "labels": [{"on": True, "why": ""}, {"on": s != 20, "why": ""}]}) + "\n")
        ds, labs, boxes, warn = merge_gen.merge([os.path.join(d, "a"), os.path.join(d, "b")])
    assert sorted(x["seed"] for x in ds) == [10, 11, 20] and sorted(labs) == [10, 11, 20]
    assert any("overlap" in w for w in warn) and any("already merged" in w for w in warn)
    assert boxes[1]["on_gist"] == 0.75
