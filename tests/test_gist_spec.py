"""Gist-first pure-chat spec, identity cards, the per-reply label pass and the fan-out merge (CPU only, no server).

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
def test_pure_chat_student_prompt_and_legacy_prompt():
    s = cards.student_system(cards.MOTH_CARD)
    assert s == persona.SYSTEM_STUDENT == cards.student_system(None)
    assert s == "you are moth, a small moth who lives on a graphics card. " + cards.STUDENT_TAIL
    for bad in ("tool", "home folder", "~", "bash", "file"):
        assert bad not in s.lower(), bad
    # the released round-3 weights keep their tools/home prompt through serve.py
    assert cards.student_system_v1(cards.MOTH_CARD) == OLD_STUDENT == cards.student_system_v1(None)
    serve_src = open(os.path.join(ROOT, "moth", "serve.py")).read()
    assert "self.system = student_system_v1(self.identity)" in serve_src


def test_student_prompt_other_cards():
    c = {"id": "x", "name": "pip", "kind": "a houseplant", "traits": ["you love puns"]}
    s = cards.student_system(c)
    assert s == "you are pip, a houseplant. you love puns. " + cards.STUDENT_TAIL
    assert cards.student_system(cards.BLANK_CARD) == cards.STUDENT_TAIL
    assert "moth" not in cards.student_system(c)
    moth_lore = {**cards.MOTH_CARD, "traits": ["you like lamps and blinking lights"]}
    assert "you like lamps and blinking lights." in cards.student_system(moth_lore)


def test_cards_have_no_home_or_folders():
    for c in cards.build_pool() + cards.heldout_cards() + cards.heldout_cards(free=True) + [cards.MOTH_CARD,
                                                                                            cards.BLANK_CARD]:
        assert "home" not in c and "folders" not in c, c
    assert not any("{home}" in t for t in cards.PHRASINGS + cards.HELDOUT_PHRASINGS)


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


def test_free_form_phrasings_and_heldout_phrasings():
    pool = cards.build_pool()
    free = [c for c in pool if c.get("lines")]
    assert 0.3 <= len(free) / len(pool) <= 0.5
    assert len({c["lines"][0].split()[0] for c in free}) >= 8     # many shapes, not one template
    for c in free:   # the student prompt is the free line + the chat tail; fields it doesn't say are cleared
        assert cards.student_system(c).startswith(c["lines"][0].rstrip(".!")) and \
            cards.student_system(c).endswith(cards.STUDENT_TAIL)
        if c["kind"]:
            assert c["kind"].split()[-1] in c["lines"][0].lower()
    held = cards.heldout_cards(free=True)
    starts = {t.split("{")[0] for t in cards.HELDOUT_PHRASINGS}
    assert all(any(c["lines"][0].startswith(st) or c["lines"][0].startswith(st.capitalize()) for st in starts if st)
               or c["lines"][0].split()[0][0].isupper() for c in held)
    train_lines = {c["lines"][0] for c in free}
    assert not train_lines & {c["lines"][0] for c in held}
    assert not set(cards.HELDOUT_PHRASINGS) & set(cards.PHRASINGS)


def test_text_card():
    c = cards.text_card("you are a cat named bo")
    assert c["name"] == "bo" and "cat" in c["refs"]
    assert cards.student_system(c) == "you are a cat named bo. " + cards.STUDENT_TAIL
    assert cards.load_card("text:You are Pip, a hamster.")["name"] == "pip"
    assert gist.follows_card("i'm bo! a cat.", c) and not gist.follows_card("i'm a moth", c)
    assert not gist.claims_other_identity("i'm a cat, silly", c)


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
    assert c["id"] == "pip" and "houseplant" in c["refs"] and "folders" not in c
    assert cards.load_card("moth")["id"] == "moth" and cards.load_card("heldout:0")["id"] == "h000"
    with pytest.raises(SystemExit):
        cards.load_card("nope-not-a-card")


def test_legacy_homes_for_the_released_model():
    """serve.py's home for the released tools model: the moth card still gets moth's six folders."""
    pool = load_pool(None)
    assert sample_home(random.Random(5), pool) == sample_home(random.Random(5), pool, cards.MOTH_CARD)
    h = sample_home(random.Random(5), pool, cards.build_pool()[1])
    assert {p.split("/")[0] for p in h} == set(cards.GENERIC_FOLDERS)


# --- the teacher sheet ----------------------------------------------------------------------------------------------
def test_core_is_identity_free_tool_free_and_lenses_gone():
    assert not hasattr(agent, "AGENT_LENSES")
    core = persona.TEACHER_CORE.lower()
    for bad in ("home folder", "tool", "`ls", "cat friends", "write a short lowercase file"):
        assert bad not in core, bad
    assert not any(w in " ".join(persona.SCENARIOS + persona.USER_PERSONAS) for w in ("home folder", "folder", "diary"))
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
               "motif_replies": 0, "card": cards.MOTH_CARD, "use_matcher": True}
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
    """TEACHER_CORE just over one 1568-token block (Qwen3.8's hybrid prefix cache), when the tokenizer is here."""
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
        {"role": "user", "content": "what are you doing"},
        {"role": "assistant", "content": "just chatting with you. and thinking about toast.", "tool_calls": []},
        {"role": "user", "content": agent.agent_turn("The box is still booting.")},
        {"role": "assistant", "content": "a box is a tiny nap spot.", "tool_calls": []}]
OLD_TOOL_ROWS = [{"role": "assistant", "content": "", "tool_calls": [{"name": "bash", "arguments": {"command": "ls"}}]},
                 {"role": "tool", "results": ["diary"]}]


def test_label_prompt_numbers_only_replies():
    text, n = gist.chat_for_label_msgs(CHAT)
    assert n == 3 and "[R3] a box is a tiny nap spot." in text
    p = gist.label_prompt(text, n, cards.MOTH_CARD)
    assert "exactly 3 entries" in p and "the box is still booting" in p.lower()
    for bad in ("tool", "file", "home folder"):
        assert bad not in gist.LABEL_PROMPT.lower(), bad
    old = CHAT[:3] + OLD_TOOL_ROWS + CHAT[3:]   # old tools data: tool rows are not shown and not numbered
    assert gist.chat_for_label_msgs(old) == (text, n)


def test_parse_labels():
    good = 'ok {"labels": [{"n": 1, "why": "asks about the toast", "on": true, "self": false}, {"n": 2, "why": "x", ' \
           '"on": "true", "self": "true"}, {"n": 3, "why": "word grab", "on": false, "self": false}]}'
    labs = gist.parse_labels(good, 3)
    assert [x["on"] for x in labs] == [True, True, False] and [x["self"] for x in labs] == [False, True, False]
    with pytest.raises(ValueError):   # "self" is required: it is the authoritative identity metric
        gist.parse_labels('{"labels": [{"n": 1, "why": "x", "on": true}]}', 1)
    with pytest.raises(ValueError):
        gist.parse_labels(good, 2)
    with pytest.raises(ValueError):
        gist.parse_labels('{"labels": [{"n": 1, "on": "maybe", "self": false}]}', 1)


def test_build_sft_cuts_at_first_off_reply():
    pytest.importorskip("torch")
    from moth import build_sft
    d = {"seed": 1, "type": "talk", "card": cards.MOTH_CARD,
         "messages": [m for m in CHAT[:4]] + [{"role": "user", "content": "lol"},
                                              {"role": "assistant", "content": "you laughed!", "tool_calls": []}]}
    on = [{"on": True, "self": False, "why": ""}] * 3
    why = collections.Counter()
    assert len(build_sft.keep(d, why, on)) == len(d["messages"])
    out = build_sft.keep(d, why, on[:2] + [{"on": False, "self": False, "why": "x"}])
    assert out[-1]["content"] == "just chatting with you. and thinking about toast." and why["off_gist:cut"] == 1
    assert len(build_sft.keep(d, why, None)) == len(d["messages"])        # unlabelled: no cut
    assert len(build_sft.keep(d, why, on[:1])) == len(d["messages"]) and why["labels_mismatch"] == 1


def test_build_sft_drops_chats_with_unprompted_identity_talk():
    pytest.importorskip("torch")
    from moth import build_sft
    msgs = []
    for u, r in [("hi", "hi! what are you up to?"), ("how are you", "good. tired."),
                 ("what did you eat", "soup. and some dust off the lamp.")]:
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": r, "tool_calls": []}]
    d = {"seed": 2, "type": "talk", "card": cards.MOTH_CARD, "messages": msgs}
    why = collections.Counter()
    assert build_sft.keep(d, why) is None and why["identity_ref:drop"] == 1          # unlabelled: the word matcher
    lab = [{"on": True, "self": False, "why": ""}] * 3
    assert len(build_sft.keep(d, why, lab)) == 6      # labelled: the label is authoritative, the matcher is off
    lab2 = lab[:1] + [{"on": True, "self": True, "why": "talks about its wings"}] + lab[2:]
    d2 = {**d, "messages": msgs[:4] + [{"role": "user", "content": "nice"}, {"role": "assistant", "content": "yes!",
                                                                          "tool_calls": []}]}
    assert build_sft.keep(d2, why, lab2) is None and why["identity_label:drop"] == 1


def test_build_sft_pure_chat_drops_tool_rows_and_file_talk():
    pytest.importorskip("torch")
    from moth import build_sft
    why = collections.Counter()
    d = {"seed": 3, "type": "talk", "card": cards.MOTH_CARD, "messages": CHAT[:3] + OLD_TOOL_ROWS + CHAT[3:4]}
    assert build_sft.keep(d, why) is None and why["tool_use:drop"] == 1
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hi! how are you?", "tool_calls": []},
            {"role": "user", "content": "good"}, {"role": "assistant", "content": "i wrote it in saved/notes.txt!",
                                                   "tool_calls": []}]
    assert build_sft.keep({"seed": 4, "type": "talk", "card": cards.MOTH_CARD, "messages": msgs}, why) is None
    assert why["file_talk:drop"] == 1


def test_identity_ref():
    m = cards.MOTH_CARD
    assert gist.identity_ref("i sat by the lamp all day", m, "how was your day")
    assert not gist.identity_ref("the lamp? i like it", m, "do you like my lamp")     # the friend brought it up
    assert not gist.identity_ref("i'm moth, i live on a graphics card", m, "who are you?")
    assert gist.identity_ref("i'm just a little guy, i don't know", m, "what's 2+2")
    plant = cards.load_card({"id": "p", "name": "fern", "kind": "a houseplant", "refs": ["leaf", "leaves", "pot", "soil"]})
    assert gist.identity_ref("my pot is too small today", plant, "how are you")
    assert not gist.identity_ref("my lamp is on", plant, "hi")   # moth words don't count for a plant
    assert gist.identity_ref_rate([("hi", "hi!"), ("yo", "my soil is dry")], plant) == 0.5


NEUTRAL = ["i've been so tired", "i hope it works", "a walk sounds nice", "go team", "that is popular",
           "i like ice cream", "the website is down", "catch it!", "my login broke", "i love potatoes", "what a season",
           "the fireworks were loud", "is it hot outside?", "did it rain?", "a fancy hat", "that sounds fun",
           "the box is still booting", "she leaves at five", "the light was green"]


def test_identity_ref_ignores_everyday_words_for_every_card():
    for c in cards.build_pool() + cards.heldout_cards() + [cards.MOTH_CARD, cards.BLANK_CARD]:
        hits = [n for n in NEUTRAL if gist.identity_ref(n, c, "hello")]
        assert not hits, (c["id"], c.get("kind"), hits)


def test_card_words_in_the_friends_line_do_not_invite_identity_talk():
    pytest.importorskip("openai")
    from moth import gen_dialogs as G
    dog = next(c for c in cards.build_pool() if c.get("kind") == "a dog")
    assert G.teacher_view("i went for a walk with my dog", card=dog).endswith(G.TALK_REMIND)


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
                    f.write(json.dumps({"seed": s, "n": 2, "labels": [{"on": True, "self": False, "why": ""},
                                                                    {"on": s != 20, "self": s == 20, "why": ""}]}) + "\n")
        ds, labs, boxes, warn = merge_gen.merge([os.path.join(d, "a"), os.path.join(d, "b")])
    assert sorted(x["seed"] for x in ds) == [10, 11, 20] and sorted(labs) == [10, 11, 20]
    assert any("overlap" in w for w in warn) and any("already merged" in w for w in warn)
    assert boxes[1]["on_gist"] == 0.75 and boxes[1]["self"] == 0.25


# --- languages and leakage (2026-10-19 review) -------------------------------------------------------------------------
def test_chat_languages():
    rng = random.Random(0)
    n = 6000
    got = collections.Counter(cards.sample_card(rng)["lang"] for _ in range(n))
    for k, p in cards.LANGS.items():
        assert abs(got[k] / n - p) < 0.03, (k, got[k] / n)
    c = {**cards.build_pool()[2], "lang": "zh"}
    t = persona.teacher_system(c)
    assert t.startswith(persona.TEACHER_CORE) and "LANGUAGE OF THIS CHAT: Chinese" in t
    assert "lowercase does not apply" in cards.language_rule(c) and "lowercase like" in cards.language_rule({"lang": "es"})
    assert "LANGUAGE OF THIS CHAT" not in persona.teacher_system({**c, "lang": "en"})
    assert "Chinese" in gist.label_prompt("friend: 你好\n[R1] 你好呀", 1, c)
    assert "Chinese" not in gist.label_prompt("friend: hi\n[R1] hi", 1, {**c, "lang": "en"})


def test_sim_and_reminders_follow_the_language():
    pytest.importorskip("openai")
    from moth import gen_dialogs as G
    c = {**cards.build_pool()[2], "lang": "hi"}
    assert "Hindi" in G.sim_prompt("p", "s", 6, 1, [], "rain", c)
    assert "Write in English." in G.sim_prompt("p", "s", 6, 1, [], "rain", {**c, "lang": "en"})
    assert G.teacher_view("नमस्ते", card=c).endswith("(reply in Hindi (Devanagari script) only.)")
    assert G.teacher_view("hi", card={**c, "lang": "en"}).endswith(G.TALK_REMIND)


def test_heldout_cards_never_train():
    assert all(cards.is_heldout(c) for c in cards.heldout_cards() + cards.heldout_cards(free=True))
    assert not any(cards.is_heldout(c) for c in cards.build_pool())
    pytest.importorskip("torch")
    from moth import build_sft
    with pytest.raises(SystemExit):
        build_sft.refuse_eval_inputs(["results/bakeoff/rented/bk-2/bakeoff/qwen27/dialogs.jsonl"])
    build_sft.refuse_eval_inputs(["results/g4/dialogs.jsonl"])
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.jsonl")
        msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hi! what's up?", "tool_calls": []},
                {"role": "user", "content": "nothing"}, {"role": "assistant", "content": "nothing is nice.", "tool_calls": []}]
        with open(p, "w") as f:
            for i, card in enumerate([cards.heldout_cards()[0], cards.build_pool()[0]]):
                f.write(json.dumps({"seed": i, "type": "talk", "card": card, "messages": msgs}) + "\n")
        why = collections.Counter()
        items, _ = build_sft.filter_chats([p], why)
    assert [it["seed"] for it in items] == [1] and why["heldout_card"] == 1
