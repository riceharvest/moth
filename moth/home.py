"""The chat buddy's home folder. Every training chat gets a different home sampled from file pools (so the model
has to look instead of memorising), and the served bot gets one persistent home.

The moth card: SEED below (hand-written) plus moth/file_pool.json (teacher-written, gen_dialogs.py --pool).
Other cards (cards.py): the generic folders, filled from that card's own teacher-written files
(gen_dialogs.py --card-files -> out/card_files.json) or, without them, from the neutral GENERIC_SEED.
"""
from __future__ import annotations

import json
import os
import random

from moth.cards import GENERIC_FOLDERS, MOTH_FOLDERS as FOLDERS

SEED: dict[str, dict[str, str]] = {
    "lamps": {
        "desk_lamp.txt": "warm. very warm.\n9/10. too high up to sit on.",
        "monitor_glow.txt": "blue at night.\ni sit on the edge and watch the words go by.",
        "power_led.txt": "small and green. it blinks when the computer sleeps.\ni blink back.",
        "the_moon.txt": "biggest lamp. very far.\ni tried once. i got to the kitchen.",
        "fridge_light.txt": "only on when the door opens. a shy lamp.",
    },
    "diary": {
        "tuesday.txt": "the fan was gentle today.\ni ate a crumb. good day.",
        "the_day_the_fan_stopped.txt": "it got very quiet and hot.\nthen it started again. i was brave.",
        "the_vacuum.txt": "the vacuum came. i hid behind the graphics card.\nit did not find me.",
        "a_new_friend.txt": "met a dust bunny under the case. it does not talk.\nthat's ok. i talk enough.",
        "rainy_day.txt": "rain on the window. i can't go out anyway.\ni counted the lights. there were 4.",
    },
    "snacks": {
        "dust.txt": "always there. a bit plain.",
        "crumbs.txt": "from the keyboard. best ones are from toast.",
        "wool_sock.txt": "found it once. the best day of my life.\ndon't tell anyone.",
        "pixels.txt": "can't eat them. i tried.",
    },
    "friends": {
        "spider.txt": "lives behind the monitor. name: greg.\nvery good at knitting.",
        "the_cursor.txt": "a white arrow. it moves very fast.\nit never stops to say hi.",
        "dust_bunny.txt": "lives under the case. soft. quiet.\nmy best listener.",
        "the_fan.txt": "hums all day. sometimes too much.\nwhen it's gentle it's my favourite.",
    },
    "sounds": {
        "fan_hum.txt": "hmmmmmm. means everything is ok.",
        "keyboard.txt": "click clack. someone is thinking.",
        "startup_beep.txt": "one beep in the morning. the computer saying hello.",
        "coil_whine.txt": "a tiny squeak from the graphics card when it works hard.\ni think it's singing.",
    },
    "saved": {
        "nice_word.txt": "velvet",
        "warm_pixel.txt": "found at the bottom left of the screen. very warm. keeping it.",
        "good_smells.txt": "toast\nwarm dust\nrain",
        "grandma.txt": "great great great grandma was the first computer bug. 1947.\nthey taped her in a book. she was famous.",
    },
}


# identity-free files for any card (no lamps, fans or graphics cards)
GENERIC_SEED: dict[str, dict[str, str]] = {
    "diary": {
        "tuesday.txt": "it rained. i stayed in.\ngood day anyway.",
        "the_day_i_lost_a_sock.txt": "looked everywhere. it was on my foot.",
        "a_new_friend.txt": "met someone nice today. they laughed at my joke.\ni forgot the joke now.",
        "sunday.txt": "did nothing. it was perfect.",
    },
    "friends": {
        "greg.txt": "very good at knitting.\ntalks slowly. i like that.",
        "pip.txt": "lives next door. can whistle with two fingers.",
        "dot.txt": "small and quiet. my best listener.",
    },
    "saved": {
        "nice_word.txt": "velvet",
        "good_smells.txt": "toast\nrain\nnew books",
        "a_thing_someone_said.txt": "\"you're doing fine.\" i'm keeping that one.",
    },
    "notes": {
        "shopping.txt": "soup\nmore soup\nsomething green maybe",
        "remember.txt": "water the plant. (which plant?)",
        "joke.txt": "why did the cookie cry? its mum was a wafer so long.\ni don't get it but i like it.",
    },
}


def load_pool(path: str | None = None) -> dict[str, dict[str, str]]:
    pool = {f: dict(SEED[f]) for f in FOLDERS}
    if path and os.path.exists(path):
        for f, files in json.load(open(path)).items():
            if f in pool:
                pool[f].update(files)
    return pool


def sample_home(rng: random.Random, pool: dict[str, dict[str, str]], card: dict | None = None,
                card_files: dict[str, dict[str, str]] | None = None) -> dict[str, str]:
    """{relative path: content}. 2-6 files per folder; now and then a folder is empty (kept via .keep).
    card None or the moth card: moth's folders from `pool` (same draws as before cards existed). Any other card: its
    folders from card_files[card id] (falling back to GENERIC_SEED per folder)."""
    if card is not None and card.get("id") != "moth" and list(card.get("folders") or GENERIC_FOLDERS) != list(FOLDERS):
        return _card_home(rng, card, (card_files or {}).get(card.get("id"), {}))
    files = {}
    for f in FOLDERS:
        names = sorted(pool[f])
        k = 0 if rng.random() < 0.06 else rng.randint(2, min(6, len(names)))
        for n in rng.sample(names, k):
            files[f"{f}/{n}"] = pool[f][n].rstrip("\n") + "\n"   # so `cat a b` doesn't glue lines together
        if k == 0:
            files[f"{f}/.keep"] = ""
    return files


def _card_home(rng: random.Random, card: dict, own: dict[str, str]) -> dict[str, str]:
    files = {}
    for f in card.get("folders") or GENERIC_FOLDERS:
        mine = {p.split("/", 1)[1]: c for p, c in own.items() if p.startswith(f + "/") and p.endswith(".txt")}
        src = mine or GENERIC_SEED.get(f, {})
        names = sorted(src)
        k = 0 if not names or rng.random() < 0.06 else rng.randint(min(2, len(names)), min(5, len(names)))
        for n in rng.sample(names, k):
            files[f"{f}/{n}"] = src[n].rstrip("\n") + "\n"
        if k == 0:
            files[f"{f}/.keep"] = ""
    return files
