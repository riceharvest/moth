"""moth's home folder. Every training chat gets a different home sampled from file pools (so the model
has to look instead of memorising), and the served bot gets one persistent home.

Pools: SEED below (hand-written) plus out/file_pool.json (teacher-written, built by gen_dialogs.py --pool).
"""
from __future__ import annotations

import json
import os
import random

from moth.persona import FOLDERS

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


def load_pool(path: str | None = None) -> dict[str, dict[str, str]]:
    pool = {f: dict(SEED[f]) for f in FOLDERS}
    if path and os.path.exists(path):
        for f, files in json.load(open(path)).items():
            if f in pool:
                pool[f].update(files)
    return pool


def sample_home(rng: random.Random, pool: dict[str, dict[str, str]]) -> dict[str, str]:
    """{relative path: content}. 2-6 files per folder; now and then a folder is empty (kept via .keep)."""
    files = {}
    for f in FOLDERS:
        names = sorted(pool[f])
        k = 0 if rng.random() < 0.06 else rng.randint(2, min(6, len(names)))
        for n in rng.sample(names, k):
            files[f"{f}/{n}"] = pool[f][n].rstrip("\n") + "\n"   # so `cat a b` doesn't glue lines together
        if k == 0:
            files[f"{f}/.keep"] = ""
    return files
