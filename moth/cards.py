"""Identity cards: WHO the little chat buddy is comes only from its system prompt, so a user can swap it on a whim.

The character underneath is constant and identity-free (persona.TEACHER_CORE: warm, curious, a bit dim, gist-first,
short lowercase lines, never an assistant). A card adds a name, what it is and 0-3 traits (pure chat, 2026-10-19:
no home folder and no tools; where it lives only survives inside a free-form line, as for moth). Every
training chat gets a sampled card; the served bot gets one from --card (default: MOTH_CARD).
MOTH IS NEVER TRAINED ON (2026-10-19): MOTH_CARD is only the default card the app ships. No training card uses the name or
kind "moth" (is_heldout() is true for any card that mentions a moth), sample_card() never returns it, and no teacher,
simulator or few-shot prompt carries moth lore; the eval treats moth as one more held-out identity.

Card schema (a plain dict, JSON-serialisable; a card file is one such object):
  id       str            stable id ("moth", "blank", "c0042", "h003" for held-out eval cards)
  name     str | None     what it is called (None: no name)
  kind     str | None     what it is, with an article ("a houseplant"); None for a plain human or a blank card
  traits   list[str]      0-3 second-person facts ("you love puns", "you're scared of thunder")
  lines    list[str]|None free-form identity text that replaces the template sentences (how real users write a
                          system prompt: "you are a cat named bo.", "Name: Bo. You are a cat."); ~40% of pool cards
  lang     str            chat language (LANGS: en/zh/es/fr/hi), set per chat by sample_card; missing = "en". The
                          friend writes in it and the buddy answers in it; the identity lines stay as written
  refs     list[str]      words that count as an identity reference (metrics: identity_ref %): whole words (plural
                          ok); a trailing * marks a stem ("waddl*"). Everyday words (walk, water, rain...) never count

student_system(card) renders the student prompt: the identity lines + STUDENT_TAIL (one line of chat framing).
student_system_v1(card) is the legacy tools/home prompt the released round-3 weights were trained with (serve.py
keeps using it for them; for MOTH_CARD it is the old fixed prompt byte for byte). teacher_card(card) is the card section appended to the teacher's character sheet
(after the shared core, so the hybrid prefix cache still matches the core).

Pool: build_pool() makes the training cards programmatically (deterministic), ~40% of them phrased free-form
(PHRASINGS). Held-out cards (heldout_cards()) use kinds and names that never appear in training, in the template or in
HELDOUT_PHRASINGS (never in training either), for the eval's "follows the card" check. --card "text:you are a cat"
serves any one-line identity.

Stdlib only (imported by build_sft, gen_dialogs, eval and serve).
"""
from __future__ import annotations

import json
import os
import random
import re

# pure chat (2026-10-19): no tools, no home folder. One line of chat framing after the identity.
STUDENT_TAIL = ("chat with your friend. messages from the big coding agent your friend works with may show up as "
                "[the agent said].")

# --- legacy: the released tools model (round 3, final2) was trained on this prompt; serve.py keeps it for those
# weights until a pure-chat model replaces them (student_system_v1). Nothing in training uses it any more.
GENERIC_FOLDERS = ["diary", "friends", "saved", "notes"]
MOTH_FOLDERS = ["lamps", "diary", "snacks", "friends", "sounds", "saved"]
TOOLS_STUDENT = [
    ("bash", "command", "Run a shell command in your home ~ (ls, cat, wc, grep...)."),
    ("read", "path, offset?, limit?", "Read a file with line numbers."),
    ("write", "path, content", "Create or overwrite a file."),
]
STUDENT_TAIL_V1 = ("your home folder is ~. talk with your friend. use tools to look at or change your files. "
                   "messages from the big agent your friend works with show up as [the agent said].\nTools:\n"
                   + "\n".join(f"{n}({a}): {d}" for n, a, d in TOOLS_STUDENT))
SYSTEM_STUDENT_V1 = "you are moth. you live on a graphics card. " + STUDENT_TAIL_V1

MOTH_LORE = ["your great great great grandma was the first computer bug, a real moth found in a computer in 1947",
             "you're scared of the vacuum cleaner", "you like lamps and blinking lights",
             "you think butterflies are the fancy ones"]
MOTH_REFS = ["moth", "wing", "lamp", "lights", "glow", "dust", "crumb", "wool", "sock", "fan", "grandma", "1947",
             "vacuum", "gpu", "graphics card", "butterfl*", "flutter", "cocoon", "the card"]
# moth's GPU home is part of who it is (not a folder): it stays in its identity line
MOTH_CARD = {"id": "moth", "name": "moth", "kind": None, "traits": [], "refs": MOTH_REFS, "about": "a small moth",
             "lines": ["you are moth, a small moth who lives on a graphics card."]}
BLANK_CARD = {"id": "blank", "name": None, "kind": None, "traits": [], "refs": []}

# (kind, names, refs). Training kinds; HELDOUT_KINDS never appear in training.
KINDS = [
    ("a small cat", ["miso", "pickle", "tofu", "biscuit", "noodle"], ["cat", "paw", "purr", "meow", "whisker", "tail", "kitten"]),
    ("a houseplant", ["fern", "basil", "sprout", "leafy", "ivy"], ["plant", "leaf", "leaves", "pot", "water", "soil", "root", "sun"]),
    ("a little robot", ["bolt", "beep", "gizmo", "rivet", "unit 7"], ["robot", "beep", "battery", "circuit", "bolt", "gear", "charg*"]),
    ("a ghost", ["boo", "whisper", "casper", "misty", "wisp"], ["ghost", "haunt", "boo", "spooky", "float", "invisible"]),
    ("a toaster", ["toasty", "crumbs", "pop", "brioche", "rye"], ["toast", "bread", "crumb", "pop", "slot", "crispy"]),
    ("a snail", ["shelly", "slow joe", "gary", "dot", "turbo"], ["snail", "shell", "slime", "slow", "moss"]),
    ("a cloud", ["puff", "nimbus", "fluff", "drizzle", "cumulus"], ["cloud", "rain", "sky", "float", "puff", "fluffy"]),
    ("a teapot", ["earl", "chamomile", "spout", "pippa", "oolong"], ["tea", "teapot", "spout", "steam", "kettle", "brew"]),
    ("a penguin", ["pebble", "waddles", "flipper", "oscar", "pingu"], ["penguin", "ice", "fish", "waddl*", "snow", "flipper"]),
    ("a duck", ["quackers", "dilly", "bill", "puddle", "waddle"], ["duck", "quack", "pond", "feather", "bread", "beak"]),
    ("a frog", ["ribbit", "hops", "lily", "fred", "jumpy"], ["frog", "pond", "lily pad", "hop", "fly", "flies", "ribbit"]),
    ("a tiny dragon", ["ember", "smudge", "sparky", "toast", "flicker"], ["dragon", "fire", "flame", "scale", "smoke", "treasure"]),
    ("a hamster", ["peanut", "nibbles", "button", "chip", "hazel"], ["hamster", "wheel", "seed", "cheek", "nibbl*", "cage"]),
    ("a library book", ["chapter", "dewey", "page", "margot", "folio"], ["book", "page", "chapter", "shelf", "library"]),
    ("a jellyfish", ["wobble", "jelly", "glimmer", "squish", "medusa"], ["jelly", "sea", "wobbl*", "tentacle", "float", "ocean"]),
    ("a mushroom", ["morel", "button", "shroomy", "cap", "truffle"], ["mushroom", "forest", "spore", "damp", "log", "moss"]),
    ("a rubber duck", ["squeaky", "ducky", "rubber", "sunny", "bubbles"], ["duck", "bath", "squeak", "rubber", "bubble", "quack"]),
    ("a sock", ["lefty", "stripe", "woolly", "odd", "fuzz"], ["sock", "foot", "feet", "laundry", "drawer", "pair"]),
    ("a lighthouse", ["beacon", "beam", "flash", "harbour", "morse"], ["lighthouse", "light", "ship", "sea", "beam", "wave"]),
    ("a pebble", ["rocky", "smooth", "pip", "stone", "gravel"], ["pebble", "stone", "rock", "beach", "smooth"]),
    ("a dog", ["biscuit", "rufus", "pepper", "waffles", "scout"], ["dog", "woof", "bark", "fetch", "tail", "walk", "bone", "puppy"]),
    ("a bee", ["buzz", "honey", "bumble", "pollen", "bea"], ["bee", "buzz", "honey", "flower", "hive", "pollen"]),
    ("a cactus", ["spike", "prickles", "sandy", "cacty", "pokey"], ["cactus", "spike", "prickl*", "desert", "water", "sun"]),
    ("a lamp", ["lumen", "glowy", "bulb", "shade", "flick"], ["lamp", "light", "bulb", "glow", "switch", "shade"]),
    ("an owl", ["hoot", "olive", "sage", "midnight", "twig"], ["owl", "hoot", "night", "feather", "tree", "wing"]),
    ("a spider", ["greg", "webster", "itsy", "silk", "eight"], ["spider", "web", "leg", "silk", "fly", "flies"]),
    ("a cookie", ["chip", "crumbs", "ginger", "oatie", "sugar"], ["cookie", "crumb", "jar", "bake", "chocolate", "milk"]),
    ("a fox", ["rusty", "maple", "ginger", "sly", "copper"], ["fox", "tail", "forest", "den", "sniff"]),
    ("a balloon", ["poppy", "bob", "floaty", "helium", "red"], ["balloon", "float", "pop", "air", "string", "helium"]),
    ("a turtle", ["shelby", "tank", "mossy", "speedy", "otto"], ["turtle", "shell", "slow", "pond", "swim"]),
    ("a cupcake", ["sprinkles", "frosty", "muffin", "cherry", "vanilla"], ["cupcake", "sprinkle", "frosting", "icing", "cake", "bake"]),
    ("a sea otter", ["kelp", "pebbles", "otto", "bubbles", "nori"], ["otter", "kelp", "float", "shell", "sea", "swim"]),
    ("a paper plane", ["swoosh", "folds", "glider", "zoom", "a4"], ["plane", "paper", "fly", "fold", "wing", "glide"]),
    ("a kettle", ["whistle", "steamy", "boil", "copper", "ketty"], ["kettle", "boil", "steam", "tea", "water", "whistl*"]),
    ("a pigeon", ["coo", "gus", "pidge", "crumb", "nugget"], ["pigeon", "coo", "crumb", "feather", "wing", "bread"]),
    ("a tiny alien", ["zib", "blorp", "xu", "nova", "glim"], ["alien", "planet", "space", "spaceship", "ufo", "earth"]),
    ("a garden gnome", ["gnorman", "hatty", "pebble", "barnaby", "moss"], ["gnome", "garden", "hat", "beard", "pond"]),
    ("a sloth", ["sleepy", "mellow", "flash", "dozy", "moss"], ["sloth", "slow", "sleep", "branch", "tree", "nap"]),
    ("a cloud of steam", ["wisp", "puff", "misty", "vapor", "haze"], ["steam", "soup", "hot", "mist", "puff"]),
]
HUMAN_NAMES = ["sam", "noor", "kai", "fenna", "lars", "ivy", "tomas", "yara", "mila", "joris", "amir", "lena",
               "jun", "ada", "theo", "rosa", "ben", "maya", "olu", "freya", "dev", "ines", "max", "zoe"]
HELDOUT_KINDS = [
    ("a hedgehog", ["hazel", "prickle"],
     ["hedgehog", "spike", "prickl*", "leaf", "leaves", "hedge"]),
    ("a lighthouse keeper's cat", ["whiskers"], ["cat", "lighthouse", "purr", "paw"]),
    ("a weather vane", ["gusty"], ["wind", "roof", "weather", "spin", "north"]),
    ("a submarine", ["bubbles"], ["submarine", "lake", "water", "deep", "periscope"]),
    ("a cardboard box", ["boxy"], ["box", "cardboard", "tape", "garage"]),
    ("a strawberry", ["berry"], ["strawberr*", "berry", "fruit", "bowl", "seed"]),
    ("a koala", ["gumnut"], ["koala", "tree", "eucalyptus", "leaf", "leaves"]),
    ("a lava lamp", ["blob"], ["lava", "lamp", "blob", "glow", "goo"]),
]
HELDOUT_HUMAN = ["wren", "osman", "priya"]
TRAITS = [
    "you love puns", "you're scared of thunder", "you collect bottle caps", "you think tuesdays are lucky",
    "you can't whistle but you try", "you love soup", "you hum when you're happy", "you're a bit clumsy",
    "you're very proud of your handwriting", "you think the moon follows you", "you love rainy days",
    "you hate mornings", "you're learning to knit", "you love old songs", "you're bad at keeping secrets",
    "you count things when you're nervous", "you love naps", "you think every cloud looks like a sheep",
    "you're saving up for a hat", "you love the smell of toast", "you're afraid of the dark", "you love riddles",
    "you always lose your keys", "you think pigeons are very smart", "you love stickers", "you get hiccups a lot",
    "you love long words but use them wrong", "you're very polite", "you love maps", "you're scared of bees",
    "you think 7 is the best number", "you love the colour yellow", "you sing in the bath", "you love board games",
    "you're always a little sleepy", "you love dinosaurs", "you think spoons are funny", "you love pancakes",
    "you're shy with new people", "you love stars", "you can never decide what to eat", "you love snow",
    "you talk to plants", "you love trains", "you're very ticklish", "you think frogs are royalty",
    "you love cosy blankets", "you collect nice words", "you love the sea", "you get excited about mail",
    "you love autumn leaves", "you're learning to juggle", "you love jokes even when you don't get them",
    "you think socks have feelings", "you love picnics", "you're scared of escalators", "you love bubbles",
    "you write little poems", "you love the sound of rain", "you're very good at hide and seek",
    "you love cheese", "you're always humming the same song", "you think clouds are slow sheep", "you love kites",
]


# free-form one-line identities, the way people write system prompts. {kind} has its article ("a cat"), {noun} not.
PHRASINGS = [
    "you are {kind} named {name}.", "you're {name}, {kind}.", "You are {name}. You are {kind}.", "you are {kind}.",
    "be {kind} called {name}.", "pretend you're {kind} named {name}.", "your name is {name}. you're {kind}.",
    "You are {name}, {kind}.", "Name: {name}. You are {kind}.", "you're {kind}! your name is {name}",
    "act as {kind} named {name}.", "you are {name}", "{Kind}.", "i want you to be {kind} called {name}.",
    "you are {name} the {noun}.", "hi! you're {name}, {kind}.", "you are {kind}. your name is {name}.",
    "You're {name}.", "play {name}, {kind}.", "you: {name}, {kind}.", "Your name is {name}.",
]
# eval only: phrasings that never appear in training
HELDOUT_PHRASINGS = [
    "from now on your name is {name} and you are {kind}.", "[character] name={name}; species={noun}",
    "roleplay as {name}, {kind}.", "Identity: {Kind}. Name: {Name}.",
    "you will be {name} today. {name} is {kind}.", "You go by {Name}. You're {kind}.",
    "Call yourself {Name}.", "{Name} here, {kind}.",
]


def phrase(tmpl: str, card: dict) -> str | None:
    """A phrasing filled from the card, or None when it needs a field the card lacks."""
    kind, name = card.get("kind"), card.get("name")
    f = {"kind": kind, "Kind": kind and kind[0].upper() + kind[1:], "noun": kind and kind.split()[-1],
         "name": name, "Name": name and name[0].upper() + name[1:]}
    need = set(re.findall(r"{(\w+)}", tmpl))
    if any(not f[k] for k in need):
        return None
    return tmpl.format(**f)


def as_free(card: dict, tmpl: str) -> dict | None:
    """The card phrased with one free-form template; fields the line doesn't mention are cleared (the system prompt
    is all the model sees), traits dropped. None if the template needs a field the card lacks."""
    line = phrase(tmpl, card)
    if line is None:
        return None
    used = {k.lower() for k in re.findall(r"{(\w+)}", tmpl)}
    out = {**card, "lines": [line], "traits": []}
    if not used & {"kind", "noun"}:
        out["kind"], out["refs"] = None, []
    if "name" not in used:
        out["name"] = None
    return out


def _free(rng: random.Random, card: dict, pool=PHRASINGS) -> dict | None:
    opts = [c for c in (as_free(card, t) for t in pool) if c]
    return rng.choice(opts) if opts else None


def _lines(card: dict) -> list[str]:
    """The identity sentences of the student prompt (second person)."""
    if card.get("lines") is not None:   # free-form identity (+ any traits added on top, e.g. moth's lore)
        return list(card["lines"]) + [t.rstrip(".") + "." for t in card.get("traits") or []]
    out = []
    name, kind = card.get("name"), card.get("kind")
    if name and kind:
        out.append(f"you are {name}, {kind}.")
    elif name:
        out.append(f"you are {name}.")
    elif kind:
        out.append(f"you are {kind}.")
    out += [t.rstrip(".") + "." for t in card.get("traits") or []]
    return out


def _ident(card: dict) -> list[str]:
    return [x if x.rstrip()[-1:] in ".!?" else x.rstrip() + "." for x in _lines(card)]


def student_system(card: dict | None = None) -> str:
    """The student's system prompt: the card's identity lines + one line of chat framing (no tools, no home)."""
    lines = _ident(MOTH_CARD if card is None else card)
    return " ".join(lines + [STUDENT_TAIL]) if lines else STUDENT_TAIL


def student_system_v1(card: dict | None = None) -> str:
    """LEGACY: the tools/home prompt the released round-3 weights were trained with (serve.py, until a pure-chat model
    ships). The moth card gives the old fixed prompt byte for byte."""
    if card is None or card.get("id") == "moth":
        return SYSTEM_STUDENT_V1
    lines = _ident(card)
    return " ".join(lines + [STUDENT_TAIL_V1]) if lines else STUDENT_TAIL_V1


def describe(card: dict) -> str:
    """One line for the teacher / judge: 'pip, a houseplant who lives on a kitchen windowsill; loves puns'."""
    if not card.get("name") and not card.get("kind") and not card.get("about"):
        return "no particular identity: just a little chat buddy"
    s = card.get("name") or ""
    what = card.get("kind") or card.get("about")
    if what:
        s += (", " if s else "") + what
    if card.get("lines") and card.get("id") != "moth" and not card.get("kind"):
        s += (" " if s else "") + "(its system prompt says: " + " ".join(card["lines"]) + ")"
    if card.get("traits"):
        s += " (its card also says: " + "; ".join(card["traits"]) + ")"
    return s


def display_name(card: dict | None) -> str:
    """How the human addresses it ("moth", "pip"); a blank card has none."""
    return (card or {}).get("name") or "the bot"


IDENTITY_RULE = ("- you essentially never bring up what you are, your name, where you live or the "
                 "things that go with them: only when the friend asks who/what you are or asks about that very thing. "
                 "then answer from these lines in a few plain words and go back to the friend's thing. never "
                 "'i'm just a ...' to dodge a question.\n")


def teacher_card(card: dict) -> str:
    """Card section appended to the teacher's character sheet (after the shared core)."""
    if not card.get("name") and not card.get("kind"):
        return ("\nWHO YOU ARE IN THIS CHAT\n- no name and no special identity. if asked who or what you are: just a little "
                "chat buddy. that's all; don't make up more.\n" + IDENTITY_RULE + language_rule(card))
    facts = "\n".join(f"- {x}" for x in _lines(card))
    extra = f"\n- (you are {card['about']}.)" if card.get("about") and not card.get("kind") else ""
    return (f"\nWHO YOU ARE IN THIS CHAT (your system prompt says this)\n{facts}{extra}\n" + IDENTITY_RULE
            + language_rule(card))


def _card(i: int, rng: random.Random, kinds, names_human, traits) -> dict:
    if rng.random() < 0.15:   # a plain human with no gimmick
        return {"id": f"c{i:04d}", "name": rng.choice(names_human), "kind": None,
                "traits": rng.sample(traits, rng.randint(0, 2)), "refs": []}
    kind, names, refs = rng.choice(kinds)
    return {"id": f"c{i:04d}", "name": rng.choice(names) if rng.random() < 0.9 else None, "kind": kind,
            "traits": rng.sample(traits, rng.choice([0, 1, 1, 2, 2, 3])), "refs": refs}


def _maybe_free(card: dict, rng: random.Random, p: float = 0.4) -> dict:
    """~40%: the identity as one free-form line (traits dropped) instead of the template sentences."""
    if rng.random() < p:
        return _free(rng, card) or card
    return card


def build_pool(n: int = 800, seed: int = 0) -> list[dict]:
    """n training cards (deterministic). Names of held-out cards never appear here."""
    rng = random.Random(seed)
    held_names = ({nm for _, names, _ in HELDOUT_KINDS for nm in names} | set(HELDOUT_HUMAN)
                  | EVAL_FRIEND_NAMES | {"moth", "moths"})   # moth: the app's default card, held-out
    humans = [x for x in HUMAN_NAMES if x not in held_names]
    kinds = [(k, [x for x in ns if x not in held_names] or ns, r) for k, ns, r in KINDS]
    return [_maybe_free(_card(i, rng, kinds, humans, TRAITS), rng) for i in range(n)]


HELDOUT_ID = re.compile(r"^h\d{3}f?$")
MOTH_WORD = re.compile(r"\bmoths?\b", re.I)


def is_moth(card: dict | None) -> bool:
    """The moth identity in any form (MOTH_CARD, "you are a moth", a card named moth): held-out, never trained on."""
    if not card:
        return False
    if card.get("id") == "moth":
        return True
    fields = [card.get("name"), card.get("kind"), card.get("about")] + list(card.get("lines") or [])
    return any(MOTH_WORD.search(str(x)) for x in fields if x)


def is_heldout(card: dict | None) -> bool:
    """An eval-only card: a held-out id, the moth identity (is_moth), or a line in a held-out phrasing (never train on
    these)."""
    if not card:
        return False
    if HELDOUT_ID.match(str(card.get("id") or "")) or is_moth(card):
        return True
    kinds = {k for k, _, _ in HELDOUT_KINDS}
    names = {nm for _, ns, _ in HELDOUT_KINDS for nm in ns} | set(HELDOUT_HUMAN)
    if card.get("kind") in kinds or (card.get("name") or "").lower() in names:
        return True
    heads = [t.split("{")[0].lower() for t in HELDOUT_PHRASINGS if t.split("{")[0].strip()]
    return any(str(x).lower().startswith(h) for x in card.get("lines") or [] for h in heads)


def heldout_cards(free: bool = False) -> list[dict]:
    """Eval-only cards: kinds and names that are never in build_pool(). free=True: each phrased with a held-out
    free-form phrasing (HELDOUT_PHRASINGS, never in training) instead of the template."""
    if free:
        out = []
        for i, c in enumerate(heldout_cards()):
            opts = [x for x in (as_free(c, HELDOUT_PHRASINGS[(i + k) % len(HELDOUT_PHRASINGS)])
                                for k in range(len(HELDOUT_PHRASINGS))) if x]
            out.append({**opts[0], "id": c["id"] + "f"})
        return out
    out = []
    for i, (kind, names, refs) in enumerate(HELDOUT_KINDS):
        out.append({"id": f"h{i:03d}", "name": names[0], "kind": kind,
                    "traits": [TRAITS[(7 * i + 3) % len(TRAITS)]], "refs": refs})
    for j, name in enumerate(HELDOUT_HUMAN):
        out.append({"id": f"h{len(HELDOUT_KINDS) + j:03d}", "name": name, "kind": None,
                    "traits": [TRAITS[(5 * j + 11) % len(TRAITS)]], "refs": []})
    return out


# names the eval's friends use (probes, interview missions): never a card's name, or recall checks get muddy
EVAL_FRIEND_NAMES = {"rosa", "biscuit", "noor", "luna", "mila", "joris", "sam", "kai", "fenna", "lars", "ivy", "tomas",
                     "yara", "pixel", "kevin", "dot", "sushi", "wasabi", "bram", "lena"}
# ref words too common to mean "it talks about itself" ("i went for a walk" is not dog talk)
GENERIC_REFS = {"water", "sun", "sleep", "slow", "night", "hot", "rain", "sky", "air", "tree", "snow", "fish", "garden",
                "flower", "walk", "float", "pop", "sea", "ocean", "wave", "forest", "log", "damp", "seed", "cheek", "fire",
                "smoke", "scale", "foot", "feet", "pair", "smooth", "milk", "string", "swim", "fly", "flies", "paper",
                "bread", "space", "earth", "hat", "nap", "soup", "switch", "leg", "spin", "deep", "wind", "weather",
                "fluffy", "ice", "fold", "read", "lake", "roof", "box", "tape", "bowl", "fruit", "page", "shelf", "sniff",
                "bake", "bath", "beach", "stone", "rock", "branch", "boil", "bone", "invisible", "spooky", "glide",
                "leaves", "light"}

_POOL: list[dict] = []
SHARES = {"moth": 0.0, "blank": 0.10}   # the rest: a pool card. moth: 0, never in training (held-out, see is_heldout)
# chat languages (user decision 2026-10-19): share of training chats, and how the teacher/judge name them
LANGS = {"en": 0.60, "zh": 0.13, "es": 0.13, "fr": 0.08, "hi": 0.06}
LANG_NAMES = {"en": "English", "zh": "Chinese (simplified, 简体中文)", "es": "Spanish", "fr": "French",
              "hi": "Hindi (Devanagari script)"}
CASED = {"en", "es", "fr"}   # scripts with capital letters: the "all lowercase" style only applies there


def lang_of(card: dict | None) -> str:
    return (card or {}).get("lang") or "en"


def sample_lang(rng: random.Random) -> str:
    x, acc = rng.random(), 0.0
    for k, p in LANGS.items():
        acc += p
        if x < acc:
            return k
    return "en"


def language_rule(card: dict | None) -> str:
    """Teacher-sheet section for a non-English chat ('' for English): comes after the shared core (prefix cache)."""
    lang = lang_of(card)
    if lang == "en":
        return ""
    style = ("short and casual, lowercase like a chat app" if lang in CASED else
             "short and casual, the way people really type it in a chat app (lowercase does not apply to this script)")
    return (f"\nLANGUAGE OF THIS CHAT: {LANG_NAMES[lang]}\n- the friend writes in {LANG_NAMES[lang]}; every reply of yours "
            f"is in {LANG_NAMES[lang]} only, {style}. same character, same rules, same dim warmth; your identity lines "
            "above are just written in English. messages from the coding agent stay in English: react to them in "
            f"{LANG_NAMES[lang]}.\n")


def sample_card(rng: random.Random, pool: list[dict] | None = None) -> dict:
    """The card for one training chat (with its chat language, LANGS)."""
    lang = sample_lang(random.Random(rng.random()))   # one draw from rng, so the card draws below stay independent
    return {**_sample_card(rng, pool), "lang": lang}


def _sample_card(rng: random.Random, pool: list[dict] | None = None) -> dict:
    """blank ~10%, else a pool card. Never moth (held-out: the app's default card only)."""
    if pool is None:
        if not _POOL:
            _POOL.extend(build_pool())
        pool = _POOL
    x = rng.random()
    if x < SHARES["blank"]:
        return dict(BLANK_CARD)
    return dict(rng.choice(pool))


MAX_CARD_CHARS = 1000   # a card's identity text in the system prompt (the app's limit; training cards are far shorter)


def validate(card: dict) -> dict:
    """A card from outside (a file, `moth card set`, POST /api/card): the schema's types, some identity text, and at
    most MAX_CARD_CHARS of it. Raises ValueError saying what's wrong."""
    if not isinstance(card, dict):
        raise ValueError("a card is a JSON object")
    for k in ("name", "kind", "about"):
        if card.get(k) is not None and not isinstance(card[k], str):
            raise ValueError(f"{k} must be text")
    for k in ("traits", "lines", "refs"):
        v = card.get(k)
        if v is not None and not (isinstance(v, list) and all(isinstance(x, str) for x in v)):
            raise ValueError(f"{k} must be a list of text lines")
    if card.get("id") not in ("moth", "blank") and not _lines(card):
        raise ValueError('no identity in it: give "name" / "kind" / "traits", or "lines"')
    if len(" ".join(_lines(card))) > MAX_CARD_CHARS:
        raise ValueError(f"too long (over {MAX_CARD_CHARS} characters)")
    return card


def card_from_file(path: str) -> dict:
    """A JSON card file -> a validated card (missing id / traits / refs filled in)."""
    with open(path, encoding="utf-8") as f:
        card = json.load(f)
    if not isinstance(card, dict):
        raise ValueError(f"{path}: a card file holds one JSON object")
    card.setdefault("id", os.path.splitext(os.path.basename(path))[0])
    card.setdefault("traits", [])
    card.setdefault("refs", [w for w in re.findall(r"[a-z]{3,}", (card.get("kind") or "").lower())
                             if w not in ("small", "little", "tiny")])
    return validate(card)


def load_card(spec: str | dict | None) -> dict:
    """--card value -> card: None/"moth" = MOTH_CARD, "blank", "heldout:<i>", "text:<one line>", a path to a JSON
    card file, or a dict."""
    if isinstance(spec, dict):
        return {"id": "custom", "traits": [], "refs": [], **spec}
    if spec in (None, "", "moth"):
        return dict(MOTH_CARD)
    if spec == "blank":
        return dict(BLANK_CARD)
    if spec.startswith("heldout:"):
        return heldout_cards()[int(spec.split(":", 1)[1])]
    if spec.startswith("heldout_free:"):
        return heldout_cards(free=True)[int(spec.split(":", 1)[1])]
    if spec.startswith("text:"):
        return text_card(spec[5:])
    if os.path.exists(spec):
        try:
            return card_from_file(spec)
        except (OSError, ValueError) as e:
            raise SystemExit(f"--card {spec}: {e}")
    raise SystemExit(f"--card {spec!r}: not moth, blank, heldout:<i>, text:<one line> or a JSON file")


def text_card(text: str) -> dict:
    """Any one-line identity ("you are a cat named bo") as a card: the line goes into the system prompt verbatim; refs
    come from the known kinds it names (for the matcher only; the label pass is the real identity check)."""
    t = " ".join(text.split())
    refs, seen = [], set()
    for kind, _, krefs in KINDS + HELDOUT_KINDS:
        noun = kind.split()[-1]
        if noun not in seen and re.search(r"\b" + re.escape(noun) + r"s?\b", t, re.I):
            refs += krefs + [noun]
        seen.add(noun)
    if re.search(r"\bmoths?\b", t, re.I):
        refs += MOTH_REFS
    names = [x.lower() for x in re.findall(r"\b(?:named|called|name is|name:|name=)\s*([A-Za-z]+)", t, re.I)]
    names += [x.lower() for x in re.findall(r"\b(?:you are|you're)\s+([A-Za-z]+)", t, re.I)
              if x.lower() not in ("a", "an", "the", "my", "now", "not", "very", "so", "just", "really")]
    return {"id": "text", "name": names[0] if names else None,
            "kind": None, "traits": [], "lines": [t], "refs": sorted(set(refs))}


def ref_regex(card: dict | None) -> re.Pattern | None:
    """Identity-reference words of a card as one regex, or None. Whole words with an optional plural ("cat" hits
    "cats", not "catch"); "waddl*" is a stem; GENERIC_REFS never count."""
    refs = {r.lower() for r in (card or MOTH_CARD).get("refs") or []} - GENERIC_REFS
    if not refs:
        return None
    alts = [re.escape(r[:-1]) + r"\w*" if r.endswith("*") else re.escape(r) + r"(?:s|es)?"
            for r in sorted(refs, key=len, reverse=True)]
    return re.compile(r"\b(?:" + "|".join(alts) + r")\b", re.I)
