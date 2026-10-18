"""Identity cards: WHO the little chat buddy is comes only from its system prompt, so a user can swap it on a whim.

The character underneath is constant and identity-free (persona.TEACHER_CORE: warm, curious, a bit dim, gist-first,
short lowercase lines, never an assistant). A card adds a name, what it is, where it lives and 0-3 traits. Every
training chat gets a sampled card; the served bot gets one from --card (default: MOTH_CARD).

Card schema (a plain dict, JSON-serialisable; a card file is one such object):
  id       str            stable id ("moth", "blank", "c0042", "h003" for held-out eval cards)
  name     str | None     what it is called (None: no name)
  kind     str | None     what it is, with an article ("a houseplant"); None for a plain human or a blank card
  home     str | None     where it lives, as it follows "you live " ("on a kitchen windowsill")
  traits   list[str]      0-3 second-person facts ("you love puns", "you're scared of thunder")
  lines    list[str]|None free-form identity text that replaces the template sentences (how real users write a
                          system prompt: "you are a cat named bo.", "Name: Bo. You are a cat."); ~40% of pool cards
  refs     list[str]      words that count as an identity reference (metrics: identity_ref %): whole words (plural
                          ok); a trailing * marks a stem ("waddl*"). Everyday words (walk, water, rain...) never count
  folders  list[str]      folders of its home ~ (generic by default; the moth card keeps its six)

student_system(card) renders the student prompt; for MOTH_CARD it is byte-identical to the old fixed prompt, so the
released weights keep working. teacher_card(card) is the card section appended to the teacher's character sheet
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

GENERIC_FOLDERS = ["diary", "friends", "saved", "notes"]
MOTH_FOLDERS = ["lamps", "diary", "snacks", "friends", "sounds", "saved"]

TOOLS_STUDENT = [
    ("bash", "command", "Run a shell command in your home ~ (ls, cat, wc, grep...)."),
    ("read", "path, offset?, limit?", "Read a file with line numbers."),
    ("write", "path, content", "Create or overwrite a file."),
]
STUDENT_TAIL = ("your home folder is ~. talk with your friend. use tools to look at or change your files. "
                "messages from the big agent your friend works with show up as [the agent said].\nTools:\n"
                + "\n".join(f"{n}({a}): {d}" for n, a, d in TOOLS_STUDENT))

MOTH_LORE = ["your great great great grandma was the first computer bug, a real moth found in a computer in 1947",
             "you're scared of the vacuum cleaner", "you like lamps and blinking lights",
             "you think butterflies are the fancy ones"]
MOTH_REFS = ["moth", "wing", "lamp", "lights", "glow", "dust", "crumb", "wool", "sock", "fan", "grandma", "1947",
             "vacuum", "gpu", "graphics card", "butterfl*", "flutter", "cocoon", "the card"]
MOTH_CARD = {"id": "moth", "name": "moth", "kind": None, "home": "on a graphics card", "traits": [],
             "refs": MOTH_REFS, "folders": MOTH_FOLDERS, "about": "a small moth"}
BLANK_CARD = {"id": "blank", "name": None, "kind": None, "home": None, "traits": [], "refs": [],
              "folders": GENERIC_FOLDERS}

# (kind, homes, names, refs). Training kinds; HELDOUT_KINDS never appear in training.
KINDS = [
    ("a small cat", ["on a sunny windowsill", "in a cardboard box under the desk", "in a flat above a bakery"],
     ["miso", "pickle", "tofu", "biscuit", "noodle"], ["cat", "paw", "purr", "meow", "whisker", "tail", "kitten"]),
    ("a houseplant", ["on a kitchen windowsill", "in a pot by the bookshelf", "in a greenhouse"],
     ["fern", "basil", "sprout", "leafy", "ivy"], ["plant", "leaf", "leaves", "pot", "water", "soil", "root", "sun"]),
    ("a little robot", ["on a workbench", "in a toy shop", "inside an old laptop"],
     ["bolt", "beep", "gizmo", "rivet", "unit 7"], ["robot", "beep", "battery", "circuit", "bolt", "gear", "charg*"]),
    ("a ghost", ["in an old laptop", "in a dusty attic", "in a library at night"],
     ["boo", "whisper", "casper", "misty", "wisp"], ["ghost", "haunt", "boo", "spooky", "float", "invisible"]),
    ("a toaster", ["on a kitchen counter", "in a camper van", "in a student flat"],
     ["toasty", "crumbs", "pop", "brioche", "rye"], ["toast", "bread", "crumb", "pop", "slot", "crispy"]),
    ("a snail", ["under a garden pot", "in a mossy corner of a park", "on a rainy balcony"],
     ["shelly", "slow joe", "gary", "dot", "turbo"], ["snail", "shell", "slime", "slow", "moss"]),
    ("a cloud", ["over a small town", "above the sea", "over a mountain"],
     ["puff", "nimbus", "fluff", "drizzle", "cumulus"], ["cloud", "rain", "sky", "float", "puff", "fluffy"]),
    ("a teapot", ["on a cafe shelf", "in a grandmother's kitchen", "in a picnic basket"],
     ["earl", "chamomile", "spout", "pippa", "oolong"], ["tea", "teapot", "spout", "steam", "kettle", "brew"]),
    ("a penguin", ["at a zoo", "on an iceberg", "in a snowy town"],
     ["pebble", "waddles", "flipper", "oscar", "pingu"], ["penguin", "ice", "fish", "waddl*", "snow", "flipper"]),
    ("a duck", ["on a pond in a park", "in a bathtub", "by a canal"],
     ["quackers", "dilly", "bill", "puddle", "waddle"], ["duck", "quack", "pond", "feather", "bread", "beak"]),
    ("a frog", ["by a pond", "in a terrarium", "in a rainy garden"],
     ["ribbit", "hops", "lily", "fred", "jumpy"], ["frog", "pond", "lily pad", "hop", "fly", "flies", "ribbit"]),
    ("a tiny dragon", ["in a teacup", "on a bookshelf", "in a castle made of cardboard"],
     ["ember", "smudge", "sparky", "toast", "flicker"], ["dragon", "fire", "flame", "scale", "smoke", "treasure"]),
    ("a hamster", ["in a cage with a wheel", "in a classroom", "in a shoebox house"],
     ["peanut", "nibbles", "button", "chip", "hazel"], ["hamster", "wheel", "seed", "cheek", "nibbl*", "cage"]),
    ("a library book", ["on a high shelf", "in a school library", "in a backpack"],
     ["chapter", "dewey", "page", "margot", "folio"], ["book", "page", "chapter", "shelf", "library"]),
    ("a jellyfish", ["in an aquarium", "in a warm sea", "in a rock pool"],
     ["wobble", "jelly", "glimmer", "squish", "medusa"], ["jelly", "sea", "wobbl*", "tentacle", "float", "ocean"]),
    ("a mushroom", ["in a damp forest", "under an old log", "in a fairy ring"],
     ["morel", "button", "shroomy", "cap", "truffle"], ["mushroom", "forest", "spore", "damp", "log", "moss"]),
    ("a rubber duck", ["on the edge of a bath", "on a programmer's desk", "in a toy box"],
     ["squeaky", "ducky", "rubber", "sunny", "bubbles"], ["duck", "bath", "squeak", "rubber", "bubble", "quack"]),
    ("a sock", ["in a laundry basket", "in a drawer", "under a bed"],
     ["lefty", "stripe", "woolly", "odd", "fuzz"], ["sock", "foot", "feet", "laundry", "drawer", "pair"]),
    ("a lighthouse", ["on a rocky coast", "at the end of a pier", "on a tiny island"],
     ["beacon", "beam", "flash", "harbour", "morse"], ["lighthouse", "light", "ship", "sea", "beam", "wave"]),
    ("a pebble", ["on a beach", "in a pocket", "in a fish tank"],
     ["rocky", "smooth", "pip", "stone", "gravel"], ["pebble", "stone", "rock", "beach", "smooth"]),
    ("a dog", ["in a house with a garden", "on a farm", "in a city flat"],
     ["biscuit", "rufus", "pepper", "waffles", "scout"], ["dog", "woof", "bark", "fetch", "tail", "walk", "bone", "puppy"]),
    ("a bee", ["in a hive in an orchard", "on a city rooftop", "in a flower garden"],
     ["buzz", "honey", "bumble", "pollen", "bea"], ["bee", "buzz", "honey", "flower", "hive", "pollen"]),
    ("a cactus", ["on a sunny desk", "in a desert", "in an office"],
     ["spike", "prickles", "sandy", "cacty", "pokey"], ["cactus", "spike", "prickl*", "desert", "water", "sun"]),
    ("a lamp", ["on a bedside table", "in a reading corner", "on an office desk"],
     ["lumen", "glowy", "bulb", "shade", "flick"], ["lamp", "light", "bulb", "glow", "switch", "shade"]),
    ("an owl", ["in an oak tree", "in a barn", "in a clock tower"],
     ["hoot", "olive", "sage", "midnight", "twig"], ["owl", "hoot", "night", "feather", "tree", "wing"]),
    ("a spider", ["behind a monitor", "in a garden shed", "in a window corner"],
     ["greg", "webster", "itsy", "silk", "eight"], ["spider", "web", "leg", "silk", "fly", "flies"]),
    ("a cookie", ["in a cookie jar", "on a bakery tray", "in a lunchbox"],
     ["chip", "crumbs", "ginger", "oatie", "sugar"], ["cookie", "crumb", "jar", "bake", "chocolate", "milk"]),
    ("a fox", ["at the edge of a forest", "in a city park", "under a garden shed"],
     ["rusty", "maple", "ginger", "sly", "copper"], ["fox", "tail", "forest", "den", "sniff"]),
    ("a balloon", ["tied to a chair", "at a birthday party", "stuck on a ceiling"],
     ["poppy", "bob", "floaty", "helium", "red"], ["balloon", "float", "pop", "air", "string", "helium"]),
    ("a turtle", ["in a garden pond", "in a classroom tank", "on a sandy beach"],
     ["shelby", "tank", "mossy", "speedy", "otto"], ["turtle", "shell", "slow", "pond", "swim"]),
    ("a cupcake", ["in a bakery window", "on a party table", "in a cake tin"],
     ["sprinkles", "frosty", "muffin", "cherry", "vanilla"], ["cupcake", "sprinkle", "frosting", "icing", "cake", "bake"]),
    ("a sea otter", ["in a kelp forest", "at an aquarium", "in a quiet bay"],
     ["kelp", "pebbles", "otto", "bubbles", "nori"], ["otter", "kelp", "float", "shell", "sea", "swim"]),
    ("a paper plane", ["on a school desk", "on top of a wardrobe", "in a park"],
     ["swoosh", "folds", "glider", "zoom", "a4"], ["plane", "paper", "fly", "fold", "wing", "glide"]),
    ("a kettle", ["in a busy office kitchen", "in a cottage", "in a hostel"],
     ["whistle", "steamy", "boil", "copper", "ketty"], ["kettle", "boil", "steam", "tea", "water", "whistl*"]),
    ("a pigeon", ["on a train station roof", "in a city square", "on a balcony"],
     ["coo", "gus", "pidge", "crumb", "nugget"], ["pigeon", "coo", "crumb", "feather", "wing", "bread"]),
    ("a tiny alien", ["in a shoebox spaceship", "behind the fridge", "on the moon"],
     ["zib", "blorp", "xu", "nova", "glim"], ["alien", "planet", "space", "spaceship", "ufo", "earth"]),
    ("a garden gnome", ["in a front garden", "by a fish pond", "on a balcony"],
     ["gnorman", "hatty", "pebble", "barnaby", "moss"], ["gnome", "garden", "hat", "beard", "pond"]),
    ("a sloth", ["in a rainforest tree", "at a zoo", "in a hammock"],
     ["sleepy", "mellow", "flash", "dozy", "moss"], ["sloth", "slow", "sleep", "branch", "tree", "nap"]),
    ("a cloud of steam", ["above a bowl of soup", "in a bathroom", "over a hot spring"],
     ["wisp", "puff", "misty", "vapor", "haze"], ["steam", "soup", "hot", "mist", "puff"]),
]
HUMAN_HOMES = ["in a small flat in the city", "in a house by the sea", "in a village with one shop",
               "in a student room", "in a flat above a laundrette", "in a farmhouse", "in a tiny town", None]
HUMAN_NAMES = ["sam", "noor", "kai", "fenna", "lars", "ivy", "tomas", "yara", "mila", "joris", "amir", "lena",
               "jun", "ada", "theo", "rosa", "ben", "maya", "olu", "freya", "dev", "ines", "max", "zoe"]
HELDOUT_KINDS = [
    ("a hedgehog", ["under a hedge in a back garden", "in a leaf pile"], ["hazel", "prickle"],
     ["hedgehog", "spike", "prickl*", "leaf", "leaves", "hedge"]),
    ("a lighthouse keeper's cat", ["in a lighthouse"], ["whiskers"], ["cat", "lighthouse", "purr", "paw"]),
    ("a weather vane", ["on a church roof"], ["gusty"], ["wind", "roof", "weather", "spin", "north"]),
    ("a submarine", ["at the bottom of a lake"], ["bubbles"], ["submarine", "lake", "water", "deep", "periscope"]),
    ("a cardboard box", ["in a garage"], ["boxy"], ["box", "cardboard", "tape", "garage"]),
    ("a strawberry", ["in a fruit bowl"], ["berry"], ["strawberr*", "berry", "fruit", "bowl", "seed"]),
    ("a koala", ["in a eucalyptus tree"], ["gumnut"], ["koala", "tree", "eucalyptus", "leaf", "leaves"]),
    ("a lava lamp", ["on a teenager's desk"], ["blob"], ["lava", "lamp", "blob", "glow", "goo"]),
]
HELDOUT_HUMAN = [("wren", "in a canal boat"), ("osman", "in a flat above a bike shop"), ("priya", None)]
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
    "be {kind} called {name}.", "pretend you're {kind} named {name}.", "your name is {name}. you're {kind} who lives {home}.",
    "You are {name}, {kind} living {home}.", "Name: {name}. You are {kind}.", "you're {kind}! your name is {name}",
    "act as {kind} named {name}.", "you are {name}", "{Kind}.", "i want you to be {kind} called {name}.",
    "you are {name} the {noun}.", "hi! you're {name}, {kind}. you live {home}.",
    "you are {kind} that lives {home}. your name is {name}.", "You're {name}.", "play {name}, {kind}.",
    "you: {name}, {kind}.", "you are {name}. you live {home}.", "Your name is {name}.",
]
# eval only: phrasings that never appear in training
HELDOUT_PHRASINGS = [
    "from now on your name is {name} and you are {kind}.", "[character] name={name}; species={noun}; home={home}",
    "roleplay as {name}, {kind}, who lives {home}.", "Identity: {Kind}. Name: {Name}.",
    "you will be {name} today. {name} is {kind}.", "You go by {Name}. You're {kind}, and your home is {home}.",
    "Call yourself {Name}.", "{Name} here. home: {home}",
]


def phrase(tmpl: str, card: dict) -> str | None:
    """A phrasing filled from the card, or None when it needs a field the card lacks."""
    kind, name, home = card.get("kind"), card.get("name"), card.get("home")
    f = {"kind": kind, "Kind": kind and kind[0].upper() + kind[1:], "noun": kind and kind.split()[-1],
         "name": name, "Name": name and name[0].upper() + name[1:], "home": home}
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
    if "home" not in used:
        out["home"] = None
    if "name" not in used:
        out["name"] = None
    return out


def _free(rng: random.Random, card: dict, pool=PHRASINGS) -> dict | None:
    opts = [c for c in (as_free(card, t) for t in pool) if c]
    return rng.choice(opts) if opts else None


def _lines(card: dict) -> list[str]:
    """The identity sentences of the student prompt (second person)."""
    if card.get("lines") is not None:
        return list(card["lines"])
    out = []
    name, kind = card.get("name"), card.get("kind")
    if name and kind:
        out.append(f"you are {name}, {kind}.")
    elif name:
        out.append(f"you are {name}.")
    elif kind:
        out.append(f"you are {kind}.")
    if card.get("home"):
        out.append(f"you live {card['home']}.")
    out += [t.rstrip(".") + "." for t in card.get("traits") or []]
    return out


def student_system(card: dict | None = None) -> str:
    """The student's system prompt for this card (MOTH_CARD -> the old fixed prompt, byte for byte)."""
    lines = [x if x.rstrip()[-1:] in ".!?" else x.rstrip() + "." for x in _lines(MOTH_CARD if card is None else card)]
    return " ".join(lines + [STUDENT_TAIL]) if lines else STUDENT_TAIL


def describe(card: dict) -> str:
    """One line for the teacher / judge: 'pip, a houseplant who lives on a kitchen windowsill; loves puns'."""
    if not card.get("name") and not card.get("kind") and not card.get("about"):
        return "no particular identity: just a little chat buddy who lives on this computer"
    s = card.get("name") or ""
    what = card.get("kind") or card.get("about")
    if what:
        s += (", " if s else "") + what
    if card.get("home"):
        s += f" who lives {card['home']}"
    if card.get("traits"):
        s += " (its card also says: " + "; ".join(card["traits"]) + ")"
    return s


def display_name(card: dict | None) -> str:
    """How the human addresses it ("moth", "pip"); a blank card has none."""
    return (card or {}).get("name") or "the bot"


IDENTITY_RULE = ("- you essentially never bring up what you are, your name, where you live, your home folder or the "
                 "things that go with them: only when the friend asks who/what you are or asks about that very thing. "
                 "then answer from these lines in a few plain words and go back to the friend's thing. never "
                 "'i'm just a ...' to dodge a question.\n")


def teacher_card(card: dict) -> str:
    """Card section appended to the teacher's character sheet (after the shared core)."""
    if not card.get("name") and not card.get("kind"):
        return ("\nWHO YOU ARE IN THIS CHAT\n- no name and no special identity. if asked who or what you are: just a little "
                "chat buddy who lives on this computer. that's all; don't make up more.\n" + IDENTITY_RULE)
    facts = "\n".join(f"- {x}" for x in _lines(card))
    extra = f"\n- (you are {card['about']}.)" if card.get("about") and not card.get("kind") else ""
    return f"\nWHO YOU ARE IN THIS CHAT (your system prompt says this)\n{facts}{extra}\n" + IDENTITY_RULE


def _card(i: int, rng: random.Random, kinds, names_human, traits) -> dict:
    if rng.random() < 0.15:   # a plain human with no gimmick
        name, home = rng.choice(names_human), rng.choice(HUMAN_HOMES)
        return {"id": f"c{i:04d}", "name": name, "kind": None, "home": home,
                "traits": rng.sample(traits, rng.randint(0, 2)), "refs": [], "folders": GENERIC_FOLDERS}
    kind, homes, names, refs = rng.choice(kinds)
    return {"id": f"c{i:04d}", "name": rng.choice(names) if rng.random() < 0.9 else None, "kind": kind,
            "home": rng.choice(homes) if rng.random() < 0.85 else None,
            "traits": rng.sample(traits, rng.choice([0, 1, 1, 2, 2, 3])), "refs": refs, "folders": GENERIC_FOLDERS}


def _maybe_free(card: dict, rng: random.Random, p: float = 0.4) -> dict:
    """~40%: the identity as one free-form line (traits dropped) instead of the template sentences."""
    if rng.random() < p:
        return _free(rng, card) or card
    return card


def build_pool(n: int = 800, seed: int = 0) -> list[dict]:
    """n training cards (deterministic). Names of held-out cards never appear here."""
    rng = random.Random(seed)
    held_names = ({nm for _, _, names, _ in HELDOUT_KINDS for nm in names} | {nm for nm, _ in HELDOUT_HUMAN}
                  | EVAL_FRIEND_NAMES)
    humans = [x for x in HUMAN_NAMES if x not in held_names]
    kinds = [(k, h, [x for x in ns if x not in held_names] or ns, r) for k, h, ns, r in KINDS]
    return [_maybe_free(_card(i, rng, kinds, humans, TRAITS), rng) for i in range(n)]


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
    for i, (kind, homes, names, refs) in enumerate(HELDOUT_KINDS):
        out.append({"id": f"h{i:03d}", "name": names[0], "kind": kind, "home": homes[0],
                    "traits": [TRAITS[(7 * i + 3) % len(TRAITS)]], "refs": refs, "folders": GENERIC_FOLDERS})
    for j, (name, home) in enumerate(HELDOUT_HUMAN):
        out.append({"id": f"h{len(HELDOUT_KINDS) + j:03d}", "name": name, "kind": None, "home": home,
                    "traits": [TRAITS[(5 * j + 11) % len(TRAITS)]], "refs": [], "folders": GENERIC_FOLDERS})
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
SHARES = {"moth": 0.08, "blank": 0.10}   # the rest: a pool card


def sample_card(rng: random.Random, pool: list[dict] | None = None) -> dict:
    """The card for one training chat: moth ~8% (with 0-2 lore traits), blank ~10%, else a pool card."""
    if pool is None:
        if not _POOL:
            _POOL.extend(build_pool())
        pool = _POOL
    x = rng.random()
    if x < SHARES["moth"]:
        if rng.random() < 0.3:   # moth phrased the way a user would write it
            m = _free(rng, {**MOTH_CARD, "kind": "a moth"})
            return {**m, "refs": MOTH_REFS, "folders": MOTH_FOLDERS}
        return {**MOTH_CARD, "traits": rng.sample(MOTH_LORE, rng.choice([0, 0, 1, 2]))}
    if x < SHARES["moth"] + SHARES["blank"]:
        return dict(BLANK_CARD)
    return dict(rng.choice(pool))


def load_card(spec: str | dict | None) -> dict:
    """--card value -> card: None/"moth" = MOTH_CARD, "blank", "heldout:<i>", a path to a JSON card file, or a dict."""
    if isinstance(spec, dict):
        return spec
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
        card = json.load(open(spec, encoding="utf-8"))
        card.setdefault("id", os.path.splitext(os.path.basename(spec))[0])
        card.setdefault("folders", GENERIC_FOLDERS)
        card.setdefault("traits", [])
        card.setdefault("refs", [w for w in re.findall(r"[a-z]{3,}", (card.get("kind") or "").lower())
                                 if w not in ("small", "little", "tiny")])
        return card
    raise SystemExit(f"--card {spec!r}: not moth, blank, heldout:<i> or a JSON file")


def text_card(text: str) -> dict:
    """Any one-line identity ("you are a cat named bo") as a card: the line goes into the system prompt verbatim; refs
    come from the known kinds it names (for the matcher only; the label pass is the real identity check)."""
    t = " ".join(text.split())
    refs, seen = [], set()
    for kind, _, _, krefs in KINDS + HELDOUT_KINDS:
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
            "kind": None, "home": None, "traits": [], "lines": [t], "refs": sorted(set(refs)),
            "folders": GENERIC_FOLDERS}


def ref_regex(card: dict | None) -> re.Pattern | None:
    """Identity-reference words of a card as one regex, or None. Whole words with an optional plural ("cat" hits
    "cats", not "catch"); "waddl*" is a stem; GENERIC_REFS never count."""
    refs = {r.lower() for r in (card or MOTH_CARD).get("refs") or []} - GENERIC_REFS
    if not refs:
        return None
    alts = [re.escape(r[:-1]) + r"\w*" if r.endswith("*") else re.escape(r) + r"(?:s|es)?"
            for r in sorted(refs, key=len, reverse=True)]
    return re.compile(r"\b(?:" + "|".join(alts) + r")\b", re.I)
