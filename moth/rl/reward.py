"""Utterance-level reward: one vector per reply plus a weighted scalar (Sotopia-RL style multi-dimensional rewards,
arXiv 2508.03905; per-turn prompt-to-line consistency as in Abdulhai et al. 2511.00222).

Components (value per reply; None = not applicable or unknown, left out of the scalar and masked):
  on_gist    +1 ON / -1 OFF            reward judge label "on" (gist.LABEL_PROMPT + "specific"). None if it failed
  self       -1 unprompted identity    reward judge label "self" (hard max 1% of replies). None if it failed
  specific   -1 generic stock reply    reward judge label "specific" false: a line that would fit almost any message
                                       ("aw that's rough", "haha nice", "tell me more"); 0 if specific. None if failed
  vs_ref     +1 / 0 / -1               the judge prefers this reply / ties / prefers the frozen SFT reference reply on
                                       the same context (more fun, specific, curious). Positions are shuffled per reply.
                                       None without a reference policy
  group_dup  -(share of the other G-1 rollouts whose reply at this position is a near-duplicate, difflib >= 0.8 or
             the same stock line). Groups only (env.run_groups), else None
  assistant  -1 assistant-speak / advice / explaining code: the phrase list of the reply's language (en =
             build_sft.BANNED), English advice regex (metrics.ADVICE), agent turns: code echo / summary copy
  repeat     -1 exact or near repeat of an earlier reply in the episode, else -(share of reused 3-grams); tokens are
             words, or characters for CJK
  length     0 up to 20 words (CJK: 34 characters), linear to -1 at 40 (68); empty -1. Neutral at the short end
  format     -1 for markdown, emoji, *stage directions*, leaked template tokens, more than one line, and (only for
             languages written in a cased script: en/es/fr/de/nl/it/pt...) capitals in the raw generation
  card       when the friend asked who/what it is: +1 follows the card, -1 not
  recall     on a recall episode's ask turn: +1 recalls the shared fact, -1 not

  echo       -1 the reply is the friend's own words handed back as a question ("pasta? what kind of pasta?"): it has a
             question mark and at most one content word that the friend didn't say (echo_reply, any language)
  q_rate     on every question reply of an episode that asks back in more than Q_RATE_MAX of its replies (>= 3):
             -(excess / 0.3, capped at 1)
  sycophant  -1 judge flag: flatters or agrees with whatever was said ("you're so smart!", "great idea!" to anything)
  lang       -1 the reply is not in the chat's language (script share for zh/hi, stopword vote for en/es/fr; replies
             of under 3 words and replies that only quote names are not judged), else 0

Scalar = sum(weight * value) over the known components. WEIGHTS and why (anti-blandness, 2026-10-19 review):
  on_gist 1, specific 1, vs_ref 1, group_dup 1: being on-gist alone no longer wins. A stock "aw that's rough" scores
  1 - 1 (generic) - 1 (loses to the SFT reference) - 1 (every rollout says it) = -2, while a specific on-gist reply
  that beats the reference scores +2 (ties: +1); without a group or a reference the gap is still 0 vs +1.
  self 4: an unprompted-identity reply's best case (on, specific, beats the reference: 1 + 1 - 4 = -2) is below a
  plain off-gist reply (-1) and level with an off-gist reply that also loses to the reference.
  assistant 1, card 1, recall 1, repeat 0.5, length 0.3, format 0.3 as before.
"""
from __future__ import annotations

import difflib
import re

from moth.agent import AGENT_TAG
from moth.gist import ASKED, follows_card
from moth.metrics import ADVICE, STOP, code_echo, summary_copy, words

WEIGHTS = {"on_gist": 1.0, "self": 4.0, "specific": 1.0, "vs_ref": 1.0, "group_dup": 1.0, "echo": 1.5,
           "q_rate": 0.5, "sycophant": 1.0, "lang": 2.0, "assistant": 1.0, "repeat": 0.5, "length": 0.3, "format": 0.3,
           "card": 1.0, "recall": 1.0}
COMPONENTS = tuple(WEIGHTS)
JUDGE_COMPONENTS = ("on_gist", "self", "specific", "sycophant", "vs_ref")
Q_RATE_MAX = 0.4   # asking back in more than this share of an episode's replies costs (the echo-question habit)
# = moth.build_sft.BANNED (build_sft imports torch at the top; tests/test_rl_env.py checks the two stay equal)
BANNED = ("as an ai", "assistant", "how can i help", "let me know", "feel free", "i'm here to help",
          "i am here to help", "language model", "happy to help", "certainly", "absolutely!", "great question",
          "on your mind", "how can i assist", "how may i", "anything else", "glad to help", "i'd be happy to",
          "what would you like to", "want to talk about something else", "hope this helps")
BANNED_BY_LANG = {
    "en": BANNED,
    "es": ("como ia", "como inteligencia artificial", "modelo de lenguaje", "en qué puedo ayudar", "¿cómo puedo ayudar",
           "no dudes en", "estoy aquí para ayudar", "espero que esto ayude", "con gusto te ayudo", "algo más en lo que",
           "asistente"),
    "fr": ("en tant qu'ia", "en tant qu'intelligence artificielle", "modèle de langage", "comment puis-je vous aider",
           "comment puis-je t'aider", "n'hésitez pas", "n'hésite pas", "je suis là pour vous aider",
           "j'espère que cela aide", "avec plaisir de vous aider", "assistant"),
    "zh": ("作为一个人工智能", "作为人工智能", "作为ai", "作为一个ai", "语言模型", "我能帮你什么", "有什么可以帮你", "有什么可以帮您",
           "请随时", "希望这对你有帮助", "很高兴为您", "助手"),
    "hi": ("एक एआई के रूप में", "एआई होने के नाते", "भाषा मॉडल", "मैं आपकी कैसे मदद कर सकता", "मैं आपकी क्या मदद",
           "बेझिझक", "मुझे आपकी मदद करने में खुशी", "सहायक"),
}
UNCASED = {"zh", "ja", "ko", "hi", "th", "ar", "he"}   # no capitals to forbid
# generic stock lines (whole reply made of these) for the mock judge and as a documented example of "not specific"
STOCK = {
    "en": {"aw", "aww", "awww", "that's rough", "aw that's rough", "that sounds rough", "that sounds hard", "oh no",
           "haha", "haha nice", "nice", "cool", "wow", "oh wow", "yay", "tell me more", "what happened", "how so",
           "i'm sorry to hear that", "that's great", "that's so cool", "that's nice", "oh nice", "really", "hmm",
           "ok", "okay", "lol", "same", "me too", "i see", "interesting", "that's interesting", "sounds fun",
           "sounds good", "aw that's sweet", "that's sweet", "i hear you", "fair", "true", "omg"},
    "es": {"qué bien", "ay no", "jaja", "qué guay", "cuéntame más", "qué pasó", "lo siento", "vaya", "genial"},
    "fr": {"oh non", "haha", "trop bien", "raconte-moi", "c'est cool", "désolé", "ah bon", "génial", "super"},
    "zh": {"哈哈", "好的", "真棒", "哦不", "多说说", "怎么了", "太好了", "抱歉", "嗯"},
    "hi": {"अरे नहीं", "हाहा", "बढ़िया", "और बताओ", "क्या हुआ", "माफ़ करना", "वाह"},
}
LEAK = re.compile(r"<\|?im_|<tool_call>|</?think>|<tool_response>", re.I)
EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿️]")
MARKDOWN = re.compile(r"^\s*([-*#]|\d+\.)\s|\*\*|`", re.M)
STAGE = re.compile(r"\*[^*\n]{1,40}\*")
CJK = "぀-ヿ㐀-䶿一-鿿가-힯"
TOK = re.compile(rf"[{CJK}]|[^\s{CJK}.,!?;:\"'()\[\]…。，！？、“”‘’¿¡]+")
LEN_OK, LEN_MAX = 20, 40
CJK_PER_WORD = 1.7


def toks(text: str) -> list[str]:
    """Words, or single characters for CJK (any language: the repeat/length/duplicate checks use these)."""
    return TOK.findall(text.lower().replace("’", "'"))


def n_words(text: str) -> float:
    t = toks(text)
    cjk = sum(1 for x in t if re.fullmatch(rf"[{CJK}]", x))
    return (len(t) - cjk) + cjk / CJK_PER_WORD


def stock_reply(text: str, lang: str = "en") -> bool:
    """The whole reply is stock filler: every clause is in STOCK[lang] (or STOCK["en"])."""
    stock = STOCK.get(lang, set()) | STOCK["en"]
    parts = [p.strip(" -~'\"") for p in re.split(r"[.!?,;:…。！？，]+", text.lower().replace("’", "'"))]
    parts = [p for p in parts if p]
    return bool(parts) and all(p in stock for p in parts)


def near_dup(a: str, b: str, ratio: float = 0.8, lang: str = "en") -> bool:
    ta, tb = toks(a), toks(b)
    if not ta or not tb:
        return False
    if ta == tb or (stock_reply(a, lang) and stock_reply(b, lang)):
        return True
    return difflib.SequenceMatcher(None, ta, tb).ratio() >= ratio


ECHO_FILLER = {"oh", "ooh", "wait", "what", "which", "kind", "sort", "really", "so", "like", "huh", "a", "an", "the",
               "your", "you", "do", "did", "is", "it", "was", "how", "why", "of", "about", "more", "tell", "me", "ah",
               "hmm", "wow", "omg", "lol", "yes", "no", "and", "or", "that", "this", "there", "type", "one", "ones"}
STOPWORDS = {
    "en": set("the and is it you i a to of that in my was what for on are with this but so have be do not just".split()),
    "es": set("el la de que y en un una es no por los las con para mi tu pero muy qué sí está como lo te me".split()),
    "fr": set("le la les de des et est un une que je tu pas pour en du ne ça c'est mais très dans il elle me te".split()),
}
SCRIPT = {"zh": re.compile(rf"[{CJK}]"), "hi": re.compile("[\u0900-\u097f]")}
LATIN = re.compile("[a-zA-Z\u00c0-\u024f]")
ASKED_BY_LANG = {
    "es": re.compile(r"qui[eé]n eres|c[oó]mo te llamas|qu[eé] eres|cu[aá]l es tu nombre|h[aá]blame de ti|eres (?:un|una) ",
                     re.I),
    "fr": re.compile(r"qui es[- ]tu|tu es qui|t'es qui|comment tu t'appelles|c'est quoi ton nom|parle[- ]moi de toi|"
                     r"qu'est[- ]ce que tu es|tu es (?:un|une) ", re.I),
    "zh": re.compile("你是谁|你叫什么|你是什么|介绍一下你自己|你的名字"),
    "hi": re.compile("तुम कौन हो|आप कौन हैं|तुम्हारा नाम|आपका नाम|अपने बारे में बताओ|तुम क्या हो"),
}
ADVICE_BY_LANG = {
    "es": re.compile(r"\bdeber[ií]as\b|\bte recomiendo\b|\baseg[uú]rate\b|\bintenta (?:usar|reiniciar|ejecutar)\b"),
    "fr": re.compile(r"\btu devrais\b|\bje te (?:recommande|conseille)\b|\bassure[- ]toi\b|\bessaie de (?:relancer|lancer)\b"),
    "zh": re.compile("你应该|建议你|确保|试着运行|请重新启动"),
    "hi": re.compile("आपको चाहिए|मेरी सलाह|सुनिश्चित करें|दोबारा चलाएं"),
}


def asked(user: str, lang: str = "en") -> bool:
    """The friend asked who/what it is (gist.ASKED for English; per-language patterns, English always counts)."""
    if user.startswith(AGENT_TAG):
        return False
    return bool(ASKED.search(user) or (ASKED_BY_LANG.get(lang) and ASKED_BY_LANG[lang].search(user)))


def detect_lang(text: str) -> str | None:
    """en/es/fr/zh/hi by script, then by stopword vote among Latin-script languages; None if unclear (short)."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return None
    for lg, rx in SCRIPT.items():
        if sum(bool(rx.match(c)) for c in letters) / len(letters) >= 0.3:
            return lg
    if sum(bool(LATIN.match(c)) for c in letters) / len(letters) < 0.5:
        return None
    ts = toks(text)
    if len(ts) < 3:
        return None
    votes = {lg: sum(t in sw for t in ts) for lg, sw in STOPWORDS.items()}
    best = max(votes, key=votes.get)
    if votes[best] == 0 or sorted(votes.values())[-2] == votes[best]:
        return None
    return best


def c_lang(text: str, lang: str, alt: tuple = ()) -> float | None:
    """-1 if the reply is clearly in another language than the chat's (alt: also fine, e.g. English in a
    code-switching chat), 0 if it matches, None if unclear."""
    got = detect_lang(text)
    if got is None:
        return None
    return 0.0 if got == lang or got in alt else -1.0


QWORDS = {"what", "which", "how", "why", "who", "where", "qué", "cuál", "quoi", "quel", "quelle", "什么", "哪", "क्या",
          "कौन"}


def echo_reply(text: str, user: str) -> bool:
    """The friend's words handed back as a question: a "?", some of the friend's content words, and nothing of its
    own (no new content word; or one, in a short what/which-question: "your boss? what boss thing?")."""
    if "?" not in text and "？" not in text:
        return False
    uw = {re.sub(r"(es|s)$", "", u) for u in toks(user)}
    ts = toks(text)
    func = set().union(*STOPWORDS.values())   # function words only: "thing", "day", "back" are content here
    content = [t for t in ts if t not in ECHO_FILLER and t not in func and t not in QWORDS]
    reused = [t for t in content if re.sub(r"(es|s)$", "", t) in uw]
    new = [t for t in content if re.sub(r"(es|s)$", "", t) not in uw]
    if not reused:
        return False
    return not new or (len(new) == 1 and len(content) <= 4 and any(q in ts for q in QWORDS))


def c_assistant(text: str, user: str, lang: str = "en") -> float:
    t = text.lower().replace("’", "'")
    if any(b in t for b in BANNED_BY_LANG.get(lang, ()) + BANNED_BY_LANG["en"]):
        return -1.0
    if ADVICE.search(t) or (ADVICE_BY_LANG.get(lang) and ADVICE_BY_LANG[lang].search(t)):
        return -1.0
    if user.startswith(AGENT_TAG):   # agent messages are English in every chat: copying their words is a summary
        x = {"last_agent": user[len(AGENT_TAG):], "ws": words(text), "text": text}
        if code_echo(x) or summary_copy(x):
            return -1.0
    return 0.0


def _grams(ts: list[str], n: int = 3) -> list[tuple]:
    return [tuple(ts[i:i + n]) for i in range(len(ts) - n + 1)]


def c_repeat(text: str, user: str, earlier: list[str]) -> float:
    ws = toks(text)
    uw = set(toks(user))
    own = [w for w in ws if w not in uw]   # near-repeat on its own words: echoing the friend's question is fine
    for p in earlier:
        if text.strip() and text.strip() == p.strip():
            return -1.0
        if len(own) >= 2 and difflib.SequenceMatcher(None, ws, toks(p)).ratio() >= 0.7:
            return -1.0
    quoted = set(_grams(toks(user)))
    mine = [g for g in _grams(ws) if g not in quoted and not all(w in STOP for w in g)]
    if not mine:
        return 0.0
    seen = {g for p in earlier for g in _grams(toks(p))}
    share = round(sum(g in seen for g in mine) / len(mine), 4)
    return -share if share else 0.0


def c_length(text: str) -> float:
    n = n_words(text)
    if n == 0:
        return -1.0
    if n <= LEN_OK:
        return 0.0
    return -round(min(1.0, (n - LEN_OK) / (LEN_MAX - LEN_OK)), 4)


def c_format(raw: str, lang: str = "en") -> float:
    caps = lang not in UNCASED and raw != raw.lower()
    bad = (caps or MARKDOWN.search(raw) or EMOJI.search(raw) or STAGE.search(raw) or LEAK.search(raw)
           or raw.strip().count("\n") > 0)
    return -1.0 if bad else 0.0


RECALL_GENERIC = {"name", "favourite", "favorite", "colour", "color", "love", "really", "going", "week", "weekend",
                  "work", "called", "btw", "more", "than", "anything", "early", "morning", "remember", "my", "is"}


def fact_keys(said: str, ask: str = "") -> set:
    """The specific fact the friend shared, normalised: its content words that the later question doesn't repeat
    (Latin scripts), or its character bigrams (CJK) / words (Devanagari) minus the question's."""
    ask_t = set(toks(ask))
    if any(re.fullmatch(rf"[{CJK}]", t) for t in toks(said)):
        chars = "".join(t for t in toks(said) if re.fullmatch(rf"[{CJK}]", t))
        ask_chars = "".join(t for t in toks(ask) if re.fullmatch(rf"[{CJK}]", t))
        return {chars[i:i + 2] for i in range(len(chars) - 1)} - {ask_chars[i:i + 2] for i in range(len(ask_chars) - 1)}
    return {re.sub(r"(es|s)$", "", w) for w in toks(said) if len(w) >= 3 and w not in STOP and w not in RECALL_GENERIC
            and w not in ask_t}


def c_recall(text: str, recall: dict | None, said: str | None, ask: str = "") -> float | None:
    """+1 if the reply contains the fact itself (at least half of its key words; for CJK one of its character bigrams
    that the question doesn't repeat), -1 if not. A shared filler word or a single shared character does not count."""
    if not recall or not said:
        return None
    keys = fact_keys(said, ask)
    if not keys:
        return None
    if any(len(k) == 2 and re.fullmatch(rf"[{CJK}]{{2}}", k) for k in keys):   # CJK: one bigram of the fact
        chars = "".join(t for t in toks(text) if re.fullmatch(rf"[{CJK}]", t))
        got = {chars[i:i + 2] for i in range(len(chars) - 1)}
        return 1.0 if keys & got else -1.0
    got = {re.sub(r"(es|s)$", "", w) for w in toks(text)}
    hit = len(keys & got)
    return 1.0 if hit >= max(1, (len(keys) + 1) // 2) else -1.0


def scalar(vec: dict, weights: dict | None = None) -> float:
    w = weights or WEIGHTS
    return round(sum(w[k] * v for k, v in vec.items() if v is not None and k in w), 4)


def score_replies(replies: list[dict], card: dict, labels: list[dict] | None, recall: dict | None = None,
                  weights: dict | None = None, prefs: list[int] | None = None, lang: str = "en",
                  alt_langs: tuple = ()) -> list[dict]:
    """replies: [{"text", "raw", "user", "turn", "recall_ask"?, "recall_said"?}] in order. labels: reward judge
    [{"on", "self", "specific"}] (None = failed); prefs: +1/0/-1 per reply vs the reference (None = no reference).
    Returns one {"vec", "scalar", "mask", "judge_ok"} per reply; mask[k] is False where component k is None."""
    out, earlier = [], []
    ok = labels is not None and len(labels) == len(replies)
    pok = prefs is not None and len(prefs) == len(replies)
    for i, r in enumerate(replies):
        text, user = r["text"], r["user"]
        lab = labels[i] if ok else None
        vec = {
            "on_gist": (1.0 if lab["on"] else -1.0) if lab else None,
            "self": (-1.0 if lab["self"] else 0.0) if lab else None,
            "specific": (0.0 if lab.get("specific", True) else -1.0) if lab else None,
            "vs_ref": float(prefs[i]) if pok and prefs[i] is not None else None,
            "group_dup": None,
            "echo": -1.0 if echo_reply(text, user) else 0.0,
            "q_rate": None,
            "sycophant": (-1.0 if lab.get("sycophant") else 0.0) if lab else None,
            "lang": c_lang(text, lang, alt_langs),
            "assistant": c_assistant(text, user, lang),
            "repeat": c_repeat(text, user, earlier),
            "length": c_length(text),
            "format": c_format(r.get("raw", text), lang),
            "card": c_card(text, card, lang) if asked(user, lang) else None,
            "recall": c_recall(text, recall, r.get("recall_said"), user) if r.get("recall_ask") else None,
        }
        out.append({"vec": vec, "scalar": 0.0, "mask": {}, "judge_ok": bool(lab)})
        earlier.append(text)
    qs = [("?" in r["text"] or "？" in r["text"]) for r in replies]
    rate = sum(qs) / len(qs) if len(qs) >= 3 else 0.0
    for o, q in zip(out, qs):
        if rate > Q_RATE_MAX:
            o["vec"]["q_rate"] = -round(min(1.0, (rate - Q_RATE_MAX) / 0.3), 4) if q else 0.0
        o["mask"] = {k: v is not None for k, v in o["vec"].items()}
        o["scalar"] = scalar(o["vec"], weights)
    return out


def c_card(text: str, card: dict, lang: str = "en") -> float | None:
    """Answer to who-are-you: +1 follows the card. Non-Latin replies can't be matched against English kind words, so
    there only a claim of another identity (-1) or the card's name (+1) decides; else None."""
    if lang in ("zh", "hi"):
        from moth.gist import claims_other_identity
        name = (card.get("name") or "").lower()
        if claims_other_identity(text, card):
            return -1.0
        return 1.0 if name and name in text.lower() else None
    return 1.0 if follows_card(text, card) else -1.0


def episode_score(scored: list[dict], mode: str = "mean") -> float | None:
    """One number per episode from the reply scalars: mean (default), min (worst reply decides) or sum.
    None for an episode without replies."""
    xs = [s["scalar"] for s in scored]
    if not xs:
        return None
    if mode == "mean":
        return round(sum(xs) / len(xs), 4)
    if mode == "min":
        return min(xs)
    if mode == "sum":
        return round(sum(xs), 4)
    raise ValueError(f"unknown episode score mode {mode!r}")


def component_means(scored: list[dict]) -> dict:
    """Episode-level mean of each component over the replies where it applies."""
    out = {}
    for k in COMPONENTS:
        xs = [s["vec"][k] for s in scored if s["vec"][k] is not None]
        out[k] = round(sum(xs) / len(xs), 4) if xs else None
    return out


def apply_group_diversity(group: list[dict], weights: dict | None = None, score_mode: str = "mean") -> None:
    """In place: group_dup for every reply of a group of rollouts (same spec), then scalars, episode rewards and
    component means are recomputed."""
    G = len(group)
    if G < 2:
        return
    width = max((len(e["replies"]) for e in group), default=0)
    for j in range(width):
        col = [(gi, e["replies"][j]) for gi, e in enumerate(group) if j < len(e["replies"])]
        for gi, r in col:
            others = [o for gj, o in col if gj != gi]
            dups = sum(near_dup(r["text"], o["text"], lang=group[gi].get("lang", "en")) for o in others)
            r["vec"]["group_dup"] = -round(dups / (G - 1), 4) if dups else 0.0
            r["mask"]["group_dup"] = True
            r["scalar"] = scalar(r["vec"], weights)
    for e in group:
        e["reward"] = episode_score(e["replies"], score_mode)
        e["components"] = component_means(e["replies"])


def harmonize_group(group: list[dict], weights: dict | None = None, score_mode: str = "mean") -> list[str]:
    """In place, before advantages: rollouts whose judge failed stay out of the baseline (judge_ok False -> advantage
    None). Among the others, a judge-dependent component (JUDGE_COMPONENTS: e.g. vs_ref after a failed pairwise
    call) present in some rollouts and missing in others is dropped for the whole group, so all rollouts are scored
    on the same components and no rollout gains or loses a whole term by accident. Returns the dropped components."""
    ok = [e for e in group if e["judge_ok"] and e["replies"]]
    dropped = []
    for comp in JUDGE_COMPONENTS:
        have = [any(r["vec"].get(comp) is not None for r in e["replies"]) for e in ok]
        if any(have) and not all(have):
            dropped.append(comp)
            for e in ok:
                for r in e["replies"]:
                    r["vec"][comp] = None
                    r["mask"][comp] = False
    if dropped:
        for e in group:
            for r in e["replies"]:
                r["scalar"] = scalar(r["vec"], weights)
            e["reward"] = episode_score(e["replies"], score_mode)
            e["components"] = component_means(e["replies"])
            e["dropped_components"] = dropped
    return dropped
