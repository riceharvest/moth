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

WEIGHTS = {"on_gist": 1.0, "self": 4.0, "specific": 1.0, "vs_ref": 1.0, "group_dup": 1.0, "assistant": 1.0,
           "repeat": 0.5, "length": 0.3, "format": 0.3, "card": 1.0, "recall": 1.0}
COMPONENTS = tuple(WEIGHTS)
JUDGE_COMPONENTS = ("on_gist", "self", "specific", "vs_ref")
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


def c_assistant(text: str, user: str, lang: str = "en") -> float:
    t = text.lower().replace("’", "'")
    if any(b in t for b in BANNED_BY_LANG.get(lang, ()) + BANNED_BY_LANG["en"]):
        return -1.0
    if lang == "en" and ADVICE.search(t):
        return -1.0
    if user.startswith(AGENT_TAG):
        x = {"last_agent": user[len(AGENT_TAG):], "ws": words(text), "text": text}
        if code_echo(x) or (lang == "en" and summary_copy(x)):
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


def c_recall(text: str, recall: dict | None, said: str | None) -> float | None:
    if not recall or not said:
        return None
    fact = {re.sub(r"(es|s)$", "", w) for w in toks(said) if len(w) >= 3 and w not in STOP} - {
        "name", "favourite", "favorite", "colour", "color", "love", "really", "going", "week", "weekend", "work",
        "called", "btw", "more", "than", "anything", "early", "morning"}
    if not fact:   # CJK: single characters
        fact = {w for w in toks(said) if re.fullmatch(rf"[{CJK}]", w)}
    got = {re.sub(r"(es|s)$", "", w) for w in toks(text)}
    return 1.0 if fact & got else -1.0


def scalar(vec: dict, weights: dict | None = None) -> float:
    w = weights or WEIGHTS
    return round(sum(w[k] * v for k, v in vec.items() if v is not None and k in w), 4)


def score_replies(replies: list[dict], card: dict, labels: list[dict] | None, recall: dict | None = None,
                  weights: dict | None = None, prefs: list[int] | None = None, lang: str = "en") -> list[dict]:
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
            "assistant": c_assistant(text, user, lang),
            "repeat": c_repeat(text, user, earlier),
            "length": c_length(text),
            "format": c_format(r.get("raw", text), lang),
            "card": ((1.0 if follows_card(text, card) else -1.0)
                     if ASKED.search(user) and not user.startswith(AGENT_TAG) else None),
            "recall": c_recall(text, recall, r.get("recall_said")) if r.get("recall_ask") else None,
        }
        out.append({"vec": vec, "scalar": scalar(vec, weights), "mask": {k: v is not None for k, v in vec.items()},
                    "judge_ok": bool(lab)})
        earlier.append(text)
    return out


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
