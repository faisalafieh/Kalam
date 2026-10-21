"""
Arabic and Arabic-dialect support for Kalam.

Whisper has one "ar" language token for every variety of Arabic. Left alone it
leans towards Modern Standard Arabic spelling and vocabulary, which hurts on
dialect speech: dialect words get "corrected" into MSA ones or dropped. Two
levers help without retraining:

1. A short primer written in the target dialect, passed to Whisper as
   `hotwords`. faster-whisper puts hotwords in front of EVERY 30 s window (unlike
   `initial_prompt`, which only reaches the first window when
   condition_on_previous_text is off), so the whole recording is nudged towards
   that dialect's spelling and vocabulary.
2. Picking the dialect automatically when the caller does not know it: a quick
   probe transcription of the opening seconds, scored against marker words that
   are common in one dialect and rare in the others.

The marker scoring is a heuristic, not a trained dialect classifier. It only
commits when the evidence is clear and otherwise falls back to MSA, so a wrong
guess costs no more than having no primer at all. Measure any of this on
labelled recordings (eval/evaluate.py) before quoting accuracy per dialect.
"""

from __future__ import annotations

import re

DIALECTS = ("msa", "gulf", "levantine", "egyptian", "maghrebi")

# One or two everyday sentences per dialect, in that dialect's usual spelling.
# Kept short: hotwords are capped at about 220 tokens and Arabic tokenises densely.
PRIMERS = {
    "msa": "افتتحت الجلسة. تفضل يا أستاذ، المحكمة تستمع إليك. نعم سيدي القاضي.",
    "gulf": "هلا والله، شلونك؟ زين الحمد لله. الحين بنبدأ، عندنا وايد أشياء نبي نناقشها.",
    "levantine": "مرحبا، كيفك؟ منيح الحمد لله. هلق رح نبلش، في كتير شغلات بدنا نحكي فيها.",
    "egyptian": "أهلا، إزيك؟ كويس الحمد لله. دلوقتي هنبدأ، فيه حاجات كتير عايزين نتكلم فيها.",
    "maghrebi": "السلام، لاباس عليك؟ بخير الحمد لله. دابا غادي نبداو، كاين بزاف ديال الحوايج بغينا نهضرو عليها.",
}

# Words that are frequent in one dialect and rare elsewhere. Deliberately leaves
# out shared words (شو, كتير, مش, عشان, ليش) that would blur the scores.
MARKERS = {
    "gulf": {
        "شلون", "وايد", "الحين", "ابغى", "أبغى", "تبغى", "يبغى", "نبغى",
        "مب", "جذي", "چذي", "هني", "شفيك", "يالله", "اشوى",
    },
    "levantine": {
        "هلق", "هلأ", "منيح", "منيحة", "بدي", "بدك", "بدنا", "بدهم",
        "هيك", "لهيك", "هاد", "هيدا", "هيدي", "كيفكن",
    },
    "egyptian": {
        "دلوقتي", "دلوقت", "عايز", "عايزة", "عاوز", "عاوزة", "ازاي", "إزاي",
        "ازيك", "إزيك", "كده", "كدا", "بتاع", "بتاعة", "بتاعت", "أوي", "اوي",
    },
    "maghrebi": {
        "دابا", "بزاف", "واش", "كيفاش", "بغيت", "بغيتي", "بغينا", "ديال",
        "ديالي", "ديالك", "غادي", "كاين", "كاينة", "شحال", "علاش", "لاباس", "مزيان",
    },
}
MIN_MARKERS = 2        # fewer hits than this is not evidence of anything
MIN_MARGIN = 1.5       # winner must beat the runner-up by this factor

_ARABIC = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿ]")
_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
_TATWEEL = "ـ"
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def has_arabic(text: str) -> bool:
    return bool(_ARABIC.search(text))


def _tokens(text: str) -> list[str]:
    text = _DIACRITICS.sub("", text).replace(_TATWEEL, "")
    words = re.findall(r"[؀-ۿ]+", text)
    out = []
    for w in words:
        out.append(w)
        # attached conjunctions: "ودابا" -> also count "دابا"
        if len(w) > 3 and w[0] in "وف":
            out.append(w[1:])
    return out


def dialect_scores(text: str) -> dict[str, int]:
    toks = _tokens(text)
    return {d: sum(t in words for t in toks) for d, words in MARKERS.items()}


def detect_dialect(text: str) -> tuple[str, dict[str, int]]:
    """Best-guess dialect of an Arabic transcript, or "msa" when unclear."""
    scores = dialect_scores(text)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    (best, top), (_, second) = ranked[0], ranked[1]
    if top >= MIN_MARKERS and top >= MIN_MARGIN * second:
        return best, scores
    return "msa", scores


def normalize_arabic(text: str) -> str:
    """Normalisation used before scoring Arabic WER.

    The usual convention for Arabic ASR evaluation: drop diacritics and tatweel,
    unify alef forms, alef maqsura and ta marbuta, map Arabic-Indic digits to
    0-9, strip punctuation (Arabic and Latin) and lowercase any Latin words, so
    spelling variants that a human would read as the same word are not errors.
    """
    text = _DIACRITICS.sub("", text).replace(_TATWEEL, "")
    text = re.sub("[إأآٱ]", "ا", text)
    text = text.replace("ى", "ي").replace("ة", "ه")
    text = text.translate(_DIGITS).lower()
    text = re.sub(r"[^\w\s]|_", " ", text)
    return " ".join(text.split())
