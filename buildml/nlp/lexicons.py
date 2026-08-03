"""Built-in linguistic resources for the core (dependency-free) NLP path.

Everything here is static data shipped with BuildML so the default NLP backends
work without optional extras and without downloading corpora at runtime:

* ``STOPWORDS`` — function-word lists per supported language.
* ``SENTIMENT_LEXICON`` — signed valence weights for rule-based sentiment.
* ``NEGATORS`` / ``INTENSIFIERS`` / ``EMOTICONS`` — rule modifiers.
* ``SCRIPT_RANGES`` — Unicode block probes for non-Latin script detection.
* ``SUFFIX_STEM_RULES`` — conservative English suffix-stripping rules.

Honesty: these lists are compact and English-centred. Wider coverage is an
opt-in extra (``buildml[nlp]`` for NLTK/langdetect, ``buildml[nlp-industry]``
for spaCy). The capability matrix reports which resource is actually in use.
"""

from __future__ import annotations

ENGLISH_STOPWORDS: frozenset[str] = frozenset(
    """
a about above after again against all am an and any are aren't as at be because
been before being below between both but by can cannot could couldn't did didn't
do does doesn't doing don't down during each few for from further had hadn't has
hasn't have haven't having he her here hers herself him himself his how i if in
into is isn't it its itself just me more most mustn't my myself no nor not of off
on once only or other ought our ours ourselves out over own same shan't she
should shouldn't so some such than that the their theirs them themselves then
there these they this those through to too under until up very was wasn't we
were weren't what when where which while who whom why with won't would wouldn't
you your yours yourself yourselves
""".split()
)

SPANISH_STOPWORDS: frozenset[str] = frozenset(
    """
a al algo algunas algunos ante antes como con contra cual cuando de del desde
donde dos el ella ellas ellos en entre era eran es esa ese eso esta estaba este
esto ha hasta hay la las le les lo los mas me mi mucho muy nada ni no nos o os
otra otro para pero poco por porque que quien se sea si sin sobre son su sus
tambien tanto te tiene todo todos tu un una uno y ya yo
""".split()
)

FRENCH_STOPWORDS: frozenset[str] = frozenset(
    """
a au aux avec ce ces dans de des du elle en et eux il ils je la le les leur lui
ma mais me meme mes moi mon ne nos notre nous on ou par pas pour qu que qui sa se
ses son sur ta te tes toi ton tu un une vos votre vous y etre avoir plus tres
comme cette celui donc
""".split()
)

GERMAN_STOPWORDS: frozenset[str] = frozenset(
    """
aber alle als am an auch auf aus bei bin bis bist da dass dein den der des dem
die das denn dir du er es euer eure fur gegen hab habe haben hat hatte hier ich
ihr im in ist ja jede jedem kann kein man mein mit nach nicht noch nun oder sein
seine sich sie sind so soll uber um und uns unser vom von vor war waren was
weil wenn wer wie wir wird zu zum zur
""".split()
)

ITALIAN_STOPWORDS: frozenset[str] = frozenset(
    """
a agli ai al alla alle allo anche che chi ci coi col come con da dai dal dalla
degli dei del della delle dello di due e ed gli ha hanno ho i il in io la le lo
ma mi ne nei nel nella no non o per piu quale quando quello questo se si sono su
sua sue sui sul sulla suo te tra tu un una uno vi
""".split()
)

PORTUGUESE_STOPWORDS: frozenset[str] = frozenset(
    """
a ao aos as com como da das de do dos e ela ele eles em entre era eram essa esse
esta este eu foi for fosse ha isso isto ja la lhe mais mas me mesmo meu muito na
nao nas no nos nossa nosso num numa o os ou para pela pelo por porque qual quando
que quem se sem ser seu sua suas seus so sobre sua tambem te tem tinha um uma
voce vos
""".split()
)

DUTCH_STOPWORDS: frozenset[str] = frozenset(
    """
aan af al als bij dan dat de der deze die dit doch doen door dus een en er ge
geen geweest haar had heb hebben heeft hem het hier hij hoe hun iemand iets ik in
is ja je kan kon kunnen maar me meer men met mij mijn na naar niet niets nog nu
of om omdat onder ons ook op over reeds te tegen toch toen tot u uit uw van veel
voor want waren was wat werd wezen wie wil worden wordt zal ze zei zelf zich zij
zijn zo zonder zou
""".split()
)

STOPWORDS: dict[str, frozenset[str]] = {
    "en": ENGLISH_STOPWORDS,
    "es": SPANISH_STOPWORDS,
    "fr": FRENCH_STOPWORDS,
    "de": GERMAN_STOPWORDS,
    "it": ITALIAN_STOPWORDS,
    "pt": PORTUGUESE_STOPWORDS,
    "nl": DUTCH_STOPWORDS,
}

SUPPORTED_STOPWORD_LANGUAGES: tuple[str, ...] = tuple(sorted(STOPWORDS))

# Function-word profiles used by the native language detector. These are the
# highest-frequency closed-class words per language; overlap between Romance
# languages is handled by scoring distinctive markers higher (see language.py).
LANGUAGE_MARKERS: dict[str, tuple[str, ...]] = {
    "en": (
        "the", "and", "of", "to", "in", "is", "that", "it", "for", "was",
        "with", "as", "on", "are", "this", "be", "have", "not", "but", "they",
    ),
    "es": (
        "el", "la", "los", "las", "de", "que", "en", "por", "con", "para",
        "una", "como", "pero", "sus", "muy", "esta", "son", "del", "al", "no",
    ),
    "fr": (
        "le", "la", "les", "des", "une", "est", "dans", "que", "pour", "pas",
        "qui", "sur", "avec", "plus", "sont", "aux", "cette", "nous", "vous", "mais",
    ),
    "de": (
        "der", "die", "das", "und", "ist", "nicht", "den", "mit", "sich", "auf",
        "eine", "auch", "als", "dem", "des", "war", "wird", "sind", "aber", "durch",
    ),
    "it": (
        "il", "lo", "la", "gli", "che", "non", "per", "con", "una", "sono",
        "come", "anche", "della", "nel", "questo", "piu", "hanno", "alla", "dei", "sul",
    ),
    "pt": (
        "os", "as", "que", "nao", "uma", "com", "para", "mais", "como", "mas",
        "dos", "das", "pelo", "pela", "sao", "foi", "seu", "sua", "muito", "isso",
    ),
    "nl": (
        "de", "het", "een", "en", "van", "is", "dat", "niet", "op", "te",
        "zijn", "voor", "met", "maar", "aan", "worden", "door", "over", "naar", "ook",
    ),
}

# Unicode block probes; the first matching block short-circuits detection because
# script identity is a much stronger signal than function-word overlap.
SCRIPT_RANGES: tuple[tuple[str, int, int], ...] = (
    ("ru", 0x0400, 0x04FF),  # Cyrillic (reported as Cyrillic-script, see language.py)
    ("el", 0x0370, 0x03FF),  # Greek
    ("he", 0x0590, 0x05FF),  # Hebrew
    ("ar", 0x0600, 0x06FF),  # Arabic
    ("hi", 0x0900, 0x097F),  # Devanagari
    ("bn", 0x0980, 0x09FF),  # Bengali
    ("ta", 0x0B80, 0x0BFF),  # Tamil
    ("th", 0x0E00, 0x0E7F),  # Thai
    ("ka", 0x10A0, 0x10FF),  # Georgian
    ("ko", 0xAC00, 0xD7AF),  # Hangul syllables
    ("ja", 0x3040, 0x30FF),  # Hiragana + Katakana
    ("zh", 0x4E00, 0x9FFF),  # CJK unified ideographs
)

SCRIPT_LABELS: dict[str, str] = {
    "ru": "cyrillic",
    "el": "greek",
    "he": "hebrew",
    "ar": "arabic",
    "hi": "devanagari",
    "bn": "bengali",
    "ta": "tamil",
    "th": "thai",
    "ka": "georgian",
    "ko": "hangul",
    "ja": "kana",
    "zh": "han",
}

# Signed valence weights in roughly [-4, 4] (VADER-style scale). Positive terms
# raise the compound score, negative terms lower it. Weights are hand-set and
# deliberately conservative; they are not learned from any labelled corpus.
SENTIMENT_LEXICON: dict[str, float] = {
    # Strong positive
    "outstanding": 3.5, "excellent": 3.4, "superb": 3.4, "phenomenal": 3.4,
    "exceptional": 3.3, "fantastic": 3.3, "brilliant": 3.2, "wonderful": 3.2,
    "amazing": 3.1, "perfect": 3.1, "delighted": 3.0, "flawless": 3.0,
    "terrific": 3.0, "marvelous": 3.0, "impeccable": 2.9, "thrilled": 2.9,
    "love": 2.8, "loved": 2.8, "loves": 2.7, "awesome": 2.8, "gorgeous": 2.7,
    "beautiful": 2.6, "delightful": 2.6, "stellar": 2.6, "admirable": 2.4,
    # Moderate positive
    "great": 2.4, "happy": 2.3, "pleased": 2.2, "enjoyed": 2.2, "enjoy": 2.1,
    "recommend": 2.1, "recommended": 2.1, "satisfied": 2.0, "impressive": 2.2,
    "impressed": 2.2, "helpful": 2.0, "friendly": 1.9, "reliable": 2.0,
    "efficient": 1.9, "responsive": 1.9, "smooth": 1.7, "comfortable": 1.7,
    "affordable": 1.6, "generous": 1.8, "polite": 1.7, "professional": 1.6,
    "clean": 1.5, "fast": 1.3, "quick": 1.3, "easy": 1.5, "clear": 1.2,
    "useful": 1.6, "valuable": 1.8, "solid": 1.4, "good": 1.9, "nice": 1.6,
    "better": 1.4, "best": 2.5, "improved": 1.5, "improvement": 1.3,
    "works": 1.1, "worked": 1.1, "resolved": 1.6, "fixed": 1.4, "praise": 2.0,
    "thanks": 1.5, "thank": 1.5, "grateful": 2.0, "appreciate": 1.8,
    "appreciated": 1.8, "recommendable": 1.9, "convenient": 1.6, "safe": 1.4,
    "accurate": 1.6, "consistent": 1.3, "durable": 1.5, "premium": 1.4,
    "worth": 1.4, "bargain": 1.6, "success": 1.9, "successful": 1.9,
    "win": 1.7, "winner": 1.9, "approved": 1.2, "confident": 1.5,
    "trust": 1.7, "trusted": 1.7, "honest": 1.8, "fair": 1.2, "calm": 1.0,
    "fun": 1.8, "funny": 1.4, "smart": 1.6, "elegant": 1.7, "quiet": 0.8,
    # Moderate negative
    "bad": -2.1, "poor": -2.1, "disappointing": -2.4, "disappointed": -2.4,
    "disappointment": -2.4, "unhappy": -2.2, "frustrated": -2.3,
    "frustrating": -2.3, "annoying": -2.0, "annoyed": -2.0, "slow": -1.5,
    "expensive": -1.4, "overpriced": -2.0, "cheap": -0.9, "flimsy": -1.8,
    "broken": -2.4, "broke": -2.2, "defective": -2.5, "faulty": -2.4,
    "damaged": -2.2, "useless": -2.7, "worthless": -2.9, "waste": -2.5,
    "wasted": -2.5, "problem": -1.5, "problems": -1.6, "issue": -1.2,
    "issues": -1.3, "bug": -1.5, "bugs": -1.6, "crash": -2.2, "crashes": -2.2,
    "crashed": -2.2, "error": -1.6, "errors": -1.7, "fail": -2.2,
    "failed": -2.3, "failure": -2.4, "fails": -2.2, "reject": -1.8,
    "rejected": -1.9, "denied": -1.9, "delay": -1.6, "delayed": -1.7,
    "late": -1.3, "confusing": -1.8, "confused": -1.6, "difficult": -1.5,
    "hard": -1.0, "complicated": -1.4, "unclear": -1.5, "misleading": -2.2,
    "dishonest": -2.7, "rude": -2.5, "unhelpful": -2.3, "unprofessional": -2.4,
    "ignored": -2.1, "unresponsive": -2.2, "dirty": -2.0, "smelly": -2.0,
    "noisy": -1.6, "uncomfortable": -1.8, "unsafe": -2.5, "risky": -1.7,
    "inaccurate": -1.9, "inconsistent": -1.7, "unreliable": -2.4,
    "worse": -2.0, "worst": -3.1, "hate": -2.9, "hated": -2.9, "hates": -2.8,
    "awful": -3.0, "terrible": -3.0, "horrible": -3.1, "dreadful": -3.0,
    "atrocious": -3.3, "abysmal": -3.3, "disgusting": -3.2, "appalling": -3.2,
    "unacceptable": -2.8, "scam": -3.2, "fraud": -3.3, "cheated": -3.0,
    "angry": -2.4, "furious": -2.9, "upset": -2.0, "regret": -2.1,
    "avoid": -2.0, "refund": -1.4, "complaint": -1.8, "complain": -1.7,
    "sad": -1.9, "painful": -2.0, "boring": -1.7, "bland": -1.3,
    "mediocre": -1.6, "lacking": -1.5, "missing": -1.3, "insufficient": -1.7,
    "wrong": -1.9, "mistake": -1.7, "damage": -2.0, "loss": -1.8,
    "lost": -1.7, "penalty": -1.6, "default": -1.4, "delinquent": -2.2,
    "overdue": -1.9, "bankrupt": -3.0, "downgrade": -1.8,
}

NEGATORS: frozenset[str] = frozenset(
    """
not no never none nobody nothing neither nor cannot cant can't dont don't
doesnt doesn't didnt didn't isnt isn't arent aren't wasnt wasn't werent
weren't wont won't wouldnt wouldn't shouldnt shouldn't couldnt couldn't
hardly barely scarcely without lack lacks lacking
""".split()
)

INTENSIFIERS: dict[str, float] = {
    "very": 0.35, "really": 0.32, "extremely": 0.45, "incredibly": 0.42,
    "absolutely": 0.42, "completely": 0.38, "totally": 0.36, "highly": 0.32,
    "super": 0.34, "so": 0.24, "quite": 0.16, "rather": 0.14, "fairly": 0.10,
    "pretty": 0.14, "particularly": 0.22, "especially": 0.24, "deeply": 0.30,
    "truly": 0.30, "utterly": 0.42, "exceptionally": 0.42, "remarkably": 0.36,
    "somewhat": -0.18, "slightly": -0.24, "marginally": -0.26, "barely": -0.30,
    "kinda": -0.16, "sorta": -0.16, "mildly": -0.20, "a_little": -0.20,
}

EMOTICONS: dict[str, float] = {
    ":)": 1.6, ":-)": 1.6, ":d": 2.0, ":-d": 2.0, "=)": 1.5, "(:": 1.4,
    ":p": 1.0, "^^": 1.3, "<3": 2.2, ":')": 1.2, ";)": 1.1,
    ":(": -1.7, ":-(": -1.7, ":'(": -2.2, ":/": -0.9, ":\\": -0.9,
    ":|": -0.5, "d:": -1.6, "):": -1.5, "</3": -2.0, ":o": 0.2,
}

# Conservative English suffix stripping (Porter-lite). Rules are applied in
# order; only the first match fires and the stem must stay long enough to be a
# plausible word. This is a deterministic fallback for the no-extras path.
SUFFIX_STEM_RULES: tuple[tuple[str, str, int], ...] = (
    ("ational", "ate", 5),
    ("fulness", "ful", 5),
    ("ousness", "ous", 5),
    ("iveness", "ive", 5),
    ("ization", "ize", 5),
    ("iveness", "ive", 5),
    ("biliti", "ble", 5),
    ("tional", "tion", 5),
    ("alism", "al", 4),
    ("aliti", "al", 4),
    ("iviti", "ive", 4),
    ("ement", "", 5),
    ("ation", "ate", 5),
    ("ingly", "", 5),
    ("edly", "", 5),
    ("ness", "", 4),
    ("ment", "", 5),
    ("ency", "ence", 4),
    ("ancy", "ance", 4),
    ("able", "", 5),
    ("ible", "", 5),
    ("ally", "al", 4),
    ("ies", "y", 4),
    ("ing", "", 5),
    ("ies", "y", 4),
    ("ive", "", 5),
    ("ful", "", 4),
    ("ous", "", 5),
    ("ers", "", 4),
    ("ies", "y", 4),
    ("ed", "", 4),
    ("es", "", 4),
    ("ly", "", 4),
    ("er", "", 4),
    ("s", "", 3),
)

# Regex-backed entity rules for the dependency-free NER path. Patterns are
# precision-first: they only fire on unambiguous surface forms.
_PHONE_PATTERN = (
    r"(?<!\d)(?:\+\d{1,3}[\s.\-]?)?(?:\(\d{2,4}\)[\s.\-]?)?"
    r"\d{3,4}[\s.\-]\d{3,4}(?:[\s.\-]\d{2,4})?(?!\d)"
)
_MONEY_PATTERN = (
    r"(?:[$\u00a3\u20ac\u00a5]\s?\d[\d,]*(?:\.\d+)?"
    r"(?:\s?(?:k|m|bn|billion|million|thousand))?)"
    r"|(?:\b\d[\d,]*(?:\.\d+)?\s?(?:USD|EUR|GBP|NGN|JPY|CAD|AUD)\b)"
)
_DATE_PATTERN = (
    r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}"
    r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
    r"[a-z]*\.?\s+\d{1,2},?\s+\d{4})\b"
)
_ORG_PATTERN = (
    r"\b(?:[A-Z][A-Za-z&.\-]+\s)*[A-Z][A-Za-z&.\-]+\s"
    r"(?:Inc|Inc\.|LLC|Ltd|Ltd\.|PLC|GmbH|Corp|Corp\.|Corporation"
    r"|Company|Bank|Group|Holdings|Limited)\b"
)

RULE_ENTITY_PATTERNS: tuple[tuple[str, str], ...] = (
    ("EMAIL", r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    ("URL", r"\bhttps?://[^\s<>\"')]+"),
    ("IP", r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    ("PHONE", _PHONE_PATTERN),
    ("MONEY", _MONEY_PATTERN),
    # The word forms take a trailing \b; '%' must not, or "12%)" and a trailing
    # "12%" would never match — '%' is itself a non-word character.
    ("PERCENT", r"\b\d+(?:\.\d+)?\s?(?:%|(?:percent|pct)\b)"),
    ("DATE", _DATE_PATTERN),
    ("TIME", r"\b(?:[01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?\s?(?:AM|PM|am|pm)?\b"),
    ("ID", r"\b(?:[A-Z]{2,5}-\d{3,8}|(?:INV|ORD|TKT|CASE|REF|ACC)[#\-]?\d{4,10})\b"),
    ("ORG", _ORG_PATTERN),
    ("PERSON", r"\b(?:Mr|Mrs|Ms|Miss|Dr|Prof|Sir|Madam)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b"),
)

RULE_ENTITY_LABELS: tuple[str, ...] = tuple(label for label, _ in RULE_ENTITY_PATTERNS)


def stopwords_for(language: str) -> frozenset[str]:
    """Return the shipped stopword list for a language.

    Stopwords are the words too common to distinguish anything — "the", "of",
    "and". Removing them shrinks the vocabulary and stops them dominating
    frequency counts. It is not always the right call: in short documents they
    can carry real signal, and phrase features like "not working" lose their
    meaning when the negation is stripped.

    Parameters
    ----------
    language:
        A language code such as ``'en'``. Matching is case-insensitive.

    Returns
    -------
    frozenset of str
        The stopword terms, already lowercased so they match normalised
        tokens.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        No list ships for that language. The message names what is available.
        For wider coverage, install ``buildml[nlp]`` or pass your own terms
        through the ``stopwords`` argument on the fit functions.

    Notes
    -----
    Consider ``max_df`` on the vectorizer as an alternative or a complement: it
    discards terms appearing in more than a given share of documents, which
    adapts to your corpus and catches domain boilerplate that no general list
    would contain.

    See Also
    --------
    buildml.nlp.fit.fit_text_classifier : Accepts a language or explicit terms.
    """
    from buildml.core.errors import ValidationError

    key = str(language).strip().lower().replace("_", "-").split("-")[0]
    try:
        return STOPWORDS[key]
    except KeyError as exc:
        raise ValidationError(
            f"No built-in stopword list for language {language!r}. "
            f"Built-in languages: {list(SUPPORTED_STOPWORD_LANGUAGES)}. "
            "Pass stopwords=(...) explicitly for other languages."
        ) from exc


__all__ = [
    "EMOTICONS",
    "INTENSIFIERS",
    "LANGUAGE_MARKERS",
    "NEGATORS",
    "RULE_ENTITY_LABELS",
    "RULE_ENTITY_PATTERNS",
    "SCRIPT_LABELS",
    "SCRIPT_RANGES",
    "SENTIMENT_LEXICON",
    "STOPWORDS",
    "SUFFIX_STEM_RULES",
    "SUPPORTED_STOPWORD_LANGUAGES",
    "stopwords_for",
]
