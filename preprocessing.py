"""Reusable NLTK-based text preprocessing for the plagiarism checker.

Pipeline (every step is a separate function so it can be reused or tested alone):

    clean_text -> tokenize -> remove_stopwords -> lemmatize_tokens

`preprocess_text` runs the whole pipeline and returns one string that is ready
for TF-IDF vectorization. Nothing in this module depends on Streamlit.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from functools import lru_cache
from typing import Dict, FrozenSet, Iterable, List, Optional, Tuple

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

logger = logging.getLogger(__name__)

# NLTK resource name -> places NLTK may keep it (extracted folder or zip file).
REQUIRED_NLTK_RESOURCES: Dict[str, Tuple[str, ...]] = {
    "stopwords": ("corpora/stopwords", "corpora/stopwords.zip"),
    "wordnet": ("corpora/wordnet", "corpora/wordnet.zip"),
}

MIN_TOKEN_LENGTH = 2

_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
_NON_TEXT_PATTERN = re.compile(r"[^a-z0-9'\s]")
# An apostrophe that is not between two letters is a quote mark, not part of a word.
_STRAY_APOSTROPHE_PATTERN = re.compile(r"(?<![a-z])'|'(?![a-z])")
_WHITESPACE_PATTERN = re.compile(r"\s+")
# Words, numbers and contractions such as "don't" (kept so NLTK's stopword list matches them).
_TOKENIZER = RegexpTokenizer(r"[a-z0-9]+(?:'[a-z]+)*")


class NLTKResourceError(RuntimeError):
    """Raised when a required NLTK resource is missing and could not be loaded."""


# --------------------------------------------------------------------------- #
# NLTK resource handling
# --------------------------------------------------------------------------- #
def _is_resource_available(locations: Iterable[str]) -> bool:
    """Return True if NLTK can find the resource at any of the given locations."""
    for location in locations:
        try:
            nltk.data.find(location)
            return True
        except LookupError:
            continue
    return False


def _lemmatizer_works() -> bool:
    """Smoke test: some NLTK versions also need the 'omw-1.4' data to lemmatize."""
    try:
        WordNetLemmatizer().lemmatize("tests")
        return True
    except LookupError:
        return False


def _download(resource: str) -> bool:
    """Download one NLTK resource; never raises (offline machines are expected)."""
    try:
        return bool(nltk.download(resource, quiet=True))
    except Exception as exc:  # network errors, permission errors, ...
        logger.warning("Could not download NLTK resource %s: %s", resource, exc)
        return False


def ensure_nltk_resources() -> Tuple[bool, str]:
    """Make sure every NLTK resource the pipeline needs is installed.

    Returns:
        (True, "ready") on success, otherwise (False, human-readable explanation).
    """
    missing: List[str] = []

    for name, locations in REQUIRED_NLTK_RESOURCES.items():
        if _is_resource_available(locations):
            continue
        if not (_download(name) and _is_resource_available(locations)):
            missing.append(name)

    if not missing and not _lemmatizer_works():
        # Older/newer NLTK releases may additionally need the multilingual WordNet data.
        _download("omw-1.4")
        if not _lemmatizer_works():
            missing.append("wordnet (lemmatizer data)")

    if missing:
        names = ", ".join(missing)
        return False, (
            f"Could not load NLTK resources: {names}. Check your internet connection "
            "and restart the app, or install them manually with: "
            'python -c "import nltk; nltk.download(\'stopwords\'); nltk.download(\'wordnet\')"'
        )
    return True, "ready"


@lru_cache(maxsize=1)
def get_stopwords() -> FrozenSet[str]:
    """Return NLTK's English stopword set (loaded once)."""
    try:
        return frozenset(stopwords.words("english"))
    except LookupError as exc:
        raise NLTKResourceError(
            "NLTK 'stopwords' data not found. Call ensure_nltk_resources() first."
        ) from exc


@lru_cache(maxsize=1)
def get_lemmatizer() -> WordNetLemmatizer:
    """Return a shared WordNet lemmatizer."""
    return WordNetLemmatizer()


# --------------------------------------------------------------------------- #
# Pipeline steps
# --------------------------------------------------------------------------- #
def clean_text(text: object) -> str:
    """Lowercase the text and strip punctuation, URLs and special characters.

    Apostrophes inside words are kept on purpose ("don't") so stopword removal
    can recognise contractions. Accented letters are folded to plain ASCII
    ("café" -> "cafe"); characters with no ASCII form (e.g. other scripts) are dropped.

    Non-string or blank input returns an empty string instead of raising.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower().replace("’", "'").replace("‘", "'")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = _URL_PATTERN.sub(" ", text)
    text = _NON_TEXT_PATTERN.sub(" ", text)
    text = _STRAY_APOSTROPHE_PATTERN.sub(" ", text)
    return _WHITESPACE_PATTERN.sub(" ", text).strip()


def tokenize(text: str) -> List[str]:
    """Split cleaned text into word tokens using NLTK's RegexpTokenizer."""
    if not text:
        return []
    return _TOKENIZER.tokenize(text)


def _strip_apostrophes(token: str) -> str:
    """Turn "john's" into "john" and "o'clock" into "oclock"."""
    if token.endswith("'s"):
        token = token[:-2]
    return token.replace("'", "")


def remove_stopwords(
    tokens: Iterable[str],
    stop_words: Optional[FrozenSet[str]] = None,
    min_length: int = MIN_TOKEN_LENGTH,
) -> List[str]:
    """Drop stopwords and very short tokens (single letters carry little meaning)."""
    stop_words = stop_words if stop_words is not None else get_stopwords()
    kept: List[str] = []
    for token in tokens:
        if token in stop_words:
            continue
        token = _strip_apostrophes(token)
        if len(token) >= min_length and token not in stop_words:
            kept.append(token)
    return kept


def lemmatize_tokens(tokens: Iterable[str]) -> List[str]:
    """Reduce each token to its dictionary form ("studies" -> "study", "running" -> "run").

    Each word is tried as a verb first and then as a noun, which gives sensible
    results without needing a part-of-speech tagger.
    """
    lemmatizer = get_lemmatizer()
    return [lemmatizer.lemmatize(lemmatizer.lemmatize(token, pos="v"), pos="n") for token in tokens]


def preprocess_to_tokens(text: object) -> List[str]:
    """Run the full pipeline and return the final list of tokens."""
    tokens = tokenize(clean_text(text))
    tokens = remove_stopwords(tokens)
    return lemmatize_tokens(tokens)


def preprocess_text(text: object) -> str:
    """Run the full pipeline and return the tokens joined by single spaces.

    Returns an empty string for empty, invalid or stopword-only input.
    """
    return " ".join(preprocess_to_tokens(text))
