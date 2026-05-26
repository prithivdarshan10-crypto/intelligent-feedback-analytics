# =============================================================================
# src/preprocessor.py
# Text cleaning and preprocessing pipeline for customer reviews
# =============================================================================

import re
import string
import logging

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

logger = logging.getLogger(__name__)

# ── Ensure required NLTK resources are available ──────────────────────────────
_NLTK_RESOURCES = [
    ("tokenizers/punkt",             "punkt"),
    ("tokenizers/punkt_tab",         "punkt_tab"),
    ("corpora/stopwords",            "stopwords"),
    ("corpora/wordnet",              "wordnet"),
    ("corpora/omw-1.4",              "omw-1.4"),
]

def _download_nltk_resources() -> None:
    """Download missing NLTK resources silently."""
    for path, package in _NLTK_RESOURCES:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)

_download_nltk_resources()

# ── Module-level singletons (expensive to create, so share them) ───────────────
_stop_words  = set(stopwords.words("english"))
_lemmatizer  = WordNetLemmatizer()

# Words that are meaningful for sentiment but sit in stopword lists
_KEEP_WORDS = {"not", "no", "never", "nor", "neither", "without", "hardly",
               "barely", "scarcely", "very", "really", "extremely", "quite",
               "rather", "absolutely", "truly", "highly", "deeply"}
_EFFECTIVE_STOPWORDS = _stop_words - _KEEP_WORDS


def clean_text(text: str) -> str:
    """
    Apply a full NLP preprocessing pipeline to a raw review string.

    Steps
    -----
    1. Lowercase
    2. Strip URLs and HTML tags
    3. Remove punctuation & digits
    4. Tokenise
    5. Remove stopwords (keeping sentiment-bearing negations)
    6. Lemmatise each token
    7. Rejoin and return

    Parameters
    ----------
    text : str
        Raw customer review text.

    Returns
    -------
    str
        Cleaned, lemmatised text ready for NLP tasks.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Step 1 – lowercase
    text = text.lower()

    # Step 2 – remove URLs (http / www)
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # Step 3 – remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Step 4 – expand common contractions (improves sentiment accuracy)
    contractions = {
        r"n't": " not", r"'re": " are", r"'s": " is",
        r"'d": " would", r"'ll": " will", r"'ve": " have",
        r"'m": " am", r"can't": "cannot", r"won't": "will not",
    }
    for pattern, replacement in contractions.items():
        text = re.sub(pattern, replacement, text)

    # Step 5 – keep only alphabetic characters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # Step 6 – collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Step 7 – tokenise
    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()

    # Step 8 – remove stopwords and short tokens (< 2 chars)
    tokens = [t for t in tokens if t not in _EFFECTIVE_STOPWORDS and len(t) > 1]

    # Step 9 – lemmatise
    tokens = [_lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def preprocess_dataframe(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """
    Apply `clean_text` to every row in `text_column` and add a
    `cleaned_text` column to the dataframe.

    Parameters
    ----------
    df          : pd.DataFrame  Source dataframe.
    text_column : str           Name of the column containing raw review text.

    Returns
    -------
    pd.DataFrame  Original dataframe with a new `cleaned_text` column appended.
    """
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in dataframe. "
                         f"Available columns: {list(df.columns)}")

    logger.info("Preprocessing %d reviews …", len(df))
    df = df.copy()
    df["cleaned_text"] = df[text_column].fillna("").apply(clean_text)

    # Flag empty results (very short reviews that became empty after cleaning)
    empty_count = (df["cleaned_text"].str.strip() == "").sum()
    if empty_count:
        logger.warning("%d reviews were empty after cleaning.", empty_count)

    logger.info("Preprocessing complete.")
    return df


def get_text_stats(df: pd.DataFrame, text_column: str) -> dict:
    """
    Return basic statistics about the raw text column.

    Returns a dict with keys: total_reviews, avg_word_count,
    max_word_count, min_word_count, empty_reviews.
    """
    if text_column not in df.columns:
        return {}

    word_counts = df[text_column].fillna("").apply(lambda x: len(x.split()))
    return {
        "total_reviews":   len(df),
        "avg_word_count":  round(word_counts.mean(), 1),
        "max_word_count":  int(word_counts.max()),
        "min_word_count":  int(word_counts.min()),
        "empty_reviews":   int((word_counts == 0).sum()),
    }
