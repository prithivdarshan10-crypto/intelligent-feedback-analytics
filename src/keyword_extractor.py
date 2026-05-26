# =============================================================================
# src/keyword_extractor.py
# Keyword and key-phrase extraction using TF-IDF and word frequency analysis.
# Supports per-sentiment and per-category breakdowns.
# =============================================================================

import logging
from collections import Counter
from typing import List, Tuple

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from config.settings import TOP_N_KEYWORDS, MAX_NGRAM

logger = logging.getLogger(__name__)

# Words to exclude even after stopword removal (too generic for keyword insight)
_EXTRA_EXCLUDE = {
    "product", "item", "thing", "stuff", "use", "used", "using",
    "get", "got", "one", "also", "would", "could", "like",
    "even", "still", "already", "made", "make", "really",
    "just", "good", "great", "bad", "nice", "best", "worst",
    "well", "much", "many", "lot", "time", "day", "week", "month",
}


# ── Core TF-IDF extraction ────────────────────────────────────────────────────

def extract_keywords_tfidf(
    texts: List[str],
    top_n: int = TOP_N_KEYWORDS,
    ngram_range: Tuple[int, int] = (1, MAX_NGRAM),
) -> List[Tuple[str, float]]:
    """
    Extract the most important keywords / key-phrases from a list of texts
    using TF-IDF scoring.

    Parameters
    ----------
    texts      : List of cleaned review strings.
    top_n      : How many top keywords to return.
    ngram_range: (min_n, max_n) for n-gram extraction.
                 (1,1) = single words; (1,2) = words + bigrams.

    Returns
    -------
    List of (keyword, tfidf_score) tuples sorted by score descending.
    """
    # Filter out empty strings
    valid_texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if len(valid_texts) < 2:
        logger.warning("Not enough texts for TF-IDF extraction (need ≥ 2).")
        return []

    try:
        vectorizer = TfidfVectorizer(
            max_features   = 500,
            ngram_range    = ngram_range,
            min_df         = 2,           # ignore terms that appear in < 2 docs
            max_df         = 0.85,        # ignore terms appearing in > 85 % of docs
            stop_words     = "english",
            sublinear_tf   = True,        # apply 1 + log(tf) smoothing
        )
        tfidf_matrix = vectorizer.fit_transform(valid_texts)
        feature_names = vectorizer.get_feature_names_out()

        # Mean TF-IDF score across all documents for ranking
        mean_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()

        # Build (term, score) pairs and filter junk words
        scored = [
            (term, float(score))
            for term, score in zip(feature_names, mean_scores)
            if term not in _EXTRA_EXCLUDE and len(term) > 2
        ]

        # Sort by score and return top_n
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]

    except Exception as exc:
        logger.error("TF-IDF extraction failed: %s", exc)
        return []


def extract_word_frequency(
    texts: List[str],
    top_n: int = TOP_N_KEYWORDS,
) -> List[Tuple[str, int]]:
    """
    Simple word-frequency counter – useful as a complement to TF-IDF.

    Parameters
    ----------
    texts : List of cleaned review strings.
    top_n : How many top words to return.

    Returns
    -------
    List of (word, count) tuples sorted by frequency descending.
    """
    counter: Counter = Counter()
    for text in texts:
        if isinstance(text, str):
            words = [
                w for w in text.split()
                if len(w) > 2 and w not in _EXTRA_EXCLUDE
            ]
            counter.update(words)
    return counter.most_common(top_n)


# ── Per-sentiment keyword extraction ─────────────────────────────────────────

def keywords_by_sentiment(
    df: pd.DataFrame,
    text_column: str  = "cleaned_text",
    top_n: int        = 15,
) -> dict:
    """
    Extract top keywords separately for Positive, Negative, and Neutral reviews.

    Parameters
    ----------
    df          : DataFrame with `text_column` and `sentiment` columns.
    text_column : Column holding cleaned text.
    top_n       : Keywords to extract per sentiment class.

    Returns
    -------
    dict  {sentiment_label: [(keyword, score), …]}
    """
    if "sentiment" not in df.columns or text_column not in df.columns:
        raise ValueError("DataFrame must have 'sentiment' and text columns.")

    result = {}
    for label in ["Positive", "Negative", "Neutral"]:
        subset = df[df["sentiment"] == label][text_column].tolist()
        result[label] = extract_keywords_tfidf(subset, top_n=top_n)

    return result


# ── Per-category keyword extraction ──────────────────────────────────────────

def keywords_by_category(
    df: pd.DataFrame,
    text_column:     str = "cleaned_text",
    category_column: str = "product_category",
    top_n: int           = 10,
) -> dict:
    """
    Extract top keywords for each product category.

    Parameters
    ----------
    df               : DataFrame with text and category columns.
    text_column      : Column holding cleaned text.
    category_column  : Column holding product/category labels.
    top_n            : Keywords per category.

    Returns
    -------
    dict  {category_label: [(keyword, score), …]}
    """
    if category_column not in df.columns or text_column not in df.columns:
        raise ValueError("DataFrame must have category and text columns.")

    result = {}
    for cat in df[category_column].unique():
        subset = df[df[category_column] == cat][text_column].tolist()
        result[str(cat)] = extract_keywords_tfidf(subset, top_n=top_n)

    return result


# ── Utility: convert keyword list to a flat DataFrame ─────────────────────────

def keywords_to_dataframe(
    keywords: List[Tuple[str, float]],
    score_label: str = "tfidf_score",
) -> pd.DataFrame:
    """
    Convert a list of (keyword, score) tuples to a tidy DataFrame.

    Parameters
    ----------
    keywords    : Output of any `extract_keywords_*` function.
    score_label : Name for the score column.

    Returns
    -------
    pd.DataFrame  with columns [keyword, score_label].
    """
    if not keywords:
        return pd.DataFrame(columns=["keyword", score_label])
    return pd.DataFrame(keywords, columns=["keyword", score_label])
