# =============================================================================
# src/sentiment_analyzer.py
# Dual-engine sentiment analysis:
#   1. VADER (rule-based, fast, no training data needed)
#   2. Logistic Regression (ML model trained on VADER pseudo-labels)
# =============================================================================

import os
import pickle
import logging

import numpy as np
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from config.settings import (
    POSITIVE_THRESHOLD, NEGATIVE_THRESHOLD, MODELS_DIR
)

logger = logging.getLogger(__name__)

# ── Ensure VADER lexicon is available ────────────────────────────────────────
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)

# Paths where the trained ML pipeline is serialised
_MODEL_PATH      = os.path.join(MODELS_DIR, "sentiment_pipeline.pkl")


def _get_vader() -> SentimentIntensityAnalyzer:
    """Return a cached VADER analyser instance."""
    if not hasattr(_get_vader, "_instance"):
        _get_vader._instance = SentimentIntensityAnalyzer()
    return _get_vader._instance


# ── Core VADER helpers ────────────────────────────────────────────────────────

def vader_scores(text: str) -> dict:
    """
    Return VADER polarity scores for a single text string.

    Returns
    -------
    dict with keys: neg, neu, pos, compound
    """
    if not isinstance(text, str) or not text.strip():
        return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}
    return _get_vader().polarity_scores(text)


def compound_to_label(compound: float) -> str:
    """
    Map a VADER compound score to a human-readable sentiment label.

    Rules
    -----
    compound >=  0.05  →  Positive
    compound <= -0.05  →  Negative
    otherwise          →  Neutral
    """
    if compound >= POSITIVE_THRESHOLD:
        return "Positive"
    elif compound <= NEGATIVE_THRESHOLD:
        return "Negative"
    return "Neutral"


def analyze_single(text: str) -> dict:
    """
    Analyse the sentiment of a single review text.

    Returns
    -------
    dict
        {label, compound, pos, neu, neg, confidence}
        confidence = the max of the three raw VADER scores.
    """
    scores  = vader_scores(text)
    label   = compound_to_label(scores["compound"])
    confidence = max(scores["pos"], scores["neu"], scores["neg"])
    return {
        "label":      label,
        "compound":   round(scores["compound"], 4),
        "pos":        round(scores["pos"],      4),
        "neu":        round(scores["neu"],      4),
        "neg":        round(scores["neg"],      4),
        "confidence": round(confidence,         4),
    }


# ── DataFrame-level analysis ─────────────────────────────────────────────────

def analyze_dataframe(df: pd.DataFrame, text_column: str = "review_text") -> pd.DataFrame:
    """
    Add sentiment columns to every row of *df*.

    New columns
    -----------
    sentiment        : Positive | Negative | Neutral
    sentiment_score  : VADER compound  (−1 … +1)
    sentiment_pos    : VADER positive component
    sentiment_neu    : VADER neutral  component
    sentiment_neg    : VADER negative component
    confidence       : rough confidence (max of pos/neu/neg)

    Parameters
    ----------
    df          : Source dataframe (not modified in-place).
    text_column : Column that holds raw or cleaned review text.

    Returns
    -------
    pd.DataFrame with the new sentiment columns appended.
    """
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found.")

    logger.info("Running VADER sentiment analysis on %d rows …", len(df))
    df = df.copy()

    results = df[text_column].fillna("").apply(analyze_single)

    df["sentiment"]       = results.apply(lambda r: r["label"])
    df["sentiment_score"] = results.apply(lambda r: r["compound"])
    df["sentiment_pos"]   = results.apply(lambda r: r["pos"])
    df["sentiment_neu"]   = results.apply(lambda r: r["neu"])
    df["sentiment_neg"]   = results.apply(lambda r: r["neg"])
    df["confidence"]      = results.apply(lambda r: r["confidence"])

    counts = df["sentiment"].value_counts().to_dict()
    logger.info("Sentiment breakdown → %s", counts)
    return df


# ── Optional ML classifier (Logistic Regression) ─────────────────────────────

def train_ml_model(df: pd.DataFrame, text_column: str = "cleaned_text") -> dict:
    """
    Train a TF-IDF + Logistic Regression pipeline using VADER pseudo-labels.

    This demonstrates an end-to-end sklearn workflow and adds a second
    opinion on top of VADER. It persists the model to disk so subsequent
    runs load it instead of retraining.

    Parameters
    ----------
    df          : DataFrame that MUST already have a 'sentiment' column.
    text_column : Column with preprocessed (cleaned) text.

    Returns
    -------
    dict  {accuracy, report_str, model_saved_path}
    """
    required = {text_column, "sentiment"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing columns: {missing}")

    # Drop rows where cleaned text is empty
    data = df[df[text_column].str.strip() != ""].copy()

    X = data[text_column]
    y = data["sentiment"]

    if len(X) < 20:
        return {"error": "Not enough data to train a model (need ≥ 20 samples)."}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build sklearn pipeline
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",  # Handle class imbalance
            random_state=42,
        )),
    ])

    logger.info("Training Logistic Regression sentiment classifier …")
    pipeline.fit(X_train, y_train)

    y_pred   = pipeline.predict(X_test)
    accuracy = round((y_pred == y_test).mean() * 100, 2)
    report   = classification_report(y_test, y_pred)

    # Save model to disk
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(_MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)

    logger.info("Model trained. Accuracy: %.2f%%", accuracy)
    return {
        "accuracy":         accuracy,
        "report_str":       report,
        "model_saved_path": _MODEL_PATH,
        "train_samples":    len(X_train),
        "test_samples":     len(X_test),
    }


def load_ml_model():
    """
    Load the persisted ML pipeline from disk.

    Returns
    -------
    sklearn Pipeline, or None if no model has been saved yet.
    """
    if os.path.exists(_MODEL_PATH):
        with open(_MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None


def get_sentiment_summary(df: pd.DataFrame) -> dict:
    """
    Compute high-level summary statistics from an already-analysed dataframe.

    Parameters
    ----------
    df : DataFrame with a 'sentiment' column.

    Returns
    -------
    dict with total, positive_pct, negative_pct, neutral_pct, avg_score.
    """
    if "sentiment" not in df.columns:
        return {}

    total      = len(df)
    counts     = df["sentiment"].value_counts()
    avg_score  = round(df.get("sentiment_score", pd.Series(dtype=float)).mean(), 4)

    return {
        "total":        total,
        "positive":     int(counts.get("Positive", 0)),
        "negative":     int(counts.get("Negative", 0)),
        "neutral":      int(counts.get("Neutral",  0)),
        "positive_pct": round(counts.get("Positive", 0) / total * 100, 1),
        "negative_pct": round(counts.get("Negative", 0) / total * 100, 1),
        "neutral_pct":  round(counts.get("Neutral",  0) / total * 100, 1),
        "avg_score":    avg_score if not np.isnan(avg_score) else 0.0,
    }
