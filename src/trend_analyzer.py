# =============================================================================
# src/trend_analyzer.py
# Time-series and categorical trend analysis for customer feedback.
# =============================================================================

import logging
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ── Date helpers ─────────────────────────────────────────────────────────────

def _ensure_datetime(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """Parse date_column to datetime if not already and return a copy."""
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    # Drop rows where date could not be parsed
    invalid = df[date_column].isna().sum()
    if invalid:
        logger.warning("Dropped %d rows with unparseable dates.", invalid)
        df = df.dropna(subset=[date_column])
    return df


# ── Monthly sentiment trend ───────────────────────────────────────────────────

def monthly_sentiment_trend(
    df: pd.DataFrame,
    date_column:      str = "date",
    sentiment_column: str = "sentiment",
) -> pd.DataFrame:
    """
    Compute the monthly count of each sentiment label.

    Returns
    -------
    pd.DataFrame  with columns: [month, Positive, Negative, Neutral, total,
                                  positive_pct, negative_pct, neutral_pct]
    """
    required = {date_column, sentiment_column}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required}")

    df = _ensure_datetime(df, date_column)
    df["month"] = df[date_column].dt.to_period("M")

    pivot = (
        df.groupby(["month", sentiment_column])
          .size()
          .unstack(fill_value=0)
          .reset_index()
    )
    pivot["month"] = pivot["month"].astype(str)

    # Ensure all three sentiment columns exist even if one is absent
    for col in ["Positive", "Negative", "Neutral"]:
        if col not in pivot.columns:
            pivot[col] = 0

    pivot["total"]        = pivot["Positive"] + pivot["Negative"] + pivot["Neutral"]
    pivot["positive_pct"] = (pivot["Positive"] / pivot["total"].replace(0, np.nan) * 100).round(1)
    pivot["negative_pct"] = (pivot["Negative"] / pivot["total"].replace(0, np.nan) * 100).round(1)
    pivot["neutral_pct"]  = (pivot["Neutral"]  / pivot["total"].replace(0, np.nan) * 100).round(1)

    return pivot.sort_values("month").reset_index(drop=True)


# ── Monthly average sentiment score ──────────────────────────────────────────

def monthly_avg_score(
    df: pd.DataFrame,
    date_column:  str = "date",
    score_column: str = "sentiment_score",
) -> pd.DataFrame:
    """
    Compute the mean VADER compound score per calendar month.

    Returns
    -------
    pd.DataFrame  with columns: [month, avg_sentiment_score, review_count]
    """
    required = {date_column, score_column}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    df = _ensure_datetime(df, date_column)
    df["month"] = df[date_column].dt.to_period("M")

    result = (
        df.groupby("month")[score_column]
          .agg(avg_sentiment_score="mean", review_count="count")
          .reset_index()
    )
    result["month"]               = result["month"].astype(str)
    result["avg_sentiment_score"] = result["avg_sentiment_score"].round(4)
    return result.sort_values("month").reset_index(drop=True)


# ── Category-level breakdown ──────────────────────────────────────────────────

def category_sentiment_breakdown(
    df: pd.DataFrame,
    category_column:  str = "product_category",
    sentiment_column: str = "sentiment",
) -> pd.DataFrame:
    """
    For each product category compute the count and percentage of each
    sentiment label.

    Returns
    -------
    pd.DataFrame  with columns:
        [category, Positive, Negative, Neutral, total,
         positive_pct, negative_pct, neutral_pct, net_sentiment_score]
    """
    required = {category_column, sentiment_column}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required}")

    pivot = (
        df.groupby([category_column, sentiment_column])
          .size()
          .unstack(fill_value=0)
          .reset_index()
          .rename(columns={category_column: "category"})
    )

    for col in ["Positive", "Negative", "Neutral"]:
        if col not in pivot.columns:
            pivot[col] = 0

    pivot["total"]        = pivot["Positive"] + pivot["Negative"] + pivot["Neutral"]
    pivot["positive_pct"] = (pivot["Positive"] / pivot["total"].replace(0, np.nan) * 100).round(1)
    pivot["negative_pct"] = (pivot["Negative"] / pivot["total"].replace(0, np.nan) * 100).round(1)
    pivot["neutral_pct"]  = (pivot["Neutral"]  / pivot["total"].replace(0, np.nan) * 100).round(1)

    # Net Sentiment Score: (Positive − Negative) / total × 100
    pivot["net_sentiment_score"] = (
        (pivot["Positive"] - pivot["Negative"]) / pivot["total"].replace(0, np.nan) * 100
    ).round(1)

    return pivot.sort_values("net_sentiment_score", ascending=False).reset_index(drop=True)


# ── Rating distribution ───────────────────────────────────────────────────────

def rating_distribution(
    df: pd.DataFrame,
    rating_column: str = "rating",
) -> pd.DataFrame:
    """
    Count reviews per star rating (1-5).

    Returns
    -------
    pd.DataFrame  with columns: [rating, count, percentage]
    """
    if rating_column not in df.columns:
        return pd.DataFrame()

    counts = df[rating_column].value_counts().sort_index().reset_index()
    counts.columns = ["rating", "count"]
    counts["percentage"] = (counts["count"] / counts["count"].sum() * 100).round(1)
    return counts


# ── Weekly volume ─────────────────────────────────────────────────────────────

def weekly_review_volume(
    df: pd.DataFrame,
    date_column: str = "date",
) -> pd.DataFrame:
    """
    Count total reviews per ISO week.

    Returns
    -------
    pd.DataFrame  with columns: [week, review_count]
    """
    if date_column not in df.columns:
        return pd.DataFrame()

    df = _ensure_datetime(df, date_column)
    df["week"] = df[date_column].dt.to_period("W").astype(str)
    result = (
        df.groupby("week")
          .size()
          .reset_index(name="review_count")
          .sort_values("week")
          .reset_index(drop=True)
    )
    return result


# ── Top & bottom products ─────────────────────────────────────────────────────

def top_bottom_products(
    df: pd.DataFrame,
    product_column:   str = "product_name",
    sentiment_column: str = "sentiment",
    top_n: int             = 5,
) -> dict:
    """
    Identify the highest-rated (most Positive) and lowest-rated
    (most Negative) products.

    Returns
    -------
    dict  {top: DataFrame, bottom: DataFrame}
    """
    if product_column not in df.columns or sentiment_column not in df.columns:
        return {"top": pd.DataFrame(), "bottom": pd.DataFrame()}

    summary = (
        df.groupby(product_column)[sentiment_column]
          .value_counts(normalize=True)
          .mul(100)
          .round(1)
          .unstack(fill_value=0)
          .reset_index()
    )

    for col in ["Positive", "Negative", "Neutral"]:
        if col not in summary.columns:
            summary[col] = 0.0

    top    = summary.nlargest (top_n, "Positive").reset_index(drop=True)
    bottom = summary.nlargest (top_n, "Negative").reset_index(drop=True)
    return {"top": top, "bottom": bottom}
