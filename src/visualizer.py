# =============================================================================
# src/visualizer.py
# All chart/visualisation helpers used by the Streamlit dashboard.
# Returns Plotly figures (interactive) or Matplotlib figures (WordCloud).
# =============================================================================

import logging
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np
import plotly.express       as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# ── Colour constants ──────────────────────────────────────────────────────────
_SENTIMENT_COLORS = {
    "Positive": "#2ecc71",
    "Negative": "#e74c3c",
    "Neutral":  "#f39c12",
}
_CATEGORY_PALETTE = px.colors.qualitative.Set2
_FONT_FAMILY      = "Inter, Arial, sans-serif"
_BG_COLOR         = "rgba(0,0,0,0)"   # transparent background


def _apply_layout(fig: go.Figure, title: str = "", height: int = 400) -> go.Figure:
    """Apply consistent branding layout to any Plotly figure."""
    fig.update_layout(
        title            = dict(text=title, font=dict(size=16, family=_FONT_FAMILY)),
        paper_bgcolor    = _BG_COLOR,
        plot_bgcolor     = _BG_COLOR,
        font             = dict(family=_FONT_FAMILY, size=13),
        height           = height,
        margin           = dict(l=40, r=40, t=60, b=40),
        legend           = dict(orientation="h", yanchor="bottom", y=1.02,
                                xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e8e8e8", linecolor="#cccccc")
    fig.update_yaxes(showgrid=True, gridcolor="#e8e8e8", linecolor="#cccccc")
    return fig


# ── 1. Sentiment Pie Chart ────────────────────────────────────────────────────

def sentiment_pie(df: pd.DataFrame) -> go.Figure:
    """Doughnut chart showing the overall sentiment distribution."""
    if "sentiment" not in df.columns:
        return go.Figure()

    counts = df["sentiment"].value_counts().reset_index()
    counts.columns = ["sentiment", "count"]
    colors = [_SENTIMENT_COLORS.get(s, "#999") for s in counts["sentiment"]]

    fig = go.Figure(go.Pie(
        labels       = counts["sentiment"],
        values       = counts["count"],
        hole         = 0.45,
        marker_colors= colors,
        textinfo     = "label+percent",
        hovertemplate= "%{label}<br>Count: %{value}<br>%{percent}<extra></extra>",
    ))
    return _apply_layout(fig, "Overall Sentiment Distribution", height=380)


# ── 2. Sentiment Bar Chart by Category ───────────────────────────────────────

def sentiment_by_category_bar(
    df: pd.DataFrame,
    category_column: str = "product_category",
) -> go.Figure:
    """Grouped bar chart – sentiment breakdown per product category."""
    if "sentiment" not in df.columns or category_column not in df.columns:
        return go.Figure()

    pivot = (
        df.groupby([category_column, "sentiment"])
          .size()
          .unstack(fill_value=0)
          .reset_index()
    )

    fig = go.Figure()
    for sentiment in ["Positive", "Neutral", "Negative"]:
        if sentiment in pivot.columns:
            fig.add_trace(go.Bar(
                name       = sentiment,
                x          = pivot[category_column],
                y          = pivot[sentiment],
                marker_color= _SENTIMENT_COLORS[sentiment],
                hovertemplate= f"<b>{sentiment}</b><br>Category: %{{x}}<br>Count: %{{y}}<extra></extra>",
            ))

    fig.update_layout(barmode="group")
    return _apply_layout(fig, "Sentiment by Product Category", height=420)


# ── 3. Monthly Trend Line ─────────────────────────────────────────────────────

def monthly_trend_line(trend_df: pd.DataFrame) -> go.Figure:
    """
    Line chart showing monthly Positive / Negative / Neutral counts.

    Parameters
    ----------
    trend_df : Output of trend_analyzer.monthly_sentiment_trend()
    """
    if trend_df.empty or "month" not in trend_df.columns:
        return go.Figure()

    fig = go.Figure()
    for sentiment in ["Positive", "Neutral", "Negative"]:
        if sentiment in trend_df.columns:
            fig.add_trace(go.Scatter(
                x          = trend_df["month"],
                y          = trend_df[sentiment],
                mode       = "lines+markers",
                name       = sentiment,
                line       = dict(color=_SENTIMENT_COLORS[sentiment], width=2.5),
                marker     = dict(size=6),
                hovertemplate= f"<b>{sentiment}</b><br>Month: %{{x}}<br>Count: %{{y}}<extra></extra>",
            ))

    return _apply_layout(fig, "Monthly Sentiment Trend", height=400)


# ── 4. Average Sentiment Score Over Time ─────────────────────────────────────

def avg_score_trend_line(score_df: pd.DataFrame) -> go.Figure:
    """
    Line chart for mean VADER compound score per month.

    Parameters
    ----------
    score_df : Output of trend_analyzer.monthly_avg_score()
    """
    if score_df.empty:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x             = score_df["month"],
        y             = score_df["avg_sentiment_score"],
        mode          = "lines+markers",
        fill          = "tozeroy",
        fillcolor     = "rgba(52,152,219,0.12)",
        line          = dict(color="#3498db", width=2.5),
        marker        = dict(size=6, color="#3498db"),
        name          = "Avg Sentiment Score",
        hovertemplate = "Month: %{x}<br>Score: %{y:.3f}<extra></extra>",
    ))

    # Add a zero reference line (neutral boundary)
    fig.add_hline(y=0, line_dash="dash", line_color="#aaa", annotation_text="Neutral baseline")

    return _apply_layout(fig, "Average Sentiment Score Over Time", height=380)


# ── 5. Rating Distribution Bar ────────────────────────────────────────────────

def rating_bar(rating_df: pd.DataFrame) -> go.Figure:
    """
    Horizontal bar chart for star rating distribution (1–5 stars).

    Parameters
    ----------
    rating_df : Output of trend_analyzer.rating_distribution()
    """
    if rating_df.empty:
        return go.Figure()

    stars  = [f"⭐ {int(r)} Star{'s' if r != 1 else ''}" for r in rating_df["rating"]]
    colors = ["#e74c3c", "#e67e22", "#f39c12", "#27ae60", "#2ecc71"][:len(stars)]

    fig = go.Figure(go.Bar(
        x             = rating_df["count"],
        y             = stars,
        orientation   = "h",
        marker_color  = colors,
        text          = rating_df["percentage"].apply(lambda p: f"{p}%"),
        textposition  = "outside",
        hovertemplate = "%{y}<br>Count: %{x}<extra></extra>",
    ))
    return _apply_layout(fig, "Rating Distribution", height=350)


# ── 6. Keyword TF-IDF Bar ─────────────────────────────────────────────────────

def keyword_bar(
    keywords: List[Tuple[str, float]],
    title: str = "Top Keywords (TF-IDF Score)",
    color: str = "#3498db",
) -> go.Figure:
    """
    Horizontal bar chart of top keywords and their TF-IDF scores.

    Parameters
    ----------
    keywords : List of (keyword, score) tuples.
    title    : Chart title string.
    color    : Bar colour (hex).
    """
    if not keywords:
        return go.Figure()

    words  = [k for k, _ in keywords]
    scores = [s for _, s in keywords]

    fig = go.Figure(go.Bar(
        x             = scores,
        y             = words,
        orientation   = "h",
        marker_color  = color,
        hovertemplate = "Keyword: %{y}<br>TF-IDF: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return _apply_layout(fig, title, height=max(350, len(keywords) * 22 + 100))


# ── 7. WordCloud ──────────────────────────────────────────────────────────────

def generate_wordcloud(
    texts: List[str],
    max_words: int = 100,
    colormap: str  = "viridis",
):
    """
    Generate a WordCloud as a Matplotlib figure.

    Parameters
    ----------
    texts    : List of cleaned text strings.
    max_words: Maximum number of words in the cloud.
    colormap : Matplotlib colormap name.

    Returns
    -------
    matplotlib.figure.Figure or None if wordcloud is not installed.
    """
    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend for server use
    except ImportError:
        logger.warning("wordcloud package not installed; skipping WordCloud.")
        return None

    combined = " ".join([t for t in texts if isinstance(t, str) and t.strip()])
    if not combined.strip():
        return None

    wc = WordCloud(
        width           = 800,
        height          = 400,
        max_words       = max_words,
        background_color= "white",
        colormap        = colormap,
        collocations    = False,
        prefer_horizontal= 0.8,
    ).generate(combined)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig


# ── 8. Net Sentiment Score by Category ───────────────────────────────────────

def net_sentiment_gauge_bar(category_df: pd.DataFrame) -> go.Figure:
    """
    Horizontal diverging bar chart showing Net Sentiment Score per category.
    NSS = (Positive − Negative) / total × 100

    Parameters
    ----------
    category_df : Output of trend_analyzer.category_sentiment_breakdown()
    """
    if category_df.empty or "net_sentiment_score" not in category_df.columns:
        return go.Figure()

    colors = [
        "#2ecc71" if v >= 0 else "#e74c3c"
        for v in category_df["net_sentiment_score"]
    ]

    fig = go.Figure(go.Bar(
        x             = category_df["net_sentiment_score"],
        y             = category_df["category"],
        orientation   = "h",
        marker_color  = colors,
        hovertemplate = "Category: %{y}<br>NSS: %{x:.1f}%<extra></extra>",
    ))
    fig.add_vline(x=0, line_width=1.5, line_color="#555")
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return _apply_layout(fig, "Net Sentiment Score by Category  (NSS = (Pos−Neg)/Total × 100%)", height=400)


# ── 9. Sentiment Heatmap (Category × Month) ───────────────────────────────────

def sentiment_heatmap(
    df: pd.DataFrame,
    category_column: str = "product_category",
    date_column:     str = "date",
) -> go.Figure:
    """
    Heatmap of Positive review percentage for each (category × month) cell.
    """
    if category_column not in df.columns or date_column not in df.columns \
            or "sentiment" not in df.columns:
        return go.Figure()

    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    df = df.dropna(subset=[date_column])
    df["month"] = df[date_column].dt.to_period("M").astype(str)

    pivot = (
        df.groupby([category_column, "month"])
          .apply(lambda g: round((g["sentiment"] == "Positive").mean() * 100, 1))
          .unstack(fill_value=0)
          .reset_index()
          .set_index(category_column)
    )

    fig = go.Figure(go.Heatmap(
        z             = pivot.values,
        x             = pivot.columns.tolist(),
        y             = pivot.index.tolist(),
        colorscale    = "RdYlGn",
        zmin          = 0,
        zmax          = 100,
        hovertemplate = "Category: %{y}<br>Month: %{x}<br>Positive %%: %{z}<extra></extra>",
        colorbar      = dict(title="% Positive"),
    ))
    return _apply_layout(fig, "Positive Sentiment % — Category × Month Heatmap", height=420)


# ── 10. Confidence Score Distribution ────────────────────────────────────────

def confidence_histogram(df: pd.DataFrame) -> go.Figure:
    """Histogram of VADER confidence scores split by sentiment label."""
    if "confidence" not in df.columns or "sentiment" not in df.columns:
        return go.Figure()

    fig = go.Figure()
    for sentiment in ["Positive", "Neutral", "Negative"]:
        subset = df[df["sentiment"] == sentiment]["confidence"]
        fig.add_trace(go.Histogram(
            x         = subset,
            name      = sentiment,
            opacity   = 0.7,
            marker_color= _SENTIMENT_COLORS[sentiment],
            nbinsx    = 20,
            hovertemplate= f"<b>{sentiment}</b><br>Confidence: %{{x:.2f}}<br>Count: %{{y}}<extra></extra>",
        ))

    fig.update_layout(barmode="overlay")
    return _apply_layout(fig, "VADER Confidence Score Distribution by Sentiment", height=360)
