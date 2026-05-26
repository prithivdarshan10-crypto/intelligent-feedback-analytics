# =============================================================================
# src/report_generator.py
# Export analysed data as CSV or multi-sheet Excel workbooks.
# =============================================================================

import io
import os
import logging
from datetime import datetime

import pandas as pd

from config.settings import EXPORTS_DIR

logger = logging.getLogger(__name__)


def _ensure_exports_dir() -> None:
    os.makedirs(EXPORTS_DIR, exist_ok=True)


# ── CSV Export ────────────────────────────────────────────────────────────────

def export_csv(df: pd.DataFrame, filename: str = "feedback_analysis.csv") -> bytes:
    """
    Convert a DataFrame to CSV bytes for Streamlit's download button.

    Parameters
    ----------
    df       : Processed DataFrame to export.
    filename : Target filename (not used for bytes, just for reference).

    Returns
    -------
    bytes  UTF-8 encoded CSV content.
    """
    return df.to_csv(index=False).encode("utf-8")


# ── Excel Export ──────────────────────────────────────────────────────────────

def export_excel(
    df: pd.DataFrame,
    trend_df:    pd.DataFrame,
    category_df: pd.DataFrame,
    keywords_dict: dict,
) -> bytes:
    """
    Build a multi-sheet Excel workbook and return its binary content.

    Sheets
    ------
    1. Raw Analysis       – every row of the processed DataFrame
    2. Summary Statistics – high-level sentiment counts + percentages
    3. Monthly Trends     – monthly sentiment trend table
    4. Category Breakdown – per-category sentiment breakdown
    5. Top Keywords       – keywords per sentiment label
    6. Metadata           – report generation info

    Parameters
    ----------
    df             : Main processed feedback DataFrame.
    trend_df       : Output of trend_analyzer.monthly_sentiment_trend().
    category_df    : Output of trend_analyzer.category_sentiment_breakdown().
    keywords_dict  : {label: [(keyword, score), …]} from keyword_extractor.

    Returns
    -------
    bytes  .xlsx file content, ready for Streamlit download button.
    """
    buffer = io.BytesIO()

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:

        # ── Sheet 1: Raw Analysis ─────────────────────────────────────────────
        # Choose the most readable columns for the report
        cols_to_include = [
            "review_id", "customer_name", "product_category", "product_name",
            "rating", "review_text", "sentiment", "sentiment_score",
            "confidence", "date", "location",
        ]
        export_cols = [c for c in cols_to_include if c in df.columns]
        df[export_cols].to_excel(writer, sheet_name="Raw Analysis", index=False)

        # ── Sheet 2: Summary Statistics ───────────────────────────────────────
        total = len(df)
        summary_rows = []
        if "sentiment" in df.columns:
            counts = df["sentiment"].value_counts()
            for label in ["Positive", "Negative", "Neutral"]:
                cnt = counts.get(label, 0)
                summary_rows.append({
                    "Metric":     f"{label} Reviews",
                    "Value":      cnt,
                    "Percentage": f"{cnt / total * 100:.1f}%",
                })
        if "rating" in df.columns:
            summary_rows.append({
                "Metric": "Average Star Rating",
                "Value":  round(df["rating"].mean(), 2),
                "Percentage": "—",
            })
        if "sentiment_score" in df.columns:
            summary_rows.append({
                "Metric": "Average VADER Score",
                "Value":  round(df["sentiment_score"].mean(), 4),
                "Percentage": "—",
            })
        summary_rows.append({
            "Metric": "Total Reviews",
            "Value":  total,
            "Percentage": "100%",
        })
        pd.DataFrame(summary_rows).to_excel(
            writer, sheet_name="Summary Statistics", index=False
        )

        # ── Sheet 3: Monthly Trends ────────────────────────────────────────────
        if not trend_df.empty:
            trend_df.to_excel(writer, sheet_name="Monthly Trends", index=False)

        # ── Sheet 4: Category Breakdown ───────────────────────────────────────
        if not category_df.empty:
            category_df.to_excel(writer, sheet_name="Category Breakdown", index=False)

        # ── Sheet 5: Top Keywords ─────────────────────────────────────────────
        kw_rows = []
        for label, kw_list in keywords_dict.items():
            for keyword, score in kw_list:
                kw_rows.append({
                    "Sentiment": label,
                    "Keyword":   keyword,
                    "TF-IDF Score": round(score, 5),
                })
        if kw_rows:
            pd.DataFrame(kw_rows).to_excel(
                writer, sheet_name="Top Keywords", index=False
            )

        # ── Sheet 6: Metadata ─────────────────────────────────────────────────
        meta = pd.DataFrame([
            {"Key": "Report Generated At", "Value": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
            {"Key": "Total Reviews Analysed", "Value": total},
            {"Key": "Platform", "Value": "Intelligent Customer Feedback Analytics Platform"},
            {"Key": "Author",   "Value": "Automated Report – Powered by VADER + Scikit-learn"},
        ])
        meta.to_excel(writer, sheet_name="Metadata", index=False)

    buffer.seek(0)
    return buffer.read()


# ── Save to disk (optional) ───────────────────────────────────────────────────

def save_report_to_disk(content: bytes, filename: str) -> str:
    """
    Save exported report bytes to the exports/ directory.

    Returns the absolute path of the saved file.
    """
    _ensure_exports_dir()
    path = os.path.join(EXPORTS_DIR, filename)
    with open(path, "wb") as f:
        f.write(content)
    logger.info("Report saved to %s", path)
    return path


# ── Summary text for display / README ────────────────────────────────────────

def generate_text_summary(df: pd.DataFrame) -> str:
    """
    Create a plain-text summary paragraph of the analysis results.
    Used in the dashboard's 'Quick Summary' card.
    """
    if df.empty:
        return "No data available for summary."

    total  = len(df)
    counts = df["sentiment"].value_counts() if "sentiment" in df.columns else {}
    pos    = counts.get("Positive", 0)
    neg    = counts.get("Negative", 0)
    neu    = counts.get("Neutral",  0)
    avg_r  = round(df["rating"].mean(), 2) if "rating" in df.columns else "N/A"
    avg_s  = round(df["sentiment_score"].mean(), 4) if "sentiment_score" in df.columns else "N/A"

    dominant = counts.idxmax() if len(counts) > 0 else "Unknown"

    return (
        f"📋 **Analysis Summary**\n\n"
        f"- **{total}** customer reviews analysed\n"
        f"- **{pos}** Positive ({pos/total*100:.1f}%), "
        f"**{neg}** Negative ({neg/total*100:.1f}%), "
        f"**{neu}** Neutral ({neu/total*100:.1f}%)\n"
        f"- Dominant sentiment: **{dominant}**\n"
        f"- Average star rating: **{avg_r} ⭐**\n"
        f"- Average VADER score: **{avg_s}** (range −1 to +1)\n"
    )
