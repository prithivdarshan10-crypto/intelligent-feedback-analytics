# =============================================================================
# config/settings.py
# Central configuration for the Intelligent Customer Feedback Analytics Platform
# =============================================================================

import os

# ── Base paths ─────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR     = os.path.join(BASE_DIR, "dataset")
DATABASE_DIR    = os.path.join(BASE_DIR, "database")
EXPORTS_DIR     = os.path.join(BASE_DIR, "exports")
MODELS_DIR      = os.path.join(BASE_DIR, "models")
VIZ_DIR         = os.path.join(BASE_DIR, "visualizations")

# ── Database ───────────────────────────────────────────────────────────────────
DB_PATH         = os.path.join(DATABASE_DIR, "feedback_analytics.db")

# ── Sentiment thresholds (VADER compound score) ────────────────────────────────
# compound score  >=  POSITIVE_THRESHOLD  → Positive
# compound score  <=  NEGATIVE_THRESHOLD  → Negative
# anything between                        → Neutral
POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05

# ── Keyword extraction ─────────────────────────────────────────────────────────
TOP_N_KEYWORDS   = 20    # Number of keywords to extract per analysis
MAX_NGRAM        = 2     # Maximum n-gram size for keyword phrases (1 = unigrams, 2 = bigrams)

# ── Colour palette (used across all Plotly charts) ────────────────────────────
COLORS = {
    "positive": "#2ecc71",   # Green
    "negative": "#e74c3c",   # Red
    "neutral":  "#f39c12",   # Amber
    "primary":  "#2c3e50",   # Dark navy
    "secondary":"#3498db",   # Blue
    "background":"#f8f9fa",  # Light grey
}

SENTIMENT_PALETTE = [
    COLORS["positive"],
    COLORS["negative"],
    COLORS["neutral"],
]

# ── Expected CSV column names (users may upload their own files) ───────────────
# The platform tries to auto-detect columns matching these keys.
DEFAULT_TEXT_COL     = "review_text"
DEFAULT_DATE_COL     = "date"
DEFAULT_CATEGORY_COL = "product_category"
DEFAULT_RATING_COL   = "rating"

# ── Report settings ────────────────────────────────────────────────────────────
REPORT_FILENAME = "feedback_analytics_report.xlsx"

# ── Streamlit page config ─────────────────────────────────────────────────────
PAGE_TITLE = "Customer Feedback Analytics Platform"
PAGE_ICON  = "📊"
LAYOUT     = "wide"
