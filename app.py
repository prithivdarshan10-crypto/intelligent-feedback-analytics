# =============================================================================
# app.py  –  Intelligent Customer Feedback Analytics Platform
# Main Streamlit dashboard entry-point.
#
# Run with:
#   streamlit run app.py
# =============================================================================

import os
import sys
import uuid
import logging
import warnings

import numpy  as np
import pandas as pd
import streamlit as st

# ── Add project root to Python path so sub-packages resolve correctly ─────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Project imports ───────────────────────────────────────────────────────────
from config.settings          import PAGE_TITLE, PAGE_ICON, LAYOUT, DEFAULT_TEXT_COL
from src.preprocessor         import preprocess_dataframe, get_text_stats
from src.sentiment_analyzer   import analyze_dataframe, get_sentiment_summary, train_ml_model
from src.keyword_extractor    import keywords_by_sentiment, extract_keywords_tfidf
from src.trend_analyzer       import (
    monthly_sentiment_trend, monthly_avg_score,
    category_sentiment_breakdown, rating_distribution, top_bottom_products,
)
from src.visualizer           import (
    sentiment_pie, sentiment_by_category_bar, monthly_trend_line,
    avg_score_trend_line, rating_bar, keyword_bar, generate_wordcloud,
    net_sentiment_gauge_bar, sentiment_heatmap, confidence_histogram,
)
from src.report_generator     import (
    export_csv, export_excel, generate_text_summary,
)
from database.db_manager      import (
    save_feedback, load_feedback, get_feedback_count,
    save_keywords, list_sessions, clear_all_data, sentiment_summary_from_db,
)

# ── Streamlit page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title = PAGE_TITLE,
    page_icon  = PAGE_ICON,
    layout     = LAYOUT,
    initial_sidebar_state = "expanded",
)

# ── Custom CSS (minimal, clean styling) ───────────────────────────────────────
st.markdown("""
<style>
  /* KPI cards */
  .kpi-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    color: white;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
  }
  .kpi-value  { font-size: 2.2rem; font-weight: 700; margin: 6px 0; }
  .kpi-label  { font-size: 0.85rem; opacity: 0.75; letter-spacing: 0.5px; }
  .kpi-delta  { font-size: 0.9rem;  margin-top: 4px; }

  /* Section divider */
  .section-title {
    font-size: 1.2rem; font-weight: 600;
    color: #2c3e50;
    border-left: 4px solid #3498db;
    padding-left: 10px;
    margin: 20px 0 12px;
  }

  /* Sidebar tweaks */
  [data-testid="stSidebar"] { background: #0f3460; }
  [data-testid="stSidebar"] * { color: white !important; }
  [data-testid="stSidebar"] .stRadio label { color: white !important; }
  [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.2); }

  /* Remove default top padding */
  .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE INITIALISATION
# =============================================================================

def _init_state():
    """Initialise all session state variables to their defaults."""
    defaults = {
        "df_raw":       None,   # Uploaded raw DataFrame
        "df_processed": None,   # After preprocessing + sentiment analysis
        "session_id":   str(uuid.uuid4())[:8],
        "filename":     None,
        "analysis_done": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("## 📊 Feedback Analytics")
    st.markdown("*Intelligent NLP-Powered Platform*")
    st.markdown("---")

    # ── Navigation ────────────────────────────────────────────────────────────
    page = st.radio(
        "Navigate to",
        options=[
            "🏠  Home & Upload",
            "📊  Analysis Dashboard",
            "🔑  Keywords & Topics",
            "📈  Trends & Insights",
            "🗄️  Database Explorer",
            "📥  Export Report",
            "ℹ️  About",
        ],
        key="nav_page",
    )

    st.markdown("---")

    # ── Upload widget (always visible) ────────────────────────────────────────
    st.markdown("### 📂 Upload Dataset")
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        help="CSV must contain a text column (e.g. review_text). "
             "You can use the sample dataset in /dataset folder.",
    )

    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file)
            st.session_state["df_raw"]   = df_raw
            st.session_state["filename"] = uploaded_file.name
            st.session_state["analysis_done"] = False   # reset on new upload
            st.success(f"✅ Loaded **{len(df_raw)}** rows")
        except Exception as e:
            st.error(f"❌ Could not read CSV: {e}")

    # ── Run Analysis button ────────────────────────────────────────────────────
    st.markdown("### ⚡ Run Analysis")
    text_col = st.text_input(
        "Text column name",
        value=DEFAULT_TEXT_COL,
        help="Name of the column containing review text.",
    )

    run_btn = st.button("🚀 Analyse Now", use_container_width=True, type="primary")

    if run_btn:
        if st.session_state["df_raw"] is None:
            st.warning("⚠️ Please upload a CSV first.")
        else:
            with st.spinner("Running NLP pipeline …"):
                try:
                    df = st.session_state["df_raw"].copy()

                    # Auto-detect text column if the specified one is absent
                    if text_col not in df.columns:
                        candidates = [c for c in df.columns
                                      if "text" in c.lower() or "review" in c.lower()
                                         or "comment" in c.lower() or "feedback" in c.lower()]
                        if candidates:
                            text_col = candidates[0]
                            st.info(f"ℹ️ Auto-detected text column: **{text_col}**")
                        else:
                            st.error(f"Column '{text_col}' not found. "
                                     f"Available: {list(df.columns)}")
                            st.stop()

                    # Step 1 – preprocess
                    df = preprocess_dataframe(df, text_col)

                    # Step 2 – sentiment analysis (VADER)
                    df = analyze_dataframe(df, text_col)

                    # Step 3 – persist to DB
                    sid = st.session_state["session_id"]
                    save_feedback(df, sid)
                    save_session_fn = lambda: None  # already in db_manager import

                    # Step 4 – save keywords
                    kw = keywords_by_sentiment(df, top_n=20)
                    save_keywords(kw, sid)

                    # Step 5 – optional ML model training
                    if len(df) >= 30:
                        train_ml_model(df, "cleaned_text")

                    st.session_state["df_processed"] = df
                    st.session_state["analysis_done"] = True
                    st.success("✅ Analysis complete!")

                except Exception as exc:
                    st.error(f"Pipeline error: {exc}")
                    logger.exception("Pipeline failure")

    st.markdown("---")
    db_count = get_feedback_count()
    st.caption(f"💾 **{db_count}** reviews in database")
    st.caption(f"🆔 Session: `{st.session_state['session_id']}`")


# =============================================================================
# HELPER: Guard clause for pages that need processed data
# =============================================================================

def _need_data(msg: str = "Run analysis first.") -> bool:
    """Return True (and show a warning) if no processed data is available."""
    if st.session_state.get("df_processed") is None:
        st.info(f"📂 {msg}")
        return True
    return False


# =============================================================================
# PAGE: HOME & UPLOAD
# =============================================================================

if "🏠" in page:
    st.title("🧠 Intelligent Customer Feedback Analytics Platform")
    st.markdown(
        "Welcome! This platform automatically analyses customer reviews using "
        "**Natural Language Processing (NLP)** and **Machine Learning** to reveal "
        "sentiment patterns, trending topics, and actionable product insights."
    )

    # ── Feature Cards ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🚀 Platform Features")

    cols = st.columns(3)
    features = [
        ("🧹", "Text Preprocessing",      "Cleans URLs, punctuation, stopwords, applies lemmatisation"),
        ("💬", "Sentiment Analysis",       "VADER rule-based + Logistic Regression ML classifier"),
        ("🔑", "Keyword Extraction",       "TF-IDF key-phrases per category and sentiment label"),
        ("📈", "Trend Analysis",           "Monthly/weekly sentiment trends & volume charts"),
        ("🗄️", "SQLite Persistence",       "Every analysis is stored for historical comparison"),
        ("📥", "Exportable Reports",       "Download full analysis as Excel or CSV"),
    ]
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(f"**{icon} {title}**\n\n{desc}")
            st.markdown("---")

    # ── How to use ────────────────────────────────────────────────────────────
    st.markdown("### 📋 Quick Start")
    st.markdown("""
1. **Upload CSV** using the sidebar uploader (or use the included `sample_feedback.csv`).
2. Set the **text column name** (default: `review_text`).
3. Click **🚀 Analyse Now** — the NLP pipeline runs automatically.
4. Explore **Analysis Dashboard**, **Keywords**, and **Trends** tabs.
5. **Export** your report as Excel or CSV.
""")

    # ── Dataset preview ───────────────────────────────────────────────────────
    if st.session_state["df_raw"] is not None:
        st.markdown("### 👁️ Uploaded Dataset Preview")
        df_raw = st.session_state["df_raw"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Rows",    len(df_raw))
        c2.metric("Total Columns", len(df_raw.columns))
        c3.metric("Filename",      st.session_state.get("filename", "—"))
        c4.metric("Memory",        f"{df_raw.memory_usage(deep=True).sum() / 1024:.1f} KB")
        st.dataframe(df_raw.head(10), use_container_width=True)

    # ── Sample dataset info ────────────────────────────────────────────────────
    sample_path = os.path.join(ROOT, "dataset", "sample_feedback.csv")
    if os.path.exists(sample_path):
        with st.expander("📄 Sample Dataset Info (dataset/sample_feedback.csv)"):
            sample = pd.read_csv(sample_path)
            st.write(f"**Rows:** {len(sample)}  |  **Columns:** {list(sample.columns)}")
            st.dataframe(sample.head(5), use_container_width=True)


# =============================================================================
# PAGE: ANALYSIS DASHBOARD
# =============================================================================

elif "📊" in page:
    st.title("📊 Sentiment Analysis Dashboard")

    if _need_data("Upload a CSV and click 🚀 Analyse Now to see the dashboard."):
        st.stop()

    df = st.session_state["df_processed"]
    summary = get_sentiment_summary(df)

    # ── KPI Row ────────────────────────────────────────────────────────────────
    st.markdown('<p class="section-title">Key Performance Indicators</p>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)

    kpi_data = [
        (c1, "Total Reviews",     summary.get("total",        0),   "#3498db", ""),
        (c2, "Positive",          f"{summary.get('positive_pct',0)}%", "#2ecc71", "😊"),
        (c3, "Negative",          f"{summary.get('negative_pct',0)}%", "#e74c3c", "😞"),
        (c4, "Neutral",           f"{summary.get('neutral_pct', 0)}%", "#f39c12", "😐"),
        (c5, "Avg VADER Score",   summary.get("avg_score",     0.0), "#9b59b6", ""),
    ]
    for col, label, value, color, emoji in kpi_data:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value" style="color:{color};">{emoji} {value}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")  # spacer

    # ── Charts Row 1 ──────────────────────────────────────────────────────────
    st.markdown('<p class="section-title">Sentiment Distribution</p>', unsafe_allow_html=True)
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.plotly_chart(sentiment_pie(df), use_container_width=True)

    with col_right:
        if "product_category" in df.columns:
            st.plotly_chart(sentiment_by_category_bar(df), use_container_width=True)
        else:
            st.info("No `product_category` column found for category breakdown.")

    # ── Rating Distribution ────────────────────────────────────────────────────
    if "rating" in df.columns:
        st.markdown('<p class="section-title">Star Rating Distribution</p>', unsafe_allow_html=True)
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            rd = rating_distribution(df)
            st.plotly_chart(rating_bar(rd), use_container_width=True)
        with col_r2:
            st.plotly_chart(confidence_histogram(df), use_container_width=True)

    # ── Text summary ──────────────────────────────────────────────────────────
    st.markdown('<p class="section-title">Quick Summary</p>', unsafe_allow_html=True)
    st.markdown(generate_text_summary(df))

    # ── Net Sentiment Score by Category ──────────────────────────────────────
    if "product_category" in df.columns:
        st.markdown('<p class="section-title">Net Sentiment Score by Category</p>',
                    unsafe_allow_html=True)
        cat_df = category_sentiment_breakdown(df)
        st.plotly_chart(net_sentiment_gauge_bar(cat_df), use_container_width=True)

    # ── Raw data table ────────────────────────────────────────────────────────
    with st.expander("🔍 View Processed Data Table"):
        display_cols = [
            c for c in ["review_id", "customer_name", "product_category",
                        "product_name", "rating", "review_text",
                        "sentiment", "sentiment_score", "confidence", "date"]
            if c in df.columns
        ]
        st.dataframe(df[display_cols], use_container_width=True)


# =============================================================================
# PAGE: KEYWORDS & TOPICS
# =============================================================================

elif "🔑" in page:
    st.title("🔑 Keyword & Topic Analysis")

    if _need_data("Run analysis first to see keyword insights."):
        st.stop()

    df = st.session_state["df_processed"]

    # ── Overall keywords ──────────────────────────────────────────────────────
    st.markdown('<p class="section-title">Top Keywords — All Reviews</p>', unsafe_allow_html=True)

    all_keywords = extract_keywords_tfidf(
        df["cleaned_text"].dropna().tolist(), top_n=20
    )
    st.plotly_chart(
        keyword_bar(all_keywords, "Top 20 Keywords Across All Reviews", "#3498db"),
        use_container_width=True,
    )

    # ── WordCloud ─────────────────────────────────────────────────────────────
    st.markdown('<p class="section-title">Word Cloud</p>', unsafe_allow_html=True)

    wc_sentiment = st.selectbox(
        "Generate word cloud for",
        ["All Reviews", "Positive Reviews", "Negative Reviews", "Neutral Reviews"],
    )

    wc_texts = df["cleaned_text"].dropna().tolist()
    if wc_sentiment != "All Reviews":
        label = wc_sentiment.split()[0]  # "Positive" / "Negative" / "Neutral"
        wc_texts = df[df["sentiment"] == label]["cleaned_text"].dropna().tolist()

    wc_fig = generate_wordcloud(wc_texts)
    if wc_fig:
        st.pyplot(wc_fig)
    else:
        st.info("Install `wordcloud` package (`pip install wordcloud`) to see the word cloud.")

    # ── Per-sentiment keywords ────────────────────────────────────────────────
    st.markdown('<p class="section-title">Keywords by Sentiment Label</p>', unsafe_allow_html=True)

    kw_by_sent = keywords_by_sentiment(df, top_n=15)
    tab_pos, tab_neg, tab_neu = st.tabs(["😊 Positive", "😞 Negative", "😐 Neutral"])

    with tab_pos:
        if kw_by_sent.get("Positive"):
            st.plotly_chart(
                keyword_bar(kw_by_sent["Positive"], "Top Keywords in Positive Reviews", "#2ecc71"),
                use_container_width=True,
            )
        else:
            st.info("No Positive reviews found.")

    with tab_neg:
        if kw_by_sent.get("Negative"):
            st.plotly_chart(
                keyword_bar(kw_by_sent["Negative"], "Top Keywords in Negative Reviews", "#e74c3c"),
                use_container_width=True,
            )
        else:
            st.info("No Negative reviews found.")

    with tab_neu:
        if kw_by_sent.get("Neutral"):
            st.plotly_chart(
                keyword_bar(kw_by_sent["Neutral"], "Top Keywords in Neutral Reviews", "#f39c12"),
                use_container_width=True,
            )
        else:
            st.info("No Neutral reviews found.")

    # ── Per-category keywords (expandable) ───────────────────────────────────
    if "product_category" in df.columns:
        st.markdown('<p class="section-title">Keywords by Product Category</p>',
                    unsafe_allow_html=True)

        selected_cat = st.selectbox(
            "Select a product category",
            sorted(df["product_category"].dropna().unique()),
        )
        cat_texts = df[df["product_category"] == selected_cat]["cleaned_text"].dropna().tolist()
        cat_kw    = extract_keywords_tfidf(cat_texts, top_n=15)

        if cat_kw:
            st.plotly_chart(
                keyword_bar(cat_kw, f"Top Keywords — {selected_cat}", "#9b59b6"),
                use_container_width=True,
            )
        else:
            st.info("Not enough data for TF-IDF extraction in this category.")


# =============================================================================
# PAGE: TRENDS & INSIGHTS
# =============================================================================

elif "📈" in page:
    st.title("📈 Trends & Insights")

    if _need_data("Run analysis first to see trend charts."):
        st.stop()

    df = st.session_state["df_processed"]

    if "date" not in df.columns:
        st.warning("⚠️ No `date` column found. Trend charts require a date column.")
        st.stop()

    # ── Monthly Sentiment Trend ───────────────────────────────────────────────
    st.markdown('<p class="section-title">Monthly Sentiment Trend</p>', unsafe_allow_html=True)
    trend_df = monthly_sentiment_trend(df)
    if not trend_df.empty:
        st.plotly_chart(monthly_trend_line(trend_df), use_container_width=True)
    else:
        st.info("Not enough date data for monthly trends.")

    # ── Average Score Over Time ───────────────────────────────────────────────
    st.markdown('<p class="section-title">Average Sentiment Score Over Time</p>',
                unsafe_allow_html=True)
    score_df = monthly_avg_score(df)
    if not score_df.empty:
        st.plotly_chart(avg_score_trend_line(score_df), use_container_width=True)

    # ── Heatmap ───────────────────────────────────────────────────────────────
    if "product_category" in df.columns:
        st.markdown('<p class="section-title">Sentiment Heatmap — Category × Month</p>',
                    unsafe_allow_html=True)
        st.plotly_chart(sentiment_heatmap(df), use_container_width=True)

    # ── Top & Bottom Products ─────────────────────────────────────────────────
    if "product_name" in df.columns:
        st.markdown('<p class="section-title">Best & Worst Reviewed Products</p>',
                    unsafe_allow_html=True)
        tb = top_bottom_products(df)

        c_top, c_bot = st.columns(2)
        with c_top:
            st.markdown("**🏆 Top Products (highest % Positive)**")
            if not tb["top"].empty:
                st.dataframe(tb["top"], use_container_width=True)

        with c_bot:
            st.markdown("**⚠️ Bottom Products (highest % Negative)**")
            if not tb["bottom"].empty:
                st.dataframe(tb["bottom"], use_container_width=True)

    # ── Category Breakdown Table ──────────────────────────────────────────────
    if "product_category" in df.columns:
        st.markdown('<p class="section-title">Category Breakdown Table</p>',
                    unsafe_allow_html=True)
        cat_df = category_sentiment_breakdown(df)
        st.dataframe(cat_df, use_container_width=True)


# =============================================================================
# PAGE: DATABASE EXPLORER
# =============================================================================

elif "🗄️" in page:
    st.title("🗄️ Database Explorer")
    st.markdown("Inspect all data stored in the local **SQLite** database.")

    total_rows = get_feedback_count()
    st.metric("Total Records in DB", total_rows)

    # ── View sessions ─────────────────────────────────────────────────────────
    st.markdown("### 📋 Upload Sessions")
    sessions = list_sessions()
    if not sessions.empty:
        st.dataframe(sessions, use_container_width=True)
    else:
        st.info("No sessions recorded yet.")

    # ── View feedback data ────────────────────────────────────────────────────
    st.markdown("### 💬 Stored Feedback Records")
    limit = st.slider("Rows to display", min_value=10, max_value=500, value=50, step=10)
    db_df = load_feedback(limit=limit)
    if not db_df.empty:
        st.dataframe(db_df, use_container_width=True)
    else:
        st.info("No feedback stored yet. Run analysis to populate the database.")

    # ── Aggregate stats from DB ───────────────────────────────────────────────
    st.markdown("### 📊 Sentiment Summary (from Database)")
    db_summary = sentiment_summary_from_db()
    if not db_summary.empty:
        st.dataframe(db_summary, use_container_width=True)

    # ── Clear data (danger zone) ──────────────────────────────────────────────
    with st.expander("⚠️ Danger Zone"):
        st.warning("This will permanently delete ALL data from the database.")
        confirm = st.checkbox("I understand. Delete all data.")
        if confirm and st.button("🗑️ Clear All Data", type="primary"):
            clear_all_data()
            st.session_state["df_processed"] = None
            st.session_state["analysis_done"] = False
            st.success("✅ All data cleared.")
            st.rerun()


# =============================================================================
# PAGE: EXPORT REPORT
# =============================================================================

elif "📥" in page:
    st.title("📥 Export Report")

    if _need_data("Run analysis first to generate an export."):
        st.stop()

    df = st.session_state["df_processed"]
    st.success(f"✅ Ready to export **{len(df)}** analysed reviews.")

    # ── CSV Download ──────────────────────────────────────────────────────────
    st.markdown("### 📄 Download as CSV")
    st.markdown("Lightweight flat-file format — best for further processing.")
    csv_bytes = export_csv(df)
    st.download_button(
        label      = "⬇️  Download CSV",
        data       = csv_bytes,
        file_name  = "feedback_analysis.csv",
        mime       = "text/csv",
        use_container_width=True,
    )

    # ── Excel Download ─────────────────────────────────────────────────────────
    st.markdown("### 📊 Download Full Excel Report")
    st.markdown(
        "Multi-sheet workbook containing: Raw Analysis, Summary Statistics, "
        "Monthly Trends, Category Breakdown, Top Keywords, and Metadata."
    )

    # Compute supporting tables
    trend_df    = monthly_sentiment_trend(df) if "date" in df.columns else pd.DataFrame()
    category_df = category_sentiment_breakdown(df) if "product_category" in df.columns else pd.DataFrame()
    kw_dict     = keywords_by_sentiment(df, top_n=20)

    xlsx_bytes = export_excel(df, trend_df, category_df, kw_dict)
    st.download_button(
        label      = "⬇️  Download Excel Report",
        data       = xlsx_bytes,
        file_name  = "feedback_analytics_report.xlsx",
        mime       = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    # ── Preview ────────────────────────────────────────────────────────────────
    with st.expander("👁️ Preview Data to be Exported (first 20 rows)"):
        preview_cols = [
            c for c in ["review_id", "customer_name", "product_category",
                        "review_text", "sentiment", "sentiment_score",
                        "confidence", "date"]
            if c in df.columns
        ]
        st.dataframe(df[preview_cols].head(20), use_container_width=True)


# =============================================================================
# PAGE: ABOUT
# =============================================================================

elif "ℹ️" in page:
    st.title("ℹ️ About This Platform")

    st.markdown("""
## 🧠 Intelligent Customer Feedback Analytics Platform

### What this project does
This platform ingests raw customer review CSV files and automatically produces:

| Feature | Technology |
|---|---|
| Text cleaning & preprocessing | NLTK (tokenisation, stopword removal, lemmatisation) |
| Sentiment analysis (rule-based) | VADER (Valence Aware Dictionary and sEntiment Reasoner) |
| Sentiment analysis (ML) | TF-IDF + Logistic Regression (scikit-learn) |
| Keyword & key-phrase extraction | TF-IDF vectorisation |
| Trend analysis | Pandas time-series groupby |
| Interactive visualisations | Plotly + Matplotlib/WordCloud |
| Data persistence | SQLite (via Python sqlite3) |
| Export reports | openpyxl (multi-sheet Excel) |
| Dashboard UI | Streamlit |

### Tech Stack
- **Language:** Python 3.9+
- **NLP:** NLTK, scikit-learn
- **Visualisation:** Plotly, Matplotlib, WordCloud
- **Database:** SQLite
- **Dashboard:** Streamlit

### Author
Built as an industry-level portfolio project demonstrating end-to-end
NLP engineering skills relevant to Data Science and Data Engineering roles.

### GitHub
[github.com/prithivdarshan10-crypto](https://github.com/prithivdarshan10-crypto)
""")

    st.markdown("---")
    st.markdown("### 📁 Project Structure")
    st.code("""
intelligent-feedback-analytics/
├── app.py                  ← Streamlit dashboard (this file)
├── requirements.txt        ← All dependencies
├── README.md               ← GitHub documentation
├── setup.py                ← NLTK data download helper
├── config/
│   └── settings.py         ← Global configuration
├── dataset/
│   └── sample_feedback.csv ← 250-row sample dataset
├── src/
│   ├── preprocessor.py     ← Text cleaning pipeline
│   ├── sentiment_analyzer.py ← VADER + ML sentiment
│   ├── keyword_extractor.py  ← TF-IDF keyword extraction
│   ├── trend_analyzer.py   ← Time-series analysis
│   ├── visualizer.py       ← Plotly/Matplotlib charts
│   └── report_generator.py ← CSV/Excel export
├── database/
│   └── db_manager.py       ← SQLite CRUD layer
├── models/                 ← Saved ML models (auto-populated)
├── visualizations/         ← Saved chart images
└── exports/                ← Generated reports
""", language="text")
