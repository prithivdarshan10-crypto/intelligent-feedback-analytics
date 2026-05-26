# =============================================================================
# tests/test_pipeline.py
# Unit tests for each module in the NLP pipeline.
# Run with:  pytest tests/ -v
# =============================================================================

import sys
import os
import pytest
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Download NLTK data before tests
import nltk
for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4", "vader_lexicon"]:
    nltk.download(pkg, quiet=True)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample_df():
    """Load and preprocess the sample dataset once for all tests."""
    from src.preprocessor       import preprocess_dataframe
    from src.sentiment_analyzer import analyze_dataframe
    df = pd.read_csv("dataset/sample_feedback.csv")
    df = preprocess_dataframe(df, "review_text")
    df = analyze_dataframe(df, "review_text")
    return df


# ── Preprocessor tests ────────────────────────────────────────────────────────

class TestPreprocessor:
    def test_clean_text_lowercase(self):
        from src.preprocessor import clean_text
        assert clean_text("HELLO WORLD") == clean_text("hello world")

    def test_clean_text_removes_url(self):
        from src.preprocessor import clean_text
        result = clean_text("Visit http://example.com for details")
        assert "http" not in result

    def test_clean_text_empty_string(self):
        from src.preprocessor import clean_text
        assert clean_text("") == ""
        assert clean_text(None) == ""

    def test_preprocess_adds_cleaned_column(self, sample_df):
        assert "cleaned_text" in sample_df.columns

    def test_preprocess_row_count_preserved(self, sample_df):
        assert len(sample_df) == 250


# ── Sentiment Analyzer tests ──────────────────────────────────────────────────

class TestSentimentAnalyzer:
    def test_positive_review_label(self):
        from src.sentiment_analyzer import analyze_single
        result = analyze_single("I absolutely love this product! It is amazing and excellent!")
        assert result["label"] == "Positive"

    def test_negative_review_label(self):
        from src.sentiment_analyzer import analyze_single
        result = analyze_single("Terrible product. Broke immediately. Worst purchase ever.")
        assert result["label"] == "Negative"

    def test_score_range(self):
        from src.sentiment_analyzer import analyze_single
        result = analyze_single("This is okay I guess.")
        assert -1.0 <= result["compound"] <= 1.0

    def test_sentiment_column_exists(self, sample_df):
        assert "sentiment"       in sample_df.columns
        assert "sentiment_score" in sample_df.columns

    def test_all_labels_valid(self, sample_df):
        valid = {"Positive", "Negative", "Neutral"}
        assert set(sample_df["sentiment"].unique()).issubset(valid)

    def test_summary_totals(self, sample_df):
        from src.sentiment_analyzer import get_sentiment_summary
        s = get_sentiment_summary(sample_df)
        assert s["positive"] + s["negative"] + s["neutral"] == s["total"]


# ── Keyword Extractor tests ───────────────────────────────────────────────────

class TestKeywordExtractor:
    def test_returns_list(self, sample_df):
        from src.keyword_extractor import extract_keywords_tfidf
        kw = extract_keywords_tfidf(sample_df["cleaned_text"].tolist(), top_n=10)
        assert isinstance(kw, list)
        assert len(kw) <= 10

    def test_tuple_format(self, sample_df):
        from src.keyword_extractor import extract_keywords_tfidf
        kw = extract_keywords_tfidf(sample_df["cleaned_text"].tolist(), top_n=5)
        for word, score in kw:
            assert isinstance(word, str)
            assert isinstance(score, float)

    def test_empty_input(self):
        from src.keyword_extractor import extract_keywords_tfidf
        result = extract_keywords_tfidf([])
        assert result == []

    def test_by_sentiment_keys(self, sample_df):
        from src.keyword_extractor import keywords_by_sentiment
        result = keywords_by_sentiment(sample_df, top_n=5)
        assert "Positive" in result
        assert "Negative" in result


# ── Trend Analyzer tests ──────────────────────────────────────────────────────

class TestTrendAnalyzer:
    def test_monthly_trend_has_12_months(self, sample_df):
        from src.trend_analyzer import monthly_sentiment_trend
        trend = monthly_sentiment_trend(sample_df)
        assert len(trend) == 12   # sample data spans Jan–Dec 2024

    def test_category_breakdown_has_all_categories(self, sample_df):
        from src.trend_analyzer import category_sentiment_breakdown
        cats = category_sentiment_breakdown(sample_df)
        assert len(cats) == 6

    def test_net_sentiment_score_column_exists(self, sample_df):
        from src.trend_analyzer import category_sentiment_breakdown
        cats = category_sentiment_breakdown(sample_df)
        assert "net_sentiment_score" in cats.columns


# ── Database tests ────────────────────────────────────────────────────────────

class TestDatabase:
    def test_save_and_load(self, sample_df, tmp_path, monkeypatch):
        from config import settings
        monkeypatch.setattr(settings, "DB_PATH", str(tmp_path / "test.db"))

        import importlib
        import database.db_manager as dbm
        importlib.reload(dbm)

        saved = dbm.save_feedback(sample_df.head(10), "pytest_session")
        assert saved == 10
        loaded = dbm.load_feedback()
        assert len(loaded) == 10
