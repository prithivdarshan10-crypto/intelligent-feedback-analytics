# =============================================================================
# database/db_manager.py
# SQLite persistence layer for the Feedback Analytics Platform.
# Uses Python's built-in sqlite3 module – no extra dependencies required.
# =============================================================================

import os
import sqlite3
import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from config.settings import DB_PATH

logger = logging.getLogger(__name__)


# ── SQL Definitions ───────────────────────────────────────────────────────────

_CREATE_FEEDBACK_TABLE = """
CREATE TABLE IF NOT EXISTS feedback (
    id                INTEGER  PRIMARY KEY AUTOINCREMENT,
    review_id         TEXT,
    customer_name     TEXT,
    product_category  TEXT,
    product_name      TEXT,
    rating            REAL,
    review_text       TEXT,
    cleaned_text      TEXT,
    date              TEXT,
    location          TEXT,
    verified_purchase TEXT,
    sentiment         TEXT,
    sentiment_score   REAL,
    sentiment_pos     REAL,
    sentiment_neu     REAL,
    sentiment_neg     REAL,
    confidence        REAL,
    processed_at      TEXT
);
"""

_CREATE_KEYWORDS_TABLE = """
CREATE TABLE IF NOT EXISTS keywords (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    session   TEXT,
    sentiment TEXT,
    keyword   TEXT,
    score     REAL,
    saved_at  TEXT
);
"""

_CREATE_SESSIONS_TABLE = """
CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT UNIQUE,
    filename    TEXT,
    row_count   INTEGER,
    created_at  TEXT
);
"""


# ── Connection helper ─────────────────────────────────────────────────────────

def _get_connection() -> sqlite3.Connection:
    """
    Open (and return) a connection to the SQLite database.
    Creates the DB file and parent directories if they don't exist.
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row   # Rows behave like dicts
    return conn


def _initialize_db() -> None:
    """Create all tables if they don't exist yet."""
    conn = _get_connection()
    try:
        cur = conn.cursor()
        cur.executescript(
            _CREATE_FEEDBACK_TABLE +
            _CREATE_KEYWORDS_TABLE +
            _CREATE_SESSIONS_TABLE
        )
        conn.commit()
        logger.info("Database initialised at: %s", DB_PATH)
    finally:
        conn.close()


# Initialise on import
_initialize_db()


# ── Session management ────────────────────────────────────────────────────────

def save_session(session_id: str, filename: str, row_count: int) -> None:
    """Record a new upload session."""
    conn = _get_connection()
    try:
        conn.execute(
            """INSERT OR REPLACE INTO sessions
               (session_id, filename, row_count, created_at)
               VALUES (?, ?, ?, ?)""",
            (session_id, filename, row_count, datetime.now().isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def list_sessions() -> pd.DataFrame:
    """Return all upload sessions as a DataFrame."""
    conn = _get_connection()
    try:
        return pd.read_sql_query(
            "SELECT session_id, filename, row_count, created_at "
            "FROM sessions ORDER BY created_at DESC",
            conn,
        )
    finally:
        conn.close()


# ── Feedback data ─────────────────────────────────────────────────────────────

def save_feedback(df: pd.DataFrame, session_id: str = "default") -> int:
    """
    Persist a processed feedback DataFrame to the `feedback` table.

    Each row is inserted with a `processed_at` timestamp and the
    provided `session_id` is stored via the session table.

    Parameters
    ----------
    df         : Analysed DataFrame (must contain at minimum review_text).
    session_id : Identifier linking this batch to a sessions record.

    Returns
    -------
    int  Number of rows inserted.
    """
    if df.empty:
        logger.warning("save_feedback called with empty DataFrame – nothing saved.")
        return 0

    df = df.copy()
    df["processed_at"] = datetime.now().isoformat()

    # Columns to persist (use only those that actually exist in df)
    desired_cols = [
        "review_id", "customer_name", "product_category", "product_name",
        "rating", "review_text", "cleaned_text", "date", "location",
        "verified_purchase", "sentiment", "sentiment_score",
        "sentiment_pos", "sentiment_neu", "sentiment_neg",
        "confidence", "processed_at",
    ]
    cols = [c for c in desired_cols if c in df.columns]
    subset = df[cols]

    conn = _get_connection()
    try:
        subset.to_sql("feedback", conn, if_exists="append", index=False)
        conn.commit()
        logger.info("Saved %d feedback rows to database.", len(subset))
        return len(subset)
    except Exception as exc:
        logger.error("Failed to save feedback: %s", exc)
        return 0
    finally:
        conn.close()


def load_feedback(limit: Optional[int] = None) -> pd.DataFrame:
    """
    Load persisted feedback from the database.

    Parameters
    ----------
    limit : Maximum rows to return (None = all).

    Returns
    -------
    pd.DataFrame  (empty DataFrame if table is empty or missing).
    """
    conn = _get_connection()
    try:
        query = "SELECT * FROM feedback ORDER BY id DESC"
        if limit:
            query += f" LIMIT {int(limit)}"
        return pd.read_sql_query(query, conn)
    except Exception as exc:
        logger.error("Failed to load feedback: %s", exc)
        return pd.DataFrame()
    finally:
        conn.close()


def get_feedback_count() -> int:
    """Return total number of rows in the feedback table."""
    conn = _get_connection()
    try:
        result = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()
        return result[0] if result else 0
    finally:
        conn.close()


# ── Keyword data ──────────────────────────────────────────────────────────────

def save_keywords(
    keywords_by_sentiment: dict,
    session_id: str = "default",
) -> None:
    """
    Persist extracted keywords (per sentiment label) to the `keywords` table.

    Parameters
    ----------
    keywords_by_sentiment : {label: [(keyword, score), …]}
    session_id            : Session identifier.
    """
    rows = []
    saved_at = datetime.now().isoformat()
    for sentiment, kw_list in keywords_by_sentiment.items():
        for keyword, score in kw_list:
            rows.append((session_id, sentiment, keyword, score, saved_at))

    if not rows:
        return

    conn = _get_connection()
    try:
        conn.executemany(
            "INSERT INTO keywords (session, sentiment, keyword, score, saved_at) "
            "VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
        logger.info("Saved %d keyword rows.", len(rows))
    finally:
        conn.close()


def load_keywords(session_id: Optional[str] = None) -> pd.DataFrame:
    """Load keywords from database, optionally filtered by session."""
    conn = _get_connection()
    try:
        if session_id:
            return pd.read_sql_query(
                "SELECT * FROM keywords WHERE session = ? ORDER BY score DESC",
                conn, params=(session_id,),
            )
        return pd.read_sql_query(
            "SELECT * FROM keywords ORDER BY score DESC", conn
        )
    finally:
        conn.close()


# ── Analytics queries (used by the dashboard) ─────────────────────────────────

def sentiment_summary_from_db() -> pd.DataFrame:
    """Aggregate sentiment counts directly from the database."""
    conn = _get_connection()
    try:
        return pd.read_sql_query(
            """SELECT sentiment,
                      COUNT(*) AS count,
                      ROUND(AVG(sentiment_score), 4) AS avg_score
               FROM feedback
               GROUP BY sentiment
               ORDER BY count DESC""",
            conn,
        )
    finally:
        conn.close()


def category_summary_from_db() -> pd.DataFrame:
    """Sentiment breakdown per product category from the database."""
    conn = _get_connection()
    try:
        return pd.read_sql_query(
            """SELECT product_category,
                      sentiment,
                      COUNT(*) AS count
               FROM feedback
               GROUP BY product_category, sentiment
               ORDER BY product_category, sentiment""",
            conn,
        )
    finally:
        conn.close()


def clear_all_data() -> None:
    """
    ⚠️ Delete ALL rows from feedback, keywords, and sessions tables.
    Used only from the admin panel in the dashboard.
    """
    conn = _get_connection()
    try:
        conn.executescript(
            "DELETE FROM feedback; DELETE FROM keywords; DELETE FROM sessions;"
        )
        conn.commit()
        logger.warning("All data cleared from database.")
    finally:
        conn.close()
