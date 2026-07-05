import pandas as pd

from config.settings import DEFAULT_CATEGORY_COL, DEFAULT_DATE_COL, DEFAULT_TEXT_COL


TEXT_COLUMN_CANDIDATES = (
    DEFAULT_TEXT_COL,
    "review",
    "review_body",
    "full_review",
    "full_text",
    "tweet_text",
    "comment",
    "message",
    "feedback",
    "content",
    "body",
    "text",
)
DATE_COLUMN_CANDIDATES = (
    "createdAt",
    "created_at",
    "createdat",
    "timestamp",
    "posted_at",
    "published_at",
    DEFAULT_DATE_COL,
)
CATEGORY_COLUMN_CANDIDATES = ("source", "platform", "tool", "topic", DEFAULT_CATEGORY_COL)


def _detect_column(columns, candidates):
    lower_to_original = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    for column in columns:
        normalized = column.lower()
        if any(candidate.lower() in normalized for candidate in candidates):
            return column
    return None


def prepare_uploaded_feedback(dataframe, requested_text_column=DEFAULT_TEXT_COL):
    prepared = dataframe.copy()
    text_column = (
        requested_text_column
        if requested_text_column in prepared.columns
        else _detect_column(prepared.columns, TEXT_COLUMN_CANDIDATES)
    )

    if text_column is None:
        raise ValueError(
            "No feedback text column found. Available columns: "
            + ", ".join(str(column) for column in prepared.columns)
        )

    text_values = prepared[text_column].fillna("").astype(str).str.strip()
    prepared = prepared[text_values != ""].copy()
    if prepared.empty:
        raise ValueError(f"Column '{text_column}' does not contain usable feedback text.")

    if DEFAULT_DATE_COL not in prepared.columns:
        date_column = _detect_column(prepared.columns, DATE_COLUMN_CANDIDATES)
        if date_column is not None:
            prepared[DEFAULT_DATE_COL] = (
                pd.to_datetime(prepared[date_column], errors="coerce")
                .dt.strftime("%Y-%m-%d")
                .fillna("")
            )

    if DEFAULT_CATEGORY_COL not in prepared.columns:
        category_column = _detect_column(prepared.columns, CATEGORY_COLUMN_CANDIDATES)
        if category_column is not None:
            prepared[DEFAULT_CATEGORY_COL] = (
                prepared[category_column].fillna("Imported Feedback").astype(str).str.strip()
            )

    return prepared, text_column
