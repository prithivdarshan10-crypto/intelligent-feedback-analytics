import unittest

import pandas as pd

from src.import_sources import prepare_uploaded_feedback


class ImportSourcesTest(unittest.TestCase):
    def test_prepare_uploaded_feedback_detects_xquik_text_column(self):
        dataframe = pd.DataFrame(
            {
                "text": ["Love the export", " "],
                "createdAt": ["2026-07-05T01:00:00Z", "2026-07-05T01:02:00Z"],
                "source": ["xquik", "xquik"],
            }
        )

        prepared, text_column = prepare_uploaded_feedback(dataframe, "missing")

        self.assertEqual(text_column, "text")
        self.assertEqual(prepared["text"].to_list(), ["Love the export"])
        self.assertEqual(prepared["date"].to_list(), ["2026-07-05"])
        self.assertEqual(prepared["product_category"].to_list(), ["xquik"])

    def test_prepare_uploaded_feedback_rejects_missing_text_columns(self):
        dataframe = pd.DataFrame({"rating": [5]})

        with self.assertRaisesRegex(ValueError, "No feedback text column found"):
            prepare_uploaded_feedback(dataframe)


if __name__ == "__main__":
    unittest.main()
