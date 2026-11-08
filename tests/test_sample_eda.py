import unittest

import pandas as pd

from viralscope.sample_eda import summarize_sample


class SampleEdaTests(unittest.TestCase):
    def test_summary_is_aggregate_only_and_tracks_text_time_quality(self) -> None:
        frame = pd.DataFrame(
            {
                "Postdate": ["1451606400", "invalid", "1454284800"],
                "Category": ["Nature", None, "Food"],
                "Title": ["A red rose", "", None],
                "Alltags": ["red flower garden", None, "fresh food"],
                "label": [5.0, 7.0, -1.0],
            }
        )

        report, _ = summarize_sample(frame)

        self.assertEqual(report["rows"], 3)
        self.assertEqual(report["missing_or_blank"]["Title"], 2)
        self.assertEqual(report["missing_or_blank"]["Alltags"], 1)
        self.assertEqual(report["invalid_timestamp_count"], 1)
        self.assertEqual(report["monthly_posts_utc"][0], {"month": "2016-01", "count": 1})
        self.assertEqual(report["title_character_lengths"]["count"], 1)
        self.assertEqual(report["title_word_counts"]["mean"], 3.0)
        self.assertEqual(report["tag_token_counts"]["median"], 2.5)
        self.assertEqual(report["label_summary"]["count"], 2)
        self.assertNotIn("A red rose", str(report))
        self.assertFalse(report["privacy"]["raw_text_written"])


if __name__ == "__main__":
    unittest.main()