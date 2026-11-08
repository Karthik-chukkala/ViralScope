import unittest

import pandas as pd

from viralscope.data_quality import audit_data_quality
from viralscope.sample_cache import CACHE_COLUMNS


class DataQualityTests(unittest.TestCase):
    def test_quality_report_checks_schema_ranges_and_duplicate_groups_without_rows(self) -> None:
        rows = [
            {
                "Postdate": "1451606400",
                "Category": "Food",
                "Subcategory": "Dessert",
                "Concept": "berries",
                "Title": "Berry tart",
                "Alltags": "fresh berries",
                "label": 6.0,
            },
            {
                "Postdate": "1451606400",
                "Category": "Food",
                "Subcategory": "Dessert",
                "Concept": "berries",
                "Title": "Berry tart",
                "Alltags": "fresh berries",
                "label": 6.0,
            },
            {
                "Postdate": "1451606400",
                "Category": "Food",
                "Subcategory": "Dessert",
                "Concept": "berries",
                "Title": "Berry tart",
                "Alltags": "fresh berries",
                "label": 7.0,
            },
            {
                "Postdate": "invalid",
                "Category": "",
                "Subcategory": "",
                "Concept": "",
                "Title": "",
                "Alltags": "",
                "label": -1.0,
            },
        ]
        frame = pd.DataFrame(rows, columns=CACHE_COLUMNS)

        report = audit_data_quality(frame)

        self.assertEqual(report["rows"], 4)
        self.assertEqual(report["timestamps"]["nonnumeric_count"], 1)
        self.assertEqual(report["timestamps"]["invalid_datetime_count"], 1)
        self.assertEqual(report["labels"]["negative_count"], 1)
        self.assertEqual(report["duplicates"]["exact_duplicate_rows"], 1)
        self.assertEqual(report["duplicates"]["duplicate_input_rows"], 2)
        self.assertEqual(
            report["duplicates"]["duplicate_input_groups_with_conflicting_labels"], 1
        )
        self.assertEqual(report["schema"]["Title"]["blank_count"], 1)
        self.assertFalse(report["privacy"]["raw_text_written"])
        self.assertNotIn("Berry tart", str(report))

    def test_quality_audit_rejects_missing_columns_and_empty_data(self) -> None:
        with self.assertRaisesRegex(ValueError, "Missing required training columns"):
            audit_data_quality(pd.DataFrame({"Title": ["example"]}))
        with self.assertRaisesRegex(ValueError, "empty training frame"):
            audit_data_quality(pd.DataFrame(columns=CACHE_COLUMNS))


if __name__ == "__main__":
    unittest.main()
