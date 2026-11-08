import unittest

import numpy as np
import pandas as pd

from viralscope.test_inference import summarize_predictions, summarize_unlabeled_inputs


class TestInferenceTests(unittest.TestCase):
    def test_summarize_unlabeled_test_inputs_excludes_targets_and_ids(self) -> None:
        frame = pd.DataFrame(
            {
                "Postdate": ["1451606400", "1454284800"],
                "Category": ["Food", "Urban"],
                "Title": ["Berry tart", ""],
                "Alltags": ["fresh sweet", "city lights"],
            }
        )

        summary = summarize_unlabeled_inputs(frame)

        self.assertEqual(summary["rows"], 2)
        self.assertEqual(summary["missing_or_blank"]["Title"], 1)
        self.assertEqual(len(summary["monthly_posts_utc"]), 2)
        self.assertEqual(summary["category_frequencies"][0], {"category": "Food", "count": 1})
        self.assertFalse(summary["labels_read"])
        self.assertFalse(summary["ids_read"])

    def test_summarize_predictions_reports_distribution_without_targets(self) -> None:
        summary = summarize_predictions(np.array([1.0, 2.0, 3.0, 4.0]))

        self.assertEqual(summary["rows"], 4)
        self.assertEqual(summary["min"], 1.0)
        self.assertEqual(summary["median"], 2.5)
        self.assertEqual(summary["max"], 4.0)

    def test_summarize_predictions_counts_values_outside_training_range(self) -> None:
        summary = summarize_predictions(
            np.array([0.0, 2.0, 8.0, 20.0]),
            training_label_range=(1.0, 16.56),
        )

        self.assertEqual(summary["outside_train_label_range_count"], 2)

    def test_summarize_predictions_rejects_empty_or_nonfinite_values(self) -> None:
        for values in (np.array([]), np.array([1.0, np.nan])):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    summarize_predictions(values)


if __name__ == "__main__":
    unittest.main()