import unittest

import numpy as np

from viralscope.test_inference import summarize_predictions


class TestInferenceTests(unittest.TestCase):
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