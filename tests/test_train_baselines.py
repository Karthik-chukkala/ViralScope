import unittest

import numpy as np
import pandas as pd

from viralscope.train_baselines import (
    _holdout_category_metrics,
    chronological_split,
    summarize_residuals,
    user_aware_split,
)


class TrainBaselineTests(unittest.TestCase):
    def test_user_aware_split_is_deterministic_and_has_no_group_overlap(self) -> None:
        features = pd.DataFrame({"value": range(30)})
        labels = np.arange(30, dtype=np.float64)
        groups = np.repeat(np.arange(10), 3)

        train_indexes, holdout_indexes = user_aware_split(features, labels, groups)
        repeated_train, repeated_holdout = user_aware_split(features, labels, groups)

        self.assertTrue(np.array_equal(train_indexes, repeated_train))
        self.assertTrue(np.array_equal(holdout_indexes, repeated_holdout))
        self.assertFalse(set(groups[train_indexes]) & set(groups[holdout_indexes]))

    def test_user_aware_split_rejects_misaligned_or_missing_groups(self) -> None:
        features = pd.DataFrame({"value": range(3)})
        labels = np.arange(3, dtype=np.float64)

        with self.assertRaises(ValueError):
            user_aware_split(features, labels, np.array(["a", "b"]))
        with self.assertRaises(ValueError):
            user_aware_split(features, labels, np.array(["a", None, "b"], dtype=object))

    def test_summarize_residuals_reports_signed_bias_and_absolute_error_bands(self) -> None:
        summary = summarize_residuals(
            np.array([1.0, 2.0, 3.0]),
            np.array([0.0, 2.0, 5.0]),
        )

        self.assertEqual(summary["rows"], 3)
        self.assertAlmostEqual(summary["mean_error_prediction_minus_actual"], 1 / 3)
        self.assertEqual(summary["median_absolute_error"], 1.0)
        self.assertAlmostEqual(summary["within_1_label_unit"], 2 / 3)

    def test_category_metrics_skip_small_groups(self) -> None:
        categories = np.array(["A", "A", "A", "B"])
        actual = np.array([1.0, 2.0, 3.0, 2.0])
        predictions = {"ridge": np.array([1.5, 2.0, 2.5, 2.0])}

        result = _holdout_category_metrics(
            categories, actual, predictions, minimum_rows=3
        )

        self.assertEqual(list(result), ["A"])
        self.assertEqual(result["A"]["rows"], 3)
        self.assertAlmostEqual(result["A"]["models"]["ridge"]["mae"], 1 / 3)

    def test_chronological_split_places_all_later_timestamps_in_holdout(self) -> None:
        features = pd.DataFrame({"value": range(10)})
        labels = np.arange(10, dtype=np.float64)
        timestamps = np.arange(10, dtype=np.int64)

        train_indexes, holdout_indexes, cutoff = chronological_split(
            features, labels, timestamps, holdout_fraction=0.2
        )

        self.assertEqual(len(train_indexes), 8)
        self.assertEqual(len(holdout_indexes), 2)
        self.assertTrue((timestamps[train_indexes] < cutoff).all())
        self.assertTrue((timestamps[holdout_indexes] >= cutoff).all())

    def test_chronological_split_rejects_invalid_fraction(self) -> None:
        with self.assertRaises(ValueError):
            chronological_split(
                pd.DataFrame(), np.array([1.0, 2.0]), np.array([1, 2]), 1.0
            )


if __name__ == "__main__":
    unittest.main()