import unittest

import numpy as np
import pandas as pd

from viralscope.train_baselines import chronological_split


class TrainBaselineTests(unittest.TestCase):
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