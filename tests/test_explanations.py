import unittest

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from viralscope.explanations import decompose_prediction, top_contributions


class ExplanationTests(unittest.TestCase):
    def test_linear_contributions_reconstruct_prediction_and_are_deterministic(self) -> None:
        model = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
                ("regressor", Ridge(alpha=1.0)),
            ]
        )
        model.fit(
            ["fresh berry tart", "bright garden bird", "fresh garden flowers"],
            np.array([7.0, 5.0, 6.0]),
        )
        features = pd.DataFrame({"text": ["fresh berry garden"]})

        names, values, baseline = decompose_prediction("tfidf_ridge", model, features)
        prediction = float(model.predict(features["text"])[0])
        first = top_contributions("tfidf_ridge", model, features)
        second = top_contributions("tfidf_ridge", model, features)

        self.assertAlmostEqual(baseline + float(values.sum()), prediction, places=10)
        self.assertEqual(first, second)
        self.assertEqual(len(names), len(values))
        self.assertLessEqual(len(first), 6)


if __name__ == "__main__":
    unittest.main()
