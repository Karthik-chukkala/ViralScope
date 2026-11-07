import unittest
from datetime import datetime

from viralscope.demo_app import PredictionRequest, make_prediction_features


class DemoAppTests(unittest.TestCase):
    def test_feature_builder_uses_utc_and_combines_text_fields(self) -> None:
        request = PredictionRequest(
            title="Coastal walk",
            tags="travel sunset",
            category="Travel&Active&Sports",
            planned_at=datetime(2016, 2, 1, 18, 30),
        )

        features, planned_at = make_prediction_features(request)

        self.assertEqual(planned_at, "2016-02-01T18:30:00+00:00")
        self.assertEqual(features.loc[0, "text"], "Coastal walk travel sunset")
        self.assertEqual(features.loc[0, "Category"], "Travel&Active&Sports")
        self.assertAlmostEqual(features.loc[0, "hour_sin"], -1.0, places=12)
        self.assertAlmostEqual(features.loc[0, "hour_cos"], 0.0, places=12)


if __name__ == "__main__":
    unittest.main()