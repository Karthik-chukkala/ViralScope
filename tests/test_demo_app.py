import json
import unittest
from datetime import datetime

from fastapi.testclient import TestClient

from viralscope.demo_app import PredictionRequest, app, make_prediction_features


class DemoAppTests(unittest.TestCase):
    def test_dataset_api_exposes_aggregate_sample_eda_only(self) -> None:
        response = TestClient(app).get("/api/dataset")

        self.assertEqual(response.status_code, 200)
        sample_eda = response.json()["sample_text_temporal_eda"]
        self.assertEqual(sample_eda["rows"], 42730)
        self.assertEqual(sample_eda["title_character_lengths"]["median"], 19.0)
        self.assertFalse(sample_eda["privacy"]["raw_text_written"])

    def test_metrics_api_exposes_aggregate_user_split_without_ids(self) -> None:
        response = TestClient(app).get("/api/metrics")

        self.assertEqual(response.status_code, 200)
        user_aware = response.json()["user_aware"]
        self.assertEqual(user_aware["evaluation"]["user_overlap_count"], 0)
        self.assertEqual(user_aware["evaluation"]["holdout_user_count"], 2196)
        self.assertNotIn('"Uid"', json.dumps(user_aware))

    def test_metrics_api_exposes_temporal_sensitivity_runs(self) -> None:
        response = TestClient(app).get("/api/metrics")

        self.assertEqual(response.status_code, 200)
        runs = response.json()["temporal_sensitivity"]["runs"]
        self.assertEqual([run["holdout_fraction"] for run in runs], [0.15, 0.2, 0.25])
        self.assertEqual([run["holdout_rows"] for run in runs], [6410, 8546, 10683])

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