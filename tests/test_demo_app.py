import json
import unittest
from datetime import datetime

from fastapi.testclient import TestClient

from viralscope.demo_app import MODEL_CHOICES, PredictionRequest, app, make_prediction_features


class DemoAppTests(unittest.TestCase):
    def test_analysis_is_on_dedicated_pages(self) -> None:
        client = TestClient(app)
        dashboard = client.get("/").text
        dataset = client.get("/dataset").text
        models = client.get("/models").text
        solutions = client.get("/solutions").text
        self.assertIn("Team 6A", dashboard)
        for heading in (
            "Dataset snapshot", "Chronological holdout", "Unseen-user holdout",
            "Temporal cutoff sensitivity", "Holdout error diagnostics", "Work completed and decisions",
        ):
            with self.subTest(heading=heading):
                self.assertNotIn(heading, dashboard)
        self.assertNotIn("/api/diagnostics/", dashboard)
        self.assertIn("sample-temporal-text.png", dataset)
        self.assertIn("Public test split: inference only", dataset)
        for heading in (
            "Chronological holdout",
            "Unseen-user holdout",
            "Future-time + unseen-user holdout",
            "Temporal cutoff sensitivity",
            "Holdout error diagnostics",
        ):
            with self.subTest(heading=heading):
                self.assertIn(heading, models)
        self.assertIn("holdout-actual-vs-predicted.png", models)
        self.assertIn("holdout-residuals.png", models)
        self.assertIn("holdout-mae-by-category.png", models)
        self.assertIn("Work completed and decisions", solutions)
        self.assertIn('id="demo-form"', client.get("/demo").text)

    def test_reference_links_serve_local_documents(self) -> None:
        client = TestClient(app)
        for path in (
            "/LICENSE-SMP-CHALLENGE.md",
            "/docs/team-handoff.md",
            "/docs/research-and-project-blueprint.md",
            "/docs/modules/dataset-access-and-audit.md",
            "/docs/modules/cleaning-and-eda.md",
            "/docs/modules/modeling-baselines.md",
            "/docs/modules/review-validation.md",
            "/docs/modules/local-demo.md",
        ):
            with self.subTest(path=path):
                self.assertEqual(client.get(path).status_code, 200)

    def test_prediction_schema_accepts_all_model_variants(self) -> None:
        for model in MODEL_CHOICES:
            with self.subTest(model=model):
                request = PredictionRequest(
                    category="Food",
                    planned_at="2016-01-01T12:00:00Z",
                    model=model,
                )
                self.assertEqual(request.model, model)

    def test_dataset_api_exposes_aggregate_sample_eda_only(self) -> None:
        response = TestClient(app).get("/api/dataset")

        self.assertEqual(response.status_code, 200)
        sample_eda = response.json()["sample_text_temporal_eda"]
        self.assertEqual(sample_eda["rows"], 42730)
        self.assertEqual(sample_eda["title_character_lengths"]["median"], 19.0)
        self.assertFalse(sample_eda["privacy"]["raw_text_written"])
        full_train = response.json()["full_train_eda"]
        test_inference = response.json()["test_inference"]
        self.assertEqual(full_train["rows"], 305613)
        self.assertEqual(len(full_train["category_frequencies"]), 11)
        self.assertEqual(test_inference["prediction_rows"], 180581)
        self.assertFalse(test_inference["test_metrics_computed"])
        self.assertEqual(test_inference["input_eda"]["rows"], 180581)
        self.assertEqual(len(test_inference["input_eda"]["category_frequencies"]), 11)
        self.assertFalse(test_inference["input_eda"]["labels_read"])
        self.assertEqual(test_inference["input_eda"]["rows"], 180581)
        self.assertFalse(test_inference["input_eda"]["labels_read"])
        self.assertFalse(test_inference["input_eda"]["ids_read"])
        chart = TestClient(app).get("/api/diagnostics/full-train-temporal-text.png")
        self.assertEqual(chart.status_code, 200)
        self.assertEqual(chart.headers["content-type"], "image/png")

    def test_metrics_api_exposes_aggregate_user_split_without_ids(self) -> None:
        response = TestClient(app).get("/api/metrics")

        self.assertEqual(response.status_code, 200)
        user_aware = response.json()["user_aware"]
        self.assertEqual(user_aware["evaluation"]["user_overlap_count"], 0)
        self.assertEqual(user_aware["evaluation"]["holdout_user_count"], 7663)
        self.assertNotIn('"Uid"', json.dumps(user_aware))

    def test_metrics_api_exposes_temporal_sensitivity_runs(self) -> None:
        response = TestClient(app).get("/api/metrics")

        self.assertEqual(response.status_code, 200)
        runs = response.json()["temporal_sensitivity"]["runs"]
        self.assertEqual([run["holdout_fraction"] for run in runs], [0.15, 0.2, 0.25])
        self.assertEqual([run["holdout_rows"] for run in runs], [45842, 61123, 76404])

    def test_metrics_api_exposes_combined_evaluation_without_user_ids(self) -> None:
        response = TestClient(app).get("/api/metrics")

        self.assertEqual(response.status_code, 200)
        if "combined_evaluation" not in response.json():
            self.skipTest("Combined evaluation report has not been generated yet")
        combined = response.json()["combined_evaluation"]
        self.assertEqual(combined["evaluation"]["user_overlap_count"], 0)
        self.assertGreater(combined["evaluation"]["purged_rows"], 0)
        self.assertNotIn('"Uid"', json.dumps(combined))

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