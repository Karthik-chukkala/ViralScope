"""Local inference API for the trained text and context baselines."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sklearn.pipeline import Pipeline

from .train_baselines import CATEGORICAL_FEATURES, TIME_FEATURES

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "artifacts/models/final"
METRICS_PATH = ROOT / "artifacts/baseline-metrics.json"
USER_AWARE_METRICS_PATH = ROOT / "artifacts/user-aware-metrics.json"
TEMPORAL_METRICS_PATH = ROOT / "artifacts/temporal-sensitivity.json"
TARGET_AUDIT_PATH = ROOT / "artifacts/train-target-audit.json"
METADATA_EDA_PATH = ROOT / "artifacts/metadata-eda.json"
SAMPLE_EDA_PATH = ROOT / "artifacts/sample-eda.json"
FULL_TRAIN_EDA_PATH = ROOT / "artifacts/full-train-eda.json"
TEST_INFERENCE_PATH = ROOT / "artifacts/test-inference-report.json"
DIAGNOSTIC_DIR = ROOT / "artifacts/diagnostics"
DIAGNOSTIC_FILES = {
    "holdout-actual-vs-predicted.png",
    "holdout-residuals.png",
    "holdout-mae-by-category.png",
    "sample-temporal-text.png",
    "full-train-temporal-text.png",
}
WEB_PAGE = ROOT / "web/index.html"
MODEL_CHOICES = (
    "tfidf_ridge",
    "tfidf_sgd",
    "tfidf_sgd_huber",
    "context_catboost",
    "context_catboost_shallow",
    "text_context_ridge",
)
TEXT_MODEL_CHOICES = {"tfidf_ridge", "tfidf_sgd", "tfidf_sgd_huber"}

app = FastAPI(title="ViralScope local demo", docs_url=None, redoc_url=None)


class PredictionRequest(BaseModel):
    title: str = Field(default="", max_length=300)
    tags: str = Field(default="", max_length=10_000)
    category: str = Field(min_length=1, max_length=100)
    subcategory: str = Field(default="", max_length=100)
    concept: str = Field(default="", max_length=200)
    planned_at: datetime
    model: Literal[
        "tfidf_ridge",
        "tfidf_sgd",
        "tfidf_sgd_huber",
        "context_catboost",
        "context_catboost_shallow",
        "text_context_ridge",
    ] = "tfidf_ridge"


def make_prediction_features(request: PredictionRequest) -> tuple[pd.DataFrame, str]:
    planned_at = request.planned_at
    if planned_at.tzinfo is None:
        planned_at = planned_at.replace(tzinfo=timezone.utc)
    planned_at = planned_at.astimezone(timezone.utc)
    hours = planned_at.hour
    weekdays = planned_at.weekday()
    months = planned_at.month - 1
    features: dict[str, Any] = {
        "Category": request.category,
        "Subcategory": request.subcategory,
        "Concept": request.concept,
        "text": f"{request.title} {request.tags}".strip(),
        "hour_sin": np.sin(2 * np.pi * hours / 24),
        "hour_cos": np.cos(2 * np.pi * hours / 24),
        "weekday_sin": np.sin(2 * np.pi * weekdays / 7),
        "weekday_cos": np.cos(2 * np.pi * weekdays / 7),
        "month_sin": np.sin(2 * np.pi * months / 12),
        "month_cos": np.cos(2 * np.pi * months / 12),
    }
    return pd.DataFrame([features]), planned_at.isoformat()


@lru_cache(maxsize=4)
def load_models(model_dir: Path = MODEL_DIR) -> dict[str, Any]:
    required = [
        model_dir / "tfidf-ridge.joblib",
        model_dir / "tfidf-sgd.joblib",
        model_dir / "tfidf-sgd-huber.joblib",
        model_dir / "context-catboost.cbm",
        model_dir / "context-catboost-shallow.cbm",
        model_dir / "text-context-ridge.joblib",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise RuntimeError("Train the baseline models before starting the demo.")
    context_model = CatBoostRegressor()
    context_model.load_model(str(required[3]))
    shallow_context_model = CatBoostRegressor()
    shallow_context_model.load_model(str(required[4]))
    return {
        "tfidf_ridge": joblib.load(required[0]),
        "tfidf_sgd": joblib.load(required[1]),
        "tfidf_sgd_huber": joblib.load(required[2]),
        "context_catboost": context_model,
        "context_catboost_shallow": shallow_context_model,
        "text_context_ridge": joblib.load(required[5]),
    }


def _top_contributions(model_name: str, model: Any, features: pd.DataFrame) -> list[dict[str, Any]]:
    if model_name.startswith("context_catboost"):
        pool = Pool(features[CATEGORICAL_FEATURES + TIME_FEATURES], cat_features=CATEGORICAL_FEATURES)
        values = model.get_feature_importance(pool, type="ShapValues")[0, :-1]
        names = CATEGORICAL_FEATURES + TIME_FEATURES
    elif model_name in TEXT_MODEL_CHOICES:
        vectorizer = model.named_steps["tfidf"]
        matrix = vectorizer.transform(features["text"])
        contributions = matrix.multiply(model.named_steps["regressor"].coef_).toarray()[0]
        values = contributions
        names = vectorizer.get_feature_names_out()
    else:
        transformer = model.named_steps["features"]
        matrix = transformer.transform(features)
        contributions = matrix.multiply(model.named_steps["regressor"].coef_).toarray()[0]
        values = contributions
        names = transformer.get_feature_names_out()

    indexes = np.argsort(np.abs(values))[-6:][::-1]
    return [
        {"feature": str(names[index]), "contribution": float(values[index])}
        for index in indexes
        if abs(float(values[index])) > 1e-10
    ]


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(WEB_PAGE)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/metrics")
def metrics() -> dict[str, Any]:
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=503, detail="Baseline metrics are not available.")
    report = json_read_metrics()
    response = {
        "evaluation": report["evaluation"],
        "metrics": report["metrics"],
        "diagnostics": report.get("diagnostics", {}),
    }
    if USER_AWARE_METRICS_PATH.exists():
        user_report = json.loads(USER_AWARE_METRICS_PATH.read_text(encoding="utf-8"))
        response["user_aware"] = {
            "evaluation": user_report["evaluation"],
            "metrics": user_report["metrics"],
        }
    if TEMPORAL_METRICS_PATH.exists():
        temporal_report = json.loads(TEMPORAL_METRICS_PATH.read_text(encoding="utf-8"))
        response["temporal_sensitivity"] = {
            "protocol": temporal_report["protocol"],
            "runs": temporal_report["runs"],
            "best_model_winner_counts": temporal_report["best_model_winner_counts"],
        }
    return response


@app.get("/api/diagnostics/{filename}")
def diagnostic_plot(filename: str) -> FileResponse:
    if filename not in DIAGNOSTIC_FILES:
        raise HTTPException(status_code=404, detail="Diagnostic plot not found.")
    path = DIAGNOSTIC_DIR / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Diagnostic plot is not available.")
    return FileResponse(path, media_type="image/png")


@app.get("/api/dataset")
def dataset_summary() -> dict[str, Any]:
    if not TARGET_AUDIT_PATH.exists() or not METADATA_EDA_PATH.exists():
        raise HTTPException(status_code=503, detail="Dataset audit reports are not available.")
    target_report = json.loads(TARGET_AUDIT_PATH.read_text(encoding="utf-8"))
    metadata_report = json.loads(METADATA_EDA_PATH.read_text(encoding="utf-8"))
    sample_eda = None
    if SAMPLE_EDA_PATH.exists():
        sample_eda_report = json.loads(SAMPLE_EDA_PATH.read_text(encoding="utf-8"))
        sample_eda = sample_eda_report["sample_eda"]
    full_train_eda = None
    if FULL_TRAIN_EDA_PATH.exists():
        full_train_report = json.loads(FULL_TRAIN_EDA_PATH.read_text(encoding="utf-8"))
        full_train_eda = full_train_report["sample_eda"]
    test_inference = None
    if TEST_INFERENCE_PATH.exists():
        test_report = json.loads(TEST_INFERENCE_PATH.read_text(encoding="utf-8"))
        test_inference = {
            "prediction_rows": test_report["prediction_rows"],
            "test_shards": len(test_report["dataset"]["shards_selected"]),
            "models": test_report["models"],
            "training_label_range": test_report["training_label_range"],
            "test_metrics_computed": test_report["test_metrics_computed"],
            "test_metric_note": test_report["test_metric_note"],
        }
    target_summary = target_report["target"]["non_sentinel_summary"]
    target_summary["histogram_counts"] = target_report["target"]["non_sentinel_histogram"]["counts"]
    target_summary["histogram_edges"] = target_report["target"]["non_sentinel_histogram"]["bin_edges"]
    return {
        "revision": target_report["dataset"]["revision"],
        "train_rows": target_report["target"]["rows_scanned"],
        "target": target_summary,
        "metadata_sample": {
            "rows": metadata_report["scan"]["rows_scanned"],
            "shards_scanned": metadata_report["scan"]["shards_scanned"],
            "shards_available": metadata_report["scan"]["shards_available"],
            "missing_titles": metadata_report["eda"]["missing_counts"]["Title"],
            "categories": metadata_report["eda"]["category_frequencies"],
        },
        "sample_text_temporal_eda": sample_eda,
        "full_train_eda": full_train_eda,
        "test_inference": test_inference,
    }


def json_read_metrics() -> dict[str, Any]:
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@app.post("/api/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    try:
        models = load_models()
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    features, planned_at = make_prediction_features(request)
    model = models[request.model]
    if request.model in TEXT_MODEL_CHOICES:
        prediction = float(model.predict(features["text"])[0])
    elif request.model.startswith("context_catboost"):
        prediction = float(model.predict(features[CATEGORICAL_FEATURES + TIME_FEATURES])[0])
    else:
        prediction = float(model.predict(features)[0])
    contributions = _top_contributions(request.model, model, features)
    return {
        "prediction": prediction,
        "model": request.model,
        "planned_at_utc": planned_at,
        "contributions": contributions,
        "explanation_note": "Feature contributions describe model behavior, not causes or guaranteed outcomes.",
    }