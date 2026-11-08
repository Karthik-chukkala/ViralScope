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
from catboost import CatBoostRegressor
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.pipeline import Pipeline

from .train_baselines import CATEGORICAL_FEATURES, TIME_FEATURES
from .explanations import top_contributions

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "artifacts/models/final"
METRICS_PATH = ROOT / "artifacts/baseline-metrics.json"
USER_AWARE_METRICS_PATH = ROOT / "artifacts/user-aware-metrics.json"
TEMPORAL_METRICS_PATH = ROOT / "artifacts/temporal-sensitivity.json"
COMBINED_METRICS_PATH = ROOT / "artifacts/combined-evaluation.json"
TARGET_AUDIT_PATH = ROOT / "artifacts/train-target-audit.json"
METADATA_EDA_PATH = ROOT / "artifacts/metadata-eda.json"
DATASET_AUDIT_PATH = ROOT / "artifacts/dataset-audit.json"
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
    "test-input-coverage.png",
}
WEB_PAGE = ROOT / "web/index.html"
WEB_DIR = ROOT / "web"
MODEL_CHOICES = (
    "tfidf_ridge",
    "tfidf_sgd",
    "tfidf_sgd_huber",
    "context_catboost",
    "context_catboost_shallow",
    "text_context_ridge",
)
TEXT_MODEL_CHOICES = {"tfidf_ridge", "tfidf_sgd", "tfidf_sgd_huber"}
MODEL_INFO = {
    "tfidf_ridge": {
        "display_name": "TF-IDF + Ridge",
        "family": "Text-only linear regression",
        "why": "Fast, sparse-text baseline that is stable on high-dimensional title/tag features.",
        "features": ["Title + Alltags", "word unigrams and bigrams", "sublinear TF", "min_df=2"],
        "training": "Ridge alpha=2; vectorizer capped at 150,000 features.",
        "explanation": "Signed TF-IDF term contributions for each prediction.",
    },
    "tfidf_sgd": {
        "display_name": "TF-IDF + SGD",
        "family": "Text-only stochastic linear regression",
        "why": "Checks a scalable stochastic optimizer against the closed-form Ridge baseline.",
        "features": ["Title + Alltags", "word unigrams and bigrams", "sublinear TF", "min_df=2"],
        "training": "Squared-error SGD; L2 alpha=1e-5; early stopping; random seed 42.",
        "explanation": "Signed TF-IDF term contributions for each prediction.",
    },
    "tfidf_sgd_huber": {
        "display_name": "TF-IDF + Huber SGD",
        "family": "Robust-loss text-only linear regression",
        "why": "Tests whether a robust loss changes sensitivity to large popularity errors.",
        "features": ["Title + Alltags", "word unigrams and bigrams", "sublinear TF", "min_df=2"],
        "training": "Huber-loss SGD; L2 alpha=1e-5; early stopping; random seed 42.",
        "explanation": "Signed TF-IDF term contributions for each prediction.",
    },
    "context_catboost": {
        "display_name": "Context CatBoost",
        "family": "Structured nonlinear regression",
        "why": "Tests nonlinear category and scheduled-time interactions without text or images.",
        "features": ["Category", "Subcategory", "Concept", "UTC hour", "weekday", "month"],
        "training": "CatBoost RMSE; 500 iterations; depth 8; learning rate 0.05; seed 42.",
        "explanation": "Per-row CatBoost SHAP values over the structured features.",
    },
    "context_catboost_shallow": {
        "display_name": "Context CatBoost shallow",
        "family": "Regularized structured nonlinear regression",
        "why": "Compares a shallower tree ensemble to test whether reduced complexity helps holdout generalization.",
        "features": ["Category", "Subcategory", "Concept", "UTC hour", "weekday", "month"],
        "training": "CatBoost RMSE; 700 iterations; depth 5; learning rate 0.04; seed 42.",
        "explanation": "Per-row CatBoost SHAP values over the structured features.",
    },
    "text_context_ridge": {
        "display_name": "Text + context Ridge",
        "family": "Fused sparse text and structured regression",
        "why": "Tests whether planned-time and category context adds value beyond text alone.",
        "features": ["Title + Alltags TF-IDF", "one-hot Category/Subcategory/Concept", "cyclic UTC hour, weekday, month"],
        "training": "ColumnTransformer + Ridge alpha=2; vectorizer capped at 150,000 terms.",
        "explanation": "Signed sparse feature contributions for each prediction.",
    },
}

app = FastAPI(title="ViralScope local demo", docs_url=None, redoc_url=None)
app.mount("/web", StaticFiles(directory=WEB_DIR), name="web")
app.mount("/docs", StaticFiles(directory=ROOT / "docs"), name="project-docs")


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


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(WEB_PAGE)


@app.get("/dataset", include_in_schema=False)
def dataset_page() -> FileResponse:
    return FileResponse(WEB_DIR / "dataset.html")


@app.get("/models", include_in_schema=False)
def models_page() -> FileResponse:
    return FileResponse(WEB_DIR / "models.html")


@app.get("/model/{model_name}", include_in_schema=False)
def model_page(model_name: str) -> FileResponse:
    if model_name not in MODEL_CHOICES:
        raise HTTPException(status_code=404, detail="Model page not found.")
    return FileResponse(WEB_DIR / "model.html")


@app.get("/demo", include_in_schema=False)
def demo_page() -> FileResponse:
    return FileResponse(WEB_DIR / "demo.html")


@app.get("/solutions", include_in_schema=False)
def solutions_page() -> FileResponse:
    return FileResponse(WEB_DIR / "solutions.html")


@app.get("/references", include_in_schema=False)
def references_page() -> FileResponse:
    return FileResponse(WEB_DIR / "references.html")


@app.get("/LICENSE-SMP-CHALLENGE.md", include_in_schema=False)
def license_file() -> FileResponse:
    return FileResponse(ROOT / "LICENSE-SMP-CHALLENGE.md")


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
    if COMBINED_METRICS_PATH.exists():
        combined_report = json.loads(COMBINED_METRICS_PATH.read_text(encoding="utf-8"))
        response["combined_evaluation"] = {
            "evaluation": combined_report["evaluation"],
            "metrics": combined_report["metrics"],
        }
    return response


@app.get("/api/models")
def models_summary() -> dict[str, Any]:
    baseline = json_read_metrics()
    result = []
    for model_name in MODEL_CHOICES:
        result.append(
            {
                "name": model_name,
                **MODEL_INFO[model_name],
                "chronological": baseline["metrics"][model_name],
            }
        )
    return {
        "fit_rows": json.loads(
            (ROOT / "artifacts/final-models-report.json").read_text(encoding="utf-8")
        )["fit_rows"],
        "models": result,
        "official_test_scored": False,
    }


@app.get("/api/model/{model_name}")
def model_summary(model_name: str) -> dict[str, Any]:
    if model_name not in MODEL_CHOICES:
        raise HTTPException(status_code=404, detail="Model not found.")
    baseline = json_read_metrics()
    final_report = json.loads(
        (ROOT / "artifacts/final-models-report.json").read_text(encoding="utf-8")
    )
    user_report = json.loads(USER_AWARE_METRICS_PATH.read_text(encoding="utf-8"))
    test_report = json.loads(TEST_INFERENCE_PATH.read_text(encoding="utf-8"))
    return {
        "name": model_name,
        **MODEL_INFO[model_name],
        "fit_rows": final_report["fit_rows"],
        "final_model_path": final_report["models"][model_name],
        "chronological": baseline["metrics"][model_name],
        "chronological_evaluation": baseline["evaluation"],
        "user_disjoint": user_report["metrics"][model_name],
        "user_disjoint_evaluation": user_report["evaluation"],
        "test_prediction_summary": test_report["models"][model_name],
        "test_rows": test_report["prediction_rows"],
        "test_metrics_computed": False,
    }


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
    audit_report = json.loads(DATASET_AUDIT_PATH.read_text(encoding="utf-8"))
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
            "input_eda": test_report["input_eda"],
            "input_eda_plot": test_report["input_eda_plot"],
            "test_metrics_computed": test_report["test_metrics_computed"],
            "test_metric_note": test_report["test_metric_note"],
        }
    target_summary = target_report["target"]["non_sentinel_summary"]
    target_summary["histogram_counts"] = target_report["target"]["non_sentinel_histogram"]["counts"]
    target_summary["histogram_edges"] = target_report["target"]["non_sentinel_histogram"]["bin_edges"]
    return {
        "revision": target_report["dataset"]["revision"],
        "train_rows": target_report["target"]["rows_scanned"],
        "split_rows": {
            split: details["rows"] for split, details in audit_report["splits"].items()
        },
        "field_schema": audit_report["features"],
        "license": audit_report["dataset"]["license"],
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
    contributions = top_contributions(request.model, model, features)
    return {
        "prediction": prediction,
        "model": request.model,
        "planned_at_utc": planned_at,
        "contributions": contributions,
        "explanation_note": "Feature contributions describe model behavior, not causes or guaranteed outcomes.",
    }