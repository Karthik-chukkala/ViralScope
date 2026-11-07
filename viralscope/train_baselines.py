"""Train lightweight text and context baselines on the local sample cache."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DEFAULT_DATA = Path("data/train-sample.parquet")
DEFAULT_MANIFEST = Path("data/train-sample-manifest.json")
DEFAULT_REPORT = Path("artifacts/baseline-metrics.json")
DEFAULT_MODEL_DIR = Path("artifacts/models")
CATEGORICAL_FEATURES = ["Category", "Subcategory", "Concept"]
TIME_FEATURES = [
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
]


def build_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    timestamps = pd.to_numeric(frame["Postdate"], errors="coerce")
    dates = pd.to_datetime(timestamps, unit="s", utc=True, errors="coerce")
    labels = pd.to_numeric(frame["label"], errors="coerce")
    valid = dates.notna() & labels.notna() & np.isfinite(labels) & (labels >= 0)
    dates = dates.loc[valid]
    labels = labels.loc[valid].to_numpy(dtype=np.float64)
    selected = frame.loc[valid]

    hours = dates.dt.hour.to_numpy()
    weekdays = dates.dt.day_of_week.to_numpy()
    months = dates.dt.month.to_numpy() - 1
    features = pd.DataFrame(index=selected.index)
    for column in CATEGORICAL_FEATURES:
        features[column] = selected[column].fillna("").astype(str)
    features["text"] = (
        selected["Title"].fillna("").astype(str)
        + " "
        + selected["Alltags"].fillna("").astype(str)
    )
    for name, values, period in (
        ("hour", hours, 24),
        ("weekday", weekdays, 7),
        ("month", months, 12),
    ):
        features[f"{name}_sin"] = np.sin(2 * np.pi * values / period)
        features[f"{name}_cos"] = np.cos(2 * np.pi * values / period)
    timestamp_seconds = timestamps.loc[valid].to_numpy(dtype=np.int64)
    return features, labels, timestamp_seconds


def chronological_split(
    features: pd.DataFrame,
    labels: np.ndarray,
    timestamps: np.ndarray,
    holdout_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, int]:
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be between zero and one")
    if len(labels) < 2:
        raise ValueError("at least two valid rows are required")
    ordered = np.sort(timestamps)
    cutoff_index = min(max(int(len(ordered) * (1 - holdout_fraction)), 1), len(ordered) - 1)
    cutoff = int(ordered[cutoff_index])
    train_mask = timestamps < cutoff
    holdout_mask = ~train_mask
    if not train_mask.any() or not holdout_mask.any():
        raise ValueError("timestamps do not support a non-empty chronological holdout")
    return np.flatnonzero(train_mask), np.flatnonzero(holdout_mask), cutoff


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    if np.unique(actual).size < 2 or np.unique(predicted).size < 2:
        correlation = None
    else:
        correlation = float(spearmanr(actual, predicted).statistic)
    return {
        "src": correlation if correlation is not None and np.isfinite(correlation) else None,
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def _record_metrics(
    labels: np.ndarray,
    train_indexes: np.ndarray,
    holdout_indexes: np.ndarray,
    train_predictions: np.ndarray,
    holdout_predictions: np.ndarray,
) -> dict[str, Any]:
    return {
        "train": _metrics(labels[train_indexes], train_predictions),
        "holdout": _metrics(labels[holdout_indexes], holdout_predictions),
    }


def train_baselines(
    data_path: Path = DEFAULT_DATA,
    manifest_path: Path = DEFAULT_MANIFEST,
    report_path: Path = DEFAULT_REPORT,
    model_dir: Path = DEFAULT_MODEL_DIR,
    workers: int = 4,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be positive")
    frame = pd.read_parquet(data_path)
    features, labels, timestamps = build_features(frame)
    train_indexes, holdout_indexes, cutoff = chronological_split(
        features, labels, timestamps
    )
    train = features.iloc[train_indexes]
    holdout = features.iloc[holdout_indexes]
    train_labels = labels[train_indexes]
    holdout_labels = labels[holdout_indexes]

    model_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    median_value = float(np.median(train_labels))
    results["median"] = _record_metrics(
        labels,
        train_indexes,
        holdout_indexes,
        np.full(len(train_indexes), median_value),
        np.full(len(holdout_indexes), median_value),
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_features=150_000, sublinear_tf=True
    )
    text_model = Pipeline([("tfidf", vectorizer), ("regressor", Ridge(alpha=2.0))])
    text_model.fit(train["text"], train_labels)
    results["tfidf_ridge"] = _record_metrics(
        labels,
        train_indexes,
        holdout_indexes,
        text_model.predict(train["text"]),
        text_model.predict(holdout["text"]),
    )
    joblib.dump(text_model, model_dir / "tfidf-ridge.joblib")

    context_features = CATEGORICAL_FEATURES + TIME_FEATURES
    context_model = CatBoostRegressor(
        iterations=500,
        depth=8,
        learning_rate=0.05,
        loss_function="RMSE",
        random_seed=42,
        thread_count=workers,
        verbose=False,
        allow_writing_files=False,
    )
    context_model.fit(
        train[context_features],
        train_labels,
        cat_features=CATEGORICAL_FEATURES,
    )
    results["context_catboost"] = _record_metrics(
        labels,
        train_indexes,
        holdout_indexes,
        context_model.predict(train[context_features]),
        context_model.predict(holdout[context_features]),
    )
    context_model.save_model(str(model_dir / "context-catboost.cbm"))

    fused_features = ColumnTransformer(
        [
            (
                "text",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=150_000,
                    sublinear_tf=True,
                ),
                "text",
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", min_frequency=2),
                CATEGORICAL_FEATURES,
            ),
            ("time", "passthrough", TIME_FEATURES),
        ],
        sparse_threshold=1.0,
    )
    fused_model = Pipeline(
        [("features", fused_features), ("regressor", Ridge(alpha=2.0))]
    )
    fused_model.fit(train, train_labels)
    results["text_context_ridge"] = _record_metrics(
        labels,
        train_indexes,
        holdout_indexes,
        fused_model.predict(train),
        fused_model.predict(holdout),
    )
    joblib.dump(fused_model, model_dir / "text-context-ridge.joblib")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": manifest,
        "evaluation": {
            "protocol": "chronological local holdout from sampled training shards",
            "holdout_fraction_requested": 0.2,
            "cutoff_utc": pd.to_datetime(cutoff, unit="s", utc=True).isoformat(),
            "valid_rows": len(labels),
            "train_rows": len(train_indexes),
            "holdout_rows": len(holdout_indexes),
            "official_test_scored": False,
        },
        "metrics": results,
        "models": {
            "tfidf_ridge": "artifacts/models/tfidf-ridge.joblib",
            "context_catboost": "artifacts/models/context-catboost.cbm",
            "text_context_ridge": "artifacts/models/text-context-ridge.joblib",
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train baseline popularity regressors.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    report = train_baselines(
        data_path=args.data,
        manifest_path=args.manifest,
        report_path=args.report,
        model_dir=args.model_dir,
        workers=args.workers,
    )
    print(f"Train rows: {report['evaluation']['train_rows']}")
    print(f"Holdout rows: {report['evaluation']['holdout_rows']}")
    print(json.dumps(report["metrics"], indent=2))
    print(f"Metrics: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())