"""Train lightweight text and context baselines on the local sample cache."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder

DEFAULT_DATA = Path("data/train-sample.parquet")
DEFAULT_MANIFEST = Path("data/train-sample-manifest.json")
DEFAULT_REPORT = Path("artifacts/baseline-metrics.json")
DEFAULT_MODEL_DIR = Path("artifacts/models")
DEFAULT_DIAGNOSTIC_DIR = Path("artifacts/diagnostics")
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


def user_aware_split(
    features: pd.DataFrame,
    labels: np.ndarray,
    groups: np.ndarray,
    holdout_fraction: float = 0.2,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be between zero and one")
    group_values = np.asarray(groups)
    if group_values.ndim != 1 or len(group_values) != len(labels):
        raise ValueError("one group ID is required for each valid row")
    if pd.isna(group_values).any():
        raise ValueError("group IDs must not be missing")
    if np.unique(group_values).size < 2:
        raise ValueError("at least two distinct groups are required")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=holdout_fraction,
        random_state=random_seed,
    )
    train_indexes, holdout_indexes = next(
        splitter.split(features, labels, groups=group_values)
    )
    return train_indexes, holdout_indexes


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    if np.unique(actual).size < 2 or np.unique(predicted).size < 2:
        correlation = None
    else:
        correlation = float(spearmanr(actual, predicted).statistic)
    return {
        "src": correlation if correlation is not None and np.isfinite(correlation) else None,
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": (
            float(r2_score(actual, predicted))
            if len(actual) > 1 and np.unique(actual).size > 1
            else None
        ),
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


def summarize_residuals(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    residuals = np.asarray(predicted, dtype=np.float64) - np.asarray(actual, dtype=np.float64)
    absolute_errors = np.abs(residuals)
    return {
        "rows": int(residuals.size),
        "mean_error_prediction_minus_actual": float(residuals.mean()),
        "median_absolute_error": float(np.median(absolute_errors)),
        "p90_absolute_error": float(np.quantile(absolute_errors, 0.9)),
        "p95_absolute_error": float(np.quantile(absolute_errors, 0.95)),
        "within_1_label_unit": float(np.mean(absolute_errors <= 1.0)),
        "within_2_label_units": float(np.mean(absolute_errors <= 2.0)),
    }


def _holdout_category_metrics(
    categories: np.ndarray,
    actual: np.ndarray,
    predictions: dict[str, np.ndarray],
    minimum_rows: int = 50,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for category in sorted(set(categories)):
        indexes = np.flatnonzero(categories == category)
        if len(indexes) < minimum_rows:
            continue
        report[str(category)] = {
            "rows": int(len(indexes)),
            "models": {
                name: _metrics(actual[indexes], values[indexes])
                for name, values in predictions.items()
            },
        }
    return report


def _save_diagnostic_plots(
    output_dir: Path,
    actual: np.ndarray,
    predictions: dict[str, np.ndarray],
    category_metrics: dict[str, Any],
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_names = ["tfidf_ridge", "context_catboost", "text_context_ridge"]
    model_labels = ["Text TF-IDF + Ridge", "Context CatBoost", "Text + context"]
    plot_paths: list[str] = []

    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    lower = float(min(actual.min(), *(predictions[name].min() for name in model_names)))
    upper = float(max(actual.max(), *(predictions[name].max() for name in model_names)))
    for axis, name, label in zip(axes, model_names, model_labels):
        axis.scatter(actual, predictions[name], s=7, alpha=0.16, color="#17685c", rasterized=True)
        axis.plot([lower, upper], [lower, upper], color="#c34f3c", linewidth=1)
        axis.set_title(label)
        axis.set_xlabel("Actual label")
        axis.set_ylabel("Predicted label")
        axis.grid(alpha=0.2)
    prediction_plot = output_dir / "holdout-actual-vs-predicted.png"
    figure.savefig(prediction_plot, dpi=150)
    plt.close(figure)
    plot_paths.append(str(prediction_plot))

    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for axis, name, label in zip(axes, model_names, model_labels):
        axis.hist(predictions[name] - actual, bins=35, color="#4b9b87", edgecolor="white")
        axis.axvline(0, color="#c34f3c", linewidth=1)
        axis.set_title(label)
        axis.set_xlabel("Prediction error")
        axis.set_ylabel("Posts")
        axis.grid(axis="y", alpha=0.2)
    residual_plot = output_dir / "holdout-residuals.png"
    figure.savefig(residual_plot, dpi=150)
    plt.close(figure)
    plot_paths.append(str(residual_plot))

    categories_to_plot = list(category_metrics)
    if categories_to_plot:
        indexes = np.arange(len(categories_to_plot))
        width = 0.24
        figure, axis = plt.subplots(figsize=(max(10, len(categories_to_plot) * 0.8), 5))
        for offset, name, label in zip((-width, 0, width), model_names, model_labels):
            values = [category_metrics[category]["models"][name]["mae"] for category in categories_to_plot]
            axis.bar(indexes + offset, values, width, label=label)
        axis.set_xticks(indexes, categories_to_plot, rotation=35, ha="right")
        axis.set_ylabel("Holdout MAE")
        axis.set_title("Holdout error by category (minimum 50 posts)")
        axis.legend(frameon=False)
        axis.grid(axis="y", alpha=0.2)
        figure.tight_layout()
        category_plot = output_dir / "holdout-mae-by-category.png"
        figure.savefig(category_plot, dpi=150)
        plt.close(figure)
        plot_paths.append(str(category_plot))
    return plot_paths


def train_baselines(
    data_path: Path = DEFAULT_DATA,
    manifest_path: Path = DEFAULT_MANIFEST,
    report_path: Path = DEFAULT_REPORT,
    model_dir: Path = DEFAULT_MODEL_DIR,
    workers: int = 4,
    split_strategy: str = "chronological",
    groups: np.ndarray | None = None,
    diagnostic_dir: Path = DEFAULT_DIAGNOSTIC_DIR,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be positive")
    frame = pd.read_parquet(data_path)
    features, labels, timestamps = build_features(frame)
    valid_groups = None
    cutoff: int | None
    if split_strategy == "chronological":
        train_indexes, holdout_indexes, cutoff = chronological_split(
            features, labels, timestamps
        )
        protocol = "chronological local holdout from sampled training shards"
    elif split_strategy == "user":
        if groups is None:
            raise ValueError("user groups are required for user-aware evaluation")
        source_groups = np.asarray(groups)
        if source_groups.ndim != 1 or len(source_groups) != len(frame):
            raise ValueError("one group ID is required for each cached sample row")
        valid_groups = source_groups[features.index.to_numpy()]
        train_indexes, holdout_indexes = user_aware_split(
            features, labels, valid_groups
        )
        cutoff = None
        protocol = "user-disjoint local holdout from sampled training shards"
    else:
        raise ValueError("split_strategy must be 'chronological' or 'user'")
    train = features.iloc[train_indexes]
    holdout = features.iloc[holdout_indexes]
    train_labels = labels[train_indexes]
    holdout_labels = labels[holdout_indexes]

    model_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    predictions: dict[str, dict[str, np.ndarray]] = {}
    median_value = float(np.median(train_labels))
    median_train_prediction = np.full(len(train_indexes), median_value)
    median_holdout_prediction = np.full(len(holdout_indexes), median_value)
    results["median"] = _record_metrics(
        labels,
        train_indexes,
        holdout_indexes,
        median_train_prediction,
        median_holdout_prediction,
    )
    predictions["median"] = {
        "train": median_train_prediction,
        "holdout": median_holdout_prediction,
    }

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_features=150_000, sublinear_tf=True
    )
    text_model = Pipeline([("tfidf", vectorizer), ("regressor", Ridge(alpha=2.0))])
    text_model.fit(train["text"], train_labels)
    text_train_prediction = text_model.predict(train["text"])
    text_holdout_prediction = text_model.predict(holdout["text"])
    results["tfidf_ridge"] = _record_metrics(
        labels,
        train_indexes,
        holdout_indexes,
        text_train_prediction,
        text_holdout_prediction,
    )
    predictions["tfidf_ridge"] = {
        "train": text_train_prediction,
        "holdout": text_holdout_prediction,
    }
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
    context_train_prediction = context_model.predict(train[context_features])
    context_holdout_prediction = context_model.predict(holdout[context_features])
    results["context_catboost"] = _record_metrics(
        labels,
        train_indexes,
        holdout_indexes,
        context_train_prediction,
        context_holdout_prediction,
    )
    predictions["context_catboost"] = {
        "train": context_train_prediction,
        "holdout": context_holdout_prediction,
    }
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
    fused_train_prediction = fused_model.predict(train)
    fused_holdout_prediction = fused_model.predict(holdout)
    results["text_context_ridge"] = _record_metrics(
        labels,
        train_indexes,
        holdout_indexes,
        fused_train_prediction,
        fused_holdout_prediction,
    )
    predictions["text_context_ridge"] = {
        "train": fused_train_prediction,
        "holdout": fused_holdout_prediction,
    }
    joblib.dump(fused_model, model_dir / "text-context-ridge.joblib")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    holdout_predictions = {
        name: values["holdout"]
        for name, values in predictions.items()
        if name != "median"
    }
    holdout_categories = features.iloc[holdout_indexes]["Category"].to_numpy()
    category_metrics = _holdout_category_metrics(
        holdout_categories,
        holdout_labels,
        holdout_predictions,
    )
    diagnostic_plots = _save_diagnostic_plots(
        diagnostic_dir,
        holdout_labels,
        holdout_predictions,
        category_metrics,
    )
    evaluation: dict[str, Any] = {
        "protocol": protocol,
        "holdout_fraction_requested": 0.2,
        "holdout_fraction_unit": "post_rows",
        "split_unit": "post_row",
        "cutoff_utc": (
            pd.to_datetime(cutoff, unit="s", utc=True).isoformat()
            if cutoff is not None
            else None
        ),
        "valid_rows": len(labels),
        "train_rows": len(train_indexes),
        "holdout_rows": len(holdout_indexes),
        "official_test_scored": False,
    }
    if valid_groups is not None:
        train_users = set(valid_groups[train_indexes])
        holdout_users = set(valid_groups[holdout_indexes])
        evaluation.update(
            {
                "train_user_count": len(train_users),
                "holdout_user_count": len(holdout_users),
                "user_overlap_count": len(train_users & holdout_users),
                "holdout_fraction_unit": "users",
                "split_unit": "user_group",
                "random_seed": 42,
            }
        )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": manifest,
        "evaluation": evaluation,
        "metrics": results,
        "diagnostics": {
            "residuals": {
                name: summarize_residuals(holdout_labels, values)
                for name, values in holdout_predictions.items()
            },
            "holdout_by_category": category_metrics,
            "minimum_category_rows": 50,
            "plots": diagnostic_plots,
        },
        "models": {
            "tfidf_ridge": str(model_dir / "tfidf-ridge.joblib"),
            "context_catboost": str(model_dir / "context-catboost.cbm"),
            "text_context_ridge": str(model_dir / "text-context-ridge.joblib"),
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