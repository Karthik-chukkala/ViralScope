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
from sklearn.linear_model import SGDRegressor

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
MODEL_DISPLAY_NAMES = {
    "tfidf_ridge": "TF-IDF + Ridge",
    "tfidf_sgd": "TF-IDF + SGD",
    "tfidf_sgd_huber": "TF-IDF + Huber SGD",
    "context_catboost": "Context CatBoost",
    "context_catboost_shallow": "Context CatBoost (shallow)",
    "text_context_ridge": "Text + context Ridge",
}
TEXT_MODEL_NAMES = {"tfidf_ridge", "tfidf_sgd", "tfidf_sgd_huber"}


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


def build_inference_features(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"Postdate", "Category", "Subcategory", "Concept", "Title", "Alltags"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required inference columns: {sorted(missing)}")
    timestamps = pd.to_numeric(frame["Postdate"], errors="coerce")
    dates = pd.to_datetime(timestamps, unit="s", utc=True, errors="coerce")
    if dates.isna().any():
        raise ValueError(f"Cannot infer rows with invalid timestamps: {int(dates.isna().sum())}")

    features = pd.DataFrame(index=frame.index)
    for column in CATEGORICAL_FEATURES:
        features[column] = frame[column].fillna("").astype(str)
    features["text"] = (
        frame["Title"].fillna("").astype(str)
        + " "
        + frame["Alltags"].fillna("").astype(str)
    )
    time_values = (
        ("hour", dates.dt.hour.to_numpy(), 24),
        ("weekday", dates.dt.day_of_week.to_numpy(), 7),
        ("month", dates.dt.month.to_numpy() - 1, 12),
    )
    for name, values, period in time_values:
        features[f"{name}_sin"] = np.sin(2 * np.pi * values / period)
        features[f"{name}_cos"] = np.cos(2 * np.pi * values / period)
    return features


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


def combined_chronological_user_split(
    features: pd.DataFrame,
    labels: np.ndarray,
    timestamps: np.ndarray,
    groups: np.ndarray,
    holdout_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    group_values = np.asarray(groups)
    if group_values.ndim != 1 or len(group_values) != len(labels):
        raise ValueError("one group ID is required for each valid row")
    if pd.isna(group_values).any():
        raise ValueError("group IDs must not be missing")

    chronological_train, holdout, cutoff = chronological_split(
        features, labels, timestamps, holdout_fraction
    )
    holdout_groups = np.unique(group_values[holdout])
    purge_mask = np.isin(group_values[chronological_train], holdout_groups)
    train = chronological_train[~purge_mask]
    purged = chronological_train[purge_mask]
    if not len(train) or not len(holdout):
        raise ValueError("combined split must retain non-empty train and holdout rows")
    if np.intersect1d(group_values[train], group_values[holdout]).size:
        raise RuntimeError("combined split failed to separate user groups")
    if (timestamps[train] >= cutoff).any() or (timestamps[holdout] < cutoff).any():
        raise RuntimeError("combined split failed to preserve chronological ordering")
    return train, holdout, cutoff, purged


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
    model_names = [name for name in MODEL_DISPLAY_NAMES if name in predictions]
    model_labels = [MODEL_DISPLAY_NAMES[name] for name in model_names]
    plot_paths: list[str] = []

    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axes = axes.ravel()
    lower = float(min(actual.min(), *(predictions[name].min() for name in model_names)))
    upper = float(max(actual.max(), *(predictions[name].max() for name in model_names)))
    for axis, name, label in zip(axes, model_names, model_labels):
        axis.scatter(actual, predictions[name], s=7, alpha=0.16, color="#17685c", rasterized=True)
        axis.plot([lower, upper], [lower, upper], color="#c34f3c", linewidth=1)
        axis.set_title(label)
        axis.set_xlabel("Actual label")
        axis.set_ylabel("Predicted label")
        axis.grid(alpha=0.2)
    for axis in axes[len(model_names):]:
        axis.set_visible(False)
    prediction_plot = output_dir / "holdout-actual-vs-predicted.png"
    figure.savefig(prediction_plot, dpi=150)
    plt.close(figure)
    plot_paths.append(str(prediction_plot))

    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axes = axes.ravel()
    for axis, name, label in zip(axes, model_names, model_labels):
        axis.hist(predictions[name] - actual, bins=35, color="#4b9b87", edgecolor="white")
        axis.axvline(0, color="#c34f3c", linewidth=1)
        axis.set_title(label)
        axis.set_xlabel("Prediction error")
        axis.set_ylabel("Posts")
        axis.grid(axis="y", alpha=0.2)
    for axis in axes[len(model_names):]:
        axis.set_visible(False)
    residual_plot = output_dir / "holdout-residuals.png"
    figure.savefig(residual_plot, dpi=150)
    plt.close(figure)
    plot_paths.append(str(residual_plot))

    categories_to_plot = list(category_metrics)
    if categories_to_plot:
        indexes = np.arange(len(categories_to_plot))
        width = 0.8 / len(model_names)
        figure, axis = plt.subplots(figsize=(max(10, len(categories_to_plot) * 0.8), 5))
        offsets = (np.arange(len(model_names)) - (len(model_names) - 1) / 2) * width
        for offset, name, label in zip(offsets, model_names, model_labels):
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


def _fit_model_suite(
    features: pd.DataFrame,
    labels: np.ndarray,
    model_dir: Path,
    workers: int,
) -> tuple[dict[str, Any], dict[str, str]]:
    model_dir.mkdir(parents=True, exist_ok=True)
    models: dict[str, Any] = {}
    filenames = {
        "tfidf_ridge": "tfidf-ridge.joblib",
        "tfidf_sgd": "tfidf-sgd.joblib",
        "tfidf_sgd_huber": "tfidf-sgd-huber.joblib",
        "context_catboost": "context-catboost.cbm",
        "context_catboost_shallow": "context-catboost-shallow.cbm",
        "text_context_ridge": "text-context-ridge.joblib",
    }
    text_estimators = {
        "tfidf_ridge": Ridge(alpha=2.0),
        "tfidf_sgd": SGDRegressor(
            loss="squared_error",
            penalty="l2",
            alpha=1e-5,
            max_iter=1000,
            tol=1e-3,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=5,
        ),
        "tfidf_sgd_huber": SGDRegressor(
            loss="huber",
            penalty="l2",
            alpha=1e-5,
            max_iter=1000,
            tol=1e-3,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=5,
        ),
    }
    for model_name, estimator in text_estimators.items():
        print(f"Training {MODEL_DISPLAY_NAMES[model_name]}...", flush=True)
        model = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        min_df=2,
                        max_features=150_000,
                        sublinear_tf=True,
                    ),
                ),
                ("regressor", estimator),
            ]
        )
        model.fit(features["text"], labels)
        joblib.dump(model, model_dir / filenames[model_name])
        models[model_name] = model

    context_features = CATEGORICAL_FEATURES + TIME_FEATURES
    context_configurations = {
        "context_catboost": {"iterations": 500, "depth": 8, "learning_rate": 0.05},
        "context_catboost_shallow": {"iterations": 700, "depth": 5, "learning_rate": 0.04},
    }
    for model_name, configuration in context_configurations.items():
        print(f"Training {MODEL_DISPLAY_NAMES[model_name]}...", flush=True)
        model = CatBoostRegressor(
            **configuration,
            loss_function="RMSE",
            random_seed=42,
            thread_count=workers,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(features[context_features], labels, cat_features=CATEGORICAL_FEATURES)
        model.save_model(str(model_dir / filenames[model_name]))
        models[model_name] = model

    print(f"Training {MODEL_DISPLAY_NAMES['text_context_ridge']}...", flush=True)
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
    fused_model.fit(features, labels)
    joblib.dump(fused_model, model_dir / filenames["text_context_ridge"])
    models["text_context_ridge"] = fused_model
    return models, {name: str(model_dir / filename) for name, filename in filenames.items()}


def _predict_model(model_name: str, model: Any, features: pd.DataFrame) -> np.ndarray:
    if model_name in TEXT_MODEL_NAMES:
        return np.asarray(model.predict(features["text"]), dtype=np.float64)
    if model_name.startswith("context_catboost"):
        return np.asarray(
            model.predict(features[CATEGORICAL_FEATURES + TIME_FEATURES]),
            dtype=np.float64,
        )
    return np.asarray(model.predict(features), dtype=np.float64)


def train_baselines(
    data_path: Path = DEFAULT_DATA,
    manifest_path: Path = DEFAULT_MANIFEST,
    report_path: Path = DEFAULT_REPORT,
    model_dir: Path = DEFAULT_MODEL_DIR,
    workers: int = 4,
    split_strategy: str = "chronological",
    groups: np.ndarray | None = None,
    diagnostic_dir: Path = DEFAULT_DIAGNOSTIC_DIR,
    holdout_fraction: float = 0.2,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be positive")
    frame = pd.read_parquet(data_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_scope = (
        "full pinned train split"
        if manifest.get("sample_method") == "all_train_shards"
        else "sampled training shards"
    )
    features, labels, timestamps = build_features(frame)
    valid_groups = None
    purged_indexes = np.array([], dtype=np.int64)
    cutoff: int | None
    if split_strategy == "chronological":
        train_indexes, holdout_indexes, cutoff = chronological_split(
            features, labels, timestamps, holdout_fraction=holdout_fraction
        )
        protocol = f"chronological local holdout from {dataset_scope}"
    elif split_strategy == "user":
        if groups is None:
            raise ValueError("user groups are required for user-aware evaluation")
        source_groups = np.asarray(groups)
        if source_groups.ndim != 1 or len(source_groups) != len(frame):
            raise ValueError("one group ID is required for each cached sample row")
        valid_groups = source_groups[features.index.to_numpy()]
        train_indexes, holdout_indexes = user_aware_split(
            features, labels, valid_groups, holdout_fraction=holdout_fraction
        )
        cutoff = None
        protocol = f"user-disjoint local holdout from {dataset_scope}"
    elif split_strategy == "combined":
        if groups is None:
            raise ValueError("user groups are required for combined evaluation")
        source_groups = np.asarray(groups)
        if source_groups.ndim != 1 or len(source_groups) != len(frame):
            raise ValueError("one group ID is required for each cached sample row")
        valid_groups = source_groups[features.index.to_numpy()]
        (
            train_indexes,
            holdout_indexes,
            cutoff,
            purged_indexes,
        ) = combined_chronological_user_split(
            features,
            labels,
            timestamps,
            valid_groups,
            holdout_fraction=holdout_fraction,
        )
        protocol = (
            f"future-time and unseen-user local holdout from {dataset_scope}; "
            "pre-cutoff rows from holdout users are purged from training"
        )
    else:
        raise ValueError("split_strategy must be 'chronological', 'user', or 'combined'")
    train = features.iloc[train_indexes]
    holdout = features.iloc[holdout_indexes]
    train_labels = labels[train_indexes]
    holdout_labels = labels[holdout_indexes]

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
    models, model_paths = _fit_model_suite(train, train_labels, model_dir, workers)
    for model_name, model in models.items():
        train_predictions = _predict_model(model_name, model, train)
        holdout_predictions = _predict_model(model_name, model, holdout)
        results[model_name] = _record_metrics(
            labels,
            train_indexes,
            holdout_indexes,
            train_predictions,
            holdout_predictions,
        )
        predictions[model_name] = {
            "train": train_predictions,
            "holdout": holdout_predictions,
        }

    holdout_predictions = {
        name: values["holdout"]
        for name, values in predictions.items()
        if name in MODEL_DISPLAY_NAMES
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
        "holdout_fraction_requested": holdout_fraction,
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
            }
        )
        if split_strategy == "user":
            evaluation.update(
                {
                    "holdout_fraction_unit": "users",
                    "split_unit": "user_group",
                    "random_seed": 42,
                }
            )
        if split_strategy == "combined":
            purged_users = set(valid_groups[purged_indexes])
            evaluation.update(
                {
                    "split_unit": "future_time_unseen_user",
                    "train_time_max_utc": pd.to_datetime(
                        timestamps[train_indexes].max(), unit="s", utc=True
                    ).isoformat(),
                    "purged_rows": len(purged_indexes),
                    "purged_user_count": len(purged_users),
                    "holdout_time_min_utc": pd.to_datetime(
                        timestamps[holdout_indexes].min(), unit="s", utc=True
                    ).isoformat(),
                    "holdout_time_max_utc": pd.to_datetime(
                        timestamps[holdout_indexes].max(), unit="s", utc=True
                    ).isoformat(),
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
            **model_paths,
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
    parser.add_argument("--diagnostic-dir", type=Path, default=DEFAULT_DIAGNOSTIC_DIR)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    args = parser.parse_args(argv)
    report = train_baselines(
        data_path=args.data,
        manifest_path=args.manifest,
        report_path=args.report,
        model_dir=args.model_dir,
        diagnostic_dir=args.diagnostic_dir,
        workers=args.workers,
        holdout_fraction=args.holdout_fraction,
    )
    print(f"Train rows: {report['evaluation']['train_rows']}")
    print(f"Holdout rows: {report['evaluation']['holdout_rows']}")
    print(json.dumps(report["metrics"], indent=2))
    print(f"Metrics: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())