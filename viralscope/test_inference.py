"""Run trained models over the full unlabeled test split without scoring labels."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .demo_app import MODEL_CHOICES, load_models
from .train_baselines import CATEGORICAL_FEATURES, TIME_FEATURES, build_inference_features

DEFAULT_DATA = Path("data/test-inference-inputs.parquet")
DEFAULT_MANIFEST = Path("data/test-inference-manifest.json")
DEFAULT_REPORT = Path("artifacts/test-inference-report.json")
DEFAULT_PREDICTIONS = Path("artifacts/test-inference-predictions.parquet")
DEFAULT_TEST_EDA_PLOT = Path("artifacts/diagnostics/test-input-coverage.png")
DEFAULT_MODEL_DIR = Path("artifacts/models/final")
DEFAULT_TARGET_AUDIT = Path("artifacts/train-target-audit.json")


def summarize_predictions(
    values: np.ndarray,
    training_label_range: tuple[float, float] | None = None,
) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("predictions must be a non-empty finite vector")
    summary: dict[str, float | int] = {
        "rows": int(values.size),
        "min": float(values.min()),
        "p10": float(np.quantile(values, 0.1)),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p90": float(np.quantile(values, 0.9)),
        "max": float(values.max()),
    }
    if training_label_range is not None:
        lower, upper = training_label_range
        summary["outside_train_label_range_count"] = int(
            np.count_nonzero((values < lower) | (values > upper))
        )
    return summary


def summarize_unlabeled_inputs(frame: pd.DataFrame) -> dict[str, Any]:
    required = ("Postdate", "Category", "Title", "Alltags")
    missing_columns = set(required) - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing required test input columns: {sorted(missing_columns)}")
    dates = pd.to_datetime(
        pd.to_numeric(frame["Postdate"], errors="coerce"),
        unit="s",
        utc=True,
        errors="coerce",
    )
    titles = frame["Title"].fillna("").astype(str).str.strip()
    tags = frame["Alltags"].fillna("").astype(str).str.strip()
    title_present = titles.ne("")
    tag_present = tags.ne("")
    title_lengths = titles[title_present].str.len().to_numpy(dtype=np.int64)
    title_words = titles[title_present].map(lambda value: len(re.findall(r"\S+", value))).to_numpy(dtype=np.int64)
    tag_tokens = tags[tag_present].map(lambda value: len(re.findall(r"\S+", value))).to_numpy(dtype=np.int64)
    monthly = dates.dropna().dt.strftime("%Y-%m").value_counts().sort_index()
    normalized_categories = frame["Category"].fillna("").astype(str).str.strip()
    category_counts = normalized_categories[normalized_categories.ne("")].value_counts()

    def length_summary(values: np.ndarray) -> dict[str, float | int | None]:
        if values.size == 0:
            return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
        return {
            "count": int(values.size),
            "min": int(values.min()),
            "median": float(np.median(values)),
            "mean": float(values.mean()),
            "max": int(values.max()),
        }

    return {
        "rows": int(len(frame)),
        "missing_or_blank": {
            "Postdate": int(dates.isna().sum()),
            "Category": int(normalized_categories.eq("").sum()),
            "Title": int((~title_present).sum()),
            "Alltags": int((~tag_present).sum()),
        },
        "invalid_timestamp_count": int(dates.isna().sum()),
        "postdate_range_utc": {
            "min": dates.min().isoformat() if dates.notna().any() else None,
            "max": dates.max().isoformat() if dates.notna().any() else None,
        },
        "monthly_posts_utc": [
            {"month": month, "count": int(count)} for month, count in monthly.items()
        ],
        "category_frequencies": [
            {"category": category, "count": int(count)}
            for category, count in category_counts.items()
        ],
        "title_character_lengths": length_summary(title_lengths),
        "title_word_counts": length_summary(title_words),
        "tag_token_counts": length_summary(tag_tokens),
        "labels_read": False,
        "ids_read": False,
        "images_read": False,
    }


def _save_test_input_plot(
    frame: pd.DataFrame,
    profile: dict[str, Any],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    axes = axes.ravel()
    monthly = profile["monthly_posts_utc"]
    axes[0].bar([item["month"] for item in monthly], [item["count"] for item in monthly], color="#17685c")
    axes[0].set_title("Test inputs by month (UTC)")
    axes[0].set_ylabel("Posts")
    axes[0].tick_params(axis="x", labelrotation=55)
    axes[0].grid(axis="y", alpha=0.2)

    categories = profile["category_frequencies"]
    axes[1].barh([item["category"] for item in reversed(categories)], [item["count"] for item in reversed(categories)], color="#c34f3c")
    axes[1].set_title("Test inputs by category")
    axes[1].set_xlabel("Posts")
    axes[1].grid(axis="x", alpha=0.2)

    titles = frame["Title"].fillna("").astype(str).str.strip()
    axes[2].hist(titles[titles.ne("")].str.len(), bins=30, color="#d7a338", edgecolor="white")
    axes[2].set_title("Test title length")
    axes[2].set_xlabel("Characters")
    axes[2].set_ylabel("Posts")
    axes[2].grid(axis="y", alpha=0.2)

    tags = frame["Alltags"].fillna("").astype(str).str.strip()
    tokens = tags[tags.ne("")].map(lambda value: len(re.findall(r"\S+", value)))
    axes[3].hist(tokens, bins=30, color="#5785a1", edgecolor="white")
    axes[3].set_title("Test tag field length")
    axes[3].set_xlabel("Whitespace-delimited tokens")
    axes[3].set_ylabel("Posts")
    axes[3].grid(axis="y", alpha=0.2)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def run_test_inference(
    data_path: Path = DEFAULT_DATA,
    manifest_path: Path = DEFAULT_MANIFEST,
    report_path: Path = DEFAULT_REPORT,
    predictions_path: Path = DEFAULT_PREDICTIONS,
    model_dir: Path = DEFAULT_MODEL_DIR,
    target_audit_path: Path = DEFAULT_TARGET_AUDIT,
    input_eda_plot_path: Path = DEFAULT_TEST_EDA_PLOT,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("split") != "test" or manifest.get("labels_cached"):
        raise ValueError("Inference input must be an unlabeled test-split cache")
    frame = pd.read_parquet(data_path, columns=list(manifest["columns"]))
    if len(frame) != manifest["rows"]:
        raise ValueError("Test-input cache row count does not match its manifest")
    if "label" in frame.columns or "Uid" in frame.columns or "Pid" in frame.columns:
        raise ValueError("Test inference inputs must not include labels or identifiers")

    features = build_inference_features(frame)
    input_profile = summarize_unlabeled_inputs(frame)
    _save_test_input_plot(frame, input_profile, input_eda_plot_path)
    models = load_models(model_dir)
    predictions: dict[str, np.ndarray] = {}
    for model_name in MODEL_CHOICES:
        model = models[model_name]
        if model_name in {"tfidf_ridge", "tfidf_sgd", "tfidf_sgd_huber"}:
            values = model.predict(features["text"])
        elif model_name.startswith("context_catboost"):
            values = model.predict(features[CATEGORICAL_FEATURES + TIME_FEATURES])
        else:
            values = model.predict(features)
        values = np.asarray(values, dtype=np.float64)
        if len(values) != len(frame) or not np.isfinite(values).all():
            raise RuntimeError(f"Model {model_name} returned invalid test predictions")
        predictions[model_name] = values

    target_audit = json.loads(target_audit_path.read_text(encoding="utf-8"))
    label_summary = target_audit["target"]["non_sentinel_summary"]
    training_label_range = (float(label_summary["min"]), float(label_summary["max"]))

    output = pd.DataFrame({"row_index": np.arange(len(frame), dtype=np.int64), **predictions})
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_predictions = predictions_path.with_suffix(predictions_path.suffix + ".tmp")
    output.to_parquet(temporary_predictions, index=False, compression="zstd")
    temporary_predictions.replace(predictions_path)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": manifest,
        "prediction_rows": len(frame),
        "input_eda": input_profile,
        "input_eda_plot": str(input_eda_plot_path),
        "models": {
            name: summarize_predictions(values, training_label_range)
            for name, values in predictions.items()
        },
        "training_label_range": {
            "min": training_label_range[0],
            "max": training_label_range[1],
        },
        "prediction_file": str(predictions_path),
        "test_metrics_computed": False,
        "test_metric_note": (
            "The public test labels are withheld (-1); inference predictions are not scored."
        ),
        "raw_text_written": False,
        "user_ids_written": False,
        "post_ids_written": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run all learned baselines over the unlabeled public test split."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--target-audit", type=Path, default=DEFAULT_TARGET_AUDIT)
    parser.add_argument("--input-eda-plot", type=Path, default=DEFAULT_TEST_EDA_PLOT)
    args = parser.parse_args(argv)
    report = run_test_inference(
        data_path=args.data,
        manifest_path=args.manifest,
        report_path=args.report,
        predictions_path=args.predictions,
        model_dir=args.model_dir,
        input_eda_plot_path=args.input_eda_plot,
        target_audit_path=args.target_audit,
    )
    print(f"Prediction rows: {report['prediction_rows']}")
    print(f"Models: {len(report['models'])}")
    print(f"Predictions: {args.predictions}")
    print(f"Summary: {args.report}")
    print("Test metrics computed: no; test labels are withheld.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())