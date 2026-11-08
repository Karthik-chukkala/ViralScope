"""Run trained models over the full unlabeled test split without scoring labels."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .demo_app import MODEL_CHOICES, load_models
from .train_baselines import CATEGORICAL_FEATURES, TIME_FEATURES, build_inference_features

DEFAULT_DATA = Path("data/test-inference-inputs.parquet")
DEFAULT_MANIFEST = Path("data/test-inference-manifest.json")
DEFAULT_REPORT = Path("artifacts/test-inference-report.json")
DEFAULT_PREDICTIONS = Path("artifacts/test-inference-predictions.parquet")
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


def run_test_inference(
    data_path: Path = DEFAULT_DATA,
    manifest_path: Path = DEFAULT_MANIFEST,
    report_path: Path = DEFAULT_REPORT,
    predictions_path: Path = DEFAULT_PREDICTIONS,
    model_dir: Path = DEFAULT_MODEL_DIR,
    target_audit_path: Path = DEFAULT_TARGET_AUDIT,
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
    args = parser.parse_args(argv)
    report = run_test_inference(
        data_path=args.data,
        manifest_path=args.manifest,
        report_path=args.report,
        predictions_path=args.predictions,
        model_dir=args.model_dir,
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