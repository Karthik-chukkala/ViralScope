"""Fit final inference models on every validated labeled training row."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .train_baselines import (
    DEFAULT_DATA,
    DEFAULT_MANIFEST,
    MODEL_DISPLAY_NAMES,
    _fit_model_suite,
    build_features,
)
import pandas as pd

DEFAULT_MODEL_DIR = Path("artifacts/models/final")
DEFAULT_REPORT = Path("artifacts/final-models-report.json")


def fit_final_models(
    data_path: Path = DEFAULT_DATA,
    manifest_path: Path = DEFAULT_MANIFEST,
    model_dir: Path = DEFAULT_MODEL_DIR,
    report_path: Path = DEFAULT_REPORT,
    workers: int = 4,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be positive")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("sample_method") != "all_train_shards":
        raise ValueError("Final models require a cache built from all pinned train shards")

    frame = pd.read_parquet(data_path)
    features, labels, _ = build_features(frame)
    if len(labels) != manifest["rows"]:
        raise RuntimeError(
            "Validated training rows do not match the all-shard manifest; "
            "refusing to fit final models."
        )

    _, model_paths = _fit_model_suite(features, labels, model_dir, workers)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": manifest,
        "fit_rows": len(labels),
        "fit_scope": "all validated labeled train rows",
        "models": model_paths,
        "model_names": MODEL_DISPLAY_NAMES,
        "holdout_metrics_report": "artifacts/baseline-metrics.json",
        "evaluation_metrics_included": False,
        "official_test_scored": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fit all learned models on all validated labeled train rows."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    report = fit_final_models(
        data_path=args.data,
        manifest_path=args.manifest,
        model_dir=args.model_dir,
        report_path=args.report,
        workers=args.workers,
    )
    print(f"Rows used for final fit: {report['fit_rows']}")
    print(f"Learned models: {len(report['models'])}")
    print(f"Model directory: {args.model_dir}")
    print(f"Fit report: {args.report}")
    print("Final models have no holdout metrics; use the separate evaluation report.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())