"""Evaluate future-time posts from users excluded from model training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .train_baselines import train_baselines
from .user_aware_eval import fetch_manifest_user_ids

DEFAULT_DATA = Path("data/train-all-safe.parquet")
DEFAULT_MANIFEST = Path("data/train-all-safe-manifest.json")
DEFAULT_REPORT = Path("artifacts/combined-evaluation.json")
DEFAULT_MODEL_DIR = Path("artifacts/models/combined")
DEFAULT_DIAGNOSTIC_DIR = Path("artifacts/diagnostics/combined")


def run_combined_evaluation(
    data_path: Path = DEFAULT_DATA,
    manifest_path: Path = DEFAULT_MANIFEST,
    report_path: Path = DEFAULT_REPORT,
    model_dir: Path = DEFAULT_MODEL_DIR,
    diagnostic_dir: Path = DEFAULT_DIAGNOSTIC_DIR,
    workers: int = 4,
    holdout_fraction: float = 0.2,
) -> dict[str, Any]:
    manifest, user_ids = fetch_manifest_user_ids(manifest_path, workers)
    if manifest.get("sample_method") != "all_train_shards":
        raise ValueError("Combined evaluation requires all pinned train shards")

    return train_baselines(
        data_path=data_path,
        manifest_path=manifest_path,
        report_path=report_path,
        model_dir=model_dir,
        diagnostic_dir=diagnostic_dir,
        workers=workers,
        split_strategy="combined",
        groups=user_ids,
        holdout_fraction=holdout_fraction,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate future-time posts only from users excluded from training."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--diagnostic-dir", type=Path, default=DEFAULT_DIAGNOSTIC_DIR)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    args = parser.parse_args(argv)

    report = run_combined_evaluation(
        data_path=args.data,
        manifest_path=args.manifest,
        report_path=args.report,
        model_dir=args.model_dir,
        diagnostic_dir=args.diagnostic_dir,
        workers=args.workers,
        holdout_fraction=args.holdout_fraction,
    )
    evaluation = report["evaluation"]
    print(f"Train rows: {evaluation['train_rows']}")
    print(f"Future-time unseen-user rows: {evaluation['holdout_rows']}")
    print(f"Purged pre-cutoff rows: {evaluation['purged_rows']}")
    print(f"User overlap: {evaluation['user_overlap_count']}")
    print(json.dumps(report["metrics"], indent=2))
    print(f"Metrics: {args.report}")
    print("Raw user IDs were kept in memory only and were not written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
