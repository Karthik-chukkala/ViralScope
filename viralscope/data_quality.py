"""Audit full-train cache schema, ranges, and duplicate patterns without storing rows."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .sample_cache import CACHE_COLUMNS

DEFAULT_DATA = Path("data/train-all-safe.parquet")
DEFAULT_MANIFEST = Path("data/train-all-safe-manifest.json")
DEFAULT_OUTPUT = Path("artifacts/full-train-quality.json")
INPUT_COLUMNS = tuple(column for column in CACHE_COLUMNS if column != "label")
TEXT_COLUMNS = ("Category", "Subcategory", "Concept", "Title", "Alltags")


def _numeric_array(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.to_numpy(dtype=np.float64, na_value=np.nan)


def audit_data_quality(frame: pd.DataFrame) -> dict[str, Any]:
    missing_columns = set(CACHE_COLUMNS) - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing required training columns: {sorted(missing_columns)}")
    if frame.empty:
        raise ValueError("Cannot audit an empty training frame")

    column_quality: dict[str, Any] = {}
    for column in CACHE_COLUMNS:
        values = frame[column]
        blank_count = (
            int(values.astype("string").str.strip().eq("").fillna(False).sum())
            if column in TEXT_COLUMNS or column == "Postdate"
            else 0
        )
        column_quality[column] = {
            "dtype": str(values.dtype),
            "null_count": int(values.isna().sum()),
            "blank_count": blank_count,
        }

    timestamp_values = _numeric_array(frame["Postdate"])
    finite_timestamps = np.isfinite(timestamp_values)
    nonnumeric_timestamps = frame["Postdate"].notna().to_numpy() & np.isnan(timestamp_values)
    dates = pd.to_datetime(
        pd.Series(timestamp_values).where(finite_timestamps),
        unit="s",
        utc=True,
        errors="coerce",
    )
    valid_dates = dates.dropna()
    timestamp_seconds = timestamp_values[finite_timestamps]

    label_values = _numeric_array(frame["label"])
    finite_labels = np.isfinite(label_values)
    nonnumeric_labels = frame["label"].notna().to_numpy() & np.isnan(label_values)
    duplicate_input_groups = frame.groupby(
        list(INPUT_COLUMNS), dropna=False, sort=False, observed=True
    )["label"]
    group_sizes = duplicate_input_groups.size()
    distinct_labels = duplicate_input_groups.nunique(dropna=False)
    categories = frame["Category"].astype("string").str.strip()
    category_counts = categories[categories.notna() & categories.ne("")].value_counts()

    return {
        "rows": int(len(frame)),
        "schema": column_quality,
        "timestamps": {
            "nonnumeric_count": int(nonnumeric_timestamps.sum()),
            "missing_count": int(frame["Postdate"].isna().sum()),
            "nonfinite_count": int(
                np.isinf(timestamp_values).sum()
            ),
            "invalid_datetime_count": int(dates.isna().sum()),
            "min_utc": valid_dates.min().isoformat() if not valid_dates.empty else None,
            "max_utc": valid_dates.max().isoformat() if not valid_dates.empty else None,
            "seconds_quantiles": {
                str(q): float(np.quantile(timestamp_seconds, q))
                for q in (0.0, 0.01, 0.5, 0.99, 1.0)
            }
            if timestamp_seconds.size
            else {},
        },
        "labels": {
            "nonnumeric_count": int(nonnumeric_labels.sum()),
            "missing_count": int(frame["label"].isna().sum()),
            "nonfinite_count": int((~finite_labels & ~np.isnan(label_values)).sum()),
            "negative_count": int((finite_labels & (label_values < 0)).sum()),
            "min": (
                round(float(label_values[finite_labels].min()), 6)
                if finite_labels.any()
                else None
            ),
            "max": (
                round(float(label_values[finite_labels].max()), 6)
                if finite_labels.any()
                else None
            ),
        },
        "categories": {
            "distinct_nonblank": int(category_counts.size),
            "top_15": [
                {"category": str(category), "count": int(count)}
                for category, count in category_counts.head(15).items()
            ],
        },
        "duplicates": {
            "exact_duplicate_rows": int(frame.duplicated(keep="first").sum()),
            "duplicate_input_rows": int(len(frame) - len(group_sizes)),
            "duplicate_input_groups": int((group_sizes > 1).sum()),
            "duplicate_input_groups_with_conflicting_labels": int(
                ((group_sizes > 1) & (distinct_labels > 1)).sum()
            ),
            "duplicate_identity_check": (
                "Post IDs are not in the safe local cache; this check compares "
                "exact cached inputs and full cached rows only."
            ),
        },
        "privacy": {
            "raw_text_written": False,
            "raw_ids_read": False,
            "raw_values_written": False,
            "image_bytes_read": False,
        },
    }


def build_quality_report(
    data_path: Path = DEFAULT_DATA,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("sample_method") != "all_train_shards":
        raise ValueError("Full-train quality audit requires all pinned train shards")

    frame = pd.read_parquet(data_path, columns=list(CACHE_COLUMNS))
    if len(frame) != manifest.get("rows"):
        raise RuntimeError(
            "Cached training row count does not match the all-shard manifest"
        )
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "repository": manifest["dataset"],
            "revision": manifest["revision"],
            "split": manifest["split"],
            "rows": len(frame),
            "shards": len(manifest["shards_selected"]),
            "sample_method": manifest["sample_method"],
        },
        "quality": audit_data_quality(frame),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit full-train data quality without writing row-level data."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    report = build_quality_report(args.data, args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    quality = report["quality"]
    print(f"Rows audited: {quality['rows']}")
    print(f"Invalid timestamps: {quality['timestamps']['invalid_datetime_count']}")
    print(f"Exact duplicate rows: {quality['duplicates']['exact_duplicate_rows']}")
    print(
        "Conflicting duplicate-input groups: "
        f"{quality['duplicates']['duplicate_input_groups_with_conflicting_labels']}"
    )
    print(f"Aggregate report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
