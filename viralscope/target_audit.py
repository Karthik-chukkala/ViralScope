"""Audit the full labeled training target using column-selective Parquet reads."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem, get_token

from .dataset_audit import (
    DEFAULT_REPOSITORY,
    DEFAULT_REVISION,
    fetch_viewer_size,
    summarize_viewer_splits,
)

DEFAULT_OUTPUT = Path("artifacts/train-target-audit.json")
HISTOGRAM_BINS = 15


def _numeric_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std_population": None,
            "quantiles": {},
        }

    array = np.asarray(values, dtype=np.float64)
    quantile_levels = (0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
    quantile_values = np.quantile(array, quantile_levels)
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std_population": float(array.std()),
        "quantiles": {
            str(level): float(value)
            for level, value in zip(quantile_levels, quantile_values)
        },
    }


def summarize_labels(raw_values: Iterable[Any]) -> dict[str, Any]:
    finite_values: list[float] = []
    missing_count = 0
    non_numeric_count = 0
    non_finite_count = 0

    for raw_value in raw_values:
        if raw_value is None:
            missing_count += 1
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            non_numeric_count += 1
            continue
        if not math.isfinite(value):
            non_finite_count += 1
            continue
        finite_values.append(value)

    sentinel_count = sum(value == -1.0 for value in finite_values)
    non_sentinel_values = [value for value in finite_values if value != -1.0]
    negative_non_sentinel_count = sum(value < 0 for value in non_sentinel_values)

    histogram = {"counts": [], "bin_edges": []}
    if non_sentinel_values:
        counts, bin_edges = np.histogram(non_sentinel_values, bins=HISTOGRAM_BINS)
        histogram = {
            "counts": counts.astype(int).tolist(),
            "bin_edges": bin_edges.astype(float).tolist(),
        }

    return {
        "rows_scanned": missing_count + non_numeric_count + non_finite_count + len(finite_values),
        "missing_count": missing_count,
        "non_numeric_count": non_numeric_count,
        "non_finite_count": non_finite_count,
        "finite_summary_including_minus_one": _numeric_summary(finite_values),
        "sentinel_minus_one_count": sentinel_count,
        "negative_non_sentinel_count": negative_non_sentinel_count,
        "non_sentinel_summary": _numeric_summary(non_sentinel_values),
        "non_sentinel_histogram": histogram,
    }


def scan_training_labels(repository: str, revision: str) -> dict[str, Any]:
    token = get_token()
    if not token:
        raise RuntimeError("Hugging Face authentication is required for this dataset.")

    api = HfApi()
    pinned_info = api.dataset_info(
        repository,
        revision=revision,
        files_metadata=True,
    )
    main_info = api.dataset_info(repository)
    if main_info.sha != pinned_info.sha:
        raise RuntimeError(
            "The repository main revision differs from the requested pin; "
            "refusing to combine pinned Parquet data with current size metadata."
        )

    train_shards = sorted(
        (
            item
            for item in pinned_info.siblings
            if item.rfilename.startswith("data/train-")
            and item.rfilename.endswith(".parquet")
        ),
        key=lambda item: item.rfilename,
    )
    if not train_shards:
        raise RuntimeError("No train Parquet shards were found at the pinned revision.")

    filesystem = HfFileSystem(token=token)
    raw_labels: list[Any] = []
    row_groups_scanned = 0
    compressed_label_bytes = 0

    for shard_index, shard in enumerate(train_shards, start=1):
        print(
            f"Reading train shard {shard_index}/{len(train_shards)}: {shard.rfilename}",
            flush=True,
        )
        remote_path = f"datasets/{repository}@{revision}/{shard.rfilename}"
        with filesystem.open(remote_path, "rb", block_size=64 * 1024) as remote_file:
            parquet = pq.ParquetFile(remote_file)
            if "label" not in parquet.schema.names:
                raise RuntimeError(f"Missing label column in {shard.rfilename}.")
            label_index = parquet.schema.names.index("label")
            row_groups_scanned += parquet.metadata.num_row_groups
            for group_index in range(parquet.metadata.num_row_groups):
                group = parquet.metadata.row_group(group_index)
                compressed_label_bytes += group.column(label_index).total_compressed_size

            table = parquet.read(columns=["label"])
            raw_labels.extend(table.column("label").to_pylist())
        print(f"  Labels scanned so far: {len(raw_labels)}", flush=True)

    viewer_size = fetch_viewer_size(repository)
    train_size = summarize_viewer_splits(viewer_size).get("train")
    if train_size is None or len(raw_labels) != train_size["rows"]:
        raise RuntimeError(
            "Scanned label count does not match the viewer train split; "
            "no target report was written."
        )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "repository": repository,
            "revision": pinned_info.sha,
            "split": "train",
            "expected_rows": train_size["rows"],
        },
        "scan": {
            "rows_scanned": len(raw_labels),
            "shards_scanned": len(train_shards),
            "row_groups_scanned": row_groups_scanned,
            "compressed_label_column_bytes": compressed_label_bytes,
            "other_columns_read": False,
            "image_bytes_read": False,
        },
        "target": summarize_labels(raw_labels),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit train labels without loading image or other columns."
    )
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    report = scan_training_labels(args.repository, args.revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    target = report["target"]
    summary = target["non_sentinel_summary"]
    print(f"Scanned labels: {report['scan']['rows_scanned']}")
    print(f"Train label range: {summary['min']} .. {summary['max']}")
    print(f"Train label mean: {summary['mean']}")
    print(f"-1 values: {target['sentinel_minus_one_count']}")
    print(f"Report: {args.output}")
    print("Only the train label column was read; image bytes were not read.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())