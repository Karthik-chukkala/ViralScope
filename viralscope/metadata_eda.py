"""Summarize selected training metadata without retaining raw post content."""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
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

DEFAULT_OUTPUT = Path("artifacts/metadata-eda.json")
DEFAULT_SAMPLE_SHARDS = 7
DEFAULT_WORKERS = 4
SELECTED_COLUMNS = (
    "Uid",
    "Pid",
    "Postdate",
    "Category",
    "Subcategory",
    "Concept",
    "Title",
    "Alltags",
)


def _length_summary(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    array = np.asarray(values, dtype=np.int64)
    return {
        "count": int(array.size),
        "min": int(array.min()),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "max": int(array.max()),
    }


class MetadataAccumulator:
    def __init__(self) -> None:
        self.row_count = 0
        self.missing_counts = Counter()
        self.category_counts = Counter()
        self.subcategory_counts = Counter()
        self.concept_counts = Counter()
        self.month_counts = Counter()
        self.user_ids: set[str] = set()
        self.post_ids: set[str] = set()
        self.duplicate_post_id_count = 0
        self.invalid_timestamp_count = 0
        self.title_lengths: list[int] = []
        self.title_word_counts: list[int] = []
        self.tag_lengths: list[int] = []
        self.tag_word_counts: list[int] = []

    @staticmethod
    def _value(row: dict[str, Any], field: str) -> str | None:
        value = row.get(field)
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    def add_rows(self, rows: Iterable[dict[str, Any]]) -> None:
        for row in rows:
            self.row_count += 1
            values = {field: self._value(row, field) for field in SELECTED_COLUMNS}
            for field, value in values.items():
                if value is None:
                    self.missing_counts[field] += 1

            user_id = values["Uid"]
            if user_id is not None:
                self.user_ids.add(user_id)

            post_id = values["Pid"]
            if post_id is not None:
                if post_id in self.post_ids:
                    self.duplicate_post_id_count += 1
                else:
                    self.post_ids.add(post_id)

            for field, counter in (
                ("Category", self.category_counts),
                ("Subcategory", self.subcategory_counts),
                ("Concept", self.concept_counts),
            ):
                if values[field] is not None:
                    counter[values[field]] += 1

            for field, char_lengths, word_counts in (
                ("Title", self.title_lengths, self.title_word_counts),
                ("Alltags", self.tag_lengths, self.tag_word_counts),
            ):
                text = values[field]
                if text is not None:
                    char_lengths.append(len(text))
                    word_counts.append(len(re.findall(r"\S+", text)))

            post_date = values["Postdate"]
            if post_date is not None:
                try:
                    timestamp = int(post_date)
                    month = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                    self.month_counts[month.strftime("%Y-%m")] += 1
                except (ValueError, OverflowError, OSError):
                    self.invalid_timestamp_count += 1

    @staticmethod
    def _frequencies(counter: Counter[str]) -> list[dict[str, Any]]:
        return [
            {"value": value, "count": count}
            for value, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
        ]

    def build_report(self) -> dict[str, Any]:
        return {
            "rows_scanned": self.row_count,
            "missing_counts": {
                field: self.missing_counts.get(field, 0)
                for field in SELECTED_COLUMNS
            },
            "unique_users": len(self.user_ids),
            "unique_post_ids": len(self.post_ids),
            "duplicate_post_id_rows": self.duplicate_post_id_count,
            "invalid_timestamp_count": self.invalid_timestamp_count,
            "category_frequencies": self._frequencies(self.category_counts),
            "subcategory_frequencies": self._frequencies(self.subcategory_counts),
            "top_20_concepts": self._frequencies(self.concept_counts)[:20],
            "monthly_post_counts_utc": [
                {"month": month, "count": count}
                for month, count in sorted(self.month_counts.items())
            ],
            "title_length_characters": _length_summary(self.title_lengths),
            "title_word_counts": _length_summary(self.title_word_counts),
            "tag_field_length_characters": _length_summary(self.tag_lengths),
            "tag_field_token_counts": _length_summary(self.tag_word_counts),
            "privacy": {
                "raw_text_written": False,
                "raw_user_ids_written": False,
                "coordinates_read": False,
                "image_bytes_read": False,
            },
        }


    def merge(self, other: MetadataAccumulator) -> None:
        self.duplicate_post_id_count += other.duplicate_post_id_count
        self.duplicate_post_id_count += len(self.post_ids & other.post_ids)
        self.row_count += other.row_count
        self.missing_counts.update(other.missing_counts)
        self.category_counts.update(other.category_counts)
        self.subcategory_counts.update(other.subcategory_counts)
        self.concept_counts.update(other.concept_counts)
        self.month_counts.update(other.month_counts)
        self.user_ids.update(other.user_ids)
        self.post_ids.update(other.post_ids)
        self.invalid_timestamp_count += other.invalid_timestamp_count
        self.title_lengths.extend(other.title_lengths)
        self.title_word_counts.extend(other.title_word_counts)
        self.tag_lengths.extend(other.tag_lengths)
        self.tag_word_counts.extend(other.tag_word_counts)


def select_evenly_spaced_shards(shards: list[Any], count: int) -> list[Any]:
    if count <= 0:
        raise ValueError("sample shard count must be positive")
    if count >= len(shards):
        return shards
    indexes = [index * (len(shards) - 1) // (count - 1) for index in range(count)] if count > 1 else [len(shards) // 2]
    return [shards[index] for index in indexes]


def _read_metadata_shard(repository: str, revision: str, shard: Any) -> tuple[MetadataAccumulator, int]:
    token = get_token()
    filesystem = HfFileSystem(token=token)
    accumulator = MetadataAccumulator()
    remote_path = f"datasets/{repository}@{revision}/{shard.rfilename}"
    with filesystem.open(remote_path, "rb", block_size=64 * 1024) as remote_file:
        parquet = pq.ParquetFile(remote_file)
        missing_columns = set(SELECTED_COLUMNS) - set(parquet.schema.names)
        if missing_columns:
            raise RuntimeError(f"Missing required columns: {sorted(missing_columns)}")
        row_groups = parquet.metadata.num_row_groups
        table = parquet.read(columns=list(SELECTED_COLUMNS))
        accumulator.add_rows(table.to_pylist())
    return accumulator, row_groups


def scan_training_metadata(
    repository: str,
    revision: str,
    sample_shards: int | None = DEFAULT_SAMPLE_SHARDS,
    workers: int = DEFAULT_WORKERS,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be positive")
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

    selected_shards = (
        train_shards
        if sample_shards is None
        else select_evenly_spaced_shards(train_shards, sample_shards)
    )
    accumulator = MetadataAccumulator()
    row_groups_scanned = 0
    print(
        f"Reading {len(selected_shards)} of {len(train_shards)} train shards "
        f"with up to {min(workers, len(selected_shards))} workers.",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=min(workers, len(selected_shards))) as executor:
        futures = [
            executor.submit(_read_metadata_shard, repository, revision, shard)
            for shard in selected_shards
        ]
        for future in futures:
            shard_accumulator, row_groups = future.result()
            accumulator.merge(shard_accumulator)
            row_groups_scanned += row_groups
            print(f"  Rows scanned so far: {accumulator.row_count}", flush=True)

    expected_rows = None
    if sample_shards is None:
        viewer_size = fetch_viewer_size(repository)
        train_size = summarize_viewer_splits(viewer_size).get("train")
        if train_size is None or accumulator.row_count != train_size["rows"]:
            raise RuntimeError(
                "Scanned metadata row count does not match the viewer train split; "
                "no EDA report was written."
            )
        expected_rows = train_size["rows"]

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "repository": repository,
            "revision": pinned_info.sha,
            "split": "train",
            "expected_rows": expected_rows,
        },
        "scan": {
            "rows_scanned": accumulator.row_count,
            "shards_scanned": len(selected_shards),
            "shards_available": len(train_shards),
            "sampled": sample_shards is not None and len(selected_shards) < len(train_shards),
            "shard_selection": "evenly_spaced" if len(selected_shards) < len(train_shards) else "all",
            "shard_names": [shard.rfilename for shard in selected_shards],
            "row_groups_scanned": row_groups_scanned,
            "columns_read": list(SELECTED_COLUMNS),
            "image_bytes_read": False,
        },
        "eda": accumulator.build_report(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate train metadata without retaining raw post content."
    )
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-shards", type=int, default=DEFAULT_SAMPLE_SHARDS)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--full", action="store_true", help="scan every train shard")
    args = parser.parse_args(argv)

    report = scan_training_metadata(
        args.repository,
        args.revision,
        sample_shards=None if args.full else args.sample_shards,
        workers=args.workers,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Rows scanned: {report['scan']['rows_scanned']}")
    print(f"Unique users: {report['eda']['unique_users']}")
    print(f"Duplicate post IDs: {report['eda']['duplicate_post_id_rows']}")
    print(f"Report: {args.output}")
    print("No raw text, user IDs, coordinates, or images were written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())