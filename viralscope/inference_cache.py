"""Cache non-image inputs from every public test shard for inference only."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, get_token

from .dataset_audit import (
    DEFAULT_REPOSITORY,
    DEFAULT_REVISION,
    fetch_viewer_size,
    summarize_viewer_splits,
)
from .sample_cache import _read_shard

DEFAULT_OUTPUT = Path("data/test-inference-inputs.parquet")
DEFAULT_MANIFEST = Path("data/test-inference-manifest.json")
INFERENCE_COLUMNS = (
    "Postdate",
    "Category",
    "Subcategory",
    "Concept",
    "Title",
    "Alltags",
)


def cache_unlabeled_test_inputs(
    repository: str = DEFAULT_REPOSITORY,
    revision: str = DEFAULT_REVISION,
    workers: int = 4,
    output: Path = DEFAULT_OUTPUT,
    manifest_output: Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    if not get_token():
        raise RuntimeError("Hugging Face authentication is required for this dataset.")
    if workers < 1:
        raise ValueError("workers must be positive")

    api = HfApi()
    pinned_info = api.dataset_info(repository, revision=revision, files_metadata=True)
    main_info = api.dataset_info(repository)
    if pinned_info.sha != revision or main_info.sha != pinned_info.sha:
        raise RuntimeError(
            "The repository main revision differs from the requested pin; "
            "refusing to cache a moving test split."
        )
    test_shards = sorted(
        (
            item
            for item in pinned_info.siblings
            if item.rfilename.startswith("data/test-")
            and item.rfilename.endswith(".parquet")
        ),
        key=lambda item: item.rfilename,
    )
    if not test_shards:
        raise RuntimeError("No test Parquet shards were found at the pinned revision.")

    with ThreadPoolExecutor(max_workers=min(workers, len(test_shards))) as executor:
        futures = {
            executor.submit(
                _read_shard,
                repository,
                pinned_info.sha,
                shard,
                INFERENCE_COLUMNS,
            ): index
            for index, shard in enumerate(test_shards)
        }
        tables: list[pa.Table | None] = [None] * len(test_shards)
        rows_loaded = 0
        for completed, future in enumerate(as_completed(futures), start=1):
            table = future.result()
            tables[futures[future]] = table
            rows_loaded += table.num_rows
            print(
                f"Read {completed}/{len(test_shards)} test shards; "
                f"rows loaded: {rows_loaded}",
                flush=True,
            )
    table = pa.concat_tables([table for table in tables if table is not None])

    viewer_size = fetch_viewer_size(repository)
    test_size = summarize_viewer_splits(viewer_size).get("test")
    expected_rows = None if test_size is None else test_size["rows"]
    if expected_rows is None or table.num_rows != expected_rows:
        raise RuntimeError(
            "Cached test-input row count does not match the dataset split size; "
            "no inference cache was written."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_suffix(output.suffix + ".tmp")
    temporary_manifest = manifest_output.with_suffix(manifest_output.suffix + ".tmp")
    pq.write_table(table, temporary_output, compression="zstd")
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": repository,
        "revision": pinned_info.sha,
        "split": "test",
        "rows": table.num_rows,
        "expected_rows": expected_rows,
        "columns": list(INFERENCE_COLUMNS),
        "shards_available": len(test_shards),
        "shards_selected": [shard.rfilename for shard in test_shards],
        "purpose": "inference_only_unlabeled_split",
        "labels_cached": False,
        "ids_cached": False,
        "images_downloaded": False,
    }
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary_output.replace(output)
    temporary_manifest.replace(manifest_output)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Cache non-image inputs from all test shards for inference only."
    )
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args(argv)
    manifest = cache_unlabeled_test_inputs(
        repository=args.repository,
        revision=args.revision,
        workers=args.workers,
        output=args.output,
        manifest_output=args.manifest,
    )
    print(f"Cached test input rows: {manifest['rows']}")
    print(f"Test shards: {len(manifest['shards_selected'])}")
    print(f"Parquet: {args.output}")
    print("Labels, IDs, and image bytes cached: no")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())