"""Cache a reproducible, image-free training sample for local experiments."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem, get_token
import numpy as np

from .dataset_audit import DEFAULT_REPOSITORY, DEFAULT_REVISION
from .metadata_eda import (
    DEFAULT_SAMPLE_SHARDS,
    DEFAULT_WORKERS,
    select_evenly_spaced_shards,
)

DEFAULT_OUTPUT = Path("data/train-sample.parquet")
DEFAULT_MANIFEST = Path("data/train-sample-manifest.json")
CACHE_COLUMNS = (
    "Postdate",
    "Category",
    "Subcategory",
    "Concept",
    "Title",
    "Alltags",
    "label",
)


def _read_shard(repository: str, revision: str, shard: Any) -> pa.Table:
    filesystem = HfFileSystem(token=get_token())
    remote_path = f"datasets/{repository}@{revision}/{shard.rfilename}"
    with filesystem.open(remote_path, "rb", block_size=64 * 1024) as remote_file:
        parquet = pq.ParquetFile(remote_file)
        missing_columns = set(CACHE_COLUMNS) - set(parquet.schema.names)
        if missing_columns:
            raise RuntimeError(f"Missing required columns: {sorted(missing_columns)}")
        return parquet.read(columns=list(CACHE_COLUMNS))


def cache_training_sample(
    repository: str = DEFAULT_REPOSITORY,
    revision: str = DEFAULT_REVISION,
    sample_shards: int = DEFAULT_SAMPLE_SHARDS,
    workers: int = DEFAULT_WORKERS,
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
    if main_info.sha != pinned_info.sha:
        raise RuntimeError(
            "The repository main revision differs from the requested pin; "
            "refusing to cache data from a moving revision."
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
    selected_shards = select_evenly_spaced_shards(train_shards, sample_shards)

    with ThreadPoolExecutor(max_workers=min(workers, len(selected_shards))) as executor:
        futures = [
            executor.submit(_read_shard, repository, pinned_info.sha, shard)
            for shard in selected_shards
        ]
        tables = [future.result() for future in futures]

    table = pa.concat_tables(tables)
    if table.num_rows == 0:
        raise RuntimeError("Selected train shards contained no rows.")
    try:
        labels = np.asarray(
            table.column("label").to_numpy(zero_copy_only=False), dtype=np.float64
        )
    except (TypeError, ValueError):
        raise RuntimeError("Training sample contains a nonnumeric target label.") from None
    if not np.isfinite(labels).all() or (labels < 0).any():
        raise RuntimeError("Training sample contains a missing or negative target label.")

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_suffix(output.suffix + ".tmp")
    temporary_manifest = manifest_output.with_suffix(manifest_output.suffix + ".tmp")
    pq.write_table(table, temporary_output, compression="zstd")
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": repository,
        "revision": pinned_info.sha,
        "split": "train",
        "rows": table.num_rows,
        "columns": list(CACHE_COLUMNS),
        "sample_method": "evenly_spaced_shards",
        "shards_available": len(train_shards),
        "shards_selected": [shard.rfilename for shard in selected_shards],
        "images_downloaded": False,
        "profile_and_location_fields_cached": False,
        "local_cache_contains_post_text": True,
    }
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary_output.replace(output)
    temporary_manifest.replace(manifest_output)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Cache an image-free, reproducible train sample for local modeling."
    )
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--sample-shards", type=int, default=DEFAULT_SAMPLE_SHARDS)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args(argv)
    manifest = cache_training_sample(
        repository=args.repository,
        revision=args.revision,
        sample_shards=args.sample_shards,
        workers=args.workers,
        output=args.output,
        manifest_output=args.manifest,
    )
    print(f"Cached rows: {manifest['rows']}")
    print(f"Selected shards: {len(manifest['shards_selected'])}")
    print(f"Parquet: {args.output}")
    print(f"Manifest: {args.manifest}")
    print("Image bytes downloaded: no")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())