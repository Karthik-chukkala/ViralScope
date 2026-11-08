"""Evaluate baselines on a user-disjoint holdout without persisting user IDs."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem, get_token

from .dataset_audit import DEFAULT_REPOSITORY, DEFAULT_REVISION
from .train_baselines import train_baselines

DEFAULT_MANIFEST = Path("data/train-sample-manifest.json")
DEFAULT_DATA = Path("data/train-sample.parquet")
DEFAULT_REPORT = Path("artifacts/user-aware-metrics.json")
DEFAULT_MODEL_DIR = Path("artifacts/models/user-aware")
DEFAULT_DIAGNOSTIC_DIR = Path("artifacts/diagnostics/user-aware")


def _read_user_ids(repository: str, revision: str, filename: str) -> list[str]:
    filesystem = HfFileSystem(token=get_token())
    remote_path = f"datasets/{repository}@{revision}/{filename}"
    with filesystem.open(remote_path, "rb", block_size=64 * 1024) as remote_file:
        parquet = pq.ParquetFile(remote_file)
        if "Uid" not in parquet.schema.names:
            raise RuntimeError(f"Missing required split-only column in {filename}: Uid")
        values = parquet.read(columns=["Uid"]).column("Uid").to_pylist()
    if any(value is None or not str(value).strip() for value in values):
        raise RuntimeError(f"Missing user IDs in selected shard: {filename}")
    return [str(value) for value in values]


def fetch_manifest_user_ids(
    manifest_path: Path = DEFAULT_MANIFEST,
    workers: int = 4,
) -> tuple[dict[str, Any], list[str]]:
    if workers < 1:
        raise ValueError("workers must be positive")
    token = get_token()
    if not token:
        raise RuntimeError("Hugging Face authentication is required for this dataset.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    repository = manifest["dataset"]
    revision = manifest["revision"]
    selected_names = manifest["shards_selected"]
    if not selected_names:
        raise RuntimeError("The sample manifest does not list selected train shards.")

    api = HfApi()
    pinned_info = api.dataset_info(repository, revision=revision, files_metadata=True)
    main_info = api.dataset_info(repository)
    if pinned_info.sha != revision or main_info.sha != pinned_info.sha:
        raise RuntimeError(
            "The repository revision no longer matches the sample manifest; "
            "refusing to align split-only IDs with a different sample."
        )
    available = {item.rfilename for item in pinned_info.siblings}
    missing_shards = set(selected_names) - available
    if missing_shards:
        raise RuntimeError(f"Manifest shards are missing at the pinned revision: {sorted(missing_shards)}")

    print(
        f"Reading split-only user IDs from {len(selected_names)} pinned train shards...",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=min(workers, len(selected_names))) as executor:
        futures = {
            executor.submit(_read_user_ids, repository, revision, filename): index
            for index, filename in enumerate(selected_names)
        }
        shard_user_ids: list[list[str] | None] = [None] * len(selected_names)
        for completed, future in enumerate(as_completed(futures), start=1):
            values = future.result()
            shard_user_ids[futures[future]] = values
            print(
                f"Read IDs from {completed}/{len(selected_names)} shards; "
                f"rows loaded: {sum(len(ids) for ids in shard_user_ids if ids is not None)}",
                flush=True,
            )
        user_ids = [user_id for ids in shard_user_ids if ids is not None for user_id in ids]

    if len(user_ids) != manifest["rows"]:
        raise RuntimeError(
            "The split-only ID row count does not match the sample cache; "
            "refusing to perform a misaligned group split."
        )
    return manifest, user_ids


def run_user_aware_evaluation(
    data_path: Path = DEFAULT_DATA,
    manifest_path: Path = DEFAULT_MANIFEST,
    report_path: Path = DEFAULT_REPORT,
    model_dir: Path = DEFAULT_MODEL_DIR,
    diagnostic_dir: Path = DEFAULT_DIAGNOSTIC_DIR,
    workers: int = 4,
) -> dict[str, Any]:
    manifest, user_ids = fetch_manifest_user_ids(manifest_path, workers)
    if data_path.stat().st_size == 0:
        raise RuntimeError("The local sample cache is empty.")

    report = train_baselines(
        data_path=data_path,
        manifest_path=manifest_path,
        report_path=report_path,
        model_dir=model_dir,
        workers=workers,
        split_strategy="user",
        groups=user_ids,
        diagnostic_dir=diagnostic_dir,
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline models on a user-disjoint holdout without saving user IDs."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--diagnostic-dir", type=Path, default=DEFAULT_DIAGNOSTIC_DIR)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)

    report = run_user_aware_evaluation(
        data_path=args.data,
        manifest_path=args.manifest,
        report_path=args.report,
        model_dir=args.model_dir,
        diagnostic_dir=args.diagnostic_dir,
        workers=args.workers,
    )
    evaluation = report["evaluation"]
    print(f"Train rows/users: {evaluation['train_rows']}/{evaluation['train_user_count']}")
    print(f"Holdout rows/users: {evaluation['holdout_rows']}/{evaluation['holdout_user_count']}")
    print(f"User overlap: {evaluation['user_overlap_count']}")
    print(json.dumps(report["metrics"], indent=2))
    print(f"Metrics: {args.report}")
    print("Raw user IDs were kept in memory only and were not written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())